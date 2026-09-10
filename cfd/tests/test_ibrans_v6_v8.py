#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/test_ibrans_v6_v8.py
==============================================
Escalera de **verificación** del hito M3 de WAVE6 (especies + EDM +
encendido): tests unitarios de la cinética (M3a–M3d) y los casos de llama
V6 de Ciencia §13.1.

    M3a  cinética molar SI vs Cantera (mismo estado) ..... <0.1 %
    M3b  maquinaria de encendido (rampa, parche, anclaje)  exacta
    M3c  Pyrometheus == tabla == Cantera ................. <0.1 %
    M3d  reactor 0D: estequiometría y energía ............ conservación
    V6   llama premezclada 1D, S_L vs Cantera FreeFlame .. ±10 %
    V6c  ENCENDIDO desde frío (riesgo #1) ................ ancla y da la
                                                           misma S_L ±5 %
    V6d  llama CH4 (2S_CH4_BFER, el mecanismo de V8) ..... ±10 % [lento]
    V7   llama de contradifusión vs Cantera .............. T_max ±10 %
                                                           [lento]
    V8   Sandia Flame D .................................. PENDIENTE
                                                           (declarado)

La auto-validación V6 (la clave de bóveda de M3)
------------------------------------------------
Nuestro solver y Cantera reciben **el mismo mecanismo** (mismos A, Ea,
órdenes — `chem.cantera_yaml` declara las unidades cgs-mol/J-kmol de la
tabla y deja que Cantera convierta), la misma hipótesis Le = 1 (Cantera
con `transport_model="unity-Lewis-number"`, nosotros con Sc = Pr), el
mismo T_ad (c_p efectivo calibrado al equilibrio HP del sistema reducido
de 5 especies) y un α(T) potencial ajustado al α de Cantera en frío y en
quemado. Así **S_L compara el SOLVER (discretización + acoplamiento), no
el mecanismo** — que es exactamente lo que Ciencia §13.1 pide («el
mecanismo global no clava S_L; lo que importa es la consistencia
solver↔Cantera»). Este montaje cazó un bug real: la tabla llevaba Ea en
J/kmol y el esqueleto dividía entre R en J/(mol·K) — tasa cero absoluto a
cualquier T (corregido 2026-08-02, `chem.py` docstring).

V8 (Sandia Flame D) — PENDIENTE Y DECLARADO
-------------------------------------------
Necesita: mecanismo de metano en la tabla (2S_CH4_BFER, con fuente),
entrada anular de piloto quemado y un caso 3D de ~0.5–1 M celdas en GPU
(~1–2 h). No cabe en esta campaña y NO se convierte en un check trivial:
se salta con motivo visible (regla de la casa). El plan está en
`ESTADO.md` §11.

Qué corre en CI y qué NO
------------------------
* **Siempre**: M3a–M3d (segundos), V6a mini (~40 s), V6c encendido (~60 s).
* **Sólo con `PHYCB_SLOW=1`**: V6b (dx = 100 µm, la medida oficial), V6d
  (metano a 60 µm — su llama es más fina y no cabe en CI) y V7.
* Cantera ausente ⇒ V6/V7/M3a/M3c se saltan con motivo (nunca trivializan).

Doble modo::

    python cfd/tests/test_ibrans_v6_v8.py           # tabla ✓/✗/–
    python cfd/tests/test_ibrans_v6_v8.py --slow    # + V6b/V7
    python -m pytest cfd/tests/test_ibrans_v6_v8.py -q
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np                                            # noqa: E402

import physics_core as pc                                     # noqa: E402

pc.assert_provenance(pc)

from cfd.ibrans import chem                                   # noqa: E402
from cfd.ibrans.backend import to_numpy                       # noqa: E402
from cfd.ibrans.energy import EnergyConfig, EnergyLowMach     # noqa: E402
from cfd.ibrans.mesh import CartesianMesh                     # noqa: E402
from cfd.ibrans.solver import (IBRANSSolver, SolverConfig,    # noqa: E402
                               make_bcs)

SLOW = bool(os.environ.get("PHYCB_SLOW")) or "--slow" in sys.argv

TOL_V6 = 0.10          # S_L vs Cantera, mismo mecanismo (Ciencia §13.1)
TOL_V6C = 0.05         # encendido vs inicializada quemada (mismo solver)
TOL_V7 = 0.10          # T_max vs Cantera counterflow (baja tension)


def _cantera():
    try:
        import cantera as ct
        return ct
    except Exception:
        return None


# ==========================================================================
# utilidades compartidas del caso de llama 1D
# ==========================================================================
def _premixed_state(phi: float = 1.0, fuel: str = "propane"):
    """Estado sin quemar + T_ad + α(T) para el paso 1 efectivo del
    combustible `fuel` de la tabla (el MISMO mecanismo que verá nuestro
    solver)."""
    ct = _cantera()
    mech = chem.global_mechanism(fuel)
    formula = mech["formula"]
    gas = ct.Solution(yaml=chem.cantera_yaml(mech))
    gas.TP = 300.0, ct.one_atm
    gas.set_equivalence_ratio(phi, formula, "O2:1.0, N2:3.76")
    y_u = {sp: float(gas.Y[gas.species_index(n)])
           for sp, n in (("FUEL", formula), ("O2", "O2"), ("N2", "N2"))}
    y_u["CO2"] = 0.0
    y_u["H2O"] = 0.0
    st = dict(y_u=y_u, mw=float(gas.mean_molecular_weight),
              rho_u=float(gas.density),
              alpha_u=float(gas.thermal_conductivity
                            / (gas.density * gas.cp_mass)))
    gas.equilibrate("HP")
    st["t_ad"] = float(gas.T)
    alpha_b = float(gas.thermal_conductivity / (gas.density * gas.cp_mass))
    st["n_pow"] = float(np.log(alpha_b / st["alpha_u"])
                        / np.log(st["t_ad"] / 300.0))
    st["mech"] = mech
    st["fuel"] = fuel
    return st


_SL_CACHE: dict = {}


def _cantera_sl(phi: float = 1.0, fuel: str = "propane") -> float:
    """S_L de Cantera FreeFlame (Le = 1) con el mecanismo de la tabla.
    Cacheada: la llama de Cantera no cambia entre tests."""
    key = (phi, fuel)
    if key in _SL_CACHE:
        return _SL_CACHE[key]
    ct = _cantera()
    mech = chem.global_mechanism(fuel)
    gas = ct.Solution(yaml=chem.cantera_yaml(mech))
    gas.TP = 300.0, ct.one_atm
    gas.set_equivalence_ratio(phi, mech["formula"], "O2:1.0, N2:3.76")
    f = ct.FreeFlame(gas, width=0.03)
    f.transport_model = "unity-Lewis-number"
    f.set_refine_criteria(ratio=3, slope=0.07, curve=0.14)
    f.solve(loglevel=0, auto=True)
    _SL_CACHE[key] = float(f.velocity[0])
    return _SL_CACHE[key]


def case_flame_1d(dx_um: float = 150.0, max_iter: int = 12_000,
                  phi: float = 1.0, ignition: bool = False,
                  u_in: float | None = None, fuel: str = "propane"):
    """Llama premezclada 1D (canal x, transversal periódica). Con
    `ignition=False` se inicializa QUEMADA (frente tanh) y se relaja; con
    `ignition=True` se inicializa FRÍA y la enciende el plan del riesgo #1
    (parche + rampa + sub-relajación)."""
    st = _premixed_state(phi, fuel)
    s_l_ct = _cantera_sl(phi, fuel)
    dx = dx_um * 1e-6
    length = 0.03
    m = CartesianMesh([0.0, 0.0], [length, 4 * dx], dx)
    m.set_sdf(None)
    pr = 0.7
    u0 = float(u_in if u_in is not None else s_l_ct)
    cfg = SolverConfig(rho=st["rho_u"], nu=st["alpha_u"] * pr,
                       u_ref=s_l_ct * st["t_ad"] / 300.0, l_ref=length,
                       mach_art=0.2, cfl=0.4, rk="rk2",
                       max_iter=max_iter, tol=0.0, residual_every=500)
    bc_in = dict(kind="inlet", u=(u0, 0.0), T=300.0)
    for sp, v in st["y_u"].items():
        bc_in[f"Y_{sp}"] = v
    bcs = make_bcs(xlo=bc_in, xhi=dict(kind="outlet", p=0.0),
                   ylo="periodic", yhi="periodic")
    s = IBRANSSolver(m, cfg, bcs)
    plan = chem.IgnitionPlan(
        patch_center_m=(0.012, 2 * dx, 0.0), patch_radius_m=2.0e-3,
        patch_T_K=2000.0, n_hold=800, ramp_from=0.1, n_ramp=1500,
        relax=0.3, n_stable=400, stable_tol=0.02)
    en = EnergyLowMach(m, EnergyConfig(
        p_thermo_Pa=101325.0, r_gas=8314.46 / st["mw"], T_inlet_K=300.0,
        prandtl=pr, nu_power_T=st["n_pow"], T_ref_transport_K=300.0))
    sp_t = chem.SpeciesTransport(m, chem.ChemConfig(
        fuel=fuel, schmidt=pr, prandtl=pr, ignition=plan))
    sp_t.calibrate_cp(st["t_ad"], st["y_u"]["FUEL"], 300.0)
    sp_t.attach(s, turbulence=None, energy=en, y_init=st["y_u"],
                ignition=ignition)
    xg = np.asarray(m.X[0])
    if not ignition:
        prog = 0.5 * (1.0 + np.tanh((xg - 0.012) / 8e-4))
        split = sp_t.split
        y_u = st["y_u"]
        yb = dict(FUEL=0.0,
                  O2=max(y_u["O2"] - split["s"] * y_u["FUEL"], 0.0),
                  CO2=split["co2"] * y_u["FUEL"],
                  H2O=split["h2o"] * y_u["FUEL"], N2=y_u["N2"])
        for spn in sp_t.cfg.species:
            fld = (1 - prog) * y_u.get(spn, 0.0) + prog * yb[spn]
            sp_t.Y[spn][...] = s.xp.asarray(
                np.broadcast_to(fld, m.shape_tot), dtype=s.dtype)
        t_init = 300.0 + prog * (st["t_ad"] - 300.0)
        en.T[...] = s.xp.asarray(np.broadcast_to(t_init, m.shape_tot),
                                 dtype=s.dtype)
        en.rho[...] = s.dtype(en.cfg.p_thermo_Pa) \
            / (s.dtype(en.cfg.r_gas) * en.T)
        s.u[0][...] = s.xp.asarray(
            np.broadcast_to(u0 * t_init / 300.0, m.shape_tot),
            dtype=s.dtype)
    else:
        s.u[0][...] = u0
    s.apply_bc()
    return s, sp_t, en, dict(st=st, s_l_ct=s_l_ct, dx=dx)


def measure_sl(s, sp_t, ex, n_avg: int = 200) -> float:
    """S_L por velocidad de consumo, promediada `n_avg` pasos."""
    vals = []
    for _ in range(int(n_avg)):
        s.step()
        vals.append(sp_t.consumption_speed(
            axis=0, y_fuel_unburned=ex["st"]["y_u"]["FUEL"],
            rho_unburned=ex["st"]["rho_u"]))
    return float(np.mean(vals))


# ==========================================================================
# M3a · cinética molar SI vs Cantera (mismo estado)
# ==========================================================================
def test_m3a_kinetics_vs_cantera():
    """La conversión de unidades de la tabla (A cgs-mol, Ea J/kmol) contra
    la tasa de Cantera con el mismo mecanismo y el mismo estado. Este es el
    test que habría cazado el bug de Ea (tasa cero absoluto)."""
    ct = _cantera()
    if ct is None:
        _skip("cantera no disponible: la cinetica no se puede verificar "
              "contra referencia")
        return
    mech = chem.global_mechanism("propane")
    gas = ct.Solution(yaml=chem.cantera_yaml(mech))
    for T in (1200.0, 1600.0, 2100.0):
        gas.TP = T, ct.one_atm
        gas.set_equivalence_ratio(1.0, "C3H8", "O2:1.0, N2:3.76")
        i_f = gas.species_index("C3H8")
        w_ct = -float(gas.net_production_rates[i_f]
                      * gas.molecular_weights[i_f])
        yf = float(gas.Y[i_f])
        yo = float(gas.Y[gas.species_index("O2")])
        w_us = float(chem.arrhenius_rate(T, gas.density, yf, yo, mech, 0))
        assert abs(w_us / w_ct - 1.0) < 1e-3, (T, w_us, w_ct)
    # la tasa a 300 K debe ser despreciable (sin encendido no hay llama)
    w_cold = float(chem.arrhenius_rate(300.0, 1.2, 0.06, 0.22, mech, 0))
    assert w_cold < 1e-12, w_cold
    # split de productos coherente con la tabla estequiométrica (<1 %)
    for fuel in ("propane", "jet_a"):
        sp = chem.product_split(chem.global_mechanism(fuel))
        assert abs(sp["s"] / chem.stoichiometric_ratio(fuel) - 1.0) < 0.01
        # el paso completo conserva la masa: 1 + s = co2 + h2o (el redondeo
        # de los MW tabulados deja ~7e-5 de residuo — se admite 1e-3)
        assert abs((1.0 + sp["s"]) - (sp["co2"] + sp["h2o"])) < 1e-3, sp


# ==========================================================================
# M3b · maquinaria de encendido
# ==========================================================================
def test_m3b_ignition_machinery():
    plan = chem.IgnitionPlan(n_hold=100, ramp_from=0.1, n_ramp=200,
                             n_stable=50, stable_tol=0.02)
    # rampa: ramp_from durante el hold, lineal hasta 1, saturada
    assert plan.a_factor(0) == 0.1 and plan.a_factor(99) == 0.1
    assert abs(plan.a_factor(200) - 0.55) < 1e-12
    assert plan.a_factor(300) == 1.0 and plan.a_factor(10_000) == 1.0
    # parche: geometría exacta
    m = CartesianMesh([0.0, 0.0], [1.0, 1.0], 0.05)
    m.set_sdf(None)
    pl = chem.IgnitionPlan(patch_center_m=(0.5, 0.5, 0.0),
                           patch_radius_m=0.2)
    mask = chem.ignition_patch(m, pl)
    xg, yg = np.meshgrid(m.xc[0], m.xc[1], indexing="ij")
    exact = np.sqrt((xg - 0.5) ** 2 + (yg - 0.5) ** 2) <= 0.2
    assert bool(np.all(mask == exact))
    # criterio de anclaje sobre series sintéticas
    sp = chem.SpeciesTransport(m, chem.ChemConfig(ignition=plan))
    sp.hrr_history = [100.0] * 60
    assert sp.anchored() is True
    sp.hrr_history = [100.0] * 40 + [100.0 * (1 + 0.05 * (-1) ** i)
                                     for i in range(20)]
    assert sp.anchored() is False
    sp.hrr_history = [0.0] * 60                    # sin llama no hay ancla
    assert sp.anchored() is False


# ==========================================================================
# M3c · Pyrometheus == tabla == Cantera
# ==========================================================================
def test_m3c_pyrometheus_equivalence():
    """La vía Pyrometheus (WAVE6 §0, instalada con autorización el
    2026-08-02) genera tasas IDÉNTICAS a la tabla y a Cantera. Si no está,
    degrada con motivo y este test se salta — nunca trivializa."""
    ct = _cantera()
    if ct is None:
        _skip("cantera no disponible")
        return
    mech = chem.global_mechanism("propane")
    tc, why = chem.pyrometheus_rates(mech)
    if tc is None:
        _skip(f"pyrometheus degradado: {why}")
        return
    gas = ct.Solution(yaml=chem.cantera_yaml(mech))
    gas.TP = 1600.0, ct.one_atm
    gas.set_equivalence_ratio(1.0, "C3H8", "O2:1.0, N2:3.76")
    i_f = gas.species_index("C3H8")
    w = tc.get_net_production_rates(gas.density, 1600.0, gas.Y)
    w_pyro = -float(w[i_f]) * float(gas.molecular_weights[i_f])
    w_ct = -float(gas.net_production_rates[i_f]
                  * gas.molecular_weights[i_f])
    w_tab = float(chem.arrhenius_rate(
        1600.0, gas.density, float(gas.Y[i_f]),
        float(gas.Y[gas.species_index("O2")]), mech, 0))
    assert abs(w_pyro / w_ct - 1.0) < 1e-3, (w_pyro, w_ct)
    assert abs(w_tab / w_ct - 1.0) < 1e-3, (w_tab, w_ct)


# ==========================================================================
# M3d · reactor 0D: estequiometría y energía acopladas
# ==========================================================================
def test_m3d_batch_stoichiometry():
    """Caja periódica sin flujo, mezcla caliente: al avanzar, los saltos de
    especies deben seguir EXACTAMENTE la estequiometría del paso efectivo
    y la T debe subir ΔY_F·LHV/c_p (conservación, no modelo)."""
    m = CartesianMesh([0.0, 0.0], [0.8, 0.8], 0.1)
    m.set_sdf(None)
    cfg = SolverConfig(rho=1.0, nu=1e-6, u_ref=1.0, l_ref=1.0,
                       max_iter=10, tol=0.0)
    bcs = make_bcs(xlo="periodic", xhi="periodic",
                   ylo="periodic", yhi="periodic")
    s = IBRANSSolver(m, cfg, bcs)
    en = EnergyLowMach(m, EnergyConfig(T_inlet_K=1500.0))
    y0 = dict(FUEL=0.05, O2=0.22, CO2=0.0, H2O=0.0, N2=0.73)
    sp = chem.SpeciesTransport(m, chem.ChemConfig(fuel="propane"))
    sp.attach(s, turbulence=None, energy=en, y_init=y0, ignition=False)
    en.cfg.T_ceiling_K = 4000.0            # el juez es la conservación
    dt = 2.0e-5
    for _ in range(400):
        sp.advance(s, dt)
        en.advance(s, dt)
    I = m.interior
    yf = float(np.mean(np.asarray(to_numpy(sp.Y["FUEL"][I]))))
    yo = float(np.mean(np.asarray(to_numpy(sp.Y["O2"][I]))))
    yc = float(np.mean(np.asarray(to_numpy(sp.Y["CO2"][I]))))
    yh = float(np.mean(np.asarray(to_numpy(sp.Y["H2O"][I]))))
    t = float(np.mean(np.asarray(to_numpy(en.T[I]))))
    d_f = y0["FUEL"] - yf
    assert d_f > 0.005, ("el reactor no ha reaccionado", d_f)
    assert abs((y0["O2"] - yo) / d_f - sp.split["s"]) < 0.02
    assert abs(yc / d_f - sp.split["co2"]) < 0.02
    assert abs(yh / d_f - sp.split["h2o"]) < 0.02
    dt_teo = d_f * float(sp.mech["lhv_J_kg"]) / sp.cfg.cp_J_kgK
    assert abs((t - 1500.0) / dt_teo - 1.0) < 0.05, (t, dt_teo)
    # la suma de especies se mantiene en 1
    tot = yf + yo + yc + yh + float(np.mean(np.asarray(
        to_numpy(sp.Y["N2"][I]))))
    assert abs(tot - 1.0) < 5e-3, tot


# ==========================================================================
# V6 · llama premezclada 1D — S_L vs Cantera (mismo mecanismo)
# ==========================================================================
def run_v6(dx_um: float = 150.0, max_iter: int = 10_000,
           ignition: bool = False, fuel: str = "propane") -> dict:
    s, sp_t, en, ex = case_flame_1d(dx_um=dx_um, max_iter=max_iter,
                                    ignition=ignition, fuel=fuel)
    info = s.run()
    s_l = measure_sl(s, sp_t, ex)
    T = np.asarray(to_numpy(en.T[s.mesh.interior]))[:, 2]
    return dict(case=f"V6 dx={dx_um:.0f}um ign={ignition} {fuel}",
                status=info["status"], iterations=info["iterations"],
                wall_time_s=info["wall_time_s"],
                s_per_cell_step=info["s_per_cell_step"],
                S_L=s_l, S_L_cantera=ex["s_l_ct"],
                err=abs(s_l / ex["s_l_ct"] - 1.0),
                T_max=float(T.max()), T_ad=ex["st"]["t_ad"],
                anchored=sp_t.anchored(),
                front_mm=float(np.argmax(np.gradient(T)) * ex["dx"] * 1e3))


def test_v6a_flame_mini():
    """V6 (dx = 150 µm) · S_L vs Cantera FreeFlame, MISMO mecanismo,
    Le = 1, mismo T_ad. ±10 % (Ciencia §13.1)."""
    if _cantera() is None:
        _skip("cantera no disponible: V6 es una comparacion contra "
              "Cantera y sin el no se puede correr")
        return
    r = run_v6(dx_um=150.0, max_iter=10_000)
    assert r["anchored"], r
    assert r["err"] < TOL_V6, r
    assert abs(r["T_max"] / r["T_ad"] - 1.0) < 0.02, r
    assert 2.0 < r["front_mm"] < 28.0, ("el frente se salio", r)


def test_v6b_flame_full():
    """V6 (dx = 100 µm, la medida oficial) · **LENTO** (~2 min)."""
    if not SLOW:
        _skip("V6 completo (dx=100um, 20k it, ~2 min): PHYCB_SLOW=1")
        return
    if _cantera() is None:
        _skip("cantera no disponible")
        return
    r = run_v6(dx_um=100.0, max_iter=20_000)
    assert r["anchored"], r
    assert r["err"] < TOL_V6, r


def test_v6d_flame_methane():
    """V6d · el MISMO juez para el METANO (2S_CH4_BFER paso 1, el
    mecanismo que necesita V8/Sandia D): S_L vs Cantera ±10 %. La
    cinética CH4 ya pasó la triple verificación puntual (tabla ==
    Pyrometheus == Cantera); esto verifica la llama ACOPLADA.

    **LENTO** (~2 min) y a dx = 60 µm: la llama BFER de CH4 es más FINA
    que la WD2 de propano (orden 0.5 del fuel) y a 150 µm no se resuelve
    — medido: −19 % a 150 µm, −9.3 % a 100 µm, −8.9 % a 60 µm
    (convergida). El −9 % residual es el sesgo de c_p constante en el
    interior del frente, declarado; pasa el ±10 % SIN margen de sobra y
    por eso vive en el tier lento, no en CI con un umbral frágil."""
    if not SLOW:
        _skip("V6d metano necesita dx=60um (~2 min); PHYCB_SLOW=1. "
              "Medido 2026-08-02: S_L -8.9% (anclada)")
        return
    if _cantera() is None:
        _skip("cantera no disponible")
        return
    r = run_v6(dx_um=60.0, max_iter=25_000, fuel="methane")
    assert r["anchored"], r
    assert r["err"] < TOL_V6, r
    assert abs(r["T_max"] / r["T_ad"] - 1.0) < 0.02, r


def test_v6c_ignition_from_cold():
    """V6c · EL RIESGO #1: campo FRÍO + plan de encendido (parche caliente
    + rampa del pre-exponencial + sub-relajación) debe (a) encender, (b)
    anclar según el criterio integral y (c) dar la MISMA S_L que la llama
    inicializada quemada (±5 %: mismo solver, distinta historia)."""
    if _cantera() is None:
        _skip("cantera no disponible")
        return
    r_cold = run_v6(dx_um=150.0, max_iter=10_000, ignition=True)
    assert r_cold["anchored"], ("NO encendio o no anclo", r_cold)
    r_hot = run_v6(dx_um=150.0, max_iter=10_000, ignition=False)
    rel = abs(r_cold["S_L"] / r_hot["S_L"] - 1.0)
    assert rel < TOL_V6C, (r_cold["S_L"], r_hot["S_L"], rel)


# ==========================================================================
# V7 · llama de contradifusión (2D opuestos) vs Cantera  [lento]
# ==========================================================================
def run_v7(dx_mm: float = 0.25, max_iter: int = 15_000) -> dict:
    """Chorros opuestos aire↔propano diluido, tensión baja. Cantera
    resuelve la contradifusión AXISIMÉTRICA y la nuestra es PLANA — la
    comparación honesta a baja tensión es T_max (que tiende al equilibrio
    estequiométrico en ambas) y la POSICIÓN de la llama del lado del
    oxidante. Perfiles finos: cualitativos, declarado."""
    ct = _cantera()
    mech = chem.global_mechanism("propane")
    st = _premixed_state(1.0)
    length = 0.02
    dx = dx_mm * 1e-3
    u_j = 0.20
    m = CartesianMesh([0.0, 0.0], [length, length], dx)
    m.set_sdf(None)
    pr = 0.7
    # lado combustible: propano diluido en N2 (60/40 en masa) — evita la
    # zona ultra-rica donde el 1 paso con orden 0.1 es duro numéricamente
    y_fuel_side = dict(FUEL=0.4, O2=0.0, CO2=0.0, H2O=0.0, N2=0.6)
    y_air = dict(FUEL=0.0, O2=0.233, CO2=0.0, H2O=0.0, N2=0.767)
    cfg = SolverConfig(rho=st["rho_u"], nu=st["alpha_u"] * pr,
                       u_ref=1.5, l_ref=length, mach_art=0.2, cfl=0.4,
                       rk="rk2", max_iter=max_iter, tol=0.0,
                       residual_every=500)
    bc_f = dict(kind="inlet", u=(u_j, 0.0), T=300.0)
    bc_o = dict(kind="inlet", u=(-u_j, 0.0), T=300.0)
    for sp, v in y_fuel_side.items():
        bc_f[f"Y_{sp}"] = v
    for sp, v in y_air.items():
        bc_o[f"Y_{sp}"] = v
    bcs = make_bcs(xlo=bc_f, xhi=bc_o,
                   ylo=dict(kind="outlet", p=0.0),
                   yhi=dict(kind="outlet", p=0.0))
    s = IBRANSSolver(m, cfg, bcs)
    en = EnergyLowMach(m, EnergyConfig(
        p_thermo_Pa=101325.0, r_gas=8314.46 / st["mw"], T_inlet_K=300.0,
        prandtl=pr, nu_power_T=st["n_pow"], T_ref_transport_K=300.0))
    sp_t = chem.SpeciesTransport(m, chem.ChemConfig(
        fuel="propane", schmidt=pr, prandtl=pr))
    sp_t.calibrate_cp(st["t_ad"], st["y_u"]["FUEL"], 300.0)
    sp_t.attach(s, turbulence=None, energy=en, y_init=y_air,
                ignition=False)
    en.cfg.T_ceiling_K = 1.10 * st["t_ad"]
    # capa de mezcla YA formada en t=0 (combustible a la izquierda, aire a
    # la derecha) + franja caliente en el contacto: sin esto la franja se
    # enfría (~5 ms) antes de que el combustible llegue por convección
    # (~50 ms) y la llama nunca nace — pasó en la primera corrida
    xg = np.asarray(m.X[0])
    side = 0.5 * (1.0 - np.tanh((xg - 0.5 * length) / 2.0e-3))  # 1=fuel
    for spn in sp_t.cfg.species:
        fld = side * y_fuel_side[spn] + (1.0 - side) * y_air[spn]
        sp_t.Y[spn][...] = s.xp.asarray(
            np.broadcast_to(fld, m.shape_tot), dtype=s.dtype)
    hot = np.exp(-((xg - 0.5 * length) / 1.5e-3) ** 2)
    en.T[...] = s.xp.asarray(
        np.broadcast_to(300.0 + 1500.0 * hot, m.shape_tot), dtype=s.dtype)
    en.rho[...] = s.dtype(en.cfg.p_thermo_Pa) \
        / (s.dtype(en.cfg.r_gas) * en.T)
    s.u[0][...] = s.xp.asarray(
        np.broadcast_to(u_j * (1.0 - 2.0 * xg / length), m.shape_tot),
        dtype=s.dtype)
    s.apply_bc()
    info = s.run()
    T = np.asarray(to_numpy(en.T[s.mesh.interior]))
    jmid = T.shape[1] // 2
    prof = T[:, jmid]
    i_max = int(np.argmax(prof))
    # referencia Cantera: contradifusion axisimetrica, misma composicion
    gas = ct.Solution(yaml=chem.cantera_yaml(mech))
    f = ct.CounterflowDiffusionFlame(gas, width=length)
    f.transport_model = "unity-Lewis-number"
    f.P = ct.one_atm
    f.fuel_inlet.mdot = u_j * 1.10          # rho del lado fuel ~1.10 kg/m3
    f.fuel_inlet.Y = "C3H8:0.4, N2:0.6"
    f.fuel_inlet.T = 300.0
    f.oxidizer_inlet.mdot = u_j * st["rho_u"]
    f.oxidizer_inlet.Y = "O2:0.233, N2:0.767"
    f.oxidizer_inlet.T = 300.0
    f.set_refine_criteria(ratio=3, slope=0.1, curve=0.2)
    f.solve(loglevel=0, auto=True)
    t_max_ct = float(np.max(f.T))
    x_max_ct = float(f.grid[int(np.argmax(f.T))])
    return dict(case=f"V7 dx={dx_mm}mm", status=info["status"],
                iterations=info["iterations"],
                wall_time_s=info["wall_time_s"],
                T_max=float(prof[i_max]), T_max_ct=t_max_ct,
                err=abs(float(prof[i_max]) / t_max_ct - 1.0),
                x_flame_mm=(i_max + 0.5) * dx * 1e3,
                x_flame_ct_mm=x_max_ct * 1e3,
                anchored=sp_t.anchored())


def test_v7_counterflow():
    """V7 · contradifusión: T_max vs Cantera ±10 % y llama del lado del
    oxidante. **LENTO** (~5 min). Geometrías DISTINTAS (plana vs
    axisimétrica) — declarado: a baja tensión T_max es comparable;
    los perfiles finos no se comparan cuantitativamente."""
    if not SLOW:
        _skip("V7: ~6.4k celdas x 15k pasos + Cantera counterflow "
              "(~5 min); PHYCB_SLOW=1")
        return
    if _cantera() is None:
        _skip("cantera no disponible")
        return
    r = run_v7()
    assert r["T_max"] > 1500.0, ("la llama no encendio/se apago", r)
    assert r["err"] < TOL_V7, r
    assert r["x_flame_mm"] > 10.0, ("llama del lado del combustible", r)


def test_v8_sandia_d_pending():
    """V8 · Sandia Flame D: **PENDIENTE, DECLARADO** (regla de la casa: se
    salta con motivo visible, no se trivializa). Necesita mecanismo de
    metano en la tabla, piloto anular y ~0.5–1 M celdas 3D en GPU (~1–2 h).
    Plan en ESTADO.md §11."""
    _skip("V8 Sandia D pendiente: mecanismo CH4 + piloto + caso 3D grande "
          "(plan en ESTADO.md §11); no corrio en la campana 2026-08-02")


# ==========================================================================
# Doble modo (tabla propia + pytest)
# ==========================================================================
class _Skip(Exception):
    pass


def _skip(reason: str):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise _Skip(reason)


_TESTS = [
    ("M3a", "cinetica molar SI vs Cantera", test_m3a_kinetics_vs_cantera),
    ("M3b", "maquinaria de encendido", test_m3b_ignition_machinery),
    ("M3c", "Pyrometheus == tabla == Cantera",
     test_m3c_pyrometheus_equivalence),
    ("M3d", "reactor 0D: estequiometria y energia",
     test_m3d_batch_stoichiometry),
    ("V6a", "llama 1D S_L vs Cantera (150um)", test_v6a_flame_mini),
    ("V6b", "llama 1D S_L (100um, oficial) [lento]", test_v6b_flame_full),
    ("V6c", "ENCENDIDO desde frio (riesgo #1)",
     test_v6c_ignition_from_cold),
    ("V6d", "llama CH4 60um vs Cantera [lento]", test_v6d_flame_methane),
    ("V7", "contradifusion vs Cantera [lento]", test_v7_counterflow),
    ("V8", "Sandia D [PENDIENTE declarado]", test_v8_sandia_d_pending),
]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only = None
    for a in argv:
        if a.startswith("--only"):
            only = set(a.split("=", 1)[-1].split(","))
    fails = 0
    print(f"\nEscalera V&V del hito M3 (WAVE6) · slow={SLOW}\n" + "-" * 72)
    for tid, name, fn in _TESTS:
        if only and tid not in only:
            continue
        t0 = time.perf_counter()
        try:
            fn()
            mark, note = "OK  ", ""
        except _Skip as exc:
            mark, note = "SKIP", f"  ({exc})"
        except BaseException as exc:
            if type(exc).__name__ == "Skipped":
                mark, note = "SKIP", f"  ({exc})"
            else:
                mark, note = "FAIL", f"  ({type(exc).__name__}: {exc})"
                fails += 1
        el = time.perf_counter() - t0
        print(f"  [{mark}] {tid:5s} {name:44s} {el:7.1f} s{note}")
    print("-" * 72)
    print(f"  {'TODO EN VERDE' if not fails else str(fails) + ' FALLO(S)'}\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
