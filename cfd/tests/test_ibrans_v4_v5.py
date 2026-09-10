#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/test_ibrans_v4_v5.py
==============================================
Escalera de **verificación** del hito M2 de WAVE6 (turbulencia + energía +
ρ variable): tests unitarios con juez analítico (M2a–M2e) y los casos de
validación de modelo V4–V5 de Ciencia §13.1.

    M2a  decaimiento homogéneo de k-ε ......... exponente analítico < 1 %
    M2b  canal turbulento plano ............... Cf vs Dean (1978) ±15 %
    M2c  expansión térmica low-Mach ........... analítico (T, u, ρu) ±3 %
    M2d  funciones puras (C_μ, f_r, Spalding) . identidades y límites
    M2e  escalar pasivo ....................... conservación y cotas
    V4   backward-facing step ................. x_r/h = 6.67 ±15 %
    V5   chorro turbulento plano libre ........ S = dy½/dx = 0.10 ±15 %

Qué corre en CI y qué NO (regla de la casa: lo que no cabe se SALTA con
motivo visible, jamás se convierte en un check trivial)
---------------------------------------------------------------------------
* **Siempre**: M2a–M2e, V4 mini (dx = h/5, dominio corto — comprueba que
  hay recirculación y una x_r de cordura; el ±15 % NO se comprueba aquí
  porque a h/5 y con 15k iteraciones el caso no está convergido y se
  DECLARA), V5 mini (ídem con banda ancha de cordura).
* **Sólo con `PHYCB_SLOW=1`** (o `--slow`): V4 completo (dx = h/8, ~30–40
  min CPU) y V5 completo (dx = d/5, ~15 min CPU) con los umbrales de ±15 %
  de la wave. Scripts manuales: `run_v4_bfs.py` y `run_v5_jet.py`. Sus
  números medidos están en `cfd/ibrans/ESTADO.md`.

Referencias (contrato §7: nada hard-coded sin fuente)
-----------------------------------------------------
* V4: Vogel, J.C. & Eaton, J.K. (1985) «Combined heat transfer and fluid
  dynamic measurements downstream of a backward-facing step», J. Heat
  Transfer 107, 922–929: Re_h = 28 000, ER = 1.25, x_r/h = 6.67 ± 0.1 —
  el Re que pide WAVE6 §2. (Driver & Seegmiller 1985 da 6.26 con
  ER = 1.125 y Re 37.5k: distinta geometría, no se mezclan.)
* V5: Pope, S.B. (2000) «Turbulent Flows» §5.4: chorro PLANO
  autosemejante, S = dy½/dx ≈ 0.10 (banda experimental 0.096–0.11).
* M2b: Dean, R.B. (1978) J. Fluids Eng. 100, 215–223:
  Cf = τ_w/(½ρU_b²) = 0.073·Re_b^{−1/4}, Re_b = U_b·2δ/ν.
* M2a: dk/dt = −ε, dε/dt = −C₂ε²/k ⇒ k ∝ (1+t/t₀)^{−1/(C₂−1)} (exacto).
* M2c: flujo tapón con fuente S y p₀ constante ⇒
  T_out = √(T₀² + 2·S·T₀·L/u₀), u_out = u₀·T_out/T₀ (gas ideal).

Los casos M2 estacionarios usan `mach_art` 0.15–0.3: la pseudo-compresibi-
lidad NO cambia el estado estacionario (∂p/∂t = 0 ⇒ ∇·u_f = q_T con el
mismo operador espacial), sólo el camino — así que subir el Mach artificial
acelera sin tocar la respuesta. En los no estacionarios (V3) sigue 0.1.

Doble modo::

    python cfd/tests/test_ibrans_v4_v5.py           # tabla ✓/✗/–
    python cfd/tests/test_ibrans_v4_v5.py --slow    # + V4/V5 completos
    python -m pytest cfd/tests/test_ibrans_v4_v5.py -q
"""

from __future__ import annotations

import os
import sys
import time

# --- contrato §0.b: Phy-AC ensombrece los módulos desde cualquier cwd
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

pc.assert_provenance(pc)          # ImportError ruidoso si es el de Phy-AC

from cfd.ibrans import turbulence as tb                       # noqa: E402
from cfd.ibrans.backend import to_numpy                       # noqa: E402
from cfd.ibrans.energy import (EnergyConfig, EnergyLowMach,   # noqa: E402
                               PassiveScalar, ScalarConfig)
from cfd.ibrans.mesh import CartesianMesh, sdf_box            # noqa: E402
from cfd.ibrans.solver import (IBRANSSolver, SolverConfig,    # noqa: E402
                               make_bcs)
from cfd.ibrans.turbulence import (KEpsilonRealizable,        # noqa: E402
                                   TurbulenceConfig)

SLOW = bool(os.environ.get("PHYCB_SLOW")) or "--slow" in sys.argv

# --- referencias con cita (ver docstring) --------------------------------
BFS_XR_REF = 6.67          # Vogel & Eaton (1985), Re_h=28k, ER=1.25
JET_S_REF = 0.10           # Pope (2000) §5.4, chorro plano
TOL_V4 = 0.15
TOL_V5 = 0.15
TOL_M2B = 0.15             # Cf vs Dean (la correlación misma es ±5-10 %)
TOL_M2C = 0.03


# ==========================================================================
# M2a · decaimiento homogéneo (juez analítico exacto)
# ==========================================================================
def test_m2a_homogeneous_decay():
    """Sin flujo medio ni paredes: dk/dt = −ε y dε/dt = −C₂ε²/k tienen
    solución exacta en ley de potencia. Verifica el integrador
    point-implicit de las fuentes y las constantes del realizable."""
    m = CartesianMesh([0.0, 0.0], [0.8, 0.8], 0.1)
    m.set_sdf(None)
    cfg = SolverConfig(rho=1.0, nu=1e-9, u_ref=1.0, l_ref=1.0,
                       max_iter=10, tol=0.0)
    bcs = make_bcs(xlo="periodic", xhi="periodic",
                   ylo="periodic", yhi="periodic")
    s = IBRANSSolver(m, cfg, bcs)
    tur = KEpsilonRealizable(m, TurbulenceConfig()).attach(s)
    k0, e0 = 1.0, 2.0
    tur.k[...] = k0
    tur.eps[...] = e0
    dt, t = 1.0e-3, 0.0
    for _ in range(4000):
        tur.advance(s, dt)
        t += dt
    k_num = float(np.mean(np.asarray(to_numpy(tur.k[m.interior]))))
    n_teo = 1.0 / (tb.C2_EPS - 1.0)
    t0 = k0 / ((tb.C2_EPS - 1.0) * e0)
    k_teo = k0 * (1.0 + t / t0) ** (-n_teo)
    err = abs(k_num / k_teo - 1.0)
    assert err < 0.01, (k_num, k_teo, err)
    # sin flujo medio la velocidad no debe moverse
    assert float(np.max(np.abs(s.field(0)))) == 0.0


# ==========================================================================
# M2b · canal turbulento plano vs Dean (1978)
# ==========================================================================
def run_m2b(ny: int = 16, max_iter: int = 12_000) -> dict:
    """Canal periódico impulsado por fuerza: τ_w = f·δ EXACTO por balance
    (no depende del modelo), así que medir U_b y comparar Cf con Dean
    verifica la ley de pared + el transporte, no la medida."""
    delta = 1.0
    nu = 2.0e-5
    force = 1.0e-3
    u_tau = (force * delta) ** 0.5
    dx = 2.0 * delta / ny
    m = CartesianMesh([0.0, 0.0], [4 * dx, 2.0 * delta], dx)
    m.set_sdf(None)
    cfg = SolverConfig(rho=1.0, nu=nu, u_ref=0.7, l_ref=delta,
                       mach_art=0.3, cfl=0.4, rk="rk2",
                       body_force=(force, 0.0), max_iter=max_iter,
                       tol=1.0e-5, residual_every=200)
    bcs = make_bcs(xlo="periodic", xhi="periodic", ylo="wall", yhi="wall")
    s = IBRANSSolver(m, cfg, bcs)
    tur = KEpsilonRealizable(m, TurbulenceConfig(
        intensity=0.05, length_scale_m=0.1)).attach(s)
    # arranque con perfil log (acorta el transitorio; el estacionario es el
    # mismo — se comprobó arrancando plano: Cf −8.1 % en 44k it)
    yg = m.X[1]
    yw = np.maximum(np.minimum(np.abs(yg), np.abs(2 * delta - yg)), dx / 4)
    up0 = np.log(tb.E_WALL * np.maximum(u_tau * yw / nu, 1.0)) / tb.KAPPA
    s.u[0][...] = s.xp.asarray(np.broadcast_to(u_tau * up0, m.shape_tot),
                               dtype=s.dtype)
    s.apply_bc()
    info = s.run()
    u = s.field(0)
    ub = float(np.mean(u))
    cf = force * delta / (0.5 * ub ** 2)
    re_b = ub * 2 * delta / nu
    cf_dean = 0.073 * re_b ** -0.25
    return dict(case=f"M2b canal ny={ny}", status=info["status"],
                iterations=info["iterations"], res_u=info["res_u"],
                wall_time_s=info["wall_time_s"], U_b=ub, Re_b=re_b,
                Cf=cf, Cf_dean=cf_dean, err=abs(cf / cf_dean - 1.0),
                yplus=u_tau * (dx / 2) / nu, ystar=tur.ystar_stats(),
                x_uniform=float(np.max(np.abs(u - u.mean(axis=0)))))


def test_m2b_channel_dean():
    r = run_m2b()
    # el canal es 1D en x: la solución no puede depender de x
    assert r["x_uniform"] < 1e-10, r
    assert r["yplus"] > 30.0, ("primera celda fuera de banda", r)
    assert r["ystar"]["frac_in_target"] == 1.0, r["ystar"]
    assert r["err"] < TOL_M2B, r
    # cuasi-estacionario aunque no llegue a tol (declarado: el acoplamiento
    # segregado deja un ciclo límite pequeño en el residual)
    assert r["res_u"] < 5.0e-4, r


# ==========================================================================
# M2c · expansión térmica low-Mach (juez analítico)
# ==========================================================================
def run_m2c(nx: int = 48, max_iter: int = 40_000) -> dict:
    L, u0, t0k = 1.0, 1.0, 300.0
    dx = L / nx
    m = CartesianMesh([0.0, 0.0], [L, 8 * dx], dx)
    m.set_sdf(None)
    cfg = SolverConfig(rho=1.2, nu=1.0e-5, u_ref=u0, l_ref=L,
                       mach_art=0.15, cfl=0.4, rk="rk2",
                       max_iter=max_iter, tol=2.0e-6, residual_every=100)
    bcs = make_bcs(xlo=dict(kind="inlet", u=(u0, 0.0), T=t0k),
                   xhi=dict(kind="outlet", p=0.0),
                   ylo="slip", yhi="slip")
    s = IBRANSSolver(m, cfg, bcs)
    en = EnergyLowMach(m, EnergyConfig(T_inlet_K=t0k)).attach(s)
    xa, xb, s_amp = 0.3, 0.6, 400.0
    src = np.where((np.asarray(m.X[0]) >= xa)
                   & (np.asarray(m.X[0]) <= xb), s_amp, 0.0)
    en.source_K_s = s.xp.asarray(np.broadcast_to(src, m.shape_tot).copy(),
                                 dtype=s.dtype)
    s.u[0][...] = u0
    s.apply_bc()
    info = s.run()
    u = s.field(0)
    T = np.asarray(to_numpy(en.T[m.interior]), dtype=float)
    rho = np.asarray(to_numpy(en.rho[m.interior]), dtype=float)
    j = 4
    t_out_a = (t0k ** 2 + 2.0 * s_amp * t0k * (xb - xa) / u0) ** 0.5
    u_out_a = u0 * t_out_a / t0k
    return dict(case="M2c expansion termica", status=info["status"],
                iterations=info["iterations"],
                wall_time_s=info["wall_time_s"],
                T_out=float(T[-1, j]), T_out_a=t_out_a,
                u_out=float(u[-1, j]), u_out_a=u_out_a,
                err_T=abs(T[-1, j] / t_out_a - 1.0),
                err_u=abs(u[-1, j] / u_out_a - 1.0),
                flux_err=abs(rho[-1, j] * u[-1, j]
                             / (rho[0, j] * u[0, j]) - 1.0))


def test_m2c_thermal_expansion():
    r = run_m2c()
    assert r["status"] == "converged", r
    assert r["err_T"] < TOL_M2C, r
    assert r["err_u"] < TOL_M2C, r
    assert r["flux_err"] < 0.01, r      # ρu constante (continuidad)


# ==========================================================================
# M2d · funciones puras del cierre (identidades y límites)
# ==========================================================================
def test_m2d_closure_functions():
    """C_μ realizable, corrección de curvatura y las dos inversiones de
    Spalding: identidades comprobables sin solver."""
    # (a) banda logarítmica en cizalla pura: Sk/ε ≈ 3.3, W = 0 ⇒ C_μ ≈ 0.09
    cmu = float(tb.c_mu_realizable(3.3, 3.3, 1.0, 1.0, w_star=0.0))
    assert abs(cmu - 0.09) < 0.01, cmu
    # (b) f_r: cizalla pura = 1; rotación sólida → 0; deformación pura
    #     recortada al techo
    assert abs(float(tb.curvature_correction(1, 1, 2.0, 2.0)) - 1.0) < 1e-12
    assert float(tb.curvature_correction(1, 1, 1e-9, 2.0)) < 1e-6
    assert abs(float(tb.curvature_correction(1, 1, 2.0, 1e-30))
               - tb.FR_MAX) < 1e-12
    # (c) u⁺(y*) invierte la ley de Spalding con residuo de máquina
    for ys in (5.0, 30.0, 300.0, 3000.0):
        up = float(tb.u_plus_of_ystar(ys))
        assert abs(float(tb.spalding_yplus(up)) - ys) / ys < 1e-6, ys
    # (d) límite laminar: y* pequeño ⇒ u⁺ ≈ y* ⇒ ν_wall ≈ 0
    up = float(tb.u_plus_of_ystar(0.5))
    assert abs(up - 0.5) < 0.01
    # (e) coherencia de las dos inversiones: u_tau_spalding(u_p, y) y
    #     u⁺(y*) describen la MISMA ley
    nu = 1.5e-5
    ut = float(tb.u_tau_spalding(30.0, 5.0e-3, nu))
    ys = ut * 5.0e-3 / nu
    assert abs(float(tb.u_plus_of_ystar(ys)) - 30.0 / ut) < 1e-4


# ==========================================================================
# M2e · escalar pasivo: conservación y cotas
# ==========================================================================
def test_m2e_passive_scalar():
    """Conducto laminar con Z de entrada: aguas abajo el escalar debe
    quedar EXACTAMENTE en el valor de entrada (0.35) y dentro de [0,1]."""
    L, u0 = 1.0, 1.0
    dx = L / 32
    m = CartesianMesh([0.0, 0.0], [L, 6 * dx], dx)
    m.set_sdf(None)
    cfg = SolverConfig(rho=1.0, nu=1e-5, u_ref=u0, l_ref=L, mach_art=0.15,
                       cfl=0.4, rk="rk2", max_iter=8_000, tol=0.0)
    bcs = make_bcs(xlo=dict(kind="inlet", u=(u0, 0.0), Z=0.35),
                   xhi=dict(kind="outlet", p=0.0),
                   ylo="slip", yhi="slip")
    s = IBRANSSolver(m, cfg, bcs)
    sc = PassiveScalar(m, ScalarConfig()).attach(s)
    s.u[0][...] = u0
    s.apply_bc()
    s.run(t_end=3.0)
    z = np.asarray(to_numpy(sc.Z[m.interior]), dtype=float)
    assert abs(float(z[-1, 3]) - 0.35) < 1e-6, z[-1, 3]
    assert float(z.min()) >= 0.0 and float(z.max()) <= 1.0 + 1e-12


# ==========================================================================
# V4 · backward-facing step (Vogel & Eaton 1985)
# ==========================================================================
def case_bfs(dx_inv: float = 8.0, x_out: float = 25.0,
             max_iter: int = 120_000, tol: float = 5.0e-6):
    """Escalón hacia atrás con el ESCALÓN COMO BLOQUE IB (prueba las wall
    functions sobre |φ|); suelo y techo son paredes del dominio (prueban
    las wall functions de contorno). ER = 1.25, Re_h = 28 000, entrada con
    capa límite 1/7 de δ = 1.1h (la condición de Vogel & Eaton)."""
    h, u0 = 1.0, 1.0
    nu = u0 * h / 28_000.0
    x_in = -4.0
    dx = h / dx_inv
    m = CartesianMesh([x_in, 0.0], [x_out, 5.0 * h], dx)
    m.set_sdf(sdf_box([x_in - 1.0, -1.0], [0.0, h], solid=True))
    k_in = 1.5 * (0.05 * u0) ** 2
    eps_in = tb.C_MU ** 0.75 * k_in ** 1.5 / (0.07 * 4 * h)
    yv = m.xc[1]
    low = np.clip((yv - h) / (1.1 * h), 1e-6, 1.0)
    top = np.clip((5.0 * h - yv) / (0.3 * h), 1e-6, 1.0)
    prof = u0 * np.minimum(low ** (1.0 / 7.0), top ** (1.0 / 7.0))
    prof[yv < h] = 0.0
    cfg = SolverConfig(rho=1.0, nu=nu, u_ref=u0, l_ref=h, mach_art=0.25,
                       cfl=0.4, rk="rk2", max_iter=max_iter, tol=tol,
                       residual_every=200)
    bcs = make_bcs(xlo=dict(kind="inlet", u=(prof, 0.0),
                            k=k_in, eps=eps_in),
                   xhi=dict(kind="outlet", p=0.0),
                   ylo="wall", yhi="wall")
    s = IBRANSSolver(m, cfg, bcs)
    tur = KEpsilonRealizable(m, TurbulenceConfig(
        intensity=0.05, length_scale_m=0.07 * 4 * h)).attach(s)
    s.u[0][...] = s.xp.asarray(
        np.broadcast_to(prof, m.shape_tot), dtype=s.dtype)
    s.apply_bc()
    return s, tur, dict(h=h, dx=dx, x_in=x_in)


def reattachment_length(s, ex) -> float:
    """x_r: último cruce − → + de u en la primera fila sobre el suelo."""
    u = s.field(0)
    xc = s.coords()[0]
    row = u[:, 0]
    for i in range(len(xc) - 1, 0, -1):
        if xc[i] <= 0.2:
            break
        if row[i - 1] < 0.0 <= row[i]:
            return float(xc[i - 1] + ex["dx"] * row[i - 1]
                         / (row[i - 1] - row[i]))
    return float("nan")


def run_v4(dx_inv: float = 8.0, x_out: float = 25.0,
           max_iter: int = 120_000) -> dict:
    s, tur, ex = case_bfs(dx_inv=dx_inv, x_out=x_out, max_iter=max_iter)
    info = s.run()
    x_r = reattachment_length(s, ex)
    u = s.field(0)
    return dict(case=f"V4 BFS dx=h/{dx_inv:.0f}", status=info["status"],
                iterations=info["iterations"], res_u=info["res_u"],
                wall_time_s=info["wall_time_s"],
                s_per_cell_step=info["s_per_cell_step"],
                cells=info["cells"], x_r=x_r,
                err=abs(x_r / BFS_XR_REF - 1.0),
                u_min_floor=float(np.min(u[:, 0])),
                ystar=tur.ystar_stats())


def test_v4_bfs_mini():
    """V4 (malla mini h/5, dominio corto, 15k it) · humo de CI: existe la
    burbuja de recirculación y x_r cae en una banda ANCHA de cordura. El
    ±15 % del criterio de la wave vive en `test_v4_bfs_full` (lento):
    a h/5 con 15k iteraciones el caso NO está convergido — medido
    (x_r crece de 4.9 a 5.7 entre 20k y 40k it) y declarado."""
    r = run_v4(dx_inv=5.0, x_out=15.0, max_iter=15_000)
    assert r["u_min_floor"] < -0.05, ("sin recirculacion", r)
    assert 3.0 < r["x_r"] < 9.0, r
    assert r["ystar"]["frac_in_target"] > 0.5, r["ystar"]


def test_v4_bfs_full():
    """V4 · x_r/h = 6.67 ±15 % (Vogel & Eaton). **LENTO** (~35 min CPU):
    sólo con `PHYCB_SLOW=1`. Script manual: `run_v4_bfs.py`."""
    if not SLOW:
        _skip("V4 completo: ~9.3k celdas x 120k pasos (~35 min CPU); corre "
              "con PHYCB_SLOW=1 o el script run_v4_bfs.py")
        return
    r = run_v4(dx_inv=8.0, x_out=25.0, max_iter=120_000)
    assert r["err"] < TOL_V4, r


# ==========================================================================
# V5 · chorro turbulento plano libre
# ==========================================================================
def case_jet(dx_inv: float = 5.0, x_out: float = 40.0, half_w: float = 13.0,
             max_iter: int = 60_000):
    """Chorro plano de ranura d con co-flujo débil (0.04·u_j, por
    estabilidad del contorno lateral; la medida usa el EXCESO de velocidad
    así que el co-flujo no sesga la tasa de expansión)."""
    d, u_j = 1.0, 1.0
    u_cof = 0.04 * u_j
    nu = u_j * d / 30_000.0
    dx = d / dx_inv
    m = CartesianMesh([0.0, -half_w], [x_out, half_w], dx)
    m.set_sdf(None)
    yv = m.xc[1]
    prof = np.where(np.abs(yv) <= 0.5 * d, u_j, u_cof)
    k_in = 1.5 * (0.05 * u_j) ** 2
    eps_in = tb.C_MU ** 0.75 * k_in ** 1.5 / (0.07 * d)
    cfg = SolverConfig(rho=1.0, nu=nu, u_ref=u_j, l_ref=d, mach_art=0.25,
                       cfl=0.4, rk="rk2", max_iter=max_iter, tol=5.0e-6,
                       residual_every=200)
    bcs = make_bcs(xlo=dict(kind="inlet", u=(prof, 0.0),
                            k=k_in, eps=eps_in),
                   xhi=dict(kind="outlet", p=0.0),
                   ylo="slip", yhi="slip")
    s = IBRANSSolver(m, cfg, bcs)
    tur = KEpsilonRealizable(m, TurbulenceConfig(
        intensity=0.05, length_scale_m=0.07 * d)).attach(s)
    s.u[0][...] = u_cof
    s.apply_bc()
    return s, tur, dict(d=d, u_j=u_j, u_cof=u_cof, dx=dx, x_out=x_out)


def jet_spreading(s, ex, x_fit=(12.0, None)) -> dict:
    """Anchura media y½(x) del exceso de velocidad y su pendiente S."""
    u = s.field(0)
    xc, yc = s.coords()
    jmid = len(yc) // 2
    rows = []
    x_hi = ex["x_out"] - 4.0 if x_fit[1] is None else x_fit[1]
    for xq in np.arange(5.0, ex["x_out"] - 2.0, 1.0):
        i = int(round(xq / ex["dx"] - 0.5))
        du = u[i, :] - ex["u_cof"]
        duc = 0.5 * (du[jmid - 1] + du[jmid]) if len(yc) % 2 == 0 \
            else du[jmid]
        half = 0.5 * duc
        jj = None
        for j in range(jmid, len(yc) - 1):
            if du[j] >= half > du[j + 1]:
                jj = j
                break
        if jj is None:
            continue
        y12 = yc[jj] + (half - du[jj]) * (yc[jj + 1] - yc[jj]) \
            / (du[jj + 1] - du[jj])
        rows.append((xq, y12, duc))
    rows = np.asarray(rows)
    if rows.size == 0 or rows.shape[0] < 6:
        return dict(ok=False, reason="perfil de chorro no medible")
    sel = (rows[:, 0] >= x_fit[0]) & (rows[:, 0] <= x_hi)
    a = np.polyfit(rows[sel, 0], rows[sel, 1], 1)
    inv2 = (ex["u_j"] / np.maximum(rows[:, 2], 1e-9)) ** 2
    b = np.polyfit(rows[sel, 0], inv2[sel], 1)
    return dict(ok=True, S=float(a[0]), x0=float(-a[1] / a[0]),
                decay_slope=float(b[0]), n_rows=int(rows.shape[0]))


def run_v5(dx_inv: float = 5.0, x_out: float = 40.0, half_w: float = 13.0,
           max_iter: int = 60_000, x_fit=(12.0, None)) -> dict:
    s, tur, ex = case_jet(dx_inv=dx_inv, x_out=x_out, half_w=half_w,
                          max_iter=max_iter)
    info = s.run()
    sp = jet_spreading(s, ex, x_fit=x_fit)
    return dict(case=f"V5 chorro dx=d/{dx_inv:.0f}", status=info["status"],
                iterations=info["iterations"], res_u=info["res_u"],
                wall_time_s=info["wall_time_s"],
                s_per_cell_step=info["s_per_cell_step"],
                cells=info["cells"], **sp,
                err=abs(sp.get("S", float("nan")) / JET_S_REF - 1.0))


def test_v5_jet_mini():
    """V5 (malla mini d/3, dominio corto) · humo de CI: el chorro se abre
    linealmente y S cae en banda ANCHA. El ±15 % vive en el lento."""
    r = run_v5(dx_inv=3.0, x_out=25.0, half_w=9.0, max_iter=20_000,
               x_fit=(8.0, 21.0))
    assert r["ok"], r
    assert 0.06 < r["S"] < 0.14, r
    assert r["decay_slope"] > 0.0, ("u_c no decae de forma autosemejante", r)


def test_v5_jet_full():
    """V5 · S = 0.10 ±15 % (Pope 2000, chorro plano). **LENTO** (~15 min
    CPU): sólo con `PHYCB_SLOW=1`. Script manual: `run_v5_jet.py`."""
    if not SLOW:
        _skip("V5 completo: ~26k celdas x 60k pasos (~15 min CPU); corre "
              "con PHYCB_SLOW=1 o el script run_v5_jet.py")
        return
    r = run_v5(dx_inv=5.0, x_out=40.0, half_w=13.0, max_iter=60_000)
    assert r["ok"], r
    assert r["err"] < TOL_V5, r


# ==========================================================================
# M2f · los ganchos M2 son opt-in (M1 intacto)
# ==========================================================================
def test_m2f_hooks_are_opt_in():
    """Sin modelos registrados el solver es el de M1: sin ν_t, sin ρ
    variable, sin fuente de divergencia — y lo dice en su summary. La
    aritmética idéntica la vigila la escalera V1–V3 completa."""
    m = CartesianMesh([0.0, 0.0], [1.0, 1.0], 0.125)
    m.set_sdf(None)
    s = IBRANSSolver(m, SolverConfig(max_iter=5, tol=0.0),
                     make_bcs(xlo="wall", xhi="wall", ylo="wall",
                              yhi=dict(kind="wall", u=(1.0, 0.0))))
    assert s.nu_field is None and s.rho_field is None
    assert s.div_source is None and s.physics == []
    assert s.summary()["solver"] == "ibrans-M1"
    s.run()
    # con modelos: el summary lo declara
    tur = KEpsilonRealizable(m, TurbulenceConfig()).attach(s)
    assert s.summary()["solver"] == "ibrans-M2"
    assert s.summary()["physics"] == ["k-eps-realizable"]
    assert tb.available() is True


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
    ("M2a", "decaimiento homogeneo k-eps (analitico)",
     test_m2a_homogeneous_decay),
    ("M2b", "canal turbulento vs Dean (wall functions)",
     test_m2b_channel_dean),
    ("M2c", "expansion termica low-Mach (analitico)",
     test_m2c_thermal_expansion),
    ("M2d", "funciones puras del cierre", test_m2d_closure_functions),
    ("M2e", "escalar pasivo: conservacion y cotas", test_m2e_passive_scalar),
    ("M2f", "ganchos M2 opt-in (M1 intacto)", test_m2f_hooks_are_opt_in),
    ("V4a", "BFS mini (humo: recirculacion)", test_v4_bfs_mini),
    ("V4b", "BFS h/8: x_r +-15% [lento]", test_v4_bfs_full),
    ("V5a", "chorro mini (humo: expansion lineal)", test_v5_jet_mini),
    ("V5b", "chorro d/5: S +-15% [lento]", test_v5_jet_full),
]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only = None
    for a in argv:
        if a.startswith("--only"):
            only = set(a.split("=", 1)[-1].split(","))
    fails = 0
    print(f"\nEscalera V&V del hito M2 (WAVE6) · slow={SLOW}\n" + "-" * 72)
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
