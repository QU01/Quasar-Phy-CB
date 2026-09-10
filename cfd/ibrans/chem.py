"""
QUASAR Phy-CB · cfd/ibrans/chem.py
==================================
**Hito M3 de WAVE6 — transporte de especies + cierre EDM + encendido.**
Verificado por la escalera M3a–M3c y V6 (`cfd/tests/test_ibrans_v6_v8.py`);
números medidos en `ESTADO.md` §11.

Qué resuelve
------------
Para cada especie Y_k (FUEL, O2, CO2, H2O y N2 como cierre)::

    DY_k/Dt = ∇·(D_eff ∇Y_k) + ω_k/ρ,     D_eff = ν/Sc + ν_t/Sc_t

con el MISMO kernel MUSCL+TVD y la MISMA uf de Rhie–Chow que el momento
(`solver.advect_diffuse_scalar`). El término fuente es el cierre **EDM**
(Magnussen & Hjertager 1976) limitado por la cinética global::

    ω_F = min( A·ρ·(ε/k)·min(Y_F, Y_O/s, B·Y_P/(1+s)) , ω_Arrhenius )

y sin turbulencia enganchada (V6/V7, llamas laminares) ω_F es la cinética
pura. La liberación de calor entra a la energía como fuente volumétrica
S_T = ω_F·LHV/(ρ·c_p) — por el MISMO gancho (`EnergyLowMach.source_K_s`)
que arrastra la dilatación térmica a la continuidad low-Mach, así que la
expansión de la llama sale sola de la termodinámica del M2.

Unidades de la cinética (bug corregido, 2026-08-02)
---------------------------------------------------
La tabla `MECHANISMS` da A en la convención cgs-mol de Westbrook & Dryer
((mol/cm³)^(1−a−b)/s) y **Ea en J/kmol** (1.256e8 J/kmol = 30 kcal/mol
para el propano). La versión esqueleto de este módulo dividía Ea entre
R = 8.314 J/(mol·K): el exponente salía ~7500 y la tasa era CERO absoluto
a cualquier temperatura. Nunca se notó porque la tasa jamás se acopló —
exactamente la clase de bug latente que la escalera V6 caza, porque ahora
el MISMO mecanismo (mismos A, Ea, órdenes) se le da a Cantera y cualquier
error de conversión aparece como desacuerdo en S_L. `arrhenius_rate` ahora
es MOLAR y en SI, con la conversión escrita y testeada (M3a).

El plan de encendido (riesgo #1 del proyecto, WAVE6 §3)
-------------------------------------------------------
`IgnitionPlan` está IMPLEMENTADO en `SpeciesTransport.advance`:
parche caliente esférico impuesto `n_hold` iteraciones + rampa del
pre-exponencial (`ramp_from`→1 en `n_ramp`) + limitador de T (el techo de
`EnergyLowMach` se fija a `T_max_factor`·T_ad al enganchar) +
sub-relajación del término fuente (`relax`) + criterio de llama anclada
(liberación de calor integral estable ±`stable_tol` durante `n_stable`
iteraciones, `anchored()`). La salida de emergencia *mixed-is-burned*
(`equilibrium_T_of_Z`) sigue escrita y usable — la decisión de la semana
15 no se pospone (WAVE6 §7).

Lo que NO hace (declarado)
--------------------------
* **Pyrometheus NO está instalado y no se instala** (regla del proyecto):
  la vía `pyrometheus_rates()` degrada con motivo. Las tasas son las de la
  tabla global de 1 paso efectivo.
* Sólo se acopla el **paso 1 efectivo** (fuel → CO2+H2O completo): el CO
  del paso 2 de los esquemas BFER/WD2 queda para el upgrade EDC/2-pasos.
  Consecuencia declarada: sin predicción de CO ni de η_comb por CO.
* c_p CONSTANTE por caso (`ChemConfig.cp_J_kgK`, calibrable a T_ad): los
  perfiles finos de T dentro de la llama heredan ese sesgo. V6 lo mide.
* H₂ premezclado/micromix EXCLUIDO (WAVE6 §0): `global_mechanism("h2")`
  devuelve `n_steps = 0` a propósito.

API pública
-----------
    available() -> bool                      (True: transporte operativo)
    STATUS · MECHANISMS · R_J_PER_KMOL_K
    global_mechanism(fuel) · stoichiometric_ratio(fuel)
    product_split(mech) -> dict              [M3]
    arrhenius_rate(T, rho, Yf, Yo, mech, step=0, a_factor=1, xp=np)
    edm_rate(...) · edm_source(...)
    cantera_yaml(mech) -> str                [M3: el MISMO mecanismo a Cantera]
    pyrometheus_rates(mech) -> (None, str)   [degradación etiquetada]
    IgnitionPlan · ignition_patch(mesh, plan)
    equilibrium_T_of_Z(...)
    ChemConfig · SpeciesTransport(mesh, cfg).attach(solver, turbulence,
                                                    energy, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- constantes universales ---------------------------------------------
R_UNIVERSAL = 8.31446261815324          # J/(mol·K)
R_J_PER_KMOL_K = 8314.46261815324       # J/(kmol·K) — el de la cinética

#: Pesos moleculares kg/kmol de las especies del esquema efectivo.
MW = {"O2": 31.998, "CO2": 44.01, "H2O": 18.015, "N2": 28.014}

# --- coeficientes EDM de Magnussen & Hjertager (1976) -------------------
#: ⚠️ PROVISIONALES a efectos del contrato §7: son los valores clásicos de
#: la literatura (A = 4.0, B = 0.5) y se recalibran si el V9 lo exige.
EDM_A = 4.0
EDM_B = 0.5

STATUS = {
    "global_mechanism": "M3 · IMPLEMENTADO (tabla con LHV y atomos)",
    "arrhenius_rate": ("M3 · IMPLEMENTADO molar SI (bug de unidades Ea "
                       "J/kmol vs J/mol CORREGIDO 2026-08-02; verificado "
                       "contra Cantera con el mismo mecanismo en V6)"),
    "edm_rate": "M3 · IMPLEMENTADO (funcion pura)",
    "species_transport": ("M3 · IMPLEMENTADO (SpeciesTransport acoplado: "
                          "adveccion MUSCL + D_eff + fuente sub-relajada "
                          "point-implicit)"),
    "energy_with_formation": ("M3 · IMPLEMENTADO via source_K_s de "
                              "EnergyLowMach (S_T = omega·LHV/(rho·cp); "
                              "arrastra la dilatacion a la continuidad)"),
    "pyrometheus_rates": ("M3 · DEGRADA con motivo (Pyrometheus no "
                          "instalado y no se instala — regla del proyecto)"),
    "ignition": ("M3 · IMPLEMENTADO (parche + rampa + limitador T + "
                 "sub-relajacion + criterio de anclaje). El RIESGO #1 es "
                 "su ROBUSTEZ en el combustor real: se juzga en M4"),
    "mixed_is_burned": "M3 · salida de emergencia escrita y usable",
}


def available() -> bool:
    """¿Hay combustión operativa? **Sí** (M3): transporte de especies +
    EDM/Arrhenius + liberación de calor + encendido. El producto L2a
    completo sigue necesitando M4 (caso del combustor)."""
    return True


# ==========================================================================
# 1. Mecanismos globales
# ==========================================================================
#: Mecanismos globales de 1–2 pasos. Coeficientes A en la convención
#: cgs-mol de Westbrook & Dryer ((mol/cm³)^(1−a−b)/s) y **Ea en J/kmol**.
#: ⚠️ PROVISIONALES hasta su verificación V6 por combustible (el propano
#: está verificado contra Cantera con el MISMO mecanismo — ESTADO.md §11;
#: el jet-A no tiene termo NASA en las bases estándar y su V6 queda
#: declarado como pendiente).
#: `lhv_J_kg`: poder calorífico inferior (fuentes estándar: jet-A 43.2 MJ/kg,
#: propano 46.35 MJ/kg — Lefebvre & Ballal, Gas Turbine Combustion, ap. A).
MECHANISMS = {
    "jet_a": dict(
        name="2S_KERO_BFER", fuel="KERO", n_steps=2,
        formula="C12H23", mw_fuel=167.31e-3,
        c_atoms=12, h_atoms=23, lhv_J_kg=43.2e6,
        A=[2.0e15, 2.0e9], Ea=[1.255e8, 1.674e8],
        order_fuel=[0.55, 0.0], order_ox=[0.9, 0.5],
        note=("Franzelli/BFER 2 pasos para queroseno; se acopla el paso 1 "
              "efectivo. SIN V6 propio (no hay termo NASA de C12H23 en las "
              "bases estandar) — declarado."),
        source="Franzelli et al., Combust. Flame 157 (2010) — via WAVE6 §0"),
    "propane": dict(
        name="WD2", fuel="C3H8", n_steps=2,
        formula="C3H8", mw_fuel=44.097e-3,
        c_atoms=3, h_atoms=8, lhv_J_kg=46.35e6,
        A=[8.6e11, 5.0e8], Ea=[1.256e8, 1.674e8],
        order_fuel=[0.1, 0.0], order_ox=[1.65, 0.5],
        note=("Westbrook & Dryer 2 pasos; se acopla el paso 1 efectivo. "
              "V6 contra Cantera con el MISMO mecanismo: ESTADO.md §11."),
        source="Westbrook & Dryer, Combust. Sci. Technol. 27 (1981)"),
    "methane": dict(
        name="2S_CH4_BFER", fuel="CH4", n_steps=2,
        formula="CH4", mw_fuel=16.043e-3,
        c_atoms=1, h_atoms=4, lhv_J_kg=50.0e6,
        # Ea de la fuente en cal/mol (35500 y 12000) × 4184 → J/kmol.
        # El paso 2 (CO⇌CO2) lleva b = 0.7 (T^0.7) que arrhenius_rate v1
        # NO soporta — se acopla el paso 1 efectivo, como en el resto.
        A=[4.9e9, 2.0e8], Ea=[1.48532e8, 5.0208e7],
        order_fuel=[0.5, 0.0], order_ox=[0.65, 0.5],
        note=("Franzelli/BFER 2 pasos para metano; se acopla el paso 1 "
              "efectivo. V6-CH4 contra Cantera con el MISMO mecanismo: "
              "ESTADO.md §11. Es el mecanismo que necesita V8 (Sandia D)."),
        source=("Franzelli et al., Combust. Flame 159 (2012) 621-637; "
                "parametros verificados en chemistry.cerfacs.fr "
                "(BFER methane-air) el 2026-08-02")),
    "h2": dict(
        name="NO-APLICA", fuel="H2", n_steps=0,
        formula="H2", mw_fuel=2.016e-3,
        c_atoms=0, h_atoms=2, lhv_J_kg=119.96e6,
        A=[], Ea=[], order_fuel=[], order_ox=[],
        note=("H2 premezclado/micromix esta EXCLUIDO de L2a v1 (WAVE6 §0): "
              "necesita FGM con difusion preferencial y EDM falla. No se "
              "devuelve mecanismo a proposito."),
        source="WAVE6 §0 / Ciencia §11.2"),
}

#: Relación estequiométrica másica oxidante/combustible (kg O₂ por kg fuel).
_S_MASS = {"jet_a": 3.404, "propane": 3.628, "methane": 3.989,
           "h2": 7.937}


def global_mechanism(fuel: str) -> dict:
    """Mecanismo global del combustible. Devuelve SIEMPRE un dict (nunca
    lanza); para `h2` devuelve el bloque con `n_steps=0` y su motivo."""
    key = str(fuel or "jet_a").strip().lower()
    return dict(MECHANISMS.get(key, MECHANISMS["jet_a"]))


def stoichiometric_ratio(fuel: str) -> float:
    """s = kg de O₂ por kg de combustible en la estequiometría global."""
    return float(_S_MASS.get(str(fuel or "jet_a").strip().lower(), 3.404))


def product_split(mech: dict) -> dict:
    """Rendimientos másicos del paso efectivo CxHy + (x+y/4)O₂ →
    x CO₂ + (y/2) H₂O, en kg por kg de combustible. `s` sale de los átomos
    (coherente con la tabla `_S_MASS` a <1 %)."""
    x = float(mech.get("c_atoms", 12))
    y = float(mech.get("h_atoms", 23))
    mwf = float(mech.get("mw_fuel", 167.31e-3)) * 1.0e3     # kg/kmol
    s = (x + 0.25 * y) * MW["O2"] / mwf
    return dict(s=s,
                co2=x * MW["CO2"] / mwf,
                h2o=0.5 * y * MW["H2O"] / mwf,
                mw_fuel_kg_kmol=mwf)


# ==========================================================================
# 2. Tasas — funciones puras
# ==========================================================================
def arrhenius_rate(T, rho, y_fuel, y_ox, mech: dict, step: int = 0,
                   a_factor: float = 1.0, xp=np):
    """Tasa de Arrhenius global del paso `step`, en **kg de combustible por
    m³ y s**, MOLAR y en SI::

        [F] = ρ·Y_F/MW_F   [kmol/m³]      (ídem O₂)
        q   = A_cgs·1e3·(1e-3·[F])^a·(1e-3·[O₂])^b·exp(−Ea/(R_kmol·T))
        ω_F = MW_F·q

    donde el factor 1e-3 convierte kmol/m³ → mol/cm³ (la convención de la
    tabla) y el 1e3 devuelve la tasa a kmol/(m³·s). `a_factor` es la rampa
    del encendido (multiplica el pre-exponencial). Verificada contra
    Cantera con el MISMO mecanismo (V6). **Función pura.**
    """
    T = xp.maximum(xp.asarray(T, dtype=float), 200.0)
    rho = xp.maximum(xp.asarray(rho, dtype=float), 1e-12)
    yf = xp.clip(xp.asarray(y_fuel, dtype=float), 0.0, 1.0)
    yo = xp.clip(xp.asarray(y_ox, dtype=float), 0.0, 1.0)
    if not mech.get("A"):
        return xp.zeros_like(T)
    i = int(np.clip(step, 0, len(mech["A"]) - 1))
    a = float(mech["A"][i]) * float(a_factor)
    ea = float(mech["Ea"][i])
    na = float(mech["order_fuel"][i])
    nb = float(mech["order_ox"][i])
    mwf = float(mech.get("mw_fuel", 167.31e-3)) * 1.0e3     # kg/kmol
    cf = 1.0e-3 * rho * yf / mwf                            # mol/cm³
    co = 1.0e-3 * rho * yo / MW["O2"]
    q = a * xp.exp(-ea / (R_J_PER_KMOL_K * T)) \
        * xp.maximum(cf, 0.0) ** na * xp.maximum(co, 0.0) ** nb
    return mwf * 1.0e3 * q                                  # kg/m³/s


def edm_rate(rho, k, eps, y_fuel, y_ox, y_prod, s: float,
             a: float = EDM_A, b: float = EDM_B, xp=np):
    """Tasa de mezcla de **Eddy Dissipation Model** (Magnussen & Hjertager
    1976), en kg de combustible por m³ y s::

        ω_mix = A·ρ·(ε/k)·min( Y_F , Y_O/s , B·Y_P/(1+s) )

    El tercer término (productos) es el que hace que la llama necesite
    productos calientes para propagarse — y es la razón física de que un
    campo frío **no encienda solo**: ahí nace el riesgo #1.
    **Función pura, verificable sin solver.**
    """
    rho = xp.maximum(xp.asarray(rho, dtype=float), 0.0)
    k = xp.maximum(xp.asarray(k, dtype=float), 1e-12)
    eps = xp.maximum(xp.asarray(eps, dtype=float), 0.0)
    yf = xp.clip(xp.asarray(y_fuel, dtype=float), 0.0, 1.0)
    yo = xp.clip(xp.asarray(y_ox, dtype=float), 0.0, 1.0)
    yp = xp.clip(xp.asarray(y_prod, dtype=float), 0.0, 1.0)
    s = max(float(s), 1e-12)
    lim = xp.minimum(yf, yo / s)
    if b > 0.0:
        lim = xp.minimum(lim, b * yp / (1.0 + s))
    return a * rho * (eps / k) * lim


def edm_source(rho, k, eps, T, y_fuel, y_ox, y_prod, mech: dict,
               s: float | None = None, a: float = EDM_A, b: float = EDM_B,
               a_factor: float = 1.0, xp=np):
    """Cierre EDM completo: ω = **min**(tasa de mezcla, Arrhenius global)
    (Ciencia §11.2). Devuelve kg de combustible consumido por m³ y s.
    **Función pura, verificable sin solver.**"""
    if s is None:
        s = stoichiometric_ratio(mech.get("formula", "jet_a"))
    mix = edm_rate(rho, k, eps, y_fuel, y_ox, y_prod, s, a, b, xp=xp)
    arr = arrhenius_rate(T, rho, y_fuel, y_ox, mech, 0, a_factor, xp=xp)
    return xp.minimum(mix, arr)


def cantera_yaml(mech: dict, thermo_source: str = "gri30.yaml") -> str:
    """YAML de Cantera con **exactamente** el paso 1 efectivo de `mech`
    (mismos A, Ea y órdenes), especies con la termodinámica NASA de
    `thermo_source`. Es la pieza de la auto-validación V6: nuestro solver
    y Cantera FreeFlame reciben el MISMO mecanismo, así que cualquier
    error de conversión de unidades aparece como desacuerdo en S_L.

    Unidades en el YAML: A viene en cgs-mol y se declara `units: {length:
    cm, quantity: mol, activation-energy: J/kmol}` — la MISMA convención
    de la tabla, para que la conversión la haga Cantera y no este módulo
    (si este módulo convirtiera, V6 no probaría nada).
    """
    x = int(mech.get("c_atoms", 3))
    y = int(mech.get("h_atoms", 8))
    fuel = mech.get("formula", "C3H8")
    a = float(mech["A"][0])
    ea = float(mech["Ea"][0])
    na = float(mech["order_fuel"][0])
    nb = float(mech["order_ox"][0])
    no2 = x + 0.25 * y
    return f"""
description: paso 1 efectivo de {mech.get('name')} (tabla de Phy-CB)
units: {{length: cm, quantity: mol, activation-energy: J/kmol}}
phases:
- name: gas
  thermo: ideal-gas
  elements: [C, H, O, N]
  species:
  - {thermo_source}/species: [{fuel}, O2, CO2, H2O, N2]
  kinetics: gas
  transport: mixture-averaged
  state: {{T: 300.0, P: 1 atm}}
reactions:
- equation: {fuel} + {no2:g} O2 => {x} CO2 + {y / 2:g} H2O
  rate-constant: {{A: {a:g}, b: 0.0, Ea: {ea:g}}}
  orders: {{{fuel}: {na:g}, O2: {nb:g}}}
  negative-orders: true
  nonreactant-orders: true
"""


def pyrometheus_rates(mech: dict, xp=np):
    """Vía **Pyrometheus** (WAVE6 §0): genera el evaluador de tasas desde
    el MISMO yaml de Cantera que usa V6 (`cantera_yaml`). Devuelve
    `(termoquimica, "")` — instancia con `get_net_production_rates(rho, T,
    Y)` en kmol/m³/s sobre el módulo de arrays `xp` — o `(None, motivo)`
    si Pyrometheus/Cantera no están. **Nunca lanza.**

    Instalado el 2026-08-02 con autorización expresa (antes la regla del
    proyecto lo vetaba y esta función degradaba siempre). Verificado
    IDÉNTICO a la tabla y a Cantera en el mismo estado (test M3c): para
    los mecanismos de 1 paso efectivo de v1 la tabla y Pyrometheus son
    intercambiables; la vía Pyrometheus es la que escala a mecanismos
    skeletal (Jacobiano incluido) cuando lleguen.
    """
    try:
        import cantera as ct
        import pyrometheus as pyro
    except Exception as exc:
        return None, (f"pyrometheus/cantera no disponibles "
                      f"({type(exc).__name__}); se usan las tasas de la "
                      "tabla global (equivalentes para 1 paso, test M3c)")
    try:
        sol = ct.Solution(yaml=cantera_yaml(mech))
        cls = pyro.get_thermochem_class(sol)
        return cls(xp), ""
    except Exception as exc:
        return None, (f"generacion Pyrometheus fallo "
                      f"({type(exc).__name__}: {exc}); se usan las tasas "
                      "de la tabla global")


# ==========================================================================
# 3. Encendido — IMPLEMENTADO (riesgo #1: su robustez se juzga en M4)
# ==========================================================================
@dataclass
class IgnitionPlan:
    """Plan de arranque de llama de WAVE6 §3.

    * parche caliente esférico en la PZ impuesto `n_hold` iteraciones,
    * rampa del pre-exponencial de `ramp_from`× a 1× en `n_ramp`,
    * limitador T ∈ [T3 − 50, `T_max_factor`·T_ad] (se aplica fijando el
      techo de `EnergyLowMach` al enganchar),
    * sub-relajación del término fuente (`relax`),
    * criterio de llama anclada: liberación de calor integral estable
      ±`stable_tol` durante `n_stable` iteraciones (`anchored()`).
    """
    patch_center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    patch_radius_m: float = 5.0e-3
    patch_T_K: float = 2000.0
    n_hold: int = 2000
    ramp_from: float = 0.1
    n_ramp: int = 5000
    T_min_offset_K: float = -50.0
    T_max_factor: float = 1.05
    relax: float = 0.3
    n_stable: int = 500
    stable_tol: float = 0.02

    def a_factor(self, it: int) -> float:
        """Factor de rampa del pre-exponencial en la iteración `it` (la
        cuenta empieza al ACABAR el hold)."""
        if it < self.n_hold:
            return float(self.ramp_from)
        f = (it - self.n_hold) / max(float(self.n_ramp), 1.0)
        return float(self.ramp_from + (1.0 - self.ramp_from)
                     * min(max(f, 0.0), 1.0))


def ignition_patch(mesh, plan: IgnitionPlan):
    """Máscara booleana del parche de encendido (geometría pura)."""
    acc = None
    for d in range(mesh.ndim):
        t = (mesh.X[d] - plan.patch_center_m[d]) ** 2
        acc = t if acc is None else acc + t
    return np.broadcast_to(np.sqrt(acc) <= plan.patch_radius_m,
                           mesh.shape_tot).copy()


def equilibrium_T_of_Z(z, T_ox: float, T_fuel: float, T_ad: float,
                       z_st: float) -> np.ndarray:
    """**Salida de emergencia declarada** (*mixed-is-burned*): T = f(Z) por
    equilibrio, con las dos ramas lineales clásicas de Burke–Schumann.

    Pierde la cinética (η_comb, CO, LBO) pero conserva PF, penetración y
    puntos calientes. Tenerla escrita es lo que permite tomar la decisión
    de la semana 15 sin reescribir nada.
    """
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    z_st = float(np.clip(z_st, 1e-6, 1.0 - 1e-6))
    lean = T_ox + (T_ad - T_ox) * z / z_st
    rich = T_ad + (T_fuel - T_ad) * (z - z_st) / (1.0 - z_st)
    return np.where(z <= z_st, lean, rich)


# ==========================================================================
# 4. Transporte de especies — IMPLEMENTADO (M3)
# ==========================================================================
@dataclass
class ChemConfig:
    """Configuración del M3. `cp_J_kgK` es el c_p EFECTIVO constante del
    caso (declarado en el docstring): calibrarlo a la T_ad del combustible
    (`SpeciesTransport.calibrate_cp`) es lo que hace que la T de llama
    salga bien aunque el c_p real varíe."""
    fuel: str = "jet_a"
    model: str = "EDM"                  # EDC como upgrade (WAVE6 §0)
    species: tuple = ("FUEL", "O2", "CO2", "H2O", "N2")
    schmidt: float = 0.7
    schmidt_t: float = 0.7
    prandtl: float = 0.7
    cp_J_kgK: float = 1300.0            # ⚠️ TUNABLE — calibrar a T_ad
    ignition: IgnitionPlan = field(default_factory=IgnitionPlan)
    T_floor_K: float = 250.0


class SpeciesTransport:
    """Transporte de Y_k + cierre EDM + encendido, acoplado por
    `solver.physics`. Orden de registro OBLIGADO (y comprobado en
    `attach`): turbulencia (si la hay) → **especies** → energía; las
    especies calculan ω con la T del paso anterior y escriben S_T, y la
    energía lo integra en el mismo paso. Uso::

        tur = KEpsilonRealizable(mesh).attach(solver)          # opcional
        sp  = SpeciesTransport(mesh, ChemConfig(fuel="propane"))
        en  = EnergyLowMach(mesh, EnergyConfig(...))
        sp.attach(solver, turbulence=tur, energy=en)   # engancha AMBOS

    Sin turbulencia (V6/V7, llamas laminares) el cierre es la cinética
    pura; con ella, min(EDM, Arrhenius).
    """

    name = "species-edm"

    def __init__(self, mesh, cfg: ChemConfig | None = None):
        self.mesh = mesh
        self.cfg = cfg or ChemConfig()
        self.mech = global_mechanism(self.cfg.fuel)
        self.split = product_split(self.mech)
        self.Y: dict = {}
        self.turbulence = None
        self.energy = None
        self.solver = None
        self.status = "detached"
        self.it = 0
        self.hrr_history: list[float] = []
        self._omega_prev = None
        self._patch = None
        self._max_diff = 0.0
        #: fuente de combustible del M4 (inyección; ver `set_fuel_source`)
        self._fuel_src = None
        self.fuel_injected_kg = 0.0
        #: pared transpirante del M4: el aire que entra por la pared porosa
        #: entra FRESCO (composición de aire), no con la mezcla quemada
        #: local — sin esto la dilución no aporta O2 (ver `set_wall_air`)
        self._wall_air = None

    # -- utilidades --------------------------------------------------------
    def t_adiabatic(self, y_fuel_unburned: float,
                    T_u: float = 300.0) -> float:
        """T_ad efectiva del modelo: T_u + Y_F·LHV/c_p (consistente con la
        energía de c_p constante — es la T que el solver puede alcanzar)."""
        return float(T_u + y_fuel_unburned
                     * float(self.mech["lhv_J_kg"]) / self.cfg.cp_J_kgK)

    def calibrate_cp(self, t_ad_target: float, y_fuel_unburned: float,
                     T_u: float = 300.0) -> float:
        """Fija c_p para que la T_ad del modelo coincida con
        `t_ad_target` (p. ej. el equilibrio de Cantera). Devuelve el c_p."""
        dt = max(float(t_ad_target) - float(T_u), 1.0)
        self.cfg.cp_J_kgK = float(y_fuel_unburned) \
            * float(self.mech["lhv_J_kg"]) / dt
        return self.cfg.cp_J_kgK

    # -- inyección de combustible (M4, tarea 4.2) --------------------------
    def set_fuel_source(self, mask_interior, mdot_kgs: float,
                        note: str = "") -> "SpeciesTransport":
        """Fuente volumétrica de combustible GASEOSO en la región `mask`
        (booleana, tamaño del interior): el spray NO se resuelve y la
        evaporación es instantánea — DECLARADO (WAVE6 §4; el retardo τ_e de
        Rizk & Mongia queda como upgrade). Mismo patrón opt-in que
        `EnergyLowMach.source_K_s`.

        La fuente es una MEZCLA conservativa: dY_F/dt = q·(1 − Y_F) y
        dY_k/dt = −q·Y_k para el resto, con q elegido cada paso para que el
        caudal inyectado sea exactamente `mdot_kgs` — así ΣY_k = 1 se
        conserva EXACTAMENTE (la lección del test M3d). El aumento de masa
        de la mezcla (~2 % del aire) NO entra a la continuidad: declarado.
        """
        self._fuel_src = dict(mask=None, mask_host=np.asarray(
            mask_interior, dtype=bool), mdot=float(mdot_kgs),
            note=note or "fuente de combustible v1 (evaporacion "
                         "instantanea declarada)")
        self.fuel_injected_kg = 0.0
        return self

    def _apply_fuel_source(self, solver, dt, rho) -> None:
        src = self._fuel_src
        if src is None or src["mdot"] <= 0.0:
            return
        xp, m = solver.xp, solver.mesh
        dt_ = solver.dtype
        I = m.interior
        if src["mask"] is None:                    # subir UNA vez al backend
            src["mask"] = xp.asarray(src["mask_host"])
        msk = src["mask"]
        dv = m.dx ** m.ndim
        yf = self.Y["FUEL"][I]
        # q uniforme sobre la región tal que ∫ρ·q·(1−Y_F) dV = ṁ_fuel
        denom = xp.sum(xp.where(msk, rho * (1.0 - yf), dt_(0.0))) * dt_(dv)
        q = dt_(src["mdot"]) / xp.maximum(denom, dt_(1e-30))
        fac = dt_(dt) * q
        for sp in self.cfg.species:
            tgt = dt_(1.0) if sp == "FUEL" else dt_(0.0)
            y = self.Y[sp][I]
            self.Y[sp][I] = xp.where(msk, y + fac * (tgt - y), y)
        self.fuel_injected_kg += src["mdot"] * float(dt)

    # -- enganche ----------------------------------------------------------
    def attach(self, solver, turbulence=None, energy=None,
               y_init: dict | None = None,
               ignition: bool = False) -> "SpeciesTransport":
        """Crea los campos en el backend, deriva contornos (el spec de una
        cara `inlet` puede traer `Y_FUEL=`, `Y_O2=`, … escalares o
        perfiles), registra este modelo Y la energía (en ese orden) y
        prepara el encendido si `ignition=True`."""
        from .solver import scalar_bc_spec
        xp, m, cfg = solver.xp, solver.mesh, self.cfg
        dt_ = solver.dtype
        self.solver = solver
        self.turbulence = turbulence
        self.energy = energy
        air = {"FUEL": 0.0, "O2": 0.233, "CO2": 0.0, "H2O": 0.0,
               "N2": 0.767}
        y0 = dict(air, **(y_init or {}))
        self.Y = {}
        self.bc = {}
        for sp in cfg.species:
            self.Y[sp] = xp.full(m.shape_tot, dt_(y0.get(sp, 0.0)),
                                 dtype=dt_)
            self.bc[sp] = scalar_bc_spec(solver, f"Y_{sp}",
                                         y0.get(sp, 0.0))
        self._omega_prev = xp.zeros(m.shape, dtype=dt_)
        self.hrr_history = []
        self.it = 0
        if energy is not None and energy.source_K_s is None:
            energy.source_K_s = xp.zeros(m.shape_tot, dtype=dt_)
        # limitador de T del plan de encendido → el techo lo aplica la
        # energía en su clip (T ∈ [T3−50, T_max_factor·T_ad])
        if energy is not None:
            yf_ref = max(float(y0.get("FUEL", 0.0)), 0.02)
            t_u = float(energy.cfg.T_inlet_K)
            t_ad = self.t_adiabatic(yf_ref, t_u)
            energy.cfg.T_ceiling_K = float(
                cfg.ignition.T_max_factor * t_ad)
            energy.cfg.T_floor_K = max(
                float(cfg.T_floor_K),
                t_u + float(cfg.ignition.T_min_offset_K))
        self._patch = None
        if ignition:
            self._patch = xp.asarray(
                ignition_patch(m, cfg.ignition)[m.interior])
        # ORDEN: especies antes que la energía (ω con T^n → S_T del paso).
        # La energía se registra aquí para que el orden quede garantizado
        # por construcción y no por disciplina del llamante.
        solver.physics.append(self)
        if energy is not None and energy.status == "detached":
            energy.attach(solver, turbulence=turbulence)
        nu = float(solver.cfg.nu)
        self._max_diff = nu / cfg.schmidt
        self.status = "attached"
        return self

    def max_diffusivity(self) -> float:
        return self._max_diff

    def set_wall_air(self, mask_per_ghost) -> "SpeciesTransport":
        """Impone composición de AIRE en los fantasmas de la pared
        transpirante del M4 (máscara por fantasma; el resto sigue
        Neumann). La máscara se sube al backend UNA vez aquí."""
        xp = self.solver.xp
        self._wall_air = xp.asarray(np.asarray(mask_per_ghost, bool))
        return self

    _AIR = {"FUEL": 0.0, "O2": 0.233, "CO2": 0.0, "H2O": 0.0, "N2": 0.767}

    def _fill_bcs(self, solver) -> None:
        from .solver import fill_scalar_bc
        m = solver.mesh
        for sp, fld in self.Y.items():
            fill_scalar_bc(solver, fld, self.bc[sp])
            if m.phi is not None:
                m.apply_neumann(fld)
                if self._wall_air is not None:     # pared transpirante
                    m.apply_dirichlet(fld, self._AIR.get(sp, 0.0),
                                      only=self._wall_air)
                m.zero_deep_solid(fld, 0.0)

    # -- un paso -----------------------------------------------------------
    def advance(self, solver, dt: float) -> bool:
        from .solver import advect_diffuse_scalar
        xp, m, cfg = solver.xp, solver.mesh, self.cfg
        dt_ = solver.dtype
        I = m.interior
        plan = cfg.ignition
        fm = solver._fmask
        en = self.energy
        # estado termodinámico del paso anterior
        if en is not None:
            T = en.T[I]
            rho = en.rho[I]
        else:
            T = xp.full(m.shape, dt_(300.0), dtype=dt_)
            rho = xp.full(m.shape, dt_(solver.cfg.rho), dtype=dt_)
        yf = self.Y["FUEL"][I]
        yo = self.Y["O2"][I]
        yp = (self.Y["CO2"][I] + self.Y["H2O"][I])
        # 1. encendido: parche caliente impuesto durante el hold
        if self._patch is not None and self.it < plan.n_hold \
                and en is not None:
            en.T[I] = xp.where(self._patch, dt_(plan.patch_T_K), en.T[I])
            T = en.T[I]
        a_f = plan.a_factor(self.it) if self._patch is not None else 1.0
        # 2. fuente: min(EDM, Arrhenius) con turbulencia; cinética pura sin
        arr = arrhenius_rate(T, rho, yf, yo, self.mech, 0, a_f, xp=xp)
        if self.turbulence is not None:
            mix = edm_rate(rho, self.turbulence.k[I],
                           self.turbulence.eps[I], yf, yo, yp,
                           self.split["s"], xp=xp)
            omega = xp.minimum(mix, arr)
            # arranque: el término de productos del EDM (B·Y_P) es un veto
            # circular al encendido — sin productos la tasa de mezcla es 0
            # en TODO el dominio y el parche caliente no puede prender nada
            # (medido en el combustor M4: no anclaba jamás). Durante el
            # hold + rampa, DENTRO del parche, el EDM va sin límite de
            # productos; fuera, y después, el cierre es el estándar.
            if self._patch is not None \
                    and self.it < plan.n_hold + plan.n_ramp:
                mix0 = edm_rate(rho, self.turbulence.k[I],
                                self.turbulence.eps[I], yf, yo, yp,
                                self.split["s"], b=0.0, xp=xp)
                omega = xp.where(self._patch,
                                 xp.minimum(mix0, arr), omega)
        else:
            omega = arr
        # no consumir más de lo que hay en la celda en un paso
        cap = rho * xp.minimum(yf, yo / dt_(self.split["s"])) / dt_(dt)
        omega = xp.minimum(omega, xp.maximum(cap, 0.0))
        # sub-relajación del plan de encendido (estabiliza el transitorio)
        omega = dt_(plan.relax) * omega \
            + dt_(1.0 - plan.relax) * self._omega_prev
        omega = omega * fm
        # 3. transporte + fuente. El sumidero del FUEL es point-implicit;
        #    TODO lo demás (O₂, productos, calor) se acopla al consumo
        #    REALIZADO por ese paso implícito, no a la ω demandada — si no,
        #    al agotarse el fuel se seguirían creando productos y calor de
        #    la nada (lo cazó el test M3d: Σ Y llegaba a 1.47).
        uf = solver.face_velocities(dt)
        if self.turbulence is not None:
            gamma = dt_(solver.cfg.nu) / dt_(cfg.schmidt) \
                + self.turbulence.nu_t_field / dt_(cfg.schmidt_t)
        elif en is not None:
            # laminar: D = ν(T)/Sc — con Sc = Pr esto es Le = 1 exacto,
            # coherente con el `unity-Lewis-number` que V6 pide a Cantera
            gamma = en.laminar_nu(solver) / dt_(cfg.schmidt)
        else:
            gamma = dt_(solver.cfg.nu) / dt_(cfg.schmidt)
        w_rho = omega / xp.maximum(rho, dt_(1e-6))          # 1/s · Y-escala
        cf = w_rho / xp.maximum(yf, dt_(1e-10))             # coef implicito
        rhs_k = {}
        for sp in cfg.species:
            rhs, _ = advect_diffuse_scalar(solver, self.Y[sp], gamma, uf)
            rhs_k[sp] = rhs * fm
        yf_hat = yf + dt_(dt) * rhs_k["FUEL"]               # tras transporte
        yf_new = yf_hat / (1.0 + dt_(dt) * cf)
        # consumo realizado, acotado además por el O₂ disponible
        dyf = xp.maximum(yf_hat - yf_new, 0.0)
        yo_hat = xp.maximum(yo + dt_(dt) * rhs_k["O2"], 0.0)
        dyf = xp.minimum(dyf, yo_hat / dt_(self.split["s"]))
        omega_eff = rho * dyf / dt_(dt)
        self._omega_prev = omega_eff
        self.Y["FUEL"][I] = xp.clip(yf_hat - dyf, 0.0, 1.0)
        self.Y["O2"][I] = xp.clip(yo_hat - dt_(self.split["s"]) * dyf,
                                  0.0, 1.0)
        self.Y["CO2"][I] = xp.clip(
            self.Y["CO2"][I] + dt_(dt) * rhs_k["CO2"]
            + dt_(self.split["co2"]) * dyf, 0.0, 1.0)
        self.Y["H2O"][I] = xp.clip(
            self.Y["H2O"][I] + dt_(dt) * rhs_k["H2O"]
            + dt_(self.split["h2o"]) * dyf, 0.0, 1.0)
        self.Y["N2"][I] = xp.clip(
            self.Y["N2"][I] + dt_(dt) * rhs_k["N2"], 0.0, 1.0)
        # 3b. inyección de combustible (M4): mezcla conservativa opt-in
        self._apply_fuel_source(solver, dt, rho)
        # 4. liberación de calor → energía (S_T en K/s), del consumo REAL
        if en is not None:
            en.source_K_s[I] = (dyf / dt_(dt)) \
                * dt_(float(self.mech["lhv_J_kg"]) / self.cfg.cp_J_kgK)
        hrr = float(xp.sum(omega_eff)) * float(self.mech["lhv_J_kg"]) \
            * m.dx ** m.ndim
        if solver.be.is_gpu:
            solver._sync_calls += 1
        self.hrr_history.append(hrr)
        self._fill_bcs(solver)
        self._max_diff = float(solver.cfg.nu) / cfg.schmidt \
            + (max(self.turbulence.max_diffusivity()
                   - float(solver.cfg.nu), 0.0) / cfg.schmidt_t
               if self.turbulence is not None else 0.0)
        self.it += 1
        return True

    # -- diagnóstico -------------------------------------------------------
    def mixture_fraction(self) -> np.ndarray:
        """Fracción de mezcla Z del campo, en NumPy float64 (post-proceso).

        Con las 5 especies del esquema y N₂ INERTE (aire Y_N2 = 0.767,
        combustible sin N₂), Z = 1 − Y_N2/0.767 es EXACTA — no hace falta
        Bilger. La usan el mixed-is-burned y el clustering del NOx (M4)."""
        from .backend import to_numpy
        yn2 = np.asarray(to_numpy(self.Y["N2"]), dtype=np.float64)
        return np.clip(1.0 - yn2 / 0.767, 0.0, 1.0)

    def z_stoichiometric(self) -> float:
        """Z estequiométrica del combustible (corriente de fuel pura):
        z_st = 1/(1 + s/0.233)."""
        s = float(self.split["s"])
        return 1.0 / (1.0 + s / 0.233)

    def heat_release(self) -> float:
        """Última liberación de calor integral (W). NaN si no ha corrido."""
        return self.hrr_history[-1] if self.hrr_history else float("nan")

    def anchored(self) -> bool:
        """Criterio de llama anclada del plan: liberación integral estable
        ±`stable_tol` durante las últimas `n_stable` iteraciones."""
        plan = self.cfg.ignition
        n = int(plan.n_stable)
        if len(self.hrr_history) < max(n, 2):
            return False
        w = np.asarray(self.hrr_history[-n:], dtype=float)
        mean = float(np.mean(w))
        if not np.isfinite(mean) or mean <= 0.0:
            return False
        return bool(float(np.max(np.abs(w - mean))) / mean
                    <= float(plan.stable_tol))

    def consumption_speed(self, axis: int = 0,
                          y_fuel_unburned: float | None = None,
                          rho_unburned: float | None = None) -> float:
        """Velocidad de consumo S_c = ∫ω dV / (ρ_u·Y_F,u·A) — la medida de
        S_L del V6 (robusta: no necesita localizar el frente)."""
        m = self.mesh
        om = self._omega_prev
        if om is None:
            return float("nan")
        from .backend import to_numpy
        w = np.asarray(to_numpy(om), dtype=float)
        yfu = float(y_fuel_unburned if y_fuel_unburned is not None else
                    np.max(np.asarray(to_numpy(self.Y["FUEL"]))))
        if self.energy is not None and rho_unburned is None:
            rho_u = float(np.max(np.asarray(to_numpy(self.energy.rho))))
        else:
            rho_u = float(rho_unburned or 1.2)
        area_cells = w.size / w.shape[axis]
        return float(w.sum() * m.dx / (rho_u * max(yfu, 1e-12)
                                       * area_cells))

    def summary(self) -> dict:
        return dict(model=self.cfg.model, fuel=self.cfg.fuel,
                    mechanism=self.mech.get("name"), status=self.status,
                    available=available(), detail=STATUS,
                    cp_J_kgK=self.cfg.cp_J_kgK,
                    coupled_step="paso 1 efectivo (CO declarado fuera)",
                    ignition=("parche+rampa+limitador+subrelajacion "
                              "IMPLEMENTADOS; robustez en combustor = "
                              "riesgo #1, se juzga en M4"),
                    heat_release_W=self.heat_release(),
                    anchored=self.anchored())
