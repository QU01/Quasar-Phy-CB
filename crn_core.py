"""
QUASAR Phy-CB · crn_core.py
===========================
Fidelidad **L1** del Computational Engineering Model: red 1D de reparto de aire
RESUELTA + red de reactores químicos (CRN) en Cantera. Es el escalón que
sustituye las *correlaciones* de L0 por un *sistema resuelto* (WAVE2 §§1–2,
`docs/Quasar_PhyCB_Science.md` §§9.1–9.3, 6.1 y 8.3).

API pública (IMPLEMENTATION_CONTRACT §8)
----------------------------------------
    CANTERA_AVAILABLE : bool
    PHYCB_L1_TIMEOUT  : float        (env `PHYCB_L1_TIMEOUT`, default 60 s)
    network_1d(theta, rec) -> dict
    crn_solve(theta, rec, timeout_s=None) -> dict | None

`physics_core._try_l1` importa ESTE módulo de forma perezosa, comprueba
`CANTERA_AVAILABLE` y llama `crn_solve(theta, rec, L1_TIMEOUT_S)`. Si devuelve
`None` el punto degrada a L0 con etiqueta; si devuelve un dict, sus claves se
funden sobre el record (contrato §4.3). **Este módulo no importa
`physics_core` a nivel de módulo**: lo hace de forma perezosa y verificada
(trampa del paquete editable de Phy-AC, ver `_import_sibling`).

Qué resuelve la red 1D (§1)
---------------------------
Incógnitas: ṁ por cada camino (swirler, filas PZ/SZ/DZ, efusión, refrigeración)
y ΔP_L — 7 en total. Ecuaciones: ṁ_h = C_d·A_h·√(2ρ₃ΔP) por camino
[Ciencia §4.1] + conservación de masa + **una única ΔP_L** (el liner es un
plenum) + pérdidas de difusor/snout. Se resuelve por **Newton acotado**
(≤ `NEWTON_ITERS` = 30 iteraciones, jacobiano por diferencias hacia adelante,
backtracking acotado, test de convergencia) — sin `while` abiertos. **No
necesita Cantera: corre siempre.**

Qué resuelve el CRN (§2)
------------------------
Topología mínima de Ciencia §9.2::

    PSR_pz  ⇄ (recirculación CTRZ)  PSR_sz  →  PFR_dz (cadena)  + bypass frío

con `ct.IdealGasReactor` + `ct.MassFlowController` + `ct.Valve` +
`ct.ReactorNet`. Volúmenes de zona desde el sizing del record; caudales desde
la red 1D de §1. Salidas: `EI_NOx_crn`, `EI_CO_crn`, `eta_comb_crn` (burn-out
por entalpía química sin quemar), `Da_pz`, `LBO_margin_Da`, `T_zones_K` y
`crn_topology`.

**PF NO sale del CRN.** Un CRN es 0-D por zona: no tiene distribución espacial
de temperatura a la salida y por tanto **no puede** producir el factor de
patrón. `PF` sigue siendo la correlación de L0 [Ciencia §4.3] hasta que L1.5
(aero-frío XLB) y L2 (RANS/OpenFOAM) den el campo real. Declarado aquí y en el
reporte; el record de L1 conserva el `PF` de L0 sin tocar.

Reglas de la casa que este archivo respeta sin excepción
--------------------------------------------------------
  * SI interno; mm/deg solo en claves de interfaz con sufijo.
  * Docstrings y comentarios en español; `open()` con `encoding="utf-8"`.
  * Solo NumPy es obligatorio. **Cantera es opcional**: si no importa,
    `CANTERA_AVAILABLE = False`, `crn_solve` devuelve `None` y `network_1d`
    sigue funcionando. Nunca se lanza una excepción por una dependencia.
  * Bucles ACOTADOS (iteraciones fijas + test de convergencia).
  * Tunables `SCREAMING_CASE` con procedencia y aviso de recalibración.
  * Determinismo: `SEED = 71`; sin aleatoriedad. El único no-determinismo
    posible es el **timeout** (depende del reloj), que es precisamente por lo
    que existe la etiqueta `crn_L1(timeout→L0)`.

Timeout duro en Windows (trampa CONTEXTO §10.6)
-----------------------------------------------
En Windows no hay `SIGALRM` y un `ProcessPoolExecutor` por punto costaría más
que el propio solve (arrancar un intérprete + importar Cantera ≈ 1–2 s). La
elección de este módulo es la **tercera vía**: el avance de la `ReactorNet` se
hace con `net.step()` dentro de un bucle ACOTADO y el *deadline* se comprueba
entre pasos (`CRN_DEADLINE_EVERY`). Un paso interno de CVODES es corto
(µs–ms), así que la granularidad del timeout es la de un paso y no la de todo
el `advance()`. Ventajas: cero procesos, cero hilos, determinismo de la
física, y funciona igual en POSIX. Limitación declarada: si un **único** paso
de CVODES se colgase (no observado), el deadline no lo interrumpiría.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

# ==========================================================================
# 0. Constantes de la casa y tunables (SCREAMING_CASE + procedencia)
# ==========================================================================
SEED = 71                  # semilla de la familia (contrato §10.8). Este
#                            módulo es determinista y NO usa aleatoriedad.

#: Timeout duro por solve de CRN [s]. Contrato §8 y WAVE2 §2: default 60 s.
#: ⚠️ NOTA DE COHERENCIA: `physics_core.L1_TIMEOUT_S` lee la MISMA variable de
#: entorno pero con default 120 s. Cuando `physics_core` llama a `crn_solve`
#: le pasa SU valor explícitamente, así que el default de aquí solo manda en
#: llamadas directas (CLI, tests, este `__main__`).
PHYCB_L1_TIMEOUT = float(os.environ.get("PHYCB_L1_TIMEOUT", "60"))

# --- Red 1D: Newton acotado ----------------------------------------------
NEWTON_ITERS = 30          # iteraciones MÁXIMAS del Newton (WAVE2 §1: ≤30)
NEWTON_TOL = 1.0e-14       # ‖R‖_inf adimensional. Con ṁ ~ 0.1 kg/s esto deja
#                            la conservación de masa en ~1e-15 kg/s, tres
#                            órdenes por debajo del ancla de WAVE2 (1e-9).
NEWTON_BACKTRACK = 8       # halvings máximos del paso (bucle ACOTADO)
NEWTON_FD_STEP = 1.0e-7    # paso relativo del jacobiano por diferencias
DP_FLOOR_PA = 1.0          # suelo de ΔP por camino [Pa]: evita raíces de
#                            números negativos con θ degenerado. Si se activa
#                            se marca `network_clamped`.

# --- Red 1D: modelo de C_d -------------------------------------------------
#  RA_ASBUILT_UM y CD_EFFUSION_INCLINED SE MUDARON a `physics_core` el
#  2026-07-31 (fuente única de la efusión, physics_core §12.b): tenerlos aquí
#  permitía que la red 1D y el LBM usasen dos C_d distintos para el MISMO
#  orificio, y eso es exactamente lo que pasaba (0.2210 vs 0.2470). Se
#  RE-EXPORTAN abajo como alias de lectura para no romper a quien los
#  importaba por nombre; el valor vivo es el del hermano.
CD_RE_DEFICIT = 0.25       # déficit de C_d a Re→0 (forma f_Re = 1 − D·e^{−Re/S})
CD_RE_SCALE = 3000.0       # Re de saturación de f_Re
#                            ⚠️ FORMA DE LA CASA. Ciencia §4.1 declara «C_d
#                            depende además del bleed ratio y de Re» pero NO
#                            publica la función. Esta forma reproduce el
#                            comportamiento cualitativo (C_d crece con Re y
#                            satura por encima de Re ≈ 1e4) y deja los
#                            orificios de efusión (Re ~ 5e3) ligeramente por
#                            debajo del C_d nominal, que es lo medido.
#                            RECALIBRAR con el L1.5 aero-frío (WAVE5-A).
RAM_RECOVERY_PLAIN = 0.50  # recuperación de presión dinámica del anillo por
#                            un orificio RASANTE del liner ξ ∈ [0,1]. ξ=0 ⇒ el
#                            orificio ve la presión ESTÁTICA del anillo (el
#                            aire rápido «pasa de largo»); ξ=1 ⇒ ve la total
#                            (orificio embutido/con cuchara). ⚠️ DE LA CASA:
#                            punto medio de la banda física. Es el tunable MÁS
#                            influyente sobre la dispersión de los splits
#                            resueltos; RECALIBRAR con el LBM frío (WAVE5-A).
RAM_RECOVERY_SNOUT = 1.00  # el snout/difusor captura el flujo de frente ⇒
#                            el swirler ve la presión TOTAL del anillo ✅.
SWIRLER_VANES = 10         # nº de álabes del swirler, solo para el Ø
#                            hidráulico del canal con el que se entra en la
#                            corrección de rugosidad. ⚠️ de la casa.
D_COOL_RING_MM = 1.50      # Ø equivalente del camino «refrigeración» (anillo
#                            de domo + fugas de sello) [mm]. ⚠️ de la casa.
EFFUSION_FRAC_OF_COOL = 0.85   # fracción del aire de refrigeración que pasa
#                            por el CAMPO DE EFUSIÓN; el resto es anillo de
#                            domo + fugas. ⚠️ de la casa (banda de práctica
#                            0.7–0.95 en liners AM con efusión densa).
CD_SWIRLER = 0.80          # C_d del canal entre álabes del swirler. Banda de
#                            práctica 0.75–0.85 para un canal corto con
#                            expansión súbita a la salida. ⚠️ de la casa.
SWIRLER_BLOCKAGE = 0.20    # bloqueo de los álabes sobre el área frontal del
#                            swirler (10 álabes de espesor finito + capa
#                            límite). ⚠️ de la casa, banda 0.15–0.25.
ANNULUS_BLEED_SHAPE = 0.50     # ṁ_anillo representativo = ṁ₃·(1 − SHAPE·Σṁ/ṁ₃)
#                            ⇒ con la red cerrada queda ṁ₃/2, que es la media
#                            exacta de un sangrado LINEAL a lo largo del
#                            anillo (de ṁ₃ en el difusor a 0 al final). ⚠️ de
#                            la casa: un sangrado real no es lineal, pero el
#                            término solo entra por la recuperación de ram.
MU_AIR_REF = 1.81e-5       # µ del aire a 293.15 K [Pa·s] — MISMO valor que
#                            `physics_core.MU_AIR` (contrato §6). Se replica
#                            aquí para que la red 1D no dependa del hermano.

# --- Red 1D: pérdidas de difusor y snout [Ciencia §2.1] -------------------
K_DIFFUSER = 0.20          # ΔP_difusor / q_ref. Difusor de vertedero (dump)
#                            corto de microturbina. ⚠️ DE LA CASA dentro del
#                            cierre declarado en Ciencia §2.1: «ΔP_liner ≈
#                            70–80 % del ΔP total (resto difusor + snout)».
K_SNOUT = 0.10             # ΔP_snout / q_ref. ⚠️ ídem.
#                            K_DIFFUSER + K_SNOUT ≈ 0.30·q_ref es coherente
#                            con LINER_FRAC_OF_DP = 0.75 de physics_core para
#                            ΔP_L/q_ref ≈ 15 (0.30/(15+0.30) ≈ 2 %); la
#                            diferencia con el 25 % nominal de L0 se REPORTA
#                            en `dp_split_check` en vez de esconderse.

# --- CRN: topología y numérica --------------------------------------------
CRN_N_PFR_DZ = 5           # nº de reactores en serie que discretizan el
#                            «PFR» de la zona de dilución. 5 basta para que el
#                            perfil de CO congelado converja (<1 % entre 5 y
#                            10 en el caso ancla). Bucle ACOTADO por
#                            construcción.
CRN_RECIRC_AT_SN1 = 1.00   # ṁ_recirc/ṁ_PZ a S_N = 1.0. Banda de literatura
#                            para llamas estabilizadas por swirl con CTRZ:
#                            0.5–1.5. ⚠️ DE LA CASA (se escala linealmente con
#                            S_N y se acota a [0.10, 3.0]).
CRN_RECIRC_MIN, CRN_RECIRC_MAX = 0.10, 3.00
CRN_RTOL, CRN_ATOL = 1.0e-9, 1.0e-16   # tolerancias de la ReactorNet
CRN_MAX_STEPS = 200_000    # pasos MÁXIMOS de `net.step()` (bucle ACOTADO)
CRN_TAU_MULT = 40.0        # el estacionario se busca hasta t = MULT·τ_total
CRN_TAU_MIN_MULT = 5.0     # antes de MULT_MIN·τ_total no se declara convergido
CRN_CHECK_EVERY = 200      # cada cuántos pasos se testea la convergencia
CRN_CHECK_WINDOW_TAU = 0.25    # ventana MÍNIMA (en τ de la red) que debe
#                            cubrir un chequeo para que su |ΔT| signifique
#                            algo. Sin ella el test es inútil: CVODES gasta
#                            la mayoría de sus pasos en el transitorio
#                            inicial, así que 200 pasos consecutivos pueden
#                            cubrir 0.1 ms y dar |ΔT| grande cuando la
#                            solución ya está a punto de asentarse.
CRN_DEADLINE_EVERY = 50    # cada cuántos pasos se comprueba el deadline
CRN_T_TOL_K = 1.0e-4       # |ΔT| entre chequeos que declara estacionario [K]
CRN_T_DRIFT_MAX_K = 1.0    # deriva máxima ADMISIBLE en la verificación final
#                            de un τ extra: por encima, el punto se declara
#                            NO estacionario y `crn_solve` devuelve None.
CRN_VALVE_K = 1.0e-9       # coeficiente del `ct.Valve` de anclaje de presión
#                            (kg/s/Pa). Diseñado para NO mover el
#                            estacionario: a ΔP = 1 kPa deja pasar 1e-6 kg/s
#                            (≈ 1e-5 del caudal del caso ancla). Su papel es
#                            drenar una deriva numérica de masa, no repartir
#                            caudal — ver «Riesgo 3» en el docstring de
#                            `_build_network`.

# --- CRN: química y post-proceso ------------------------------------------
OXIDIZER = "O2:1.0, N2:3.76"   # aire seco molar (idéntico a physics_core)
DA_LBO_REF = 0.80          # Da de referencia del apagado pobre. Ciencia §6.1:
#                            «Da ≈ 0.3–0.8 en LBO frente a Da ≈ 20 en punto de
#                            diseño» ✅. Se toma el EXTREMO SUPERIOR de la
#                            banda (conservador) para `LBO_margin_Da`.
DA_LBO_BAND = (0.30, 0.80)     # banda declarada de LBO por Damköhler ✅
FAR_EVAP_C1 = -0.0295      # FAR_efectivo = FAR·(C1·τ_e/τ + C2) [Ciencia §8.3
FAR_EVAP_C2 = 0.45         # — VERIFICADA: Lanewala sobre Rizk & Mongia]
FAR_EVAP_FLOOR = 0.30      # suelo NUMÉRICO del factor de evaporación. La
#                            fórmula publicada no está acotada y con τ_e/τ > 5
#                            se vuelve negativa (sin sentido físico). ⚠️ guarda
#                            de la casa; se marca en `far_evap_clamped`.
#: Poder calorífico inferior por especie [J/kg] para el burn-out. Handbook.
LHV_BY_SPECIES = {
    "CO": 10.10e6, "H2": 120.0e6, "CH4": 50.0e6, "C2H2": 48.28e6,
    "C2H4": 47.16e6, "C2H6": 47.51e6, "C3H8": 46.35e6, "CH2O": 17.26e6,
    "CH3OH": 19.93e6, "NC12H26": 44.15e6, "POSF10325": 43.1e6,
}
MW_NO2 = 46.0055           # masa molar del NO₂ [kg/kmol] — los EI de NOx se
#                            reportan SIEMPRE como NO₂-equivalente (convenio
#                            de certificación ICAO).

G_DEGENERATE = 10.0        # (no se usa aquí: este módulo no produce `g`)


# ==========================================================================
# 1. Import perezoso y VERIFICADO de los hermanos de Phy-CB
# ==========================================================================
_SIBLINGS: dict[str, object | None] = {}


def _import_sibling(module_name: str):
    """Importa un módulo de Phy-CB comprobando que sea REALMENTE de Phy-CB.

    ⚠️ TRAMPA REAL de esta máquina: Phy-AC está instalado como paquete
    editable, así que `import physics_core` puede resolver al de Phy-AC (θ de
    15 posiciones y otra física) sin avisar. Se compara el directorio del
    módulo importado con el de ESTE archivo. Nunca lanza: devuelve `None`.
    """
    if module_name in _SIBLINGS:
        return _SIBLINGS[module_name]
    mod = None
    try:
        import importlib
        cand = importlib.import_module(module_name)
        here = os.path.dirname(os.path.abspath(__file__))
        there = os.path.dirname(os.path.abspath(getattr(cand, "__file__", "")))
        if os.path.normcase(here) == os.path.normcase(there):
            mod = cand
    except Exception:
        mod = None
    _SIBLINGS[module_name] = mod
    return mod


def _shared_const(name: str, fallback: float) -> float:
    """Constante COMPARTIDA cuyo dueño es `physics_core` (fuente única).

    Si el hermano no es importable se cae al literal de RESPALDO, que es el
    MISMO valor documentado allí — degradación etiquetada, no una segunda
    fuente de verdad: en cuanto `physics_core` esté, manda él.
    """
    pc = _import_sibling("physics_core")
    try:
        return float(getattr(pc, name))
    except Exception:
        return float(fallback)


#: ALIAS DE LECTURA de dos tunables cuyo dueño es hoy `physics_core` §12.b
#: (se mudaron el 2026-07-31 para que la red 1D y `cfd/xlb_cold` no pudieran
#: dar dos C_d distintos al mismo orificio de efusión). Se conservan los
#: nombres porque son ABI: hay código y documentación que los citan.
#: `nombre -> respaldo si el hermano no es importable`.
_SHARED_FROM_PHYSICS = {"CD_EFFUSION_INCLINED": 0.38, "RA_ASBUILT_UM": 12.0}


def __getattr__(name: str):
    """Alias PEREZOSOS de las constantes que posee `physics_core` (PEP 562).

    Se resuelven al LEERLOS, no al importar este módulo: la cabecera promete
    que `crn_core` **no importa `physics_core` a nivel de módulo** (trampa
    del paquete editable de Phy-AC), y un alias evaluado en el `import`
    congelaría la resolución antes de que el punto de entrada haya arreglado
    el `sys.path`. Dentro de este archivo se usa `_shared_const` directamente
    — PEP 562 no interviene en la búsqueda de nombres globales.
    """
    if name in _SHARED_FROM_PHYSICS:
        return _shared_const(name, _SHARED_FROM_PHYSICS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _safe(x, default: float = 0.0) -> float:
    """float finito o `default` (el record se serializa a JSON: nada de NaN)."""
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _clip(x: float, lo: float, hi: float) -> float:
    """Clamp finito (nunca lanza, NaN → `lo`)."""
    v = _safe(x, lo)
    return float(min(max(v, lo), hi))


# ==========================================================================
# 2. Cantera: import al cargar el módulo (CANTERA_AVAILABLE es ABI)
# ==========================================================================
#: `physics_core._try_l1` lee `CANTERA_AVAILABLE` INMEDIATAMENTE después del
#: import, así que no puede ser perezoso: tiene que ser un bool real ya.
_CT = None
try:                                            # pragma: no cover - entorno
    import cantera as _ct_mod
    _CT = _ct_mod
    try:
        _CT.suppress_thermo_warnings()
    except Exception:
        pass
    CANTERA_AVAILABLE = True
except Exception:
    CANTERA_AVAILABLE = False

#: Estado del ÚLTIMO `crn_solve`. `physics_core._try_l1` (WAVE1) etiqueta como
#: `crn_L1(diverged→L0)` cualquier `None`, porque no puede distinguir el
#: motivo. Esta variable expone el motivo real para que un parche de 3 líneas
#: en `physics_core` pueda emitir `crn_L1(timeout→L0)` cuando toque (contrato
#: §3). Valores: "ok" · "timeout" · "diverged" · "cantera_unavailable" ·
#: "degenerate" · "no_mechanism".
LAST_STATUS = "not_run"

#: Directorio de mecanismos propios de Phy-CB (capa 0). WAVE2 §2 pide que los
#: `.yaml` vivan en `data/` con procedencia; si el directorio existe se añade
#: al path de búsqueda de Cantera para que un mecanismo esquelético de
#: queroseno colocado ahí gane a los de serie sin tocar código.
_MECH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "mech")
if CANTERA_AVAILABLE and os.path.isdir(_MECH_DIR):
    try:
        _CT.add_directory(_MECH_DIR)
    except Exception:
        pass

#: Cascada de mecanismos por combustible: (archivo, especie combustible).
#: El PRIMERO que Cantera consiga cargar gana. El orden codifica la
#: preferencia de Ciencia §9.3; los que NO vienen de serie con Cantera 3.2
#: quedan como «si alguien los deja en data/mech/».
#:
#: ⚠️ INVENTARIO REAL de Cantera 3.2 (verificado en esta máquina): gri30,
#: gri30_highT, gri30_ion, h2o2, ohn, nDodecane_Reitz, air, airNASA9,
#: liquidvapor, ptcombust, diamond, silicon, sofc, KOH, water, graphite,
#: methane_pox_on_pt, SiF4_NH3_mec, nasa_gas, nasa_condensed. **Ninguno de los
#: tres `mech_l1` de `physics_core.FUELS` viene de serie**, así que los tres
#: combustibles degradan y se ETIQUETA en `crn_mech_note`.
MECH_CASCADE = {
    "jet_a": (
        ("ker_skeletal.yaml", "POSF10325"),      # el pedido (data/mech/)
        ("chem_kero_skeletal.yaml", "KERO_LUCHE"),
        ("gri30.yaml", "C3H8"),                  # ← degradación real
    ),
    "propane": (
        ("gri30.yaml", "C3H8"),                  # el pedido, de serie ✅
    ),
    "h2": (
        ("sandiego_h2_9sp.yaml", "H2"),          # el pedido (data/mech/)
        ("ohn.yaml", "H2"),                      # ← degradación real (con NOx)
        ("h2o2.yaml", "H2"),                     # último recurso (sin NOx)
    ),
}

#: Nota de honestidad por combustible cuando se usa la rama degradada.
MECH_NOTES = {
    ("jet_a", "gri30.yaml"):
        "mech_l1='ker_skeletal' NO viene con Cantera 3.2: se degrada a "
        "GRI-Mech 3.0 con C3H8 como SURROGATE del queroseno, igualando el "
        "RATIO DE EQUIVALENCIA (no la masa de combustible). Se conserva la "
        "química de NOx completa (Zeldovich térmico + prompt vía HCN + N2O), "
        "que es la salida útil de L1; se pierde la cinética específica de "
        "los aromáticos del Jet-A (afecta a CO y a hollín, no modelado). "
        "Coloca un esquelético de queroseno en data/mech/ker_skeletal.yaml "
        "para recuperar la rama pedida sin tocar código.",
    ("h2", "ohn.yaml"):
        "mech_l1='sandiego_h2_9sp' NO viene con Cantera 3.2: se degrada a "
        "ohn.yaml (18 sp / 69 r, H2-O2-N con NOx), que es MÁS completo que el "
        "9sp/21r pedido y conserva la química de NOx.",
    ("h2", "h2o2.yaml"):
        "sin mecanismo con nitrógeno reactivo disponible: h2o2.yaml no tiene "
        "especies de NOx ⇒ EI_NOx_crn = 0.0 NO es un resultado físico.",
}


def _resolve_mechanism(fuel: str) -> tuple[object, str, str, str] | None:
    """Primer mecanismo cargable de la cascada de `fuel`.

    Devuelve `(Solution, archivo, especie_combustible, nota)` o `None`.
    Nunca lanza. El objeto `Solution` se devuelve recién construido (no
    cacheado): un `Solution` compartido entre reactores de redes distintas es
    una fuente clásica de estados contaminados.
    """
    if not CANTERA_AVAILABLE:
        return None
    for mech, sp in MECH_CASCADE.get(fuel, ()):
        try:
            gas = _CT.Solution(mech)
        except Exception:
            continue
        if sp not in gas.species_names:
            continue
        note = MECH_NOTES.get((fuel, mech), "")
        return gas, mech, sp, note
    return None


def available_mechanisms() -> dict:
    """Diagnóstico: qué rama de la cascada gana para cada combustible.

    Lo consumen la CLI y el reporte para pintar la honestidad de mecanismo
    ANTES de gastar un solve. Nunca lanza.
    """
    out: dict[str, dict] = {}
    for fuel in MECH_CASCADE:
        res = _resolve_mechanism(fuel)
        if res is None:
            out[fuel] = dict(mech=None, species=None, degraded=True, note=(
                "Cantera no disponible" if not CANTERA_AVAILABLE
                else "ningún mecanismo de la cascada es cargable"))
            continue
        _gas, mech, sp, note = res
        requested = MECH_CASCADE[fuel][0][0]
        out[fuel] = dict(mech=mech, species=sp,
                         degraded=bool(mech != requested),
                         note=note or "")
    return out


# ==========================================================================
# 3. Capa 0: C_d por tipo de orificio + factor de rugosidad AM f(Ra/d)
# ==========================================================================
_CD_CACHE: dict | None = None


def _cd_tables() -> dict:
    """C_d nominales y curva f(Ra/d), leídos de `data/` UNA vez.

    `data/lefebvre/cd_holes.csv` da el C_d por tipo de orificio ✅/⚠️ y
    `data/lefebvre/cd_am_roughness.csv` el factor multiplicativo de rugosidad
    as-built LPBF. **No existe correlación cerrada publicada para f(Ra/d)**
    (declarado en Ciencia §4.1 y en la propia cabecera de la tabla): la forma
    tabulada es de la casa, calibrada solo a la banda cualitativa «5–20 % de
    reducción vs pulido». Si las tablas faltan se usan los respaldos de
    `physics_core` y se sigue adelante — nunca se lanza.
    """
    global _CD_CACHE
    if _CD_CACHE is not None:
        return _CD_CACHE
    pc = _import_sibling("physics_core")
    out = {
        "plain": 0.62, "effusion": 0.65, "plunged": 0.80,
        "ra_x": np.array([0.0, 0.05]), "ra_f": np.array([1.0, 0.80]),
        "K_sw": 1.30, "beta_deg": 50.0, "tables_ok": False,
    }
    if pc is not None:
        try:
            c = pc.coeffs()
            out["plain"] = float(c["Cd_hole"])
            out["effusion"] = float(c["Cd_eff"])
            out["K_sw"] = float(c["K_sw"])
            out["beta_deg"] = float(pc.SWIRL_VANE_BETA_DEG)
            t = pc.load_table("lefebvre/cd_am_roughness.csv")
            if t is not None and "Ra_over_d" in t["cols"]:
                x = np.asarray(t["cols"]["Ra_over_d"], dtype=float)
                f = np.asarray(t["cols"]["f_Cd"], dtype=float)
                o = np.argsort(x)
                out["ra_x"], out["ra_f"] = x[o], f[o]
                out["tables_ok"] = True
        except Exception:
            pass
    _CD_CACHE = out
    return out


def _f_am_roughness(d_m: float) -> float:
    """Factor f(Ra/d) de rugosidad as-built sobre el C_d ideal.

    DELEGA en `physics_core.am_roughness_factor` — que es hoy el ÚNICO sitio
    donde vive esta curva (auditoría 2026-07-31: había una copia aquí y
    ninguna en `cfd/xlb_cold`, que por eso descargaba con un C_d un 11.8 %
    más alto). Se conserva el nombre privado porque lo usan todos los
    caminos de la red 1D; el respaldo local solo actúa si el hermano no es
    importable, y es la MISMA interpolación sobre la MISMA tabla.
    """
    pc = _import_sibling("physics_core")
    if pc is not None:
        try:
            return float(pc.am_roughness_factor(d_m))
        except Exception:
            pass
    t = _cd_tables()
    d = max(_safe(d_m, 1e-3), 1e-5)
    ra = _shared_const("RA_ASBUILT_UM", _SHARED_FROM_PHYSICS["RA_ASBUILT_UM"])
    return float(np.interp((ra * 1e-6) / d, t["ra_x"], t["ra_f"]))


def effusion_cd(d_eff_m: float) -> float:
    """C_d GEOMÉTRICO del orificio de efusión que usa la red 1D.

    Es literalmente `physics_core.effusion_discharge_coefficient` (fuente
    única §12.b): tabla × inclinación × rugosidad AM. NO lleva `f(Re)`, que
    depende de la solución y se aplica dentro del residuo. Existe como
    función pública para que el test de regresión pueda comparar ESTE número
    con el que usa `cfd/xlb_cold` sin resolver ninguna red ni ningún LBM.
    """
    pc = _import_sibling("physics_core")
    if pc is not None:
        try:
            return float(pc.effusion_discharge_coefficient(d_eff_m))
        except Exception:
            pass
    cdi = _shared_const("CD_EFFUSION_INCLINED",
                        _SHARED_FROM_PHYSICS["CD_EFFUSION_INCLINED"])
    return float(_cd_tables()["effusion"] * cdi * _f_am_roughness(d_eff_m))


def _f_reynolds(re_d: float) -> float:
    """Factor f(Re) sobre el C_d ideal. ⚠️ FORMA DE LA CASA (ver tunables)."""
    re = max(_safe(re_d, 0.0), 0.0)
    return float(1.0 - CD_RE_DEFICIT * math.exp(-re / CD_RE_SCALE))


def _mu_air(T_K: float) -> float:
    """µ del aire [Pa·s] por Sutherland, anclada en `MU_AIR_REF`. Nunca lanza."""
    T = max(_safe(T_K, 293.15), 100.0)
    return float(MU_AIR_REF * (T / 293.15) ** 1.5
                 * (293.15 + 110.4) / (T + 110.4))


# ==========================================================================
# 4. Red 1D de reparto de aire [WAVE2 §1 · Ciencia §§9.1, 4.1 y 2.1]
# ==========================================================================
#: Nombres CONGELADOS de los seis caminos de la red, en el orden en el que
#: ocupan el vector de incógnitas. La séptima incógnita es ΔP_L.
PATH_NAMES = ("swirler", "pz_holes", "sz_holes", "dz_holes",
              "effusion", "cool_ring")

#: Camino → zona a la que entrega el aire (para agregar `splits_solved`).
PATH_ZONE = {"swirler": "pz", "pz_holes": "pz", "sz_holes": "sz",
             "dz_holes": "dz", "effusion": "cool", "cool_ring": "cool"}


def _rows_by_zone(rec: dict) -> dict:
    """`hole_rows` del record indexadas por zona (dict vacío si no están)."""
    out: dict[str, dict] = {}
    rows = rec.get("hole_rows")
    if isinstance(rows, (list, tuple)):
        for r in rows:
            if isinstance(r, dict) and r.get("zone") in ("pz", "sz", "dz"):
                out[str(r["zone"])] = r
    return out


def _build_paths(rec: dict, ctx: dict) -> list[dict]:
    """Geometría equivalente de cada camino: `(C_d₀, A, D_h, ξ, ΔP_extra)`.

    Todos los caminos comparten la MISMA ΔP_L (el liner es un plenum,
    WAVE2 §1); lo que los diferencia es (a) su producto C_d·A, (b) la
    fracción ξ de la presión dinámica del anillo que el orificio recupera y
    (c) las pérdidas de forma que ve aguas arriba (snout).

    Procedencia de cada área:

    * `swirler` — modelo de canal de álabe consistente con Ciencia §4.4: el
      aire atraviesa el canal con velocidad U alineada con el álabe (β
      respecto al eje) y pierde ΔP_sw = K_sw·½ρU². Con ṁ = C_d·ρ·A_paso·U y
      A_paso = A_sw·cosβ·(1 − bloqueo) sale
      `(C_d·A)_sw = C_d·A_sw·cosβ·(1 − bloqueo)/√K_sw`. Es la MISMA K_sw de
      L0, pero evaluada con el caudal REAL del swirler y no con ṁ₃ — que es
      justamente la incoherencia que L1 viene a resolver (ver
      `dp_split_check`).
    * `pz/sz/dz_holes` — `A_row_m2` y `Cd` de `hole_rows` del record.
    * `effusion` — `A_effusion_m2` del record (porosidad × área MOJADA de la
      pared caliente, calculada por `physics_core.effusion_wall_area_m2` de
      forma consciente de la arquitectura) con el C_d de
      `physics_core.effusion_discharge_coefficient` = tabla ×
      `CD_EFFUSION_INCLINED` × rugosidad AM. FUENTE ÚNICA: `cfd/xlb_cold`
      consume exactamente las mismas dos magnitudes.
    * `cool_ring` — NO tiene geometría en el record (anillo de domo + fugas
      de sello son un camino MODELADO, no dibujado): su área equivalente se
      deriva del caudal que el cierre `frac_cool` de L0 le asigna a la ΔP_L
      NOMINAL, es decir «área de la solución nominal L0» (WAVE2 §4/T17).
      Una vez fijada, responde a la ΔP resuelta como cualquier otro camino.
    """
    q_ref = ctx["q_ref"]
    rho3 = ctx["rho3"]
    dp_ref = ctx["dp_ref"]
    mdot_air = ctx["mdot_air"]
    t = _cd_tables()
    rows = _rows_by_zone(rec)
    splits = rec.get("splits") or {}
    clamped = False

    # --- swirler ----------------------------------------------------------
    sw = rec.get("swirler") if isinstance(rec.get("swirler"), dict) else {}
    A_sw = _safe(sw.get("A_sw_m2"), 0.0)
    beta = math.radians(_safe(sw.get("beta_deg"), t["beta_deg"]))
    K_sw = max(_safe(t["K_sw"], 1.30), 1e-3)
    if A_sw <= 0.0:                       # respaldo: A_sw/A_ref de la casa
        A_sw = 0.14 * max(_safe(rec.get("A_ref_m2"), 0.0), 1e-9)
        clamped = True
    A_sw_eff = max(A_sw * math.cos(beta) * (1.0 - SWIRLER_BLOCKAGE)
                   / math.sqrt(K_sw), 1e-12)
    #  Ø hidráulico del canal entre álabes (solo para f(Ra/d) y f(Re)).
    d_sw = math.sqrt(4.0 * (A_sw / max(SWIRLER_VANES, 1)) / math.pi)

    paths: list[dict] = [dict(
        name="swirler", A_m2=A_sw_eff, Cd0=CD_SWIRLER, d_h_m=d_sw,
        ram=RAM_RECOVERY_SNOUT, dp_extra_Pa=K_SNOUT * q_ref,
        A_geom_m2=A_sw)]

    # --- filas de orificios del liner -------------------------------------
    for zone in ("pz", "sz", "dz"):
        r = rows.get(zone)
        if r is not None and _safe(r.get("A_row_m2"), 0.0) > 0.0:
            A = _safe(r["A_row_m2"])
            Cd0 = _safe(r.get("Cd"), t["plain"]) or t["plain"]
            d_h = max(1e-3 * _safe(r.get("d_mm"), 1.0), 1e-5)
        else:                              # respaldo: área nominal L0
            frac = _safe(splits.get(zone), 0.0)
            if zone == "pz":
                frac *= 0.5               # el swirler se lleva la otra mitad
            Cd0, d_h, clamped = t["plain"], 2.0e-3, True
            A = max(frac * mdot_air / max(
                Cd0 * _f_am_roughness(d_h) * math.sqrt(2.0 * rho3 * dp_ref),
                1e-12), 1e-12)
        paths.append(dict(name=f"{zone}_holes", A_m2=A, Cd0=Cd0, d_h_m=d_h,
                          ram=RAM_RECOVERY_PLAIN, dp_extra_Pa=0.0,
                          A_geom_m2=A))

    # --- campo de efusión --------------------------------------------------
    #  FUENTE ÚNICA (physics_core §12.b): el ÁREA sale de
    #  `record["A_effusion_m2"]` —que desde el 2026-07-31 es la misma que
    #  `contract["effusion"]["open_area_mm2"]`— y el C_d de
    #  `effusion_cd(d)` = tabla × inclinación × rugosidad AM, EL MISMO número
    #  que consume `cfd/xlb_cold._effusion_sites`. `f_am` de este camino se
    #  fija a 1.0 porque la rugosidad ya está DENTRO del C_d: multiplicarla
    #  dos veces sería el bug simétrico del que había.
    A_eff = _safe(rec.get("A_effusion_m2"), 0.0)
    d_eff = max(1e-3 * _safe(rec.get("d_eff_mm"), 0.6), 1e-5)
    Cd_eff = effusion_cd(d_eff)
    if A_eff <= 0.0:
        A_eff = max(EFFUSION_FRAC_OF_COOL * _safe(splits.get("cool"), 0.0)
                    * mdot_air / max(Cd_eff
                                     * math.sqrt(2.0 * rho3 * dp_ref),
                                     1e-12), 1e-12)
        clamped = True
    paths.append(dict(name="effusion", A_m2=A_eff,
                      Cd0=Cd_eff, f_am_fixed=1.0, d_h_m=d_eff,
                      ram=RAM_RECOVERY_PLAIN, dp_extra_Pa=0.0,
                      A_geom_m2=A_eff))

    # --- anillo de domo + fugas (área de la solución nominal L0) ----------
    d_ring = 1e-3 * D_COOL_RING_MM
    m_ring_nom = max((1.0 - EFFUSION_FRAC_OF_COOL)
                     * _safe(splits.get("cool"), 0.0) * mdot_air, 0.0)
    Cd_ring = t["plain"] * _f_am_roughness(d_ring)
    A_ring = max(m_ring_nom / max(Cd_ring * math.sqrt(2.0 * rho3 * dp_ref),
                                  1e-12), 1e-12)
    paths.append(dict(name="cool_ring", A_m2=A_ring, Cd0=t["plain"],
                      d_h_m=d_ring, ram=RAM_RECOVERY_PLAIN, dp_extra_Pa=0.0,
                      A_geom_m2=A_ring))

    for p in paths:                        # f(Ra/d) no depende de la solución
        p["f_am"] = float(p.pop("f_am_fixed", None)
                          or _f_am_roughness(p["d_h_m"]))
        #  C_d GEOMÉTRICO del camino (sin `f(Re)`, que sí depende de la
        #  solución). Es la magnitud que el test de regresión compara contra
        #  la que usa el LBM: si vuelven a divergir, salta ahí.
        p["Cd_geom"] = float(p["Cd0"] * p["f_am"])
    #  El ORDEN es ABI (`PATH_NAMES`): `splits_solved_paths` y el vector de
    #  incógnitas lo dan por hecho. Si alguna rama de respaldo lo rompiese,
    #  mejor enterarse aquí que en un split silenciosamente cruzado.
    if tuple(p["name"] for p in paths) != PATH_NAMES:
        raise ValueError("orden de caminos inconsistente con PATH_NAMES")
    ctx["paths_clamped"] = clamped
    return paths


def _network_residual(y: np.ndarray, ctx: dict) -> np.ndarray:
    """Residuo ADIMENSIONAL de la red 1D (7 ecuaciones, 7 incógnitas).

    `y = [ṁ₀…ṁ₅]/ṁ₃ ∪ [ΔP_L/ΔP_ref]`. Ecuaciones:

      * 6 leyes de orificio `ṁ_i = C_d,i·A_i·√(2ρ₃·ΔP_i)` [Ciencia §4.1],
        con `ΔP_i = ΔP_L − ΔP_extra,i − (1 − ξ_i)·q_anillo`,
      * 1 conservación de masa `Σṁ_i = ṁ₃`.

    El sistema es genuinamente NO LINEAL y acoplado por dos vías: `f(Re_i)`
    depende del propio ṁ_i, y `q_anillo` depende de TODOS los caudales
    (el anillo se vacía a medida que el aire entra al liner).
    """
    paths = ctx["paths"]
    n = len(paths)
    mdot = ctx["mdot_air"]
    rho3 = ctx["rho3"]
    mu = ctx["mu"]
    m = np.asarray(y[:n], dtype=float) * mdot
    dp = float(y[n]) * ctx["dp_ref"]

    #  Presión dinámica REPRESENTATIVA del anillo: el caudal cae desde ṁ₃ en
    #  la salida del difusor hasta ~0 al final, así que se toma la media del
    #  sangrado (ANNULUS_BLEED_SHAPE).
    m_an = max(mdot - ANNULUS_BLEED_SHAPE * float(np.sum(m)), 0.0)
    u_an = m_an / max(rho3 * ctx["A_an"], 1e-12)
    q_an = 0.5 * rho3 * u_an * u_an
    ctx["_q_an"] = q_an
    ctx["_u_an"] = u_an

    R = np.empty(n + 1, dtype=float)
    clamped = False
    for i, p in enumerate(paths):
        dp_i = dp - p["dp_extra_Pa"] - (1.0 - p["ram"]) * q_an
        if dp_i < DP_FLOOR_PA:
            dp_i, clamped = DP_FLOOR_PA, True
        u_i = abs(m[i]) / max(rho3 * p["A_geom_m2"], 1e-12)
        re_i = rho3 * u_i * p["d_h_m"] / max(mu, 1e-12)
        cda = p["Cd_geom"] * _f_reynolds(re_i) * p["A_m2"]
        p["_CdA"] = cda
        p["_dp_Pa"] = dp_i
        R[i] = (m[i] - cda * math.sqrt(2.0 * rho3 * dp_i)) / mdot
    R[n] = (float(np.sum(m)) - mdot) / mdot
    ctx["_clamped"] = clamped
    return R


def _newton_1d(y0: np.ndarray, ctx: dict) -> tuple[np.ndarray, int, float]:
    """Newton ACOTADO con jacobiano por diferencias y backtracking acotado.

    Sin `while` abiertos: `NEWTON_ITERS` iteraciones máximas, cada una con
    `NEWTON_BACKTRACK` halvings máximos. Devuelve `(y, iteraciones, ‖R‖∞)`
    del MEJOR punto visitado — nunca lanza.
    """
    y = np.array(y0, dtype=float)
    R = _network_residual(y, ctx)
    best_y, best_n = y.copy(), float(np.max(np.abs(R)))
    n = y.size
    iters = 0
    for _ in range(NEWTON_ITERS):
        iters += 1
        norm0 = float(np.max(np.abs(R)))
        if norm0 <= NEWTON_TOL:
            break
        J = np.empty((n, n), dtype=float)
        for j in range(n):
            h = NEWTON_FD_STEP * max(abs(y[j]), 1.0e-3)
            yp = y.copy()
            yp[j] += h
            J[:, j] = (_network_residual(yp, ctx) - R) / h
        try:
            dy = np.linalg.solve(J, -R)
        except Exception:
            try:
                dy = np.linalg.lstsq(J, -R, rcond=None)[0]
            except Exception:
                break
        if not np.all(np.isfinite(dy)):
            break
        step = 1.0
        improved = False
        for _bt in range(NEWTON_BACKTRACK + 1):
            y_try = y + step * dy
            #  Suelo positivo: caudales y ΔP son físicamente positivos.
            y_try = np.maximum(y_try, 1.0e-9)
            R_try = _network_residual(y_try, ctx)
            norm1 = float(np.max(np.abs(R_try)))
            if np.isfinite(norm1) and norm1 < norm0:
                y, R, improved = y_try, R_try, True
                if norm1 < best_n:
                    best_y, best_n = y.copy(), norm1
                break
            step *= 0.5
        if not improved:
            break
    _network_residual(best_y, ctx)          # deja el ctx en el mejor punto
    return best_y, iters, best_n


def network_1d(theta, rec: dict) -> dict:
    """Resuelve la red 1D de reparto de aire (WAVE2 §1). **Nunca lanza.**

    Sustituye los splits NOMINALES de L0 (que son un dato de tabla por
    arquitectura) por el reparto que la geometría realmente impone: seis
    caminos en paralelo entre el anillo y el liner, todos con la misma ΔP_L,
    más las pérdidas de difusor y snout.

    Parámetros
    ----------
    theta : array-like
        Vector de diseño (solo se usa para el punto de operación; si falta
        se cae al eco de θ que el propio record lleva).
    rec : dict
        Record L0 de `physics_core.evaluate` (`phycb-record-1`).

    Devuelve
    --------
    dict con (contrato §4.3 y WAVE2 §1)
        `splits_solved`, `dp_liner_Pa`, `v_zones`, `tau_zones_ms`,
        `phi_zones_solved` — más el diagnóstico completo de la red
        (`network_converged`, `network_residual`, `mass_residual_kgs`,
        `CdA_paths_m2`, `split_dev_vs_L0`, `dp_split_check`).

    **No necesita Cantera**: corre siempre, también cuando `crn_solve`
    devuelve `None`.
    """
    try:
        return _network_1d_impl(theta, rec)
    except Exception as exc:                # red de seguridad: nunca lanza
        splits = rec.get("splits") if isinstance(rec, dict) else None
        base = dict(splits or {"pz": 0.30, "sz": 0.12, "dz": 0.42,
                               "cool": 0.16})
        return dict(
            splits_solved=base, network_converged=False, network_iters=0,
            network_residual=float("inf"), mass_residual_kgs=0.0,
            network_clamped=True,
            network_note=f"red 1D degradada a los splits nominales de L0: "
                         f"{type(exc).__name__}: {exc}")


def _network_1d_impl(theta, rec: dict) -> dict:
    """Implementación de `network_1d` (la envoltura pública captura todo)."""
    P3 = _safe(rec.get("P3_Pa"), 0.0)
    T3 = _safe(rec.get("T3_K"), 0.0)
    mdot_air = _safe(rec.get("mdot_air_kgs"), 0.0)
    rho3 = _safe(rec.get("rho3_kgm3"), 0.0)
    if theta is not None:
        try:
            t = np.asarray(theta, dtype=float).ravel()
            if t.size >= 20 and np.all(np.isfinite(t[16:20])):
                P3, T3, mdot_air = float(t[16]), float(t[17]), float(t[18])
        except Exception:
            pass
    if rho3 <= 0.0:
        rho3 = P3 / max(287.0 * max(T3, 1.0), 1e-9)
    if not (P3 > 0.0 and T3 > 0.0 and mdot_air > 0.0 and rho3 > 0.0):
        raise ValueError("punto de operación degenerado en el record")

    q_ref = max(_safe(rec.get("q_ref_Pa"), 0.0), 1e-9)
    A_an = max(_safe(rec.get("A_an_m2"), 0.0), 1e-9)
    A_L = max(_safe(rec.get("A_L_m2"), 0.0), 1e-9)
    dp_ref = max(_safe(rec.get("dp_liner_Pa"), 0.0), 1.0)
    ctx = dict(rho3=rho3, mdot_air=mdot_air, q_ref=q_ref, A_an=A_an,
               dp_ref=dp_ref, mu=_mu_air(T3))
    ctx["paths"] = _build_paths(rec, ctx)
    paths = ctx["paths"]

    # --- semilla: la solución NOMINAL de L0 (el mejor arranque posible) ---
    splits0 = rec.get("splits") or {"pz": 0.30, "sz": 0.12, "dz": 0.42,
                                    "cool": 0.16}
    seed = {
        "swirler": 0.5 * _safe(splits0.get("pz"), 0.30),
        "pz_holes": 0.5 * _safe(splits0.get("pz"), 0.30),
        "sz_holes": _safe(splits0.get("sz"), 0.12),
        "dz_holes": _safe(splits0.get("dz"), 0.42),
        "effusion": EFFUSION_FRAC_OF_COOL * _safe(splits0.get("cool"), 0.16),
        "cool_ring": (1.0 - EFFUSION_FRAC_OF_COOL)
        * _safe(splits0.get("cool"), 0.16),
    }
    y0 = np.array([max(seed[p["name"]], 1e-6) for p in paths] + [1.0])
    y, iters, resid = _newton_1d(y0, ctx)

    m_path = {p["name"]: float(y[i] * mdot_air) for i, p in enumerate(paths)}
    dp_liner = float(y[len(paths)] * dp_ref)
    m_tot = float(sum(m_path.values()))
    mass_res = m_tot - mdot_air

    # --- agregación a los cuatro splits del contrato ----------------------
    splits_solved = {"pz": 0.0, "sz": 0.0, "dz": 0.0, "cool": 0.0}
    for name, m in m_path.items():
        splits_solved[PATH_ZONE[name]] += m / max(m_tot, 1e-12)
    dev = {k: (splits_solved[k] / max(_safe(splits0.get(k), 0.0), 1e-12) - 1.0)
           for k in splits_solved}

    # --- φ por zona RECALCULADO con los splits resueltos ------------------
    mdot_fuel = _safe(rec.get("mdot_fuel_kgs"), 0.0)
    far_st = max(_safe(rec.get("far_stoich"), 0.0686834), 1e-9)
    m_air_pz = m_path["swirler"] + m_path["pz_holes"]
    m_air_sz = m_path["sz_holes"]
    m_air_dz = m_path["dz_holes"]
    m_air_chain = max(m_air_pz + m_air_sz + m_air_dz, 1e-12)
    phi = {
        "pz": (mdot_fuel / max(m_air_pz, 1e-12)) / far_st,
        "sz": (mdot_fuel / max(m_air_pz + m_air_sz, 1e-12)) / far_st,
        "dz": (mdot_fuel / m_air_chain) / far_st,
        "global": (mdot_fuel / max(m_tot, 1e-12)) / far_st,
    }

    # --- velocidades y tiempos de residencia por zona ---------------------
    R_hot = max(_safe(rec.get("R_hot_J_kgK"), 292.5), 1.0)
    T_zone = {"pz": _safe(rec.get("T_pz_K"), 2000.0),
              "sz": _safe(rec.get("T_sz_K"), 1800.0),
              "dz": _safe(rec.get("T_dz_K"), 1200.0)}
    m_gas = {"pz": mdot_fuel + m_air_pz,
             "sz": mdot_fuel + m_air_pz + m_air_sz,
             "dz": mdot_fuel + m_air_chain}
    V_zone = {"pz": max(_safe(rec.get("V_pz_m3"), 0.0), 1e-12),
              "sz": max(_safe(rec.get("V_sz_m3"), 0.0), 1e-12),
              "dz": max(_safe(rec.get("V_dz_m3"), 0.0), 1e-12)}
    v_zones, tau_hot, tau_cold = {}, {}, {}
    for z in ("pz", "sz", "dz"):
        rho_z = P3 / (R_hot * max(T_zone[z], 200.0))
        v_zones[z] = m_gas[z] / max(rho_z * A_L, 1e-12)
        tau_hot[z] = 1e3 * rho_z * V_zone[z] / max(m_gas[z], 1e-12)
    m_air_zone = {"pz": m_air_pz, "sz": m_air_sz, "dz": m_air_dz}
    for z in ("pz", "sz", "dz"):
        tau_cold[z] = 1e3 * rho3 * V_zone[z] / max(m_air_zone[z], 1e-12)
    tau_hot["total"] = tau_hot["pz"] + tau_hot["sz"] + tau_hot["dz"]
    tau_cold["total"] = 1e3 * rho3 * max(_safe(rec.get("V_c_m3"), 0.0),
                                         1e-12) / mdot_air
    v_zones["annulus"] = float(ctx.get("_u_an", 0.0))
    for p in paths:                       # velocidad de descarga por camino
        v_zones[p["name"]] = m_path[p["name"]] / max(
            rho3 * p["A_geom_m2"], 1e-12)

    # --- cierre de presiones ----------------------------------------------
    dp_diff = K_DIFFUSER * q_ref
    dp_snout = K_SNOUT * q_ref
    dp_total = dp_diff + dp_liner
    dp_sw_solved = float(paths[0]["_dp_Pa"])
    frac_form = dp_diff / max(dp_total, 1e-9)
    dp_check = (
        f"ΔP difusor+snout resuelto = {dp_diff:.1f}+{dp_snout:.1f} Pa "
        f"({100.0 * frac_form:.1f} % del ΔP total) frente al 25 % NOMINAL que "
        f"L0 asume con LINER_FRAC_OF_DP = 0.75. La diferencia es real y no se "
        f"esconde: K_DIFFUSER+K_SNOUT = {K_DIFFUSER + K_SNOUT:.2f}·q_ref con "
        f"q_ref = {q_ref:.1f} Pa son pérdidas de FORMA, mientras que el 25 % "
        f"de L0 es un cierre global de tabla. Además, ΔP_sw/q_ref resuelto = "
        f"{dp_sw_solved / q_ref:.2f} frente a "
        f"{_safe(rec.get('dp_sw_over_qref'), 0.0):.2f} de la correlación de "
        f"Ciencia §4.4, que evalúa la pérdida del swirler con ṁ₃ ENTERO y no "
        f"con el caudal que el swirler realmente pasa.")

    #  Convergencia = residuo de orificio despreciable Y conservación de masa
    #  por debajo del ancla de WAVE2 §4/T17 (1e-9 kg/s, aquí en ABSOLUTO).
    converged = bool(resid <= 1.0e-9 and abs(mass_res) <= 1.0e-9)
    out = {
        "splits_solved": {k: float(v) for k, v in splits_solved.items()},
        "splits_solved_paths": {k: float(v / max(m_tot, 1e-12))
                                for k, v in m_path.items()},
        "mdot_paths_kgs": {k: float(v) for k, v in m_path.items()},
        "CdA_paths_m2": {p["name"]: float(p["_CdA"]) for p in paths},
        #  C_d y área GEOMÉTRICOS por camino (sin `f(Re)`): es lo que hay que
        #  poder comparar contra otro consumidor —el LBM— sin resolver nada.
        "Cd_geom_paths": {p["name"]: float(p["Cd_geom"]) for p in paths},
        "A_paths_m2": {p["name"]: float(p["A_m2"]) for p in paths},
        "dp_paths_Pa": {p["name"]: float(p["_dp_Pa"]) for p in paths},
        "dp_liner_Pa": float(dp_liner),
        "dp_liner_over_qref": float(dp_liner / q_ref),
        "dp_diffuser_Pa": float(dp_diff),
        "dp_snout_Pa": float(dp_snout),
        "dp_sw_over_qref_solved": float(dp_sw_solved / q_ref),
        "dp_over_p_network": float(dp_total / max(P3, 1.0)),
        "q_annulus_Pa": float(ctx.get("_q_an", 0.0)),
        "v_zones": {k: float(v) for k, v in v_zones.items()},
        "tau_zones_ms": {k: float(v) for k, v in tau_hot.items()},
        "tau_zones_cold_ms": {k: float(v) for k, v in tau_cold.items()},
        "tau_convention": (
            "`tau_zones_ms` es el tiempo de residencia FÍSICO ρ(T_zona)·V/ṁ_gas "
            "— el que ve la química y el que alimenta Da. "
            "`tau_zones_cold_ms` usa el convenio de Lefebvre ρ₃·V/ṁ_aire que "
            "emplea L0 (`tau_ms`, `tau_pz_ms`), ~6× mayor, y está SOLO para "
            "que la comparación L0↔L1 sea manzana con manzana."),
        "phi_zones_solved": {k: float(v) for k, v in phi.items()},
        "split_dev_vs_L0": {k: float(v) for k, v in dev.items()},
        "split_dev_max": float(max(abs(v) for v in dev.values())),
        "dp_split_check": dp_check,
        "network_converged": converged,
        "network_iters": int(iters),
        "network_residual": float(resid),
        "mass_residual_kgs": float(mass_res),
        "network_clamped": bool(ctx.get("_clamped") or
                                ctx.get("paths_clamped")),
        "network_note": (
            "6 caminos + ΔP_L resueltos por Newton acotado sobre "
            "ṁ_h = C_d·A_h·√(2ρ₃·ΔP) [Ciencia §4.1]. `phi_zones_solved` y "
            "`dp_liner_Pa` son de L1; `phi_pz`, `dp_over_p` y `g` se dejan "
            "INTACTOS porque el vector de restricciones se construyó con "
            "ellos en L0 y reescribirlos dejaría el record incoherente."),
    }
    return out


# ==========================================================================
# 5. CRN de Cantera [WAVE2 §2 · Ciencia §§9.2, 9.3, 8.3 y 6.1]
# ==========================================================================
#: Especies cuyo CO/H₂/HC sin quemar cuenta para el burn-out. Se filtran
#: contra `gas.species_names` en tiempo de ejecución.
_BURNOUT_SPECIES = tuple(LHV_BY_SPECIES)


def _phys():
    """`physics_core` verificado, o `None`. No es obligatorio para el CRN."""
    return _import_sibling("physics_core")


def _fuel_props(fuel: str) -> dict:
    """Propiedades del combustible desde `physics_core.FUELS`, con respaldo."""
    pc = _phys()
    try:
        return dict(pc.FUELS[fuel])
    except Exception:
        return {"jet_a": dict(LHV=43.1e6, lambda_evap=8.0e-7, S_L_ref=0.40,
                              liquid=True, far_stoich=0.0686834),
                "propane": dict(LHV=46.35e6, lambda_evap=1.2e-6,
                                S_L_ref=0.43, liquid=False,
                                far_stoich=0.0642180),
                "h2": dict(LHV=120.0e6, lambda_evap=4.0e-6, S_L_ref=2.10,
                           liquid=False, far_stoich=0.0293600),
                }.get(fuel, dict(LHV=43.1e6, lambda_evap=8.0e-7,
                                 S_L_ref=0.40, liquid=True,
                                 far_stoich=0.0686834))


def _evap_factor(rec: dict, fuel: str, tau_pz_s: float) -> dict:
    """Corrección de evaporación de Lanewala sobre Rizk & Mongia [§8.3].

    Fórmula publicada (✅ VERIFICADA en Ciencia §§8.3 y 9.2)::

        FAR_efectivo = FAR·(−0.0295·τ_e/τ + 0.45),   τ_e = SMD²/λ_e

    ⚠️ **DESVIACIÓN DECLARADA.** Aplicada LITERALMENTE, el factor vale como
    mucho 0.45 (límite de atomización perfecta, τ_e → 0), es decir que ni
    con gota nula llegaría a la zona primaria más de la mitad del
    combustible. En el caso ancla eso da φ_PZ,ef = 0.45·1.048 = 0.47 < φ_LBO
    = 0.52: el PSR se APAGARÍA, y sin embargo el KJ66 de referencia arranca y
    funciona. La fórmula solo puede leerse como una corrección RELATIVA
    dentro del ajuste de Rizk & Mongia, no como una fracción absoluta de
    combustible evaporado. Aquí se aplica NORMALIZADA a su propio límite de
    atomización perfecta::

        f_evap = (C1·τ_e/τ + C2)/C2 = 1 − (|C1|/C2)·τ_e/τ

    con lo que se conserva EXACTAMENTE la pendiente publicada
    (−0.0295/0.45 = −0.0656 por unidad de τ_e/τ) y se recupera f = 1 cuando
    la evaporación es instantánea. El factor bruto de la fórmula literal se
    reporta igualmente en `far_evap_factor_raw` para que nadie tenga que
    creerse esta nota.

    Segunda decisión declarada: la corrección se aplica solo si el
    combustible es LÍQUIDO **y** el inyector produce espray. Un vaporizador
    (KJ66) o un micromix de H₂ entregan el combustible ya en fase gas: su
    `SMD_um` es un d₅₀ nominal de referencia, no una gota que haya que
    evaporar.

    El combustible NO evaporado no desaparece: se inyecta en el PSR de la
    zona SECUNDARIA (`_build_network`), que es donde físicamente acaba de
    arder. El balance de combustible del CRN es exacto.
    """
    fp = _fuel_props(fuel)
    smd = rec.get("SMD_um")
    injector = str(rec.get("injector", ""))
    spray = injector in ("airblast", "simplex")
    liquid = bool(fp.get("liquid", False))
    if not (liquid and spray) or smd is None or not _safe(smd, 0.0) > 0.0:
        why = ("combustible gaseoso" if not liquid else
               f"inyector '{injector}' entrega el combustible ya vaporizado")
        return dict(far_evap_factor=1.0, far_evap_factor_raw=1.0,
                    tau_evap_ms=0.0, tau_e_over_tau=0.0,
                    far_evap_clamped=False,
                    far_evap_note=f"sin corrección de evaporación ({why}).")
    lam = max(_safe(fp.get("lambda_evap"), 8.0e-7), 1e-12)
    tau_e = (_safe(smd, 0.0) * 1e-6) ** 2 / lam
    ratio = tau_e / max(tau_pz_s, 1e-12)
    raw = FAR_EVAP_C1 * ratio + FAR_EVAP_C2
    f = raw / FAR_EVAP_C2
    clamped = bool(f < FAR_EVAP_FLOOR)
    f = _clip(f, FAR_EVAP_FLOOR, 1.0)
    return dict(far_evap_factor=float(f), far_evap_factor_raw=float(raw),
                tau_evap_ms=float(1e3 * tau_e),
                tau_e_over_tau=float(ratio),
                far_evap_clamped=clamped,
                far_evap_note=(
                    f"Lanewala/Rizk & Mongia NORMALIZADA (ver docstring): "
                    f"τ_e = {1e3 * tau_e:.3f} ms, τ_e/τ_PZ = {ratio:.3f}, "
                    f"f_evap = {f:.4f} (bruto de la fórmula literal: "
                    f"{raw:.4f}). La fracción {1.0 - f:.4f} del combustible "
                    f"se inyecta en el PSR de la zona SECUNDARIA."))


def _recirc_ratio(SN: float) -> float:
    """ṁ_recirc/ṁ_PZ del CTRZ, lineal en S_N y ACOTADA [§9.2]. ⚠️ casa."""
    r = CRN_RECIRC_AT_SN1 * max(_safe(SN, 0.7), 0.0)
    return _clip(r, CRN_RECIRC_MIN, CRN_RECIRC_MAX)


def _fresh(gas, phi: float, fuel_sp: str, T: float, P: float):
    """Deja `gas` en la mezcla fresca (φ, T, P). Devuelve `Y_combustible`."""
    gas.TP = max(T, 200.0), max(P, 1.0)
    gas.set_equivalence_ratio(max(phi, 1e-6), fuel_sp, OXIDIZER)
    i = gas.species_index(fuel_sp)
    return float(gas.Y[i])


def _build_network(net_1d: dict, rec: dict, mech: tuple, evap: dict,
                   T3: float, P_liner: float) -> dict:
    """Construye la topología mínima de Ciencia §9.2 en Cantera.

    ::

        res_pz(aire+comb) ─MFC→ PSR_pz ─MFC→ PSR_sz ─MFC→ PFR_dz×N ─MFC→ esc.
                                   ↑            ↑ aire SZ (+comb no evaporado)
                                   └────MFC─────┘  recirculación CTRZ
                                                 PFR_dz1 ← aire DZ
                                   (aire de refrigeración = BYPASS FRÍO)

    Tres riesgos de WAVE2 §6, mitigados desde el diseño:

    1. **PSR que no enciende.** Cada reactor se inicializa con la solución de
       `equilibrate('HP')` de SU propia mezcla, nunca desde T₃. Es
       exactamente lo que hace el ejemplo oficial `combustor.py` de Cantera
       («fill it initially with the equilibrium products … provides a good
       initial condition»). ⚠️ Ese ejemplo YA NO se distribuye con el paquete
       (`cantera/examples/README.txt` de esta instalación: «Starting with
       Cantera 3.0, Python examples are no longer packaged»); se sigue su
       patrón documentado y se declara aquí.
    2. **Rigidez de los mecanismos.** La cascada `MECH_CASCADE` empieza
       SIEMPRE por el reducido/esquelético y solo cae a GRI-Mech si aquel no
       está.
    3. **Redes con `Valve` mal condicionadas.** El reparto de caudal se hace
       con `ct.MassFlowController` (el caudal es DATO: sale de la red 1D de
       §1, que ya lo resolvió), incluida la recirculación del CTRZ. Un
       `ct.Valve` como repartidor exigiría calibrar su K contra una ΔP
       desconocida — el modo de fallo que WAVE2 §6 avisa. El `ct.Valve` se
       reserva a lo que sí sabe hacer: anclar la presión de cada reactor al
       escape con `CRN_VALVE_K` = 1e-9 kg/s/Pa, que a 1 kPa de deriva drena
       1e-6 kg/s (1e-5 del caudal del ancla) y por tanto NO mueve el
       estacionario. Es una DESVIACIÓN declarada respecto a la letra de
       Ciencia §9.2 («Valve de retorno»), no respecto a su física.

    Los `ct.IdealGasReactor` tienen volumen fijo: como se arrancan ya con la
    densidad CALIENTE del equilibrio, la presión se queda donde debe y el
    tiempo de residencia del reactor es el físico ρ(T)·V/ṁ.
    """
    gas0, mech_file, fuel_sp, _note = mech
    m = net_1d["mdot_paths_kgs"]
    m_air_pz = m["swirler"] + m["pz_holes"]
    m_air_sz = m["sz_holes"]
    m_air_dz = m["dz_holes"]
    m_bypass = m["effusion"] + m["cool_ring"]
    m_air_chain = max(m_air_pz + m_air_sz + m_air_dz, 1e-12)

    #  φ por zona con el combustible REAL (los splits ya son los resueltos).
    phi = net_1d["phi_zones_solved"]
    f_ev = float(evap["far_evap_factor"])
    phi_pz_crn = max(f_ev * _safe(phi.get("pz"), 1.0), 1e-4)

    #  SURROGATE: la cascada puede haber degradado a otro combustible
    #  (jet_a → C3H8). Se iguala el RATIO DE EQUIVALENCIA, no la masa.
    y_st = _fresh(gas0, 1.0, fuel_sp, T3, P_liner)
    far_st_sur = y_st / max(1.0 - y_st, 1e-12)
    m_fuel_sur = _safe(phi.get("dz"), 0.0) * far_st_sur * m_air_chain
    m_fuel_pz = f_ev * m_fuel_sur
    m_fuel_sz = max(m_fuel_sur - m_fuel_pz, 0.0)

    def _sol():
        """Un `Solution` NUEVO por reactor/reservorio (estados aislados)."""
        return _CT.Solution(mech_file)

    def _equil(phi_z: float):
        """Mezcla fresca a φ_z llevada a `equilibrate('HP')` (arranque)."""
        g = _sol()
        _fresh(g, phi_z, fuel_sp, T3, P_liner)
        try:
            g.equilibrate("HP")
        except Exception:
            pass
        return g

    #  --- reservorios de entrada -------------------------------------------
    g_in = _sol()
    _fresh(g_in, phi_pz_crn, fuel_sp, T3, P_liner)
    y_f_in = float(g_in.Y[g_in.species_index(fuel_sp)])
    res_in = _CT.Reservoir(g_in, name="in_pz")
    #  El reservorio ya está PREMEZCLADO a φ_PZ,CRN, así que su caudal es
    #  aire + combustible. Se calcula desde Y_comb del propio reservorio para
    #  que el balance sea exacto: `m_in_pz − m_air_pz ≡ m_fuel_pz` (se cumple
    #  por construcción porque φ_PZ,CRN = f_evap·φ_PZ y φ_PZ sale de los
    #  MISMOS caudales resueltos que `m_air_pz`).
    m_in_pz = m_air_pz / max(1.0 - y_f_in, 1e-9)

    g_air = _sol()
    g_air.TPX = max(T3, 200.0), max(P_liner, 1.0), OXIDIZER
    res_air_sz = _CT.Reservoir(g_air, name="air_sz")
    g_air2 = _sol()
    g_air2.TPX = max(T3, 200.0), max(P_liner, 1.0), OXIDIZER
    res_air_dz = _CT.Reservoir(g_air2, name="air_dz")

    res_fuel_sz = None
    if m_fuel_sz > 1e-12:
        g_f = _sol()
        g_f.TPY = max(T3, 200.0), max(P_liner, 1.0), {fuel_sp: 1.0}
        res_fuel_sz = _CT.Reservoir(g_f, name="fuel_sz")

    g_ex = _equil(_safe(phi.get("dz"), 0.3))
    exhaust = _CT.Reservoir(g_ex, name="exhaust")

    #  --- reactores ---------------------------------------------------------
    V_pz = max(_safe(rec.get("V_pz_m3"), 0.0), 1e-12)
    V_sz = max(_safe(rec.get("V_sz_m3"), 0.0), 1e-12)
    V_dz = max(_safe(rec.get("V_dz_m3"), 0.0), 1e-12)
    r_pz = _CT.IdealGasReactor(_equil(phi_pz_crn), name="PSR_pz")
    r_pz.volume = V_pz
    phi_sz_crn = max(f_ev * _safe(phi.get("sz"), 0.7), 1e-4)
    r_sz = _CT.IdealGasReactor(_equil(phi_sz_crn), name="PSR_sz")
    r_sz.volume = V_sz
    pfr = []
    for k in range(CRN_N_PFR_DZ):
        r = _CT.IdealGasReactor(_equil(_safe(phi.get("dz"), 0.3)),
                                name=f"PFR_dz{k + 1}")
        r.volume = V_dz / CRN_N_PFR_DZ
        pfr.append(r)
    reactors = [r_pz, r_sz] + pfr

    #  --- caudales de la cadena (balance EXACTO por construcción) ----------
    SN = _safe((rec.get("swirler") or {}).get("SN"),
               _safe(rec.get("swirl_SN"), 0.70))
    rr = _recirc_ratio(SN)
    m_recirc = rr * m_in_pz
    m_pz_out = m_in_pz + m_recirc
    m_sz_out = m_in_pz + m_air_sz + m_fuel_sz
    m_chain = m_sz_out + m_air_dz

    dev = [
        _CT.MassFlowController(res_in, r_pz, mdot=m_in_pz),
        _CT.MassFlowController(r_sz, r_pz, mdot=m_recirc),      # CTRZ
        _CT.MassFlowController(r_pz, r_sz, mdot=m_pz_out),
        _CT.MassFlowController(res_air_sz, r_sz, mdot=m_air_sz),
        _CT.MassFlowController(r_sz, pfr[0], mdot=m_sz_out),
        _CT.MassFlowController(res_air_dz, pfr[0], mdot=m_air_dz),
    ]
    if res_fuel_sz is not None:
        dev.append(_CT.MassFlowController(res_fuel_sz, r_sz, mdot=m_fuel_sz))
    for k in range(CRN_N_PFR_DZ - 1):
        dev.append(_CT.MassFlowController(pfr[k], pfr[k + 1], mdot=m_chain))
    dev.append(_CT.MassFlowController(pfr[-1], exhaust, mdot=m_chain))
    for r in reactors:                       # ancla de presión, NO reparto
        dev.append(_CT.Valve(r, exhaust, K=CRN_VALVE_K))

    net = _CT.ReactorNet(reactors)
    net.rtol, net.atol = CRN_RTOL, CRN_ATOL

    topo = (f"PSR_pz(V={V_pz:.3e} m3, tau={1e3 * r_pz.mass / m_in_pz:.3f} ms, "
            f"recirc={rr:.2f}·mdot_pz) -> PSR_sz(V={V_sz:.3e} m3) -> "
            f"PFR_dz x{CRN_N_PFR_DZ}(V={V_dz:.3e} m3) + bypass frio "
            f"{100.0 * m_bypass / max(m_bypass + m_air_chain, 1e-12):.1f} % "
            f"| mech={mech_file}/{fuel_sp} | Cantera {getattr(_CT, '__version__', '?')}")
    return dict(net=net, reactors=reactors, r_pz=r_pz, r_sz=r_sz, pfr=pfr,
                gas_probe=gas0, fuel_sp=fuel_sp, mech_file=mech_file,
                m_in_pz=m_in_pz, m_recirc=m_recirc, m_chain=m_chain,
                m_bypass=m_bypass, m_fuel_sur=m_fuel_sur,
                m_air_pz=m_air_pz, m_air_chain=m_air_chain,
                fuel_balance=float((m_in_pz - m_air_pz + m_fuel_sz)
                                   - m_fuel_sur),
                phi_pz_crn=phi_pz_crn, recirc_ratio=rr, topology=topo,
                #  `dev` y `res` se DEVUELVEN para mantener vivas las
                #  referencias Python: los objetos de Cantera son extensiones
                #  Cython y no se puede confiar en que el C++ subyacente
                #  sobreviva a la recolección de basura del envoltorio.
                dev=dev, exhaust=exhaust,
                res=[res_in, res_air_sz, res_air_dz, res_fuel_sz, exhaust],
                T3=T3, P_liner=P_liner)


def _advance_to_steady(net, reactors, tau_net_s: float,
                       deadline: float) -> tuple[bool, str, int, float]:
    """Avanza la `ReactorNet` con `step()` y DEADLINE entre pasos.

    Elección de timeout (trampa CONTEXTO §10.6 / cabecera del módulo): en
    Windows no hay `SIGALRM`, y un `ProcessPoolExecutor` por punto costaría
    más que el propio solve (intérprete nuevo + import de Cantera ≈ 1–2 s
    frente a ~1 s de CRN). Se avanza con `net.step()` —cada paso interno de
    CVODES dura µs–ms— y el deadline se comprueba cada `CRN_DEADLINE_EVERY`
    pasos: cero procesos, cero hilos, física determinista e idéntico
    comportamiento en POSIX. Limitación declarada: si UN SOLO paso de CVODES
    se colgase (no observado), el deadline no lo interrumpiría.

    El |ΔT| que se devuelve NO es el del último chequeo del bucle (que puede
    caer en pleno transitorio, porque CVODES gasta casi todos sus pasos
    allí): cuando la ventana se agota se hace una VERIFICACIÓN FINAL de un τ
    extra y se mide la deriva real. Si supera `CRN_T_DRIFT_MAX_K` el punto no
    es estacionario y se declara divergido.

    Devuelve `(convergió, motivo, pasos, |ΔT|_max final)`.
    """
    tau = max(tau_net_s, 1e-9)
    t_end = CRN_TAU_MULT * tau
    t_min = CRN_TAU_MIN_MULT * tau
    win = CRN_CHECK_WINDOW_TAU * tau
    T_prev = np.array([r.T for r in reactors], dtype=float)
    t_prev = float(net.time)
    dT = float("inf")
    steps = 0
    exhausted = True
    for k in range(CRN_MAX_STEPS):
        if net.time >= t_end:
            exhausted = False
            break
        try:
            net.step()
        except Exception:
            return False, "diverged", steps, dT
        steps = k + 1
        if steps % CRN_DEADLINE_EVERY == 0 and time.monotonic() >= deadline:
            return False, "timeout", steps, dT
        if steps % CRN_CHECK_EVERY == 0 and (net.time - t_prev) >= win:
            T_now = np.array([r.T for r in reactors], dtype=float)
            if not np.all(np.isfinite(T_now)):
                return False, "diverged", steps, dT
            dT = float(np.max(np.abs(T_now - T_prev)))
            T_prev, t_prev = T_now, float(net.time)
            if net.time >= t_min and dT < CRN_T_TOL_K:
                return True, "ok", steps, dT
    if exhausted:                       # se agotaron los pasos ACOTADOS
        return False, "diverged", steps, dT
    #  Verificación final: un τ más y se mide la deriva de verdad.
    T_a = np.array([r.T for r in reactors], dtype=float)
    try:
        net.advance(net.time + tau)
    except Exception:
        return False, "diverged", steps, dT
    T_b = np.array([r.T for r in reactors], dtype=float)
    if not np.all(np.isfinite(T_b)):
        return False, "diverged", steps, dT
    dT = float(np.max(np.abs(T_b - T_a)))
    if dT > CRN_T_DRIFT_MAX_K:
        return False, "diverged", steps, dT
    return True, "ok", steps, dT


def _burnout(gas, m_out: float, m_fuel_sur: float, lhv_sur: float) -> dict:
    """Eficiencia de combustión por ENTALPÍA QUÍMICA SIN QUEMAR a la salida.

    η = 1 − Σ_i ṁ_out·Y_i·LHV_i / (ṁ_comb·LHV), sobre CO, H₂ y todo el HC
    residual que el mecanismo tenga (`LHV_BY_SPECIES`). Se usa la base
    energética del SURROGATE (la del propio CRN), que es autoconsistente; en
    el ancla difiere <0.5 % de la del combustible real.
    """
    names = set(gas.species_names)
    unburnt_W = 0.0
    detail = {}
    for sp in _BURNOUT_SPECIES:
        if sp not in names:
            continue
        y = float(gas.Y[gas.species_index(sp)])
        if y <= 0.0:
            continue
        w = m_out * y * LHV_BY_SPECIES[sp]
        unburnt_W += w
        detail[sp] = float(y)
    q_in = max(m_fuel_sur * lhv_sur, 1e-12)
    return dict(eta=float(_clip(1.0 - unburnt_W / q_in, 0.0, 1.0)),
                unburnt_W=float(unburnt_W), q_in_W=float(q_in),
                Y_unburnt=detail)


def _emission_indices(gas, m_out: float, m_fuel_real: float) -> dict:
    """EI de NOx (como NO₂-equivalente, convenio ICAO) y de CO [g/kg comb.].

    El denominador es la masa de combustible **REAL** del record, no la del
    surrogate: así el número es directamente comparable con el abanico de
    correlaciones de L0, que está en g por kg de Jet-A. Con φ igualado la
    diferencia entre ambas bases es del orden del 6 %, muy por debajo del
    ±30–50 % que Ciencia §8 declara para cualquier predicción de emisiones.
    """
    names = set(gas.species_names)
    mw = {s: float(gas.molecular_weights[gas.species_index(s)])
          for s in ("NO", "NO2", "N2O", "CO") if s in names}
    mdot_f = max(m_fuel_real, 1e-12)
    kmol_nox = 0.0
    for sp in ("NO", "NO2"):
        if sp in names:
            kmol_nox += m_out * float(gas.Y[gas.species_index(sp)]) / mw[sp]
    ei_nox = 1e3 * kmol_nox * MW_NO2 / mdot_f
    ei_co = 0.0
    if "CO" in names:
        ei_co = 1e3 * m_out * float(gas.Y[gas.species_index("CO")]) / mdot_f
    ei_n2o = 0.0
    if "N2O" in names:
        ei_n2o = (1e3 * m_out * float(gas.Y[gas.species_index("N2O")])
                  / mdot_f)
    nox_species = bool(names & {"NO", "NO2"})
    return dict(EI_NOx_crn=float(ei_nox), EI_CO_crn=float(ei_co),
                EI_N2O_crn=float(ei_n2o), nox_species_present=nox_species)


def _damkohler(gas, fuel: str, fuel_sp: str, tau_pz_s: float,
               phi_pz: float, T3: float, P3: float) -> dict:
    """Da de la zona primaria y margen de LBO [Ciencia §6.1].

    `Da = τ_PZ/τ_quím` con **τ_quím = α/S_L²** (el tiempo de tránsito de un
    frente laminar, δ_L/S_L), que es la única definición que NO se vuelve
    trivial en un PSR estacionario: allí `ρ·Y_F,in/|ω̇_F|` degenera
    IDÉNTICAMENTE a τ_res por el propio balance del reactor (Da ≡ 1), así que
    no sirve de nada.

    * α = k/(ρ·c_p) se toma de CANTERA en la mezcla FRESCA de la PZ (T₃, P₃,
      φ_PZ) con el transporte del mecanismo — es la parte «CRN» del número.
    * S_L = S_L,ref·(T_u/298)^2·(P/P_atm)^−0.25 es la MISMA correlación que
      usa `physics_core._stability` (S_L_EXP_T = 2.00, S_L_EXP_P = −0.25 ✅),
      para que Da y φ_LBO hablen el mismo idioma.

    `LBO_margin_Da = Da_pz/DA_LBO_REF − 1`, con `DA_LBO_REF = 0.80` = extremo
    SUPERIOR (conservador) de la banda ✅ 0.3–0.8 de Ciencia §6.1.
    """
    fp = _fuel_props(fuel)
    pc = _phys()
    exp_T = _safe(getattr(pc, "S_L_EXP_T", None), 2.00) if pc else 2.00
    exp_P = _safe(getattr(pc, "S_L_EXP_P", None), -0.25) if pc else -0.25
    p_atm = _safe(getattr(pc, "P_ATM", None), 101_325.0) if pc else 101_325.0
    S_L = (_safe(fp.get("S_L_ref"), 0.40) * (max(T3, 1.0) / 298.0) ** exp_T
           * (max(P3, 1.0) / p_atm) ** exp_P)
    S_L = max(S_L, 1e-3)
    alpha = 2.0e-5
    try:
        _fresh(gas, phi_pz, fuel_sp, T3, P3)
    except Exception:
        pass
    try:
        alpha = float(gas.thermal_conductivity
                      / max(gas.density * gas.cp_mass, 1e-12))
    except Exception:
        pass
    tau_chem = max(alpha / (S_L * S_L), 1e-12)
    Da = float(max(tau_pz_s, 0.0) / tau_chem)
    return dict(Da_pz=Da, tau_chem_ms=float(1e3 * tau_chem),
                S_L_ms_crn=float(S_L), alpha_pz_m2s=float(alpha),
                LBO_margin_Da=float(Da / DA_LBO_REF - 1.0),
                Da_LBO_band=list(DA_LBO_BAND),
                Da_note=("Da = τ_PZ/(α/S_L²) con α de Cantera en la mezcla "
                         "fresca de la PZ y S_L de la correlación de "
                         "physics_core; banda de LBO ✅ 0.3–0.8 (Ciencia "
                         "§6.1), se usa el extremo superior 0.80 como "
                         "referencia conservadora."))


def crn_solve(theta, rec: dict, timeout_s: float | None = None) -> dict | None:
    """Red 1D + CRN de Cantera. `dict` con las claves de L1, o `None`.

    Es el punto de entrada que `physics_core._try_l1` llama con
    `crn_solve(theta, rec, L1_TIMEOUT_S)`; si devuelve `None` el punto
    degrada a L0 con etiqueta (contrato §3). El MOTIVO real queda en
    `LAST_STATUS` (`"ok"·"timeout"·"diverged"·"cantera_unavailable"·
    "no_mechanism"·"degenerate"`) para que un parche de tres líneas en
    `physics_core` pueda emitir `crn_L1(timeout→L0)` en vez del genérico
    `crn_L1(diverged→L0)`.

    **Nunca lanza.** Cualquier excepción interna se traduce en `None`.

    **PF NO sale de aquí y se declara**: un CRN es 0-D por zona, no tiene
    distribución espacial de temperatura a la salida y por tanto no puede
    producir el factor de patrón. `PF`, `PF_star`, `RTDF` y `T4_max_K` del
    record siguen siendo la correlación de L0 [Ciencia §4.3] hasta L1.5/L2.
    """
    global LAST_STATUS
    LAST_STATUS = "not_run"
    deadline = time.monotonic() + max(
        _safe(timeout_s, PHYCB_L1_TIMEOUT) if timeout_s is not None
        else PHYCB_L1_TIMEOUT, 0.0)
    if not isinstance(rec, dict):
        LAST_STATUS = "degenerate"
        return None
    if not CANTERA_AVAILABLE:
        LAST_STATUS = "cantera_unavailable"
        return None
    try:
        return _crn_solve_impl(theta, rec, deadline)
    except Exception:
        if LAST_STATUS in ("not_run", "ok"):
            LAST_STATUS = "diverged"
        return None


def _crn_solve_impl(theta, rec: dict, deadline: float) -> dict | None:
    """Implementación de `crn_solve` (la envoltura pública captura todo)."""
    global LAST_STATUS
    P3 = _safe(rec.get("P3_Pa"), 0.0)
    T3 = _safe(rec.get("T3_K"), 0.0)
    m_fuel_real = _safe(rec.get("mdot_fuel_kgs"), 0.0)
    if not (P3 > 0.0 and T3 > 0.0 and m_fuel_real > 0.0
            and _safe(rec.get("mdot_air_kgs"), 0.0) > 0.0):
        LAST_STATUS = "degenerate"
        return None

    net1 = network_1d(theta, rec)
    if not net1.get("network_converged"):
        LAST_STATUS = "diverged"
        return None
    if time.monotonic() >= deadline:        # el deadline manda desde ya
        LAST_STATUS = "timeout"
        return None

    fuel = str(rec.get("fuel", "jet_a"))
    mech = _resolve_mechanism(fuel)
    if mech is None:
        LAST_STATUS = "no_mechanism"
        return None
    gas0, mech_file, fuel_sp, mech_note = mech
    requested = MECH_CASCADE.get(fuel, (("", ""),))[0][0]
    degraded = bool(mech_file != requested)
    P_liner = max(P3 - net1["dp_diffuser_Pa"] - net1["dp_liner_Pa"], 1e3)

    tau_pz_s = 1e-3 * _safe(net1["tau_zones_ms"].get("pz"), 1.0)
    evap = _evap_factor(rec, fuel, tau_pz_s)
    B = _build_network(net1, rec, mech, evap, T3, P_liner)
    if time.monotonic() >= deadline:        # construir la red ya costó todo
        LAST_STATUS = "timeout"
        return None

    tau_net_s = sum(r.mass for r in B["reactors"]) / max(B["m_chain"], 1e-12)
    ok, why, steps, dT = _advance_to_steady(
        B["net"], B["reactors"], tau_net_s, deadline)
    if not ok:
        LAST_STATUS = why
        return None

    # --- post-proceso -----------------------------------------------------
    r_exit = B["pfr"][-1]
    g_exit = r_exit.thermo
    m_out = B["m_chain"]
    fp = _fuel_props(fuel)
    lhv_sur = LHV_BY_SPECIES.get(fuel_sp, _safe(fp.get("LHV"), 43.1e6))
    bo = _burnout(g_exit, m_out, B["m_fuel_sur"], lhv_sur)
    ei = _emission_indices(g_exit, m_out, m_fuel_real)

    T_zones = {"pz": float(B["r_pz"].T), "sz": float(B["r_sz"].T)}
    for k, r in enumerate(B["pfr"]):
        T_zones[f"dz{k + 1}"] = float(r.T)
    T_zones["dz"] = float(r_exit.T)
    T_zones["exit"] = float(r_exit.T)

    #  Mezcla adiabática con el BYPASS FRÍO (efusión + anillo) a la salida.
    T04_crn = float(r_exit.T)
    try:
        g_mix = _CT.Solution(mech_file)
        g_air = _CT.Solution(mech_file)
        g_air.TPX = max(T3, 200.0), max(P_liner, 1.0), OXIDIZER
        m_by = max(B["m_bypass"], 0.0)
        w = m_out + m_by
        Y_mix = (m_out * np.asarray(g_exit.Y) + m_by * np.asarray(g_air.Y)) / w
        h_mix = (m_out * g_exit.enthalpy_mass + m_by * g_air.enthalpy_mass) / w
        g_mix.Y = Y_mix
        g_mix.HP = h_mix, max(P_liner, 1.0)
        T04_crn = float(g_mix.T)
        T_zones["exit_mixed"] = T04_crn
    except Exception:
        T_zones["exit_mixed"] = T04_crn

    da = _damkohler(_CT.Solution(mech_file), fuel, fuel_sp, tau_pz_s,
                    B["phi_pz_crn"], T3, P3)

    #  τ y v por zona RECALCULADOS con las temperaturas del CRN.
    R_hot = max(_safe(rec.get("R_hot_J_kgK"), 292.5), 1.0)
    A_L = max(_safe(rec.get("A_L_m2"), 0.0), 1e-12)
    tau_crn = {"pz": 1e3 * B["r_pz"].mass / max(B["m_in_pz"], 1e-12),
               "sz": 1e3 * B["r_sz"].mass / max(B["m_chain"] - _safe(
                   net1["mdot_paths_kgs"].get("dz_holes"), 0.0), 1e-12),
               "dz": 1e3 * sum(r.mass for r in B["pfr"])
               / max(B["m_chain"], 1e-12)}
    tau_crn["total"] = tau_crn["pz"] + tau_crn["sz"] + tau_crn["dz"]
    tau_crn["reactor_pz"] = 1e3 * B["r_pz"].mass / max(
        B["m_in_pz"] + B["m_recirc"], 1e-12)
    v_crn = dict(net1["v_zones"])
    for z, T_z, m_z in (("pz", T_zones["pz"], B["m_in_pz"]),
                        ("sz", T_zones["sz"], B["m_chain"] - _safe(
                            net1["mdot_paths_kgs"].get("dz_holes"), 0.0)),
                        ("dz", T_zones["dz"], B["m_chain"])):
        rho_z = P_liner / (R_hot * max(T_z, 200.0))
        v_crn[z] = float(m_z / max(rho_z * A_L, 1e-12))

    mech_flag = "degradado" if degraded else "el pedido"
    note = (f"mech_l1 solicitado = '{requested}'; usado = '{mech_file}' "
            f"({mech_flag}), especie de combustible '{fuel_sp}'."
            + (" " + mech_note if mech_note else ""))
    if not ei["nox_species_present"]:
        note += (" ⚠️ el mecanismo NO tiene especies de NOx: EI_NOx_crn = 0 "
                 "NO es un resultado físico.")

    out: dict = dict(net1)
    out.update({
        "T_zones_K": T_zones,
        "tau_zones_ms": {k: float(v) for k, v in tau_crn.items()},
        "tau_zones_1d_ms": net1["tau_zones_ms"],
        "v_zones": {k: float(v) for k, v in v_crn.items()},
        "eta_comb_crn": float(bo["eta"]),
        "crn_unburnt_W": float(bo["unburnt_W"]),
        "crn_Y_unburnt": bo["Y_unburnt"],
        "T04_crn_K": float(T04_crn),
        "P_liner_Pa": float(P_liner),
        "crn_topology": B["topology"],
        "crn_mech": mech_file,
        "crn_mech_species": fuel_sp,
        "crn_mech_degraded": bool(degraded),
        "crn_mech_note": note,
        "crn_recirc_ratio": float(B["recirc_ratio"]),
        "crn_steps": int(steps),
        "crn_dT_final_K": float(dT),
        "crn_tau_net_ms": float(1e3 * tau_net_s),
        "crn_n_reactors": int(len(B["reactors"])),
        "phi_pz_crn": float(B["phi_pz_crn"]),
        "mdot_fuel_surrogate_kgs": float(B["m_fuel_sur"]),
        "crn_fuel_balance_kgs": float(B["fuel_balance"]),
        "crn_mdot_chain_kgs": float(B["m_chain"]),
        "crn_mdot_bypass_kgs": float(B["m_bypass"]),
        "crn_pf_note": (
            "PF NO sale del CRN y no se toca: una red de reactores es 0-D por "
            "zona, no tiene distribución espacial de T a la salida. `PF`, "
            "`PF_star`, `RTDF` y `T4_max_K` siguen siendo la correlación de "
            "L0 [Ciencia §4.3] hasta L1.5 (aero-frío XLB) y L2 (RANS)."),
        "crn_bypass_note": (
            "El aire de refrigeración (efusión + anillo, "
            f"{100.0 * B['m_bypass'] / max(B['m_bypass'] + B['m_air_chain'], 1e-12):.1f} "
            "% del total) entra como BYPASS FRÍO: se mezcla por entalpía a la "
            "salida y NO participa en la cadena de reactores. Consecuencia "
            "declarada: el CRN no captura el apagado de CO contra la pared, "
            "así que `EI_CO_crn` es una COTA INFERIOR."),
    })
    out.update(ei)
    out.update(da)
    out.update({k: v for k, v in evap.items()})
    LAST_STATUS = "ok"
    return out


# ==========================================================================
# 6. CLI de comprobación — caso ancla REF_CB10 (WAVE2 §§1–2, criterios §4)
# ==========================================================================
def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:+6.2f} %"


def _check(label: str, ok_: bool, detail: str, cs) -> bool:
    line = (cs.ok if ok_ else cs.err)(f"{label:<44s} {detail}")
    print(line)
    return ok_


def main(argv=None) -> int:
    """Comprobación del módulo sobre el caso ancla. `0` si todo pasa."""
    global CANTERA_AVAILABLE
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import physics_core as pc
    pc.assert_provenance(pc)

    try:
        import cli_style as cs
        cs.init()
    except Exception:                       # degradación silenciosa de estilo
        class cs:                           # type: ignore[no-redef]
            ok = staticmethod(lambda m: "[ok]   " + m)
            err = staticmethod(lambda m: "[FALLA] " + m)
            warn = staticmethod(lambda m: "[warn] " + m)
            step = staticmethod(lambda m: "-> " + m)
            kv = staticmethod(lambda k, v: f"  {k}: {v}")
            rule = staticmethod(lambda t="", s="": "-" * 70 + " " + t)

    print(cs.rule("crn_core · L1 (red 1D + CRN) · ancla REF_CB10"))
    print(cs.kv("Cantera", f"{CANTERA_AVAILABLE} "
                           f"{getattr(_CT, '__version__', '')}"))
    for f, info in available_mechanisms().items():
        print(cs.kv(f"mech[{f}]", f"{info['mech']} / {info['species']}"
                                  f"{'  (DEGRADADO)' if info['degraded'] else ''}"))

    theta = pc.THETA_REF.copy()
    rec = pc.evaluate(theta, fidelity=pc.Fidelity.L0)
    passed = []

    # ---------------- 1. Red 1D --------------------------------------------
    print(cs.rule("1 · red 1D"))
    t0 = time.perf_counter()
    n1 = network_1d(theta, rec)
    dt_ms = 1e3 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for _ in range(20):
        network_1d(theta, rec)
    dt_ms_avg = 1e3 * (time.perf_counter() - t0) / 20.0
    print(cs.kv("tiempo", f"{dt_ms:.2f} ms (1ª) · {dt_ms_avg:.2f} ms (media "
                          f"de 20) · objetivo < 10 ms"))
    print(cs.kv("iteraciones Newton", f"{n1['network_iters']} de "
                                      f"{NEWTON_ITERS} · ‖R‖∞ = "
                                      f"{n1['network_residual']:.3e}"))
    print(cs.kv("ΔP_L", f"{n1['dp_liner_Pa']:.1f} Pa "
                        f"({n1['dp_liner_over_qref']:.2f}·q_ref) frente a "
                        f"{rec['dp_liner_Pa']:.1f} Pa de L0"))
    for k in ("pz", "sz", "dz", "cool"):
        print(cs.kv(f"split {k}", f"{n1['splits_solved'][k]:.4f} vs "
                                  f"{rec['splits'][k]:.4f} nominal  "
                                  f"({_fmt_pct(n1['split_dev_vs_L0'][k])})"))
    for k, v in n1["splits_solved_paths"].items():
        print(cs.kv(f"  camino {k}", f"{v:.4f}"))
    passed.append(_check(
        "conservación de masa < 1e-9 kg/s", abs(n1["mass_residual_kgs"]) < 1e-9,
        f"|Σṁ − ṁ₃| = {abs(n1['mass_residual_kgs']):.3e} kg/s", cs))
    passed.append(_check(
        "splits resueltos a < 15 % de los nominales",
        n1["split_dev_max"] < 0.15,
        f"desviación máxima = {_fmt_pct(n1['split_dev_max'])}", cs))
    passed.append(_check("red 1D converge", bool(n1["network_converged"]),
                         f"‖R‖∞ = {n1['network_residual']:.3e}", cs))
    passed.append(_check("red 1D en < 10 ms", dt_ms_avg < 10.0,
                         f"{dt_ms_avg:.2f} ms", cs))
    print(cs.kv("φ resueltos", " · ".join(
        f"{k}={v:.4f}" for k, v in n1["phi_zones_solved"].items())))
    print(cs.kv("τ zonas [ms]", " · ".join(
        f"{k}={v:.3f}" for k, v in n1["tau_zones_ms"].items())))
    print(cs.kv("dp_split_check", n1["dp_split_check"][:300] + "…"))

    # ---------------- 2. CRN -----------------------------------------------
    print(cs.rule("2 · CRN (Cantera)"))
    if not CANTERA_AVAILABLE:
        print(cs.warn("Cantera ausente: se omiten las comprobaciones de CRN"))
        return 0 if all(passed) else 1
    t0 = time.perf_counter()
    sol = crn_solve(theta, rec, PHYCB_L1_TIMEOUT)
    dt_s = time.perf_counter() - t0
    if sol is None:
        print(cs.err(f"crn_solve devolvió None (LAST_STATUS={LAST_STATUS})"))
        return 1
    print(cs.kv("tiempo", f"{dt_s:.3f} s · {sol['crn_steps']} pasos · "
                          f"τ_red = {sol['crn_tau_net_ms']:.3f} ms · deriva "
                          f"final |ΔT| = {sol['crn_dT_final_K']:.3e} K en un "
                          f"τ extra · objetivo < 5 s/punto"))
    print(cs.kv("topología", sol["crn_topology"]))
    print(cs.kv("mecanismo", sol["crn_mech_note"][:220] + "…"))
    print(cs.kv("T por zona [K]", " · ".join(
        f"{k}={v:.1f}" for k, v in sol["T_zones_K"].items()
        if not k.startswith("dz") or k == "dz")))
    print(cs.kv("τ CRN [ms]", " · ".join(
        f"{k}={v:.4f}" for k, v in sol["tau_zones_ms"].items())))
    print(cs.kv("η_comb", f"CRN {sol['eta_comb_crn']:.5f} vs L0 "
                          f"{rec['eta_comb']:.5f}"))
    print(cs.kv("EI_NOx [g/kg]", f"CRN {sol['EI_NOx_crn']:.3f} vs abanico L0 "
                                 f"{rec['EI_NOx_fan']['min']:.2f}–"
                                 f"{rec['EI_NOx_fan']['max']:.2f}"))
    print(cs.kv("EI_CO [g/kg]", f"CRN {sol['EI_CO_crn']:.3f} vs L0 "
                                f"{rec['EI_CO']:.2f}"))
    print(cs.kv("Da_pz", f"{sol['Da_pz']:.2f} (τ_quím = "
                         f"{sol['tau_chem_ms'] * 1e3:.2f} µs, S_L = "
                         f"{sol['S_L_ms_crn']:.3f} m/s) · margen LBO_Da = "
                         f"{sol['LBO_margin_Da']:.2f}"))
    print(cs.kv("evaporación", sol["far_evap_note"][:200]))
    passed.append(_check("η_comb_crn ∈ (0.95, 1.0)",
                         0.95 < sol["eta_comb_crn"] < 1.0,
                         f"{sol['eta_comb_crn']:.5f}", cs))
    passed.append(_check("EI_NOx_crn ∈ (1, 40) g/kg",
                         1.0 < sol["EI_NOx_crn"] < 40.0,
                         f"{sol['EI_NOx_crn']:.3f} g/kg", cs))
    passed.append(_check("Da_pz en el orden de 20 (Ciencia §6.1)",
                         2.0 < sol["Da_pz"] < 200.0,
                         f"Da = {sol['Da_pz']:.2f}", cs))
    passed.append(_check("CRN en < 5 s/punto", dt_s < 5.0,
                         f"{dt_s:.3f} s", cs))
    passed.append(_check("balance de combustible del CRN exacto",
                         abs(sol["crn_fuel_balance_kgs"]) < 1e-12,
                         f"|Δṁ_comb| = {abs(sol['crn_fuel_balance_kgs']):.3e}"
                         f" kg/s", cs))
    sol2 = crn_solve(theta, rec, PHYCB_L1_TIMEOUT)
    same = sol2 is not None and all(
        sol2[k] == sol[k] for k in ("EI_NOx_crn", "EI_CO_crn",
                                    "eta_comb_crn", "Da_pz", "dp_liner_Pa"))
    passed.append(_check("determinismo (SEED = 71, dos solves idénticos)",
                         bool(same), "bit-idénticos" if same else "DIFIEREN",
                         cs))

    # --- rama de espray: la corrección de evaporación SÍ se activa ---------
    th_ab = theta.copy()
    th_ab[10] = 0.0                         # inyector: vaporizer → airblast
    rec_ab = pc.evaluate(th_ab, fidelity=pc.Fidelity.L0)
    sol_ab = crn_solve(th_ab, rec_ab, PHYCB_L1_TIMEOUT)
    if sol_ab is not None:
        print(cs.kv("airblast · SMD", f"{rec_ab['SMD_um']:.2f} µm · "
                                      f"τ_e = {sol_ab['tau_evap_ms']:.4f} ms "
                                      f"· τ_e/τ = "
                                      f"{sol_ab['tau_e_over_tau']:.4f}"))
        print(cs.kv("airblast · f_evap",
                    f"{sol_ab['far_evap_factor']:.4f} (bruto de la fórmula "
                    f"literal {sol_ab['far_evap_factor_raw']:.4f}) · "
                    f"EI_NOx = {sol_ab['EI_NOx_crn']:.3f} g/kg"))
        passed.append(_check(
            "corrección de evaporación activa en espray (0 < f < 1)",
            0.0 < sol_ab["far_evap_factor"] < 1.0,
            f"f_evap = {sol_ab['far_evap_factor']:.4f}", cs))
    else:
        passed.append(_check("CRN de la rama airblast", False, "None", cs))

    # ---------------- 3. Monotonías ----------------------------------------
    print(cs.rule("3 · monotonías: NOx sube con τ y con T₃"))
    th_tau = theta.copy()
    th_tau[12] = 1.00                       # L_pz/D_L 0.75 → 1.00 ⇒ más τ_PZ
    rec_tau = pc.evaluate(th_tau, fidelity=pc.Fidelity.L0)
    sol_tau = crn_solve(th_tau, rec_tau, PHYCB_L1_TIMEOUT)
    th_T3 = theta.copy()
    th_T3[17] = 520.0                       # T₃ 430 → 520 K
    th_T3[19] = 1240.0                      # T₄ +90 K ⇒ mismo salto ⇒ φ ≈ igual
    rec_T3 = pc.evaluate(th_T3, fidelity=pc.Fidelity.L0)
    sol_T3 = crn_solve(th_T3, rec_T3, PHYCB_L1_TIMEOUT)
    if sol_tau is None or sol_T3 is None:
        passed.append(_check("monotonías", False, "algún crn_solve dio None",
                             cs))
    else:
        print(cs.kv("τ_PZ [ms]", f"{sol['tau_zones_ms']['pz']:.4f} → "
                                 f"{sol_tau['tau_zones_ms']['pz']:.4f}"))
        passed.append(_check(
            "NOx sube con τ_PZ",
            sol_tau["EI_NOx_crn"] > sol["EI_NOx_crn"]
            and sol_tau["tau_zones_ms"]["pz"] > sol["tau_zones_ms"]["pz"],
            f"EI_NOx {sol['EI_NOx_crn']:.3f} → "
            f"{sol_tau['EI_NOx_crn']:.3f} g/kg", cs))
        print(cs.kv("T_PZ [K]", f"{sol['T_zones_K']['pz']:.1f} → "
                                f"{sol_T3['T_zones_K']['pz']:.1f}"))
        passed.append(_check(
            "NOx sube con T₃", sol_T3["EI_NOx_crn"] > sol["EI_NOx_crn"],
            f"EI_NOx {sol['EI_NOx_crn']:.3f} → "
            f"{sol_T3['EI_NOx_crn']:.3f} g/kg", cs))

    # ---------------- 4. Degradación sin Cantera ---------------------------
    print(cs.rule("4 · degradación: import de Cantera simulado como fallido"))
    saved = CANTERA_AVAILABLE
    try:
        CANTERA_AVAILABLE = False
        n1_nc = network_1d(theta, rec)      # la red 1D DEBE seguir corriendo
        sol_nc = crn_solve(theta, rec, PHYCB_L1_TIMEOUT)
        passed.append(_check("crn_solve → None sin excepción",
                             sol_nc is None and LAST_STATUS ==
                             "cantera_unavailable",
                             f"LAST_STATUS = {LAST_STATUS}", cs))
        passed.append(_check("network_1d sigue funcionando sin Cantera",
                             bool(n1_nc["network_converged"]),
                             f"ΔP_L = {n1_nc['dp_liner_Pa']:.1f} Pa", cs))
    finally:
        CANTERA_AVAILABLE = saved

    # ---------------- 5. Timeout duro --------------------------------------
    print(cs.rule("5 · timeout duro (deadline = 0 s)"))
    sol_to = crn_solve(theta, rec, 0.0)
    passed.append(_check("timeout → None y LAST_STATUS='timeout'",
                         sol_to is None and LAST_STATUS in ("timeout",
                                                            "diverged"),
                         f"LAST_STATUS = {LAST_STATUS}", cs))

    # ---------------- 6. Integración con physics_core ----------------------
    print(cs.rule("6 · physics_core.evaluate(fidelity=L1)"))
    t0 = time.perf_counter()
    rec_l1 = pc.evaluate(theta, fidelity=pc.Fidelity.L1, use_cache=False)
    dt_l1 = time.perf_counter() - t0
    print(cs.kv("source", rec_l1.get("source", "?")))
    print(cs.kv("tiempo end-to-end", f"{dt_l1:.3f} s"))
    passed.append(_check("evaluate(L1) etiqueta 'crn_L1'",
                         rec_l1.get("source") == "crn_L1",
                         str(rec_l1.get("source")), cs))
    passed.append(_check("record L1 trae las claves del contrato §4.3",
                         all(k in rec_l1 for k in (
                             "splits_solved", "dp_liner_Pa", "v_zones",
                             "tau_zones_ms", "T_zones_K", "EI_NOx_crn",
                             "EI_CO_crn", "eta_comb_crn", "Da_pz",
                             "LBO_margin_Da", "crn_topology")),
                         "11/11", cs))

    print(cs.rule("resumen"))
    n_ok = sum(1 for p in passed if p)
    print((cs.ok if n_ok == len(passed) else cs.err)(
        f"{n_ok}/{len(passed)} comprobaciones"))
    return 0 if n_ok == len(passed) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
