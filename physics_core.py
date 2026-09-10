"""
QUASAR Phy-CB · physics_core.py
===============================
Capa 1 del Computational Engineering Model: NÚCLEO FÍSICO MULTI-FIDELIDAD
para CÁMARAS DE COMBUSTIÓN Brayton imprimibles en 3D (hermano de Phy-AC,
compresores axiales, y de Phy-CC, centrífugos).

Contrato único (idéntico al de los hermanos, `IMPLEMENTATION_CONTRACT` §2):

    evaluate(theta, fidelity=Fidelity.L0, use_cache=True,
             calibrate=True) -> dict (record `phycb-record-1`)

Escalera de fidelidad
---------------------
  L0   Correlaciones de Lefebvre + equilibrio químico (~1 ms/punto). ESTE
       ARCHIVO. Dimensionado A_ref = max(criterio aerodinámico, criterio
       químico θ), reparto de aire, orificios y penetración de chorro,
       atomización, eficiencia descompuesta η=min(η_θ,η_evap,η_mix),
       estabilidad (LBO + encendido), térmico 1D del liner con efusión,
       factor de patrón y el abanico COMPLETO de correlaciones de emisiones.
       Devuelve el vector de restricciones `g <= 0` (12 aero/térmicas), la
       interfaz de la dominancia restringida de Deb en la capa de búsqueda.
  L1   Red 1D de flujo + CRN de Cantera (`crn_core.py`, WAVE2). Import
       PEREZOSO: mientras `crn_core` no exista, L1 degrada a L0 etiquetado.
  L1.5 Aero-frío LBM (XLB) sobre voxels → auto-topología de CRN (WAVE5-A).
  L2a  IB-RANS nativo sobre el SDF de PicoGK (WAVE6).
  L2b  Backend OpenFOAM por subproceso (WAVE5-B) — **FUERA DE ALCANCE**
       (decisión de proyecto 2026-07-31: la máquina objetivo es Windows y
       OpenFOAM exigiría WSL2/Docker). El valor del enum SE CONSERVA porque
       es ABI (viaja en `record["fidelity"]`, checkpoints y caché), pero
       `--fidelity L2B` degrada a L2a con
       `source="openfoam_L2B(out_of_scope→L2A)"`. Consecuencia declarada:
       Phy-CB NO predice T de metal cuantitativa en ningún nivel — era lo
       que L2b aportaba (CHT + radiación). Ver contrato §12.
  Hook de calibración `register_hifi_pair(theta, y_hifi)`: corrección afín
  por salida sobre L0/L1, alimentada por L2a o por banco de ensayo.

Otras capacidades (patrón de la familia)
----------------------------------------
  * Caché persistente por hash de θ+fidelidad (JSONL, **solo L1+**; con
    `use_cache=False` no lee NI escribe — trampa CONTEXTO §10.5).
  * `evaluate_batch(..., n_workers)` — paralelismo por procesos para L1+.
  * `physics_features(theta)` — embedding físico cerrado de 37 componentes
    para el surrogate: el ensemble aprende el RESIDUO sobre la física.

Reglas de la casa que este archivo respeta sin excepción
--------------------------------------------------------
  * SI interno; mm/deg/L solo en las claves de interfaz, con sufijo.
  * Todo coeficiente marcado ⚠️ [L·verificar] en `docs/Quasar_PhyCB_Science.md`
    se lee de `data/` vía `data_pipeline.load_table()`. Si la tabla falta se
    usa el valor provisional de la constante SCREAMING_CASE correspondiente y
    el aviso se registra en `PROVISIONAL_COEFFS` para que el reporte lo pinte.
  * `g` NUNCA lanza: entrada degenerada ⇒ record degenerado con `g` todo a
    `G_DEGENERATE = 10.0`, `feasible=False` y sufijo `(degenerate)` en
    `source`.
  * Bucles ACOTADOS (iteraciones fijas + test de convergencia), sin `while`
    abiertos.
  * Determinismo: `SEED = 71`, sin aleatoriedad ni relojes dentro de la
    física. Dos `evaluate` idénticos ⇒ records bit-idénticos.
  * Dependencias opcionales (Cantera) con import perezoso y degradación
    etiquetada; **solo NumPy es obligatorio**.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from enum import IntEnum

import numpy as np

warnings.filterwarnings("ignore")

# ==========================================================================
# 0.a Centinela de procedencia — TRAMPA REAL DE ESTA MÁQUINA
# ==========================================================================
#: Phy-AC está instalado como paquete EDITABLE en el entorno de desarrollo
#: (`__editable__.phyac-0.1.0.pth` en site-packages). Consecuencia verificada
#: el 2026-07-30: desde cualquier cwd que no sea el de Phy-CB, un
#: `import physics_core` resuelve al `physics_core.py` de **Phy-AC** (θ de 15
#: posiciones, física de compresor axial) SIN error — devolvería números
#: perfectamente plausibles y completamente equivocados.
#:
#: Por eso todo punto de entrada de Phy-CB (CLI, tests, bloques __main__)
#: hace `sys.path.insert(0, dirname(abspath(__file__)))` Y comprueba este
#: centinela con `assert_provenance()`. Un fallo ruidoso siempre es mejor
#: que un resultado silencioso equivocado.
PROJECT = "Phy-CB"
MODULE_SCHEMA = "phycb-physics-core-1"


def assert_provenance(*modules) -> None:
    """Verifica que los módulos hermanos importados son los de Phy-CB y no
    los de otro hermano de la familia instalado en site-packages.

    Sin argumentos comprueba solo este módulo. Lanza `ImportError` con un
    mensaje accionable — es la única excepción deliberada del núcleo (una
    dependencia mal resuelta no es un caso degradable: es un bug de entorno
    que debe parar la ejecución)."""
    here = os.path.dirname(os.path.abspath(__file__))
    mods = modules or (sys.modules[__name__],)
    for m in mods:
        got = os.path.dirname(os.path.abspath(getattr(m, "__file__", "") or ""))
        if os.path.normcase(got) != os.path.normcase(here):
            raise ImportError(
                f"módulo '{getattr(m, '__name__', m)}' resuelto en '{got}' en "
                f"lugar de '{here}'. Phy-AC está instalado como paquete "
                "editable y ensombrece los módulos de Phy-CB: añade "
                "sys.path.insert(0, os.path.dirname(os.path.abspath("
                "__file__))) al arrancar, o corre desde el directorio de "
                "Phy-CB.")
        if getattr(m, "PROJECT", "Phy-CB") != "Phy-CB":
            raise ImportError(
                f"módulo '{getattr(m, '__name__', m)}' no es de Phy-CB "
                f"(PROJECT={getattr(m, 'PROJECT')!r})")


# ==========================================================================
# 0. Constantes compartidas (IMPLEMENTATION_CONTRACT §6) — NO redefinir
# ==========================================================================
GAMMA, CP, RGAS = 1.4, 1005.0, 287.0     # aire frío (idéntico a los hermanos)
GAMMA_HOT, CP_HOT = 4.0 / 3.0, 1148.0    # gas caliente (CONTEXTO §4)
MU_AIR = 1.81e-5                          # Pa·s a ~293 K
SIGMA_SB = 5.670374419e-8                 # W/m²/K⁴ (CODATA)
P_ATM = 101_325.0                         # Pa
SEED = 71                                 # semilla de la familia (§10.8)

#: Eficiencia de combustión y pérdida del relleno de ciclo de Phy-AC
#: (CONTEXTO §4). `_far_from_T4` DEBE reproducir su `f` bit a bit para que la
#: integración de WAVE4 (`cycle_seam.hot_cycle`) sea coherente.
ETA_COMB_CYCLE = 0.995

#: Combustibles. `LHV_JET_A = 43.1e6` DEBE coincidir con el del ciclo.
#: Los campos `rho_l/sigma/mu_l` son de handbook a ~300 K (queroseno Jet-A);
#: `lambda_evap` es la constante de evaporación efectiva [m²/s] de Ciencia
#: §6.1/§8.3; `S_L_ref` la velocidad laminar de llama a 298 K, 1 atm, φ=1.
FUELS = {
    "jet_a": dict(
        LHV=43.1e6, HC_ratio=1.91, formula="C12H23",
        mech_l1="ker_skeletal", mech_l2="2S_KERO_BFER",
        ct_species="POSF10325", luminosity=1.7, liquid=True,
        far_stoich=0.0686834, rho_l=780.0, sigma=0.0270, mu_l=1.30e-3,
        lambda_evap=8.0e-7, S_L_ref=0.40, T_boil_K=440.0,
        mw_products=28.6),
    "propane": dict(
        LHV=46.35e6, HC_ratio=2.667, formula="C3H8",
        mech_l1="gri30", mech_l2="WD2",
        ct_species="C3H8", luminosity=1.0, liquid=False,
        far_stoich=0.0642180, rho_l=493.0, sigma=0.0070, mu_l=0.11e-3,
        lambda_evap=1.2e-6, S_L_ref=0.43, T_boil_K=231.0,
        mw_products=28.5),
    "h2": dict(
        LHV=120.0e6, HC_ratio=math.inf, formula="H2",
        mech_l1="sandiego_h2_9sp", mech_l2="sandiego_h2_9sp",
        ct_species="H2", luminosity=1.0, liquid=False,
        far_stoich=0.0293600, rho_l=70.8, sigma=0.0019, mu_l=0.013e-3,
        lambda_evap=4.0e-6, S_L_ref=2.10, T_boil_K=20.3,
        mw_products=24.0),
}
LHV_JET_A = FUELS["jet_a"]["LHV"]

#: Nombres de las tres compuertas (gates) del contrato §1. El ÍNDICE es el
#: valor entero de la gate: `ARCH_NAMES[int(round(theta[0]))]`.
ARCH_NAMES = ("can", "annular", "reverse_annular")
INJECTOR_NAMES = ("airblast", "simplex", "vaporizer", "micromix")
FUEL_NAMES = tuple(FUELS)          # ("jet_a", "propane", "h2") — orden fijo

G_DEGENERATE = 10.0     # violación grande y continua (contrato §5.1)


class Fidelity(IntEnum):
    """Escalera de fidelidad del contrato §2 (los enteros son ABI)."""
    L0 = 0      # correlaciones + equilibrio            (WAVE1 — este archivo)
    L1 = 1      # red 1D + CRN Cantera                  (WAVE2, crn_core.py)
    L15 = 2     # aero-frío XLB → auto-CRN              (WAVE5-A)
    L2A = 3     # IB-RANS nativo                        (WAVE6)
    L2B = 4     # backend OpenFOAM (WAVE5-B) — FUERA DE ALCANCE, contrato
    #             §12. Se conserva el valor: los enteros son ABI y ya viajan
    #             en records, checkpoints y phys_cache.jsonl.


# ==========================================================================
# 1. Espacio de diseño θ (20-D) — IMPLEMENTATION_CONTRACT §1, CONGELADO
# ==========================================================================
DESIGN_VARS = [
    # nombre               min        max         unidad / notas
    ("arch",                0.0,       2.0),      # gate: can/annular/reverse
    ("AL_over_Aref",        0.55,      0.80),     # -    [Ciencia §2.5]
    ("frac_pz",             0.15,      0.60),     # -    aire zona primaria
    ("frac_sz",             0.00,      0.25),     # -    aire zona secundaria
    ("frac_dz",             0.15,      0.60),     # -    aire de dilución
    ("swirl_SN",            0.50,      1.20),     # -    número de swirl
    ("n_holes_pz",          4.0,      16.0),      # -    entero relajado
    ("n_holes_dz",          6.0,      18.0),      # -    entero relajado
    ("d_eff_mm",            0.50,      1.00),     # mm   Ø efusión (LPBF)
    ("pitch_over_d_eff",    4.0,      10.0),      # -    paso/Ø de efusión
    ("injector",            0.0,       3.0),      # gate: airblast..micromix
    ("ALR",                 2.0,       6.0),      # -    solo airblast
    ("L_pz_over_DL",        0.50,      1.00),     # -
    ("L_dz_over_DL",        1.00,      3.00),     # -
    ("t_liner_mm",          0.50,      1.50),     # mm
    ("fuel",                0.0,       2.0),      # gate: jet_a/propane/h2
    ("P3_Pa",           1.0e5,     2.0e6),        # Pa   punto de operación
    ("T3_K",              250.0,   1150.0),       # K    punto de operación
    ("mdot_kgs",            0.02,     30.0),      # kg/s punto de operación
    ("T4_K",              800.0,   1800.0),       # K    punto de operación
]
# Reglas duras del contrato §1:
#  * `frac_cool = 1 − (frac_pz + frac_sz + frac_dz)` es CIERRE, no variable.
#  * Los índices 16–19 son el punto de operación y NUNCA se mueven
#    (`DesignSpec.fix_operating_point()` los sobreescribe siempre).
#  * Las variables nuevas se AÑADEN en el índice 20+ y `pad_theta()` las
#    rellena con su default neutro; `NDIM_LEGACY` no cambia.
#  * Las tres gates (0, 10, 15) se decodifican con `int(round(...))` + clip
#    ANTES de cualquier operación float (invariante CONTEXTO §2.1).
VAR_NAMES = [v[0] for v in DESIGN_VARS]
BOUNDS_LO = np.array([v[1] for v in DESIGN_VARS])
BOUNDS_HI = np.array([v[2] for v in DESIGN_VARS])
NDIM = len(DESIGN_VARS)
NDIM_LEGACY = 20          # θ de la WAVE0/WAVE1 — ver pad_theta

#: Índices con nombre (los consumen structures_core, dfam_checker y la CLI).
IDX_ARCH, IDX_INJECTOR, IDX_FUEL = 0, 10, 15
IDX_P3_PA, IDX_T3_K, IDX_MDOT_KGS, IDX_T4_K = 16, 17, 18, 19

#: θ de referencia — caso ancla KJ66 / `REF_CB10` (contrato §1, literal).
THETA_REF = np.array([
    2.0,        # arch = reverse_annular
    0.70,       # AL_over_Aref
    0.30,       # frac_pz
    0.12,       # frac_sz
    0.42,       # frac_dz         (frac_cool = 0.16)
    0.70,       # swirl_SN
    8.0,        # n_holes_pz
    10.0,       # n_holes_dz
    0.60,       # d_eff_mm
    6.0,        # pitch_over_d_eff
    2.0,        # injector = vaporizer  (arquitectura KJ66)
    4.0,        # ALR (no aplica al vaporizador; queda inerte)
    0.75,       # L_pz_over_DL
    1.80,       # L_dz_over_DL
    0.80,       # t_liner_mm
    0.0,        # fuel = jet_a
    250_000.0,  # P3_Pa
    430.0,      # T3_K
    0.13,       # mdot_kgs
    1150.0,     # T4_K
])


def pad_theta(theta: np.ndarray) -> np.ndarray:
    """θ legacy → θ de NDIM componentes rellenando con defaults neutros.

    Hoy `NDIM == NDIM_LEGACY == 20`, así que la función es la identidad
    (con copia a float y `ravel`): existe desde el día 0 para que añadir
    variables en el índice 20+ no rompa checkpoints, anclas de regresión ni
    datasets del surrogate — el patrón exacto de `Phy-AC/physics_core.py`.
    Cuando se añada la variable 21 basta con extender `_THETA_PAD_DEFAULTS`.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    n = theta.shape[0]
    if n == NDIM:
        return theta
    if NDIM_LEGACY <= n < NDIM:
        pad = np.array(_THETA_PAD_DEFAULTS[n - NDIM_LEGACY:], dtype=float)
        return np.concatenate([theta, pad])
    raise ValueError(
        f"theta debe tener entre {NDIM_LEGACY} (legacy) y {NDIM} componentes "
        f"— llegaron {n}")


#: Defaults neutros de las variables añadidas por encima de NDIM_LEGACY.
#: Hoy vacío (NDIM == NDIM_LEGACY); cada variable nueva añade aquí el valor
#: que reproduce EXACTAMENTE el comportamiento anterior a su introducción.
_THETA_PAD_DEFAULTS: list[float] = []


def denormalize(u: np.ndarray) -> np.ndarray:
    """[0,1]^NDIM → unidades físicas (SI, salvo los sufijos `_mm`)."""
    return BOUNDS_LO + np.asarray(u, dtype=float) * (BOUNDS_HI - BOUNDS_LO)


def normalize(theta: np.ndarray) -> np.ndarray:
    """Unidades físicas → [0,1]^NDIM (sin clip: puede salirse si θ se sale)."""
    return (pad_theta(theta) - BOUNDS_LO) / (BOUNDS_HI - BOUNDS_LO)


def decode_gates(theta: np.ndarray) -> dict:
    """Decodifica las TRES gates de θ **antes de cualquier operación float**.

    Invariante duro del contrato §1 y de CONTEXTO §2.1: una gate es un
    entero disfrazado de float; redondearla tarde (o interpolar entre dos
    arquitecturas) produce híbridos sin sentido físico. Aquí se hace
    `int(round(x))` + clip al rango de la gate y ya no se vuelve a mirar el
    float. Nunca lanza: NaN/inf caen al valor 0 de su gate.

    Devuelve `{arch_id, arch, injector_id, injector, fuel_id, fuel}`.
    """
    t = np.asarray(theta, dtype=float).ravel()

    def _gate(idx: int, names: tuple) -> int:
        v = t[idx] if idx < t.shape[0] else 0.0
        if not np.isfinite(v):
            return 0
        return int(min(max(int(round(float(v))), 0), len(names) - 1))

    a = _gate(IDX_ARCH, ARCH_NAMES)
    i = _gate(IDX_INJECTOR, INJECTOR_NAMES)
    f = _gate(IDX_FUEL, FUEL_NAMES)
    return dict(arch_id=a, arch=ARCH_NAMES[a],
                injector_id=i, injector=INJECTOR_NAMES[i],
                fuel_id=f, fuel=FUEL_NAMES[f])


# ==========================================================================
# 2. Capa 0 — acceso a `data/` y registro de coeficientes provisionales
# ==========================================================================
#: Avisos de coeficientes provisionales que han llegado a un número del
#: record. El reporte los pinta en el bloque «limitaciones» (contrato §7:
#: «todo valor provisional que llega a un número del reporte enciende un
#: warning visible»). Cada entrada: (clave, mensaje).
PROVISIONAL_COEFFS: list[tuple[str, str]] = []
_PROV_SEEN: set[str] = set()


#: Colector de la evaluación EN CURSO. `PROVISIONAL_COEFFS` es el inventario
#: del PROCESO (crece una vez por aviso distinto y alimenta el reporte y
#: `docs/PENDIENTES_Lverificar.md`); este otro recoge, sin deduplicar entre
#: llamadas, los avisos que dispara **una** evaluación concreta.
#:
#: Los dos existen porque confundirlos era un bug real, cazado por el test
#: T16b: `rec["limitations"]` salía del inventario global, así que el MISMO θ
#: devolvía 5 avisos en un intérprete virgen y 9 después de evaluar otros 64
#: diseños. Un record debe ser función pura de θ (contrato §10.8): si depende
#: de la historia del proceso, las anclas de regresión no valen nada.
_EVAL_STACK: list[dict] = []


def _warn_provisional(key: str, msg: str) -> None:
    """Registra un aviso de coeficiente provisional.

    Idempotente en el inventario del proceso; SIEMPRE registrado en la
    evaluación en curso (si la hay), que es lo que viaja en el record.
    """
    if key not in _PROV_SEEN:
        _PROV_SEEN.add(key)
        PROVISIONAL_COEFFS.append((key, msg))
    if _EVAL_STACK:
        _EVAL_STACK[-1].setdefault(key, msg)


class _capture_limitations:
    """Contexto que aísla los avisos de UNA evaluación. Reentrante y a
    prueba de excepciones (el `g` continuo del núcleo nunca lanza, pero el
    contexto tiene que cerrarse igual si algo revienta más adentro)."""

    def __enter__(self):
        _EVAL_STACK.append({})
        return self

    def __exit__(self, *exc):
        self.collected = _EVAL_STACK.pop() if _EVAL_STACK else {}
        return False        # no traga excepciones

    def report(self) -> list[dict]:
        got = getattr(self, "collected", None)
        if got is None:
            got = _EVAL_STACK[-1] if _EVAL_STACK else {}
        return [dict(key=k, message=m) for k, m in sorted(got.items())]


def _current_limitations() -> list[dict]:
    """Avisos de la evaluación EN CURSO (lo que va en `rec["limitations"]`).

    Fuera de un `_capture_limitations` cae al inventario del proceso, que es
    el comportamiento antiguo — así un `_record_from_chain` llamado a mano
    desde un test o un script sigue devolviendo algo útil.
    """
    if _EVAL_STACK:
        return [dict(key=k, message=m)
                for k, m in sorted(_EVAL_STACK[-1].items())]
    return provisional_report()


def provisional_report() -> list[dict]:
    """Inventario del PROCESO: todos los avisos provisionales acumulados
    desde que se importó el módulo. Es lo que consume el reporte global y
    `docs/PENDIENTES_Lverificar.md` — NO lo que va en `rec["limitations"]`
    (ver `_capture_limitations`)."""
    return [dict(key=k, message=m) for k, m in PROVISIONAL_COEFFS]


def _data_pipeline():
    """Import perezoso y VERIFICADO de `data_pipeline` (nunca lanza).

    ⚠️ Trampa real de esta máquina: Phy-AC está instalado como paquete
    editable, así que `import data_pipeline` desde otro directorio puede
    resolver al de Phy-AC. Se compara el directorio del módulo con el de
    ESTE archivo y se devuelve `None` si no coinciden.
    """
    global _DP_MOD, _DP_TRIED
    if _DP_TRIED:
        return _DP_MOD
    _DP_TRIED = True
    try:
        import importlib
        mod = importlib.import_module("data_pipeline")
        here = os.path.dirname(os.path.abspath(__file__))
        there = os.path.dirname(os.path.abspath(getattr(mod, "__file__", "")))
        _DP_MOD = mod if os.path.normcase(here) == os.path.normcase(there) \
            else None
    except Exception:
        _DP_MOD = None
    if _DP_MOD is None:
        _warn_provisional(
            "data_pipeline",
            "data_pipeline is not importable: EVERY coefficient comes from "
            "the provisional constants of physics_core, not from data/.")
    return _DP_MOD


_DP_MOD = None
_DP_TRIED = False
_TABLES: dict[str, dict | None] = {}


def load_table(rel: str) -> dict | None:
    """`data_pipeline.load_table` con caché de módulo. Nunca lanza."""
    if rel in _TABLES:
        return _TABLES[rel]
    dp = _data_pipeline()
    t = None
    if dp is not None:
        try:
            t = dp.load_table(rel)
        except Exception:
            t = None
    if t is None:
        _warn_provisional(f"table:{rel}",
                          f"table '{rel}' missing or unreadable: the "
                          f"provisional values of physics_core are used.")
    _TABLES[rel] = t
    return t


def _tab_param(rel: str, param: str, default: float,
               model: str | None = None, why: str = "") -> float:
    """Valor de una tabla en formato largo (`param`/`value`), con default.

    `model` acota la búsqueda en las tablas que además llevan columna
    `model` (atomización, emisiones). Si la fila no está —o su `status` es
    `provisional`— se registra el aviso correspondiente. Nunca lanza.
    """
    t = load_table(rel)
    if t is not None:
        cols = t["cols"]
        keys = cols.get("param")
        vals = cols.get("value")
        if keys is not None and vals is not None and vals.dtype.kind in "fiu":
            mask = np.asarray(keys, dtype=str) == param
            if model is not None and "model" in cols:
                mask = mask & (np.asarray(cols["model"], dtype=str) == model)
            idx = np.nonzero(mask)[0]
            if idx.size:
                i = int(idx[0])
                st = t["status"]
                if "status" in cols:
                    st = str(np.asarray(cols["status"], dtype=str)[i])
                if st != "verified":
                    _warn_provisional(
                        f"{rel}:{model or ''}:{param}",
                        f"{rel} -> {model + '.' if model else ''}{param} = "
                        f"{float(vals[i]):.6g} is marked '{st}'. {why}")
                v = float(vals[i])
                if np.isfinite(v):
                    return v
    _warn_provisional(
        f"missing:{rel}:{model or ''}:{param}",
        f"{rel} -> {model + '.' if model else ''}{param} not available: "
        f"the provisional value {default:.6g} of physics_core is used. "
        f"{why}")
    return float(default)


def _tab_row(rel: str, key_col: str, key: str, val_col: str,
             default: float) -> float:
    """Valor de una tabla ancha: fila donde `key_col == key`, columna
    `val_col`. Nunca lanza; registra aviso si falta."""
    t = load_table(rel)
    if t is not None:
        cols = t["cols"]
        kc, vc = cols.get(key_col), cols.get(val_col)
        if kc is not None and vc is not None and vc.dtype.kind in "fiu":
            idx = np.nonzero(np.asarray(kc, dtype=str) == key)[0]
            if idx.size:
                v = float(vc[int(idx[0])])
                if np.isfinite(v):
                    return v
    _warn_provisional(f"missing:{rel}:{key}:{val_col}",
                      f"{rel} -> {key}.{val_col} not available: the "
                      f"provisional {default:.6g} is used.")
    return float(default)


# ==========================================================================
# 3. Tunables del modelo L0 (SCREAMING_CASE, procedencia + recalibración)
#
# Regla de la casa: cada número lleva su fidelidad de origen. Los que están
# marcados ⚠️ se leen de `data/` cuando la tabla existe y estos valores son
# el respaldo documentado; los que NO tienen tabla son cierres geométricos
# o de modelado propios de Phy-CB y se declaran como tales.
# ==========================================================================

# --- Dimensionado (Ciencia §2) -------------------------------------------
DP_OVER_P_DESIGN = 0.030   # Δp/p de diseño [-]. En microturbinas Δp/p es
#                            RESTRICCIÓN DURA del ciclo (2–4 %), no salida
#                            (Ciencia §2.1). Default = `dp_over_p_nom` de la
#                            fila `micro_target` de plf_by_arch.csv; la CLI y
#                            DesignSpec lo sobreescriben con el del usuario.
PLF_MICRO_DEFAULT = 20.0   # ΔP₃₋₄/q_ref de la clase micro (banda 15–25).
LINER_FRAC_OF_DP = 0.75    # ΔP_liner / ΔP_total. Ciencia §2.1: «ΔP_liner ≈
#                            70–80 % del ΔP total (resto difusor + snout)».
#                            ⚠️ punto medio de la banda; RECALIBRAR con el
#                            difusor real en WAVE5.
ETA_SIZING_TARGET = 0.98   # η_comb objetivo que dispara el criterio químico
#                            θ. Procedencia: `reference/class_table.csv`,
#                            `eta_comb_design` de la clase micro = 0.970–0.990
#                            (centro de banda). NO es 0.999: exigir el nivel
#                            de una aeronáutica grande a una microturbina
#                            duplica A_ref y saca V_c y τ de TODAS las bandas
#                            de referencia de su clase. ⚠️ RECALIBRAR junto
#                            con la carta η_c(θ) (eta_vs_theta.csv es
#                            provisional: un solo punto de anclaje).
ANNULUS_HTR = 0.60         # D_hub/D_casing de las arquitecturas anulares.
#                            Cierre geométrico de la casa (no hay fuente):
#                            una anular inversa de microturbina envuelve eje
#                            y turbina, HTR 0.5–0.65. Solo afecta a
#                            `D_casing_mm`. ⚠️ RECALIBRAR con la envolvente
#                            real del banco (WAVE7).
THETA_B_K = 300.0          # constante b del parámetro θ de Lefebvre [K]
#                            (Ciencia §2.2: b ≈ 300 K). ✅ forma estándar.
SIZING_ITERS = 32          # iteraciones del criterio químico (bucle ACOTADO)
SIZING_TOL = 1.0e-10       # test de convergencia relativo sobre A_ref

# --- Reparto de aire (Ciencia §3) ----------------------------------------
FRAC_COOL_MIN = 0.02       # refrigeración mínima que se reserva al reescalar
#                            un θ con Σ(fracciones) ≥ 1 (mantiene el cierre
#                            frac_cool = 1 − Σ estrictamente positivo).
SWIRLER_FRAC_OF_PZ = 0.50  # fracción del aire de PZ que entra por el
#                            swirler (el resto, por los orificios de PZ).
#                            Cierre de la casa; banda de práctica 0.4–0.7.

# --- Orificios y chorros (Ciencia §4) ------------------------------------
CD_HOLE_DEFAULT = 0.62     # C_d de orificio plano de canto vivo (banda ✅
#                            0.60–0.65) — respaldo de cd_holes.csv.
CD_EFFUSION_DEFAULT = 0.65  # C_d de orificio de efusión cilíndrico ⚠️.
JET_PENETRATION_C = 1.15   # Y_max/d_j = C·√J ✅ (verificada en 2 fuentes).
Y_OVER_DL_SINGLE_ROW = 0.50   # ✅ Lefebvre & Norster 1968: fila única.
Y_OVER_DL_TWO_ROWS = 0.33     # ✅ dos filas desfasadas.
K_SW_DEFAULT = 1.30        # K_sw de álabes planos ⚠️ (swirler_K.csv).
SWIRL_VANE_BETA_DEG = 50.0    # β del álabe del swirler (banda ✅ 45–60°).
A_SW_OVER_AREF = 0.14      # A_swirler/A_ref. Cierre de la casa elegido para
#                            que ΔP_sw/q_ref caiga en la banda del PLF
#                            (15–25) en el caso ancla. ⚠️ RECALIBRAR.
D_PREFILMER_OVER_DL = 0.25    # Ø del prefilmer del airblast / D_L. Cierre
#                            de la casa (el atomizador ocupa ~1/4 del domo).

# --- Factor de patrón (Ciencia §4.3) -------------------------------------
PF_K_DILUTION_DEFAULT = 0.038882   # k de PF = exp[−k(L_DZ/D_L)(ΔP_L/q_ref)]
#                            ⚠️ provisional (pf_constant.csv `k_dilution`).
PF_MAX_DEFAULT = 0.45      # límite de PF de la restricción g[0].
#                            DESVIACIÓN DECLARADA: la tabla trae también
#                            `PF_max_default` = 0.35, pero ese 0.35 es
#                            EXACTAMENTE el punto en el que se calibró
#                            `k_dilution` (L_DZ/D_L = 1.8, ΔP_L/q_ref = 15),
#                            así que usarlo deja todo diseño nominal justo
#                            sobre la frontera. Phy-CB usa el techo ✅ de la
#                            clase micro (`PF_target_micro_max` = 0.45:
#                            «0.30–0.45 aceptable en microturbina con turbina
#                            radial no refrigerada», Ciencia §4.3) y deja que
#                            DesignSpec lo apriete.
RTDF_OVER_PF = 0.75        # RTDF/PF ⚠️ (pf_constant.csv `rtdf_over_pf`).

# --- Térmico 1D del liner (Ciencia §5) -----------------------------------
RAD_INTERNAL_PREFACTOR = 0.50   # ⚠️ thermal_coeffs.csv
GAS_EMISSIVITY_COEFF = 0.29     # ⚠️ thermal_coeffs.csv
CONV_COEFF = 0.020              # ⚠️ Dittus-Boelter modificado de Lefebvre
CONV_RE_EXP = 0.80              # ✅ exponente de Dittus-Boelter
CONV_D_EXP = -0.20              # ✅
EPS_WALL_DEFAULT = 0.80         # ⚠️ (manda `emissivity_wall` del material)
EPS_G_FLOOR_LUMINOUS = 0.15     # suelo de emisividad de gas para llamas
#                            LUMINOSAS (queroseno). ⚠️ CALIBRACIÓN DE LA
#                            CASA: la forma documentada en Ciencia §5 con
#                            P₃ en Pa da ε_g ≈ 0.01 en el caso ancla, un
#                            orden de magnitud por debajo de la banda 0.2–0.6
#                            que la literatura reporta para queroseno — es
#                            decir, dejaría el gas transparente y subestimaría
#                            T_pared. El suelo se eligió para que T_w1 del
#                            caso ancla caiga en 900–1050 K
#                            (`class_table.csv`, `T_wall_liner_K`, clase 0).
#                            RECALIBRAR con el libro cap. 8 o con termopares.
EPS_G_FLOOR_NONLUMINOUS = 0.05  # ídem para gas natural / H₂ (no luminosas)
EPS_G_MAX = 0.90
THERMAL_ITERS = 60         # iteraciones del balance R1+C1=K=R2+C2 (ACOTADO)
THERMAL_TOL_K = 1.0e-7     # test de convergencia [K]
N_WALL_STATIONS = 7        # puntos del perfil axial `T_wall_profile`
EFFUSION_ROWS_SAT = 10.0   # ✅ la eficacia de efusión satura a ~10 filas
PRANDTL_GAS = 0.71         # Pr del aire/gases de combustión (≈cte)
M_EFF_MIN = 0.30           # relación de soplado de DISEÑO de la efusión ✅
M_EFF_MAX = 0.80           # (Ciencia §5.1: M = ρ_cU_c/(ρ_gU_g) = 0.3–0.8;
#                            se leen de film_effusion.csv `effusion.M_design_
#                            min/max`, esto es el respaldo). Fuera de la banda
#                            el diseño de efusión está fuera del régimen en el
#                            que la eficacia está tabulada: a M < 0.3 el
#                            refrigerante se pega y a M > 0.8 hay blow-off.
#                            Dispara `_warn_provisional` (aviso de diseño).
M_EFF_VALID_MIN = 0.20     # extremos de la REJILLA del mapa η_film(M) de
M_EFF_VALID_MAX = 1.00     # film_effusion.csv (se leen del propio mapa; esto
#                            es el respaldo). Fuera de [0.2, 1.0] el mapa se
#                            evalúa CLAMPEADO —la eficacia deja de estar
#                            modelada y pasa a estar saturada—, y por eso este
#                            rango (no el de diseño) es el que consume la
#                            penalización continua de g[8]. Ver `_constraints`.

# --- Efusión: FUENTE ÚNICA de área, patrón y C_d (auditoría 2026-07-31) ---
# Cuatro módulos hablaban de la MISMA efusión con dos áreas y dos C_d
# distintos (ver §13.b). Las constantes que la geometría y la física
# COMPARTEN viven a partir de aquí y SOLO aquí; `geometry_generator` y
# `crn_core` las consumen desde este módulo (principio del radio de fillet,
# CONTEXTO §8).
LINER_R_OVER_RCAS = 0.92   # r_liner_exterior / r_carcasa en las anulares.
#                            Fija a la vez la columna `r_mm` de las filas de
#                            orificios (`_r_liner_mm`) y el radio sobre el
#                            que `geometry_generator._annulus_geometry`
#                            construye la corona — luego también el ÁREA
#                            MOJADA que lleva la efusión. ⚠️ cierre de la
#                            casa (holgura radial liner↔carcasa del 8 %).
ANNULUS_AREA_REL_TOL = 0.02  # holgura antes de avisar de que el atajo
#                            `A_ref − A_L` se aparta de la corona
#                            geométrica (ver `_sizing`). 2 % es el
#                            redondeo del dimensionado, no física.
ANNULUS_R_MIN_FRAC = 0.15  # r_li mínimo / r_carcasa: suelo con el que la
#                            corona deja de degenerar cuando A_paso no cabe
#                            dentro de `LINER_R_OVER_RCAS·r_carcasa`. MISMA
#                            rama de respaldo que `_annulus_geometry`.
#: Paredes del liner que llevan efusión en **v1**, por arquitectura.
#: DECISIÓN DECLARADA (2026-07-31): en `annular` y `reverse_annular` la pared
#: caliente tiene cara EXTERIOR e INTERIOR, y v1 perfora **solo la exterior**
#: — que es exactamente lo que dibuja `geometry_generator` (integra
#: `prof[k][2]`, el radio exterior). Motivo: en la anular inversa la cara
#: interior envuelve el `conducto_turbina`, que el propio contrato de
#: geometría declara fuera de alcance (es hardware de Phy-TB), y en la anular
#: recta el liner interior necesitaría su propio anillo de alimentación que
#: v1 no dimensiona. Cuando se abra la cara interior, aquí se añade `"inner"`
#: y `effusion_wall_area_m2` suma `2π·r_li·L_tot` — el resto de la cadena no
#: se toca.
EFFUSION_WALLS = {"can": ("outer",),
                  "annular": ("outer",),
                  "reverse_annular": ("outer",)}
EFFUSION_AREA_REL_TOL = 2.0e-3   # tolerancia RELATIVA declarada entre el área
#                            mojada NOMINAL de este módulo y la que
#                            `geometry_generator` integra sobre la polilínea
#                            REAL. Con el modelo consciente de arquitectura
#                            las dos son la MISMA fórmula cerrada y solo las
#                            separa el redondeo de la polilínea a
#                            `geometry_generator.ROUND_N` decimales (~1e-7
#                            relativo): 2e-3 deja tres órdenes de margen y
#                            sigue cazando cualquier divergencia de MODELO
#                            (la que había era ×1.375). Por encima de esta
#                            tolerancia NO es una nota, es una violación del
#                            contrato entre capas.
EFFUSION_N_HOLES_TOL = 1   # holguras admitidas al comparar nº de orificios:
#                            ambos lados hacen `int(A_pared/paso²)`, y un
#                            truncamiento que caiga justo en el borde puede
#                            separarlos EXACTAMENTE en uno. Más de uno es un
#                            desacuerdo real.
CD_EFFUSION_INCLINED = 0.38    # factor MULTIPLICATIVO sobre el C_d de
#                            `cd_holes.csv` para orificios de efusión
#                            INCLINADOS (20–30° respecto a la pared) y con
#                            crossflow por AMBAS caras. La literatura de
#                            film-cooling (Hay & Lampard 1998; Gritsch et al.
#                            1998) sitúa ese cociente en 0.35–0.60 respecto a
#                            un orificio normal sin crossflow; se toma el
#                            extremo BAJO porque el crossflow interno del
#                            anillo de un liner AM es fuerte (U_anillo ≈ 60
#                            m/s en el ancla). ⚠️ ES EL TUNABLE MÁS
#                            INFLUYENTE sobre `splits_solved["cool"]`.
#                            VIVÍA EN `crn_core`; se movió aquí el 2026-07-31
#                            para que la red 1D y el LBM no puedan usar dos
#                            C_d distintos. `crn_core.CD_EFFUSION_INCLINED`
#                            es hoy un alias de lectura de esta constante.
RA_ASBUILT_UM = 12.0       # rugosidad Ra as-built LPBF [µm] con la que se
#                            entra en `cd_am_roughness.csv`. ⚠️ banda típica
#                            de superficie descendente sin post-proceso
#                            (6–20 µm). VIVÍA EN `crn_core`; mismo motivo de
#                            traslado que `CD_EFFUSION_INCLINED`.

# --- Estabilidad (Ciencia §6) --------------------------------------------
LBO_CONST_SI_DEFAULT = 6.95e16   # ⚠️ lbo_constant.csv `C_LBO_SI`
LBO_EXP_P3 = -1.30         # ✅ estabilidad ∝ P₃^−1.3
LBO_EXP_SMD = 2.00         # ✅ ∝ D₃₂²
LBO_MARGIN_MIN = 0.20      # ⚠️ margen mínimo exigido por g[2]
#: DEFINICIÓN EXPLÍCITA de `LBO_margin` (auditoría 2026-07-30). El nombre
#: «margen» con un valor ≈1.0 induce a error —parece un 100 % de margen, y de
#: hecho lo es, pero medido como EXCESO RELATIVO, no como fracción—, así que
#: el record publica las tres formas y esta cadena junto a ellas:
#:   LBO_margin       = φ_PZ/φ_LBO − 1   (exceso relativo; 0 = justo en LBO)
#:   LBO_margin_ratio = φ_PZ/φ_LBO       (razón; 1 = justo en LBO)
#:   LBO_margin_frac  = 1 − φ_LBO/φ_PZ   (fracción de φ_PZ que se puede
#:                                        perder antes de apagarse; ∈ [0,1))
#: Se CONSERVA `LBO_margin` con su definición original: es ABI (contrato §4.2
#: y g[2], que compara φ_LBO·(1+LBO_MARGIN_MIN) con φ_PZ). Renombrarla
#: rompería el vector g y las anclas de regresión.
LBO_MARGIN_DEF = ("LBO_margin = phi_pz/phi_LBO - 1 (exceso relativo; "
                  "0 = justo en el limite de extincion pobre). "
                  "LBO_margin_ratio = phi_pz/phi_LBO. "
                  "LBO_margin_frac = 1 - phi_LBO/phi_pz.")
LBO_GAS_EASE = 0.70        # φ_LBO(gas) / φ_LBO(líquido equivalente).
#                            El término de evaporación D₃₂²/λ_eff NO aplica a
#                            un combustible gaseoso (Ciencia §6.1), así que
#                            la constante cambia de unidades: la rama gas usa
#                            C_gas = C_LBO_SI · (D₃₂²/λ)_ref · LBO_GAS_EASE.
#                            ⚠️ CALIBRACIÓN DE LA CASA (los gases estabilizan
#                            mejor); RECALIBRAR en WAVE7.
LBO_EVAP_REF = (50.0e-6) ** 2 / 8.0e-7   # (D₃₂²/λ_eff) del punto de
#                            calibración de C_LBO_SI (D₃₂ = 50 µm)
QUENCH_PECLET = 50.0       # d_q = Pe·α/S_L (Ciencia §6.2: d_q ∝ α/S_L).
#                            Pe_q ≈ 40–60 es la banda clásica de distancia de
#                            apagado. ⚠️ punto medio.
SPRAY_DQ_OVER_SMD = 10.0   # en rama spray d_q = max(d_q,gas, C·D₃₂) — da la
#                            dependencia ✅ E_min ∝ D₃₂³ cuando el espray
#                            manda. ⚠️ constante de la casa.
S_L_EXP_T = 2.00           # S_L ∝ (T_u/298)^2 ✅ (banda clásica 1.7–2.2)
S_L_EXP_P = -0.25          # S_L ∝ (P/P_atm)^−0.25 ✅

# --- Atomización (Ciencia §7) --------------------------------------------
SMD_SIMPLEX_C = 2.25       # ⚠️ atomization_coeffs.csv `simplex_lefebvre.C`
DP_FUEL_SIMPLEX_PA = 8.0e5  # ΔP de combustible del simplex [Pa] cuando el
#                            θ no lo fija (bomba de 8 bar — banda de práctica
#                            micro 3–8 bar, Ciencia §7.2). ⚠️ de la casa.
VAPORIZER_D50_UM = 50.0    # ✅ d₅₀ ≤ 50 µm del vaporizador clase KJ66
MICROMIX_D_HOLE_MM = 0.50  # ✅ Ø típico del micromix H₂ (0.3–1.0 mm)
MICROMIX_KW_PER_INJ = 2.0  # ✅ 1–3 kW por inyector
U_ATOMIZING_MIN_MS = 20.0  # clamp inferior de U_aire del atomizador para que
#                            la correlación no explote con θ degenerado

# --- Emisiones (Ciencia §8) ----------------------------------------------
EI_NOX_LEFEBVRE_C = 9.0e-8      # ✅ verificada (tesis U. Toronto ec. 1.4)
EI_P3_SCALE_LEFEBVRE = 1.0e-3   # Pa → kPa. ⚠️ PENDIENTE DECLARADO: la unidad
#                            de P₃ no es explícita en la fuente verificada
#                            (el original de 1984 usa kPa). Se LEE de
#                            emissions_coeffs.csv (`P3_unit_scale_from_Pa`) y
#                            se fija ahí, nunca aquí: este valor es solo el
#                            respaldo si la tabla falta. AUDITORÍA DIMENSIONAL
#                            2026-07-30: kPa CONFIRMADA (ver §15.0).
EI_CO_LEFEBVRE_C = 86.0         # ⚠️ NO verificada
NOX_CO_WEIGHT_RATIO = 10.0      # peso relativo NOx:CO del óptimo analítico
#                            de V_c (Ciencia §8.2). NO es física: es una
#                            preferencia ambiental (1 g de NOx «cuesta» como
#                            10 g de CO). Cámbialo y el V_c óptimo se mueve.
EI_REPORT_MAX = 1.0e4      # tope de reporte de los EI [g/kg]. Algunas de las
#                            correlaciones del abanico son extremadamente
#                            sensibles a T_st fuera de su rango de calibración;
#                            el tope mantiene el record serializable y finito.
#                            El EI_NOx de cabecera es SIEMPRE el de Lefebvre
#                            (la única ✅ verificada con su unidad declarada).

#: **PANTALLA DE PLAUSIBILIDAD DEL ABANICO DE EI_NOx** [g/kg de combustible].
#: PROCEDENCIA de la banda (se escribe además en `emissions_coeffs.csv`,
#: modelo `plausibility`, y se lee de allí; esto es el respaldo):
#:   * Suelo 0.01 g/kg — el combustor más limpio documentado es el micromix
#:     H₂ de un dígito de ppm (`reference_micro.NOx_ppm15O2_gas_min` = 9 ppm
#:     ≈ 0.1 g/kg, Ciencia §7.4); se deja un orden de magnitud de margen por
#:     debajo. Nada por debajo de 0.01 g/kg es un combustor, es un error.
#:   * Techo 200 g/kg — `reference_micro.NOx_ppm15O2_kero_max` = 80 ppm
#:     ≈ 5 g/kg y el máximo de motor grande de la literatura es 40 g/kg
#:     (Ciencia §8.4); 200 deja un factor 5 sobre el peor caso real.
#: REGLA (contrato §7 + Ciencia §8, honestidad): una correlación fuera de la
#: banda SE SIGUE REPORTANDO con su nombre —el usuario ve el abanico
#: completo— pero (a) se marca en `record["EI_NOx_flags"]`, (b) queda
#: EXCLUIDA del min/max y de los estadísticos de `record["EI_NOx_fan"]`, y
#: (c) dispara `_warn_provisional()`.
EI_NOX_PLAUSIBLE_RANGE = (0.01, 200.0)
EI_NOX_PLAUSIBLE_MIN, EI_NOX_PLAUSIBLE_MAX = EI_NOX_PLAUSIBLE_RANGE

#: Nombres cortos del abanico de EI_NOx, en el orden en que se reportan.
#: Son las claves de `record["EI_NOx_flags"]` y de `nox_fan()`; el record
#: además las expone con el prefijo `EI_NOx_<nombre>` (contrato §4.2).
NOX_FAN_NAMES = ("lefebvre", "rizk_mongia", "zonal", "odgers", "laviolette")

#: PUNTO DE REFERENCIA DE MOTOR GRANDE para el ensayo de coherencia del
#: abanico de EI_NOx (§15.0). NO es un θ de Phy-CB —P₃ = 30 atm queda por
#: encima de `BOUNDS_HI[16]`— sino un punto analítico de la clase
#: aeronáutica, donde la literatura sitúa EI_NOx en **20–40 g/kg**. Es el
#: segundo extremo del ensayo dimensional: una unidad que solo funciona en
#: la microturbina no es evidencia, es ajuste. Cifras derivadas:
#: ρ₃ = P₃/(R·T₃) = 13.24 kg/m³ y V_c = τ·ṁ₃/ρ₃ = 6.80 L.
NOX_REF_LARGE_ENGINE = dict(
    P3_Pa=30.0 * P_ATM, T3_K=800.0, mdot_air_kgs=30.0, T_st_K=2600.0,
    T_pz_K=2300.0, T_sz_K=2000.0, tau_s=3.0e-3, tau_e_s=2.0e-3,
    tau_pz_s=1.0e-3, tau_sz_s=2.0e-3, dp_over_p=0.05,
    phi_pz=0.95, phi_sz=0.60, band_g_kg=(20.0, 40.0))

#: Banda objetivo de intensidad de combustión [MW/m³·atm] (Ciencia §2.4:
#: 20–60 para microturbina). Es la ENVOLVENTE de las tres clases de
#: `reference/class_table.csv` (`I_MW_m3_atm`: 25–60 / 20–50 / 20–45); se lee
#: de la tabla y esto es el respaldo. Consume la restricción BLANDA
#: `g_soft["I_band"]` — no entra en el `g` de 12 (contrato §5.1 congelado).
I_BAND_MIN_DEFAULT, I_BAND_MAX_DEFAULT = 20.0, 60.0

# --- Salida y ciclo -------------------------------------------------------
A4_OVER_AL = 1.00          # área de salida / área de liner (sin contracción
#                            en L0; la tobera de transición llega en WAVE3).
TURBINE_UNCOOLED_T_MAX_K = 1200.0   # T04 por encima de la cual una turbina
#                            radial no refrigerada exige sangrado. Banda de
#                            práctica micro (class_table `T4_K` ≤ 1350 con
#                            materiales avanzados). ⚠️ de la casa.
BLEED_FRAC_PER_K = 2.5e-4  # sangrado extra por K sobre el umbral. ⚠️

# --- Límites de las restricciones g (contrato §5.1) ----------------------
TAU_MIN_MS = 4.0           # τ mínimo ✅ (class_table `tau_ms`, clase micro)
TAU_MAX_MS = 10.0          # τ máximo ✅ (class_table `tau_ms`, clases 0–2)
M_EXIT_MAX = 0.30          # Mach de salida amigable a la turbina radial
DP_MAX_DEFAULT = 0.040     # Δp/p máximo ✅ (banda micro 2–4 %)
D_EFF_MIN_MM = 0.50        # Ø mínimo de efusión imprimible en LPBF ✅
PHI_PZ_MAX = 1.40          # φ_PZ máximo antes de zona rica tipo RQL
ETA_COMB_MIN = 0.97        # η_comb mínimo ✅ (class_table, clase micro)
Y_TARGET_DEFAULT = Y_OVER_DL_SINGLE_ROW   # objetivo de Y/D_L en dilución
Y_TOL = 0.15               # semiancho admisible de Y/D_L
T_WALL_MARGIN_K = 50.0     # margen de MODELO sobre T_max_flame_wall_K
#                            (idéntico a structures_core.T_WALL_MARGIN_K)
MATERIAL_DEFAULT = "Inconel-625"    # pared caliente por defecto

N_CONSTRAINTS = 12         # dim del vector g(theta) — contrato §5.1

#: RESTRICCIONES BLANDAS (`record["g_soft"]`, mismo convenio `≤ 0` ⇔ OK).
#: NO entran en `g` —el contrato §5.1 congela las 12 y su orden— ni en
#: `feasible`: son información de diseño que el reporte pinta y que el
#: usuario puede promover a dura desde `DesignSpec.constraints`.
#:   * `I_band`   — intensidad de combustión fuera de la banda de clase
#:                  (Ciencia §2.4 / `reference/class_table.csv`).
#:   * `M_design` — relación de soplado de la efusión fuera de la banda de
#:                  DISEÑO 0.3–0.8 (Ciencia §5.1). El rango de VALIDEZ del
#:                  mapa, más ancho, sí consume g[8] (ver `_constraints`).
SOFT_CONSTRAINT_NAMES = ("I_band", "M_design")


# ==========================================================================
# 4. Paquete de coeficientes de la capa 0 (se construye UNA vez)
# ==========================================================================
_COEFFS: dict | None = None


def coeffs() -> dict:
    """Coeficientes de `data/` con respaldo provisional. Cacheado y puro.

    Se construye en el primer `evaluate` y nunca se vuelve a leer disco: el
    objetivo de rendimiento es ~1 ms por punto y una lectura de CSV por
    evaluación lo destrozaría.
    """
    global _COEFFS
    if _COEFFS is not None:
        return _COEFFS
    L = "lefebvre/"
    c: dict = {}
    # --- dimensionado ----------------------------------------------------
    c["PLF"] = _tab_row(L + "plf_by_arch.csv", "arch", "micro_target",
                        "PLF_nom", PLF_MICRO_DEFAULT)
    c["dp_over_p"] = _tab_row(L + "plf_by_arch.csv", "arch", "micro_target",
                              "dp_over_p_nom", DP_OVER_P_DESIGN)
    c["AL_over_Aref"] = {}
    c["L_sz_over_DL"] = {}
    for a in ARCH_NAMES:
        c["AL_over_Aref"][a] = _tab_row(L + "zone_lengths.csv", "arch", a,
                                        "AL_over_Aref", 0.70)
        c["L_sz_over_DL"][a] = _tab_row(L + "zone_lengths.csv", "arch", a,
                                        "L_sz_over_DL_nom", 0.50)
    # curva η_c(θ) digitalizada
    t = load_table(L + "eta_vs_theta.csv")
    if t is not None and "theta_SI" in t["cols"] and "eta_theta" in t["cols"]:
        th = np.asarray(t["cols"]["theta_SI"], dtype=float)
        et = np.asarray(t["cols"]["eta_theta"], dtype=float)
        order = np.argsort(th)
        c["eta_theta_x"] = np.log(np.maximum(th[order], 1.0))
        c["eta_theta_y"] = et[order]
        if t["status"] != "verified":
            _warn_provisional(
                "eta_vs_theta",
                "the η_c(θ) chart of eta_vs_theta.csv is PROVISIONAL (a "
                "single anchor point, θ=7e7 SI -> η=0.999): η_θ and the "
                "chemical sizing criterion inherit that uncertainty.")
    else:
        c["eta_theta_x"] = np.log(np.geomspace(1.0e6, 8.0e7, 25))
        c["eta_theta_y"] = 0.9995 * (
            1.0 - np.exp(-7.6004 * np.geomspace(1.0e6, 8.0e7, 25) / 7.0e7))
    # --- orificios y swirler ---------------------------------------------
    c["Cd_hole"] = _tab_row(L + "cd_holes.csv", "hole_type",
                            "plain_sharp_edged", "Cd_nom", CD_HOLE_DEFAULT)
    c["Cd_eff"] = _tab_row(L + "cd_holes.csv", "hole_type",
                           "effusion_cylindrical", "Cd_nom",
                           CD_EFFUSION_DEFAULT)
    c["K_sw"] = _tab_row(L + "swirler_K.csv", "vane_type", "flat_vanes",
                         "K_sw", K_SW_DEFAULT)
    c["jet_C"] = _tab_param(L + "atomization_coeffs.csv", "C",
                            JET_PENETRATION_C, model="jet_penetration")
    c["Y_single"] = _tab_param(L + "atomization_coeffs.csv",
                               "Ymax_over_DL_single_row",
                               Y_OVER_DL_SINGLE_ROW, model="jet_penetration")
    c["Y_two"] = _tab_param(L + "atomization_coeffs.csv",
                            "Ymax_over_DL_two_rows", Y_OVER_DL_TWO_ROWS,
                            model="jet_penetration")
    # --- factor de patrón -------------------------------------------------
    c["pf_k"] = _tab_param(L + "pf_constant.csv", "k_dilution",
                           PF_K_DILUTION_DEFAULT,
                           why="PF k calibrated to ONE anchor point.")
    c["pf_max"] = _tab_param(L + "pf_constant.csv", "PF_target_micro_max",
                             PF_MAX_DEFAULT)
    c["rtdf_over_pf"] = _tab_param(L + "pf_constant.csv", "rtdf_over_pf",
                                   RTDF_OVER_PF)
    # --- térmico ----------------------------------------------------------
    th_rel = L + "thermal_coeffs.csv"
    c["rad_pref"] = _tab_param(th_rel, "rad_internal_prefactor",
                               RAD_INTERNAL_PREFACTOR)
    c["eps_g_coeff"] = _tab_param(th_rel, "gas_emissivity_coeff",
                                  GAS_EMISSIVITY_COEFF)
    c["conv_in"] = _tab_param(th_rel, "conv_internal_coeff", CONV_COEFF)
    c["conv_out"] = _tab_param(th_rel, "conv_external_coeff", CONV_COEFF)
    c["conv_re_exp"] = _tab_param(th_rel, "conv_reynolds_exp", CONV_RE_EXP)
    c["conv_d_exp"] = _tab_param(th_rel, "conv_diameter_exp", CONV_D_EXP)
    c["eps_wall"] = _tab_param(th_rel, "eps_wall_default", EPS_WALL_DEFAULT)
    # mapa de eficacia de efusión η(M, ángulo) — formato largo
    c["eff_map"] = _load_effusion_map(L + "film_effusion.csv")
    c["eff_angle_deg"] = _tab_param_scheme(L + "film_effusion.csv",
                                           "effusion", "angle_deg_max", 30.0)
    c["eff_rows_sat"] = _tab_param_scheme(L + "film_effusion.csv", "effusion",
                                          "n_rows_saturation",
                                          EFFUSION_ROWS_SAT)
    #  Banda de DISEÑO de la relación de soplado (aviso) y rango de VALIDEZ
    #  del mapa (penalización continua). El segundo sale del propio mapa.
    c["M_eff_min"] = _tab_param_scheme(L + "film_effusion.csv", "effusion",
                                       "M_design_min", M_EFF_MIN)
    c["M_eff_max"] = _tab_param_scheme(L + "film_effusion.csv", "effusion",
                                       "M_design_max", M_EFF_MAX)
    _Mg = c["eff_map"]["M"]
    c["M_valid_min"] = float(_Mg[0]) if len(_Mg) else M_EFF_VALID_MIN
    c["M_valid_max"] = float(_Mg[-1]) if len(_Mg) else M_EFF_VALID_MAX
    # --- banda de intensidad de combustión (envolvente de las 3 clases) ---
    c["I_min"], c["I_max"] = _class_band("I_MW_m3_atm",
                                         I_BAND_MIN_DEFAULT,
                                         I_BAND_MAX_DEFAULT)
    # --- estabilidad ------------------------------------------------------
    lb = L + "lbo_constant.csv"
    c["C_LBO"] = _tab_param(lb, "C_LBO_SI", LBO_CONST_SI_DEFAULT,
                            why="global φ_LBO constant calibrated to ONE "
                                "anchor case; declared accuracy ±45 %.")
    c["lbo_exp_P3"] = _tab_param(lb, "exp_P3", LBO_EXP_P3)
    c["lbo_exp_SMD"] = _tab_param(lb, "exp_SMD", LBO_EXP_SMD)
    c["lbo_margin_min"] = _tab_param(lb, "LBO_margin_min", LBO_MARGIN_MIN)
    c["lbo_T3_scale"] = _tab_param(lb, "exp_T3_scale_K", THETA_B_K)
    # --- atomización ------------------------------------------------------
    at = L + "atomization_coeffs.csv"
    c["smd_simplex_C"] = _tab_param(at, "C", SMD_SIMPLEX_C,
                                    model="simplex_lefebvre",
                                    why="the 2.25 coefficient is not "
                                        "verified (ch. 6).")
    c["rl_C1"] = _tab_param(at, "C1", 3.33e-3,
                            model="prefilm_rizkalla_lefebvre")
    c["rl_C2"] = _tab_param(at, "C2", 13.0e-3,
                            model="prefilm_rizkalla_lefebvre")
    c["esl_C1"] = _tab_param(at, "C1", 0.073,
                             model="prefilm_elshanawany_lefebvre")
    c["esl_C2"] = _tab_param(at, "C2", 0.0015,
                             model="prefilm_elshanawany_lefebvre")
    c["nd_C1"] = _tab_param(at, "C1", 0.33, model="prefilm_nondimensional",
                            why="NON-DIMENSIONAL form not verified: it is "
                                "computed and reported, but it is NOT the "
                                "chosen model.")
    c["nd_C2"] = _tab_param(at, "C2", 0.068, model="prefilm_nondimensional")
    c["pj_C1"] = _tab_param(at, "C1", 0.48, model="plain_jet_rizk_lefebvre")
    c["pj_C2"] = _tab_param(at, "C2", 0.15, model="plain_jet_rizk_lefebvre")
    c["vap_d50_um"] = _tab_param(at, "d50_um_max", VAPORIZER_D50_UM,
                                 model="vaporizer_kj66")
    c["vap_n_tubes"] = _tab_param(at, "n_tubes_min", 6.0,
                                  model="vaporizer_kj66")
    c["vap_L_mm"] = _tab_param(at, "tube_length_mm", 64.0,
                               model="vaporizer_kj66")
    c["mm_d_mm"] = _tab_param(at, "d_hole_mm_typ", MICROMIX_D_HOLE_MM,
                              model="micromix_h2")
    c["mm_kW"] = _tab_param(at, "kW_per_injector_max", MICROMIX_KW_PER_INJ,
                            model="micromix_h2")
    c["evap_lambda_C"] = _tab_param(L + "emissions_coeffs.csv", "lambda_e_C",
                                    8.0, model="evaporation")
    # --- emisiones --------------------------------------------------------
    em = L + "emissions_coeffs.csv"
    c["nox_lef_C"] = _tab_param(em, "C", EI_NOX_LEFEBVRE_C,
                                model="nox_lefebvre")
    c["nox_lef_expP3"] = _tab_param(em, "exp_P3", 1.25, model="nox_lefebvre")
    c["nox_lef_Tst"] = _tab_param(em, "exp_Tst_coeff", 0.01,
                                  model="nox_lefebvre")
    #  ⚠️ EL PENDIENTE DECLARADO: la unidad de P₃ del EI_NOx de Lefebvre se
    #  FIJA AQUÍ, leyéndola de la tabla (nunca hard-coded en la fórmula).
    c["nox_lef_P3scale"] = _tab_param(
        em, "P3_unit_scale_from_Pa", EI_P3_SCALE_LEFEBVRE,
        model="nox_lefebvre",
        why="P₃ unit of Lefebvre's EI_NOx: DECLARED OPEN ITEM (Ciencia "
            "§8.1). It is suspect no. 1 whenever the band does not add up.")
    c["co_lef_C"] = _tab_param(em, "C", EI_CO_LEFEBVRE_C, model="co_lefebvre")
    c["co_lef_Tpz"] = _tab_param(em, "exp_Tpz_coeff", -0.00345,
                                 model="co_lefebvre")
    c["co_lef_expdp"] = _tab_param(em, "exp_dp_over_p", -0.50,
                                   model="co_lefebvre")
    c["co_lef_expP3"] = _tab_param(em, "exp_P3", -1.50, model="co_lefebvre")
    c["co_lef_P3scale"] = _tab_param(em, "P3_unit_scale_from_Pa", 1.0e-3,
                                     model="co_lefebvre")
    c["rm_C"] = _tab_param(em, "C", 15.0e14, model="nox_rizk_mongia")
    c["rm_Ea"] = _tab_param(em, "Ea_over_R_K", 71100.0,
                            model="nox_rizk_mongia")
    c["rm_expP3"] = _tab_param(em, "exp_P3", -0.05, model="nox_rizk_mongia")
    c["rm_expdp"] = _tab_param(em, "exp_dp_over_p", -0.50,
                               model="nox_rizk_mongia")
    z = "nox_rizk_mongia_zonal"
    c["rmz_C"] = _tab_param(em, "C", 1.0e13, model=z)
    c["rmz_Ea"] = _tab_param(em, "Ea_over_R_K", 71442.0, model=z)
    c["rmz_aa_C1"] = _tab_param(em, "aa_C1", 1.95, model=z)
    c["rmz_aa_C2"] = _tab_param(em, "aa_C2", 5.76, model=z)
    c["rmz_lean_C"] = _tab_param(em, "pz_lean_C", 7.56, model=z)
    c["rmz_lean_e"] = _tab_param(em, "pz_lean_exp_phi", -7.20, model=z)
    c["rmz_rich_C"] = _tab_param(em, "pz_rich_C", 5.21, model=z)
    c["rmz_rich_e"] = _tab_param(em, "pz_rich_exp_phi", -2.99, model=z)
    c["rmz_split"] = _tab_param(em, "phi_split", 1.08, model=z)
    c["rmz_off"] = _tab_param(em, "offset", -1.60, model=z)
    c["rmz_tau_pz"] = _tab_param(em, "exp_tau_pz", 0.64, model=z)
    c["rmz_sz_C"] = _tab_param(em, "sz_dz_C", 1.172, model=z)
    c["rmz_sz_e"] = _tab_param(em, "sz_dz_exp_phi", -4.56, model=z)
    c["rmz_tau_sz"] = _tab_param(em, "exp_tau_sz_dz", 0.876, model=z)
    c["od_C"] = _tab_param(em, "C", 29.0, model="nox_odgers")
    c["od_Ea"] = _tab_param(em, "Ea_over_R_K", 21670.0, model="nox_odgers")
    c["od_expP3"] = _tab_param(em, "exp_P3", 0.66, model="nox_odgers")
    c["od_tau"] = _tab_param(em, "tau_sat_coeff", 250.0, model="nox_odgers")
    c["od_P3scale"] = _tab_param(em, "P3_unit_scale_from_Pa", 1.0,
                                 model="nox_odgers",
                                 why="P₃ unit of Odgers: the dimensional "
                                     "audit at two extremes pins it to Pa "
                                     "(NOT atm, as Ciencia §8.4 assumed).")
    c["od_tauscale"] = _tab_param(em, "tau_unit_scale_from_s", 1.0,
                                  model="nox_odgers")
    c["lv_C"] = _tab_param(em, "C", 1.902e5, model="nox_laviolette")
    c["lv_expP3"] = _tab_param(em, "exp_P3", 0.4048, model="nox_laviolette")
    c["lv_tau"] = _tab_param(em, "tau_sat_coeff", 480.0,
                             model="nox_laviolette")
    c["lv_Ea"] = _tab_param(em, "Ea_over_R_K", 37902.0,
                            model="nox_laviolette")
    #  BUG DE UNIDADES CORREGIDO: Laviolette reutilizaba el factor de Odgers.
    #  Son correlaciones distintas de la misma revisión y cada una lleva el
    #  suyo (aunque la auditoría acabe fijando las dos en Pa).
    c["lv_P3scale"] = _tab_param(em, "P3_unit_scale_from_Pa", 1.0,
                                 model="nox_laviolette",
                                 why="Laviolette underestimates the large "
                                     "engine by a factor 5-9 with ANY unit: "
                                     "it stays in the fan, marked, without "
                                     "being forced.")
    c["lv_tauscale"] = _tab_param(em, "tau_unit_scale_from_s", 1.0,
                                  model="nox_laviolette")
    #  --- escalas de unidad de Rizk & Mongia (§15.0) ----------------------
    c["rm_P3scale"] = _tab_param(em, "P3_unit_scale_from_Pa", 1.0e-3,
                                 model="nox_rizk_mongia",
                                 why="exp_P3 = −0.05 makes the P₃ unit "
                                     "dimensionally UNDECIDABLE here; kPa is "
                                     "adopted for source consistency.")
    c["rm_tauscale"] = _tab_param(em, "tau_unit_scale_from_s", 1.0,
                                  model="nox_rizk_mongia",
                                  why="τ in SECONDS, not in ms as Ciencia "
                                      "§8.3 declared (ms gives 1147 g/kg on "
                                      "the anchor and 8447 on the large "
                                      "engine).")
    c["rm_dppscale"] = _tab_param(em, "dp_over_p_unit_scale_from_frac", 100.0,
                                  model="nox_rizk_mongia",
                                  why="Δp/p in PER CENT: it is the factor "
                                      "10 missing to land in 20-40 g/kg on "
                                      "the large engine.")
    c["rmz_P3scale"] = _tab_param(em, "P3_unit_scale_from_Pa", 1.0, model=z,
                                  why="P₃ in Pa: no other unit leaves the "
                                      "zonal one inside the plausibility "
                                      "band at both extremes (P3^1.4 is "
                                      "raised to aa and sinks the result by "
                                      "4-7 orders).")
    c["rmz_tauscale"] = _tab_param(em, "tau_unit_scale_from_s", 1.0, model=z,
                                   why="τ in SECONDS (same finding as the "
                                       "global Rizk & Mongia one).")
    #  --- pantalla de plausibilidad del abanico ---------------------------
    c["ei_nox_min"] = _tab_param(em, "EI_NOx_min", EI_NOX_PLAUSIBLE_MIN,
                                 model="plausibility")
    c["ei_nox_max"] = _tab_param(em, "EI_NOx_max", EI_NOX_PLAUSIBLE_MAX,
                                 model="plausibility")
    _COEFFS = c
    return c


def _class_band(param: str, lo_default: float,
                hi_default: float) -> tuple[float, float]:
    """Envolvente (min de `min`, max de `max`) de un parámetro de
    `reference/class_table.csv` sobre las TRES clases de potencia.

    Se usa la envolvente y no la clase concreta porque Phy-CB no tiene
    todavía un mapeo potencia→clase declarado (la tabla indexa por potencia
    de eje y el L0 conoce la potencia TÉRMICA): elegir la clase «a ojo»
    sería un ajuste encubierto. Nunca lanza.
    """
    t = load_table("reference/class_table.csv")
    if t is not None:
        cols = t["cols"]
        pa, mn, mx = cols.get("param"), cols.get("min"), cols.get("max")
        if pa is not None and mn is not None and mx is not None:
            m = np.asarray(pa, dtype=str) == param
            lo = np.asarray(mn, dtype=float)[m]
            hi = np.asarray(mx, dtype=float)[m]
            lo = lo[np.isfinite(lo)]
            hi = hi[np.isfinite(hi)]
            if lo.size and hi.size:
                return float(np.min(lo)), float(np.max(hi))
    _warn_provisional(
        f"missing:class_table:{param}",
        f"reference/class_table.csv -> {param} not available: the "
        f"provisional band ({lo_default:g}, {hi_default:g}) of physics_core "
        f"is used.")
    return float(lo_default), float(hi_default)


def _tab_param_scheme(rel: str, scheme: str, param: str,
                      default: float) -> float:
    """Valor de `film_effusion.csv` (formato largo con columna `scheme`)."""
    t = load_table(rel)
    if t is not None:
        cols = t["cols"]
        sc, pa, va = cols.get("scheme"), cols.get("param"), cols.get("value")
        if sc is not None and pa is not None and va is not None \
                and va.dtype.kind in "fiu":
            m = (np.asarray(sc, dtype=str) == scheme) \
                & (np.asarray(pa, dtype=str) == param)
            idx = np.nonzero(m)[0]
            if idx.size:
                v = float(va[int(idx[0])])
                if np.isfinite(v):
                    return v
    _warn_provisional(f"missing:{rel}:{scheme}.{param}",
                      f"{rel} -> {scheme}.{param} not available: provisional "
                      f"{default:.6g}.")
    return float(default)


def _load_effusion_map(rel: str) -> dict:
    """Mapa η_film(M) de efusión cilíndrica al ángulo de diseño.

    Devuelve `{"M": array, "eta": array, "status": str}`; si la tabla falta
    se emite la forma exponencial saturada de respaldo (documentada en
    `data_pipeline.EFFUSION_*`) y se registra el aviso.
    """
    t = load_table(rel)
    if t is not None:
        cols = t["cols"]
        sc, pa = cols.get("scheme"), cols.get("param")
        Mc, ac, vc = cols.get("M"), cols.get("angle_deg"), cols.get("value")
        if all(x is not None for x in (sc, pa, Mc, ac, vc)):
            m = (np.asarray(sc, dtype=str) == "effusion_cylindrical") \
                & (np.asarray(pa, dtype=str) == "eta_cool") \
                & (np.asarray(ac, dtype=float) == 30.0)
            if m.any():
                M = np.asarray(Mc, dtype=float)[m]
                e = np.asarray(vc, dtype=float)[m]
                o = np.argsort(M)
                _warn_provisional(
                    "effusion_map",
                    "the effusion η_film(M) map of film_effusion.csv is "
                    "PROVISIONAL: no closed published correlation exists "
                    "(gap declared in Ciencia §5.1).")
                return {"M": M[o], "eta": e[o], "status": "provisional"}
    M = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00])
    e = 0.72 * (1.0 - np.exp(-M / 0.22))
    _warn_provisional("effusion_map_fallback",
                      "film_effusion.csv missing: effusion η_film falls back "
                      "to the saturated exponential form.")
    return {"M": M, "eta": e, "status": "provisional"}


# ==========================================================================
# 5. Propiedades de gas (funciones puras, sin tablas — Sutherland + Pr)
# ==========================================================================
def _mu_gas(T: float) -> float:
    """Viscosidad dinámica del aire/gas quemado [Pa·s] — ley de Sutherland.

    μ = μ₀ (T/T₀)^1.5 (T₀+S)/(T+S) con μ₀ = 1.716e-5 Pa·s, T₀ = 273.15 K,
    S = 110.4 K (constantes clásicas del aire; el gas quemado difiere <10 %
    a estas temperaturas, dentro del error de las correlaciones que la
    consumen). Clampeada para θ degenerado.
    """
    T = float(min(max(T, 100.0), 3500.0))
    return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)


def _k_gas(T: float, cp: float = CP) -> float:
    """Conductividad térmica [W/m/K] desde μ y Pr (k = μ·c_p/Pr)."""
    return _mu_gas(T) * cp / PRANDTL_GAS


def _r_gas(mw: float) -> float:
    """Constante del gas [J/kg/K] desde la masa molar [kg/kmol]."""
    return 8314.462618 / max(float(mw), 2.0)


# ==========================================================================
# 6. Termoquímica L0 [Ciencia §9.3] y FAR del ciclo [CONTEXTO §4]
# ==========================================================================
_TAD_CACHE: dict[str, dict | None] = {}
_CT_CACHE: dict[str, object] = {}
_CANTERA_TRIED = False
_CANTERA_OK = False


def _cantera():
    """Import perezoso de Cantera. Devuelve el módulo o `None`; nunca lanza.

    Nota de rendimiento (Ciencia §9.3 + objetivo de ~1 ms/punto): el camino
    caliente de L0 usa SIEMPRE la tabla `data/thermo/tad_fit_<fuel>.csv`
    (generada CON Cantera por la capa 0, error de interpolación < 1 %).
    Cantera solo se toca cuando se pide explícitamente `use_cantera=True`
    en `_thermo_hot`, y entonces el objeto `Solution` queda cacheado por
    combustible.
    """
    global _CANTERA_TRIED, _CANTERA_OK
    if _CANTERA_TRIED:
        return _CT_CACHE.get("module")
    _CANTERA_TRIED = True
    try:
        import cantera as ct
        _CT_CACHE["module"] = ct
        _CANTERA_OK = True
        return ct
    except Exception:
        _CT_CACHE["module"] = None
        return None


def cantera_available() -> bool:
    """True si Cantera es importable (informativo para la CLI/reporte)."""
    return _cantera() is not None


def _tad_table(fuel: str) -> dict | None:
    """Carga y estructura la tabla de T_ad de un combustible (cacheada).

    Devuelve `{phi, T3, P3(log), T_ad[3D], cp[3D], MW[3D], status,
    far_stoich}` o `None` si la tabla no está.
    """
    if fuel in _TAD_CACHE:
        return _TAD_CACHE[fuel]
    t = load_table(f"thermo/tad_fit_{fuel}.csv")
    out = None
    if t is not None:
        try:
            prov = t["provenance"]
            phi = np.array([float(x) for x in prov["grid_phi"].split()])
            t3 = np.array([float(x) for x in prov["grid_T3_K"].split()])
            p3 = np.array([float(x) for x in prov["grid_P3_Pa"].split()])
            shape = (phi.size, t3.size, p3.size)
            cols = t["cols"]
            n = int(np.prod(shape))
            if cols["T_ad_K"].size != n:
                raise ValueError("rejilla incoherente")
            out = {
                "phi": phi, "T3": t3, "logP3": np.log(p3),
                "T_ad": np.asarray(cols["T_ad_K"], float).reshape(shape),
                "cp": np.asarray(cols["cp_prod_J_kgK"], float).reshape(shape),
                "MW": np.asarray(cols["MW_prod_kgkmol"], float).reshape(shape),
                "status": t["status"],
                "far_stoich": float(prov.get("far_stoich_mass",
                                             FUELS[fuel]["far_stoich"])),
            }
            if t["status"] != "verified":
                _warn_provisional(
                    f"tad:{fuel}",
                    f"thermo/tad_fit_{fuel}.csv is marked '{t['status']}' "
                    f"(generated without Cantera): T_ad, c_p and MW of the "
                    f"burnt gas come from a fallback analytical model.")
        except Exception:
            out = None
    if out is None:
        _warn_provisional(
            f"tad_missing:{fuel}",
            f"thermo/tad_fit_{fuel}.csv missing: T_ad falls back to the "
            f"enthalpy balance with constant c_p (no dissociation) — it "
            f"overestimates T_ad above ~2000 K.")
    _TAD_CACHE[fuel] = out
    return out


def _trilinear(grid: np.ndarray, axes: tuple, vals: tuple) -> float:
    """Interpolación trilineal con CLAMP en los extremos (sin extrapolar).

    Misma convención que `data_pipeline._interp_tad` (que es quien mide el
    error de interpolación anotado en la cabecera del CSV): así el fallback
    y el error declarado hablan del mismo esquema.
    """
    idx, frac = [], []
    for ax, v in zip(axes, vals):
        i = int(np.clip(np.searchsorted(ax, v) - 1, 0, ax.size - 2))
        idx.append(i)
        f = (v - ax[i]) / (ax[i + 1] - ax[i])
        frac.append(float(min(max(f, 0.0), 1.0)))   # clamp = sin extrapolar
    out = 0.0
    for di, wa in ((0, 1.0 - frac[0]), (1, frac[0])):
        for dj, wb in ((0, 1.0 - frac[1]), (1, frac[1])):
            for dk, wc in ((0, 1.0 - frac[2]), (1, frac[2])):
                out += wa * wb * wc * grid[idx[0] + di, idx[1] + dj,
                                           idx[2] + dk]
    return float(out)


def _thermo_hot(fuel: str, phi: float, P3: float, T3: float,
                use_cantera: bool = False) -> dict:
    """Estado de equilibrio del gas quemado [Ciencia §9.3].

    Con Cantera (`use_cantera=True`) resuelve `gas.equilibrate('HP')` desde
    (φ, T₃, P₃) — el L0 termoquímico canónico, equivalente a NASA CEA con
    desvío verificado 0.06–3.5 % en T_ad. Sin Cantera (el camino por
    defecto, ~30 µs) interpola la tabla `data/thermo/tad_fit_<fuel>.csv`,
    que la capa 0 generó CON Cantera sobre una rejilla (φ, T₃, P₃) y cuyo
    error de interpolación viene anotado en la propia cabecera del CSV.

    Devuelve `dict(T_ad_K, cp_J_kgK, MW_kgkmol, R_J_kgK, far_stoich,
    source)`. `source` es la etiqueta que se propaga al record:
    `equilibrium_cantera` | `tad_table` | `tad_table(provisional)` |
    `enthalpy_balance(no_table)`. **Nunca lanza.**
    """
    fp = FUELS.get(fuel) or FUELS["jet_a"]
    phi = float(phi) if np.isfinite(phi) else 0.0
    phi = min(max(phi, 1.0e-6), 4.0)
    P3 = float(P3) if np.isfinite(P3) and P3 > 0.0 else P_ATM
    T3 = float(T3) if np.isfinite(T3) and T3 > 0.0 else 288.15

    if use_cantera:
        res = _thermo_hot_cantera(fuel, phi, P3, T3)
        if res is not None:
            return res

    tab = _tad_table(fuel)
    if tab is not None:
        axes = (tab["phi"], tab["T3"], tab["logP3"])
        vals = (phi, T3, math.log(P3))
        T_ad = _trilinear(tab["T_ad"], axes, vals)
        cp = _trilinear(tab["cp"], axes, vals)
        mw = _trilinear(tab["MW"], axes, vals)
        tag = "tad_table" if tab["status"] == "verified" \
            else "tad_table(provisional)"
        return dict(T_ad_K=T_ad, cp_J_kgK=cp, MW_kgkmol=mw,
                    R_J_kgK=_r_gas(mw), far_stoich=tab["far_stoich"],
                    source=tag)

    # Último recurso: balance de entalpía con c_p del gas caliente y sin
    # disociación. Sobreestima T_ad por encima de ~2000 K — declarado.
    far_st = float(fp["far_stoich"])
    far = phi * far_st
    burned = min(1.0, 1.0 / phi)
    q = (far / (1.0 + far)) * float(fp["LHV"]) * burned
    T_ad = T3 + q / CP_HOT
    mw = float(fp["mw_products"])
    return dict(T_ad_K=T_ad, cp_J_kgK=CP_HOT, MW_kgkmol=mw,
                R_J_kgK=_r_gas(mw), far_stoich=far_st,
                source="enthalpy_balance(no_table)")


def _thermo_hot_cantera(fuel: str, phi: float, P3: float,
                        T3: float) -> dict | None:
    """`gas.equilibrate('HP')` con el objeto `Solution` CACHEADO por
    combustible. Devuelve `None` (nunca lanza) si Cantera no está o falla.

    El mecanismo y el surrogate de Jet-A son exactamente los que usó la
    capa 0 para generar las tablas (`data_pipeline._ct_equilibrium_gas`),
    de modo que la rama con Cantera y la tabulada NO pueden divergir de
    modelo, solo de error de interpolación.
    """
    ct = _cantera()
    if ct is None:
        return None
    key = f"gas:{fuel}"
    if key not in _CT_CACHE:
        dp = _data_pipeline()
        setup = None
        if dp is not None:
            try:
                setup = dp._ct_equilibrium_gas(ct, fuel)
            except Exception:
                setup = None
        if setup is None:
            _CT_CACHE[key] = None
        else:
            gas, fuel_X, _desc = setup
            try:
                lhv = dp._ct_lhv(ct, gas, fuel_X)
                dh = lhv - float(FUELS[fuel]["LHV"])
            except Exception:
                dh = 0.0
            _CT_CACHE[key] = (gas, fuel_X, dh)
    entry = _CT_CACHE.get(key)
    if entry is None:
        return None
    gas, fuel_X, dh = entry
    try:
        gas.set_equivalence_ratio(phi, fuel_X, "O2:1.0, N2:3.76",
                                  basis="mole")
        gas.TP = T3, P3
        y_fuel = sum(float(gas.Y[gas.species_index(k)]) for k in fuel_X)
        gas.HP = gas.enthalpy_mass - y_fuel * dh, P3
        gas.equilibrate("HP")
        mw = float(gas.mean_molecular_weight)
        return dict(T_ad_K=float(gas.T), cp_J_kgK=float(gas.cp_mass),
                    MW_kgkmol=mw, R_J_kgK=_r_gas(mw),
                    far_stoich=float(FUELS[fuel]["far_stoich"]),
                    source="equilibrium_cantera")
    except Exception:
        return None


def _far_from_T4(fuel: str, T3: float, T4: float,
                 eta_comb: float = ETA_COMB_CYCLE) -> float:
    """FAR por balance de entalpía — **idéntico al del ciclo** [CONTEXTO §4].

    `fan_core.hot_cycle` de Phy-AC (y su reimplementación de
    `cycle_seam.py`) calcula

        f = (cph·T04 − cpc·T03) / max(η_b·LHV − cph·T04, 1e3)

    con `cph = CP_HOT = 1148`, `cpc = CP = 1005` y `η_b = ETA_COMB = 0.995`.
    Esta función reproduce ESA expresión carácter a carácter (mismo `max`,
    mismo suelo 1e3) para que la integración de WAVE4 no introduzca ni un
    ULP de diferencia: es la condición del test T8 (<1e-10).

    El único grado de libertad es `eta_comb`: llamándola con la η resuelta
    por Phy-CB en vez del relleno constante 0.995 se obtiene el FAR real
    del combustor diseñado, que es justo lo que el seam de ciclo va a
    consumir. Con el default reproduce el ciclo de referencia.
    """
    cph, cpc = CP_HOT, CP
    LHV = float((FUELS.get(fuel) or FUELS["jet_a"])["LHV"])
    T03, T04 = float(T3), float(T4)
    return (cph * T04 - cpc * T03) / max(eta_comb * LHV - cph * T04, 1e3)


# ==========================================================================
# 7. Dimensionado [Ciencia §2]
# ==========================================================================
def _eta_theta(theta_si: float) -> float:
    """η_θ(θ) por interpolación lineal en log θ sobre `eta_vs_theta.csv`."""
    c = coeffs()
    x = math.log(max(float(theta_si), 1.0))
    return float(np.interp(x, c["eta_theta_x"], c["eta_theta_y"]))


def _theta_required(eta_target: float) -> float:
    """θ mínimo que la carta η_c(θ) asocia a `eta_target` (inversa)."""
    c = coeffs()
    y, x = c["eta_theta_y"], c["eta_theta_x"]
    e = float(min(max(eta_target, float(y[0])), float(y[-1])))
    return float(math.exp(float(np.interp(e, y, x))))


def _d_casing(arch: str, A_ref: float) -> float:
    """Ø exterior de carcasa [m] desde el área de referencia.

    * `can`: la carcasa ES el círculo de área A_ref.
    * `annular` / `reverse_annular`: la sección de referencia es una corona
      de relación cubo/punta `ANNULUS_HTR` alrededor del eje/turbina, luego
      A_ref = (π/4)·D_cas²·(1 − HTR²).
    """
    A = max(float(A_ref), 1e-9)
    if arch == "can":
        return math.sqrt(4.0 * A / math.pi)
    return math.sqrt(4.0 * A / (math.pi * max(1.0 - ANNULUS_HTR ** 2, 1e-3)))


def _sizing(theta: np.ndarray, gates: dict, mdot_air: float,
            mdot_fuel: float, dp_over_p: float) -> dict:
    """Área de referencia, geometría, longitudes de zona y tiempos [§2].

    A_ref = **max(criterio aerodinámico, criterio químico θ)** [§2.1–2.2]:

    * Aerodinámico (✅ forma estándar):
      `A_ref = [ (R/2)·(ṁ₃√T₃/P₃)²·(ΔP₃₋₄/q_ref)/(ΔP₃₋₄/P₃) ]^0.5`.
      En microturbinas Δp/p es RESTRICCIÓN del ciclo (2–4 %), no salida:
      entra como dato y el área sale de él.
    * Químico: θ = P₃^1.75·A_ref·D_ref^0.75·exp(T₃/b)/ṁ₃ debe alcanzar el
      θ que la carta η_c(θ) asocia a `ETA_SIZING_TARGET`. Como D_ref
      depende de A_ref, la inversión es un PUNTO FIJO: se resuelve con un
      bucle ACOTADO de `SIZING_ITERS` iteraciones y test de convergencia
      relativo (`SIZING_TOL`), nunca con un `while` abierto. El mapa es
      contractivo (D_ref ∝ A^0.5 entra con exponente 0.75 ⇒ A^0.375).

    DESVIACIÓN DECLARADA respecto a Ciencia §2.2: el D_ref del parámetro θ
    se toma como el **diámetro equivalente** √(4A_ref/π) también en las
    arquitecturas anulares, en vez de la altura de la corona. Motivo: el
    único punto de anclaje de la carta η_c(θ) (θ = 7e7 SI ↔ η = 99.9 %) no
    declara arquitectura, y mezclar la convención «altura de corona» con un
    θ_req calibrado sobre tubular duplica A_ref y saca V_c, τ y Ø de
    carcasa de TODAS las bandas de `reference/class_table.csv` para la
    clase micro. La altura de la corona sí se usa para la carcasa
    (`_d_casing`). Cerrar cuando se digitalice la carta real (WAVE1 §0).

    Devuelve un dict con todas las claves de dimensionado del contrato §4.2
    más `sizing_driver` ∈ {"aero", "chem"}.
    """
    c = coeffs()
    arch = gates["arch"]
    P3 = float(theta[IDX_P3_PA])
    T3 = float(theta[IDX_T3_K])
    rho3 = P3 / (RGAS * T3)
    mdot3 = max(float(mdot_air), 1e-9)
    al_over_aref = float(min(max(theta[1], 0.30), 0.95))

    # --- criterio aerodinámico -------------------------------------------
    dpp = float(min(max(dp_over_p, 1e-4), 0.30))
    A_aero = math.sqrt(max(
        0.5 * RGAS * (mdot3 * math.sqrt(T3) / P3) ** 2 * c["PLF"] / dpp,
        1e-12))

    # --- criterio químico (punto fijo ACOTADO) ---------------------------
    theta_req = _theta_required(ETA_SIZING_TARGET)
    k_theta = (P3 ** 1.75) * math.exp(T3 / THETA_B_K) / mdot3
    A_chem = A_aero          # arranque: el aerodinámico es buen semilla
    converged = False
    for _ in range(SIZING_ITERS):
        D_ref_k = math.sqrt(4.0 * A_chem / math.pi)
        A_new = theta_req / max(k_theta * D_ref_k ** 0.75, 1e-30)
        A_new = float(min(max(A_new, 1e-9), 10.0))
        if abs(A_new - A_chem) <= SIZING_TOL * max(A_new, 1e-12):
            A_chem = A_new
            converged = True
            break
        A_chem = 0.5 * (A_chem + A_new)     # sub-relajación (robustez)
    A_ref = max(A_aero, A_chem)
    driver = "aero" if A_aero >= A_chem else "chem"

    # --- geometría de referencia -----------------------------------------
    D_ref = math.sqrt(4.0 * A_ref / math.pi)
    D_casing = _d_casing(arch, A_ref)
    A_L = al_over_aref * A_ref
    D_L = math.sqrt(4.0 * A_L / math.pi)
    # --- corona liner↔carcasa: GEOMÉTRICA, no A_ref − A_L -----------------
    #
    # Era `A_an = A_ref − A_L`, y es un atajo que NO casa con la geometría que
    # se construye. Medido en el ancla el 2026-08-01:
    #
    #     A_ref − A_L                      = 1000.6 mm²   ⇒ U_an = 64.1 m/s
    #     π·(r_cas² − r_lo²) (lo que se construye) = 800.4 mm²  ⇒ U_an = 80.2 m/s
    #
    # ×1.25 de área, ×1.56 de cabeza dinámica en el lado frío. La causa es que
    # `A_ref` sale del criterio de dimensionado (Ciencia §2.1) y `D_casing` se
    # deriva aparte, así que `A_ref − A_L` no es la corona de NINGÚN sólido:
    # es la resta de dos áreas que viven en planos distintos. La misma familia
    # que el bug de la efusión (§13.b) — un atajo analítico contra el perfil
    # real — y por eso se arregla igual: con la constante COMPARTIDA
    # `LINER_R_OVER_RCAS`, que es la que `geometry_generator._annulus_geometry`
    # y `effusion_wall_area_m2` ya usan para construir la corona.
    #
    # `A_an` alimenta la convección externa del térmico 1D (C₂, Ciencia §5):
    # una corona más estrecha ⇒ más flujo másico por unidad de área ⇒ MÁS
    # refrigeración del lado frío. El cambio baja T_w2, no la sube.
    r_cas_m = 0.5 * D_casing
    if gates["arch"] == "can":
        A_an = max(0.25 * math.pi * (D_casing ** 2 - D_L ** 2), 1e-6)
    else:
        r_lo_m = LINER_R_OVER_RCAS * r_cas_m
        A_an = max(math.pi * (r_cas_m ** 2 - r_lo_m ** 2), 1e-6)
    A_an_nominal = max(A_ref - A_L, 1e-6)         # el atajo, solo para el aviso
    if abs(A_an_nominal - A_an) > ANNULUS_AREA_REL_TOL * A_an:
        _warn_provisional(
            "annulus_area_shortcut",
            f"liner<->casing annulus: the geometric one is "
            f"{1e6 * A_an:.1f} mm² and the A_ref−A_L shortcut gives "
            f"{1e6 * A_an_nominal:.1f} mm² (x{A_an_nominal / A_an:.2f}). The "
            f"GEOMETRIC one is used, which is the one that gets "
            "manufactured. It affects the external convection of the 1-D "
            "thermal.")
    D_an = 4.0 * A_an / max(math.pi * (D_casing + D_L), 1e-9)   # Ø hidráulico
    U_ref = mdot3 / max(rho3 * A_ref, 1e-9)
    q_ref = 0.5 * rho3 * U_ref ** 2

    # --- longitudes de zona [§2.5] ---------------------------------------
    L_pz = float(min(max(theta[12], 0.20), 2.0)) * D_L
    L_sz = c["L_sz_over_DL"].get(arch, 0.50) * D_L
    L_dz = float(min(max(theta[13], 0.20), 5.0)) * D_L
    L_tot = L_pz + L_sz + L_dz

    V_pz, V_sz, V_dz = A_L * L_pz, A_L * L_sz, A_L * L_dz
    V_c = V_pz + V_sz + V_dz

    # --- tiempos, intensidad y cargas [§2.4] ------------------------------
    tau = V_c * rho3 / mdot3
    theta_lefebvre = k_theta * A_ref * D_ref ** 0.75
    Omega = mdot3 / max((P3 ** 1.8) * V_c * math.exp(T3 / THETA_B_K), 1e-30)
    #  INTENSIDAD DE COMBUSTIÓN — auditoría 2026-07-30 [Ciencia §2.4]
    #      I = ṁ_f·LHV/(V_c·P₃)  [MW/m³·atm]
    #  P₃ va en **atm** (`P3/P_ATM`) y el numerador en W: es la única
    #  combinación que da MW/m³·atm, y está bien desde el principio. El
    #  ancla sale en 126 MW/m³·atm, 2.1× por encima de la banda 20–60 de
    #  Ciencia §2.4 — y NO es un error de unidades, es una identidad:
    #      I = FAR·LHV·P_atm/(R·T₃·τ)
    #  (sustituyendo V_c = τ·ṁ₃/ρ₃ y ρ₃ = P₃/(R·T₃): V_c y P₃ se cancelan).
    #  Con el FAR del ancla (0.0216, impuesto por T₄ = 1150 K desde
    #  T₃ = 430 K) y τ = 6.05 ms —que está DENTRO de la banda 5–10 ms—, I no
    #  puede valer otra cosa. Las dos bandas de Ciencia §2.4 (I = 20–60 y
    #  τ = 5–10 ms) solo son compatibles entre sí para FAR ≲ 0.014, es
    #  decir, para saltos de temperatura bastante menores. Se reporta el
    #  valor real, se declara la banda y se enciende la restricción BLANDA
    #  `g_soft["I_band"]`; no se falsea el número.
    I_MW = (max(mdot_fuel, 0.0) * float(FUELS[gates["fuel"]]["LHV"])
            / max(V_c * (P3 / P_ATM), 1e-12)) / 1.0e6

    return dict(
        rho3_kgm3=rho3, A_ref_m2=A_ref, A_ref_aero_m2=A_aero,
        A_ref_chem_m2=A_chem, sizing_driver=driver,
        sizing_converged=bool(converged), theta_req_SI=theta_req,
        D_ref_mm=1e3 * D_ref, D_L_mm=1e3 * D_L, D_casing_mm=1e3 * D_casing,
        A_L_m2=A_L, A_an_m2=A_an, D_an_m=D_an,
        U_ref_ms=U_ref, q_ref_Pa=q_ref,
        L_pz_mm=1e3 * L_pz, L_sz_mm=1e3 * L_sz, L_dz_mm=1e3 * L_dz,
        L_tot_mm=1e3 * L_tot,
        V_c_m3=V_c, V_pz_m3=V_pz, V_sz_m3=V_sz, V_dz_m3=V_dz,
        tau_ms=1e3 * tau, I_MW_m3_atm=I_MW,
        theta_lefebvre=theta_lefebvre, Omega_load=Omega,
        _D_L=D_L, _L_tot=L_tot, _A_L=A_L, _A_ref=A_ref)


# ==========================================================================
# 8. Reparto de aire y riquezas de zona [Ciencia §3]
# ==========================================================================
def _air_partition(theta: np.ndarray, far_global: float,
                   far_stoich: float) -> dict:
    """Fracciones de aire por zona y φ por zona, con CIERRE exacto.

    `frac_cool = 1 − (frac_pz + frac_sz + frac_dz)` es cierre, no variable
    (contrato §1). Si un θ degenerado pide más aire del que hay, las tres
    fracciones se reescalan proporcionalmente dejando `FRAC_COOL_MIN` de
    refrigeración; el cierre se calcula SIEMPRE al final, de modo que
    Σ splits = 1 exacto a la precisión de la máquina (test T7: 1e-12).

    Riquezas (acumulativas aguas abajo, como el aire real):
      φ_PZ  = FAR_global / (frac_pz · FAR_st)
      φ_SZ  = FAR_global / ((frac_pz+frac_sz) · FAR_st)
      φ_DZ  = φ_global = FAR_global / FAR_st
    Bandas de referencia [§3]: φ_PZ 0.85–1.15 (micro), φ_SZ 0.6–0.8,
    φ_global 0.15–0.30. La tabla `data/lefebvre/air_split_by_arch.csv`
    (fila `microturbine`) es el ancla contra la que se valida el reparto.
    """
    def _f(x, lo=0.0, hi=1.0):
        v = float(x)
        if not np.isfinite(v):
            return lo
        return float(min(max(v, lo), hi))

    pz, sz, dz = _f(theta[2]), _f(theta[3]), _f(theta[4])
    s = pz + sz + dz
    s_max = 1.0 - FRAC_COOL_MIN
    if s > s_max:
        k = s_max / max(s, 1e-12)
        pz, sz, dz = pz * k, sz * k, dz * k
    cool = 1.0 - (pz + sz + dz)          # CIERRE — nunca una variable
    far_st = max(float(far_stoich), 1e-6)
    fg = max(float(far_global), 0.0)
    phi_pz = fg / max(pz * far_st, 1e-9)
    phi_sz = fg / max((pz + sz) * far_st, 1e-9)
    phi_global = fg / far_st
    return dict(splits=dict(pz=pz, sz=sz, dz=dz, cool=cool),
                frac_cool=cool, phi_pz=phi_pz, phi_sz=phi_sz,
                phi_dz=phi_global, phi_global=phi_global)


def air_split_reference(regime: str = "microturbine") -> dict | None:
    """Reparto nominal de `air_split_by_arch.csv` (ancla de validación).

    Lo consumen la CLI (modo «reglas nominales», WAVE1 §3) y el test T7.
    Devuelve `None` si la tabla no está.
    """
    t = load_table("lefebvre/air_split_by_arch.csv")
    if t is None or "regime" not in t["cols"]:
        return None
    idx = np.nonzero(np.asarray(t["cols"]["regime"], dtype=str) == regime)[0]
    if not idx.size:
        return None
    i = int(idx[0])
    out = {}
    for zone in ("pz", "sz", "dz", "cool"):
        col = t["cols"].get(f"frac_{zone}_nom")
        if col is None:
            return None
        out[zone] = float(col[i])
    return out


# ==========================================================================
# 9. Swirler, orificios, penetración y cierre de pérdidas [Ciencia §4]
# ==========================================================================
def _swirler_hub_ratio(SN: float, beta_rad: float) -> float:
    """D_hub/D_sw que produce el número de swirl pedido [Ciencia §4.4].

    S_N = (2/3)·[(1−x³)/(1−x²)]·tanβ con x = D_hub/D_sw. El factor entre
    corchetes crece monótonamente de 1 (x→0) a 1.5 (x→1), así que la
    inversión se resuelve por BISECCIÓN ACOTADA (40 pasos ⇒ 1e-12) y se
    clampa cuando el S_N pedido cae fuera del rango alcanzable con ese β.
    """
    tb = math.tan(beta_rad)
    if tb <= 1e-6:
        return 0.5
    target = 1.5 * float(SN) / tb          # = (1−x³)/(1−x²)
    if target <= 1.0:
        return 0.0
    if target >= 1.5:
        return 0.999
    lo, hi = 0.0, 0.999
    for _ in range(40):                    # bucle ACOTADO
        mid = 0.5 * (lo + hi)
        f = (1.0 - mid ** 3) / max(1.0 - mid ** 2, 1e-9)
        if f < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _swirler(theta: np.ndarray, sz: dict, q_ref: float) -> dict:
    """Geometría y pérdida del swirler [Ciencia §4.4].

    ΔP_sw/q_ref = K_sw·[(A_ref/(A_sw·secβ))² − (A_ref/A_L)²]  ⚠️ K_sw.
    S_N > 0.6 ⇒ zona de recirculación toroidal central (CTRZ), que es lo
    que ancla la llama: se reporta como bandera `ctrz`.
    """
    c = coeffs()
    SN = float(min(max(theta[5], 0.05), 3.0)) if np.isfinite(theta[5]) else 0.7
    beta = math.radians(SWIRL_VANE_BETA_DEG)
    A_ref, A_L = sz["_A_ref"], sz["_A_L"]
    A_sw = max(A_SW_OVER_AREF * A_ref, 1e-9)
    sec_b = 1.0 / max(math.cos(beta), 1e-6)
    term = (A_ref / (A_sw * sec_b)) ** 2 - (A_ref / max(A_L, 1e-9)) ** 2
    dp_sw_over_qref = max(c["K_sw"] * term, 0.0)
    return dict(SN=SN, beta_deg=SWIRL_VANE_BETA_DEG,
                hub_over_tip=_swirler_hub_ratio(SN, beta),
                A_sw_m2=A_sw, dp_sw_over_qref=dp_sw_over_qref,
                dp_sw_Pa=dp_sw_over_qref * q_ref, ctrz=bool(SN > 0.6))


def _r_liner_mm(gates: dict, sz: dict) -> float:
    """Radio de la superficie del liner donde se practican los orificios.

    En `can` es el propio radio del liner; en las anulares, el radio
    exterior de la corona menos la holgura al casing (`LINER_R_OVER_RCAS`,
    la MISMA constante con la que `effusion_wall_area_m2` y
    `geometry_generator._annulus_geometry` construyen la corona: era un 0.92
    escrito a mano aquí y otro allí, y esa duplicación es de la misma
    familia que el bug de la efusión). Solo alimenta la columna `r_mm` del
    contrato de geometría (WAVE3), no la física.
    """
    if gates["arch"] == "can":
        return 0.5 * sz["D_L_mm"]
    return 0.5 * sz["D_casing_mm"] * LINER_R_OVER_RCAS


def _holes(theta: np.ndarray, gates: dict, sz: dict, part: dict,
           mdot_air: float, mdot_fuel: float, T_zone: dict, mw_hot: float,
           dp_over_p: float) -> dict:
    """Filas de orificios, penetración de chorro y cierre de Δp [§4].

    Cadena:
      1. ΔP_total = (Δp/p)·P₃ (dato del ciclo, Ciencia §2.1) y
         ΔP_liner = `LINER_FRAC_OF_DP`·ΔP_total (70–80 % ⚠️).
      2. Velocidad de chorro U_j = √(2·ΔP_L/ρ₃) y área por fila desde
         ṁ_h = C_d·A_h·√(2ρ₃ΔP_L) [§4.1] ⇒ Ø de cada orificio con el
         número de orificios de θ (gates enteras ya decodificadas).
      3. Penetración Y_max/d_j = 1.15·√J con J = ρ_jU_j²/(ρ_gU_g²) ✅, y
         regla de mezcla óptima de Lefebvre & Norster 1968: Y ≈ 0.5·D_L
         (fila única) o 0.33·D_L (dos filas desfasadas).
      4. Cierre: Δp/p total y ΔP_L/q_ref (que alimentan PF y η_mix).

    El C_d lleva la corrección por rugosidad AM `f(Ra/d)` de
    `cd_am_roughness.csv` cuando la tabla está (⚠️ forma de la casa, no hay
    correlación cerrada publicada).
    """
    c = coeffs()
    P3 = float(theta[IDX_P3_PA])
    rho3 = sz["rho3_kgm3"]
    dp_Pa = float(min(max(dp_over_p, 1e-4), 0.30)) * P3
    dp_liner_Pa = LINER_FRAC_OF_DP * dp_Pa
    U_j = math.sqrt(max(2.0 * dp_liner_Pa / max(rho3, 1e-6), 1e-9))
    Cd = c["Cd_hole"]
    splits = part["splits"]
    D_L = sz["_D_L"]
    A_L = sz["_A_L"]
    R_hot = _r_gas(mw_hot)
    r_mm = _r_liner_mm(gates, sz)

    n_pz = int(min(max(round(float(theta[6])), 1), 200)) \
        if np.isfinite(theta[6]) else 8
    n_dz = int(min(max(round(float(theta[7])), 1), 200)) \
        if np.isfinite(theta[7]) else 10
    n_sz = n_pz          # las filas secundarias replican el conteo de PZ

    # Aire por fila: el swirler se lleva `SWIRLER_FRAC_OF_PZ` de la PZ.
    m_sw = SWIRLER_FRAC_OF_PZ * splits["pz"] * mdot_air
    rows_spec = [
        ("pz", n_pz, (1.0 - SWIRLER_FRAC_OF_PZ) * splits["pz"] * mdot_air,
         0.5 * sz["L_pz_mm"], T_zone["T_pz_K"],
         mdot_fuel + m_sw),
        ("sz", n_sz, splits["sz"] * mdot_air,
         sz["L_pz_mm"] + 0.5 * sz["L_sz_mm"], T_zone["T_pz_K"],
         mdot_fuel + splits["pz"] * mdot_air),
        ("dz", n_dz, splits["dz"] * mdot_air,
         sz["L_pz_mm"] + sz["L_sz_mm"], T_zone["T_sz_K"],
         mdot_fuel + (splits["pz"] + splits["sz"]) * mdot_air),
    ]
    hole_rows: list[dict] = []
    A_h_total = 0.0
    J_dz = Y_over_DL_dz = 0.0
    for zone, n, m_row, z_mm, T_g, m_gas in rows_spec:
        A_row = max(m_row, 0.0) / max(Cd * rho3 * U_j, 1e-12)
        d = math.sqrt(4.0 * A_row / (math.pi * n)) if A_row > 0.0 else 0.0
        rho_g = P3 / (R_hot * max(T_g, 200.0))
        U_g = max(m_gas, 1e-9) / max(rho_g * A_L, 1e-12)
        J = (rho3 * U_j ** 2) / max(rho_g * U_g ** 2, 1e-9)
        Y = c["jet_C"] * math.sqrt(max(J, 0.0)) * d
        A_h_total += A_row
        hole_rows.append(dict(
            zone=zone, z_mm=float(z_mm), r_mm=float(r_mm), n=int(n),
            d_mm=float(1e3 * d), Cd=float(Cd), J=float(J),
            Y_mm=float(1e3 * Y), Y_over_DL=float(Y / max(D_L, 1e-9)),
            A_row_m2=float(A_row), shape="circle", plunged=False,
            mdot_kgs=float(max(m_row, 0.0)), U_jet_ms=float(U_j)))
        if zone == "dz":
            J_dz, Y_over_DL_dz = J, Y / max(D_L, 1e-9)

    q_ref = max(sz["q_ref_Pa"], 1e-9)
    sw = _swirler(theta, sz, q_ref)
    return dict(hole_rows=hole_rows, A_h_total_m2=A_h_total,
                dp_liner_Pa=dp_liner_Pa, dp_Pa=dp_Pa,
                dp_over_p=dp_Pa / max(P3, 1.0),
                dp_liner_over_qref=dp_liner_Pa / q_ref,
                dp_sw_over_qref=sw["dp_sw_over_qref"],
                dp_sw_Pa=sw["dp_sw_Pa"], U_jet_ms=U_j,
                J_dz=J_dz, Y_over_DL_dz=Y_over_DL_dz,
                Y_target=Y_TARGET_DEFAULT, swirler=sw,
                mdot_swirler_kgs=m_sw)


# ==========================================================================
# 10. Atomización [Ciencia §7] — se reporta el ABANICO, no un número
# ==========================================================================
def _atomization(theta: np.ndarray, gates: dict, sz: dict, rho3: float,
                 dp_liner_Pa: float, mdot_fuel: float, T3: float) -> dict:
    """SMD por familia de inyector, según la gate `injector` [§7].

    * **airblast** (por defecto de Phy-CB): se usan las formas
      DIMENSIONALES ✅ VERIFICADAS contra arXiv:1906.04042 (Tabla 7) —
      Rizkalla & Lefebvre (1975) y El-Shanawany & Lefebvre (1980) — y NO
      la adimensional 0.33/0.068, que no pudo verificarse (se calcula
      igualmente y se reporta, pero no es el modelo elegido). El aire de
      atomización se acelera con la caída del liner: U_g = √(2ΔP_L/ρ₃),
      que en el caso ancla cae en la banda de diseño ✅ 60–100 m/s.
    * **simplex** [§7.1]: SMD = C·σ^0.25 μ_L^0.25 ṁ_L^0.25 ΔP_L^−0.5 ρ_A^−0.25
      ⚠️ (coef. 2.25). Dependencia ✅ SMD ∝ ΔP^−0.5.
    * **vaporizer** [§7.3]: reglas KJ66 ✅ (d₅₀ ≤ 50 µm, 6–12 tubos de
      ~64 mm con codo de 30°); no hay espray fino que correlacionar.
    * **micromix** H₂ [§7.4]: RAMA GAS. `SMD_um = None` (no `NaN`: el
      record se serializa a JSON y `NaN` no es JSON válido — decisión
      declarada del contrato §4). Se reportan Ø de orificio, nº de
      inyectores y J de penetración del chorro de hidrógeno.

    Cualquier combustible NO líquido (`FUELS[fuel]["liquid"] is False`)
    fuerza la rama gas aunque la gate pida un atomizador de líquido: el
    propano y el H₂ llegan al domo ya vaporizados a estas presiones.

    Devuelve `SMD_um_by_corr` con TODAS las correlaciones evaluadas (el
    usuario ve el abanico) + `atomizer_model`, `SMD_um` y `ALR_eff`.
    """
    c = coeffs()
    fuel = gates["fuel"]
    fp = FUELS[fuel]
    inj = gates["injector"]
    liquid = bool(fp["liquid"])
    sigma = float(fp["sigma"])
    rho_l = float(fp["rho_l"])
    mu_l = float(fp["mu_l"])
    mdot_f = max(float(mdot_fuel), 1e-9)
    U_g = max(math.sqrt(max(2.0 * dp_liner_Pa / max(rho3, 1e-6), 1e-9)),
              U_ATOMIZING_MIN_MS)
    D_p = max(D_PREFILMER_OVER_DL * sz["_D_L"], 1e-4)     # Ø prefilmer [m]
    ALR = float(min(max(theta[11], 0.10), 50.0)) \
        if np.isfinite(theta[11]) else 4.0
    mr = 1.0 / ALR                                        # ṁ_l/ṁ_g
    by_corr: dict[str, float | None] = {}

    # --- airblast prefilming: DIMENSIONALES ✅ (las de defecto) ----------
    esl = ((c["esl_C1"] * (sigma / max(rho3 * U_g ** 2, 1e-9)) ** 0.6
            * (rho_l / max(rho3, 1e-9)) ** 0.1 * D_p ** 0.4)
           + c["esl_C2"] * math.sqrt(mu_l ** 2 * D_p
                                     / max(sigma * rho_l, 1e-12))
           ) * (1.0 + mr)
    rl = (c["rl_C1"] * math.sqrt(sigma * rho_l * D_p)
          / max(rho3 * U_g, 1e-9) * (1.0 + mr)
          + c["rl_C2"] * (mu_l ** 2 / max(sigma * rho_l, 1e-12)) ** 0.425
          * D_p ** 0.575 * (1.0 + mr) ** 2)
    # --- airblast adimensional ⚠️ NO verificada (solo se reporta) --------
    we = rho3 * U_g ** 2 * D_p / max(sigma, 1e-9)
    nd = D_p * (c["nd_C1"] * (1.0 + mr) / max(we, 1e-9) ** 0.5
                + c["nd_C2"] * (mu_l ** 2 / max(sigma * rho_l * D_p, 1e-15))
                ** 0.5 * (1.0 + mr) ** 2)
    # --- plain-jet ⚠️ coeficientes no verificados ------------------------
    pj = D_p * (c["pj_C1"] / max(we, 1e-9) ** 0.5 * (1.0 + mr)
                + c["pj_C2"] * (mu_l ** 2 / max(sigma * rho_l * D_p, 1e-15))
                ** 0.5 * (1.0 + mr) ** 2)
    # --- Chaussonnet 2016 ✅ (extra útil a microescala) -------------------
    h_a = 0.10 * D_p
    ch = (1.69 * (0.70 / max(U_g, 1e-6))
          * math.sqrt(h_a * sigma / max(rho_l * rho3, 1e-12))
          * (math.sqrt(rho_l) + math.sqrt(rho3)))
    # --- simplex (presión-swirl) ⚠️ coef. 2.25 ---------------------------
    dP_f = DP_FUEL_SIMPLEX_PA
    sx = (c["smd_simplex_C"] * sigma ** 0.25 * mu_l ** 0.25 * mdot_f ** 0.25
          * dP_f ** -0.5 * max(rho3, 1e-6) ** -0.25)

    if liquid:
        by_corr["elshanawany_lefebvre"] = 1e6 * esl
        by_corr["rizkalla_lefebvre"] = 1e6 * rl
        by_corr["prefilm_nondimensional"] = 1e6 * nd
        by_corr["plain_jet_rizk_lefebvre"] = 1e6 * pj
        by_corr["chaussonnet_2016"] = 1e6 * ch
        by_corr["simplex_lefebvre"] = 1e6 * sx
        by_corr["vaporizer_kj66_d50"] = c["vap_d50_um"]

    extra: dict = {}
    if not liquid or inj == "micromix":
        # RAMA GAS: no hay espray. SMD = None (JSON `null`).
        model = "micromix_gas" if inj == "micromix" else "gaseous_fuel"
        smd = None
        power_W = mdot_f * float(fp["LHV"])
        n_inj = max(1, int(round(power_W / (c["mm_kW"] * 1000.0))))
        d_inj = c["mm_d_mm"] * 1e-3
        rho_f = float(P_ATM) / (_r_gas(2.016 if fuel == "h2" else 44.1)
                                * max(T3, 200.0)) if fuel == "h2" else rho3
        A_inj = n_inj * 0.25 * math.pi * d_inj ** 2
        U_f = mdot_f / max(rho_f * A_inj, 1e-12)
        J_gas = (rho_f * U_f ** 2) / max(rho3 * U_g ** 2, 1e-9)
        extra = dict(n_injectors=n_inj, d_injector_mm=c["mm_d_mm"],
                     J_fuel_jet=J_gas, U_fuel_ms=U_f,
                     y_penetration_mm=1e3 * c["jet_C"]
                     * math.sqrt(max(J_gas, 0.0)) * d_inj,
                     power_kW=power_W / 1e3)
        by_corr["gas_jet_no_smd"] = None
    elif inj == "airblast":
        model = "elshanawany_lefebvre"
        smd = 1e6 * esl
        extra = dict(D_prefilmer_mm=1e3 * D_p, U_atomizing_ms=U_g)
    elif inj == "simplex":
        model = "simplex_lefebvre"
        smd = 1e6 * sx
        extra = dict(dP_fuel_Pa=dP_f,
                     flow_number_m2=mdot_f / math.sqrt(max(rho_l * dP_f,
                                                           1e-9)))
    else:   # vaporizer
        model = "vaporizer_kj66"
        smd = c["vap_d50_um"]
        extra = dict(n_tubes=int(c["vap_n_tubes"]),
                     tube_length_mm=c["vap_L_mm"], elbow_deg=30.0)
    ALR_eff = ALR if (liquid and inj == "airblast") else float("nan")
    return dict(SMD_um=smd, SMD_um_by_corr=by_corr, atomizer_model=model,
                ALR_eff=(None if not np.isfinite(ALR_eff) else ALR_eff),
                U_atomizing_ms=U_g, atomizer=extra, is_gas_branch=not liquid
                or inj == "micromix")


# ==========================================================================
# 11. Eficiencia de combustión descompuesta [Ciencia §2.3]
# ==========================================================================
ETA_CEILING = 0.9995       # techo asintótico de η_comb (nunca llega a 1).
#                            = `ETA_THETA_CEILING` de data_pipeline.
K_MIX_DP = 0.30            # η_mix = 1 − exp(−K·ΔP_L/q_ref). ⚠️ FORMA DE LA
#                            CASA: Ciencia §2.3 declara η_mix = f(ΔP_L/q_ref,
#                            geometría de chorros) sin dar la función. K se
#                            eligió para que η_mix ≈ 0.99 en el punto de
#                            diseño clásico ΔP_L/q_ref ≈ 15 y caiga rápido
#                            por debajo de 8. RECALIBRAR con L1/L2.
K_MIX_PENETRATION = 0.50   # penalización cuadrática por desviación relativa
#                            de Y/D_L respecto al óptimo de Lefebvre &
#                            Norster. ⚠️ de la casa.
EVAP_MASS_EXP = 1.50       # ley D²: fracción evaporada en t = 1−(1−t/τ_e)^1.5


def _efficiency(sz: dict, part: dict, holes: dict, atom: dict,
                mdot_air: float, fuel: str) -> dict:
    """η_comb = min(η_θ, η_evap, η_mix) [Ciencia §2.3] + el driver.

    A escala micro **dominan evaporación y mezcla**, no la cinética: por eso
    Lefebvre descompone la eficiencia en tres mecanismos en serie y toma el
    mínimo (el proceso limitante manda). Se reporta cuál mandó
    (`eta_driver`), que es la información de diseño útil.

    * η_θ: carta η_c(θ) digitalizada en `data/lefebvre/eta_vs_theta.csv`
      ⚠️ (un solo punto de anclaje).
    * η_evap: ley D² — un gotón de D₃₂ tarda τ_e = D₃₂²/λ_eff en
      evaporarse y la masa evaporada en t vale 1−(1−t/τ_e)^{3/2}. Se
      compara con el tiempo de residencia de la ZONA PRIMARIA (es ahí donde
      hay que haber evaporado). Combustible gaseoso ⇒ η_evap = techo.
    * η_mix: crece con ΔP_L/q_ref (energía disponible para mezclar) y se
      penaliza cuando la penetración de los chorros de dilución se aleja
      del óptimo Y ≈ 0.5·D_L.
    """
    fp = FUELS[fuel]
    eta_theta = min(_eta_theta(sz["theta_lefebvre"]), ETA_CEILING)

    # --- evaporación ------------------------------------------------------
    mdot_pz = max(part["splits"]["pz"] * mdot_air, 1e-9)
    tau_pz = sz["V_pz_m3"] * sz["rho3_kgm3"] / mdot_pz
    smd = atom.get("SMD_um")
    if atom.get("is_gas_branch") or not fp["liquid"] or smd is None:
        eta_evap = ETA_CEILING
        tau_e = 0.0
    else:
        tau_e = (float(smd) * 1e-6) ** 2 / max(float(fp["lambda_evap"]), 1e-12)
        if tau_pz >= tau_e:
            eta_evap = ETA_CEILING
        else:
            eta_evap = ETA_CEILING * (
                1.0 - (1.0 - tau_pz / max(tau_e, 1e-12)) ** EVAP_MASS_EXP)

    # --- mezcla -----------------------------------------------------------
    dpq = max(holes["dp_liner_over_qref"], 0.0)
    miss = abs(holes["Y_over_DL_dz"] - holes["Y_target"]) \
        / max(holes["Y_target"], 1e-6)
    eta_mix = ETA_CEILING * (1.0 - math.exp(-K_MIX_DP * dpq)) \
        * max(1.0 - K_MIX_PENETRATION * min(miss, 3.0) ** 2, 0.05)

    trio = {"theta": eta_theta, "evap": eta_evap, "mix": eta_mix}
    driver = min(trio, key=lambda k: trio[k])
    return dict(eta_theta=eta_theta, eta_evap=eta_evap, eta_mix=eta_mix,
                eta_comb=float(min(max(trio[driver], 0.05), ETA_CEILING)),
                eta_driver=driver, tau_pz_ms=1e3 * tau_pz,
                tau_evap_ms=1e3 * tau_e)


# ==========================================================================
# 12. Estabilidad: LBO y encendido [Ciencia §6]
# ==========================================================================
def _stability(theta: np.ndarray, gates: dict, sz: dict, part: dict,
               atom: dict, mdot_air: float, T_st: float,
               mw_hot: float) -> dict:
    """φ_LBO, margen, energía mínima de chispa y distancia de apagado.

    **LBO** [§6.1] — estructura funcional ✅, constante global ⚠️ (±45 %):

        φ_LBO = C · (A_ref/V_PZ) · ṁ_A/(P₃^1.3·e^{T₃/300}) · (D₃₂²/λ_eff)/LHV

    Lecturas de diseño que el modelo conserva por construcción: P₃^−1.3 (la
    estabilidad se degrada a baja presión — crítico a π≈3) y D₃₂² (la
    atomización es LA palanca a escala pequeña).

    **Combustible gaseoso**: el término de evaporación D₃₂²/λ_eff NO
    contribuye (no hay gotas). Se sustituye por la constante equivalente
    `C_gas = C·(D₃₂²/λ)_ref·LBO_GAS_EASE`, que conserva las unidades y
    reconoce que un gas estabiliza mejor. ⚠️ calibración de la casa.

    **Encendido** [§6.2] — Ballal & Lefebvre, estructura ✅:
        E_min = (π/6)·ρ_g·c_p·(T_st − T₀)·d_q³,   d_q = Pe·α/S_L
    En espray d_q se satura por el tamaño de gota (`SPRAY_DQ_OVER_SMD`·D₃₂),
    que es de donde sale la dependencia ✅ E_min ∝ D₃₂³ («un factor 2 en SMD
    son 8 en energía de chispa»).
    """
    c = coeffs()
    fuel = gates["fuel"]
    fp = FUELS[fuel]
    P3 = float(theta[IDX_P3_PA])
    T3 = float(theta[IDX_T3_K])
    rho3 = sz["rho3_kgm3"]
    LHV = float(fp["LHV"])
    A_over_V = sz["_A_ref"] / max(sz["V_pz_m3"], 1e-12)
    press_term = max(mdot_air, 1e-9) / max(
        P3 ** (-c["lbo_exp_P3"]) * math.exp(T3 / c["lbo_T3_scale"]), 1e-30)

    smd = atom.get("SMD_um")
    gas_branch = bool(atom.get("is_gas_branch")) or smd is None \
        or not fp["liquid"]
    if gas_branch:
        evap_term = LBO_EVAP_REF * LBO_GAS_EASE
        smd_m = 0.0
    else:
        smd_m = float(smd) * 1e-6
        evap_term = smd_m ** c["lbo_exp_SMD"] \
            / max(float(fp["lambda_evap"]), 1e-12)
    phi_lbo = c["C_LBO"] * A_over_V * press_term * evap_term / LHV
    phi_lbo = float(min(max(phi_lbo, 1e-4), 10.0))
    phi_pz = max(part["phi_pz"], 1e-6)
    #  TRES formas del margen — ver `LBO_MARGIN_DEF`. `LBO_margin` conserva
    #  su definición original (exceso relativo) porque es ABI: la consume
    #  g[2] y las anclas de regresión. Las otras dos se AÑADEN porque un
    #  «margen» que vale 0.998 se lee mal: no es «al 99.8 % del límite», es
    #  «φ_PZ es 2.0 veces φ_LBO», o sea que se puede perder el 50 % de la
    #  riqueza de la zona primaria antes de apagarse.
    margin = phi_pz / phi_lbo - 1.0

    # --- encendido --------------------------------------------------------
    mu3 = _mu_gas(T3)
    alpha = _k_gas(T3, CP) / max(rho3 * CP, 1e-9)          # difusividad [m²/s]
    S_L = (float(fp["S_L_ref"]) * (T3 / 298.0) ** S_L_EXP_T
           * (P3 / P_ATM) ** S_L_EXP_P)
    d_q_gas = QUENCH_PECLET * alpha / max(S_L, 1e-3)
    d_q = max(d_q_gas, SPRAY_DQ_OVER_SMD * smd_m)
    # ρ y c_p son los de la MEZCLA FRESCA que la chispa tiene que calentar
    # hasta T_st (no los del gas ya quemado): ese es el sentido del término
    # ρ·c_p·(T_st − T₀) de Ballal & Lefebvre.
    E_min = (math.pi / 6.0) * rho3 * CP * max(T_st - T3, 1.0) * d_q ** 3
    return dict(phi_LBO=phi_lbo, LBO_margin=float(margin),
                LBO_margin_ratio=float(phi_pz / phi_lbo),
                LBO_margin_frac=float(1.0 - phi_lbo / phi_pz),
                LBO_margin_def=LBO_MARGIN_DEF,
                E_min_J=float(E_min), d_q_mm=1e3 * d_q,
                S_L_ms=float(S_L), mu3_Pas=mu3,
                lbo_branch="gas" if gas_branch else "spray")


# ==========================================================================
# 12.b Efusión — FUENTE ÚNICA del área, del patrón y del C_d
#      [Ciencia §4.1 (C_d y rugosidad AM) y §5.1 (película)]
# ==========================================================================
#  POR QUÉ EXISTE ESTA SECCIÓN (auditoría 2026-07-31)
#  ---------------------------------------------------
#  Cuatro módulos hablaban de la MISMA efusión y no coincidían:
#
#    * `_liner_thermal_1d`  aplicaba la porosidad sobre `π·D_L·L_tot`,
#    * `geometry_generator._effusion` la aplicaba sobre la superficie REAL
#      que integra la polilínea del liner,
#    * `crn_core` usaba `C_d_tabla · CD_EFFUSION_INCLINED · f(Ra/d)`,
#    * `cfd/xlb_cold` usaba `C_d_tabla · CD_EFFUSION_INCLINED` (sin f(Ra/d)).
#
#  En el ancla REF_CB10 eso daba ×1.375 de área y ×1.118 de C_d, es decir
#  ×1.536 de conductancia C_d·A al MISMO Δp para el MISMO agujero.
#
#  CAUSA RAÍZ DEL ×1.375 — no era la longitud ni el redondeo
#  ---------------------------------------------------------
#  `D_L = √(4·A_L/π)` es el diámetro EQUIVALENTE EN ÁREA del paso del liner,
#  y `π·D_L` solo es el perímetro mojado cuando el paso ES un círculo, o
#  sea, en `can`. En una corona `A_L = π(r_lo² − r_li²)`, así que
#
#      π·D_L = 2π·√(r_lo² − r_li²)   <   2π·r_lo = perímetro real exterior
#
#  y el cociente es EXACTAMENTE `r_lo/√(r_lo² − r_li²)` — 1.3745 con los
#  radios del ancla (r_lo = 37.470 mm, r_li = 25.707 mm). Además `r_lo` NO
#  sale de `A_L`: sale de la CARCASA (`LINER_R_OVER_RCAS·r_carcasa`). El
#  círculo de área equivalente es la figura de perímetro MÍNIMO, así que el
#  error era sistemático y siempre en el mismo sentido (área mojada de menos
#  ⇒ menos orificios, menos área abierta, más flujo por orificio, M de más).
#  Comprobado: en `can` el cociente es 1.0000 exacto, y el tramo axial
#  (`z_chamber` de la geometría) coincide con `L_tot` en las tres
#  arquitecturas — ni la longitud ni `prof[k][2]` eran el problema.
def am_roughness_factor(d_hole_m: float) -> float:
    """Factor f(Ra/d) de rugosidad as-built sobre el C_d ideal [Ciencia §4.1].

    Ciencia §4.1 es literal: en los orificios de aire del liner la rugosidad
    AM **reduce** C_d un 5–20 % frente a pulido. Por eso este factor es
    OBLIGATORIO en cualquier consumidor del C_d de un orificio impreso — el
    que estaba mal era `cfd/xlb_cold`, que no lo aplicaba, no `crn_core`.

    Interpolación LINEAL con CLAMP en los extremos (la propia tabla
    `lefebvre/cd_am_roughness.csv` lo pide) sobre `Ra/d` con
    `Ra = RA_ASBUILT_UM`. `d_hole_m` en metros. Nunca lanza.

    ⚠️ FORMA DE LA CASA: no hay correlación cerrada publicada; la tabla está
    calibrada solo a la banda cualitativa de Ciencia §4.1.

    VIVÍA EN `crn_core._f_am_roughness`; se movió aquí el 2026-07-31 para que
    no hubiera dos copias. `crn_core._f_am_roughness` delega en esta.
    """
    d = max(_safe(d_hole_m, 1.0e-3), 1.0e-5)
    ra_over_d = (RA_ASBUILT_UM * 1.0e-6) / d
    t = load_table("lefebvre/cd_am_roughness.csv")
    if t is not None and "Ra_over_d" in t["cols"] and "f_Cd" in t["cols"]:
        x = np.asarray(t["cols"]["Ra_over_d"], dtype=float)
        f = np.asarray(t["cols"]["f_Cd"], dtype=float)
        o = np.argsort(x)
        return float(np.interp(ra_over_d, x[o], f[o]))
    #  Respaldo ETIQUETADO si la tabla no está: recta 1.00 → 0.80.
    _warn_provisional(
        "cd_am_roughness_fallback",
        "there is no `lefebvre/cd_am_roughness.csv`: f(Ra/d) falls back to "
        "the linear 1.00->0.80 over Ra/d ∈ [0, 0.05]. The effusion C_d and "
        "the liner-hole C_d inherit that uncertainty.")
    return float(np.interp(ra_over_d, [0.0, 0.05], [1.0, 0.80]))


def effusion_discharge_coefficient(d_eff_m: float) -> float:
    """C_d de un orificio de efusión — **el único** que puede usar nadie.

        C_d = C_d(`cd_holes.csv:effusion_cylindrical`)
              · CD_EFFUSION_INCLINED
              · am_roughness_factor(d)

    Los tres factores son obligatorios: el tabulado (orificio cilíndrico de
    efusión), el de INCLINACIÓN + crossflow por ambas caras, y el de
    RUGOSIDAD as-built de Ciencia §4.1. En el ancla vale 0.6500 · 0.3800 ·
    0.8947 = 0.2210.

    NO incluye la corrección por Reynolds `f(Re)`: esa depende de la solución
    (el caudal del propio orificio) y la aplica quien resuelve la red 1D.
    Este número es la parte GEOMÉTRICA del C_d y es la que deben compartir
    la red 1D y el LBM. Puro y sin estado. Nunca lanza.
    """
    return float(coeffs()["Cd_eff"] * CD_EFFUSION_INCLINED
                 * am_roughness_factor(d_eff_m))


def effusion_wall_area_m2(arch, sizing: dict) -> float:
    """Área MOJADA [m²] de la pared caliente que lleva efusión.

    Consciente de la arquitectura y **fiel a la geometría que dibuja
    `geometry_generator`** (ver la causa raíz al principio de la sección):

    * `can` — el paso ES un círculo: `A = π·D_L·L_tot`.
    * `annular` / `reverse_annular` — la corona se construye con radio
      exterior `r_lo = LINER_R_OVER_RCAS·r_carcasa` (y el respaldo de corona
      degenerada `r_li = ANNULUS_R_MIN_FRAC·r_carcasa` ⇒
      `r_lo = √(r_li² + A_paso/π)` cuando A_paso no cabe), luego la cara
      EXTERIOR moja `2π·r_lo·L_tot`. Solo esa cara lleva efusión en v1
      (`EFFUSION_WALLS`); si algún día se abre la interior, se suma aquí
      `2π·r_li·L_tot` y nada más cambia.

    `sizing` puede ser el dict de `_sizing` **o** un `record` ya evaluado:
    se leen `D_L_mm`, `D_casing_mm`, `L_tot_mm`, `A_L_m2` y, si está,
    `A4_m2` (el área de paso con la que se construye la corona es
    `max(A_L, A4)`, igual que `geometry_generator._dims`). Lectura
    defensiva: nunca lanza y devuelve 0.0 si no hay con qué.
    """
    s = sizing if isinstance(sizing, dict) else {}
    #  Gate decodificada con `round` y clip ANTES de cualquier float
    #  (invariante CONTEXTO §2.1), por si llega el θ crudo en vez del nombre.
    name = arch if isinstance(arch, str) else ARCH_NAMES[
        int(min(max(round(_safe(arch, 0.0)), 0), len(ARCH_NAMES) - 1))]
    D_L = _safe(s.get("_D_L"), 0.0) or 1e-3 * _safe(s.get("D_L_mm"), 0.0)
    L_tot = _safe(s.get("_L_tot"), 0.0) or 1e-3 * _safe(s.get("L_tot_mm"), 0.0)
    if D_L <= 0.0 or L_tot <= 0.0:
        return 0.0
    A_L = _safe(s.get("_A_L"), 0.0) or _safe(s.get("A_L_m2"), 0.0) \
        or 0.25 * math.pi * D_L * D_L
    if name == "can":
        #  `geometry_generator._dims` usa r_L = max(D_L/2, √(A_L/π)); con la
        #  física coherente los dos son iguales, y así lo son también cuando
        #  entra un record cosido a mano.
        return float(2.0 * math.pi * max(0.5 * D_L,
                                         math.sqrt(A_L / math.pi)) * L_tot)
    A4 = _safe(s.get("A4_m2"), 0.0) or A4_OVER_AL * A_L
    A_pass = max(A_L, A4, 1e-12)
    D_cas = 1e-3 * _safe(s.get("D_casing_mm"), 0.0)
    if D_cas <= 0.0:                      # respaldo: corona HTR de `_d_casing`
        D_cas = _d_casing(name, A_L / max(_safe(s.get("AL_over_Aref"), 0.70),
                                          1e-3))
    r_cas = 0.5 * D_cas
    r_lo = LINER_R_OVER_RCAS * r_cas
    r_min = ANNULUS_R_MIN_FRAC * r_cas
    if r_lo * r_lo - A_pass / math.pi <= r_min * r_min:
        #  MISMA rama de respaldo que `_annulus_geometry`: la corona no cabe,
        #  se fija r_li en el suelo y se abre r_lo conservando el área.
        r_lo = math.sqrt(r_min * r_min + A_pass / math.pi)
        r_li = r_min
    else:
        r_li = math.sqrt(r_lo * r_lo - A_pass / math.pi)
    walls = EFFUSION_WALLS.get(name, ("outer",))
    A = 0.0
    if "outer" in walls:
        A += 2.0 * math.pi * r_lo * L_tot
    if "inner" in walls:
        A += 2.0 * math.pi * r_li * L_tot
    return float(A)


def effusion_pattern(d_eff_m: float, pitch_over_d: float,
                     A_wall_m2: float) -> dict:
    """Patrón de efusión de paso CUADRADO sobre un área mojada dada.

    Hipótesis (la MISMA que dibuja `geometry_generator._effusion`): paso
    cuadrado `P_x = P_y = (paso/Ø)·d`, un orificio por celda de paso.

        paso     = (paso/Ø)·d
        porosity = (π/4)·d²/paso²
        n_holes  = int(A_pared/paso²)
        A_open   = porosity · A_pared

    `n_holes` se trunca con `int()` **a propósito**: es el mismo truncamiento
    que hace la geometría, y así los dos lados dan el mismo entero (ver
    `EFFUSION_N_HOLES_TOL`). Puro, sin estado y nunca lanza; devuelve
    `porosity`, `n_holes`, `A_open_m2` y `pitch_m`.
    """
    d = max(_safe(d_eff_m, 6.0e-4), 5.0e-5)
    pod = max(_safe(pitch_over_d, 6.0), 1.0)
    pitch = pod * d
    A_wall = max(_safe(A_wall_m2, 0.0), 0.0)
    porosity = 0.25 * math.pi * d * d / max(pitch * pitch, 1e-12)
    return dict(porosity=float(porosity),
                n_holes=int(max(A_wall / max(pitch * pitch, 1e-12), 0.0)),
                A_open_m2=float(max(porosity * A_wall, 0.0)),
                pitch_m=float(pitch))


# ==========================================================================
# 13. Térmico 1D del liner con efusión [Ciencia §5 y §5.1]
# ==========================================================================
_SC_MOD = None
_SC_TRIED = False


def _structures():
    """Import perezoso y VERIFICADO de `structures_core` (nunca lanza).

    Mismo guardia anti-Phy-AC que `_data_pipeline`: la biblioteca de
    materiales de Phy-CB (con `T_max_flame_wall_K`, `lpbf_t_min_mm`,
    `emissivity_wall` y `k_thermal_W_mK`) NO existe en el hermano.
    """
    global _SC_MOD, _SC_TRIED
    if _SC_TRIED:
        return _SC_MOD
    _SC_TRIED = True
    try:
        import importlib
        mod = importlib.import_module("structures_core")
        here = os.path.dirname(os.path.abspath(__file__))
        there = os.path.dirname(os.path.abspath(getattr(mod, "__file__", "")))
        _SC_MOD = mod if os.path.normcase(here) == os.path.normcase(there) \
            else None
    except Exception:
        _SC_MOD = None
    if _SC_MOD is None:
        _warn_provisional(
            "structures_core",
            "structures_core is not importable: the wall limit of g[1] uses "
            "the IN625 fallback (1253 K) instead of MATERIALS.")
    return _SC_MOD


def material_limits(material: str = MATERIAL_DEFAULT) -> dict:
    """Límites de pared del material desde `structures_core.MATERIALS`.

    Devuelve `T_max_flame_wall_K`, `lpbf_t_min_mm`, `emissivity_wall` y un
    callable `k_wall(T)`. Con `structures_core` ausente cae a los valores de
    IN625 (el default de liner) y lo declara en `PROVISIONAL_COEFFS`.
    """
    sc = _structures()
    if sc is not None and material in getattr(sc, "MATERIALS", {}):
        m = sc.MATERIALS[material]

        def k_wall(T: float, _m=m, _sc=sc) -> float:
            try:
                return float(_sc.material_at_T(material, T)["k_thermal_W_mK_T"])
            except Exception:
                return 20.0

        return dict(name=material,
                    T_max_flame_wall_K=float(m["T_max_flame_wall_K"]),
                    lpbf_t_min_mm=float(m["lpbf_t_min_mm"]),
                    emissivity_wall=float(m["emissivity_wall"]),
                    k_wall=k_wall)
    return dict(name=material, T_max_flame_wall_K=1253.0, lpbf_t_min_mm=0.50,
                emissivity_wall=0.85, k_wall=lambda T: 20.0)


def _gas_emissivity(P3: float, L_beam: float, far: float, lum: float,
                    T_g: float, liquid: bool) -> float:
    """ε_g del gas [Ciencia §5]: 1 − exp(−0.29·P₃·L·(f·l_b)^0.5·T_g^−1.5).

    ⚠️ El coeficiente 0.29 está sin verificar y, con P₃ en Pa y L en m, la
    expresión da ε_g ≈ 0.01 en el caso ancla — un orden de magnitud por
    debajo de la banda 0.2–0.6 que la literatura reporta para llamas de
    queroseno. Se aplica el suelo `EPS_G_FLOOR_*` (ver su comentario de
    procedencia) para que el térmico no deje el gas transparente, y el
    aviso queda registrado.
    """
    x = (GAS_EMISSIVITY_COEFF * P3 * L_beam
         * math.sqrt(max(far * lum, 1e-12)) * max(T_g, 300.0) ** -1.5)
    eps = 1.0 - math.exp(-min(max(x, 0.0), 50.0))
    floor = EPS_G_FLOOR_LUMINOUS if (liquid and lum > 1.2) \
        else EPS_G_FLOOR_NONLUMINOUS
    if eps < floor:
        _warn_provisional(
            "eps_gas_floor",
            f"ε_g from the form of Ciencia §5 is {eps:.4f}, below the "
            f"in-house calibrated floor {floor:.2f} (literature band 0.2–0.6 "
            f"for kerosene): the 1-D thermal would underestimate the "
            f"radiation and T_wall. RECALIBRATE the 0.29 coefficient.")
    return float(min(max(eps, floor), EPS_G_MAX))


def _solve_wall(T_g: float, T_aw: float, T3: float, h1: float, h2: float,
                eps_g: float, eps_w: float, k_over_t: float,
                rad_pref: float) -> tuple[float, float, float, float]:
    """Resuelve R1+C1 = K₁₋₂ = R2+C2 por BISECCIÓN ACOTADA sobre T_w1.

    q_in(T_w1) es monótona decreciente y q_out(T_w2(T_w1)) monótona
    creciente, así que el residuo tiene una única raíz y la bisección
    converge siempre (60 pasos ⇒ |ΔT| < 1e-13 K sobre un intervalo de
    2000 K). Devuelve `(T_w1, T_w2, q, residuo_relativo)`.
    """
    def q_in(Tw1: float) -> float:
        R1 = (rad_pref * SIGMA_SB * (1.0 + eps_w) * eps_g
              * max(T_g, 1.0) ** 1.5
              * (max(T_g, 1.0) ** 2.5 - max(Tw1, 1.0) ** 2.5))
        C1 = h1 * (T_aw - Tw1)
        return R1 + C1

    def q_out(Tw2: float) -> float:
        R2 = SIGMA_SB * eps_w * (max(Tw2, 1.0) ** 4 - max(T3, 1.0) ** 4)
        C2 = h2 * (Tw2 - T3)
        return R2 + C2

    def residual(Tw1: float) -> tuple[float, float, float]:
        q = q_in(Tw1)
        Tw2 = Tw1 - q / max(k_over_t, 1e-6)
        return q - q_out(Tw2), q, Tw2

    lo, hi = min(T3, T_aw) - 5.0, max(T_g, T_aw, T3) + 5.0
    r_lo, _, _ = residual(lo)
    r_hi, _, _ = residual(hi)
    if r_lo * r_hi > 0.0:                      # sin cambio de signo: extremo
        Tw1 = lo if abs(r_lo) < abs(r_hi) else hi
        _, q, Tw2 = residual(Tw1)
        return Tw1, Tw2, q, abs(residual(Tw1)[0]) / max(abs(q), 1.0)
    for _ in range(THERMAL_ITERS):             # bucle ACOTADO
        mid = 0.5 * (lo + hi)
        r, _, _ = residual(mid)
        if r > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < THERMAL_TOL_K:
            break
    Tw1 = 0.5 * (lo + hi)
    r, q, Tw2 = residual(Tw1)
    return Tw1, Tw2, q, abs(r) / max(abs(q), 1.0)


def _liner_thermal_1d(theta: np.ndarray, gates: dict, sz: dict, part: dict,
                      holes: dict, T_zone: dict, mw_hot: float, far: float,
                      material: str = MATERIAL_DEFAULT,
                      blowing_ratio: float | None = None,
                      cooling: bool = True) -> dict:
    """Balance térmico 1D del liner con refrigeración por efusión [§5].

    Por unidad de área y en régimen estacionario:
        R₁ + C₁ = K₁₋₂ = R₂ + C₂
    con radiación interna luminosa (ε_g del gas), convección interna y
    externa tipo Dittus-Boelter (exponente 0.8 ✅, coeficientes 0.020 ⚠️),
    radiación externa al casing y conducción a través del espesor.

    La **efusión** entra como reducción de la temperatura adiabática de
    pared que ve la convección interna, T_aw = T_g − η_film·(T_g − T₃)
    (la radiación sigue viendo T_g: la película es transparente). η_film
    sale del mapa η(M) de `data/lefebvre/film_effusion.csv` ⚠️;
    `blowing_ratio` permite imponer M (lo usan los tests y los estudios
    paramétricos) y `cooling=False` apaga la película («liner desnudo»).

    RELACIÓN DE SOPLADO — CORREGIDA EN LA AUDITORÍA 2026-07-30
    ----------------------------------------------------------
    El smoke del ancla daba M = 5.48, siete veces la banda de diseño 0.3–0.8
    de Ciencia §5.1, con η_film clavado en 0.712357 = el ÚLTIMO punto de la
    tabla: la eficacia no estaba modelada, estaba saturada. Dos defectos
    reales, los dos de cierre de masa:

    1. **U_c salía de la caída de presión, no del caudal.** Se usaba
       `U_c = C_d·√(2ΔP_L/ρ₃)` = 48.4 m/s en el ancla, que es la velocidad
       que los orificios PODRÍAN dar, no la que el reparto de aire les
       permite. Con la porosidad del ancla esa velocidad implicaría
       ṁ_efusión = 0.061 kg/s = 47 % del aire total, cuando `frac_cool`
       —que es CIERRE del contrato §1, no una variable— solo asigna el
       16 %. Ahora **U_c sale de conservación de masa**,
       `U_c = ṁ_cool/(ρ₃·A_efusión)` = 16.5 m/s, y el valor de ΔP se
       reporta aparte (`U_cool_dp_ms`) junto con el cociente de áreas
       `effusion_area_match` = A_requerida(ΔP)/A_geométrica, que es la
       señal honesta de que el patrón de efusión y el reparto de aire no
       están casados (0.34 en el ancla: el patrón sobra 3×).
    2. **M se evaluaba en UNA estación (la PZ).** La densidad se cancela en
       M = (ṁ_c/A_ef)/(ṁ_g/A_L), así que M solo depende de flujos másicos
       por unidad de área — y ṁ_g crece un factor 3 del domo a la salida al
       ir entrando el aire de SZ y DZ. Ahora M se evalúa **local a cada
       estación**, el mapa se interpola con el M local y se reportan la
       media (`blowing_ratio_M`), el mínimo y el máximo.

    Si M queda fuera de la banda de DISEÑO (`M_design_min/max` de la tabla)
    se dispara `_warn_provisional` y se enciende `g_soft["M_design"]`; si
    queda fuera del rango de VALIDEZ del mapa —donde la interpolación es un
    clamp y la eficacia deja de estar modelada— se añade además una
    penalización CONTINUA a `g[8]` (ver `_constraints`).

    Devuelve además `T_wall_profile` = [[z_mm, T_w1_K, T_w2_K], …] con
    `N_WALL_STATIONS` estaciones en los centros de celda a lo largo del
    liner: la eficacia de la efusión crece con el número de filas aguas
    arriba y satura a ~10 filas (dato ✅), así que el domo sale más caliente
    que la salida — que es lo que se mide en un liner real.
    """
    c = coeffs()
    lim = material_limits(material)
    P3 = float(theta[IDX_P3_PA])
    T3 = float(theta[IDX_T3_K])
    rho3 = sz["rho3_kgm3"]
    fp = FUELS[gates["fuel"]]
    eps_w = lim["emissivity_wall"]
    t_w = max(float(theta[14]) if np.isfinite(theta[14]) else 0.8, 0.05) * 1e-3
    A_L, A_an = sz["_A_L"], sz["A_an_m2"]
    D_L, D_an = sz["_D_L"], max(sz["D_an_m"], 1e-3)
    L_tot = sz["_L_tot"]

    # --- geometría de la efusión (ANTES que M: M depende de A_efusión) ----
    #  FUENTE ÚNICA §12.b: el área mojada, el patrón y el C_d salen de las
    #  MISMAS funciones que consumen `geometry_generator`, `crn_core` y
    #  `cfd/xlb_cold`. Aquí ya NO se escribe `π·D_L·L_tot`: esa fórmula solo
    #  vale para `can` y en las anulares se quedaba ×1.375 corta.
    d_eff = max(float(theta[8]) if np.isfinite(theta[8]) else 0.6, 0.05) * 1e-3
    pod = max(float(theta[9]) if np.isfinite(theta[9]) else 6.0, 1.0)
    rows_sat = max(c["eff_rows_sat"], 1.0)
    A_wall = effusion_wall_area_m2(gates["arch"], sz)
    pat = effusion_pattern(d_eff, pod, A_wall)
    pitch = pat["pitch_m"]
    n_eff_holes = pat["n_holes"]
    porosity = pat["porosity"]
    A_eff = max(pat["A_open_m2"], 1e-12)            # área perforada total
    Cd_eff_hole = effusion_discharge_coefficient(d_eff)

    # --- flujo de refrigerante: CONSERVACIÓN DE MASA, no ΔP ---------------
    mdot_air = max(float(T_zone.get("mdot_air_kgs", 0.0)), 1e-9)
    mdot_cool = max(part["splits"]["cool"], 0.0) * mdot_air
    G_cool = mdot_cool / A_eff                      # flujo másico [kg/m²/s]
    U_cool = G_cool / max(rho3, 1e-6)
    #  Diagnóstico: la velocidad que la caída del liner PODRÍA imponer y el
    #  cociente de áreas que revela si el patrón está casado con el reparto.
    #  AUDITORÍA 2026-07-31: usaba `c["Cd_eff"]` PELADO (0.65, la fila de la
    #  tabla), que es un orificio de efusión PULIDO, NORMAL a la pared y sin
    #  crossflow — un cuarto C_d de efusión distinto dentro de este mismo
    #  módulo. Ahora usa el de §12.b (0.2210 en el ancla), que es el que
    #  descarga de verdad. `effusion_area_match` deja de acusar un patrón
    #  «3× sobredimensionado» que solo existía por el C_d equivocado.
    U_cool_dp = Cd_eff_hole * math.sqrt(
        max(2.0 * holes["dp_liner_Pa"] / max(rho3, 1e-6), 1e-9))
    A_eff_req = mdot_cool / max(rho3 * U_cool_dp, 1e-12)
    area_match = float(A_eff_req / A_eff)

    # --- estaciones axiales ----------------------------------------------
    emap = c["eff_map"]
    M_map_lo, M_map_hi = float(emap["M"][0]), float(emap["M"][-1])
    L_pz, L_sz = 1e-3 * sz["L_pz_mm"], 1e-3 * sz["L_sz_mm"]
    T04 = float(theta[IDX_T4_K])
    profile: list[list[float]] = []
    best = None
    M_list: list[float] = []
    eta_list: list[float] = []
    saturated = False
    for i in range(N_WALL_STATIONS):
        z = (i + 0.5) / N_WALL_STATIONS * L_tot
        if z <= L_pz:
            T_g, m_g = T_zone["T_pz_K"], T_zone["mdot_pz_kgs"]
            m_an = T_zone["mdot_an_pz_kgs"]
        elif z <= L_pz + L_sz:
            T_g, m_g = T_zone["T_sz_K"], T_zone["mdot_sz_kgs"]
            m_an = T_zone["mdot_an_sz_kgs"]
        else:
            frac = (z - L_pz - L_sz) / max(L_tot - L_pz - L_sz, 1e-9)
            T_g = T_zone["T_sz_K"] + frac * (T04 - T_zone["T_sz_K"])
            m_g = T_zone["mdot_sz_kgs"] + frac * (T_zone["mdot4_kgs"]
                                                  - T_zone["mdot_sz_kgs"])
            m_an = max(T_zone["mdot_an_dz_kgs"] * (1.0 - frac), 1e-6)
        #  M local: la densidad se cancela, M = (ṁ_c/A_ef)/(ṁ_g/A_L).
        G_g = max(m_g, 1e-9) / max(A_L, 1e-12)
        M_loc = float(blowing_ratio) if blowing_ratio is not None \
            else G_cool / max(G_g, 1e-12)
        M_list.append(M_loc)
        M_clip = min(max(M_loc, M_map_lo), M_map_hi)
        if abs(M_clip - M_loc) > 1e-12:
            saturated = True
        eta_M = float(np.interp(M_clip, emap["M"], emap["eta"]))
        if not cooling or part["splits"]["cool"] <= 1e-6:
            eta_M = 0.0
        eta_list.append(eta_M)
        n_rows = z / max(pitch, 1e-9)
        eta_loc = eta_M * (1.0 - math.exp(-3.0 * n_rows / rows_sat))
        T_aw = T_g - eta_loc * (T_g - T3)
        mu_g, mu_a = _mu_gas(T_g), _mu_gas(T3)
        k_g = _k_gas(T_g, CP_HOT)
        k_a = _k_gas(T3, CP)
        h1 = (c["conv_in"] * k_g * D_L ** c["conv_d_exp"]
              * (max(m_g, 1e-9) / max(A_L * mu_g, 1e-15)) ** c["conv_re_exp"])
        h2 = (c["conv_out"] * k_a * D_an ** c["conv_d_exp"]
              * (max(m_an, 1e-9) / max(A_an * mu_a, 1e-15)) ** c["conv_re_exp"])
        eps_g = _gas_emissivity(P3, D_L, far, float(fp["luminosity"]), T_g,
                                bool(fp["liquid"]))
        k_w = lim["k_wall"](0.5 * (T_g + T3))
        Tw1, Tw2, q, res = _solve_wall(T_g, T_aw, T3, h1, h2, eps_g, eps_w,
                                       k_w / t_w, c["rad_pref"])
        profile.append([float(1e3 * z), float(Tw1), float(Tw2)])
        if best is None or Tw1 > best[0]:
            best = (Tw1, Tw2, q, res, eps_g, eta_loc)

    Tw1, Tw2, q, res, eps_g, eta_loc = best
    M_mean = float(sum(M_list) / max(len(M_list), 1))
    eta_film = float(sum(eta_list) / max(len(eta_list), 1))
    #  Avisos de diseño (banda de DISEÑO) y de modelo (rango del MAPA).
    if cooling and part["splits"]["cool"] > 1e-6:
        if not (c["M_eff_min"] <= M_mean <= c["M_eff_max"]):
            _warn_provisional(
                "blowing_ratio_out_of_design_band",
                f"effusion blowing ratio M = {M_mean:.3f} (liner average, "
                f"{min(M_list):.3f}–{max(M_list):.3f}) OUTSIDE the design "
                f"band [{c['M_eff_min']:.2f}, {c['M_eff_max']:.2f}] of "
                f"Ciencia §5.1. The film effectiveness is not in the regime "
                f"there is data for: it lights g_soft['M_design'].")
        if saturated:
            _warn_provisional(
                "effusion_map_clamped",
                f"the η_film(M) map was evaluated CLAMPED at some station "
                f"(tabulated range [{M_map_lo:.2f}, {M_map_hi:.2f}], design "
                f"M {min(M_list):.3f}–{max(M_list):.3f}): there the cooling "
                f"effectiveness is NOT modelled, it is saturated. It "
                f"penalises g[8] continuously.")
        if area_match < 0.5 or area_match > 2.0:
            _warn_provisional(
                "effusion_area_mismatch",
                f"the effusion pattern and the air split are not matched: "
                f"A_required(ΔP_liner)/A_geometric = {area_match:.3f} (1.0 = "
                f"matched). With porosity {100.0 * porosity:.2f} % the holes "
                f"would pass {1.0 / max(area_match, 1e-9):.2f}x the air that "
                f"`frac_cool` assigns them. Levers: `frac_cool` (closure of "
                f"the split) and `pitch_over_d_eff`.")
    return dict(T_w1_K=float(Tw1), T_w2_K=float(Tw2),
                T_wall_max_K=float(Tw1), q_wall_W_m2=float(q),
                thermal_residual_rel=float(res), eps_gas=float(eps_g),
                eta_film=eta_film, eta_film_local=float(eta_loc),
                eta_film_saturated=bool(saturated),
                blowing_ratio_M=M_mean,
                blowing_ratio_M_min=float(min(M_list)),
                blowing_ratio_M_max=float(max(M_list)),
                blowing_ratio_M_profile=[float(v) for v in M_list],
                U_cool_ms=float(U_cool), U_cool_dp_ms=float(U_cool_dp),
                effusion_area_match=area_match, A_effusion_m2=float(A_eff),
                mdot_cool_kgs=float(mdot_cool),
                M_design_band=[float(c["M_eff_min"]), float(c["M_eff_max"])],
                M_map_range=[M_map_lo, M_map_hi],
                T_wall_profile=profile, n_eff_holes=int(n_eff_holes),
                effusion_porosity=float(porosity),
                effusion_wall_area_m2=float(A_wall),
                effusion_Cd=float(Cd_eff_hole),
                effusion_walls=list(EFFUSION_WALLS.get(gates["arch"],
                                                       ("outer",))),
                effusion_d_mm=1e3 * d_eff, effusion_pitch_mm=1e3 * pitch,
                material=lim["name"],
                T_max_flame_wall_K=lim["T_max_flame_wall_K"],
                lpbf_t_min_mm=lim["lpbf_t_min_mm"])


# ==========================================================================
# 14. Factor de patrón [Ciencia §4.3] — se exponen AMBAS definiciones
# ==========================================================================
def _pattern_factor(sz: dict, holes: dict, theta: np.ndarray) -> dict:
    """PF, PF* y RTDF desde la correlación exponencial de dilución.

        PF = exp[−k·(L_DZ/D_L)·(ΔP_L/q_ref)]      ⚠️ k calibrable (capa 0)

    Coexisten dos definiciones en la literatura y Phy-CB expone las dos
    (evitar la confusión es parte del contrato):

        PF  = (T₄max − T₄avg)/(T₄avg − T₃)   ← estándar de diseño
        PF* = (T₄peak − T₄avg)/T₄avg
        RTDF = (T₄max,radial − T₄avg)/(T₄avg − T₃),  típico 0.12–0.30

    Bandas objetivo: 0.15–0.25 (turbina refrigerada), 0.25–0.35
    (convencional) y **0.30–0.45 aceptable en microturbina con turbina
    radial no refrigerada**.

    AUDITORÍA 2026-07-30 — POR QUÉ EL ANCLA DA PF = 0.35000000857
    -------------------------------------------------------------
    NO es saturación ni un punto de calibración devuelto tal cual: la
    correlación responde, pero solo por UN camino. Con el criterio
    aerodinámico dimensionando A_ref (§2.1), ΔP_L/q_ref es una IDENTIDAD
    ALGEBRAICA independiente del punto de operación:

        A_ref² = (R/2)·(ṁ₃√T₃/P₃)²·PLF/(Δp/p)   [criterio aero]
        q_ref/P₃ = (R/2)·(ṁ₃√T₃/(A_ref·P₃))²  =  (Δp/p)/PLF
        ⇒ ΔP_L/q_ref = LINER_FRAC_OF_DP·(Δp/p)·P₃/q_ref
                     = LINER_FRAC_OF_DP·PLF = 0.75·20 = **15 siempre**

    y 15 es exactamente el ΔP_L/q_ref con el que `pf_constant.csv` calibró
    `k_dilution` (junto con L_DZ/D_L = 1.8, que es el del θ ancla). El
    ancla cae, por construcción, en el punto de calibración: los dígitos
    `…00857` son el redondeo de `k_dilution` a 7 cifras en el CSV.

    Consecuencias que hay que decir en voz alta:
      * PF **sí** responde a la geometría: baja monótonamente al alargar
        L_DZ/D_L (1.4 → 0.442 · 1.8 → 0.350 · 2.2 → 0.277) y al subir
        ΔP_L/q_ref. Ambas monotonías se comprueban en el `__main__`.
      * PF **no** responde a Δp/p ni a P₃ mientras dimensione el criterio
        aerodinámico, porque A_ref se ajusta para dejar ΔP_L/q_ref fijo.
        Solo se desacopla cuando manda el criterio químico
        (`sizing_driver == "chem"`, A_ref > A_aero ⇒ q_ref menor ⇒
        ΔP_L/q_ref mayor ⇒ PF menor).
      * La única palanca real de PF en L0 es, por tanto, la longitud de
        dilución. Es una limitación del modelo de nivel 0, no del diseño:
        el número de filas de dilución, su desfase y su penetración —que
        es lo que de verdad fija el PF— solo entran a partir de L1.5/L2.
    """
    c = coeffs()
    T3 = float(theta[IDX_T3_K])
    T4 = float(theta[IDX_T4_K])
    ratio = sz["L_dz_mm"] / max(sz["D_L_mm"], 1e-9)
    PF = math.exp(-max(c["pf_k"], 0.0) * ratio
                  * max(holes["dp_liner_over_qref"], 0.0))
    PF = float(min(max(PF, 0.0), 5.0))
    T4_max = T4 + PF * (T4 - T3)
    return dict(PF=PF, PF_star=float(PF * (T4 - T3) / max(T4, 1.0)),
                RTDF=float(c["rtdf_over_pf"] * PF), T4_max_K=float(T4_max))


# ==========================================================================
# 15. Emisiones [Ciencia §8] — TODAS calculadas, TODAS con su nombre
#
# §15.0 AUDITORÍA DIMENSIONAL DEL ABANICO DE EI_NOx (2026-07-30)
# --------------------------------------------------------------
# El smoke del caso ancla REF_CB10 destapó un abanico de SEIS órdenes de
# magnitud (lefebvre 2.0 · rizk_mongia 1147 · zonal 313 · odgers 0.0044 ·
# laviolette 0.030 g/kg) — el error clásico que Ciencia §8 anticipa:
# UNIDADES. Ciencia §8.1/§8.4 declara que la unidad de P₃ NO es explícita en
# las fuentes de Lefebvre ni de Odgers, y §8.3 avisa de que la extracción de
# texto de Rizk & Mongia degradó signos y paréntesis.
#
# MÉTODO (evidencia, no ajuste): cada correlación se re-derivó barriendo
# SISTEMÁTICAMENTE la unidad de P₃ (Pa · kPa · bar · atm · psia), la de τ
# (s · ms) y la de Δp/p (fracción · porcentaje), evaluando las 4–24
# combinaciones en DOS extremos a la vez:
#   * ancla REF_CB10  — 250 kPa, 430 K, 0.13 kg/s, τ = 6.05 ms, T_st = 2360 K
#   * motor grande    — 30 atm, 800 K, τ = 3 ms, T_st = 2600 K, Δp/p = 5 %,
#                       donde la literatura da EI_NOx = 20–40 g/kg
# Se conserva la ÚNICA combinación que cae en rango en los DOS extremos.
# Resultado (todas las escalas viven en `emissions_coeffs.csv`, nunca aquí):
#
#   correlación      P₃      τ    Δp/p   ancla   grande   veredicto
#   lefebvre         kPa     –     –      2.00    39.2    ✅ confirmada
#   rizk_mongia      kPa*    s     %      3.63    26.7    ⚠️ τ y Δp/p corregidas
#   zonal            Pa      s     –      3.76    41.0    ⚠️ P₃ y τ corregidas
#   odgers           Pa      s     –      8.91    23.5    ⚠️ P₃ corregida (NO atm)
#   laviolette       Pa      s     –      3.17     4.3    ⚠️ PROVISIONAL (baja x5)
#   * indecidible: exp_P3 = −0.05 (todo Pa↔atm cabe en un factor 1.8).
#
# Dispersión del abanico en el ancla: ×2.6e5 antes → ×4.5 después.
# Ninguna se «arregla» moviendo su coeficiente: solo se han movido factores
# de UNIDAD, y todos siguen marcados `provisional` en la tabla (son
# inferencia dimensional, no lectura de fuente primaria — ver
# `docs/PENDIENTES_Lverificar.md`).
# ==========================================================================
def _clip_ei(x: float, name: str = "") -> float:
    """Tope de reporte de un índice de emisión [g/kg], finito y serializable.

    Solo acota y sanea (NaN → 0.0, tope `EI_REPORT_MAX`); la PANTALLA de
    plausibilidad —marcado, exclusión del abanico y aviso— la aplica
    `_screen_nox_fan`, que es quien conoce el abanico completo.
    """
    if not np.isfinite(x):
        return 0.0
    return float(min(max(x, 0.0), EI_REPORT_MAX))


def _screen_nox_fan(fan: dict) -> tuple[dict, dict]:
    """Pantalla de plausibilidad del abanico de EI_NOx (contrato §7).

    `fan` es `{nombre_corto: EI_NOx [g/kg]}`. Devuelve `(flags, stats)`:

    * `flags[nombre]` ∈ {`"ok"`, `"out_of_range_low"`, `"out_of_range_high"`}
      — la clave hermana que el contrato pide en `record["EI_NOx_flags"]`.
      TODAS las correlaciones se siguen reportando con su valor: la bandera
      NO oculta nada, dice qué mirar con lupa (regla de honestidad §8).
    * `stats` = `{min, max, median, mean, spread_ratio, n_used, n_excluded,
      used, excluded}` calculado **solo** con las que caen dentro de
      `EI_NOX_PLAUSIBLE_RANGE`: una correlación fuera de su rango de
      calibración no debe contaminar la dispersión que el reporte enseña.

    Cada exclusión dispara `_warn_provisional` una vez.
    """
    c = coeffs()
    lo, hi = c["ei_nox_min"], c["ei_nox_max"]
    flags, used = {}, {}
    for name, v in fan.items():
        if not np.isfinite(v) or v < lo:
            flags[name] = "out_of_range_low"
        elif v > hi:
            flags[name] = "out_of_range_high"
        else:
            flags[name] = "ok"
            used[name] = float(v)
        if flags[name] != "ok":
            _warn_provisional(
                f"ei_out_of_band:{name}",
                f"EI_NOx_{name} = {v:.4g} g/kg falls OUTSIDE the "
                f"plausibility band [{lo:g}, {hi:g}] g/kg: the correlation "
                f"is beyond its calibration range. It is reported for "
                f"transparency (contract §7) but is EXCLUDED from the "
                f"min/max and from the fan statistics, and it is NEVER the "
                f"headline (the record's EI_NOx is always Lefebvre's).")
    vals = sorted(used.values())
    if vals:
        stats = dict(
            min=float(vals[0]), max=float(vals[-1]),
            median=float(vals[len(vals) // 2] if len(vals) % 2
                         else 0.5 * (vals[len(vals) // 2 - 1]
                                     + vals[len(vals) // 2])),
            mean=float(sum(vals) / len(vals)),
            spread_ratio=float(vals[-1] / max(vals[0], 1e-12)))
    else:
        stats = dict(min=0.0, max=0.0, median=0.0, mean=0.0, spread_ratio=1.0)
    stats.update(n_used=len(used), n_excluded=len(fan) - len(used),
                 used=sorted(used), excluded=sorted(set(fan) - set(used)),
                 band=[float(lo), float(hi)])
    return flags, stats


def nox_fan(P3_Pa: float, T_st_K: float, T_pz_K: float, T_sz_K: float,
            V_c_m3: float, mdot_air_kgs: float, tau_s: float,
            tau_e_s: float, tau_pz_s: float, tau_sz_s: float,
            dp_over_p: float, phi_pz: float, phi_sz: float) -> dict:
    """Abanico de EI_NOx [g/kg] a partir de escalares — función PURA.

    La comparten `_emissions` (cadena L0) y el test de coherencia del
    `__main__`, que la evalúa en un punto de MOTOR GRANDE que no es un θ de
    Phy-CB: por eso la firma es de escalares y no de `record`.

    Cada correlación aplica sus tres factores de unidad
    (`P3_unit_scale_from_Pa`, `tau_unit_scale_from_s`,
    `dp_over_p_unit_scale_from_frac`) leídos de
    `data/lefebvre/emissions_coeffs.csv` — NUNCA hard-codeados aquí. La
    evidencia dimensional que los fija está en §15.0 y en la cabecera de la
    propia tabla.
    """
    c = coeffs()
    P3 = max(float(P3_Pa), 1.0)
    T_st = max(float(T_st_K), 300.0)
    T_pz = max(float(T_pz_K), 300.0)
    T_sz = max(float(T_sz_K), 300.0)
    V_c = max(float(V_c_m3), 1e-12)
    mA = max(float(mdot_air_kgs), 1e-9)
    dpp = max(float(dp_over_p), 1e-6)

    # --- Lefebvre ✅ (P₃ en kPa, confirmada en los dos extremos) ----------
    P3_lef = P3 * c["nox_lef_P3scale"]
    ei_lef = (c["nox_lef_C"] * P3_lef ** c["nox_lef_expP3"] * V_c
              * math.exp(min(c["nox_lef_Tst"] * T_st, 60.0)) / (mA * T_pz))

    # --- Rizk & Mongia global (τ en s, Δp/p en %) ------------------------
    tau_grp = max((float(tau_s) - 0.5 * float(tau_e_s)) * c["rm_tauscale"],
                  1e-12)
    ei_rm = (c["rm_C"] * tau_grp ** 0.5
             * math.exp(-c["rm_Ea"] / T_st)
             * (P3 * c["rm_P3scale"]) ** c["rm_expP3"]
             * (dpp * c["rm_dppscale"]) ** c["rm_expdp"])

    # --- Rizk & Mongia zonal (P₃ en Pa, τ en s) --------------------------
    P3_z = P3 * c["rmz_P3scale"]
    aa = 1.0 - c["rmz_aa_C1"] * math.exp(-c["rmz_aa_C2"] * max(phi_pz, 1e-6))
    base = c["rmz_C"] * ((P3_z ** 1.4) / 1.0e6) ** aa
    ph_pz = max(float(phi_pz), 1e-6)
    if ph_pz <= c["rmz_split"]:
        f_phi = c["rmz_lean_C"] * ph_pz ** c["rmz_lean_e"] + c["rmz_off"]
    else:
        f_phi = c["rmz_rich_C"] * ph_pz ** c["rmz_rich_e"] + c["rmz_off"]
    ph_sz = max(float(phi_sz), 1e-6)
    f_sz = c["rmz_sz_C"] * ph_sz ** c["rmz_sz_e"] + c["rmz_off"]
    t_pz = max(float(tau_pz_s) * c["rmz_tauscale"], 1e-12)
    t_sz = max(float(tau_sz_s) * c["rmz_tauscale"], 1e-12)
    ei_zonal = (base * math.exp(-c["rmz_Ea"] / T_pz) * max(f_phi, 0.0)
                * t_pz ** c["rmz_tau_pz"]
                + base * math.exp(-c["rmz_Ea"] / T_sz) * max(f_sz, 0.0)
                * t_sz ** c["rmz_tau_sz"])

    # --- Odgers & Kretschmer (P₃ en Pa) / Laviolette (P₃ en Pa) ----------
    tau_od = max(float(tau_s) * c["od_tauscale"], 0.0)
    ei_od = (c["od_C"] * math.exp(-c["od_Ea"] / T_pz)
             * (P3 * c["od_P3scale"]) ** c["od_expP3"]
             * (1.0 - math.exp(-c["od_tau"] * tau_od)))
    tau_lv = max(float(tau_s) * c["lv_tauscale"], 0.0)
    ei_lv = ((P3 * c["lv_P3scale"]) ** c["lv_expP3"] * c["lv_C"]
             * (1.0 - math.exp(-c["lv_tau"] * tau_lv))
             / math.exp(min(c["lv_Ea"] / T_pz, 300.0)))

    return {"lefebvre": _clip_ei(ei_lef), "rizk_mongia": _clip_ei(ei_rm),
            "zonal": _clip_ei(ei_zonal), "odgers": _clip_ei(ei_od),
            "laviolette": _clip_ei(ei_lv)}


def _emissions(theta: np.ndarray, sz: dict, part: dict, holes: dict,
               T_zone: dict, atom: dict, mdot_air: float,
               T_st: float) -> dict:
    """El abanico completo de correlaciones de NOx y CO [Ciencia §8].

    Se calculan y se reportan TODAS con su nombre —el usuario ve la
    dispersión, no un número único— y el `EI_NOx` de cabecera es el de
    **Lefebvre**, la única ✅ verificada cuya unidad de P₃ está declarada:

      * `EI_NOx_lefebvre`   ✅ 9e-8·P₃^1.25·V_c·e^{0.01·T_st}/(ṁ_A·T_PZ)
      * `EI_CO_lefebvre`    ⚠️ 86·ṁ_A·T_PZ·e^{−0.00345·T_PZ}/((V_c−V_e)·…)
      * `EI_NOx_rizk_mongia` ✅ global, y `EI_NOx_zonal` ✅ por zonas
      * `EI_NOx_odgers`     ✅ Odgers & Kretschmer (satura para τ > ~20 ms)
      * `EI_NOx_laviolette` ✅ variante optimizada de la misma fuente

    **Unidades**: PENDIENTE DECLARADO (Ciencia §8.1/§8.3/§8.4) cerrado por
    ENSAYO DIMENSIONAL EN DOS EXTREMOS — ver §15.0 arriba. Nada se
    hard-codea: cada correlación aplica sus factores
    `P3_unit_scale_from_Pa`, `tau_unit_scale_from_s` y
    `dp_over_p_unit_scale_from_frac` leídos de
    `data/lefebvre/emissions_coeffs.csv`, que es donde se fijan y donde hay
    que tocarlos si una banda no cuadra.

    **Pantalla de plausibilidad**: `_screen_nox_fan` marca en
    `EI_NOx_flags` y excluye de `EI_NOx_fan` (min/max/mediana/dispersión)
    toda correlación fuera de `EI_NOX_PLAUSIBLE_RANGE`. Se siguen
    reportando TODAS con su valor: la bandera no oculta, señala.

    **V_c óptimo analítico** [§8.2]: NOx ∝ +V_c y CO ∝ 1/(V_c−V_e), luego
    ∂(w₁·EI_NOx + w₂·EI_CO)/∂V_c = 0 tiene solución cerrada
    `V_c,opt = V_e + √(w₂·b/(w₁·a))`. El peso relativo `NOX_CO_WEIGHT_RATIO`
    NO es física: es una preferencia ambiental declarada.

    REGLA DE HONESTIDAD (contrato §7): estas correlaciones dan TENDENCIAS
    (±30–50 % en el mejor caso). El reporte nunca las presenta como
    absolutos certificables.
    """
    c = coeffs()
    P3 = float(theta[IDX_P3_PA])
    V_c = max(sz["V_c_m3"], 1e-12)
    T_pz = max(T_zone["T_pz_K"], 300.0)
    T_sz = max(T_zone["T_sz_K"], 300.0)
    mA = max(mdot_air, 1e-9)
    tau_s = 1e-3 * sz["tau_ms"]
    dpp = max(holes["dp_over_p"], 1e-6)
    smd = atom.get("SMD_um")
    fp = FUELS.get(_gate_fuel(theta), FUELS["jet_a"])
    lam = float(fp["lambda_evap"])
    tau_e_s = 0.0 if smd is None else \
        (float(smd) * 1e-6) ** 2 / max(lam, 1e-12)
    tau_pz_s = max(sz["V_pz_m3"] * sz["rho3_kgm3"]
                   / max(part["splits"]["pz"] * mA, 1e-9), 1e-9)
    tau_sz_s = max(tau_s - tau_pz_s, 1e-9)

    # --- abanico de EI_NOx (función pura, unidades desde `data/`) ---------
    fan = nox_fan(P3_Pa=P3, T_st_K=T_st, T_pz_K=T_pz, T_sz_K=T_sz,
                  V_c_m3=V_c, mdot_air_kgs=mA, tau_s=tau_s, tau_e_s=tau_e_s,
                  tau_pz_s=tau_pz_s, tau_sz_s=tau_sz_s, dp_over_p=dpp,
                  phi_pz=part["phi_pz"], phi_sz=part["phi_sz"])
    flags, stats = _screen_nox_fan(fan)

    # --- Lefebvre CO ⚠️ (P₃ en kPa: 57.6 g/kg en el ancla, 8.3 en el motor
    #     grande; Pa daría 9e-4 y atm 2.8e4 — fuera de toda banda física) --
    V_e = 0.0 if smd is None else min(
        V_c * min(tau_e_s / max(tau_s, 1e-12), 0.90), 0.90 * V_c)
    P3_co = P3 * c["co_lef_P3scale"]
    ei_co_lef = (c["co_lef_C"] * mA * T_pz
                 * math.exp(max(c["co_lef_Tpz"] * T_pz, -60.0))
                 / (max(V_c - V_e, 1e-12) * dpp ** -c["co_lef_expdp"]
                    * P3_co ** -c["co_lef_expP3"]))
    ei_co_lef = _clip_ei(ei_co_lef)

    # --- V_c óptimo analítico NOx ↔ CO [§8.2] -----------------------------
    a_nox = fan["lefebvre"] / V_c                  # EI_NOx = a·V_c
    b_co = ei_co_lef * max(V_c - V_e, 1e-12)       # EI_CO  = b/(V_c−V_e)
    V_opt = V_e + math.sqrt(max(b_co / max(NOX_CO_WEIGHT_RATIO * a_nox,
                                           1e-30), 0.0))
    return dict(
        EI_NOx_lefebvre=fan["lefebvre"],
        EI_CO_lefebvre=ei_co_lef,
        EI_NOx_rizk_mongia=fan["rizk_mongia"],
        EI_NOx_zonal=fan["zonal"],
        EI_NOx_odgers=fan["odgers"],
        EI_NOx_laviolette=fan["laviolette"],
        EI_NOx_flags=flags, EI_NOx_fan=stats,
        V_c_opt_m3=float(min(max(V_opt, 0.0), 1e3)),
        V_evap_m3=float(V_e), tau_evap_ms=float(1e3 * tau_e_s),
        tau_pz_ms_emis=float(1e3 * tau_pz_s))


def _gate_fuel(theta: np.ndarray) -> str:
    """Nombre del combustible desde la gate (helper local de emisiones)."""
    return decode_gates(theta)["fuel"]


# ==========================================================================
# 16. Vector de restricciones g <= 0 (contrato §5.1, orden CONGELADO)
# ==========================================================================
CONSTRAINT_NAMES = (
    "PF", "T_wall", "LBO", "tau_min", "tau_max", "M_exit", "dp",
    "t_liner", "d_eff", "phi_pz_rich", "eta_comb", "mixing",
)
assert len(CONSTRAINT_NAMES) == N_CONSTRAINTS


def _band_violation(x: float, lo: float, hi: float) -> float:
    """Violación CONTINUA y normalizada de la banda `[lo, hi]`.

    `≤ 0` dentro de la banda (y tanto más negativo cuanto más centrado el
    valor), `> 0` fuera y creciendo linealmente con la distancia relativa
    al borde infringido. Es la forma que la dominancia restringida de Deb
    necesita para poder ORDENAR diseños infactibles entre sí. Nunca lanza:
    entrada no finita ⇒ `G_DEGENERATE`.
    """
    if not np.isfinite(x):
        return G_DEGENERATE
    lo = float(lo) if np.isfinite(lo) and lo > 0.0 else 1e-9
    hi = float(hi) if np.isfinite(hi) and hi > lo else lo * 10.0
    return float(max((lo - x) / lo, (x - hi) / hi))


def soft_constraints(record: dict) -> dict:
    """Restricciones BLANDAS del record (`≤ 0` ⇔ dentro de banda).

    No entran en `g` (contrato §5.1 congela las 12) ni en `feasible`: son
    información de diseño que el reporte pinta y que `DesignSpec` puede
    promover a dura. Nunca lanza. Ver `SOFT_CONSTRAINT_NAMES`.
    """
    try:
        c = coeffs()
        return {
            "I_band": _band_violation(float(record.get("I_MW_m3_atm", 0.0)),
                                      c["I_min"], c["I_max"]),
            "M_design": _band_violation(
                float(record.get("blowing_ratio_M", 0.0)),
                c["M_eff_min"], c["M_eff_max"]),
        }
    except Exception:
        return {k: G_DEGENERATE for k in SOFT_CONSTRAINT_NAMES}


def _constraints(record: dict, theta: np.ndarray) -> list[float]:
    """`g <= 0` ⇔ factible. Longitud fija `N_CONSTRAINTS = 12`. **Nunca lanza**.

    La MAGNITUD es el grado de violación, y es CONTINUA: la dominancia
    restringida de Deb (capa de búsqueda) necesita ordenar individuos
    infactibles entre sí, no solo saber que lo son. Entrada degenerada
    (NaN/inf/negativos, record incompleto) ⇒ TODOS los g a `G_DEGENERATE`.

    Orden congelado (contrato §5.1) — cambiarlo obliga a tocar
    `neural_optimizer`, `dfam_checker`, los tests y las anclas de regresión:
    PF · T_wall · LBO · tau_min · tau_max · M_exit · dp · t_liner · d_eff ·
    phi_pz_rich · eta_comb · mixing.

    El límite de pared sale de `structures_core.MATERIALS`
    (`T_max_flame_wall_K − T_WALL_MARGIN_K`) y el espesor mínimo, de
    `lpbf_t_min_mm` del MISMO material — física y fabricabilidad comparten
    la fuente de verdad. La envolvente (D y L máximos) NO va aquí: es
    restricción de *spec* y vive en `DesignSpec.constraints` (§5.1).

    DESVIACIÓN DECLARADA DEL CONTRATO §5.1 (auditoría 2026-07-30)
    ------------------------------------------------------------
    `g[8]` («d_eff») pasa de ser `D_EFF_MIN_MM/d_eff_mm − 1` a ser el
    **máximo** de esa forma y de la violación continua del rango de VALIDEZ
    del mapa de eficacia de efusión:

        g[8] = max( D_EFF_MIN_MM/d_eff_mm − 1,
                    (M_valid_min − M)/M_valid_min,
                    (M − M_valid_max)/M_valid_max )

    Sigue siendo una sola restricción sobre el diseño de efusión, sigue
    siendo continua, sigue reduciéndose a la forma del contrato cuando M
    está dentro del mapa (los dos términos nuevos son negativos), y la
    longitud y el orden de `g` no cambian. El motivo: fuera del rango
    tabulado la eficacia de la película NO está modelada —la interpolación
    es un clamp— y un diseño así no puede declararse factible sin decirlo.
    La banda de DISEÑO, más estrecha (0.3–0.8), no endurece `g`: enciende
    `g_soft["M_design"]` y un aviso.
    """
    try:
        c = coeffs()
        t = np.asarray(theta, dtype=float).ravel()
        t_liner = float(t[14])
        d_eff = float(t[8])
        pf_max = max(c["pf_max"], 1e-6)
        T_lim = max(float(record["T_max_flame_wall_K"]) - T_WALL_MARGIN_K, 1.0)
        g = [
            float(record["PF"]) / pf_max - 1.0,
            float(record["T_w1_K"]) / T_lim - 1.0,
            (float(record["phi_LBO"]) * (1.0 + c["lbo_margin_min"])
             / max(float(record["phi_pz"]), 1e-9) - 1.0),
            TAU_MIN_MS / max(float(record["tau_ms"]), 1e-9) - 1.0,
            float(record["tau_ms"]) / TAU_MAX_MS - 1.0,
            float(record["M_exit"]) / M_EXIT_MAX - 1.0,
            float(record["dp_over_p"]) / DP_MAX_DEFAULT - 1.0,
            float(record["lpbf_t_min_mm"]) / max(t_liner, 1e-9) - 1.0,
            max(D_EFF_MIN_MM / max(d_eff, 1e-9) - 1.0,
                _band_violation(float(record.get("blowing_ratio_M", 0.0)),
                                c["M_valid_min"], c["M_valid_max"])),
            float(record["phi_pz"]) / PHI_PZ_MAX - 1.0,
            ETA_COMB_MIN / max(float(record["eta_comb"]), 1e-9) - 1.0,
            (abs(float(record["Y_over_DL_dz"]) - Y_TARGET_DEFAULT) / Y_TOL
             - 1.0),
        ]
        out = []
        for v in g:
            out.append(float(v) if np.isfinite(v) else G_DEGENERATE)
        return out
    except Exception:
        return [G_DEGENERATE] * N_CONSTRAINTS


# ==========================================================================
# 17. Calibración de alta fidelidad (hook L2 — patrón de la familia)
# ==========================================================================
OUT_KEYS = ["eta_comb", "dp_over_p", "PF", "T_wall_max_K", "EI_NOx"]


class HiFiCalibration:
    """Corrección afín por salida: `y_cal = a·y + b`, ajustada con pares
    (y_modelo, y_hifi). **Con menos de 2 pares aplica la identidad**, que es
    lo que garantiza que un sistema sin datos de CFD/banco se comporte
    exactamente como el L0 puro."""

    KEYS = tuple(OUT_KEYS)

    def __init__(self):
        self.pairs: dict[str, list[tuple[float, float]]] = {
            k: [] for k in self.KEYS}
        self.coef: dict[str, tuple[float, float]] = {
            k: (1.0, 0.0) for k in self.KEYS}

    def register(self, y_model: dict, y_hifi: dict):
        for k in self.KEYS:
            if k in y_model and k in y_hifi:
                try:
                    self.pairs[k].append((float(y_model[k]),
                                          float(y_hifi[k])))
                except (TypeError, ValueError):
                    continue
        self._refit()

    def _refit(self):
        for k, p in self.pairs.items():
            if len(p) >= 2:
                x = np.array([q[0] for q in p])
                y = np.array([q[1] for q in p])
                if np.ptp(x) > 1e-12:
                    a, b = np.polyfit(x, y, 1)
                    self.coef[k] = (float(a), float(b))

    def apply(self, rec: dict) -> dict:
        rec = dict(rec)
        for k, (a, b) in self.coef.items():
            if k in rec and (a, b) != (1.0, 0.0) \
                    and isinstance(rec[k], (int, float)):
                rec[k + "_raw"] = rec[k]
                rec[k] = a * rec[k] + b
                rec["calibrated"] = True
        return rec


CALIBRATION = HiFiCalibration()


def register_hifi_pair(theta: np.ndarray, y_hifi: dict,
                       fidelity: Fidelity = Fidelity.L0):
    """Reinyecta un resultado de CFD/banco al lazo (corrección afín con
    datos de alta fidelidad).

    Evalúa el MISMO θ sin calibrar (para no realimentar la corrección sobre
    sí misma) y registra el par (modelo, alta fidelidad).
    """
    base = evaluate(theta, fidelity=fidelity, use_cache=True,
                    calibrate=False)
    CALIBRATION.register(base, y_hifi)


# ==========================================================================
# 18. Caché persistente (contrato §2: SOLO fidelity >= L1 y use_cache=True)
# ==========================================================================
_CACHE: dict[str, dict] = {}

#: Nombre canónico del caché persistente (contrato §2). NO se activa solo:
#: hay que pedirlo con `set_cache_path()` o con la variable de entorno
#: `PHYCB_CACHE` — igual que en los hermanos, para que importar el módulo
#: nunca cree archivos por sorpresa.
CACHE_FILENAME = "phys_cache.jsonl"
CACHE_PATH = os.environ.get("PHYCB_CACHE", "")


def _theta_key(theta: np.ndarray, fidelity: int, extra: str = "") -> str:
    arr = np.round(np.asarray(theta, dtype=float), 9)
    return hashlib.sha1(arr.tobytes() + bytes([int(fidelity) & 0xFF])
                        + extra.encode()).hexdigest()


def set_cache_path(path: str):
    """Activa (o cambia) el caché persistente y carga lo ya escrito."""
    global CACHE_PATH
    CACHE_PATH = path or ""
    _cache_load()


def _cache_load():
    if CACHE_PATH and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        _CACHE[rec.pop("_key")] = rec
                    except Exception:
                        pass
        except Exception:
            pass


def _cache_store(key: str, rec: dict):
    """Guarda en memoria y —solo para L1+— también en disco.

    El L0 se recomputa en ~1 ms: escribirlo a disco costaría más que
    calcularlo y ensuciaría el JSONL. Con `use_cache=False` esta función NO
    se llama en absoluto (ni lectura ni escritura) — es el bug real de
    Phy-CC que el contrato §2 obliga a no repetir.
    """
    _CACHE[key] = rec
    if CACHE_PATH and int(rec.get("fidelity", 0)) >= int(Fidelity.L1):
        try:
            with open(CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"_key": key, **rec},
                                   ensure_ascii=False) + "\n")
        except Exception:
            pass


_cache_load()


# ==========================================================================
# 19. Cadena L0 completa y contrato `evaluate`
# ==========================================================================
SCHEMA = "phycb-record-1"
FAR_ITERS = 3          # pasadas del cierre η_comb ↔ FAR (bucle ACOTADO)
FAR_TOL = 1.0e-9       # test de convergencia sobre η_comb


def _safe(x, default: float = 0.0) -> float:
    """float finito o `default` (el record se serializa a JSON)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def _l0_chain(theta: np.ndarray, gates: dict, dp_target: float,
              material: str, use_cantera: bool) -> dict:
    """Compone la cadena L0 completa (§§2–8 de Ciencia) sobre un θ válido.

    Cierre η_comb ↔ FAR: el combustible necesario para llegar a T₄ depende
    de la eficiencia de combustión, que a su vez depende de las riquezas de
    zona (y por tanto del combustible). Se cierra con un bucle ACOTADO de
    `FAR_ITERS` pasadas y test de convergencia sobre η_comb — nunca un
    `while` abierto. La primera pasada arranca con `ETA_COMB_CYCLE = 0.995`
    (el relleno del ciclo), de modo que un combustor «perfecto» reproduce
    exactamente el FAR de `cycle_seam.hot_cycle`.
    """
    P3 = float(theta[IDX_P3_PA])
    T3 = float(theta[IDX_T3_K])
    mdot_air = float(theta[IDX_MDOT_KGS])
    T4 = float(theta[IDX_T4_K])
    fuel = gates["fuel"]

    th_st = _thermo_hot(fuel, 1.0, P3, T3, use_cantera=use_cantera)
    far_st = th_st["far_stoich"]
    T_st = th_st["T_ad_K"]

    eta_guess = ETA_COMB_CYCLE
    far_cycle = _far_from_T4(fuel, T3, T4, ETA_COMB_CYCLE)
    out: dict = {}
    for it in range(FAR_ITERS):
        far = _far_from_T4(fuel, T3, T4, eta_guess)
        mdot_fuel = far * mdot_air
        mdot4 = mdot_air + mdot_fuel

        part = _air_partition(theta, far, far_st)
        sz = _sizing(theta, gates, mdot_air, mdot_fuel, dp_target)

        th_pz = _thermo_hot(fuel, part["phi_pz"], P3, T3, use_cantera)
        th_sz = _thermo_hot(fuel, part["phi_sz"], P3, T3, use_cantera)
        th_dz = _thermo_hot(fuel, part["phi_global"], P3, T3, use_cantera)
        mw_hot = th_pz["MW_kgkmol"]
        sp = part["splits"]
        T_zone = dict(
            T_pz_K=th_pz["T_ad_K"], T_sz_K=th_sz["T_ad_K"],
            T_dz_K=th_dz["T_ad_K"],
            # `mdot_air_kgs`/`mdot_fuel_kgs`: los necesita el térmico 1D para
            # cerrar por MASA el caudal de efusión (ṁ_cool = frac_cool·ṁ_A).
            mdot_air_kgs=mdot_air, mdot_fuel_kgs=mdot_fuel,
            mdot_pz_kgs=mdot_fuel + sp["pz"] * mdot_air,
            mdot_sz_kgs=mdot_fuel + (sp["pz"] + sp["sz"]) * mdot_air,
            mdot4_kgs=mdot4,
            mdot_an_pz_kgs=max((1.0 - sp["pz"]) * mdot_air, 1e-6),
            mdot_an_sz_kgs=max((1.0 - sp["pz"] - sp["sz"]) * mdot_air, 1e-6),
            mdot_an_dz_kgs=max((sp["dz"] + sp["cool"]) * mdot_air, 1e-6))

        holes = _holes(theta, gates, sz, part, mdot_air, mdot_fuel, T_zone,
                       mw_hot, dp_target)
        atom = _atomization(theta, gates, sz, sz["rho3_kgm3"],
                            holes["dp_liner_Pa"], mdot_fuel, T3)
        eff = _efficiency(sz, part, holes, atom, mdot_air, fuel)
        out = dict(far=far, mdot_fuel=mdot_fuel, mdot4=mdot4, part=part,
                   sz=sz, T_zone=T_zone, holes=holes, atom=atom, eff=eff,
                   mw_hot=mw_hot, T_st=T_st, th_st=th_st, th_pz=th_pz,
                   th_sz=th_sz, th_dz=th_dz, far_cycle=far_cycle,
                   far_iters=it + 1)
        if abs(eff["eta_comb"] - eta_guess) <= FAR_TOL:
            break
        eta_guess = eff["eta_comb"]

    part, sz, holes, atom = out["part"], out["sz"], out["holes"], out["atom"]
    T_zone, mw_hot = out["T_zone"], out["mw_hot"]
    stab = _stability(theta, gates, sz, part, atom, mdot_air, T_st, mw_hot)
    therm = _liner_thermal_1d(theta, gates, sz, part, holes, T_zone, mw_hot,
                              out["far"], material=material)
    pf = _pattern_factor(sz, holes, theta)
    emis = _emissions(theta, sz, part, holes, T_zone, atom, mdot_air, T_st)

    # --- salida y acople de ciclo -----------------------------------------
    R_hot = _r_gas(mw_hot)
    P04 = P3 * (1.0 - holes["dp_over_p"])
    A4 = A4_OVER_AL * sz["_A_L"]
    rho4 = P04 / (R_hot * max(T4, 200.0))
    U4 = out["mdot4"] / max(rho4 * A4, 1e-12)
    a4 = math.sqrt(GAMMA_HOT * R_hot * max(T4, 200.0))
    M_exit = U4 / max(a4, 1e-6)
    bleed = max(0.0, (T4 - TURBINE_UNCOOLED_T_MAX_K)) * BLEED_FRAC_PER_K

    out.update(stab=stab, therm=therm, pf=pf, emis=emis, P04=P04, A4=A4,
               M_exit=M_exit, bleed=bleed, R_hot=R_hot)
    return out


def _record_from_chain(theta: np.ndarray, gates: dict, ch: dict,
                       source: str, fidelity: int) -> dict:
    """Ensambla el record `phycb-record-1` completo (contrato §4.1 + §4.2)."""
    sz, part, holes = ch["sz"], ch["part"], ch["holes"]
    atom, eff, stab = ch["atom"], ch["eff"], ch["stab"]
    therm, pf, emis, T_zone = ch["therm"], ch["pf"], ch["emis"], ch["T_zone"]
    T4 = float(theta[IDX_T4_K])
    rec: dict = {
        # --- meta ---------------------------------------------------------
        "schema": SCHEMA, "fidelity": int(fidelity), "source": source,
        "seed": SEED,
        "arch": gates["arch"], "injector": gates["injector"],
        "fuel": gates["fuel"],
        # --- mínimo obligatorio (contrato §4.1) ---------------------------
        "eta_comb": _safe(eff["eta_comb"]),
        "dp_over_p": _safe(holes["dp_over_p"]),
        "PF": _safe(pf["PF"]),
        "T_wall_max_K": _safe(therm["T_wall_max_K"]),
        "T04_K": _safe(T4),
        "P04_Pa": _safe(ch["P04"]),
        "f_fuel_air": _safe(ch["far"]),
        "EI_NOx": _safe(emis["EI_NOx_lefebvre"]),
        "EI_CO": _safe(emis["EI_CO_lefebvre"]),
        "phi_pz": _safe(part["phi_pz"]),
        "tau_ms": _safe(sz["tau_ms"]),
        "SMD_um": (None if atom["SMD_um"] is None
                   else _safe(atom["SMD_um"])),
        # --- termoquímica [Ciencia §9.3, §2] -------------------------------
        "T_ad_st_K": _safe(ch["T_st"]),
        "T_pz_K": _safe(T_zone["T_pz_K"]),
        "T_sz_K": _safe(T_zone["T_sz_K"]),
        "T_dz_K": _safe(T_zone["T_dz_K"]),
        "phi_sz": _safe(part["phi_sz"]),
        "phi_global": _safe(part["phi_global"]),
        "far_stoich": _safe(ch["th_st"]["far_stoich"]),
        "f_fuel_air_cycle": _safe(ch["far_cycle"]),
        "rho3_kgm3": _safe(sz["rho3_kgm3"]),
        "mdot_air_kgs": _safe(theta[IDX_MDOT_KGS]),
        "mdot_fuel_kgs": _safe(ch["mdot_fuel"]),
        "mdot4_kgs": _safe(ch["mdot4"]),
        "LHV_J_kg": _safe(FUELS[gates["fuel"]]["LHV"]),
        "MW_hot_kgkmol": _safe(ch["mw_hot"]),
        "R_hot_J_kgK": _safe(ch["R_hot"]),
        "thermo_source": ch["th_pz"]["source"],
        # --- dimensionado [Ciencia §2] -------------------------------------
        "A_ref_m2": _safe(sz["A_ref_m2"]),
        "A_ref_aero_m2": _safe(sz["A_ref_aero_m2"]),
        "A_ref_chem_m2": _safe(sz["A_ref_chem_m2"]),
        "sizing_driver": sz["sizing_driver"],
        "sizing_converged": bool(sz["sizing_converged"]),
        "D_ref_mm": _safe(sz["D_ref_mm"]),
        "D_L_mm": _safe(sz["D_L_mm"]),
        "D_casing_mm": _safe(sz["D_casing_mm"]),
        "A_L_m2": _safe(sz["A_L_m2"]),
        "A_an_m2": _safe(sz["A_an_m2"]),
        "U_ref_ms": _safe(sz["U_ref_ms"]),
        "q_ref_Pa": _safe(sz["q_ref_Pa"]),
        "L_pz_mm": _safe(sz["L_pz_mm"]),
        "L_sz_mm": _safe(sz["L_sz_mm"]),
        "L_dz_mm": _safe(sz["L_dz_mm"]),
        "L_tot_mm": _safe(sz["L_tot_mm"]),
        "V_c_m3": _safe(sz["V_c_m3"]),
        "V_pz_m3": _safe(sz["V_pz_m3"]),
        "V_sz_m3": _safe(sz["V_sz_m3"]),
        "V_dz_m3": _safe(sz["V_dz_m3"]),
        "tau_pz_ms": _safe(eff["tau_pz_ms"]),
        "I_MW_m3_atm": _safe(sz["I_MW_m3_atm"]),
        "theta_lefebvre": _safe(sz["theta_lefebvre"]),
        "Omega_load": _safe(sz["Omega_load"]),
        # --- reparto de aire [Ciencia §3] ----------------------------------
        "frac_cool": _safe(part["frac_cool"]),
        "splits": {k: _safe(v) for k, v in part["splits"].items()},
        # --- orificios y pérdidas [Ciencia §4] -----------------------------
        "hole_rows": holes["hole_rows"],
        "A_h_total_m2": _safe(holes["A_h_total_m2"]),
        "dp_liner_over_qref": _safe(holes["dp_liner_over_qref"]),
        "dp_sw_over_qref": _safe(holes["dp_sw_over_qref"]),
        "dp_liner_Pa": _safe(holes["dp_liner_Pa"]),
        "dp_sw_Pa": _safe(holes["dp_sw_Pa"]),
        "dp_Pa": _safe(holes["dp_Pa"]),
        "J_dz": _safe(holes["J_dz"]),
        "Y_over_DL_dz": _safe(holes["Y_over_DL_dz"]),
        "swirler": {k: (v if isinstance(v, bool) else _safe(v))
                    for k, v in holes["swirler"].items()},
        # --- atomización [Ciencia §7] --------------------------------------
        "SMD_um_by_corr": {k: (None if v is None else _safe(v))
                           for k, v in atom["SMD_um_by_corr"].items()},
        "atomizer_model": atom["atomizer_model"],
        "ALR_eff": (None if atom["ALR_eff"] is None
                    else _safe(atom["ALR_eff"])),
        "atomizer": {k: (v if isinstance(v, bool) else _safe(v))
                     for k, v in atom["atomizer"].items()},
        # --- eficiencia [Ciencia §2.3] -------------------------------------
        "eta_theta": _safe(eff["eta_theta"]),
        "eta_evap": _safe(eff["eta_evap"]),
        "eta_mix": _safe(eff["eta_mix"]),
        "eta_driver": eff["eta_driver"],
        # --- estabilidad [Ciencia §6] --------------------------------------
        "phi_LBO": _safe(stab["phi_LBO"]),
        "LBO_margin": _safe(stab["LBO_margin"]),
        "LBO_margin_ratio": _safe(stab["LBO_margin_ratio"]),
        "LBO_margin_frac": _safe(stab["LBO_margin_frac"]),
        "LBO_margin_def": stab["LBO_margin_def"],
        "E_min_J": _safe(stab["E_min_J"]),
        "d_q_mm": _safe(stab["d_q_mm"]),
        "lbo_branch": stab["lbo_branch"],
        # --- térmico 1D [Ciencia §5] ---------------------------------------
        "T_w1_K": _safe(therm["T_w1_K"]),
        "T_w2_K": _safe(therm["T_w2_K"]),
        "eta_film": _safe(therm["eta_film"]),
        "eta_film_saturated": bool(therm["eta_film_saturated"]),
        "blowing_ratio_M": _safe(therm["blowing_ratio_M"]),
        "blowing_ratio_M_min": _safe(therm["blowing_ratio_M_min"]),
        "blowing_ratio_M_max": _safe(therm["blowing_ratio_M_max"]),
        "blowing_ratio_M_profile": [_safe(v)
                                    for v in therm["blowing_ratio_M_profile"]],
        "M_design_band": [_safe(v) for v in therm["M_design_band"]],
        "M_map_range": [_safe(v) for v in therm["M_map_range"]],
        "U_cool_ms": _safe(therm["U_cool_ms"]),
        "U_cool_dp_ms": _safe(therm["U_cool_dp_ms"]),
        "effusion_area_match": _safe(therm["effusion_area_match"]),
        "A_effusion_m2": _safe(therm["A_effusion_m2"]),
        "mdot_cool_kgs": _safe(therm["mdot_cool_kgs"]),
        "T_wall_profile": therm["T_wall_profile"],
        "q_wall_W_m2": _safe(therm["q_wall_W_m2"]),
        "thermal_residual_rel": _safe(therm["thermal_residual_rel"]),
        "eps_gas": _safe(therm["eps_gas"]),
        "n_eff_holes": int(therm["n_eff_holes"]),
        "effusion_porosity": _safe(therm["effusion_porosity"]),
        #  FUENTE ÚNICA §12.b — el contrato de geometría, la red 1D y el LBM
        #  contrastan CONTRA estas tres claves.
        "effusion_wall_area_m2": _safe(therm["effusion_wall_area_m2"]),
        "effusion_Cd": _safe(therm["effusion_Cd"]),
        "effusion_walls": list(therm["effusion_walls"]),
        "material": therm["material"],
        "T_max_flame_wall_K": _safe(therm["T_max_flame_wall_K"]),
        "lpbf_t_min_mm": _safe(therm["lpbf_t_min_mm"]),
        # --- factor de patrón [Ciencia §4.3] -------------------------------
        "PF_star": _safe(pf["PF_star"]),
        "RTDF": _safe(pf["RTDF"]),
        "T4_max_K": _safe(pf["T4_max_K"]),
        # --- emisiones [Ciencia §8] — TODAS reportadas ----------------------
        "EI_NOx_lefebvre": _safe(emis["EI_NOx_lefebvre"]),
        "EI_NOx_rizk_mongia": _safe(emis["EI_NOx_rizk_mongia"]),
        "EI_NOx_odgers": _safe(emis["EI_NOx_odgers"]),
        "EI_NOx_laviolette": _safe(emis["EI_NOx_laviolette"]),
        "EI_NOx_zonal": _safe(emis["EI_NOx_zonal"]),
        "EI_CO_lefebvre": _safe(emis["EI_CO_lefebvre"]),
        # Pantalla de plausibilidad (§15.0): banderas hermanas y estadísticos
        # del abanico calculados SOLO con las correlaciones en rango.
        "EI_NOx_flags": dict(emis["EI_NOx_flags"]),
        "EI_NOx_fan": dict(emis["EI_NOx_fan"]),
        "V_c_opt_m3": _safe(emis["V_c_opt_m3"]),
        # --- salida --------------------------------------------------------
        "M_exit": _safe(ch["M_exit"]),
        "A4_m2": _safe(ch["A4"]),
        # --- integración de ciclo [WAVE4] ----------------------------------
        "bleed_frac_required": _safe(ch["bleed"]),
    }
    # Eco de θ: una clave float por cada VAR_NAMES (contrato §4.2).
    # COLISIÓN RESUELTA: `arch`, `injector` y `fuel` son a la vez nombres de
    # variable de θ y claves de meta con valor STRING (contrato §4.2, primera
    # línea). Manda el string —es lo que leen el reporte, la CLI y
    # `neural_optimizer`— y el eco numérico de las tres gates va a
    # `arch_id`/`injector_id`/`fuel_id`, que son además los enteros ya
    # decodificados por `decode_gates` (no el float crudo de θ).
    for n, v in zip(VAR_NAMES, theta):
        if n in ("arch", "injector", "fuel"):
            continue
        rec[n] = _safe(v)
    rec["arch_id"] = int(gates["arch_id"])
    rec["injector_id"] = int(gates["injector_id"])
    rec["fuel_id"] = int(gates["fuel_id"])
    rec["I_band_MW_m3_atm"] = [_safe(coeffs()["I_min"]),
                               _safe(coeffs()["I_max"])]
    rec["g"] = _constraints(rec, theta)
    rec["g_soft"] = soft_constraints(rec)
    rec["feasible"] = bool(all(v <= 0.0 for v in rec["g"]))
    if rec["g_soft"].get("I_band", 0.0) > 0.0:
        _warn_provisional(
            "I_out_of_band",
            f"combustion intensity I = {rec['I_MW_m3_atm']:.1f} MW/m³·atm "
            f"OUTSIDE the band {rec['I_band_MW_m3_atm'][0]:.0f}–"
            f"{rec['I_band_MW_m3_atm'][1]:.0f} of "
            f"reference/class_table.csv (Ciencia §2.4). The units are "
            f"correct (P₃ in atm): it is design information, "
            f"I = FAR·LHV·P_atm/(R·T₃·τ) with "
            f"FAR = {rec['f_fuel_air']:.4f} and τ = {rec['tau_ms']:.2f} ms "
            f"admits no other value. SOFT constraint g_soft['I_band'] = "
            f"{rec['g_soft']['I_band']:+.3f}.")
    rec["limitations"] = _current_limitations()
    return rec


def _degenerate_record(theta, source: str, reason: str,
                       fidelity: int = int(Fidelity.L0)) -> dict:
    """Record canónico ante entrada degenerada (contrato §5.1).

    Todas las claves mínimas presentes y con el tipo correcto, `g` a
    `G_DEGENERATE`, `feasible=False` y `source` con sufijo `(degenerate)`.
    Ninguna excepción sale nunca de `evaluate`.
    """
    try:
        gates = decode_gates(np.asarray(theta, dtype=float).ravel())
    except Exception:
        gates = dict(arch="can", injector="airblast", fuel="jet_a")
    rec = {
        "schema": SCHEMA, "fidelity": int(fidelity),
        "source": source + "(degenerate)", "seed": SEED,
        "degenerate_reason": reason,
        "arch": gates["arch"], "injector": gates["injector"],
        "fuel": gates["fuel"],
        "eta_comb": 0.0, "dp_over_p": 1.0, "PF": 10.0, "T_wall_max_K": 0.0,
        "T04_K": 0.0, "P04_Pa": 0.0, "f_fuel_air": 0.0, "EI_NOx": 0.0,
        "EI_CO": 0.0, "phi_pz": 0.0, "tau_ms": 0.0, "SMD_um": None,
        "T_ad_st_K": 0.0, "T_pz_K": 0.0, "T_sz_K": 0.0, "T_dz_K": 0.0,
        "phi_sz": 0.0, "phi_global": 0.0, "far_stoich": 0.0,
        "f_fuel_air_cycle": 0.0, "rho3_kgm3": 0.0,
        "mdot_air_kgs": 0.0, "mdot_fuel_kgs": 0.0, "mdot4_kgs": 0.0,
        "LHV_J_kg": 0.0, "MW_hot_kgkmol": 0.0, "R_hot_J_kgK": 0.0,
        "thermo_source": "none", "A_ref_m2": 0.0, "A_ref_aero_m2": 0.0,
        "A_ref_chem_m2": 0.0, "sizing_driver": "none",
        "sizing_converged": False, "D_ref_mm": 0.0,
        "D_L_mm": 0.0, "D_casing_mm": 0.0, "A_L_m2": 0.0, "A_an_m2": 0.0,
        "U_ref_ms": 0.0, "q_ref_Pa": 0.0, "L_pz_mm": 0.0, "L_sz_mm": 0.0,
        "L_dz_mm": 0.0, "L_tot_mm": 0.0, "V_c_m3": 0.0, "V_pz_m3": 0.0,
        "V_sz_m3": 0.0, "V_dz_m3": 0.0, "tau_pz_ms": 0.0,
        "I_MW_m3_atm": 0.0, "theta_lefebvre": 0.0, "Omega_load": 0.0,
        "frac_cool": 0.0,
        "splits": {"pz": 0.0, "sz": 0.0, "dz": 0.0, "cool": 0.0},
        "hole_rows": [], "A_h_total_m2": 0.0, "dp_liner_over_qref": 0.0,
        "dp_sw_over_qref": 0.0, "dp_liner_Pa": 0.0, "dp_sw_Pa": 0.0,
        "dp_Pa": 0.0, "J_dz": 0.0, "Y_over_DL_dz": 0.0, "swirler": {},
        "SMD_um_by_corr": {}, "atomizer_model": "none", "ALR_eff": None,
        "atomizer": {}, "eta_theta": 0.0, "eta_evap": 0.0, "eta_mix": 0.0,
        "eta_driver": "none", "phi_LBO": 10.0, "LBO_margin": -1.0,
        "E_min_J": 0.0, "d_q_mm": 0.0, "lbo_branch": "none",
        "T_w1_K": 0.0, "T_w2_K": 0.0, "eta_film": 0.0,
        "blowing_ratio_M": 0.0, "T_wall_profile": [], "q_wall_W_m2": 0.0,
        "thermal_residual_rel": 0.0, "eps_gas": 0.0, "n_eff_holes": 0,
        "effusion_porosity": 0.0, "material": MATERIAL_DEFAULT,
        "T_max_flame_wall_K": 0.0, "lpbf_t_min_mm": 0.0,
        "PF_star": 0.0, "RTDF": 0.0, "T4_max_K": 0.0,
        "EI_NOx_lefebvre": 0.0, "EI_NOx_rizk_mongia": 0.0,
        "EI_NOx_odgers": 0.0, "EI_NOx_laviolette": 0.0, "EI_NOx_zonal": 0.0,
        "EI_CO_lefebvre": 0.0, "V_c_opt_m3": 0.0, "M_exit": 0.0,
        "A4_m2": 0.0, "bleed_frac_required": 0.0,
        "EI_NOx_flags": {k: "degenerate" for k in NOX_FAN_NAMES},
        "EI_NOx_fan": dict(min=0.0, max=0.0, median=0.0, mean=0.0,
                           spread_ratio=1.0, n_used=0,
                           n_excluded=len(NOX_FAN_NAMES), used=[],
                           excluded=list(NOX_FAN_NAMES),
                           band=list(EI_NOX_PLAUSIBLE_RANGE)),
        "blowing_ratio_M_min": 0.0, "blowing_ratio_M_max": 0.0,
        "U_cool_ms": 0.0, "U_cool_dp_ms": 0.0, "effusion_area_match": 0.0,
        "A_effusion_m2": 0.0, "eta_film_saturated": False,
        "effusion_wall_area_m2": 0.0, "effusion_Cd": 0.0,
        "effusion_walls": [],
        "LBO_margin_ratio": 0.0, "LBO_margin_frac": -1.0,
        "LBO_margin_def": LBO_MARGIN_DEF,
        "I_band_MW_m3_atm": [0.0, 0.0],
        "g_soft": {k: G_DEGENERATE for k in SOFT_CONSTRAINT_NAMES},
    }
    try:
        t = pad_theta(theta)
    except Exception:
        t = np.zeros(NDIM)
    for n, v in zip(VAR_NAMES, t):
        if n not in ("arch", "injector", "fuel"):
            rec[n] = _safe(v)
    rec["arch_id"] = int(gates.get("arch_id", 0))
    rec["injector_id"] = int(gates.get("injector_id", 0))
    rec["fuel_id"] = int(gates.get("fuel_id", 0))
    rec["g"] = [G_DEGENERATE] * N_CONSTRAINTS
    rec["feasible"] = False
    rec["limitations"] = _current_limitations()
    return rec


def _validate_operating_point(theta: np.ndarray) -> str | None:
    """Devuelve el motivo de degeneración, o `None` si el θ es utilizable."""
    t = np.asarray(theta, dtype=float).ravel()
    if t.shape[0] < NDIM_LEGACY or not np.all(np.isfinite(t)):
        return "theta con NaN/inf o longitud insuficiente"
    for idx, name in ((IDX_P3_PA, "P3_Pa"), (IDX_T3_K, "T3_K"),
                      (IDX_MDOT_KGS, "mdot_kgs"), (IDX_T4_K, "T4_K")):
        if not (t[idx] > 0.0):
            return f"{name} <= 0"
    if t[IDX_T4_K] <= t[IDX_T3_K]:
        return "T4_K <= T3_K (no hay aporte de calor)"
    if t[14] <= 0.0 or t[8] <= 0.0:
        return "espesor de liner o Ø de efusión <= 0"
    if not (t[2] > 0.0):
        return "frac_pz <= 0 (no hay zona primaria)"
    return None


L1_TIMEOUT_S = float(os.environ.get("PHYCB_L1_TIMEOUT", "120"))


def _try_l1(theta: np.ndarray, rec: dict, fid: int) -> dict:
    """Intento de L1 (red 1D + CRN) con degradación etiquetada.

    Contrato §3 de etiquetas `source`:
      `crn_L1` · `crn_L1(cantera_unavailable→L0)` · `crn_L1(timeout→L0)` ·
      `crn_L1(diverged→L0)`.

    `crn_core` es de WAVE2 y **hoy NO EXISTE**: el `ImportError` se captura
    y el punto degrada a L0 etiquetado, sin excepción y sin coste. Cuando
    aparezca, debe exponer `crn_solve(theta, rec, timeout_s) -> dict | None`
    (`None` = divergido/timeout) y `CANTERA_AVAILABLE`; sus claves se funden
    sobre el record (contrato §4.3: `splits_solved`, `v_zones`,
    `tau_zones_ms`, `T_zones_K`, `EI_NOx_crn`, `Da_pz`, …).
    **Este archivo no habrá que tocarlo el día que WAVE2 aterrice.**
    """
    try:
        import importlib
        crn = importlib.import_module("crn_core")
        here = os.path.dirname(os.path.abspath(__file__))
        there = os.path.dirname(os.path.abspath(getattr(crn, "__file__", "")))
        if os.path.normcase(here) != os.path.normcase(there):
            raise ImportError("crn_core resuelto fuera de Phy-CB")
    except ImportError:
        rec["source"] = "crn_L1(cantera_unavailable→L0)"
        return rec
    except Exception:
        rec["source"] = "crn_L1(cantera_unavailable→L0)"
        return rec
    if not getattr(crn, "CANTERA_AVAILABLE", False):
        rec["source"] = "crn_L1(cantera_unavailable→L0)"
        return rec
    try:
        sol = crn.crn_solve(theta, rec, L1_TIMEOUT_S)
    except Exception:
        sol = None
    if not sol:
        # `crn_core.LAST_STATUS` distingue POR QUÉ no hubo solución. El
        # contrato §3 exige etiquetas distintas: un timeout se arregla
        # subiendo PHYCB_L1_TIMEOUT, una divergencia no — colapsarlas
        # escondía información accionable (test T20c).
        status = str(getattr(crn, "LAST_STATUS", "") or "")
        why = _L1_STATUS_LABEL.get(status, "diverged")
        rec["source"] = f"crn_L1({why}→L0)"
        return rec
    rec.update(sol)
    rec["source"] = "crn_L1"
    return rec


#: `crn_core.LAST_STATUS` → sufijo de `source` (contrato §3).
_L1_STATUS_LABEL = {
    "timeout": "timeout",
    "diverged": "diverged",
    "degenerate": "diverged",
    "cantera_unavailable": "cantera_unavailable",
    "no_mechanism": "cantera_unavailable",
}


def _try_higher(theta: np.ndarray, rec: dict, fid: int) -> dict:
    """Fidelidades por encima de L1 (L1.5 · L2a · L2b), con la degradación
    etiquetada del contrato §3. Se llama DESPUÉS de `_try_l1`, así que el
    record ya trae lo mejor que L1 pudo dar.

    * **L1.5** (WAVE5-A): aero-frío LBM con XLB sobre los voxels → splits
      reales, penetraciones medidas y auto-topología del CRN. Import
      perezoso de `cfd.crn_from_field`; sin XLB/JAX degrada a lo que haya
      con `crn_L15(xlb_unavailable→L1)`.
    * **L2a** (WAVE6): IB-RANS nativo sobre el SDF. Import perezoso de
      `cfd.ibrans`; sin él, `ibrans_L2A(unavailable→L1)`.
    * **L2b**: **FUERA DE ALCANCE** por decisión de proyecto (contrato §12 —
      máquina Windows, OpenFOAM exigiría WSL2/Docker). No se intenta
      importar nada: se etiqueta y se sigue con lo que L2a/L1 dieran. El
      usuario ve en `source` exactamente por qué no obtuvo lo que pidió.
    """
    if fid == int(Fidelity.L2B):
        rec["source"] = f"openfoam_L2B(out_of_scope→{rec.get('source', 'L0')})"
        rec["fidelity_note"] = (
            "L2b (OpenFOAM) está fuera del alcance del proyecto: máquina "
            "Windows, requeriría WSL2/Docker. Consecuencia declarada: Phy-CB "
            "no predice temperatura de metal cuantitativa en ningún nivel de "
            "fidelidad. Ver docs/IMPLEMENTATION_CONTRACT.md §12.")
        return rec

    mods = {int(Fidelity.L15): ("cfd.crn_from_field", "topology_from_field",
                                "crn_L15", "xlb_unavailable"),
            int(Fidelity.L2A): ("cfd.ibrans", "solve", "ibrans_L2A",
                                "unavailable")}
    name, func, tag, why = mods[fid]
    mod = None
    try:
        import importlib
        mod = importlib.import_module(name)
        sol = getattr(mod, func)(theta, rec)
    except Exception:
        sol = None
    if not sol:
        # el módulo puede declarar el porqué corto de SU degradación
        # (p. ej. cfd.ibrans.DEGRADE_TAG = "no_product" | "diverged"):
        # el sufijo del contrato §3 explica POR QUÉ se degradó, y
        # "unavailable" sería mentira si el solver corrió y falló.
        why = str(getattr(mod, "DEGRADE_TAG", "") or why) if mod else why
        rec["source"] = f"{tag}({why}→{rec.get('source', 'L0')})"
        return rec
    rec.update(sol)
    rec["source"] = tag
    return rec


def evaluate(theta: np.ndarray, fidelity: Fidelity = Fidelity.L0,
             use_cache: bool = True, calibrate: bool = True,
             dp_over_p_target: float | None = None,
             material: str = MATERIAL_DEFAULT,
             use_cantera: bool = False) -> dict:
    """Contrato único de la capa 1: θ → record `phycb-record-1`.

    * `fidelity = L0` (default) corre la cadena de correlaciones de este
      archivo en ~1 ms.
    * `fidelity >= L1` intenta el import **PEREZOSO** de `crn_core` (red 1D
      + CRN de Cantera, WAVE2) — ver `_try_l1`, que documenta el hook.
    * `use_cache=False` **no lee ni escribe** el caché (trampa CONTEXTO
      §10.5); el caché en disco solo recibe fidelidades ≥ L1.
    * `dp_over_p_target` sobreescribe el Δp/p de diseño (que en microturbina
      es RESTRICCIÓN del ciclo, no salida); `None` usa el nominal de la
      clase micro de `data/lefebvre/plf_by_arch.csv`.
    * `use_cantera=True` fuerza `equilibrate('HP')` en vez de la tabla de la
      capa 0 (más exacto, ~1000× más lento — fuera del lazo caliente).

    **Nunca lanza.** Cualquier fallo devuelve el record degenerado con
    `g = [10.0]·12`, `feasible=False` y `source` con sufijo `(degenerate)`.
    """
    with _capture_limitations():
        return _evaluate_inner(theta, fidelity, use_cache, calibrate,
                               dp_over_p_target, material, use_cantera)


def _evaluate_inner(theta, fidelity, use_cache, calibrate,
                    dp_over_p_target, material, use_cantera) -> dict:
    """Cuerpo de `evaluate`, ya dentro del contexto que aísla los avisos
    provisionales de ESTA evaluación (ver `_capture_limitations`)."""
    try:
        theta = pad_theta(theta)
    except Exception as exc:
        return _degenerate_record(theta, "correlations_L0",
                                  f"{type(exc).__name__}: {exc}")

    fid = int(fidelity)
    extra = f"{dp_over_p_target}|{material}|{int(bool(use_cantera))}"
    key = _theta_key(theta, fid, extra)
    if use_cache and key in _CACHE:
        rec = dict(_CACHE[key])
        return CALIBRATION.apply(rec) if calibrate else rec

    gates = decode_gates(theta)          # gates ANTES de cualquier float
    reason = _validate_operating_point(theta)
    if reason is not None:
        rec = _degenerate_record(theta, "correlations_L0", reason, fid)
        return CALIBRATION.apply(rec) if calibrate else rec

    dp_target = coeffs()["dp_over_p"] if dp_over_p_target is None \
        else float(dp_over_p_target)
    try:
        ch = _l0_chain(theta, gates, dp_target, material, bool(use_cantera))
        src = "correlations_L0" if cantera_available() \
            else "correlations_L0(cantera_unavailable)"
        rec = _record_from_chain(theta, gates, ch, src, fid)
    except Exception as exc:             # red de seguridad: g nunca lanza
        rec = _degenerate_record(theta, "correlations_L0",
                                 f"{type(exc).__name__}: {exc}", fid)
        return CALIBRATION.apply(rec) if calibrate else rec

    if fid >= int(Fidelity.L1):
        rec = _try_l1(theta, rec, fid)
    if fid >= int(Fidelity.L15):
        rec = _try_higher(theta, rec, fid)

    if use_cache:
        _cache_store(key, rec)
    return CALIBRATION.apply(rec) if calibrate else rec


def _eval_worker(args):
    theta, fid = args
    return evaluate(np.array(theta), Fidelity(fid))


def evaluate_batch(thetas: np.ndarray, fidelity: Fidelity = Fidelity.L0,
                   n_workers: int = 1) -> list[dict]:
    """Evalúa un lote. `n_workers>1` paraleliza por procesos (útil en L1+).

    En L0 el paralelismo no compensa (~1 ms por punto frente al coste de
    arrancar procesos), así que se ignora — igual que en los hermanos.
    """
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    if n_workers <= 1 or int(fidelity) <= int(Fidelity.L0):
        return [evaluate(t, fidelity) for t in thetas]
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(_eval_worker, [(t.tolist(), int(fidelity))
                                              for t in thetas]))
    except Exception:
        return [evaluate(t, fidelity) for t in thetas]


# ==========================================================================
# 20. Embedding físico para el surrogate (capa 2) — contrato §8
# ==========================================================================
FEATURE_NAMES = VAR_NAMES + [
    "f_phi_pz", "f_logtheta", "f_Omega", "f_tau", "f_I", "f_J_dz",
    "f_M_eff", "f_Tw1", "f_PF", "f_LBO", "f_SMD", "f_dpqref",
    "L0_eta_comb", "L0_dp", "L0_PF", "L0_Twall", "L0_logNOx",
]
N_FEATURES = NDIM + 17          # = 37 (contrato §8)
assert len(FEATURE_NAMES) == N_FEATURES


def physics_features(theta: np.ndarray, record: dict | None = None
                     ) -> np.ndarray:
    """θ normalizado + 12 features físicos + 5 predicciones L0 ⇒ 37-D.

    El ensemble aprende el **RESIDUO** sobre la física, no la física desde
    cero: por eso las cinco últimas componentes son las salidas del L0
    (`OUT_KEYS`) y no hay que reaprenderlas. `record` DEBE ser un record L0
    del MISMO θ (un record L1/L2 corrompería el residual learning); si no
    se pasa, se calcula.

    Todas las componentes están escaladas a O(1) y ACOTADAS: un θ degenerado
    produce un vector finito, nunca NaN (el MLP no puede recibir NaN).
    """
    theta = pad_theta(theta)
    rec = record if record is not None else evaluate(
        theta, Fidelity.L0, use_cache=True, calibrate=False)
    smd = rec.get("SMD_um")
    feats = np.array([
        min(max(_safe(rec.get("phi_pz")), 0.0), 3.0) / 3.0,
        math.log(max(_safe(rec.get("theta_lefebvre")), 1.0)) / math.log(1e9),
        min(max(_safe(rec.get("Omega_load")), 0.0), 1e-6) / 1e-6,
        min(max(_safe(rec.get("tau_ms")), 0.0), 30.0) / 30.0,
        min(max(_safe(rec.get("I_MW_m3_atm")), 0.0), 200.0) / 200.0,
        min(max(_safe(rec.get("J_dz")), 0.0), 100.0) / 100.0,
        min(max(_safe(rec.get("blowing_ratio_M")), 0.0), 10.0) / 10.0,
        min(max(_safe(rec.get("T_w1_K")), 0.0), 2000.0) / 2000.0,
        min(max(_safe(rec.get("PF")), 0.0), 2.0) / 2.0,
        min(max(_safe(rec.get("phi_LBO")), 0.0), 3.0) / 3.0,
        min(max(0.0 if smd is None else _safe(smd), 0.0), 300.0) / 300.0,
        min(max(_safe(rec.get("dp_liner_over_qref")), 0.0), 100.0) / 100.0,
        min(max(_safe(rec.get("eta_comb")), 0.0), 1.0),
        min(max(_safe(rec.get("dp_over_p")), 0.0), 0.2) / 0.2,
        min(max(_safe(rec.get("PF")), 0.0), 2.0) / 2.0,
        min(max(_safe(rec.get("T_wall_max_K")), 0.0), 2000.0) / 2000.0,
        math.log(max(_safe(rec.get("EI_NOx")), 1e-6)) / math.log(1e4),
    ])
    u = normalize(theta)
    u = np.where(np.isfinite(u), u, 0.0)
    return np.concatenate([u, feats])


# ==========================================================================
# 21. Smoke test / demostración (estilo del __main__ de Phy-AC)
# ==========================================================================
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    r = evaluate(THETA_REF, fidelity=Fidelity.L0)
    print("Caso ancla REF_CB10 (clase KJ66, anular inversa, jet-A):")
    for k in ["arch", "injector", "fuel", "source", "thermo_source",
              "sizing_driver", "A_ref_m2", "D_casing_mm", "D_L_mm",
              "L_tot_mm", "V_c_m3", "tau_ms", "U_ref_ms", "I_MW_m3_atm",
              "f_fuel_air", "phi_pz", "phi_global", "T_ad_st_K", "T_pz_K",
              "eta_comb", "eta_driver", "dp_over_p", "PF", "T_w1_K",
              "T_w2_K", "eta_film", "blowing_ratio_M", "SMD_um", "phi_LBO",
              "LBO_margin", "E_min_J", "EI_NOx", "EI_CO", "M_exit",
              "feasible"]:
        print(f"  {k:22s} = {r[k]}")
    print(f"  {'g':22s} = [" + ", ".join(f"{v:+.3f}" for v in r["g"]) + "]")
    print("  filas de orificios:")
    for row in r["hole_rows"]:
        print(f"    {row['zone']:>3s}  n={row['n']:3d}  "
              f"D={row['d_mm']:6.2f} mm  J={row['J']:7.2f}  "
              f"Y/D_L={row['Y_over_DL']:.3f}")
    print("  perfil de pared (z_mm, T_w1_K, T_w2_K): "
          f"{[[round(v, 1) for v in p] for p in r['T_wall_profile']]}")
    print("  SMD por correlacion [um]: "
          + str({k: (None if v is None else round(v, 2))
                 for k, v in r["SMD_um_by_corr"].items()}))
    print(f"  V_c optimo NOx-CO = {r['V_c_opt_m3'] * 1e3:.3f} L "
          f"(V_c = {r['V_c_m3'] * 1e3:.3f} L)")
    print(f"  coeficientes provisionales en juego: {len(r['limitations'])}")
    print(f"  features del surrogate: {physics_features(THETA_REF).shape} "
          f"(N_FEATURES = {N_FEATURES})")
    print("  record serializable a JSON: "
          f"{len(json.dumps(r, ensure_ascii=False))} caracteres")

    # ======================================================================
    # Ensayos de COHERENCIA de la auditoria dimensional (2026-07-30)
    # ======================================================================
    def _print_fan(title, fan, flags, stats, band=None):
        print(f"\n{title}")
        for k in NOX_FAN_NAMES:
            mark = "" if flags[k] == "ok" else f"   <-- {flags[k]}"
            print(f"    EI_NOx_{k:<12s} = {fan[k]:10.4g} g/kg{mark}")
        print(f"    abanico en rango  : min={stats['min']:.4g} "
              f"max={stats['max']:.4g} mediana={stats['median']:.4g} "
              f"dispersion=x{stats['spread_ratio']:.2f}  "
              f"({stats['n_used']} usadas / {stats['n_excluded']} excluidas)")
        if band is not None:
            ins = [k for k in NOX_FAN_NAMES if band[0] <= fan[k] <= band[1]]
            print(f"    banda de literatura {band[0]:.0f}-{band[1]:.0f} g/kg: "
                  f"dentro {ins}")

    print("\n" + "=" * 70)
    print("TEST 1 - abanico de EI_NOx en los DOS extremos (Ciencia 8, §15.0)")
    print("=" * 70)
    fan_a = {k: r[f"EI_NOx_{k}"] for k in NOX_FAN_NAMES}
    _print_fan("  [ancla REF_CB10 - microturbina 250 kPa / 430 K]",
               fan_a, r["EI_NOx_flags"], r["EI_NOx_fan"])
    _big = dict(NOX_REF_LARGE_ENGINE)
    _band = _big.pop("band_g_kg")
    # T3 solo sirve para cerrar ρ₃ y de ahí V_c = τ·ṁ/ρ₃; `nox_fan` recibe
    # escalares ya cerrados, así que se consume aquí y no se le pasa.
    _T3_big = _big.pop("T3_K")
    _big["V_c_m3"] = (_big["tau_s"] * _big["mdot_air_kgs"]
                      / (_big["P3_Pa"] / (RGAS * _T3_big)))
    fan_b = nox_fan(**_big)
    flg_b, st_b = _screen_nox_fan(fan_b)
    _print_fan(f"  [motor grande - {_big['P3_Pa'] / P_ATM:.0f} atm / "
               f"{_T3_big:.0f} K / tau {1e3 * _big['tau_s']:.0f} ms, "
               f"V_c = {1e3 * _big['V_c_m3']:.2f} L]",
               fan_b, flg_b, st_b, band=_band)
    print(f"    VEREDICTO: la dispersion del abanico pasa de x2.6e5 (antes de "
          f"la auditoria) a x{r['EI_NOx_fan']['spread_ratio']:.1f} en el ancla "
          f"y x{st_b['spread_ratio']:.1f} en el motor grande.")
    print("    Cabecera del record = EI_NOx de Lefebvre (unica verificada): "
          f"{r['EI_NOx']:.4g} g/kg [contrato §4].")

    print("\n" + "=" * 70)
    print("TEST 2 - efusion: relacion de soplado y cierre de masa (Ciencia 5.1)")
    print("=" * 70)
    print(f"  M (media del liner)   = {r['blowing_ratio_M']:.4f}   perfil "
          f"{[round(v, 3) for v in r['blowing_ratio_M_profile']]}")
    print(f"  banda de diseno       = {r['M_design_band']}   "
          f"rango del mapa = {r['M_map_range']}   "
          f"mapa clampeado = {r['eta_film_saturated']}")
    print(f"  U_cool (masa)         = {r['U_cool_ms']:.3f} m/s   "
          f"U_cool (dP_liner) = {r['U_cool_dp_ms']:.3f} m/s   "
          f"A_req/A_geom = {r['effusion_area_match']:.4f}")
    print(f"  eta_film              = {r['eta_film']:.4f}  (antes 0.712357 = "
          f"ultimo punto de la tabla: saturado, no modelado)")
    print(f"  g[8] (d_eff + banda M) = {r['g'][8]:+.4f}   "
          f"g_soft['M_design'] = {r['g_soft']['M_design']:+.4f}")
    print("  respuesta al paso de efusion (pitch/d_eff):")
    for _p in (4.0, 5.0, 6.0, 8.0, 10.0):
        _t = THETA_REF.copy()
        _t[9] = _p
        _rp = evaluate(_t, use_cache=False)
        print(f"    pitch/d={_p:5.1f}  porosidad={100 * _rp['effusion_porosity']:5.2f} %"
              f"  M={_rp['blowing_ratio_M']:6.3f}  eta_film={_rp['eta_film']:.4f}"
              f"  g[8]={_rp['g'][8]:+.3f}  factible={_rp['feasible']}")

    print("\n" + "=" * 70)
    print("TEST 3 - monotonia del factor de patron (Ciencia 4.3)")
    print("=" * 70)
    print("  PF vs L_DZ/D_L (debe BAJAR al alargar la dilucion):")
    _pf_prev, _mono1 = None, True
    for _l in (1.0, 1.4, 1.8, 2.2, 3.0):
        _t = THETA_REF.copy()
        _t[13] = _l
        _rp = evaluate(_t, use_cache=False)
        _mono1 &= (_pf_prev is None or _rp["PF"] < _pf_prev)
        _pf_prev = _rp["PF"]
        print(f"    L_DZ/D_L={_l:4.1f}  dP_L/q_ref="
              f"{_rp['dp_liner_over_qref']:6.3f}  PF={_rp['PF']:.5f}")
    print("  PF vs dP_L/q_ref a geometria fija (correlacion desnuda):")
    _sz_fake = dict(L_dz_mm=98.13, D_L_mm=54.52)
    _pf_prev, _mono2 = None, True
    for _q in (5.0, 10.0, 15.0, 25.0, 40.0):
        _pfv = _pattern_factor(_sz_fake, dict(dp_liner_over_qref=_q),
                               THETA_REF)["PF"]
        _mono2 &= (_pf_prev is None or _pfv < _pf_prev)
        _pf_prev = _pfv
        print(f"    dP_L/q_ref={_q:5.1f}  PF={_pfv:.5f}")
    print(f"  monotonia en L_DZ: {'OK' if _mono1 else 'FALLA'}   "
          f"monotonia en dP_L/q_ref: {'OK' if _mono2 else 'FALLA'}")
    print(f"  NOTA: con el criterio aerodinamico dimensionando A_ref, "
          f"dP_L/q_ref = LINER_FRAC_OF_DP·PLF = "
          f"{LINER_FRAC_OF_DP * coeffs()['PLF']:.1f} SIEMPRE (identidad "
          f"algebraica, ver _pattern_factor): el ancla cae en el punto de "
          f"calibracion de k_dilution, de ahi el 0.35 clavado.")

    print("\n" + "=" * 70)
    print("TEST 4 - definiciones explicitas y bandas declaradas")
    print("=" * 70)
    print(f"  {r['LBO_margin_def']}")
    print(f"    phi_pz={r['phi_pz']:.4f}  phi_LBO={r['phi_LBO']:.4f}  ->  "
          f"LBO_margin={r['LBO_margin']:.4f}  "
          f"LBO_margin_ratio={r['LBO_margin_ratio']:.4f}  "
          f"LBO_margin_frac={r['LBO_margin_frac']:.4f}")
    _I_id = (r["f_fuel_air"] * r["LHV_J_kg"] * P_ATM
             / (RGAS * r["T3_K"] * 1e-3 * r["tau_ms"])) / 1e6
    print(f"  I = mdot_f·LHV/(V_c·P3[atm]) = {r['I_MW_m3_atm']:.2f} "
          f"MW/m3·atm   (identidad FAR·LHV·P_atm/(R·T3·tau) = {_I_id:.2f}, "
          f"error {abs(_I_id / r['I_MW_m3_atm'] - 1) * 100:.3f} %)")
    print(f"    banda de clase {r['I_band_MW_m3_atm']} -> "
          f"g_soft['I_band'] = {r['g_soft']['I_band']:+.4f}  "
          f"(unidades correctas: P3 en atm; el valor alto es V_c pequeno)")
    print(f"  g_soft = {({k: round(v, 4) for k, v in r['g_soft'].items()})} "
          f"(BLANDAS: no entran en g ni en feasible)")

    print("\n" + "=" * 70)
    print("TEST 5 - determinismo, degradacion y pantalla fuera de rango")
    print("=" * 70)
    _r2 = evaluate(THETA_REF, use_cache=False)
    _same = all(_r2[k] == r[k] for k in ("EI_NOx", "PF", "T_w1_K",
                                         "blowing_ratio_M"))
    print(f"  dos evaluate identicos -> record identico: "
          f"{'OK' if _same else 'FALLA'}")
    _deg = evaluate(np.full(NDIM, np.nan))
    print(f"  theta degenerado -> g = {_deg['g'][0]:.1f} x {len(_deg['g'])}, "
          f"feasible={_deg['feasible']}, source='{_deg['source']}', "
          f"flags={sorted(set(_deg['EI_NOx_flags'].values()))}")
    _t = THETA_REF.copy()
    _t[IDX_P3_PA], _t[IDX_T4_K], _t[IDX_T3_K] = 2.0e6, 1800.0, 900.0
    _t[2] = 0.55
    _rx = evaluate(_t, use_cache=False)
    print(f"  caso extremo (P3=20 bar, T3=900 K, T4=1800 K): "
          f"abanico {[round(_rx[f'EI_NOx_{k}'], 3) for k in NOX_FAN_NAMES]}")
    print(f"    flags={_rx['EI_NOx_flags']}")
    print(f"    en rango: {_rx['EI_NOx_fan']['used']}  "
          f"excluidas: {_rx['EI_NOx_fan']['excluded']}  "
          f"dispersion=x{_rx['EI_NOx_fan']['spread_ratio']:.2f}")
    print(f"\n  avisos provisionales acumulados: {len(provisional_report())}")
