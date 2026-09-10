"""
QUASAR Phy-CB · geometry_generator.py
=====================================
Capa 5a del CEM: del record VERIFICADO de `physics_core.evaluate` al contrato
de geometría `combustor.json` (schema **`phycb-combustor-1`**, definido
literalmente por `build_contract()` de este módulo y descrito en
`docs/Quasar_PhyCB_Science.md` §16) + las tablas CSV de
análisis + las condiciones de contorno de CFD.

Es la frontera entre Python y el resto del mundo: la capa 5c (C#/PicoGK,
`CombustorDesigner/`) y los solvers de `cfd/` NO recalculan nada — el
contrato lleva la geometría **totalmente resuelta**, en mm y grados, con el
convenio de signos documentado en el propio JSON (clave `_conventions`,
trampa CONTEXTO §10.2).

API pública (IMPLEMENTATION_CONTRACT §8):

    SCHEMA = "phycb-combustor-1"
    build_contract(theta, record, structural=None, dfam=None) -> dict
    generate(theta, record, outdir, structural=None, dfam=None, ...) -> dict
    cfd_boundary_conditions(theta, record) -> dict
    export_step(contract, outdir, mode="parts") -> list[str]  # CadQuery OPC.
    try_export_step(contract, path) -> bool                   # compatibilidad
    apply_dfam(contract, dfam) -> dict     # aplica el veredicto del checker
    var_names() -> list[str]
    main(argv=None) -> int

---------------------------------------------------------------------------
PATRÓN «DOBLE VÍA» (CONTEXTO §10.4)
---------------------------------------------------------------------------
El STL de la capa 5c es una **aproximación imprimible**; la **referencia
dimensional de análisis** son los CSV que escribe `generate()`:
`hole_table.csv` (posición, Ø, C_d, forma de cada fila), `liner_profile.csv`
(polilínea + áreas de paso) y `effusion_field.csv`. Cuando el STL y el CSV
discrepen, **manda el CSV**.

---------------------------------------------------------------------------
GEOMETRÍA REAL DE LAS TRES ARQUITECTURAS (WAVE3 §1.1, gate `arch` de θ[0])
---------------------------------------------------------------------------
`Z` es SIEMPRE el eje de revolución y la polilínea del liner va ordenada por
`z` creciente. Lo que cambia entre arquitecturas es **el recorrido del gas**,
que se declara aparte en el bloque `flow_path`:

* **`can`** — tubo de radio `D_L/2` con domo (hemisférico o plano según el
  inyector) en `z` mínimo y contracción de salida a la tobera en `z` máximo.
  `r_inner = 0` en toda la polilínea. El gas recorre +Z.
* **`annular`** — corona entre `r_li` y `r_lo` con
  `π(r_lo² − r_li²) = A_L`; el radio exterior es
  `physics_core.LINER_R_OVER_RCAS·r_carcasa` = 0.92·r_carcasa (**el dueño de
  ese cociente es `physics_core`**: fija a la vez la columna `r_mm` de las
  filas de orificios del record —que así cae EXACTAMENTE sobre la pared— y
  el ÁREA MOJADA con la que la física reparte el aire de efusión). El gas
  recorre +Z. **OJO**: `π·D_L` NO es el perímetro de esta corona —es el de
  un círculo de área equivalente, un factor `r_lo/√(r_lo²−r_li²)` más
  pequeño—; ese error valía ×1.375 en el ancla y lo caza hoy el test T49.
* **`reverse_annular`** — anular de **flujo inverso** (la arquitectura del
  caso ancla): el compresor centrífugo descarga en el frente (z = 0) al
  anillo exterior, el aire lo recorre **+Z** precalentándose contra el liner,
  **gira 180°** alrededor del domo trasero y quema recorriendo el liner
  **−Z**; en el frente **gira otros 180°** hacia la turbina, que va alojada
  dentro del propio combustor («el combustor envuelve la turbina»,
  EstadoDelArte §2.1). En la polilínea eso se ve como: corona de salida a
  radio pequeño en z=0 → transición radial a 55° → cámara anular a radio
  grande → domo toroidal en z máximo. El **eje geométrico sigue siendo Z**
  (WAVE3 §1.1); el signo del recorrido vive en `flow_path[].dir_z`.

Reglas de diseño que se imponen por construcción y se verifican (T31):

  1. la polilínea meridional **no se auto-interseca** (`_self_intersects`);
  2. el área de paso `A(z) = π(r_out² − r_in²)` **nunca baja de `A_L`**
     (la sección de referencia del liner del contrato) — en la transición de
     salida el radio exterior se resuelve como `√(r_in² + A/π)`, así que el
     área es constante EXACTA en todo el tramo;
  3. las **cortinas de giro** de 180° tienen área ≥ `A_ref` (regla de que el
     giro no sea la garganta): de ahí sale la longitud que la carcasa
     sobresale del domo, `L_turn = 1.1·A_ref/(2π·r_lo)`.

---------------------------------------------------------------------------
CONVENCIONES (IMPLEMENTATION_CONTRACT §10, CONTEXTO §9)
---------------------------------------------------------------------------
SI interno; **mm/deg solo en la interfaz y con sufijo en la clave**. Ángulos:
`hole_rows[].angle_deg` = inclinación del EJE del orificio desde la dirección
RADIAL hacia +Z (0 = radial puro, 90 = axial puro) — **idéntico a
`dfam_checker._row_axis` y a `Parameters.cs`**; `effusion.angle_deg` se mide
desde la SUPERFICIE del liner. Solo NumPy es obligatorio: CadQuery degrada a
`[]` sin excepción. Determinismo: sin relojes ni aleatoriedad dentro del
contrato (`SEED = 71` se declara por coherencia), de modo que dos
`build_contract` con el mismo record son bit-idénticos y el round-trip
write→read es exacto (T25).
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

# ==========================================================================
# 0. Identidad, constantes y tunables (SCREAMING_CASE + procedencia)
# ==========================================================================
SCHEMA = "phycb-combustor-1"       # contrato §9 — versionado desde el día 0
SEED = 71                          # semilla de la familia (contrato §10.8).
#                                    Este módulo es puramente determinista.
GENERATOR = "phycb.geometry_generator"

ROUND_N = 6                        # decimales de las magnitudes en mm/deg del
#                                    JSON. Suficiente para que el round-trip
#                                    sea exacto y el archivo legible.

# --- radios y proporciones de la envolvente -------------------------------
LINER_R_OVER_RCAS = 0.92   # r_liner_exterior / r_carcasa en las anulares.
#                            ⚠️ EL DUEÑO ES `physics_core.LINER_R_OVER_RCAS`
#                            desde el 2026-07-31: además de fijar la columna
#                            `r_mm` de las filas de orificios, fija el ÁREA
#                            MOJADA de la pared caliente que lleva efusión,
#                            así que la física necesita el MISMO número para
#                            predecir la geometría. Este literal es solo el
#                            RESPALDO etiquetado de `_shared_const` para
#                            cuando el hermano no sea importable; el valor
#                            vivo se lee de él.
ANNULUS_HTR = 0.60         # D_hub/D_casing de las anulares = el mismo cierre
#                            de `physics_core.ANNULUS_HTR` (solo informativo
#                            aquí: fija el radio del cubo que reportamos).
EXIT_HUB_OVER_RLI = 0.35   # radio interior del conducto de salida / r_li en
#                            flujo inverso (túnel del eje y del cubo de la
#                            turbina). ⚠️ CIERRE DE LA CASA: la turbina no
#                            es de Phy-CB; se elige el mayor cubo que deja
#                            pasar A4 sin superar r_li. RECALIBRAR cuando
#                            exista Phy-TB.
EXIT_CONE_DEG = 55.0       # inclinación de las transiciones radiales respecto
#                            a la placa. > 45° por diseño ⇒ auto-soportadas
#                            (EstadoDelArte §4.3): el ángulo de voladizo NO se
#                            deja al azar, se impone en la geometría.
EXIT_LIP_MM = 4.0          # tramo recto en la brida de salida (asiento).
CASING_GAP_MIN_MM = 2.0    # holgura radial mínima carcasa↔liner en la salida.
N_TRANS_PTS = 5            # puntos de la transición radial (área constante).
AREA_REL_TOL = 1e-6        # tolerancia RELATIVA con la que se compara el área
AREA_ABS_TOL_MM2 = 1e-6    # de paso contra la de referencia (T31): la
#                            polilínea se redondea a ROUND_N decimales antes
#                            de salir, y 2πr·5e-7 ya mueve el área ~1e-4 mm².
DOME_MAX_DR_MM = 1.5       # paso radial máximo al muestrear un domo si se
#                            incluye en la polilínea (hoy el domo se declara,
#                            no se discretiza: lo cierra la capa 5c).
ANNULUS_R_MIN_FRAC = 0.15  # r_li mínimo / r_carcasa (evita coronas
#                            degeneradas con A_L enorme). MISMO régimen que
#                            `LINER_R_OVER_RCAS`: el dueño es
#                            `physics_core.ANNULUS_R_MIN_FRAC` (la física
#                            reproduce esta rama de respaldo al predecir el
#                            área mojada) y esto es el respaldo etiquetado.
TURN_AREA_SF = 1.10        # factor de seguridad del área de cortina de un
#                            giro de 180° respecto a A_ref. ⚠️ CRITERIO
#                            PROPIO (el giro no debe ser la garganta).

# --- espesores y features (todos ≥ suelo LPBF de `dfam_checker`) ----------
VOXEL_DEFAULT_MM = 0.5     # voxel de iteración de la capa 5c (CONTEXTO §8).
FEATURE_MIN_MM = 0.6       # suelo absoluto de un feature en LPBF.
CASING_T_OVER_T_LINER = 1.5   # espesor de carcasa / espesor de liner cuando
#                            `structures_core` no lo dice (mismo cierre que
#                            `structures_core.CASING_T_OVER_T_LINER`).
CASING_T_MIN_MM = 1.0      # suelo del espesor de carcasa (recipiente a
#                            presión). ⚠️ criterio propio.
LINER_FILLET_R_OVER_T = 1.5   # radio de acuerdo domo↔barril / espesor de
#                            liner. ⚠️ CRITERIO PROPIO (concentración de
#                            tensiones en la unión); es el `fillet_r_mm` que
#                            la capa 5c materializa con el cierre morfológico
#                            restringido de `Liner.voxWithDomeFillet`.
FLANGE_W_MM = 10.0         # ancho radial de la brida.
FLANGE_T_MM = 4.0          # espesor axial de la brida.
HEAD_PLATE_T_MM = 2.0      # tapa de la carcasa (cierra el extremo).
BOLT_SPACING_MM = 25.0     # paso de pernos sobre la circunferencia de brida
#                            (práctica de taller). Da 12 pernos en el ancla,
#                            que es el número del patrón de Phy-AC.
BOLT_D_MM = 6.0            # Ø de perno (M6) — patrón de Phy-AC.
BOLT_COUNT_MIN, BOLT_COUNT_MAX = 6, 24

# --- swirler ---------------------------------------------------------------
SWIRLER_N_VANES = 8        # nº de álabes por swirler. ✅ banda de práctica
#                            8–12; se reduce si el paso entre álabes no cabe.
SWIRLER_VANE_T_MM = 1.2    # espesor de álabe. ⚠️ criterio propio: por encima
#                            del suelo de feature max(0.6, 2·voxel) = 1.0 mm
#                            a voxel 0.5 con margen del 20 %.
SWIRLER_HUB_MIN_FRAC = 0.35   # D_hub/D_tip mínimo del swirler. El hub que
#                            devuelve `physics_core._swirler_hub_ratio` puede
#                            ser 0 (S_N bajo con β=50°) y un swirler sin cubo
#                            no es fabricable ni sujeta al inyector: se
#                            impone este mínimo CONSERVANDO A_sw.
SWIRLER_VANE_GAP_MIN_MM = 1.0   # paso libre mínimo entre álabes.
SWIRLER_N_VANES_MIN = 4    # suelo del nº de álabes cuando el paso libre en
#                            el cubo no deja los 8 nominales.
SWIRLER_A_OVER_AREF = 0.14   # A_sw/A_ref de RESPALDO cuando el record no
#                            trae `swirler.A_sw_m2`. ⚠️ ESPEJO EXACTO de
#                            `physics_core.A_SW_OVER_AREF`: si se toca allí,
#                            hay que tocarlo aquí (el swirler dejaría de
#                            casar con la pérdida que cobra la física).
INJ_SPACING_FACTOR = 1.25  # (Ø punta del swirler)·factor = paso mínimo entre
#                            inyectores sobre el anillo del domo.
N_INJ_ANNULAR_DEFAULT = 8  # inyectores de una anular cuando la física no los
#                            fija. ⚠️ banda 6–12 en microturbinas (KJ66: 6).

# --- inyectores -----------------------------------------------------------
VAPORIZER_TUBE_D_MM = 6.0  # Ø exterior del tubo «walking stick». ✅ clase
#                            KJ66 (tubos de ~64 mm con codo ~30°,
#                            EstadoDelArte §2.3); el Ø no está publicado —
#                            ⚠️ criterio propio dentro de la práctica.
VAPORIZER_L_MAX_FRAC = 0.60   # longitud de tubo / L_tot máxima (que no
#                            atraviese la cámara).
INJECTOR_BORE_CD = 0.70    # C_d del orificio simplex para deducir su Ø.

# --- orificios ------------------------------------------------------------
IGNITER_D_MM = 5.0         # Ø del alojamiento de bujía/incandescente en el
#                            liner. ⚠️ criterio propio (bujía micro M8–M10).
#                            No lleva caudal (`flow: false`): el tapón la
#                            sella, así que NO contamina el reparto de aire.
IGNITER_S_OVER_LPZ = 0.35  # posición de la bujía dentro de la PZ.
IGNITER_PHI0_DEG = 22.5    # fase para no chocar con la primera fila de PZ.
HOLE_R_TOL_MM = 0.5        # tolerancia con la que se acepta el `r_mm` del
#                            record antes de reproyectarlo sobre la pared.

# --- efusión ---------------------------------------------------------------
EFFUSION_ANGLE_DEG = 25.0  # ángulo del orificio de efusión respecto a la
#                            SUPERFICIE. ✅ centro de la banda 20–30° de
#                            Ciencia §5.1 (y la banda que exige
#                            `dfam_checker.EFF_ANGLE_DEG_BAND`).
EFFUSION_SHAPE = "conical"  # cónico = mejor eficacia de película (difusor de
#                            salida); la capa 5c lo resuelve como campo
#                            implícito, nunca como booleana por orificio.
EFFUSION_CONE_EXPANSION = 1.6   # r_salida/r_entrada del orificio cónico. ⚠️

# --- drenaje de polvo (EstadoDelArte §4.4) --------------------------------
DRAIN_D_OVER_W = 2.0       # Ø de puerto ≥ 2× el ancho del canal ✅.
DRAIN_D_MIN_MM, DRAIN_D_CAP_MM = 2.0, 8.0   # mismos topes que dfam_checker.
DRAIN_INSET_FRAC = 0.05    # retranqueo del puerto respecto al extremo.
DRAIN_INSET_MAX_MM = 5.0

# --- CFD -------------------------------------------------------------------
TURBULENCE_INTENSITY = 0.05   # 5 % a la entrada (mismo valor que el bloque
#                            `cfx_boundary_conditions` de Phy-AC).
LENGTH_SCALE_OVER_DH = 0.07   # ℓ = 0.07·D_h (regla estándar de conducto).

#: Nombres de θ. Respaldo CONGELADO por si `physics_core` no es importable
#: (contrato §1): el orden es ABI, no se reordena nunca.
_VAR_NAMES_FALLBACK = [
    "arch", "AL_over_Aref", "frac_pz", "frac_sz", "frac_dz", "swirl_SN",
    "n_holes_pz", "n_holes_dz", "d_eff_mm", "pitch_over_d_eff", "injector",
    "ALR", "L_pz_over_DL", "L_dz_over_DL", "t_liner_mm", "fuel", "P3_Pa",
    "T3_K", "mdot_kgs", "T4_K",
]
_ARCH_NAMES = ("can", "annular", "reverse_annular")
_INJECTOR_NAMES = ("airblast", "simplex", "vaporizer", "micromix")
_FUEL_NAMES = ("jet_a", "propane", "h2")

#: Convenios que viajan DENTRO del JSON (trampa CONTEXTO §10.2: el β₂ de
#: Phy-CC es negativo en Python y el C# le hace `Math.Abs()` porque el
#: convenio no estaba escrito en ningún sitio). Aquí sí lo está.
CONVENTIONS = {
    "axis": "Z is the axis of revolution. Every polyline is ordered by "
            "increasing z_mm.",
    "units": "Lengths in mm, angles in deg, pressures in Pa, temperatures "
             "in K, mass flows in kg/s — ALWAYS with the suffix in the key. "
             "Internally Phy-CB works in SI.",
    "flow": "The nominal gas goes to +Z in 'can' and 'annular'. In "
            "'reverse_annular' the gas path makes two 180° turns and inside "
            "the liner it goes to −Z: the geometric axis is still Z and the "
            "sign of each leg is in flow_path[].dir_z.",
    "liner_profile": "Rows [z_mm, r_inner_mm, r_outer_mm] of the GAS "
                     "CONTOUR (inner wetted surface). r_inner = 0 <=> can. "
                     "The wall grows outwards at r_outer and inwards at "
                     "r_inner, with thickness liner.t_wall_mm.",
    "casing_profile": "Rows [z_mm, r_mm] with r_mm = INNER radius of the "
                      "casing; the wall grows outwards.",
    "hole_angle_deg": "hole_rows[].angle_deg = tilt of the hole AXIS from "
                      "the RADIAL direction towards +Z: 0 = purely radial "
                      "(jet perpendicular to the axis), 90 = purely axial. "
                      "Positive = the jet points to +Z. An explicit 'axis' "
                      "vector [radial, tangential, axial] overrides "
                      "angle_deg.",
    "hole_phi0_deg": "phi0_deg = phase of the row's first hole, measured "
                     "from +X counter-clockwise seen from +Z; the n holes "
                     "are spread every 360/n degrees.",
    "effusion_angle_deg": "effusion.angle_deg is measured from the liner "
                          "SURFACE (not from the normal): 20–30° on a nearly "
                          "axial wall leaves the hole self-supporting.",
    "effusion_walls": "In v1 ONLY the OUTER liner wall carries effusion "
                      "(effusion.walls = ['outer']): in the annular ones the "
                      "inner face wraps the shaft/turbine duct, which is not "
                      "hardware of this contract. effusion.wall_area_mm2 is "
                      "that surface, integrated over the REAL polyline, and "
                      "validation.effusion cross-checks it against the "
                      "nominal one of physics_core.effusion_wall_area_m2: if "
                      "they do not match it is a VIOLATION of the "
                      "cross-layer contract, not a note.",
    "swirler_beta_deg": "swirler.beta_deg is the vane angle with respect "
                        "to the AXIS (0 = axial, 90 = tangential), always "
                        "POSITIVE; the swirl direction is counter-clockwise "
                        "seen from +Z.",
    "signs": "Constraints g ≤ 0 <=> feasible (the magnitude is the degree "
             "of violation). No angle of the contract is negative.",
    "truth": "The CSV overrules the STL: hole_table.csv, "
             "liner_profile.csv and effusion_field.csv are the dimensional "
             "reference; the layer-5c STL is the printable approximation "
             "(CONTEXTO §10.4).",
}


# ==========================================================================
# 1. Imports perezosos VERIFICADOS y lectura defensiva
# ==========================================================================
def _import_sibling(module_name: str):
    """Importa un módulo de Phy-CB comprobando que sea REALMENTE de Phy-CB.

    ⚠️ TRAMPA REAL de esta máquina: Phy-AC está instalado como paquete
    editable, así que `import physics_core` desde otro directorio puede
    resolver silenciosamente al de Phy-AC (θ de 13–15 posiciones y otra
    física). Se compara el directorio del módulo con el de ESTE archivo.
    Devuelve el módulo o `None` — nunca lanza (CONTEXTO §2.3).
    """
    try:
        import importlib
        mod = importlib.import_module(module_name)
        here = os.path.dirname(os.path.abspath(__file__))
        there = os.path.dirname(os.path.abspath(getattr(mod, "__file__", "")))
        return mod if os.path.normcase(here) == os.path.normcase(there) \
            else None
    except Exception:
        return None


_PC_MOD, _PC_TRIED = None, False
_DFAM_MOD, _DFAM_TRIED = None, False


def _pc():
    """`physics_core` verificado (o `None`). Cacheado."""
    global _PC_MOD, _PC_TRIED
    if not _PC_TRIED:
        _PC_TRIED = True
        _PC_MOD = _import_sibling("physics_core")
    return _PC_MOD


def _shared_const(name: str, fallback: float) -> float:
    """Constante COMPARTIDA con la física, cuyo dueño es `physics_core`.

    Principio de la casa (CONTEXTO §8, el ejemplo del radio de fillet): una
    magnitud que comparten física y geometría se calcula UNA vez, en UN
    sitio. Si el hermano no es importable se cae al literal de respaldo de
    este módulo —degradación etiquetada, mismo valor documentado—; en cuanto
    esté, manda él. Nunca lanza.
    """
    pc = _pc()
    try:
        return float(getattr(pc, name))
    except Exception:
        return float(fallback)


def _dfam():
    """`dfam_checker` verificado (o `None`). Cacheado."""
    global _DFAM_MOD, _DFAM_TRIED
    if not _DFAM_TRIED:
        _DFAM_TRIED = True
        _DFAM_MOD = _import_sibling("dfam_checker")
    return _DFAM_MOD


def var_names() -> list[str]:
    """Nombres de θ: los de `physics_core` si está, si no el respaldo."""
    pc = _pc()
    names = getattr(pc, "VAR_NAMES", None) if pc is not None else None
    if isinstance(names, (list, tuple)) and len(names) >= 20:
        return [str(n) for n in names]
    return list(_VAR_NAMES_FALLBACK)


def _num(x):
    """float finito o `None`. Nunca lanza (`bool` se rechaza a propósito)."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pos(x):
    """float finito y estrictamente positivo, o `None`."""
    v = _num(x)
    return v if (v is not None and v > 0.0) else None


def _rget(record, key, default=None):
    """`record[key]` tolerante: no-dict / clave ausente / `None` ⇒ default."""
    if not isinstance(record, dict):
        return default
    v = record.get(key, None)
    return default if v is None else v


def _rnum(record, key, default=None):
    """Número finito del record, o `default` (lectura DEFENSIVA total)."""
    v = _num(_rget(record, key))
    return default if v is None else v


def _rpos(record, key, default=None):
    v = _pos(_rget(record, key))
    return default if v is None else v


def _r(x, n: int = ROUND_N):
    """Redondeo estable para la interfaz. No finito ⇒ se deja pasar tal cual
    (el JSON lo sanea al escribir y `dfam_checker` lo denuncia)."""
    v = _num(x)
    return float(x) if v is None else float(round(v, n))


def _theta_list(theta) -> list[float]:
    """θ como lista de floats de longitud ≥ 20 (rellena con 0.0). Nunca
    lanza: un θ corto o con basura sigue produciendo un `design_vector`."""
    try:
        t = np.asarray(theta, dtype=float).ravel().tolist()
    except Exception:
        t = []
    out = [(_num(v) if _num(v) is not None else float("nan")) for v in t]
    while len(out) < len(_VAR_NAMES_FALLBACK):
        out.append(0.0)
    return out


def _gates(theta, record) -> dict:
    """Gates decodificadas. Manda el record (ya son strings verificados);
    si falta, se decodifican de θ con `int(round(...))` + clip, que es el
    invariante CONTEXTO §2.1."""
    t = _theta_list(theta)

    def _gate(idx, names, key):
        s = _rget(record, key)
        if isinstance(s, str) and s in names:
            return s
        v = _num(t[idx]) if idx < len(t) else None
        i = 0 if v is None else int(min(max(int(round(v)), 0), len(names) - 1))
        return names[i]

    return dict(arch=_gate(0, _ARCH_NAMES, "arch"),
                injector=_gate(10, _INJECTOR_NAMES, "injector"),
                fuel=_gate(15, _FUEL_NAMES, "fuel"))


# ==========================================================================
# 2. Geometría de la polilínea: áreas, auto-intersección, interpolación
# ==========================================================================
def _seg_cross(p, q, r, s) -> bool:
    """¿Se cruzan PROPIAMENTE los segmentos pq y rs? (sin contar extremos
    compartidos, que en una polilínea son adyacencias legítimas)."""
    def d(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2, d3, d4 = d(r, s, p), d(r, s, q), d(p, q, r), d(p, q, s)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False


def _self_intersects(points: list) -> bool:
    """¿La poligonal CERRADA `points` se auto-interseca? (test T31).

    Se prueban todos los pares de segmentos no adyacentes — O(n²) con n ≤ 60
    es despreciable. Devuelve `True` también si la entrada es ilegible: la
    respuesta conservadora es «no la des por buena»."""
    try:
        p = [(float(a), float(b)) for a, b in points]
    except Exception:
        return True
    n = len(p)
    if n < 4:
        return False
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            c, d2 = p[j], p[(j + 1) % n]
            if _seg_cross(a, b, c, d2):
                return True
    return False


def _meridional_polygon(profile: list) -> list:
    """Contorno meridional CERRADO del volumen de gas: pared exterior de
    z_min→z_max y pared interior de vuelta."""
    outer = [(row[0], row[2]) for row in profile]
    inner = [(row[0], row[1]) for row in reversed(profile)]
    return outer + inner


def _passage_area_mm2(row) -> float:
    """Área de paso anular de una estación: π(r_out² − r_in²) [mm²]."""
    ri, ro = float(row[1]), float(row[2])
    return math.pi * max(ro * ro - ri * ri, 0.0)


def _profile_ok(profile: list) -> tuple[bool, list]:
    """Validez de la polilínea del liner (los MISMOS criterios que el
    `Validate()` de `Parameters.cs`, para que un contrato escrito aquí nunca
    sea rechazado allí): ≥2 puntos, z no decreciente, 0 ≤ r_in < r_out,
    longitud axial no nula y sin auto-intersección."""
    errs: list[str] = []
    if not isinstance(profile, (list, tuple)) or len(profile) < 2:
        return False, ["la polilínea del liner necesita ≥2 puntos"]
    z_prev = None
    for i, row in enumerate(profile):
        if len(row) != 3 or any(_num(v) is None for v in row):
            errs.append(f"liner.profile[{i}] no es [z, r_in, r_out] finito")
            continue
        z, ri, ro = (float(v) for v in row)
        if ro <= 0.0:
            errs.append(f"liner.profile[{i}]: r_outer_mm debe ser > 0")
        if ri < 0.0 or ri >= ro:
            errs.append(f"liner.profile[{i}]: exige 0 <= r_inner < r_outer")
        if z_prev is not None and z < z_prev - 1e-9:
            errs.append(f"liner.profile[{i}]: z_mm retrocede")
        z_prev = z
    if not errs:
        if abs(float(profile[-1][0]) - float(profile[0][0])) < 1e-3:
            errs.append("liner.profile tiene longitud axial nula")
        if _self_intersects(_meridional_polygon(profile)):
            errs.append("liner.profile se AUTO-INTERSECA en el plano "
                        "meridional")
    return (not errs), errs


def _interp(profile: list, z: float, col: int) -> float:
    """Interpolación lineal en z de una columna de la polilínea."""
    try:
        arr = np.asarray(profile, dtype=float)
        idx = np.argsort(arr[:, 0], kind="stable")
        return float(np.interp(float(z), arr[idx, 0], arr[idx, col]))
    except Exception:
        return 0.0


# ==========================================================================
# 3. Saneo del record: dimensiones de la envolvente y espesores de pared
# ==========================================================================
def _clean(obj):
    """Estructura serializable a JSON ESTRICTO.

    Sin `NaN`/`inf` (que NO son JSON válido — misma decisión declarada que
    `physics_core` con `SMD_um = None`), sin tipos de NumPy y sin tuplas (una
    tupla vuelve del JSON como lista y rompería la igualdad del round-trip
    write→read del test T25). Nunca lanza."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _dims(record: dict, gates: dict, t: list[float]) -> dict:
    """Dimensiones de la envolvente en **mm y mm²**, saneadas y cerradas.

    Todo sale del record VERIFICADO; cada clave lleva un respaldo geométrico
    coherente para que un record incompleto siga produciendo un contrato
    construible (lectura defensiva, CONTEXTO §10.3). Se devuelven ya en
    mm/mm² porque el contrato **es** una interfaz (convención §10.1).

    `A_pass_mm2 = max(A_L, A4)` es el **área de paso de referencia** con la
    que se construyen todas las coronas: garantiza por construcción que
    ninguna sección del liner estrangula ni la sección de referencia ni la
    salida (regla 2 de la cabecera y test T31)."""
    D_L = _rpos(record, "D_L_mm", 50.0)
    a = _rpos(record, "A_L_m2")
    A_L = 1e6 * a if a is not None else 0.25 * math.pi * D_L * D_L
    a = _rpos(record, "A_ref_m2")
    A_ref = 1e6 * a if a is not None else \
        A_L / min(max(_pos(t[1]) or 0.70, 0.30), 0.95)
    a = _rpos(record, "A4_m2")
    A4 = 1e6 * a if a is not None else A_L
    a = _rpos(record, "A_an_m2")
    A_an = 1e6 * a if a is not None else max(A_ref - A_L, 0.05 * A_L)
    D_cas = _rpos(record, "D_casing_mm")
    if D_cas is None:
        D_cas = math.sqrt(4.0 * A_ref / math.pi) if gates["arch"] == "can" \
            else math.sqrt(4.0 * A_ref
                           / (math.pi * max(1.0 - ANNULUS_HTR ** 2, 1e-3)))
    L_pz = _rpos(record, "L_pz_mm", 0.75 * D_L)
    L_sz = _rpos(record, "L_sz_mm", 0.50 * D_L)
    L_dz = _rpos(record, "L_dz_mm", 1.80 * D_L)
    return dict(D_L_mm=D_L, D_casing_mm=D_cas,
                r_L_mm=max(0.5 * D_L, math.sqrt(A_L / math.pi)),
                r_casing_mm=0.5 * D_cas,
                r_hub_mm=0.5 * ANNULUS_HTR * D_cas,
                A_ref_mm2=A_ref, A_L_mm2=A_L, A4_mm2=A4, A_an_mm2=A_an,
                A_pass_mm2=max(A_L, A4),
                L_pz_mm=L_pz, L_sz_mm=L_sz, L_dz_mm=L_dz,
                L_tot_mm=L_pz + L_sz + L_dz,
                L_tot_record_mm=_rpos(record, "L_tot_mm", L_pz + L_sz + L_dz))


def _walls(record: dict, structural: dict | None, t: list[float]) -> dict:
    """Espesores y materiales de liner y carcasa.

    Fuente única de verdad: `structures_core.evaluate_structural` cuando se
    pasa (trae `t_casing_mm`, `material`, `material_casing` ya derateados).
    Sin él, el liner viene de θ[14]/record y la carcasa del cierre
    `CASING_T_OVER_T_LINER` con suelo `CASING_T_MIN_MM`."""
    t_liner = _rpos(record, "t_liner_mm") or _pos(t[14]) or 0.8
    mat_l = str(_rget(record, "material", "Inconel-625"))
    mat_c, t_cas = "Inconel-718", None
    if isinstance(structural, dict):
        t_liner = _pos(structural.get("t_liner_mm")) or t_liner
        t_cas = _pos(structural.get("t_casing_mm"))
        mat_l = str(structural.get("material") or mat_l)
        mat_c = str(structural.get("material_casing") or mat_c)
    if t_cas is None:
        t_cas = CASING_T_OVER_T_LINER * t_liner
    return dict(t_liner_mm=float(t_liner),
                t_casing_mm=float(max(t_cas, CASING_T_MIN_MM)),
                material_liner=mat_l, material_casing=mat_c,
                fillet_r_mm=float(min(max(LINER_FILLET_R_OVER_T * t_liner,
                                          FEATURE_MIN_MM), 3.0)))


# ==========================================================================
# 4. Geometría REAL de las tres arquitecturas
# ==========================================================================
def _transition(z0: float, r_in0: float, r_in1: float, A_mm2: float,
                n_pts: int = N_TRANS_PTS) -> tuple[list, float]:
    """Transición radial de **área EXACTAMENTE constante** a
    `EXIT_CONE_DEG` sobre la placa.

    La pared interior sube (o baja) de `r_in0` a `r_in1` con pendiente
    `dz = |Δr|·tan(EXIT_CONE_DEG)` — > 45° por construcción, así que la
    generatriz es auto-soportada (EstadoDelArte §4.3) — y la exterior se
    resuelve como `√(r_in² + A/π)`, de modo que `A(z) = A_mm2` en TODA la
    transición y no solo en sus extremos. Devuelve las filas NUEVAS (sin
    repetir el punto de partida) y la longitud axial consumida."""
    dr = float(r_in1) - float(r_in0)
    dz = abs(dr) * math.tan(math.radians(EXIT_CONE_DEG))
    rows: list[list[float]] = []
    n = max(int(n_pts), 1)
    for k in range(1, n + 1):
        f = k / n
        ri = float(r_in0) + f * dr
        rows.append([z0 + f * dz, ri,
                     math.sqrt(ri * ri + max(A_mm2, 1e-9) / math.pi)])
    return rows, dz


def _annulus_geometry(dm: dict, w: dict, notes: list) -> tuple:
    """Radios de la corona del liner de una arquitectura anular.

    `r_lo = LINER_R_OVER_RCAS·r_carcasa` (el cociente lo posee
    `physics_core`, para que la columna `r_mm` de las filas de orificios del
    record caiga sobre la pared **y** para que `effusion_wall_area_m2`
    prediga esta misma corona) y
    `r_li = √(r_lo² − A_pass/π)`. Si esa corona degeneraría en un radio
    interior menor que `ANNULUS_R_MIN_FRAC·r_carcasa`, se fija `r_li` en ese
    suelo, se recalcula `r_lo` CONSERVANDO el área y, si hace falta, se abre
    la carcasa — con nota, nunca en silencio."""
    r_cas = dm["r_casing_mm"]
    A = dm["A_pass_mm2"]
    #  FUENTE ÚNICA: los dos cocientes los posee `physics_core`, que necesita
    #  reproducir ESTA construcción para predecir el área mojada de la
    #  efusión (`effusion_wall_area_m2`).
    f_lo = _shared_const("LINER_R_OVER_RCAS", LINER_R_OVER_RCAS)
    f_min = _shared_const("ANNULUS_R_MIN_FRAC", ANNULUS_R_MIN_FRAC)
    r_lo = f_lo * r_cas
    r_min = f_min * r_cas
    x = r_lo * r_lo - A / math.pi
    if x <= r_min * r_min:
        r_li = r_min
        r_lo = math.sqrt(r_min * r_min + A / math.pi)
        notes.append(
            f"corona degenerada: A_paso = {A:.1f} mm² no cabe dentro de "
            f"r_lo = {f_lo:.2f}·r_carcasa; se fija "
            f"r_inner = {r_li:.2f} mm y se abre r_outer a {r_lo:.2f} mm "
            "conservando el área")
        need = r_lo + w["t_liner_mm"] + CASING_GAP_MIN_MM
        if need > r_cas:
            notes.append(f"carcasa abierta de {r_cas:.2f} a {need:.2f} mm "
                         "para dejar la holgura radial mínima al liner")
            r_cas = need
    else:
        r_li = math.sqrt(x)
    return r_li, r_lo, r_cas


def _note_annulus(notes: list, A_geo: float, A_rec: float,
                  inner_corona: bool) -> None:
    """Anota la diferencia entre el anillo de refrigeración GEOMÉTRICO y el
    `A_an` del record.

    Hay dos causas legítimas y ninguna es un error: (1) `physics_core` calcula
    `A_an = A_ref − A_L` con radios de línea media, sin descontar el espesor
    de la pared, y (2) en las anulares el `A_ref` de la física es la corona
    completa `r_cubo → r_carcasa`, así que su `A_an` incluye también la corona
    INTERIOR, que aloja el eje y la turbina y no es hardware de Phy-CB. Se
    declara en voz alta porque el coeficiente convectivo del lado frío del
    record se calculó con `A_an`."""
    if A_rec <= 0.0 or abs(A_geo - A_rec) <= 0.10 * A_rec:
        return
    extra = (" y la corona INTERIOR r_cubo→r_li, que aloja el eje y la "
             "turbina y NO es hardware de este contrato (Phy-TB)"
             if inner_corona else "")
    notes.append(
        f"anillo de refrigeración geométrico = {A_geo:.1f} mm² frente al "
        f"A_an = {A_rec:.1f} mm² del record: el record no descuenta el "
        f"espesor de pared{extra}. La velocidad del lado frío real es "
        f"~{A_rec / max(A_geo, 1e-9):.2f}× la que usó el mapa térmico 1D")


def _geom_can(dm: dict, w: dict, gates: dict) -> dict:
    """`can` — tubo de radio `D_L/2` con domo en z = 0 y contracción de
    salida a la tobera. `r_inner = 0` en toda la polilínea; el gas va a +Z."""
    notes: list[str] = []
    r_L, t = dm["r_L_mm"], w["t_liner_mm"]
    r_cas = max(dm["r_casing_mm"], r_L + t + CASING_GAP_MIN_MM)
    L_pz, L_sz, L_tot = dm["L_pz_mm"], dm["L_sz_mm"], dm["L_tot_mm"]
    prof = [[0.0, 0.0, r_L], [L_pz, 0.0, r_L], [L_pz + L_sz, 0.0, r_L],
            [L_tot, 0.0, r_L]]
    z = L_tot
    r_exit = math.sqrt(max(dm["A4_mm2"], 1e-9) / math.pi)
    if r_L - r_exit > 0.5:                       # contracción de salida
        dz = (r_L - r_exit) * math.tan(math.radians(EXIT_CONE_DEG))
        prof.append([z + dz, 0.0, r_exit])
        z += dz
    else:
        r_exit = r_L
        notes.append("A4 ≈ A_L: la salida del can no contrae (el difusor de "
                     "la tobera vive aguas abajo, fuera del contrato)")
    prof.append([z + EXIT_LIP_MM, 0.0, r_exit])
    z += EXIT_LIP_MM

    flat = gates["injector"] in ("vaporizer", "micromix")
    dome = "flat" if flat else "hemispherical"
    bulge = 0.0 if flat else r_L
    z_cas0 = 0.0 if flat else -(bulge + CASING_GAP_MIN_MM)
    A_an_geo = math.pi * (r_cas ** 2 - (r_L + t) ** 2)
    _note_annulus(notes, A_an_geo, dm["A_an_mm2"], False)
    return dict(
        liner=prof, casing=[[z_cas0, r_cas], [z, r_cas]],
        z_dome_mm=0.0, dome_type=dome, dome_bulge_mm=bulge, flow_sign=1.0,
        z_chamber=[0.0, L_tot], r_casing_mm=r_cas, r_inner_mm=0.0,
        r_outer_mm=r_L, head_plate_t_mm=HEAD_PLATE_T_MM, notes=notes,
        flow_path=[
            dict(id="anillo_carcasa", role="aire de refrigeración y dilución",
                 dir_z=1.0, turn_deg=0.0, z0_mm=z_cas0, z1_mm=z,
                 r_in_mm=r_L + t, r_out_mm=r_cas, area_mm2=A_an_geo),
            dict(id="liner", role="gas de combustión", dir_z=1.0,
                 turn_deg=0.0, z0_mm=0.0, z1_mm=z, r_in_mm=0.0, r_out_mm=r_L,
                 area_mm2=math.pi * r_L * r_L)])


def _geom_annular(dm: dict, w: dict, gates: dict) -> dict:
    """`annular` — corona entre `r_li` y `r_lo` con `π(r_lo²−r_li²) = A_paso`,
    domo toroidal en z = 0 y salida a la corona de la turbina. Gas a +Z."""
    notes: list[str] = []
    t, A = w["t_liner_mm"], dm["A_pass_mm2"]
    r_li, r_lo, r_cas = _annulus_geometry(dm, w, notes)
    L_pz, L_sz, L_tot = dm["L_pz_mm"], dm["L_sz_mm"], dm["L_tot_mm"]
    prof = [[0.0, r_li, r_lo], [L_pz, r_li, r_lo],
            [L_pz + L_sz, r_li, r_lo], [L_tot, r_li, r_lo]]
    z = L_tot
    r_out_ex = math.sqrt(r_li * r_li + max(dm["A4_mm2"], 1e-9) / math.pi)
    if r_lo - r_out_ex > 0.5:
        # contracción de salida a radio INTERIOR fijo (el cubo de la turbina
        # no se mueve): baja la pared exterior con la generatriz a 55°.
        dz = (r_lo - r_out_ex) * math.tan(math.radians(EXIT_CONE_DEG))
        prof.append([z + dz, r_li, r_out_ex])
        z += dz
    else:
        r_out_ex = r_lo
        notes.append("A4 ≈ A_L: la corona de salida no contrae")
    prof.append([z + EXIT_LIP_MM, r_li, r_out_ex])
    z += EXIT_LIP_MM

    bulge = math.sqrt(max(r_lo * r_lo - r_li * r_li, 0.0))
    z_cas0 = -(bulge + CASING_GAP_MIN_MM)
    A_an_geo = math.pi * (r_cas ** 2 - (r_lo + t) ** 2)
    _note_annulus(notes, A_an_geo, dm["A_an_mm2"], True)
    return dict(
        liner=prof, casing=[[z_cas0, r_cas], [z, r_cas]],
        z_dome_mm=0.0, dome_type="hemispherical", dome_bulge_mm=bulge,
        flow_sign=1.0, z_chamber=[0.0, L_tot], r_casing_mm=r_cas,
        r_inner_mm=r_li, r_outer_mm=r_lo, head_plate_t_mm=HEAD_PLATE_T_MM,
        notes=notes,
        flow_path=[
            dict(id="anillo_exterior", role="aire de refrigeración y dilución",
                 dir_z=1.0, turn_deg=0.0, z0_mm=z_cas0, z1_mm=z,
                 r_in_mm=r_lo + t, r_out_mm=r_cas, area_mm2=A_an_geo),
            dict(id="liner", role="gas de combustión", dir_z=1.0,
                 turn_deg=0.0, z0_mm=0.0, z1_mm=z, r_in_mm=r_li,
                 r_out_mm=r_lo, area_mm2=A)])


def _geom_reverse(dm: dict, w: dict, gates: dict) -> dict:
    """`reverse_annular` — anular de FLUJO INVERSO (la arquitectura del caso
    ancla): el gas hace **dos giros de 180°** y el eje geométrico sigue
    siendo Z (WAVE3 §1.1).

    Polilínea, ordenada por z creciente:

      1. **corona de salida** a radio pequeño en `z = 0` (r_in = cubo del eje
         y de la turbina, r_out resuelto para dar `A_paso`), con un tramo
         recto tan largo como pide la cortina del segundo giro;
      2. **transición radial** a `EXIT_CONE_DEG` y **área constante**;
      3. **cámara anular** a radio grande hasta el domo en `z = z_domo`.

    El recorrido del gas —outer +Z, giro de 180° en el domo, liner −Z, giro
    de 180° en el frente hacia la turbina que va ALOJADA dentro del propio
    combustor— vive en `flow_path[].dir_z`, nunca en el eje geométrico."""
    notes: list[str] = []
    t, A = w["t_liner_mm"], dm["A_pass_mm2"]
    r_li, r_lo, r_cas = _annulus_geometry(dm, w, notes)
    r_in_ex = max(EXIT_HUB_OVER_RLI * r_li, FEATURE_MIN_MM)
    r_out_ex = math.sqrt(r_in_ex * r_in_ex + A / math.pi)

    # cortina del giro de salida: el giro NO puede ser la garganta
    L_turn_out = TURN_AREA_SF * A / max(2.0 * math.pi * r_out_ex, 1e-9)
    L_lip = max(EXIT_LIP_MM, L_turn_out)
    prof = [[0.0, r_in_ex, r_out_ex], [L_lip, r_in_ex, r_out_ex]]
    rows, dz = _transition(L_lip, r_in_ex, r_li, A)
    prof += rows
    z_ch0 = L_lip + dz
    L_pz, L_sz, L_dz = dm["L_pz_mm"], dm["L_sz_mm"], dm["L_dz_mm"]
    # el gas va a −Z: DZ toca la salida, PZ toca el domo
    for zz in (z_ch0 + L_dz, z_ch0 + L_dz + L_sz, z_ch0 + L_dz + L_sz + L_pz):
        prof.append([zz, r_li, r_lo])
    z_dome = z_ch0 + dm["L_tot_mm"]
    L_turn_dome = TURN_AREA_SF * A / max(2.0 * math.pi * r_lo, 1e-9)
    bulge = math.sqrt(max(r_lo * r_lo - r_li * r_li, 0.0))
    z_cas1 = z_dome + max(L_turn_dome, bulge + CASING_GAP_MIN_MM)
    A_an_geo = math.pi * (r_cas ** 2 - (r_lo + t) ** 2)
    _note_annulus(notes, A_an_geo, dm["A_an_mm2"], True)
    return dict(
        liner=prof, casing=[[0.0, r_cas], [z_cas1, r_cas]],
        z_dome_mm=z_dome, dome_type="hemispherical", dome_bulge_mm=bulge,
        flow_sign=-1.0, z_chamber=[z_ch0, z_dome], r_casing_mm=r_cas,
        r_inner_mm=r_li, r_outer_mm=r_lo, head_plate_t_mm=0.0, notes=notes,
        flow_path=[
            dict(id="anillo_exterior",
                 role="aire del difusor, se precalienta contra el liner",
                 dir_z=1.0, turn_deg=0.0, z0_mm=0.0, z1_mm=z_dome,
                 r_in_mm=r_lo + t, r_out_mm=r_cas, area_mm2=A_an_geo),
            dict(id="giro_domo", role="primer giro de 180° hacia el liner",
                 dir_z=0.0, turn_deg=180.0, z0_mm=z_dome, z1_mm=z_cas1,
                 r_in_mm=r_li, r_out_mm=r_cas,
                 area_mm2=2.0 * math.pi * r_lo * L_turn_dome),
            dict(id="liner", role="gas de combustión (PZ en el domo, DZ en "
                                  "la salida)",
                 dir_z=-1.0, turn_deg=0.0, z0_mm=z_dome, z1_mm=z_ch0,
                 r_in_mm=r_li, r_out_mm=r_lo, area_mm2=A),
            dict(id="giro_salida",
                 role="segundo giro de 180° hacia la turbina",
                 dir_z=0.0, turn_deg=180.0, z0_mm=z_ch0, z1_mm=0.0,
                 r_in_mm=r_in_ex, r_out_mm=r_out_ex,
                 area_mm2=2.0 * math.pi * r_out_ex * L_lip),
            dict(id="conducto_turbina",
                 role="la turbina va ALOJADA dentro del combustor; su "
                      "conducto NO pertenece a este contrato (Phy-TB)",
                 dir_z=1.0, turn_deg=0.0, z0_mm=0.0, z1_mm=z_dome,
                 r_in_mm=0.0, r_out_mm=r_in_ex, area_mm2=None)])


_GEOM_BY_ARCH = {"can": _geom_can, "annular": _geom_annular,
                 "reverse_annular": _geom_reverse}


def _geometry(dm: dict, w: dict, gates: dict) -> dict:
    """Geometría de la arquitectura pedida + su validación de perfil."""
    geo = _GEOM_BY_ARCH.get(gates["arch"], _geom_can)(dm, w, gates)
    geo["liner"] = [[_r(v) for v in row] for row in geo["liner"]]
    geo["casing"] = [[_r(v) for v in row] for row in geo["casing"]]
    ok, errs = _profile_ok(geo["liner"])
    areas = [_passage_area_mm2(row) for row in geo["liner"]]
    a_min = min(areas) if areas else 0.0
    # tolerancia RELATIVA: la polilínea se redondea a ROUND_N decimales y
    # 2πr·5e-7 ya mueve el área en ~1e-4 mm². Comparar con 0 absoluto haría
    # fallar una geometría exacta por el redondeo de la interfaz.
    if a_min < dm["A_pass_mm2"] * (1.0 - AREA_REL_TOL) - AREA_ABS_TOL_MM2:
        ok = False
        errs.append(f"área de paso mínima {a_min:.3f} mm² < área de "
                    f"referencia {dm['A_pass_mm2']:.3f} mm² (T31)")
    geo["profile_ok"] = bool(ok)
    geo["profile_errors"] = errs
    geo["A_pass_min_mm2"] = float(a_min)
    return geo


def _z_of_s(geo: dict, s: float) -> float:
    """Coordenada geométrica z de una estación `s` medida a lo largo del
    RECORRIDO DEL GAS desde el domo (que es como el record da `z_mm`)."""
    return float(geo["z_dome_mm"] + geo["flow_sign"] * float(s))


# ==========================================================================
# 5. Swirler, inyectores, filas de orificios, efusión y drenaje
# ==========================================================================
def _injector_count(record: dict, gates: dict) -> int:
    """Nº de inyectores: el del record (`atomizer.n_tubes` del vaporizador
    KJ66) si está; si no, 1 en `can` y `N_INJ_ANNULAR_DEFAULT` en anular."""
    atom = _rget(record, "atomizer", {})
    n = _num(atom.get("n_tubes")) if isinstance(atom, dict) else None
    if n is None:
        n = 1.0 if gates["arch"] == "can" else float(N_INJ_ANNULAR_DEFAULT)
    return int(min(max(round(n), 1), 64))


def _swirler(record: dict, dm: dict, geo: dict, n_inj: int,
             notes: list) -> dict:
    """Bloque `swirler`: UN swirler concéntrico por inyector.

    El área total `A_sw` la fija la física (`record["swirler"]["A_sw_m2"]`,
    que es quien cobra ΔP_sw); aquí solo se reparte entre inyectores y se
    resuelve en Ø. El cubo del record puede salir 0 (S_N bajo con β = 50°) y
    un swirler sin cubo no es fabricable ni sujeta al inyector: se impone
    `SWIRLER_HUB_MIN_FRAC` **conservando A_sw** (crece D_tip, no baja el
    área). El nº de álabes baja de los 8 nominales si el paso libre en el
    cubo no los admite — con nota."""
    sw = _rget(record, "swirler", {})
    sw = sw if isinstance(sw, dict) else {}
    a = _pos(sw.get("A_sw_m2"))
    A_tot = 1e6 * a if a is not None else SWIRLER_A_OVER_AREF * dm["A_ref_mm2"]
    A_one = max(A_tot / max(n_inj, 1), 1e-6)
    x = min(max(_num(sw.get("hub_over_tip")) or 0.0, 0.0), 0.95)
    if x < SWIRLER_HUB_MIN_FRAC:
        notes.append(
            f"D_cubo/D_punta del swirler = {x:.3f} no es fabricable; se sube "
            f"a {SWIRLER_HUB_MIN_FRAC:.2f} CONSERVANDO A_sw "
            f"({A_tot:.1f} mm² totales)")
        x = SWIRLER_HUB_MIN_FRAC
    d_tip = math.sqrt(4.0 * A_one / (math.pi * max(1.0 - x * x, 1e-3)))
    d_hub = x * d_tip
    n_v = int(SWIRLER_N_VANES)
    paso = math.pi * d_hub / max(n_v, 1) - SWIRLER_VANE_T_MM
    if paso < SWIRLER_VANE_GAP_MIN_MM:
        n_v = int(min(max(math.floor(
            math.pi * d_hub / (SWIRLER_VANE_T_MM + SWIRLER_VANE_GAP_MIN_MM)),
            SWIRLER_N_VANES_MIN), SWIRLER_N_VANES))
        notes.append(
            f"álabes del swirler reducidos a {n_v} (los "
            f"{SWIRLER_N_VANES} nominales dejarían un paso libre de "
            f"{paso:.2f} mm < {SWIRLER_VANE_GAP_MIN_MM:.1f} mm en el cubo)")
    beta = _num(sw.get("beta_deg"))
    return dict(type="axial", n_vanes=int(n_v),
                beta_deg=_r(abs(beta) if beta is not None else 50.0),
                SN=_r(_num(sw.get("SN")) or 0.0),
                d_hub_mm=_r(d_hub), d_tip_mm=_r(d_tip),
                vane_t_mm=_r(SWIRLER_VANE_T_MM),
                chord_mm=_r(max(1.5 * (0.5 * (d_tip - d_hub)), 3.0)),
                z_mm=_r(geo["z_dome_mm"]), count=int(n_inj),
                A_sw_total_mm2=_r(A_tot), A_sw_each_mm2=_r(A_one),
                monolithic_with_dome=True)


def _injectors(record: dict, dm: dict, geo: dict, w: dict, gates: dict,
               swirler: dict, n_inj: int, notes: list) -> dict:
    """Bloque `injectors` según la gate θ[10], con el anillo del domo y —
    para el vaporizador— el tubo «walking stick» completo."""
    r_out = geo["r_outer_mm"]
    r_in = geo["r_inner_mm"]
    ring = 0.0 if (n_inj == 1 and r_in <= 1e-3) else \
        (0.5 * (r_out + r_in) if r_in > 1e-3 else 0.5 * r_out)
    if ring > 1e-3 and n_inj > 1:
        paso = 2.0 * math.pi * ring / n_inj
        if paso < INJ_SPACING_FACTOR * swirler["d_tip_mm"]:
            notes.append(
                f"paso entre inyectores {paso:.1f} mm < "
                f"{INJ_SPACING_FACTOR:.2f}·D_punta del swirler "
                f"({INJ_SPACING_FACTOR * swirler['d_tip_mm']:.1f} mm): los "
                "asientos del domo se tocan — revisa el nº de inyectores")
    out = dict(type=gates["injector"], count=int(n_inj),
               ALR=_r(_rnum(record, "ALR", 4.0)),
               ring_r_mm=_r(ring), phi0_deg=0.0,
               body_d_mm=_r(swirler["d_tip_mm"]),
               SMD_um=_rnum(record, "SMD_um"),
               atomizer_model=str(_rget(record, "atomizer_model", "-")))
    if gates["injector"] == "vaporizer":
        atom = _rget(record, "atomizer", {})
        atom = atom if isinstance(atom, dict) else {}
        L_tube = _pos(atom.get("tube_length_mm")) or (0.35 * dm["L_tot_mm"])
        L_max = VAPORIZER_L_MAX_FRAC * dm["L_tot_mm"]
        if L_tube > L_max:
            notes.append(f"tubo vaporizador recortado de {L_tube:.1f} a "
                         f"{L_max:.1f} mm ({VAPORIZER_L_MAX_FRAC:.0%} de "
                         "L_tot): no puede atravesar la cámara")
            L_tube = L_max
        bend = _num(atom.get("elbow_deg"))
        out.update(tube_d_mm=_r(VAPORIZER_TUBE_D_MM),
                   tube_t_mm=_r(w["t_liner_mm"]),
                   tube_z0_mm=_r(geo["z_dome_mm"]),
                   tube_z1_mm=_r(_z_of_s(geo, L_tube)),
                   bend_deg=_r(abs(bend) if bend is not None else 180.0),
                   bend_r_mm=_r(1.5 * VAPORIZER_TUBE_D_MM),
                   n_tubes=int(n_inj))
    elif gates["injector"] == "airblast":
        out.update(prefilmer_t_mm=_r(SWIRLER_VANE_T_MM),
                   bore_d_mm=_r(max(0.25 * swirler["d_hub_mm"],
                                    FEATURE_MIN_MM)))
    elif gates["injector"] == "simplex":
        mdot_f = (max(_rnum(record, "mdot_fuel_kgs", 1e-4), 1e-9)
                  / max(n_inj, 1))
        dp = max(_rnum(record, "dp_liner_Pa", 5e3), 1e2)
        rho_f = 800.0                      # jet-A líquido ⚠️ valor de clase
        A_b = mdot_f / max(INJECTOR_BORE_CD * math.sqrt(2.0 * rho_f * dp),
                           1e-12)
        out.update(bore_d_mm=_r(max(1e3 * math.sqrt(4.0 * A_b / math.pi),
                                    FEATURE_MIN_MM)),
                   bore_Cd=INJECTOR_BORE_CD)
    elif gates["injector"] == "micromix":
        n_jets = int(min(max(8 * n_inj, 8), 512))
        out.update(micromix_n=n_jets,
                   micromix_d_mm=_r(max(FEATURE_MIN_MM,
                                        0.20 * swirler["d_hub_mm"])))
    return out


def _hole_rows(record: dict, dm: dict, geo: dict, notes: list) -> list[dict]:
    """Filas de orificios del contrato a partir de las del record.

    Se traduce `z_mm` (que el record da a lo largo del RECORRIDO DEL GAS
    desde el domo) a la coordenada geométrica Z, se **reproyecta** `r_mm`
    sobre la pared real del liner si el record se desvía más de
    `HOLE_R_TOL_MM`, se desfasan las filas consecutivas media división
    (regla de mezcla de Lefebvre & Norster) y se añade el alojamiento de la
    bujía, que NO lleva caudal (`flow: false`) para no contaminar el reparto
    de aire."""
    out: list[dict] = []
    src = _rget(record, "hole_rows", [])
    src = src if isinstance(src, (list, tuple)) else []
    for i, hr in enumerate(src):
        if not isinstance(hr, dict):
            notes.append(f"record.hole_rows[{i}] no es un dict: descartada")
            continue
        d = _pos(hr.get("d_mm"))
        if d is None:
            notes.append(f"record.hole_rows[{i}] sin Ø legible: descartada")
            continue
        zone = str(hr.get("zone", "pz"))
        n = int(min(max(round(_num(hr.get("n")) or 1.0), 1), 400))
        z = _z_of_s(geo, _num(hr.get("z_mm")) or 0.0)
        r_wall = _interp(geo["liner"], z, 2)
        r_rec = _num(hr.get("r_mm"))
        r = r_wall
        if r_rec is not None and abs(r_rec - r_wall) <= HOLE_R_TOL_MM:
            r = r_rec
        elif r_rec is not None:
            notes.append(
                f"fila {zone}: r_mm del record {r_rec:.2f} mm reproyectado a "
                f"{r_wall:.2f} mm (pared real del liner en z = {z:.1f} mm)")
        out.append(dict(
            zone=zone, id=f"{zone}{i:02d}", z_mm=_r(z), r_mm=_r(r),
            n=n, d_mm=_r(d), Cd=_r(_num(hr.get("Cd")) or 0.62),
            shape="circle", plunged=False, angle_deg=0.0,
            phi0_deg=_r(180.0 / n if (i % 2) else 0.0), flow=True,
            A_row_mm2=_r(1e6 * (_num(hr.get("A_row_m2")) or 0.0)),
            J=_r(_num(hr.get("J")) or 0.0),
            Y_mm=_r(_num(hr.get("Y_mm")) or 0.0),
            Y_over_DL=_r(_num(hr.get("Y_over_DL")) or 0.0),
            mdot_kgs=_r(_num(hr.get("mdot_kgs")) or 0.0),
            U_jet_ms=_r(_num(hr.get("U_jet_ms")) or 0.0)))
    z_ig = _z_of_s(geo, IGNITER_S_OVER_LPZ * dm["L_pz_mm"])
    out.append(dict(
        zone="pz", id="bujia", z_mm=_r(z_ig),
        r_mm=_r(_interp(geo["liner"], z_ig, 2)), n=1, d_mm=_r(IGNITER_D_MM),
        Cd=0.0, shape="circle", plunged=False, angle_deg=0.0,
        phi0_deg=_r(IGNITER_PHI0_DEG), flow=False, A_row_mm2=0.0, J=0.0,
        Y_mm=0.0, Y_over_DL=0.0, mdot_kgs=0.0, U_jet_ms=0.0))
    return out


def _effusion(record: dict, dm: dict, geo: dict, t: list[float], gates: dict,
              notes: list) -> dict:
    """Campo de efusión: Ø y paso de θ[8]/θ[9], ángulo `EFFUSION_ANGLE_DEG`
    desde la SUPERFICIE y zona axial = la cámara.

    El paso es CUADRADO (P_x = P_y = θ[9]·d), que es exactamente la hipótesis
    con la que `physics_core.effusion_pattern` calcula `n_eff_holes` y la
    porosidad: si aquí se pusieran pasos distintos, el área perforada del
    contrato dejaría de ser la que refrigera en la física.

    CONTRASTE OBLIGATORIO CONTRA LA FÍSICA (auditoría 2026-07-31)
    -------------------------------------------------------------
    Este módulo sigue integrando el área mojada sobre la polilínea REAL —es
    el mejor dato que hay, y es el que se fabrica—, pero ya NO se limita a
    dejar una nota amable cuando no coincide con la nominal de
    `physics_core.effusion_wall_area_m2`. La efusión es UNA magnitud
    compartida entre capas: si las dos no casan dentro de
    `physics_core.EFFUSION_AREA_REL_TOL`, el reparto de aire de la física y
    el hardware que se imprime describen combustores distintos. Eso es una
    **violación del contrato entre capas** y se registra como tal en
    `contract["validation"]["effusion"]`, en `notes` y en el log.

    Hasta el 2026-07-31 el desacuerdo era de ×1.375 en las anulares porque la
    física aplicaba la porosidad sobre `π·D_L·L_tot`, y `π·D_L` es el
    perímetro de un CÍRCULO de área equivalente, no el de la corona real.
    """
    d = _pos(_rget(record, "d_eff_mm")) or _pos(t[8]) or 0.6
    pod = _pos(_rget(record, "pitch_over_d_eff")) or _pos(t[9]) or 6.0
    pitch = pod * d
    z0, z1 = sorted(geo["z_chamber"])
    L = max(z1 - z0, pitch)
    rows = int(max(math.floor(L / max(pitch, 1e-6)), 1))
    # superficie mojada REAL de la pared exterior de la cámara
    A_wall = 0.0
    prof = geo["liner"]
    for k in range(len(prof) - 1):
        za, zb = prof[k][0], prof[k + 1][0]
        lo, hi = max(min(za, zb), z0), min(max(za, zb), z1)
        if hi <= lo:
            continue
        A_wall += math.pi * (prof[k][2] + prof[k + 1][2]) * (hi - lo)
    #  El ÁREA MOJADA la pone este módulo (integra el perfil REAL, que es el
    #  mejor dato); el PATRÓN lo pone `physics_core.effusion_pattern`, que es
    #  su dueño. Así porosidad y nº de orificios no pueden divergir por tener
    #  la fórmula escrita dos veces. Respaldo etiquetado si no hay hermano.
    poro, n_geo, A_open = _effusion_pattern_mm(d, pod, A_wall)
    n_rec = int(_rnum(record, "n_eff_holes", 0))
    A_open_rec_mm2 = 1e6 * _rnum(record, "A_effusion_m2", 0.0)

    # --- contraste contra la FUENTE ÚNICA de `physics_core` ---------------
    eff = dict(d_mm=_r(d), pitch_x_over_d=_r(pod), pitch_y_over_d=_r(pod),
               pitch_x_mm=_r(pitch), pitch_y_mm=_r(pitch),
               angle_deg=_r(EFFUSION_ANGLE_DEG), shape=EFFUSION_SHAPE,
               cone_expansion=_r(EFFUSION_CONE_EXPANSION), staggered=True,
               porosity=_r(poro), n_holes=n_geo,
               n_holes_record=n_rec,
               wall_area_mm2=_r(A_wall),
               open_area_mm2=_r(A_open),
               open_area_record_mm2=_r(A_open_rec_mm2),
               zones=[dict(z0_mm=_r(z0), z1_mm=_r(z1), rows=rows)])
    eff["check"] = _effusion_check(record, gates, A_wall, A_open, n_geo,
                                   n_rec, A_open_rec_mm2, d, pod, notes)
    eff["Cd"] = eff["check"].get("Cd")
    eff["walls"] = eff["check"].get("walls")
    return eff


def _effusion_pattern_mm(d_mm: float, pod: float,
                         A_wall_mm2: float) -> tuple[float, int, float]:
    """`(porosidad, nº de orificios, área abierta [mm²])` en unidades de la
    interfaz, delegando en `physics_core.effusion_pattern` (fuente única).

    El respaldo local reproduce la MISMA fórmula de paso cuadrado y solo
    entra si el hermano no es importable — degradación etiquetada, no una
    segunda definición. Nunca lanza."""
    pc = _pc()
    if pc is not None:
        try:
            p = pc.effusion_pattern(1e-3 * d_mm, pod, 1e-6 * A_wall_mm2)
            return (float(p["porosity"]), int(p["n_holes"]),
                    1e6 * float(p["A_open_m2"]))
        except Exception:
            pass
    pitch = max(pod, 1.0) * max(d_mm, 0.05)
    poro = 0.25 * math.pi * d_mm * d_mm / max(pitch * pitch, 1e-12)
    return (poro, int(max(A_wall_mm2 / max(pitch * pitch, 1e-9), 0.0)),
            poro * A_wall_mm2)


def _effusion_check(record: dict, gates: dict, A_wall_geo: float,
                    A_open_geo: float, n_geo: int, n_rec: int,
                    A_open_rec_mm2: float, d_mm: float, pod: float,
                    notes: list) -> dict:
    """Contraste efusión GEOMÉTRICA ↔ efusión NOMINAL de `physics_core`.

    Devuelve el bloque que `build_contract` escribe en
    `contract["validation"]["effusion"]`. Una violación se deja en `notes`
    (que el reporte pinta) y `generate()` la repite en el LOG y en
    `warnings`. Nunca lanza: si `physics_core` no es importable se marca
    `source = "-"` y `ok = None` (no se puede afirmar ni negar), que es
    honesto.
    """
    out = dict(source="-", ok=None, wall_area_nominal_mm2=None,
               open_area_nominal_mm2=None, n_holes_nominal=None,
               wall_area_rel_dev=None, open_area_rel_dev=None,
               n_holes_dev=None, tol_rel=None, Cd=None, walls=None,
               violations=[])
    pc = _pc()
    if pc is None:
        notes.append(
            "no se pudo contrastar el campo de efusión contra `physics_core` "
            "(hermano no importable desde este directorio): el contrato lleva "
            "el área GEOMÉTRICA sin verificar contra la física")
        return out
    try:
        arch = gates.get("arch", "can")
        A_wall_nom = 1e6 * float(pc.effusion_wall_area_m2(arch, record))
        pat = pc.effusion_pattern(1e-3 * d_mm, pod, 1e-6 * A_wall_nom)
        A_open_nom = 1e6 * float(pat["A_open_m2"])
        tol = float(getattr(pc, "EFFUSION_AREA_REL_TOL", 2.0e-3))
        n_tol = int(getattr(pc, "EFFUSION_N_HOLES_TOL", 1))
        out.update(
            source="physics_core.effusion_wall_area_m2",
            wall_area_nominal_mm2=_r(A_wall_nom),
            open_area_nominal_mm2=_r(A_open_nom),
            n_holes_nominal=int(pat["n_holes"]),
            walls=list(getattr(pc, "EFFUSION_WALLS", {}).get(arch,
                                                             ("outer",))),
            Cd=_r(float(pc.effusion_discharge_coefficient(1e-3 * d_mm)), 9),
            tol_rel=tol)
        dev_w = (A_wall_geo - A_wall_nom) / max(abs(A_wall_nom), 1e-12)
        dev_o = (A_open_geo - A_open_nom) / max(abs(A_open_nom), 1e-12)
        out["wall_area_rel_dev"] = _r(dev_w, 9)
        out["open_area_rel_dev"] = _r(dev_o, 9)
        out["n_holes_dev"] = int(n_geo - int(pat["n_holes"]))
        viol = []
        if abs(dev_w) > tol:
            viol.append(
                f"área MOJADA de la pared caliente: geometría "
                f"{A_wall_geo:.1f} mm² frente a {A_wall_nom:.1f} mm² de "
                f"`physics_core.effusion_wall_area_m2` ({100.0 * dev_w:+.2f} "
                f"%, tolerancia declarada ±{100.0 * tol:.2f} %)")
        if abs(dev_o) > tol:
            viol.append(
                f"área ABIERTA de efusión: contrato {A_open_geo:.1f} mm² "
                f"frente a {A_open_nom:.1f} mm² de la física "
                f"({100.0 * dev_o:+.2f} %)")
        if abs(int(n_geo) - int(pat["n_holes"])) > n_tol:
            viol.append(
                f"nº de orificios de efusión: {n_geo} geométricos frente a "
                f"{int(pat['n_holes'])} nominales (holgura admitida "
                f"±{n_tol})")
        #  El record es lo que consume la red 1D: si NO es lo que este
        #  contrato declara, la física y el hardware no describen el mismo
        #  combustor aunque la fórmula nominal case.
        if A_open_rec_mm2 > 0.0:
            dev_r = (A_open_geo - A_open_rec_mm2) / max(A_open_rec_mm2, 1e-12)
            out["open_area_record_rel_dev"] = _r(dev_r, 9)
            if abs(dev_r) > tol:
                viol.append(
                    f"`record['A_effusion_m2']` = {A_open_rec_mm2:.1f} mm² no "
                    f"casa con el área abierta del contrato "
                    f"{A_open_geo:.1f} mm² ({100.0 * dev_r:+.2f} %): la red "
                    f"1D y el LBM verían conductancias distintas")
        if n_rec > 0 and abs(n_geo - n_rec) > n_tol:
            viol.append(
                f"`record['n_eff_holes']` = {n_rec} frente a {n_geo} del "
                f"contrato")
        out["violations"] = viol
        out["ok"] = not viol
        if viol:
            msg = ("VIOLACIÓN DEL CONTRATO ENTRE CAPAS (efusión): "
                   + "; ".join(viol)
                   + ". Física y geometría describen combustores distintos: "
                     "no es una nota, hay que arreglar el modelo NOMINAL de "
                     "`physics_core` o la construcción de la corona de este "
                     "módulo.")
            notes.append(msg)
    except Exception as exc:                  # nunca lanza (CONTEXTO §2.3)
        notes.append(f"el contraste de efusión contra `physics_core` falló "
                     f"({type(exc).__name__}): el contrato lleva el área "
                     f"GEOMÉTRICA sin verificar")
    return out


def _annulus_width_mm(liner: list, casing: list, t_liner: float) -> tuple:
    """Ancho del canal liner↔carcasa (mediana y mínimo) sobre 33 estaciones.

    Réplica EXACTA del muestreo de `dfam_checker._cavity_nodes`, para que el
    Ø de puerto que se declara aquí sea el mismo que el checker exige."""
    try:
        zl = [liner[0][0], liner[-1][0]]
        zc = [casing[0][0], casing[-1][0]]
        z0, z1 = max(min(zl), min(zc)), min(max(zl), max(zc))
        if z1 <= z0:
            return 0.0, 0.0
        ws = []
        for k in range(33):
            z = z0 + (z1 - z0) * k / 32.0
            w = _interp([[c[0], 0.0, c[1]] for c in casing], z, 2) \
                - _interp(liner, z, 2) - t_liner
            if w > 1e-6:
                ws.append(w)
        if not ws:
            return 0.0, 0.0
        return float(np.median(ws)), float(min(ws))
    except Exception:
        return 0.0, 0.0


def _drain_ports(liner: list, casing: list, t_liner: float) -> list[dict]:
    """Puertos de drenaje de polvo del anillo liner↔carcasa.

    EstadoDelArte §4.4 («la restricción más subestimada»): ≥2 salidas con
    línea de visión por cavidad y Ø ≥ 2× el ancho del canal. Se declaran DOS,
    uno en cada extremo axial, para que la cavidad drene con **cualquiera**
    de las dos orientaciones de build — el checker elige la orientación y no
    puede quedarse sin puerto bajo."""
    w_med, _ = _annulus_width_mm(liner, casing, t_liner)
    if w_med <= 0.0:
        return []
    d = min(max(DRAIN_D_OVER_W * w_med, DRAIN_D_MIN_MM), DRAIN_D_CAP_MM)
    z0 = max(liner[0][0], casing[0][0])
    z1 = min(liner[-1][0], casing[-1][0])
    inset = min(DRAIN_INSET_FRAC * abs(z1 - z0), DRAIN_INSET_MAX_MM)
    return [dict(z_mm=_r(z0 + inset), d_mm=_r(d), phi_deg=0.0, boss=True,
                 role="drenaje/venteo del anillo liner-carcasa"),
            dict(z_mm=_r(z1 - inset), d_mm=_r(d), phi_deg=180.0, boss=True,
                 role="drenaje/venteo del anillo liner-carcasa")]


# ==========================================================================
# 6. Condiciones de contorno de CFD
# ==========================================================================
def cfd_boundary_conditions(theta, record: dict) -> dict:
    """Condiciones de contorno **copy-pasteables** para `cfd/` (WAVE5/6) y
    para cualquier solver externo.

    Entrada = anillo de la carcasa (aire del difusor del compresor), salida =
    presión estática del ciclo, paredes = el mapa 1D de `physics_core`
    (`walls = "1d_thermal"`), no adiabáticas: la temperatura de pared es un
    RESULTADO verificado del L0 y tirarla sería regalar información.

    Nunca lanza: todo se lee con respaldo y las magnitudes van en SI con el
    sufijo en la clave."""
    t = _theta_list(theta)
    g = _gates(theta, record)
    P3 = _rpos(record, "P3_Pa") or _pos(t[16]) or 101_325.0
    T3 = _rpos(record, "T3_K") or _pos(t[17]) or 288.15
    T4 = _rpos(record, "T4_K") or _pos(t[19]) or 1100.0
    mdot_air = _rpos(record, "mdot_air_kgs") or _pos(t[18]) or 0.1
    mdot_fuel = _rpos(record, "mdot_fuel_kgs") or \
        (mdot_air * (_rnum(record, "f_fuel_air", 0.02)))
    rho3 = _rpos(record, "rho3_kgm3") or (P3 / (287.0 * max(T3, 1.0)))
    D_L = _rpos(record, "D_L_mm", 50.0)
    D_cas = _rpos(record, "D_casing_mm", 1.4 * D_L)
    t_liner = _rpos(record, "t_liner_mm") or _pos(t[14]) or 0.8
    # D_h del anillo de refrigeración = 2·holgura (conducto anular). La
    # holgura se calcula con la MISMA regla con la que se construye la
    # geometría: en `can` el liner es D_L/2, en las anulares es
    # LINER_R_OVER_RCAS·r_carcasa (si no, D_h sale 5× grande y ℓ con él).
    r_cas = 0.5 * D_cas
    gap = (r_cas - 0.5 * D_L if g["arch"] == "can"
           else (1.0 - LINER_R_OVER_RCAS) * r_cas) - t_liner
    D_h_mm = max(2.0 * gap, 1.0)
    return dict(
        P3_Pa=_r(P3, 3), T3_K=_r(T3, 3), mdot_air_kgs=_r(mdot_air, 9),
        mdot_fuel_kgs=_r(mdot_fuel, 9), T4_target_K=_r(T4, 3),
        walls="1d_thermal", turbulence_intensity=TURBULENCE_INTENSITY,
        turbulence_length_scale_mm=_r(LENGTH_SCALE_OVER_DH * D_h_mm),
        inlet_hydraulic_d_mm=_r(D_h_mm), fuel=g["fuel"],
        fuel_LHV_J_kg=_r(_rnum(record, "LHV_J_kg", 43.1e6), 1),
        f_fuel_air=_r(_rnum(record, "f_fuel_air", 0.0), 9),
        rho3_kgm3=_r(rho3, 9),
        P_out_static_Pa=_r((_rnum(record, "P04_Pa", P3))
                           * (1.0 - _rnum(record, "dp_over_p", 0.0)), 3),
        P04_Pa=_r(_rnum(record, "P04_Pa", P3), 3),
        T04_K=_r(_rnum(record, "T04_K", T4), 3),
        mdot_out_kgs=_r(_rnum(record, "mdot4_kgs", mdot_air + mdot_fuel), 9),
        swirl_number=_r(_rnum(record, "swirl_SN", 0.0)),
        arch=g["arch"], injector=g["injector"],
        source=str(_rget(record, "source", "correlations_L0")),
        notes=["patch de entrada = anillo de la carcasa (aire del difusor); "
               "patch de salida = presión estática",
               "paredes NO adiabáticas: T_w1/T_w2 del mapa 1D del record",
               "ℓ = 0.07·D_h con D_h del anillo de refrigeración"])


# ==========================================================================
# 7. Contrato `phycb-combustor-1`
# ==========================================================================
def build_contract(theta, record: dict, structural: dict | None = None,
                   dfam: dict | None = None) -> dict:
    """θ + record VERIFICADO → contrato de geometría `phycb-combustor-1`.

    Devuelve el diccionario COMPLETO de WAVE3 §1.1 con la geometría
    **totalmente resuelta** en mm/deg: la capa 5c (C#/PicoGK) y los solvers
    de `cfd/` no recalculan nada.

    Parámetros
    ----------
    theta : array-like
        Vector de diseño de 20 componentes (se rellena si viene corto).
    record : dict
        Record `phycb-record-1` de `physics_core.evaluate`. Lectura
        DEFENSIVA total: un record incompleto degrada a respaldos
        geométricos coherentes y lo anota en `notes`.
    structural : dict | None
        Salida de `structures_core.evaluate_structural` (manda para los
        espesores y los materiales).
    dfam : dict | None
        Salida de `dfam_checker.check`: si se pasa, sus `hole_shapes`, su
        `orientation` y sus `drain_ports` sugeridos se APLICAN al contrato.

    **Nunca lanza** (lo llama el lazo del optimizador a través de
    `neural_optimizer.DesignSpec`): un perfil inválido se reporta en
    `validation.profile_ok`, no con una excepción. Es `generate()` quien
    falla ruidosamente."""
    t = _theta_list(theta)
    gates = _gates(theta, record)
    notes: list[str] = []
    dm = _dims(record, gates, t)
    w = _walls(record, structural, t)
    if abs(dm["L_tot_record_mm"] - dm["L_tot_mm"]) > 1e-6:
        notes.append(f"L_tot del record ({dm['L_tot_record_mm']:.3f} mm) no "
                     f"casa con L_pz+L_sz+L_dz ({dm['L_tot_mm']:.3f} mm): "
                     "manda la suma de las zonas")
    geo = _geometry(dm, w, gates)
    notes += geo["notes"]

    n_inj = _injector_count(record, gates)
    swirler = _swirler(record, dm, geo, n_inj, notes)
    injectors = _injectors(record, dm, geo, w, gates, swirler, n_inj, notes)
    rows = _hole_rows(record, dm, geo, notes)
    eff = _effusion(record, dm, geo, t, gates, notes)
    #  El contraste entre capas vive en `validation`, no dentro del bloque
    #  geométrico: `effusion` describe hardware, `validation` lo audita.
    eff_check = eff.pop("check")
    ports = _drain_ports(geo["liner"], geo["casing"], w["t_liner_mm"])

    r_bolt = geo["r_casing_mm"] + w["t_casing_mm"] + 0.5 * FLANGE_W_MM
    n_bolt = int(min(max(round(2.0 * math.pi * r_bolt / BOLT_SPACING_MM),
                         BOLT_COUNT_MIN), BOLT_COUNT_MAX))

    contract = dict(
        schema=SCHEMA,
        arch=gates["arch"],
        _conventions=" ".join(CONVENTIONS[k] for k in
                              ("axis", "units", "flow", "liner_profile",
                               "casing_profile", "hole_angle_deg",
                               "hole_phi0_deg", "effusion_angle_deg",
                               "effusion_walls",
                               "swirler_beta_deg", "signs", "truth")),
        _conventions_detail=dict(CONVENTIONS),
        design_vector={n: _r(v, 9) for n, v in zip(var_names(), t)},
        derived=dict(
            eta_comb=_r(_rnum(record, "eta_comb", 0.0), 9),
            dp_over_p=_r(_rnum(record, "dp_over_p", 0.0), 9),
            PF=_r(_rnum(record, "PF", 0.0), 9),
            T_wall_max_K=_r(_rnum(record, "T_wall_max_K", 0.0), 6),
            T04_K=_r(_rnum(record, "T04_K", 0.0), 6),
            P04_Pa=_r(_rnum(record, "P04_Pa", 0.0), 3),
            f_fuel_air=_r(_rnum(record, "f_fuel_air", 0.0), 9),
            EI_NOx=_r(_rnum(record, "EI_NOx", 0.0), 6),
            EI_CO=_r(_rnum(record, "EI_CO", 0.0), 6),
            phi_pz=_r(_rnum(record, "phi_pz", 0.0), 6),
            tau_ms=_r(_rnum(record, "tau_ms", 0.0), 6),
            SMD_um=_rnum(record, "SMD_um"),
            M_exit=_r(_rnum(record, "M_exit", 0.0), 6),
            mdot_air_kgs=_r(_rnum(record, "mdot_air_kgs", 0.0), 9),
            mdot_fuel_kgs=_r(_rnum(record, "mdot_fuel_kgs", 0.0), 9),
            A_ref_mm2=_r(dm["A_ref_mm2"]), A_L_mm2=_r(dm["A_L_mm2"]),
            A_pass_mm2=_r(dm["A_pass_mm2"]),
            A_pass_min_mm2=_r(geo["A_pass_min_mm2"]),
            A_annulus_mm2=_r(dm["A_an_mm2"]),
            D_L_mm=_r(dm["D_L_mm"]), D_casing_mm=_r(dm["D_casing_mm"]),
            L_pz_mm=_r(dm["L_pz_mm"]), L_sz_mm=_r(dm["L_sz_mm"]),
            L_dz_mm=_r(dm["L_dz_mm"]), L_tot_mm=_r(dm["L_tot_mm"]),
            feasible=bool(_rget(record, "feasible", False)),
            source=str(_rget(record, "source", "correlations_L0"))),
        liner=dict(
            profile=geo["liner"], t_wall_mm=_r(w["t_liner_mm"]),
            material=w["material_liner"], fillet_r_mm=_r(w["fillet_r_mm"]),
            dome=dict(type=geo["dome_type"], z_mm=_r(geo["z_dome_mm"]),
                      bulge_mm=_r(geo["dome_bulge_mm"]))),
        casing=dict(
            profile=geo["casing"], t_wall_mm=_r(w["t_casing_mm"]),
            material=w["material_casing"], flange_bolt_count=n_bolt,
            flange_bolt_d_mm=_r(BOLT_D_MM), flange_w_mm=_r(FLANGE_W_MM),
            flange_t_mm=_r(FLANGE_T_MM),
            head_plate_t_mm=_r(geo["head_plate_t_mm"]),
            boss_h_mm=_r(FLANGE_T_MM),
            note="carcasa cilíndrica de radio interior constante: contiene "
                 "el liner con holgura y NO es la garganta de ningún paso"),
        swirler=swirler,
        injectors=injectors,
        hole_rows=rows,
        effusion=eff,
        flow_path=[{k: (_r(v) if isinstance(v, float) else v)
                    for k, v in fp.items()} for fp in geo["flow_path"]],
        dfam=dict(build_orientation="z+",
                  min_wall_mm=_r(min(w["t_liner_mm"], w["t_casing_mm"])),
                  voxel_mm=_r(VOXEL_DEFAULT_MM),
                  coarse_voxel_mm=1.0, overhang_frac_max=0.005,
                  feature_floor_mm=_r(max(FEATURE_MIN_MM,
                                          2.0 * VOXEL_DEFAULT_MM)),
                  drain_ports=ports),
        cfd_boundary_conditions=cfd_boundary_conditions(theta, record),
        validation=dict(profile_ok=bool(geo["profile_ok"]),
                        profile_errors=list(geo["profile_errors"]),
                        self_intersects=bool(
                            _self_intersects(_meridional_polygon(
                                geo["liner"]))),
                        #  Contrato ENTRE CAPAS: la efusión que dibuja este
                        #  módulo y la que reparte aire `physics_core` tienen
                        #  que ser la misma. `ok=None` = no se pudo contrastar.
                        effusion_ok=eff_check.get("ok"),
                        effusion=eff_check),
        notes=notes,
        run_meta=dict(
            name=f"PhyCB_{gates['arch']}",
            schema=SCHEMA, seed=SEED,
            source=str(_rget(record, "source", "correlations_L0")),
            fidelity=f"L{int(_rnum(record, 'fidelity', 0))}",
            generator=GENERATOR,
            versions=dict(contract=SCHEMA,
                          record=str(_rget(record, "schema",
                                           "phycb-record-1")),
                          builder="CombustorDesigner")))

    if isinstance(structural, dict):
        contract["structural"] = dict(
            material=w["material_liner"], material_casing=w["material_casing"],
            t_liner_mm=_r(w["t_liner_mm"]), t_casing_mm=_r(w["t_casing_mm"]),
            T_metal_K=_r(_num(structural.get("T_metal_K")) or 0.0, 6),
            T_wall_allow_K=_r(
                _num(structural.get("T_wall_allow_K")) or 0.0, 6),
            sigma_th_MPa=_r(_num(structural.get("sigma_th_MPa")) or 0.0, 6),
            g_struct=[_r(_num(v) or 10.0, 9)
                      for v in (structural.get("g_struct") or [])],
            feasible=bool(structural.get("feasible_struct", False)),
            source=str(structural.get("source", "-")))

    if isinstance(dfam, dict):
        apply_dfam(contract, dfam)
    return _clean(contract)


def apply_dfam(contract: dict, dfam: dict) -> dict:
    """Aplica el veredicto de `dfam_checker.check` SOBRE el contrato.

    Tres cosas y ninguna más: la forma de cada fila (`circle`/`teardrop`, que
    es literalmente el trabajo del checker), la orientación de build ganadora
    y los puertos de drenaje sugeridos que aún no estuvieran declarados.
    Muta y devuelve el mismo dict. Nunca lanza."""
    if not (isinstance(contract, dict) and isinstance(dfam, dict)):
        return contract
    shapes = dfam.get("hole_shapes")
    shapes = shapes if isinstance(shapes, dict) else {}
    for i, row in enumerate(contract.get("hole_rows") or []):
        if not isinstance(row, dict):
            continue
        s = shapes.get(str(i)) or shapes.get(f"row{i:02d}") \
            or shapes.get(str(row.get("id", "")))
        if s in ("circle", "teardrop"):
            row["shape"] = s
    if isinstance(contract.get("effusion"), dict) \
            and shapes.get("effusion") == "teardrop":
        contract["effusion"]["shape"] = "teardrop"
    blk = contract.setdefault("dfam", {})
    if isinstance(dfam.get("orientation"), str):
        blk["build_orientation"] = dfam["orientation"]
    have = blk.setdefault("drain_ports", [])
    zs = [round(float(p.get("z_mm", 0.0)), 3) for p in have
          if isinstance(p, dict)]
    for p in (dfam.get("drain_ports") or []):
        if not isinstance(p, dict):
            continue
        z = _num(p.get("z_mm"))
        d = _pos(p.get("d_mm"))
        if z is None or d is None or round(z, 3) in zs:
            continue
        have.append(dict(z_mm=_r(z), d_mm=_r(d), phi_deg=90.0, boss=True,
                         role=str(p.get("role", "sugerido por dfam_checker"))))
        zs.append(round(z, 3))
    blk["g_dfam"] = [_r(_num(v) if _num(v) is not None else 10.0, 9)
                     for v in (dfam.get("g_dfam") or [])]
    blk["feasible"] = bool(dfam.get("feasible", False))
    blk["violations"] = [str(v) for v in (dfam.get("violations") or [])]
    return contract


# ==========================================================================
# 8. STEP opcional (CadQuery) — sólidos de revolución, NUNCA la efusión
# ==========================================================================
def _cq():
    """CadQuery o `None`. Dependencia OPCIONAL (contrato §10.4)."""
    try:
        import cadquery as cq          # noqa: PLC0415
        return cq
    except Exception:
        return None


def _cq_shell(cq, profile: list, col: int, t: float, outward: bool):
    """Cáscara de revolución que sigue una generatriz de la polilínea.

    `outward=True` engorda hacia +r (pared exterior del liner, carcasa);
    `False` hacia −r (pared interior de una corona). Es el mismo patrón que
    `Phy-AC.geometry_generator._cq_line_ring`."""
    inner = [(float(row[col]) + (0.0 if outward else -t), float(row[0]))
             for row in profile]
    outer = [(float(row[col]) + (t if outward else 0.0), float(row[0]))
             for row in profile]
    return (cq.Workplane("XZ").polyline(inner + outer[::-1]).close()
            .revolve(360.0, (0, 0), (0, 1)))


def _step_readme(contract: dict) -> str:
    """Notas de re-CAD del STEP (ASCII: lo lee cualquier CAD)."""
    liner = contract.get("liner") or {}
    eff = contract.get("effusion") or {}
    lines = [
        "STEP de re-CAD Phy-CB - notas",
        "=============================",
        f"Contrato: {contract.get('schema', '-')}  arquitectura: "
        f"{contract.get('arch', '-')}",
        "Coordenadas de maquina: Z = eje de revolucion, milimetros.",
        "",
        "Lo que SI esta en el STEP:",
        "  * liner (cascara de revolucion del perfil, espesor "
        f"{liner.get('t_wall_mm', '-')} mm)",
        "  * carcasa (cascara de revolucion)",
        "",
        "Lo que NO esta (solo en el STL de la capa 5c):",
        "  * campo de efusion: "
        f"{eff.get('n_holes', 0)} orificios de {eff.get('d_mm', '-')} mm.",
        "    OCC no aguanta esa booleana - se resuelve como campo implicito",
        "    en PicoGK (WAVE3 seccion 6). Las posiciones exactas estan en",
        "    effusion_field.csv y hole_table.csv, que son la REFERENCIA",
        "    DIMENSIONAL (patron doble via: el STL aproxima, el CSV manda).",
        "  * filas de orificios discretos, swirler, inyectores, bridas,",
        "    puertos de drenaje y boss de bujia.",
        "",
        "Filas de orificios a practicar en el CAD destino:",
    ]
    for row in (contract.get("hole_rows") or []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"  {str(row.get('id', '?')):<8s}"
            f" zona {str(row.get('zone', '?')):<3s}"
            f"  n={row.get('n', 0):<3d} d={row.get('d_mm', 0):.2f} mm"
            f"  z={row.get('z_mm', 0):.1f} mm  r={row.get('r_mm', 0):.1f} mm"
            f"  {row.get('shape', 'circle')}")
    lines.append("")
    return "\n".join(lines)


def export_step(contract: dict, outdir: str, mode: str = "parts") -> list[str]:
    """Exporta el STEP del diseño. Devuelve las rutas escritas.

    `[]` sin CadQuery — **degradación silenciosa etiquetada**, nunca una
    excepción (contrato §10.4: solo NumPy es obligatorio).

    mode:
      `"liner"`    — solo el liner (rápido, compatibilidad).
      `"parts"`    — una pieza por parte + `parts/README.txt` (default).
      `"assembly"` — además `assembly.step` con las piezas nombradas.
    """
    cq = _cq()
    if cq is None or not isinstance(contract, dict):
        return []
    written: list[str] = []
    try:
        prof = (contract.get("liner") or {}).get("profile") or []
        if len(prof) < 2:
            return []
        t_l = _pos((contract.get("liner") or {}).get("t_wall_mm")) or 0.8
        parts = {"liner": _cq_shell(cq, prof, 2, t_l, True)}
        if max(float(r[1]) for r in prof) > 1e-3:
            parts["liner_inner"] = _cq_shell(cq, prof, 1, t_l, False)
        cas = (contract.get("casing") or {}).get("profile") or []
        if len(cas) >= 2:
            t_c = _pos((contract.get("casing") or {}).get("t_wall_mm")) or 1.2
            parts["casing"] = _cq_shell(
                cq, [[c[0], 0.0, c[1]] for c in cas], 2, t_c, True)
        if mode == "liner":
            path = os.path.join(outdir, "liner.step")
            os.makedirs(outdir, exist_ok=True)
            cq.exporters.export(parts["liner"], path)
            return [path]
        pdir = os.path.join(outdir, "parts")
        os.makedirs(pdir, exist_ok=True)
        for name, solid in parts.items():
            path = os.path.join(pdir, f"{name}.step")
            cq.exporters.export(solid, path)
            written.append(path)
        readme = os.path.join(pdir, "README.txt")
        with open(readme, "w", encoding="utf-8") as f:
            f.write(_step_readme(contract))
        written.append(readme)
        if mode == "assembly":
            asm = cq.Assembly()
            for name, solid in parts.items():
                asm.add(solid, name=name)
            path = os.path.join(outdir, "assembly.step")
            asm.save(path)
            written.append(path)
        return written
    except Exception:
        return written


def try_export_step(contract: dict, path: str) -> bool:
    """Compatibilidad: intenta el STEP del liner en `path`. `False` sin
    error si CadQuery no está instalado."""
    return bool(export_step(contract, os.path.dirname(path) or ".",
                            mode="liner"))


# ==========================================================================
# 9. Paquete de geometría en disco
# ==========================================================================
def _write_csv(path: str, header: list, rows: list) -> str:
    """CSV UTF-8 con final de línea estable (convención §10.3)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for row in rows:
            wr.writerow(row)
    return path


def _hole_table_rows(contract: dict) -> list:
    out = []
    for i, r in enumerate(contract.get("hole_rows") or []):
        out.append([i, r.get("id"), r.get("zone"), r.get("z_mm"),
                    r.get("r_mm"), r.get("phi0_deg"), r.get("n"),
                    r.get("d_mm"), r.get("Cd"), r.get("shape"),
                    bool(r.get("plunged")), r.get("angle_deg"),
                    r.get("A_row_mm2"), r.get("J"), r.get("Y_mm"),
                    r.get("Y_over_DL"), r.get("mdot_kgs"), r.get("U_jet_ms"),
                    bool(r.get("flow", True))])
    return out


def _liner_profile_rows(contract: dict) -> list:
    liner = (contract.get("liner") or {}).get("profile") or []
    cas = (contract.get("casing") or {}).get("profile") or []
    t = _pos((contract.get("liner") or {}).get("t_wall_mm")) or 0.0
    cas3 = [[c[0], 0.0, c[1]] for c in cas] if len(cas) >= 2 else None
    out = []
    for row in liner:
        z, ri, ro = float(row[0]), float(row[1]), float(row[2])
        rc = _interp(cas3, z, 2) if cas3 else 0.0
        out.append([_r(z), _r(ri), _r(ro), _r(_passage_area_mm2(row)),
                    _r(rc), _r(max(rc - ro - t, 0.0)),
                    _r(math.pi * max(rc * rc - (ro + t) ** 2, 0.0))])
    return out


def _effusion_rows(contract: dict) -> list:
    eff = contract.get("effusion") or {}
    out = []
    for k, zn in enumerate(eff.get("zones") or []):
        z0, z1 = float(zn.get("z0_mm", 0.0)), float(zn.get("z1_mm", 0.0))
        n_rows = int(zn.get("rows", 0))
        for j in range(max(n_rows, 0)):
            z = z0 + (z1 - z0) * (j + 0.5) / max(n_rows, 1)
            r = _interp((contract.get("liner") or {}).get("profile") or [],
                        z, 2)
            n_hole = int(max(round(2.0 * math.pi * r
                                   / max(eff.get("pitch_y_mm", 1.0), 1e-6)),
                             1))
            stag = bool(eff.get("staggered")) and bool(j % 2)
            phi0 = (180.0 / n_hole) if stag else 0.0
            out.append([k, j, _r(z), _r(r), n_hole, eff.get("d_mm"),
                        _r(phi0), eff.get("angle_deg"), eff.get("shape"),
                        eff.get("cone_expansion")])
    return out


def _bom_rows(contract: dict) -> list:
    """Lista de materiales — ítems en ESPAÑOL (convención de la familia)."""
    liner = contract.get("liner") or {}
    cas = contract.get("casing") or {}
    sw = contract.get("swirler") or {}
    inj = contract.get("injectors") or {}
    eff = contract.get("effusion") or {}
    prof = liner.get("profile") or [[0, 0, 0]]
    cprof = cas.get("profile") or [[0.0, 0.0]]
    r_cas = float(cprof[0][1])
    L = float(prof[-1][0]) - float(prof[0][0])
    r_max = max(float(p[2]) for p in prof)
    n_inj = int(inj.get("count", 1))
    def q(key, blk=None, n=2):
        """Magnitud del contrato formateada para la lista (2 decimales:
        una BOM con 6 no la lee nadie; el número exacto vive en el JSON)."""
        v = _num((blk if blk is not None else cas).get(key))
        return "-" if v is None else f"{v:.{n}f}"

    dome = liner.get("dome") or {}
    rows = [
        ["liner (tubo de llama)", 1, liner.get("material"),
         f"L{L:.1f} x r{r_max:.1f} x t{q('t_wall_mm', liner)}",
         f"{contract.get('arch')} · efusión Ø{q('d_mm', eff)} mm x "
         f"{eff.get('n_holes', 0)} orificios"],
        [f"domo del liner ({dome.get('type', '-')})", 1,
         liner.get("material"),
         f"z{q('z_mm', dome, 1)} x t{q('t_wall_mm', liner)}",
         "monolítico con el liner en LPBF"],
        ["carcasa (recipiente a presión)", 1, cas.get("material"),
         f"r{r_cas:.1f} x t{q('t_wall_mm')}",
         "cilíndrica; contiene el liner con holgura"],
        ["brida de carcasa", 2, cas.get("material"),
         f"w{q('flange_w_mm')} x t{q('flange_t_mm')}",
         f"{cas.get('flange_bolt_count')} pernos cada una"],
        [f"perno de brida M{int(cas.get('flange_bolt_d_mm') or 6)}",
         2 * int(cas.get("flange_bolt_count") or 0), "acero 8.8",
         f"M{int(cas.get('flange_bolt_d_mm') or 6)}", "elemento comercial"],
        ["swirler axial", n_inj, liner.get("material"),
         f"D{q('d_tip_mm', sw)}/{q('d_hub_mm', sw)} x "
         f"{sw.get('n_vanes')} álabes t{q('vane_t_mm', sw)}",
         f"beta = {q('beta_deg', sw, 1)} deg, S_N = {q('SN', sw)}"],
        [f"inyector ({inj.get('type')})", n_inj, liner.get("material"),
         f"anillo r{q('ring_r_mm', inj)}",
         f"ALR = {q('ALR', inj)}, SMD = {q('SMD_um', inj, 1)} um"],
    ]
    if inj.get("type") == "vaporizer":
        rows.append(["tubo vaporizador «walking stick»", n_inj,
                     liner.get("material"),
                     f"Ø{q('tube_d_mm', inj)} x t{q('tube_t_mm', inj)} "
                     f"x codo {q('bend_deg', inj, 1)} deg",
                     "barrido a lo largo de la polilínea en la capa 5c"])
    if float(cas.get("head_plate_t_mm") or 0.0) > 0.0:
        rows.append(["placa de cabeza de la carcasa", 1, cas.get("material"),
                     f"t{q('head_plate_t_mm')}",
                     "taladrada para los inyectores"])
    rows.append(["bujía / encendedor", 1, "comercial",
                 f"Ø{IGNITER_D_MM:.2f}",
                 "alojamiento sellado: NO lleva caudal"])
    ports = (contract.get("dfam") or {}).get("drain_ports") or []
    if ports:
        rows.append(["puerto de drenaje de polvo", len(ports),
                     cas.get("material"), f"Ø{q('d_mm', ports[0])}",
                     "EstadoDelArte §4.4: ≥2 salidas con línea de visión"])
    return rows


def generate(theta, record: dict, outdir: str,
             structural: dict | None = None, dfam: dict | None = None,
             voxel_mm: float = VOXEL_DEFAULT_MM,
             build_orientation: str = "z+", step: bool = True,
             step_mode: str = "parts", log=None) -> dict:
    """Escribe el paquete de geometría COMPLETO en `outdir`.

    Archivos (contrato §9): `combustor.json`, `hole_table.csv`,
    `liner_profile.csv`, `effusion_field.csv`, `combustor_bom.csv`,
    `cfd_boundary_conditions.json`, `dfam_report.json` y, si CadQuery está
    instalado, `parts/*.step`.

    Encadena el verificador DfAM: `check` → se aplican las formas
    (`circle`/`teardrop`) y los puertos sugeridos → `check` otra vez sobre el
    contrato YA corregido, que es el que se escribe. Es un bucle **acotado**
    de dos pasadas, nunca abierto.

    **Falla ruidosamente** con `ValueError` si la polilínea del liner no
    valida (auto-intersección o área de paso por debajo de la de
    referencia): escribir un contrato inconstruible sería peor que no
    escribir nada. `build_contract` sí es silencioso — lo llama el lazo del
    optimizador.

    Devuelve un dict con `contract`, `files`, `dfam`, `notes` y `warnings`.
    """
    say = log if callable(log) else (lambda *_a, **_k: None)
    # El contrato se construye ANTES de tocar el disco: si el perfil no
    # valida no se crea ni el directorio (nada de paquetes a medias).
    contract = build_contract(theta, record, structural, dfam)
    val = contract.get("validation") or {}
    if not val.get("profile_ok", False):
        raise ValueError(
            "la polilínea del liner de la arquitectura "
            f"'{contract.get('arch')}' NO valida: "
            + "; ".join(val.get("profile_errors") or ["motivo desconocido"]))
    os.makedirs(outdir, exist_ok=True)

    # --- contrato ENTRE CAPAS: la efusión de la física y la del dibujo ----
    #  No se aborta (el paquete es fabricable y el área geométrica es la
    #  buena), pero se dice EN VOZ ALTA en el log y en `warnings`, que es lo
    #  que el reporte pinta. En silencio nunca.
    res, warnings = None, []
    _ec = (val.get("effusion") or {}) if isinstance(val, dict) else {}
    if _ec.get("ok") is False:
        w = ("CROSS-LAYER CONTRACT VIOLATION (effusion): "
             + "; ".join(_ec.get("violations") or ["reason unknown"]))
        warnings.append(w)
        say("  ⚠️ " + w)
    elif _ec.get("ok") is None:
        warnings.append(
            "the effusion field could NOT be cross-checked against "
            "`physics_core`: the contract carries the GEOMETRIC area "
            "unverified")

    # --- verificador DfAM: dos pasadas ACOTADAS ---------------------------
    dc = _dfam()
    if dc is None:
        warnings.append("dfam_checker is not importable from this "
                        "directory: the contract is written WITHOUT DfAM "
                        "verification")
    else:
        for _ in range(2):                  # bucle ACOTADO (contrato §10.5)
            res = dc.check(contract, build_orientation=build_orientation,
                           voxel_mm=voxel_mm)
            apply_dfam(contract, res)
        contract = _clean(contract)
        if not res.get("feasible", False):
            warnings.append(
                "DfAM INFEASIBLE: " + "; ".join(res.get("violations") or []))

    files: list[str] = []
    p = os.path.join(outdir, "combustor.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=1, ensure_ascii=False, sort_keys=False,
                  allow_nan=False)
    files.append(p)

    files.append(_write_csv(
        os.path.join(outdir, "hole_table.csv"),
        ["row", "id", "zone", "z_mm", "r_mm", "phi0_deg", "n", "d_mm", "Cd",
         "shape", "plunged", "angle_deg", "A_row_mm2", "J", "Y_mm",
         "Y_over_DL", "mdot_kgs", "U_jet_ms", "flow"],
        _hole_table_rows(contract)))
    files.append(_write_csv(
        os.path.join(outdir, "liner_profile.csv"),
        ["z_mm", "r_inner_mm", "r_outer_mm", "A_passage_mm2", "r_casing_mm",
         "gap_mm", "A_annulus_mm2"],
        _liner_profile_rows(contract)))
    files.append(_write_csv(
        os.path.join(outdir, "effusion_field.csv"),
        ["zone", "row", "z_mm", "r_mm", "n_holes", "d_mm", "phi0_deg",
         "angle_deg", "shape", "cone_expansion"],
        _effusion_rows(contract)))
    files.append(_write_csv(
        os.path.join(outdir, "combustor_bom.csv"),
        ["item", "cantidad", "material", "dimensiones_mm", "nota"],
        _bom_rows(contract)))

    p = os.path.join(outdir, "cfd_boundary_conditions.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(contract["cfd_boundary_conditions"], f, indent=1,
                  ensure_ascii=False, allow_nan=False)
    files.append(p)

    p = os.path.join(outdir, "dfam_report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_clean(dict(
            schema=SCHEMA, arch=contract.get("arch"),
            generator=GENERATOR, seed=SEED,
            voxel_mm=voxel_mm, build_orientation_requested=build_orientation,
            result=res if res is not None else
            dict(note="dfam_checker not available"))),
            f, indent=1, ensure_ascii=False, allow_nan=False)
    files.append(p)

    steps = export_step(contract, outdir, mode=step_mode) if step else []
    if step and not steps:
        warnings.append("CadQuery is not installed: no STEP export "
                        "(OPTIONAL dependency, labelled degradation)")
    files += steps

    for msg in warnings:
        say(f"  ! {msg}")
    return dict(outdir=outdir, files=files, contract=contract, dfam=res,
                step=steps, warnings=warnings,
                notes=list(contract.get("notes") or []),
                schema=SCHEMA, arch=contract.get("arch"),
                n_hole_rows=len(contract.get("hole_rows") or []),
                source="phycb_geometry_5a" + ("" if steps else "(sin_step)"))


# ==========================================================================
# 10. Smoke test — el ancla y las TRES arquitecturas
# ==========================================================================
def main(argv: list | None = None) -> int:
    """Smoke test: contrato del ancla, round-trip, DfAM y las tres
    arquitecturas. Devuelve 0 si todo pasa, 1 si algo falla."""
    argv = list(sys.argv[1:] if argv is None else argv)
    outdir = argv[0] if argv else os.path.join("runs", "_geom_smoke")

    import time                        # noqa: PLC0415
    import physics_core as pc          # noqa: PLC0415
    import dfam_checker as dc          # noqa: PLC0415
    import structures_core as sc       # noqa: PLC0415
    pc.assert_provenance(pc, dc, sc, sys.modules[__name__])
    print(f"[procedencia OK · physics_core NDIM={pc.NDIM} "
          f"PROJECT={pc.PROJECT}]")

    ok = True
    theta = pc.THETA_REF
    rec = pc.evaluate(theta, fidelity=pc.Fidelity.L0)
    struct = sc.evaluate_structural(theta, rec,
                                    material=rec.get("material",
                                                     "Inconel-625"))
    res = generate(theta, rec, outdir, structural=struct, log=print)
    c = res["contract"]
    print(f"\n=== ancla REF_CB10 · {c['arch']} → {outdir} ===")
    for f in res["files"]:
        print(f"  {os.path.basename(f):<28s} {os.path.getsize(f):>8d} B")
    d = c["derived"]
    print(f"  A_ref = {d['A_ref_mm2']:.1f} mm²   A_L = {d['A_L_mm2']:.1f} "
          f"mm²   A_paso = {d['A_pass_mm2']:.1f} mm²   "
          f"A_paso_min = {d['A_pass_min_mm2']:.1f} mm²")
    print(f"  liner: {len(c['liner']['profile'])} puntos  "
          f"z {c['liner']['profile'][0][0]:.1f} → "
          f"{c['liner']['profile'][-1][0]:.1f} mm   t = "
          f"{c['liner']['t_wall_mm']} mm  {c['liner']['material']}   domo "
          f"{c['liner']['dome']['type']} en z = {c['liner']['dome']['z_mm']}")
    print(f"  carcasa: r = {c['casing']['profile'][0][1]:.2f} mm  t = "
          f"{c['casing']['t_wall_mm']} mm  "
          f"{c['casing']['flange_bolt_count']} pernos M"
          f"{int(c['casing']['flange_bolt_d_mm'])}")
    print(f"  swirler: {c['swirler']['n_vanes']} álabes  "
          f"D {c['swirler']['d_tip_mm']:.2f}/{c['swirler']['d_hub_mm']:.2f} mm"
          f"   x{c['swirler']['count']}")
    print(f"  inyectores: {c['injectors']['count']} × "
          f"{c['injectors']['type']}  anillo r = "
          f"{c['injectors']['ring_r_mm']:.2f} mm")
    for row in c["hole_rows"]:
        print(f"    {row['id']:<8s} z={row['z_mm']:8.2f}  r={row['r_mm']:6.2f}"
              f"  n={row['n']:3d}  Ø={row['d_mm']:5.2f}  {row['shape']:<9s}"
              f"  {'con caudal' if row['flow'] else 'sin caudal'}")
    e = c["effusion"]
    print(f"  efusión: Ø{e['d_mm']} mm  P/d {e['pitch_x_over_d']}×"
          f"{e['pitch_y_over_d']}  {e['angle_deg']}°  porosidad "
          f"{e['porosity']:.4f}  {e['n_holes']} orificios "
          f"(record: {e['n_holes_record']})")
    for fp in c["flow_path"]:
        a = fp.get("area_mm2")
        print(f"    · {fp['id']:<18s} dir_z={fp['dir_z']:+.0f} "
              f"giro={fp['turn_deg']:5.1f}°  área="
              f"{('%9.1f mm²' % a) if a is not None else '      n/d'}")
    for n in c["notes"]:
        print(f"    ~ {n}")

    # --- round-trip write→read (T25) --------------------------------------
    with open(os.path.join(outdir, "combustor.json"), encoding="utf-8") as f:
        back = json.load(f)
    same = (back == c)
    print(f"\n  round-trip write→read: {'IDÉNTICO' if same else 'DIVERGE'}")
    ok = ok and same
    for k in ("schema", "arch", "design_vector", "derived", "liner", "casing",
              "swirler", "injectors", "hole_rows", "effusion", "dfam",
              "cfd_boundary_conditions", "run_meta", "_conventions"):
        if k not in back:
            print(f"  ! falta el bloque obligatorio '{k}'")
            ok = False

    # --- DfAM --------------------------------------------------------------
    r = dc.check(c, voxel_mm=VOXEL_DEFAULT_MM)
    print(f"  DfAM: orientación {r['orientation']}  factible: "
          f"{'SÍ' if r['feasible'] else 'NO'}")
    for nm, g in zip(dc.G_NAMES, r["g_dfam"]):
        print(f"    g[{dc.G_NAMES.index(nm)}] {nm:<15s} = {g:+8.4f}  "
              f"{'ok' if g <= 0.0 else 'VIOLA'}")
    for v in r["violations"]:
        print(f"    ! {v}")
    ok = ok and bool(r["feasible"])

    # --- las TRES arquitecturas (T31) --------------------------------------
    print("\n=== las tres arquitecturas ===")
    for arch_id, name in enumerate(_ARCH_NAMES):
        th = np.array(pc.THETA_REF, dtype=float).copy()
        th[0] = float(arch_id)
        rc = pc.evaluate(th, fidelity=pc.Fidelity.L0)
        ct = build_contract(th, rc, None, None)
        prof = ct["liner"]["profile"]
        good, errs = _profile_ok(prof)
        cross = _self_intersects(_meridional_polygon(prof))
        areas = [_passage_area_mm2(p) for p in prof]
        a_ref = ct["derived"]["A_pass_mm2"]
        rr = dc.check(ct, voxel_mm=VOXEL_DEFAULT_MM)
        print(f"  {name:<16s} pts={len(prof):2d}  "
              f"z {prof[0][0]:7.2f}→{prof[-1][0]:7.2f} mm  "
              f"r_in {min(p[1] for p in prof):6.2f}–"
              f"{max(p[1] for p in prof):6.2f}  r_out "
              f"{min(p[2] for p in prof):6.2f}–{max(p[2] for p in prof):6.2f}")
        print(f"                   perfil {'OK ' if good else 'MAL'}  "
              f"auto-intersección: {'SÍ' if cross else 'NO'}  "
              f"A_min={min(areas):8.1f} mm² (ref {a_ref:8.1f})  "
              f"DfAM {rr['orientation']} "
              f"{'factible' if rr['feasible'] else 'INFACTIBLE'}")
        for er in errs:
            print(f"                   ! {er}")
        for v in rr["violations"]:
            print(f"                   ! {v}")
        if (not good) or cross or \
                min(areas) < a_ref * (1.0 - AREA_REL_TOL) - AREA_ABS_TOL_MM2:
            ok = False

    # --- robustez: nunca lanza con entrada degenerada ----------------------
    print("\n=== robustez de build_contract (nunca lanza) ===")
    for nombre, th, rr2 in (("record vacío", pc.THETA_REF, {}),
                            ("record None", pc.THETA_REF, None),
                            ("θ corto", np.zeros(3), {}),
                            ("θ con NaN",
                             np.full(20, float("nan")), {})):
        try:
            cc = build_contract(th, rr2, None, None)
            s = json.dumps(cc, ensure_ascii=False, allow_nan=False)
            bien = cc["validation"]["profile_ok"]
            print(f"  {nombre:<16s} OK  {len(s):6d} caracteres  perfil "
                  f"{'válido' if bien else 'inválido'}")
        except Exception as exc:
            print(f"  {nombre:<16s} LANZÓ {type(exc).__name__}: {exc}")
            ok = False

    # --- las cuatro familias de inyector × los tres combustibles ----------
    print("\n=== gates de inyector y combustible ===")
    for i_inj, nm_inj in enumerate(_INJECTOR_NAMES):
        for i_f, nm_f in enumerate(_FUEL_NAMES):
            th = np.array(pc.THETA_REF, dtype=float).copy()
            th[10], th[15] = float(i_inj), float(i_f)
            cc = build_contract(th, pc.evaluate(th, fidelity=pc.Fidelity.L0),
                                None, None)
            rr = dc.check(cc, voxel_mm=VOXEL_DEFAULT_MM)
            bien = cc["validation"]["profile_ok"]
            print(f"  {nm_inj:<10s} {nm_f:<8s} "
                  f"inyectores={cc['injectors']['count']:2d}  "
                  f"perfil={'OK ' if bien else 'MAL'}"
                  f"  DfAM {'factible' if rr['feasible'] else 'INFACTIBLE'}"
                  f"  ({rr['orientation']})")
            ok = ok and bien

    # --- generate() DEBE fallar ruidosamente con un perfil inválido -------
    print("\n=== generate() con un perfil inválido ===")
    orig = _GEOM_BY_ARCH["can"]
    try:
        def _roto(dm, w, gates, _o=orig):
            g = _o(dm, w, gates)
            g["liner"] = [[0.0, 0.0, 10.0], [50.0, 0.0, 10.0],
                          [10.0, 0.0, 40.0]]        # z retrocede + cruce
            return g
        _GEOM_BY_ARCH["can"] = _roto
        th = np.array(pc.THETA_REF, dtype=float).copy()
        th[0] = 0.0
        generate(th, pc.evaluate(th), os.path.join(outdir, "_roto"),
                 step=False)
        print("  ! NO lanzó: generate() debe fallar con un perfil inválido")
        ok = False
    except ValueError as exc:
        print(f"  ValueError como se espera: {str(exc)[:110]}…")
    finally:
        _GEOM_BY_ARCH["can"] = orig

    # --- claves que consume la capa 5c en C# (PhyCBImport.cs) -------------
    print("\n=== claves que lee PhyCBImport.cs ===")
    faltan = [k for k in ("schema", "arch", "_conventions", "liner", "casing",
                          "swirler", "injectors", "hole_rows", "effusion",
                          "dfam", "run_meta") if k not in c]
    sub = {"liner": ("profile", "t_wall_mm", "material", "fillet_r_mm",
                     "dome"),
           "casing": ("profile", "t_wall_mm", "material", "flange_bolt_count",
                      "flange_bolt_d_mm", "flange_w_mm", "flange_t_mm",
                      "head_plate_t_mm", "boss_h_mm"),
           "swirler": ("type", "n_vanes", "beta_deg", "SN", "d_hub_mm",
                       "d_tip_mm", "vane_t_mm", "z_mm", "chord_mm",
                       "monolithic_with_dome"),
           "injectors": ("type", "count", "ALR", "ring_r_mm", "phi0_deg"),
           "effusion": ("d_mm", "pitch_x_over_d", "pitch_y_over_d",
                        "angle_deg", "shape", "cone_expansion", "staggered",
                        "zones"),
           "dfam": ("build_orientation", "min_wall_mm", "voxel_mm",
                    "coarse_voxel_mm", "overhang_frac_max", "drain_ports")}
    for blk, keys in sub.items():
        faltan += [f"{blk}.{k}" for k in keys if k not in (c.get(blk) or {})]
    for k in ("zone", "id", "z_mm", "r_mm", "n", "d_mm", "Cd", "shape",
              "plunged", "angle_deg", "phi0_deg"):
        if c["hole_rows"] and k not in c["hole_rows"][0]:
            faltan.append(f"hole_rows[].{k}")
    print(f"  {'todas presentes' if not faltan else 'FALTAN: ' + str(faltan)}")
    ok = ok and not faltan

    # --- determinismo y coste (build_contract vive en el lazo del NSGA-II) -
    a = build_contract(theta, rec, struct, None)
    b = build_contract(theta, rec, struct, None)
    igual = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    n_rep = 100
    t0 = time.perf_counter()
    for _ in range(n_rep):
        build_contract(theta, rec, struct, None)
    dt = 1000.0 * (time.perf_counter() - t0) / n_rep
    print(f"\n[determinismo: {'bit-idéntico' if igual else 'DIVERGE'} · "
          f"{dt:.3f} ms/build_contract · {n_rep} repeticiones]")
    ok = ok and igual

    print(f"\n[{'TODO OK' if ok else 'HAY FALLOS'}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
