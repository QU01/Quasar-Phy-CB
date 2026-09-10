"""
QUASAR Phy-CB · dfam_checker.py
===============================
Capa 5a-bis del CEM: VERIFICADOR DfAM (Design for Additive Manufacturing) de
fabricabilidad LPBF, en geometría analítica sobre el contrato
`phycb-combustor-1` — sin voxels, sin dependencias opcionales, <10 ms por
llamada. Es el PRE-FILTRO que corre DENTRO del lazo del optimizador; la
librería C#/PicoGK (capa 5c) re-verifica lo mismo sobre los voxels reales.

Contrato público (IMPLEMENTATION_CONTRACT §5.3 y §8):

    N_DFAM_CONSTRAINTS = 5
    check(contract, build_orientation="z+", voxel_mm=0.5) -> dict
    teardrop_needed(axis_vec, build_dir) -> bool
    drainability_graph(contract, build_dir) -> dict

`check()` devuelve `g_dfam` (5 floats CONTINUOS, g<=0 factible),
`violations` (list[str] en español), `hole_shapes` (id_fila `str` →
"circle"|"teardrop"; alias "0"/"row00"/`id` propio), `drain_ports`
(puertos de drenaje CONCRETOS sugeridos), `orientation`
(la elegida) y `report` (métricas detalladas), más dos claves de cortesía
del estilo de la casa que el contrato no exige: `feasible` (bool) y
`source` (str, con el motivo de degradación entre paréntesis — §3).
**Nunca lanza**: entrada degenerada (no-dict, NaN/inf, ceros, negativos)
⇒ los 5 g a `DEGENERATE_G` (10.0) y el motivo en `violations`.

Orden congelado de `g_dfam` (§5.3):

  | i | nombre           | regla                                            |
  |---|------------------|--------------------------------------------------|
  | 0 | `t_min`          | paredes ≥ `lpbf_t_min_mm(material)`; features     |
  |   |                  | ≥ max(0.6 mm, 2·voxel)                            |
  | 1 | `d_hole_min`     | todo orificio Ø ≥ 0.5 mm                          |
  | 2 | `overhang`       | generatrices ≥ 45° respecto a la placa            |
  | 3 | `drainability`   | ≥2 salidas con línea de visión por cavidad        |
  | 4 | `effusion_band`  | P_x/d, P_y/d, porosidad y ángulo en banda         |

---------------------------------------------------------------------------
CLAVES DEL CONTRATO QUE SE CONSUMEN (todas con lectura DEFENSIVA: cualquier
clave ausente degrada a un valor neutro y se anota en `report["notes"]`;
`geometry_generator.py` todavía no existe, así que este subconjunto se lee
del esquema que emite `geometry_generator.build_contract()`, descrito en
`docs/Quasar_PhyCB_Science.md` §16)
---------------------------------------------------------------------------

    schema                                   (str, informativo)
    arch                                     (str, informativo)
    liner.profile        [[z_mm, r_inner_mm, r_outer_mm], …]
                                             (2 columnas ⇒ r_inner = 0)
    liner.t_wall_mm                          espesor de pared del liner
    liner.material                           p.ej. "Inconel-625"
    liner.dome.type                          "hemispherical" | "flat"
    liner.dome.z_mm                          plano del domo
    liner.double_wall.{gap_mm,z0_mm,z1_mm,t_inner_mm,t_outer_mm}   (opcional)
        · alias: contract.double_wall, liner.cooling(type=="double_wall")
    casing.profile       [[z_mm, r_mm], …]   r = radio INTERIOR de la carcasa
    casing.t_wall_mm, casing.material
    casing.flange_t_mm                       (opcional, feature)
    swirler.vane_t_mm, swirler.d_tip_mm, swirler.d_hub_mm, swirler.type
    injectors.type              "airblast|simplex|vaporizer|micromix"
    injectors.count
    injectors.tube_d_mm / tube_t_mm / tube_z0_mm / tube_z1_mm / bend_deg
        · alias: injectors.vaporizer.{…}, d_tube_mm, tube_id_mm
    injectors.prefilmer_t_mm                 (opcional, feature)
    hole_rows[]          {zone, z_mm, r_mm, n, d_mm, shape, plunged,
                          angle_deg | axis, id}
        · `angle_deg` = inclinación del EJE del orificio desde la dirección
          RADIAL hacia +Z (0 = radial puro, 90 = axial puro). `axis` (3-vector
          [radial, tangencial, axial]) tiene prioridad si está presente.
    effusion.{d_mm, pitch_x_over_d, pitch_y_over_d, angle_deg, shape}
    effusion.zones[]     {z0_mm, z1_mm, rows}
    dfam.{build_orientation, min_wall_mm, voxel_mm}
    dfam.drain_ports[]   {z_mm, d_mm}        puertos YA declarados
    cavities[]           {id, kind, width_mm, z0_mm, z1_mm}  (extensión libre)

---------------------------------------------------------------------------
FÍSICA CODIFICADA (EstadoDelArte §4 completa)
---------------------------------------------------------------------------
§4.1/§4.2  Espesor mínimo dependiente del MATERIAL: al adelgazar la pared
           crecen porosidad y fisuras (Hastelloy-X es el caso severo, de ahí
           su 0.8 mm frente a 0.5 mm de IN625).
§4.3       Ø mínimo de orificio >0.5 mm; ángulo autosoportante ≈45°; eje de
           orificio a menos de 45° de la horizontal del build ⇒ sección en
           GOTA (autosoportante, hay patente específica de efusión
           autosoportante); dilución Ø≥5 mm SIEMPRE gota (el «techo» del
           orificio necesitaría soporte irretirable en un liner cerrado).
§4.4       Depowdering — «la restricción más subestimada»: el polvo atrapado
           en una doble pared sinteriza en el HIP y obstruye de forma
           permanente. Reglas: ≥2 puertos de drenaje CON LÍNEA DE VISIÓN por
           cavidad, Ø de puerto ≥2–3× el ancho del canal, sin cavidades
           ciegas, y **los orificios de efusión NO cuentan como drenaje**.
           El grafo de cavidades con gravedad según la orientación es el
           diferenciador del sistema: ninguna herramienta de diseño
           preliminar lo comprueba.
Ciencia §5.1  Banda imprimible de efusión: P_x/d 5–8, P_y/d 4–8, porosidad
           1–5 %, ángulo 20–30°.

Convenciones (IMPLEMENTATION_CONTRACT §10): SI interno, mm/deg SOLO en la
interfaz y con sufijo en la clave; comentarios en español; sin relojes ni
aleatoriedad dentro del cálculo (el resultado es bit-idéntico para el mismo
contrato).
"""

from __future__ import annotations

import math

import numpy as np

# ==========================================================================
# 0. Constantes de módulo y tunables
# ==========================================================================
SEED = 71                  # semilla de la familia. Este módulo es puramente
#                            determinista (ni RNG ni relojes): se declara por
#                            coherencia con el resto de Phy-CB.

N_DFAM_CONSTRAINTS = 5     # longitud CONGELADA de g_dfam (contrato §5.3)

G_NAMES = ("t_min", "d_hole_min", "overhang", "drainability",
           "effusion_band")

DEGENERATE_G = 10.0        # violación grande y CONTINUA para entrada
#                            degenerada (invariante CONTEXTO §2.2). Es
#                            también el techo de saturación de cualquier g
#                            real: la dominancia de Deb consume magnitudes,
#                            pero magnitudes acotadas.

G_UNVERIFIABLE = 1.0       # violación de «no verificable»: el contrato no
#                            describe la geometría que esta restricción
#                            necesita. Menor que DEGENERATE_G (el contrato
#                            no está corrupto, está incompleto) pero > 0:
#                            la ausencia de datos NUNCA es factibilidad.

#: Qué restricciones deja sin verificar cada bloque ausente del contrato.
#: Índices de G_NAMES: 0 t_min · 1 d_hole_min · 2 overhang ·
#: 3 drainability · 4 effusion_band.
UNVERIFIABLE_G_BY_BLOCK = {
    "liner": (0, 2, 3),      # espesor, voladizos y cavidad liner-carcasa
    "casing": (0, 3),        # espesor de pared fría y cierre de la cavidad
    "hole_rows": (1,),       # Ø mínimo y conversión a gota
}

SOURCE = "dfam_analytic"   # etiqueta base del campo `source` (contrato §3)

# --- Espesores y features -------------------------------------------------
FEATURE_MIN_MM = 0.6       # ⚠ mínimo absoluto de un FEATURE en LPBF
#                            [EstadoDelArte §4.3 + CONTEXTO §8: «grosor
#                            mínimo de cualquier pared/feature: max(0.6 mm,
#                            2·voxel)»]. Recalibrar con la máquina real.
VOXEL_FACTOR = 2.0         # nº de voxels mínimo para RESOLVER un feature
#                            (regla de la familia; el C# clampea a 2·voxel
#                            con warning logueado — ver nota de §1).

# Espesor mínimo LPBF por material [mm]. Procedencia: WAVE0 §5 (tabla
# MATERIALS ampliada) + EstadoDelArte §4.2/§4.3. ⚠ PROVISIONAL: son valores
# de handbook/criterio, no de cupones propios — recalibrar con la ventana de
# proceso de la máquina antes de fabricar. Si `structures_core.py` existe,
# manda SU tabla (fuente única de verdad); esto es solo el respaldo.
LPBF_T_MIN_MM = {
    "inconel-625": 0.5,    # liner micro; «polvo preferido para combustor»
    "hastelloy-x": 0.8,    # ⚠ fisuración en pared fina LPBF (§4.2)
    "haynes-230": 0.6,
    "haynes-282": 0.6,
    "inconel-718": 0.5,    # solo carcasa/estructura fría (HOT_WALL_BANNED)
    "inconel-939": 0.7,
    "ti-6al-4v": 0.4,
    "17-4ph-h900": 0.5,
    "al-2618-t61": 0.5,
    "al-7075-t6": 0.5,
    "316l": 0.4,
    "grcop-42": 0.5,
}
LPBF_T_MIN_DEFAULT_MM = 0.6   # material desconocido ⇒ el mínimo genérico

# --- Orificios ------------------------------------------------------------
D_HOLE_MIN_MM = 0.5        # ✅ EstadoDelArte §4.3: Ø mínimo DMLS en cara
#                            plana, independiente de la profundidad. Coincide
#                            con D_EFF_MIN_MM del contrato §5.1 (g[8] aero).
TEARDROP_ANGLE_DEG = 45.0  # ✅ ángulo autosoportante (§4.3). Eje del orificio
#                            a MENOS de 45° de la horizontal del build ⇒ gota.
TEARDROP_D_DILUTION_MM = 5.0   # ✅ §4.3 regla 3: dilución Ø≥5 mm SIEMPRE
#                                gota (el soporte sería irretirable).
ANGLE_EPS_DEG = 1e-9       # margen anti-ruido en las comparaciones de ángulo

# --- Voladizos ------------------------------------------------------------
OVERHANG_MIN_DEG = 45.0    # ✅ ángulo autosoportante respecto a la placa
OVERHANG_SPAN_TOL_MM = 2.0 # ⚠ vano radial por debajo del cual un tramo casi
#                            horizontal se autosostiene por puenteo (polos de
#                            domo, chaflanes). CRITERIO PROPIO — recalibrar.
PLATE_TOL_MM = 0.5         # tramo a esta distancia de la placa ⇒ apoyado

# --- Drenabilidad de polvo ------------------------------------------------
DRAIN_EXITS_MIN = 2.0      # ✅ §4.4: ≥2 salidas con línea de visión/cavidad
DRAIN_D_OVER_W_MIN = 2.0   # ✅ §4.4: Ø de puerto ≥2–3× el ancho del canal
DRAIN_PORT_D_MIN_MM = 2.0  # ⚠ suelo absoluto del puerto (canales muy finos:
#                            2·0.4 mm sería impracticable de limpiar y de
#                            inspeccionar). CRITERIO PROPIO.
DRAIN_PORT_D_CAP_MM = 8.0  # ⚠ techo del requisito 2–3×: la regla nació para
#                            canales estrechos; por encima de ~8 mm el polvo
#                            fluye libre y exigir 2× de un anillo de 12 mm
#                            (24 mm) no tiene sentido físico. CRITERIO PROPIO
#                            — el punto más recalibrable del módulo.
DRAIN_HOLE_D_MIN_MM = 2.0  # ⚠ Ø individual por debajo del cual el polvo
#                            puentea y el orificio NO drena (por eso la
#                            efusión de 0.5–0.8 mm no cuenta ni aunque
#                            hubiera miles). CRITERIO PROPIO.
DRAIN_N_CREDIT_MAX = 16    # tope de orificios de una fila que acreditan área
#                            equivalente (d_eq = d·√n) — evita que una nube
#                            de agujeros pequeños finja un puerto.
GRAVITY_W_MIN = 0.5        # peso de una salida en el extremo OPUESTO a la
#                            gravedad de la orientación (0.5 = media
#                            eficacia). Interpolación lineal hasta 1.0.
GRAV_BEST_MIN = 0.75       # al menos UNA salida debe alcanzar este peso
#                            efectivo ⇔ estar en la mitad baja de la cavidad.
LOS_FALLOFF_MM = 5.0       # caída lineal de la línea de visión cuando el
#                            puerto queda fuera del rango axial de la cavidad
BEND_LOS_PENALTY = 0.8     # ⚠ penalización de línea de visión de un tubo con
#                            codo ≥90° (vaporizador walking-stick): el polvo
#                            sale, pero no en línea recta. CRITERIO PROPIO —
#                            deliberadamente distinto de GRAV_BEST_MIN para
#                            que un tubo abierto legítimo no quede clavado
#                            en la frontera de factibilidad.
BEND_LOS_DEG = 90.0        # codo a partir del cual se aplica la penalización

# --- Efusión (banda imprimible) — Ciencia §5.1, datos ✅ -------------------
EFF_PX_OVER_D_BAND = (5.0, 8.0)
EFF_PY_OVER_D_BAND = (4.0, 8.0)
EFF_POROSITY_BAND = (0.01, 0.05)
EFF_ANGLE_DEG_BAND = (20.0, 30.0)

BUILD_ORIENTATIONS = ("z+", "z-")   # v1: solo axiales (WAVE3 §1.3 punto 5)


# ==========================================================================
# 1. Utilidades de lectura defensiva
# ==========================================================================
def _num(x) -> float | None:
    """float finito o None. Nunca lanza (bool se rechaza a propósito:
    `True` colándose como espesor sería 1.0 mm silencioso)."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pos(x) -> float | None:
    """float finito y estrictamente positivo, o None."""
    v = _num(x)
    return v if (v is not None and v > 0.0) else None


def _dget(obj, *keys, default=None):
    """`obj[k1][k2]…` tolerante: cualquier eslabón no-dict devuelve default."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _first(obj, keys, default=None):
    """Primer alias presente de `keys` dentro del dict `obj`."""
    if not isinstance(obj, dict):
        return default
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _clip_g(x: float) -> float:
    """Satura g a ±DEGENERATE_G y sanea NaN/inf (g SIEMPRE finito)."""
    v = _num(x)
    if v is None:
        return DEGENERATE_G
    return float(min(max(v, -DEGENERATE_G), DEGENERATE_G))


def _band_violation(x, band: tuple[float, float]) -> float:
    """Violación relativa continua de una banda [lo, hi].

    Devuelve max((lo−x)/|lo|, (x−hi)/|hi|): 0 justo en los bordes, negativa
    dentro de la banda, positiva y creciente fuera. Entrada no finita ⇒
    DEGENERATE_G (nunca lanza)."""
    v = _num(x)
    lo, hi = float(band[0]), float(band[1])
    if v is None:
        return DEGENERATE_G
    v_lo = (lo - v) / abs(lo) if lo != 0.0 else (lo - v)
    v_hi = (v - hi) / abs(hi) if hi != 0.0 else (v - hi)
    return float(max(v_lo, v_hi))


def _norm_material(name) -> str:
    """Clave canónica de material: minúsculas, sin espacios ni guiones
    bajos, con alias frecuentes (`IN625`, `Inconel 625`, `HX`…)."""
    s = str(name or "").strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in s:
        s = s.replace("--", "-")
    alias = {
        "in625": "inconel-625", "in-625": "inconel-625",
        "inconel625": "inconel-625", "ni625": "inconel-625",
        "in718": "inconel-718", "in-718": "inconel-718",
        "inconel718": "inconel-718",
        "hx": "hastelloy-x", "hastelloyx": "hastelloy-x",
        "hastelloy": "hastelloy-x",
        "haynes230": "haynes-230", "haynes282": "haynes-282",
        "ti64": "ti-6al-4v", "tial6v4": "ti-6al-4v",
        "ss316l": "316l", "316-l": "316l",
    }
    return alias.get(s, s)


def lpbf_t_min_mm(material) -> float:
    """Espesor mínimo LPBF del material [mm].

    Fuente única de verdad: `structures_core.MATERIALS[m]["lpbf_t_min_mm"]`
    si el módulo existe (import PEREZOSO — todavía no está escrito en el
    repo). Respaldo: la tabla local `LPBF_T_MIN_MM`. Material desconocido o
    ilegible ⇒ `LPBF_T_MIN_DEFAULT_MM` (nunca lanza)."""
    key = _norm_material(material)
    try:                                   # lazy: structures_core es WAVE0/2
        import structures_core as _sc      # noqa: PLC0415
        for name, props in getattr(_sc, "MATERIALS", {}).items():
            if _norm_material(name) == key:
                v = _pos(props.get("lpbf_t_min_mm"))
                if v is not None:
                    return v
    except Exception:                      # módulo ausente o incompleto
        pass
    return float(LPBF_T_MIN_MM.get(key, LPBF_T_MIN_DEFAULT_MM))


def _build_vec(build_dir) -> np.ndarray:
    """Vector unitario de la dirección de crecimiento del build.

    Acepta "z+", "z-", "z−" (guion Unicode), "+z", "-z", "up"/"down" o un
    3-vector explícito. Cualquier cosa irreconocible ⇒ +Z (el default del
    contrato), nunca una excepción."""
    if isinstance(build_dir, (list, tuple, np.ndarray)):
        v = np.asarray(build_dir, dtype=float).ravel()
        if v.size == 3 and np.all(np.isfinite(v)):
            n = float(np.linalg.norm(v))
            if n > 1e-12:
                return v / n
        return np.array([0.0, 0.0, 1.0])
    s = str(build_dir or "z+").strip().lower().replace("−", "-")
    if s in ("z-", "-z", "down", "z_minus", "zminus"):
        return np.array([0.0, 0.0, -1.0])
    return np.array([0.0, 0.0, 1.0])


def _norm_orientation(build_dir) -> str:
    """Etiqueta canónica "z+"/"z-" de una orientación de build."""
    return "z+" if _build_vec(build_dir)[2] >= 0.0 else "z-"


def _profile_zr(block, key: str = "profile") -> np.ndarray | None:
    """Polilínea (n,3) [z_mm, r_inner_mm, r_outer_mm] a partir de una lista
    de filas de 2 o 3 columnas. Filas de 2 columnas ⇒ [z, r] se interpreta
    como pared única (r_inner = r_outer = r). Devuelve None si no hay al
    menos 2 puntos utilizables — nunca lanza."""
    raw = _dget(block, key) if isinstance(block, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    rows: list[list[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple, np.ndarray)):
            continue
        vals = [_num(x) for x in list(item)[:3]]
        if len(vals) < 2 or any(v is None for v in vals[:2]):
            continue
        if len(vals) == 2 or vals[2] is None:
            rows.append([vals[0], vals[1], vals[1]])
        else:
            rows.append([vals[0], vals[1], vals[2]])
    if len(rows) < 2:
        return None
    return np.asarray(rows, dtype=float)


def _z_range(prof: np.ndarray | None) -> tuple[float, float] | None:
    if prof is None or prof.shape[0] < 2:
        return None
    return float(np.min(prof[:, 0])), float(np.max(prof[:, 0]))


def _interp_col(prof: np.ndarray, z: float, col: int) -> float:
    """Interpolación lineal en z de una columna del perfil (los perfiles
    llegan ordenados por z; se ordena por si acaso — el contrato no lo
    garantiza y la lectura es defensiva)."""
    idx = np.argsort(prof[:, 0], kind="stable")
    return float(np.interp(z, prof[idx, 0], prof[idx, col]))


# ==========================================================================
# 2. Forma de orificio: gota autosoportante  (T27)
# ==========================================================================
def teardrop_needed(axis_vec, build_dir="z+") -> bool:
    """¿El orificio necesita sección en GOTA para ser autosoportante?

    Regla (EstadoDelArte §4.3 + WAVE3 §1.3.2): si el EJE del orificio queda
    a menos de `TEARDROP_ANGLE_DEG` (45°) de la horizontal de la placa de
    construcción, su «techo» es un voladizo que exigiría soporte interno
    irretirable ⇒ la sección debe ser gota/diamante.

    Con φ = ángulo AGUDO entre el eje y la dirección de build (el eje es una
    recta, no un rayo: ±axis describen el mismo orificio), la elevación
    sobre la horizontal es 90°−φ, y la condición es 90°−φ < 45° ⟺ φ > 45°.

    `axis_vec`: 3-vector [radial, tangencial, axial] o 2-vector meridional
    [radial, axial]. `build_dir`: "z+"/"z-" o 3-vector.

    Entrada degenerada (vector nulo, NaN, dimensión rara) ⇒ **True**: la
    gota siempre es imprimible, así que el fallback conservador es pedirla.
    Nunca lanza.
    """
    try:
        b = _build_vec(build_dir)
        a = np.asarray(axis_vec, dtype=float).ravel()
        if a.size == 2:                       # meridional [radial, axial]
            a = np.array([a[0], 0.0, a[1]])
        if a.size != 3 or not np.all(np.isfinite(a)):
            return True
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            return True
        cos_phi = abs(float(np.dot(a / n, b)))
        phi_deg = math.degrees(math.acos(min(cos_phi, 1.0)))
        return bool(phi_deg > TEARDROP_ANGLE_DEG + ANGLE_EPS_DEG)
    except Exception:
        return True


def _row_axis(row: dict) -> np.ndarray:
    """Eje de una fila de orificios en [radial, tangencial, axial].

    Prioridad: `axis` explícito → `angle_deg` (inclinación desde la
    dirección RADIAL hacia +Z: 0 = radial puro, 90 = axial puro) → radial
    puro, que es el caso físico por defecto de los chorros de PZ/SZ/DZ en
    un liner axisimétrico."""
    ax = _first(row, ("axis", "axis_vec", "eje"))
    if isinstance(ax, (list, tuple, np.ndarray)):
        v = np.asarray(ax, dtype=float).ravel()
        if v.size == 2:
            v = np.array([v[0], 0.0, v[1]])
        if (v.size == 3 and np.all(np.isfinite(v))
                and np.linalg.norm(v) > 1e-12):
            return v
    a = _num(_first(row, ("angle_deg", "axis_angle_deg", "tilt_deg")))
    if a is None:
        a = 0.0
    ar = math.radians(a)
    return np.array([math.cos(ar), 0.0, math.sin(ar)])


def _effusion_axis(contract: dict, liner: np.ndarray | None,
                   angle_deg: float) -> np.ndarray:
    """Eje de un orificio de efusión en [radial, tangencial, axial].

    El orificio se inclina `angle_deg` desde la SUPERFICIE del liner (no
    desde la normal): con la pared casi axial y ángulos de 20–30° el eje
    queda a 20–30° del eje de build ⇒ autosoportante sin gota. Es
    exactamente la regla 2 de EstadoDelArte §4.3 («un orificio inclinado
    20–30° no debe generar voladizo <45°»), y por eso el ángulo de efusión
    y la orientación de build están acoplados."""
    t_z, t_r = 1.0, 0.0                    # tangente por defecto: axial
    if liner is not None and liner.shape[0] >= 2:
        z_mid = _effusion_z_mid(contract, liner)
        idx = np.argsort(liner[:, 0], kind="stable")
        p = liner[idx]
        for i in range(p.shape[0] - 1):
            if p[i, 0] - 1e-9 <= z_mid <= p[i + 1, 0] + 1e-9:
                dz = p[i + 1, 0] - p[i, 0]
                dr = p[i + 1, 2] - p[i, 2]
                nrm = math.hypot(dz, dr)
                if nrm > 1e-12:
                    t_z, t_r = dz / nrm, dr / nrm
                break
    n_z, n_r = -t_r, t_z                   # normal saliente (giro +90°)
    ar = math.radians(float(angle_deg))
    az = math.cos(ar) * t_z + math.sin(ar) * n_z
    arad = math.cos(ar) * t_r + math.sin(ar) * n_r
    return np.array([arad, 0.0, az])


def _effusion_z_mid(contract: dict, liner: np.ndarray | None) -> float:
    """z representativo del campo de efusión (centro de la primera zona
    declarada; si no hay zonas, centro del liner)."""
    zones = _dget(contract, "effusion", "zones")
    if isinstance(zones, (list, tuple)):
        for zn in zones:
            if isinstance(zn, dict):
                z0 = _num(_first(zn, ("z0_mm", "z_start_mm")))
                z1 = _num(_first(zn, ("z1_mm", "z_end_mm")))
                if z0 is not None and z1 is not None:
                    return 0.5 * (z0 + z1)
    rng = _z_range(liner)
    return 0.5 * (rng[0] + rng[1]) if rng else 0.0


def _is_effusion_row(row: dict) -> bool:
    """¿La fila es campo de efusión? (nunca cuenta como drenaje, §4.4)."""
    tag = f"{row.get('zone', '')}|{row.get('id', '')}|{row.get('kind', '')}"
    return "eff" in str(tag).lower()


# ==========================================================================
# 3. Grafo de drenabilidad de polvo  (T28) — el diferenciador del sistema
# ==========================================================================
def _cavity_nodes(contract: dict, liner: np.ndarray | None,
                  casing: np.ndarray | None) -> list[dict]:
    """Cavidades del contrato como NODOS del grafo.

    Se modelan (WAVE3 §1.3.4): el anillo liner–carcasa, la doble pared si la
    hay, los tubos de vaporizador y — como sumidero explícito — el interior
    del liner (abierto al swirler y a la tobera). `contract["cavities"]`
    permite declarar cavidades extra sin tocar este módulo."""
    nodes: list[dict] = []

    # --- anillo liner ↔ carcasa (siempre que haya ambos perfiles) ---------
    if liner is not None and casing is not None:
        t_liner = _pos(_dget(contract, "liner", "t_wall_mm")) or 0.0
        rl, rc = _z_range(liner), _z_range(casing)
        z0 = max(rl[0], rc[0])
        z1 = min(rl[1], rc[1])
        if z1 > z0:
            zs = np.linspace(z0, z1, 33)
            w = np.array([_interp_col(casing, z, 1)
                          - _interp_col(liner, z, 2) - t_liner for z in zs])
            w_ok = w[w > 1e-6]
            if w_ok.size:
                nodes.append(dict(
                    id="anillo_liner_carcasa", kind="annulus",
                    z0_mm=z0, z1_mm=z1,
                    width_mm=float(np.median(w_ok)),
                    width_min_mm=float(np.min(w_ok)),
                    accepts=("drain_port", "hole_row"),
                    note="anillo de refrigeración entre liner y carcasa"))

    # --- doble pared (impingement + efusión, patente Honeywell) ----------
    dw = _first(contract, ("double_wall",)) or \
        _dget(contract, "liner", "double_wall")
    if dw is None:
        cool = _dget(contract, "liner", "cooling")
        if isinstance(cool, dict) and "double" in str(
                cool.get("type", "")).lower():
            dw = cool
    if isinstance(dw, dict):
        gap = _pos(_first(dw, ("gap_mm", "h_mm", "channel_mm", "t_gap_mm")))
        rl = _z_range(liner) or (0.0, 1.0)
        z0 = _num(_first(dw, ("z0_mm", "z_start_mm")))
        z1 = _num(_first(dw, ("z1_mm", "z_end_mm")))
        z0 = rl[0] if z0 is None else z0
        z1 = rl[1] if z1 is None else z1
        if z1 < z0:
            z0, z1 = z1, z0
        nodes.append(dict(
            id="doble_pared", kind="double_wall",
            z0_mm=float(z0), z1_mm=float(z1),
            width_mm=float(gap) if gap else 1.0,
            width_min_mm=float(gap) if gap else 1.0,
            accepts=("drain_port",),
            note="cavidad cerrada: la efusión NO cuenta como drenaje "
                 "(EstadoDelArte §4.4)"))

    # --- tubos de vaporizador (walking stick con codo) --------------------
    inj = contract.get("injectors") if isinstance(contract, dict) else None
    inj = inj if isinstance(inj, dict) else {}
    vap = inj.get("vaporizer") if isinstance(inj.get("vaporizer"), dict) \
        else inj
    if "vapor" in str(inj.get("type", "")).lower():
        d_tube = _pos(_first(vap, ("tube_d_mm", "d_tube_mm", "tube_id_mm",
                                   "d_mm")))
        n_tubes = _num(_first(inj, ("count", "n", "n_injectors"))) or 1.0
        rl = _z_range(liner) or (0.0, 1.0)
        z0 = _num(_first(vap, ("tube_z0_mm", "z0_mm"))) or rl[0]
        z1 = _num(_first(vap, ("tube_z1_mm", "z1_mm"))) or \
            min(rl[1], rl[0] + 0.5 * (rl[1] - rl[0]))
        bend = _num(_first(vap, ("bend_deg", "elbow_deg"))) or 180.0
        if z1 < z0:
            z0, z1 = z1, z0
        nodes.append(dict(
            id="tubos_vaporizador", kind="vaporizer_tube",
            z0_mm=float(z0), z1_mm=float(z1),
            width_mm=float(d_tube) if d_tube else 3.0,
            width_min_mm=float(d_tube) if d_tube else 3.0,
            n_instances=int(max(round(n_tubes), 1)),
            bend_deg=float(bend),
            accepts=("drain_port", "open_end"),
            note="tubo abierto por ambos extremos; el codo penaliza la "
                 "línea de visión"))

    # --- interior del liner: SUMIDERO (abierto al swirler y a la salida) --
    if liner is not None:
        rl = _z_range(liner)
        nodes.append(dict(
            id="interior_liner", kind="core",
            z0_mm=rl[0], z1_mm=rl[1],
            width_mm=float(2.0 * np.max(liner[:, 2])),
            width_min_mm=float(2.0 * np.max(liner[:, 2])),
            accepts=("open_end",),
            note="volumen de combustión: abierto al domo/swirler y a la "
                 "tobera — es el sumidero al que drenan los orificios"))

    # --- cavidades declaradas explícitamente (extensión libre) ------------
    extra = contract.get("cavities") if isinstance(contract, dict) else None
    if isinstance(extra, (list, tuple)):
        for i, cav in enumerate(extra):
            if not isinstance(cav, dict):
                continue
            w = _pos(_first(cav, ("width_mm", "gap_mm", "h_mm")))
            rl = _z_range(liner) or (0.0, 1.0)
            z0 = _num(_first(cav, ("z0_mm",)))
            z1 = _num(_first(cav, ("z1_mm",)))
            nodes.append(dict(
                id=str(cav.get("id", f"cavidad_{i}")),
                kind=str(cav.get("kind", "custom")),
                z0_mm=float(rl[0] if z0 is None else z0),
                z1_mm=float(rl[1] if z1 is None else z1),
                width_mm=float(w) if w else 1.0,
                width_min_mm=float(w) if w else 1.0,
                accepts=("drain_port",),
                note="cavidad declarada en el contrato"))
    return nodes


def _d_required_mm(width_mm: float) -> float:
    """Ø de puerto exigido para un canal de ancho `width_mm` [mm].

    Regla ✅ de EstadoDelArte §4.4 (≥2–3× el canal) acotada por criterio
    propio ⚠ entre `DRAIN_PORT_D_MIN_MM` y `DRAIN_PORT_D_CAP_MM`: la regla
    2–3× nació para canales estrechos y, aplicada literalmente a un anillo
    de 12 mm, pediría puertos de 24–36 mm."""
    w = max(float(width_mm), 1e-6)
    return float(min(max(DRAIN_D_OVER_W_MIN * w, DRAIN_PORT_D_MIN_MM),
                     DRAIN_PORT_D_CAP_MM))


def _q_gravity(z_mm: float, node: dict, grav_sign: float) -> float:
    """Peso de gravedad de una salida [GRAVITY_W_MIN, 1].

    `grav_sign` = −1 si el polvo cae hacia −Z (build "z+", placa abajo) y
    +1 si cae hacia +Z (build "z-"). u = 1 en el extremo BAJO de la cavidad
    (donde el polvo se acumula y por tanto la salida es plenamente eficaz),
    u = 0 en el extremo alto (la salida solo respira)."""
    z0, z1 = float(node["z0_mm"]), float(node["z1_mm"])
    span = z1 - z0
    if span <= 1e-9:
        return 1.0
    frac = (float(z_mm) - z0) / span                      # 0 en z0, 1 en z1
    u = (1.0 - frac) if grav_sign < 0.0 else frac
    u = min(max(u, 0.0), 1.0)
    return float(GRAVITY_W_MIN + (1.0 - GRAVITY_W_MIN) * u)


def _q_los(z_mm: float, node: dict) -> float:
    """Calidad de LÍNEA DE VISIÓN de una salida por su posición axial: 1
    dentro del rango de la cavidad, caída lineal hasta 0 en
    `LOS_FALLOFF_MM` fuera (un puerto fuera del rango no abre a la
    cavidad)."""
    z0, z1 = float(node["z0_mm"]), float(node["z1_mm"])
    z = float(z_mm)
    if z0 - 1e-9 <= z <= z1 + 1e-9:
        return 1.0
    d = (z0 - z) if z < z0 else (z - z1)
    return float(max(0.0, 1.0 - d / LOS_FALLOFF_MM))


def drainability_graph(contract: dict, build_dir="z+") -> dict:
    """Grafo de drenabilidad de polvo: nodos (cavidades), aristas (salidas)
    y veredicto continuo.

    Modelo (EstadoDelArte §4.4, WAVE3 §1.3.4):

      * **Nodos** = cavidades cerradas: anillo liner–carcasa, doble pared,
        tubos de vaporizador, cavidades declaradas; más el interior del
        liner como sumidero abierto.
      * **Aristas** = salidas candidatas: `dfam.drain_ports`, filas de
        orificios suficientemente grandes (anillo) y extremos abiertos
        (tubos, interior). Cada arista pesa
        `q = q_size · q_los · q_grav`:
          - `q_size` = min(1, d_eq/d_req) con d_eq = d·√min(n, 16) y d_req
            de `_d_required_mm`; una fila solo acredita si su Ø INDIVIDUAL
            supera `DRAIN_HOLE_D_MIN_MM` (por eso la efusión nunca cuenta,
            además de la exclusión categórica de la regla).
          - `q_los` por posición axial dentro de la cavidad; el codo del
            vaporizador la penaliza a `BEND_LOS_PENALTY`.
          - `q_grav` según la gravedad de la ORIENTACIÓN de build.
      * **Veredicto** por cavidad, con dos términos continuos:
          `g_count = (2 − Σ q_size·q_los) / 2`          (≥2 salidas)
          `g_grav  = GRAV_BEST_MIN − max(q_size·q_los·q_grav)`
                                                        (≥1 salida baja)
        `g_cavidad = max(g_count, g_grav)`; el g del grafo es el peor.

    Devuelve `dict(build_dir, gravity_dir_z, nodes, edges, verdict,
    suggested_ports, notes)`. **Nunca lanza**: sin cavidades detectables el
    veredicto es factible con `g = −1.0` y una nota."""
    try:
        return _drainability_graph_impl(contract, build_dir)
    except Exception as exc:                     # red de seguridad absoluta
        return dict(build_dir=_norm_orientation(build_dir),
                    gravity_dir_z=0.0, nodes=[], edges=[],
                    verdict=dict(ok=False, g=DEGENERATE_G,
                                 worst_node="", n_cavities=0),
                    suggested_ports=[],
                    notes=[f"grafo de drenabilidad no evaluable: {exc}"])


def _drainability_graph_impl(contract: dict, build_dir) -> dict:
    orient = _norm_orientation(build_dir)
    grav_sign = -1.0 if orient == "z+" else 1.0     # dirección de caída
    notes: list[str] = []
    if not isinstance(contract, dict):
        return dict(build_dir=orient, gravity_dir_z=grav_sign, nodes=[],
                    edges=[], verdict=dict(ok=False, g=DEGENERATE_G,
                                           worst_node="", n_cavities=0),
                    suggested_ports=[],
                    notes=["el contrato no es un dict"])

    liner = _profile_zr(contract.get("liner"))
    casing = _profile_zr(contract.get("casing"))
    nodes = _cavity_nodes(contract, liner, casing)

    # --- salidas candidatas del contrato ----------------------------------
    ports = _dget(contract, "dfam", "drain_ports", default=[])
    ports = ports if isinstance(ports, (list, tuple)) else []
    rows = contract.get("hole_rows")
    rows = rows if isinstance(rows, (list, tuple)) else []

    edges: list[dict] = []
    suggested: list[dict] = []
    worst = dict(g=-1.0, node="")
    n_cav = 0

    for node in nodes:
        accepts = node.get("accepts", ())
        d_req = _d_required_mm(node["width_mm"])
        cav_edges: list[dict] = []

        # (a) puertos de drenaje declarados
        if "drain_port" in accepts:
            for j, p in enumerate(ports):
                if not isinstance(p, dict):
                    continue
                z = _num(_first(p, ("z_mm", "z")))
                d = _pos(_first(p, ("d_mm", "diameter_mm", "d")))
                if z is None or d is None:
                    continue
                q_los = _q_los(z, node)
                if q_los <= 0.0:
                    continue
                q_size = min(1.0, d / d_req)
                cav_edges.append(dict(
                    src=node["id"], dst="exterior", kind="drain_port",
                    ref=f"drain_ports[{j}]", z_mm=z, d_mm=d, n=1,
                    d_eq_mm=d, d_req_mm=d_req,
                    q_size=q_size, q_los=q_los,
                    q_grav=_q_gravity(z, node, grav_sign)))

        # (b) filas de orificios (drenan el anillo hacia el interior)
        if "hole_row" in accepts:
            for i, r in enumerate(rows):
                if not isinstance(r, dict):
                    continue
                if _is_effusion_row(r):
                    continue          # ✅ §4.4: la efusión NO cuenta
                z = _num(_first(r, ("z_mm", "z")))
                d = _pos(_first(r, ("d_mm", "diameter_mm")))
                n = _num(_first(r, ("n", "count"))) or 1.0
                if z is None or d is None:
                    continue
                q_los = _q_los(z, node)
                if q_los <= 0.0:
                    continue
                if d < DRAIN_HOLE_D_MIN_MM:
                    edges.append(dict(
                        src=node["id"], dst="interior_liner",
                        kind="hole_row", ref=f"hole_rows[{i}]", z_mm=z,
                        d_mm=d, n=int(n), d_eq_mm=d, d_req_mm=d_req,
                        q_size=0.0, q_los=q_los, q_grav=0.0, q=0.0,
                        counts=False,
                        why=f"Ø {d:.2f} mm < {DRAIN_HOLE_D_MIN_MM} mm: el "
                            "polvo puentea, no drena"))
                    continue
                d_eq = d * math.sqrt(min(max(n, 1.0), DRAIN_N_CREDIT_MAX))
                cav_edges.append(dict(
                    src=node["id"], dst="interior_liner", kind="hole_row",
                    ref=f"hole_rows[{i}]", z_mm=z, d_mm=d, n=int(n),
                    d_eq_mm=d_eq, d_req_mm=d_req,
                    q_size=min(1.0, d_eq / d_req), q_los=q_los,
                    q_grav=_q_gravity(z, node, grav_sign)))

        # (c) extremos abiertos (tubos de vaporizador, interior del liner)
        if "open_end" in accepts:
            q_los_end = BEND_LOS_PENALTY \
                if float(node.get("bend_deg", 0.0)) >= BEND_LOS_DEG else 1.0
            for z_end, tag in ((node["z0_mm"], "extremo_z0"),
                               (node["z1_mm"], "extremo_z1")):
                cav_edges.append(dict(
                    src=node["id"], dst="exterior", kind="open_end",
                    ref=tag, z_mm=float(z_end),
                    d_mm=float(node["width_mm"]), n=1,
                    d_eq_mm=float(node["width_mm"]), d_req_mm=d_req,
                    q_size=1.0,          # la sección completa ES la salida
                    q_los=q_los_end,
                    q_grav=_q_gravity(z_end, node, grav_sign)))

        for e in cav_edges:
            e["q"] = float(e["q_size"] * e["q_los"] * e["q_grav"])
            e["counts"] = bool(e["q_size"] * e["q_los"] > 0.0)
            e.setdefault("why", "")
        edges.extend(cav_edges)

        n_eff = float(sum(e["q_size"] * e["q_los"] for e in cav_edges))
        q_best = float(max((e["q"] for e in cav_edges), default=0.0))
        g_count = (DRAIN_EXITS_MIN - n_eff) / DRAIN_EXITS_MIN
        g_grav = GRAV_BEST_MIN - q_best
        g_cav = max(g_count, g_grav)

        node.update(n_exits=len(cav_edges), n_eff=n_eff, q_best=q_best,
                    d_req_mm=d_req, g_count=float(g_count),
                    g_grav=float(g_grav), g=float(g_cav),
                    ok=bool(g_cav <= 0.0))

        if node["kind"] != "core":
            n_cav += 1
            if g_cav > worst["g"]:
                worst = dict(g=float(g_cav), node=node["id"])
            if g_cav > 0.0:
                suggested.extend(_suggest_ports(node, cav_edges, grav_sign,
                                                casing, liner))

    if n_cav == 0:
        notes.append("no closed cavity was detected in the contract (are "
                     "the liner/casing profiles missing?) — "
                     "drainability not evaluable, reported as feasible")
        worst = dict(g=-1.0, node="")

    return dict(build_dir=orient, gravity_dir_z=grav_sign,
                nodes=nodes, edges=edges,
                verdict=dict(ok=bool(worst["g"] <= 0.0),
                             g=float(worst["g"]),
                             worst_node=worst["node"], n_cavities=n_cav),
                suggested_ports=suggested, notes=notes)


def _suggest_ports(node: dict, cav_edges: list[dict], grav_sign: float,
                   casing: np.ndarray | None,
                   liner: np.ndarray | None) -> list[dict]:
    """Puertos de drenaje CONCRETOS para una cavidad que no drena.

    Se colocan en los extremos axiales de la cavidad: el primero en el
    extremo BAJO según la gravedad de la orientación (drenaje real), el
    segundo en el opuesto (venteo — sin venteo el polvo no fluye). El Ø es
    el exigido por `_d_required_mm`. Bucle ACOTADO: como mucho
    `DRAIN_EXITS_MIN` puertos por cavidad."""
    z0, z1 = float(node["z0_mm"]), float(node["z1_mm"])
    inset = min(0.05 * abs(z1 - z0), 5.0)
    z_low = (z0 + inset) if grav_sign < 0.0 else (z1 - inset)
    z_high = (z1 - inset) if grav_sign < 0.0 else (z0 + inset)
    d_req = float(node.get("d_req_mm") or _d_required_mm(node["width_mm"]))
    n_eff = float(node.get("n_eff", 0.0))
    n_missing = int(min(max(math.ceil(DRAIN_EXITS_MIN - n_eff), 1),
                        int(DRAIN_EXITS_MIN)))
    plan = [(z_low, "drenaje"), (z_high, "venteo")][:n_missing]
    out: list[dict] = []
    for z, role in plan:
        port = dict(z_mm=round(float(z), 2), d_mm=round(d_req, 2),
                    cavity=node["id"], role=role,
                    reason=f"cavidad '{node['id']}' con {n_eff:.2f} salidas "
                           f"efectivas (<{DRAIN_EXITS_MIN:.0f}) — "
                           f"canal de {node['width_mm']:.2f} mm")
        ref = casing if casing is not None else liner
        if ref is not None:
            port["r_mm"] = round(_interp_col(ref, float(z), 1), 2)
        out.append(port)
    return out


# ==========================================================================
# 4. Comprobaciones 1, 2, 3 y 5 (espesor, Ø, voladizos, efusión)
# ==========================================================================
def _check_thickness(contract: dict, voxel_mm: float) -> dict:
    """Regla 1 (WAVE3 §1.3.1): `t_wall ≥ lpbf_t_min_mm(material)` para las
    PAREDES y `≥ max(0.6, 2·voxel)` para los FEATURES.

    Nota de interpretación (declarada, no silenciosa): el suelo `2·voxel`
    se aplica solo a FEATURES. Para las PAREDES se usa el mínimo del
    MATERIAL, coherente con `physics_core.g[7]` y `structures_core.g[3]`
    del contrato §5.1/§5.2 (que usan `lpbf_t_min_mm/t_liner_mm − 1`); si se
    aplicase 2·voxel a las paredes, un liner de 0.8 mm sería infactible a
    voxel 0.5 y el ancla REF_CB10 se caería, además de contradecir a los
    otros dos vectores g. En las paredes la limitación de resolución se
    reporta como AVISO (el C# clampea a 2·voxel con warning logueado —
    CONTEXTO §8), no como violación.

    ⚠ ACOPLAMIENTO CON `voxel_mm`: el suelo de features SÍ depende del
    voxel, así que `check()` debe llamarse con el voxel FINAL (0.3–0.5),
    no con el de borrador: a voxel 0.8 el suelo sube a 1.6 mm y un álabe
    de swirler legítimo de 1.0–1.2 mm se marcaría como infactible. Se
    emite un aviso explícito cuando `voxel_mm > 0.5`."""
    feat_floor = max(FEATURE_MIN_MM, VOXEL_FACTOR * max(voxel_mm, 0.0))
    mat_liner = _dget(contract, "liner", "material", default="Inconel-625")
    mat_casing = _dget(contract, "casing", "material",
                       default="Inconel-718")
    items: list[dict] = []

    def _add(name, value, kind, material=None):
        """Añade una pared/feature. Clave AUSENTE ⇒ se ignora (lectura
        defensiva); clave PRESENTE pero ilegible (NaN/inf/texto) o ≤0 ⇒
        violación degenerada — un espesor NaN jamás puede salir como
        factible."""
        if value is None:
            return
        t_req = lpbf_t_min_mm(material) if kind == "wall" else feat_floor
        t = _num(value)
        if t is None or t <= 0.0:
            items.append(dict(name=name, kind=kind, t_mm=t,
                              t_req_mm=float(t_req),
                              material=(str(material) if material else "-"),
                              g=DEGENERATE_G, degenerate=True))
            return
        items.append(dict(name=name, kind=kind, t_mm=float(t),
                          t_req_mm=float(t_req),
                          material=(str(material) if material else "-"),
                          g=_clip_g(t_req / t - 1.0), degenerate=False))

    _add("liner.t_wall_mm", _dget(contract, "liner", "t_wall_mm"),
         "wall", mat_liner)
    _add("casing.t_wall_mm", _dget(contract, "casing", "t_wall_mm"),
         "wall", mat_casing)
    _add("dfam.min_wall_mm", _dget(contract, "dfam", "min_wall_mm"),
         "wall", mat_liner)
    dw = _first(contract, ("double_wall",)) or \
        _dget(contract, "liner", "double_wall")
    if isinstance(dw, dict):
        _add("double_wall.t_inner_mm", _first(dw, ("t_inner_mm", "t_mm")),
             "wall", mat_liner)
        _add("double_wall.t_outer_mm", dw.get("t_outer_mm"),
             "wall", mat_liner)
    inj = contract.get("injectors") if isinstance(contract, dict) else None
    inj = inj if isinstance(inj, dict) else {}
    vap = inj.get("vaporizer") if isinstance(inj.get("vaporizer"), dict) \
        else inj
    _add("injectors.tube_t_mm", _first(vap, ("tube_t_mm", "t_tube_mm")),
         "wall", mat_liner)
    _add("swirler.vane_t_mm", _dget(contract, "swirler", "vane_t_mm"),
         "feature", mat_liner)
    _add("casing.flange_t_mm", _dget(contract, "casing", "flange_t_mm"),
         "feature", mat_casing)
    _add("injectors.prefilmer_t_mm", _first(vap, ("prefilmer_t_mm",)),
         "feature", mat_liner)

    g = max((it["g"] for it in items), default=-1.0)
    warns = [f"{it['name']} = {it['t_mm']:.2f} mm < 2*voxel = "
             f"{VOXEL_FACTOR * voxel_mm:.2f} mm: layer 5c will clamp it to "
             f"2 voxels (use --voxel <= {it['t_mm'] / 2.0:.2f} to resolve "
             f"it)"
             for it in items
             if not it["degenerate"] and it["t_mm"] < VOXEL_FACTOR * voxel_mm]
    viols = []
    for it in items:
        if it["g"] <= 0.0:
            continue
        if it["degenerate"]:
            viols.append(f"thickness: {it['name']} is not a finite "
                         "positive number (NaN/inf/negative)")
            continue
        viols.append(
            f"thickness: {it['name']} = {it['t_mm']:.3f} mm < "
            f"{it['t_req_mm']:.3f} mm required"
            + (f" by {it['material']} (LPBF)" if it["kind"] == "wall"
               else f" (feature: max({FEATURE_MIN_MM}, 2*voxel))"))
    if voxel_mm > 0.5 and any(it["kind"] == "feature" for it in items):
        warns.append(
            f"draft voxel ({voxel_mm:.2f} mm): the feature floor rises to "
            f"{feat_floor:.2f} mm and may flag legitimate features — "
            "re-check with the final voxel (0.3-0.5 mm)")
    return dict(g=_clip_g(g), items=items, violations=viols,
                warnings=warns, feature_floor_mm=float(feat_floor))


def _check_holes(contract: dict, build_dir, liner: np.ndarray | None,
                 voxel_mm: float) -> dict:
    """Regla 2 (WAVE3 §1.3.2): Ø ≥ 0.5 mm y asignación AUTOMÁTICA de gota.

    Devuelve además `hole_shapes` con tres alias por fila — `"0"` (índice),
    `"row00"` y el `id` propio si el contrato lo trae — para que el
    consumidor pueda indexar como le convenga. **Todas las claves son
    `str`**: el dict viaja en `dfam_report.json` y una clave `int` ni
    sobrevive al round-trip de JSON ni admite `sort_keys=True`. Añade
    `overrides` con las filas cuyo `shape` declarado contradice al exigido
    (corrección automática, NO una violación: la conversión a gota es
    precisamente el trabajo del checker)."""
    rows = contract.get("hole_rows") if isinstance(contract, dict) else None
    rows = rows if isinstance(rows, (list, tuple)) else []
    shapes: dict = {}
    table: list[dict] = []
    overrides: list[str] = []
    viols: list[str] = []
    warns: list[str] = []
    g = -1.0

    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        d = _num(_first(r, ("d_mm", "diameter_mm")))
        zone = str(r.get("zone", "?"))
        axis = _row_axis(r)
        need = teardrop_needed(axis, build_dir)
        big_dz = (zone.lower().startswith("dz") or "dil" in zone.lower()) \
            and d is not None and d >= TEARDROP_D_DILUTION_MM
        shape = "teardrop" if (need or big_dz) else "circle"
        why = ("eje a <45° de la horizontal del build" if need
               else (f"dilución Ø≥{TEARDROP_D_DILUTION_MM:.0f} mm "
                     "(regla dura §4.3)" if big_dz else "autosoportante"))
        rid = f"row{i:02d}"
        shapes[str(i)] = shape         # claves SIEMPRE str: `hole_shapes`
        shapes[rid] = shape            # viaja en dfam_report.json y una
        if r.get("id") is not None:    # clave int no sobrevive al JSON
            shapes[str(r["id"])] = shape
        if d is None or d <= 0.0:
            g = max(g, DEGENERATE_G)
            viols.append(f"holes: hole_rows[{i}] (zone {zone}) has no "
                         "readable diameter, or d <= 0")
        else:
            g = max(g, D_HOLE_MIN_MM / d - 1.0)
            if d < D_HOLE_MIN_MM:
                viols.append(f"holes: hole_rows[{i}] (zone {zone}) "
                             f"d = {d:.3f} mm < {D_HOLE_MIN_MM} mm LPBF "
                             "minimum")
            if d < VOXEL_FACTOR * voxel_mm:
                warns.append(f"hole_rows[{i}] d = {d:.2f} mm < 2*voxel: "
                             "it is not resolved by the voxelisation")
        declared = str(r.get("shape", "")).lower()
        if declared in ("circle", "teardrop") and declared != shape:
            overrides.append(f"hole_rows[{i}] ({zone}): '{declared}' → "
                             f"'{shape}' ({why})")
        table.append(dict(index=i, id=rid, zone=zone, d_mm=d,
                          z_mm=_num(_first(r, ("z_mm", "z"))),
                          n=_num(_first(r, ("n", "count"))),
                          axis=[float(a) for a in axis],
                          shape=shape, reason=why))

    # --- campo de efusión: mismo Ø mínimo, forma por el ángulo real -------
    eff = contract.get("effusion") if isinstance(contract, dict) else None
    if isinstance(eff, dict):
        d_eff = _num(_first(eff, ("d_mm", "d_eff_mm")))
        a_eff = _num(_first(eff, ("angle_deg", "alpha_deg")))
        a_eff = 25.0 if a_eff is None else a_eff
        axis = _effusion_axis(contract, liner, a_eff)
        shape = "teardrop" if teardrop_needed(axis, build_dir) else "circle"
        shapes["effusion"] = shape
        if d_eff is None or d_eff <= 0.0:
            g = max(g, DEGENERATE_G)
            viols.append("holes: effusion.d_mm missing or <= 0")
        else:
            g = max(g, D_HOLE_MIN_MM / d_eff - 1.0)
            if d_eff < D_HOLE_MIN_MM:
                viols.append(f"holes: effusion d = {d_eff:.3f} mm < "
                             f"{D_HOLE_MIN_MM} mm LPBF minimum "
                             "(EstadoDelArte §4.3)")
        table.append(dict(index=-1, id="effusion", zone="effusion",
                          d_mm=d_eff, z_mm=None, n=None,
                          axis=[float(a) for a in axis], shape=shape,
                          reason="effusion angle vs build direction"))

    return dict(g=_clip_g(g), hole_shapes=shapes, table=table,
                violations=viols, warnings=warns, overrides=overrides)


def _segments(prof: np.ndarray, col: int, part: str) -> list[dict]:
    """Tramos de una generatriz (z, r) como lista de dicts."""
    idx = np.argsort(prof[:, 0], kind="stable")
    p = prof[idx]
    out = []
    for i in range(p.shape[0] - 1):
        out.append(dict(part=part, z0_mm=float(p[i, 0]),
                        z1_mm=float(p[i + 1, 0]), r0_mm=float(p[i, col]),
                        r1_mm=float(p[i + 1, col])))
    return out


def _check_overhangs(contract: dict, build_dir, liner: np.ndarray | None,
                     casing: np.ndarray | None) -> dict:
    """Regla 3 (WAVE3 §1.3.3): ángulo de las generatrices ≥ 45° respecto a
    la placa, salvo tramos auto-soportados.

    El liner/carcasa son CÁSCARAS delgadas de revolución: su cara inferior
    es down-skin en todo tramo, así que la inclinación del tramo respecto a
    la placa (α = atan|Δz|/|Δr|) gobierna, independientemente de a qué lado
    esté el material. Exenciones (la revolución continua alrededor de Z es
    auto-soportada, WAVE3 §1.3.3):
      * tramo APOYADO en la placa (el extremo del build correspondiente):
        de aquí sale el acoplamiento fuerte con la orientación — un domo
        plano en z_min es gratis en "z+" y un voladizo total en "z-";
      * vano radial |Δr| ≤ `OVERHANG_SPAN_TOL_MM` (polos de domo,
        chaflanes): el puenteo lo resuelve;
      * marcado explícito `liner.self_supported` / `casing.self_supported`
        o `dfam.assume_revolution_self_supported`.
    """
    orient = _norm_orientation(build_dir)
    segs: list[dict] = []
    if liner is not None:
        segs += _segments(liner, 2, "liner.outer")
        if float(np.max(liner[:, 1])) > 0.1:      # anular: hay pared interna
            segs += _segments(liner, 1, "liner.inner")
    if casing is not None:
        segs += _segments(casing, 1, "casing")
    dome_t = str(_dget(contract, "liner", "dome", "type", default="")).lower()
    z_dome = _num(_dget(contract, "liner", "dome", "z_mm"))
    if dome_t.startswith("flat") and z_dome is not None and liner is not None:
        segs.append(dict(part="liner.dome(flat)", z0_mm=z_dome,
                         z1_mm=z_dome, r0_mm=0.0,
                         r1_mm=_interp_col(liner, z_dome, 2)))

    if not segs:
        return dict(g=-1.0, orientation=orient, segments=[],
                    violations=[], warnings=[],
                    note="sin polilíneas de liner/carcasa: voladizos no "
                         "evaluables")

    z_all = [s["z0_mm"] for s in segs] + [s["z1_mm"] for s in segs]
    z_plate = min(z_all) if orient == "z+" else max(z_all)
    forced = bool(_dget(contract, "dfam",
                        "assume_revolution_self_supported", default=False)) \
        or bool(_dget(contract, "liner", "self_supported", default=False))

    g = -1.0
    viols: list[str] = []
    for s in segs:
        dz = abs(s["z1_mm"] - s["z0_mm"])
        dr = abs(s["r1_mm"] - s["r0_mm"])
        if dz < 1e-9 and dr < 1e-9:
            continue
        alpha = math.degrees(math.atan2(dz, dr))     # 90° = pared vertical
        on_plate = (abs(s["z0_mm"] - z_plate) <= PLATE_TOL_MM
                    and abs(s["z1_mm"] - z_plate) <= PLATE_TOL_MM)
        short = dr <= OVERHANG_SPAN_TOL_MM
        exempt = on_plate or short or forced
        reason = ("apoyado en la placa" if on_plate else
                  ("vano radial corto (puenteo)" if short else
                   ("marcado auto-soportado" if forced else "")))
        g_seg = (OVERHANG_MIN_DEG - alpha) / OVERHANG_MIN_DEG
        s.update(angle_deg=float(alpha), exempt=bool(exempt),
                 reason=reason, g=0.0 if exempt else float(g_seg))
        if not exempt:
            g = max(g, g_seg)
            if g_seg > 0.0:
                viols.append(
                    f"overhang: {s['part']} z {s['z0_mm']:.1f}->"
                    f"{s['z1_mm']:.1f} mm at {alpha:.1f} deg < "
                    f"{OVERHANG_MIN_DEG:.0f} deg (radial span {dr:.1f} mm, "
                    f"build {orient})")
    return dict(g=_clip_g(g), orientation=orient, segments=segs,
                z_plate_mm=float(z_plate), violations=viols, warnings=[])


def _check_effusion(contract: dict) -> dict:
    """Regla 5 (WAVE3 §1.3.5 / Ciencia §5.1): paso y porosidad del campo de
    efusión dentro de la banda imprimible y aerotérmicamente válida.

    Porosidad geométrica σ = (π/4)·d² / (P_x·P_y) = (π/4)/((P_x/d)(P_y/d)):
    con la banda de pasos (5–8, 4–8) da 1.2–3.9 %, dentro de la banda
    experimental 1–5 % ✅ — las tres restricciones son consistentes."""
    eff = contract.get("effusion") if isinstance(contract, dict) else None
    if not isinstance(eff, dict):
        return dict(g=-1.0, metrics={}, violations=[], warnings=[],
                    note="sin bloque `effusion`: banda no aplicable")
    px = _num(_first(eff, ("pitch_x_over_d", "px_over_d", "Px_over_d")))
    py = _num(_first(eff, ("pitch_y_over_d", "py_over_d", "Py_over_d")))
    ang = _num(_first(eff, ("angle_deg", "alpha_deg")))
    checks = []
    if px is not None:
        checks.append(("P_x/d", px, EFF_PX_OVER_D_BAND,
                       _band_violation(px, EFF_PX_OVER_D_BAND)))
    if py is not None:
        checks.append(("P_y/d", py, EFF_PY_OVER_D_BAND,
                       _band_violation(py, EFF_PY_OVER_D_BAND)))
    if ang is not None:
        checks.append(("ángulo [deg]", ang, EFF_ANGLE_DEG_BAND,
                       _band_violation(ang, EFF_ANGLE_DEG_BAND)))
    poro = None
    if px is not None and py is not None and px > 0.0 and py > 0.0:
        poro = (math.pi / 4.0) / (px * py)
        checks.append(("porosidad", poro, EFF_POROSITY_BAND,
                       _band_violation(poro, EFF_POROSITY_BAND)))
    if not checks:
        return dict(g=-1.0, metrics={}, violations=[],
                    warnings=["bloque `effusion` sin pasos ni ángulo: "
                              "band not evaluated"],
                    note="insufficient data")
    g = max(c[3] for c in checks)
    viols = [f"effusion: {n} = {v:.4g} outside the printable band "
             f"[{b[0]:g}, {b[1]:g}]" for n, v, b, gv in checks if gv > 0.0]
    return dict(g=_clip_g(g),
                metrics=dict(pitch_x_over_d=px, pitch_y_over_d=py,
                             angle_deg=ang, porosity=poro,
                             bands=dict(pitch_x_over_d=EFF_PX_OVER_D_BAND,
                                        pitch_y_over_d=EFF_PY_OVER_D_BAND,
                                        angle_deg=EFF_ANGLE_DEG_BAND,
                                        porosity=EFF_POROSITY_BAND)),
                checks=[dict(name=n, value=v, band=list(b), g=float(gv))
                        for n, v, b, gv in checks],
                violations=viols, warnings=[])


# ==========================================================================
# 5. Evaluación de UNA orientación y API pública `check`
# ==========================================================================
def _check_one_orientation(contract: dict, build_dir: str,
                           voxel_mm: float) -> dict:
    """Las 5 reglas para una orientación de build concreta."""
    liner = _profile_zr(contract.get("liner"))
    casing = _profile_zr(contract.get("casing"))
    th = _check_thickness(contract, voxel_mm)
    ho = _check_holes(contract, build_dir, liner, voxel_mm)
    ov = _check_overhangs(contract, build_dir, liner, casing)
    dr = drainability_graph(contract, build_dir)
    ef = _check_effusion(contract)

    g_dfam = [_clip_g(th["g"]), _clip_g(ho["g"]), _clip_g(ov["g"]),
              _clip_g(dr["verdict"]["g"]), _clip_g(ef["g"])]
    viols = list(th["violations"]) + list(ho["violations"]) + \
        list(ov["violations"]) + list(ef["violations"])
    for node in dr["nodes"]:
        if node["kind"] != "core" and node.get("g", -1.0) > 0.0:
            viols.append(
                f"drainability: cavity '{node['id']}' (channel "
                f"{node['width_mm']:.2f} mm, z {node['z0_mm']:.1f}->"
                f"{node['z1_mm']:.1f} mm) has {node['n_eff']:.2f} effective "
                f"line-of-sight exits (minimum {DRAIN_EXITS_MIN:.0f}) and a "
                f"best gravity weight of {node['q_best']:.2f} (minimum "
                f"{GRAV_BEST_MIN:.2f}) in build {dr['build_dir']}")
    warns = list(th["warnings"]) + list(ho["warnings"]) + \
        list(ov["warnings"]) + list(ef["warnings"])
    notes = list(dr["notes"])
    for blk in (ov, ef):
        if blk.get("note"):
            notes.append(blk["note"])

    total = float(sum(max(g, 0.0) for g in g_dfam))
    return dict(orientation=_norm_orientation(build_dir), g_dfam=g_dfam,
                total_violation=total, sum_g=float(sum(g_dfam)),
                violations=viols, warnings=warns, notes=notes,
                hole_shapes=ho["hole_shapes"], shape_overrides=ho["overrides"],
                drain_ports=dr["suggested_ports"],
                thickness=th, holes=ho, overhangs=ov, drainability=dr,
                effusion=ef)


def _degenerate_result(reason: str, orientation: str) -> dict:
    """Resultado con violación máxima CONTINUA (invariante CONTEXTO §2.2):
    los 5 g a `DEGENERATE_G`, sin excepción y sin claves ausentes."""
    return dict(
        g_dfam=[DEGENERATE_G] * N_DFAM_CONSTRAINTS,
        violations=[f"degenerate input: {reason}"],
        hole_shapes={}, drain_ports=[], orientation=orientation,
        feasible=False, source=f"{SOURCE}(entrada_degenerada)",
        report=dict(constraint_names=list(G_NAMES), reason=reason,
                    warnings=[], notes=[reason], orientation_trials=[],
                    params=dict(voxel_mm=None)))


def check(contract: dict, build_orientation: str = "z+",
          voxel_mm: float = 0.5) -> dict:
    """Verificador DfAM completo del contrato `phycb-combustor-1`.

    Prueba las DOS orientaciones axiales {z+, z-} (WAVE3 §1.3.5), elige la
    que MINIMIZA la violación total Σ max(g_i, 0) — con desempate por
    Σ g_i y, si aún empatan, por la orientación pedida en
    `build_orientation` — y devuelve sus resultados.

    Parámetros
    ----------
    contract : dict
        Contrato de geometría (esquema en el docstring del módulo). Lectura
        totalmente defensiva: las claves ausentes degradan a neutro.
    build_orientation : str
        Preferencia/desempate ("z+" por defecto). NO fija la orientación:
        el contrato §5.3 exige que la orientación sea un grado de libertad.
    voxel_mm : float
        Voxel de la capa 5c (0.5 iteración, 0.8 borrador, 0.3–0.4 final).
        Fija el suelo de los FEATURES en max(0.6, 2·voxel).

    Devuelve
    --------
    dict con `g_dfam` (5 floats, g≤0 factible), `violations` (español),
    `hole_shapes`, `drain_ports` (sugeridos), `orientation`, `feasible`,
    `source` y `report`. **Nunca lanza.**
    """
    pref = _norm_orientation(build_orientation)
    try:
        if not isinstance(contract, dict) or not contract:
            return _degenerate_result(
                "el contrato no es un dict no vacío", pref)
        v = _num(voxel_mm)
        if v is None or v <= 0.0:
            v = 0.5
        v = float(min(max(v, 0.05), 5.0))       # voxel físicamente sensato

        trials = []
        for orient in BUILD_ORIENTATIONS:
            trials.append(_check_one_orientation(contract, orient, v))
        # Selección determinista: menor violación total → menor Σg →
        # preferencia del llamante → orden congelado de BUILD_ORIENTATIONS.
        best = min(trials, key=lambda t: (
            round(t["total_violation"], 12), round(t["sum_g"], 12),
            0 if t["orientation"] == pref else 1,
            BUILD_ORIENTATIONS.index(t["orientation"])))

        source = SOURCE
        missing = [k for k in ("liner", "casing", "hole_rows")
                   if not contract.get(k)]
        if missing:
            source = f"{SOURCE}(contrato_incompleto:{'+'.join(missing)})"
            # AUSENCIA DE GEOMETRÍA ≠ GEOMETRÍA IMPRIMIBLE (test T26).
            # Etiquetar el `source` no basta: si `g_dfam` sale neutro, un
            # contrato sin liner, sin carcasa y sin orificios se declara
            # FACTIBLE porque no hay nada que violar — y el optimizador
            # premiaría el vacío. Cada bloque ausente marca como NO
            # VERIFICABLE las restricciones que ese bloque alimenta.
            g = list(best["g_dfam"])
            for key in missing:
                for i in UNVERIFIABLE_G_BY_BLOCK[key]:
                    g[i] = max(g[i], G_UNVERIFIABLE)
            best = dict(best, g_dfam=g,
                        total_violation=float(sum(max(x, 0.0) for x in g)),
                        sum_g=float(sum(g)),
                        notes=list(best["notes"]) + [
                            f"bloque(s) ausente(s) del contrato: "
                            f"{', '.join(missing)} — las restricciones "
                            f"correspondientes se marcan NO VERIFICABLES "
                            f"(g = {G_UNVERIFIABLE:+.1f}), no factibles"],
                        violations=list(best["violations"]) + [
                            f"incomplete contract: {k} is missing — "
                            f"what is not described cannot be declared "
                            f"printable"
                            for k in missing])

        notes = list(best["notes"])
        v_ctr = _pos(_dget(contract, "dfam", "voxel_mm"))
        if v_ctr is not None and abs(v_ctr - v) > 1e-9:
            notes.append(f"the contract declares dfam.voxel_mm = "
                         f"{v_ctr:g} mm but it was verified with {v:g} mm "
                         "(the `check` argument wins)")
        o_ctr = _dget(contract, "dfam", "build_orientation")
        if (o_ctr is not None
                and _norm_orientation(o_ctr) != best["orientation"]):
            notes.append(f"the contract declares build_orientation = "
                         f"'{_norm_orientation(o_ctr)}' but '"
                         f"{best['orientation']}' minimises the total "
                         "violation — update it in "
                         "dfam.build_orientation")

        report = dict(
            constraint_names=list(G_NAMES),
            schema=str(contract.get("schema", "")),
            arch=str(contract.get("arch", "")),
            orientation=best["orientation"],
            orientation_requested=pref,
            orientation_trials=[
                dict(orientation=t["orientation"],
                     g_dfam=list(t["g_dfam"]),
                     total_violation=t["total_violation"],
                     n_violations=len(t["violations"])) for t in trials],
            thickness=best["thickness"],
            holes=best["holes"],
            overhangs=best["overhangs"],
            drainability=best["drainability"],
            effusion=best["effusion"],
            shape_overrides=best["shape_overrides"],
            drain_ports_declared=_dget(contract, "dfam", "drain_ports",
                                       default=[]),
            warnings=best["warnings"],
            notes=notes,
            params=dict(voxel_mm=v, feature_floor_mm=max(
                FEATURE_MIN_MM, VOXEL_FACTOR * v),
                d_hole_min_mm=D_HOLE_MIN_MM,
                overhang_min_deg=OVERHANG_MIN_DEG,
                drain_exits_min=DRAIN_EXITS_MIN,
                drain_d_over_w_min=DRAIN_D_OVER_W_MIN,
                teardrop_angle_deg=TEARDROP_ANGLE_DEG,
                seed=SEED),
        )
        return dict(g_dfam=[float(g) for g in best["g_dfam"]],
                    violations=list(best["violations"]),
                    hole_shapes=dict(best["hole_shapes"]),
                    drain_ports=list(best["drain_ports"]),
                    orientation=best["orientation"],
                    feasible=bool(all(g <= 0.0 for g in best["g_dfam"])),
                    source=source, report=report)
    except Exception as exc:                     # red de seguridad absoluta
        return _degenerate_result(f"excepción interna {type(exc).__name__}: "
                                  f"{exc}", pref)


# ==========================================================================
# 6. Smoke test — can de 10 kW, caso limpio y caso infactible
# ==========================================================================
def _demo_contract_clean() -> dict:
    """Can de 10 kW imprimible (clase 5–50 kW, EstadoDelArte §7).

    Domo con cara plana anular en z=0 (gratis en build "z+", voladizo total
    en "z-": es lo que decide la orientación), cono de domo a 57°, cuerpo
    cilíndrico y contracción de salida a 62°. Efusión 0.6 mm / 25° / P_x/d 6
    / P_y/d 5 (porosidad 2.6 %), liner IN625 de 0.8 mm."""
    return {
        "schema": "phycb-combustor-1",
        "arch": "can",
        "_conventions": ("z en mm sobre el eje, gas a +Z; "
                         "hole_rows[].angle_deg = inclinación del eje del "
                         "orificio desde la dirección radial hacia +Z"),
        "liner": {
            "profile": [[0.0, 0.0, 12.0], [0.0, 0.0, 18.0],
                        [26.0, 0.0, 35.0], [95.0, 0.0, 35.0],
                        [120.0, 0.0, 22.0]],
            "t_wall_mm": 0.8, "material": "Inconel-625",
            "dome": {"type": "hemispherical", "z_mm": 0.0},
        },
        "casing": {
            "profile": [[0.0, 48.0], [95.0, 48.0], [120.0, 30.0]],
            "t_wall_mm": 1.2, "material": "Inconel-718",
            "flange_bolt_count": 12, "flange_bolt_d_mm": 6.0,
            "flange_t_mm": 3.0,
        },
        "swirler": {"type": "axial", "n_vanes": 8, "beta_deg": 55.0,
                    "SN": 0.78, "d_hub_mm": 10.0, "d_tip_mm": 24.0,
                    "vane_t_mm": 1.2},
        "injectors": {"type": "airblast", "count": 1, "ALR": 4.0,
                      "prefilmer_t_mm": 1.2},
        "hole_rows": [
            {"zone": "pz", "z_mm": 6.0, "r_mm": 20.0, "n": 1, "d_mm": 3.0,
             "Cd": 0.62, "angle_deg": 80.0, "shape": "circle",
             "id": "bujia"},
            {"zone": "pz", "z_mm": 32.0, "r_mm": 35.0, "n": 8, "d_mm": 4.0,
             "Cd": 0.62, "shape": "circle"},
            {"zone": "sz", "z_mm": 55.0, "r_mm": 35.0, "n": 8, "d_mm": 4.5,
             "Cd": 0.62, "shape": "circle"},
            {"zone": "dz", "z_mm": 78.0, "r_mm": 35.0, "n": 10, "d_mm": 6.5,
             "Cd": 0.62, "shape": "circle"},
        ],
        "effusion": {"d_mm": 0.6, "pitch_x_over_d": 6.0,
                     "pitch_y_over_d": 5.0, "angle_deg": 25.0,
                     "shape": "conical",
                     "zones": [{"z0_mm": 30.0, "z1_mm": 92.0, "rows": 12}]},
        "dfam": {"build_orientation": "z+", "min_wall_mm": 0.8,
                 "voxel_mm": 0.5,
                 "drain_ports": [{"z_mm": 8.0, "d_mm": 5.0},
                                 {"z_mm": 112.0, "d_mm": 5.0}]},
        "run_meta": {"schema": "phycb-combustor-1", "seed": SEED,
                     "source": "demo", "fidelity": "L0"},
    }


def _demo_contract_bad() -> dict:
    """El mismo can, deliberadamente INFACTIBLE — ejercita las 5 reglas:
    pared de 0.3 mm (g0), efusión de 0.4 mm (g1), cono de domo a 20° (g2),
    doble pared sin puertos de drenaje (g3) y paso/ángulo de efusión fuera
    de banda (g4)."""
    c = _demo_contract_clean()
    c["liner"] = dict(c["liner"])
    c["liner"]["t_wall_mm"] = 0.3                       # g0
    c["liner"]["profile"] = [[0.0, 0.0, 12.0], [0.0, 0.0, 18.0],
                             [9.0, 0.0, 35.0], [95.0, 0.0, 35.0],
                             [120.0, 0.0, 22.0]]        # g2: cono a 20.7°
    c["liner"]["double_wall"] = {"gap_mm": 1.2, "z0_mm": 30.0,
                                 "z1_mm": 92.0, "t_inner_mm": 0.3,
                                 "t_outer_mm": 0.4}     # g3
    c["effusion"] = {"d_mm": 0.4, "pitch_x_over_d": 3.0,   # g1 + g4
                     "pitch_y_over_d": 3.0, "angle_deg": 45.0,
                     "shape": "cyl",
                     "zones": [{"z0_mm": 30.0, "z1_mm": 92.0, "rows": 12}]}
    c["dfam"] = dict(c["dfam"])
    c["dfam"]["min_wall_mm"] = 0.3
    c["dfam"]["drain_ports"] = []                       # g3: sin puertos
    return c


if __name__ == "__main__":
    import sys
    import time

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def _tabla(nombre: str, res: dict) -> None:
        print(f"\n=== {nombre} ===")
        print(f"  orientación elegida : {res['orientation']}   "
              f"factible: {'SÍ' if res['feasible'] else 'NO'}   "
              f"source: {res['source']}")
        for n, g in zip(G_NAMES, res["g_dfam"]):
            print(f"    g[{G_NAMES.index(n)}] {n:<15s} = {g:+8.4f}  "
                  f"{'ok' if g <= 0.0 else 'VIOLA'}")
        for t in res["report"]["orientation_trials"]:
            print(f"    · ensayo {t['orientation']}: "
                  f"violación total {t['total_violation']:.4f} "
                  f"({t['n_violations']} avisos)")
        for v in res["violations"]:
            print(f"    ! {v}")
        for w in res["report"]["warnings"]:
            print(f"    ~ {w}")
        for o in res["report"]["shape_overrides"]:
            print(f"    → forma corregida: {o}")
        shapes = {k: v for k, v in res["hole_shapes"].items()
                  if isinstance(k, str)}
        print(f"    formas: {shapes}")
        gr = res["report"]["drainability"]
        print(f"    grafo: {len(gr['nodes'])} nodos / "
              f"{len(gr['edges'])} aristas · veredicto "
              f"g={gr['verdict']['g']:+.3f} "
              f"(peor: {gr['verdict']['worst_node'] or '-'})")
        for nd in gr["nodes"]:
            print(f"      - {nd['id']:<22s} canal {nd['width_mm']:6.2f} mm  "
                  f"salidas={nd['n_exits']:2d}  n_eff={nd['n_eff']:5.2f}  "
                  f"q_best={nd['q_best']:.2f}  g={nd['g']:+.3f}")
        for p in res["drain_ports"]:
            print(f"    + puerto sugerido: z={p['z_mm']} mm  "
                  f"Ø={p['d_mm']} mm  [{p['role']} de {p['cavity']}]")

    limpio = _demo_contract_clean()
    malo = _demo_contract_bad()

    r1 = check(limpio, build_orientation="z+", voxel_mm=0.5)
    r2 = check(malo, build_orientation="z+", voxel_mm=0.5)
    _tabla("can 10 kW LIMPIO", r1)
    _tabla("can 10 kW INFACTIBLE", r2)

    # --- funciones auxiliares públicas (T27/T28 las usan directamente) ----
    print("\n=== teardrop_needed (T27) ===")
    for nombre, eje in (("radial (dilución)", [1.0, 0.0, 0.0]),
                        ("axial (vertical)", [0.0, 0.0, 1.0]),
                        ("efusión 25°", [0.423, 0.0, 0.906]),
                        ("a 45° exactos", [1.0, 0.0, 1.0]),
                        ("degenerado (0,0,0)", [0.0, 0.0, 0.0])):
        print(f"  {nombre:<20s} z+ → "
              f"{'teardrop' if teardrop_needed(eje, 'z+') else 'circle'}")

    print("\n=== robustez: entradas degeneradas (nunca lanza) ===")
    for nombre, ctr in (("None", None), ("dict vacío", {}),
                        ("lista", [1, 2, 3]),
                        ("NaN en el espesor",
                         {"liner": {"t_wall_mm": float("nan"),
                                    "material": "Inconel-625"}}),
                        ("espesor negativo",
                         {"liner": {"t_wall_mm": -0.5}}),
                        ("basura",
                         {"liner": {"profile": "no-es-una-lista"},
                          "hole_rows": [None, 7, {"d_mm": "x"}]})):
        r = check(ctr)
        print(f"  {nombre:<20s} g={[round(g, 3) for g in r['g_dfam']]}  "
              f"factible={r['feasible']}  {r['source']}")

    # --- coste (debe quedar MUY por debajo de 10 ms — lazo del optimizador)
    n_rep = 200
    t0 = time.perf_counter()
    for _ in range(n_rep):
        check(limpio, voxel_mm=0.5)
    dt_ms = 1000.0 * (time.perf_counter() - t0) / n_rep
    print(f"\n[{dt_ms:.3f} ms/check · presupuesto 10 ms · "
          f"{n_rep} repeticiones]")

    # --- determinismo (contrato §10.8) -----------------------------------
    a = check(limpio, voxel_mm=0.5)
    b = check(limpio, voxel_mm=0.5)
    igual = (a["g_dfam"] == b["g_dfam"]
             and a["orientation"] == b["orientation"]
             and a["violations"] == b["violations"])
    print(f"[determinismo: {'bit-idéntico' if igual else 'DIVERGE'}]")
