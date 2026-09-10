"""QUASAR Phy-CB — Capa 0: pipeline de datos y anclas de correlación.

Política Phy-* (heredada de Phy-HL/Phy-Wing vía Phy-AC): SOLO agregados pequeños
con whitelist hard-coded y manifiesto SHA-256 — nunca tiers masivos de campos.

Para combustores de microturbina **no existe ningún dataset público agregado**
(el estado del arte lo confirma: `docs/Quasar_PhyCB_EstadoDelArte.md` §5 — «no
existe ninguna implementación de código abierto» del método preliminar). Lo que
sí existe son coeficientes de correlación dispersos en literatura. La regla del
contrato (`docs/IMPLEMENTATION_CONTRACT.md` §7) es que **todo coeficiente vive
aquí, nunca hard-coded en `physics_core`**, y que cada tabla declara su
procedencia y su estado de verificación.

`REMOTE_FILES` está VACÍO deliberadamente: ninguna de las fuentes primarias
(Cranfield OA, tesis U. Toronto, MDPI, arXiv — ver Ciencia §14) publica los
coeficientes como archivo descargable estable; son PDFs. Todo agregado se
versiona en el repo y la validación funciona **offline**. La maquinaria de
descarga + manifiesto se conserva por si aparece un mirror estable.

Estado de verificación (la «honestidad calibrada» de la casa). Cada CSV lleva
en su cabecera `#`:

  * líneas de prosa: qué es la tabla y de dónde sale;
  * `@source:` fuente primaria (URL o cita cap./ec. de Lefebvre & Ballal);
  * `@status:` `verified` | `provisional`;
  * `@status_reason:` obligatorio cuando `@status: provisional`.

La arroba distingue metadato de prosa sin ambigüedad.

Ese estado **es dato, no adorno**: `load_table()` lo devuelve en la clave
`status`, y las tablas heterogéneas llevan además una columna `status` por
fila (la columna manda sobre la cabecera para esa fila).

Las tablas `thermo/tad_fit_<fuel>.csv` son el fallback de T_ad SIN Cantera. Se
generan CON Cantera (equilibrio 'HP', ≡ NASA CEA con desvío 0.06–3.5 %,
arXiv:2503.11826) sobre una rejilla (φ, T3, P3); el error de interpolación
trilineal medido contra Cantera se anota en la propia cabecera (`interp_*`) y
está por debajo del 1 % exigido. Sin Cantera, `build()` las emite con un modelo
analítico de respaldo y las marca `provisional`.

El dataset de ENTRENAMIENTO del surrogate no se descarga: se autogenera
muestreando L0 (flywheel `dataset.csv` de cada corrida — ver
`neural_optimizer.py`), misma estrategia que los hermanos.

Uso CLI:
  python data_pipeline.py                  # build + validate
  python data_pipeline.py --build          # regenera todas las tablas
  python data_pipeline.py --validate       # verifica manifiesto + tablas
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Whitelist remota (vacía — ver docstring) y registro de tablas en repo
# ----------------------------------------------------------------------------

REMOTE_FILES: dict[str, str] = {}

#: Tablas agregadas versionadas en el repo, en el orden del contrato §7.
#: Cada CSV lleva su procedencia y su estado en las líneas de comentario (#).
IN_REPO_FILES = (
    "lefebvre/eta_vs_theta.csv",        # η_c(θ)                     [Ciencia §2.2]
    "lefebvre/plf_by_arch.csv",         # PLF, Δp/p, U_ref por arq.  [Ciencia §2.1]
    "lefebvre/air_split_by_arch.csv",   # reparto de aire            [Ciencia §3]
    "lefebvre/zone_lengths.csv",        # A_L/A_ref y L_PZ/L_SZ/L_DZ [Ciencia §2.5]
    "lefebvre/cd_holes.csv",            # C_d por tipo de orificio   [Ciencia §4.1]
    "lefebvre/cd_am_roughness.csv",     # factor f(Ra/d) de AM       [Ciencia §4.1]
    "lefebvre/pf_constant.csv",         # k de la correlación de PF  [Ciencia §4.3]
    "lefebvre/swirler_K.csv",           # K_sw álabes planos/curvos  [Ciencia §4.4]
    "lefebvre/thermal_coeffs.csv",      # 0.020 / 0.29 / η_film      [Ciencia §5]
    "lefebvre/film_effusion.csv",       # efusión vs M y ángulo      [Ciencia §5.1]
    "lefebvre/lbo_constant.csv",        # constante de φ_LBO         [Ciencia §6.1]
    "lefebvre/atomization_coeffs.csv",  # SMD (5 familias)           [Ciencia §7]
    "lefebvre/emissions_coeffs.csv",    # EI_NOx / EI_CO             [Ciencia §8]
    "reference/class_table.csv",        # tabla por clase de potencia [EstadoDelArte §7]
    "thermo/tad_fit_jet_a.csv",         # T_ad fallback              [Ciencia §9.3]
    "thermo/tad_fit_propane.csv",
    "thermo/tad_fit_h2.csv",
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

#: Semilla de la familia (CONTRATO §10.8). Aquí no se usa aleatoriedad —
#: se declara para que el determinismo del módulo sea explícito.
SEED = 71

P_ATM = 101_325.0  # Pa — idéntico a physics_core.P_ATM (NO redefinir con otro valor)


# ----------------------------------------------------------------------------
# Tunables de la capa 0 (SCREAMING_CASE con procedencia y aviso de recalibración)
#
# Regla de la casa: si el número no está verificado contra fuente primaria, se
# marca `provisional` en el CSV que lo consume Y se recalibra en WAVE1/WAVE7.
# ----------------------------------------------------------------------------

#: θ (SI) al que la literatura sitúa η_c ≈ 99.9 % [Ciencia §2.2]. ⚠️ PROVISIONAL:
#: la carta original de Lefebvre cap. 5 no se ha digitalizado; este es el único
#: punto de anclaje citado en la investigación. RECALIBRAR con la carta real.
ETA_THETA_REF_SI = 7.0e7
ETA_THETA_AT_REF = 0.999      # η_c en θ = ETA_THETA_REF_SI
ETA_THETA_CEILING = 0.9995    # techo asintótico (η_comb real nunca llega a 1)

#: k de PF = exp[−k·(L_DZ/D_L)·(ΔP_L/q_ref)] [Ciencia §4.3]. ⚠️ PROVISIONAL:
#: calibrado a mano para que el caso ancla micro (L_DZ/D_L = 1.8, ΔP_L/q_ref =
#: 15) caiga en PF ≈ 0.35, centro de la banda 0.30–0.45 de la clase micro
#: (EstadoDelArte §7). RECALIBRAR contra el libro cap. 5 o contra ensayo.
PF_K_DILUTION = -math.log(0.35) / (1.8 * 15.0)

#: Constante global de φ_LBO (SI) [Ciencia §6.1]. ⚠️ PROVISIONAL: la forma
#: funcional está verificada, la constante NO. Calibrada para que el caso ancla
#: REF_CB10 (P3 = 250 kPa, T3 = 430 K, ṁ = 0.13 kg/s, A_ref/V_PZ ≈ 33.3 1/m,
#: D32 = 50 µm, λ_eff = 8e-7 m²/s, LHV = 43.1 MJ/kg) dé φ_LBO ≈ 0.50, es decir
#: margen ≈ 2× sobre φ_PZ ≈ 1.0. Precisión declarada de Lefebvre: ±45 %.
#: RECALIBRAR en WAVE7 contra el banco propio.
LBO_CONST_SI = 6.95e16

#: Pérdida máxima de C_d por rugosidad AM en orificios de aire del liner
#: [Ciencia §4.1: «reduce C_d típicamente 5–20 % vs pulido»]. ⚠️ PROVISIONAL:
#: no existe correlación cerrada publicada — la forma f = 1 − LOSS·(Ra/d /
#: RD_MAX)^EXP es de la casa. El único dato discreto verificado (DOI
#: 10.1016/j.jmapro.2022.11.031) vive en `cd_holes.csv`, no aquí.
CD_AM_LOSS_MAX = 0.20
CD_AM_RD_MAX = 0.05
CD_AM_EXP = 0.70

#: Eficacia de efusión: η = SAT·(1 − exp(−M/M_SCALE))·f(ángulo) [Ciencia §5.1].
#: ⚠️ PROVISIONAL: «correlación de eficacia de efusión en forma cerrada: no
#: verificada» (hueco declarado). SAT y M_SCALE se ajustan para reproducir la
#: banda de dato ✅ η = 0.50–0.70 en el rango de diseño M = 0.3–0.8.
EFFUSION_ETA_SAT = 0.72
EFFUSION_M_SCALE = 0.22
EFFUSION_ANGLE_FACTOR = {20.0: 1.00, 30.0: 1.00, 45.0: 0.92, 60.0: 0.83}
EFFUSION_CONICAL_GAIN = 1.25  # ✅ +25 % vs cilíndrico a M = 0.25 [Ciencia §5.1]

#: Surrogate de Jet-A: mezcla molar n-dodecano (c12h26) + benceno (A1) ajustada
#: para reproducir EXACTAMENTE el H/C = 1.91 del contrato §6 (≡ C12H23).
#:   x_A1 = 3.08/8.54 resuelve (26 − 20x)/(12 − 6x) = 1.91.
#: Elección de mecanismo: Cantera 3.2 NO trae ningún mecanismo con POSF10325 ni
#: con C12H23; `nDodecane_Reitz.yaml` es el único bundled con una especie C12.
#: El benceno aporta el aromático que baja el H/C al valor real del queroseno.
JET_A_X_A1 = 3.08 / 8.54

#: Rejilla de las tablas de T_ad. φ densificado alrededor del pico (φ ≈ 1.05)
#: porque es donde la interpolación lineal más sufre.
TAD_PHI_GRID = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
                1.00, 1.05, 1.10, 1.20, 1.30, 1.40, 1.60, 1.80, 2.00)
TAD_T3_GRID = (250.0, 350.0, 450.0, 550.0, 650.0, 750.0, 850.0, 950.0,
               1050.0, 1150.0)                      # bounds de θ[17] ±
TAD_P3_GRID = (1.0e5, 2.0e5, 4.0e5, 8.0e5, 1.4e6, 2.0e6)   # bounds de θ[16]

#: Anclas de T_ad a φ = 1, T3 = 300 K, P = 1 atm [K]. Bandas de literatura
#: recogidas en WAVE1 §4 (T6): queroseno 2200–2350, H₂ 2350–2450. Se usan SOLO
#: para calibrar el modelo analítico de respaldo (sin Cantera). ⚠️ El respaldo
#: es PROVISIONAL por construcción.
TAD_ANCHOR_K = {"jet_a": 2273.7, "propane": 2266.5, "h2": 2381.3}

#: Modelo analítico de respaldo (solo si falta Cantera). Déficit por disociación
#:   D = A·(sqrt(exc² + E0²) − E0)·(P3/P_atm)^P_EXP,  exc = max(0, T_ideal − T_ON)
#: La forma «palo de hockey» garantiza dD/dT_ideal = A·exc/sqrt(exc²+E0²) < A < 1,
#: es decir T_ad SIEMPRE crece con T3 (monotonía estructural, no ajustada).
#: A se calibra por combustible con su ancla; el resto son globales ajustados
#: contra las propias tablas de Cantera de este repo (2026-07-30):
#: error de T_ad medio 1.4–2.2 %, máximo 5.5 %; cp medio 1.8 %; MW medio 0.5 %.
#: RECALIBRAR si se cambian las anclas o la rejilla.
TAD_DISS_ONSET_K = 1900.0   # T por encima de la cual la disociación muerde
TAD_DISS_E0_K = 100.0       # suavizado del arranque del déficit
TAD_DISS_P_EXP = -0.16      # más presión ⇒ menos disociación ⇒ menos déficit
TAD_DISS_A_MAX = 0.95       # tope de A (blinda la monotonía ante recalibración)
#: Penalización rica: a φ > 1 los productos son CO/H₂ en vez de CO₂/H₂O, así que
#: el calor liberado cae por debajo de la fracción quemable 1/φ. El efecto es
#: proporcional a la fracción de O₂ estequiométrico que consume el CARBONO
#: (0.677 en jet-A, 0.600 en propano, 0.0 en H₂ — que por eso no la sufre).
TAD_RICH_A = 0.40
TAD_RICH_B = 1.6
TAD_RICH_FLOOR = 0.30

#: LHV de contrato (CONTRATO §6). `LHV_JET_A` DEBE coincidir con el del ciclo
#: (CONTEXTO §4: fan_core.LHV_JET_A = 43.1e6). El surrogate de Cantera tiene un
#: LHV ligeramente distinto (43.67 MJ/kg): `build()` corrige la entalpía de los
#: reactivos para que la tabla entregue EXACTAMENTE esta energía — así T_ad y el
#: balance de FAR del seam de ciclo hablan del mismo combustible.
FUEL_LHV_J_KG = {"jet_a": 43.1e6, "propane": 46.35e6, "h2": 120.0e6}

#: Átomos (C, H) por mol de combustible del surrogate de cada rama.
FUEL_ATOMS_CH = {
    "jet_a": (12.0 - 6.0 * JET_A_X_A1, 26.0 - 20.0 * JET_A_X_A1),
    "propane": (3.0, 8.0),
    "h2": (0.0, 2.0),
}

MW_C, MW_H, MW_O2, MW_N2 = 12.011, 1.008, 31.998, 28.014
AIR_N2_PER_O2 = 3.76  # aire seco simplificado O2:1, N2:3.76 (convenio Cantera)


def _far_stoich(fuel: str) -> float:
    """FAR estequiométrico másico del surrogate (exacto dada la fórmula)."""
    nC, nH = FUEL_ATOMS_CH[fuel]
    mw_f = nC * MW_C + nH * MW_H
    n_o2 = nC + nH / 4.0
    mw_air_per_o2 = MW_O2 + AIR_N2_PER_O2 * MW_N2
    return mw_f / (n_o2 * mw_air_per_o2)


def _carbon_o2_fraction(fuel: str) -> float:
    """Fracción del O₂ estequiométrico que consume el carbono del combustible."""
    nC, nH = FUEL_ATOMS_CH[fuel]
    return nC / (nC + nH / 4.0)


# ----------------------------------------------------------------------------
# Manifiesto SHA-256
# ----------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "phy-cb/0.1"})
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                dest.write_bytes(r.read())
            return
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"No se pudo descargar {url}: {last_err}")


def fetch(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> dict:
    """Descarga los archivos de la whitelist remota (hoy: ninguno).

    Idempotente; devuelve/actualiza el manifiesto de descargas.
    """
    raw = Path(data_dir) / "raw"
    manifest_path = raw / "fetch_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})
    if not REMOTE_FILES:
        return manifest
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in REMOTE_FILES.items():
        dest = raw / name
        if dest.exists() and not force:
            continue
        print(f"[fetch] {name} <- {url}")
        _download(url, dest)
        manifest[name] = {
            "url": url, "sha256": _sha256(dest), "bytes": dest.stat().st_size,
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_manifest(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Escribe data/manifest.json con el SHA-256 y el estado de cada tabla."""
    data_dir = Path(data_dir)
    man: dict[str, dict] = {}
    for rel in IN_REPO_FILES:
        p = data_dir / rel
        if not p.exists():
            continue
        t = load_table(rel, data_dir)
        man[rel] = {
            "sha256": _sha256(p),
            "bytes": p.stat().st_size,
            "rows": int(t["n_rows"]) if t else 0,
            "status": t["status"] if t else "unknown",
        }
    out = data_dir / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


def verify_manifest(data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    """Devuelve la lista de problemas (vacía si todo verifica)."""
    data_dir = Path(data_dir)
    problems: list[str] = []
    man_path = data_dir / "manifest.json"
    if not man_path.exists():
        return [f"falta {man_path}"]
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for rel in IN_REPO_FILES:
        p = data_dir / rel
        if not p.exists():
            problems.append(f"falta {rel}")
        elif rel not in man:
            problems.append(f"{rel} sin entrada en el manifiesto")
        elif _sha256(p) != man[rel]["sha256"]:
            problems.append(f"{rel}: SHA-256 no coincide con el manifiesto")
    return problems


# ----------------------------------------------------------------------------
# Escritura y lectura de CSV con cabecera de procedencia
# ----------------------------------------------------------------------------

#: Cabeceras `# @clave: valor` reconocidas como metadatos (el resto es prosa).
#: La arroba es deliberada: sin marcador, una línea de prosa como
#: «Odgers: 29·exp(...)» se colaría como metadato (bug real, 2026-07-30).
_META_RE = re.compile(r"^@([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*)$")


def _fmt(v) -> str:
    """Formato determinista de celda: 6 cifras significativas para floats."""
    if isinstance(v, str):
        return v
    fv = float(v)
    if math.isnan(fv):
        return "nan"
    if math.isinf(fv):
        return "inf" if fv > 0 else "-inf"
    return f"{fv:.6g}"


def _write_csv(path: Path, doc: list[str], meta: dict[str, str],
               cols: list[str], rows) -> None:
    """Escribe un CSV con cabecera `#` de procedencia + estado.

    `doc` son líneas de prosa; `meta` se emite como `# @clave: valor` y es lo
    que `load_table()` devuelve en `provenance` (de ahí sale `status`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF determinista (csv.writer emite \r\n por defecto en Windows): los
    # hashes del manifiesto deben ser idénticos en cualquier plataforma.
    with open(path, "w", newline="", encoding="utf-8") as f:
        for line in doc:
            f.write(f"# {line}\n")
        for k, v in meta.items():
            f.write(f"# @{k}: {v}\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for row in rows:
            w.writerow([_fmt(v) for v in row])


def _read_csv(path: Path) -> tuple[list[str], dict[str, str], list[str], list[list[str]]]:
    """Lee un CSV con cabecera `#`: (prosa, metadatos, columnas, filas)."""
    doc: list[str] = []
    meta: dict[str, str] = {}
    body: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                txt = line[1:].strip()
                m = _META_RE.match(txt)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
                elif txt:
                    doc.append(txt)
            else:
                body.append(line)
    rows = [r for r in csv.reader(body) if r]
    if not rows:
        return doc, meta, [], []
    return doc, meta, rows[0], rows[1:]


# ----------------------------------------------------------------------------
# Build — tablas de correlación (Ciencia §§2–8 + EstadoDelArte §7)
# ----------------------------------------------------------------------------

def _build_eta_vs_theta(data_dir: Path) -> None:
    """η_c(θ) — carta de eficiencia vs parámetro θ de Lefebvre [Ciencia §2.2]."""
    b = -math.log(1.0 - ETA_THETA_AT_REF / ETA_THETA_CEILING)
    theta = np.geomspace(1.0e6, 8.0e7, 25)
    eta = ETA_THETA_CEILING * (1.0 - np.exp(-b * theta / ETA_THETA_REF_SI))
    _write_csv(
        data_dir / "lefebvre" / "eta_vs_theta.csv",
        ["Eficiencia de combustion cinetica eta_theta frente al parametro theta",
         "de Lefebvre, theta = P3^1.75 · A_ref · D_ref^0.75 · exp(T3/300) / mdot3",
         "en SI (Pa, m2, m, kg/s, K). Curva monotona creciente que satura en un",
         "techo asintotico; se usa como tabla lookup (interpolar lineal en",
         "log10(theta_SI)). Fuera de rango: clampear a los extremos.",
         "AVISO: la carta original de Lefebvre cap. 5 NO se ha digitalizado.",
         "Esta curva reproduce el UNICO punto de anclaje citado en la",
         "investigacion (eta = 99.9 % en theta = 7e7 SI) con una forma",
         "exponencial saturada de la casa. Es una forma plausible, no la carta."],
        {"source": "Lefebvre & Ballal, 'Gas Turbine Combustion' 3.a ed., cap. 5 "
                   "(carta eta_c vs theta); anclaje theta=7e7 SI segun "
                   "Quasar_PhyCB_Science.md 2.2",
         "status": "provisional",
         "status_reason": "carta sin digitalizar; solo se conoce un punto de "
                          "anclaje (theta=7e7 -> eta=0.999). Bloqueante "
                          "declarado en EstadoDelArte 11 (prioridad alta 2). "
                          "RECALIBRAR en WAVE1 con la carta original.",
         "model": f"eta = {ETA_THETA_CEILING:.4f}*(1 - exp(-{b:.4f}*theta/"
                  f"{ETA_THETA_REF_SI:.3g}))",
         "generated_by": "data_pipeline.build()"},
        ["theta_SI", "eta_theta"],
        np.column_stack([theta, eta]))


def _build_plf_by_arch(data_dir: Path) -> None:
    """PLF = ΔP3-4/q_ref, Δp/p y U_ref por arquitectura [Ciencia §2.1]."""
    rows = [
        # arch, arch_id, PLF min/max/nom, dp/p min/max/nom, U_ref min/max/nom
        ("can", 0, 35.0, 37.0, 36.0, 0.060, 0.070, 0.065, 25.0, 35.0, 30.0,
         "verified"),
        ("tuboannular", -1, 25.0, 28.0, 26.5, 0.050, 0.060, 0.055, 25.0, 40.0,
         32.5, "verified"),
        ("annular", 1, 18.0, 20.0, 19.0, 0.050, 0.060, 0.055, 25.0, 45.0, 35.0,
         "verified"),
        ("reverse_annular", 2, 20.0, 25.0, 22.5, 0.050, 0.060, 0.055, 20.0,
         40.0, 30.0, "provisional"),
        ("micro_target", -2, 15.0, 25.0, 20.0, 0.020, 0.040, 0.030, 10.0, 25.0,
         17.5, "verified"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "plf_by_arch.csv",
        ["Factor de perdida de presion PLF = DeltaP_3-4/q_ref, banda de Dp/p y",
         "velocidad de referencia U_ref por arquitectura de camara.",
         "arch_id sigue el gate theta[0] del contrato: 0=can, 1=annular,",
         "2=reverse_annular. arch_id = -1 -> arquitectura fuera del gate;",
         "arch_id = -2 -> fila de OBJETIVO de clase, no de arquitectura.",
         "La fila micro_target es la regla dura de Phy-CB: en microturbinas",
         "Dp/p (2-4 %) es RESTRICCION del ciclo, no salida; cada punto extra",
         "vale ~ -0.5 % de empuje o +0.25 % de consumo especifico.",
         "Fraccion tipica: DP_liner = 70-80 % del DP total (resto difusor+snout).",
         "La columna 'status' manda por fila: reverse_annular no aparece en la",
         "tabla de PLF de la literatura y se estima por analogia."],
        {"source": "Quasar_PhyCB_Science.md 2.1 (tabla PLF por arquitectura) "
                   "-> Lefebvre & Ballal 3.a ed. cap. 4; banda micro verificada "
                   "en EstadoDelArte 1",
         "status": "verified",
         "status_rowcol": "la columna 'status' manda por fila "
                          "(reverse_annular = provisional)",
         "generated_by": "data_pipeline.build()"},
        ["arch", "arch_id", "PLF_min", "PLF_max", "PLF_nom",
         "dp_over_p_min", "dp_over_p_max", "dp_over_p_nom",
         "U_ref_min_ms", "U_ref_max_ms", "U_ref_nom_ms", "status"],
        rows)


def _build_air_split_by_arch(data_dir: Path) -> None:
    """Reparto de aire y riquezas de zona por concepto de combustion [§3]."""
    rows = [
        # regime, phycb_default, pz(min,max,nom), sz, dz, cool, phi_pz, phi_sz
        ("conventional_aero", 0,
         0.18, 0.25, 0.22, 0.10, 0.15, 0.13, 0.25, 0.45, 0.35, 0.20, 0.40, 0.30,
         0.80, 1.20, 0.60, 0.80, "verified"),
        ("microturbine", 1,
         0.25, 0.35, 0.30, 0.10, 0.20, 0.12, 0.35, 0.55, 0.42, 0.10, 0.25, 0.16,
         0.85, 1.15, 0.60, 0.80, "verified"),
        ("rql", 0,
         0.45, 0.60, 0.50, 0.00, 0.00, 0.00, 0.25, 0.35, 0.30, 0.15, 0.25, 0.20,
         1.20, 1.80, float("nan"), float("nan"), "verified"),
        ("lean_premix", 0,
         0.60, 0.75, 0.65, 0.00, 0.00, 0.00, 0.15, 0.25, 0.20, 0.10, 0.20, 0.15,
         0.50, 0.60, float("nan"), float("nan"), "verified"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "air_split_by_arch.csv",
        ["Reparto de aire (fracciones de mdot3) y riquezas objetivo por zona.",
         "El reparto de Phy-CB NO depende de la arquitectura geometrica sino",
         "del concepto de combustion: la fila por defecto para las tres",
         "arquitecturas del gate theta[0] es 'microturbine' (phycb_default=1).",
         "frac_cool es CIERRE, no variable: frac_cool = 1 - (pz + sz + dz).",
         "Los valores _nom de cada fila suman exactamente 1 (validado).",
         "Los _nom de microturbine son los de THETA_REF (contrato 1) para que",
         "el caso ancla REF_CB10 y esta tabla no puedan divergir.",
         "phi_sz = nan donde el concepto no tiene zona secundaria.",
         "Dato de calibracion medido (EstadoDelArte 1.8): en un",
         "microturborreactor real la optimizacion bajo el aire de PZ de",
         "39.35 % a 32 % (phi_PZ 0.89 -> 1.09) para estabilizar en aceleracion."],
        {"source": "Quasar_PhyCB_Science.md 3 (tabla de reparto y riquezas); "
                   "banda micro de phi_PZ en EstadoDelArte 7",
         "status": "verified",
         "generated_by": "data_pipeline.build()"},
        ["regime", "phycb_default",
         "frac_pz_min", "frac_pz_max", "frac_pz_nom",
         "frac_sz_min", "frac_sz_max", "frac_sz_nom",
         "frac_dz_min", "frac_dz_max", "frac_dz_nom",
         "frac_cool_min", "frac_cool_max", "frac_cool_nom",
         "phi_pz_min", "phi_pz_max", "phi_sz_min", "phi_sz_max", "status"],
        rows)


def _build_zone_lengths(data_dir: Path) -> None:
    """A_L/A_ref y longitudes de zona normalizadas por D_L [Ciencia §2.5]."""
    rows = [
        # arch, arch_id, AL/Aref, Lpz(min,max,nom), Lsz_nom, Ldz(min,max,nom),
        # Ltot(min,max)
        ("can", 0, 0.66, 0.66, 0.75, 0.70, 0.50, 1.50, 2.50, 2.00, 2.50, 4.00,
         "provisional"),
        ("tuboannular", -1, 0.66, 0.66, 0.75, 0.70, 0.50, 1.50, 2.50, 2.00,
         2.50, 4.00, "provisional"),
        ("annular", 1, 0.70, 0.66, 0.75, 0.70, 0.50, 1.50, 2.50, 2.00, 2.50,
         4.00, "provisional"),
        ("reverse_annular", 2, 0.70, 0.66, 0.75, 0.70, 0.50, 1.50, 2.50, 2.00,
         2.50, 4.00, "provisional"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "zone_lengths.csv",
        ["Cociente area de liner / area de referencia y longitudes de zona",
         "normalizadas por el diametro del liner D_L.",
         "L_DZ/D_L lo fija en realidad el factor de patron (ver pf_constant.csv):",
         "esta banda es solo el punto de partida del dimensionado.",
         "reverse_annular hereda los cocientes del anular: los dos giros de",
         "180 grados cambian la perdida de forma, NO estos cocientes de area.",
         "AVISO: los cocientes estan marcados [L-verificar] en Ciencia 2.5."],
        {"source": "Quasar_PhyCB_Science.md 2.5 -> Lefebvre & Ballal 3.a ed. "
                   "caps. 4-5",
         "status": "provisional",
         "status_reason": "cocientes A_L/A_ref y L_zona/D_L marcados "
                          "[L-verificar] en Ciencia 2.5; pendientes de cotejo "
                          "contra el libro (tarea bloqueante de WAVE1 0).",
         "generated_by": "data_pipeline.build()"},
        ["arch", "arch_id", "AL_over_Aref",
         "L_pz_over_DL_min", "L_pz_over_DL_max", "L_pz_over_DL_nom",
         "L_sz_over_DL_nom",
         "L_dz_over_DL_min", "L_dz_over_DL_max", "L_dz_over_DL_nom",
         "L_tot_over_DL_min", "L_tot_over_DL_max", "status"],
        rows)


def _build_cd_holes(data_dir: Path) -> None:
    """C_d por tipo de orificio [Ciencia §4.1]."""
    rows = [
        ("plain_sharp_edged", 0.60, 0.65, 0.62, "verified"),
        ("plunged_flared", 0.75, 0.85, 0.80, "verified"),
        ("first_iteration_conservative", 0.50, 0.50, 0.50, "verified"),
        ("effusion_cylindrical", 0.55, 0.75, 0.65, "provisional"),
        ("effusion_conical", 0.60, 0.80, 0.70, "provisional"),
        ("teardrop_am_selfsupporting", 0.55, 0.70, 0.62, "provisional"),
        ("simplex_orifice_am_fine", 0.37, 0.41, 0.39, "verified"),
        ("simplex_orifice_am_rough", 0.50, 0.58, 0.54, "verified"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "cd_holes.csv",
        ["Coeficiente de descarga C_d por tipo de orificio, para",
         "mdot_h = C_d · A_h · sqrt(2 · rho3 · DeltaP_L).",
         "C_d depende ademas del bleed ratio y de Re: estas son bandas de",
         "primera iteracion, no valores de una geometria concreta.",
         "Las dos ultimas filas son el DATO DISCRETO VERIFICADO de rugosidad",
         "AM: tobera simplex impresa de d_o = 1.2 mm, C_d 0.37-0.41 con acabado",
         "fino y 0.50-0.58 con rugosidad alta (menor nucleo de aire). OJO: ahi",
         "la rugosidad SUBE C_d; en orificios de aire del liner lo BAJA - ver",
         "cd_am_roughness.csv. No mezclar los dos efectos.",
         "Las filas de efusion y teardrop no tienen fuente primaria: son",
         "extrapolaciones de la casa (columna status = provisional)."],
        {"source": "Quasar_PhyCB_Science.md 4.1; dato AM discreto en "
                   "DOI 10.1016/j.jmapro.2022.11.031 (Brno UT, OA)",
         "status": "verified",
         "status_rowcol": "la columna 'status' manda por fila (efusion, "
                          "efusion conica y teardrop son provisionales)",
         "generated_by": "data_pipeline.build()"},
        ["hole_type", "Cd_min", "Cd_max", "Cd_nom", "status"],
        rows)


def _build_cd_am_roughness(data_dir: Path) -> None:
    """Factor f(Ra/d) de rugosidad AM sobre C_d de orificios del liner [§4.1]."""
    rd = np.array([0.0, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030,
                   0.040, 0.050])
    f = 1.0 - CD_AM_LOSS_MAX * (rd / CD_AM_RD_MAX) ** CD_AM_EXP
    _write_csv(
        data_dir / "lefebvre" / "cd_am_roughness.csv",
        ["Factor multiplicativo f(Ra/d) que aplica la rugosidad as-built LPBF",
         "al C_d ideal de un orificio de aire del liner:  C_d,AM = C_d,ideal · f.",
         "Ra as-built tipica 6-20 um en vertical y 15-40 um en down-skins;",
         "con d = 0.5-5 mm, Ra/d cae en 0.002-0.05 (rango de esta tabla).",
         "Interpolar lineal en Ra/d; fuera de rango, clampear.",
         "AVISO IMPORTANTE: no existe correlacion cerrada publicada. Lo unico",
         "verificado es (a) la banda cualitativa '5-20 % de reduccion vs",
         "pulido' de Ciencia 4.1 y (b) el dato discreto de tobera de",
         "cd_holes.csv (que ademas va en el sentido contrario, por tratarse de",
         "un atomizador y no de un orificio de aire). La forma de esta curva es",
         "de la casa y esta declarada como tal."],
        {"source": "Quasar_PhyCB_Science.md 4.1 (banda 5-20 %); dato discreto "
                   "DOI 10.1016/j.jmapro.2022.11.031",
         "status": "provisional",
         "status_reason": "no hay correlacion cerrada publicada (hueco "
                          "declarado en EstadoDelArte 11, prioridad baja 9). "
                          "La forma f = 1 - LOSS·(Ra/d / RD_MAX)^EXP es de la "
                          "casa, calibrada solo a la banda 5-20 %. RECALIBRAR "
                          "con medidas propias de orificios impresos (WAVE7).",
         "model": f"f = 1 - {CD_AM_LOSS_MAX:.2f}*(Ra_over_d/"
                  f"{CD_AM_RD_MAX:.3f})^{CD_AM_EXP:.2f}",
         "generated_by": "data_pipeline.build()"},
        ["Ra_over_d", "f_Cd", "applies_to", "status"],
        [(rd[i], f[i], "liner_air_hole", "provisional") for i in range(len(rd))])


def _build_pf_constant(data_dir: Path) -> None:
    """Constante k del factor de patron y bandas objetivo [Ciencia §4.3]."""
    rows = [
        ("k_dilution", PF_K_DILUTION, "-", "provisional"),
        ("PF_max_default", 0.35, "-", "provisional"),
        ("PF_target_cooled_turbine_min", 0.15, "-", "verified"),
        ("PF_target_cooled_turbine_max", 0.25, "-", "verified"),
        ("PF_target_conventional_min", 0.25, "-", "verified"),
        ("PF_target_conventional_max", 0.35, "-", "verified"),
        ("PF_target_micro_min", 0.30, "-", "verified"),
        ("PF_target_micro_max", 0.45, "-", "verified"),
        ("PF_measured_small_min", 0.21, "-", "verified"),
        ("PF_measured_small_max", 0.24, "-", "verified"),
        ("RTDF_min", 0.12, "-", "verified"),
        ("RTDF_max", 0.30, "-", "verified"),
        ("rtdf_over_pf", 0.75, "-", "provisional"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "pf_constant.csv",
        ["Constantes del factor de patron.",
         "Correlacion de dilucion (forma exponencial):",
         "   PF = exp[ -k_dilution · (L_DZ/D_L) · (DeltaP_L/q_ref) ]",
         "Definiciones que coexisten en la literatura (exponer AMBAS):",
         "   PF  = (T4_max - T4_avg)/(T4_avg - T3)   <- estandar de diseno",
         "   PF* = (T4_peak - T4_avg)/T4_avg",
         "   RTDF = (T4_max_radial - T4_avg)/(T4_avg - T3)",
         "k_dilution esta CALIBRADO A MANO, no digitalizado: se eligio para que",
         "el caso ancla micro (L_DZ/D_L = 1.8, DeltaP_L/q_ref = 15) caiga en",
         "PF = 0.35, centro de la banda 0.30-0.45 de la clase micro.",
         "PF_max_default es el limite que consume la restriccion g[0] del",
         "contrato 5.1; se puede sobreescribir desde DesignSpec."],
        {"source": "Quasar_PhyCB_Science.md 4.3 -> Lefebvre & Ballal 3.a ed. "
                   "cap. 5; bandas medidas en EstadoDelArte 1 y 7",
         "status": "provisional",
         "status_reason": "la constante k de la correlacion de PF esta marcada "
                          "[L-verificar] en Ciencia 4.3 y aqui esta calibrada a "
                          "un solo punto ancla. Las filas PF_target_* y "
                          "PF_measured_* SI son dato verificado (ver columna "
                          "status). RECALIBRAR k en WAVE1/WAVE7.",
         "generated_by": "data_pipeline.build()"},
        ["param", "value", "units", "status"],
        rows)


def _build_swirler_K(data_dir: Path) -> None:
    """K_sw de la perdida del swirler y bandas de diseno [Ciencia §4.4]."""
    rows = [
        ("flat_vanes", 1.30, 45.0, 60.0, 0.60, 1.00, 0.60, "provisional"),
        ("curved_vanes", 1.15, 45.0, 60.0, 0.60, 1.00, 0.60, "provisional"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "swirler_K.csv",
        ["Coeficiente K_sw de la perdida de presion del swirler:",
         "   DeltaP_sw/q_ref = K_sw · [ (A_ref/(A_sw·sec beta))^2",
         "                              - (A_ref/A_L)^2 ]",
         "y bandas de diseno del numero de swirl",
         "   S_N = (2/3)·[(1-(D_hub/D_sw)^3)/(1-(D_hub/D_sw)^2)]·tan(beta).",
         "S_N > SN_ctrz_min genera zona de recirculacion toroidal central",
         "(CTRZ) - este umbral SI esta verificado; K_sw no.",
         "beta en grados (interfaz); internamente radianes."],
        {"source": "Quasar_PhyCB_Science.md 4.4 -> Lefebvre & Ballal 3.a ed. "
                   "cap. 4",
         "status": "provisional",
         "status_reason": "K_sw = 1.3 / 1.15 marcado [L-verificar] en Ciencia "
                          "4.4; la forma de la ecuacion si es estandar. "
                          "RECALIBRAR contra el libro cap. 4.",
         "generated_by": "data_pipeline.build()"},
        ["vane_type", "K_sw", "beta_deg_min", "beta_deg_max",
         "SN_design_min", "SN_design_max", "SN_ctrz_min", "status"],
        rows)


def _build_thermal_coeffs(data_dir: Path) -> None:
    """Coeficientes del balance termico 1D del liner [Ciencia §5]."""
    rows = [
        ("rad_internal_prefactor", 0.50, "-", "provisional"),
        ("gas_emissivity_coeff", 0.29, "-", "provisional"),
        ("conv_internal_coeff", 0.020, "-", "provisional"),
        ("conv_external_coeff", 0.020, "-", "provisional"),
        ("conv_reynolds_exp", 0.80, "-", "verified"),
        ("conv_diameter_exp", -0.20, "-", "verified"),
        ("rad_internal_Tg_exp", 1.50, "-", "provisional"),
        ("rad_internal_dT_exp", 2.50, "-", "provisional"),
        ("luminosity_jet_a", 1.70, "-", "verified"),
        ("luminosity_propane", 1.00, "-", "verified"),
        ("luminosity_h2", 1.00, "-", "verified"),
        ("eps_wall_default", 0.80, "-", "provisional"),
        ("film_slot_C", 1.10, "-", "provisional"),
        ("film_slot_exp_m", 0.65, "-", "provisional"),
        ("film_slot_exp_mu_ratio", 0.15, "-", "provisional"),
        ("film_slot_exp_x_over_s", -0.20, "-", "provisional"),
        ("film_slot_exp_t_over_s", -0.20, "-", "provisional"),
        ("film_slot_m_min", 0.50, "-", "verified"),
        ("film_slot_m_max", 1.30, "-", "verified"),
        ("T3_recuperated_max_K", 1093.0, "K", "verified"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "thermal_coeffs.csv",
        ["Coeficientes del balance termico estacionario 1D del liner:",
         "   R1 + C1 = K_1-2 = R2 + C2   (por unidad de area)",
         "   R1 = rad_internal_prefactor · sigma · (1+eps_w) · eps_g",
         "        · Tg^1.5 · (Tg^2.5 - Tw1^2.5)",
         "   eps_g = 1 - exp(-gas_emissivity_coeff · P3 · L · (f·l_b)^0.5",
         "                   · Tg^-1.5)",
         "   C1 = conv_internal_coeff · (k_g/D_L^0.2)",
         "        · (mdot_g/(A_L·mu_g))^0.8 · (Tg - Tw1)",
         "   C2 = conv_external_coeff · (k_a/D_an^0.2)",
         "        · (mdot_an/(A_an·mu_a))^0.8 · (Tw2 - T3)",
         "y de la refrigeracion por ranura de Lefebvre:",
         "   eta_film = film_slot_C · m^0.65 · (mu_g/mu_c)^0.15",
         "              · (x/s)^-0.2 · (t/s)^-0.2,  valido 0.5 < m < 1.3.",
         "La ESTRUCTURA (Dittus-Boelter, exponente 0.8) es solida; los",
         "coeficientes 0.020 / 0.29 / 1.10 estan marcados [L-verificar].",
         "eps_wall_default es de handbook: el valor por material vive en",
         "structures_core.MATERIALS (emissivity_wall) y manda sobre este.",
         "Punto critico micro-recuperado: con T3 hasta 1093 K (recuperador ~90 %,",
         "HiETA/Bath) C2 se desploma y el aire deja de refrigerar por conveccion",
         "- es EL argumento para efusion densa fabricada por AM."],
        {"source": "Quasar_PhyCB_Science.md 5 y 5.1 -> Lefebvre & Ballal 3.a "
                   "ed. caps. 8-10; T3 recuperada medida en EstadoDelArte 3.4",
         "status": "provisional",
         "status_reason": "coeficientes 0.020, 0.29, 0.5 y 1.10 marcados "
                          "[L-verificar] en Ciencia 5. Los exponentes de "
                          "Dittus-Boelter y las luminosidades SI estan "
                          "verificados (ver columna status). RECALIBRAR contra "
                          "el libro caps. 8-9.",
         "generated_by": "data_pipeline.build()"},
        ["param", "value", "units", "status"],
        rows)


def _build_film_effusion(data_dir: Path) -> None:
    """Parametros y eficacia de la refrigeracion por efusion [Ciencia §5.1]."""
    nan = float("nan")
    rows: list[tuple] = [
        # bandas de diseno geometrico — dato experimental verificado
        ("effusion", "d_hole_mm_min", nan, nan, 0.40, "mm", "verified"),
        ("effusion", "d_hole_mm_max", nan, nan, 1.00, "mm", "verified"),
        ("effusion", "d_hole_mm_printable_min", nan, nan, 0.50, "mm", "verified"),
        ("effusion", "d_hole_mm_printable_max", nan, nan, 0.80, "mm", "verified"),
        ("effusion", "angle_deg_min", nan, nan, 20.0, "deg", "verified"),
        ("effusion", "angle_deg_max", nan, nan, 30.0, "deg", "verified"),
        ("effusion", "pitch_x_over_d_min", nan, nan, 5.0, "-", "verified"),
        ("effusion", "pitch_x_over_d_max", nan, nan, 8.0, "-", "verified"),
        ("effusion", "pitch_y_over_d_min", nan, nan, 4.0, "-", "verified"),
        ("effusion", "pitch_y_over_d_max", nan, nan, 8.0, "-", "verified"),
        ("effusion", "porosity_min", nan, nan, 0.01, "-", "verified"),
        ("effusion", "porosity_max", nan, nan, 0.05, "-", "verified"),
        ("effusion", "n_rows_min", nan, nan, 10.0, "-", "verified"),
        ("effusion", "n_rows_max", nan, nan, 20.0, "-", "verified"),
        ("effusion", "n_rows_saturation", nan, nan, 10.0, "-", "verified"),
        ("effusion", "M_design_min", nan, nan, 0.30, "-", "verified"),
        ("effusion", "M_design_max", nan, nan, 0.80, "-", "verified"),
        ("effusion", "conical_gain_at_M0p25", nan, nan,
         EFFUSION_CONICAL_GAIN, "-", "verified"),
        # eficacia global por esquema de refrigeracion — dato verificado
        ("bare_liner", "eta_cool_min", nan, nan, 0.10, "-", "verified"),
        ("bare_liner", "eta_cool_max", nan, nan, 0.25, "-", "verified"),
        ("slot_film", "eta_cool_min", nan, nan, 0.40, "-", "verified"),
        ("slot_film", "eta_cool_max", nan, nan, 0.50, "-", "verified"),
        ("effusion", "eta_cool_min", nan, nan, 0.50, "-", "verified"),
        ("effusion", "eta_cool_max", nan, nan, 0.70, "-", "verified"),
        ("impingement_effusion", "eta_cool_min", nan, nan, 0.65, "-", "verified"),
        ("impingement_effusion", "eta_cool_max", nan, nan, 0.80, "-", "verified"),
        ("transpiration_lattice", "eta_cool_min", nan, nan, 0.80, "-", "verified"),
        ("transpiration_lattice", "eta_cool_max", nan, nan, 0.95, "-", "verified"),
    ]
    # mapa eta_cool(M, angulo) — forma de la casa, PROVISIONAL
    for ang in (20.0, 30.0, 45.0, 60.0):
        fa = EFFUSION_ANGLE_FACTOR[ang]
        for M in (0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00):
            eta = EFFUSION_ETA_SAT * (1.0 - math.exp(-M / EFFUSION_M_SCALE)) * fa
            rows.append(("effusion_cylindrical", "eta_cool", M, ang, eta, "-",
                         "provisional"))
    _write_csv(
        data_dir / "lefebvre" / "film_effusion.csv",
        ["Refrigeracion por pelicula y efusion: bandas de diseno verificadas",
         "(experimentales) + mapa de eficacia adiabatica frente a la relacion",
         "de soplado M = rho_c·U_c/(rho_g·U_g) y al angulo del orificio.",
         "Formato largo (tidy): filas de banda llevan M y angle_deg = nan; las",
         "filas del mapa llevan param = 'eta_cool' con M y angle_deg definidos.",
         "eta_cool = (T_g - T_w)/(T_g - T_c).",
         "Los orificios CONICOS (triviales en AM) ganan un factor",
         "conical_gain_at_M0p25 = 1.25 sobre los cilindricos a M = 0.25 (dato",
         "verificado); aplicar sobre el mapa cilindrico y saturar en el maximo",
         "de la banda del esquema.",
         "30 grados supera a 60 grados en eficacia adiabatica (confirmado): por",
         "eso f(angulo) del mapa decrece con el angulo.",
         "A M bajo el refrigerante queda adherido; a M alto hay blow-off local",
         "que mejora aguas abajo - por eso el mapa satura, no decrece.",
         "AVISO: el mapa es de la casa. La forma cerrada de la eficacia de",
         "efusion NO se pudo verificar en la investigacion (hueco declarado)."],
        {"source": "Quasar_PhyCB_Science.md 5.1 y EstadoDelArte 4.5 (bandas "
                   "experimentales); mapa eta_cool(M, angulo) sin fuente "
                   "primaria",
         "status": "verified",
         "status_rowcol": "la columna 'status' manda por fila: las bandas son "
                          "dato experimental verificado, el mapa eta_cool es "
                          "provisional",
         "model": f"eta_cool = {EFFUSION_ETA_SAT:.2f}*(1 - "
                  f"exp(-M/{EFFUSION_M_SCALE:.2f}))*f(angle)",
         "generated_by": "data_pipeline.build()"},
        ["scheme", "param", "M", "angle_deg", "value", "units", "status"],
        rows)


def _build_lbo_constant(data_dir: Path) -> None:
    """Constante de phi_LBO, margenes y datos de encendido [Ciencia §6]."""
    rows = [
        ("C_LBO_SI", LBO_CONST_SI, "SI", "provisional"),
        ("exp_P3", -1.30, "-", "verified"),
        ("exp_T3_scale_K", 300.0, "K", "verified"),
        ("exp_SMD", 2.00, "-", "verified"),
        ("LBO_margin_min", 0.20, "-", "provisional"),
        ("accuracy_lefebvre", 0.45, "-", "verified"),
        ("accuracy_flame_volume", 0.15, "-", "verified"),
        ("Da_lbo_min", 0.30, "-", "verified"),
        ("Da_lbo_max", 0.80, "-", "verified"),
        ("Da_design", 20.0, "-", "verified"),
        ("gas_fuel_evap_term", 0.0, "-", "verified"),
        ("E_spark_J_min", 1.0, "J", "verified"),
        ("E_spark_J_max", 5.0, "J", "verified"),
        ("x_spark_over_Lpz_min", 0.30, "-", "verified"),
        ("x_spark_over_Lpz_max", 0.50, "-", "verified"),
        ("phi_local_ignition_min", 0.70, "-", "verified"),
        ("phi_local_ignition_max", 1.30, "-", "verified"),
        ("E_min_prefactor", math.pi / 6.0, "-", "verified"),
        ("E_min_exp_SMD", 3.00, "-", "verified"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "lbo_constant.csv",
        ["Extincion pobre (LBO) y encendido.",
         "Correlacion de Lefebvre para llamas de spray:",
         "   phi_LBO = C_LBO_SI · (A_ref/V_PZ)",
         "             · (mdot_A / (P3^-1.30_inv · exp(T3/300)))",
         "             · (D32^2/lambda_eff) · (1/LHV)",
         "   -- con P3 elevado a exp_P3 = -1.30 en el denominador, es decir",
         "      el termino es mdot_A / (P3^1.30 · exp(T3/300)).",
         "Lecturas de diseno: P3^-1.3 (la estabilidad se degrada a baja presion,",
         "critico a pi ~ 3); D32^2 (la atomizacion es LA palanca a escala",
         "pequena); combustible gaseoso -> el termino de evaporacion desaparece",
         "(gas_fuel_evap_term = 0) y el margen es estructuralmente distinto.",
         "En L1 el margen se mide por numero de Damkohler: Da ~ 0.3-0.8 en LBO",
         "frente a Da ~ 20 en punto de diseno (dato verificado).",
         "Encendido (Ballal & Lefebvre):",
         "   E_min = (pi/6)·rho_g·cp·(T_st - T0)·d_q^3,  E_min ~ D32^3.",
         "Un factor 2 en SMD son 8 en energia de chispa: por eso los",
         "microturborreactores usan encendido asistido.",
         "AVISO: la constante global C_LBO_SI esta calibrada a UN punto ancla."],
        {"source": "Quasar_PhyCB_Science.md 6.1 y 6.2 -> Lefebvre & Ballal "
                   "3.a ed. cap. 5; precisiones y Da en EstadoDelArte 1",
         "status": "provisional",
         "status_reason": "la constante global de phi_LBO esta marcada "
                          "[L-verificar] en Ciencia 6.1 y aqui esta calibrada "
                          "a un unico caso ancla (REF_CB10 -> phi_LBO = 0.50). "
                          "Precision declarada de la correlacion: +-45 %. Los "
                          "exponentes y los datos de encendido SI estan "
                          "verificados (ver columna status). RECALIBRAR en "
                          "WAVE7 contra banco propio.",
         "calibration": "A_ref/V_PZ=33.3 1/m, mdot_A=0.13 kg/s, P3=250000 Pa, "
                        "T3=430 K, D32=50e-6 m, lambda_eff=8e-7 m2/s, "
                        "LHV=43.1e6 J/kg -> phi_LBO=0.50",
         "generated_by": "data_pipeline.build()"},
        ["param", "value", "units", "status"],
        rows)


def _build_atomization_coeffs(data_dir: Path) -> None:
    """Coeficientes de las correlaciones de SMD [Ciencia §7]."""
    rows = [
        # --- simplex (presion-swirl) [7.1]
        ("simplex_lefebvre", "C", 2.25, "SI", "provisional"),
        ("simplex_lefebvre", "exp_sigma", 0.25, "-", "provisional"),
        ("simplex_lefebvre", "exp_mu_l", 0.25, "-", "provisional"),
        ("simplex_lefebvre", "exp_mdot_l", 0.25, "-", "provisional"),
        ("simplex_lefebvre", "exp_dP_l", -0.50, "-", "provisional"),
        ("simplex_lefebvre", "exp_rho_a", -0.25, "-", "provisional"),
        ("simplex_radcliffe", "C", 7.30, "mixed", "provisional"),
        ("simplex_radcliffe", "exp_sigma", 0.60, "-", "provisional"),
        ("simplex_radcliffe", "exp_nu", 0.20, "-", "provisional"),
        ("simplex_radcliffe", "exp_mdot", 0.25, "-", "provisional"),
        ("simplex_radcliffe", "exp_dP", -0.40, "-", "provisional"),
        ("simplex_jasuja", "C", 4.40, "mixed", "provisional"),
        ("simplex_jasuja", "exp_sigma", 0.60, "-", "provisional"),
        ("simplex_jasuja", "exp_nu", 0.16, "-", "provisional"),
        ("simplex_jasuja", "exp_mdot", 0.22, "-", "provisional"),
        ("simplex_jasuja", "exp_dP", -0.43, "-", "provisional"),
        # --- airblast prefilming dimensional VERIFICADO [7.2]
        ("prefilm_rizkalla_lefebvre", "C1", 3.33e-3, "SI", "verified"),
        ("prefilm_rizkalla_lefebvre", "C2", 13.0e-3, "SI", "verified"),
        ("prefilm_rizkalla_lefebvre", "exp_visc_group", 0.425, "-", "verified"),
        ("prefilm_rizkalla_lefebvre", "exp_Lc", 0.575, "-", "verified"),
        ("prefilm_elshanawany_lefebvre", "C1", 0.073, "SI", "verified"),
        ("prefilm_elshanawany_lefebvre", "exp_weber_group", 0.60, "-", "verified"),
        ("prefilm_elshanawany_lefebvre", "exp_rho_ratio", 0.10, "-", "verified"),
        ("prefilm_elshanawany_lefebvre", "exp_Dp", 0.40, "-", "verified"),
        ("prefilm_elshanawany_lefebvre", "C2", 0.0015, "SI", "verified"),
        ("prefilm_nondimensional", "C1", 0.33, "-", "provisional"),
        ("prefilm_nondimensional", "C2", 0.068, "-", "provisional"),
        ("plain_jet_rizk_lefebvre", "C1", 0.48, "-", "provisional"),
        ("plain_jet_rizk_lefebvre", "C2", 0.15, "-", "provisional"),
        ("gepperth_2013", "C", 12.08, "-", "verified"),
        ("gepperth_2013", "exp_We", -0.31, "-", "verified"),
        ("gepperth_2013", "exp_Re", -0.28, "-", "verified"),
        ("gepperth_2013", "exp_rho_ratio", -0.037, "-", "verified"),
        ("gepperth_2013", "exp_ha_over_delta", 0.27, "-", "verified"),
        ("chaussonnet_2016", "C", 1.69, "-", "verified"),
        ("chaussonnet_2016", "U_ref_ms", 0.70, "m/s", "verified"),
        ("airblast", "ALR_design_min", 2.0, "-", "verified"),
        ("airblast", "ALR_design_max", 6.0, "-", "verified"),
        ("airblast", "exp_U_air", -1.20, "-", "verified"),
        ("airblast", "U_air_design_min_ms", 60.0, "m/s", "verified"),
        ("airblast", "U_air_design_max_ms", 100.0, "m/s", "verified"),
        # --- vaporizador clase KJ66 [7.3] — dato verificado
        ("vaporizer_kj66", "n_tubes_min", 6.0, "-", "verified"),
        ("vaporizer_kj66", "n_tubes_max", 12.0, "-", "verified"),
        ("vaporizer_kj66", "tube_length_mm", 64.0, "mm", "verified"),
        ("vaporizer_kj66", "elbow_deg", 30.0, "deg", "verified"),
        ("vaporizer_kj66", "area_holes_over_tube_min", 0.25, "-", "verified"),
        ("vaporizer_kj66", "area_holes_over_tube_max", 0.50, "-", "verified"),
        ("vaporizer_kj66", "tau_tube_ms_min", 6.0, "ms", "verified"),
        ("vaporizer_kj66", "tau_tube_ms_max", 8.0, "ms", "verified"),
        ("vaporizer_kj66", "d50_um_max", 50.0, "um", "verified"),
        ("vaporizer_kj66", "T_tube_wall_K_max", 1173.15, "K", "verified"),
        # --- micromix H2 [7.4] — dato verificado
        ("micromix_h2", "J_min", 1.25, "-", "verified"),
        ("micromix_h2", "J_max", 2.44, "-", "verified"),
        ("micromix_h2", "d_hole_mm_min", 0.30, "mm", "verified"),
        ("micromix_h2", "d_hole_mm_max", 1.00, "mm", "verified"),
        ("micromix_h2", "d_hole_mm_typ", 0.50, "mm", "verified"),
        ("micromix_h2", "kW_per_injector_min", 1.0, "kW", "verified"),
        ("micromix_h2", "kW_per_injector_max", 3.0, "kW", "verified"),
        ("micromix_h2", "phi_max", 0.30, "-", "verified"),
        ("micromix_h2", "U_air_min_ms", 50.0, "m/s", "verified"),
        ("micromix_h2", "T_wall_K_max", 600.0, "K", "verified"),
        # --- comun
        ("jet_penetration", "C", 1.15, "-", "verified"),
        ("jet_penetration", "Ymax_over_DL_single_row", 0.50, "-", "verified"),
        ("jet_penetration", "Ymax_over_DL_two_rows", 0.33, "-", "verified"),
        ("flow_number", "definition_is_mdot_over_sqrt_rho_dP", 1.0, "-",
         "verified"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "atomization_coeffs.csv",
        ["Coeficientes de las correlaciones de diametro medio de Sauter (SMD)",
         "y datos de diseno de cada familia de inyector (gate theta[10]:",
         "0=airblast, 1=simplex, 2=vaporizador, 3=micromix).",
         "Formas (SI salvo donde 'units' dice mixed):",
         "  simplex Lefebvre:  SMD = C·sigma^0.25·mu_l^0.25·mdot_l^0.25",
         "                           ·dP_l^-0.5·rho_a^-0.25",
         "  prefilming Rizkalla & Lefebvre (1975), DIMENSIONAL - USAR ESTA:",
         "     SMD = C1·sqrt(sigma·rho_l·Lc)/(rho_g·U_g)·(1 + mdot_l/mdot_g)",
         "         + C2·(mu_l^2/(sigma·rho_l))^0.425·Lc^0.575",
         "               ·(1 + mdot_l/mdot_g)^2",
         "  prefilming El-Shanawany & Lefebvre (1980):",
         "     SMD = [C1·(sigma/(rho_g·U_g^2))^0.6·(rho_l/rho_g)^0.1·Dp^0.4",
         "            + C2·sqrt(mu_l^2·Dp/(sigma·rho_l))]·(1 + mdot_l/mdot_g)",
         "  penetracion de chorro:  Y_max/d_j = C·sqrt(J).",
         "  flow number:  FN = mdot_L/sqrt(rho_L·dP_L)  [m2]  (forma SI).",
         "Decision de diseno de Phy-CB: airblast por defecto (SMD independiente",
         "del caudal de combustible, usa la energia del swirler, SMD ~ U_A^-1.2);",
         "el simplex puro es mala eleccion por debajo de ~100 kW porque exigiria",
         "20-60 bar de combustible para SMD ~ 50 um.",
         "Para H2 (rama micromix) el SMD no aplica: es rama gaseosa.",
         "AVISO: la forma ADIMENSIONAL habitual del prefilming (0.33/0.068) NO",
         "pudo verificarse; se conserva por compatibilidad pero no es la de",
         "defecto."],
        {"source": "Quasar_PhyCB_Science.md 7.1-7.4; verificaciones en "
                   "arXiv:1906.04042 (Tabla 7), OA Cranfield de Rizkalla & "
                   "Lefebvre 1975 y El-Shanawany & Lefebvre 1980, ILASS 2023 "
                   "paper 37 (flow number), gtba.co.uk/engine_designs/kj66.htm "
                   "y MDPI Energies 18(23):6168 (micromix)",
         "status": "verified",
         "status_rowcol": "la columna 'status' manda por fila: simplex "
                          "(2.25/7.3/4.4), prefilming adimensional "
                          "(0.33/0.068) y plain-jet (0.48/0.15) son "
                          "PROVISIONALES - cotejar Lefebvre & Ballal 3.a ed. "
                          "caps. 6 y 10 antes de usarlos en un numero de "
                          "reporte",
         "generated_by": "data_pipeline.build()"},
        ["model", "param", "value", "units", "status"],
        rows)


def _build_emissions_coeffs(data_dir: Path) -> None:
    """Coeficientes de las correlaciones de emisiones [Ciencia §8]."""
    rows = [
        # --- EI_NOx Lefebvre [8.1] — coeficiente verificado, unidad de P3 no
        ("nox_lefebvre", "C", 9.0e-8, "mixed", "verified"),
        ("nox_lefebvre", "exp_P3", 1.25, "-", "verified"),
        ("nox_lefebvre", "exp_Tst_coeff", 0.01, "1/K", "verified"),
        # kPa: UNICA unidad que cae en rango en los DOS extremos (ancla
        # REF_CB10 -> 2.00 g/kg; motor grande 30 atm/800 K -> 39.2 g/kg,
        # dentro de la banda 20-40 de la literatura). Pa da 1.1e4 y atm
        # 6.2e-3 en el ancla: exp_P3 = 1.25 amplifica un factor 1000 de
        # unidad en 5.6e3, asi que la evidencia dimensional es concluyente.
        ("nox_lefebvre", "P3_unit_scale_from_Pa", 1.0e-3, "-", "provisional"),
        # --- EI_CO Lefebvre [8.2] — NO verificada
        ("co_lefebvre", "C", 86.0, "mixed", "provisional"),
        ("co_lefebvre", "exp_Tpz_coeff", -0.00345, "1/K", "provisional"),
        ("co_lefebvre", "exp_dp_over_p", -0.50, "-", "provisional"),
        ("co_lefebvre", "exp_P3", -1.50, "-", "provisional"),
        # kPa: ancla 57.6 g/kg y motor grande 8.3 g/kg (EI_CO cae con la
        # potencia, tendencia correcta). Pa da 9e-4 y atm 2.8e4: fuera de
        # toda banda fisica en ambos extremos.
        ("co_lefebvre", "P3_unit_scale_from_Pa", 1.0e-3, "-", "provisional"),
        # --- Rizk & Mongia global [8.3] — verificada
        ("nox_rizk_mongia", "C", 15.0e14, "mixed", "verified"),
        ("nox_rizk_mongia", "exp_tau_group", 0.50, "-", "verified"),
        ("nox_rizk_mongia", "tau_e_factor", 0.50, "-", "verified"),
        ("nox_rizk_mongia", "Ea_over_R_K", 71100.0, "K", "verified"),
        ("nox_rizk_mongia", "exp_P3", -0.05, "-", "verified"),
        ("nox_rizk_mongia", "exp_dp_over_p", -0.50, "-", "verified"),
        # exp_P3 = -0.05 hace la unidad de P3 dimensionalmente INDECIDIBLE
        # con este ensayo (todo el rango Pa<->atm cabe en un factor 1.8);
        # se adopta kPa por coherencia con el EI_NOx de Lefebvre de la
        # MISMA tesis. Lo decisivo son las otras dos unidades.
        ("nox_rizk_mongia", "P3_unit_scale_from_Pa", 1.0e-3, "-", "provisional"),
        # tau en SEGUNDOS, no en ms como dice Ciencia 8.3: con ms el ancla
        # da 1147 g/kg y el motor grande 8447 g/kg (absurdo); con s da 3.63
        # y 26.7 g/kg. El exponente 0.5 convierte el factor 1000 de unidad
        # en 31.6, que es justo lo que separa el absurdo del rango fisico.
        ("nox_rizk_mongia", "tau_unit_scale_from_s", 1.0, "-", "provisional"),
        # dP3/P3 en PORCENTAJE (Ciencia 2.1 tabula esa magnitud en %): el
        # exponente -0.5 aporta el factor 10 que falta. (s, %) es la UNICA
        # de las 4 combinaciones (tau)x(dp/p) que cae en 20-40 en el motor
        # grande; (ms,%) y (s,frac) coinciden en 327 y (ms,frac) en 8447.
        ("nox_rizk_mongia", "dp_over_p_unit_scale_from_frac", 100.0, "-",
         "provisional"),
        # --- Rizk & Mongia zonal [8.3]
        ("nox_rizk_mongia_zonal", "C", 1.0e13, "mixed", "verified"),
        # Pa + tau en s: ancla 3.76 g/kg, motor grande 41.0 g/kg (borde
        # superior de la banda de literatura). Ninguna otra de las 8
        # combinaciones (P3)x(tau) queda dentro de la banda de
        # plausibilidad en los dos extremos: kPa/atm/psia hunden el
        # resultado 4-7 ordenes de magnitud porque P3^1.4 va elevado a aa.
        ("nox_rizk_mongia_zonal", "P3_unit_scale_from_Pa", 1.0, "-",
         "provisional"),
        ("nox_rizk_mongia_zonal", "tau_unit_scale_from_s", 1.0, "-",
         "provisional"),
        ("nox_rizk_mongia_zonal", "Ea_over_R_K", 71442.0, "K", "verified"),
        ("nox_rizk_mongia_zonal", "aa_C1", 1.95, "-", "provisional"),
        ("nox_rizk_mongia_zonal", "aa_C2", 5.76, "-", "provisional"),
        ("nox_rizk_mongia_zonal", "pz_lean_C", 7.56, "-", "provisional"),
        ("nox_rizk_mongia_zonal", "pz_lean_exp_phi", -7.20, "-", "provisional"),
        ("nox_rizk_mongia_zonal", "pz_rich_C", 5.21, "-", "provisional"),
        ("nox_rizk_mongia_zonal", "pz_rich_exp_phi", -2.99, "-", "provisional"),
        ("nox_rizk_mongia_zonal", "phi_split", 1.08, "-", "verified"),
        ("nox_rizk_mongia_zonal", "offset", -1.60, "-", "provisional"),
        ("nox_rizk_mongia_zonal", "exp_tau_pz", 0.64, "-", "verified"),
        ("nox_rizk_mongia_zonal", "sz_dz_C", 1.172, "-", "provisional"),
        ("nox_rizk_mongia_zonal", "sz_dz_exp_phi", -4.56, "-", "provisional"),
        ("nox_rizk_mongia_zonal", "exp_tau_sz_dz", 0.876, "-", "verified"),
        # --- Odgers & Kretschmer [8.4] — verificada
        ("nox_odgers", "C", 29.0, "mixed", "verified"),
        ("nox_odgers", "Ea_over_R_K", 21670.0, "K", "verified"),
        ("nox_odgers", "exp_P3", 0.66, "-", "verified"),
        ("nox_odgers", "tau_sat_coeff", 250.0, "1/s", "verified"),
        # Pa, NO atm: con atm el ancla da 4.4e-3 g/kg y el motor grande
        # 1.2e-2 (cuatro ordenes por debajo de la banda 20-40); con Pa da
        # 8.9 y 23.5 g/kg. kPa (0.093/0.25) y psia (0.026/0.069) tampoco
        # llegan. La "mayoria de las citas" que decia Ciencia 8.4 NO
        # sobrevive al ensayo dimensional en los dos extremos.
        ("nox_odgers", "P3_unit_scale_from_Pa", 1.0, "-", "provisional"),
        ("nox_odgers", "tau_unit_scale_from_s", 1.0, "-", "verified"),
        # --- Laviolette & Perez [8.4] — verificada
        ("nox_laviolette", "C", 1.902e5, "mixed", "verified"),
        ("nox_laviolette", "exp_P3", 0.4048, "-", "verified"),
        ("nox_laviolette", "tau_sat_coeff", 480.0, "1/s", "verified"),
        ("nox_laviolette", "Ea_over_R_K", 37902.0, "K", "verified"),
        # Pa es la MEJOR de las 8 combinaciones (ancla 3.17, motor grande
        # 4.26 g/kg) pero NINGUNA alcanza la banda 20-40 del motor grande:
        # la correlacion subestima un factor 5-9 al escalar por su
        # exponente de presion bajisimo (0.4048). Queda en el abanico,
        # dentro de la banda de plausibilidad, y marcada PROVISIONAL.
        ("nox_laviolette", "P3_unit_scale_from_Pa", 1.0, "-", "provisional"),
        ("nox_laviolette", "tau_unit_scale_from_s", 1.0, "-", "verified"),
        # --- contrastes empiricos [8.4]
        ("nox_nepp", "C", 33.2, "psia,degR", "verified"),
        ("nox_nepp", "P3_ref_psia", 432.7, "psia", "verified"),
        ("nox_nepp", "exp_P3", 0.40, "-", "verified"),
        ("nox_nepp", "T3_ref_degR", 1487.6, "degR", "verified"),
        ("nox_nepp", "T3_scale_degR", 350.0, "degR", "verified"),
        ("nox_gasturb", "C", 32.0, "kPa,K", "verified"),
        ("nox_gasturb", "P3_ref_kPa", 2965.0, "kPa", "verified"),
        ("nox_gasturb", "exp_P3", 0.40, "-", "verified"),
        ("nox_gasturb", "T3_ref_K", 826.0, "K", "verified"),
        ("nox_gasturb", "T3_scale_K", 194.0, "K", "verified"),
        # --- evaporacion (Rizk & Mongia / Lanewala) [8.3, 9.2]
        ("evaporation", "lambda_e_C", 8.0, "-", "verified"),
        ("evaporation", "lambda_e_Re_coeff", 0.30, "-", "verified"),
        ("evaporation", "lambda_e_exp_Re", 0.50, "-", "verified"),
        ("evaporation", "lambda_e_exp_Pr", 0.33, "-", "verified"),
        ("evaporation", "far_corr_slope", -0.0295, "-", "verified"),
        ("evaporation", "far_corr_offset", 0.45, "-", "verified"),
        # --- honestidad
        ("honesty", "uncertainty_band_min", 0.30, "-", "verified"),
        ("honesty", "uncertainty_band_max", 0.50, "-", "verified"),
        # --- pantalla de plausibilidad del abanico de EI_NOx
        # Banda de un EI_NOx de combustor REAL. Suelo 0.01 g/kg: por
        # debajo esta el micromix H2 de un digito de ppm (reference_micro
        # NOx_ppm15O2_gas_min = 9 ppm ~ 0.1 g/kg) con un orden de margen.
        # Techo 200 g/kg: dos veces largos por encima del peor combustor
        # de la banda kerosene (NOx_ppm15O2_kero_max = 80 ppm ~ 5 g/kg) y
        # cinco veces el maximo de motor grande de la literatura (40 g/kg).
        # Toda correlacion fuera de la banda SE SIGUE REPORTANDO pero
        # queda fuera del min/max y de los estadisticos del abanico.
        ("plausibility", "EI_NOx_min", 0.01, "g/kg", "verified"),
        ("plausibility", "EI_NOx_max", 200.0, "g/kg", "verified"),
        ("reference_micro", "NOx_ppm15O2_gas_min", 9.0, "ppm", "verified"),
        ("reference_micro", "NOx_ppm15O2_gas_max", 25.0, "ppm", "verified"),
        ("reference_micro", "CO_ppm15O2_gas_min", 10.0, "ppm", "verified"),
        ("reference_micro", "CO_ppm15O2_gas_max", 50.0, "ppm", "verified"),
        ("reference_micro", "NOx_ppm15O2_kero_min", 35.0, "ppm", "verified"),
        ("reference_micro", "NOx_ppm15O2_kero_max", 80.0, "ppm", "verified"),
    ]
    _write_csv(
        data_dir / "lefebvre" / "emissions_coeffs.csv",
        ["Coeficientes de las correlaciones de emisiones de nivel preliminar.",
         "TODAS se calculan y se reportan con su nombre: el usuario ve el",
         "abanico, no un numero unico.",
         "  EI_NOx Lefebvre: C·P3^1.25·V_c·exp(0.01·T_st)/(mdot_A·T_PZ)  [g/kg]",
         "  EI_CO  Lefebvre: C·mdot_A·T_PZ·exp(-0.00345·T_PZ)",
         "                   / ((V_c - V_e)·(dp/p)^0.5·P3^1.5)",
         "  Rizk & Mongia global: C·(tau - 0.5·tau_e)^0.5·exp(-71100/T_st)",
         "                        ·P3^-0.05·(dP3/P3)^-0.5",
         "  Odgers: C·exp(-21670/T_f)·P3^0.66·[1 - exp(-250·tau)]",
         "  Laviolette: C·P3^0.4048·[1 - exp(-480·tau)]/exp(37902/T_f)",
         "TRAMPA DE UNIDADES (auditoria dimensional 2026-07-30): la unidad de",
         "P3 NO es explicita en las fuentes verificadas y la de tau de Rizk &",
         "Mongia estaba mal declarada. Las tres familias de factores",
         "  P3_unit_scale_from_Pa · tau_unit_scale_from_s ·",
         "  dp_over_p_unit_scale_from_frac",
         "se fijaron por ENSAYO SISTEMATICO en DOS extremos: el caso ancla",
         "REF_CB10 (250 kPa, 430 K, 0.13 kg/s, tau 6.05 ms) y un motor grande",
         "(30 atm, 800 K, tau 3 ms, T_st 2600 K, dp/p 5 %, donde la literatura",
         "da EI_NOx = 20-40 g/kg). Se conserva la combinacion que cae en rango",
         "en AMBOS extremos - eso es evidencia dimensional, no ajuste. El",
         "resultado por correlacion esta comentado fila a fila abajo. Siguen",
         "marcadas PROVISIONAL: son inferencia dimensional, no lectura de la",
         "fuente primaria; cerrarlas exige Lefebvre & Ballal 3.a ed. cap. 9.",
         "Abanico resultante en el ancla [g/kg]: lefebvre 2.00 · rizk_mongia",
         "3.63 · zonal 3.76 · odgers 8.91 · laviolette 3.17 (dispersion x4.5,",
         "antes x2.6e5). En el motor grande: 39.2 / 26.7 / 41.0 / 23.5 / 4.3.",
         "Compromiso NOx<->CO capturado explicitamente: NOx ~ +V_c·exp(0.01·T_st)",
         "y CO ~ 1/V_c, luego existe un V_c optimo analitico",
         "d(w1·EI_NOx + w2·EI_CO)/dV_c = 0 - implementarlo en L0.",
         "REGLA DE HONESTIDAD: estas correlaciones dan TENDENCIAS (+-30-50 % en",
         "el mejor caso, incluso con CRN calibrado). El reporte de Phy-CB nunca",
         "presenta NOx/CO como absolutos certificables."],
        {"source": "Quasar_PhyCB_Science.md 8.1-8.4; tesis U. Toronto "
                   "(EI_NOx Lefebvre ec. 1.4, Rizk & Mongia ecs. 3.5/1.5-1.8, "
                   "NEPP/GasTurb), MDPI Appl. Sci. 14:6104 (Odgers, "
                   "Laviolette)",
         "status": "verified",
         "status_rowcol": "la columna 'status' manda por fila: EI_CO "
                          "(86/-0.00345), la funcion de phi de Rizk & Mongia "
                          "zonal y TODOS los *_unit_scale_* son PROVISIONALES "
                          "(evidencia dimensional en dos extremos, no lectura "
                          "de fuente primaria)",
         "generated_by": "data_pipeline.build()"},
        ["model", "param", "value", "units", "status"],
        rows)


def _build_class_table(data_dir: Path) -> None:
    """Tabla de valores de referencia por clase de potencia [EstadoDelArte §7]."""
    names = ("micro_0p1_5kW", "small_5_50kW", "medium_50_250kW")
    # param -> (units, [(min, max) por clase])
    spec: list[tuple[str, str, list[tuple[float, float]]]] = [
        ("power_kW", "kW", [(0.1, 5.0), (5.0, 50.0), (50.0, 250.0)]),
        ("pressure_ratio", "-", [(2.0, 3.5), (3.0, 4.5), (3.5, 5.0)]),
        ("T3_K_unrecuperated", "K", [(400.0, 470.0), (430.0, 500.0),
                                     (450.0, 530.0)]),
        ("T3_K_recuperated", "K", [(float("nan"), float("nan")),
                                   (800.0, 950.0), (900.0, 1100.0)]),
        ("T4_K", "K", [(1100.0, 1250.0), (1150.0, 1300.0), (1200.0, 1350.0)]),
        ("mdot_air_kgs", "kg/s", [(0.05, 0.25), (0.15, 0.60), (0.50, 1.50)]),
        ("FAR_global", "-", [(0.010, 0.018), (0.012, 0.020), (0.012, 0.020)]),
        ("phi_global", "-", [(0.15, 0.27), (0.18, 0.30), (0.18, 0.30)]),
        ("phi_pz", "-", [(0.85, 1.15), (0.80, 1.10), (0.60, 0.90)]),
        ("dp_over_p", "-", [(0.02, 0.04), (0.03, 0.05), (0.03, 0.05)]),
        ("U_ref_ms", "m/s", [(8.0, 20.0), (10.0, 25.0), (15.0, 30.0)]),
        ("tau_ms", "ms", [(4.0, 8.0), (5.0, 10.0), (5.0, 10.0)]),
        ("V_c_L", "L", [(0.05, 0.40), (0.30, 1.50), (1.00, 4.00)]),
        ("D_casing_mm", "mm", [(80.0, 130.0), (120.0, 200.0), (180.0, 300.0)]),
        ("I_MW_m3_atm", "MW/m3/atm", [(25.0, 60.0), (20.0, 50.0), (20.0, 45.0)]),
        ("eta_comb_design", "-", [(0.970, 0.990), (0.990, 0.999),
                                  (0.995, 0.999)]),
        ("eta_comb_idle", "-", [(0.900, 0.960), (0.940, 0.980), (0.960, 0.990)]),
        ("PF", "-", [(0.35, 0.50), (0.25, 0.40), (0.20, 0.35)]),
        ("frac_cool", "-", [(0.00, 0.15), (0.10, 0.25), (0.15, 0.30)]),
        ("T_wall_liner_K", "K", [(900.0, 1050.0), (1000.0, 1150.0),
                                 (1050.0, 1200.0)]),
        ("t_liner_mm", "mm", [(0.50, 0.80), (0.60, 1.00), (0.80, 1.50)]),
        ("d_eff_mm", "mm", [(0.50, 0.70), (0.50, 0.80), (0.60, 1.00)]),
        ("n_eff_holes", "-", [(300.0, 1500.0), (1000.0, 4000.0),
                              (2000.0, 8000.0)]),
        ("d_dilution_mm", "mm", [(3.0, 6.0), (5.0, 9.0), (7.0, 14.0)]),
        ("n_dilution_per_row", "-", [(6.0, 10.0), (8.0, 12.0), (10.0, 16.0)]),
        ("n_injectors", "-", [(1.0, 1.0), (1.0, 6.0), (4.0, 12.0)]),
        ("n_vaporizer_tubes", "-", [(6.0, 12.0), (float("nan"), float("nan")),
                                    (float("nan"), float("nan"))]),
        ("SMD_um", "um", [(40.0, 70.0), (35.0, 60.0), (30.0, 50.0)]),
        ("ALR_airblast", "-", [(2.0, 5.0), (3.0, 6.0), (3.0, 6.0)]),
        ("SN_swirl", "-", [(0.70, 1.00), (0.60, 0.90), (0.60, 0.90)]),
    ]
    rows = []
    for cid, cname in enumerate(names):
        for param, units, bands in spec:
            lo, hi = bands[cid]
            rows.append((cid, cname, param, lo, hi, units, "verified"))
    _write_csv(
        data_dir / "reference" / "class_table.csv",
        ["Tabla de valores de referencia por clase de potencia de microturbina.",
         "Es la ESCALERA DE ANCLAS numero 1 del plan de validacion: cualquier",
         "diseno de Phy-CB que se salga de estas bandas debe justificarse.",
         "class_id: 0 = 0.1-5 kW (microturborreactor), 1 = 5-50 kW,",
         "2 = 50-250 kW.",
         "nan = el parametro no aplica a esa clase (p. ej. no hay recuperador",
         "en la clase 0, ni vaporizadores por encima de la clase 0).",
         "Material de liner por clase (no numerico, vive en",
         "structures_core.MATERIALS): clase 0 IN625/Hastelloy-X, clase 1 IN625",
         "(caso Bath/HiETA), clase 2 Hastelloy X (caso Sierra Turbines).",
         "T3 recuperada de la clase 2: 820 C = 1093 K medidos (HiETA/Bath).",
         "PF medido en combustores pequenos reales: 0.21-0.24."],
        {"source": "Quasar_PhyCB_EstadoDelArte.md 7 (tabla de valores de "
                   "referencia, marcas [W] verificadas en fuente web)",
         "status": "verified",
         "caveat": "las cifras certificadas exactas de Capstone C30/C65 y "
                   "Turbec T100 NO se verificaron (EstadoDelArte 7 y 11, "
                   "prioridad media 4); las emisiones de referencia viven en "
                   "emissions_coeffs.csv (modelo reference_micro).",
         "generated_by": "data_pipeline.build()"},
        ["class_id", "class_name", "param", "min", "max", "units", "status"],
        rows)


# ----------------------------------------------------------------------------
# Build — tablas de T_ad (fallback de termoquímica sin Cantera) [Ciencia §9.3]
# ----------------------------------------------------------------------------

def _cantera():
    """Import perezoso de Cantera. Devuelve el módulo o None (nunca lanza)."""
    try:
        import cantera as ct  # noqa: PLC0415 — import perezoso deliberado
        return ct
    except Exception:
        return None


#: Mecanismos bundled con Cantera 3.2 usados para el EQUILIBRIO (no cinética).
#: Nota: para `equilibrate('HP')` solo importa la termoquímica de las especies,
#: no las reacciones — por eso las soluciones se construyen con
#: `kinetics='none'` (2 órdenes de magnitud más rápido con el mecanismo de C12).
_CT_MECH_MAIN = "gri30.yaml"
_CT_MECH_C12 = "nDodecane_Reitz.yaml"


def _ct_equilibrium_gas(ct, fuel: str):
    """Construye el gas de equilibrio y la composición molar del combustible.

    Devuelve `(gas, fuel_X, descripcion)` o `None` si el mecanismo no está.
    Selección de especies (criterio CEA: todo lo que el equilibrio puebla):

    * `jet_a`   — GRI-Mech 3.0 completo (incluye química de N ⇒ NO en equilibrio,
      como hace CEA) + `c12h26` y `A1` (benceno) tomados de `nDodecane_Reitz`.
    * `propane` — GRI-Mech 3.0 completo (trae C3H8).
    * `h2`      — subconjunto H/O/N/Ar de GRI-Mech 3.0 (19 especies).
    """
    try:
        sp_main = {s.name: s for s in ct.Species.list_from_file(_CT_MECH_MAIN)}
    except Exception:
        return None
    if fuel == "jet_a":
        try:
            sp_c12 = {s.name: s
                      for s in ct.Species.list_from_file(_CT_MECH_C12)}
        except Exception:
            return None
        if "c12h26" not in sp_c12 or "A1" not in sp_c12:
            return None
        species = list(sp_main.values()) + [sp_c12["c12h26"], sp_c12["A1"]]
        fuel_X = {"c12h26": 1.0 - JET_A_X_A1, "A1": JET_A_X_A1}
        desc = (f"{_CT_MECH_MAIN} + c12h26/A1 de {_CT_MECH_C12}; surrogate "
                f"molar n-dodecano {1.0 - JET_A_X_A1:.4f} + benceno "
                f"{JET_A_X_A1:.4f} (H/C = 1.91 exacto)")
    elif fuel == "propane":
        if "C3H8" not in sp_main:
            return None
        species = list(sp_main.values())
        fuel_X = {"C3H8": 1.0}
        desc = f"{_CT_MECH_MAIN} (C3H8 nativo)"
    elif fuel == "h2":
        species = [s for s in sp_main.values()
                   if set(s.composition) <= {"H", "O", "N", "Ar"}]
        if not any(s.name == "H2" for s in species):
            return None
        fuel_X = {"H2": 1.0}
        desc = f"{_CT_MECH_MAIN} restringido al subconjunto H/O/N/Ar"
    else:
        return None
    try:
        gas = ct.Solution(thermo="ideal-gas", kinetics="none", species=species)
    except Exception:
        return None
    return gas, fuel_X, desc


_CT_OXIDIZER = f"O2:1.0, N2:{AIR_N2_PER_O2}"


def _ct_lhv(ct, gas, fuel_X: dict) -> float:
    """LHV [J/kg] del surrogate según la termoquímica del propio mecanismo.

    Combustión completa a 298.15 K con agua en fase gas (definición de LHV).
    """
    gas.TPX = 298.15, ct.one_atm, fuel_X
    mw_f = gas.mean_molecular_weight
    h_f_molar = gas.enthalpy_mole
    nC = sum(v * gas.species(k).composition.get("C", 0.0)
             for k, v in fuel_X.items())
    nH = sum(v * gas.species(k).composition.get("H", 0.0)
             for k, v in fuel_X.items())
    prod = {}
    if nC > 0.0:
        prod["CO2"] = nC
    prod["H2O"] = nH / 2.0
    n_p = sum(prod.values())
    gas.TPX = 298.15, ct.one_atm, prod
    h_p_molar = gas.enthalpy_mole
    # O2 y N2 tienen entalpía de formación nula por definición.
    d_h = n_p * h_p_molar - h_f_molar        # J por kmol de combustible
    return -d_h / mw_f


def _ct_state(ct, gas, fuel_X, phi, T3, P3, dh_lhv):
    """Equilibrio 'HP' con corrección de entalpía al LHV de contrato."""
    gas.set_equivalence_ratio(phi, fuel_X, _CT_OXIDIZER, basis="mole")
    gas.TP = T3, P3
    y_fuel = sum(float(gas.Y[gas.species_index(k)]) for k in fuel_X)
    gas.HP = gas.enthalpy_mass - y_fuel * dh_lhv, P3
    gas.equilibrate("HP")
    return float(gas.T), float(gas.cp_mass), float(gas.mean_molecular_weight)


def _mw_products(fuel: str, phi: float) -> float:
    """Masa molar de los productos [kg/kmol] por balance atómico.

    Combustión completa a CO₂/H₂O si φ ≤ 1; a φ > 1 la fracción no quemable
    va a CO + H₂, lo que da el MISMO recuento de moles (nC + nH/2) y por eso
    una sola expresión sirve en las dos ramas. Error frente al equilibrio de
    Cantera: medio 0.4–0.6 %, máximo 4 % (medido 2026-07-30).
    """
    nC, nH = FUEL_ATOMS_CH[fuel]
    mw_f = nC * MW_C + nH * MW_H
    n_o2_st = nC + nH / 4.0
    n_o2 = n_o2_st / max(phi, 1.0e-9)
    mass = mw_f + n_o2 * (MW_O2 + AIR_N2_PER_O2 * MW_N2)
    moles = nC + nH / 2.0 + max(0.0, n_o2 - n_o2_st) + AIR_N2_PER_O2 * n_o2
    return mass / moles


def _cp_products(T: float, mw: float = 28.6) -> float:
    """cp medio de gases de combustión [J/kg/K] — ajuste de la casa.

    Base ajustada a productos de queroseno (MW ≈ 28.6) y escalada por 1/MW,
    que es como cp másico depende de la composición a cp molar comparable.
    Solo lo usa el modelo analítico de respaldo (sin Cantera).
    """
    return (950.0 + 0.30 * T - 4.0e-5 * T * T) * (28.6 / mw)


def _tad_ideal(fuel: str, phi: float, T3: float, mw: float) -> float:
    """T de llama SIN disociación: balance de entalpía con cp medio."""
    far = phi * _far_stoich(fuel)
    y_fuel = far / (1.0 + far)
    burned = min(1.0, 1.0 / max(phi, 1.0e-9))
    rich = max(TAD_RICH_FLOOR,
               1.0 - TAD_RICH_A * _carbon_o2_fraction(fuel)
               * max(0.0, phi - 1.0) ** TAD_RICH_B)
    q = y_fuel * FUEL_LHV_J_KG[fuel] * burned * rich   # J por kg de mezcla
    T = T3 + 1500.0
    for _ in range(60):                            # bucle ACOTADO (sin while)
        t_new = T3 + q / _cp_products(0.5 * (T3 + T), mw)
        if abs(t_new - T) < 1.0e-6:
            T = t_new
            break
        T = 0.5 * (T + t_new)                      # sub-relajación
    return T


def _tad_calibrate_A(fuel: str) -> float:
    """Coeficiente A del déficit que hace pasar el modelo por su ancla."""
    t_ideal = _tad_ideal(fuel, 1.0, 300.0, _mw_products(fuel, 1.0))
    exc = max(1.0, t_ideal - TAD_DISS_ONSET_K)
    sat = math.sqrt(exc * exc + TAD_DISS_E0_K ** 2) - TAD_DISS_E0_K
    a = max(0.0, t_ideal - TAD_ANCHOR_K[fuel]) / max(sat, 1.0e-9)
    return min(a, TAD_DISS_A_MAX)                  # blinda dT_ad/dT3 > 0


def _tad_analytic(fuel: str, phi: float, T3: float, P3: float,
                  a_diss: float) -> tuple[float, float, float]:
    """Modelo analítico de respaldo de T_ad (solo si falta Cantera).

    Balance de entalpía con cp medio dependiente de T y de la composición,
    atenuación rica proporcional al carbono del combustible y déficit por
    disociación en «palo de hockey» calibrado a un único punto ancla por
    combustible. PROVISIONAL por construcción: error frente a Cantera medio
    1.4–2.2 % y máximo 5.5 % sobre la rejilla completa (medido 2026-07-30).
    """
    mw = _mw_products(fuel, phi)
    t_ideal = _tad_ideal(fuel, phi, T3, mw)
    exc = max(0.0, t_ideal - TAD_DISS_ONSET_K)
    deficit = (a_diss
               * (math.sqrt(exc * exc + TAD_DISS_E0_K ** 2) - TAD_DISS_E0_K)
               * (P3 / P_ATM) ** TAD_DISS_P_EXP)
    t_ad = max(T3, t_ideal - deficit)
    return t_ad, _cp_products(t_ad, mw), mw


def _interp_tad(grid: np.ndarray, phi: float, t3: float, p3: float) -> float:
    """Interpolación trilineal en (phi, T3, log P3) — la que se documenta."""
    axes = (np.asarray(TAD_PHI_GRID), np.asarray(TAD_T3_GRID),
            np.log(np.asarray(TAD_P3_GRID)))
    vals = (phi, t3, math.log(p3))
    idx, frac = [], []
    for ax, v in zip(axes, vals):
        i = int(np.clip(np.searchsorted(ax, v) - 1, 0, len(ax) - 2))
        idx.append(i)
        frac.append((v - ax[i]) / (ax[i + 1] - ax[i]))
    out = 0.0
    for di, wa in ((0, 1.0 - frac[0]), (1, frac[0])):
        for dj, wb in ((0, 1.0 - frac[1]), (1, frac[1])):
            for dk, wc in ((0, 1.0 - frac[2]), (1, frac[2])):
                out += wa * wb * wc * grid[idx[0] + di, idx[1] + dj,
                                           idx[2] + dk]
    return float(out)


def _build_tad_table(data_dir: Path, fuel: str, ct) -> str:
    """Genera `thermo/tad_fit_<fuel>.csv`. Devuelve la etiqueta de estado."""
    n_phi, n_t3, n_p3 = len(TAD_PHI_GRID), len(TAD_T3_GRID), len(TAD_P3_GRID)
    grid_t = np.zeros((n_phi, n_t3, n_p3))
    grid_cp = np.zeros_like(grid_t)
    grid_mw = np.zeros_like(grid_t)

    setup = _ct_equilibrium_gas(ct, fuel) if ct is not None else None
    if setup is not None:
        gas, fuel_X, mech_desc = setup
        try:
            lhv_surr = _ct_lhv(ct, gas, fuel_X)
            dh_lhv = lhv_surr - FUEL_LHV_J_KG[fuel]
            for i, phi in enumerate(TAD_PHI_GRID):
                for j, t3 in enumerate(TAD_T3_GRID):
                    for k, p3 in enumerate(TAD_P3_GRID):
                        (grid_t[i, j, k], grid_cp[i, j, k],
                         grid_mw[i, j, k]) = _ct_state(ct, gas, fuel_X, phi,
                                                       t3, p3, dh_lhv)
            # error de interpolación medido en los puntos medios de la rejilla
            errs = []
            for i in range(n_phi - 1):
                for j in range(0, n_t3 - 1, 2):
                    for k in range(0, n_p3 - 1, 2):
                        phi = 0.5 * (TAD_PHI_GRID[i] + TAD_PHI_GRID[i + 1])
                        t3 = 0.5 * (TAD_T3_GRID[j] + TAD_T3_GRID[j + 1])
                        p3 = math.exp(0.5 * (math.log(TAD_P3_GRID[k])
                                             + math.log(TAD_P3_GRID[k + 1])))
                        ref = _ct_state(ct, gas, fuel_X, phi, t3, p3,
                                        dh_lhv)[0]
                        errs.append(abs(_interp_tad(grid_t, phi, t3, p3) - ref)
                                    / ref)
            err_max = float(np.max(errs))
            gas.set_equivalence_ratio(1.0, fuel_X, _CT_OXIDIZER, basis="mole")
            z_st = float(gas.mixture_fraction(fuel_X, _CT_OXIDIZER,
                                              basis="mole"))
            far_st = z_st / (1.0 - z_st)
            status, reason = "verified", ""
            meta_extra = {
                "generator": f"Cantera {ct.__version__} equilibrate('HP')",
                "mechanism": mech_desc,
                "lhv_surrogate_J_kg": f"{lhv_surr:.6g}",
                "lhv_contract_J_kg": f"{FUEL_LHV_J_KG[fuel]:.6g}",
                "enthalpy_correction_J_kg_fuel": f"{dh_lhv:.6g}",
                "far_stoich_mass": f"{far_st:.6g}",
                "interp_scheme": "trilineal en (phi, T3_K, log P3_Pa)",
                "interp_max_rel_err": f"{err_max:.3e}",
                "interp_n_probes": str(len(errs)),
            }
            doc_extra = [
                "Generada CON Cantera: equilibrio 'HP' desde (phi, T3, P3) con",
                "el combustible en fase VAPOR a T3 (hipotesis estandar de L0).",
                "La entalpia de los reactivos se corrige para que el surrogate",
                "entregue EXACTAMENTE el LHV de contrato: asi T_ad y el balance",
                "de FAR del seam de ciclo hablan del mismo combustible.",
                "Cantera equilibrate('HP') coincide con NASA CEA con desvio",
                "0.06-3.5 % en T_ad (arXiv:2503.11826).",
            ]
        except Exception as exc:      # degradación elegante, nunca excepción
            setup = None
            print(f"[build] Cantera fallo en '{fuel}' ({type(exc).__name__}: "
                  f"{exc}) -> modelo analitico de respaldo")
    if setup is None:
        a_diss = _tad_calibrate_A(fuel)
        for i, phi in enumerate(TAD_PHI_GRID):
            for j, t3 in enumerate(TAD_T3_GRID):
                for k, p3 in enumerate(TAD_P3_GRID):
                    (grid_t[i, j, k], grid_cp[i, j, k],
                     grid_mw[i, j, k]) = _tad_analytic(fuel, phi, t3, p3,
                                                       a_diss)
        far_st = _far_stoich(fuel)
        status = "provisional"
        reason = ("generada SIN Cantera con el modelo analitico de respaldo "
                  "(_tad_analytic): balance de entalpia con cp medio, "
                  "atenuacion rica por fraccion de carbono y deficit por "
                  "disociacion calibrado a un unico punto ancla por "
                  "combustible. Error frente a Cantera: medio 1.4-2.2 %, "
                  "maximo 5.5 %. REGENERAR con Cantera instalado "
                  "(conda-forge, NUNCA pip en Windows).")
        meta_extra = {
            "generator": "data_pipeline._tad_analytic (respaldo sin Cantera)",
            "mechanism": "ninguno (modelo analitico)",
            "lhv_contract_J_kg": f"{FUEL_LHV_J_KG[fuel]:.6g}",
            "far_stoich_mass": f"{far_st:.6g}",
            "anchor_T_ad_K": f"{TAD_ANCHOR_K[fuel]:.6g}",
            "diss_coeff_A": f"{a_diss:.6g}",
            "interp_scheme": "trilineal en (phi, T3_K, log P3_Pa)",
        }
        doc_extra = [
            "Generada SIN Cantera: modelo analitico de respaldo. Los numeros",
            "son plausibles pero NO son equilibrio quimico. Regenerar.",
        ]

    rows = []
    for i, phi in enumerate(TAD_PHI_GRID):
        for j, t3 in enumerate(TAD_T3_GRID):
            for k, p3 in enumerate(TAD_P3_GRID):
                rows.append((phi, t3, p3, grid_t[i, j, k], grid_cp[i, j, k],
                             grid_mw[i, j, k]))

    meta = {"source": "Quasar_PhyCB_Science.md 9.3 (L0 termoquimico = "
                      "gas.equilibrate('HP') de Cantera); LHV de contrato en "
                      "IMPLEMENTATION_CONTRACT.md 6",
            "status": status}
    if reason:
        meta["status_reason"] = reason
    meta.update(meta_extra)
    meta["grid_phi"] = " ".join(f"{v:g}" for v in TAD_PHI_GRID)
    meta["grid_T3_K"] = " ".join(f"{v:g}" for v in TAD_T3_GRID)
    meta["grid_P3_Pa"] = " ".join(f"{v:g}" for v in TAD_P3_GRID)
    meta["generated_by"] = "data_pipeline.build()"

    _write_csv(
        data_dir / "thermo" / f"tad_fit_{fuel}.csv",
        [f"Tabla lookup de temperatura adiabatica de llama de '{fuel}' sobre la",
         "rejilla (phi, T3_K, P3_Pa). ES EL FALLBACK DE T_ad SIN CANTERA:",
         "physics_core la usa cuando Cantera no esta disponible y etiqueta el",
         "record con source = 'correlations_L0(cantera_unavailable)'.",
         "Recorre la rejilla en orden phi (lento) > T3 (medio) > P3 (rapido).",
         "Interpolar como dice 'interp_scheme'; fuera de rango, clampear.",
         "cp_prod_J_kgK y MW_prod_kgkmol son del gas quemado en equilibrio y",
         "permiten cerrar rho4, gamma_hot y el numero de Mach de salida sin",
         "volver a llamar a Cantera."] + doc_extra,
        meta,
        ["phi", "T3_K", "P3_Pa", "T_ad_K", "cp_prod_J_kgK", "MW_prod_kgkmol"],
        rows)
    return status


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------

def build(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Regenera deterministamente TODAS las tablas del contrato §7.

    Las tablas de correlación son **anclas**: fijan los coeficientes que
    `physics_core` consume, de modo que un cambio accidental dispare
    `validate()` (mismo rol que las anclas de Phy-AC/Phy-CC). Las tablas
    `thermo/tad_fit_*` son datos calculados: se generan con Cantera si está
    disponible y con un modelo analítico de respaldo si no, **sin lanzar nunca**
    por dependencia ausente (CONTRATO §10.4).
    """
    data_dir = Path(data_dir)
    _build_eta_vs_theta(data_dir)
    _build_plf_by_arch(data_dir)
    _build_air_split_by_arch(data_dir)
    _build_zone_lengths(data_dir)
    _build_cd_holes(data_dir)
    _build_cd_am_roughness(data_dir)
    _build_pf_constant(data_dir)
    _build_swirler_K(data_dir)
    _build_thermal_coeffs(data_dir)
    _build_film_effusion(data_dir)
    _build_lbo_constant(data_dir)
    _build_atomization_coeffs(data_dir)
    _build_emissions_coeffs(data_dir)
    _build_class_table(data_dir)

    ct = _cantera()
    if ct is None:
        print("[build] Cantera no disponible -> tablas de T_ad con el modelo "
              "analitico de respaldo (marcadas 'provisional')")
    for fuel in ("jet_a", "propane", "h2"):
        st = _build_tad_table(data_dir, fuel, ct)
        print(f"[build] thermo/tad_fit_{fuel}.csv ({st})")

    write_manifest(data_dir)
    n_prov = sum(1 for rel in IN_REPO_FILES
                 if (load_table(rel, data_dir) or {}).get("status")
                 == "provisional")
    print(f"[build] {len(IN_REPO_FILES)} tablas regeneradas + manifest.json "
          f"({n_prov} provisionales)")


# ----------------------------------------------------------------------------
# Acceso
# ----------------------------------------------------------------------------

def load_table(rel: str, data_dir: Path = DEFAULT_DATA_DIR) -> dict | None:
    """Lee una tabla de `data/` con su procedencia. **Nunca lanza.**

    Devuelve `None` si el archivo no existe o no se puede parsear — los
    consumidores tienen fallback (regla de degradación elegante del contrato).

    El dict devuelto:

    * `cols`       — dict `nombre -> np.ndarray`. Columna numérica ⇒ `float64`;
      columna con texto ⇒ array de strings (`<U`).
    * `col_names`  — orden original de las columnas.
    * `n_rows`     — número de filas de datos.
    * `status`     — `verified` | `provisional` | `unknown` (del `# @status:`).
      Si la tabla tiene columna `status`, **esa columna manda por fila**.
    * `provenance` — todos los `# @clave: valor` de la cabecera (incluye
      `source`, `status_reason`, `model`, rejillas de las tablas de T_ad…).
    * `doc`        — líneas de prosa de la cabecera.
    * `path`       — ruta absoluta como string.
    """
    try:
        p = Path(data_dir) / rel
        if not p.is_file():
            return None
        doc, meta, names, body = _read_csv(p)
        if not names:
            return None
        cols: dict[str, np.ndarray] = {}
        for j, name in enumerate(names):
            raw = [(r[j] if j < len(r) else "") for r in body]
            try:
                cols[name] = np.array([float(v) for v in raw], dtype=float)
            except (TypeError, ValueError):
                cols[name] = np.array(raw, dtype=str)
        return {
            "path": str(p),
            "cols": cols,
            "col_names": list(names),
            "n_rows": len(body),
            "status": meta.get("status", "unknown"),
            "provenance": meta,
            "doc": doc,
        }
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Validación
# ----------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _num(t: dict | None, name: str) -> np.ndarray | None:
    """Columna numérica de una tabla, o None si no existe / no es numérica."""
    if t is None:
        return None
    c = t["cols"].get(name)
    if c is None or c.dtype.kind not in "fiu":
        return None
    return np.asarray(c, dtype=float)


def _pval(t: dict | None, key: str, model: str | None = None,
          key_col: str = "param", val_col: str = "value"):
    """Valor de una tabla en formato largo (`param`/`value`).

    `model` acota la búsqueda en las tablas que además llevan columna `model`
    (atomización, emisiones), donde el mismo `param` se repite por familia.
    """
    if t is None:
        return None
    keys = t["cols"].get(key_col)
    vals = _num(t, val_col)
    if keys is None or vals is None:
        return None
    m = np.asarray(keys, dtype=str) == key
    if model is not None:
        mcol = t["cols"].get("model")
        if mcol is None:
            return None
        m = m & (np.asarray(mcol, dtype=str) == model)
    return float(vals[m][0]) if m.any() else None


def validate(data_dir: Path = DEFAULT_DATA_DIR) -> list[Check]:
    """Verifica manifiesto, cabeceras, monotonías y rangos físicos.

    **Nunca lanza**: un CSV corrupto o editado a mano produce un `Check` en
    FAIL con la excepción como detalle, no un traceback (misma disciplina que
    el vector `g` del contrato §10.4).
    """
    data_dir = Path(data_dir)
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), detail))

    def guard(name: str, fn) -> None:
        """Evalúa `fn() -> (ok, detalle)` capturando cualquier excepción."""
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"excepcion {type(exc).__name__}: {exc}"
        checks.append(Check(name, bool(ok), str(detail)))

    problems = verify_manifest(data_dir)
    add("manifiesto SHA-256 verifica", not problems, "; ".join(problems))
    if problems:
        return checks

    tables = {rel: load_table(rel, data_dir) for rel in IN_REPO_FILES}

    missing = [r for r, t in tables.items() if t is None]
    add("todas las tablas se parsean", not missing, "; ".join(missing))
    if missing:
        return checks

    empty = [r for r, t in tables.items() if t["n_rows"] == 0]
    add("ninguna tabla esta vacia", not empty, "; ".join(empty))

    bad_status = [r for r, t in tables.items()
                  if t["status"] not in ("verified", "provisional")]
    add("toda tabla declara status verified|provisional", not bad_status,
        "; ".join(bad_status))

    no_reason = [r for r, t in tables.items()
                 if t["status"] == "provisional"
                 and not t["provenance"].get("status_reason")]
    add("toda tabla provisional declara su motivo", not no_reason,
        "; ".join(no_reason))

    no_src = [r for r, t in tables.items() if not t["provenance"].get("source")]
    add("toda tabla declara su fuente primaria", not no_src, "; ".join(no_src))

    # Regresión: las claves `# @clave: valor` con mayúsculas se perdían en la
    # prosa antes de introducir el marcador `@` (bug real, 2026-07-30).
    need = ("far_stoich_mass", "grid_phi", "grid_T3_K", "grid_P3_Pa",
            "interp_scheme", "lhv_contract_J_kg", "generator", "mechanism")
    lost = [f"{f}:{k}" for f in ("jet_a", "propane", "h2") for k in need
            if k not in tables[f"thermo/tad_fit_{f}.csv"]["provenance"]]
    add("las tablas de T_ad exponen su metadato de rejilla y procedencia",
        not lost, "; ".join(lost))

    n_prov = sum(1 for t in tables.values() if t["status"] == "provisional")
    add("inventario de estado", True,
        f"{len(tables) - n_prov} verified / {n_prov} provisional")

    # --- eta_vs_theta -------------------------------------------------------
    def _eta_vs_theta():
        t = tables["lefebvre/eta_vs_theta.csv"]
        th, eta = _num(t, "theta_SI"), _num(t, "eta_theta")
        ok = (np.all(np.diff(th) > 0) and np.all(np.diff(eta) > 0)
              and eta.min() > 0.0 and eta.max() < 1.0)
        return ok, f"eta in [{eta.min():.4f}, {eta.max():.4f}]"

    guard("eta_vs_theta: theta y eta crecientes, eta en (0,1)", _eta_vs_theta)

    # --- plf_by_arch --------------------------------------------------------
    def _plf_bands():
        t = tables["lefebvre/plf_by_arch.csv"]
        plf_lo, plf_hi = _num(t, "PLF_min"), _num(t, "PLF_max")
        dp_lo, dp_hi = _num(t, "dp_over_p_min"), _num(t, "dp_over_p_max")
        u_lo, u_hi = _num(t, "U_ref_min_ms"), _num(t, "U_ref_max_ms")
        nom = _num(t, "PLF_nom")
        ok = (np.all(plf_lo <= nom) and np.all(nom <= plf_hi)
              and plf_lo.min() >= 10.0 and plf_hi.max() <= 60.0
              and np.all(dp_lo <= dp_hi) and dp_lo.min() > 0.0
              and dp_hi.max() <= 0.12
              and np.all(u_lo <= u_hi) and u_lo.min() > 0.0
              and u_hi.max() <= 80.0)
        return ok, (f"PLF in [{plf_lo.min():.0f}, {plf_hi.max():.0f}], "
                    f"dp/p in [{dp_lo.min():.3f}, {dp_hi.max():.3f}]")

    guard("plf_by_arch: bandas ordenadas y en rango fisico", _plf_bands)

    def _plf_archs():
        t = tables["lefebvre/plf_by_arch.csv"]
        archs = set(np.asarray(t["cols"]["arch"], dtype=str))
        return ({"can", "annular", "reverse_annular"} <= archs,
                ", ".join(sorted(archs)))

    guard("plf_by_arch: estan las 3 arquitecturas del gate theta[0]",
          _plf_archs)

    # --- air_split_by_arch --------------------------------------------------
    def _air_split():
        t = tables["lefebvre/air_split_by_arch.csv"]
        zones = ("pz", "sz", "dz", "cool")
        s = sum(_num(t, f"frac_{z}_nom") for z in zones)
        ok = bool(np.all(np.abs(s - 1.0) < 1e-12))
        for z in zones:
            lo, nom, hi = (_num(t, f"frac_{z}_min"), _num(t, f"frac_{z}_nom"),
                           _num(t, f"frac_{z}_max"))
            ok &= bool(np.all(lo <= nom) and np.all(nom <= hi)
                       and lo.min() >= 0.0 and hi.max() <= 1.0)
        return ok, f"max |sum-1| = {np.max(np.abs(s - 1.0)):.2e}"

    guard("air_split: los nominales suman 1 y las bandas estan ordenadas",
          _air_split)

    def _air_default():
        t = tables["lefebvre/air_split_by_arch.csv"]
        dflt = _num(t, "phycb_default")
        reg = np.asarray(t["cols"]["regime"], dtype=str)
        ok = int(np.sum(dflt)) == 1
        return ok, f"regimen: {reg[dflt == 1][0]}" if ok else "no hay 1 default"

    guard("air_split: hay exactamente un regimen por defecto", _air_default)

    # --- zone_lengths -------------------------------------------------------
    def _zone_lengths():
        t = tables["lefebvre/zone_lengths.csv"]
        al = _num(t, "AL_over_Aref")
        lpz_lo, lpz_hi = (_num(t, "L_pz_over_DL_min"),
                          _num(t, "L_pz_over_DL_max"))
        ldz_lo, ldz_hi = (_num(t, "L_dz_over_DL_min"),
                          _num(t, "L_dz_over_DL_max"))
        lt_lo, lt_hi = (_num(t, "L_tot_over_DL_min"),
                        _num(t, "L_tot_over_DL_max"))
        # la suma de longitudes NOMINALES debe caer dentro de la banda total
        suma = (_num(t, "L_pz_over_DL_nom") + _num(t, "L_sz_over_DL_nom")
                + _num(t, "L_dz_over_DL_nom"))
        ok = (al.min() >= 0.5 and al.max() <= 0.9
              and np.all(lpz_lo <= lpz_hi) and np.all(ldz_lo <= ldz_hi)
              and np.all(lt_lo <= lt_hi) and lpz_lo.min() > 0.0
              and np.all(suma >= lt_lo) and np.all(suma <= lt_hi))
        return ok, (f"A_L/A_ref in [{al.min():.2f}, {al.max():.2f}], "
                    f"L_tot_nom = {suma.min():.2f}-{suma.max():.2f} D_L")

    guard("zone_lengths: A_L/A_ref en [0.5,0.9] y longitudes coherentes",
          _zone_lengths)

    # --- cd_holes / cd_am_roughness ----------------------------------------
    def _cd_holes():
        t = tables["lefebvre/cd_holes.csv"]
        lo, nom, hi = _num(t, "Cd_min"), _num(t, "Cd_nom"), _num(t, "Cd_max")
        ok = (lo.min() > 0.0 and hi.max() <= 1.0 and np.all(lo <= nom)
              and np.all(nom <= hi))
        return ok, f"C_d in [{lo.min():.2f}, {hi.max():.2f}]"

    guard("cd_holes: C_d en (0,1] y nominal dentro de su banda", _cd_holes)

    def _cd_am():
        t = tables["lefebvre/cd_am_roughness.csv"]
        rd, f_cd = _num(t, "Ra_over_d"), _num(t, "f_Cd")
        ok = (np.all(np.diff(rd) > 0) and np.all(np.diff(f_cd) < 0)
              and abs(f_cd[0] - 1.0) < 1e-9 and f_cd.min() >= 0.75)
        return ok, f"f in [{f_cd.min():.3f}, {f_cd.max():.3f}]"

    guard("cd_am_roughness: f decreciente con Ra/d y en [0.75,1.0]", _cd_am)

    # --- pf_constant / swirler_K -------------------------------------------
    def _pf():
        t = tables["lefebvre/pf_constant.csv"]
        k_pf = _pval(t, "k_dilution")
        pf_max = _pval(t, "PF_max_default")
        pf_ref = math.exp(-k_pf * 1.8 * 15.0)
        ok = (k_pf > 0.0 and 0.30 <= pf_ref <= 0.45
              and 0.15 <= pf_max <= 0.50)
        return ok, f"k = {k_pf:.5f} -> PF(ancla) = {pf_ref:.3f}"

    guard("pf_constant: k>0 y el caso ancla cae en la banda micro 0.30-0.45",
          _pf)

    def _swirler():
        t = tables["lefebvre/swirler_K.csv"]
        ksw, sn = _num(t, "K_sw"), _num(t, "SN_ctrz_min")
        b_lo, b_hi = _num(t, "beta_deg_min"), _num(t, "beta_deg_max")
        ok = (ksw.min() >= 1.0 and ksw.max() <= 1.6 and b_lo.min() >= 30.0
              and b_hi.max() <= 75.0 and sn.min() > 0.5)
        return ok, f"K_sw in [{ksw.min():.2f}, {ksw.max():.2f}]"

    guard("swirler_K: K_sw en [1.0,1.6], beta en [30,75] deg, S_N CTRZ > 0.5",
          _swirler)

    # --- thermal_coeffs / film_effusion ------------------------------------
    def _thermal():
        t = tables["lefebvre/thermal_coeffs.csv"]
        v = _num(t, "value")
        ok = (np.all(np.isfinite(v))
              and _pval(t, "conv_internal_coeff") == 0.020
              and _pval(t, "gas_emissivity_coeff") == 0.29
              and _pval(t, "conv_reynolds_exp") == 0.80
              and _pval(t, "film_slot_C") == 1.10)
        return ok, f"{t['n_rows']} coeficientes"

    guard("thermal_coeffs: finitos y con los coeficientes clave presentes",
          _thermal)

    def _effusion_map():
        t = tables["lefebvre/film_effusion.csv"]
        param = np.asarray(t["cols"]["param"], dtype=str)
        m, ang, val = _num(t, "M"), _num(t, "angle_deg"), _num(t, "value")
        mp = (param == "eta_cool") & np.isfinite(m)
        ok = bool(mp.sum() > 0 and val[mp].min() > 0.0 and val[mp].max() < 1.0)
        for a in np.unique(ang[mp]):              # monotonía en M por ángulo
            sel = mp & (ang == a)
            ok &= bool(np.all(np.diff(val[sel][np.argsort(m[sel])]) > 0))
        return ok, (f"{int(mp.sum())} puntos de mapa, eta in "
                    f"[{val[mp].min():.3f}, {val[mp].max():.3f}]")

    guard("film_effusion: eta_cool(M) creciente por angulo y en (0,1)",
          _effusion_map)

    def _effusion_hierarchy():
        t = tables["lefebvre/film_effusion.csv"]
        sch = np.asarray(t["cols"]["scheme"], dtype=str)
        param = np.asarray(t["cols"]["param"], dtype=str)
        val = _num(t, "value")
        b = (param == "eta_cool_min") | (param == "eta_cool_max")
        ok = (float(val[(sch == "bare_liner") & b].max())
              < float(val[(sch == "effusion") & b].min())
              < float(val[(sch == "transpiration_lattice") & b].min()))
        return ok, "bare < effusion < transpiration"

    guard("film_effusion: la jerarquia de esquemas de refrigeracion crece",
          _effusion_hierarchy)

    # --- lbo / atomizacion / emisiones -------------------------------------
    def _lbo():
        t = tables["lefebvre/lbo_constant.csv"]
        c = _pval(t, "C_LBO_SI")
        # mismo caso ancla que documenta la cabecera (@calibration)
        phi_lbo = (c * 33.3
                   * (0.13 / (250_000.0 ** 1.3 * math.exp(430.0 / 300.0)))
                   * ((50e-6) ** 2 / 8.0e-7) / 43.1e6)
        ok = (c > 0.0 and 0.30 <= phi_lbo <= 0.70
              and _pval(t, "exp_P3") == -1.30
              and _pval(t, "gas_fuel_evap_term") == 0.0)
        return ok, f"phi_LBO(ancla) = {phi_lbo:.3f}"

    guard("lbo_constant: C>0 y el caso ancla da phi_LBO en 0.3-0.7", _lbo)

    def _atomization():
        t = tables["lefebvre/atomization_coeffs.csv"]
        v = _num(t, "value")
        ok = (np.all(np.isfinite(v))
              and _pval(t, "C", "simplex_lefebvre") == 2.25
              and _pval(t, "C1", "prefilm_rizkalla_lefebvre") == 3.33e-3
              and _pval(t, "C1", "prefilm_elshanawany_lefebvre") == 0.073
              and _pval(t, "C", "jet_penetration") == 1.15)
        return ok, f"{t['n_rows']} coeficientes"

    guard("atomization_coeffs: finitos y con las formas verificadas presentes",
          _atomization)

    def _emissions():
        t = tables["lefebvre/emissions_coeffs.csv"]
        v = _num(t, "value")
        model = set(np.asarray(t["cols"]["model"], dtype=str))
        ok = (np.all(np.isfinite(v))
              and {"nox_lefebvre", "nox_rizk_mongia", "nox_rizk_mongia_zonal",
                   "nox_odgers", "nox_laviolette"} <= model
              and _pval(t, "C", "nox_lefebvre") == 9.0e-8
              and _pval(t, "C", "co_lefebvre") == 86.0
              and _pval(t, "C", "nox_odgers") == 29.0)
        return ok, f"{len(model)} modelos, {t['n_rows']} coeficientes"

    guard("emissions_coeffs: finitos y con las 5 correlaciones de NOx",
          _emissions)

    # --- class_table --------------------------------------------------------
    def _class_table():
        t = tables["reference/class_table.csv"]
        lo, hi = _num(t, "min"), _num(t, "max")
        fin = np.isfinite(lo) & np.isfinite(hi)
        cid = _num(t, "class_id")
        params = set(np.asarray(t["cols"]["param"], dtype=str))
        ok = (np.all(lo[fin] <= hi[fin])
              and set(np.unique(cid)) == {0.0, 1.0, 2.0}
              and len(params) >= 25
              and all(int(np.sum(cid == c)) == len(params) for c in (0, 1, 2)))
        return ok, f"{t['n_rows']} filas, {len(params)} parametros x 3 clases"

    guard("class_table: min<=max en toda fila y 3 clases completas",
          _class_table)

    # --- tablas de T_ad -----------------------------------------------------
    #: Bandas de T_ad(phi=1, T3=350 K, 1 atm). Las de WAVE1 T6 son a T3=300 K
    #: (queroseno 2200-2350, H2 2350-2450); a 350 K suben ~35 K.
    bands_tad = {"jet_a": (2200.0, 2400.0), "propane": (2200.0, 2400.0),
                 "h2": (2320.0, 2500.0)}
    for fuel, band in bands_tad.items():
        def _tad(fuel=fuel, band=band):
            t = tables[f"thermo/tad_fit_{fuel}.csv"]
            phi, t3, p3 = _num(t, "phi"), _num(t, "T3_K"), _num(t, "P3_Pa")
            tad, cp, mw = (_num(t, "T_ad_K"), _num(t, "cp_prod_J_kgK"),
                           _num(t, "MW_prod_kgkmol"))
            n_exp = len(TAD_PHI_GRID) * len(TAD_T3_GRID) * len(TAD_P3_GRID)
            ref = tad[(np.abs(phi - 1.0) < 1e-9) & (np.abs(t3 - 350.0) < 1e-9)
                      & (np.abs(p3 - 1.0e5) < 1e-9)]
            mono = True                    # T_ad crece con T3 a (phi, P3) fijos
            for p in np.unique(phi):
                for pr in np.unique(p3):
                    sel = (phi == p) & (p3 == pr)
                    mono &= bool(np.all(
                        np.diff(tad[sel][np.argsort(t3[sel])]) > 0))
            ok = (t["n_rows"] == n_exp and np.all(tad > t3) and mono
                  and np.all(cp > 800.0) and np.all(cp < 4000.0)
                  and np.all(mw > 10.0) and np.all(mw < 40.0)
                  and len(ref) == 1 and band[0] <= ref[0] <= band[1])
            det = (f"{t['n_rows']}/{n_exp} filas, T_ad(phi=1,T3=350,1atm) = "
                   f"{ref[0]:.0f} K" if len(ref) == 1 else "falta el ancla")
            return ok, det

        guard(f"tad_fit_{fuel}: rejilla completa, T_ad>T3, monotona en T3",
              _tad)

        def _tad_err(fuel=fuel):
            t = tables[f"thermo/tad_fit_{fuel}.csv"]
            err = t["provenance"].get("interp_max_rel_err")
            if err is None:
                return True, ("sin Cantera: tabla analitica provisional "
                              "(error no medido)")
            return float(err) < 0.01, f"max = {float(err) * 100:.3f} %"

        guard(f"tad_fit_{fuel}: error de interpolacion < 1 %", _tad_err)

    return checks


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args(argv)

    do_all = not (args.fetch or args.build or args.validate)
    if args.fetch:
        fetch(args.data_dir, force=args.force)
    if args.build or do_all:
        build(args.data_dir)
    if args.validate or do_all:
        checks = validate(args.data_dir)
        n_fail = sum(not c.ok for c in checks)
        for c in checks:
            mark = "OK " if c.ok else "FAIL"
            print(f"  [{mark}] {c.name}" + (f"  ({c.detail})" if c.detail else ""))
        print(f"[validate] {len(checks) - n_fail}/{len(checks)} checks OK")
        return 1 if n_fail else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
