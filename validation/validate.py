#!/usr/bin/env python3
"""
QUASAR Phy-CB · validation/validate.py
======================================
Corre la campaña de VALIDACIÓN (¿son las ecuaciones CORRECTAS?) contra los
casos de `cases.py` y regenera `validation/RESULTS.md`.

    python validation/validate.py                 # tabla + RESULTS.md
    python validation/validate.py --strict        # exit 1 en fallo duro
    python validation/validate.py --only A1,C1    # subconjunto por cid
    python validation/validate.py --no-write      # no toca RESULTS.md
    python validation/validate.py --json out.json # volcado legible a máquina
    python validation/validate.py --l1            # añade el CRN L1 (lento)

Distinción VV&UQ del proyecto: `test_phycb.py` **verifica** (ecuaciones bien
resueltas, T1–T40); esto **valida** (predicciones vs datos publicados), sin
recalibración por-caso (WAVE7 §1 y §5).

Política de fallos — la razón de ser del script
------------------------------------------------
* Un check sobre un valor `derived` que falla es un **FAIL** y sale en el
  reporte con su número. No se esconde, no se suaviza, no se recalibra.
* Los FAIL ya explicados en `docs/VALIDATION.md` viven en `KNOWN_GAPS` con
  su motivo y su ticket de recalibración. `--strict` **no** falla por
  ellos (así la CI detecta REGRESIONES, no el estado conocido) pero **sí
  falla si un `known_gap` empieza a pasar**: un hueco cerrado hay que
  quitarlo del documento, no dejarlo mintiendo.
* Un criterio que no se puede medir en esta máquina se DECLARA
  (`cases.NON_EVALUABLE`), nunca se convierte en un check que aprueba.

Trampa de entorno (contrato §0.b): Phy-AC está instalado como paquete
EDITABLE; sin el `sys.path.insert` de abajo un `import physics_core` desde
otro cwd resuelve SILENCIOSAMENTE al de Phy-AC (NDIM = 15, compresor
axial). `assert_provenance` es el centinela.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, timezone, datetime

# --- TRAMPA DE ENTORNO (contrato §0.b) — antes de cualquier otro import ----
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

for _s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
    # Windows reporta cp1252 y truena con η, φ, ✓ y flechas. Trampa REAL.
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np                                              # noqa: E402

import physics_core as pc                                       # noqa: E402
import crn_core                                                 # noqa: E402
import data_pipeline as dp                                      # noqa: E402
import structures_core as sc                                    # noqa: E402

pc.assert_provenance(pc, crn_core, dp, sc)

from validation.cases import (                                  # noqa: E402
    CASES, CLASS_PARAMS, DECLARED_NOT_RUN, DP_TOL_REL, EI_ORDER_FACTOR,
    EXPECTED_ANCHORS, LBO_TOL_FACTOR, NON_EVALUABLE, PHI_PZ_TOL_REL,
    PHI_RATIO_TOL_REL, T4_TOL_REL, THETA_INDEX, UQ_MEASURED_SWEEP,
    UQ_SIGMA_REL, theta_from_spec)

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHORS_JSON = os.path.join(HERE, "anchors.json")
RESULTS_MD = os.path.join(HERE, "RESULTS.md")

#: Banda de orden de magnitud del EI_NOx para queroseno de clase micro
#: [g/kg]. Derivación DE LA CASA a partir de
#: `emissions_coeffs.csv::reference_micro` (NOx 35–80 ppm@15 %O₂), tal como
#: la documenta `physics_core.py` §3 (comentario de
#: `EI_NOX_PLAUSIBLE_RANGE`: «80 ppm ≈ 5 g/kg»). NO es una medición directa;
#: solo se usa para el criterio de orden de magnitud ×2 que permite WAVE7.
EI_NOX_MICRO_KERO_BAND = (0.1, 5.0)

#: Bisección de T₄ (bucle ACOTADO, contrato §10.5): 60 pasos sobre el rango
#: de θ[19]. Sin `while` abierto.
T4_BISECT_STEPS = 60
T4_BISECT_LO, T4_BISECT_HI = 800.0, 1800.0

# ---------------------------------------------------------------------------
# HUECOS CONOCIDOS — el libro mayor de la honestidad
# ---------------------------------------------------------------------------
#: (cid, etiqueta) → motivo + ticket de recalibración. Cada entrada tiene su
#: párrafo en `docs/VALIDATION.md`. Añadir aquí SIN escribirlo allí es
#: exactamente lo que este archivo existe para impedir.
#: Cada motivo empieza con su identificador de hueco `G#`, que es la
#: sección de `docs/VALIDATION.md` §4 donde está explicado y priorizado.
_G1 = ("G1 · La tabla de clase se contradice a sí misma: el FAR que impone "
       "el balance de entalpía a partir de sus propias bandas de (T₃, T₄) "
       "no cabe en su banda de FAR/φ_global (ver A4). φ_global = FAR/FAR_st "
       "e I = FAR·LHV·P_atm/(R·T₃·τ) heredan el defecto por identidad. "
       "Recalibración: NINGUNA — hay que corregir `class_table.csv`, no el "
       "modelo. Tocar una constante para «cuadrar» esto sería calibrar "
       "contra un dato inconsistente.")
_G2 = ("G2 · Dos tablas `verified` del paquete se contradicen en 1 punto: "
       "`air_split_by_arch.csv::microturbine.frac_cool_nom` = 0.16 frente a "
       "`class_table.csv::clase 0 frac_cool` = 0–0.15. THETA_REF usa 0.16. "
       "Es un defecto de DATO, de impacto nulo en la física.")
_G3 = ("G3 · Nº de orificios de efusión por encima de la banda de clase. No "
       "es un error de cálculo: n = A_efusión/(π/4·d²) con la porosidad que "
       "sale de pitch/d, y `PENDIENTES §4.2` demuestra que hacer FACTIBLE "
       "el ancla exige apretar el paso a pitch/d = 4–5, lo que SUBE aún más "
       "el número. La banda 300–1500 de la clase 0 y las reglas de efusión "
       "densa del propio proyecto son incompatibles. Corregir el DATO. "
       "AMPLIADO EL 2026-07-31: la corrección del área mojada de la efusión "
       "(physics_core §12.b — π·D_L es el perímetro de un CÍRCULO de área "
       "equivalente y no el de la corona real, ×1.375 en el ancla) sube n en "
       "las arquitecturas ANULARES en ese mismo factor y mete también a A2 "
       "(clase 1, anular inversa: 3696 → 5080 frente a [1000, 4000]) en este "
       "hueco. Es la MISMA incompatibilidad de dato, ahora con el área "
       "correcta: el número de antes estaba dentro de la banda por un error "
       "de geometría, no por acuerdo con la clase. NO se ha movido ningún "
       "umbral para taparlo.")
_G4 = ("G4 · V_c fuera de banda por continuidad: V_c = τ·ṁ/ρ₃. Con τ dentro "
       "de banda, V_c queda determinado por ṁ y ρ₃; en el extremo alto de "
       "la banda de ṁ de la clase 0 el V_c que exige la continuidad "
       "sobrepasa el techo de 0.4 L. Ver la pata 2 de A4.")
_G5 = ("G5 · PF = 0.35000 en TODOS los casos, con cualquier clase y "
       "cualquier punto de operación. No es una predicción: es la "
       "calibración devuelta. PF = exp[−k·(L_DZ/D_L)·(ΔP_L/q_ref)] con k "
       "calibrada a UN punto, y con el criterio aerodinámico dimensionando "
       "A_ref el cociente ΔP_L/q_ref = 15 es una identidad algebraica "
       "(PENDIENTES §5). Es el pendiente de recalibración nº 1 de esta "
       "campaña.")
_G6 = ("G6 · Ø de carcasa por debajo de la banda de clase en las clases 1–2 "
       "(y en D1). A_ref sale del criterio aerodinámico con "
       "`PLF = 20` (fila `micro_target` de `plf_by_arch.csv`) para TODAS "
       "las arquitecturas, mientras la fila de la arquitectura da PLF = 36 "
       "(can) o 22.5 (reverse_annular): usar el PLF de la arquitectura "
       "subiría A_ref ×√(36/20) = 1.34 en `can`. Es el pendiente de "
       "recalibración nº 4.")
_G7 = ("G7 · η_comb por debajo de la banda en las clases 1–2. El driver es "
       "η_mix = ETA_CEILING·(1−e^{−K_MIX_DP·ΔP_L/q_ref})·penalización de "
       "penetración; `K_MIX_DP` es constante de la casa. La banda de clase "
       "pide >99 % / >99.5 % y el modelo da 98.8 % / 98.0 %.")
_G8 = ("G8 · SMD de airblast ×3–4 por debajo de la banda de clase (10.7 µm "
       "frente a 35–60 µm en la clase 1). Las dos correlaciones "
       "dimensionales de airblast SÍ están verificadas (arXiv:1906.04042 "
       "Tabla 7), así que el conflicto es entre dos fuentes `verified`. "
       "CONSECUENCIA GRAVE: φ_LBO ∝ D₃₂², de modo que un SMD 3.5× pequeño "
       "hace el margen de LBO ~12× optimista (LBO_margin = 42 en A2, o sea "
       "«esta cámara no se apaga nunca», que no es un resultado creíble). "
       "Es el pendiente de recalibración nº 2.")
_G9 = ("G9 · φ_PZ fuera de banda. En A2/A3 hereda G1 (el FAR viene "
       "inflado). En D1 es el hallazgo de recuperación extrema: con "
       "T₃ = 1093 K el φ_global es 0.118 y ni con frac_pz en su MÍNIMO de "
       "contrato (0.15) se alcanza φ_PZ ≥ 0.8.")
_G10 = ("G10 · τ fuera de banda. τ = V_c·ρ₃/ṁ es continuidad; hereda el "
        "dimensionado (G6) y, en la clase 0 con ṁ alto, el mismo defecto "
        "de banda que G4.")
_G11 = ("G11 · T de pared fuera de la banda de clase. **Solo queda D1**, y "
        "CAMBIÓ DE SIGNO el 2026-08-01 al corregir la corona geométrica: "
        "era 1466 K (+27 % sobre 1000–1150 K) y ahora es 952 K (−4.8 % por "
        "DEBAJO). Está mucho más cerca, pero el modelo pasó de sobre- a "
        "sub-predecir. B1 y B2 salieron del hueco (994.6 K y 1012 K, dentro "
        "de banda). Sospechoso nº 1 de PENDIENTES §6.1: el suelo "
        "`EPS_G_FLOOR_LUMINOUS = 0.15`, calibración de la casa que hoy "
        "SUSTITUYE a la física declarada (ε_g calculado = 0.0065) y que al "
        "barrer su banda de literatura mueve T_w1 entre 808 K y 1573 K. "
        "Phy-CB **no predice T de metal cuantitativa en ningún nivel** "
        "(contrato §12): esto es una comprobación de banda.")
_G16 = ("G16 · A2 (clase 1) se sale de la banda de T de pared POR DEBAJO "
        "(952 K frente a 1000–1150 K) desde el 2026-08-01, al corregir la "
        "corona liner↔carcasa: `A_an` pasó del atajo `A_ref − A_L` a la "
        "geométrica π(r_cas²−r_lo²), que es ×0.80 y por tanto refrigera más "
        "(más flujo másico por unidad de área en la convección externa C₂, "
        "Ciencia §5). La corrección es correcta —es la corona que se "
        "fabrica— y en el balance GANA: metió a B1 (994.6 K) y B2 (1012 K) "
        "dentro de banda y subió el marcador de 107 a 108 PASS. Pero deja a "
        "A2 sub-prediciendo un 4.8 %. Como G11, apunta al mismo sospechoso "
        "nº 1 de PENDIENTES §6.1 (`EPS_G_FLOOR_LUMINOUS`), que es quien "
        "domina T_w1 y está calibrado a un solo punto: con el término "
        "radiante mal anclado, cualquier cambio en el convectivo mueve la "
        "pared entera. Phy-CB **no predice T de metal cuantitativa en "
        "ningún nivel** (contrato §12): esto es una comprobación de banda.")

_G12 = ("G12 · El modelo declara IN625 insuficiente (T_w1 = 1466 K frente "
        "al límite 1253 K − 50 K de margen) en el caso Bath/HiETA, que "
        "está FABRICADO y OPERADO en IN625. Es la falsación más limpia de "
        "la campaña y apunta al mismo sitio que G11.")
_G13 = ("G13 · La cabecera `record['EI_NOx']` es EI_NOx_lefebvre "
        "(physics_core línea 3042) y tiene el signo EQUIVOCADO frente a "
        "φ_PZ: la forma de Lefebvre no depende de φ_PZ (va con T_st, P₃, "
        "V_c y ṁ) y el −25 % que se ve al barrer φ_PZ es puro artefacto "
        "del denominador (g NOx por kg de combustible, y el combustible "
        "cambia con η_comb). Las tres formas ZONALES —zonal, Odgers y "
        "Laviolette— SÍ dan el signo correcto y el pico en φ_PZ ≈ 1.05. "
        "ACCIÓN: la cabecera de NOx debe ser la zonal, no la de Lefebvre. "
        "Es el pendiente de recalibración nº 3 y **no requiere ningún dato "
        "nuevo** — es una decisión de modelo.")
_G14 = ("G14 · Rizk & Mongia global es INSENSIBLE a φ_PZ (variación "
        "relativa 0.00 % en todo el barrido). No es un fallo de "
        "implementación: la forma global no lleva φ. Se reporta para que "
        "nadie la use como indicador de tendencia.")
_G15 = ("G15 · La forma zonal es INSENSIBLE a τ (variación < 2 %): es "
        "función de φ y T de zona, no del tiempo de residencia. "
        "Complementaria de G13: para tendencia en φ, la zonal; para "
        "tendencia en τ, Odgers o Lefebvre.")

KNOWN_GAPS = {
    # --- G1 ---------------------------------------------------------------
    ("A1", "FAR_global"): _G1, ("A1", "phi_global"): _G1,
    ("A1", "I_MW_m3_atm"): _G1,
    ("A2", "FAR_global"): _G1, ("A2", "phi_global"): _G1,
    ("A2", "I_MW_m3_atm"): _G1,
    ("A3", "FAR_global"): _G1, ("A3", "phi_global"): _G1,
    ("A3", "I_MW_m3_atm"): _G1,
    ("B1", "FAR_global"): _G1, ("B1", "phi_global"): _G1,
    ("B1", "I_MW_m3_atm"): _G1,
    ("B2", "FAR_global"): _G1, ("B2", "phi_global"): _G1,
    ("B2", "I_MW_m3_atm"): _G1,
    ("A4", "clase 0 · T₃ no recuperada"): _G1,
    ("A4", "clase 2 · T₃ no recuperada"): _G1,
    # --- G2 ---------------------------------------------------------------
    ("A1", "frac_cool"): _G2, ("B1", "frac_cool"): _G2,
    ("B2", "frac_cool"): _G2,
    # --- G3 ---------------------------------------------------------------
    ("A1", "n_eff_holes"): _G3, ("B1", "n_eff_holes"): _G3,
    ("B2", "n_eff_holes"): _G3,
    #  A2 entra en el hueco el 2026-07-31 con la corrección del área mojada
    #  de la efusión (×1.375 en las anulares). NO es un umbral movido: es el
    #  mismo hueco con el área correcta. Ver `_G3` y docs/VALIDATION.md §G3.
    ("A2", "n_eff_holes"): _G3,
    # --- G4 ---------------------------------------------------------------
    ("A1", "V_c_L"): _G4, ("B2", "V_c_L"): _G4,
    # --- G5 ---------------------------------------------------------------
    ("A3", "PF"): _G5,
    # --- G6 ---------------------------------------------------------------
    ("A2", "D_casing_mm"): _G6, ("A3", "D_casing_mm"): _G6,
    ("D1", "D_casing_mm"): _G6,
    # --- G7 ---------------------------------------------------------------
    ("A2", "eta_comb_design"): _G7, ("A3", "eta_comb_design"): _G7,
    ("D1", "eta_comb_design"): _G7,
    # --- G8 ---------------------------------------------------------------
    ("A2", "SMD_um"): _G8, ("A3", "SMD_um"): _G8,
    # --- G9 ---------------------------------------------------------------
    ("A2", "phi_pz"): _G9, ("A3", "phi_pz"): _G9, ("D1", "phi_pz"): _G9,
    # --- G10 --------------------------------------------------------------
    ("A3", "tau_ms"): _G10, ("B2", "tau_ms"): _G10, ("D1", "tau_ms"): _G10,
    # --- G11 / G12 --------------------------------------------------------
    #  B1/B2 SALIERON de este hueco el 2026-08-01 al corregir la corona
    #  liner↔carcasa (`A_an` geométrica en vez del atajo `A_ref − A_L`):
    #  T_w1 pasó de 1059 K a 994.6 K en B1 y a 1012 K en B2, ambas dentro de
    #  la banda 900–1050 K. Se quitan de KNOWN_GAPS porque el propio
    #  `validate.py` los marcaba ya como «hueco obsoleto». D1 sigue dentro.
    ("D1", "T_wall_liner_K"): _G11,
    ("A2", "T_wall_liner_K"): _G16,
    ("D1", "Inconel-625 soporta la pared predicha"): _G12,
    # --- G13 / G14 / G15 --------------------------------------------------
    ("D2", "signo ∂/∂φ_PZ · EI_NOx (cabecera = lefebvre)"): _G13,
    ("D2", "signo ∂/∂φ_PZ · EI_NOx_lefebvre"): _G13,
    ("D2", "signo ∂/∂φ_PZ · EI_NOx_rizk_mongia"): _G14,
    ("D2", "signo ∂/∂τ · EI_NOx_zonal"): _G15,
}

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
_STATUS_TXT = {True: "PASS", False: "FAIL", None: "N/E"}


def _fmt(v) -> str:
    """Formato compacto y estable para la tabla (no localiza decimales)."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(x):
        return "no-finito"
    a = abs(x)
    if a and (a < 1e-3 or a >= 1e5):
        return f"{x:.3e}"
    return f"{x:.4g}"


def check(cid: str, label: str, ok, *, role: str = "derived",
          value=None, target: str = "", detail: str = "") -> dict:
    """Construye un check. `ok=None` ⇒ NO EVALUABLE (nunca aprueba).

    `role`:
      * `derived`  — el modelo lo calcula ⇒ cuenta para el veredicto.
      * `input`    — se lo damos nosotros ⇒ informativo, jamás bloquea.
      * `data`     — se valida un DATO del paquete, no el modelo.
      * `declared` — criterio declarado no evaluable.
    """
    gap = KNOWN_GAPS.get((cid, label))
    blocking = role in ("derived", "data") and ok is not None
    return dict(cid=cid, label=label, ok=ok, role=role, value=value,
                target=target, detail=detail, blocking=blocking,
                known_gap=gap, stale_gap=bool(gap) and ok is True)


def _in_band(x: float, lo: float, hi: float) -> bool:
    return bool(np.isfinite(x) and lo <= x <= hi)


def _theta(case: dict) -> np.ndarray:
    return np.asarray(theta_from_spec(case.get("spec", {})), dtype=float)


def _assert_theta_layout() -> None:
    """El orden de θ de `cases.py` DEBE coincidir con el contrato §1.

    Es la única excepción deliberada a «nunca lanzar»: si el ABI se movió,
    todo número de esta campaña sería basura plausible.
    """
    for name, idx in THETA_INDEX.items():
        if pc.VAR_NAMES[idx] != name:
            raise RuntimeError(
                f"θ desalineado: cases.THETA_INDEX['{name}'] = {idx} pero "
                f"physics_core.VAR_NAMES[{idx}] = '{pc.VAR_NAMES[idx]}'. "
                "El contrato §1 se ha movido — actualiza cases.py antes de "
                "creerte ningún resultado.")


def _eval(theta: np.ndarray, material: str = "Inconel-625",
          fidelity=None) -> dict:
    """`evaluate` de campaña: sin caché y **sin calibración afín**.

    `calibrate=False` es obligatorio: la campaña mide el modelo desnudo. Si
    la calibración afín estuviera activa estaríamos validando la
    calibración, no la física (regla de la familia, `Phy-AC/validate.py`).
    """
    return pc.evaluate(theta, fidelity=fidelity or pc.Fidelity.L0,
                       use_cache=False, calibrate=False, material=material)


# ---------------------------------------------------------------------------
# Tabla de clase
# ---------------------------------------------------------------------------
_CLASS_CACHE: dict | None = None


def class_bands() -> dict:
    """`{class_id: {param: (min, max, units, status)}}` de `class_table.csv`."""
    global _CLASS_CACHE
    if _CLASS_CACHE is not None:
        return _CLASS_CACHE
    t = dp.load_table("reference/class_table.csv")
    out: dict = {}
    if t is not None:
        c = t["cols"]
        for i in range(t["n_rows"]):
            cid = int(c["class_id"][i])
            out.setdefault(cid, {})[str(c["param"][i])] = (
                float(c["min"][i]), float(c["max"][i]),
                str(c["units"][i]), str(c["status"][i]))
    _CLASS_CACHE = out
    return out


def _class_value(rec: dict, key: str, theta: np.ndarray):
    """Valor del record para una fila de `class_table` (o derivación)."""
    if key == "_pi":
        return float(theta[THETA_INDEX["P3_Pa"]]) / pc.P_ATM
    if key == "_V_c_L":
        return 1e3 * float(rec.get("V_c_m3", float("nan")))
    v = rec.get(key)
    return None if v is None else float(v)


# ---------------------------------------------------------------------------
# Runners por tipo de caso
# ---------------------------------------------------------------------------
def run_class_bands(case: dict) -> dict:
    """Compara el record contra las bandas de su clase de potencia."""
    theta = _theta(case)
    rec = _eval(theta, case.get("material", "Inconel-625"))
    bands = class_bands().get(int(case["class_id"]), {})
    checks = []
    for p in CLASS_PARAMS:
        band = bands.get(p["key"])
        if band is None:
            continue
        lo, hi, units, status = band
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue                       # `nan` = no aplica a esta clase
        val = _class_value(rec, p["rec"], theta)
        if val is None:
            continue
        checks.append(check(
            case["cid"], p["key"], _in_band(val, lo, hi), role=p["role"],
            value=val, target=f"[{_fmt(lo)}, {_fmt(hi)}] {units}",
            detail=f"status de la fila: {status}"))

    # Arquitectura / inyector: consistencia con el dato verificado.
    if case["cid"] == "B1":
        at = rec.get("atomizer") or {}
        checks.append(check(
            case["cid"], "vaporizador KJ66 (nº tubos)",
            6.0 <= float(at.get("n_tubes", 0.0)) <= 12.0, role="data",
            value=at.get("n_tubes"), target="[6, 12] tubos ✅ [W]",
            detail="CONSISTENCIA con el dato, no predicción"))
        checks.append(check(
            case["cid"], "vaporizador KJ66 (L tubo)",
            abs(float(at.get("tube_length_mm", 0.0)) - 64.0) <= 5.0,
            role="data", value=at.get("tube_length_mm"),
            target="64 ± 5 mm ✅ [W]", detail="ídem"))
        checks.append(check(
            case["cid"], "vaporizador KJ66 (codo)",
            abs(float(at.get("elbow_deg", 0.0)) - 30.0) <= 5.0, role="data",
            value=at.get("elbow_deg"), target="30 ± 5 deg ✅ [W]",
            detail="ídem"))
        checks.append(check(
            case["cid"], "arquitectura", rec.get("arch") == "reverse_annular",
            role="input", value=rec.get("arch"), target="reverse_annular"))

    # Cierre del balance de energía (VERIFICACIÓN, no validación).
    checks.append(_energy_closure(case["cid"], rec, theta))
    return dict(case=case, rec=rec, theta=theta, checks=checks)


def _energy_closure(cid: str, rec: dict, theta: np.ndarray) -> dict:
    """Round-trip T₄ → FAR → T₄: cierre del balance de entalpía.

    NO es validación (T₄ es entrada, `cases.NON_EVALUABLE['T4']`): mide que
    la inversión analítica del balance devuelve el T₄ que se pidió. Es el
    único sentido en que el criterio «T4 ±3 %» de WAVE7 §3.2 se puede
    evaluar hoy, y se etiqueta como tal.
    """
    far = float(rec["f_fuel_air"])
    eta = float(rec["eta_comb"])
    lhv = float(rec["LHV_J_kg"])
    T3 = float(theta[THETA_INDEX["T3_K"]])
    T4_req = float(theta[THETA_INDEX["T4_K"]])
    # Inversa exacta de _far_from_T4 con la misma η resuelta:
    T4_back = (far * eta * lhv + pc.CP * T3) / (pc.CP_HOT * (1.0 + far))
    err = (T4_back - T4_req) / max(T4_req, 1e-9)
    return check(cid, "cierre de energía T₄→FAR→T₄", abs(err) < T4_TOL_REL,
                 role="derived", value=T4_back,
                 target=f"{_fmt(T4_req)} K ±{100 * T4_TOL_REL:.0f} %",
                 detail=f"Δ = {100 * err:+.3f} % · VERIFICACIÓN, no "
                        f"validación (T₄ es θ[19])")


def run_table_consistency(case: dict) -> dict:
    """A4 · ¿es la tabla de clase compatible consigo misma?

    Para cada clase y cada rama de T₃ (no recuperada / recuperada) se
    calcula la banda de FAR que IMPONE el balance de entalpía a partir de
    las bandas de (T₃, T₄), y se comprueba que solape con la banda de FAR
    que la tabla declara. Es álgebra pura — ningún coeficiente provisional.
    """
    bands = class_bands()
    checks, rows = [], []
    for cid_class in sorted(bands):
        b = bands[cid_class]
        far_band = b.get("FAR_global")
        T4b = b.get("T4_K")
        if far_band is None or T4b is None:
            continue
        for branch, key in (("no recuperada", "T3_K_unrecuperated"),
                            ("recuperada", "T3_K_recuperated")):
            T3b = b.get(key)
            if T3b is None or not np.isfinite(T3b[0]):
                continue
            # f crece con T₄ y decrece con T₃ ⇒ extremos en las esquinas.
            f_lo = pc._far_from_T4("jet_a", T3b[1], T4b[0])
            f_hi = pc._far_from_T4("jet_a", T3b[0], T4b[1])
            overlap = not (f_hi < far_band[0] or f_lo > far_band[1])
            label = f"clase {cid_class} · T₃ {branch}"
            rows.append(dict(label=label, f_lo=f_lo, f_hi=f_hi,
                             decl_lo=far_band[0], decl_hi=far_band[1],
                             ok=overlap))
            checks.append(check(
                case["cid"], label, overlap, role="data",
                value=f"[{_fmt(f_lo)}, {_fmt(f_hi)}]",
                target=f"solapa con [{_fmt(far_band[0])}, "
                       f"{_fmt(far_band[1])}]",
                detail=(f"T₃ ∈ [{_fmt(T3b[0])}, {_fmt(T3b[1])}] K · "
                        f"T₄ ∈ [{_fmt(T4b[0])}, {_fmt(T4b[1])}] K")))

        # --- pata 2: V_c ↔ (τ, ṁ, ρ₃) ------------------------------------
        # V_c = τ·ṁ/ρ₃ con ρ₃ = π·P_atm/(R·T₃). Es continuidad, no una
        # correlación: si la banda de V_c no contiene lo que imponen las
        # bandas de τ, ṁ, π y T₃, la tabla se contradice a sí misma.
        Vb, tb, mb, pib = (b.get("V_c_L"), b.get("tau_ms"),
                           b.get("mdot_air_kgs"), b.get("pressure_ratio"))
        T3b = b.get("T3_K_unrecuperated")
        if all(x is not None for x in (Vb, tb, mb, pib, T3b)):
            rho_lo = pib[0] * pc.P_ATM / (pc.RGAS * T3b[1])   # ρ₃ mínima
            rho_hi = pib[1] * pc.P_ATM / (pc.RGAS * T3b[0])   # ρ₃ máxima
            v_lo = 1e3 * (1e-3 * tb[0]) * mb[0] / rho_hi
            v_hi = 1e3 * (1e-3 * tb[1]) * mb[1] / rho_lo
            ok = not (v_hi < Vb[0] or v_lo > Vb[1])
            rho_c = (0.5 * (pib[0] + pib[1]) * pc.P_ATM
                     / (pc.RGAS * 0.5 * (T3b[0] + T3b[1])))
            v_c = 1e3 * (1e-3 * 0.5 * (tb[0] + tb[1])) \
                * 0.5 * (mb[0] + mb[1]) / rho_c
            label = f"clase {cid_class} · V_c ↔ (τ, ṁ, ρ₃)"
            checks.append(check(
                case["cid"], label, ok, role="data",
                value=f"[{_fmt(v_lo)}, {_fmt(v_hi)}] L",
                target=f"solapa con [{_fmt(Vb[0])}, {_fmt(Vb[1])}] L",
                detail=f"continuidad V_c = τ·ṁ/ρ₃ (álgebra) · en el CENTRO "
                       f"de las bandas V_c = {_fmt(v_c)} L"))

        # --- pata 3: I ↔ (FAR, T₃, τ) --------------------------------------
        # Identidad ya anotada en `physics_core._sizing` (auditoría
        # 2026-07-30): I = FAR·LHV·P_atm/(R·T₃·τ). V_c y P₃ se cancelan.
        Ib = b.get("I_MW_m3_atm")
        if all(x is not None for x in (Ib, far_band, tb, T3b)):
            lhv = float(pc.FUELS["jet_a"]["LHV"])
            i_lo = (far_band[0] * lhv * pc.P_ATM
                    / (pc.RGAS * T3b[1] * 1e-3 * tb[1])) / 1e6
            i_hi = (far_band[1] * lhv * pc.P_ATM
                    / (pc.RGAS * T3b[0] * 1e-3 * tb[0])) / 1e6
            ok = not (i_hi < Ib[0] or i_lo > Ib[1])
            i_c = (0.5 * (far_band[0] + far_band[1]) * lhv * pc.P_ATM
                   / (pc.RGAS * 0.5 * (T3b[0] + T3b[1])
                      * 1e-3 * 0.5 * (tb[0] + tb[1]))) / 1e6
            label = f"clase {cid_class} · I ↔ (FAR, T₃, τ)"
            checks.append(check(
                case["cid"], label, ok, role="data",
                value=f"[{_fmt(i_lo)}, {_fmt(i_hi)}] MW/m³·atm",
                target=f"solapa con [{_fmt(Ib[0])}, {_fmt(Ib[1])}]",
                detail=f"identidad I = FAR·LHV·P_atm/(R·T₃·τ) "
                       f"(physics_core §7, auditoría 2026-07-30) · en el "
                       f"CENTRO de las bandas I = {_fmt(i_c)}"))
    return dict(case=case, rec=None, theta=None, checks=checks, rows=rows)


def _solve_T4_for_phi_global(theta: np.ndarray, phi_target: float) -> float:
    """T₄ que hace que `record['phi_global']` valga `phi_target`.

    Bisección ACOTADA de `T4_BISECT_STEPS` pasos (contrato §10.5, sin
    `while` abierto). φ_global crece monótonamente con T₄ a T₃ fija.
    Misma técnica que `Phy-AC/validation/validate.build_theta`: se le da al
    modelo el dato medido y se califica lo que deduce de él.
    """
    lo, hi = T4_BISECT_LO, T4_BISECT_HI
    t = theta.copy()
    for _ in range(T4_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        t[THETA_INDEX["T4_K"]] = mid
        phi = float(_eval(t)["phi_global"])
        if phi < phi_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_split_anchor(case: dict) -> dict:
    """C1 · reparto de aire publicado → φ_PZ. El caso más falsable."""
    pts = case["points"]
    phi_g = float(case["phi_global_implied"])
    base = _theta(case)
    T4 = _solve_T4_for_phi_global(base, phi_g)
    base[THETA_INDEX["T4_K"]] = T4

    results = []
    for p in pts:
        t = base.copy()
        frac_pz = float(p["frac_pz"])
        # frac_sz/frac_dz se reescalan para dejar frac_cool = 0.16 nominal.
        rest = max(1.0 - frac_pz - 0.16, 1e-6)
        w_sz = 0.12 / (0.12 + 0.42)
        t[THETA_INDEX["frac_pz"]] = frac_pz
        t[THETA_INDEX["frac_sz"]] = rest * w_sz
        t[THETA_INDEX["frac_dz"]] = rest * (1.0 - w_sz)
        rec = _eval(t)
        results.append(dict(label=p["label"], frac_pz=frac_pz,
                            phi_pub=float(p["phi_pz_pub"]),
                            phi_mod=float(rec["phi_pz"]), rec=rec, theta=t))

    checks = []
    # (a) INVARIANTE de reparto: no depende del punto de operación.
    r_pub = results[1]["phi_pub"] / results[0]["phi_pub"]
    r_mod = results[1]["phi_mod"] / results[0]["phi_mod"]
    err_r = (r_mod - r_pub) / r_pub
    checks.append(check(
        case["cid"], "invariante φ_PZ,opt/φ_PZ,base",
        abs(err_r) < PHI_RATIO_TOL_REL, role="derived", value=r_mod,
        target=f"{_fmt(r_pub)} ±{100 * PHI_RATIO_TOL_REL:.0f} %",
        detail=f"Δ = {100 * err_r:+.2f} % · independiente del punto de "
               f"operación: mide SOLO la cadena reparto→φ_PZ"))
    # (b) valor ABSOLUTO en cada punto publicado.
    for r in results:
        err = (r["phi_mod"] - r["phi_pub"]) / r["phi_pub"]
        checks.append(check(
            case["cid"], f"φ_PZ · {r['label']} (frac_pz = {r['frac_pz']:.4f})",
            abs(err) < PHI_PZ_TOL_REL, role="derived", value=r["phi_mod"],
            target=f"{_fmt(r['phi_pub'])} ±{100 * PHI_PZ_TOL_REL:.0f} %",
            detail=f"Δ = {100 * err:+.2f} %"))
    # (c) el modelo debe reproducir la estabilización: φ_PZ SUBE al bajar
    #     el aire de PZ (es la razón física por la que el paper lo hizo).
    checks.append(check(
        case["cid"], "signo · menos aire de PZ ⇒ φ_PZ mayor",
        results[1]["phi_mod"] > results[0]["phi_mod"], role="derived",
        value=f"{_fmt(results[0]['phi_mod'])} → {_fmt(results[1]['phi_mod'])}",
        target="creciente", detail="tendencia publicada: 0.89 → 1.09"))
    checks.append(check(
        case["cid"], "T₄ resuelto por bisección", True, role="input",
        value=T4, target="—",
        detail=f"φ_global objetivo = {_fmt(phi_g)} (derivada de la fuente)"))
    return dict(case=case, rec=results[0]["rec"], theta=results[0]["theta"],
                checks=checks, points=results)


def run_thermal_extreme(case: dict) -> dict:
    """D1 · térmico con T₃ = 1093 K (recuperación extrema) e IN625."""
    theta = _theta(case)
    material = case.get("material", "Inconel-625")
    rec = _eval(theta, material)
    bands = class_bands().get(int(case["class_id"]), {})
    checks = []

    tw = float(rec["T_wall_max_K"])
    b = bands.get("T_wall_liner_K")
    if b:
        checks.append(check(
            case["cid"], "T_wall_liner_K", _in_band(tw, b[0], b[1]),
            role="derived", value=tw,
            target=f"[{_fmt(b[0])}, {_fmt(b[1])}] K",
            detail="BANDA, no valor: Phy-CB no predice T de metal "
                   "cuantitativa en ningún nivel (contrato §12)"))
    # ¿Sobrevive el material que Bath USÓ de verdad?
    lim = float(rec["T_max_flame_wall_K"])
    checks.append(check(
        case["cid"], f"{material} soporta la pared predicha",
        tw <= lim - pc.T_WALL_MARGIN_K, role="derived", value=tw,
        target=f"≤ {_fmt(lim - pc.T_WALL_MARGIN_K)} K "
               f"({material} − margen {pc.T_WALL_MARGIN_K:.0f} K)",
        detail="Bath FABRICÓ y operó este combustor en IN625: si aquí sale "
               "infactible, el conservadurismo es del modelo"))
    # Coherencia física mínima: la pared no puede bajar del aire que la enfría.
    T3 = float(theta[THETA_INDEX["T3_K"]])
    checks.append(check(
        case["cid"], "T_w1 > T₃ (el refrigerante ya llega a 1093 K)",
        tw > T3, role="derived", value=tw, target=f"> {_fmt(T3)} K",
        detail="test de cordura del térmico 1D en recuperación extrema"))
    # φ_PZ alcanzable con los bounds del contrato.
    b = bands.get("phi_pz")
    if b:
        checks.append(check(
            case["cid"], "phi_pz", _in_band(float(rec["phi_pz"]), b[0], b[1]),
            role="derived", value=float(rec["phi_pz"]),
            target=f"[{_fmt(b[0])}, {_fmt(b[1])}]",
            detail=f"frac_pz = {theta[THETA_INDEX['frac_pz']]:.3f} = MÍNIMO "
                   f"del contrato §1 (bound inferior 0.15)"))
    for key, rk in (("tau_ms", "tau_ms"), ("V_c_L", "_V_c_L"),
                    ("D_casing_mm", "D_casing_mm"),
                    ("eta_comb_design", "eta_comb"), ("PF", "PF")):
        b = bands.get(key)
        if not b or not np.isfinite(b[0]):
            continue
        v = _class_value(rec, rk, theta)
        checks.append(check(case["cid"], key, _in_band(v, b[0], b[1]),
                            role="derived", value=v,
                            target=f"[{_fmt(b[0])}, {_fmt(b[1])}] {b[2]}"))
    checks.append(_energy_closure(case["cid"], rec, theta))
    return dict(case=case, rec=rec, theta=theta, checks=checks)


#: Variación relativa mínima para que un signo signifique algo. Por debajo
#: de esto la correlación es INSENSIBLE a la variable y aprobar su signo
#: sería aprobar ruido de coma flotante (caso real: Rizk & Mongia global
#: da pendiente 1.4e-15 frente a φ_PZ — «> 0» y completamente vacío).
TREND_MIN_REL_SPAN = 0.02


def _slope_sign(xs, ys) -> float:
    """Pendiente por mínimos cuadrados (solo interesa el signo)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2 or not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _rel_span(ys) -> float:
    """(max−min)/|media| de la serie: cuánto se mueve realmente."""
    y = np.asarray(ys, dtype=float)
    if y.size < 2 or not np.all(np.isfinite(y)):
        return float("nan")
    m = abs(float(np.mean(y)))
    return float(np.ptp(y)) / max(m, 1e-30)


def run_emission_trends(case: dict, with_l1: bool = False) -> dict:
    """D2 · signo de ∂EI/∂φ_PZ y ∂EI/∂τ + orden de magnitud (×2)."""
    base = _theta(case)
    sweep = case["sweep"]
    keys = case["trend_keys"]
    checks, series = [], {}

    # --- (1) barrido de φ_PZ por frac_pz, en el lado POBRE ----------------
    phis, ph_vals = [], {k["key"]: [] for k in keys}
    for f in sweep["frac_pz"]:
        t = base.copy()
        rest = max(1.0 - f - 0.16, 1e-6)
        w_sz = 0.12 / (0.12 + 0.42)
        t[THETA_INDEX["frac_pz"]] = f
        t[THETA_INDEX["frac_sz"]] = rest * w_sz
        t[THETA_INDEX["frac_dz"]] = rest * (1.0 - w_sz)
        r = _eval(t)
        phis.append(float(r["phi_pz"]))
        for k in keys:
            ph_vals[k["key"]].append(float(r.get(k["key"], float("nan"))))

    # --- (2) barrido de τ por L_dz/D_L ------------------------------------
    taus, tau_vals = [], {k["key"]: [] for k in keys}
    for L in sweep["L_dz_over_DL"]:
        t = base.copy()
        t[THETA_INDEX["L_dz_over_DL"]] = L
        r = _eval(t)
        taus.append(float(r["tau_ms"]))
        for k in keys:
            tau_vals[k["key"]].append(float(r.get(k["key"], float("nan"))))

    series["phi_pz"] = dict(x=phis, vals=ph_vals)
    series["tau_ms"] = dict(x=taus, vals=tau_vals)

    # Solo el lado POBRE: en φ ≈ 1.05 la derivada del NOx es ≈ 0 por
    # física (pico de T_ad) y pedirle un signo sería pedir ruido.
    lean = [i for i, p in enumerate(phis) if p < 0.95]
    for k in keys:
        ys = [ph_vals[k["key"]][i] for i in lean]
        s = _slope_sign([phis[i] for i in lean], ys)
        span = _rel_span(ys)
        want = k["phi_sign"]
        flat = not (span > TREND_MIN_REL_SPAN)
        ok = (bool(np.isfinite(s)) and not flat
              and (s > 0 if want > 0 else s < 0))
        checks.append(check(
            case["cid"], f"signo ∂/∂φ_PZ · {k['label']}", ok,
            role="derived", value=s,
            target=("> 0 (Zeldovich: T_ad ↑ con φ en el lado pobre)"
                    if want > 0 else "< 0 (más T ⇒ más oxidación de CO)"),
            detail=(f"variación relativa {100 * span:.2f} % "
                    f"{'⇒ INSENSIBLE (< 2 %): el signo no significa nada'
                       if flat else ''} · φ_PZ ∈ [{_fmt(min(phis))}, "
                    f"{_fmt(max(phis))}], {len(lean)} puntos pobres")))
    for k in keys:
        ys = tau_vals[k["key"]]
        s = _slope_sign(taus, ys)
        span = _rel_span(ys)
        want = k["tau_sign"]
        flat = not (span > TREND_MIN_REL_SPAN)
        ok = (bool(np.isfinite(s)) and not flat
              and (s > 0 if want > 0 else s < 0))
        checks.append(check(
            case["cid"], f"signo ∂/∂τ · {k['label']}", ok,
            role="derived", value=s,
            target=("> 0 (más tiempo ⇒ más NOx térmico)" if want > 0
                    else "< 0 (más tiempo ⇒ más CO quemado)"),
            detail=(f"variación relativa {100 * span:.2f} % "
                    f"{'⇒ INSENSIBLE (< 2 %)' if flat else ''} · "
                    f"τ ∈ [{_fmt(min(taus))}, {_fmt(max(taus))}] ms")))

    # --- (3) orden de magnitud ×2 frente a la banda de referencia ---------
    rec = _eval(base)
    lo, hi = EI_NOX_MICRO_KERO_BAND
    fan = rec.get("EI_NOx_fan") or {}
    f_lo = float(fan.get("min", rec["EI_NOx"]))
    f_hi = float(fan.get("max", rec["EI_NOx"]))
    checks.append(check(
        case["cid"], "orden de magnitud EI_NOx (abanico L0)",
        not (f_hi < lo / EI_ORDER_FACTOR or f_lo > hi * EI_ORDER_FACTOR),
        role="derived", value=f"[{_fmt(f_lo)}, {_fmt(f_hi)}] g/kg",
        target=f"solapa [{lo / EI_ORDER_FACTOR:.3g}, "
               f"{hi * EI_ORDER_FACTOR:.3g}] g/kg (×{EI_ORDER_FACTOR:.0f})",
        detail=f"abanico de {fan.get('n_used', '?')} correlaciones, "
               f"dispersión ×{_fmt(fan.get('spread_ratio'))}"))

    # --- (4) L1 (CRN Cantera) — el número menos empírico -------------------
    if with_l1:
        r1 = _eval(base, fidelity=pc.Fidelity.L1)
        if str(r1.get("source", "")).startswith("crn_L1") \
                and "→L0" not in str(r1.get("source", "")):
            v = float(r1["EI_NOx_crn"])
            checks.append(check(
                case["cid"], "orden de magnitud EI_NOx (CRN L1)",
                lo / EI_ORDER_FACTOR <= v <= hi * EI_ORDER_FACTOR,
                role="derived", value=v,
                target=f"[{lo / EI_ORDER_FACTOR:.3g}, "
                       f"{hi * EI_ORDER_FACTOR:.3g}] g/kg",
                detail=f"source = {r1['source'][:40]} · "
                       f"EI_CO_crn = {_fmt(r1.get('EI_CO_crn'))} g/kg · "
                       f"η_comb_crn = {_fmt(r1.get('eta_comb_crn'))}"))
            series["L1"] = dict(EI_NOx_crn=v,
                                EI_CO_crn=float(r1["EI_CO_crn"]),
                                eta_comb_crn=float(r1["eta_comb_crn"]),
                                Da_pz=float(r1.get("Da_pz", float("nan"))))
        else:
            checks.append(check(
                case["cid"], "orden de magnitud EI_NOx (CRN L1)", None,
                role="declared", value=None, target="—",
                detail=f"NO EVALUABLE: el CRN degradó a "
                       f"'{r1.get('source')}'"))
    else:
        checks.append(check(
            case["cid"], "orden de magnitud EI_NOx (CRN L1)", None,
            role="declared", value=None, target="—",
            detail="NO EVALUADO: L1 cuesta ~2.2 s/punto — usa `--l1`"))

    return dict(case=case, rec=rec, theta=base, checks=checks,
                series=series)


# ---------------------------------------------------------------------------
# Anclas de regresión (viven en validation/anchors.json — NO son mediciones)
# ---------------------------------------------------------------------------
ANCHOR_RTOL = 1e-6


def run_anchors() -> list[dict]:
    """Reevalúa las anclas congeladas y declara las que faltan y por qué."""
    frozen = {}
    if os.path.isfile(ANCHORS_JSON):
        try:
            with open(ANCHORS_JSON, encoding="utf-8") as f:
                frozen = (json.load(f) or {}).get("anchors", {}) or {}
        except Exception as exc:                       # nunca revienta
            frozen = {}
            print(f"  aviso: no se pudo leer {ANCHORS_JSON}: {exc}")

    out = []
    for spec in EXPECTED_ANCHORS:
        name = spec["name"]
        a = frozen.get(name)
        if a is None:
            out.append(dict(name=name, ok=None, present=False,
                            required=spec["required"], why=spec["why"],
                            diffs={}))
            continue
        theta = np.asarray(a["theta"], dtype=float)
        fid = getattr(pc.Fidelity, str(a.get("fidelity", "L0")).upper())
        rec = _eval(theta, fidelity=fid)
        diffs = {}
        for k, v in a.items():
            if k in ("theta", "fidelity") or not isinstance(v, (int, float)):
                continue
            got = rec.get(k)
            if not isinstance(got, (int, float)):
                continue
            diffs[k] = (float(got), float(v),
                        abs(float(got) - float(v)) / max(abs(float(v)), 1e-12))
        ok = all(d[2] < ANCHOR_RTOL for d in diffs.values()) and bool(diffs)
        out.append(dict(name=name, ok=ok, present=True,
                        required=spec["required"], why=spec["why"],
                        diffs=diffs, source=rec.get("source")))
    return out


# ---------------------------------------------------------------------------
# UQ mínima (WAVE7 §3.5)
# ---------------------------------------------------------------------------
def uq_report(rec: dict, sigma_ens: dict | None = None) -> list[dict]:
    """Bandas de incertidumbre: σ_tot = √(σ_modelo² + σ_ensemble²).

    APROXIMACIÓN DECLARADA (se repite en RESULTS.md y en VALIDATION.md):
    supone errores relativos, gaussianos, independientes y un modelo lineal
    en el entorno del punto. Ninguna de las cuatro está demostrada. En la
    campaña σ_ensemble = 0 porque no hay surrogate entrenado: la banda es
    **solo** incertidumbre de modelo físico.
    """
    sigma_ens = sigma_ens or {}
    out = []
    for key, s_m in UQ_SIGMA_REL.items():
        v = rec.get(key)
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            continue
        s_e = float(sigma_ens.get(key, 0.0))
        s_t = math.sqrt(s_m ** 2 + s_e ** 2)
        out.append(dict(key=key, value=float(v), sigma_model=s_m,
                        sigma_ens=s_e, sigma_total=s_t,
                        lo=float(v) * (1.0 - s_t), hi=float(v) * (1.0 + s_t),
                        kind="sigma"))
    for key, sw in UQ_MEASURED_SWEEP.items():
        v = rec.get(key)
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            continue
        lo = sw.get("lo_K", float(v) * (1.0 + sw.get("lo_rel", 0.0)))
        hi = sw.get("hi_K", float(v) * (1.0 + sw.get("hi_rel", 0.0)))
        out.append(dict(key=key, value=float(v), lo=float(lo), hi=float(hi),
                        driver=sw["driver"], note=sw["note"],
                        kind="barrido medido"))
    return out


# ---------------------------------------------------------------------------
# Ejecución y reporte
# ---------------------------------------------------------------------------
RUNNERS = {
    "class_bands": run_class_bands,
    "table_consistency": run_table_consistency,
    "split_anchor": run_split_anchor,
    "thermal_extreme": run_thermal_extreme,
    "emission_trends": run_emission_trends,
}


def run_case(case: dict, with_l1: bool = False) -> dict:
    fn = RUNNERS[case["kind"]]
    return fn(case, with_l1) if case["kind"] == "emission_trends" else fn(case)


def _tally(results: list[dict]) -> dict:
    n_pass = n_fail = n_ne = n_gap = n_stale = 0
    for r in results:
        for c in r["checks"]:
            if c["ok"] is None:
                n_ne += 1
            elif c["ok"]:
                n_pass += 1
                n_stale += bool(c["stale_gap"])
            else:
                n_fail += 1
                n_gap += bool(c["known_gap"])
    return dict(passed=n_pass, failed=n_fail, non_evaluable=n_ne,
                known_gaps=n_gap, stale_gaps=n_stale)


def _env_line() -> str:
    opt = []
    for mod in ("cantera", "jax", "xlb", "warp", "matplotlib"):
        try:
            __import__(mod)
            opt.append(f"{mod}=sí")
        except Exception:
            opt.append(f"{mod}=NO")
    return (f"Python {sys.version.split()[0]} · NumPy {np.__version__} · "
            + " · ".join(opt))


def write_results_md(results: list[dict], anchors: list[dict],
                     uq: list[dict], path: str, with_l1: bool) -> None:
    t = _tally(results)
    L = [
        "# Resultados de validación — Quasar Phy-CB",
        "",
        f"Generado por `validation/validate.py` el {date.today()} "
        f"({datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}). "
        "**Regenerar tras cualquier cambio en `physics_core.py` o en "
        "`data/`** — este archivo se versiona como evidencia.",
        "",
        f"Entorno de esta corrida: {_env_line()}",
        "",
        "Modo: `fidelity=L0`, `use_cache=False`, **`calibrate=False`** (la "
        "campaña mide el modelo desnudo; con la calibración afín activa "
        "estaríamos validando la calibración, no la física). "
        + ("L1 (CRN Cantera) INCLUIDO (`--l1`)." if with_l1
           else "L1 (CRN Cantera) no incluido en esta corrida (`--l1`)."),
        "",
        "## 0. Marcador",
        "",
        f"| PASS | FAIL | de ellos huecos conocidos | NO EVALUABLE | "
        f"huecos obsoletos |",
        "|---|---|---|---|---|",
        f"| {t['passed']} | {t['failed']} | {t['known_gaps']} | "
        f"{t['non_evaluable']} | {t['stale_gaps']} |",
        "",
        "*Huecos obsoletos* = checks marcados `known_gap` en "
        "`validate.KNOWN_GAPS` que hoy PASAN. Si este número no es 0, el "
        "documento miente y hay que quitar la entrada.",
        "",
        "Leyenda de `rol`: **derived** = lo calcula el modelo ⇒ puede "
        "validar o falsar · **input** = se lo damos nosotros ⇒ solo "
        "trazabilidad, jamás evidencia · **data** = se valida un dato del "
        "paquete, no el modelo · **declared** = criterio no evaluable.",
        "",
        "## 1. Resumen por caso",
        "",
        "| Caso | Nivel | Nombre | PASS | FAIL | N/E |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        cs = r["checks"]
        p = sum(1 for c in cs if c["ok"] is True)
        f = sum(1 for c in cs if c["ok"] is False)
        n = sum(1 for c in cs if c["ok"] is None)
        L.append(f"| `{r['case']['cid']}` | {r['case']['level']} | "
                 f"{r['case']['name']} | {p} | {f} | {n} |")

    L += ["", "## 2. Detalle por caso", ""]
    for r in results:
        c0 = r["case"]
        L += [f"### `{c0['cid']}` · {c0['name']}", "",
              f"**Fuente.** {c0['source']}", "", "**Supuestos declarados.**",
              ""]
        for a in c0.get("assumptions", []):
            L.append(f"* {a}")
        L += ["", f"**Notas.** {c0.get('notes', '—')}", "",
              "| Check | rol | valor | referencia | veredicto | detalle |",
              "|---|---|---|---|---|---|"]
        for c in r["checks"]:
            v = _STATUS_TXT[c["ok"]]
            if c["ok"] is False and c["known_gap"]:
                v = "FAIL (hueco conocido)"
            if c["stale_gap"]:
                v = "PASS ⚠️ hueco obsoleto"
            L.append(f"| {c['label']} | {c['role']} | {_fmt(c['value'])} | "
                     f"{c['target'] or '—'} | {v} | {c['detail'] or '—'} |")
        gaps = [c for c in r["checks"] if c["ok"] is False and c["known_gap"]]
        if gaps:
            L += ["", "**Huecos conocidos de este caso** (explicados en "
                  "`docs/VALIDATION.md`):", ""]
            for c in gaps:
                L.append(f"* `{c['label']}` — {c['known_gap']}")
        if c0["cid"] == "C1":
            L += ["", "| punto | frac_pz | φ_PZ publicada | φ_PZ modelo | Δ |",
                  "|---|---|---|---|---|"]
            for p in r["points"]:
                d = (p["phi_mod"] - p["phi_pub"]) / p["phi_pub"]
                L.append(f"| {p['label']} | {p['frac_pz']:.4f} | "
                         f"{p['phi_pub']:.3f} | {p['phi_mod']:.4f} | "
                         f"{100 * d:+.2f} % |")
        if c0["cid"] == "D2":
            s = r["series"]
            ks = [k["key"] for k in c0["trend_keys"]]
            for tag, xlab in (("phi_pz", "φ_PZ"), ("tau_ms", "τ [ms]")):
                blk = s[tag]
                L += ["", f"**Barrido de {xlab}** "
                      + ("(por `frac_pz`, lado pobre)" if tag == "phi_pz"
                         else "(por `L_dz/D_L`)") + " — todo en g/kg:", "",
                      "| " + xlab + " | " + " | ".join(ks) + " |",
                      "|" + "---|" * (len(ks) + 1)]
                for i, x in enumerate(blk["x"]):
                    L.append(f"| {x:.4f} | " + " | ".join(
                        f"{blk['vals'][k][i]:.4g}" for k in ks) + " |")
        L.append("")

    L += ["## 3. Anclas de regresión (NO son mediciones)", "",
          "Congelan la salida ACTUAL del modelo para detectar deriva "
          "silenciosa de la física. Se congelan con "
          "`python test_phycb.py --freeze-anchors` y moverlas exige "
          "justificación escrita (contrato §10.9).", "",
          "| Ancla | Estado | Detalle |", "|---|---|---|"]
    for a in anchors:
        if not a["present"]:
            est = "AUSENTE (obligatoria)" if a["required"] else "AUSENTE"
            L.append(f"| {a['name']} | {est} | {a['why']} |")
        else:
            det = "; ".join(f"{k} = {got:.6g} (esp. {exp:.6g})"
                            for k, (got, exp, _) in
                            sorted(a["diffs"].items()))
            L.append(f"| {a['name']} | {_STATUS_TXT[a['ok']]} | {det} |")

    L += ["", "## 4. UQ mínima — bandas sobre el caso ancla REF_CB10", "",
          "σ_total = √(σ_modelo² + σ_ensemble²). **APROXIMACIÓN "
          "DECLARADA**: supone errores relativos, gaussianos, "
          "independientes y modelo lineal en el entorno del punto; ninguna "
          "de las cuatro está demostrada. σ_ensemble = 0 en la campaña "
          "(no hay surrogate entrenado) ⇒ la banda es **solo** "
          "incertidumbre de modelo físico.", "",
          "| Magnitud | valor | tipo | banda | origen |",
          "|---|---|---|---|---|"]
    for u in uq:
        if u["kind"] == "sigma":
            org = (f"σ_modelo = ±{100 * u['sigma_model']:.0f} %, "
                   f"σ_ens = ±{100 * u['sigma_ens']:.0f} %")
        else:
            org = f"{u['driver']} — {u['note']}"
        L.append(f"| {u['key']} | {_fmt(u['value'])} | {u['kind']} | "
                 f"[{_fmt(u['lo'])}, {_fmt(u['hi'])}] | {org} |")

    L += ["", "## 5. Criterios v1 que NO son evaluables (declarados)", "",
          "Regla de la casa: un criterio que no se puede medir se DECLARA; "
          "nunca se convierte en un check que aprueba trivialmente.", "",
          "| Criterio | Por qué no |", "|---|---|"]
    for k, v in NON_EVALUABLE.items():
        L.append(f"| `{k}` | {v} |")

    L += ["", "## 6. Niveles de la escalera declarados y no corridos", "",
          "| Nivel | Caso | Motivo |", "|---|---|---|"]
    for d in DECLARED_NOT_RUN:
        L.append(f"| {d['level']} | {d['name']} | {d['why']} |")

    L += ["", "## 7. Cómo añadir un caso", "",
          "1. Reunir el dato publicado **con su URL o cita**; lo que la "
          "fuente no diga va en `assumptions` con la palabra SUPUESTO.",
          "2. Añadir la entrada a `validation/cases.py` marcando cada "
          "parámetro comparado como `derived` o `input`.",
          "3. `python validation/validate.py` y versionar este archivo.",
          "4. Si algo falla: **no se recalibra por caso**. Se abre entrada "
          "en `docs/VALIDATION.md` §prioridad de recalibración, se toca "
          "**una** constante global y se re-corren TODOS los casos.", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Campaña de validación de Phy-CB (WAVE7)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 si hay fallo duro (ver política arriba)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--only", default="",
                    help="subconjunto de cids separados por coma")
    ap.add_argument("--json", default="",
                    help="volcado legible a máquina")
    ap.add_argument("--l1", action="store_true",
                    help="incluye el CRN L1 (Cantera, ~2 s/punto)")
    args = ap.parse_args(argv)

    _assert_theta_layout()
    solo = {s.strip().upper() for s in args.only.split(",") if s.strip()}

    print("=" * 78)
    print("QUASAR Phy-CB · campaña de VALIDACIÓN (WAVE7)")
    print(f"  physics_core : {pc.__file__}")
    print(f"  PROJECT={pc.PROJECT}  NDIM={pc.NDIM}  SEED={pc.SEED}")
    print(f"  {_env_line()}")
    print("=" * 78)

    results = []
    for case in CASES:
        if solo and case["cid"].upper() not in solo:
            continue
        results.append(run_case(case, with_l1=args.l1))

    for r in results:
        c0 = r["case"]
        print(f"\n— [{c0['cid']}] nivel {c0['level']} · {c0['name']}")
        for c in r["checks"]:
            mark = {True: "✓", False: "✗", None: "–"}[c["ok"]]
            tag = ""
            if c["ok"] is False and c["known_gap"]:
                tag = "  (hueco conocido)"
            if c["stale_gap"]:
                tag = "  (⚠️ HUECO OBSOLETO: quítalo de KNOWN_GAPS)"
            print(f"   {mark} {c['label']:<44s} {_fmt(c['value']):>12s}  "
                  f"vs {c['target']}{tag}")

    anchors = run_anchors()
    print("\n— anclas de regresión")
    for a in anchors:
        mark = {True: "✓", False: "✗", None: "–"}[a["ok"]]
        est = "OK" if a["ok"] else ("AUSENTE" if not a["present"] else "MOVIDA")
        print(f"   {mark} {a['name']:<16s} {est}")

    ref = _eval(np.asarray(pc.THETA_REF, dtype=float))
    uq = uq_report(ref)
    print("\n— UQ mínima sobre REF_CB10 (σ_ens = 0, aproximación declarada)")
    for u in uq:
        print(f"   {u['key']:<14s} {_fmt(u['value']):>10s}  "
              f"[{_fmt(u['lo'])}, {_fmt(u['hi'])}]  ({u['kind']})")

    t = _tally(results)
    print(f"\nMARCADOR  PASS={t['passed']}  FAIL={t['failed']} "
          f"(de ellos {t['known_gaps']} huecos conocidos)  "
          f"N/E={t['non_evaluable']}  huecos obsoletos={t['stale_gaps']}")

    if args.no_write:
        pass
    elif solo:
        # RESULTS.md es la EVIDENCIA versionada: reescribirla con un
        # subconjunto la truncaría en silencio. `--only` es para depurar.
        print("\n(no se reescribe RESULTS.md: --only corre un subconjunto; "
              "usa la campaña completa para regenerar la evidencia)")
    else:
        write_results_md(results, anchors, uq, RESULTS_MD, args.l1)
        print(f"→ {RESULTS_MD}")
    if args.json:
        payload = dict(
            schema="phycb-validation-1", date=str(date.today()),
            env=_env_line(), tally=t,
            cases=[dict(cid=r["case"]["cid"], level=r["case"]["level"],
                        name=r["case"]["name"],
                        checks=[{k: v for k, v in c.items()
                                 if k != "known_gap"} for c in r["checks"]])
                   for r in results],
            anchors=[{k: v for k, v in a.items() if k != "diffs"}
                     for a in anchors],
            uq=uq)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
        print(f"→ {args.json}")

    unexpected = [c for r in results for c in r["checks"]
                  if c["blocking"] and c["ok"] is False and not c["known_gap"]]
    stale = [c for r in results for c in r["checks"] if c["stale_gap"]]
    anchor_bad = [a for a in anchors
                  if a["ok"] is False or (a["required"] and not a["present"])]
    hard = bool(unexpected or stale or anchor_bad)
    if hard:
        print("\nSTRICT: fallo duro —")
        for c in unexpected:
            print(f"  · [{c['cid']}] {c['label']}: FAIL no documentado en "
                  f"KNOWN_GAPS ni en docs/VALIDATION.md")
        for c in stale:
            print(f"  · [{c['cid']}] {c['label']}: hueco conocido que ahora "
                  f"PASA — quítalo de KNOWN_GAPS y de docs/VALIDATION.md")
        for a in anchor_bad:
            print(f"  · ancla {a['name']}: "
                  f"{'ausente' if not a['present'] else 'movida'}")
    return 1 if (args.strict and hard) else 0


if __name__ == "__main__":
    sys.exit(main())
