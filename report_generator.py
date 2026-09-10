"""
QUASAR Phy-CB · report_generator.py
===================================
CEM layer 5b: **self-contained HTML report** of the combustion-chamber
design (WAVE1 §2 + WAVE4 §4).

One single `.html` file, `lang="en"`, embedded CSS, **zero dependencies**:
every chart is an SVG generated in pure Python (same technique as the
`_svg_*` of `Phy-AC/report_generator.py`), with no matplotlib and no
JavaScript. The matplotlib figures of `visualization.py` are an optional
EXTRA: if they are there they are embedded in base64, and if not, the
report is still complete.

Numbered sections
-----------------
  1  Specification and verdict
  2  Sizing and air split (SVG sankey of the splits)
  3  Holes and atomization (row table + SMD fan)
  4  Liner thermal (axial T_w profile in SVG)
  5  Stability (phi_LBO, margin, ignition energy)
  6  Emissions (the correlation FAN, side by side)
  7  g constraints and feasibility (aero + spec + structural + DfAM)
  8  Traceability (`run_meta`, `source`, seed, versions)
  9  Verified Pareto front                  [WAVE4]
 10  Surrogate round history                [WAVE4]
 11  L0 vs L1 comparison of the winners     [WAVE4]
 12  Integration with the compressor        [WAVE4]
 13  DfAM (checker, orientation, drains)    [WAVE4]
     + final «DECLARED LIMITS» block

House rules this module follows to the letter
---------------------------------------------
* **Calibrated honesty**: EVERY number carries the fidelity it came from
  (L0 · L0s · L1 · L1.5 · L2a · L2b · spec · DfAM) glued to the value.
* **Never raises**: if a record key is missing the row is painted with a
  dash (`—`); if a whole section is missing, it is painted with the reason
  why it is empty. `generate_report` always returns the written path.
* **Every section is always emitted** (even when empty): the report is a
  reading contract, not a variable template — and test T15 checks exactly
  that.
* SI internally; mm/deg only at the interface, with a key suffix.
* `open()` always with `encoding="utf-8"`.

Public API
----------
``generate_report(spec, result, path, structural=None, contract=None,
dfam=None, figures=None, log=print) -> str`` (returns the written path).
"""

from __future__ import annotations

import base64
import html
import json
import math
import os
import sys
from datetime import datetime
from string import Template

import numpy as np

from physics_core import (CONSTRAINT_NAMES, EI_NOX_PLAUSIBLE_MAX,
                          N_CONSTRAINTS, SEED, VAR_NAMES, provisional_report)

# ==========================================================================
# 0. Presentation tunables (SCREAMING_CASE with provenance)
# ==========================================================================
#: Product version. Mirror of `pyproject.toml [project].version` — if it is
#: bumped there, it is bumped here (there is no `__version__` in the core).
PHYCB_VERSION = "0.1.0"

#: Palette. Consistent with `visualization.py` (_C_BRAND/_C_HOT/…) so the
#: embedded matplotlib figures do not clash with the native SVGs.
ACCENT = "#2e6ff2"
INK = "#1a1d27"
MUT = "#6b7280"
OK = "#0a8f5b"
BAD = "#c43d3d"
WARN = "#c2761b"
HOT = "#d6453c"
COOL = "#3bb0a8"
GOLD = "#f2a73b"
GRID = "#e5e7eb"

#: Nominal SVG width [viewBox px]. They all scale with `width:100%`.
SVG_W = 620

#: «Missing data» dash. A missing value is NEVER invented and NEVER set to
#: zero: it is painted with this character (honesty rule, contract §7).
DASH = "—"

#: Maximum number of provisional coefficients listed unfolded in the limits
#: block (the rest goes inside a `<details>`, pure HTML, no JS).
PROVISIONAL_INLINE_MAX = 6

#: Maximum number of Pareto-front points tabulated under the SVG.
PARETO_TABLE_MAX = 24


# ==========================================================================
# 1. Source fidelity — the label that travels with every number
# ==========================================================================
#: Keys that are ONLY born in L1 (contract §4.3). If they are in the record,
#: the number that uses them is labelled L1 even if the rest of the record
#: is L0.
L1_KEYS = frozenset({
    "splits_solved", "v_zones", "tau_zones_ms", "T_zones_K", "EI_NOx_crn",
    "EI_CO_crn", "eta_comb_crn", "Da_pz", "LBO_margin_Da", "crn_topology",
})
#: L1.5 keys (cold aero LBM) — contract §4.4.
L15_KEYS = frozenset({
    "splits_measured", "Y_over_d_measured", "V_zones_measured",
})
#: L2a/L2b keys (CFD) — contract §4.4.
L2_KEYS = frozenset({
    "PF_cfd", "T4_cfd_K", "dp_over_p_cfd", "yplus_hist", "cfd_mesh",
})

#: Level → CSS class of the fidelity «chip».
_FID_CLASS = {
    "L0": "f-l0", "L0s": "f-l0s", "L1": "f-l1", "L1.5": "f-l15",
    "L2a": "f-l2", "L2b": "f-l2", "spec": "f-spec", "DfAM": "f-dfam",
    "geom": "f-spec", "meta": "f-spec",
}

#: `record["fidelity"]` (IntEnum int) → readable label.
_FID_BY_INT = {0: "L0", 1: "L1", 2: "L1.5", 3: "L2a", 4: "L2b"}


def _fid_of(key: str) -> str:
    """SOURCE fidelity of a record key.

    L0 ALWAYS runs (inside L1/L2 too): a sizing number is still L0 even if
    the point was verified with a CRN. Only the keys that are born higher up
    carry the higher label.
    """
    if key in L1_KEYS:
        return "L1"
    if key in L15_KEYS:
        return "L1.5"
    if key in L2_KEYS:
        return "L2a"
    return "L0"


def _chip(fid: str | None) -> str:
    """Fidelity pill. `None` ⇒ no pill."""
    if not fid:
        return ""
    cls = _FID_CLASS.get(fid, "f-l0")
    return (f'<span class="fid {cls}" title="source fidelity">'
            f"{_esc(fid)}</span>")


# ==========================================================================
# 2. Fidelity ladder — the EXACT table of Ciencia §10
# ==========================================================================
#: (level, model, cost, predicts reliably, does NOT predict). Transcribed
#: literally from `docs/Quasar_PhyCB_Science.md` §10 — if that table
#: changes, this one changes too.
FIDELITY_LADDER = (
    ("L0", "Correlations §§1–8 + equilibrium (Cantera)", "~ms",
     "T_ad, T4, FAR, sizing, nominal splits, Δp/p, trends of "
     "η/PF/LBO/EI",
     "nothing spatial"),
    ("L1", "1-D network + CRN", "0.1–2 s",
     "real 1-D splits, τ per zone, NOx/CO trends (±30–50 %), LBO through "
     "Da, η_comb",
     "PF, local wall temperature"),
    ("L1.5", "Cold aero LBM (XLB) over PicoGK voxels", "s–min (GPU)",
     "REAL 3-D air splits, jet penetration, recirculation topology → "
     "automatic CRN topology",
     "everything thermal (the per-zone density is corrected from L0)"),
    ("L2a", "Native reactive RANS (IB over SDF)", "5–30 min (GPU)",
     "flow topology, dilution penetration, ranking of PF and hot spots "
     "between designs, Δp/p ±15–20 %, T4 ±3–5 %",
     "absolute NOx (×2–10), absolute PF, quantitative LBO, "
     "thermoacoustics, quantitative metal temperature"),
    ("L2b", "OpenFOAM (reactingFoam) — OUT OF SCOPE (contract §12)",
     "not available",
     "—  (it would have added CHT + radiation ⇒ quantitative metal "
     "temperature)",
     "EVERYTHING: withdrawn from the system; Phy-CB predicts no metal "
     "temperature at any level"),
)

#: FIXED honesty label for any L2 result (Ciencia §10).
L2_HONESTY_LABEL = ("Design comparison / hot-spot detection — not fit for "
                    "emissions certification")


# ==========================================================================
# 3. Correlation metadata (verification mark of Ciencia §§7–8)
# ==========================================================================
#: (record key, human name, mark, provenance/note).
#: Marks: "✅" verified · "⚠️" provisional · "L1" only exists with a CRN.
EMISSION_MODELS = (
    ("EI_NOx_lefebvre", "NOx · Lefebvre", "✅",
     "Ciencia §8.1 — verified (U. Toronto thesis, eq. 1.4). It is the "
     "record's headline `EI_NOx`."),
    ("EI_NOx_rizk_mongia", "NOx · Rizk & Mongia (global)", "✅",
     "Ciencia §8.3 — verified (U. Toronto thesis, eqs. 3.5 and 1.5–1.8)."),
    ("EI_NOx_zonal", "NOx · Rizk & Mongia (zonal)", "✅",
     "Ciencia §8.3 — structure verified; VERY sensitive to T_st through "
     "the exp(−Ea/T)."),
    ("EI_NOx_odgers", "NOx · Odgers & Kretschmer", "✅",
     "Ciencia §8.4 — verified against a peer-reviewed review (MDPI 2024); "
     "saturates for τ > ~20 ms."),
    ("EI_NOx_laviolette", "NOx · Laviolette & Perez", "✅",
     "Ciencia §8.4 — optimised variant of the same source."),
    ("EI_NOx_crn", "NOx · CRN (Cantera)", "L1",
     "Ciencia §9.2 — only present if the point was verified at L1."),
    ("EI_CO_lefebvre", "CO · Lefebvre", "⚠️",
     "Ciencia §8.2 — NOT verified: the coefficients 86 and −0.00345 are "
     "still to be checked against the book."),
    ("EI_CO_crn", "CO · CRN (Cantera)", "L1",
     "Ciencia §9.2 — only present if the point was verified at L1."),
)

#: Same for the SMD fan (Ciencia §7).
SMD_MODELS = (
    ("elshanawany_lefebvre",
     "Prefilming airblast · El-Shanawany & Lefebvre (1980)", "✅",
     "DIMENSIONAL form verified (arXiv:1906.04042, table 7)"),
    ("rizkalla_lefebvre",
     "Prefilming airblast · Rizkalla & Lefebvre (1975)", "✅",
     "dimensional form verified (same source)"),
    ("prefilm_nondimensional",
     "Non-dimensional prefilming airblast (0.33 / 0.068)", "⚠️",
     "NOT verified — it is computed and reported, but it is NOT the chosen "
     "model"),
    ("plain_jet_rizk_lefebvre", "Plain-jet airblast (0.48 / 0.15)", "⚠️",
     "coefficients still to be verified"),
    ("chaussonnet_2016", "Airblast · Chaussonnet (2016)", "✅",
     "verified; especially useful at micro scale"),
    ("simplex_lefebvre", "Pressure-swirl (simplex)", "⚠️",
     "the 2.25 coefficient is pending; the SMD ∝ ΔP^−0.5 dependence IS ✅"),
    ("vaporizer_kj66_d50", "Vaporizer (KJ66 reference d50)", "✅",
     "KJ66-class rule (Ciencia §7.3)"),
    ("gas_jet_no_smd", "Gaseous branch (no spray)", "—",
     "H₂ / micromix: SMD does not apply, the fuel enters already gaseous"),
)

#: Human name + form of each aero g (contract §5.1, FROZEN order).
G_AERO_DESC = {
    "PF": ("Pattern factor", "PF / PF_max − 1"),
    "T_wall": ("Flame-side wall temperature",
               "T_w1 / (T_max_wall − margin) − 1"),
    "LBO": ("Lean-blowout margin", "φ_LBO·(1+margin) / φ_PZ − 1"),
    "tau_min": ("Minimum residence time", "τ_min / τ − 1"),
    "tau_max": ("Maximum residence time", "τ / τ_max − 1"),
    "M_exit": ("Exit Mach (turbine-friendly)", "M_exit / M_max − 1"),
    "dp": ("Total pressure loss", "Δp/p / Δp/p_max − 1"),
    "t_liner": ("Printable liner thickness (LPBF)",
                "t_min_LPBF / t_liner − 1"),
    "d_eff": ("Effusion hole Ø", "Ø_min / d_eff − 1"),
    "phi_pz_rich": ("Primary-zone richness", "φ_PZ / φ_PZ_max − 1"),
    "eta_comb": ("Combustion efficiency", "η_min / η_comb − 1"),
    "mixing": ("Dilution penetration (mixing)",
               "|Y/D_L − target| / tolerance − 1"),
}

#: g of the USER's spec (contract §5.4: FIXED length 4).
G_SPEC_DESC = (
    ("dp_spec", "Δp/p ≤ the spec's dp_max", "Δp/p / dp_max − 1"),
    ("PF_spec", "PF ≤ the spec's PF_max", "PF / PF_max − 1"),
    ("envelope_D", "Casing Ø within the envelope", "D_casing / D_env − 1"),
    ("envelope_L", "Length within the envelope", "L_tot / L_env − 1"),
)

#: Structural g (contract §5.2) and DfAM g (§5.3).
G_STRUCT_DESC = (
    ("T_wall", "Wall T vs the derated material limit"),
    ("sigma_th", "Thermal stress vs k(T)·σ_y"),
    ("hoop", "Hoop stress, liner + casing"),
    ("t_wall", "Thickness ≥ the material's LPBF t_min"),
    ("banned", "Material banned on the hot wall"),
)
G_DFAM_DESC = (
    ("t_min", "Minimum printable thickness (max(0.6 mm, 2·voxel))"),
    ("d_hole_min", "Minimum hole Ø (0.5 mm)"),
    ("overhang", "Overhangs ≥ 45° or self-supporting"),
    ("drainability", "Powder drainability (≥2 exits per cavity)"),
    ("effusion_band", "Effusion pitch/porosity within the printable band"),
)


# ==========================================================================
# 4. Formatting helpers — they NEVER raise, they return `DASH` if the datum
#    is missing
# ==========================================================================
def _esc(x) -> str:
    """`html.escape`, tolerant to any type."""
    try:
        return html.escape(str(x))
    except Exception:
        return DASH


def _fin(x) -> float | None:
    """Finite float or `None` (accepts numeric str, np.float64, bool…)."""
    try:
        if x is None or isinstance(x, (dict, list, tuple)):
            return None
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


#: Thousands separator of the report. The document is in English, so the
#: comma that `format(…, ",")` produces IS the thousands separator and is
#: kept as such. (The Spanish edition of this file replaced it with a narrow
#: no-break space, because there the comma is the DECIMAL separator and
#: «250,000 Pa» would have read as 250.)
THOUSANDS_SEP = ","


def _num(value, fmt: str = "{:.3f}", scale: float = 1.0,
         unit: str = "") -> str:
    """Formats a number; `DASH` if it is not finite."""
    v = _fin(value)
    if v is None:
        return DASH
    try:
        txt = fmt.format(v * scale)
    except Exception:
        return DASH
    if "," in fmt:
        txt = txt.replace(",", THOUSANDS_SEP)
    return _esc(txt) + (f" {_esc(unit)}" if unit else "")


def _get(rec, key, fmt: str = "{:.3f}", scale: float = 1.0,
         unit: str = "") -> str:
    """Formatted value of `rec[key]`. Missing key ⇒ `DASH`."""
    if not isinstance(rec, dict):
        return DASH
    return _num(rec.get(key), fmt, scale, unit)


def _txt(rec, key) -> str:
    """Escaped text value of `rec[key]`; `DASH` if missing or empty."""
    if not isinstance(rec, dict):
        return DASH
    v = rec.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return DASH
    return _esc(v)


def _row(label, value, fid: str | None = None, note: str = "") -> str:
    """Label → value row (+ fidelity chip + note at the foot of the cell).

    `value` is inserted AS IS: whoever builds it must escape it.
    """
    note_html = f'<div class="mut">{_esc(note)}</div>' if note else ""
    return (f"<tr><td>{_esc(label)}{note_html}</td>"
            f'<td class="v"><b>{value}</b>{_chip(fid)}</td></tr>')


def _krow(rec, key, label, fmt="{:.3f}", scale=1.0, unit="",
          fid: str | None = None, note: str = "") -> str:
    """Row straight from the record, fidelity deduced from the key."""
    return _row(label, _num((rec or {}).get(key), fmt, scale, unit),
                fid or _fid_of(key), note)


def _table(rows: str, head: str = "") -> str:
    if not rows:
        return '<p class="mut">(no data)</p>'
    return f"<table>{head}{rows}</table>"


def _verdict_span(ok: bool, text_ok: str, text_bad: str) -> str:
    color = OK if ok else BAD
    return (f'<span class="verdict" style="background:{color}">'
            f"{_esc(text_ok if ok else text_bad)}</span>")


def _g_cell(g: float | None) -> str:
    v = _fin(g)
    if v is None:
        return f'<span class="mut">{DASH}</span>'
    cls = "ok" if v <= 0.0 else "bad"
    return f'<span class="{cls}">{v:+.4f}</span>'


def _g_verdict(g: float | None) -> str:
    v = _fin(g)
    if v is None:
        return f'<span class="mut">{DASH}</span>'
    return ('<span class="ok">met</span>' if v <= 0.0
            else '<span class="bad">VIOLATED</span>')


def _note(text: str) -> str:
    return f'<p class="mut">{_esc(text)}</p>'


def _lbo_margin_min() -> float:
    """The core's `LBO_MARGIN_MIN` (⚠️ provisional). Single source of truth:
    the report does NOT redefine tunables, it reads them."""
    try:
        import physics_core as _pc
        v = _fin(getattr(_pc, "LBO_MARGIN_MIN", None))
        return v if v is not None else 0.20
    except Exception:
        return 0.20


def _empty(reason: str) -> str:
    """Body of a section with no data: it is emitted anyway, with the why."""
    return (f'<p class="mut">(no data in this report — {_esc(reason)})</p>')


# ==========================================================================
# 5. SVG scaffolding (pure Python — no matplotlib, no JS)
# ==========================================================================
def _span(lo: float, hi: float, pad: float = 0.06) -> tuple[float, float]:
    """Range with padding and a guaranteed non-zero width."""
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return 0.0, 1.0
    d = hi - lo
    if d <= 1e-12:
        d = max(abs(hi), 1.0) * 0.1
    return lo - pad * d, hi + pad * d


def _frame(w: int, h: int, x0: float, x1: float, y0: float, y1: float,
           xlab: str, ylab: str, xfmt: str = "{:.2f}", yfmt: str = "{:.3f}",
           ml: int = 66, mb: int = 42, mt: int = 18, mr: int = 18,
           nticks: int = 5):
    """Cartesian frame: grid, ticks and labels. Returns (parts, X, Y)."""
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0

    def X(v: float) -> float:
        return ml + (float(v) - x0) / dx * (w - ml - mr)

    def Y(v: float) -> float:
        return h - mb - (float(v) - y0) / dy * (h - mb - mt)

    s: list[str] = []
    for i in range(nticks):
        gx = x0 + i * dx / (nticks - 1)
        gy = y0 + i * dy / (nticks - 1)
        s.append(f'<line x1="{X(gx):.1f}" y1="{mt}" x2="{X(gx):.1f}" '
                 f'y2="{h - mb}" stroke="{GRID}"/>')
        s.append(f'<line x1="{ml}" y1="{Y(gy):.1f}" x2="{w - mr}" '
                 f'y2="{Y(gy):.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{X(gx):.1f}" y="{h - mb + 16}" font-size="11" '
                 f'text-anchor="middle" fill="{MUT}">'
                 f'{_esc(xfmt.format(gx))}</text>')
        s.append(f'<text x="{ml - 6}" y="{Y(gy) + 4:.1f}" font-size="11" '
                 f'text-anchor="end" fill="{MUT}">'
                 f'{_esc(yfmt.format(gy))}</text>')
    s.append(f'<text x="{(w + ml - mr) / 2:.0f}" y="{h - 6}" font-size="12" '
             f'text-anchor="middle" fill="{INK}">{_esc(xlab)}</text>')
    cy = (h - mb + mt) / 2
    s.append(f'<text x="14" y="{cy:.0f}" font-size="12" text-anchor="middle" '
             f'fill="{INK}" transform="rotate(-90 14 {cy:.0f})">'
             f'{_esc(ylab)}</text>')
    return s, X, Y


def _svg_open(w: int, h: int) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'style="max-width:{w}px;width:100%;height:auto">')


def _bar_chart(items: list[tuple[str, float | None, str, str]],
               unit: str, w: int = SVG_W, log: bool = False,
               value_fmt: str = "{:.3g}") -> str:
    """Labelled horizontal bars. `items` = (name, value, colour, note).

    `log=True` uses a logarithmic scale (indispensable in the NOx fan: the
    correlations separate by several orders of magnitude).
    """
    pts = [(n, _fin(v), c, t) for n, v, c, t in items]
    floor = 0.0 if log else -1e30
    pts = [p for p in pts if p[1] is not None and p[1] > floor]
    if not pts:
        return _empty("no correlation returned a finite value")
    ml, mr = 250, 84
    bh, gap = 22, 9
    h = len(pts) * (bh + gap) + 26
    span = max(w - ml - mr, 40)
    vals = [p[1] for p in pts]
    if log:
        lo = math.log10(max(min(vals), 1e-6))
        hi = math.log10(max(max(vals), 1e-5))
        if hi - lo < 0.5:
            lo, hi = lo - 0.5, hi + 0.5

        def frac(v):
            return (math.log10(max(v, 1e-6)) - lo) / max(hi - lo, 1e-9)
    else:
        hi = max(max(vals), 1e-12)

        def frac(v):
            return max(v, 0.0) / hi
    s = [_svg_open(w, h)]
    for i, (name, v, color, tip) in enumerate(pts):
        y = i * (bh + gap) + 6
        bw = max(frac(v) * span, 2.0)
        s.append(f'<text x="{ml - 8}" y="{y + bh - 7}" font-size="11.5" '
                 f'text-anchor="end" fill="{INK}">{_esc(name)}</text>')
        s.append(f'<rect x="{ml}" y="{y}" width="{bw:.1f}" height="{bh}" '
                 f'rx="4" fill="{color}" fill-opacity="0.82">'
                 f'<title>{_esc(name)}: {v:.4g} {_esc(unit)}'
                 + (f" — {_esc(tip)}" if tip else "") + "</title></rect>")
        s.append(f'<text x="{ml + bw + 6:.1f}" y="{y + bh - 7}" '
                 f'font-size="11.5" fill="{MUT}">'
                 f'{_esc(value_fmt.format(v))}</text>')
    scale_txt = ("logarithmic scale" if log else "linear scale")
    s.append(f'<text x="{ml}" y="{h - 6}" font-size="11" fill="{MUT}">'
             f'{_esc(unit)} · {scale_txt}</text>')
    s.append("</svg>")
    return "".join(s)


def _svg_sankey(splits: dict, mdot_air: float | None,
                w: int = SVG_W, h: int = 250) -> str:
    """Simple sankey of the air split: ṁ₃ → PZ / SZ / DZ / cooling.

    Hand-drawn cubic Bézier ribbons — the technique of Phy-AC's `_svg_*`
    brought to the air split of Ciencia §3.
    """
    order = (("pz", "Primary zone", HOT),
             ("sz", "Secondary zone", GOLD),
             ("dz", "Dilution", ACCENT),
             ("cool", "Cooling / effusion", COOL))
    vals = []
    for k, _lbl, _c in order:
        v = _fin((splits or {}).get(k))
        vals.append(max(v, 0.0) if v is not None else 0.0)
    tot = sum(vals)
    if tot <= 1e-12:
        return _empty("the record carries no `splits` block")
    mt, mb = 26, 26
    H = h - mt - mb
    gap = 10.0
    Hs = max(H - gap * (len(order) - 1), 20.0)
    xa, xb = 60.0, 82.0          # inlet bar
    xc, xd = 372.0, 394.0        # outlet bars
    s = [_svg_open(w, h)]
    s.append(f'<rect x="{xa}" y="{mt}" width="{xb - xa}" height="{H}" rx="3" '
             f'fill="{INK}" fill-opacity="0.82"><title>inlet air '
             f'ṁ₃</title></rect>')
    m_txt = ("" if _fin(mdot_air) is None
             else f" = {_fin(mdot_air):.4g} kg/s")
    s.append(f'<text x="{xa - 8}" y="{mt + H / 2:.0f}" font-size="12" '
             f'text-anchor="end" fill="{INK}">ṁ₃{_esc(m_txt)}</text>')
    yl, yr = float(mt), float(mt)
    for (k, lbl, color), v in zip(order, vals):
        fr = v / tot
        hl = fr * H
        hr = fr * Hs
        path = (f"M{xb},{yl:.1f} C{(xb + xc) / 2:.0f},{yl:.1f} "
                f"{(xb + xc) / 2:.0f},{yr:.1f} {xc},{yr:.1f} "
                f"L{xc},{yr + hr:.1f} C{(xb + xc) / 2:.0f},{yr + hr:.1f} "
                f"{(xb + xc) / 2:.0f},{yl + hl:.1f} {xb},{yl + hl:.1f} Z")
        s.append(f'<path d="{path}" fill="{color}" fill-opacity="0.42">'
                 f'<title>{_esc(lbl)}: {100 * fr:.1f} % of ṁ₃</title></path>')
        s.append(f'<rect x="{xc}" y="{yr:.1f}" width="{xd - xc}" '
                 f'height="{max(hr, 2):.1f}" rx="3" fill="{color}" '
                 f'fill-opacity="0.9"/>')
        m_row = ("" if _fin(mdot_air) is None
                 else f"  ·  {fr * _fin(mdot_air):.4g} kg/s")
        s.append(f'<text x="{xd + 8}" y="{yr + max(hr, 12) / 2 + 4:.1f}" '
                 f'font-size="12" fill="{INK}">{_esc(lbl)}  '
                 f'<tspan fill="{MUT}">{100 * fr:.1f} %'
                 f'{_esc(m_row)}</tspan></text>')
        yl += hl
        yr += hr + gap
    s.append("</svg>")
    return "".join(s)


def _svg_wall_profile(rec: dict, limit_K: float | None,
                      w: int = SVG_W, h: int = 300) -> str:
    """Axial wall-temperature profile T_w1 / T_w2 (Ciencia §5)."""
    prof = (rec or {}).get("T_wall_profile") or []
    pts = []
    for p in prof:
        try:
            z, t1, t2 = float(p[0]), float(p[1]), float(p[2])
        except Exception:
            continue
        if all(math.isfinite(v) for v in (z, t1, t2)):
            pts.append((z, t1, t2))
    if len(pts) < 2:
        return _empty("the record carries no `T_wall_profile` with ≥2 "
                      "stations")
    pts.sort(key=lambda q: q[0])
    zs = [p[0] for p in pts]
    temps = [p[1] for p in pts] + [p[2] for p in pts]
    t3 = _fin(rec.get("T3_K"))
    lim = _fin(limit_K)
    for extra in (t3, lim):
        if extra is not None:
            temps.append(extra)
    x0, x1 = _span(min(zs), max(zs), 0.02)
    y0, y1 = _span(min(temps), max(temps), 0.10)
    s, X, Y = _frame(w, h, x0, x1, y0, y1, "z [mm] (the gas flows to +Z)",
                     "Wall T [K]", "{:.0f}", "{:.0f}")
    parts = [_svg_open(w, h)] + s
    if lim is not None and y0 < lim < y1:
        parts.append(f'<line x1="{X(x0):.0f}" y1="{Y(lim):.1f}" '
                     f'x2="{X(x1):.0f}" y2="{Y(lim):.1f}" stroke="{BAD}" '
                     f'stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{X(x0) + 6:.0f}" y="{Y(lim) - 5:.1f}" '
                     f'font-size="11" fill="{BAD}">material limit '
                     f'{lim:.0f} K</text>')
    if t3 is not None and y0 < t3 < y1:
        parts.append(f'<line x1="{X(x0):.0f}" y1="{Y(t3):.1f}" '
                     f'x2="{X(x1):.0f}" y2="{Y(t3):.1f}" stroke="{MUT}" '
                     f'stroke-dasharray="3 4"/>')
        parts.append(f'<text x="{X(x1) - 4:.0f}" y="{Y(t3) - 5:.1f}" '
                     f'font-size="11" text-anchor="end" fill="{MUT}">'
                     f'T₃ = {t3:.0f} K</text>')
    for idx, (color, name) in ((1, (HOT, "T_w1 (flame side)")),
                               (2, (COOL, "T_w2 (annulus side)"))):
        poly = " ".join(f"{X(p[0]):.1f},{Y(p[idx]):.1f}" for p in pts)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" '
                     f'stroke-width="2.4"/>')
        for p in pts:
            parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[idx]):.1f}" '
                         f'r="3" fill="{color}"><title>z = {p[0]:.1f} mm · '
                         f'{_esc(name)} = {p[idx]:.1f} K</title></circle>')
    lx = 96
    for i, (color, name) in enumerate(((HOT, "T_w1 — flame side"),
                                       (COOL, "T_w2 — annulus side"))):
        parts.append(f'<rect x="{lx + i * 190}" y="6" width="14" height="7" '
                     f'rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 18 + i * 190}" y="13" font-size="11.5" '
                     f'fill="{INK}">{_esc(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_phi_band(rec: dict, w: int = SVG_W, h: int = 132) -> str:
    """Stability band: φ_LBO · required φ_LBO · φ_PZ · global φ."""
    phi_lbo = _fin((rec or {}).get("phi_LBO"))
    phi_pz = _fin((rec or {}).get("phi_pz"))
    if phi_lbo is None or phi_pz is None:
        return _empty("`phi_LBO` or `phi_pz` missing from the record")
    margin = _fin(rec.get("LBO_margin"))
    phi_glob = _fin(rec.get("phi_global"))
    phi_sz = _fin(rec.get("phi_sz"))
    hi = max(phi_pz, phi_lbo, phi_glob or 0.0, 1.5) * 1.12
    ml, mr = 56, 26
    y = 54
    span = w - ml - mr

    def X(v):
        return ml + max(min(v / hi, 1.0), 0.0) * span

    s = [_svg_open(w, h)]
    s.append(f'<rect x="{ml}" y="{y}" width="{X(phi_lbo) - ml:.1f}" '
             f'height="20" fill="{BAD}" fill-opacity="0.16"/>')
    s.append(f'<rect x="{X(phi_lbo):.1f}" y="{y}" '
             f'width="{max(span + ml - X(phi_lbo), 0):.1f}" height="20" '
             f'fill="{OK}" fill-opacity="0.13"/>')
    s.append(f'<line x1="{ml}" y1="{y + 20}" x2="{ml + span}" y2="{y + 20}" '
             f'stroke="{GRID}"/>')
    for i in range(6):
        v = hi * i / 5.0
        s.append(f'<text x="{X(v):.1f}" y="{y + 36}" font-size="11" '
                 f'text-anchor="middle" fill="{MUT}">{v:.2f}</text>')
    mmin = _lbo_margin_min()
    marks = [(phi_lbo, BAD, "φ_LBO (lean blowout)"),
             (phi_lbo * (1.0 + mmin), WARN,
              f"required φ_LBO (+{100 * mmin:.0f} % margin)"),
             (phi_pz, HOT, "φ_PZ (primary zone)")]
    if phi_sz is not None:
        marks.append((phi_sz, GOLD, "φ_SZ (secondary zone)"))
    if phi_glob is not None:
        marks.append((phi_glob, ACCENT, "global φ"))
    for i, (v, color, name) in enumerate(marks):
        x = X(v)
        dy = 0 if i % 2 == 0 else -16
        s.append(f'<line x1="{x:.1f}" y1="{y - 6}" x2="{x:.1f}" '
                 f'y2="{y + 26}" stroke="{color}" stroke-width="2.4">'
                 f'<title>{_esc(name)} = {v:.3f}</title></line>')
        s.append(f'<text x="{x:.1f}" y="{y - 12 + dy}" font-size="10.5" '
                 f'text-anchor="middle" fill="{color}">{v:.2f}</text>')
    lbl = "LBO margin = " + (f"{margin:+.3f}" if margin is not None
                             else DASH)
    s.append(f'<text x="{ml}" y="{h - 6}" font-size="11.5" fill="{INK}">'
             f'φ [-] · {_esc(lbl)} · red zone = blowout, green zone = '
             f'sustained flame</text>')
    s.append("</svg>")
    return "".join(s)


def _svg_pareto(front: list, spec, w: int = SVG_W, h: int = 380) -> str:
    """Verified Pareto front: EI_NOx (min) vs η_comb (max), with hover."""
    pts = []
    for p in front or []:
        if not isinstance(p, dict):
            continue
        x = _fin(p.get("EI_NOx"))
        y = _fin(p.get("eta_comb"))
        if x is None or y is None:
            continue
        pts.append((x, y, bool(p.get("feasible")), p))
    if not pts:
        return _empty("the result carries no `pareto_front` with valid "
                      "points")
    x0, x1 = _span(min(p[0] for p in pts), max(p[0] for p in pts), 0.10)
    y0, y1 = _span(min(p[1] for p in pts), max(p[1] for p in pts), 0.10)
    s, X, Y = _frame(w, h, x0, x1, y0, y1,
                     "EI_NOx [g/kg of fuel] — minimise",
                     "η_comb [-] — maximise", "{:.2f}", "{:.4f}")
    out = [_svg_open(w, h)] + s
    for x, y, feas, p in sorted(pts):
        color = OK if feas else BAD
        tip = (f"η_comb={y:.4f} · EI_NOx={x:.3f} g/kg · "
               f"Δp/p={_num(p.get('dp_over_p'), '{:.4f}')} · "
               f"PF={_num(p.get('PF'), '{:.3f}')} · "
               f"T_wall={_num(p.get('T_wall_max_K'), '{:.0f}')} K · "
               f"{p.get('arch', '?')}/{p.get('injector', '?')} · "
               f"{'feasible' if feas else 'infeasible'}")
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5.5" '
                   f'fill="{color}" fill-opacity="0.85" stroke="white">'
                   f"<title>{_esc(tip)}</title></circle>")
    dp_max = _fin(getattr(spec, "dp_max", None))
    if dp_max is not None:
        out.append(f'<text x="{w - 22}" y="{30}" font-size="11" '
                   f'text-anchor="end" fill="{MUT}">Δp/p ≤ '
                   f'{100 * dp_max:.1f} % is a hard CONSTRAINT, not an '
                   f'axis</text>')
    out.append("</svg>")
    return "".join(out)


# ==========================================================================
# 6. Optional matplotlib figures (visualization.py) — extra, never required
# ==========================================================================
#: `visualization.py` figure → section it is embedded in.
_FIG_SECTION = {
    "air_split": 2, "liner_thermal": 4, "emissions_fan": 6,
    "pareto": 9, "liner_profile": 13,
}


def _b64_png(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def _collect_figures(figures, rec: dict, front: list, contract,
                     material_limit_K, outdir: str, log) -> dict:
    """Normalises `figures` to `{stem: b64}`. **Never raises.**

    Accepts `None` (tries to generate them with `visualization.py`), a list
    of paths, a dict `{name: path}` or `{name: {"b64": …}}`. If matplotlib
    is missing it returns `{}` and the report is still complete.
    """
    out: dict[str, str] = {}
    try:
        if figures is None:
            try:
                import visualization
            except Exception:
                return {}
            if not visualization.available():
                log("      · matplotlib not available: the report ships "
                    "with its native SVGs only.")
                return {}
            figures = visualization.generate_all(
                rec, os.path.join(outdir, "figures"), front=front,
                contract=contract, material_limit_K=material_limit_K)
        if isinstance(figures, dict):
            items = list(figures.items())
        else:
            items = [(os.path.splitext(os.path.basename(str(p)))[0], p)
                     for p in (figures or [])]
        for name, val in items:
            stem = os.path.splitext(os.path.basename(str(name)))[0]
            b64 = None
            if isinstance(val, dict):
                b64 = val.get("b64")
                if b64 is None and val.get("path"):
                    b64 = _b64_png(str(val["path"]))
            elif isinstance(val, (str, os.PathLike)):
                b64 = _b64_png(str(val))
            if b64:
                out[stem] = b64
    except Exception as exc:
        log(f"      ⚠ figures skipped: {type(exc).__name__}: {exc}")
        return out
    return out


def _fig_html(figs: dict, section: int) -> str:
    """Base64 `<img>` block of the figures that belong to this section."""
    blocks = []
    for stem, b64 in sorted(figs.items()):
        if _FIG_SECTION.get(stem, 2) != section:
            continue
        blocks.append(
            f'<div class="card" style="margin-top:14px">'
            f'<img class="fig" src="data:image/png;base64,{b64}" '
            f'alt="{_esc(stem)}"/>'
            f'<p class="mut" style="margin:8px 0 0">Figure '
            f'<code>{_esc(stem)}.png</code> — matplotlib (optional).</p>'
            f"</div>")
    return "".join(blocks)


# ==========================================================================
# 7. Versions and traceability
# ==========================================================================
def _version_of(mod_name: str) -> str:
    """Version of an optional dependency WITHOUT importing it if it is not
    necessary."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return str(version(mod_name))
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        return str(getattr(mod, "__version__", "present"))
    except Exception:
        return "absent"


def _dependency_rows() -> str:
    rows = [_row("Python", _esc(sys.version.split()[0]), "meta"),
            _row("NumPy", _esc(getattr(np, "__version__", "?")), "meta")]
    try:
        import physics_core as _pc
        cantera_ok = bool(_pc.cantera_available())
    except Exception:
        cantera_ok = False
    rows.append(_row(
        "Cantera", f'{_esc(_version_of("cantera"))} '
        f'({"in use" if cantera_ok else "not available → layer-0 table"})',
        "meta"))
    try:
        import visualization
        mpl_ok = bool(visualization.available())
    except Exception:
        mpl_ok = False
    mpl_txt = "embedded figures" if mpl_ok else "native SVGs only"
    rows.append(_row("matplotlib",
                     f'{_esc(_version_of("matplotlib"))} ({mpl_txt})',
                     "meta"))
    for name, why in (("cadquery", "optional STEP export"),
                      ("jax", "L1.5 XLB backend"),
                      ("warp-lang", "native L2a backend")):
        rows.append(_row(name, f'{_esc(_version_of(name))} '
                               f'<span class="mut">({_esc(why)})</span>',
                         "meta"))
    return rows


def _manifest_status() -> str:
    """State of the SHA-256 manifest of `data/` (layer 0). Never raises."""
    try:
        import importlib
        dp = importlib.import_module("data_pipeline")
        here = os.path.dirname(os.path.abspath(__file__))
        there = os.path.dirname(os.path.abspath(getattr(dp, "__file__", "")))
        if os.path.normcase(here) != os.path.normcase(there):
            return ('<span class="warn">data_pipeline resolved outside '
                    "Phy-CB</span>")
        problems = dp.verify_manifest()
    except Exception as exc:
        return (f'<span class="warn">not verifiable '
                f"({_esc(type(exc).__name__)})</span>")
    if not problems:
        return '<span class="ok">SHA-256 verified, no discrepancies</span>'
    return (f'<span class="bad">{len(problems)} discrepancy(ies)</span>'
            f'<div class="mut">{_esc("; ".join(str(p) for p in problems[:4]))}'
            "</div>")


# ==========================================================================
# 8. Sections
# ==========================================================================
def _sec1_spec(spec, rec: dict, theta, result: dict) -> str:
    """1 · Input specification, verdict and design vector."""
    g = list(rec.get("g") or [])
    feas = bool(rec.get("feasible"))
    n_viol = sum(1 for v in g if (_fin(v) or 0.0) > 0.0)
    verdict = _verdict_span(
        feas, "VERIFIED FEASIBLE DESIGN",
        f"BEST CANDIDATE — {n_viol} violated constraint(s)")

    def sp(attr, fmt="{:.4g}", scale=1.0, unit=""):
        return _num(getattr(spec, attr, None), fmt, scale, unit)

    if spec is not None:
        spec_rows = "".join([
            _row("P₃ (inlet total pressure)",
                 sp("P3_Pa", "{:,.0f}", unit="Pa"), "spec"),
            _row("T₃ (inlet total temperature)",
                 sp("T3_K", "{:.1f}", unit="K"), "spec"),
            _row("ṁ₃ (air mass flow)", sp("mdot_kgs", "{:.4g}", unit="kg/s"),
                 "spec"),
            _row("T₄ (exit temperature the turbine asks for)",
                 sp("T4_K", "{:.1f}", unit="K"), "spec"),
            _row("Fuel", _esc(getattr(spec, "fuel", DASH)), "spec"),
            _row("Maximum Δp/p the cycle allows",
                 sp("dp_max", "{:.2f}", 100.0, "%"), "spec",
                 "HARD constraint: in a combustor Δp is SPENT on mixing up "
                 "to the allowance, it is not minimised (WAVE4 §1.2)"),
            _row("Maximum PF allowed", sp("PF_max", "{:.3f}"), "spec"),
            _row("Imposed architecture",
                 _esc(getattr(spec, "arch", None)
                      or "free (design variable θ[0])"), "spec"),
            _row("Hot-wall material",
                 _esc(getattr(spec, "material", None)
                      or "no structural constraint"), "spec"),
            _row("Casing material",
                 _esc(getattr(spec, "material_casing", None) or DASH), "spec"),
            _row("Envelope Ø × L",
                 f'{sp("envelope_D_mm", "{:.1f}")} × '
                 f'{sp("envelope_L_mm", "{:.1f}")} mm', "spec"),
            _row("DfAM constraints active",
                 "yes" if getattr(spec, "dfam", False) else "no", "spec"),
        ])
    else:
        spec_rows = "".join([
            _krow(rec, "P3_Pa", "P₃ (from the operating point of θ)",
                  "{:,.0f}", unit="Pa"),
            _krow(rec, "T3_K", "T₃", "{:.1f}", unit="K"),
            _krow(rec, "mdot_kgs", "ṁ₃", "{:.4g}", unit="kg/s"),
            _krow(rec, "T4_K", "T₄", "{:.1f}", unit="K"),
            _row("Fuel", _txt(rec, "fuel"), "L0"),
        ])

    res_rows = "".join([
        _krow(rec, "eta_comb", "η_comb (combustion efficiency)", "{:.4f}"),
        _krow(rec, "dp_over_p", "Δp/p (total pressure loss)",
              "{:.3f}", 100.0, "%"),
        _krow(rec, "PF", "PF (pattern factor)", "{:.4f}"),
        _krow(rec, "T_wall_max_K", "Maximum wall T", "{:.1f}", unit="K"),
        _krow(rec, "EI_NOx", "EI_NOx (headline: Lefebvre)", "{:.3f}",
              unit="g/kg"),
        _krow(rec, "EI_CO", "EI_CO (Lefebvre ⚠️ not verified)", "{:.2f}",
              unit="g/kg"),
        _krow(rec, "T04_K", "Exit T₀₄", "{:.1f}", unit="K"),
        _krow(rec, "P04_Pa", "Exit P₀₄", "{:,.0f}", unit="Pa"),
        _krow(rec, "f_fuel_air", "FAR (fuel/air ratio)", "{:.5f}"),
        _krow(rec, "M_exit", "Exit Mach", "{:.4f}"),
        _row("Architecture", _txt(rec, "arch"), "L0"),
        _row("Injector", _txt(rec, "injector"), "L0"),
        _row("Optimiser scalar fitness",
             _num(result.get("fitness"), "{:.5f}"), "L0",
             "η_comb penalised by NOx; infeasible ⇒ −10·violation"),
        _row("Source of the calculation",
             f"<code>{_txt(rec, 'source')}</code>", "meta"),
    ])

    th = np.asarray(theta, dtype=float).ravel() if theta is not None \
        else np.array([])
    theta_rows = ""
    for i, name in enumerate(VAR_NAMES):
        val = _num(th[i], "{:.5g}") if i < th.shape[0] else DASH
        gate = name in ("arch", "injector", "fuel")
        op = name in ("P3_Pa", "T3_K", "mdot_kgs", "T4_K")
        tag = " (gate)" if gate else (" (operating point)" if op else "")
        theta_rows += (f"<tr><td>{i}</td><td><code>{_esc(name)}</code>"
                       f'<span class="mut">{_esc(tag)}</span></td>'
                       f'<td class="v">{val}</td></tr>')

    theta_head = "<tr><th>i</th><th>variable</th><th>value</th></tr>"
    return (f"<p>{verdict}</p>"
            f'<div class="grid"><div><h3>Input specification</h3>'
            f"{_table(spec_rows)}</div>"
            f"<div><h3>Verified result</h3>{_table(res_rows)}"
            f'<p class="mut">These numbers come from a real PHYSICAL '
            f"evaluation (not from the surrogate): the model uncertainty "
            f"applies to the search, not to this point.</p></div></div>"
            f"<h3>Design vector θ</h3>{_table(theta_rows, theta_head)}")


def _sec2_sizing(rec: dict, figs: dict) -> str:
    """2 · Sizing and air split."""
    splits = rec.get("splits") or {}
    sankey = _svg_sankey(splits, _fin(rec.get("mdot_air_kgs")))
    size_rows = "".join([
        _krow(rec, "A_ref_m2", "A_ref (reference area)", "{:.6g}",
              1e4, "cm²"),
        _krow(rec, "A_ref_aero_m2", "A_ref by the aerodynamic criterion",
              "{:.6g}", 1e4, "cm²"),
        _krow(rec, "A_ref_chem_m2", "A_ref by the chemical criterion (θ)",
              "{:.6g}", 1e4, "cm²"),
        _row("Criterion that sizes it", _txt(rec, "sizing_driver"), "L0",
             "A_ref = max(aero, chemical) — Ciencia §2.1–2.2"),
        _row("Sizing loop converged",
             '<span class="ok">yes</span>' if rec.get("sizing_converged")
             else '<span class="warn">no (bounded loop exhausted)</span>',
             "L0"),
        _krow(rec, "D_ref_mm", "Reference Ø", "{:.2f}", unit="mm"),
        _krow(rec, "D_L_mm", "Liner Ø D_L", "{:.2f}", unit="mm"),
        _krow(rec, "D_casing_mm", "Casing Ø", "{:.2f}", unit="mm"),
        _krow(rec, "L_pz_mm", "Primary-zone length", "{:.2f}", unit="mm"),
        _krow(rec, "L_sz_mm", "Secondary-zone length", "{:.2f}", unit="mm"),
        _krow(rec, "L_dz_mm", "Dilution-zone length", "{:.2f}", unit="mm"),
        _krow(rec, "L_tot_mm", "Total length", "{:.2f}", unit="mm"),
        _krow(rec, "V_c_m3", "Combustion volume V_c", "{:.4f}", 1e3, "L"),
        _krow(rec, "U_ref_ms", "Reference velocity U_ref", "{:.2f}",
              unit="m/s"),
        _krow(rec, "q_ref_Pa", "Reference dynamic pressure q_ref",
              "{:.1f}", unit="Pa"),
        _krow(rec, "tau_ms", "Residence time τ", "{:.3f}", unit="ms"),
        _krow(rec, "I_MW_m3_atm", "Combustion intensity", "{:.1f}",
              unit="MW/m³·atm"),
        _krow(rec, "theta_lefebvre", "Lefebvre θ parameter", "{:.4g}"),
        _krow(rec, "Omega_load", "Combustor loading Ω", "{:.4g}"),
    ])
    thermo_rows = "".join([
        _krow(rec, "T_ad_st_K", "Stoichiometric adiabatic T", "{:.1f}",
              unit="K"),
        _krow(rec, "T_pz_K", "Primary-zone T", "{:.1f}", unit="K"),
        _krow(rec, "T_sz_K", "Secondary-zone T", "{:.1f}", unit="K"),
        _krow(rec, "T_dz_K", "Dilution-zone T", "{:.1f}", unit="K"),
        _krow(rec, "phi_pz", "Primary-zone φ", "{:.4f}"),
        _krow(rec, "phi_sz", "Secondary-zone φ", "{:.4f}"),
        _krow(rec, "phi_global", "Global φ", "{:.4f}"),
        _krow(rec, "far_stoich", "Stoichiometric FAR", "{:.5f}"),
        _krow(rec, "mdot_air_kgs", "Air ṁ", "{:.5g}", unit="kg/s"),
        _krow(rec, "mdot_fuel_kgs", "Fuel ṁ", "{:.5g}", unit="kg/s"),
        _krow(rec, "mdot4_kgs", "Exit ṁ", "{:.5g}", unit="kg/s"),
        _krow(rec, "LHV_J_kg", "Lower heating value", "{:.4g}",
              1e-6, "MJ/kg"),
        _row("Thermochemistry source", _txt(rec, "thermo_source"), "L0",
             "Cantera's `equilibrate('HP')` or the tabulated layer-0 fit"),
        _krow(rec, "eta_theta", "η by the θ parameter", "{:.4f}"),
        _krow(rec, "eta_evap", "η by evaporation", "{:.4f}"),
        _krow(rec, "eta_mix", "η by mixing", "{:.4f}"),
        _row("Dominant η_comb", _txt(rec, "eta_driver"), "L0",
             "η_comb = min(η_θ, η_evap, η_mix) — Ciencia §2.3"),
    ])
    split_rows = ""
    for key, lbl in (("pz", "Primary zone"), ("sz", "Secondary zone"),
                     ("dz", "Dilution"), ("cool", "Cooling/effusion")):
        split_rows += _row(lbl, _num(splits.get(key), "{:.2f}", 100.0, "%"),
                           "L0")
    tot = sum((_fin(splits.get(k)) or 0.0)
              for k in ("pz", "sz", "dz", "cool"))
    split_rows += _row("Σ (exact closure by construction)",
                       f"{100 * tot:.6f} %", "L0",
                       "frac_cool = 1 − (pz+sz+dz) is CLOSURE, not a "
                       "variable")
    return (f'<div class="card">{sankey}</div>'
            f"{_fig_html(figs, 2)}"
            f'<div class="grid"><div><h3>Sizing (Ciencia §2)</h3>'
            f"{_table(size_rows)}</div>"
            f"<div><h3>Thermochemistry and efficiency "
            f"(Ciencia §§2.3, 9.3)</h3>"
            f"{_table(thermo_rows)}</div></div>"
            f"<h3>Air split (Ciencia §3)</h3>{_table(split_rows)}")


def _sec3_holes(rec: dict) -> str:
    """3 · Holes, swirler and atomization."""
    rows = ""
    for r in rec.get("hole_rows") or []:
        if not isinstance(r, dict):
            continue
        rows += ("<tr>"
                 f"<td>{_txt(r, 'zone')}</td>"
                 f'<td class="v">{_get(r, "z_mm", "{:.1f}")}</td>'
                 f'<td class="v">{_get(r, "r_mm", "{:.1f}")}</td>'
                 f'<td class="v">{_get(r, "n", "{:.0f}")}</td>'
                 f'<td class="v">{_get(r, "d_mm", "{:.2f}")}</td>'
                 f'<td class="v">{_get(r, "Cd", "{:.3f}")}</td>'
                 f'<td class="v">{_get(r, "J", "{:.2f}")}</td>'
                 f'<td class="v">{_get(r, "Y_mm", "{:.2f}")}</td>'
                 f'<td class="v">{_get(r, "Y_over_DL", "{:.3f}")}</td>'
                 f'<td class="v">{_get(r, "A_row_m2", "{:.4g}", 1e6)}</td>'
                 "</tr>")
    head = ("<tr><th>Zone</th><th>z [mm]</th><th>r [mm]</th><th>n</th>"
            "<th>Ø [mm]</th><th>C_d</th><th>J</th><th>Y [mm]</th>"
            "<th>Y/D_L</th><th>A_row [mm²]</th></tr>")
    holes_tbl = _table(rows, head) if rows else _empty("there are no "
                                                      "`hole_rows`")

    sw = rec.get("swirler") or {}
    dp_rows = "".join([
        _krow(rec, "dp_Pa", "Total ΔP", "{:.1f}", unit="Pa"),
        _krow(rec, "dp_liner_Pa", "Liner ΔP", "{:.1f}", unit="Pa"),
        _krow(rec, "dp_sw_Pa", "Swirler ΔP", "{:.1f}", unit="Pa"),
        _krow(rec, "dp_liner_over_qref", "ΔP_liner / q_ref", "{:.2f}"),
        _krow(rec, "dp_sw_over_qref", "ΔP_swirler / q_ref", "{:.2f}"),
        _krow(rec, "A_h_total_m2", "Total hole area", "{:.4g}", 1e6, "mm²"),
        _krow(rec, "J_dz", "J of the dilution row", "{:.2f}"),
        _krow(rec, "Y_over_DL_dz", "Dilution Y/D_L", "{:.3f}", note=(
            "target 0.50 (single row) or 0.33 (two staggered rows) — "
            "Lefebvre & Norster 1968 ✅")),
        _row("Swirler · S_N", _num(sw.get("SN"), "{:.3f}"), "L0"),
        _row("Swirler · vane count", _num(sw.get("n_vanes"), "{:.0f}"),
             "L0"),
        _row("Swirler · vane β", _num(sw.get("beta_deg"), "{:.1f}",
                                      unit="deg"), "L0"),
    ])

    by_corr = rec.get("SMD_um_by_corr") or {}
    model = str(rec.get("atomizer_model") or "")
    items, srows = [], ""
    for key, name, mark, why in SMD_MODELS:
        if key not in by_corr:
            continue
        v = _fin(by_corr.get(key))
        chosen = (key == model)
        color = ACCENT if chosen else (COOL if mark == "✅" else GOLD)
        items.append((name + (" ← CHOSEN" if chosen else ""), v, color, why))
        srows += ("<tr>"
                  f"<td>{_esc(name)}"
                  + (' <span class="ok">(chosen model)</span>'
                     if chosen else "")
                  + f'<div class="mut">{_esc(why)}</div></td>'
                  f'<td class="v">{_esc(mark)}</td>'
                  f'<td class="v"><b>{_num(v, "{:.2f}")}</b>{_chip("L0")}'
                  "</td></tr>")
    fan = (_bar_chart(items, "SMD [µm]", value_fmt="{:.1f}")
           if items else _empty("gaseous branch: SMD does not apply"))
    atom_rows = "".join([
        _row("Atomization model used",
             f"<code>{_esc(model or DASH)}</code>", "L0"),
        _krow(rec, "SMD_um", "Headline SMD (D₃₂)", "{:.2f}", unit="µm"),
        _krow(rec, "ALR_eff", "Effective ALR (airblast only)", "{:.3f}"),
    ])
    smd_head = ("<tr><th>Correlation</th><th>State</th>"
                "<th>SMD [µm]</th></tr>")
    return (f"<h3>Hole rows (Ciencia §4.1–4.2)</h3>{holes_tbl}"
            f'<div class="grid"><div><h3>Losses and swirler '
            f"(Ciencia §4.4)</h3>{_table(dp_rows)}</div>"
            f"<div><h3>Atomization (Ciencia §7)</h3>{_table(atom_rows)}"
            f"</div></div>"
            f"<h3>SMD fan by correlation</h3>"
            f'<div class="card">{fan}</div>'
            f"{_table(srows, smd_head)}"
            + _note("EVERY correlation of the family is computed and "
                    "published: the reader sees the spread, not a single "
                    "number. The ⚠️ mark flags coefficients not yet checked "
                    "against Lefebvre & Ballal 3rd ed."))


def _sec4_thermal(rec: dict, structural: dict | None, figs: dict) -> str:
    """4 · 1-D liner thermal."""
    limit = _fin(rec.get("T_max_flame_wall_K"))
    if limit is None and structural:
        limit = _fin(structural.get("T_wall_allow_K"))
    svg = _svg_wall_profile(rec, limit)
    rows = "".join([
        _krow(rec, "T_w1_K", "T_w1 (flame side)", "{:.1f}", unit="K"),
        _krow(rec, "T_w2_K", "T_w2 (annulus side)", "{:.1f}", unit="K"),
        _krow(rec, "T_wall_max_K", "Maximum wall T of the profile", "{:.1f}",
              unit="K"),
        _row("Material limit (flame side)",
             _num(limit, "{:.1f}", unit="K"), "L0s",
             "from `structures_core.MATERIALS` — the same source of truth "
             "as constraint g[1]"),
        _krow(rec, "eta_film", "Film/effusion effectiveness η_film",
              "{:.4f}"),
        _krow(rec, "blowing_ratio_M", "Blowing ratio M", "{:.3f}"),
        _krow(rec, "n_eff_holes", "Effusion hole count", "{:.0f}"),
        _krow(rec, "effusion_porosity", "Effusion porosity", "{:.3f}",
              100.0, "%"),
        _krow(rec, "q_wall_W_m2", "Heat flux to the wall", "{:.4g}",
              1e-3, "kW/m²"),
        _krow(rec, "eps_gas", "Gas emissivity", "{:.4f}"),
        _krow(rec, "thermal_residual_rel",
              "Residual of the balance R1+C1=K=R2+C2", "{:.2e}", note=(
                  "BOUNDED loop with a convergence test — there is no open "
                  "`while`")),
        _row("Liner material", _txt(rec, "material"), "L0s"),
        _krow(rec, "lpbf_t_min_mm", "Material's minimum LPBF thickness",
              "{:.2f}", unit="mm", fid="DfAM"),
        _krow(rec, "t_liner_mm", "Liner thickness from θ", "{:.2f}",
              unit="mm"),
    ])
    pf_rows = "".join([
        _krow(rec, "PF", "Pattern factor PF", "{:.4f}"),
        _krow(rec, "PF_star", "PF* (raw correlation)", "{:.4f}"),
        _krow(rec, "RTDF", "RTDF (radial profile)", "{:.4f}"),
        _krow(rec, "T4_max_K", "Maximum local T₄", "{:.1f}", unit="K"),
    ])
    return (f'<div class="card">{svg}</div>{_fig_html(figs, 4)}'
            f'<div class="grid"><div><h3>1-D wall balance '
            f"(Ciencia §5)</h3>{_table(rows)}</div>"
            f"<div><h3>Pattern factor (Ciencia §4.3)</h3>{_table(pf_rows)}"
            + _note("The L0 PF is a CORRELATION: its absolute value is not "
                    "to be trusted; it serves to compare designs. A "
                    "quantitative PF needs L2 (Ciencia §10).")
            + "</div></div>")


def _sec5_stability(rec: dict) -> str:
    """5 · Stability: LBO and ignition."""
    svg = _svg_phi_band(rec)
    rows = "".join([
        _krow(rec, "phi_LBO", "φ_LBO (lean blowout)", "{:.4f}"),
        _krow(rec, "phi_pz", "Operating φ_PZ", "{:.4f}"),
        _krow(rec, "LBO_margin", "LBO margin (φ_PZ/φ_LBO − 1)", "{:.4f}",
              note=f"g[2] requires ≥ {_lbo_margin_min():.2f} "
                   f"(LBO_MARGIN_MIN ⚠️ provisional)"),
        _row("Branch of the LBO correlation", _txt(rec, "lbo_branch"), "L0",
             "in gases the evaporation term does not contribute"),
        _krow(rec, "E_min_J", "Minimum ignition energy", "{:.4g}", unit="J"),
        _krow(rec, "d_q_mm", "Quenching distance d_q", "{:.3f}", unit="mm"),
        _krow(rec, "tau_pz_ms", "Primary-zone τ", "{:.3f}", unit="ms"),
        _krow(rec, "Da_pz", "Primary-zone Damköhler", "{:.4g}"),
        _krow(rec, "LBO_margin_Da", "LBO margin through Da (L1)", "{:.4f}"),
    ])
    return (f'<div class="card">{svg}</div>{_table(rows)}'
            + _note("φ_LBO ∝ P₃^−1.3 and ∝ SMD²: lowering the inlet "
                    "pressure or fattening the droplet brings blowout "
                    "closer. The global constant of the correlation is ⚠️ "
                    "provisional (Ciencia §6.1)."))


def _sec6_emissions(rec: dict, figs: dict) -> str:
    """6 · Emissions: the full FAN, with its verification mark."""
    rows, items = "", []
    for key, name, mark, why in EMISSION_MODELS:
        if key not in rec:
            continue
        v = _fin(rec.get(key))
        out_band = (v is not None and "NOx" in name
                    and v > EI_NOX_PLAUSIBLE_MAX)
        fid = "L1" if key.endswith("_crn") else "L0"
        flag = ('<span class="bad">OUTSIDE THE PHYSICAL RANGE</span>'
                if out_band else '<span class="ok">within band</span>'
                if v is not None else f'<span class="mut">{DASH}</span>')
        rows += ("<tr>"
                 f"<td>{_esc(name)}<div class='mut'>{_esc(why)}</div></td>"
                 f'<td class="v">{_esc(mark)}</td>'
                 f'<td class="v"><b>{_num(v, "{:.4g}")}</b>{_chip(fid)}</td>'
                 f'<td class="v">{flag}</td></tr>')
        color = BAD if out_band else (COOL if mark == "✅" else GOLD)
        items.append((name + (" ⚠ out of band" if out_band else ""),
                      v, color, why))
    head = ("<tr><th>Correlation</th><th>State</th><th>EI [g/kg]</th>"
            "<th>Plausibility</th></tr>")
    tbl = _table(rows, head) if rows else _empty("the record carries no "
                                                "emissions")
    fan = _bar_chart(items, "EI [g/kg of fuel]", log=True)
    vc = _fin(rec.get("V_c_m3"))
    vopt = _fin(rec.get("V_c_opt_m3"))
    opt_rows = "".join([
        _krow(rec, "V_c_m3", "Current V_c", "{:.4f}", 1e3, "L"),
        _krow(rec, "V_c_opt_m3", "Analytical NOx↔CO optimum V_c", "{:.4f}",
              1e3, "L", note=("the relative NOx:CO weight is NOT physics, "
                              "it is a declared environmental preference — "
                              "Ciencia §8.2")),
        _row("Direction of the trade-off",
             ("raising V_c would lower CO and raise NOx"
              if (vc is not None and vopt is not None and vopt > vc)
              else "lowering V_c would lower NOx and raise CO"
              if (vc is not None and vopt is not None) else DASH), "L0"),
    ])
    return (f"{tbl}<h3>Correlation fan</h3>"
            f'<div class="card">{fan}</div>{_fig_html(figs, 6)}'
            f"<h3>NOx ↔ CO trade-off (Ciencia §8.2)</h3>{_table(opt_rows)}"
            + _note(f"Plausibility band used: EI_NOx ≤ "
                    f"{EI_NOX_PLAUSIBLE_MAX:.0f} g/kg. Outside it the "
                    "correlation is working beyond its calibration range "
                    "(the Rizk & Mongia ones depend on exp(−Ea/T_st) and "
                    "blow up): they are published all the same, marked. "
                    "These correlations give TRENDS (±30–50 % at best), "
                    "never certifiable absolutes."))


def _sec7_constraints(rec: dict, spec, structural: dict | None,
                      dfam: dict | None) -> str:
    """7 · Global g vector (contract §§5.1–5.4) and feasibility verdict."""
    def block(title, names_desc, values, fid):
        rows = ""
        for i, item in enumerate(names_desc):
            name = item[0]
            desc = item[1]
            form = item[2] if len(item) > 2 else ""
            v = values[i] if i < len(values) else None
            rows += ("<tr>"
                     f"<td><code>{_esc(name)}</code></td>"
                     f"<td>{_esc(desc)}"
                     + (f'<div class="mut"><code>{_esc(form)}</code></div>'
                        if form else "")
                     + f"</td>"
                     f'<td class="v">{_g_cell(v)}{_chip(fid)}</td>'
                     f'<td class="v">{_g_verdict(v)}</td></tr>')
        return f"<h3>{_esc(title)}</h3>" + _table(
            rows, "<tr><th>g</th><th>Meaning</th><th>Value</th>"
                  "<th>Verdict</th></tr>")

    g_aero = list(rec.get("g") or [])
    aero_desc = [(n, G_AERO_DESC.get(n, (n, ""))[0],
                  G_AERO_DESC.get(n, (n, ""))[1]) for n in CONSTRAINT_NAMES]
    html_out = block(f"Aerodynamic and thermal — contract §5.1 "
                     f"({N_CONSTRAINTS})", aero_desc, g_aero, "L0")

    g_spec = []
    if spec is not None:
        dp_max = _fin(getattr(spec, "dp_max", None)) or 1e-9
        pf_max = _fin(getattr(spec, "PF_max", None)) or 1e-9
        env_d = _fin(getattr(spec, "envelope_D_mm", None))
        env_l = _fin(getattr(spec, "envelope_L_mm", None))
        dpv = _fin(rec.get("dp_over_p"))
        pfv = _fin(rec.get("PF"))
        dcv = _fin(rec.get("D_casing_mm"))
        lcv = _fin(rec.get("L_tot_mm"))
        g_spec = [
            (dpv / dp_max - 1.0) if dpv is not None else None,
            (pfv / pf_max - 1.0) if pfv is not None else None,
            (dcv / env_d - 1.0) if (env_d and dcv is not None) else -1.0,
            (lcv / env_l - 1.0) if (env_l and lcv is not None) else -1.0,
        ]
    html_out += block("From the user's specification — contract §5.4 (4)",
                      G_SPEC_DESC, g_spec, "spec") if g_spec else (
        "<h3>From the user's specification — contract §5.4 (4)</h3>"
        + _empty("no `spec` was passed: the 4 spec constraints cannot be "
                 "evaluated"))

    if structural:
        names = structural.get("g_struct_names") or [n for n, _ in
                                                     G_STRUCT_DESC]
        desc = dict(G_STRUCT_DESC)
        sd = [(n, desc.get(n, n)) for n in names]
        html_out += block("Structural-thermal — contract §5.2 (5)", sd,
                          structural.get("g_struct") or [], "L0s")
    else:
        html_out += ("<h3>Structural-thermal — contract §5.2 (5)</h3>"
                     + _empty("no `structural` was passed"))

    if dfam:
        html_out += block("LPBF manufacturability — contract §5.3 (5)",
                          G_DFAM_DESC, dfam.get("g_dfam") or [], "DfAM")
    else:
        html_out += ("<h3>LPBF manufacturability — contract §5.3 (5)</h3>"
                     + _empty("no `dfam` was passed (it needs the layer-5a "
                              "geometry contract)"))

    all_g = ([v for v in g_aero]
             + [v for v in g_spec]
             + list((structural or {}).get("g_struct") or [])
             + list((dfam or {}).get("g_dfam") or []))
    fin = [_fin(v) for v in all_g]
    fin = [v for v in fin if v is not None]
    viol = sum(max(v, 0.0) for v in fin)
    n_bad = sum(1 for v in fin if v > 0.0)
    summary = _table("".join([
        _row("Length of the g evaluated in this report", f"{len(fin)}",
             "meta",
             "aero 12 + spec 4 + structural 5 + DfAM 5 = 26 with everything "
             "active"),
        _row("Violated constraints",
             (f'<span class="bad">{n_bad}</span>' if n_bad
              else '<span class="ok">0</span>'), "meta"),
        _row("Total violation Σ max(g,0)", f"{viol:.5f}", "meta",
             "Deb's constrained dominance consumes this MAGNITUDE, not just "
             "the boolean"),
        _row("Verdict", _verdict_span(n_bad == 0, "FEASIBLE",
                                      "INFEASIBLE"), "meta"),
    ]))
    return html_out + "<h3>Summary</h3>" + summary


def _sec8_trace(rec: dict, result: dict, path: str, spec) -> str:
    """8 · Traceability: run_meta, source, seed, versions."""
    fid_int = rec.get("fidelity")
    fid_lbl = _FID_BY_INT.get(int(fid_int) if _fin(fid_int) is not None
                              else -1, DASH)
    run_meta = dict(result.get("run_meta") or {})
    run_meta.setdefault("schema", rec.get("schema"))
    run_meta.setdefault("seed", rec.get("seed", SEED))
    run_meta.setdefault("source", rec.get("source"))
    run_meta.setdefault("fidelity", fid_lbl)
    run_meta.setdefault("n_evals", len(result.get("dataset") or [])
                        or result.get("n_evals"))
    run_meta.setdefault("rounds", len(result.get("history") or []))
    run_meta.setdefault("report_path", os.path.abspath(path))
    run_meta.setdefault("generated_at", datetime.now().isoformat(
        timespec="seconds"))
    if spec is not None:
        run_meta.setdefault("spec", {
            k: v for k, v in vars(spec).items() if not k.startswith("_")
        } if hasattr(spec, "__dict__") else str(spec))

    rows = "".join([
        _row("Record schema", f"<code>{_txt(rec, 'schema')}</code>", "meta"),
        _row("`source` label", f"<code>{_txt(rec, 'source')}</code>", "meta",
             "the suffix in parentheses explains WHY it degraded "
             "(contract §3)"),
        _row("Fidelity of the point", _esc(fid_lbl), "meta"),
        _row("Seed", _num(rec.get("seed", SEED), "{:.0f}"), "meta",
             "two identical `evaluate` calls ⇒ bit-identical records"),
        _row("Degeneracy reason", _txt(rec, "degenerate_reason"), "meta"),
        _row("SHA-256 manifest of `data/` (layer 0)",
             _manifest_status(), "meta"),
        _row("Accumulated physical evaluations",
             _num(run_meta.get("n_evals"), "{:.0f}"), "meta"),
        _row("Optimiser rounds",
             _num(run_meta.get("rounds"), "{:.0f}"), "meta"),
        _row("Generated", _esc(run_meta.get("generated_at", "")), "meta"),
        _row("Phy-CB version", _esc(PHYCB_VERSION), "meta"),
    ])
    try:
        meta_json = json.dumps(run_meta, indent=1, ensure_ascii=False,
                               default=str)
    except Exception:
        meta_json = "{}"
    return (f'<div class="grid"><div><h3>Provenance of the result</h3>'
            f"{_table(rows)}</div>"
            f"<div><h3>Dependency versions</h3>"
            f"{_table(''.join(_dependency_rows()))}"
            + _note("Only NumPy is mandatory. Cantera, matplotlib, CadQuery, "
                    "XLB and Warp are optional: if they are missing, the "
                    "system degrades WITH A LABEL in `source` and never "
                    "raises.")
            + "</div></div>"
            f"<h3><code>run_meta</code> block</h3>"
            f"<pre>{_esc(meta_json)}</pre>")


def _sec9_pareto(result: dict, spec, figs: dict) -> str:
    """9 · Verified Pareto front (WAVE4 §4)."""
    front = result.get("pareto_front") or []
    if not front:
        return _empty("this `result` does not come from `design()` — there "
                      "is no Pareto front to show")
    svg = _svg_pareto(front, spec)
    n_feas = sum(1 for p in front if isinstance(p, dict)
                 and p.get("feasible"))
    rows = ""
    for i, p in enumerate(front[:PARETO_TABLE_MAX]):
        if not isinstance(p, dict):
            continue
        feas = bool(p.get("feasible"))
        rows += ("<tr>"
                 f"<td>{i}</td>"
                 f'<td class="v">{_get(p, "eta_comb", "{:.4f}")}</td>'
                 f'<td class="v">{_get(p, "EI_NOx", "{:.3f}")}</td>'
                 f'<td class="v">{_get(p, "dp_over_p", "{:.4f}")}</td>'
                 f'<td class="v">{_get(p, "PF", "{:.3f}")}</td>'
                 f'<td class="v">{_get(p, "T_wall_max_K", "{:.0f}")}</td>'
                 f"<td>{_txt(p, 'arch')} / {_txt(p, 'injector')}</td>"
                 f'<td class="v">'
                 + ('<span class="ok">yes</span>' if feas
                    else '<span class="bad">no</span>')
                 + "</td></tr>")
    head = ("<tr><th>#</th><th>η_comb</th><th>EI_NOx [g/kg]</th>"
            "<th>Δp/p</th><th>PF</th><th>T_wall [K]</th>"
            "<th>architecture / injector</th><th>feasible</th></tr>")
    extra = ("" if len(front) <= PARETO_TABLE_MAX else
             _note(f"(the first {PARETO_TABLE_MAX} of {len(front)} points "
                   f"are tabulated; the complete front is in the JSON "
                   f"checkpoint)"))
    plural = "verified point" if len(front) == 1 else "verified points"
    return (f'<p class="mut">{len(front)} {plural}, {n_feas} '
            f'feasible.</p><div class="card">{svg}</div>'
            f"{_fig_html(figs, 9)}{_table(rows, head)}{extra}"
            + _note("A front of real PHYSICAL evaluations (not surrogate "
                    "predictions). Δp/p is not an axis: it is a hard cycle "
                    "constraint — in a combustor Δp is spent on mixing."))


def _sec10_rounds(result: dict) -> str:
    """10 · Surrogate round history (WAVE4 §4)."""
    hist = result.get("history") or []
    if not hist:
        return _empty("no rounds recorded: the `result` does not come from "
                      "`design()`, or the surrogate quality gate was never "
                      "cleared and the rounds went on space-filling "
                      "sampling")
    rows = ""
    for h in hist:
        if not isinstance(h, dict):
            continue
        sur = h.get("surrogate") or {}
        r2 = sur.get("r2") or {}
        rows += ("<tr>"
                 f'<td class="v">{_get(h, "round", "{:.0f}")}</td>'
                 f'<td class="v">{_get(h, "eta_comb", "{:.4f}")}</td>'
                 f'<td class="v">{_get(h, "dp_over_p", "{:.4f}")}</td>'
                 f'<td class="v">{_get(h, "PF", "{:.3f}")}</td>'
                 f'<td class="v">{_get(h, "EI_NOx", "{:.3f}")}</td>'
                 f'<td class="v">{_get(h, "T_wall_max_K", "{:.0f}")}</td>'
                 f'<td class="v">{_get(h, "best_fitness", "{:.4f}")}</td>'
                 f'<td class="v">{_get(h, "n_evals", "{:.0f}")}</td>'
                 f'<td class="v">{_get(r2, "eta_comb", "{:.3f}")} / '
                 f'{_get(r2, "dp_over_p", "{:.3f}")}</td>'
                 f'<td class="v">{_get(sur, "coverage_2sigma", "{:.3f}")}'
                 "</td></tr>")
    head = ("<tr><th>Round</th><th>η_comb</th><th>Δp/p</th><th>PF</th>"
            "<th>EI_NOx</th><th>T_wall [K]</th><th>fitness</th>"
            "<th>cum. evals</th><th>R² η/Δp</th><th>±2σ coverage</th>"
            "</tr>")
    return (_table(rows, head)
            + _note("Quality gate of the family: R²(η_comb) ≥ 0.90, "
                    "R²(Δp/p) ≥ 0.85 and ±2σ coverage ∈ [0.80 · 1.00]. If a "
                    "round does not clear it, the budget goes on extra "
                    "space-filling sampling instead of NSGA-II — that is "
                    "deliberate."))


def _sec11_l0_l1(rec: dict, result: dict) -> str:
    """11 · L0 vs L1 comparison of the winners (WAVE4 §4)."""
    pairs = result.get("l1_pairs") or result.get("winners_l1") or []
    blocks = []
    if pairs:
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                continue
            r0 = pair.get("L0") or pair.get("l0") or {}
            r1 = pair.get("L1") or pair.get("l1") or {}
            label = pair.get("label", f"Winner #{i}")
            rows = ""
            for key, name, fmt in (("eta_comb", "η_comb", "{:.4f}"),
                                   ("dp_over_p", "Δp/p", "{:.4f}"),
                                   ("PF", "PF", "{:.3f}"),
                                   ("T_wall_max_K", "Wall T [K]", "{:.1f}"),
                                   ("EI_NOx", "EI_NOx [g/kg]", "{:.3f}"),
                                   ("EI_CO", "EI_CO [g/kg]", "{:.2f}")):
                a, b = _fin(r0.get(key)), _fin(r1.get(key))
                d = (f"{100 * (b - a) / abs(a):+.1f} %"
                     if (a not in (None, 0.0) and b is not None) else DASH)
                rows += (f"<tr><td>{_esc(name)}</td>"
                         f'<td class="v">{_num(a, fmt)}{_chip("L0")}</td>'
                         f'<td class="v">{_num(b, fmt)}{_chip("L1")}</td>'
                         f'<td class="v">{_esc(d)}</td></tr>')
            blocks.append(f"<h3>{_esc(label)}</h3>" + _table(
                rows, "<tr><th>Quantity</th><th>L0</th><th>L1</th>"
                      "<th>Relative Δ</th></tr>"))
    # Fallback: the L1 record itself carries BOTH columns (contract §4.3).
    inplace = (("eta_comb", "eta_comb_crn", "η_comb", "{:.4f}"),
               ("EI_NOx", "EI_NOx_crn", "EI_NOx [g/kg]", "{:.3f}"),
               ("EI_CO", "EI_CO_crn", "EI_CO [g/kg]", "{:.2f}"),
               ("LBO_margin", "LBO_margin_Da", "LBO margin", "{:.4f}"))
    rows = ""
    for k0, k1, name, fmt in inplace:
        if k1 not in rec:
            continue
        a, b = _fin(rec.get(k0)), _fin(rec.get(k1))
        d = (f"{100 * (b - a) / abs(a):+.1f} %"
             if (a not in (None, 0.0) and b is not None) else DASH)
        rows += (f"<tr><td>{_esc(name)}</td>"
                 f'<td class="v">{_num(a, fmt)}{_chip("L0")}</td>'
                 f'<td class="v">{_num(b, fmt)}{_chip("L1")}</td>'
                 f'<td class="v">{_esc(d)}</td></tr>')
    if rows:
        head = ("<tr><th>Quantity</th><th>L0</th><th>L1</th>"
                "<th>Relative Δ</th></tr>")
        blocks.append("<h3>Recommended point (same record, two "
                      "columns)</h3>" + _table(rows, head))
        blocks.append(_table("".join([
            _row("CRN topology", _txt(rec, "crn_topology"), "L1"),
            _krow(rec, "Da_pz", "Primary-zone Damköhler", "{:.4g}"),
            _row("Source of the point",
                 f"<code>{_txt(rec, 'source')}</code>", "meta"),
        ])))
    if not blocks:
        return _empty("no winner was verified at L1 (Cantera missing, "
                      "`crn_core` not landed yet, or the point was not "
                      "re-evaluated): `source` says exactly why")
    return "".join(blocks) + _note(
        "L0 explores cheaply and L1 verifies candidates — the exact pattern "
        "of the sibling tools. A large discrepancy does not invalidate the "
        "L0: it points at where to recalibrate (`register_hifi_pair` hook).")


def _sec12_integration(rec: dict, result: dict, spec) -> str:
    """12 · Integration with the upstream compressor (WAVE4 §§2–3)."""
    info = result.get("integration") or getattr(spec, "integration", None) \
        or {}
    bleed_frac = _fin(rec.get("bleed_frac_required"))
    mdot = _fin(rec.get("mdot_air_kgs"))
    rows = "".join([
        _row("Origin of the specification",
             _esc({"phyac": "Phy-AC (multistage axial compressor)",
                   "phycc": "Phy-CC (centrifugal compressor)"}.get(
                       info.get("kind"), "the engineer's manual spec")),
             "meta"),
        _row("Source run",
             f"<code>{_esc(info.get('path') or DASH)}</code>", "meta"),
        _row("Schema of the contract read",
             f"<code>{_esc(info.get('schema') or DASH)}</code>", "meta"),
        _row("Plane used", _esc(info.get("plane") or DASH), "meta",
             "in Phy-CC the STAGE plane rules, not the impeller one: mixing "
             "them is mistake #1 (VALIDACION §2b)"),
        _krow(rec, "P3_Pa", "P₃ entering the combustor", "{:,.0f}",
              unit="Pa"),
        _krow(rec, "T3_K", "T₃ entering the combustor", "{:.1f}", unit="K"),
        _krow(rec, "mdot_kgs", "ṁ₃", "{:.5g}", unit="kg/s"),
        _krow(rec, "P04_Pa", "P₀₄ returned to the turbine", "{:,.0f}",
              unit="Pa"),
        _krow(rec, "T04_K", "T₀₄ returned to the turbine", "{:.1f}",
              unit="K"),
    ])
    derived = info.get("derived") or []
    notes = info.get("notes") or []
    der_html = ("".join(f"<li>{_esc(x)}</li>" for x in derived)
                if derived else "")
    note_html = "".join(f"<li>{_esc(x)}</li>" for x in notes) if notes else ""

    bleed = info.get("bleed") or {}
    bleed_rows = "".join([
        _row("Bleed REQUIRED by the liner",
             (f"{100 * bleed_frac:.2f} % of ṁ₃" if bleed_frac is not None
              else DASH)
             + (f' <span class="mut">= {bleed_frac * mdot:.4g} kg/s</span>'
                if (bleed_frac is not None and mdot is not None) else ""),
             "L0",
             "it replaces the `cooling_bleed=\"not modelled\"` of the Phy-AC "
             "cycle (CONTEXTO §5.3)"),
        _row("Port AVAILABLE on the compressor",
             (f"stage {_esc(bleed.get('stage'))}, Ø "
              f"{_num(bleed.get('hole_d_mm'), '{:.1f}', unit='mm')}")
             if bleed else DASH, "meta"),
        _row("Hub / tip radius of the inlet annulus",
             f"{_num(info.get('inlet_r_hub_mm'), '{:.1f}')} / "
             f"{_num(info.get('inlet_r_tip_mm'), '{:.1f}')} mm", "meta"),
    ])
    warn = ""
    if derived:
        warn = (f'<div class="warnbox"><b>DERIVED numbers, not read</b>'
                f"<ul>{der_html}</ul></div>")
    notes_block = (f'<p class="mut">Notes from the link:</p>'
                   f'<ul class="mut">{note_html}</ul>') if note_html else ""
    return (f"{_table(rows)}{warn}{notes_block}"
            f"<h3>Cooling air (bleed)</h3>{_table(bleed_rows)}"
            + _note("Cycle seam: `cycle_seam.hot_cycle(..., "
                    "combustor=record)` replaces the filler constants "
                    "(η_b = 0.995 and Δp/p = 5 %) with the values resolved "
                    "here. With `combustor=None` the cycle is byte-identical "
                    "to the original — the family's gate rule."))


def _sec13_effusion_cross_layer(contract: dict) -> str:
    """The «effusion: CROSS-LAYER contract» block of section 13.

    Effusion is ONE quantity shared by the physics (which splits the air
    with it) and the geometry (which drills it). Until 2026-07-31 each layer
    computed its own — ×1.375 in area and ×1.118 in C_d on the anchor,
    ×1.536 in conductance — and the disagreement only surfaced as a polite
    note. `geometry_generator` now cross-checks it against
    `physics_core.effusion_wall_area_m2` and writes it to
    `contract["validation"]["effusion"]`; here it is PAINTED, and if it is
    violated it is painted in red. It never raises."""
    chk = ((contract.get("validation") or {}).get("effusion") or {}) \
        if isinstance(contract, dict) else {}
    eff = (contract.get("effusion") or {}) \
        if isinstance(contract, dict) else {}
    if not chk:
        return ("<h3>Effusion · cross-layer contract</h3>"
                + _empty("the contract carries no `validation.effusion` "
                         "(geometry from an earlier version?)"))
    ok = chk.get("ok")
    if ok is True:
        verdict = ('<span class="ok">consistent</span> — the physics and '
                   'the geometry describe the SAME effusion')
    elif ok is None:
        verdict = ('<span class="mut">not verified</span> — it could not be '
                   'cross-checked against `physics_core`')
    else:
        verdict = ('<span class="bad">CROSS-LAYER CONTRACT '
                   'VIOLATION</span> — the physics splits air with one '
                   'effusion and the hardware drills another')
    tol = _fin(chk.get("tol_rel"))
    rows = "".join([
        _row("Verdict", verdict, "geom"),
        _row("Walls that carry effusion (v1)",
             _esc(", ".join(chk.get("walls") or []) or DASH), "geom"),
        _row("Wetted area · REAL profile [mm²]",
             _num(eff.get("wall_area_mm2"), "{:.1f}"), "geom"),
        _row("Wetted area · NOMINAL from physics_core [mm²]",
             _num(chk.get("wall_area_nominal_mm2"), "{:.1f}"), "L0"),
        _row("Open area · contract [mm²]",
             _num(eff.get("open_area_mm2"), "{:.2f}"), "geom"),
        _row("Open area · record (the one that splits the air) [mm²]",
             _num(eff.get("open_area_record_mm2"), "{:.2f}"), "L0"),
        _row("Hole count · contract / record",
             f"{_num(eff.get('n_holes'), '{:.0f}')} / "
             f"{_num(eff.get('n_holes_record'), '{:.0f}')}", "geom"),
        _row("Effusion C_d (table × inclination × AM roughness)",
             _num(eff.get("Cd"), "{:.4f}"), "L0",
             "the SAME one the 1-D network and the LBM use"),
        _row("Relative deviation of the open area",
             _num(chk.get("open_area_rel_dev"), "{:+.3%}"), "geom",
             f"declared tolerance ±{100.0 * tol:.2f} %" if tol else ""),
    ])
    viol = chk.get("violations") or []
    viol_html = ("<ul>" + "".join(f'<li class="bad">{_esc(v)}</li>'
                                  for v in viol) + "</ul>") if viol else ""
    return ("<h3>Effusion · cross-layer contract</h3>"
            + _table(rows) + viol_html)


#: Backwards-compatible alias: the function used to carry a Spanish name.
_sec13_efusion_entre_capas = _sec13_effusion_cross_layer


def _sec13_dfam(dfam: dict | None, contract: dict | None) -> str:
    """13 · DfAM: checker, build orientation and drains."""
    if not dfam:
        return _empty("no `dfam` was passed: the checker needs the "
                      "`phycb-combustor-1` geometry contract of layer 5a")
    rep = dfam.get("report")
    rep = rep if isinstance(rep, dict) else {}
    params = rep.get("params")
    params = params if isinstance(params, dict) else {}
    rows = "".join([
        _row("DfAM verdict",
             _verdict_span(bool(dfam.get("feasible")), "PRINTABLE",
                           "NOT PRINTABLE AS IT STANDS"), "DfAM"),
        _row("Build orientation chosen",
             f"<code>{_esc(dfam.get('orientation', DASH))}</code>", "DfAM",
             "the orientation is a DEGREE OF FREEDOM: z+ and z− are both "
             "tried and the one with the lower total violation wins"),
        _row("Orientation requested",
             f"<code>{_esc(rep.get('orientation_requested', DASH))}</code>",
             "DfAM"),
        _row("Verification voxel",
             _num(params.get("voxel_mm"), "{:.2f}", unit="mm"), "DfAM"),
        _row("Thickness floor (max(0.6 mm, 2·voxel))",
             _num(params.get("feature_floor_mm"), "{:.2f}", unit="mm"),
             "DfAM"),
        _row("Minimum hole Ø",
             _num(params.get("d_hole_min_mm"), "{:.2f}", unit="mm"), "DfAM"),
        _row("Minimum overhang angle",
             _num(params.get("overhang_min_deg"), "{:.0f}", unit="deg"),
             "DfAM"),
        _row("Source", f"<code>{_esc(dfam.get('source', DASH))}</code>",
             "meta"),
    ])
    trials = rep.get("orientation_trials") or []
    trial_rows = ""
    for t in trials:
        if not isinstance(t, dict):
            continue
        gs = " · ".join(f"{_fin(v):+.3f}" if _fin(v) is not None else DASH
                        for v in (t.get("g_dfam") or []))
        trial_rows += (f"<tr><td><code>{_esc(t.get('orientation'))}</code>"
                       f'</td><td class="v">'
                       f'{_num(t.get("total_violation"), "{:.4f}")}</td>'
                       f'<td class="v">{_num(t.get("n_violations"), "{:.0f}")}'
                       f"</td><td><code>{_esc(gs)}</code></td></tr>")
    trials_tbl = _table(trial_rows,
                        "<tr><th>Orientation</th><th>Σ max(g,0)</th>"
                        "<th>violations</th><th>g_dfam</th></tr>") \
        if trial_rows else _empty("no orientation trials")

    shapes = dfam.get("hole_shapes")
    shapes = shapes if isinstance(shapes, dict) else {}
    tears = [str(k) for k, v in shapes.items() if str(v) == "teardrop"]
    ports = dfam.get("drain_ports") or []
    port_rows = ""
    for p in ports:
        if isinstance(p, dict):
            port_rows += (f"<tr><td>{_esc(p.get('id', p.get('zone', DASH)))}"
                          f'</td><td class="v">'
                          f'{_get(p, "z_mm", "{:.1f}")}</td>'
                          f'<td class="v">{_get(p, "r_mm", "{:.1f}")}</td>'
                          f'<td class="v">{_get(p, "d_mm", "{:.2f}")}</td>'
                          "</tr>")
        else:
            port_rows += f'<tr><td colspan="4">{_esc(p)}</td></tr>'
    ports_tbl = _table(port_rows, "<tr><th>Port</th><th>z [mm]</th>"
                                  "<th>r [mm]</th><th>Ø [mm]</th></tr>") \
        if port_rows else _empty("the checker suggested no drain ports")

    viol = dfam.get("violations") or []
    viol_html = ("<ul>" + "".join(f"<li>{_esc(v)}</li>" for v in viol)
                 + "</ul>") if viol else _note("No DfAM violations.")
    notes = (rep.get("notes") or []) + (rep.get("warnings") or [])
    notes_html = ('<ul class="mut">'
                  + "".join(f"<li>{_esc(n)}</li>" for n in notes)
                  + "</ul>") if notes else ""
    ctr = ""
    inter = ""
    if isinstance(contract, dict) and contract:
        ctr = _table("".join([
            _row("Contract schema",
                 f"<code>{_esc(contract.get('schema', DASH))}</code>", "geom"),
            _row("Architecture", _esc(contract.get("arch", DASH)), "geom"),
            _row("Sign conventions declared in the JSON",
                 _esc(contract.get("_conventions", DASH)), "geom"),
        ]))
        #  CROSS-LAYER CONTRACT (effusion): the geometry that gets printed
        #  and the physics' air split have to be talking about the SAME
        #  effusion. If they do not match it is a violation, and it is
        #  painted as one — never in silence (audit 2026-07-31).
        inter = _sec13_effusion_cross_layer(contract)
    ctr = ctr or _empty("no `contract` was passed")
    return (f"{_table(rows)}"
            f"<h3>Geometry contract</h3>{ctr}"
            f"{inter}"
            f"<h3>Orientation trial</h3>{trials_tbl}"
            f"<h3>Holes converted to teardrops</h3>"
            + (f'<p><code>{_esc(", ".join(tears))}</code></p>' if tears
               else _note("No hole needed a teardrop shape."))
            + f"<h3>Powder drain ports</h3>{ports_tbl}"
            f"<h3>Violations</h3>{viol_html}{notes_html}")


def _limits_block(rec: dict) -> str:
    """Final «DECLARED LIMITS» block.

    Ciencia §10 (what each level does NOT predict) + the provisional
    coefficients that reached a number of this report + the fixed L2 label.
    """
    fid_int = _fin(rec.get("fidelity"))
    level = _FID_BY_INT.get(int(fid_int) if fid_int is not None else 0, "L0")
    rows = ""
    for name, model, cost, good, bad in FIDELITY_LADDER:
        here = (name == level)
        rows += ("<tr" + (' style="background:#fff2d8"' if here else "")
                 + f"><td><b>{_esc(name)}</b>"
                 + (' <span class="ok">← this report</span>' if here else "")
                 + f"</td><td>{_esc(model)}</td><td>{_esc(cost)}</td>"
                 f"<td>{_esc(good)}</td>"
                 f'<td><b>{_esc(bad)}</b></td></tr>')
    ladder = _table(rows, "<tr><th>Level</th><th>Model</th><th>Cost</th>"
                          "<th>Predicts reliably</th><th>Does NOT predict"
                          "</th></tr>")
    l2_label = ""
    if level in ("L2a", "L2b"):
        l2_label = (f'<p><span class="verdict" style="background:{WARN}">'
                    f"{_esc(L2_HONESTY_LABEL)}</span></p>")

    prov = rec.get("limitations")
    if not isinstance(prov, list) or not prov:
        try:
            prov = provisional_report()
        except Exception:
            prov = []
    items = []
    for p in prov:
        if isinstance(p, dict):
            items.append((str(p.get("key", "?")), str(p.get("message", ""))))
        else:
            items.append((str(p), ""))
    head = items[:PROVISIONAL_INLINE_MAX]
    tail = items[PROVISIONAL_INLINE_MAX:]

    def li(pair):
        k, m = pair
        return (f"<li><code>{_esc(k)}</code>"
                + (f'<div class="mut">{_esc(m)}</div>' if m else "") + "</li>")

    prov_html = (f"<ul>{''.join(li(p) for p in head)}</ul>"
                 if head else _note("No provisional coefficient is in play "
                                    "in this calculation."))
    if tail:
        prov_html += (f"<details><summary>… and {len(tail)} more warning(s) "
                      f"(click to unfold)</summary>"
                      f"<ul>{''.join(li(p) for p in tail)}</ul></details>")

    return (f'<div class="limits" id="limits">'
            f"<h2>DECLARED LIMITS</h2>"
            f'<p class="mut">What this fidelity level does <b>NOT</b> '
            f"predict. Exact table of "
            f"<code>docs/Quasar_PhyCB_Science.md</code> §10.</p>"
            f"{ladder}{l2_label}"
            f"<h3>Provisional coefficients that reached a number of this "
            f"report ({len(items)})</h3>{prov_html}"
            + _note("Honesty rule of contract §7: every value marked "
                    "`provisional` that reaches a number of the report "
                    "lights a visible warning and appears in "
                    "docs/PENDIENTES_Lverificar.md. Closing them requires "
                    "checking against Lefebvre & Ballal, «Gas Turbine "
                    "Combustion», 3rd ed.")
            + _note("This report is NOT a certificate. The emission indices "
                    "are TRENDS; the pattern factor and the metal "
                    "temperature are comparative between designs; the LBO "
                    "is a correlation margin, not a guarantee of operation. "
                    "Before manufacturing: structural verification with "
                    "certified material specs and a rig test.")
            + "</div>")


# ==========================================================================
# 9. Template and assembly
# ==========================================================================
_CSS = Template("""
 *{box-sizing:border-box}
 body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;color:$ink;
   margin:0;background:$bg;line-height:1.45}
 .wrap{max-width:1000px;margin:0 auto;padding:30px 20px 64px}
 h1{font-size:26px;margin:.25em 0}
 h2{font-size:18px;margin-top:40px;border-bottom:2px solid $grid;
   padding-bottom:6px;scroll-margin-top:10px}
 h3{font-size:15px;margin:20px 0 6px}
 table{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;
   overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.07);margin:8px 0 6px}
 td,th{padding:7px 12px;border-bottom:1px solid #eef0f3;font-size:13.5px;
   text-align:left;vertical-align:top}
 th{background:#f2f4f8;font-weight:600;font-size:12.5px}
 td.v{text-align:right;white-space:nowrap}
 code{font-family:Consolas,'SF Mono',monospace;font-size:12.5px;
   background:#f2f4f8;padding:1px 4px;border-radius:4px}
 pre{background:#fff;padding:14px;border-radius:8px;overflow:auto;
   font-size:12.5px;box-shadow:0 1px 3px rgba(0,0,0,.07)}
 .mut{color:$mut;font-size:12.5px;margin:4px 0}
 .verdict{display:inline-block;padding:6px 14px;border-radius:8px;color:#fff;
   font-weight:600;font-size:13.5px}
 .fid{display:inline-block;margin-left:7px;padding:0 5px;border-radius:5px;
   font-size:10px;font-weight:700;vertical-align:middle;
   border:1px solid currentColor;background:#fff}
 .f-l0{color:$mut}.f-l0s{color:#7a5af5}.f-l1{color:$accent}
 .f-l15{color:#0e9488}.f-l2{color:#b45309}.f-spec{color:#4b5563}
 .f-dfam{color:#a21caf}
 .card{background:#fff;border-radius:10px;padding:14px;
   box-shadow:0 1px 3px rgba(0,0,0,.07);overflow-x:auto}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}
 @media(max-width:860px){.grid{grid-template-columns:1fr}}
 .ok{color:$ok;font-weight:600}.bad{color:$bad;font-weight:600}
 .warn{color:$warn;font-weight:600}
 .limits{background:#fffaf0;border:1px solid #f0d9ac;border-radius:10px;
   padding:18px 20px;margin-top:44px}
 .limits h2{border-bottom-color:#f0d9ac;margin-top:4px}
 .warnbox{background:#fff6e8;border-left:4px solid $warn;padding:10px 14px;
   border-radius:6px;margin:10px 0;font-size:13px}
 details{margin-top:10px}
 summary{cursor:pointer;font-size:12.5px;color:$mut}
 img.fig{width:100%;height:auto;border-radius:6px;display:block}
 .toc{font-size:12.5px;margin:14px 0 0}
 .toc a{color:$accent;text-decoration:none;margin:0 10px 4px 0;
   display:inline-block}
 .toc a:hover{text-decoration:underline}
 footer{margin-top:36px;font-size:12px;color:$mut}
""").safe_substitute(ink=INK, mut=MUT, accent=ACCENT, ok=OK, bad=BAD,
                     warn=WARN, grid=GRID, bg="#f7f8fa")

#: Titles of the 13 sections (the order IS the reading contract).
SECTION_TITLES = (
    "Specification and verdict",
    "Sizing and air split",
    "Holes and atomization",
    "Liner thermal",
    "Stability",
    "Emissions",
    "g constraints and feasibility",
    "Traceability",
    "Verified Pareto front",
    "Surrogate round history",
    "L0 vs L1 comparison of the winners",
    "Integration with the compressor",
    "DfAM — LPBF manufacturability",
)


def _section(n: int, body: str, subtitle: str = "") -> str:
    sub = f' <span class="mut">({_esc(subtitle)})</span>' if subtitle else ""
    return (f'<h2 id="s{n}">{n} · {_esc(SECTION_TITLES[n - 1])}{sub}</h2>'
            f"{body}")


def _toc() -> str:
    links = "".join(f'<a href="#s{i + 1}">{i + 1} · {_esc(t)}</a>'
                    for i, t in enumerate(SECTION_TITLES))
    return (f'<nav class="toc">{links}'
            f'<a href="#limits"><b>DECLARED LIMITS</b></a></nav>')


# ==========================================================================
# 10. Public API
# ==========================================================================
def generate_report(spec, result: dict, path: str,
                    structural: dict | None = None,
                    contract: dict | None = None,
                    dfam: dict | None = None,
                    figures=None, log=print) -> str:
    """Writes the self-contained HTML report and returns the written path.

    Parameters
    ----------
    spec :
        `neural_optimizer.DesignSpec` (or any object with the same
        attributes, or `None`). It is only read with `getattr` — never
        assumed.
    result : dict
        Output of `neural_optimizer.design()` / `evaluate_design()`:
        `{theta, record, fitness, history, pareto_front}`. Extra keys this
        report uses if present: `run_meta`, `integration` (the `info` of
        `cycle_seam.spec_from_run`), `l1_pairs` (`[{label, L0, L1}, …]`),
        `dataset`, `n_evals`.
    path : str
        Path of the `.html` to write (missing directories are created).
    structural : dict | None
        Output of `structures_core.evaluate_structural`.
    contract : dict | None
        `phycb-combustor-1` contract of layer 5a.
    dfam : dict | None
        Output of `dfam_checker.check`.
    figures :
        `None` (tries to generate them with `visualization.py`), a list of
        PNG paths, or a dict `{name: path}`. matplotlib is OPTIONAL.
    log : callable
        Warning channel (`print` by default).

    **Never raises**: any section that fails is emitted with the reason, and
    any missing key is painted with a dash `—`.
    """
    try:
        result = result if isinstance(result, dict) else {}
        rec = result.get("record")
        rec = rec if isinstance(rec, dict) else {}
        theta = result.get("theta")
        if theta is None:
            theta = [rec.get(n) for n in VAR_NAMES]
        front = result.get("pareto_front") or []
        outdir = os.path.dirname(os.path.abspath(path)) or "."
        try:
            os.makedirs(outdir, exist_ok=True)
        except Exception:
            pass

        limit_K = _fin(rec.get("T_max_flame_wall_K"))
        if limit_K is None and structural:
            limit_K = _fin(structural.get("T_wall_allow_K"))
        figs = _collect_figures(figures, rec, front, contract, limit_K,
                                outdir, log)

        builders = (
            (1, lambda: _sec1_spec(spec, rec, theta, result), ""),
            (2, lambda: _sec2_sizing(rec, figs), "Ciencia §§2–3"),
            (3, lambda: _sec3_holes(rec), "Ciencia §§4, 7"),
            (4, lambda: _sec4_thermal(rec, structural, figs),
             "Ciencia §§5, 4.3"),
            (5, lambda: _sec5_stability(rec), "Ciencia §6"),
            (6, lambda: _sec6_emissions(rec, figs), "Ciencia §8"),
            (7, lambda: _sec7_constraints(rec, spec, structural, dfam),
             "contract §§5.1–5.4"),
            (8, lambda: _sec8_trace(rec, result, path, spec), "contract §9"),
            (9, lambda: _sec9_pareto(result, spec, figs), "WAVE4 §4"),
            (10, lambda: _sec10_rounds(result), "WAVE4 §4"),
            (11, lambda: _sec11_l0_l1(rec, result), "WAVE4 §4"),
            (12, lambda: _sec12_integration(rec, result, spec),
             "WAVE4 §§2–3"),
            (13, lambda: _sec13_dfam(dfam, contract), "WAVE4 §4 · §5.3"),
        )
        body = []
        for n, fn, sub in builders:
            try:
                body.append(_section(n, fn(), sub))
            except Exception as exc:           # a section NEVER takes it down
                log(f"      ⚠ report section {n} degraded: "
                    f"{type(exc).__name__}: {exc}")
                body.append(_section(
                    n, _empty(f"internal error while composing it: "
                              f"{type(exc).__name__}"), sub))
        try:
            limits = _limits_block(rec)
        except Exception as exc:
            log(f"      ⚠ declared-limits block degraded: "
                f"{type(exc).__name__}")
            limits = ('<div class="limits" id="limits"><h2>DECLARED '
                      "LIMITS</h2>" + _empty("internal error") + "</div>")

        feas = bool(rec.get("feasible"))
        head_chip = _verdict_span(feas, "VERIFIED FEASIBLE DESIGN",
                                  "BEST CANDIDATE — review the constraints")
        doc = (
            '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,'
            'initial-scale=1">'
            "<title>Quasar Phy-CB — Combustion chamber design report"
            "</title>"
            f"<style>{_CSS}</style></head><body><div class=\"wrap\">"
            f'<p class="mut">Quasar Solutions Lab · Phy-CB v{PHYCB_VERSION} · '
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')} · seed "
            f"{int(_fin(rec.get('seed')) or SEED)}</p>"
            "<h1>Autonomous design report — printable Brayton combustion "
            "chamber</h1>"
            f"<p>{head_chip}</p>"
            f'<p class="mut">Every number carries its <b>source '
            "fidelity</b> glued to it: "
            f'{_chip("L0")} correlations · {_chip("L0s")} analytical '
            f'structural · {_chip("L1")} 1-D network + CRN · '
            f'{_chip("L1.5")} cold aero LBM · {_chip("L2a")} CFD · '
            f'{_chip("spec")} the engineer sets it · {_chip("DfAM")} '
            f"manufacturability. A dash «{DASH}» means <b>missing "
            f"data</b>, never zero.</p>"
            f"{_toc()}"
            + "".join(body)
            + limits
            + "<footer>Self-contained report: SVGs generated in pure "
              "Python, no JavaScript and no external dependencies. The PNG "
              "figures (if any) are embedded in base64. Regenerable with "
              "<code>report_generator.generate_report()</code>.</footer>"
            "</div></body></html>")

        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        return path
    except Exception as exc:                        # final safety net
        # The contract demands that this module NEVER raises: if everything
        # fails, a minimal HTML is written that at least says what happened.
        try:
            log(f"      ⚠ report degraded to the minimal one: "
                f"{type(exc).__name__}: {exc}")
            with open(path, "w", encoding="utf-8") as f:
                f.write('<!DOCTYPE html><html lang="en"><head>'
                        '<meta charset="utf-8"><title>Quasar Phy-CB</title>'
                        "</head><body><h1>Report could not be generated</h1>"
                        f"<p>{_esc(type(exc).__name__)}: {_esc(exc)}</p>"
                        "</body></html>")
        except Exception:
            pass
        return path


# ==========================================================================
# 11. Smoke test — T15: section tokens + traceability block
# ==========================================================================
def _smoke_front(theta_ref, n: int = 64) -> list[dict]:
    """A REAL and cheap Pareto front: random sampling of the box + a
    non-dominated filter (constrained dominance: feasible ≫ infeasible).

    It is not an optimiser — it is a deterministic sample (seed 71) of
    genuine L0 evaluations, so that the §9 SVG shows real data in the smoke
    test and not invented numbers. The architecture and the fuel are pinned
    the way `DesignSpec.fix_operating_point` would.
    """
    from physics_core import BOUNDS_HI, BOUNDS_LO, Fidelity, evaluate

    rng = np.random.default_rng(SEED)
    ref = np.array(theta_ref, dtype=float)
    pts = []
    for _i in range(n):
        t = ref.copy()
        u = rng.random(16)
        t[:16] = BOUNDS_LO[:16] + u * (BOUNDS_HI[:16] - BOUNDS_LO[:16])
        t[0], t[15] = ref[0], ref[15]    # arch and fuel are fixed by the spec
        r = evaluate(t, fidelity=Fidelity.L0, use_cache=False)
        pts.append(dict(theta=t.tolist(), eta_comb=r["eta_comb"],
                        dp_over_p=r["dp_over_p"], PF=r["PF"],
                        EI_NOx=r["EI_NOx"], T_wall_max_K=r["T_wall_max_K"],
                        arch=r.get("arch"), injector=r.get("injector"),
                        feasible=bool(r.get("feasible")),
                        source=r.get("source", "")))
    front = []
    for p in pts:
        dominated = False
        for q in pts:
            if q is p:
                continue
            if q["feasible"] and not p["feasible"]:
                dominated = True
                break
            if q["feasible"] == p["feasible"] and \
                    q["eta_comb"] >= p["eta_comb"] and \
                    q["EI_NOx"] <= p["EI_NOx"] and \
                    (q["eta_comb"] > p["eta_comb"] or
                     q["EI_NOx"] < p["EI_NOx"]):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front or pts[:4]


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    from physics_core import THETA_REF, Fidelity, evaluate

    _out = os.path.join("runs", "_report_smoke", "report.html")
    _theta = THETA_REF
    _rec = evaluate(_theta, fidelity=Fidelity.L0)

    # --- structural (exists today) ---------------------------------------
    _struct = None
    try:
        import structures_core
        _struct = structures_core.evaluate_structural(
            _theta, _rec, material=_rec.get("material", "Inconel-625"))
    except Exception as _e:
        print(f"  · structural skipped: {type(_e).__name__}: {_e}")

    # --- geometry + DfAM (layer 5a; may not be there yet) ----------------
    _contract = _dfam = None
    try:
        import geometry_generator as _gg
        import dfam_checker as _dc
        _contract = _gg.build_contract(_theta, _rec, _struct, None)
        _dfam = _dc.check(_contract)
    except Exception as _e:
        print(f"  · DfAM skipped ({type(_e).__name__}) — §13 is emitted "
              f"anyway with the reason.")

    # --- spec (if the optimizer is available) ----------------------------
    _spec = None
    try:
        from neural_optimizer import DesignSpec
        _spec = DesignSpec(P3_Pa=250_000.0, T3_K=430.0, mdot_kgs=0.13,
                           T4_K=1150.0, fuel="jet_a", dp_max=0.04,
                           arch="reverse_annular", material="Inconel-625")
    except Exception as _e:
        print(f"  · spec skipped: {type(_e).__name__}: {_e}")

    _result = dict(
        theta=_theta, record=_rec, fitness=None, history=[],
        pareto_front=_smoke_front(_theta),
        run_meta=dict(checkpoint="runs/_report_smoke/phycb_run.json",
                      note="report_generator smoke test (T15)"),
        integration=dict(kind=None, path=None,
                         notes=["manual spec: this smoke does not come from "
                                "a compressor run"]))

    _path = generate_report(_spec, _result, _out, structural=_struct,
                            contract=_contract, dfam=_dfam, log=print)

    with open(_path, encoding="utf-8") as _f:
        _doc = _f.read()

    _found, _missing = [], []
    for _i, _t in enumerate(SECTION_TITLES, start=1):
        if f'id="s{_i}"' in _doc and _t in _doc:
            _found.append(f"{_i} · {_t}")
        else:
            _missing.append(f"{_i} · {_t}")
    _tokens = {
        "DECLARED LIMITS block": "DECLARED LIMITS" in _doc,
        "run_meta block": "run_meta" in _doc,
        "source label": "source" in _doc,
        "seed": f"seed {SEED}" in _doc,
        "lang=en": '<html lang="en">' in _doc,
        "zero JS": "<script" not in _doc.lower(),
        "native SVG": "<svg" in _doc,
        "fidelity chips": 'class="fid' in _doc,
    }

    print(f"\nreport: {os.path.abspath(_path)}")
    print(f"size  : {len(_doc.encode('utf-8')) / 1024:.1f} KiB "
          f"({len(_doc)} characters)")
    print(f"sections found ({len(_found)}/{len(SECTION_TITLES)}):")
    for _s2 in _found:
        print(f"    ✓ {_s2}")
    for _s2 in _missing:
        print(f"    ✗ {_s2}")
    print("traceability tokens:")
    for _k, _v in _tokens.items():
        print(f"    {'✓' if _v else '✗'} {_k}")
    print(f"provisional coefficients in play: "
          f"{len(_rec.get('limitations') or [])}")
    _ok = not _missing and all(_tokens.values())
    print(f"\nT15: {'✓ PASSES' if _ok else '✗ FAILS'}")
    sys.exit(0 if _ok else 1)
