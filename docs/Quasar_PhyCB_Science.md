# QUASAR Phy-CB — Science Document

Equations, correlations, algorithms, verification status and measured
results behind every layer of the Phy-CB stack: **3-D-printable Brayton
combustors** (micro gas turbines first, full scale range declared).
Companion to the root [README.md](../README.md) and to
[VALIDATION.md](VALIDATION.md); sibling of `Quasar_PhyCC_Science.md`
(centrifugal compressors), `Quasar_PhyAC_Science.md` (axial compressors and
fans) and `Phy_Wing_Science.md`.

> **The section numbering of §§1–14 is an interface, not a table of
> contents.** Around 250 references of the form *"Ciencia §4.1"* live
> inside the code, inside the `@source:` headers of the layer-0 CSV tables
> and inside the wave plans, so §§1–14 are preserved verbatim across this
> rewrite and new material is appended as §§15–19. The file itself was
> renamed `Quasar_PhyCB_Ciencia.md` → **`Quasar_PhyCB_Science.md`** on
> 2026-09-10; the 25 pointers that named the old file (the 17 layer-0
> tables, `physics_core`, `crn_core`, `data_pipeline`, `report_generator`,
> both READMEs and two wave plans) were updated with it, and
> `data/manifest.json` was regenerated because the table headers are
> hashed. The short prefix "Ciencia §" is kept in the code as the citation
> form.

**Verification-status convention** (the house's *calibrated honesty*, kept
per equation):

- ✅ **[verified]** — form *and* numerical coefficients checked against an
  accessible primary source (URL in §14).
- ⚠️ **[L·verify]** — the functional structure is right (exponents and
  dependencies, which is what fixes the software architecture) but the
  numerical coefficient still has to be checked against Lefebvre & Ballal,
  *Gas Turbine Combustion*, 3rd ed., or against the original paper. **Zero
  of these have been closed against the canonical book**, which is not on
  the target machine; that is why WAVE1 is declared *complete but not
  closed* (§19, limit 8).

**Notation.** Station 3 = compressor exit / combustor inlet; station 4 =
combustor exit / turbine inlet (standard gas-turbine numbering, identical
to the siblings' hot cycle). SI internally; `mm`, `deg`, `L` appear only in
interface keys, always with a unit suffix.

**Measurement machine.** Every number reported as *measured* in this
document was measured on the project machine: Windows 11, Python 3.13.7,
NumPy 2.4.6, Cantera 3.2.0 (conda-forge), .NET SDK 10, **NVIDIA RTX 3070
Ti 8 GiB (cc 8.6) with CuPy 14.1.1**. No Warp, no JAX, no XLB, no OpenFOAM
— and none of them get installed (house rule): what is absent degrades
with a label, it never raises.

**A note on what this document reports.** Several sections below record
results that are negative, self-refuting, or worse than an earlier
revision. That is deliberate. The headline NOx correlation carries the
wrong sign against φ_PZ (§8.6); the pattern factor returns its own
calibration instead of predicting (§4.3); the reference table that anchors
the whole validation campaign contradicts itself (§13.2); a prediction
this very document used to make about L1.5 was refuted by measurement
(§11.5). All of it is here, with the measurement that settles it, because
a refuted hypothesis with its number is worth more than a silent
workaround.

---

## 0. Design philosophy

Phy-CB inherits the family's Computational Engineering Model — *calibrated
physics prior → residual deep ensemble with an uncertainty gate →
constrained NSGA-II → self-contained report* — and specialises it for a
component where the physics is **correlational, not tabulated**: no public
aggregated dataset of micro-turbine combustors exists. Four consequences
run through every section.

**Every ⚠️ coefficient lives in a versioned table, never in the code.**
Layer 0 (`data_pipeline.py`) ships **17 CSV tables** with a SHA-256
manifest; each declares `@source:`, `@status: verified|provisional` and,
when provisional, `@status_reason:`. `physics_core` reads them through
`load_table()` and falls back to a documented SCREAMING_CASE constant when
a table is missing — registering the event in `PROVISIONAL_COEFFS` so that
**every provisional value that reaches a reported number lights a visible
warning** in the record and in `report.html`. Measured on the anchor case:
**43 such warnings** reach a published number.

**Δp/p is a cycle constraint, not an output.** In a micro turbine
(π ≈ 2–5) each extra point of pressure loss costs ≈ −0.5 % thrust or
+0.25 % specific fuel consumption. So Δp/p enters as a *hard bound* and the
hole areas are sized to meet it. The honest consequence is stated rather
than hidden: at L0 `record["dp_over_p"]` reproduces its target **by
construction**, and validating it requires CFD (§13.2, §19 limit 5).

**Manufacturability is a constraint inside the loop, not a post-check.**
`dfam_checker.py` returns 5 continuous LPBF constraints — minimum wall,
minimum hole diameter, ≥45° overhang or self-supporting teardrop, powder
**drainability** (≥2 line-of-sight exits per cavity), effusion
pitch/porosity band — that the optimizer sees at the same level as φ_PZ or
wall temperature. That is what lets the tool expose the real tension of an
AM combustor: denser effusion cools better *and* leaves the printable
band (§5.1).

**Degrade with a label; never raise.** `g(θ)` is continuous and never
throws (degenerate input ⇒ all twelve components at `G_DEGENERATE = 10.0`);
Cantera, matplotlib, CadQuery, XLB, JAX and Warp are optional and their
absence is written into `record["source"]`
(`crn_L1(cantera_unavailable→L0)`, `ibrans_L2A(unavailable→crn_L1)`, …).
The single deliberate exception is provenance: a mis-resolved sibling
import is an environment bug, not a degradable case, and
`assert_provenance()` stops execution (§17).

---

## 1. Structure of the preliminary design method (Lefebvre / Melconian–Modak)

### 1.1 The sequential procedure

Two competing sizing criteria and a closure loop over pressure loss and
pattern factor:

```
Cycle (P3, T3, ṁ3, T4, FAR, fuel)
   ↓
[1] Reference area A_ref ← max(aerodynamic criterion, chemical θ criterion)
[2] Reference geometry (D_ref, D_L, A_L/A_ref)
[3] Air split (PZ / SZ / DZ / cooling) ← targets φ_PZ, T_wall
[4] Hole sizing (C_d, jet penetration)
[5] Zone lengths (L_PZ, L_SZ, L_DZ) ← pattern factor
[6] Total pressure loss (cold + hot) → back to [1]
[7] Verification: η_comb, LBO, ignition, T_wall, emissions
```

It is evaluated at **every relevant operating point** (design, idle, cold
start; in aviation also windmilling relight at altitude) and the largest
A_ref wins. In ground micro turbines the critical point is usually
start/idle.

### 1.2 The design vector θ (20-D)

Frozen by contract. Indices 16–19 are the operating point and are never
moved by the optimizer; new variables are appended at index 20+ and
`pad_theta()` fills them with neutral defaults so that checkpoints,
regression anchors and surrogate datasets survive.

| i | name | range | note |
|---|---|---|---|
| 0 | `arch` | 0–2 | **gate**: can · annular · reverse_annular |
| 1 | `AL_over_Aref` | 0.55–0.80 | §2.5 |
| 2–4 | `frac_pz`, `frac_sz`, `frac_dz` | 0.15–0.60 / 0–0.25 / 0.15–0.60 | §3 |
| 5 | `swirl_SN` | 0.50–1.20 | §4.4 |
| 6–7 | `n_holes_pz`, `n_holes_dz` | 4–16 / 6–18 | relaxed integers |
| 8–9 | `d_eff_mm`, `pitch_over_d_eff` | 0.50–1.00 mm / 4–10 | effusion, §5.1 |
| 10 | `injector` | 0–3 | **gate**: airblast · simplex · vaporizer · micromix |
| 11 | `ALR` | 2–6 | airblast only |
| 12–13 | `L_pz_over_DL`, `L_dz_over_DL` | 0.50–1.00 / 1.00–3.00 | §2.5 |
| 14 | `t_liner_mm` | 0.50–1.50 mm | |
| 15 | `fuel` | 0–2 | **gate**: jet_a · propane · h2 |
| 16–19 | `P3_Pa`, `T3_K`, `mdot_kgs`, `T4_K` | 1e5–2e6 / 250–1150 / 0.02–30 / 800–1800 | operating point |

Two hard rules. **`frac_cool = 1 − (frac_pz + frac_sz + frac_dz)` is a
closure, never a variable** — Σ splits = 1 to machine precision. And the
three gates are decoded with `int(round(·))` + clip **before any float
operation**: a gate is an integer in disguise, and interpolating between
two architectures produces a hybrid with no physical meaning.

### 1.3 The record and the constraint vector

`evaluate(theta, fidelity) -> dict` returns the `phycb-record-1` record —
a pure function of θ (two identical calls give bit-identical records,
`SEED = 71`, no clocks inside the physics) — carrying the sizing, the
split, the hole table, atomization, the 1-D thermal profile, stability,
the emissions fan, the fidelity actually used (`source`), the active
`limitations` and `g`.

`g ≤ 0` ⇔ feasible, **12 components, frozen order**:

| i | name | rule |
|---|---|---|
| 0 | `PF` | PF ≤ PF_max |
| 1 | `T_wall` | T_w1 ≤ T_max,flame_wall(material) − 50 K |
| 2 | `LBO` | φ_LBO·(1 + margin_min) ≤ φ_PZ |
| 3–4 | `tau_min`, `tau_max` | 4 ms ≤ τ ≤ 10 ms |
| 5 | `M_exit` | M₄ ≤ 0.30 (radial turbine friendly) |
| 6 | `dp` | Δp/p ≤ 4 % |
| 7 | `t_liner` | t ≥ `lpbf_t_min_mm(material)` |
| 8 | `d_eff` | Ø ≥ 0.5 mm **and** blowing ratio inside the film map |
| 9 | `phi_pz_rich` | φ_PZ ≤ 1.40 |
| 10 | `eta_comb` | η ≥ 0.97 |
| 11 | `mixing` | \|Y/D_L − 0.5\| ≤ 0.15 |

The magnitude *is* the degree of violation and it is **continuous**,
because Deb's constrained dominance has to rank infeasible individuals
against each other, not merely know that they are infeasible.

`g[8]` carries a declared deviation from the original contract: besides
the printable-diameter rule it absorbs the continuous violation of the
**validity** range of the effusion film map (M ∈ [0.2, 1.0]). Outside the
tabulated range the film effectiveness is not modelled — the interpolation
is a clamp — and a design like that cannot be called feasible without
saying so. The narrower **design** band (M = 0.3–0.8) does *not* harden
`g`: it lights the soft constraint `g_soft["M_design"]`.

Two soft constraints (`≤ 0` ⇔ in band) are reported but do not enter `g`
nor `feasible`: `I_band` (combustion intensity outside its class band) and
`M_design`. In the full run the global constraint vector reaches **26
components**: 12 aero/thermal ⊕ 4 spec ⊕ 5 structural (§12) ⊕ 5 DfAM
(§16).

---

## 2. Reference-area sizing

### 2.1 Aerodynamic criterion ✅ [verified — standard form]

$$A_{ref}=\left[\frac{R}{2}\left(\frac{\dot m_3\sqrt{T_3}}{P_3}\right)^{2}\frac{(\Delta P_{3-4}/q_{ref})}{(\Delta P_{3-4}/P_3)}\right]^{1/2}$$

with $q_{ref}=\tfrac12\rho_3U_{ref}^2$ and
$\dfrac{q_{ref}}{P_3}=\dfrac{R}{2}\left(\dfrac{\dot m_3\sqrt{T_3}}{A_{ref}P_3}\right)^{2}$.

Pressure-loss factors (PLF = ΔP₃₋₄/q_ref):

| Architecture | PLF | ΔP₃₋₄/P₃ | U_ref (m/s) |
|---|---|---|---|
| Tubular (can) | 35–37 | 6–7 % | 25–35 |
| Tuboannular | 25–28 | 5–6 % | 25–40 |
| Annular | 18–20 | 5–6 % | 25–45 |
| **Micro turbine (Phy-CB target)** | 15–25 | **2–4 %** | **10–25** |

**Phy-CB's rule:** in micro turbines Δp/p is a **hard cycle constraint
(2–4 %)**, not an output. Typical share: ΔP_liner ≈ 70–80 % of the total
(the rest is diffuser + snout); the implementation takes the mid-band
`LINER_FRAC_OF_DP = 0.75` ⚠️.

**Measured consequence, and it is not a small one.** With the aerodynamic
criterion doing the sizing, ΔP_L/q_ref is an **algebraic identity**
independent of the operating point:

$$\frac{\Delta P_L}{q_{ref}}
= \text{LINER\_FRAC\_OF\_DP}\cdot\frac{\Delta p}{p}\cdot\frac{P_3}{q_{ref}}
= \text{LINER\_FRAC\_OF\_DP}\cdot \text{PLF} = 0.75\times20 = \mathbf{15}\ \text{always.}$$

The aerodynamic criterion drives the sizing in **all six** validation
cases (`sizing_driver = "aero"`), so this identity is not a corner case —
it is the operating regime, and it is what turns the pattern factor into a
tautology (§4.3).

A second declared deviation, found on 2026-08-01: the liner↔casing annulus
area was being taken as the shortcut `A_ref − A_L`, which is the
difference of two areas that live in **different planes** and matches no
solid. Measured on the anchor: `A_ref − A_L` = 1000.6 mm² (U_an = 64.1 m/s)
against the geometric annulus `π(r_cas² − r_lo²)` = 800.4 mm²
(U_an = 80.2 m/s) — ×1.25 in area, ×1.56 in cold-side dynamic head. It now
uses the shared constant `LINER_R_OVER_RCAS` that `geometry_generator`
already builds the annulus with. Same family of bug as §4.1.

### 2.2 Chemical criterion — Lefebvre's θ parameter ⚠️ [L·verify the required θ level]

$$\theta=\frac{P_3^{1.75}\,A_{ref}\,D_{ref}^{0.75}\,\exp(T_3/b)}{\dot m_3},\qquad b\simeq300\ \mathrm{K}$$

(SI: Pa, m², m, kg/s, K). η_comb grows monotonically with θ; the
literature places η ≈ 99.9 % at θ ≈ 7×10⁷ (SI) ⚠️. The chart η_c(θ) lives
as a versioned lookup table (`data/lefebvre/eta_vs_theta.csv`,
`@status: provisional` — **one anchoring point**), never as a hard-coded
fit.

Because D_ref depends on A_ref, the inversion is a **fixed point**: solved
with a *bounded* loop (`SIZING_ITERS = 32`, relative convergence test,
under-relaxation), never an open `while`. The map is contractive
(D_ref ∝ A^0.5 enters with exponent 0.75 ⇒ A^0.375).

**Declared deviation.** D_ref inside θ is taken as the equivalent diameter
√(4A_ref/π) *also* in the annular architectures, instead of the annulus
height. Reason: the single anchoring point of the η_c(θ) chart does not
declare an architecture, and mixing the "annulus height" convention with a
θ_req calibrated on tubular geometry doubles A_ref and throws V_c, τ and
casing diameter out of *every* class band. The annulus height is still
used for the casing.

The sizing target is `ETA_SIZING_TARGET = 0.98`, the centre of the micro
class band (0.970–0.990) — **not** 0.999: demanding large-aero efficiency
from a micro turbine doubles A_ref. Latent risk, quantified: if the real
chart required θ ×2, A_ref would grow **+58.8 %** and V_c and τ would
double.

### 2.3 Combustor loading (Odgers/Longwell) and the efficiency decomposition

$$\Omega=\frac{\dot m_3}{P_3^{1.8}\,V_c\,\exp(T_3/300)},\qquad \eta_{comb}=f(\Omega)\ \text{decreasing}$$

**Lefebvre's decomposition, which is how Phy-CB implements it:**

$$\eta_c=\min(\eta_{\theta,\text{kinetics}},\ \eta_{\text{evaporation}},\ \eta_{\text{mixing}})$$

- η_θ = f(θ) — kinetics, from the layer-0 chart.
- η_evap — D² law: a droplet of D₃₂ takes τ_e = D₃₂²/λ_eff to evaporate
  and the evaporated mass fraction at t is $1-(1-t/\tau_e)^{3/2}$,
  compared against the residence time **of the primary zone** (that is
  where evaporation has to be finished). A gaseous fuel goes to the
  ceiling.
- η_mix = $\eta_{ceil}\,(1-e^{-K\,\Delta P_L/q_{ref}})\,
  \bigl[1-K_p(\Delta Y/Y^\*)^2\bigr]$ ⚠️ **house form**: the science
  declares η_mix = f(ΔP_L/q_ref, jet geometry) without giving the
  function. `K_MIX_DP = 0.30` was chosen so that η_mix ≈ 0.99 at the
  classical design point ΔP_L/q_ref ≈ 15 and falls fast below 8; the
  quadratic penalty punishes dilution penetration away from the Lefebvre &
  Norster optimum.

At micro scale **evaporation and wall quenching dominate, not kinetics** —
which is exactly why this decomposition is what makes the model useful at
the target scale. The record publishes which mechanism won
(`eta_driver`); on the anchor it is `theta`, and η_mix is the driver in 4
of the 6 validation cases (gap G7).

### 2.4 Combustion intensity and residence time

$$I=\frac{\dot m_f\,LHV}{V_c\,P_3}\ \left[\mathrm{MW/m^3\,atm}\right];\qquad
\tau=\frac{V_c\,\rho_3}{\dot m_3};\quad \tau_{PZ}=\frac{V_{PZ}\,\rho}{\dot m_{PZ}}$$

Micro-turbine targets: **I = 20–60 MW/m³·atm**, **τ = 5–10 ms** (large
aero engines: 2–5 ms; the high micro value is imposed by evaporation and
CO burn-out).

**Audit, and it is a data finding.** The two expressions are not
independent: substituting V_c = τ·ṁ₃/ρ₃ gives the identity

$$I=\frac{FAR\cdot LHV\cdot P_{atm}}{R\,T_3\,\tau},$$

verified numerically to 0.000 % on the anchor. Therefore the I band
(20–60) and the τ band (5–10 ms) are only mutually compatible for
FAR ≲ 0.014 — and §13.2 shows that this FAR is incompatible with the
(T₃, T₄) bands of the very same reference table. Measured on the anchor:
I = **126.35 MW/m³·atm**, i.e. `g_soft["I_band"] = +1.11`; the value is
high because V_c is small, and it is reported as a *soft* violation rather
than silently rescaled.

### 2.5 Liner and zone lengths ⚠️ [L·verify the ratios]

- $A_L/A_{ref}\approx0.66$ (tubular/tuboannular), $\approx0.70$ (annular).
- $L_{PZ}\approx0.66\!-\!0.75\,D_L$ · $L_{SZ}\approx0.5\,D_L$ ·
  $L_{DZ}/D_L\approx1.5\!-\!2.5$ (fixed by PF) · $L_{tot}/D_L\approx2.5\!-\!4$
  (annular).

---

## 3. Air split and zone equivalence ratios

| Zone | Conventional aero | **Micro turbine** | RQL | Lean premixed (DLE) |
|---|---|---|---|---|
| Primary (swirler + PZ holes) | 18–25 % | **25–35 %** | 45–60 % | 60–75 % |
| Intermediate/secondary | 10–15 % | 10–20 % | — | — |
| Dilution | 25–45 % | **35–55 %** | 25–35 % (quench) | 15–25 % |
| Liner cooling | 20–40 % | **10–25 %** | 15–25 % | 10–20 % |

Equivalence ratios are **cumulative downstream**, exactly as the air is:

$$\phi_{PZ}=\frac{FAR_{global}}{f_{PZ}\,FAR_{st}},\quad
\phi_{SZ}=\frac{FAR_{global}}{(f_{PZ}+f_{SZ})\,FAR_{st}},\quad
\phi_{DZ}=\phi_{global}=\frac{FAR_{global}}{FAR_{st}}.$$

Targets: φ_PZ = 0.8–1.2 conventional (cited optimum ≈ 0.79 for maximum η
at low emissions); 1.2–1.8 in the rich zone of an RQL; 0.5–0.6 lean
premixed. φ_SZ ≈ 0.6–0.8. φ_global ≈ 0.15–0.30 (FAR 0.010–0.020).

**Measured calibration datum** (real micro turbojet, Wiley IJAE
2021:6616537): the published optimisation moved PZ air from 39.35 % → 32 %
(φ_PZ 0.89 → 1.09) to stabilise the flame during acceleration.

**This is the one chain in the whole package validated against hard data
from a real machine, and it passes without any per-case recalibration:**

| point | frac_pz | φ_PZ published | φ_PZ Phy-CB | Δ |
|---|---|---|---|---|
| baseline | 0.3935 | 0.890 | 0.9038 | **+1.55 %** |
| optimised | 0.3200 | 1.090 | 1.0922 | **+0.20 %** |
| invariant φ_PZ,opt/φ_PZ,base | — | 1.225 | 1.208 | **−1.32 %** |

The invariant is the strong check: it does not depend on the operating
point, so it measures the air-split → φ_PZ chain and nothing else. The
model also reproduces the *reason* the paper made the change.

---

## 4. Holes, jets and pattern factor

### 4.1 Mass flow and discharge coefficient ✅/⚠️

$$\dot m_h=C_d\,A_h\sqrt{2\rho_3\,\Delta P_L}$$

- Plain sharp-edged hole: C_d ≈ 0.60–0.65 · plunged/flared: 0.75–0.85 ·
  conservative first iteration: 0.5. C_d depends on bleed ratio and Re.
- Liner-loss closure:
  $\dfrac{\Delta P_L}{q_{ref}}\approx\left(\dfrac{(\dot m_L/\dot m_3)\,A_{ref}}{C_d\,A_{h,tot}}\right)^{2}$
- **AM roughness correction** (✅ discrete datum, no closed form
  published): in a small nozzle d_o = 1.2 mm, C_d rose from 0.37–0.41
  (fine finish) to 0.50–0.58 with high roughness — DOI
  10.1016/j.jmapro.2022.11.031. For liner air holes roughness typically
  **reduces** C_d by 5–20 % versus polished. Implemented as
  `C_d,AM = C_d,ideal · f(Ra/d)` with `f` tabulated in layer 0 and
  `RA_ASBUILT_UM = 12.0` ⚠️; the absence of a published closed correlation
  is declared, not papered over.
- Inclined effusion holes with crossflow on **both** faces get the
  multiplicative factor `CD_EFFUSION_INCLINED = 0.38` (film-cooling
  literature places the ratio at 0.35–0.60; the low end is taken because
  the internal crossflow of an AM liner annulus is strong, U_an ≈ 60–80
  m/s). ⚠️ **This is the single most influential tunable on the solved
  cooling split.** Anchor value of the resulting effusion C_d: 0.220988.

**The effusion single-source rule, and the bug that forced it.** Four
modules used to talk about the *same* effusion with **two different open
areas and two different C_d** (854.1 mm² / C_d 0.221 in the LBM against
621.4 mm² / C_d 0.247 in the 1-D network: a ×1.54 conductance discrepancy
at equal Δp). Root cause: `π·D_L` is the perimeter of an **equal-area
circle**, correct only for `can`; in an annulus the real outer perimeter
is `2π·r_lo`, larger by

$$\frac{r_{lo}}{\sqrt{r_{lo}^{2}-r_{li}^{2}}} = 1.374513 \quad\text{(measured on the anchor, } r_{lo}=37.470,\ r_{li}=25.707\text{ mm)}.$$

There is now exactly one owner (`physics_core`: `effusion_wall_area_m2`,
`effusion_pattern`, `effusion_discharge_coefficient`);
`geometry_generator` cross-checks its integrated area against the nominal
one and records any disagreement as a **cross-layer contract violation**;
and test **T49** (a–i) guards the whole chain across the three
architectures — including the identity above, checked as an identity and
not as a tolerance. The correction changed the anchor: 2197 → **3020**
effusion holes, blowing ratio M 1.21 → **0.884** (inside the map), and the
design moved from INFEASIBLE to feasible (§5.1).

### 4.2 Jet penetration ✅ [verified — the 1.15 constant confirmed in 2 sources]

$$\frac{Y_{max}}{d_j}=1.15\sqrt{J},\qquad J=\frac{\rho_jU_j^{2}}{\rho_gU_g^{2}}$$

Optimum mixing rule (Lefebvre & Norster 1968): single row → Y_max ≈ 0.5·D_L;
two staggered rows → Y_max ≈ 0.33·D_L. Dilution: 6–12 holes per row in
small chambers. Measured on the anchor: J_dz = 7.72, Y/D_L = 0.505 —
i.e. the dilution row lands on the single-row optimum, which is what
`g[11]` measures.

### 4.3 Pattern factor ⚠️ [L·verify the constant k]

**Both** definitions are exposed, because they coexist in the literature
and confusing them is a classic error:

$$\text{PF}=\frac{T_{4,max}-T_{4,avg}}{T_{4,avg}-T_3}\ \text{(design standard)};\qquad
\text{PF}^{*}=\frac{T_{4,peak}-T_{4,avg}}{T_{4,avg}}$$

Radial profile: $\text{RTDF}=\dfrac{T_{4,max,rad}-T_{4,avg}}{T_{4,avg}-T_3}$,
typically 0.12–0.30. Targets: PF 0.15–0.25 (cooled turbine), 0.25–0.35
(conventional), **0.30–0.45 acceptable in a micro turbine with an
uncooled radial turbine** (0.21–0.24 measured in real small combustors).

Dilution correlation (exponential form, k calibratable ⚠️):

$$\text{PF}=\exp\!\left[-k\,\left(\frac{L_{DZ}}{D_L}\right)\left(\frac{\Delta P_L}{q_{ref}}\right)\right]$$

**Measured finding: the pattern factor returns its own calibration.** PF =
0.35000 in all six validation cases, across three power classes and both
KJ66 points. The correlation *does* respond — monotonically, and both
monotonies are unit-tested:

| L_DZ/D_L | 1.0 | 1.4 | **1.8** | 2.2 | 3.0 |
|---|---|---|---|---|---|
| PF | 0.558 | 0.442 | **0.350** | 0.277 | 0.174 |

| ΔP_L/q_ref | 5 | 10 | **15** | 25 | 40 |
|---|---|---|---|---|---|
| PF | 0.705 | 0.497 | **0.350** | 0.174 | 0.061 |

— but with the aerodynamic criterion sizing A_ref, ΔP_L/q_ref = 15 is the
identity of §2.1, and 15 (together with L_DZ/D_L = 1.8) is **exactly the
point where `k_dilution = 0.0388823` was calibrated**. The anchor falls on
its own calibration point by construction; the trailing digits of
0.35000000857 are the rounding of k in the CSV.

Consequences, said out loud: PF has **no case-dependent degree of freedom
left** at L0, its only real lever is the dilution length, and PF is
therefore **not validated** (gap G5). Perturbing k measures how much of the
number is model: k ×2 → PF 0.1225 (−65 %); k ×0.5 → PF 0.592 (+69 %). What
actually fixes the pattern factor — number of dilution rows, their
staggering, their penetration — only enters at L1.5/L2.

`PF_MAX_DEFAULT = 0.45` is a declared deviation from the table's own
`PF_max_default = 0.35`: that 0.35 is *precisely* the calibration point, so
using it as the bound would leave every nominal design exactly on the
constraint boundary.

### 4.4 Swirler ✅ [standard form]

$$S_N=\frac{2}{3}\left[\frac{1-(D_{hub}/D_{sw})^{3}}{1-(D_{hub}/D_{sw})^{2}}\right]\tan\beta$$

S_N > 0.6 ⇒ central toroidal recirculation zone (CTRZ). Design: β = 45–60°
(implementation: 50°), S_N = 0.6–1.0. Loss ⚠️ [L·verify K_sw]:

$$\frac{\Delta P_{sw}}{q_{ref}}=K_{sw}\left[\left(\frac{A_{ref}}{A_{sw}\sec\beta}\right)^{2}-\left(\frac{A_{ref}}{A_L}\right)^{2}\right],\ K_{sw}=1.3\ \text{(flat vanes)},\ 1.15\ \text{(curved)}$$

`A_SW_OVER_AREF = 0.14` is a house closure chosen so that ΔP_sw/q_ref
falls inside the PLF band on the anchor ⚠️; `SWIRLER_FRAC_OF_PZ = 0.50` of
the primary air enters through the swirler (practice band 0.4–0.7).
Measured at L2a on the real combustor case: swirl number **0.769 against
the 0.70 imposed by the contract (+9.9 %, stable over three runs)** —
i.e. the swirler inlet profile of the CFD case reproduces the design
number to within 10 %.

---

## 5. 1-D liner thermal model ⚠️ [L·verify the 0.020 and 0.29 coefficients]

Steady balance per unit area: $R_1+C_1=K_{1\text{-}2}=R_2+C_2$

- Internal radiation: $R_1=0.5\,\sigma(1+\varepsilon_w)\,\varepsilon_g\,T_g^{1.5}(T_g^{2.5}-T_{w1}^{2.5})$
- Gas emissivity: $\varepsilon_g=1-\exp(-0.29\,P_3\,L\,(f\,l_b)^{0.5}\,T_g^{-1.5})$;
  luminosity l_b ≈ 1.7 kerosene, ≈ 1.0 natural gas/H₂
- Internal convection: $C_1=0.020\,\dfrac{k_g}{D_L^{0.2}}\left(\dfrac{\dot m_g}{A_L\mu_g}\right)^{0.8}(T_g-T_{w1})$
- External convection: $C_2=0.020\,\dfrac{k_a}{D_{an}^{0.2}}\left(\dfrac{\dot m_{an}}{A_{an}\mu_a}\right)^{0.8}(T_{w2}-T_3)$
- External radiation: $R_2=\sigma\,\varepsilon_w(T_{w2}^{4}-T_3^{4})$ · conduction:
  $K_{1\text{-}2}=\dfrac{k_w}{t_w}(T_{w1}-T_{w2})$

The structure (Dittus–Boelter, exponent 0.8) is solid; the coefficients are
⚠️. The system is solved by **bounded bisection** on T_w1 (q_in decreasing,
q_out increasing ⇒ a single root; 60 steps ⇒ |ΔT| < 1e-13 K), never by an
open iteration, at `N_WALL_STATIONS = 7` axial stations.

**The emissivity floor is the weakest number in the whole L0 stack, and it
is measured as such.** With P₃ in Pa the declared form gives ε_g ≈ 0.0065
on the anchor — two orders of magnitude below the 0.2–0.6 band the
literature reports for kerosene flames, i.e. it would leave the gas
transparent. A house floor `EPS_G_FLOOR_LUMINOUS = 0.15` *substitutes* for
the declared physics. Sweeping that floor over the literature band
0.05–0.60 moves the anchor wall temperature between **808 K and 1573 K**,
and `g[1]` **changes sign** (−0.33 ↔ +0.31). Doubling
`rad_internal_prefactor` alone moves T_w1 +23.0 %.

**Critical point of a recuperated micro turbine:** with T₃ up to 1093 K
(≈90 % recuperator, Bath/HiETA), C₂ collapses — the air no longer cools by
convection. That is *the* argument for dense AM-manufactured
effusion/transpiration, and it is why the 1-D thermal model must report the
margin `T_w,max − T_limit(material)` as a constraint g.

Measured anchor profile (z_mm, T_w1, T_w2), today, from
`python physics_core.py`:

```
[11.9, 994.6, 986.2] [35.6, 941.3, 933.9] [59.4, 796.1, 791.6]
[83.1, 758.5, 755.0] [106.9, 717.3, 714.6] [130.7, 692.5, 690.6]
[154.4, 687.2, 686.1]
```

**And the cleanest falsification of the campaign.** In validation case D1
(Bath/HiETA, T₃ = 1093 K, Inconel 625) the model declares that **IN625
does not survive**: T_w1 = 1466 K against a 1253 K limit. Bath/HiETA built
and ran that combustor in IN625. The result is robust across 18
combinations of air split and hole count (T_w1 moves only 1466–1540 K),
which is what makes it evidence rather than noise — and the only two
verified data of that case are precisely T₃ and the material, so the burden
of proof falls on the model (gap G12).

### 5.1 Film and effusion cooling

Slot (Lefebvre) ⚠️ [L·verify the exponents]:

$$\eta_{film}=1.10\,m^{0.65}\left(\frac{\mu_g}{\mu_c}\right)^{0.15}\left(\frac{x}{s}\right)^{-0.2}\left(\frac{t}{s}\right)^{-0.2}\quad(0.5<m<1.3)$$

Effusion — design parameters (✅ experimental data): d = 0.4–1.0 mm
(printable 0.5–0.8), angle 20–30° (30° beats 60° in adiabatic
effectiveness, confirmed), P_x/d = 5–8, P_y/d = 4–8, porosity 1–5 %, 10–20
rows (saturates around 10). Design blowing ratio
**M = ρ_cU_c/ρ_gU_g = 0.3–0.8** (at low M the coolant stays attached; at
high M there is local blow-off that improves downstream). **Conical**
holes (trivial in AM): +25 % effectiveness versus cylindrical at M = 0.25
(✅). Overall effectiveness targets: effusion 0.5–0.7;
impingement+effusion (AM double wall) 0.65–0.8; lattice transpiration
0.8–0.95.

**A closed-form effusion effectiveness correlation could not be verified**
— declared gap; implemented as a perforated-wall model plus a calibratable
map (`film_effusion.csv`), with the map's validity range consuming `g[8]`
so that a clamped (i.e. *unmodelled*) effectiveness can never be reported
as feasible.

**The measured design tension, which is the reason Phy-CB exists.**
Sweeping the effusion pitch on the anchor:

| pitch/d | porosity | M | η_film | g[8] | feasible |
|---|---|---|---|---|---|
| 4.0 | 4.91 % | 0.393 | 0.569 | −0.167 | yes |
| 5.0 | 3.14 % | 0.614 | 0.647 | −0.167 | yes |
| **6.0** | **2.18 %** | **0.884** | **0.686** | **−0.116** | **yes (anchor)** |
| 8.0 | 1.23 % | 1.571 | 0.711 | +0.571 | **no** |
| 10.0 | 0.79 % | 2.455 | 0.712 | +1.455 | **no** |

Drilling denser lowers M back into the design band — and collides with the
DfAM printable-effusion band (`g_dfam[4]`), which rejects
`pitch_over_d_eff = 4.5`. The anchor sits at M = 0.884: inside the map
(so `g[8] < 0`) but outside the 0.3–0.8 *design* band, so
`g_soft["M_design"] = +0.10` stays lit and is printed rather than
smoothed. **Fixing a violated g is a design change, not a knob** — that is
the lesson the tool is built to teach, and both ends of it are constraints
the optimizer sees.

---

## 6. Stability: LBO and ignition

### 6.1 Lean blow-out (LBO) ⚠️ [L·verify the global constant]

Lefebvre (spray flames) — functional structure:

$$\phi_{LBO}\propto\left(\frac{A_{ref}}{V_{PZ}}\right)\left(\frac{\dot m_A}{P_3^{1.3}\exp(T_3/300)}\right)\left(\frac{D_{32}^{2}}{\lambda_{eff}}\right)\frac{1}{LHV}$$

Design readings: P₃^−1.3 (stability degrades at low pressure — critical at
π ≈ 3); D₃₂² (atomization is *the* lever at small scale); a gaseous fuel
kills the evaporation term, so the margins are structurally different for
propane/H₂ (implemented as an equivalent constant
`C_gas = C·(D₃₂²/λ)_ref·LBO_GAS_EASE`, ⚠️ house calibration). Declared
accuracy: Lefebvre ±45 %; a Flame Volume model reaches ±15 % and is the
planned pluggable upgrade.

At L1 (CRN) the LBO margin comes from the **Damköhler number**: Da ≈
0.3–0.8 at blow-out against Da ≈ 20 at the design point (✅ CRN
literature, §9).

The record publishes **three** forms of the margin, because a "margin" of
0.998 reads wrong:

```
LBO_margin      = φ_PZ/φ_LBO − 1     (relative excess; 0 = exactly at LBO)
LBO_margin_ratio= φ_PZ/φ_LBO
LBO_margin_frac = 1 − φ_LBO/φ_PZ     (fraction of φ_PZ that can be lost)
```

Anchor, measured: φ_PZ = 1.0483, φ_LBO = 0.5245 ⇒ margin 0.9985, ratio
1.9985, frac 0.4996.

**Measured finding: with airblast the LBO numbers are not credible**
(gap G8). φ_LBO ∝ D₃₂², and the airblast SMD comes out ×3–4 small against
the class band:

| Case | SMD [µm] | φ_LBO | LBO margin | credible? |
|---|---|---|---|---|
| B1 (vaporizer, verified d₅₀ = 50 µm) | 50.0 | 0.525 | **1.00** | yes |
| A1 (vaporizer) | 50.0 | 0.507 | 1.13 | yes |
| A2 (airblast) | 10.7 | 0.025 | **42.5** | **no** |
| A3 (airblast) | 11.6 | 0.044 | 25.3 | **no** |
| D1 (airblast, T₃ = 1093 K) | 10.4 | 0.0017 | **475.8** | **no** |

A margin of ×476 says "this chamber never blows out": that is a symptom,
not a result. The vaporizer cases survive because their SMD is the
verified KJ66 d₅₀, not a correlation. Note that the conflict is between
two `verified` sources (the airblast correlations of §7.2 *are* verified),
so the suspect list includes the class band itself and the effective ALR —
which is why the recalibration order (§13.2) puts SMD ahead of `C_LBO_SI`.

### 6.2 Ignition (Ballal & Lefebvre) ✅ [structure verified]

$$E_{min}=\frac{\pi}{6}\rho_g c_p(T_{st}-T_0)d_q^{3},\qquad d_q\propto\frac{\alpha}{S_L};\qquad
E_{min}\propto D_{32}^{3}\ \text{(sprays)}$$

with $d_q=\max(Pe\,\alpha/S_L,\ C\cdot D_{32})$, `QUENCH_PECLET = 50`
(classical band 40–60) ⚠️ and ρ, c_p those of the **fresh mixture** the
spark has to heat. A factor 2 in SMD ⇒ a factor 8 in spark energy — which
is why micro turbojets use assisted ignition. Practical rule: 1–5 J plug
in the outer recirculation zone at 0.3–0.5·L_PZ from the dome, local φ
0.7–1.3 at start. Measured on the anchor: E_min = 4.10 mJ, d_q = 0.50 mm.

---

## 7. Atomization (SMD) — with per-correlation verification status

The record reports the **whole fan** of correlations by name
(`SMD_um_by_corr`), never a single number pretending to be the answer.
Measured on the anchor (vaporizer gate, so the headline is the verified
d₅₀): elshanawany_lefebvre 12.67 · rizkalla_lefebvre 16.43 ·
prefilm_nondimensional 78.12 · plain_jet 116.27 · chaussonnet_2016 71.11 ·
simplex_lefebvre 37.36 · **vaporizer_kj66_d50 50.0 µm ← headline**.

### 7.1 Pressure-swirl (simplex)

| Correlation | Form | Status |
|---|---|---|
| Lefebvre | $\text{SMD}=2.25\,\sigma^{0.25}\mu_L^{0.25}\dot m_L^{0.25}\Delta P_L^{-0.5}\rho_A^{-0.25}$ (SI) | ⚠️ **[L·verify]** coeff. 2.25 |
| Radcliffe | $\text{SMD}\propto\sigma^{0.6}\nu^{0.2}\dot m^{0.25}\Delta P^{-0.4}$ (coeff. 7.3, mixed units) | ⚠️ not verified; **over-predicts** (✅ qualitative) |
| Jasuja | $\text{SMD}\propto\sigma^{0.6}\nu^{0.16}\dot m^{0.22}\Delta P^{-0.43}$ (coeff. 4.4) | ⚠️ not verified; over-predicts at low flow (✅ qualitative) |

Flow number ✅: **FN = ṁ_L/√(ρ_L·ΔP_L) [m²]** (SI form verified, ILASS
2023); the industrial form ṁ_L/√ΔP_L coexists — check which one each
correlation uses before substituting. Small simplex C_d measured ✅:
0.37–0.41 (fine) → 0.50–0.58 (rough).

**Design decision**: SMD ∝ ΔP_L^−0.4…−0.5 ⇒ halving the SMD costs ×4 fuel
pressure. At micro scale (ṁ_f = 1–5 g/s) a simplex for SMD ~50 µm would
demand 20–60 bar ⇒ **the pure simplex is a bad choice below ~100 kW**.

### 7.2 Airblast — Phy-CB's default option

**Prefilming, dimensional form ✅ [VERIFIED against the table of a
peer-reviewed paper (arXiv:1906.04042, Table 7)] — use THIS one, not the
non-dimensional:**

Rizkalla & Lefebvre (1975):
$$\text{SMD}=3.33\times10^{-3}\,\frac{\sqrt{\sigma\rho_l L_c}}{\rho_g U_g}\left(1+\frac{\dot m_l}{\dot m_g}\right)
+13.0\times10^{-3}\left(\frac{\mu_l^{2}}{\sigma\rho_l}\right)^{0.425} L_c^{0.575}\left(1+\frac{\dot m_l}{\dot m_g}\right)^{2}$$

El-Shanawany & Lefebvre (1980) ✅:
$$\text{SMD}=\left[0.073\left(\frac{\sigma}{\rho_g U_g^{2}}\right)^{0.6}\left(\frac{\rho_l}{\rho_g}\right)^{0.1} D_p^{0.4}
+0.0015\sqrt{\frac{\mu_l^{2} D_p}{\sigma\rho_l}}\right]\left(1+\frac{\dot m_l}{\dot m_g}\right)$$

(SI: σ [N/m], μ_l [Pa·s], L_c/D_p [m] = characteristic length / prefilmer
Ø; the implementation takes `D_PREFILMER_OVER_DL = 0.25`, a house closure.)
The usual non-dimensional 0.33/0.068 form with (1+1/ALR) ⚠️ **could not be
verified**; it is computed and reported, but it is not the model.

Plain-jet (Rizk & Lefebvre 1984): two-term structure ✅ confirmed;
coefficients 0.48/0.15 ⚠️ not verified.

Verified extras ✅ useful at micro scale (same Table 7): Gepperth 2013
(SMD/δ = 12.08·We^−0.31·Re^−0.28·(ρ_l/ρ_g)^−0.037·(h_a/δ)^0.27) and
Chaussonnet 2016 (SMD = 1.69·(0.7/U_g)·√(h_aσ/(ρ_lρ_g))·(√ρ_l+√ρ_g)).
Exponent of ρ_g: classical −0.5…−1; newer data ≈ −0.3.

**Why airblast at micro scale:** SMD independent of fuel flow (turndown),
it uses the swirler's energy (3–8 bar pump), SMD ∝ U_A^−1.2 (60–100 m/s of
air is enough — the atomizing air is accelerated by the liner drop,
U_g = √(2ΔP_L/ρ₃), which lands in that band on the anchor), and a more
uniform spray ⇒ less soot ⇒ less radiation to the liner. **Design ALR:
2–6.**

### 7.3 Vaporizers and slinger

Vaporizer (KJ66 class, ✅ verified data): 6–12 "walking stick" tubes of
~64 mm with a ~30° bend; hole-area/tube-area ratio 0.25–0.5; optimum
in-tube residence 6–8 ms; d₅₀ ≤ 50 µm; tube wall T ≤ 900 °C. Advantage: no
high-pressure pump and no atomizer; drawbacks: auxiliary-gas start,
coking, slow transients. AM transforms it: internal fins, variable
section, monolithic joint to the dome. Slinger (rotary injection): SMD
independent of flow, no HP pump; requires an annulus around the shaft.

### 7.4 Micromix H₂ ✅ [verified data]

J = ρ_H₂u_H₂²/(ρ_airu_air²) with operating range 1.25–2.44; penetration
y = 1.15·d·√J; d_H₂ = 0.3–1.0 mm (0.5 mm typical; the original APU used
1600 holes of 0.3 mm); ~1–3 kW per injector ⇒ 30–100 injectors for 100 kW;
single-digit ppm NOx at φ ≤ 0.3, V_air ≥ 50 m/s, T_wall ≤ 600 K (≈9 ppm at
φ = 0.25). Short non-premixed flames ⇒ no flashback. **In practice only
manufacturable by AM.** Any non-liquid fuel forces the gas branch even if
the gate asks for a liquid atomizer, and `SMD_um` becomes `None` (not
`NaN`: the record must stay valid JSON).

---

## 8. Emissions — preliminary-level correlations

### 8.1 Lefebvre's EI_NOx ✅ [VERIFIED — U. Toronto thesis, eq. 1.4]

$$EI_{NOx}=9\times10^{-8}\,\frac{P_3^{1.25}\,V_c\,\exp(0.01\,T_{st})}{\dot m_A\,T_{PZ}}\ \ [\mathrm{g/kg\ fuel}]$$

V_c [m³], T_st stoichiometric temperature [K], T_PZ primary-zone
equilibrium temperature [K], ṁ_A [kg/s]. The unit of P₃ is not explicit in
the verified source (the 1984 original uses kPa); the dimensional audit of
§8.5 **confirms kPa**, and the scale lives in `emissions_coeffs.csv`
(`P3_unit_scale_from_Pa`), never in the code.

### 8.2 Lefebvre's EI_CO ⚠️ [L·verify — NOT verified]

$$EI_{CO}=\frac{86\,\dot m_A\,T_{PZ}\,\exp(-0.00345\,T_{PZ})}{(V_c-V_e)\,(\Delta P/P)^{0.5}\,P_3^{1.5}}$$

(V_e = volume occupied by evaporation.) Do not use 86 / −0.00345 without
checking the book, ch. 9.

**The NOx↔CO trade-off is captured explicitly**: NOx ∝ +V_c·exp(0.01T_st),
CO ∝ 1/V_c ⇒ an analytic optimum V_c exists,
∂(w₁·EI_NOx + w₂·EI_CO)/∂V_c = 0, implemented at L0 at near-zero cost with
`NOX_CO_WEIGHT_RATIO = 10` — declared as an *environmental preference*,
not physics: change it and the optimum V_c moves. Measured on the anchor:
optimum V_c = 0.659 L against the design 0.388 L.

### 8.3 Rizk & Mongia ✅ [VERIFIED — U. Toronto thesis, eqs. 3.5 and 1.5–1.8]

Global:
$$EI_{NOx}=15\times10^{14}\,(\tau-0.5\tau_e)^{0.5}\exp(-71100/T_{st})\,P_3^{-0.05}\,(\Delta P_3/P_3)^{-0.5}$$

τ, τ_e in ms; τ_e = SMD²/λ_e with
λ_e = (8k_g/(ρ_lc_{pg}))·ln(1+B_m)·(1+0.3Re^{0.5}Pr^{0.33}) (✅ also verified).

Zonal (structure ✅; ⚠️ the φ function and the exponent
`aa = 1 − 1.95·exp(−5.76·φ)` still need a visual check against pp. 9–10 of
the source PDF — text extraction degraded signs and parentheses):

```
PZ (φ ≤ 1.08):  EI_NOx = 1e13 · (P3^1.4/1e6)^aa · exp(−71442/T_c) · (7.56·φ^−7.2 − 1.6) · τ^0.64
PZ (φ > 1.08):  same with (5.21·φ^−2.99 − 1.6) · τ^0.64
SZ/DZ:          same with (1.172·φ^−4.56 − 1.6) · τ^0.876
```

### 8.4 Odgers & Kretschmer ✅ [verified against a peer-reviewed MDPI 2024 review]

$$EI_{NOx}=29\,\exp(-21670/T_f)\,P_3^{0.66}\,[1-\exp(-250\,\tau)]\ [\mathrm{g/kg}]$$

T_f [K], τ [s]. The saturating term [1−exp(−250τ)] flattens for τ > ~20 ms
⇒ especially appropriate for micro turbines (τ = 5–10 ms, the sensitive
regime). Optimised variant (same source ✅): Laviolette & Perez,
EI_NOx = 1.902×10⁵·P3^0.4048·[1−exp(−480τ)]/exp(37902/T_f). Empirical
contrast ✅: NEPP `EI_NOx = 33.2·(P3/432.7)^0.4·exp((T3−1487.6)/350)`
(psia, °R); GasTurb `EI_NOx = 32·(P3/2965)^0.4·exp((T3−826)/194)` (kPa, K).

### 8.5 The dimensional audit of the fan (measured, not argued)

The first anchor smoke exposed a fan spanning **six orders of magnitude**
(lefebvre 2.0 · rizk_mongia 1147 · zonal 313 · odgers 0.0044 · laviolette
0.030 g/kg) — the classical error this document anticipates: **units**.
Method: each correlation was re-derived sweeping the unit of P₃
(Pa · kPa · bar · atm · psia), of τ (s · ms) and of Δp/p (fraction ·
percent), evaluating the 4–24 combinations at **two extremes at once** —
the anchor (250 kPa, 430 K, τ = 6.05 ms, T_st = 2360 K) and a large engine
(30 atm, 800 K, τ = 3 ms, T_st = 2600 K, where the literature gives
EI_NOx = 20–40 g/kg). Only the combination that lands in range at **both**
extremes survives. A unit that only works in the micro turbine is not
evidence, it is fitting.

Result (all the scales live in `emissions_coeffs.csv`), re-measured today:

| correlation | P₃ | τ | Δp/p | anchor | large engine | verdict |
|---|---|---|---|---|---|---|
| lefebvre | kPa | – | – | 2.00 | 39.17 | ✅ confirmed |
| rizk_mongia | kPa* | s | % | 3.63 | 26.71 | ⚠️ τ and Δp/p corrected |
| zonal | Pa | s | – | 3.76 | 40.99 | ⚠️ P₃ and τ corrected |
| odgers | Pa | s | – | 8.91 | 23.53 | ⚠️ P₃ corrected (NOT atm) |
| laviolette | Pa | s | – | 3.17 | 4.26 | ⚠️ PROVISIONAL (×5 low) |

\* undecidable: exp_P3 = −0.05 makes Pa↔atm fit inside a factor 1.8.

Fan spread on the anchor: **×2.6e5 before the audit → ×4.46 after**
(×9.63 on the large engine). No coefficient was "fixed": only unit factors
moved, and every one of them is still marked `provisional` — dimensional
inference is not a reading of a primary source.

A **plausibility screen** (0.01–200 g/kg, with its provenance written into
the table) marks out-of-range members in `EI_NOx_flags`, excludes them from
the fan statistics and fires a warning — but keeps reporting them by name.
Measured on an extreme case (20 bar, 900 K, T₄ = 1800 K) the zonal form
goes `out_of_range_low` and is excluded; the user still sees it.

### 8.6 The headline NOx correlation has the WRONG SIGN in ∂EI/∂φ_PZ

This deserves its own subsection because **EI_NOx is an optimizer
objective**.

`record["EI_NOx"]` is `EI_NOx_lefebvre`. Lefebvre's form uses the *fixed*
stoichiometric temperature inside the exponential and the primary-zone
temperature in the denominator, so it **does not actually depend on φ_PZ**;
the −25 % slope seen across a lean sweep is a pure artefact of the
denominator (grams of NOx *per kg of fuel*, and the fuel flow moves with
η_comb). Measured with the geometry frozen (`--sweep phi_pz`):

| φ_PZ | **lefebvre (headline)** | rizk_mongia | zonal | odgers | laviolette | EI_CO |
|---|---|---|---|---|---|---|
| 0.604 | 2.619 | 3.627 | 0.013 | 0.527 | 0.023 | 304.4 |
| 0.797 | 2.221 | 3.627 | 1.048 | 3.251 | 0.544 | 117.0 |
| **1.048** | **2.000** | 3.627 | **3.756** | **8.911** | **3.172** | **57.6** |
| 1.260 | 2.102 | 3.627 | 0.246 | 5.594 | 1.405 | 81.5 |
| 1.599 | 2.376 | 3.627 | 0.000 | 1.597 | 0.157 | 177.1 |

The headline has its **minimum exactly at φ_PZ ≈ 1.05** — the point where
the adiabatic flame temperature is highest and thermal NOx should peak.
The three **zonal** forms (zonal, Odgers, Laviolette) give the right sign
*and* peak there. EI_CO behaves correctly (bottoms out at 57.6 g/kg and
rises on both sides).

Two companion findings: **G14** — the Rizk & Mongia *global* form is
insensitive to φ_PZ by construction (0.00 % over the whole sweep);
**G15** — the *zonal* form is insensitive to τ (< 2 %). For a φ trend use
the zonal; for a τ trend use Odgers or Lefebvre.

**Action, and it costs no new data: the headline must become the zonal
form.** It is a model choice, one line in `physics_core`. It is documented
rather than silently patched because changing the record's headline key
moves the frozen `REF_CB10` anchor, and moving an anchor is a deliberate
act that must cite the correction that moved it.

**Honesty rule.** These correlations give **trends** (±30–50 % at best,
even with a calibrated CRN). Phy-CB never presents NOx/CO as certifiable
absolutes.

---

## 9. L1 — 1-D flow network + CRN (Cantera)

### 9.1 The 1-D network

Unknowns: ṁ through each path (swirler, PZ/SZ/DZ rows, effusion, cooling)
and ΔP_L — **7 in total**. Equations: ṁ_h = C_d·A_h·√(2ρ₃ΔP) per path
(§4.1) + mass conservation + **a single ΔP_L** (the liner is a plenum) +
diffuser/snout form losses (`K_DIFFUSER = 0.20`, `K_SNOUT = 0.10` ⚠️) and
ram recovery in the annulus. Solved by **bounded Newton** (≤ 30 iterations,
forward-difference Jacobian, bounded backtracking, convergence test) — no
open `while`. **It does not need Cantera: it always runs.** Output: real
splits, per-zone velocities, per-zone τ.

The C_d of each path carries the Reynolds deficit
`f_Re = 1 − 0.25·e^{−Re/3000}` and, for the effusion path, the shared
inclined-hole factor of §4.1 — the same number the LBM uses, guarded
bit-for-bit by T49c.

### 9.2 CRN — minimal topology

```
Swirler/PZ:  PSR with recirculation  → φ_PZ, τ_PZ   (2 parallel PSR if double recirculation)
   ‖ lateral mixing
SZ:          PSR (or PSR+PFR)         → φ_SZ, τ_SZ
   ‖ dilution jets
DZ:          PFR + cold bypass        → φ_global, τ_DZ
```

Cantera: `IdealGasReactor`, `MassFlowController`, `Valve`, `ReactorNet`;
the DZ PFR is discretised as `CRN_N_PFR_DZ = 5` reactors in series and the
CTRZ recirculation scales with the swirl number
(`CRN_RECIRC_AT_SN1 = 1.00`). Configurations of 12 PSR + 1 PFR + 3 mixers
reach < 12 % error in NOx (✅). Real-time LBO with 3 PSR in series plus a
recirculation path (✅). Evaporation correction (Lanewala over Rizk &
Mongia, ✅ verified): FAR = FAR₀·(−0.0295·τ_e/τ + 0.45).

**Hard timeout on Windows.** There is no `SIGALRM`, and a
`ProcessPoolExecutor` per point would cost more than the solve itself. The
choice is a third way: the network is advanced with `net.step()` inside a
**bounded** loop (`CRN_MAX_STEPS = 200_000`) and the deadline is checked
between steps. Zero processes, zero threads, deterministic physics.
Declared limitation: if a *single* CVODES step hung (never observed), the
deadline would not interrupt it. Measured: `source='crn_L1'` in **2 495
ms**, degrading as `crn_L1(cantera_unavailable→L0)`,
`crn_L1(timeout→L0)` or `crn_L1(diverged→L0)` — three distinct labels,
because collapsing a timeout and a divergence into one label was a real
bug the suite caught.

**PF does NOT come out of the CRN.** A CRN is 0-D per zone: it has no
spatial temperature distribution at the exit and therefore *cannot*
produce a pattern factor. The L1 record keeps L0's PF untouched and says
so.

Measured on the anchor: EI_NOx_crn = **3.07 g/kg**, EI_CO_crn = 19.6 g/kg,
η_comb_crn = 0.9954 — the least empirical emissions number in the package,
and inside the reference band for micro kerosene (0.1–5 g/kg with the
declared ×2 factor).

### 9.3 Mechanisms by fuel

| Fuel | L1 (CRN) | L2 (EDM/EDC) | Notes |
|---|---|---|---|
| Jet-A/kerosene | skeletal 39sp/153r or 32sp/73r; reduced 18sp | **2S_KERO_BFER** (2-step) or 4-step | |
| Propane/NG | GRI-Mech 3.0 (53sp) or DRM-19 | 2-step (WD2) / JL4 | the WD2 step 1 is what V6 validates |
| CH₄ | GRI-Mech 3.0 | **2S_CH4_BFER** (A = 4.9e9 cgs, Ea = 35 500 cal/mol, orders 0.5/0.65 — confirmed against the CERFACS database) | the flame front is *thinner* than propane's: dx ≤ 100 µm in laminar |
| H₂ | San Diego reduced **9sp/21r** | 9sp/21r directly (cheap) | ⚠️ standard EDM/flamelet **FAIL** with H₂ (preferential diffusion, Le≠1) ⇒ an L2 for H₂ needs FGM with preferential diffusion. Micromix H₂ enters through L0/L1 first |

L0 thermochemistry: Cantera's `gas.equilibrate('HP')` ≡ NASA CEA with a
verified 0.06–3.5 % deviation in T_ad (✅ arXiv:2503.11826). Without
Cantera, T_ad comes from trilinear interpolation over versioned
`thermo/tad_fit_<fuel>.csv` grids **generated with** Cantera, with the
measured interpolation error written into the table header (below the 1 %
requirement). **Cantera on Windows: install from conda-forge, NEVER pip**
(the PyPI wheel links a single-threaded LAPACK: 4–8× slower reactor
networks). This is a house rule, encoded in `pyproject.toml` and in CI.

Chemistry rates at L2a are additionally cross-checked against
**Pyrometheus** (Cantera → NumPy code with an AD Jacobian): table ==
Pyrometheus == Cantera to 5 significant figures at the same state
(171.66 kg/m³/s at 1600 K, φ = 1).

### 9.4 Post-processing NOx from L2

Standard CFD-CRN method (✅): cluster the L2 field cells by (T, φ, mixture
fraction) → post-processing CRN in Cantera → thermal NOx (extended
Zeldovich) + prompt + N₂O. This turns L2 into an **automatic CRN topology
generator** and reuses every line of the L1 code — the highest
value-added path in the system. **Implemented** in `post.py` and run at
M4: clustering on (T, Z) → `crn_core`, measured EI_NOx = 3.30 g/kg on the
anchor case. Without Cantera it degrades with a reason instead of
returning a number.

---

## 10. The complete fidelity ladder

The `Fidelity` IntEnum is frozen ABI: `L0=0 · L1=1 · L15=2 · L2A=3 ·
L2B=4`. Columns 3–4 are the *science*; the last is the **measured**
implementation state on the project machine.

| Level | Model | Cost (measured) | Predicts reliably | Does NOT predict | State |
|---|---|---|---|---|---|
| **L0** | correlations §§1–8 + Cantera equilibrium | **17 ms** | T_ad, T₄, FAR, sizing, nominal splits, Δp/p, trends of η/PF/LBO/EI | anything spatial | ✅ complete |
| **L1** | 1-D network + CRN | **2.5 s** | real 1-D splits, per-zone τ, NOx/CO trends (±30–50 %), LBO via Da, η_comb | PF, local wall temperature | ✅ complete |
| **L1.5** | cold LBM over the PicoGK voxels | 23 min (GPU, 7 M nodes) | recirculation topology, which way the air goes, CRN auto-topology | **anything quantitative** (see §11.5) | ⚠️ runs, **fails its own acceptance** |
| **L2a** | native reactive RANS (IB over the SDF) | **21.7 min** (206 k cells, GPU) | flow topology, dilution penetration, **PF ranking and hot spots between designs**, Δp/p ±15–20 %, T₄ ±3–5 % | absolute NOx (×2–10), absolute PF, quantitative LBO, thermoacoustics, **quantitative metal temperature** | ⚠️ **M1–M3 done and verified; M4 implemented, not closed** |
| **L2b** | OpenFOAM (`reactingFoam`) via subprocess | 0.5–12 h | L2a with more solidity + CHT + radiation | same absolute NOx | ❌ **OUT OF SCOPE** (§11.4) |
| L3 | LES | out of scope | — | — | — |

Honesty label attached to every L2 figure: **"design comparison /
hot-spot detection — not suitable for emissions certification"**.

The family's high-fidelity feedback hook is preserved:
`register_hifi_pair(theta, y_hifi)` applies a per-output affine correction
so that L2a/L2b (or a test rig) can calibrate L0/L1. It is currently the
**identity**: it needs ≥2 high-fidelity pairs and there are zero. The
validation campaign deliberately runs with `calibrate=False` — with the
affine correction active it would be validating the calibration, not the
physics.

---

## 11. The native L2a solver (IB-RANS over PicoGK's SDF)

### 11.1 Why this architecture

PicoGK works internally with a **narrow-band SDF voxel field** — the
canonical input of a cut-cell / immersed-boundary method. No conversion, no
meshing (the dominant bottleneck of industrial CFD). Structural bonus:
**the wall distance = |φ| is available exactly in every cell, for free** —
in a body-fitted solver computing it is an expensive problem, and wall
functions and turbulence models need it. Precedents: `sdfibm` (SDF-based
IB in OpenFOAM, CPC), INCA (cut-cell + reactive + AMR).

```
PicoGK voxel field (SDF φ)
  → resample onto the solver's uniform Cartesian grid (dx = 0.5–2 mm)
  → φ<0 solid; φ>0 fluid; normals = ∇φ (analytic)
  → ghost-cell IB (v1; conservative cut-cell as an upgrade)
  → wall distance = |φ| ⇒ y⁺ and wall functions directly
```

**Declared limitation of the chain as built**: `mesh.sample_vdb()` exists,
imports lazily and degrades with a reason — there is no OpenVDB binding
for Windows/CPython 3.13 — so what actually runs is
`sdf_combustor_from_contract()`, built analytically from the geometry
contract (liner ∪ annulus ∪ resolved rows, the d/dx ≥ 4 rule, wall
inflated to ≥2·dx). Unresolved rows and the effusion enter as a calibrated
volumetric source rather than as geometry.

### 11.2 Numerical decisions: planned v1 vs as-built

| Component | v1 choice (plan) | **As built, and why** |
|---|---|---|
| Regime | pseudo-compressibility / explicit low-Mach with preconditioning | **same** — avoids the implicit pressure Poisson, which is where home-made reactive solvers die |
| Discretization | FV 2nd order, upwind + TVD limiter | **same** (MUSCL+TVD, SSP-RK, Rhie–Chow) |
| Turbulence | realizable k-ε + wall functions (via \|φ\|) | **realizable k-ε (Shih 1995) with full A_s**, C₁ = max(0.43, η/(η+5)), P_k ≤ 10·ε, **two-scale wall functions** (u\* from k + inverted Spalding law) and **simplified Spalart–Shur** curvature correction (r̃ = 0, declared) |
| Combustion | EDM (Magnussen): rate = min(mixing ε/k, global Arrhenius) | **same**: `min(EDM, Arrhenius)` with a consistent point-implicit consumption (products, O₂ and heat all follow the *realised* consumption) |
| Chemistry | rates generated by **Pyrometheus** | done and used as a **third independent check** of the rate table |
| Precision | **FP32 mandatory** | **FP64 by default** — declared deviation: in FP32 the V1/V2 laminar cases lose accuracy; FP64 is free on CPU and costs ×1.8 on large GPU meshes, and the V&V is worth more than the 1.8 |
| Backend | NVIDIA Warp (1st choice) or JAX | **CuPy** — neither Warp nor JAX is installed and neither gets installed; CuPy is a near drop-in NumPy replacement, so **there is one kernel, not two implementations of the physics that can diverge** |
| Radiation/CHT | v1: none (thermal wall function) | **none** — and therefore v1 promises **no metal temperature** |
| Mesh | 200 k–800 k cells for a micro turbine (Ø40–120 mm) | measured 206 k cells at dx = 2 mm; **dx ≤ 1 mm (~1.6 M cells) is the declared next step** (§11.3) |
| Small holes | dilution Ø2–5 mm: local AMR or **C_d-calibrated mass/momentum sources**; effusion Ø0.3–0.5 mm: **unresolvable** ⇒ porous-wall model | as planned: `TranspirationSource` with the discharge spot sized as `s_r = ½·d·√C_d` so that the discharge area is exactly C_d·πd²/4 |

**Performance budget — measured, not estimated** (`run_perf_budget.py`,
3-D cube with an immersed sphere so the ghost-cell path is in the count):

| cells | NumPy/CPU s/cell/step | CuPy/GPU s/cell/step | winner |
|---|---|---|---|
| 8 000 | 7.37e-7 | 7.68e-6 | CPU ×10.4 |
| 64 000 | 5.91e-7 | 9.05e-7 | CPU ×1.5 |
| 140 608 | 2.00e-6 | 4.02e-7 | **GPU ×5.0** |
| **512 000** | **2.01e-6** | **1.17e-7** | **GPU ×17.1** |
| 1 000 000 | 2.04e-6 | 9.66e-8 | GPU ×21.1 |

The wave's target — 500 k cells × 10 k iterations in 5–40 min on an
RTX 4070+ — **is met on this RTX 3070 Ti: 9.8 min with the GPU free,
16.8 min with the GPU shared at 99 % by another job**, against **2.6–2.8 h**
on NumPy/CPU. Three measured facts that travel with that headline:

* **Below ~100 k cells the GPU loses**, and badly (×0.4 at 32 k): an RK3
  step launches several hundred kernels and the launch cost dominates.
  `backend="auto"` therefore stays on NumPy below
  `GPU_MIN_CELLS = 100 000` — a rule with a table behind it, not a hunch.
* Where the time goes on GPU (512 k cells, FP64, phase-by-phase with
  `sync()`): `rhs` 65 %, full `apply_bc` 31 %, **of which the IB
  ghost-cell is 9 %** — so the immersed-boundary indexing *does*
  vectorise; it is not the bottleneck. Device↔host transfers inside the
  loop: **1.33 per step** (the CFL scalar, plus residuals every N steps);
  no field ever moves.
* The remaining ×11.7 against the public FP32 reference (JAX-Fluids, 1e-8
  s/cell/step) is explained: one array operation = one kernel, with no
  fusion, where XLA fuses. Recovering it is optional optimisation, not a
  blocker.

VRAM, measured with CuPy's pool: 19.7 MiB (FP32) / 38.8 MiB (FP64) at
512 k cells — the science's ~30 MB estimate was conservative by ~1.5×,
which is the right direction for an estimate. The guard
`backend.fits_in_vram` (60 % margin) degrades to CPU **with a label**
rather than dying; the limit on this machine is 46.4 M cells in FP32.

### 11.3 The three risks that kill reactive solvers — and where each one stands

1. **Flame start-up**: a cold field does not ignite by itself. Mitigation
   as planned — imposed hot patch for N iterations + pre-exponential ramp
   + T limiter + aggressive under-relaxation. **Measured (V6c): a cold
   1-D propane field ignites, anchors (integral heat release stable within
   ±2 % over 400 iterations) and gives the same S_L as the pre-burnt
   initialisation within ±5 %.** And at M4, the flame **anchors in the
   real combustor**: 110.5 kW with 0.70 % oscillation against a 2 %
   threshold, 21.7 min on GPU. **Risk #1 is resolved.**
2. **Chemical stiffness**: global 2–4-step mechanisms in v1; point-implicit
   integration with Pyrometheus' AD Jacobian if skeletal chemistry ever
   enters. The point-implicit source had to be made *consistent* — an
   early version consumed fuel implicitly but created products and heat
   from the *demanded* rate, so mass appeared out of nothing (Σ Y reached
   1.47). Caught by test M3d before it could contaminate V6.
3. **Thermal wall accuracy with IB**: the literature documents degraded
   thermal boundary layers over immersed boundaries ⇒ **no quantitative
   metal temperature is promised at L2a**; the wall map is **adiabatic and
   for ranking**, with that label pinned to it in `post.HONESTY`. A
   two-layer wall model is the mitigation; quantitative liner temperature
   was delegated to L2b — which is out of scope (§11.4), so the capability
   disappeared from the system rather than moving elsewhere.

**M4's open items, declared rather than disguised** (measured 2026-08-03,
KJ66 anchor, dx = 2 mm, 206 550 cells):

| measurement | value | wave criterion |
|---|---|---|
| flame anchored | **YES** | ✅ risk #1 |
| heat release | 110.5 kW, oscillation 0.70 % | ±2 % ✅ |
| budget | 21.7 min | < 30 min GPU ✅ |
| mass closure | −4.2 % | (was −42 % before the bug fixes below) |
| penetrations Y/d | 0.53–0.72 | (were 0.003–0.022) |
| **T₄** | **1445.8 K** vs 1150 K | **+25.7 %, FAILS ±5 %** ❌ |
| Δp/p | 0.0189 vs 0.030 | **not comparable** (see below) |
| PF | 0.162 (L0: 0.350) | no absolute criterion yet |

Two structural limits at dx = 2 mm, and neither is a bug: in the KJ66 the
liner↔casing clearance is ~3 mm and the liner wall 0.8 mm, so with the wall
inflated to 2 cells **the annulus does not fit**. The domain is the liner
interior only and every row enters as a source. Therefore (a) **the air
split is an INPUT, not a prediction** — it would only be a prediction with
the annulus resolved, where the paths compete for one ΔP, which is exactly
what the 1-D network does; and (b) **the orifice loss does not happen
inside the domain**, so Δp/p is not comparable with the network and
`post.dp_over_p` declares `comparable_with_1d=False`. The missing part is
*not* synthesised from the same C_d that would be validated — that would
make the contrast tautological. Partial diagnosis of the T₄ excess: with
the model's calibrated c_p a fully-mixed T₄ would be ~1028 K, so the excess
is **incomplete mixing** (the 16 % effusion air enters distributed and
slow, hugs the wall and weighs little in a ρu-weighted average), not c_p.
The pattern factor is currently governed by the **n = 4 azimuthal mode of
the Cartesian mesh** (rows of 8, 8 and 10 holes; 6 injectors), so the M4c
ranking test would measure mesh noise and **is not run**. All of it points
at dx ≤ 1 mm (~1.6 M cells, 1–2 h/case).

**Three real bugs the M4 ladder caught**, listed because "the tests found
nothing" is usually a statement about the tests:

1. **A transpired row that injected no momentum.** The spot was sized for
   geometric convenience instead of the discharge area C_d·πd²/4, so air
   entered at ~15 m/s instead of ~70 m/s: no orifice loss (Δp/p halved)
   and no penetration (Y/d ≈ 0.003).
2. **A velocity BC does not deliver a mass flow** — the field decides the
   flow, and the Dirichlet wall delivered 17 % less mass. Replaced by a
   **volumetric source**, which makes the flow rate exact by construction.
3. **The mass source was scaled with ρ₃ instead of the local ρ.** In the
   flame (ρ ≈ ρ₃/5) five times less dilution air entered: T₄ went to
   1791 K (+56 %) and mass closure to −42 %. **Invisible in cold flow**,
   because there ρ ≡ ρ₃.

Plus two post-processing bugs: the wall map was reading the **ghost** cell
(i.e. its own boundary condition: ~430 K next to a 2400 K flame) and the
penetrations included that same layer, so Y/d came out ≈ 0 by
construction.

### 11.4 L2b — the OpenFOAM backend (plan B / confirmation) — **OUT OF SCOPE**

Chain, as designed: contract → STL (PicoGK) → `snappyHexMesh`
(castellate→snap→layers) → `reactingFoam`/`rhoReactingFoam` (EDM/PaSR,
CHT, radiation) → read back with **`foamlib`**. Distribution: WSL2/Docker,
**never bundled** (GPL-3.0: subprocess invocation with file exchange is
"mere aggregation"; if packaging were ever required, the LGPL alternative
is SU2 v8 with FGM, with meshing as a separate problem). Known risk:
`snappyHexMesh` over generative geometry (lattices, internal channels) can
fail or produce poor meshes.

**Project decision (2026-07-31): out of scope.** The target machine is
Windows; OpenFOAM would need WSL2/Docker, i.e. non-runnable code on the
target machine, and this house does not ship numbers it cannot run. The
enum value is **kept** because it is ABI (it travels in
`record["fidelity"]`, in checkpoints and in `phys_cache.jsonl`), and
`--fidelity L2B` degrades with `source="openfoam_L2B(out_of_scope→…)"`.
The consequence is declared, not absorbed: **quantitative metal
temperature disappeared from the system** (§19, limit 1).

### 11.5 L1.5 — cold LBM (XLB) — and the prediction this document got wrong

Design intent: a cold isothermal lattice-Boltzmann field over PicoGK's
voxels → real air splits, jet penetration, recirculation zones →
**automatic construction of the L1 CRN topology** (the CFD-CRN approach of
§9.4). Implemented with three backends behind one API: XLB (absent),
**CuPy** and a pure-NumPy D3Q19 BGK fallback with Smagorinsky LES — one
core, one physics, five isolated device-specific spots.

Measured: **193 MLUPS (5.17e-9 s/cell/step)** on the GPU, which *beats*
the 1e-8 reference. That bought resolution, and

> **resolution was not the binding constraint.**

The anchor went from 0.92 M nodes at dx = 1.229 mm to **6.98 M nodes at
dx = 0.613 mm** (annulus 4.0 cells, min hole Ø 8.4 cells) and it
*converged* (37 800 steps, 1 365 s). It still fails:

> **τ stays pinned at `TAU_FLOOR` = 0.5015 against a molecular
> 0.50001–0.50003, so the effective viscosity is ×53–106 the real one and
> Δp/p comes out 0.587 against a design 0.030.** Refining the mesh halved
> the viscosity ratio and did not fix it. What is needed is a collision
> operator that survives τ → 0.5 (MRT/TRT/regularised BGK), **not more
> cells.**

An earlier revision predicted that "dx ≤ 0.6 mm would make it
quantitative". That prediction has been **refuted by measurement** and is
recorded here rather than quietly dropped. Acceptance measured at the
coarse mesh (`test_cfd_l15.py --slow`):

| Check | Criterion | Measured | |
|---|---|---|---|
| **T43** jet in crossflow, J = 4 | Y/d within ±25 % of 1.15·√J | 3.00 vs 2.30 → **+30.4 %** | ✗ |
| T43, J = 16 / J = 36 | ±25 % | +13.0 % / −2.9 % | ✓ |
| T43 monotonicity | Y/d grows with J | 3.00 → 5.20 → 6.70 | ✓ |
| **T44** KJ66 splits vs the 1-D network | per zone < 20 % | pz −22.5 · sz **−62.1** · dz **−51.4** · cool **+242.5 %** | ✗ |
| T44e mass closure | < 1 % | 1.013 % | ✗ |
| T44g Δp/p vs design | < ×3 | **3.70 vs 0.030 → ×123** | ✗ |
| T44h steady state | converged | **not converged in 9 400 steps** | ✗ |
| **T45** auto-topology CRN | same NOx trends | dragged by T44 (same field) | ✗ |

The failure pattern is itself evidence: the weak jet (J = 4) is exactly the
case an over-viscous field keeps standing up too long, while the
momentum-dominated jets pass. And the similarity station used for T43 was
derived **a priori** from the classical trajectory (1.6·s₀^⅓ = 1.15 ⇒
s₀ = 0.371) and **was not moved to make the test pass**. Two further
declared distortions at dx = 1.229 mm: the 0.80 mm liner wall is thinner
than one cell, so 27 188 gas cells (9.6 % of the liner volume) had to be
**forced solid** to seal it; and the Ø0.6 mm effusion holes are
unresolvable and enter as distributed transpiration with no momentum.

**Standing limit: the L1.5 cold field is usable for TOPOLOGY —
recirculation structure, which way the air goes, CRN auto-topology — and
is NOT quantitative.** That is not a prediction; it is what the acceptance
tests measure, and they are reported as failing rather than re-tuned into
compliance.

---

## 12. Structural-thermal model of the liner (layer 1s)

Analytic (< 1 ms), mirroring the aerodynamic ladder (L0s analytic → L1s
axisymmetric FEA → L2s 3-D FEA), and it is a **hard** constraint block of
the optimizer:

- **Materials** (9 alloys) with piecewise-linear thermal derating k(T),
  clamped at the ends (never extrapolated), service limits, flame-wall
  limits, minimum LPBF thickness, emissivity: IN625 (micro liner, ≤ ~980 K
  continuous service with limited creep), **Hastelloy X** (industry
  standard for liners, ~900–1150 °C; note: susceptible to LPBF cracking in
  thin walls), Haynes 230/282. **Hard rule: IN718 is automatically
  rejected for any flame-exposed surface** (limit ~650 °C — γ″
  over-ages). The casing *may* be IN718 because it lives at T₃, not at
  flame temperature.
- The `alpha` (CTE) field that the sibling projects left unused is used
  here: **gradient thermal stress** σ_th ≈ E·α·(T_w1−T_w2)/(2(1−ν))
  against the derated yield k(T)·σ_y.
- **Start-stop LCF proxy** (simplified Coffin–Manson): **a reported
  metric, not a constraint in v1** (`is_constraint=False`) — the same
  criterion the siblings apply to their own too-coarse screens.
- Pressure: liner hoop (ΔP_liner) and casing hoop (P₃ − P_atm), each
  against its own wall's derating **at its own temperature**.

`g_struct ≤ 0` has 5 components with frozen order, is continuous, never
raises, and shares its source of truth with the physics: the wall limit in
`g[1]` and the minimum thickness in `g[7]` both come from the *same*
material entry. Materials are handbook-typical: *substitute certified
specs for real design.*

---

## 13. Verification and validation

The family keeps the two strictly apart: **verification** asks whether we
solve the equations right (`test_phycb.py` + the CFD ladder);
**validation** asks whether they are the right equations (`validation/` →
[VALIDATION.md](VALIDATION.md)).

Measured today on this machine: `python test_phycb.py` →
**355/355 checks OK · 1 declared skip · 32.9 s** (the skip is the expensive
C#/PicoGK smoke T29, which runs under `--slow` and takes the count to
359/359). A skipped case is never a check that passes trivially: it is run
by hand with its own script and its numbers are recorded.

### 13.1 The CFD V&V ladder

Thresholds were fixed **before** the runs and **none has been loosened**.

| # | Case | Verifies | Criterion | Measured | |
|---|---|---|---|---|---|
| V1a | Poiseuille, both walls immersed and offset 0.25 dx | diffusion + IB BCs | < 2 % | **0.17 %** | ✅ |
| V1b | Couette, moving immersed wall | moving IB | < 2 % | **0.04 %** | ✅ |
| V2 | lid-driven cavity Re = 100 (32²) / Re = 1000 (128²) | p–v coupling (Ghia et al. 1982) | max\|Δu\| < 0.05 | **0.0044/0.0070** · **0.0048/0.0099** | ✅ |
| V3 | cylinder in cross-flow with IB, D/10 | IB accuracy | C_d ±10 %, St ±10 % | **1.3958 (+3.4 %)** · **0.16638 (+0.8 %)** | ✅ |
| M2a–M2e | k-ε decay, channel (Dean), low-Mach thermal expansion, scalar | model machinery | analytic / ±15 % | 0.14 % · −3.7 % · +0.94 % · < 1e-6 | ✅ |
| V4 | backward-facing step Re 28k (Vogel & Eaton) | k-ε + wall functions (the most discriminating test) | x_r/h = 6.67 ±15 % | **6.23 (−6.6 %)** | ✅ |
| V5 | plane turbulent jet (Pope) | turbulent diffusion | S = 0.10 ±15 % | **0.0877 (−12.3 %)** | ✅ |
| V5-JICF | hot jet in crossflow (Kamotani & Greber) | k-ε + energy + ρ(T) + scalar together | ±25 % | **+19.7 %** | ✅ |
| V6 | 1-D premixed propane flame, S_L vs Cantera `FreeFlame`, **same mechanism** | chemistry + coupling (**exact self-validation**) | ±10 % | **−2.3 %** (0.2476 vs 0.2534 m/s) | ✅ |
| V6c | **ignition from cold** (risk #1) | start-up plan | anchors, same S_L ±5 % | **anchors; ΔS_L < 5 %** | ✅ |
| V6d | CH₄ flame (2S_CH4_BFER) | the V8 mechanism | ±10 % | **−8.9 %** at dx = 60 µm (−19 % at 150 µm) | ✅ |
| V7 | counterflow diffusion flame | diffusion structure, extinction | ±10 % | **−0.3 %** (2143.7 vs 2150.1 K) | ✅ |
| V8 | Sandia Flame D (CH₄/air piloted, Re 22400) | turbulence–chemistry interaction | ±15 % qualitative | **PENDING, declared** | — |
| V9 | PRECCINSTA (DLR swirl burner) | swirl + recirculation + confined flame — the closest case | qualitative expectation | **PENDING, declared** | — |
| V10 | own printed chamber on a rig | closes the loop | — | **does not exist** | — |

Three points worth keeping. **V6 is the keystone**: our solver and Cantera
get the *same* mechanism (through a YAML that declares the table's units,
so our own unit conversion is under surveillance), the same Le = 1
assumption and the same T_ad — so a ±10 % agreement is a statement about
the solver, not about the chemistry. **V9 will need wall heat loss**
(it changes the flame from M to V shape) and RANS is hard there: the
expectation is declared as qualitative *before* the run. And the ladder
**caught real bugs**: a sealed fluid volume that made the residual
identically zero (a solver "converging" without moving anything — exactly
the false green worth hunting), a Newton iteration for Spalding's law
replaced by bounded bisection (machine residual across all three bands),
and a moving immersed wall that crashed on GPU because `np.asarray` throws
on a CuPy array by design.

One methodological result deserves its own line. V3 did **not** pass at
first (C_d = 1.682, +24 %). It was diagnosed by measuring rather than
reasoning: first the *measurement* was verified (the control-volume force
over three very different interior boxes scattered by 0.5 %, so the
measurement was right and the bias was in the field), then the ghost-cell
image point was swept **against V1's analytic solution** —
`IMAGE_MIN_DX` 0.05 → 1.50 — and fixed at **0.55**, the value that
minimises the error against the only exact judge available. It was **not**
calibrated against C_d: that would be fitting to the answer. Effect on the
full V3: C_d 1.682 → 1.396.

### 13.2 Validation of the preliminary design (L0/L1)

Ladder of anchors: (1) the per-power-class reference table; (2) the
Bath/HiETA case (IN625, radial swirl with vane injection, T₃ = 1093 K
recuperated); (3) the 39.35 % → 32 % PZ split of the optimised micro
turbojet; (4) KJ66 (verified vaporizer geometry) as the hobby
architecture; (5) an own rig. v1 criteria: Δp/p ±20 %, T₄ ±3 %, PF inside
the class band, correct NOx trend against φ_PZ and τ — **with no
per-machine recalibration, ever**. If different constants were needed per
machine, the model would be wrong, not the constant.

**Marker** (`validation/RESULTS.md`, regenerated 2026-08-01, `L0` +
Cantera CRN, `calibrate=False`): **PASS 108 · FAIL 48 · all 48 are
documented gaps · 0 obsolete gaps**. The prose of `VALIDATION.md` was cut
one day earlier, on 2026-07-31, at **107 PASS · 49 FAIL**, and has not
been regenerated since the annulus correction of §2.1 (which also moved
the anchor's wall temperature from 1059 K to 994.6 K). The two documents
therefore disagree by one check; the regenerated evidence file is the one
to believe, and the drift is reported here rather than averaged away.

Per case: A1 16/6 · A2 13/9 · A3 13/9 · A4 9/2 · B1 21/5 · B2 15/7 ·
**C1 5/0** · D1 4/6 · D2 12/4.

**Two of the five v1 criteria are NOT evaluable on this machine, and
saying so is part of the deliverable**: Δp/p (it is an *input* at L0) and
T₄ (it is θ[19], a pinned operating point; what is evaluated instead is
the **closure** of the enthalpy balance, T₄→FAR→T₄, which shuts to
< 4e-3 % in all six cases). LBO is not evaluable either: there is no
measured blow-out datum in the package.

**The gaps, by family** (identifiers `G#` live in
`validation/validate.py::KNOWN_GAPS`; `--strict` fails on a regression, on
an undocumented finding, **and on a known gap that starts passing** — you
cannot fix the code without updating the document, nor the reverse):

*Data gaps (the model is not at fault).* **G1** is the campaign's main
finding and explains **17 of the 48 failures**: the cycle enthalpy balance
— pure algebra, no provisional coefficient inside — applied to the
reference table's own (T₃, T₄) bands imposes a FAR that does not overlap
the FAR band the same table declares, and for class 0 (non-recuperated by
definition) it is outright impossible. The table's FAR band is only
compatible with **recuperated** operation. **Action: fix the DATA, not the
model.** Calibrating against an inconsistent datum is the worst possible
move in a validation campaign. G2 (two `verified` tables disagree on
frac_cool), G3 (effusion hole count out of the table's band — and this gap
*grew* with the ×1.375 correction of §4.1: A2 went from 3696, inside the
band, to 5080, outside, and the marker was **not** repaired by moving a
threshold: A2 had been passing because of a geometry error) and G4 (V_c out
of band by continuity) are of the same family.

*Model gaps.* G5 pattern factor returns its calibration (§4.3) · G6 casing
diameter systematically small in classes 1–2 (measured suspect: the
`micro_target` PLF = 20 is used for every architecture while the
architecture row gives 36 for `can` — and A_ref ∝ √PLF gives exactly the
order of the deficit) · G7 η_comb below band in classes 1–2 (driver:
η_mix) · G8 airblast SMD and the LBO margins (§6.1) · G9 φ_PZ out of band
— in D1 this is a finding of its own: at T₃ = 1093 K the balance imposes
φ_global = 0.118 and **not even with `frac_pz` at its contract minimum**
does φ_PZ reach 0.8, which in design terms means that in extreme
recuperation a real combustor cannot split its air by these rules; it needs
piloting/staging · G10 τ out of band · G11/G12 wall temperature and the
IN625 falsification (§5) · G13/G14/G15 the emissions correlations (§8.6).

**Recalibration priority** — if something fails, it is not all touched at
once; it is touched in this order, the movement is recorded with its
justification, and **all cases are re-run after each adjustment**:

| # | What to touch | Why it is the first suspect | Closes |
|---|---|---|---|
| **0** | `class_table.csv` — the **data**, not a constant | its FAR and (T₃,T₄) bands cannot both be true | G1–G4 |
| **1** | headline EI_NOx: `lefebvre` → `zonal` | wrong sign vs φ_PZ, demonstrated by sweep; **needs no new data** | G13 |
| **2** | ε_g floor (and behind it the 0.29 coefficient) | house calibration currently *substituting* for the declared physics; ±765 K on wall temperature, `g[1]` changes sign; it is what declares a combustor that exists to be unbuildable | G11, G12 |
| **3** | airblast SMD → φ_LBO chain | SMD ×3–4 small ⇒ LBO margins ×42–476. Careful: the SMD correlations are *verified*, so the suspect may be the class band or the effective ALR | G8 |
| **4** | PF's k (`pf_constant.csv`) | calibrated at one point, and that point *is* the anchor | G5 |
| **5** | PLF per architecture | explains the casing-diameter deficit | G6, G10 |
| **6** | `K_MIX_DP` and the penetration penalty | η_mix is the driver in 4 of 6 cases | G7 |
| **7** | `C_LBO_SI` | one point, ±45 % declared; sits behind the SMD | — |
| **8** | the η_c(θ) chart | one anchoring point; it does not bite today (aero criterion wins everywhere) but if the real chart is stricter the whole geometry moves | latent risk |

**What will not be done:** tuning a constant per case until it fits. That
is what separates this campaign from a curve-fitting exercise.

**Minimum UQ.** Bands are computed as σ_total = √(σ_model² + σ_ensemble²)
and are a **declared approximation**: they assume relative, Gaussian,
mutually independent errors and a locally linear model — none of the four
is proven, and σ_ensemble = 0 in the campaign because no surrogate is
trained. On the anchor: φ_LBO 0.5245 → [0.289, 0.761] (±45 %, the declared
accuracy of Lefebvre's LBO model); EI_NOx 2.00 → [1.00, 3.00] and EI_CO
57.57 → [28.8, 86.4] (±50 %). Two quantities get **no** Gaussian σ,
because their uncertainty is not a correlation's but the measured sweep of
a coefficient calibrated at a single point — inventing a σ there would be
inventing: `T_wall_max_K` = 994.6 K with the measured interval
**[808, 1573] K**, and `PF` = 0.3500 with **[0.1225, 0.5915]**.

**Level F — rig specification, ready for when a supplier exists.** It is
the only level that closes the three critical pending items (ε_g floor, PF
k, C_LBO) and G8/G11/G12. Part: the `REF_CB10` anchor chamber
(reverse-flow annular, vaporizer, Ø casing 81.5 mm, L 166.3 mm) exactly as
`geometry_generator.generate()` emits it. Material IN625; LPBF per SAE
AMS7003, alloy per AMS7000; **change `pitch/d` from 6.0 to 4–5 before
printing** (that is what brings the blowing ratio back into the map);
build orientation as `dfam_checker` emits it; stress relief **before**
removal from the plate, HIP 1160–1200 °C / 100–150 MPa / 3–4 h; CT
inspection **plus permeability verification of the ~3000 effusion holes**
(with that count, "holes open" is a quality requirement, not a detail) and
of the ≥2 drain ports per cavity the checker places — trapped powder
sinters during HIP and the blockage is permanent. Instrumentation, in
order of value to this campaign: (1) **liner wall thermocouples** (≥5
axial) — the single most valuable datum, it closes G11/G12 and the ε_g
floor; (2) **lean blow-out test** (fuel ramp at fixed P₃, T₃, ≥3
repetitions) — cheap and highly discriminating, it arbitrates G8;
(3) **cold Δp** sweep — the only path to validating Δp/p without CFD;
(4) exit-plane thermocouple rake (≥12 points, 2 radii) for measured PF and
RTDF; (5) gas analyser (NOx/CO/O₂ dry, corrected to 15 % O₂);
(6) laser-diffraction SMD of the injector.

---

## 14. Verified primary sources (URLs)

Correlation verification (session 2026-07-29):

1. Airblast (Table 7: Rizkalla & Lefebvre + El-Shanawany & Lefebvre +
   Gepperth + Chaussonnet): https://arxiv.org/pdf/1906.04042
2. Rizkalla & Lefebvre 1975 OA: https://dspace.lib.cranfield.ac.uk/bitstreams/d679e116-3818-47f4-a5e9-603b4764963b/download
3. El-Shanawany & Lefebvre 1980 OA: https://dspace.lib.cranfield.ac.uk/bitstreams/80accbd0-3fd0-4a7f-afd6-7710dac0414e/download
4. Lanewala thesis, U. Toronto (Lefebvre EI_NOx eq. 1.4; Rizk & Mongia
   eqs. 3.5/1.5–1.8; NEPP/GasTurb): https://utoronto.scholaris.ca/bitstreams/96edd947-0a38-4b1a-8711-0691e934de7b/download
5. Odgers & Kretschmer + Laviolette (review): https://www.mdpi.com/2076-3417/14/14/6104
6. Rizk & Mongia 1993 (ASME, DOI 10.1115/1.2906747): https://asmedigitalcollection.asme.org/gasturbinespower/article-abstract/115/3/612/406892
7. Flow number and simplex (ILASS 2023, paper 37): https://ilasseurope.org/ConferencePapers/ILASS_Eur_2023_37.pdf
8. C_d vs roughness (Brno UT, DOI 10.1016/j.jmapro.2022.11.031): https://dspace.vut.cz/bitstreams/138e5676-73a1-48cb-b137-03e6d118b10e/download
9. Micromix H₂: https://www.mdpi.com/1996-1073/18/23/6168 · https://link.springer.com/content/pdf/10.1007/s42496-021-00091-5.pdf
10. Micro vaporizers: https://doi.org/10.3390/app16083764 · KJ66: https://www.gtba.co.uk/engine_designs/kj66.htm
11. Cantera ≈ CEA (T_ad): https://arxiv.org/pdf/2503.11826
12. Simplex comparisons (qualitative): https://www.mdpi.com/2226-4310/11/8/658

**Mandatory canonical reference for closing every ⚠️:** Lefebvre, A. H. &
Ballal, D. R., *Gas Turbine Combustion: Alternative Fuels and Emissions*,
3rd ed., CRC Press — ch. 4 (aerodynamics), 5 (efficiency, θ, PF), 6
(atomization), 8 (gas emissivity), 9 (emissions), 10 (airblast, cooling).
It is **not available on this machine**, which is why WAVE1 is complete
but not closed.

*Added for the CFD, optimization and manufacturing layers:*

13. Ghia, U., Ghia, K. N. & Shin, C. T., *J. Comput. Phys.* 48(3), 387–411,
    1982 (lid-driven cavity, Tables I–II, entered as data with their
    citation).
14. Liu, Zheng & Sung, *J. Comput. Phys.* 139, 35–57, 1998; Braza,
    Chassaing & Ha Minh, *JFM* 165, 79–130, 1986 (cylinder C_d);
    Williamson, *JFM* 206, 579–627, 1989 (Strouhal).
15. Dean, R. B., 1978 (channel friction correlation); Vogel & Eaton, 1985
    (backward-facing step); Pope, S. B., *Turbulent Flows* (plane jet
    spreading rate).
16. Kamotani, Y. & Greber, I., NASA TN D-6966, 1972, eq. 3 (thermal
    centreline of a jet in crossflow — the correct comparator; Pratte &
    Baines 1967 is the momentum/visual centreline and reads
    systematically higher).
17. Shih, T.-H. et al., 1995 (realizable k-ε); Spalart & Shur, 1997
    (curvature correction); Spalding's law of the wall.
18. Magnussen, B. F. & Hjertager, 1976 (Eddy Dissipation Model);
    CERFACS 2S_CH4_BFER and 2S_KERO_BFER two-step mechanisms; San Diego
    reduced H₂ mechanism 9sp/21r.
19. Pyrometheus (Cantera → differentiable code with an AD Jacobian);
    Cantera (BSD-3); NASA CEA equivalence, arXiv:2503.11826.
20. Deb, K. et al., "A Fast and Elitist Multiobjective Genetic Algorithm:
    NSGA-II," *IEEE TEC* 6(2), 2002 (constrained dominance, crowding).
21. PicoGK (Apache-2.0) and its narrow-band SDF voxel kernel; OpenVDB
    narrow-band grids; `sdfibm` (SDF immersed boundary in OpenFOAM, CPC).
22. XLB (Autodesk, Apache-2.0, D3Q19/D3Q27 LBM + Smagorinsky LES);
    JAX-Fluids and JANC as published GPU performance references.
23. SAE AMS7003 (LPBF process control) / AMS7000 (IN625 powder);
    film-cooling discharge coefficients: Hay & Lampard 1998, Gritsch et
    al. 1998.

---

## 15. Design space, surrogate and inverse search (layers 2–4)

The search engine is **domain-agnostic** and ported verbatim from the
siblings; only the domain adapter was written. That is a deliberate reuse
decision, and the one documented deviation is recorded: the combustor
needs ~30 keys of the L0 record to evaluate `g_struct` and `g_dfam`, so
the population evaluation starts from the full L0 record and then
overwrites the outputs with the surrogate's LCB values, instead of copying
a short tuple.

**Physics embedding (37-D).** The ensemble learns the **residual over the
physics**, not the physics from scratch: `physics_features(θ)` = θ
normalised (20) + 12 closed physical features (φ_PZ, log θ_Lefebvre, Ω,
τ, I, J_dz, blowing ratio M, T_w1, PF, φ_LBO, SMD, ΔP_L/q_ref) + **the 5
L0 predictions themselves** (`eta_comb`, `dp_over_p`, `PF`,
`T_wall_max_K`, log EI_NOx). Every component is scaled to O(1) and
**bounded**: a degenerate θ produces a finite vector, never a NaN, because
an MLP cannot be handed a NaN.

**Learned outputs** (`OUT_KEYS`, frozen order): η_comb, Δp/p, PF,
T_wall_max_K, EI_NOx — the last one learned in log, because it spans
orders of magnitude.

**Objectives** (minimized): **−η_comb** and **EI_NOx**. Δp/p is *not* an
objective: in a combustor Δp is **spent** on mixing up to the cycle
allowance, so it is a hard constraint. Feasibility is Deb's constrained
dominance (feasible ≻ infeasible; between infeasibles, lower total
violation; between feasibles, Pareto), over the 26-component global g.
Acquisition: LCB with k-means batching; bounded rounds with checkpointing
and a `dataset.csv` flywheel per run.

**The uncertainty gate is what makes this honest, and it is measured
closing.** On the `--quick` budget (132 physical evaluations, 14.5 s) the
gate **never opened**: the budget was spent on space-filling sampling and
NSGA-II never ran, so the reported incumbent is the best *sampled* point —
and the CLI says exactly that on screen instead of presenting it as an
optimum. On the full budget (460 evaluations, 5 rounds, 108.5 s) it opens
and produces a real, physically verified 3-point Pareto front:

```
 #  feasible  architecture      η_comb    Δp/p     PF   T_w [K]  EI_NOx
 0     OK     annular           0.9846  0.0300  0.365    940.5    2.42
 1     OK     annular           0.9797  0.0300  0.313   1120.2    2.12
 2     OK     can               0.9761  0.0300  0.389   1035.5    2.09
```

Three feasible, printable designs trading efficiency against NOx, with two
different architectures chosen by the gate θ[0]. **And the front measures
the defect of §8.6 in the product**: the NOx axis spans only 2.42 → 2.09
g/kg (±7 %) because Lefebvre's form barely depends on the design, so as an
objective it hardly discriminates — while the recommended point sits at
φ_PZ = 0.616 with EI_CO = 214 g/kg, ×3.7 the nominal design's 57.6. The
front is buying a small NOx gain with a large CO penalty that no objective
is watching. The house prescription when the gate stays shut is to enrich
`physics_features`, never to lower the gate threshold.

---

## 16. Geometry contract, printable parts and DfAM (layers 5a and 5c)

**The contract.** `geometry_generator.build_contract()` emits
`combustor.json` (schema `phycb-combustor-1`) with the geometry **fully
resolved** in mm and degrees, plus its sign conventions inside the JSON
itself: neither the C# layer nor the CFD layer recomputes anything. Seven
artefacts per run: the contract, `hole_table.csv`, `liner_profile.csv`,
`effusion_field.csv`, `combustor_bom.csv`, `cfd_boundary_conditions.json`,
`dfam_report.json`. **Double-path rule:** the STL is a *printable
approximation*; the **dimensional reference for analysis are the CSVs**.
When STL and CSV disagree, the CSV governs.

The three architectures are real geometry, not labels. `reverse_annular`
(the anchor's) is a reverse-flow annulus: the centrifugal compressor
discharges at the front into the outer annulus, the air travels +Z
preheating against the liner, turns 180° around the rear dome, burns
travelling −Z, and turns another 180° at the front towards the turbine,
which lives inside the combustor itself. Three rules are imposed by
construction and unit-tested: the meridional polyline does not
self-intersect; the flow area A(z) = π(r_out² − r_in²) never drops below
A_L (in the exit transition the outer radius is solved as √(r_in² + A/π),
so the area is *exactly* constant); and the 180° turning curtains have
area ≥ A_ref, which is what fixes how far the casing overhangs the dome.

**The effusion field is one implicit SDF, not N booleans.** An effusion
liner carries hundreds to thousands of holes; subtracting one boolean per
hole is the wrong operation for a narrow-band SDF kernel. The whole
pattern is described as a signed-distance function evaluated once —
`sd(p) = min over the holes near p` — on a regular (z, φ) lattice, so the
cost is **O(1) per sample, not O(N)**: building 900 holes costs what
building 9 costs. The hole axis is measured **from the liner surface**
(identical to `dfam_checker`, and the two must never diverge): with a
near-axial wall and 20–30° the axis stays more than 45° from the build
horizontal, so the hole is self-supporting and a round section is legal;
when it is not, the section switches to a **teardrop** and the switch is
logged. That coupling between the effusion angle and the build orientation
is the design rule, obtained rather than assumed.

**Two DfAM checks, and they disagree on the anchor — which is the point.**
`dfam_checker.check()` (Python, 5 analytic constraints, < 10 ms, inside the
optimizer loop) calls the smoke design **PRINTABLE**. `DfamVoxelCheck`
(C#, 3 constraints measured on the **built voxels**, after PicoGK has
actually made the part) calls it **FAIL**. Measured by building the same
contract twice:

* **The casing overhang violation is REAL, not a discretization
  artefact**: 2.359 % of solid cells unsupported at 0.8 mm voxel and
  2.366 % at 0.4 mm — i.e. it does not move. The casing genuinely has
  down-skin surfaces below 45° in the `z+` build; the analytic checker
  cannot see it because it reasons about the *contract*, not about the
  swept solid.
* **The thickness clamp IS a discretization artefact**: at 0.8 mm voxel
  the 0.8 mm liner wall and the 1.2 mm swirler vanes are clamped up to
  1.6 mm (a 2-voxel floor), so **the printed part is not the verified
  design** — the warning says exactly that, and it disappears at 0.4 mm.
  Print final parts at 0.3–0.4 mm, never at draft resolution.

The C# report frames the relationship itself: *"COMPLEMENTARY to
`dfam_checker.check()` … neither replaces the other."* The practical
consequence is blunt and belongs in this document: **a design the optimizer
calls printable can still fail the as-built check, so `dfam_report.json`
must be read before anything is sent to a printer** (§19, limit 10).

Powder drainability is settled **topologically**, by flood-filling the
void on a deliberately coarse 1 mm grid: enclosed voids (zero exits) are
unconditionally fatal, and every cavity must break out of the envelope
through ≥2 distinct patches. The coarse grid is not laziness — it is
physically right: a channel narrower than the cell does not drain powder,
so sub-millimetre effusion holes are automatically not counted as
drainage, which is exactly the rule, obtained for free instead of by a
special case.

Finally, the **voxel field is exported** (`<Name>_field.vdb`,
`<Name>_Liner_field.vdb`, OpenVDB narrow band, mm) because it *is* the
kernel's native representation: `cfd/` is meant to sample the identical
field the STL was meshed from, not a re-voxelization of a triangle soup.
The assembly STL is **inspection only** — the parts interpenetrate at
their seats on purpose.

---

## 17. The cycle seam and the integration findings

`hot_cycle(record, TIT_K, …, combustor=None)` is the Brayton hot-cycle
seam. With `combustor=None` it reproduces the documented constant filler
exactly (η_comb = 0.995, Δp = 5 %); passing a Phy-CB record or callable
substitutes the solved η_b and Δp/p and adds `pattern_factor`, `EI_NOx`,
`EI_CO` and `bleed_frac_required` to the output.
`cycle_adapter(record)` freezes a design's geometry and re-evaluates η and
Δp/p at whatever conditions the cycle asks for (simple off-design, with the
declared proxy Δp/p ∝ (ṁ√T₃/P₃)² — the right scaling for a fixed orifice,
not a validated correlation). `spec_from_run(path)` reads a
`runs/<n>/` directory of a sibling compressor project and returns a
combustor `DesignSpec`.

Three implementation findings changed the plan and are frozen in the
contract:

**§0 — the sibling's `fan_core.py` does not exist in this repo.** The
integration document described `hot_cycle()` at specific line numbers; no
module in the Phy-AC copy defines `hot_cycle`, `GAMMA_HOT`, `CP_HOT`,
`ETA_COMB`, `DP_COMB`, `LHV_JET_A` or `BPR`. Decision: Phy-CB implements
the seam **on its own side**, byte-faithful to the documented formulas,
with a hook ready to delegate the day the sibling grows its own. **No file
of Phy-AC or Phy-CC was modified.**

**§0.b — the sibling shadows Phy-CB's modules, silently.** Phy-AC is
installed as an *editable* package on this machine, so from any other
working directory `import physics_core` resolves to **Phy-AC's**
(NDIM = 15, axial-compressor physics). No error, no warning, a plausible
and completely wrong answer. Every entry point therefore inserts the
Phy-CB root at the head of `sys.path` and calls
`pc.assert_provenance(...)`, with `physics_core.PROJECT == "Phy-CB"` as
the sentinel. This is the **single deliberate exception to the
degrade-never-fail rule**: a mis-resolved dependency is not a degradable
case.

**§12 — L2b out of scope takes a capability with it** (§11.4, §19 limit 1).

Four more defects were found by the project's own suites and are listed
because "the tests found nothing" is usually a statement about the tests:
the C# overhang checker exempted padding slices instead of the first
**solid** slice (and therefore reported 100 % unsupported for every part);
`limitations` accumulated process state, so the record was not a pure
function of θ; `crn_L1(diverged)` did not distinguish a timeout from a
divergence; and an **empty** geometry contract came back PRINTABLE,
because there was nothing to violate.

---

## 18. How to reproduce every number in this document

```bash
cd "<Phy-CB root>"                       # ALWAYS — finding §0.b

python physics_core.py                   # the anchor + the provisional inventory
                                         #   + the EI_NOx fan at both extremes
                                         #   + the effusion and PF sweeps
python data_pipeline.py --validate       # layer 0: 31/31 checks, SHA-256 manifest
python test_phycb.py                     # verification: 355/355, 1 declared skip (33 s)
python test_phycb.py --slow              # + the C#/PicoGK smoke: 359/359
python validation/validate.py --l1       # validation campaign → validation/RESULTS.md
python validation/validate.py --strict   # what CI runs

python -m pytest cfd/tests/ -q                 # CFD ladder, fast
PHYCB_SLOW=1 python -m pytest cfd/tests/ -q    # + the expensive cases (~1–2 h)
python cfd/tests/test_ibrans_v1_v3.py --slow   # L2a V&V ladder incl. V2b/V3b
python cfd/tests/run_perf_budget.py            # the performance budget of §11.2
python cfd/tests/run_m4_kj66.py                # the M4 combustor case of §11.3
python cfd/ibrans/backend.py                   # backends available + benchmark

dotnet build CombustorDesigner.sln -c Debug    # layer 5c: 0 warnings, 0 errors
```

Read the CFD counts carefully: `pytest cfd/tests/ -q` under-reports
skipping, because `test_cfd_l15.py` registers T43–T45 through a **soft**
skip that records the reason in its own table while the test function
returns normally. The table mode tells the truth
(`python cfd/tests/test_cfd_l15.py` → 7/7 OK · 7 SKIPPED · 14 total) and
prints the reason for each skip.

---

## 19. Declared limits

1. **No quantitative metal temperature, at any fidelity level.** L2b —
   OpenFOAM with CHT and radiation, ≤10 % documented error — was the level
   that was going to produce it and is out of scope (§11.4). The L0 1-D
   thermal model gives the *trend* and the *margin* against the material
   limit; L2a gives **adiabatic** wall maps for *ranking*. Closing this
   needs a rig (level F) or reactivating L2b on a machine with
   WSL2/Docker. How much it matters is measured, not asserted: case D1
   (§5).
2. **Absolute NOx is never promised** — not at L0, not at L1, not at L2a.
   Trend and order of magnitude (×2), with the honesty label on every L2
   figure.
3. **The headline NOx correlation carries the wrong gradient against
   φ_PZ** (§8.6), and EI_NOx is an optimizer objective. Fixing it costs no
   data; it is documented rather than silently patched because it moves a
   frozen anchor.
4. **L1.5 is topological, not quantitative, and more mesh will not fix
   it** (§11.5). The BGK collision operator pinned at `TAU_FLOOR` caps the
   achievable Reynolds number; a τ-robust operator (MRT/TRT) is the open
   task.
5. **Δp/p is imposed, not predicted, at L0**: the hole areas are sized to
   meet it, so the record reproduces it by construction. The v1 criterion
   "Δp/p ±20 %" is therefore **not evaluable** today.
6. **The pattern factor returns its own calibration** (§4.3): PF = 0.35000
   in all six validation cases. PF is not validated.
7. **LBO is not validated, and with airblast the numbers are not
   credible** (§6.1): margins of ×42 and ×476 are a symptom, not a result.
8. **Zero ⚠️ coefficients have been closed against the canonical primary
   source.** Layer 0 has 17 tables; 7 carry `@status: provisional` at
   table level and 107 of 368 status-bearing rows are provisional. In the
   anchor, **43 warnings reach a reported number**, travel inside the
   record under `limitations` and are printed in every `report.html`. Five
   of them move a headline number by more than 20 %, each measured by
   perturbing the layer-0 table and re-evaluating the anchor (§5, §4.3,
   §6.1, §2.2). **WAVE1 is declared COMPLETE but NOT CLOSED** for exactly
   this reason.
9. **The class reference table contradicts itself** (§13.2, G1) and that
   single data defect explains 17 of the 48 failures. Fix the data, not
   the model.
10. **The in-loop DfAM check passes a design the as-built check rejects**
    (§16). Read `dfam_report.json` before printing.
11. **L2a's M4 is implemented but not closed**: the flame anchors in the
    real combustor (risk #1 resolved) but T₄ is +25.7 % off the balance,
    Δp/p is not comparable while the liner↔casing annulus is unresolved,
    and the pattern factor is governed by the mesh's n = 4 azimuthal mode
    — so the ranking test is not run rather than run on noise. Declaring
    it closed would require either loosening a threshold or keeping quiet
    about the "no-annulus" mode; neither is done here.
12. **V8 (Sandia Flame D) and V9 (PRECCINSTA) are pending, declared.** The
    mechanisms and the scripts exist; the tests skip with a visible
    reason and are never trivialised into passing checks.
13. **The surrogate's uncertainty gate does not open at the smoke
    budget** (§15). The gate closing is the gate *working* — but smoke
    runs are space-filling search, not optimization, and must not be
    quoted as optimized results.
14. **Thermoacoustics is outside the scope of L0–L2.** Passive mitigation
    (AM-integrable Helmholtz resonators) plus rig testing; declared.
15. **The high-fidelity affine calibration is still the identity**: it
    needs ≥2 hi-fi pairs and there are zero (no closed L2a product, no
    L2b, no rig). The campaign deliberately runs with `calibrate=False`.
16. **Materials are handbook-typical** and there is **no physical test**:
    no coupon data, no printed part measured. Safety factors and the
    banned-material rule are the mitigation, not a substitute.
17. **The assembly STL is for inspection only** — print the parts, at
    0.3–0.4 mm voxel, never at draft resolution.
