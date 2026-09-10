#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/test_cfd_l15.py — VERIFICACIÓN de WAVE5.

    T41  canal vacío: ṁ entrada = ṁ salida a 1e-3 relativo tras promedio.
    T42  placa con orificio: C_d medido ∈ (0.55, 0.75) para canto vivo.
    T43  chorro en flujo cruzado: Y/d vs 1.15·√J a ±25 % para J ∈ {4,16,36}.
    T44  splits del ancla KJ66: |medido − red 1D| < 20 % por fila.
    T45  CRN auto vs manual: mismas tendencias de EI_NOx en el barrido de
         φ_PZ; τ_pz medido dentro de 2× del de dimensionado.
    T46  (WAVE5-B) malla de OpenFOAM: **SKIP declarado**.
    T47  (WAVE5-B) Δp/p del L2b vs red 1D: **SKIP declarado**.
    T48  (WAVE5-B) T4 con reacción del L2b: **SKIP declarado**.
    T50  operadores de colisión: TRT con τ_a = τ_s y MRT con todas las tasas
         iguales reproducen BGK a precisión de máquina (identidad algebraica).

Doble modo, igual que `test_phycb.py`::

    python cfd/tests/test_cfd_l15.py            # tabla ✓/✗/–, exit 1 si falla
    python cfd/tests/test_cfd_l15.py --slow     # incluye los caros (T43–T45)
    python cfd/tests/test_cfd_l15.py --only T42
    python -m pytest cfd/tests/test_cfd_l15.py -q
    PHYCB_SLOW=1 python -m pytest cfd/tests/test_cfd_l15.py -q

Un test que no se puede correr se SALTA con motivo visible, **nunca** se
convierte en un check que pasa trivialmente. T46–T48 son exactamente eso:
L2b (OpenFOAM) está fuera de alcance por decisión de proyecto
(`docs/IMPLEMENTATION_CONTRACT.md` §12, máquina Windows sin WSL2/Docker), y
se declara aquí en vez de fingir que se comprobó algo.

Backend y operador de colisión
------------------------------
`cfd.xlb_cold` resuelve el backend solo: XLB → CuPy (GPU) → NumPy, y ETIQUETA
cuál salió. Los tests NO lo fijan: comprueban física, y el motor que la calcula
se reporta en el detalle de cada check. Para reproducir en CPU un número medido
en GPU: `set PHYCB_LBM_BACKEND=numpy` (o `PHYCB_NO_CUPY=1`).

Desde 2026-08-01 el operador de colisión también se elige
(`PHYCB_LBM_COLLISION=bgk|trt|mrt`). El defecto es **TRT** con ω_a = 1.80,
que es el que aguanta τ → ½. Para re-medir un número publicado antes de esa
fecha hay que pedir BGK explícitamente (`PHYCB_LBM_COLLISION=bgk`), que es
con el que se midió todo lo anterior.

**Los tres operadores pasan T41 y T42 con los umbrales actuales, sin tocar
ninguno.** Medido el 2026-08-01 en esta máquina, a la τ = 0.600 de los casos
canónicos:

    operador   T41a cierre de masa      T42a C_d (banda 0.55–0.75)
    bgk        3.260e-05                0.6774
    trt        3.391e-05                0.6824
    mrt        1.025e-05                0.6874   ← el mejor cierre de masa

(Una revisión anterior de este docstring afirmaba que «T41c y T42a los rompen
TRT y MRT». **Es falso y queda corregido**: T41 da 4/4 y T42 da 3/3 con los
tres. La afirmación describía un estado intermedio del desarrollo y no se
actualizó al cerrar la calibración de `TRT_LAMBDA`. Se deja escrita la
corrección en vez de borrarla en silencio.)

Donde los operadores SÍ se separan es en el suelo de τ que aguantan, que es
justo el motivo de haberlos implementado — ver `TAU_FLOOR_*` en
`cfd/xlb_cold.py`.

⚠️ El resultado **depende del backend en UN sitio declarado**: la malla del
caso real (T44/T45), que en GPU se afina hasta donde llega el presupuesto de
pared. Los casos canónicos (T41–T43) corren con la MISMA configuración en los
dos backends a propósito — un veredicto de verificación que cambia según la
máquina es un defecto, no una mejora. Las reducciones en GPU no son
bit-idénticas a las de CPU (contracciones por GEMM); medido en T41/T42, la
diferencia no llega al último dígito reportado. Nada de esto mueve un umbral.

Coste de pared medido en esta máquina
-------------------------------------
* CPU (NumPy, ~6.6 MLUPS): T41 ≈ 8 s · T42 ≈ 48 s · T43 ≈ 5 min (3 casos) ·
  T44/T45 ≈ 15–25 min con malla de 0.9 M nodos a dx = 1.23 mm.
* GPU (CuPy, RTX 3070 Ti, 182 MLUPS medidos sobre el ancla de 3 M nodos =
  5.5e-9 s/celda/paso): T41 se queda en NumPy (malla de 16 k nodos: la GPU
  pierde contra la latencia de lanzamiento) · T42 ≈ 35 s · T43 ≈ 5 min ·
  T44/T45 ≈ 25–30 min con malla de 7 M nodos a dx = 0.61 mm.

El presupuesto de pared de L1.5 se controla con `PHYCB_L15_TIMEOUT`
(segundos, por defecto 900) y **la malla de GPU se adapta a él**, midiendo la
VRAM libre real y estimando los pasos que pide el transitorio:

    PHYCB_L15_TIMEOUT   dx        nodos    celdas en el anillo del liner
    900  s              1.04 mm   1.5 M    2.4
    1800 s              0.73 mm   4.2 M    3.4
    3600 s              0.61 mm   7.0 M    4.0   ← el objetivo de WAVE5 §A.2.1
    7200 s              0.55 mm  10.0 M    4.5

En CPU la malla se queda en dx = 1.23 mm (0.92 M nodos, 2.0 celdas en el
anillo), que es el mínimo por debajo del cual el reparto de aire medido deja
de significar nada.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

# --- TRAMPA DE ENTORNO (contrato §0.b): Phy-AC está instalado como paquete
#     EDITABLE y ensombrece los módulos de Phy-CB desde cualquier otro cwd. --
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

for _s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import physics_core as pc                                        # noqa: E402
import geometry_generator as gg                                  # noqa: E402
import crn_core as crn                                           # noqa: E402
from cfd import xlb_cold as xc                                   # noqa: E402
from cfd import cold_post as cp                                  # noqa: E402
from cfd import crn_from_field as cff                            # noqa: E402

pc.assert_provenance(pc, gg, crn)

THETA_REF = np.array(pc.THETA_REF, dtype=float)

SLOW = os.environ.get("PHYCB_SLOW", "").strip().lower() not in (
    "", "0", "no", "false")

#: Motivo ÚNICO de los SKIP de WAVE5-B. Se imprime tal cual.
L2B_REASON = ("L2b (OpenFOAM) está FUERA DE ALCANCE por decisión de proyecto: "
              "docs/IMPLEMENTATION_CONTRACT.md §12 — la máquina de "
              "implementación es Windows y OpenFOAM exigiría WSL2/Docker. "
              "`cfd/openfoam_backend.py` no existe. Consecuencia declarada: "
              "Phy-CB NO predice temperatura de metal cuantitativa en ningún "
              "nivel de fidelidad.")

# ==========================================================================
# Infraestructura de checks — mismo patrón que `test_phycb.py`
# ==========================================================================
_RESULTS: list[tuple[str, str, str, str]] = []
_CURRENT_ID = ""


def check(name: str, cond, detail: str = ""):
    ok = bool(cond)
    _RESULTS.append((_CURRENT_ID, name, "ok" if ok else "fail", str(detail)))
    assert ok, f"{name}: {detail}"


def skip(name: str, reason: str, *, hard: bool = True):
    """Salta un check DEJANDO CONSTANCIA. Nunca cuenta como aprobado.

    BUG REAL CORREGIDO (2026-07-31). Esta función solo anotaba en
    `_RESULTS` y el llamante hacía `return`. La tabla del modo standalone
    decía la verdad («7/7 OK · 7 SALTADOS»), pero **bajo pytest un test que
    retorna sin excepción cuenta como PASSED**: `pytest -q` informaba
    «8 passed» de unos tests de los que la mitad ni se había ejecutado. Y
    pytest es justo lo que corre la CI.

    Ahora, bajo pytest, se lanza `pytest.skip`, que es el único mecanismo
    que pytest cuenta como salto. `hard=False` para los saltos PARCIALES
    dentro de un barrido, donde el test debe seguir midiendo los demás
    puntos (ahí el `check` global de cierre es el que decide).
    """
    _RESULTS.append((_CURRENT_ID, name, "skip", reason))
    if not hard:
        return
    try:
        import pytest as _pytest
    except Exception:
        return                       # modo standalone: basta con la tabla
    if os.environ.get("PYTEST_CURRENT_TEST"):
        _pytest.skip(f"{name}: {reason}")


def record(name: str, cond, detail: str = "") -> bool:
    """Anota un resultado SIN abortar el test.

    `check` lanza `AssertionError` (es lo que hace que pytest funcione), y
    eso corta el resto del barrido: si J=4 falla, J=16 y J=36 no llegan a
    medirse y el informe pierde justo la información que explica el fallo.
    Los barridos anotan con `record` y cierran con UN `check` global."""
    ok = bool(cond)
    _RESULTS.append((_CURRENT_ID, name, "ok" if ok else "fail", str(detail)))
    return ok


def _rel(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)


# ==========================================================================
# Ancla compartida por T44/T45 (una sola resolución del campo frío)
# ==========================================================================
_ANCHOR: dict = {}


def _anchor(log=None) -> dict:
    """Campo frío del ancla KJ66 resuelto UNA vez y memorizado.

    T44 y T45 comparten la misma medida: volver a lanzar el LBM costaría
    otros 15 min y daría exactamente lo mismo (SEED = 71, determinista)."""
    if _ANCHOR:
        return _ANCHOR
    rec = pc.evaluate(THETA_REF, pc.Fidelity.L0, use_cache=False)
    t0 = time.monotonic()
    post = cff.measure(THETA_REF, rec, log=log)
    _ANCHOR.update(rec=rec, post=post, status=cff.LAST_STATUS,
                   wall_s=time.monotonic() - t0,
                   net1d=crn.network_1d(THETA_REF, rec))
    return _ANCHOR


# ==========================================================================
# T41 — canal vacío
# ==========================================================================
def test_T41_canal_vacio():
    """T41 Canal vacío: conservación de masa del solver LBM."""
    r = xc.canonical_channel(n=18, nz=48, avg=300)
    check("T41a ṁ entrada = ṁ salida a 1e-3 relativo",
          r["rel_error"] < 1.0e-3,
          f"rel = {r['rel_error']:.3e} (in {r['mdot_in_lat']:.6f}, "
          f"out {r['mdot_out_lat']:.6f} lu)")
    check("T41b el corte central da el mismo caudal",
          _rel(r["mdot_mid_lat"], r["mdot_in_lat"]) < 1.0e-3,
          f"medio {r['mdot_mid_lat']:.6f} vs entrada {r['mdot_in_lat']:.6f}")
    #  El caudal es algo menor que el «tapón» u·A porque la condición de
    #  no-deslizamiento de media celda frena las celdas del borde: si saliera
    #  MAYOR, el bounce-back estaría creando masa.
    check("T41c el caudal no supera el tapón u·A",
          r["mdot_in_lat"] <= r["mdot_plug_lat"] * 1.001,
          f"{r['mdot_in_lat']:.4f} vs tapón {r['mdot_plug_lat']:.4f}")
    check("T41d el solver converge", r["stats"]["converged"],
          f"{r['stats']['steps']} pasos, {r['stats']['mlups']:.1f} MLUPS")


# ==========================================================================
# T42 — C_d de una placa con orificio de canto vivo
# ==========================================================================
def test_T42_cd_orificio():
    """T42 Placa con orificio: C_d medido en la banda de canto vivo."""
    r = xc.canonical_orifice()
    lo, hi = 0.55, 0.75
    check(f"T42a C_d medido ∈ ({lo}, {hi}) para canto vivo",
          lo < r["Cd"] < hi,
          f"C_d = {r['Cd']:.4f} (t/d = {r['t_over_d']:.2f}, "
          f"Re_d = {r['Re_d']:.0f}, u_lattice = {r['u_hole_lat']:.4f})")
    check("T42b la velocidad de lattice del chorro se queda por debajo de 0.1",
          r["u_hole_lat"] < xc.U_LATTICE_MAX,
          f"u_hole = {r['u_hole_lat']:.4f} (WAVE5 §D)")
    check("T42c el orificio no ahoga el conducto (A_h/A_duct < 0.15)",
          r["area_ratio"] < 0.15, f"A_h/A_duct = {r['area_ratio']:.3f}")


# ==========================================================================
# T43 — chorro en flujo cruzado
# ==========================================================================
def test_T43_chorro_en_flujo_cruzado():
    """T43 Chorro en flujo cruzado: Y/d vs 1.15·√J a ±25 %."""
    if not SLOW:
        skip("T43 penetración Y/d vs 1.15·√J",
             "caro (~5 min: tres LBM de 0.087 M nodos) — usa --slow")
        return
    res, oks = [], []
    for J in (4.0, 16.0, 36.0):
        r = xc.canonical_jicf(J=J)
        corr = r["Y_over_d_lefebvre"]
        dev = r["Y_over_d"] / max(corr, 1e-30) - 1.0
        res.append((J, r, dev))
        oks.append(record(
            f"T43 J={J:.0f}: Y/d medido dentro del ±25 % de 1.15·√J",
            abs(dev) <= 0.25,
            f"Y/d = {r['Y_over_d']:.2f} vs {corr:.2f} ({100 * dev:+.1f} %) "
            f"en la estación de semejanza x/(r·d) = {r['station_over_rd']:.3f} "
            f"⇒ x/d = {r['station_over_d']:.2f}; {r['stats']['steps']} pasos, "
            f"d = {r['d_lat']} lu (refine ×{r['refine']}), "
            f"{r['n_nodes'] / 1e6:.2f} M nodos, {r['backend']}, "
            f"{r['stats']['mlups']:.0f} MLUPS"))
    record("T43d la penetración medida crece con J",
           all(res[i][1]["Y_over_d"] < res[i + 1][1]["Y_over_d"]
               for i in range(len(res) - 1)),
           " · ".join(f"J={J:.0f}: Y/d={r['Y_over_d']:.2f}"
                      for J, r, _ in res))
    check("T43 el barrido completo J ∈ {4,16,36} entra en el ±25 %",
          all(oks),
          "desviaciones: " + " · ".join(
              f"J={J:.0f} {100 * d:+.1f} %" for J, _, d in res)
          + ". El chorro DÉBIL (J=4) es el que se sale: a d = 5 lu el LBM lo "
            "mantiene erguido ~30 % más de lo que dice la correlación. La "
            "estación se derivó A PRIORI de la trayectoria clásica "
            "(1.6·s₀^⅓ = 1.15 ⇒ s₀ = 0.371) y NO se ha movido para aprobar.")


# ==========================================================================
# T44 — splits del ancla KJ66
# ==========================================================================
def test_T44_splits_ancla():
    """T44 Splits medidos del ancla KJ66 vs la red 1D de L1."""
    if not SLOW:
        skip("T44 splits medidos vs red 1D",
             "caro (~15–25 min: LBM de 0.9 M nodos a dx = 1.23 mm en CPU, o "
             "~23 min de 7.0 M nodos a dx = 0.61 mm en GPU con "
             "PHYCB_L15_TIMEOUT=3600) — usa --slow")
        return
    a = _anchor(log=print)
    if a["post"] is None:
        skip("T44 splits medidos vs red 1D",
             f"el campo frío no resolvió: {a['status']} "
             f"(LBM: {xc.LAST_STATUS})")
        return
    post, net = a["post"], a["net1d"]
    cmpr = cp.compare_with_1d(post, net)
    oks = []
    for row in cmpr["rows"]:
        if not math.isfinite(row["dev"]):
            skip(f"T44 split {row['zone']}",
                 "la red 1D no asigna caudal a esa zona", hard=False)
            continue
        oks.append(record(
            f"T44 split {row['zone']}: |LBM − red 1D| < 20 %",
            abs(row["dev"]) < 0.20,
            f"LBM {row['lbm']:.4f} vs red 1D {row['network_1d']:.4f} "
            f"({100 * row['dev']:+.1f} %)"))
    record("T44 ṁ por camino medido [kg/s]", True,
           " · ".join(f"{k} {v:.5f}" for k, v in
                      sorted((post.get("mdot_paths_measured_kgs") or {}).items())))
    record("T44 discretización de las filas (A_voxel/A_contrato)", True,
           " · ".join(f"{r['id']} {r['A_ratio']:.2f}"
                      for r in (post.get("rows_discretization") or [])))
    #  DIAGNÓSTICO: los cuatro splits desviados no dicen POR QUÉ. Estos
    #  cuatro checks sí, y por eso se anotan ANTES del veredicto agregado
    #  (que aborta el test). Si el Δp/p del LBM se dispara, la resistencia
    #  del anillo domina y se come el Δp que deberían ver los orificios.
    mesh = post.get("cfd_mesh") or {}
    oks.append(record(
        "T44e cierre de masa entrada/salida del LBM < 1 %",
        post["mass_closure_rel"] < 0.01,
        f"{100 * post['mass_closure_rel']:.3f} % "
        f"(in {post['mdot_in_measured_kgs']:.5f} kg/s, "
        f"out {post['mdot_out_measured_kgs']:.5f} kg/s)"))
    oks.append(record(
        "T44f el caudal de entrada reproduce el del punto de operación",
        _rel(post["mdot_in_measured_kgs"], a["rec"]["mdot_air_kgs"]) < 0.05,
        f"{post['mdot_in_measured_kgs']:.5f} vs "
        f"{a['rec']['mdot_air_kgs']:.5f} kg/s"))
    nu_ratio = float(mesh.get("nu_ratio_effective") or float("nan"))
    oks.append(record(
        "T44g el Δp/p del LBM no se dispara frente al de diseño (×3)",
        post["dp_over_p_cfd"] < 3.0 * max(a["rec"]["dp_over_p"], 1e-9),
        f"Δp/p_LBM = {post['dp_over_p_cfd']:.4f} vs diseño "
        f"{a['rec']['dp_over_p']:.4f}; el anillo se resuelve con "
        f"{mesh.get('annulus_cells', float('nan')):.2f} celdas "
        f"({mesh.get('annulus_gap_mm', float('nan')):.2f} mm a dx = "
        f"{mesh.get('dx_mm', float('nan')):.3f} mm), el Ø de orificio más "
        f"pequeño con {mesh.get('d_hole_min_cells', float('nan')):.2f}, y τ "
        f"{'SIGUE suelado' if mesh.get('tau_floored') else 'ya NO está suelado'}"
        f" en {mesh.get('tau', float('nan')):.5f} frente al molecular "
        f"{mesh.get('tau_molecular', float('nan')):.6f} — viscosidad efectiva "
        f"×{nu_ratio:.0f} la real, así que el Δp del anillo lleva componente "
        "NUMÉRICA y el reparto entre caminos hereda ese sesgo"))
    oks.append(record(
        "T44h el LBM alcanzó el estacionario estadístico",
        bool(mesh.get("converged")),
        f"{mesh.get('steps')} pasos, {mesh.get('wall_s', 0.0):.0f} s de "
        f"pared, {mesh.get('mlups', 0.0):.1f} MLUPS, backend "
        f"{mesh.get('backend', '?')} ({mesh.get('precision', '?')}, "
        f"acumulador {mesh.get('acc_precision', '?')}), "
        f"{mesh.get('n_nodes', 0) / 1e6:.2f} M nodos"))
    #  DIAGNÓSTICO del camino `cool`: si la cota de seguridad de la efusión
    #  muerde, `splits.cool` NO es una medida del campo sino el recorte, y
    #  refinar la malla no lo puede arreglar. Se anota SIEMPRE, pase o falle.
    record("T44i la efusión NO está saturada en su cota de seguridad",
           not bool(mesh.get("effusion_capped")),
           f"cota mordió en {mesh.get('effusion_cap_hits', 0)} refrescos "
           f"(la ley de orificio pedía {mesh.get('effusion_demand_lat', 0.0):.3f} "
           f"lu frente a la cota {mesh.get('effusion_cap_lat', 0.0):.3f} = "
           f"{100 * xc.EFFUSION_MAX_FRAC:.0f} % del caudal de entrada). "
           "Mientras muerda, la desviación de `cool` mide el recorte, no el "
           "campo: es el MODELO de efusión (C_d·A frente al Δp local), no la "
           "resolución. NO entra en el veredicto agregado a propósito — es "
           "diagnóstico, y moverlo aquí sería taparlo.")
    record("T44j sellado del liner (celdas de gas sacrificadas)", True,
           f"{100 * float(mesh.get('seal_loss_frac', 0.0)):.1f} % del volumen "
           f"del liner (t_liner frente a dx = "
           f"{mesh.get('dx_mm', float('nan')):.3f} mm)")
    check("T44 splits medidos, cierre de masa y Δp/p del ancla",
          all(oks), "desviaciones de split: " + " · ".join(
              f"{r['zone']} {100 * r['dev']:+.1f} %" for r in cmpr["rows"]
              if math.isfinite(r["dev"]))
          + f" · Δp/p_LBM {post['dp_over_p_cfd']:.3f} vs diseño "
            f"{a['rec']['dp_over_p']:.3f}")


# ==========================================================================
# T45 — CRN auto-topología vs CRN manual
# ==========================================================================
def test_T45_crn_auto_vs_manual():
    """T45 CRN de auto-topología (L1.5) frente al de dimensionado (L1)."""
    if not SLOW:
        skip("T45 CRN auto vs manual",
             "caro (comparte el LBM del ancla de T44) — usa --slow")
        return
    a = _anchor(log=print)
    if a["post"] is None:
        skip("T45 CRN auto vs manual",
             f"el campo frío no resolvió: {a['status']}")
        return
    rec, post = a["rec"], a["post"]

    # --- τ_pz medido vs el de dimensionado (MISMO convenio de Lefebvre) ---
    #  `rec["tau_pz_ms"]` es ρ₃·V_pz/ṁ_aire,PZ (convenio FRÍO de Lefebvre,
    #  Ciencia §2.4). El homólogo medido es `tau_zones_measured_cold_ms`,
    #  que usa exactamente eso con el V y el ṁ MEDIDOS. Compararlo contra el
    #  corregido por densidad sería mezclar convenios y dar un factor ~5.5
    #  de regalo.
    tau_pz_meas = float((post["tau_zones_measured_cold_ms"] or {}).get("pz", 0.0))
    tau_pz_size = float(rec["tau_pz_ms"])
    ratio = tau_pz_meas / max(tau_pz_size, 1e-30)
    V_pz_m = float((post["V_zones_measured_m3"] or {}).get("pz", 0.0))
    oks = [record("T45a τ_pz medido dentro de 2× del de dimensionado",
                  0.5 <= ratio <= 2.0,
                  f"τ_pz medido {tau_pz_meas:.3f} ms vs dimensionado "
                  f"{tau_pz_size:.3f} ms (×{ratio:.2f}; convenio FRÍO "
                  f"ρ₃·V/ṁ_aire,PZ en ambos). V_pz medido "
                  f"{1e6 * V_pz_m:.2f} cm³ vs "
                  f"{1e6 * float(rec['V_pz_m3']):.2f} cm³ de dimensionado "
                  f"(×{V_pz_m / max(float(rec['V_pz_m3']), 1e-30):.2f})")]

    rec2, diag = cff.build_measured_record(THETA_REF, rec, post)
    oks.append(record(
        "T45b el ajuste de C_d reproduce los caudales medidos",
        diag["l15_match_dev_max"] < 0.10,
        f"desviación máxima por camino {100 * diag['l15_match_dev_max']:.2f} "
        f"% en {diag['l15_match_iters']} iteraciones "
        f"(clamp {'SÍ' if diag['l15_match_clamped'] else 'no'}); factores "
        f"{ {k: round(v, 3) for k, v in diag['l15_match_factors'].items()} }"))

    if not getattr(crn, "CANTERA_AVAILABLE", False):
        skip("T45c tendencia de EI_NOx auto vs manual",
             "Cantera no está disponible: el CRN no se puede resolver",
             hard=False)
        check("T45 auto-topología: τ_pz y ajuste de caudales", all(oks), "")
        return

    # --- barrido de φ_PZ con AMBAS topologías ------------------------------
    #  Se barre el caudal de combustible (φ ∝ ṁ_f a reparto de aire fijo). El
    #  campo frío NO cambia con el combustible, así que reutilizar la misma
    #  medida para los tres puntos es correcto y se declara.
    scales = (0.80, 1.00, 1.25)
    man, aut, phis = [], [], []
    for s in scales:
        r_man = dict(rec)
        r_man["mdot_fuel_kgs"] = float(rec["mdot_fuel_kgs"]) * s
        r_aut = dict(rec2)
        r_aut["mdot_fuel_kgs"] = float(rec["mdot_fuel_kgs"]) * s
        sm = crn.crn_solve(THETA_REF, r_man, 180.0)
        sa = crn.crn_solve(THETA_REF, r_aut, 180.0)
        if not sm or not sa:
            skip("T45c tendencia de EI_NOx auto vs manual",
                 f"el CRN no convergió en el punto ×{s:.2f} "
                 f"({crn.LAST_STATUS})", hard=False)
            check("T45 auto-topología: τ_pz y ajuste de caudales",
                  all(oks), "")
            return
        man.append(float(sm["EI_NOx_crn"]))
        aut.append(float(sa["EI_NOx_crn"]))
        phis.append((float(sm["phi_pz_crn"]), float(sa["phi_pz_crn"])))
    d_man = [man[i + 1] - man[i] for i in range(len(man) - 1)]
    d_aut = [aut[i + 1] - aut[i] for i in range(len(aut) - 1)]
    same = all((x > 0) == (y > 0) for x, y in zip(d_man, d_aut))
    oks.append(record(
        "T45c EI_NOx sigue la MISMA tendencia con φ_PZ en ambas topologías",
        same,
        "φ_pz manual/auto " + " · ".join(f"{m:.3f}/{a_:.3f}"
                                         for m, a_ in phis)
        + " · EI_NOx manual " + " ".join(f"{v:.3f}" for v in man)
        + " · auto " + " ".join(f"{v:.3f}" for v in aut)))
    oks.append(record(
        "T45d la auto-topología no cambia el orden de magnitud del NOx",
        all(0.2 <= a_ / max(m_, 1e-12) <= 5.0 for a_, m_ in zip(aut, man)),
        "ratio auto/manual = " + " ".join(
            f"{a_ / max(m_, 1e-12):.2f}" for a_, m_ in zip(aut, man))))
    check("T45 auto-topología: τ_pz, ajuste de caudales y tendencia de NOx",
          all(oks),
          "el barrido de φ_PZ NO coincide en tendencia porque el reparto de "
          "aire MEDIDO (T44) manda a la PZ mucho menos aire que la red 1D — "
          "el modelo de efusión del LBM se lleva mucho más que su cuota (ver "
          "T44i: mientras la cota de seguridad muerda, `cool` mide el "
          "recorte, no el campo) —, así que el auto-CRN arranca en la PZ ya "
          "rico y a ×1.25 se va a una riqueza donde el NOx CAE. Es una "
          "consecuencia AGUAS ABAJO del fallo de T44, no de la maquinaria de "
          "auto-topología: T45a (τ_pz) y T45b (ajuste de C_d) pasan. NOTA "
          "2026-07-31: el desacuerdo de conductancia C_d·A entre el LBM y "
          "`crn_core` (×1.536) que se citaba aquí YA NO EXISTE — los dos "
          "consumen `physics_core` §12.b y el test T49 lo vigila. Lo que "
          "queda es Δp local, τ suelado en TAU_FLOOR y la cota de efusión.")


# ==========================================================================
# T46–T48 — WAVE5-B (OpenFOAM): SKIP DECLARADO
# ==========================================================================
def test_T46_openfoam_malla():
    """T46 Malla de OpenFOAM (checkMesh + 50 iteraciones): SKIP declarado."""
    skip("T46 caso de OpenFOAM: checkMesh limpio y 50 iteraciones", L2B_REASON)


def test_T47_openfoam_dp():
    """T47 Δp/p del L2b vs red 1D ±20 % en frío: SKIP declarado."""
    skip("T47 Δp/p del L2b vs red 1D (±20 %, caso ancla frío)", L2B_REASON)


def test_T48_openfoam_t4():
    """T48 T4 media del L2b con reacción ±5 %: SKIP declarado."""
    skip("T48 T4 media del L2b con reacción (±5 % del balance L0)",
         L2B_REASON)
    #  Lo que se pierde al retirar L2b no es solo este check: es la ÚNICA vía
    #  que Ciencia §11.3 dejaba para la temperatura de metal cuantitativa.
    skip("T48b temperatura de metal cuantitativa (CHT + radiación)",
         "capacidad RETIRADA del sistema, no trasladada a otro nivel: L2a da "
         "mapas de pared ADIABÁTICA para ranking, y el térmico 1D de L0 da "
         "tendencia y margen. Cerrar el hueco exige banco de ensayo (WAVE7 "
         "nivel F) o reactivar L2b en una máquina con WSL2/Docker.")


# ==========================================================================
# T50 — operadores de colisión: identidades algebraicas exactas
#
#  El ID es T50 y no T49: `test_phycb.py` ya usa T49a–T49i para el contrato
#  entre capas de la efusión (contrato §11). Los IDs son un espacio de
#  nombres del PROYECTO, no de cada archivo — dos tests distintos con el
#  mismo ID hacen ilegible cualquier informe que los cruce.
# ==========================================================================
def test_T50_operadores_de_colision():
    """T50 TRT y MRT degenerados reproducen BGK a precisión de máquina.

    NO es un test de física: es la prueba de que la descomposición
    simétrica/antisimétrica de TRT y la matriz de momentos de MRT están bien
    escritas. Las dos identidades son EXACTAS en álgebra:

    * TRT con τ_a = τ_s ≡ BGK (sale de `f⁺ + f⁻ = f_i`);
    * MRT con todas las tasas iguales a 1/τ ≡ BGK (porque `M⁻¹(1/τ)I M`
      es `(1/τ)I`).

    Se comparan campos `f` COMPLETOS tras 120 pasos en FP64, con y sin LES
    (el camino del LES es distinto: las tasas pasan a ser por celda).
    Barato: dos cajas de 5 k nodos, ~10 s en CPU."""
    r = xc.collision_equivalence()
    for row in r["rows"]:
        les = "con LES" if row["les"] else "sin LES"
        record(f"T50 {row['case']} ≡ BGK {les}", row["max_rel"] < 1e-12,
               f"|Δf|_max = {row['max_abs']:.3e}, relativo "
               f"{row['max_rel']:.3e}, bit a bit = {row['bit_exact']}")
    check("T50 los tres operadores coinciden en su límite común",
          r["ok"],
          " · ".join(f"{x['case']}{'/LES' if x['les'] else ''} "
                     f"{x['max_rel']:.1e}" for x in r["rows"])
          + ". TRT sale BIT A BIT idéntico a BGK (la descomposición por pares "
            "no cambia el orden de las sumas); MRT queda en ~4e-15 relativo "
            "porque `M⁻¹SM` se ensambla numéricamente.")


ALL = [test_T41_canal_vacio, test_T42_cd_orificio,
       test_T43_chorro_en_flujo_cruzado, test_T44_splits_ancla,
       test_T45_crn_auto_vs_manual, test_T46_openfoam_malla,
       test_T47_openfoam_dp, test_T48_openfoam_t4,
       test_T50_operadores_de_colision]


def _id_de(fn) -> str:
    return fn.__name__.split("_")[1].upper()


def _uso() -> str:
    return __doc__ or ""


def main(argv=None) -> int:
    """Ejecuta la suite de WAVE5 en modo standalone."""
    global _CURRENT_ID, SLOW
    argv = list(sys.argv[1:] if argv is None else argv)
    solo: set[str] = set()
    for i, a in enumerate(argv):
        if a in ("-h", "--help"):
            print(_uso())
            return 0
        if a == "--slow":
            SLOW = True
        elif a == "--only" and i + 1 < len(argv):
            solo = {s.strip().upper() for s in argv[i + 1].split(",")
                    if s.strip()}
        elif a.startswith("--only="):
            solo = {s.strip().upper() for s in a.split("=", 1)[1].split(",")
                    if s.strip()}

    print("=" * 74)
    print("QUASAR Phy-CB · VERIFICACIÓN de WAVE5 (L1.5 aero-frío) — T41…T48")
    print(f"  physics_core : {pc.__file__}")
    print(f"  XLB/JAX      : {'sí' if xc.XLB_AVAILABLE else 'NO'}")
    _gi = xc.gpu_info()
    print(f"  CuPy/GPU     : "
          + (f"{_gi['name']} · {_gi['vram_free_GiB']:.2f}/"
             f"{_gi['vram_total_GiB']:.2f} GiB libres · compute "
             f"{_gi['compute_capability']} · cupy {_gi['cupy_version']}"
             if _gi else f"NO ({xc.cupy_reason()})"))
    print(f"  backend LBM  : {xc.resolve_backend('auto')[0]} "
          f"(PHYCB_LBM_BACKEND para forzar otro)")
    print(f"  Cantera      : {'sí' if crn.CANTERA_AVAILABLE else 'NO'}")
    print(f"  OpenVDB      : {'sí' if xc.openvdb_module() else 'NO → '
                              're-voxelización analítica del contrato'}")
    print(f"  tests caros  : {'SÍ (--slow)' if SLOW else 'no (usa --slow)'}")
    print("=" * 74)

    t_ini = time.time()
    fallos: list[tuple[str, str]] = []
    for fn in ALL:
        tid = _id_de(fn)
        if solo and tid not in solo:
            continue
        _CURRENT_ID = tid
        t0 = time.time()
        try:
            fn()
            estado = "ok"
        except AssertionError as exc:
            estado = "fail"
            fallos.append((tid, str(exc).splitlines()[0]))
        except Exception as exc:
            estado = "error"
            _RESULTS.append((tid, f"{tid} · EXCEPCIÓN NO ESPERADA", "fail",
                             f"{type(exc).__name__}: {exc}"))
            fallos.append((tid, f"{type(exc).__name__}: {exc}"))
        print(f"— {tid:>4s} {fn.__doc__.splitlines()[0][:56]:<58s}"
              f"{time.time() - t0:7.2f} s  "
              f"{'✓' if estado == 'ok' else '✗'}")
    _CURRENT_ID = ""

    print("\n" + "=" * 74)
    print("TABLA DE CHECKS")
    print("=" * 74)
    n_ok = n_fail = n_skip = 0
    for tid, nombre, estado, detalle in _RESULTS:
        marca = {"ok": "✓", "fail": "✗", "skip": "–"}[estado]
        n_ok += estado == "ok"
        n_fail += estado == "fail"
        n_skip += estado == "skip"
        linea = f"  {marca} [{tid:>4s}] {nombre}"
        if detalle and estado != "ok":
            linea += f"\n        → {detalle}"
        elif detalle:
            linea += f"  [{detalle}]"
        print(linea)

    total = n_ok + n_fail + n_skip
    print("=" * 74)
    print(f"{n_ok}/{n_ok + n_fail} checks OK   ·   {n_skip} SALTADOS   ·   "
          f"{total} en total   ·   {time.time() - t_ini:.1f} s")
    if n_skip:
        print("\nSALTADOS (ninguno cuenta como aprobado):")
        for tid, nombre, estado, detalle in _RESULTS:
            if estado == "skip":
                print(f"  – [{tid:>4s}] {nombre}:\n        {detalle}")
    if fallos:
        print(f"\n{len(fallos)} TEST(S) FALLARON:")
        for tid, msg in fallos:
            print(f"  ✗ {tid}: {msg[:220]}")
        return 1
    print("\nVerificación de WAVE5 completa ✓  (los caros con `--slow`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
