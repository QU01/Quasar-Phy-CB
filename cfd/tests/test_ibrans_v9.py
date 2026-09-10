#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/test_ibrans_v9.py
===========================================
Escalera de **verificación del hito M4** de WAVE6 (combustor completo +
producto). IDs del espacio de nombres del PROYECTO (contrato §11;
comprobados libres en `test_phycb.py` y en `cfd/tests/`):

    M4a  caso frío del ancla: conductancias del contrato,
         momento de descarga, masa y swirl ................ ver el
                                                            docstring
    M4b  ENCENDIDO del ancla KJ66 (el riesgo #1 en el
         combustor real) + producto completo ............. ancla y
                                                           T3 < T4 < T_ad
                                                           [lento]
    M4c  RANKING: dilución 6 vs 10 orificios → el PF
         relativo ordena igual que la correlación L0 ..... orden [lento]
    V9a  PRECCINSTA con pérdida de calor en paredes ...... PENDIENTE
                                                           (declarado)

Qué mide cada uno (y qué NO)
----------------------------
* **M4a** es el criterio de cierre de la tarea 4.1 del plan
  (`M4_PLAN.md`): la cadena contrato → SDF completo → transpiración
  C_d·A → campo, con los splits MEDIDOS por balance de volumen de control
  axial (el flujo radial local en la PZ con swirl no es medible fila a
  fila: la recirculación lo contamina — medido). En el ancla KJ66 el
  anillo liner↔carcasa NO cabe en la malla (holgura ~3 mm) y el caso corre
  en el modo DECLARADO «sin anillo»: todas las filas transpiradas, así que
  M4a verifica que la transpiración discreta ENTREGA las conductancias del
  contrato, no una redistribución por Δp.
* **M4b** ejercita `solve()` completo (contrato → caso → encendido →
  producto §4.4). El criterio de anclaje es el del plan
  (`sp.anchored()`); T4 se comprueba FÍSICAMENTE acotada (T3 < T4 < T_ad
  del modelo). El contraste fino T4 vs balance ±5 % es del caso ancla
  oficial (`run_m4_kj66.py`, con sus números en `ESTADO.md`).
* **M4c** es el entregable de PRECISIÓN de L2a (criterio de aceptación #3
  de la wave): ranking, no valores absolutos.
* **V9a** (PRECCINSTA) tiene su script manual (`run_v9_preccinsta.py`) y
  queda PENDIENTE DECLARADO hasta que se corra y sus números estén en
  `ESTADO.md` — regla de la casa: un caso que no cabe en CI se salta con
  motivo visible, jamás se convierte en check trivial.

Doble modo::

    python cfd/tests/test_ibrans_v9.py            # tabla ✓/✗/–
    PHYCB_SLOW=1 python cfd/tests/test_ibrans_v9.py   # + M4b/M4c
    python -m pytest cfd/tests/test_ibrans_v9.py -q
"""

from __future__ import annotations

import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np                                            # noqa: E402

import physics_core as pc                                     # noqa: E402
import geometry_generator as gg                               # noqa: E402

pc.assert_provenance(pc, gg)

import cfd.ibrans as ib                                       # noqa: E402
from cfd.ibrans.case import (CaseConfig, build_case,          # noqa: E402
                             measure_splits, run_case,
                             swirl_number_measured)

SLOW = os.environ.get("PHYCB_SLOW", "") not in ("", "0")

# Umbrales (de M4_PLAN.md §2 y WAVE6 §4 — no se aflojan)
#: cierre de masa (plano 4 vs ṁ del contrato). ⚠️ El caso a dx = 2 mm no
#: converge más allá de esto en el presupuesto de CI: medido +9.9 % a 6000
#: iteraciones. NO se afloja para que pase — si sube, es una regresión.
TOL_CLOSURE = 0.12
TOL_SN = 0.20             # SN medido vs swirler.SN
#: banda admisible de v_inyeccion / U_jet en las filas transpiradas. No es
#: ±20 %: la malla sólo puede aproximar el área de descarga C_d·A con un
#: número ENTERO de celdas, y con dx = 2 mm eso deja un escalón grueso.
#: Lo que vigila es el orden de magnitud del MOMENTO, que es lo que se
#: perdía (factor ~5) antes de la corrección.
TOL_UJET = (0.5, 2.0)
# Presupuesto del caso frío de CI (medido 2026-08-02: ~5 min en CPU)
M4A_DX_MM = 2.0
M4A_ITERS = 4000

_CACHE: dict = {}


def _anchor_cold():
    """Caso frío del ancla, corrido UNA vez y compartido por los asserts."""
    if "cold" in _CACHE:
        return _CACHE["cold"]
    rec = pc.evaluate(pc.THETA_REF)
    c = gg.build_contract(pc.THETA_REF, rec)
    cfg = CaseConfig(dx_mm=M4A_DX_MM, reactive=False, ignition=False,
                     max_iter=M4A_ITERS, timeout_s=900.0)
    s = build_case(c, cfg)
    info = run_case(s)
    _CACHE["cold"] = (s, info, c, rec)
    return _CACHE["cold"]


# ==========================================================================
# M4a · caso frío: SDF completo + splits por C_d·A
# ==========================================================================
def test_m4a_cold_splits():
    """M4a · La cadena θ → contrato → SDF completo → caso frío monta el
    combustor y respeta las conductancias del contrato.

    QUÉ VERIFICA Y QUÉ NO (importa, porque el nombre engaña)
    --------------------------------------------------------
    En el ancla KJ66 el anillo liner↔carcasa no cabe a dx = 2 mm (holgura
    ~3 mm) y el caso corre en el modo DECLARADO «sin anillo»: cada fila
    entra como fuente volumétrica con SU caudal C_d·A. En ese modo el
    reparto de aire es un **input**, no una predicción — sólo lo sería con
    el anillo resuelto, donde los caminos compiten por una misma Δp. Así
    que este test comprueba lo que sí es comprobable y significativo:

    1. las declaraciones del SDF (pared inflada, regla d/dx, sin anillo);
    2. **T49**: la conductancia de efusión es EXACTAMENTE la del contrato;
    3. la fuente inyecta EXACTAMENTE Σ C_d·A_i (conservación por
       construcción, la razón de usar fuente volumétrica y no una BC de
       velocidad — que entregaba un 17 % menos);
    4. el MOMENTO de cada fila es el de descarga del orificio (guardián de
       la regresión que dejaba Δp/p a la mitad y Y/d ≈ 0);
    5. el cierre de masa global y el número de swirl.

    El reparto medido fila a fila NO se contrasta contra ±20 %: el balance
    de volumen de control axial deja de ser limpio en cuanto los chorros
    tienen momento y reorganizan el flujo (medido: ±30 % con el caudal
    inyectado exacto). Ese contraste pertenece al caso con anillo resuelto
    (dx ≤ 1 mm), y está declarado como pendiente en `ESTADO.md`.
    """
    s, info, c, rec = _anchor_cold()
    meta = s.m4["meta"]
    # (1) declaraciones del SDF
    assert meta["wall_inflated"], "la pared sub-malla debe inflarse"
    assert meta["t_liner_mm"] >= 2.0 * M4A_DX_MM * 0.999
    assert "rows" in meta and len(meta["rows"]) >= 2
    # (2) espejo T49: la conductancia de efusión del caso ES la del contrato
    eff = c["effusion"]
    cda = s.m4["paths"]["effusion"]["CdA_m2"]
    assert abs(cda - float(eff["Cd"]) * float(eff["open_area_mm2"]) * 1e-6) \
        < 1e-12, "la conductancia de efusion no es la del contrato (T49)"
    assert info["status"] in ("max_iter", "converged", "t_end"), info
    # (3) la fuente inyecta EXACTAMENTE lo que el contrato manda
    trs = s.m4["models"]["transpiration"]
    assert trs is not None, "el caso ancla debe llevar transpiracion"
    inj = sum(r["mdot_kgs"] for r in trs.rows)
    tgt = sum(p["mdot_kgs"] for p in s.m4["paths"].values()
              if p["kind"] == "transpired")
    assert abs(inj / max(tgt, 1e-12) - 1.0) < 1e-9, (inj, tgt)
    # (5) cierre de masa global y swirl
    sm = measure_splits(s)
    assert abs(sm["mass_closure_rel"]) < TOL_CLOSURE, sm["mass_closure_rel"]
    sn = swirl_number_measured(s)
    assert abs(sn["SN_measured"] / max(sn["SN_target"], 1e-9) - 1.0) \
        < TOL_SN, sn
    # (4) MOMENTO: cada fila inyecta a la velocidad de DESCARGA del
    # orificio, U_jet = ṁ/(ρ·C_d·A_geom). Sin esto el aire entraba difuso
    # a ~1/5 de U_jet y no había ni pérdida de orificio ni penetración
    # (medido el 2026-08-03: Δp/p a la mitad, Y/d ≈ 0.003).
    for r in trs.rows:
        if r["id"] == "effusion":
            continue                     # la efusión es lenta a propósito
        row = next((p["row"] for p in s.m4["paths"].values()
                    if p.get("row") and p["row"]["id"] == r["id"]), None)
        assert row is not None, r
        a_disc = row["n"] * math.pi * (0.5 * row["d_m"]) ** 2 * row["Cd"]
        u_ref = r["mdot_kgs"] / (s.m4["rho3_kgm3"] * a_disc)
        assert TOL_UJET[0] < r["u_jet_ms"] / u_ref < TOL_UJET[1], (r, u_ref)
        assert r["cells"] >= 1, r


# ==========================================================================
# M4b · encendido del ancla + producto completo (solve end-to-end) [lento]
# ==========================================================================
def test_m4b_anchor_ignition():
    """M4b · `solve()` end-to-end sobre el ancla: el caso reactivo
    ENCIENDE (criterio `anchored()` del plan — el riesgo #1), y el
    producto trae las claves del §4.4 con T4 físicamente acotada
    (T3 < T4 < 1.05·T_ad). El contraste fino T4 ±5 % vs balance es del
    script oficial `run_m4_kj66.py` (números en ESTADO.md)."""
    if not SLOW:
        _skip("caso reactivo (decenas de min) — PHYCB_SLOW=1")
        return
    rec = pc.evaluate(pc.THETA_REF)
    sol = ib.solve(pc.THETA_REF, rec, dx_mm=2.0, timeout_s=2400,
                   max_iter=9000)
    assert sol is not None, (ib.DEGRADE_TAG, ib.LAST_STATUS)
    assert ib.LAST_STATUS["status"] == "product", \
        ("no anclo (mixed-is-burned no vale para M4b)", ib.LAST_STATUS)
    for k in ("PF_cfd", "T4_cfd_K", "dp_over_p_cfd", "splits_measured",
              "cfd_mesh"):
        assert k in sol, (k, sorted(sol))
    t3 = float(rec["T3_K"])
    t_ad = float(rec.get("T_ad_st_K", 2600.0))
    assert t3 < sol["T4_cfd_K"] < 1.05 * t_ad, (sol["T4_cfd_K"], t3, t_ad)
    assert sol["PF_cfd"] > 0.0, sol["PF_cfd"]
    # T4 vs balance: ESTE es el contraste oficial de la wave (±5 %)
    assert abs(sol["T4_cfd_K"] / float(rec["T04_K"]) - 1.0) < 0.05, \
        (sol["T4_cfd_K"], rec["T04_K"])
    # Δp/p vs red 1D (±20 %) SÓLO es exigible si el anillo está resuelto:
    # sin él la pérdida de orificio no ocurre dentro del dominio y el
    # número no es comparable (lo declara `post.dp_over_p`). Exigirlo aquí
    # sería pedirle al caso algo que su discretización no puede dar.
    if bool(ib.LAST_STATUS.get("annulus_resolved")):
        assert abs(sol["dp_over_p_cfd"] / float(rec["dp_over_p"]) - 1.0) \
            < 0.20, (sol["dp_over_p_cfd"], rec["dp_over_p"])


# ==========================================================================
# M4c · ranking 6 vs 10 orificios de dilución [lento]
# ==========================================================================
def test_m4c_ranking():
    """M4c · El entregable de PRECISIÓN de L2a (aceptación #3 de la wave):
    dos variantes del ancla (n_holes_dz = 6 y 10) → el PF relativo del CFD
    ordena IGUAL que la correlación de L0. Ranking, no absolutos."""
    if not SLOW:
        _skip("dos casos reactivos (≈1 h) — PHYCB_SLOW=1")
        return
    pf_cfd, pf_l0 = {}, {}
    for n_dz in (6.0, 10.0):
        th = np.array(pc.THETA_REF, dtype=float)
        th[7] = n_dz
        rec = pc.evaluate(th, use_cache=False)
        pf_l0[n_dz] = float(rec["PF"])
        sol = ib.solve(th, rec, dx_mm=2.0, timeout_s=2400, max_iter=9000)
        assert sol is not None and "PF_cfd" in sol, \
            (n_dz, ib.DEGRADE_TAG, ib.LAST_STATUS)
        pf_cfd[n_dz] = float(sol["PF_cfd"])
    orden_l0 = pf_l0[6.0] > pf_l0[10.0]
    orden_cfd = pf_cfd[6.0] > pf_cfd[10.0]
    assert orden_l0 == orden_cfd, dict(PF_L0=pf_l0, PF_cfd=pf_cfd)


# ==========================================================================
# V9 · PRECCINSTA — PENDIENTE DECLARADO (script manual escrito)
# ==========================================================================
def test_v9a_preccinsta_pending():
    """V9a · PRECCINSTA con pérdida de calor en paredes: **PENDIENTE,
    DECLARADO** (regla de la casa). El script manual está escrito
    (`run_v9_preccinsta.py`: quemador DLR con `T_wall_ib_K` isoterma,
    metano 2S_CH4_BFER, expectativa CUALITATIVA forma M vs V); corre horas
    en GPU y sus números deben anotarse en `ESTADO.md` antes de que esto
    pase a verde. Si el swirl sale muy mal con k-ε+curvatura, el plan B
    declarado es SST (WAVE6 §7) — no inventar nada exótico."""
    _skip("V9 PRECCINSTA pendiente: correr run_v9_preccinsta.py (horas, "
          "GPU) y anotar los numeros en ESTADO.md; no corrio en la "
          "campana 2026-08-02")


# ==========================================================================
# Doble modo (tabla propia + pytest)
# ==========================================================================
class _Skip(Exception):
    pass


def _skip(reason: str):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise _Skip(reason)


_TESTS = [
    ("M4a", "caso frio: conductancias, momento y masa",
     test_m4a_cold_splits),
    ("M4b", "ENCENDIDO del ancla + producto [lento]",
     test_m4b_anchor_ignition),
    ("M4c", "ranking dilucion 6 vs 10 [lento]", test_m4c_ranking),
    ("V9a", "PRECCINSTA [PENDIENTE declarado]",
     test_v9a_preccinsta_pending),
]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only = None
    for a in argv:
        if a.startswith("--only"):
            only = set(a.split("=", 1)[-1].split(","))
    fails = 0
    print(f"\nEscalera V&V del hito M4 (WAVE6) · slow={SLOW}\n" + "-" * 72)
    for tid, name, fn in _TESTS:
        if only and tid not in only:
            continue
        t0 = time.perf_counter()
        try:
            fn()
            mark, note = "OK  ", ""
        except _Skip as exc:
            mark, note = "SKIP", f"  ({exc})"
        except BaseException as exc:
            if type(exc).__name__ == "Skipped":
                mark, note = "SKIP", f"  ({exc})"
            else:
                mark, note = "FAIL", f"  ({type(exc).__name__}: {exc})"
                fails += 1
        el = time.perf_counter() - t0
        print(f"  [{mark}] {tid:5s} {name:44s} {el:7.1f} s{note}")
    print("-" * 72)
    print(f"  {'TODO EN VERDE' if not fails else str(fails) + ' FALLO(S)'}\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
