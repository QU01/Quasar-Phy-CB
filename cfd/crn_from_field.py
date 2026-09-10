#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/crn_from_field.py — auto-topología del CRN desde el campo
frío (L1.5). **El entregable estrella de WAVE5-A.**

Idea (Ciencia §9.4, WAVE5 §A.2.4): el CRN de L1 monta sus reactores con los
volúmenes del DIMENSIONADO y los caudales de la red 1D. Aquí se sustituyen
ambos por lo MEDIDO en el campo tridimensional:

    contrato → xlb_cold.solve_cold → cold_post.post_process
             → V_pz/V_sz/V_dz y ṁ por camino MEDIDOS
             → se re-calibra la red 1D de `crn_core` para que reproduzca esos
               caudales (bucle ACOTADO sobre los C_d de cada camino)
             → `crn_core.crn_solve` con esos números
             → record de L1.5

Punto de entrada — FIRMA EXACTA que `physics_core._try_higher` espera::

    topology_from_field(theta, rec) -> dict | None

Devolver `None` hace que `physics_core` degrade con
`crn_L15(xlb_unavailable→…)`. El MOTIVO real queda en `LAST_STATUS`, igual
que hace `crn_core.LAST_STATUS` con L1.

Desviación declarada del contrato §3
------------------------------------
La tabla del contrato §3 etiqueta el éxito de L1.5 como `crn_L15(xlb_cold)`,
pero `physics_core._try_higher` escribe `rec["source"] = "crn_L15"` DESPUÉS
del `rec.update(sol)` — y este módulo no toca `physics_core`. La etiqueta
larga se conserva en `l15_source` y en `crn_topology`, que sí sobreviven.

Honestidad de la degradación
----------------------------
Si el campo frío sale pero **el CRN no se puede reconstruir** (sin Cantera,
timeout, divergencia) se devuelve `None`: reclamar `crn_L15` sin haber
resuelto ningún CRN sería mentir. Las medidas aerodinámicas no se pierden —
quedan en `LAST_POST` y `LAST_FIELD_META` para quien las quiera (CLI,
reporte), y `crn_from_field.measure(theta, rec)` las expone como API pública.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import physics_core as pc                                        # noqa: E402
import geometry_generator as gg                                  # noqa: E402
import crn_core as crn                                           # noqa: E402
from cfd import xlb_cold as xc                                   # noqa: E402
from cfd import cold_post as cp                                  # noqa: E402

pc.assert_provenance(pc, gg, crn)

SCHEMA = "phycb-record-1"

# ==========================================================================
# Tunables
# ==========================================================================
#: Iteraciones MÁXIMAS del ajuste de C_d que hace que la red 1D reproduzca
#: los caudales medidos. Bucle ACOTADO (contrato §10.5).
N_MATCH_ITERS = 8
MATCH_TOL = 0.02                 # desviación relativa objetivo por camino
MATCH_RELAX = 0.80               # sub-relajación del ajuste (exponente)
MATCH_CD_CLAMP = (0.10, 6.0)     # factor acumulado admisible sobre C_d·A

#: Presupuesto de pared del campo frío dentro de `topology_from_field`.
L15_TIMEOUT_S = float(os.environ.get("PHYCB_L15_TIMEOUT", "900"))
#: Presupuesto del CRN posterior (independiente del de L1).
L15_CRN_TIMEOUT_S = float(os.environ.get("PHYCB_L15_CRN_TIMEOUT", "180"))

#: Estado y últimos resultados (patrón `crn_core.LAST_STATUS`).
LAST_STATUS = "not_run"
LAST_POST: dict | None = None
LAST_FIELD_META: dict | None = None
#: Campo COMPLETO de la última medida (arrays incluidos). Lo consume
#: `visualization.generate_cfd` para dibujar los cortes; se guarda
#: aparte de `LAST_FIELD_META` porque pesa MB y no debe viajar en
#: ningún artefacto JSON.
LAST_FIELD: dict | None = None


def _f(x, d=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(d)
    except Exception:
        return float(d)


# ==========================================================================
# 1. Medida: contrato → campo frío → post-proceso
# ==========================================================================
def measure(theta, rec: dict, *, dx_mm=None, max_nodes=None,
            steps=xc.LBM_STEPS_MAX, timeout_s=None, vdb_path=None,
            backend="auto", collision=None, log=None) -> dict | None:
    """θ + record → medidas aerodinámicas de L1.5 (`cold_post.post_process`).

    API pública para la CLI y el reporte: da los splits medidos, la
    penetración y los volúmenes de zona **sin** exigir Cantera.

    `max_nodes=None` (defecto) deja que `xlb_cold.solve_cold` elija: el
    presupuesto de CPU si el backend sale NumPy, y lo que quepa en la VRAM
    libre MEDIDA si sale CuPy. Pasar un número fuerza el techo.

    `collision=None` deja mandar a `xlb_cold.resolve_collision`
    (`PHYCB_LBM_COLLISION` o el defecto del módulo); pasar `"bgk"` re-mide
    con el operador con el que se publicaron los números anteriores."""
    global LAST_STATUS, LAST_POST, LAST_FIELD_META, LAST_FIELD
    try:
        contract = gg.build_contract(theta, rec)
    except Exception as exc:
        LAST_STATUS = f"contract_error:{type(exc).__name__}"
        return None
    field = xc.solve_cold(contract, dx_mm=dx_mm, max_nodes=max_nodes,
                          steps=steps, vdb_path=vdb_path, backend=backend,
                          collision=collision,
                          timeout_s=L15_TIMEOUT_S if timeout_s is None
                          else timeout_s, log=log)
    LAST_FIELD = field
    if field is None:
        LAST_STATUS = f"cold_failed:{xc.LAST_STATUS}"
        return None
    post = cp.post_process(field, contract, rec)
    post["contract_schema"] = contract.get("schema")
    LAST_POST = post
    LAST_FIELD_META = dict(field.get("cfd_mesh") or {})
    LAST_FIELD_META["lbm_status"] = xc.LAST_STATUS
    if not post.get("ok"):
        LAST_STATUS = "post_failed"
        return None
    LAST_STATUS = "measured"
    return post


# ==========================================================================
# 2. Inyección de lo medido en el record que consume `crn_core`
# ==========================================================================
def _target_path_fracs(post: dict, net0: dict) -> dict:
    """Fracciones objetivo por camino de `crn_core.PATH_NAMES`.

    El LBM mide cinco de los seis caminos (swirler, tres filas de orificios y
    efusión). El sexto, `cool_ring`, es un camino MODELADO (anillo de domo +
    fugas de sello) que no está dibujado en la geometría y por tanto no
    existe en el campo: se le conserva la fracción que le da la red 1D y el
    resto se normaliza sobre lo que queda."""
    meas = post.get("mdot_paths_measured_kgs") or {}
    zone_of = {}
    for k in meas:
        if k in ("swirler", "effusion"):
            continue
        zone_of[k] = k
    by_path = {"swirler": abs(_f(meas.get("swirler"))),
               "effusion": abs(_f(meas.get("effusion")))}
    for zone in ("pz", "sz", "dz"):
        tot = 0.0
        for k, v in meas.items():
            if k in ("swirler", "effusion"):
                continue
            if k.startswith(zone):
                tot += abs(_f(v))
        by_path[f"{zone}_holes"] = tot
    f_ring = _f((net0.get("splits_solved_paths") or {}).get("cool_ring"), 0.0)
    f_ring = min(max(f_ring, 0.0), 0.5)
    s = sum(by_path.values())
    if s <= 0.0:
        return {}
    return {k: (1.0 - f_ring) * v / s for k, v in by_path.items()}


def _scale_path(rec2: dict, path: str, factor: float, acc: dict) -> None:
    """Escala el `C_d·A` efectivo del camino en el record (bucle de ajuste).

    * `swirler`   → `swirler.A_sw_m2` (el C_d del swirler es constante en
      `crn_core`, así que el grado de libertad es el área efectiva).
    * `*_holes`   → `hole_rows[zona].Cd` (es el coeficiente que la propia
      Ciencia §4.1 declara calibrable: «C_d,AM = C_d,ideal·f(Ra/d)», sin
      correlación cerrada publicada).
    * `effusion`  → `A_effusion_m2`.
    """
    lo, hi = MATCH_CD_CLAMP
    factor = min(max(float(factor), 1e-3), 1e3)
    new_acc = min(max(acc.get(path, 1.0) * factor, lo), hi)
    factor = new_acc / max(acc.get(path, 1.0), 1e-12)
    acc[path] = new_acc
    if path == "swirler":
        sw = dict(rec2.get("swirler") or {})
        sw["A_sw_m2"] = max(_f(sw.get("A_sw_m2")) * factor, 1e-12)
        rec2["swirler"] = sw
    elif path == "effusion":
        rec2["A_effusion_m2"] = max(_f(rec2.get("A_effusion_m2")) * factor,
                                    1e-12)
    elif path.endswith("_holes"):
        zone = path.split("_", 1)[0]
        rows = [dict(r) for r in (rec2.get("hole_rows") or [])]
        for r in rows:
            if str(r.get("zone")) == zone and _f(r.get("A_row_m2")) > 0.0:
                r["Cd"] = max(_f(r.get("Cd"), 0.62) * factor, 1e-3)
        rec2["hole_rows"] = rows


def build_measured_record(theta, rec: dict, post: dict) -> tuple[dict, dict]:
    """Record `phycb-record-1` con los VOLÚMENES y CAUDALES medidos dentro.

    Devuelve `(rec_medido, diagnóstico)`. El diagnóstico lleva la traza del
    ajuste (desviación por camino en cada iteración, factores acumulados y si
    se saturó el clamp) — sin eso, un split medido raro sería indistinguible
    de un bug del ajuste."""
    rec2 = dict(rec)
    V = post.get("V_zones_measured_m3") or {}
    v_note = []
    for key, zname in (("V_pz_m3", "pz"), ("V_sz_m3", "sz"),
                       ("V_dz_m3", "dz")):
        vm = _f(V.get(zname), 0.0)
        if vm > 0.0:
            v_note.append(f"{zname}: {1e6 * _f(rec.get(key)):.2f} → "
                          f"{1e6 * vm:.2f} cm³")
            rec2[key] = vm
    V_tot = sum(_f(V.get(z)) for z in ("pz", "sz", "dz"))
    if V_tot > 0.0:
        rec2["V_c_m3"] = V_tot

    net0 = crn.network_1d(theta, rec)
    target = _target_path_fracs(post, net0)
    acc: dict[str, float] = {}
    hist = []
    dev_max = float("nan")
    net = net0
    if target:
        for it in range(N_MATCH_ITERS):
            net = crn.network_1d(theta, rec2)
            now = net.get("splits_solved_paths") or {}
            devs = {}
            for p, t in target.items():
                cur = _f(now.get(p), 0.0)
                devs[p] = (cur / t - 1.0) if t > 1e-12 else 0.0
            dev_max = max((abs(v) for v in devs.values()), default=0.0)
            hist.append(dict(iter=it, dev_max=float(dev_max),
                             dev={k: float(v) for k, v in devs.items()}))
            if dev_max <= MATCH_TOL:
                break
            for p, t in target.items():
                cur = _f(now.get(p), 0.0)
                if cur <= 1e-12 or t <= 1e-12:
                    continue
                _scale_path(rec2, p, (t / cur) ** MATCH_RELAX, acc)
        net = crn.network_1d(theta, rec2)

    diag = dict(
        l15_volume_note="volúmenes de zona MEDIDOS en el campo frío: "
                        + "; ".join(v_note) if v_note else
                        "no se pudo medir ningún volumen de zona",
        l15_match_iters=len(hist), l15_match_dev_max=float(dev_max),
        l15_match_history=hist,
        l15_match_factors={k: float(v) for k, v in acc.items()},
        l15_match_clamped=bool(any(
            v <= MATCH_CD_CLAMP[0] * (1 + 1e-9) or
            v >= MATCH_CD_CLAMP[1] * (1 - 1e-9) for v in acc.values())),
        l15_target_path_fracs={k: float(v) for k, v in target.items()},
        l15_network_after=dict(
            splits_solved=dict(net.get("splits_solved") or {}),
            splits_solved_paths=dict(net.get("splits_solved_paths") or {}),
            converged=bool(net.get("network_converged"))))
    return rec2, diag


# ==========================================================================
# 3. Punto de entrada de `physics_core._try_higher`
# ==========================================================================
def topology_from_field(theta, rec: dict) -> dict | None:
    """L1.5: campo frío 3D → CRN con volúmenes y caudales MEDIDOS.

    Firma EXACTA que `physics_core._try_higher` invoca. `None` ⇒ el punto
    degrada con etiqueta; el motivo queda en `LAST_STATUS`.

    **Nunca lanza.**
    """
    global LAST_STATUS
    LAST_STATUS = "not_run"
    if not isinstance(rec, dict):
        LAST_STATUS = "degenerate"
        return None
    try:
        return _topology_impl(theta, rec)
    except Exception as exc:                     # red de seguridad
        LAST_STATUS = f"error:{type(exc).__name__}"
        return None


def _topology_impl(theta, rec: dict) -> dict | None:
    global LAST_STATUS
    t0 = time.monotonic()
    post = measure(theta, rec)
    if post is None:
        return None                              # LAST_STATUS ya puesto
    rec2, diag = build_measured_record(theta, rec, post)

    if not getattr(crn, "CANTERA_AVAILABLE", False):
        LAST_STATUS = "cantera_unavailable"
        return None
    sol = crn.crn_solve(theta, rec2, L15_CRN_TIMEOUT_S)
    if not sol:
        LAST_STATUS = f"crn_failed:{getattr(crn, 'LAST_STATUS', '?')}"
        return None

    out: dict = dict(sol)
    out.update(diag)
    for k in ("splits_measured", "mdot_paths_measured_kgs",
              "mdot_in_measured_kgs", "mdot_out_measured_kgs",
              "mass_closure_rel", "rows_discretization",
              "Y_over_d_measured", "jet_penetration_measured",
              "V_zones_measured_m3", "V_recirc_pz_m3", "recirc_frac_pz",
              "mdot_zone_exchange_kgs", "tau_zones_measured_ms",
              "tau_zones_measured_cold_ms", "dp_over_p_cfd",
              "dp_annulus_liner_Pa", "cfd_mesh"):
        if k in post:
            out[k] = post[k]

    #  El campo frío NO produce ni y+ (bounce-back de media celda no tiene el
    #  concepto) ni PF ni T4: se declara con la clave puesta a vacío en vez de
    #  inventar un número.
    out["yplus_hist"] = []
    out["l15_source"] = "crn_L15(xlb_cold)"
    out["l15_backend"] = str((post.get("cfd_mesh") or {}).get("backend", "?"))
    out["crn_topology"] = (
        f"{sol.get('crn_topology', '')} | AUTO-TOPOLOGÍA L1.5 "
        f"({out['l15_backend']}, malla "
        f"{(post.get('cfd_mesh') or {}).get('nx', '?')}×"
        f"{(post.get('cfd_mesh') or {}).get('ny', '?')}×"
        f"{(post.get('cfd_mesh') or {}).get('nz', '?')} a dx="
        f"{(post.get('cfd_mesh') or {}).get('dx_mm', '?')} mm): volúmenes de "
        f"zona y caudales por camino MEDIDOS en el campo 3D, no del "
        f"dimensionado")
    out["l15_wall_s"] = float(time.monotonic() - t0)
    out["l15_notes"] = list(post.get("cold_post_notes") or [])
    out["l15_limitations"] = list(post.get("cold_post_limitations") or []) + [
        "L1.5 mide AERODINÁMICA fría: no predice PF, ni T4, ni temperatura de "
        "pared. Phy-CB no predice temperatura de metal cuantitativa en ningún "
        "nivel de fidelidad (contrato §12: L2b fuera de alcance).",
        "El CRN de L1.5 hereda TODAS las limitaciones del de L1 (0-D por "
        "zona, EI_CO cota inferior, NOx ±30–50 %); lo que cambia son los "
        "volúmenes y los caudales, que ahora son medidos."]
    LAST_STATUS = "ok"
    return out


# ==========================================================================
# 4. CLI de comprobación
# ==========================================================================
def main(argv=None) -> int:
    """Smoke de la auto-topología sobre el caso ancla."""
    import argparse
    ap = argparse.ArgumentParser(description="Phy-CB L1.5 auto-CRN smoke")
    ap.add_argument("--dx-mm", type=float, default=None)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--backend", default="auto",
                    choices=("auto", "xlb", "cupy", "numpy"))
    ap.add_argument("--measure-only", action="store_true")
    args = ap.parse_args(argv)
    theta = np.array(pc.THETA_REF, dtype=float)
    rec = pc.evaluate(theta, pc.Fidelity.L0, use_cache=False)
    if args.measure_only:
        post = measure(theta, rec, dx_mm=args.dx_mm, backend=args.backend,
                       max_nodes=args.max_nodes, log=print)
        if post is None:
            print(f"measure() devolvió None ({LAST_STATUS})")
            return 1
        print("splits medidos:", {k: round(v, 4)
                                  for k, v in post["splits_measured"].items()})
        print("ṁ por camino  :", {k: round(v, 6) for k, v in
                                  post["mdot_paths_measured_kgs"].items()})
        print("V zonas [cm³] :", {k: round(1e6 * v, 2) for k, v in
                                  post["V_zones_measured_m3"].items()})
        print("Y/d medidos   :", {k: round(v, 3) for k, v in
                                  post["Y_over_d_measured"].items()})
        return 0
    sol = topology_from_field(theta, rec)
    if sol is None:
        print(f"topology_from_field devolvió None ({LAST_STATUS})")
        return 1
    print("source L1.5   :", sol["l15_source"])
    print("splits medidos:", {k: round(v, 4)
                              for k, v in sol["splits_measured"].items()})
    print("τ zonas [ms]  :", {k: round(v, 4)
                              for k, v in sol["tau_zones_measured_ms"].items()})
    print("EI_NOx_crn    :", round(_f(sol.get("EI_NOx_crn")), 4))
    print("eta_comb_crn  :", round(_f(sol.get("eta_comb_crn")), 6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
