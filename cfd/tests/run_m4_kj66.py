#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/run_m4_kj66.py
========================================
**Script MANUAL del caso ancla KJ66** (WAVE6 §4, criterios de aceptación
de la wave): el producto L2a completo desde θ_REF, con sus dos contrastes
oficiales y (opcional) el test de ranking:

    Δp/p CFD vs red 1D del record .......... ±20 %
    T4 CFD vs balance (T04_K del record) ... ±5 %
    --ranking: PF(dz 6 orificios) vs PF(dz 10) ordena como L0

Es el caso que fija los NÚMEROS de `ESTADO.md` (regla de la casa: todo
número reportado, medido en esta máquina). El presupuesto objetivo es
<30 min en GPU (criterio #2 de la wave).

Uso::

    python cfd/tests/run_m4_kj66.py                     # dx=1.5, reactivo
    python cfd/tests/run_m4_kj66.py --dx-mm 2 --max-iter 9000
    python cfd/tests/run_m4_kj66.py --ranking --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np                                            # noqa: E402

import physics_core as pc                                     # noqa: E402

pc.assert_provenance(pc)

import cfd.ibrans as ib                                       # noqa: E402

TOL_DP = 0.20          # Δp/p vs red 1D (WAVE6 §4)
TOL_T4 = 0.05          # T4 vs balance (WAVE6 §4)


def _one(theta, dx_mm, max_iter, timeout_s, label=""):
    rec = pc.evaluate(theta, pc.Fidelity.L1, use_cache=False)
    t0 = time.perf_counter()
    sol = ib.solve(theta, rec, dx_mm=dx_mm, timeout_s=timeout_s,
                   max_iter=max_iter)
    el = time.perf_counter() - t0
    print(f"\n=== caso {label or 'ancla'} · {el:.0f} s · "
          f"estado {ib.LAST_STATUS.get('status')} ===")
    if sol is None:
        print(f"  solve() devolvió None ({ib.DEGRADE_TAG}): "
              f"{ib.LAST_STATUS.get('reason')}")
        return rec, None, el
    for k in ("PF_cfd", "T4_cfd_K", "RTDF_cfd", "dp_over_p_cfd",
              "T_wall_adiabatic_max_K", "mass_closure_rel", "EI_NOx_crn"):
        if k in sol:
            print(f"  {k:24s}: {sol[k]:.6g}" if isinstance(
                sol[k], float) else f"  {k:24s}: {sol[k]}")
    print(f"  splits medidos          : "
          f"{ {k: round(v, 4) for k, v in (sol.get('splits_measured') or {}).items()} }")
    print(f"  Y/d medidos             : "
          f"{ {k: round(v, 3) for k, v in (sol.get('Y_over_d_measured') or {}).items()} }")
    print(f"  nota                    : {sol.get('fidelity_note')}")
    return rec, sol, el


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="caso ancla KJ66 (M4, manual)")
    ap.add_argument("--dx-mm", type=float, default=1.5)
    ap.add_argument("--max-iter", type=int, default=15_000)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--ranking", action="store_true",
                    help="corre ademas dz=6 y dz=10 y compara el orden")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    out = dict(dx_mm=a.dx_mm, max_iter=a.max_iter)

    rec, sol, el = _one(pc.THETA_REF, a.dx_mm, a.max_iter, a.timeout_s)
    ok = sol is not None
    if ok:
        anchored = ib.LAST_STATUS.get("status") == "product"
        dp_ref = float(rec.get("dp_over_p", 0.0))
        t4_ref = float(rec.get("T04_K", 0.0))
        dev_dp = sol.get("dp_over_p_cfd", float("nan")) / dp_ref - 1.0 \
            if dp_ref else float("nan")
        dev_t4 = sol.get("T4_cfd_K", float("nan")) / t4_ref - 1.0 \
            if t4_ref else float("nan")
        print(f"\n  CONTRASTES OFICIALES (WAVE6 §4)")
        print(f"  llama anclada          : {anchored}")
        print(f"  dp/p  {sol.get('dp_over_p_cfd', float('nan')):.4f} vs "
              f"red 1D {dp_ref:.4f}  → {100 * dev_dp:+.1f} % "
              f"({'PASA' if abs(dev_dp) < TOL_DP else 'NO PASA'} ±20 %)")
        print(f"  T4    {sol.get('T4_cfd_K', float('nan')):.0f} K vs "
              f"balance {t4_ref:.0f} K → {100 * dev_t4:+.1f} % "
              f"({'PASA' if abs(dev_t4) < TOL_T4 else 'NO PASA'} ±5 %)")
        print(f"  presupuesto <30 min    : {el / 60:.1f} min "
              f"({'PASA' if el < 1800 else 'NO PASA'})")
        ok = anchored and abs(dev_dp) < TOL_DP and abs(dev_t4) < TOL_T4
        out.update(anchored=anchored, dev_dp=dev_dp, dev_t4=dev_t4,
                   wall_s=el, product=sol,
                   record_refs=dict(dp_over_p=dp_ref, T04_K=t4_ref))

    if a.ranking and ok:
        pf = {}
        for n_dz, lbl in ((6.0, "dz6"), (10.0, "dz10")):
            th = np.array(pc.THETA_REF, dtype=float)
            th[7] = n_dz
            r2, s2, _ = _one(th, a.dx_mm, a.max_iter, a.timeout_s, lbl)
            if s2 is None or "PF_cfd" not in s2:
                ok = False
                break
            pf[lbl] = dict(PF_cfd=float(s2["PF_cfd"]),
                           PF_L0=float(r2["PF"]))
        if len(pf) == 2:
            o_cfd = pf["dz6"]["PF_cfd"] > pf["dz10"]["PF_cfd"]
            o_l0 = pf["dz6"]["PF_L0"] > pf["dz10"]["PF_L0"]
            print(f"\n  RANKING dz6 vs dz10: CFD "
                  f"{'dz6>dz10' if o_cfd else 'dz10>=dz6'}, L0 "
                  f"{'dz6>dz10' if o_l0 else 'dz10>=dz6'} → "
                  f"{'COHERENTE' if o_cfd == o_l0 else 'INCOHERENTE'}")
            out["ranking"] = dict(pf=pf, coherent=bool(o_cfd == o_l0))
            ok = ok and (o_cfd == o_l0)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(out, passed=bool(ok)), fh, indent=2,
                      ensure_ascii=False, default=str)
        print(f"\n  JSON: {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
