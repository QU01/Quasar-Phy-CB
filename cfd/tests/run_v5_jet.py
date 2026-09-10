#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/run_v5_jet.py
=======================================
**Script MANUAL** del V5 completo: chorro turbulento PLANO libre — mide la
difusión turbulenta del modelo (WAVE6 §2), que es la que gobierna la
penetración y la mezcla de los chorros de dilución del combustor.

Caso y referencia
-----------------
Chorro de ranura d con co-flujo débil (0.04·u_j, estabilidad del contorno
lateral). Referencia: Pope (2000) «Turbulent Flows» §5.4, chorro plano
autosemejante: **S = dy½/dx ≈ 0.10** (banda experimental 0.096–0.11).
Criterio: **±15 %**. La medida usa el EXCESO de velocidad sobre el
co-flujo, así que este no sesga S. Comprobación independiente: el
decaimiento (u_j/Δu_c)² debe ser lineal en x (autosemejanza).

NOTA honesta: el chorro REDONDO es la anomalía clásica de k-ε (predice
0.116 vs 0.094); aquí se verifica el PLANO, que es el que tiene banda
experimental estrecha. El realizable en plano tiende a quedar ~10 % corto
con esta discretización (medido: ver ESTADO.md).

Uso::

    python cfd/tests/run_v5_jet.py                  # dx = d/5, 60k it
    python cfd/tests/run_v5_jet.py --dx-inv 8 --max-iter 120000
    python cfd/tests/run_v5_jet.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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

import physics_core as pc                                     # noqa: E402

pc.assert_provenance(pc)          # contrato §0.b

import test_ibrans_v4_v5 as vv                                # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V5 manual: chorro plano libre")
    ap.add_argument("--dx-inv", type=float, default=5.0,
                    help="celdas por anchura de ranura d")
    ap.add_argument("--x-out", type=float, default=40.0)
    ap.add_argument("--half-w", type=float, default=13.0)
    ap.add_argument("--max-iter", type=int, default=60_000)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    print(f"V5 · chorro plano Re=30k, dx=d/{a.dx_inv:.0f}, dominio "
          f"[0,{a.x_out}]x[±{a.half_w}] d — esto tarda ~minutos")
    r = vv.run_v5(dx_inv=a.dx_inv, x_out=a.x_out, half_w=a.half_w,
                  max_iter=a.max_iter)
    ok = r.get("ok", False) and r["err"] < vv.TOL_V5
    print(f"  malla        : {r['cells']} celdas")
    print(f"  estado       : {r['status']}  ({r['iterations']} iter, "
          f"{r['wall_time_s']:.1f} s, res_u {r['res_u']:.2e})")
    print(f"  rendimiento  : {r['s_per_cell_step']:.3e} s/celda/paso")
    if r.get("ok"):
        print(f"  S = dy12/dx  : {r['S']:.4f}  vs {vv.JET_S_REF} "
              f"({100 * r['err']:+.1f} %)  → {'PASA' if ok else 'NO PASA'} "
              f"(umbral ±{100 * vv.TOL_V5:.0f} %)")
        print(f"  origen virt. : {r['x0']:.1f} d")
        print(f"  decaimiento  : pendiente de (u_j/du_c)^2 = "
              f"{r['decay_slope']:.4f} (>0 = autosemejante)")
    else:
        print(f"  MEDIDA FALLIDA: {r.get('reason', '?')}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(r, passed=bool(ok)), fh, indent=2,
                      ensure_ascii=False)
        print(f"  JSON         : {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
