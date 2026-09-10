#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/run_v2_re1000.py
==========================================
**Script MANUAL** de la V2 a Re = 1000 (cavidad de tapa deslizante de Ghia).

Por qué es manual y no un test de CI: a Re = 1000 la cavidad necesita malla
128² y ~50 tiempos convectivos, lo que son varios minutos — fuera del
presupuesto de una suite que tiene que correr en cada cambio. El test de CI
corre Re = 100 en 32² y este script cubre Re = 1000; **el resultado medido
está anotado en `cfd/ibrans/ESTADO.md`**, no se deduce ni se promete.

Backend: 128² son 16 384 celdas, muy por debajo del cruce CPU/GPU medido
(`backend.GPU_MIN_CELLS` = 100 000), así que con `auto` esto corre en
NumPy/CPU **aunque haya GPU** — a este tamaño la GPU pierde por coste de
lanzamiento de kernels. Para forzarla: `PHYCB_L2A_BACKEND=cupy`.

Uso::

    python cfd/tests/run_v2_re1000.py                # 128^2, Re=1000
    python cfd/tests/run_v2_re1000.py --n 96 --re 400
    python cfd/tests/run_v2_re1000.py --json out.json

Criterio (Ciencia §13.1): perfiles de las Tablas I y II de Ghia et al.
(1982) dentro de **±5 % de la velocidad de la tapa** (U = 1), excluidos los
dos puntos de contorno.
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

import test_ibrans_v1_v3 as vv                                # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V2 manual: cavidad de Ghia")
    ap.add_argument("--n", type=int, default=128, help="celdas por lado")
    ap.add_argument("--re", type=int, default=1000, choices=(100, 1000))
    ap.add_argument("--tol-res", type=float, default=2.0e-5,
                    help="residual de parada")
    ap.add_argument("--max-iter", type=int, default=600_000)
    ap.add_argument("--json", default="", help="vuelca el resultado a JSON")
    a = ap.parse_args(argv)

    print(f"V2 · cavidad de tapa deslizante Re={a.re} en {a.n}^2 "
          "— esto tarda minutos, no segundos")
    r = vv.run_v2(n=a.n, re=a.re, tol=a.tol_res, max_iter=a.max_iter)
    ok = max(r["err_u"], r["err_v"]) < vv.TOL_V2
    print(f"  estado          : {r['status']}  ({r['iterations']} iter, "
          f"{r['wall_time_s']:.1f} s)")
    print(f"  residuales      : res_u={r['res_u']:.2e}  "
          f"res_div={r['res_div']:.2e}")
    print(f"  rendimiento     : {r['s_per_cell_step']:.3e} s/celda/paso "
          f"({r['cells']} celdas, backend {r['backend']}/{r['dtype']})")
    print(f"  err max |u−Ghia|: {r['err_u']:.4f}   (tol {vv.TOL_V2})")
    print(f"  err max |v−Ghia|: {r['err_v']:.4f}   (tol {vv.TOL_V2})")
    print(f"  V2 Re={a.re}      : {'PASA' if ok else 'NO PASA'}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(r, passed=bool(ok)), fh, indent=2,
                      ensure_ascii=False)
        print(f"  JSON            : {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
