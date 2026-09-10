#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/run_v4_bfs.py
=======================================
**Script MANUAL** del V4 completo: backward-facing step turbulento — EL
test discriminante del hito M2 (WAVE6 §2). Si la longitud de readherencia
sale mal, el k-ε + wall functions no vale, por bonitos que sean los demás
números.

Caso y referencia
-----------------
Vogel & Eaton (1985), J. Heat Transfer 107, 922–929: Re_h = 28 000,
razón de expansión 1.25, **x_r/h = 6.67 ± 0.1**. Criterio: **±15 %**
(banda 5.67–7.67). El escalón es un BLOQUE INMERSO (prueba las wall
functions sobre |φ|); suelo y techo son paredes del dominio (prueban las
de contorno). Entrada con capa límite 1/7 de δ = 1.1h, como el experimento.

x_r se mide por el último cruce − → + de u en la primera fila de celdas
sobre el suelo (interpolado). El caso es ESTACIONARIO: se usa
mach_art = 0.25 porque la pseudo-compresibilidad no cambia el estado
estacionario, sólo el camino (ver docstring de test_ibrans_v4_v5).

Número medido en esta máquina (2026-08-02, CPU): dx = h/8, 120k
iteraciones → **x_r = 6.23h (−6.6 %)**, 22 min. A dx = h/5 con 40k it sale
5.7h: el caso NO está convergido ahí — por eso el ±15 % vive aquí y no en
la suite rápida.

Uso::

    python cfd/tests/run_v4_bfs.py                 # dx = h/8, 120k it
    python cfd/tests/run_v4_bfs.py --dx-inv 10 --max-iter 200000
    python cfd/tests/run_v4_bfs.py --json out.json
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
    ap = argparse.ArgumentParser(
        description="V4 manual: backward-facing step (Vogel & Eaton)")
    ap.add_argument("--dx-inv", type=float, default=8.0,
                    help="celdas por altura de escalon h")
    ap.add_argument("--x-out", type=float, default=25.0)
    ap.add_argument("--max-iter", type=int, default=120_000)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    print(f"V4 · BFS Re_h=28k ER=1.25, dx=h/{a.dx_inv:.0f}, dominio "
          f"[-4,{a.x_out}]x[0,5] h — esto tarda ~decenas de minutos en CPU")
    r = vv.run_v4(dx_inv=a.dx_inv, x_out=a.x_out, max_iter=a.max_iter)
    ok = r["err"] < vv.TOL_V4
    ys = r["ystar"]
    print(f"  malla        : {r['cells']} celdas")
    print(f"  estado       : {r['status']}  ({r['iterations']} iter, "
          f"{r['wall_time_s']:.1f} s, res_u {r['res_u']:.2e})")
    print(f"  rendimiento  : {r['s_per_cell_step']:.3e} s/celda/paso")
    print(f"  x_r/h        : {r['x_r']:.2f}  vs {vv.BFS_XR_REF} "
          f"({100 * r['err']:+.1f} %)  → {'PASA' if ok else 'NO PASA'} "
          f"(umbral ±{100 * vv.TOL_V4:.0f} %)")
    print(f"  u_min suelo  : {r['u_min_floor']:.3f} (recirculacion)")
    print(f"  y*           : medio {ys.get('ystar_mean', float('nan')):.0f}, "
          f"max {ys.get('ystar_max', float('nan')):.0f}, "
          f"{100 * ys.get('frac_in_target', 0.0):.0f} % en banda [30,150]")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(r, passed=bool(ok)), fh, indent=2,
                      ensure_ascii=False)
        print(f"  JSON         : {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
