#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/run_v3_cylinder.py
============================================
**Script MANUAL** de la V3 completa: cilindro en flujo cruzado a Re = 100
resuelto con **frontera inmersa pura** (el cilindro sólo existe como SDF;
no hay malla ajustada al cuerpo). Es el test que mide la precisión del IB.

Qué mide y cómo
---------------
* **Cd**: balance de cantidad de movimiento sobre un volumen de control
  INTERIOR al dominio (`post.control_volume_force`), promediado sobre la
  ventana temporal. No se integra la tensión sobre la superficie inmersa a
  propósito: sobre IB esa integral es la parte frágil, y el balance de
  cantidad de movimiento es exacto para el sistema discreto salvo los
  flujos de la caja, que sí se calculan bien porque la caja está en el
  interior del dominio.
* **St**: FFT de la velocidad transversal en una sonda de la estela
  (x = 1.5 D, y = 0), con comprobación independiente por cruces por cero.

Referencias (Ciencia §13.1, «clásico»)
--------------------------------------
* Cd(Re=100) ≈ 1.35 — Liu, Zheng & Sung (1998) JCP 139, 35–57; 1.364 en
  Braza, Chassaing & Ha Minh (1986) JFM 165, 79–130.
* St(Re=100) ≈ 0.164–0.166 — Williamson (1989) JFM 206, 579–627 (exp.).

Criterio: **±10 %** en ambos.

Backend: a dx = D/10 el caso tiene 38 400 celdas, por debajo del cruce
CPU/GPU medido (`backend.GPU_MIN_CELLS` = 100 000), así que con `auto`
corre en NumPy/CPU aunque haya GPU. Para forzar la GPU (y comprobar que la
V&V pasa también allí, con los mismos umbrales):
`PHYCB_L2A_BACKEND=cupy python cfd/tests/run_v3_cylinder.py --dx 0.1`.

Uso::

    python cfd/tests/run_v3_cylinder.py                  # dx = D/10
    python cfd/tests/run_v3_cylinder.py --dx 0.0625 --t-end 250
    python cfd/tests/run_v3_cylinder.py --json out.json
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
    ap = argparse.ArgumentParser(description="V3 manual: cilindro con IB")
    ap.add_argument("--dx", type=float, default=0.1, help="paso de malla / D")
    ap.add_argument("--xlo", type=float, default=-8.0)
    ap.add_argument("--xhi", type=float, default=16.0)
    ap.add_argument("--yh", type=float, default=8.0,
                    help="semialtura del dominio en D (bloqueo = D/2yh)")
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--t-end", type=float, default=180.0,
                    help="tiempo final en D/U")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    blockage = 1.0 / (2.0 * a.yh)
    print(f"V3 · cilindro Re={a.re:.0f}, dx=D/{1.0 / a.dx:.0f}, dominio "
          f"[{a.xlo},{a.xhi}]x[±{a.yh}] D, bloqueo {100 * blockage:.1f} % — "
          "esto tarda minutos")
    r = vv.run_v3(dx=a.dx, xlo=a.xlo, xhi=a.xhi, yh=a.yh, re=a.re,
                  t_end=a.t_end)
    ok_cd = r["err_Cd"] < vv.TOL_V3
    ok_st = r["err_St"] < vv.TOL_V3
    print(f"  malla           : {r['cells']} celdas, {r['ghost']} fantasmas, "
          f"{r['orphan']} huerfanas")
    print(f"  estado          : {r['status']}  ({r['iterations']} iter, "
          f"{r['wall_time_s']:.1f} s)")
    print(f"  rendimiento     : {r['s_per_cell_step']:.3e} s/celda/paso "
          f"(backend {r['backend']}/{r['dtype']})")
    print(f"  Cd              : {r['Cd']:.4f}  vs {vv.CYL_CD_REF} "
          f"({100 * r['err_Cd']:.1f} %)  → {'PASA' if ok_cd else 'NO PASA'}")
    print(f"  St (FFT)        : {r['St']:.4f}  vs {vv.CYL_ST_REF} "
          f"({100 * r['err_St']:.1f} %)  → {'PASA' if ok_st else 'NO PASA'}")
    print(f"  St (cruce cero) : {r['St_zerocross']:.4f}  (comprobacion "
          "independiente)")
    print(f"  Cl medio        : {r['Cl']:+.4f}  (debe ser ~0 en media)")
    print(f"  amplitud sonda  : {r['amplitude']:.4f}")
    print(f"  NOTA            : el bloqueo lateral de {100 * blockage:.1f} % "
          "infla el Cd; con ±10 D baja ~2-3 %.")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(r, passed_Cd=bool(ok_cd), passed_St=bool(ok_st)),
                      fh, indent=2, ensure_ascii=False)
        print(f"  JSON            : {a.json}")
    return 0 if (ok_cd and ok_st) else 1


if __name__ == "__main__":
    sys.exit(main())
