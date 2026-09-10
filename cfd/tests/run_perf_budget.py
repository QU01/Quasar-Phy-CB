#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/run_perf_budget.py
============================================
**Script MANUAL** que RE-MIDE el presupuesto de rendimiento del solver L2a
en los dos backends (NumPy/CPU y CuPy/GPU) y decide si el objetivo de
WAVE6 §5 —**500 000 celdas × 10 000 iteraciones**— se cumple.

Por qué existe
--------------
La primera versión de `cfd/ibrans/ESTADO.md` §5 declaraba el objetivo
incumplido con una premisa FALSA: «esta máquina no tiene GPU». La máquina
tiene una RTX 3070 Ti y CuPy 14.1.1 instalado y funcionando. La medida en
CPU era buena; la conclusión, no. Este script vuelve a medirlo TODO con el
mismo caso y deja los dos números uno al lado del otro.

Qué mide, en este orden
-----------------------
1. Ficha real de la GPU y del entorno (`backend.gpu_info`).
2. Kernel sintético de advección-difusión (el que la wave manda usar).
3. **El solver real** en 3D, que es la cifra que cuenta, barriendo tamaños
   en los dos backends: `s/celda/paso`, VRAM usada y transferencias
   device→host por paso (debe ser ≈ 1: sólo el escalar del CFL).
4. Extrapolación al objetivo de la wave y comparación con la referencia
   pública de 1e-8 s/celda/paso (JAX-Fluids, Ciencia §11.2).

El caso 3D es el MISMO en los dos backends (misma malla, mismas BC, mismo
número de pasos), así que la razón CPU/GPU es una razón limpia.

Uso::

    python cfd/tests/run_perf_budget.py                  # completo
    python cfd/tests/run_perf_budget.py --steps 40
    python cfd/tests/run_perf_budget.py --only-gpu
    python cfd/tests/run_perf_budget.py --json perf.json
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

pc.assert_provenance(pc)          # contrato §0.b

from cfd.ibrans import backend as bk                          # noqa: E402
from cfd.ibrans.mesh import CartesianMesh, sdf_sphere         # noqa: E402
from cfd.ibrans.solver import (IBRANSSolver, SolverConfig,    # noqa: E402
                               make_bcs)

#: Tamaños del barrido 3D (celdas por lado → n³ celdas). 80³ = 512 000 es
#: EL caso del objetivo de la wave; los demás sirven para ver dónde está el
#: cruce CPU/GPU (en mallas pequeñas la GPU pierde por coste de lanzamiento).
SIZES_3D = (20, 32, 40, 60, 80, 100, 128)


def case_3d(n: int, ib: bool = True, rk: str = "rk3", backend: str = "auto",
            precision: str = "auto", steps: int = 30):
    """Cubo n³ con entrada/salida en Z, paredes laterales de deslizamiento y
    —si `ib`— una **esfera inmersa** en el centro (D = 0.4 L) para que el
    ghost-cell del IB entre en la cuenta. Sin la esfera se mediría un
    kernel de volumen puro y el IB saldría gratis, que no es el caso real.
    """
    L = 1.0
    dx = L / int(n)
    m = CartesianMesh([0.0, 0.0, 0.0], [L, L, L], dx)
    m.set_sdf(sdf_sphere((0.5 * L, 0.5 * L, 0.5 * L), 0.2 * L, solid=True)
              if ib else None)
    cfg = SolverConfig(rho=1.0, nu=1.0e-3, u_ref=1.0, l_ref=L, mach_art=0.1,
                       cfl=0.4, rk=rk, max_iter=int(steps), tol=0.0,
                       residual_every=10 ** 9, backend=backend,
                       precision=precision)
    bcs = make_bcs(xlo="slip", xhi="slip", ylo="slip", yhi="slip",
                   zlo=dict(kind="inlet", u=(0.0, 0.0, 1.0)),
                   zhi=dict(kind="outlet", p=0.0))
    return IBRANSSolver(m, cfg, bcs)


def measure(n: int, backend: str, ib: bool = True, rk: str = "rk3",
            precision: str = "auto", steps: int = 30,
            warmup: int = 3) -> dict:
    """Un punto del barrido. `warmup` pasos sin cronometrar (en GPU el
    primer paso paga la reserva del pool y la compilación de kernels).
    **Nunca lanza**: un fallo se devuelve como `ok=False` + motivo."""
    try:
        s = case_3d(n, ib=ib, rk=rk, backend=backend, precision=precision,
                    steps=steps)
        for _ in range(int(warmup)):
            s.step()
        s.be.sync()
        info = s.run(max_iter=int(steps))
        return dict(ok=True, n=n, cells=info["cells"],
                    backend=s.be.name, dtype=np.dtype(s.be.dtype).name,
                    is_gpu=bool(s.be.is_gpu), rk=rk, ib=bool(ib),
                    ghost=int(s.mesh.n_ghost),
                    s_per_cell_step=info["s_per_cell_step"],
                    wall_time_s=info["wall_time_s"], steps=info["steps"],
                    d2h_per_step=info["d2h_per_step"],
                    vram_used_MiB=info["vram_used_MiB"],
                    note=s.be.note)
    except Exception as exc:
        return dict(ok=False, n=n, backend=backend,
                    error=f"{type(exc).__name__}: {exc}")


def _target(scs: float) -> tuple[float, float]:
    """(segundos, horas) del objetivo 500k×10k a ese ritmo."""
    t = scs * bk.TARGET_CELLS * bk.TARGET_STEPS
    return t, t / 3600.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-medida del presupuesto de rendimiento del L2a")
    ap.add_argument("--steps", type=int, default=30,
                    help="pasos cronometrados por punto")
    ap.add_argument("--sizes", default="",
                    help="lista de n (celdas por lado), coma-separada")
    ap.add_argument("--only-gpu", action="store_true")
    ap.add_argument("--only-cpu", action="store_true")
    ap.add_argument("--no-ib", action="store_true",
                    help="barrido sin esfera inmersa")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    sizes = ([int(v) for v in a.sizes.split(",") if v.strip()]
             if a.sizes else list(SIZES_3D))
    backends = []
    if not a.only_gpu:
        backends.append("numpy")
    if not a.only_cpu and bk.available_backends().get("cupy"):
        backends.append("cupy")

    out: dict = dict(when=time.strftime("%Y-%m-%d %H:%M:%S"),
                     gpu=bk.gpu_info(), available=bk.available_backends())
    print("=" * 74)
    print("PRESUPUESTO DE RENDIMIENTO DEL SOLVER L2a — re-medida")
    print("=" * 74)
    g = out["gpu"]
    if g.get("ok"):
        print(f"GPU : {g['name']} · cc {g['compute_capability']} · "
              f"{g['vram_free_GiB']:.2f}/{g['vram_total_GiB']:.2f} GiB libres")
        print(f"      cupy {g['cupy_version']} · CUDA runtime "
              f"{g['cuda_runtime']} · driver {g['cuda_driver']}")
    else:
        print(f"GPU : NO disponible ({g.get('reason', '')})")
    print(f"Backends: {out['available']}")

    # --- 1. kernel sintetico ---------------------------------------------
    print("\n1) Kernel sintetico adveccion-difusion 96^3 (884 736 celdas)")
    out["synthetic"] = []
    for b in backends:
        for prec in ("fp32", "fp64"):
            r = bk.benchmark(n=96, steps=20, ndim=3, precision=prec,
                             backend=b)
            out["synthetic"].append(r)
            if r.get("ok"):
                print(f"   {b:5s} {prec}: {r['s_per_cell_step']:.3e} "
                      f"s/celda/paso · objetivo {r['target_h']:.4f} h · "
                      f"x{r['ratio_vs_ref_gpu']:.2f} vs ref GPU")
            else:
                print(f"   {b:5s} {prec}: FALLO {r.get('error')}")

    # --- 2. solver real ---------------------------------------------------
    ib = not a.no_ib
    print(f"\n2) SOLVER REAL, cubo 3D con {'esfera inmersa' if ib else 'IB'}"
          f"{'' if ib else ' desactivado'}, rk3, {a.steps} pasos "
          "cronometrados")
    print(f"   {'n^3':>6} {'celdas':>9} {'backend':>7} {'dtype':>7} "
          f"{'s/celda/paso':>13} {'obj 500kx10k':>13} {'VRAM MiB':>9} "
          f"{'D2H/paso':>9}")
    out["solver"] = []
    for n in sizes:
        for b in backends:
            r = measure(n, b, ib=ib, steps=a.steps)
            out["solver"].append(r)
            if not r.get("ok"):
                print(f"   {n:>6} {'-':>9} {b:>7}   FALLO: {r['error']}")
                continue
            _, th = _target(r["s_per_cell_step"])
            hh = f"{th:.3f} h" if th >= 0.01 else f"{th * 3600:.1f} s"
            print(f"   {n:>6} {r['cells']:>9} {r['backend']:>7} "
                  f"{r['dtype']:>7} {r['s_per_cell_step']:>13.3e} "
                  f"{hh:>13} {r['vram_used_MiB']:>9.1f} "
                  f"{r['d2h_per_step']:>9.2f}")

    # --- 3. veredicto -----------------------------------------------------
    tgt = [r for r in out["solver"] if r.get("ok") and r["cells"] >= 500_000]
    best = {}
    for r in tgt:
        k = r["backend"]
        if k not in best or r["s_per_cell_step"] < best[k]["s_per_cell_step"]:
            best[k] = r
    print("\n3) Veredicto sobre el objetivo de la wave "
          f"({bk.TARGET_CELLS} celdas x {bk.TARGET_STEPS} iteraciones, "
          "5-40 min)")
    out["verdict"] = {}
    for k, r in best.items():
        ts, th = _target(r["s_per_cell_step"])
        ok = (300.0 <= ts <= 2400.0) or ts < 300.0
        out["verdict"][k] = dict(s_per_cell_step=r["s_per_cell_step"],
                                 target_s=ts, target_h=th, meets=bool(ok),
                                 cells=r["cells"])
        print(f"   {k:5s}: {r['s_per_cell_step']:.3e} s/celda/paso "
              f"(caso de {r['cells']} celdas) → {ts / 60.0:.1f} min "
              f"({th:.2f} h) → objetivo {'SE CUMPLE' if ok else 'NO se cumple'}")
    if "numpy" in best and "cupy" in best:
        sp = (best["numpy"]["s_per_cell_step"]
              / best["cupy"]["s_per_cell_step"])
        out["verdict"]["speedup_gpu_vs_cpu"] = sp
        print(f"   aceleracion GPU/CPU medida en el caso del objetivo: x{sp:.1f}")
    print(f"   referencia publica GPU FP32 (Ciencia §11.2): "
          f"{bk.REF_S_PER_CELL_STEP_FP32_GPU:.0e} s/celda/paso")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
        print(f"\nJSON: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
