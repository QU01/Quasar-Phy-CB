#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/run_v5_jcf.py
=======================================
**Script MANUAL** del chorro CALIENTE en flujo cruzado (JICF) — el tercer
caso de la tarea 2.6 de M2 (WAVE6 §2: «chorro caliente en cruzado,
trayectoria vs correlación»). Es 3D y ejercita a la vez: k-ε realizable,
energía low-Mach, ρ(T) variable, escalar pasivo y entrada por perfil
(la ranura circular del chorro en la pared inferior).

Caso y referencia
-----------------
Chorro redondo de diámetro d a T_j = 600 K entrando perpendicular a un
cruzado a T_∞ = 300 K, con J = ρ_j·u_j²/(ρ_∞·u_∞²) y r = √J; con gas
ideal a p₀ constante ρ_j/ρ_∞ = T_∞/T_j.

La trayectoria se mide como el lugar del MÁXIMO de exceso de temperatura
en cada plano x del plano de simetría — el mismo criterio que usará el M4
para las penetraciones de dilución (`post.penetrations`). El comparador
correcto para ESA medida es la **línea central de temperatura** de
Kamotani & Greber (NASA TN D-6966, 1972, ec. (3))::

    Z_c/D = 0.76·(ρ_j/ρ_∞)^0.15 · J^0.52 · (x/D)^0.27

y NO la de Pratte & Baines (1967) — esa es la línea central visual/de
momento, que va sistemáticamente MÁS ALTA que la térmica (la primera
corrida de este script comparaba contra P&B y daba −35 %: correlación
equivocada, no solver roto; se deja P&B como contexto en la salida).
Criterio: **±25 % vs K&G en x/(r·d) ∈ [1, 4]** — verifica TENDENCIA con
ρ variable + turbulencia + escalar juntos, no precisión fina (la banda de
los propios datos de K&G es ~±15–20 % y el RANS k-ε añade lo suyo).

Malla y coste: con --dx-inv 4 son ~184k celdas ⇒ el backend `auto` elige
GPU (por encima de `GPU_MIN_CELLS`). En la RTX 3070 Ti de esta máquina:
ver el número medido en ESTADO.md.

Uso::

    python cfd/tests/run_v5_jcf.py                    # dx = d/4, GPU
    python cfd/tests/run_v5_jcf.py --dx-inv 3 --max-iter 8000   # humo
    python cfd/tests/run_v5_jcf.py --json out.json
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

from cfd.ibrans.backend import to_numpy                       # noqa: E402
from cfd.ibrans.energy import (EnergyConfig, EnergyLowMach,   # noqa: E402
                               PassiveScalar, ScalarConfig)
from cfd.ibrans.mesh import CartesianMesh                     # noqa: E402
from cfd.ibrans.solver import (IBRANSSolver, SolverConfig,    # noqa: E402
                               make_bcs)
from cfd.ibrans.turbulence import (C_MU, KEpsilonRealizable,  # noqa: E402
                                   TurbulenceConfig)

#: Kamotani & Greber (NASA TN D-6966, 1972), ec. (3): línea central de
#: TEMPERATURA Z_c/D = KG_A·(ρj/ρ∞)^KG_RHO·J^KG_J·(x/D)^KG_M — el
#: comparador correcto para una trayectoria medida por el máximo de T.
KG_A = 0.76
KG_RHO = 0.15
KG_J = 0.52
KG_M = 0.27
#: Pratte & Baines (1967) (línea central visual/de momento, MÁS ALTA que
#: la térmica) — sólo como contexto en la salida.
PB_A = 2.05
PB_M = 0.28
TOL_TRAJ = 0.25


def run_jcf(dx_inv: float = 4.0, r_eff: float = 4.0, t_jet: float = 600.0,
            t_inf: float = 300.0, max_iter: int = 15_000,
            timeout_s: float = 3600.0) -> dict:
    d = 1.0
    u_inf = 1.0
    # r = sqrt(rho_j u_j^2 / rho_inf u_inf^2), rho_j/rho_inf = T_inf/T_j
    u_j = r_eff * u_inf * (t_jet / t_inf) ** 0.5
    nu = u_inf * d / 20_000.0
    dx = d / dx_inv
    lox, hix = -4.0, 16.0
    hiy, hiz = 12.0, 6.0
    m = CartesianMesh([lox, 0.0, -hiz], [hix, hiy, hiz], dx)
    m.set_sdf(None)
    # ranura circular del chorro en la pared inferior (perfil por celda)
    xg, zg = np.meshgrid(m.xc[0], m.xc[2], indexing="ij")
    rr = np.sqrt(xg ** 2 + zg ** 2)
    jet_v = np.where(rr <= 0.5 * d, u_j, 0.0)
    # perfiles por celda también para T, Z, k y ε: fuera de la ranura la
    # cara inferior queda a T_inf, Z=0 y k ambiente (sin suelo caliente
    # espurio). La k de la RANURA escala con u_j (I = 10 %, ℓ = 0.07·d —
    # salida de tobera turbulenta): con la k del cruzado (u_inf) el chorro
    # entraba casi laminar, se mezclaba poco y sobre-penetraba +26 % vs
    # K&G — medido en las corridas d/4 y d/6 del 2026-08-02.
    t_prof = np.where(rr <= 0.5 * d, t_jet, t_inf)
    z_prof = np.where(rr <= 0.5 * d, 1.0, 0.0)
    k_in = 1.5 * (0.05 * u_inf) ** 2
    eps_in = C_MU ** 0.75 * k_in ** 1.5 / (0.07 * d)
    k_slot = 1.5 * (0.10 * u_j) ** 2
    eps_slot = C_MU ** 0.75 * k_slot ** 1.5 / (0.07 * d)
    k_prof = np.where(rr <= 0.5 * d, k_slot, k_in)
    e_prof = np.where(rr <= 0.5 * d, eps_slot, eps_in)
    cfg = SolverConfig(rho=1.2, nu=nu, u_ref=u_j, l_ref=d, mach_art=0.2,
                       cfl=0.4, rk="rk2", max_iter=max_iter, tol=5.0e-6,
                       residual_every=200, timeout_s=timeout_s)
    bcs = make_bcs(
        xlo=dict(kind="inlet", u=(u_inf, 0.0, 0.0), k=k_in, eps=eps_in),
        xhi=dict(kind="outlet", p=0.0),
        # la pared inferior es "inlet" con perfil: chorro en la ranura y
        # u = 0 fuera (equivale a pared no deslizante con inyección)
        ylo=dict(kind="inlet", u=(0.0, jet_v, 0.0), k=k_prof, eps=e_prof,
                 T=t_prof, Z=z_prof),
        yhi="slip", zlo="slip", zhi="slip")
    s = IBRANSSolver(m, cfg, bcs)
    tur = KEpsilonRealizable(m, TurbulenceConfig(
        intensity=0.05, length_scale_m=0.07 * d)).attach(s)
    en = EnergyLowMach(m, EnergyConfig(T_inlet_K=t_inf)).attach(
        s, turbulence=tur)
    sc = PassiveScalar(m, ScalarConfig(key="Z")).attach(s, turbulence=tur)
    s.u[0][...] = u_inf
    s.apply_bc()
    t0 = time.perf_counter()
    info = s.run()
    el = time.perf_counter() - t0
    T = np.asarray(to_numpy(en.T[m.interior]), dtype=float)
    kz = T.shape[2] // 2
    tc = T[:, :, kz]                        # plano de simetría z=0
    xc, yc = s.coords()[0], s.coords()[1]
    rows = []
    rd = r_eff * d
    jmom = r_eff ** 2
    rho_ratio = t_inf / t_jet
    for xq in np.arange(0.5 * rd, 6.5 * rd, 0.5 * rd):
        i = int(round((xq - lox) / dx - 0.5))
        if i < 0 or i >= tc.shape[0]:
            continue
        j = int(np.argmax(tc[i, :]))
        if tc[i, j] < t_inf + 5.0:
            continue
        z_kg = KG_A * rho_ratio ** KG_RHO * jmom ** KG_J \
            * (xq / d) ** KG_M
        rows.append((xq / rd, yc[j] / rd, z_kg / rd))
    rows = np.asarray(rows)
    errs = [yr / zk - 1.0 for xr, yr, zk in rows if 1.0 <= xr <= 4.0]
    err_max = float(np.max(np.abs(errs))) if errs else float("nan")
    return dict(case=f"JICF r={r_eff} dx=d/{dx_inv:.0f}",
                status=info["status"], iterations=info["iterations"],
                res_u=info["res_u"], wall_time_s=el,
                s_per_cell_step=info["s_per_cell_step"],
                cells=info["cells"], backend=info["backend"]["backend"],
                u_j=u_j, r_eff=r_eff, J=jmom,
                trajectory=[[float(a), float(b), float(c)]
                            for a, b, c in rows],
                err_traj_max=err_max,
                T_max=float(T.max()), T_min=float(T.min()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="JICF caliente manual (M2, tarea 2.6)")
    ap.add_argument("--dx-inv", type=float, default=4.0)
    ap.add_argument("--r", type=float, default=4.0,
                    help="razon de velocidades efectiva")
    ap.add_argument("--max-iter", type=int, default=15_000)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    print(f"JICF caliente · r={a.r}, dx=d/{a.dx_inv:.0f} — 3D, puede tardar "
          "decenas de minutos (GPU si la malla supera GPU_MIN_CELLS)")
    r = run_jcf(dx_inv=a.dx_inv, r_eff=a.r, max_iter=a.max_iter,
                timeout_s=a.timeout_s)
    ok = np.isfinite(r["err_traj_max"]) and r["err_traj_max"] < TOL_TRAJ
    print(f"  malla        : {r['cells']} celdas (backend {r['backend']})")
    print(f"  estado       : {r['status']}  ({r['iterations']} iter, "
          f"{r['wall_time_s']:.1f} s, res_u {r['res_u']:.2e})")
    print(f"  rendimiento  : {r['s_per_cell_step']:.3e} s/celda/paso")
    print(f"  T            : [{r['T_min']:.0f}, {r['T_max']:.0f}] K")
    print("  trayectoria del maximo de T (x/rd, y/rd) vs Kamotani&Greber "
          "(TN D-6966 ec.3) [P&B como contexto]:")
    rd = r["r_eff"]
    for xr, yr, zk in r["trajectory"]:
        pb = PB_A * xr ** PB_M
        print(f"    x/rd={xr:5.2f}  y/rd={yr:5.2f}  K&G={zk:5.2f}  "
              f"(P&B={pb:5.2f})")
    print(f"  desviacion max vs K&G: {100 * r['err_traj_max']:.1f} %  → "
          f"{'PASA' if ok else 'NO PASA'} (umbral ±{100 * TOL_TRAJ:.0f} % "
          "en x/rd ∈ [1,4])")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(r, passed=bool(ok)), fh, indent=2,
                      ensure_ascii=False)
        print(f"  JSON         : {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
