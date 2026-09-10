#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/run_m4_figures.py
===========================================
**Script MANUAL de FIGURAS del caso M4** (WAVE6 §4: «export de cortes PNG
para el reporte»). Monta el caso ancla, lo corre y vuelca los cortes que
un combustor pide mirar:

    1. corte meridional (φ = 0): T, |u|, Y_FUEL y liberación de calor
    2. plano 4: mapa de T (el que produce el PF) + perfil radial
    3. mapa (ζ, φ) de pared ADIABÁTICA — ranking, NO valor absoluto

Todo lo que dibuja sale del MISMO caso que produce los números de
`combustor_product`, así que la figura y el número no pueden divergir.

matplotlib es OPCIONAL en Phy-CB (contrato §10.4): sin él, el script
guarda igualmente los cortes en `.npz` y lo dice — no falla.

Uso::

    python cfd/tests/run_m4_figures.py                     # reactivo
    python cfd/tests/run_m4_figures.py --cold --dx-mm 2.5  # frio, rapido
    python cfd/tests/run_m4_figures.py --outdir runs/m4_kj66
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
import geometry_generator as gg                               # noqa: E402

pc.assert_provenance(pc, gg)

from cfd.ibrans import post                                   # noqa: E402
from cfd.ibrans.backend import to_numpy                       # noqa: E402
from cfd.ibrans.case import (CaseConfig, build_case,          # noqa: E402
                             measure_splits, run_case,
                             swirl_number_measured)

_MM = 1.0e-3


def _plt():
    """matplotlib o `None` (dependencia OPCIONAL, contrato §10.4)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def extract_slices(s) -> dict:
    """Cortes del caso en coordenadas de RECORRIDO (ζ desde el domo, r con
    signo en el plano φ = 0), en NumPy float64. Es post-proceso: baja del
    dispositivo una vez y no toca el bucle."""
    m, m4 = s.mesh, s.m4
    sign = m4["flow_sign"]
    I = m.interior
    itp = m4["interp"]

    def _dn(a):
        return np.asarray(to_numpy(a[I]), dtype=np.float64)

    ux, uy, uz = (_dn(s.u[c]) for c in range(3))
    p = _dn(s.p)
    fl = m.fluid[I]
    xs = m.xc[0][I[0]]
    ys = m.xc[1][I[1]]
    zs = m.xc[2][I[2]]
    jy = int(np.argmin(np.abs(ys)))          # plano φ = 0 (y ≈ 0)
    zeta = sign * (zs - m4["z_dome_m"])      # recorrido desde el domo
    order = np.argsort(zeta)
    umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    def _mer(a):
        """Corte meridional (x, ζ) enmascarado fuera del fluido."""
        sl = a[:, jy, :][:, order]
        msk = fl[:, jy, :][:, order]
        return np.where(msk, sl, np.nan)

    out = dict(x_mm=xs / _MM, zeta_mm=zeta[order] / _MM,
               umag=_mer(umag), p=_mer(p), fluid=fl[:, jy, :][:, order],
               r_out_mm=np.interp(zs[order], itp["z"], itp["r_out"]) / _MM,
               r_in_mm=np.interp(zs[order], itp["z"], itp["r_in"]) / _MM)
    en = (m4["models"] or {}).get("energy")
    sp = (m4["models"] or {}).get("species")
    if en is not None and en.T is not None:
        out["T"] = _mer(_dn(en.T))
    if sp is not None:
        out["Y_FUEL"] = _mer(_dn(sp.Y["FUEL"]))
        out["Y_O2"] = _mer(_dn(sp.Y["O2"]))
        if sp._omega_prev is not None:
            om = np.asarray(to_numpy(sp._omega_prev), dtype=np.float64)
            out["hrr"] = _mer(om * float(sp.mech["lhv_J_kg"]))
    # plano 4
    k4 = m.shape[2] - 2 if sign > 0 else 1
    if en is not None and en.T is not None:
        T4 = np.asarray(to_numpy(en.T[I]), dtype=np.float64)[:, :, k4]
        liner4 = post._liner_mask_interior(s)[:, :, k4]
        out["plane4_T"] = np.where(liner4, T4, np.nan)
        out["plane4_x_mm"] = xs / _MM
        out["plane4_y_mm"] = ys / _MM
    return out


def make_figures(s, cuts: dict, outdir: str, label: str) -> list[str]:
    """Dibuja los cortes. Devuelve la lista de PNG escritos (vacía si no
    hay matplotlib — se dice, no se falla)."""
    plt = _plt()
    if plt is None:
        return []
    os.makedirs(outdir, exist_ok=True)
    files = []
    ext = [cuts["zeta_mm"][0], cuts["zeta_mm"][-1],
           cuts["x_mm"][0], cuts["x_mm"][-1]]

    panels = [("T", "Temperatura [K]", "inferno"),
              ("umag", "|u| [m/s]", "viridis"),
              ("Y_FUEL", "Y combustible [-]", "magma"),
              ("hrr", "Liberacion de calor [W/m3]", "hot")]
    panels = [q for q in panels if q[0] in cuts]
    if panels:
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.6 * len(panels)),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, (key, ttl, cmap) in zip(axes, panels):
            # los cortes son (nx, nz): filas = r (vertical), columnas = zeta
            # (horizontal), que es justo el orden que `extent` declara. NO
            # se transpone — hacerlo cruza los ejes y el anillo aparece
            # como un hueco axial que no existe (bug cazado el 2026-08-03).
            im = ax.imshow(cuts[key], origin="lower", extent=ext,
                           aspect="auto", cmap=cmap)
            ax.plot(cuts["zeta_mm"], cuts["r_out_mm"], "w--", lw=0.8)
            ax.plot(cuts["zeta_mm"], -cuts["r_out_mm"], "w--", lw=0.8)
            ax.set_ylabel("r [mm]")
            ax.set_title(ttl, fontsize=10)
            fig.colorbar(im, ax=ax, pad=0.01)
        axes[-1].set_xlabel("zeta desde el domo [mm]  (recorrido del gas)")
        fig.suptitle(f"Phy-CB L2a · corte meridional · {label}", fontsize=11)
        f = os.path.join(outdir, "m4_meridional.png")
        fig.savefig(f, dpi=130)
        plt.close(fig)
        files.append(f)

    if "plane4_T" in cuts:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2),
                                     constrained_layout=True)
        e4 = [cuts["plane4_x_mm"][0], cuts["plane4_x_mm"][-1],
              cuts["plane4_y_mm"][0], cuts["plane4_y_mm"][-1]]
        im = a1.imshow(cuts["plane4_T"].T, origin="lower", extent=e4,
                       cmap="inferno")
        a1.set_title("Plano 4 · T [K] (de aqui sale el PF)", fontsize=10)
        a1.set_xlabel("x [mm]")
        a1.set_ylabel("y [mm]")
        a1.set_aspect("equal")
        fig.colorbar(im, ax=a1, pad=0.01)
        t4 = post.T4_and_pattern_factor(s)
        if t4.get("ok"):
            prof = np.asarray(t4["radial_profile_K"], dtype=float)
            a2.plot(np.linspace(0, 1, prof.size), prof, "o-")
            a2.axhline(t4["T4_cfd_K"], ls="--", c="k",
                       label=f"T4 media {t4['T4_cfd_K']:.0f} K")
            a2.set_xlabel("posicion radial normalizada [-]")
            a2.set_ylabel("T media masica [K]")
            a2.set_title(f"Perfil radial · PF = {t4['PF_cfd']:.3f}, "
                         f"RTDF = {t4['RTDF_cfd']:.3f}", fontsize=10)
            a2.legend(fontsize=8)
            a2.grid(alpha=0.3)
        f = os.path.join(outdir, "m4_plane4.png")
        fig.savefig(f, dpi=130)
        plt.close(fig)
        files.append(f)

    wall = post.adiabatic_wall_map(s)
    if wall.get("ok"):
        grid = np.array([[np.nan if v is None else v for v in row]
                         for row in wall["map_zeta_phi_K"]], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
        im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="inferno",
                       extent=[0, float(s.m4["meta"]["L_tot_mm"]), -180, 180])
        ax.set_xlabel("zeta desde el domo [mm]")
        ax.set_ylabel("phi [deg]")
        ax.set_title("Pared ADIABATICA · T [K] — RANKING, no valor absoluto "
                     "(contrato §12)", fontsize=10)
        fig.colorbar(im, ax=ax, pad=0.01)
        f = os.path.join(outdir, "m4_wall_map.png")
        fig.savefig(f, dpi=130)
        plt.close(fig)
        files.append(f)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="figuras del caso M4 (manual)")
    ap.add_argument("--dx-mm", type=float, default=2.0)
    ap.add_argument("--max-iter", type=int, default=12_000)
    ap.add_argument("--timeout-s", type=float, default=2400.0)
    ap.add_argument("--cold", action="store_true",
                    help="caso frio (sin combustion): mas rapido")
    ap.add_argument("--outdir", default=os.path.join(_ROOT, "runs",
                                                     "m4_kj66"))
    a = ap.parse_args(argv)
    rec = pc.evaluate(pc.THETA_REF)
    c = gg.build_contract(pc.THETA_REF, rec)
    cfg = CaseConfig(dx_mm=a.dx_mm, reactive=not a.cold,
                     ignition=not a.cold, max_iter=a.max_iter,
                     timeout_s=a.timeout_s,
                     t_ad_K=float(rec.get("T_ad_st_K") or 0) or None)
    s = build_case(c, cfg)
    print(f"caso {'FRIO' if a.cold else 'REACTIVO'} · {s.mesh.ncell} celdas "
          f"(dx = {a.dx_mm} mm) — corriendo...", flush=True)
    t0 = time.perf_counter()
    info = run_case(s)
    el = time.perf_counter() - t0
    print(f"  {info['status']} · {info['iterations']} it · {el / 60:.1f} min"
          f" · anclada={info.get('anchored')}", flush=True)
    cuts = extract_slices(s)
    os.makedirs(a.outdir, exist_ok=True)
    npz = os.path.join(a.outdir, "m4_slices.npz")
    np.savez_compressed(npz, **{k: v for k, v in cuts.items()
                                if isinstance(v, np.ndarray)})
    files = make_figures(s, cuts, a.outdir,
                         f"KJ66 {'frio' if a.cold else 'reactivo'} "
                         f"dx={a.dx_mm} mm")
    prod = post.combustor_product(s, pc.THETA_REF, rec, c)
    sm = measure_splits(s)
    meta = dict(
        case="KJ66" + (" frio" if a.cold else " reactivo"),
        dx_mm=a.dx_mm, cells=int(s.mesh.ncell),
        cells_fluid=int(s.mesh.summary()["cells_fluid"]),
        status=info["status"], iterations=int(info["iterations"]),
        wall_min=el / 60.0, anchored=bool(info.get("anchored", False)),
        heat_release_W=info.get("heat_release_W"),
        backend=(info.get("backend") or {}).get("backend"),
        splits_target=sm["splits_target_CdA"],
        splits_measured=sm["splits_measured"],
        mass_closure_rel=sm["mass_closure_rel"],
        swirl=swirl_number_measured(s),
        product={k: v for k, v in prod.items()
                 if isinstance(v, (int, float, str, list, dict))
                 and k not in ("wall_map", "cfd_mesh")},
        record_refs=dict(dp_over_p=float(rec["dp_over_p"]),
                         T04_K=float(rec["T04_K"]), PF=float(rec["PF"])),
        mesh_meta=s.m4["meta"], figures=[os.path.basename(f) for f in files])
    jf = os.path.join(a.outdir, "m4_figures.json")
    with open(jf, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    print(f"  cortes  : {npz}")
    print(f"  meta    : {jf}")
    for f in files:
        print(f"  figura  : {f}")
    if not files:
        print("  (matplotlib ausente: solo .npz — dependencia opcional)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
