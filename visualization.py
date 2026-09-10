"""
QUASAR Phy-CB · visualization.py
================================
Capa 5b (opcional): figuras matplotlib del diseño de combustor.

REGLA DE LA FAMILIA: matplotlib es una dependencia OPCIONAL. Este módulo
degrada en SILENCIO — si no está instalado, todas las funciones devuelven
`[]` y el reporte HTML sigue siendo completo (sus SVG los dibuja
`report_generator.py` en Python puro, sin dependencias). Nunca lanza.

Cada función `fig_*` devuelve la lista de rutas escritas, para que el
llamante componga el bloque ENTREGABLES.

Convenio: mm en los ejes de geometría (es una interfaz), SI en el resto.
Textos de las figuras en español (van al reporte, `lang="es"`).
"""

from __future__ import annotations

import math
import os

import numpy as np

_MPL = None
_MPL_TRIED = False


def _mpl():
    """Import perezoso de matplotlib con backend no interactivo."""
    global _MPL, _MPL_TRIED
    if _MPL_TRIED:
        return _MPL
    _MPL_TRIED = True
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _MPL = plt
    except Exception:
        _MPL = None
    return _MPL


def available() -> bool:
    """¿Se pueden generar figuras? La CLI lo consulta UNA vez, al arranque,
    para avisar en voz alta (patrón de degradación de la familia)."""
    return _mpl() is not None


# --- paleta coherente con cli_style / report_generator -------------------
_C_BRAND = "#2E6FF2"
_C_ACCENT = "#F2A73B"
_C_HOT = "#D6453C"
_C_COOL = "#3BB0A8"
_C_GRID = "#D8DCE3"


def _finish(fig, ax_list, path: str, plt) -> list[str]:
    for ax in ax_list:
        ax.grid(True, color=_C_GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=140)
    except Exception:
        plt.close(fig)
        return []
    plt.close(fig)
    return [path]


def fig_air_split(record: dict, outdir: str) -> list[str]:
    """Reparto de aire por zona (barras apiladas + φ de zona)."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        sp = record.get("splits") or {}
        keys = ["pz", "sz", "dz", "cool"]
        vals = [float(sp.get(k, 0.0)) for k in keys]
        if not any(vals):
            return []
        labels = ["Primaria", "Secundaria", "Dilución", "Refrigeración"]
        colors = [_C_HOT, _C_ACCENT, _C_BRAND, _C_COOL]
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        left = 0.0
        for v, lab, c in zip(vals, labels, colors):
            ax.barh(0, v, left=left, color=c, edgecolor="white", lw=1.2)
            if v > 0.05:
                ax.text(left + v / 2, 0, f"{lab}\n{100 * v:.1f} %",
                        ha="center", va="center", fontsize=8.5,
                        color="white", weight="bold")
            left += v
        ax.set_xlim(0, max(left, 1.0))
        ax.set_yticks([])
        ax.set_xlabel("fracción del caudal de aire ṁ₃  [-]")
        phis = (f"φ_PZ = {record.get('phi_pz', float('nan')):.2f}   "
                f"φ_SZ = {record.get('phi_sz', float('nan')):.2f}   "
                f"φ_global = {record.get('phi_global', float('nan')):.3f}")
        ax.set_title(f"Reparto de aire · {phis}", fontsize=10)
        return _finish(fig, [ax], os.path.join(outdir, "air_split.png"), plt)
    except Exception:
        return []


def fig_liner_thermal(record: dict, outdir: str,
                      material_limit_K: float | None = None) -> list[str]:
    """Perfil axial de temperatura de pared T_w1 (llama) y T_w2 (anillo)."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        prof = record.get("T_wall_profile") or []
        if len(prof) < 2:
            return []
        p = np.asarray(prof, dtype=float)
        z, t1, t2 = p[:, 0], p[:, 1], p[:, 2]
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.plot(z, t1, color=_C_HOT, lw=2.0, label="T_w1 (cara de llama)")
        ax.plot(z, t2, color=_C_COOL, lw=2.0, label="T_w2 (cara de anillo)")
        t3 = record.get("T3_K")
        if t3:
            ax.axhline(float(t3), color=_C_GRID, lw=1.4, ls=":",
                       label=f"T₃ = {float(t3):.0f} K")
        if material_limit_K:
            ax.axhline(float(material_limit_K), color=_C_ACCENT, lw=1.6,
                       ls="--", label=f"límite material "
                                      f"{float(material_limit_K):.0f} K")
        ax.set_xlabel("z  [mm]")
        ax.set_ylabel("T de pared  [K]")
        ax.set_title("Térmico 1D del liner", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        return _finish(fig, [ax], os.path.join(outdir, "liner_thermal.png"),
                       plt)
    except Exception:
        return []


def fig_emissions_fan(record: dict, outdir: str) -> list[str]:
    """El ABANICO de correlaciones de emisiones: el usuario ve la dispersión
    entre modelos, no un número único (regla de honestidad de Ciencia §8)."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        names = [("EI_NOx_lefebvre", "Lefebvre"),
                 ("EI_NOx_rizk_mongia", "Rizk & Mongia"),
                 ("EI_NOx_zonal", "Rizk & Mongia zonal"),
                 ("EI_NOx_odgers", "Odgers & Kretschmer"),
                 ("EI_NOx_laviolette", "Laviolette & Perez"),
                 ("EI_NOx_crn", "CRN (L1)")]
        vals, labs = [], []
        for k, lab in names:
            v = record.get(k)
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))
                labs.append(lab)
        if len(vals) < 2:
            return []
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        y = np.arange(len(vals))
        ax.barh(y, vals, color=_C_BRAND, alpha=0.85, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("EI_NOx  [g / kg de combustible]")
        lo, hi = min(vals), max(vals)
        ax.set_title(
            f"Abanico de correlaciones de NOx · dispersión ×{hi / max(lo, 1e-9):.1f}"
            "  — TENDENCIA, no valor certificable", fontsize=9.5)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
        return _finish(fig, [ax], os.path.join(outdir, "emissions_fan.png"),
                       plt)
    except Exception:
        return []


def fig_pareto(front: list[dict], outdir: str) -> list[str]:
    """Frente de Pareto verificado: η_comb vs EI_NOx, factibles resaltados."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        pts = [p for p in (front or [])
               if p.get("eta_comb") is not None
               and p.get("EI_NOx") is not None]
        if len(pts) < 2:
            return []
        eta = np.array([float(p["eta_comb"]) for p in pts])
        nox = np.array([float(p["EI_NOx"]) for p in pts])
        feas = np.array([bool(p.get("feasible")) for p in pts])
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if (~feas).any():
            ax.scatter(nox[~feas], eta[~feas], s=34, facecolors="none",
                       edgecolors=_C_GRID, lw=1.2, label="infactible")
        if feas.any():
            ax.scatter(nox[feas], eta[feas], s=48, color=_C_BRAND,
                       label="factible")
            o = np.argsort(nox[feas])
            ax.plot(nox[feas][o], eta[feas][o], color=_C_BRAND, lw=1.0,
                    alpha=0.5)
        ax.set_xlabel("EI_NOx  [g/kg]  (minimizar)")
        ax.set_ylabel("η_comb  [-]  (maximizar)")
        ax.set_title("Frente de Pareto verificado", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        return _finish(fig, [ax], os.path.join(outdir, "pareto.png"), plt)
    except Exception:
        return []


def fig_liner_profile(contract: dict, outdir: str) -> list[str]:
    """Corte meridional: perfil del liner y de la carcasa + filas de
    orificios (Z = eje, el gas va a +Z)."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        liner = (contract or {}).get("liner", {}) or {}
        prof = np.asarray(liner.get("profile") or [], dtype=float)
        if prof.ndim != 2 or prof.shape[0] < 2:
            return []
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        z, ri, ro = prof[:, 0], prof[:, 1], prof[:, 2]
        for s in (1.0, -1.0):
            ax.plot(z, s * ri, color=_C_HOT, lw=1.8)
            ax.plot(z, s * ro, color=_C_HOT, lw=1.8)
        cas = np.asarray(((contract.get("casing") or {}).get("profile")
                          or []), dtype=float)
        if cas.ndim == 2 and cas.shape[0] >= 2:
            for s in (1.0, -1.0):
                ax.plot(cas[:, 0], s * cas[:, 1], color=_C_BRAND, lw=1.6,
                        ls="--")
        for row in (contract.get("hole_rows") or []):
            try:
                zr = float(row.get("z_mm", 0.0))
                rr = float(row.get("r_mm", np.interp(zr, z, ro)))
                d = float(row.get("d_mm", 1.0))
                for s in (1.0, -1.0):
                    ax.plot([zr], [s * rr], marker="o",
                            ms=max(2.5, min(9.0, d)), color=_C_ACCENT)
            except Exception:
                continue
        ax.set_xlabel("z  [mm]   (eje de simetría; el gas fluye +Z)")
        ax.set_ylabel("r  [mm]")
        ax.set_aspect("equal", adjustable="datalim")
        arch = contract.get("arch", "")
        ax.set_title(f"Corte meridional · {arch}", fontsize=10)
        return _finish(fig, [ax], os.path.join(outdir, "liner_profile.png"),
                       plt)
    except Exception:
        return []


# ==========================================================================
# CFD — visualización de L1.5 (LBM frío) y L2a (IB-RANS nativo)
# ==========================================================================
# El campo que consumen estas funciones es el dict de
# `cfd.xlb_cold.solve_cold(contract)`: `rho` (nx,ny,nz), `j` (3,nx,ny,nz) en
# unidades de red, `dom` (máscaras, coordenadas, radios) y `units` (escalas
# de vuelta a SI). Degradan como el resto del módulo: sin matplotlib, o con
# un campo incompleto, devuelven [].
#
# CONVENIO: Z es el eje de simetría y el gas va a +Z (CONTEXTO §8). El corte
# meridional es el plano y ≈ 0, que da (z, x) ≡ (z, ±r) — el mismo corte que
# `fig_liner_profile`, para poder superponerlos mentalmente.
#
# REGLA DE HONESTIDAD: ninguna figura de campo se publica sin su pie de
# procedencia (`_cfd_caption`). Un solver que no ha convergido, o una malla
# que no resuelve el anillo, produce figuras igual de bonitas — y ese es
# exactamente el modo de fallo peligroso.

_C_SOLID = "#8A9099"


def _cfd_arrays(field: dict):
    """(u_ms, fluido, z_mm, x_mm) del corte meridional. None si no se puede."""
    dom = field.get("dom") or {}
    rho, j, fluid = field.get("rho"), field.get("j"), dom.get("fluid")
    if rho is None or j is None or fluid is None:
        return None
    rho, j = np.asarray(rho), np.asarray(j)
    fluid = np.asarray(fluid, dtype=bool)
    if j.ndim != 4 or rho.ndim != 3 or fluid.shape != rho.shape:
        return None
    u_scale = float((field.get("units") or {}).get("u_scale", 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = j / np.maximum(rho, 1e-12)[None, ...] * u_scale
    u = np.where(fluid[None, ...], u, np.nan)     # en el sólido no hay u

    jy = u.shape[2] // 2                          # plano meridional y ≈ 0
    dx = float(dom.get("dx_mm", 1.0))
    org = dom.get("origin_mm") or (0.0, 0.0, 0.0)
    nx, nz = u.shape[1], u.shape[3]
    x_mm = float(org[0]) + (np.arange(nx) + 0.5) * dx
    z_mm = float(org[2]) + (np.arange(nz) + 0.5) * dx
    return u[:, :, jy, :], fluid[:, jy, :], z_mm, x_mm


def _cfd_caption(field: dict) -> str:
    """Pie con procedencia y calidad de malla. Va en TODAS las figuras."""
    m = field.get("cfd_mesh") or {}
    u = field.get("units") or {}
    conv = "convergido" if m.get("converged") else "NO convergido"
    txt = (f"{field.get('source', '?')} · dx = {float(m.get('dx_mm', 0)):.2f} mm"
           f" · {int(m.get('n_fluid', 0)):,} celdas fluidas · {conv}")
    tau, tau_m = u.get("tau"), u.get("tau_molecular")
    if tau and tau_m and u.get("tau_floored"):
        ratio = (float(tau) - 0.5) / max(float(tau_m) - 0.5, 1e-12)
        txt += (f"\nτ suelado {float(tau):.4f} vs molecular {float(tau_m):.6f}"
                f" ⇒ ν efectiva ×{ratio:.0f} — resolución insuficiente, el "
                "campo NO es cuantitativo (sirve para topología)")
    return txt


def _draw_solid(ax, z, x, fluid) -> None:
    """Pinta el sólido en gris sobre el corte."""
    ax.contourf(z, x, (~fluid).astype(float), levels=[0.5, 1.5],
                colors=[_C_SOLID], alpha=0.92)


def _cfd_finish(fig, axes, path, plt, field):
    """Cierra una figura de CFD reservando sitio para el pie.

    NO puede delegar en `_finish`: aquel llama a `tight_layout()` sin `rect`,
    que anularía la banda inferior y dejaría el pie encima de la etiqueta del
    eje x. El resto (rejilla, guardado, cierre) se replica aquí.
    """
    for ax in axes:
        ax.grid(True, color=_C_GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
    cap = _cfd_caption(field)
    band = 0.075 + 0.035 * cap.count("\n")        # una banda por línea
    fig.tight_layout(rect=(0, band, 1, 1))
    fig.text(0.008, 0.006, cap, fontsize=7, va="bottom", color="#666")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=140)
    except Exception:
        plt.close(fig)
        return []
    plt.close(fig)
    return [path]


def fig_cfd_velocity(field: dict, outdir: str) -> list[str]:
    """Corte meridional del módulo de velocidad, con el sólido en gris."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        got = _cfd_arrays(field)
        if got is None:
            return []
        u, fluid, z, x = got
        umag = np.sqrt(np.nansum(u ** 2, axis=0))
        fig, ax = plt.subplots(figsize=(9.2, 4.0))
        pc = ax.pcolormesh(z, x, umag, shading="auto", cmap="viridis")
        _draw_solid(ax, z, x, fluid)
        fig.colorbar(pc, ax=ax, pad=0.02).set_label("|u|  [m/s]")
        ax.set_xlabel("z  [mm]   (eje de simetría; el gas fluye +Z)")
        ax.set_ylabel("x ≡ ±r  [mm]")
        ax.set_title("L1.5 · campo frío: módulo de velocidad "
                     "(corte meridional)", fontsize=10)
        return _cfd_finish(fig, [ax], os.path.join(outdir, "cfd_velocity.png"),
                           plt, field)
    except Exception:
        return []


def fig_cfd_recirculation(field: dict, outdir: str) -> list[str]:
    """Zonas de recirculación (u_z < 0). Es la CTRZ que el swirler debe
    generar con S_N > 0.6 (Ciencia §4.4) y la que define la PZ del CRN."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        got = _cfd_arrays(field)
        if got is None:
            return []
        u, fluid, z, x = got
        uz = u[2]
        lim = float(np.nanpercentile(np.abs(uz), 98)) or 1.0
        fig, ax = plt.subplots(figsize=(9.2, 4.0))
        pc = ax.pcolormesh(z, x, uz, shading="auto", cmap="RdBu_r",
                           vmin=-lim, vmax=lim)
        ax.contour(z, x, np.nan_to_num(uz), levels=[0.0], colors="k",
                   linewidths=0.7)
        _draw_solid(ax, z, x, fluid)
        fig.colorbar(pc, ax=ax, pad=0.02).set_label(
            "u_z  [m/s]   (azul = contracorriente)")
        frac = (float(np.nanmean(uz[fluid] < 0.0)) if fluid.any() else 0.0)
        ax.set_xlabel("z  [mm]")
        ax.set_ylabel("x ≡ ±r  [mm]")
        ax.set_title(f"L1.5 · recirculación: u_z < 0 en el "
                     f"{100 * frac:.1f} % del fluido — la isolínea negra es "
                     "u_z = 0", fontsize=10)
        return _cfd_finish(fig, [ax],
                           os.path.join(outdir, "cfd_recirculation.png"),
                           plt, field)
    except Exception:
        return []


def fig_cfd_streamlines(field: dict, outdir: str) -> list[str]:
    """Líneas de corriente del corte meridional: el recorrido real del gas
    (en anular inverso, los dos giros de 180°) y dónde se despega."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        got = _cfd_arrays(field)
        if got is None:
            return []
        u, fluid, z, x = got
        uz, ux = np.nan_to_num(u[2]), np.nan_to_num(u[0])
        fig, ax = plt.subplots(figsize=(9.2, 4.0))
        ax.pcolormesh(z, x, np.hypot(uz, ux), shading="auto", cmap="Blues",
                      alpha=0.6)
        # streamplot exige rejilla regular y (fila, columna) = (x, z)
        ax.streamplot(z, x, uz, ux, density=1.4, linewidth=0.7,
                      color="#1b3a6b", arrowsize=0.7)
        _draw_solid(ax, z, x, fluid)
        ax.set_xlim(z[0], z[-1])
        ax.set_ylim(x[0], x[-1])
        ax.set_xlabel("z  [mm]")
        ax.set_ylabel("x ≡ ±r  [mm]")
        ax.set_title("L1.5 · líneas de corriente del campo frío", fontsize=10)
        return _cfd_finish(fig, [ax],
                           os.path.join(outdir, "cfd_streamlines.png"),
                           plt, field)
    except Exception:
        return []


def fig_cfd_splits(field: dict, post: dict | None, record: dict | None,
                   outdir: str) -> list[str]:
    """EL cierre de bucle de fidelidades (WAVE5 §C.2): el mismo reparto de
    aire visto por la correlación (L0), por la red 1D (L1) y medido sobre el
    campo (L1.5). Las tres barras juntas son el argumento del sistema."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        zones = ["pz", "sz", "dz", "cool"]
        rec = record or {}
        cand = [(rec.get("splits"), "L0 · correlación", _C_GRID),
                (rec.get("splits_solved"), "L1 · red 1D", _C_ACCENT),
                ((post or {}).get("splits_measured"), "L1.5 · medido (LBM)",
                 _C_BRAND)]
        series = [([float((d or {}).get(k, 0.0)) for k in zones], lab, c)
                  for d, lab, c in cand if d]
        if len(series) < 2:
            return []
        fig, ax = plt.subplots(figsize=(7.8, 3.6))
        n, idx = len(series), np.arange(len(zones))
        w = 0.8 / n
        for i, (s, lab, c) in enumerate(series):
            ax.bar(idx + (i - (n - 1) / 2) * w, s, width=w, label=lab, color=c)
        ax.set_xticks(idx)
        ax.set_xticklabels(["Primaria", "Secundaria", "Dilución",
                            "Refrigeración"])
        ax.set_ylabel("fracción de ṁ₃  [-]")
        ax.set_title("Reparto de aire: correlación vs red 1D vs medido",
                     fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        return _cfd_finish(fig, [ax], os.path.join(outdir, "cfd_splits.png"),
                           plt, field)
    except Exception:
        return []


def fig_cfd_convergence(field: dict, outdir: str) -> list[str]:
    """Historial de ⟨|u|⟩ y celdas por rasgo. Acompaña SIEMPRE al campo:
    sin esto no se sabe si las otras figuras significan algo."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        st = field.get("stats") or {}
        hist = [float(v) for v in (st.get("u_mean_history") or [])]
        if len(hist) < 3:
            return []
        m = field.get("cfd_mesh") or {}
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.3),
                                      gridspec_kw=dict(width_ratios=[2, 1]))
        step = max(int(st.get("steps", len(hist))) // max(len(hist), 1), 1)
        ax.plot(np.arange(1, len(hist) + 1) * step, hist, color=_C_BRAND,
                lw=1.6)
        ax.set_xlabel("paso LBM")
        ax.set_ylabel("⟨|u|⟩ de red")
        ok = bool(st.get("converged"))
        ax.set_title(("convergido" if ok else "NO convergido")
                     + f" · {st.get('steps', '?')} pasos", fontsize=10,
                     color=("#1a7f37" if ok else _C_HOT))
        # celdas por rasgo: el diagnóstico que decide si el campo se puede
        # creer (Ciencia §11.2: la dilución se resuelve si d/dx ≥ 4)
        vals = [float(m.get("annulus_cells", 0.0)),
                float(m.get("d_hole_min_cells", 0.0))]
        ax2.barh(np.arange(2), vals, height=0.55,
                 color=[(_C_COOL if v >= 4.0 else _C_HOT) for v in vals])
        ax2.axvline(4.0, color="k", ls="--", lw=1.0)
        ax2.text(4.1, -0.42, "mínimo 4", fontsize=7.5)
        ax2.set_yticks(np.arange(2))
        ax2.set_yticklabels(["anillo", "Ø orificio\nmínimo"], fontsize=8)
        ax2.set_xlabel("celdas por rasgo")
        ax2.set_title("resolución", fontsize=10)
        return _cfd_finish(fig, [ax, ax2],
                           os.path.join(outdir, "cfd_convergence.png"),
                           plt, field)
    except Exception:
        return []


def fig_l2a_yplus(result: dict, outdir: str) -> list[str]:
    """Histograma de y⁺ del L2a (WAVE6 §2): las funciones de pared solo
    valen en 30 ≤ y⁺ ≤ 150; las celdas fuera son un aviso, no un detalle."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        h = (result or {}).get("yplus_hist") or {}
        edges = np.asarray(h.get("edges") or [], dtype=float)
        counts = np.asarray(h.get("counts") or [], dtype=float)
        if edges.size < 2 or counts.size != edges.size - 1:
            return []
        ctr = 0.5 * (edges[:-1] + edges[1:])
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.bar(ctr, counts, width=np.diff(edges), align="center",
               color=[(_C_COOL if 30.0 <= c <= 150.0 else _C_HOT)
                      for c in ctr])
        for v in (30.0, 150.0):
            ax.axvline(v, color="k", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("y⁺  [-]")
        ax.set_ylabel("celdas de pared")
        frac = float(counts[(ctr >= 30) & (ctr <= 150)].sum()
                     / max(counts.sum(), 1.0))
        ax.set_title(f"L2a · y⁺ en el liner — {100 * frac:.0f} % dentro de "
                     "[30, 150] (validez de las funciones de pared)",
                     fontsize=10)
        return _finish(fig, [ax], os.path.join(outdir, "cfd_yplus.png"), plt)
    except Exception:
        return []


def fig_cfd_panel(field: dict, outdir: str) -> list[str]:
    """Panel meridional de 4 campos a la vez: |u|, axial, radial y presión
    manométrica. Es la figura de lectura rápida — de un vistazo se ve si el
    chorro penetra, si el anillo alimenta y si hay gradiente de presión
    donde debe haberlo."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        got = _cfd_arrays(field)
        if got is None:
            return []
        u, fluid, z, x = got
        rho = np.asarray(field["rho"])
        jy = rho.shape[1] // 2
        p_scale = float((field.get("units") or {}).get("p_scale", 1.0))
        # p = c_s²·ρ en unidades de red; se muestra manométrica respecto a la
        # media del dominio fluido, que es lo que tiene sentido físico aquí.
        p = np.where(fluid, (rho[:, jy, :] - 1.0) / 3.0 * p_scale, np.nan)
        p = p - np.nanmean(p)

        campos = [(np.sqrt(np.nansum(u ** 2, axis=0)), "|u|  [m/s]", "viridis",
                   None),
                  (u[2], "u_z  [m/s]  (axial)", "RdBu_r", "sym"),
                  (u[0], "u_x ≈ u_r  [m/s]  (radial)", "PuOr_r", "sym"),
                  (p, "p − p̄  [Pa]", "cividis", "robust")]
        fig, axes = plt.subplots(2, 2, figsize=(14.0, 7.2))
        for ax, (dat, lab, cmap, mode) in zip(axes.ravel(), campos):
            kw = {}
            if mode == "sym":
                lim = float(np.nanpercentile(np.abs(dat), 98)) or 1.0
                kw = dict(vmin=-lim, vmax=lim)
            elif mode == "robust":
                # el pico del difusor aplastaba toda la estructura interna:
                # se recorta a [2, 98] % y la barra sigue mostrando el valor
                # real, así que no se falsea nada, solo se hace legible
                kw = dict(vmin=float(np.nanpercentile(dat, 2)),
                          vmax=float(np.nanpercentile(dat, 98)))
            pc = ax.pcolormesh(z, x, dat, shading="auto", cmap=cmap, **kw)
            _draw_solid(ax, z, x, fluid)
            fig.colorbar(pc, ax=ax, pad=0.02).set_label(lab, fontsize=8)
            ax.set_xlabel("z  [mm]", fontsize=8)
            ax.set_ylabel("x ≡ ±r  [mm]", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.suptitle("L1.5 · campo frío, corte meridional (y ≈ 0)",
                     fontsize=11)
        return _cfd_finish(fig, list(axes.ravel()),
                           os.path.join(outdir, "cfd_panel.png"), plt, field)
    except Exception:
        return []


def fig_cfd_sections(field: dict, outdir: str, n: int = 4) -> list[str]:
    """Cortes TRANSVERSALES (planos x–y) a varias estaciones z.

    Es la única vista que enseña lo que el corte meridional no puede: si los
    chorros de dilución son discretos o ya se han fundido en un anillo, y si
    el reparto azimutal es uniforme — que es justo lo que fija el factor de
    patrón (Ciencia §4.3)."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        dom = field.get("dom") or {}
        rho, j = np.asarray(field.get("rho")), np.asarray(field.get("j"))
        fluid = np.asarray(dom.get("fluid"), dtype=bool)
        if j.ndim != 4:
            return []
        u_scale = float((field.get("units") or {}).get("u_scale", 1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            umag = np.sqrt((j ** 2).sum(axis=0)) / np.maximum(rho, 1e-12)
        umag = np.where(fluid, umag * u_scale, np.nan)
        dx = float(dom.get("dx_mm", 1.0))
        org = dom.get("origin_mm") or (0.0, 0.0, 0.0)
        nz = umag.shape[2]
        # estaciones equiespaciadas dentro del tramo con fluido
        has = np.where(fluid.any(axis=(0, 1)))[0]
        if has.size < n:
            return []
        ks = np.linspace(has[0], has[-1], n + 2)[1:-1].astype(int)
        ax_mm = float(org[0]) + (np.arange(umag.shape[0]) + 0.5) * dx
        ay_mm = float(org[1]) + (np.arange(umag.shape[1]) + 0.5) * dx
        vmax = float(np.nanpercentile(umag, 99)) or 1.0

        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.6))
        for ax, k in zip(np.atleast_1d(axes), ks):
            sl = umag[:, :, k]
            pc = ax.pcolormesh(ax_mm, ay_mm, sl.T, shading="auto",
                               cmap="viridis", vmin=0.0, vmax=vmax)
            ax.contourf(ax_mm, ay_mm, (~fluid[:, :, k]).astype(float).T,
                        levels=[0.5, 1.5], colors=[_C_SOLID], alpha=0.9)
            ax.set_aspect("equal")
            ax.set_title(f"z = {float(org[2]) + (k + 0.5) * dx:.0f} mm",
                         fontsize=9)
            ax.set_xlabel("x  [mm]", fontsize=8)
            ax.tick_params(labelsize=7)
        np.atleast_1d(axes)[0].set_ylabel("y  [mm]", fontsize=8)
        fig.subplots_adjust(right=0.90)
        cax = fig.add_axes([0.925, 0.16, 0.012, 0.66])
        fig.colorbar(pc, cax=cax).set_label("|u|  [m/s]", fontsize=8)
        cax.tick_params(labelsize=7)
        fig.suptitle("L1.5 · cortes transversales: uniformidad azimutal y "
                     "chorros discretos", fontsize=10)
        for ax in np.atleast_1d(axes):
            ax.grid(True, color=_C_GRID, lw=0.6, alpha=0.8)
            ax.set_axisbelow(True)
        fig.text(0.008, 0.006, _cfd_caption(field), fontsize=7, va="bottom",
                 color="#666")
        fig.subplots_adjust(bottom=0.22, top=0.86, left=0.05)
        path = os.path.join(outdir, "cfd_sections.png")
        try:
            os.makedirs(outdir, exist_ok=True)
            fig.savefig(path, dpi=140)
        except Exception:
            plt.close(fig)
            return []
        plt.close(fig)
        return [path]
    except Exception:
        return []


def fig_cfd_axial_flux(field: dict, outdir: str,
                       contract: dict | None = None) -> list[str]:
    """Caudal axial estación a estación: cuánto aire lleva el gas y cuánto
    el anillo en cada z, con las filas de orificios marcadas.

    Es el reparto de aire RESUELTO ESPACIALMENTE — donde el gráfico de
    barras solo da cuatro números, este dice en qué z entra cada cosa."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        dom = field.get("dom") or {}
        j = np.asarray(field.get("j"))
        fluid = np.asarray(dom.get("fluid"), dtype=bool)
        if j.ndim != 4:
            return []
        units = field.get("units") or {}
        ms = float(units.get("mdot_scale", 1.0))
        dx = float(dom.get("dx_mm", 1.0))
        org = dom.get("origin_mm") or (0.0, 0.0, 0.0)
        RR = dom.get("RR_mm")
        r_li = float(dom.get("r_liner_in_mm", 0.0))
        r_lo = float(dom.get("r_liner_out_mm", 1e9))
        jz = np.where(fluid, j[2], 0.0)
        # separa el gas (dentro del liner) del aire del anillo por radio
        if RR is not None:
            RR = np.asarray(RR)
            m_gas = (RR >= r_li) & (RR <= r_lo)
            gas = (jz * m_gas[:, :, None]).sum(axis=(0, 1)) * ms
            ann = (jz * (~m_gas)[:, :, None]).sum(axis=(0, 1)) * ms
        else:
            gas = jz.sum(axis=(0, 1)) * ms
            ann = None
        z = float(org[2]) + (np.arange(jz.shape[2]) + 0.5) * dx

        fig, ax = plt.subplots(figsize=(9.4, 3.8))
        ax.plot(z, gas, color=_C_HOT, lw=1.8, label="interior del liner (gas)")
        if ann is not None:
            ax.plot(z, ann, color=_C_COOL, lw=1.8, label="anillo (aire frío)")
            ax.plot(z, gas + ann, color="#444", lw=1.0, ls="--",
                    label="suma (debe ser ≈ constante)")
        ax.axhline(0.0, color=_C_GRID, lw=1.0)
        for row in ((contract or {}).get("hole_rows") or []):
            try:
                zr = float(row.get("z_mm"))
                ax.axvline(zr, color=_C_ACCENT, lw=1.0, alpha=0.8)
                ax.text(zr, ax.get_ylim()[1], f" {row.get('zone', '')}",
                        fontsize=7, va="top", color="#8a6d1f", rotation=90)
            except Exception:
                continue
        ax.set_xlabel("z  [mm]")
        ax.set_ylabel("ṁ axial  [kg/s]   (+Z)")
        ax.set_title("L1.5 · caudal axial por estación — el reparto de aire "
                     "resuelto en z", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="best")
        return _cfd_finish(fig, [ax],
                           os.path.join(outdir, "cfd_axial_flux.png"), plt,
                           field)
    except Exception:
        return []


def fig_cfd_row_splits(field: dict, post: dict | None, record: dict | None,
                       outdir: str) -> list[str]:
    """Caudal POR FILA de orificios: medido vs red 1D. Más fino que el
    gráfico por zonas — enseña QUÉ fila se desvía, no solo qué zona."""
    plt = _mpl()
    if plt is None:
        return []
    try:
        meas = (post or {}).get("mdot_paths_measured_kgs") or {}
        net = (record or {}).get("mdot_paths_kgs") or {}

        # Los dos lados NO nombran igual los caminos: el LBM devuelve el ID de
        # FILA del contrato (`pz00`, `sz01`, `dz02`…) y la red 1D devuelve el
        # camino agregado por zona (`pz_holes`, `sz_holes`, `dz_holes`). Sin
        # normalizar, la mitad de las barras salían vacías y la figura parecía
        # decir que la red 1D no predice esas filas — cuando lo que pasaba era
        # que se comparaban claves distintas.
        def _canon(k: str) -> str:
            k = str(k)
            if k in ("effusion", "swirler", "cool_ring"):
                return k
            for z in ("pz", "sz", "dz"):
                if k.startswith(z):
                    return f"{z}_holes"
            return k

        def _fold(d: dict) -> dict:
            out: dict[str, float] = {}
            for k, v in d.items():
                try:
                    out[_canon(k)] = out.get(_canon(k), 0.0) + abs(float(v))
                except Exception:
                    continue
            return out

        meas, net = _fold(meas), _fold(net)
        keys = sorted(set(meas) | set(net))
        keys = [k for k in keys if meas.get(k, 0.0) > 0.0
                or net.get(k, 0.0) > 0.0]
        if len(keys) < 2:
            return []
        m = [meas.get(k, 0.0) for k in keys]
        n = [net.get(k, np.nan) for k in keys]
        fig, ax = plt.subplots(figsize=(8.2, 3.6))
        idx = np.arange(len(keys))
        ax.bar(idx - 0.2, n, width=0.4, color=_C_ACCENT, label="L1 · red 1D")
        ax.bar(idx + 0.2, m, width=0.4, color=_C_BRAND, label="L1.5 · medido")
        for i, (a, b) in enumerate(zip(n, m)):
            if np.isfinite(a) and a > 0:
                ax.text(i, max(a, b), f"{100 * (b / a - 1):+.0f} %",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(idx)
        ax.set_xticklabels(keys, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("ṁ  [kg/s]")
        ax.set_title("L1.5 · caudal por camino: medido vs red 1D "
                     "(el % es la desviación)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        return _cfd_finish(fig, [ax],
                           os.path.join(outdir, "cfd_row_splits.png"), plt,
                           field)
    except Exception:
        return []


def generate_cfd(field: dict, outdir: str, post: dict | None = None,
                 record: dict | None = None,
                 l2a: dict | None = None,
                 contract: dict | None = None) -> list[str]:
    """Todas las figuras de CFD que se puedan con lo que haya. Lista vacía
    si no hay matplotlib o el campo no sirve — degradación silenciosa."""
    if not available() or not isinstance(field, dict):
        return []
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        return []
    out: list[str] = []
    out += fig_cfd_velocity(field, outdir)
    out += fig_cfd_recirculation(field, outdir)
    out += fig_cfd_streamlines(field, outdir)
    out += fig_cfd_panel(field, outdir)
    out += fig_cfd_sections(field, outdir)
    out += fig_cfd_axial_flux(field, outdir, contract)
    out += fig_cfd_convergence(field, outdir)
    out += fig_cfd_splits(field, post, record, outdir)
    out += fig_cfd_row_splits(field, post, record, outdir)
    if l2a:
        out += fig_l2a_yplus(l2a, outdir)
    return out


def generate_all(record: dict, outdir: str, front: list[dict] | None = None,
                 contract: dict | None = None,
                 material_limit_K: float | None = None) -> list[str]:
    """Genera todas las figuras posibles. Devuelve las rutas escritas
    (lista vacía si matplotlib no está — degradación silenciosa)."""
    if not available():
        return []
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        return []
    out: list[str] = []
    out += fig_air_split(record, outdir)
    out += fig_liner_thermal(record, outdir, material_limit_K)
    out += fig_emissions_fan(record, outdir)
    if front:
        out += fig_pareto(front, outdir)
    if contract:
        out += fig_liner_profile(contract, outdir)
    return out


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("matplotlib disponible:", available())
    rec = dict(
        splits=dict(pz=0.30, sz=0.12, dz=0.42, cool=0.16),
        phi_pz=1.05, phi_sz=0.72, phi_global=0.21, T3_K=430.0,
        T_wall_profile=[[float(z), 900.0 + 4.0 * z, 820.0 + 3.2 * z]
                        for z in np.linspace(0, 60, 25)],
        EI_NOx_lefebvre=4.2, EI_NOx_rizk_mongia=7.9, EI_NOx_odgers=3.1,
        EI_NOx_laviolette=5.5, EI_NOx_zonal=9.4)
    paths = generate_all(rec, "runs/_viz_smoke",
                         front=[dict(eta_comb=0.99 + 0.002 * i,
                                     EI_NOx=3.0 + 0.9 * i, feasible=i % 3 != 0)
                                for i in range(8)],
                         material_limit_K=1253.0)
    print("figuras:", paths or "(ninguna — degradación silenciosa)")
