"""
QUASAR Phy-CB · cfd/ibrans/post.py
==================================
Post-proceso del solver L2a. El módulo tiene **dos mitades con estado muy
distinto** y aquí se declaran sin adornos:

1. **IMPLEMENTADO (M1)** — las medidas que necesita la escalera V&V y que
   son útiles por sí solas: fuerza sobre un cuerpo inmerso por balance de
   cantidad de movimiento en un volumen de control, número de Strouhal por
   FFT de una señal de estela, tensión de pared y **histograma de y⁺** en
   la banda del IB, cortes de campo para figuras.

2. **IMPLEMENTADO (M4)** — el *producto* de combustor: Δp/p, T4, PF/RTDF
   en el plano 4, mapa de T de pared adiabática (RANKING), penetraciones
   (la maquinaria del JICF) y clustering (T, Z) → CRN de post-proceso →
   NOx (`crn_core`). Todas exigen que el solver lleve el CASO M4 completo
   (`solver.m4`, lo pone `case.build_case`): sin él degradan con motivo,
   no inventan números.

CPU y GPU
---------
El post-proceso trabaja SIEMPRE en NumPy/float64: es diagnóstico, no bucle
caliente, y en float64 no arrastra el redondeo del campo. Cuando el solver
corre en GPU los campos se bajan con `backend.to_numpy` (que es lo que hace
`_np` aquí abajo) — nunca con `np.asarray`, porque CuPy **lanza a
propósito** ante una conversión device→host implícita para que nadie mueva
datos sin enterarse. `control_volume_force` se llama cada N iteraciones
desde la V3: sólo baja las CARAS de la caja de control, no el campo entero.

Etiqueta de honestidad fija de L2a (Ciencia §10) — la repite `HONESTY`:
«Comparación de diseños / detección de puntos calientes — no apto para
certificación de emisiones». Y el límite duro del proyecto (contrato §12):
**Phy-CB no predice temperatura de metal cuantitativa en ninguna fidelidad**;
el mapa de pared de M4 será ADIABÁTICO y sólo sirve para *ranking*.

API pública
-----------
    HONESTY, STATUS
    control_volume_force(solver, box) -> dict            [M1]
    momentum_in_box(solver, box) -> np.ndarray           [M1]
    force_from_history(...) -> dict                      [M1]
    strouhal_from_signal(t, y, ...) -> dict              [M1]
    wall_shear(solver) -> dict                           [M1]
    yplus_histogram(solver, ...) -> dict                 [M1]
    slice_field(solver, comp, axis, index) -> np.ndarray [M1]
    dp_over_p(...)  T4_and_pattern_factor(...)
    adiabatic_wall_map(...)  penetrations(...)
    nox_post_crn(...)  combustor_product(...)            [M4 · esqueleto]
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .backend import to_numpy as _to_numpy


def _np(a) -> np.ndarray:
    """Array de NumPy en float64 a partir de un array de NumPy o de CuPy.

    Es el único puente device→host del post-proceso. `np.asarray` NO vale:
    CuPy lo bloquea a propósito.
    """
    return np.asarray(_to_numpy(a), dtype=np.float64)


HONESTY = ("Comparacion de disenos / deteccion de puntos calientes - no apto "
           "para certificacion de emisiones (Ciencia §10). Phy-CB no predice "
           "temperatura de metal cuantitativa en ninguna fidelidad "
           "(IMPLEMENTATION_CONTRACT §12).")

#: Estado por bloque. Lo leen `ESTADO.md`, los tests y el reporte.
STATUS = {
    "control_volume_force": "M1 · implementado y verificado en V3",
    "strouhal_from_signal": "M1 · implementado y verificado en V3",
    "wall_shear": ("M1 · implementado (estimacion LAMINAR de diagnostico; "
                   "las wall functions acopladas de M2 viven en "
                   "turbulence.KEpsilonRealizable y reportan su y* en "
                   "ystar_stats)"),
    "yplus_histogram": "M1 · implementado como diagnostico de malla",
    "dp_over_p": ("M4 · IMPLEMENTADO (presion total pesada por caudal, "
                  "entrada vs plano 4, /P3; requiere solver.m4)"),
    "T4_and_pattern_factor": ("M4 · IMPLEMENTADO (T4 media masica, PF y "
                              "RTDF con perfil radial en el plano 4)"),
    "adiabatic_wall_map": ("M4 · IMPLEMENTADO (mapa (zeta,phi) ADIABATICO "
                           "para RANKING, etiqueta fija — contrato §12)"),
    "penetrations": ("M4 · IMPLEMENTADO (defecto de T por fila, la "
                     "maquinaria del JICF; en frio, u_r entrante)"),
    "nox_post_crn": ("M4 · IMPLEMENTADO (k-means (T,Z) SEED=71 → zonas → "
                     "crn_core.crn_solve; degrada sin Cantera)"),
    "combustor_product": ("M4 · IMPLEMENTADO (agrega §4.4; cada bloque "
                          "degrada por separado con motivo)"),
}


# ==========================================================================
# 1. Balance de cantidad de movimiento en un volumen de control  [M1]
# ==========================================================================
def _box_indices(mesh, box) -> tuple[list[int], list[int]]:
    """Convierte una caja en coordenadas a índices de CARA del interior.
    La cara `j` del eje `d` está en `x = lo[d] + j·dx`."""
    i0, i1 = [], []
    for d in range(mesh.ndim):
        a = int(round((float(box[d][0]) - mesh.lo[d]) / mesh.dx))
        b = int(round((float(box[d][1]) - mesh.lo[d]) / mesh.dx))
        a = int(np.clip(a, 1, mesh.shape[d] - 1))
        b = int(np.clip(b, a + 1, mesh.shape[d] - 1))
        i0.append(a)
        i1.append(b)
    return i0, i1


def _face_planes(mesh, d: int, j: int, i0, i1):
    """Índices (celda izquierda, celda derecha) de la cara `j` del eje `d`,
    recortados a la sección transversal de la caja."""
    L, R = [], []
    for e in range(mesh.ndim):
        if e == d:
            L.append(mesh.ng + j - 1)
            R.append(mesh.ng + j)
        else:
            L.append(slice(mesh.ng + i0[e], mesh.ng + i1[e]))
            R.append(slice(mesh.ng + i0[e], mesh.ng + i1[e]))
    return tuple(L), tuple(R)


def momentum_in_box(solver, box) -> np.ndarray:
    """∫_CV ρ u dV (vector, SI). Las celdas sólidas aportan 0 porque el IB
    las mantiene a u = 0."""
    m = solver.mesh
    i0, i1 = _box_indices(m, box)
    sl = tuple(slice(m.ng + i0[d], m.ng + i1[d]) for d in range(m.ndim))
    dv = m.dx ** m.ndim
    return np.array([float(np.sum(_np(solver.u[c][sl])))
                     * solver.cfg.rho * dv for c in range(m.ndim)])


def control_volume_force(solver, box) -> dict:
    """Integral de superficie del flujo de cantidad de movimiento sobre la
    frontera de la caja `box = ((x0,x1),(y0,y1)[,(z0,z1)])`.

    Balance (Ω = caja, que CONTIENE al cuerpo inmerso)::

        d/dt ∫_Ω ρu dV = −∮ (ρ u u·n + p n − τ·n) dS + F_cuerpo→fluido
        F_sobre_el_cuerpo = −F_cuerpo→fluido = −( d/dt∫ρu dV + ∮(...) dS )

    Esta función devuelve `S` = ∮(...)dS y `M` = ∫ρu dV; el término
    temporal lo cierra el llamante con `force_from_history` (así se puede
    promediar sobre un número entero de periodos, que es como se mide un
    Cd medio en flujo con desprendimiento).

    Se integra sobre una caja INTERIOR al dominio (no sobre el contorno
    exterior) para que la medida no dependa de la calidad de las
    condiciones de salida — es la práctica del «wake survey».
    """
    m, cfg = solver.mesh, solver.cfg
    nd = m.ndim
    i0, i1 = _box_indices(m, box)
    dA = m.dx ** (nd - 1)
    mu = cfg.rho * cfg.nu
    S = np.zeros(nd)
    for d in range(nd):
        for side, j in (("lo", i0[d]), ("hi", i1[d])):
            sgn = -1.0 if side == "lo" else 1.0
            L, R = _face_planes(m, d, j, i0, i1)
            uL = [_np(solver.u[c][L]) for c in range(nd)]
            uR = [_np(solver.u[c][R]) for c in range(nd)]
            uf = [0.5 * (uL[c] + uR[c]) for c in range(nd)]
            pf = 0.5 * (_np(solver.p[L]) + _np(solver.p[R]))
            # ∂/∂n en la cara (2.º orden, compacta)
            dudn = [(uR[c] - uL[c]) / m.dx for c in range(nd)]
            for c in range(nd):
                # convectivo + presión
                val = cfg.rho * uf[c] * uf[d] + (pf if c == d else 0.0)
                # viscoso: τ_cd = μ(∂u_c/∂x_d + ∂u_d/∂x_c)
                tau = mu * dudn[c]
                if c != d:
                    tau = tau + mu * _tangential_grad(solver, d, c, L, R)
                else:
                    tau = tau + mu * dudn[d]
                val = val - tau
                S[c] += sgn * float(np.sum(val)) * dA
    return dict(S=S, M=momentum_in_box(solver, box), t=solver.t,
                box=[[float(b[0]), float(b[1])] for b in box])


def _tangential_grad(solver, d: int, c: int, L, R) -> np.ndarray:
    """∂u_d/∂x_c en una cara normal a `d` (derivada TANGENCIAL), por
    diferencias centradas promediadas entre las dos celdas de la cara."""
    m = solver.mesh
    fld = solver.u[d]

    def shift(idx, axis, k):
        idx = list(idx)
        s = idx[axis]
        if isinstance(s, slice):
            idx[axis] = slice(s.start + k, s.stop + k)
        else:
            idx[axis] = s + k
        return tuple(idx)

    gl = (_np(fld[shift(L, c, 1)])
          - _np(fld[shift(L, c, -1)])) / (2 * m.dx)
    gr = (_np(fld[shift(R, c, 1)])
          - _np(fld[shift(R, c, -1)])) / (2 * m.dx)
    return 0.5 * (gl + gr)


def force_from_history(samples: Sequence[dict], u_ref: float, rho: float,
                       length: float) -> dict:
    """Cierra el balance con la derivada temporal y adimensionaliza.

    `samples` = lista de salidas de `control_volume_force` ordenadas en
    tiempo. Devuelve Cd/Cl medios sobre TODA la ventana (la derivada
    temporal se evalúa con los extremos, así que promediar sobre un número
    entero de periodos la anula casi exactamente).
    """
    if len(samples) < 2:
        return dict(ok=False, reason="hacen falta >= 2 muestras")
    t = np.array([s["t"] for s in samples], dtype=float)
    S = np.stack([np.asarray(s["S"], dtype=float) for s in samples])
    M = np.stack([np.asarray(s["M"], dtype=float) for s in samples])
    T = float(t[-1] - t[0])
    if T <= 0:
        return dict(ok=False, reason="ventana temporal nula")
    dMdt = (M[-1] - M[0]) / T
    S_mean = np.trapezoid(S, t, axis=0) / T
    F = -(dMdt + S_mean)
    q = 0.5 * rho * u_ref ** 2 * length
    coef = F / max(q, 1e-30)
    out = dict(ok=True, F=F.tolist(), Cd=float(coef[0]), window_s=T,
               dMdt=dMdt.tolist(), S_mean=S_mean.tolist())
    if coef.size > 1:
        out["Cl"] = float(coef[1])
    return out


# ==========================================================================
# 2. Strouhal  [M1]
# ==========================================================================
def strouhal_from_signal(t, y, u_ref: float, length: float,
                         t_skip_frac: float = 0.4) -> dict:
    """Frecuencia dominante de una señal de estela → St = f·L/U.

    Método primario: FFT sobre la señal remuestreada uniformemente (el
    `dt` del solver varía ligeramente con |u|max). Cruce por cero como
    comprobación independiente. `t_skip_frac` descarta el transitorio
    inicial. **Nunca lanza.**
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size < 32:
        return dict(ok=False, reason=f"muestras insuficientes ({t.size})")
    k0 = int(t.size * float(np.clip(t_skip_frac, 0.0, 0.9)))
    t, y = t[k0:], y[k0:]
    if t.size < 32 or t[-1] <= t[0]:
        return dict(ok=False, reason="ventana util insuficiente")
    n = int(2 ** math.floor(math.log2(t.size)))
    tu = np.linspace(t[0], t[-1], n)
    yu = np.interp(tu, t, y)
    yu = yu - yu.mean()
    dt = float(tu[1] - tu[0])
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(yu * win))
    freq = np.fft.rfftfreq(n, dt)
    spec[0] = 0.0
    kmax = int(np.argmax(spec))
    f_fft = float(freq[kmax])
    # cruces por cero ascendentes
    s = np.sign(yu)
    cross = np.count_nonzero((s[:-1] < 0) & (s[1:] >= 0))
    f_zc = cross / max(float(tu[-1] - tu[0]), 1e-30)
    amp = float(np.max(np.abs(yu)))
    return dict(ok=bool(f_fft > 0 and amp > 1e-6), f_hz=f_fft,
                St=f_fft * length / max(u_ref, 1e-30),
                St_zerocross=f_zc * length / max(u_ref, 1e-30),
                amplitude=amp, n_samples=int(n), df_hz=float(freq[1]),
                reason="" if f_fft > 0 else "sin pico espectral")


# ==========================================================================
# 3. Pared: tensión, y+ y su histograma  [M1]
# ==========================================================================
def wall_shear(solver) -> dict:
    """Tensión de pared en las celdas fantasma del IB.

    Se toma la velocidad TANGENCIAL en el punto imagen y se divide por su
    distancia a la pared: τ_w = μ·|u_t,imagen| / d_ip (aproximación lineal
    de la capa límite, coherente con la interpolación del ghost-cell).
    Válida mientras la primera celda esté en la subcapa viscosa — que es lo
    que ocurre en laminar. **Con turbulencia (M2) esto se sustituye por la
    ley de pared de `turbulence.py`; aquí es sólo diagnóstico.**
    """
    m, cfg = solver.mesh, solver.cfg
    if m.phi is None or m.n_ghost == 0:
        return dict(ok=False, reason="la malla no tiene frontera inmersa")
    nd = m.ndim
    d_g = np.abs(m.phi[m._g_idx])
    d_ip = np.maximum(d_g, 1.5 * m.dx)
    nrm = np.stack([n[m._g_idx] for n in m.normals], axis=1)
    u_ip = np.zeros((m.n_ghost, nd))
    for c in range(nd):
        flat = _np(solver.u[c]).reshape(-1)
        u_ip[:, c] = (flat[m._g_nodes] * m._g_wfluid).sum(axis=1)
    un = np.sum(u_ip * nrm, axis=1)
    ut = u_ip - un[:, None] * nrm
    ut_mag = np.sqrt(np.sum(ut ** 2, axis=1))
    mu = cfg.rho * cfg.nu
    tau_w = mu * ut_mag / np.maximum(d_ip, 1e-30)
    u_tau = np.sqrt(np.maximum(tau_w, 0.0) / max(cfg.rho, 1e-30))
    return dict(ok=True, tau_w=tau_w, u_tau=u_tau, d_ip=d_ip,
                n_wall_cells=int(m.n_ghost),
                tau_w_mean=float(np.mean(tau_w)),
                tau_w_max=float(np.max(tau_w)))


def yplus_histogram(solver, bins=(0, 1, 5, 11, 30, 100, 150, 300, 1000)
                    ) -> dict:
    """Histograma de y⁺ = u_τ·y/ν con y = distancia a pared de la primera
    celda FLUIDA (= |φ|, gratis).

    La banda objetivo del liner es y⁺ ∈ [30, 150] (WAVE6 §2); las celdas
    fuera de banda son un *warning*, no un error. **En M1 esto es
    diagnóstico de malla**: sin modelo de turbulencia, y⁺ no gobierna
    ninguna ley de pared todavía.
    """
    ws = wall_shear(solver)
    if not ws.get("ok"):
        return dict(ok=False, reason=ws.get("reason", ""), status=STATUS[
            "yplus_histogram"])
    m, cfg = solver.mesh, solver.cfg
    # distancia de la PRIMERA celda fluida: |φ| en las celdas fluidas de la
    # banda (φ < 2·dx), que es lo que ve la ley de pared
    band = m.fluid & (m.phi < 2.0 * m.dx)
    y = np.abs(m.phi[band])
    if y.size == 0:
        return dict(ok=False, reason="banda fluida vacia")
    # u_τ representativo: el medio de la pared (M1 laminar, sin mapa 1:1)
    u_tau = float(np.mean(ws["u_tau"]))
    yp = u_tau * y / max(cfg.nu, 1e-30)
    edges = np.asarray(bins, dtype=float)
    counts, _ = np.histogram(yp, bins=edges)
    infr = float(np.count_nonzero((yp >= 30.0) & (yp <= 150.0)) / yp.size)
    return dict(ok=True, bins=edges.tolist(), counts=counts.tolist(),
                yplus_mean=float(np.mean(yp)), yplus_max=float(np.max(yp)),
                yplus_min=float(np.min(yp)), frac_in_30_150=infr,
                n_cells=int(yp.size), u_tau_mean=u_tau,
                warning=("y+ fuera de [30,150] en "
                         f"{100.0 * (1 - infr):.0f} % de la banda"),
                status=STATUS["yplus_histogram"])


def slice_field(solver, comp: int | str = 0, axis: int = 2,
                index: int | None = None) -> np.ndarray:
    """Corte 2D del campo (`comp` = índice de componente o `"p"`), para las
    figuras del reporte. Devuelve el array del interior."""
    m = solver.mesh
    fld = solver.pressure() if comp == "p" else solver.field(int(comp))
    if m.ndim == 2:
        return fld
    idx = [slice(None)] * m.ndim
    idx[axis] = m.shape[axis] // 2 if index is None else int(index)
    return fld[tuple(idx)]


# ==========================================================================
# 4. Producto de combustor — M4 (firmas CONGELADAS, ahora implementadas)
# ==========================================================================
def _m4_or_reason(solver) -> tuple[dict | None, dict | None]:
    """El caso M4 que `case.build_case` deja en `solver.m4`, o el motivo de
    que no haya producto. Sin el caso completo NO se inventa nada."""
    m4 = getattr(solver, "m4", None)
    if not isinstance(m4, dict):
        return None, dict(ok=False, honesty=HONESTY,
                          reason=("el solver no lleva el caso M4: monta el "
                                  "caso con case.build_case(contract) — "
                                  "sin SDF completo no hay producto"))
    return m4, None


def _model(solver, key: str):
    m4 = getattr(solver, "m4", None)
    if isinstance(m4, dict):
        return (m4.get("models") or {}).get(key)
    return None


def _plane4_k(solver) -> int:
    """Índice (del interior) del plano 4: el último plano fluido del lado
    de salida."""
    sign = solver.m4["flow_sign"]
    return solver.mesh.shape[2] - 2 if sign > 0 else 1


def _liner_mask_interior(solver) -> np.ndarray:
    """Máscara del interior del LINER (excluye el anillo de la carcasa)."""
    from .case import _interior_coords
    m4 = solver.m4
    itp = m4["interp"]
    xi, yi, zi = _interior_coords(solver)
    ri = np.sqrt(xi ** 2 + yi ** 2)
    rout = np.interp(zi, itp["z"], itp["r_out"])
    rin = np.interp(zi, itp["z"], itp["r_in"])
    return solver.mesh.fluid[solver.mesh.interior] \
        & (ri < rout) & (ri > rin - 0.5 * solver.mesh.dx)


def dp_over_p(solver, patches=None) -> dict:
    """Δp/p_3 entre el parche de entrada y el plano 4: presión total media
    (p + ½ρ|u|²) **pesada por el flujo másico** en cada plano, dividida por
    P₃ (la presión termodinámica del caso). La p del solver es manométrica
    (la salida ancla p = 0): sólo la DIFERENCIA tiene sentido, y es la que
    se reporta."""
    m4, why = _m4_or_reason(solver)
    if why:
        return dict(why, status=STATUS["dp_over_p"])
    from .case import _rho_field
    m = solver.mesh
    sign = m4["flow_sign"]
    rho = _rho_field(solver)
    u = [_np(solver.u[c][m.interior]) for c in range(3)]
    p = _np(solver.p[m.interior])
    fl = m.fluid[m.interior]
    liner = _liner_mask_interior(solver)
    q = 0.5 * rho * (u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    p_tot = p + q
    w = rho * np.maximum(sign * u[2], 0.0)          # peso: flujo másico +
    k0 = 1 if sign > 0 else m.shape[2] - 2
    k4 = _plane4_k(solver)

    def _wmean(k, mask):
        ww = np.where(mask[:, :, k], w[:, :, k], 0.0)
        s_w = float(np.sum(ww))
        if s_w <= 0.0:
            return float("nan"), 0.0
        return float(np.sum(ww * p_tot[:, :, k]) / s_w), s_w

    pt_in, w_in = _wmean(k0, fl)
    pt_4, w_4 = _wmean(k4, liner)
    P3 = float(m4["P3_Pa"])
    dp = (pt_in - pt_4) / max(P3, 1e-12)
    ok = math.isfinite(dp) and w_in > 0.0 and w_4 > 0.0
    # ⚠️ QUÉ ES ESTE NÚMERO Y QUÉ NO ES. La pérdida de un combustor se
    # reparte entre el swirler, los orificios del liner y la cámara. Este
    # Δp/p mide lo que ocurre DENTRO del dominio: si el anillo
    # liner↔carcasa está resuelto, el aire de cada fila lo atraviesa de
    # verdad y el número es comparable con el de la red 1D. Si NO lo está
    # (modo «sin anillo»: holgura < 2·dx), las filas entran por
    # transpiración ya dentro del liner y **la pérdida de orificio no
    # ocurre en el dominio**: el número mide sólo la rama del swirler más
    # la cámara y NO es comparable con `record["dp_over_p"]`. Se declara
    # en `comparable_with_1d` en vez de fabricar la parte que falta desde
    # el propio C_d, que haría el contraste tautológico.
    annulus = bool(m4["meta"].get("annulus_resolved"))
    return dict(ok=ok, dp_over_p_cfd=float(dp), pt_in_Pa=pt_in,
                pt_plane4_Pa=pt_4, P3_Pa=P3,
                annulus_resolved=annulus,
                comparable_with_1d=annulus,
                note=("presion total MECANICA (manometrica) pesada por "
                      "caudal entre la cara de entrada y el plano 4"
                      + ("" if annulus else
                         "; ANILLO NO RESUELTO: las filas entran por "
                         "transpiracion dentro del liner, la perdida de "
                         "orificio NO ocurre en el dominio y este numero "
                         "NO es comparable con la red 1D — hace falta "
                         "dx <= 1 mm (ver meta.annulus_gap_min_mm)")),
                honesty=HONESTY,
                reason="" if ok else "sin flujo medible en algun plano")


def T4_and_pattern_factor(solver, record=None, plane_z=None) -> dict:
    """T4 media (pesada por ρ·u_ζ), T4 máx, **PF y RTDF en el plano 4**::

        PF   = (T4_max − T̄4) / (T̄4 − T3)
        RTDF = (max_r T̄4(r) − T̄4) / (T̄4 − T3)

    con T̄4(r) el perfil radial (media másica por corona). Necesita la
    energía del caso (M2) y, para que el número hable del combustor, la
    llama de M3 encendida."""
    m4, why = _m4_or_reason(solver)
    if why:
        return dict(why, status=STATUS["T4_and_pattern_factor"])
    en = _model(solver, "energy")
    if en is None or en.T is None:
        return dict(ok=False, status=STATUS["T4_and_pattern_factor"],
                    honesty=HONESTY, reason="el caso no lleva energia (M2)")
    from .case import _interior_coords, _rho_field
    m = solver.mesh
    sign = m4["flow_sign"]
    k4 = _plane4_k(solver) if plane_z is None else int(plane_z)
    T = _np(en.T[m.interior])[:, :, k4]
    rho = _rho_field(solver)[:, :, k4]
    uz = _np(solver.u[2][m.interior])[:, :, k4] * sign
    liner = _liner_mask_interior(solver)[:, :, k4]
    w = np.where(liner, rho * np.maximum(uz, 0.0), 0.0)
    s_w = float(np.sum(w))
    if s_w <= 0.0:
        return dict(ok=False, status=STATUS["T4_and_pattern_factor"],
                    honesty=HONESTY, reason="sin flujo en el plano 4")
    T3 = float(m4["T3_K"])
    t4 = float(np.sum(w * T) / s_w)
    t4max = float(np.max(np.where(w > 0.0, T, -np.inf)))
    dT = max(t4 - T3, 1.0)
    xi, yi, _ = _interior_coords(solver)
    ri = np.sqrt(xi ** 2 + yi ** 2)[:, :, k4]
    r_lo, r_hi = float(ri[liner].min()), float(ri[liner].max())
    nbin = 12
    prof = []
    for b in range(nbin):
        m_b = liner & (ri >= r_lo + (r_hi - r_lo) * b / nbin) \
            & (ri < r_lo + (r_hi - r_lo) * (b + 1) / nbin)
        wb = np.where(m_b, w, 0.0)
        sb = float(np.sum(wb))
        prof.append(float(np.sum(wb * T) / sb) if sb > 0.0
                    else float("nan"))
    prof_ok = [v for v in prof if math.isfinite(v)]
    rtdf = (max(prof_ok) - t4) / dT if prof_ok else float("nan")
    return dict(ok=True, T4_cfd_K=t4, T4_max_K=t4max, T3_K=T3,
                PF_cfd=(t4max - t4) / dT, RTDF_cfd=float(rtdf),
                radial_profile_K=prof, honesty=HONESTY)


def adiabatic_wall_map(solver, record=None, n_z: int = 24,
                       n_phi: int = 16) -> dict:
    """Mapa (ζ, φ) de temperatura de la banda IB del liner. **ETIQUETA
    FIJA: es un mapa ADIABÁTICO para RANKING entre diseños, no un valor
    absoluto** — sin radiación ni CHT el absoluto está sesgado hasta
    ~200 K, y Phy-CB no promete T de metal en ninguna fidelidad
    (contrato §12)."""
    m4, why = _m4_or_reason(solver)
    if why:
        return dict(why, status=STATUS["adiabatic_wall_map"])
    en = _model(solver, "energy")
    if en is None or en.T is None:
        return dict(ok=False, status=STATUS["adiabatic_wall_map"],
                    honesty=HONESTY, reason="el caso no lleva energia (M2)")
    m = solver.mesh
    itp = m4["interp"]
    t_w = m4["t_liner_m"]
    gc = m.ghost_coords()
    if gc.shape[0] == 0:
        return dict(ok=False, status=STATUS["adiabatic_wall_map"],
                    honesty=HONESTY, reason="sin celdas fantasma")
    # ⚠️ La T se lee en el PUNTO IMAGEN (primera celda FLUIDA), no en la
    # celda fantasma. En el fantasma vive la CONDICIÓN DE CONTORNO —y en
    # la pared transpirante del M4 esa condición es T3 impuesta—, así que
    # leerla ahí devuelve el aire de refrigeración, no el gas: el mapa
    # salía a ~430 K justo donde la llama está a 2400 K (cazado el
    # 2026-08-03 con la figura de `run_m4_figures.py`).
    tflat = _np(en.T).reshape(-1)
    wsum = np.maximum(m._g_wsum, 1e-12)
    Tg = (tflat[m._g_nodes] * m._g_wfluid).sum(axis=1) / wsum
    # fantasmas huérfanos (estencil casi todo sólido): sin gas adyacente
    # fiable, se excluyen del mapa en vez de contaminar el ranking
    valid = ~m._g_orphan
    x, y, z = gc[:, 0], gc[:, 1], gc[:, 2]
    r = np.sqrt(x * x + y * y)
    rout = np.interp(z, itp["z"], itp["r_out"])
    sel = (r >= rout - 0.51 * m.dx) & (r <= rout + t_w + 0.51 * m.dx) \
        & (r < rout + 0.5 * t_w) & valid  # cara CALIENTE (lado del liner)
    if not np.any(sel):
        return dict(ok=False, status=STATUS["adiabatic_wall_map"],
                    honesty=HONESTY, reason="sin fantasmas en el liner")
    zeta = m4["flow_sign"] * (z[sel] - m4["z_dome_m"])
    phi = np.arctan2(y[sel], x[sel])
    Tw = Tg[sel]
    L = m4["meta"]["L_tot_mm"] * 1e-3
    grid = np.full((n_z, n_phi), np.nan)
    iz = np.clip((zeta / max(L, 1e-9) * n_z).astype(int), 0, n_z - 1)
    ip = np.clip(((phi + math.pi) / (2 * math.pi) * n_phi).astype(int),
                 0, n_phi - 1)
    for a in range(n_z):
        for b in range(n_phi):
            mm = (iz == a) & (ip == b)
            if np.any(mm):
                grid[a, b] = float(np.max(Tw[mm]))
    k_hot = int(np.argmax(Tw))
    return dict(ok=True, T_wall_adiabatic_max_K=float(np.max(Tw)),
                T_wall_adiabatic_mean_K=float(np.mean(Tw)),
                hottest_zeta_mm=float(zeta[k_hot] * 1e3),
                hottest_phi_deg=float(math.degrees(phi[k_hot])),
                map_zeta_phi_K=[[None if not math.isfinite(v) else
                                 round(float(v), 1) for v in row]
                                for row in grid],
                n_wall_cells=int(Tw.size),
                label=("RANKING, no absoluto: pared adiabatica sin "
                       "radiacion ni CHT (contrato §12)"),
                honesty=HONESTY)


def penetrations(solver, contract=None) -> dict:
    """Penetración MEDIDA de los chorros por fila de orificios: trayectoria
    del extremo de exceso/defecto de T en una losa aguas abajo de la fila —
    exactamente la maquinaria del JICF (`run_v5_jcf.py`), con el defecto de
    T porque los chorros de dilución entran FRÍOS en un núcleo caliente. En
    frío (sin energía) degrada al máximo de velocidad radial entrante, con
    el método declarado en la salida."""
    m4, why = _m4_or_reason(solver)
    if why:
        return dict(why, status=STATUS["penetrations"])
    from .case import _interior_coords, _rho_field  # noqa: F401
    m = solver.mesh
    itp = m4["interp"]
    sign = m4["flow_sign"]
    en = _model(solver, "energy")
    xi, yi, zi = _interior_coords(solver)
    ri = np.sqrt(xi ** 2 + yi ** 2)
    liner = _liner_mask_interior(solver)
    if en is not None and en.T is not None:
        fld = _np(en.T[m.interior])
        method = "defecto de T (chorro frio en nucleo caliente)"
        use_T = True
    else:
        ux = _np(solver.u[0][m.interior])
        uy = _np(solver.u[1][m.interior])
        fld = -(xi * ux + yi * uy) / np.maximum(ri, 1e-12)  # u_r entrante
        method = "maximo de velocidad radial entrante (caso frio)"
        use_T = False
    # la capa pegada a la pared lleva la CONDICIÓN de transpiración (aire
    # a T3 impuesto), no el chorro: incluirla hacía que el máximo déficit
    # de T cayera siempre en la pared y Y/d saliera ~0 por construcción
    rout_f = np.interp(zi, itp["z"], itp["r_out"])
    core = liner & (ri < rout_f - 1.5 * m.dx)
    out = {}
    detail = []
    for pid, p in (m4.get("paths") or {}).items():
        row = p.get("row")
        if row is None:
            continue
        d = row["d_m"]
        slab = core & (sign * (zi - row["z_m"]) >= 0.5 * d) \
            & (sign * (zi - row["z_m"]) <= 4.0 * d)
        if not np.any(slab):
            continue
        rout_here = np.interp(row["z_m"], itp["z"], itp["r_out"])
        if use_T:
            t_ref = float(np.mean(fld[slab]))
            deficit = np.where(slab, t_ref - fld, -np.inf)
        else:
            deficit = np.where(slab, fld, -np.inf)
        k = int(np.argmax(deficit))
        idx = np.unravel_index(k, fld.shape)
        r_core = float(ri[idx])
        y_pen = max(rout_here - r_core, 0.0)
        out[pid] = float(y_pen / max(d, 1e-9))
        detail.append(dict(id=pid, Y_mm=y_pen * 1e3, d_mm=d * 1e3,
                           Y_over_d=out[pid], resolved=row["resolved"]))
    return dict(ok=bool(out), Y_over_d_measured=out, method=method,
                detail=detail, honesty=HONESTY,
                caveat=("las filas TRANSPIRADAS entran sin su k/eps de "
                        "chorro (limite declarado del gancho IB): su "
                        "penetracion es aproximada; la de las resueltas "
                        "usa la maquinaria verificada del JICF"))


def nox_post_crn(solver, record=None, n_clusters: int = 32,
                 timeout_s: float = 180.0) -> dict:
    """NOx de post-proceso (Ciencia §9.4): clustering k-means en (T, Z) del
    campo → los clusters se agregan a las zonas pz/sz/dz (volúmenes y T
    MEDIDOS) → `crn_core.crn_solve` re-resuelve el CRN con esos números
    (el mismo patrón que L1.5). Degrada con motivo si no hay Cantera o si
    el caso no lleva especies. Bucle del k-means ACOTADO, `SEED = 71`."""
    m4, why = _m4_or_reason(solver)
    if why:
        return dict(why, status=STATUS["nox_post_crn"])
    sp = _model(solver, "species")
    en = _model(solver, "energy")
    if sp is None or en is None:
        return dict(ok=False, status=STATUS["nox_post_crn"], honesty=HONESTY,
                    reason="el caso no lleva especies+energia (M3)")
    from .case import _interior_coords
    m = solver.mesh
    liner = _liner_mask_interior(solver)
    T = _np(en.T[m.interior])[liner]
    Z = sp.mixture_fraction()[m.interior][liner]
    _, _, zi = _interior_coords(solver)
    zeta = (m4["flow_sign"] * (zi - m4["z_dome_m"]))[liner]
    n = T.size
    if n < 64:
        return dict(ok=False, status=STATUS["nox_post_crn"], honesty=HONESTY,
                    reason="campo demasiado pequeno para clustering")
    rng = np.random.default_rng(71)
    sel = rng.choice(n, size=min(n, 100_000), replace=False)
    ts, zs, zzs = T[sel], Z[sel], zeta[sel]
    feats = np.stack([(ts - ts.min()) / max(np.ptp(ts), 1e-9),
                      (zs - zs.min()) / max(np.ptp(zs), 1e-9)], axis=1)
    k = int(min(n_clusters, max(4, sel.size // 50)))
    cent = feats[rng.choice(sel.size, size=k, replace=False)]
    lab = np.zeros(sel.size, dtype=int)
    for _ in range(25):                              # bucle ACOTADO
        d2 = ((feats[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            mj = lab == j
            if np.any(mj):
                cent[j] = feats[mj].mean(axis=0)
    dv = m.dx ** 3 * n / sel.size                    # volumen por muestra
    L = dict(m4["L_zones_mm"])
    z_pz = L.get("pz", 0.0) * 1e-3
    z_sz = z_pz + L.get("sz", 0.0) * 1e-3
    vol = {"pz": 0.0, "sz": 0.0, "dz": 0.0}
    clusters = []
    for j in range(k):
        mj = lab == j
        if not np.any(mj):
            continue
        zc = float(np.mean(zzs[mj]))
        zone = "pz" if zc <= z_pz else ("sz" if zc <= z_sz else "dz")
        vj = float(np.count_nonzero(mj)) * dv
        vol[zone] += vj
        clusters.append(dict(zone=zone, V_m3=vj, T_K=float(np.mean(ts[mj])),
                             Z=float(np.mean(zs[mj]))))
    out = dict(ok=False, status=STATUS["nox_post_crn"], honesty=HONESTY,
               V_zones_measured_m3={z: float(v) for z, v in vol.items()},
               n_clusters=len(clusters), clusters=clusters)
    if not isinstance(record, dict):
        out["reason"] = ("sin record no se puede montar el CRN "
                         "(se devuelven los clusters y volumenes)")
        return out
    try:
        import crn_core as crn
    except Exception as exc:
        out["reason"] = f"crn_core no importable ({type(exc).__name__})"
        return out
    if not getattr(crn, "CANTERA_AVAILABLE", False):
        out["reason"] = "cantera_unavailable (declarado; sin NOx de CRN)"
        return out
    rec2 = dict(record)
    for zone, key in (("pz", "V_pz_m3"), ("sz", "V_sz_m3"),
                      ("dz", "V_dz_m3")):
        if vol[zone] > 0.0:
            rec2[key] = vol[zone]
    rec2["V_c_m3"] = sum(vol.values()) or rec2.get("V_c_m3")
    sol = crn.crn_solve(None, rec2, timeout_s)
    if not sol:
        out["reason"] = (f"crn_solve degrado "
                         f"({getattr(crn, 'LAST_STATUS', '?')})")
        return out
    out.update(ok=True, EI_NOx_crn=sol.get("EI_NOx_crn"),
               EI_CO_crn=sol.get("EI_CO_crn"),
               eta_comb_crn=sol.get("eta_comb_crn"),
               reason="",
               note=("CRN de post-proceso con VOLUMENES de zona medidos en "
                     "el campo L2a (clustering (T,Z) k-means, SEED=71); "
                     "topologia 1D de crn_core — clusters→zonas es la "
                     "agregacion v1 DECLARADA"))
    return out


def combustor_product(solver, theta=None, record=None, contract=None) -> dict:
    """Agrega el producto L2a en las claves del contrato §4.4 (`PF_cfd`,
    `T4_cfd_K`, `dp_over_p_cfd`, `yplus_hist`, `cfd_mesh`,
    `splits_measured`, `Y_over_d_measured`, `V_zones_measured`). Es lo que
    `__init__.solve()` devuelve a `physics_core`. **Nunca lanza**: cada
    bloque degrada por separado con su motivo."""
    m4, why = _m4_or_reason(solver)
    if why:
        return dict(why, status=STATUS["combustor_product"])
    out: dict = dict(ok=True, honesty=HONESTY, blocks={})
    try:
        out["cfd_mesh"] = solver.mesh.summary()
    except Exception as exc:
        out["blocks"]["cfd_mesh"] = f"{type(exc).__name__}: {exc}"
    # y+ del modelo de pared ACOPLADO (M2), no el diagnostico laminar
    try:
        tur = _model(solver, "turbulence")
        ys = tur._last_ystar if tur is not None else None
        if ys is not None:
            arr = _np(ys)
            edges = np.asarray((0, 1, 5, 11, 30, 100, 150, 300, 1000),
                               dtype=float)
            counts, _ = np.histogram(arr, bins=edges)
            out["yplus_hist"] = counts.tolist()
            out["ystar_stats"] = tur.ystar_stats()
        else:
            out["yplus_hist"] = []
            out["blocks"]["yplus"] = "sin turbulencia acoplada"
    except Exception as exc:
        out["yplus_hist"] = []
        out["blocks"]["yplus"] = f"{type(exc).__name__}: {exc}"
    try:
        from .case import measure_splits
        sm = measure_splits(solver)
        out["splits_measured"] = sm["splits_measured"]
        out["splits_target_CdA"] = sm["splits_target_CdA"]
        out["mass_closure_rel"] = sm["mass_closure_rel"]
        out["rows_discretization"] = sm["rows_discretization"]
    except Exception as exc:
        out["blocks"]["splits"] = f"{type(exc).__name__}: {exc}"
    try:
        dp = dp_over_p(solver)
        if dp.get("ok"):
            out["dp_over_p_cfd"] = dp["dp_over_p_cfd"]
        else:
            out["blocks"]["dp_over_p"] = dp.get("reason", "?")
    except Exception as exc:
        out["blocks"]["dp_over_p"] = f"{type(exc).__name__}: {exc}"
    try:
        t4 = T4_and_pattern_factor(solver, record)
        if t4.get("ok"):
            out["T4_cfd_K"] = t4["T4_cfd_K"]
            out["PF_cfd"] = t4["PF_cfd"]
            out["RTDF_cfd"] = t4["RTDF_cfd"]
        else:
            out["blocks"]["T4_PF"] = t4.get("reason", "?")
    except Exception as exc:
        out["blocks"]["T4_PF"] = f"{type(exc).__name__}: {exc}"
    try:
        pen = penetrations(solver, contract)
        if pen.get("ok"):
            out["Y_over_d_measured"] = pen["Y_over_d_measured"]
            out["penetration_method"] = pen["method"]
        else:
            out["blocks"]["penetrations"] = pen.get("reason", "?")
    except Exception as exc:
        out["blocks"]["penetrations"] = f"{type(exc).__name__}: {exc}"
    try:
        nox = nox_post_crn(solver, record)
        out["V_zones_measured"] = nox.get("V_zones_measured_m3", {})
        if nox.get("ok"):
            out["EI_NOx_crn"] = nox["EI_NOx_crn"]
            out["eta_comb_crn"] = nox.get("eta_comb_crn")
        else:
            out["blocks"]["nox_post_crn"] = nox.get("reason", "?")
    except Exception as exc:
        out["blocks"]["nox_post_crn"] = f"{type(exc).__name__}: {exc}"
    try:
        wall = adiabatic_wall_map(solver, record)
        if wall.get("ok"):
            out["T_wall_adiabatic_max_K"] = wall["T_wall_adiabatic_max_K"]
            out["wall_map"] = dict(
                map_zeta_phi_K=wall["map_zeta_phi_K"],
                hottest_zeta_mm=wall["hottest_zeta_mm"],
                hottest_phi_deg=wall["hottest_phi_deg"],
                label=wall["label"])
    except Exception as exc:
        out["blocks"]["wall_map"] = f"{type(exc).__name__}: {exc}"
    return out
