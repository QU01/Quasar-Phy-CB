#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/cold_post.py — post-proceso del campo frío L1.5.

Convierte el campo promediado de `cfd/xlb_cold.solve_cold` en los cuatro
números que la wave persigue (WAVE5 §A.2.3):

1. **Splits REALES** — se integra el caudal que cruza el plano de CADA fila de
   `hole_rows[]`, el del swirler y el de la efusión, y se compara con la red
   1D de `crn_core.network_1d`. La integración usa el estimador EXACTO por
   poblaciones de `xlb_cold.region_flux`, no una interpolación de ρu.
2. **Penetración de los chorros** — trayectoria del máximo de cantidad de
   movimiento radial entrante de cada chorro de dilución → `Y_max/d` medido
   frente a la correlación `1.15·√J` de Ciencia §4.2.
3. **Zonas** — el interior del liner se segmenta por posición axial (PZ/SZ/DZ
   con las longitudes del contrato) y por recirculación (`u_s < 0` ⇒ parte
   recirculante de la PZ) → `V_pz`, `V_sz`, `V_dz` y los caudales de
   intercambio MEDIDOS entre zonas.
4. **Corrección de densidad por zona** — limitación declarada del frío: sin
   expansión térmica los τ salen sesgados, así que se escalan con
   ρ ∝ 1/T_zona tomando la T de L0/L1 (`ρ_z/ρ₃ = T₃/T_z` a P constante).
   Se devuelven los DOS convenios, igual que `crn_core.tau_convention`.

Todo lo que sale de aquí lleva el sufijo `_measured` y convive con lo
correlacionado; nunca lo pisa. **Nunca lanza**: un campo incompleto devuelve
lo que se pudo medir más una nota.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import physics_core as pc                                        # noqa: E402
from cfd import xlb_cold as xc                                   # noqa: E402

pc.assert_provenance(pc)

SCHEMA = "phycb-coldpost-1"

# ==========================================================================
# Tunables (SCREAMING_CASE + procedencia, contrato §10.7)
# ==========================================================================
#: Ventana axial, en diámetros, en la que se sigue la traza de cada chorro.
PEN_WINDOW_OVER_D = 8.0

#: Estación de lectura de la penetración, en la coordenada de SEMEJANZA del
#: chorro en flujo cruzado `x/(r·d)` con `r = u_j/u_c = √J`. ⚠️ CONVENIO
#: PROPIO: la correlación de Ciencia §4.2 (`Y_max/d = 1.15√J`) **no
#: especifica estación** y la trayectoria crece como x^⅓, así que a estación
#: fija en x/d NINGUNA elección puede reproducir un √J (medido: la traza
#: escala como J^0.25). En la coordenada de semejanza sí. Es EL MISMO valor
#: que `xlb_cold.JICF_STATION_OVER_RD`, con el que se calibró en T43:
#: cambiarlo obliga a rehacer T43.
PEN_STATION_OVER_RD = xc.JICF_STATION_OVER_RD

#: Fracción del pico de cantidad de movimiento radial por debajo de la cual se
#: considera que la traza del chorro se ha perdido en el flujo cruzado.
PEN_SIGNAL_FLOOR = 0.15

#: Umbral de recirculación: velocidad ALONG-FLOW negativa. Es el criterio de
#: WAVE5 §A.2.3 («u_z < 0 → PZ recirculante»), expresado en la coordenada de
#: recorrido `s` para que valga también en `reverse_annular`.
RECIRC_U_LAT = 0.0


def _f(x, d=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(d)
    except Exception:
        return float(d)


def _zone_coordinate(dom, contract):
    """Coordenada de recorrido `s` (mm) creciente en el sentido del gas.

    En `reverse_annular` el gas va a −Z desde el domo, así que
    `s = z_domo − z`; en `can`/`annular`, `s = z − z_domo`. Es exactamente la
    convención de `flow_path[].dir_z` del contrato (§9)."""
    z = dom["coords_mm"]["z"]
    z_dome = _f(dom.get("z_dome_mm"), 0.0)
    return (z_dome - z) if dom.get("reverse") else (z - z_dome)


def _zone_masks(dom, contract):
    """PZ / SZ / DZ / salida del interior del liner por posición axial."""
    d = contract.get("derived") or {}
    L_pz = _f(d.get("L_pz_mm"), 0.0)
    L_sz = _f(d.get("L_sz_mm"), 0.0)
    L_dz = _f(d.get("L_dz_mm"), 0.0)
    s = _zone_coordinate(dom, contract)
    gas = dom["regions"]["liner"]
    edges = [(-1e9, L_pz, "pz"), (L_pz, L_pz + L_sz, "sz"),
             (L_pz + L_sz, L_pz + L_sz + L_dz, "dz"),
             (L_pz + L_sz + L_dz, 1e9, "exit")]
    out = {}
    for lo, hi, name in edges:
        sel = (s > lo) & (s <= hi)
        m = np.zeros(dom["shape"], dtype=bool)
        m[:, :, sel] = True
        out[name] = m & gas
    return out, s


def _radial_unit(dom):
    """Versores radiales (e_x, e_y) por celda del plano (nx, ny)."""
    RR = np.maximum(dom["RR_mm"], 1e-9)
    xs = dom["coords_mm"]["x"][:, None]
    ys = dom["coords_mm"]["y"][None, :]
    return xs / RR, ys / RR


def _jet_penetration(field, contract, dom, rows):
    """`Y/d` medido de cada fila de orificios y su trayectoria.

    Para cada orificio se sigue, aguas abajo, la profundidad (medida desde la
    pared del liner) a la que la cantidad de movimiento radial ENTRANTE es
    máxima. El resultado de la fila es la media sobre sus orificios."""
    j = field["j"]
    rho = field["rho"]
    dx = dom["dx_mm"]
    xs = dom["coords_mm"]["x"]
    ys = dom["coords_mm"]["y"]
    zs = dom["coords_mm"]["z"]
    nx, ny, nz = dom["shape"]
    gas = dom["regions"]["liner"]
    flow_dir = -1.0 if dom.get("reverse") else 1.0
    r_in = _f(dom.get("r_liner_in_mm"), 0.0)
    out = {}
    for row in rows:
        d_h = _f(row.get("d_mm"), 0.0)
        n_h = int(row.get("n", 0) or 0)
        if d_h <= 0.0 or n_h <= 0:
            continue
        z_h = _f(row.get("z_mm"), 0.0)
        r_h = _f(row.get("r_mm"), 0.0)
        phi0 = math.radians(_f(row.get("phi0_deg"), 0.0))
        depth_max = max(r_h - r_in - 0.5 * dx, dx)
        n_s = max(int(PEN_WINDOW_OVER_D * d_h / dx), 3)
        traj_acc = np.zeros(n_s)
        peak_acc = np.zeros(n_s)
        used = 0
        for kk in range(n_h):
            phi = phi0 + 2.0 * math.pi * kk / n_h
            ex, ey = math.cos(phi), math.sin(phi)
            tj = np.zeros(n_s)
            pk = np.zeros(n_s)
            for isx in range(n_s):
                z = z_h + flow_dir * (isx * dx)
                kz = int(round((z - zs[0]) / dx))
                if not (1 <= kz < nz - 1):
                    break
                best, best_v = 0.0, 0.0
                depth = dx
                while depth <= depth_max:
                    r = r_h - depth
                    ix = int(round((r * ex - xs[0]) / dx))
                    iy = int(round((r * ey - ys[0]) / dx))
                    if 0 <= ix < nx and 0 <= iy < ny and gas[ix, iy, kz]:
                        jr = -(j[0][ix, iy, kz] * ex + j[1][ix, iy, kz] * ey)
                        if jr > best_v:
                            best_v, best = float(jr), depth
                    depth += dx
                tj[isx], pk[isx] = best, best_v
            traj_acc += tj
            peak_acc += pk
            used += 1
        if not used:
            continue
        traj = traj_acc / used
        peak = peak_acc / used
        #  Estación en la coordenada de SEMEJANZA x/(r·d), r = √J (ver el
        #  tunable): es la única en la que `Y/d ∝ √J` tiene sentido.
        J_row = max(_f(row.get("J"), 0.0), 0.0)
        st_over_d = PEN_STATION_OVER_RD * math.sqrt(max(J_row, 1e-9))
        i_st = min(max(int(round(st_over_d * d_h / dx)), 0), n_s - 1)
        p0 = float(peak.max()) if peak.size else 0.0
        i_lost = n_s
        for i in range(n_s):
            if p0 > 0.0 and peak[i] < PEN_SIGNAL_FLOOR * p0:
                i_lost = i
                break
        lo_i = max(int(round(0.7 * st_over_d * d_h / dx)), 0)
        hi_i = min(int(round(1.3 * st_over_d * d_h / dx)) + 1, n_s)
        win = traj[lo_i:max(hi_i, lo_i + 1)]
        Y = float(np.median(win)) if win.size else float(traj[i_st])
        out[str(row.get("id", "row"))] = dict(
            zone=str(row.get("zone", "")), d_mm=d_h, J=J_row,
            Y_mm_measured=Y, Y_over_d_measured=Y / max(d_h, 1e-9),
            Y_over_d_correlation=1.15 * math.sqrt(J_row),
            Y_over_DL_measured=Y / max(_f(
                (contract.get("derived") or {}).get("D_L_mm"), 1.0), 1e-9),
            station_over_rd=PEN_STATION_OVER_RD, station_over_d=st_over_d,
            signal_lost_over_d=float(i_lost * dx / max(d_h, 1e-9)),
            traj_over_d=[[float(i * dx / d_h), float(traj[i] / d_h)]
                         for i in range(min(n_s, 24))])
    return out


def post_process(field: dict, contract: dict, record: dict | None = None) -> dict:
    """Campo frío + contrato → medidas de L1.5. **Nunca lanza.**

    `record` es OPCIONAL y solo se usa para la corrección de densidad por
    zona (`T_pz_K`/`T_sz_K`/`T_dz_K` viven en el record `phycb-record-1`, no
    en el contrato de geometría). Sin él, la corrección se desactiva y se
    declara: los τ devueltos son entonces los fríos.

    Devuelve (contrato §4.4 y WAVE5 §A.2.3):
    `splits_measured`, `mdot_paths_measured_kgs`, `Y_over_d_measured`,
    `V_zones_measured_m3`, `tau_zones_measured_ms`,
    `tau_zones_measured_cold_ms`, `mdot_zone_exchange_kgs`,
    `recirc_frac_pz`, `dp_over_p_cfd`, `cfd_mesh`, `cold_post_notes`.
    """
    try:
        return _post_impl(field, contract, record)
    except Exception as exc:                    # red de seguridad
        return dict(schema=SCHEMA, ok=False,
                    cold_post_notes=[f"post-proceso degradado: "
                                     f"{type(exc).__name__}: {exc}"])


def _post_impl(field: dict, contract: dict, record: dict | None) -> dict:
    dom = field["dom"]
    units = field["units"]
    sh = dom["shape"]
    dx = dom["dx_mm"]
    dx3_m3 = (dx * 1e-3) ** 3
    ms = units["mdot_scale"]
    notes: list[str] = list(field.get("notes") or [])
    lim: list[str] = []

    # ---- 1. splits medidos ------------------------------------------------
    fl = field["flux_lat"]
    mdot_in = abs(fl.get("inlet", 0.0)) * ms
    mdot_out = abs(fl.get("outlet", 0.0)) * ms
    rows = {r["id"]: r for r in dom["rows"]}
    paths: dict[str, float] = {}
    for name, v in fl.items():
        if name.startswith("hole:"):
            paths[name.split(":", 1)[1]] = float(v) * ms
    paths["swirler"] = float(fl.get("swirler", 0.0)) * ms
    paths["effusion"] = float(fl.get("effusion", 0.0)) * ms
    tot = sum(abs(v) for v in paths.values())
    zone_of = {rid: str(r.get("zone", "")) for rid, r in rows.items()}
    splits = {"pz": 0.0, "sz": 0.0, "dz": 0.0, "cool": 0.0}
    for name, v in paths.items():
        z = ("pz" if name == "swirler" else
             "cool" if name == "effusion" else zone_of.get(name, "dz"))
        splits[z] += abs(v) / max(tot, 1e-30)

    mass_close = abs(mdot_in - mdot_out) / max(mdot_in, 1e-30)
    if abs(tot - mdot_in) / max(mdot_in, 1e-30) > 0.10:
        lim.append(
            f"la suma de los caudales por camino ({tot:.5f} kg/s) se aparta "
            f"{100.0 * abs(tot - mdot_in) / max(mdot_in, 1e-30):.1f} % del "
            f"caudal de entrada ({mdot_in:.5f} kg/s): a este dx quedan fugas "
            "por caminos no etiquetados (huecos de sellado, giro del domo). "
            "Los splits están NORMALIZADOS a la suma de caminos.")

    # ---- 2. zonas y volúmenes --------------------------------------------
    zones, s_coord = _zone_masks(dom, contract)
    V = {k: float(m.sum()) * dx3_m3 for k, m in zones.items()}
    # recirculación: componente de la velocidad a lo largo del recorrido
    j = field["j"]
    rho = field["rho"]
    flow_dir = -1.0 if dom.get("reverse") else 1.0
    u_s = flow_dir * j[2] / np.maximum(rho, 1e-12)
    recirc = (u_s < RECIRC_U_LAT) & zones["pz"]
    V_recirc = float(recirc.sum()) * dx3_m3
    frac_recirc = V_recirc / max(V["pz"], 1e-30)

    # caudales de intercambio MEDIDOS entre zonas consecutivas
    ex = {}
    order = ("pz", "sz", "dz", "exit")
    for a, b in zip(order[:-1], order[1:]):
        ex[f"{a}->{b}"] = region_flux_kg(field, zones[a], zones[b], sh, ms)
    ex["out"] = mdot_out

    # ---- 3. penetración ----------------------------------------------------
    pen = _jet_penetration(field, contract, dom,
                           [r for r in (contract.get("hole_rows") or [])
                            if isinstance(r, dict) and r.get("flow", True)])

    # ---- 4. τ por zona con CORRECCIÓN DE DENSIDAD -------------------------
    d = contract.get("derived") or {}
    T3 = _f(field.get("T3_K"), 300.0)
    rho3 = _f(field.get("rho3_kgm3"), 1.0)
    rz = record if isinstance(record, dict) else {}
    T_zone = {"pz": _f(rz.get("T_pz_K"), 0.0), "sz": _f(rz.get("T_sz_K"), 0.0),
              "dz": _f(rz.get("T_dz_K"), 0.0)}
    if not any(v > 0.0 for v in T_zone.values()):
        lim.append("sin `record` no hay T por zona: la corrección de densidad "
                   "queda DESACTIVADA y `tau_zones_measured_ms` es el τ frío.")
    m_zone = {"pz": abs(ex.get("pz->sz", 0.0)),
              "sz": abs(ex.get("sz->dz", 0.0)),
              "dz": abs(ex.get("dz->exit", 0.0))}
    tau_cold, tau_hot = {}, {}
    for z in ("pz", "sz", "dz"):
        m = max(m_zone[z], 1e-12)
        tau_cold[z] = 1e3 * rho3 * V[z] / m
        Tz = T_zone[z] if T_zone[z] > 0.0 else T3
        tau_hot[z] = tau_cold[z] * (T3 / max(Tz, 1.0))
    tau_cold["total"] = sum(tau_cold[z] for z in ("pz", "sz", "dz"))
    tau_hot["total"] = sum(tau_hot[z] for z in ("pz", "sz", "dz"))

    lim.append(
        "L1.5 es flujo FRÍO e isotermo: los τ medidos con ρ₃ están sesgados "
        "por el factor T_zona/T₃ (≈"
        f"{max(T_zone.values() or [T3]) / max(T3, 1.0):.1f}× en la PZ). "
        "`tau_zones_measured_ms` ya lleva la corrección ρ ∝ 1/T con las T de "
        "L0/L1; `tau_zones_measured_cold_ms` es el crudo del LBM. Ninguno "
        "sustituye a un cálculo caliente.")
    lim.append(
        "L1.5 NO predice ni PF ni T4 ni temperatura de pared: no hay reacción "
        "ni transporte de energía en el campo frío. `PF`, `PF_star`, `RTDF` y "
        "`T4_max_K` siguen siendo la correlación de L0 [Ciencia §4.3].")

    # ---- 5. Δp medido -----------------------------------------------------
    #
    # ⚠️ `dp_over_p_cfd` NO es comparable con el Δp/p de diseño del record.
    # Diagnosticado el 2026-08-01 sobre el ancla, con TRT y ν efectiva ×1
    # (o sea, sin la excusa del suelo de τ):
    #
    #     A_ref del diseño   = 3335 mm²  →  U_ref = 19.2 m/s  →  q_ref =   375 Pa
    #     A_inlet del dominio=  621 mm²  →  U_in  = 103.4 m/s →  q_in  = 10832 Pa
    #                                          cociente de áreas ×5.37 ⇒ q ×28.9
    #
    #     Δp de diseño = 7500 Pa   = 20.0 · q_ref
    #     Δp del LBM   = 119748 Pa = 11.1 · q_in
    #
    # En cabezas dinámicas el LBM pierde MENOS que el diseño (11.1 vs 20.0):
    # el solver no está inflando nada. Lo que pasa es que el parche de entrada
    # del dominio es ×5.37 más pequeño que el área de referencia del
    # combustor, porque el dominio NO incluye el difusor que frena el flujo de
    # salida del compresor hasta U_ref. Todo el campo corre a ~5× la velocidad
    # de diseño y la pérdida, que va con el cuadrado, sale ×29.
    #
    # Consecuencias, para no volver a diagnosticar mal:
    #   * el Δp/p absoluto de L1.5 NO es utilizable como predicción; sirve
    #     para comparar diseños entre sí con el mismo dominio;
    #   * por eso afinar la malla (×2 en dx) y arreglar τ (×55 → ×1) apenas lo
    #     movieron: la pérdida es INERCIAL (forma, a través de los orificios),
    #     no viscosa;
    #   * el arreglo de verdad es de GEOMETRÍA DEL DOMINIO —modelar el difusor
    #     o imponer la entrada sobre A_ref—, no del operador de colisión.
    # Queda anotado en `cold_post_notes` para que salga en el reporte.
    inlet = dom["regions"]["inlet"]
    outlet = dom["regions"]["outlet"]
    #  Presiones MANOMÉTRICAS respecto a la salida (ρ_lat = 1): el LBM es
    #  débilmente compresible y su presión absoluta ρ·c_s²·escala no tiene
    #  significado físico, solo lo tienen las diferencias.
    p_in = (float(rho[inlet].mean()) - 1.0) * xc.CS2 * units["p_scale"]
    p_out = (float(rho[outlet].mean()) - 1.0) * xc.CS2 * units["p_scale"]
    P3 = _f(field.get("P3_Pa"), 1.0)
    dp_over_p = (p_in - p_out) / max(P3, 1.0)
    #  Diagnóstico que acompaña SIEMPRE al número (ver el comentario largo de
    #  arriba): sin él, `dp_over_p_cfd` se lee como una predicción y no lo es.
    A_in_m2 = _f(dom.get("A_inlet_m2"), 0.0)
    A_ref_m2 = _f((record or {}).get("A_ref_m2"), 0.0)
    if A_in_m2 > 0.0 and A_ref_m2 > 0.0:
        lim.append(
            f"Δp/p medido = {dp_over_p:.4f} NO es comparable con el de diseño: "
            f"el parche de entrada del dominio ({1e6 * A_in_m2:.0f} mm²) es "
            f"×{A_ref_m2 / A_in_m2:.2f} más pequeño que el área de referencia "
            f"del combustor ({1e6 * A_ref_m2:.0f} mm²) —el dominio no incluye "
            f"el difusor—, así que la cabeza dinámica va ×"
            f"{(A_ref_m2 / A_in_m2) ** 2:.0f} y la pérdida, que va con el "
            "cuadrado, también. Úsese para COMPARAR diseños con el mismo "
            "dominio, nunca como valor absoluto.")
    # Δp anillo→liner por cota axial (lo que gobierna la efusión)
    ann = dom["regions"]["annulus"]
    gas = dom["regions"]["liner"]
    dp_axial = []
    zs = dom["coords_mm"]["z"]
    for k in range(1, sh[2] - 1, max(1, sh[2] // 24)):
        ca, cg = ann[:, :, k], gas[:, :, k]
        if ca.any() and cg.any():
            dpa = (float(rho[:, :, k][ca].mean())
                   - float(rho[:, :, k][cg].mean())) * xc.CS2 * units["p_scale"]
            dp_axial.append([float(zs[k]), float(dpa)])

    out = dict(
        schema=SCHEMA, ok=True,
        splits_measured={k: float(v) for k, v in splits.items()},
        mdot_paths_measured_kgs={k: float(v) for k, v in paths.items()},
        mdot_in_measured_kgs=float(mdot_in),
        mdot_out_measured_kgs=float(mdot_out),
        mass_closure_rel=float(mass_close),
        rows_discretization=[dict(
            id=r["id"], zone=r["zone"], A_row_mm2=r["A_row_mm2"],
            A_vox_mm2=r["A_vox_mm2"], n_vox=r["n_vox"],
            A_ratio=(r["A_vox_mm2"] / r["A_row_mm2"]
                     if r["A_row_mm2"] > 0 else 0.0)) for r in dom["rows"]],
        Y_over_d_measured={k: v["Y_over_d_measured"] for k, v in pen.items()},
        jet_penetration_measured=pen,
        V_zones_measured_m3={k: float(v) for k, v in V.items()},
        V_recirc_pz_m3=float(V_recirc), recirc_frac_pz=float(frac_recirc),
        mdot_zone_exchange_kgs={k: float(v) for k, v in ex.items()},
        tau_zones_measured_ms={k: float(v) for k, v in tau_hot.items()},
        tau_zones_measured_cold_ms={k: float(v) for k, v in tau_cold.items()},
        dp_over_p_cfd=float(dp_over_p),
        dp_annulus_liner_Pa=dp_axial,
        p_inlet_Pa=float(p_in), p_outlet_Pa=float(p_out),
        cfd_mesh=dict(field.get("cfd_mesh") or {}),
        cold_post_notes=notes, cold_post_limitations=lim)
    return out


def region_flux_kg(field, A, B, shape, mdot_scale) -> float:
    """`xlb_cold.region_flux` en kg/s (azúcar para el post-proceso)."""
    return float(xc.region_flux(field["gbar"], A, B, shape) * mdot_scale)


#: Camino de `crn_core.PATH_NAMES` → zona a la que entrega el aire. Copia
#: deliberada de `crn_core.PATH_ZONE` para no depender de un privado.
PATH_ZONE = {"swirler": "pz", "pz_holes": "pz", "sz_holes": "sz",
             "dz_holes": "dz", "effusion": "cool", "cool_ring": "cool"}


def compare_with_1d(post: dict, net1d: dict) -> dict:
    """Tabla «red 1D vs LBM» — el primer cierre de bucle de fidelidades del
    sistema (criterio de aceptación WAVE5 §C.2).

    La red 1D tiene un sexto camino, `cool_ring` (anillo de domo + fugas de
    sello), que **no está dibujado** en la geometría y por tanto no existe en
    el campo de voxels. Compararlo contra un LBM que solo ve cinco caminos
    metería un sesgo del tamaño de ese camino, así que se re-normaliza la
    solución 1D SIN él antes de comparar. Se devuelve también la versión sin
    re-normalizar para que el lector vea la diferencia."""
    meas = post.get("splits_measured") or {}
    paths = (net1d or {}).get("splits_solved_paths") or {}
    raw = (net1d or {}).get("splits_solved") or {}
    tot = sum(_f(v) for k, v in paths.items() if k != "cool_ring")
    norm: dict[str, float] = {"pz": 0.0, "sz": 0.0, "dz": 0.0, "cool": 0.0}
    for k, v in paths.items():
        if k == "cool_ring" or tot <= 0.0:
            continue
        norm[PATH_ZONE.get(k, "dz")] += _f(v) / tot
    rows = []
    for z in ("pz", "sz", "dz", "cool"):
        m, s = _f(meas.get(z)), _f(norm.get(z))
        rows.append(dict(zone=z, lbm=m, network_1d=s,
                         network_1d_raw=_f(raw.get(z)),
                         dev=(m / s - 1.0) if s > 1e-12 else float("nan")))
    dev = [abs(r["dev"]) for r in rows if math.isfinite(r["dev"])]
    return dict(rows=rows, dev_max=max(dev) if dev else float("nan"),
                cool_ring_frac=_f(paths.get("cool_ring")),
                note="la red 1D se re-normaliza sin `cool_ring` (camino "
                     "modelado, no dibujado) para comparar manzana con "
                     "manzana contra los cinco caminos que el LBM ve")
