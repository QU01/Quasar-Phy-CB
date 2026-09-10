#!/usr/bin/env python3
"""
QUASAR Phy-CB · cfd/tests/run_v9_preccinsta.py
==============================================
**Script MANUAL del V9 — quemador swirl PRECCINSTA (DLR)**, el último caso
de la escalera V&V de WAVE6 (§4). Corre HORAS en GPU: sus números van a
`ESTADO.md` y el test `V9a` de `test_ibrans_v9.py` sigue en SKIP declarado
hasta que estén.

Caso y expectativa (DECLARADAS por la wave)
-------------------------------------------
Quemador PRECCINSTA (base pública DLR; Meier et al., Combust. Flame 150
(2007)): premezcla metano-aire φ = 0.83, ṁ_aire = 734.2 g/min, boquilla
Ø 27.85 mm con swirler SN ≈ 0.6, cámara de 85×85×110 mm. Aquí se
aproxima con la maquinaria del caso M4 y un contrato SINTÉTICO:

* cámara CILÍNDRICA de radio equivalente (la real es cuadrada — la
  esquina afecta poco a la forma de la llama; declarado),
* premezcla representada como fuente de combustible en la boquilla
  (evaporación/premezcla instantánea; declarado),
* **pérdida de calor en paredes**: `T_wall_ib_K` isoterma — SIN ella la
  llama cambia de forma M↔V y el caso no discrimina nada (WAVE6 §4).

Expectativa DECLARADA: topología y posición de llama CUALITATIVAS (forma
M — anclada en las DOS capas de cortadura — frente a V), no perfiles
cuantitativos: RANS+EDM ahí es duro y la literatura de referencia es LES.
El diagnóstico imprime el mapa (r, ζ) de liberación de calor cerca de la
boquilla y una métrica de forma: fracción del HRR en la capa de cortadura
EXTERIOR (M ≳ 0.25; V ≈ 0). Si el swirl sale muy mal con k-ε+curvatura,
el plan B declarado es SST (WAVE6 §7).

Uso::

    python cfd/tests/run_v9_preccinsta.py --dx-mm 1.0 --max-iter 30000
    python cfd/tests/run_v9_preccinsta.py --t-wall 1050 --json v9.json
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

pc.assert_provenance(pc)

from cfd.ibrans.backend import to_numpy                       # noqa: E402
from cfd.ibrans.case import CaseConfig, build_case, run_case  # noqa: E402

# --- datos del caso (Meier et al. 2007; el punto quieto phi=0.83) --------
MDOT_AIR_KGS = 734.2 / 60.0 / 1000.0        # 734.2 g/min
PHI = 0.83
S_CH4 = 3.989                               # kg O2 / kg CH4 (chem._S_MASS)
MDOT_FUEL_KGS = PHI * MDOT_AIR_KGS * 0.233 / S_CH4
R_CHAMBER_MM = 47.96        # radio del cilindro de área = 85×85 mm²
L_CHAMBER_MM = 110.0
D_NOZZLE_MM = 27.85
D_HUB_MM = 16.0
SN_PRECC = 0.6
T_IN_K = 320.0
P_ATM = 101_325.0


def synthetic_contract() -> dict:
    """Contrato sintético `phycb-combustor-1`-compatible del PRECCINSTA
    (SOLO las claves que `case.build_case` lee)."""
    a_sw = 0.25 * np.pi * (D_NOZZLE_MM ** 2 - D_HUB_MM ** 2)
    return dict(
        schema="phycb-combustor-1(v9-preccinsta-sintetico)",
        liner=dict(profile=[[0.0, 0.0, R_CHAMBER_MM],
                            [L_CHAMBER_MM, 0.0, R_CHAMBER_MM]],
                   t_wall_mm=1.5,
                   dome=dict(type="flat", z_mm=0.0)),
        casing=dict(profile=[[0.0, R_CHAMBER_MM + 2.0],
                             [L_CHAMBER_MM, R_CHAMBER_MM + 2.0]]),
        swirler=dict(d_hub_mm=D_HUB_MM, d_tip_mm=D_NOZZLE_MM,
                     beta_deg=50.0, SN=SN_PRECC, A_sw_total_mm2=a_sw,
                     count=1, z_mm=0.0),
        injectors=dict(type="premix", count=1, ring_r_mm=0.0),
        hole_rows=[],
        effusion={},
        derived=dict(D_L_mm=2.0 * R_CHAMBER_MM, L_pz_mm=40.0,
                     L_sz_mm=30.0, L_dz_mm=40.0, L_tot_mm=L_CHAMBER_MM),
        cfd_boundary_conditions=dict(
            P3_Pa=P_ATM, T3_K=T_IN_K, rho3_kgm3=P_ATM / (287.05 * T_IN_K),
            mdot_air_kgs=MDOT_AIR_KGS, mdot_fuel_kgs=MDOT_FUEL_KGS,
            fuel="methane", turbulence_intensity=0.07,
            turbulence_length_scale_mm=0.07 * D_NOZZLE_MM))


def flame_shape_metric(s) -> dict:
    """Mapa (r, ζ) de liberación de calor cerca de la boquilla y métrica
    de forma: fracción del HRR a r > r_boquilla (capa EXTERIOR) en
    ζ < 1.5·D — M ≳ 0.25, V ≈ 0. CUALITATIVO (declarado)."""
    from cfd.ibrans.case import _interior_coords
    m = s.mesh
    sp = s.m4["models"]["species"]
    om = np.asarray(to_numpy(sp._omega_prev), dtype=float)
    xi, yi, zi = _interior_coords(s)
    ri = np.sqrt(xi ** 2 + yi ** 2)
    r_noz = 0.5 * D_NOZZLE_MM * 1e-3
    near = zi <= 1.5 * D_NOZZLE_MM * 1e-3
    tot = float(np.sum(np.where(near, om, 0.0)))
    outer = float(np.sum(np.where(near & (ri > 1.15 * r_noz), om, 0.0)))
    nb_r, nb_z = 16, 12
    grid = np.zeros((nb_z, nb_r))
    rmx = R_CHAMBER_MM * 1e-3
    zmx = 1.5 * D_NOZZLE_MM * 1e-3
    for a in range(nb_z):
        for b in range(nb_r):
            mm = near & (zi >= zmx * a / nb_z) & (zi < zmx * (a + 1) / nb_z) \
                & (ri >= rmx * b / nb_r) & (ri < rmx * (b + 1) / nb_r)
            grid[a, b] = float(np.sum(np.where(mm, om, 0.0)))
    return dict(outer_frac=outer / max(tot, 1e-30),
                hrr_map=grid.tolist(),
                shape="M" if outer / max(tot, 1e-30) > 0.25 else "V",
                note=("metrica CUALITATIVA declarada: fraccion del HRR en "
                      "la capa de cortadura exterior a menos de 1.5·D de "
                      "la boquilla"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V9 PRECCINSTA (manual)")
    ap.add_argument("--dx-mm", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=30_000)
    ap.add_argument("--timeout-s", type=float, default=14_400.0)
    ap.add_argument("--t-wall", type=float, default=1050.0,
                    help="T de pared isoterma [K] (0 = adiabatica, para "
                         "comparar la forma M<->V)")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    print("V9 PRECCINSTA — puede tardar HORAS (GPU); expectativa "
          "CUALITATIVA declarada (forma M vs V)")
    c = synthetic_contract()
    cfg = CaseConfig(
        dx_mm=a.dx_mm, reactive=True, ignition=True,
        max_iter=a.max_iter, timeout_s=a.timeout_s,
        T_wall_ib_K=(a.t_wall if a.t_wall > 0 else None),
        fuel_src_r_mm=0.6 * D_NOZZLE_MM,
        n_hold=2000, n_ramp=5000, n_stable=500)
    s = build_case(c, cfg)
    print(f"  malla: {s.mesh.ncell} celdas "
          f"({s.mesh.summary()['cells_fluid']} fluidas), "
          f"pared {'isoterma %g K' % a.t_wall if a.t_wall > 0 else 'adiab.'}")
    t0 = time.perf_counter()
    info = run_case(s)
    el = time.perf_counter() - t0
    print(f"  estado: {info['status']} ({info['iterations']} it, "
          f"{el / 60:.1f} min, res_u {info['res_u']:.2e})")
    print(f"  anclada: {info.get('anchored')}  "
          f"HRR {info.get('heat_release_W', float('nan')):.3g} W "
          f"(osc {100 * info.get('hrr_window_osc_rel', float('nan')):.1f} %)")
    shp = flame_shape_metric(s)
    print(f"  forma de llama: {shp['shape']} "
          f"(fraccion exterior {shp['outer_frac']:.3f}) — {shp['note']}")
    ok = bool(info.get("anchored"))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(case="V9 PRECCINSTA", dx_mm=a.dx_mm,
                           t_wall_K=a.t_wall, info={
                               k: v for k, v in info.items()
                               if isinstance(v, (int, float, str, bool))},
                           shape=shp, passed=ok), fh, indent=2,
                      ensure_ascii=False)
        print(f"  JSON: {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
