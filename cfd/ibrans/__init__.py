"""
QUASAR Phy-CB · cfd/ibrans/ — solver **L2a** nativo (IB-RANS sobre el SDF)
==========================================================================
Fidelidad L2a de la escalera de Ciencia §10: RANS reactivo sobre malla
cartesiana con **frontera inmersa alimentada por el SDF de PicoGK, sin
mallado**. Diseño completo y justificación de cada decisión: Ciencia §11;
estado medido por hitos: `ESTADO.md` (el plan de wave original ya no
viaja en el repo).

ESTADO REAL DE LA WAVE (leer antes de usar — detalle en `ESTADO.md`)
--------------------------------------------------------------------
    M1  solver frío laminar + IB + escalera V1–V3 ............ **HECHO**
    M2  turbulencia k-ε realizable + ρ variable + energía .... **HECHO**
        (escalera M2a–M2f + V4 backward-step + V5 chorro)
    M3  especies + EDM + arranque de llama ................... **HECHO**
        (escalera M3a–M3d + V6 S_L vs Cantera + V6c encendido
         desde frío + V7 contradifusión; V8 Sandia D PENDIENTE
         declarado; `chem.py` + Pyrometheus verificado)
    M4  combustor completo + producto (PF, T4, Δp/p, NOx) ....
        **IMPLEMENTADO, no cerrado** (caso en `case.py`, producto en
         `post.py`, escalera en `cfd/tests/test_ibrans_v9.py`). La llama
         ANCLA en el combustor real —el riesgo #1 del proyecto queda
         resuelto— pero T4 sale a +25.7 % del balance y el Δp/p no es
         comparable con la red 1D mientras el anillo liner↔carcasa no se
         resuelva (pide dx ≤ 1 mm). Números y diagnóstico: `ESTADO.md`
         §12; plan original en `cfd/ibrans/M4_PLAN.md`

`solve(theta, rec)` — el punto de entrada que llama
`physics_core._try_higher` — monta el caso completo desde el contrato y
devuelve el producto del §4.4; si el caso no puede producir, devuelve
`None` con el porqué en `LAST_STATUS` y el sufijo corto en `DEGRADE_TAG`
(el sistema degrada con `source="ibrans_L2A(<tag>→…)"`, que sigue siendo
la verdad).

Lo que M1 **sí** entrega hoy
----------------------------
* `IBRANSSolver` verificado por la escalera V1–V3 (Poiseuille/Couette,
  cavidad de Ghia, cilindro con IB): ver `cfd/tests/test_ibrans_v1_v3.py`.
* `cold_smoke(contract)`: la cadena contrato → SDF → malla cartesiana →
  IB → solver, corriendo sobre la geometría REAL del combustor. Es una
  prueba de fontanería de arquitectura, **no un producto**: no da splits
  ni penetraciones (eso es M4).
* `mesh.py` con la distancia a pared = |φ| gratis, que es el argumento
  estructural de toda la arquitectura.
* Toda la maquinaria de post-proceso que la escalera V&V necesita
  (`post.control_volume_force`, `post.strouhal_from_signal`,
  `post.yplus_histogram`).
* **Backend GPU real (CuPy)**: el mismo kernel corre en CPU y en GPU. Con
  la RTX 3070 Ti de esta máquina el objetivo de rendimiento de la wave
  (500 k celdas × 10 k iteraciones en 5–40 min) **se cumple**: medido
  1.17e-7 s/celda/paso ⇒ 9.8 min con la GPU libre (2.01e-7 ⇒ 16.8 min si la
  GPU está compartida), frente a ~1.9e-6 s/celda/paso ⇒ 2.6 h en NumPy/CPU.
  Detalle y método en `ESTADO.md` §5.

Límite duro heredado del proyecto (contrato §12): **Phy-CB no predice
temperatura de metal cuantitativa en ningún nivel de fidelidad.** Eso NO
cambia por tener GPU: es una limitación de modelo, no de cómputo.

API pública
-----------
    solve(theta, rec) -> dict | None        ← contrato §8 (hoy: None)
    LAST_STATUS : dict
    milestone_status() -> dict
    cold_smoke(contract, ...) -> dict
    IBRANSSolver, SolverConfig, make_bcs, CartesianMesh
    mesh, solver, turbulence, chem, post, backend  (submódulos)
"""

from __future__ import annotations

import os

import numpy as np

from . import backend, case, chem, energy, mesh, post, solver, turbulence
from .case import CaseConfig, build_case, run_case
from .energy import EnergyLowMach, PassiveScalar
from .mesh import CartesianMesh, mesh_from_contract, patches_from_contract
from .solver import IBRANSSolver, SolverConfig, make_bcs
from .turbulence import KEpsilonRealizable

__all__ = ["solve", "LAST_STATUS", "DEGRADE_TAG", "milestone_status",
           "cold_smoke", "IBRANSSolver", "SolverConfig", "make_bcs",
           "CartesianMesh", "KEpsilonRealizable", "EnergyLowMach",
           "PassiveScalar", "CaseConfig", "build_case", "run_case",
           "mesh", "solver", "turbulence", "chem", "energy", "post",
           "backend", "case"]

#: Estado por hito. Es la fuente de verdad que consultan `ESTADO.md`, los
#: tests y el reporte — si un hito se completa, se cambia AQUÍ.
MILESTONES = {
    "M1": dict(done=True, name="solver frio laminar + IB",
               verified=("V1 Poiseuille 0.17 % / Couette 0.04 % (<2 %); "
                         "V2 Ghia Re=100 y Re=1000 <1 % de U_tapa (<5 %); "
                         "V3 cilindro D/10 Cd 1.396 (+3.4 %) y St 0.1664 "
                         "(+0.8 %) (<10 %) — medido 2026-07-31"),
               caveat=("objetivo de RENDIMIENTO CUMPLIDO en GPU (CuPy): "
                       "500k celdas x 10k pasos = 9.8 min con la GPU libre "
                       "y 16.8 min con la GPU compartida (1.17e-7 y 2.01e-7 "
                       "s/celda/paso, RTX 3070 Ti, fp64), dentro de la banda "
                       "5-40 min de la wave; en NumPy/CPU son 2.6 h. En "
                       "mallas < 100k celdas la GPU pierde por coste de "
                       "lanzamiento y 'auto' se queda en CPU. Ver "
                       "ESTADO.md §5")),
    "M2": dict(done=True, name="k-eps realizable + rho variable + energia",
               verified=("V4 backward-facing step (Vogel&Eaton Re=28k "
                         "ER=1.25): x_r=6.23h vs 6.67 (-6.6%, umbral 15%), "
                         "y* medio 67 con 90% en [30,150]; V5 chorro plano: "
                         "S=0.0877 vs 0.10 (-12.3%, umbral 15%); M2a "
                         "decaimiento homogeneo 0.14% vs analitico; M2b "
                         "canal Cf a -3.7% de Dean; M2c expansion termica "
                         "low-Mach +0.9% vs analitico - medido 2026-08-02"),
               caveat=("wall functions de dos escalas (u* de k + ley de "
                       "Spalding) sobre IB: imprecisas en capa limite "
                       "termica (riesgo declarado Ciencia §11.3); "
                       "correccion de curvatura Spalart-Shur SIMPLIFICADA "
                       "(r_tilde=0)")),
    "M3": dict(done=True, name="especies + EDM + arranque de llama",
               verified=("V6 llama premezclada 1D propano: S_L=0.2476 vs "
                         "Cantera FreeFlame 0.2534 con el MISMO mecanismo "
                         "(-2.3%, umbral 10%); V6c ENCENDIDO desde frio "
                         "(riesgo #1): parche+rampa+subrelajacion enciende, "
                         "ancla y da la misma S_L (±5%); V7 contradifusion: "
                         "T_max=2144 vs 2150 K de Cantera (-0.3%); M3a-M3d "
                         "cinetica==Pyrometheus==Cantera y reactor 0D "
                         "conservativo - medido 2026-08-02"),
               caveat=("V8 Sandia D PENDIENTE (declarado: falta piloto "
                       "anular + caso 3D grande; el mecanismo CH4 ya esta "
                       "verificado — V6d: S_L -8.9%); solo el paso 1 "
                       "efectivo acoplado (sin CO); cp constante calibrado "
                       "a T_ad; V6 del jet-A sin referencia (sin termo "
                       "NASA de C12H23). La ROBUSTEZ del encendido en el "
                       "combustor real se juzga en M4"),
               risk="RIESGO #1 del proyecto: robustez del encendido en M4"),
    "M4": dict(done=False, name="combustor completo + producto",
               verified=("caso e integracion IMPLEMENTADOS y corriendo de "
                         "punta a punta (SDF completo, transpiracion como "
                         "fuente de masa+momento por C_d·A, inyeccion, "
                         "encendido, producto §4.4). Ancla KJ66 dx=2mm, "
                         "206550 celdas, GPU: llama ANCLADA (riesgo #1 "
                         "RESUELTO) con 110.5 kW y oscilacion 0.70% "
                         "(umbral 2%), 21.7 min (<30 min), cierre de masa "
                         "-4.2%, penetraciones Y/d 0.53-0.72, SN 0.769 vs "
                         "0.70; caudal inyectado EXACTO (0.08801 kg/s). "
                         "Medido 2026-08-03 — ESTADO.md §12"),
               caveat=("NO cerrado: T4 = 1445.8 K vs 1150 del balance "
                       "(+25.7%, umbral ±5%) por mezcla incompleta del "
                       "aire de efusion; Δp/p NO comparable con la red 1D "
                       "porque a dx=2mm el anillo liner-carcasa no cabe "
                       "(modo «sin anillo» DECLARADO: el reparto de aire "
                       "es un input, no una prediccion, y la perdida de "
                       "orificio no ocurre en el dominio); el PF esta "
                       "dominado por el modo azimutal n=4 de la MALLA, "
                       "asi que el ranking M4c mediria ruido y no se "
                       "corre. Cerrarlo pide dx <= 1 mm (~1.6 M celdas, "
                       "1-2 h/caso). V9 PRECCINSTA no corrido (script "
                       "escrito)"),
               blocks=["T4 ±5%", "dp/p ±20% (necesita anillo resuelto)",
                       "M4c ranking", "V9 PRECCINSTA"]),
}

#: Qué hace falta para que `solve()` devuelva producto y cómo degrada.
#: (El nombre se conserva de M1–M3, cuando era el motivo de devolver None;
#: hoy documenta la política de degradación.)
UNAVAILABLE_REASON = (
    "M4 IMPLEMENTADO: solve() monta el caso completo del combustor desde "
    "el contrato (SDF con anillo/orificios/swirler, transpiracion de "
    "efusion, inyeccion, encendido) y devuelve el producto del contrato "
    "§4.4. Degrada etiquetado (None + DEGRADE_TAG) si el contrato no "
    "produce caso, el caso diverge o no hay producto medible; si la llama "
    "no ancla dentro del presupuesto, devuelve el producto mixed-is-burned "
    "DECLARADO en fidelity_note (salida de emergencia de WAVE6 §7).")

LAST_STATUS: dict = dict(status="not_run", reason="")

#: Sufijo de degradación que `physics_core._try_higher` puede leer cuando
#: `solve()` devuelve None: `ibrans_L2A(<DEGRADE_TAG>→...)`.
DEGRADE_TAG: str = "unavailable"

#: Presupuestos del caso M4 (ajustables por entorno, contrato §10.7).
L2A_TIMEOUT_S = float(os.environ.get("PHYCB_L2A_TIMEOUT", "1500"))
L2A_MAX_ITER = int(os.environ.get("PHYCB_L2A_MAXITER", "15000"))
L2A_DX_MM = float(os.environ.get("PHYCB_L2A_DX", "0"))    # 0 = automatico
#: Techo de celdas del dx automático: mantiene el caso en la banda
#: 0.5–1 M celdas del plan (GPU, <30 min).
L2A_CELL_CAP = int(os.environ.get("PHYCB_L2A_CELLS", "900000"))
#: Iteraciones MÍNIMAS para fiarse de un producto: un caso cortado antes
#: (presupuesto ridículo, timeout inmediato) devuelve None etiquetado en
#: vez de un PF de un campo sin establecer — eso sería mentir.
L2A_MIN_ITERS = int(os.environ.get("PHYCB_L2A_MIN_ITERS", "1000"))


def milestone_status() -> dict:
    """Estado de los cuatro hitos + disponibilidad de submodelos. Lo lee el
    reporte para que la etiqueta de fidelidad no dependa de la memoria de
    nadie."""
    return dict(
        milestones={k: dict(v) for k, v in MILESTONES.items()},
        turbulence_available=turbulence.available(),
        chemistry_available=chem.available(),
        backend=backend.get_backend().summary(),
        honesty=post.HONESTY, reason=UNAVAILABLE_REASON)


def _auto_dx_mm(contract: dict) -> float:
    """dx del caso: el más fino de la banda 1–3 mm cuyo dominio cabe en
    `L2A_CELL_CAP` celdas (estimación geométrica, bucle ACOTADO)."""
    try:
        cas = (contract.get("casing") or {}).get("profile") or []
        r = max(float(p[1]) for p in cas)
        prof = (contract.get("liner") or {}).get("profile") or []
        zs = [float(p[0]) for p in prof]
        L = max(zs) - min(zs)
    except Exception:
        return 2.0
    for dx in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        cells = (2.0 * r / dx) ** 2 * (L / dx)
        if cells <= L2A_CELL_CAP:
            return dx
    return 3.0


def solve(theta, rec: dict | None = None, *, dx_mm: float | None = None,
          timeout_s: float | None = None, max_iter: int | None = None,
          reactive: bool = True) -> dict | None:
    """Punto de entrada L2a del contrato §8 — **la firma exacta que llama
    `physics_core._try_higher`** (los kwargs extra tienen default).

    Cadena: θ (+record) → `geometry_generator.build_contract` → caso M4
    (`case.build_case`) → `run_case` (criterio: llama anclada / presupuesto)
    → `post.combustor_product` → SOLO las claves del contrato §4.4 +
    `fidelity_note`/`honesty`. Después registra el par (θ, medidas) en
    `physics_core.register_hifi_pair` (realimentación de alta fidelidad).

    Degradación (contrato §3): **nunca lanza**; devuelve `None` con el
    porqué en `LAST_STATUS` y el sufijo corto en `DEGRADE_TAG`
    (p. ej. `ibrans_L2A(case_error→crn_L1)`). Si la llama NO ancla dentro
    del presupuesto, aplica la salida de emergencia DECLARADA
    *mixed-is-burned* (T = f(Z) de equilibrio, `chem.equilibrium_T_of_Z`) y
    devuelve producto con esa etiqueta: conserva PF/penetraciones/puntos
    calientes, pierde cinética (decisión de la semana 15, WAVE6 §7).
    """
    global LAST_STATUS, DEGRADE_TAG

    def _fail(tag: str, reason: str):
        global LAST_STATUS, DEGRADE_TAG
        DEGRADE_TAG = tag
        LAST_STATUS = dict(status=f"unavailable:{tag}", reason=reason)
        return None

    DEGRADE_TAG = "unavailable"
    try:
        import geometry_generator as gg
        import physics_core as pc
    except Exception as exc:
        return _fail("no_siblings",
                     f"geometry_generator/physics_core no importables "
                     f"({type(exc).__name__}: {exc})")
    try:
        if not isinstance(rec, dict) or not rec:
            rec = pc.evaluate(theta, use_cache=False)
        contract = gg.build_contract(theta, rec)
    except Exception as exc:
        return _fail("contract_error",
                     f"build_contract fallo: {type(exc).__name__}: {exc}")
    budget = float(L2A_TIMEOUT_S if timeout_s is None else timeout_s)
    cfg = CaseConfig(
        dx_mm=float(dx_mm if dx_mm else (L2A_DX_MM or
                                         _auto_dx_mm(contract))),
        reactive=bool(reactive), ignition=bool(reactive),
        max_iter=int(L2A_MAX_ITER if max_iter is None else max_iter),
        timeout_s=0.9 * budget,
        t_ad_K=(float(rec["T_ad_st_K"]) if isinstance(
            rec.get("T_ad_st_K"), (int, float)) else None))
    try:
        s = build_case(contract, cfg)
    except Exception as exc:
        return _fail("case_error",
                     f"build_case fallo: {type(exc).__name__}: {exc}")
    try:
        info = run_case(s)
    except Exception as exc:
        return _fail("run_error", f"{type(exc).__name__}: {exc}")
    if info.get("status") == "diverged":
        return _fail("diverged", f"caso divergido: {info.get('note', '')}")
    if int(info.get("iterations", 0)) < L2A_MIN_ITERS \
            and not info.get("anchored"):
        return _fail("budget_too_small",
                     f"el caso corrió {info.get('iterations')} iteraciones "
                     f"(< {L2A_MIN_ITERS}, estado '{info.get('status')}'): "
                     "un campo sin establecer no es producto")

    note = [f"caso M4 dx={cfg.dx_mm} mm, {info.get('cells')} celdas, "
            f"{info.get('iterations')} iteraciones, "
            f"{info.get('wall_time_s', 0.0):.0f} s "
            f"({(info.get('backend') or {}).get('backend', '?')}), "
            f"estado '{info.get('status')}'"]
    mib = False
    pre: dict = {}
    sp = (s.m4.get("models") or {}).get("species")
    en = (s.m4.get("models") or {}).get("energy")
    if reactive and sp is not None and en is not None \
            and not info.get("anchored"):
        # salida de emergencia DECLARADA (WAVE6 §7): mixed-is-burned.
        # Los splits y el Δp se miden ANTES de sobrescribir T: con la ρ
        # de equilibrio y la u del caso no quemado el flujo másico es
        # inconsistente (medido: cierre de masa −0.78 espurio).
        try:
            sm = case.measure_splits(s)
            pre["splits_measured"] = sm["splits_measured"]
            pre["mass_closure_rel"] = sm["mass_closure_rel"]
            dp0 = post.dp_over_p(s)
            if dp0.get("ok"):
                pre["dp_over_p_cfd"] = dp0["dp_over_p_cfd"]
        except Exception:
            pass
        mib = True
        z = sp.mixture_fraction()
        t3 = float(s.m4["T3_K"])
        t_ad = float(cfg.t_ad_K or sp.t_adiabatic(sp.z_stoichiometric(),
                                                  t3))
        t_mib = chem.equilibrium_T_of_Z(z, t3, t3, t_ad,
                                        sp.z_stoichiometric())
        en.T[...] = s._up(t_mib)
        en.rho[...] = s._up(float(en.cfg.p_thermo_Pa)
                            / (float(en.cfg.r_gas)
                               * np.maximum(t_mib, 150.0)))
        note.append("la llama NO anclo dentro del presupuesto: producto "
                    "MIXED-IS-BURNED (T = f(Z) de equilibrio, "
                    "chem.equilibrium_T_of_Z) — conserva PF/penetraciones/"
                    "puntos calientes, pierde cinetica (eta_comb, CO, LBO); "
                    "salida de emergencia declarada de WAVE6 §7")
    elif reactive and info.get("anchored"):
        note.append(f"llama ANCLADA (criterio anchored() del plan; "
                    f"liberacion {info.get('heat_release_W', 0.0):.3g} W)")

    prod = post.combustor_product(s, theta, rec, contract)
    prod.update(pre)          # medidas consistentes previas al overwrite
    keys = ("PF_cfd", "T4_cfd_K", "dp_over_p_cfd", "yplus_hist",
            "cfd_mesh", "splits_measured", "Y_over_d_measured",
            "V_zones_measured", "EI_NOx_crn", "eta_comb_crn", "RTDF_cfd",
            "T_wall_adiabatic_max_K", "mass_closure_rel")
    sol = {k: prod[k] for k in keys if k in prod and prod[k] is not None}
    if not any(k in sol for k in ("PF_cfd", "T4_cfd_K", "dp_over_p_cfd")):
        return _fail("no_product",
                     "el caso corrio pero no produjo PF/T4/dp medibles: "
                     + str(prod.get("blocks")))
    if prod.get("blocks"):
        note.append("bloques degradados: " + str(prod["blocks"]))
    if mib:
        DEGRADE_TAG = "mixed_is_burned"      # informativo (hay producto)
    sol["fidelity_note"] = "; ".join(note)
    sol["honesty"] = post.HONESTY
    try:        # par (θ, alta fidelidad) → calibración afín de L0/L1.
        # SOLO con llama anclada: calibrar L0/L1 con un producto
        # mixed-is-burned contaminaría la corrección con física degradada.
        y_hifi = {}
        if not mib:
            if "PF_cfd" in sol:
                y_hifi["PF"] = float(sol["PF_cfd"])
            if "dp_over_p_cfd" in sol:
                y_hifi["dp_over_p"] = float(sol["dp_over_p_cfd"])
            if "EI_NOx_crn" in sol:
                y_hifi["EI_NOx"] = float(sol["EI_NOx_crn"])
        if y_hifi:
            pc.register_hifi_pair(theta, y_hifi)
            sol["hifi_pair_registered"] = sorted(y_hifi)
    except Exception:
        pass
    LAST_STATUS = dict(status="mixed_is_burned" if mib else "product",
                       reason="", info={k: v for k, v in info.items()
                                        if not isinstance(v, (list, dict))},
                       anchored=bool(info.get("anchored", False)),
                       annulus_resolved=bool(
                           s.m4["meta"].get("annulus_resolved")),
                       dx_mm=cfg.dx_mm)
    return sol


def cold_smoke(contract: dict, dx_mm: float = 3.0, max_iter: int = 200,
               timeout_s: float = 60.0, cfl: float = 0.4,
               u_ref: float | None = None) -> dict:
    """Prueba de **fontanería de arquitectura** sobre la geometría real:
    contrato → SDF analítico del liner → malla cartesiana → máscaras y
    fantasmas del IB → N iteraciones del solver M1 en frío.

    Sirve para tres cosas y sólo para tres: (1) demostrar que la cadena sin
    mallado funciona sobre el contrato de verdad, (2) medir el coste real
    en s/celda/paso a tamaño realista, (3) dar el recuento de celdas
    fluido/sólido/fantasma que el M4 heredará.

    **NO es un producto**: no da splits de aire, ni penetraciones, ni Δp/p
    — eso necesita el caso completo del M4 (swirler, orificios, fuentes de
    combustible, pared efusiva). **Nunca lanza.**
    """
    try:
        m, meta = mesh_from_contract(contract, dx_mm=dx_mm)
        patches = patches_from_contract(contract)
        pin = next((p for p in patches if p.kind == "inlet"), None)
        pout = next((p for p in patches if p.kind == "outlet"), None)
        uin = float(pin.u[2]) if (pin and pin.u) else 20.0
        uref = float(u_ref if u_ref else max(abs(uin), 1.0))
        bc = make_bcs(
            xlo="slip", xhi="slip", ylo="slip", yhi="slip",
            zlo=dict(kind="inlet", u=(0.0, 0.0, uin)),
            zhi=dict(kind="outlet", p=(pout.p if pout else 0.0)))
        cfg = SolverConfig(rho=float((contract.get(
            "cfd_boundary_conditions") or {}).get("rho3_kgm3", 1.2)),
            nu=1.5e-5, u_ref=uref, l_ref=max(m.dx * 10, 1e-3),
            cfl=cfl, rk="rk2", max_iter=int(max_iter),
            timeout_s=float(timeout_s), tol=0.0, residual_every=25)
        s = IBRANSSolver(m, cfg, bc)
        info = s.run()
        info["mesh_meta"] = meta
        info["patches"] = [dict(name=p.name, kind=p.kind, key=p.key)
                           for p in patches]
        info["is_product"] = False
        info["note"] = ("prueba de fonteneria de M1 sobre la geometria real; "
                        "NO es producto L2a (ver UNAVAILABLE_REASON)")
        info["honesty"] = post.HONESTY
        return info
    except Exception as exc:
        return dict(status="error", error=f"{type(exc).__name__}: {exc}",
                    is_product=False)
