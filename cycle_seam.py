"""
QUASAR Phy-CB · cycle_seam.py
=============================
EL SEAM: el punto donde el combustor de Phy-CB se enchufa al ciclo Brayton
de los hermanos, y donde los contratos de Phy-CC / Phy-AC se leen como
especificación de entrada (WAVE4 §§2–3).

Contiene tres cosas:

  1. `hot_cycle(record, TIT_K, ..., combustor=None)` — el ciclo caliente con
     el parámetro `combustor=` que WAVE4 §3 especifica. `combustor=None`
     reproduce EXACTAMENTE el relleno constante documentado
     (ETA_COMB=0.995, DP_COMB=0.05); pasarle un record/callable de Phy-CB
     sustituye η_b y Δp/p por los valores resueltos y añade al dict de
     salida `pattern_factor`, `EI_NOx`, `EI_CO` y `bleed_frac_required`.
  2. `cycle_adapter(design_record)` — el helper que congela la geometría de
     un diseño y expone la firma que el ciclo espera, re-evaluando η y Δp/p
     en las condiciones que el ciclo pida (off-design simple).
  3. `spec_from_run(path)` — lee `runs/<n>/` de Phy-AC o de Phy-CC y
     devuelve un `DesignSpec` de combustor.

--------------------------------------------------------------------------
NOTA DE PROCEDENCIA (importante, leer antes de tocar nada)
--------------------------------------------------------------------------
`docs/CONTEXTO_Integracion.md` §4 describe `Phy-AC/fan_core.py` con
`hot_cycle()` en las líneas 516–629. **Ese archivo no existe en la copia de
Phy-AC de este workspace** (verificado el 2026-07-30: ni `fan_core.py`, ni
`hot_cycle`, ni `GAMMA_HOT`/`CP_HOT`/`ETA_COMB`/`DP_COMB`/`LHV_JET_A`, ni
`BPR` aparecen en ningún `.py` de Phy-AC). La «fase 9.8» del ciclo caliente
no está en este repo.

Por eso se toma la alternativa que la propia WAVE4 deja abierta en su §7
(«sin tocar Phy-AC, cero riesgo»): el seam vive **de este lado**.

  * Si algún día Phy-AC publica `fan_core.hot_cycle`, `_phyac_fan_core()`
    lo detecta e `hot_cycle()` DELEGA en él (y, si su firma ya acepta
    `combustor=`, se lo pasa tal cual). Esa es la ruta preferente.
  * Mientras tanto se usa la implementación de referencia de este módulo:
    el bloque de combustor **copiado literalmente** de la especificación de
    CONTEXTO §4 más un núcleo turbojet ideal (turbina de alta que equilibra
    al compresor + tobera) para poder medir el efecto del combustor sobre
    el TSFC. Va etiquetada con `source="phycb_cycle_reference"` — NO es el
    ciclo autorizado de Phy-AC, es el banco de pruebas del seam.

Todas las salidas llevan el bloque `assumptions` con lo NO modelado, igual
que el original.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

# ==========================================================================
# 0. Constantes del ciclo caliente — copia literal de CONTEXTO §4
#    (Phy-AC/fan_core.py líneas 504-513 según ese documento)
# ==========================================================================
GAMMA_HOT = 4.0 / 3.0     # γ del gas de combustión
CP_HOT = 1148.0           # Mattingly / Walsh & Fletcher
GAMMA_COLD = 1.4
CP_COLD = 1005.0
RGAS = 287.0
LHV_JET_A = 43.1e6        # poder calorífico inferior Jet-A [J/kg]
ETA_COMB = 0.995          # eficiencia de combustión        ← RELLENO CONSTANTE
DP_COMB = 0.05            # pérdida de presión total (5 %)  ← RELLENO CONSTANTE
ETA_POLY_TURB = 0.90
ETA_MECH = 0.995
PSI_LPT_STAGE = 2.2

#: exponente del proxy de off-design del combustor: con la geometría
#: congelada, Δp/p ∝ (ṁ·√T3/P3)². Documentado como PROXY en WAVE4 §3 — no
#: es una correlación validada, es la escala correcta de un orificio fijo.
OFFDESIGN_EXP = 2.0

_PHYAC_HINT = os.environ.get("PHYCB_PHYAC_DIR", "")


# ==========================================================================
# 1. Hook de delegación a Phy-AC (import perezoso, nunca a nivel de módulo)
# ==========================================================================
_FAN_CORE_CACHE: dict = {}


def _phyac_fan_core():
    """Devuelve el módulo `fan_core` de Phy-AC si existe, o None.

    Import PEREZOSO y silencioso (invariante 3 de CONTEXTO §2): Phy-CB
    nunca depende de que Phy-AC esté presente. Se busca en `PYTHONPATH`,
    en `$PHYCB_PHYAC_DIR` y en las rutas hermanas habituales del workspace.
    """
    if "mod" in _FAN_CORE_CACHE:
        return _FAN_CORE_CACHE["mod"]
    mod = None
    try:
        import fan_core as mod           # noqa: F401  (ya en el path)
    except Exception:
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        cands = [_PHYAC_HINT] if _PHYAC_HINT else []
        cands += [
            os.path.join(os.path.dirname(here), "Phy-AC", "Phy-AC"),
            os.path.join(os.path.dirname(here), "Phy-AC"),
        ]
        for d in cands:
            if d and os.path.isfile(os.path.join(d, "fan_core.py")):
                sys.path.insert(0, d)
                try:
                    import fan_core as mod  # noqa: F811
                except Exception:
                    mod = None
                break
    _FAN_CORE_CACHE["mod"] = mod
    return mod


# ==========================================================================
# 2. Resolución del combustor (el seam propiamente dicho)
# ==========================================================================
def _resolve_combustor(combustor, P03, T03, mdot_c, T04, fuel) -> dict:
    """Normaliza el argumento `combustor=` a un dict de resultados.

    Acepta: None (relleno constante), un callable con la firma
    `(P03, T03, mdot, T04, fuel) -> dict`, o directamente un record de
    Phy-CB. Lectura DEFENSIVA: un record incompleto degrada a los valores
    de relleno en las claves que falten, nunca lanza.
    """
    if combustor is None:
        return dict(eta_comb=ETA_COMB, dp_over_p=DP_COMB, extra={},
                    source="constant_filler")
    cb = combustor
    if callable(combustor):
        try:
            cb = combustor(P03=P03, T03=T03, mdot=mdot_c, T04=T04, fuel=fuel)
        except TypeError:
            cb = combustor(P03, T03, mdot_c, T04)
    if not isinstance(cb, dict):
        return dict(eta_comb=ETA_COMB, dp_over_p=DP_COMB, extra={},
                    source="constant_filler(combustor_no_dict)")
    eta_b = float(cb.get("eta_comb", ETA_COMB))
    dp_b = float(cb.get("dp_over_p", DP_COMB))
    if not (0.5 < eta_b <= 1.0) or not (0.0 <= dp_b < 0.5):
        # valores absurdos (θ degenerado): se cae al relleno con etiqueta
        return dict(eta_comb=ETA_COMB, dp_over_p=DP_COMB, extra={},
                    source="constant_filler(combustor_out_of_range)")
    extra = dict(
        pattern_factor=cb.get("PF"),
        profile_radial=cb.get("RTDF"),
        EI_NOx=cb.get("EI_NOx"),
        EI_CO=cb.get("EI_CO"),
        bleed_frac_required=cb.get("bleed_frac_required"),
        T_wall_max_K=cb.get("T_wall_max_K"),
        combustor_source=cb.get("source"),
    )
    return dict(eta_comb=eta_b, dp_over_p=dp_b, extra=extra,
                source=cb.get("source", "phycb"))


def hot_cycle(record: dict, TIT_K: float = 1550.0, M_flight: float = 0.0,
              eta_nozzle: float = 0.98, LHV: float = LHV_JET_A,
              P0a: float | None = None, T0a: float = 288.15,
              fuel: str = "jet_a", combustor=None) -> dict | None:
    """Ciclo caliente: combustor + turbina + tobera sobre un record de
    compresor (Phy-AC o Phy-CC).

    `combustor=None` ⇒ comportamiento del relleno constante documentado en
    CONTEXTO §4, **byte-idéntico** entre llamadas (es la regla de gates de
    la familia: el camino desactivado no cambia ni un bit).

    Si Phy-AC publica su `fan_core.hot_cycle`, esta función DELEGA en él
    (ver la nota de procedencia del módulo). En caso contrario usa la
    implementación de referencia de Phy-CB, etiquetada como tal.
    """
    fc = _phyac_fan_core()
    if fc is not None and hasattr(fc, "hot_cycle"):
        try:
            import inspect
            sig = inspect.signature(fc.hot_cycle)
            if "combustor" in sig.parameters:
                return fc.hot_cycle(record, TIT_K=TIT_K, M_flight=M_flight,
                                    eta_nozzle=eta_nozzle, LHV=LHV,
                                    combustor=combustor)
            # Phy-AC aún sin el parámetro: se corrige a posteriori (WAVE4 §7)
            out = fc.hot_cycle(record, TIT_K=TIT_K, M_flight=M_flight,
                               eta_nozzle=eta_nozzle, LHV=LHV)
            return _retrofit_combustor(out, record, TIT_K, LHV, fuel,
                                       combustor)
        except Exception:
            pass   # cualquier incompatibilidad → implementación propia

    return _reference_hot_cycle(record, TIT_K, M_flight, eta_nozzle, LHV,
                                P0a, T0a, fuel, combustor)


def _retrofit_combustor(out: dict, record: dict, TIT_K, LHV, fuel,
                        combustor) -> dict:
    """Reaplica η_b y Δp/p sobre la salida de un `hot_cycle` de Phy-AC que
    aún no acepta `combustor=` (envoltorio de WAVE4 §7: cero riesgo)."""
    if combustor is None or not isinstance(out, dict):
        return out
    T03 = float(out.get("T03_K", 0.0))
    P03 = float(out.get("P03_Pa", 0.0))
    mdot_c = float(out.get("mdot_core_kg_s", record.get("massflow", 0.0)))
    cb = _resolve_combustor(combustor, P03, T03, mdot_c, TIT_K, fuel)
    eta_b, dp_b = cb["eta_comb"], cb["dp_over_p"]
    f = (CP_HOT * TIT_K - CP_COLD * T03) / max(eta_b * LHV - CP_HOT * TIT_K,
                                               1e3)
    out = dict(out)
    f_old = float(out.get("f_fuel_air", f)) or f
    out["f_fuel_air"] = f
    out["mdot_fuel_kg_s"] = f * mdot_c
    out["P04_Pa"] = P03 * (1.0 - dp_b)
    for k in ("TSFC", "TSFC_kg_Ns", "TSFC_g_kNs"):
        if k in out and f_old:
            out[k] = out[k] * (f / f_old)
    out.update({k: v for k, v in cb["extra"].items() if v is not None})
    asm = dict(out.get("assumptions", {}))
    asm.update(eta_comb=eta_b, dp_comb=dp_b,
               seam="phycb.cycle_seam._retrofit_combustor")
    out["assumptions"] = asm
    return out


def _reference_hot_cycle(record, TIT_K, M_flight, eta_nozzle, LHV,
                         P0a, T0a, fuel, combustor) -> dict | None:
    """Implementación de referencia del ciclo caliente de Phy-CB.

    Bloque de combustor **copiado literalmente** de CONTEXTO §4; el resto
    (turbina de alta que equilibra al compresor, tobera convergente ideal)
    es un núcleo turbojet mínimo cuyo único fin es medir el efecto del
    combustor sobre el TSFC. NO sustituye al ciclo de Phy-AC.
    """
    # --- estación 3: salida del compresor -------------------------------
    PR = record.get("PR") or record.get("PR_tt_stage")
    if PR is None:
        return None
    PR = float(PR)
    P0_in = float(P0a if P0a is not None
                  else record.get("P0_in", record.get("P0a", 101_325.0)))
    T0_in = float(record.get("T0_in", T0a))
    T03 = record.get("T0_out")
    if T03 is None:
        T03 = record.get("T02")
    if T03 is None:
        eta_p = float(record.get("eta_poly", record.get("eta_poly_stage",
                                                        0.88)))
        T03 = T0_in * PR ** ((GAMMA_COLD - 1.0) / (GAMMA_COLD * eta_p))
    T03 = float(T03)
    P03 = P0_in * PR
    mdot_c = float(record.get("massflow", record.get("mdot_kg_s", 1.0)))

    # --- combustor (CONTEXTO §4, líneas 552-561) ------------------------
    T04 = float(TIT_K)
    cb = _resolve_combustor(combustor, P03, T03, mdot_c, T04, fuel)
    eta_b, dp_b = cb["eta_comb"], cb["dp_over_p"]
    cph, cpc = CP_HOT, CP_COLD
    # f por balance de entalpía: f·η_b·LHV = (1+f)·cp_g·T04 − cp·T03
    f = (cph * T04 - cpc * T03) / max(eta_b * LHV - cph * T04, 1e3)
    mdot_fuel = f * mdot_c
    mdot4 = mdot_c * (1.0 + f)
    P04 = P03 * (1.0 - dp_b)

    # --- turbina de alta: equilibra la potencia del compresor -----------
    w_comp = cpc * (T03 - T0_in)                       # J/kg de aire
    dT_turb = w_comp / (ETA_MECH * cph * (1.0 + f))
    T045 = T04 - dT_turb
    if T045 <= 0.0:
        return None
    expo = GAMMA_HOT / ((GAMMA_HOT - 1.0) * ETA_POLY_TURB)
    P045 = P04 * (T045 / T04) ** expo

    # --- tobera convergente ideal a ambiente ----------------------------
    P_amb = P0_in / (1.0 + 0.5 * (GAMMA_COLD - 1.0) * M_flight ** 2) ** (
        GAMMA_COLD / (GAMMA_COLD - 1.0))
    pr_n = max(P045 / max(P_amb, 1.0), 1.0 + 1e-9)
    T9 = T045 * pr_n ** (-(GAMMA_HOT - 1.0) / GAMMA_HOT)
    v9 = math.sqrt(max(2.0 * eta_nozzle * cph * (T045 - T9), 0.0))
    v0 = M_flight * math.sqrt(GAMMA_COLD * RGAS * T0_in)
    thrust = mdot4 * v9 - mdot_c * v0                   # N (tobera adaptada)
    tsfc = mdot_fuel / max(thrust, 1e-6)                # kg/(N·s)

    eta_thermal = (0.5 * (mdot4 * v9 ** 2 - mdot_c * v0 ** 2)
                   / max(mdot_fuel * LHV, 1e-6))

    out = dict(
        schema="phycb-cycle-1",
        source="phycb_cycle_reference",
        T03_K=T03, P03_Pa=P03, T04_K=T04, P04_Pa=P04,
        T045_K=T045, P045_Pa=P045, T9_K=T9,
        f_fuel_air=f, mdot_fuel_kg_s=mdot_fuel, mdot_core_kg_s=mdot_c,
        mdot4_kg_s=mdot4, v9_m_s=v9,
        thrust_N=thrust, TSFC_kg_Ns=tsfc, TSFC_g_kNs=tsfc * 1e6,
        eta_thermal=eta_thermal,
        assumptions=dict(
            cooling_bleed=("no modelado" if combustor is None
                           else "estimado por el combustor"),
            turbine="1 etapa ideal que equilibra al compresor",
            nozzle="convergente adaptada, eta_nozzle constante",
            eta_comb=eta_b, dp_comb=dp_b,
            combustor_seam=cb["source"],
            provenance=("implementación de referencia de Phy-CB: "
                        "Phy-AC/fan_core.py no existe en este repo — "
                        "ver docs/IMPLEMENTATION_CONTRACT.md §0"),
        ),
    )
    out.update({k: v for k, v in cb["extra"].items() if v is not None})
    return out


# ==========================================================================
# 3. Adaptador de combustor para el ciclo (WAVE4 §3)
# ==========================================================================
def cycle_adapter(design_record: dict, theta=None, fidelity=None):
    """Congela un diseño de combustor y devuelve el callable que
    `hot_cycle(..., combustor=...)` espera.

    Off-design (declarado como PROXY, WAVE4 §3): con la geometría fija,
    el parámetro de flujo corregido ṁ·√T3/P3 fija la velocidad de
    referencia, así que Δp/p escala con su cuadrado respecto al punto de
    diseño. η_comb se re-evalúa con la física real si se pasó `theta`
    (re-llamando a `physics_core.evaluate` en las nuevas condiciones); si
    no, se mantiene la del diseño y se etiqueta.
    """
    rec0 = dict(design_record)
    P3_0 = float(rec0.get("P3_Pa", rec0.get("P3", 1.0)))
    T3_0 = float(rec0.get("T3_K", rec0.get("T3", 288.15)))
    m0 = float(rec0.get("mdot_kgs", rec0.get("mdot_air_kgs", 1.0)))
    dp0 = float(rec0.get("dp_over_p", DP_COMB))
    w0 = m0 * math.sqrt(max(T3_0, 1.0)) / max(P3_0, 1.0)

    def _combustor(P03, T03, mdot, T04, fuel=None):
        # --- ruta exacta: re-evaluar la física con el θ congelado -------
        if theta is not None:
            try:
                import physics_core as pc
                th = np.asarray(theta, dtype=float).copy()
                th[16], th[17], th[18], th[19] = (float(P03), float(T03),
                                                  float(mdot), float(T04))
                fid = fidelity if fidelity is not None else pc.Fidelity.L0
                r = pc.evaluate(th, fidelity=fid)
                r = dict(r)
                r["source"] = r.get("source", "") + "|cycle_adapter(exact)"
                return r
            except Exception:
                pass
        # --- ruta proxy: escalado de Δp/p con geometría congelada -------
        w = float(mdot) * math.sqrt(max(float(T03), 1.0)) / max(float(P03),
                                                               1.0)
        dp = dp0 * (w / max(w0, 1e-12)) ** OFFDESIGN_EXP
        out = dict(rec0)
        out["dp_over_p"] = float(np.clip(dp, 0.0, 0.49))
        out["source"] = (str(rec0.get("source", "")) +
                         "|cycle_adapter(offdesign_proxy)")
        return out

    return _combustor


# ==========================================================================
# 4. Lectura de los contratos de los hermanos (WAVE4 §2)
# ==========================================================================
def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_contract(run_dir: str) -> tuple[str, dict] | tuple[None, None]:
    """Localiza el contrato de geometría de un run de Phy-AC o Phy-CC.
    Acepta tanto la carpeta del run como el JSON directo."""
    if os.path.isfile(run_dir):
        d = _load_json(run_dir)
        if d is None:
            return None, None
        schema = str(d.get("schema", ""))
        return ("phyac" if "axial" in schema else "phycc"), d
    for rel, kind in (("geometry/axial_compressor.json", "phyac"),
                      ("geometry/impeller.json", "phycc"),
                      ("axial_compressor.json", "phyac"),
                      ("impeller.json", "phycc")):
        p = os.path.join(run_dir, rel)
        if os.path.isfile(p):
            d = _load_json(p)
            if d is not None:
                return kind, d
    return None, None


def spec_from_run(run_dir: str, T4_K: float | None = None,
                  fuel: str = "jet_a", dp_max: float = 0.04,
                  material: str = "Inconel-625", arch: str | None = None,
                  log=lambda *a: None):
    """`runs/<n>/` de Phy-AC o Phy-CC → `DesignSpec` de combustor.

    Devuelve `(spec, info)` donde `info` documenta de dónde salió cada
    número (el reporte lo publica: es trazabilidad, no adorno) y qué se
    tuvo que derivar. LECTURA DEFENSIVA siempre: los artefactos JSON
    comprometidos pueden ser más viejos que el código (CONTEXTO §10.3).
    """
    from neural_optimizer import DesignSpec       # import perezoso

    kind, d = _find_contract(run_dir)
    if d is None:
        raise FileNotFoundError(
            f"no se encontró contrato de compresor en '{run_dir}' "
            "(se buscó geometry/axial_compressor.json y geometry/impeller.json)")

    info: dict = dict(kind=kind, path=run_dir, derived=[], notes=[],
                      schema=d.get("schema"))

    if kind == "phyac":
        bc = d.get("cfx_boundary_conditions", {}) or {}
        derived = d.get("derived", {}) or {}
        P3 = bc.get("P0_out_Pa_est")
        T3 = bc.get("T0_out_K_est")
        mdot = bc.get("mdot_kg_s")
        if P3 is None:
            P3 = float(bc.get("P0_in_Pa", 101_325.0)) * float(
                derived.get("PR", 1.0))
            info["derived"].append("P3 = P0_in · PR (there was no P0_out_Pa_est)")
        if T3 is None:
            T3 = float(bc.get("T0_in_K", 288.15)) * float(
                derived.get("PR", 1.0)) ** (
                (GAMMA_COLD - 1.0) / (GAMMA_COLD *
                                      float(derived.get("eta_poly", 0.88))))
            info["derived"].append("T3 = T0_in·PR^((gamma-1)/(gamma·eta_poly))")
        if mdot is None:
            mdot = float(derived.get("massflow", 1.0))
            info["derived"].append("mdot taken from the `derived` block")
        m_exit = derived.get("M_exit")
        if m_exit is not None:
            info["notes"].append(
                f"compressor M_exit = {float(m_exit):.3f} (Phy-AC "
                "guarantees <= 0.55 through its g vector: a fit inlet for "
                "the combustor diffuser)")
        ann = d.get("annulus", {}) or {}
        try:
            hub = ann.get("hub") or []
            tip = ann.get("tip") or []
            if hub and tip:
                info["inlet_r_hub_mm"] = float(hub[-1][1])
                info["inlet_r_tip_mm"] = float(tip[-1][1])
                info["notes"].append(
                    f"diffuser inlet annulus: r_hub="
                    f"{info['inlet_r_hub_mm']:.1f} mm, r_tip="
                    f"{info['inlet_r_tip_mm']:.1f} mm")
        except Exception:
            pass
        asm = d.get("assembly", {}) or {}
        if "bleed_stage" in asm:
            info["bleed"] = dict(stage=asm.get("bleed_stage"),
                                 hole_d_mm=asm.get("bleed_hole_d_mm"),
                                 boss_d_mm=asm.get("bleed_boss_d_mm"))
            info["notes"].append(
                f"bleed port available at stage "
                f"{asm.get('bleed_stage')} (d = "
                f"{asm.get('bleed_hole_d_mm')} mm)")
    else:
        perf = d.get("performance", {}) or {}
        bc = d.get("cfx_boundary_conditions", {}) or {}
        inlet = (bc.get("inlet") or {}) if isinstance(bc, dict) else {}
        P0_in = float(inlet.get("P0_Pa", bc.get("P0_in_Pa", 101_325.0)))
        T0_in = float(inlet.get("T0_K", bc.get("T0_in_K", 288.15)))
        # plano de ETAPA, no de impulsor: es la brida física a la que se
        # atornilla el combustor (Phy-CC/VALIDACION.md §2b: mezclarlos es el
        # error #1)
        PR = perf.get("PR_tt_stage")
        eta_p = perf.get("eta_poly_stage")
        plane = "etapa"
        if PR is None:
            PR, eta_p, plane = perf.get("PR"), perf.get("eta_poly"), "impulsor"
            info["notes"].append(
                "⚠ the contract carries no STAGE plane "
                "(PR_tt_stage): the IMPELLER plane is used instead — "
                "it overestimates the pressure at the flange")
        PR = float(PR or 1.0)
        eta_p = float(eta_p or 0.85)
        info["plane"] = plane
        P3 = P0_in * PR
        T3 = perf.get("T02_stage") or perf.get("T02")
        if T3 is None:
            T3 = T0_in * PR ** ((GAMMA_COLD - 1.0) / (GAMMA_COLD * eta_p))
            info["derived"].append(
                "T02 = T0_in·PR^((gamma-1)/(gamma·eta_poly)) "
                "— the Phy-CC contract does not export T02 (gap "
                "documented in CONTEXTO §5.2); the value is DERIVED, "
                "not measured")
        T3 = float(T3)
        mdot = float(bc.get("mdot_kg_s", d.get("massflow", 0.0)) or 0.0)
        if mdot <= 0.0:
            mdot = float(perf.get("massflow", 0.5))
            info["derived"].append("mdot taken from the `performance` block")

    if T4_K is None:
        # default por clase: microturbina con turbina radial no refrigerada
        T4_K = 1150.0
        info["derived"].append("T4 = 1150 K (micro-class default; the "
                               "engineer should pin it with --t4)")

    spec = DesignSpec(P3_Pa=float(P3), T3_K=float(T3), mdot_kgs=float(mdot),
                      T4_K=float(T4_K), fuel=fuel, dp_max=dp_max,
                      arch=arch, material=material)
    info["spec"] = dict(P3_Pa=spec.P3_Pa, T3_K=spec.T3_K,
                        mdot_kgs=spec.mdot_kgs, T4_K=spec.T4_K)
    for line in info["derived"]:
        log(f"  ⚠ derived: {line}")
    for line in info["notes"]:
        log(f"  · {line}")
    return spec, info


# ==========================================================================
if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    rec = dict(PR=4.0, eta_poly=0.90, T0_in=288.15, P0_in=101_325.0,
               T0_out=460.0, massflow=25.0)

    base = hot_cycle(rec, TIT_K=1550.0)
    print("— ciclo con relleno constante (combustor=None) —")
    for k in ("T03_K", "P03_Pa", "P04_Pa", "f_fuel_air", "thrust_N",
              "TSFC_g_kNs", "source"):
        print(f"  {k:12s} = {base[k]}")

    mejor = dict(eta_comb=0.999, dp_over_p=0.030, PF=0.28, EI_NOx=6.4,
                 EI_CO=12.0, bleed_frac_required=0.08, source="correlations_L0")
    con = hot_cycle(rec, TIT_K=1550.0, combustor=mejor)
    print("\n— ciclo con un record de Phy-CB (η=0.999, Δp/p=3 %) —")
    for k in ("f_fuel_air", "P04_Pa", "thrust_N", "TSFC_g_kNs",
              "pattern_factor", "EI_NOx", "bleed_frac_required"):
        print(f"  {k:20s} = {con[k]}")
    d_tsfc = (con["TSFC_g_kNs"] - base["TSFC_g_kNs"]) / base["TSFC_g_kNs"]
    print(f"\n  ΔTSFC = {100 * d_tsfc:+.2f} %  "
          "(η_b mayor y Δp/p menor ⇒ debe MEJORAR)")
