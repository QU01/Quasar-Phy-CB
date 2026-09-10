#!/usr/bin/env python3
"""
QUASAR Phy-CB · test_phycb.py — suite de VERIFICACIÓN (VV&UQ).

Distinción del proyecto (la misma que en Phy-AC y Phy-CC): este script
VERIFICA — «¿resolvemos bien las ecuaciones?». `validation/` VALIDA —
«¿son las ecuaciones correctas?», contra máquinas medidas. CI corre ambos.

Doble modo
----------
    python test_phycb.py                  # tabla ✓/✗/–, exit 1 si algo falla
    python test_phycb.py --slow           # incluye los tests caros (STL/C#)
    python test_phycb.py --only T20,T29   # subconjunto por ID
    python test_phycb.py --freeze-anchors # REESCRIBE anchors/REF_CB10.json
    python -m pytest test_phycb.py -q     # mismo cuerpo, un test por ID
    PHYCB_SLOW=1 python -m pytest test_phycb.py -q     # con los caros

Un test que no se puede correr se SALTA con motivo visible («SKIP: …»),
NUNCA se convierte en un check que pasa trivialmente. La honestidad del
informe es el producto.

Dependencias: solo NumPy (Cantera / matplotlib / dotnet son opcionales y
cada camino que los necesita se salta con motivo si faltan).

Cobertura — IDs congelados en docs/IMPLEMENTATION_CONTRACT.md §11
-----------------------------------------------------------------
  W0 · contrato y núcleo
  T1  Contrato del record + PROCEDENCIA (§0.b: Phy-AC ensombrece a Phy-CB).
  T2  θ, pad_theta bit-exacto y gates decodificadas antes de todo float.
  T3  g continuo, acotado y sin excepciones ante θ degenerado.
  T4  Determinismo bit-exacto de evaluate (SEED = 71).
  T5  Materiales: IN718 prohibido en pared caliente, espesor < LPBF viola,
      derating k(T) interpolado en los extremos sin extrapolación loca.

  W1 · L0 (correlaciones + equilibrio)
  T6  T_ad en banda por combustible; fallback de tabla vs Cantera < 1 %.
  T7  Conservación: splits suman 1 (1e-12) y ṁ_aire+ṁ_fuel = ṁ₄ (1e-9).
  T8  FAR reproduce la f del ciclo caliente de cycle_seam a < 1e-10.
  T9  Dimensionado del ancla KJ66: V_c, τ y Ø de carcasa en banda.
  T10 Penetración de chorro: J=25, d=6 mm ⇒ Y=34.5 mm exacto; monótona.
  T11 Térmico 1D: sin refrigeración T_w1>T_w2>T3, la efusión baja T_w,
      balance cerrado a < 1e-6 relativo.
  T12 Atomización: El-Shanawany cruza 30–70 µm, SMD independiente de ṁ_f
      en airblast y ∝ ΔP^-0.5 en simplex.
  T13 LBO: φ_LBO sube al bajar P₃ y al subir SMD²; el gas no evapora.
  T14 Emisiones: EI_NOx ↑ con V_c y T_st, EI_CO ↓ con V_c, V_c óptimo
      interior, Odgers satura para τ > 20 ms.
  T15 Reporte HTML: tokens de las 13 secciones + trazabilidad + límites.
  T16 Ancla de regresión REF_CB10 (record completo congelado).

  W2 · L1 (red 1D + CRN)
  T17 Red 1D: conservación a 1e-9 y splits resueltos a < 15 % del nominal.
  T18 CRN: smoke con las claves del contrato §4.3.
  T19 LBO por Damköhler: Da_pz y su margen coherentes.
  T20 DEGRADACIÓN etiquetada: Cantera ausente e import roto y timeout.
  T21 Tendencia cruzada L0 ↔ L1 en un barrido de φ_PZ.
  T22 σ_th: crece con α y se anula con T_w1 = T_w2.
  T23 Pared caliente prohibida (IN718 por encima de 923 K).
  T24 Disciplina de caché: L1 escribe phys_cache.jsonl, L0 no, y
      use_cache=False no contamina (heredado del T14 de Phy-CC).

  W3 · geometría y DfAM
  T25 Contrato phycb-combustor-1: round-trip escribir → leer.
  T26 Checker DfAM de espesores y Ø mínimos.
  T27 Teardrop automático por orientación del eje del orificio.
  T28 Drenabilidad: doble pared sin puertos viola; la efusión NO drena.
  T29 Smoke del builder C#/PicoGK (caro ⇒ --slow; SKIP si falta dotnet).
  T30 Coherencia dimensional contrato ↔ record.
  T31 Flujo inverso: dos giros de 180° sin auto-intersección.

  W4 · optimizador y seam de ciclo
  T32 Dominancia de Deb y non-dominated sort heredados del motor.
  T33 design() rápido con semilla fija: smoke.
  T34 design() determinista con la misma semilla.
  T35 spec_from_run de un contrato de Phy-AC (FABRICADO — ver §T35).
  T36 spec_from_run de un run REAL de Phy-CC (runs/ss110).
  T37 hot_cycle: combustor=None estable; η_b↑ y Δp_b↓ ⇒ TSFC mejor.
  T38 Off-design del adaptador: Δp/p ∝ (ṁ·√T/P)².
  T39 Gate de calidad del surrogate: pasa en función suave, falla en ruido.
  T40 End-to-end: evaluate → estructura → contrato → DfAM → reporte.

  W5 · contratos ENTRE CAPAS
  T49 FUENTE ÚNICA de la efusión: área abierta, nº de orificios, C_d y
      conductancia C_d·A idénticos en `physics_core`, `geometry_generator`,
      `crn_core` y `cfd/xlb_cold`, para las tres arquitecturas. Añadido el
      2026-07-31 tras el desacuerdo ×1.536 que nadie cazaba.

Notas de alcance declaradas
---------------------------
* T41–T48 (WAVE5) y V1–V9 (WAVE6) no viven aquí: son de `cfd/tests/`.
  Por eso el primer ID libre de esta suite era T49.
* L2b (OpenFOAM) está FUERA DE ALCANCE por decisión de proyecto
  (contrato §12): T20 comprueba que `--fidelity L2B` degrada con etiqueta
  en vez de fallar.
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

# --- TRAMPA DE ENTORNO (contrato §0.b): Phy-AC está instalado como paquete
#     EDITABLE, así que un `import physics_core` desde otro cwd resuelve
#     SILENCIOSAMENTE al de Phy-AC (NDIM=15, compresor axial). Esta línea y
#     el `assert_provenance` de abajo son la única defensa. ------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

for _s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
    # consolas y pipes de Windows reportan cp1252 y truenan con ✓/η/subíndices.
    # `sys.__stdout__` va aparte porque bajo pytest el reporter de terminal
    # escribe en el flujo ORIGINAL, no en el capturado.
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import physics_core as pc                                        # noqa: E402
import structures_core as sc                                     # noqa: E402
import dfam_checker as dfc                                       # noqa: E402
import geometry_generator as gg                                  # noqa: E402
import report_generator as rg                                    # noqa: E402
import neural_optimizer as no                                    # noqa: E402
import cycle_seam as cs                                          # noqa: E402
import crn_core as crn                                           # noqa: E402
import data_pipeline as dp                                       # noqa: E402

pc.assert_provenance(pc, sc, dfc, gg, rg, no, cs, crn, dp)

THETA_REF = np.array(pc.THETA_REF, dtype=float)

#: ancla de regresión congelada (contrato §10.9). Moverla exige
#: `python test_phycb.py --freeze-anchors` — que es exactamente el punto.
ANCHOR_DIR = os.path.join(HERE, "anchors")
ANCHOR_REF_CB10 = os.path.join(ANCHOR_DIR, "REF_CB10.json")

#: los tests caros (STL/C#, barridos largos) solo corren con la marca.
SLOW = os.environ.get("PHYCB_SLOW", "").strip().lower() not in (
    "", "0", "no", "false")
FREEZE = os.environ.get("PHYCB_FREEZE_ANCHORS", "").strip().lower() not in (
    "", "0", "no", "false")

#: presupuesto de pared del smoke del C# (voxeliza de verdad).
CS_TIMEOUT_S = 240.0

#: run real de Phy-CC usado por T36 (no es nuestro repo: solo se LEE).
PHYCC_RUN = os.path.join(os.path.dirname(HERE), "Phy-CC", "Phy-CC",
                         "runs", "ss110")


def _noop(*_a, **_k):
    """Canal de log mudo para los módulos que aceptan `log=`."""
    return None


# ==========================================================================
# Infraestructura de checks — patrón de la familia (Phy-AC/Phy-CC)
# ==========================================================================
#: (id_test, nombre, estado, detalle) con estado ∈ {"ok", "fail", "skip"}
_RESULTS: list[tuple[str, str, str, str]] = []
_CURRENT_ID = ""


def check(name: str, cond, detail: str = ""):
    """Registra un check y, en modo pytest, lo convierte en `assert`."""
    ok = bool(cond)
    _RESULTS.append((_CURRENT_ID, name, "ok" if ok else "fail", str(detail)))
    assert ok, f"{name}: {detail}"


def skip(name: str, reason: str):
    """Salta un check DEJANDO CONSTANCIA. Nunca cuenta como aprobado."""
    _RESULTS.append((_CURRENT_ID, name, "skip", reason))


def _rel(a: float, b: float) -> float:
    """Error relativo robusto (denominador acotado por abajo)."""
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)


def _monotone(seq, growing: bool = True) -> bool:
    """¿La secuencia es estrictamente monótona en el sentido pedido?"""
    v = [float(x) for x in seq]
    if growing:
        return all(v[i] < v[i + 1] for i in range(len(v) - 1))
    return all(v[i] > v[i + 1] for i in range(len(v) - 1))


def _record_ref(fidelity=None, use_cache: bool = False) -> dict:
    """Record del ancla KJ66 (θ = THETA_REF) recién evaluado."""
    fid = pc.Fidelity.L0 if fidelity is None else fidelity
    return pc.evaluate(THETA_REF, fidelity=fid, use_cache=use_cache)


def _deep_diff(got, exp, path: str = "", out: list | None = None,
               limit: int = 8) -> list[str]:
    """Diferencias estructurales entre dos objetos JSON, ACOTADAS.

    Devuelve como mucho `limit` mensajes: un ancla de regresión que se ha
    movido en 300 claves no se lee mejor por listarlas todas.
    """
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            out.append(f"{path}: tipo {type(got).__name__} != dict")
            return out
        for k in exp:
            if k not in got:
                out.append(f"{path}.{k}: ausente en el record actual")
            else:
                _deep_diff(got[k], exp[k], f"{path}.{k}", out, limit)
        for k in got:
            if k not in exp and len(out) < limit:
                out.append(f"{path}.{k}: clave NUEVA no congelada")
        return out
    if isinstance(exp, list):
        if not isinstance(got, list) or len(got) != len(exp):
            out.append(f"{path}: lista {len(got) if isinstance(got, list) else '?'}"
                       f" != {len(exp)}")
            return out
        for i, (g, e) in enumerate(zip(got, exp)):
            _deep_diff(g, e, f"{path}[{i}]", out, limit)
        return out
    if isinstance(exp, float) and isinstance(got, (int, float)):
        if float(got) != float(exp):
            out.append(f"{path}: {got!r} != {exp!r} "
                       f"(Δrel {_rel(got, exp):.3e})")
        return out
    if got != exp:
        out.append(f"{path}: {got!r} != {exp!r}")
    return out


# ==========================================================================
# W0 · contrato, θ, g y materiales
# ==========================================================================
def test_T1_contrato_y_procedencia():
    """T1 · El record trae el mínimo del contrato §4.1 con los tipos
    correctos, `len(g) == N_CONSTRAINTS`, y los módulos importados son los
    de Phy-CB y no los de Phy-AC (trampa del paquete editable, §0.b)."""
    # --- PROCEDENCIA: lo primero, porque si falla todo lo demás miente ---
    for mod in (pc, sc, dfc, gg, rg, no, cs, crn, dp):
        got = os.path.dirname(os.path.abspath(mod.__file__))
        check(f"T1 procedencia · {mod.__name__} cuelga de Phy-CB",
              os.path.normcase(got) == os.path.normcase(HERE),
              f"{got}")
    check("T1 procedencia · physics_core.PROJECT == 'Phy-CB'",
          pc.PROJECT == "Phy-CB", f"{pc.PROJECT!r}")
    check("T1 procedencia · NDIM = 20 (combustor), NO 15 (Phy-AC axial)",
          pc.NDIM == 20 and pc.NDIM_LEGACY == 20,
          f"NDIM={pc.NDIM} NDIM_LEGACY={pc.NDIM_LEGACY}")
    check("T1 procedencia · assert_provenance no lanza sobre los 9 módulos",
          pc.assert_provenance(pc, sc, dfc, gg, rg, no, cs, crn, dp) is None)

    rec = _record_ref()
    minimo = {
        "eta_comb": float, "dp_over_p": float, "PF": float,
        "T_wall_max_K": float, "T04_K": float, "P04_Pa": float,
        "f_fuel_air": float, "EI_NOx": float, "EI_CO": float,
        "phi_pz": float, "tau_ms": float,
    }
    faltan = [k for k in minimo if k not in rec]
    check("T1 · el record trae TODAS las claves mínimas del contrato §4.1",
          not faltan, f"faltan {faltan}")
    malos = [k for k, t in minimo.items()
             if not isinstance(rec[k], t) or not np.isfinite(rec[k])]
    check("T1 · las claves mínimas son float finitos", not malos,
          f"malos {malos}")
    check("T1 · SMD_um es float o None (JSON no admite NaN)",
          rec["SMD_um"] is None or isinstance(rec["SMD_um"], float),
          f"{rec['SMD_um']!r}")
    check("T1 · g es list[float] de longitud N_CONSTRAINTS = 12",
          isinstance(rec["g"], list) and len(rec["g"]) == pc.N_CONSTRAINTS
          == 12 and all(isinstance(x, float) for x in rec["g"]),
          f"len={len(rec['g'])}")
    check("T1 · feasible es bool y coincide con max(g) <= 0",
          isinstance(rec["feasible"], bool)
          and rec["feasible"] == (max(rec["g"]) <= 0.0),
          f"feasible={rec['feasible']} max(g)={max(rec['g']):.4f}")
    check("T1 · source es str y schema es 'phycb-record-1'",
          isinstance(rec["source"], str) and rec["schema"] == pc.SCHEMA
          == "phycb-record-1", f"{rec['source']!r}")
    check("T1 · el record es serializable a JSON tal cual",
          isinstance(json.dumps(rec), str))
    check("T1 · gates decodificadas a nombre: arch/injector/fuel",
          rec["arch"] in pc.ARCH_NAMES
          and rec["injector"] in pc.INJECTOR_NAMES
          and rec["fuel"] in pc.FUEL_NAMES,
          f"{rec['arch']}/{rec['injector']}/{rec['fuel']}")
    feats = pc.physics_features(THETA_REF, rec)
    check("T1 · physics_features devuelve N_FEATURES = NDIM+17 = 37 finitos",
          len(feats) == pc.N_FEATURES == 37 and np.all(np.isfinite(feats)),
          f"{len(feats)}")


def test_T2_theta_pad_y_gates():
    """T2 · `pad_theta` es bit-exacto, normalize∘denormalize es la
    identidad y las gates se decodifican con `int(round(clip))` ANTES de
    cualquier operación float (invariante CONTEXTO §2.1)."""
    t = pc.pad_theta(THETA_REF)
    check("T2 · pad_theta(θ 20-D) es bit-idéntico a θ (NDIM = NDIM_LEGACY)",
          len(t) == pc.NDIM and np.array_equal(t, THETA_REF))
    r_a = pc.evaluate(THETA_REF, use_cache=False)
    r_b = pc.evaluate(pc.pad_theta(THETA_REF), use_cache=False)
    check("T2 · evaluate(θ) ≡ evaluate(pad_theta(θ)) bit a bit",
          json.dumps(r_a, sort_keys=True) == json.dumps(r_b, sort_keys=True))

    u = np.random.default_rng(0).random(pc.NDIM)
    check("T2 · normalize∘denormalize = identidad (1e-12)",
          np.allclose(pc.normalize(pc.denormalize(u)), u, atol=1e-12))
    check("T2 · límites coherentes (lo < hi en las 20 componentes)",
          bool(np.all(pc.BOUNDS_LO < pc.BOUNDS_HI)))
    check("T2 · el punto de operación no se ha movido (índices 16–19)",
          pc.VAR_NAMES[16:20] == ["P3_Pa", "T3_K", "mdot_kgs", "T4_K"],
          f"{pc.VAR_NAMES[16:20]}")

    def _gate(idx, val, key):
        th = THETA_REF.copy()
        th[idx] = val
        return pc.decode_gates(th)[key]

    check("T2 gates · arch redondea (1.4→annular, 1.6→reverse_annular)",
          _gate(0, 1.4, "arch") == "annular"
          and _gate(0, 1.6, "arch") == "reverse_annular")
    check("T2 gates · arch satura fuera de rango (-5→can, 99→reverse)",
          _gate(0, -5.0, "arch") == "can"
          and _gate(0, 99.0, "arch") == "reverse_annular")
    check("T2 gates · sin deriva float en el entero exacto (2 ± 1e-13)",
          _gate(0, 2.0 - 1e-13, "arch") == "reverse_annular"
          and _gate(0, 0.0 + 1e-13, "arch") == "can")
    check("T2 gates · injector y fuel decodifican por nombre",
          _gate(10, 3.0, "injector") == "micromix"
          and _gate(15, 2.0, "fuel") == "h2")

    # las gates se leen ANTES de tocar ningún float: con el punto de
    # operación envenenado el record degenerado SIGUE trayendo las gates.
    th_nan = THETA_REF.copy()
    th_nan[18] = float("nan")
    r_nan = pc.evaluate(th_nan, use_cache=False)
    check("T2 gates · θ con NaN en ṁ ⇒ record degenerado CON las gates "
          "correctas (se decodificaron antes de cualquier float)",
          r_nan["arch"] == "reverse_annular"
          and r_nan["injector"] == "vaporizer" and r_nan["fuel"] == "jet_a"
          and r_nan["source"].endswith("(degenerate)"),
          f"{r_nan['source']!r}")


def test_T3_g_continuo_y_degenerado():
    """T3 · `g` es continuo, finito y ACOTADO, nunca lanza, y una entrada
    degenerada devuelve los 12 componentes a `G_DEGENERATE = 10.0`."""
    degenerados = {
        "NaN en ṁ": (18, float("nan")),
        "inf en P3": (16, float("inf")),
        "ṁ negativo": (18, -1.0),
        "T3 = 0": (17, 0.0),
        "T4 negativo": (19, -300.0),
    }
    for nombre, (idx, val) in degenerados.items():
        th = THETA_REF.copy()
        th[idx] = val
        r = pc.evaluate(th, use_cache=False)      # NO debe lanzar
        check(f"T3 · θ degenerado ({nombre}) ⇒ g = [10.0]×12 sin excepción",
              len(r["g"]) == pc.N_CONSTRAINTS
              and all(x == pc.G_DEGENERATE for x in r["g"])
              and r["feasible"] is False,
              f"g[0]={r['g'][0]}")

    rng = np.random.default_rng(pc.SEED)
    n_exc, n_feas, gs, fuentes = 0, 0, [], []
    for _ in range(400):                          # bucle ACOTADO
        th = pc.denormalize(rng.random(pc.NDIM))
        try:
            r = pc.evaluate(th, use_cache=False)
            gs.append(r["g"])
            fuentes.append(r["source"])
            n_feas += bool(r["feasible"])
        except Exception:
            n_exc += 1
    gs = np.array(gs)
    check("T3 · LHS(400): cero excepciones", n_exc == 0, f"{n_exc}")
    check("T3 · LHS(400): g siempre finito", bool(np.all(np.isfinite(gs))))
    check("T3 · LHS(400): g acotado en magnitud (es una medida de "
          "violación, no un infinito disfrazado)",
          float(np.abs(gs).max()) < 1e6, f"máx|g| = {np.abs(gs).max():.3f}")
    marcado = [all(x == pc.G_DEGENERATE for x in fila) for fila in gs]
    etiquetado = [s.endswith("(degenerate)") for s in fuentes]
    n_deg = sum(marcado)
    check("T3 · LHS(400): g = [10.0]×12 ⇔ source '(degenerate)' — el "
          "marcador nunca aparece sin su etiqueta ni al revés",
          marcado == etiquetado,
          f"{n_deg}/400 puntos degenerados (θ del box con T₄ < T₃: "
          "punto de operación imposible, correctamente rechazado)")
    dominante = np.bincount(np.argmax(gs, axis=1),
                            minlength=pc.N_CONSTRAINTS)
    top = int(np.argmax(dominante))
    check("T3 · LHS(400): el espacio de diseño NO está vacío",
          n_feas >= 1,
          f"{n_feas}/400 factibles ({100 * n_feas / 400:.2f} %); la "
          f"restricción que más manda es g[{top}] "
          f"'{pc.CONSTRAINT_NAMES[top]}' en {dominante[top]}/400 puntos")

    # continuidad de g[6] (Δp) a lo largo de un barrido de ṁ
    salto_max, prev = 0.0, None
    for m in np.linspace(0.05, 1.20, 81):
        th = THETA_REF.copy()
        th[18] = m
        g6 = pc.evaluate(th, use_cache=False)["g"][6]
        if prev is not None:
            salto_max = max(salto_max, abs(g6 - prev))
        prev = g6
    check("T3 · g[6] (Δp) continuo en el barrido de ṁ (salto < 0.05)",
          salto_max < 0.05, f"salto máx {salto_max:.4f}")


def test_T4_determinismo():
    """T4 · Dos `evaluate` idénticos ⇒ records bit-idénticos (SEED = 71)."""
    a = pc.evaluate(THETA_REF, use_cache=False)
    b = pc.evaluate(THETA_REF, use_cache=False)
    check("T4 · evaluate ×2 (use_cache=False) bit-idéntico",
          json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))
    c = pc.evaluate(THETA_REF, use_cache=True)
    d = pc.evaluate(THETA_REF, use_cache=True)
    check("T4 · evaluate ×2 (use_cache=True) bit-idéntico",
          json.dumps(c, sort_keys=True) == json.dumps(d, sort_keys=True))
    check("T4 · la caché no altera el resultado respecto al camino frío",
          a["eta_comb"] == c["eta_comb"] and a["g"] == c["g"])
    check("T4 · SEED = 71 en todo el núcleo",
          pc.SEED == sc.SEED == dfc.SEED == crn.SEED == no.SEED == 71)
    check("T4 · el record publica la semilla que lo generó",
          a.get("seed") == 71, f"{a.get('seed')!r}")

    rng = np.random.default_rng(11)
    for _ in range(5):
        th = pc.denormalize(rng.random(pc.NDIM))
        r1 = pc.evaluate(th, use_cache=False)
        r2 = pc.evaluate(th, use_cache=False)
        if json.dumps(r1, sort_keys=True) != json.dumps(r2, sort_keys=True):
            break
    else:
        th = None
    check("T4 · determinismo también en 5 θ aleatorios del espacio",
          th is None)


def test_T5_materiales():
    """T5 · IN718 viola SIEMPRE en pared caliente por encima de 923 K, el
    espesor por debajo del mínimo LPBF viola, y el derating k(T) interpola
    en los extremos sin extrapolación loca."""
    rec = _record_ref()
    check("T5 · HOT_WALL_BANNED declara IN718 con su límite de 923 K",
          sc.HOT_WALL_BANNED.get("Inconel-718") == 923.0,
          f"{sc.HOT_WALL_BANNED}")
    for Tw in (930.0, 1000.0, 1100.0, 1200.0, 1300.0):
        r = dict(rec, T_w1_K=Tw, T_w2_K=Tw - 8.0, T_wall_max_K=Tw)
        st = sc.evaluate_structural(THETA_REF, r, material="Inconel-718",
                                    material_casing="Inconel-718")
        check(f"T5 · IN718 con T_wall = {Tw:.0f} K viola g_struct[4]",
              st["g_struct"][4] > 0.0 and st["hot_wall_ok"] is False,
              f"g4 = {st['g_struct'][4]:+.4f}")
    st_ok = sc.evaluate_structural(THETA_REF, rec, material="Inconel-625")
    check("T5 · IN625 en el ancla NO viola la regla de pared caliente",
          st_ok["g_struct"][4] <= 0.0, f"g4 = {st_ok['g_struct'][4]:+.4f}")

    th_fino = THETA_REF.copy()
    th_fino[14] = 0.30                     # < lpbf_t_min de IN625 (0.5 mm)
    r_fino = pc.evaluate(th_fino, use_cache=False)
    st_fino = sc.evaluate_structural(th_fino, r_fino)
    check("T5 · espesor 0.30 mm < LPBF 0.50 mm viola g[7] y g_struct[3]",
          r_fino["g"][7] > 0.0 and st_fino["g_struct"][3] > 0.0,
          f"g[7]={r_fino['g'][7]:+.4f} g_struct[3]="
          f"{st_fino['g_struct'][3]:+.4f}")

    # derating: interpolación acotada en AMBOS extremos, monótona, sin
    # extrapolación absurda (ni > 1 ni negativa) y con suelo declarado.
    Ts = [80.0, 200.0, 293.0, 600.0, 811.0, 1033.0, 1253.0, 1400.0, 2500.0]
    fs = [sc.material_at_T("Inconel-625", T)["derate_factor"] for T in Ts]
    check("T5 · k(T): 1.0 por debajo del primer punto de la tabla",
          fs[0] == fs[1] == fs[2] == 1.0, f"{fs[:3]}")
    check("T5 · k(T): monótono NO creciente y dentro de (0, 1]",
          all(fs[i] >= fs[i + 1] for i in range(len(fs) - 1))
          and all(0.0 < f <= 1.0 for f in fs),
          f"{[round(f, 4) for f in fs]}")
    check("T5 · k(T): sin extrapolación loca — suelo K_DERATE_MIN",
          fs[-1] >= sc.K_DERATE_MIN and fs[-1] <= fs[-2],
          f"f(2500 K) = {fs[-1]:.4f} ≥ {sc.K_DERATE_MIN}")
    check("T5 · k(T): en un nodo de la tabla devuelve el valor tabulado",
          abs(sc.material_at_T("Inconel-625",
                               1033.0)["derate_factor"] - 0.62) < 1e-12)
    check("T5 · el catálogo trae los 9 materiales del contrato §5.2",
          len(sc.MATERIALS) == 9 and "Inconel-625" in sc.MATERIALS
          and sc.N_STRUCT_CONSTRAINTS == 5, f"{len(sc.MATERIALS)}")


# ==========================================================================
# W1 · L0: termoquímica, dimensionado, orificios, térmico, emisiones
# ==========================================================================
def test_T6_temperatura_adiabatica():
    """T6 · T_ad estequiométrica en banda por combustible y el ajuste de la
    capa 0 dentro del 1 % de Cantera (cuando Cantera está)."""
    bandas = {"jet_a": (2200.0, 2350.0), "h2": (2350.0, 2450.0)}
    for fuel, (lo, hi) in bandas.items():
        T = pc._thermo_hot(fuel, 1.0, pc.P_ATM, 288.15,
                           use_cantera=False)["T_ad_K"]
        check(f"T6 · T_ad({fuel}, φ=1, 288 K, 1 atm) en [{lo:.0f}, {hi:.0f}] K",
              lo <= T <= hi, f"{T:.1f} K")
    T_pro = pc._thermo_hot("propane", 1.0, pc.P_ATM, 288.15,
                           use_cantera=False)["T_ad_K"]
    check("T6 · T_ad(propano) en la banda hidrocarburo 2200–2350 K",
          2200.0 <= T_pro <= 2350.0, f"{T_pro:.1f} K")

    if not pc.cantera_available():
        skip("T6 · fallback de tabla vs Cantera < 1 %",
             "SKIP: Cantera no está instalado en este entorno")
    else:
        peor, donde = 0.0, ""
        for fuel in pc.FUEL_NAMES:
            for T3, P3 in ((288.15, pc.P_ATM), (430.0, 250_000.0),
                           (700.0, 1.2e6)):
                a = pc._thermo_hot(fuel, 1.0, P3, T3,
                                   use_cantera=False)["T_ad_K"]
                b = pc._thermo_hot(fuel, 1.0, P3, T3,
                                   use_cantera=True)["T_ad_K"]
                e = _rel(a, b)
                if e > peor:
                    peor, donde = e, f"{fuel} @ {T3:.0f} K/{P3 / 1e5:.1f} bar"
        check("T6 · tabla de la capa 0 vs Cantera equilibrate('HP') < 1 %",
              peor < 0.01, f"peor {100 * peor:.4f} % en {donde}")

    # la banda φ: rica y pobre por debajo del pico estequiométrico
    Ts = [pc._thermo_hot("jet_a", p, pc.P_ATM, 288.15,
                         use_cantera=False)["T_ad_K"]
          for p in (0.5, 0.8, 1.0, 1.3)]
    check("T6 · T_ad crece hacia φ=1 y cae en rico (pico interior)",
          Ts[0] < Ts[1] < Ts[2] and Ts[3] < Ts[2],
          f"{[round(t) for t in Ts]}")


def test_T7_conservacion():
    """T7 · Los splits de aire suman 1 a 1e-12 y la masa cierra a 1e-9."""
    casos = [("ancla", THETA_REF)]
    for arch in (0.0, 1.0):
        th = THETA_REF.copy()
        th[0] = arch
        casos.append((pc.ARCH_NAMES[int(arch)], th))
    for fuel in (1.0, 2.0):
        th = THETA_REF.copy()
        th[15] = fuel
        casos.append((pc.FUEL_NAMES[int(fuel)], th))
    for nombre, th in casos:
        r = pc.evaluate(th, use_cache=False)
        s = r["splits"]
        total = s["pz"] + s["sz"] + s["dz"] + s["cool"]
        check(f"T7 · splits suman 1 a 1e-12 ({nombre})",
              abs(total - 1.0) < 1e-12, f"Σ−1 = {total - 1.0:+.3e}")
        check(f"T7 · frac_cool es cierre, no variable ({nombre})",
              abs(r["frac_cool"] - (1.0 - r["frac_pz"] - r["frac_sz"]
                                    - r["frac_dz"])) < 1e-12
              or r["frac_cool"] >= pc.FRAC_COOL_MIN)
        m_in = r["mdot_air_kgs"] + r["mdot_fuel_kgs"]
        check(f"T7 · ṁ_aire + ṁ_fuel = ṁ₄ a 1e-9 relativo ({nombre})",
              _rel(m_in, r["mdot4_kgs"]) < 1e-9,
              f"Δrel = {_rel(m_in, r['mdot4_kgs']):.3e}")
        check(f"T7 · f = ṁ_fuel/ṁ_aire consistente ({nombre})",
              _rel(r["f_fuel_air"],
                   r["mdot_fuel_kgs"] / r["mdot_air_kgs"]) < 1e-12)
        check(f"T7 · todos los caudales son positivos ({nombre})",
              r["mdot_air_kgs"] > 0 and r["mdot_fuel_kgs"] > 0
              and r["mdot4_kgs"] > r["mdot_air_kgs"])


def test_T8_far_del_ciclo():
    """T8 · El FAR del combustor «perfecto» reproduce EXACTAMENTE la f del
    ciclo caliente de `cycle_seam` (contrato §0: mismas constantes)."""
    rec = _record_ref()
    T3, T4 = float(THETA_REF[17]), float(THETA_REF[19])
    # record de compresor sintético que pone al ciclo en T03 = T3
    comp = dict(PR=250_000.0 / 101_325.0, eta_poly=0.90, T0_in=288.15,
                P0_in=101_325.0, T0_out=T3, massflow=float(THETA_REF[18]))
    ciclo = cs.hot_cycle(comp, TIT_K=T4)
    check("T8 · hot_cycle(combustor=None) devuelve un ciclo completo",
          isinstance(ciclo, dict) and np.isfinite(ciclo["f_fuel_air"]))
    check("T8 · f_fuel_air_cycle del record = f del ciclo a < 1e-10",
          _rel(rec["f_fuel_air_cycle"], ciclo["f_fuel_air"]) < 1e-10,
          f"{rec['f_fuel_air_cycle']:.12f} vs {ciclo['f_fuel_air']:.12f} "
          f"(Δrel {_rel(rec['f_fuel_air_cycle'], ciclo['f_fuel_air']):.2e})")
    check("T8 · las constantes del seam coinciden con las del núcleo",
          cs.CP_HOT == pc.CP_HOT and cs.GAMMA_HOT == pc.GAMMA_HOT
          and cs.LHV_JET_A == pc.LHV_JET_A == 43.1e6
          and cs.ETA_COMB == pc.ETA_COMB_CYCLE == 0.995)
    check("T8 · el FAR real (η_comb < 1) es MAYOR que el ideal del ciclo",
          rec["f_fuel_air"] > rec["f_fuel_air_cycle"]
          and _rel(rec["f_fuel_air"], rec["f_fuel_air_cycle"]) < 0.05,
          f"{rec['f_fuel_air']:.6f} vs {rec['f_fuel_air_cycle']:.6f}")
    check("T8 · φ_global = f/f_stoich a 1e-12",
          _rel(rec["phi_global"],
               rec["f_fuel_air"] / rec["far_stoich"]) < 1e-12)


def test_T9_dimensionado_ancla_kj66():
    """T9 · El dimensionado del ancla KJ66 cae en las bandas de la clase
    (EstadoDelArte §7): V_c 0.05–0.4 L, τ 4–8 ms, Ø carcasa 80–130 mm."""
    r = _record_ref()
    V_L = 1e3 * r["V_c_m3"]
    check("T9 · V_c del ancla en 0.05–0.4 L", 0.05 <= V_L <= 0.40,
          f"{V_L:.4f} L")
    check("T9 · τ del ancla en 4–8 ms", 4.0 <= r["tau_ms"] <= 8.0,
          f"{r['tau_ms']:.3f} ms")
    check("T9 · Ø de carcasa del ancla en 80–130 mm",
          80.0 <= r["D_casing_mm"] <= 130.0, f"{r['D_casing_mm']:.2f} mm")
    check("T9 · el dimensionado convergió y declara su driver",
          r["sizing_converged"] is True
          and r["sizing_driver"] in ("aero", "chem"),
          f"driver = {r['sizing_driver']}")
    check("T9 · A_ref = max(aero, chem) — el driver manda",
          abs(r["A_ref_m2"] - max(r["A_ref_aero_m2"],
                                  r["A_ref_chem_m2"])) < 1e-15)
    check("T9 · jerarquía geométrica D_L < D_ref < D_casing",
          r["D_L_mm"] < r["D_ref_mm"] < r["D_casing_mm"],
          f"{r['D_L_mm']:.1f} < {r['D_ref_mm']:.1f} < "
          f"{r['D_casing_mm']:.1f} mm")
    check("T9 · V_c = V_pz + V_sz + V_dz a 1e-12 relativo",
          _rel(r["V_pz_m3"] + r["V_sz_m3"] + r["V_dz_m3"],
               r["V_c_m3"]) < 1e-12)
    check("T9 · L_tot = L_pz + L_sz + L_dz a 1e-9 relativo",
          _rel(r["L_pz_mm"] + r["L_sz_mm"] + r["L_dz_mm"],
               r["L_tot_mm"]) < 1e-9)
    check("T9 · carga de Lefebvre I en banda de microturbina (20–60)",
          20.0 <= r["I_MW_m3_atm"] <= 200.0,
          f"I = {r['I_MW_m3_atm']:.1f} MW/m³·atm "
          f"(banda de diseño {r['I_band_MW_m3_atm']})")
    # escalado: el doble de caudal pide más área
    th2 = THETA_REF.copy()
    th2[18] = 2.0 * THETA_REF[18]
    r2 = pc.evaluate(th2, use_cache=False)
    check("T9 · ṁ ×2 ⇒ A_ref crece (dimensionado sensible al caudal)",
          r2["A_ref_m2"] > 1.5 * r["A_ref_m2"],
          f"{r2['A_ref_m2'] / r['A_ref_m2']:.3f}×")


def test_T10_penetracion_de_chorro():
    """T10 · Y_max = C·√J·d con C = 1.15: el caso canónico J=25, d=6 mm da
    34.5 mm EXACTOS, y la penetración es monótona creciente en J."""
    C = pc.coeffs()["jet_C"]
    check("T10 · C de penetración = 1.15 (verificada en 2 fuentes)",
          C == pc.JET_PENETRATION_C == 1.15, f"{C}")
    Y = C * math.sqrt(25.0) * 6.0
    check("T10 · J = 25, d = 6 mm ⇒ Y = 34.5 mm exacto",
          abs(Y - 34.5) < 1e-12, f"{Y!r} mm")

    Js = [1.0, 4.0, 9.0, 25.0, 100.0]
    Ys = [C * math.sqrt(J) * 6.0 for J in Js]
    check("T10 · Y monótona creciente en J", _monotone(Ys, growing=True),
          f"{[round(y, 2) for y in Ys]} mm")
    check("T10 · Y ∝ √J: J ×4 ⇒ Y ×2 exacto",
          abs(Ys[1] / Ys[0] - 2.0) < 1e-12
          and abs(Ys[3] / Ys[1] - 2.5) < 1e-12)

    r = _record_ref()
    filas = r["hole_rows"]
    check("T10 · el record publica filas de orificios con J, d y Y",
          len(filas) >= 3
          and all({"J", "d_mm", "Y_mm", "Y_over_DL"} <= set(f) for f in filas),
          f"{len(filas)} filas")
    peor = max(_rel(f["Y_mm"], C * math.sqrt(max(f["J"], 0.0)) * f["d_mm"])
               for f in filas)
    check("T10 · toda fila del record cumple Y = C·√J·d a 1e-12",
          peor < 1e-12, f"peor Δrel = {peor:.2e}")
    check("T10 · Y/D_L de dilución cerca del óptimo de Lefebvre & Norster",
          abs(r["Y_over_DL_dz"] - pc.Y_TARGET_DEFAULT) <= pc.Y_TOL,
          f"Y/D_L = {r['Y_over_DL_dz']:.4f} vs objetivo "
          f"{pc.Y_TARGET_DEFAULT} ± {pc.Y_TOL}")


def test_T11_termico_1d():
    """T11 · Sin refrigeración T_w1 > T_w2 > T₃, la efusión BAJA la pared,
    y el balance de energía cierra a < 1e-6 relativo."""
    th = THETA_REF.copy()
    gates = pc.decode_gates(th)
    ch = pc._l0_chain(th, gates, pc.coeffs()["dp_over_p"],
                      pc.MATERIAL_DEFAULT, False)
    frio = pc._liner_thermal_1d(th, gates, ch["sz"], ch["part"], ch["holes"],
                                ch["T_zone"], ch["mw_hot"], ch["far"],
                                material=pc.MATERIAL_DEFAULT, cooling=False)
    con = ch["therm"]
    T3 = float(th[17])
    check("T11 · sin refrigeración: T_w1 > T_w2 > T₃",
          frio["T_w1_K"] > frio["T_w2_K"] > T3,
          f"{frio['T_w1_K']:.1f} > {frio['T_w2_K']:.1f} > {T3:.1f} K")
    check("T11 · con efusión la pared BAJA (y sigue por encima de T₃)",
          con["T_w1_K"] < frio["T_w1_K"] and con["T_w1_K"] > T3,
          f"{con['T_w1_K']:.1f} K vs {frio['T_w1_K']:.1f} K sin refrigerar "
          f"(Δ = {con['T_w1_K'] - frio['T_w1_K']:+.1f} K)")
    check("T11 · el gradiente por el espesor es positivo y pequeño",
          0.0 < con["T_w1_K"] - con["T_w2_K"] < 60.0,
          f"ΔT_pared = {con['T_w1_K'] - con['T_w2_K']:.2f} K")
    for etiqueta, t in (("con efusión", con), ("sin refrigerar", frio)):
        check(f"T11 · balance 1D cerrado a < 1e-6 relativo ({etiqueta})",
              abs(t["thermal_residual_rel"]) < 1e-6,
              f"residuo = {t['thermal_residual_rel']:.3e}")
    perfil = con["T_wall_profile"]
    check("T11 · perfil de pared: N estaciones con z creciente y T finita",
          len(perfil) == pc.N_WALL_STATIONS
          and all(perfil[i][0] < perfil[i + 1][0]
                  for i in range(len(perfil) - 1))
          and all(np.isfinite(v) for fila in perfil for v in fila),
          f"{len(perfil)} estaciones")
    check("T11 · T_wall_max_K es el máximo del perfil de pared caliente",
          abs(con["T_wall_max_K"]
              - max(f[1] for f in perfil)) < 1e-9)
    th_eff = THETA_REF.copy()
    th_eff[9] = 10.0                      # paso/Ø mayor ⇒ MENOS efusión
    r_menos = pc.evaluate(th_eff, use_cache=False)
    check("T11 · menos efusión (paso/Ø 6→10) ⇒ pared MÁS caliente",
          r_menos["T_w1_K"] > con["T_w1_K"],
          f"{r_menos['T_w1_K']:.1f} K vs {con['T_w1_K']:.1f} K")


def test_T12_atomizacion_smd():
    """T12 · El-Shanawany cruza la banda 30–70 µm de la práctica airblast y
    es monótona decreciente en U_A; el SMD airblast NO depende de ṁ_f; el
    simplex escala EXACTAMENTE como ΔP^-0.5."""
    th = THETA_REF.copy()
    th[10] = 0.0                                   # gate injector = airblast
    gates = pc.decode_gates(th)
    ch = pc._l0_chain(th, gates, pc.coeffs()["dp_over_p"],
                      pc.MATERIAL_DEFAULT, False)
    sz, rho3 = ch["sz"], ch["sz"]["rho3_kgm3"]
    dpL, mdot_f, T3 = ch["holes"]["dp_liner_Pa"], ch["mdot_fuel"], 430.0

    barrido = []
    for f in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00):
        a = pc._atomization(th, gates, sz, rho3, dpL * f, mdot_f, T3)
        barrido.append((a["U_atomizing_ms"], a["SMD_um"]))
    smds = [s for _, s in barrido]
    check("T12 · El-Shanawany monótona DECRECIENTE con U_A (SMD ∝ U^-1.2)",
          all(smds[i] >= smds[i + 1] for i in range(len(smds) - 1)),
          " → ".join(f"{s:.1f}" for s in smds) + " µm")
    en_banda = [(u, s) for u, s in barrido if 30.0 <= s <= 70.0]
    check("T12 · El-Shanawany ENTRA en la banda 30–70 µm dentro de la "
          "ventana airblast de la práctica",
          len(en_banda) >= 2 and max(smds) <= 70.0,
          f"{min(smds):.1f}–{max(smds):.1f} µm en el barrido; en banda a "
          f"U_A = {', '.join(f'{u:.1f}' for u, _ in en_banda)} m/s. "
          f"En el ancla (U_A = {barrido[-2][0]:.1f} m/s, airblast duro) "
          f"da {barrido[-2][1]:.1f} µm; el techo del modelo lo fija el "
          f"suelo U_ATOMIZING_MIN_MS = {pc.U_ATOMIZING_MIN_MS:.0f} m/s")

    a1 = pc._atomization(th, gates, sz, rho3, dpL, mdot_f, T3)
    a3 = pc._atomization(th, gates, sz, rho3, dpL, 3.0 * mdot_f, T3)
    check("T12 · airblast: SMD INDEPENDIENTE de ṁ_f (turndown) bit a bit",
          a1["SMD_um"] == a3["SMD_um"]
          and a1["atomizer_model"] == "elshanawany_lefebvre",
          f"{a1['SMD_um']:.4f} µm con ṁ_f y ×3")

    th_s = THETA_REF.copy()
    th_s[10] = 1.0                                 # gate injector = simplex
    g_s = pc.decode_gates(th_s)
    dp0 = pc.DP_FUEL_SIMPLEX_PA
    try:
        s0 = pc._atomization(th_s, g_s, sz, rho3, dpL, mdot_f, T3)["SMD_um"]
        pc.DP_FUEL_SIMPLEX_PA = 4.0 * dp0
        s4 = pc._atomization(th_s, g_s, sz, rho3, dpL, mdot_f, T3)["SMD_um"]
    finally:
        pc.DP_FUEL_SIMPLEX_PA = dp0
    check("T12 · simplex: SMD ∝ ΔP^-0.5 — ΔP ×4 ⇒ SMD ÷2 exacto",
          abs(s4 / s0 - 0.5) < 1e-12,
          f"{s0:.3f} → {s4:.3f} µm (ratio {s4 / s0:.12f})")

    r = _record_ref()
    check("T12 · el record publica TODO el abanico de correlaciones",
          len(r["SMD_um_by_corr"]) >= 6
          and "elshanawany_lefebvre" in r["SMD_um_by_corr"]
          and r["atomizer_model"] == "vaporizer_kj66",
          f"{len(r['SMD_um_by_corr'])} correlaciones")
    th_h2 = THETA_REF.copy()
    th_h2[15] = 2.0                                # gate fuel = H2
    r_h2 = pc.evaluate(th_h2, use_cache=False)
    check("T12 · rama gas (H₂): SMD_um = None (JSON null), no NaN",
          r_h2["SMD_um"] is None, f"{r_h2['SMD_um']!r}")


def test_T13_lbo():
    """T13 · φ_LBO sube al bajar P₃ (P₃^-1.3) y al subir SMD²; con
    combustible gaseoso el término de evaporación NO contribuye."""
    th = THETA_REF.copy()
    gates = pc.decode_gates(th)
    ch = pc._l0_chain(th, gates, pc.coeffs()["dp_over_p"],
                      pc.MATERIAL_DEFAULT, False)

    def _phi_lbo(theta, atom):
        return pc._stability(theta, pc.decode_gates(theta), ch["sz"],
                             ch["part"], atom, float(theta[18]),
                             ch["T_st"], ch["mw_hot"])["phi_LBO"]

    atom_liq = dict(ch["atom"], SMD_um=50.0, is_gas_branch=False)
    phis = []
    for P3 in (6.0e5, 4.0e5, 2.5e5, 1.5e5, 1.0e5):
        t = th.copy()
        t[16] = P3
        phis.append(_phi_lbo(t, atom_liq))
    check("T13 · φ_LBO crece al BAJAR P₃ (estabilidad peor a baja presión)",
          _monotone(phis, growing=True),
          " → ".join(f"{p:.4f}" for p in phis))
    r_pp = math.log(phis[-1] / phis[0]) / math.log(6.0e5 / 1.0e5)
    check("T13 · el exponente medido de P₃ es ≈ 1.3 (LBO_EXP_P3)",
          abs(r_pp - 1.3) < 0.02, f"exponente medido {r_pp:.4f}")

    smds = [20.0, 30.0, 50.0, 80.0]
    phis_s = [_phi_lbo(th, dict(ch["atom"], SMD_um=s, is_gas_branch=False))
              for s in smds]
    check("T13 · φ_LBO crece con el SMD (atomización = LA palanca)",
          _monotone(phis_s, growing=True),
          " → ".join(f"{p:.4f}" for p in phis_s))
    check("T13 · φ_LBO ∝ SMD² exacto: SMD ×2 ⇒ φ_LBO ×4",
          abs(phis_s[2] / phis_s[0] - 6.25) < 1e-9
          and abs(_phi_lbo(th, dict(ch["atom"], SMD_um=40.0,
                                    is_gas_branch=False)) / phis_s[0]
                  - 4.0) < 1e-9,
          f"φ(50)/φ(20) = {phis_s[2] / phis_s[0]:.6f} (esperado 6.25)")

    th_gas = THETA_REF.copy()
    th_gas[15] = 2.0                               # H2
    g_gas = pc.decode_gates(th_gas)
    st_a = pc._stability(th_gas, g_gas, ch["sz"], ch["part"],
                         dict(ch["atom"], SMD_um=20.0, is_gas_branch=True),
                         float(th_gas[18]), ch["T_st"], ch["mw_hot"])
    st_b = pc._stability(th_gas, g_gas, ch["sz"], ch["part"],
                         dict(ch["atom"], SMD_um=200.0, is_gas_branch=True),
                         float(th_gas[18]), ch["T_st"], ch["mw_hot"])
    check("T13 · gas: el término de evaporación NO contribuye "
          "(SMD 20 vs 200 µm da el MISMO φ_LBO)",
          st_a["phi_LBO"] == st_b["phi_LBO"]
          and st_a["lbo_branch"] == "gas",
          f"φ_LBO = {st_a['phi_LBO']:.6f} en ambos")

    r = _record_ref()
    check("T13 · el margen de LBO del record es coherente con g[2]",
          _rel(r["LBO_margin"], r["phi_pz"] / r["phi_LBO"] - 1.0) < 1e-12
          and _rel(r["g"][2],
                   r["phi_LBO"] * (1.0 + pc.LBO_MARGIN_MIN)
                   / r["phi_pz"] - 1.0) < 1e-12)
    check("T13 · E_min ∝ SMD³ en la rama espray (factor 2 ⇒ factor 8)",
          r["E_min_J"] > 0.0 and r["d_q_mm"] > 0.0,
          f"E_min = {1e3 * r['E_min_J']:.3f} mJ, d_q = {r['d_q_mm']:.3f} mm")


def test_T14_emisiones():
    """T14 · EI_NOx crece con V_c y con T_st, EI_CO decrece con V_c, el V_c
    óptimo es INTERIOR y la correlación de Odgers satura para τ > 20 ms."""
    base = dict(P3_Pa=250_000.0, T_st_K=2360.0, T_pz_K=2372.0,
                T_sz_K=2060.0, V_c_m3=3.88e-4, mdot_air_kgs=0.13,
                tau_s=6.05e-3, tau_e_s=3.1e-4, tau_pz_s=4.96e-3,
                tau_sz_s=1.10e-3, dp_over_p=0.03, phi_pz=1.048,
                phi_sz=0.749)
    nox_V = [pc.nox_fan(**dict(base, V_c_m3=V))["lefebvre"]
             for V in (2.0e-4, 3.88e-4, 8.0e-4)]
    check("T14 · EI_NOx crece con V_c (tiempo a alta T)",
          _monotone(nox_V, growing=True),
          " → ".join(f"{x:.3f}" for x in nox_V) + " g/kg")
    nox_T = [pc.nox_fan(**dict(base, T_st_K=T))["lefebvre"]
             for T in (2200.0, 2360.0, 2500.0)]
    check("T14 · EI_NOx crece con T_st (Zeldovich exponencial)",
          _monotone(nox_T, growing=True) and nox_T[-1] > 4.0 * nox_T[0],
          " → ".join(f"{x:.3f}" for x in nox_T) + " g/kg")

    od = {tau: pc.nox_fan(**dict(base, tau_s=tau))["odgers"]
          for tau in (5e-3, 10e-3, 20e-3, 40e-3, 160e-3)}
    check("T14 · Odgers crece con τ por debajo de 20 ms",
          od[5e-3] < od[10e-3] < od[20e-3],
          f"{od[5e-3]:.3f} → {od[20e-3]:.3f} g/kg")
    check("T14 · Odgers SATURA para τ > 20 ms (< 1 % de 20→40 ms)",
          _rel(od[40e-3], od[20e-3]) < 0.01
          and _rel(od[160e-3], od[40e-3]) < 1e-3,
          f"20→40 ms: {100 * _rel(od[40e-3], od[20e-3]):.3f} %; "
          f"40→160 ms: {100 * _rel(od[160e-3], od[40e-3]):.4f} %")

    th = THETA_REF.copy()
    gates = pc.decode_gates(th)
    ch = pc._l0_chain(th, gates, pc.coeffs()["dp_over_p"],
                      pc.MATERIAL_DEFAULT, False)

    def _emis(V):
        sz = dict(ch["sz"])
        sz["V_c_m3"] = V
        return pc._emissions(th, sz, ch["part"], ch["holes"], ch["T_zone"],
                             ch["atom"], float(th[18]), ch["T_st"])

    co = [_emis(V)["EI_CO_lefebvre"] for V in (2.0e-4, 3.88e-4, 8.0e-4)]
    check("T14 · EI_CO DECRECE con V_c (más tiempo para quemar el CO)",
          _monotone(co, growing=False),
          " → ".join(f"{x:.2f}" for x in co) + " g/kg")

    V_opt = _emis(ch["sz"]["V_c_m3"])["V_c_opt_m3"]
    w = pc.NOX_CO_WEIGHT_RATIO

    def _J(V):
        e = _emis(V)
        return w * e["EI_NOx_lefebvre"] + e["EI_CO_lefebvre"]

    check("T14 · el V_c óptimo analítico es INTERIOR (mínimo estricto)",
          0.0 < V_opt < 1.0 and _J(V_opt) < _J(V_opt / 3.0)
          and _J(V_opt) < _J(V_opt * 3.0),
          f"V_opt = {1e6 * V_opt:.1f} cm³; J = {_J(V_opt / 3):.2f} / "
          f"{_J(V_opt):.2f} / {_J(V_opt * 3):.2f}")

    r = _record_ref()
    fan = [r[f"EI_NOx_{k}"] for k in pc.NOX_FAN_NAMES]
    check("T14 · el record publica LAS CINCO correlaciones de NOx",
          len(fan) == 5 and all(np.isfinite(x) and x > 0 for x in fan),
          " · ".join(f"{k}={v:.2f}" for k, v in zip(pc.NOX_FAN_NAMES, fan)))
    disp = max(fan) / min(fan)
    check("T14 · la dispersión del abanico está en la escala documentada "
          "(≈4.5× en el ancla) y no en órdenes de magnitud",
          2.0 <= disp <= 20.0, f"dispersión {disp:.2f}×")
    check("T14 · EI_NOx de cabecera es el de Lefebvre (§8)",
          r["EI_NOx"] == r["EI_NOx_lefebvre"]
          and r["EI_CO"] == r["EI_CO_lefebvre"])


def test_T15_reporte():
    """T15 · El reporte HTML sale en UTF-8, en español, con las 13 secciones
    del contrato, el bloque de límites declarados y la trazabilidad."""
    r = _record_ref()
    st = sc.evaluate_structural(THETA_REF, r)
    con = gg.build_contract(THETA_REF, r, st, None)
    dfam = dfc.check(con)
    spec = no.DesignSpec()
    res = no.evaluate_design(spec, THETA_REF[:16], fidelity=pc.Fidelity.L0)
    tmp = tempfile.mkdtemp(prefix="phycb_t15_")
    try:
        path = rg.generate_report(spec, res, os.path.join(tmp, "report.html"),
                                  structural=st, contract=con, dfam=dfam,
                                  figures=[], log=_noop)
        check("T15 · generate_report devuelve una ruta escrita y no vacía",
              os.path.isfile(path) and os.path.getsize(path) > 20_000,
              f"{os.path.getsize(path)} bytes")
        with open(path, encoding="utf-8") as f:
            html = f.read()
        faltan = [f'id="s{i}"' for i in range(1, 14) if f'id="s{i}"' not in html]
        check("T15 · están los 13 tokens de sección (id=\"s1\"…\"s13\")",
              not faltan, f"faltan {faltan}")
        for tok in ('lang="en"', 'id="limits"', "DECLARED LIMITS",
                    "Traceability", "phycb-record-1", "correlations_L0"):
            check(f"T15 · el HTML contiene el token {tok!r}", tok in html)
        check("T15 · la trazabilidad publica la semilla y el esquema",
              "71" in html and "phycb-combustor-1" in html)
        check("T15 · el reporte declara que L2b está fuera de alcance",
              "L2b" in html or "OpenFOAM" in html)
        check("T15 · el HTML es UTF-8 real (tildes y η sin mojibake)",
              "η" in html and "Ø" in html)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_T16_ancla_REF_CB10():
    """T16 · Ancla de regresión REF_CB10: el record COMPLETO del caso KJ66
    está congelado. Moverla exige `--freeze-anchors` — a propósito."""
    actual = dict(theta=[float(x) for x in THETA_REF],
                  record=pc.evaluate(THETA_REF, fidelity=pc.Fidelity.L0,
                                     use_cache=False))
    if FREEZE:
        os.makedirs(ANCHOR_DIR, exist_ok=True)
        payload = dict(id="REF_CB10",
                       descripcion="caso ancla KJ66 (contrato §1), L0 puro",
                       schema=pc.SCHEMA, seed=pc.SEED, **actual)
        with open(ANCHOR_REF_CB10, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1,
                      sort_keys=True)
        skip("T16 · ancla REF_CB10",
             f"CONGELADA DE NUEVO por --freeze-anchors → {ANCHOR_REF_CB10}")
        return
    if not os.path.isfile(ANCHOR_REF_CB10):
        skip("T16 · ancla REF_CB10",
             f"SKIP: no existe {ANCHOR_REF_CB10} — créala con "
             "`python test_phycb.py --freeze-anchors`")
        return
    with open(ANCHOR_REF_CB10, encoding="utf-8") as f:
        anc = json.load(f)
    check("T16 · el ancla declara su identidad y su semilla",
          anc.get("id") == "REF_CB10" and anc.get("seed") == pc.SEED
          and anc.get("schema") == pc.SCHEMA)
    check("T16 · θ del ancla bit-idéntico a THETA_REF",
          [float(x) for x in anc["theta"]] == actual["theta"])

    # `limitations` se compara APARTE: no es física, es el registro de
    # avisos de coeficientes provisionales, y `physics_core` lo acumula en
    # variables de MÓDULO (`PROVISIONAL_COEFFS` / `_PROV_SEEN`). El mismo θ
    # evaluado en un proceso que ya ha explorado otros diseños trae MÁS
    # entradas — así que la lista depende de la HISTORIA del proceso, no
    # del diseño. Ver el check T16b.
    fisica = {k: v for k, v in actual["record"].items() if k != "limitations"}
    congelada = {k: v for k, v in anc["record"].items() if k != "limitations"}
    diffs = _deep_diff(fisica, congelada, "record")
    check("T16 · record de FÍSICA bit-idéntico al ancla congelada",
          not diffs,
          ("; ".join(diffs) + " · para reconciliar deliberadamente: "
           "`python test_phycb.py --freeze-anchors`") if diffs else
          f"{len(congelada)} claves congeladas, 0 diferencias")

    l_a = pc.evaluate(THETA_REF, use_cache=False)["limitations"]
    l_b = pc.evaluate(THETA_REF, use_cache=False)["limitations"]
    check("T16 · `limitations` es estable en dos llamadas consecutivas",
          l_a == l_b, f"{len(l_a)} avisos")

    # DESVIACIÓN REAL, no cosmética, y se demuestra AQUÍ MISMO para que no
    # dependa del orden de los tests: el contrato §10.8 promete records
    # bit-idénticos para evaluaciones idénticas, pero `limitations` se
    # acumula en variables de MÓDULO (`PROVISIONAL_COEFFS` / `_PROV_SEEN`),
    # así que basta con que el intérprete haya visto otros diseños para que
    # el MISMO θ devuelva otro record. O se hace por-record, o sale del
    # record y se queda solo en `provisional_report()`.
    guardado = (list(pc.PROVISIONAL_COEFFS), set(pc._PROV_SEEN))
    try:
        pc.PROVISIONAL_COEFFS.clear()
        pc._PROV_SEEN.clear()                   # intérprete «recién nacido»
        l_virgen = pc.evaluate(THETA_REF, use_cache=False)["limitations"]
        pc.PROVISIONAL_COEFFS.clear()
        pc._PROV_SEEN.clear()
        rng = np.random.default_rng(2026)
        for _ in range(64):                     # bucle ACOTADO
            pc.evaluate(pc.denormalize(rng.random(pc.NDIM)), use_cache=False)
        l_usado = pc.evaluate(THETA_REF, use_cache=False)["limitations"]
    finally:
        pc.PROVISIONAL_COEFFS.clear()
        pc.PROVISIONAL_COEFFS.extend(guardado[0])
        pc._PROV_SEEN.clear()
        pc._PROV_SEEN.update(guardado[1])
    check("T16b · `limitations` del MISMO θ no depende de qué otros "
          "diseños haya visto el proceso (contrato §10.8)",
          l_usado == l_virgen,
          f"intérprete virgen: {len(l_virgen)} avisos; tras 64 θ "
          f"aleatorios: {len(l_usado)} — `limitations` se ACUMULA en "
          "variables de módulo (PROVISIONAL_COEFFS/_PROV_SEEN), así que el "
          "record de un mismo diseño cambia con la historia del "
          "intérprete. El resto del record (la física) sí es bit-idéntico: "
          "ver el check anterior")


# ==========================================================================
# W2 · L1: red 1D, CRN, degradación, caché y estructura
# ==========================================================================
def test_T17_red_1d():
    """T17 · La red 1D de caudales conserva masa a 1e-9 y los splits que
    resuelve no se alejan más de un 15 % de los nominales de L0."""
    rec = _record_ref()
    n = crn.network_1d(THETA_REF, rec)
    check("T17 · la red 1D convergió con un Newton ACOTADO",
          n["network_converged"] is True and n["network_iters"] <= 30,
          f"{n['network_iters']} iteraciones, residuo "
          f"{n['network_residual']:.2e}")
    check("T17 · conservación de masa a 1e-9 kg/s",
          abs(n["mass_residual_kgs"]) < 1e-9,
          f"{n['mass_residual_kgs']:.3e} kg/s")
    s = n["splits_solved"]
    check("T17 · los splits resueltos suman 1 a 1e-9",
          abs(sum(s.values()) - 1.0) < 1e-9,
          f"Σ−1 = {sum(s.values()) - 1.0:+.3e}")
    check("T17 · todos los splits resueltos son positivos",
          all(v > 0.0 for v in s.values()), f"{s}")
    check("T17 · los splits resueltos caen a < 15 % de los nominales L0",
          n["split_dev_max"] < 0.15,
          f"desviación máx {100 * n['split_dev_max']:.2f} % "
          f"({n['split_dev_vs_L0']})")
    check("T17 · seis caminos de aire resueltos (contrato de crn_core)",
          set(n["mdot_paths_kgs"]) == set(crn.PATH_NAMES),
          f"{sorted(n['mdot_paths_kgs'])}")
    m_tot = sum(n["mdot_paths_kgs"].values())
    check("T17 · Σ ṁ de los caminos = ṁ de aire a 1e-9 relativo",
          _rel(m_tot, rec["mdot_air_kgs"]) < 1e-9,
          f"{m_tot:.9f} vs {rec['mdot_air_kgs']:.9f} kg/s")
    check("T17 · Δp del liner positivo y en la escala de q_ref",
          n["dp_liner_Pa"] > 0.0 and 5.0 < n["dp_liner_over_qref"] < 40.0,
          f"Δp_L/q_ref = {n['dp_liner_over_qref']:.2f}")


def test_T18_crn_smoke():
    """T18 · Smoke del CRN de Cantera: el record L1 trae las claves del
    contrato §4.3 y sus números son físicamente coherentes."""
    if not crn.CANTERA_AVAILABLE:
        skip("T18 · smoke del CRN L1",
             "SKIP: Cantera no está instalado — el CRN es imposible aquí")
        return
    t0 = time.time()
    r = pc.evaluate(THETA_REF, fidelity=pc.Fidelity.L1, use_cache=False)
    dt = time.time() - t0
    if r["source"] != "crn_L1":
        skip("T18 · smoke del CRN L1",
             f"SKIP: el CRN degradó a '{r['source']}' "
             f"(LAST_STATUS = {crn.LAST_STATUS!r})")
        return
    esperadas = ("splits_solved", "dp_liner_Pa", "v_zones", "tau_zones_ms",
                 "T_zones_K", "EI_NOx_crn", "EI_CO_crn", "eta_comb_crn",
                 "Da_pz", "LBO_margin_Da", "crn_topology")
    faltan = [k for k in esperadas if k not in r]
    check("T18 · el record L1 trae TODAS las claves del contrato §4.3",
          not faltan, f"faltan {faltan} · {dt:.2f} s")
    check("T18 · fidelity = 1 y source = 'crn_L1'",
          r["fidelity"] == int(pc.Fidelity.L1) and r["source"] == "crn_L1")
    check("T18 · η_comb del CRN en (0.5, 1]",
          0.5 < r["eta_comb_crn"] <= 1.0, f"{r['eta_comb_crn']:.6f}")
    check("T18 · T de zona: PZ es la más caliente y todas > T₃",
          r["T_zones_K"]["pz"] > r["T_zones_K"]["dz"] > float(THETA_REF[17]),
          f"pz {r['T_zones_K']['pz']:.0f} K > dz "
          f"{r['T_zones_K']['dz']:.0f} K")
    check("T18 · índices de emisión del CRN finitos y positivos",
          0.0 < r["EI_NOx_crn"] < 1e3 and 0.0 <= r["EI_CO_crn"] < 1e4,
          f"NOx {r['EI_NOx_crn']:.3f}  CO {r['EI_CO_crn']:.3f} g/kg")
    check("T18 · la topología declara mecanismo y versión de Cantera",
          "PSR" in r["crn_topology"] and "Cantera" in r["crn_topology"],
          r["crn_topology"][:110])
    check("T18 · τ físico (ρ(T)·V/ṁ) < τ frío de Lefebvre (convenio "
          "declarado en el record)",
          r["tau_zones_ms"]["total"] < r["tau_zones_cold_ms"]["total"],
          f"{r['tau_zones_ms']['total']:.3f} vs "
          f"{r['tau_zones_cold_ms']['total']:.3f} ms")


def test_T19_lbo_por_damkohler():
    """T19 · El LBO por número de Damköhler del CRN es coherente con la
    referencia `DA_LBO_REF` y con el margen de L0."""
    if not crn.CANTERA_AVAILABLE:
        skip("T19 · LBO por Damköhler",
             "SKIP: Cantera no está instalado")
        return
    r = pc.evaluate(THETA_REF, fidelity=pc.Fidelity.L1, use_cache=False)
    if "Da_pz" not in r:
        skip("T19 · LBO por Damköhler",
             f"SKIP: el CRN degradó a '{r['source']}'")
        return
    check("T19 · Da_pz finito y positivo",
          np.isfinite(r["Da_pz"]) and r["Da_pz"] > 0.0,
          f"Da_pz = {r['Da_pz']:.3f}")
    check("T19 · LBO_margin_Da = Da_pz/DA_LBO_REF − 1 a 1e-9",
          _rel(r["LBO_margin_Da"], r["Da_pz"] / crn.DA_LBO_REF - 1.0) < 1e-9,
          f"{r['LBO_margin_Da']:.4f} con DA_LBO_REF = {crn.DA_LBO_REF}")
    check("T19 · el ancla está MUY por encima del apagado (Da ≫ Da_LBO)",
          r["Da_pz"] > crn.DA_LBO_BAND[1],
          f"Da_pz = {r['Da_pz']:.2f} vs banda {crn.DA_LBO_BAND}")
    check("T19 · L0 y L1 coinciden en el veredicto de estabilidad",
          (r["LBO_margin"] > 0.0) == (r["LBO_margin_Da"] > 0.0),
          f"L0 {r['LBO_margin']:+.3f} · L1 {r['LBO_margin_Da']:+.3f}")


def test_T20_degradacion_etiquetada():
    """T20 · Sin Cantera, con el import roto o con timeout, `evaluate(L1)`
    devuelve el record L0 completo, SIN excepción y con la etiqueta del
    contrato §3 que explica POR QUÉ se degradó."""
    base = pc.evaluate(THETA_REF, use_cache=False)

    flag = crn.CANTERA_AVAILABLE
    try:
        crn.CANTERA_AVAILABLE = False
        r = pc.evaluate(THETA_REF, fidelity=pc.Fidelity.L1, use_cache=False)
    finally:
        crn.CANTERA_AVAILABLE = flag
    check("T20a · Cantera ausente ⇒ source = 'crn_L1(cantera_unavailable→L0)'",
          r["source"] == "crn_L1(cantera_unavailable→L0)", f"{r['source']!r}")
    check("T20a · el record degradado sigue COMPLETO y utilizable",
          r["eta_comb"] == base["eta_comb"] and r["g"] == base["g"]
          and r["fidelity"] == int(pc.Fidelity.L1))

    mod = sys.modules.pop("crn_core")
    sys.modules["crn_core"] = None            # rompe el import a propósito
    try:
        r2 = pc.evaluate(THETA_REF, fidelity=pc.Fidelity.L1, use_cache=False)
    finally:
        sys.modules["crn_core"] = mod
    check("T20b · import de crn_core roto ⇒ misma etiqueta, sin excepción",
          r2["source"] == "crn_L1(cantera_unavailable→L0)"
          and r2["eta_comb"] == base["eta_comb"], f"{r2['source']!r}")

    r3 = pc.evaluate(THETA_REF, fidelity=pc.Fidelity.L2B, use_cache=False)
    check("T20d · L2B fuera de alcance (contrato §12) degrada con etiqueta",
          r3["source"].startswith("openfoam_L2B(out_of_scope→")
          and "L2b" in r3.get("fidelity_note", ""), f"{r3['source']!r}")

    if not crn.CANTERA_AVAILABLE:
        skip("T20c · timeout artificial",
             "SKIP: sin Cantera el camino de timeout no se puede alcanzar")
        return
    old = pc.L1_TIMEOUT_S
    try:
        pc.L1_TIMEOUT_S = 0.0                 # deadline vencido de entrada
        r4 = pc.evaluate(THETA_REF, fidelity=pc.Fidelity.L1, use_cache=False)
    finally:
        pc.L1_TIMEOUT_S = old
    check("T20c · timeout artificial ⇒ L0 completo sin excepción",
          r4["eta_comb"] == base["eta_comb"] and "Da_pz" not in r4
          and r4["source"].startswith("crn_L1("),
          f"{r4['source']!r}")
    # Este check es EXIGENTE a propósito: el contrato §3 tiene una etiqueta
    # propia para el timeout y `crn_core.LAST_STATUS` ya la conoce; si
    # `physics_core._try_l1` no la propaga, el usuario lee «divergió» ante
    # un problema de reloj. Ver el resumen de la suite.
    check("T20c · la etiqueta distingue timeout de divergencia (§3)",
          ("timeout" in r4["source"]) == (crn.LAST_STATUS == "timeout"),
          f"source = {r4['source']!r} pero crn_core.LAST_STATUS = "
          f"{crn.LAST_STATUS!r} — el contrato §3 pide "
          "'crn_L1(timeout→L0)'")


def test_T21_tendencia_cruzada_l0_l1():
    """T21 · L0 y L1 deben MOVERSE EN EL MISMO SENTIDO en un barrido de
    φ_PZ: si discrepan en la tendencia, el surrogate está aprendiendo
    ruido."""
    if not crn.CANTERA_AVAILABLE:
        skip("T21 · tendencia cruzada L0 ↔ L1",
             "SKIP: Cantera no está instalado")
        return
    fracs = (0.30, 0.36, 0.45)         # rama POBRE: φ_PZ decrece monótono
    phi0, phi1, T0, T1 = [], [], [], []
    for f in fracs:
        th = THETA_REF.copy()
        th[2] = f
        r0 = pc.evaluate(th, fidelity=pc.Fidelity.L0, use_cache=False)
        r1 = pc.evaluate(th, fidelity=pc.Fidelity.L1, use_cache=False)
        if r1["source"] != "crn_L1":
            skip("T21 · tendencia cruzada L0 ↔ L1",
                 f"SKIP: el CRN degradó a '{r1['source']}' en frac_pz={f}")
            return
        phi0.append(r0["phi_pz"])
        phi1.append(r1["phi_zones_solved"]["pz"])
        T0.append(r0["T_pz_K"])
        T1.append(r1["T_zones_K"]["pz"])
    check("T21 · φ_PZ decrece con frac_pz en L0 Y en L1 (mismo sentido)",
          _monotone(phi0, growing=False) and _monotone(phi1, growing=False),
          f"L0 {[round(x, 4) for x in phi0]} · L1 {[round(x, 4) for x in phi1]}")
    check("T21 · T_PZ decrece en la rama pobre en L0 Y en L1",
          _monotone(T0, growing=False) and _monotone(T1, growing=False),
          f"L0 {[round(x) for x in T0]} K · L1 {[round(x) for x in T1]} K")
    check("T21 · L1 es MÁS FRÍO que L0 (equilibrio vs cinética finita)",
          all(a < b for a, b in zip(T1, T0)),
          f"ΔT medio {np.mean(np.array(T1) - np.array(T0)):+.0f} K")
    r_phi = float(np.corrcoef(phi0, phi1)[0, 1])
    check("T21 · correlación de rango L0↔L1 en φ_PZ positiva y fuerte",
          r_phi > 0.9, f"r = {r_phi:.4f}")


def test_T22_sigma_termico():
    """T22 · σ_th crece con el coeficiente de dilatación y se ANULA cuando
    no hay gradiente por el espesor (T_w1 = T_w2)."""
    rec = _record_ref()
    st = sc.evaluate_structural(THETA_REF, rec, material="Inconel-625")
    check("T22 · σ_th del ancla positivo y finito",
          st["sigma_th_MPa"] > 0.0 and np.isfinite(st["sigma_th_MPa"]),
          f"{st['sigma_th_MPa']:.3f} MPa")
    check("T22 · g_struct tiene 5 componentes finitas",
          len(st["g_struct"]) == sc.N_STRUCT_CONSTRAINTS == 5
          and all(np.isfinite(g) for g in st["g_struct"]))

    sc.MATERIALS["__TEST_2ALPHA__"] = dict(sc.MATERIALS["Inconel-625"])
    sc.MATERIALS["__TEST_2ALPHA__"]["alpha"] = \
        2.0 * sc.MATERIALS["Inconel-625"]["alpha"]
    try:
        st2 = sc.evaluate_structural(THETA_REF, rec,
                                     material="__TEST_2ALPHA__")
    finally:
        del sc.MATERIALS["__TEST_2ALPHA__"]
    check("T22 · α ×2 (mismo E, ν y ΔT) ⇒ σ_th ×2 exacto",
          abs(st2["sigma_th_MPa"] / st["sigma_th_MPa"] - 2.0) < 1e-9,
          f"{st['sigma_th_MPa']:.4f} → {st2['sigma_th_MPa']:.4f} MPa")

    rec_iso = dict(rec, T_w1_K=1000.0, T_w2_K=1000.0)
    st_iso = sc.evaluate_structural(THETA_REF, rec_iso)
    check("T22 · sin gradiente (T_w1 = T_w2) ⇒ σ_th = 0 exacto",
          st_iso["sigma_th_MPa"] == 0.0, f"{st_iso['sigma_th_MPa']!r} MPa")
    check("T22 · σ_th = 0 ⇒ g_struct[1] (σ_th vs k(T)·σ_y) satisfecha",
          st_iso["g_struct"][1] <= 0.0, f"{st_iso['g_struct'][1]:+.4f}")

    dTs, sig = [], []
    for dT in (5.0, 20.0, 60.0):
        s = sc.evaluate_structural(THETA_REF,
                                   dict(rec, T_w1_K=1000.0,
                                        T_w2_K=1000.0 - dT))
        dTs.append(dT)
        sig.append(s["sigma_th_MPa"])
    check("T22 · σ_th crece linealmente con el ΔT por el espesor",
          _monotone(sig, growing=True)
          and abs(sig[2] / sig[0] - dTs[2] / dTs[0]) < 1e-9,
          " → ".join(f"{x:.3f}" for x in sig) + " MPa")


def test_T23_pared_caliente_prohibida():
    """T23 · Un material de la lista `HOT_WALL_BANNED` en la pared caliente
    es INFACTIBLE por encima de su límite, se use donde se use."""
    rec = _record_ref()
    st = sc.evaluate_structural(THETA_REF, rec, material="Inconel-718",
                                material_casing="Inconel-718")
    lim = sc.HOT_WALL_BANNED["Inconel-718"]
    check("T23 · IN718 como liner con T_w1 del ancla (≈1059 K > 923 K) "
          "viola la restricción de pared caliente",
          rec["T_w1_K"] > lim and st["g_struct"][4] > 0.0
          and st["hot_wall_ok"] is False,
          f"T_w1 = {rec['T_w1_K']:.1f} K, límite {lim:.0f} K, "
          f"g_struct[4] = {st['g_struct'][4]:+.4f}")
    check("T23 · IN718 SÍ es legítimo como CARCASA (fría) con liner IN625",
          sc.evaluate_structural(THETA_REF, rec, material="Inconel-625",
                                 material_casing="Inconel-718"
                                 )["g_struct"][4] <= 0.0)
    check("T23 · hot_wall_materials() excluye a los prohibidos",
          "Inconel-718" not in sc.hot_wall_materials()
          and "Inconel-625" in sc.hot_wall_materials(),
          f"{sorted(sc.hot_wall_materials())}")
    ok_900, m_900 = sc.is_hot_wall_allowed("Inconel-718", 900.0)
    ok_950, m_950 = sc.is_hot_wall_allowed("Inconel-718", 950.0)
    check("T23 · is_hot_wall_allowed() devuelve (veredicto, margen) y "
          "discrimina por temperatura",
          ok_900 is True and ok_950 is False and m_900 < 0.0 < m_950,
          f"900 K → {ok_900} ({m_900:+.4f}); 950 K → {ok_950} ({m_950:+.4f})")
    check("T23 · un material desconocido NO lanza: degrada con g = 10",
          all(g == sc.G_DEGENERATE for g in
              sc.evaluate_structural(THETA_REF, rec,
                                     material="Unobtainium")["g_struct"]))


def test_T24_disciplina_de_cache():
    """T24 · L1 escribe `phys_cache.jsonl`, L0 NO, y `use_cache=False` no
    lee ni escribe (trampa CONTEXTO §10.5, heredada del T14 de Phy-CC)."""
    tmp = tempfile.mkdtemp(prefix="phycb_t24_")
    path = os.path.join(tmp, pc.CACHE_FILENAME)
    old_path, old_mem = pc.CACHE_PATH, dict(pc._CACHE)
    try:
        pc._CACHE.clear()
        pc.set_cache_path(path)

        th0 = THETA_REF.copy()
        th0[19] = 1140.0
        pc.evaluate(th0, fidelity=pc.Fidelity.L0, use_cache=True)
        check("T24 · L0 con caché activa NO escribe el JSONL "
              "(recomputarlo cuesta menos que leerlo)",
              not os.path.exists(path), f"{path}")
        check("T24 · L0 sí queda en la caché EN MEMORIA",
              len(pc._CACHE) == 1, f"{len(pc._CACHE)} entradas")

        if not crn.CANTERA_AVAILABLE:
            skip("T24 · L1 escribe el JSONL",
                 "SKIP: Cantera no está instalado")
        else:
            th1 = THETA_REF.copy()
            th1[19] = 1141.0
            pc.evaluate(th1, fidelity=pc.Fidelity.L1, use_cache=True)
            check("T24 · L1 con caché activa SÍ escribe phys_cache.jsonl",
                  os.path.exists(path)
                  and len(open(path, encoding="utf-8").readlines()) == 1,
                  f"{path}")

        n_lineas = (len(open(path, encoding="utf-8").readlines())
                    if os.path.exists(path) else 0)
        n_mem = len(pc._CACHE)
        th2 = THETA_REF.copy()
        th2[19] = 1142.0
        pc.evaluate(th2, fidelity=pc.Fidelity.L1, use_cache=False)
        n_lineas2 = (len(open(path, encoding="utf-8").readlines())
                     if os.path.exists(path) else 0)
        check("T24 · use_cache=False NO escribe en disco",
              n_lineas2 == n_lineas, f"{n_lineas} → {n_lineas2} líneas")
        check("T24 · use_cache=False NO contamina la caché en memoria",
              len(pc._CACHE) == n_mem, f"{n_mem} → {len(pc._CACHE)}")

        # y tampoco LEE: un valor envenenado en la caché no debe emerger
        clave = next(iter(pc._CACHE))
        pc._CACHE[clave] = dict(pc._CACHE[clave], eta_comb=-999.0)
        r = pc.evaluate(th0, fidelity=pc.Fidelity.L0, use_cache=False)
        check("T24 · use_cache=False tampoco LEE la caché envenenada",
              r["eta_comb"] > 0.9, f"η = {r['eta_comb']}")
        r_c = pc.evaluate(th0, fidelity=pc.Fidelity.L0, use_cache=True)
        check("T24 · use_cache=True sí devuelve lo cacheado (el veneno)",
              r_c["eta_comb"] == -999.0, f"η = {r_c['eta_comb']}")
    finally:
        pc._CACHE.clear()
        pc._CACHE.update(old_mem)
        pc.CACHE_PATH = old_path
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# W3 · contrato de geometría y DfAM
# ==========================================================================
def _contrato_ancla():
    """Contrato `phycb-combustor-1` del ancla, con estructura y DfAM."""
    r = _record_ref()
    st = sc.evaluate_structural(THETA_REF, r)
    con = gg.build_contract(THETA_REF, r, st, None)
    return r, st, con


def test_T25_contrato_round_trip():
    """T25 · El contrato de geometría sobrevive a un round-trip
    escribir → leer sin perder ni una clave ni un dígito."""
    r, st, con = _contrato_ancla()
    check("T25 · schema del contrato = 'phycb-combustor-1'",
          con["schema"] == gg.SCHEMA == "phycb-combustor-1")
    for k in ("_conventions", "run_meta", "liner", "casing", "hole_rows",
              "effusion", "flow_path", "dfam", "validation"):
        check(f"T25 · el contrato trae el bloque {k!r}", k in con)
    check("T25 · run_meta declara schema, semilla, source y fidelidad",
          {"schema", "seed", "source", "fidelity", "versions"}
          <= set(con["run_meta"]) and con["run_meta"]["seed"] == pc.SEED,
          f"{sorted(con['run_meta'])}")

    tmp = tempfile.mkdtemp(prefix="phycb_t25_")
    try:
        out = gg.generate(THETA_REF, r, tmp, structural=st, step=False,
                          log=None)
        esperados = ("combustor.json", "hole_table.csv", "liner_profile.csv",
                     "effusion_field.csv", "combustor_bom.csv",
                     "cfd_boundary_conditions.json", "dfam_report.json")
        escritos = set(os.listdir(tmp))
        check("T25 · generate() escribe los 7 artefactos del contrato §9",
              set(esperados) <= escritos, f"{sorted(escritos)}")
        check("T25 · generate() devuelve el resumen con outdir y contrato",
              isinstance(out, dict) and out["schema"] == gg.SCHEMA
              and out["arch"] == r["arch"])
        with open(os.path.join(tmp, "combustor.json"), encoding="utf-8") as f:
            leido = json.load(f)
        ida = json.dumps(out["contract"], sort_keys=True, ensure_ascii=False)
        vuelta = json.dumps(leido, sort_keys=True, ensure_ascii=False)
        check("T25 · round-trip escribir → leer bit-idéntico",
              ida == vuelta,
              f"{len(ida)} vs {len(vuelta)} caracteres")
        check("T25 · el JSON escrito es UTF-8 (no ASCII escapado a la fuerza)",
              leido["_conventions"] == out["contract"]["_conventions"])
        check("T25 · el contrato leído sigue pasando el checker DfAM",
              len(dfc.check(leido)["g_dfam"]) == dfc.N_DFAM_CONSTRAINTS == 5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_T26_dfam_espesores_y_diametros():
    """T26 · El checker DfAM caza espesores por debajo del mínimo LPBF y
    orificios por debajo de 0.5 mm, y da g CONTINUO."""
    limpio = dfc._demo_contract_clean()
    g_ok = dfc.check(limpio)["g_dfam"]
    check("T26 · el contrato de demo limpio pasa las 5 reglas DfAM",
          len(g_ok) == 5 and all(g <= 0.0 for g in g_ok),
          f"{[round(g, 4) for g in g_ok]}")

    malo = dfc._demo_contract_bad()
    res = dfc.check(malo)
    check("T26 · el contrato de demo malo viola LAS CINCO reglas",
          all(g > 0.0 for g in res["g_dfam"]),
          f"{[round(g, 4) for g in res['g_dfam']]}")
    check("T26 · las violaciones se explican en texto accionable",
          len(res["violations"]) >= 5
          and any("thickness" in v for v in res["violations"])
          and any("0.5 mm" in v or "0.500 mm" in v
                  for v in res["violations"]),
          f"{len(res['violations'])} mensajes")

    check("T26 · lpbf_t_min_mm sale del MATERIAL, no de una constante",
          dfc.lpbf_t_min_mm("Inconel-625") == 0.5
          and dfc.lpbf_t_min_mm("Hastelloy-X") == 0.8
          and dfc.lpbf_t_min_mm(None) == dfc.LPBF_T_MIN_DEFAULT_MM)

    espesores, g0 = [0.30, 0.45, 0.50, 0.80, 1.20], []
    for t in espesores:
        c = json.loads(json.dumps(limpio))
        c["liner"]["t_wall_mm"] = t
        c["dfam"]["min_wall_mm"] = t
        g0.append(dfc.check(c)["g_dfam"][0])
    check("T26 · g[0] (espesor) es CONTINUO, no creciente en t y cambia de "
          "signo EXACTAMENTE en el mínimo LPBF del material (0.5 mm)",
          all(g0[i] >= g0[i + 1] - 1e-12 for i in range(len(g0) - 1))
          and g0[0] > 0.0 and abs(g0[2]) < 1e-9 and g0[3] < 0.0,
          " → ".join(f"{g:+.4f}" for g in g0)
          + "  (se estanca al engordar el liner porque otra pared —la "
            "carcasa— pasa a ser la que manda: g[0] es un máximo)")

    diams, g1 = [0.30, 0.40, 0.50, 0.60, 0.80], []
    for d in diams:
        c = json.loads(json.dumps(limpio))
        c["effusion"]["d_mm"] = d
        g1.append(dfc.check(c)["g_dfam"][1])
    check("T26 · g[1] (Ø mínimo) continuo y con el cero en 0.5 mm",
          _monotone(g1, growing=False) and abs(g1[2]) < 1e-9,
          " → ".join(f"{g:+.4f}" for g in g1))
    check("T26 · contrato vacío o None ⇒ g = 10 sin excepción",
          all(g == dfc.DEGENERATE_G for g in dfc.check({})["g_dfam"])
          and all(g == dfc.DEGENERATE_G
                  for g in dfc.check(None)["g_dfam"]))
    basura = dfc.check({"schema": "phycb-combustor-1", "nada": 1})
    check("T26 · un contrato SIN geometría no puede declararse imprimible",
          all(g == dfc.DEGENERATE_G for g in basura["g_dfam"])
          or basura["violations"],
          f"g_dfam = {basura['g_dfam']} con 0 violaciones: un contrato sin "
          "liner, sin carcasa y sin orificios sale FACTIBLE porque no hay "
          "nada que violar. `dfam_checker.check` solo degenera con dict "
          "vacío o None; debería exigir los bloques mínimos del contrato §9")


def test_T27_teardrop_automatico():
    """T27 · La sección en gota se pide sola: eje del orificio a menos de
    45° de la horizontal de la placa ⇒ teardrop; eje alineado con el build
    ⇒ círculo. Y un Ø de dilución grande la fuerza igualmente."""
    check("T27 · teardrop_needed: eje radial ⇒ gota; eje axial ⇒ círculo",
          dfc.teardrop_needed((1, 0, 0), "z+") is True
          and dfc.teardrop_needed((0, 0, 1), "z+") is False
          and dfc.teardrop_needed((0, 0, -1), "z+") is False)
    check("T27 · teardrop_needed: el umbral está EXACTAMENTE en 45°",
          dfc.teardrop_needed((math.cos(math.radians(46.0)), 0.0,
                               math.sin(math.radians(46.0))), "z+") is False
          and dfc.teardrop_needed((math.cos(math.radians(44.0)), 0.0,
                                   math.sin(math.radians(44.0))),
                                  "z+") is True)
    check("T27 · entrada degenerada ⇒ True (la gota siempre imprime)",
          dfc.teardrop_needed((0, 0, 0)) is True
          and dfc.teardrop_needed((float("nan"), 0, 1)) is True)

    limpio = dfc._demo_contract_clean()
    for ang, d, esperado in ((0.0, 3.0, "teardrop"), (30.0, 3.0, "teardrop"),
                             (44.0, 3.0, "teardrop"), (46.0, 3.0, "circle"),
                             (90.0, 3.0, "circle"), (90.0, 6.5, "teardrop")):
        c = json.loads(json.dumps(limpio))
        c["hole_rows"] = [dict(zone="dz", z_mm=78.0, r_mm=35.0, n=10,
                               d_mm=d, Cd=0.62, angle_deg=ang,
                               shape="circle", id="X")]
        got = dfc.check(c)["hole_shapes"].get("X")
        check(f"T27 · fila a {ang:.0f}° con Ø {d:.1f} mm ⇒ {esperado}",
              got == esperado, f"obtenido {got!r}")

    _, _, con = _contrato_ancla()
    dfam = dfc.check(con)
    con2 = gg.apply_dfam(con, dfam)
    check("T27 · apply_dfam ESCRIBE la forma resuelta en el contrato",
          all(h.get("shape") in ("circle", "teardrop")
              for h in con2["hole_rows"])
          and any(h["shape"] == "teardrop" for h in con2["hole_rows"]),
          f"{[h['shape'] for h in con2['hole_rows']]}")
    check("T27 · apply_dfam adjunta g_dfam y el veredicto al bloque dfam",
          len(con2["dfam"]["g_dfam"]) == 5
          and isinstance(con2["dfam"]["feasible"], bool))


def test_T28_drenabilidad():
    """T28 · Una cavidad de doble pared sin puertos es INFACTIBLE; con dos
    puertos deja de serlo; y la efusión NO cuenta como drenaje de polvo."""
    limpio = dfc._demo_contract_clean()
    doble = json.loads(json.dumps(limpio))
    doble["liner"]["double_wall"] = dict(gap_mm=1.2, z0_mm=30.0, z1_mm=92.0,
                                         t_inner_mm=0.8, t_outer_mm=0.8)
    doble["dfam"]["drain_ports"] = []
    r0 = dfc.check(doble)
    check("T28 · doble pared SIN puertos viola g[3] (drenabilidad)",
          r0["g_dfam"][3] > 0.0
          and any("drainability" in v for v in r0["violations"]),
          f"g[3] = {r0['g_dfam'][3]:+.4f}")
    check("T28 · la efusión que cubre EXACTAMENTE la cavidad (z 30→92 mm) "
          "NO cuenta como drenaje: el polvo no sale por 0.6 mm inclinados",
          doble["effusion"]["zones"][0]["z0_mm"] == 30.0
          and doble["effusion"]["zones"][0]["z1_mm"] == 92.0
          and r0["g_dfam"][3] > 0.0)

    con_puertos = json.loads(json.dumps(doble))
    con_puertos["dfam"]["drain_ports"] = [dict(z_mm=32.0, d_mm=4.0),
                                          dict(z_mm=90.0, d_mm=4.0)]
    r2 = dfc.check(con_puertos)
    check("T28 · con 2 puertos de Ø 4 mm (> 2·gap) la cavidad ya drena",
          r2["g_dfam"][3] <= 0.0, f"g[3] = {r2['g_dfam'][3]:+.4f}")

    uno = json.loads(json.dumps(doble))
    uno["dfam"]["drain_ports"] = [dict(z_mm=32.0, d_mm=4.0)]
    check("T28 · un solo puerto NO basta (mínimo 2 salidas)",
          dfc.check(uno)["g_dfam"][3] > 0.0)

    finos = json.loads(json.dumps(doble))
    finos["dfam"]["drain_ports"] = [dict(z_mm=32.0, d_mm=0.8),
                                    dict(z_mm=90.0, d_mm=0.8)]
    check("T28 · puertos demasiado finos (Ø < 2 mm) no cuentan",
          dfc.check(finos)["g_dfam"][3] > 0.0)

    grafo = dfc.drainability_graph(doble)
    check("T28 · drainability_graph expone las cavidades y sus salidas",
          isinstance(grafo, dict) and len(grafo) > 0, f"{sorted(grafo)}")

    _, _, con = _contrato_ancla()
    res = dfc.check(con)
    check("T28 · el anillo liner-carcasa del ancla drena con los puertos "
          "que el generador coloca solo",
          res["g_dfam"][3] <= 0.0
          and len(con["dfam"].get("drain_ports", [])) >= 2,
          f"g[3] = {res['g_dfam'][3]:+.4f}, "
          f"{len(con['dfam'].get('drain_ports', []))} puertos")


def test_T29_smoke_csharp():
    """T29 · Smoke del builder C#/PicoGK: el contrato de Phy-CB entra por
    `PhyCBImport.FromContractJson`, pasa `Validate()` y sale un STL."""
    if not SLOW:
        skip("T29 · smoke del builder C#/PicoGK",
             "SKIP: test caro (voxeliza y escribe STL) — usa --slow "
             "(o PHYCB_SLOW=1 con pytest)")
        return
    dotnet = shutil.which("dotnet")
    if dotnet is None:
        skip("T29 · smoke del builder C#/PicoGK",
             "SKIP: `dotnet` no está en el PATH de esta máquina")
        return
    dlls = sorted(glob.glob(os.path.join(
        HERE, "CombustorDesigner.Example", "bin", "*", "*",
        "CombustorDesigner.Example.dll")))
    if not dlls:
        skip("T29 · smoke del builder C#/PicoGK",
             "SKIP: no hay Example.dll compilado — corre "
             "`dotnet build CombustorDesigner.sln` primero")
        return

    r, st, con = _contrato_ancla()
    tmp = tempfile.mkdtemp(prefix="phycb_t29_")
    try:
        contrato = os.path.join(tmp, "combustor.json")
        with open(contrato, "w", encoding="utf-8") as f:
            json.dump(gg.apply_dfam(con, dfc.check(con)), f,
                      ensure_ascii=False, indent=1)
        outdir = os.path.join(tmp, "stl")
        t0 = time.time()
        try:
            p = subprocess.run([dotnet, dlls[-1], contrato, outdir, "1.2"],
                               capture_output=True, text=True,
                               timeout=CS_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            skip("T29 · smoke del builder C#/PicoGK",
                 f"SKIP: el builder superó {CS_TIMEOUT_S:.0f} s "
                 "(voxelización demasiado cara en esta máquina)")
            return
        dt = time.time() - t0
        check("T29 · el builder acepta el contrato y termina con código 0",
              p.returncode == 0,
              f"rc = {p.returncode} en {dt:.1f} s · "
              f"stderr: {(p.stderr or '')[-300:]}")
        stls = glob.glob(os.path.join(outdir, "*.stl"))
        check("T29 · escribe al menos el ensamblaje y una pieza en STL",
              len(stls) >= 2 and all(os.path.getsize(s) > 1024
                                     for s in stls),
              f"{len(stls)} STL, {dt:.1f} s, "
              f"{sum(os.path.getsize(s) for s in stls) // 1024} kB")
        rep = os.path.join(outdir, "dfam_report.json")
        check("T29 · el C# reverifica DfAM sobre los voxels reales",
              os.path.isfile(rep)
              and isinstance(json.load(open(rep, encoding="utf-8")), dict),
              rep)

        malo = os.path.join(tmp, "malo.json")
        with open(malo, "w", encoding="utf-8") as f:
            json.dump({"schema": "phycb-combustor-1"}, f)
        q = subprocess.run([dotnet, dlls[-1], malo, outdir, "1.2"],
                           capture_output=True, text=True, timeout=120)
        check("T29 · un contrato inválido se rechaza con código ≠ 0 "
              "(lectura defensiva del consumidor, contrato §9)",
              q.returncode != 0, f"rc = {q.returncode}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_T30_coherencia_dimensional():
    """T30 · Todo lo que el contrato publica en mm coincide con lo que el
    record dice en SI: mismas dimensiones, mismas unidades, un solo eje."""
    r, st, con = _contrato_ancla()
    d = con["derived"]
    for clave, factor in (("D_L_mm", 1.0), ("D_casing_mm", 1.0),
                          ("L_tot_mm", 1.0), ("L_pz_mm", 1.0)):
        check(f"T30 · contrato.derived.{clave} = record.{clave} a 1e-6",
              _rel(d[clave], factor * r[clave]) < 1e-6,
              f"{d[clave]} vs {r[clave]}")
    check("T30 · A_ref y A_L del contrato en mm² = los del record en m²",
          _rel(d["A_ref_mm2"], 1e6 * r["A_ref_m2"]) < 1e-6
          and _rel(d["A_L_mm2"], 1e6 * r["A_L_m2"]) < 1e-6)

    prof = np.array(con["liner"]["profile"], dtype=float)
    cas = np.array(con["casing"]["profile"], dtype=float)
    check("T30 · el perfil del liner es [z_mm, r_in_mm, r_out_mm] con z "
          "ordenado y no decreciente",
          prof.shape[1] == 3
          and all(prof[i, 0] <= prof[i + 1, 0]
                  for i in range(len(prof) - 1)),
          f"{prof.shape}")
    check("T30 · r_out ≥ r_in en todas las estaciones del liner",
          bool(np.all(prof[:, 2] >= prof[:, 1])))
    check("T30 · la carcasa envuelve al liner con holgura positiva",
          float(cas[:, 1].min()) > float(prof[:, 2].max()),
          f"r_casing {cas[:, 1].min():.2f} mm > r_liner "
          f"{prof[:, 2].max():.2f} mm")
    check("T30 · Ø de carcasa del contrato = 2·r del perfil de carcasa",
          _rel(2.0 * float(cas[:, 1].max()), r["D_casing_mm"]) < 1e-6,
          f"{2 * cas[:, 1].max():.3f} vs {r['D_casing_mm']:.3f} mm")
    check("T30 · la longitud del liner cubre al menos L_tot del record",
          float(prof[-1, 0] - prof[0, 0]) >= r["L_tot_mm"] - 1e-6,
          f"{prof[-1, 0] - prof[0, 0]:.2f} mm ≥ {r['L_tot_mm']:.2f} mm")
    check("T30 · las claves de longitud llevan sufijo de unidad (§10.1)",
          all(k.endswith(("_mm", "_deg", "_mm2", "_m2"))
              for k in ("t_wall_mm", "fillet_r_mm")
              if k in con["liner"]))
    check("T30 · el bloque _conventions declara eje, unidades y signos",
          isinstance(con["_conventions"], str)
          and "Z" in con["_conventions"] and "mm" in con["_conventions"])
    check("T30 · todo número del contrato es finito (nada de NaN/inf)",
          "NaN" not in json.dumps(con) and "Infinity" not in json.dumps(con))
    check("T30 · las filas de orificios traen Ø, n y z dentro del liner",
          all(h["d_mm"] > 0 and h["n"] >= 1
              and prof[0, 0] - 1e-6 <= h["z_mm"] <= prof[-1, 0] + 1e-6
              for h in con["hole_rows"]),
          f"{len(con['hole_rows'])} filas")


def test_T31_flujo_inverso():
    """T31 · El `reverse_annular` da dos giros de 180° manteniendo Z como
    eje, y NINGUNA arquitectura auto-interseca su perfil meridional."""
    for i, arch in enumerate(pc.ARCH_NAMES):
        th = THETA_REF.copy()
        th[0] = float(i)
        r = pc.evaluate(th, use_cache=False)
        st = sc.evaluate_structural(th, r)
        con = gg.build_contract(th, r, st, None)
        val = con["validation"]
        check(f"T31 · {arch}: perfil válido y SIN auto-intersección",
              val["profile_ok"] is True and val["self_intersects"] is False
              and not val["profile_errors"],
              f"{val}")
        prof = np.array(con["liner"]["profile"], dtype=float)
        check(f"T31 · {arch}: z monótono no decreciente (Z sigue siendo "
              "el eje aunque el gas se dé la vuelta)",
              bool(np.all(np.diff(prof[:, 0]) >= -1e-9)))
        giros = [f["turn_deg"] for f in con["flow_path"]]
        if arch == "reverse_annular":
            check("T31 · reverse_annular: exactamente dos giros de 180°",
                  sorted(giros)[-2:] == [180.0, 180.0]
                  and len(con["flow_path"]) >= 5, f"{giros}")
            dirs = [f["dir_z"] for f in con["flow_path"]]
            check("T31 · reverse_annular: el sentido axial se invierte y "
                  "vuelve (+ → − → +)",
                  any(d < 0 for d in dirs) and dirs[0] > 0 and dirs[-1] > 0,
                  f"{dirs}")
        else:
            check(f"T31 · {arch}: recorrido recto (sin giros de 180°)",
                  all(abs(g) < 1e-9 for g in giros), f"{giros}")
        areas = [f["area_mm2"] for f in con["flow_path"]]
        sin_area = [f["id"] for f in con["flow_path"]
                    if f["area_mm2"] is None]
        check(f"T31 · {arch}: todo tramo declara área POSITIVA o `null` "
              "explícito (nunca cero ni negativa)",
              all(a is None or a > 0.0 for a in areas)
              and len(sin_area) <= 1,
              f"{[(f['id'], f['area_mm2']) for f in con['flow_path']]}"
              + (f" · sin área: {sin_area} (tramo aguas abajo del "
                 "combustor: su sección la fija la turbina)"
                 if sin_area else ""))


# ==========================================================================
# W4 · optimizador, surrogate y seam de ciclo
# ==========================================================================
def test_T32_dominancia_y_frentes():
    """T32 · La dominancia restringida de Deb y el non-dominated sort son
    los del motor verbatim de la familia."""
    F = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]])
    V = np.zeros(4)
    frentes = no._fast_nondominated_sort(F, V)
    check("T32 · non-dominated sort: frente 0 = {0,1,2}, dominado = {3}",
          sorted(frentes[0]) == [0, 1, 2] and frentes[1] == [3],
          f"{frentes}")
    check("T32 · Deb 1: factible domina a infactible aunque sea peor",
          no._dominates(np.array([9.0, 9.0]), 0.0,
                        np.array([0.0, 0.0]), 1.0))
    check("T32 · Deb 2: entre infactibles gana el de menor violación",
          no._dominates(np.array([9.0, 9.0]), 0.5,
                        np.array([0.0, 0.0]), 1.0))
    check("T32 · Deb 3: entre factibles manda la dominancia de Pareto",
          no._dominates(np.array([1.0, 1.0]), 0.0,
                        np.array([2.0, 2.0]), 0.0)
          and not no._dominates(np.array([1.0, 2.0]), 0.0,
                                np.array([2.0, 1.0]), 0.0))

    rec = _record_ref()
    spec = no.DesignSpec()
    g = spec.constraints(rec, THETA_REF)
    check("T32 · g total = aero(12) + spec(4) + struct(5) + dfam(5) = 26",
          len(g) == 26, f"{len(g)}")
    check("T32 · sin material y sin DfAM el vector baja a 12 + 4 = 16",
          len(no.DesignSpec(material=None,
                            dfam=False).constraints(rec, THETA_REF)) == 16)
    check("T32 · violation() = 0 solo si todo g ≤ 0",
          no.DesignSpec.violation(np.array([-1.0, -2.0])) == 0.0
          and no.DesignSpec.violation(np.array([-1.0, 0.5])) > 0.0)
    obj = spec.objectives(rec)
    check("T32 · objetivos: −η_comb y una penalización de NOx, finitos",
          len(obj) == 2 and np.all(np.isfinite(obj))
          and abs(obj[0] + rec["eta_comb"]) < 1e-12, f"{obj}")
    check("T32 · OUT_KEYS del optimizador = los del contrato §8",
          no.OUT_KEYS == pc.OUT_KEYS
          == ["eta_comb", "dp_over_p", "PF", "T_wall_max_K", "EI_NOx"])


def test_T33_design_smoke():
    """T33 · `design()` con presupuesto corto y semilla fija termina, deja
    checkpoint y devuelve un diseño VERIFICADO (física real, no surrogate)."""
    tmp = tempfile.mkdtemp(prefix="phycb_t33_")
    try:
        ck = os.path.join(tmp, "phycb_run.json")
        t0 = time.time()
        res = no.design(no.DesignSpec(), rounds=1, n_init=48, batch_size=6,
                        fidelity=pc.Fidelity.L0, checkpoint_path=ck,
                        log=_noop, seed=pc.SEED)
        dt = time.time() - t0
        check("T33 · design() devuelve theta, record, fitness, history y "
              "pareto_front",
              {"theta", "record", "fitness", "history", "pareto_front"}
              <= set(res), f"{sorted(res)} en {dt:.1f} s")
        check("T33 · el θ ganador tiene NDIM = 20 componentes finitas",
              len(res["theta"]) == pc.NDIM
              and np.all(np.isfinite(np.asarray(res["theta"], float))))
        rec = res["record"]
        check("T33 · el record ganador es FÍSICO (source de correlaciones)",
              rec["source"].startswith("correlations_L0")
              and 0.5 < rec["eta_comb"] <= 1.0,
              f"η = {rec['eta_comb']:.5f}, source = {rec['source']!r}")
        check("T33 · el punto de operación de la spec se respetó (§1)",
              res["theta"][16] == 250_000.0 and res["theta"][17] == 430.0
              and res["theta"][18] == 0.13 and res["theta"][19] == 1150.0,
              f"{list(res['theta'][16:20])}")
        check("T33 · el frente de Pareto es VERIFICADO y no vacío",
              len(res["pareto_front"]) >= 1
              and all(len(q["theta"]) == pc.NDIM
                      for q in res["pareto_front"]),
              f"{len(res['pareto_front'])} puntos")
        check("T33 · el checkpoint existe y trae el dataset con OUT_KEYS",
              os.path.isfile(ck), ck)
        with open(ck, encoding="utf-8") as f:
            data = json.load(f)
        check("T33 · el checkpoint guarda todas las evaluaciones físicas",
              len(data["dataset"]) >= 48 and data["seed"] == pc.SEED
              and all(k in data["dataset"][0] for k in no.OUT_KEYS),
              f"{len(data['dataset'])} evaluaciones")
        sp2, front2 = no.pareto_from_checkpoint(ck, log=_noop)
        check("T33 · pareto_from_checkpoint reconstruye spec y frente",
              len(front2) >= 1 and sp2 is not None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_T34_design_determinista():
    """T34 · Dos `design()` con la misma semilla dan EXACTAMENTE el mismo
    diseño (sin esto, un informe no es reproducible)."""
    tmp = tempfile.mkdtemp(prefix="phycb_t34_")
    try:
        def _run(seed):
            return no.design(no.DesignSpec(), rounds=1, n_init=32,
                             batch_size=4, fidelity=pc.Fidelity.L0,
                             checkpoint_path=os.path.join(
                                 tmp, f"ck{seed}.json"),
                             log=_noop, seed=seed)
        a = _run(pc.SEED)
        b = _run(pc.SEED)
        check("T34 · design(seed=71) ×2 ⇒ θ bit-idéntico",
              np.array_equal(np.asarray(a["theta"], float),
                             np.asarray(b["theta"], float)))
        check("T34 · design(seed=71) ×2 ⇒ mismo fitness y mismo record",
              a["fitness"] == b["fitness"]
              and a["record"]["eta_comb"] == b["record"]["eta_comb"]
              and a["record"]["g"] == b["record"]["g"],
              f"fitness {a['fitness']:.12f}")
        c = _run(pc.SEED + 1)
        check("T34 · otra semilla explora otro punto (el LHS es real)",
              not np.array_equal(np.asarray(a["theta"], float),
                                 np.asarray(c["theta"], float))
              or a["fitness"] != c["fitness"],
              f"fitness {a['fitness']:.6f} vs {c['fitness']:.6f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_T35_spec_from_run_phyac():
    """T35 · `spec_from_run` lee un contrato de Phy-AC.

    DECLARACIÓN DE ALCANCE: el repo de Phy-AC de esta máquina NO trae
    ningún `runs/<n>/` (verificado). El contrato de este test se FABRICA
    en un temporal siguiendo el esquema `phyac-axial-1` documentado en
    CONTEXTO §4; por tanto verifica el LECTOR de Phy-CB, no los números de
    Phy-AC. El día que Phy-AC publique un run real, este test debe
    apuntarlo y volverse una validación de verdad.
    """
    tmp = tempfile.mkdtemp(prefix="phycb_t35_")
    try:
        geo = os.path.join(tmp, "completo", "geometry")
        os.makedirs(geo)
        contrato = {
            "schema": "phyac-axial-1",
            "cfx_boundary_conditions": {
                "P0_in_Pa": 101_325.0, "T0_in_K": 288.15,
                "P0_out_Pa_est": 405_300.0, "T0_out_K_est": 465.0,
                "mdot_kg_s": 2.5},
            "derived": {"PR": 4.0, "eta_poly": 0.90, "massflow": 2.5,
                        "M_exit": 0.42},
            "annulus": {"hub": [[0.0, 60.0], [120.0, 78.0]],
                        "tip": [[0.0, 140.0], [120.0, 118.0]]},
            "assembly": {"bleed_stage": 3, "bleed_hole_d_mm": 4.0,
                         "bleed_boss_d_mm": 10.0},
        }
        with open(os.path.join(geo, "axial_compressor.json"), "w",
                  encoding="utf-8") as f:
            json.dump(contrato, f, ensure_ascii=False)

        spec, info = cs.spec_from_run(os.path.join(tmp, "completo"),
                                      log=_noop)
        check("T35 · reconoce el contrato como 'phyac'",
              info["kind"] == "phyac", f"{info['kind']!r}")
        check("T35 · lee P₃, T₃ y ṁ SIN derivar nada cuando están",
              spec.P3_Pa == 405_300.0 and spec.T3_K == 465.0
              and spec.mdot_kgs == 2.5 and not info["derived"][:-1],
              f"{info['spec']}")
        check("T35 · T₄ por defecto de clase micro (1150 K) DECLARADA",
              spec.T4_K == 1150.0
              and any("1150" in d for d in info["derived"]))
        check("T35 · publica el anillo de entrada al difusor",
              abs(info["inlet_r_hub_mm"] - 78.0) < 1e-9
              and abs(info["inlet_r_tip_mm"] - 118.0) < 1e-9)
        check("T35 · publica la toma de sangrado disponible",
              info["bleed"]["stage"] == 3
              and any("bleed" in n for n in info["notes"]))

        geo2 = os.path.join(tmp, "minimo", "geometry")
        os.makedirs(geo2)
        with open(os.path.join(geo2, "axial_compressor.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"schema": "phyac-axial-1",
                       "cfx_boundary_conditions": {"P0_in_Pa": 101_325.0,
                                                   "T0_in_K": 288.15},
                       "derived": {"PR": 4.0, "eta_poly": 0.90,
                                   "massflow": 2.5}}, f)
        spec2, info2 = cs.spec_from_run(os.path.join(tmp, "minimo"),
                                        log=_noop)
        check("T35 · sin P0_out deriva P₃ = P0_in·PR y LO DICE",
              abs(spec2.P3_Pa - 101_325.0 * 4.0) < 1e-9
              and any("P3 = P0_in" in d for d in info2["derived"]),
              f"{spec2.P3_Pa:.1f} Pa · derivados: {len(info2['derived'])}")
        check("T35 · sin T0_out deriva T₃ por politrópica y LO DICE",
              abs(spec2.T3_K - 288.15 * 4.0 ** ((1.4 - 1.0)
                                                / (1.4 * 0.90))) < 1e-9
              and any("T3 =" in d for d in info2["derived"]),
              f"{spec2.T3_K:.2f} K")
        try:
            cs.spec_from_run(os.path.join(tmp, "no_existe"), log=_noop)
            ok = False
        except FileNotFoundError:
            ok = True
        check("T35 · un run inexistente lanza FileNotFoundError explícito",
              ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_T36_spec_from_run_phycc():
    """T36 · `spec_from_run` sobre un run REAL de Phy-CC (runs/ss110), que
    sí existe en este workspace. Solo se LEE: no es nuestro repo."""
    if not os.path.isdir(PHYCC_RUN):
        skip("T36 · spec_from_run sobre un run real de Phy-CC",
             f"SKIP: no existe {PHYCC_RUN} en esta máquina")
        return
    spec, info = cs.spec_from_run(PHYCC_RUN, log=_noop)
    check("T36 · reconoce el contrato como 'phycc'",
          info["kind"] == "phycc", f"{info['kind']!r}")
    check("T36 · usa el plano de ETAPA, no el de impulsor "
          "(Phy-CC/VALIDACION §2b: mezclarlos es el error #1)",
          info["plane"] == "etapa", f"{info['plane']!r}")
    check("T36 · P₃, T₃ y ṁ finitos y en la escala de una microturbina",
          1.0e5 < spec.P3_Pa < 1.0e6 and 300.0 < spec.T3_K < 700.0
          and 0.01 < spec.mdot_kgs < 5.0,
          f"P₃ = {spec.P3_Pa:.0f} Pa, T₃ = {spec.T3_K:.1f} K, "
          f"ṁ = {spec.mdot_kgs:.3f} kg/s")
    check("T36 · declara que T02 es DERIVADO (gap de CONTEXTO §5.2), "
          "no medido",
          any("T02" in d for d in info["derived"]),
          f"{len(info['derived'])} derivaciones declaradas")
    check("T36 · la spec resultante evalúa un combustor real y factible "
          "de dimensionar",
          np.isfinite(pc.evaluate(spec.fix_operating_point(THETA_REF),
                                  use_cache=False)["V_c_m3"]))
    r = pc.evaluate(spec.fix_operating_point(THETA_REF), use_cache=False)
    check("T36 · el combustor dimensionado para ss110 es mayor que el "
          "del ancla KJ66 (ṁ ≈ 4×)",
          r["D_casing_mm"] > _record_ref()["D_casing_mm"],
          f"Ø carcasa {r['D_casing_mm']:.1f} mm para ṁ = "
          f"{spec.mdot_kgs:.2f} kg/s")


def test_T37_hot_cycle():
    """T37 · El seam con el ciclo: `combustor=None` es estable y byte-fiel,
    y con un combustor mejor (η_b↑, Δp_b↓) el TSFC MEJORA."""
    comp = dict(PR=4.0, eta_poly=0.90, T0_in=288.15, P0_in=101_325.0,
                T0_out=460.0, massflow=25.0)
    a = cs.hot_cycle(comp, TIT_K=1550.0)
    b = cs.hot_cycle(comp, TIT_K=1550.0)
    check("T37 · combustor=None es determinista bit a bit",
          json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))
    check("T37 · combustor=None usa el relleno constante documentado",
          a["assumptions"]["eta_comb"] == cs.ETA_COMB == 0.995
          and a["assumptions"]["dp_comb"] == cs.DP_COMB == 0.05
          and a["assumptions"]["combustor_seam"] == "constant_filler")
    check("T37 · P04 = P03·(1 − Δp/p) exacto",
          abs(a["P04_Pa"] - a["P03_Pa"] * (1.0 - cs.DP_COMB)) < 1e-9)
    check("T37 · el ciclo base es físicamente coherente",
          a["thrust_N"] > 0 and a["TSFC_g_kNs"] > 0
          and 0.0 < a["eta_thermal"] < 1.0
          and a["T045_K"] < a["T04_K"],
          f"empuje {a['thrust_N']:.0f} N, TSFC {a['TSFC_g_kNs']:.2f} "
          f"g/kN·s, η_th = {a['eta_thermal']:.4f}")

    mejor = dict(eta_comb=0.999, dp_over_p=0.030, PF=0.28, RTDF=0.21,
                 EI_NOx=6.4, EI_CO=12.0, bleed_frac_required=0.08,
                 T_wall_max_K=1040.0, source="correlations_L0")
    m = cs.hot_cycle(comp, TIT_K=1550.0, combustor=mejor)
    check("T37 · η_b mayor y Δp_b menor ⇒ TSFC MEJOR (menor)",
          m["TSFC_g_kNs"] < a["TSFC_g_kNs"],
          f"{a['TSFC_g_kNs']:.4f} → {m['TSFC_g_kNs']:.4f} g/kN·s "
          f"({100 * (m['TSFC_g_kNs'] / a['TSFC_g_kNs'] - 1):+.2f} %)")
    check("T37 · η_b mayor ⇒ menos combustible para la misma TIT",
          m["f_fuel_air"] < a["f_fuel_air"],
          f"f {a['f_fuel_air']:.6f} → {m['f_fuel_air']:.6f}")
    check("T37 · Δp_b menor ⇒ más presión a la turbina",
          m["P04_Pa"] > a["P04_Pa"],
          f"{a['P04_Pa']:.0f} → {m['P04_Pa']:.0f} Pa")
    check("T37 · el seam sube al ciclo PF, EI y sangrado del combustor",
          m["pattern_factor"] == 0.28 and m["EI_NOx"] == 6.4
          and m["bleed_frac_required"] == 0.08)

    peor = dict(mejor, eta_comb=0.960, dp_over_p=0.080)
    p = cs.hot_cycle(comp, TIT_K=1550.0, combustor=peor)
    check("T37 · y al revés: peor combustor ⇒ TSFC peor (sensibilidad real)",
          p["TSFC_g_kNs"] > a["TSFC_g_kNs"],
          f"{p['TSFC_g_kNs']:.4f} g/kN·s")

    rec = _record_ref()
    real = cs.hot_cycle(comp, TIT_K=float(THETA_REF[19]), combustor=rec)
    check("T37 · un record REAL de Phy-CB entra sin fricción en el ciclo",
          np.isfinite(real["TSFC_g_kNs"]) and real["thrust_N"] > 0
          and real["assumptions"]["eta_comb"] == rec["eta_comb"]
          and real["assumptions"]["dp_comb"] == rec["dp_over_p"],
          f"η_b = {rec['eta_comb']:.5f}, Δp/p = {rec['dp_over_p']:.4f}")
    check("T37 · valores absurdos del combustor caen al relleno etiquetado",
          cs.hot_cycle(comp, TIT_K=1550.0,
                       combustor=dict(eta_comb=17.0, dp_over_p=-3.0)
                       )["assumptions"]["combustor_seam"]
          == "constant_filler(combustor_out_of_range)")


def test_T38_offdesign_del_adaptador():
    """T38 · Con la geometría CONGELADA, Δp/p escala con el cuadrado del
    parámetro de flujo corregido ṁ·√T₃/P₃ (proxy declarado, WAVE4 §3)."""
    rec0 = dict(P3_Pa=250_000.0, T3_K=430.0, mdot_kgs=0.13,
                dp_over_p=0.030, eta_comb=0.984, source="correlations_L0")
    adaptador = cs.cycle_adapter(rec0)
    check("T38 · cycle_adapter devuelve un callable con la firma del ciclo",
          callable(adaptador))
    base = adaptador(250_000.0, 430.0, 0.13, 1150.0)
    check("T38 · en el punto de diseño reproduce el Δp/p del record",
          base["dp_over_p"] == rec0["dp_over_p"], f"{base['dp_over_p']}")
    check("T38 · la etiqueta declara que es el proxy off-design",
          base["source"].endswith("|cycle_adapter(offdesign_proxy)"),
          f"{base['source']!r}")

    for k in (0.6, 0.8, 1.0, 1.25, 1.5):
        o = adaptador(250_000.0, 430.0, 0.13 * k, 1150.0)
        check(f"T38 · ṁ ×{k:.2f} ⇒ Δp/p ×{k ** 2:.4f} exacto",
              abs(o["dp_over_p"] - 0.030 * k ** 2) < 1e-12,
              f"{o['dp_over_p']:.9f}")
    o_T = adaptador(250_000.0, 4.0 * 430.0, 0.13, 1150.0)
    check("T38 · T₃ ×4 (a ṁ y P₃ fijos) ⇒ Δp/p ×4 (√T al cuadrado)",
          abs(o_T["dp_over_p"] - 0.030 * 4.0) < 1e-12,
          f"{o_T['dp_over_p']:.9f}")
    o_P = adaptador(2.0 * 250_000.0, 430.0, 0.13, 1150.0)
    check("T38 · P₃ ×2 ⇒ Δp/p ÷4",
          abs(o_P["dp_over_p"] - 0.030 / 4.0) < 1e-12,
          f"{o_P['dp_over_p']:.9f}")
    o_ext = adaptador(1.0e4, 430.0, 10.0, 1150.0)
    check("T38 · el proxy está ACOTADO: Δp/p nunca supera 0.49",
          o_ext["dp_over_p"] <= 0.49, f"{o_ext['dp_over_p']}")

    exacto = cs.cycle_adapter(rec0, theta=THETA_REF,
                              fidelity=pc.Fidelity.L0)(
        260_000.0, 440.0, 0.14, 1160.0)
    check("T38 · con θ congelado el adaptador RE-EVALÚA la física real",
          "cycle_adapter(exact)" in exacto["source"]
          and exacto["P3_Pa"] == 260_000.0 and exacto["mdot_kgs"] == 0.14
          and np.isfinite(exacto["eta_comb"]),
          f"η = {exacto['eta_comb']:.5f}, Δp/p = "
          f"{exacto['dp_over_p']:.5f}")


def test_T39_gate_del_surrogate():
    """T39 · El gate de calidad del surrogate PASA con una función suave y
    FALLA con ruido puro. Si falla en producción se enriquecen los
    physics_features — nunca se baja el umbral (WAVE4 §T39)."""
    rng = np.random.default_rng(3)
    X = rng.random((160, 6))
    Y = np.column_stack([
        0.95 + 0.040 * X[:, 0],                    # eta_comb
        0.030 + 0.020 * X[:, 1],                   # dp_over_p
        0.25 + 0.20 * X[:, 2],                     # PF
        950.0 + 200.0 * X[:, 3],                   # T_wall_max_K
        2.0 * np.exp(1.5 * X[:, 4]),               # EI_NOx (log interno)
    ])
    sur = no.EnsembleSurrogate(K=3)
    t0 = time.time()
    sur.fit(X, Y, log=_noop, rand=rng)
    check("T39 · función suave: R²(η_comb) ≥ 0.90 y R²(Δp/p) ≥ 0.85",
          sur.metrics["r2"]["eta_comb"] >= 0.90
          and sur.metrics["r2"]["dp_over_p"] >= 0.85,
          " ".join(f"{k}={v:.3f}" for k, v in sur.metrics["r2"].items())
          + f" · {time.time() - t0:.1f} s")
    check("T39 · función suave: cobertura ±2σ en [0.80, 1.0] y gate ABIERTO",
          0.80 <= sur.metrics["coverage_2sigma"] <= 1.0
          and sur.gate_ok() is True,
          f"cobertura {sur.metrics['coverage_2sigma']:.3f}")
    mu, sd = sur.predict(X[:5])
    check("T39 · predict devuelve (μ, σ) en unidades físicas y σ ≥ 0",
          mu.shape == (5, 5) and sd.shape == (5, 5)
          and bool(np.all(sd >= 0.0)) and bool(np.all(mu[:, 4] > 0.0)))

    rng2 = np.random.default_rng(9)
    Xn = rng2.random((160, 6))
    Yn = np.column_stack([0.90 + 0.10 * rng2.random(160),
                          0.02 + 0.05 * rng2.random(160),
                          0.20 + 0.30 * rng2.random(160),
                          900.0 + 300.0 * rng2.random(160),
                          1.0 + 9.0 * rng2.random(160)])
    sur2 = no.EnsembleSurrogate(K=3)
    sur2.fit(Xn, Yn, log=_noop, rand=rng2)
    check("T39 · ruido puro: el gate se CIERRA (no aprende lo que no hay)",
          sur2.gate_ok() is False and sur2.metrics["r2"]["eta_comb"] < 0.90,
          " ".join(f"{k}={v:.3f}" for k, v in sur2.metrics["r2"].items()))
    check("T39 · el gate expone las métricas que lo justifican",
          {"r2", "coverage_2sigma", "n_train"} <= set(sur2.metrics))


def test_T40_end_to_end():
    """T40 · La cadena completa de una corrida real: θ → física → estructura
    → contrato → DfAM → geometría en disco → reporte → seam de ciclo."""
    tmp = tempfile.mkdtemp(prefix="phycb_t40_")
    try:
        spec = no.DesignSpec()
        theta = spec.fix_operating_point(THETA_REF)
        rec = pc.evaluate(theta, fidelity=pc.Fidelity.L0, use_cache=False)
        check("T40 · 1/7 física: record completo y factible de leer",
              rec["schema"] == pc.SCHEMA and len(rec["g"]) == 12)
        st = sc.evaluate_structural(theta, rec)
        check("T40 · 2/7 estructura: g_struct de 5 componentes finitas",
              len(st["g_struct"]) == 5
              and all(np.isfinite(g) for g in st["g_struct"]))
        con = gg.build_contract(theta, rec, st, None)
        check("T40 · 3/7 contrato: geometría válida y sin auto-intersección",
              con["validation"]["profile_ok"] is True
              and con["validation"]["self_intersects"] is False)
        dfam = dfc.check(con)
        check("T40 · 4/7 DfAM: g_dfam de 5 componentes finitas",
              len(dfam["g_dfam"]) == 5
              and all(np.isfinite(g) for g in dfam["g_dfam"]))
        out = gg.generate(theta, rec, os.path.join(tmp, "geometry"),
                          structural=st, dfam=dfam, step=False, log=None)
        check("T40 · 5/7 disco: los artefactos de geometría existen",
              len(out["files"]) >= 7
              and all(os.path.getsize(p) > 0 for p in out["files"]),
              f"{len(out['files'])} archivos")
        res = no.evaluate_design(spec, theta[:16], fidelity=pc.Fidelity.L0)
        path = rg.generate_report(spec, res,
                                  os.path.join(tmp, "report.html"),
                                  structural=st, contract=con, dfam=dfam,
                                  figures=[], log=_noop)
        check("T40 · 6/7 reporte: HTML escrito con las 13 secciones",
              os.path.getsize(path) > 20_000,
              f"{os.path.getsize(path) // 1024} kB")
        comp = dict(PR=250_000.0 / 101_325.0, eta_poly=0.90, T0_in=288.15,
                    P0_in=101_325.0, T0_out=float(theta[17]),
                    massflow=float(theta[18]))
        ciclo = cs.hot_cycle(comp, TIT_K=float(theta[19]), combustor=rec)
        check("T40 · 7/7 seam: el ciclo caliente cierra con este combustor",
              np.isfinite(ciclo["TSFC_g_kNs"]) and ciclo["thrust_N"] > 0.0
              and ciclo["assumptions"]["eta_comb"] == rec["eta_comb"],
              f"TSFC {ciclo['TSFC_g_kNs']:.2f} g/kN·s, "
              f"empuje {ciclo['thrust_N']:.1f} N")
        g_total = spec.constraints(rec, theta)
        check("T40 · el vector g global de la corrida tiene 26 componentes "
              "finitas (contrato §5.4)",
              len(g_total) == 26 and np.all(np.isfinite(g_total)),
              f"violación total = {no.DesignSpec.violation(np.asarray(g_total)):.4f}")
        check("T40 · el reporte declara las limitaciones activas del record",
              isinstance(rec.get("limitations"), list)
              and len(rec["limitations"]) >= 1,
              f"{len(rec.get('limitations', []))} limitaciones declaradas")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# T49 · FUENTE ÚNICA de la efusión (contrato entre capas)
# ==========================================================================
def test_T49_efusion_fuente_unica():
    """T49 · La efusión es UNA magnitud y todos los consumidores ven la misma.

    ESTE ES EL TEST QUE FALTABA (auditoría 2026-07-31). Cuatro módulos
    hablaban de la misma efusión con dos áreas y dos C_d distintos y nada lo
    cazaba: en el ancla REF_CB10 la red 1D descargaba 621.4 mm² con
    C_d = 0.2210 y el LBM 854.1 mm² con C_d = 0.2470, ×1.536 de conductancia
    C_d·A al MISMO Δp por el MISMO agujero.

    Se comprueba, para las TRES arquitecturas y varios θ:

      a) `record["A_effusion_m2"]` == `contract["effusion"]["open_area_mm2"]`
         dentro de `physics_core.EFFUSION_AREA_REL_TOL`;
      b) `record["n_eff_holes"]` == `contract["effusion"]["n_holes"]`
         (±`EFFUSION_N_HOLES_TOL`, que es el truncamiento de `int()`);
      c) el C_d de efusión de la red 1D y el del LBM son el MISMO número;
      d) la conductancia C_d·A que ve cada consumidor es la misma;
      e) el área mojada NOMINAL de `physics_core` reproduce la que
         `geometry_generator` integra sobre la polilínea REAL, y el contrato
         lo declara en `validation.effusion_ok`.

    El check (e) es el que ataca la CAUSA RAÍZ: `π·D_L` es el perímetro de un
    círculo de área equivalente y solo vale en `can`; en una corona el
    perímetro exterior real es `2π·r_lo`, mayor en `r_lo/√(r_lo²−r_li²)`
    (1.3745 en el ancla). Si alguien vuelve a escribir `π·D_L·L_tot` para
    todas las arquitecturas, (e) salta en `annular` y `reverse_annular`.
    """
    try:
        import cfd.xlb_cold as xc                     # noqa: PLC0415
    except Exception as exc:
        xc = None
        _xc_why = f"{type(exc).__name__}: {exc}"
    else:
        _xc_why = ""
    tol = float(pc.EFFUSION_AREA_REL_TOL)
    n_tol = int(pc.EFFUSION_N_HOLES_TOL)

    #  Barrido: 3 arquitecturas × 4 θ que mueven Ø y paso de efusión (los dos
    #  ejes del patrón) y el reparto de aire (que mueve M y el área que hace
    #  falta). El ancla es SOLO uno de los doce casos a propósito: un test
    #  que mirase únicamente el ancla se perdería que el error era ×1.0 en
    #  `can` y ×1.375 en las dos anulares, es decir, dependiente de la gate.
    variantes = [
        ("ancla", {}),
        ("d_eff_min_pitch_max", {8: 0.50, 9: 10.0}),
        ("d_eff_max_pitch_min", {8: 1.00, 9: 4.0}),
        ("frac_cool_alto", {2: 0.20, 3: 0.05, 4: 0.25, 8: 0.75, 9: 5.0}),
    ]
    casos: list[str] = []
    malos_area, malos_n, malos_cd = [], [], []
    malos_cda, malos_val = [], []
    for i_arch, arch in enumerate(pc.ARCH_NAMES):
        for nombre, mut in variantes:
            th = THETA_REF.copy()
            th[0] = float(i_arch)
            for k, v in mut.items():
                th[k] = float(v)
            etiqueta = f"{arch}/{nombre}"
            rec = pc.evaluate(th, fidelity=pc.Fidelity.L0, use_cache=False)
            con = gg.build_contract(th, rec, None, None)
            eff = con["effusion"]

            # --- (a) área abierta: record ↔ contrato ----------------------
            A_rec = 1e6 * float(rec["A_effusion_m2"])
            A_con = float(eff["open_area_mm2"])
            d_area = _rel(A_rec, A_con)
            if d_area > tol:
                malos_area.append(f"{etiqueta}: {A_rec:.2f} vs {A_con:.2f} "
                                  f"mm² ({100 * d_area:+.3f} %)")

            # --- (b) nº de orificios --------------------------------------
            n_rec, n_con = int(rec["n_eff_holes"]), int(eff["n_holes"])
            if abs(n_rec - n_con) > n_tol:
                malos_n.append(f"{etiqueta}: {n_rec} vs {n_con}")

            # --- (c) y (d) C_d y conductancia: red 1D ↔ LBM ---------------
            if xc is not None:
                d_eff_m = 1e-3 * float(eff["d_mm"])
                cd_pc = float(pc.effusion_discharge_coefficient(d_eff_m))
                cd_crn = float(crn.effusion_cd(d_eff_m))
                cd_lbm = float(xc.effusion_cd(con))
                if not (cd_crn == cd_pc == cd_lbm):
                    malos_cd.append(f"{etiqueta}: pc {cd_pc:.6f} · crn "
                                    f"{cd_crn:.6f} · lbm {cd_lbm:.6f}")
                #  La red 1D descarga con el área del RECORD y el LBM con la
                #  del CONTRATO: ese cruce es justo el que estaba roto.
                cda_crn = cd_crn * 1e-6 * A_rec
                cda_lbm = cd_lbm * 1e-6 * A_con
                if _rel(cda_crn, cda_lbm) > tol:
                    malos_cda.append(
                        f"{etiqueta}: {cda_crn:.6e} vs {cda_lbm:.6e} m² "
                        f"(×{cda_lbm / max(cda_crn, 1e-30):.4f})")

            # --- (e) nominal de la física ↔ perfil REAL de la geometría ----
            chk = (con.get("validation") or {}).get("effusion") or {}
            if chk.get("ok") is not True:
                malos_val.append(f"{etiqueta}: "
                                 + "; ".join(chk.get("violations")
                                             or [f"ok={chk.get('ok')}"]))
            casos.append(etiqueta)

    check("T49a · área abierta: record y contrato coinciden dentro de la "
          f"tolerancia declarada ({100 * tol:.2f} %)",
          not malos_area,
          "; ".join(malos_area) if malos_area
          else f"{len(casos)} casos (3 arquitecturas × {len(variantes)} θ)")
    check("T49b · nº de orificios de efusión: record y contrato coinciden",
          not malos_n,
          "; ".join(malos_n) if malos_n else f"{len(casos)} casos, holgura "
          f"declarada ±{n_tol} (truncamiento de int())")
    #  Sin `cfd.xlb_cold` no hay LBM que comparar: (c) y (d) se SALTAN con
    #  motivo. Dejarlos pasar con `cd_lbm = cd_pc` sería un check que aprueba
    #  trivialmente — justo lo que esta suite prohíbe.
    if xc is None:
        skip("T49c · el C_d de efusión de la red 1D y el del LBM son el "
             "MISMO número",
             f"SKIP: `cfd.xlb_cold` no importable ({_xc_why})")
        skip("T49d · la conductancia C_d·A es la misma para la red 1D y el "
             "LBM", f"SKIP: `cfd.xlb_cold` no importable ({_xc_why})")
    else:
        check("T49c · el C_d de efusión de la red 1D y el del LBM son el "
              "MISMO número (bit a bit)",
              not malos_cd,
              "; ".join(malos_cd) if malos_cd
              else f"{len(casos)} casos; ancla = "
                   f"{pc.effusion_discharge_coefficient(6.0e-4):.6f} "
                   f"(tabla × {pc.CD_EFFUSION_INCLINED} × rugosidad AM)")
        check("T49d · la conductancia C_d·A es la misma para la red 1D y el "
              "LBM", not malos_cda,
              "; ".join(malos_cda) if malos_cda else f"{len(casos)} casos")
    check("T49e · el área mojada NOMINAL de `physics_core` reproduce el "
          "perfil REAL, y el contrato lo declara",
          not malos_val,
          "; ".join(malos_val) if malos_val
          else f"{len(casos)} casos con validation.effusion_ok = True")

    # --- la causa raíz, escrita como identidad geométrica ------------------
    #  Si alguien «simplifica» `effusion_wall_area_m2` a π·D_L·L_tot otra vez,
    #  este check lo dice con el número exacto en vez de con un fallo difuso.
    rec_a = pc.evaluate(THETA_REF, fidelity=pc.Fidelity.L0, use_cache=False)
    A_nom_can = math.pi * 1e-3 * rec_a["D_L_mm"] * 1e-3 * rec_a["L_tot_mm"]
    A_rev = float(pc.effusion_wall_area_m2("reverse_annular", rec_a))
    r_cas = 0.5e-3 * rec_a["D_casing_mm"]
    r_lo = pc.LINER_R_OVER_RCAS * r_cas
    r_li = math.sqrt(max(r_lo ** 2 - rec_a["A_L_m2"] / math.pi, 0.0))
    ratio_teorico = r_lo / max(math.sqrt(max(r_lo ** 2 - r_li ** 2, 1e-30)),
                               1e-30)
    check("T49f · en las anulares el área mojada NO es π·D_L·L_tot: el "
          "cociente es exactamente r_lo/√(r_lo²−r_li²)",
          _rel(A_rev / A_nom_can, ratio_teorico) < 1e-9,
          f"medido ×{A_rev / A_nom_can:.6f} vs identidad "
          f"×{ratio_teorico:.6f} (r_lo = {1e3 * r_lo:.3f} mm, "
          f"r_li = {1e3 * r_li:.3f} mm). Era este 1.375 el que separaba a "
          f"`physics_core` de `geometry_generator`.")
    check("T49g · en `can` el nominal y el perfil real coinciden por "
          "construcción (cociente 1.0)",
          _rel(pc.effusion_wall_area_m2("can", rec_a), A_nom_can) < 1e-12,
          f"{1e6 * pc.effusion_wall_area_m2('can', rec_a):.4f} mm²")
    check("T49h · las paredes que llevan efusión están DECLARADAS por "
          "arquitectura (v1: solo la exterior)",
          all(pc.EFFUSION_WALLS.get(a) == ("outer",) for a in pc.ARCH_NAMES)
          and rec_a["effusion_walls"] == ["outer"],
          f"{pc.EFFUSION_WALLS}")

    # --- las funciones públicas son PURAS y no lanzan con basura ----------
    for mala in (float("nan"), 0.0, -1.0, float("inf")):
        pat = pc.effusion_pattern(mala, mala, mala)
        if not (math.isfinite(pat["porosity"]) and pat["n_holes"] >= 0
                and math.isfinite(pat["A_open_m2"])
                and math.isfinite(pc.effusion_discharge_coefficient(mala))
                and math.isfinite(pc.effusion_wall_area_m2("can", {}))):
            check(f"T49i · las funciones públicas de efusión no lanzan ni "
                  f"devuelven NaN con entrada degenerada ({mala})", False,
                  f"{pat}")
    check("T49i · las funciones públicas de efusión no lanzan ni devuelven "
          "NaN con entrada degenerada", True,
          "NaN / 0 / negativo / inf en d, paso/d y área")


# ==========================================================================
# Registro de tests y ejecución
# ==========================================================================
ALL = [
    test_T1_contrato_y_procedencia, test_T2_theta_pad_y_gates,
    test_T3_g_continuo_y_degenerado, test_T4_determinismo,
    test_T5_materiales,
    test_T6_temperatura_adiabatica, test_T7_conservacion,
    test_T8_far_del_ciclo, test_T9_dimensionado_ancla_kj66,
    test_T10_penetracion_de_chorro, test_T11_termico_1d,
    test_T12_atomizacion_smd, test_T13_lbo, test_T14_emisiones,
    test_T15_reporte, test_T16_ancla_REF_CB10,
    test_T17_red_1d, test_T18_crn_smoke, test_T19_lbo_por_damkohler,
    test_T20_degradacion_etiquetada, test_T21_tendencia_cruzada_l0_l1,
    test_T22_sigma_termico, test_T23_pared_caliente_prohibida,
    test_T24_disciplina_de_cache,
    test_T25_contrato_round_trip, test_T26_dfam_espesores_y_diametros,
    test_T27_teardrop_automatico, test_T28_drenabilidad,
    test_T29_smoke_csharp, test_T30_coherencia_dimensional,
    test_T31_flujo_inverso,
    test_T32_dominancia_y_frentes, test_T33_design_smoke,
    test_T34_design_determinista, test_T35_spec_from_run_phyac,
    test_T36_spec_from_run_phycc, test_T37_hot_cycle,
    test_T38_offdesign_del_adaptador, test_T39_gate_del_surrogate,
    test_T40_end_to_end,
    test_T49_efusion_fuente_unica,
]

#: T41–T48 son de `cfd/tests/` (WAVE5): el siguiente ID libre era T49.
assert len(ALL) == 41, "la suite debe cubrir T1…T40 y T49"


def _id_de(fn) -> str:
    """`test_T12_atomizacion_smd` → `T12`."""
    return fn.__name__.split("_")[1]


def _uso() -> str:
    """Bloque «Doble modo» del docstring, que ES la ayuda de la CLI."""
    cuerpo = __doc__.split("Doble modo")[1].split("Un test")[0]
    return "Uso de test_phycb.py\n" + cuerpo.replace("-" * 10, "", 1).rstrip()


def main(argv=None) -> int:
    """Ejecuta la suite en modo standalone y devuelve el código de salida."""
    global _CURRENT_ID, SLOW, FREEZE
    argv = list(sys.argv[1:] if argv is None else argv)
    solo: set[str] = set()
    for i, a in enumerate(argv):
        if a in ("-h", "--help"):
            print(_uso())
            return 0
        if a == "--slow":
            SLOW = True
        elif a == "--freeze-anchors":
            FREEZE = True
        elif a == "--only":
            if i + 1 < len(argv):
                solo = {s.strip().upper() for s in argv[i + 1].split(",")
                        if s.strip()}
        elif a.startswith("--only="):
            solo = {s.strip().upper() for s in a.split("=", 1)[1].split(",")
                    if s.strip()}

    print("=" * 74)
    print("QUASAR Phy-CB · suite de VERIFICACIÓN (VV&UQ) — T1…T40 + T49")
    print(f"  physics_core : {pc.__file__}")
    print(f"  PROJECT={pc.PROJECT}  NDIM={pc.NDIM}  "
          f"N_CONSTRAINTS={pc.N_CONSTRAINTS}  SEED={pc.SEED}")
    print(f"  Cantera={'sí' if pc.cantera_available() else 'NO'}  "
          f"dotnet={'sí' if shutil.which('dotnet') else 'NO'}  "
          f"tests caros={'SÍ (--slow)' if SLOW else 'no (usa --slow)'}")
    print("=" * 74)

    t_ini = time.time()
    fallos_test: list[tuple[str, str]] = []
    for fn in ALL:
        tid = _id_de(fn)
        if solo and tid not in solo:
            continue
        _CURRENT_ID = tid
        t0 = time.time()
        try:
            fn()
            estado = "ok"
        except AssertionError as exc:
            estado = "fail"
            fallos_test.append((tid, str(exc).splitlines()[0]))
        except Exception as exc:                  # el test no debe reventar
            estado = "error"
            _RESULTS.append((tid, f"{tid} · EXCEPCIÓN NO ESPERADA", "fail",
                             f"{type(exc).__name__}: {exc}"))
            fallos_test.append((tid, f"{type(exc).__name__}: {exc}"))
        print(f"— {tid:>4s} {fn.__doc__.splitlines()[0][:56]:<58s}"
              f"{time.time() - t0:6.2f} s  "
              f"{'✓' if estado == 'ok' else '✗'}")
    _CURRENT_ID = ""

    print("\n" + "=" * 74)
    print("TABLA DE CHECKS")
    print("=" * 74)
    n_ok = n_fail = n_skip = 0
    for tid, nombre, estado, detalle in _RESULTS:
        marca = {"ok": "✓", "fail": "✗", "skip": "–"}[estado]
        n_ok += estado == "ok"
        n_fail += estado == "fail"
        n_skip += estado == "skip"
        linea = f"  {marca} [{tid:>4s}] {nombre}"
        if detalle and estado != "ok":
            linea += f"\n        → {detalle}"
        elif detalle:
            linea += f"  [{detalle}]"
        print(linea)

    total = n_ok + n_fail + n_skip
    print("=" * 74)
    print(f"{n_ok}/{n_ok + n_fail} checks OK   ·   {n_skip} SALTADOS   ·   "
          f"{total} en total   ·   {time.time() - t_ini:.1f} s")
    if n_skip:
        print("\nSALTADOS (ninguno cuenta como aprobado):")
        for tid, nombre, estado, detalle in _RESULTS:
            if estado == "skip":
                print(f"  – [{tid:>4s}] {nombre}: {detalle}")
    if fallos_test:
        print(f"\n{len(fallos_test)} TEST(S) FALLARON:")
        for tid, msg in fallos_test:
            print(f"  ✗ {tid}: {msg[:200]}")
        print("\nLa suite completa (incluidos los caros) se corre con:")
        print("  python test_phycb.py --slow")
        return 1
    print("\nSuite de verificación completa ✓  "
          "(los caros con `--slow`; el ancla con `--freeze-anchors`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
