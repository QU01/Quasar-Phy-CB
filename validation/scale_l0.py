"""Fase 17.1 (QSTUDIO) · evidencia reproducible de Phy-CB a mayor escala.

Subcomandos (ejecutar desde el checkout ORIGINAL de Phy-CB):

    python -B -X utf8 validation/scale_l0.py acquire  --mass 8 --out-root DIR
    python -B -X utf8 validation/scale_l0.py acquire  --mass 5 --out-root DIR
    python -B -X utf8 validation/scale_l0.py archive-uat14 --campaign F --run-dir D
    python -B -X utf8 validation/scale_l0.py diagnose --mass 8 --out-root DIR
    python -B -X utf8 validation/scale_l0.py check-inlets
    python -B -X utf8 validation/scale_l0.py verify-handoff --wave 1 --summary F
    python -B -X utf8 validation/scale_l0.py accept   --mass 8 --out-root DIR
    python -B -X utf8 validation/scale_l0.py verify   --masses 8
    python -B -X utf8 validation/scale_l0.py verify-candidate --candidate F
    python -B -X utf8 validation/scale_l0.py verify-handoff --wave 2 --summary F
    python -B -X utf8 validation/scale_l0.py accept   --mass 5,25 --out-root DIR
    python -B -X utf8 validation/scale_l0.py cad --out-root DIR --evidence F
    python -B -X utf8 validation/scale_l0.py archive-cad --evidence F
    python -B -X utf8 validation/scale_l0.py report
    python -B -X utf8 validation/scale_l0.py verify-handoff --wave 3 --summary F

`cad` (17.1-03, D-386) compila UNA vez el generador C#/PicoGK, construye en
serie los dos contratos de 8 kg/s de la matriz y el smoke KJ66 (T29 en modo
explícito) y comprueba STL, reporte DfAM de vóxel, envolvente y topología
por copa. `archive-cad` incorpora ese bloque `cad_runs` y la recomendación
D-377 a la matriz y `report` genera `validation/SCALE-L0-RESULTS.md`.

`accept` (17.1-02) corre por la CLI normal de Phy-CB una búsqueda L0 por
arquitectura anular con la descarga congelada del caudal, determina el θ
seleccionado desde el checkpoint y el contrato exportado, lo verifica de
forma independiente (`verify_scale_candidate`) y guarda la celda en
`validation/scale_acceptance.json`. `verify` repite esa verificación sobre
el archivo sin lanzar ninguna búsqueda.

`acquire` lanza Phy-AC (checkout original fijado a `AC_PINNED_SHA`, árbol
limpio antes y después) con una receta L0 congelada, toma del contrato
`geometry/axial_compressor.json` las condiciones TOTALES de descarga
(`cfx_boundary_conditions.mdot_kg_s / P0_out_Pa_est / T0_out_K_est`) y
`derived.PR`, reevalúa el θ del candidato con el propio Phy-AC en L0 y
escribe la fila en `validation/scale_inlets.json` con los bytes del
contrato (lzma+base64, sin pérdida) y su SHA256. Si la fila ya existe con
la misma receta y SHA de Phy-AC, la verifica sin relanzar ni sobrescribir.

`diagnose` corre UNA búsqueda L0 corta de Phy-CB con la descarga congelada
(fase diagnóstica: NO cuenta para la matriz de aceptación), reevalúa el θ
devuelto y archiva el resumen en `validation/scale_basis.md` entre
marcadores. Un resultado infactible es un resultado; una ejecución sin
evidencia identificable falla.

Las comprobaciones de las filas (`check_row`, `check_inlets`) no necesitan
QSTUDIO ni el archivo histórico: todo lo necesario va dentro del fixture.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import json
import lzma
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

CB_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = CB_ROOT / "validation"
INLETS_PATH = VALIDATION_DIR / "scale_inlets.json"
BASIS_PATH = VALIDATION_DIR / "scale_basis.md"

AC_ROOT = Path(os.environ.get(
    "PHYCB_SCALE_AC_ROOT", "C:/Users/nickv/projects/Proyectos/Phy-AC/Phy-AC"))
AC_PINNED_SHA = "081b6d05d68a32b6bd8a39dcedaf6e7dc39e93c8"
AC_REPO_URL = "https://github.com/QU01/Quasar-Phy-AC.git"

def ac_python() -> str:
    """Intérprete de los subprocesos Phy-AC.

    Por defecto `sys.executable`. `PHYCB_SCALE_AC_PYTHON` lo sustituye
    cuando el intérprete que escribe en Phy-CB no termina limpio con la
    CLI de Phy-AC (observado 2026-11-10: el Python base con las
    site-packages del venv por PYTHONPATH completa la corrida y cae al
    salir con 0xC0000374). El intérprete usado queda en la procedencia."""
    return os.environ.get("PHYCB_SCALE_AC_PYTHON") or sys.executable


INLETS_SCHEMA = "phycb.scale_inlets/1"
DIAG_SCHEMA = "phycb.scale_diagnose/1"
MASS_KEYS = ("5", "8", "25")
PR_BAND = (3.0, 7.0)
CONTRACT_REL = "geometry/axial_compressor.json"

#: Receta común de admisión y topes de Phy-AC (RESEARCH §4, A4).
AC_COMMON = dict(t0=288.15, p0=101325.0, rpm_max=25000.0, utip_max=480.0,
                 rtip_max=400.0, nstages_max=8, sm_flow_min=0.15,
                 material="Ti-6Al-4V")
#: PR objetivo por caudal. D-389 (fundador, 2026-11-10): a 5 kg/s pasa de
#: 3,2 a 3,5 tras agotar dos veces la compuerta con 3,2 (recomendados PR
#: 2,70 y 2,40); criterio intacto: recomendado con PR observado en [3,7].
#: Los intentos con objetivo 3,2 son evidencia y no se reutilizan.
AC_TARGET_PR = {"5": 3.5, "8": 3.5}
AC_TARGET_POLICY = {"5": "D-389", "8": "plan-17.1-01"}
AC_SUPERSEDED_TARGETS = {"5": [3.2]}
#: Intentos predefinidos por caudal: el 2.º sólo si el 1.º falla.
AC_ATTEMPTS = (dict(seed=71, n_init=320, rounds=5, batch_size=14),
               dict(seed=137, n_init=480, rounds=7, batch_size=18))
#: Tope por intento. D-387 (fundador, 2026-11-10): el 1.º conserva 150 s y
#: el 2.º predefinido sube a 210 s; criterio, receta y objetivo intactos.
AC_TIMEOUTS_S = (150.0, 210.0)
AC_TIMEOUT_POLICY = "D-387"
AC_TIMEOUT_S = AC_TIMEOUTS_S[0]
#: Margen de bombeo mínimo por caudal. D-390 (fundador, 2026-11-10): SÓLO a
#: 5 kg/s `sm_flow_min` baja de 0,15 a 0,12 tras agotar la compuerta con
#: D-389 (0 de 51 diseños en banda factibles con todas las restricciones,
#: siempre por el margen de bombeo). 8 y 25 kg/s conservan 0,15. La
#: factibilidad se evalúa con TODAS las g del spec efectivo (spec.sm_flow_min),
#: no con la constante fija SM_FLOW_MIN de la CLI de Phy-AC, cuya etiqueta
#: «min 15%» es cosmética.
AC_SM_FLOW_MIN = {"5": 0.12}
AC_SM_FLOW_EXCEPTION = {"5": dict(
    policy="D-390", sm_flow_min=0.12, baseline_sm_flow_min=0.15,
    vetable=True,
    risk=("excepción vetable: margen de bombeo en gasto ~12,26 % (0,26 "
          "puntos sobre el umbral 0,12) frente a la práctica del 15-25 % que "
          "cita Phy-AC; el diseño usa 8 de 8 etapas del tope y es el único "
          "factible en banda PR [3,7] del sondeo de diagnóstico; se acepta "
          "porque Phy-CB necesita P03/T03 realistas y el compresor no es el "
          "objeto de la fase 17.1"))}


def ac_common(key: str) -> dict:
    """Receta común efectiva por caudal (AC_COMMON + excepción D-390)."""
    c = dict(AC_COMMON)
    if key in AC_SM_FLOW_MIN:
        c["sm_flow_min"] = AC_SM_FLOW_MIN[key]
    return c


def expected_sm_flow_min(key: str) -> float:
    return float(ac_common(key)["sm_flow_min"])

#: Receta del diagnóstico L0 de Phy-CB (Tarea 1 de 17.1-01).
DIAG_RECIPE = dict(arch="annular", fidelity="L0", t4=1200.0, fuel="jet_a",
                   material="Inconel-625", material_casing="Inconel-718",
                   dp_max=0.04, pf_max=0.40, dfam=True, seed=71, n_init=40,
                   rounds=1, batch_size=6)
DIAG_TIMEOUT_S = 60.0
DIAG_BEGIN = "<!-- scale-l0:diagnose-8:begin -->"
DIAG_END = "<!-- scale-l0:diagnose-8:end -->"

#: Valores congelados de la descarga histórica UAT 14 (25 kg/s).
UAT14 = dict(run_id="1f9e4fa586c2", mdot_kg_s=25.0,
             P03_Pa=322771.4057302644, T03_K=420.15317853003546,
             PR=3.185506101458321, seed=2118878624,
             contract_sha256=("cff8c2f092857d6888ef2a911d8ac3a7"
                              "107eb50e83624903a4ec2ab7168526ee"),
             manifest_sha256=("2eb49eaaf0dc7823e86d8b6aa32ed00e"
                              "e88debece05439437be99240db05ee8f"),
             branch="master", commit=AC_PINNED_SHA)


class ScaleError(RuntimeError):
    """Evidencia inválida o incompleta (la tarea falla)."""


# --------------------------------------------------------------- utilidades
def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def encode_bytes(data: bytes) -> str:
    return base64.b64encode(lzma.compress(data, preset=9)).decode("ascii")


def decode_bytes(text: str) -> bytes:
    return lzma.decompress(base64.b64decode(text.encode("ascii")))


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                         text=True, timeout=60)
    if out.returncode != 0:
        raise ScaleError(f"git {' '.join(args)} falló en {repo}: "
                         f"{out.stderr.strip()}")
    return out.stdout


def git_state(repo: Path) -> dict:
    return dict(head=_git(repo, "rev-parse", "HEAD").strip(),
                branch=_git(repo, "symbolic-ref", "--short", "HEAD").strip(),
                clean=_git(repo, "status", "--porcelain").strip() == "")


def _require_ac_pinned() -> dict:
    st = git_state(AC_ROOT)
    if st["head"] != AC_PINNED_SHA:
        raise ScaleError(f"Phy-AC en {st['head']}, se exige {AC_PINNED_SHA}")
    if not st["clean"]:
        raise ScaleError("árbol de Phy-AC con cambios: no se adquiere")
    return st


def _finite(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def load_inlets(path: Path = INLETS_PATH) -> dict:
    if not Path(path).exists():
        return {"schema": INLETS_SCHEMA, "inlets": {}}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_inlets(doc: dict, path: Path = INLETS_PATH) -> None:
    doc["schema"] = INLETS_SCHEMA
    doc["description"] = (
        "Descargas Phy-AC L0 (condiciones TOTALES a la salida del compresor) "
        "que alimentan la matriz de escala de Phy-CB (fase 17.1, D-378). "
        "Cada fila es inmutable y se usa por clave de masa en las dos "
        "arquitecturas anulares. P03_Pa = cfx_boundary_conditions."
        "P0_out_Pa_est, T03_K = T0_out_K_est, mdot_kg_s = mdot_kg_s; "
        "P_static_out_Pa_est queda excluida. contract.data = lzma+base64 de "
        "los bytes exactos del contrato Phy-AC.")
    doc["inlets"] = {k: doc["inlets"][k] for k in sorted(
        doc["inlets"], key=lambda s: float(s))}
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------- verificación filas
def check_row(key: str, row: dict) -> dict:
    """Verifica una fila de descarga contra sus propios bytes de contrato.

    Lanza `ScaleError` ante cualquier incoherencia; devuelve el contrato
    decodificado si todo cuadra. No toca la red ni QSTUDIO."""
    if key not in MASS_KEYS:
        raise ScaleError(f"clave de masa inesperada: {key!r}")
    for f in ("mdot_kg_s", "P03_Pa", "T03_K", "PR"):
        if not _finite(row.get(f)):
            raise ScaleError(f"[{key}] {f} ausente o no finito")
    if float(row["mdot_kg_s"]) != float(key):
        raise ScaleError(f"[{key}] mdot_kg_s={row['mdot_kg_s']} no coincide "
                         "con la clave de masa")
    if row.get("pressure_kind") != "total":
        raise ScaleError(f"[{key}] la presión de entrada debe ser total")
    pr = float(row["PR"])
    if not (PR_BAND[0] <= pr <= PR_BAND[1]):
        raise ScaleError(f"[{key}] PR observado {pr} fuera de {PR_BAND}")
    c = row.get("contract") or {}
    try:
        raw = decode_bytes(c["data"])
    except Exception as exc:  # noqa: BLE001
        raise ScaleError(f"[{key}] bytes del contrato ilegibles: {exc}")
    if len(raw) != int(c.get("size_bytes", -1)):
        raise ScaleError(f"[{key}] tamaño del contrato distinto")
    if sha256_bytes(raw) != c.get("sha256"):
        raise ScaleError(f"[{key}] SHA256 del contrato no reproduce")
    contract = json.loads(raw.decode("utf-8"))
    bc = contract["cfx_boundary_conditions"]
    der = contract["derived"]
    if float(bc["mdot_kg_s"]) != float(row["mdot_kg_s"]):
        raise ScaleError(f"[{key}] caudal distinto del contrato")
    if float(bc["P0_out_Pa_est"]) != float(row["P03_Pa"]):
        if float(bc["P_static_out_Pa_est"]) == float(row["P03_Pa"]):
            raise ScaleError(f"[{key}] P03_Pa es la presión ESTÁTICA")
        raise ScaleError(f"[{key}] P03_Pa distinto de P0_out_Pa_est")
    if float(bc["T0_out_K_est"]) != float(row["T03_K"]):
        raise ScaleError(f"[{key}] T03_K distinto de T0_out_K_est")
    if float(der["PR"]) != pr:
        raise ScaleError(f"[{key}] PR distinto de derived.PR")
    p0_in = float(bc["P0_in_Pa"])
    if abs(float(row["P03_Pa"]) - p0_in * pr) > 1e-9 * float(row["P03_Pa"]):
        raise ScaleError(f"[{key}] P03 != P0_in·PR")
    if der.get("feasible") is not True:
        raise ScaleError(f"[{key}] el candidato Phy-AC no es factible")
    if (contract.get("run_meta") or {}).get("fidelity") != "L0":
        raise ScaleError(f"[{key}] el contrato no es de fidelidad L0")
    cand = row.get("candidate") or {}
    names = cand.get("var_names") or []
    theta = cand.get("theta") or []
    dv = contract["design_vector"]
    if len(names) != len(theta) or list(dv) != list(names):
        raise ScaleError(f"[{key}] θ/nombres del candidato incoherentes")
    for n, v in zip(names, theta):
        if float(dv[n]) != float(v):
            raise ScaleError(f"[{key}] θ[{n}] distinto del contrato")
    for gname in ("g", "reeval_g"):
        g = cand.get(gname)
        if not g or not all(_finite(x) and float(x) <= 0.0 for x in g):
            raise ScaleError(f"[{key}] {gname} ausente o con violación")
    if cand.get("reeval_feasible") is not True:
        raise ScaleError(f"[{key}] la reevaluación L0 no es factible")
    if float(cand.get("reeval_PR", float("nan"))) != pr:
        raise ScaleError(f"[{key}] la reevaluación no reproduce el PR")
    if float(cand.get("reeval_T0_out_K", float("nan"))) != float(row["T03_K"]):
        raise ScaleError(f"[{key}] la reevaluación no reproduce T03")
    if len(cand["reeval_g"]) < len(cand["g"]):
        raise ScaleError(f"[{key}] la reevaluación no usa todas las "
                         "restricciones del spec")
    spec = row.get("spec_effective") or {}
    sm_min = expected_sm_flow_min(key)
    if float(spec.get("sm_flow_min", float("nan"))) != sm_min:
        raise ScaleError(f"[{key}] spec efectivo con sm_flow_min="
                         f"{spec.get('sm_flow_min')!r}, se exige {sm_min!r}")
    exc = AC_SM_FLOW_EXCEPTION.get(key)
    if exc is not None:
        if (row.get("sm_flow_min") != exc["sm_flow_min"]
                or row.get("sm_flow_policy") != exc["policy"]
                or (row.get("sm_flow_exception") or {}).get("vetable")
                is not True
                or (row.get("recipe") or {}).get("sm_flow_min")
                != exc["sm_flow_min"]):
            raise ScaleError(f"[{key}] falta declarar la excepción "
                             f"{exc['policy']} de sm_flow_min")
        sm = cand.get("reeval_sm_flow")
        if not _finite(sm) or float(sm) < exc["sm_flow_min"]:
            raise ScaleError(f"[{key}] margen de bombeo reevaluado {sm!r} "
                             f"bajo el umbral {exc['sm_flow_min']}")
    elif row.get("sm_flow_policy") is not None:
        raise ScaleError(f"[{key}] excepción de sm_flow_min no autorizada")
    prov = row.get("provenance") or {}
    if prov.get("ac_commit") != AC_PINNED_SHA:
        raise ScaleError(f"[{key}] commit de Phy-AC no fijado")
    if not re.fullmatch(r"[0-9a-f]{64}", str(c.get("sha256", ""))):
        raise ScaleError(f"[{key}] SHA256 con formato inválido")
    return contract


def check_inlets(doc: dict) -> dict:
    """Exactamente tres filas 5/8/25, únicas y válidas. Devuelve totales."""
    if doc.get("schema") != INLETS_SCHEMA:
        raise ScaleError("esquema del fixture inesperado")
    inl = doc.get("inlets") or {}
    if sorted(inl, key=float) != sorted(MASS_KEYS, key=float):
        raise ScaleError(f"se esperan las masas {MASS_KEYS}, hay {list(inl)}")
    hashes = [r["contract"]["sha256"] for r in inl.values()]
    if len(set(hashes)) != len(hashes):
        raise ScaleError("contratos repetidos entre filas")
    pairs = [(r["P03_Pa"], r["T03_K"]) for r in inl.values()]
    if len(set(pairs)) != len(pairs):
        raise ScaleError("parejas P03/T03 repetidas entre filas")
    for k, r in inl.items():
        check_row(k, r)
    r25 = inl["25"]
    for f in ("mdot_kg_s", "P03_Pa", "T03_K", "PR"):
        if float(r25[f]) != float(UAT14[f]):
            raise ScaleError(f"[25] {f} difiere del valor histórico UAT 14")
    if r25["contract"]["sha256"] != UAT14["contract_sha256"]:
        raise ScaleError("[25] SHA del contrato distinto del UAT 14")
    if r25["provenance"].get("run_id") != UAT14["run_id"]:
        raise ScaleError("[25] id histórico distinto")
    if r25["provenance"].get("manifest_sha256") != UAT14["manifest_sha256"]:
        raise ScaleError("[25] huella canónica del manifest distinta")
    return dict(count=len(inl), keys=sorted(inl, key=float),
                contract_sha256={k: r["contract"]["sha256"]
                                 for k, r in inl.items()})


# ----------------------------------------------- reevaluación dentro de AC
def _ac_reeval_main(args) -> int:
    """Subproceso con cwd = checkout de Phy-AC: reevalúa θ en L0."""
    sys.path.insert(0, str(AC_ROOT))
    import numpy as np  # noqa: F401
    import physics_core as pc  # Phy-AC
    import neural_optimizer as no  # Phy-AC
    if Path(pc.__file__).resolve().parent != AC_ROOT.resolve():
        raise ScaleError(f"physics_core cargado de {pc.__file__}")
    req = json.loads(Path(args.request).read_text(encoding="utf-8"))
    spec = no.DesignSpec(**req["spec"])
    res = no.evaluate_design(spec, req["theta"], fidelity=pc.Fidelity.L0)
    rec = res["record"]
    g = spec.constraints(rec, res["theta"])
    out = dict(var_names=list(pc.VAR_NAMES),
               theta=[float(x) for x in res["theta"]],
               g=[float(x) for x in g],
               feasible=bool(spec.violation(g) <= 1e-9),
               PR=float(rec["PR"]), T0_out_K=float(rec["T0_out"]),
               sm_flow=(float(rec["sm_flow"]) if _finite(rec.get("sm_flow"))
                        else None),
               sm_flow_min=float(spec.sm_flow_min),
               source=str(rec.get("source", "")))
    Path(args.response).write_text(json.dumps(out), encoding="utf-8")
    return 0


def _ac_reeval(spec: dict, theta: list, workdir: Path) -> dict:
    req = workdir / f"reeval-{uuid.uuid4().hex[:8]}.json"
    resp = req.with_suffix(".out.json")
    req.write_text(json.dumps(dict(spec=spec, theta=theta)), encoding="utf-8")
    p = subprocess.run([ac_python(), "-B", "-X", "utf8", str(Path(__file__)),
                        "_ac-reeval", "--request", str(req),
                        "--response", str(resp)],
                       cwd=str(AC_ROOT), capture_output=True, text=True,
                       timeout=120)
    if p.returncode != 0 or not resp.exists():
        raise ScaleError(f"reevaluación Phy-AC falló: {p.stderr[-800:]}")
    return json.loads(resp.read_text(encoding="utf-8"))


def _match_checkpoint_row(ckpt: dict, theta: list) -> dict:
    best = None
    for row in ckpt.get("dataset", []):
        t = [float(x) for x in row["theta"]]
        if len(t) == len(theta) and all(
                abs(a - b) <= 1e-12 * max(1.0, abs(b)) for a, b in zip(t, theta)):
            best = row
            break
    if best is None:
        raise ScaleError("el θ del contrato no aparece en el checkpoint")
    return best


def _row_from_ac_run(key: str, outdir: Path, recipe: dict, prov: dict,
                     contract_raw: bytes, ckpt: dict) -> dict:
    contract = json.loads(contract_raw.decode("utf-8"))
    bc = contract["cfx_boundary_conditions"]
    der = contract["derived"]
    names = list(contract["design_vector"])
    theta = [float(contract["design_vector"][n]) for n in names]
    ck_row = _match_checkpoint_row(ckpt, theta)
    spec = ckpt["spec"]
    rr = _ac_reeval(spec, theta, outdir)
    if rr["var_names"] != names:
        raise ScaleError("VAR_NAMES de Phy-AC distinto del contrato")
    row = dict(
        mass_key=key,
        mdot_kg_s=float(bc["mdot_kg_s"]),
        P03_Pa=float(bc["P0_out_Pa_est"]),
        T03_K=float(bc["T0_out_K_est"]),
        PR=float(der["PR"]),
        pressure_kind="total",
        boundary_fields=dict(
            mdot_kg_s=bc["mdot_kg_s"], P0_out_Pa_est=bc["P0_out_Pa_est"],
            T0_out_K_est=bc["T0_out_K_est"], P0_in_Pa=bc["P0_in_Pa"],
            T0_in_K=bc["T0_in_K"], RPM=bc["RPM"]),
        excluded_fields=["P_static_out_Pa_est"],
        candidate=dict(
            var_names=names, theta=theta,
            g=[float(x) for x in ck_row["g"]],
            feasible=bool(der["feasible"]),
            n_stages=der.get("n_stages"), eta_poly=der.get("eta_poly"),
            min_SM=der.get("min_SM"), sm_flow=der.get("sm_flow"),
            U_tip_m_s=der.get("U_tip_m_s"), r_tip_in_mm=der.get("r_tip_in_mm"),
            source=der.get("source"),
            reeval_g=rr["g"], reeval_feasible=rr["feasible"],
            reeval_PR=rr["PR"], reeval_T0_out_K=rr["T0_out_K"],
            reeval_sm_flow=rr.get("sm_flow"),
            reeval_sm_flow_min=rr.get("sm_flow_min"),
            reeval_source=rr["source"]),
        spec_effective=spec,
        recipe=recipe,
        provenance=prov,
        contract=dict(path_in_run=CONTRACT_REL, sha256=sha256_bytes(contract_raw),
                      size_bytes=len(contract_raw), encoding="lzma+base64",
                      data=encode_bytes(contract_raw)),
    )
    return row


def _ac_argv(key: str, att: dict, outdir: Path) -> list[str]:
    c = ac_common(key)
    return ["phyac_cli.py", "--fidelity", "L0", "--mdot", key,
            "--pr", repr(AC_TARGET_PR[key]), "--t0", repr(c["t0"]),
            "--p0", repr(c["p0"]), "--rpm-max", repr(c["rpm_max"]),
            "--utip-max", repr(c["utip_max"]), "--rtip-max", repr(c["rtip_max"]),
            "--nstages-max", str(c["nstages_max"]),
            "--sm-flow-min", repr(c["sm_flow_min"]), "--material", c["material"],
            "--seed", str(att["seed"]), "--rounds", str(att["rounds"]),
            "--n-init", str(att["n_init"]), "--batch-size", str(att["batch_size"]),
            "--no-figures", "--no-color", "--outdir", str(outdir)]


def _same_recipe(row: dict, key: str) -> bool:
    rec = row.get("recipe") or {}
    att = rec.get("attempt_params")
    return (row.get("provenance", {}).get("ac_commit") == AC_PINNED_SHA
            and rec.get("target_PR") == AC_TARGET_PR.get(key)
            and att in [dict(a) for a in AC_ATTEMPTS]
            and rec.get("common") == ac_common(key))


def _verify_prior_attempt1(key: str, outdir: Path) -> dict:
    """Reutiliza la evidencia de un intento 1 ya ejecutado (D-387).

    Sólo vale si el directorio es de la receta exacta del intento 1 para
    esta masa (spec, semilla, presupuesto, fidelidad, receta común) y si su
    candidato recomendado falló el criterio (PR fuera de banda o no
    factible). Un intento 1 que hubiera pasado no se puede saltar."""
    att = AC_ATTEMPTS[0]
    contract_path = outdir / CONTRACT_REL
    ckpt_path = outdir / "phyac_run.json"
    log = outdir / "phyac_cli.log"
    for p in (contract_path, ckpt_path, log):
        if not p.exists():
            raise ScaleError(f"intento 1 previo incompleto: falta {p.name}")
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    raw = contract_path.read_bytes()
    contract = json.loads(raw.decode("utf-8"))
    spec = ckpt.get("spec") or {}
    meta = contract.get("run_meta") or {}
    c = ac_common(key)
    expect = dict(PR_target=AC_TARGET_PR[key], massflow=float(key),
                  RPM_max=c["rpm_max"], U_tip_max=c["utip_max"],
                  r_tip_max_mm=c["rtip_max"], n_stages_max=c["nstages_max"],
                  sm_flow_min=c["sm_flow_min"], T0_in=c["t0"], P0_in=c["p0"],
                  material=c["material"])
    for f, v in expect.items():
        if spec.get(f) != v:
            raise ScaleError(f"intento 1 previo con {f}={spec.get(f)!r}, "
                             f"se espera {v!r}")
    n_expected = att["n_init"] + att["rounds"] * att["batch_size"]
    if (int(ckpt.get("seed", -1)) != att["seed"]
            or int(ckpt.get("n_evals", -1)) != n_expected
            or meta.get("rounds") != att["rounds"]
            or meta.get("n_init") != att["n_init"]
            or meta.get("fidelity") != "L0"):
        raise ScaleError("el intento 1 previo no es de la receta predefinida")
    pr = float(contract["derived"]["PR"])
    feas = contract["derived"].get("feasible") is True
    if feas and PR_BAND[0] <= pr <= PR_BAND[1]:
        raise ScaleError("el intento 1 previo pasa el criterio: no se salta")
    reason = (f"[{key}] PR observado {pr!r} fuera de {PR_BAND}" if feas
              else f"[{key}] el candidato Phy-AC no es factible")
    return dict(attempt=1, attempt_params=dict(att), timeout_s=AC_TIMEOUTS_S[0],
                outdir_name=outdir.name, n_evals=ckpt.get("n_evals"),
                seed=ckpt.get("seed"), recommended_PR=pr,
                recommended_feasible=feas, contract_sha256=sha256_bytes(raw),
                checkpoint_sha256=sha256_file(ckpt_path),
                log_sha256=sha256_file(log), reason=reason,
                reused="evidencia del intento 1 verificada sin relanzar (D-387)")


def cmd_acquire(args) -> int:
    key = str(args.mass).rstrip("0").rstrip(".") if "." in str(args.mass) \
        else str(args.mass)
    if key == "25":
        doc = load_inlets()
        if "25" not in doc.get("inlets", {}):
            raise ScaleError("25 kg/s se congela desde UAT 14 con "
                             "archive-uat14, no se relanza")
        check_row("25", doc["inlets"]["25"])
        print("[acquire 25] fila histórica UAT 14 verificada (sin relanzar)")
        return 0
    if key not in AC_TARGET_PR:
        raise ScaleError(f"masa sin receta: {key}")
    doc = load_inlets()
    if key in doc.get("inlets", {}) and not args.force:
        row = doc["inlets"][key]
        if not _same_recipe(row, key):
            raise ScaleError(f"[{key}] la fila existente no es de esta "
                             "receta/SHA; usar --force para relanzar")
        check_row(key, row)
        print(f"[acquire {key}] evidencia existente válida "
              f"(contrato {row['contract']['sha256']}, PR {row['PR']}); "
              "no se relanza ni sobrescribe")
        return 0
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    failures = []
    prior = []
    start = 1
    if getattr(args, "prior_attempt1", None):
        prior.append(_verify_prior_attempt1(key, Path(args.prior_attempt1)))
        failures.append(dict(attempt=1, reason=prior[0]["reason"],
                             outdir=str(Path(args.prior_attempt1))))
        start = 2
    for n, att in enumerate(AC_ATTEMPTS, start=1):
        if n < start:
            continue
        timeout_s = AC_TIMEOUTS_S[n - 1]
        st0 = _require_ac_pinned()
        run_id = uuid.uuid4().hex[:12]
        outdir = out_root / f"ac-{key}-a{n}-{run_id}"
        outdir.mkdir(parents=True, exist_ok=False)
        argv = _ac_argv(key, att, outdir)
        log = outdir / "phyac_cli.log"
        t0, started = time.monotonic(), _now()
        try:
            with open(log, "w", encoding="utf-8") as fh:
                p = subprocess.run([ac_python(), "-B", "-X", "utf8", *argv],
                                   cwd=str(AC_ROOT), stdout=fh,
                                   stderr=subprocess.STDOUT,
                                   timeout=timeout_s)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        elapsed = round(time.monotonic() - t0, 3)
        st1 = git_state(AC_ROOT)
        if st1["head"] != AC_PINNED_SHA or not st1["clean"]:
            raise ScaleError("Phy-AC cambió durante la adquisición")
        why = None
        contract_path = outdir / CONTRACT_REL
        ckpt_path = outdir / "phyac_run.json"
        if rc != 0:
            why = f"código de retorno {rc}"
        elif not contract_path.exists() or not ckpt_path.exists():
            why = "faltan contrato o checkpoint"
        if why is None:
            raw = contract_path.read_bytes()
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            recipe = dict(
                argv=["python", "-B", "-X", "utf8"] + [
                    a if a != str(outdir) else "<outdir>" for a in argv],
                cwd="<Phy-AC>", target_PR=AC_TARGET_PR[key],
                target_policy=AC_TARGET_POLICY[key],
                superseded_targets=AC_SUPERSEDED_TARGETS.get(key, []),
                common=ac_common(key),
                attempt=n, attempt_params=dict(att), timeout_s=timeout_s,
                timeouts_s=list(AC_TIMEOUTS_S),
                timeout_policy=AC_TIMEOUT_POLICY, fidelity="L0")
            sm_exc = AC_SM_FLOW_EXCEPTION.get(key)
            if sm_exc is not None:
                recipe.update(sm_flow_min=sm_exc["sm_flow_min"],
                              sm_flow_policy=sm_exc["policy"],
                              sm_flow_exception=dict(sm_exc))
            prov = dict(
                kind="acquired", repo="Phy-AC", repo_url=AC_REPO_URL,
                ac_commit=st0["head"], branch=st0["branch"],
                tree_clean_before=st0["clean"], tree_clean_after=st1["clean"],
                run_id=run_id, outdir_name=outdir.name, started_utc=started,
                finished_utc=_now(), elapsed_s=elapsed,
                n_evals=ckpt.get("n_evals"), seed=ckpt.get("seed"),
                python=sys.version.split()[0],
                ac_interpreter=Path(ac_python()).as_posix(),
                log_sha256=sha256_file(log), prior_attempts=prior,
                target_PR=AC_TARGET_PR[key],
                target_rule=AC_TARGET_POLICY[key],
                feasibility_rule=("spec.constraints completo (todas las g) "
                                  "con el spec efectivo del checkpoint; "
                                  f"sm_flow_min={expected_sm_flow_min(key)}"))
            if sm_exc is not None:
                prov.update(sm_flow_rule=sm_exc["policy"])
            try:
                row = _row_from_ac_run(key, outdir, recipe, prov, raw, ckpt)
                if sm_exc is not None:
                    row.update(sm_flow_min=sm_exc["sm_flow_min"],
                               sm_flow_policy=sm_exc["policy"],
                               sm_flow_exception=dict(sm_exc))
                check_row(key, row)
            except ScaleError as exc:
                why = str(exc)
        print(f"[acquire {key}] intento {n}: rc={rc} t={elapsed}s "
              f"→ {'OK' if why is None else why}")
        if why is None:
            doc = load_inlets()
            doc.setdefault("inlets", {})[key] = row
            save_inlets(doc)
            print(f"[acquire {key}] PR={row['PR']!r} P03={row['P03_Pa']!r} "
                  f"T03={row['T03_K']!r} contrato={row['contract']['sha256']}")
            return 0
        failures.append(dict(attempt=n, reason=why, outdir=str(outdir)))
    print(f"[acquire {key}] GATE AGOTADO: {json.dumps(failures)}")
    return 2


# --------------------------------------------------------- 25 kg/s (UAT 14)
def cmd_archive_uat14(args) -> int:
    """Congela la fila de 25 kg/s desde la corrida histórica (sólo lectura)."""
    _require_ac_pinned()
    run_dir = Path(args.run_dir)
    campaign = Path(args.campaign) if args.campaign else None
    raw = (run_dir / CONTRACT_REL).read_bytes()
    if sha256_bytes(raw) != UAT14["contract_sha256"]:
        raise ScaleError("el contrato histórico no reproduce su SHA publicado")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    ckpt = json.loads((run_dir / "phyac_run.json").read_text(encoding="utf-8"))
    if status.get("run_id") != UAT14["run_id"] or status.get("status") != "done":
        raise ScaleError("status.json no corresponde a la corrida UAT 14")
    cfg = status["config"]
    res = status["result"]
    if not (res.get("feasible") is True and res.get("fidelity_reached") == "L0"):
        raise ScaleError("la corrida histórica no es L0 factible")
    if int(cfg["seed"]) != UAT14["seed"] or int(ckpt["seed"]) != UAT14["seed"]:
        raise ScaleError("semilla histórica distinta")
    huella = None
    campaign_sha = None
    if campaign is not None and campaign.exists():
        campaign_sha = sha256_file(campaign)
        cj = json.loads(campaign.read_text(encoding="utf-8"))
        for h in cj["campana"]["huellas"]:
            if h.get("run_id") == UAT14["run_id"]:
                huella = h
        if huella is None:
            raise ScaleError("la campaña no cita la corrida 1f9e4fa586c2")
        hc = huella["huella_corrida"]
        if (hc["commit"] != UAT14["commit"] or hc["branch"] != UAT14["branch"]
                or hc["manifest_sha256"] != UAT14["manifest_sha256"]):
            raise ScaleError("la huella de la campaña no cuadra con UAT 14")
        source_used = "14-campana-real.json + status.json + contrato"
    else:
        source_used = "huellas de RESEARCH §4 + status.json + contrato"
    recipe = dict(
        argv=["python", "-B", "-X", "utf8", "phyac_cli.py", "--fidelity", "L0",
              "--mdot", "25", "--pr", "4", "--t0", "288.15", "--p0", "101325",
              "--rpm-max", "20000", "--utip-max", "480", "--rtip-max", "400",
              "--nstages-max", "8", "--material", "Ti-6Al-4V",
              "--seed", str(UAT14["seed"]), "--rounds", "2", "--n-init", "100",
              "--batch-size", "8", "--outdir", "<outdir>"],
        note=("receta equivalente a la config de Studio de la corrida "
              "histórica (ruta Studio, no CLI); sm_flow_min por defecto 0.15"),
        target_PR=4.0, attempt=None, attempt_params=None,
        studio_config=cfg, fidelity="L0")
    prov = dict(
        kind="archived_uat14", repo="Phy-AC", repo_url=AC_REPO_URL,
        ac_commit=UAT14["commit"], branch=UAT14["branch"],
        manifest_sha256=UAT14["manifest_sha256"], run_id=UAT14["run_id"],
        historical_path=("backend/app/workspace/phy_models/phyac/runs/"
                         "1f9e4fa586c2"),
        campaign_path=(".planning/phases/14-campa-a-de-sistema-e-integraci-"
                       "n-iterativa/14-campana-real.json"),
        campaign_sha256=campaign_sha, campaign_huella=huella,
        status_created_at=status.get("created_at"),
        wall_clock_s=res.get("wall_clock_s"),
        n_evals=ckpt.get("n_evals"), seed=UAT14["seed"],
        status_sha256=sha256_file(run_dir / "status.json"),
        checkpoint_sha256=sha256_file(run_dir / "phyac_run.json"),
        source_used=source_used, archived_utc=_now())
    tmp = Path(args.out_root).resolve() if args.out_root else run_dir
    tmp.mkdir(parents=True, exist_ok=True)
    row = _row_from_ac_run("25", tmp, recipe, prov, raw, ckpt)
    for f in ("mdot_kg_s", "P03_Pa", "T03_K", "PR"):
        if float(row[f]) != float(UAT14[f]):
            raise ScaleError(f"[25] {f}={row[f]!r} difiere del valor publicado")
    check_row("25", row)
    doc = load_inlets()
    doc.setdefault("inlets", {})["25"] = row
    save_inlets(doc)
    print(f"[archive-uat14] fila 25 congelada ({source_used}); "
          f"contrato {row['contract']['sha256']}")
    return 0


# ------------------------------------------------------- diagnóstico de CB
def _import_cb():
    if str(CB_ROOT) not in sys.path:
        sys.path.insert(0, str(CB_ROOT))
    import physics_core as pc
    import neural_optimizer as no
    import geometry_generator as gg
    import dfam_checker as dc
    if Path(pc.__file__).resolve().parent != CB_ROOT.resolve():
        raise ScaleError(f"physics_core cargado de {pc.__file__}")
    return pc, no, gg, dc


def _class_table_rows() -> list[dict]:
    rows = []
    path = CB_ROOT / "data" / "reference" / "class_table.csv"
    header = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if header is None:
            header = parts
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def historical_class(mdot: float) -> dict:
    rows = _class_table_rows()
    cover = [r["class_name"] for r in rows if r["param"] == "mdot_air_kgs"
             and float(r["min"]) <= mdot <= float(r["max"])]
    top = max(float(r["max"]) for r in rows if r["param"] == "mdot_air_kgs")
    return dict(classes_covering=cover, documented_max_kgs=top,
                label=(cover[0] if cover else
                       f"fuera de la banda documentada (máx. {top} kg/s)"))


def _diag_argv(row: dict, outdir: Path) -> list[str]:
    r = DIAG_RECIPE
    argv = ["phycb_cli.py", "--optimize", "--fidelity", r["fidelity"],
            "--mdot", repr(float(row["mdot_kg_s"])),
            "--p3", repr(float(row["P03_Pa"])), "--t3", repr(float(row["T03_K"])),
            "--t4", repr(r["t4"]), "--arch", r["arch"], "--fuel", r["fuel"],
            "--material", r["material"], "--material-casing", r["material_casing"],
            "--dp-max", repr(r["dp_max"]), "--pf-max", repr(r["pf_max"]),
            "--seed", str(r["seed"]), "--n-init", str(r["n_init"]),
            "--rounds", str(r["rounds"]), "--batch-size", str(r["batch_size"]),
            "--no-figures", "--no-color", "--outdir", str(outdir)]
    if not r["dfam"]:
        argv.insert(-2, "--no-dfam")
    return argv


def _named_g(g: list[float], names: list[str]) -> dict:
    return {n: float(v) for n, v in zip(names, g)}


def _read_archived_diag() -> dict | None:
    if not BASIS_PATH.exists():
        return None
    txt = BASIS_PATH.read_text(encoding="utf-8")
    if DIAG_BEGIN not in txt:
        return None
    block = txt.split(DIAG_BEGIN, 1)[1].split(DIAG_END, 1)[0]
    m = re.search(r"```json\n(.*?)\n```", block, re.S)
    return json.loads(m.group(1)) if m else None


def _diag_markdown(d: dict) -> str:
    g = d["g_blocks"]
    viol = [f"`{k}` = {v:+.4f}" for blk in g.values() for k, v in blk.items()
            if v > 0]
    rec = d["record"]
    lines = [
        DIAG_BEGIN,
        "### Traza diagnóstica L0 a 8 kg/s (generada por `scale_l0.py "
        "diagnose`)",
        "",
        "Fase **diagnóstica**: no cuenta para la matriz de aceptación.",
        "",
        f"- Entrada: fila `8` de `scale_inlets.json`, contrato Phy-AC "
        f"`{d['inlet']['contract_sha256']}`; P03 = {d['inlet']['P03_Pa']!r} Pa, "
        f"T03 = {d['inlet']['T03_K']!r} K, ṁ = {d['inlet']['mdot_kg_s']!r} kg/s.",
        f"- Receta: `{' '.join(d['recipe']['argv'])}` (cwd Phy-CB, tope "
        f"{d['recipe']['timeout_s']:.0f} s).",
        f"- Phy-CB `{d['cb']['head']}` ({d['cb']['branch']}), clase "
        f"histórica: {d['historical_class']['label']}; límites τ aplicados "
        f"{d['tau_limits_ms']['min']}–{d['tau_limits_ms']['max']} ms.",
        f"- Resultado: factible = **{d['feasible']}**; violaciones: "
        + (", ".join(viol) if viol else "ninguna") + ".",
        f"- τ = {rec['tau_ms']:.3f} ms, τ_PZ = {rec['tau_pz_ms']:.3f} ms, "
        f"φ_PZ = {rec['phi_pz']:.4f}, φ_LBO = {rec['phi_LBO']:.4f}, margen "
        f"LBO = {rec['LBO_margin']:.4f}, D_L = {rec['D_L_mm']:.1f} mm, "
        f"L_tot = {rec['L_tot_mm']:.1f} mm, V_c = {rec['V_c_m3']:.5f} m³, "
        f"sizing_driver = {rec['sizing_driver']}.",
        f"- Contrato Phy-CB `{d['contract']['sha256']}`; θ "
        f"`{d['theta_sha256']}`; DfAM real: {d['dfam']['checker']}.",
        f"- Tiempo: {d['timing']['elapsed_s']} s ({d['timing']['started_utc']}).",
        "",
        "```json",
        json.dumps(d, indent=1, ensure_ascii=False, allow_nan=False),
        "```",
        DIAG_END,
    ]
    return "\n".join(lines)


def _write_diag_block(d: dict) -> None:
    block = _diag_markdown(d)
    if BASIS_PATH.exists():
        txt = BASIS_PATH.read_text(encoding="utf-8")
    else:
        txt = "# Base técnica de la escala 5–25 kg/s de Phy-CB\n\n"
    if DIAG_BEGIN in txt:
        pre, rest = txt.split(DIAG_BEGIN, 1)
        post = rest.split(DIAG_END, 1)[1]
        txt = pre + block + post
    else:
        txt = txt.rstrip("\n") + "\n\n" + block + "\n"
    BASIS_PATH.write_text(txt, encoding="utf-8", newline="\n")


def cmd_diagnose(args) -> int:
    key = str(args.mass)
    if key != "8":
        raise ScaleError("el diagnóstico de la Tarea 1 es a 8 kg/s")
    doc = load_inlets()
    row = (doc.get("inlets") or {}).get(key)
    if row is None:
        raise ScaleError("falta la fila 8 de scale_inlets.json")
    check_row(key, row)
    cb_state = git_state(CB_ROOT)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    outdir = out_root / f"cb-diag-{key}-{run_id}"
    outdir.mkdir(parents=True, exist_ok=False)
    argv = _diag_argv(row, outdir)
    log = outdir / "phycb_cli.log"
    t0, started = time.monotonic(), _now()
    try:
        with open(log, "w", encoding="utf-8") as fh:
            p = subprocess.run([sys.executable, "-B", "-X", "utf8", *argv],
                               cwd=str(CB_ROOT), stdout=fh,
                               stderr=subprocess.STDOUT, timeout=DIAG_TIMEOUT_S)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        raise ScaleError(f"diagnóstico sin terminar en {DIAG_TIMEOUT_S} s")
    elapsed = round(time.monotonic() - t0, 3)
    if rc not in (0, 3):           # 3 = sin diseño factible (resultado)
        raise ScaleError(f"phycb_cli devolvió {rc}; ver {log}")
    ckpt_p = outdir / "phycb_run.json"
    con_p = outdir / "geometry" / "combustor.json"
    dfam_p = outdir / "geometry" / "dfam_report.json"
    for f in (ckpt_p, con_p, dfam_p):
        if not f.exists():
            raise ScaleError(f"falta evidencia: {f.name}")
    if git_state(CB_ROOT) != cb_state:
        raise ScaleError("el árbol de Phy-CB cambió durante el diagnóstico")
    pc, no, gg, dc = _import_cb()
    import numpy as np
    ckpt = json.loads(ckpt_p.read_text(encoding="utf-8"))
    con_raw = con_p.read_bytes()
    contract = json.loads(con_raw.decode("utf-8"))
    dv = contract["design_vector"]
    spec_d = ckpt["spec"]
    for f, rk in (("P3_Pa", "P03_Pa"), ("T3_K", "T03_K"),
                  ("mdot_kgs", "mdot_kg_s")):
        if float(spec_d[f]) != float(row[rk]):
            raise ScaleError(f"el spec de la corrida no usa {rk} de la fila")
    names = list(pc.VAR_NAMES)
    theta = None
    for r in ckpt["dataset"]:
        t = [float(x) for x in pc.pad_theta(np.asarray(r["theta"], float))]
        if all(abs(t[i] - float(dv[n])) <= 1e-8 * max(1.0, abs(t[i]))
               for i, n in enumerate(names)):
            theta, ck_row = t, r
            break
    if theta is None:
        raise ScaleError("el θ del contrato no aparece en el checkpoint")
    spec = no.DesignSpec(**spec_d)
    th = np.asarray(theta, dtype=float)
    rec = pc.evaluate(th, fidelity=pc.Fidelity.L0, use_cache=False,
                      calibrate=False)
    g = [float(x) for x in spec.constraints(rec, th)]
    if ck_row.get("g") is not None:
        # el checkpoint guarda el g FÍSICO (12); el global añade spec,
        # estructura y DfAM, que se recalculan aquí sobre el record L0.
        gk = [float(x) for x in ck_row["g"]]
        if len(gk) not in (pc.N_CONSTRAINTS, len(g)) or any(
                abs(a - b) > 1e-9 * max(1.0, abs(b)) for a, b in zip(g, gk)):
            raise ScaleError("la reevaluación no reproduce los g archivados")
    if no._dfam_module() is None:
        raise ScaleError("dfam_checker no disponible: DfAM neutro")
    g_dfam_file = [float(x) for x in dc.check(contract)["g_dfam"]]
    n_aero = pc.N_CONSTRAINTS
    blocks = dict(
        aero=_named_g(g[:n_aero], list(pc.CONSTRAINT_NAMES)),
        spec=_named_g(g[n_aero:n_aero + 4],
                      ["spec_dp_max", "spec_PF_max", "spec_envelope_D",
                       "spec_envelope_L"]),
        struct=_named_g(g[n_aero + 4:n_aero + 9],
                        ["T_wall", "sigma_th", "hoop", "t_wall", "banned"]),
        dfam=_named_g(g[n_aero + 9:], list(dc.G_NAMES)),
    )
    keep = ("tau_ms", "tau_pz_ms", "phi_pz", "phi_LBO", "LBO_margin",
            "LBO_margin_ratio", "V_c_m3", "V_pz_m3", "V_sz_m3", "V_dz_m3",
            "D_L_mm", "D_ref_mm", "D_casing_mm", "A_ref_m2", "A_ref_aero_m2",
            "A_ref_chem_m2", "A_L_m2", "A_an_m2", "L_pz_mm", "L_sz_mm",
            "L_dz_mm", "L_tot_mm", "U_ref_ms", "rho3_kgm3", "SMD_um",
            "sizing_driver", "eta_comb", "dp_over_p", "PF", "T04_K",
            "P04_Pa", "mdot_air_kgs", "mdot_fuel_kgs", "mdot4_kgs",
            "I_MW_m3_atm", "theta_lefebvre", "Omega_load", "atomizer_model",
            "source", "fidelity", "arch")
    rec_sub = {k: (rec.get(k) if not isinstance(rec.get(k), (np.floating,))
                   else float(rec.get(k))) for k in keep}
    theta_bytes = json.dumps(theta).encode("utf-8")
    hist = ckpt_p.read_bytes()
    d = dict(
        schema=DIAG_SCHEMA, phase="diagnostic", counts_for_acceptance=False,
        inlet=dict(mass_key=key, mdot_kg_s=row["mdot_kg_s"],
                   P03_Pa=row["P03_Pa"], T03_K=row["T03_K"], PR=row["PR"],
                   contract_sha256=row["contract"]["sha256"]),
        recipe=dict(DIAG_RECIPE, timeout_s=DIAG_TIMEOUT_S,
                    argv=["python", "-B", "-X", "utf8"] + [
                        a if a != str(outdir) else "<outdir>" for a in argv]),
        cb=cb_state, run_id=run_id, outdir_name=outdir.name, returncode=rc,
        historical_class=historical_class(float(row["mdot_kg_s"])),
        tau_limits_ms=dict(min=pc.TAU_MIN_MS, max=pc.TAU_MAX_MS),
        n_evals=ckpt.get("n_evals"),
        theta=dict(zip(names, theta)),
        theta_sha256=sha256_bytes(theta_bytes),
        feasible=bool(all(v <= 0.0 for v in g)),
        g_blocks=blocks,
        g_dfam_from_contract=g_dfam_file,
        dfam=dict(checker="dfam_checker.check (real)",
                  violations=json.loads(dfam_p.read_text(encoding="utf-8"))
                  ["result"].get("violations")),
        record=rec_sub,
        contract=dict(sha256=sha256_bytes(con_raw), size_bytes=len(con_raw),
                      derived={k: contract["derived"].get(k) for k in (
                          "T04_K", "P04_Pa", "mdot_air_kgs", "mdot_fuel_kgs",
                          "tau_ms", "dp_over_p", "PF")}),
        checkpoint_sha256=sha256_bytes(hist),
        log_sha256=sha256_file(log),
        timing=dict(started_utc=started, finished_utc=_now(),
                    elapsed_s=elapsed),
    )
    (outdir / "diagnose_8.json").write_text(
        json.dumps(d, indent=1, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    old = _read_archived_diag()
    if old is not None and not args.rewrite:
        same = (old.get("inlet") == d["inlet"]
                and old.get("theta_sha256") == d["theta_sha256"]
                and old.get("contract", {}).get("sha256") == d["contract"]["sha256"]
                and all(abs(old["g_blocks"][b][k] - d["g_blocks"][b][k]) <= 1e-9
                        for b in d["g_blocks"] for k in d["g_blocks"][b]))
        if not same:
            raise ScaleError("el diagnóstico no reproduce el archivado en "
                             "scale_basis.md (usar --rewrite si cambió la "
                             "física a propósito)")
        print(f"[diagnose 8] reproduce el diagnóstico archivado "
              f"(θ {d['theta_sha256'][:12]}, contrato "
              f"{d['contract']['sha256'][:12]}); evidencia en {outdir}")
        return 0
    _write_diag_block(d)
    viol = {k: v for blk in blocks.values() for k, v in blk.items() if v > 0}
    print(f"[diagnose 8] factible={d['feasible']} violaciones={viol} "
          f"τ={rec_sub['tau_ms']:.3f} ms; archivado en scale_basis.md")
    return 0


# ------------------------------------------------------------- handoff ola 1
SUMMARY_FIELDS = ("model_basis_sha", "inlets_sha256", "basis_sha256",
                  "inlets_count", "engineering_gate")


def parse_summary_fields(text: str, fields=SUMMARY_FIELDS) -> dict:
    """Campos `clave: valor` de UNA línea; una clave repetida es error."""
    found: dict = {}
    for line in text.splitlines():
        m = re.fullmatch(r"\s*-?\s*`?([a-z_0-9]+)`?\s*:\s*`?([^`\s]+)`?\s*",
                         line)
        if m and m.group(1) in fields:
            if m.group(1) in found and found[m.group(1)] != m.group(2):
                raise ScaleError(f"campo repetido con valores distintos: "
                                 f"{m.group(1)}")
            found[m.group(1)] = m.group(2)
    missing = [f for f in fields if f not in found]
    if missing:
        raise ScaleError(f"faltan campos en el SUMMARY: {missing}")
    return found


def cmd_verify_handoff(args) -> int:
    if args.wave == 2:
        return cmd_verify_handoff_2(args)
    if args.wave == 3:
        return cmd_verify_handoff_3(args)
    if args.wave != 1:
        raise ScaleError("sólo se definen los handoff de las olas 1, 2 y 3")
    f = parse_summary_fields(Path(args.summary).read_text(encoding="utf-8"))
    for k in ("model_basis_sha",):
        if not re.fullmatch(r"[0-9a-f]{40}", f[k]):
            raise ScaleError(f"{k} debe ser un SHA de commit en minúsculas")
    for k in ("inlets_sha256", "basis_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", f[k]):
            raise ScaleError(f"{k} debe ser un SHA256 en minúsculas")
    if f["engineering_gate"] != "PASS":
        raise ScaleError(f"engineering_gate = {f['engineering_gate']}: "
                         "la base técnica no habilita las olas siguientes")
    try:
        _git(CB_ROOT, "cat-file", "-e", f["model_basis_sha"] + "^{commit}")
    except ScaleError:
        raise ScaleError("model_basis_sha no existe en el checkout de Phy-CB")
    for rel, k in (("validation/scale_inlets.json", "inlets_sha256"),
                   ("validation/scale_basis.md", "basis_sha256")):
        blob = subprocess.run(
            ["git", "-C", str(CB_ROOT), "show", f"{f['model_basis_sha']}:{rel}"],
            capture_output=True, timeout=60)
        if blob.returncode != 0:
            raise ScaleError(f"{rel} no está en {f['model_basis_sha']}")
        if sha256_bytes(blob.stdout) != f[k]:
            raise ScaleError(f"{k} no coincide con {rel} en el commit")
        # el digest canónico es el del blob confirmado (LF); el árbol puede
        # llevar CRLF si core.autocrlf lo reescribe al hacer checkout.
        work = (CB_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        if sha256_bytes(work) != f[k]:
            raise ScaleError(f"{rel} del árbol difiere del digest del SUMMARY")
    doc = load_inlets()
    tot = check_inlets(doc)
    if int(f["inlets_count"]) != 3 or tot["count"] != 3:
        raise ScaleError("inlets_count distinto de 3")
    print(f"[verify-handoff 1] OK · base {f['model_basis_sha']} · "
          f"{tot['count']} descargas · gate PASS")
    return 0


def cmd_check_inlets(args) -> int:
    tot = check_inlets(load_inlets())
    print(json.dumps(tot, indent=1))
    return 0


# ============================================================ ola 2 (17.1-02)
#: Aceptación de la matriz de escala (D-377–D-379, D-381). Una celda es UNA
#: búsqueda por la CLI normal con una arquitectura explícita, la descarga
#: congelada del caudal y esta receta; la factibilidad la decide la
#: verificación independiente, nunca el código de salida de la CLI.
ACCEPT_SCHEMA = "phycb.scale_acceptance/1"
ACCEPTANCE_PATH = VALIDATION_DIR / "scale_acceptance.json"
ACCEPT_ARCHS = ("annular", "reverse_annular")
ACCEPT_RECIPE = dict(fidelity="L0", t4=1200.0, t4_max=1250.0, fuel="jet_a",
                     material="Inconel-625", material_casing="Inconel-718",
                     dp_max=0.04, pf_max=0.40, dfam=True, workers=1,
                     figures=False)
#: Etapa 1 y ÚNICA escalada predefinida; si escala, se reintentan AMBAS.
ACCEPT_STAGES = (dict(stage=1, seed=71, n_init=320, rounds=5, batch_size=14),
                 dict(stage=2, seed=137, n_init=480, rounds=7, batch_size=18))
ACCEPT_TIMEOUT_S = 120.0
#: Tope por celda e intento (D-391, 17.1-03): 240 s en las aceptaciones de 5
#: y 25 kg/s para que una celda no se bloquee por reloj y no por física. El
#: par de 8 kg/s ya aceptado conserva el tope de 17.1-02. Recetas, semillas,
#: escalada y criterios NO cambian.
ACCEPT_TIMEOUTS_BY_MASS = {"5": 240.0, "8": ACCEPT_TIMEOUT_S, "25": 240.0}
ACCEPT_TIMEOUT_POLICY = ("D-391: 240 s por celda e intento en 5 y 25 kg/s; "
                         "8 kg/s conserva 120 s (17.1-02)")
#: Rutas de SOPORTE: arnés, pruebas, informe, evidencia y generador C#. No
#: forman parte de la física Python que ejecuta la matriz, así que una
#: revisión que sólo las cambia conserva el `model_source_sha` de la matriz.
SUPPORT_PATHS = frozenset({
    "validation/scale_acceptance.json", "validation/scale_l0.py",
    "validation/test_scale_l0.py", "validation/SCALE-L0-RESULTS.md",
    "test_phycb.py", "CombustorDesigner.sln"})
SUPPORT_PREFIXES = ("CombustorDesigner/", "CombustorDesigner.Example/")
T04_TARGET_K = 1200.0
T04_CAP_K = 1250.0
TOL_MDOT_KGS = 2e-9
G_BLOCKS = (("aero", 12), ("spec", 4), ("struct", 5), ("dfam", 5))
G_SPEC_NAMES = ("spec_dp_max", "spec_PF_max", "spec_envelope_D",
                "spec_envelope_L")
G_STRUCT_NAMES = ("T_wall", "sigma_th", "hoop", "t_wall", "banned")
DFAM_NEUTRAL = [-1.0] * 5
SUMMARY_FIELDS_2 = ("model_source_sha", "pilot_evidence_commit",
                    "pilot_evidence_sha256", "pilot_acceptance_count")
EVIDENCE_REL = "validation/scale_acceptance.json"


def _strict_json(data: bytes):
    def _bad(tok):
        raise ValueError(f"constante JSON no estricta: {tok}")
    return json.loads(data.decode("utf-8"), parse_constant=_bad)


def _r9(x) -> float:
    return float(round(float(x), 9))


def _g_names(dc) -> list[str]:
    import physics_core as pc
    return (list(pc.CONSTRAINT_NAMES) + list(G_SPEC_NAMES)
            + list(G_STRUCT_NAMES) + list(dc.G_NAMES))


def expected_spec(key: str, arch: str) -> dict:
    """Spec efectivo que la receta exige a la CLI para (caudal, arquitectura)."""
    row = load_inlets()["inlets"][key]
    r = ACCEPT_RECIPE
    return dict(P3_Pa=float(row["P03_Pa"]), T3_K=float(row["T03_K"]),
                mdot_kgs=float(row["mdot_kg_s"]), T4_K=r["t4"], fuel=r["fuel"],
                dp_max=r["dp_max"], arch=arch, material=r["material"],
                material_casing=r["material_casing"], PF_max=r["pf_max"],
                envelope_D_mm=None, envelope_L_mm=None, dfam=True,
                fixed_vars={})


def _real_dfam():
    """`dfam_checker` REAL del checkout original (o None)."""
    try:
        _pc, no, _gg, dc = _import_cb()
    except Exception:
        return None
    if Path(dc.__file__).resolve().parent != CB_ROOT.resolve():
        return None
    if no._dfam_module() is None:
        return None
    return dc


def rebuild_exported_contract(theta, rec: dict, spec) -> dict:
    """Contrato exportado reconstruido como lo escribe la CLI
    (`emit_deliverables` → `geometry_generator.generate`): estructura del
    spec, dos pasadas del checker DfAM y JSON limpio."""
    _pc, _no, gg, dc = _import_cb()
    import structures_core as sc
    t = np.asarray(theta, dtype=float)
    structural = sc.evaluate_structural(t, rec, material=spec.material,
                                        material_casing=spec.material_casing)
    c = gg.build_contract(t, rec, structural, None)
    for _ in range(2):
        res = dc.check(c, build_orientation="z+", voxel_mm=gg.VOXEL_DEFAULT_MM)
        gg.apply_dfam(c, res)
    return gg._clean(c)


def export_candidate_contract(theta, spec_kw: dict, outdir: Path) -> bytes:
    """Escribe el paquete de geometría como la CLI y devuelve los bytes de
    `combustor.json` (lo usan las pruebas y `verify-candidate`)."""
    pc, no, gg, _dc = _import_cb()
    import structures_core as sc
    t = np.asarray(theta, dtype=float)
    spec = no.DesignSpec(**spec_kw)
    rec = pc.evaluate(t, fidelity=pc.Fidelity.L0, use_cache=False,
                      calibrate=False)
    st = sc.evaluate_structural(t, rec, material=spec.material,
                                material_casing=spec.material_casing)
    geo = Path(outdir) / "geometry"
    gg.generate(t, rec, str(geo), structural=st,
                voxel_mm=gg.VOXEL_DEFAULT_MM, step=False, log=lambda *a: None)
    return (geo / "combustor.json").read_bytes()


def raw_pf_guard(theta, rec: dict, contract: dict | None, pf_max: float,
                 chain_fn=None) -> tuple:
    """PF CRUDO de la cadena L0 (antes de `_record_from_chain`/`_safe`).

    Usa los mismos θ, gates, material y Δp/p que la evaluación L0 por
    defecto y `use_cantera=False`. Exige PF presente, real, finito y no
    negativo, igual al del record (y al del contrato a 9 decimales) y no
    mayor que el tope del spec. Un PF sanitizado a 0 en el record publicado
    NO lo salva: el control mira la cadena. El arreglo de producción de PF
    queda para 18-02; esto es un control del arnés."""
    pc = _import_cb()[0]
    t = np.asarray(theta, dtype=float)
    fn = chain_fn or pc._l0_chain
    try:
        ch = fn(t, pc.decode_gates(t), pc.coeffs()["dp_over_p"],
                pc.MATERIAL_DEFAULT, False)
        raw = ch["pf"]["PF"]
    except Exception as exc:
        return None, [f"raw_pf_missing: PF ausente en la cadena L0 "
                      f"({type(exc).__name__}: {exc})"]
    if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, float, np.integer, np.floating)):
        return None, [f"raw_pf_type: PF de la cadena no es real ({raw!r})"]
    raw = float(raw)
    if not math.isfinite(raw):
        return None, [f"raw_pf_nonfinite: PF de la cadena = {raw!r}"]
    if raw < 0.0:
        return None, [f"raw_pf_negative: PF de la cadena = {raw!r}"]
    fails = []
    rpf = rec.get("PF") if isinstance(rec, dict) else None
    if not _finite(rpf) or abs(raw - float(rpf)) > 1e-12 * max(1.0, abs(raw)):
        fails.append(f"raw_pf_record: PF crudo {raw!r} ≠ record {rpf!r}")
    if contract is not None:
        cpf = (contract.get("derived") or {}).get("PF")
        if not _finite(cpf) or abs(_r9(raw) - float(cpf)) > 1e-12:
            fails.append(f"raw_pf_contract: PF crudo {raw!r} ≠ contrato "
                         f"{cpf!r}")
    if raw > float(pf_max):
        fails.append(f"raw_pf_cap: PF crudo {raw:.6f} > tope {pf_max}")
    return raw, fails


def verify_scale_candidate(theta, spec_kw: dict, contract_bytes, *,
                           expect: dict | None = None,
                           contract_sha256: str | None = None,
                           constraints_fn=None, chain_fn=None,
                           dfam_fn=None) -> dict:
    """Verificación INDEPENDIENTE de un candidato de la clase de escala.

    Reevalúa θ con `physics_core.evaluate` (L0, sin caché ni calibración) y
    `DesignSpec.constraints` con el spec completo; exige los cuatro bloques
    12/4/5/5 de g finitos y ≤ 0, el checker DfAM real sobre el contrato
    exportado, la identidad de clase, revisión, τ, fidelidad y
    arquitectura, el contrato reproducible, T04/P04 y A7 y el PF crudo.
    Devuelve `{ok, failures, ...}`; nunca lanza por un candidato malo.
    `constraints_fn`, `chain_fn` y `dfam_fn` sólo existen para los dobles
    de las pruebas focalizadas."""
    pc, no, gg, _dc = _import_cb()
    fails: list[str] = []
    out: dict = dict(ok=False, failures=fails)
    expect = dict(expect or {})

    contract = None
    if not contract_bytes:
        fails.append("contract_missing: no hay contrato exportado")
    else:
        sha = sha256_bytes(contract_bytes)
        out["contract_sha256"] = sha
        if contract_sha256 is not None and sha != contract_sha256:
            fails.append(f"contract_hash: {sha} ≠ {contract_sha256}")
        try:
            contract = _strict_json(contract_bytes)
        except Exception as exc:
            fails.append(f"contract_json: {type(exc).__name__}: {exc}")

    if (not isinstance(theta, (list, tuple, np.ndarray))
            or len(theta) != pc.NDIM
            or not all(_finite(x if isinstance(x, (bool, int, float, str))
                               else float(x)) for x in theta)):
        fails.append("theta_invalid: θ ausente, corto o no finito")
        return out
    t = np.asarray(theta, dtype=float)
    out["theta"] = [float(x) for x in t]
    try:
        spec = no.DesignSpec(**dict(spec_kw))
    except Exception as exc:
        fails.append(f"spec_invalid: {type(exc).__name__}: {exc}")
        return out
    r = ACCEPT_RECIPE
    if not spec.dfam:
        fails.append("no_dfam: la aceptación exige DfAM activo (D-379)")
    for name, want in (("material", r["material"]),
                       ("material_casing", r["material_casing"]),
                       ("fuel", r["fuel"]), ("dp_max", r["dp_max"]),
                       ("PF_max", r["pf_max"]), ("T4_K", r["t4"]),
                       ("envelope_D_mm", None), ("envelope_L_mm", None),
                       ("fixed_vars", {})):
        if getattr(spec, name) != want:
            fails.append(f"spec_recipe: {name}={getattr(spec, name)!r} "
                         f"(receta {want!r})")
    op = spec.operating
    if op is None or op["class_id"] != pc.OPERATING_CLASS_SCALE:
        fails.append(f"class: el spec no es de la clase "
                     f"{pc.OPERATING_CLASS_SCALE}")
    arch = pc.decode_gates(t)["arch"]
    out["arch"] = arch
    if spec.arch is None:
        fails.append("arch_not_fixed: la celda exige arquitectura explícita")
    if spec.arch is not None and arch != spec.arch:
        fails.append(f"theta_arch: θ trae {arch}, el spec {spec.arch}")
    if expect.get("arch") and (arch != expect["arch"]
                               or spec.arch != expect["arch"]):
        fails.append(f"theta_arch: se esperaba {expect['arch']}")
    point = (spec.P3_Pa, spec.T3_K, spec.mdot_kgs, spec.T4_K)
    if tuple(float(x) for x in t[16:20]) != point:
        fails.append("theta_point: el punto de operación de θ no es el del "
                     "spec")
    for k, idx in (("P03_Pa", 16), ("T03_K", 17), ("mdot_kg_s", 18)):
        if k in expect and float(t[idx]) != float(expect[k]):
            fails.append(f"theta_point: θ[{idx}]={t[idx]!r} ≠ descarga "
                         f"{expect[k]!r}")
    if not np.array_equal(spec.fix_operating_point(t), t):
        fails.append("theta_point: θ no es un punto fijo del spec "
                     "(combustible o fixed_vars)")

    rec = pc.evaluate(t, fidelity=pc.Fidelity.L0, use_cache=False,
                      calibrate=False)
    src = str(rec.get("source", ""))
    if ("degenerate" in src or rec.get("fidelity_achieved") != "L0"
            or rec.get("fidelity_requested") != "L0"
            or rec.get("fidelity_degradation") is not None):
        fails.append(f"fidelity: record {src!r} no es L0 alcanzado")
    if rec.get("operating_class") != pc.OPERATING_CLASS_SCALE:
        fails.append(f"class: record {rec.get('operating_class')!r}")
    if rec.get("scale_policy_rev") != pc.SCALE_POLICY_REV:
        fails.append(f"policy_rev: record {rec.get('scale_policy_rev')!r}")
    if rec.get("tau_max_ms_applied") != pc.SCALE_TAU_MAX_MS:
        fails.append(f"tau_limit: record {rec.get('tau_max_ms_applied')!r}")

    # --- g: cuatro bloques 12/4/5/5 finitos y ≤ 0 ---------------------------
    try:
        g = (constraints_fn or spec.constraints)(rec, t)
        g = [float(x) for x in g]
    except Exception as exc:
        g = None
        fails.append(f"g_missing: {type(exc).__name__}: {exc}")
    dcm = dfam_fn() if dfam_fn is not None else _real_dfam()
    names = _g_names(dcm or _dc)
    if g is not None:
        if len(g) != sum(n for _b, n in G_BLOCKS):
            fails.append(f"g_missing: {len(g)} valores (se exigen 12/4/5/5)")
        elif not all(math.isfinite(x) for x in g):
            fails.append("g_nonfinite: g con NaN/inf")
        else:
            pos = [f"{n}={v:+.4g}" for n, v in zip(names, g) if v > 0.0]
            if pos:
                fails.append("g_positive: " + ", ".join(pos))
            i = 0
            blocks = {}
            for b, n in G_BLOCKS:
                blocks[b] = dict(zip(names[i:i + n], g[i:i + n]))
                i += n
            out["g_blocks"] = blocks
        out["g"] = g

    # --- checker DfAM real sobre el contrato del lazo y el exportado -------
    if dcm is None:
        fails.append("dfam_checker_missing: sin checker DfAM real")
    else:
        try:
            inloop = [float(x) for x in dcm.check(
                gg.build_contract(t, rec, None, None))["g_dfam"]]
        except Exception as exc:
            inloop = None
            fails.append(f"dfam_checker: {type(exc).__name__}: {exc}")
        if inloop is not None:
            if inloop == DFAM_NEUTRAL:
                fails.append("dfam_checker_neutral: g_dfam neutro")
            if g is not None and len(g) == 26 and g[21:26] != inloop:
                fails.append("dfam_inloop_mismatch: el bloque DfAM de g no "
                             "es el del checker")
        if contract is not None:
            try:
                ex = dcm.check(contract)
                gx = [float(x) for x in ex["g_dfam"]]
                out["dfam_exported"] = dict(g_dfam=gx,
                                            feasible=bool(ex["feasible"]),
                                            violations=list(
                                                ex.get("violations") or []))
                if (len(gx) != 5 or not all(math.isfinite(x) for x in gx)
                        or any(x > 0.0 for x in gx) or not ex["feasible"]):
                    fails.append(f"dfam_exported: contrato exportado no "
                                 f"imprimible {gx}")
                if gx == DFAM_NEUTRAL:
                    fails.append("dfam_checker_neutral: g_dfam neutro en el "
                                 "contrato exportado")
            except Exception as exc:
                fails.append(f"dfam_exported: {type(exc).__name__}: {exc}")

    # --- contrato ------------------------------------------------------------
    if contract is not None:
        _check_contract(contract, t, rec, spec, arch, fails, out)

    # --- PF crudo ------------------------------------------------------------
    raw, pf_fails = raw_pf_guard(t, rec, contract, spec.PF_max,
                                 chain_fn=chain_fn)
    out["raw_pf"] = raw
    fails += pf_fails
    out["record"] = {k: rec.get(k) for k in (
        "eta_comb", "dp_over_p", "PF", "T_wall_max_K", "EI_NOx", "tau_ms",
        "phi_pz", "LBO_margin", "D_casing_mm", "L_tot_mm", "D_L_mm",
        "h_L_mm", "T04_K", "P04_Pa", "mdot_air_kgs", "mdot_fuel_kgs",
        "mdot4_kgs", "source", "operating_class", "scale_policy_rev",
        "tau_max_ms_applied", "fidelity_achieved")}
    out["ok"] = not fails
    return out


def _check_contract(c: dict, t, rec: dict, spec, arch: str, fails: list,
                    out: dict) -> None:
    pc, _no, gg, _dc = _import_cb()
    if c.get("schema") != gg.SCHEMA:
        fails.append(f"contract_schema: {c.get('schema')!r}")
    if c.get("arch") != arch:
        fails.append(f"contract_arch: {c.get('arch')!r} ≠ {arch}")
    dv = c.get("design_vector") or {}
    if any(dv.get(n) != _r9(x) for n, x in zip(pc.VAR_NAMES, t)):
        fails.append("contract_theta: design_vector no es el θ verificado")
    if c.get("operating_class") != pc.OPERATING_CLASS_SCALE:
        fails.append(f"contract_class: {c.get('operating_class')!r}")
    if c.get("scale_policy_revision") != pc.SCALE_POLICY_REV:
        fails.append(f"contract_policy_rev: "
                     f"{c.get('scale_policy_revision')!r}")
    if c.get("tau_max_ms") != pc.SCALE_TAU_MAX_MS:
        fails.append(f"contract_tau: {c.get('tau_max_ms')!r}")
    d = c.get("derived") or {}
    src = str(d.get("source", ""))
    if ((c.get("run_meta") or {}).get("fidelity") != "L0" or "→" in src
            or "degenerate" in src or src != rec.get("source")):
        fails.append(f"contract_fidelity: {src!r} / "
                     f"{(c.get('run_meta') or {}).get('fidelity')!r}")
    # T04
    T04 = d.get("T04_K")
    if not _finite(T04) or not (0.0 < T04 <= T04_CAP_K) \
            or abs(T04 - T04_TARGET_K) > 1e-6 \
            or T04 != float(round(rec.get("T04_K", float("nan")), 6)):
        fails.append(f"T04: {T04!r} (objetivo {T04_TARGET_K}, tope "
                     f"{T04_CAP_K})")
    # P04 = P03·(1 − Δp/p)
    P04, dp, P03 = d.get("P04_Pa"), d.get("dp_over_p"), float(t[16])
    if not (_finite(P04) and P04 > 0.0 and _finite(dp)):
        fails.append(f"P04: {P04!r}")
    else:
        resid = abs(P04 - P03 * (1.0 - dp))
        tol = 5e-4 + 5e-10 * P03 + 8.0 * sys.float_info.epsilon * P03
        out["P04_closure"] = dict(P03_Pa=P03, P04_Pa=P04, dp_over_p=dp,
                                  residual_Pa=resid, tol_Pa=tol)
        if resid > tol:
            fails.append(f"P04_closure: |P04 − P03(1−Δp/p)| = {resid:.3g} Pa "
                         f"> {tol:.3g}")
        if P04 != float(round(rec.get("P04_Pa", float("nan")), 3)):
            fails.append("P04: el contrato no publica el P04 del record")
    # A7: caudal total
    mo = d.get("mdot_out_kgs")
    ma, mf = d.get("mdot_air_kgs"), d.get("mdot_fuel_kgs")
    src_sum = gg.mdot_out_kgs(rec)
    if (not _finite(mo) or mo <= 0.0 or not _finite(ma) or not _finite(mf)
            or abs(mo - (ma + mf)) > TOL_MDOT_KGS or src_sum is None
            or mo != src_sum or ma != _r9(spec.mdot_kgs)):
        fails.append(f"A7: mdot_out_kgs={mo!r} aire={ma!r} comb={mf!r} "
                     f"(suma fuente {src_sum!r})")
    else:
        out["mdot_out"] = dict(mdot_out_kgs=mo, mdot_air_kgs=ma,
                               mdot_fuel_kgs=mf,
                               residual_kgs=abs(mo - (ma + mf)))
    # contrato reproducible (dimensiones, DfAM aplicado, estructura)
    try:
        rebuilt = rebuild_exported_contract(t, rec, spec)
        if json.dumps(rebuilt, sort_keys=True) != json.dumps(c, sort_keys=True):
            fails.append("contract_reproducible: el contrato exportado no es "
                         "el que reconstruye el θ verificado")
    except Exception as exc:
        fails.append(f"contract_reproducible: {type(exc).__name__}: {exc}")
    st = c.get("structural") or {}
    if not st.get("feasible"):
        fails.append("contract_structural: márgenes estructurales violados")
    out["dims"] = {k: d.get(k) for k in ("D_L_mm", "D_casing_mm", "L_tot_mm",
                                         "A_ref_mm2", "tau_ms", "dp_over_p",
                                         "PF", "eta_comb")}


def select_from_checkpoint(ck_path: Path, contract: dict) -> dict:
    """θ realmente seleccionado por la CLI, sin confiar en el código de
    salida: el design_vector del contrato exportado debe identificar UN θ
    del checkpoint (a 9 decimales) y ese θ debe ser el incumbente que da la
    regla vigente (máximo de `scalar_fitness` sobre todas las evaluaciones
    físicas, el primero en caso de empate, como `_update_best`). El load
    también valida revisión, clase, fidelidad y filas del checkpoint."""
    pc, no, _gg, _dc = _import_cb()
    payload = json.loads(Path(ck_path).read_text(encoding="utf-8"))
    d = no.AutonomousCombustorDesigner.load(str(ck_path),
                                            fidelity=pc.Fidelity.L0,
                                            log=lambda *a: None)
    best = None
    for i, r in enumerate(d.recs):
        t = np.asarray(r["_theta"], dtype=float)
        fit = no.scalar_fitness(d.spec, r, t)
        if best is None or fit > best[2]:
            best = (i, t, fit)
    dv = contract.get("design_vector") or {}
    match = {tuple(r["_theta"]) for r in d.recs
             if all(dv.get(n) == _r9(x)
                    for n, x in zip(pc.VAR_NAMES, r["_theta"]))}
    if len(match) != 1:
        raise ScaleError(f"identidad de θ: el contrato casa con {len(match)} "
                         "θ distintos del checkpoint")
    if tuple(best[1].tolist()) not in match:
        raise ScaleError("regla de selección: el θ del contrato no es el "
                         "incumbente del checkpoint")
    spec_kw = {k: v for k, v in payload["spec"].items()
               if not k.startswith("_")}
    return dict(theta=[float(x) for x in best[1]], row_index=int(best[0]),
                fitness=float(best[2]), n_evals=len(d.recs), spec=spec_kw,
                rule="max scalar_fitness sobre todas las evaluaciones "
                     "físicas (primer máximo)")


def _accept_argv(row: dict, arch: str, stage: dict, outdir: Path) -> list:
    r = ACCEPT_RECIPE
    return ["phycb_cli.py", "--optimize", "--fidelity", r["fidelity"],
            "--mdot", repr(float(row["mdot_kg_s"])),
            "--p3", repr(float(row["P03_Pa"])),
            "--t3", repr(float(row["T03_K"])), "--t4", repr(r["t4"]),
            "--arch", arch, "--fuel", r["fuel"], "--material", r["material"],
            "--material-casing", r["material_casing"],
            "--dp-max", repr(r["dp_max"]), "--pf-max", repr(r["pf_max"]),
            "--seed", str(stage["seed"]), "--n-init", str(stage["n_init"]),
            "--rounds", str(stage["rounds"]),
            "--batch-size", str(stage["batch_size"]),
            "--workers", str(r["workers"]), "--no-figures", "--no-color",
            "--outdir", str(outdir)]


def accept_timeout_s(key) -> float:
    """Tope por celda e intento para el caudal `key` (D-391)."""
    return float(ACCEPT_TIMEOUTS_BY_MASS.get(str(key), ACCEPT_TIMEOUT_S))


def is_support_path(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return rel in SUPPORT_PATHS or rel.startswith(SUPPORT_PREFIXES)


def _run_cell_process(argv: list, log_path: Path,
                      timeout_s: float = ACCEPT_TIMEOUT_S) -> dict:
    t0 = time.time()
    started = _now()
    try:
        p = subprocess.run([sys.executable, "-B", "-X", "utf8", *argv],
                           cwd=str(CB_ROOT), capture_output=True,
                           timeout=timeout_s)
        rc, timed_out, raw = p.returncode, False, p.stdout + p.stderr
    except subprocess.TimeoutExpired as exc:
        rc, timed_out = None, True
        raw = (exc.stdout or b"") + (exc.stderr or b"")
    log_path.write_bytes(raw)
    return dict(rc=rc, timed_out=timed_out, timeout_s=float(timeout_s),
                started_utc=started,
                finished_utc=_now(), elapsed_s=round(time.time() - t0, 3),
                log=str(log_path), log_sha256=sha256_bytes(raw),
                interpreter=sys.executable)


def assess_cell(key: str, arch: str, stage: dict, outdir: Path,
                run: dict, row: dict) -> dict:
    """Celda a partir de una corrida: `accepted` sólo si hay contrato,
    checkpoint y selección identificables y la verificación independiente
    pasa. Timeout, archivo ausente o fallo de traza = `blocked`."""
    cell = dict(key=f"{key}/{arch}", mass=key, arch=arch,
                stage=stage["stage"], budget=dict(stage), status="blocked",
                cause=None, run=run, outdir=str(outdir),
                argv=_accept_argv(row, arch, stage, Path("<outdir>")),
                inlet=dict(key=key, mdot_kg_s=row["mdot_kg_s"],
                           P03_Pa=row["P03_Pa"], T03_K=row["T03_K"],
                           contract_sha256=row["contract"]["sha256"],
                           run_id=row["provenance"].get("run_id"),
                           ac_commit=row["provenance"].get("ac_commit")))
    if run.get("timed_out"):
        cell["cause"] = (f"timeout: la búsqueda superó "
                         f"{run.get('timeout_s', ACCEPT_TIMEOUT_S):g} s")
        return cell
    ck = Path(outdir) / "phycb_run.json"
    cpath = Path(outdir) / "geometry" / "combustor.json"
    missing = [p.name for p in (ck, cpath) if not p.exists()]
    if missing:
        cell["cause"] = f"resultado parcial: faltan {missing} (rc {run.get('rc')})"
        return cell
    cb = cpath.read_bytes()
    try:
        contract = _strict_json(cb)
        sel = select_from_checkpoint(ck, contract)
    except Exception as exc:
        cell["cause"] = f"traza: {type(exc).__name__}: {exc}"
        return cell
    want = expected_spec(key, arch)
    diff = {k: (sel["spec"].get(k), v) for k, v in want.items()
            if sel["spec"].get(k) != v}
    v = verify_scale_candidate(sel["theta"], sel["spec"], cb,
                               expect=dict(arch=arch, P03_Pa=row["P03_Pa"],
                                           T03_K=row["T03_K"],
                                           mdot_kg_s=row["mdot_kg_s"]),
                               contract_sha256=sha256_bytes(cb))
    arts = {}
    for name in ("phycb_run.json", "dataset.csv", "report.html"):
        p = Path(outdir) / name
        arts[name] = dict(path=str(p), sha256=sha256_file(p)) \
            if p.exists() else None
    cell.update(theta=sel["theta"], spec=sel["spec"], selection={
        k: sel[k] for k in ("row_index", "fitness", "n_evals", "rule")},
        contract=dict(sha256=sha256_bytes(cb), size_bytes=len(cb),
                      encoding="lzma+base64", data=encode_bytes(cb)),
        artifacts=arts, verification={k: v.get(k) for k in (
            "g_blocks", "dfam_exported", "P04_closure", "mdot_out", "dims",
            "record", "raw_pf", "arch")}, g=v.get("g"), raw_pf=v.get("raw_pf"),
        failures=list(v["failures"]))
    causes = list(v["failures"])
    if diff:
        causes.append(f"spec distinto de la receta: {diff}")
    if run.get("rc") not in (0, 3):
        causes.append(f"rc {run.get('rc')} de la CLI")
    elif run.get("rc") == 3 and v["ok"]:
        causes.append("la CLI declaró infactible (rc 3) y la verificación "
                      "no: discrepancia")
    if causes:
        cell["cause"] = "; ".join(causes)
    else:
        cell["status"] = "accepted"
    return cell


def _changed_since(sha: str) -> list[str]:
    """Rutas de MODELO que difieren entre `sha` y el árbol de trabajo. Las
    rutas de soporte (`is_support_path`: arnés, pruebas, informe, evidencia
    y generador C#) no cambian la física que ejecuta la matriz."""
    out = _git(CB_ROOT, "diff", "--name-only", sha, "--").splitlines()
    out += [ln[3:].strip() for ln in
            _git(CB_ROOT, "status", "--porcelain").splitlines() if ln.strip()]
    return sorted({p for p in out if p and not is_support_path(p)})


def matrix_model_source(doc: dict) -> str | None:
    """`model_source_sha` común de las masas ya aceptadas (una revisión
    única por matriz). Dos valores distintos son un error."""
    shas = {m.get("model_source_sha") for m in (doc.get("masses") or {})
            .values() if m.get("status") == "accepted"}
    shas.discard(None)
    if len(shas) > 1:
        raise ScaleError(f"la matriz mezcla fuentes de modelo: {sorted(shas)}")
    return next(iter(shas), None)


def source_identity(doc: dict | None = None) -> dict:
    """Fuente confirmada para la corrida: árbol limpio salvo la evidencia.

    Si la matriz ya tiene masas aceptadas, la física debe ser la de su
    `model_source_sha`: HEAD sólo puede diferir en rutas de soporte (el
    arnés con el tope D-391, por ejemplo). Se registran ambos SHA."""
    st = git_state(CB_ROOT)
    dirty = [ln for ln in _git(CB_ROOT, "status", "--porcelain").splitlines()
             if ln.strip() and ln[3:].strip() != EVIDENCE_REL]
    if dirty:
        raise ScaleError("árbol de Phy-CB con cambios sin confirmar: "
                         f"{dirty[:5]} — confirme la fuente antes de aceptar")
    base = matrix_model_source(doc or {})
    if base is None:
        return dict(model_source_sha=st["head"], harness_sha=st["head"],
                    branch=st["branch"])
    anc = subprocess.run(["git", "-C", str(CB_ROOT), "merge-base",
                          "--is-ancestor", base, st["head"]], timeout=60)
    if anc.returncode != 0:
        raise ScaleError(f"HEAD no desciende de la fuente de la matriz {base}")
    changed = _changed_since(base)
    if changed:
        raise ScaleError(
            f"la física cambió desde la fuente de la matriz {base} "
            f"({changed[:5]}): invalide los éxitos previos y vuelva a los "
            "gates de ingeniería y piloto antes de ampliar la matriz")
    return dict(model_source_sha=base, harness_sha=st["head"],
                branch=st["branch"])


def load_acceptance(path: Path = ACCEPTANCE_PATH) -> dict:
    if not Path(path).exists():
        return dict(schema=ACCEPT_SCHEMA, cells={}, attempts=[], masses={})
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_acceptance(doc: dict, path: Path = ACCEPTANCE_PATH) -> None:
    import physics_core as pc
    doc.update(schema=ACCEPT_SCHEMA, recipe=ACCEPT_RECIPE,
               stages=list(ACCEPT_STAGES), timeout_s=ACCEPT_TIMEOUT_S,
               timeouts_s=dict(ACCEPT_TIMEOUTS_BY_MASS),
               timeout_policy=ACCEPT_TIMEOUT_POLICY,
               policy_rev=pc.SCALE_POLICY_REV)
    text = json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True,
                      allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8", newline="\n")


def _pair_comparison(cells: list[dict]) -> dict:
    """Comparación D-377 a igualdad de condiciones. En L0 `annular` y
    `reverse_annular` comparten la física (sólo el generador las distingue):
    se declara si los g físicos coinciden en vez de inventar diferencias."""
    rows = {}
    for c in cells:
        rec = c["verification"]["record"]
        g = c["g"]
        rows[c["arch"]] = dict(
            dp_over_p=rec["dp_over_p"], D_casing_mm=rec["D_casing_mm"],
            L_tot_mm=rec["L_tot_mm"], T04_K=rec["T04_K"], P04_Pa=rec["P04_Pa"],
            mdot_out_kgs=c["verification"]["mdot_out"]["mdot_out_kgs"],
            eta_comb=rec["eta_comb"], PF=rec["PF"], tau_ms=rec["tau_ms"],
            g_max=max(g), g_min_margin_name=_g_names(_import_cb()[3])[
                int(np.argmax(g))],
            elapsed_s=c["run"]["elapsed_s"],
            n_evals=c["selection"]["n_evals"])
    a, b = (cells[0], cells[1])
    same_aero = a["g"][:12] == b["g"][:12]
    same_theta = a["theta"][1:] == b["theta"][1:]
    return dict(rows=rows, same_aero_g=same_aero, same_theta_except_arch=same_theta,
                note=("En L0 ambas anulares usan la misma física: los g "
                      "aerodinámicos y el θ salvo la gate coinciden; sólo "
                      "el contrato y el generador las distinguen (D-377).")
                if same_aero else "Los g aerodinámicos difieren.")


def cmd_accept(args) -> int:
    """`--mass` admite una lista (`5,25`): la identidad de la fuente se fija
    UNA vez al empezar y cada caudal se guarda al terminar su par."""
    keys = [k.strip() for k in str(args.mass).split(",") if k.strip()]
    if not keys or len(set(keys)) != len(keys):
        raise ScaleError(f"lista de caudales inválida: {args.mass!r}")
    for key in keys:
        if key not in MASS_KEYS:
            raise ScaleError(f"caudal {key} fuera de la matriz {MASS_KEYS}")
    doc = load_acceptance()
    pending = []
    rc = 0
    for key in keys:
        prev = doc.get("masses", {}).get(key)
        if prev and prev.get("status") == "accepted" and not args.force:
            changed = _changed_since(prev["model_source_sha"])
            if changed:
                raise ScaleError(f"la fuente cambió desde la aceptación de "
                                 f"{key} ({changed[:5]}): use --force para "
                                 "rehacerla")
            print(f"[accept {key}] ya aceptado sobre "
                  f"{prev['model_source_sha']}; se verifica sin relanzar ni "
                  "sobrescribir")
            rc = max(rc, _verify_masses(doc, [key]))
        else:
            pending.append(key)
    if not pending:
        return rc
    if args.force:
        doc.setdefault("masses", {})
        for key in pending:
            doc["masses"].pop(key, None)
    ident = source_identity(doc)
    if args.out_root is None:
        raise ScaleError("accept necesita --out-root para lanzar la CLI")
    for key in pending:
        rc = max(rc, _accept_one(key, args, ident, doc))
    return rc


def _accept_one(key: str, args, ident: dict, doc: dict) -> int:
    pc = _import_cb()[0]
    inlets = load_inlets()
    row = inlets["inlets"].get(key)
    if row is None:
        raise ScaleError(f"no hay descarga congelada para {key} kg/s")
    check_row(key, row)
    timeout_s = accept_timeout_s(key)
    root = Path(args.out_root) / f"accept-{key}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    print(f"[accept {key}] tope {timeout_s:g} s por celda e intento "
          f"({ACCEPT_TIMEOUT_POLICY}) · fuente de modelo "
          f"{ident['model_source_sha'][:12]} · arnés "
          f"{ident['harness_sha'][:12]} · {root}", flush=True)
    accepted = None
    for stage in ACCEPT_STAGES:
        cells = []
        for arch in ACCEPT_ARCHS:
            outdir = root / f"s{stage['stage']}-{arch}"
            run = _run_cell_process(_accept_argv(row, arch, stage, outdir),
                                    root / f"s{stage['stage']}-{arch}.log",
                                    timeout_s)
            run["timeout_policy"] = ACCEPT_TIMEOUT_POLICY
            cell = assess_cell(key, arch, stage, outdir, run, row)
            cell.update(ident)
            cell["policy_rev"] = pc.SCALE_POLICY_REV
            cells.append(cell)
            att = {k: v for k, v in cell.items() if k != "contract"}
            att["contract_sha256"] = (cell.get("contract") or {}).get("sha256")
            doc.setdefault("attempts", []).append(att)
            print(f"[accept {key}] etapa {stage['stage']} {arch}: "
                  f"{cell['status']} · {run['elapsed_s']} s"
                  + (f" · {cell['cause']}" if cell["cause"] else ""),
                  flush=True)
        if all(c["status"] == "accepted" for c in cells):
            accepted = (stage, cells)
            break
    if accepted is None:
        doc.setdefault("masses", {})[key] = dict(
            status="blocked", model_source_sha=ident["model_source_sha"],
            harness_sha=ident["harness_sha"], timeout_s=timeout_s,
            timeout_policy=ACCEPT_TIMEOUT_POLICY, finished_utc=_now(),
            cause="ninguna etapa dio las dos arquitecturas aceptadas")
        save_acceptance(doc)
        print(f"[accept {key}] BLOQUEADO · evidencia en {ACCEPTANCE_PATH}")
        return 1
    stage, cells = accepted
    for c in cells:
        doc.setdefault("cells", {})[c["key"]] = c
    doc.setdefault("masses", {})[key] = dict(
        status="accepted", stage=stage["stage"],
        model_source_sha=ident["model_source_sha"],
        harness_sha=ident["harness_sha"], timeout_s=timeout_s,
        timeout_policy=ACCEPT_TIMEOUT_POLICY, finished_utc=_now(),
        inlets_sha256=sha256_bytes(INLETS_PATH.read_bytes()
                                   .replace(b"\r\n", b"\n")),
        comparison=_pair_comparison(cells))
    save_acceptance(doc)
    print(f"[accept {key}] ACEPTADO en la etapa {stage['stage']} · "
          f"{ACCEPTANCE_PATH}")
    return 0


def _verify_masses(doc: dict, masses: list[str]) -> int:
    ok = True
    inlets = load_inlets()["inlets"]
    for key in masses:
        m = (doc.get("masses") or {}).get(key)
        if not m or m.get("status") != "accepted":
            print(f"[verify {key}] sin aceptación")
            ok = False
            continue
        cells = [(doc.get("cells") or {}).get(f"{key}/{a}")
                 for a in ACCEPT_ARCHS]
        if any(c is None or c.get("status") != "accepted" for c in cells):
            print(f"[verify {key}] faltan celdas aceptadas")
            ok = False
            continue
        same = {(c["stage"], json.dumps(c["budget"], sort_keys=True),
                 c["inlet"]["contract_sha256"], c["model_source_sha"])
                for c in cells}
        if len(same) != 1:
            print(f"[verify {key}] las dos celdas no comparten etapa, receta, "
                  "descarga y fuente")
            ok = False
        if cells[0]["inlet"]["contract_sha256"] != \
                inlets[key]["contract"]["sha256"]:
            print(f"[verify {key}] la descarga no es la congelada")
            ok = False
        changed = _changed_since(cells[0]["model_source_sha"])
        if changed:
            print(f"[verify {key}] la fuente cambió desde "
                  f"{cells[0]['model_source_sha']}: {changed[:5]}")
            ok = False
        for c in cells:
            raw = decode_bytes(c["contract"]["data"])
            v = verify_scale_candidate(
                c["theta"], c["spec"], raw,
                expect=dict(arch=c["arch"], P03_Pa=c["inlet"]["P03_Pa"],
                            T03_K=c["inlet"]["T03_K"],
                            mdot_kg_s=c["inlet"]["mdot_kg_s"]),
                contract_sha256=c["contract"]["sha256"])
            same_g = v.get("g") == c["g"] and v.get("raw_pf") == c["raw_pf"]
            print(f"[verify {c['key']}] {'OK' if v['ok'] and same_g else 'FALLO'}"
                  f" · g máx {max(v.get('g') or [float('nan')]):+.4f}"
                  f" · PF crudo {v.get('raw_pf')}"
                  + ("" if v["ok"] else f" · {v['failures']}")
                  + ("" if same_g else " · g o PF distintos de los guardados"))
            ok = ok and v["ok"] and same_g
    return 0 if ok else 1


def cmd_verify(args) -> int:
    masses = [m.strip() for m in str(args.masses).split(",") if m.strip()]
    return _verify_masses(load_acceptance(), masses)


def cmd_verify_candidate(args) -> int:
    path = Path(args.candidate)
    cand = json.loads(path.read_text(encoding="utf-8"))
    if cand.get("contract_path"):
        cp = Path(cand["contract_path"])
        raw = (cp if cp.is_absolute() else path.parent / cp).read_bytes()
    elif isinstance(cand.get("contract"), dict):
        raw = json.dumps(cand["contract"], ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    else:
        raw = None
    v = verify_scale_candidate(cand.get("theta"), cand.get("spec") or {},
                               raw, expect=cand.get("expect"),
                               contract_sha256=cand.get("contract_sha256"))
    print(json.dumps(dict(ok=v["ok"], failures=v["failures"],
                          raw_pf=v.get("raw_pf"),
                          contract_sha256=v.get("contract_sha256"),
                          g_max=max(v["g"]) if v.get("g") else None),
                     indent=1, ensure_ascii=False))
    return 0 if v["ok"] else 1


def cmd_verify_handoff_2(args) -> int:
    f = parse_summary_fields(Path(args.summary).read_text(encoding="utf-8"),
                             SUMMARY_FIELDS_2)
    for k in ("model_source_sha", "pilot_evidence_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", f[k]):
            raise ScaleError(f"{k} debe ser un SHA de commit completo en "
                             "minúsculas")
        try:
            _git(CB_ROOT, "cat-file", "-e", f[k] + "^{commit}")
        except ScaleError:
            raise ScaleError(f"{k} no existe en el checkout de Phy-CB")
    if not re.fullmatch(r"[0-9a-f]{64}", f["pilot_evidence_sha256"]):
        raise ScaleError("pilot_evidence_sha256 debe ser un SHA256")
    if f["pilot_acceptance_count"] != "2":
        raise ScaleError("pilot_acceptance_count distinto de 2")
    src, ev = f["model_source_sha"], f["pilot_evidence_commit"]
    anc = subprocess.run(["git", "-C", str(CB_ROOT), "merge-base",
                          "--is-ancestor", src, ev], timeout=60)
    if anc.returncode != 0:
        raise ScaleError("model_source_sha no es antecesor de la evidencia")
    changed = [p for p in _git(CB_ROOT, "diff", "--name-only", src, ev,
                               "--").splitlines() if p and p != EVIDENCE_REL]
    if changed:
        raise ScaleError(f"el commit de evidencia cambia fuente: {changed}")

    def _show(rev: str, rel: str) -> bytes:
        p = subprocess.run(["git", "-C", str(CB_ROOT), "show",
                            f"{rev}:{rel}"], capture_output=True, timeout=60)
        if p.returncode != 0:
            raise ScaleError(f"{rel} no está en {rev}")
        return p.stdout

    blob = _show(ev, EVIDENCE_REL)
    if sha256_bytes(blob) != f["pilot_evidence_sha256"]:
        raise ScaleError("pilot_evidence_sha256 no coincide con la evidencia "
                         "del commit")
    doc = _strict_json(blob)
    inl = _strict_json(_show(src, "validation/scale_inlets.json"))
    row8 = inl["inlets"]["8"]
    cells = [c for c in (doc.get("cells") or {}).values()
             if c.get("mass") == "8" and c.get("status") == "accepted"]
    archs = sorted(c["arch"] for c in cells)
    if archs != sorted(ACCEPT_ARCHS) or len(cells) != 2:
        raise ScaleError(f"se exigen dos celdas distintas de 8 kg/s; hay "
                         f"{archs}")
    for c in cells:
        if c.get("model_source_sha") != src:
            raise ScaleError(f"{c['key']} no se ejecutó sobre "
                             "model_source_sha")
        if c["inlet"]["contract_sha256"] != row8["contract"]["sha256"]:
            raise ScaleError(f"{c['key']} no parte de la descarga congelada")
        raw = decode_bytes(c["contract"]["data"])
        if sha256_bytes(raw) != c["contract"]["sha256"]:
            raise ScaleError(f"{c['key']}: bytes de contrato alterados")
    if len({c["stage"] for c in cells}) != 1:
        raise ScaleError("las dos celdas no comparten etapa")
    print(f"[verify-handoff 2] OK · fuente {src} · evidencia {ev} · "
          f"2 celdas de 8 kg/s (etapa {cells[0]['stage']})")
    return 0


# ------------------------------------------------------- gate CAD (17.1-03)
#: Gate de geometría (D-386): el generador C#/PicoGK de este checkout se
#: compila UNA vez, se ejecuta en serie con los dos contratos reales de
#: 8 kg/s (extraídos de la matriz por SHA) y con el smoke KJ66 (T29 en modo
#: explícito con los mismos binarios), y cada salida se comprueba: STL de
#: piezas y ensamblaje, reporte DfAM de vóxel estricto, envolvente, y
#: topología por copa (copa, garganta, paso central) por puntos de
#: ocupación sobre el STL del liner a tolerancia de vóxel.
CAD_SCHEMA = "phycb.cad_runs/1"
CAD_VOXEL_MM = 0.8
CAD_BUILD_TIMEOUT_S = 180.0
CAD_GEOM_TIMEOUT_S = 180.0
CAD_T29_TIMEOUT_S = 420.0
CAD_MEM_MAX_BYTES = 18 * 1024 ** 3
CAD_BIN_REL = "CombustorDesigner.Example/bin/Debug/net9.0-windows"
CAD_LAUNCHER_REL = CAD_BIN_REL + "/CombustorDesigner.Example.dll"
CAD_LIBRARY_REL = CAD_BIN_REL + "/CombustorDesigner.dll"
CAD_BUILD_ARGV = ("dotnet", "build", "CombustorDesigner.sln", "-t:Rebuild",
                  "-c", "Debug", "--no-restore", "-nodeReuse:false",
                  "-p:UseSharedCompilation=false")
CAD_RUN_KEYS = ("kj66", "annular", "reverse_annular")
DFAM_VOXEL_SCHEMA = "phycb-dfam-voxel-1"
DFAM_VOXEL_SOURCE = "dfam_voxel(picogk)"
DFAM_VOXEL_G_NAMES = ("overhang_voxel", "trapped_powder", "drain_exits")
DFAM_DETAIL_KEYS = ("overhang", "enclosed_voids", "drainability", "params")
SCALE_CLASS = "annular_5_25"
REPORT_REL = "validation/SCALE-L0-RESULTS.md"
EVIDENCE_PATHS = frozenset({EVIDENCE_REL, REPORT_REL})


def canonical_sha256(obj) -> str:
    """SHA256 del JSON canónico: claves ordenadas, sin NaN, separadores
    sin espacios, UTF-8."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return sha256_bytes(text.encode("utf-8"))


def sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


# -- STL ------------------------------------------------------------------
def read_stl(path: Path) -> np.ndarray:
    """Triángulos (N, 3, 3) float64 de un STL BINARIO; error si el tamaño no
    casa con la cabecera o el fichero es ASCII/vacío."""
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ScaleError(f"STL truncado: {path}")
    n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    if n == 0 or len(raw) != 84 + 50 * n:
        raise ScaleError(f"STL no binario o incoherente ({n} triángulos, "
                         f"{len(raw)} B): {path}")
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tris = np.frombuffer(raw, dtype=rec, count=n, offset=84)["v"]
    return tris.astype(np.float64)


def stl_stats(tris: np.ndarray) -> dict:
    finite = bool(np.isfinite(tris).all())
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    vol = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    return dict(n_triangles=int(len(tris)), finite=finite,
                volume_mm3=abs(vol),
                bbox_mm=[float(x) for x in (*lo, *hi)])


class PointInMesh:
    """Ocupación de puntos en una malla cerrada por paridad de rayos (+X,
    +Y y, si discrepan, -X), con los triángulos indexados en rejillas 2D."""

    _JITTER = np.array([1.2345e-3, 2.3456e-3, 3.4567e-3])

    def __init__(self, tris: np.ndarray, cell_mm: float = 4.0):
        self.t = tris
        self.cell = float(cell_mm)
        self.grids = {0: self._grid(1, 2), 1: self._grid(0, 2)}

    def _grid(self, a: int, b: int):
        lo = self.t[:, :, [a, b]].min(axis=1)
        hi = self.t[:, :, [a, b]].max(axis=1)
        org = lo.min(axis=0) - self.cell
        i0 = np.floor((lo - org) / self.cell).astype(np.int64)
        i1 = np.floor((hi - org) / self.cell).astype(np.int64)
        nb = int(i1[:, 1].max()) + 2
        na_ = i1[:, 0] - i0[:, 0] + 1
        nb_ = i1[:, 1] - i0[:, 1] + 1
        cnt = na_ * nb_
        idx = np.repeat(np.arange(len(self.t)), cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        local = np.arange(int(cnt.sum())) - start
        nbr = np.repeat(nb_, cnt)
        ca = np.repeat(i0[:, 0], cnt) + local // nbr
        cb = np.repeat(i0[:, 1], cnt) + local % nbr
        key = ca * nb + cb
        order = np.argsort(key, kind="stable")
        return dict(a=a, b=b, org=org, nb=nb, keys=key[order],
                    idx=idx[order])

    def _cand(self, g, p):
        ia = int(np.floor((p[g["a"]] - g["org"][0]) / self.cell))
        ib = int(np.floor((p[g["b"]] - g["org"][1]) / self.cell))
        k = ia * g["nb"] + ib
        s = np.searchsorted(g["keys"], k, side="left")
        e = np.searchsorted(g["keys"], k, side="right")
        return g["idx"][s:e]

    def _crossings(self, p, axis: int, sign: float) -> int:
        g = self.grids[axis]
        c = self._cand(g, p)
        if len(c) == 0:
            return 0
        tr = self.t[c]
        d = np.zeros(3)
        d[axis] = sign
        e1 = tr[:, 1] - tr[:, 0]
        e2 = tr[:, 2] - tr[:, 0]
        h = np.cross(d, e2)
        a = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(a) > 1e-12
        f = np.zeros_like(a)
        f[ok] = 1.0 / a[ok]
        s = p - tr[:, 0]
        u = f * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = f * (q @ d)
        t = f * np.einsum("ij,ij->i", e2, q)
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        return int(hit.sum())

    def inside(self, p) -> bool:
        p = np.asarray(p, dtype=np.float64) + self._JITTER
        rx = self._crossings(p, 0, 1.0) % 2 == 1
        ry = self._crossings(p, 1, 1.0) % 2 == 1
        if rx == ry:
            return bool(rx)
        return bool(self._crossings(p, 0, -1.0) % 2 == 1)

    def frac_inside(self, pts) -> float:
        pts = list(pts)
        return sum(self.inside(p) for p in pts) / max(len(pts), 1)


# -- contrato → estaciones de copa (espejo de Liner.aCupStations) ----------
def _interp_profile(prof, z: float, col: int) -> float:
    zs = [r[0] for r in prof]
    vs = [r[col] for r in prof]
    return float(np.interp(z, zs, vs))


def cup_geometry(contract: dict, voxel: float) -> dict:
    """Estaciones de copa calculadas desde el contrato con las mismas reglas
    que el C# (independiente del STL que se comprueba)."""
    lin = contract["liner"]
    prof = lin["profile"]
    sw, inj = contract["swirler"], contract["injectors"]
    t = max(float(lin["t_wall_mm"]), max(0.6, 2.0 * voxel))
    z0, z1 = prof[0][0], prof[-1][0]
    dome = lin.get("dome") or {}
    zd = float(dome.get("z_mm", z0))
    sign = -1.0 if zd <= 0.5 * (z0 + z1) else 1.0
    hemi = str(dome.get("type", "")).lower().startswith("hemi")
    big_r = _interp_profile(prof, zd, 2)

    def z_out(r):
        if not hemi:
            return zd
        return zd + sign * math.sqrt(max(big_r * big_r - r * r, 0.0))

    def z_in(r):
        if not hemi:
            return zd - sign * t
        ri = max(big_r - t, 0.05)
        return zd + sign * math.sqrt(max(ri * ri - r * r, 0.0))

    ring = float(inj["ring_r_mm"])
    n = int(inj["count"])
    r_tip, r_hub = 0.5 * float(sw["d_tip_mm"]), 0.5 * float(sw["d_hub_mm"])
    r_cup = r_tip + t
    chord = float(sw["chord_mm"])
    if str(inj.get("type", "")).lower().startswith("vapor") \
            and float(inj.get("tube_d_mm") or 0) > 0:
        r_seat = 0.5 * float(inj["tube_d_mm"]) + voxel
    else:
        r_seat = 0.5 * float(inj.get("body_d_mm") or 0) + voxel
    zs = [f(r) for r in (max(ring - r_cup, 0.0), ring, ring + r_cup)
          for f in (z_out, z_in)]
    st = []
    for k in range(n):
        phi = math.radians(float(inj.get("phi0_deg", 0.0)) + 360.0 * k / n)
        zc = z_out(ring)
        st.append(dict(phi_rad=phi, x=ring * math.cos(phi),
                       y=ring * math.sin(phi), z=zc,
                       z_lo=min(min(zs), zc - 0.5 * chord) - 2 * voxel,
                       z_hi=max(max(zs), zc + 0.5 * chord) + 2 * voxel))
    return dict(stations=st, ring=ring, n=n, r_tip=r_tip, r_hub=r_hub,
                r_seat=r_seat, r_cup=r_cup, t=t, chord=chord, sign=sign,
                z_out=z_out, z_in=z_in, z_dome=zd, profile=prof,
                n_vanes=int(sw["n_vanes"]),
                a_each=float(sw["A_sw_each_mm2"]))


def topology_checks(contract: dict, pim: PointInMesh, voxel: float) -> dict:
    """Copa, garganta y paso central por puntos de ocupación sobre el STL
    del liner (la copa es monolítica con el domo). Umbrales: el anillo del
    cubo y la camisa deben estar ocupados (>= 75 %); la garganta en el plano
    de álabes y a la altura de la pared del domo, libre (>= 60 %: los
    álabes ocupan ~15 %); el domo entre copas, ocupado (la garganta es
    local); el centro de cada cubo (asiento del inyector) y el paso
    central, libres."""
    g = cup_geometry(contract, voxel)
    out, ok = [], True
    for s in g["stations"]:
        cx, cy, zc = s["x"], s["y"], s["z"]

        def ring_pts(r, z_fn, m):
            for j in range(m):
                a = 2 * math.pi * (j + 0.5) / m
                x, y = cx + r * math.cos(a), cy + r * math.sin(a)
                yield (x, y, z_fn(x, y))

        def wall_mid(x, y):
            r = math.hypot(x, y)
            return 0.5 * (g["z_out"](r) + g["z_in"](r))

        hub = pim.frac_inside(ring_pts(0.5 * (g["r_seat"] + g["r_hub"]),
                                       lambda x, y: zc, 8))
        shroud = pim.frac_inside(ring_pts(g["r_tip"] + 0.5 * g["t"],
                                          lambda x, y: zc, 12))
        throat_vanes = 1.0 - pim.frac_inside(ring_pts(
            0.5 * (g["r_hub"] + g["r_tip"]), lambda x, y: zc, 24))
        throat_wall = 1.0 - pim.frac_inside(ring_pts(
            0.5 * (g["r_hub"] + g["r_tip"]), wall_mid, 24))
        seat_open = not pim.inside((cx, cy, zc))
        mid = s["phi_rad"] + math.pi / g["n"]
        xm, ym = g["ring"] * math.cos(mid), g["ring"] * math.sin(mid)
        dome_between = pim.inside((xm, ym, wall_mid(xm, ym)))
        r_ok = (hub >= 0.75 and shroud >= 0.75 and throat_vanes >= 0.60
                and throat_wall >= 0.60 and seat_open and dome_between)
        ok = ok and r_ok
        out.append(dict(phi_deg=round(math.degrees(s["phi_rad"]), 6),
                        center_mm=[round(cx, 4), round(cy, 4), round(zc, 4)],
                        z_span_mm=[round(s["z_lo"], 4), round(s["z_hi"], 4)],
                        hub_solid=hub, shroud_solid=shroud,
                        throat_free_vane_plane=throat_vanes,
                        throat_free_dome_wall=throat_wall,
                        injector_seat_open=seat_open,
                        dome_solid_between_cups=dome_between, ok=r_ok))
    prof = g["profile"]
    zs = [s["z"] for s in g["stations"]]
    z_a = min(min(zs), prof[0][0])
    z_b = max(max(zs), prof[-1][0])
    central = []
    for k in range(12):
        z = z_a + (z_b - z_a) * (k + 0.5) / 12
        r_in = _interp_profile(prof, z, 1)
        pts = [(0.0, 0.0, z)] + [(0.5 * r_in * math.cos(a),
                                  0.5 * r_in * math.sin(a), z)
                                 for a in (0.3, 1.9, 3.5, 5.1)]
        central.append(not any(pim.inside(p) for p in pts))
    c_ok = all(central)
    return dict(ok=bool(ok and c_ok), stations=out,
                central_passage_open=c_ok, central_samples=len(central) * 5,
                n_cups=g["n"], ring_r_mm=g["ring"], r_tip_mm=g["r_tip"],
                r_hub_mm=g["r_hub"], r_seat_mm=g["r_seat"],
                shroud_t_mm=g["t"], n_vanes=g["n_vanes"],
                a_each_mm2=g["a_each"],
                a_each_from_d_mm2=math.pi * (g["r_tip"] ** 2 - g["r_hub"] ** 2),
                thresholds=dict(hub=0.75, shroud=0.75, throat=0.60))


# -- reporte DfAM de vóxel ------------------------------------------------
def validate_dfam_report(rep: dict, arch: str, voxel: float,
                         printable_parts: list) -> dict:
    """Reporte estricto: un fallo de forma es error; un g > 0 con reporte
    íntegro es un incumplimiento voxelizado REGISTRADO, no un error."""
    errs = []
    if rep.get("schema") != DFAM_VOXEL_SCHEMA:
        errs.append(f"schema {rep.get('schema')!r}")
    if rep.get("source") != DFAM_VOXEL_SOURCE:
        errs.append(f"source {rep.get('source')!r}")
    if rep.get("arch") != arch:
        errs.append(f"arch {rep.get('arch')!r} != {arch!r}")
    vx = rep.get("voxel_mm")
    if not (isinstance(vx, (int, float)) and abs(float(vx) - voxel) < 1e-6):
        errs.append(f"voxel_mm {vx!r} != {voxel}")
    g = rep.get("g_voxel")
    if not (isinstance(g, list) and len(g) == 3
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in g)):
        errs.append(f"g_voxel inválido: {g!r}")
        g = None
    if list(rep.get("g_voxel_names") or []) != list(DFAM_VOXEL_G_NAMES):
        errs.append(f"g_voxel_names {rep.get('g_voxel_names')!r}")
    det = rep.get("detail")
    if not isinstance(det, dict) or any(k not in det for k in DFAM_DETAIL_KEYS):
        errs.append(f"detail sin {list(DFAM_DETAIL_KEYS)}")
    feas = rep.get("feasible")
    viol = rep.get("violations")
    if g is not None:
        if not isinstance(feas, bool) or feas != all(x <= 0 for x in g):
            errs.append(f"feasible {feas!r} incoherente con g {g}")
        if feas is False and not (isinstance(viol, list) and viol
                                  and all(isinstance(v, str) for v in viol)):
            errs.append("incumplimientos sin violations explicadas")
    parts = {p.get("name"): p for p in rep.get("parts") or []}
    for name in printable_parts:
        p = parts.get(name)
        fr = (p or {}).get("frac_unsupported")
        if p is None:
            errs.append(f"parte {name} ausente del reporte")
        elif not (isinstance(fr, (int, float)) and math.isfinite(fr)):
            errs.append(f"frac_unsupported de {name} ausente o no finita")
    if errs:
        raise ScaleError("dfam_report.json inválido: " + "; ".join(errs))
    return dict(g_voxel=[float(x) for x in g], g_voxel_names=list(
        DFAM_VOXEL_G_NAMES), feasible=bool(feas),
        violations=list(viol or []),
        frac_unsupported={n: float(parts[n]["frac_unsupported"])
                          for n in printable_parts})


# -- procesos -------------------------------------------------------------
def _picogk_processes() -> list:
    import psutil
    out = []
    for pr in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(pr.info.get("cmdline") or [])
        except Exception:
            continue
        if "CombustorDesigner.Example" in cmd or (
                (pr.info.get("name") or "").lower().startswith("dotnet")
                and ("MSBuild" in cmd or "VBCSCompiler" in cmd)):
            out.append(dict(pid=pr.info["pid"], cmd=cmd[:200]))
    return out


def _require_no_picogk(where: str) -> None:
    live = _picogk_processes()
    if live:
        raise ScaleError(f"{where}: hay procesos PicoGK/dotnet vivos {live} "
                         "— nunca dos construcciones a la vez")


def _kill_tree(proc) -> None:
    import psutil
    try:
        root = psutil.Process(proc.pid)
        for ch in root.children(recursive=True):
            ch.kill()
        root.kill()
    except psutil.NoSuchProcess:
        pass


def run_bounded(argv: list, log_path: Path, timeout_s: float,
                mem_max: int = CAD_MEM_MAX_BYTES, env=None,
                cwd: Path = CB_ROOT) -> dict:
    """Proceso con tope de reloj y de memoria (RSS del árbol); al exceder
    cualquiera se detienen proceso y descendientes."""
    import psutil
    t0, started = time.time(), _now()
    peak, cause = 0, None
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(argv, cwd=str(cwd), stdout=log,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env)
        while True:
            try:
                rc = proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            try:
                root = psutil.Process(proc.pid)
                rss = root.memory_info().rss + sum(
                    c.memory_info().rss for c in root.children(recursive=True))
                peak = max(peak, rss)
            except psutil.Error:
                rss = 0
            if time.time() - t0 > timeout_s:
                cause = f"timeout {timeout_s:g} s"
            elif rss > mem_max:
                cause = f"memoria {rss / 2**30:.1f} GiB > {mem_max / 2**30:g}"
            if cause:
                _kill_tree(proc)
                proc.wait(timeout=30)
                rc = None
                break
    raw = Path(log_path).read_bytes()
    return dict(argv=[str(a) for a in argv], rc=rc, killed=cause,
                started_utc=started, finished_utc=_now(),
                elapsed_s=round(time.time() - t0, 3),
                peak_rss_gib=round(peak / 2 ** 30, 3), log=str(log_path),
                log_sha256=sha256_bytes(raw), timeout_s=timeout_s)


def _bin_hashes() -> dict:
    out = {}
    for k, rel in (("launcher", CAD_LAUNCHER_REL), ("library", CAD_LIBRARY_REL)):
        p = CB_ROOT / rel
        if not p.is_file():
            raise ScaleError(f"falta {rel} tras la compilación")
        out[k] = sha256_stream(p)
    return out


def generator_identity() -> dict:
    """La fuente del generador debe estar confirmada: árbol limpio (salvo la
    evidencia) y física idéntica a la de la matriz."""
    st = git_state(CB_ROOT)
    dirty = [ln for ln in _git(CB_ROOT, "status", "--porcelain").splitlines()
             if ln.strip() and ln[3:].strip() not in EVIDENCE_PATHS]
    if dirty:
        raise ScaleError(f"árbol de Phy-CB con cambios sin confirmar {dirty[:5]}"
                         " — confirme la fuente C#, el arnés y el test antes "
                         "del gate fresco")
    return dict(generator_source_sha=st["head"], branch=st["branch"])


# -- gate -----------------------------------------------------------------
def _pilot_contracts(doc: dict, dest: Path) -> dict:
    out = {}
    for arch in ACCEPT_ARCHS:
        c = (doc.get("cells") or {}).get(f"8/{arch}")
        if not c or c.get("status") != "accepted":
            raise ScaleError(f"falta la celda aceptada 8/{arch}")
        raw = decode_bytes(c["contract"]["data"])
        if sha256_bytes(raw) != c["contract"]["sha256"]:
            raise ScaleError(f"8/{arch}: bytes de contrato alterados")
        p = dest / f"contract_8_{arch}.json"
        p.write_bytes(raw)
        out[arch] = dict(path=p, raw=raw, sha256=c["contract"]["sha256"],
                         cell=f"8/{arch}")
    return out


def _part_checks(rep: dict, contract: dict, voxel: float,
                 need: list) -> tuple:
    """STL de cada parte del reporte: triángulos finitos, caja coherente con
    el reporte y con la envolvente, volumen de malla ~ volumen de vóxel."""
    errs, parts = [], []
    cas = contract.get("casing") or {}
    cprof = cas.get("profile") or [[0, 0]]
    r_env = (max(r[1] for r in cprof) + float(cas.get("t_wall_mm", 0))
             + float(cas.get("flange_w_mm", 0)) + float(cas.get("boss_h_mm", 0))
             + 4 * voxel)
    z_pad = (float(cas.get("flange_t_mm", 0)) + float(cas.get("t_wall_mm", 0))
             + float(cas.get("head_plate_t_mm", 0)) + 4 * voxel)
    z_env = (min(r[0] for r in cprof) - z_pad, max(r[0] for r in cprof) + z_pad)
    names = [p.get("name") for p in rep.get("parts") or []]
    for n in need:
        if n not in names:
            errs.append(f"falta la parte {n}")
    vols = {}
    for p in rep.get("parts") or []:
        path = Path(p["stl"])
        if not path.is_file():
            errs.append(f"STL ausente: {path}")
            continue
        tris = read_stl(path)
        s = stl_stats(tris)
        del tris
        bb_r = p.get("bbox_mm")
        e = []
        if not s["finite"] or s["n_triangles"] < 12:
            e.append("triángulos no finitos o malla vacía")
        if not (isinstance(bb_r, list) and len(bb_r) == 6) or max(
                abs(a - b) for a, b in zip(bb_r, s["bbox_mm"])) > 2 * voxel:
            e.append(f"caja STL {s['bbox_mm']} != reporte {bb_r}")
        lo, hi = s["bbox_mm"][:3], s["bbox_mm"][3:]
        if max(abs(lo[0]), abs(lo[1]), abs(hi[0]), abs(hi[1])) > r_env \
                or lo[2] < z_env[0] or hi[2] > z_env[1]:
            e.append(f"fuera de la envolvente r <= {r_env:.1f}, z en "
                     f"[{z_env[0]:.1f}, {z_env[1]:.1f}]")
        v_rep = float(p.get("volume_mm3") or 0)
        if not (v_rep > 0 and abs(s["volume_mm3"] - v_rep) <= 0.05 * v_rep):
            e.append(f"volumen de malla {s['volume_mm3']:.0f} != vóxel "
                     f"{v_rep:.0f} (5 %)")
        vols[p["name"]] = v_rep
        parts.append(dict(name=p["name"], stl=str(path),
                          stl_sha256=sha256_stream(path),
                          size_bytes=path.stat().st_size,
                          printable=p.get("printable"),
                          build_s=p.get("build_s"),
                          volume_voxel_mm3=v_rep, **{k: s[k] for k in (
                              "n_triangles", "volume_mm3", "bbox_mm")},
                          ok=not e, errors=e))
        errs += [f"{p['name']}: {x}" for x in e]
    if "Assembly" in vols:
        others = [v for k, v in vols.items() if k != "Assembly"]
        if others and not (max(others) <= vols["Assembly"]
                           <= 1.02 * sum(others)):
            errs.append("volumen del ensamblaje incoherente con sus partes")
    return parts, errs


def cad_run_geometry(key: str, contract_path: Path, contract: dict,
                     outdir: Path, bins: dict, voxel: float) -> dict:
    _require_no_picogk(f"antes de {key}")
    if _bin_hashes() != bins:
        raise ScaleError(f"{key}: binarios distintos de la compilación")
    launcher = str(CB_ROOT / CAD_LAUNCHER_REL)
    run = run_bounded(["dotnet", launcher, str(contract_path), str(outdir),
                       repr(voxel)], outdir.parent / f"{key}.log",
                      CAD_GEOM_TIMEOUT_S)
    after = _bin_hashes()
    _require_no_picogk(f"después de {key}")
    res = dict(key=key, run=run, voxel_mm=voxel, outdir=str(outdir),
               bins_unchanged=after == bins, ok=False, errors=[])
    if after != bins:
        res["errors"].append("los binarios cambiaron durante la ejecución")
    if run["rc"] != 0:
        res["errors"].append(f"rc {run['rc']} ({run['killed'] or 'fallo'})")
        return res
    rp = outdir / "dfam_report.json"
    if not rp.is_file():
        res["errors"].append("falta dfam_report.json")
        return res
    rb = rp.read_bytes()
    rep = _strict_json(rb)
    n_inj = int(contract["injectors"]["count"])
    need = ["Liner", "Casing"] + [f"Vaporizer{i + 1}" for i in range(n_inj)]
    try:
        res["dfam"] = validate_dfam_report(rep, contract["arch"], voxel, need)
    except ScaleError as exc:
        res["errors"].append(str(exc))
    parts, errs = _part_checks(rep, contract, voxel, need + ["Assembly"])
    res["parts"] = parts
    res["errors"] += errs
    res["dfam_report"] = dict(path=str(rp), sha256=sha256_bytes(rb),
                              size_bytes=len(rb), encoding="lzma+base64",
                              data=encode_bytes(rb))
    pl = outdir / "PicoGK.log"
    if pl.is_file():
        res["picogk_log"] = dict(path=str(pl), sha256=sha256_stream(pl))
    liner = next((p for p in parts if p["name"] == "Liner"), None)
    if liner is not None:
        t0 = time.time()
        pim = PointInMesh(read_stl(Path(liner["stl"])))
        res["topology"] = topology_checks(contract, pim, voxel)
        res["topology"]["elapsed_s"] = round(time.time() - t0, 1)
        del pim
        if not res["topology"]["ok"]:
            res["errors"].append("topología de copa/garganta/paso central "
                                 "incorrecta")
    res["ok"] = not res["errors"]
    return res


def cad_negative_count(raw: bytes, dest: Path, bins: dict) -> dict:
    """Contrato de escala con swirler.count != injectors.count: debe fallar
    en Validate (rc 3) ANTES de voxelizar (sin PicoGK.log ni STL)."""
    c = json.loads(raw.decode("utf-8"))
    c["swirler"]["count"] = int(c["injectors"]["count"]) - 1
    p = dest / "contract_negative_count.json"
    p.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    out = dest / "negative_out"
    _require_no_picogk("antes del ensayo negativo")
    run = run_bounded(["dotnet", str(CB_ROOT / CAD_LAUNCHER_REL), str(p),
                       str(out), repr(CAD_VOXEL_MM)],
                      dest / "negative.log", 60.0)
    log = Path(run["log"]).read_text(encoding="utf-8", errors="replace")
    voxelized = out.exists() and any(out.iterdir())
    ok = (run["rc"] == 3 and "scale cups" in log and "swirler.count" in log
          and not voxelized and _bin_hashes() == bins)
    return dict(contract_sha256=sha256_file(p), run=run, voxelized=voxelized,
                message=[ln for ln in log.splitlines() if "scale cups" in ln],
                ok=ok)


def cad_t29(bins: dict, outdir: Path) -> dict:
    env = dict(os.environ, PHYCB_T29_LAUNCHER_DLL=str(CB_ROOT / CAD_LAUNCHER_REL),
               PHYCB_T29_LAUNCHER_SHA256=bins["launcher"],
               PHYCB_T29_LIBRARY_SHA256=bins["library"],
               PHYCB_T29_OUTDIR=str(outdir), PYTHONIOENCODING="utf-8")
    _require_no_picogk("antes de T29")
    run = run_bounded([sys.executable, "-B", "-X", "utf8", "test_phycb.py",
                       "--slow", "--only", "T29"], outdir.parent / "t29.log",
                      CAD_T29_TIMEOUT_S, env=env)
    _require_no_picogk("después de T29")
    log = Path(run["log"]).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(\d+)/(\d+) checks OK\s+·\s+(\d+) SALTADOS", log)
    counts = tuple(int(x) for x in m.groups()) if m else None
    res = dict(key="kj66", run=run, voxel_mm=1.2, outdir=str(outdir),
               checks=dict(ok=counts[0], total=counts[1], skipped=counts[2])
               if counts else None, bins_unchanged=_bin_hashes() == bins,
               ok=False, errors=[])
    if counts != (4, 4, 0):
        res["errors"].append(f"T29 dio {counts}, se exige 4/4 y 0 SALTADOS")
    if run["rc"] != 0:
        res["errors"].append(f"rc {run['rc']} ({run['killed'] or 'fallo'})")
    cp = outdir / "combustor.json"
    rp = outdir / "stl" / "dfam_report.json"
    if not (cp.is_file() and rp.is_file()):
        res["errors"].append("T29 no dejó contrato y reporte en el outdir")
        return res
    craw, rb = cp.read_bytes(), rp.read_bytes()
    contract = _strict_json(craw)
    rep = _strict_json(rb)
    res["contract"] = dict(path=str(cp), sha256=sha256_bytes(craw),
                           operating_class=contract.get("operating_class"),
                           arch=contract.get("arch"))
    res["dfam_report"] = dict(path=str(rp), sha256=sha256_bytes(rb),
                              size_bytes=len(rb), encoding="lzma+base64",
                              data=encode_bytes(rb))
    n_inj = int(contract["injectors"]["count"])
    need = ["Liner", "Casing"] + [f"Vaporizer{i + 1}" for i in range(n_inj)]
    try:
        res["dfam"] = validate_dfam_report(rep, contract["arch"], 1.2, need)
    except ScaleError as exc:
        res["errors"].append(str(exc))
    parts, errs = _part_checks(rep, contract, 1.2, need + ["Assembly"])
    res["parts"] = parts
    res["errors"] += errs
    res["ok"] = not res["errors"]
    return res


def cmd_cad(args) -> int:
    ident = generator_identity()
    doc = load_acceptance()
    model = matrix_model_source(doc)
    if model is None or _changed_since(model):
        raise ScaleError("la física del checkout no es la de la matriz")
    root = Path(args.out_root) / f"cad-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    print(f"[cad] generador {ident['generator_source_sha'][:12]} · modelo "
          f"{model[:12]} · {root}", flush=True)
    contracts = _pilot_contracts(doc, root)
    _require_no_picogk("antes de compilar")
    build = run_bounded(list(CAD_BUILD_ARGV), root / "build.log",
                        CAD_BUILD_TIMEOUT_S)
    if build["rc"] != 0:
        raise ScaleError(f"la compilación falló (rc {build['rc']}, "
                         f"{build['killed']}); ver {build['log']}")
    bins = _bin_hashes()
    print(f"[cad] compilado en {build['elapsed_s']} s · launcher "
          f"{bins['launcher'][:12]} · biblioteca {bins['library'][:12]}",
          flush=True)
    runs = {}
    for arch in ACCEPT_ARCHS:
        c = contracts[arch]
        r = cad_run_geometry(arch, c["path"], json.loads(c["raw"]),
                             root / arch, bins, CAD_VOXEL_MM)
        r["contract"] = dict(path=str(c["path"]), sha256=c["sha256"],
                             cell=c["cell"], arch=arch,
                             operating_class=SCALE_CLASS)
        runs[arch] = r
        print(f"[cad] {arch}: {'OK' if r['ok'] else 'FALLO'} · "
              f"{r['run']['elapsed_s']} s · pico {r['run']['peak_rss_gib']} "
              f"GiB" + (f" · {r['errors']}" if r["errors"] else ""), flush=True)
        if not r["ok"] and r["run"]["rc"] != 0:
            break
    neg = cad_negative_count(contracts["annular"]["raw"], root, bins)
    print(f"[cad] negativo (count incoherente): "
          f"{'OK' if neg['ok'] else 'FALLO'} · rc {neg['run']['rc']}",
          flush=True)
    runs["kj66"] = cad_t29(bins, root / "kj66")
    k = runs["kj66"]
    print(f"[cad] kj66 (T29): {'OK' if k['ok'] else 'FALLO'} · "
          f"{k['checks']} · {k['run']['elapsed_s']} s"
          + (f" · {k['errors']}" if k["errors"] else ""), flush=True)
    cad_runs = dict(schema=CAD_SCHEMA, generated_utc=_now(),
                    generator_source_sha=ident["generator_source_sha"],
                    model_source_sha=model, voxel_mm=CAD_VOXEL_MM,
                    launcher=CAD_LAUNCHER_REL, library=CAD_LIBRARY_REL,
                    launcher_sha256=bins["launcher"],
                    library_sha256=bins["library"], build=build,
                    budgets=dict(build_s=CAD_BUILD_TIMEOUT_S,
                                 geometry_s=CAD_GEOM_TIMEOUT_S,
                                 t29_s=CAD_T29_TIMEOUT_S,
                                 mem_gib=CAD_MEM_MAX_BYTES / 2 ** 30),
                    out_root=str(root), runs=runs, negative_count=neg,
                    ok=(sorted(runs) == sorted(CAD_RUN_KEYS)
                        and all(r["ok"] for r in runs.values())
                        and neg["ok"]))
    ev = dict(cad_runs=cad_runs, cad_evidence_sha256=canonical_sha256(cad_runs))
    Path(args.evidence).write_text(json.dumps(
        ev, indent=1, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n", encoding="utf-8", newline="\n")
    print(f"[cad] {'OK' if cad_runs['ok'] else 'BLOQUEADO'} · evidencia "
          f"{args.evidence} · cad_evidence_sha256 {ev['cad_evidence_sha256']}")
    return 0 if cad_runs["ok"] else 1


def check_cad_runs(cad: dict, *, generator_sha=None, bins=None,
                   matrix=None) -> list:
    """Schema, hashes y los tres resultados del bloque cad_runs."""
    errs = []
    if cad.get("schema") != CAD_SCHEMA:
        errs.append(f"schema {cad.get('schema')!r}")
    runs = cad.get("runs") or {}
    if sorted(runs) != sorted(CAD_RUN_KEYS):
        errs.append(f"runs {sorted(runs)} != {sorted(CAD_RUN_KEYS)}")
    for k, r in runs.items():
        if not r.get("ok") or r.get("errors"):
            errs.append(f"{k} no pasó: {r.get('errors')}")
        d = r.get("dfam_report") or {}
        try:
            if sha256_bytes(decode_bytes(d["data"])) != d["sha256"]:
                errs.append(f"{k}: bytes del reporte as-built alterados")
        except Exception as exc:
            errs.append(f"{k}: reporte as-built irrecuperable ({exc})")
        if k != "kj66" and not (r.get("topology") or {}).get("ok"):
            errs.append(f"{k}: topología no verificada")
    if not (cad.get("negative_count") or {}).get("ok"):
        errs.append("el ensayo negativo de conteo no se rechazó antes de "
                    "voxelizar")
    if not cad.get("ok"):
        errs.append("cad_runs.ok es falso")
    if generator_sha and cad.get("generator_source_sha") != generator_sha:
        errs.append("generator_source_sha distinto")
    if bins and (cad.get("launcher_sha256"), cad.get("library_sha256")) != (
            bins.get("launcher"), bins.get("library")):
        errs.append("SHA256 de binarios distintos")
    if matrix is not None:
        for arch in ACCEPT_ARCHS:
            cell = (matrix.get("cells") or {}).get(f"8/{arch}") or {}
            got = ((runs.get(arch) or {}).get("contract") or {}).get("sha256")
            if got != (cell.get("contract") or {}).get("sha256"):
                errs.append(f"{arch}: el contrato CAD no es el de la matriz")
    return errs


# ------------------------------------------- entrega: cad_runs, D-377, informe
SUMMARY_FIELDS_3 = ("model_source_sha", "generator_source_sha",
                    "model_delivery_sha", "matrix_sha256", "report_sha256",
                    "cad_evidence_sha256", "launcher_sha256",
                    "library_sha256", "matrix_acceptance_count",
                    "cad_runs_count")
MATRIX_MASSES = ("5", "8", "25")
GENERATOR_PREFIXES = ("CombustorDesigner/", "CombustorDesigner.Example/",
                      "CombustorDesigner.sln")
TIE_G = 1e-9
TIE_MARGIN = 1e-6


def _cell_contract(c: dict) -> dict:
    return _strict_json(decode_bytes(c["contract"]["data"]))


def envelope_dims(contract: dict) -> dict:
    """Embalaje con la MISMA fórmula para ambas arquitecturas: diámetro
    exterior de la carcasa, longitud axial de la carcasa y volumen del
    cilindro envolvente."""
    cas = contract["casing"]
    prof = cas["profile"]
    d = 2.0 * (max(r[1] for r in prof) + float(cas["t_wall_mm"]))
    ln = max(r[0] for r in prof) - min(r[0] for r in prof)
    return dict(D_env_mm=d, L_env_mm=ln,
                V_env_m3=math.pi / 4.0 * d * d * ln * 1e-9,
                formula="D = 2 (r_casing + t_casing); L = z_max - z_min de "
                        "la carcasa; V = pi/4 D^2 L")


def recommend_pilot(doc: dict) -> dict:
    """D-377: regla reproducible a 8 kg/s.

    R1 pérdida de presión sólo si es una diferencia CALCULADA de forma
    independiente y mayor que su incertidumbre (en L0 Δp/p es impuesto por
    DP_OVER_P_DESIGN: empate por construcción, no ordena). R2 peor g global
    (menor gana). R3 vector de márgenes ordenado de peor a mejor, el primer
    valor que difiere en más de TIE_MARGIN decide (más negativo gana). R4
    embalaje (volumen envolvente; menor gana) para desempatar o explicar el
    compromiso."""
    names = _g_names(_import_cb()[3])
    rows = {}
    for a in ACCEPT_ARCHS:
        c = doc["cells"][f"8/{a}"]
        con = _cell_contract(c)
        rec = c["verification"]["record"]
        g = [float(x) for x in c["g"]]
        blocks, i = {}, 0
        for b, n in G_BLOCKS:
            blk = dict(zip(names[i:i + n], g[i:i + n]))
            i += n
            k = max(blk, key=blk.get)
            blocks[b] = dict(max=blk[k], max_name=k)
        rows[a] = dict(
            key=f"8/{a}", stage=c["stage"],
            contract_sha256=c["contract"]["sha256"],
            selection_row=c["selection"]["row_index"],
            theta=c["theta"], dp_over_p=rec["dp_over_p"],
            dp_over_p_kind="impuesta (DP_OVER_P_DESIGN en L0)",
            T04_K=rec["T04_K"], P04_Pa=rec["P04_Pa"],
            mdot_out_kgs=c["verification"]["mdot_out"]["mdot_out_kgs"],
            g_max=max(g), g_max_name=names[int(np.argmax(g))],
            g_sorted=sorted(g, reverse=True), blocks=blocks,
            g_dfam=g[-5:], **envelope_dims(con))
    a, b = (rows[x] for x in ACCEPT_ARCHS)
    steps = [dict(rule="R1 pérdida de presión", result="empate por "
                  "construcción: Δp/p = 0,030 impuesto en L0 en ambas; no "
                  "se usa para ordenar")]
    winner, decided = None, None
    if abs(a["g_max"] - b["g_max"]) > TIE_G:
        winner = min(rows, key=lambda k: rows[k]["g_max"])
        decided = "R2"
        steps.append(dict(rule="R2 peor g global", result=f"gana {winner}"))
    else:
        steps.append(dict(rule="R2 peor g global", result=(
            f"empate: g_max = {a['g_max']:+.6f} ({a['g_max_name']}) en ambas")))
        for j, (x, y) in enumerate(zip(a["g_sorted"], b["g_sorted"])):
            if abs(x - y) > TIE_MARGIN:
                winner = ACCEPT_ARCHS[0] if x < y else ACCEPT_ARCHS[1]
                decided = "R3"
                steps.append(dict(rule="R3 márgenes ordenados", result=(
                    f"posición {j} del vector ordenado: annular {x:+.6f} "
                    f"frente a reverse_annular {y:+.6f}; gana {winner}")))
                break
        else:
            steps.append(dict(rule="R3 márgenes ordenados", result="empate"))
    va, vb = a["V_env_m3"], b["V_env_m3"]
    pack = min(rows, key=lambda k: rows[k]["V_env_m3"])
    steps.append(dict(rule="R4 embalaje", result=(
        f"V_env annular {va:.4f} m³ frente a reverse_annular {vb:.4f} m³; "
        f"menor: {pack}")))
    if winner is None:
        winner, decided = pack, "R4"
    trade = ("sin compromiso: el embalaje confirma la regla"
             if pack == winner else
             f"compromiso: {winner} gana por {decided} pero {pack} embala "
             "en menos volumen")
    w = rows[winner]
    return dict(key=f"8/{winner}", arch=winner, decided_by=decided,
                stage=w["stage"], contract_sha256=w["contract_sha256"],
                selection_row=w["selection_row"], steps=steps,
                tradeoff=trade, rows=rows, vetable=True,
                note=("En L0 las dos anulares comparten la física: la "
                      "decisión sale de los márgenes DfAM del contrato y del "
                      "embalaje, no de una diferencia aerotérmica."))


def cmd_archive_cad(args) -> int:
    ev = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    cad = ev["cad_runs"]
    if canonical_sha256(cad) != ev.get("cad_evidence_sha256"):
        raise ScaleError("cad_evidence_sha256 no coincide con el bloque")
    doc = load_acceptance()
    errs = check_cad_runs(cad, matrix=doc)
    if errs:
        raise ScaleError(f"cad_runs inválido: {errs}")
    doc["cad_runs"] = cad
    doc["cad_evidence_sha256"] = ev["cad_evidence_sha256"]
    doc["recommendation"] = recommend_pilot(doc)
    save_acceptance(doc)
    rec = doc["recommendation"]
    print(f"[archive-cad] cad_runs archivado ({ev['cad_evidence_sha256']}) · "
          f"recomendación {rec['key']} por {rec['decided_by']}")
    return 0


# -- informe --------------------------------------------------------------
def _f(x, nd=3):
    return "—" if x is None else f"{x:,.{nd}f}".replace(",", " ")


def render_report(doc: dict) -> str:
    inl = load_inlets()["inlets"]
    cad = doc.get("cad_runs") or {}
    rec = doc.get("recommendation") or {}
    names = _g_names(_import_cb()[3])
    m8 = doc["masses"]["8"]
    L = []
    w = L.append
    w("# Phy-CB a mayor escala · resultados L0 de la matriz 5–25 kg/s")
    w("")
    w("Informe generado por `validation/scale_l0.py report` desde "
      "`validation/scale_acceptance.json` (no se edita a mano). Fase 17.1 de "
      "QSTUDIO, planes 17.1-02 y 17.1-03. **Todas las cifras son salidas de "
      "software L0 (correlaciones), no ensayos experimentales.**")
    w("")
    w("## Identidad")
    w("")
    w(f"- Fuente del modelo (física Python que ejecutó la matriz): "
      f"`{m8['model_source_sha']}`")
    hs = sorted({doc['masses'][k].get('harness_sha') or
                 doc['masses'][k]['model_source_sha'] for k in MATRIX_MASSES})
    w(f"- Arnés de las corridas: {', '.join(f'`{h}`' for h in hs)} (sólo "
      "rutas de soporte respecto a la fuente)")
    if cad:
        w(f"- Generador C#/PicoGK de los STL: `{cad['generator_source_sha']}`"
          f" · launcher `{cad['launcher_sha256']}` · biblioteca "
          f"`{cad['library_sha256']}`")
        w(f"- `cad_evidence_sha256`: `{doc.get('cad_evidence_sha256')}`")
    w(f"- Descargas Phy-AC congeladas: `validation/scale_inlets.json` "
      f"(`inlets_sha256` {m8['inlets_sha256']})")
    w(f"- Revisión de política: `{doc.get('policy_rev')}`")
    w(f"- Receta: L0, T04 objetivo {ACCEPT_RECIPE['t4']:g} K, tope "
      f"{ACCEPT_RECIPE['t4_max']:g} K, Jet-A, Inconel-625/718, Δp/p máx "
      f"{ACCEPT_RECIPE['dp_max']}, PF máx {ACCEPT_RECIPE['pf_max']}, DfAM "
      "activo, workers 1. Etapa 1 semilla 71 · 320/5/14; escalada única "
      "semilla 137 · 480/7/18.")
    w(f"- Tope por celda: {doc.get('timeout_policy', ACCEPT_TIMEOUT_POLICY)}")
    w("")
    w("## Etiquetas de las salidas")
    w("")
    w("- **T04**: impuesto por la receta (1200 K) y verificado ≤ 1250 K.")
    w("- **P04**: calculado, P04 = P03 · (1 − Δp/p), con cierre comprobado.")
    w("- **Caudal de salida (A7)**: calculado, aire + combustible.")
    w("- **Δp/p**: impuesto por el cierre de diseño L0 (`DP_OVER_P_DESIGN` = "
      "0,030); no es una pérdida calculada de forma independiente y no se "
      "usa para ordenar arquitecturas.")
    w("- **τ, η, PF, LBO, T pared**: calculados por correlaciones L0 con la "
      "base D-388 (banda de τ 5–10 ms y LBO intensiva por transferencia, "
      "vetable).")
    w("")
    w("## Matriz de aceptación (6/6)")
    w("")
    w("| Celda | Etapa | Tiempo (s) | Evals | P03 (Pa) | T03 (K) | PR AC | "
      "T04 (K) | P04 (Pa) | ṁ salida (kg/s) | Δp/p | η | PF | τ (ms) [cota] "
      "| LBO margen | g máx (nombre) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in MATRIX_MASSES:
        for a in ACCEPT_ARCHS:
            c = doc["cells"][f"{k}/{a}"]
            r = c["verification"]["record"]
            g = c["g"]
            con = _cell_contract(c)
            d = con["derived"]
            w(f"| {k}/{a} | {c['stage']} | {c['run']['elapsed_s']} | "
              f"{c['selection']['n_evals']} | {_f(c['inlet']['P03_Pa'], 2)} | "
              f"{_f(c['inlet']['T03_K'], 2)} | {inl[k]['PR']:.3f} | "
              f"{r['T04_K']:.1f} | {_f(r['P04_Pa'], 3)} | "
              f"{c['verification']['mdot_out']['mdot_out_kgs']:.9f} | "
              f"{r['dp_over_p']:.4f} | {r['eta_comb']:.5f} | {r['PF']:.5f} | "
              f"{r['tau_ms']:.3f} [{d.get('tau_min_ms_applied')}–"
              f"{d.get('tau_max_ms_applied')}] | {r['LBO_margin']:.3f} | "
              f"{max(g):+.4f} ({names[int(np.argmax(g))]}) |")
    w("")
    w("Detalle por bloque (máximo de cada bloque de g, todos ≤ 0) y "
      "dimensiones:")
    w("")
    w("| Celda | aero | spec | struct | dfam | D_L (mm) | D_casing (mm) | "
      "L_tot (mm) | h_L (mm) | T pared máx (K) | SMD (µm) / atomizador | "
      "Corona: copas × d_tip/d_hub (mm), A por copa (mm²) | Envolvente D×L "
      "(mm), V (m³) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in MATRIX_MASSES:
        for a in ACCEPT_ARCHS:
            c = doc["cells"][f"{k}/{a}"]
            r = c["verification"]["record"]
            con = _cell_contract(c)
            gb = c["verification"]["g_blocks"]
            sw, inj = con["swirler"], con["injectors"]
            env = envelope_dims(con)
            bl = " | ".join(
                f"{max(gb[b].values()):+.4f} ({max(gb[b], key=gb[b].get)})"
                for b, _n in G_BLOCKS)
            w(f"| {k}/{a} | {bl} | {r['D_L_mm']:.1f} | "
              f"{r['D_casing_mm']:.1f} | {r['L_tot_mm']:.1f} | "
              f"{r['h_L_mm']:.1f} | {r['T_wall_max_K']:.1f} | "
              f"{con['derived'].get('SMD_um')} / {inj.get('type')} | "
              f"{sw['count']} × {sw['d_tip_mm']:.2f}/{sw['d_hub_mm']:.2f}, "
              f"{sw['A_sw_each_mm2']:.1f} (anillo r {inj['ring_r_mm']:.1f}) | "
              f"{env['D_env_mm']:.1f} × {env['L_env_mm']:.1f}, "
              f"{env['V_env_m3']:.4f} |")
    w("")
    w("DfAM analítico del contrato exportado (checker real, "
      "`g_dfam` = t_min, d_hole_min, overhang, drainability, effusion_band):")
    w("")
    for k in MATRIX_MASSES:
        for a in ACCEPT_ARCHS:
            c = doc["cells"][f"{k}/{a}"]
            de = c["verification"]["dfam_exported"]
            w(f"- {k}/{a}: imprimible = {de['feasible']} · g_dfam "
              f"[{', '.join(f'{x:+.4f}' for x in de['g_dfam'])}]")
    w("")
    w("Contratos (SHA256 de los bytes archivados) y semillas:")
    w("")
    for k in MATRIX_MASSES:
        for a in ACCEPT_ARCHS:
            c = doc["cells"][f"{k}/{a}"]
            w(f"- {k}/{a}: `{c['contract']['sha256']}` · semilla "
              f"{c['budget']['seed']} · descarga AC "
              f"`{c['inlet']['contract_sha256'][:16]}…` · fila del "
              f"checkpoint {c['selection']['row_index']}")
    w("")
    w("## Comparación D-377 a 8 kg/s y recomendación")
    w("")
    if rec:
        rows = rec["rows"]
        w("| Magnitud | annular | reverse_annular |")
        w("|---|---|---|")
        for lab, key, fmt in (("Δp/p (impuesta)", "dp_over_p", "{:.4f}"),
                              ("T04 (K)", "T04_K", "{:.1f}"),
                              ("P04 (Pa)", "P04_Pa", "{:.3f}"),
                              ("ṁ salida (kg/s)", "mdot_out_kgs", "{:.9f}"),
                              ("g máx", "g_max", "{:+.6f}"),
                              ("D envolvente (mm)", "D_env_mm", "{:.1f}"),
                              ("L envolvente (mm)", "L_env_mm", "{:.1f}"),
                              ("V envolvente (m³)", "V_env_m3", "{:.4f}")):
            w(f"| {lab} | {fmt.format(rows['annular'][key])} | "
              f"{fmt.format(rows['reverse_annular'][key])} |")
        w("| g_dfam | " + " | ".join(
            "[" + ", ".join(f"{x:+.3f}" for x in rows[a]["g_dfam"]) + "]"
            for a in ACCEPT_ARCHS) + " |")
        if cad:
            w("| DfAM vóxel g (overhang, polvo, drenaje) | " + " | ".join(
                "[" + ", ".join(f"{x:+.3f}" for x in
                                cad["runs"][a]["dfam"]["g_voxel"]) + "]"
                for a in ACCEPT_ARCHS) + " |")
        w("")
        w("Regla aplicada (en orden):")
        w("")
        for s in rec["steps"]:
            w(f"- {s['rule']}: {s['result']}")
        w("")
        w(f"**Recomendación (vetable): `{rec['key']}`** (decide "
          f"{rec['decided_by']}; etapa {rec['stage']}; contrato "
          f"`{rec['contract_sha256']}`; fila {rec['selection_row']}). "
          f"{rec['tradeoff']}. {rec['note']}")
        if cad:
            va = {a: cad["runs"][a]["dfam"] for a in ACCEPT_ARCHS}
            w("")
            w("El DfAM sobre vóxeles as-built se registra como evidencia "
              "separada del gate analítico: annular imprimible = "
              f"{va['annular']['feasible']}, reverse_annular imprimible = "
              f"{va['reverse_annular']['feasible']}. No entra en la regla "
              "porque mide la geometría generada a vóxel 0,8 mm (paredes "
              "subidas a 2 vóxeles), no el diseño verificado.")
    w("")
    w("## Geometría PicoGK (D-386)")
    w("")
    if cad:
        w(f"Compilación única `{' '.join(CAD_BUILD_ARGV)}` en "
          f"{cad['build']['elapsed_s']} s; vóxel {cad['voxel_mm']} mm para "
          "los pilotos y 1,2 mm para el smoke KJ66 (T29). Ejecución en "
          "serie con tope de 180 s y 18 GiB por geometría.")
        w("")
        w("| Caso | Tiempo (s) | Pico RAM (GiB) | Piezas | DfAM vóxel g | "
          "Imprimible | Topología |")
        w("|---|---|---|---|---|---|---|")
        for key in CAD_RUN_KEYS:
            r = cad["runs"][key]
            dd = r.get("dfam") or {}
            topo = r.get("topology")
            tt = ("—" if topo is None else
                  f"{topo['n_cups']} copas OK, paso central "
                  f"{'abierto' if topo['central_passage_open'] else 'CERRADO'}")
            w(f"| {key} | {r['run']['elapsed_s']} | "
              f"{r['run']['peak_rss_gib']} | {len(r.get('parts') or [])} | "
              f"[{', '.join(f'{x:+.3f}' for x in dd.get('g_voxel', []))}] | "
              f"{dd.get('feasible')} | {tt} |")
        w("")
        for key in CAD_RUN_KEYS:
            r = cad["runs"][key]
            w(f"**{key}** · contrato `{r['contract']['sha256']}` · reporte "
              f"as-built `{r['dfam_report']['sha256']}`")
            w("")
            w("| Pieza | Volumen vóxel (mm³) | Caja (mm) | Triángulos | "
              "STL SHA256 |")
            w("|---|---|---|---|---|")
            for p in r.get("parts") or []:
                bb = ", ".join(f"{x:.1f}" for x in p["bbox_mm"])
                w(f"| {p['name']} | {_f(p['volume_voxel_mm3'], 0)} | [{bb}] | "
                  f"{p['n_triangles']} | `{p['stl_sha256'][:16]}…` |")
            if r.get("dfam", {}).get("violations"):
                w("")
                w("Incumplimientos voxelizados registrados: " + "; ".join(
                    r["dfam"]["violations"]))
            w("")
        neg = cad["negative_count"]
        w(f"Ensayo negativo: contrato de escala con `swirler.count` ≠ "
          f"`injectors.count` → rc {neg['run']['rc']} antes de voxelizar "
          f"({'; '.join(neg['message'])}).")
        w("")
        w(f"Los STL quedan en `{cad['out_root']}` (directorio de ejecución, "
          "fuera de Git), ligados por SHA256. Los tres STL de campaña G4 "
          "corresponden a las corridas finales de la fase 18.")
    w("")
    w("## Limitaciones de la evidencia L0")
    w("")
    w("- Correlaciones L0 sin CFD ni ensayo; la base de residencia y LBO es "
      "una transferencia (D-388) vetable.")
    w("- En L0 annular y reverse_annular comparten la física: misma θ salvo "
      "la gate y mismos g aerotérmicos; sólo el contrato y el generador las "
      "distinguen.")
    w("- A 8 kg/s el vaporizador conserva `n_tubes` = 6 copas; 5 y 25 kg/s "
      "usan micromix con el conteo por sector.")
    w("- El generador C# coloca cada copa con eje paralelo a Z sobre la "
      "estación del domo hemisférico; la camisa atraviesa toda la pared "
      "oblicua del domo. Los tubos del vaporizador siguen las cotas "
      "`tube_z0/z1` del contrato (plano del domo), no la estación de la "
      "copa: es la convención actual del contrato.")
    w("- El DfAM as-built usa paredes de 2 vóxeles (1,6 mm a 0,8 mm), más "
      "gruesas que el diseño verificado.")
    w("")
    w("## Reproducción")
    w("")
    w("```")
    w("python -B -X utf8 validation/scale_l0.py verify --masses 5,8,25")
    w("python -B -X utf8 validation/scale_l0.py accept --mass 5,25 "
      "--out-root DIR   # tope D-391")
    w("python -B -X utf8 validation/scale_l0.py cad --out-root DIR "
      "--evidence DIR/cad_evidence.json")
    w("python -B -X utf8 validation/scale_l0.py archive-cad --evidence "
      "DIR/cad_evidence.json")
    w("python -B -X utf8 validation/scale_l0.py report")
    w("python -B -X utf8 validation/scale_l0.py verify-handoff --wave 3 "
      "--summary 17.1-03-SUMMARY.md")
    w("```")
    w("")
    return "\n".join(L)


def cmd_report(args) -> int:
    doc = load_acceptance()
    text = render_report(doc)
    (CB_ROOT / REPORT_REL).write_text(text, encoding="utf-8", newline="\n")
    print(f"[report] {REPORT_REL} · {sha256_bytes(text.encode('utf-8'))}")
    return 0


# -- handoff de la ola 3 ----------------------------------------------------
def _show_blob(rev: str, rel: str) -> bytes:
    p = subprocess.run(["git", "-C", str(CB_ROOT), "show", f"{rev}:{rel}"],
                       capture_output=True, timeout=60)
    if p.returncode != 0:
        raise ScaleError(f"{rel} no está en {rev}")
    return p.stdout


def _is_ancestor(a: str, b: str) -> bool:
    return subprocess.run(["git", "-C", str(CB_ROOT), "merge-base",
                           "--is-ancestor", a, b], timeout=60).returncode == 0


def cmd_verify_handoff_3(args) -> int:
    f = parse_summary_fields(Path(args.summary).read_text(encoding="utf-8"),
                             SUMMARY_FIELDS_3)
    for k in ("model_source_sha", "generator_source_sha", "model_delivery_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", f[k]):
            raise ScaleError(f"{k} debe ser un SHA de commit completo")
        try:
            _git(CB_ROOT, "cat-file", "-e", f[k] + "^{commit}")
        except ScaleError:
            raise ScaleError(f"{k} no existe en el checkout de Phy-CB")
    for k in ("matrix_sha256", "report_sha256", "cad_evidence_sha256",
              "launcher_sha256", "library_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", f[k]):
            raise ScaleError(f"{k} debe ser un SHA256 en minúsculas")
    if f["matrix_acceptance_count"] != "6" or f["cad_runs_count"] != "3":
        raise ScaleError("se exigen matrix_acceptance_count 6 y "
                         "cad_runs_count 3")
    src, gen, dlv = (f["model_source_sha"], f["generator_source_sha"],
                     f["model_delivery_sha"])
    if not (_is_ancestor(src, gen) and _is_ancestor(gen, dlv)):
        raise ScaleError("se exige model_source_sha ⊑ generator_source_sha ⊑ "
                         "model_delivery_sha")
    prod = [p for p in _git(CB_ROOT, "diff", "--name-only", src, dlv,
                            "--").splitlines()
            if p and not is_support_path(p)]
    if prod:
        raise ScaleError(f"la entrega cambia física respecto a la fuente: "
                         f"{prod}")
    gen_ch = [p for p in _git(CB_ROOT, "diff", "--name-only", gen, dlv,
                              "--").splitlines()
              if p and p.startswith(GENERATOR_PREFIXES)]
    if gen_ch:
        raise ScaleError(f"el generador C# cambió tras el gate CAD: {gen_ch}")
    mblob = _show_blob(dlv, EVIDENCE_REL)
    if sha256_bytes(mblob) != f["matrix_sha256"]:
        raise ScaleError("matrix_sha256 no coincide con la matriz entregada")
    rblob = _show_blob(dlv, REPORT_REL)
    if sha256_bytes(rblob) != f["report_sha256"]:
        raise ScaleError("report_sha256 no coincide con el informe entregado")
    doc = _strict_json(mblob)
    cells = [c for c in (doc.get("cells") or {}).values()
             if c.get("status") == "accepted"]
    keys = sorted((c["mass"], c["arch"]) for c in cells)
    want = sorted((m, a) for m in MATRIX_MASSES for a in ACCEPT_ARCHS)
    if keys != want:
        raise ScaleError(f"la matriz no tiene las 6 celdas únicas: {keys}")
    if {c["model_source_sha"] for c in cells} != {src}:
        raise ScaleError("las celdas no comparten model_source_sha")
    if len({c.get("policy_rev") for c in cells}) != 1:
        raise ScaleError("las celdas no comparten revisión de política")
    if matrix_model_source(doc) != src:
        raise ScaleError("las masas aceptadas no son de model_source_sha")
    cad = doc.get("cad_runs") or {}
    if canonical_sha256(cad) != f["cad_evidence_sha256"] \
            or doc.get("cad_evidence_sha256") != f["cad_evidence_sha256"]:
        raise ScaleError("cad_evidence_sha256 no coincide con cad_runs")
    errs = check_cad_runs(cad, generator_sha=gen, matrix=doc,
                          bins=dict(launcher=f["launcher_sha256"],
                                    library=f["library_sha256"]))
    if errs:
        raise ScaleError(f"cad_runs inválido: {errs}")
    rec = doc.get("recommendation") or {}
    if rec.get("key") not in (f"8/{a}" for a in ACCEPT_ARCHS) \
            or rec.get("contract_sha256") != doc["cells"][rec["key"]][
                "contract"]["sha256"] \
            or f"`{rec['key']}`".encode("utf-8") not in rblob:
        raise ScaleError("recomendación ausente o incoherente con el informe")
    if _changed_since(src):
        raise ScaleError("la física del árbol no es la de model_source_sha: "
                         "no se puede reevaluar la matriz")
    if _verify_masses(doc, list(MATRIX_MASSES)) != 0:
        raise ScaleError("la reevaluación de la matriz entregada falló")
    print(f"[verify-handoff 3] OK · fuente {src[:12]} · generador {gen[:12]} "
          f"· entrega {dlv[:12]} · 6 celdas · 3 corridas CAD · "
          f"recomendación {rec['key']}")
    return 0


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scale_l0.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--mass", required=True)
    a.add_argument("--out-root", required=False, default=None)
    a.add_argument("--force", action="store_true")
    a.add_argument("--prior-attempt1", default=None,
                   help="outdir de un intento 1 ya ejecutado y fallido; se "
                        "verifica y se pasa al intento 2 (D-387)")
    u = sub.add_parser("archive-uat14")
    u.add_argument("--run-dir", required=True)
    u.add_argument("--campaign", default=None)
    u.add_argument("--out-root", default=None)
    d = sub.add_parser("diagnose")
    d.add_argument("--mass", required=True)
    d.add_argument("--out-root", required=True)
    d.add_argument("--rewrite", action="store_true")
    h = sub.add_parser("verify-handoff")
    h.add_argument("--wave", type=int, required=True)
    h.add_argument("--summary", required=True)
    sub.add_parser("check-inlets")
    ac = sub.add_parser("accept")
    ac.add_argument("--mass", required=True)
    ac.add_argument("--out-root", default=None)
    ac.add_argument("--force", action="store_true")
    vf = sub.add_parser("verify")
    vf.add_argument("--masses", required=True)
    vc = sub.add_parser("verify-candidate")
    vc.add_argument("--candidate", required=True)
    cd = sub.add_parser("cad")
    cd.add_argument("--out-root", required=True)
    cd.add_argument("--evidence", required=True)
    ar = sub.add_parser("archive-cad")
    ar.add_argument("--evidence", required=True)
    sub.add_parser("report")
    r = sub.add_parser("_ac-reeval")
    r.add_argument("--request", required=True)
    r.add_argument("--response", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    fn = {"acquire": cmd_acquire, "archive-uat14": cmd_archive_uat14,
          "diagnose": cmd_diagnose, "verify-handoff": cmd_verify_handoff,
          "check-inlets": cmd_check_inlets, "_ac-reeval": _ac_reeval_main,
          "accept": cmd_accept, "verify": cmd_verify,
          "verify-candidate": cmd_verify_candidate, "cad": cmd_cad,
          "archive-cad": cmd_archive_cad, "report": cmd_report}
    if args.cmd == "acquire" and args.out_root is None:
        doc = load_inlets()
        key = str(args.mass)
        if key not in doc.get("inlets", {}) or args.force:
            print("acquire necesita --out-root para lanzar Phy-AC")
            return 2
    try:
        return fn[args.cmd](args)
    except ScaleError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
