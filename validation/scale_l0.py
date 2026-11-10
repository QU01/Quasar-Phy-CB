"""Fase 17.1 (QSTUDIO) · evidencia reproducible de Phy-CB a mayor escala.

Subcomandos (ejecutar desde el checkout ORIGINAL de Phy-CB):

    python -B -X utf8 validation/scale_l0.py acquire  --mass 8 --out-root DIR
    python -B -X utf8 validation/scale_l0.py acquire  --mass 5 --out-root DIR
    python -B -X utf8 validation/scale_l0.py archive-uat14 --campaign F --run-dir D
    python -B -X utf8 validation/scale_l0.py diagnose --mass 8 --out-root DIR
    python -B -X utf8 validation/scale_l0.py check-inlets
    python -B -X utf8 validation/scale_l0.py verify-handoff --wave 1 --summary F

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
AC_TARGET_PR = {"5": 3.2, "8": 3.5}
#: Intentos predefinidos por caudal: el 2.º sólo si el 1.º falla.
AC_ATTEMPTS = (dict(seed=71, n_init=320, rounds=5, batch_size=14),
               dict(seed=137, n_init=480, rounds=7, batch_size=18))
AC_TIMEOUT_S = 150.0

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
    c = AC_COMMON
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
            and rec.get("common") == AC_COMMON)


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
    for n, att in enumerate(AC_ATTEMPTS, start=1):
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
                                   timeout=AC_TIMEOUT_S)
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
                cwd="<Phy-AC>", target_PR=AC_TARGET_PR[key], common=AC_COMMON,
                attempt=n, attempt_params=dict(att), timeout_s=AC_TIMEOUT_S,
                fidelity="L0")
            prov = dict(
                kind="acquired", repo="Phy-AC", repo_url=AC_REPO_URL,
                ac_commit=st0["head"], branch=st0["branch"],
                tree_clean_before=st0["clean"], tree_clean_after=st1["clean"],
                run_id=run_id, outdir_name=outdir.name, started_utc=started,
                finished_utc=_now(), elapsed_s=elapsed,
                n_evals=ckpt.get("n_evals"), seed=ckpt.get("seed"),
                python=sys.version.split()[0],
                ac_interpreter=Path(ac_python()).as_posix(),
                log_sha256=sha256_file(log))
            try:
                row = _row_from_ac_run(key, outdir, recipe, prov, raw, ckpt)
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


def parse_summary_fields(text: str) -> dict:
    """Campos `clave: valor` de UNA línea; una clave repetida es error."""
    found: dict = {}
    for line in text.splitlines():
        m = re.fullmatch(r"\s*-?\s*`?([a-z_0-9]+)`?\s*:\s*`?([^`\s]+)`?\s*",
                         line)
        if m and m.group(1) in SUMMARY_FIELDS:
            if m.group(1) in found and found[m.group(1)] != m.group(2):
                raise ScaleError(f"campo repetido con valores distintos: "
                                 f"{m.group(1)}")
            found[m.group(1)] = m.group(2)
    missing = [f for f in SUMMARY_FIELDS if f not in found]
    if missing:
        raise ScaleError(f"faltan campos en el SUMMARY: {missing}")
    return found


def cmd_verify_handoff(args) -> int:
    if args.wave != 1:
        raise ScaleError("sólo se define el handoff de la ola 1")
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
        if sha256_file(CB_ROOT / rel) != f[k]:
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


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scale_l0.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--mass", required=True)
    a.add_argument("--out-root", required=False, default=None)
    a.add_argument("--force", action="store_true")
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
    r = sub.add_parser("_ac-reeval")
    r.add_argument("--request", required=True)
    r.add_argument("--response", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    fn = {"acquire": cmd_acquire, "archive-uat14": cmd_archive_uat14,
          "diagnose": cmd_diagnose, "verify-handoff": cmd_verify_handoff,
          "check-inlets": cmd_check_inlets, "_ac-reeval": _ac_reeval_main}
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
