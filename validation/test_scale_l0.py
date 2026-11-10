"""Pruebas de la evidencia de escala de Phy-CB (fase 17.1 de QSTUDIO).

Ejecutar desde el checkout original de Phy-CB:

    python -B -m pytest -q -p no:cacheprovider validation/test_scale_l0.py -k inlet

Las pruebas `test_inlet_*` sólo leen `validation/scale_inlets.json`: no
necesitan QSTUDIO, el workspace ni el archivo histórico de la UAT 14.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CB_ROOT = HERE.parent
for _p in (str(HERE), str(CB_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scale_l0 as sl  # noqa: E402


@pytest.fixture(scope="module")
def inlets() -> dict:
    return json.loads(sl.INLETS_PATH.read_text(encoding="utf-8"))


def _contract(row: dict) -> dict:
    return json.loads(sl.decode_bytes(row["contract"]["data"]).decode("utf-8"))


# ------------------------------------------------------------------ inlets
def test_inlet_three_masses_unique_and_valid(inlets):
    tot = sl.check_inlets(inlets)
    assert tot["count"] == 3
    assert tot["keys"] == ["5", "8", "25"]
    assert len(set(tot["contract_sha256"].values())) == 3
    pairs = {(r["P03_Pa"], r["T03_K"]) for r in inlets["inlets"].values()}
    assert len(pairs) == 3


@pytest.mark.parametrize("key", ["5", "8", "25"])
def test_inlet_contract_hash_and_totals(inlets, key):
    row = inlets["inlets"][key]
    raw = sl.decode_bytes(row["contract"]["data"])
    assert sl.sha256_bytes(raw) == row["contract"]["sha256"]
    assert len(raw) == row["contract"]["size_bytes"]
    c = json.loads(raw.decode("utf-8"))
    bc = c["cfx_boundary_conditions"]
    # condiciones TOTALES, sin redondear, idénticas a las del contrato
    assert row["P03_Pa"] == bc["P0_out_Pa_est"]
    assert row["T03_K"] == bc["T0_out_K_est"]
    assert row["mdot_kg_s"] == bc["mdot_kg_s"] == float(key)
    assert row["P03_Pa"] != bc["P_static_out_Pa_est"]
    assert row["PR"] == c["derived"]["PR"]
    assert abs(row["P03_Pa"] - bc["P0_in_Pa"] * row["PR"]) <= 1e-9 * row["P03_Pa"]
    assert sl.PR_BAND[0] <= row["PR"] <= sl.PR_BAND[1]
    assert c["derived"]["feasible"] is True
    assert c["run_meta"]["fidelity"] == "L0"
    cand = row["candidate"]
    assert all(g <= 0.0 for g in cand["g"])
    # factibilidad con TODAS las restricciones del spec efectivo
    assert len(cand["reeval_g"]) >= len(cand["g"])
    assert all(g <= 0.0 for g in cand["reeval_g"])
    assert cand["reeval_feasible"] is True
    assert cand["reeval_PR"] == row["PR"]
    assert cand["reeval_T0_out_K"] == row["T03_K"]
    assert row["provenance"]["ac_commit"] == sl.AC_PINNED_SHA


def test_inlet_uat14_exact_historical_values(inlets):
    r = inlets["inlets"]["25"]
    assert r["P03_Pa"] == 322771.4057302644
    assert r["T03_K"] == 420.15317853003546
    assert r["PR"] == 3.185506101458321
    assert r["mdot_kg_s"] == 25.0
    assert r["contract"]["sha256"] == (
        "cff8c2f092857d6888ef2a911d8ac3a7107eb50e83624903a4ec2ab7168526ee")
    p = r["provenance"]
    assert p["kind"] == "archived_uat14"
    assert p["run_id"] == "1f9e4fa586c2"
    assert p["seed"] == 2118878624
    assert p["branch"] == "master"
    assert p["manifest_sha256"] == (
        "2eb49eaaf0dc7823e86d8b6aa32ed00ee88debece05439437be99240db05ee8f")
    # receta reproducible (objetivo PR 4, RPM 20000, 100/2/8, semilla)
    cfg = r["recipe"]["studio_config"]
    assert (cfg["pr"], cfg["rpm_max"], cfg["n_init"], cfg["rounds"],
            cfg["batch_size"], cfg["seed"]) == (4.0, 20000, 100, 2, 8,
                                                2118878624)
    assert r["recipe"]["target_PR"] == 4.0


@pytest.mark.parametrize("key", ["5", "8"])
def test_inlet_acquired_recipe_is_predefined(inlets, key):
    r = inlets["inlets"][key]
    rec = r["recipe"]
    assert rec["target_PR"] == sl.AC_TARGET_PR[key]
    assert rec["attempt_params"] in [dict(a) for a in sl.AC_ATTEMPTS]
    assert rec["attempt_params"] == dict(sl.AC_ATTEMPTS[rec["attempt"] - 1])
    assert rec["common"] == sl.ac_common(key)
    # tope del intento que pasó (D-387), no un tope fijo
    assert rec["timeout_s"] == sl.AC_TIMEOUTS_S[rec["attempt"] - 1]
    p = r["provenance"]
    assert p["kind"] == "acquired"
    assert p["tree_clean_before"] and p["tree_clean_after"]
    assert p["elapsed_s"] <= rec["timeout_s"]


def test_inlet_d390_surge_margin_exception_declared(inlets):
    """D-390: sólo 5 kg/s usa sm_flow_min 0,12, declarado y vetable."""
    r = inlets["inlets"]["5"]
    exc = sl.AC_SM_FLOW_EXCEPTION["5"]
    assert r["sm_flow_min"] == r["recipe"]["sm_flow_min"] == 0.12
    assert r["sm_flow_policy"] == r["recipe"]["sm_flow_policy"] == "D-390"
    assert r["sm_flow_exception"] == exc
    assert exc["vetable"] is True and exc["baseline_sm_flow_min"] == 0.15
    for frag in ("12,26", "15-25", "8 de 8", "único"):
        assert frag in exc["risk"]
    assert r["spec_effective"]["sm_flow_min"] == 0.12
    assert r["recipe"]["common"]["sm_flow_min"] == 0.12
    assert r["provenance"]["sm_flow_rule"] == "D-390"
    assert "--sm-flow-min" in r["recipe"]["argv"]
    i = r["recipe"]["argv"].index("--sm-flow-min")
    assert r["recipe"]["argv"][i + 1] == "0.12"
    cand = r["candidate"]
    assert cand["reeval_sm_flow_min"] == 0.12
    assert 0.12 <= cand["reeval_sm_flow"] < 0.15
    # 8 y 25 kg/s conservan 0,15 y no declaran excepción
    for k in ("8", "25"):
        o = inlets["inlets"][k]
        assert o["spec_effective"]["sm_flow_min"] == 0.15
        assert "sm_flow_policy" not in o
    assert inlets["inlets"]["8"]["recipe"]["common"] == sl.AC_COMMON
    assert sl.AC_COMMON["sm_flow_min"] == 0.15


def test_inlet_rejects_undeclared_or_foreign_sm_exception(inlets):
    bad = copy.deepcopy(inlets)
    bad["inlets"]["5"].pop("sm_flow_policy")
    with pytest.raises(sl.ScaleError, match="D-390"):
        sl.check_row("5", bad["inlets"]["5"])
    bad = copy.deepcopy(inlets)
    bad["inlets"]["8"]["spec_effective"]["sm_flow_min"] = 0.12
    with pytest.raises(sl.ScaleError, match="sm_flow_min"):
        sl.check_row("8", bad["inlets"]["8"])
    bad = copy.deepcopy(inlets)
    bad["inlets"]["25"]["sm_flow_policy"] = "D-390"
    with pytest.raises(sl.ScaleError, match="no autorizada"):
        sl.check_row("25", bad["inlets"]["25"])
    bad = copy.deepcopy(inlets)
    bad["inlets"]["5"]["candidate"]["reeval_sm_flow"] = 0.119
    with pytest.raises(sl.ScaleError, match="bombeo"):
        sl.check_row("5", bad["inlets"]["5"])


def test_inlet_rejects_altered_fixture(inlets):
    bad = copy.deepcopy(inlets)
    bad["inlets"]["8"]["P03_Pa"] = bad["inlets"]["8"]["P03_Pa"] * (1 + 1e-12)
    with pytest.raises(sl.ScaleError):
        sl.check_inlets(bad)
    bad = copy.deepcopy(inlets)
    raw = bytearray(sl.decode_bytes(bad["inlets"]["5"]["contract"]["data"]))
    i = raw.index(b'"PR"')
    raw[i + 7] = ord("9") if raw[i + 7] != ord("9") else ord("8")
    bad["inlets"]["5"]["contract"]["data"] = sl.encode_bytes(bytes(raw))
    with pytest.raises(sl.ScaleError, match="SHA256"):
        sl.check_inlets(bad)


def test_inlet_rejects_static_pressure(inlets):
    bad = copy.deepcopy(inlets)
    row = bad["inlets"]["25"]
    row["P03_Pa"] = _contract(row)["cfx_boundary_conditions"][
        "P_static_out_Pa_est"]
    with pytest.raises(sl.ScaleError, match="ESTÁTICA"):
        sl.check_row("25", row)


def test_inlet_rejects_invalid_pr(inlets):
    bad = copy.deepcopy(inlets)
    row = bad["inlets"]["8"]
    row["PR"] = 2.5
    with pytest.raises(sl.ScaleError, match="fuera"):
        sl.check_row("8", row)
    row["PR"] = float("nan")
    with pytest.raises(sl.ScaleError):
        sl.check_row("8", row)


def test_inlet_rejects_duplicates_and_wrong_keys(inlets):
    bad = copy.deepcopy(inlets)
    bad["inlets"]["5"] = copy.deepcopy(bad["inlets"]["8"])
    with pytest.raises(sl.ScaleError):
        sl.check_inlets(bad)
    bad = copy.deepcopy(inlets)
    bad["inlets"].pop("25")
    with pytest.raises(sl.ScaleError):
        sl.check_inlets(bad)


def test_inlet_rejects_infeasible_candidate(inlets):
    bad = copy.deepcopy(inlets)
    row = bad["inlets"]["8"]
    row["candidate"]["reeval_g"] = list(row["candidate"]["reeval_g"])
    row["candidate"]["reeval_g"][0] = 0.01
    with pytest.raises(sl.ScaleError, match="violación"):
        sl.check_row("8", row)
