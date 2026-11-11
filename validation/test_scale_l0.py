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


# =====================================================================
# Tarea 3 · dimensionado de corona, residencia y LBO (base D-388)
#
#   python -B -m pytest -q -p no:cacheprovider validation/test_scale_l0.py \
#       -k "sizing or lbo or residence"
#
# La base técnica, sus fuentes, bandas y sensibilidad están en
# validation/scale_basis.md («Base de escala D-388»). Las pruebas fijan
# identidades, signos y tendencias; las marcadas [CÁLCULO] reproducen
# cifras del informe con la fuente de este commit.
# =====================================================================
import math  # noqa: E402

import numpy as np  # noqa: E402

import physics_core as pc  # noqa: E402
import geometry_generator as gg  # noqa: E402
import neural_optimizer as no  # noqa: E402

NAMES = list(pc.CONSTRAINT_NAMES)
ARCHS = ("annular", "reverse_annular")


def _diag_theta() -> np.ndarray:
    d = sl._read_archived_diag()
    assert d is not None, "falta la traza diagnóstica archivada"
    return np.array([d["theta"][n] for n in pc.VAR_NAMES], dtype=float)


def _eval(theta) -> dict:
    return pc.evaluate(np.asarray(theta, float), fidelity=pc.Fidelity.L0,
                       use_cache=False, calibrate=False)


def _theta_at(key: str, arch: str, base=None) -> np.ndarray:
    row = sl.load_inlets()["inlets"][key]
    t = (_diag_theta() if base is None else base).copy()
    t[pc.IDX_ARCH] = float(pc.ARCH_NAMES.index(arch))
    t[pc.IDX_P3_PA], t[pc.IDX_T3_K] = row["P03_Pa"], row["T03_K"]
    t[pc.IDX_MDOT_KGS], t[pc.IDX_T4_K] = row["mdot_kg_s"], 1200.0
    return t


def _cases():
    out = {"KJ66": pc.THETA_REF.copy()}
    for key in ("5", "8", "25"):
        for arch in ARCHS:
            out[f"{key}/{arch}"] = _theta_at(key, arch)
    return out


CASES = list(_cases())
SCALE_CASES = [c for c in CASES if c != "KJ66"]


@pytest.fixture(scope="module")
def records():
    return {k: (t, _eval(t)) for k, t in _cases().items()}


# ------------------------------------------------------------------ sizing
def test_sizing_class_selection_policy():
    """D-373/D-374: [5, 25] kg/s con extremos y sólo anulares."""
    for m, arch, cls in ((5.0, "annular", "scale"),
                         (25.0, "reverse_annular", "scale"),
                         (8.0, "annular", "scale"),
                         (4.999, "annular", "micro"),
                         (25.001, "annular", "micro"),
                         (8.0, "can", "micro"),
                         (0.18, "reverse_annular", "micro"),
                         (float("nan"), "annular", "micro")):
        assert pc.combustor_class(m, arch) == cls, (m, arch)
    assert pc.tau_limits_ms("micro") == (pc.TAU_MIN_MS, pc.TAU_MAX_MS) \
        == (4.0, 10.0)
    assert pc.tau_limits_ms("scale") == (5.0, 10.0)


@pytest.mark.parametrize("case", CASES)
def test_sizing_zone_area_volume_continuity(records, case):
    t, r = records[case]
    assert "degenerate" not in r["source"]
    c = pc.coeffs()
    A_L = r["A_L_m2"]
    assert A_L == pytest.approx(min(max(t[1], 0.30), 0.95) * r["A_ref_m2"],
                                rel=1e-12)
    assert r["D_L_mm"] == pytest.approx(1e3 * math.sqrt(4 * A_L / math.pi),
                                        rel=1e-12)
    if case == "KJ66":
        assert r["combustor_class"] == "micro"
        assert r["length_basis"] == "equivalent_diameter"
        assert r["D_T_mm"] == r["D_L_mm"]
    else:
        assert r["combustor_class"] == "scale"
        assert r["length_basis"] == "liner_height"
        assert r["D_T_mm"] == pytest.approx(r["h_L_mm"], rel=1e-12)
        assert r["scale_policy_rev"] == pc.SCALE_POLICY_REV
    D_T = r["D_T_mm"]
    arch = pc.ARCH_NAMES[int(round(t[0]))]
    assert r["L_pz_mm"] == pytest.approx(min(max(t[12], .2), 2.) * D_T,
                                         rel=1e-12)
    assert r["L_sz_mm"] == pytest.approx(c["L_sz_over_DL"][arch] * D_T,
                                         rel=1e-12)
    assert r["L_dz_mm"] == pytest.approx(min(max(t[13], .2), 5.) * D_T,
                                         rel=1e-12)
    for z in ("pz", "sz", "dz"):
        assert r[f"V_{z}_m3"] == pytest.approx(A_L * r[f"L_{z}_mm"] * 1e-3,
                                               rel=1e-12)
    assert r["V_c_m3"] == pytest.approx(
        r["V_pz_m3"] + r["V_sz_m3"] + r["V_dz_m3"], rel=1e-12)
    assert r["L_tot_mm"] == pytest.approx(
        r["L_pz_mm"] + r["L_sz_mm"] + r["L_dz_mm"], rel=1e-12)
    for k in ("A_ref_m2", "A_L_m2", "A_an_m2", "D_L_mm", "D_casing_mm",
              "D_T_mm", "L_pz_mm", "L_sz_mm", "L_dz_mm"):
        assert r[k] > 0.0, k
    assert r["A_ref_m2"] == max(r["A_ref_aero_m2"], r["A_ref_chem_m2"])


@pytest.mark.parametrize("case", SCALE_CASES)
def test_sizing_liner_height_is_the_built_corona(records, case):
    """h_L de la física = r_out − r_in del liner que construye el generador,
    con r_out = LINER_R_OVER_RCAS·r_carcasa; el contrato declara la clase."""
    t, r = records[case]
    c = gg.build_contract(t, r, None, None)
    prof = c["liner"]["profile"]
    # estaciones de la corona recta (la inversa añade los giros)
    r_lo = pc.LINER_R_OVER_RCAS * 0.5 * r["D_casing_mm"]
    corona = [p for p in prof if abs(p[2] - r_lo) <= 1e-4]
    assert corona, "ninguna estación del liner en r_lo"
    for p in corona:
        assert p[2] - p[1] == pytest.approx(r["h_L_mm"], abs=1e-4)
    der = c["derived"]
    assert der["combustor_class"] == "scale"
    assert der["length_basis"] == "liner_height"
    assert der["D_T_mm"] == pytest.approx(r["D_T_mm"], abs=1e-5)
    assert der["tau_max_ms_applied"] == pc.SCALE_TAU_MAX_MS
    assert der["tau_min_ms_applied"] == pc.SCALE_TAU_MIN_MS
    assert der["lbo_law"] == "lefebvre_intensive_fpz"
    # el ancla micro declara su clase y su límite histórico
    rk = _eval(pc.THETA_REF)
    ck = gg.build_contract(pc.THETA_REF, rk, None, None)["derived"]
    assert (ck["combustor_class"], ck["tau_max_ms_applied"]) == ("micro", 10.0)


@pytest.mark.parametrize("case", CASES)
def test_sizing_air_balance_identity(records, case):
    _, r = records[case]
    sp = r["splits"]
    assert sum(sp.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(v >= 0.0 for v in sp.values())
    assert r["mdot_fuel_kgs"] == pytest.approx(
        r["f_fuel_air"] * r["mdot_air_kgs"], rel=1e-12)
    assert r["mdot4_kgs"] == pytest.approx(
        r["mdot_air_kgs"] + r["mdot_fuel_kgs"], rel=1e-12)


@pytest.mark.parametrize("case", ["KJ66", "8/annular", "8/reverse_annular",
                                  "25/annular"])
def test_sizing_contract_profile_and_dfam_match_record(records, case):
    t, r = records[case]
    c = gg.build_contract(t, r, None, None)
    der = c["derived"]
    assert der["D_L_mm"] == pytest.approx(r["D_L_mm"], abs=1e-6)
    assert der["L_pz_mm"] == pytest.approx(r["L_pz_mm"], abs=1e-6)
    assert der["L_tot_mm"] == pytest.approx(r["L_tot_mm"], abs=1e-6)
    assert der["tau_ms"] == pytest.approx(r["tau_ms"], abs=1e-6)
    prof = c["liner"]["profile"]
    zs = [p[0] for p in prof]
    assert zs == sorted(zs)
    r_cas = min(p[1] for p in c["casing"]["profile"])
    for p in prof:
        r_in, r_out = p[1], p[2]
        assert r_out > r_in >= 0.0            # radios positivos, sin cruce
        assert r_out + c["liner"]["t_wall_mm"] < r_cas   # holgura radial
    assert c["validation"]["profile_ok"] is True
    assert c["validation"]["self_intersects"] is False
    # el DfAM del contrato es el mismo que entra en g (checker real)
    spec = no.DesignSpec(P3_Pa=t[16], T3_K=t[17], mdot_kgs=t[18], T4_K=t[19],
                         arch=pc.ARCH_NAMES[int(round(t[0]))])
    g = spec.constraints(r, t)
    assert no._dfam_module() is not None
    assert list(g[-5:]) == pytest.approx(
        list(no._dfam_module().check(c)["g_dfam"]), abs=1e-12)


@pytest.mark.parametrize("case", ["8/annular", "8/reverse_annular",
                                  "5/annular", "25/reverse_annular"])
def test_sizing_cup_pitch_and_seats_fit_the_corona(records, case):
    """Capa 5c de 17.1-03 (D-386): copas en el radio medio de la corona,
    paso angular uniforme, asientos que no se tocan y cuerpo de copa que
    cabe en la altura de la corona (airblast del θ diagnóstico)."""
    t, r = records[case]
    c = gg.build_contract(t, r, None, None)
    inj = c["injectors"]
    r_out = pc.LINER_R_OVER_RCAS * 0.5 * r["D_casing_mm"]
    h = r["h_L_mm"]
    r_in = r_out - h
    n = inj["count"]
    expect = max(gg.N_INJ_ANNULAR_DEFAULT, round(
        2 * math.pi * 0.5 * (r_in + r_out) / (gg.CUP_PITCH_OVER_H * h)))
    assert inj["type"] == "airblast"
    assert n == expect and n == c["swirler"]["count"]
    assert r_in < inj["ring_r_mm"] < r_out
    assert inj["ring_r_mm"] == pytest.approx(0.5 * (r_in + r_out), abs=1e-4)
    pitch = 2 * math.pi * inj["ring_r_mm"] / n
    assert pitch >= gg.INJ_SPACING_FACTOR * c["swirler"]["d_tip_mm"]
    assert c["swirler"]["d_tip_mm"] < h
    # prefilmer por copa (D_T): con D_L no cabría en la corona
    assert pc.D_PREFILMER_OVER_DL * r["D_T_mm"] < h
    assert pc.D_PREFILMER_OVER_DL * r["D_L_mm"] > h


def test_sizing_reverse_annular_turns_and_packaging(records):
    t, r = records["8/reverse_annular"]
    c = gg.build_contract(t, r, None, None)
    legs = c["flow_path"]
    dirs = [leg.get("dir_z") for leg in legs]
    assert any(d is not None and d < 0 for d in dirs)   # tramo a −Z
    turns = [leg.get("turn_deg", 0.0) for leg in legs]
    assert sum(1 for x in turns if abs(x - 180.0) < 1e-9) == 2
    for leg in legs:
        if "r_in_mm" in leg and "r_out_mm" in leg:
            assert leg["r_out_mm"] > leg["r_in_mm"] >= 0.0


def test_sizing_dimensionless_ratios_are_scale_invariant(records):
    """[CÁLCULO] Fundamento del reparto de convenciones: a θ fijo, Y/D_L y
    L_DZ/D_T (factor de patrón) no cambian con ṁ; τ sí (∝ √ṁ)."""
    ys = {k: records[f"{k}/annular"][1]["Y_over_DL_dz"] for k in
          ("5", "8", "25")}
    pfs = {k: records[f"{k}/annular"][1]["PF"] for k in ("5", "8", "25")}
    assert max(ys.values()) - min(ys.values()) < 0.03
    assert max(pfs.values()) - min(pfs.values()) < 1e-9


# --------------------------------------------------------------- residence
@pytest.mark.parametrize("case", CASES)
def test_residence_reproduces_declared_definition(records, case):
    t, r = records[case]
    rho3 = t[16] / (pc.RGAS * t[17])
    assert r["rho3_kgm3"] == pytest.approx(rho3, rel=1e-12)
    assert r["tau_ms"] == pytest.approx(
        1e3 * r["V_c_m3"] * rho3 / r["mdot_air_kgs"], rel=1e-12)
    assert r["tau_pz_ms"] == pytest.approx(
        1e3 * r["V_pz_m3"] * rho3 / (r["splits"]["pz"] * r["mdot_air_kgs"]),
        rel=1e-12)
    lo, hi = pc.tau_limits_ms(r["combustor_class"])
    assert (r["tau_min_ms_applied"], r["tau_max_ms_applied"]) == (lo, hi)


@pytest.mark.parametrize("case", ["KJ66", "8/annular"])
def test_residence_g_tau_sign_on_both_sides(records, case):
    t, r = records[case]
    lo, hi = pc.tau_limits_ms(r["combustor_class"])
    i_min = NAMES.index("tau_min")
    i_max = NAMES.index("tau_max")
    for tau, s_min, s_max in ((hi * 0.999, -1, -1), (hi * 1.001, -1, +1),
                              (lo * 1.001, -1, -1), (lo * 0.999, +1, -1)):
        g = pc._constraints(dict(r, tau_ms=tau), t)
        assert math.copysign(1, g[i_min]) == s_min, (case, tau)
        assert math.copysign(1, g[i_max]) == s_max, (case, tau)


def test_residence_scaling_exponent_with_mdot():
    """[CÁLCULO] Clase de escala, rama aero, P3/T3 y θ fijos: τ ∝ ṁ^(1/2)
    también con la altura de corona (similitud homotética a HTR fijo)."""
    t = _diag_theta()
    taus = {}
    for m in (5.0, 25.0):
        tt = t.copy()
        tt[pc.IDX_MDOT_KGS] = m
        rr = _eval(tt)
        assert rr["sizing_driver"] == "aero"
        assert rr["combustor_class"] == "scale"
        taus[m] = rr["tau_ms"]
    assert math.log(taus[25.0] / taus[5.0]) / math.log(5.0) == pytest.approx(
        0.5, abs=1e-9)


def _box_corner(key, arch, side):
    t = _theta_at(key, arch)
    bounds = pc.BOUNDS_LO if side == "lo" else pc.BOUNDS_HI
    for v in ("AL_over_Aref", "L_pz_over_DL", "L_dz_over_DL"):
        t[pc.VAR_NAMES.index(v)] = bounds[pc.VAR_NAMES.index(v)]
    return t


@pytest.mark.parametrize("key", ["5", "8", "25"])
@pytest.mark.parametrize("arch", ARCHS)
def test_residence_scale_band_is_attainable_in_the_box(key, arch):
    """[CÁLCULO] Con la corona, la caja de diseño cubre la banda [5, 10] ms
    en las tres descargas y las dos anulares; con D_L (convención micro)
    el mínimo de la caja supera 10 ms (la pared que motivó D-388)."""
    lo = _eval(_box_corner(key, arch, "lo"))
    hi = _eval(_box_corner(key, arch, "hi"))
    assert lo["sizing_driver"] == "aero"
    assert lo["tau_ms"] < pc.SCALE_TAU_MAX_MS
    assert hi["tau_ms"] > pc.SCALE_TAU_MIN_MS
    # cota inferior física: la evaporación D² de 50 µm (3,1 ms) cabe bajo τ_min
    tau_evap = 1e3 * (50e-6) ** 2 / pc.FUELS["jet_a"]["lambda_evap"]
    assert tau_evap < pc.SCALE_TAU_MIN_MS
    # misma esquina con la convención micro (D_L)
    saved = pc.SCALE_MDOT_MIN_KGS
    try:
        pc.SCALE_MDOT_MIN_KGS = 1e9
        micro = _eval(_box_corner(key, arch, "lo"))
    finally:
        pc.SCALE_MDOT_MIN_KGS = saved
    assert micro["combustor_class"] == "micro"
    assert micro["tau_ms"] > pc.TAU_MAX_MS
    assert micro["tau_ms"] / lo["tau_ms"] == pytest.approx(
        micro["D_L_mm"] / lo["h_L_mm"], rel=1e-9)


def test_residence_length_factor_sensitivity():
    """[CÁLCULO] D_T = 2h (diámetro hidráulico) duplica longitudes y τ a θ
    fijo: es la sensibilidad declarada de la convención de longitud."""
    t = _theta_at("8", "annular")
    base = _eval(t)
    saved = pc.SCALE_LENGTH_FACTOR
    try:
        pc.SCALE_LENGTH_FACTOR = 2.0
        dbl = _eval(t)
    finally:
        pc.SCALE_LENGTH_FACTOR = saved
    assert dbl["length_basis"] == "liner_height_x2"
    assert dbl["D_T_mm"] == pytest.approx(2 * base["D_T_mm"], rel=1e-12)
    assert dbl["tau_ms"] == pytest.approx(2 * base["tau_ms"], rel=1e-12)
    assert dbl["L_tot_mm"] == pytest.approx(2 * base["L_tot_mm"], rel=1e-12)


# --------------------------------------------------------------------- lbo
def _stab(P3=3.5e5, T3=430.0, mdot=8.0, A_ref=0.14, V_pz=0.018,
          smd=40.0, phi_pz=0.9, f_pz=0.3, cls="scale"):
    t = pc.THETA_REF.copy()
    t[16], t[17] = P3, T3
    sz = {"rho3_kgm3": P3 / (pc.RGAS * T3), "_A_ref": A_ref, "V_pz_m3": V_pz,
          "combustor_class": cls}
    part = {"phi_pz": phi_pz, "splits": {"pz": f_pz}}
    return pc._stability(t, {"fuel": "jet_a", "arch": "annular"}, sz, part,
                         {"SMD_um": smd, "is_gas_branch": False}, mdot,
                         2300.0, 28.9)


def test_lbo_intensive_invariant_to_sector_replication():
    """N sectores idénticos (ṁ, A_ref, V_PZ ×N; f_PZ igual): misma φ_LBO."""
    one = _stab(mdot=0.5, A_ref=0.01, V_pz=0.002)
    for N in (2, 8, 18):
        rep = _stab(mdot=0.5 * N, A_ref=0.01 * N, V_pz=0.002 * N)
        assert rep["phi_LBO_raw"] == pytest.approx(one["phi_LBO_raw"],
                                                   rel=1e-12)
        assert rep["LBO_load_intensive"] == pytest.approx(
            one["LBO_load_intensive"], rel=1e-12)
    assert one["lbo_law"] == "lefebvre_intensive_fpz"


def test_lbo_micro_law_is_extensive_and_untouched():
    """La ley micro (ancla) sigue escalando ×N: por eso no se transfiere."""
    one = _stab(mdot=0.5, A_ref=0.01, V_pz=0.002, cls="micro")
    n18 = _stab(mdot=9.0, A_ref=0.18, V_pz=0.036, cls="micro")
    assert one["lbo_law"] == "lefebvre_aref_micro"
    assert n18["phi_LBO_raw"] == pytest.approx(18 * one["phi_LBO_raw"],
                                               rel=1e-9)


def test_lbo_c_int_transfer_reproduces_the_anchor():
    """C_int = C_LBO·A_ref/f_PZ en REF_CB10: ambas leyes dan la misma φ en
    el ancla (transferencia de la única calibración, sin ajuste)."""
    anchors = json.loads((HERE / "anchors.json").read_text(encoding="utf-8"))
    ref = anchors["anchors"]["REF_CB10"]
    t = np.array(ref["theta"], float)
    r = _eval(t)
    c = pc.coeffs()
    assert pc.LBO_C_INT_SI == pytest.approx(
        c["C_LBO"] * r["A_ref_m2"] / r["splits"]["pz"], rel=1e-12)
    P3, T3 = float(t[16]), float(t[17])
    press = r["mdot_air_kgs"] / (P3 ** (-c["lbo_exp_P3"])
                                 * math.exp(T3 / c["lbo_T3_scale"]))
    evap = ((r["SMD_um"] * 1e-6) ** c["lbo_exp_SMD"]
            / pc.FUELS[r["fuel"]]["lambda_evap"])
    phi_int = (pc.LBO_C_INT_SI * r["splits"]["pz"] / r["V_pz_m3"]
               * press * evap / r["LHV_J_kg"])
    assert phi_int == pytest.approx(r["phi_LBO"], rel=1e-9)
    lo, hi = pc.LBO_INT_BAND
    assert lo < 1.0 < hi


def test_lbo_intensive_units_and_trends():
    base = _stab()["phi_LBO_raw"]
    assert 1e-4 < base < 10.0               # fuera de la saturación
    assert _stab(P3=7e5)["phi_LBO_raw"] < base   # P3^-1.3
    assert _stab(T3=500.0)["phi_LBO_raw"] < base
    assert _stab(smd=80.0)["phi_LBO_raw"] == pytest.approx(4 * base, rel=1e-12)
    assert _stab(V_pz=0.036)["phi_LBO_raw"] == pytest.approx(base / 2,
                                                             rel=1e-12)
    assert _stab(f_pz=0.6)["phi_LBO_raw"] == pytest.approx(2 * base, rel=1e-12)
    assert _stab(A_ref=0.5)["phi_LBO_raw"] == pytest.approx(base, rel=1e-12)
    m = _stab(P3=7e5)["phi_LBO_raw"] / base
    assert m == pytest.approx(2.0 ** (pc.coeffs()["lbo_exp_P3"]), rel=1e-12)


def test_lbo_unsaturated_value_observable():
    s = _stab(smd=4000.0)                    # satura a 10
    assert s["phi_LBO"] == 10.0
    assert s["phi_LBO_raw"] > s["phi_LBO"]
    u = _stab()
    assert u["phi_LBO_raw"] == u["phi_LBO"]


def test_lbo_margin_definition_and_g_sign(records):
    s = _stab()
    assert s["LBO_margin"] == pytest.approx(0.9 / s["phi_LBO"] - 1, rel=1e-12)
    assert s["LBO_margin_ratio"] == pytest.approx(0.9 / s["phi_LBO"],
                                                  rel=1e-12)
    t, r = records["8/annular"]
    i = NAMES.index("LBO")
    c = pc.coeffs()
    assert r["g"][i] == pytest.approx(
        r["phi_LBO"] * (1 + c["lbo_margin_min"]) / r["phi_pz"] - 1, rel=1e-12)


@pytest.mark.parametrize("key", ["5", "8", "25"])
def test_lbo_scale_margin_robust_across_transfer_band(records, key):
    """[CÁLCULO] Con φ_LBO en el extremo alto de la banda de transferencia
    (×1,45·4²), el LBO sigue cumpliendo con el θ diagnóstico."""
    t, r = records[f"{key}/annular"]
    assert r["lbo_law"] == "lefebvre_intensive_fpz"
    hi = r["phi_LBO_raw"] * pc.LBO_INT_BAND[1]
    c = pc.coeffs()
    assert hi * (1 + c["lbo_margin_min"]) / r["phi_pz"] - 1 < 0.0


def test_lbo_kj66_anchor_and_degenerate_unchanged():
    anchors = json.loads((HERE / "anchors.json").read_text(encoding="utf-8"))
    ref = anchors["anchors"]["REF_CB10"]
    r = _eval(np.array(ref["theta"], float))
    for k in ("tau_ms", "phi_pz", "eta_comb", "PF", "D_ref_mm", "L_tot_mm"):
        assert r[k] == pytest.approx(ref[k], rel=1e-9), k
    assert r["combustor_class"] == "micro"
    assert r["lbo_law"] == "lefebvre_aref_micro"
    assert r["tau_max_ms_applied"] == pc.TAU_MAX_MS == 10.0
    assert r["tau_ms"] < pc.TAU_MAX_MS
    i_lbo = NAMES.index("LBO")
    assert r["g"][i_lbo] <= 0.0
    # θ degenerado: ṁ = 0 ⇒ record degenerado infactible con g = 10
    bad = np.array(ref["theta"], float)
    bad[pc.IDX_MDOT_KGS] = 0.0
    rb = _eval(bad)
    assert rb["feasible"] is False
    assert "degenerate" in rb["source"]
    assert all(v == 10.0 for v in rb["g"])


# =====================================================================
# Política de caudal efectiva (fase 17.1-02, Tarea 1): D-373–D-376
#
# La misma política decide la clase en el spec, en la CLI y en la
# evaluación directa de θ. Las pruebas `test_selection_*` recorren los
# bordes, `test_architecture_*` el dominio anular, `test_fidelity_*` las
# etiquetas de fidelidad (con un doble de solver) y `test_resume_*` los
# checkpoints.
# =====================================================================
import phycb_cli as cli  # noqa: E402

P3_8, T3_8 = 358961.8805265866, 432.4902351398321


def _spec(m, T4=1200.0, **kw):
    return no.DesignSpec(P3_Pa=P3_8, T3_K=T3_8, mdot_kgs=m, T4_K=T4, **kw)


def _cli_spec(m, *extra, T4=1200.0):
    argv = ["--p3", repr(P3_8), "--t3", repr(T3_8), "--mdot", repr(m),
            "--t4", repr(T4), *extra]
    return cli.build_spec(cli.parse_args(argv))[0]


SCALE = pc.OPERATING_CLASS_SCALE
SEL = [(0.05, "micro", 0), (1.5, "micro", 0), (1.5 + 1e-6, "gap_1p5_5", 1),
       (5.0 - 1e-6, "gap_1p5_5", 1), (5.0, SCALE, 0), (5.0 + 1e-6, SCALE, 0),
       (8.0, SCALE, 0), (25.0 - 1e-6, SCALE, 0), (25.0, SCALE, 0)]


@pytest.mark.parametrize("m,cls,nwarn", SEL)
def test_selection_spec_cli_and_theta_share_policy(m, cls, nwarn):
    s = _spec(m)
    c = _cli_spec(m)
    for sp in (s, c):
        op = sp.operating
        assert op["class_id"] == cls
        assert len(op["warnings"]) == nwarn
    r = _eval(s.fix_operating_point(pc.THETA_REF))
    assert r["operating_class"] == cls
    assert r["operating_warnings"] == s.operating["warnings"]
    assert r["combustor_class"] == ("scale" if cls == SCALE else "micro")
    assert r["tau_max_ms_applied"] == s.operating["tau_max_ms"] == 10.0
    if cls == SCALE:
        assert r["scale_policy_rev"] == pc.SCALE_POLICY_REV
        assert s.operating["archs"] == pc.SCALE_ARCHS
        assert r["tau_min_ms_applied"] == pc.SCALE_TAU_MIN_MS
    else:
        assert r["scale_policy_rev"] is None
        assert s.operating["archs"] == pc.ARCH_NAMES


def test_selection_below_documented_band_warns_with_history():
    op = pc.resolve_operating_class(0.03, 1150.0)
    assert op["class_id"] == "micro_below_band" and len(op["warnings"]) == 1
    assert op["physics_class"] == "micro"


@pytest.mark.parametrize("m", [25.0 + 1e-6, 30.0])
def test_selection_above_25_rejected_everywhere(m):
    for build in (lambda: _spec(m), lambda: _cli_spec(m)):
        with pytest.raises(pc.OperatingDomainError) as e:
            build()
        assert (e.value.field, e.value.rule) == ("mdot_kgs", "max_25")
        assert "25 kg/s" in str(e.value)
    th = pc.THETA_REF.copy()
    th[pc.IDX_MDOT_KGS] = m
    r = _eval(th)
    assert r["feasible"] is False and "degenerate" in r["source"]
    assert "max_25" in r["degenerate_reason"]
    assert all(v == pc.G_DEGENERATE for v in r["g"])


@pytest.mark.parametrize("bad,reason", [
    (0.0, "mdot_kgs <= 0"), (-1.0, "mdot_kgs <= 0"),
    (float("nan"), "theta con NaN/inf o longitud insuficiente"),
    (float("inf"), "theta con NaN/inf o longitud insuficiente")])
def test_selection_invalid_mass_spec_cli_and_degenerate_theta(bad, reason):
    for build in (lambda: _spec(bad), lambda: _cli_spec(bad)):
        with pytest.raises(pc.OperatingDomainError) as e:
            build()
        assert (e.value.field, e.value.rule) == ("mdot_kgs", "finite_positive")
    th = pc.THETA_REF.copy()
    th[pc.IDX_MDOT_KGS] = bad
    r = _eval(th)
    # forma histórica del degenerado: mismo motivo, g positivo e infactible
    assert r["degenerate_reason"] == reason
    assert r["g"] == [pc.G_DEGENERATE] * pc.N_CONSTRAINTS
    assert r["feasible"] is False and r["operating_class"] == "none"


@pytest.mark.parametrize("bad", [True, None, "texto", float("nan")])
def test_selection_resolver_rejects_non_numeric(bad):
    with pytest.raises(pc.OperatingDomainError) as e:
        pc.resolve_operating_class(bad, 1200.0)
    assert e.value.as_dict()["rule"] == "finite_positive"


def test_selection_t4_cap_only_in_scale_class():
    assert _spec(8.0, T4=1250.0).operating["T4_max_K"] == 1250.0
    for build in (lambda: _spec(8.0, T4=1250.0 + 1e-6),
                  lambda: _cli_spec(8.0, T4=1250.0 + 1e-6)):
        with pytest.raises(pc.OperatingDomainError) as e:
            build()
        assert (e.value.field, e.value.rule) == ("T4_K", "max_1250")
    th = _spec(8.0).fix_operating_point(pc.THETA_REF)
    th[pc.IDX_T4_K] = 1250.0 + 1e-6
    r = _eval(th)
    assert "max_1250" in r["degenerate_reason"] and r["feasible"] is False
    th[pc.IDX_T4_K] = 1250.0
    assert "degenerate" not in _eval(th)["source"]
    # la clase micro no tiene ese tope (comportamiento histórico)
    assert no.DesignSpec(mdot_kgs=0.13, T4_K=1300.0).operating["T4_max_K"] \
        is None


def test_selection_kj66_micro_reverse_flow_unchanged():
    r = _eval(pc.THETA_REF)
    assert (r["arch"], r["operating_class"], r["combustor_class"]) == \
        ("reverse_annular", "micro", "micro")
    assert r["operating_warnings"] == [] and r["tau_max_ms_applied"] == 10.0
    assert r["fidelity_requested"] == r["fidelity_achieved"] == "L0"
    assert r["fidelity_degradation"] is None


def test_selection_cli_main_reports_flow_cap(tmp_path, capsys):
    rc = cli.main(["--mdot", "30", "--p3", repr(P3_8), "--t3", repr(T3_8),
                   "--t4", "1200", "--no-figures", "--no-color",
                   "--outdir", str(tmp_path / "o")])
    assert rc == 2
    assert "25 kg/s" in capsys.readouterr().out


# ------------------------------------------------------------- architecture
def test_architecture_can_explicit_rejected_at_scale():
    for build in (lambda: _spec(8.0, arch="can"),
                  lambda: _cli_spec(8.0, "--arch", "can")):
        with pytest.raises(pc.OperatingDomainError) as e:
            build()
        assert e.value.rule == "annular_only" and "anular" in str(e.value)
    assert _spec(0.13, arch="can").operating["archs"] == ("can",)


@pytest.mark.parametrize("val", [0.0, 0.4, -3.0])
def test_architecture_fix_can_rejected_at_scale(val):
    with pytest.raises(pc.OperatingDomainError) as e:
        _spec(8.0, fixed_vars={"arch": val})
    assert (e.value.field, e.value.rule) == ("fixed_vars.arch", "annular_only")
    with pytest.raises(pc.OperatingDomainError):
        _cli_spec(8.0, "--fix", f"arch={val}")


def test_architecture_explicit_vs_fix_conflict():
    with pytest.raises(pc.OperatingDomainError) as e:
        _spec(8.0, arch="annular", fixed_vars={"arch": 2.0})
    assert e.value.rule == "fix_conflict"
    ok = _spec(8.0, arch="annular", fixed_vars={"arch": 1.0})
    assert ok.operating["archs"] == ("annular",)
    fixed = _spec(8.0, fixed_vars={"arch": 2.0})
    assert fixed.operating["archs"] == ("reverse_annular",)
    # micro: semántica histórica, el fix gana a la arquitectura explícita
    m = no.DesignSpec(arch="annular", fixed_vars={"arch": 2.0})
    assert pc.decode_gates(m.fix_operating_point(pc.THETA_REF))["arch"] \
        == "reverse_annular"


def test_architecture_auto_domain_is_exactly_the_annulars():
    assert _spec(8.0).operating["archs"] == ("annular", "reverse_annular")
    assert _spec(8.0, arch="reverse_annular").operating["archs"] == \
        ("reverse_annular",)
    assert no.DesignSpec().operating["archs"] == pc.ARCH_NAMES


def test_architecture_nominal_is_deterministic_and_labelled():
    s = _spec(8.0)
    a = cli.nominal_theta(s)
    b = cli.nominal_theta(_spec(8.0))
    assert np.array_equal(a, b)
    assert pc.decode_gates(a)["arch"] in pc.SCALE_ARCHS
    assert pc.decode_gates(cli.nominal_theta(_spec(8.0, arch="annular")))[
        "arch"] == "annular"
    # micro conserva la gate del ancla
    assert cli.nominal_theta(no.DesignSpec())[pc.IDX_ARCH] == \
        pc.THETA_REF[pc.IDX_ARCH]


def test_architecture_explicit_theta_validated_not_transformed():
    s = _spec(8.0)
    th = pc.THETA_REF.copy()
    th[pc.IDX_ARCH] = 0.0
    assert s.fix_operating_point(th)[pc.IDX_ARCH] == 0.0     # sin transformar
    with pytest.raises(pc.OperatingDomainError) as e:
        no.evaluate_design(s, th)
    assert e.value.field == "theta.arch"
    th[pc.IDX_ARCH] = 1.0
    out = no.evaluate_design(s, th)
    assert out["record"]["arch"] == "annular"
    assert out["record"]["operating_class"] == SCALE


def test_architecture_cli_full_theta_overrides_and_validates(tmp_path, capsys):
    th = _spec(8.0).fix_operating_point(pc.THETA_REF)
    th[pc.IDX_ARCH] = 0.0
    rc = cli.main(["--theta", ",".join(repr(float(x)) for x in th),
                   "--no-figures", "--no-report", "--no-color",
                   "--outdir", str(tmp_path / "o")])
    assert rc == 2
    assert "anular" in capsys.readouterr().out


def test_architecture_sampling_is_balanced_and_in_domain():
    U = no.AutonomousCombustorDesigner._lhs(40, pc.NDIM,
                                            np.random.default_rng(71))
    s = _spec(8.0)
    archs = [pc.decode_gates(s.fix_operating_point(pc.denormalize(u),
                                                   sampled=True))["arch"]
             for u in U]
    assert archs.count("annular") == archs.count("reverse_annular") == 20
    micro = no.DesignSpec()
    archs_m = {pc.decode_gates(micro.fix_operating_point(pc.denormalize(u),
                                                         sampled=True))["arch"]
               for u in U}
    assert archs_m == set(pc.ARCH_NAMES)          # micro sin cambios
    fixed = _spec(8.0, arch="annular")
    assert {pc.decode_gates(fixed.fix_operating_point(pc.denormalize(u),
                                                      sampled=True))["arch"]
            for u in U} == {"annular"}


class _StubSurrogate:
    def predict(self, X):
        n = len(X)
        return (np.tile([0.99, 0.03, 0.2, 900.0, 5.0], (n, 1)),
                np.full((n, 5), 1e-3))


def test_architecture_search_and_acquisition_keep_domain():
    s = _spec(8.0)
    res = no.NSGA2(s, _StubSurrogate(), pop=12, generations=2, seed=71).run()
    archs = {pc.decode_gates(t)["arch"] for t in res["thetas"]}
    assert archs <= set(pc.SCALE_ARCHS)
    U = no.select_acquisition_batch(res, s, batch_size=6, seed=71)
    d = no.AutonomousCombustorDesigner(s, fidelity=pc.Fidelity.L0,
                                       log=lambda *a: None)
    recs = d._eval_and_store(np.array([pc.denormalize(u) for u in U]))
    assert recs and all(r["arch"] in pc.SCALE_ARCHS for r in recs)
    assert all(r["operating_class"] == SCALE for r in recs)


def _mini(spec, n=6, fid=pc.Fidelity.L0):
    d = no.AutonomousCombustorDesigner(spec, fidelity=fid,
                                       log=lambda *a: None)
    U = d._lhs(n, pc.NDIM, d.rng)
    d._eval_and_store(np.array([pc.denormalize(u) for u in U]))
    return d


def test_architecture_candidates_labelled_in_checkpoint(tmp_path):
    p = tmp_path / "ck.json"
    _mini(_spec(8.0)).save(str(p))
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["scale_policy_revision"] == pc.SCALE_POLICY_REV
    assert payload["operating_class"] == SCALE
    assert payload["fidelity"] == "L0"
    assert payload["arch_domain"] == list(pc.SCALE_ARCHS)
    assert {r["arch"] for r in payload["dataset"]} <= set(pc.SCALE_ARCHS)


# ----------------------------------------------------------------- fidelity
def _fake_l1(label):
    def fake(theta, rec, fid):
        rec["source"] = label
        return rec
    return fake


def test_fidelity_exploratory_warning_and_achieved_label(monkeypatch):
    monkeypatch.setattr(pc, "_try_l1", _fake_l1("crn_L1"))
    th = _spec(8.0).fix_operating_point(pc.THETA_REF)
    r = pc.evaluate(th, fidelity=pc.Fidelity.L1, use_cache=False,
                    calibrate=False)
    assert any("EXPLORATORIO" in w for w in r["operating_warnings"])
    assert (r["fidelity_requested"], r["fidelity_achieved"],
            r["fidelity_degradation"]) == ("L1", "L1", None)


def test_fidelity_fallback_is_degradation_not_warning(monkeypatch):
    label = "crn_L1(timeout→L0)"
    monkeypatch.setattr(pc, "_try_l1", _fake_l1(label))
    th = _spec(8.0).fix_operating_point(pc.THETA_REF)
    r = pc.evaluate(th, fidelity=pc.Fidelity.L1, use_cache=False,
                    calibrate=False)
    assert (r["fidelity_achieved"], r["fidelity_degradation"]) == ("L0", label)
    assert label not in " ".join(r["operating_warnings"])
    assert len(r["operating_warnings"]) == 1          # sólo el exploratorio


def test_fidelity_no_exploratory_warning_in_l0_or_micro(monkeypatch):
    monkeypatch.setattr(pc, "_try_l1", _fake_l1("crn_L1"))
    th8 = _spec(8.0).fix_operating_point(pc.THETA_REF)
    assert _eval(th8)["operating_warnings"] == []
    rm = pc.evaluate(pc.THETA_REF, fidelity=pc.Fidelity.L1, use_cache=False,
                     calibrate=False)
    assert rm["operating_warnings"] == [] and rm["fidelity_achieved"] == "L1"


def test_fidelity_spec_offer_unchanged_and_exploratory():
    assert set(cli.FIDELITY_BY_NAME) == {"L0", "L1", "L15", "L2A", "L2B"}
    s = _spec(8.0)
    assert s.resolve_operating_point(pc.Fidelity.L2A)["exploratory"] is True
    assert s.resolve_operating_point(pc.Fidelity.L0)["warnings"] == []
    assert no.DesignSpec().resolve_operating_point(
        pc.Fidelity.L2A)["warnings"] == []


@pytest.mark.parametrize("fid,src,exp", [
    (0, "correlations_L0", ("L0", None)),
    (0, "correlations_L0(cantera_unavailable)", ("L0", None)),
    (2, "crn_L15", ("L15", None)),
    (3, "ibrans_L2A(unavailable→crn_L1)", ("L1", "ibrans_L2A(unavailable→crn_L1)")),
    (4, "openfoam_L2B(out_of_scope→ibrans_L2A)",
     ("L2A", "openfoam_L2B(out_of_scope→ibrans_L2A)")),
    (1, "crn_L1(cantera_unavailable→L0)",
     ("L0", "crn_L1(cantera_unavailable→L0)"))])
def test_fidelity_achieved_parses_degradation_chain(fid, src, exp):
    assert pc.fidelity_achieved(fid, src) == exp


# ------------------------------------------------------------------- resume
def test_resume_micro_checkpoint_shape_and_reload(tmp_path):
    p = tmp_path / "micro.json"
    d = _mini(no.DesignSpec())
    d.save(str(p))
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert set(payload) == {"spec", "seed", "n_evals", "history", "dataset"}
    assert set(payload["dataset"][0]) == {"theta", *no.OUT_KEYS, "g",
                                          "source"}
    back = no.AutonomousCombustorDesigner.load(str(p),
                                               fidelity=pc.Fidelity.L0,
                                               log=lambda *a: None)
    assert [r["g"] for r in back.recs] == [r["g"] for r in d.recs]
    assert back.checkpoint_identity == "current"


def test_resume_scale_current_restores_g(tmp_path):
    p = tmp_path / "s.json"
    d = _mini(_spec(8.0))
    d.save(str(p))
    back = no.AutonomousCombustorDesigner.load(str(p),
                                               fidelity=pc.Fidelity.L0,
                                               log=lambda *a: None)
    assert [r["g"] for r in back.recs] == [r["g"] for r in d.recs]
    assert back.checkpoint_identity == "current"


def _tamper(p, fn):
    payload = json.loads(p.read_text(encoding="utf-8"))
    fn(payload)
    p.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("how", ["no_rev", "old_rev", "theta_op", "theta_can",
                                 "class"])
def test_resume_scale_incompatible_rejected_before_g(tmp_path, how):
    p = tmp_path / "s.json"
    _mini(_spec(8.0)).save(str(p))
    edit = {
        "no_rev": lambda q: q.pop("scale_policy_revision"),
        "old_rev": lambda q: q.update(
            scale_policy_revision="phycb-scale/1 (17.1-01, D-388)"),
        "theta_op": lambda q: q["dataset"][2]["theta"].__setitem__(18, 7.5),
        "theta_can": lambda q: q["dataset"][1]["theta"].__setitem__(0, 0.0),
        "class": lambda q: q.update(operating_class="micro"),
    }[how]
    _tamper(p, edit)
    with pytest.raises(no.CheckpointIncompatible) as e:
        no.AutonomousCombustorDesigner.load(str(p), fidelity=pc.Fidelity.L0,
                                            log=lambda *a: None)
    assert "búsqueda nueva" in str(e.value)


def test_resume_fidelity_mismatch_rejected(tmp_path):
    p = tmp_path / "s.json"
    _mini(_spec(8.0)).save(str(p))
    with pytest.raises(no.CheckpointIncompatible) as e:
        no.AutonomousCombustorDesigner.load(str(p), fidelity=pc.Fidelity.L1,
                                            log=lambda *a: None)
    assert "fidelidad" in str(e.value)


def test_resume_historical_scale_checkpoint_query_keeps_identity(tmp_path):
    p = tmp_path / "s.json"
    d = _mini(_spec(8.0))
    d.save(str(p))
    _tamper(p, lambda q: q.pop("scale_policy_revision"))
    spec, front = no.pareto_from_checkpoint(str(p))
    assert front and all(q["checkpoint_identity"] == "historical"
                         for q in front)
    q = no.AutonomousCombustorDesigner.load(str(p), fidelity=pc.Fidelity.L0,
                                            log=lambda *a: None,
                                            purpose="query")
    assert q.checkpoint_identity == "historical"
    assert [r["g"] for r in q.recs] == [r["g"] for r in d.recs]
    # el archivo histórico no se reescribe
    assert "scale_policy_revision" not in json.loads(
        p.read_text(encoding="utf-8"))
    # un spec histórico fuera del dominio (can a 8 kg/s) también se consulta
    _tamper(p, lambda q: q["spec"].update(arch="can"))
    spec2, front2 = no.pareto_from_checkpoint(str(p))
    assert spec2.operating is None
    assert all(q["checkpoint_identity"] == "historical" for q in front2)


def test_resume_cli_rejects_incompatible_checkpoint(tmp_path, capsys):
    p = tmp_path / "s.json"
    _mini(_spec(8.0)).save(str(p))
    _tamper(p, lambda q: q.pop("scale_policy_revision"))
    rc = cli.main(["--resume", str(p), "--rounds", "1", "--no-figures",
                   "--no-color", "--outdir", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == 2 and "could not resume" in out and "búsqueda nueva" in out


# =====================================================================
# Contrato A7 y verificación independiente de una celda (17.1-02, Tarea 2)
# =====================================================================
#: θ factible a 8 kg/s con las 26 g y DfAM (búsqueda corta de diagnóstico
#: de la ola 1 sobre 2c44c7f; en L0 ambas anulares dan el mismo θ salvo la
#: gate). El punto de operación se toma EXACTO de la fila congelada.
THETA_8_DIAG = [1.0, 0.625000046, 0.307366741, 0.126953473, 0.578594152,
                0.704734228, 9.843091112, 10.794938403, 0.681881151,
                7.181109515, 1.810823879, 2.641562967, 0.541332794,
                2.181716514, 0.507587533, 0.0, 0.0, 0.0, 0.0, 1200.0]


def _theta8(arch="annular"):
    row = sl.load_inlets()["inlets"]["8"]
    t = list(THETA_8_DIAG)
    t[pc.IDX_ARCH] = float(pc.ARCH_NAMES.index(arch))
    t[pc.IDX_P3_PA], t[pc.IDX_T3_K] = row["P03_Pa"], row["T03_K"]
    t[pc.IDX_MDOT_KGS] = row["mdot_kg_s"]
    return t


@pytest.fixture(scope="module")
def cand8(tmp_path_factory):
    out = {}
    for arch in ARCHS:
        t = _theta8(arch)
        spec = sl.expected_spec("8", arch)
        raw = sl.export_candidate_contract(
            t, spec, tmp_path_factory.mktemp(f"c8{arch}"))
        out[arch] = dict(theta=t, spec=spec, raw=raw,
                         sha=sl.sha256_bytes(raw))
    return out


def _verify(c, raw=None, **kw):
    raw = c["raw"] if raw is None else raw
    kw.setdefault("contract_sha256", sl.sha256_bytes(raw) if raw else None)
    return sl.verify_scale_candidate(c["theta"], c["spec"], raw,
                                     expect=dict(arch=c["spec"]["arch"]), **kw)


def _mutated(c, fn):
    d = json.loads(c["raw"].decode("utf-8"))
    fn(d)
    return json.dumps(d, indent=1, ensure_ascii=False).encode("utf-8")


# ----------------------------------------------------------------- outflow
@pytest.mark.parametrize("air,fuel,exp", [
    (8.0, 0.16, 8.16), (8.0, 0.0, 8.0), (0.0, 0.0, 0.0),
    (8.1234567891, 0.1234567894, round(8.1234567891 + 0.1234567894, 9)),
    (np.float64(8.0), np.float64(0.183147061), 8.183147061)])
def test_outflow_sum_of_source_operands(air, fuel, exp):
    assert gg.mdot_out_kgs({"mdot_air_kgs": air, "mdot_fuel_kgs": fuel}) == exp


@pytest.mark.parametrize("bad", [None, True, False, "8.0", -1e-9,
                                 float("nan"), float("inf"), "missing"])
@pytest.mark.parametrize("side", ["mdot_air_kgs", "mdot_fuel_kgs"])
def test_outflow_invalid_operand_is_null(bad, side):
    rec = {"mdot_air_kgs": 8.0, "mdot_fuel_kgs": 0.16}
    if bad == "missing":
        rec.pop(side)
    else:
        rec[side] = bad
    assert gg.mdot_out_kgs(rec) is None


def test_outflow_overflow_and_non_dict_are_null():
    assert gg.mdot_out_kgs({"mdot_air_kgs": 1.7e308,
                            "mdot_fuel_kgs": 1.7e308}) is None
    assert gg.mdot_out_kgs(None) is None and gg.mdot_out_kgs([]) is None


def test_outflow_contract_field_rounding_and_strict_json():
    t = np.array(_theta8())
    rec = _eval(t)
    c = gg.build_contract(t, rec)
    d = c["derived"]
    assert d["mdot_out_kgs"] == round(rec["mdot_air_kgs"]
                                      + rec["mdot_fuel_kgs"], 9)
    assert abs(d["mdot_out_kgs"] - (d["mdot_air_kgs"] + d["mdot_fuel_kgs"])) \
        <= sl.TOL_MDOT_KGS
    json.dumps(c, allow_nan=False)                 # JSON estricto
    # el respaldo CFD no sustituye a la suma: sin combustible fuente, null
    rec2 = dict(rec)
    rec2.pop("mdot_fuel_kgs")
    c2 = gg.build_contract(t, rec2)
    assert c2["derived"]["mdot_out_kgs"] is None
    assert c2["cfd_boundary_conditions"]["mdot_out_kgs"] is not None
    assert json.loads(json.dumps(c2, allow_nan=False))["derived"][
        "mdot_out_kgs"] is None


def test_outflow_degenerate_record_stays_infeasible():
    th = pc.THETA_REF.copy()
    th[pc.IDX_MDOT_KGS] = 0.0
    rec = _eval(th)
    c = gg.build_contract(th, rec)
    assert c["derived"]["mdot_out_kgs"] == 0.0
    assert c["derived"]["feasible"] is False and rec["feasible"] is False


# ---------------------------------------------------------------- contract
def test_contract_root_keys_and_paired_cups(cand8):
    for arch, c in cand8.items():
        d = json.loads(c["raw"].decode("utf-8"))
        assert d["schema"] == gg.SCHEMA and d["arch"] == arch
        assert d["operating_class"] == pc.OPERATING_CLASS_SCALE
        assert d["scale_policy_revision"] == pc.SCALE_POLICY_REV
        assert d["tau_max_ms"] == 10.0 and d["operating_warnings"] == []
        sw, inj = d["swirler"], d["injectors"]
        # D-386: dimensiones por copa, conteo y áreas; anillo y fase emparejados
        assert sw["count"] == inj["count"] >= 1
        if inj["type"] == "vaporizer":     # conserva n_tubes (limitación 5c)
            assert inj["n_tubes"] == inj["count"]
        assert sw["A_sw_total_mm2"] == pytest.approx(
            sw["A_sw_each_mm2"] * sw["count"], rel=1e-6)
        for k in ("d_hub_mm", "d_tip_mm", "n_vanes"):
            assert sw[k] > 0
        assert inj["ring_r_mm"] > 0 and inj["phi0_deg"] == 0.0
    kj = gg.build_contract(pc.THETA_REF, _eval(pc.THETA_REF))
    assert kj["operating_class"] == "micro" and kj["scale_policy_revision"] \
        is None


# --------------------------------------------------------- acceptance guard
def test_acceptance_guard_accepts_reproducible_candidate(cand8):
    for c in cand8.values():
        v = _verify(c)
        assert v["ok"], v["failures"]
        assert [len(v["g_blocks"][b]) for b, _n in sl.G_BLOCKS] == \
            [12, 4, 5, 5]
        assert v["dfam_exported"]["feasible"] is True
        assert v["P04_closure"]["residual_Pa"] <= v["P04_closure"]["tol_Pa"]
        assert v["raw_pf"] == v["record"]["PF"]


def test_acceptance_guard_rejects_missing_or_foreign_contract(cand8):
    c = cand8["annular"]
    assert any(f.startswith("contract_missing") for f in
               sl.verify_scale_candidate(c["theta"], c["spec"], None)[
                   "failures"])
    v = _verify(c, contract_sha256="0" * 64)
    assert any(f.startswith("contract_hash") for f in v["failures"])
    other = cand8["reverse_annular"]["raw"]
    v = _verify(c, raw=other)
    assert any(f.startswith("contract_arch") for f in v["failures"])


def test_acceptance_guard_rejects_foreign_theta_arch_or_point(cand8):
    c = dict(cand8["annular"])
    v = _verify(dict(c, theta=_theta8("reverse_annular")))
    assert any(f.startswith("theta_arch") for f in v["failures"])
    t = list(c["theta"])
    t[pc.IDX_P3_PA] += 1.0
    v = _verify(dict(c, theta=t))
    assert any(f.startswith("theta_point") for f in v["failures"])
    v = sl.verify_scale_candidate(c["theta"], c["spec"], c["raw"],
                                  expect=dict(arch="annular",
                                              P03_Pa=c["theta"][16] + 1.0))
    assert any(f.startswith("theta_point") for f in v["failures"])


@pytest.mark.parametrize("val", [None, "8.18", True, -1.0, 8.5])
def test_acceptance_guard_rejects_invalid_a7(cand8, val):
    c = cand8["annular"]
    raw = _mutated(c, lambda d: d["derived"].__setitem__("mdot_out_kgs", val))
    v = _verify(c, raw=raw)
    assert any(f.startswith("A7") for f in v["failures"]), v["failures"]


@pytest.mark.parametrize("edit", [
    lambda d: d["derived"].__setitem__("source", "crn_L1(timeout→L0)"),
    lambda d: d["run_meta"].__setitem__("fidelity", "L1")])
def test_acceptance_guard_rejects_degraded_fidelity(cand8, edit):
    c = cand8["annular"]
    v = _verify(c, raw=_mutated(c, edit))
    assert any(f.startswith("contract_fidelity") for f in v["failures"])


@pytest.mark.parametrize("g,code", [
    ([-0.1] * 25 + [0.2], "g_positive"),
    ([-0.1] * 25 + [float("nan")], "g_nonfinite"),
    ([-0.1] * 21, "g_missing")])
def test_acceptance_guard_rejects_bad_g(cand8, g, code):
    c = cand8["annular"]
    v = _verify(c, constraints_fn=lambda rec, t: g)
    assert any(f.startswith(code) for f in v["failures"]), v["failures"]


def test_acceptance_guard_rejects_absent_neutral_or_disabled_dfam(cand8):
    c = cand8["annular"]
    v = _verify(c, dfam_fn=lambda: None)
    assert any(f.startswith("dfam_checker_missing") for f in v["failures"])

    class _Neutral:
        G_NAMES = gg._dfam().G_NAMES

        @staticmethod
        def check(contract, **kw):
            return dict(g_dfam=[-1.0] * 5, feasible=True, violations=[])
    v = _verify(c, dfam_fn=lambda: _Neutral)
    assert any(f.startswith("dfam_checker_neutral") for f in v["failures"])
    v = _verify(dict(c, spec=dict(c["spec"], dfam=False)))
    assert any(f.startswith("no_dfam") for f in v["failures"])


def test_acceptance_guard_partial_or_timeout_cell_is_blocked(tmp_path):
    row = sl.load_inlets()["inlets"]["8"]
    st = sl.ACCEPT_STAGES[0]
    run = dict(rc=None, timed_out=True, elapsed_s=120.0)
    cell = sl.assess_cell("8", "annular", st, tmp_path, run, row)
    assert cell["status"] == "blocked" and cell["cause"].startswith("timeout")
    run = dict(rc=0, timed_out=False, elapsed_s=3.0)
    cell = sl.assess_cell("8", "annular", st, tmp_path / "vacio", run, row)
    assert cell["status"] == "blocked" and "parcial" in cell["cause"]


def _chain_with_pf(pf, missing=False):
    def chain(theta, gates, dp, material, use_cantera):
        return {"pf": {} if missing else {"PF": pf}}
    return chain


@pytest.mark.parametrize("pf,missing,code", [
    (None, True, "raw_pf_missing"), (float("nan"), False, "raw_pf_nonfinite"),
    (float("inf"), False, "raw_pf_nonfinite"), (True, False, "raw_pf_type"),
    (-0.1, False, "raw_pf_negative")])
def test_acceptance_guard_raw_pf(cand8, pf, missing, code):
    """El record publicado lleva PF = 0 (lo que dejaría `_safe`) y pasa el g
    de PF; el guard mira la cadena y falla igualmente."""
    c = cand8["annular"]
    rec = dict(_eval(np.array(c["theta"])), PF=0.0)
    contract = json.loads(c["raw"].decode("utf-8"))
    contract["derived"]["PF"] = 0.0
    assert rec["PF"] / 0.40 - 1.0 < 0.0             # el g de PF «pasa»
    raw, fails = sl.raw_pf_guard(c["theta"], rec, contract, 0.40,
                                 chain_fn=_chain_with_pf(pf, missing))
    assert raw is None and fails and fails[0].startswith(code)
    v = _verify(c, chain_fn=_chain_with_pf(pf, missing))
    assert any(f.startswith(code) for f in v["failures"])


def test_acceptance_guard_raw_pf_true_zero_is_distinguished(cand8):
    c = cand8["annular"]
    rec = dict(_eval(np.array(c["theta"])), PF=0.0)
    contract = {"derived": {"PF": 0.0}}
    raw, fails = sl.raw_pf_guard(c["theta"], rec, contract, 0.40,
                                 chain_fn=_chain_with_pf(0.0))
    assert raw == 0.0 and fails == []
    # y el PF crudo real coincide con el record y el contrato del candidato
    rec_real = _eval(np.array(c["theta"]))
    raw, fails = sl.raw_pf_guard(c["theta"], rec_real,
                                 json.loads(c["raw"].decode("utf-8")), 0.40)
    assert fails == [] and raw == rec_real["PF"] and 0.0 < raw < 0.40


def test_acceptance_guard_verify_candidate_cli(cand8, tmp_path, capsys):
    c = cand8["reverse_annular"]
    (tmp_path / "combustor.json").write_bytes(c["raw"])
    cand = dict(theta=c["theta"], spec=c["spec"],
                contract_path="combustor.json",
                expect=dict(arch="reverse_annular"))
    p = tmp_path / "cand.json"
    p.write_text(json.dumps(cand), encoding="utf-8")
    assert sl.main(["verify-candidate", "--candidate", str(p)]) == 0
    cand["theta"] = _theta8("annular")
    p.write_text(json.dumps(cand), encoding="utf-8")
    assert sl.main(["verify-candidate", "--candidate", str(p)]) == 1
    assert "theta_arch" in capsys.readouterr().out


# ------------------------------------------------------- 17.1-03 · matriz
def test_matrix_timeout_d391_by_mass():
    assert sl.accept_timeout_s("5") == 240.0
    assert sl.accept_timeout_s("25") == 240.0
    assert sl.accept_timeout_s("8") == sl.ACCEPT_TIMEOUT_S == 120.0
    assert "D-391" in sl.ACCEPT_TIMEOUT_POLICY


def test_matrix_support_paths_do_not_change_model():
    for rel in ("validation/scale_l0.py", "validation/test_scale_l0.py",
                "validation/scale_acceptance.json", "test_phycb.py",
                "validation/SCALE-L0-RESULTS.md",
                "CombustorDesigner/src/CombustorDesigner/Swirler.cs",
                "CombustorDesigner.Example/Program.cs"):
        assert sl.is_support_path(rel), rel
    for rel in ("physics_core.py", "neural_optimizer.py", "phycb_cli.py",
                "geometry_generator.py", "dfam_checker.py",
                "validation/scale_inlets.json", "validation/cases.py"):
        assert not sl.is_support_path(rel), rel


def test_matrix_single_model_source():
    a = "a" * 40
    doc = {"masses": {"8": {"status": "accepted", "model_source_sha": a},
                      "5": {"status": "blocked", "model_source_sha": "b" * 40}}}
    assert sl.matrix_model_source(doc) == a
    assert sl.matrix_model_source({}) is None
    doc["masses"]["25"] = {"status": "accepted", "model_source_sha": "c" * 40}
    with pytest.raises(sl.ScaleError):
        sl.matrix_model_source(doc)


# ------------------------------------------------------- 17.1-03 · gate CAD
def _cube_stl(path, lo=0.0, hi=10.0):
    import struct
    import numpy as np
    c = np.array([[lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
                  [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi]])
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(faces)))
        for a, b, d in faces:
            f.write(struct.pack("<3f", 0, 0, 0))
            for v in (c[a], c[b], c[d]):
                f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")


def test_cad_canonical_sha_is_order_independent():
    a = sl.canonical_sha256({"b": 1, "a": [1, 2]})
    assert a == sl.canonical_sha256({"a": [1, 2], "b": 1})
    with pytest.raises(ValueError):
        sl.canonical_sha256({"x": float("nan")})


def test_cad_stl_reader_and_point_in_mesh(tmp_path):
    p = tmp_path / "cube.stl"
    _cube_stl(p)
    tris = sl.read_stl(p)
    s = sl.stl_stats(tris)
    assert s["n_triangles"] == 12 and s["finite"]
    assert abs(s["volume_mm3"] - 1000.0) < 1e-6
    assert s["bbox_mm"] == [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
    pim = sl.PointInMesh(tris, cell_mm=2.0)
    assert pim.inside((5, 5, 5)) and pim.inside((1, 9, 2))
    assert not pim.inside((11, 5, 5)) and not pim.inside((5, -1, 5))
    (tmp_path / "bad.stl").write_bytes(b"solid x\nendsolid x\n" + b" " * 80)
    with pytest.raises(sl.ScaleError):
        sl.read_stl(tmp_path / "bad.stl")


def _rep(**kw):
    rep = dict(schema="phycb-dfam-voxel-1", source="dfam_voxel(picogk)",
               arch="annular", voxel_mm=0.8, g_voxel=[-0.5, -1.0, -1.0],
               g_voxel_names=list(sl.DFAM_VOXEL_G_NAMES), feasible=True,
               violations=[], detail={k: {} for k in sl.DFAM_DETAIL_KEYS},
               parts=[dict(name="Liner", frac_unsupported=0.001)])
    rep.update(kw)
    return rep


def test_cad_dfam_report_strict():
    ok = sl.validate_dfam_report(_rep(), "annular", 0.8, ["Liner"])
    assert ok["feasible"] and ok["g_voxel"] == [-0.5, -1.0, -1.0]
    # g > 0 con reporte íntegro: incumplimiento REGISTRADO, no error
    v = sl.validate_dfam_report(
        _rep(g_voxel=[0.2, -1.0, -1.0], feasible=False,
             violations=["overhang"]), "annular", 0.8, ["Liner"])
    assert v["feasible"] is False and v["violations"] == ["overhang"]
    for bad in (dict(source="dfam_voxel(failed)"), dict(arch="can"),
                dict(voxel_mm=0.5), dict(g_voxel=[0.1, float("nan"), 0]),
                dict(g_voxel=[-1, -1]), dict(feasible=False),
                dict(g_voxel=[0.2, -1, -1], feasible=False, violations=[]),
                dict(detail={"overhang": {}}),
                dict(parts=[dict(name="Liner")]),
                dict(g_voxel_names=["trapped_powder", "overhang_voxel",
                                    "drain_exits"])):
        with pytest.raises(sl.ScaleError):
            sl.validate_dfam_report(_rep(**bad), "annular", 0.8, ["Liner"])


def test_cad_cup_stations_from_pilot_contracts():
    doc = sl.load_acceptance()
    for arch, sign in (("annular", -1.0), ("reverse_annular", 1.0)):
        con = json.loads(sl.decode_bytes(
            doc["cells"][f"8/{arch}"]["contract"]["data"]))
        g = sl.cup_geometry(con, 0.8)
        assert g["n"] == con["injectors"]["count"] == con["swirler"]["count"]
        assert g["sign"] == sign and len(g["stations"]) == g["n"]
        # área por copa del contrato, no el total dividido otra vez
        assert abs(math.pi * (g["r_tip"] ** 2 - g["r_hub"] ** 2)
                   - g["a_each"]) < 1e-3 * g["a_each"]
        for s in g["stations"]:
            assert abs(math.hypot(s["x"], s["y"]) - g["ring"]) < 1e-9
            assert s["z_lo"] < s["z"] < s["z_hi"]
            # el tramo de la copa no entra en el barril del liner
            z0, z1 = con["liner"]["profile"][0][0], \
                con["liner"]["profile"][-1][0]
            assert s["z_hi"] < z0 or s["z_lo"] > z1


def test_cad_recommendation_rule_is_reproducible():
    doc = sl.load_acceptance()
    r1, r2 = sl.recommend_pilot(doc), sl.recommend_pilot(doc)
    assert r1 == r2
    assert r1["key"] in ("8/annular", "8/reverse_annular")
    assert r1["steps"][0]["rule"].startswith("R1")
    assert "empate" in r1["steps"][0]["result"]
    assert r1["contract_sha256"] == doc["cells"][r1["key"]]["contract"][
        "sha256"]


def test_cad_runs_check_rejects_incomplete_block():
    errs = sl.check_cad_runs({"schema": sl.CAD_SCHEMA, "runs": {}})
    assert any("runs" in e for e in errs)
    assert any("negativo" in e for e in errs)
