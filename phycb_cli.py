#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUASAR Phy-CB · phycb_cli.py
============================
PRODUCT entry point: the engineer writes four numbers (the operating point
that comes out of the compressor) and receives the complete package —
verified design, Pareto front, geometry contract, self-contained HTML
report, accumulated dataset and, optionally, the printable STLs of layer 5c
(C#/PicoGK).

    # manual design with the nominal rules (WAVE1 §3) — ~1 s
    python phycb_cli.py --p3 250000 --t3 430 --mdot 0.13 --t4 1150 \\
        --fuel jet_a --arch reverse_annular --outdir runs/kj66

    # autonomous optimisation (WAVE4) — the full product
    python phycb_cli.py --p3 250000 --t3 430 --mdot 0.13 --t4 1150 \\
        --optimize --rounds 5 --n-init 320 --outdir runs/x
    python phycb_cli.py --quick --outdir runs/quick     # smoke (~1-2 min)

    # --fidelity governs EVERY expensive evaluation of the loop, not just
    # the last one. Above L0 it is paid per point, so the CLI times one
    # point and quotes the projected wall clock before spending it:
    python phycb_cli.py --optimize --fidelity L1
    python phycb_cli.py --optimize --fidelity L1 --search-fidelity L0

    # spec read from a run of the upstream sibling
    python phycb_cli.py --from-run ../Phy-AC/runs/demo --t4 1150 --quick

    # interactive wizard (asks for the whole configuration):
    python phycb_cli.py --interactive
    python phycb_cli.py                    # no arguments = wizard
    python phycb_cli.py --fidelity L1 --stl -i    # flags PRE-SEED the wizard

    # the OTHER path: predict a PUBLISHED machine from the validation set
    python phycb_cli.py --analyze B1
    python phycb_cli.py --analyze list

    # environment, optional dependencies and provisional data
    python phycb_cli.py --info

    # over an existing checkpoint (without re-optimising)
    python phycb_cli.py --outdir runs/x --list-pareto
    python phycb_cli.py --outdir runs/x --pareto-pick 3 --stl --voxel 0.8

Outputs in <outdir>/ (CONTEXTO §6.3):
    phycb_run.json    checkpoint (spec + seed + history + dataset)
    dataset.csv       accumulated physical dataset (flywheel seed)
    phys_cache.jsonl  persistent cache — ONLY written at fidelity >= L1
    report.html       self-contained report (layer 5b)
    figures/          *.png when matplotlib is available
    geometry/         combustor.json (`phycb-combustor-1` contract), *.csv,
                      cfd_boundary_conditions.json, dfam_report.json, *.step
    sweep_*.csv       phi_PZ / load sweep — the combustor "map"
    *.stl             with --stl/--voxel (layer 5c, C#/PicoGK)

TWO THINGS THIS CLI DOES THAT A PARAMETERISER DOES NOT:

  * **`--analyze` exists.** A parameteriser is not an analyser. The default
    path SYNTHESISES a combustor; `--analyze` takes a PUBLISHED case from
    `validation/cases.py` and only predicts it. Mixing the two is how a
    model gets graded against a machine it also designed, so the CLI
    rejects the synthesis flags next to `--analyze` instead of ignoring
    them.
  * **An infeasible spec is reported as such, naming the binding
    constraint over the WHOLE population** — WALLS (no sampled design
    clears it: change the spec) versus TIGHT (some clear it: spend more
    budget). A `violation = 0.149` that does not say which constraint is
    not information, it is a number.

Everything optional degrades WITH A WARNING (contract §10.4): without
CadQuery there is no STEP and it says so; without the .NET SDK there are no
STLs and it says so; a fidelity that cannot be reached falls back down the
ladder with every reason attached. `--strict` turns any announced
degradation into a failure (exit 4), which is the single question a CI
needs to ask.

Exit codes (family convention): 0 done · 2 bad usage or unreadable input ·
3 no feasible design for this spec · 4 --strict and something degraded.

Language: the whole CLI — help, prompts and progress — is in ENGLISH.
"""

from __future__ import annotations

import os
import sys

# ==========================================================================
# CRITICAL TRAP OF THIS ENVIRONMENT — contract §0.b. DO NOT MOVE A LINE.
# Phy-AC is installed as an EDITABLE package: from any cwd other than
# Phy-CB's, `import physics_core` would SILENTLY resolve to Phy-AC's
# (NDIM=15, axial-compressor physics) and return plausible, completely
# wrong numbers. The insert goes BEFORE every sibling import, and
# `assert_provenance()` right after.
# ==========================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse           # noqa: E402
import csv                # noqa: E402
import importlib.util     # noqa: E402
import json               # noqa: E402
import math               # noqa: E402
import re                 # noqa: E402
import shutil             # noqa: E402
import subprocess         # noqa: E402
import time               # noqa: E402
from datetime import datetime, timezone   # noqa: E402

import numpy as np        # noqa: E402

import cli_style as ui             # noqa: E402
import cycle_seam                  # noqa: E402
import dfam_checker                # noqa: E402
import geometry_generator          # noqa: E402
import neural_optimizer            # noqa: E402
import physics_core as pc          # noqa: E402
import report_generator            # noqa: E402
import structures_core             # noqa: E402
import visualization               # noqa: E402
from neural_optimizer import (AutonomousCombustorDesigner,  # noqa: E402
                              DesignSpec, OUT_KEYS, evaluate_design,
                              pareto_from_checkpoint)
from physics_core import (ARCH_NAMES, BOUNDS_HI, BOUNDS_LO,  # noqa: E402
                          FUELS, Fidelity, NDIM, SEED, THETA_REF, VAR_NAMES,
                          register_hifi_pair, set_cache_path)

# A loud failure always beats a silently wrong result.
pc.assert_provenance(pc, neural_optimizer, geometry_generator,
                     report_generator, structures_core, dfam_checker,
                     visualization, cycle_seam, ui)


# ==========================================================================
# 0. CLI constants
# ==========================================================================
#: Project root — every path is built from here so that the CLI works from
#: any cwd (trap §0.b also reaches the C# side).
HERE = os.path.dirname(os.path.abspath(__file__))

#: Layer 5c (C#/PicoGK). Byte-compatible invocation with Phy-AC's:
#: `dotnet build <sln>` + `dotnet <Example.dll> <contract> <outdir> <voxel>`.
SLN_PATH = os.path.join(HERE, "CombustorDesigner.sln")
EXAMPLE_DIR = os.path.join(HERE, "CombustorDesigner.Example")
EXAMPLE_DLL = "CombustorDesigner.Example.dll"
DOTNET_FALLBACK = r"C:\Program Files\dotnet\dotnet.exe"

#: Default voxel of layer 5c (CONTEXTO §8): 0.8 draft · 0.5 iteration ·
#: 0.3–0.4 final. Cost scales ~1/voxel³.
VOXEL_DEFAULT_MM = 0.5

#: Frozen regression anchors (contract §10.9). Moving them takes the
#: deliberate act `--freeze-anchors`, citing the correction that moved them.
ANCHOR_PATH = os.path.join(HERE, "validation", "anchors.json")
ANCHOR_KEYS = ("eta_comb", "dp_over_p", "PF", "T_wall_max_K", "EI_NOx",
               "phi_pz", "tau_ms", "T04_K", "D_ref_mm", "L_tot_mm",
               "SMD_um", "EI_CO")
ANCHOR_RTOL = 1.0e-9          # bit-exact determinism (contract §10.8)

#: Default operating point = anchor case REF_CB10 / KJ66 (contract §1).
DEFAULT_OP = dict(p3=250_000.0, t3=430.0, mdot=0.13, t4=1150.0)

FIDELITY_BY_NAME = {"L0": Fidelity.L0, "L1": Fidelity.L1,
                    "L15": Fidelity.L15, "L2A": Fidelity.L2A,
                    "L2B": Fidelity.L2B}

#: One line per rung of the ladder, for `--info`.
FIDELITY_WHAT = {
    "L0": "meanline + correlations (~1.3 ms/point) — the cheap rung the "
          "loop can afford on every point",
    "L1": "chemical reactor network with Cantera (seconds/point)",
    "L15": "cold aerodynamic field (D3Q19 LBM) feeding the CRN",
    "L2A": "native reactive IB-RANS, full combustor case (tens of minutes)",
    "L2B": "external OpenFOAM backend — OUT OF SCOPE in v1 (contract §12)",
}

#: Names of the `g_spec` stretch of the global vector (contract §5.4), in
#: the EXACT order `DesignSpec.constraints` concatenates them.
G_SPEC_NAMES = ("spec_dp_max", "spec_PF_max", "spec_envelope_D",
                "spec_envelope_L")
G_STRUCT_NAMES = ("T_wall", "sigma_th", "hoop", "t_wall", "banned")

#: Sweep points (`--sweep`). BOUNDED loop — contract §10.5.
SWEEP_N = 13
SWEEP_LOAD_LO, SWEEP_LOAD_HI = 0.40, 1.20     # fraction of design mdot

#: Family budget (WAVE4 §1.6) — do NOT re-tune without data.
BUDGET_FULL = dict(n_init=320, rounds=5, batch_size=14)
BUDGET_QUICK = dict(n_init=100, rounds=2, batch_size=8)

#: Flags that say WHAT to compute. Anything outside this list only says
#: HOW (fidelity, output format, colour, verbosity), so a command line made
#: only of those has told the CLI nothing about the job — and that is
#: exactly when the wizard should open, pre-seeded with what was passed.
#: `phycb --fidelity L15` is a configured wizard, not a default design.
ACTION_FLAGS = (
    # the spec
    "--p3", "--t3", "--mdot", "--t4", "--fuel", "--arch", "--dp-max",
    "--pf-max", "--envelope-d", "--envelope-l", "--no-dfam", "--spec-file",
    "--fix", "--from-run",
    # the budget / mode
    "--optimize", "--quick", "--rounds", "--n-init", "--batch-size",
    "--resume",
    # the direct modes
    "--eval-theta", "--theta", "--list-pareto", "--pareto-pick",
    "--from-checkpoint", "--analyze", "--info", "--freeze-anchors",
)

#: Synthesis flags. `--analyze` receives a PUBLISHED case, so accepting any
#: of these next to it would suggest the analysis path parameterises
#: something. It does not, and the CLI says so instead of ignoring them.
SYNTHESIS_FLAGS = ("--optimize", "--rounds", "--n-init", "--batch-size",
                   "--quick", "--resume", "--fix", "--eval-theta", "--theta",
                   "--from-run", "--spec-file", "--list-pareto",
                   "--pareto-pick", "--p3", "--t3", "--mdot", "--t4",
                   "--arch", "--fuel", "--envelope-d", "--envelope-l")

#: English display labels for the published validation cases. The dataset
#: itself (`validation/cases.py`) keeps its own wording — it is cited
#: source material and `validate.py` writes RESULTS.md from it — so the
#: CLI carries its own label instead of rewording the evidence.
CASE_LABELS = {
    "A1": "Class 0 · micro-turbojet 0.1-5 kW (reference band table)",
    "A2": "Class 1 · microturbine 5-50 kW (reference band table)",
    "A3": "Class 2 · microturbine 50-250 kW (reference band table)",
    "A4": "Internal consistency of class_table.csv (FAR <-> T3, T4)",
    "B1": "KJ66 / hobby class · reverse annular + vaporizer",
    "B2": "KJ66 · sensitivity to the community mass flow (0.23 kg/s)",
    "C1": "Optimised micro-turbojet · PZ air split 39.35 % -> 32 %",
    "D1": "Bath/IAAPS/HiETA · IN625, radial swirl, T3 = 1093 K",
    "D2": "Emission trends · sign of dEI/dphi_PZ and dEI/dtau",
}

#: What each validation level can and cannot falsify, in one line.
LEVEL_WHAT = {
    "A": "reference band table by machine class",
    "B": "architecture and class bands (NO published performance data)",
    "C": "published air-split anchor — the most falsifiable case here",
    "D": "thermal extreme with a published wall material",
}


# ==========================================================================
# 1. Arguments — FLAT, like the rest of the family
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phycb", add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="QUASAR Phy-CB — autonomous design of 3D-printable "
                    "Brayton combustion chambers (spec -> verified design "
                    "-> geometry contract -> STL).",
        epilog="examples:\n"
        "  phycb --interactive                        guided wizard\n"
        "  phycb --fidelity L1 --stl -i               flags pre-seed the "
        "wizard\n"
        "  phycb --p3 250000 --t3 430 --mdot 0.13 --t4 1150\n"
        "                                             nominal design (~1 s)\n"
        "  phycb --quick --outdir runs/quick          quick smoke run\n"
        "  phycb --optimize --rounds 5 --outdir runs/x\n"
        "  phycb --from-run ../Phy-AC/runs/demo --t4 1150 --quick\n"
        "  phycb --analyze B1                         predict a PUBLISHED "
        "case\n"
        "  phycb --info                               environment and "
        "provisional data\n"
        "  phycb --outdir runs/x --list-pareto\n"
        "  phycb --outdir runs/x --pareto-pick 3 --stl --voxel 0.8\n")

    s = p.add_argument_group("specification (the four numbers)")
    s.add_argument("--p3", type=float, default=None,
                   help="combustor inlet total pressure [Pa] "
                        f"(default {DEFAULT_OP['p3']:,.0f})")
    s.add_argument("--t3", type=float, default=None,
                   help="combustor inlet total temperature [K] "
                        f"(default {DEFAULT_OP['t3']:.0f})")
    s.add_argument("--mdot", type=float, default=None,
                   help="air mass flow [kg/s] "
                        f"(default {DEFAULT_OP['mdot']})")
    s.add_argument("--t4", type=float, default=None,
                   help="turbine inlet total temperature [K] "
                        f"(default {DEFAULT_OP['t4']:.0f})")
    s.add_argument("--fuel", choices=list(FUELS), default="jet_a",
                   help="fuel (default jet_a)")
    s.add_argument("--dp-max", type=float, default=0.04,
                   help="HARD cycle constraint on total pressure loss "
                        "(default 0.04 = 4%%; micro turbines run 2-4%%). In "
                        "a combustor dp is SPENT on mixing up to the "
                        "allowance, it is not minimized")
    s.add_argument("--arch", choices=list(ARCH_NAMES), default=None,
                   help="combustor architecture; omit to leave it as a "
                        "design variable (gate theta[0])")
    s.add_argument("--material", type=str, default="Inconel-625",
                   choices=structures_core.list_materials(),
                   help="HOT wall (liner) alloy (default Inconel-625)")
    s.add_argument("--material-casing", type=str, default="Inconel-718",
                   choices=structures_core.list_materials(),
                   help="COLD wall (casing) alloy (default Inconel-718)")
    s.add_argument("--pf-max", type=float, default=0.40,
                   help="pattern factor allowed by the turbine "
                        "(default 0.40)")
    s.add_argument("--envelope-d", type=float, default=None,
                   help="max casing diameter [mm] (spec constraint)")
    s.add_argument("--envelope-l", type=float, default=None,
                   help="max total length [mm] (spec constraint)")
    s.add_argument("--no-dfam", action="store_true",
                   help="disable the 5 LPBF manufacturability constraints "
                        "(study/debug only — the design may not be "
                        "printable)")
    s.add_argument("--spec-file", type=str, default=None,
                   help="JSON with the spec (overwrites flags)")
    s.add_argument("--fix", action="append", default=None,
                   metavar="VAR=VALUE",
                   help="pin a design variable and optimize the rest "
                        "(repeatable), e.g. --fix swirl_SN=0.8 --fix "
                        "t_liner_mm=1.0. Vars: "
                        + ", ".join(VAR_NAMES[:16]))

    b = p.add_argument_group("optimization budget")
    b.add_argument("--optimize", action="store_true",
                   help="run the autonomous optimizer (NSGA-II + deep "
                        "ensemble). Without it — and without --quick / "
                        "--resume / --rounds / --n-init / --batch-size — "
                        "the CLI sizes ONE nominal design and evaluates it")
    b.add_argument("--rounds", type=int, default=None,
                   help="active-learning rounds (default 5); implies "
                        "--optimize")
    b.add_argument("--n-init", type=int, default=None,
                   help="initial LHS physical evaluations (default 320); "
                        "implies --optimize")
    b.add_argument("--batch-size", type=int, default=None,
                   help="acquisition batch per round (default 14); implies "
                        "--optimize")
    b.add_argument("--workers", type=int, default=1,
                   help="parallel processes for the expensive fidelities")
    b.add_argument("--quick", action="store_true",
                   help="quick smoke run (n_init=100, rounds=2, batch=8)")
    b.add_argument("--seed", type=int, default=None,
                   help=f"random seed of the whole run (default {SEED}; a "
                        "resumed run keeps its checkpoint's seed)")
    b.add_argument("--resume", nargs="?", const="__auto__", default=None,
                   metavar="CHECKPOINT",
                   help="warm-start from a phycb_run.json checkpoint "
                        "(without a value: <outdir>/phycb_run.json); the "
                        "spec is taken from the checkpoint")

    x = p.add_argument_group("sibling input (Phy-AC / Phy-CC)")
    x.add_argument("--from-run", type=str, default=None, metavar="RUN_DIR",
                   help="read the design point from a sibling run directory "
                        "(runs/<n>/ of Phy-AC or Phy-CC, or the geometry "
                        "JSON itself). P3/T3/mdot come from the compressor "
                        "contract; T4 stays yours (--t4)")

    f = p.add_argument_group("fidelity")
    f.add_argument("--fidelity", choices=list(FIDELITY_BY_NAME), default="L0",
                   help="fidelity of EVERY EXPENSIVE PHYSICAL EVALUATION: "
                        "the initial LHS, each acquisition batch and the "
                        "final verification (default L0). The inner NSGA-II "
                        "generations always run over the surrogate, never "
                        "over the physics. L2B is out of scope (contract "
                        "§12): it is accepted and degrades with a tagged "
                        "'source'")
    f.add_argument("--search-fidelity", choices=list(FIDELITY_BY_NAME),
                   default=None,
                   help="decouple the SEARCH from the VERIFICATION: explore "
                        "at this level and still verify the winner at "
                        "--fidelity. `--search-fidelity L0` buys back the "
                        "cheap loop when --fidelity is L15 or L2A, where a "
                        "full LHS would take hours. Defaults to --fidelity")

    d = p.add_argument_group("direct evaluation / existing checkpoint")
    d.add_argument("--eval-theta", "--theta", dest="eval_theta", type=str,
                   default=None, metavar="V1,V2,...",
                   help="evaluate ONE given design and emit the full "
                        f"deliverables. Accepts 16 design values or {NDIM} "
                        "(full theta; the operating point lives at "
                        "positions 16-19)")
    d.add_argument("--from-checkpoint", type=str, default=None,
                   help="checkpoint for --list-pareto / --pareto-pick "
                        "(default <outdir>/phycb_run.json)")
    d.add_argument("--list-pareto", action="store_true",
                   help="print the verified Pareto front of a checkpoint "
                        "and exit")
    d.add_argument("--pareto-pick", type=int, default=None, metavar="N",
                   help="regenerate the deliverables for point #N of the "
                        "checkpoint's Pareto front (see --list-pareto); "
                        "written to <outdir>/pareto_NN/")

    o = p.add_argument_group("outputs")
    o.add_argument("--outdir", type=str, default="phycb_out",
                   help="run directory (default phycb_out)")
    o.add_argument("--stl", action="store_true",
                   help="build the printable STLs through layer 5c "
                        "(C#/PicoGK, needs the .NET SDK)")
    o.add_argument("--voxel", type=float, default=None,
                   help=f"voxel size in mm for the STLs (default "
                        f"{VOXEL_DEFAULT_MM}; 0.8 draft, 0.3-0.4 final; "
                        "cost ~ 1/voxel^3); implies --stl")
    o.add_argument("--step", action="store_true",
                   help="export STEP for re-CAD (optional CadQuery)")
    o.add_argument("--step-mode", choices=["parts", "assembly"],
                   default="parts",
                   help="STEP scope: one file per physical part (default) "
                        "or a single assembly")
    o.add_argument("--sweep", choices=["phi_pz", "load"], default=None,
                   help="the combustor 'map' (WAVE4 §4): sweep of the "
                        "primary-zone equivalence ratio (phi_pz) or of the "
                        "load line (mass flow with the geometry frozen)")
    o.add_argument("--hifi-pairs", type=str, default=None, metavar="FILE",
                   help="JSON list of CFD/rig calibration pairs "
                        "[{\"theta\": [...], \"eta_comb\": ..., ...}, ...]; "
                        "affine per-output correction (identity with <2 "
                        "pairs)")
    o.add_argument("--no-figures", action="store_true",
                   help="skip the matplotlib figures (report keeps its "
                        "native SVGs)")
    o.add_argument("--no-report", action="store_true",
                   help="skip the HTML report (layer 5b)")

    m = p.add_argument_group("modes (mutually exclusive with the design "
                             "path)")
    m.add_argument("--analyze", metavar="CASE", default=None,
                   help="predict a PUBLISHED case from validation/cases.py "
                        "(A1, B1, C1, D1, ...); 'list' prints the catalogue")
    m.add_argument("--p3-override", type=float, default=None,
                   help="--analyze: move the operating point off the "
                        "published one")
    m.add_argument("--t3-override", type=float, default=None)
    m.add_argument("--mdot-override", type=float, default=None)
    m.add_argument("--t4-override", type=float, default=None)
    m.add_argument("--info", action="store_true",
                   help="environment, dependencies, fidelity ladder and "
                        "provisional data")

    u = p.add_argument_group("interface")
    u.add_argument("--interactive", "-i", action="store_true",
                   help="wizard that asks for the whole configuration "
                        "(spec, fidelity, budget and outputs); any flag "
                        "given on the command line PRE-SEEDS it")
    u.add_argument("--no-color", action="store_true",
                   help="disable color and ASCII art (also NO_COLOR env)")
    u.add_argument("--quiet", "-q", action="store_true",
                   help="milestones only")
    u.add_argument("--strict", action="store_true",
                   help="an announced degradation is a FAILURE (exit 4)")
    u.add_argument("--freeze-anchors", action="store_true",
                   help="RE-FREEZE the REF_* regression anchors with the "
                        "current code (validation/anchors.json). A "
                        "deliberate act: cite in docs/PENDIENTES_"
                        "Lverificar.md the correction that moved them")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def parse_fixed(items) -> dict:
    """['swirl_SN=0.8', 't_liner_mm=1.0'] -> {'swirl_SN': 0.8, ...}.
    Name validation belongs to DesignSpec (single source of truth)."""
    fixed = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--fix expects VAR=VALUE, got '{it}'")
        name, _, raw = it.partition("=")
        try:
            fixed[name.strip()] = float(raw)
        except ValueError:
            raise ValueError(f"--fix {name}: non-numeric value '{raw}'")
    return fixed


def wants_optimization(args) -> bool:
    """Did the engineer ask for the autonomous optimizer?

    Without any of these signals the CLI does what WAVE1 §3 says: it sizes
    ONE design with the nominal rules and evaluates it (~1 s). That is the
    difference between "give me a number now" and "spend 500 evaluations"."""
    return bool(args.optimize or args.quick or args.resume is not None
                or args.rounds is not None or args.n_init is not None
                or args.batch_size is not None)


def build_spec(args, log=print) -> tuple[DesignSpec, dict]:
    """Builds the `DesignSpec` and the report's `integration` block.

    Priority: `--from-run` (sibling contract) < flags < `--spec-file`.
    """
    info: dict = {}
    if args.from_run:
        base, info = cycle_seam.spec_from_run(
            args.from_run, T4_K=args.t4, fuel=args.fuel,
            dp_max=args.dp_max, material=args.material, arch=args.arch,
            log=log)
        # Only the operating point comes from the sibling; everything else
        # is the engineer's call. The whole spec is rebuilt (no patching of
        # private attributes behind its back).
        spec = DesignSpec(
            P3_Pa=base.P3_Pa, T3_K=base.T3_K, mdot_kgs=base.mdot_kgs,
            T4_K=base.T4_K, fuel=args.fuel, dp_max=args.dp_max,
            arch=args.arch, material=args.material,
            material_casing=args.material_casing, PF_max=args.pf_max,
            envelope_D_mm=args.envelope_d, envelope_L_mm=args.envelope_l,
            dfam=not args.no_dfam, fixed_vars=parse_fixed(args.fix))
        if args.spec_file:
            spec = _spec_from_file(args.spec_file, spec)
        return spec, info

    kw = dict(P3_Pa=_or(args.p3, DEFAULT_OP["p3"]),
              T3_K=_or(args.t3, DEFAULT_OP["t3"]),
              mdot_kgs=_or(args.mdot, DEFAULT_OP["mdot"]),
              T4_K=_or(args.t4, DEFAULT_OP["t4"]),
              fuel=args.fuel, dp_max=args.dp_max, arch=args.arch,
              material=args.material, material_casing=args.material_casing,
              PF_max=args.pf_max, envelope_D_mm=args.envelope_d,
              envelope_L_mm=args.envelope_l, dfam=not args.no_dfam,
              fixed_vars=parse_fixed(args.fix))
    if args.spec_file:
        with open(args.spec_file, encoding="utf-8") as fh:
            kw.update(json.load(fh))
    return DesignSpec(**kw), info


def _spec_from_file(path: str, base: DesignSpec) -> DesignSpec:
    """`--spec-file` on top of an already built spec (`--from-run` path)."""
    kw = {k: v for k, v in vars(base).items() if not k.startswith("_")}
    with open(path, encoding="utf-8") as fh:
        kw.update(json.load(fh))
    return DesignSpec(**kw)


def _or(value, default):
    return default if value is None else value


def _f(rec: dict, key: str, default: float = float("nan")) -> float:
    """A float that is ALWAYS formattable: a missing key or a `None` (e.g.
    `SMD_um` in micromix, contract §4) cannot take down the final panel."""
    try:
        v = rec.get(key)
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def warn_fixed_out_of_bounds(spec: DesignSpec) -> None:
    """Pinning outside the optimizer bounds is legal (the physics tolerates
    it) but deserves a warning: the surrogate extrapolates there."""
    for name, val in spec.fixed_vars.items():
        i = VAR_NAMES.index(name)
        if not (BOUNDS_LO[i] <= val <= BOUNDS_HI[i]):
            print("  " + ui.warn(f"--fix {name}={val:g} outside the "
                                 f"optimizer bounds [{BOUNDS_LO[i]:g}, "
                                 f"{BOUNDS_HI[i]:g}] — the surrogate will "
                                 "extrapolate"))


# ==========================================================================
# 2. Nominal design (WAVE1 §3): house rules, no optimizer
# ==========================================================================
#: Nominal injector per fuel (gate theta[10]). Jet-A inherits the KJ66
#: anchor's vaporizer; H2 goes to micromix (its dedicated branch, Ciencia
#: §7.4); propane reaches the dome already vaporised, so airblast is the
#: neutral choice (the gas branch is forced anyway).
NOMINAL_INJECTOR = {"jet_a": 2.0, "propane": 0.0, "h2": 3.0}


def nominal_theta(spec: DesignSpec) -> np.ndarray:
    """Nominal theta: `THETA_REF` anchor + the layer-0 air-split table +
    the per-fuel injector; the operating point and the `fixed_vars` are
    imposed by the spec (`fix_operating_point`).

    This is WAVE1 §3's "nominal rules" mode: it optimises nothing, it sizes
    with the reference values of `data/lefebvre/air_split_by_arch.csv` (row
    `microturbine`, the one that declares `phycb_default=1`).
    """
    theta = THETA_REF.copy()
    split = pc.air_split_reference("microturbine")
    if split:                       # if the table is missing, keep the anchor
        theta[2], theta[3], theta[4] = split["pz"], split["sz"], split["dz"]
    theta[10] = NOMINAL_INJECTOR.get(spec.fuel, theta[10])
    return spec.fix_operating_point(theta)


# ==========================================================================
# 3. Interactive wizard
# ==========================================================================
def _ask(prompt: str, default, cast=float):
    try:
        raw = input(ui.c("  ? ", "brand", bold=True)
                    + prompt + ui.c(f" [{default}]: ", "dim")).strip()
    except EOFError:
        print()
        return default
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print("  " + ui.warn(f"invalid value, using {default}"))
        return default


def _ask_choice(prompt: str, options, default):
    try:
        raw = input(ui.c("  ? ", "brand", bold=True)
                    + f"{prompt} ({', '.join(options)})"
                    + ui.c(f" [{default}]: ", "dim")).strip()
    except EOFError:
        print()
        return default
    if not raw:
        return default
    if raw in options:
        return raw
    print("  " + ui.warn(f"unknown value, using {default}"))
    return default


def _ask_yes_no(prompt: str, default: bool) -> bool:
    d = "y" if default else "n"
    raw = _ask(f"{prompt} (y/n)", d, str)
    return str(raw).strip().lower().startswith("y")


def run_wizard(args):
    """Asks for the WHOLE configuration, not only the four numbers.

    Every flag already given on the command line arrives here as the
    DEFAULT of its question, so `--fidelity L1 --stl -i` enters the wizard
    with the fidelity and the STL output already chosen and the engineer
    just presses Enter through them. That is the point of allowing flags
    next to `--interactive`: the wizard configures, it does not reset.
    """
    print(ui.rule("Configuration wizard", "accent"))
    print(ui.c("  Press Enter to accept the value in brackets — the flags "
               "you passed are already in there.\n", "dim"))

    print(ui.c("  -- operating point (the four numbers) " + "-" * 30, "dim"))
    args.p3 = _ask("Inlet total pressure P3 [Pa]",
                   _or(args.p3, DEFAULT_OP["p3"]))
    args.t3 = _ask("Inlet total temperature T3 [K]",
                   _or(args.t3, DEFAULT_OP["t3"]))
    args.mdot = _ask("Air mass flow [kg/s]",
                     _or(args.mdot, DEFAULT_OP["mdot"]))
    args.t4 = _ask("Turbine inlet temperature T4 [K]",
                   _or(args.t4, DEFAULT_OP["t4"]))
    args.fuel = _ask_choice("Fuel", list(FUELS), args.fuel)
    args.arch = _ask_choice("Architecture (or 'auto' = design variable)",
                            list(ARCH_NAMES) + ["auto"],
                            args.arch or "auto")
    if args.arch == "auto":
        args.arch = None
    args.dp_max = _ask("Max total pressure loss dp/p", args.dp_max)
    args.pf_max = _ask("Max pattern factor allowed by the turbine",
                       args.pf_max)
    args.material = _ask_choice("Liner (hot wall) material",
                                structures_core.list_materials(),
                                args.material)

    print()
    print(ui.c("  -- fidelity and budget " + "-" * 44, "dim"))
    print(ui.c("     this is the level of EVERY expensive physical "
               "evaluation: the LHS, each acquisition batch and the final "
               "verification.", "dim"))
    print(ui.c("     above L0 it is paid per point, so the optimiser will "
               "quote you the measured cost before spending it.", "dim"))
    args.fidelity = _ask_choice("Fidelity", list(FIDELITY_BY_NAME),
                                args.fidelity)
    mode_default = ("f" if args.optimize else
                    "q" if args.quick or wants_optimization(args) else "n")
    mode = _ask("Mode: [n]ominal (~1 s) / [q]uick optimization / [f]ull "
                "optimization", mode_default, str)
    mode = str(mode).strip().lower()
    args.quick = mode.startswith("q")
    args.optimize = mode.startswith("f")
    if args.optimize:
        args.rounds = int(_ask("Active-learning rounds",
                               _or(args.rounds, BUDGET_FULL["rounds"])))
        args.n_init = int(_ask("Initial LHS physical evaluations",
                               _or(args.n_init, BUDGET_FULL["n_init"])))
    if (args.optimize or args.quick) and args.fidelity != "L0":
        #   The one question that turns an overnight run back into a
        #   coffee-break one. It is only worth asking when there IS a loop
        #   and the rung above L0 would be paid on every point of it.
        cheap = _ask_yes_no(
            f"Explore in L0 and verify only the winner at "
            f"{args.fidelity}? (no = {args.fidelity} on every point)",
            False)
        args.search_fidelity = "L0" if cheap else args.fidelity

    print()
    print(ui.c("  -- outputs " + "-" * 56, "dim"))
    args.outdir = _ask("Run directory", args.outdir, str)
    args.stl = _ask_yes_no("Printable STLs (layer 5c, needs the .NET SDK)?",
                           bool(args.stl or args.voxel is not None))
    if args.stl:
        args.voxel = _ask("Voxel size [mm] (0.8 draft · 0.3-0.4 final)",
                          _or(args.voxel, VOXEL_DEFAULT_MM))
    args.step = _ask_yes_no("STEP export for re-CAD (needs CadQuery)?",
                            bool(args.step))
    sweep = _ask_choice("Sweep (the combustor 'map')",
                        ["none", "phi_pz", "load"], args.sweep or "none")
    args.sweep = None if sweep == "none" else sweep
    print()
    return args


# ==========================================================================
# 4. Styled logger (Phy-AC pattern)
# ==========================================================================
class StyledLog:
    """Colours and formats the optimizer output."""

    def __init__(self, quiet=False):
        self.quiet = quiet

    def __call__(self, msg="") -> None:
        text = str(msg)
        stripped = text.strip()
        m = re.search(r"Round\s+(\d+)/(\d+)", stripped)
        if m:
            n, tot = int(m.group(1)), int(m.group(2))
            print()
            print(ui.rule(f"Round {n}/{tot}", "brand2"))
            print("  " + ui.progress_bar(n / tot, ui.c("exploring...", "dim")))
            return
        if "⚠" in text or "FAILED" in text:
            print("  " + ui.warn(stripped.lstrip("⚠ ").strip()))
            return
        if "Best verified" in stripped:
            print("  " + ui.ok(stripped))
            return
        if stripped and not self.quiet:
            print("    " + ui.c(stripped, "dim"))


# ==========================================================================
# 5. Degradation OUT LOUD, ONCE, AT STARTUP (contract §10.4)
# ==========================================================================
def _have(module: str) -> bool:
    """Is the module installed? `find_spec` does NOT run the import."""
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _dotnet_bin() -> str | None:
    """The dotnet executable, with Phy-AC's full-path fallback."""
    exe = shutil.which("dotnet")
    if exe:
        return exe
    return DOTNET_FALLBACK if os.path.exists(DOTNET_FALLBACK) else None


def probe_environment(args) -> dict:
    """SINGLE probe of the optional dependencies.

    Returns the map of what is there and what is missing. It NEVER raises:
    every absence is a labelled degradation, not an error (contract §10.4).
    """
    fid_req = FIDELITY_BY_NAME[args.fidelity]
    env = dict(fidelity_requested=fid_req)
    #   The SEARCH rung. It follows `--fidelity` (the Phy-CT contract: the
    #   flag governs every expensive evaluation, not just the last one) and
    #   only parts company from it when the engineer says so explicitly.
    env["fidelity_search"] = FIDELITY_BY_NAME[
        getattr(args, "search_fidelity", None) or args.fidelity]
    env["cantera"] = pc.cantera_available()
    env["matplotlib"] = _have("matplotlib") and not args.no_figures
    env["matplotlib_installed"] = _have("matplotlib")
    env["cadquery"] = _have("cadquery")
    env["dotnet"] = _dotnet_bin()
    env["xlb"] = _have("xlb") and _have("jax")
    #  L1.5 does NOT require XLB: `cfd.xlb_cold` delivers the SAME D3Q19 BGK
    #  over three backends — XLB, CuPy (GPU) and NumPy — and labels which
    #  one ran. `env["xlb"]` says whether the first is there; `env["cupy"]`,
    #  whether there is a working GPU (the path that actually runs on this
    #  machine, ~x10 over NumPy); `env["l15"]`, whether the L1.5 fidelity is
    #  reachable at all.
    env["cupy"] = False
    env["gpu"] = {}
    try:
        from cfd import xlb_cold as _xc                   # noqa: PLC0415
        env["cupy"] = bool(_xc.cupy_available())
        env["gpu"] = _xc.gpu_info()
    except Exception:
        pass
    env["l15"] = os.path.isfile(os.path.join(HERE, "cfd", "crn_from_field.py"))
    env["ibrans"] = os.path.isfile(os.path.join(HERE, "cfd", "ibrans",
                                                "__init__.py"))
    env["crn"] = False
    env["l1_timeout_s"] = None
    if int(fid_req) >= int(Fidelity.L1):
        # crn_core drags Cantera in: only touched when it is really needed
        try:
            import crn_core
            env["crn"] = bool(getattr(crn_core, "CANTERA_AVAILABLE", False))
            env["l1_timeout_s"] = float(
                getattr(crn_core, "PHYCB_L1_TIMEOUT", 60.0))
        except Exception:
            env["crn"] = False

    # --- the REAL fidelity the run will use ------------------------------
    eff, why = _degrade_ladder(fid_req, env)
    env["fidelity_effective"] = eff
    env["fidelity_reason"] = "; ".join(why)
    eff_s, why_s = _degrade_ladder(env["fidelity_search"], env)
    env["fidelity_search_effective"] = eff_s
    env["fidelity_search_reason"] = "; ".join(why_s)
    return env


def _degrade_ladder(fid_req, env) -> tuple:
    """Walks one requested rung down to what this machine can really run.

    The ladder degrades in CASCADE and ALL reasons accumulate: saying only
    the last one would hide that L2b was already out of reach.
    """
    eff, why = fid_req, []
    if int(fid_req) >= int(Fidelity.L2B):
        eff = Fidelity.L2A
        why.append("L2b is out of scope (contract §12)")
    if int(eff) >= int(Fidelity.L2A) and not env["ibrans"]:
        eff = Fidelity.L15
        why.append("cfd.ibrans missing (WAVE6)")
    if int(eff) >= int(Fidelity.L15) and not env["l15"]:
        eff = Fidelity.L1
        why.append("cfd.crn_from_field missing (WAVE5-A)")
    elif int(eff) >= int(Fidelity.L15) and not env["xlb"]:
        why.append("L1.5 without XLB/JAX: the same D3Q19 runs on "
                   + ("CuPy/GPU" if env.get("cupy") else "NumPy (CPU)")
                   + " — same physics, different engine")
    if int(eff) >= int(Fidelity.L1) and not env["crn"]:
        eff = Fidelity.L0
        why.append("Cantera/crn_core not available")
    return eff, why


def announce_environment(env: dict, args) -> None:
    """Said ONCE, at startup, out loud — never per worker."""
    print(ui.rule("Environment and declared degradation", "brand"))
    ok, warn = ui.ok, ui.warn
    print("  " + (ok("Cantera        · real equilibrium and L1 CRN "
                     "available")
                  if env["cantera"] else
                  ui.degraded("Cantera", "not installed — L0 falls back to "
                                         "the data/thermo fit (the record's "
                                         "`source` says so)")))
    if args.no_figures:
        print("  " + ui.c("· matplotlib     · figures disabled by "
                          "--no-figures", "dim"))
    else:
        print("  " + (ok("matplotlib     · PNG figures in figures/")
                      if env["matplotlib"] else
                      ui.degraded("matplotlib", "not installed — the report "
                                                "ships with its native SVGs "
                                                "only")))
    if args.step:
        print("  " + (ok("CadQuery       · STEP export available")
                      if env["cadquery"] else
                      ui.degraded("CadQuery", "not installed — no STEP "
                                              "export (pip install "
                                              "cadquery)")))
    else:
        print("  " + (ok("CadQuery       · STEP export available")
                      if env["cadquery"] else
                      ui.c("· CadQuery absent  · no STEP export "
                           "(pip install cadquery)", "dim")))
    if args.stl or args.voxel is not None:
        print("  " + (ok(f"dotnet         · layer 5c (STL) at {env['dotnet']}")
                      if env["dotnet"] else
                      ui.degraded("dotnet", "not found — no printable STLs "
                                            "(install the .NET SDK)")))
    else:
        print("  " + (ok(f"dotnet         · layer 5c (STL) at {env['dotnet']}")
                      if env["dotnet"] else
                      ui.c("· dotnet absent    · no printable STLs "
                           "(install the .NET SDK)", "dim")))
    if env.get("cupy"):
        g = env.get("gpu") or {}
        print("  " + ok(
            f"GPU/CuPy       · L1.5 (cold aero LBM) on {g.get('name', 'GPU')} "
            f"({g.get('vram_free_GiB', float('nan')):.1f} GiB free) — fine "
            "mesh, ~x10 over NumPy"))
    if not env["xlb"]:
        print("  " + ui.c("· XLB/JAX absent   · L1.5 (cold aero LBM) will "
                          "run the same D3Q19 on "
                          + ("CuPy/GPU" if env.get("cupy")
                             else "NumPy on CPU (minutes, not seconds)")
                          if env.get("l15") else
                          "· XLB/JAX absent   · L1.5 (cold aero LBM) will "
                          "degrade", "dim"))
    if not env["ibrans"]:
        print("  " + ui.c("· cfd.ibrans absent · L2a (native IB-RANS) will "
                          "degrade", "dim"))
    elif int(env["fidelity_effective"]) >= int(Fidelity.L2A):
        print("  " + ui.ok(
            "cfd.ibrans      · L2a (reactive IB-RANS) AVAILABLE — full "
            "combustor case, tens of minutes per point; a product for "
            "RANKING (post.HONESTY)"))

    req = Fidelity(env["fidelity_requested"]).name
    eff = Fidelity(env["fidelity_effective"]).name
    if req == eff:
        print("  " + ui.ok(f"Fidelity       · {req} was asked for and {eff} "
                           f"is what will really run"))
    else:
        print("  " + ui.degraded(
            "fidelity", f"{req} was asked for but {eff} is what will really "
                        f"run — {env['fidelity_reason']}"))
        print("    " + ui.c("each record's `source` label will explain WHY "
                            "it degraded (contract §3)", "dim"))
    #   The SEARCH rung only earns a line of its own when it differs from
    #   the verification one: otherwise it would repeat the line above.
    srch = Fidelity(env["fidelity_search_effective"]).name
    if srch != eff:
        print("  " + ui.ok(f"Search         · every expensive evaluation "
                           f"runs {srch}; the winner is verified at {eff}"))
    else:
        print("  " + ui.c(f"· Search           · every expensive evaluation "
                          f"runs {srch} too (--fidelity governs the whole "
                          f"loop)", "dim"))
    prov = pc.provisional_report()
    if prov:
        print("  " + ui.warn(f"{len(prov)} PROVISIONAL coefficient(s) in "
                             "play — see docs/PENDIENTES_Lverificar.md "
                             "(`--info` lists them)"))
    print()


# ==========================================================================
# 6. Frozen regression anchors (contract §10.9)
# ==========================================================================
def _anchor_now() -> dict:
    """Record of the REF_CB10 anchor case with TODAY's code (L0, no cache)."""
    rec = pc.evaluate(THETA_REF.copy(), fidelity=Fidelity.L0,
                      use_cache=False, calibrate=False)
    out = {}
    for k in ANCHOR_KEYS:
        v = rec.get(k)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    out["source"] = str(rec.get("source", ""))
    return out


def check_anchors(freeze: bool, quiet: bool = False) -> None:
    """Compares (or RE-FREEZES) the `REF_*` anchors. Never raises.

    Without `--freeze-anchors` this only COMPARES and warns: moving an
    anchor is a deliberate act that must be cited in
    `docs/PENDIENTES_Lverificar.md`.
    """
    try:
        now = _anchor_now()
        if freeze:
            os.makedirs(os.path.dirname(ANCHOR_PATH), exist_ok=True)
            payload = dict(
                schema="phycb-anchors-1", seed=SEED,
                generated_at=datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                note="Anclas de regresión CONGELADAS. Moverlas exige "
                     "--freeze-anchors y citar en "
                     "docs/PENDIENTES_Lverificar.md la corrección que las "
                     "movió (contrato §10.9).",
                anchors={"REF_CB10": dict(
                    theta=THETA_REF.tolist(), fidelity="L0", **now)})
            with open(ANCHOR_PATH, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, ensure_ascii=False)
            print("  " + ui.warn(f"ANCHORS RE-FROZEN -> {ANCHOR_PATH}"))
            print("  " + ui.c("deliberate act: cite in "
                              "docs/PENDIENTES_Lverificar.md the correction "
                              "that moved them.", "dim"))
            return
        if not os.path.exists(ANCHOR_PATH):
            return
        with open(ANCHOR_PATH, encoding="utf-8") as fh:
            old = (json.load(fh).get("anchors") or {}).get("REF_CB10") or {}
        moved = []
        for k, v_new in now.items():
            if k not in old or not isinstance(v_new, float):
                continue
            v_old = old[k]
            if not isinstance(v_old, (int, float)):
                continue
            if abs(v_new - float(v_old)) > ANCHOR_RTOL * max(
                    1.0, abs(float(v_old))):
                moved.append((k, float(v_old), v_new))
        if moved:
            print("  " + ui.degraded(
                "regression anchors",
                f"{len(moved)} REF_CB10 anchor(s) MOVED with respect to "
                f"validation/anchors.json"))
            for k, a, b in moved[:8]:
                print("    " + ui.c(f"{k:<14s} {a:.6g} -> {b:.6g}", "warn"))
            print("  " + ui.c("if the change is deliberate: "
                              "--freeze-anchors (and cite it in "
                              "docs/PENDIENTES_Lverificar.md)", "dim"))
        elif not quiet:
            print("  " + ui.ok("REF_CB10 anchors intact "
                               "(validation/anchors.json)"))
    except Exception as exc:                     # never takes the run down
        print("  " + ui.warn(f"anchor check skipped: {type(exc).__name__}"))


# ==========================================================================
# 7. Flywheel export
# ==========================================================================
def export_dataset_csv(run_json: str, csv_path: str) -> int:
    """checkpoint -> `dataset.csv` (theta + OUT_KEYS + g + source columns)."""
    if not os.path.exists(run_json):
        return 0
    try:
        with open(run_json, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return 0
    rows = data.get("dataset", [])
    if not rows:
        return 0
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(VAR_NAMES + OUT_KEYS + ["g", "source"])
        for r in rows:
            w.writerow(list(r["theta"]) + [r.get(k) for k in OUT_KEYS]
                       + [json.dumps(r.get("g")), r.get("source")])
    return len(rows)


# ==========================================================================
# 8. Layer 5c — STL by subprocess (EXACT phyac_cli.generate_stls pattern)
# ==========================================================================
def _example_dll() -> str | None:
    """Path of the layer-5c executable (tolerant to the csproj TFM)."""
    base = os.path.join(EXAMPLE_DIR, "bin", "Debug")
    exact = os.path.join(base, "net9.0-windows", EXAMPLE_DLL)
    if os.path.exists(exact):
        return exact
    try:
        for tfm in sorted(os.listdir(base)):
            cand = os.path.join(base, tfm, EXAMPLE_DLL)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    return None


def generate_stls(contract_json: str, outdir: str, voxel: float) -> bool:
    """`dotnet build <sln>` + `dotnet <Example.dll> <contract> <out> <voxel>`.

    Handoff BY FILE (CONTEXTO §2 invariant 4): no sockets, no P/Invoke. It
    never raises: if the SDK is missing or the build fails it says so and
    the run carries on with the remaining deliverables.
    """
    dotnet_bin = _dotnet_bin()
    if not dotnet_bin:
        print("  " + ui.degraded("STL (layer 5c)",
                                 "'dotnet' not found — install the .NET SDK"))
        return False

    print(ui.c("\n  Building CombustorDesigner...", "dim"))
    try:
        res_build = subprocess.run(
            [dotnet_bin, "build", SLN_PATH, "-c", "Debug"],
            capture_output=True, text=True, cwd=HERE)
        if res_build.returncode != 0:
            print("  " + ui.degraded("STL (layer 5c)",
                                     "CombustorDesigner.sln failed to build"))
            print(res_build.stderr or res_build.stdout)
            return False
    except Exception as e:
        print("  " + ui.degraded("STL (layer 5c)",
                                 f"dotnet build could not run: {e}"))
        return False

    dll_path = _example_dll()
    if not dll_path:
        print("  " + ui.degraded(
            "STL (layer 5c)",
            f"{EXAMPLE_DLL} not found in {EXAMPLE_DIR}/bin/Debug/"))
        return False

    print(ui.c(f"  Generating STLs (voxel={voxel} mm)... this can take "
               "seconds to minutes...", "accent"))
    try:
        res_run = subprocess.run(
            [dotnet_bin, dll_path, contract_json, outdir, str(voxel)],
            capture_output=True, text=True, cwd=HERE)
        if res_run.returncode != 0:
            print("  " + ui.degraded("STL (layer 5c)",
                                     "CombustorDesigner.Example failed"))
            print(res_run.stderr or res_run.stdout)
            return False
        for line in res_run.stdout.splitlines():
            line_s = line.strip()
            if line_s and ("->" in line_s or "[PhyCB]" in line_s
                           or "WARN" in line_s or "NOTE" in line_s):
                print("  " + ui.ok(line_s))
        return True
    except Exception as e:
        print("  " + ui.degraded("STL (layer 5c)", f"{e}"))
        return False


# ==========================================================================
# 9. The combustor "map": phi_PZ and load sweeps (WAVE4 §4)
# ==========================================================================
SWEEP_COLS = ("eta_comb", "dp_over_p", "PF", "T_wall_max_K", "EI_NOx",
              "EI_CO", "phi_pz", "LBO_margin", "tau_ms", "T04_K")


def run_sweep(mode: str, spec: DesignSpec, theta: np.ndarray, outdir: str,
              fidelity=Fidelity.L0, quiet: bool = False) -> str | None:
    """Sweep with FROZEN GEOMETRY — the equivalent of the siblings'
    off-design map (there is no combustor map in v1: WAVE4 §4).

    * `phi_pz`: walks `frac_pz` across its bounds; phi_PZ moves the other
      way. It is the curve that shows where it blows out (LBO) and where it
      gets dirty (NOx).
    * `load`: load line with mdot from 40 % to 120 % of design and P3 scaled
      proportionally (PROXY of a coupled compressor, declared in the CSV
      header): the corrected mass flow mdot*sqrt(T3)/P3 stays constant, so
      what changes is Lefebvre's loading parameter — and with it eta_comb
      and the LBO margin. T3 and T4 are held (constant-TIT line).

    BOUNDED loop of `SWEEP_N` points. Returns the CSV path or `None`.
    """
    rows, head = [], []
    base = np.asarray(theta, dtype=float).copy()
    try:
        if mode == "phi_pz":
            head = ["frac_pz"]
            xs = np.linspace(BOUNDS_LO[2], BOUNDS_HI[2], SWEEP_N)
            for x in xs:
                th = base.copy()
                th[2] = float(x)
                rec = pc.evaluate(th, fidelity=fidelity, use_cache=False)
                rows.append([float(x)] + _sweep_row(rec))
        else:
            head = ["load_frac", "mdot_kgs", "P3_Pa"]
            xs = np.linspace(SWEEP_LOAD_LO, SWEEP_LOAD_HI, SWEEP_N)
            for q in xs:
                th = base.copy()
                th[18] = float(spec.mdot_kgs * q)
                th[16] = float(spec.P3_Pa * q)
                rec = pc.evaluate(th, fidelity=fidelity, use_cache=False)
                rows.append([float(q), th[18], th[16]] + _sweep_row(rec))
    except Exception as exc:
        print("  " + ui.degraded(f"sweep '{mode}'",
                                 f"{type(exc).__name__}: {exc}"))
        return None

    path = os.path.join(outdir, f"sweep_{mode}.csv")
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(f"# Sweep '{mode}' with the geometry FROZEN (the "
                     f"design theta), {SWEEP_N} points.\n")
            if mode == "load":
                fh.write("# DECLARED PROXY: mdot and P3 scale together "
                         "(coupled compressor), T3 and T4 constant => the "
                         "corrected mass flow does not change and what "
                         "moves is Lefebvre's loading parameter.\n")
            else:
                fh.write("# frac_cool is CLOSURE: as frac_pz goes up the "
                         "cooling air goes down (contract §1).\n")
            fh.write("# 'feasible' is the RECORD's (the 12 aero "
                     "constraints of contract §5.1), not the spec's global "
                     "g of 26.\n")
            fh.write(f"# fidelity: {Fidelity(fidelity).name} · seed {SEED} "
                     f"· generated by phycb_cli.py\n")
            w = csv.writer(fh)
            w.writerow(head + list(SWEEP_COLS) + ["feasible", "source"])
            w.writerows(rows)
    except Exception as exc:
        print("  " + ui.degraded(f"sweep '{mode}'",
                                 f"could not write {path}: "
                                 f"{type(exc).__name__}"))
        return None

    if not quiet:
        n_head = len(head)
        print(ui.c(f"\n  Sweep '{mode}' (frozen geometry, "
                   f"{Fidelity(fidelity).name}):", "dim"))
        print(ui.c(f"    {head[0]:>9}  {'eta_comb':>8} {'dp/p':>7} "
                   f"{'PF':>6} {'T_w [K]':>8} {'EI_NOx':>7} {'phi_PZ':>6} "
                   f"{'LBOmg':>6}  feasible", "key"))
        for r in rows:
            feas = r[n_head + len(SWEEP_COLS)]
            mark = ui.c("  OK ", "ok") if feas else ui.c("  -- ", "warn")
            print(f"    {r[0]:9.3f}  {r[n_head]:8.4f} {r[n_head + 1]:7.4f} "
                  f"{r[n_head + 2]:6.3f} {r[n_head + 3]:8.1f} "
                  f"{r[n_head + 4]:7.2f} {r[n_head + 6]:6.3f} "
                  f"{r[n_head + 7]:6.3f} {mark}")
    return path


def _sweep_row(rec: dict) -> list:
    out = []
    for k in SWEEP_COLS:
        v = rec.get(k)
        out.append(float(v) if isinstance(v, (int, float))
                   and math.isfinite(float(v)) else float("nan"))
    out.append(bool(rec.get("feasible")))
    out.append(str(rec.get("source", "")))
    return out


# ==========================================================================
# 10. Violated constraints — the "why it is not feasible", in plain words
# ==========================================================================
def constraint_names(spec: DesignSpec) -> list[str]:
    """Names of the global `g` in the EXACT order of
    `DesignSpec.constraints` (contract §5.4): aero(12) + spec(4) +
    struct(5) + dfam(5)."""
    names = list(pc.CONSTRAINT_NAMES) + list(G_SPEC_NAMES)
    if spec.material is not None:
        names += [f"struct_{n}" for n in G_STRUCT_NAMES]
    if spec.dfam:
        names += [f"dfam_{n}" for n in dfam_checker.G_NAMES]
    return names


#: Hints for the constraints whose FROZEN name does not tell the whole
#: story (the name is ABI, contract §5.1: it is not renamed, it is
#: explained).
G_HINTS = {
    "d_eff": "the name comes from the contract, but g[8] is the MAXIMUM of "
             "effusion diameter < 0.5 mm and the validity band of the "
             "blowing ratio M; here M usually binds (Ciencia §5.1)",
    "mixing": "dilution penetration Y/D_L outside the target band",
    "LBO": "not enough lean-blowout margin: raise frac_pz or lower phi_PZ",
}


def global_constraints(spec: DesignSpec, rec: dict, theta) -> list | None:
    """Global `g` of 26 (contract §5.4), or `None` if not even that could
    be composed. Never raises: it is the final panel's safety net."""
    try:
        return list(spec.constraints(rec, np.asarray(theta, dtype=float)))
    except Exception as exc:
        print("  " + ui.warn(f"the g vector could not be composed: "
                             f"{type(exc).__name__}: {exc}"))
        return None


def print_violations(spec: DesignSpec, rec: dict, theta: np.ndarray,
                     g: list | None = None) -> int:
    """Prints the violated constraints with their magnitude. Returns how
    many.

    `g <= 0` <=> feasible; the magnitude is the degree of violation.
    """
    g = global_constraints(spec, rec, theta) if g is None else g
    if g is None:
        return -1
    names = constraint_names(spec)
    if len(names) != len(g):        # defensive read: never misalign
        names = [f"g[{i}]" for i in range(len(g))]
    bad = [(i, n, float(v))
           for i, (n, v) in enumerate(zip(names, g)) if v > 1e-9]
    if not bad:
        return 0
    bad.sort(key=lambda z: -z[2])
    print()
    print("  " + ui.warn(f"{len(bad)} violated constraint(s) out of "
                         f"{len(g)} — this is why it is NOT feasible:"))
    for i, n, v in bad[:12]:
        print("    " + ui.c(f"g[{i}] {n:<18s} = {v:+.4f}", "warn"))
        hint = G_HINTS.get(n)
        if hint:
            print("      " + ui.c(hint, "dim"))
    if len(bad) > 12:
        print("    " + ui.c(f"... and {len(bad) - 12} more (section 7 of the "
                            "report)", "dim"))
    return len(bad)


def infeasibility_report(spec: DesignSpec, population: list) -> None:
    """WHY there is no feasible design, read off the WHOLE population.

    THE POINT-WISE ANSWER IS THE WRONG ONE, and this is the function that
    exists to say so. Naming the constraint that binds *at the returned
    design* answers "what stopped this particular geometry", which is
    almost never the question. The question is "what stops EVERY
    geometry", and that is a property of the population: a constraint that
    no sampled point ever satisfies is a WALL, and one that most points
    miss but some clear is merely TIGHT. The two want opposite things from
    the engineer — a wall wants the spec changed, a tight constraint wants
    more search budget — so telling them apart is the whole value of the
    message.

    `population` is a list of `(record, theta)` pairs. With fewer than 8
    usable rows there is no population to diagnose and the function says
    nothing: the point-wise list from `print_violations` is all there is.
    """
    names = constraint_names(spec)
    rows = []
    for rec, theta in population or []:
        try:
            g = np.asarray(spec.constraints(rec, np.asarray(theta,
                                                            dtype=float)),
                           dtype=float)
        except Exception:
            continue
        if g.size == len(names):
            rows.append(g)
    if len(rows) < 8:
        return
    G = np.array(rows)
    frac = (G > 0.0).mean(axis=0)
    gmin = G.min(axis=0)

    walls = [i for i in range(len(names)) if frac[i] >= 0.999 and gmin[i] > 0]
    tight = [i for i in range(len(names))
             if i not in walls and frac[i] >= 0.50]

    ui.section("Why there is no feasible design")
    ui.line("population", f"{len(G)} evaluated designs", "",
            "the diagnosis is over ALL of them, not over the one returned")

    if walls:
        print()
        print("  " + ui.c("WALLS", "err", bold=True)
              + ui.dim("  -- not one design in the box satisfies these. "
                       "More search budget will not help; the SPEC has to "
                       "change."))
        for i in sorted(walls, key=lambda k: -gmin[k]):
            ui.line(names[i], f"violated by {100 * frac[i]:.0f} %", "",
                    f"best achievable g = {gmin[i]:+.3f}  (needs <= 0)")
    if tight:
        print()
        print("  " + ui.c("TIGHT", "warn", bold=True)
              + ui.dim("  -- most designs miss these, but some clear them. "
                       "They are what the search is fighting."))
        for i in sorted(tight, key=lambda k: -frac[k])[:6]:
            ui.line(names[i], f"violated by {100 * frac[i]:.0f} %", "",
                    f"best achievable g = {gmin[i]:+.3f}")

    #   DROP ONE. For each constraint: how many designs of the population
    #   would be feasible if THAT one did not exist. It is the question
    #   someone staring at a failed run actually asks — "what is the last
    #   thing between me and a design" — and it is answered neither by the
    #   violation frequency nor by whatever binds at the returned point. A
    #   constraint with a high count here is THE blocker: everything else
    #   is already solved in those points.
    loo = []
    for i in range(len(names)):
        others = np.delete(G, i, axis=1)
        n_ok = int((others.max(axis=1) <= 0.0).sum())
        if n_ok > 0:
            loo.append((n_ok, i))
    loo.sort(reverse=True)
    if loo:
        print()
        print("  " + ui.c("THE LAST BLOCKER", "warn", bold=True)
              + ui.dim("  -- designs that would be feasible if this ONE "
                       "constraint were dropped. Everything else is already "
                       "satisfied in those designs."))
        for n_ok, i in loo[:4]:
            gb = G[(np.delete(G, i, axis=1).max(axis=1) <= 0.0), i].min()
            ui.line(names[i], f"{n_ok} design(s) blocked by it alone", "",
                    f"closest of those sits at g = {gb:+.3f}")

    if loo and loo[0][0] >= 5:
        n_ok, i = loo[0]
        print()
        print("  " + ui.warn(
            f"THIS IS NOT A SEARCH PROBLEM. {n_ok} designs already satisfy "
            f"every other constraint; the single thing standing between you "
            f"and a feasible combustor is {ui.bold(names[i])}. More budget "
            f"will find more of the same."))
        _blocker_advice(spec, names[i])
    elif not walls:
        print()
        print("  " + ui.warn(
            "NO WALLS and no single dominant blocker: every constraint is "
            "clearable on its own, so the spec is reachable in principle — "
            "what is hard is the INTERSECTION. That IS a search problem, "
            "and more budget is the right first move:"))
        ui.line("try", "--optimize --rounds 6 --n-init 400", "",
                "and drop --quick, which caps the search at 100 + 2x8")


#: The concrete advice, when the blocker has an arithmetic way out. Only
#: the constraints whose escape is a SPEC number are listed: the rest is
#: search, and saying otherwise would send the engineer to move a knob
#: that was never the problem.
BLOCKER_ADVICE = {
    "spec_dp_max": ("dp/p is a HARD cycle constraint, not an objective: a "
                    "combustor SPENDS pressure on mixing. If the cycle can "
                    "afford it, raise --dp-max (micro turbines run 2-4 %); "
                    "if it cannot, the mixing has to get cheaper — lower "
                    "swirl_SN or fewer, bigger dilution holes."),
    "spec_PF_max": ("the pattern factor is the TURBINE's allowance, not "
                    "the combustor's: --pf-max is a number your turbine "
                    "gave you. Raising it is a conversation with Phy-CT, "
                    "not a design change. Otherwise lengthen the dilution "
                    "zone (L_dz_over_DL) so the profile has room to flatten."),
    "spec_envelope_D": ("the casing does not fit the envelope. Raise "
                        "--envelope-d, or accept a longer chamber: D and L "
                        "trade against each other at constant residence "
                        "time."),
    "spec_envelope_L": ("the chamber does not fit lengthwise. Raise "
                        "--envelope-l, or accept a wider casing: residence "
                        "time is volume, and volume can be bought either "
                        "way."),
    "LBO": ("lean blowout: the primary zone is too lean to hold the flame. "
            "Raise frac_pz (more air to the PZ is NOT the fix — phi_PZ goes "
            "DOWN with frac_pz), or raise T4 so the global FAR rises."),
    "struct_T_wall": ("the liner runs hotter than the alloy allows. A "
                      "stronger --material, more cooling air (lower "
                      "frac_pz/sz/dz so frac_cool closes higher), or a "
                      "lower T4. This one is arithmetic: the wall does not "
                      "negotiate."),
}


def _blocker_advice(spec: DesignSpec, name: str) -> None:
    advice = BLOCKER_ADVICE.get(name)
    if not advice:
        for key, txt in BLOCKER_ADVICE.items():
            if name.startswith(key):
                advice = txt
                break
    if advice:
        print()
        ui.line("what to change", "", "", advice)
    if name == "spec_dp_max":
        ui.line("this spec asks", f"dp/p <= {spec.dp_max:.4f}", "",
                "a HARD cycle constraint — it is YOUR number, not a "
                "measurement")
    elif name == "spec_PF_max":
        ui.line("this spec asks", f"PF <= {spec.PF_max:.3f}", "",
                "the turbine's allowance — change it with --pf-max only if "
                "the turbine agrees")


# ==========================================================================
# 11. Deliverables (layer 5) — shared by the CLI's three paths
# ==========================================================================
def emit_deliverables(args, spec, result, outdir, env, ckpt=None, logger=None,
                      title="Recommended design", integration=None,
                      numbered=True):
    """Structural + geometry + DfAM + figures + report + dataset + sweep +
    STL + final panel + DELIVERABLES block with the REAL paths.

    `numbered` carries the [3/4] / [4/4] step labels. The design paths run
    all four steps and want them; `--analyze` and `--pareto-pick` skip the
    first two, and a run whose first heading reads [3/4] looks like a run
    that lost something.
    """
    logger = logger or StyledLog(quiet=args.quiet)
    s3 = "[3/4] " if numbered else ""
    s4 = "[4/4] " if numbered else ""
    rec = result["record"]
    theta = np.asarray(result["theta"], dtype=float)
    voxel = args.voxel if args.voxel is not None else VOXEL_DEFAULT_MM
    delivered: list[tuple[str, str]] = []

    # ---- [3/4] Structural + geometry + DfAM -----------------------------
    print()
    print(ui.rule(f"{s3}Geometry, structures and DfAM (layers 1s and 5a)",
                  "brand"))
    structural = None
    if spec.material is not None:
        try:
            structural = structures_core.evaluate_structural(
                theta, rec, material=spec.material,
                material_casing=spec.material_casing)
            s_ok = structural.get("feasible_struct")
            print("  " + (ui.ok if s_ok else ui.warn)(
                f"structural  · {structural.get('material')} · "
                f"g_struct {'OK' if s_ok else 'VIOLATED'}"))
        except Exception as exc:
            print("  " + ui.degraded("structural evaluation",
                                     f"{type(exc).__name__}: {exc}"))

    geo_dir = os.path.join(outdir, "geometry")
    manifest, contract, dfam = None, None, None
    try:
        manifest = geometry_generator.generate(
            theta, rec, geo_dir, structural=structural, voxel_mm=voxel,
            step=bool(args.step), step_mode=args.step_mode,
            log=(lambda m: print("  " + ui.c(str(m).strip(), "dim"))))
        contract, dfam = manifest.get("contract"), manifest.get("dfam")
        print("  " + ui.ok(f"geometry    -> {geo_dir}{os.sep} "
                           f"({len(manifest.get('files') or [])} files, "
                           f"{manifest.get('n_hole_rows')} hole rows)"))
        delivered.append(("geometry", geo_dir + os.sep))
        if dfam is not None:
            d_ok = dfam.get("feasible")
            print("  " + (ui.ok if d_ok else ui.warn)(
                f"DfAM        · orientation '{dfam.get('orientation')}' · "
                f"voxel {voxel} mm · "
                f"{'PRINTABLE' if d_ok else 'NOT printable'}"))
            for v in (dfam.get("violations") or [])[:4]:
                print("    " + ui.c(str(v), "warn"))
        #  Warnings from the geometry package itself — today they come from
        #  the effusion cross-check BETWEEN LAYERS (`validation.effusion`):
        #  a discrepancy between what the physics cools and what the machine
        #  drills CANNOT pass in silence.
        for w in (manifest.get("warnings") or [])[:4]:
            print("    " + ui.warn(str(w)))
        if args.step:
            steps = manifest.get("step") or []
            if steps:
                print("  " + ui.ok(f"STEP        -> {len(steps)} file(s) "
                                   f"({args.step_mode})"))
            else:
                print("  " + ui.degraded("STEP export",
                                         "CadQuery is not installed "
                                         "(pip install cadquery)"))
    except Exception as exc:
        # The contract fails LOUDLY when the polyline does not validate;
        # it is caught here so the report still comes out and explains why.
        print("  " + ui.degraded("geometry package",
                                 f"not written: {type(exc).__name__}: {exc}"))

    # ---- [4/4] Figures + report + dataset -------------------------------
    print()
    print(ui.rule(f"{s4}Report, figures and dataset (layer 5b)", "brand"))
    figs_dir = os.path.join(outdir, "figures")
    figures = []
    if not args.no_figures:
        limit_K = rec.get("T_max_flame_wall_K")
        if limit_K is None and structural:
            limit_K = structural.get("T_wall_allow_K")
        figures = visualization.generate_all(
            rec, figs_dir, front=result.get("pareto_front"),
            contract=contract, material_limit_K=limit_K)
        # CFD figures: they only exist if the run really reached L1.5/L2a.
        # `crn_from_field.LAST_FIELD` holds the full field of the last
        # measurement (it travels in no JSON: it weighs MB).
        try:
            import cfd.crn_from_field as _cff
            _fld = getattr(_cff, "LAST_FIELD", None)
            if _fld:
                _cfd_figs = visualization.generate_cfd(
                    _fld, figs_dir, post=getattr(_cff, "LAST_POST", None),
                    record=rec)
                if _cfd_figs:
                    figures = list(figures) + list(_cfd_figs)
                    print("  " + ui.ok(f"CFD figures -> {len(_cfd_figs)} PNG "
                                       "(cold field, L1.5)"))
        except Exception:
            pass                    # the CFD is optional: never breaks a run

        if figures:
            print("  " + ui.ok(f"figures     -> {figs_dir}{os.sep} "
                               f"({len(figures)} PNG)"))
            delivered.append(("figures", figs_dir + os.sep))
        else:
            print("  " + ui.c("· no figures (matplotlib missing or no data) "
                              "— the report ships with its native SVGs",
                              "dim"))

    if integration:
        result = dict(result)
        result["integration"] = integration
    report = None
    if args.no_report:
        #   A skip the engineer ASKED for is a choice, not a degradation:
        #   registering it would make `--strict` fail on the user's own
        #   decision and drown the flags that really did fall back.
        print("  " + ui.c("· report        · skipped by --no-report",
                          "dim"))
    else:
        report = report_generator.generate_report(
            spec, result, os.path.join(outdir, "report.html"),
            structural=structural, contract=contract, dfam=dfam,
            figures=figures, log=logger)
        print("  " + ui.ok(f"report      -> {report}"))
        delivered.append(("report.html", report))

    if ckpt and os.path.exists(ckpt):
        n_rows = export_dataset_csv(ckpt, os.path.join(outdir, "dataset.csv"))
        print("  " + ui.ok(f"dataset     -> {n_rows} rows (flywheel)"))
        delivered.append(("checkpoint", ckpt))
        delivered.append(("dataset.csv",
                          os.path.join(outdir, "dataset.csv")))
    cache = pc.CACHE_PATH
    if cache and os.path.exists(cache):
        delivered.append(("phys_cache.jsonl", cache))

    # ---- sweep (the combustor "map") ------------------------------------
    if args.sweep:
        # the USER's fidelity is asked for (not the effective one): that way
        # each point's `source` column documents WHY it degraded, contract §3
        sw = run_sweep(args.sweep, spec, theta, outdir,
                       fidelity=env["fidelity_requested"], quiet=args.quiet)
        if sw:
            delivered.append((f"sweep_{args.sweep}.csv", sw))

    # ---- layer 5c: printable STLs ---------------------------------------
    if args.stl or args.voxel is not None:
        contract_path = os.path.join(geo_dir, "combustor.json")
        if os.path.exists(contract_path):
            if generate_stls(contract_path, outdir, voxel):
                stls = sorted(f for f in os.listdir(outdir)
                              if f.lower().endswith(".stl"))
                delivered.append((f"STL ({len(stls)})", outdir + os.sep))
        else:
            print("  " + ui.degraded("STL (layer 5c)",
                                     f"{contract_path} does not exist"))

    # ---- final panel ----------------------------------------------------
    # The status that rules is the GLOBAL `g` of 26 (aero + spec + struct +
    # DfAM), not the record's `feasible` (the 12 aero ones only): a design
    # can be aerodynamically sound and violate the envelope or the
    # manufacturability.
    g_total = global_constraints(spec, rec, theta)
    feasible_aero = bool(rec.get("feasible"))
    feasible = (feasible_aero if g_total is None
                else bool(DesignSpec.violation(np.asarray(g_total)) <= 1e-9))
    status = (ui.c(" FEASIBLE ", "ok", bold=True) if feasible
              else ui.c(" INFEASIBLE ", "err", bold=True))
    front = result.get("pareto_front") or []
    n_feas = sum(1 for q in front if q.get("feasible"))
    rows = [
        ("eta_comb", ui.c(f"{_f(rec, 'eta_comb'):.4f}", "val", bold=True)),
        ("dp/p", f"{_f(rec, 'dp_over_p'):.4f}"
                 + ui.c(f"   (max {spec.dp_max:.3f})", "dim")),
        ("Pattern factor", f"{_f(rec, 'PF'):.3f}"
                           + ui.c(f"   (max {spec.PF_max:.2f})", "dim")),
        ("Max wall T", f"{_f(rec, 'T_wall_max_K'):.0f} K"),
        ("EI_NOx / EI_CO", f"{_f(rec, 'EI_NOx'):.2f} / "
                           f"{_f(rec, 'EI_CO'):.2f} g/kg"),
        ("phi_PZ · tau", f"{_f(rec, 'phi_pz'):.3f} · "
                         f"{_f(rec, 'tau_ms'):.2f} ms"),
        ("Envelope", f"D {_f(rec, 'D_casing_mm'):.1f} x "
                     f"{_f(rec, 'L_tot_mm'):.1f} mm"),
        ("T04 / P04", f"{_f(rec, 'T04_K'):.1f} K / "
                      f"{_f(rec, 'P04_Pa'):,.0f} Pa"),
        ("Bleed required",
         f"{100.0 * _f(rec, 'bleed_frac_required'):.2f} % of mdot3"),
        ("Fidelity", ui.c(str(rec.get("source", "?")), "brand")),
        ("Status", status),
    ]
    if feasible_aero != feasible:
        rows.append(("Aero status (12)",
                     ui.c(" FEASIBLE ", "ok") if feasible_aero
                     else ui.c(" INFEASIBLE ", "warn")))
    if front:
        rows.append(("Pareto",
                     f"{len(front)} point{'s' if len(front) != 1 else ''} "
                     f"({n_feas} feasible)"))
    if structural:
        s_ok = structural.get("feasible_struct")
        rows.append(("Structure",
                     (ui.c(" MARGINS OK ", "ok", bold=True) if s_ok
                      else ui.c(" MARGIN VIOLATED ", "err", bold=True))
                     + ui.c(f"  {structural.get('material')}", "dim")))
    if dfam is not None:
        d_ok = dfam.get("feasible")
        rows.append(("DfAM",
                     (ui.c(" PRINTABLE ", "ok", bold=True) if d_ok
                      else ui.c(" NOT PRINTABLE ", "err", bold=True))
                     + ui.c(f"  voxel {voxel} mm · "
                            f"{dfam.get('orientation')}", "dim")))
    print()
    print(ui.panel(title, rows, style="ok" if feasible else "warn"))

    if not feasible:
        print_violations(spec, rec, theta, g=g_total)

    # ---- DELIVERABLES ---------------------------------------------------
    print()
    print(ui.rule("DELIVERABLES", "accent"))
    for label, path in delivered:
        # padding is computed over the PLAIN text: ANSI codes take no
        # columns on screen but they do take len()
        print("  " + ui.c(f"{label + ':':<26}", "key") + " "
              + ui.c(path, "val"))
    print()
    if report:
        print("  " + ui.c("Open the report:", "dim") + " "
              + ui.c(os.path.abspath(report), "brand", bold=True))
        print()
    return dict(report=report, feasible=feasible, structural=structural,
                contract=contract, dfam=dfam)


# ==========================================================================
# 12. Pareto mode over an existing checkpoint
# ==========================================================================
def run_pareto_mode(args, env) -> int:
    """--list-pareto / --pareto-pick: works over an already written
    checkpoint without re-optimising. The front is re-verified with the
    physics (L0)."""
    ck = args.from_checkpoint or os.path.join(args.outdir, "phycb_run.json")
    if not os.path.exists(ck):
        print("  " + ui.err(f"checkpoint not found: {ck} "
                            "(use --from-checkpoint)"))
        return 2
    # the persistent cache matters here too: --pareto-pick at L1 is expensive
    set_cache_path(os.environ.get("PHYCB_CACHE")
                   or os.path.join(args.outdir, "phys_cache.jsonl"))
    try:
        spec, front = pareto_from_checkpoint(ck)
    except Exception as exc:
        print("  " + ui.err(f"unreadable checkpoint: {type(exc).__name__}: "
                            f"{exc}"))
        return 2
    front.sort(key=lambda q: (not q["feasible"], -q["eta_comb"],
                              q["EI_NOx"]))
    print(ui.rule("Verified Pareto front", "brand"))
    print(ui.c(f"  {ck} — {len(front)} "
               f"point{'s' if len(front) != 1 else ''}\n", "dim"))
    print(ui.c(f"  {'#':>3}  {'feas':^4} {'architecture':<18} "
               f"{'eta_comb':>8} {'dp/p':>7} {'PF':>6} {'T_w [K]':>8} "
               f"{'EI_NOx':>7}  source", "key"))
    for i, q in enumerate(front):
        feas_txt = (ui.c("  OK ", "ok") if q["feasible"]
                    else ui.c("  -- ", "warn"))
        print(f"  {i:>3} {feas_txt} {str(q.get('arch') or '?'):<18} "
              f"{q['eta_comb']:8.4f} {q['dp_over_p']:7.4f} {q['PF']:6.3f} "
              f"{q['T_wall_max_K']:8.1f} {q['EI_NOx']:7.2f}  "
              f"{q.get('source', '')}")
    if args.pareto_pick is None:
        print(ui.c("\n  deliverables of one point:  --pareto-pick N "
                   "[--stl] [--sweep phi_pz]", "dim"))
        return 0
    n = args.pareto_pick
    if not 0 <= n < len(front):
        print("  " + ui.err(f"--pareto-pick {n} out of range "
                            f"0..{len(front) - 1}"))
        return 2
    result = evaluate_design(spec, np.array(front[n]["theta"]),
                             fidelity=env["fidelity_requested"])
    outdir = os.path.join(args.outdir, f"pareto_{n:02d}")
    os.makedirs(outdir, exist_ok=True)
    print()
    print("  " + ui.ok(f"Pareto point #{n} -> {outdir}{os.sep}"))
    out = emit_deliverables(args, spec, result, outdir, env, ckpt=None,
                            title=f"Pareto point #{n}", numbered=False)
    return 0 if out["feasible"] else 3


# ==========================================================================
# 13. --analyze: predict a PUBLISHED case (it synthesises NOTHING)
# ==========================================================================
def _load_cases():
    """Loads `validation/cases.py` without turning it into a package."""
    path = os.path.join(HERE, "validation", "cases.py")
    spec = importlib.util.spec_from_file_location("phycb_cases", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case_label(case: dict) -> str:
    return CASE_LABELS.get(case.get("cid", ""), str(case.get("name", "")))


def _print_case_catalogue(cases_mod) -> None:
    print(ui.rule("Published cases available to --analyze", "brand"))
    print(ui.c("  the dataset is validation/cases.py; the campaign and its "
               "verdicts live in docs/VALIDATION.md\n", "dim"))
    print(ui.c(f"  {'id':<5}{'lvl':<5}{'analysable':<12}description", "key"))
    for c in cases_mod.CASES:
        ok_ = "spec" in c
        print(f"  {c['cid']:<5}{c.get('level', '?'):<5}"
              + (ui.c("yes", "ok") + " " * 9 if ok_
                 else ui.c("no", "warn") + " " * 10)
              + _case_label(c))
    print()
    print("  " + ui.c("'no' means the case declares no theta (it grades a "
                      "table, not a machine): there is nothing to predict.",
                      "dim"))


def run_analyze(args, env) -> int:
    """Takes a PUBLISHED case and only PREDICTS it.

    A parameteriser is not an analyser. The default path SYNTHESISES a
    combustor from a spec; this one reads the theta a published source
    pins down and reports what the model says about it. The two must not
    mix: grading a model against a machine it also designed is how a
    campaign turns into a mirror.
    """
    try:
        cases_mod = _load_cases()
    except Exception as exc:
        print("  " + ui.err(f"validation/cases.py could not be loaded: "
                            f"{type(exc).__name__}: {exc}"))
        return 2

    if str(args.analyze).strip().lower() in ("list", "?", "help"):
        _print_case_catalogue(cases_mod)
        return 0

    cid = str(args.analyze).strip().upper()
    case = next((c for c in cases_mod.CASES if c["cid"].upper() == cid), None)
    if case is None:
        print("  " + ui.err(f"unknown case '{args.analyze}'"))
        print()
        _print_case_catalogue(cases_mod)
        return 2
    if "spec" not in case:
        print("  " + ui.err(
            f"{cid} declares no theta: it grades a data table, not a "
            "machine, so there is nothing to predict."))
        print("  " + ui.c("run `--analyze list` for the analysable cases, "
                          "or `python validation/validate.py` for the whole "
                          "campaign.", "dim"))
        return 2

    theta = np.array(cases_mod.theta_from_spec(case["spec"]), dtype=float)
    overrides = []
    for idx, val, name in ((16, args.p3_override, "P3"),
                           (17, args.t3_override, "T3"),
                           (18, args.mdot_override, "mdot"),
                           (19, args.t4_override, "T4")):
        if val is not None:
            overrides.append(f"{name}: {theta[idx]:g} -> {val:g}")
            theta[idx] = float(val)

    gates = pc.decode_gates(theta)
    spec = DesignSpec(P3_Pa=float(theta[16]), T3_K=float(theta[17]),
                      mdot_kgs=float(theta[18]), T4_K=float(theta[19]),
                      fuel=gates["fuel"], dp_max=args.dp_max,
                      arch=gates["arch"],
                      material=case.get("material", args.material),
                      material_casing=args.material_casing,
                      PF_max=args.pf_max, dfam=not args.no_dfam)

    print(ui.rule(f"Analysis of published case {cid}", "brand"))
    ui.line("case", _case_label(case), "", f"level {case.get('level', '?')}")
    if case.get("level") in LEVEL_WHAT:
        ui.line("what this level grades", "", "", LEVEL_WHAT[case["level"]])
    ui.line("kind", str(case.get("kind", "-")))
    ui.line("operating point",
            f"P3 = {theta[16]:,.0f} Pa   T3 = {theta[17]:.1f} K", "",
            f"mdot = {theta[18]:g} kg/s   T4 = {theta[19]:.1f} K")
    ui.line("gates", f"{gates['arch']} · {gates['injector']} · "
                     f"{gates['fuel']}", "",
            "architecture · injector · fuel (contract §1)")
    ui.line("theta", f"{len(case['spec'])} value(s) pinned by the case", "",
            "everything else stays at THETA_BASE (= THETA_REF, contract §1)")
    for k, v in sorted(case["spec"].items()):
        ui.line(f"  {k}", f"{float(v):g}", "", "", width_=24)
    if overrides:
        print()
        print("  " + ui.warn("operating point moved OFF the published one: "
                             + "; ".join(overrides)))
        print("  " + ui.c("what comes out is no longer comparable with the "
                          "source. It is a what-if, not a validation.",
                          "dim"))
    print()
    print("  " + ui.warn("PREDICTION ONLY — nothing here is synthesised or "
                         "fitted to this case. The declared assumptions and "
                         "sources of the case are quoted verbatim in "
                         "validation/cases.py and graded in "
                         "docs/VALIDATION.md."))
    n_assump = len(case.get("assumptions") or [])
    if n_assump:
        ui.line("declared assumptions", f"{n_assump}", "",
                f"validation/cases.py :: {cid} — read them before quoting "
                f"any number below")

    try:
        result = evaluate_design(spec, theta,
                                 fidelity=env["fidelity_requested"])
    except Exception as exc:
        print("  " + ui.err(f"evaluation failed: {type(exc).__name__}: "
                            f"{exc}"))
        return 2

    outdir = args.outdir if args.outdir != "phycb_out" \
        else os.path.join("phycb_out", f"analyze_{cid}")
    os.makedirs(outdir, exist_ok=True)
    set_cache_path(os.environ.get("PHYCB_CACHE")
                   or os.path.join(outdir, "phys_cache.jsonl"))
    out = emit_deliverables(args, spec, result, outdir, env, ckpt=None,
                            title=f"Prediction for {cid}", numbered=False)
    #  Feasibility is a SYNTHESIS verdict: a published machine is not
    #  graded by whether it clears our design constraints, so the analysis
    #  path never returns 3 for it.
    print("  " + ui.c("note: the FEASIBLE/INFEASIBLE badge above is a "
                      "design verdict against OUR constraints. A published "
                      "machine is not judged by it — the campaign verdict "
                      "is in docs/VALIDATION.md.", "dim"))
    return 0 if out is not None else 2


def _asked_for_a_job(argv) -> bool:
    """Did this command line say WHAT to compute?

    An empty command line obviously did not — that is the classic "no
    arguments = wizard". But neither did `--fidelity L15`, `--stl` or
    `--outdir runs/x`: those choose HOW the job is done and leave the job
    itself unstated. Treating them as a full request is how an engineer who
    meant to configure the wizard ends up watching a default design run at
    a fidelity they never got to pair with a spec.

    So the wizard opens for both, PRE-SEEDED with whatever was passed, and
    an explicit `--interactive` still forces it even next to a full spec.
    """
    return any(t == f or t.startswith(f + "=")
               for t in argv for f in ACTION_FLAGS)


def reject_synthesis_flags(argv) -> list[str]:
    """`--analyze` reads a published case and parameterises nothing.
    Accepting a synthesis flag next to it would suggest it does, so the
    flag is rejected instead of silently ignored. Returns the offenders
    (empty = the combination is legal)."""
    return [f for f in SYNTHESIS_FLAGS
            if any(t == f or t.startswith(f + "=") for t in argv)]


# ==========================================================================
# 14. --info: environment, ladder and provisional data
# ==========================================================================
def _dataset_status() -> list[tuple[str, str]]:
    """(relative path, @status) of every data CSV. Cheap: only headers."""
    out = []
    root = os.path.join(HERE, "data")
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".csv"):
                continue
            p = os.path.join(dirpath, fn)
            status = "unknown"
            try:
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.startswith("#"):
                            break
                        if "@status:" in line:
                            status = line.split("@status:", 1)[1].strip()
                            break
            except Exception:
                pass
            out.append((os.path.relpath(p, HERE).replace("\\", "/"), status))
    return sorted(out)


#: `@status` -> traceability mark of the family.
STATUS_MARK = {"verified": "V", "provisional": "?", "unknown": "?"}


def run_info(args) -> int:
    env = probe_environment(args)
    ui.title("QUASAR Phy-CB - environment")
    ui.line("record schema", pc.SCHEMA)
    ui.line("physics module", pc.MODULE_SCHEMA)
    ui.line("geometry contract", geometry_generator.SCHEMA)
    ui.line("theta dimension", NDIM, "",
            "16 design variables + the 4-number operating point")
    ui.line("seed", SEED, "", "the whole run is deterministic (contract §10.8)")
    ui.line("architectures", ", ".join(ARCH_NAMES))
    ui.line("fuels", ", ".join(FUELS))
    ui.line("materials", ", ".join(structures_core.list_materials()))

    ui.section("Optional dependencies (they degrade WITH A WARNING)")
    ui.line("numpy", np.__version__)
    ui.line("Cantera (L1 CRN)", ui.status(env["cantera"], "YES", "NO"), "",
            "real equilibrium and reactor network" if env["cantera"]
            else "L0 falls back to the data/thermo fit, labelled in `source`")
    ui.line("matplotlib (PNG)",
            ui.status(env["matplotlib_installed"], "YES", "NO"), "",
            "figures/*.png" if env["matplotlib_installed"] else
            "the HTML report does NOT need it: its figures are SVG")
    ui.line("CadQuery (STEP)", ui.status(env["cadquery"], "YES", "NO"), "",
            "`--step` writes the STEP files" if env["cadquery"]
            else "no STEP; the rest of the contract is complete")
    ui.line("dotnet + PicoGK (STL)",
            ui.status(bool(env["dotnet"]), "YES", "NO"), "",
            str(env["dotnet"]) if env["dotnet"]
            else "install the .NET SDK, then `dotnet build "
                 "CombustorDesigner.sln`")
    ui.line("CuPy / GPU (L1.5)", ui.status(env["cupy"], "YES", "NO"), "",
            (f"{(env.get('gpu') or {}).get('name', 'GPU')} — same D3Q19, "
             "~x10 over NumPy") if env["cupy"] else
            "the LBM falls back to NumPy on CPU (minutes, not seconds)")
    ui.line("XLB + JAX (L1.5)", ui.status(env["xlb"], "YES", "NO"), "",
            "the reference backend" if env["xlb"] else
            "not required: cfd.xlb_cold ships the same D3Q19 on CuPy/NumPy")
    ui.line("cfd.ibrans (L2a)", ui.status(env["ibrans"], "YES", "NO"), "",
            "native reactive IB-RANS" if env["ibrans"] else
            "L2a will degrade down the ladder")

    ui.section("Fidelity ladder")
    #   Each rung is asked ITS OWN question — «would `--fidelity X` really
    #   run X on this machine?» — instead of being compared against the
    #   rung this invocation happens to have asked for. Reading the ladder
    #   off a single probe would report every rung above the default as
    #   degrading even when the dependencies are all installed.
    import copy as _copy
    for name in FIDELITY_BY_NAME:
        probe = _copy.copy(args)
        probe.fidelity = name
        e = probe_environment(probe)
        eff_name = Fidelity(e["fidelity_effective"]).name
        ui.line(name, ui.status(eff_name == name, "reachable",
                                f"degrades to {eff_name}"), "",
                FIDELITY_WHAT[name])
    req = Fidelity(env["fidelity_requested"]).name
    eff = Fidelity(env["fidelity_effective"]).name
    if req != eff:
        print()
        print("  " + ui.warn(f"as invoked ({'--fidelity ' + req}) this "
                             f"machine would really run {eff} — "
                             f"{env['fidelity_reason']}"))
    ui.line("search fidelity", "= --fidelity", "",
            "every expensive evaluation (LHS + acquisition + verification) "
            "runs at that rung; `--search-fidelity L0` decouples them")
    ui.line("inner NSGA-II", "surrogate always", "",
            "~1e4 points per generation: no rung above L0 survives that")

    ui.section("Data tables, with their MARK")
    ds = _dataset_status()
    n_prov = 0
    for rel, status in ds:
        m = STATUS_MARK.get(status, "?")
        n_prov += m == "?"
        print(f"  {ui.mark(m)} {rel:<44} {ui.dim(status)}")
    print()
    print("  " + ui.warn(f"{n_prov} of {len(ds)} tables are NOT verified: "
                         "anchoring a conclusion on one of them is "
                         "FORBIDDEN — the reason of each is in its own "
                         "`@status_reason` header"))

    prov = pc.provisional_report()
    if prov:
        ui.section("Provisional coefficients touched by THIS import")
        for it in prov[:12]:
            ui.line(str(it.get("key")), "", "", str(it.get("message"))[:70])
        print()
        print("  " + ui.c("the list grows as the physics is exercised: run "
                          "a design and it will be longer.", "dim"))

    ui.section("Validation campaign")
    try:
        cases_mod = _load_cases()
        n_cases = len(cases_mod.CASES)
        n_an = sum(1 for c in cases_mod.CASES if "spec" in c)
        ui.line("published cases", f"{n_cases}", "",
                f"{n_an} carry a theta and can be run with `--analyze <id>`")
        ui.line("levels", ", ".join(sorted({str(c.get("level"))
                                            for c in cases_mod.CASES})), "",
                "A bands · B architecture · C air-split anchor · D thermal "
                "extreme")
    except Exception as exc:
        print("  " + ui.warn(f"validation/cases.py unreadable: "
                             f"{type(exc).__name__}"))
    ui.line("runner", "`python validation/validate.py`", "",
            "`--strict` fails on any regression; known gaps are declared in "
            "docs/VALIDATION.md")
    ui.line("regression anchors",
            ui.status(os.path.exists(ANCHOR_PATH), "frozen", "absent"), "",
            "validation/anchors.json — every run compares REF_CB10 against "
            "them")
    print()
    print("  " + ui.warn("a case marked `known_gap` is a DECLARED DEFECT "
                         "with a named mechanism, not an unknown: read its "
                         "note in docs/VALIDATION.md"))

    ui.section("What is NOT covered")
    for k, v in (
        ("L2b (OpenFOAM)",
         "out of scope in v1 (contract §12): it is accepted on the command "
         "line and degrades to L2a with a tagged `source`"),
        ("combustor map",
         "there is no off-design map in v1 (WAVE4 §4). `--sweep phi_pz` and "
         "`--sweep load` are the frozen-geometry substitute"),
        ("KJ66 performance",
         "the package holds NO measurement of thrust, EGT, consumption or "
         "cold dp for the anchor machine: level B can falsify architecture "
         "and class bands, not performance"),
        ("dp/p as an output",
         "in L0 dp/p is an INPUT (a hard cycle constraint), so the +-20 % "
         "criterion is not evaluable without L1.5/L2a")):
        ui.line(k, "", "", v, width_=22)
    print()
    return 0


# ==========================================================================
# 15. Verification of the winner at high fidelity (WAVE4 §1.7)
# ==========================================================================
def warn_if_gate_never_passed(result: dict, rounds: int,
                              n_history_before: int = 0) -> None:
    """The surrogate quality gate (R^2 eta >= 0.90, R^2 dp/p >= 0.85, 2sigma
    coverage in [0.80, 1]) decides whether a round spends its budget on
    NSGA-II or on extra space filling. `history` ONLY gets an entry when the
    gate passes: if it did not grow in this run, not a single round ever
    optimised and the result is an expensive LHS. That has to be said, not
    hidden (WAVE4 §7). `n_history_before` discounts what a resumed
    checkpoint already carried."""
    if len(result.get("history") or []) > n_history_before:
        return
    print()
    print("  " + ui.degraded(
        "surrogate quality gate",
        f"it was not cleared in any of the {rounds} rounds: the budget went "
        "on space filling and NSGA-II never ran"))
    print("  " + ui.c("the incumbent is the best SAMPLED point, not an "
                      "optimum of the front. WAVE4 §7 says enrich "
                      "physics_features before lowering the threshold.",
                      "dim"))


def _announce_search_cost(spec, fid_search, n_expensive: int, env) -> None:
    """Says what the search rung is, and what it will COST, before paying.

    `--fidelity` governs every expensive evaluation, so the difference
    between L0 and L1 is not a label: it is the difference between a run
    that ends while you read the panel and one that ends after lunch. The
    projection is not a guess — ONE point is timed here, at that exact
    rung, and multiplied by the budget the engineer actually asked for.
    Quoting a remembered "~2 s/point" would be a number about some other
    machine.
    """
    name = Fidelity(fid_search).name
    if int(fid_search) <= int(Fidelity.L0):
        print("  " + ui.c(f"the loop explores in {name} (~1.3 ms/point); "
                          "the inner NSGA-II generations always run over "
                          "the surrogate, never over the physics", "dim"))
        return

    print("  " + ui.warn(
        f"--fidelity {name} runs the {name} physics on EVERY EXPENSIVE "
        f"EVALUATION — the initial LHS and each acquisition batch — and the "
        f"surrogate is trained on those records. The inner NSGA-II "
        f"generations stay on the surrogate: they are ~1e4 points per "
        f"generation and no rung above L0 survives that."))
    t0 = time.time()
    try:
        pc.evaluate(nominal_theta(spec), fidelity=fid_search,
                    use_cache=False)
        dt = time.time() - t0
    except Exception as exc:
        print("  " + ui.warn(f"the cost probe failed ({type(exc).__name__}): "
                             "the run continues, but with no projection of "
                             "how long it will take"))
        return
    total = dt * n_expensive
    ui.line("measured cost", f"{dt:.2f} s/point at {name}", "",
            "timed here, on this machine, at this operating point")
    ui.line("budget", f"{n_expensive} expensive evaluations", "",
            f"projected {total / 60.0:.1f} min "
            f"({total / 3600.0:.1f} h) of physics alone")
    if total > 1800.0:
        print("  " + ui.warn(
            "that is over half an hour of physics. `--search-fidelity L0` "
            "explores cheaply and still verifies the winner at "
            f"{Fidelity(env['fidelity_requested']).name}; --quick cuts the "
            "budget to 100 + 2x8."))
    #   The cache is the reason a resumed or repeated run is not charged
    #   twice, and it only exists at >= L1. Saying so here is cheaper than
    #   having the engineer discover it by waiting.
    ui.line("cache", str(pc.CACHE_PATH or "-"), "",
            "every >= L1 record is persisted: a resumed run does not pay "
            "for a theta twice")


def _report_search_fidelity(recs: list, fid_search) -> None:
    """How many expensive evaluations REALLY reached the requested rung.

    A rung that silently fell back on 40 % of the population is not the
    rung the panel claims to have searched at, and the surrogate learned
    from a mixture. The `source` tag carries an arrow whenever a record
    dropped down the ladder (contract §3), so the count is read off the
    records themselves rather than assumed.
    """
    if int(fid_search) <= int(Fidelity.L0) or not recs:
        return
    n = len(recs)
    fell = [r for r in recs if "→" in str(r.get("source", ""))]
    name = Fidelity(fid_search).name
    if not fell:
        print("  " + ui.ok(f"{name} reached on {n}/{n} expensive "
                           f"evaluations"))
        return
    why = {}
    for r in fell:
        tag = str(r.get("source", ""))
        why[tag] = why.get(tag, 0) + 1
    print("  " + ui.degraded(
        f"search fidelity {name}",
        f"{len(fell)} of {n} expensive evaluations fell back down the "
        f"ladder; the surrogate learned from a MIXTURE of rungs"))
    for tag, k in sorted(why.items(), key=lambda z: -z[1])[:4]:
        ui.line(f"  {tag}", f"{k} point(s)", "", "", width_=34)


def verify_winner(spec, result, env, logger) -> None:
    """L0 explores, L1+ verifies: re-evaluates the incumbent at the
    requested fidelity and leaves the (L0, L1) pair in `result['l1_pairs']`
    for section 11 of the report. Never raises."""
    fid = env["fidelity_requested"]
    if int(fid) <= int(Fidelity.L0):
        return
    theta = np.asarray(result["theta"], dtype=float)
    rec0 = dict(result["record"])
    print()
    print("  " + ui.c(f"verifying the incumbent at "
                      f"{Fidelity(fid).name}...", "dim"))
    t0 = time.time()
    try:
        rec1 = pc.evaluate(theta, fidelity=fid)
    except Exception as exc:                       # the core must not raise
        print("  " + ui.degraded(f"{Fidelity(fid).name} verification",
                                 f"{type(exc).__name__}: {exc}"))
        return
    result["record"] = rec1
    result["l1_pairs"] = [dict(label="Incumbent", L0=rec0, L1=rec1)]
    print("  " + ui.ok(f"{Fidelity(fid).name} in {time.time() - t0:.1f} s · "
                       f"source='{rec1.get('source')}'"))


# ==========================================================================
# 16. main
# ==========================================================================
def main(argv=None) -> int:
    # Windows consoles and pipes report cp1252 and blow up on eta/subscripts.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    ui.init(use_color=False if args.no_color else None)
    ui.reset_degradations()

    if args.analyze:
        bad = reject_synthesis_flags(raw_argv)
        if bad:
            print("  " + ui.err(
                "--analyze takes a PUBLISHED case from validation/cases.py "
                "and synthesises nothing, so it cannot be combined with "
                + ", ".join(bad) + "."))
            print("  " + ui.c("A parameteriser is not an analyser. Use "
                              "--p3-override / --t3-override / "
                              "--mdot-override / --t4-override to move the "
                              "operating point off the published one.",
                              "dim"))
            return 2

    # `--info` is a help mode: it must not need an outdir, a spec or a
    # single physical evaluation to answer.
    if args.info:
        return run_info(args)

    if not _asked_for_a_job(raw_argv):
        args.interactive = True

    print()
    print(ui.banner())
    print()

    env = probe_environment(args)

    if args.analyze:
        #   The catalogue is a help mode: it evaluates nothing, so it does
        #   not owe the reader an environment probe either.
        if str(args.analyze).strip().lower() not in ("list", "?", "help"):
            announce_environment(env, args)
        rc = run_analyze(args, env)
        return _strict_verdict(args, rc)

    # ---- modes over an existing checkpoint (no optimisation) ------------
    if args.list_pareto or args.pareto_pick is not None:
        rc = run_pareto_mode(args, env)
        return _strict_verdict(args, rc)

    if args.interactive:
        args = run_wizard(args)
        env = probe_environment(args)       # the wizard may have moved it

    # The mode is decided with the flags AS THEY ARRIVED: filling the
    # default budgets before looking at them would make every run look like
    # an optimisation (and the ~1 s nominal path would not exist).
    optimize = wants_optimization(args)
    budget_default = BUDGET_QUICK if args.quick else BUDGET_FULL
    args.n_init = _or(args.n_init, budget_default["n_init"])
    args.rounds = _or(args.rounds, budget_default["rounds"])
    args.batch_size = _or(args.batch_size, budget_default["batch_size"])

    os.makedirs(args.outdir, exist_ok=True)
    announce_environment(env, args)

    logger = StyledLog(quiet=args.quiet)

    # ---- [1/4] specification --------------------------------------------
    print(ui.rule("[1/4] Specification and operating point", "brand"))
    try:
        spec, integration = build_spec(
            args, log=(lambda m: print("  " + ui.c(str(m).strip(), "dim"))))
    except (ValueError, OSError, FileNotFoundError) as exc:
        print("  " + ui.err(str(exc)))
        return 2
    warn_fixed_out_of_bounds(spec)

    # The persistent cache ONLY receives fidelities >= L1 (contract §2).
    set_cache_path(os.environ.get("PHYCB_CACHE")
                   or os.path.join(args.outdir, "phys_cache.jsonl"))

    if args.hifi_pairs:
        try:
            with open(args.hifi_pairs, encoding="utf-8") as fh:
                pairs = json.load(fh)
            for pair in pairs:
                register_hifi_pair(np.array(pair["theta"], dtype=float),
                                   {k: v for k, v in pair.items()
                                    if k != "theta"})
            print("  " + ui.ok(f"calibration loaded: {args.hifi_pairs} "
                               f"({len(pairs)} pairs) — affine per-output "
                               "correction (identity with <2 pairs)"))
        except Exception as exc:
            print("  " + ui.degraded("--hifi-pairs",
                                     f"ignored: {type(exc).__name__}: {exc}"))

    check_anchors(args.freeze_anchors, quiet=args.quiet)

    budget = (f"{args.n_init} init + {args.rounds}x{args.batch_size} "
              "acquisition" + ("  (quick)" if args.quick else "")) \
        if optimize else "1 physical evaluation (nominal rules, WAVE1 §3)"
    panel_rows = [
        ("P3", ui.c(f"{spec.P3_Pa:,.0f} Pa", "val", bold=True)),
        ("T3", ui.c(f"{spec.T3_K:.1f} K", "val", bold=True)),
        ("Air mass flow", f"{spec.mdot_kgs:.4g} kg/s"),
        ("T4 (turbine)", f"{spec.T4_K:.1f} K"),
        ("Fuel", ui.c(spec.fuel, "brand")),
        ("Max dp/p", f"{spec.dp_max:.3f}   (HARD cycle constraint)"),
        ("Max PF", f"{spec.PF_max:.2f}"),
        ("Architecture", spec.arch or ui.c("free (design variable)",
                                           "accent")),
        ("Liner material", f"{spec.material}  ·  casing "
                           f"{spec.material_casing}"),
        ("DfAM", "5 LPBF constraints active" if spec.dfam
                 else ui.c("DISABLED (--no-dfam)", "warn")),
        ("Budget", ui.c(budget, "accent")),
        ("Fidelity", ui.c(Fidelity(env["fidelity_requested"]).name,
                          "brand")
                     + ui.c("   (verification)", "dim")),
        ("Search fidelity", ui.c(Fidelity(env["fidelity_search"]).name,
                                 "brand")
                            + ui.c("   (every expensive evaluation)",
                                   "dim")),
    ]
    if spec.envelope_D_mm or spec.envelope_L_mm:
        panel_rows.append(("Max envelope",
                           f"D {spec.envelope_D_mm or '—'} x "
                           f"{spec.envelope_L_mm or '—'} mm"))
    if spec.fixed_vars:
        panel_rows.append(("Pinned variables", ui.c(
            "  ".join(f"{k}={v:g}" for k, v in spec.fixed_vars.items()),
            "accent")))
    if integration:
        src = str(integration.get("path") or "")
        if len(src) > 34:                     # the panel does not clip
            src = "…" + src[-33:]
        panel_rows.append(("Spec read from",
                           ui.c(f"{integration.get('kind')} · {src}",
                                "accent")))
    if args.seed is not None:
        panel_rows.append(("Seed", f"{args.seed}"))
    print(ui.panel("Specification", panel_rows))

    # ---- [2/4] physics --------------------------------------------------
    print()
    ckpt = os.path.join(args.outdir, "phycb_run.json")
    t_start = time.time()
    population: list = []

    # (a) direct evaluation of a given theta
    if args.eval_theta:
        print(ui.rule("[2/4] Direct evaluation (no optimisation)", "brand"))
        try:
            vals = [float(x) for x in re.split(r"[,\s]+",
                                               args.eval_theta.strip()) if x]
            # FULL theta => the theta rules: the operating point travels in
            # positions 16-19 and the fuel in gate 15 (exact Phy-AC
            # pattern). Otherwise `fix_operating_point` would overwrite them
            # with the flags and the engineer would evaluate another design
            # without noticing.
            if len(vals) == NDIM:
                fuel_th = pc.decode_gates(np.array(vals))["fuel"]
                if fuel_th != spec.fuel:
                    print("  " + ui.warn(f"the theta carries fuel={fuel_th} "
                                         f"and it overrides --fuel "
                                         f"{spec.fuel}"))
                spec.fuel = fuel_th
                (spec.P3_Pa, spec.T3_K, spec.mdot_kgs, spec.T4_K) = (
                    vals[16], vals[17], vals[18], vals[19])
                print("  " + ui.c(f"operating point TAKEN FROM THE THETA: "
                                  f"P3={spec.P3_Pa:,.0f} Pa  "
                                  f"T3={spec.T3_K:.1f} K  "
                                  f"mdot={spec.mdot_kgs:g} kg/s  "
                                  f"T4={spec.T4_K:.1f} K", "accent"))
            result = evaluate_design(spec, np.array(vals),
                                     fidelity=env["fidelity_requested"])
        except (ValueError, OSError) as exc:
            print("  " + ui.err(str(exc)))
            return 2
        print("  " + ui.ok(f"theta of {len(vals)} values evaluated in "
                           f"{1000 * (time.time() - t_start):.0f} ms"))
        print(ui.panel("Evaluated theta", [
            (name, f"{val:g}")
            for name, val in zip(VAR_NAMES, result["theta"])]))
        out = emit_deliverables(args, spec, result, args.outdir, env,
                                ckpt=None, logger=logger,
                                title="Evaluated design",
                                integration=integration)
        return _strict_verdict(args, 0 if out["feasible"] else 3)

    # (b) autonomous optimisation (new or resumed)
    if optimize:
        print(ui.rule("[2/4] Autonomous optimisation (layers 2-4)", "brand"))
        fid_search = env["fidelity_search"]
        n_expensive = int(args.n_init) + int(args.rounds) * int(
            args.batch_size)
        _announce_search_cost(spec, fid_search, n_expensive, env)
        designer, n_hist_before = None, 0
        if args.resume is not None:
            resume_path = (ckpt if args.resume == "__auto__"
                           else args.resume)
            if not os.path.exists(resume_path):
                print("  " + ui.err(f"checkpoint not found: {resume_path}"))
                return 2
            try:
                designer = AutonomousCombustorDesigner.load(
                    resume_path, fidelity=fid_search,
                    n_workers=args.workers, log=logger, seed=args.seed)
            except Exception as exc:
                print("  " + ui.err(f"could not resume: "
                                    f"{type(exc).__name__}: {exc}"))
                return 2
            spec = designer.spec        # the dataset was evaluated under it
            n_hist_before = len(designer.history)
            print("  " + ui.ok(f"resumed: {resume_path} · "
                               f"{len(designer.recs)} previous evaluations · "
                               f"seed {designer.seed}"))
        else:
            designer = AutonomousCombustorDesigner(
                spec, fidelity=fid_search, n_workers=args.workers,
                log=logger, seed=SEED if args.seed is None else args.seed)
        result = designer.run(n_init=args.n_init, rounds=args.rounds,
                              batch_size=args.batch_size,
                              checkpoint_path=ckpt)
        #  The population is kept so that an infeasible run can be
        #  diagnosed over ALL evaluated designs, not over the one returned.
        population = [(r, np.array(r["_theta"], dtype=float))
                      for r in designer.recs if "_theta" in r]
        _report_search_fidelity(designer.recs, fid_search)
        warn_if_gate_never_passed(result, args.rounds, n_hist_before)
        verify_winner(spec, result, env, logger)
        print("  " + ui.ok(f"optimisation completed in "
                           f"{time.time() - t_start:.1f} s"))
        out = emit_deliverables(args, spec, result, args.outdir, env,
                                ckpt=ckpt, logger=logger,
                                integration=integration)
        if not out["feasible"]:
            infeasibility_report(spec, population)
            print()
            print("  " + ui.err("NO feasible design was found for this "
                                "spec."))
            print("  " + ui.c("read the WALLS above before touching the "
                              "budget: that is what no design in the box "
                              "clears.", "dim"))
        return _strict_verdict(args, 0 if out["feasible"] else 3)

    # (c) nominal design (WAVE1 §3) — the ~1 s path
    print(ui.rule("[2/4] Nominal sizing (layer 1, WAVE1 §3)", "brand"))
    theta = nominal_theta(spec)
    result = evaluate_design(spec, theta,
                             fidelity=env["fidelity_requested"])
    rec = result["record"]
    print("  " + ui.ok(f"record `{rec.get('schema')}` in "
                       f"{1000 * (time.time() - t_start):.0f} ms · "
                       f"source='{rec.get('source')}'"))
    print("  " + ui.c("nominal air split: "
                      + "  ".join(f"{k}={v:.3f}" for k, v in
                                  (rec.get("splits") or {}).items()), "dim"))
    print("  " + ui.c(f"sizing driver: "
                      f"{rec.get('sizing_driver')} · A_ref="
                      f"{rec.get('A_ref_m2', float('nan')) * 1e4:.2f} cm2 · "
                      f"D_ref={rec.get('D_ref_mm', float('nan')):.1f} mm",
                      "dim"))
    out = emit_deliverables(args, spec, result, args.outdir, env, ckpt=None,
                            logger=logger, title="Nominal design",
                            integration=integration)
    if not out["feasible"]:
        print("  " + ui.c("the nominal path evaluates ONE design, so there "
                          "is no population to diagnose: run --optimize to "
                          "learn whether the spec has a WALL or is merely "
                          "TIGHT.", "dim"))
    return _strict_verdict(args, 0 if out["feasible"] else 3)


def _strict_verdict(args, rc: int) -> int:
    """`--strict`: what degraded stops being a warning and becomes a
    failure. For a human, announcing each degradation is enough; a CI needs
    ONE question at the end — "did this run in full?" — and this is it. It
    does not change what the program does: it changes what the program
    CONCEDES."""
    if not getattr(args, "strict", False):
        return rc
    deg = ui.strict_verdict()
    if deg:
        print()
        print("  " + ui.err(f"--strict: {len(deg)} degradation(s); in "
                            "strict mode that is a failure"))
        for what, why in deg:
            print("    " + ui.c(f"{what}: {why[:160]}", "err"))
        return 4
    print("  " + ui.ok("--strict: nothing degraded"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
