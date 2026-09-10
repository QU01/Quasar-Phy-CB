//
// DfamVoxelCheck.cs - DfAM RE-VERIFICATION ON THE REAL VOXELS.
//
// dfam_checker.py already screens the contract analytically inside the
// optimizer loop (5 constraints, <10 ms, no voxels). This class re-checks on
// the field that is actually going to be printed the two rules only the
// built geometry can settle:
//
//   1. OVERHANG, PER PRINTED PART, by comparing consecutive Z slices. A
//      solid cell is supported when at least one of the 3x3 cells directly
//      beneath it (in BUILD order) is solid. On a cubic grid that
//      neighbourhood is exactly the 45-degree self-supporting cone, so
//      "unsupported cell" means "down-skin steeper than 45 degrees from the
//      plate" - EstadoDelArte 4.3. The first layer rests on the build plate
//      and is supported by definition. This sees what the analytic profile
//      check cannot: hole roofs, boss under-skins, flange overhangs, the
//      vaporizer elbow. It runs PER PART because each part is printed on its
//      own - checking the assembly would blame the liner for the casing.
//
//   2. POWDER DRAINABILITY, on the assembly, by FLOOD FILLING the void. This
//      is the differentiator (EstadoDelArte 4.4: trapped powder sinters
//      during HIP and blocks the passage permanently). Two verdicts:
//        (a) ENCLOSED VOIDS - void unreachable from outside the machine at
//            all. Zero exits, unconditionally fatal. Purely topological: no
//            analytic cavity graph can find these.
//        (b) EXITS PER CAVITY - the void inside the casing envelope is split
//            into connected cavities, and for each one we count the DISTINCT
//            PATCHES through which it breaks out of the envelope. Two exits
//            minimum. Counting break-out patches (rather than special-casing
//            ports vs open ends) treats every opening the same way and needs
//            no arbitrary classification: a drain port, the diffuser inlet
//            and the turbine exit are all just places where powder gets out.
//
// WHY THE GRID IS COARSE (1 mm by default, WAVE3 section 6): a flood fill at
// the build voxel is needlessly expensive and the TOPOLOGY does not change.
// It is also physically right: a channel narrower than the cell does not
// drain powder, so the sub-millimetre effusion holes are automatically NOT
// counted as drainage - exactly the rule EstadoDelArte 4.4 states, obtained
// for free instead of by a special case.
//
// The verdict goes to PicoGK.log AND to <outDir>/dfam_report.json.
//

using System;
using System.Collections.Generic;
using System.Numerics;
using PicoGK;

namespace CombustorDesigner
{
    /// <summary>Overhang statistics of ONE printed part.</summary>
    public sealed class PartOverhang
    {
        public string strName;
        public long nSolid;
        public long nUnsupported;
        public float fFrac;
        public float fWorstZMm;
        public float fWorstFrac;
        public float fCellMm;

        public Dictionary<string, object> oAsDict()
            => new Dictionary<string, object>
            {
                { "part", strName },
                { "n_solid_cells", nSolid },
                { "n_unsupported_cells", nUnsupported },
                { "frac_unsupported", fFrac },
                { "worst_slice_z_mm", fWorstZMm },
                { "worst_slice_frac", fWorstFrac },
                { "coarse_voxel_mm", fCellMm },
            };
    }

    /// <summary>Result of the voxel-level DfAM re-verification.</summary>
    public sealed class DfamVoxelReport
    {
        /// <summary>Frozen order of the voxel-level constraint vector. It is
        /// COMPLEMENTARY to dfam_checker.g_dfam (5 analytic entries), never a
        /// replacement: this one is measured on the built field.</summary>
        public static readonly string[] astrGNames =
            { "overhang_voxel", "trapped_powder", "drain_exits" };

        public float[] afG = { -1f, -1f, -1f };
        public bool bFeasible = true;
        public List<string> aViolations = new List<string>();
        public List<string> aNotes = new List<string>();
        public Dictionary<string, object> oDetail =
            new Dictionary<string, object>();
    }

    public static class DfamVoxelCheck
    {
        /// <summary>Upper bound on the number of coarse cells. The cell size
        /// is grown until the grid fits, and the growth is logged.</summary>
        const long lMaxCells = 20_000_000L;

        /// <summary>Minimum number of exits per cavity (EstadoDelArte 4.4:
        /// ">= 2 exits with line of sight per cavity").</summary>
        const int nDrainExitsMin = 2;

        /// <summary>Voids below this volume are staircase noise, not a
        /// powder pocket.</summary>
        const float fVoidNoiseMm3 = 4f;

        // ==================================================================
        // 1. Overhang, one part at a time
        // ==================================================================

        /// <summary>
        /// Overhang statistics of one printed part. Called during the
        /// streaming build loop, right after the part is voxelized and
        /// before it is released, so peak memory stays at one part.
        /// </summary>
        public static PartOverhang CheckPart(Voxels vox,
                                             CombustorParameters p,
                                             string strName)
        {
            BBox3 oBB = vox.oCalculateBoundingBox();
            float fCell = MathF.Max(p.Dfam.CoarseVoxelMm, p.VoxelSizeMm);
            Vector3 vecMin = oBB.vecMin - new Vector3(2f * fCell);
            Vector3 vecMax = oBB.vecMax + new Vector3(2f * fCell);
            GridSize(vecMax - vecMin, ref fCell, out int nx, out int ny,
                     out int nz, strName);

            var abSolid = new bool[nx * ny * nz];
            for (int k = 0; k < nz; k++)
            {
                float fZ = vecMin.Z + (k + 0.5f) * fCell;
                for (int j = 0; j < ny; j++)
                {
                    float fY = vecMin.Y + (j + 0.5f) * fCell;
                    for (int i = 0; i < nx; i++)
                        abSolid[nIdx(i, j, k, nx, ny)] = vox.bIsInside(
                            new Vector3(vecMin.X + (i + 0.5f) * fCell,
                                        fY, fZ));
                }
            }

            var oRes = new PartOverhang
            {
                strName = strName,
                fCellMm = fCell,
                fWorstZMm = float.NaN,
            };
            float fBuildSign = p.fBuildSign();

            // Index (in BUILD order) of the first slice that actually holds
            // solid. This is NOT s == 0: the sampling grid starts two cells
            // BELOW the bounding box (see vecMin above), so slices 0 and 1
            // are pure padding. Exempting s == 0 therefore exempted empty
            // air and left the real first layer to be tested against an
            // empty slice underneath -- reporting 100 % unsupported at the
            // build plate for every part. Found by the Phy-CB geometry
            // cross-check, 2026-07-31.
            int sFirstSolid = -1;
            for (int s = 0; s < nz && sFirstSolid < 0; s++)
            {
                int k = fBuildSign >= 0f ? s : nz - 1 - s;
                for (int j = 0; j < ny && sFirstSolid < 0; j++)
                    for (int i = 0; i < nx; i++)
                        if (abSolid[nIdx(i, j, k, nx, ny)])
                        {
                            sFirstSolid = s;
                            break;
                        }
            }

            for (int s = 0; s < nz; s++)
            {
                // s = index in BUILD order; the first SOLID one rests on
                // the plate and is self-supported by definition.
                int k = fBuildSign >= 0f ? s : nz - 1 - s;
                int kBelow = fBuildSign >= 0f ? k - 1 : k + 1;
                long nLayer = 0, nLayerUnsup = 0;
                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++)
                    {
                        if (!abSolid[nIdx(i, j, k, nx, ny)])
                            continue;
                        nLayer++;
                        if (s <= sFirstSolid)
                            continue;               // on the build plate
                        bool bSup = false;
                        for (int dj = -1; dj <= 1 && !bSup; dj++)
                            for (int di = -1; di <= 1; di++)
                            {
                                int ii = i + di, jj = j + dj;
                                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny)
                                    continue;
                                if (abSolid[nIdx(ii, jj, kBelow, nx, ny)])
                                {
                                    bSup = true;
                                    break;
                                }
                            }
                        if (!bSup)
                            nLayerUnsup++;
                    }
                oRes.nSolid += nLayer;
                oRes.nUnsupported += nLayerUnsup;
                if (nLayer > 0)
                {
                    float fLayerFrac = (float)(nLayerUnsup / (double)nLayer);
                    if (float.IsNaN(oRes.fWorstZMm)
                        || fLayerFrac > oRes.fWorstFrac)
                    {
                        oRes.fWorstFrac = fLayerFrac;
                        oRes.fWorstZMm = vecMin.Z + (k + 0.5f) * fCell;
                    }
                }
            }
            oRes.fFrac = oRes.nSolid > 0
                ? (float)(oRes.nUnsupported / (double)oRes.nSolid) : 0f;
            Library.Log($"DfAM overhang '{strName}' (build "
                      + $"{p.Dfam.BuildOrientation}, cell {fCell:0.##} mm): "
                      + $"{100f * oRes.fFrac:0.##} % of the solid cells have "
                      + $"no 45-deg support beneath; worst slice at z = "
                      + $"{oRes.fWorstZMm:0.#} mm "
                      + $"({100f * oRes.fWorstFrac:0.#} %)");
            return oRes;
        }

        // ==================================================================
        // 2. Powder drainability on the assembly + the overhang verdict
        // ==================================================================

        public static DfamVoxelReport Run(Voxels voxAssembly,
                                          CombustorParameters p,
                                          List<PartOverhang> aOverhang)
        {
            var oRep = new DfamVoxelReport();
            var oWatch = System.Diagnostics.Stopwatch.StartNew();

            // -- overhang verdict: the WORST printed part drives g --------
            float fFracMax = MathF.Max(p.Dfam.OverhangFracMax, 1e-6f);
            float fWorst = 0f;
            string strWorst = "(none)";
            var aOverDicts = new List<Dictionary<string, object>>();
            foreach (PartOverhang o in aOverhang)
            {
                aOverDicts.Add(o.oAsDict());
                if (o.fFrac > fWorst) { fWorst = o.fFrac; strWorst = o.strName; }
            }
            oRep.afG[0] = fWorst / fFracMax - 1f;
            if (oRep.afG[0] > 0f)
                oRep.aViolations.Add(
                    $"overhang: part '{strWorst}' has {100f * fWorst:0.###} % "
                    + "of its solid cells with no support within 45 deg "
                    + $"beneath them in build '{p.Dfam.BuildOrientation}' "
                    + $"(tolerance {100f * fFracMax:0.###} %). Either "
                    + "re-orient the build, chamfer the down-skins, or raise "
                    + "dfam.overhang_frac_max if supports are acceptable on "
                    + "that surface.");
            oRep.oDetail["overhang"] = new Dictionary<string, object>
            {
                { "method", "per printed part, consecutive Z-slice "
                          + "comparison; a solid cell is supported when any "
                          + "of the 3x3 cells beneath it (build order) is "
                          + "solid = the 45 deg self-supporting cone" },
                { "build_orientation", p.Dfam.BuildOrientation },
                { "frac_unsupported_max", fFracMax },
                { "worst_part", strWorst },
                { "worst_frac", fWorst },
                { "parts", aOverDicts },
            };

            // -- coarse grid over the assembly ----------------------------
            BBox3 oBB = voxAssembly.oCalculateBoundingBox();
            float fCell = MathF.Max(p.Dfam.CoarseVoxelMm, p.VoxelSizeMm);
            Vector3 vecMin = oBB.vecMin - new Vector3(3f * fCell);
            Vector3 vecMax = oBB.vecMax + new Vector3(3f * fCell);
            GridSize(vecMax - vecMin, ref fCell, out int nx, out int ny,
                     out int nz, "assembly");
            int nCells = nx * ny * nz;
            Library.Log($"DfAM drainability: sampling a {nx} x {ny} x {nz} "
                      + $"grid at {fCell:0.###} mm ({nCells / 1000} k cells)");

            // label: -1 solid, 0 unvisited void, >0 labelled
            var anLabel = new int[nCells];
            var abEnvelope = new bool[nCells];
            long lSolid = 0;
            for (int k = 0; k < nz; k++)
            {
                float fZ = vecMin.Z + (k + 0.5f) * fCell;
                bool bZIn = bZInsideEnvelope(p, fZ);
                float fREnv = bZIn ? fEnvelopeRAt(p, fZ) : -1f;
                float fREnv2 = fREnv * fREnv;
                for (int j = 0; j < ny; j++)
                {
                    float fY = vecMin.Y + (j + 0.5f) * fCell;
                    for (int i = 0; i < nx; i++)
                    {
                        float fX = vecMin.X + (i + 0.5f) * fCell;
                        int n = nIdx(i, j, k, nx, ny);
                        if (voxAssembly.bIsInside(new Vector3(fX, fY, fZ)))
                        {
                            anLabel[n] = -1;
                            lSolid++;
                        }
                        else if (bZIn && fX * fX + fY * fY <= fREnv2)
                            abEnvelope[n] = true;
                    }
                }
            }
            Library.Log($"DfAM: {lSolid / 1000} k solid cells sampled in "
                      + $"{oWatch.ElapsedMilliseconds / 1000f:0.0} s");

            // -- 2a. enclosed voids (zero exits) --------------------------
            FloodFromGridBoundary(anLabel, nx, ny, nz);   // label 1 = outside
            float fCellVol = fCell * fCell * fCell;
            var aEnclosed = new List<Dictionary<string, object>>();
            float fTrapped = 0f;
            int nNextLabel = 2;
            for (int n = 0; n < nCells; n++)
            {
                if (anLabel[n] != 0)
                    continue;
                CompResult oComp = LabelComponent(anLabel, n, nNextLabel++,
                                                  nx, ny, nz);
                float fVol = oComp.nCells * fCellVol;
                if (fVol < fVoidNoiseMm3)
                    continue;
                fTrapped += fVol;
                aEnclosed.Add(new Dictionary<string, object>
                {
                    { "volume_mm3", fVol },
                    { "n_cells", oComp.nCells },
                    // fCx/fCy/fCz are mean CELL INDICES
                    { "centroid_mm", new float[]
                        {
                            vecMin.X + (oComp.fCx + 0.5f) * fCell,
                            vecMin.Y + (oComp.fCy + 0.5f) * fCell,
                            vecMin.Z + (oComp.fCz + 0.5f) * fCell,
                        } },
                    { "z_range_mm", new float[]
                        {
                            vecMin.Z + (oComp.nK0 + 0.5f) * fCell,
                            vecMin.Z + (oComp.nK1 + 0.5f) * fCell,
                        } },
                });
            }
            oRep.afG[1] = fTrapped / MathF.Max(fVoidNoiseMm3, 1e-6f) - 1f;
            if (aEnclosed.Count > 0)
                oRep.aViolations.Add(
                    $"trapped powder: {aEnclosed.Count} fully ENCLOSED "
                    + $"void(s) totalling {fTrapped:0.#} mm3 - no exit at "
                    + "all. Powder left there sinters during HIP and blocks "
                    + "the cavity permanently (EstadoDelArte 4.4).");
            oRep.oDetail["enclosed_voids"] = new Dictionary<string, object>
            {
                { "method", "flood fill of the void from the grid boundary; "
                          + "whatever is not reached is sealed inside the "
                          + "part" },
                { "count", aEnclosed.Count },
                { "total_volume_mm3", fTrapped },
                { "noise_floor_mm3", fVoidNoiseMm3 },
                { "cavities", aEnclosed },
            };

            // -- 2b. exits per cavity -------------------------------------
            oRep.oDetail["drainability"] = CheckDrainExits(
                anLabel, abEnvelope, nx, ny, nz, vecMin, fCell, p, oRep);

            oRep.bFeasible = true;
            foreach (float f in oRep.afG)
                if (f > 0f) oRep.bFeasible = false;

            oRep.oDetail["params"] = new Dictionary<string, object>
            {
                { "coarse_voxel_mm", fCell },
                { "build_voxel_mm", p.VoxelSizeMm },
                { "grid", new int[] { nx, ny, nz } },
                { "origin_mm", new float[]
                    { vecMin.X, vecMin.Y, vecMin.Z } },
                { "drain_exits_min", nDrainExitsMin },
                { "note", "the coarse cell is also the physical statement "
                        + "that a channel finer than the cell does not drain "
                        + "powder - this is why the sub-millimetre effusion "
                        + "holes never count as a drain (EstadoDelArte 4.4)" },
            };
            oRep.aNotes.Add("the drainability verdict is taken on the "
                          + "ASSEMBLY (the machine as designed); the overhang "
                          + "verdict is taken per PRINTED PART, because each "
                          + "part goes on the plate on its own");
            oRep.aNotes.Add("voxel re-verification took "
                          + $"{oWatch.ElapsedMilliseconds / 1000f:0.0} s");

            foreach (string s in oRep.aViolations)
                Library.Log("DfAM VIOLATION - " + s);
            Library.Log("DfAM voxel verdict: "
                      + (oRep.bFeasible ? "PASS" : "FAIL")
                      + $"  g = [{oRep.afG[0]:+0.0000;-0.0000}, "
                      + $"{oRep.afG[1]:+0.0000;-0.0000}, "
                      + $"{oRep.afG[2]:+0.0000;-0.0000}] "
                      + $"({string.Join(", ", DfamVoxelReport.astrGNames)})");
            return oRep;
        }

        // -- drainability -------------------------------------------------

        /// <summary>
        /// Splits the void inside the casing envelope into cavities and
        /// counts, for each one, the DISTINCT PATCHES through which it
        /// breaks out of the envelope. Every opening counts the same way -
        /// a drain port, the diffuser inlet, the turbine exit - so no
        /// arbitrary classification is needed. Sub-cell openings (the
        /// effusion holes) simply do not resolve, which is exactly the rule.
        /// </summary>
        static Dictionary<string, object> CheckDrainExits(
            int[] anLabel, bool[] abEnvelope, int nx, int ny, int nz,
            Vector3 vecMin, float fCell, CombustorParameters p,
            DfamVoxelReport oRep)
        {
            int nCells = anLabel.Length;
            // -1 = not part of the powder domain, 0 = unlabelled cavity cell
            var anCav = new int[nCells];
            for (int n = 0; n < nCells; n++)
                anCav[n] = (anLabel[n] == -1 || !abEnvelope[n]) ? -1 : 0;

            float fCellVol = fCell * fCell * fCell;
            var aPorts = Casing.aDrainPorts(p);
            var anPortCell = new int[aPorts.Count];
            for (int i = 0; i < aPorts.Count; i++)
                anPortCell[i] = nNearestVoidCell(
                    anCav, Casing.vecPortSeed(p, aPorts[i]), vecMin, fCell,
                    nx, ny, nz);

            // label every cavity
            var aComps = new List<CompResult>();
            int nLabel = 1;
            for (int n = 0; n < nCells; n++)
            {
                if (anCav[n] != 0)
                    continue;
                aComps.Add(LabelComponent(anCav, n, nLabel++, nx, ny, nz));
            }

            // mark the break-out cells: a cavity cell with a 6-neighbour that
            // is void but OUTSIDE the envelope
            var anEsc = new int[nCells];
            int nXY = nx * ny;
            for (int n = 0; n < nCells; n++)
            {
                anEsc[n] = -1;
                if (anCav[n] <= 0)
                    continue;
                int k = n / nXY;
                int r = n - k * nXY;
                int j = r / nx;
                int i = r - j * nx;
                bool bEsc =
                    (i > 0 && bOutsideVoid(anLabel, abEnvelope, n - 1))
                 || (i < nx - 1 && bOutsideVoid(anLabel, abEnvelope, n + 1))
                 || (j > 0 && bOutsideVoid(anLabel, abEnvelope, n - nx))
                 || (j < ny - 1 && bOutsideVoid(anLabel, abEnvelope, n + nx))
                 || (k > 0 && bOutsideVoid(anLabel, abEnvelope, n - nXY))
                 || (k < nz - 1 && bOutsideVoid(anLabel, abEnvelope, n + nXY));
                if (bEsc)
                    anEsc[n] = 0;                 // unlabelled break-out cell
            }

            // count the connected break-out patches of each cavity
            var oExits = new Dictionary<int, int>();
            int nEscLabel = 1;
            for (int n = 0; n < nCells; n++)
            {
                if (anEsc[n] != 0)
                    continue;
                int nCav = anCav[n];
                LabelComponent(anEsc, n, nEscLabel++, nx, ny, nz);
                oExits[nCav] = oExits.TryGetValue(nCav, out int nPrev)
                    ? nPrev + 1 : 1;
            }

            var aCavities = new List<Dictionary<string, object>>();
            int nMinExits = int.MaxValue;
            int nCounted = 0;
            foreach (CompResult oComp in aComps)
            {
                float fVol = oComp.nCells * fCellVol;
                if (fVol < 10f * fCellVol)
                    continue;                     // staircase noise

                int nExits = oExits.TryGetValue(oComp.nLabel, out int nE)
                    ? nE : 0;
                var aPortIdx = new List<int>();
                for (int i = 0; i < aPorts.Count; i++)
                    if (anPortCell[i] >= 0
                        && anCav[anPortCell[i]] == oComp.nLabel)
                        aPortIdx.Add(i);

                float fZ0 = vecMin.Z + (oComp.nK0 + 0.5f) * fCell;
                float fZ1 = vecMin.Z + (oComp.nK1 + 0.5f) * fCell;
                nMinExits = Math.Min(nMinExits, nExits);
                nCounted++;

                string strId = "cavity" + oComp.nLabel.ToString("000");
                aCavities.Add(new Dictionary<string, object>
                {
                    { "id", strId },
                    { "volume_mm3", fVol },
                    { "z_range_mm", new float[] { fZ0, fZ1 } },
                    { "centroid_mm", new float[]
                        {
                            vecMin.X + (oComp.fCx + 0.5f) * fCell,
                            vecMin.Y + (oComp.fCy + 0.5f) * fCell,
                            vecMin.Z + (oComp.fCz + 0.5f) * fCell,
                        } },
                    { "n_exits", nExits },
                    { "served_by_drain_ports", aPortIdx.ToArray() },
                    { "ok", nExits >= nDrainExitsMin },
                });
                if (nExits < nDrainExitsMin)
                    oRep.aViolations.Add(
                        $"drainability: cavity {strId} ({fVol:0.#} mm3, z "
                        + $"{fZ0:0.#} -> {fZ1:0.#} mm, centroid r = "
                        + $"{fRadiusOf(oComp, vecMin, fCell):0.#} mm) breaks "
                        + $"out of the machine at {nExits} place(s), minimum "
                        + $"{nDrainExitsMin}. Add drain ports to "
                        + "dfam.drain_ports; effusion holes do NOT count "
                        + "(EstadoDelArte 4.4).");
            }
            if (nMinExits == int.MaxValue)
                nMinExits = nDrainExitsMin;       // nothing to drain
            oRep.afG[2] = (nDrainExitsMin - nMinExits) / (float)nDrainExitsMin;

            var aPortDicts = new List<Dictionary<string, object>>();
            for (int i = 0; i < aPorts.Count; i++)
                aPortDicts.Add(new Dictionary<string, object>
                {
                    { "index", i },
                    { "z_mm", aPorts[i].ZMm },
                    { "d_mm", aPorts[i].DMm },
                    { "phi_deg", aPorts[i].PhiDeg },
                    { "reaches_a_cavity", anPortCell[i] >= 0 },
                });

            return new Dictionary<string, object>
            {
                { "method", "the void inside the casing envelope is split "
                          + "into connected cavities; the exits of a cavity "
                          + "are the DISTINCT connected patches through "
                          + "which it breaks out of the envelope (drain "
                          + "ports, inlet, exit - all counted alike)" },
                { "n_declared_ports", aPorts.Count },
                { "declared_ports", aPortDicts },
                { "n_cavities", nCounted },
                { "min_exits", nMinExits },
                { "exits_min_required", nDrainExitsMin },
                { "cavities", aCavities },
                { "effusion_counts_as_drain", false },
            };
        }

        static bool bOutsideVoid(int[] anLabel, bool[] abEnvelope, int n)
            => anLabel[n] != -1 && !abEnvelope[n];

        static float fRadiusOf(CompResult oComp, Vector3 vecMin, float fCell)
        {
            float fX = vecMin.X + (oComp.fCx + 0.5f) * fCell;
            float fY = vecMin.Y + (oComp.fCy + 0.5f) * fCell;
            return MathF.Sqrt(fX * fX + fY * fY);
        }

        /// <summary>Index of the void cell nearest to a world point, searched
        /// in a small neighbourhood (the exact seed may land in the wall).
        /// Returns -1 when nothing usable is within 3 cells.</summary>
        static int nNearestVoidCell(int[] anCav, Vector3 vecPt,
                                    Vector3 vecMin, float fCell,
                                    int nx, int ny, int nz)
        {
            int i0 = (int)MathF.Floor((vecPt.X - vecMin.X) / fCell);
            int j0 = (int)MathF.Floor((vecPt.Y - vecMin.Y) / fCell);
            int k0 = (int)MathF.Floor((vecPt.Z - vecMin.Z) / fCell);
            for (int r = 0; r <= 3; r++)
                for (int dk = -r; dk <= r; dk++)
                    for (int dj = -r; dj <= r; dj++)
                        for (int di = -r; di <= r; di++)
                        {
                            int i = i0 + di, j = j0 + dj, k = k0 + dk;
                            if (i < 0 || i >= nx || j < 0 || j >= ny
                                || k < 0 || k >= nz)
                                continue;
                            int n = nIdx(i, j, k, nx, ny);
                            if (anCav[n] == 0)
                                return n;
                        }
            return -1;
        }

        // -- connected components -----------------------------------------

        struct CompResult
        {
            public int nLabel;
            public int nCells;
            public int nK0, nK1;
            public float fCx, fCy, fCz;
        }

        /// <summary>Flood fill of the void from the grid boundary. Every cell
        /// reached gets label 1 = "connected to the outside world". Whatever
        /// stays at 0 afterwards is sealed inside the part.</summary>
        static void FloodFromGridBoundary(int[] anLabel, int nx, int ny,
                                          int nz)
        {
            var aStack = new Stack<int>();
            void Seed(int i, int j, int k)
            {
                int n = nIdx(i, j, k, nx, ny);
                if (anLabel[n] == 0)
                {
                    anLabel[n] = 1;
                    aStack.Push(n);
                }
            }
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                {
                    Seed(i, j, 0);
                    Seed(i, j, nz - 1);
                }
            for (int k = 0; k < nz; k++)
            {
                for (int i = 0; i < nx; i++)
                {
                    Seed(i, 0, k);
                    Seed(i, ny - 1, k);
                }
                for (int j = 0; j < ny; j++)
                {
                    Seed(0, j, k);
                    Seed(nx - 1, j, k);
                }
            }
            Propagate(anLabel, aStack, 1, nx, ny, nz, null);
        }

        /// <summary>Labels one 6-connected component of cells currently at 0
        /// and returns its statistics (cell count, z extent, centroid in
        /// CELL INDEX units).</summary>
        static CompResult LabelComponent(int[] anLabel, int nStart,
                                         int nLabel, int nx, int ny, int nz)
        {
            int nXY = nx * ny;
            long lCount = 0;
            double fSx = 0.0, fSy = 0.0, fSz = 0.0;
            int nK0 = int.MaxValue, nK1 = int.MinValue;

            var aStack = new Stack<int>();
            anLabel[nStart] = nLabel;
            aStack.Push(nStart);
            Propagate(anLabel, aStack, nLabel, nx, ny, nz, n =>
            {
                int k = n / nXY;
                int r = n - k * nXY;
                int j = r / nx;
                int i = r - j * nx;
                lCount++;
                fSx += i; fSy += j; fSz += k;
                if (k < nK0) nK0 = k;
                if (k > nK1) nK1 = k;
            });

            double fInv = lCount > 0 ? 1.0 / lCount : 0.0;
            return new CompResult
            {
                nLabel = nLabel,
                nCells = (int)Math.Min(lCount, int.MaxValue),
                nK0 = nK0 == int.MaxValue ? 0 : nK0,
                nK1 = nK1 == int.MinValue ? 0 : nK1,
                fCx = (float)(fSx * fInv),
                fCy = (float)(fSy * fInv),
                fCz = (float)(fSz * fInv),
            };
        }

        /// <summary>
        /// Propagation over cells whose label is 0, using 18-connectivity
        /// (the 6 faces plus the 12 edges).
        ///
        /// Digital topology requires COMPLEMENTARY connectivities for the
        /// solid and the void: taking 6 for both makes thin channels read as
        /// sealed purely because the grid phase happened to land badly, and
        /// then two identical vaporizer tubes get opposite verdicts, which is
        /// worse than useless. Edge connectivity fixes that. Corners (26) are
        /// deliberately NOT used: they would let the void leak diagonally
        /// through a one-cell wall, and a false "drains fine" is the
        /// dangerous direction here (EstadoDelArte 4.4). With the wall floor
        /// at 2 voxels and the coarse cell at 1 mm, every printable wall is
        /// at least one full cell thick, so edge leakage cannot happen.
        /// </summary>
        static void Propagate(int[] anLabel, Stack<int> aStack, int nLabel,
                              int nx, int ny, int nz, Action<int> fnVisit)
        {
            int nXY = nx * ny;
            while (aStack.Count > 0)
            {
                int n = aStack.Pop();
                fnVisit?.Invoke(n);
                int k = n / nXY;
                int r = n - k * nXY;
                int j = r / nx;
                int i = r - j * nx;

                for (int dk = -1; dk <= 1; dk++)
                {
                    int kk = k + dk;
                    if (kk < 0 || kk >= nz) continue;
                    for (int dj = -1; dj <= 1; dj++)
                    {
                        int jj = j + dj;
                        if (jj < 0 || jj >= ny) continue;
                        for (int di = -1; di <= 1; di++)
                        {
                            // 18-connectivity: skip the 8 corners and self
                            if (Math.Abs(di) + Math.Abs(dj) + Math.Abs(dk) > 2
                                || (di == 0 && dj == 0 && dk == 0))
                                continue;
                            int ii = i + di;
                            if (ii < 0 || ii >= nx) continue;
                            int m = nIdx(ii, jj, kk, nx, ny);
                            if (anLabel[m] == 0)
                            {
                                anLabel[m] = nLabel;
                                aStack.Push(m);
                            }
                        }
                    }
                }
            }
        }

        static int nIdx(int i, int j, int k, int nx, int ny)
            => (k * ny + j) * nx + i;

        static void GridSize(Vector3 vecSize, ref float fCell, out int nx,
                             out int ny, out int nz, string strWhat)
        {
            while (true)
            {
                nx = Math.Max(3, (int)MathF.Ceiling(vecSize.X / fCell));
                ny = Math.Max(3, (int)MathF.Ceiling(vecSize.Y / fCell));
                nz = Math.Max(3, (int)MathF.Ceiling(vecSize.Z / fCell));
                if ((long)nx * ny * nz <= lMaxCells)
                    return;
                float fNew = fCell * 1.25f;
                Library.Log($"DfAM ({strWhat}): coarse cell {fCell:0.###} -> "
                          + $"{fNew:0.###} mm so the grid fits under "
                          + $"{lMaxCells / 1_000_000} M cells");
                fCell = fNew;
            }
        }

        // -- envelope of the machine --------------------------------------
        //
        // The envelope is the casing BORE, not its outer skin: it is what
        // separates "powder trapped inside the machine" from "powder in the
        // rest of the build chamber". Two consequences, both wanted:
        //   * no margin anywhere. A margin would wrap the part in a shell of
        //     free space, and the staircase of the outer surface would then
        //     read as a swarm of spurious slivers.
        //   * the drain-port bores, which pierce the wall, sit OUTSIDE the
        //     envelope. That is precisely what makes a cavity reaching a
        //     port count as breaking out of the machine.

        static bool bZInsideEnvelope(CombustorParameters p, float fZ)
        {
            float fZ0 = p.Casing.Profile[0][0];
            float fZ1 = p.Casing.Profile[p.Casing.Profile.Length - 1][0];
            return fZ >= fZ0 && fZ <= fZ1;
        }

        static float fEnvelopeRAt(CombustorParameters p, float fZ)
            => p.fCasingRAt(fZ);
    }
}
