//
// PhyCBImport.cs - bridge from the Quasar Phy-CB design pipeline.
//
// Phy-CB (physics_core.py / neural_optimizer.py / geometry_generator.py)
// optimizes a 20-D design vector and writes the verified result as
// `combustor.json` (schema phycb-combustor-1) inside <outdir>/geometry/.
// This class maps that file onto CombustorParameters so the optimized
// design can be built as printable STLs:
//
//   var p = PhyCBImport.FromContractJson(@"runs\x\geometry\combustor.json");
//   CombustorBuilder.Build(p, outputDir);
//
// READING IS DEFENSIVE BY CONTRACT (CONTEXTO section 10.3): a committed
// artefact may be OLDER than the code. Every optional key falls back to a
// documented neutral default and the fallback is recorded in
// p.Notes / p.Warnings, which the builder logs. Nothing here throws except
// for a file that is not a Phy-CB geometry contract at all.
//

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;

namespace CombustorDesigner
{
    public static class PhyCBImport
    {
        public static CombustorParameters FromContractJson(
            string strJsonPath, string strOutputName = null)
        {
            using JsonDocument oDoc =
                JsonDocument.Parse(File.ReadAllText(strJsonPath));
            JsonElement oRoot = oDoc.RootElement;

            if (oRoot.ValueKind != JsonValueKind.Object)
                throw new ArgumentException(
                    $"'{strJsonPath}' is not a JSON object");

            var p = new CombustorParameters();

            // -- meta ------------------------------------------------------
            p.Schema = strOpt(oRoot, "schema", "");
            if (p.Schema.Length > 0 && !p.Schema.StartsWith("phycb-combustor"))
                throw new ArgumentException(
                    $"'{strJsonPath}' declares schema '{p.Schema}' - this " +
                    "builder consumes 'phycb-combustor-1'");
            p.Arch = strOpt(oRoot, "arch", "can");
            if (!oRoot.TryGetProperty("_conventions", out _))
                p.Notes.Add("the contract carries no `_conventions` key; the "
                          + "sign conventions assumed are the ones documented "
                          + "in Parameters.cs (WAVE3 1.1).");

            // -- liner -----------------------------------------------------
            if (!oTry(oRoot, "liner", out JsonElement oLiner))
                throw new ArgumentException(
                    $"'{strJsonPath}' has no `liner` block - nothing to build");
            p.Liner.Profile = afProfile(oLiner, "profile", 3);
            p.Liner.TWallMm = fOpt(oLiner, "t_wall_mm", p.Liner.TWallMm,
                                   p, "liner.t_wall_mm");
            p.Liner.Material = strOpt(oLiner, "material", p.Liner.Material);
            p.Liner.FilletRMm = fFirst(oLiner,
                new[] { "fillet_r_mm", "dome_fillet_r_mm", "root_fillet_r_mm" },
                p.Liner.FilletRMm);
            if (oTry(oLiner, "dome", out JsonElement oDome))
            {
                p.Liner.DomeType = strOpt(oDome, "type", p.Liner.DomeType);
                if (oTry(oDome, "z_mm", out JsonElement eDz)
                    && bNum(eDz, out float fDz))
                {
                    p.Liner.DomeZMm = fDz;
                    p.Liner.bDomeZGiven = true;
                }
            }
            else
            {
                p.Notes.Add("liner.dome absent - a hemispherical dome is "
                          + "closed on the upstream end of the profile.");
            }
            // double wall: three documented aliases (dfam_checker reads the
            // same three), all optional
            JsonElement oDW = default;
            bool bDW = oTry(oLiner, "double_wall", out oDW)
                    || oTry(oRoot, "double_wall", out oDW);
            if (!bDW && oTry(oLiner, "cooling", out JsonElement oCool)
                && strOpt(oCool, "type", "") == "double_wall")
            {
                oDW = oCool;
                bDW = true;
            }
            if (bDW)
            {
                p.Liner.DoubleWall = new DoubleWallParams
                {
                    GapMm = fOpt(oDW, "gap_mm", 1.0f),
                    Z0Mm = fOpt(oDW, "z0_mm", p.Liner.Profile[0][0]),
                    Z1Mm = fOpt(oDW, "z1_mm",
                        p.Liner.Profile[p.Liner.Profile.Length - 1][0]),
                    TInnerMm = fOpt(oDW, "t_inner_mm", p.Liner.TWallMm),
                    TOuterMm = fOpt(oDW, "t_outer_mm", p.Liner.TWallMm),
                };
            }

            // -- casing ----------------------------------------------------
            if (oTry(oRoot, "casing", out JsonElement oCase))
            {
                p.Casing.Profile = afProfile(oCase, "profile", 2);
                p.Casing.TWallMm = fOpt(oCase, "t_wall_mm", p.Casing.TWallMm,
                                        p, "casing.t_wall_mm");
                p.Casing.Material = strOpt(oCase, "material",
                                           p.Casing.Material);
                p.Casing.FlangeBoltCount = (int)fOpt(oCase,
                    "flange_bolt_count", p.Casing.FlangeBoltCount);
                p.Casing.FlangeBoltDMm = fOpt(oCase, "flange_bolt_d_mm",
                                              p.Casing.FlangeBoltDMm);
                p.Casing.FlangeWMm = fOpt(oCase, "flange_w_mm",
                                          p.Casing.FlangeWMm);
                p.Casing.FlangeTMm = fOpt(oCase, "flange_t_mm",
                                          p.Casing.FlangeTMm);
                p.Casing.HeadPlateTMm = fOpt(oCase, "head_plate_t_mm",
                                             p.Casing.HeadPlateTMm);
                p.Casing.BossHMm = fOpt(oCase, "boss_h_mm", p.Casing.BossHMm);
            }
            if (p.Casing.Profile == null && p.Liner.Profile != null)
            {
                // No casing in the contract: synthesise a plain sleeve 20 %
                // outside the liner so the assembly is still buildable, and
                // say so loudly.
                var af = new float[p.Liner.Profile.Length][];
                for (int i = 0; i < af.Length; i++)
                    af[i] = new float[]
                    {
                        p.Liner.Profile[i][0],
                        p.Liner.Profile[i][2] * 1.2f + 6f,
                    };
                p.Casing.Profile = af;
                p.Warnings.Add("the contract has no `casing` block - a plain "
                             + "sleeve 20 % + 6 mm outside the liner was "
                             + "synthesised so the assembly builds. Do NOT "
                             + "print it.");
            }

            // -- swirler ---------------------------------------------------
            if (oTry(oRoot, "swirler", out JsonElement oSw))
            {
                p.Swirler.Type = strOpt(oSw, "type", p.Swirler.Type);
                p.Swirler.NVanes = (int)fFirst(oSw,
                    new[] { "n_vanes", "n_blades", "count" }, 0f);
                p.Swirler.BetaDeg = fFirst(oSw,
                    new[] { "beta_deg", "vane_angle_deg" }, p.Swirler.BetaDeg);
                p.Swirler.BetaHubDeg = fOpt(oSw, "beta_hub_deg",
                                            p.Swirler.BetaHubDeg);
                p.Swirler.SN = fOpt(oSw, "SN", p.Swirler.SN);
                p.Swirler.DHubMm = fOpt(oSw, "d_hub_mm", p.Swirler.DHubMm);
                p.Swirler.DTipMm = fOpt(oSw, "d_tip_mm", p.Swirler.DTipMm);
                p.Swirler.VaneTMm = fFirst(oSw,
                    new[] { "vane_t_mm", "t_vane_mm", "thickness_mm" },
                    p.Swirler.VaneTMm);
                p.Swirler.ZMm = fOpt(oSw, "z_mm", p.Swirler.ZMm);
                p.Swirler.ChordMm = fFirst(oSw,
                    new[] { "chord_mm", "l_mm", "length_mm" }, 0f);
                p.Swirler.MonolithicWithDome =
                    bOpt(oSw, "monolithic_with_dome", false);
            }

            // -- injectors -------------------------------------------------
            if (oTry(oRoot, "injectors", out JsonElement oInj))
            {
                p.Injectors.Type = strOpt(oInj, "type", p.Injectors.Type);
                p.Injectors.Count = Math.Max(1,
                    (int)fOpt(oInj, "count", p.Injectors.Count));
                p.Injectors.ALR = fOpt(oInj, "ALR", p.Injectors.ALR);
                p.Injectors.RingRMm = fFirst(oInj,
                    new[] { "ring_r_mm", "pitch_r_mm", "r_mm" }, 0f);
                p.Injectors.Phi0Deg = fOpt(oInj, "phi0_deg", 0f);
                p.Injectors.BodyDMm = fFirst(oInj,
                    new[] { "body_d_mm", "d_out_mm", "D_p_mm" }, 0f);
                p.Injectors.BoreDMm = fFirst(oInj,
                    new[] { "bore_d_mm", "d_in_mm", "orifice_d_mm" }, 0f);
                p.Injectors.PrefilmerTMm = fOpt(oInj, "prefilmer_t_mm", 0f);
                p.Injectors.MicromixN = (int)fFirst(oInj,
                    new[] { "micromix_n", "n_jets", "n" }, 0f);
                p.Injectors.MicromixDMm = fFirst(oInj,
                    new[] { "micromix_d_mm", "jet_d_mm", "d_jet_mm" }, 0f);

                // vaporizer tube: the flat keys win, the nested block is the
                // documented alias
                JsonElement oVap = oInj;
                if (oTry(oInj, "vaporizer", out JsonElement oV))
                    oVap = oV;
                p.Injectors.TubeDMm = fFirst(oVap,
                    new[] { "tube_d_mm", "d_tube_mm", "tube_od_mm" }, 0f);
                if (p.Injectors.TubeDMm <= 0f)
                    p.Injectors.TubeDMm = fFirst(oInj,
                        new[] { "tube_d_mm", "d_tube_mm", "tube_od_mm" }, 0f);
                p.Injectors.TubeTMm = fFirst(oVap,
                    new[] { "tube_t_mm", "t_tube_mm" }, 0f);
                p.Injectors.TubeZ0Mm = fFirst(oVap,
                    new[] { "tube_z0_mm", "z0_mm" }, float.NaN);
                p.Injectors.TubeZ1Mm = fFirst(oVap,
                    new[] { "tube_z1_mm", "z1_mm" }, float.NaN);
                p.Injectors.BendDeg = fFirst(oVap,
                    new[] { "bend_deg", "elbow_deg" }, p.Injectors.BendDeg);
                p.Injectors.BendRMm = fFirst(oVap,
                    new[] { "bend_r_mm", "elbow_r_mm" }, 0f);
                // `tube_id_mm` is the third documented alias: it gives the
                // BORE, so the wall follows from the outer diameter
                float fId = fFirst(oVap, new[] { "tube_id_mm" }, 0f);
                if (fId > 0f && p.Injectors.TubeDMm > fId)
                    p.Injectors.TubeTMm = 0.5f * (p.Injectors.TubeDMm - fId);
            }

            // -- hole rows -------------------------------------------------
            if (oTry(oRoot, "hole_rows", out JsonElement oRows)
                && oRows.ValueKind == JsonValueKind.Array)
            {
                int i = 0;
                foreach (JsonElement oRow in oRows.EnumerateArray())
                {
                    if (oRow.ValueKind != JsonValueKind.Object) { i++; continue; }
                    var row = new HoleRowParams
                    {
                        Zone = strOpt(oRow, "zone", "pz"),
                        Id = strOpt(oRow, "id", null),
                        ZMm = fOpt(oRow, "z_mm", 0f),
                        RMm = fOpt(oRow, "r_mm", 0f),
                        N = Math.Max(1, (int)fOpt(oRow, "n", 1f)),
                        DMm = fOpt(oRow, "d_mm", 1f),
                        Cd = fOpt(oRow, "Cd", 0.62f),
                        Shape = strOpt(oRow, "shape", "circle"),
                        Plunged = bOpt(oRow, "plunged", false),
                        AngleDeg = fOpt(oRow, "angle_deg", 0f),
                        Phi0Deg = fOpt(oRow, "phi0_deg", 0f),
                    };
                    if (oTry(oRow, "axis", out JsonElement oAx)
                        && oAx.ValueKind == JsonValueKind.Array)
                    {
                        var af = new List<float>();
                        foreach (JsonElement e in oAx.EnumerateArray())
                            if (bNum(e, out float f)) af.Add(f);
                        if (af.Count >= 3) row.Axis = af.ToArray();
                    }
                    if (row.Id == null) row.Id = "row" + i.ToString("00");
                    p.HoleRows.Add(row);
                    i++;
                }
            }

            // -- effusion field --------------------------------------------
            if (oTry(oRoot, "effusion", out JsonElement oEff))
            {
                p.Effusion.DMm = fOpt(oEff, "d_mm", 0f);
                p.Effusion.PitchXOverD = fOpt(oEff, "pitch_x_over_d",
                                              p.Effusion.PitchXOverD);
                p.Effusion.PitchYOverD = fOpt(oEff, "pitch_y_over_d",
                                              p.Effusion.PitchYOverD);
                p.Effusion.AngleDeg = fOpt(oEff, "angle_deg",
                                           p.Effusion.AngleDeg);
                p.Effusion.Shape = strOpt(oEff, "shape", p.Effusion.Shape);
                p.Effusion.ConeExpansion = fOpt(oEff, "cone_expansion",
                                                p.Effusion.ConeExpansion);
                p.Effusion.Staggered = bOpt(oEff, "staggered", true);
                if (oTry(oEff, "zones", out JsonElement oZones)
                    && oZones.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement oZ in oZones.EnumerateArray())
                    {
                        if (oZ.ValueKind != JsonValueKind.Object) continue;
                        p.Effusion.Zones.Add(new EffusionZoneParams
                        {
                            Z0Mm = fFirst(oZ, new[] { "z0_mm", "z_start_mm" }, 0f),
                            Z1Mm = fFirst(oZ, new[] { "z1_mm", "z_end_mm" }, 0f),
                            Rows = (int)fOpt(oZ, "rows", 0f),
                        });
                    }
                }
            }

            // -- DfAM block ------------------------------------------------
            if (oTry(oRoot, "dfam", out JsonElement oDfam))
            {
                p.Dfam.BuildOrientation = strOpt(oDfam, "build_orientation",
                                                 p.Dfam.BuildOrientation);
                p.Dfam.MinWallMm = fOpt(oDfam, "min_wall_mm", 0f);
                p.Dfam.VoxelMm = fOpt(oDfam, "voxel_mm", p.Dfam.VoxelMm);
                p.Dfam.CoarseVoxelMm = fOpt(oDfam, "coarse_voxel_mm",
                                            p.Dfam.CoarseVoxelMm);
                p.Dfam.OverhangFracMax = fOpt(oDfam, "overhang_frac_max",
                                              p.Dfam.OverhangFracMax);
                if (oTry(oDfam, "drain_ports", out JsonElement oPorts)
                    && oPorts.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement oP in oPorts.EnumerateArray())
                    {
                        if (oP.ValueKind != JsonValueKind.Object) continue;
                        p.Dfam.DrainPorts.Add(new DrainPortParams
                        {
                            ZMm = fOpt(oP, "z_mm", 0f),
                            DMm = fOpt(oP, "d_mm", 5f),
                            PhiDeg = fOpt(oP, "phi_deg", float.NaN),
                            bBoss = bOpt(oP, "boss", true),
                        });
                    }
                }
            }
            else
            {
                p.Notes.Add("no `dfam` block - build orientation z+, voxel "
                          + "0.5 mm and NO declared drain ports assumed.");
            }
            if (p.Dfam.DrainPorts.Count == 0)
                p.Warnings.Add("the contract declares no drain ports. Powder "
                             + "drainability will be judged on the voxel field "
                             + "alone (EstadoDelArte 4.4 asks for >= 2 exits "
                             + "with line of sight per cavity).");
            p.VoxelSizeMm = p.Dfam.VoxelMm > 0f ? p.Dfam.VoxelMm : 0.5f;

            // -- output name -----------------------------------------------
            string strName = strOutputName;
            if (string.IsNullOrEmpty(strName))
            {
                strName = null;
                if (oTry(oRoot, "run_meta", out JsonElement oMeta))
                    strName = strOpt(oMeta, "name", null);
                if (string.IsNullOrEmpty(strName))
                    strName = "PhyCB_" + strSanitize(p.Arch);
            }
            p.OutputName = strSanitize(strName);
            return p;
        }

        // -- defensive readers --------------------------------------------

        static bool oTry(JsonElement o, string strName, out JsonElement e)
        {
            e = default;
            return o.ValueKind == JsonValueKind.Object
                && o.TryGetProperty(strName, out e)
                && e.ValueKind != JsonValueKind.Null;
        }

        static bool bNum(JsonElement e, out float f)
        {
            f = 0f;
            if (e.ValueKind == JsonValueKind.Number)
            {
                if (!e.TryGetDouble(out double d)) return false;
                if (double.IsNaN(d) || double.IsInfinity(d)) return false;
                f = (float)d;
                return true;
            }
            if (e.ValueKind == JsonValueKind.String
                && float.TryParse(e.GetString(), NumberStyles.Float,
                                  CultureInfo.InvariantCulture, out f))
                return true;
            return false;
        }

        static float fOpt(JsonElement o, string strName, float fDefault)
            => oTry(o, strName, out JsonElement e) && bNum(e, out float f)
                ? f : fDefault;

        /// <summary>Numeric read that RECORDS the fallback: used for the
        /// keys whose absence changes the printed part.</summary>
        static float fOpt(JsonElement o, string strName, float fDefault,
                          CombustorParameters p, string strPath)
        {
            if (oTry(o, strName, out JsonElement e) && bNum(e, out float f))
                return f;
            p.Notes.Add($"`{strPath}` absent or unreadable - defaulted to "
                      + fDefault.ToString("0.###", CultureInfo.InvariantCulture));
            return fDefault;
        }

        static float fFirst(JsonElement o, string[] astrNames, float fDefault)
        {
            foreach (string s in astrNames)
                if (oTry(o, s, out JsonElement e) && bNum(e, out float f))
                    return f;
            return fDefault;
        }

        static string strOpt(JsonElement o, string strName, string strDefault)
        {
            if (!oTry(o, strName, out JsonElement e))
                return strDefault;
            if (e.ValueKind == JsonValueKind.String)
                return e.GetString();
            if (e.ValueKind == JsonValueKind.Number)
                return e.ToString();
            return strDefault;
        }

        static bool bOpt(JsonElement o, string strName, bool bDefault)
        {
            if (!oTry(o, strName, out JsonElement e))
                return bDefault;
            if (e.ValueKind == JsonValueKind.True) return true;
            if (e.ValueKind == JsonValueKind.False) return false;
            if (e.ValueKind == JsonValueKind.Number && bNum(e, out float f))
                return f != 0f;
            return bDefault;
        }

        /// <summary>Reads a [[z, ...], ...] polyline, pads short rows with
        /// zeros up to nCols, drops unreadable rows and sorts by z. Returns
        /// null when the key is absent.</summary>
        static float[][] afProfile(JsonElement o, string strName, int nCols)
        {
            if (!oTry(o, strName, out JsonElement e)
                || e.ValueKind != JsonValueKind.Array)
                return null;

            var aRows = new List<float[]>();
            foreach (JsonElement oPt in e.EnumerateArray())
            {
                if (oPt.ValueKind != JsonValueKind.Array) continue;
                var afVals = new List<float>();
                foreach (JsonElement v in oPt.EnumerateArray())
                    if (bNum(v, out float f)) afVals.Add(f);
                if (afVals.Count < 2) continue;

                var afRow = new float[nCols];
                if (nCols == 3 && afVals.Count == 2)
                {
                    // [z, r] on a 3-column profile => r_inner = 0 (a can)
                    afRow[0] = afVals[0];
                    afRow[1] = 0f;
                    afRow[2] = afVals[1];
                }
                else
                {
                    for (int i = 0; i < nCols; i++)
                        afRow[i] = i < afVals.Count ? afVals[i] : 0f;
                }
                aRows.Add(afRow);
            }
            if (aRows.Count == 0) return null;
            aRows.Sort((a, b) => a[0].CompareTo(b[0]));
            return aRows.ToArray();
        }

        static string strSanitize(string s)
        {
            if (string.IsNullOrEmpty(s)) return "PhyCB";
            var ac = s.ToCharArray();
            for (int i = 0; i < ac.Length; i++)
                if (!char.IsLetterOrDigit(ac[i]) && ac[i] != '_'
                    && ac[i] != '-')
                    ac[i] = '_';
            return new string(ac);
        }
    }
}
