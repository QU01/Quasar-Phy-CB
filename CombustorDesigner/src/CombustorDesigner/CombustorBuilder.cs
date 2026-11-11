//
// CombustorBuilder.cs - public entry point (Phy-AC / Phy-CC Build pattern).
//
//   1. Validates the parameters
//   2. Calls PicoGK.Library.Go(voxelSize, task, logFile, ...)
//   3. The task builds one part at a time, exports its STL and accumulates
//      the assembly (streaming keeps the peak memory at ONE part and gives
//      per-part timings in the log - the Phy-AC recipe)
//   4. Re-verifies DfAM on the built voxels and writes dfam_report.json
//   5. Exports the voxel field so the CFD layer consumes the SAME geometry
//   6. Returns the absolute path of the assembly STL
//
// PARTS (WAVE3 section 2 table)
//   <Name>_Liner.stl        revolution - hole rows - effusion field, dome,
//                           injector seats
//   <Name>_Casing.stl       shell + bolted flanges + drain ports + igniter
//                           boss
//   <Name>_Swirler.stl      lofted vanes on the hub (unioned into the liner
//                           when the contract says monolithic_with_dome)
//   <Name>_Injector.stl     or <Name>_Vaporizer{i}.stl, by injector type
//   <Name>.stl              the assembly - INSPECTION ONLY, DO NOT PRINT
//                           (the convention of both sibling systems). The
//                           parts interpenetrate at their seats on purpose.
//
// VOXEL FIELD EXPORT (WAVE3 section 2, "export adicional"). The assembly and
// the liner are written as OpenVDB narrow-band signed-distance grids
// (`<Name>_field.vdb`, `<Name>_Liner_field.vdb`) with
// Voxels.SaveToVdbFile. That format was chosen because
//   * it IS the kernel's native representation - no conversion, no loss;
//   * it round-trips through Voxels.voxFromVdbFile(path), so WAVE5/6 gets
//     the identical field this STL was meshed from, not a re-voxelization
//     of a triangle soup;
//   * OpenVDB readers exist for Python (openvdb / pyopenvdb) and for the
//     usual CFD meshers, so cfd/ can sample it directly.
// The choice, the units (mm) and the reload call are recorded in
// dfam_report.json under "voxel_field" so the CFD layer never has to guess.
//

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using PicoGK;

namespace CombustorDesigner
{
    public static class CombustorBuilder
    {
        /// <summary>Tag carried by the assembly STL in every artefact that
        /// mentions it.</summary>
        public const string strAssemblyTag =
            "inspection only - do not print";

        /// <summary>
        /// Build every part STL plus the assembly into
        /// <paramref name="strOutputDir"/> and return the assembly STL path.
        /// <paramref name="strPartFilter"/> (optional, case-insensitive
        /// substring, e.g. "Liner" or "Vaporizer3") builds only the matching
        /// parts and skips the assembly, the DfAM check and the field export
        /// - the fast path for inspecting a single piece.
        /// </summary>
        public static string Build(CombustorParameters p,
                                   string strOutputDir,
                                   bool bShowViewer = false,
                                   string strPartFilter = null)
        {
            string strErr = p.Validate();
            if (strErr != null)
                throw new ArgumentException("Invalid contract: " + strErr);

            Directory.CreateDirectory(strOutputDir);
            string strOutputStl = Path.GetFullPath(
                Path.Combine(strOutputDir, p.OutputName + ".stl"));
            string strLogFile = Path.Combine(strOutputDir, "PicoGK.log");

            PicoGK.Library.Go(
                p.VoxelSizeMm,
                () => BuildTask(p, strOutputStl, bShowViewer, strPartFilter),
                strLogFile,
                !bShowViewer,
                $"CombustorDesigner - {p.OutputName}");

            if (strPartFilter == null && !File.Exists(strOutputStl))
                throw new Exception(
                    "Build failed: the assembly STL was not written. Check "
                    + "the log at " + strLogFile);
            return strOutputStl;
        }

        // -- the task that runs inside the PicoGK library -----------------

        static void BuildTask(CombustorParameters p, string strOutputStl,
                              bool bShowViewer, string strFilter)
        {
            Library.Log($"QUASAR Phy-CB - building combustor '{p.OutputName}'"
                      + $" (arch '{p.Arch}', schema '{p.Schema}') at voxel "
                      + $"{p.VoxelSizeMm} mm"
                      + (strFilter != null ? $" - filter '{strFilter}'" : ""));
            Library.Log("Conventions: Z is the axis, the gas flows +Z, all "
                      + "lengths mm, STL binary in mm. Printable floor "
                      + $"max(0.6, 2*voxel) = {p.fFeatureFloorMm():0.###} mm.");
            foreach (string s in p.Notes)
                Library.Log("NOTE  " + s);
            foreach (string s in p.Warnings)
                Library.Log("WARN  " + s);
            int nWarnAlready = p.Warnings.Count;
            LatticeUtils.fVoxelSizeMm = p.VoxelSizeMm;

            string strDir = Path.GetDirectoryName(strOutputStl) ?? ".";
            string strBase = Path.GetFileNameWithoutExtension(strOutputStl);

            // the effusion field is shared: the liner subtracts it and the
            // report quotes its statistics
            var aEffLog = new List<string>();
            EffusionField oEff = p.Effusion.DMm > 0f
                              && p.Effusion.Zones.Count > 0
                ? new EffusionField(p, aEffLog) : null;
            foreach (string s in aEffLog)
                Library.Log("EFFUSION  " + s);

            var aParts = new List<Dictionary<string, object>>();
            var aOverhang = new List<PartOverhang>();
            Voxels voxAssembly = null;
            int nPart = 0;
            var oWatch = System.Diagnostics.Stopwatch.StartNew();

            foreach ((string strName, Func<Voxels> fnBuild)
                     in aPartBuilders(p, oEff))
            {
                if (strFilter != null && !strName.Contains(
                        strFilter, StringComparison.OrdinalIgnoreCase))
                    continue;

                long lMs0 = oWatch.ElapsedMilliseconds;
                Voxels vox = fnBuild();
                if (vox == null || vox.bIsEmpty())
                {
                    if (strName == "Liner" || strName == "Casing")
                        throw new Exception(
                            $"part '{strName}' is empty - check the contract "
                            + "profiles and the voxel size.");
                    Library.Log($"WARN  part '{strName}' came out empty - "
                              + "skipped (it is optional).");
                    continue;
                }
                long lMsBuilt = oWatch.ElapsedMilliseconds;
                vox.CalculateProperties(out float fVol, out BBox3 oBB);
                string strPartStl = Path.Combine(
                    strDir, $"{strBase}_{strName}.stl");
                long lMsMesh = SaveStl(vox, strPartStl, oWatch);
                float fSecs = (oWatch.ElapsedMilliseconds - lMs0) / 1000f;
                Library.Log($"TIMING part {strName}: build "
                          + $"{(lMsBuilt - lMs0) / 1000f:0.0} s, properties "
                          + $"{(lMsMesh - lMsBuilt) / 1000f:0.0} s, mesh+STL "
                          + $"{(oWatch.ElapsedMilliseconds - lMsMesh) / 1000f:0.0} s");
                Library.Log($"Part {strName}: {fVol:F0} mm^3 in {fSecs:F1} s "
                          + $"-> {strPartStl}");

                // overhang is a PER-PART question (each part goes on the
                // plate on its own), so it is measured here, while the part
                // is still in memory, so streaming is preserved.
                PartOverhang oOver = null;
                try
                {
                    oOver = DfamVoxelCheck.CheckPart(vox, p, strName);
                    aOverhang.Add(oOver);
                }
                catch (Exception e)
                {
                    Library.Log($"WARN  overhang check of '{strName}' failed: "
                              + e.GetType().Name + ": " + e.Message);
                }

                aParts.Add(new Dictionary<string, object>
                {
                    { "name", strName },
                    { "stl", Path.GetFullPath(strPartStl) },
                    { "volume_mm3", fVol },
                    { "bbox_mm", new float[]
                        { oBB.vecMin.X, oBB.vecMin.Y, oBB.vecMin.Z,
                          oBB.vecMax.X, oBB.vecMax.Y, oBB.vecMax.Z } },
                    { "build_s", fSecs },
                    { "printable", true },
                    { "frac_unsupported", oOver != null
                        ? (object)oOver.fFrac : null },
                });

                // D-392: the assembly grows IN PLACE and each part's grid is
                // released as soon as it is merged (the first part becomes
                // the assembly itself). Streaming then really keeps one part
                // plus the assembly in memory.
                if (bShowViewer)
                    Library.oViewer().Add(vox, nPart++ % 4);
                if (strFilter == null)
                {
                    if (voxAssembly == null)
                        voxAssembly = vox;
                    else
                    {
                        voxAssembly.BoolAdd(vox);
                        if (!bShowViewer)
                            vox.Dispose();
                    }
                }
                else if (!bShowViewer)
                    vox.Dispose();
            }

            if (strFilter != null)
            {
                Library.Log("Part filter active - assembly, DfAM "
                          + "re-verification and field export skipped.");
                return;
            }

            // -- assembly (inspection only) -------------------------------
            long lMsAsm = oWatch.ElapsedMilliseconds;
            voxAssembly.CalculateProperties(out float fVolAsm,
                                            out BBox3 oBBAsm);
            SaveStl(voxAssembly, strOutputStl, oWatch);
            Library.Log($"TIMING assembly mesh+STL: "
                      + $"{(oWatch.ElapsedMilliseconds - lMsAsm) / 1000f:0.0} s");
            Library.Log($"Exported STL: {strOutputStl} ({fVolAsm:F0} mm^3) - "
                      + strAssemblyTag.ToUpperInvariant()
                      + ". The parts interpenetrate at their seats on "
                      + "purpose; print the per-part STLs.");
            aParts.Add(new Dictionary<string, object>
            {
                { "name", "Assembly" },
                { "stl", Path.GetFullPath(strOutputStl) },
                { "volume_mm3", fVolAsm },
                { "bbox_mm", new float[]
                    { oBBAsm.vecMin.X, oBBAsm.vecMin.Y, oBBAsm.vecMin.Z,
                      oBBAsm.vecMax.X, oBBAsm.vecMax.Y, oBBAsm.vecMax.Z } },
                { "printable", false },
                { "note", strAssemblyTag },
            });

            // -- DfAM re-verification on the built voxels -----------------
            DfamVoxelReport oDfam = null;
            long lMsDfam = oWatch.ElapsedMilliseconds;
            try
            {
                oDfam = DfamVoxelCheck.Run(voxAssembly, p, aOverhang);
                Library.Log($"TIMING DfAM voxel re-verification: "
                          + $"{(oWatch.ElapsedMilliseconds - lMsDfam) / 1000f:0.0} s");
            }
            catch (Exception e)
            {
                Library.Log("WARN  the DfAM voxel re-verification failed: "
                          + e.GetType().Name + ": " + e.Message
                          + " - the STLs are still valid; the report will "
                          + "carry the failure.");
            }

            // -- voxel field export for the CFD layer ---------------------
            var oField = new Dictionary<string, object>();
            try
            {
                string strVdb = Path.Combine(strDir, strBase + "_field.vdb");
                long lMsVdb = oWatch.ElapsedMilliseconds;
                voxAssembly.SaveToVdbFile(strVdb);
                Library.Log($"TIMING voxel field export: "
                          + $"{(oWatch.ElapsedMilliseconds - lMsVdb) / 1000f:0.0} s");
                string strVdbLiner = null;
                oField["format"] = "openvdb-sdf-narrowband";
                oField["units"] = "mm";
                oField["voxel_mm"] = p.VoxelSizeMm;
                oField["assembly_vdb"] = Path.GetFullPath(strVdb);
                oField["liner_vdb"] = strVdbLiner;
                oField["reload_csharp"] =
                    "PicoGK.Voxels.voxFromVdbFile(path)";
                oField["reload_python"] =
                    "openvdb.read(path) / pyopenvdb.readAllGridMetadata(path)";
                oField["why"] =
                    "the kernel's own narrow-band SDF: WAVE5/6 samples the "
                    + "SAME field this STL was meshed from, not a "
                    + "re-voxelization of a triangle soup (WAVE3 section 2, "
                    + "CONTEXTO trap 10.4)";
                oField["convention"] =
                    "negative inside the solid, positive outside, distances "
                    + "in mm; Z is the machine axis and the gas flows +Z";
                Library.Log($"Exported voxel field (OpenVDB narrow-band SDF, "
                          + $"mm): {strVdb}");
            }
            catch (Exception e)
            {
                oField["error"] = e.GetType().Name + ": " + e.Message;
                Library.Log("WARN  could not export the voxel field: "
                          + e.Message);
            }

            for (int i = nWarnAlready; i < p.Warnings.Count; i++)
                Library.Log("WARN  " + p.Warnings[i]);

            WriteReport(p, strDir, aParts, oEff, oDfam, oField);
            Library.Log($"TIMING total: {oWatch.ElapsedMilliseconds / 1000f:0.0} s");
        }

        /// <summary>Meshes the field and writes the binary STL, releasing
        /// the native mesh right away (D-392). Returns the stopwatch
        /// reading at the start, and logs meshing and writing apart.
        /// </summary>
        static long SaveStl(Voxels vox, string strPath,
                            System.Diagnostics.Stopwatch oWatch)
        {
            long lMs0 = oWatch.ElapsedMilliseconds;
            using (Mesh msh = new Mesh(vox))
            {
                long lMs1 = oWatch.ElapsedMilliseconds;
                msh.SaveToStlFile(strPath);
                Library.Log($"TIMING STL {Path.GetFileName(strPath)}: mesh "
                          + $"{(lMs1 - lMs0) / 1000f:0.0} s "
                          + $"({msh.nTriangleCount()} triangles), write "
                          + $"{(oWatch.ElapsedMilliseconds - lMs1) / 1000f:0.0} s");
            }
            return lMs0;
        }

        // -- part list (lazy: one part in memory at a time) ---------------

        /// <summary>Lazy part builders - the caller builds ONE part, exports
        /// it and releases the field. Peak memory stays at one part and the
        /// log gets per-part timings (Phy-AC's streaming recipe).</summary>
        public static List<(string strName, Func<Voxels> fnBuild)>
            aPartBuilders(CombustorParameters p, EffusionField oEff)
        {
            var aParts = new List<(string, Func<Voxels>)>();
            bool bMonoSwirler = p.Swirler.NVanes > 0
                             && p.Swirler.MonolithicWithDome;

            aParts.Add(("Liner", () =>
            {
                Voxels vox = Liner.voxLiner(p, oEff);
                if (bMonoSwirler)
                {
                    Library.Log("Liner: the swirler is MONOLITHIC with the "
                              + "dome (contract swirler.monolithic_with_dome)"
                              + " - unioned into this part.");
                    vox += Swirler.voxSwirler(p);
                }
                return vox;
            }));
            aParts.Add(("Casing", () => Casing.voxCasing(p)));
            if (p.Swirler.NVanes > 0 && !bMonoSwirler)
                aParts.Add(("Swirler", () => Swirler.voxSwirler(p)));

            int nInj = Injector.nPartCount(p);
            for (int i = 0; i < nInj; i++)
            {
                int n = i;                       // capture by value
                aParts.Add((Injector.strPartName(p, n),
                            () => Injector.voxPart(p, n)));
            }
            return aParts;
        }

        // -- report -------------------------------------------------------

        static void WriteReport(CombustorParameters p, string strDir,
                                List<Dictionary<string, object>> aParts,
                                EffusionField oEff, DfamVoxelReport oDfam,
                                Dictionary<string, object> oField)
        {
            var oRoot = new Dictionary<string, object>
            {
                { "schema", "phycb-dfam-voxel-1" },
                { "source", oDfam != null
                    ? "dfam_voxel(picogk)" : "dfam_voxel(failed)" },
                { "contract_schema", p.Schema },
                { "arch", p.Arch },
                { "name", p.OutputName },
                { "units", "mm / deg" },
                { "conventions",
                  "Z is the axis of revolution, the gas flows +Z (in "
                  + "reverse_annular the GAS turns 180 deg twice but the "
                  + "geometric axis is still Z); STL binary in mm; "
                  + "hole_rows[].angle_deg is the tilt of the hole axis from "
                  + "RADIAL towards +Z; effusion.angle_deg is measured from "
                  + "the liner SURFACE" },
                { "voxel_mm", p.VoxelSizeMm },
                { "feature_floor_mm", p.fFeatureFloorMm() },
                { "build_orientation", p.Dfam.BuildOrientation },
                { "parts", aParts },
                { "assembly_note", strAssemblyTag },
                { "voxel_field", oField },
                { "warnings", p.Warnings },
                { "notes", p.Notes },
            };

            if (oEff != null)
                oRoot["effusion"] = new Dictionary<string, object>
                {
                    { "n_holes", oEff.nHoleCount },
                    { "d_mm", p.Effusion.DMm },
                    { "shape", oEff.bIsTeardrop ? "teardrop"
                                                : p.Effusion.Shape },
                    { "angle_deg_from_surface", p.Effusion.AngleDeg },
                    { "pitch_x_over_d", p.Effusion.PitchXOverD },
                    { "pitch_y_over_d", p.Effusion.PitchYOverD },
                    { "porosity", oEff.fPorosity(p) },
                    { "method", "one implicit signed-distance field for the "
                              + "whole pattern, evaluated during "
                              + "voxelization - never one boolean per hole" },
                };

            if (oDfam != null)
            {
                oRoot["g_voxel"] = oDfam.afG;
                oRoot["g_voxel_names"] = DfamVoxelReport.astrGNames;
                oRoot["feasible"] = oDfam.bFeasible;
                oRoot["violations"] = oDfam.aViolations;
                oRoot["dfam_notes"] = oDfam.aNotes;
                oRoot["detail"] = oDfam.oDetail;
                oRoot["relation_to_python"] =
                    "COMPLEMENTARY to dfam_checker.check(): those 5 analytic "
                    + "constraints pre-filter the contract inside the "
                    + "optimizer loop; these 3 are measured on the built "
                    + "voxels. Neither replaces the other.";
            }
            else
            {
                oRoot["feasible"] = false;
                oRoot["violations"] = new List<string>
                    { "the voxel re-verification did not run" };
            }

            string strPath = Path.Combine(strDir, "dfam_report.json");
            var oOpt = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(strPath,
                JsonSerializer.Serialize(oRoot, oOpt));
            Library.Log($"Wrote DfAM report: {strPath}");
        }
    }
}
