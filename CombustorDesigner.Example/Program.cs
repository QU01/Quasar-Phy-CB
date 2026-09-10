//
// Program.cs - Phy-CB bridge: combustor.json -> printable combustor STLs.
//
//   Example.dll <path\to\combustor.json> [outDir] [voxelMm] [partFilter]
//
// Byte-compatible with the subprocess call the Python CLI makes (the exact
// pattern of Phy-AC/phyac_cli.py:348-390: `dotnet build` then
// `dotnet <Example>.dll <contract> <outdir> <voxel>`, with a fallback to
// C:\Program Files\dotnet\dotnet.exe).
//
//   voxelMm     0.8 draft, 0.5 iteration, 0.3-0.4 final. Cost ~ 1/voxel^3.
//   partFilter  case-insensitive substring: "Liner", "Casing", "Swirler",
//               "Vaporizer2" - builds only those parts and skips the
//               assembly, the DfAM re-verification and the field export.
//

using System;
using System.Globalization;
using System.IO;
using CombustorDesigner;

namespace CombustorDesigner.Example
{
    internal static class Program
    {
        public static int Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.Error.WriteLine(
                    "Usage: CombustorDesigner.Example.dll <combustor.json> "
                    + "[outDir] [voxelMm] [partFilter]");
                Console.Error.WriteLine(
                    "  voxelMm     0.8 draft, 0.5 iteration, 0.3-0.4 final "
                    + "(cost ~ 1/voxel^3)");
                Console.Error.WriteLine(
                    "  partFilter  substring, e.g. 'Liner', 'Casing', "
                    + "'Swirler', 'Vaporizer2' - builds only those parts");
                return 1;
            }

            string strJson = Path.GetFullPath(args[0]);
            if (!File.Exists(strJson))
            {
                Console.Error.WriteLine($"Does not exist: {strJson}");
                return 1;
            }
            string strOutDir = args.Length > 1
                ? Path.GetFullPath(args[1])
                : Path.Combine(Environment.CurrentDirectory, "output");

            CombustorParameters p;
            try
            {
                p = PhyCBImport.FromContractJson(strJson);
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(
                    $"Could not read the contract: {e.Message}");
                return 2;
            }

            if (args.Length > 2 && float.TryParse(args[2], NumberStyles.Float,
                    CultureInfo.InvariantCulture, out float fVoxel)
                && fVoxel > 0f)
                p.VoxelSizeMm = fVoxel;

            string strFilter = args.Length > 3 && args[3].Length > 0
                ? args[3] : null;

            string strErr = p.Validate();
            if (strErr != null)
            {
                Console.Error.WriteLine($"Invalid contract: {strErr}");
                return 3;
            }

            Console.WriteLine($"[PhyCB] {strJson}");
            Console.WriteLine(
                $"        arch '{p.Arch}', liner "
                + $"{p.Liner.Profile[0][0]:0.#}->"
                + $"{p.Liner.Profile[p.Liner.Profile.Length - 1][0]:0.#} mm "
                + $"(r_out max {p.fRMaxOuter():0.#} mm), wall "
                + $"{p.Liner.TWallMm:0.##} mm {p.Liner.Material}");
            Console.WriteLine(
                $"        {p.HoleRows.Count} hole rows, effusion d "
                + $"{p.Effusion.DMm:0.###} mm in {p.Effusion.Zones.Count} "
                + $"zone(s), {p.Injectors.Count} x {p.Injectors.Type}, "
                + $"swirler {p.Swirler.NVanes} vanes");
            Console.WriteLine(
                $"        voxel {p.VoxelSizeMm} mm, build "
                + $"'{p.Dfam.BuildOrientation}', "
                + $"{p.Dfam.DrainPorts.Count} drain port(s)"
                + (strFilter != null ? $", filter '{strFilter}'" : ""));
            foreach (string s in p.Warnings)
                Console.WriteLine("        WARN " + s);

            try
            {
                string strStl = CombustorBuilder.Build(
                    p, strOutDir, bShowViewer: false, strPartFilter: strFilter);
                Console.WriteLine($"        -> {strStl}");
                if (strFilter == null)
                    Console.WriteLine(
                        "        -> " + Path.Combine(strOutDir,
                                                     "dfam_report.json"));
                Console.WriteLine(
                    "        NOTE the assembly STL is "
                    + CombustorBuilder.strAssemblyTag + ".");
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(
                    $"Build failed: {e.GetType().Name}: {e.Message}");
                Console.Error.WriteLine(
                    "Check " + Path.Combine(strOutDir, "PicoGK.log"));
                return 4;
            }
            return 0;
        }
    }
}
