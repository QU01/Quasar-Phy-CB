//
// Parameters.cs - data-driven parameters for the combustor builder.
//
// Like Phy-AC's AxialCompressorParameters this record is DATA-DRIVEN: the
// Python layer 5a (geometry_generator.py) emits a fully resolved geometry
// contract (`combustor.json`, schema `phycb-combustor-1`) and this class
// simply carries it. The C# is deliberately "dumb": every bit of physics
// and every dimension has already been solved upstream.
//
// UNITS AND SIGNS (contract section 1.1 / IMPLEMENTATION_CONTRACT section 9)
//   * every length is in MILLIMETRES, every angle in DEGREES; the key
//     carries the suffix (_mm, _deg, _Pa, _K).
//   * Z is the axis of revolution and the gas flows towards +Z. In
//     `reverse_annular` the GAS makes two 180-degree turns but the
//     geometric axis is still Z - the contract profile already encodes it.
//   * liner.profile rows are [z_mm, r_inner_mm, r_outer_mm]; a two-column
//     row means r_inner = 0 (a can). casing.profile rows are [z_mm, r_mm]
//     with r the INNER radius of the casing.
//   * hole_rows[].angle_deg is the tilt of the hole AXIS away from the
//     RADIAL direction towards +Z (0 = purely radial, 90 = purely axial);
//     an explicit `axis` 3-vector [radial, tangential, axial] wins over it.
//     Same convention as dfam_checker.py - the two must never diverge.
//   * effusion.angle_deg is measured from the liner SURFACE (not from the
//     normal): 20-30 degrees on a near-axial wall keeps the hole
//     self-supporting. Again identical to dfam_checker._effusion_axis.
//
// PRINTABILITY: any wall/feature thinner than max(0.6 mm, 2*voxel) is
// CLAMPED to that floor and the clamp is LOGGED (never silent) - the house
// rule from CONTEXTO section 8.
//

using System;
using System.Collections.Generic;
using System.Globalization;

namespace CombustorDesigner
{
    /// <summary>One row of discrete holes (primary / secondary / dilution,
    /// igniter port, ...). The row is the dimensional truth shared with
    /// `hole_table.csv`; the STL is the printable approximation.</summary>
    public sealed class HoleRowParams
    {
        public string Zone = "pz";           // "pz" | "sz" | "dz" | free
        public string Id = null;             // optional contract id
        public float ZMm;
        public float RMm = 0f;               // 0 => taken from the liner
        public int N = 1;
        public float DMm = 1f;
        public float Cd = 0.62f;
        public string Shape = "circle";      // "circle" | "teardrop"
        public bool Plunged = false;
        public float AngleDeg = 0f;          // tilt from RADIAL towards +Z
        public float[] Axis = null;          // [radial, tangential, axial]
        public float Phi0Deg = 0f;           // phase of the first hole

        /// <summary>True when this row is the igniter/spark port, which the
        /// casing answers with a boss and a through bore.</summary>
        public bool bIsIgniter()
        {
            string s = ((Id ?? "") + "|" + (Zone ?? "")).ToLowerInvariant();
            return s.Contains("bujia") || s.Contains("buj")
                || s.Contains("spark") || s.Contains("ignit");
        }

        /// <summary>True when the row belongs to the effusion field, which
        /// never counts as a powder drain (EstadoDelArte 4.4).</summary>
        public bool bIsEffusion()
        {
            string s = ((Id ?? "") + "|" + (Zone ?? "")).ToLowerInvariant();
            return s.Contains("eff");
        }
    }

    /// <summary>One axial band of the effusion field.</summary>
    public sealed class EffusionZoneParams
    {
        public float Z0Mm;
        public float Z1Mm;
        public int Rows = 0;                 // 0 => derived from the pitch
    }

    /// <summary>Effusion cooling field - hundreds to thousands of holes,
    /// rendered as ONE implicit, never as one boolean per hole.</summary>
    public sealed class EffusionParams
    {
        public float DMm = 0f;               // 0 => no effusion field
        public float PitchXOverD = 6f;       // axial pitch / d
        public float PitchYOverD = 5f;       // circumferential pitch / d
        public float AngleDeg = 25f;         // from the liner SURFACE
        public string Shape = "cyl";         // "cyl" | "conical" | "teardrop"
        public float ConeExpansion = 1.6f;   // exit/inlet radius, conical
        public bool Staggered = true;        // alternate rows by half pitch
        public List<EffusionZoneParams> Zones = new List<EffusionZoneParams>();
    }

    /// <summary>Optional impingement/effusion double wall - the cavity that
    /// makes powder drainability the binding constraint.</summary>
    public sealed class DoubleWallParams
    {
        public float GapMm;
        public float Z0Mm;
        public float Z1Mm;
        public float TInnerMm;
        public float TOuterMm;
    }

    public sealed class LinerParams
    {
        public float[][] Profile;            // [[z, r_in, r_out], ...]
        public float TWallMm = 0.8f;
        public string Material = "Inconel-625";
        public string DomeType = "hemispherical";   // | "flat" | "none"
        public float DomeZMm = 0f;
        public bool bDomeZGiven = false;
        public DoubleWallParams DoubleWall = null;
        /// <summary>Root-fillet radius of the dome/barrel junction.
        /// SINGLE SOURCE OF TRUTH with the structural side: the same number
        /// the notch factor upstream credits (Phy-AC couples geometry and
        /// K_t through `blade_fillet_r_mm` exactly like this).</summary>
        public float FilletRMm = 0f;
    }

    public sealed class DrainPortParams
    {
        public float ZMm;
        public float DMm = 5f;
        public float PhiDeg = float.NaN;     // NaN => auto-spread
        public bool bBoss = true;
    }

    public sealed class CasingParams
    {
        public float[][] Profile;            // [[z, r_inner], ...]
        public float TWallMm = 1.2f;
        public string Material = "Inconel-718";
        public int FlangeBoltCount = 12;
        public float FlangeBoltDMm = 6f;
        public float FlangeWMm = 10f;        // radial width of the flange
        public float FlangeTMm = 4f;         // axial thickness
        public float HeadPlateTMm = 0f;      // 0 => open head end
        public float BossHMm = 4f;           // boss stand-off height
    }

    public sealed class SwirlerParams
    {
        public string Type = "axial";        // "axial" | "radial" | "none"
        public int NVanes = 0;               // 0 => no swirler part
        public float BetaDeg = 55f;          // vane angle at the tip
        public float BetaHubDeg = float.NaN; // NaN => constant beta
        public float SN = 0f;                // informative (swirl number)
        public float DHubMm = 10f;
        public float DTipMm = 24f;
        public float VaneTMm = 1f;
        public float ZMm = float.NaN;        // NaN => the dome plane
        public float ChordMm = 0f;           // 0 => derived from the span
        public bool MonolithicWithDome = false;
    }

    public sealed class InjectorParams
    {
        public string Type = "airblast";     // airblast|simplex|vaporizer|micromix
        public int Count = 1;
        public float ALR = 4f;
        public float RingRMm = 0f;           // 0 => on the axis when Count==1
        public float Phi0Deg = 0f;
        public float BodyDMm = 0f;           // 0 => derived from the swirler
        public float BoreDMm = 0f;
        public float PrefilmerTMm = 0f;
        // vaporizer ("walking stick")
        public float TubeDMm = 0f;           // outer diameter
        public float TubeTMm = 0f;           // wall thickness
        public float TubeZ0Mm = float.NaN;   // inlet plane
        public float TubeZ1Mm = float.NaN;   // elbow plane
        public float BendDeg = 180f;
        public float BendRMm = 0f;           // 0 => 1.5 * TubeDMm
        // micromix
        public int MicromixN = 0;
        public float MicromixDMm = 0f;
    }

    public sealed class DfamParams
    {
        public string BuildOrientation = "z+";
        public float MinWallMm = 0f;         // 0 => material table
        public float VoxelMm = 0.5f;
        public List<DrainPortParams> DrainPorts = new List<DrainPortParams>();
        /// <summary>Cell size of the coarse grid used for the voxel-level
        /// re-verification. WAVE3 section 6: the flood fill is done COARSE
        /// (1 mm) on the same field - the topology does not change and it is
        /// far cheaper. It is also physically meaningful: a channel finer
        /// than the cell does not drain powder, which is exactly why the
        /// sub-millimetre effusion holes must NOT count as a drain.</summary>
        public float CoarseVoxelMm = 1.0f;
        /// <summary>Tolerated fraction of unsupported solid cells before the
        /// overhang constraint is reported as violated.</summary>
        public float OverhangFracMax = 0.005f;
    }

    /// <summary>Full combustor parameters consumed by CombustorBuilder.</summary>
    public sealed class CombustorParameters
    {
        // -- printability floors (EstadoDelArte 4.3) -----------------------
        public const float FEATURE_MIN_MM = 0.6f;   // absolute LPBF floor
        public const float VOXEL_FACTOR = 2.0f;     // voxels to RESOLVE it
        public const float D_HOLE_MIN_MM = 0.5f;    // min printable hole
        public const float TEARDROP_ANGLE_DEG = 45f;

        /// <summary>Material-dependent minimum wall. Mirror of
        /// dfam_checker.LPBF_T_MIN_MM / structures_core.MATERIALS - if you
        /// touch one, touch the other.</summary>
        static readonly Dictionary<string, float> s_oTMin =
            new Dictionary<string, float>
            {
                { "inconel-625", 0.5f }, { "hastelloy-x", 0.8f },
                { "haynes-230", 0.6f },  { "haynes-282", 0.6f },
                { "inconel-718", 0.5f }, { "inconel-939", 0.7f },
                { "ti-6al-4v", 0.4f },   { "17-4ph-h900", 0.5f },
                { "al-2618-t61", 1.0f }, { "al-7075-t6", 1.0f },
                { "316l", 0.4f },        { "grcop-42", 0.5f },
            };
        public const float T_MIN_DEFAULT_MM = 0.6f;

        public string OutputName = "PhyCB";
        public string Schema = "";
        public string Arch = "can";          // can | annular | reverse_annular
        public float VoxelSizeMm = 0.5f;

        public LinerParams Liner = new LinerParams();
        public CasingParams Casing = new CasingParams();
        public SwirlerParams Swirler = new SwirlerParams();
        public InjectorParams Injectors = new InjectorParams();
        public EffusionParams Effusion = new EffusionParams();
        public DfamParams Dfam = new DfamParams();
        public List<HoleRowParams> HoleRows = new List<HoleRowParams>();

        /// <summary>Non-fatal findings: unknown materials, thickness clamps,
        /// keys missing from an older contract. Logged by the builder - the
        /// house rule is that a degradation is never silent.</summary>
        public List<string> Warnings = new List<string>();
        /// <summary>Notes from the defensive JSON reader (keys defaulted).</summary>
        public List<string> Notes = new List<string>();

        // -- printability helpers -----------------------------------------

        /// <summary>The printable floor for ANY feature: max(0.6 mm,
        /// 2*voxel). CONTEXTO section 8.</summary>
        public float fFeatureFloorMm()
            => MathF.Max(FEATURE_MIN_MM, VOXEL_FACTOR * VoxelSizeMm);

        /// <summary>Minimum LPBF wall of a material (canonical name, alias
        /// tolerant). Unknown material -> the generic floor + a warning.</summary>
        public float fMaterialTMinMm(string strMaterial)
        {
            string strKey = strNormMaterial(strMaterial);
            if (s_oTMin.TryGetValue(strKey, out float f))
                return f;
            Warnings.Add(
                $"unknown material '{strMaterial}' - using the generic LPBF " +
                $"floor {T_MIN_DEFAULT_MM:0.##} mm. Known materials: " +
                string.Join(", ", s_oTMin.Keys));
            return T_MIN_DEFAULT_MM;
        }

        /// <summary>Clamp a requested thickness to the printable floor,
        /// LOGGING the clamp (never silent - CONTEXTO section 8). Returns
        /// the value actually built, which may be THICKER than the verified
        /// design; the caller is told to use a finer voxel.</summary>
        public float fPrintableMm(float fRequestedMm, string strWhat)
        {
            float fFloor = fFeatureFloorMm();
            if (fRequestedMm >= fFloor)
                return fRequestedMm;
            Warnings.Add(
                $"WARNING {strWhat}: thickness clamped " +
                $"{fRequestedMm.ToString("0.###", CultureInfo.InvariantCulture)}" +
                $" -> {fFloor.ToString("0.###", CultureInfo.InvariantCulture)} mm " +
                $"(2 voxels at {VoxelSizeMm.ToString("0.###", CultureInfo.InvariantCulture)} mm). " +
                "The PRINTED part is thicker than the verified design - " +
                "use a finer voxel for a faithful part.");
            return fFloor;
        }

        /// <summary>The same clamp WITHOUT the warning. Use it when the
        /// value has already been reported once for that feature (a wall is
        /// consumed by half a dozen sub-builders; warning once is the point,
        /// warning six times is noise).</summary>
        public float fPrintableQuietMm(float fRequestedMm)
            => MathF.Max(fRequestedMm, fFeatureFloorMm());

        /// <summary>Canonical material key: lowercase, dashes, common
        /// aliases. Mirror of dfam_checker._norm_material.</summary>
        public static string strNormMaterial(string strName)
        {
            string s = (strName ?? "").Trim().ToLowerInvariant()
                                     .Replace('_', '-').Replace(' ', '-');
            while (s.Contains("--"))
                s = s.Replace("--", "-");
            switch (s)
            {
                case "in625": case "in-625": case "inconel625": case "ni625":
                    return "inconel-625";
                case "in718": case "in-718": case "inconel718":
                    return "inconel-718";
                case "hx": case "hastelloyx": case "hastelloy":
                    return "hastelloy-x";
                case "haynes230": return "haynes-230";
                case "haynes282": return "haynes-282";
                case "ti64": case "tial6v4": return "ti-6al-4v";
                case "ss316l": case "316-l": return "316l";
                default: return s;
            }
        }

        /// <summary>+1 when the part grows towards +Z on the plate, -1 for
        /// "z-". Everything DfAM depends on this sign.</summary>
        public float fBuildSign()
        {
            string s = (Dfam.BuildOrientation ?? "z+").Trim().ToLowerInvariant()
                                                     .Replace("-", "-");
            if (s.Contains("-")) return -1f;
            return 1f;
        }

        // -- contract geometry helpers ------------------------------------

        /// <summary>Outer radius of the liner at an axial station.</summary>
        public float fLinerROuterAt(float fZ)
            => LatticeUtils.fInterpAt(Liner.Profile, fZ, 2);

        /// <summary>Inner radius of the liner annulus (0 for a can).</summary>
        public float fLinerRInnerAt(float fZ)
            => LatticeUtils.fInterpAt(Liner.Profile, fZ, 1);

        /// <summary>Inner radius of the casing at an axial station.</summary>
        public float fCasingRAt(float fZ)
            => LatticeUtils.fInterpAt(Casing.Profile, fZ, 1);

        public float fZMin()
        {
            float f = Liner.Profile[0][0];
            if (Casing.Profile != null && Casing.Profile.Length > 0)
                f = MathF.Min(f, Casing.Profile[0][0]);
            return f;
        }

        public float fZMax()
        {
            float f = Liner.Profile[Liner.Profile.Length - 1][0];
            if (Casing.Profile != null && Casing.Profile.Length > 0)
                f = MathF.Max(f,
                    Casing.Profile[Casing.Profile.Length - 1][0]);
            return f;
        }

        public float fRMaxOuter()
        {
            float f = 0f;
            foreach (float[] afRow in Liner.Profile)
                f = MathF.Max(f, afRow[2]);
            if (Casing.Profile != null)
                foreach (float[] afRow in Casing.Profile)
                    f = MathF.Max(f, afRow[1] + Casing.TWallMm);
            return f;
        }

        /// <summary>Dome plane: the contract value when given, otherwise the
        /// upstream end of the liner.</summary>
        public float fDomeZMm()
            => Liner.bDomeZGiven ? Liner.DomeZMm : Liner.Profile[0][0];

        // -- validation ---------------------------------------------------

        /// <summary>
        /// Returns an error message, or null when the contract can be built.
        /// Checks the schema, the presence of unit-suffixed values, the
        /// consistency of the polyline point counts, and known materials.
        /// Soft problems (unknown material, missing optional block) go to
        /// <see cref="Warnings"/> instead of failing the build - an old
        /// contract must still build (CONTEXTO section 10.3).
        /// </summary>
        public string Validate()
        {
            if (string.IsNullOrEmpty(Schema))
                Warnings.Add("the contract carries no `schema` key - assuming "
                           + "phycb-combustor-1; add it (CONTEXTO 10.3).");
            else if (!Schema.StartsWith("phycb-combustor"))
                return $"schema '{Schema}' is not a phycb-combustor-* " +
                       "geometry contract (expected 'phycb-combustor-1')";

            if (Liner.Profile == null || Liner.Profile.Length < 2)
                return "liner.profile needs at least 2 points [z_mm, "
                     + "r_inner_mm, r_outer_mm]";
            int nCols = Liner.Profile[0].Length;
            for (int i = 0; i < Liner.Profile.Length; i++)
            {
                if (Liner.Profile[i] == null || Liner.Profile[i].Length < 3)
                    return $"liner.profile[{i}] must have 3 columns "
                         + "[z_mm, r_inner_mm, r_outer_mm] (2-column rows are "
                         + "accepted on import and padded with r_inner = 0)";
                if (Liner.Profile[i].Length != nCols)
                    return "liner.profile: inconsistent number of columns "
                         + $"(row 0 has {nCols}, row {i} has "
                         + $"{Liner.Profile[i].Length})";
                if (Liner.Profile[i][2] <= 0f)
                    return $"liner.profile[{i}]: r_outer_mm must be > 0";
                if (Liner.Profile[i][1] < 0f
                    || Liner.Profile[i][1] >= Liner.Profile[i][2])
                    return $"liner.profile[{i}]: needs 0 <= r_inner_mm < "
                         + "r_outer_mm";
                if (i > 0 && Liner.Profile[i][0] < Liner.Profile[i - 1][0])
                    return "liner.profile must be sorted by z_mm "
                         + $"(row {i} goes backwards)";
            }
            if (MathF.Abs(Liner.Profile[Liner.Profile.Length - 1][0]
                          - Liner.Profile[0][0]) < 1e-3f)
                return "liner.profile has zero axial length";

            if (Casing.Profile == null || Casing.Profile.Length < 2)
                return "casing.profile needs at least 2 points [z_mm, r_mm]";
            for (int i = 0; i < Casing.Profile.Length; i++)
            {
                if (Casing.Profile[i] == null || Casing.Profile[i].Length < 2)
                    return $"casing.profile[{i}] must have 2 columns "
                         + "[z_mm, r_mm]";
                if (Casing.Profile[i][1] <= 0f)
                    return $"casing.profile[{i}]: r_mm must be > 0";
                if (i > 0 && Casing.Profile[i][0] < Casing.Profile[i - 1][0])
                    return "casing.profile must be sorted by z_mm "
                         + $"(row {i} goes backwards)";
            }

            if (Liner.TWallMm <= 0f)
                return "liner.t_wall_mm must be > 0 (units are millimetres)";
            if (Casing.TWallMm <= 0f)
                return "casing.t_wall_mm must be > 0 (units are millimetres)";
            if (VoxelSizeMm <= 0.05f)
                return $"voxel {VoxelSizeMm} mm is too small (>0.05 mm); the "
                     + "cost scales as 1/voxel^3 - use 0.8 draft, 0.5 "
                     + "iteration, 0.3-0.4 final";

            // the casing has to actually contain the liner
            float fZ0 = MathF.Max(Liner.Profile[0][0], Casing.Profile[0][0]);
            float fZ1 = MathF.Min(Liner.Profile[Liner.Profile.Length - 1][0],
                                  Casing.Profile[Casing.Profile.Length - 1][0]);
            if (fZ1 <= fZ0)
                return "the liner and casing axial ranges do not overlap - "
                     + "check the z_mm datum of both profiles";
            for (int k = 0; k <= 8; k++)
            {
                float fZ = fZ0 + (fZ1 - fZ0) * k / 8f;
                if (fCasingRAt(fZ) <= fLinerROuterAt(fZ))
                    return $"the casing inner radius ({fCasingRAt(fZ):0.##} mm)"
                         + $" is not outside the liner ({fLinerROuterAt(fZ):0.##}"
                         + $" mm) at z = {fZ:0.#} mm";
            }

            // holes: the contract is the dimensional truth, so it must be sane
            for (int i = 0; i < HoleRows.Count; i++)
            {
                HoleRowParams row = HoleRows[i];
                if (row.N < 1)
                    return $"hole_rows[{i}]: n must be >= 1";
                if (row.DMm <= 0f)
                    return $"hole_rows[{i}]: d_mm must be > 0";
                if (row.DMm < D_HOLE_MIN_MM)
                    Warnings.Add(
                        $"hole_rows[{i}] ({row.Zone}) d = {row.DMm:0.###} mm "
                        + $"< the LPBF floor {D_HOLE_MIN_MM:0.##} mm - it will "
                        + "be built but it is not printable (EstadoDelArte 4.3)");
            }

            if (Effusion.DMm > 0f)
            {
                if (Effusion.Zones.Count == 0)
                    Warnings.Add("effusion.d_mm is given but effusion.zones is "
                               + "empty - no effusion field will be built");
                if (Effusion.DMm < D_HOLE_MIN_MM)
                    Warnings.Add(
                        $"effusion d = {Effusion.DMm:0.###} mm < the LPBF floor "
                        + $"{D_HOLE_MIN_MM:0.##} mm (0.5-0.8 mm is the realistic "
                        + "as-built band, EstadoDelArte 4.3)");
                foreach (EffusionZoneParams z in Effusion.Zones)
                    if (z.Z1Mm <= z.Z0Mm)
                        return "effusion.zones: every zone needs z1_mm > z0_mm";
            }

            // materials: soft check with a useful message
            fMaterialTMinMm(Liner.Material);
            fMaterialTMinMm(Casing.Material);
            if (strNormMaterial(Liner.Material) == "inconel-718")
                Warnings.Add("liner material is Inconel-718, which is BANNED on "
                           + "a hot wall (gamma'' over-ages above ~650 C, "
                           + "EstadoDelArte 4.1). The part will still be built.");
            float fTMinLiner = fMaterialTMinMm(Liner.Material);
            if (Liner.TWallMm < fTMinLiner)
                Warnings.Add(
                    $"liner t_wall = {Liner.TWallMm:0.###} mm is below the LPBF "
                    + $"minimum of {Liner.Material} ({fTMinLiner:0.##} mm) - "
                    + "porosity and cracking grow non-linearly below it "
                    + "(EstadoDelArte 4.2)");

            if (!(Dfam.CoarseVoxelMm > 0f))
                return "dfam coarse voxel must be > 0";
            return null;
        }
    }
}
