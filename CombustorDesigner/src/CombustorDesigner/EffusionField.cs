//
// EffusionField.cs - the effusion cooling pattern as ONE implicit field.
//
// WHY THIS FILE EXISTS (WAVE3 section 2, "detalles criticos"): an effusion
// liner carries hundreds to thousands of holes. Subtracting one boolean per
// hole is the wrong operation for a narrow-band SDF kernel - it is O(N)
// expensive voxel operations and it makes PicoGK pointless. The natural
// operation is to describe the WHOLE pattern as a signed-distance function
// and let the voxelizer evaluate it once:
//
//   sd(p) = min over the holes near p of the signed distance to that hole
//
// Cost is O(1) per sample, not O(N): the holes sit on a regular (z, phi)
// lattice, so for any sample point we can jump straight to the 3x3
// neighbourhood of candidate holes and take the minimum. Building 900 holes
// then costs exactly what building 9 would.
//
// HOLE AXIS. `effusion.angle_deg` is measured FROM THE LINER SURFACE, not
// from the normal - identical to dfam_checker._effusion_axis, and the two
// must never diverge:
//
//   t = unit meridional tangent of the liner outer profile at the row
//   n = outward normal = rot90(t)
//   a = cos(angle)*t + sin(angle)*n          (points outwards + downstream)
//
// With a near-axial wall and 20-30 degrees the hole axis stays more than 45
// degrees away from the build horizontal, so it is self-supporting and the
// round section is legal (EstadoDelArte 4.3 rule 2 - this is exactly why
// the effusion angle and the build orientation are coupled). When it is
// NOT, the section switches to a teardrop and the switch is logged.
//
// SECTION SHAPES
//   "cyl"      constant radius            - the metering baseline
//   "conical"  radius grows towards the GAS side (the air enters from the
//              annulus through the metering inlet on the outer surface and
//              diffuses into the combustor) - "radio funcion de la
//              profundidad" in WAVE3
//   "teardrop" convex hull of the circle and a 45-degree apex along the
//              build direction - "gota = union cilindro + cuna"
//

using System;
using System.Collections.Generic;
using System.Numerics;
using PicoGK;

namespace CombustorDesigner
{
    public sealed class EffusionField : IImplicit
    {
        /// <summary>One precomputed row of holes on the liner.</summary>
        struct Row
        {
            public float fZc;        // axial station of the row centre
            public float fRMid;      // mid-wall radius of the hole centres
            public int nHoles;       // holes around the circumference
            public float fDPhi;      // angular pitch
            public float fPhi0;      // phase (staggered rows)
            public float fAr;        // axis, radial component
            public float fAz;        // axis, axial component
            public float fHalfLen;   // half length of the drilled segment
        }

        readonly Row[] m_aRows;
        readonly float[] m_afRowZ;       // sorted keys for the binary search
        readonly float m_fR;             // nominal hole radius
        readonly float m_fReach;         // |dz| beyond which a row cannot hit
        readonly bool m_bConical;
        readonly bool m_bTeardrop;
        readonly float m_fConeExp;
        readonly Vector3 m_vecUp;        // build-up direction
        readonly BBox3 m_oBounds;
        readonly int m_nHoles;

        // D-392: per-row tables of the hole centre and axis for every hole
        // index j the 3x3 neighbourhood search can reach, built with the
        // very expressions fSignedDistance used to evaluate per sample, so
        // the field is bit-for-bit the same and ~3x cheaper to evaluate.
        readonly int[] m_anJMin;
        readonly Vector3[][] m_aavecC;
        readonly Vector3[][] m_aavecA;

        /// <summary>Total number of holes in the field (reported to the log
        /// and to dfam_report.json - with ~2000 holes "open holes" becomes a
        /// quality requirement, EstadoDelArte 4.4).</summary>
        public int nHoleCount => m_nHoles;

        /// <summary>Bounding box the field has to be rendered in: the
        /// annular shell that contains every hole, nothing more.</summary>
        public BBox3 oBounds => m_oBounds;

        public bool bIsTeardrop => m_bTeardrop;

        /// <summary>True when the contract asks for effusion and at least
        /// one hole was actually generated.</summary>
        public bool bIsEmpty => m_aRows.Length == 0;

        public EffusionField(CombustorParameters p, IList<string> aLog)
        {
            EffusionParams oEff = p.Effusion;
            m_fR = 0.5f * oEff.DMm;
            m_bConical = (oEff.Shape ?? "").ToLowerInvariant()
                                           .StartsWith("con");
            m_fConeExp = MathF.Max(1f, oEff.ConeExpansion);
            m_vecUp = new Vector3(0f, 0f, p.fBuildSign());

            float fT = p.Liner.TWallMm;
            float fAngle = oEff.AngleDeg * MathF.PI / 180f;

            var aRows = new List<Row>();
            float fZLo = float.MaxValue, fZHi = float.MinValue;
            float fRLo = float.MaxValue, fRHi = float.MinValue;
            int nHoles = 0;
            bool bTeardrop = (oEff.Shape ?? "").ToLowerInvariant()
                                               .StartsWith("tear");
            float fWorstDeg = 90f;

            float fPitchZNominal = MathF.Max(oEff.PitchXOverD * oEff.DMm,
                                             2f * oEff.DMm);
            float fPitchPhi = MathF.Max(oEff.PitchYOverD * oEff.DMm,
                                        2f * oEff.DMm);

            foreach (EffusionZoneParams oZone in oEff.Zones)
            {
                float fZ0 = MathF.Min(oZone.Z0Mm, oZone.Z1Mm);
                float fZ1 = MathF.Max(oZone.Z0Mm, oZone.Z1Mm);
                float fSpan = fZ1 - fZ0;
                if (fSpan <= 0f || m_fR <= 0f)
                    continue;

                int nRows = oZone.Rows;
                if (nRows <= 0)
                    nRows = Math.Max(1, (int)MathF.Round(fSpan
                                                        / fPitchZNominal));
                else
                {
                    int nFromPitch = Math.Max(1,
                        (int)MathF.Round(fSpan / fPitchZNominal));
                    if (Math.Abs(nFromPitch - nRows) > 1)
                        aLog?.Add(
                            $"effusion zone z {fZ0:0.#}-{fZ1:0.#} mm declares "
                            + $"{nRows} rows but pitch_x/d = "
                            + $"{oEff.PitchXOverD:0.##} implies {nFromPitch} - "
                            + "the DECLARED row count wins (the contract is "
                            + "the resolved truth); check hole_table.csv.");
                }
                float fPitchZ = fSpan / nRows;

                for (int i = 0; i < nRows; i++)
                {
                    float fZc = fZ0 + (i + 0.5f) * fPitchZ;
                    float fROut = p.fLinerROuterAt(fZc);
                    if (fROut <= 0f)
                        continue;
                    float fRMid = MathF.Max(fROut - 0.5f * fT, 0.5f * fT);

                    // local wall tangent / outward normal in (z, r)
                    Vector2 vecT = LatticeUtils.vecTangentAt(
                        p.Liner.Profile, fZc, 2);
                    float fTz = vecT.X, fTr = vecT.Y;
                    float fNz = -fTr, fNr = fTz;
                    float fAz = MathF.Cos(fAngle) * fTz
                              + MathF.Sin(fAngle) * fNz;
                    float fAr = MathF.Cos(fAngle) * fTr
                              + MathF.Sin(fAngle) * fNr;
                    float fNorm = MathF.Sqrt(fAz * fAz + fAr * fAr);
                    if (fNorm < 1e-9f) { fAz = 1f; fAr = 0f; fNorm = 1f; }
                    fAz /= fNorm; fAr /= fNorm;

                    // self-support test: angle between the hole axis and the
                    // build plane (the plane normal to the build direction).
                    // The axis has no tangential component, so the angle from
                    // the horizontal is asin(|az|).
                    float fDeg = MathF.Asin(MathF.Min(1f, MathF.Abs(fAz)))
                               * 180f / MathF.PI;
                    fWorstDeg = MathF.Min(fWorstDeg, fDeg);

                    // incidence with the wall drives how deep we must drill
                    float fSinInc = MathF.Max(
                        MathF.Abs(fAz * fNz + fAr * fNr), 0.2f);
                    float fHalf = 0.75f * fT / fSinInc + 1.0f;

                    int nAround = Math.Max(3,
                        (int)MathF.Round(2f * MathF.PI * fRMid / fPitchPhi));
                    float fDPhi = 2f * MathF.PI / nAround;

                    aRows.Add(new Row
                    {
                        fZc = fZc,
                        fRMid = fRMid,
                        nHoles = nAround,
                        fDPhi = fDPhi,
                        fPhi0 = (oEff.Staggered && (i % 2) == 1)
                                ? 0.5f * fDPhi : 0f,
                        fAr = fAr,
                        fAz = fAz,
                        fHalfLen = fHalf,
                    });
                    nHoles += nAround;
                    fZLo = MathF.Min(fZLo, fZc - fHalf - m_fR);
                    fZHi = MathF.Max(fZHi, fZc + fHalf + m_fR);
                    float fRSpan = fHalf + m_fR * m_fConeExp;
                    fRLo = MathF.Min(fRLo, fRMid - fRSpan);
                    fRHi = MathF.Max(fRHi, fRMid + fRSpan);
                }
            }

            // Auto teardrop (EstadoDelArte 4.3 rule 1): if the hole axis ends
            // up closer than 45 degrees to the build horizontal it needs a
            // self-supporting section. Never silent.
            if (!bTeardrop && aRows.Count > 0
                && fWorstDeg < CombustorParameters.TEARDROP_ANGLE_DEG)
            {
                bTeardrop = true;
                aLog?.Add(
                    $"effusion holes sit at {fWorstDeg:0.#} deg from the build "
                    + $"horizontal (< {CombustorParameters.TEARDROP_ANGLE_DEG:0} "
                    + "deg) - the section was switched to TEARDROP "
                    + "automatically (EstadoDelArte 4.3 rule 1).");
            }
            m_bTeardrop = bTeardrop;

            m_aRows = aRows.ToArray();
            m_afRowZ = new float[m_aRows.Length];
            for (int i = 0; i < m_aRows.Length; i++)
                m_afRowZ[i] = m_aRows[i].fZc;
            Array.Sort(m_afRowZ, m_aRows);
            m_nHoles = nHoles;

            m_anJMin = new int[m_aRows.Length];
            m_aavecC = new Vector3[m_aRows.Length][];
            m_aavecA = new Vector3[m_aRows.Length][];
            for (int i = 0; i < m_aRows.Length; i++)
            {
                Row row = m_aRows[i];
                int jMin = (int)MathF.Floor((-MathF.PI - row.fPhi0)
                                            / row.fDPhi) - 2;
                int jMax = (int)MathF.Ceiling((MathF.PI - row.fPhi0)
                                              / row.fDPhi) + 2;
                m_anJMin[i] = jMin;
                m_aavecC[i] = new Vector3[jMax - jMin + 1];
                m_aavecA[i] = new Vector3[jMax - jMin + 1];
                for (int j = jMin; j <= jMax; j++)
                {
                    HoleFrame(row, j, out m_aavecC[i][j - jMin],
                              out m_aavecA[i][j - jMin]);
                }
            }

            float fMaxHalf = 0f;
            foreach (Row r in m_aRows)
                fMaxHalf = MathF.Max(fMaxHalf, r.fHalfLen);
            m_fReach = fMaxHalf + m_fR * m_fConeExp * MathF.Sqrt(2f) + 1e-3f;

            if (m_aRows.Length == 0)
            {
                m_oBounds = LatticeUtils.oBBox(Vector3.Zero, Vector3.Zero);
                return;
            }
            float fRad = MathF.Max(MathF.Abs(fRHi), 1f) + 1f;
            m_oBounds = LatticeUtils.oBBox(
                new Vector3(-fRad, -fRad, fZLo - 1f),
                new Vector3(fRad, fRad, fZHi + 1f));
        }

        /// <summary>
        /// The whole field as ONE voxel field, ready to subtract.
        ///
        /// D-392 (17.1-03): the SAME implicit, evaluated only where it can
        /// matter. Rendering it over m_oBounds evaluated fSignedDistance on
        /// every voxel of the +-r_max square of the liner (118 s for the
        /// 39 064 holes of the 8 kg/s liner at 0.8 mm), although the holes
        /// live in a thin annular shell around the wall. Here the shell is
        /// covered by axis-aligned tiles: one z-slab at a time, the (z, r)
        /// extent of every row that reaches the slab is padded by
        /// fPadVoxels voxels (wider than the narrow band) and only the xy
        /// tiles that touch that annulus are rendered, with half a voxel of
        /// overlap between neighbours. Every voxel where the old render
        /// stored a value inside the band is evaluated with the identical
        /// function, so the subtracted field is the same; outside the tiles
        /// the old render had only background (tiles are also clipped to
        /// m_oBounds, the box the old render used). Not a lattice: PicoGK
        /// rasterizes sub-voxel beams differently from the implicit (the
        /// KJ66 liner lost 3.5 % more volume that way), so it is not
        /// equivalent.
        /// </summary>
        public Voxels voxRender(float fVoxelMm, out string strHow)
        {
            if (m_aRows.Length == 0)
            {
                strHow = "empty field";
                return new Voxels();
            }
            const float fPadVoxels = 4f;
            // thin in xy (the shell is ~10 mm thick radially), tall in z
            // (rows follow each other every few mm along the wall)
            const int nTileVoxels = 10;
            const int nTileZVoxels = 80;
            float fPad = fPadVoxels * fVoxelMm;
            float fTile = nTileVoxels * fVoxelMm;
            float fTileZ = nTileZVoxels * fVoxelMm;
            float fLap = 0.5f * fVoxelMm;

            // largest perpendicular reach of a section: the conical gas
            // side, and the 45-degree apex of a teardrop
            float fRSec = m_fR * (m_bConical ? m_fConeExp : 1f);
            if (m_bTeardrop)
                fRSec *= MathF.Sqrt(2f);

            int n = m_aRows.Length;
            var afZLo = new float[n];
            var afZHi = new float[n];
            var afRLo = new float[n];
            var afRHi = new float[n];
            float fZMin = float.MaxValue, fZMax = float.MinValue;
            float fRAll = 0f;
            for (int i = 0; i < n; i++)
            {
                Row row = m_aRows[i];
                float fDz = row.fHalfLen * MathF.Abs(row.fAz) + fRSec + fPad;
                float fDr = row.fHalfLen * MathF.Abs(row.fAr) + fRSec + fPad
                          + fRSec * fRSec / MathF.Max(row.fRMid, 1f);
                afZLo[i] = row.fZc - fDz;
                afZHi[i] = row.fZc + fDz;
                afRLo[i] = MathF.Max(row.fRMid - fDr, 0f);
                afRHi[i] = row.fRMid + fDr;
                fZMin = MathF.Min(fZMin, afZLo[i]);
                fZMax = MathF.Max(fZMax, afZHi[i]);
                fRAll = MathF.Max(fRAll, afRHi[i]);
            }

            int nXY = (int)MathF.Ceiling(2f * fRAll / fTile);
            var aBoxes = new List<BBox3>();
            double dEvalMm3 = 0.0;
            for (float fZ0 = fZMin; fZ0 < fZMax; fZ0 += fTileZ)
            {
                float fZ1 = fZ0 + fTileZ;
                float fRLo = float.MaxValue, fRHi = -1f;
                for (int i = 0; i < n; i++)
                {
                    if (afZHi[i] < fZ0 || afZLo[i] > fZ1)
                        continue;
                    fRLo = MathF.Min(fRLo, afRLo[i]);
                    fRHi = MathF.Max(fRHi, afRHi[i]);
                }
                if (fRHi < 0f)
                    continue;

                for (int ix = 0; ix < nXY; ix++)
                {
                    float fX0 = -fRAll + ix * fTile, fX1 = fX0 + fTile;
                    for (int iy = 0; iy < nXY; iy++)
                    {
                        float fY0 = -fRAll + iy * fTile, fY1 = fY0 + fTile;
                        // nearest / farthest distance of the tile to the axis
                        float fNx = MathF.Max(0f, MathF.Max(fX0, -fX1));
                        float fNy = MathF.Max(0f, MathF.Max(fY0, -fY1));
                        float fFx = MathF.Max(MathF.Abs(fX0), MathF.Abs(fX1));
                        float fFy = MathF.Max(MathF.Abs(fY0), MathF.Abs(fY1));
                        float fNear = MathF.Sqrt(fNx * fNx + fNy * fNy);
                        float fFar = MathF.Sqrt(fFx * fFx + fFy * fFy);
                        if (fNear > fRHi || fFar < fRLo)
                            continue;
                        // never outside the box the old render used
                        Vector3 vecLo = Vector3.Max(m_oBounds.vecMin,
                            new Vector3(fX0 - fLap, fY0 - fLap, fZ0 - fLap));
                        Vector3 vecHi = Vector3.Min(m_oBounds.vecMax,
                            new Vector3(fX1 + fLap, fY1 + fLap, fZ1 + fLap));
                        if (vecHi.X <= vecLo.X || vecHi.Y <= vecLo.Y
                            || vecHi.Z <= vecLo.Z)
                            continue;
                        aBoxes.Add(LatticeUtils.oBBox(vecLo, vecHi));
                        Vector3 vecE = vecHi - vecLo;
                        dEvalMm3 += (double)vecE.X * vecE.Y * vecE.Z;
                    }
                }
            }

            double dBox = (double)(m_oBounds.vecMax.X - m_oBounds.vecMin.X)
                        * (m_oBounds.vecMax.Y - m_oBounds.vecMin.Y)
                        * (m_oBounds.vecMax.Z - m_oBounds.vecMin.Z);
            if (dEvalMm3 >= 0.5 * dBox)
            {
                // a small liner: the shell IS most of the box
                strHow = $"implicit over the full box ({dBox / 1e6:0.0} dm3; "
                       + "the shell tiles would cover "
                       + $"{100.0 * dEvalMm3 / dBox:0} % of it)";
                return new Voxels(this, m_oBounds);
            }

            // every tile is rendered into the SAME field: the overlaps
            // receive the identical value twice and nothing is unioned (a
            // field per tile plus unions gave the same triangles, slower)
            Voxels vox = new Voxels();
            foreach (BBox3 oBox in aBoxes)
                vox.RenderImplicit(this, oBox);
            string strJoin = "rendered into one field";
            strHow = $"implicit over {aBoxes.Count} shell tiles of "
                   + $"{fTile:0.##} x {fTile:0.##} x {fTileZ:0.##} mm, {strJoin} ({dEvalMm3 / 1e6:0.0} of "
                   + $"{dBox / 1e6:0.0} dm3 of the full box evaluated)";
            return vox;
        }

        /// <summary>
        /// Signed distance to the nearest effusion hole. Negative inside a
        /// hole. The result is the SDF of a capped (possibly tapered,
        /// possibly teardrop) cylinder - exact on the axis-normal section
        /// and a mild under-estimate near the rim corners, which is
        /// irrelevant for a narrow-band kernel.
        /// </summary>
        public float fSignedDistance(in Vector3 vec)
        {
            if (m_aRows.Length == 0)
                return 1e6f;

            float fZ = vec.Z;
            // rows are sorted by z: jump to the neighbourhood
            int i0 = Array.BinarySearch(m_afRowZ, fZ);
            if (i0 < 0) i0 = ~i0;

            float fBest = 1e6f;
            bool bPhi = false;
            float fPhi = 0f;
            for (int i = i0 - 1; i <= i0 + 1; i++)
            {
                if (i < 0 || i >= m_aRows.Length)
                    continue;
                Row row = m_aRows[i];
                if (MathF.Abs(fZ - row.fZc) > m_fReach + row.fHalfLen)
                    continue;

                if (!bPhi)
                {
                    fPhi = MathF.Atan2(vec.Y, vec.X);
                    bPhi = true;
                }
                float fJ = (fPhi - row.fPhi0) / row.fDPhi;
                int j0 = (int)MathF.Round(fJ);
                Vector3[] aC = m_aavecC[i], aA = m_aavecA[i];
                int jMin = m_anJMin[i];
                for (int j = j0 - 1; j <= j0 + 1; j++)
                {
                    Vector3 vecC, vecA;
                    int k = j - jMin;
                    if ((uint)k < (uint)aC.Length)
                    {
                        vecC = aC[k];
                        vecA = aA[k];
                    }
                    else
                        HoleFrame(row, j, out vecC, out vecA);

                    Vector3 vecD = vec - vecC;
                    float fT = Vector3.Dot(vecD, vecA);
                    Vector3 vecPerp = vecD - fT * vecA;

                    float fRLocal = m_fR;
                    if (m_bConical)
                    {
                        // s = 0 at the metering inlet (+a, the annulus side),
                        // s = 1 at the diffusing exit (-a, the gas side)
                        float s = (row.fHalfLen - fT)
                                / MathF.Max(2f * row.fHalfLen, 1e-6f);
                        s = MathF.Min(1f, MathF.Max(0f, s));
                        fRLocal = m_fR * (1f + (m_fConeExp - 1f) * s);
                    }

                    float fRadial;
                    if (m_bTeardrop)
                        fRadial = fSdTeardrop(vecPerp, vecA, fRLocal);
                    else
                        fRadial = vecPerp.Length() - fRLocal;

                    float fCap = MathF.Abs(fT) - row.fHalfLen;
                    float fSd = MathF.Max(fRadial, fCap);
                    if (fSd < fBest) fBest = fSd;
                }
            }
            return fBest;
        }

        /// <summary>Centre (mid-wall) and axis of hole j of a row - the
        /// expressions of the original per-sample evaluation, unchanged.
        /// </summary>
        static void HoleFrame(in Row row, int j, out Vector3 vecC,
                              out Vector3 vecA)
        {
            float fPhiH = row.fPhi0 + j * row.fDPhi;
            float fCos = MathF.Cos(fPhiH), fSin = MathF.Sin(fPhiH);
            Vector3 vecRad = new Vector3(fCos, fSin, 0f);
            vecC = vecRad * row.fRMid + new Vector3(0f, 0f, row.fZc);
            vecA = vecRad * row.fAr + new Vector3(0f, 0f, row.fAz);
        }

        /// <summary>
        /// 2-D signed distance to the teardrop section = the convex hull of
        /// the circle of radius fR and an apex at fR*sqrt(2) along the build
        /// direction (a 45-degree roof). Standard round-cone SDF with the
        /// far radius set to zero.
        /// </summary>
        float fSdTeardrop(Vector3 vecPerp, Vector3 vecAxis, float fR)
        {
            Vector3 vecV = m_vecUp - Vector3.Dot(m_vecUp, vecAxis) * vecAxis;
            float fVLen = vecV.Length();
            if (fVLen < 0.15f)                       // axis parallel to build
                return vecPerp.Length() - fR;
            vecV /= fVLen;

            float fQy = Vector3.Dot(vecPerp, vecV);
            float fQx = MathF.Sqrt(MathF.Max(
                vecPerp.LengthSquared() - fQy * fQy, 0f));

            float fH = fR * MathF.Sqrt(2f);          // apex distance
            const float fB = 0.70710678f;            // (r - 0) / h
            const float fA = 0.70710678f;            // sqrt(1 - b^2)

            float fK = -fB * fQx + fA * fQy;
            if (fK < 0f)
                return MathF.Sqrt(fQx * fQx + fQy * fQy) - fR;
            if (fK > fA * fH)
            {
                float fDy = fQy - fH;
                return MathF.Sqrt(fQx * fQx + fDy * fDy);
            }
            return fA * fQx + fB * fQy - fR;
        }

        /// <summary>Effusion open-area ratio (porosity) of the field, for the
        /// report. Printable band is 1-5 % (Ciencia 5.1).</summary>
        public float fPorosity(CombustorParameters p)
        {
            if (m_aRows.Length == 0) return 0f;
            // wall area = sum over rows of (circumference x axial row pitch)
            float fSpan = m_afRowZ[m_afRowZ.Length - 1] - m_afRowZ[0];
            float fPitchZ = m_aRows.Length > 1
                ? fSpan / (m_aRows.Length - 1) : 1f;
            float fHoleArea = 0f, fWallArea = 0f;
            foreach (Row row in m_aRows)
            {
                fHoleArea += row.nHoles * MathF.PI * m_fR * m_fR;
                fWallArea += 2f * MathF.PI * row.fRMid * fPitchZ;
            }
            return fWallArea > 0f ? fHoleArea / fWallArea : 0f;
        }
    }
}
