//
// Liner.cs - the flame tube: revolution of the contract profile, minus the
// discrete hole rows, minus the effusion field, plus the dome and the
// injector seats.
//
// Recipe (WAVE3 section 2 table):
//
//   1. revolve the OUTER contour            -> solid body
//   2. subtract the same contour offset by  -> open-ended tube of wall t
//      -t_wall (extended past both ends)
//   3. for an annular/reverse-annular liner, add the INNER wall the same way
//   4. weld the dome (hemispherical or flat) onto the head end
//   5. add the injector seats (bosses) and the swirler seat
//   6. restricted morphological closing -> root fillet at the dome/barrel
//      junction, with the SAME radius the structural notch factor credits
//   7. subtract the discrete hole rows (circle or self-supporting teardrop)
//   8. subtract the effusion field (ONE implicit, never one boolean per hole)
//   9. subtract the seat bores last so nothing plugs them
//
// Order matters: the fillet closing runs BEFORE any hole is cut, otherwise
// it would fill the holes back in.
//
// The dome bulges AWAY from the barrel - a purely geometric decision taken
// from where the dome plane sits on the profile. The build orientation must
// never change the geometry; it only changes the DfAM verdict.
//

using System;
using System.Collections.Generic;
using System.Numerics;
using PicoGK;

namespace CombustorDesigner
{
    public static class Liner
    {
        /// <summary>Axial overshoot used when subtracting an inner solid so
        /// the resulting shell is an OPEN tube (Phy-AC casing trick).</summary>
        const float fOpenEndMm = 2f;

        /// <summary>Builds the complete liner part.</summary>
        public static Voxels voxLiner(CombustorParameters p,
                                      EffusionField oEffusion)
        {
            float fT = p.fPrintableMm(p.Liner.TWallMm, "liner wall");

            // D-392: every boolean below is IN PLACE and its tool solid is
            // disposed at once. The operators (LatticeUtils.Sub(vox, x) allocate a new
            // field per step and leave the old one and the tool to the
            // finalizer); at the 8 kg/s scale those native grids piled up to
            // ~10 GiB (see Casing). Same booleans, same order, same result.
            var oStage = System.Diagnostics.Stopwatch.StartNew();

            // 1-2. outer wall as an open tube
            Voxels vox = LatticeUtils.voxRevolveProfileZ(p.Liner.Profile, 2);
            LatticeUtils.Sub(vox, LatticeUtils.voxRevolveProfileZ(
                p.Liner.Profile, 2, -fT, fOpenEndMm));

            // 3. inner wall of an annular / reverse-annular liner
            if (bHasInnerWall(p))
            {
                Voxels voxIn = LatticeUtils.voxRevolveProfileZ(
                    p.Liner.Profile, 1, fT);
                LatticeUtils.Sub(voxIn, LatticeUtils.voxRevolveProfileZ(
                    p.Liner.Profile, 1, 0f, fOpenEndMm));
                LatticeUtils.Add(vox, voxIn);
                Library.Log("Liner: annular flame tube - inner wall added "
                          + $"(r_inner {p.Liner.Profile[0][1]:0.#} -> "
                          + $"{p.Liner.Profile[p.Liner.Profile.Length - 1][1]:0.#}"
                          + " mm)");
            }
            LogStage("walls", oStage);

            // 3b. optional impingement/effusion double wall - the cavity that
            //     makes powder drainability the binding constraint (4.4)
            if (p.Liner.DoubleWall != null)
                LatticeUtils.Add(vox, voxDoubleWall(p, fT));

            // 4. dome
            LatticeUtils.Add(vox, voxDome(p, fT));
            LogStage("dome", oStage);

            // 5. injector + swirler seats (material first, bores at step 9)
            LatticeUtils.Add(vox, voxSeats(p, fT));

            // 6. root fillet by RESTRICTED morphological closing
            LatticeUtils.FlushFinalizers();
            vox = voxWithDomeFillet(p, vox);
            LogStage("seats + root fillet", oStage);

            // 7. discrete hole rows
            Voxels voxHoles = voxHoleRows(p);
            if (voxHoles != null && !voxHoles.bIsEmpty())
                vox.BoolSubtract(voxHoles);
            voxHoles?.Dispose();
            LogStage("hole rows", oStage);

            // 8. the effusion field - hundreds to thousands of holes in ONE
            //    voxel field, the implicit evaluated over shell tiles only
            if (oEffusion != null && !oEffusion.bIsEmpty)
            {
                LatticeUtils.FlushFinalizers();
                var oWatch = System.Diagnostics.Stopwatch.StartNew();
                Voxels voxEff = oEffusion.voxRender(p.VoxelSizeMm,
                                                    out string strHow);
                LatticeUtils.Sub(vox, voxEff);
                Library.Log(
                    $"Liner: effusion field - {oEffusion.nHoleCount} holes ("
                    + (oEffusion.bIsTeardrop ? "teardrop" : p.Effusion.Shape)
                    + $", d {p.Effusion.DMm:0.###} mm, "
                    + $"{p.Effusion.AngleDeg:0.#} deg from the wall, porosity "
                    + $"{100f * oEffusion.fPorosity(p):0.##} %) rendered as ONE "
                    + $"{strHow} in {oWatch.ElapsedMilliseconds / 1000f:0.0} s");
            }
            LogStage("effusion", oStage);

            // 9. bores last
            Voxels voxBores = voxSeatBores(p);
            if (!voxBores.bIsEmpty())
                vox.BoolSubtract(voxBores);
            voxBores.Dispose();
            LogStage("seat bores / cup throats", oStage);
            return vox;
        }

        /// <summary>Per-stage timing line (D-392 profiling evidence).</summary>
        static void LogStage(string strStage,
                             System.Diagnostics.Stopwatch oWatch)
        {
            Library.Log($"TIMING liner {strStage}: "
                      + $"{oWatch.ElapsedMilliseconds / 1000f:0.0} s");
            oWatch.Restart();
        }

        /// <summary>True when the contract carries a real inner annulus wall
        /// (any r_inner > 0). A can has r_inner = 0 everywhere.</summary>
        public static bool bHasInnerWall(CombustorParameters p)
        {
            foreach (float[] afRow in p.Liner.Profile)
                if (afRow[1] > 1e-3f)
                    return true;
            return false;
        }

        // -- dome ---------------------------------------------------------

        /// <summary>+1 when the dome bulges towards +Z, -1 towards -Z. Set
        /// by where the dome plane sits on the liner profile, NOT by the
        /// build orientation.</summary>
        public static float fDomeSign(CombustorParameters p)
        {
            float fZ0 = p.Liner.Profile[0][0];
            float fZ1 = p.Liner.Profile[p.Liner.Profile.Length - 1][0];
            return p.fDomeZMm() <= 0.5f * (fZ0 + fZ1) ? -1f : 1f;
        }

        /// <summary>Axial station of the dome OUTER surface at a given
        /// radius - where an injector seat has to sit. Flat domes are at the
        /// dome plane; hemispherical ones follow the sphere.</summary>
        public static float fDomeSurfaceZ(CombustorParameters p, float fRadius)
        {
            float fZD = p.fDomeZMm();
            string strType = (p.Liner.DomeType ?? "").ToLowerInvariant();
            if (!strType.StartsWith("hemi"))
                return fZD;
            float fRo = p.fLinerROuterAt(fZD);
            float fArg = fRo * fRo - fRadius * fRadius;
            float fH = fArg > 0f ? MathF.Sqrt(fArg) : 0f;
            return fZD + fDomeSign(p) * fH;
        }

        /// <summary>
        /// Head-end closure. "hemispherical" is a concentric spherical shell
        /// of radius r_outer(z_dome) centred on the dome plane, so the wall
        /// thickness is uniform and it welds flush onto the barrel;
        /// "flat" is an annular plate of thickness t. "none" builds nothing.
        /// </summary>
        public static Voxels voxDome(CombustorParameters p, float fT)
        {
            string strType = (p.Liner.DomeType ?? "").ToLowerInvariant();
            if (strType.StartsWith("none"))
                return new Voxels();

            float fZD = p.fDomeZMm();
            float fRo = p.fLinerROuterAt(fZD);
            float fRi = p.fLinerRInnerAt(fZD);
            float fSign = fDomeSign(p);

            if (strType.StartsWith("flat"))
            {
                float fZ0 = fSign < 0f ? fZD - fT : fZD;
                Voxels voxFlat = LatticeUtils.voxCylinderZ(
                    new Vector3(0f, 0f, fZ0), fRo, fT);
                if (fRi > 1e-3f)
                    LatticeUtils.Sub(voxFlat, LatticeUtils.voxCylinderZ(
                        new Vector3(0f, 0f, fZ0 - 1f), fRi, fT + 2f));
                return voxFlat;
            }

            // hemispherical: concentric shell, contour sampled fine enough
            // that the voxelizer never sees a terrace
            const int nSeg = 48;
            float fRin = MathF.Max(fRo - fT, 0.05f);
            var afOuter = new float[nSeg + 1][];
            var afInner = new float[nSeg + 1][];
            for (int i = 0; i <= nSeg; i++)
            {
                // a = 0 at the pole, 90 deg at the equator (= dome plane)
                float fA = 0.5f * MathF.PI * i / nSeg;
                int k = fSign < 0f ? i : nSeg - i;    // keep z ascending
                afOuter[k] = new float[]
                {
                    fZD + fSign * fRo * MathF.Cos(fA),
                    fRo * MathF.Sin(fA),
                };
                afInner[k] = new float[]
                {
                    fZD + fSign * fRin * MathF.Cos(fA),
                    fRin * MathF.Sin(fA),
                };
            }
            Voxels vox = LatticeUtils.voxRevolveProfileZ(afOuter, 1);
            LatticeUtils.Sub(vox, LatticeUtils.voxRevolveProfileZ(
                afInner, 1, 0f, fOpenEndMm));
            if (fRi > 1e-3f)
            {
                // annular dome: open the bore for the inner flow path
                LatticeUtils.Sub(vox, LatticeUtils.voxCylinderZ(
                    new Vector3(0f, 0f, fZD - fRo - 2f), fRi, 2f * fRo + 4f));
            }
            return vox;
        }

        // -- double wall --------------------------------------------------

        /// <summary>Outer skin of an impingement/effusion double wall, held
        /// off the liner by gap_mm. Its cavity is exactly the one the powder
        /// drainability check has to clear.</summary>
        static Voxels voxDoubleWall(CombustorParameters p, float fT)
        {
            DoubleWallParams o = p.Liner.DoubleWall;
            float fTo = p.fPrintableMm(o.TOuterMm > 0f ? o.TOuterMm : fT,
                                       "liner double wall (outer skin)");
            float fGap = MathF.Max(o.GapMm, 2f * p.VoxelSizeMm);
            float fZ0 = MathF.Min(o.Z0Mm, o.Z1Mm);
            float fZ1 = MathF.Max(o.Z0Mm, o.Z1Mm);
            if (fZ1 - fZ0 < 1e-3f)
                return new Voxels();

            const int nSeg = 32;
            var afOut = new float[nSeg + 1][];
            var afIn = new float[nSeg + 1][];
            for (int i = 0; i <= nSeg; i++)
            {
                float fZ = fZ0 + (fZ1 - fZ0) * i / nSeg;
                float fR = p.fLinerROuterAt(fZ);
                afOut[i] = new float[] { fZ, fR + fGap + fTo };
                afIn[i] = new float[] { fZ, fR + fGap };
            }
            Voxels vox = LatticeUtils.voxRevolveProfileZ(afOut, 1);
            LatticeUtils.Sub(vox, LatticeUtils.voxRevolveProfileZ(
                afIn, 1, 0f, fOpenEndMm));
            Library.Log($"Liner: double wall gap {fGap:0.##} mm over z "
                      + $"{fZ0:0.#}-{fZ1:0.#} mm - this cavity MUST clear the "
                      + "powder drainability check (EstadoDelArte 4.4)");
            return vox;
        }

        // -- discrete hole rows -------------------------------------------

        /// <summary>
        /// The tool solid for every discrete hole: n cylinders (or teardrop
        /// prisms) per row. Dozens of holes, so plain booleans are the right
        /// operation here - the implicit-field treatment is reserved for the
        /// effusion field, which has hundreds.
        ///
        /// Shape rules (EstadoDelArte 4.3): the contract shape wins; when it
        /// says "circle" but the axis is closer than 45 degrees to the build
        /// horizontal, or the hole is a dilution hole of 5 mm or more, the
        /// section is upgraded to a teardrop and the upgrade is LOGGED.
        /// </summary>
        public static Voxels voxHoleRows(CombustorParameters p)
        {
            if (p.HoleRows.Count == 0)
                return null;

            Vector3 vecUp = new Vector3(0f, 0f, p.fBuildSign());
            Voxels voxAll = null;
            foreach (HoleRowParams row in p.HoleRows)
            {
                if (row.bIsEffusion())
                    continue;                     // owned by EffusionField
                float fR = 0.5f * row.DMm;
                if (fR <= 0f)
                    continue;

                float fRWall = row.RMm > 0f
                    ? row.RMm : p.fLinerROuterAt(row.ZMm);
                Vector3 vecAxLocal = vecRowAxis(row);   // (radial, tang, axial)

                bool bTeardrop = bTeardropFor(p, row, vecAxLocal,
                                              out string strWhy);
                if (strWhy != null)
                    Library.Log(strWhy);

                for (int k = 0; k < row.N; k++)
                {
                    float fPhi = row.Phi0Deg * MathF.PI / 180f
                               + 2f * MathF.PI * k / row.N;
                    Vector3 vecRad = new Vector3(MathF.Cos(fPhi),
                                                 MathF.Sin(fPhi), 0f);
                    Vector3 vecTan = new Vector3(-MathF.Sin(fPhi),
                                                  MathF.Cos(fPhi), 0f);
                    Vector3 vecAxis = vecAxLocal.X * vecRad
                                    + vecAxLocal.Y * vecTan
                                    + vecAxLocal.Z * Vector3.UnitZ;
                    vecAxis = MathHelpers.vecSafeNormalize(vecAxis, vecRad);

                    Vector3 vecC = vecRad * fRWall
                                 + new Vector3(0f, 0f, row.ZMm);
                    float fHalf = 2f * p.Liner.TWallMm + row.DMm + 1f;
                    Vector3 vecA = vecC - fHalf * vecAxis;
                    Vector3 vecB = vecC + fHalf * vecAxis;

                    Voxels voxHole = bTeardrop
                        ? LatticeUtils.voxTeardropBetween(vecA, vecB, fR, vecUp)
                        : LatticeUtils.voxBeam(vecA, vecB, fR);
                    if (voxHole.bIsEmpty())
                    {
                        voxHole.Dispose();
                        continue;
                    }
                    voxAll = voxAll == null ? voxHole : LatticeUtils.Add(voxAll, voxHole);
                }
            }
            return voxAll;
        }

        /// <summary>Hole axis in the local (radial, tangential, axial) frame.
        /// `axis` wins over `angle_deg`, which is the tilt away from RADIAL
        /// towards +Z - the dfam_checker convention.</summary>
        public static Vector3 vecRowAxis(HoleRowParams row)
        {
            if (row.Axis != null && row.Axis.Length >= 3)
            {
                Vector3 v = new Vector3(row.Axis[0], row.Axis[1], row.Axis[2]);
                return MathHelpers.vecSafeNormalize(v, Vector3.UnitX);
            }
            float fA = row.AngleDeg * MathF.PI / 180f;
            return new Vector3(MathF.Cos(fA), 0f, MathF.Sin(fA));
        }

        /// <summary>Decides circle vs teardrop and explains why.</summary>
        static bool bTeardropFor(CombustorParameters p, HoleRowParams row,
                                 Vector3 vecAxLocal, out string strWhy)
        {
            strWhy = null;
            if ((row.Shape ?? "").ToLowerInvariant().StartsWith("tear"))
                return true;

            // the axis has no tangential component in the meridional plane,
            // so the angle from the build horizontal is asin(|axial|)
            float fDeg = MathF.Asin(MathF.Min(1f, MathF.Abs(vecAxLocal.Z)))
                       * 180f / MathF.PI;
            if (fDeg < CombustorParameters.TEARDROP_ANGLE_DEG)
            {
                strWhy = $"WARNING hole row '{row.Id}' ({row.Zone}, d "
                       + $"{row.DMm:0.##} mm): axis at {fDeg:0.#} deg from the "
                       + "build horizontal (< "
                       + $"{CombustorParameters.TEARDROP_ANGLE_DEG:0} deg) - "
                       + "section upgraded to TEARDROP (EstadoDelArte 4.3 "
                       + "rule 1). Update hole_rows[].shape in the contract "
                       + "so Python and C# agree.";
                return true;
            }
            if (row.DMm >= 5f)
            {
                strWhy = $"WARNING hole row '{row.Id}' ({row.Zone}): dilution "
                       + $"hole d = {row.DMm:0.##} mm >= 5 mm - section "
                       + "upgraded to TEARDROP; its roof would need supports "
                       + "that cannot be removed from a closed liner "
                       + "(EstadoDelArte 4.3 rule 3).";
                return true;
            }
            return false;
        }

        // -- injector / swirler seats -------------------------------------

        /// <summary>Bosses that receive the injectors. The matching bores are
        /// cut by voxSeatBores AFTER the fillet pass.</summary>
        public static Voxels voxSeats(CombustorParameters p, float fT)
        {
            if (p.bScaleCups())
            {
                // annular_5_25: the seat of each station IS its swirl cup
                // (shroud tube welded through the dome, hub bored for the
                // injector) - built by Swirler.voxScaleCups.
                Library.Log($"Liner: {p.Swirler.CupCount} swirl-cup stations "
                          + "on the injector ring - the cup shroud is the "
                          + "seat; no axis boss is built.");
                return new Voxels();
            }
            Voxels vox = null;
            float fSign = fDomeSign(p);
            float fDSeat = fSeatDMm(p);
            float fBossT = p.fPrintableMm(1.5f * fT, "injector seat boss");

            foreach (Vector3 vecPos in aInjectorPositions(p))
            {
                Vector3 vecOut = new Vector3(0f, 0f, fSign);
                Voxels voxB = LatticeUtils.voxBeam(
                    vecPos + vecOut * fBossT,
                    vecPos - vecOut * (2f * fBossT),
                    0.5f * fDSeat + fBossT);
                vox = vox == null ? voxB : LatticeUtils.Add(vox, voxB);
            }
            return vox ?? new Voxels();
        }

        /// <summary>The bores through the dome: injector seats and, when the
        /// swirler sits on the axis, the swirler throat.</summary>
        public static Voxels voxSeatBores(CombustorParameters p)
        {
            if (p.bScaleCups())
                return voxCupThroats(p);
            Voxels vox = null;
            float fZD = p.fDomeZMm();
            float fRo = p.fLinerROuterAt(fZD);
            float fDSeat = fSeatDMm(p);
            float fReach = 2f * fRo + 8f;

            foreach (Vector3 vecPos in aInjectorPositions(p))
            {
                Voxels voxB = LatticeUtils.voxBeam(
                    vecPos + new Vector3(0f, 0f, fReach),
                    vecPos - new Vector3(0f, 0f, fReach),
                    0.5f * fDSeat);
                vox = vox == null ? voxB : LatticeUtils.Add(vox, voxB);
            }

            if (p.Swirler.NVanes > 0 && p.Swirler.DTipMm > 0f
                && p.Injectors.RingRMm <= 1e-3f && p.Injectors.Count == 1)
            {
                float fRT = 0.5f * p.Swirler.DTipMm + p.VoxelSizeMm;
                Voxels voxS = LatticeUtils.voxCylinderZ(
                    new Vector3(0f, 0f, fZD - fReach), fRT, 2f * fReach);
                vox = vox == null ? voxS : LatticeUtils.Add(vox, voxS);
            }
            return vox ?? new Voxels();
        }

        /// <summary>Diameter of one injector seat.</summary>
        public static float fSeatDMm(CombustorParameters p)
        {
            string strType = (p.Injectors.Type ?? "").ToLowerInvariant();
            if (strType.StartsWith("vapor") && p.Injectors.TubeDMm > 0f)
                return p.Injectors.TubeDMm + 2f * p.VoxelSizeMm;
            if (p.Injectors.BodyDMm > 0f)
                return p.Injectors.BodyDMm + 2f * p.VoxelSizeMm;
            if (p.Injectors.Count == 1 && p.Swirler.DTipMm > 0f)
                return p.Swirler.DTipMm;
            return MathF.Max(6f, 0.25f * p.fLinerROuterAt(p.fDomeZMm()));
        }

        /// <summary>Ring radius the injectors sit on (0 = on the axis).</summary>
        public static float fInjectorRingRMm(CombustorParameters p)
        {
            if (p.Injectors.RingRMm > 1e-3f)
                return p.Injectors.RingRMm;
            if (Math.Max(1, p.Injectors.Count) == 1 && !bHasInnerWall(p))
                return 0f;
            float fZD = p.fDomeZMm();
            return 0.5f * (p.fLinerROuterAt(fZD) + p.fLinerRInnerAt(fZD));
        }

        /// <summary>Injector seat positions ON the dome outer surface. One
        /// injector with no declared ring radius sits on the axis (a can);
        /// several are spread evenly on the ring.</summary>
        public static List<Vector3> aInjectorPositions(CombustorParameters p)
        {
            var aPos = new List<Vector3>();
            int n = Math.Max(1, p.Injectors.Count);
            float fRing = fInjectorRingRMm(p);
            float fZ = fDomeSurfaceZ(p, fRing);
            if (fRing <= 1e-3f)
            {
                aPos.Add(new Vector3(0f, 0f, fZ));
                return aPos;
            }
            for (int k = 0; k < n; k++)
            {
                float fPhi = p.Injectors.Phi0Deg * MathF.PI / 180f
                           + 2f * MathF.PI * k / n;
                aPos.Add(new Vector3(fRing * MathF.Cos(fPhi),
                                     fRing * MathF.Sin(fPhi), fZ));
            }
            return aPos;
        }

        // -- swirl-cup ring of the annular_5_25 class (17.1-03, D-386) ----

        /// <summary>One cup station: centre on the dome OUTER surface at the
        /// injector ring radius and phase, and the axial span [fZLo, fZHi]
        /// of its shroud tube and throat. The span covers the whole crossing
        /// of the (oblique) dome wall by the cup footprint plus the vane
        /// band, two voxels past each side, and nothing else - the trim is
        /// local to the dome.</summary>
        public struct CupStation
        {
            public Vector3 vecC;
            public float fPhiDeg;
            public float fZLo, fZHi;
        }

        /// <summary>Outer radius of one cup: d_tip/2 + shroud wall.</summary>
        public static float fCupOuterRMm(CombustorParameters p)
            => 0.5f * p.Swirler.DTipMm + p.fPrintableQuietMm(p.Liner.TWallMm);

        /// <summary>Axial station of the dome INNER surface at a radius
        /// (hemispherical: concentric sphere of radius r_outer - t).</summary>
        public static float fDomeInnerSurfaceZ(CombustorParameters p,
                                               float fRadius)
        {
            float fZD = p.fDomeZMm();
            float fT = p.fPrintableQuietMm(p.Liner.TWallMm);
            string strType = (p.Liner.DomeType ?? "").ToLowerInvariant();
            if (!strType.StartsWith("hemi"))
                return fZD - fDomeSign(p) * fT;
            float fRin = MathF.Max(p.fLinerROuterAt(fZD) - fT, 0.05f);
            float fArg = fRin * fRin - fRadius * fRadius;
            float fH = fArg > 0f ? MathF.Sqrt(fArg) : 0f;
            return fZD + fDomeSign(p) * fH;
        }

        public static List<CupStation> aCupStations(CombustorParameters p)
        {
            var aSt = new List<CupStation>();
            float fRCup = fCupOuterRMm(p);
            float fRing = fInjectorRingRMm(p);
            float fHalf = 0.5f * Swirler.fChordMm(p);
            float fM = 2f * p.VoxelSizeMm;
            float[] afR = { MathF.Max(fRing - fRCup, 0f), fRing,
                            fRing + fRCup };
            float fLo = float.PositiveInfinity, fHi = float.NegativeInfinity;
            foreach (float fR in afR)
            {
                foreach (float fZ in new[] { fDomeSurfaceZ(p, fR),
                                             fDomeInnerSurfaceZ(p, fR) })
                {
                    fLo = MathF.Min(fLo, fZ);
                    fHi = MathF.Max(fHi, fZ);
                }
            }
            int k = 0;
            foreach (Vector3 vecPos in aInjectorPositions(p))
            {
                float fPhi = p.Injectors.Phi0Deg
                           + 360f * k / Math.Max(1, p.Injectors.Count);
                aSt.Add(new CupStation
                {
                    vecC = vecPos,
                    fPhiDeg = fPhi,
                    fZLo = MathF.Min(fLo, vecPos.Z - fHalf) - fM,
                    fZHi = MathF.Max(fHi, vecPos.Z + fHalf) + fM,
                });
                k++;
            }
            return aSt;
        }

        /// <summary>The throat of every cup: the dome wall inside the
        /// shroud bore (r = d_tip/2) is removed over the station span only,
        /// so the central passage and the foreign legs of the flame tube
        /// are never touched (Validate checks the footprint).</summary>
        static Voxels voxCupThroats(CombustorParameters p)
        {
            Voxels vox = null;
            float fRT = 0.5f * p.Swirler.DTipMm;
            foreach (CupStation st in aCupStations(p))
            {
                Voxels voxT = LatticeUtils.voxCylinderZ(
                    new Vector3(st.vecC.X, st.vecC.Y, st.fZLo), fRT,
                    st.fZHi - st.fZLo);
                vox = vox == null ? voxT : LatticeUtils.Add(vox, voxT);
                Library.Log($"Liner: cup station phi {st.fPhiDeg:0.###} deg at "
                          + $"({st.vecC.X:0.###}, {st.vecC.Y:0.###}, "
                          + $"{st.vecC.Z:0.###}) mm, throat r {fRT:0.###} mm "
                          + $"over z {st.fZLo:0.###}..{st.fZHi:0.###} mm");
            }
            return vox ?? new Voxels();
        }

        // -- fillets ------------------------------------------------------

        /// <summary>
        /// Root fillet at the dome/barrel junction by RESTRICTED
        /// morphological closing - the identical recipe Phy-AC uses for the
        /// blade-root fillet: ONE Fillet pass on the whole part (the
        /// expensive step), then masked to a band around the junction and
        /// unioned back, so nothing else in the part is rounded.
        ///
        /// The radius is `liner.fillet_r_mm`, which is the SAME number the
        /// structural side must credit in its stress concentration - one
        /// source of truth between geometry and structures, exactly as
        /// Phy-AC couples `blade_fillet_r_mm` to its Peterson K_t.
        /// </summary>
        public static Voxels voxWithDomeFillet(CombustorParameters p,
                                               Voxels vox)
        {
            float fR = p.Liner.FilletRMm;
            if (fR <= 0f)
                return vox;

            float fZD = p.fDomeZMm();
            float fBand = fR + 2f * p.VoxelSizeMm;
            float fRo = p.fLinerROuterAt(fZD) + fBand;

            Voxels voxBand = LatticeUtils.voxCylinderZ(
                new Vector3(0f, 0f, fZD - fBand), fRo, 2f * fBand);
            // D-392: the closing only reads the part within 2 r of a point,
            // so on a large part it runs on a copy TRIMMED to the band plus
            // 2 r and four voxels of margin: inside the band the result is
            // the one the whole-part pass gives (identical triangles at
            // 1e-4 mm on the KJ66 liner), at a fraction of the cost (43 s
            // on the 8 kg/s liner at 0.8 mm). The trim faces are farther
            // than 2 r from the band and never reach it. A part whose whole
            // box is under the lattice budget keeps the historical
            // whole-part pass bit for bit.
            float fReach = 2f * fR + 4f * p.VoxelSizeMm;
            float fXY = fRo + fReach;
            float fZ0 = p.Liner.Profile[0][0];
            float fZ1 = p.Liner.Profile[p.Liner.Profile.Length - 1][0];
            double dV = p.VoxelSizeMm;
            double dPartVoxels = (2.0 * fXY / dV) * (2.0 * fXY / dV)
                               * (Math.Abs(fZ1 - fZ0) + 2.0 * fRo) / dV;
            Voxels voxFil = dPartVoxels <= LatticeUtils.dLatticeBudgetVoxels
                ? new Voxels(vox)
                : vox.voxTrim(LatticeUtils.oBBox(
                    new Vector3(-fXY, -fXY, fZD - fBand - fReach),
                    new Vector3(fXY, fXY, fZD + fBand + fReach)));
            voxFil.Fillet(fR);
            Library.Log($"Liner: dome root fillet r = {fR:0.##} mm by "
                      + "restricted morphological closing (same radius the "
                      + "structural notch factor credits)");
            voxFil.BoolIntersect(voxBand);
            voxBand.Dispose();
            vox.BoolAdd(voxFil);
            voxFil.Dispose();
            return vox;
        }
    }
}
