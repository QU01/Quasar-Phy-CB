//
// Injector.cs - the fuel injection hardware, one part family per type
// (WAVE3 section 2 table):
//
//   airblast   prefilming body: outer barrel, central fuel post, thin
//              prefilmer lip that the swirled air shears
//   simplex    pressure-swirl body: barrel with a stepped bore and the
//              final metering orifice
//   vaporizer  the "walking stick" - a tube swept along a polyline with a
//              bend (180 deg on a KJ66-class machine): it goes downstream
//              inside the flame, turns back and discharges upstream into
//              the primary zone. Exported as <Name>_Vaporizer{i}.stl
//   micromix   a ring of small tubes, one part carrying them all
//
// The sweep is a chain of ROUND-capped lattice beams: consecutive segments
// weld at any bend angle with no notch, which is what a swept tube needs.
// The bore is the same polyline swept at the inner radius and EXTENDED past
// both ends, so the tube comes out open rather than capped.
//

using System;
using System.Collections.Generic;
using System.Numerics;
using PicoGK;

namespace CombustorDesigner
{
    public static class Injector
    {
        /// <summary>Number of parts this injector family exports. The
        /// vaporizer exports one STL per tube (they are separate pieces on a
        /// real engine); every other type is one part.</summary>
        public static int nPartCount(CombustorParameters p)
        {
            string s = strType(p);
            if (s == "vaporizer")
                return Math.Max(1, p.Injectors.Count);
            return 1;
        }

        public static string strPartName(CombustorParameters p, int i)
            => strType(p) == "vaporizer" ? $"Vaporizer{i + 1}" : "Injector";

        public static string strType(CombustorParameters p)
        {
            string s = (p.Injectors.Type ?? "").ToLowerInvariant();
            if (s.StartsWith("vapor")) return "vaporizer";
            if (s.StartsWith("simplex") || s.StartsWith("pressure"))
                return "simplex";
            if (s.StartsWith("micro")) return "micromix";
            return "airblast";
        }

        public static Voxels voxPart(CombustorParameters p, int i)
        {
            switch (strType(p))
            {
                case "vaporizer": return voxVaporizer(p, i);
                case "micromix": return voxMicromix(p);
                case "simplex": return voxSimplex(p);
                default: return voxAirblast(p);
            }
        }

        // -- vaporizer: the walking stick ---------------------------------

        /// <summary>
        /// Centreline of vaporizer i, in the meridional plane at its
        /// angular station: straight leg downstream, circular bend of
        /// `bend_deg`, straight return leg. The bend turns the tube inwards
        /// (towards the axis), which is how a walking stick discharges back
        /// into the primary zone.
        /// </summary>
        public static List<Vector3> aVaporizerPath(CombustorParameters p,
                                                   int i)
        {
            int n = Math.Max(1, p.Injectors.Count);
            float fPhi = p.Injectors.Phi0Deg * MathF.PI / 180f
                       + 2f * MathF.PI * i / n;
            Vector3 vecRad = new Vector3(MathF.Cos(fPhi), MathF.Sin(fPhi), 0f);

            float fD = p.Injectors.TubeDMm > 0f
                ? p.Injectors.TubeDMm
                : MathF.Max(4f, 0.12f * p.fLinerROuterAt(p.fDomeZMm()));
            float fRb = p.Injectors.BendRMm > 0f
                ? p.Injectors.BendRMm : 1.5f * fD;

            float fZDome = p.fDomeZMm();
            float fZ0 = float.IsNaN(p.Injectors.TubeZ0Mm)
                ? fZDome : p.Injectors.TubeZ0Mm;
            float fZ1 = float.IsNaN(p.Injectors.TubeZ1Mm)
                ? fZ0 + MathF.Max(6f * fD, 12f) : p.Injectors.TubeZ1Mm;
            float fSign = MathF.Sign(fZ1 - fZ0);
            if (fSign == 0f) { fSign = 1f; fZ1 = fZ0 + 12f; }

            float fRing = Liner.fInjectorRingRMm(p);
            if (fRing <= 1e-3f)
                fRing = 0.6f * p.fLinerROuterAt(fZDome);
            // the bend centre sits fRb inboard: the return leg ends up at
            // fRing - 2*fRb, i.e. closer to the axis
            float fRCentre = MathF.Max(fRing - fRb, 0.5f * fD);

            float fBend = MathF.Abs(p.Injectors.BendDeg);
            if (fBend < 5f) fBend = 180f;
            fBend = MathF.Min(fBend, 180f);
            int nArc = Math.Max(6, (int)MathF.Round(fBend / 10f));

            var aPts = new List<Vector3>();
            // inlet leg: starts one diameter upstream of the dome plane so
            // the tube pierces the dome seat
            aPts.Add(vecRad * fRing
                     + new Vector3(0f, 0f, fZ0 - fSign * 1.5f * fD));
            aPts.Add(vecRad * fRing + new Vector3(0f, 0f, fZ1));
            // bend in the (radial, axial) plane about (fRCentre, fZ1)
            for (int k = 1; k <= nArc; k++)
            {
                float fA = fBend * MathF.PI / 180f * k / nArc;
                float fRr = fRCentre + fRb * MathF.Cos(fA);
                float fZz = fZ1 + fSign * fRb * MathF.Sin(fA);
                aPts.Add(vecRad * fRr + new Vector3(0f, 0f, fZz));
            }
            // Return leg back upstream, discharging at the dome plane so the
            // fuel-air jet fires INTO the primary recirculation. Ending it
            // any further upstream would push the mouth outside the liner.
            Vector3 vecLast = aPts[aPts.Count - 1];
            float fRLast = MathF.Sqrt(vecLast.X * vecLast.X
                                    + vecLast.Y * vecLast.Y);
            float fZEnd = fZ0;
            if (MathF.Abs(fZEnd - vecLast.Z) < 2f * fD)
                fZEnd = vecLast.Z - fSign * 4f * fD;
            aPts.Add(vecRad * fRLast + new Vector3(0f, 0f, fZEnd));
            return aPts;
        }

        public static Voxels voxVaporizer(CombustorParameters p, int i)
        {
            var aPts = aVaporizerPath(p, i);
            float fD = p.Injectors.TubeDMm > 0f
                ? p.Injectors.TubeDMm
                : MathF.Max(4f, 0.12f * p.fLinerROuterAt(p.fDomeZMm()));
            float fT = p.fPrintableMm(
                p.Injectors.TubeTMm > 0f ? p.Injectors.TubeTMm
                                         : p.Liner.TWallMm,
                $"vaporizer {i + 1} tube wall");
            float fRo = 0.5f * fD;
            float fRi = MathF.Max(fRo - fT, 0.15f * fRo);

            Voxels vox = LatticeUtils.voxSweepPolyline(aPts, fRo, true);
            // open the bore: same path, extended past both ends
            var aBore = new List<Vector3>(aPts);
            aBore.Insert(0, aPts[0] + 3f * Vector3.Normalize(aPts[0] - aPts[1]));
            aBore.Add(aPts[aPts.Count - 1]
                      + 3f * Vector3.Normalize(aPts[aPts.Count - 1]
                                               - aPts[aPts.Count - 2]));
            LatticeUtils.Sub(vox, LatticeUtils.voxSweepPolyline(aBore, fRi, true));

            Library.Log($"Vaporizer {i + 1}: walking stick swept over "
                      + $"{aPts.Count} points, d {fD:0.##} mm, wall "
                      + $"{fT:0.##} mm, bend {p.Injectors.BendDeg:0.#} deg");
            return vox;
        }

        // -- airblast / simplex bodies ------------------------------------

        static Voxels voxAirblast(CombustorParameters p)
        {
            float fZ = Swirler.fZMm(p);
            float fDBody = fBodyDMm(p);
            float fL = MathF.Max(1.6f * fDBody, 8f);
            float fT = p.fPrintableMm(
                p.Injectors.PrefilmerTMm > 0f ? p.Injectors.PrefilmerTMm
                                              : p.Liner.TWallMm,
                "injector prefilmer");
            float fBore = fBoreDMm(p);
            float fSign = Liner.fDomeSign(p);

            Voxels vox = null;
            foreach (Vector3 vecPos in Liner.aInjectorPositions(p))
            {
                Vector3 vecBase = new Vector3(vecPos.X, vecPos.Y,
                                              fZ + fSign * 0.5f * fL);
                // outer barrel (open tube)
                Voxels voxB = LatticeUtils.voxBeam(
                    vecBase, vecBase - new Vector3(0f, 0f, fSign * fL),
                    0.5f * fDBody);
                LatticeUtils.Sub(voxB, LatticeUtils.voxBeam(
                    vecBase + new Vector3(0f, 0f, fSign),
                    vecBase - new Vector3(0f, 0f, fSign * (fL + 1f)),
                    0.5f * fDBody - fT));
                // central fuel post with its metering bore
                Voxels voxPost = LatticeUtils.voxBeam(
                    vecBase, vecBase - new Vector3(0f, 0f, fSign * 0.65f * fL),
                    0.5f * fBore + fT);
                LatticeUtils.Sub(voxPost, LatticeUtils.voxBeam(
                    vecBase + new Vector3(0f, 0f, fSign),
                    vecBase - new Vector3(0f, 0f, fSign * (0.65f * fL + 1f)),
                    0.5f * fBore));
                // prefilmer lip: a thin ring bridging post and barrel
                Voxels voxLip = LatticeUtils.voxBeam(
                    vecBase - new Vector3(0f, 0f, fSign * 0.65f * fL),
                    vecBase - new Vector3(0f, 0f,
                                          fSign * (0.65f * fL + fT)),
                    0.5f * fDBody - fT);
                LatticeUtils.Sub(voxLip, LatticeUtils.voxBeam(
                    vecBase - new Vector3(0f, 0f, fSign * (0.65f * fL - 1f)),
                    vecBase - new Vector3(0f, 0f,
                                          fSign * (0.65f * fL + fT + 1f)),
                    0.5f * fBore));
                Voxels voxOne = LatticeUtils.Add(LatticeUtils.Add(voxB, voxPost), voxLip);
                vox = vox == null ? voxOne : LatticeUtils.Add(vox, voxOne);
            }
            Library.Log($"Injector: {p.Injectors.Count} airblast body(ies), "
                      + $"d_body {fDBody:0.##} mm, bore {fBore:0.##} mm, "
                      + $"ALR {p.Injectors.ALR:0.##}");
            return vox ?? new Voxels();
        }

        static Voxels voxSimplex(CombustorParameters p)
        {
            float fZ = Swirler.fZMm(p);
            float fDBody = fBodyDMm(p);
            float fL = MathF.Max(1.8f * fDBody, 9f);
            float fBore = fBoreDMm(p);
            float fSign = Liner.fDomeSign(p);

            Voxels vox = null;
            foreach (Vector3 vecPos in Liner.aInjectorPositions(p))
            {
                Vector3 vecBase = new Vector3(vecPos.X, vecPos.Y,
                                              fZ + fSign * 0.5f * fL);
                Voxels voxB = LatticeUtils.voxBeam(
                    vecBase, vecBase - new Vector3(0f, 0f, fSign * fL),
                    0.5f * fDBody);
                // stepped bore: wide swirl chamber then the metering orifice
                LatticeUtils.Sub(voxB, LatticeUtils.voxBeam(
                    vecBase + new Vector3(0f, 0f, fSign),
                    vecBase - new Vector3(0f, 0f, fSign * 0.7f * fL),
                    MathF.Max(1.8f * 0.5f * fBore, 0.5f * fBore
                              + 2f * p.VoxelSizeMm)));
                LatticeUtils.Sub(voxB, LatticeUtils.voxBeam(
                    vecBase - new Vector3(0f, 0f, fSign * 0.6f * fL),
                    vecBase - new Vector3(0f, 0f, fSign * (fL + 1f)),
                    0.5f * fBore));
                vox = vox == null ? voxB : LatticeUtils.Add(vox, voxB);
            }
            Library.Log($"Injector: {p.Injectors.Count} simplex body(ies), "
                      + $"d_body {fDBody:0.##} mm, orifice {fBore:0.##} mm");
            return vox ?? new Voxels();
        }

        // -- micromix -----------------------------------------------------

        static Voxels voxMicromix(CombustorParameters p)
        {
            int n = p.Injectors.MicromixN > 0
                ? p.Injectors.MicromixN : Math.Max(8, 6 * p.Injectors.Count);
            float fDJet = p.Injectors.MicromixDMm > 0f
                ? p.Injectors.MicromixDMm : 0.8f;
            if (fDJet < CombustorParameters.D_HOLE_MIN_MM)
                p.Warnings.Add(
                    $"micromix jet d = {fDJet:0.###} mm is below the LPBF hole "
                    + $"floor {CombustorParameters.D_HOLE_MIN_MM:0.##} mm");
            float fT = p.fPrintableMm(p.Liner.TWallMm, "micromix plate");
            float fZ = Liner.fDomeSurfaceZ(p, Liner.fInjectorRingRMm(p));
            float fSign = Liner.fDomeSign(p);
            float fRing = Liner.fInjectorRingRMm(p);
            if (fRing <= 1e-3f)
                fRing = 0.6f * p.fLinerROuterAt(p.fDomeZMm());

            // one annular manifold plate carrying n jets
            float fWidth = MathF.Max(4f * fDJet, 4f);
            Voxels vox = LatticeUtils.voxCylinderZ(
                new Vector3(0f, 0f, fZ - 0.5f * fT), fRing + 0.5f * fWidth,
                fT);
            LatticeUtils.Sub(vox, LatticeUtils.voxCylinderZ(
                new Vector3(0f, 0f, fZ - 0.5f * fT - 1f),
                MathF.Max(fRing - 0.5f * fWidth, 0.5f), fT + 2f));
            for (int k = 0; k < n; k++)
            {
                float fPhi = 2f * MathF.PI * k / n;
                Vector3 vecC = new Vector3(fRing * MathF.Cos(fPhi),
                                           fRing * MathF.Sin(fPhi), fZ);
                LatticeUtils.Sub(vox, LatticeUtils.voxBeam(
                    vecC + new Vector3(0f, 0f, fSign * (fT + 2f)),
                    vecC - new Vector3(0f, 0f, fSign * (fT + 2f)),
                    0.5f * fDJet));
            }
            Library.Log($"Injector: micromix plate with {n} jets of "
                      + $"{fDJet:0.###} mm on r = {fRing:0.#} mm");
            return vox;
        }

        // -- shared sizing ------------------------------------------------

        static float fBodyDMm(CombustorParameters p)
        {
            if (p.Injectors.BodyDMm > 0f)
                return p.Injectors.BodyDMm;
            if (p.Swirler.DTipMm > 0f)
                return MathF.Max(0.8f * p.Swirler.DTipMm, 4f);
            return MathF.Max(6f, 0.2f * p.fLinerROuterAt(p.fDomeZMm()));
        }

        static float fBoreDMm(CombustorParameters p)
        {
            if (p.Injectors.BoreDMm > 0f)
                return p.Injectors.BoreDMm;
            return MathF.Max(0.35f * fBodyDMm(p),
                             CombustorParameters.D_HOLE_MIN_MM);
        }
    }
}
