//
// Swirler.cs - the swirler: hub, shroud and n_vanes lofted vanes.
//
// The vane recipe is Phy-AC's solid blade loft, reduced to what a swirler
// needs: a closed "stadium" section (flat plate with rounded leading and
// trailing edges) is lofted between span ribs, each rib rotated by the
// SIGNED vane angle and wrapped onto the cylinder of its stacking radius
//
//   phi = phi0 + y'/r      (arc length preserved)
//   P   = (r cos phi, r sin phi, zCenter + x')
//
// so the printed vane is a true solid, watertight, with the real thickness.
// Vane thickness below two voxels is clamped and the clamp is LOGGED.
//
// beta_deg is the vane angle at the TIP measured from the machine axis
// (55 deg is the classic value for SN ~ 0.8). If the contract carries
// beta_hub_deg the twist is interpolated linearly along the span, which is
// how a free-vortex swirler is drawn; otherwise beta is constant.
//
// "radial" swirlers are built as straight vanes in the plane normal to Z,
// each running from the tip to the hub with the same tangential lean.
//

using System;
using System.Collections.Generic;
using System.Numerics;
using PicoGK;

namespace CombustorDesigner
{
    public static class Swirler
    {
        /// <summary>Interpolated ribs inserted between the hub and tip
        /// stations (the swirler span is short, so 6 ribs are plenty).</summary>
        const int nRibs = 7;
        /// <summary>Points around the closed stadium section.</summary>
        const int nSectionPts = 24;

        /// <summary>Axial station of the swirler: the contract value when
        /// given, otherwise the dome surface on the axis.</summary>
        public static float fZMm(CombustorParameters p)
        {
            if (!float.IsNaN(p.Swirler.ZMm))
                return p.Swirler.ZMm;
            return Liner.fDomeSurfaceZ(p, 0f);
        }

        /// <summary>Axial chord of a vane.</summary>
        public static float fChordMm(CombustorParameters p)
        {
            if (p.Swirler.ChordMm > 0f)
                return p.Swirler.ChordMm;
            float fSpan = 0.5f * MathF.Max(p.Swirler.DTipMm
                                           - p.Swirler.DHubMm, 1f);
            return MathF.Max(1.2f * fSpan, 4f);
        }

        public static Voxels voxSwirler(CombustorParameters p)
        {
            if (p.Swirler.NVanes <= 0)
                return new Voxels();
            if (p.bScaleCups())
                return voxScaleCups(p);

            string strType = (p.Swirler.Type ?? "axial").ToLowerInvariant();
            float fRHub = 0.5f * MathF.Max(p.Swirler.DHubMm, 0.5f);
            float fRTip = 0.5f * MathF.Max(p.Swirler.DTipMm,
                                           p.Swirler.DHubMm + 1f);
            float fZ = fZMm(p);
            float fChord = fChordMm(p);
            float fVaneT = p.fPrintableMm(p.Swirler.VaneTMm, "swirler vane");

            // hub: a solid stub the vanes root into
            Voxels vox = LatticeUtils.voxCylinderZ(
                new Vector3(0f, 0f, fZ - 0.5f * fChord), fRHub, fChord);

            // vanes
            Voxels voxVanes = strType.StartsWith("rad")
                ? voxRadialVanes(p, fRHub, fRTip, fZ, fChord, fVaneT)
                : voxAxialVanes(p, fRHub, fRTip, fZ, fChord, fVaneT);
            if (voxVanes != null && !voxVanes.bIsEmpty())
                vox += voxVanes;

            // shroud ring: welds the vane tips together so the part is one
            // rigid piece (and gives the dome bore something to seat on)
            float fShroud = p.fPrintableMm(p.Liner.TWallMm, "swirler shroud");
            Voxels voxShroud = LatticeUtils.voxCylinderZ(
                new Vector3(0f, 0f, fZ - 0.5f * fChord), fRTip + fShroud,
                fChord);
            voxShroud -= LatticeUtils.voxCylinderZ(
                new Vector3(0f, 0f, fZ - 0.5f * fChord - 1f), fRTip,
                fChord + 2f);
            vox += voxShroud;

            Library.Log($"Swirler: {p.Swirler.NVanes} {strType} vanes, beta "
                      + $"{p.Swirler.BetaDeg:0.#} deg (SN {p.Swirler.SN:0.##}), "
                      + $"d_hub {p.Swirler.DHubMm:0.#} -> d_tip "
                      + $"{p.Swirler.DTipMm:0.#} mm, t {fVaneT:0.##} mm at "
                      + $"z = {fZ:0.#} mm");
            return vox;
        }

        // -- swirl-cup ring (annular_5_25, 17.1-03 / D-386) ---------------

        /// <summary>
        /// One swirl cup per injector station, identical in the LOCAL frame
        /// of its station (the vane pattern turns with the station phase):
        /// hub of d_hub bored for the injector seat, n_vanes axial vanes of
        /// the contract chord and angle, and a shroud tube of d_tip that
        /// runs through the whole dome-wall crossing so all the cup air goes
        /// through the vanes. Every cup carries the contract per-cup
        /// dimensions - the area per cup is A_sw_each_mm2, never the total
        /// divided again. Axis parallel to Z (vertical walls on the plate).
        /// </summary>
        public static Voxels voxScaleCups(CombustorParameters p)
        {
            float fRHub = 0.5f * p.Swirler.DHubMm;
            float fRTip = 0.5f * p.Swirler.DTipMm;
            float fChord = fChordMm(p);
            float fVaneT = p.fPrintableMm(p.Swirler.VaneTMm, "swirler vane");
            float fShroud = p.fPrintableMm(p.Liner.TWallMm, "swirler shroud");
            float fRSeat = 0.5f * Liner.fSeatDMm(p);

            Voxels vox = null;
            var aSt = Liner.aCupStations(p);
            foreach (Liner.CupStation st in aSt)
            {
                float fCx = st.vecC.X, fCy = st.vecC.Y, fZ = st.vecC.Z;
                float fPhiSt = st.fPhiDeg * MathF.PI / 180f;

                // hub, bored for the injector
                Voxels voxCup = LatticeUtils.voxCylinderZ(
                    new Vector3(fCx, fCy, fZ - 0.5f * fChord), fRHub, fChord);
                // vanes in the local frame of the station
                Voxels voxVanes = voxAxialVanes(p, fRHub, fRTip, fZ, fChord,
                                                fVaneT, fCx, fCy, fPhiSt);
                if (voxVanes != null && !voxVanes.bIsEmpty())
                    voxCup += voxVanes;
                voxCup -= LatticeUtils.voxCylinderZ(
                    new Vector3(fCx, fCy, fZ - 0.5f * fChord - 1f), fRSeat,
                    fChord + 2f);

                // shroud tube over the full station span
                Voxels voxShroud = LatticeUtils.voxCylinderZ(
                    new Vector3(fCx, fCy, st.fZLo), fRTip + fShroud,
                    st.fZHi - st.fZLo);
                voxShroud -= LatticeUtils.voxCylinderZ(
                    new Vector3(fCx, fCy, st.fZLo - 1f), fRTip,
                    st.fZHi - st.fZLo + 2f);
                voxCup += voxShroud;
                vox = vox == null ? voxCup : vox + voxCup;
            }

            Library.Log($"Swirler: {aSt.Count} swirl cups (annular_5_25) on "
                      + $"r = {Liner.fInjectorRingRMm(p):0.###} mm, each "
                      + $"{p.Swirler.NVanes} axial vanes, beta "
                      + $"{p.Swirler.BetaDeg:0.#} deg, d_hub "
                      + $"{p.Swirler.DHubMm:0.###} -> d_tip "
                      + $"{p.Swirler.DTipMm:0.###} mm (A per cup "
                      + $"{p.Swirler.AEachMm2:0.###} mm2), chord "
                      + $"{fChord:0.###} mm, vane t {fVaneT:0.##} mm, shroud "
                      + $"t {fShroud:0.##} mm, injector bore r {fRSeat:0.###} mm");
            return vox ?? new Voxels();
        }

        // -- axial swirler ------------------------------------------------

        struct Rib
        {
            public float fR;
            public float fStaggerDeg;
            public float[][] afPts;      // closed section, chord frame
        }

        static Voxels voxAxialVanes(CombustorParameters p, float fRHub,
                                    float fRTip, float fZ, float fChord,
                                    float fVaneT)
            => voxAxialVanes(p, fRHub, fRTip, fZ, fChord, fVaneT,
                             0f, 0f, 0f);

        /// <summary>Axial vanes around a local axis parallel to Z through
        /// (fCx, fCy), the pattern rotated by fPhiBase (0, 0, 0 = the
        /// historical swirler on the machine axis).</summary>
        static Voxels voxAxialVanes(CombustorParameters p, float fRHub,
                                    float fRTip, float fZ, float fChord,
                                    float fVaneT, float fCx, float fCy,
                                    float fPhiBase)
        {
            float fBetaTip = p.Swirler.BetaDeg;
            float fBetaHub = float.IsNaN(p.Swirler.BetaHubDeg)
                ? fBetaTip : p.Swirler.BetaHubDeg;

            // the root sinks into the hub, the tip into the shroud, so the
            // vane welds instead of merely touching
            float fSink = MathF.Max(2f * p.VoxelSizeMm, 0.3f);
            var aRibs = new Rib[nRibs];
            for (int i = 0; i < nRibs; i++)
            {
                float t = i / (float)(nRibs - 1);
                float fR = fRHub + (fRTip - fRHub) * t;
                if (i == 0) fR -= fSink;
                if (i == nRibs - 1) fR += fSink;
                aRibs[i] = new Rib
                {
                    fR = MathF.Max(fR, 0.5f),
                    fStaggerDeg = fBetaHub + (fBetaTip - fBetaHub) * t,
                    afPts = afStadium(fChord, fVaneT, nSectionPts),
                };
            }

            Mesh msh = new Mesh();
            for (int k = 0; k < p.Swirler.NVanes; k++)
            {
                float fPhi0 = fPhiBase + 2f * MathF.PI * k / p.Swirler.NVanes;
                AddVaneSolid(msh, aRibs, fZ, fPhi0, fCx, fCy);
            }
            return new Voxels(msh);
        }

        /// <summary>Closed "stadium" section in the chord frame: a flat
        /// plate of chord fC and thickness fT with semicircular LE/TE,
        /// centred on the origin, CCW.</summary>
        static float[][] afStadium(float fC, float fT, int nPts)
        {
            float fRe = 0.5f * fT;
            float fHalf = MathF.Max(0.5f * fC - fRe, 0.1f);
            int nCap = Math.Max(3, nPts / 2 - 1);
            var aPts = new List<float[]>();
            // trailing edge cap (+x), from -y to +y
            for (int i = 0; i <= nCap; i++)
            {
                float fA = -0.5f * MathF.PI + MathF.PI * i / nCap;
                aPts.Add(new float[]
                {
                    fHalf + fRe * MathF.Cos(fA),
                    fRe * MathF.Sin(fA),
                });
            }
            // leading edge cap (-x), from +y to -y
            for (int i = 0; i <= nCap; i++)
            {
                float fA = 0.5f * MathF.PI + MathF.PI * i / nCap;
                aPts.Add(new float[]
                {
                    -fHalf + fRe * MathF.Cos(fA),
                    fRe * MathF.Sin(fA),
                });
            }
            return aPts.ToArray();
        }

        /// <summary>Closed watertight vane: ruled side walls around the
        /// closed contour + hub/tip caps. Phy-AC's AddBladeSolid, with the
        /// same LE/TE zipper cap so the triangulation never crosses.</summary>
        static void AddVaneSolid(Mesh msh, Rib[] aRibs, float fZCenter,
                                 float fPhi0, float fCx = 0f, float fCy = 0f)
        {
            int nR = aRibs.Length;
            int nPts = aRibs[0].afPts.Length;
            var an = new int[nR, nPts];

            for (int i = 0; i < nR; i++)
            {
                Rib rib = aRibs[i];
                float fR = MathF.Max(rib.fR, 0.5f);
                float fG = rib.fStaggerDeg * MathF.PI / 180f;
                float fCosG = MathF.Cos(fG), fSinG = MathF.Sin(fG);
                for (int j = 0; j < nPts; j++)
                {
                    float x = rib.afPts[j][0];
                    float y = rib.afPts[j][1];
                    float xr = x * fCosG - y * fSinG;    // axial (staggered)
                    float yr = x * fSinG + y * fCosG;    // tangential
                    float fPhi = fPhi0 + yr / fR;        // wrap on cylinder
                    an[i, j] = msh.nAddVertex(new Vector3(
                        fCx + fR * MathF.Cos(fPhi), fCy + fR * MathF.Sin(fPhi),
                        fZCenter + xr));
                }
            }

            for (int i = 0; i < nR - 1; i++)
                for (int j = 0; j < nPts; j++)
                {
                    int j1 = (j + 1) % nPts;
                    msh.AddQuad(an[i, j], an[i + 1, j],
                                an[i + 1, j1], an[i, j1]);
                }
            AddCap(msh, aRibs[0], an, 0, nPts);
            AddCap(msh, aRibs[nR - 1], an, nR - 1, nPts);
        }

        /// <summary>Triangulates one rib polygon by zipping the two
        /// x-monotone chains between LE and TE - no assumption on the start
        /// index or the winding. Phy-AC's BladeRow.AddCap.</summary>
        static void AddCap(Mesh msh, Rib rib, int[,] an, int iRib, int nPts)
        {
            int iLe = 0, iTe = 0;
            for (int j = 1; j < nPts; j++)
            {
                if (rib.afPts[j][0] < rib.afPts[iLe][0]) iLe = j;
                if (rib.afPts[j][0] > rib.afPts[iTe][0]) iTe = j;
            }
            int nA = (iTe - iLe + nPts) % nPts + 1;
            int nB = (iLe - iTe + nPts) % nPts + 1;
            var aA = new int[nA];
            var aB = new int[nB];
            for (int k = 0; k < nA; k++) aA[k] = (iLe + k) % nPts;
            for (int k = 0; k < nB; k++) aB[k] = (iLe - k + nPts) % nPts;

            int a = 0, b = 0;
            while (a < nA - 1 || b < nB - 1)
            {
                bool bAdvanceA =
                    b == nB - 1
                    || (a < nA - 1
                        && rib.afPts[aA[a + 1]][0] <= rib.afPts[aB[b + 1]][0]);
                if (bAdvanceA)
                {
                    AddTri(msh, an[iRib, aA[a]], an[iRib, aA[a + 1]],
                           an[iRib, aB[b]]);
                    a++;
                }
                else
                {
                    AddTri(msh, an[iRib, aA[a]], an[iRib, aB[b + 1]],
                           an[iRib, aB[b]]);
                    b++;
                }
            }
        }

        static void AddTri(Mesh msh, int n0, int n1, int n2)
        {
            if (n0 == n1 || n1 == n2 || n0 == n2)
                return;                       // degenerate at LE/TE
            msh.nAddTriangle(n0, n1, n2);
        }

        // -- radial swirler -----------------------------------------------

        /// <summary>Radial (cyclone) swirler: straight vanes in the plane
        /// normal to Z, leaning by beta from the radial direction, each
        /// running from the tip radius inwards. Built as flat prisms of
        /// height = the axial chord.</summary>
        static Voxels voxRadialVanes(CombustorParameters p, float fRHub,
                                     float fRTip, float fZ, float fChord,
                                     float fVaneT)
        {
            float fBeta = p.Swirler.BetaDeg * MathF.PI / 180f;
            Voxels vox = null;
            for (int k = 0; k < p.Swirler.NVanes; k++)
            {
                float fPhi = 2f * MathF.PI * k / p.Swirler.NVanes;
                // outer point on the tip circle, inner point rotated by the
                // lean so the chord makes beta with the radial direction
                float fDPhi = MathF.Tan(fBeta)
                            * MathF.Log(MathF.Max(fRTip / fRHub, 1.001f));
                Vector3 vecOut = new Vector3(fRTip * MathF.Cos(fPhi),
                                             fRTip * MathF.Sin(fPhi), fZ);
                Vector3 vecIn = new Vector3(
                    fRHub * MathF.Cos(fPhi + fDPhi),
                    fRHub * MathF.Sin(fPhi + fDPhi), fZ);
                Voxels voxV = voxFlatVane(vecOut, vecIn, fVaneT, fChord);
                vox = vox == null ? voxV : vox + voxV;
            }
            return vox ?? new Voxels();
        }

        /// <summary>Flat rectangular vane spanning vecA-vecB, thickness fT
        /// in the in-plane normal direction, height fH along Z.</summary>
        static Voxels voxFlatVane(Vector3 vecA, Vector3 vecB, float fT,
                                  float fH)
        {
            Vector3 vecDir = vecB - vecA;
            float fLen = vecDir.Length();
            if (fLen < 1e-6f)
                return new Voxels();
            vecDir /= fLen;
            Vector3 vecN = new Vector3(-vecDir.Y, vecDir.X, 0f);
            if (vecN.Length() < 1e-6f)
                vecN = Vector3.UnitY;
            vecN = Vector3.Normalize(vecN) * (0.5f * fT);
            Vector3 vecUp = new Vector3(0f, 0f, 0.5f * fH);

            Mesh msh = new Mesh();
            Vector3[] aCorner =
            {
                vecA - vecN - vecUp, vecB - vecN - vecUp,
                vecB + vecN - vecUp, vecA + vecN - vecUp,
                vecA - vecN + vecUp, vecB - vecN + vecUp,
                vecB + vecN + vecUp, vecA + vecN + vecUp,
            };
            var an = new int[8];
            for (int i = 0; i < 8; i++)
                an[i] = msh.nAddVertex(aCorner[i]);
            msh.AddQuad(an[3], an[2], an[1], an[0]);   // bottom (-Z)
            msh.AddQuad(an[4], an[5], an[6], an[7]);   // top (+Z)
            msh.AddQuad(an[0], an[1], an[5], an[4]);
            msh.AddQuad(an[1], an[2], an[6], an[5]);
            msh.AddQuad(an[2], an[3], an[7], an[6]);
            msh.AddQuad(an[3], an[0], an[4], an[7]);
            return new Voxels(msh);
        }
    }
}
