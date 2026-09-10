//
// Casing.cs - the pressure casing: revolved shell, bolted end flanges,
// powder drain ports and the igniter boss.
//
// The flange pattern is Phy-AC's, reused verbatim (ring + annular relief +
// bolt circle) so a bolted joint looks the same in both machines.
//
// The drain ports are NOT decoration: EstadoDelArte 4.4 calls trapped
// powder "the most underestimated constraint" - powder left in the
// liner/casing annulus sinters during HIP and blocks it permanently. Each
// declared port becomes a real radial through-hole with a boss, and
// DfamVoxelCheck then proves on the voxel field that the cavity it serves
// really does break out of the machine in at least two places.
//

using System;
using System.Collections.Generic;
using System.Numerics;
using PicoGK;

namespace CombustorDesigner
{
    public static class Casing
    {
        /// <summary>Axial lip beyond the ends of the casing profile.</summary>
        internal const float fLipMm = 2f;

        public static Voxels voxCasing(CombustorParameters p)
        {
            // clamped once here; every sub-builder below uses the quiet
            // clamp so the same wall is not reported half a dozen times
            float fT = p.fPrintableMm(p.Casing.TWallMm, "casing wall");

            // shell: outer revolve minus the inner revolve, opened at both
            // ends by extending the inner solid axially
            Voxels vox = LatticeUtils.voxRevolveProfileZ(p.Casing.Profile, 1,
                                                         fT);
            vox -= LatticeUtils.voxRevolveProfileZ(p.Casing.Profile, 1, 0f,
                                                   fLipMm + 2f);

            float fZ0 = p.Casing.Profile[0][0];
            float fZ1 = p.Casing.Profile[p.Casing.Profile.Length - 1][0];

            // bolted end flanges
            vox += voxFlange(p, fZ0, bForward: true);
            vox += voxFlange(p, fZ1, bForward: false);

            // optional head-end closure plate (only when declared: a can
            // casing that feeds through the head annulus has none, and
            // inventing one would falsify the drainability verdict)
            if (p.Casing.HeadPlateTMm > 0f)
                vox += voxHeadPlate(p, fT);

            // igniter boss + through bore
            foreach (HoleRowParams row in p.HoleRows)
            {
                if (!row.bIsIgniter())
                    continue;
                float fDBoss = MathF.Max(2.5f * row.DMm, row.DMm + 6f);
                vox += voxRadialBoss(p, row.ZMm, row.Phi0Deg, fDBoss);
                vox -= voxRadialBore(p, row.ZMm, row.Phi0Deg, row.DMm);
                Library.Log($"Casing: igniter boss at z = {row.ZMm:0.#} mm, "
                          + $"bore d = {row.DMm:0.##} mm");
            }

            // powder drain ports
            var aPorts = aDrainPorts(p);
            for (int i = 0; i < aPorts.Count; i++)
            {
                DrainPortParams o = aPorts[i];
                if (o.bBoss)
                    vox += voxRadialBoss(p, o.ZMm, o.PhiDeg,
                                         MathF.Max(2f * o.DMm, o.DMm + 5f));
                vox -= voxRadialBore(p, o.ZMm, o.PhiDeg, o.DMm);
                Library.Log($"Casing: drain port {i + 1}/{aPorts.Count} at "
                          + $"z = {o.ZMm:0.#} mm, phi = {o.PhiDeg:0.#} deg, "
                          + $"d = {o.DMm:0.##} mm");
            }
            return vox;
        }

        /// <summary>The declared drain ports with a resolved angular
        /// position: ports that do not declare `phi_deg` are spread evenly
        /// so two of them never end up on the same generatrix (which would
        /// leave the far side of the annulus with no line of sight).</summary>
        public static List<DrainPortParams> aDrainPorts(CombustorParameters p)
        {
            var aOut = new List<DrainPortParams>();
            int n = p.Dfam.DrainPorts.Count;
            for (int i = 0; i < n; i++)
            {
                DrainPortParams o = p.Dfam.DrainPorts[i];
                float fPhi = o.PhiDeg;
                if (float.IsNaN(fPhi))
                    fPhi = 180f * i;              // 0, 180, 360=0, ...
                aOut.Add(new DrainPortParams
                {
                    ZMm = o.ZMm,
                    DMm = o.DMm > 0f ? o.DMm : 5f,
                    PhiDeg = fPhi,
                    bBoss = o.bBoss,
                });
            }
            return aOut;
        }

        /// <summary>Bolted flange ring at axial station fZ (thickness towards
        /// +Z if bForward). Phy-AC's pattern: ring, annular relief at the
        /// bore, bolt circle at mid-width.</summary>
        public static Voxels voxFlange(CombustorParameters p, float fZ,
                                       bool bForward)
        {
            float fT = p.fPrintableQuietMm(p.Casing.TWallMm);
            float fFT = p.fPrintableMm(p.Casing.FlangeTMm, "casing flange");
            float fRIn = p.fCasingRAt(fZ);
            float fRO = fRIn + fT;
            float fRF = fRO + MathF.Max(p.Casing.FlangeWMm,
                                        p.Casing.FlangeBoltDMm * 2f);
            float fZBase = bForward ? fZ : fZ - fFT;

            Voxels vox = LatticeUtils.voxCylinderZ(
                new Vector3(0f, 0f, fZBase), fRF, fFT);
            vox -= LatticeUtils.voxCylinderZ(
                new Vector3(0f, 0f, fZBase - 1f), fRIn, fFT + 2f);
            float fRBolt = fRO + 0.5f * (fRF - fRO);
            vox -= LatticeUtils.voxBoltCircle(fRBolt, p.Casing.FlangeBoltCount,
                                              p.Casing.FlangeBoltDMm,
                                              fZBase, fFT);
            return vox;
        }

        /// <summary>Head-end closure plate (only when the contract declares
        /// `casing.head_plate_t_mm`). Bored for the injector feed.</summary>
        static Voxels voxHeadPlate(CombustorParameters p, float fT)
        {
            float fTP = p.fPrintableMm(p.Casing.HeadPlateTMm,
                                       "casing head plate");
            float fZ0 = p.Casing.Profile[0][0];
            float fR = p.fCasingRAt(fZ0) + fT;
            Voxels vox = LatticeUtils.voxCylinderZ(
                new Vector3(0f, 0f, fZ0 - fTP), fR, fTP);
            float fBore = MathF.Max(0.5f * Liner.fSeatDMm(p), 2f);
            foreach (Vector3 vecPos in Liner.aInjectorPositions(p))
                vox -= LatticeUtils.voxCylinderZ(
                    new Vector3(vecPos.X, vecPos.Y, fZ0 - fTP - 1f),
                    fBore, fTP + 2f);
            return vox;
        }

        /// <summary>Radial boss standing off the casing at (z, phi).</summary>
        public static Voxels voxRadialBoss(CombustorParameters p, float fZ,
                                           float fPhiDeg, float fDBoss)
        {
            float fT = p.fPrintableQuietMm(p.Casing.TWallMm);
            float fPhi = fPhiDeg * MathF.PI / 180f;
            float fRIn = p.fCasingRAt(fZ);
            float fROut = fRIn + fT + MathF.Max(p.Casing.BossHMm, 1f);
            Vector3 vecDir = new Vector3(MathF.Cos(fPhi), MathF.Sin(fPhi), 0f);
            return LatticeUtils.voxBeam(
                vecDir * MathF.Max(fRIn - 1f, 0.5f) + new Vector3(0f, 0f, fZ),
                vecDir * fROut + new Vector3(0f, 0f, fZ),
                0.5f * fDBoss);
        }

        /// <summary>Radial through bore at (z, phi), long enough to cut the
        /// wall and the boss.</summary>
        public static Voxels voxRadialBore(CombustorParameters p, float fZ,
                                           float fPhiDeg, float fD)
        {
            float fT = p.fPrintableQuietMm(p.Casing.TWallMm);
            float fPhi = fPhiDeg * MathF.PI / 180f;
            float fRIn = p.fCasingRAt(fZ);
            float fROut = fRIn + fT + p.Casing.BossHMm + 3f;
            Vector3 vecDir = new Vector3(MathF.Cos(fPhi), MathF.Sin(fPhi), 0f);
            return LatticeUtils.voxBeam(
                vecDir * MathF.Max(fRIn - 3f, 0.1f) + new Vector3(0f, 0f, fZ),
                vecDir * fROut + new Vector3(0f, 0f, fZ),
                0.5f * fD);
        }

        /// <summary>Point just inside the casing bore of a drain port - the
        /// SEED used to tell which cavity that port serves. Sits a little
        /// inboard of the wall so it lands in the annulus, not in the
        /// wall.</summary>
        public static Vector3 vecPortSeed(CombustorParameters p,
                                          DrainPortParams o)
        {
            float fPhi = o.PhiDeg * MathF.PI / 180f;
            float fRIn = p.fCasingRAt(o.ZMm);
            float fR = MathF.Max(fRIn - MathF.Max(0.6f * o.DMm, 1.5f), 0.1f);
            return new Vector3(fR * MathF.Cos(fPhi), fR * MathF.Sin(fPhi),
                               o.ZMm);
        }

        /// <summary>Outer radius of the casing at fZ: the envelope of the
        /// whole machine, which bounds the powder domain. Uses the CLAMPED
        /// wall so the envelope matches the metal that was actually built at
        /// this voxel size.</summary>
        public static float fROuterAt(CombustorParameters p, float fZ)
            => p.fCasingRAt(fZ) + p.fPrintableQuietMm(p.Casing.TWallMm);
    }
}
