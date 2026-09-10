//
// LatticeUtils.cs - minimal solid primitives built on top of PicoGK.
// No ShapeKernel dependency.
//
// Adapted from Phy-AC (AxialCompressorDesigner/LatticeUtils.cs): voxBox,
// voxCylinderZ, voxCylinderAlongAxis and voxRevolveZ are the sibling's
// recipes; the combustor adds
//
//   voxRevolveProfileZ   revolution straight off a contract polyline
//                        ([[z, r], ...]), so profile kinks land EXACTLY on
//                        band boundaries instead of being smeared by a
//                        uniform-t parameterization.
//   voxSweepPolyline     round-jointed sweep along a 3-D polyline - the
//                        vaporizer "walking stick" tube.
//   voxTeardropBetween   self-supporting teardrop hole (DfAM section 4.3 rule 1):
//                        the convex hull of the circle and a 45-degree apex,
//                        extruded along the hole axis as a watertight prism.
//
// All lengths are millimetres. Z is the machine axis.
//

using System;
using System.Collections.Generic;
using System.Numerics;
using PicoGK;

namespace CombustorDesigner
{
    public static class LatticeUtils
    {
        /// <summary>
        /// Solid axis-aligned box (centered at vecCenter) with full
        /// dimensions fX x fY x fZ. Voxelized from a watertight cube mesh -
        /// the result is a true solid, not a shell.
        /// </summary>
        public static Voxels voxBox(Vector3 vecCenter, float fX, float fY,
                                    float fZ)
        {
            Mesh msh = Utils.mshCreateCube(new Vector3(fX, fY, fZ), vecCenter);
            return new Voxels(msh);
        }

        /// <summary>
        /// Solid cylinder of length fL along +Z, radius fR, bottom face
        /// centred at vecBaseBottom. A single flat-capped lattice beam -
        /// a true solid, not a shell.
        /// </summary>
        public static Voxels voxCylinderZ(Vector3 vecBaseBottom, float fR,
                                          float fL)
        {
            Lattice lat = new Lattice();
            lat.AddBeam(vecBaseBottom, fR,
                        vecBaseBottom + new Vector3(0, 0, fL), fR,
                        false);
            return new Voxels(lat);
        }

        /// <summary>Solid cylinder along the local +Z of a LocalFrame,
        /// length fL, radius fR, base at the frame origin.</summary>
        public static Voxels voxCylinderAlongAxis(LocalFrame oFrame, float fR,
                                                  float fL)
        {
            Vector3 vecBase = oFrame.vecGetPosition();
            Vector3 vecZ = oFrame.vecGetLocalZ();

            Lattice lat = new Lattice();
            lat.AddBeam(vecBase, fR, vecBase + fL * vecZ, fR, false);
            return new Voxels(lat);
        }

        /// <summary>Solid beam (cylinder) between two arbitrary points.
        /// bRound = hemispherical caps (use it for swept tube joints so the
        /// segments weld without notches).</summary>
        public static Voxels voxBeam(Vector3 vecA, Vector3 vecB, float fR,
                                     bool bRound = false)
        {
            Lattice lat = new Lattice();
            lat.AddBeam(vecA, fR, vecB, fR, bRound);
            return new Voxels(lat);
        }

        /// <summary>
        /// Solid sweep of a constant-radius circular section along a 3-D
        /// polyline. Round caps at every joint so bends of any angle weld
        /// smoothly (the vaporizer elbow is 180 degrees). The end caps are
        /// round too - the tube is later opened by subtracting the inner
        /// sweep, which is extended past both ends.
        /// </summary>
        public static Voxels voxSweepPolyline(IReadOnlyList<Vector3> aPts,
                                              float fR, bool bRound = true)
        {
            Lattice lat = new Lattice();
            int nAdded = 0;
            for (int i = 0; i + 1 < aPts.Count; i++)
            {
                if ((aPts[i + 1] - aPts[i]).LengthSquared() < 1e-12f)
                    continue;
                lat.AddBeam(aPts[i], fR, aPts[i + 1], fR, bRound);
                nAdded++;
            }
            if (nAdded == 0)
                return new Voxels();
            return new Voxels(lat);
        }

        /// <summary>
        /// Solid of revolution about the Z axis. fnContour maps t in [0,1]
        /// to a (radius, z) point; the solid fills everything inside the
        /// revolved contour, built as a stack of tapered (conical) frustums
        /// so the surface follows the contour instead of terracing.
        /// fROffset is added to the contour radius. The contour must be
        /// monotonic in z.
        /// </summary>
        public static Voxels voxRevolveZ(Func<float, Vector2> fnContour,
                                         float fROffset = 0f, int nBands = 64)
        {
            Lattice lat = new Lattice();
            for (int i = 0; i < nBands; i++)
            {
                Vector2 a = fnContour(i / (float)nBands);
                Vector2 b = fnContour((i + 1) / (float)nBands);
                float fRa = a.X + fROffset;
                float fRb = b.X + fROffset;
                if (fRa <= 0.1f && fRb <= 0.1f)
                    continue;
                lat.AddBeam(new Vector3(0f, 0f, a.Y), MathF.Max(fRa, 0.05f),
                            new Vector3(0f, 0f, b.Y), MathF.Max(fRb, 0.05f),
                            false);
            }
            return new Voxels(lat);
        }

        /// <summary>
        /// Solid of revolution straight off a contract polyline. afProfile
        /// rows are [z_mm, ...] with the radius in column nRCol (1 for
        /// [z, r] casing lines, 2 for [z, r_inner, r_outer] liner lines).
        ///
        /// One conical frustum per polyline segment: every kink of the
        /// contract profile lands exactly on a band boundary. Segments with
        /// zero axial length are SKIPPED - a radius jump at constant z is a
        /// flat annular face and is produced automatically by the two
        /// neighbouring frustums.
        ///
        /// fROffset shifts the whole contour radially (wall offsets),
        /// fZExtendMm extends the first/last segment axially at constant
        /// radius (use it on the subtracted inner solid so the resulting
        /// shell is an OPEN tube, the Phy-AC casing-shell trick).
        /// Returns an empty field when nothing is left to revolve.
        /// </summary>
        public static Voxels voxRevolveProfileZ(float[][] afProfile,
                                                int nRCol,
                                                float fROffset = 0f,
                                                float fZExtendMm = 0f,
                                                float fRMinMm = 0.05f)
        {
            if (afProfile == null || afProfile.Length < 2)
                return new Voxels();

            var aZ = new List<float>();
            var aR = new List<float>();
            for (int i = 0; i < afProfile.Length; i++)
            {
                float[] afRow = afProfile[i];
                if (afRow == null || afRow.Length <= nRCol)
                    continue;
                aZ.Add(afRow[0]);
                aR.Add(afRow[nRCol] + fROffset);
            }
            if (aZ.Count < 2)
                return new Voxels();

            if (fZExtendMm > 0f)
            {
                float fSign = aZ[aZ.Count - 1] >= aZ[0] ? 1f : -1f;
                aZ.Insert(0, aZ[0] - fSign * fZExtendMm);
                aR.Insert(0, aR[0]);
                aZ.Add(aZ[aZ.Count - 1] + fSign * fZExtendMm);
                aR.Add(aR[aR.Count - 1]);
            }

            Lattice lat = new Lattice();
            int nBeams = 0;
            for (int i = 0; i + 1 < aZ.Count; i++)
            {
                if (MathF.Abs(aZ[i + 1] - aZ[i]) < 1e-6f)
                    continue;                  // flat annular face
                float fRa = aR[i], fRb = aR[i + 1];
                if (fRa <= fRMinMm && fRb <= fRMinMm)
                    continue;
                lat.AddBeam(new Vector3(0f, 0f, aZ[i]),
                            MathF.Max(fRa, fRMinMm),
                            new Vector3(0f, 0f, aZ[i + 1]),
                            MathF.Max(fRb, fRMinMm), false);
                nBeams++;
            }
            if (nBeams == 0)
                return new Voxels();
            return new Voxels(lat);
        }

        /// <summary>
        /// Watertight TEARDROP prism from vecA to vecB: the cross-section is
        /// the convex hull of a circle of radius fR and an apex at
        /// fR*sqrt(2) along vecUp, i.e. a roof whose two flanks sit at
        /// exactly 45 degrees - the self-supporting section EstadoDelArte
        /// section 4.3 rule 1 demands for any hole whose axis is closer than
        /// 45 degrees to the build horizontal.
        ///
        /// vecUp must be the BUILD-UP direction; its component along the
        /// hole axis is removed. If the axis is (nearly) parallel to the
        /// build direction the teardrop is meaningless and a plain cylinder
        /// is returned.
        /// </summary>
        public static Voxels voxTeardropBetween(Vector3 vecA, Vector3 vecB,
                                                float fR, Vector3 vecUp,
                                                int nArc = 28)
        {
            Vector3 vecAxis = vecB - vecA;
            float fLen = vecAxis.Length();
            if (fLen < 1e-6f || fR <= 0f)
                return new Voxels();
            vecAxis /= fLen;

            Vector3 vecV = vecUp - Vector3.Dot(vecUp, vecAxis) * vecAxis;
            if (vecV.Length() < 0.15f)         // axis within ~8.6 deg of up
                return voxBeam(vecA, vecB, fR);
            vecV = Vector3.Normalize(vecV);
            Vector3 vecU = Vector3.Cross(vecV, vecAxis);   // right-handed

            // Cross-section polygon, CCW seen from +axis: 270-degree arc
            // from 135 to 405 degrees, then the apex.
            int nPts = Math.Max(nArc, 8) + 1;
            var aSec = new Vector3[nPts];
            for (int i = 0; i < nPts - 1; i++)
            {
                float fA = (135f + 270f * i / (nPts - 2)) * MathF.PI / 180f;
                aSec[i] = fR * (MathF.Cos(fA) * vecU + MathF.Sin(fA) * vecV);
            }
            aSec[nPts - 1] = fR * MathF.Sqrt(2f) * vecV;   // 45-degree apex

            Mesh msh = new Mesh();
            var anBase = new int[nPts];
            var anTop = new int[nPts];
            for (int i = 0; i < nPts; i++)
            {
                Vector3 vecPB = vecA + aSec[i];
                Vector3 vecPT = vecB + aSec[i];
                anBase[i] = msh.nAddVertex(vecPB);
                anTop[i] = msh.nAddVertex(vecPT);
            }
            // side walls (outward normals with the CCW section above)
            for (int i = 0; i < nPts; i++)
            {
                int j = (i + 1) % nPts;
                msh.AddQuad(anBase[i], anBase[j], anTop[j], anTop[i]);
            }
            // caps: base normal -axis, top normal +axis (convex => fan)
            for (int i = 1; i + 1 < nPts; i++)
            {
                msh.nAddTriangle(anBase[0], anBase[i + 1], anBase[i]);
                msh.nAddTriangle(anTop[0], anTop[i], anTop[i + 1]);
            }
            return new Voxels(msh);
        }

        /// <summary>Clamped linear interpolation of a [[z, ...]] polyline:
        /// value of column nCol at axial station fZ. Rows are assumed
        /// sorted by z (the contract guarantees it; PhyCBImport sorts
        /// defensively anyway).</summary>
        public static float fInterpAt(float[][] afLine, float fZ, int nCol)
        {
            if (afLine == null || afLine.Length == 0)
                return 0f;
            if (afLine.Length == 1 || fZ <= afLine[0][0])
                return afLine[0][nCol];
            for (int i = 1; i < afLine.Length; i++)
            {
                if (fZ <= afLine[i][0])
                {
                    float fDz = afLine[i][0] - afLine[i - 1][0];
                    if (MathF.Abs(fDz) < 1e-9f)
                        return afLine[i][nCol];
                    float t = (fZ - afLine[i - 1][0]) / fDz;
                    return afLine[i - 1][nCol]
                         + t * (afLine[i][nCol] - afLine[i - 1][nCol]);
                }
            }
            return afLine[afLine.Length - 1][nCol];
        }

        /// <summary>Unit meridional tangent (dz, dr) of a [[z, ...]]
        /// polyline at fZ, column nCol. Used by the effusion field to tilt
        /// each hole away from the LOCAL wall, which is what the DfAM
        /// checker assumes.</summary>
        public static Vector2 vecTangentAt(float[][] afLine, float fZ,
                                           int nCol)
        {
            if (afLine == null || afLine.Length < 2)
                return new Vector2(1f, 0f);
            int nSeg = 0;
            for (int i = 1; i < afLine.Length; i++)
            {
                nSeg = i - 1;
                if (fZ <= afLine[i][0])
                    break;
            }
            float fDz = afLine[nSeg + 1][0] - afLine[nSeg][0];
            float fDr = afLine[nSeg + 1][nCol] - afLine[nSeg][nCol];
            float fN = MathF.Sqrt(fDz * fDz + fDr * fDr);
            if (fN < 1e-9f)
                return new Vector2(1f, 0f);
            return new Vector2(fDz / fN, fDr / fN);
        }

        /// <summary>Bounding box from two corners. Built with Include()
        /// rather than the two-vector constructor so it is immune to the
        /// by-ref/by-value ambiguity of the PicoGK signature.</summary>
        public static BBox3 oBBox(Vector3 vecMin, Vector3 vecMax)
        {
            BBox3 oBox = new BBox3();
            oBox.Include(vecMin);
            oBox.Include(vecMax);
            return oBox;
        }

        /// <summary>Ring of bolt holes on a bolt circle at axial station
        /// fZ0, drilled through a plate of thickness fT. Phy-AC's flange
        /// pattern, reused verbatim so bolted joints look the same in both
        /// machines.</summary>
        public static Voxels voxBoltCircle(float fRBolt, int nBolts,
                                           float fDBolt, float fZ0, float fT)
        {
            if (nBolts <= 0 || fDBolt <= 0f)
                return new Voxels();
            Lattice lat = new Lattice();
            for (int k = 0; k < nBolts; k++)
            {
                float fA = 2f * MathF.PI * k / nBolts;
                Vector3 vecB = new Vector3(fRBolt * MathF.Cos(fA),
                                           fRBolt * MathF.Sin(fA), fZ0 - 1f);
                lat.AddBeam(vecB, 0.5f * fDBolt,
                            vecB + new Vector3(0f, 0f, fT + 2f),
                            0.5f * fDBolt, false);
            }
            return new Voxels(lat);
        }
    }
}
