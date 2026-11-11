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
// D-392 (17.1-03): a lattice beam is rendered over its whole bounding box,
// padded by the radius in all three axes, so a revolution or cylinder of a
// few hundred mm costs 10^8-10^10 voxel evaluations at 0.8 mm. Above
// dLatticeBudgetVoxels voxCylinderZ and voxRevolveProfileZ build the same
// solid as a watertight revolved mesh (mshRevolveZ) instead; below it the
// historical lattice is kept bit for bit. Add/Sub do the booleans in place
// and dispose the tool so no native grid waits for the finalizer.
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
            return voxFrom(msh);
        }

        /// <summary>
        /// Solid cylinder of length fL along +Z, radius fR, bottom face
        /// centred at vecBaseBottom. A single flat-capped lattice beam -
        /// a true solid, not a shell.
        /// </summary>
        public static Voxels voxCylinderZ(Vector3 vecBaseBottom, float fR,
                                          float fL)
        {
            // 17.1-03 / D-392: PicoGK renders a beam by evaluating its SDF
            // over the whole beam bounding box, which carries +-r in Z too:
            // a flange of r = 280 mm and 4 mm thick costs a 560^3 mm box.
            // Beyond dLatticeBudgetVoxels the cylinder is built as a
            // watertight revolved MESH (voxelized over its surface only;
            // same solid, polygon sag under fRevolveSagMm). Below it the
            // historical lattice beam is kept bit for bit (KJ66, cups).
            if (dBeamBoxVoxels(vecBaseBottom.Z, fR, vecBaseBottom.Z + fL, fR)
                <= dLatticeBudgetVoxels)
            {
                Lattice lat = new Lattice();
                lat.AddBeam(vecBaseBottom, fR,
                            vecBaseBottom + new Vector3(0, 0, fL), fR,
                            false);
                return voxFrom(lat);
            }
            if (MathF.Abs(fL) < 1e-6f || fR <= 0f)
                return new Voxels();
            Mesh msh = mshRevolveZ(
                new[] { vecBaseBottom.Z, vecBaseBottom.Z + fL },
                new[] { fR, fR }, vecBaseBottom.X, vecBaseBottom.Y);
            Voxels vox = new Voxels(msh);
            msh.Dispose();
            return vox;
        }

        /// <summary>
        /// D-392: runs every pending finalizer NOW, between native calls.
        /// PicoGK objects that reach the finalizer are destroyed on the
        /// finalizer thread, concurrently with whatever the build thread is
        /// doing inside the (single-threaded by design) native library;
        /// the 8 kg/s builds died twice with 0xC0000374 (heap corruption)
        /// in the middle of long native operations. Every transient object
        /// is now disposed explicitly; this flush drains whatever is left
        /// at the safe points (before each part, the fillet, the effusion,
        /// the assembly and the DfAM pass).
        /// </summary>
        public static void FlushFinalizers()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        /// <summary>Voxelizes a lattice and releases it at once (D-392:
        /// no native object is left to the finalizer thread).</summary>
        public static Voxels voxFrom(Lattice lat)
        {
            Voxels vox = new Voxels(lat);
            lat.Dispose();
            return vox;
        }

        /// <summary>Voxelizes a mesh and releases it at once.</summary>
        public static Voxels voxFrom(Mesh msh)
        {
            Voxels vox = new Voxels(msh);
            msh.Dispose();
            return vox;
        }

        /// <summary>Voxel size of the running build (mm); set once by
        /// CombustorBuilder before any part is built. Only used to price a
        /// lattice beam against dLatticeBudgetVoxels.</summary>
        public static float fVoxelSizeMm = 0.8f;

        /// <summary>
        /// D-392: largest lattice render (voxels evaluated, summed over the
        /// beams of one solid) still built as a lattice. Every KJ66 solid
        /// and every cup of the scale class stays well under it (at most ~12 M at
        /// 1.2 mm), so they keep the historical geometry bit for bit; the
        /// liner, dome and casing revolutions of the 5-25 kg/s class at 0.8
        /// mm cost 10^8-10^10 and switch to the revolved mesh.
        /// </summary>
        public const double dLatticeBudgetVoxels = 5e7;

        /// <summary>Voxels PicoGK evaluates to render one flat-capped beam
        /// along Z: its bounding box, which is padded by the radius in all
        /// three axes.</summary>
        public static double dBeamBoxVoxels(float fZa, float fRa, float fZb,
                                            float fRb)
        {
            double dR = Math.Max(Math.Max(fRa, fRb), 0f);
            double dS = 2.0 * dR;
            double dV = Math.Max(fVoxelSizeMm, 1e-3f);
            return dS * dS * (Math.Abs(fZb - fZa) + dS) / (dV * dV * dV);
        }

        /// <summary>In-place union vox += voxTool; the tool is disposed
        /// (D-392: no native grid is left to the finalizer). Returns vox.
        /// </summary>
        public static Voxels Add(Voxels vox, Voxels voxTool)
        {
            if (voxTool == null)
                return vox;
            if (!voxTool.bIsEmpty())
                vox.BoolAdd(voxTool);
            voxTool.Dispose();
            return vox;
        }

        /// <summary>In-place difference vox -= voxTool; the tool is
        /// disposed. Returns vox.</summary>
        public static Voxels Sub(Voxels vox, Voxels voxTool)
        {
            if (voxTool == null)
                return vox;
            if (!voxTool.bIsEmpty())
                vox.BoolSubtract(voxTool);
            voxTool.Dispose();
            return vox;
        }

        /// <summary>Largest chord sag (mm) the polygonal revolutions may
        /// show against the true circle: two orders of magnitude under the
        /// finest voxel used (0.3 mm), so the voxel field cannot tell the
        /// polygon from the circle.</summary>
        public const float fRevolveSagMm = 0.005f;

        /// <summary>Angular segments of a revolved mesh whose largest radius
        /// is fRMaxMm, so the inscribed polygon stays within fRevolveSagMm
        /// of the circle. Multiple of 8 (symmetric about both axes).</summary>
        public static int nRevolveSegments(float fRMaxMm)
        {
            if (!(fRMaxMm > fRevolveSagMm))
                return 48;
            double dHalf = Math.Acos(1.0 - fRevolveSagMm / fRMaxMm);
            int n = (int)Math.Ceiling(Math.PI / dHalf);
            n = Math.Clamp(n, 48, 4096);
            return (n + 7) / 8 * 8;
        }

        /// <summary>
        /// Watertight solid of revolution about the axis parallel to Z
        /// through (fCx, fCy): the meridian polyline (aZ[i], aR[i]), z
        /// ascending and r > 0, revolved into rings joined by quads, with a
        /// flat disc cap at each end. A radius step at constant z becomes
        /// the flat annular face it represents. Outward winding (CCW seen
        /// from outside).
        /// </summary>
        public static Mesh mshRevolveZ(IReadOnlyList<float> aZ,
                                       IReadOnlyList<float> aR,
                                       float fCx = 0f, float fCy = 0f)
        {
            int nRings = aZ.Count;
            float fRMax = 0f;
            for (int i = 0; i < nRings; i++)
                fRMax = MathF.Max(fRMax, aR[i]);
            int nSeg = nRevolveSegments(fRMax);
            var afC = new float[nSeg];
            var afS = new float[nSeg];
            for (int k = 0; k < nSeg; k++)
            {
                double dA = 2.0 * Math.PI * k / nSeg;
                afC[k] = (float)Math.Cos(dA);
                afS[k] = (float)Math.Sin(dA);
            }

            Mesh msh = new Mesh();
            var anRing = new int[nRings][];
            for (int i = 0; i < nRings; i++)
            {
                anRing[i] = new int[nSeg];
                for (int k = 0; k < nSeg; k++)
                    anRing[i][k] = msh.nAddVertex(new Vector3(
                        fCx + aR[i] * afC[k], fCy + aR[i] * afS[k], aZ[i]));
            }
            // side walls: (a, b, c) with a->b tangential (+phi) and b->c
            // meridional (+z or radial) gives the outward normal
            for (int i = 0; i + 1 < nRings; i++)
            {
                int[] a0 = anRing[i], a1 = anRing[i + 1];
                for (int k = 0; k < nSeg; k++)
                {
                    int k1 = (k + 1) % nSeg;
                    msh.nAddTriangle(a0[k], a0[k1], a1[k1]);
                    msh.nAddTriangle(a0[k], a1[k1], a1[k]);
                }
            }
            // end caps: bottom faces -Z, top faces +Z
            int nBot = msh.nAddVertex(new Vector3(fCx, fCy, aZ[0]));
            int nTop = msh.nAddVertex(new Vector3(fCx, fCy, aZ[nRings - 1]));
            int[] aB = anRing[0], aT = anRing[nRings - 1];
            for (int k = 0; k < nSeg; k++)
            {
                int k1 = (k + 1) % nSeg;
                msh.nAddTriangle(nBot, aB[k1], aB[k]);
                msh.nAddTriangle(nTop, aT[k], aT[k1]);
            }
            return msh;
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
            return voxFrom(lat);
        }

        /// <summary>Solid beam (cylinder) between two arbitrary points.
        /// bRound = hemispherical caps (use it for swept tube joints so the
        /// segments weld without notches).</summary>
        public static Voxels voxBeam(Vector3 vecA, Vector3 vecB, float fR,
                                     bool bRound = false)
        {
            Lattice lat = new Lattice();
            lat.AddBeam(vecA, fR, vecB, fR, bRound);
            return voxFrom(lat);
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
            return voxFrom(lat);
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
            return voxFrom(lat);
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

            // 17.1-03 / D-392: one lattice beam per segment costs a box of
            // +-r in all three axes per beam (the 8 kg/s liner walls took
            // 98 s and the 48-segment dome 246 s at 0.8 mm). Under
            // dLatticeBudgetVoxels the historical lattice is kept bit for
            // bit; above it the SAME union of conical frustums is built as
            // ONE watertight revolved mesh, voxelized over its surface.
            int nBeams = 0;
            double dCost = 0.0;
            for (int i = 0; i + 1 < aZ.Count; i++)
            {
                if (MathF.Abs(aZ[i + 1] - aZ[i]) < 1e-6f)
                    continue;                  // flat annular face
                if (aR[i] <= fRMinMm && aR[i + 1] <= fRMinMm)
                    continue;
                nBeams++;
                dCost += dBeamBoxVoxels(aZ[i], aR[i], aZ[i + 1], aR[i + 1]);
            }
            if (nBeams == 0)
                return new Voxels();

            if (dCost <= dLatticeBudgetVoxels)
            {
                Lattice lat = new Lattice();
                for (int i = 0; i + 1 < aZ.Count; i++)
                {
                    if (MathF.Abs(aZ[i + 1] - aZ[i]) < 1e-6f)
                        continue;              // flat annular face
                    float fRa = aR[i], fRb = aR[i + 1];
                    if (fRa <= fRMinMm && fRb <= fRMinMm)
                        continue;
                    lat.AddBeam(new Vector3(0f, 0f, aZ[i]),
                                MathF.Max(fRa, fRMinMm),
                                new Vector3(0f, 0f, aZ[i + 1]),
                                MathF.Max(fRb, fRMinMm), false);
                }
                return voxFrom(lat);
            }

            if (aZ[aZ.Count - 1] < aZ[0])
            {
                aZ.Reverse();
                aR.Reverse();
            }
            var aZm = new List<float>(aZ.Count);
            var aRm = new List<float>(aZ.Count);
            for (int i = 0; i < aZ.Count; i++)
            {
                float fR = MathF.Max(aR[i], fRMinMm);
                if (aZm.Count > 0
                    && MathF.Abs(aZ[i] - aZm[aZm.Count - 1]) < 1e-6f
                    && MathF.Abs(fR - aRm[aRm.Count - 1]) < 1e-6f)
                    continue;                  // repeated vertex
                aZm.Add(aZ[i]);
                aRm.Add(fR);
            }
            // a radius step at either END is not a face of the lattice
            // union (its first/last frustum starts at the inner point):
            // drop it so the end cap is never folded over itself
            while (aZm.Count > 2 && MathF.Abs(aZm[1] - aZm[0]) < 1e-6f)
            {
                aZm.RemoveAt(0);
                aRm.RemoveAt(0);
            }
            while (aZm.Count > 2
                   && MathF.Abs(aZm[aZm.Count - 1] - aZm[aZm.Count - 2])
                      < 1e-6f)
            {
                aZm.RemoveAt(aZm.Count - 1);
                aRm.RemoveAt(aRm.Count - 1);
            }
            Mesh msh = mshRevolveZ(aZm, aRm);
            Voxels vox = new Voxels(msh);
            msh.Dispose();
            return vox;
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
            return voxFrom(msh);
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
            return voxFrom(lat);
        }
    }
}
