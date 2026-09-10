//
// LocalFrame.cs - shim wrapper around PicoGK.Shapes.Frame3d.
//
// Copied verbatim from Phy-AC (AxialCompressorDesigner/LocalFrame.cs), only
// the namespace changed. Keeping the two siblings byte-identical here is
// deliberate: any fix to the frame algebra must land in both.
//
// PicoGK's public API exposes Frame3d with a constructor that takes
// (Vector3 pos, Vector3 vecLx, Vector3 vecLy, Vector3 vecLz) but the
// typical use case is "give me a frame at this position pointing in
// this direction". This shim gives a friendlier API that matches the
// one used in the ShapeKernel examples:
//
//   var f = new LocalFrame(position, xDir, zDir);   // yDir = zDir x xDir
//   var g = LocalFrame.oGetTranslatedFrame(f, dx);
//   var h = LocalFrame.oGetInvertFrame(f, flipX: true, flipY: false);
//

using System;
using System.Numerics;

namespace CombustorDesigner
{
    /// <summary>
    /// Friendly coordinate-frame wrapper. Use this for all sub-component
    /// frames (injector seats, vaporizer tubes, swirler vanes, drain
    /// ports). Z is the machine axis; the gas flows towards +Z.
    /// </summary>
    public struct LocalFrame
    {
        public Vector3 vecPos;
        public Vector3 vecLx;
        public Vector3 vecLy;
        public Vector3 vecLz;

        public LocalFrame(Vector3 pos)
        {
            vecPos = pos;
            vecLx = Vector3.UnitX;
            vecLy = Vector3.UnitY;
            vecLz = Vector3.UnitZ;
        }

        /// <summary>Position + local Z axis (X and Y are derived from Z via
        /// the right-hand rule).</summary>
        public LocalFrame(Vector3 pos, Vector3 vecZDir)
        {
            vecPos = pos;
            vecLz = Vector3.Normalize(vecZDir);
            vecLx = MathHelpers.vecArbitraryOrthogonal(vecLz);
            vecLy = Vector3.Cross(vecLz, vecLx);
        }

        /// <summary>Position + X and Z axes (Y derived from Z x X).</summary>
        public LocalFrame(Vector3 pos, Vector3 vecXDir, Vector3 vecZDir)
        {
            vecPos = pos;
            vecLx = Vector3.Normalize(vecXDir);
            vecLz = Vector3.Normalize(vecZDir);
            vecLy = Vector3.Cross(vecLz, vecLx);
        }

        public Vector3 vecGetPosition() => vecPos;
        public Vector3 vecGetLocalX()   => vecLx;
        public Vector3 vecGetLocalY()   => vecLy;
        public Vector3 vecGetLocalZ()   => vecLz;

        /// <summary>Translate this frame by a vector.</summary>
        public static LocalFrame oGetTranslatedFrame(LocalFrame src,
                                                     Vector3 translation)
            => new LocalFrame(src.vecPos + translation, src.vecLx, src.vecLz);

        /// <summary>Invert this frame: negate the X and/or Y axes.</summary>
        public static LocalFrame oGetInvertFrame(LocalFrame src, bool bFlipX,
                                                 bool bFlipY)
        {
            LocalFrame f = src;
            if (bFlipX) f.vecLx = -f.vecLx;
            if (bFlipY) f.vecLy = -f.vecLy;
            f.vecLz = Vector3.Cross(f.vecLx, f.vecLy);
            return f;
        }

        /// <summary>Frame sitting on the cylinder of radius fR at angle
        /// fPhi and axial station fZ, with local Z pointing radially
        /// outwards and local X along the machine axis (+Z world). This is
        /// THE frame for every radial feature of a combustor: dilution
        /// holes, drain ports, spark boss, vaporizer feed.</summary>
        public static LocalFrame oRadialAt(float fR, float fPhi, float fZ)
        {
            Vector3 vecRadial = new Vector3(MathF.Cos(fPhi),
                                            MathF.Sin(fPhi), 0f);
            return new LocalFrame(vecRadial * fR + new Vector3(0f, 0f, fZ),
                                  Vector3.UnitZ, vecRadial);
        }
    }

    /// <summary>Small vector helpers used across the library.</summary>
    public static class MathHelpers
    {
        /// <summary>Return any unit vector perpendicular to v.</summary>
        public static Vector3 vecArbitraryOrthogonal(Vector3 v)
        {
            Vector3 vNorm = Vector3.Normalize(v);
            // Pick whichever world axis is least parallel to vNorm.
            Vector3 axisX = Math.Abs(vNorm.X) < 0.9f
                ? Vector3.UnitX : Vector3.UnitY;
            return Vector3.Normalize(Vector3.Cross(vNorm, axisX));
        }

        public static Vector3 vecGetCylPoint(float r, float phi, float z)
            => new Vector3(r * MathF.Cos(phi), r * MathF.Sin(phi), z);

        public static Vector3 vecTranslatePointOntoFrame(LocalFrame frame,
                                                         Vector3 pt)
            => frame.vecPos + pt.X * frame.vecLx + pt.Y * frame.vecLy
                            + pt.Z * frame.vecLz;

        public static Vector3 vecRotateAroundZ(Vector3 pt, float angle,
                                               Vector3 pivot)
        {
            float c = MathF.Cos(angle), s = MathF.Sin(angle);
            Vector3 d = pt - pivot;
            return pivot + new Vector3(c * d.X - s * d.Y,
                                       s * d.X + c * d.Y, d.Z);
        }

        public static Vector3 vecRotateAroundAxis(Vector3 v, float angle,
                                                  Vector3 axis)
        {
            Vector3 a = Vector3.Normalize(axis);
            float c = MathF.Cos(angle), s = MathF.Sin(angle);
            return c * v + s * Vector3.Cross(a, v)
                 + (1f - c) * Vector3.Dot(a, v) * a;
        }

        /// <summary>Safe normalize: returns vecFallback for a null vector so
        /// no NaN ever reaches the voxel kernel.</summary>
        public static Vector3 vecSafeNormalize(Vector3 v, Vector3 vecFallback)
        {
            float fLen = v.Length();
            return fLen > 1e-9f ? v / fLen : vecFallback;
        }
    }
}
