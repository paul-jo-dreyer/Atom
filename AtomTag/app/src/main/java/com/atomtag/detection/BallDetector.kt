package com.atomtag.detection

import com.atomtag.model.FieldConfig
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * One green-ball candidate located in the current frame.
 *
 * Pixel coordinates are in the unrotated frame (same as AprilTag corners), so the
 * overlay's existing rotation handling works without modification.
 *
 * If the origin tag was not visible at capture, [fieldXYZ] / [expectedPixelRadius]
 * are null and [passedRadiusGate] is false. Downstream consumers should branch on
 * [BallDetectionResult.originVisible].
 */
data class BallCandidate(
    val pixelU: Double,
    val pixelV: Double,
    val pixelRadius: Double,
    val fieldXYZ: DoubleArray?,
    val expectedPixelRadius: Double?,
    val passedRadiusGate: Boolean,
)

data class BallDetectionResult(
    val candidates: List<BallCandidate>,
    val originVisible: Boolean,
)

/**
 * Detects a homogeneous green ball by HSV thresholding + min-enclosing-circle.
 * When the camera→field transform is available we additionally compute, per
 * candidate, the field-frame ground-plane intersection and an expected pixel
 * radius (from the known physical radius and the candidate's distance), and
 * gate against the observed radius for robustness.
 */
class BallDetector {

    /** HSV thresholds for green (OpenCV: H in 0..179, S/V in 0..255). */
    private val hsvLow = Scalar(35.0, 80.0, 40.0)
    private val hsvHigh = Scalar(85.0, 255.0, 255.0)

    private val morphKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))

    private val minContourAreaPx = 50.0
    private val minPixelRadius = 4.0

    /** Tolerance for the geometric radius gate: |obs - exp| / exp must be < this. */
    private val radiusGateTolerance = 0.75

    /** Reusable Mats to avoid per-frame allocation churn. */
    private val hsv = Mat()
    private val mask = Mat()

    fun detect(
        rgb: Mat,
        cameraMatrix: Mat,
        cameraToField: FloatArray?,
    ): BallDetectionResult {
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        Core.inRange(hsv, hsvLow, hsvHigh, mask)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, morphKernel)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, morphKernel)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()

        val originVisible = cameraToField != null
        val fx = cameraMatrix.get(0, 0)[0]
        val cx = cameraMatrix.get(0, 2)[0]
        val cy = cameraMatrix.get(1, 2)[0]

        val out = mutableListOf<BallCandidate>()
        for (contour in contours) {
            try {
                val area = Imgproc.contourArea(contour)
                if (area < minContourAreaPx) continue

                val center = Point()
                val radiusArr = FloatArray(1)
                val curve = MatOfPoint2f(*contour.toArray())
                Imgproc.minEnclosingCircle(curve, center, radiusArr)
                curve.release()

                val rPx = radiusArr[0].toDouble()
                if (rPx < minPixelRadius) continue

                if (cameraToField == null) {
                    out += BallCandidate(
                        pixelU = center.x,
                        pixelV = center.y,
                        pixelRadius = rPx,
                        fieldXYZ = null,
                        expectedPixelRadius = null,
                        passedRadiusGate = false,
                    )
                    continue
                }

                val ground = unprojectToGround(center.x, center.y, fx, cx, cy, cameraToField)
                val expectedR = ground?.let { fx * FieldConfig.BALL_RADIUS_M / it.distance }
                val passed = expectedR != null &&
                    abs(rPx - expectedR) / expectedR < radiusGateTolerance

                out += BallCandidate(
                    pixelU = center.x,
                    pixelV = center.y,
                    pixelRadius = rPx,
                    fieldXYZ = ground?.fieldPoint,
                    expectedPixelRadius = expectedR,
                    passedRadiusGate = passed,
                )
            } finally {
                contour.release()
            }
        }

        return BallDetectionResult(candidates = out.toList(), originVisible = originVisible)
    }

    private data class GroundIntersection(val fieldPoint: DoubleArray, val distance: Double)

    /**
     * Cast a ray from the camera origin through pixel (u, v) and intersect the
     * field-frame plane z=0 (the floor). Returns the field-frame intersection
     * point and the distance from the camera to that point (for use in the
     * expected-radius computation). Returns null when the ray does not point at
     * the floor (e.g. the camera is looking up).
     *
     * The cameraToField matrix is row-major 4×4 with rotation in the upper-left
     * 3×3 block and the camera origin in field frame in the translation column
     * (T[3], T[7], T[11]).
     */
    private fun unprojectToGround(
        u: Double, v: Double,
        fx: Double, cx: Double, cy: Double,
        T: FloatArray,
    ): GroundIntersection? {
        val dxCam = (u - cx) / fx
        val dyCam = (v - cy) / fx  // square pixels: fy == fx
        val dzCam = 1.0

        val dxF = T[0] * dxCam + T[1] * dyCam + T[2]  * dzCam
        val dyF = T[4] * dxCam + T[5] * dyCam + T[6]  * dzCam
        val dzF = T[8] * dxCam + T[9] * dyCam + T[10] * dzCam

        val camOriginZ = T[11].toDouble()
        if (abs(dzF) < 1e-9) return null
        val t = -camOriginZ / dzF
        if (t <= 0.0) return null

        val fx0 = T[3].toDouble() + t * dxF
        val fy0 = T[7].toDouble() + t * dyF
        val fz0 = 0.0

        val dirNorm = sqrt(dxF * dxF + dyF * dyF + dzF * dzF)
        val distance = t * dirNorm
        return GroundIntersection(doubleArrayOf(fx0, fy0, fz0), distance)
    }

    fun release() {
        hsv.release()
        mask.release()
        morphKernel.release()
    }
}
