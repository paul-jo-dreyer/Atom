package com.atomtag.detection

import com.atomtag.model.DetectionResult
import com.atomtag.model.TagConfig
import com.atomtag.model.TagPose
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import org.opencv.core.Rect
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.DetectorParameters
import org.opencv.objdetect.Objdetect
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class AprilTagDetector {

    private val dictionary = Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11)
    private val detectorParams = DetectorParameters().apply {
        set_minMarkerPerimeterRate(0.01)
        set_minDistanceToBorder(1)
    }
    private val detector = ArucoDetector(dictionary, detectorParams)

    private var cameraMatrix = Mat(3, 3, CvType.CV_64F)
    private var distCoeffs = MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0)
    private var intrinsicsInitialized = false

    /** Previous frame's detected corner bounding boxes, keyed by tag ID. */
    private val previousDetections = mutableMapOf<Int, Rect>()

    /** How much to expand the ROI around a previous detection (multiplier on bbox size). */
    private val roiExpansion = 2.0f

    /** Max frames a tag can be missing before we stop doing ROI search for it. */
    private val maxMissedFrames = 10
    private val missedFrameCounts = mutableMapOf<Int, Int>()

    /**
     * Origin-tag pose smoothing. Bot tags pass through unfiltered so robot motion stays
     * responsive. Rotation gets a much smaller alpha than translation: the field frame is
     * offset from the origin tag, so small angular noise in the origin's rotation is
     * amplified to mm-level jitter at the field frame origin (leverage = offset · sin θ).
     * Translation noise from solvePnP is already small, so translation alpha stays moderate.
     */
    private val ORIGIN_FILTER_ALPHA_R = 0.03
    private val ORIGIN_FILTER_ALPHA_T = 0.15
    private val originFilterRvec = DoubleArray(3)
    private val originFilterTvec = DoubleArray(3)
    private var originFilterInitialized = false

    private fun filterOriginPoseInPlace(rvec: Mat, tvec: Mat) {
        val r = DoubleArray(3); rvec.get(0, 0, r)
        val t = DoubleArray(3); tvec.get(0, 0, t)
        if (!originFilterInitialized) {
            for (i in 0..2) {
                originFilterRvec[i] = r[i]
                originFilterTvec[i] = t[i]
            }
            originFilterInitialized = true
        } else {
            for (i in 0..2) {
                originFilterRvec[i] = ORIGIN_FILTER_ALPHA_R * r[i] +
                    (1.0 - ORIGIN_FILTER_ALPHA_R) * originFilterRvec[i]
                originFilterTvec[i] = ORIGIN_FILTER_ALPHA_T * t[i] +
                    (1.0 - ORIGIN_FILTER_ALPHA_T) * originFilterTvec[i]
            }
        }
        rvec.put(0, 0, *originFilterRvec)
        tvec.put(0, 0, *originFilterTvec)
    }

    fun initIntrinsics(width: Int, height: Int) {
        if (intrinsicsInitialized) return
        val focalLength = width.toDouble()
        cameraMatrix.put(0, 0,
            focalLength, 0.0, width / 2.0,
            0.0, focalLength, height / 2.0,
            0.0, 0.0, 1.0
        )
        intrinsicsInitialized = true
    }

    fun detect(grayFrame: Mat, projectAxes: Boolean = false): List<DetectionResult> {
        if (!intrinsicsInitialized) return emptyList()

        // Step 1: Full-frame detection
        val results = detectInRegion(grayFrame, 0, 0, projectAxes)
        val foundIds = results.map { it.pose.tagId }.toSet()

        // Step 2: ROI re-detection for missing tags that were recently seen
        val roiResults = mutableListOf<DetectionResult>()
        for ((tagId, bbox) in previousDetections) {
            if (tagId in foundIds) {
                missedFrameCounts.remove(tagId)
                continue
            }

            val missed = missedFrameCounts.getOrDefault(tagId, 0) + 1
            missedFrameCounts[tagId] = missed
            if (missed > maxMissedFrames) continue

            // Expand the bbox and clamp to frame bounds
            val expandW = (bbox.width * roiExpansion).roundToInt()
            val expandH = (bbox.height * roiExpansion).roundToInt()
            val roiX = max(0, bbox.x - expandW)
            val roiY = max(0, bbox.y - expandH)
            val roiW = min(grayFrame.cols() - roiX, bbox.width + expandW * 2)
            val roiH = min(grayFrame.rows() - roiY, bbox.height + expandH * 2)

            if (roiW <= 0 || roiH <= 0) continue

            val roi = Rect(roiX, roiY, roiW, roiH)
            val cropped = Mat(grayFrame, roi)
            val roiDetections = detectInRegion(cropped, roiX, roiY, projectAxes)
            cropped.release()

            // Only keep the tag we were looking for
            for (det in roiDetections) {
                if (det.pose.tagId == tagId && det.pose.tagId !in foundIds) {
                    roiResults.add(det)
                }
            }
        }

        val allResults = results + roiResults

        // Update previous detections for next frame
        previousDetections.clear()
        for (det in allResults) {
            det.cornerBounds?.let { previousDetections[det.pose.tagId] = it }
        }

        // Clean up missed counts for tags no longer tracked
        missedFrameCounts.keys.removeAll { it !in previousDetections && (missedFrameCounts[it] ?: 0) > maxMissedFrames }

        return allResults
    }

    /**
     * Run detection on a (possibly cropped) region.
     * offsetX/offsetY are added back to corner coordinates so they're in full-frame space.
     */
    private fun detectInRegion(
        region: Mat, offsetX: Int, offsetY: Int, projectAxes: Boolean
    ): List<DetectionResult> {
        val corners = mutableListOf<Mat>()
        val ids = Mat()
        val rejected = mutableListOf<Mat>()

        detector.detectMarkers(region, corners, ids, rejected)

        if (ids.empty()) {
            ids.release()
            rejected.forEach { it.release() }
            return emptyList()
        }

        val results = mutableListOf<DetectionResult>()

        for (i in 0 until ids.rows()) {
            val tagId = ids.get(i, 0)[0].toInt()
            if (!TagConfig.isTracked(tagId)) continue

            val tagSize = TagConfig.getTagSize(tagId)
            val half = tagSize / 2.0

            val objPoints = MatOfPoint3f(
                Point3(-half, half, 0.0),
                Point3(half, half, 0.0),
                Point3(half, -half, 0.0),
                Point3(-half, -half, 0.0)
            )

            // Extract corners and offset to full-frame coordinates
            val cornerMat = corners[i]
            val points = Array(4) { idx ->
                val data = cornerMat.get(0, idx)
                Point(data[0] + offsetX, data[1] + offsetY)
            }
            val imagePoints = MatOfPoint2f(*points)

            // Compute bounding box from corners (in full-frame coords)
            var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE
            var maxX = Float.MIN_VALUE; var maxY = Float.MIN_VALUE
            for (pt in points) {
                minX = min(minX, pt.x.toFloat()); minY = min(minY, pt.y.toFloat())
                maxX = max(maxX, pt.x.toFloat()); maxY = max(maxY, pt.y.toFloat())
            }
            val cornerBounds = Rect(
                minX.roundToInt(), minY.roundToInt(),
                (maxX - minX).roundToInt(), (maxY - minY).roundToInt()
            )

            val rvec = Mat()
            val tvec = Mat()

            val solved = Calib3d.solvePnP(
                objPoints, imagePoints,
                cameraMatrix, distCoeffs,
                rvec, tvec
            )

            if (solved) {
                if (tagId == TagConfig.ORIGIN_TAG_ID) {
                    filterOriginPoseInPlace(rvec, tvec)
                }
                val transform = buildTransformMatrix(rvec, tvec)
                val pose = TagPose(tagId, transform)

                var axisProjected: Array<FloatArray>? = null
                if (projectAxes) {
                    val axisLength = tagSize * 0.5
                    val axisPoints3d = MatOfPoint3f(
                        Point3(0.0, 0.0, 0.0),
                        Point3(axisLength, 0.0, 0.0),
                        Point3(0.0, axisLength, 0.0),
                        Point3(0.0, 0.0, axisLength)
                    )
                    axisProjected = projectPoints(rvec, tvec, axisPoints3d)
                    axisPoints3d.release()
                }

                var fieldFrameAxes: Array<FloatArray>? = null
                if (projectAxes && tagId == TagConfig.ORIGIN_TAG_ID) {
                    val dx = TagConfig.FIELD_FRAME_X_M.toDouble()
                    val dy = TagConfig.FIELD_FRAME_Y_M.toDouble()
                    val dz = TagConfig.FIELD_FRAME_Z_M.toDouble()
                    val fieldAxisLength = tagSize * 0.5
                    val fieldAxisPoints3d = MatOfPoint3f(
                        Point3(dx, dy, dz),
                        Point3(dx + fieldAxisLength, dy, dz),
                        Point3(dx, dy + fieldAxisLength, dz),
                        Point3(dx, dy, dz + fieldAxisLength),
                    )
                    fieldFrameAxes = projectPoints(rvec, tvec, fieldAxisPoints3d)
                    fieldAxisPoints3d.release()
                }

                val bottomCenter3d = MatOfPoint3f(Point3(0.0, -half, 0.0))
                val bottomProjected = projectPoints(rvec, tvec, bottomCenter3d)[0]
                bottomCenter3d.release()

                results.add(
                    DetectionResult(
                        pose = pose,
                        axisPoints = axisProjected,
                        bottomCenter = bottomProjected,
                        cornerBounds = cornerBounds,
                        fieldFrameAxes = fieldFrameAxes,
                    )
                )
            }

            rvec.release()
            tvec.release()
            imagePoints.release()
            objPoints.release()
        }

        ids.release()
        corners.forEach { it.release() }
        rejected.forEach { it.release() }

        return results
    }

    private fun projectPoints(rvec: Mat, tvec: Mat, points3d: MatOfPoint3f): Array<FloatArray> {
        val projected = MatOfPoint2f()
        Calib3d.projectPoints(points3d, rvec, tvec, cameraMatrix, distCoeffs, projected)

        val pts = projected.toArray()
        projected.release()

        return Array(pts.size) { i ->
            floatArrayOf(pts[i].x.toFloat(), pts[i].y.toFloat())
        }
    }

    private fun buildTransformMatrix(rvec: Mat, tvec: Mat): FloatArray {
        val rotMat = Mat()
        Calib3d.Rodrigues(rvec, rotMat)

        val t = FloatArray(16)
        for (r in 0..2) {
            for (c in 0..2) {
                t[r * 4 + c] = rotMat.get(r, c)[0].toFloat()
            }
            t[r * 4 + 3] = tvec.get(r, 0)[0].toFloat()
        }
        t[12] = 0f; t[13] = 0f; t[14] = 0f; t[15] = 1f

        rotMat.release()
        return t
    }

    fun release() {
        cameraMatrix.release()
    }
}
