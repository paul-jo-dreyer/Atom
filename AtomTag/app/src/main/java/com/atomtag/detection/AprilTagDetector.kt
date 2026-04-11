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
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.DetectorParameters
import org.opencv.objdetect.Objdetect

class AprilTagDetector {

    private val dictionary = Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11)
    private val detectorParams = DetectorParameters()
    private val detector = ArucoDetector(dictionary, detectorParams)

    private var cameraMatrix = Mat(3, 3, CvType.CV_64F)
    private var distCoeffs = MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0)
    private var intrinsicsInitialized = false

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

    /**
     * Detect AprilTags and estimate poses using per-tag sizes from config.
     */
    fun detect(grayFrame: Mat, projectAxes: Boolean = false): List<DetectionResult> {
        if (!intrinsicsInitialized) return emptyList()

        val corners = mutableListOf<Mat>()
        val ids = Mat()
        val rejected = mutableListOf<Mat>()

        detector.detectMarkers(grayFrame, corners, ids, rejected)

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

            val cornerMat = corners[i]
            val points = Array(4) { idx ->
                val data = cornerMat.get(0, idx)
                Point(data[0], data[1])
            }
            val imagePoints = MatOfPoint2f(*points)

            val rvec = Mat()
            val tvec = Mat()

            val solved = Calib3d.solvePnP(
                objPoints, imagePoints,
                cameraMatrix, distCoeffs,
                rvec, tvec
            )

            if (solved) {
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

                val bottomCenter3d = MatOfPoint3f(Point3(0.0, -half, 0.0))
                val bottomProjected = projectPoints(rvec, tvec, bottomCenter3d)[0]
                bottomCenter3d.release()

                results.add(DetectionResult(pose, axisProjected, bottomProjected))
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
