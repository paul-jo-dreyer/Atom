package com.atomtag.detection

import com.atomtag.model.DetectionResult
import com.atomtag.model.FieldConfig
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
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

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
    private val ORIGIN_FILTER_ALPHA_R = 1.0
    private val ORIGIN_FILTER_ALPHA_T = 0.5
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

    /** Camera intrinsics matrix; valid after initIntrinsics has been called. */
    fun cameraMatrix(): Mat = cameraMatrix
    fun intrinsicsReady(): Boolean = intrinsicsInitialized

    fun detect(
        grayFrame: Mat,
        projectAxes: Boolean = false,
        projectFieldOverlay: Boolean = false,
    ): List<DetectionResult> {
        if (!intrinsicsInitialized) return emptyList()

        // Step 1: Full-frame detection
        val results = detectInRegion(grayFrame, 0, 0, projectAxes, projectFieldOverlay)
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
            val roiDetections = detectInRegion(cropped, roiX, roiY, projectAxes, projectFieldOverlay)
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

    /** Per-tag inputs needed by both pose solve and axis-projection paths. */
    private data class TagInputs(
        val objPoints: MatOfPoint3f,
        val imagePoints: MatOfPoint2f,
        val cornerBounds: Rect,
        val tagSize: Double,
        val half: Double,
    )

    private fun prepareTagInputs(
        tagId: Int, cornerMat: Mat, offsetX: Int, offsetY: Int,
    ): TagInputs {
        val tagSize = TagConfig.getTagSize(tagId)
        val half = tagSize / 2.0

        val objPoints = MatOfPoint3f(
            Point3(-half, half, 0.0),
            Point3(half, half, 0.0),
            Point3(half, -half, 0.0),
            Point3(-half, -half, 0.0)
        )

        val points = Array(4) { idx ->
            val data = cornerMat.get(0, idx)
            Point(data[0] + offsetX, data[1] + offsetY)
        }
        val imagePoints = MatOfPoint2f(*points)

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

        return TagInputs(objPoints, imagePoints, cornerBounds, tagSize, half)
    }

    private fun buildDetectionResult(
        tagId: Int, rvec: Mat, tvec: Mat, inputs: TagInputs,
        projectAxes: Boolean, projectFieldOverlay: Boolean,
    ): DetectionResult {
        val transform = buildTransformMatrix(rvec, tvec)
        val pose = TagPose(tagId, transform)

        var axisProjected: Array<FloatArray>? = null
        if (projectAxes) {
            val axisLength = inputs.tagSize * 0.5
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
            val dx = FieldConfig.FIELD_FRAME_X_M.toDouble()
            val dy = FieldConfig.FIELD_FRAME_Y_M.toDouble()
            val dz = FieldConfig.FIELD_FRAME_Z_M.toDouble()
            val fieldAxisLength = inputs.tagSize * 0.5
            val fieldAxisPoints3d = MatOfPoint3f(
                Point3(dx, dy, dz),
                Point3(dx + fieldAxisLength, dy, dz),
                Point3(dx, dy + fieldAxisLength, dz),
                Point3(dx, dy, dz + fieldAxisLength),
            )
            fieldFrameAxes = projectPoints(rvec, tvec, fieldAxisPoints3d)
            fieldAxisPoints3d.release()
        }

        var fieldLines: List<Array<FloatArray>>? = null
        if (projectFieldOverlay && tagId == TagConfig.ORIGIN_TAG_ID && FieldConfig.LINES.isNotEmpty()) {
            val dx = FieldConfig.FIELD_FRAME_X_M.toDouble()
            val dy = FieldConfig.FIELD_FRAME_Y_M.toDouble()
            val dz = FieldConfig.FIELD_FRAME_Z_M.toDouble()
            val out = mutableListOf<Array<FloatArray>>()
            for (line in FieldConfig.LINES) {
                val pts3d = MatOfPoint3f(
                    Point3(dx + line.fromX, dy + line.fromY, dz + line.fromZ),
                    Point3(dx + line.toX,   dy + line.toY,   dz + line.toZ),
                )
                out += projectPoints(rvec, tvec, pts3d)
                pts3d.release()
            }
            fieldLines = out.toList()
        }

        var goalieBoxOutline: List<Array<FloatArray>>? = null
        if (projectFieldOverlay && tagId == TagConfig.ORIGIN_TAG_ID) {
            goalieBoxOutline = projectGoalieBoxOutline(rvec, tvec)
        }

        val bottomCenter3d = MatOfPoint3f(Point3(0.0, -inputs.half, 0.0))
        val bottomProjected = projectPoints(rvec, tvec, bottomCenter3d)[0]
        bottomCenter3d.release()

        return DetectionResult(
            pose = pose,
            axisPoints = axisProjected,
            bottomCenter = bottomProjected,
            cornerBounds = inputs.cornerBounds,
            fieldFrameAxes = fieldFrameAxes,
            fieldLines = fieldLines,
            goalieBoxOutline = goalieBoxOutline,
        )
    }

    /**
     * Project the goalie-box footprint as a list of 2D image-space line segments.
     *
     * The box is a rounded rectangle on the floor centered at
     * FieldConfig.GOALIE_BOX_{X,Y,Z}_M, width along tag-X, height along tag-Y.
     * Each corner arc is sampled at SAMPLES_PER_CORNER+1 points so perspective
     * distortion comes out smooth.
     *
     * Each tiny segment of the boundary is parametrically clipped against the
     * field-boundary half-planes built from FieldConfig.LINES (each line is
     * treated as a half-plane with the "inside" being the side the field
     * origin lies on). Surviving portions are emitted as `[start, end]` pairs.
     * The result is that the part of the goalie box outside the field bounds
     * (e.g. behind the goal line) is not drawn, and because the chord on each
     * clipping line is also not emitted, there's no double-draw with
     * `fieldLines`.
     *
     * Returns null when no segments survive (whole box outside) or the box
     * isn't configured.
     */
    private fun projectGoalieBoxOutline(rvec: Mat, tvec: Mat): List<Array<FloatArray>>? {
        val w = FieldConfig.GOALIE_BOX_WIDTH_M.toDouble()
        val h = FieldConfig.GOALIE_BOX_HEIGHT_M.toDouble()
        if (w <= 0.0 || h <= 0.0) return null

        val cx = FieldConfig.GOALIE_BOX_X_M.toDouble()
        val cy = FieldConfig.GOALIE_BOX_Y_M.toDouble()
        val cz = FieldConfig.GOALIE_BOX_Z_M.toDouble()
        val r = min(FieldConfig.GOALIE_BOX_CORNER_RADIUS_M.toDouble(), min(w, h) / 2.0)
        val hw = w / 2.0
        val hh = h / 2.0

        // Walking counterclockwise from the upper-right corner.
        val corners = arrayOf(
            doubleArrayOf(cx + hw - r, cy + hh - r, 0.0),               // UR
            doubleArrayOf(cx - hw + r, cy + hh - r, PI / 2.0),          // UL
            doubleArrayOf(cx - hw + r, cy - hh + r, PI),                // LL
            doubleArrayOf(cx + hw - r, cy - hh + r, 3.0 * PI / 2.0),    // LR
        )

        val boundary = mutableListOf<Point3>()
        for (c in corners) {
            val ccx = c[0]; val ccy = c[1]; val a0 = c[2]
            for (i in 0..SAMPLES_PER_CORNER) {
                val angle = a0 + (PI / 2.0) * i / SAMPLES_PER_CORNER
                boundary.add(Point3(ccx + r * cos(angle), ccy + r * sin(angle), cz))
            }
        }

        val planes = buildFieldBoundaryHalfPlanes()
        val visible3d = mutableListOf<Pair<Point3, Point3>>()
        val n = boundary.size
        for (i in 0 until n) {
            val a = boundary[i]
            val b = boundary[(i + 1) % n]
            val clipped = clipSegmentToHalfPlanes(a, b, planes)
            if (clipped != null) visible3d.add(clipped)
        }

        if (visible3d.isEmpty()) return null

        // Project all segment endpoints in one solvePnP-style call.
        val flat = mutableListOf<Point3>()
        for ((p, q) in visible3d) { flat.add(p); flat.add(q) }
        val mat = MatOfPoint3f(*flat.toTypedArray())
        val projected = projectPoints(rvec, tvec, mat)
        mat.release()

        val out = ArrayList<Array<FloatArray>>(visible3d.size)
        for (i in visible3d.indices) {
            out.add(arrayOf(projected[2 * i], projected[2 * i + 1]))
        }
        return out
    }

    /**
     * Half-plane (anchor + outward normal) representing one field-boundary
     * line. A point p is "inside" iff `n · (p - anchor) >= 0`.
     */
    private data class HalfPlane(val ax: Double, val ay: Double, val nx: Double, val ny: Double)

    /**
     * Build a half-plane for each FieldConfig.LINES entry, oriented so the
     * field origin lies inside. Lines are interpreted in tag-local coordinates
     * (we add the field-frame offset to each endpoint). Degenerate (zero-length)
     * lines are skipped.
     */
    private fun buildFieldBoundaryHalfPlanes(): List<HalfPlane> {
        if (FieldConfig.LINES.isEmpty()) return emptyList()
        val dx = FieldConfig.FIELD_FRAME_X_M.toDouble()
        val dy = FieldConfig.FIELD_FRAME_Y_M.toDouble()
        val out = mutableListOf<HalfPlane>()
        for (line in FieldConfig.LINES) {
            val ax = dx + line.fromX
            val ay = dy + line.fromY
            val bx = dx + line.toX
            val by = dy + line.toY
            // Perpendicular to the line direction.
            var nx = -(by - ay)
            var ny = (bx - ax)
            val len = sqrt(nx * nx + ny * ny)
            if (len < 1e-9) continue
            nx /= len
            ny /= len
            // Flip the normal so the field origin (at (dx, dy) in tag-local)
            // is on the "inside" side.
            if (nx * (dx - ax) + ny * (dy - ay) < 0.0) { nx = -nx; ny = -ny }
            out.add(HalfPlane(ax, ay, nx, ny))
        }
        return out
    }

    /**
     * Parametric clip of segment a→b against multiple half-planes (Liang-Barsky
     * style on a 2D ground plane; z is interpolated linearly along with x,y).
     * Returns the visible sub-segment, or null if it's entirely clipped away.
     */
    private fun clipSegmentToHalfPlanes(
        a: Point3, b: Point3, planes: List<HalfPlane>,
    ): Pair<Point3, Point3>? {
        if (planes.isEmpty()) return a to b
        var t0 = 0.0
        var t1 = 1.0
        for (p in planes) {
            val da = p.nx * (a.x - p.ax) + p.ny * (a.y - p.ay)
            val db = p.nx * (b.x - p.ax) + p.ny * (b.y - p.ay)
            when {
                da >= 0 && db >= 0 -> { /* fully inside this plane */ }
                da < 0 && db < 0 -> return null  // fully outside
                else -> {
                    val tCross = da / (da - db)
                    if (da >= 0) t1 = min(t1, tCross) else t0 = max(t0, tCross)
                    if (t0 > t1) return null
                }
            }
        }
        return Point3(
            a.x + t0 * (b.x - a.x),
            a.y + t0 * (b.y - a.y),
            a.z + t0 * (b.z - a.z),
        ) to Point3(
            a.x + t1 * (b.x - a.x),
            a.y + t1 * (b.y - a.y),
            a.z + t1 * (b.z - a.z),
        )
    }

    private companion object {
        const val SAMPLES_PER_CORNER = 8
    }

    /**
     * Run detection on a (possibly cropped) region.
     * offsetX/offsetY are added back to corner coordinates so they're in full-frame space.
     *
     * Two-pass: solve non-origin tags first, accumulating their tag-Z direction
     * in camera frame as a "world up" reference, then solve the origin tag with
     * IPPE_SQUARE and use the reference to break the planar ambiguity. The bots
     * are flat on the floor with their Z pointing up just like the origin, but
     * they're not filtered and don't suffer from the same flip in practice, so
     * their Z direction is a reliable per-frame proxy for gravity.
     */
    private fun detectInRegion(
        region: Mat, offsetX: Int, offsetY: Int,
        projectAxes: Boolean, projectFieldOverlay: Boolean,
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
        val upRefSum = DoubleArray(3)
        var upRefCount = 0
        var originRowIdx = -1

        // Pass 1: non-origin tags. Accumulate tag-Z (third column of R) in
        // camera frame to use as the "world up" reference for the origin.
        for (i in 0 until ids.rows()) {
            val tagId = ids.get(i, 0)[0].toInt()
            if (!TagConfig.isTracked(tagId)) continue
            if (tagId == TagConfig.ORIGIN_TAG_ID) { originRowIdx = i; continue }

            val inputs = prepareTagInputs(tagId, corners[i], offsetX, offsetY)
            val rvec = Mat()
            val tvec = Mat()
            val solved = Calib3d.solvePnP(
                inputs.objPoints, inputs.imagePoints,
                cameraMatrix, distCoeffs,
                rvec, tvec
            )
            if (solved) {
                val det = buildDetectionResult(tagId, rvec, tvec, inputs, projectAxes, projectFieldOverlay)
                results.add(det)
                // Tag's local +Z in camera frame = third column of R, which is
                // (transform[2], transform[6], transform[10]) in row-major form.
                upRefSum[0] += det.pose.transform[2].toDouble()
                upRefSum[1] += det.pose.transform[6].toDouble()
                upRefSum[2] += det.pose.transform[10].toDouble()
                upRefCount++
            }
            rvec.release()
            tvec.release()
            inputs.imagePoints.release()
            inputs.objPoints.release()
        }

        // Pass 2: origin tag (if detected), disambiguated by the up reference.
        if (originRowIdx >= 0) {
            val inputs = prepareTagInputs(
                TagConfig.ORIGIN_TAG_ID, corners[originRowIdx], offsetX, offsetY
            )
            val upRef = if (upRefCount > 0) {
                doubleArrayOf(
                    upRefSum[0] / upRefCount,
                    upRefSum[1] / upRefCount,
                    upRefSum[2] / upRefCount,
                )
            } else null

            val rvec = Mat()
            val tvec = Mat()
            val solved = solveOriginPoseDisambiguated(
                inputs.objPoints, inputs.imagePoints,
                rvec, tvec, upRef
            )
            if (solved) {
                filterOriginPoseInPlace(rvec, tvec)
                val det = buildDetectionResult(
                    TagConfig.ORIGIN_TAG_ID, rvec, tvec, inputs, projectAxes, projectFieldOverlay
                )
                results.add(det)
            }
            rvec.release()
            tvec.release()
            inputs.imagePoints.release()
            inputs.objPoints.release()
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

    /**
     * Origin-tag pose solve with planar-pose ambiguity resolved.
     *
     * For a square planar marker, solvePnP has two physically valid solutions
     * (Oberkampf "fold" ambiguity); at oblique viewing angles they differ in
     * tag orientation by tens of degrees, and corner-detection noise in low
     * light flips between them — visible as the origin's orientation jumping
     * between "Z up" and "Z down".
     *
     * SOLVEPNP_IPPE_SQUARE returns BOTH candidates explicitly. To pick the
     * right one we use [upReference]: the average of the bot tags' local +Z
     * directions in camera frame, accumulated by the caller. All tags are
     * flat on the floor with Z pointing up, so the bots' (unfiltered, stable)
     * Z directions form a reliable per-frame proxy for the gravity vector in
     * camera space. The IPPE candidate whose own Z direction has the largest
     * dot product with [upReference] is the "tag-Z up" solution.
     *
     * If [upReference] is null (no bots in frame) we fall back to the lower-
     * reprojection-error solution at index 0.
     */
    private fun solveOriginPoseDisambiguated(
        objPoints: MatOfPoint3f,
        imagePoints: MatOfPoint2f,
        outRvec: Mat,
        outTvec: Mat,
        upReference: DoubleArray?,
    ): Boolean {
        val rvecs = mutableListOf<Mat>()
        val tvecs = mutableListOf<Mat>()
        val n = Calib3d.solvePnPGeneric(
            objPoints, imagePoints,
            cameraMatrix, distCoeffs,
            rvecs, tvecs,
            false,
            Calib3d.SOLVEPNP_IPPE_SQUARE,
        )
        if (n == 0) return false

        val bestIdx = if (upReference != null && n > 1) {
            var bestI = 0
            var bestDot = -Double.POSITIVE_INFINITY
            val rotMat = Mat()
            for (i in 0 until n) {
                Calib3d.Rodrigues(rvecs[i], rotMat)
                val z0 = rotMat.get(0, 2)[0]
                val z1 = rotMat.get(1, 2)[0]
                val z2 = rotMat.get(2, 2)[0]
                val dot = z0 * upReference[0] + z1 * upReference[1] + z2 * upReference[2]
                if (dot > bestDot) { bestDot = dot; bestI = i }
            }
            rotMat.release()
            bestI
        } else {
            0
        }

        rvecs[bestIdx].copyTo(outRvec)
        tvecs[bestIdx].copyTo(outTvec)
        rvecs.forEach { it.release() }
        tvecs.forEach { it.release() }
        return true
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
