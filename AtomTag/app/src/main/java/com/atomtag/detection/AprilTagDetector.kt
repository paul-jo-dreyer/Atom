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

/**
 * AprilTag detection: ArUco marker detection (DICT_APRILTAG_36h11), pose
 * recovery via solvePnP (with IPPE_SQUARE planar-pose disambiguation for the
 * origin tag), an ROI re-detection pass for tags missing from the full-frame
 * scan, and an EMA filter on the origin tag's pose.
 *
 * Returns `DetectionResult`s containing only pose + corner bounds — visualization
 * geometry (axes, field lines, goalie box, scoreboard, robot silhouettes) is
 * produced separately by `FieldOverlayProjector`. The detector deliberately
 * doesn't import `FieldConfig`.
 */
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
    private val ORIGIN_FILTER_ALPHA_T = 1.0
    private val originFilterRvec = DoubleArray(3)
    private val originFilterTvec = DoubleArray(3)
    private var originFilterInitialized = false

    /**
     * Temporal-continuity anchor for `solveOriginPoseDisambiguated`. Holds the
     * last accepted (raw, pre-filter) origin rvec. When non-null, the
     * disambiguator picks the IPPE_SQUARE candidate with the smallest rotation
     * delta from this anchor instead of using the bot-Z reference.
     *
     * Invalidated when the origin has been missing for [ORIGIN_STALE_FRAMES]
     * consecutive frames — at that point we re-bootstrap from bot-Z on the next
     * sighting so we don't lock onto a stale wrong branch.
     */
    private val previousOriginRvec = DoubleArray(3)
    private var hasPreviousOriginRvec = false
    private var framesSinceOriginSeen = 0
    private val ORIGIN_STALE_FRAMES = 30   // ~1 second at 30 fps

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
    /** Distortion coefficients; valid for the lifetime of the detector. */
    fun distCoeffs(): MatOfDouble = distCoeffs
    fun intrinsicsReady(): Boolean = intrinsicsInitialized

    fun detect(grayFrame: Mat): List<DetectionResult> {
        if (!intrinsicsInitialized) return emptyList()

        // Step 1: Full-frame detection
        val results = detectInRegion(grayFrame, 0, 0)
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
            val roiDetections = detectInRegion(cropped, roiX, roiY)
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

        // Track origin-tag visibility for the temporal-continuity anchor.
        // After ORIGIN_STALE_FRAMES consecutive misses, invalidate both the
        // disambiguator anchor and the EMA filter so we re-bootstrap from
        // bot-Z on the next sighting.
        val originSeenThisFrame = allResults.any { it.pose.tagId == TagConfig.ORIGIN_TAG_ID }
        if (originSeenThisFrame) {
            framesSinceOriginSeen = 0
        } else {
            framesSinceOriginSeen++
            if (framesSinceOriginSeen >= ORIGIN_STALE_FRAMES) {
                hasPreviousOriginRvec = false
                originFilterInitialized = false
            }
        }

        return allResults
    }

    /** Per-tag inputs needed by the pose solve. */
    private data class TagInputs(
        val objPoints: MatOfPoint3f,
        val imagePoints: MatOfPoint2f,
        val cornerBounds: Rect,
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

        return TagInputs(objPoints, imagePoints, cornerBounds)
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
                val det = buildDetection(tagId, rvec, tvec, inputs)
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
                results.add(buildDetection(TagConfig.ORIGIN_TAG_ID, rvec, tvec, inputs))
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

    private fun buildDetection(
        tagId: Int, rvec: Mat, tvec: Mat, inputs: TagInputs,
    ): DetectionResult = DetectionResult(
        pose = TagPose(tagId, buildTransformMatrix(rvec, tvec)),
        cornerBounds = inputs.cornerBounds,
    )

    /**
     * Origin-tag pose solve with planar-pose ambiguity resolved.
     *
     * For a square planar marker, solvePnP has two physically valid solutions
     * (Oberkampf "fold" ambiguity); at oblique viewing angles they differ in
     * tag orientation by tens of degrees, and corner-detection noise in low
     * light flips between them — visible as the origin's orientation jumping
     * between "Z up" and "Z down".
     *
     * SOLVEPNP_IPPE_SQUARE returns BOTH candidates explicitly. Picking the
     * right one uses three criteria, in priority order:
     *
     *  1. **Temporal continuity** — when [hasPreviousOriginRvec] is true, the
     *     candidate with the smallest rotation-angle delta from
     *     [previousOriginRvec] wins. Once locked onto a solution, this keeps
     *     us locked unless physics genuinely says otherwise. This is the
     *     primary criterion in steady-state.
     *
     *  2. **Bot-Z reference (bootstrap)** — when there's no temporal anchor
     *     (first frame, or after ORIGIN_STALE_FRAMES of occlusion), use
     *     [upReference] — the average of the bot tags' local +Z directions in
     *     camera frame. All tags lie flat on the floor with Z up, so the
     *     bots' (unfiltered) Z directions are a per-frame proxy for gravity.
     *
     *  3. **Lower reprojection error** — last resort, when neither anchor is
     *     available (no bots visible and no prior). Picks index 0.
     *
     * On success, [previousOriginRvec] is updated with the accepted raw rvec
     * (pre-filter), so the next frame's call sees this frame's decision.
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

        val bestIdx = when {
            n == 1 -> 0
            hasPreviousOriginRvec -> pickClosestToPreviousOrigin(rvecs)
            upReference != null -> pickByUpReference(rvecs, upReference)
            else -> 0
        }

        rvecs[bestIdx].copyTo(outRvec)
        tvecs[bestIdx].copyTo(outTvec)

        // Refresh the temporal anchor with the accepted (raw, pre-filter) rvec.
        outRvec.get(0, 0, previousOriginRvec)
        hasPreviousOriginRvec = true

        rvecs.forEach { it.release() }
        tvecs.forEach { it.release() }
        return true
    }

    /**
     * Pick the IPPE candidate whose rotation is closest to [previousOriginRvec]
     * by 3D rotation-angle. Uses the Frobenius inner product identity
     * `trace(R₁ᵀ R₂) = sum_{i,j} R₁[i,j] · R₂[i,j]`, which lets us compute
     * the cos of the rotation angle between two rotation matrices without
     * actually multiplying the matrices. Larger cos = smaller angle = closer.
     */
    private fun pickClosestToPreviousOrigin(rvecs: List<Mat>): Int {
        // Materialize the anchor as a 3×3 rotation matrix once.
        val prevRvecMat = Mat(3, 1, CvType.CV_64F)
        prevRvecMat.put(0, 0, *previousOriginRvec)
        val prevRotMat = Mat()
        Calib3d.Rodrigues(prevRvecMat, prevRotMat)
        prevRvecMat.release()

        var bestI = 0
        var bestCosAngle = -Double.POSITIVE_INFINITY
        val candidateRot = Mat()
        for (i in rvecs.indices) {
            Calib3d.Rodrigues(rvecs[i], candidateRot)
            var frob = 0.0
            for (r in 0..2) for (c in 0..2) {
                frob += prevRotMat.get(r, c)[0] * candidateRot.get(r, c)[0]
            }
            // trace(R1ᵀ R2) = frob; rotation angle satisfies cos θ = (trace - 1) / 2.
            val cosAngle = ((frob - 1.0) / 2.0).coerceIn(-1.0, 1.0)
            if (cosAngle > bestCosAngle) {
                bestCosAngle = cosAngle
                bestI = i
            }
        }
        candidateRot.release()
        prevRotMat.release()
        return bestI
    }

    /** Pick the IPPE candidate whose tag-Z axis (third column of R) best
     *  aligns with [upReference]. Used only as the bootstrap path when no
     *  temporal anchor exists yet. */
    private fun pickByUpReference(rvecs: List<Mat>, upReference: DoubleArray): Int {
        var bestI = 0
        var bestDot = -Double.POSITIVE_INFINITY
        val rotMat = Mat()
        for (i in rvecs.indices) {
            Calib3d.Rodrigues(rvecs[i], rotMat)
            val z0 = rotMat.get(0, 2)[0]
            val z1 = rotMat.get(1, 2)[0]
            val z2 = rotMat.get(2, 2)[0]
            val dot = z0 * upReference[0] + z1 * upReference[1] + z2 * upReference[2]
            if (dot > bestDot) { bestDot = dot; bestI = i }
        }
        rotMat.release()
        return bestI
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
