package com.atomtag.detection

import com.atomtag.model.DetectionResult
import com.atomtag.model.FieldConfig
import com.atomtag.model.OverlayFrame
import com.atomtag.model.ScoreboardData
import com.atomtag.model.ScoreboardOverlay
import com.atomtag.model.TagAxisOverlay
import com.atomtag.model.TagConfig
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Turns a frame's worth of `DetectionResult`s into an `OverlayFrame` of
 * already-projected image-space geometry: per-tag axes, the field-frame triad,
 * goalie-box outline + fill, field lines, scoreboard (plate + per-team blocks
 * + 7-segment glyphs), and per-robot body silhouettes.
 *
 * Fully decoupled from `AprilTagDetector` — takes only camera intrinsics +
 * `FieldConfig`. To project, it reconstitutes rvec/tvec from each detection's
 * 4×4 transform via Rodrigues; the cost is one 3×3 matrix and one Rodrigues
 * call per tag per frame (microseconds), which is the minor wart of decoupling.
 *
 * Reusable: anything that has tag poses + camera intrinsics — recorded
 * detections, offline rendering, tests — can drive it.
 */
class FieldOverlayProjector(
    private val cameraMatrix: Mat,
    private val distCoeffs: MatOfDouble,
) {

    /**
     * Build the overlay for one frame.
     *
     * @param drawTagAxes when true, projects per-tag axis triads and (for the
     *   origin tag) the field-frame axis triad. Skipping is a real saving —
     *   each triad is 4 projected points per tag.
     * @param drawFieldOverlay when true, projects field lines, goalie-box
     *   outline + fill (if configured), scoreboard, and per-robot silhouettes.
     */
    fun project(
        detections: List<DetectionResult>,
        scoreboardData: ScoreboardData = ScoreboardData(),
        drawTagAxes: Boolean = false,
        drawFieldOverlay: Boolean = false,
    ): OverlayFrame {
        val tagAxes = ArrayList<TagAxisOverlay>(detections.size)
        var fieldFrameAxes: Array<FloatArray>? = null
        var fieldLines: List<Array<FloatArray>>? = null
        var goalieBoxOutline: List<Array<FloatArray>>? = null
        var goalieBoxFill: Array<FloatArray>? = null
        var scoreboard: ScoreboardOverlay? = null
        val robotSilhouettes = mutableMapOf<Int, Array<FloatArray>>()
        var turfBase: Array<FloatArray>? = null
        var turfStripes: List<Array<FloatArray>> = emptyList()
        var halfwayLine: List<Array<FloatArray>> = emptyList()
        var centerCircle: Array<FloatArray>? = null
        var centerDot: Array<FloatArray>? = null
        var logoQuad: Array<FloatArray>? = null

        for (det in detections) {
            val tagId = det.pose.tagId
            val (rvec, tvec) = rvecTvecFromTransform(det.pose.transform)
            try {
                tagAxes.add(buildTagAxisOverlay(tagId, rvec, tvec, drawTagAxes))

                if (tagId == TagConfig.ORIGIN_TAG_ID) {
                    if (drawTagAxes) {
                        fieldFrameAxes = projectFieldFrameAxes(tagId, rvec, tvec)
                    }
                    if (drawFieldOverlay) {
                        fieldLines = projectFieldLines(rvec, tvec)
                        goalieBoxOutline = projectGoalieBoxOutline(rvec, tvec)
                        if (FieldConfig.GOALIE_BOX_FILL != FieldConfig.GoalieBoxFill.None) {
                            goalieBoxFill = projectGoalieBoxFill(rvec, tvec)
                        }
                        scoreboard = projectScoreboard(rvec, tvec, scoreboardData)
                        val bg = projectFieldBackground(rvec, tvec)
                        turfBase = bg.turfBase
                        turfStripes = bg.turfStripes
                        halfwayLine = bg.halfwayLine
                        centerCircle = bg.centerCircle
                        centerDot = bg.centerDot
                        logoQuad = bg.logoQuad
                    }
                } else if (drawFieldOverlay) {
                    val sil = projectRobotSilhouette(rvec, tvec)
                    if (sil != null) robotSilhouettes[tagId] = sil
                }
            } finally {
                rvec.release()
                tvec.release()
            }
        }

        return OverlayFrame(
            tagAxes = tagAxes,
            fieldFrameAxes = fieldFrameAxes,
            fieldLines = fieldLines,
            goalieBoxOutline = goalieBoxOutline,
            goalieBoxFill = goalieBoxFill,
            scoreboard = scoreboard,
            robotSilhouettes = robotSilhouettes,
            turfBase = turfBase,
            turfStripes = turfStripes,
            halfwayLine = halfwayLine,
            centerCircle = centerCircle,
            centerDot = centerDot,
            logoQuad = logoQuad,
        )
    }

    // ── per-tag axes + label position ─────────────────────────────────────

    private fun buildTagAxisOverlay(
        tagId: Int, rvec: Mat, tvec: Mat, drawAxes: Boolean,
    ): TagAxisOverlay {
        val tagSize = TagConfig.getTagSize(tagId)
        val half = tagSize / 2.0

        val axes: Array<FloatArray>? = if (drawAxes) {
            val axisLength = tagSize * 0.5
            val pts = MatOfPoint3f(
                Point3(0.0, 0.0, 0.0),
                Point3(axisLength, 0.0, 0.0),
                Point3(0.0, axisLength, 0.0),
                Point3(0.0, 0.0, axisLength),
            )
            val out = projectPoints(rvec, tvec, pts)
            pts.release()
            out
        } else null

        val bottomCenter3d = MatOfPoint3f(Point3(0.0, -half, 0.0))
        val bottomCenter = projectPoints(rvec, tvec, bottomCenter3d)[0]
        bottomCenter3d.release()

        return TagAxisOverlay(tagId = tagId, axisPoints = axes, bottomCenter = bottomCenter)
    }

    private fun projectFieldFrameAxes(tagId: Int, rvec: Mat, tvec: Mat): Array<FloatArray> {
        val tagSize = TagConfig.getTagSize(tagId)
        val dx = FieldConfig.FIELD_FRAME_X_M.toDouble()
        val dy = FieldConfig.FIELD_FRAME_Y_M.toDouble()
        val dz = FieldConfig.FIELD_FRAME_Z_M.toDouble()
        val len = tagSize * 0.5
        val pts = MatOfPoint3f(
            Point3(dx, dy, dz),
            Point3(dx + len, dy, dz),
            Point3(dx, dy + len, dz),
            Point3(dx, dy, dz + len),
        )
        val out = projectPoints(rvec, tvec, pts)
        pts.release()
        return out
    }

    // ── field lines ───────────────────────────────────────────────────────

    private fun projectFieldLines(rvec: Mat, tvec: Mat): List<Array<FloatArray>>? {
        if (FieldConfig.LINES.isEmpty()) return null
        val dx = FieldConfig.FIELD_FRAME_X_M.toDouble()
        val dy = FieldConfig.FIELD_FRAME_Y_M.toDouble()
        val dz = FieldConfig.FIELD_FRAME_Z_M.toDouble()
        val out = mutableListOf<Array<FloatArray>>()
        for (line in FieldConfig.LINES) {
            val pts = MatOfPoint3f(
                Point3(dx + line.fromX, dy + line.fromY, dz + line.fromZ),
                Point3(dx + line.toX,   dy + line.toY,   dz + line.toZ),
            )
            out += projectPoints(rvec, tvec, pts)
            pts.release()
        }
        return out
    }

    // ── goalie box: outline (segments, parametric clip) ───────────────────

    /**
     * Visible portion of the rounded-rect goalie-box footprint, as `[start,
     * end]` image-space segments. Each tiny boundary segment is independently
     * clipped against the field-boundary half-planes (Liang-Barsky), so the
     * chord on each clip line is *not* emitted — no double-draw with field
     * lines, no closed-loop artifact behind the goal.
     */
    private fun projectGoalieBoxOutline(rvec: Mat, tvec: Mat): List<Array<FloatArray>>? {
        val w = FieldConfig.GOALIE_BOX_WIDTH_M.toDouble()
        val h = FieldConfig.GOALIE_BOX_HEIGHT_M.toDouble()
        if (w <= 0.0 || h <= 0.0) return null

        val boundary = goalieBoxBoundary(w, h)
        val planes = buildFieldBoundaryHalfPlanes()
        val visible = mutableListOf<Pair<Point3, Point3>>()
        for (i in boundary.indices) {
            val a = boundary[i]
            val b = boundary[(i + 1) % boundary.size]
            val clipped = clipSegmentToHalfPlanes(a, b, planes)
            if (clipped != null) visible.add(clipped)
        }
        if (visible.isEmpty()) return null

        val flat = ArrayList<Point3>(visible.size * 2)
        for ((p, q) in visible) { flat.add(p); flat.add(q) }
        val mat = MatOfPoint3f(*flat.toTypedArray())
        val projected = projectPoints(rvec, tvec, mat)
        mat.release()

        val out = ArrayList<Array<FloatArray>>(visible.size)
        for (i in visible.indices) {
            out.add(arrayOf(projected[2 * i], projected[2 * i + 1]))
        }
        return out
    }

    /**
     * Visible portion of the rounded-rect goalie-box footprint as a CCW closed
     * polygon (3+ vertices), Sutherland-Hodgman clipped against the field-
     * boundary half-planes. Suitable for a fill paint; chord edges along the
     * clip lines ARE present (which closes the polygon).
     */
    private fun projectGoalieBoxFill(rvec: Mat, tvec: Mat): Array<FloatArray>? {
        val w = FieldConfig.GOALIE_BOX_WIDTH_M.toDouble()
        val h = FieldConfig.GOALIE_BOX_HEIGHT_M.toDouble()
        if (w <= 0.0 || h <= 0.0) return null

        val boundary = goalieBoxBoundary(w, h)
        val planes = buildFieldBoundaryHalfPlanes()
        val clipped = if (planes.isEmpty()) boundary else clipPolygonToHalfPlanes(boundary, planes)
        if (clipped.size < 3) return null

        val mat = MatOfPoint3f(*clipped.toTypedArray())
        val projected = projectPoints(rvec, tvec, mat)
        mat.release()
        return projected
    }

    /** Sampled rounded-rect boundary in tag-local coords (CCW from upper-right).
     *  For `r <= 0` returns the 4 sharp corners. For `r > 0` returns 4 ×
     *  (SAMPLES_PER_CORNER + 1) points with each corner replaced by a quarter-
     *  arc. `r` is clamped to `min(w/2, h/2)`. */
    private fun roundedRectBoundary(
        cx: Double, cy: Double, w: Double, h: Double, r: Double, cz: Double,
    ): List<Point3> {
        val rClamped = max(0.0, min(r, min(w, h) / 2.0))
        val hw = w / 2.0
        val hh = h / 2.0

        if (rClamped <= 0.0) {
            return listOf(
                Point3(cx + hw, cy + hh, cz),
                Point3(cx - hw, cy + hh, cz),
                Point3(cx - hw, cy - hh, cz),
                Point3(cx + hw, cy - hh, cz),
            )
        }

        val corners = arrayOf(
            doubleArrayOf(cx + hw - rClamped, cy + hh - rClamped, 0.0),
            doubleArrayOf(cx - hw + rClamped, cy + hh - rClamped, PI / 2.0),
            doubleArrayOf(cx - hw + rClamped, cy - hh + rClamped, PI),
            doubleArrayOf(cx + hw - rClamped, cy - hh + rClamped, 3.0 * PI / 2.0),
        )
        val out = mutableListOf<Point3>()
        for (c in corners) {
            val ccx = c[0]; val ccy = c[1]; val a0 = c[2]
            for (i in 0..SAMPLES_PER_CORNER) {
                val angle = a0 + (PI / 2.0) * i / SAMPLES_PER_CORNER
                out.add(Point3(ccx + rClamped * cos(angle), ccy + rClamped * sin(angle), cz))
            }
        }
        return out
    }

    /** Goalie-box boundary — thin wrapper around `roundedRectBoundary` reading
     *  the goalie-box center, dims, radius, and z plane from `FieldConfig`. */
    private fun goalieBoxBoundary(w: Double, h: Double): List<Point3> =
        roundedRectBoundary(
            cx = FieldConfig.GOALIE_BOX_X_M.toDouble(),
            cy = FieldConfig.GOALIE_BOX_Y_M.toDouble(),
            w = w, h = h,
            r = FieldConfig.GOALIE_BOX_CORNER_RADIUS_M.toDouble(),
            cz = FieldConfig.GOALIE_BOX_Z_M.toDouble(),
        )

    // ── scoreboard ────────────────────────────────────────────────────────

    /**
     * Project the scoreboard plate, the per-team score blocks, and the 7-
     * segment glyphs for the clock + scores. Layout is built in scoreboard-
     * local 2D (origin at center, +X right, +Y up); a single rotation +
     * translation step lifts everything into tag-local coords before the
     * batched `projectPoints` call, so `FieldConfig.SCOREBOARD_ROTATION_DEG`
     * rotates the whole layout as one rigid unit.
     */
    private fun projectScoreboard(
        rvec: Mat, tvec: Mat, data: ScoreboardData,
    ): ScoreboardOverlay? {
        val w = FieldConfig.SCOREBOARD_WIDTH_M.toDouble()
        val h = FieldConfig.SCOREBOARD_HEIGHT_M.toDouble()
        if (w <= 0.0 || h <= 0.0) return null

        val cx = FieldConfig.SCOREBOARD_X_M.toDouble()
        val cy = FieldConfig.SCOREBOARD_Y_M.toDouble()
        val cz = FieldConfig.SCOREBOARD_Z_M.toDouble()
        val hw = w / 2.0
        val hh = h / 2.0

        // === scoreboard-local 2D ===
        val plateLocal = listOf(
            doubleArrayOf( hw, -hh),
            doubleArrayOf( hw,  hh),
            doubleArrayOf(-hw,  hh),
            doubleArrayOf(-hw, -hh),
        )

        val scoreRowYTop = -h * 0.04
        val scoreRowYBot = -h * 0.46
        val orangeXMin = -w * 0.42
        val orangeXMax = -w * 0.04
        val blueXMin   =  w * 0.04
        val blueXMax   =  w * 0.42
        val orangeLocal = listOf(
            doubleArrayOf(orangeXMax, scoreRowYBot),
            doubleArrayOf(orangeXMax, scoreRowYTop),
            doubleArrayOf(orangeXMin, scoreRowYTop),
            doubleArrayOf(orangeXMin, scoreRowYBot),
        )
        val blueLocal = listOf(
            doubleArrayOf(blueXMax, scoreRowYBot),
            doubleArrayOf(blueXMax, scoreRowYTop),
            doubleArrayOf(blueXMin, scoreRowYTop),
            doubleArrayOf(blueXMin, scoreRowYBot),
        )

        // Glyph cell ~10mm × 22mm on a 120×80 plate.
        val digitW = w * 0.083
        val digitH = h * 0.275
        val halfDW = digitW / 2.0
        val halfDH = digitH / 2.0

        val clockY =  h * 0.22
        val scoreY = -h * 0.25
        val pitch = digitW + w * 0.025
        val clockXs = doubleArrayOf(
            -pitch * 1.5, -pitch * 0.5, pitch * 0.5, pitch * 1.5,
        )
        val orangeScoreX = (orangeXMin + orangeXMax) / 2.0
        val blueScoreX = (blueXMin + blueXMax) / 2.0

        val totalSec = (data.clockMs / 1000L).coerceAtLeast(0L)
        val mm = (totalSec / 60L).coerceAtMost(99L)
        val ss = totalSec % 60L
        val clockDigits = intArrayOf(
            (mm / 10).toInt(), (mm % 10).toInt(),
            (ss / 10).toInt(), (ss % 10).toInt(),
        )

        val glyphLocal = mutableListOf<DoubleArray>()
        for (i in clockDigits.indices) {
            emitDigitLocal(glyphLocal, clockXs[i], clockY, halfDW, halfDH, clockDigits[i])
        }
        // Colon: two short horizontal dashes.
        val colonR = digitW * 0.12
        val colonYTop = clockY + halfDH * 0.45
        val colonYBot = clockY - halfDH * 0.45
        glyphLocal.add(doubleArrayOf(-colonR, colonYTop))
        glyphLocal.add(doubleArrayOf( colonR, colonYTop))
        glyphLocal.add(doubleArrayOf(-colonR, colonYBot))
        glyphLocal.add(doubleArrayOf( colonR, colonYBot))

        emitDigitLocal(glyphLocal, orangeScoreX, scoreY, halfDW, halfDH, data.orangeScore.coerceIn(0, 9))
        emitDigitLocal(glyphLocal, blueScoreX,   scoreY, halfDW, halfDH, data.blueScore.coerceIn(0, 9))

        // === scoreboard-local → tag-local ===
        val rotRad = Math.toRadians(FieldConfig.SCOREBOARD_ROTATION_DEG.toDouble())
        val cosA = cos(rotRad)
        val sinA = sin(rotRad)
        fun toTag(p: DoubleArray): Point3 = Point3(
            cx + p[0] * cosA - p[1] * sinA,
            cy + p[0] * sinA + p[1] * cosA,
            cz,
        )

        val plateBase = 0
        val orangeBase = plateBase + plateLocal.size
        val blueBase = orangeBase + orangeLocal.size
        val glyphsBase = blueBase + blueLocal.size
        val flat = ArrayList<Point3>(glyphsBase + glyphLocal.size)
        for (p in plateLocal) flat.add(toTag(p))
        for (p in orangeLocal) flat.add(toTag(p))
        for (p in blueLocal) flat.add(toTag(p))
        for (p in glyphLocal) flat.add(toTag(p))

        val mat = MatOfPoint3f(*flat.toTypedArray())
        val projected = projectPoints(rvec, tvec, mat)
        mat.release()

        val plateQuad = Array(plateLocal.size) { projected[plateBase + it] }
        val orangeQuad = Array(orangeLocal.size) { projected[orangeBase + it] }
        val blueQuad = Array(blueLocal.size) { projected[blueBase + it] }
        val segments = ArrayList<Array<FloatArray>>(glyphLocal.size / 2)
        var idx = glyphsBase
        while (idx + 1 < projected.size) {
            segments.add(arrayOf(projected[idx], projected[idx + 1]))
            idx += 2
        }
        return ScoreboardOverlay(plateQuad, orangeQuad, blueQuad, segments)
    }

    /** Append the lit segments of a 7-seg digit (in scoreboard-LOCAL 2D, before
     *  rotation) to `out` as endpoint pairs. */
    private fun emitDigitLocal(
        out: MutableList<DoubleArray>,
        cx: Double, cy: Double, hw: Double, hh: Double,
        digit: Int,
    ) {
        val mask = SEVEN_SEG[digit.coerceIn(0, 9)]
        val tl = doubleArrayOf(cx - hw, cy + hh)
        val tr = doubleArrayOf(cx + hw, cy + hh)
        val ml = doubleArrayOf(cx - hw, cy)
        val mr = doubleArrayOf(cx + hw, cy)
        val bl = doubleArrayOf(cx - hw, cy - hh)
        val br = doubleArrayOf(cx + hw, cy - hh)
        // a top, b upper-right, c lower-right, d bottom, e lower-left, f upper-left, g middle
        val segs = arrayOf(
            tl to tr, tr to mr, mr to br, bl to br, ml to bl, tl to ml, ml to mr,
        )
        for (i in 0..6) {
            if ((mask shr i) and 1 == 1) {
                out.add(segs[i].first)
                out.add(segs[i].second)
            }
        }
    }

    // ── field background (turf + stripes + halfway line + center circle + logo) ──

    /** Bundle returned by `projectFieldBackground`. Fields are nullable when
     *  their corresponding config is disabled (or geometry is degenerate). */
    private data class FieldBackground(
        val turfBase: Array<FloatArray>?,
        val turfStripes: List<Array<FloatArray>>,
        val halfwayLine: List<Array<FloatArray>>,
        val centerCircle: Array<FloatArray>?,
        val centerDot: Array<FloatArray>?,
        val logoQuad: Array<FloatArray>?,
    )

    /**
     * Project everything that lives on the "ground" between the bare floor and
     * the white field markings: the turf base rectangle, mowed-stripe sub-rects,
     * the halfway line, the center circle + dot, and the logo's destination
     * quad. All built in field-frame coords (origin = field center), translated
     * into tag-local, and batched into one `projectPoints` call.
     */
    private fun projectFieldBackground(rvec: Mat, tvec: Mat): FieldBackground {
        val w = FieldConfig.SIZE_X_M.toDouble()
        val h = FieldConfig.SIZE_Y_M.toDouble()
        if (w <= 0.0 || h <= 0.0) {
            return FieldBackground(null, emptyList(), emptyList(), null, null, null)
        }
        val dx = FieldConfig.FIELD_FRAME_X_M.toDouble()
        val dy = FieldConfig.FIELD_FRAME_Y_M.toDouble()
        val dz = FieldConfig.FIELD_FRAME_Z_M.toDouble()
        val hw = w / 2.0
        val hh = h / 2.0

        // Convert field-frame (fx, fy) → tag-local Point3 at z=floor.
        fun toTag(fx: Double, fy: Double): Point3 = Point3(dx + fx, dy + fy, dz)

        // === scoreboard-style batched buffer: indices keyed off section bases ===
        val flat = ArrayList<Point3>(128)

        // Turf base — sampled rounded-rect boundary (sharp 4-corner rect if
        // corner_radius_mm == 0). Built in tag-local coords directly via the
        // shared helper; we re-add to `flat` so it goes through the same
        // batched projection as everything else below.
        val turfBaseStart = if (FieldConfig.TURF_ENABLED) flat.size else -1
        val turfBaseCount: Int
        if (FieldConfig.TURF_ENABLED) {
            val turfBoundary = roundedRectBoundary(
                cx = dx, cy = dy, w = w, h = h,
                r = FieldConfig.TURF_CORNER_RADIUS_M.toDouble(),
                cz = dz,
            )
            turfBaseCount = turfBoundary.size
            flat.addAll(turfBoundary)
        } else {
            turfBaseCount = 0
        }

        // Mowed stripes: each stripe is a 4-corner sub-rect along the chosen axis.
        val n = if (FieldConfig.TURF_ENABLED) FieldConfig.TURF_STRIPES_N.coerceAtLeast(0) else 0
        val stripeAxis = FieldConfig.TURF_STRIPES_AXIS
        val stripeStart = flat.size
        if (n > 0 && stripeAxis != FieldConfig.TurfStripesAxis.None) {
            if (stripeAxis == FieldConfig.TurfStripesAxis.Vertical) {
                // Split along X — N vertical bands spanning Y from -hh to +hh.
                for (i in 0 until n) {
                    val x0 = -hw + (2.0 * hw * i) / n
                    val x1 = -hw + (2.0 * hw * (i + 1)) / n
                    flat.add(toTag(x1, -hh))
                    flat.add(toTag(x1, +hh))
                    flat.add(toTag(x0, +hh))
                    flat.add(toTag(x0, -hh))
                }
            } else {
                // Horizontal — split along Y.
                for (i in 0 until n) {
                    val y0 = -hh + (2.0 * hh * i) / n
                    val y1 = -hh + (2.0 * hh * (i + 1)) / n
                    flat.add(toTag(+hw, y0))
                    flat.add(toTag(+hw, y1))
                    flat.add(toTag(-hw, y1))
                    flat.add(toTag(-hw, y0))
                }
            }
        }

        // Halfway line at field-X = 0. When the center circle is enabled and
        // small enough to fit in-field, we split the line into two pieces so
        // it doesn't run through the circle. Otherwise we emit one full
        // segment from the bottom to the top edge of the field.
        val halfwayStart = flat.size
        val wantHalfway = FieldConfig.MARKINGS_ENABLED && FieldConfig.MARKINGS_HALFWAY_LINE
        val circleR = FieldConfig.MARKINGS_CENTER_CIRCLE_RADIUS_M.toDouble()
        val splitHalfway = wantHalfway && circleR > 0.0 && circleR < hh
        val halfwaySegmentCount: Int
        if (wantHalfway) {
            if (splitHalfway) {
                flat.add(toTag(0.0, -hh))
                flat.add(toTag(0.0, -circleR))
                flat.add(toTag(0.0, +circleR))
                flat.add(toTag(0.0, +hh))
                halfwaySegmentCount = 2
            } else {
                flat.add(toTag(0.0, -hh))
                flat.add(toTag(0.0, +hh))
                halfwaySegmentCount = 1
            }
        } else {
            halfwaySegmentCount = 0
        }

        // Center circle outline. Reuses `circleR` declared in the halfway block.
        val circleStart = flat.size
        val wantCircle = FieldConfig.MARKINGS_ENABLED && circleR > 0.0
        if (wantCircle) {
            for (i in 0 until CIRCLE_SAMPLES) {
                val t = 2.0 * PI * i / CIRCLE_SAMPLES
                flat.add(toTag(circleR * cos(t), circleR * sin(t)))
            }
        }

        // Center dot fill (small filled circle).
        val dotStart = flat.size
        val dotR = FieldConfig.MARKINGS_CENTER_DOT_RADIUS_M.toDouble()
        val wantDot = FieldConfig.MARKINGS_ENABLED && dotR > 0.0
        if (wantDot) {
            for (i in 0 until DOT_SAMPLES) {
                val t = 2.0 * PI * i / DOT_SAMPLES
                flat.add(toTag(dotR * cos(t), dotR * sin(t)))
            }
        }

        // Logo quad — corners ordered to match the source bitmap (0,0)/(W,0)/(W,H)/(0,H).
        // Bitmap +Y is DOWN, field +Y is up — so top-of-image maps to +Y in field.
        val logoStart = flat.size
        val lw = FieldConfig.LOGO_WIDTH_M.toDouble()
        val lh = FieldConfig.LOGO_HEIGHT_M.toDouble()
        val wantLogo = FieldConfig.LOGO_ENABLED && lw > 0.0 && lh > 0.0
        if (wantLogo) {
            val hlw = lw / 2.0
            val hlh = lh / 2.0
            flat.add(toTag(-hlw, +hlh))  // bitmap top-left
            flat.add(toTag(+hlw, +hlh))  // bitmap top-right
            flat.add(toTag(+hlw, -hlh))  // bitmap bottom-right
            flat.add(toTag(-hlw, -hlh))  // bitmap bottom-left
        }

        if (flat.isEmpty()) {
            return FieldBackground(null, emptyList(), emptyList(), null, null, null)
        }

        val mat = MatOfPoint3f(*flat.toTypedArray())
        val projected = projectPoints(rvec, tvec, mat)
        mat.release()

        // === split projected back into named sections ===
        val turfBaseOut: Array<FloatArray>? = if (turfBaseStart >= 0 && turfBaseCount >= 3) {
            Array(turfBaseCount) { projected[turfBaseStart + it] }
        } else null

        val stripesOut: List<Array<FloatArray>> = if (n > 0 && stripeAxis != FieldConfig.TurfStripesAxis.None) {
            ArrayList<Array<FloatArray>>(n).also { list ->
                for (i in 0 until n) {
                    val base = stripeStart + i * 4
                    list.add(arrayOf(projected[base], projected[base + 1], projected[base + 2], projected[base + 3]))
                }
            }
        } else emptyList()

        val halfwayOut: List<Array<FloatArray>> = if (halfwaySegmentCount > 0) {
            (0 until halfwaySegmentCount).map { i ->
                arrayOf(projected[halfwayStart + 2 * i], projected[halfwayStart + 2 * i + 1])
            }
        } else emptyList()

        val circleOut: Array<FloatArray>? = if (wantCircle) {
            Array(CIRCLE_SAMPLES) { projected[circleStart + it] }
        } else null

        val dotOut: Array<FloatArray>? = if (wantDot) {
            Array(DOT_SAMPLES) { projected[dotStart + it] }
        } else null

        val logoOut: Array<FloatArray>? = if (wantLogo) {
            Array(4) { projected[logoStart + it] }
        } else null

        return FieldBackground(turfBaseOut, stripesOut, halfwayOut, circleOut, dotOut, logoOut)
    }

    // ── robot silhouette ──────────────────────────────────────────────────

    /**
     * Convex hull (in image pixels) of the 8 corners of the robot's body cube
     * (side `FieldConfig.ROBOT_BODY_SIZE_M`, tag on the top face). The renderer
     * uses this as a `clipOutPath` mask for field overlays.
     */
    private fun projectRobotSilhouette(rvec: Mat, tvec: Mat): Array<FloatArray>? {
        val s = FieldConfig.ROBOT_BODY_SIZE_M.toDouble()
        if (s <= 0.0) return null
        val h = s / 2.0
        val cubePts = MatOfPoint3f(
            Point3(-h, -h,  0.0), Point3( h, -h,  0.0),
            Point3( h,  h,  0.0), Point3(-h,  h,  0.0),
            Point3(-h, -h, -s),   Point3( h, -h, -s),
            Point3( h,  h, -s),   Point3(-h,  h, -s),
        )
        val projected = projectPoints(rvec, tvec, cubePts)
        cubePts.release()

        val cvPts = MatOfPoint(*Array(projected.size) {
            Point(projected[it][0].toDouble(), projected[it][1].toDouble())
        })
        val hullIdx = MatOfInt()
        Imgproc.convexHull(cvPts, hullIdx)
        val idxArr = hullIdx.toArray()
        cvPts.release()
        hullIdx.release()
        if (idxArr.size < 3) return null
        return Array(idxArr.size) { projected[idxArr[it]] }
    }

    // ── geometry: half-plane clipping (lines + polygons) ──────────────────

    private data class HalfPlane(val ax: Double, val ay: Double, val nx: Double, val ny: Double)

    /** Four half-planes — the four edges of the field rectangle defined by
     *  `FieldConfig.SIZE_X_M` × `SIZE_Y_M`, each oriented so the field origin
     *  is on the inside. Returned in tag-local coords (we add the field-frame
     *  offset). Empty if the field has no configured size. */
    private fun buildFieldBoundaryHalfPlanes(): List<HalfPlane> {
        val w = FieldConfig.SIZE_X_M.toDouble()
        val h = FieldConfig.SIZE_Y_M.toDouble()
        if (w <= 0.0 || h <= 0.0) return emptyList()
        val dx = FieldConfig.FIELD_FRAME_X_M.toDouble()
        val dy = FieldConfig.FIELD_FRAME_Y_M.toDouble()
        val hw = w / 2.0
        val hh = h / 2.0
        // Anchor a point on each edge in tag-local coords; outward normal
        // would point AWAY from origin, so the inward normal is the negation.
        return listOf(
            HalfPlane(dx - hw, dy, +1.0, 0.0),  // left edge,  inside = +X
            HalfPlane(dx + hw, dy, -1.0, 0.0),  // right edge, inside = -X
            HalfPlane(dx, dy - hh, 0.0, +1.0),  // bottom edge,inside = +Y
            HalfPlane(dx, dy + hh, 0.0, -1.0),  // top edge,   inside = -Y
        )
    }

    /** Liang-Barsky-style parametric segment clip against multiple half-planes.
     *  z is interpolated linearly along the segment so the projection later
     *  stays consistent. Returns the visible sub-segment, or null if entirely
     *  clipped away. */
    private fun clipSegmentToHalfPlanes(
        a: Point3, b: Point3, planes: List<HalfPlane>,
    ): Pair<Point3, Point3>? {
        if (planes.isEmpty()) return a to b
        var t0 = 0.0; var t1 = 1.0
        for (p in planes) {
            val da = p.nx * (a.x - p.ax) + p.ny * (a.y - p.ay)
            val db = p.nx * (b.x - p.ax) + p.ny * (b.y - p.ay)
            when {
                da >= 0 && db >= 0 -> { /* fully inside this plane */ }
                da < 0 && db < 0 -> return null
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

    /** Sutherland-Hodgman polygon clip against a series of half-planes (each
     *  plane keeps points on its `+` side). Operates on the ground plane (x,y);
     *  z is interpolated along introduced edges. CCW in, CCW out. */
    private fun clipPolygonToHalfPlanes(
        poly: List<Point3>, planes: List<HalfPlane>,
    ): List<Point3> {
        if (poly.isEmpty()) return emptyList()
        var current: List<Point3> = poly
        for (p in planes) {
            if (current.isEmpty()) return emptyList()
            val next = ArrayList<Point3>(current.size + 2)
            val m = current.size
            for (i in 0 until m) {
                val a = current[i]
                val b = current[(i + 1) % m]
                val da = p.nx * (a.x - p.ax) + p.ny * (a.y - p.ay)
                val db = p.nx * (b.x - p.ax) + p.ny * (b.y - p.ay)
                val aIn = da >= 0
                val bIn = db >= 0
                if (aIn) next.add(a)
                if (aIn != bIn) {
                    val tCross = da / (da - db)
                    next.add(Point3(
                        a.x + tCross * (b.x - a.x),
                        a.y + tCross * (b.y - a.y),
                        a.z + tCross * (b.z - a.z),
                    ))
                }
            }
            current = next
        }
        return current
    }

    // ── lower-level helpers ───────────────────────────────────────────────

    /** Reconstitute (rvec, tvec) from a row-major 4×4 transform via Rodrigues.
     *  Caller releases the returned Mats. */
    private fun rvecTvecFromTransform(transform: FloatArray): Pair<Mat, Mat> {
        val rotMat = Mat(3, 3, CvType.CV_64F)
        val rotData = DoubleArray(9)
        for (r in 0..2) for (c in 0..2) {
            rotData[r * 3 + c] = transform[r * 4 + c].toDouble()
        }
        rotMat.put(0, 0, *rotData)
        val rvec = Mat()
        Calib3d.Rodrigues(rotMat, rvec)
        rotMat.release()

        val tvec = Mat(3, 1, CvType.CV_64F)
        tvec.put(0, 0,
            transform[3].toDouble(),
            transform[7].toDouble(),
            transform[11].toDouble(),
        )
        return rvec to tvec
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

    private companion object {
        const val SAMPLES_PER_CORNER = 8
        const val CIRCLE_SAMPLES = 48
        const val DOT_SAMPLES = 16

        /** Bit i = segment i lit (a=0, b=1, c=2, d=3, e=4, f=5, g=6). */
        val SEVEN_SEG = intArrayOf(
            0b0111111, // 0
            0b0000110, // 1
            0b1011011, // 2
            0b1001111, // 3
            0b1100110, // 4
            0b1101101, // 5
            0b1111101, // 6
            0b0000111, // 7
            0b1111111, // 8
            0b1101111, // 9
        )
    }
}
