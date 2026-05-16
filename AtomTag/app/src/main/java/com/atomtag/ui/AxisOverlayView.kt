package com.atomtag.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffColorFilter
import android.util.AttributeSet
import android.view.View
import com.atomtag.model.FieldConfig
import com.atomtag.model.ScoreboardOverlay

/**
 * Transparent overlay that draws coordinate frame axes and tag index labels.
 * X = red, Y = green, Z = blue.
 * Each tag index gets a unique bright saturated color.
 */
class AxisOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    data class TagOverlayData(
        val tagId: Int,
        val axisPoints: Array<FloatArray>?,
        val bottomCenter: FloatArray?,
        /** Convex-hull silhouette of the robot body in image space. When the
         *  field overlay is on, this gets `clipOutPath`'d before field lines /
         *  goalie box are drawn so painted lines don't run across the robot. */
        val silhouette: Array<FloatArray>? = null,
    )

    /** Hollow-circle marker for a green-ball candidate in unrotated image space. */
    data class BallOverlayData(
        val pixelU: Float,
        val pixelV: Float,
        val pixelRadius: Float,
        val passedGate: Boolean,
    )

    private var tagData: List<TagOverlayData> = emptyList()
    private var fieldFrameAxes: Array<FloatArray>? = null
    private var ballData: List<BallOverlayData> = emptyList()
    private var fieldLineSegments: List<Array<FloatArray>>? = null
    private var goalieBoxSegments: List<Array<FloatArray>>? = null
    private var goalieBoxFillPolygon: Array<FloatArray>? = null
    private var scoreboardOverlay: ScoreboardOverlay? = null
    private var turfBaseQuad: Array<FloatArray>? = null
    private var turfStripeQuads: List<Array<FloatArray>> = emptyList()
    private var halfwayLineSegments: List<Array<FloatArray>> = emptyList()
    private var centerCircleSamples: Array<FloatArray>? = null
    private var centerDotSamples: Array<FloatArray>? = null
    private var logoQuad: Array<FloatArray>? = null
    private var imageWidth = 1
    private var imageHeight = 1
    private var rotationDegrees = 0

    /** Loaded once from `FieldConfig.LOGO_ASSET`. Null if disabled or missing. */
    private val logoBitmap: Bitmap? = run {
        if (!FieldConfig.LOGO_ENABLED) return@run null
        val name = FieldConfig.LOGO_ASSET ?: return@run null
        try {
            context.assets.open(name).use { BitmapFactory.decodeStream(it) }
        } catch (_: Throwable) {
            null
        }
    }

    private val paintX = Paint().apply {
        color = Color.RED; strokeWidth = 6f; style = Paint.Style.STROKE; isAntiAlias = true
    }
    private val paintY = Paint().apply {
        color = Color.GREEN; strokeWidth = 6f; style = Paint.Style.STROKE; isAntiAlias = true
    }
    private val paintZ = Paint().apply {
        color = Color.BLUE; strokeWidth = 6f; style = Paint.Style.STROKE; isAntiAlias = true
    }
    private val paintOrigin = Paint().apply {
        color = Color.WHITE; style = Paint.Style.FILL; isAntiAlias = true
    }
    private val labelPaint = Paint().apply {
        textSize = 48f; isFakeBoldText = true; isAntiAlias = true; textAlign = Paint.Align.CENTER
    }
    private val labelBgPaint = Paint().apply {
        color = Color.argb(160, 0, 0, 0); style = Paint.Style.FILL; isAntiAlias = true
    }
    private val ballPaintPassed = Paint().apply {
        color = Color.GREEN; strokeWidth = 4f; style = Paint.Style.STROKE; isAntiAlias = true
    }
    private val ballPaintRejected = Paint().apply {
        color = Color.argb(140, 0, 200, 0); strokeWidth = 2f; style = Paint.Style.STROKE; isAntiAlias = true
    }
    // === White-marking paints (consolidated) ===
    // One color + alpha + line width drives the halfway line, center circle,
    // center dot, existing field lines (goal_line + future entries in
    // FieldConfig.LINES), AND the goalie-box outline. Configured via the
    // `markings:` block in field_config.yaml.
    private val markingsStrokePaint = Paint().apply {
        color = FieldConfig.MARKINGS_COLOR_ARGB
        strokeWidth = FieldConfig.MARKINGS_LINE_WIDTH_PX
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }
    private val markingsFillPaint = Paint().apply {
        color = FieldConfig.MARKINGS_COLOR_ARGB
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // === Turf paints (base + mowed stripes) ===
    private val turfBasePaint = Paint().apply {
        color = FieldConfig.TURF_COLOR_ARGB; style = Paint.Style.FILL; isAntiAlias = true
    }
    private val turfStripeLightPaint = Paint().apply {
        color = shiftRgb(FieldConfig.TURF_COLOR_ARGB, +FieldConfig.TURF_STRIPES_DELTA / 2)
        style = Paint.Style.FILL; isAntiAlias = true
    }
    private val turfStripeDarkPaint = Paint().apply {
        color = shiftRgb(FieldConfig.TURF_COLOR_ARGB, -FieldConfig.TURF_STRIPES_DELTA / 2)
        style = Paint.Style.FILL; isAntiAlias = true
    }

    // === Logo paint ===
    // When `FieldConfig.LOGO_TINT_ARGB` is non-null, every non-transparent
    // pixel of the bitmap is replaced with the tint via PorterDuff SRC_IN
    // (alpha mask preserved). Use this to render a black-silhouette PNG in
    // white — set `logo.tint: "#FFFFFF"` in field_config.yaml.
    private val logoPaint = Paint().apply {
        isFilterBitmap = true
        isAntiAlias = true
        alpha = FieldConfig.LOGO_ALPHA
        FieldConfig.LOGO_TINT_ARGB?.let { tint ->
            colorFilter = PorterDuffColorFilter(tint, PorterDuff.Mode.SRC_IN)
        }
    }
    // === Team color palette ===
    // Single source of truth for the team-color overlays (goalie-box fill +
    // scoreboard score blocks). Tweak these to retune both at once. RGB values
    // are 0–255; per-paint alpha is set on the paints below to keep the fill
    // softer than the score block's tint.

    private val orangeR = 250; private val orangeG = 136; private val orangeB = 25
    private val blueR   =  25; private val blueG   = 129; private val blueB   = 255

    private val goalieFillOrangePaint = Paint().apply {
        color = Color.argb(150, orangeR, orangeG, orangeB); style = Paint.Style.FILL; isAntiAlias = true
    }
    private val goalieFillBluePaint = Paint().apply {
        color = Color.argb(150, blueR, blueG, blueB); style = Paint.Style.FILL; isAntiAlias = true
    }
    private val scoreboardPlatePaint = Paint().apply {
        color = Color.argb(220, 22, 22, 22); style = Paint.Style.FILL; isAntiAlias = true
    }
    private val scoreboardOrangeBlockPaint = Paint().apply {
        color = Color.argb(200, orangeR, orangeG, orangeB); style = Paint.Style.FILL; isAntiAlias = true
    }
    private val scoreboardBlueBlockPaint = Paint().apply {
        color = Color.argb(200, blueR, blueG, blueB); style = Paint.Style.FILL; isAntiAlias = true
    }
    private val scoreboardGlyphPaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 4f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    fun update(
        data: List<TagOverlayData>,
        imgWidth: Int,
        imgHeight: Int,
        rotation: Int,
        fieldAxes: Array<FloatArray>? = null,
        balls: List<BallOverlayData> = emptyList(),
        fieldLines: List<Array<FloatArray>>? = null,
        goalieBoxOutline: List<Array<FloatArray>>? = null,
        goalieBoxFill: Array<FloatArray>? = null,
        scoreboard: ScoreboardOverlay? = null,
        turfBase: Array<FloatArray>? = null,
        turfStripes: List<Array<FloatArray>> = emptyList(),
        halfwayLine: List<Array<FloatArray>> = emptyList(),
        centerCircle: Array<FloatArray>? = null,
        centerDot: Array<FloatArray>? = null,
        logo: Array<FloatArray>? = null,
    ) {
        tagData = data
        fieldFrameAxes = fieldAxes
        ballData = balls
        fieldLineSegments = fieldLines
        goalieBoxSegments = goalieBoxOutline
        goalieBoxFillPolygon = goalieBoxFill
        scoreboardOverlay = scoreboard
        turfBaseQuad = turfBase
        turfStripeQuads = turfStripes
        halfwayLineSegments = halfwayLine
        centerCircleSamples = centerCircle
        centerDotSamples = centerDot
        logoQuad = logo
        imageWidth = imgWidth
        imageHeight = imgHeight
        rotationDegrees = rotation
        postInvalidate()
    }

    fun clear() {
        tagData = emptyList()
        fieldFrameAxes = null
        ballData = emptyList()
        fieldLineSegments = null
        goalieBoxSegments = null
        goalieBoxFillPolygon = null
        scoreboardOverlay = null
        turfBaseQuad = null
        turfStripeQuads = emptyList()
        halfwayLineSegments = emptyList()
        centerCircleSamples = null
        centerDotSamples = null
        logoQuad = null
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // All flat-on-the-floor overlays — turf, mowed stripes, logo, halfway
        // line, center circle, center dot, goalie-box fill, field lines,
        // goalie-box outline, scoreboard — share one clip-out region so robots
        // and a gated ball punch through every one of them uniformly.
        val hasFieldOverlay = !fieldLineSegments.isNullOrEmpty() ||
            !goalieBoxSegments.isNullOrEmpty() ||
            goalieBoxFillPolygon != null ||
            scoreboardOverlay != null ||
            turfBaseQuad != null ||
            turfStripeQuads.isNotEmpty() ||
            halfwayLineSegments.isNotEmpty() ||
            centerCircleSamples != null ||
            centerDotSamples != null ||
            logoQuad != null
        if (hasFieldOverlay) {
            canvas.save()
            for (tag in tagData) {
                val sil = tag.silhouette ?: continue
                if (sil.size < 3) continue
                val path = Path()
                val first = mapPoint(sil[0])
                path.moveTo(first[0], first[1])
                for (i in 1 until sil.size) {
                    val p = mapPoint(sil[i])
                    path.lineTo(p[0], p[1])
                }
                path.close()
                canvas.clipOutPath(path)
            }
            val ballScale = viewScale()
            for (ball in ballData) {
                if (!ball.passedGate) continue
                val c = mapPoint(floatArrayOf(ball.pixelU, ball.pixelV))
                val r = ball.pixelRadius * ballScale
                if (r <= 0f) continue
                val path = Path()
                path.addCircle(c[0], c[1], r, Path.Direction.CW)
                canvas.clipOutPath(path)
            }

            // Back layer: turf. Stripes (when configured) replace the base —
            // drawing both would double-blend through alpha. When the turf has
            // rounded corners (`corner_radius_mm` > 0) the base polygon is
            // already a sampled rounded-rect; for the striped case we use it
            // as a clip mask so each stripe gets sliced at the rounded edges.
            val turfPath = turfBaseQuad?.takeIf { it.size >= 3 }?.let { buildClosedPath(it) }
            if (turfStripeQuads.isNotEmpty()) {
                canvas.save()
                if (turfPath != null) canvas.clipPath(turfPath)
                for ((i, stripe) in turfStripeQuads.withIndex()) {
                    if (stripe.size < 3) continue
                    val p = if (i % 2 == 0) turfStripeLightPaint else turfStripeDarkPaint
                    canvas.drawPath(buildClosedPath(stripe), p)
                }
                canvas.restore()
            } else if (turfPath != null) {
                canvas.drawPath(turfPath, turfBasePaint)
            }

            // Logo (above turf, below white markings).
            logoQuad?.let { quad ->
                val bm = logoBitmap
                if (bm != null && quad.size == 4) {
                    val src = floatArrayOf(
                        0f, 0f,
                        bm.width.toFloat(), 0f,
                        bm.width.toFloat(), bm.height.toFloat(),
                        0f, bm.height.toFloat(),
                    )
                    val dst = FloatArray(8)
                    for (i in 0 until 4) {
                        val mapped = mapPoint(quad[i])
                        dst[i * 2] = mapped[0]
                        dst[i * 2 + 1] = mapped[1]
                    }
                    val m = Matrix()
                    if (m.setPolyToPoly(src, 0, dst, 0, 4)) {
                        canvas.drawBitmap(bm, m, logoPaint)
                    }
                }
            }

            // Halfway line — one or two segments (split around the center circle).
            for (line in halfwayLineSegments) {
                if (line.size != 2) continue
                val a = mapPoint(line[0])
                val b = mapPoint(line[1])
                canvas.drawLine(a[0], a[1], b[0], b[1], markingsStrokePaint)
            }
            // Center circle outline.
            centerCircleSamples?.let { samples ->
                if (samples.size >= 3) canvas.drawPath(buildClosedPath(samples), markingsStrokePaint)
            }
            // Center dot (filled).
            centerDotSamples?.let { samples ->
                if (samples.size >= 3) canvas.drawPath(buildClosedPath(samples), markingsFillPaint)
            }

            // Goalie-box interior fill (on top of turf so its tint is faithful).
            goalieBoxFillPolygon?.let { poly ->
                val paint = when (FieldConfig.GOALIE_BOX_FILL) {
                    FieldConfig.GoalieBoxFill.Orange -> goalieFillOrangePaint
                    FieldConfig.GoalieBoxFill.Blue -> goalieFillBluePaint
                    FieldConfig.GoalieBoxFill.None -> null
                }
                if (paint != null && poly.size >= 3) {
                    canvas.drawPath(buildClosedPath(poly), paint)
                }
            }
            // Existing field lines + goalie-box outline (white, consolidated paint).
            fieldLineSegments?.forEach { line ->
                val a = mapPoint(line[0])
                val b = mapPoint(line[1])
                canvas.drawLine(a[0], a[1], b[0], b[1], markingsStrokePaint)
            }
            goalieBoxSegments?.forEach { segment ->
                val a = mapPoint(segment[0])
                val b = mapPoint(segment[1])
                canvas.drawLine(a[0], a[1], b[0], b[1], markingsStrokePaint)
            }
            // Scoreboard sits on top.
            scoreboardOverlay?.let { sb ->
                canvas.drawPath(buildClosedPath(sb.backgroundQuad), scoreboardPlatePaint)
                canvas.drawPath(buildClosedPath(sb.orangeBlock), scoreboardOrangeBlockPaint)
                canvas.drawPath(buildClosedPath(sb.blueBlock), scoreboardBlueBlockPaint)
                for (seg in sb.glyphSegments) {
                    val a = mapPoint(seg[0])
                    val b = mapPoint(seg[1])
                    canvas.drawLine(a[0], a[1], b[0], b[1], scoreboardGlyphPaint)
                }
            }
            canvas.restore()
        }

        fieldFrameAxes?.let { axes ->
            val origin = mapPoint(axes[0])
            val xTip = mapPoint(axes[1])
            val yTip = mapPoint(axes[2])
            val zTip = mapPoint(axes[3])
            canvas.drawLine(origin[0], origin[1], xTip[0], xTip[1], paintX)
            canvas.drawLine(origin[0], origin[1], yTip[0], yTip[1], paintY)
            canvas.drawLine(origin[0], origin[1], zTip[0], zTip[1], paintZ)
            canvas.drawCircle(origin[0], origin[1], 8f, paintOrigin)
            val fieldLabel = "F"
            val tw = labelPaint.measureText(fieldLabel)
            val padding = 8f
            val lx = origin[0]
            val ly = origin[1] - 16f
            canvas.drawRoundRect(
                lx - tw / 2 - padding,
                ly - labelPaint.textSize,
                lx + tw / 2 + padding,
                ly + padding,
                12f, 12f,
                labelBgPaint,
            )
            labelPaint.color = Color.WHITE
            canvas.drawText(fieldLabel, lx, ly, labelPaint)
        }

        if (ballData.isNotEmpty()) {
            val s = viewScale()
            for (ball in ballData) {
                val center = mapPoint(floatArrayOf(ball.pixelU, ball.pixelV))
                val r = ball.pixelRadius * s
                val paint = if (ball.passedGate) ballPaintPassed else ballPaintRejected
                canvas.drawCircle(center[0], center[1], r, paint)
            }
        }

        if (tagData.isEmpty()) return

        for (tag in tagData) {
            // Draw axes if available
            if (tag.axisPoints != null) {
                val origin = mapPoint(tag.axisPoints[0])
                val xTip = mapPoint(tag.axisPoints[1])
                val yTip = mapPoint(tag.axisPoints[2])
                val zTip = mapPoint(tag.axisPoints[3])

                canvas.drawLine(origin[0], origin[1], xTip[0], xTip[1], paintX)
                canvas.drawLine(origin[0], origin[1], yTip[0], yTip[1], paintY)
                canvas.drawLine(origin[0], origin[1], zTip[0], zTip[1], paintZ)
                canvas.drawCircle(origin[0], origin[1], 8f, paintOrigin)
            }

            // Always draw tag index label at bottom edge
            if (tag.bottomCenter != null) {
                val labelPos = mapPoint(tag.bottomCenter)
                val tagColor = colorForIndex(tag.tagId)
                labelPaint.color = tagColor

                val label = "${tag.tagId}"
                val textWidth = labelPaint.measureText(label)
                val padding = 8f
                val x = labelPos[0]
                val y = labelPos[1] + labelPaint.textSize + 4f

                // Background pill
                canvas.drawRoundRect(
                    x - textWidth / 2 - padding,
                    y - labelPaint.textSize,
                    x + textWidth / 2 + padding,
                    y + padding,
                    12f, 12f,
                    labelBgPaint
                )

                // Color indicator dot
                val dotPaint = Paint().apply { color = tagColor; style = Paint.Style.FILL; isAntiAlias = true }
                canvas.drawCircle(x - textWidth / 2 - padding - 12f, y - labelPaint.textSize / 2 + padding / 2, 8f, dotPaint)

                // Label text
                canvas.drawText(label, x, y, labelPaint)
            }
        }
    }

    private fun mapPoint(pt: FloatArray): FloatArray {
        val imgX = pt[0]
        val imgY = pt[1]

        val rotatedX: Float
        val rotatedY: Float
        val rotatedW: Int
        val rotatedH: Int

        when (rotationDegrees) {
            90 -> {
                rotatedX = imageHeight - 1f - imgY
                rotatedY = imgX
                rotatedW = imageHeight
                rotatedH = imageWidth
            }
            180 -> {
                rotatedX = imageWidth - 1f - imgX
                rotatedY = imageHeight - 1f - imgY
                rotatedW = imageWidth
                rotatedH = imageHeight
            }
            270 -> {
                rotatedX = imgY
                rotatedY = imageWidth - 1f - imgX
                rotatedW = imageHeight
                rotatedH = imageWidth
            }
            else -> {
                rotatedX = imgX
                rotatedY = imgY
                rotatedW = imageWidth
                rotatedH = imageHeight
            }
        }

        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val scale = maxOf(viewW / rotatedW, viewH / rotatedH)

        val scaledW = rotatedW * scale
        val scaledH = rotatedH * scale
        val offsetX = (viewW - scaledW) / 2f
        val offsetY = (viewH - scaledH) / 2f

        val vx = rotatedX * scale + offsetX
        val vy = rotatedY * scale + offsetY

        return floatArrayOf(vx, vy)
    }

    /** Shift the RGB channels of an ARGB int by [delta] (each clamped 0..255).
     *  Alpha is preserved. Used to derive the mowed-stripe tones from
     *  `FieldConfig.TURF_COLOR_ARGB`. */
    private fun shiftRgb(argb: Int, delta: Int): Int {
        val a = (argb ushr 24) and 0xFF
        val r = (((argb shr 16) and 0xFF) + delta).coerceIn(0, 255)
        val g = (((argb shr 8) and 0xFF) + delta).coerceIn(0, 255)
        val b = ((argb and 0xFF) + delta).coerceIn(0, 255)
        return Color.argb(a, r, g, b)
    }

    /** Build a closed Path from an array of [u, v] image-space points, mapping
     *  each through [mapPoint] into view-space first. */
    private fun buildClosedPath(points: Array<FloatArray>): Path {
        val path = Path()
        if (points.isEmpty()) return path
        val first = mapPoint(points[0])
        path.moveTo(first[0], first[1])
        for (i in 1 until points.size) {
            val p = mapPoint(points[i])
            path.lineTo(p[0], p[1])
        }
        path.close()
        return path
    }

    /** Same scale factor mapPoint uses, exposed so radii in image pixels can be scaled to view pixels. */
    private fun viewScale(): Float {
        val rotatedW: Int
        val rotatedH: Int
        when (rotationDegrees) {
            90, 270 -> { rotatedW = imageHeight; rotatedH = imageWidth }
            else    -> { rotatedW = imageWidth;  rotatedH = imageHeight }
        }
        return maxOf(width.toFloat() / rotatedW, height.toFloat() / rotatedH)
    }

    companion object {
        fun colorForIndex(index: Int): Int = TagColors.argbForIndex(index)
    }
}
