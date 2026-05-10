package com.atomtag.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

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
        val bottomCenter: FloatArray?
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
    private var imageWidth = 1
    private var imageHeight = 1
    private var rotationDegrees = 0

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
    private val fieldLinePaint = Paint().apply {
        color = Color.argb(200, 255, 255, 255)
        strokeWidth = 5f
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
    ) {
        tagData = data
        fieldFrameAxes = fieldAxes
        ballData = balls
        fieldLineSegments = fieldLines
        goalieBoxSegments = goalieBoxOutline
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
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Field markings (drawn first so tags / ball / axes sit on top).
        fieldLineSegments?.forEach { line ->
            val a = mapPoint(line[0])
            val b = mapPoint(line[1])
            canvas.drawLine(a[0], a[1], b[0], b[1], fieldLinePaint)
        }
        goalieBoxSegments?.forEach { segment ->
            val a = mapPoint(segment[0])
            val b = mapPoint(segment[1])
            canvas.drawLine(a[0], a[1], b[0], b[1], fieldLinePaint)
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
