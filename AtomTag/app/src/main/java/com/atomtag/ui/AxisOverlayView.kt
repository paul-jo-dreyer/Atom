package com.atomtag.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Transparent overlay that draws coordinate frame axes on detected AprilTags.
 * X = red, Y = green, Z = blue.
 */
class AxisOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // Each entry: [origin, x-tip, y-tip, z-tip] as FloatArray(2) each
    private var axisData: List<Array<FloatArray>> = emptyList()
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

    fun updateAxes(axes: List<Array<FloatArray>>, imgWidth: Int, imgHeight: Int, rotation: Int) {
        axisData = axes
        imageWidth = imgWidth
        imageHeight = imgHeight
        rotationDegrees = rotation
        postInvalidate()
    }

    fun clear() {
        axisData = emptyList()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (axisData.isEmpty()) return

        for (axis in axisData) {
            val origin = mapPoint(axis[0])
            val xTip = mapPoint(axis[1])
            val yTip = mapPoint(axis[2])
            val zTip = mapPoint(axis[3])

            canvas.drawLine(origin[0], origin[1], xTip[0], xTip[1], paintX)
            canvas.drawLine(origin[0], origin[1], yTip[0], yTip[1], paintY)
            canvas.drawLine(origin[0], origin[1], zTip[0], zTip[1], paintZ)
            canvas.drawCircle(origin[0], origin[1], 8f, paintOrigin)
        }
    }

    /**
     * Map image coordinates to view coordinates, matching PreviewView's FILL_CENTER behavior.
     *
     * 1. Rotate image coords by rotationDegrees to get display-oriented coords
     * 2. Scale to fill the view (same as FILL_CENTER — scale to cover, then center)
     */
    private fun mapPoint(pt: FloatArray): FloatArray {
        val imgX = pt[0]
        val imgY = pt[1]

        // Step 1: Rotate image point to display orientation
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
            else -> { // 0
                rotatedX = imgX
                rotatedY = imgY
                rotatedW = imageWidth
                rotatedH = imageHeight
            }
        }

        // Step 2: FILL_CENTER scaling — scale to cover the view, then center-crop
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
}
