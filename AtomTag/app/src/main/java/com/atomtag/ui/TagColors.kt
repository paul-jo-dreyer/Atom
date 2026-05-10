package com.atomtag.ui

import android.graphics.Color

object TagColors {
    fun argbForIndex(index: Int): Int {
        val hue = (index * 137.508f) % 360f
        return Color.HSVToColor(floatArrayOf(hue, 1f, 1f))
    }
}
