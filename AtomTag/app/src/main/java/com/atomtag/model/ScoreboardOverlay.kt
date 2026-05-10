package com.atomtag.model

/**
 * Projected 2D geometry for one frame's scoreboard render.
 *
 * The scoreboard is a flat rectangle on the tag's plane (z=0 in tag-local
 * coords) at FieldConfig.SCOREBOARD_*_M. The detector projects:
 *  - the four corners of the background plate,
 *  - the four corners of each team's score block (orange left, blue right),
 *  - the line segments for the clock digits (MM:SS) and the per-team score
 *    digits, all rendered in the same color by the overlay.
 *
 * The overlay decides paint colors; this struct only carries pixel geometry.
 */
data class ScoreboardOverlay(
    /** 4 corners CCW of the scoreboard rectangle, image space. */
    val backgroundQuad: Array<FloatArray>,
    /** 4 corners CCW of the orange-team score block (lower-left half). */
    val orangeBlock: Array<FloatArray>,
    /** 4 corners CCW of the blue-team score block (lower-right half). */
    val blueBlock: Array<FloatArray>,
    /** All glyph segments — clock digits, colon dots, both score digits.
     *  Each entry is `[start, end]` in image-space pixels. */
    val glyphSegments: List<Array<FloatArray>>,
)
