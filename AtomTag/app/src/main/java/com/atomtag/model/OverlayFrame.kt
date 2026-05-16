package com.atomtag.model

/**
 * Bundle of every painted-on-the-floor (or floating-on-the-tag) shape the UI
 * overlay needs for one frame, all already projected into image-space pixels.
 * Produced by `FieldOverlayProjector` from a list of `DetectionResult`s;
 * consumed by `AxisOverlayView` via `DetectionService`.
 *
 * Decouples detection from visualization: the detector knows nothing about any
 * of this — it just produces poses, and a separate pass turns them into shapes.
 */
data class OverlayFrame(
    /** Per-tag axis triads + label center. One entry per detected tag. */
    val tagAxes: List<TagAxisOverlay> = emptyList(),
    /** Field-frame axis triad rooted at the field origin. Origin tag only. */
    val fieldFrameAxes: Array<FloatArray>? = null,
    /** Each entry is `[start, end]` for one line in `FieldConfig.LINES`. */
    val fieldLines: List<Array<FloatArray>>? = null,
    /** Visible portion of the goalie-box outline as `[start, end]` segments. */
    val goalieBoxOutline: List<Array<FloatArray>>? = null,
    /** Closed polygon (3+ vertices) for the visible goalie-box interior fill. */
    val goalieBoxFill: Array<FloatArray>? = null,
    /** Scoreboard plate + per-team blocks + clock/score glyph segments. */
    val scoreboard: ScoreboardOverlay? = null,
    /** tagId → convex-hull silhouette of the robot body in image pixels.
     *  Used by the renderer as a `clipOutPath` mask. */
    val robotSilhouettes: Map<Int, Array<FloatArray>> = emptyMap(),
    /** 4 CCW corners of the field rectangle (turf base). Image-space pixels. */
    val turfBase: Array<FloatArray>? = null,
    /** Per-stripe 4-corner polygons that overlay the mowed bands on top of the
     *  base turf, alternating light/dark by list index. Empty list when stripes
     *  are disabled (mowed_stripes_n == 0). */
    val turfStripes: List<Array<FloatArray>> = emptyList(),
    /** Halfway-line segments — each entry is `[from, to]`. Split into two
     *  pieces around the center circle when the circle is enabled (so the
     *  line doesn't run through it); a single full-height segment otherwise.
     *  Empty list when the halfway line is disabled. */
    val halfwayLine: List<Array<FloatArray>> = emptyList(),
    /** Sampled vertices around the center-circle outline (CCW). Null when disabled. */
    val centerCircle: Array<FloatArray>? = null,
    /** Sampled vertices around the center-dot fill (CCW). Null when disabled. */
    val centerDot: Array<FloatArray>? = null,
    /** 4 corners of the logo bitmap's destination quad in image-space pixels,
     *  ordered (top-left, top-right, bottom-right, bottom-left) consistent
     *  with the source bitmap's (0,0)→(W,0)→(W,H)→(0,H) layout. The renderer
     *  uses `Matrix.setPolyToPoly` to warp the bitmap onto these corners. */
    val logoQuad: Array<FloatArray>? = null,
)

/**
 * Per-tag axis triad and label-position projection.
 */
data class TagAxisOverlay(
    val tagId: Int,
    /** `[origin, x-tip, y-tip, z-tip]` in image pixels, or null when axes are
     *  toggled off (the projector skips the work then). */
    val axisPoints: Array<FloatArray>?,
    /** Bottom-center of the tag's footprint in image pixels — used to anchor
     *  the tag-id label. Always populated when the tag is detected. */
    val bottomCenter: FloatArray?,
)
