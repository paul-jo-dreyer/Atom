package com.atomtag.model

import org.opencv.core.Rect

/**
 * A detection result including the pose and projected 2D points for visualization.
 *
 * @param pose The estimated tag pose
 * @param axisPoints Projected 2D points: [origin, x-tip, y-tip, z-tip] in image coordinates.
 * @param bottomCenter Projected 2D point at the bottom edge center of the tag, for label placement.
 * @param cornerBounds Bounding box of the detected tag corners in full-frame image coordinates.
 *                     Used for ROI-based re-detection on subsequent frames.
 */
data class DetectionResult(
    val pose: TagPose,
    val axisPoints: Array<FloatArray>? = null,
    val bottomCenter: FloatArray? = null,
    val cornerBounds: Rect? = null,
    /** Projected 2D points for the field-frame axes [origin, x-tip, y-tip, z-tip].
     *  Only populated on the origin tag's detection result. */
    val fieldFrameAxes: Array<FloatArray>? = null,
    /** Projected 2D endpoints for each field-line marking from FieldConfig.LINES,
     *  in load order. Each entry is `[start, end]`. Only populated on the origin
     *  tag's detection result, and only when the field overlay flag is on.
     *  Renderer draws these behind tags / ball / axes. */
    val fieldLines: List<Array<FloatArray>>? = null,
    /** Projected 2D segments for the visible portion of the goalie-box footprint.
     *  Each entry is `[start, end]`. Segments are emitted after parametric
     *  clipping against the field-boundary half-planes (built from
     *  FieldConfig.LINES), so the part of the box outside the field — and any
     *  edge that lies on a field-boundary line — is not in the output. Only
     *  populated on the origin tag's detection result, and only when the field
     *  overlay flag is on. */
    val goalieBoxOutline: List<Array<FloatArray>>? = null,
    /** Projected convex-hull silhouette of the robot's 3D body (a cube centered
     *  on the tag's XY with the tag on its top face). In image pixels, CCW or CW
     *  order — the renderer just walks them. Populated for non-origin tags only,
     *  and only when the field overlay flag is on; the renderer uses it as a
     *  clip-out mask so field lines don't paint across the robot. */
    val robotSilhouette: Array<FloatArray>? = null,
    /** Projected closed polygon (3+ vertices) for the visible portion of the
     *  goalie-box interior, after Sutherland-Hodgman clipping against the
     *  field-boundary half-planes. Only populated on the origin tag's result,
     *  only when the field overlay flag is on and FieldConfig.GOALIE_BOX_FILL
     *  is non-None. The renderer fills it under the outline. */
    val goalieBoxFill: Array<FloatArray>? = null,
    /** Scoreboard projection — background plate, per-team score blocks, and
     *  clock + score glyph segments. Only populated on the origin tag's
     *  result, only when the field overlay flag is on. */
    val scoreboard: com.atomtag.model.ScoreboardOverlay? = null,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DetectionResult) return false
        return pose == other.pose
    }

    override fun hashCode(): Int = pose.hashCode()
}
