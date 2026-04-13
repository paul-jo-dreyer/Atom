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
    val cornerBounds: Rect? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DetectionResult) return false
        return pose == other.pose
    }

    override fun hashCode(): Int = pose.hashCode()
}
