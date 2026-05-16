package com.atomtag.model

import org.opencv.core.Rect

/**
 * Per-tag detection output. Identity (`pose`) and the corner bounding box used
 * by the detector's own ROI re-detection across frames — nothing else.
 *
 * Visualization geometry lives on [OverlayFrame], produced separately by
 * `FieldOverlayProjector`. Keeping this struct lean keeps detection and
 * rendering decoupled: the detector knows about ArUco markers and pose
 * recovery; everything that paints lives downstream.
 */
data class DetectionResult(
    val pose: TagPose,
    /** Bounding box of the detected tag corners in full-frame image
     *  coordinates. Used by `AprilTagDetector` to restrict the next frame's
     *  search to a small ROI when the tag was visible last frame. */
    val cornerBounds: Rect? = null,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DetectionResult) return false
        return pose == other.pose
    }

    override fun hashCode(): Int = pose.hashCode()
}
