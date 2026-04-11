package com.atomtag.model

/**
 * A detection result including the pose and projected 2D axis points for visualization.
 *
 * @param pose The estimated tag pose
 * @param axisPoints Projected 2D points: [origin, x-tip, y-tip, z-tip] in image coordinates.
 *                   Null if axis projection wasn't requested.
 */
data class DetectionResult(
    val pose: TagPose,
    val axisPoints: Array<FloatArray>? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DetectionResult) return false
        return pose == other.pose
    }

    override fun hashCode(): Int = pose.hashCode()
}
