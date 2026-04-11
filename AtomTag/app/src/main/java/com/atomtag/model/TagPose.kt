package com.atomtag.model

/**
 * Represents a detected AprilTag's pose.
 *
 * @param tagId The AprilTag ID (index into the pose vector)
 * @param transform 4x4 homogeneous transformation matrix (row-major, 16 floats)
 *                  Encodes both rotation and translation of the tag relative to the camera.
 * @param timestampMs When this detection occurred
 */
data class TagPose(
    val tagId: Int,
    val transform: FloatArray,
    val timestampMs: Long = System.currentTimeMillis()
) {
    init {
        require(transform.size == 16) { "Transform must be a 4x4 matrix (16 floats)" }
    }

    val tx: Float get() = transform[3]
    val ty: Float get() = transform[7]
    val tz: Float get() = transform[11]

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TagPose) return false
        return tagId == other.tagId && transform.contentEquals(other.transform)
    }

    override fun hashCode(): Int = 31 * tagId + transform.contentHashCode()
}
