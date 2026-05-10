package com.atomtag.detection

import com.atomtag.model.FieldConfig
import com.atomtag.model.TagConfig
import com.atomtag.model.TagPose
import kotlin.math.atan2

/**
 * Transforms detected tag poses from camera frame into the origin tag's coordinate frame.
 *
 * When the origin tag is visible, all poses are expressed relative to it (in meters).
 * When the origin tag is not visible, poses are returned in camera frame as a fallback.
 */
object PoseTransformer {

    /**
     * Transform poses into the field frame: origin-tag frame translated by
     * (FIELD_FRAME_X_M, FIELD_FRAME_Y_M, FIELD_FRAME_Z_M).
     *
     * If the origin tag is not visible, returns camera-frame poses unchanged.
     */
    fun transformToFieldFrame(detections: List<TagPose>): List<TagPose> {
        val originVisible = detections.any { it.tagId == TagConfig.ORIGIN_TAG_ID }
        val zeroFrame = transformToOriginFrame(detections)
        if (!originVisible) return zeroFrame

        val dx = FieldConfig.FIELD_FRAME_X_M
        val dy = FieldConfig.FIELD_FRAME_Y_M
        val dz = FieldConfig.FIELD_FRAME_Z_M
        return zeroFrame.map { pose ->
            val t = pose.transform.copyOf()
            t[3] = pose.transform[3] - dx
            t[7] = pose.transform[7] - dy
            t[11] = pose.transform[11] - dz
            TagPose(pose.tagId, t, pose.timestampMs)
        }
    }

    /**
     * Yaw (rotation about field-Z) of a tag from its field-frame transform,
     * in radians. Defined as the angle of the tag's local +X axis in the
     * field XY plane: atan2(R[1,0], R[0,0]) — i.e. atan2(transform[4],
     * transform[0]) for the row-major 4×4 used everywhere else.
     */
    fun yawFromFieldFrameTransform(transform: FloatArray): Float {
        return atan2(transform[4], transform[0])
    }

    /**
     * Returns the camera's pose expressed in the field frame, as a 4×4 row-major
     * matrix. Returns null if the origin tag is not visible. The translation
     * column is the camera origin in field-frame coordinates; the 3×3 rotation
     * block rotates a vector from camera frame into field frame.
     */
    fun cameraToFieldTransform(detections: List<TagPose>): FloatArray? {
        val origin = detections.firstOrNull { it.tagId == TagConfig.ORIGIN_TAG_ID } ?: return null
        val inv = invertTransform(origin.transform)
        inv[3]  -= FieldConfig.FIELD_FRAME_X_M
        inv[7]  -= FieldConfig.FIELD_FRAME_Y_M
        inv[11] -= FieldConfig.FIELD_FRAME_Z_M
        return inv
    }

    /**
     * Given all detected poses (in camera frame), return them transformed into the
     * origin tag's coordinate frame.
     */
    fun transformToOriginFrame(detections: List<TagPose>): List<TagPose> {
        val originPose = detections.find { it.tagId == TagConfig.ORIGIN_TAG_ID }

        if (originPose == null) {
            // Origin tag not visible — return camera-frame poses as-is
            return detections
        }

        val originInv = invertTransform(originPose.transform)

        return detections.map { pose ->
            if (pose.tagId == TagConfig.ORIGIN_TAG_ID) {
                TagPose(pose.tagId, IDENTITY_4X4, pose.timestampMs)
            } else {
                val worldTransform = multiply4x4(originInv, pose.transform)
                TagPose(pose.tagId, worldTransform, pose.timestampMs)
            }
        }
    }

    /**
     * Invert a 4x4 homogeneous transform matrix (row-major).
     * For a rigid transform [R|t; 0 0 0 1], the inverse is [R^T | -R^T * t; 0 0 0 1].
     */
    private fun invertTransform(m: FloatArray): FloatArray {
        val inv = FloatArray(16)

        // R^T (transpose the 3x3 rotation block)
        inv[0]  = m[0]; inv[1]  = m[4]; inv[2]  = m[8]
        inv[4]  = m[1]; inv[5]  = m[5]; inv[6]  = m[9]
        inv[8]  = m[2]; inv[9]  = m[6]; inv[10] = m[10]

        // -R^T * t
        val tx = m[3]; val ty = m[7]; val tz = m[11]
        inv[3]  = -(inv[0] * tx + inv[1] * ty + inv[2]  * tz)
        inv[7]  = -(inv[4] * tx + inv[5] * ty + inv[6]  * tz)
        inv[11] = -(inv[8] * tx + inv[9] * ty + inv[10] * tz)

        // Bottom row
        inv[12] = 0f; inv[13] = 0f; inv[14] = 0f; inv[15] = 1f

        return inv
    }

    /**
     * Multiply two 4x4 row-major matrices: result = A * B
     */
    private fun multiply4x4(a: FloatArray, b: FloatArray): FloatArray {
        val r = FloatArray(16)
        for (row in 0..3) {
            for (col in 0..3) {
                var sum = 0f
                for (k in 0..3) {
                    sum += a[row * 4 + k] * b[k * 4 + col]
                }
                r[row * 4 + col] = sum
            }
        }
        return r
    }

    private val IDENTITY_4X4 = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f
    )
}
