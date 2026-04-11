package com.atomtag.detection

import com.atomtag.model.TagConfig
import com.atomtag.model.TagPose

/**
 * Transforms detected tag poses from camera frame into the origin tag's coordinate frame.
 *
 * The origin tag (TagConfig.ORIGIN_TAG_ID) defines the world frame.
 * All other tag poses are expressed relative to it.
 *
 * Currently a pass-through — implement the actual transform when ready.
 */
object PoseTransformer {

    /**
     * Given all detected poses (in camera frame), return them transformed into the
     * origin tag's coordinate frame.
     *
     * @param detections Raw poses in camera frame
     * @return Poses in origin-tag coordinate frame
     */
    fun transformToOriginFrame(detections: List<TagPose>): List<TagPose> {
        val originPose = detections.find { it.tagId == TagConfig.ORIGIN_TAG_ID }

        if (originPose == null) {
            // Origin tag not visible — return camera-frame poses as-is
            return detections
        }

        // TODO: Compute T_origin_inv = inverse(originPose.transform),
        //       then for each detection: T_world = T_origin_inv * T_tag
        //       This expresses every tag's pose relative to the origin tag.
        return detections.map { pose ->
            if (pose.tagId == TagConfig.ORIGIN_TAG_ID) {
                // Origin tag is identity in its own frame
                TagPose(pose.tagId, IDENTITY_4X4, pose.timestampMs)
            } else {
                // Pass-through for now — replace with actual transform
                pose
            }
        }
    }

    private val IDENTITY_4X4 = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f
    )
}
