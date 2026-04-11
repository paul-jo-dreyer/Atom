package com.atomtag.model

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Thread-safe container for the current poses of all tracked tags.
 * Index corresponds to tag ID.
 */
class PoseVector(private val size: Int = TagConfig.NUM_TAGS) {

    private val poses = arrayOfNulls<TagPose>(size)
    private val lock = Any()

    fun update(pose: TagPose) {
        require(pose.tagId in 0 until size) { "Tag ID ${pose.tagId} out of range [0, $size)" }
        synchronized(lock) {
            poses[pose.tagId] = pose
        }
    }

    fun get(tagId: Int): TagPose? {
        synchronized(lock) {
            return poses[tagId]
        }
    }

    fun snapshot(): Array<TagPose?> {
        synchronized(lock) {
            return poses.copyOf()
        }
    }

    /**
     * Serialize the full pose vector to bytes for UDP transmission.
     * Format: [numTags: int] then for each slot:
     *   [present: byte] [tagId: int] [transform: 16 floats]
     */
    fun toBytes(): ByteArray {
        synchronized(lock) {
            // 4 (numTags) + for each: 1 (present) + 4 (tagId) + 64 (16 floats)
            val buf = ByteBuffer.allocate(4 + size * (1 + 4 + 64))
            buf.order(ByteOrder.LITTLE_ENDIAN)
            buf.putInt(size)
            for (i in 0 until size) {
                val pose = poses[i]
                if (pose != null) {
                    buf.put(1.toByte())
                    buf.putInt(pose.tagId)
                    for (v in pose.transform) buf.putFloat(v)
                } else {
                    buf.put(0.toByte())
                    buf.putInt(i)
                    repeat(16) { buf.putFloat(0f) }
                }
            }
            return buf.array()
        }
    }
}
