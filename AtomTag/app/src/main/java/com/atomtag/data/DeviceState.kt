package com.atomtag.data

import com.atomtag.model.TagPose

enum class DeviceStatus { Online, Stale, Offline }

data class DeviceState(
    val tagId: Int,
    val name: String,
    val colorArgb: Int,
    val batteryVolts: Float?,
    val lastSeenMs: Long?,
    val pose: TagPose?,
) {
    fun statusAt(nowMs: Long): DeviceStatus {
        val seen = lastSeenMs ?: return DeviceStatus.Offline
        val age = nowMs - seen
        return when {
            age < ONLINE_THRESHOLD_MS -> DeviceStatus.Online
            age < OFFLINE_THRESHOLD_MS -> DeviceStatus.Stale
            else -> DeviceStatus.Offline
        }
    }

    companion object {
        const val ONLINE_THRESHOLD_MS = 2_000L
        const val OFFLINE_THRESHOLD_MS = 10_000L
    }
}
