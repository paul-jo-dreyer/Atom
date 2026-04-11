package com.atomtag.model

/**
 * Central configuration. Edit these values to adjust the system.
 */
object TagConfig {
    /** How many tags the system tracks. Indices 0..NUM_TAGS-1 in the pose vector. */
    const val NUM_TAGS = 2

    /** Tag ID that defines the world-frame origin. All poses are transformed relative to this. */
    const val ORIGIN_TAG_ID = 0

    /** Physical tag size in meters (outer edge to outer edge). Needed for pose estimation. */
    const val TAG_SIZE_METERS = 0.05

    /** UDP multicast group address. All devices join this group to receive pose updates. */
    const val MULTICAST_GROUP = "239.1.1.1"

    /** UDP multicast port. */
    const val MULTICAST_PORT = 5000

    /** Broadcast interval in milliseconds. Limits how often we send UDP packets. */
    const val BROADCAST_INTERVAL_MS = 50L
}
