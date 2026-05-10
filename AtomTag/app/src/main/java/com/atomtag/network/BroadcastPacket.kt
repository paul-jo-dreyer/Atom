package com.atomtag.network

import com.atomtag.data.AppMode
import com.atomtag.detection.StateTracker
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Serializes a frame of state into the AtomTag broadcast wire format.
 *
 * Layout (little-endian):
 *
 *   Header (16 bytes):
 *     uint32  magic         = 'ATOM' (0x4D4F5441 read LE)
 *     uint8   version       = 1
 *     uint8   mode          // AppMode ordinal
 *     uint16  flags         // bit 0: ball_present, bit 1: origin_visible
 *     uint64  timestamp_us  // sender wall-clock at send time
 *
 *   Ball (20 bytes):
 *     uint8   present       // 1 if ball detected this frame
 *     uint8   reserved[3]   // alignment
 *     float32 bx, by        // m, field frame (zeroed if !present)
 *     float32 dbx, dby      // m/s
 *
 *   Robots (28 bytes × N, slots ordered by tag_id ascending):
 *     uint8   tag_id        // sanity check; matches expected slot order
 *     uint8   present       // 1 if detected this frame
 *     uint8   command_bits  // bit 0: reset, bit 1: zero_theta, 2-7 reserved
 *     uint8   reserved
 *     float32 rx, ry        // m
 *     float32 rtheta        // rad
 *     float32 drx, dry      // m/s
 *     float32 dtheta        // rad/s
 *
 * Total for 6 bots: 16 + 20 + 6×28 = 204 bytes.
 */
object BroadcastPacket {

    private const val MAGIC = 0x4D4F5441  // 'ATOM' read little-endian
    private const val VERSION: Byte = 1

    private const val HEADER_SIZE = 16
    private const val BALL_SIZE = 20
    private const val ROBOT_SIZE = 28

    const val FLAG_BALL_PRESENT: Int = 0x0001
    const val FLAG_ORIGIN_VISIBLE: Int = 0x0002

    const val COMMAND_RESET: Int = 0x01
    const val COMMAND_ZERO_THETA: Int = 0x02

    fun packetSizeFor(numRobots: Int): Int = HEADER_SIZE + BALL_SIZE + numRobots * ROBOT_SIZE

    /**
     * Build the wire packet. [robotStates] must already be sorted by tag_id
     * ascending; the order on the wire matches the iteration order.
     */
    fun build(
        timestampUs: Long,
        mode: AppMode,
        originVisible: Boolean,
        ball: StateTracker.BallState,
        robotStates: List<Pair<Int, StateTracker.RobotState>>,
        commandBits: Map<Int, Int> = emptyMap(),
    ): ByteArray {
        val size = packetSizeFor(robotStates.size)
        val buf = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)

        buf.putInt(MAGIC)
        buf.put(VERSION)
        buf.put(mode.ordinal.toByte())
        var flags = 0
        if (ball.present) flags = flags or FLAG_BALL_PRESENT
        if (originVisible) flags = flags or FLAG_ORIGIN_VISIBLE
        buf.putShort(flags.toShort())
        buf.putLong(timestampUs)

        buf.put(if (ball.present) 1 else 0)
        buf.put(0); buf.put(0); buf.put(0)
        buf.putFloat(ball.x); buf.putFloat(ball.y)
        buf.putFloat(ball.dx); buf.putFloat(ball.dy)

        for ((tagId, state) in robotStates) {
            buf.put(tagId.toByte())
            buf.put(if (state.present) 1 else 0)
            buf.put((commandBits[tagId] ?: 0).toByte())
            buf.put(0)
            buf.putFloat(state.x); buf.putFloat(state.y)
            buf.putFloat(state.theta)
            buf.putFloat(state.dx); buf.putFloat(state.dy)
            buf.putFloat(state.dtheta)
        }

        return buf.array()
    }
}
