package com.atomtag.detection

import kotlin.math.PI

/**
 * Tracks per-tag and ball state over frames, producing position + smoothed
 * velocity estimates suitable for the UDP broadcast packet.
 *
 * Velocity is computed from the position delta between consecutive detections
 * and EMA-smoothed against the previous velocity estimate. Theta velocities
 * use shortest-angle deltas so the ±π wrap doesn't fake a giant spike.
 *
 * The tracker holds last-known state; when a tag is not detected this frame
 * the caller should call [robotNotDetected] (or [ballNotDetected]) which
 * preserves the prior state internally but returns a state with present=false
 * and zeroed values, leaving the receiver to ignore the slot.
 */
class StateTracker(
    private val velocityAlpha: Float = 0.4f,
) {

    data class RobotState(
        val x: Float, val y: Float, val theta: Float,
        val dx: Float, val dy: Float, val dtheta: Float,
        val present: Boolean,
    )

    data class BallState(
        val x: Float, val y: Float,
        val dx: Float, val dy: Float,
        val present: Boolean,
    )

    private data class RobotPrev(
        val x: Float, val y: Float, val theta: Float,
        val timestampMs: Long,
        val dx: Float, val dy: Float, val dtheta: Float,
    )

    private data class BallPrev(
        val x: Float, val y: Float,
        val timestampMs: Long,
        val dx: Float, val dy: Float,
    )

    private val robots = mutableMapOf<Int, RobotPrev>()
    private var ballPrev: BallPrev? = null

    fun updateRobot(tagId: Int, x: Float, y: Float, theta: Float, timestampMs: Long): RobotState {
        val prev = robots[tagId]
        if (prev == null || timestampMs <= prev.timestampMs) {
            // First frame, or duplicate / out-of-order timestamp: no velocity update.
            val (dx, dy, dtheta) = if (prev == null) Triple(0f, 0f, 0f)
                                   else Triple(prev.dx, prev.dy, prev.dtheta)
            robots[tagId] = RobotPrev(x, y, theta, timestampMs, dx, dy, dtheta)
            return RobotState(x, y, theta, dx, dy, dtheta, present = true)
        }
        val dt = (timestampMs - prev.timestampMs) / 1000f
        val dxInst = (x - prev.x) / dt
        val dyInst = (y - prev.y) / dt
        val dthetaInst = shortestAngleDelta(theta, prev.theta) / dt
        val a = velocityAlpha
        val dx = a * dxInst + (1 - a) * prev.dx
        val dy = a * dyInst + (1 - a) * prev.dy
        val dtheta = a * dthetaInst + (1 - a) * prev.dtheta
        robots[tagId] = RobotPrev(x, y, theta, timestampMs, dx, dy, dtheta)
        return RobotState(x, y, theta, dx, dy, dtheta, present = true)
    }

    fun robotNotDetected(): RobotState {
        return RobotState(0f, 0f, 0f, 0f, 0f, 0f, present = false)
    }

    fun updateBall(x: Float, y: Float, timestampMs: Long): BallState {
        val prev = ballPrev
        if (prev == null || timestampMs <= prev.timestampMs) {
            val (dx, dy) = if (prev == null) 0f to 0f else prev.dx to prev.dy
            ballPrev = BallPrev(x, y, timestampMs, dx, dy)
            return BallState(x, y, dx, dy, present = true)
        }
        val dt = (timestampMs - prev.timestampMs) / 1000f
        val dxInst = (x - prev.x) / dt
        val dyInst = (y - prev.y) / dt
        val a = velocityAlpha
        val dx = a * dxInst + (1 - a) * prev.dx
        val dy = a * dyInst + (1 - a) * prev.dy
        ballPrev = BallPrev(x, y, timestampMs, dx, dy)
        return BallState(x, y, dx, dy, present = true)
    }

    fun ballNotDetected(): BallState {
        return BallState(0f, 0f, 0f, 0f, present = false)
    }

    companion object {
        private val TWO_PI = (2 * PI).toFloat()
        private val PI_F = PI.toFloat()

        /** Shortest signed delta from b to a, wrapped to (-π, π]. */
        private fun shortestAngleDelta(a: Float, b: Float): Float {
            var d = a - b
            while (d > PI_F) d -= TWO_PI
            while (d < -PI_F) d += TWO_PI
            return d
        }
    }
}
