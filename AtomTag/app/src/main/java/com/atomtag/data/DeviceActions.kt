package com.atomtag.data

data class PingResult(
    val timestampMs: Long,
    val success: Boolean,
)

sealed class RestartState {
    abstract val startedAtMs: Long

    data class Pending(override val startedAtMs: Long) : RestartState()
    data class Acknowledged(override val startedAtMs: Long, val ackAtMs: Long) : RestartState()
    data class TimedOut(override val startedAtMs: Long) : RestartState()

    companion object {
        const val TIMEOUT_MS: Long = 20_000L
    }
}
