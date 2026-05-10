package com.atomtag.model

/**
 * Runtime input to the scoreboard projection. The detector consumes this each
 * frame to choose which 7-segment glyphs to light up. There's no game-state
 * source plumbed through the app yet, so the default is an unstarted match.
 */
data class ScoreboardData(
    /** Match clock in milliseconds. Rendered as MM:SS, clamped 0..99:59. */
    val clockMs: Long = 0L,
    /** Orange team score, clamped 0..9 by the renderer. */
    val orangeScore: Int = 0,
    /** Blue team score, clamped 0..9 by the renderer. */
    val blueScore: Int = 0,
)
