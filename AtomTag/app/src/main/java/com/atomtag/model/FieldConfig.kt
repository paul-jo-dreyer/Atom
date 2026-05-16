package com.atomtag.model

import android.content.Context
import org.yaml.snakeyaml.Yaml

/**
 * Field-layout config loaded from `assets/field_config.yaml`.
 *
 * Holds everything that describes the physical playing surface — overall
 * dimensions, the rigid-translation offsets from the origin tag (tag 0) to
 * other reference frames, the goalie box, ball geometry, and any line
 * markings to draw on the floor.
 *
 * `TagConfig` continues to own AprilTag identity + multicast plumbing.
 *
 * All public fields are exposed in **meters**; on-disk values are mm.
 * Call [load] once at startup before reading anything.
 */
object FieldConfig {

    /** Field dimensions, in meters. */
    var SIZE_X_M = 0f
        private set
    var SIZE_Y_M = 0f
        private set

    /** Tag0 → field-frame translation, in tag-local coordinates (m). */
    var FIELD_FRAME_X_M = 0f
        private set
    var FIELD_FRAME_Y_M = 0f
        private set
    var FIELD_FRAME_Z_M = 0f
        private set

    /** Tag0 → goalie-box-center translation, in tag-local coordinates (m). */
    var GOALIE_BOX_X_M = 0f
        private set
    var GOALIE_BOX_Y_M = 0f
        private set
    var GOALIE_BOX_Z_M = 0f
        private set

    /** Goalie box footprint dimensions (rounded rectangle on the ground), m. */
    var GOALIE_BOX_WIDTH_M = 0f
        private set
    var GOALIE_BOX_HEIGHT_M = 0f
        private set
    var GOALIE_BOX_CORNER_RADIUS_M = 0f
        private set

    /** Optional fill color for the goalie-box interior. */
    enum class GoalieBoxFill { None, Orange, Blue }
    var GOALIE_BOX_FILL: GoalieBoxFill = GoalieBoxFill.None
        private set

    /** Tag0 → scoreboard-center translation, in tag-local coordinates (m). */
    var SCOREBOARD_X_M = 0f
        private set
    var SCOREBOARD_Y_M = 0f
        private set
    var SCOREBOARD_Z_M = 0f
        private set

    /** Scoreboard rectangle dimensions (flat plate on the tag's plane), m. */
    var SCOREBOARD_WIDTH_M = 0f
        private set
    var SCOREBOARD_HEIGHT_M = 0f
        private set

    /** CCW rotation of the scoreboard about its vertical axis, in degrees.
     *  Applied before translating into tag-local coords; rotates the whole
     *  layout (plate, score blocks, clock + score glyphs) together. */
    var SCOREBOARD_ROTATION_DEG = 0f
        private set

    /** Physical ball radius, in meters. Used by the ball detector's geometric gate. */
    var BALL_RADIUS_M = 0.028f
        private set

    /** Robot body side length, in meters. The body is treated as a cube centered on
     *  the tag's XY origin with the tag mounted on the top face. Used by the overlay
     *  renderer to mask field lines under each robot. */
    var ROBOT_BODY_SIZE_M = 0.060f
        private set

    /** A straight line marking, expressed in field-frame coordinates (m). */
    data class Line(
        val name: String,
        val fromX: Float, val fromY: Float, val fromZ: Float,
        val toX: Float, val toY: Float, val toZ: Float,
    )

    /** All field-line markings, in load order. */
    var LINES: List<Line> = emptyList()
        private set

    // === Turf (green field background with optional mowed-stripe overlay) ===

    enum class TurfStripesAxis { None, Vertical, Horizontal }

    var TURF_ENABLED: Boolean = false
        private set
    /** ARGB int, alpha bits already merged with TURF_ALPHA. */
    var TURF_COLOR_ARGB: Int = 0xB42E7D32.toInt()
        private set
    /** Number of stripes spanning the field along TURF_STRIPES_AXIS. 0 = flat turf. */
    var TURF_STRIPES_N: Int = 0
        private set
    /** ±RGB shift between adjacent stripes (mean stays at TURF_COLOR_ARGB). */
    var TURF_STRIPES_DELTA: Int = 14
        private set
    var TURF_STRIPES_AXIS: TurfStripesAxis = TurfStripesAxis.None
        private set
    /** Turf corner radius in meters. 0 = sharp rectangle. Clamped to min(W/2, H/2)
     *  by the renderer. The mowed stripes are clipped to the rounded boundary. */
    var TURF_CORNER_RADIUS_M: Float = 0f
        private set

    // === Field markings (halfway line, center circle, center dot) ===

    var MARKINGS_ENABLED: Boolean = false
        private set
    /** ARGB int for all white field markings — halfway line, center circle, center dot,
     *  and the existing field lines + goalie-box outline (consolidated). */
    var MARKINGS_COLOR_ARGB: Int = 0xC8FFFFFF.toInt()
        private set
    var MARKINGS_LINE_WIDTH_PX: Float = 4f
        private set
    var MARKINGS_HALFWAY_LINE: Boolean = false
        private set
    /** Center-circle outline radius, in meters. 0 to disable. */
    var MARKINGS_CENTER_CIRCLE_RADIUS_M: Float = 0f
        private set
    /** Center dot fill radius, in meters. 0 to disable. */
    var MARKINGS_CENTER_DOT_RADIUS_M: Float = 0f
        private set

    // === Logo (bitmap rendered as a perspective-correct quad at field center) ===

    var LOGO_ENABLED: Boolean = false
        private set
    var LOGO_ASSET: String? = null
        private set
    var LOGO_WIDTH_M: Float = 0f
        private set
    var LOGO_HEIGHT_M: Float = 0f
        private set
    var LOGO_ALPHA: Int = 220
        private set
    /** Optional tint applied to the logo bitmap via PorterDuff SRC_IN —
     *  replaces every non-transparent pixel with this color while preserving
     *  the bitmap's alpha mask. Null = render bitmap unchanged. */
    var LOGO_TINT_ARGB: Int? = null
        private set

    fun load(context: Context) {
        val text = context.assets.open("field_config.yaml").bufferedReader().readText()
        val root = (Yaml().load<Any?>(text) as? Map<*, *>) ?: emptyMap<String, Any>()

        val size = root["size_mm"] as? Map<*, *>
        SIZE_X_M = mm(size, "x")
        SIZE_Y_M = mm(size, "y")

        val xforms = root["transforms_mm"] as? Map<*, *>
        val tagToField = xforms?.get("tag0_to_field") as? Map<*, *>
        FIELD_FRAME_X_M = mm(tagToField, "x")
        FIELD_FRAME_Y_M = mm(tagToField, "y")
        FIELD_FRAME_Z_M = mm(tagToField, "z")

        val tagToGoalieBox = xforms?.get("tag0_to_goalie_box") as? Map<*, *>
        GOALIE_BOX_X_M = mm(tagToGoalieBox, "x")
        GOALIE_BOX_Y_M = mm(tagToGoalieBox, "y")
        GOALIE_BOX_Z_M = mm(tagToGoalieBox, "z")

        val tagToScoreboard = xforms?.get("tag0_to_scoreboard") as? Map<*, *>
        SCOREBOARD_X_M = mm(tagToScoreboard, "x")
        SCOREBOARD_Y_M = mm(tagToScoreboard, "y")
        SCOREBOARD_Z_M = mm(tagToScoreboard, "z")

        val gbox = root["goalie_box"] as? Map<*, *>
        GOALIE_BOX_WIDTH_M = mm(gbox, "width_mm")
        GOALIE_BOX_HEIGHT_M = mm(gbox, "height_mm")
        GOALIE_BOX_CORNER_RADIUS_M = mm(gbox, "corner_radius_mm")
        GOALIE_BOX_FILL = parseFill(gbox?.get("fill_color"))

        val scoreboard = root["scoreboard"] as? Map<*, *>
        SCOREBOARD_WIDTH_M = mm(scoreboard, "width_mm")
        SCOREBOARD_HEIGHT_M = mm(scoreboard, "height_mm")
        SCOREBOARD_ROTATION_DEG = asDouble(scoreboard?.get("rotation_deg")).toFloat()

        val ball = root["ball"] as? Map<*, *>
        BALL_RADIUS_M = if (ball != null) mm(ball, "radius_mm") else 0.028f

        val robot = root["robot"] as? Map<*, *>
        ROBOT_BODY_SIZE_M = if (robot != null) mm(robot, "body_mm") else 0.060f

        val linesNode = root["lines"] as? List<*>
        val parsedLines = mutableListOf<Line>()
        if (linesNode != null) {
            for ((i, item) in linesNode.withIndex()) {
                val obj = item as? Map<*, *> ?: continue
                val from = obj["from_mm"] as? List<*> ?: continue
                val to = obj["to_mm"] as? List<*> ?: continue
                if (from.size < 3 || to.size < 3) continue
                parsedLines += Line(
                    name  = (obj["name"] as? String) ?: "line$i",
                    fromX = (asDouble(from[0]) / 1000.0).toFloat(),
                    fromY = (asDouble(from[1]) / 1000.0).toFloat(),
                    fromZ = (asDouble(from[2]) / 1000.0).toFloat(),
                    toX   = (asDouble(to[0]) / 1000.0).toFloat(),
                    toY   = (asDouble(to[1]) / 1000.0).toFloat(),
                    toZ   = (asDouble(to[2]) / 1000.0).toFloat(),
                )
            }
        }
        LINES = parsedLines.toList()

        val turf = root["turf"] as? Map<*, *>
        if (turf != null) {
            TURF_ENABLED = asBool(turf["enabled"], default = true)
            TURF_COLOR_ARGB = mergeAlpha(
                parseHexColor(turf["color"], default = 0xFF2E7D32.toInt()),
                asInt(turf["alpha"], default = 180),
            )
            TURF_STRIPES_N = asInt(turf["mowed_stripes_n"], default = 0)
            TURF_STRIPES_DELTA = asInt(turf["mowed_stripes_delta"], default = 14)
            TURF_STRIPES_AXIS = parseStripesAxis(turf["mowed_stripes_axis"])
            TURF_CORNER_RADIUS_M = mm(turf, "corner_radius_mm")
        }

        val markings = root["markings"] as? Map<*, *>
        if (markings != null) {
            MARKINGS_ENABLED = asBool(markings["enabled"], default = true)
            MARKINGS_COLOR_ARGB = mergeAlpha(
                parseHexColor(markings["color"], default = 0xFFFFFFFF.toInt()),
                asInt(markings["alpha"], default = 220),
            )
            MARKINGS_LINE_WIDTH_PX = asDouble(markings["line_width_px"]).toFloat().let {
                if (it > 0f) it else 4f
            }
            MARKINGS_HALFWAY_LINE = asBool(markings["halfway_line"], default = false)
            MARKINGS_CENTER_CIRCLE_RADIUS_M = mm(markings, "center_circle_radius_mm")
            MARKINGS_CENTER_DOT_RADIUS_M = mm(markings, "center_dot_radius_mm")
        }

        val logo = root["logo"] as? Map<*, *>
        if (logo != null) {
            LOGO_ENABLED = asBool(logo["enabled"], default = true)
            LOGO_ASSET = (logo["asset"] as? String)?.takeIf { it.isNotBlank() }
            LOGO_WIDTH_M = mm(logo, "width_mm")
            LOGO_HEIGHT_M = mm(logo, "height_mm")
            LOGO_ALPHA = asInt(logo["alpha"], default = 220).coerceIn(0, 255)
            LOGO_TINT_ARGB = (logo["tint"] as? String)?.let { s ->
                if (s.isBlank() || s.equals("none", ignoreCase = true)) null
                else parseHexColor(s, default = 0xFFFFFFFF.toInt())
            }
        }
    }

    private fun mm(obj: Map<*, *>?, key: String): Float =
        (asDouble(obj?.get(key)) / 1000.0).toFloat()

    private fun asDouble(v: Any?): Double = when (v) {
        is Number -> v.toDouble()
        is String -> v.toDoubleOrNull() ?: 0.0
        else -> 0.0
    }

    private fun parseFill(v: Any?): GoalieBoxFill = when ((v as? String)?.lowercase()) {
        "orange" -> GoalieBoxFill.Orange
        "blue" -> GoalieBoxFill.Blue
        else -> GoalieBoxFill.None
    }

    private fun parseStripesAxis(v: Any?): TurfStripesAxis = when ((v as? String)?.lowercase()) {
        "vertical" -> TurfStripesAxis.Vertical
        "horizontal" -> TurfStripesAxis.Horizontal
        else -> TurfStripesAxis.None
    }

    /** Parse `"#RRGGBB"` or `"#AARRGGBB"` into a 0xAARRGGBB Int. Alpha defaults
     *  to `0xFF` for 6-digit forms. Returns [default] on null / malformed input. */
    private fun parseHexColor(v: Any?, default: Int): Int {
        val s = (v as? String)?.trim() ?: return default
        if (s.isEmpty()) return default
        val hex = if (s.startsWith("#")) s.substring(1) else s
        return try {
            when (hex.length) {
                6 -> ("FF$hex").toLong(16).toInt()
                8 -> hex.toLong(16).toInt()
                else -> default
            }
        } catch (e: NumberFormatException) {
            default
        }
    }

    /** Replace the alpha channel of [rgb] with [alpha] (0..255). */
    private fun mergeAlpha(rgb: Int, alpha: Int): Int =
        (alpha.coerceIn(0, 255) shl 24) or (rgb and 0x00FFFFFF)

    private fun asInt(v: Any?, default: Int): Int = when (v) {
        is Number -> v.toInt()
        is String -> v.toIntOrNull() ?: default
        else -> default
    }

    private fun asBool(v: Any?, default: Boolean): Boolean = when (v) {
        is Boolean -> v
        is String -> v.equals("true", ignoreCase = true) || v == "1"
        is Number -> v.toInt() != 0
        else -> default
    }
}
