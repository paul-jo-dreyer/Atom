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

    /** Physical ball radius, in meters. Used by the ball detector's geometric gate. */
    var BALL_RADIUS_M = 0.028f
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

        val gbox = root["goalie_box"] as? Map<*, *>
        GOALIE_BOX_WIDTH_M = mm(gbox, "width_mm")
        GOALIE_BOX_HEIGHT_M = mm(gbox, "height_mm")
        GOALIE_BOX_CORNER_RADIUS_M = mm(gbox, "corner_radius_mm")

        val ball = root["ball"] as? Map<*, *>
        BALL_RADIUS_M = if (ball != null) mm(ball, "radius_mm") else 0.028f

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
    }

    private fun mm(obj: Map<*, *>?, key: String): Float =
        (asDouble(obj?.get(key)) / 1000.0).toFloat()

    private fun asDouble(v: Any?): Double = when (v) {
        is Number -> v.toDouble()
        is String -> v.toDoubleOrNull() ?: 0.0
        else -> 0.0
    }
}
