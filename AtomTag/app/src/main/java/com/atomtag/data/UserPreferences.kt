package com.atomtag.data

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Persistent per-tag user overrides written to `device_overrides.json` in the
 * app's internal storage. Initialize once at app startup via [init].
 */
object UserPreferences {

    private const val TAG = "UserPreferences"
    private const val FILE_NAME = "device_overrides.json"

    private lateinit var file: File
    private val cache = mutableMapOf<Int, DeviceOverride>()

    fun init(context: Context) {
        file = File(context.filesDir, FILE_NAME)
        load()
    }

    fun getOverride(tagId: Int): DeviceOverride? = cache[tagId]

    fun setName(tagId: Int, name: String?) {
        update(tagId) { it.copy(name = name?.takeIf { s -> s.isNotBlank() }) }
    }

    fun setColor(tagId: Int, colorArgb: Int?) {
        update(tagId) { it.copy(colorArgb = colorArgb) }
    }

    fun setTeam(tagId: Int, team: String?) {
        update(tagId) { it.copy(team = team?.takeIf { s -> s.isNotBlank() }) }
    }

    private fun update(tagId: Int, transform: (DeviceOverride) -> DeviceOverride) {
        val current = cache[tagId] ?: DeviceOverride()
        val next = transform(current)
        if (next.isEmpty) cache.remove(tagId) else cache[tagId] = next
        save()
    }

    private fun load() {
        cache.clear()
        if (!file.exists()) return
        try {
            val root = JSONObject(file.readText())
            val keys = root.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                val id = key.toIntOrNull() ?: continue
                val obj = root.getJSONObject(key)
                cache[id] = DeviceOverride(
                    name = obj.optString("name").takeIf { it.isNotEmpty() },
                    colorArgb = if (obj.has("color_argb")) obj.getInt("color_argb") else null,
                    team = obj.optString("team").takeIf { it.isNotEmpty() },
                )
            }
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to load $FILE_NAME, ignoring", t)
        }
    }

    private fun save() {
        try {
            val root = JSONObject()
            for ((tagId, override) in cache) {
                if (override.isEmpty) continue
                val obj = JSONObject()
                override.name?.let { obj.put("name", it) }
                override.colorArgb?.let { obj.put("color_argb", it) }
                override.team?.let { obj.put("team", it) }
                root.put(tagId.toString(), obj)
            }
            file.writeText(root.toString(2))
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to save $FILE_NAME", t)
        }
    }
}

data class DeviceOverride(
    val name: String? = null,
    val colorArgb: Int? = null,
    val team: String? = null,
) {
    val isEmpty: Boolean
        get() = name == null && colorArgb == null && team == null
}
