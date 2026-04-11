package com.atomtag.model

import android.content.Context
import org.json.JSONObject

/**
 * Central configuration loaded from assets/tag_config.json.
 * Call TagConfig.load(context) once at startup.
 */
object TagConfig {

    var ORIGIN_TAG_ID = 0
        private set

    var MULTICAST_GROUP = "239.1.1.1"
        private set

    var MULTICAST_PORT = 5000
        private set

    var BROADCAST_INTERVAL_MS = 50L
        private set

    /** Per-tag configuration: id -> TagInfo */
    private val tags = mutableMapOf<Int, TagInfo>()

    val NUM_TAGS: Int get() = tags.size

    data class TagInfo(
        val id: Int,
        val sizeMeters: Double,
        val label: String
    )

    fun load(context: Context) {
        val json = context.assets.open("tag_config.json").bufferedReader().readText()
        val root = JSONObject(json)

        ORIGIN_TAG_ID = root.optInt("origin_tag_id", 0)
        MULTICAST_GROUP = root.optString("multicast_group", "239.1.1.1")
        MULTICAST_PORT = root.optInt("multicast_port", 5000)
        BROADCAST_INTERVAL_MS = root.optLong("broadcast_interval_ms", 50L)

        tags.clear()
        val tagsArray = root.getJSONArray("tags")
        for (i in 0 until tagsArray.length()) {
            val tagObj = tagsArray.getJSONObject(i)
            val info = TagInfo(
                id = tagObj.getInt("id"),
                sizeMeters = tagObj.getDouble("size_meters"),
                label = tagObj.optString("label", "tag${tagObj.getInt("id")}")
            )
            tags[info.id] = info
        }
    }

    fun getTagInfo(tagId: Int): TagInfo? = tags[tagId]

    fun getTagSize(tagId: Int): Double = tags[tagId]?.sizeMeters ?: 0.05

    fun isTracked(tagId: Int): Boolean = tags.containsKey(tagId)

    fun allTagIds(): Set<Int> = tags.keys
}
