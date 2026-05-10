package com.atomtag.data

import kotlinx.coroutines.flow.StateFlow

interface DeviceStateRepository {
    val devices: StateFlow<List<DeviceState>>
    fun start()
    fun stop()
    fun setColor(tagId: Int, colorArgb: Int)
}
