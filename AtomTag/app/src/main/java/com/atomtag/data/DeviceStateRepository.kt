package com.atomtag.data

import kotlinx.coroutines.flow.StateFlow

interface DeviceStateRepository {
    val devices: StateFlow<List<DeviceState>>
    fun start()
    fun stop()
    fun setName(tagId: Int, name: String)
    fun setColor(tagId: Int, colorArgb: Int)
    fun setTeam(tagId: Int, team: String?)

    /**
     * Begin propagating a mode to all known devices. In the real implementation
     * this would broadcast a UDP "set mode" message; devices acknowledge by reporting
     * their current mode back, which appears in [DeviceState.mode]. The mock simulates
     * per-device ack timing and the same observable update pattern.
     */
    fun applyMode(mode: AppMode)
}
