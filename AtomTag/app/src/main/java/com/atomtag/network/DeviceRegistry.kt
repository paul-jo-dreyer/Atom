package com.atomtag.network

/**
 * Placeholder for network device discovery.
 * Will eventually scan for devices on the local network and maintain a registry.
 */
class DeviceRegistry {

    data class Device(
        val name: String,
        val ip: String,
        val status: String = "unknown"
    )

    private val devices = mutableListOf<Device>()

    /** Returns mock devices for now. Replace with actual network discovery. */
    fun getDevices(): List<Device> {
        if (devices.isEmpty()) {
            // Mock data — remove when real discovery is implemented
            devices.add(Device("Device A", "192.168.1.100", "mock"))
            devices.add(Device("Device B", "192.168.1.101", "mock"))
        }
        return devices.toList()
    }

    /** Placeholder for network scan. */
    fun refresh() {
        // TODO: Scan network for devices, ping them, update statuses
    }
}
