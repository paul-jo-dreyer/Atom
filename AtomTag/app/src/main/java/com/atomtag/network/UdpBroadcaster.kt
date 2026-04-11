package com.atomtag.network

import com.atomtag.model.PoseVector
import com.atomtag.model.TagConfig
import java.net.DatagramPacket
import java.net.InetAddress
import java.net.MulticastSocket

/**
 * Broadcasts the pose vector to all devices via UDP multicast.
 * All listeners join the multicast group to receive updates — no per-device connections needed.
 */
class UdpBroadcaster {

    private var socket: MulticastSocket? = null
    private var group: InetAddress? = null
    @Volatile private var running = false

    fun start() {
        if (running) return
        running = true
        socket = MulticastSocket()
        group = InetAddress.getByName(TagConfig.MULTICAST_GROUP)
    }

    /**
     * Send the current pose vector as a single UDP multicast packet.
     * Call this from a background thread.
     */
    fun broadcast(poseVector: PoseVector) {
        if (!running) return
        val data = poseVector.toBytes()
        val packet = DatagramPacket(data, data.size, group, TagConfig.MULTICAST_PORT)
        socket?.send(packet)
    }

    fun stop() {
        running = false
        socket?.close()
        socket = null
    }

    val isRunning: Boolean get() = running
}
