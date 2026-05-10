package com.atomtag.network

import com.atomtag.model.TagConfig
import java.net.DatagramPacket
import java.net.InetAddress
import java.net.MulticastSocket

/**
 * Broadcasts a pre-serialized packet to all listeners via UDP multicast.
 * All listeners join the multicast group to receive updates — no per-device
 * connections needed. The packet format is the caller's responsibility
 * (see [BroadcastPacket]).
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

    /** Send the supplied bytes as a single UDP multicast packet. */
    fun broadcast(data: ByteArray) {
        if (!running) return
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
