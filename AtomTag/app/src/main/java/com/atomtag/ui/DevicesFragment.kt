package com.atomtag.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.atomtag.R
import com.atomtag.network.DeviceRegistry

class DevicesFragment : Fragment() {

    private val registry = DeviceRegistry()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        return inflater.inflate(R.layout.fragment_devices, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        val recycler = view.findViewById<RecyclerView>(R.id.deviceList)
        val btnRefresh = view.findViewById<Button>(R.id.btnRefresh)

        val adapter = DeviceAdapter(registry.getDevices())
        recycler.layoutManager = LinearLayoutManager(requireContext())
        recycler.adapter = adapter

        btnRefresh.setOnClickListener {
            registry.refresh()
            adapter.update(registry.getDevices())
        }
    }

    private class DeviceAdapter(
        private var devices: List<DeviceRegistry.Device>
    ) : RecyclerView.Adapter<DeviceAdapter.VH>() {

        class VH(view: View) : RecyclerView.ViewHolder(view) {
            val name: TextView = view.findViewById(R.id.deviceName)
            val ip: TextView = view.findViewById(R.id.deviceIp)
            val status: TextView = view.findViewById(R.id.deviceStatus)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
            val view = LayoutInflater.from(parent.context).inflate(R.layout.item_device, parent, false)
            return VH(view)
        }

        override fun onBindViewHolder(holder: VH, position: Int) {
            val device = devices[position]
            holder.name.text = device.name
            holder.ip.text = device.ip
            holder.status.text = device.status
        }

        override fun getItemCount() = devices.size

        fun update(newDevices: List<DeviceRegistry.Device>) {
            devices = newDevices
            notifyDataSetChanged()
        }
    }
}
