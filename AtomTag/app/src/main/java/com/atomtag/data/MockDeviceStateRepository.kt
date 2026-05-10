package com.atomtag.data

import com.atomtag.model.TagConfig
import com.atomtag.ui.TagColors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.random.Random

class MockDeviceStateRepository : DeviceStateRepository {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var tickJob: Job? = null

    private val colorOverrides = mutableMapOf<Int, Int>()

    private val _devices = MutableStateFlow(buildInitialDevices())
    override val devices: StateFlow<List<DeviceState>> = _devices.asStateFlow()

    override fun start() {
        if (tickJob?.isActive == true) return
        tickJob = scope.launch {
            while (true) {
                delay(TICK_MS)
                tick()
            }
        }
    }

    override fun stop() {
        tickJob?.cancel()
        tickJob = null
    }

    override fun setColor(tagId: Int, colorArgb: Int) {
        colorOverrides[tagId] = colorArgb
        _devices.update { current ->
            current.map { d ->
                if (d.tagId == tagId) d.copy(colorArgb = colorArgb) else d
            }
        }
    }

    private fun tick() {
        val now = System.currentTimeMillis()
        _devices.update { current ->
            current.map { d ->
                d.copy(
                    batteryVolts = d.batteryVolts?.let { drift(it) },
                    lastSeenMs = now,
                )
            }
        }
    }

    private fun drift(volts: Float): Float {
        val delta = -0.004f + Random.nextFloat() * 0.002f
        return (volts + delta).coerceIn(6.0f, 8.4f)
    }

    private fun colorFor(id: Int): Int = colorOverrides[id] ?: TagColors.argbForIndex(id)

    private fun buildInitialDevices(): List<DeviceState> {
        val now = System.currentTimeMillis()
        return TagConfig.allTagIds().sorted().map { id ->
            val info = TagConfig.getTagInfo(id)
            DeviceState(
                tagId = id,
                name = info?.label ?: "tag$id",
                colorArgb = colorFor(id),
                batteryVolts = 7.4f + Random.nextFloat() * 0.8f,
                lastSeenMs = now,
                pose = null,
            )
        }
    }

    companion object {
        private const val TICK_MS = 1_000L
    }
}
