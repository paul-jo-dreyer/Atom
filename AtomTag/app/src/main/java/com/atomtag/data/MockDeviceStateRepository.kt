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

    override fun setName(tagId: Int, name: String) {
        UserPreferences.setName(tagId, name)
        _devices.update { current ->
            current.map { d -> if (d.tagId == tagId) d.copy(name = name) else d }
        }
    }

    override fun setColor(tagId: Int, colorArgb: Int) {
        UserPreferences.setColor(tagId, colorArgb)
        _devices.update { current ->
            current.map { d -> if (d.tagId == tagId) d.copy(colorArgb = colorArgb) else d }
        }
    }

    override fun setTeam(tagId: Int, team: String?) {
        UserPreferences.setTeam(tagId, team)
        _devices.update { current ->
            current.map { d -> if (d.tagId == tagId) d.copy(team = team) else d }
        }
    }

    override fun applyMode(mode: AppMode) {
        scope.launch {
            for (device in _devices.value) {
                launch {
                    val willAck = Random.nextFloat() < 0.85f
                    if (!willAck) return@launch
                    delay(200L + Random.nextLong(1300L))
                    _devices.update { current ->
                        current.map { d -> if (d.tagId == device.tagId) d.copy(mode = mode) else d }
                    }
                }
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

    private fun buildInitialDevices(): List<DeviceState> {
        val now = System.currentTimeMillis()
        return TagConfig.allTagIds()
            .asSequence()
            .filter { it != TagConfig.ORIGIN_TAG_ID }
            .sorted()
            .map { id ->
                val info = TagConfig.getTagInfo(id)
                val override = UserPreferences.getOverride(id)
                DeviceState(
                    tagId = id,
                    name = override?.name ?: info?.label ?: "tag$id",
                    colorArgb = override?.colorArgb ?: TagColors.argbForIndex(id),
                    batteryVolts = 7.4f + Random.nextFloat() * 0.8f,
                    lastSeenMs = now,
                    pose = null,
                    mode = AppMode.Sandbox,
                    health = mockHealthFor(id),
                    statusMessages = mockMessagesFor(id),
                    team = override?.team,
                )
            }
            .toList()
    }

    private fun mockHealthFor(id: Int): DeviceHealth = when (id) {
        2 -> DeviceHealth.Warning
        4 -> DeviceHealth.Error
        else -> DeviceHealth.Ok
    }

    private fun mockMessagesFor(id: Int): List<String> = when (id) {
        2 -> listOf("Battery below 20%", "Calibration drift exceeds 2°")
        4 -> listOf("Motor 2 stalled", "IMU unresponsive")
        else -> emptyList()
    }

    companion object {
        private const val TICK_MS = 1_000L
    }
}
