package com.atomtag.ui.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.atomtag.data.AppMode
import com.atomtag.data.DeviceState
import com.atomtag.data.DeviceStateRepository
import com.atomtag.data.MockDeviceStateRepository
import com.atomtag.data.PingResult
import com.atomtag.data.RestartState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.random.Random

data class DashboardUiState(
    val devices: List<DeviceState> = emptyList(),
    val nowMs: Long = System.currentTimeMillis(),
)

data class ApplyState(
    val inProgress: Boolean = false,
    val pendingMode: AppMode? = null,
    val totalDevices: Int = 0,
    val ackedCount: Int = 0,
    val timedOut: Boolean = false,
)

class DevicesViewModel(
    private val repository: DeviceStateRepository = MockDeviceStateRepository(),
) : ViewModel() {

    val state: StateFlow<DashboardUiState> = combine(
        repository.devices,
        clockTicker(),
    ) { devices, now ->
        DashboardUiState(devices = devices, nowMs = now)
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000L),
        initialValue = DashboardUiState(),
    )

    private val _selectedMode = MutableStateFlow(AppMode.Sandbox)
    val selectedMode: StateFlow<AppMode> = _selectedMode.asStateFlow()

    private val _applyState = MutableStateFlow(ApplyState())
    val applyState: StateFlow<ApplyState> = _applyState.asStateFlow()

    private val _pingResults = MutableStateFlow<Map<Int, PingResult>>(emptyMap())
    val pingResults: StateFlow<Map<Int, PingResult>> = _pingResults.asStateFlow()

    private val _restartStates = MutableStateFlow<Map<Int, RestartState>>(emptyMap())
    val restartStates: StateFlow<Map<Int, RestartState>> = _restartStates.asStateFlow()

    init {
        repository.start()
        viewModelScope.launch {
            repository.devices.collect { devices ->
                val current = _applyState.value
                if (current.inProgress) {
                    val acked = devices.count { it.mode == current.pendingMode }
                    if (acked != current.ackedCount) {
                        _applyState.value = current.copy(ackedCount = acked)
                    }
                    if (acked >= current.totalDevices && current.totalDevices > 0) {
                        _applyState.value = current.copy(
                            inProgress = false,
                            ackedCount = acked,
                        )
                    }
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        repository.stop()
    }

    fun selectMode(mode: AppMode) {
        if (_selectedMode.value == mode) return
        _selectedMode.value = mode
        _applyState.value = ApplyState()
    }

    fun applySelectedMode() {
        val mode = _selectedMode.value
        val devices = state.value.devices
        if (devices.isEmpty()) return
        _applyState.value = ApplyState(
            inProgress = true,
            pendingMode = mode,
            totalDevices = devices.size,
            ackedCount = devices.count { it.mode == mode },
            timedOut = false,
        )
        repository.applyMode(mode)
        viewModelScope.launch {
            delay(APPLY_TIMEOUT_MS)
            val current = _applyState.value
            if (current.inProgress && current.pendingMode == mode) {
                _applyState.value = current.copy(
                    inProgress = false,
                    timedOut = current.ackedCount < current.totalDevices,
                )
            }
        }
    }

    fun setName(tagId: Int, name: String) {
        repository.setName(tagId, name)
    }

    fun setColor(tagId: Int, colorArgb: Int) {
        repository.setColor(tagId, colorArgb)
    }

    fun setTeam(tagId: Int, team: String?) {
        repository.setTeam(tagId, team)
    }

    fun ping(tagId: Int) {
        viewModelScope.launch {
            delay(150L + Random.nextLong(700L))
            val success = Random.nextFloat() > 0.2f
            _pingResults.update {
                it + (tagId to PingResult(System.currentTimeMillis(), success))
            }
        }
    }

    fun restart(tagId: Int) {
        viewModelScope.launch {
            val started = System.currentTimeMillis()
            _restartStates.update { it + (tagId to RestartState.Pending(started)) }

            val willAck = Random.nextFloat() > 0.3f
            val ackDelay = if (willAck) 3_000L + Random.nextLong(5_000L) else RestartState.TIMEOUT_MS + 5_000L
            val waitMs = minOf(ackDelay, RestartState.TIMEOUT_MS)
            delay(waitMs)

            val current = _restartStates.value[tagId]
            if (current is RestartState.Pending && current.startedAtMs == started) {
                val resolved = if (willAck) {
                    RestartState.Acknowledged(started, System.currentTimeMillis())
                } else {
                    RestartState.TimedOut(started)
                }
                _restartStates.update { it + (tagId to resolved) }
            }
        }
    }

    fun clearCompletedRestart(tagId: Int) {
        _restartStates.update { current ->
            when (current[tagId]) {
                is RestartState.Acknowledged, is RestartState.TimedOut -> current - tagId
                else -> current
            }
        }
    }

    private fun clockTicker() = flow {
        while (true) {
            emit(System.currentTimeMillis())
            delay(CLOCK_TICK_MS)
        }
    }

    companion object {
        private const val CLOCK_TICK_MS = 500L
        private const val APPLY_TIMEOUT_MS = 8_000L
    }
}
