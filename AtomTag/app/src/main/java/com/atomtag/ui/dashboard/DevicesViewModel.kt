package com.atomtag.ui.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
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

    private val _pingResults = MutableStateFlow<Map<Int, PingResult>>(emptyMap())
    val pingResults: StateFlow<Map<Int, PingResult>> = _pingResults.asStateFlow()

    private val _restartStates = MutableStateFlow<Map<Int, RestartState>>(emptyMap())
    val restartStates: StateFlow<Map<Int, RestartState>> = _restartStates.asStateFlow()

    init {
        repository.start()
    }

    override fun onCleared() {
        super.onCleared()
        repository.stop()
    }

    fun setColor(tagId: Int, colorArgb: Int) {
        repository.setColor(tagId, colorArgb)
    }

    /**
     * Drop a completed (Acknowledged or TimedOut) restart entry so the action sheet
     * shows a fresh button next time it opens. Pending restarts are preserved.
     */
    fun clearCompletedRestart(tagId: Int) {
        _restartStates.update { current ->
            when (current[tagId]) {
                is RestartState.Acknowledged, is RestartState.TimedOut -> current - tagId
                else -> current
            }
        }
    }

    /**
     * Stubbed ping: simulates a small RTT and an 80% success rate.
     * Real implementation should send a ping over UDP and listen for a reply.
     */
    fun ping(tagId: Int) {
        viewModelScope.launch {
            delay(150L + Random.nextLong(700L))
            val success = Random.nextFloat() > 0.2f
            _pingResults.update {
                it + (tagId to PingResult(System.currentTimeMillis(), success))
            }
        }
    }

    /**
     * Stubbed restart: ~70% of attempts simulate an ack within 3-8 s; the rest
     * exceed the timeout window and resolve as TimedOut after 20 s.
     */
    fun restart(tagId: Int) {
        viewModelScope.launch {
            val started = System.currentTimeMillis()
            _restartStates.update { it + (tagId to RestartState.Pending(started)) }

            val willAck = Random.nextFloat() > 0.3f
            val ackDelay = if (willAck) 3_000L + Random.nextLong(5_000L) else RestartState.TIMEOUT_MS + 5_000L
            val waitMs = minOf(ackDelay, RestartState.TIMEOUT_MS)
            delay(waitMs)

            // Re-check the current state; user may have triggered another restart.
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

    private fun clockTicker() = flow {
        while (true) {
            emit(System.currentTimeMillis())
            delay(CLOCK_TICK_MS)
        }
    }

    companion object {
        private const val CLOCK_TICK_MS = 500L
    }
}
