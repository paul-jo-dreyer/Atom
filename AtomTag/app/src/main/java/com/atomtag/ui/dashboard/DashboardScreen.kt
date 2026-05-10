package com.atomtag.ui.dashboard

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun DashboardScreen(
    onCameraClick: () -> Unit,
    onSettingsClick: () -> Unit,
    viewModel: DevicesViewModel = viewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val pingResults by viewModel.pingResults.collectAsStateWithLifecycle()
    val restartStates by viewModel.restartStates.collectAsStateWithLifecycle()
    var selectedTagId by remember { mutableStateOf<Int?>(null) }

    Scaffold { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            if (state.devices.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text(
                        text = "No devices configured",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 16.dp, bottom = 96.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(items = state.devices, key = { it.tagId }) { device ->
                        DeviceCard(
                            device = device,
                            nowMs = state.nowMs,
                            onClick = { selectedTagId = device.tagId },
                        )
                    }
                }
            }

            FloatingActionButton(
                onClick = onSettingsClick,
                modifier = Modifier.align(Alignment.BottomStart).padding(16.dp),
            ) {
                Icon(Icons.Default.Settings, contentDescription = "Settings")
            }
            FloatingActionButton(
                onClick = onCameraClick,
                modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp),
            ) {
                Icon(Icons.Default.PhotoCamera, contentDescription = "Open camera")
            }
        }
    }

    val tagId = selectedTagId
    LaunchedEffect(tagId) {
        if (tagId != null) viewModel.clearCompletedRestart(tagId)
    }
    if (tagId != null) {
        val device = state.devices.firstOrNull { it.tagId == tagId }
        if (device != null) {
            DeviceActionSheet(
                device = device,
                pingResult = pingResults[tagId],
                restartState = restartStates[tagId],
                nowMs = state.nowMs,
                onDismiss = { selectedTagId = null },
                onPing = { viewModel.ping(tagId) },
                onRestart = { viewModel.restart(tagId) },
                onColorChange = { viewModel.setColor(tagId, it) },
            )
        } else {
            LaunchedEffect(tagId) { selectedTagId = null }
        }
    }
}
