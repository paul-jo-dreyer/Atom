package com.atomtag.ui.dashboard

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.NetworkPing
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import com.atomtag.data.DeviceHealth
import com.atomtag.data.DeviceState
import com.atomtag.data.PingResult
import com.atomtag.data.RestartState
import com.atomtag.ui.theme.StatusOffline
import com.atomtag.ui.theme.StatusOnline
import com.atomtag.ui.theme.StatusStale

private val PresetColors = listOf(
    Color(0xFFEF4444),
    Color(0xFFF97316),
    Color(0xFFF59E0B),
    Color(0xFF10B981),
    Color(0xFF14B8A6),
    Color(0xFF3B82F6),
    Color(0xFF8B5CF6),
    Color(0xFFEC4899),
)

private val ExtendedPalette = listOf(
    Color(0xFFFF0000), Color(0xFFFF4000), Color(0xFFFF8000), Color(0xFFFFBF00),
    Color(0xFFFFFF00), Color(0xFFBFFF00), Color(0xFF80FF00), Color(0xFF40FF00),
    Color(0xFF00FF00), Color(0xFF00FF80), Color(0xFF00FFFF), Color(0xFF0080FF),
    Color(0xFF0000FF), Color(0xFF8000FF), Color(0xFFFF00FF), Color(0xFFFF0080),
)

private val TeamOptions = listOf("None", "Red", "Blue", "Yellow")

private val RainbowSweep = Brush.sweepGradient(
    colors = listOf(
        Color(0xFFFF0000), Color(0xFFFFFF00), Color(0xFF00FF00),
        Color(0xFF00FFFF), Color(0xFF0000FF), Color(0xFFFF00FF), Color(0xFFFF0000),
    )
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DeviceActionSheet(
    device: DeviceState,
    pingResult: PingResult?,
    restartState: RestartState?,
    nowMs: Long,
    onDismiss: () -> Unit,
    onPing: () -> Unit,
    onRestart: () -> Unit,
    onColorChange: (Int) -> Unit,
    onNameChange: (String) -> Unit,
    onTeamChange: (String?) -> Unit,
) {
    var showSettings by remember(device.tagId) { mutableStateOf(false) }
    var customPickerOpen by remember(device.tagId) { mutableStateOf(false) }
    var editingName by remember(device.tagId) { mutableStateOf(false) }

    val currentColor = Color(device.colorArgb)
    val selectedTeam = device.team ?: "None"

    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Header(
                device = device,
                pingResult = pingResult,
                nowMs = nowMs,
                showSettings = showSettings,
                editingName = editingName,
                onToggleSettings = { showSettings = !showSettings },
                onStartEditName = { editingName = true },
                onCommitName = { newName ->
                    if (newName.isNotBlank() && newName != device.name) onNameChange(newName)
                    editingName = false
                },
                onCancelEditName = { editingName = false },
            )

            Spacer(Modifier.height(4.dp))

            ActionButton(label = "Ping", icon = Icons.Default.NetworkPing, onClick = onPing)
            RestartButton(state = restartState, nowMs = nowMs, onClick = onRestart)

            DeviceStatusSection(
                health = device.health,
                messages = device.statusMessages,
            )

            if (showSettings) {
                Spacer(Modifier.height(8.dp))
                HorizontalDivider()
                Spacer(Modifier.height(4.dp))

                if (customPickerOpen) {
                    InlineCustomColorPicker(
                        initial = currentColor,
                        onConfirm = {
                            onColorChange(it.toArgb())
                            customPickerOpen = false
                        },
                        onCancel = { customPickerOpen = false },
                    )
                } else {
                    SettingsSection(title = "Color") {
                        ColorPickerRow(
                            currentColor = currentColor,
                            onPickPreset = { onColorChange(it.toArgb()) },
                            onPickCustom = { customPickerOpen = true },
                        )
                    }

                    SettingsSection(title = "Team") {
                        TeamSelectorRow(
                            options = TeamOptions,
                            selected = selectedTeam,
                            onSelect = { team ->
                                onTeamChange(if (team == "None") null else team)
                            },
                        )
                    }
                }
            }

            Spacer(Modifier.height(12.dp))
        }
    }
}

@Composable
private fun Header(
    device: DeviceState,
    pingResult: PingResult?,
    nowMs: Long,
    showSettings: Boolean,
    editingName: Boolean,
    onToggleSettings: () -> Unit,
    onStartEditName: () -> Unit,
    onCommitName: (String) -> Unit,
    onCancelEditName: () -> Unit,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(20.dp)
                .clip(CircleShape)
                .background(Color(device.colorArgb))
        )
        Spacer(Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            if (editingName) {
                NameEditRow(
                    initial = device.name,
                    onCommit = onCommitName,
                    onCancel = onCancelEditName,
                )
            } else {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = device.name,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f, fill = false),
                    )
                    IconButton(
                        onClick = onStartEditName,
                        modifier = Modifier.size(28.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Default.Edit,
                            contentDescription = "Rename device",
                            modifier = Modifier.size(16.dp),
                        )
                    }
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "tag ${device.tagId}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (pingResult != null) {
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = "·",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.width(8.dp))
                    val ageS = ((nowMs - pingResult.timestampMs) / 1000L).coerceAtLeast(0L)
                    val ageText = if (ageS < 1L) "now" else "${ageS}s ago"
                    val (color, mark) = if (pingResult.success) {
                        StatusOnline to "✓"
                    } else {
                        StatusOffline to "✗"
                    }
                    Text(
                        text = "ping $mark $ageText",
                        style = MaterialTheme.typography.bodyMedium,
                        color = color,
                    )
                }
            }
        }
        IconButton(
            onClick = onToggleSettings,
            colors = if (showSettings) {
                IconButtonDefaults.iconButtonColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = MaterialTheme.colorScheme.onPrimary,
                )
            } else {
                IconButtonDefaults.iconButtonColors()
            },
        ) {
            Icon(Icons.Default.Tune, contentDescription = "Device settings")
        }
    }
}

@Composable
private fun NameEditRow(
    initial: String,
    onCommit: (String) -> Unit,
    onCancel: () -> Unit,
) {
    var text by remember(initial) { mutableStateOf(initial) }
    Row(verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            singleLine = true,
            modifier = Modifier.weight(1f),
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
            keyboardActions = KeyboardActions(onDone = { onCommit(text) }),
        )
        IconButton(
            onClick = { onCommit(text) },
            modifier = Modifier.size(40.dp),
        ) {
            Icon(Icons.Default.Check, contentDescription = "Save name")
        }
        IconButton(
            onClick = onCancel,
            modifier = Modifier.size(40.dp),
        ) {
            Icon(Icons.Default.Close, contentDescription = "Cancel")
        }
    }
}

@Composable
private fun ActionButton(label: String, icon: ImageVector, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
    ) {
        Icon(imageVector = icon, contentDescription = null)
        Spacer(Modifier.width(12.dp))
        Text(text = label, style = MaterialTheme.typography.bodyLarge)
    }
}

@Composable
private fun RestartButton(state: RestartState?, nowMs: Long, onClick: () -> Unit) {
    val isPending = state is RestartState.Pending
    OutlinedButton(
        onClick = onClick,
        enabled = !isPending,
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
    ) {
        when (state) {
            is RestartState.Pending -> {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    strokeWidth = 2.dp,
                )
                Spacer(Modifier.width(12.dp))
                val elapsed = ((nowMs - state.startedAtMs) / 1000L).coerceAtLeast(0L)
                Text(text = "Restarting… ${elapsed}s", style = MaterialTheme.typography.bodyLarge)
            }
            is RestartState.Acknowledged -> {
                Icon(Icons.Default.CheckCircle, contentDescription = null, tint = StatusOnline)
                Spacer(Modifier.width(12.dp))
                val ageS = ((nowMs - state.ackAtMs) / 1000L).coerceAtLeast(0L)
                val ageText = if (ageS < 1L) "just now" else "${ageS}s ago"
                Text(text = "Restarted $ageText", style = MaterialTheme.typography.bodyLarge)
            }
            is RestartState.TimedOut -> {
                Icon(Icons.Default.ErrorOutline, contentDescription = null, tint = StatusOffline)
                Spacer(Modifier.width(12.dp))
                Text(text = "Restart timed out", style = MaterialTheme.typography.bodyLarge)
            }
            null -> {
                Icon(Icons.Default.RestartAlt, contentDescription = null)
                Spacer(Modifier.width(12.dp))
                Text(text = "Restart", style = MaterialTheme.typography.bodyLarge)
            }
        }
    }
}

@Composable
private fun DeviceStatusSection(health: DeviceHealth, messages: List<String>) {
    val (label, color) = when (health) {
        DeviceHealth.Ok -> "Good" to StatusOnline
        DeviceHealth.Warning -> "Warning" to StatusStale
        DeviceHealth.Error -> "Error" to StatusOffline
    }
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .clip(CircleShape)
                    .background(color)
            )
            Spacer(Modifier.width(8.dp))
            Text(
                text = "Status: $label",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
        }
        if (messages.isNotEmpty()) {
            Spacer(Modifier.height(6.dp))
            for (message in messages) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                    verticalAlignment = Alignment.Top,
                ) {
                    Text(
                        text = "•",
                        modifier = Modifier.padding(end = 8.dp),
                        color = color,
                    )
                    Text(
                        text = message,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        } else {
            Text(
                text = "No reported issues.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
    }
}

@Composable
private fun SettingsSection(title: String, content: @Composable () -> Unit) {
    Column {
        Text(
            text = title,
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.padding(bottom = 8.dp),
        )
        content()
    }
}

@Composable
private fun ColorPickerRow(
    currentColor: Color,
    onPickPreset: (Color) -> Unit,
    onPickCustom: () -> Unit,
) {
    val isPreset = PresetColors.any { it.value == currentColor.value }
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        for (color in PresetColors) {
            val selected = color.value == currentColor.value
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(CircleShape)
                    .background(color)
                    .border(
                        width = if (selected) 3.dp else 0.dp,
                        color = MaterialTheme.colorScheme.onSurface,
                        shape = CircleShape,
                    )
                    .clickable { onPickPreset(color) }
            )
        }
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .background(RainbowSweep)
                .border(
                    width = if (!isPreset) 3.dp else 0.dp,
                    color = MaterialTheme.colorScheme.onSurface,
                    shape = CircleShape,
                )
                .clickable(onClick = onPickCustom)
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TeamSelectorRow(
    options: List<String>,
    selected: String,
    onSelect: (String) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        for (option in options) {
            FilterChip(
                selected = option == selected,
                onClick = { onSelect(option) },
                label = { Text(option) },
            )
        }
    }
}

@Composable
private fun InlineCustomColorPicker(
    initial: Color,
    onConfirm: (Color) -> Unit,
    onCancel: () -> Unit,
) {
    var selected by remember(initial) { mutableStateOf(initial) }

    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = "Pick a color",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.padding(bottom = 12.dp),
        )

        for (rowStart in 0 until ExtendedPalette.size step 8) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                for (i in rowStart until rowStart + 8) {
                    val color = ExtendedPalette[i]
                    val isSelected = color.value == selected.value
                    Box(
                        modifier = Modifier
                            .size(32.dp)
                            .clip(CircleShape)
                            .background(color)
                            .border(
                                width = if (isSelected) 3.dp else 0.dp,
                                color = MaterialTheme.colorScheme.onSurface,
                                shape = CircleShape,
                            )
                            .clickable { selected = color }
                    )
                }
            }
        }

        Spacer(Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
        ) {
            TextButton(onClick = onCancel) { Text("Cancel") }
            Spacer(Modifier.width(8.dp))
            TextButton(onClick = { onConfirm(selected) }) { Text("OK") }
        }
    }
}
