package com.atomtag.ui.dashboard

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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.atomtag.data.AppMode
import com.atomtag.ui.theme.StatusOffline
import com.atomtag.ui.theme.StatusOnline

@Composable
fun ModeSelectorPanel(
    selected: AppMode,
    onSelect: (AppMode) -> Unit,
    applyState: ApplyState,
    onApply: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surface,
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Mode",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                ModeDropdown(
                    selected = selected,
                    onSelect = onSelect,
                    enabled = !applyState.inProgress,
                    modifier = Modifier.weight(1f),
                )
                ApplyButton(
                    state = applyState,
                    onClick = onApply,
                )
            }
        }
    }
}

@Composable
private fun ModeDropdown(
    selected: AppMode,
    onSelect: (AppMode) -> Unit,
    enabled: Boolean,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }
    Box(modifier = modifier) {
        OutlinedButton(
            onClick = { expanded = true },
            enabled = enabled,
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
        ) {
            Text(
                text = selected.label,
                modifier = Modifier.weight(1f),
                style = MaterialTheme.typography.bodyLarge,
            )
            Icon(Icons.Default.ArrowDropDown, contentDescription = null)
        }
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            for (mode in AppMode.values()) {
                DropdownMenuItem(
                    text = { Text(mode.label) },
                    onClick = {
                        onSelect(mode)
                        expanded = false
                    },
                )
            }
        }
    }
}

@Composable
private fun ApplyButton(state: ApplyState, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = !state.inProgress,
        shape = RoundedCornerShape(12.dp),
    ) {
        when {
            state.inProgress -> {
                CircularProgressIndicator(
                    modifier = Modifier.size(18.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.onPrimary,
                )
                Spacer(Modifier.width(10.dp))
                Text(text = "${state.ackedCount}/${state.totalDevices}")
            }
            state.pendingMode != null && state.timedOut -> {
                Icon(
                    Icons.Default.ErrorOutline,
                    contentDescription = null,
                    tint = StatusOffline,
                )
                Spacer(Modifier.width(8.dp))
                Text(text = "${state.ackedCount}/${state.totalDevices}")
            }
            state.pendingMode != null && state.ackedCount >= state.totalDevices && state.totalDevices > 0 -> {
                Icon(
                    Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = StatusOnline,
                )
                Spacer(Modifier.width(8.dp))
                Text(text = "${state.ackedCount}/${state.totalDevices}")
            }
            else -> {
                Icon(Icons.Default.Send, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text(text = "Apply")
            }
        }
    }
}
