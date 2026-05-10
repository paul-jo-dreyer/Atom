package com.atomtag.ui.dashboard

import androidx.compose.foundation.background
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
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.atomtag.data.DeviceState
import com.atomtag.data.DeviceStatus
import com.atomtag.ui.theme.StatusOffline
import com.atomtag.ui.theme.StatusOnline
import com.atomtag.ui.theme.StatusStale

@Composable
fun DeviceCard(
    device: DeviceState,
    nowMs: Long,
    modifier: Modifier = Modifier,
    onClick: () -> Unit = {},
) {
    val status = device.statusAt(nowMs)
    Card(
        modifier = modifier.fillMaxWidth().clickable(onClick = onClick),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .size(width = 4.dp, height = 44.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(Color(device.colorArgb))
            )
            Spacer(Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = device.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = "tag ${device.tagId}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Column(
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                BatteryReadout(volts = device.batteryVolts)
                StatusChip(status = status, lastSeenMs = device.lastSeenMs, nowMs = nowMs)
            }
        }
    }
}

@Composable
private fun BatteryReadout(volts: Float?) {
    val text = volts?.let { "%.2f V".format(it) } ?: "—"
    val color = when {
        volts == null -> MaterialTheme.colorScheme.onSurfaceVariant
        volts >= 7.4f -> StatusOnline
        volts >= 6.6f -> StatusStale
        else -> StatusOffline
    }
    Text(text = text, color = color, style = MaterialTheme.typography.bodyLarge)
}

@Composable
private fun StatusChip(status: DeviceStatus, lastSeenMs: Long?, nowMs: Long) {
    val (dotColor, label) = when (status) {
        DeviceStatus.Online -> StatusOnline to "online"
        DeviceStatus.Stale -> StatusStale to "stale"
        DeviceStatus.Offline -> StatusOffline to "offline"
    }
    val ageText = formatAge(lastSeenMs, nowMs)
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(dotColor)
        )
        Spacer(Modifier.width(6.dp))
        Text(
            text = "$label · $ageText",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private fun formatAge(lastSeenMs: Long?, nowMs: Long): String {
    if (lastSeenMs == null) return "never"
    val ageS = ((nowMs - lastSeenMs) / 1000L).coerceAtLeast(0L)
    return when {
        ageS < 1L -> "now"
        ageS < 60L -> "${ageS}s ago"
        ageS < 3600L -> "${ageS / 60L}m ago"
        else -> "${ageS / 3600L}h ago"
    }
}
