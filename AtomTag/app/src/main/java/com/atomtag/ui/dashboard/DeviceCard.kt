package com.atomtag.ui.dashboard

import androidx.compose.foundation.Canvas
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
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.atomtag.data.DeviceHealth
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
        Box(modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(
                    modifier = Modifier
                        .size(width = 4.dp, height = 54.dp)
                        .clip(RoundedCornerShape(2.dp))
                        .background(Color(device.colorArgb))
                )
                Spacer(Modifier.width(12.dp))

                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = device.name,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = "tag ${device.tagId}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Column(
                    modifier = Modifier.padding(end = 54.dp),
                    horizontalAlignment = Alignment.Start,
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    StatusChip(status = status, lastSeenMs = device.lastSeenMs, nowMs = nowMs)
                    ModeChip(mode = device.mode?.label ?: "—", health = device.health)
                }
            }

            BatteryReadout(
                volts = device.batteryVolts,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp),
            )
        }
    }
}

@Composable
private fun BatteryReadout(volts: Float?, modifier: Modifier = Modifier) {
    val fill = batteryFill(volts)
    val borderColor = MaterialTheme.colorScheme.onSurface
    val fillColor = batteryColor(fill)
    Canvas(modifier = modifier.size(width = 28.dp, height = 14.dp)) {
        val w = size.width
        val h = size.height
        val tipW = w * 0.10f
        val bodyW = w - tipW
        val strokePx = 1.2.dp.toPx()
        val outerCorner = CornerRadius(2.dp.toPx())
        val innerCorner = CornerRadius(1.dp.toPx())

        drawRoundRect(
            color = borderColor,
            topLeft = Offset(strokePx / 2f, strokePx / 2f),
            size = Size(bodyW - strokePx, h - strokePx),
            cornerRadius = outerCorner,
            style = Stroke(width = strokePx),
        )
        val tipH = h * 0.5f
        drawRoundRect(
            color = borderColor,
            topLeft = Offset(bodyW, (h - tipH) / 2f),
            size = Size(tipW, tipH),
            cornerRadius = CornerRadius(0.5.dp.toPx()),
        )

        if (fill > 0f) {
            val padding = strokePx + 1.dp.toPx()
            val maxFillWidth = bodyW - 2f * padding
            val fillWidth = maxFillWidth * fill
            drawRoundRect(
                color = fillColor,
                topLeft = Offset(padding, padding),
                size = Size(fillWidth, h - 2f * padding),
                cornerRadius = innerCorner,
            )
        }
    }
}

private val BatteryRed = Color(0xFFEF4444)
private val BatteryYellow = Color(0xFFFBBF24)
private val BatteryGreen = Color(0xFF34D399)

private fun batteryColor(fill: Float): Color {
    val t = ((fill - 0.2f) / 0.6f).coerceIn(0f, 1f)
    return when {
        t <= 0.5f -> lerp(BatteryRed, BatteryYellow, t * 2f)
        else -> lerp(BatteryYellow, BatteryGreen, (t - 0.5f) * 2f)
    }
}

private const val VOLTAGE_FULL = 8.3f
private const val VOLTAGE_EMPTY = 6.1f

/**
 * 2S LiPo voltage → state-of-charge in [0, 1]. Cubic anchored at:
 *   f(6.1V)=0, f(7.4V)=0.65 (nominal sits above half), f(8.3V)=1,
 *   plus f'(0)=0.4 to model LiPo's flat ends — voltage barely changes near
 *   empty or full, but drops sharply through the middle "discharge plateau".
 * The curve is S-shaped: slow at both ends, steep in the middle.
 * Coefficients computed in normalized t = (V - 6.1) / 2.2.
 */
private fun batteryFill(volts: Float?): Float {
    if (volts == null) return 0f
    if (volts >= VOLTAGE_FULL) return 1f
    if (volts <= VOLTAGE_EMPTY) return 0f
    val t = (volts - VOLTAGE_EMPTY) / (VOLTAGE_FULL - VOLTAGE_EMPTY)
    val t2 = t * t
    val t3 = t2 * t
    return (-1.43f * t3 + 2.03f * t2 + 0.4f * t).coerceIn(0f, 1f)
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

@Composable
private fun ModeChip(mode: String, health: DeviceHealth) {
    val dotColor = when (health) {
        DeviceHealth.Ok -> StatusOnline
        DeviceHealth.Warning -> StatusStale
        DeviceHealth.Error -> StatusOffline
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(dotColor)
        )
        Spacer(Modifier.width(6.dp))
        Text(
            text = mode,
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
