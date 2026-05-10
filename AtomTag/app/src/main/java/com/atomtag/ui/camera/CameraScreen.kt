package com.atomtag.ui.camera

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.atomtag.model.TagConfig
import com.atomtag.service.CameraStats
import com.atomtag.service.DetectionService
import com.atomtag.ui.AxisOverlayView
import com.atomtag.ui.LocalDetectionService

@Composable
fun CameraScreen(modifier: Modifier = Modifier) {
    val service = LocalDetectionService.current
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        when {
            !hasPermission -> PermissionPrompt(
                onRequest = { permissionLauncher.launch(Manifest.permission.CAMERA) }
            )
            service == null -> Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("Connecting to detection service…")
            }
            else -> CameraContent(service = service)
        }
    }
}

@Composable
private fun CameraContent(service: DetectionService) {
    val context = LocalContext.current
    val previewView = remember { PreviewView(context) }
    val overlayView = remember { AxisOverlayView(context) }

    DisposableEffect(service) {
        service.attachPreview(previewView)
        service.attachOverlay(overlayView)
        onDispose {
            service.detachPreview()
            service.detachOverlay()
        }
    }

    val drawAxes by service.drawAxes.collectAsStateWithLifecycle()
    val drawLabels by service.drawLabels.collectAsStateWithLifecycle()
    val showVirtualBg by service.showVirtualBackground.collectAsStateWithLifecycle()
    val stats by service.stats.collectAsStateWithLifecycle()

    Column(modifier = Modifier.fillMaxSize()) {
        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
            AndroidView(factory = { overlayView }, modifier = Modifier.fillMaxSize())
        }

        ToggleChipRow(
            drawAxes = drawAxes,
            drawLabels = drawLabels,
            showVirtualBg = showVirtualBg,
            onAxes = service::setDrawAxes,
            onLabels = service::setDrawLabels,
            onVirtualBg = service::setShowVirtualBackground,
        )

        StatsStrip(stats = stats)
    }
}

@Composable
private fun PermissionPrompt(onRequest: () -> Unit) {
    Box(modifier = Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "Camera permission required",
                style = MaterialTheme.typography.titleMedium,
            )
            Spacer(Modifier.height(12.dp))
            Text(
                text = "AtomTag needs camera access to detect AprilTags.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(20.dp))
            Button(onClick = onRequest) { Text("Grant access") }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ToggleChipRow(
    drawAxes: Boolean,
    drawLabels: Boolean,
    showVirtualBg: Boolean,
    onAxes: (Boolean) -> Unit,
    onLabels: (Boolean) -> Unit,
    onVirtualBg: (Boolean) -> Unit,
) {
    Surface(color = MaterialTheme.colorScheme.surface, modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            ToggleChip(label = "Axes", selected = drawAxes, onChange = onAxes)
            ToggleChip(label = "Labels", selected = drawLabels, onChange = onLabels)
            ToggleChip(label = "Virtual bg", selected = showVirtualBg, onChange = onVirtualBg)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ToggleChip(label: String, selected: Boolean, onChange: (Boolean) -> Unit) {
    FilterChip(
        selected = selected,
        onClick = { onChange(!selected) },
        label = { Text(label) },
        leadingIcon = if (selected) {
            { Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.height(16.dp)) }
        } else null,
        colors = FilterChipDefaults.filterChipColors(),
    )
}

@Composable
private fun StatsStrip(stats: CameraStats) {
    var expanded by remember { mutableStateOf(false) }
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
    ) {
        Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                val frameLabel = if (stats.originVisible) "tag ${TagConfig.ORIGIN_TAG_ID}" else "cam"
                Text(
                    text = "%.0f fps · %d tags · %.0f Hz · ref:%s".format(
                        stats.fps, stats.tagIds.size, stats.broadcastRateHz, frameLabel,
                    ),
                    modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                )
                IconButton(
                    onClick = { expanded = !expanded },
                    modifier = Modifier.height(32.dp),
                ) {
                    Icon(
                        imageVector = if (expanded) Icons.Default.ExpandMore else Icons.Default.ExpandLess,
                        contentDescription = if (expanded) "Collapse stats" else "Expand stats",
                    )
                }
            }
            if (expanded) {
                Spacer(Modifier.height(4.dp))
                StatLine(label = "FPS", value = "%.1f".format(stats.fps))
                StatLine(label = "Detections", value = stats.tagIds.size.toString())
                StatLine(label = "Broadcast", value = "%.1f Hz".format(stats.broadcastRateHz))
                if (stats.poseLines.isNotEmpty()) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = stats.poseLines.joinToString("\n"),
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun StatLine(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
        Text(
            text = label,
            modifier = Modifier.width(110.dp),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
        )
    }
}
