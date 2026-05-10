package com.atomtag.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.atomtag.ui.camera.CameraScreen
import com.atomtag.ui.dashboard.DashboardScreen
import com.atomtag.ui.settings.SettingsScreen
import com.atomtag.ui.theme.AtomTagTheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppRoot() {
    AtomTagTheme {
        var cameraOpen by remember { mutableStateOf(false) }
        var settingsOpen by remember { mutableStateOf(false) }
        val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

        if (settingsOpen) {
            SettingsScreen(onBack = { settingsOpen = false })
        } else {
            DashboardScreen(
                onCameraClick = { cameraOpen = true },
                onSettingsClick = { settingsOpen = true },
            )
        }

        if (cameraOpen) {
            ModalBottomSheet(
                onDismissRequest = { cameraOpen = false },
                sheetState = sheetState,
            ) {
                CameraScreen(modifier = Modifier.fillMaxSize())
            }
        }
    }
}
