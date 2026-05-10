package com.atomtag.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val AtomTagDarkColors = darkColorScheme(
    primary = Primary,
    onPrimary = OnPrimary,
    background = Background,
    onBackground = OnSurface,
    surface = Surface,
    onSurface = OnSurface,
    surfaceVariant = SurfaceVariant,
    onSurfaceVariant = OnSurfaceVariant,
    outline = Outline,
)

@Composable
fun AtomTagTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = AtomTagDarkColors,
        content = content,
    )
}
