package com.atomtag.ui

import androidx.compose.runtime.compositionLocalOf
import com.atomtag.service.DetectionService

val LocalDetectionService = compositionLocalOf<DetectionService?> { null }
