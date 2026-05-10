package com.atomtag

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import com.atomtag.data.UserPreferences
import com.atomtag.model.FieldConfig
import com.atomtag.model.TagConfig
import com.atomtag.service.DetectionService
import com.atomtag.ui.AppRoot
import com.atomtag.ui.LocalDetectionService
import org.opencv.android.OpenCVLoader

class MainActivity : AppCompatActivity() {

    private var service: DetectionService? by mutableStateOf(null)

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as? DetectionService.LocalBinder)?.service
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
        }
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startAndBindDetectionService()
        else Log.w(TAG, "Camera permission denied; detection service not started")
    }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* ignore — notification just won't show */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        TagConfig.load(this)
        FieldConfig.load(this)
        UserPreferences.init(applicationContext)
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV init failed")
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startAndBindDetectionService()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            CompositionLocalProvider(LocalDetectionService provides service) {
                AppRoot()
            }
        }
    }

    private fun startAndBindDetectionService() {
        val intent = Intent(this, DetectionService::class.java)
        ContextCompat.startForegroundService(this, intent)
        bindService(intent, connection, BIND_AUTO_CREATE)
    }

    override fun onDestroy() {
        super.onDestroy()
        runCatching { unbindService(connection) }
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
