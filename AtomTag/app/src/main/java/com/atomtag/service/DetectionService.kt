package com.atomtag.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import com.atomtag.MainActivity
import com.atomtag.R
import com.atomtag.detection.AprilTagDetector
import com.atomtag.detection.PoseTransformer
import com.atomtag.model.PoseVector
import com.atomtag.model.TagConfig
import com.atomtag.network.UdpBroadcaster
import com.atomtag.ui.AxisOverlayView
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

data class CameraStats(
    val fps: Float = 0f,
    val tagIds: List<Int> = emptyList(),
    val originVisible: Boolean = false,
    val poseLines: List<String> = emptyList(),
    val broadcastRateHz: Float = 0f,
)

class DetectionService : LifecycleService() {

    private val poseVector = PoseVector()
    private val broadcaster = UdpBroadcaster()
    private val detector = AprilTagDetector()
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private var cameraProvider: ProcessCameraProvider? = null
    private var preview: Preview? = null
    private var analysis: ImageAnalysis? = null
    private var overlayRef: AxisOverlayView? = null

    private val _stats = MutableStateFlow(CameraStats())
    val stats: StateFlow<CameraStats> = _stats.asStateFlow()

    private val _drawAxes = MutableStateFlow(false)
    val drawAxes: StateFlow<Boolean> = _drawAxes.asStateFlow()

    private val _drawLabels = MutableStateFlow(true)
    val drawLabels: StateFlow<Boolean> = _drawLabels.asStateFlow()

    private val _showVirtualBackground = MutableStateFlow(false)
    val showVirtualBackground: StateFlow<Boolean> = _showVirtualBackground.asStateFlow()

    private val _isRunning = MutableStateFlow(false)
    val isRunning: StateFlow<Boolean> = _isRunning.asStateFlow()

    fun setDrawAxes(v: Boolean) { _drawAxes.value = v }
    fun setDrawLabels(v: Boolean) { _drawLabels.value = v }
    fun setShowVirtualBackground(v: Boolean) { _showVirtualBackground.value = v }

    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var lastBroadcastTime = 0L
    private var broadcastsInWindow = 0
    private var lastBroadcastWindowStart = 0L

    inner class LocalBinder : Binder() {
        val service: DetectionService = this@DetectionService
    }

    private val binder = LocalBinder()

    override fun onBind(intent: Intent): IBinder {
        super.onBind(intent)
        return binder
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification(0))
        broadcaster.start()
        startCamera()
        _isRunning.value = true
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        if (intent?.action == ACTION_STOP) {
            stopSelf()
        }
        return START_STICKY
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        super.onTaskRemoved(rootIntent)
        stopSelf()
    }

    override fun onDestroy() {
        super.onDestroy()
        _isRunning.value = false
        cameraProvider?.unbindAll()
        cameraProvider = null
        broadcaster.stop()
        analysisExecutor.shutdown()
        detector.release()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider
            preview = Preview.Builder().build()
            analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::analyzeFrame) }
            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis,
                )
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to bind camera", t)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    fun attachPreview(view: PreviewView) {
        preview?.setSurfaceProvider(view.surfaceProvider)
    }

    fun detachPreview() {
        preview?.setSurfaceProvider(null)
    }

    fun attachOverlay(view: AxisOverlayView) {
        overlayRef = view
    }

    fun detachOverlay() {
        overlayRef?.clear()
        overlayRef = null
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        var gray: Mat? = null
        try {
            val rotation = imageProxy.imageInfo.rotationDegrees
            gray = imageProxyToGrayMat(imageProxy)
            if (gray.cols() <= 0 || gray.rows() <= 0) return

            val shouldDrawAxes = _drawAxes.value
            val shouldDrawLabels = _drawLabels.value

            detector.initIntrinsics(gray.cols(), gray.rows())
            val detections = detector.detect(gray, projectAxes = shouldDrawAxes)

            val rawPoses = detections.map { it.pose }
            val transformed = PoseTransformer.transformToFieldFrame(rawPoses)
            val originVisible = rawPoses.any { it.tagId == TagConfig.ORIGIN_TAG_ID }
            for (pose in transformed) poseVector.update(pose)

            val now = System.currentTimeMillis()
            if (now - lastBroadcastTime >= TagConfig.BROADCAST_INTERVAL_MS) {
                lastBroadcastTime = now
                broadcaster.broadcast(poseVector)
                broadcastsInWindow++
            }

            frameCount++
            var fps = _stats.value.fps
            val fpsElapsed = now - lastFpsTime
            if (fpsElapsed >= 1000) {
                fps = frameCount * 1000f / fpsElapsed
                frameCount = 0
                lastFpsTime = now
            }

            var broadcastRate = _stats.value.broadcastRateHz
            if (lastBroadcastWindowStart == 0L) lastBroadcastWindowStart = now
            val winElapsed = now - lastBroadcastWindowStart
            if (winElapsed >= 1000) {
                broadcastRate = broadcastsInWindow * 1000f / winElapsed
                broadcastsInWindow = 0
                lastBroadcastWindowStart = now
                updateNotification(transformed.size)
            }

            _stats.value = CameraStats(
                fps = fps,
                tagIds = transformed.map { it.tagId },
                originVisible = originVisible,
                poseLines = transformed.map { p ->
                    "Tag ${p.tagId}: (${"%.3f".format(p.tx)}, ${"%.3f".format(p.ty)}, ${"%.3f".format(p.tz)})"
                },
                broadcastRateHz = broadcastRate,
            )

            val overlayData = detections.map { det ->
                AxisOverlayView.TagOverlayData(
                    tagId = det.pose.tagId,
                    axisPoints = if (shouldDrawAxes) det.axisPoints else null,
                    bottomCenter = if (shouldDrawLabels) det.bottomCenter else null,
                )
            }
            val fieldAxes = if (shouldDrawAxes) {
                detections.firstOrNull { it.pose.tagId == TagConfig.ORIGIN_TAG_ID }?.fieldFrameAxes
            } else null
            val cols = gray.cols()
            val rows = gray.rows()
            overlayRef?.let { ov ->
                ov.post {
                    if (overlayData.isNotEmpty() || fieldAxes != null) {
                        ov.update(overlayData, cols, rows, rotation, fieldAxes)
                    } else {
                        ov.clear()
                    }
                }
            }
        } finally {
            gray?.release()
            imageProxy.close()
        }
    }

    private fun imageProxyToGrayMat(imageProxy: ImageProxy): Mat {
        val plane = imageProxy.planes[0]
        val buffer: ByteBuffer = plane.buffer
        val rowStride = plane.rowStride
        val width = imageProxy.width
        val height = imageProxy.height

        val mat = Mat(height, width, CvType.CV_8UC1)
        if (rowStride == width) {
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            mat.put(0, 0, bytes)
        } else {
            val bytes = ByteArray(width)
            for (row in 0 until height) {
                buffer.position(row * rowStride)
                buffer.get(bytes, 0, width)
                mat.put(row, 0, bytes)
            }
        }
        return mat
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Detection",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "AtomTag detection and broadcast"
                setShowBadge(false)
            }
            val mgr = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            mgr.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(detectionCount: Int): Notification {
        val openIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stopIntent = PendingIntent.getService(
            this,
            1,
            Intent(this, DetectionService::class.java).apply { action = ACTION_STOP },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("AtomTag broadcasting")
            .setContentText("$detectionCount tag${if (detectionCount == 1) "" else "s"} tracked")
            .setSmallIcon(R.drawable.ic_atomtag_notification)
            .setOngoing(true)
            .setContentIntent(openIntent)
            .addAction(0, "Stop", stopIntent)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    private fun updateNotification(detectionCount: Int) {
        val mgr = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        mgr.notify(NOTIFICATION_ID, buildNotification(detectionCount))
    }

    companion object {
        private const val TAG = "DetectionService"
        private const val CHANNEL_ID = "atomtag_detection"
        private const val NOTIFICATION_ID = 1001
        const val ACTION_STOP = "com.atomtag.action.STOP_DETECTION"
    }
}
