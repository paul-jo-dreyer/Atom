package com.atomtag.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.atomtag.R
import com.atomtag.detection.AprilTagDetector
import com.atomtag.detection.PoseTransformer
import com.atomtag.model.PoseVector
import com.atomtag.model.TagConfig
import com.atomtag.network.UdpBroadcaster
import com.google.android.material.switchmaterial.SwitchMaterial
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment() {

    private lateinit var previewView: PreviewView
    private lateinit var fpsText: TextView
    private lateinit var detectionText: TextView
    private lateinit var broadcastText: TextView
    private lateinit var axisOverlay: AxisOverlayView
    private lateinit var axisToggle: SwitchMaterial

    private lateinit var analysisExecutor: ExecutorService
    private lateinit var detector: AprilTagDetector
    private val poseVector = PoseVector()
    private val broadcaster = UdpBroadcaster()

    @Volatile private var drawAxes = false
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var lastBroadcastTime = 0L

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else Toast.makeText(requireContext(), "Camera permission required", Toast.LENGTH_SHORT).show()
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        return inflater.inflate(R.layout.fragment_camera, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        previewView = view.findViewById(R.id.previewView)
        fpsText = view.findViewById(R.id.fpsText)
        detectionText = view.findViewById(R.id.detectionText)
        broadcastText = view.findViewById(R.id.broadcastText)
        axisOverlay = view.findViewById(R.id.axisOverlay)
        axisToggle = view.findViewById(R.id.axisToggle)

        axisToggle.setOnCheckedChangeListener { _, isChecked ->
            drawAxes = isChecked
            if (!isChecked) axisOverlay.clear()
        }

        analysisExecutor = Executors.newSingleThreadExecutor()

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV init failed")
            return
        }

        detector = AprilTagDetector()
        broadcaster.start()
        broadcastText.text = "Broadcast: ${TagConfig.MULTICAST_GROUP}:${TagConfig.MULTICAST_PORT}"

        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::analyzeFrame) }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                viewLifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val rotation = imageProxy.imageInfo.rotationDegrees
        val gray = imageProxyToGrayMat(imageProxy)
        if (gray != null) {
            if (gray.cols() > 0 && gray.rows() > 0) {
                val shouldDrawAxes = drawAxes
                detector.initIntrinsics(gray.cols(), gray.rows())
                val detections = detector.detect(gray, projectAxes = shouldDrawAxes)
                val rawPoses = detections.map { it.pose }
                val transformed = PoseTransformer.transformToOriginFrame(rawPoses)
                val originVisible = rawPoses.any { it.tagId == TagConfig.ORIGIN_TAG_ID }

                for (pose in transformed) {
                    poseVector.update(pose)
                }

                // Broadcast at configured rate
                val now = System.currentTimeMillis()
                if (now - lastBroadcastTime >= TagConfig.BROADCAST_INTERVAL_MS) {
                    lastBroadcastTime = now
                    broadcaster.broadcast(poseVector)
                }

                // Build overlay data — labels always shown, axes only when toggled
                val overlayData = detections.map { det ->
                    AxisOverlayView.TagOverlayData(
                        tagId = det.pose.tagId,
                        axisPoints = if (shouldDrawAxes) det.axisPoints else null,
                        bottomCenter = det.bottomCenter
                    )
                }

                val imgW = gray.cols()
                val imgH = gray.rows()

                // Show transformed poses (origin-frame when tag 0 visible, camera-frame otherwise)
                val frameLabel = if (originVisible) "ref: tag ${TagConfig.ORIGIN_TAG_ID}" else "ref: camera"
                val tagIds = transformed.map { it.tagId }
                val poseInfo = transformed.joinToString("\n") { p ->
                    "  Tag ${p.tagId}: (${String.format("%.3f", p.tx)}, ${String.format("%.3f", p.ty)}, ${String.format("%.3f", p.tz)})"
                }

                requireActivity().runOnUiThread {
                    detectionText.text = if (tagIds.isEmpty()) "Tags: none"
                    else "[$frameLabel]\nTags: ${tagIds.joinToString(", ")}\n$poseInfo"
                    updateFps()

                    if (overlayData.isNotEmpty()) {
                        axisOverlay.update(overlayData, imgW, imgH, rotation)
                    } else {
                        axisOverlay.clear()
                    }
                }
            }
            gray.release()
        }
        imageProxy.close()
    }

    private fun imageProxyToGrayMat(imageProxy: ImageProxy): Mat? {
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

    private fun updateFps() {
        frameCount++
        val now = System.currentTimeMillis()
        val elapsed = now - lastFpsTime
        if (elapsed >= 1000) {
            val fps = frameCount * 1000.0 / elapsed
            fpsText.text = "FPS: ${String.format("%.1f", fps)}"
            frameCount = 0
            lastFpsTime = now
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        analysisExecutor.shutdown()
        broadcaster.stop()
        detector.release()
    }

    companion object {
        private const val TAG = "CameraFragment"
    }
}
