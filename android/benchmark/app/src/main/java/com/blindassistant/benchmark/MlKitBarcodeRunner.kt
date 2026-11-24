package com.blindassistant.benchmark

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.common.InputImage
import java.util.concurrent.TimeUnit

object MlKitBarcodeRunner {
    fun run(context: Context, width: Int, height: Int, warmup: Int, timed: Int): ModelBenchmarkResult {
        val scanner = BarcodeScanning.getClient()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        return try {
            val image = InputImage.fromBitmap(bitmap, 0)
            BenchmarkHarness.measure(context, "mlkit_barcode", warmup, timed) {
                Tasks.await(scanner.process(image), 10, TimeUnit.SECONDS)
            }
        } catch (err: Exception) {
            ModelBenchmarkResult("mlkit_barcode", "error", detail = err.message ?: "failed")
        } finally {
            bitmap.recycle()
            scanner.close()
        }
    }
}
