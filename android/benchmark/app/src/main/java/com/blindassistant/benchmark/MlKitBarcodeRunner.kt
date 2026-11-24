package com.blindassistant.benchmark

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.common.InputImage
import java.util.concurrent.TimeUnit

object MlKitBarcodeRunner {
    fun run(width: Int, height: Int, warmup: Int, timed: Int): ModelBenchmarkResult {
        val scanner = BarcodeScanning.getClient()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        return try {
            val image = InputImage.fromBitmap(bitmap, 0)
            repeat(warmup) {
                Tasks.await(scanner.process(image), 10, TimeUnit.SECONDS)
            }
            val times = (1..timed).map {
                val t0 = System.nanoTime()
                Tasks.await(scanner.process(image), 10, TimeUnit.SECONDS)
                (System.nanoTime() - t0) / 1_000_000.0
            }
            ModelBenchmarkResult("mlkit_barcode", "ok", StatsCalculator.summarize(times))
        } catch (err: Exception) {
            ModelBenchmarkResult("mlkit_barcode", "error", detail = err.message ?: "failed")
        } finally {
            bitmap.recycle()
            scanner.close()
        }
    }
}
