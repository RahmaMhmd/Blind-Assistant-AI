package com.blindassistant.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.common.InputImage
import org.json.JSONArray
import java.io.File
import java.util.concurrent.TimeUnit

object MlKitBarcodeRunner {
    private fun ensureMinSize(src: Bitmap, minPx: Int = 32): Bitmap {
        if (src.width >= minPx && src.height >= minPx) return src
        val w = maxOf(src.width, minPx)
        val h = maxOf(src.height, minPx)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(Color.WHITE)
            drawBitmap(src, (w - src.width) / 2f, (h - src.height) / 2f, null)
        }
        src.recycle()
        return out
    }

    private fun runRealCases(context: Context, casesFile: File, warmup: Int): ModelBenchmarkResult {
        val scanner = BarcodeScanning.getClient()
        val arr = JSONArray(casesFile.readText())
        val baseDir = casesFile.parentFile ?: return ModelBenchmarkResult("mlkit_barcode", "error")
        val before = DeviceMetrics.capture(context)
        var correct = 0
        var wrong = 0
        val times = mutableListOf<Double>()
        return try {
            for (idx in 0 until arr.length()) {
                val item = arr.getJSONObject(idx)
                val imgFile = File(baseDir, item.getString("file"))
                val raw = BitmapFactory.decodeFile(imgFile.absolutePath) ?: continue
                val bmp = ensureMinSize(raw)
                val decoded = try {
                    val input = InputImage.fromBitmap(bmp, 0)
                    if (idx < warmup) Tasks.await(scanner.process(input), 10, TimeUnit.SECONDS)
                    val t0 = System.nanoTime()
                    val found = Tasks.await(scanner.process(input), 10, TimeUnit.SECONDS)
                    times.add((System.nanoTime() - t0) / 1_000_000.0)
                    found.mapNotNull { it.rawValue }
                } catch (_: Exception) {
                    emptyList()
                } finally {
                    bmp.recycle()
                }
                val expected = item.getString("value")
                if (expected in decoded) correct++ else if (decoded.isNotEmpty()) wrong++
            }
            val after = DeviceMetrics.capture(context)
            val delta = ((after.nativeHeapMb - before.nativeHeapMb) * 100.0).toLong() / 100.0
            val n = times.size
            ModelBenchmarkResult(
                name = "mlkit_barcode",
                status = "ok",
                stats = StatsCalculator.summarize(times),
                memDeltaMb = delta,
                tempBeforeC = before.tempC,
                tempAfterC = after.tempC,
                correctRate = if (n > 0) correct.toDouble() / n else 0.0,
                wrongRate = if (n > 0) wrong.toDouble() / n else 0.0,
                total = n,
                correct = correct,
                detail = "real_qr_eval"
            )
        } finally {
            scanner.close()
        }
    }

    fun run(context: Context, width: Int, height: Int, warmup: Int, timed: Int): ModelBenchmarkResult {
        val intCases = File(context.filesDir, "qr_eval/cases.json")
        val extCases = File(context.getExternalFilesDir(null), "qr_eval/cases.json")
        val casesFile = if (intCases.isFile) intCases else extCases
        if (casesFile.isFile) return runRealCases(context, casesFile, warmup)
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
