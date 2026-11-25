package com.blindassistant.benchmark

import android.content.Context
import android.graphics.BitmapFactory
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.json.JSONArray
import java.io.File
import java.util.concurrent.TimeUnit

object MlKitTextRunner {
    private fun norm(text: String): String {
        return text.trim().lowercase().replace(Regex("\\s+"), " ")
    }

    fun run(context: Context, lang: String, warmup: Int): ModelBenchmarkResult {
        val intCases = File(context.filesDir, "ocr_eval/cases.json")
        val extCases = File(context.getExternalFilesDir(null), "ocr_eval/cases.json")
        val casesFile = if (intCases.isFile) intCases else extCases
        val name = "mlkit_text_$lang"
        if (!casesFile.isFile) return ModelBenchmarkResult(name, "missing_eval", detail = "no ocr eval")
        val baseDir = casesFile.parentFile ?: return ModelBenchmarkResult(name, "error")
        val arr = JSONArray(casesFile.readText())
        val items = (0 until arr.length()).map { arr.getJSONObject(it) }.filter { it.getString("lang") == lang }
        if (items.isEmpty()) return ModelBenchmarkResult(name, "empty", detail = "no $lang samples")

        val client = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        val before = DeviceMetrics.capture(context)
        var correct = 0
        var wrong = 0
        val times = mutableListOf<Double>()
        return try {
            for ((idx, item) in items.withIndex()) {
                val bmp = BitmapFactory.decodeFile(File(baseDir, item.getString("file")).absolutePath) ?: continue
                val got = try {
                    val input = InputImage.fromBitmap(bmp, 0)
                    if (idx < warmup) Tasks.await(client.process(input), 10, TimeUnit.SECONDS)
                    val t0 = System.nanoTime()
                    val res = Tasks.await(client.process(input), 10, TimeUnit.SECONDS)
                    times.add((System.nanoTime() - t0) / 1_000_000.0)
                    norm(res.text)
                } catch (_: Exception) {
                    ""
                } finally {
                    bmp.recycle()
                }
                val expected = norm(item.getString("text"))
                if (got == expected) correct++ else if (got.isNotEmpty()) wrong++
            }
            val after = DeviceMetrics.capture(context)
            val delta = ((after.nativeHeapMb - before.nativeHeapMb) * 100.0).toLong() / 100.0
            val n = times.size
            ModelBenchmarkResult(
                name = name,
                status = "ok",
                stats = StatsCalculator.summarize(times),
                memDeltaMb = delta,
                tempBeforeC = before.tempC,
                tempAfterC = after.tempC,
                correctRate = if (n > 0) correct.toDouble() / n else 0.0,
                wrongRate = if (n > 0) wrong.toDouble() / n else 0.0,
                total = n,
                correct = correct,
                detail = "real_ocr_$lang"
            )
        } finally {
            client.close()
        }
    }
}
