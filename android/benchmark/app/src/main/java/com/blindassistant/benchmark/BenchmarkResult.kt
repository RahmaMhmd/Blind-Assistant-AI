package com.blindassistant.benchmark

import android.content.Context
import org.json.JSONObject

data class ModelBenchmarkResult(
    val name: String,
    val status: String,
    val stats: LatencyStats? = null,
    val memDeltaMb: Double = 0.0,
    val tempBeforeC: Double = 0.0,
    val tempAfterC: Double = 0.0,
    val correctRate: Double? = null,
    val wrongRate: Double? = null,
    val total: Int = 0,
    val correct: Int = 0,
    val detail: String = ""
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("name", name)
        put("status", status)
        put("memDeltaMb", memDeltaMb)
        put("tempBeforeC", tempBeforeC)
        put("tempAfterC", tempAfterC)
        put("detail", detail)
        if (correctRate != null) {
            put("correctRate", correctRate)
            put("wrongRate", wrongRate ?: 0.0)
            put("total", total)
            put("correct", correct)
        }
        if (stats != null) {
            put("meanMs", stats.meanMs)
            put("medianMs", stats.medianMs)
            put("p95Ms", stats.p95Ms)
            put("minMs", stats.minMs)
            put("maxMs", stats.maxMs)
        }
    }
}

object BenchmarkHarness {
    fun measure(
        context: Context,
        name: String,
        warmup: Int,
        timed: Int,
        step: () -> Unit
    ): ModelBenchmarkResult {
        val before = DeviceMetrics.capture(context)
        repeat(warmup) { step() }
        val times = (1..timed).map {
            val t0 = System.nanoTime()
            step()
            (System.nanoTime() - t0) / 1_000_000.0
        }
        val after = DeviceMetrics.capture(context)
        val delta = ((after.nativeHeapMb - before.nativeHeapMb) * 100.0).toLong() / 100.0
        return ModelBenchmarkResult(name, "ok", StatsCalculator.summarize(times), delta, before.tempC, after.tempC)
    }
}
