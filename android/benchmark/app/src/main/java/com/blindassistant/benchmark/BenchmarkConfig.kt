package com.blindassistant.benchmark

import android.content.Context
import org.json.JSONObject

data class OnnxModelSpec(
    val name: String,
    val file: String,
    val inputName: String,
    val inputSize: Int
)

data class BenchmarkConfig(
    val warmupRuns: Int,
    val timedRuns: Int,
    val reportFile: String,
    val onnxModels: List<OnnxModelSpec>,
    val barcodeWidth: Int,
    val barcodeHeight: Int
)

object ConfigLoader {
    fun load(context: Context): BenchmarkConfig {
        val raw = context.assets.open("benchmark_config.json").bufferedReader().use { it.readText() }
        val root = JSONObject(raw)
        val arr = root.getJSONArray("onnxModels")
        val models = (0 until arr.length()).map { idx ->
            val obj = arr.getJSONObject(idx)
            OnnxModelSpec(
                name = obj.getString("name"),
                file = obj.getString("file"),
                inputName = obj.getString("inputName"),
                inputSize = obj.getInt("inputSize")
            )
        }
        return BenchmarkConfig(
            warmupRuns = root.getInt("warmupRuns"),
            timedRuns = root.getInt("timedRuns"),
            reportFile = root.getString("reportFile"),
            onnxModels = models,
            barcodeWidth = root.getInt("barcodeWidth"),
            barcodeHeight = root.getInt("barcodeHeight")
        )
    }
}
