package com.blindassistant.benchmark

import android.content.Context

object BenchmarkSuite {
    suspend fun execute(ctx: Context, onProgress: suspend (String) -> Unit): String {
        val cfg = ConfigLoader.load(ctx)
        val before = DeviceMetrics.capture(ctx)
        val results = mutableListOf<ModelBenchmarkResult>()
        for (spec in cfg.onnxModels) {
            onProgress("Running ${spec.name}")
            results.add(OnnxRunner.run(ctx, spec, cfg.warmupRuns, cfg.timedRuns))
        }
        for (spec in cfg.tfliteModels) {
            onProgress("Running ${spec.name}")
            results.add(LiteRtRunner.run(ctx, spec, cfg.warmupRuns, cfg.timedRuns))
        }
        onProgress("Running mlkit_barcode")
        results.add(MlKitBarcodeRunner.run(ctx, cfg.barcodeWidth, cfg.barcodeHeight, cfg.warmupRuns, cfg.timedRuns))
        onProgress("Running mlkit_text")
        results.add(MlKitTextRunner.run(ctx, "en", cfg.warmupRuns))
        results.add(MlKitTextRunner.run(ctx, "ar", cfg.warmupRuns))
        val after = DeviceMetrics.capture(ctx)
        val saved = ReportWriter.write(ctx, cfg.reportFile, before, after, results)
        val head = "Device ${before.model} Temp ${before.tempC} to ${after.tempC} C"
        val thermal = "Thermal ${after.thermalStatus} Battery ${after.batteryPct} pct"
        val lines = results.map { MainActivity.formatLine(it) }
        return (listOf(head, thermal) + lines + "Saved ${saved.name}").joinToString("\n\n")
    }
}
