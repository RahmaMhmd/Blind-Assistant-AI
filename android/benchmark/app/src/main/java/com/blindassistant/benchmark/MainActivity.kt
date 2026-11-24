package com.blindassistant.benchmark

import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private val scope = CoroutineScope(Dispatchers.Main + Job())
    private lateinit var outputView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val runBtn = Button(this).apply { text = "Run Phone Benchmark" }
        outputView = TextView(this).apply { text = "Tap Run Phone Benchmark to start" }
        val scroll = ScrollView(this).apply { addView(outputView) }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(36, 36, 36, 36)
            addView(runBtn)
            addView(scroll)
        }
        setContentView(root)
        runBtn.setOnClickListener { startBenchmark(runBtn) }
    }

    private fun startBenchmark(runBtn: Button) {
        runBtn.isEnabled = false
        outputView.text = "Running benchmarks"
        scope.launch {
            val summary = withContext(Dispatchers.Default) { executeSuite() }
            outputView.text = summary
            runBtn.isEnabled = true
        }
    }

    private fun executeSuite(): String {
        val cfg = ConfigLoader.load(this)
        val before = DeviceMetrics.capture(this)
        val onnx = cfg.onnxModels.map { OnnxRunner.run(this, it, cfg.warmupRuns, cfg.timedRuns) }
        val mlkit = MlKitBarcodeRunner.run(cfg.barcodeWidth, cfg.barcodeHeight, cfg.warmupRuns, cfg.timedRuns)
        val all = onnx + mlkit
        val after = DeviceMetrics.capture(this)
        val saved = ReportWriter.write(this, cfg.reportFile, before, after, all)
        val lines = all.map { r ->
            if (r.stats != null) {
                String.format(Locale.US, "%s median %.1f ms p95 %.1f ms", r.name, r.stats.medianMs, r.stats.p95Ms)
            } else {
                "${r.name} ${r.status} ${r.detail}"
            }
        }
        return (listOf("Device ${before.model} Battery ${after.batteryPct}") + lines + "Saved ${saved.name}").joinToString("\n")
    }
}
