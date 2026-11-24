package com.blindassistant.benchmark

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private val scope = CoroutineScope(Dispatchers.Main + Job())
    private lateinit var outputView: TextView
    private lateinit var runBtn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val cfg = ConfigLoader.load(this)
        val found = OnnxRunner.availableCount(this, cfg.onnxModels)
        val header = TextView(this).apply {
            text = "Blind Assistant AI Benchmark"
            textSize = 20f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.BLACK)
        }
        runBtn = Button(this).apply { text = "Run Phone Benchmark" }
        outputView = TextView(this).apply {
            text = "Ready with $found of ${cfg.onnxModels.size} ONNX models and ML Kit Barcode"
            textSize = 14f
            setTextColor(Color.DKGRAY)
            setPadding(0, 24, 0, 24)
        }
        val scroll = ScrollView(this).apply { addView(outputView) }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            addView(header)
            addView(runBtn)
            addView(scroll)
        }
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.setPadding(bars.left + 40, bars.top + 32, bars.right + 40, bars.bottom + 32)
            insets
        }
        setContentView(root)
        runBtn.setOnClickListener { startBenchmark() }
    }

    private fun formatLine(r: ModelBenchmarkResult): String {
        val s = r.stats ?: return "${r.name} ${r.status} ${r.detail}"
        val acc = r.correctRate?.let { String.format(Locale.US, " acc %.1f%%", it * 100.0) } ?: ""
        return String.format(
            Locale.US,
            "%s median %.1f ms p95 %.1f ms%s ram %.2f MB temp %.1f C",
            r.name, s.medianMs, s.p95Ms, acc, r.memDeltaMb, r.tempAfterC
        )
    }

    private fun startBenchmark() {
        runBtn.isEnabled = false
        scope.launch {
            val summary = withContext(Dispatchers.Default) { executeSuite(this@MainActivity) }
            outputView.text = summary
            runBtn.isEnabled = true
        }
    }

    private suspend fun executeSuite(ctx: Context): String {
        val cfg = ConfigLoader.load(ctx)
        val before = DeviceMetrics.capture(ctx)
        val results = mutableListOf<ModelBenchmarkResult>()
        for (spec in cfg.onnxModels) {
            withContext(Dispatchers.Main) { outputView.text = "Running ${spec.name}" }
            results.add(OnnxRunner.run(ctx, spec, cfg.warmupRuns, cfg.timedRuns))
        }
        withContext(Dispatchers.Main) { outputView.text = "Running mlkit_barcode" }
        results.add(MlKitBarcodeRunner.run(ctx, cfg.barcodeWidth, cfg.barcodeHeight, cfg.warmupRuns, cfg.timedRuns))
        val after = DeviceMetrics.capture(ctx)
        val saved = ReportWriter.write(ctx, cfg.reportFile, before, after, results)
        val head = "Device ${before.model} Temp ${before.tempC} to ${after.tempC} C"
        val thermal = "Thermal ${after.thermalStatus} Battery ${after.batteryPct} pct"
        return (listOf(head, thermal) + results.map(::formatLine) + "Saved ${saved.name}").joinToString("\n\n")
    }
}
