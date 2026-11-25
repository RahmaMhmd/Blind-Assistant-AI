package com.blindassistant.benchmark

import kotlin.math.ceil

data class LatencyStats(
    val meanMs: Double,
    val medianMs: Double,
    val p95Ms: Double,
    val minMs: Double,
    val maxMs: Double
)

object StatsCalculator {
    fun summarize(samplesMs: List<Double>): LatencyStats {
        require(samplesMs.isNotEmpty()) { "samples must not be empty" }
        val sorted = samplesMs.sorted()
        val n = sorted.size
        val median = if (n % 2 == 1) {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
        val p95Index = (ceil(0.95 * n).toInt() - 1).coerceIn(0, n - 1)
        return LatencyStats(
            meanMs = samplesMs.average(),
            medianMs = median,
            p95Ms = sorted[p95Index],
            minMs = sorted.first(),
            maxMs = sorted.last()
        )
    }
}
