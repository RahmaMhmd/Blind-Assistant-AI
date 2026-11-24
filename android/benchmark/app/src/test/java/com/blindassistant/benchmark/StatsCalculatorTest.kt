package com.blindassistant.benchmark

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class StatsCalculatorTest {
    @Test
    fun summarizeComputesOddAndEvenMediansAndP95() {
        val odd = StatsCalculator.summarize(listOf(10.0, 30.0, 20.0))
        assertEquals(20.0, odd.medianMs, 0.01)
        assertEquals(30.0, odd.p95Ms, 0.01)

        val twenty = (1..20).map { it.toDouble() }
        val even = StatsCalculator.summarize(twenty)
        assertEquals(10.5, even.meanMs, 0.01)
        assertEquals(10.5, even.medianMs, 0.01)
        assertEquals(19.0, even.p95Ms, 0.01)
        assertEquals(1.0, even.minMs, 0.01)
        assertEquals(20.0, even.maxMs, 0.01)
    }

    @Test
    fun summarizeRejectsEmptyInput() {
        assertThrows(IllegalArgumentException::class.java) {
            StatsCalculator.summarize(emptyList())
        }
    }
}
