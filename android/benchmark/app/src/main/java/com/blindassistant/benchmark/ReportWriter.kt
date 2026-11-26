package com.blindassistant.benchmark

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

object ReportWriter {
    fun write(
        context: Context,
        fileName: String,
        before: DeviceSnapshot,
        after: DeviceSnapshot,
        results: List<ModelBenchmarkResult>
    ): File {
        val payload = JSONObject().apply {
            put("deviceBefore", before.toJson())
            put("deviceAfter", after.toJson())
            put("results", JSONArray(results.map { it.toJson() }))
        }
        val outFile = File(context.getExternalFilesDir(null), fileName)
        outFile.writeText(payload.toString(2))
        return outFile
    }
}
