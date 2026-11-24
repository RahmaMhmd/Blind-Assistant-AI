package com.blindassistant.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer

object OnnxRunner {
    private fun findModel(context: Context, fileName: String): File? {
        val extFile = File(context.getExternalFilesDir(null), fileName)
        if (extFile.isFile) return extFile
        val cacheFile = File(context.cacheDir, fileName)
        val assetList = context.assets.list("")?.toSet().orEmpty()
        if (fileName !in assetList) return null
        context.assets.open(fileName).use { input ->
            cacheFile.outputStream().use { output -> input.copyTo(output) }
        }
        return cacheFile
    }

    fun availableCount(context: Context, specs: List<OnnxModelSpec>): Int {
        return specs.count { findModel(context, it.file) != null }
    }

    fun run(context: Context, spec: OnnxModelSpec, warmup: Int, timed: Int): ModelBenchmarkResult {
        val modelFile = findModel(context, spec.file)
            ?: return ModelBenchmarkResult(spec.name, "missing_model", detail = "copy ${spec.file} to app files dir")
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
        val shape = longArrayOf(1, 3, spec.inputSize.toLong(), spec.inputSize.toLong())
        val floats = FloatArray(3 * spec.inputSize * spec.inputSize) { 0.5f }
        env.createSession(modelFile.absolutePath, opts).use { session ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), shape).use { tensor ->
                val feed = mapOf(spec.inputName to tensor)
                return BenchmarkHarness.measure(context, spec.name, warmup, timed) {
                    session.run(feed).close()
                }
            }
        }
    }
}
