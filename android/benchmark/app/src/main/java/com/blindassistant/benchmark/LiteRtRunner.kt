package com.blindassistant.benchmark

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

object LiteRtRunner {
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

    fun availableCount(context: Context, specs: List<TfLiteModelSpec>): Int {
        return specs.count { findModel(context, it.file) != null }
    }

    fun run(context: Context, spec: TfLiteModelSpec, warmup: Int, timed: Int): ModelBenchmarkResult {
        val modelFile = findModel(context, spec.file)
            ?: return ModelBenchmarkResult(spec.name, "missing_model", detail = "copy ${spec.file} to app files dir")
        val opts = Interpreter.Options().apply { setNumThreads(4) }
        return try {
            Interpreter(modelFile, opts).use { interp ->
                val inBytes = interp.getInputTensor(0).numBytes()
                val inputBuf = ByteBuffer.allocateDirect(inBytes).order(ByteOrder.nativeOrder())
                val outputs = HashMap<Int, Any>()
                for (i in 0 until interp.outputTensorCount) {
                    val outBytes = interp.getOutputTensor(i).numBytes()
                    outputs[i] = ByteBuffer.allocateDirect(outBytes).order(ByteOrder.nativeOrder())
                }
                val inputs = arrayOf<Any>(inputBuf)
                BenchmarkHarness.measure(context, spec.name, warmup, timed) {
                    inputBuf.rewind()
                    outputs.values.forEach { (it as ByteBuffer).rewind() }
                    interp.runForMultipleInputsOutputs(inputs, outputs)
                }
            }
        } catch (err: Exception) {
            ModelBenchmarkResult(spec.name, "error", detail = err.message ?: "failed")
        }
    }
}
