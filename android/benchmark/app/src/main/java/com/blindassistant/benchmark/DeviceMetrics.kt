package com.blindassistant.benchmark

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import org.json.JSONArray
import org.json.JSONObject

data class DeviceSnapshot(
    val model: String,
    val sdkInt: Int,
    val abis: List<String>,
    val availMemMb: Long,
    val nativeHeapMb: Double,
    val batteryPct: Int,
    val tempC: Double,
    val thermalStatus: Int
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("model", model)
        put("sdkInt", sdkInt)
        put("abis", JSONArray(abis))
        put("availMemMb", availMemMb)
        put("nativeHeapMb", nativeHeapMb)
        put("batteryPct", batteryPct)
        put("tempC", tempC)
        put("thermalStatus", thermalStatus)
    }
}

object DeviceMetrics {
    private fun batteryTempC(context: Context): Double {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        return tenths / 10.0
    }

    private fun currentThermal(context: Context): Int {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return -1
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        return pm.currentThermalStatus
    }

    fun capture(context: Context): DeviceSnapshot {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val pct = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val heapMb = Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)
        return DeviceSnapshot(
            model = "${Build.MANUFACTURER} ${Build.MODEL}",
            sdkInt = Build.VERSION.SDK_INT,
            abis = Build.SUPPORTED_ABIS.toList(),
            availMemMb = memInfo.availMem / (1024L * 1024L),
            nativeHeapMb = (heapMb * 100.0).toLong() / 100.0,
            batteryPct = pct,
            tempC = batteryTempC(context),
            thermalStatus = currentThermal(context)
        )
    }
}
