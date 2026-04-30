package com.retirementreadinesslab.ui

import java.text.NumberFormat
import java.util.Locale
import kotlin.math.abs

fun Double.asCurrency(): String {
    val format = NumberFormat.getCurrencyInstance(Locale.US).apply {
        maximumFractionDigits = 0
    }
    return format.format(this)
}

fun Double.asCompactCurrency(): String {
    val value = abs(this)
    val sign = if (this < 0) "-" else ""
    return when {
        value >= 1_000_000 -> "$sign\$${"%.1f".format(Locale.US, value / 1_000_000.0)}M"
        value >= 1_000 -> "$sign\$${"%.0f".format(Locale.US, value / 1_000.0)}k"
        else -> "$sign\$${"%.0f".format(Locale.US, value)}"
    }
}

fun Double.asPercent(): String {
    return "${"%.0f".format(Locale.US, this * 100.0)}%"
}
