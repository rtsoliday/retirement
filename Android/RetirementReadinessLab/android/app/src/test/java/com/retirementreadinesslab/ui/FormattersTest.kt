package com.retirementreadinesslab.ui

import org.junit.Assert.assertEquals
import org.junit.Test

class FormattersTest {
    @Test
    fun editablePercentPreservesEnteredPrecisionAcrossRepeatedEdits() {
        for (rate in listOf(0.02345, 0.0001, -0.01234, 0.133, 1.0, 0.0)) {
            var savedRate = rate
            repeat(3) {
                savedRate = savedRate.asEditablePercent().toDouble() / 100.0
            }
            assertEquals(rate, savedRate, 1e-15)
        }
        assertEquals("2.345", 0.02345.asEditablePercent())
        assertEquals("13.3", 0.133.asEditablePercent())
        assertEquals("100", 1.0.asEditablePercent())
    }

    @Test
    fun editableMoneyPreservesCentsWithoutAddingNoiseToWholeDollars() {
        assertEquals("1250.75", 1250.75.asEditableMoney())
        assertEquals("1250", 1250.0.asEditableMoney())
    }

    @Test
    fun editableMoneyCanRenderZeroAsAnEmptyOptionalField() {
        assertEquals("", 0.0.asEditableMoney(blankWhenZero = true))
        assertEquals("0", 0.0.asEditableMoney())
    }
}
