package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.FilingStatus
import org.junit.Assert.assertEquals
import org.junit.Test

class MedicarePremiumsTest {
    @Test
    fun lowIncomeSingleFilerUsesBasePartBAndPartDEstimate() {
        val estimate = MedicarePremiums.estimateAnnualPremium(
            modifiedAdjustedGrossIncome = 75_000.0,
            filingStatus = FilingStatus.Single
        )

        assertEquals("Base premium", estimate.tierLabel)
        assertEquals(230.20, estimate.monthlyPremium, 0.01)
        assertEquals(2_762.40, estimate.annualPremium, 0.01)
        assertEquals(0.0, estimate.partBIrmaaMonthly, 0.01)
        assertEquals(0.0, estimate.partDIrmaaMonthly, 0.01)
    }

    @Test
    fun highIncomeSingleFilerUsesTopIrmaaTier() {
        val estimate = MedicarePremiums.estimateAnnualPremium(
            modifiedAdjustedGrossIncome = 650_000.0,
            filingStatus = FilingStatus.Single
        )

        assertEquals("IRMAA tier 5", estimate.tierLabel)
        assertEquals(730.50, estimate.monthlyPremium, 0.01)
        assertEquals(8_766.00, estimate.annualPremium, 0.01)
        assertEquals(419.30, estimate.partBIrmaaMonthly, 0.01)
        assertEquals(81.00, estimate.partDIrmaaMonthly, 0.01)
    }

    @Test
    fun marriedFilerUsesMarriedIncomeThresholds() {
        val singleEstimate = MedicarePremiums.estimateAnnualPremium(
            modifiedAdjustedGrossIncome = 150_000.0,
            filingStatus = FilingStatus.Single
        )
        val marriedEstimate = MedicarePremiums.estimateAnnualPremium(
            modifiedAdjustedGrossIncome = 150_000.0,
            filingStatus = FilingStatus.Married
        )

        assertEquals("IRMAA tier 2", singleEstimate.tierLabel)
        assertEquals("Base premium", marriedEstimate.tierLabel)
    }

    @Test
    fun inflationMultiplierIndexesThresholdsAndPremiums() {
        val estimate = MedicarePremiums.estimateAnnualPremium(
            modifiedAdjustedGrossIncome = 110_000.0,
            filingStatus = FilingStatus.Single,
            inflationMultiplier = 1.20
        )

        assertEquals("Base premium", estimate.tierLabel)
        assertEquals(276.24, estimate.monthlyPremium, 0.01)
        assertEquals(3_314.88, estimate.annualPremium, 0.01)
    }

    @Test
    fun coveredPeopleScalesPremiumEstimate() {
        val estimate = MedicarePremiums.estimateAnnualPremium(
            modifiedAdjustedGrossIncome = 75_000.0,
            filingStatus = FilingStatus.Married,
            coveredPeople = 2
        )

        assertEquals(460.40, estimate.monthlyPremium, 0.01)
        assertEquals(5_524.80, estimate.annualPremium, 0.01)
    }
}
