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
        assertEquals(241.89, estimate.monthlyPremium, 0.01)
        assertEquals(2_902.68, estimate.annualPremium, 0.01)
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
        assertEquals(819.89, estimate.monthlyPremium, 0.01)
        assertEquals(9_838.68, estimate.annualPremium, 0.01)
        assertEquals(487.00, estimate.partBIrmaaMonthly, 0.01)
        assertEquals(91.00, estimate.partDIrmaaMonthly, 0.01)
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
        assertEquals(290.27, estimate.monthlyPremium, 0.01)
        assertEquals(3_483.22, estimate.annualPremium, 0.01)
    }

    @Test
    fun premiumAndIncomeThresholdInflationCanUseDifferentAssumptions() {
        val estimate = MedicarePremiums.estimateAnnualPremium(
            modifiedAdjustedGrossIncome = 110_000.0,
            filingStatus = FilingStatus.Single,
            inflationMultiplier = 1.20,
            incomeThresholdInflationMultiplier = 1.0
        )

        assertEquals("IRMAA tier 1", estimate.tierLabel)
        assertEquals(405.11, estimate.monthlyPremium, 0.01)
    }

    @Test
    fun coveredPeopleScalesPremiumEstimate() {
        val estimate = MedicarePremiums.estimateAnnualPremium(
            modifiedAdjustedGrossIncome = 75_000.0,
            filingStatus = FilingStatus.Married,
            coveredPeople = 2
        )

        assertEquals(483.78, estimate.monthlyPremium, 0.01)
        assertEquals(5_805.36, estimate.annualPremium, 0.01)
    }

    @Test
    fun zeroCoveredPeopleProducesNoPremium() {
        val estimate = MedicarePremiums.estimateAnnualPremium(
            modifiedAdjustedGrossIncome = 75_000.0,
            filingStatus = FilingStatus.Single,
            coveredPeople = 0
        )

        assertEquals(0.0, estimate.monthlyPremium, 0.01)
        assertEquals(0.0, estimate.annualPremium, 0.01)
    }
}
