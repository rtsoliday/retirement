package com.retirementreadinesslab.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FeatureAccessTest {
    @Test
    fun freeAccessCapsSimulationsAndDisablesProOnlyAssumptions() {
        val scenario = sampleBaseScenario().copy(
            household = sampleBaseScenario().household.copy(retirementAge = 58),
            spending = sampleBaseScenario().spending.copy(lowPortfolioSpendingReduction = 0.25),
            healthcare = sampleBaseScenario().healthcare.copy(
                healthcareInflationMean = 0.09,
                healthcareInflationStdDev = 0.05
            ),
            rothConversion = RothConversionStrategy(enabled = true, marginalRateCap = 0.24),
            withdrawalStrategy = WithdrawalStrategy(
                useCashReserveDuringDrawdowns = true,
                drawdownTrigger = -0.08,
                applyEarlyWithdrawalPenalty = false,
                ruleOf55Eligible = true,
                seppEligible = true
            ),
            postRetirementAllocation = PostRetirementAllocationStrategy(
                stockUnder30x = 0.20,
                stock30xTo35x = 0.25,
                stock35xTo40x = 0.30,
                stock40xTo45x = 0.35,
                stock45xTo50x = 0.40,
                stock50xOrMore = 0.45
            ),
            longTermCare = LongTermCareAssumption(enabled = true),
            numberOfSimulations = 1_500
        )

        val runnable = scenario.forFeatureAccess(FeatureAccess(isProUnlocked = false))

        assertEquals(FREE_SIMULATION_LIMIT, runnable.numberOfSimulations)
        assertEquals(0.0, runnable.spending.lowPortfolioSpendingReduction, 0.001)
        assertEquals(DEFAULT_HEALTHCARE_INFLATION_MEAN, runnable.healthcare.healthcareInflationMean, 0.001)
        assertEquals(DEFAULT_HEALTHCARE_INFLATION_STD_DEV, runnable.healthcare.healthcareInflationStdDev, 0.001)
        assertFalse(runnable.rothConversion.enabled)
        assertFalse(runnable.withdrawalStrategy.useCashReserveDuringDrawdowns)
        assertEquals(PostRetirementAllocationStrategy(), runnable.postRetirementAllocation)
        assertTrue(runnable.withdrawalStrategy.applyEarlyWithdrawalPenalty)
        assertFalse(runnable.withdrawalStrategy.ruleOf55Eligible)
        assertFalse(runnable.withdrawalStrategy.seppEligible)
        assertFalse(runnable.longTermCare.enabled)
    }

    @Test
    fun proAccessKeepsProAssumptionsAndCapsAtProLimit() {
        val scenario = sampleBaseScenario().copy(
            rothConversion = RothConversionStrategy(enabled = true, marginalRateCap = 0.24),
            withdrawalStrategy = WithdrawalStrategy(
                useCashReserveDuringDrawdowns = true,
                drawdownTrigger = -0.08,
                applyEarlyWithdrawalPenalty = false,
                ruleOf55Eligible = true,
                seppEligible = true
            ),
            postRetirementAllocation = PostRetirementAllocationStrategy(stockUnder30x = 0.20),
            longTermCare = LongTermCareAssumption(enabled = true),
            numberOfSimulations = PRO_SIMULATION_LIMIT + 250
        )

        val runnable = scenario.forFeatureAccess(FeatureAccess(isProUnlocked = true))

        assertEquals(PRO_SIMULATION_LIMIT, runnable.numberOfSimulations)
        assertTrue(runnable.rothConversion.enabled)
        assertEquals(0.20, runnable.postRetirementAllocation.stockUnder30x, 0.001)
        assertFalse(runnable.withdrawalStrategy.useCashReserveDuringDrawdowns)
        assertTrue(runnable.withdrawalStrategy.ruleOf55Eligible)
        assertTrue(runnable.withdrawalStrategy.seppEligible)
        assertTrue(runnable.longTermCare.enabled)
    }
}
