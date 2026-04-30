package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.DEFAULT_PROJECTION_END_AGE
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPlan
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class RetirementSimulatorTest {
    @Test
    fun deterministicSeedProducesStableResult() {
        val scenario = sampleScenario()

        val first = RetirementSimulator.run(scenario)
        val second = RetirementSimulator.run(scenario)

        assertEquals(first.successProbability, second.successProbability, 0.0001)
        assertEquals(first.medianEndingBalance, second.medianEndingBalance, 0.01)
        assertEquals("2026.04-medicare-parity", first.provenance.engineVersion)
        assertEquals("Monthly cashflow model with annual result bands", first.provenance.engineCadence)
        assertEquals("2024 federal brackets", first.provenance.taxTableVersion)
        assertEquals(
            "SSA Trustees Alt2 2025 annual death probabilities",
            first.provenance.mortalityModelVersion
        )
        assertEquals(scenario.seed, first.provenance.randomSeed)
        assertEquals(scenario.numberOfSimulations, first.provenance.simulationCount)
        assertEquals(first.provenance.assumptionFingerprint, second.provenance.assumptionFingerprint)
        assertTrue(first.provenance.assumptionFingerprint.length >= 8)
    }

    @Test
    fun noSpendingScenarioAlwaysSucceeds() {
        val result = RetirementSimulator.run(
            sampleScenario().copy(
                spending = SpendingPlan(annualBaseSpending = 0.0),
                healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false)
            )
        )

        assertEquals(1.0, result.successProbability, 0.0001)
    }

    @Test
    fun noAssetsWithSpendingUsuallyFails() {
        val result = RetirementSimulator.run(
            sampleScenario().copy(
                accounts = AccountBalances(pretax = 0.0, roth = 0.0, taxable = 0.0, cash = 0.0),
                socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67)
            )
        )

        assertTrue(result.successProbability < 0.10)
        assertTrue(result.failureAgeBuckets.isNotEmpty())
        assertTrue(result.failureAgeBuckets.sumOf { it.count } > 0)
    }

    @Test
    fun preRetirementGrowthDoesNotFloorMonthlyLosses() {
        val random = object : Random(0L) {
            override fun nextGaussian(): Double = -3.0
        }

        val growth = RetirementSimulator.samplePreRetirementMonthlyGrowth(
            random = random,
            mean = 0.0,
            stdDev = 1.0
        )

        assertEquals(-3.0, growth, 0.0001)
    }

    @Test
    fun longTermCareReplacesNormalSpendingWhileActive() {
        val normalNeed = RetirementSimulator.monthlyRetirementNeed(
            monthlySpending = 5_000.0,
            mortgageCost = 1_500.0,
            healthcareCost = 300.0,
            longTermCareCost = 8_000.0,
            inLongTermCare = false
        )
        val careNeed = RetirementSimulator.monthlyRetirementNeed(
            monthlySpending = 5_000.0,
            mortgageCost = 1_500.0,
            healthcareCost = 300.0,
            longTermCareCost = 8_000.0,
            inLongTermCare = true
        )

        assertEquals(6_800.0, normalNeed, 0.01)
        assertEquals(8_300.0, careNeed, 0.01)
    }

    private fun sampleScenario(): RetirementScenario {
        return RetirementScenario(
            id = "test",
            name = "Test",
            household = HouseholdProfile(currentAge = 50, retirementAge = 60, targetEndAge = DEFAULT_PROJECTION_END_AGE),
            accounts = AccountBalances(pretax = 700_000.0, roth = 100_000.0, taxable = 25_000.0, cash = 50_000.0),
            spending = SpendingPlan(annualBaseSpending = 60_000.0),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 500.0),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 28_000.0, claimAge = 67),
            numberOfSimulations = 200,
            seed = 1234L
        )
    }
}
