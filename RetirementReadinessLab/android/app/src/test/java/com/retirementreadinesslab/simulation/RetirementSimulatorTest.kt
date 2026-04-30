package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPlan
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RetirementSimulatorTest {
    @Test
    fun deterministicSeedProducesStableResult() {
        val scenario = sampleScenario()

        val first = RetirementSimulator.run(scenario)
        val second = RetirementSimulator.run(scenario)

        assertEquals(first.successProbability, second.successProbability, 0.0001)
        assertEquals(first.medianEndingBalance, second.medianEndingBalance, 0.01)
        assertEquals("2026.04-monthly-mvp", first.provenance.engineVersion)
        assertEquals("Monthly cashflow model with annual result bands", first.provenance.engineCadence)
        assertEquals("2024 federal brackets", first.provenance.taxTableVersion)
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

    private fun sampleScenario(): RetirementScenario {
        return RetirementScenario(
            id = "test",
            name = "Test",
            household = HouseholdProfile(currentAge = 50, retirementAge = 60, targetEndAge = 90),
            accounts = AccountBalances(pretax = 700_000.0, roth = 100_000.0, taxable = 25_000.0, cash = 50_000.0),
            spending = SpendingPlan(annualBaseSpending = 60_000.0),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 500.0),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 28_000.0, claimAge = 67),
            numberOfSimulations = 200,
            seed = 1234L
        )
    }
}
