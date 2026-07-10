package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.LongTermCareAssumption
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPlan
import com.retirementreadinesslab.model.sampleBaseScenario
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RetirementOptimizerTest {
    @Test
    fun estimateReturnsDecisionTargetsForSamplePlan() {
        val scenario = sampleBaseScenario().copy(
            longTermCare = LongTermCareAssumption(enabled = false),
            numberOfSimulations = 100
        )

        val estimate = RetirementOptimizer.estimate(
            scenario = scenario,
            targetReadiness = 0.45,
            simulationCount = 80
        )

        assertEquals(80, estimate.simulationCount)
        assertNotNull(estimate.earliestRetirementAge)
        assertTrue(estimate.earliestRetirementAge!! in scenario.household.currentAge..70)
        assertNotNull(estimate.safeAnnualSpending)
        assertTrue(estimate.safeAnnualSpending!! >= 0.0)
    }

    @Test
    fun noAssetsAndNoIncomeCanFailTargetFinders() {
        val scenario = sampleBaseScenario().copy(
            accounts = AccountBalances(pretax = 0.0, roth = 0.0, taxable = 0.0, cash = 0.0),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 500.0),
            spending = SpendingPlan(annualBaseSpending = 60_000.0)
        )

        val estimate = RetirementOptimizer.estimate(
            scenario = scenario,
            targetReadiness = 0.80,
            simulationCount = 80
        )

        assertNull(estimate.earliestRetirementAge)
        assertNull(estimate.safeAnnualSpending)
    }

    @Test
    fun roundedSafeSpendingReportsReadinessForTheReturnedAmount() {
        val scenario = sampleBaseScenario().copy(
            longTermCare = LongTermCareAssumption(enabled = false)
        )
        val simulationCount = 80
        val target = 0.45

        val estimate = RetirementOptimizer.estimate(
            scenario = scenario,
            targetReadiness = target,
            simulationCount = simulationCount
        )
        val returnedSpending = estimate.safeAnnualSpending!!
        val verifiedReadiness = RetirementSimulator.run(
            scenario.copy(
                spending = scenario.spending.copy(annualBaseSpending = returnedSpending),
                numberOfSimulations = simulationCount,
                seed = scenario.seed + 20_000L
            )
        ).successProbability

        assertEquals(0.0, returnedSpending % 500.0, 0.01)
        assertEquals(verifiedReadiness, estimate.safeSpendingReadiness!!, 0.0001)
        assertTrue(estimate.safeSpendingReadiness >= target)
    }

    @Test(expected = IllegalArgumentException::class)
    fun targetReadinessMustBeAProbability() {
        RetirementOptimizer.estimate(
            scenario = sampleBaseScenario(),
            targetReadiness = 1.01,
            simulationCount = 50
        )
    }
}
