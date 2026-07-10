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
}
