package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.LongTermCareAssumption
import com.retirementreadinesslab.model.MarketAssumptions
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPlan
import com.retirementreadinesslab.model.SpendingPathModel
import com.retirementreadinesslab.model.WithdrawalStrategy
import com.retirementreadinesslab.model.sampleBaseScenario
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RetirementOptimizerTest {
    @Test
    fun analysisRetirementAgeChangesRefreshWithdrawalDefaults() {
        val scenario = sampleBaseScenario().copy(
            household = HouseholdProfile(currentAge = 54, retirementAge = 67),
            withdrawalStrategy = WithdrawalStrategy(
                useCashReserveDuringDrawdowns = true,
                drawdownTrigger = -0.08,
                applyEarlyWithdrawalPenalty = false,
                ruleOf55Eligible = false,
                seppEligible = true
            )
        )

        val age54 = scenario.withRetirementAgeForAnalysis(54)
        val age58 = scenario.withRetirementAgeForAnalysis(58)
        val originalAge = scenario.withRetirementAgeForAnalysis(67)

        assertTrue(age54.withdrawalStrategy.applyEarlyWithdrawalPenalty)
        assertFalse(age54.withdrawalStrategy.ruleOf55Eligible)
        assertFalse(age54.withdrawalStrategy.seppEligible)
        assertTrue(age58.withdrawalStrategy.applyEarlyWithdrawalPenalty)
        assertTrue(age58.withdrawalStrategy.ruleOf55Eligible)
        assertFalse(age58.withdrawalStrategy.seppEligible)
        assertTrue(age58.withdrawalStrategy.useCashReserveDuringDrawdowns)
        assertEquals(-0.08, age58.withdrawalStrategy.drawdownTrigger, 0.001)
        assertEquals(scenario.withdrawalStrategy, originalAge.withdrawalStrategy)
    }

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
    fun safeSpendingRoundsDownAndTestsReturnedAmount() {
        val scenario = sampleBaseScenario().copy(
            household = HouseholdProfile(currentAge = 40, retirementAge = 40, targetEndAge = 41),
            accounts = AccountBalances(pretax = 0.0, roth = 12_300.0, taxable = 0.0, cash = 0.0),
            spending = SpendingPlan(
                annualBaseSpending = 12_300.0,
                generalInflationMean = 0.0,
                generalInflationStdDev = 0.0,
                spendingPathModel = SpendingPathModel.Flat,
                lowPortfolioSpendingReduction = 0.0
            ),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            longTermCare = LongTermCareAssumption(enabled = false),
            market = MarketAssumptions(
                preRetirementMeanReturn = 0.0,
                preRetirementStdDev = 0.0,
                stockMeanReturn = 0.0,
                stockStdDev = 0.0,
                bondMeanReturn = 0.0,
                bondStdDev = 0.0
            ),
            numberOfSimulations = 50,
            seed = 72L
        )

        val estimate = RetirementOptimizer.estimate(
            scenario = scenario,
            targetReadiness = 1.0,
            simulationCount = 50
        )
        val roundedUpResult = RetirementSimulator.run(
            scenario.copy(
                spending = scenario.spending.copy(annualBaseSpending = 12_500.0),
                numberOfSimulations = 50,
                seed = scenario.seed + 20_000L
            )
        )

        assertEquals(12_000.0, estimate.safeAnnualSpending ?: -1.0, 0.01)
        assertEquals(1.0, estimate.safeSpendingReadiness ?: 0.0, 0.0001)
        assertTrue(roundedUpResult.successProbability < 1.0)
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
