package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.LongTermCareAssumption
import com.retirementreadinesslab.model.MarketAssumptions
import com.retirementreadinesslab.model.MortgagePlan
import com.retirementreadinesslab.model.PostRetirementAllocationStrategy
import com.retirementreadinesslab.model.RothConversionStrategy
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPathModel
import com.retirementreadinesslab.model.SpendingPlan
import com.retirementreadinesslab.model.WithdrawalStrategy
import com.retirementreadinesslab.model.sampleBaseScenario
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScenarioLabAnalyzerTest {
    @Test
    fun analyzerBuildsTargetFindersSweepsAndComparisons() {
        val scenario = sampleBaseScenario()

        val analysis = ScenarioLabAnalyzer.analyze(
            scenario = scenario,
            quickSimulations = 60,
            targetEstimateSimulations = 60
        )

        assertEquals(scenario.id, analysis.scenarioId)
        assertEquals(3, analysis.sweeps.size)
        assertTrue(analysis.sweeps.all { it.rows.isNotEmpty() })
        assertEquals(5, analysis.comparisons.size)
        assertTrue(analysis.comparisons.any { it.type == LabComparisonType.RothConversion })
        assertTrue(analysis.comparisons.any { it.type == LabComparisonType.LongTermCare })
        assertTrue(analysis.comparisons.any { it.type == LabComparisonType.HealthcareInflation })
        assertTrue(analysis.comparisons.any { it.type == LabComparisonType.MarketDownturn })
        assertTrue(analysis.comparisons.any { it.type == LabComparisonType.MortgagePayoff })
        assertTrue(analysis.decisionEstimate.simulationCount >= 50)
    }

    @Test
    fun strategyComparisonsFlipAlreadySelectedOptions() {
        val scenario = sampleBaseScenario().copy(
            rothConversion = RothConversionStrategy(enabled = true, marginalRateCap = 0.22),
            longTermCare = LongTermCareAssumption(enabled = true),
            mortgage = MortgagePlan(monthlyPayment = 0.0, yearsLeft = 0, monthsLeft = 0)
        )

        val analysis = ScenarioLabAnalyzer.analyze(
            scenario = scenario,
            quickSimulations = 50,
            targetEstimateSimulations = 50
        )

        assertEquals(
            "Disable Roth conversions",
            analysis.comparisons.first { it.type == LabComparisonType.RothConversion }.title
        )
        assertEquals(
            "Remove long-term care risk",
            analysis.comparisons.first { it.type == LabComparisonType.LongTermCare }.title
        )
        assertEquals(
            "Add mortgage payment",
            analysis.comparisons.first { it.type == LabComparisonType.MortgagePayoff }.title
        )
    }

    @Test
    fun allocationOptimizerDefaultUsesFiveHundredSimulationsPerMix() {
        assertEquals(500, ScenarioLabAnalyzer.ALLOCATION_OPTIMIZATION_SIMULATIONS)
    }

    @Test
    fun spendingSweepUsesComparableRandomPaths() {
        val analysis = ScenarioLabAnalyzer.analyze(
            scenario = sampleBaseScenario(),
            quickSimulations = 100,
            targetEstimateSimulations = 50
        )
        val rows = analysis.sweeps
            .first { it.type == LabSweepType.SpendingFlexibility }
            .rows

        assertTrue(rows.zipWithNext().all { (lower, higher) -> lower.readiness >= higher.readiness })
    }

    @Test
    fun allocationOptimizerFindsBestMixWithoutRunningWholeLab() {
        val scenario = sampleBaseScenario().copy(
            household = HouseholdProfile(currentAge = 40, retirementAge = 40, targetEndAge = 41),
            accounts = AccountBalances(pretax = 0.0, roth = 480_000.0, taxable = 0.0, cash = 0.0),
            spending = SpendingPlan(
                annualBaseSpending = 12_000.0,
                generalInflationMean = 0.0,
                generalInflationStdDev = 0.0,
                spendingPathModel = SpendingPathModel.Flat,
                lowPortfolioSpendingReduction = 0.0
            ),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            market = MarketAssumptions(
                preRetirementMeanReturn = 0.0,
                preRetirementStdDev = 0.0,
                stockMeanReturn = 0.12,
                stockStdDev = 0.0,
                bondMeanReturn = 0.0,
                bondStdDev = 0.0
            ),
            postRetirementAllocation = PostRetirementAllocationStrategy(stock40xTo45x = 0.0),
            withdrawalStrategy = WithdrawalStrategy(useCashReserveDuringDrawdowns = false)
        )

        val optimization = ScenarioLabAnalyzer.optimizePostRetirementAllocation(
            scenario = scenario,
            quickSimulations = 50
        )

        assertEquals(scenario.id, optimization.scenarioId)
        assertEquals(127, optimization.testedAllocations)
        assertEquals(50, optimization.simulationCount)
        assertEquals(1.0, optimization.recommendedAllocation.stock40xTo45x, 0.0001)
        assertTrue(optimization.recommendedReadiness >= optimization.startingReadiness)
    }
}
