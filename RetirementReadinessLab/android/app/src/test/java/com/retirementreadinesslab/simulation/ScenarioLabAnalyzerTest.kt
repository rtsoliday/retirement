package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.MarketAssumptions
import com.retirementreadinesslab.model.PostRetirementAllocationStrategy
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
        assertEquals(8, analysis.comparisons.size)
        assertTrue(analysis.comparisons.any { it.type == LabComparisonType.HealthcareInflation })
        assertTrue(analysis.comparisons.any { it.type == LabComparisonType.MarketDownturn })
        assertTrue(analysis.comparisons.any { it.type == LabComparisonType.MortgagePayoff })
        assertTrue(analysis.decisionEstimate.simulationCount >= 50)
    }

    @Test
    fun analyzerNeverBuildsRetirementAgeAtProjectionCap() {
        val scenario = sampleBaseScenario().copy(
            household = HouseholdProfile(currentAge = 58, retirementAge = 59, targetEndAge = 60)
        )

        val analysis = ScenarioLabAnalyzer.analyze(
            scenario = scenario,
            quickSimulations = 50,
            targetEstimateSimulations = 50
        )

        val retirementRows = analysis.sweeps.first { it.type == LabSweepType.RetirementAge }.rows
        assertTrue(retirementRows.isNotEmpty())
        assertTrue(retirementRows.all { row -> row.label.removePrefix("Age ").toInt() < 60 })
        assertEquals(
            0.0,
            analysis.comparisons.first { it.type == LabComparisonType.RetireLater }.readinessDelta,
            0.0001
        )
    }

    @Test
    fun noOpStrategyComparisonUsesSameMonteCarloSample() {
        val scenario = sampleBaseScenario().copy(
            socialSecurity = sampleBaseScenario().socialSecurity.copy(claimAge = 70)
        )

        val analysis = ScenarioLabAnalyzer.analyze(
            scenario = scenario,
            quickSimulations = 50,
            targetEstimateSimulations = 50
        )

        assertEquals(
            0.0,
            analysis.comparisons.first { it.type == LabComparisonType.ClaimLater }.readinessDelta,
            0.0001
        )
    }

    @Test
    fun allocationOptimizerDefaultUsesFiveHundredSimulationsPerMix() {
        assertEquals(500, ScenarioLabAnalyzer.ALLOCATION_OPTIMIZATION_SIMULATIONS)
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
