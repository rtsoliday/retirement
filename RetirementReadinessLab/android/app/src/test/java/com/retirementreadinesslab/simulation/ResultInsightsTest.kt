package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.LongTermCareAssumption
import com.retirementreadinesslab.model.RiskLevel
import com.retirementreadinesslab.model.sampleBaseScenario
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ResultInsightsTest {
    @Test
    fun highReadinessScenarioGetsHealthyInterpretation() {
        val scenario = sampleBaseScenario().copy(
            accounts = AccountBalances(pretax = 2_000_000.0, roth = 1_000_000.0, taxable = 1_000_000.0, cash = 250_000.0),
            spending = sampleBaseScenario().spending.copy(annualBaseSpending = 35_000.0),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 650.0),
            longTermCare = LongTermCareAssumption(enabled = false),
            numberOfSimulations = 80
        )
        val result = RetirementSimulator.run(scenario)

        val insight = ResultInsights.summarize(scenario, result)

        assertEquals(RiskLevel.Healthy, insight.level)
        assertTrue(insight.title.contains("Durable"))
        assertTrue(insight.summary.contains("Primary modeled pressure"))
        assertTrue(insight.bullets.any { it.contains("Median ending balance") })
        assertTrue(insight.bullets.any { it.contains("pre-Medicare years") })
    }

    @Test
    fun stressedScenarioGetsAtRiskInterpretation() {
        val scenario = sampleBaseScenario().copy(
            spending = sampleBaseScenario().spending.copy(annualBaseSpending = 140_000.0),
            numberOfSimulations = 80
        )
        val result = RetirementSimulator.run(scenario)

        val insight = ResultInsights.summarize(scenario, result)

        assertEquals(RiskLevel.AtRisk, insight.level)
        assertTrue(insight.title.contains("Needs pressure relief"))
        assertTrue(insight.bullets.any { it.contains("Next useful test") })
    }

    @Test
    fun medicareExclusionIsCalledOutAfterAge65() {
        val scenario = sampleBaseScenario().copy(
            household = sampleBaseScenario().household.copy(currentAge = 66, retirementAge = 66),
            healthcare = sampleBaseScenario().healthcare.copy(includeMedicarePremiums = false),
            numberOfSimulations = 80
        )
        val result = RetirementSimulator.run(scenario)

        val insight = ResultInsights.summarize(scenario, result)

        assertTrue(insight.bullets.any { it.contains("Medicare premiums are excluded") })
    }
}
