package com.retirementreadinesslab.simulation

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
}
