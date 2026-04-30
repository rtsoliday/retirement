package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.RothConversionStrategy
import com.retirementreadinesslab.model.sampleBaseScenario
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScenarioComparisonTest {
    @Test
    fun comparisonHighlightsChangedAssumptionsAndBestRow() {
        val baseline = sampleBaseScenario().copy(numberOfSimulations = 60)
        val alternative = baseline.copy(
            id = "later",
            name = "Later with Roth",
            household = baseline.household.copy(retirementAge = baseline.household.retirementAge + 2),
            socialSecurity = baseline.socialSecurity.copy(claimAge = 70),
            rothConversion = RothConversionStrategy(enabled = true),
            numberOfSimulations = 60,
            seed = baseline.seed + 1
        )
        val results = listOf(baseline, alternative).associate { scenario ->
            scenario.id to RetirementSimulator.run(scenario)
        }

        val summary = ScenarioComparison.build(
            scenarios = listOf(baseline, alternative),
            results = results,
            baseline = baseline
        )

        assertEquals(2, summary.rows.size)
        assertEquals("Baseline", summary.rows.first().changedAssumptions.single())
        assertTrue(summary.rows[1].changedAssumptions.any { it.contains("Retire") })
        assertTrue(summary.rows[1].changedAssumptions.any { it == "Claim 70" })
        assertTrue(summary.rows[1].changedAssumptions.any { it == "Roth on" })
        assertTrue(summary.bestReadinessRow != null)
    }
}
