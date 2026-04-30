package com.retirementreadinesslab.reports

import com.retirementreadinesslab.model.sampleBaseScenario
import com.retirementreadinesslab.simulation.RetirementSimulator
import org.junit.Assert.assertTrue
import org.junit.Test

class ReportBuilderTest {
    @Test
    fun reportIncludesScenarioResultAssumptionsAndDisclaimer() {
        val scenario = sampleBaseScenario().copy(numberOfSimulations = 50)
        val result = RetirementSimulator.run(scenario)

        val report = ReportBuilder.buildTextReport(scenario, result)

        assertTrue(report.contains("Retirement Readiness Lab"))
        assertTrue(report.contains("Scenario: Base plan"))
        assertTrue(report.contains("Retirement age: 58"))
        assertTrue(report.contains("Readiness:"))
        assertTrue(report.contains("Annual spending: $75,000"))
        assertTrue(report.contains("Healthcare inflation mean: 6%"))
        assertTrue(report.contains("Stock return volatility: 18%"))
        assertTrue(report.contains("Federal tax table: 2024 brackets"))
        assertTrue(report.contains("Calculation provenance"))
        assertTrue(report.contains("Assumption fingerprint"))
        assertTrue(report.contains("2026.04-monthly-mvp"))
        assertTrue(report.contains("Privacy note"))
        assertTrue(report.contains("generated locally from user-entered scenario data"))
        assertTrue(report.contains("not financial, tax, legal, or investment advice"))
    }

    @Test
    fun csvIncludesScenarioComparisonRows() {
        val scenario = sampleBaseScenario().copy(name = "Base, plan", numberOfSimulations = 50)
        val result = RetirementSimulator.run(scenario)

        val csv = ReportBuilder.buildScenarioCsv(
            scenarios = listOf(scenario),
            results = mapOf(scenario.id to result)
        )

        assertTrue(csv.contains("Scenario,Retirement age,Claim age"))
        assertTrue(csv.contains("\"Base, plan\""))
        assertTrue(csv.contains("Most common failure window"))
        assertTrue(csv.contains("Changed assumptions"))
        assertTrue(csv.contains("Baseline"))
        assertTrue(csv.contains("Primary risk"))
    }
}
