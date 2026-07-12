package com.retirementreadinesslab.reports

import com.retirementreadinesslab.model.sampleBaseScenario
import com.retirementreadinesslab.simulation.RetirementSimulator
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReportBuilderTest {
    @Test
    fun reportIncludesResultAssumptionsAndDisclaimer() {
        val scenario = sampleBaseScenario().copy(numberOfSimulations = 50)
        val result = RetirementSimulator.run(scenario)

        val report = ReportBuilder.buildTextReport(scenario, result)

        assertTrue(report.contains("Retirement Readiness Lab"))
        assertTrue(report.contains("Readiness report"))
        assertFalse(report.contains("Base plan"))
        assertFalse(report.contains("Scenario: Base plan"))
        assertTrue(report.contains("Retirement age: 67"))
        assertTrue(report.contains("Horizon model: Mortality table"))
        assertFalse(report.contains("Target end age"))
        assertTrue(report.contains("Readiness:"))
        assertTrue(report.contains("Annual spending: $75,000"))
        assertTrue(report.contains("Spending path: Empirical age decline"))
        assertTrue(report.contains("Spending cut below 50% portfolio: 10%"))
        assertTrue(report.contains("Primary Social Security at FRA: $30,000"))
        assertTrue(report.contains("Primary Social Security claim age: 67"))
        assertTrue(report.contains("Social Security model: SSA retirement, spousal, and survivor rules"))
        assertTrue(report.contains("Annual guaranteed income: $0"))
        assertTrue(report.contains("Guaranteed income survivor benefit: 100%"))
        assertTrue(report.contains("Pre-Medicare monthly premium: $1,250"))
        assertTrue(report.contains("Healthcare inflation average: 4%"))
        assertTrue(report.contains("Healthcare inflation Std Dev: 1.8%"))
        assertTrue(report.contains("Medicare premium model: 2026 modeled Medicare Parts B/D with indexed IRMAA"))
        assertTrue(report.contains("Mortgage time left: 0 years, 0 months"))
        assertTrue(report.contains("Current mortgage balance: $0"))
        assertTrue(report.contains("Current home value: $0"))
        assertTrue(report.contains("Monthly rent: $0"))
        assertTrue(report.contains("Senior apartment rent after home sale: $3,000 in 2026 dollars"))
        assertTrue(report.contains("Current portfolio returns average: 13.3%"))
        assertTrue(report.contains("Current portfolio returns Std Dev: 16.2%"))
        assertTrue(report.contains("Post-retirement stock returns average: 13.3%"))
        assertTrue(report.contains("Post-retirement stock returns Std Dev: 16.2%"))
        assertTrue(report.contains("Post-retirement stock returns basis: Default 13.3% ± 16.2% values based on the last 50 years of the S&P 500"))
        assertTrue(report.contains("Post-retirement investment ratios:"))
        assertTrue(report.contains("Under 30x annual spending: 100% stocks / 0% bonds"))
        assertTrue(report.contains("10% early withdrawal tax: Excluded"))
        assertTrue(report.contains("Rule of 55 assumption: Not applied"))
        assertTrue(report.contains("72(t) / SEPP assumption: Not applied"))
        assertTrue(report.contains("Federal tax table: 2026 brackets with senior-aware deductions"))
        assertTrue(report.contains("Readiness interpretation"))
        assertTrue(report.contains("Next useful test:"))
        assertTrue(report.contains("Calculation provenance"))
        assertTrue(report.contains("Assumption fingerprint"))
        assertTrue(report.contains("2026.07-senior-tax-deductions"))
        assertTrue(report.contains("SSA Trustees Alt2 2025 annual death probabilities"))
        assertTrue(report.contains("Privacy note"))
        assertTrue(report.contains("generated locally from user-entered scenario data"))
        assertTrue(report.contains("not financial, tax, legal, or investment advice"))
    }
}
