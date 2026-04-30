package com.retirementreadinesslab.model

import org.junit.Assert.assertTrue
import org.junit.Test

class RetirementScenarioValidationTest {
    @Test
    fun sampleScenarioIsValid() {
        assertTrue(sampleBaseScenario().validate().isEmpty())
    }

    @Test
    fun rejectsNegativeAdvancedAssumptions() {
        val scenario = sampleBaseScenario().copy(
            healthcare = sampleBaseScenario().healthcare.copy(healthcareInflationStdDev = -0.01),
            market = sampleBaseScenario().market.copy(stockStdDev = -0.10),
            longTermCare = sampleBaseScenario().longTermCare.copy(annualCost = -1.0)
        )

        val errors = scenario.validate()

        assertTrue(errors.any { it.contains("Healthcare inflation volatility") })
        assertTrue(errors.any { it.contains("Market return volatility") })
        assertTrue(errors.any { it.contains("Long-term care cost") })
    }

    @Test
    fun rejectsUnsupportedRothBracketCapWhenEnabled() {
        val scenario = sampleBaseScenario().copy(
            rothConversion = RothConversionStrategy(enabled = true, marginalRateCap = 0.23)
        )

        val errors = scenario.validate()

        assertTrue(errors.any { it.contains("Roth conversion bracket cap") })
    }

    @Test
    fun warningsFlagSuspiciousButAllowedAssumptions() {
        val scenario = sampleBaseScenario().copy(
            household = sampleBaseScenario().household.copy(retirementAge = 48),
            socialSecurity = sampleBaseScenario().socialSecurity.copy(annualBenefitAt67 = 0.0),
            healthcare = sampleBaseScenario().healthcare.copy(preMedicareMonthlyPremium = 0.0),
            longTermCare = LongTermCareAssumption(enabled = false),
            market = sampleBaseScenario().market.copy(stockMeanReturn = 0.12),
            numberOfSimulations = 250
        )

        val warnings = scenario.warnings()

        assertTrue(warnings.any { it.title.contains("early retirement", ignoreCase = true) })
        assertTrue(warnings.any { it.title.contains("Social Security") })
        assertTrue(warnings.any { it.title.contains("healthcare premium", ignoreCase = true) })
        assertTrue(warnings.any { it.title.contains("High return") })
        assertTrue(warnings.any { it.title.contains("simulation count", ignoreCase = true) })
    }
}
