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
            household = sampleBaseScenario().household.copy(
                filingStatus = FilingStatus.Married,
                spouseCurrentAge = -1
            ),
            spending = sampleBaseScenario().spending.copy(lowPortfolioSpendingReduction = 1.50),
            healthcare = sampleBaseScenario().healthcare.copy(healthcareInflationStdDev = -0.01),
            mortgage = sampleBaseScenario().mortgage.copy(currentBalance = -1.0, monthsLeft = 12),
            rent = RentPlan(monthlyRent = -1.0),
            home = HomePlan(currentValue = -1.0),
            socialSecurity = sampleBaseScenario().socialSecurity.copy(spouseClaimAge = 59),
            guaranteedIncome = GuaranteedIncomePlan(annualIncome = -1.0, startAge = -1, survivorPercent = 1.50),
            market = sampleBaseScenario().market.copy(stockStdDev = -0.10),
            postRetirementAllocation = sampleBaseScenario().postRetirementAllocation.copy(stock50xOrMore = 1.20),
            longTermCare = sampleBaseScenario().longTermCare.copy(annualCost = -1.0)
        )

        val errors = scenario.validate()

        assertTrue(errors.any { it.contains("Spending reduction") })
        assertTrue(errors.any { it.contains("Healthcare inflation std dev") })
        assertTrue(errors.any { it.contains("Mortgage") })
        assertTrue(errors.any { it.contains("Mortgage months left") })
        assertTrue(errors.any { it.contains("Rent") })
        assertTrue(errors.any { it.contains("Home value") })
        assertTrue(errors.any { it.contains("Spouse age") })
        assertTrue(errors.any { it.contains("Spouse Social Security") })
        assertTrue(errors.any { it.contains("Guaranteed income cannot be negative") })
        assertTrue(errors.any { it.contains("Guaranteed income start age") })
        assertTrue(errors.any { it.contains("survivor benefit") })
        assertTrue(errors.any { it.contains("Market return std dev") })
        assertTrue(errors.any { it.contains("Post-retirement investment ratios") })
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
    fun rejectsNonFiniteAndNegativeImportedBudgetAmounts() {
        val scenario = sampleBaseScenario().copy(
            accounts = sampleBaseScenario().accounts.copy(cash = Double.NaN),
            budget = BudgetProfile(
                annualPropertyTaxes = -1.0,
                monthlyBudgets = listOf(
                    MonthlyBudget(
                        month = "2026-06",
                        cashAndAtmWithdrawals = -25.0
                    )
                )
            )
        )

        val errors = scenario.validate()

        assertTrue(errors.any { it.contains("finite") })
        assertTrue(errors.any { it.contains("Budget amounts") })
    }

    @Test
    fun rejectsImportedPercentagesOutsideUiSupportedRanges() {
        val scenario = sampleBaseScenario().copy(
            spending = sampleBaseScenario().spending.copy(generalInflationMean = 0.50),
            healthcare = sampleBaseScenario().healthcare.copy(healthcareInflationMean = 0.50),
            guaranteedIncome = sampleBaseScenario().guaranteedIncome.copy(annualIncrease = 0.50),
            market = sampleBaseScenario().market.copy(stockMeanReturn = 0.50),
            withdrawalStrategy = sampleBaseScenario().withdrawalStrategy.copy(drawdownTrigger = 0.50),
            numberOfSimulations = 10_001
        )

        val errors = scenario.validate()

        assertTrue(errors.any { it.contains("General inflation assumptions") })
        assertTrue(errors.any { it.contains("Healthcare inflation assumptions") })
        assertTrue(errors.any { it.contains("Guaranteed income annual increase") })
        assertTrue(errors.any { it.contains("Market return assumptions") })
        assertTrue(errors.any { it.contains("drawdown trigger") })
        assertTrue(errors.any { it.contains("10,000") })
    }

    @Test
    fun warningsFlagSuspiciousButAllowedAssumptions() {
        val scenario = sampleBaseScenario().copy(
            household = sampleBaseScenario().household.copy(retirementAge = 48),
            socialSecurity = sampleBaseScenario().socialSecurity.copy(annualBenefitAt67 = 0.0),
            healthcare = sampleBaseScenario().healthcare.copy(preMedicareMonthlyPremium = 0.0),
            longTermCare = LongTermCareAssumption(enabled = false),
            market = sampleBaseScenario().market.copy(stockMeanReturn = 0.15),
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
