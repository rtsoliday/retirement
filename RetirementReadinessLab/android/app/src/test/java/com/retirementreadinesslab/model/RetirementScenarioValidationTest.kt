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
    fun rejectsNonFiniteOutOfRangeAndUnsafeImportedValues() {
        val scenario = sampleBaseScenario().copy(
            spending = sampleBaseScenario().spending.copy(annualBaseSpending = Double.NaN),
            budget = BudgetProfile(annualPropertyTaxes = -1.0),
            market = sampleBaseScenario().market.copy(stockMeanReturn = 2.0),
            numberOfSimulations = PRO_SIMULATION_LIMIT + 1
        )

        val errors = scenario.validate()

        assertTrue(errors.any { it.contains("finite") })
        assertTrue(errors.any { it.contains("Budget amounts") })
        assertTrue(errors.any { it.contains("Market return assumptions") })
        assertTrue(errors.any { it.contains("Simulation count cannot exceed") })
    }

    @Test
    fun withdrawalStrategyDefaultsFollowRetirementAge() {
        val age54 = WithdrawalStrategy.defaultsForRetirementAge(54)
        val age58 = WithdrawalStrategy.defaultsForRetirementAge(58)
        val age60 = WithdrawalStrategy.defaultsForRetirementAge(60)

        assertTrue(age54.applyEarlyWithdrawalPenalty)
        assertTrue(!age54.ruleOf55Eligible)
        assertTrue(!age54.seppEligible)
        assertTrue(age58.applyEarlyWithdrawalPenalty)
        assertTrue(!age58.ruleOf55Eligible)
        assertTrue(!age58.seppEligible)
        assertTrue(!age60.applyEarlyWithdrawalPenalty)
        assertTrue(!age60.ruleOf55Eligible)
        assertTrue(!age60.seppEligible)
    }

    @Test
    fun rejectsImportedDurationsThatCanOverflowOrExceedSupportedUiRanges() {
        val scenario = sampleBaseScenario().copy(
            mortgage = sampleBaseScenario().mortgage.copy(yearsLeft = Int.MAX_VALUE),
            longTermCare = sampleBaseScenario().longTermCare.copy(averageDurationYears = Int.MAX_VALUE)
        )

        val errors = scenario.validate()

        assertTrue(errors.any { it.contains("years left") })
        assertTrue(errors.any { it.contains("duration") })
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
