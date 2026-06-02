package com.retirementreadinesslab.model

fun sampleScenarios(): List<RetirementScenario> {
    return listOf(sampleBaseScenario(), sampleLaterScenario(), sampleLeanScenario())
}

fun sampleBaseScenario(): RetirementScenario {
    return RetirementScenario(
        id = "base-plan",
        name = "Base plan",
        household = HouseholdProfile(
            currentAge = 50,
            retirementAge = 58,
            filingStatus = FilingStatus.Single,
            gender = Gender.Male
        ),
        accounts = AccountBalances(
            pretax = 800_000.0,
            roth = 100_000.0,
            taxable = 0.0,
            cash = 50_000.0
        ),
        spending = SpendingPlan(annualBaseSpending = 75_000.0),
        mortgage = MortgagePlan(monthlyPayment = 0.0, yearsLeft = 0),
        healthcare = HealthcarePlan(),
        socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 30_000.0, claimAge = 67),
        market = MarketAssumptions(
            preRetirementMeanReturn = 0.133,
            preRetirementStdDev = 0.162,
            stockMeanReturn = 0.133,
            stockStdDev = 0.162,
            bondMeanReturn = 0.03,
            bondStdDev = 0.06
        ),
        rothConversion = RothConversionStrategy(enabled = false),
        withdrawalStrategy = WithdrawalStrategy(useCashReserveDuringDrawdowns = true),
        longTermCare = LongTermCareAssumption(enabled = true),
        numberOfSimulations = 1_500
    )
}

private fun sampleLaterScenario(): RetirementScenario {
    val base = sampleBaseScenario()
    return base.copy(
        id = "later-retirement",
        name = "Retire at 62",
        household = base.household.copy(retirementAge = 62),
        socialSecurity = base.socialSecurity.copy(claimAge = 70),
        rothConversion = RothConversionStrategy(enabled = true, marginalRateCap = 0.22),
        seed = 20260430L
    )
}

private fun sampleLeanScenario(): RetirementScenario {
    val base = sampleBaseScenario()
    return base.copy(
        id = "lean-plan",
        name = "Lower spending",
        spending = base.spending.copy(annualBaseSpending = 68_000.0),
        socialSecurity = base.socialSecurity.copy(claimAge = 67),
        seed = 20260431L
    )
}
