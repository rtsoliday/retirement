package com.retirementreadinesslab.model

enum class FilingStatus(val label: String) {
    Single("Single"),
    Married("Married"),
    HeadOfHousehold("Head of household")
}

enum class Gender(val label: String) {
    Male("Male"),
    Female("Female")
}

enum class RiskLevel {
    Healthy,
    Watch,
    AtRisk
}

enum class ScenarioWarningSeverity {
    Note,
    Watch
}

const val DEFAULT_PROJECTION_END_AGE = 119

data class ScenarioWarning(
    val title: String,
    val detail: String,
    val severity: ScenarioWarningSeverity = ScenarioWarningSeverity.Watch
)

data class HouseholdProfile(
    val currentAge: Int,
    val retirementAge: Int,
    val targetEndAge: Int = DEFAULT_PROJECTION_END_AGE,
    val filingStatus: FilingStatus = FilingStatus.Single,
    val gender: Gender = Gender.Male
)

data class AccountBalances(
    val pretax: Double,
    val roth: Double,
    val taxable: Double = 0.0,
    val cash: Double
) {
    val total: Double
        get() = pretax + roth + taxable + cash
}

data class SpendingPlan(
    val annualBaseSpending: Double,
    val generalInflationMean: Double = 0.033,
    val generalInflationStdDev: Double = 0.04
)

data class MortgagePlan(
    val monthlyPayment: Double = 0.0,
    val yearsLeft: Int = 0
)

data class HealthcarePlan(
    val preMedicareMonthlyPremium: Double = 650.0,
    val healthcareInflationMean: Double = 0.055,
    val healthcareInflationStdDev: Double = 0.02,
    val includeMedicarePremiums: Boolean = true
)

data class SocialSecurityPlan(
    val annualBenefitAt67: Double,
    val claimAge: Int = 67
)

data class MarketAssumptions(
    val preRetirementMeanReturn: Double = 0.08,
    val preRetirementStdDev: Double = 0.16,
    val stockMeanReturn: Double = 0.08,
    val stockStdDev: Double = 0.18,
    val bondMeanReturn: Double = 0.03,
    val bondStdDev: Double = 0.06
)

data class RothConversionStrategy(
    val enabled: Boolean = false,
    val marginalRateCap: Double = 0.22
)

data class WithdrawalStrategy(
    val useCashReserveDuringDrawdowns: Boolean = true,
    val drawdownTrigger: Double = -0.01
)

data class LongTermCareAssumption(
    val enabled: Boolean = false,
    val annualCost: Double = 100_000.0,
    val averageDurationYears: Int = 3
)

data class RetirementScenario(
    val id: String,
    val name: String,
    val household: HouseholdProfile,
    val accounts: AccountBalances,
    val spending: SpendingPlan,
    val mortgage: MortgagePlan = MortgagePlan(),
    val healthcare: HealthcarePlan = HealthcarePlan(),
    val socialSecurity: SocialSecurityPlan,
    val market: MarketAssumptions = MarketAssumptions(),
    val rothConversion: RothConversionStrategy = RothConversionStrategy(),
    val withdrawalStrategy: WithdrawalStrategy = WithdrawalStrategy(),
    val longTermCare: LongTermCareAssumption = LongTermCareAssumption(),
    val numberOfSimulations: Int = 1_500,
    val seed: Long = 20260429L
)

data class OutcomeBand(
    val age: Int,
    val pessimistic: Double,
    val median: Double,
    val optimistic: Double
)

data class FailureAgeBucket(
    val label: String,
    val count: Int,
    val shareOfFailures: Double
)

data class RiskBreakdown(
    val market: RiskLevel,
    val longevity: RiskLevel,
    val healthcare: RiskLevel,
    val taxes: RiskLevel,
    val spending: RiskLevel,
    val primaryRisk: String,
    val recommendedNextTest: String
)

data class CalculationProvenance(
    val engineVersion: String,
    val engineCadence: String,
    val taxTableVersion: String,
    val mortalityModelVersion: String,
    val randomSeed: Long,
    val simulationCount: Int,
    val assumptionFingerprint: String
)

data class SimulationResult(
    val scenarioId: String,
    val successProbability: Double,
    val medianEndingBalance: Double,
    val pessimisticEndingBalance: Double,
    val optimisticEndingBalance: Double,
    val medianFailureAge: Int?,
    val failureAgeBuckets: List<FailureAgeBucket>,
    val balanceBands: List<OutcomeBand>,
    val riskBreakdown: RiskBreakdown,
    val provenance: CalculationProvenance,
    val generatedAtEpochMillis: Long
)

fun RetirementScenario.validate(): List<String> {
    val errors = mutableListOf<String>()
    if (household.currentAge <= 0) errors += "Current age must be positive."
    if (household.retirementAge < household.currentAge) {
        errors += "Retirement age must be greater than or equal to current age."
    }
    if (household.targetEndAge <= household.retirementAge) {
        errors += "Projection end age must be greater than retirement age."
    }
    if (socialSecurity.claimAge !in 62..70) {
        errors += "Social Security claim age must be between 62 and 70."
    }
    if (accounts.pretax < 0 || accounts.roth < 0 || accounts.taxable < 0 || accounts.cash < 0) {
        errors += "Account balances cannot be negative."
    }
    if (spending.annualBaseSpending < 0) errors += "Annual spending cannot be negative."
    if (spending.generalInflationStdDev < 0) errors += "General inflation swing cannot be negative."
    if (mortgage.monthlyPayment < 0 || mortgage.yearsLeft < 0) {
        errors += "Mortgage payment and years left cannot be negative."
    }
    if (healthcare.preMedicareMonthlyPremium < 0) errors += "Healthcare premium cannot be negative."
    if (healthcare.healthcareInflationStdDev < 0) errors += "Healthcare inflation swing cannot be negative."
    if (socialSecurity.annualBenefitAt67 < 0) errors += "Social Security estimate cannot be negative."
    if (
        market.preRetirementStdDev < 0 ||
        market.stockStdDev < 0 ||
        market.bondStdDev < 0
    ) {
        errors += "Market return swing cannot be negative."
    }
    if (rothConversion.enabled && !supportedRothBracketCaps.any { kotlin.math.abs(it - rothConversion.marginalRateCap) < 0.0001 }) {
        errors += "Roth conversion bracket cap must match a supported federal tax bracket."
    }
    if (longTermCare.annualCost < 0) errors += "Long-term care cost cannot be negative."
    if (longTermCare.averageDurationYears < 1) errors += "Long-term care duration must be at least one year."
    if (numberOfSimulations < 1) errors += "Simulation count must be positive."
    return errors
}

fun RetirementScenario.warnings(): List<ScenarioWarning> {
    val warnings = mutableListOf<ScenarioWarning>()
    val yearsToRetirement = household.retirementAge - household.currentAge
    val retirementHorizon = household.targetEndAge - household.retirementAge
    val spendingRatio = spending.annualBaseSpending / accounts.total.coerceAtLeast(1.0)

    if (household.retirementAge < 50) {
        warnings += ScenarioWarning(
            title = "Very early retirement age",
            detail = "Retiring before 50 creates a long drawdown period and makes healthcare assumptions especially important."
        )
    }
    if (household.retirementAge < 55 && retirementHorizon > 50) {
        warnings += ScenarioWarning(
            title = "Long retirement horizon",
            detail = "The mortality-modeled projection includes a long potential drawdown period, so inflation and healthcare assumptions carry extra weight."
        )
    }
    if (spendingRatio > 0.07) {
        warnings += ScenarioWarning(
            title = "High spending draw",
            detail = "Annual spending is more than 7% of current assets before taxes, healthcare, and mortgage costs."
        )
    }
    if (spending.generalInflationMean < 0.015) {
        warnings += ScenarioWarning(
            title = "Low inflation assumption",
            detail = "General inflation below 1.5% may understate long-term spending pressure.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if (market.stockMeanReturn > 0.10 || market.preRetirementMeanReturn > 0.10) {
        warnings += ScenarioWarning(
            title = "High return assumption",
            detail = "Expected stock or pre-retirement returns above 10% may make readiness look stronger than a conservative model."
        )
    }
    if (household.retirementAge < 65 && healthcare.preMedicareMonthlyPremium <= 0.0) {
        warnings += ScenarioWarning(
            title = "Missing pre-Medicare healthcare premium",
            detail = "Retiring before 65 without a healthcare premium can materially understate spending."
        )
    }
    if (!healthcare.includeMedicarePremiums) {
        warnings += ScenarioWarning(
            title = "Medicare premiums excluded",
            detail = "Medicare Parts B/D premiums are not included in this scenario.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if (!longTermCare.enabled) {
        warnings += ScenarioWarning(
            title = "Long-term care stress disabled",
            detail = "The plan does not include a late-life care shock.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if (socialSecurity.annualBenefitAt67 <= 0.0) {
        warnings += ScenarioWarning(
            title = "No Social Security estimate",
            detail = "A zero Social Security estimate may be intentional, but it materially changes bridge and withdrawal needs.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if (mortgage.yearsLeft > 0 && mortgage.yearsLeft > yearsToRetirement + retirementHorizon) {
        warnings += ScenarioWarning(
            title = "Mortgage extends beyond horizon",
            detail = "Mortgage years left exceeds the internal mortality projection cap."
        )
    }
    if (numberOfSimulations < 500) {
        warnings += ScenarioWarning(
            title = "Low simulation count",
            detail = "Use at least 500 simulations before relying on scenario comparisons.",
            severity = ScenarioWarningSeverity.Note
        )
    }

    return warnings
}

private val supportedRothBracketCaps = listOf(0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37)
