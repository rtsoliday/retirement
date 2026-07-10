package com.retirementreadinesslab.model

const val FREE_SIMULATION_LIMIT = 500
const val PRO_SIMULATION_LIMIT = 10_000

data class FeatureAccess(
    val isProUnlocked: Boolean = false
) {
    val maxSimulationCount: Int
        get() = if (isProUnlocked) PRO_SIMULATION_LIMIT else FREE_SIMULATION_LIMIT

    val tierName: String
        get() = if (isProUnlocked) "Pro" else "Free"
}

fun RetirementScenario.forFeatureAccess(access: FeatureAccess): RetirementScenario {
    val simulationCount = numberOfSimulations.coerceIn(1, access.maxSimulationCount)
    val withoutFirstReleaseHiddenFeatures = copy(
        withdrawalStrategy = withdrawalStrategy.copy(useCashReserveDuringDrawdowns = false),
        numberOfSimulations = simulationCount
    )
    if (access.isProUnlocked) {
        return withoutFirstReleaseHiddenFeatures
    }

    return withoutFirstReleaseHiddenFeatures.copy(
        spending = spending.copy(
            lowPortfolioSpendingReduction = 0.0
        ),
        healthcare = healthcare.copy(
            healthcareInflationMean = DEFAULT_HEALTHCARE_INFLATION_MEAN,
            healthcareInflationStdDev = DEFAULT_HEALTHCARE_INFLATION_STD_DEV,
            includeMedicarePremiums = true
        ),
        postRetirementAllocation = PostRetirementAllocationStrategy(),
        rothConversion = RothConversionStrategy(enabled = false),
        withdrawalStrategy = WithdrawalStrategy(
            useCashReserveDuringDrawdowns = false,
            drawdownTrigger = WithdrawalStrategy().drawdownTrigger,
            applyEarlyWithdrawalPenalty =
                household.retirementAge * 12 < PENALTY_FREE_WITHDRAWAL_AGE_MONTHS,
            ruleOf55Eligible = false,
            seppEligible = false
        ),
        longTermCare = longTermCare.copy(enabled = false),
        numberOfSimulations = simulationCount
    )
}
