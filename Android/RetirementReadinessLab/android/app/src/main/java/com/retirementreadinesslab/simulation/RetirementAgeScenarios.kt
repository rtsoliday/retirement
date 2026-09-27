package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.FilingStatus
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.WithdrawalStrategy

internal fun RetirementScenario.withRetirementAgeForAnalysis(retirementAge: Int): RetirementScenario {
    if (retirementAge == household.retirementAge) return this

    val defaults = WithdrawalStrategy.defaultsForRetirementAge(retirementAge)
    return copy(
        household = household.copy(retirementAge = retirementAge),
        withdrawalStrategy = withdrawalStrategy.copy(
            applyEarlyWithdrawalPenalty = defaults.applyEarlyWithdrawalPenalty
        )
    )
}

internal fun RetirementScenario.latestRetirementAgeForAnalysis(maxRetirementAge: Int): Int {
    val primaryLimit = minOf(maxRetirementAge, household.targetEndAge - 1)
    if (household.filingStatus != FilingStatus.Married) return primaryLimit

    // Both people must still be below the projection cap when retirement starts.
    val spouseLimit = household.currentAge +
        (household.targetEndAge - household.spouseCurrentAge) - 1
    return minOf(primaryLimit, spouseLimit)
}
