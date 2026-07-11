package com.retirementreadinesslab.simulation

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
