package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SimulationResult
import kotlin.math.abs

data class ScenarioComparisonSummary(
    val baselineScenarioId: String,
    val rows: List<ScenarioComparisonRow>
) {
    val bestReadinessRow: ScenarioComparisonRow?
        get() = rows
            .filter { it.readiness != null }
            .maxWithOrNull(
                compareBy<ScenarioComparisonRow> { it.readiness ?: -1.0 }
                    .thenBy { it.medianEndingBalance ?: Double.NEGATIVE_INFINITY }
            )
}

data class ScenarioComparisonRow(
    val scenarioId: String,
    val scenarioName: String,
    val readiness: Double?,
    val medianEndingBalance: Double?,
    val pessimisticEndingBalance: Double?,
    val optimisticEndingBalance: Double?,
    val medianFailureAge: Int?,
    val primaryRisk: String?,
    val mostCommonFailureWindow: String,
    val changedAssumptions: List<String>
)

object ScenarioComparison {
    fun build(
        scenarios: List<RetirementScenario>,
        results: Map<String, SimulationResult>,
        baseline: RetirementScenario = scenarios.first()
    ): ScenarioComparisonSummary {
        return ScenarioComparisonSummary(
            baselineScenarioId = baseline.id,
            rows = scenarios.map { scenario ->
                val result = results[scenario.id]
                ScenarioComparisonRow(
                    scenarioId = scenario.id,
                    scenarioName = scenario.name,
                    readiness = result?.successProbability,
                    medianEndingBalance = result?.medianEndingBalance,
                    pessimisticEndingBalance = result?.pessimisticEndingBalance,
                    optimisticEndingBalance = result?.optimisticEndingBalance,
                    medianFailureAge = result?.medianFailureAge,
                    primaryRisk = result?.riskBreakdown?.primaryRisk,
                    mostCommonFailureWindow = result?.mostCommonFailureWindow() ?: "Not run",
                    changedAssumptions = changedAssumptions(baseline, scenario)
                )
            }
        )
    }

    private fun SimulationResult.mostCommonFailureWindow(): String {
        val bucket = failureAgeBuckets.maxByOrNull { it.count }
        return bucket?.let { "Ages ${it.label}" } ?: "No failures"
    }

    private fun changedAssumptions(
        baseline: RetirementScenario,
        scenario: RetirementScenario
    ): List<String> {
        if (baseline.id == scenario.id) return listOf("Baseline")

        val changes = mutableListOf<String>()
        if (baseline.household.retirementAge != scenario.household.retirementAge) {
            changes += "Retire ${signedInt(scenario.household.retirementAge - baseline.household.retirementAge)} years"
        }
        if (!baseline.spending.annualBaseSpending.closeTo(scenario.spending.annualBaseSpending)) {
            changes += "Spend ${signedMoney(scenario.spending.annualBaseSpending - baseline.spending.annualBaseSpending)}"
        }
        if (baseline.socialSecurity.claimAge != scenario.socialSecurity.claimAge) {
            changes += "Claim ${scenario.socialSecurity.claimAge}"
        }
        if (baseline.rothConversion.enabled != scenario.rothConversion.enabled) {
            changes += if (scenario.rothConversion.enabled) "Roth on" else "Roth off"
        } else if (
            scenario.rothConversion.enabled &&
            !baseline.rothConversion.marginalRateCap.closeTo(scenario.rothConversion.marginalRateCap)
        ) {
            changes += "Roth cap ${wholePercent(scenario.rothConversion.marginalRateCap)}"
        }
        if (baseline.withdrawalStrategy.useCashReserveDuringDrawdowns != scenario.withdrawalStrategy.useCashReserveDuringDrawdowns) {
            changes += if (scenario.withdrawalStrategy.useCashReserveDuringDrawdowns) "Cash reserve on" else "Cash reserve off"
        }
        if (baseline.longTermCare.enabled != scenario.longTermCare.enabled) {
            changes += if (scenario.longTermCare.enabled) "LTC on" else "LTC off"
        }
        if (!baseline.healthcare.preMedicareMonthlyPremium.closeTo(scenario.healthcare.preMedicareMonthlyPremium)) {
            changes += "Health premium ${signedMoney((scenario.healthcare.preMedicareMonthlyPremium - baseline.healthcare.preMedicareMonthlyPremium) * 12.0)}/yr"
        }
        if (!baseline.healthcare.healthcareInflationMean.closeTo(scenario.healthcare.healthcareInflationMean)) {
            changes += "Health inflation ${wholePercent(scenario.healthcare.healthcareInflationMean)}"
        }
        if (
            !baseline.mortgage.monthlyPayment.closeTo(scenario.mortgage.monthlyPayment) ||
            baseline.mortgage.yearsLeft != scenario.mortgage.yearsLeft
        ) {
            changes += if (scenario.mortgage.monthlyPayment <= 0.0 || scenario.mortgage.yearsLeft <= 0) {
                "Mortgage removed"
            } else {
                "Mortgage changed"
            }
        }
        if (!baseline.accounts.total.closeTo(scenario.accounts.total)) {
            changes += "Assets ${signedMoney(scenario.accounts.total - baseline.accounts.total)}"
        }

        return changes.ifEmpty { listOf("Name only") }
    }

    private fun Double.closeTo(other: Double): Boolean = abs(this - other) < 0.01

    private fun signedInt(value: Int): String {
        return if (value >= 0) "+$value" else value.toString()
    }

    private fun signedMoney(value: Double): String {
        val sign = if (value >= 0.0) "+" else "-"
        val absValue = abs(value)
        val formatted = when {
            absValue >= 1_000_000.0 -> "${"%.1f".format(absValue / 1_000_000.0)}M"
            absValue >= 1_000.0 -> "${"%.0f".format(absValue / 1_000.0)}k"
            else -> "%.0f".format(absValue)
        }
        return "$sign${'$'}$formatted"
    }

    private fun wholePercent(value: Double): String {
        return "${"%.0f".format(value * 100.0)}%"
    }
}
