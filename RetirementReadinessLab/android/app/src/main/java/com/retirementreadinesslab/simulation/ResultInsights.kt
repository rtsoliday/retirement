package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.RiskLevel
import com.retirementreadinesslab.model.SimulationResult
import java.text.NumberFormat
import java.util.Locale

data class ResultInsightSummary(
    val title: String,
    val summary: String,
    val level: RiskLevel,
    val bullets: List<String>
)

object ResultInsights {
    fun summarize(
        scenario: RetirementScenario,
        result: SimulationResult
    ): ResultInsightSummary {
        val level = when {
            result.successProbability >= 0.82 -> RiskLevel.Healthy
            result.successProbability >= 0.65 -> RiskLevel.Watch
            else -> RiskLevel.AtRisk
        }
        val title = when (level) {
            RiskLevel.Healthy -> "Durable in most modeled paths"
            RiskLevel.Watch -> "Workable but assumption-sensitive"
            RiskLevel.AtRisk -> "Needs pressure relief"
        }
        val summary = "${result.successProbability.percent()} of simulations avoided portfolio depletion " +
            "across the mortality-modeled horizon. Primary modeled pressure: ${result.riskBreakdown.primaryRisk}."

        return ResultInsightSummary(
            title = title,
            summary = summary,
            level = level,
            bullets = buildBullets(scenario, result)
        )
    }

    private fun buildBullets(
        scenario: RetirementScenario,
        result: SimulationResult
    ): List<String> {
        val bullets = mutableListOf<String>()
        bullets += "Median ending balance is ${result.medianEndingBalance.money()}; the 10th percentile path ends at ${result.pessimisticEndingBalance.money()}."
        bullets += failureTiming(result)
        bullets += healthcareBridge(scenario)
        bullets += taxExposure(scenario)
        bullets += "Next useful test: ${result.riskBreakdown.recommendedNextTest}"
        return bullets
    }

    private fun failureTiming(result: SimulationResult): String {
        val bucket = result.failureAgeBuckets.maxByOrNull { it.count }
        return if (bucket == null) {
            "No simulated path depleted assets before death or the projection cap in this run."
        } else {
            "Failed paths most often depleted assets around ages ${bucket.label}."
        }
    }

    private fun healthcareBridge(scenario: RetirementScenario): String {
        val preMedicareYears = (65 - scenario.household.retirementAge).coerceAtLeast(0)
        return when {
            preMedicareYears > 0 -> {
                "The plan bridges $preMedicareYears pre-Medicare years with " +
                    "${(scenario.healthcare.preMedicareMonthlyPremium * 12.0).money()} of starting annual premiums."
            }
            scenario.healthcare.includeMedicarePremiums -> {
                "Medicare Parts B/D premiums and modeled IRMAA tiers are included after age 65."
            }
            else -> {
                "Medicare premiums are excluded, so modeled healthcare costs may be understated after age 65."
            }
        }
    }

    private fun taxExposure(scenario: RetirementScenario): String {
        val pretaxShare = scenario.accounts.pretax / scenario.accounts.total.coerceAtLeast(1.0)
        return if (pretaxShare >= 0.60) {
            "Pre-tax accounts are ${pretaxShare.percent()} of current assets, so withdrawal timing and conversions can affect taxes."
        } else {
            "Assets are spread across tax buckets, which gives the withdrawal strategy more flexibility."
        }
    }

    private fun Double.money(): String {
        return NumberFormat.getCurrencyInstance(Locale.US).apply {
            maximumFractionDigits = 0
        }.format(this)
    }

    private fun Double.percent(): String {
        return "${"%.0f".format(Locale.US, this * 100.0)}%"
    }
}
