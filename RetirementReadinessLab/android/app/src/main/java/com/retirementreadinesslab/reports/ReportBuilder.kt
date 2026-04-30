package com.retirementreadinesslab.reports

import com.retirementreadinesslab.compliance.ComplianceText
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SimulationResult
import com.retirementreadinesslab.model.warnings
import com.retirementreadinesslab.simulation.MedicarePremiums
import com.retirementreadinesslab.simulation.ResultInsights
import com.retirementreadinesslab.simulation.ScenarioComparison
import java.text.NumberFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ReportBuilder {
    fun buildScenarioCsv(
        scenarios: List<RetirementScenario>,
        results: Map<String, SimulationResult>
    ): String {
        val comparison = scenarios.firstOrNull()?.let { baseline ->
            ScenarioComparison.build(scenarios, results, baseline)
        }
        val rowsById = comparison?.rows.orEmpty().associateBy { it.scenarioId }
        return buildString {
            appendLine(
                listOf(
                    "Scenario",
                    "Retirement age",
                    "Claim age",
                    "Annual spending",
                    "Total assets",
                    "Readiness",
                    "Median ending balance",
                    "10th percentile ending balance",
                    "90th percentile ending balance",
                    "Median failure age",
                    "Most common failure window",
                    "Changed assumptions",
                    "Primary risk"
                ).joinToString(",")
            )
            scenarios.forEach { scenario ->
                val result = results[scenario.id]
                val comparisonRow = rowsById[scenario.id]
                appendLine(
                    listOf(
                        scenario.name.csv(),
                        scenario.household.retirementAge.toString(),
                        scenario.socialSecurity.claimAge.toString(),
                        scenario.spending.annualBaseSpending.csvNumber(),
                        scenario.accounts.total.csvNumber(),
                        result?.successProbability?.percent() ?: "",
                        result?.medianEndingBalance?.csvNumber() ?: "",
                        result?.pessimisticEndingBalance?.csvNumber() ?: "",
                        result?.optimisticEndingBalance?.csvNumber() ?: "",
                        result?.medianFailureAge?.toString() ?: "",
                        comparisonRow?.mostCommonFailureWindow?.csv() ?: "",
                        comparisonRow?.changedAssumptions.orEmpty().joinToString("; ").csv(),
                        result?.riskBreakdown?.primaryRisk?.csv() ?: ""
                    ).joinToString(",")
                )
            }
        }
    }

    fun buildTextReport(
        scenario: RetirementScenario,
        result: SimulationResult?
    ): String {
        val generated = result?.generatedAtEpochMillis ?: System.currentTimeMillis()
        return buildString {
            appendLine("Retirement Readiness Lab")
            appendLine("Scenario report")
            appendLine("Generated: ${formatDate(generated)}")
            appendLine()
            appendLine("Summary")
            appendLine("- Scenario: ${scenario.name}")
            appendLine("- Retirement age: ${scenario.household.retirementAge}")
            appendLine("- Horizon model: Mortality table, capped at age ${scenario.household.targetEndAge}")
            appendLine("- Readiness: ${result?.successProbability?.percent() ?: "Not run"}")
            appendLine("- Median ending balance: ${result?.medianEndingBalance?.money() ?: "Not run"}")
            appendLine("- 10th percentile ending balance: ${result?.pessimisticEndingBalance?.money() ?: "Not run"}")
            appendLine("- 90th percentile ending balance: ${result?.optimisticEndingBalance?.money() ?: "Not run"}")
            appendLine("- Median failure age: ${result?.medianFailureAge?.toString() ?: "N/A"}")
            appendLine("- Primary modeled risk: ${result?.riskBreakdown?.primaryRisk ?: "Not run"}")
            if (result != null) {
                appendLine("- Market risk: ${result.riskBreakdown.market.label()}")
                appendLine("- Longevity risk: ${result.riskBreakdown.longevity.label()}")
                appendLine("- Healthcare risk: ${result.riskBreakdown.healthcare.label()}")
                appendLine("- Tax risk: ${result.riskBreakdown.taxes.label()}")
                appendLine("- Spending risk: ${result.riskBreakdown.spending.label()}")
                val insight = ResultInsights.summarize(scenario, result)
                appendLine()
                appendLine("Readiness interpretation")
                appendLine("- ${insight.title}: ${insight.summary}")
                insight.bullets.forEach { bullet ->
                    appendLine("- $bullet")
                }
            }
            if (!result?.failureAgeBuckets.isNullOrEmpty()) {
                appendLine()
                appendLine("Failure timing")
                result.failureAgeBuckets.forEach { bucket ->
                    appendLine("- Ages ${bucket.label}: ${bucket.count} runs, ${bucket.shareOfFailures.percent()} of failed runs")
                }
            }
            if (result != null) {
                appendLine()
                appendLine("Calculation provenance")
                appendLine("- Engine: ${result.provenance.engineVersion}")
                appendLine("- Cadence: ${result.provenance.engineCadence}")
                appendLine("- Tax table: ${result.provenance.taxTableVersion}")
                appendLine("- Mortality model: ${result.provenance.mortalityModelVersion}")
                appendLine("- Random seed: ${result.provenance.randomSeed}")
                appendLine("- Simulations: ${result.provenance.simulationCount}")
                appendLine("- Assumption fingerprint: ${result.provenance.assumptionFingerprint}")
            }
            val warnings = scenario.warnings()
            if (warnings.isNotEmpty()) {
                appendLine()
                appendLine("Assumption checks")
                warnings.forEach { warning ->
                    appendLine("- ${warning.title}: ${warning.detail}")
                }
            }
            appendLine()
            appendLine("Key assumptions")
            appendLine("- Filing status: ${scenario.household.filingStatus.label}")
            appendLine("- Mortality table: ${scenario.household.gender.label}")
            appendLine("- Current age: ${scenario.household.currentAge}")
            appendLine("- Projection cap: Age ${scenario.household.targetEndAge}")
            appendLine("- Annual spending: ${scenario.spending.annualBaseSpending.money()}")
            appendLine("- General inflation average: ${scenario.spending.generalInflationMean.percent()}")
            appendLine("- General inflation +/- swing: ${scenario.spending.generalInflationStdDev.percent()}")
            appendLine("- Pre-tax balance: ${scenario.accounts.pretax.money()}")
            appendLine("- Roth balance: ${scenario.accounts.roth.money()}")
            appendLine("- Taxable balance: ${scenario.accounts.taxable.money()}")
            appendLine("- Cash reserve: ${scenario.accounts.cash.money()}")
            appendLine("- Social Security at 67: ${scenario.socialSecurity.annualBenefitAt67.money()}")
            appendLine("- Social Security claim age: ${scenario.socialSecurity.claimAge}")
            appendLine("- Pre-Medicare monthly premium: ${scenario.healthcare.preMedicareMonthlyPremium.money()}")
            appendLine("- Healthcare inflation average: ${scenario.healthcare.healthcareInflationMean.percent()}")
            appendLine("- Healthcare inflation +/- swing: ${scenario.healthcare.healthcareInflationStdDev.percent()}")
            appendLine("- Medicare premiums: ${if (scenario.healthcare.includeMedicarePremiums) "Included" else "Excluded"}")
            appendLine("- Medicare premium model: ${MedicarePremiums.PREMIUM_TABLE_VERSION}")
            appendLine("- Mortgage payment: ${scenario.mortgage.monthlyPayment.money()}")
            appendLine("- Mortgage years left: ${scenario.mortgage.yearsLeft}")
            appendLine("- Pre-retirement return average: ${scenario.market.preRetirementMeanReturn.percent()}")
            appendLine("- Pre-retirement return +/- swing: ${scenario.market.preRetirementStdDev.percent()}")
            appendLine("- Stock return average: ${scenario.market.stockMeanReturn.percent()}")
            appendLine("- Stock return +/- swing: ${scenario.market.stockStdDev.percent()}")
            appendLine("- Bond return average: ${scenario.market.bondMeanReturn.percent()}")
            appendLine("- Bond return +/- swing: ${scenario.market.bondStdDev.percent()}")
            appendLine("- Roth conversions: ${if (scenario.rothConversion.enabled) "Enabled" else "Disabled"}")
            appendLine("- Roth conversion bracket cap: ${scenario.rothConversion.marginalRateCap.percent()}")
            appendLine("- Cash reserve drawdown: ${if (scenario.withdrawalStrategy.useCashReserveDuringDrawdowns) "Enabled" else "Disabled"}")
            appendLine("- Drawdown trigger: ${scenario.withdrawalStrategy.drawdownTrigger.percent()}")
            appendLine("- Long-term care risk: ${if (scenario.longTermCare.enabled) "Enabled" else "Disabled"}")
            appendLine("- Long-term care annual cost: ${scenario.longTermCare.annualCost.money()}")
            appendLine("- Long-term care duration: ${scenario.longTermCare.averageDurationYears} years")
            appendLine("- Simulations: ${scenario.numberOfSimulations}")
            appendLine("- Random seed: ${scenario.seed}")
            appendLine("- Federal tax table: 2024 brackets")
            appendLine()
            appendLine("Suggested next test")
            appendLine(result?.riskBreakdown?.recommendedNextTest ?: "Run this scenario before comparing next steps.")
            appendLine()
            appendLine("Privacy note")
            appendLine(ComplianceText.reportPrivacyNotice)
            appendLine()
            appendLine("Disclaimer")
            appendLine(ComplianceText.educationalDisclaimer)
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

    private fun com.retirementreadinesslab.model.RiskLevel.label(): String {
        return when (this) {
            com.retirementreadinesslab.model.RiskLevel.Healthy -> "Healthy"
            com.retirementreadinesslab.model.RiskLevel.Watch -> "Watch"
            com.retirementreadinesslab.model.RiskLevel.AtRisk -> "At risk"
        }
    }

    private fun Double.csvNumber(): String {
        return "%.2f".format(Locale.US, this)
    }

    private fun String.csv(): String {
        val escaped = replace("\"", "\"\"")
        return if (escaped.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
            "\"$escaped\""
        } else {
            escaped
        }
    }

    private fun formatDate(epochMillis: Long): String {
        return SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date(epochMillis))
    }
}
