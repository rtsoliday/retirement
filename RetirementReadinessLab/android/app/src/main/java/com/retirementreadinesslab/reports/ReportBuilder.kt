package com.retirementreadinesslab.reports

import com.retirementreadinesslab.compliance.ComplianceText
import com.retirementreadinesslab.model.PostRetirementAllocationTier
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SENIOR_APARTMENT_MONTHLY_RENT_2026
import com.retirementreadinesslab.model.SimulationResult
import com.retirementreadinesslab.model.SpendingPathModel
import com.retirementreadinesslab.model.warnings
import com.retirementreadinesslab.simulation.MedicarePremiums
import com.retirementreadinesslab.simulation.ResultInsights
import com.retirementreadinesslab.simulation.SocialSecurity
import java.text.NumberFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ReportBuilder {
    fun buildTextReport(
        scenario: RetirementScenario,
        result: SimulationResult?
    ): String {
        val generated = result?.generatedAtEpochMillis ?: System.currentTimeMillis()
        return buildString {
            appendLine("Retirement Readiness Lab")
            appendLine("Readiness report")
            appendLine("Generated: ${formatDate(generated)}")
            appendLine()
            appendLine("Summary")
            appendLine("- Retirement age: ${scenario.household.retirementAge}")
            appendLine("- Horizon model: Mortality table, capped at age ${scenario.household.targetEndAge} per modeled person")
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
            if (scenario.household.filingStatus == com.retirementreadinesslab.model.FilingStatus.Married) {
                appendLine("- Spouse current age: ${scenario.household.spouseCurrentAge}")
                appendLine("- Spouse mortality table: opposite gender, no separate spouse income")
            }
            appendLine("- Current age: ${scenario.household.currentAge}")
            appendLine("- Projection cap: Age ${scenario.household.targetEndAge}")
            appendLine("- Annual spending: ${scenario.spending.annualBaseSpending.money()}")
            appendLine("- Spending path: ${scenario.spending.spendingPathModel.label}")
            appendLine("- Spending path detail: ${scenario.spending.spendingPathModel.detail()}")
            appendLine("- General inflation average: ${scenario.spending.generalInflationMean.percent()}")
            appendLine("- General inflation Std Dev: ${scenario.spending.generalInflationStdDev.percent()}")
            appendLine("- Spending cut below 50% portfolio: ${scenario.spending.lowPortfolioSpendingReduction.percent()}")
            appendLine("- Pre-tax balance: ${scenario.accounts.pretax.money()}")
            appendLine("- Roth balance: ${scenario.accounts.roth.money()}")
            appendLine("- Taxable balance: ${scenario.accounts.taxable.money()}")
            appendLine("- Cash reserve: ${scenario.accounts.cash.money()}")
            val primaryBirthYear = SocialSecurity.primaryBirthYear(scenario.household.currentAge)
            appendLine("- Primary Social Security at FRA: ${scenario.socialSecurity.annualBenefitAt67.money()}")
            appendLine("- Primary Social Security FRA: ${SocialSecurity.fullRetirementAgeText(primaryBirthYear)}")
            appendLine("- Primary Social Security claim age: ${scenario.socialSecurity.claimAge}")
            if (scenario.household.filingStatus == com.retirementreadinesslab.model.FilingStatus.Married) {
                val spouseBirthYear = SocialSecurity.primaryBirthYear(scenario.household.spouseCurrentAge)
                appendLine("- Spouse Social Security FRA: ${SocialSecurity.fullRetirementAgeText(spouseBirthYear)}")
                appendLine("- Spouse survivor FRA: ${SocialSecurity.survivorFullRetirementAgeText(spouseBirthYear)}")
                appendLine("- Spouse benefit claim age: ${scenario.socialSecurity.spouseClaimAge}")
            }
            appendLine("- Social Security model: ${SocialSecurity.MODEL_VERSION}; benefits indexed with general inflation")
            appendLine("- Annual guaranteed income: ${scenario.guaranteedIncome.annualIncome.money()}")
            appendLine("- Guaranteed income start age: ${scenario.guaranteedIncome.startAge}")
            appendLine("- Guaranteed income annual increase: ${scenario.guaranteedIncome.annualIncrease.percent()}")
            appendLine("- Guaranteed income survivor benefit: ${scenario.guaranteedIncome.survivorPercent.percent()}")
            appendLine("- Pre-Medicare monthly premium: ${scenario.healthcare.preMedicareMonthlyPremium.money()}")
            appendLine("- Healthcare inflation average: ${scenario.healthcare.healthcareInflationMean.percent()}")
            appendLine("- Healthcare inflation Std Dev: ${scenario.healthcare.healthcareInflationStdDev.precisePercent()}")
            appendLine("- Medicare premiums: ${if (scenario.healthcare.includeMedicarePremiums) "Included" else "Excluded"}")
            appendLine("- Medicare premium model: ${MedicarePremiums.PREMIUM_TABLE_VERSION}")
            appendLine("- Mortgage payment: ${scenario.mortgage.monthlyPayment.money()}")
            appendLine("- Mortgage time left: ${scenario.mortgage.yearsLeft} years, ${scenario.mortgage.monthsLeft} months")
            appendLine("- Current mortgage balance: ${scenario.mortgage.currentBalance.money()}")
            appendLine("- Current home value: ${scenario.home.currentValue.money()}")
            appendLine("- Monthly rent: ${scenario.rent.monthlyRent.money()}")
            appendLine("- Senior apartment rent after home sale: ${SENIOR_APARTMENT_MONTHLY_RENT_2026.money()} in 2026 dollars, inflated with general inflation")
            appendLine("- Current portfolio returns average: ${scenario.market.preRetirementMeanReturn.precisePercent()}")
            appendLine("- Current portfolio returns Std Dev: ${scenario.market.preRetirementStdDev.precisePercent()}")
            appendLine("- Post-retirement stock returns average: ${scenario.market.stockMeanReturn.precisePercent()}")
            appendLine("- Post-retirement stock returns Std Dev: ${scenario.market.stockStdDev.precisePercent()}")
            appendLine("- Post-retirement stock returns basis: Default 13.3% ± 16.2% values based on the last 50 years of the S&P 500")
            appendLine("- Post-retirement bond returns average: ${scenario.market.bondMeanReturn.percent()}")
            appendLine("- Post-retirement bond returns Std Dev: ${scenario.market.bondStdDev.percent()}")
            appendLine("- Post-retirement investment ratios:")
            PostRetirementAllocationTier.entries.forEach { tier ->
                val stockAllocation = scenario.postRetirementAllocation.stockAllocationFor(tier)
                appendLine("  - ${tier.label}: ${stockAllocation.percent()} stocks / ${(1.0 - stockAllocation).percent()} bonds")
            }
            appendLine("- Roth conversions: ${if (scenario.rothConversion.enabled) "Enabled" else "Disabled"}")
            appendLine("- Roth conversion bracket cap: ${scenario.rothConversion.marginalRateCap.percent()}")
            appendLine("- 10% early withdrawal tax: ${if (scenario.withdrawalStrategy.applyEarlyWithdrawalPenalty) "Included before age 59 1/2" else "Excluded"}")
            appendLine("- Rule of 55 assumption: ${if (scenario.withdrawalStrategy.ruleOf55Eligible) "Eligible employer-plan withdrawals" else "Not applied"}")
            appendLine("- 72(t) / SEPP assumption: ${if (scenario.withdrawalStrategy.seppEligible) "Fixed amortization schedule from pre-tax balance" else "Not applied"}")
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

    private fun SpendingPathModel.detail(): String {
        return when (this) {
            SpendingPathModel.Flat -> "Annual base spending keeps pace with general inflation."
            SpendingPathModel.EmpiricalAgeDecline ->
                "Annual base spending uses Hurd and Rohwedder 2023 HRS/CAMS real spending declines after age 65, capped at age 85."
        }
    }

    private fun Double.percent(): String {
        return "${"%.0f".format(Locale.US, this * 100.0)}%"
    }

    private fun Double.precisePercent(): String {
        val percentText = "%.1f".format(Locale.US, this * 100.0)
            .trimEnd('0')
            .trimEnd('.')
        return "$percentText%"
    }

    private fun com.retirementreadinesslab.model.RiskLevel.label(): String {
        return when (this) {
            com.retirementreadinesslab.model.RiskLevel.Healthy -> "Healthy"
            com.retirementreadinesslab.model.RiskLevel.Watch -> "Watch"
            com.retirementreadinesslab.model.RiskLevel.AtRisk -> "At risk"
        }
    }

    private fun formatDate(epochMillis: Long): String {
        return SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date(epochMillis))
    }
}
