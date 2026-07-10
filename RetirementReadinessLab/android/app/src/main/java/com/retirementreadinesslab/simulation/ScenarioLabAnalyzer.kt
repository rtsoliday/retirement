package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.MortgagePlan
import com.retirementreadinesslab.model.PostRetirementAllocationStrategy
import com.retirementreadinesslab.model.PostRetirementAllocationTier
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.RothConversionStrategy
import java.util.Locale
import kotlin.math.abs

data class ScenarioLabAnalysis(
    val scenarioId: String,
    val decisionEstimate: RetirementDecisionEstimate,
    val sweeps: List<LabSweepAnalysis>,
    val comparisons: List<LabComparisonAnalysis>
)

data class LabSweepAnalysis(
    val type: LabSweepType,
    val rows: List<LabSweepRowAnalysis>,
    val takeaway: String
)

data class LabSweepRowAnalysis(
    val label: String,
    val detail: String,
    val readiness: Double,
    val medianEndingBalance: Double,
    val failureAge: Int?,
    val isBase: Boolean
) {
    val failureLabel: String
        get() = failureAge?.let { "Fail $it" } ?: "No fail"
}

data class LabComparisonAnalysis(
    val type: LabComparisonType,
    val title: String,
    val subtitle: String,
    val readinessDelta: Double,
    val takeaway: String
)

data class PostRetirementAllocationOptimization(
    val scenarioId: String,
    val startingAllocation: PostRetirementAllocationStrategy,
    val recommendedAllocation: PostRetirementAllocationStrategy,
    val startingReadiness: Double,
    val recommendedReadiness: Double,
    val recommendedMedianEndingBalance: Double,
    val testedAllocations: Int,
    val simulationCount: Int
) {
    val readinessDelta: Double
        get() = recommendedReadiness - startingReadiness
}

private data class ComparisonScenario(
    val scenario: RetirementScenario,
    val title: String,
    val subtitle: String
)

enum class LabSweepType {
    RetirementAge,
    SpendingFlexibility,
    SocialSecurityTiming
}

enum class LabComparisonType {
    RothConversion,
    LongTermCare,
    HealthcareInflation,
    MarketDownturn,
    MortgagePayoff
}

object ScenarioLabAnalyzer {
    const val QUICK_LAB_SIMULATIONS = 300
    const val ALLOCATION_OPTIMIZATION_SIMULATIONS = 500

    fun analyze(
        scenario: RetirementScenario,
        quickSimulations: Int = QUICK_LAB_SIMULATIONS,
        targetEstimateSimulations: Int = RetirementOptimizer.QUICK_SIMULATIONS
    ): ScenarioLabAnalysis {
        val boundedQuickSimulations = quickSimulations.coerceAtLeast(50)
        val baseResult = runQuick(scenario, boundedQuickSimulations, seedOffset = 0)
        return ScenarioLabAnalysis(
            scenarioId = scenario.id,
            decisionEstimate = RetirementOptimizer.estimate(
                scenario = scenario,
                simulationCount = targetEstimateSimulations
            ),
            sweeps = listOf(
                retirementAgeSweep(scenario, boundedQuickSimulations),
                spendingSweep(scenario, boundedQuickSimulations),
                socialSecuritySweep(scenario, boundedQuickSimulations)
            ),
            comparisons = strategyComparisons(scenario, baseResult.successProbability, boundedQuickSimulations)
        )
    }

    fun optimizePostRetirementAllocation(
        scenario: RetirementScenario,
        quickSimulations: Int = ALLOCATION_OPTIMIZATION_SIMULATIONS
    ): PostRetirementAllocationOptimization {
        val boundedQuickSimulations = quickSimulations.coerceAtLeast(50)
        val candidateAllocations = (0..20).map { it / 20.0 }
        var testedAllocations = 0

        fun score(allocation: PostRetirementAllocationStrategy): AllocationScore {
            testedAllocations += 1
            val result = RetirementSimulator.run(
                scenario.copy(
                    postRetirementAllocation = allocation,
                    numberOfSimulations = boundedQuickSimulations,
                    seed = scenario.seed + 900
                )
            )
            return AllocationScore(
                allocation = allocation,
                readiness = result.successProbability,
                medianEndingBalance = result.medianEndingBalance
            )
        }

        val startingAllocation = scenario.postRetirementAllocation
        val startingScore = score(startingAllocation)
        var bestScore = startingScore

        PostRetirementAllocationTier.entries.forEach { tier ->
            bestScore = candidateAllocations
                .map { value -> score(bestScore.allocation.withStockAllocation(tier, value)) }
                .maxWith(compareBy<AllocationScore> { it.readiness }.thenBy { it.medianEndingBalance })
        }

        return PostRetirementAllocationOptimization(
            scenarioId = scenario.id,
            startingAllocation = startingAllocation,
            recommendedAllocation = bestScore.allocation,
            startingReadiness = startingScore.readiness,
            recommendedReadiness = bestScore.readiness,
            recommendedMedianEndingBalance = bestScore.medianEndingBalance,
            testedAllocations = testedAllocations,
            simulationCount = boundedQuickSimulations
        )
    }

    private fun retirementAgeSweep(base: RetirementScenario, quickSimulations: Int): LabSweepAnalysis {
        val currentAge = base.household.currentAge
        val baseAge = base.household.retirementAge
        val ages = listOf(baseAge - 2, baseAge, baseAge + 2, baseAge + 4)
            .filter { it in currentAge..70 }
            .distinct()

        val rows = ages.mapIndexed { index, age ->
            val scenario = base
                .withRetirementAgeForAnalysis(age)
                .copy(seed = base.seed + 100)
            val result = runQuick(scenario, quickSimulations, seedOffset = index.toLong())
            LabSweepRowAnalysis(
                label = "Age $age",
                detail = "${age - currentAge} years from now",
                readiness = result.successProbability,
                medianEndingBalance = result.medianEndingBalance,
                failureAge = result.medianFailureAge,
                isBase = age == baseAge
            )
        }
        return LabSweepAnalysis(
            type = LabSweepType.RetirementAge,
            rows = rows,
            takeaway = sweepTakeaway(rows)
        )
    }

    private fun spendingSweep(base: RetirementScenario, quickSimulations: Int): LabSweepAnalysis {
        val factors = listOf(0.90, 0.95, 1.00, 1.05, 1.10)
        val rows = factors.mapIndexed { index, factor ->
            val spending = base.spending.annualBaseSpending * factor
            val scenario = base.copy(
                spending = base.spending.copy(annualBaseSpending = spending),
                seed = base.seed + 200
            )
            val result = runQuick(scenario, quickSimulations, seedOffset = index.toLong())
            val pct = ((factor - 1.0) * 100.0)
            LabSweepRowAnalysis(
                label = if (factor == 1.0) "Current" else "${signedWholePercent(pct)} spending",
                detail = spending.shortCurrency(),
                readiness = result.successProbability,
                medianEndingBalance = result.medianEndingBalance,
                failureAge = result.medianFailureAge,
                isBase = factor == 1.0
            )
        }
        return LabSweepAnalysis(
            type = LabSweepType.SpendingFlexibility,
            rows = rows,
            takeaway = sweepTakeaway(rows)
        )
    }

    private fun socialSecuritySweep(base: RetirementScenario, quickSimulations: Int): LabSweepAnalysis {
        val ages = setOf(62, 67, 70, base.socialSecurity.claimAge).sorted()
        val rows = ages.mapIndexed { index, age ->
            val scenario = base.copy(
                socialSecurity = base.socialSecurity.copy(claimAge = age),
                seed = base.seed + 300
            )
            val result = runQuick(scenario, quickSimulations, seedOffset = index.toLong())
            LabSweepRowAnalysis(
                label = "Claim $age",
                detail = if (age < base.household.retirementAge) "Available at retirement" else "Bridge ${age - base.household.retirementAge} years",
                readiness = result.successProbability,
                medianEndingBalance = result.medianEndingBalance,
                failureAge = result.medianFailureAge,
                isBase = age == base.socialSecurity.claimAge
            )
        }
        return LabSweepAnalysis(
            type = LabSweepType.SocialSecurityTiming,
            rows = rows,
            takeaway = sweepTakeaway(rows)
        )
    }

    private fun strategyComparisons(
        base: RetirementScenario,
        baseReadiness: Double,
        quickSimulations: Int
    ): List<LabComparisonAnalysis> {
        fun score(updated: RetirementScenario, seedOffset: Long): Double {
            return runQuick(updated, quickSimulations, seedOffset).successProbability
        }

        val rothComparison = rothComparison(base)
        val longTermCareComparison = longTermCareComparison(base)
        val healthcareInflationComparison = healthcareInflationComparison(base)
        val marketComparison = marketComparison(base)
        val mortgageComparison = mortgageComparison(base)

        val roth = score(rothComparison.scenario, seedOffset = 40)
        val ltc = score(longTermCareComparison.scenario, seedOffset = 50)
        val healthcareInflation = score(healthcareInflationComparison.scenario, seedOffset = 60)
        val marketDownturn = score(marketComparison.scenario, seedOffset = 70)
        val mortgagePayoff = score(mortgageComparison.scenario, seedOffset = 80)

        return listOf(
            LabComparisonAnalysis(
                type = LabComparisonType.RothConversion,
                title = rothComparison.title,
                subtitle = rothComparison.subtitle,
                readinessDelta = roth - baseReadiness,
                takeaway = if (base.rothConversion.enabled) {
                    takeaway(roth - baseReadiness, "Disabling conversions improves this quick estimate.", "The current Roth conversion setting is not the main pressure point.")
                } else {
                    takeaway(roth - baseReadiness, "Tax location matters for this plan.", "Roth conversions have limited modeled impact so far.")
                }
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.LongTermCare,
                title = longTermCareComparison.title,
                subtitle = longTermCareComparison.subtitle,
                readinessDelta = ltc - baseReadiness,
                takeaway = if (!base.longTermCare.enabled && ltc < baseReadiness - 0.05) {
                    "Long-term care is a material downside risk."
                } else if (base.longTermCare.enabled && ltc > baseReadiness + 0.05) {
                    "The selected long-term care risk materially reduces modeled durability."
                } else {
                    "The plan is not highly sensitive to this LTC assumption."
                }
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.HealthcareInflation,
                title = healthcareInflationComparison.title,
                subtitle = healthcareInflationComparison.subtitle,
                readinessDelta = healthcareInflation - baseReadiness,
                takeaway = if (healthcareInflation < baseReadiness - 0.05) {
                    "Healthcare inflation is a material downside risk."
                } else if (healthcareInflation > baseReadiness + 0.05) {
                    "Lower healthcare inflation materially improves the quick estimate."
                } else {
                    "This healthcare inflation change does not materially change the quick estimate."
                }
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.MarketDownturn,
                title = marketComparison.title,
                subtitle = marketComparison.subtitle,
                readinessDelta = marketDownturn - baseReadiness,
                takeaway = if (marketDownturn < baseReadiness - 0.05) {
                    "Lower returns and wider swings materially weaken this plan."
                } else if (marketDownturn > baseReadiness + 0.05) {
                    "Less stressful returns materially improve this quick estimate."
                } else {
                    "The plan is not highly sensitive to this market change in the quick estimate."
                }
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.MortgagePayoff,
                title = mortgageComparison.title,
                subtitle = mortgageComparison.subtitle,
                readinessDelta = mortgagePayoff - baseReadiness,
                takeaway = if (base.mortgage.monthlyPayment > 0.0 && base.mortgage.totalMonthsLeft > 0 && mortgagePayoff > baseReadiness + 0.03) {
                    "Removing the mortgage payment materially improves modeled durability."
                } else if ((base.mortgage.monthlyPayment <= 0.0 || base.mortgage.totalMonthsLeft <= 0) && mortgagePayoff < baseReadiness - 0.03) {
                    "Adding a mortgage payment materially weakens modeled durability."
                } else {
                    "Mortgage cash flow is not the strongest lever in this quick estimate."
                }
            )
        )
    }

    private fun rothComparison(base: RetirementScenario): ComparisonScenario {
        val enabled = !base.rothConversion.enabled
        return ComparisonScenario(
            scenario = base.copy(
                rothConversion = RothConversionStrategy(enabled = enabled, marginalRateCap = 0.22),
                seed = base.seed + 40
            ),
            title = if (enabled) "Enable Roth conversions" else "Disable Roth conversions",
            subtitle = if (enabled) "Fill the 22% bracket after retirement" else "Compare without planned conversions"
        )
    }

    private fun longTermCareComparison(base: RetirementScenario): ComparisonScenario {
        val enabled = !base.longTermCare.enabled
        return ComparisonScenario(
            scenario = base.copy(
                longTermCare = base.longTermCare.copy(enabled = enabled),
                seed = base.seed + 50
            ),
            title = if (enabled) "Add long-term care risk" else "Remove long-term care risk",
            subtitle = if (enabled) "Late-life care shock" else "Compare without the care shock"
        )
    }

    private fun healthcareInflationComparison(base: RetirementScenario): ComparisonScenario {
        val stressHigher = base.healthcare.healthcareInflationMean < 0.20 ||
            base.healthcare.healthcareInflationStdDev < 0.30
        val updatedHealthcare = if (stressHigher) {
            base.healthcare.copy(
                healthcareInflationMean = (base.healthcare.healthcareInflationMean + 0.025).coerceAtMost(0.20),
                healthcareInflationStdDev = (base.healthcare.healthcareInflationStdDev + 0.01).coerceAtMost(0.30)
            )
        } else {
            base.healthcare.copy(
                healthcareInflationMean = (base.healthcare.healthcareInflationMean - 0.025).coerceAtLeast(0.0),
                healthcareInflationStdDev = (base.healthcare.healthcareInflationStdDev - 0.01).coerceAtLeast(0.0)
            )
        }
        return ComparisonScenario(
            scenario = base.copy(healthcare = updatedHealthcare, seed = base.seed + 60),
            title = if (stressHigher) "Stress healthcare inflation" else "Ease healthcare inflation",
            subtitle = if (stressHigher) "Higher medical cost growth" else "Lower medical cost growth"
        )
    }

    private fun marketComparison(base: RetirementScenario): ComparisonScenario {
        val stressLower = base.market.stockMeanReturn > -0.20 ||
            base.market.preRetirementMeanReturn > -0.20
        val updatedMarket = if (stressLower) {
            base.market.copy(
                preRetirementMeanReturn = (base.market.preRetirementMeanReturn - 0.02).coerceAtLeast(-0.20),
                preRetirementStdDev = (base.market.preRetirementStdDev + 0.05).coerceAtMost(0.60),
                stockMeanReturn = (base.market.stockMeanReturn - 0.025).coerceAtLeast(-0.20),
                stockStdDev = (base.market.stockStdDev + 0.07).coerceAtMost(0.60),
                bondMeanReturn = (base.market.bondMeanReturn - 0.01).coerceAtLeast(-0.20),
                bondStdDev = (base.market.bondStdDev + 0.02).coerceAtMost(0.40)
            )
        } else {
            base.market.copy(
                preRetirementMeanReturn = (base.market.preRetirementMeanReturn + 0.02).coerceAtMost(0.25),
                preRetirementStdDev = (base.market.preRetirementStdDev - 0.05).coerceAtLeast(0.0),
                stockMeanReturn = (base.market.stockMeanReturn + 0.025).coerceAtMost(0.25),
                stockStdDev = (base.market.stockStdDev - 0.07).coerceAtLeast(0.0),
                bondMeanReturn = (base.market.bondMeanReturn + 0.01).coerceAtMost(0.20),
                bondStdDev = (base.market.bondStdDev - 0.02).coerceAtLeast(0.0)
            )
        }
        return ComparisonScenario(
            scenario = base.copy(market = updatedMarket, seed = base.seed + 70),
            title = if (stressLower) "Stress lower returns" else "Ease return stress",
            subtitle = if (stressLower) "Lower expected returns, wider swings" else "Higher expected returns, narrower swings"
        )
    }

    private fun mortgageComparison(base: RetirementScenario): ComparisonScenario {
        val hasMortgage = base.mortgage.monthlyPayment > 0.0 && base.mortgage.totalMonthsLeft > 0
        val updatedMortgage = if (hasMortgage) {
            MortgagePlan(monthlyPayment = 0.0, yearsLeft = 0, monthsLeft = 0, currentBalance = 0.0)
        } else {
            MortgagePlan(monthlyPayment = 1_500.0, yearsLeft = 10, monthsLeft = 0, currentBalance = 150_000.0)
        }
        return ComparisonScenario(
            scenario = base.copy(mortgage = updatedMortgage, seed = base.seed + 80),
            title = if (hasMortgage) "Remove mortgage payment" else "Add mortgage payment",
            subtitle = if (hasMortgage) {
                "Tests retirement cash flow without mortgage payments"
            } else {
                "Tests a $1,500 monthly payment for 10 years"
            }
        )
    }

    private fun runQuick(
        scenario: RetirementScenario,
        quickSimulations: Int,
        seedOffset: Long
    ) = RetirementSimulator.run(
        scenario.copy(
            numberOfSimulations = quickSimulations,
            seed = scenario.seed + seedOffset
        )
    )

    private fun sweepTakeaway(rows: List<LabSweepRowAnalysis>): String {
        if (rows.isEmpty()) return "No valid sweep points for this scenario."
        val best = rows.maxBy { it.readiness }
        val base = rows.firstOrNull { it.isBase }
        val spread = rows.maxOf { it.readiness } - rows.minOf { it.readiness }
        if (spread < 0.02) {
            return "These options are close in the quick estimate. Use the main stress test before deciding."
        }
        return if (base != null && best.label == base.label) {
            "The current setting is the strongest quick estimate in this sweep."
        } else {
            "Best quick estimate: ${best.label} at ${best.readiness.wholePercent()} readiness."
        }
    }

    private fun takeaway(delta: Double, positive: String, neutral: String): String {
        return if (delta > 0.03) positive else neutral
    }

    private fun signedWholePercent(value: Double): String {
        val sign = if (value >= 0.0) "+" else ""
        return "$sign${String.format(Locale.US, "%.0f%%", value)}"
    }

    private fun Double.wholePercent(): String {
        return "${String.format(Locale.US, "%.0f", this * 100.0)}%"
    }

    private fun Double.shortCurrency(): String {
        val value = abs(this)
        val sign = if (this < 0) "-" else ""
        return when {
            value >= 1_000_000 -> "$sign\$${"%.1f".format(Locale.US, value / 1_000_000.0)}M"
            value >= 1_000 -> "$sign\$${"%.0f".format(Locale.US, value / 1_000.0)}k"
            else -> "$sign\$${"%.0f".format(Locale.US, value)}"
        }
    }

    private data class AllocationScore(
        val allocation: PostRetirementAllocationStrategy,
        val readiness: Double,
        val medianEndingBalance: Double
    )
}
