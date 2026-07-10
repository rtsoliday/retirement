package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.LongTermCareAssumption
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

enum class LabSweepType {
    RetirementAge,
    SpendingFlexibility,
    SocialSecurityTiming
}

enum class LabComparisonType {
    RetireLater,
    SpendLess,
    ClaimLater,
    RothConversion,
    LongTermCare,
    HealthcareInflation,
    MarketDownturn,
    MortgagePayoff
}

object ScenarioLabAnalyzer {
    const val QUICK_LAB_SIMULATIONS = 300
    const val ALLOCATION_OPTIMIZATION_SIMULATIONS = 500
    private const val COMPARISON_SEED_OFFSET = 400L

    fun analyze(
        scenario: RetirementScenario,
        quickSimulations: Int = QUICK_LAB_SIMULATIONS,
        targetEstimateSimulations: Int = RetirementOptimizer.QUICK_SIMULATIONS
    ): ScenarioLabAnalysis {
        val boundedQuickSimulations = quickSimulations.coerceAtLeast(50)
        val baseResult = runQuick(
            scenario,
            boundedQuickSimulations,
            seedOffset = COMPARISON_SEED_OFFSET
        )
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
            val candidateBest = candidateAllocations
                .map { value -> score(bestScore.allocation.withStockAllocation(tier, value)) }
                .maxWith(compareBy<AllocationScore> { it.readiness }.thenBy { it.medianEndingBalance })
            if (allocationScoreComparator.compare(candidateBest, bestScore) > 0) {
                bestScore = candidateBest
            }
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
        val lastValidAge = minOf(70, base.household.targetEndAge - 1)
        val ages = listOf(baseAge - 2, baseAge, baseAge + 2, baseAge + 4)
            .filter { it in currentAge..lastValidAge }
            .distinct()

        val rows = ages.map { age ->
            val scenario = base.copy(
                household = base.household.copy(retirementAge = age)
            )
            val result = runQuick(scenario, quickSimulations, seedOffset = 100L)
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
        val rows = factors.map { factor ->
            val spending = base.spending.annualBaseSpending * factor
            val scenario = base.copy(
                spending = base.spending.copy(annualBaseSpending = spending)
            )
            val result = runQuick(scenario, quickSimulations, seedOffset = 200L)
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
        val rows = ages.map { age ->
            val scenario = base.copy(
                socialSecurity = base.socialSecurity.copy(claimAge = age)
            )
            val result = runQuick(scenario, quickSimulations, seedOffset = 300L)
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
        fun score(updated: RetirementScenario): Double {
            return runQuick(
                updated,
                quickSimulations,
                seedOffset = COMPARISON_SEED_OFFSET
            ).successProbability
        }

        val latestRetirementAge = minOf(70, base.household.targetEndAge - 1)
        val retireLaterAge = (base.household.retirementAge + 2)
            .coerceAtMost(latestRetirementAge)
            .coerceAtLeast(base.household.retirementAge)
        val retireLater = score(
            base.copy(
                household = base.household.copy(
                    retirementAge = retireLaterAge
                )
            )
        )
        val spendLess = score(
            base.copy(
                spending = base.spending.copy(annualBaseSpending = base.spending.annualBaseSpending * 0.95)
            )
        )
        val claimLater = score(
            base.copy(
                socialSecurity = base.socialSecurity.copy(claimAge = 70)
            )
        )
        val roth = score(
            base.copy(
                rothConversion = RothConversionStrategy(enabled = true, marginalRateCap = 0.22)
            )
        )
        val ltc = score(
            base.copy(
                longTermCare = LongTermCareAssumption(enabled = true)
            )
        )
        val healthcareInflation = score(
            base.copy(
                healthcare = base.healthcare.copy(
                    healthcareInflationMean = (base.healthcare.healthcareInflationMean + 0.025).coerceAtMost(0.20),
                    healthcareInflationStdDev = (base.healthcare.healthcareInflationStdDev + 0.01).coerceAtMost(0.30)
                )
            )
        )
        val marketDownturn = score(
            base.copy(
                market = base.market.copy(
                    preRetirementMeanReturn = base.market.preRetirementMeanReturn - 0.02,
                    preRetirementStdDev = (base.market.preRetirementStdDev + 0.05).coerceAtMost(0.60),
                    stockMeanReturn = base.market.stockMeanReturn - 0.025,
                    stockStdDev = (base.market.stockStdDev + 0.07).coerceAtMost(0.60),
                    bondMeanReturn = base.market.bondMeanReturn - 0.01,
                    bondStdDev = (base.market.bondStdDev + 0.02).coerceAtMost(0.40)
                )
            )
        )
        val mortgagePayoff = score(
            base.copy(
                mortgage = base.mortgage.copy(monthlyPayment = 0.0, yearsLeft = 0, monthsLeft = 0)
            )
        )

        return listOf(
            LabComparisonAnalysis(
                type = LabComparisonType.RetireLater,
                readinessDelta = retireLater - baseReadiness,
                takeaway = takeaway(retireLater - baseReadiness, "Delaying retirement helps this scenario.", "Retirement age is not the main pressure point.")
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.SpendLess,
                readinessDelta = spendLess - baseReadiness,
                takeaway = takeaway(spendLess - baseReadiness, "Spending flexibility is powerful here.", "A small spending change does not solve the main risk.")
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.ClaimLater,
                readinessDelta = claimLater - baseReadiness,
                takeaway = takeaway(claimLater - baseReadiness, "Delayed claiming improves modeled durability.", "Claim age is secondary in this scenario.")
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.RothConversion,
                readinessDelta = roth - baseReadiness,
                takeaway = takeaway(roth - baseReadiness, "Tax location matters for this plan.", "Roth conversions have limited modeled impact so far.")
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.LongTermCare,
                readinessDelta = ltc - baseReadiness,
                takeaway = if (ltc < baseReadiness - 0.05) {
                    "Long-term care is a material downside risk."
                } else {
                    "The plan is not highly sensitive to this LTC assumption."
                }
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.HealthcareInflation,
                readinessDelta = healthcareInflation - baseReadiness,
                takeaway = if (healthcareInflation < baseReadiness - 0.05) {
                    "Healthcare inflation is a material downside risk."
                } else {
                    "Higher healthcare inflation does not materially change this quick estimate."
                }
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.MarketDownturn,
                readinessDelta = marketDownturn - baseReadiness,
                takeaway = if (marketDownturn < baseReadiness - 0.05) {
                    "Lower returns and wider swings materially weaken this plan."
                } else {
                    "The plan is not highly sensitive to this market stress in the quick estimate."
                }
            ),
            LabComparisonAnalysis(
                type = LabComparisonType.MortgagePayoff,
                readinessDelta = mortgagePayoff - baseReadiness,
                takeaway = if (base.mortgage.monthlyPayment <= 0.0 || base.mortgage.totalMonthsLeft <= 0) {
                    "No active mortgage payment is modeled in this scenario."
                } else if (mortgagePayoff > baseReadiness + 0.03) {
                    "Removing the mortgage payment materially improves modeled durability."
                } else {
                    "Mortgage payoff is not the strongest lever in this quick estimate."
                }
            )
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

    private val allocationScoreComparator =
        compareBy<AllocationScore> { it.readiness }.thenBy { it.medianEndingBalance }
}
