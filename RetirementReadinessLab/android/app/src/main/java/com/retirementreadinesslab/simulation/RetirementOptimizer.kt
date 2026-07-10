package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.RetirementScenario
import kotlin.math.floor

data class RetirementDecisionEstimate(
    val targetReadiness: Double,
    val simulationCount: Int,
    val earliestRetirementAge: Int?,
    val earliestRetirementReadiness: Double?,
    val safeAnnualSpending: Double?,
    val safeSpendingReadiness: Double?
)

object RetirementOptimizer {
    const val DEFAULT_TARGET_READINESS = 0.80
    const val QUICK_SIMULATIONS = 180

    fun estimate(
        scenario: RetirementScenario,
        targetReadiness: Double = DEFAULT_TARGET_READINESS,
        simulationCount: Int = QUICK_SIMULATIONS,
        maxRetirementAge: Int = 70
    ): RetirementDecisionEstimate {
        val boundedSimulations = simulationCount.coerceAtLeast(50)
        val earliest = findEarliestRetirementAge(
            scenario = scenario,
            targetReadiness = targetReadiness,
            simulationCount = boundedSimulations,
            maxRetirementAge = maxRetirementAge
        )
        val safeSpending = findSafeSpending(
            scenario = scenario,
            targetReadiness = targetReadiness,
            simulationCount = boundedSimulations
        )

        return RetirementDecisionEstimate(
            targetReadiness = targetReadiness,
            simulationCount = boundedSimulations,
            earliestRetirementAge = earliest?.first,
            earliestRetirementReadiness = earliest?.second,
            safeAnnualSpending = safeSpending?.first,
            safeSpendingReadiness = safeSpending?.second
        )
    }

    private fun findEarliestRetirementAge(
        scenario: RetirementScenario,
        targetReadiness: Double,
        simulationCount: Int,
        maxRetirementAge: Int
    ): Pair<Int, Double>? {
        val firstAge = scenario.household.currentAge
        val lastAge = minOf(maxRetirementAge, scenario.household.targetEndAge - 1)
        if (firstAge > lastAge) return null

        for (age in firstAge..lastAge) {
            val result = runQuick(
                scenario = scenario.withRetirementAgeForAnalysis(age),
                simulationCount = simulationCount,
                seedOffset = 10_000L + age
            )
            if (result.successProbability >= targetReadiness) {
                return age to result.successProbability
            }
        }
        return null
    }

    private fun findSafeSpending(
        scenario: RetirementScenario,
        targetReadiness: Double,
        simulationCount: Int
    ): Pair<Double, Double>? {
        fun readinessFor(spending: Double): Double {
            return runQuick(
                scenario = scenario.copy(
                    spending = scenario.spending.copy(annualBaseSpending = spending)
                ),
                simulationCount = simulationCount,
                seedOffset = 20_000L
            ).successProbability
        }

        val zeroReadiness = readinessFor(0.0)
        if (zeroReadiness < targetReadiness) return null

        val maxSearchSpending = maxOf(
            scenario.spending.annualBaseSpending * 3.0,
            250_000.0
        )
        var low = 0.0
        var high = maxOf(scenario.spending.annualBaseSpending * 1.5, 40_000.0)
            .coerceAtMost(maxSearchSpending)
        var highReadiness = readinessFor(high)

        while (highReadiness >= targetReadiness && high < maxSearchSpending) {
            low = high
            high = (high * 1.35).coerceAtMost(maxSearchSpending)
            highReadiness = readinessFor(high)
        }

        if (highReadiness >= targetReadiness) {
            return roundedSafeSpending(
                safeSpending = high,
                targetReadiness = targetReadiness,
                increment = 500.0,
                readinessFor = ::readinessFor
            )
        }

        repeat(11) { index ->
            val mid = (low + high) / 2.0
            val readiness = readinessFor(mid)
            if (readiness >= targetReadiness) {
                low = mid
            } else {
                high = mid
            }
        }

        return roundedSafeSpending(
            safeSpending = low,
            targetReadiness = targetReadiness,
            increment = 500.0,
            readinessFor = ::readinessFor
        )
    }

    private fun runQuick(
        scenario: RetirementScenario,
        simulationCount: Int,
        seedOffset: Long
    ) = RetirementSimulator.run(
        scenario.copy(
            numberOfSimulations = simulationCount,
            seed = scenario.seed + seedOffset
        )
    )

    private fun roundedSafeSpending(
        safeSpending: Double,
        targetReadiness: Double,
        increment: Double,
        readinessFor: (Double) -> Double
    ): Pair<Double, Double> {
        var candidate = roundDownToIncrement(safeSpending, increment)
        while (candidate > 0.0) {
            val readiness = readinessFor(candidate)
            if (readiness >= targetReadiness) {
                return candidate to readiness
            }
            candidate = (candidate - increment).coerceAtLeast(0.0)
        }
        return 0.0 to readinessFor(0.0)
    }

    private fun roundDownToIncrement(value: Double, increment: Double): Double {
        return floor(value / increment) * increment
    }
}
