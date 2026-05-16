package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.sampleBaseScenario
import kotlin.system.measureNanoTime
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Ignore
import org.junit.Test

class RetirementSimulatorPerformanceTest {
    @Test
    fun thousandSimulationProfileCompletesQuickly() {
        val scenario = sampleBaseScenario().copy(numberOfSimulations = 1_000)
        var result = RetirementSimulator.run(scenario.copy(numberOfSimulations = 25))

        val elapsedMillis = measureNanoTime {
            result = RetirementSimulator.run(scenario)
        } / 1_000_000L

        println("RetirementSimulator performance: 1,000 simulations completed in ${elapsedMillis}ms")
        assertEquals(1_000, result.provenance.simulationCount)
        assertEquals(
            scenario.household.targetEndAge - scenario.household.retirementAge + 1,
            result.balanceBands.size
        )
        assertTrue(result.notFailedByAge.size <= result.balanceBands.size)
        assertTrue(result.successProbability in 0.0..1.0)
        assertTrue(
            "1,000 simulations took ${elapsedMillis}ms",
            elapsedMillis < maxAutomatedProfileMillis()
        )
    }

    @Ignore("Manual release-device profile. Run directly from Android Studio or remove @Ignore locally.")
    @Test
    fun tenThousandSimulationManualProfileCompletesWithoutChangingResultContract() {
        val scenario = sampleBaseScenario().copy(numberOfSimulations = 10_000)
        lateinit var result: com.retirementreadinesslab.model.SimulationResult

        val elapsedMillis = measureNanoTime {
            result = RetirementSimulator.run(scenario)
        } / 1_000_000L

        println("RetirementSimulator performance: 10,000 simulations completed in ${elapsedMillis}ms")
        assertEquals(10_000, result.provenance.simulationCount)
        assertTrue(result.successProbability in 0.0..1.0)
        assertTrue(result.balanceBands.isNotEmpty())
        assertTrue(result.notFailedByAge.isNotEmpty())
    }

    private fun maxAutomatedProfileMillis(): Long {
        return System.getProperty("retirementLab.maxPerfMillis")?.toLongOrNull() ?: 15_000L
    }
}
