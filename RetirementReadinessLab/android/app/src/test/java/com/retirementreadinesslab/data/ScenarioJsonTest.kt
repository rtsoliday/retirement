package com.retirementreadinesslab.data

import com.retirementreadinesslab.model.DEFAULT_PROJECTION_END_AGE
import com.retirementreadinesslab.model.sampleBaseScenario
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScenarioJsonTest {
    @Test
    fun scenariosRoundTripThroughJsonBackup() {
        val scenario = sampleBaseScenario().copy(name = "Base, with comma")

        val raw = ScenarioJson.encodeScenarios(listOf(scenario))
        val decoded = ScenarioJson.decodeScenarios(raw)

        assertEquals(1, decoded.size)
        assertEquals("Base, with comma", decoded.first().name)
        assertEquals(scenario.household.retirementAge, decoded.first().household.retirementAge)
        assertEquals(scenario.accounts.pretax, decoded.first().accounts.pretax, 0.01)
        assertEquals(scenario.socialSecurity.claimAge, decoded.first().socialSecurity.claimAge)
        assertTrue(raw.startsWith("["))
    }

    @Test
    fun legacyTargetEndAgeImportsUseMortalityProjectionCap() {
        val scenario = sampleBaseScenario().copy(
            household = sampleBaseScenario().household.copy(targetEndAge = 95)
        )

        val decoded = ScenarioJson.decodeScenarios(ScenarioJson.encodeScenarios(listOf(scenario)))

        assertEquals(DEFAULT_PROJECTION_END_AGE, decoded.first().household.targetEndAge)
    }
}
