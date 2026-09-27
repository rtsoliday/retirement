package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.Gender
import org.junit.Assert.assertEquals
import org.junit.Test

class MortalityTablesTest {
    @Test
    fun maleTableUsesPythonCsvValues() {
        assertEquals(0.004895, MortalityTables.annualDeathProbability(Gender.Male, 51), 0.000001)
        assertEquals(0.009251, MortalityTables.annualDeathProbability(Gender.Male, 58), 0.000001)
        assertEquals(0.015623, MortalityTables.annualDeathProbability(Gender.Male, 65), 0.000001)
        assertEquals(0.091358, MortalityTables.annualDeathProbability(Gender.Male, 85), 0.000001)
        assertEquals(0.949709, MortalityTables.annualDeathProbability(Gender.Male, 119), 0.000001)
    }

    @Test
    fun femaleTableUsesPythonCsvValues() {
        assertEquals(0.003098, MortalityTables.annualDeathProbability(Gender.Female, 51), 0.000001)
        assertEquals(0.005866, MortalityTables.annualDeathProbability(Gender.Female, 58), 0.000001)
        assertEquals(0.009208, MortalityTables.annualDeathProbability(Gender.Female, 65), 0.000001)
        assertEquals(0.070833, MortalityTables.annualDeathProbability(Gender.Female, 85), 0.000001)
        assertEquals(0.949709, MortalityTables.annualDeathProbability(Gender.Female, 119), 0.000001)
    }

    @Test
    fun agesOutsideTableAreCertainDeath() {
        assertEquals(1.0, MortalityTables.annualDeathProbability(Gender.Male, -1), 0.000001)
        assertEquals(1.0, MortalityTables.annualDeathProbability(Gender.Female, 120), 0.000001)
    }
}
