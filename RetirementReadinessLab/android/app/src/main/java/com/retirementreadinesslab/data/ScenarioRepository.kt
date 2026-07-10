package com.retirementreadinesslab.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.sampleScenarios
import kotlinx.coroutines.flow.first

private val Context.scenarioDataStore by preferencesDataStore(name = "retirement_scenarios")

class ScenarioRepository(private val context: Context) {
    private val scenariosKey = stringPreferencesKey("scenarios_json")
    private val selectedScenarioKey = stringPreferencesKey("selected_scenario_id")
    private val firstLaunchCompleteKey = booleanPreferencesKey("first_launch_complete")

    suspend fun loadState(): ScenarioStoreState {
        val preferences = context.scenarioDataStore.data.first()
        val rawScenarios = preferences[scenariosKey]
        val scenarios = rawScenarios
            ?.let { runCatching { ScenarioJson.decodeScenarios(it) }.getOrNull() }
            ?.takeIf { it.isNotEmpty() }
            ?: sampleScenarios()
        val selectedId = preferences[selectedScenarioKey]
            ?.takeIf { id -> scenarios.any { it.id == id } }
            ?: scenarios.first().id
        val hasCompletedFirstLaunch = preferences[firstLaunchCompleteKey] ?: (rawScenarios != null)
        return ScenarioStoreState(
            scenarios = scenarios,
            selectedScenarioId = selectedId,
            hasCompletedFirstLaunch = hasCompletedFirstLaunch
        )
    }

    suspend fun saveState(scenarios: List<RetirementScenario>, selectedScenarioId: String) {
        context.scenarioDataStore.edit { preferences ->
            preferences[scenariosKey] = ScenarioJson.encodeScenarios(scenarios)
            preferences[selectedScenarioKey] = selectedScenarioId
        }
    }

    suspend fun clearState() {
        context.scenarioDataStore.edit { preferences ->
            preferences.clear()
        }
    }

    suspend fun setFirstLaunchComplete(value: Boolean) {
        context.scenarioDataStore.edit { preferences ->
            preferences[firstLaunchCompleteKey] = value
        }
    }
}

data class ScenarioStoreState(
    val scenarios: List<RetirementScenario>,
    val selectedScenarioId: String,
    val hasCompletedFirstLaunch: Boolean
)
