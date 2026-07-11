package com.retirementreadinesslab.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.sampleScenarios
import com.retirementreadinesslab.model.validate
import kotlinx.coroutines.flow.first

private val Context.scenarioDataStore by preferencesDataStore(name = "retirement_scenarios")

class ScenarioRepository(private val context: Context) {
    private val scenariosKey = stringPreferencesKey("scenarios_json")
    private val selectedScenarioKey = stringPreferencesKey("selected_scenario_id")
    private val firstLaunchCompleteKey = booleanPreferencesKey("first_launch_complete")
    private val proUnlockedKey = booleanPreferencesKey("pro_unlocked")

    suspend fun loadState(): ScenarioStoreState {
        val preferences = context.scenarioDataStore.data.first()
        val rawScenarios = preferences[scenariosKey]
        val scenarios = rawScenarios
            ?.let { runCatching { ScenarioJson.decodeScenarios(it) }.getOrNull() }
            ?.takeIf { loaded ->
                loaded.isNotEmpty() &&
                    loaded.map { it.id }.distinct().size == loaded.size &&
                    loaded.all { it.validate().isEmpty() }
            }
            ?: sampleScenarios()
        val selectedId = preferences[selectedScenarioKey]
            ?.takeIf { id -> scenarios.any { it.id == id } }
            ?: scenarios.first().id
        val hasCompletedFirstLaunch = preferences[firstLaunchCompleteKey] ?: (rawScenarios != null)
        return ScenarioStoreState(
            scenarios = scenarios,
            selectedScenarioId = selectedId,
            hasCompletedFirstLaunch = hasCompletedFirstLaunch,
            isProUnlocked = preferences[proUnlockedKey] ?: false
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
            val proUnlocked = preferences[proUnlockedKey] ?: false
            preferences.clear()
            if (proUnlocked) {
                preferences[proUnlockedKey] = true
            }
        }
    }

    suspend fun setFirstLaunchComplete(value: Boolean) {
        context.scenarioDataStore.edit { preferences ->
            preferences[firstLaunchCompleteKey] = value
        }
    }

    suspend fun setProUnlocked(value: Boolean) {
        context.scenarioDataStore.edit { preferences ->
            preferences[proUnlockedKey] = value
        }
    }
}

data class ScenarioStoreState(
    val scenarios: List<RetirementScenario>,
    val selectedScenarioId: String,
    val hasCompletedFirstLaunch: Boolean,
    val isProUnlocked: Boolean
)
