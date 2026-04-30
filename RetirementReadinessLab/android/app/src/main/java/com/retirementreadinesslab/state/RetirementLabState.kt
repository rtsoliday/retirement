package com.retirementreadinesslab.state

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import com.retirementreadinesslab.data.ScenarioJson
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SimulationResult
import com.retirementreadinesslab.model.sampleScenarios
import com.retirementreadinesslab.data.ScenarioRepository
import com.retirementreadinesslab.model.validate
import com.retirementreadinesslab.simulation.RetirementDecisionEstimate
import com.retirementreadinesslab.simulation.RetirementSimulator
import com.retirementreadinesslab.simulation.ScenarioLabAnalysis
import com.retirementreadinesslab.simulation.ScenarioLabAnalyzer
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class RetirementLabState(
    private val repository: ScenarioRepository,
    private val scope: CoroutineScope,
    initialScenarios: List<RetirementScenario>
) {
    val scenarios = mutableStateListOf<RetirementScenario>().apply {
        addAll(initialScenarios)
    }
    private val results = mutableStateMapOf<String, SimulationResult>()
    private val labAnalyses = mutableStateMapOf<String, ScenarioLabAnalysis>()
    private var labAnalysisRunId = 0L

    var selectedScenarioId by mutableStateOf(scenarios.first().id)
        private set

    var isLoading by mutableStateOf(true)
        private set

    var isRunning by mutableStateOf(false)
        private set

    var isAnalyzingLab by mutableStateOf(false)
        private set

    var storageMessage by mutableStateOf<String?>(null)
        private set

    var lastRunMessage by mutableStateOf<String?>(null)
        private set

    var hasCompletedFirstLaunch by mutableStateOf(false)
        private set

    val selectedScenario: RetirementScenario
        get() = scenarios.firstOrNull { it.id == selectedScenarioId } ?: scenarios.first()

    val selectedResult: SimulationResult?
        get() = results[selectedScenarioId]

    val selectedLabAnalysis: ScenarioLabAnalysis?
        get() = labAnalyses[selectedScenarioId]

    val selectedDecisionEstimate: RetirementDecisionEstimate?
        get() = selectedLabAnalysis?.decisionEstimate

    fun resultFor(id: String): SimulationResult? = results[id]

    fun allResults(): Map<String, SimulationResult> = results.toMap()

    suspend fun loadPersistedState() {
        isLoading = true
        storageMessage = null
        val loaded = runCatching { repository.loadState() }
        loaded.onSuccess { stored ->
            scenarios.clear()
            scenarios.addAll(stored.scenarios)
            selectedScenarioId = stored.selectedScenarioId
            hasCompletedFirstLaunch = stored.hasCompletedFirstLaunch
            results.clear()
            labAnalyses.clear()
            runScenarioAsync(selectedScenarioId)
            runLabAnalysisAsync(selectedScenarioId)
        }.onFailure {
            storageMessage = "Saved scenarios could not be loaded. Sample scenarios are shown."
            runScenarioAsync(selectedScenarioId)
            runLabAnalysisAsync(selectedScenarioId)
        }
        isLoading = false
    }

    fun completeFirstLaunch() {
        hasCompletedFirstLaunch = true
        scope.launch {
            runCatching {
                repository.setFirstLaunchComplete(true)
            }.onFailure {
                storageMessage = "Welcome preference could not be saved locally."
            }
        }
    }

    fun selectScenario(id: String) {
        selectedScenarioId = id
        if (results[id] == null) {
            runScenarioAsync(id)
        }
        if (labAnalyses[id] == null) {
            runLabAnalysisAsync(id)
        }
        persist()
    }

    fun updateSelected(transform: (RetirementScenario) -> RetirementScenario) {
        val index = scenarios.indexOfFirst { it.id == selectedScenarioId }
        if (index >= 0) {
            val updated = transform(scenarios[index])
            scenarios[index] = updated
            results.remove(updated.id)
            labAnalyses.remove(updated.id)
            lastRunMessage = "Changes applied. Running stress test for ${updated.name}..."
            runScenarioAsync(updated.id)
            runLabAnalysisAsync(updated.id)
            persist()
        }
    }

    fun duplicateSelected() {
        val source = selectedScenario
        val copy = source.copy(
            id = "scenario-${System.currentTimeMillis()}",
            name = "${source.name} copy",
            seed = source.seed + scenarios.size + 1
        )
        scenarios += copy
        selectedScenarioId = copy.id
        runScenarioAsync(copy.id)
        runLabAnalysisAsync(copy.id)
        persist()
    }

    fun deleteSelected() {
        if (scenarios.size <= 1) return

        val index = scenarios.indexOfFirst { it.id == selectedScenarioId }
        if (index < 0) return

        val removed = scenarios.removeAt(index)
        results.remove(removed.id)
        labAnalyses.remove(removed.id)
        selectedScenarioId = scenarios.getOrNull(index)?.id ?: scenarios.last().id
        if (results[selectedScenarioId] == null) {
            runScenarioAsync(selectedScenarioId)
        }
        if (labAnalyses[selectedScenarioId] == null) {
            runLabAnalysisAsync(selectedScenarioId)
        }
        persist()
    }

    fun restoreSamplePlans() {
        scenarios.clear()
        scenarios.addAll(sampleScenarios())
        selectedScenarioId = scenarios.first().id
        results.clear()
        labAnalyses.clear()
        runScenarioAsync(selectedScenarioId)
        runLabAnalysisAsync(selectedScenarioId)
        persist()
    }

    fun deleteLocalData() {
        scenarios.clear()
        scenarios.addAll(sampleScenarios())
        selectedScenarioId = scenarios.first().id
        results.clear()
        labAnalyses.clear()
        storageMessage = "Local saved data deleted. Sample scenarios are loaded."
        runScenarioAsync(selectedScenarioId)
        runLabAnalysisAsync(selectedScenarioId)
        scope.launch {
            runCatching {
                repository.clearState()
            }.onFailure {
                storageMessage = "Local saved data could not be deleted."
            }
        }
    }

    fun importScenarioBackup(rawBackup: String): String? {
        val imported = runCatching {
            ScenarioJson.decodeScenarios(rawBackup.trim())
        }.getOrElse {
            return "Backup format could not be read. Paste the full JSON backup text."
        }

        if (imported.isEmpty()) {
            return "Backup did not contain any scenarios."
        }

        val invalidScenario = imported.firstOrNull { it.validate().isNotEmpty() }
        if (invalidScenario != null) {
            return "Backup contains an invalid scenario: ${invalidScenario.name}."
        }

        scenarios.clear()
        scenarios.addAll(imported)
        selectedScenarioId = scenarios.first().id
        results.clear()
        labAnalyses.clear()
        runScenarioAsync(selectedScenarioId)
        runLabAnalysisAsync(selectedScenarioId)
        storageMessage = "Imported ${scenarios.size} scenario${if (scenarios.size == 1) "" else "s"}."
        persist()
        return null
    }

    fun runSelectedScenario() {
        runScenarioAsync(selectedScenarioId)
    }

    fun runAllScenarios() {
        isRunning = true
        scenarios.forEach { scenario ->
            runScenario(scenario.id)
        }
        isRunning = false
        lastRunMessage = "Compared ${scenarios.size} scenario${if (scenarios.size == 1) "" else "s"}."
    }

    private fun runScenario(id: String) {
        val scenario = scenarios.firstOrNull { it.id == id } ?: return
        val result = RetirementSimulator.run(scenario)
        results[id] = result
        lastRunMessage = buildRunMessage(scenario, result)
    }

    private fun runScenarioAsync(id: String) {
        val scenario = scenarios.firstOrNull { it.id == id } ?: return
        isRunning = true
        lastRunMessage = "Running stress test for ${scenario.name}..."
        scope.launch {
            val run = runCatching {
                withContext(Dispatchers.Default) {
                    RetirementSimulator.run(scenario)
                }
            }
            run.onSuccess { result ->
                results[id] = result
                lastRunMessage = buildRunMessage(scenario, result)
            }.onFailure {
                storageMessage = "Stress test failed: ${it.message ?: "unknown error"}"
            }
            isRunning = false
        }
    }

    private fun runLabAnalysisAsync(id: String) {
        val scenario = scenarios.firstOrNull { it.id == id } ?: return
        val requestId = ++labAnalysisRunId
        isAnalyzingLab = id == selectedScenarioId
        scope.launch {
            val run = runCatching {
                withContext(Dispatchers.Default) {
                    ScenarioLabAnalyzer.analyze(scenario)
                }
            }
            run.onSuccess { analysis ->
                val stillCurrent = scenarios.firstOrNull { it.id == id } == scenario
                if (stillCurrent && requestId == labAnalysisRunId) {
                    labAnalyses[id] = analysis
                }
            }.onFailure {
                if (requestId == labAnalysisRunId) {
                    storageMessage = "Lab estimates failed: ${it.message ?: "unknown error"}"
                }
            }
            if (requestId == labAnalysisRunId) {
                isAnalyzingLab = false
            }
        }
    }

    private fun buildRunMessage(scenario: RetirementScenario, result: SimulationResult): String {
        val pct = String.format(Locale.US, "%.0f%%", result.successProbability * 100.0)
        return "Stress test complete: $pct readiness for ${scenario.name} at retirement age ${scenario.household.retirementAge}."
    }

    private fun persist() {
        val snapshot = scenarios.toList()
        val selectedId = selectedScenarioId
        scope.launch {
            runCatching {
                repository.saveState(snapshot, selectedId)
            }.onFailure {
                storageMessage = "Changes could not be saved locally."
            }
        }
    }
}

@Composable
fun rememberRetirementLabState(): RetirementLabState {
    val context = LocalContext.current.applicationContext
    val scope = rememberCoroutineScope()
    val state = remember(context) {
        RetirementLabState(
            repository = ScenarioRepository(context),
            scope = scope,
            initialScenarios = sampleScenarios()
        )
    }
    LaunchedEffect(state) {
        state.loadPersistedState()
    }
    return state
}
