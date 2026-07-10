package com.retirementreadinesslab.state

import android.content.Context
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.retirementreadinesslab.data.ScenarioJson
import com.retirementreadinesslab.data.ScenarioRepository
import com.retirementreadinesslab.model.BudgetProfile
import com.retirementreadinesslab.model.PostRetirementAllocationStrategy
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SimulationResult
import com.retirementreadinesslab.model.sampleScenarios
import com.retirementreadinesslab.model.validate
import com.retirementreadinesslab.simulation.RetirementDecisionEstimate
import com.retirementreadinesslab.simulation.RetirementSimulator
import com.retirementreadinesslab.simulation.PostRetirementAllocationOptimization
import com.retirementreadinesslab.simulation.ScenarioLabAnalysis
import com.retirementreadinesslab.simulation.ScenarioLabAnalyzer
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
    private val allocationOptimizations = mutableStateMapOf<String, PostRetirementAllocationOptimization>()
    private var scenarioRunId = 0L
    private var labAnalysisRunId = 0L
    private var allocationOptimizationRunId = 0L

    var selectedScenarioId by mutableStateOf(scenarios.first().id)
        private set

    var isLoading by mutableStateOf(true)
        private set

    var isRunning by mutableStateOf(false)
        private set

    var isAnalyzingLab by mutableStateOf(false)
        private set

    var isOptimizingPostRetirementAllocation by mutableStateOf(false)
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

    val selectedPostRetirementAllocationOptimization: PostRetirementAllocationOptimization?
        get() = allocationOptimizations[selectedScenarioId]

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
            allocationOptimizations.clear()
            runScenarioAsync(selectedScenarioId)
            runLabAnalysisAsync(selectedScenarioId)
        }.onFailure {
            storageMessage = "Saved scenario data could not be loaded. Sample data is shown."
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

    fun updateSelected(transform: (RetirementScenario) -> RetirementScenario) {
        val index = scenarios.indexOfFirst { it.id == selectedScenarioId }
        if (index >= 0) {
            val updated = transform(scenarios[index])
            scenarios[index] = updated
            results.remove(updated.id)
            labAnalyses.remove(updated.id)
            lastRunMessage = "Changes applied. Running stress test..."
            runScenarioAsync(updated.id)
            runLabAnalysisAsync(updated.id)
            persist()
        }
    }

    fun updateSelectedPostRetirementAllocation(allocation: PostRetirementAllocationStrategy) {
        val index = scenarios.indexOfFirst { it.id == selectedScenarioId }
        if (index >= 0) {
            val current = scenarios[index]
            val updated = current.copy(postRetirementAllocation = allocation)
            scenarios[index] = updated
            results.remove(updated.id)
            allocationOptimizations.remove(updated.id)
            lastRunMessage = "Investment ratios saved. Run a stress test to update readiness."
            persist()
        }
    }

    fun optimizeSelectedPostRetirementAllocation(startingAllocation: PostRetirementAllocationStrategy) {
        val index = scenarios.indexOfFirst { it.id == selectedScenarioId }
        if (index < 0) return

        val startingScenario = scenarios[index].copy(postRetirementAllocation = startingAllocation)
        scenarios[index] = startingScenario
        results.remove(startingScenario.id)
        allocationOptimizations.remove(startingScenario.id)
        lastRunMessage = "Testing investment ratios..."
        persist()

        val requestId = ++allocationOptimizationRunId
        isOptimizingPostRetirementAllocation = true
        scope.launch {
            val run = runCatching {
                withContext(Dispatchers.Default) {
                    ScenarioLabAnalyzer.optimizePostRetirementAllocation(startingScenario)
                }
            }
            run.onSuccess { optimization ->
                val currentIndex = scenarios.indexOfFirst { it.id == startingScenario.id }
                val current = scenarios.getOrNull(currentIndex)
                val unchangedDuringRun = current?.postRetirementAllocation == startingAllocation
                if (current != null && unchangedDuringRun && requestId == allocationOptimizationRunId) {
                    val updated = current.copy(postRetirementAllocation = optimization.recommendedAllocation)
                    scenarios[currentIndex] = updated
                    results.remove(updated.id)
                    allocationOptimizations[updated.id] = optimization
                    lastRunMessage = buildAllocationOptimizationMessage(optimization)
                    persist()
                }
            }.onFailure {
                if (requestId == allocationOptimizationRunId) {
                    storageMessage = "Investment ratio test failed: ${it.message ?: "unknown error"}"
                }
            }
            if (requestId == allocationOptimizationRunId) {
                isOptimizingPostRetirementAllocation = false
            }
        }
    }

    fun saveSelectedBudget(budget: BudgetProfile) {
        val index = scenarios.indexOfFirst { it.id == selectedScenarioId }
        if (index >= 0) {
            val current = scenarios[index]
            scenarios[index] = current.copy(budget = budget)
            storageMessage = "Budget saved."
            persist()
        }
    }

    fun applyBudgetToAnnualBaseSpending(budget: BudgetProfile): String? {
        val estimate = budget.annualBaseSpendingEstimate
        if (estimate <= 0.0) {
            return "Enter at least one spending amount before applying the budget estimate."
        }
        updateSelected { scenario ->
            scenario.copy(
                budget = budget,
                spending = scenario.spending.copy(annualBaseSpending = estimate)
            )
        }
        return null
    }

    fun deleteLocalData() {
        scenarios.clear()
        scenarios.addAll(sampleScenarios())
        selectedScenarioId = scenarios.first().id
        results.clear()
        labAnalyses.clear()
        allocationOptimizations.clear()
        storageMessage = "Local saved data deleted. Sample data is loaded."
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
        allocationOptimizations.clear()
        runScenarioAsync(selectedScenarioId)
        runLabAnalysisAsync(selectedScenarioId)
        storageMessage = "Imported ${scenarios.size} scenario${if (scenarios.size == 1) "" else "s"}."
        persist()
        return null
    }

    fun runSelectedScenario() {
        runScenarioAsync(selectedScenarioId)
    }

    private fun runScenarioAsync(id: String) {
        val scenario = scenarios.firstOrNull { it.id == id } ?: return
        val requestId = ++scenarioRunId
        isRunning = true
        lastRunMessage = "Running stress test..."
        scope.launch {
            val run = runCatching {
                withContext(Dispatchers.Default) {
                    RetirementSimulator.run(scenario)
                }
            }
            run.onSuccess { result ->
                val stillCurrent = scenarios.firstOrNull { it.id == id } == scenario
                if (stillCurrent && requestId == scenarioRunId) {
                    results[id] = result
                    lastRunMessage = buildRunMessage(scenario, result)
                }
            }.onFailure {
                if (requestId == scenarioRunId) {
                    storageMessage = "Stress test failed: ${it.message ?: "unknown error"}"
                }
            }
            if (requestId == scenarioRunId) {
                isRunning = false
            }
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
        return "Stress test complete: $pct readiness at retirement age ${scenario.household.retirementAge}."
    }

    private fun buildAllocationOptimizationMessage(optimization: PostRetirementAllocationOptimization): String {
        val best = String.format(Locale.US, "%.0f%%", optimization.recommendedReadiness * 100.0)
        val delta = String.format(Locale.US, "%+.0f", optimization.readinessDelta * 100.0)
        return "Investment ratio test complete: best quick estimate $best readiness ($delta pts)."
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

private class RetirementLabViewModel(applicationContext: Context) : ViewModel() {
    val state = RetirementLabState(
        repository = ScenarioRepository(applicationContext),
        scope = viewModelScope,
        initialScenarios = sampleScenarios()
    )

    private var hasLoadedPersistedState = false

    fun loadPersistedStateOnce() {
        if (hasLoadedPersistedState) return
        hasLoadedPersistedState = true
        viewModelScope.launch {
            state.loadPersistedState()
        }
    }
}

@Composable
fun rememberRetirementLabState(): RetirementLabState {
    val context = LocalContext.current.applicationContext
    val factory = remember(context) {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return RetirementLabViewModel(context) as T
            }
        }
    }
    val labViewModel: RetirementLabViewModel = viewModel(factory = factory)

    LaunchedEffect(labViewModel) {
        labViewModel.loadPersistedStateOnce()
    }
    return labViewModel.state
}
