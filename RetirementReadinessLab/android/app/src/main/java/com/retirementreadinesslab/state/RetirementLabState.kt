package com.retirementreadinesslab.state

import android.app.Activity
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
import com.retirementreadinesslab.entitlements.DefaultEntitlementProvider
import com.retirementreadinesslab.entitlements.ProEntitlement
import com.retirementreadinesslab.entitlements.ProEntitlementProvider
import com.retirementreadinesslab.model.BudgetProfile
import com.retirementreadinesslab.model.FeatureAccess
import com.retirementreadinesslab.model.PostRetirementAllocationStrategy
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SimulationResult
import com.retirementreadinesslab.model.forFeatureAccess
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
    private val entitlementProvider: ProEntitlementProvider,
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

    var isPurchasingPro by mutableStateOf(false)
        private set

    var isRestoringPro by mutableStateOf(false)
        private set

    var storageMessage by mutableStateOf<String?>(null)
        private set

    var lastRunMessage by mutableStateOf<String?>(null)
        private set

    var hasCompletedFirstLaunch by mutableStateOf(false)
        private set

    var isProUnlocked by mutableStateOf(false)
        private set

    val featureAccess: FeatureAccess
        get() = FeatureAccess(isProUnlocked = isProUnlocked)

    val entitlementProviderName: String
        get() = entitlementProvider.providerName

    val hasDeveloperEntitlementControls: Boolean
        get() = entitlementProvider.allowsDeveloperOverrides

    val supportsUserPurchases: Boolean
        get() = entitlementProvider.supportsUserPurchases

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
            val entitlement = entitlementProvider.currentEntitlement(stored.isProUnlocked)
            isProUnlocked = entitlement.isProUnlocked
            if (entitlement.shouldPersist && entitlement.isProUnlocked != stored.isProUnlocked) {
                runCatching {
                    repository.setProUnlocked(entitlement.isProUnlocked)
                }.onFailure {
                    storageMessage = "Pro unlock could not be saved locally."
                }
            }
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
        if (!isProUnlocked) {
            storageMessage = "Post-retirement allocation optimization requires Pro."
            return
        }
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

    fun applyVerifiedProUnlock() {
        applyProEntitlement(
            entitlement = ProEntitlement(
                isProUnlocked = true,
                message = "Pro unlock restored."
            ),
            persist = true
        )
    }

    fun purchasePro(activity: Activity?) {
        if (activity == null) {
            storageMessage = "Pro purchase could not start from this screen."
            return
        }
        if (!entitlementProvider.supportsUserPurchases) {
            storageMessage = "Pro purchase is not available in this build."
            return
        }
        if (isPurchasingPro) return

        isPurchasingPro = true
        storageMessage = "Opening Google Play purchase..."
        scope.launch {
            val entitlement = runCatching {
                entitlementProvider.purchasePro(
                    activity = activity,
                    storedLocalUnlock = isProUnlocked
                )
            }.getOrElse {
                ProEntitlement(
                    isProUnlocked = isProUnlocked,
                    message = "Pro purchase failed: ${it.message ?: "unknown error"}",
                    shouldPersist = false
                )
            }
            applyProEntitlement(entitlement, persist = true)
            isPurchasingPro = false
        }
    }

    fun restoreProPurchase() {
        if (!entitlementProvider.supportsUserPurchases) {
            storageMessage = "Restore purchase is not available in this build."
            return
        }
        if (isRestoringPro) return

        isRestoringPro = true
        storageMessage = "Checking Google Play purchase..."
        scope.launch {
            val entitlement = runCatching {
                entitlementProvider.currentEntitlement(isProUnlocked)
            }.getOrElse {
                ProEntitlement(
                    isProUnlocked = isProUnlocked,
                    message = "Pro purchase could not be checked: ${it.message ?: "unknown error"}",
                    shouldPersist = false
                )
            }
            val withMessage = if (entitlement.message == null) {
                entitlement.copy(
                    message = if (entitlement.isProUnlocked) {
                        "Pro purchase restored."
                    } else {
                        "No Pro purchase was found for this Google Play account."
                    }
                )
            } else {
                entitlement
            }
            applyProEntitlement(withMessage, persist = true)
            isRestoringPro = false
        }
    }

    fun unlockProWithPromoCode(activity: Activity?, promoCode: String) {
        if (!entitlementProvider.allowsDeveloperOverrides && !entitlementProvider.supportsUserPurchases) {
            storageMessage = "Promo code unlock is not available in this build."
            return
        }
        scope.launch {
            val entitlement = entitlementProvider.redeemPromoCode(
                activity = activity,
                promoCode = promoCode,
                storedLocalUnlock = isProUnlocked
            )
            applyProEntitlement(entitlement, persist = true)
        }
    }

    fun setDeveloperProUnlocked(isUnlocked: Boolean) {
        if (!entitlementProvider.allowsDeveloperOverrides) return
        scope.launch {
            val entitlement = entitlementProvider.setDeveloperOverride(
                isProUnlocked = isUnlocked,
                storedLocalUnlock = isProUnlocked
            )
            applyProEntitlement(entitlement, persist = true)
        }
    }

    private fun runScenarioAsync(id: String) {
        val scenario = scenarios.firstOrNull { it.id == id } ?: return
        val runnableScenario = scenario.forFeatureAccess(featureAccess)
        val requestId = ++scenarioRunId
        isRunning = true
        lastRunMessage = "Running stress test..."
        scope.launch {
            val run = runCatching {
                withContext(Dispatchers.Default) {
                    RetirementSimulator.run(runnableScenario)
                }
            }
            run.onSuccess { result ->
                val stillCurrent = scenarios.firstOrNull { it.id == id } == scenario
                if (stillCurrent && requestId == scenarioRunId) {
                    results[id] = result
                    lastRunMessage = buildRunMessage(runnableScenario, result)
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
        if (!isProUnlocked) {
            labAnalyses.remove(id)
            isAnalyzingLab = false
            return
        }
        val scenario = scenarios.firstOrNull { it.id == id } ?: return
        val runnableScenario = scenario.forFeatureAccess(featureAccess)
        val requestId = ++labAnalysisRunId
        isAnalyzingLab = id == selectedScenarioId
        scope.launch {
            val run = runCatching {
                withContext(Dispatchers.Default) {
                    ScenarioLabAnalyzer.analyze(runnableScenario)
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

    private fun applyProEntitlement(entitlement: ProEntitlement, persist: Boolean) {
        val changed = isProUnlocked != entitlement.isProUnlocked
        isProUnlocked = entitlement.isProUnlocked
        entitlement.message?.let { storageMessage = it }
        if (!isProUnlocked) {
            labAnalyses.clear()
            allocationOptimizations.clear()
        }
        if (changed) {
            results.remove(selectedScenarioId)
            runScenarioAsync(selectedScenarioId)
            runLabAnalysisAsync(selectedScenarioId)
        }
        if (persist && entitlement.shouldPersist) {
            scope.launch {
                runCatching {
                    repository.setProUnlocked(entitlement.isProUnlocked)
                }.onFailure {
                    storageMessage = "Pro unlock could not be saved locally."
                }
            }
        }
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
        entitlementProvider = DefaultEntitlementProvider(applicationContext),
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
