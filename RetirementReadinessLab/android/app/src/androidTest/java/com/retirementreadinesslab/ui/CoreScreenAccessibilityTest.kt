package com.retirementreadinesslab.ui

import androidx.compose.ui.test.assertHasClickAction
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.longClick
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performTouchInput
import androidx.test.platform.app.InstrumentationRegistry
import com.retirementreadinesslab.data.ScenarioRepository
import com.retirementreadinesslab.model.sampleScenarios
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.screens.AssumptionsScreen
import com.retirementreadinesslab.ui.screens.DashboardScreen
import com.retirementreadinesslab.ui.screens.LabScreen
import com.retirementreadinesslab.ui.screens.OnboardingScreen
import com.retirementreadinesslab.ui.screens.ReportsScreen
import com.retirementreadinesslab.ui.screens.ResultsScreen
import com.retirementreadinesslab.ui.screens.ScenariosScreen
import com.retirementreadinesslab.ui.theme.RetirementReadinessLabTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class CoreScreenAccessibilityTest {
    @get:Rule
    val compose = createComposeRule()

    @Test
    fun dashboardExposesPrimaryActionAndReadinessGauge() {
        compose.setContent {
            RetirementReadinessLabTheme {
                DashboardScreen(
                    state = testState(),
                    onViewResults = {}
                )
            }
        }

        compose.onNodeWithTag("dashboard-screen").assertIsDisplayed()
        compose.onNodeWithContentDescription("Readiness gauge: 0% readiness").assertIsDisplayed()
        compose.onNodeWithText("Run Stress Test").assertHasClickAction()
        compose.onNodeWithText("Retirement Readiness Lab").assertIsDisplayed()
    }

    @Test
    fun setupExposesEditableInputsAndPrimaryAction() {
        compose.setContent {
            RetirementReadinessLabTheme {
                OnboardingScreen(state = testState())
            }
        }

        compose.onNodeWithTag("setup-screen").assertIsDisplayed()
        compose.onNodeWithText("Guided setup").assertIsDisplayed()
        compose.onNodeWithText("Retirement target").assertIsDisplayed()
        compose.onNodeWithText("Pre-tax accounts").assertIsDisplayed()
        scrollScreenToText("setup-screen", "Long-term care risk")
        compose.onNodeWithText("Long-term care risk").assertIsDisplayed()
        scrollScreenToText("setup-screen", "Run Current Scenario")
        compose.onNodeWithText("Run Current Scenario").assertHasClickAction()
    }

    @Test
    fun scenariosExposeComparisonControlsAndDuplicateScenario() {
        val state = testState()
        compose.setContent {
            RetirementReadinessLabTheme {
                ScenariosScreen(state = state)
            }
        }

        compose.onNodeWithTag("scenarios-screen").assertIsDisplayed()
        compose.onNodeWithText("Scenarios").assertIsDisplayed()
        scrollScreenToTag("scenarios-screen", "run-all-scenarios-button")
        compose.onNodeWithTag("run-all-scenarios-button").assertHasClickAction().performClick()
        compose.waitUntil(timeoutMillis = 15_000) {
            !state.isRunning && state.resultFor("base-plan") != null
        }
        val baseReadiness = state.resultFor("base-plan")!!.successProbability.asPercent()
        scrollScreenToTag("scenarios-screen", "comparison-readiness-base-plan")
        compose.onNodeWithTag("comparison-readiness-base-plan").assertIsDisplayed()
        compose.onNodeWithContentDescription("Comparison readiness for Base plan: $baseReadiness").assertIsDisplayed()
        scrollScreenToTag("scenarios-screen", "scenario-card-base-plan")
        compose.onNodeWithTag("scenario-card-base-plan").performTouchInput { longClick() }
        compose.onNodeWithText("Scenario actions").assertIsDisplayed()
        compose.onNodeWithTag("scenario-action-name-input")
            .performTextClearance()
        compose.onNodeWithTag("scenario-action-name-input")
            .performTextInput("Travel plan")
        compose.onNodeWithTag("scenario-action-rename-button").assertHasClickAction().performClick()
        compose.waitForIdle()

        assertEquals("Travel plan", state.selectedScenario.name)
        scrollScreenToTag("scenarios-screen", "duplicate-scenario-button")
        compose.onNodeWithTag("duplicate-scenario-button").assertHasClickAction().performClick()
        compose.waitForIdle()

        assertEquals(4, state.scenarios.size)
        assertEquals("Travel plan copy", state.selectedScenario.name)
        val copiedScenarioId = state.selectedScenarioId
        scrollScreenToTag("scenarios-screen", "scenario-card-$copiedScenarioId")
        compose.onNodeWithTag("scenario-card-$copiedScenarioId").performTouchInput { longClick() }
        compose.onNodeWithText("Scenario actions").assertIsDisplayed()
        compose.onNodeWithTag("scenario-action-delete-button").assertHasClickAction().performClick()
        compose.waitForIdle()

        assertEquals(3, state.scenarios.size)
        compose.waitUntil(timeoutMillis = 15_000) { !state.isRunning }
        scrollScreenToTag("scenarios-screen", "run-all-scenarios-button")
        compose.onNodeWithTag("run-all-scenarios-button").assertHasClickAction()
        scrollScreenToTag("scenarios-screen", "restore-samples-button")
        compose.onNodeWithTag("restore-samples-button").assertHasClickAction()
    }

    @Test
    fun reportsExposePrivacyExportImportAndDeleteConfirmation() {
        compose.setContent {
            RetirementReadinessLabTheme {
                ReportsScreen(state = testState())
            }
        }

        compose.onNodeWithTag("reports-screen").assertIsDisplayed()
        compose.onNodeWithText("Privacy and disclosures").assertIsDisplayed()
        compose.onNodeWithTag("share-pdf-report-button").assertHasClickAction()
        compose.onNodeWithTag("share-text-report-button").assertHasClickAction()
        compose.onNodeWithTag("share-scenario-backup-button").assertHasClickAction()
        scrollScreenToTag("reports-screen", "share-comparison-csv-button")
        compose.onNodeWithTag("share-comparison-csv-button").assertHasClickAction()

        scrollScreenToTag("reports-screen", "json-backup-input")
        compose.onNodeWithTag("json-backup-input").performTextInput("not json")
        scrollScreenToTag("reports-screen", "import-backup-button")
        compose.onNodeWithTag("import-backup-button").assertHasClickAction().performClick()
        compose.onNodeWithText("Backup format could not be read. Paste the full JSON backup text.").assertIsDisplayed()

        scrollScreenToTag("reports-screen", "delete-local-data-button")
        compose.onNodeWithTag("delete-local-data-button").assertHasClickAction().performClick()
        compose.onNodeWithTag("confirm-delete-local-data-button").assertHasClickAction()
        compose.onNodeWithTag("cancel-delete-local-data-button").assertHasClickAction().performClick()
    }

    @Test
    fun assumptionsShowValidationErrorForInvalidAge() {
        compose.setContent {
            RetirementReadinessLabTheme {
                AssumptionsScreen(state = testState())
            }
        }

        compose.onNodeWithTag("assumptions-screen").assertIsDisplayed()
        compose.onNodeWithTag("current-age-input")
            .performScrollTo()
            .performTextClearance()
        compose.onNodeWithTag("current-age-input")
            .performTextInput("10")
        scrollScreenToTag("assumptions-screen", "apply-assumptions-button")
        compose.onNodeWithTag("apply-assumptions-button")
            .assertHasClickAction()
            .performClick()

        compose.onNodeWithText("Current age must be between 18 and 90.").assertIsDisplayed()
    }

    @Test
    fun resultsExposeCalculationProvenance() {
        val state = testState()

        compose.setContent {
            RetirementReadinessLabTheme {
                ResultsScreen(state = state)
            }
        }
        state.runSelectedScenario()
        compose.waitUntil(timeoutMillis = 15_000) {
            !state.isRunning && state.selectedResult != null
        }

        compose.onNodeWithTag("results-screen").assertIsDisplayed()
        scrollScreenToText("results-screen", "Calculation provenance")
        compose.onNodeWithText("Calculation provenance").assertIsDisplayed()
        scrollScreenToText("results-screen", "Monthly cashflow model with annual result bands")
        compose.onNodeWithText("Monthly cashflow model with annual result bands").assertIsDisplayed()
        scrollScreenToText("results-screen", "Assumption fingerprint")
        compose.onNodeWithText("Assumption fingerprint").assertIsDisplayed()
    }

    @Test
    fun labScreenExposesStableScreenAnchor() {
        compose.setContent {
            RetirementReadinessLabTheme {
                LabScreen(state = testState())
            }
        }

        compose.onNodeWithTag("lab-screen").assertIsDisplayed()
        compose.onNodeWithText("Scenario Lab").assertIsDisplayed()
        compose.onNodeWithText("Quick lab mode").assertIsDisplayed()
    }

    private fun testState(): RetirementLabState {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        return RetirementLabState(
            repository = ScenarioRepository(context),
            scope = CoroutineScope(Dispatchers.Main),
            initialScenarios = sampleScenarios().map { scenario ->
                scenario.copy(numberOfSimulations = 75)
            }
        )
    }

    private fun scrollScreenToText(screenTag: String, text: String) {
        compose.onNodeWithTag(screenTag).performScrollToNode(hasText(text))
    }

    private fun scrollScreenToTag(screenTag: String, targetTag: String) {
        compose.onNodeWithTag(screenTag).performScrollToNode(hasTestTag(targetTag))
    }
}
