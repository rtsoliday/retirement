package com.retirementreadinesslab.ui

import androidx.compose.ui.test.assertHasClickAction
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import androidx.test.platform.app.InstrumentationRegistry
import com.retirementreadinesslab.data.ScenarioRepository
import com.retirementreadinesslab.model.sampleScenarios
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.screens.BudgetScreen
import com.retirementreadinesslab.ui.screens.DashboardScreen
import com.retirementreadinesslab.ui.screens.LabScreen
import com.retirementreadinesslab.ui.screens.ReportsScreen
import com.retirementreadinesslab.ui.screens.ResultsScreen
import com.retirementreadinesslab.ui.screens.SetupScreen
import com.retirementreadinesslab.ui.theme.RetirementReadinessLabTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
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
        compose.onAllNodesWithText("Base plan").assertCountEquals(0)
    }

    @Test
    fun setupExposesEditableInputsAndPrimaryAction() {
        compose.setContent {
            RetirementReadinessLabTheme {
                SetupScreen(state = testState())
            }
        }

        compose.onNodeWithTag("setup-screen").assertIsDisplayed()
        compose.onNodeWithText("Setup").assertIsDisplayed()
        compose.onNodeWithText("Household profile").assertIsDisplayed()
        scrollScreenToText("setup-screen", "Max spending cut (%)")
        compose.onNodeWithText("Max spending cut (%)").assertIsDisplayed()
        scrollScreenToText("setup-screen", "Mortgage time left")
        compose.onNodeWithText("Mortgage time left").assertIsDisplayed()
        compose.onNodeWithText("Years left").assertIsDisplayed()
        compose.onNodeWithText("Months left").assertIsDisplayed()
        scrollScreenToText("setup-screen", "Monthly rent")
        compose.onNodeWithText("Current mortgage balance").assertIsDisplayed()
        compose.onNodeWithText("Current home value").assertIsDisplayed()
        compose.onNodeWithText("Monthly rent").assertIsDisplayed()
        scrollScreenToText("setup-screen", "Long-term care shock")
        compose.onNodeWithText("Long-term care shock").assertIsDisplayed()
        scrollScreenToText("setup-screen", "Pre-tax accounts")
        compose.onNodeWithText("Pre-tax accounts").assertIsDisplayed()
        scrollScreenToText("setup-screen", "Post-retirement investment ratios")
        compose.onNodeWithText("Post-retirement investment ratios").assertIsDisplayed()
        compose.onNodeWithTag("restore-setup-post-retirement-allocation-defaults").assertHasClickAction()
        scrollScreenToText("setup-screen", "Primary Social Security at FRA")
        compose.onNodeWithText("Social Security").assertIsDisplayed()
        compose.onNodeWithText("Primary Social Security at FRA").assertIsDisplayed()
        compose.onNodeWithText("Run Current Scenario").assertHasClickAction()
    }

    @Test
    fun budgetExposesMonthlyEditorsAndApplyAction() {
        compose.setContent {
            RetirementReadinessLabTheme {
                BudgetScreen(state = testState())
            }
        }

        compose.onNodeWithTag("budget-screen").assertIsDisplayed()
        compose.onNodeWithText("Budget").assertIsDisplayed()
        compose.onNodeWithText("Yearly property taxes").assertIsDisplayed()
        scrollScreenToText("budget-screen", "Monthly spending")
        compose.onNodeWithTag("budget-month-label").assertIsDisplayed()
        compose.onNodeWithTag("add-checking-bill-button").assertHasClickAction()
        compose.onNodeWithTag("add-credit-card-bill-button").assertHasClickAction()
        scrollScreenToTag("budget-screen", "apply-budget-spending-button")
        compose.onNodeWithText("Use Estimate For Annual Base Spending").assertIsDisplayed()
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
    fun setupShowsValidationErrorForInvalidAge() {
        compose.setContent {
            RetirementReadinessLabTheme {
                SetupScreen(state = testState())
            }
        }

        compose.onNodeWithTag("setup-screen").assertIsDisplayed()
        compose.onNodeWithTag("current-age-input")
            .performScrollTo()
            .performTextClearance()
        compose.onNodeWithTag("current-age-input")
            .performTextInput("10")
        compose.onNodeWithTag("apply-setup-button")
            .assertHasClickAction()
            .performClick()

        compose.onNodeWithText("Current age must be between 18 and 90.").assertIsDisplayed()
    }

    @Test
    fun resultsHideScenarioNameAndCalculationProvenance() {
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
        compose.onNodeWithText("Results Detail").assertIsDisplayed()
        compose.onAllNodesWithText("Base plan").assertCountEquals(0)
        compose.onAllNodesWithText("Calculation provenance").assertCountEquals(0)
        compose.onAllNodesWithText("Monthly cashflow model with annual result bands").assertCountEquals(0)
        compose.onAllNodesWithText("Assumption fingerprint").assertCountEquals(0)
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
