package com.retirementreadinesslab.ui

import androidx.compose.ui.test.assertHasClickAction
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
import androidx.test.platform.app.InstrumentationRegistry
import com.retirementreadinesslab.data.ScenarioRepository
import com.retirementreadinesslab.model.sampleScenarios
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.screens.DashboardScreen
import com.retirementreadinesslab.ui.screens.OnboardingScreen
import com.retirementreadinesslab.ui.screens.ReportsScreen
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
        compose.onNodeWithText("Long-term care risk").assertIsDisplayed()
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
        compose.onNodeWithTag("duplicate-scenario-button").assertHasClickAction().performClick()
        compose.waitForIdle()

        assertEquals(4, state.scenarios.size)
        assertEquals("Base plan copy", state.selectedScenario.name)
        compose.onNodeWithTag("run-all-scenarios-button").assertHasClickAction()
        compose.onNodeWithTag("delete-scenario-button").assertHasClickAction()
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

        compose.onNodeWithTag("json-backup-input").performScrollTo().performTextInput("not json")
        compose.onNodeWithTag("import-backup-button").performScrollTo().assertHasClickAction().performClick()
        compose.onNodeWithText("Backup format could not be read. Paste the full JSON backup text.").assertIsDisplayed()

        compose.onNodeWithTag("delete-local-data-button").performScrollTo().assertHasClickAction().performClick()
        compose.onNodeWithTag("confirm-delete-local-data-button").assertHasClickAction()
        compose.onNodeWithTag("cancel-delete-local-data-button").assertHasClickAction().performClick()
        compose.onNodeWithTag("delete-local-data-button").assertHasClickAction()
    }

    private fun testState(): RetirementLabState {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        return RetirementLabState(
            repository = ScenarioRepository(context),
            scope = CoroutineScope(Dispatchers.Main),
            initialScenarios = sampleScenarios()
        )
    }
}
