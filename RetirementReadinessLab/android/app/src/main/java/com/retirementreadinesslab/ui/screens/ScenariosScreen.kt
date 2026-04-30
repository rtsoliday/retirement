package com.retirementreadinesslab.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CopyAll
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Restore
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.simulation.ScenarioComparison
import com.retirementreadinesslab.simulation.ScenarioComparisonRow
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCompactCurrency
import com.retirementreadinesslab.ui.asPercent
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabCaution
import com.retirementreadinesslab.ui.theme.LabMutedText
import com.retirementreadinesslab.ui.theme.LabRisk
import com.retirementreadinesslab.ui.theme.LabSuccess

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ScenariosScreen(state: RetirementLabState) {
    val results = state.allResults()
    val comparisonSummary = if (state.scenarios.isNotEmpty()) {
        ScenarioComparison.build(
            scenarios = state.scenarios.toList(),
            results = results,
            baseline = state.selectedScenario
        )
    } else {
        null
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("scenarios-screen"),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp)
    ) {
        item {
            SectionHeader(
                title = "Scenarios",
                subtitle = "Save alternatives and compare the decisions that move the result."
            )
        }

        if (state.isRunning) {
            item {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
        }

        state.lastRunMessage?.let { message ->
            item {
                Text(message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
            }
        }

        item {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                Button(
                    onClick = state::duplicateSelected,
                    enabled = !state.isRunning,
                    modifier = Modifier
                        .weight(1f)
                        .testTag("duplicate-scenario-button")
                ) {
                    Icon(Icons.Filled.CopyAll, contentDescription = "Duplicate scenario")
                    Text("Duplicate")
                }
                OutlinedButton(
                    onClick = state::runSelectedScenario,
                    enabled = !state.isRunning,
                    modifier = Modifier
                        .weight(1f)
                        .testTag("run-scenario-button")
                ) {
                    Icon(Icons.Filled.PlayArrow, contentDescription = "Run selected scenario")
                    Text("Run")
                }
            }
        }

        item {
            Button(
                onClick = state::runAllScenarios,
                enabled = !state.isRunning,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("run-all-scenarios-button")
            ) {
                Icon(Icons.Filled.PlayArrow, contentDescription = "Run all scenarios")
                Text(if (state.isRunning) "Running..." else "Run All And Compare")
            }
        }

        item {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(
                    onClick = state::deleteSelected,
                    enabled = state.scenarios.size > 1 && !state.isRunning,
                    modifier = Modifier
                        .weight(1f)
                        .testTag("delete-scenario-button")
                ) {
                    Icon(Icons.Filled.Delete, contentDescription = "Delete selected scenario")
                    Text("Delete")
                }
                OutlinedButton(
                    onClick = state::restoreSamplePlans,
                    enabled = !state.isRunning,
                    modifier = Modifier
                        .weight(1f)
                        .testTag("restore-samples-button")
                ) {
                    Icon(Icons.Filled.Restore, contentDescription = "Restore sample scenarios")
                    Text("Restore Samples")
                }
            }
        }

        state.storageMessage?.let { message ->
            item {
                Text(message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            }
        }

        if (results.isNotEmpty()) {
            item {
                ComparisonCard(
                    rows = comparisonSummary?.rows.orEmpty(),
                    bestRow = comparisonSummary?.bestReadinessRow
                )
            }
        }

        items(state.scenarios, key = { it.id }) { scenario ->
            ScenarioCard(
                scenario = scenario,
                selected = scenario.id == state.selectedScenarioId,
                successProbability = state.resultFor(scenario.id)?.successProbability,
                onSelect = { state.selectScenario(scenario.id) }
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ComparisonCard(
    rows: List<ScenarioComparisonRow>,
    bestRow: ScenarioComparisonRow?
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Comparison", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(
                "Readiness, failure timing, and changed assumptions against the selected baseline.",
                style = MaterialTheme.typography.bodySmall,
                color = LabMutedText
            )
            bestRow?.let { row ->
                BestPlanSummary(row)
            }
            rows.forEach { row ->
                ComparisonRow(row)
            }
        }
    }
}

@Composable
private fun BestPlanSummary(row: ScenarioComparisonRow) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.08f), RoundedCornerShape(8.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Text("Strongest quick comparison", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
        Text(
            "${row.scenarioName}: ${row.readiness?.asPercent() ?: "Not run"} readiness",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold
        )
        Text(
            "Median ending ${row.medianEndingBalance?.asCompactCurrency() ?: "N/A"} · ${row.mostCommonFailureWindow}",
            style = MaterialTheme.typography.bodySmall,
            color = LabMutedText
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ComparisonRow(
    row: ScenarioComparisonRow
) {
    val probability = row.readiness ?: 0.0
    val color = readinessColor(probability)

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(row.scenarioName, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Medium)
                Text(
                    "Median ${row.medianEndingBalance?.asCompactCurrency() ?: "N/A"} · ${row.mostCommonFailureWindow}",
                    style = MaterialTheme.typography.labelSmall,
                    color = LabMutedText
                )
            }
            Text(
                row.readiness?.asPercent() ?: "Not run",
                style = MaterialTheme.typography.labelLarge,
                color = color,
                fontWeight = FontWeight.SemiBold
            )
        }
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .height(10.dp)
                    .background(MaterialTheme.colorScheme.outline.copy(alpha = 0.30f), RoundedCornerShape(8.dp))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(probability.toFloat().coerceIn(0f, 1f))
                        .height(10.dp)
                        .background(color, RoundedCornerShape(8.dp))
                )
            }
            Text(
                row.primaryRisk?.let { "Risk $it" } ?: "",
                style = MaterialTheme.typography.labelSmall,
                color = LabMutedText,
                modifier = Modifier.width(92.dp)
            )
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(7.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            row.changedAssumptions.take(6).forEach { change ->
                ChangeChip(change)
            }
        }
    }
}

@Composable
private fun ChangeChip(label: String) {
    Row(
        modifier = Modifier
            .background(MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f), RoundedCornerShape(8.dp))
            .padding(horizontal = 9.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(7.dp)
                .background(MaterialTheme.colorScheme.secondary, RoundedCornerShape(8.dp))
        )
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurface)
    }
}

private fun readinessColor(probability: Double): Color {
    return when {
        probability >= 0.82 -> LabSuccess
        probability >= 0.65 -> LabCaution
        else -> LabRisk
    }
}

@Composable
private fun ScenarioCard(
    scenario: RetirementScenario,
    selected: Boolean,
    successProbability: Double?,
    onSelect: () -> Unit
) {
    Card(
        onClick = onSelect,
        colors = CardDefaults.cardColors(
            containerColor = if (selected) MaterialTheme.colorScheme.primary.copy(alpha = 0.10f) else MaterialTheme.colorScheme.surface
        ),
        modifier = Modifier
            .fillMaxWidth()
            .testTag("scenario-card-${scenario.id}")
            .semantics {
                contentDescription = if (selected) {
                    "${scenario.name} scenario selected"
                } else {
                    "${scenario.name} scenario"
                }
            }
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(scenario.name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        "Retire ${scenario.household.retirementAge} · Claim ${scenario.socialSecurity.claimAge}",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                }
                Text(
                    successProbability?.asPercent() ?: "Not run",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.SemiBold
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                Text(
                    "Spend ${scenario.spending.annualBaseSpending.asCompactCurrency()}",
                    style = MaterialTheme.typography.bodySmall,
                    color = LabMutedText
                )
                Text(
                    "Assets ${scenario.accounts.total.asCompactCurrency()}",
                    style = MaterialTheme.typography.bodySmall,
                    color = LabMutedText
                )
            }
        }
    }
}
