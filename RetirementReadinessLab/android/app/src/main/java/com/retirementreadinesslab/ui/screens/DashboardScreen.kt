package com.retirementreadinesslab.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.TrendingUp
import androidx.compose.material.icons.filled.HealthAndSafety
import androidx.compose.material.icons.filled.Savings
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.retirementreadinesslab.model.warnings
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCompactCurrency
import com.retirementreadinesslab.ui.asPercent
import com.retirementreadinesslab.ui.components.BalancePathChart
import com.retirementreadinesslab.ui.components.MetricCard
import com.retirementreadinesslab.ui.components.ReadinessGauge
import com.retirementreadinesslab.ui.components.RiskPill
import com.retirementreadinesslab.ui.components.ScenarioWarningCard
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabMutedText

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun DashboardScreen(
    state: RetirementLabState,
    onViewResults: () -> Unit
) {
    val scenario = state.selectedScenario
    val result = state.selectedResult
    val decisionEstimate = state.selectedDecisionEstimate
    val warnings = scenario.warnings()

    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
            .testTag("dashboard-screen"),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        item {
            SectionHeader(
                title = "Retirement Readiness Lab",
                subtitle = scenario.name
            )
        }

        if (state.isRunning) {
            item {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
        }

        state.lastRunMessage?.let { message ->
            item {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.08f))) {
                    Text(
                        text = message,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }
        }

        item {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.spacedBy(18.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    ReadinessGauge(probability = result?.successProbability ?: 0.0)
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.weight(1f)) {
                        Text(
                            text = if (result == null) "Run a stress test to see your result." else readinessText(result.successProbability),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = result?.riskBreakdown?.recommendedNextTest
                                ?: "Start with the base plan, then compare retirement age and spending.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = LabMutedText
                        )
                        Button(
                            onClick = state::runSelectedScenario,
                            enabled = !state.isRunning
                        ) {
                            Text(if (state.isRunning) "Running..." else "Run Stress Test")
                        }
                        if (result != null) {
                            Button(
                                onClick = onViewResults,
                                enabled = !state.isRunning
                            ) {
                                Text("View Details")
                            }
                        }
                    }
                }
            }
        }

        if (result != null) {
            item {
                BalancePathChart(bands = result.balanceBands)
            }

            item {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricCard(
                        title = "Retirement age",
                        value = scenario.household.retirementAge.toString(),
                        detail = "Target through age ${scenario.household.targetEndAge}",
                        icon = Icons.Filled.Schedule,
                        modifier = Modifier.weight(1f)
                    )
                    MetricCard(
                        title = "Median ending",
                        value = result.medianEndingBalance.asCompactCurrency(),
                        detail = "10th percentile ${result.pessimisticEndingBalance.asCompactCurrency()}",
                        icon = Icons.Filled.Savings,
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            item {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricCard(
                        title = "Earliest age",
                        value = decisionEstimate?.earliestRetirementAge?.toString()
                            ?: if (state.isAnalyzingLab) "Finding..." else "N/A",
                        detail = decisionEstimate?.earliestRetirementReadiness?.let {
                            "${it.asPercent()} quick readiness"
                        } ?: decisionEstimate?.let {
                            "${it.targetReadiness.asPercent()} not found by age 70"
                        } ?: "Quick target estimate",
                        icon = Icons.Filled.Schedule,
                        modifier = Modifier.weight(1f)
                    )
                    MetricCard(
                        title = "Safe spending",
                        value = decisionEstimate?.safeAnnualSpending?.asCompactCurrency()
                            ?: if (state.isAnalyzingLab) "Finding..." else "N/A",
                        detail = decisionEstimate?.safeSpendingReadiness?.let {
                            "${it.asPercent()} quick readiness"
                        } ?: "Target not found",
                        icon = Icons.AutoMirrored.Filled.TrendingUp,
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            item {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricCard(
                        title = "Annual spending",
                        value = scenario.spending.annualBaseSpending.asCompactCurrency(),
                        detail = "In today's dollars",
                        icon = Icons.AutoMirrored.Filled.TrendingUp,
                        modifier = Modifier.weight(1f)
                    )
                    MetricCard(
                        title = "Healthcare",
                        value = (scenario.healthcare.preMedicareMonthlyPremium * 12.0).asCompactCurrency(),
                        detail = "Pre-Medicare annual premium",
                        icon = Icons.Filled.HealthAndSafety,
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            item {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                    Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("Risk scan", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            RiskPill("Market", result.riskBreakdown.market)
                            RiskPill("Longevity", result.riskBreakdown.longevity)
                            RiskPill("Healthcare", result.riskBreakdown.healthcare)
                            RiskPill("Taxes", result.riskBreakdown.taxes)
                            RiskPill("Spending", result.riskBreakdown.spending)
                        }
                    }
                }
            }

            item {
                ScenarioWarningCard(title = "Assumption checks", warnings = warnings)
            }

            item {
                Text(
                    text = "Educational estimate only. Results depend on user-entered assumptions and are not financial, tax, legal, or investment advice.",
                    style = MaterialTheme.typography.bodySmall,
                    color = LabMutedText
                )
            }
        }
    }
}

private fun readinessText(probability: Double): String {
    return when {
        probability >= 0.82 -> "Most simulated paths lasted. Compare tradeoffs before changing course."
        probability >= 0.65 -> "The plan is workable but sensitive to assumptions."
        else -> "This plan needs pressure relief before it looks durable."
    }
}
