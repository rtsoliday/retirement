package com.retirementreadinesslab.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.HourglassBottom
import androidx.compose.material.icons.filled.Savings
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.retirementreadinesslab.model.FailureAgeBucket
import com.retirementreadinesslab.model.SimulationResult
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCompactCurrency
import com.retirementreadinesslab.ui.asPercent
import com.retirementreadinesslab.ui.components.BalancePathChart
import com.retirementreadinesslab.ui.components.KeyValueRow
import com.retirementreadinesslab.ui.components.MetricCard
import com.retirementreadinesslab.ui.components.RiskPill
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabCaution
import com.retirementreadinesslab.ui.theme.LabMutedText
import com.retirementreadinesslab.ui.theme.LabPrimary
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun ResultsScreen(state: RetirementLabState) {
    val scenario = state.selectedScenario
    val result = state.selectedResult

    LazyColumn(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp)
    ) {
        item {
            SectionHeader(
                title = "Results Detail",
                subtitle = scenario.name
            )
        }

        if (result == null) {
            item {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                    Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("No result yet", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(
                            "Run a stress test from Home to see detailed paths, failure timing, and assumptions.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = LabMutedText
                        )
                    }
                }
            }
            return@LazyColumn
        }

        item {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                MetricCard(
                    title = "Readiness",
                    value = result.successProbability.asPercent(),
                    detail = "Simulations lasting through age ${scenario.household.targetEndAge}",
                    icon = Icons.Filled.Savings,
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    title = "Median failure",
                    value = result.medianFailureAge?.toString() ?: "N/A",
                    detail = "Only failed runs",
                    icon = Icons.Filled.HourglassBottom,
                    modifier = Modifier.weight(1f)
                )
            }
        }

        item {
            BalancePathChart(bands = result.balanceBands)
        }

        item {
            EndingBalanceCard(result)
        }

        item {
            FailureTimingCard(result.failureAgeBuckets)
        }

        item {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Risk readout", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        RiskPill("Market", result.riskBreakdown.market)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        RiskPill("Healthcare", result.riskBreakdown.healthcare)
                        RiskPill("Taxes", result.riskBreakdown.taxes)
                    }
                    Text(
                        result.riskBreakdown.recommendedNextTest,
                        style = MaterialTheme.typography.bodyMedium,
                        color = LabMutedText
                    )
                }
            }
        }

        item {
            ProvenanceCard(result)
        }
    }
}

@Composable
private fun ProvenanceCard(result: SimulationResult) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Calculation provenance", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            KeyValueRow("Generated", result.generatedAtEpochMillis.formatGeneratedAt())
            KeyValueRow("Engine", result.provenance.engineVersion)
            KeyValueRow("Cadence", result.provenance.engineCadence)
            KeyValueRow("Tax table", result.provenance.taxTableVersion)
            KeyValueRow("Mortality model", result.provenance.mortalityModelVersion)
            KeyValueRow("Random seed", result.provenance.randomSeed.toString())
            KeyValueRow("Simulations", result.provenance.simulationCount.toString())
            KeyValueRow("Assumption fingerprint", result.provenance.assumptionFingerprint)
        }
    }
}

@Composable
private fun EndingBalanceCard(result: SimulationResult) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Ending balance range", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            KeyValueRow("10th percentile", result.pessimisticEndingBalance.asCompactCurrency())
            KeyValueRow("Median", result.medianEndingBalance.asCompactCurrency())
            KeyValueRow("90th percentile", result.optimisticEndingBalance.asCompactCurrency())
        }
    }
}

@Composable
private fun FailureTimingCard(buckets: List<FailureAgeBucket>) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                androidx.compose.material3.Icon(Icons.Filled.Warning, contentDescription = null, tint = LabCaution)
                Text("Failure timing", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            }
            if (buckets.isEmpty()) {
                Text(
                    "No failed runs in this simulation set.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = LabMutedText
                )
            } else {
                buckets.forEach { bucket ->
                    FailureBucketRow(bucket)
                }
            }
        }
    }
}

@Composable
private fun FailureBucketRow(bucket: FailureAgeBucket) {
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Age ${bucket.label}", style = MaterialTheme.typography.labelLarge)
            Text(
                "${bucket.count} runs · ${bucket.shareOfFailures.asPercent()}",
                style = MaterialTheme.typography.labelMedium,
                color = LabMutedText
            )
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(9.dp)
                .background(MaterialTheme.colorScheme.outline.copy(alpha = 0.25f), RoundedCornerShape(8.dp))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(bucket.shareOfFailures.toFloat().coerceIn(0f, 1f))
                    .height(9.dp)
                    .background(LabPrimary, RoundedCornerShape(8.dp))
            )
        }
    }
}

private fun Long.formatGeneratedAt(): String {
    return SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date(this))
}
