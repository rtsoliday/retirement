package com.retirementreadinesslab.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.TrendingDown
import androidx.compose.material.icons.filled.HealthAndSafety
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Savings
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Toll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.retirementreadinesslab.simulation.LabComparisonAnalysis
import com.retirementreadinesslab.simulation.LabComparisonType
import com.retirementreadinesslab.simulation.LabSweepAnalysis
import com.retirementreadinesslab.simulation.LabSweepRowAnalysis
import com.retirementreadinesslab.simulation.LabSweepType
import com.retirementreadinesslab.simulation.RetirementDecisionEstimate
import com.retirementreadinesslab.simulation.ScenarioLabAnalyzer
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCompactCurrency
import com.retirementreadinesslab.ui.asPercent
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabCaution
import com.retirementreadinesslab.ui.theme.LabMutedText
import com.retirementreadinesslab.ui.theme.LabPrimary
import com.retirementreadinesslab.ui.theme.LabRisk
import com.retirementreadinesslab.ui.theme.LabSuccess

@Composable
fun LabScreen(state: RetirementLabState) {
    val analysis = state.selectedLabAnalysis
    val isPending = state.isAnalyzingLab && analysis == null

    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("lab-screen"),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp)
    ) {
        item {
            SectionHeader(
                title = "Scenario Lab",
                subtitle = "Quick estimates for the decisions most likely to move the plan."
            )
        }

        item {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Quick lab mode", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        "Each point uses ${ScenarioLabAnalyzer.QUICK_LAB_SIMULATIONS} simulations for direction. Run the main stress test after choosing a scenario.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = LabMutedText
                    )
                }
            }
        }

        if (isPending) {
            item {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
        }

        item {
            SectionHeader(
                title = "Target finders",
                subtitle = "Quick estimates for the two headline planning thresholds."
            )
        }

        item {
            TargetFinderCard(
                title = "Earliest retirement age",
                value = analysis?.decisionEstimate?.earliestRetirementAge?.let { "Age $it" }
                    ?: if (isPending) "Finding..." else "Unavailable",
                detail = earliestAgeDetail(analysis?.decisionEstimate, isPending),
                icon = Icons.Filled.Schedule
            )
        }

        item {
            TargetFinderCard(
                title = "Safe annual spending",
                value = analysis?.decisionEstimate?.safeAnnualSpending?.asCompactCurrency()
                    ?: if (isPending) "Finding..." else "Unavailable",
                detail = safeSpendingDetail(analysis?.decisionEstimate, isPending),
                icon = Icons.Filled.Savings
            )
        }

        analysis?.sweeps.orEmpty().forEach { sweep ->
            item {
                LabSweepCard(sweep)
            }
        }

        item {
            SectionHeader(
                title = "Strategy checks",
                subtitle = "Focused one-change tests against the active plan."
            )
        }

        if (analysis == null) {
            item {
                Text(
                    text = if (isPending) "Lab estimates are running in the background." else "Lab estimates are unavailable for the active scenario.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = LabMutedText
                )
            }
        } else {
            analysis.comparisons.forEach { comparison ->
                item {
                    LabComparisonCard(comparison)
                }
            }
        }
    }
}

@Composable
private fun TargetFinderCard(
    title: String,
    value: String,
    detail: String,
    icon: ImageVector
) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(icon, contentDescription = null, tint = LabPrimary)
            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(value, style = MaterialTheme.typography.headlineSmall, color = LabPrimary, fontWeight = FontWeight.SemiBold)
                Text(detail, style = MaterialTheme.typography.bodyMedium, color = LabMutedText)
            }
        }
    }
}

@Composable
private fun LabSweepCard(sweep: LabSweepAnalysis) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(sweep.type.icon(), contentDescription = null, tint = LabPrimary)
                Column {
                    Text(sweep.type.title(), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(sweep.type.subtitle(), style = MaterialTheme.typography.bodySmall, color = LabMutedText)
                }
            }
            sweep.rows.forEach { row ->
                SweepResultRow(row)
            }
            Text(sweep.takeaway, style = MaterialTheme.typography.bodyMedium, color = LabMutedText)
        }
    }
}

@Composable
private fun SweepResultRow(row: LabSweepRowAnalysis) {
    val color = readinessColor(row.readiness)
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(row.label, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                    if (row.isBase) {
                        Text("Current", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                    }
                }
                Text(row.detail, style = MaterialTheme.typography.labelSmall, color = LabMutedText)
            }
            Column(horizontalAlignment = Alignment.End) {
                Text(row.readiness.asPercent(), style = MaterialTheme.typography.labelLarge, color = color, fontWeight = FontWeight.SemiBold)
                Text(row.medianEndingBalance.asCompactCurrency(), style = MaterialTheme.typography.labelSmall, color = LabMutedText)
            }
        }
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .height(9.dp)
                    .testTag("lab-sweep-readiness-${row.label.accessibilityTagSuffix()}")
                    .semantics {
                        contentDescription = "Lab sweep ${row.label}: ${row.readiness.asPercent()} readiness, ${row.failureLabel}"
                        progressBarRangeInfo = ProgressBarRangeInfo(
                            current = row.readiness.toFloat().coerceIn(0f, 1f),
                            range = 0f..1f
                        )
                    }
                    .background(MaterialTheme.colorScheme.outline.copy(alpha = 0.25f), RoundedCornerShape(8.dp))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(row.readiness.toFloat().coerceIn(0f, 1f))
                        .height(9.dp)
                        .background(color, RoundedCornerShape(8.dp))
                )
            }
            Text(row.failureLabel, style = MaterialTheme.typography.labelSmall, color = LabMutedText, modifier = Modifier.width(76.dp))
        }
    }
}

@Composable
private fun LabComparisonCard(comparison: LabComparisonAnalysis) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(comparison.type.icon(), contentDescription = null, tint = LabPrimary)
                    Column {
                        Text(comparison.type.title(), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(comparison.type.subtitle(), style = MaterialTheme.typography.bodySmall, color = LabMutedText)
                    }
                }
                Text(signedPercent(comparison.readinessDelta), style = MaterialTheme.typography.titleMedium, color = deltaColor(comparison.readinessDelta))
            }
            Text(comparison.takeaway, style = MaterialTheme.typography.bodyMedium, color = LabMutedText)
        }
    }
}

private fun earliestAgeDetail(estimate: RetirementDecisionEstimate?, isPending: Boolean): String {
    if (estimate == null) return if (isPending) "Running quick target estimate." else "Quick target estimate could not be completed."
    return estimate.earliestRetirementReadiness?.let {
        "${it.asPercent()} readiness at the first age that clears ${estimate.targetReadiness.asPercent()}."
    } ?: "No age through 70 cleared ${estimate.targetReadiness.asPercent()} in the quick estimate."
}

private fun safeSpendingDetail(estimate: RetirementDecisionEstimate?, isPending: Boolean): String {
    if (estimate == null) return if (isPending) "Running quick target estimate." else "Quick target estimate could not be completed."
    return estimate.safeSpendingReadiness?.let {
        "${it.asPercent()} readiness at the current retirement age."
    } ?: "The target was not reached even at zero base spending because fixed costs still apply."
}

private fun readinessColor(probability: Double): Color {
    return when {
        probability >= 0.82 -> LabSuccess
        probability >= 0.65 -> LabCaution
        else -> LabRisk
    }
}

private fun deltaColor(delta: Double): Color {
    return when {
        delta > 0.01 -> LabSuccess
        delta < -0.01 -> LabRisk
        else -> LabPrimary
    }
}

private fun signedPercent(value: Double): String {
    val sign = if (value >= 0.0) "+" else ""
    return "$sign${value.asPercent()}"
}

private fun String.accessibilityTagSuffix(): String {
    return lowercase()
        .replace(Regex("[^a-z0-9]+"), "-")
        .trim('-')
}

private fun LabSweepType.title(): String {
    return when (this) {
        LabSweepType.RetirementAge -> "Retirement age sweep"
        LabSweepType.SpendingFlexibility -> "Spending flexibility"
        LabSweepType.SocialSecurityTiming -> "Social Security timing"
    }
}

private fun LabSweepType.subtitle(): String {
    return when (this) {
        LabSweepType.RetirementAge -> "Tests earlier and later start dates."
        LabSweepType.SpendingFlexibility -> "Tests small spending changes."
        LabSweepType.SocialSecurityTiming -> "Compares claim age tradeoffs."
    }
}

private fun LabSweepType.icon(): ImageVector {
    return when (this) {
        LabSweepType.RetirementAge -> Icons.Filled.Schedule
        LabSweepType.SpendingFlexibility -> Icons.Filled.Savings
        LabSweepType.SocialSecurityTiming -> Icons.Filled.Security
    }
}

private fun LabComparisonType.title(): String {
    return when (this) {
        LabComparisonType.RetireLater -> "Retire two years later"
        LabComparisonType.SpendLess -> "Spend 5% less"
        LabComparisonType.ClaimLater -> "Claim Social Security at 70"
        LabComparisonType.RothConversion -> "Enable Roth conversions"
        LabComparisonType.LongTermCare -> "Add long-term care risk"
        LabComparisonType.HealthcareInflation -> "Stress healthcare inflation"
        LabComparisonType.MarketDownturn -> "Stress lower returns"
        LabComparisonType.MortgagePayoff -> "Remove mortgage payment"
    }
}

private fun LabComparisonType.subtitle(): String {
    return when (this) {
        LabComparisonType.RetireLater -> "More compounding, fewer drawdown years"
        LabComparisonType.SpendLess -> "Lower annual base spending"
        LabComparisonType.ClaimLater -> "Higher benefit, longer bridge period"
        LabComparisonType.RothConversion -> "Fill the 22% bracket after retirement"
        LabComparisonType.LongTermCare -> "Late-life care shock"
        LabComparisonType.HealthcareInflation -> "Higher medical cost growth"
        LabComparisonType.MarketDownturn -> "Lower expected returns, higher volatility"
        LabComparisonType.MortgagePayoff -> "Tests retirement cash flow without mortgage payments"
    }
}

private fun LabComparisonType.icon(): ImageVector {
    return when (this) {
        LabComparisonType.RetireLater -> Icons.Filled.Schedule
        LabComparisonType.SpendLess -> Icons.Filled.Savings
        LabComparisonType.ClaimLater -> Icons.Filled.Security
        LabComparisonType.RothConversion -> Icons.Filled.Toll
        LabComparisonType.LongTermCare -> Icons.Filled.HealthAndSafety
        LabComparisonType.HealthcareInflation -> Icons.Filled.HealthAndSafety
        LabComparisonType.MarketDownturn -> Icons.AutoMirrored.Filled.TrendingDown
        LabComparisonType.MortgagePayoff -> Icons.Filled.Home
    }
}
