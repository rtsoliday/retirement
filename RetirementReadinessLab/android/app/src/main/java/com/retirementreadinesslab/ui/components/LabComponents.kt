package com.retirementreadinesslab.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.retirementreadinesslab.model.OutcomeBand
import com.retirementreadinesslab.model.RiskLevel
import com.retirementreadinesslab.model.ScenarioWarning
import com.retirementreadinesslab.model.ScenarioWarningSeverity
import com.retirementreadinesslab.ui.asCompactCurrency
import com.retirementreadinesslab.ui.asPercent
import com.retirementreadinesslab.ui.theme.LabCaution
import com.retirementreadinesslab.ui.theme.LabDivider
import com.retirementreadinesslab.ui.theme.LabMutedText
import com.retirementreadinesslab.ui.theme.LabPrimary
import com.retirementreadinesslab.ui.theme.LabRisk
import com.retirementreadinesslab.ui.theme.LabSuccess
import kotlin.math.max
import kotlin.math.min

@Composable
fun ScreenColumn(content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.background)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        content = content
    )
}

@Composable
fun SectionHeader(title: String, subtitle: String? = null) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold
        )
        if (subtitle != null) {
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodyMedium,
                color = LabMutedText
            )
        }
    }
}

@Composable
fun MetricCard(
    title: String,
    value: String,
    detail: String,
    icon: ImageVector,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(title, style = MaterialTheme.typography.labelLarge, color = LabMutedText)
                Icon(icon, contentDescription = null, tint = LabPrimary, modifier = Modifier.size(20.dp))
            }
            Text(value, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
            Text(detail, style = MaterialTheme.typography.bodySmall, color = LabMutedText)
        }
    }
}

@Composable
fun ReadinessGauge(probability: Double, modifier: Modifier = Modifier) {
    val color = when {
        probability >= 0.82 -> LabSuccess
        probability >= 0.65 -> LabCaution
        else -> LabRisk
    }
    Box(
        modifier = modifier.semantics {
            contentDescription = "Readiness gauge: ${probability.asPercent()} readiness"
            progressBarRangeInfo = ProgressBarRangeInfo(
                current = probability.toFloat().coerceIn(0f, 1f),
                range = 0f..1f
            )
        },
        contentAlignment = Alignment.Center
    ) {
        Canvas(modifier = Modifier.size(156.dp)) {
            val stroke = Stroke(width = 18.dp.toPx(), cap = StrokeCap.Round)
            drawArc(
                color = LabDivider,
                startAngle = 150f,
                sweepAngle = 240f,
                useCenter = false,
                style = stroke,
                size = Size(size.width, size.height)
            )
            drawArc(
                color = color,
                startAngle = 150f,
                sweepAngle = (240f * probability).toFloat(),
                useCenter = false,
                style = stroke,
                size = Size(size.width, size.height)
            )
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(probability.asPercent(), style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
            Text("readiness", style = MaterialTheme.typography.labelMedium, color = LabMutedText)
        }
    }
}

@Composable
fun BalancePathChart(bands: List<OutcomeBand>, modifier: Modifier = Modifier) {
    if (bands.isEmpty()) return

    val minValue = min(0.0, bands.minOf { it.pessimistic })
    val maxValue = max(1.0, bands.maxOf { it.optimistic })
    val description = "Portfolio range chart from age ${bands.first().age} to age ${bands.last().age}. " +
        "Median ending balance ${bands.last().median.asCompactCurrency()}."

    Card(
        modifier = modifier
            .fillMaxWidth()
            .semantics { contentDescription = description },
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Portfolio range", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(180.dp)
            ) {
                fun x(index: Int): Float {
                    if (bands.size == 1) return 0f
                    return (index.toFloat() / (bands.size - 1).toFloat()) * size.width
                }

                fun y(value: Double): Float {
                    val normalized = ((value - minValue) / (maxValue - minValue)).coerceIn(0.0, 1.0)
                    return size.height - (normalized.toFloat() * size.height)
                }

                for (i in 0 until bands.lastIndex) {
                    drawLine(
                        color = LabDivider,
                        start = Offset(x(i), y(bands[i].optimistic)),
                        end = Offset(x(i + 1), y(bands[i + 1].optimistic)),
                        strokeWidth = 2.dp.toPx()
                    )
                    drawLine(
                        color = LabPrimary,
                        start = Offset(x(i), y(bands[i].median)),
                        end = Offset(x(i + 1), y(bands[i + 1].median)),
                        strokeWidth = 3.dp.toPx()
                    )
                    drawLine(
                        color = LabRisk.copy(alpha = 0.75f),
                        start = Offset(x(i), y(bands[i].pessimistic)),
                        end = Offset(x(i + 1), y(bands[i + 1].pessimistic)),
                        strokeWidth = 2.dp.toPx()
                    )
                }
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                ChartLegendItem(color = LabDivider, label = "90th percentile")
                ChartLegendItem(color = LabPrimary, label = "Median")
                ChartLegendItem(color = LabRisk.copy(alpha = 0.75f), label = "10th percentile")
            }
            Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                Text("Age ${bands.first().age}", style = MaterialTheme.typography.labelSmall, color = LabMutedText)
                Text(bands.last().median.asCompactCurrency(), style = MaterialTheme.typography.labelSmall, color = LabMutedText)
                Text("Age ${bands.last().age}", style = MaterialTheme.typography.labelSmall, color = LabMutedText)
            }
        }
    }
}

@Composable
private fun ChartLegendItem(color: Color, label: String) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(9.dp)
                .background(color, RoundedCornerShape(8.dp))
        )
        Text(label, style = MaterialTheme.typography.labelSmall, color = LabMutedText)
    }
}

@Composable
fun RiskPill(label: String, level: RiskLevel) {
    val color = when (level) {
        RiskLevel.Healthy -> LabSuccess
        RiskLevel.Watch -> LabCaution
        RiskLevel.AtRisk -> LabRisk
    }
    val text = when (level) {
        RiskLevel.Healthy -> "Healthy"
        RiskLevel.Watch -> "Watch"
        RiskLevel.AtRisk -> "At risk"
    }
    Row(
        modifier = Modifier
            .background(color.copy(alpha = 0.12f), RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 7.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .background(color, RoundedCornerShape(8.dp))
        )
        Text("$label: $text", style = MaterialTheme.typography.labelMedium, color = color)
    }
}

@Composable
fun KeyValueRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = LabMutedText)
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
    }
}

@Composable
fun ScenarioWarningCard(title: String, warnings: List<ScenarioWarning>, modifier: Modifier = Modifier) {
    if (warnings.isEmpty()) return

    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            warnings.take(5).forEach { warning ->
                ScenarioWarningRow(warning)
            }
            if (warnings.size > 5) {
                Text(
                    text = "+${warnings.size - 5} more warnings in assumptions",
                    style = MaterialTheme.typography.bodySmall,
                    color = LabMutedText
                )
            }
        }
    }
}

@Composable
private fun ScenarioWarningRow(warning: ScenarioWarning) {
    val color = when (warning.severity) {
        ScenarioWarningSeverity.Note -> LabPrimary
        ScenarioWarningSeverity.Watch -> LabCaution
    }
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.Top
    ) {
        Box(
            modifier = Modifier
                .padding(top = 6.dp)
                .size(8.dp)
                .background(color, RoundedCornerShape(8.dp))
        )
        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(warning.title, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
            Text(warning.detail, style = MaterialTheme.typography.bodySmall, color = LabMutedText)
        }
    }
}

@Composable
fun VerticalGap() {
    Spacer(modifier = Modifier.height(4.dp))
}
