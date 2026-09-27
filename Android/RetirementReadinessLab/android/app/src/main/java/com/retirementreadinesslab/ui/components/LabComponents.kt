package com.retirementreadinesslab.ui.components

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.retirementreadinesslab.model.PortfolioSurvivalPoint
import com.retirementreadinesslab.model.RiskLevel
import com.retirementreadinesslab.model.ScenarioWarning
import com.retirementreadinesslab.model.ScenarioWarningSeverity
import com.retirementreadinesslab.model.SimulationMeanPoint
import com.retirementreadinesslab.model.SimulationPathPoint
import com.retirementreadinesslab.ui.asCompactCurrency
import com.retirementreadinesslab.ui.asPercent
import com.retirementreadinesslab.ui.theme.LabCaution
import com.retirementreadinesslab.ui.theme.LabDivider
import com.retirementreadinesslab.ui.theme.LabMutedText
import com.retirementreadinesslab.ui.theme.LabPrimary
import com.retirementreadinesslab.ui.theme.LabRisk
import com.retirementreadinesslab.ui.theme.LabSuccess
import kotlin.math.exp
import kotlin.math.ln

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
fun PortfolioSurvivalChart(points: List<PortfolioSurvivalPoint>, modifier: Modifier = Modifier) {
    if (points.isEmpty()) return

    var isExpanded by rememberSaveable { mutableStateOf(false) }
    val description = "Funding and survival by age chart from age ${points.first().age} to age ${points.last().age}. " +
        "${points.last().notFailedShare.asPercent()} remain not failed and " +
        "${points.last().aliveShare.asPercent()} remain alive at age ${points.last().age}."

    Card(
        modifier = modifier
            .fillMaxWidth()
            .testTag("portfolio-survival-chart")
            .clickable(onClickLabel = "Expand funding and survival chart") { isExpanded = true }
            .semantics { contentDescription = description },
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Funding and survival by age", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            PortfolioSurvivalPlot(
                points = points,
                interactive = false,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(190.dp)
            )
            PortfolioSurvivalLegend()
        }
    }

    if (isExpanded) {
        ExpandedPortfolioSurvivalDialog(
            points = points,
            onDismiss = { isExpanded = false }
        )
    }
}

@Composable
private fun ExpandedPortfolioSurvivalDialog(
    points: List<PortfolioSurvivalPoint>,
    onDismiss: () -> Unit
) {
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        "Funding and survival by age",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold
                    )
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Filled.Close, contentDescription = "Close")
                    }
                }
                PortfolioSurvivalPlot(
                    points = points,
                    interactive = true,
                    onDoubleTap = onDismiss,
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                )
                PortfolioSurvivalLegend()
            }
        }
    }
}

@Composable
private fun PortfolioSurvivalPlot(
    points: List<PortfolioSurvivalPoint>,
    interactive: Boolean,
    onDoubleTap: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    var canvasSize by remember { mutableStateOf(IntSize.Zero) }
    var zoomScale by rememberSaveable { mutableStateOf(1f) }
    var offsetX by rememberSaveable { mutableStateOf(0f) }
    var offsetY by rememberSaveable { mutableStateOf(0f) }
    var markerY by rememberSaveable { mutableStateOf<Float?>(null) }
    val density = LocalDensity.current
    val markerColor = MaterialTheme.colorScheme.primary
    val markerLabelBackground = MaterialTheme.colorScheme.surface
    val pointerModifier = if (interactive) {
        val leftInset = with(density) { PortfolioChartYAxisWidth.toPx() }
        val rightInset = with(density) { PortfolioChartRightPadding.toPx() }
        val topInset = with(density) { PortfolioChartTopPadding.toPx() }
        val bottomInset = with(density) { PortfolioChartBottomPadding.toPx() }
        Modifier.pointerInput(canvasSize) {
            detectTransformGestures { centroid, pan, zoom, _ ->
                val plotWidth = (canvasSize.width - leftInset - rightInset).coerceAtLeast(1f)
                val plotHeight = (canvasSize.height - topInset - bottomInset).coerceAtLeast(1f)
                val nextScale = (zoomScale * zoom).coerceIn(1f, 12f)
                val scaleChange = nextScale / zoomScale
                val centroidX = (centroid.x - leftInset).coerceIn(0f, plotWidth)
                val centroidY = (centroid.y - topInset).coerceIn(0f, plotHeight)
                val minOffsetX = -plotWidth * (nextScale - 1f)
                val minOffsetY = -plotHeight * (nextScale - 1f)

                offsetX = (((offsetX - centroidX) * scaleChange) + centroidX + pan.x)
                    .coerceIn(minOffsetX, 0f)
                offsetY = (((offsetY - centroidY) * scaleChange) + centroidY + pan.y)
                    .coerceIn(minOffsetY, 0f)
                zoomScale = nextScale
            }
        }
    } else {
        Modifier
    }
    val markerModifier = if (interactive) {
        val topInset = with(density) { PortfolioChartTopPadding.toPx() }
        val bottomInset = with(density) { PortfolioChartBottomPadding.toPx() }
        Modifier.pointerInput(canvasSize) {
            awaitPointerEventScope {
                while (true) {
                    val event = awaitPointerEvent(PointerEventPass.Initial)
                    val pressed = event.changes.filter { it.pressed }
                    if (pressed.size == 1 && canvasSize.height > 0) {
                        markerY = pressed.first().position.y
                            .coerceIn(topInset, canvasSize.height.toFloat() - bottomInset)
                        if (pressed.first().positionChanged()) {
                            pressed.forEach { it.consume() }
                        }
                    }
                }
            }
        }
    } else {
        Modifier
    }
    val doubleTapModifier = if (onDoubleTap != null) {
        Modifier.pointerInput(onDoubleTap) {
            detectTapGestures(onDoubleTap = { onDoubleTap() })
        }
    } else {
        Modifier
    }

    Canvas(
        modifier = modifier
            .onSizeChanged { canvasSize = it }
            .then(markerModifier)
            .then(pointerModifier)
            .then(doubleTapModifier)
    ) {
        val plotLeft = PortfolioChartYAxisWidth.toPx()
        val plotRight = size.width - PortfolioChartRightPadding.toPx()
        val plotTop = PortfolioChartTopPadding.toPx()
        val plotBottom = size.height - PortfolioChartBottomPadding.toPx()
        val plotWidth = (plotRight - plotLeft).coerceAtLeast(1f)
        val plotHeight = (plotBottom - plotTop).coerceAtLeast(1f)
        val axisColor = LabMutedText.copy(alpha = 0.78f)
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = LabMutedText.toArgb()
            textAlign = Paint.Align.RIGHT
            textSize = 10.dp.toPx()
        }
        val xAxisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = LabMutedText.toArgb()
            textAlign = Paint.Align.CENTER
            textSize = 10.dp.toPx()
        }
        val markerLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = markerColor.toArgb()
            textAlign = Paint.Align.RIGHT
            textSize = 10.dp.toPx()
            isFakeBoldText = true
        }

        val ageRange = (points.last().age - points.first().age).coerceAtLeast(1)

        fun baseXForAge(age: Int): Float {
            return ((age - points.first().age).toFloat() / ageRange.toFloat()) * plotWidth
        }

        fun baseY(value: Double): Float {
            val normalized = value.coerceIn(0.0, 1.0)
            return (1f - normalized.toFloat()) * plotHeight
        }

        fun xForAge(age: Int): Float = plotLeft + (baseXForAge(age) * zoomScale) + offsetX

        fun y(value: Double): Float = plotTop + (baseY(value) * zoomScale) + offsetY

        fun percentAt(screenY: Float): Double {
            val baseY = ((screenY - plotTop - offsetY) / zoomScale).coerceIn(0f, plotHeight)
            return (1f - (baseY / plotHeight)).toDouble().coerceIn(0.0, 1.0)
        }

        repeat(PortfolioChartTickCount) { index ->
            val fraction = if (PortfolioChartTickCount == 1) 0f else index.toFloat() / (PortfolioChartTickCount - 1)
            val tickY = plotBottom - (fraction * plotHeight)
            drawLine(
                color = LabDivider.copy(alpha = 0.7f),
                start = Offset(plotLeft, tickY),
                end = Offset(plotRight, tickY),
                strokeWidth = 1.dp.toPx()
            )
            drawContext.canvas.nativeCanvas.drawText(
                percentAt(tickY).asPercent(),
                plotLeft - 7.dp.toPx(),
                tickY + 3.dp.toPx(),
                labelPaint
            )
        }

        clipRect(left = plotLeft, top = plotTop, right = plotRight, bottom = plotBottom) {
            for (i in 0 until points.lastIndex) {
                drawLine(
                    color = LabPrimary,
                    start = Offset(xForAge(points[i].age), y(points[i].notFailedShare)),
                    end = Offset(xForAge(points[i + 1].age), y(points[i + 1].notFailedShare)),
                    strokeWidth = 3.dp.toPx(),
                    cap = StrokeCap.Round
                )
            }

            for (i in 0 until points.lastIndex) {
                drawLine(
                    color = LabCaution,
                    start = Offset(xForAge(points[i].age), y(points[i].aliveShare)),
                    end = Offset(xForAge(points[i + 1].age), y(points[i + 1].aliveShare)),
                    strokeWidth = 3.dp.toPx(),
                    cap = StrokeCap.Round
                )
            }
        }

        drawLine(
            color = axisColor,
            start = Offset(plotLeft, plotTop),
            end = Offset(plotLeft, plotBottom),
            strokeWidth = 1.2.dp.toPx()
        )
        drawLine(
            color = axisColor,
            start = Offset(plotLeft, plotBottom),
            end = Offset(plotRight, plotBottom),
            strokeWidth = 1.2.dp.toPx()
        )

        ageTicks(points.first().age, points.last().age).forEach { tickAge ->
            val tickX = xForAge(tickAge)
            if (tickX < plotLeft || tickX > plotRight) return@forEach
            drawLine(
                color = axisColor,
                start = Offset(tickX, plotBottom),
                end = Offset(tickX, plotBottom + 4.dp.toPx()),
                strokeWidth = 1.dp.toPx()
            )
            drawContext.canvas.nativeCanvas.drawText(
                "$tickAge",
                tickX,
                plotBottom + 17.dp.toPx(),
                xAxisPaint
            )
        }

        if (interactive && markerY != null) {
            val y = markerY!!.coerceIn(plotTop, plotBottom)
            drawHorizontalMarker(
                markerY = y,
                label = percentAt(y).asPercent(),
                plotLeft = plotLeft,
                plotRight = plotRight,
                plotTop = plotTop,
                plotBottom = plotBottom,
                lineColor = markerColor,
                labelBackground = markerLabelBackground,
                labelPaint = markerLabelPaint
            )
        }
    }
}

@Composable
private fun PortfolioSurvivalLegend() {
    Row(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth()
    ) {
        ChartLegendItem(color = LabPrimary, label = "Still funded")
        ChartLegendItem(color = LabCaution, label = "Still alive")
    }
}

@Composable
fun SimulationPathsChart(
    points: List<SimulationPathPoint>,
    meanPath: List<SimulationMeanPoint>,
    retirementAge: Int,
    modifier: Modifier = Modifier
) {
    if (points.isEmpty() && meanPath.isEmpty()) return

    var isExpanded by rememberSaveable { mutableStateOf(false) }
    val bounds = remember(points, meanPath) { simulationPathBounds(points, meanPath) }
    val colors = simulationPathColors()
    val description = "Simulation paths scatter plot with ${points.size} sampled path points and " +
        "${meanPath.size} mean path points from age $retirementAge to age ${retirementAge + bounds.maxYear}."

    Card(
        modifier = modifier
            .fillMaxWidth()
            .testTag("simulation-paths-chart")
            .clickable(onClickLabel = "Expand simulation paths") { isExpanded = true }
            .semantics { contentDescription = description },
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Simulation paths", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            SimulationPathsPlot(
                points = points,
                meanPath = meanPath,
                bounds = bounds,
                colors = colors,
                retirementAge = retirementAge,
                interactive = false,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(190.dp)
            )
            SimulationPathsLegend(colors = colors, includeCertainOutcomes = false)
            Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                Text("Age $retirementAge", style = MaterialTheme.typography.labelSmall, color = LabMutedText)
                Text("Log scale", style = MaterialTheme.typography.labelSmall, color = LabMutedText)
                Text("Age ${retirementAge + bounds.maxYear}", style = MaterialTheme.typography.labelSmall, color = LabMutedText)
            }
        }
    }

    if (isExpanded) {
        ExpandedSimulationPathsDialog(
            points = points,
            meanPath = meanPath,
            bounds = bounds,
            colors = colors,
            retirementAge = retirementAge,
            onDismiss = { isExpanded = false }
        )
    }
}

@Composable
private fun ExpandedSimulationPathsDialog(
    points: List<SimulationPathPoint>,
    meanPath: List<SimulationMeanPoint>,
    bounds: SimulationPathBounds,
    colors: SimulationPathColors,
    retirementAge: Int,
    onDismiss: () -> Unit
) {
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        "Simulation paths",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold
                    )
                    IconButton(onClick = onDismiss) {
                        Icon(Icons.Filled.Close, contentDescription = "Close")
                    }
                }
                SimulationPathsPlot(
                    points = points,
                    meanPath = meanPath,
                    bounds = bounds,
                    colors = colors,
                    retirementAge = retirementAge,
                    interactive = true,
                    onDoubleTap = onDismiss,
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                )
                SimulationPathsLegend(colors = colors, includeCertainOutcomes = true)
                Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                    Text("Age $retirementAge", style = MaterialTheme.typography.labelSmall, color = LabMutedText)
                    Text("Log scale", style = MaterialTheme.typography.labelSmall, color = LabMutedText)
                    Text("Age ${retirementAge + bounds.maxYear}", style = MaterialTheme.typography.labelSmall, color = LabMutedText)
                }
            }
        }
    }
}

@Composable
private fun SimulationPathsPlot(
    points: List<SimulationPathPoint>,
    meanPath: List<SimulationMeanPoint>,
    bounds: SimulationPathBounds,
    colors: SimulationPathColors,
    retirementAge: Int,
    interactive: Boolean,
    onDoubleTap: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    var canvasSize by remember { mutableStateOf(IntSize.Zero) }
    var zoomScale by rememberSaveable { mutableStateOf(1f) }
    var offsetX by rememberSaveable { mutableStateOf(0f) }
    var offsetY by rememberSaveable { mutableStateOf(0f) }
    var markerY by rememberSaveable { mutableStateOf<Float?>(null) }
    val density = LocalDensity.current
    val markerColor = MaterialTheme.colorScheme.primary
    val markerLabelBackground = MaterialTheme.colorScheme.surface
    val pointerModifier = if (interactive) {
        val leftInset = with(density) { SimulationChartYAxisWidth.toPx() }
        val rightInset = with(density) { SimulationChartRightPadding.toPx() }
        val topInset = with(density) { SimulationChartTopPadding.toPx() }
        val bottomInset = with(density) { SimulationChartBottomPadding.toPx() }
        Modifier.pointerInput(canvasSize) {
            detectTransformGestures { centroid, pan, zoom, _ ->
                val plotWidth = (canvasSize.width - leftInset - rightInset).coerceAtLeast(1f)
                val plotHeight = (canvasSize.height - topInset - bottomInset).coerceAtLeast(1f)
                val nextScale = (zoomScale * zoom).coerceIn(1f, 12f)
                val scaleChange = nextScale / zoomScale
                val centroidX = (centroid.x - leftInset).coerceIn(0f, plotWidth)
                val centroidY = (centroid.y - topInset).coerceIn(0f, plotHeight)
                val minOffsetX = -plotWidth * (nextScale - 1f)
                val minOffsetY = -plotHeight * (nextScale - 1f)

                offsetX = (((offsetX - centroidX) * scaleChange) + centroidX + pan.x)
                    .coerceIn(minOffsetX, 0f)
                offsetY = (((offsetY - centroidY) * scaleChange) + centroidY + pan.y)
                    .coerceIn(minOffsetY, 0f)
                zoomScale = nextScale
            }
        }
    } else {
        Modifier
    }
    val markerModifier = if (interactive) {
        val topInset = with(density) { SimulationChartTopPadding.toPx() }
        val bottomInset = with(density) { SimulationChartBottomPadding.toPx() }
        Modifier.pointerInput(canvasSize) {
            awaitPointerEventScope {
                while (true) {
                    val event = awaitPointerEvent(PointerEventPass.Initial)
                    val pressed = event.changes.filter { it.pressed }
                    if (pressed.size == 1 && canvasSize.height > 0) {
                        markerY = pressed.first().position.y
                            .coerceIn(topInset, canvasSize.height.toFloat() - bottomInset)
                        if (pressed.first().positionChanged()) {
                            pressed.forEach { it.consume() }
                        }
                    }
                }
            }
        }
    } else {
        Modifier
    }
    val doubleTapModifier = if (onDoubleTap != null) {
        Modifier.pointerInput(onDoubleTap) {
            detectTapGestures(onDoubleTap = { onDoubleTap() })
        }
    } else {
        Modifier
    }

    Canvas(
        modifier = modifier
            .onSizeChanged { canvasSize = it }
            .then(markerModifier)
            .then(pointerModifier)
            .then(doubleTapModifier)
    ) {
        val plotLeft = SimulationChartYAxisWidth.toPx()
        val plotRight = size.width - SimulationChartRightPadding.toPx()
        val plotTop = SimulationChartTopPadding.toPx()
        val plotBottom = size.height - SimulationChartBottomPadding.toPx()
        val plotWidth = (plotRight - plotLeft).coerceAtLeast(1f)
        val plotHeight = (plotBottom - plotTop).coerceAtLeast(1f)
        val axisColor = LabMutedText.copy(alpha = 0.78f)
        val labelColor = LabMutedText.toArgb()
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = labelColor
            textAlign = Paint.Align.RIGHT
            textSize = 10.dp.toPx()
        }
        val xAxisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = labelColor
            textAlign = Paint.Align.CENTER
            textSize = 10.dp.toPx()
        }
        val markerLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = markerColor.toArgb()
            textAlign = Paint.Align.RIGHT
            textSize = 10.dp.toPx()
            isFakeBoldText = true
        }

        fun baseX(yearsInRetirement: Int): Float {
            return (yearsInRetirement.toFloat() / bounds.maxYear.toFloat()) * plotWidth
        }

        fun baseY(balance: Double): Float {
            val normalized = ((ln(balance.coerceAtLeast(1.0)) - bounds.logMin) / bounds.logRange)
                .coerceIn(0.0, 1.0)
            return (1f - normalized.toFloat()) * plotHeight
        }

        fun x(yearsInRetirement: Int): Float = plotLeft + (baseX(yearsInRetirement) * zoomScale) + offsetX

        fun y(balance: Double): Float = plotTop + (baseY(balance) * zoomScale) + offsetY

        fun balanceAt(screenY: Float): Double {
            val baseY = ((screenY - plotTop - offsetY) / zoomScale).coerceIn(0f, plotHeight)
            val normalized = 1f - (baseY / plotHeight)
            return exp(bounds.logMin + (bounds.logRange * normalized))
        }

        repeat(SimulationChartTickCount) { index ->
            val fraction = if (SimulationChartTickCount == 1) 0f else index.toFloat() / (SimulationChartTickCount - 1)
            val tickY = plotBottom - (fraction * plotHeight)
            drawLine(
                color = LabDivider.copy(alpha = 0.7f),
                start = Offset(plotLeft, tickY),
                end = Offset(plotRight, tickY),
                strokeWidth = 1.dp.toPx()
            )
            drawContext.canvas.nativeCanvas.drawText(
                balanceAt(tickY).asCompactCurrency(),
                plotLeft - 7.dp.toPx(),
                tickY + 3.dp.toPx(),
                labelPaint
            )
        }

        clipRect(left = plotLeft, top = plotTop, right = plotRight, bottom = plotBottom) {
            val radius = if (interactive) 1.45.dp.toPx() else 1.15.dp.toPx()
            val successPointOffset = SimulationSuccessPointOffset.toPx()
            val failurePointOffset = SimulationFailurePointOffset.toPx()
            points.forEach { point ->
                val pointOffset = if (point.successfulPath) successPointOffset else failurePointOffset
                drawCircle(
                    color = colors.colorFor(point).copy(alpha = 0.62f),
                    radius = radius,
                    center = Offset(x(point.yearsInRetirement) + pointOffset, y(point.balance))
                )
            }

            for (i in 0 until meanPath.lastIndex) {
                drawLine(
                    color = colors.mean,
                    start = Offset(x(meanPath[i].yearsInRetirement), y(meanPath[i].balance)),
                    end = Offset(x(meanPath[i + 1].yearsInRetirement), y(meanPath[i + 1].balance)),
                    strokeWidth = if (interactive) 2.dp.toPx() else 1.6.dp.toPx()
                )
            }
        }

        drawLine(
            color = axisColor,
            start = Offset(plotLeft, plotTop),
            end = Offset(plotLeft, plotBottom),
            strokeWidth = 1.2.dp.toPx()
        )
        drawLine(
            color = axisColor,
            start = Offset(plotLeft, plotBottom),
            end = Offset(plotRight, plotBottom),
            strokeWidth = 1.2.dp.toPx()
        )

        ageTicks(retirementAge, retirementAge + bounds.maxYear).forEach { tickAge ->
            val tickX = x(tickAge - retirementAge)
            if (tickX < plotLeft || tickX > plotRight) return@forEach
            drawLine(
                color = axisColor,
                start = Offset(tickX, plotBottom),
                end = Offset(tickX, plotBottom + 4.dp.toPx()),
                strokeWidth = 1.dp.toPx()
            )
            drawContext.canvas.nativeCanvas.drawText(
                "$tickAge",
                tickX,
                plotBottom + 17.dp.toPx(),
                xAxisPaint
            )
        }

        if (interactive && markerY != null) {
            val y = markerY!!.coerceIn(plotTop, plotBottom)
            drawHorizontalMarker(
                markerY = y,
                label = balanceAt(y).asCompactCurrency(),
                plotLeft = plotLeft,
                plotRight = plotRight,
                plotTop = plotTop,
                plotBottom = plotBottom,
                lineColor = markerColor,
                labelBackground = markerLabelBackground,
                labelPaint = markerLabelPaint
            )
        }
    }
}

@Composable
private fun SimulationPathsLegend(colors: SimulationPathColors, includeCertainOutcomes: Boolean) {
    if (includeCertainOutcomes) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                ChartLegendItem(color = colors.strongSuccess, label = "100% Success")
                ChartLegendItem(color = colors.success, label = "Success")
                ChartLegendItem(color = colors.mean, label = "Mean")
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                ChartLegendItem(color = colors.failure, label = "Failure")
                ChartLegendItem(color = colors.clearFailure, label = "100% Failure")
            }
        }
    } else {
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            ChartLegendItem(color = colors.success, label = "Success")
            ChartLegendItem(color = colors.failure, label = "Failure")
            ChartLegendItem(color = colors.mean, label = "Mean")
        }
    }
}

@Composable
private fun simulationPathColors(): SimulationPathColors {
    return SimulationPathColors(
        strongSuccess = Color(0xFF38D66B),
        success = LabSuccess,
        clearFailure = Color(0xFFE53935),
        failure = Color(0xFF7F1D1D),
        mean = MaterialTheme.colorScheme.onSurface
    )
}

private fun simulationPathBounds(
    points: List<SimulationPathPoint>,
    meanPath: List<SimulationMeanPoint>
): SimulationPathBounds {
    val positivePointBalances = points.map { it.balance }.filter { it > 0.0 }
    val positiveMeanBalances = meanPath.map { it.balance }.filter { it > 0.0 }
    val minValue = (positivePointBalances + positiveMeanBalances).minOrNull()?.coerceAtLeast(1.0) ?: 1.0
    val maxValue = (positivePointBalances + positiveMeanBalances).maxOrNull()?.coerceAtLeast(minValue * 10.0) ?: 10.0
    val maxYear = maxOf(
        points.maxOfOrNull { it.yearsInRetirement } ?: 0,
        meanPath.maxOfOrNull { it.yearsInRetirement } ?: 0
    ).coerceAtLeast(1)
    val logMin = ln(minValue)
    val logMax = ln(maxValue).takeIf { it > logMin } ?: (logMin + 1.0)

    return SimulationPathBounds(
        maxYear = maxYear,
        logMin = logMin,
        logMax = logMax
    )
}

private fun SimulationPathColors.colorFor(point: SimulationPathPoint): Color {
    return when {
        point.successfulPath && point.separatedFromOppositeOutcome -> strongSuccess
        point.successfulPath -> success
        point.separatedFromOppositeOutcome -> clearFailure
        else -> failure
    }
}

private data class SimulationPathBounds(
    val maxYear: Int,
    val logMin: Double,
    val logMax: Double
) {
    val logRange: Double = (logMax - logMin).takeIf { it > 0.0 } ?: 1.0
}

private data class SimulationPathColors(
    val strongSuccess: Color,
    val success: Color,
    val clearFailure: Color,
    val failure: Color,
    val mean: Color
)

private val SimulationChartYAxisWidth = 62.dp
private val SimulationChartRightPadding = 8.dp
private val SimulationChartTopPadding = 10.dp
private val SimulationChartBottomPadding = 30.dp
private val SimulationSuccessPointOffset = 2.dp
private val SimulationFailurePointOffset = 5.dp
private const val SimulationChartTickCount = 5
private val PortfolioChartYAxisWidth = 44.dp
private val PortfolioChartRightPadding = 6.dp
private val PortfolioChartTopPadding = 8.dp
private val PortfolioChartBottomPadding = 30.dp
private const val PortfolioChartTickCount = 5
private const val ChartAgeTickInterval = 5

private fun DrawScope.drawHorizontalMarker(
    markerY: Float,
    label: String,
    plotLeft: Float,
    plotRight: Float,
    plotTop: Float,
    plotBottom: Float,
    lineColor: Color,
    labelBackground: Color,
    labelPaint: Paint
) {
    val y = markerY.coerceIn(plotTop, plotBottom)
    val labelHeight = 18.dp.toPx()
    val labelTop = (y - (labelHeight / 2f)).coerceIn(0f, size.height - labelHeight)
    drawLine(
        color = lineColor.copy(alpha = 0.88f),
        start = Offset(plotLeft, y),
        end = Offset(plotRight, y),
        strokeWidth = 1.4.dp.toPx()
    )
    drawRect(
        color = labelBackground.copy(alpha = 0.96f),
        topLeft = Offset(0f, labelTop),
        size = Size((plotLeft - 4.dp.toPx()).coerceAtLeast(1f), labelHeight)
    )
    drawLine(
        color = lineColor.copy(alpha = 0.88f),
        start = Offset(plotLeft - 5.dp.toPx(), y),
        end = Offset(plotLeft, y),
        strokeWidth = 1.4.dp.toPx()
    )
    drawContext.canvas.nativeCanvas.drawText(
        label,
        plotLeft - 7.dp.toPx(),
        labelTop + 12.5.dp.toPx(),
        labelPaint
    )
}

private fun ageTicks(startAge: Int, endAge: Int): List<Int> {
    val ticks = mutableListOf<Int>()
    var age = startAge
    while (age <= endAge) {
        ticks += age
        age += ChartAgeTickInterval
    }
    return ticks.ifEmpty { listOf(startAge) }
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
    val color = level.color()
    val text = level.label()
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
fun RiskScanRow(label: String, level: RiskLevel, description: String) {
    val color = level.color()
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.Top
    ) {
        Box(
            modifier = Modifier
                .padding(top = 5.dp)
                .size(8.dp)
                .background(color, RoundedCornerShape(8.dp))
        )
        Column(verticalArrangement = Arrangement.spacedBy(2.dp), modifier = Modifier.weight(1f)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    label,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f)
                )
                Text(level.label(), style = MaterialTheme.typography.labelMedium, color = color)
            }
            Text(description, style = MaterialTheme.typography.bodySmall, color = LabMutedText)
        }
    }
}

private fun RiskLevel.color(): Color = when (this) {
    RiskLevel.Healthy -> LabSuccess
    RiskLevel.Watch -> LabCaution
    RiskLevel.AtRisk -> LabRisk
}

private fun RiskLevel.label(): String = when (this) {
    RiskLevel.Healthy -> "Healthy"
    RiskLevel.Watch -> "Watch"
    RiskLevel.AtRisk -> "At risk"
}

@Composable
fun KeyValueRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            color = LabMutedText,
            modifier = Modifier.weight(0.45f)
        )
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
            textAlign = TextAlign.End,
            modifier = Modifier.weight(0.55f)
        )
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
                    text = "+${warnings.size - 5} more warnings in setup",
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
