package com.retirementreadinesslab.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightScheme = lightColorScheme(
    primary = LabPrimary,
    onPrimary = LabSurface,
    secondary = LabAccent,
    background = LabBackground,
    onBackground = LabText,
    surface = LabSurface,
    onSurface = LabText,
    outline = LabDivider,
    error = LabRisk
)

private val DarkScheme = darkColorScheme(
    primary = LabAccent,
    onPrimary = LabPrimaryDark,
    secondary = LabPrimary,
    background = LabDarkBackground,
    onBackground = LabDarkText,
    surface = LabDarkSurface,
    onSurface = LabDarkText,
    outline = LabMutedText,
    error = LabRisk
)

@Composable
fun RetirementReadinessLabTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkScheme else LightScheme,
        content = content
    )
}
