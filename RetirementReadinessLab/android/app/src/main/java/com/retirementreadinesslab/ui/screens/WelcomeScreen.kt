package com.retirementreadinesslab.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.Backup
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Science
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.retirementreadinesslab.R
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabMutedText
import com.retirementreadinesslab.ui.theme.LabPrimary

@Composable
fun WelcomeScreen(
    state: RetirementLabState,
    onStartSetup: () -> Unit,
    onUseSamplePlans: () -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = PaddingValues(16.dp)
    ) {
        item {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Image(
                    painter = painterResource(R.drawable.retirement_readiness_icon),
                    contentDescription = null,
                    modifier = Modifier.size(82.dp)
                )
                Text(
                    text = "Retirement Readiness Lab",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Stress-test when you can retire.",
                    style = MaterialTheme.typography.titleMedium,
                    color = LabPrimary,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = "Compare retirement age, spending, Social Security, healthcare, taxes, market risk, and long-term care assumptions without creating an account.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = LabMutedText
                )
            }
        }

        item {
            WelcomeActionCard(
                state = state,
                onStartSetup = onStartSetup,
                onUseSamplePlans = onUseSamplePlans
            )
        }

        item {
            SectionHeader(title = "Built for private planning")
        }

        item {
            WelcomePointCard(
                title = "Local by default",
                detail = "Scenario data is stored on this device unless you choose to share or export it.",
                icon = Icons.Filled.Lock
            )
        }

        item {
            WelcomePointCard(
                title = "Decision lab",
                detail = "Run focused comparisons for the levers that can materially change a retirement plan.",
                icon = Icons.Filled.Science
            )
        }

        item {
            WelcomePointCard(
                title = "Portable reports",
                detail = "Export a summary, backup scenarios, or share a PDF when the assumptions are ready.",
                icon = Icons.Filled.Backup
            )
        }

        item {
            Text(
                text = "Educational estimates only. Retirement Readiness Lab is not financial, tax, legal, or investment advice.",
                style = MaterialTheme.typography.bodySmall,
                color = LabMutedText
            )
        }
    }
}

@Composable
private fun WelcomeActionCard(
    state: RetirementLabState,
    onStartSetup: () -> Unit,
    onUseSamplePlans: () -> Unit
) {
    Card(
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text("Start with your numbers", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(
                "The guided setup starts with the active sample plan, then saves your changes locally.",
                style = MaterialTheme.typography.bodyMedium,
                color = LabMutedText
            )
            Button(
                onClick = onStartSetup,
                enabled = !state.isLoading,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Start Setup")
                Spacer(modifier = Modifier.width(8.dp))
                Icon(Icons.AutoMirrored.Filled.ArrowForward, contentDescription = null)
            }
            OutlinedButton(
                onClick = onUseSamplePlans,
                enabled = !state.isLoading,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Use Sample Plans")
            }
        }
    }
}

@Composable
private fun WelcomePointCard(title: String, detail: String, icon: ImageVector) {
    Card(
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(icon, contentDescription = null, tint = LabPrimary)
            Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(detail, style = MaterialTheme.typography.bodyMedium, color = LabMutedText)
            }
        }
    }
}
