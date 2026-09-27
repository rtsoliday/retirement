package com.retirementreadinesslab.ui.screens

import android.content.ClipData
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Backup
import androidx.compose.material.icons.filled.DeleteForever
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.LockOpen
import androidx.compose.material.icons.filled.PictureAsPdf
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.retirementreadinesslab.compliance.ComplianceText
import com.retirementreadinesslab.data.ScenarioJson
import com.retirementreadinesslab.model.forFeatureAccess
import com.retirementreadinesslab.reports.ReportBuilder
import com.retirementreadinesslab.reports.ReportPdfExporter
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCompactCurrency
import com.retirementreadinesslab.ui.asPercent
import com.retirementreadinesslab.ui.components.KeyValueRow
import com.retirementreadinesslab.ui.components.ProLockedInlineNotice
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.findActivity
import com.retirementreadinesslab.ui.theme.LabMutedText

private const val SHOW_SCENARIO_BACKUP_TOOLS = false

@Composable
fun ReportsScreen(state: RetirementLabState) {
    val context = LocalContext.current
    val activity = context.findActivity()
    val scenario = state.selectedScenario.forFeatureAccess(state.featureAccess)
    val result = state.selectedResult
    val report = ReportBuilder.buildTextReport(scenario, result)
    val scenarioBackup = ScenarioJson.encodeScenarios(state.scenarios.toList())
    var importText by remember { mutableStateOf("") }
    var importMessage by remember { mutableStateOf<String?>(null) }
    var exportMessage by remember { mutableStateOf<String?>(null) }
    var promoCodeText by remember { mutableStateOf("") }
    var confirmDeleteData by remember { mutableStateOf(false) }

    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("reports-screen"),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp)
    ) {
        item {
            SectionHeader(
                title = "Reports",
                subtitle = "Prepare a shareable summary with assumptions and educational disclaimers."
            )
        }

        item {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Privacy and disclosures", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        text = ComplianceText.localDataNotice,
                        style = MaterialTheme.typography.bodyMedium,
                        color = LabMutedText
                    )
                    ComplianceText.disclosureBullets.forEach { disclosure ->
                        Text(
                            text = "- $disclosure",
                            style = MaterialTheme.typography.bodySmall,
                            color = LabMutedText
                        )
                    }
                }
            }
        }

        item {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Report preview", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    KeyValueRow("Retirement age", scenario.household.retirementAge.toString())
                    KeyValueRow("Readiness", result?.successProbability?.asPercent() ?: "Not run")
                    KeyValueRow("Median ending", result?.medianEndingBalance?.asCompactCurrency() ?: "Not run")
                    KeyValueRow("Primary risk", result?.riskBreakdown?.primaryRisk ?: "Not run")
                }
            }
        }

        item {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Share and backup", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        ComplianceText.exportNotice,
                        style = MaterialTheme.typography.bodyMedium,
                        color = LabMutedText
                    )
                    if (!state.isProUnlocked) {
                        ProLockedInlineNotice(
                            title = "Reports and exports are Pro",
                            detail = "Unlock Pro to share PDF reports and full text reports. JSON backup/import stays hidden for the first release."
                        )
                        if (state.supportsUserPurchases) {
                            Button(
                                onClick = { state.purchasePro(activity) },
                                enabled = !state.isPurchasingPro,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("unlock-pro-button")
                            ) {
                                Icon(Icons.Filled.LockOpen, contentDescription = null)
                                Text(if (state.isPurchasingPro) "Opening Google Play..." else "Unlock Pro")
                            }
                            OutlinedButton(
                                onClick = state::restoreProPurchase,
                                enabled = !state.isRestoringPro,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("restore-pro-purchase-button")
                            ) {
                                Icon(Icons.Filled.LockOpen, contentDescription = null)
                                Text(if (state.isRestoringPro) "Checking Google Play..." else "Restore Purchase")
                            }
                            Text(
                                "For Google review access, enter an official Google Play promo code for Pro Unlock. After redeeming it in Google Play, return here and tap Restore Purchase.",
                                style = MaterialTheme.typography.bodySmall,
                                color = LabMutedText
                            )
                            OutlinedTextField(
                                value = promoCodeText,
                                onValueChange = { promoCodeText = it },
                                label = { Text("Google Play promo code") },
                                singleLine = true,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("pro-promo-code-input")
                            )
                            OutlinedButton(
                                onClick = { state.unlockProWithPromoCode(activity, promoCodeText) },
                                enabled = promoCodeText.isNotBlank(),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("unlock-pro-with-promo-code-button")
                            ) {
                                Icon(Icons.Filled.LockOpen, contentDescription = null)
                                Text("Redeem Google Play Promo Code")
                            }
                        }
                    }
                    Button(
                        onClick = {
                            val exported = runCatching {
                                ReportPdfExporter.createReportPdf(context, scenario, result)
                            }
                            exported.onSuccess { file ->
                                exportMessage = null
                                context.shareFile(
                                    title = "Retirement Readiness Lab PDF report",
                                    uri = FileProvider.getUriForFile(
                                        context,
                                        "${context.packageName}.fileprovider",
                                        file
                                    ),
                                    mimeType = "application/pdf"
                                )
                            }.onFailure {
                                exportMessage = "PDF report could not be created: ${it.message ?: "unknown error"}"
                            }
                        },
                        enabled = state.isProUnlocked,
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("share-pdf-report-button")
                    ) {
                        Icon(Icons.Filled.PictureAsPdf, contentDescription = null)
                        Text(if (state.isProUnlocked) "Share PDF Report" else "Share PDF Report (Pro)")
                    }
                    OutlinedButton(
                        onClick = {
                            val exported = runCatching {
                                ReportPdfExporter.createReportPdf(context, scenario, result)
                            }
                            exported.onSuccess { file ->
                                val opened = runCatching {
                                    context.viewFile(
                                        uri = FileProvider.getUriForFile(
                                            context,
                                            "${context.packageName}.fileprovider",
                                            file
                                        ),
                                        mimeType = "application/pdf"
                                    )
                                }
                                opened.onSuccess {
                                    exportMessage = null
                                }.onFailure {
                                    exportMessage = "PDF report could not be opened: ${it.message ?: "no PDF viewer found"}"
                                }
                            }.onFailure {
                                exportMessage = "PDF report could not be created: ${it.message ?: "unknown error"}"
                            }
                        },
                        enabled = state.isProUnlocked,
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("view-pdf-report-button")
                    ) {
                        Icon(Icons.Filled.PictureAsPdf, contentDescription = null)
                        Text(if (state.isProUnlocked) "View PDF Report" else "View PDF Report (Pro)")
                    }
                    Button(
                        onClick = {
                            context.shareText(
                                title = "Retirement Readiness Lab report",
                                text = report
                            )
                        },
                        enabled = state.isProUnlocked,
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("share-text-report-button")
                    ) {
                        Icon(Icons.Filled.Share, contentDescription = null)
                        Text(if (state.isProUnlocked) "Share Report" else "Share Report (Pro)")
                    }
                    if (SHOW_SCENARIO_BACKUP_TOOLS) {
                        OutlinedButton(
                            onClick = {
                                context.shareText(
                                    title = "Retirement Readiness Lab scenario backup",
                                    text = scenarioBackup
                                )
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("share-scenario-backup-button")
                        ) {
                            Icon(Icons.Filled.Backup, contentDescription = null)
                            Text("Share Scenario Backup")
                        }
                    }
                    exportMessage?.let { message ->
                        Text(
                            text = message,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }
        }

        if (SHOW_SCENARIO_BACKUP_TOOLS) {
            item {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                    Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("Restore from backup", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(
                            "Paste JSON backup text exported from this app. Import replaces saved scenarios on this device.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = LabMutedText
                        )
                        OutlinedTextField(
                            value = importText,
                            onValueChange = {
                                importText = it
                                importMessage = null
                            },
                            label = { Text("JSON backup") },
                            minLines = 4,
                            maxLines = 8,
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("json-backup-input")
                        )
                        Button(
                            onClick = {
                                val error = state.importScenarioBackup(importText)
                                if (error == null) {
                                    importText = ""
                                    importMessage = "Backup imported and saved locally."
                                } else {
                                    importMessage = error
                                }
                            },
                            enabled = importText.isNotBlank(),
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("import-backup-button")
                        ) {
                            Icon(Icons.Filled.UploadFile, contentDescription = null)
                            Text("Import Backup")
                        }
                        importMessage?.let { message ->
                            Text(
                                text = message,
                                style = MaterialTheme.typography.bodySmall,
                                color = if (message.startsWith("Backup imported")) {
                                    MaterialTheme.colorScheme.primary
                                } else {
                                    MaterialTheme.colorScheme.error
                                }
                            )
                        }
                    }
                }
            }
        }

        if (state.hasDeveloperEntitlementControls) {
            item {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                    Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("Developer entitlement", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(
                            "Fake entitlement controls for debug testing. These controls are not available in release builds.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = LabMutedText
                        )
                        KeyValueRow("Provider", state.entitlementProviderName)
                        KeyValueRow("Current access", if (state.isProUnlocked) "Pro" else "Free")
                        OutlinedTextField(
                            value = promoCodeText,
                            onValueChange = { promoCodeText = it },
                            label = { Text("Debug promo code") },
                            singleLine = true,
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("debug-pro-promo-code-input")
                        )
                        Button(
                            onClick = { state.unlockProWithPromoCode(activity, promoCodeText) },
                            enabled = promoCodeText.isNotBlank() && !state.isProUnlocked,
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("debug-unlock-pro-with-promo-code-button")
                        ) {
                            Icon(Icons.Filled.LockOpen, contentDescription = null)
                            Text("Unlock Debug Pro with Promo Code")
                        }
                        Button(
                            onClick = { state.setDeveloperProUnlocked(true) },
                            enabled = !state.isProUnlocked,
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("debug-unlock-pro-button")
                        ) {
                            Icon(Icons.Filled.LockOpen, contentDescription = null)
                            Text("Unlock Pro For Testing")
                        }
                        OutlinedButton(
                            onClick = { state.setDeveloperProUnlocked(false) },
                            enabled = state.isProUnlocked,
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("debug-reset-free-button")
                        ) {
                            Icon(Icons.Filled.Lock, contentDescription = null)
                            Text("Reset To Free")
                        }
                    }
                }
            }
        }

        item {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("Local data", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        ComplianceText.localDataNotice,
                        style = MaterialTheme.typography.bodyMedium,
                        color = LabMutedText
                    )
                    Text(
                        ComplianceText.deletionNotice,
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    state.storageMessage?.let { message ->
                        Text(
                            text = message,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    if (confirmDeleteData) {
                        Button(
                            onClick = {
                                state.deleteLocalData()
                                importText = ""
                                importMessage = null
                                exportMessage = null
                                confirmDeleteData = false
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("confirm-delete-local-data-button")
                        ) {
                            Icon(Icons.Filled.DeleteForever, contentDescription = null)
                            Text("Confirm Delete Local Data")
                        }
                        OutlinedButton(
                            onClick = { confirmDeleteData = false },
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("cancel-delete-local-data-button")
                        ) {
                            Text("Cancel")
                        }
                    } else {
                        OutlinedButton(
                            onClick = { confirmDeleteData = true },
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("delete-local-data-button")
                        ) {
                            Icon(Icons.Filled.DeleteForever, contentDescription = null)
                            Text("Delete Local Data")
                        }
                    }
                }
            }
        }

        item {
            Text(
                text = ComplianceText.educationalDisclaimer,
                style = MaterialTheme.typography.bodySmall,
                color = LabMutedText
            )
        }
    }
}

private fun android.content.Context.shareText(title: String, text: String) {
    val sendIntent = Intent(Intent.ACTION_SEND).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_SUBJECT, title)
        putExtra(Intent.EXTRA_TEXT, text)
    }
    startActivity(Intent.createChooser(sendIntent, title))
}

private fun android.content.Context.shareFile(title: String, uri: Uri, mimeType: String) {
    val sendIntent = Intent(Intent.ACTION_SEND).apply {
        type = mimeType
        putExtra(Intent.EXTRA_SUBJECT, title)
        putExtra(Intent.EXTRA_STREAM, uri)
        clipData = ClipData.newUri(contentResolver, title, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    startActivity(Intent.createChooser(sendIntent, title))
}

private fun android.content.Context.viewFile(uri: Uri, mimeType: String) {
    val viewIntent = Intent(Intent.ACTION_VIEW).apply {
        setDataAndType(uri, mimeType)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        clipData = ClipData.newUri(contentResolver, "Retirement Readiness Lab PDF report", uri)
    }
    startActivity(viewIntent)
}
