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
import androidx.compose.material.icons.filled.PictureAsPdf
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.TableChart
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
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.retirementreadinesslab.compliance.ComplianceText
import com.retirementreadinesslab.data.ScenarioJson
import com.retirementreadinesslab.reports.ReportBuilder
import com.retirementreadinesslab.reports.ReportPdfExporter
import com.retirementreadinesslab.simulation.RetirementSimulator
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCompactCurrency
import com.retirementreadinesslab.ui.asPercent
import com.retirementreadinesslab.ui.components.KeyValueRow
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabMutedText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun ReportsScreen(state: RetirementLabState) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val scenario = state.selectedScenario
    val result = state.selectedResult
    val report = ReportBuilder.buildTextReport(scenario, result)
    val scenarioBackup = ScenarioJson.encodeScenarios(state.scenarios.toList())
    var importText by remember { mutableStateOf("") }
    var importMessage by remember { mutableStateOf<String?>(null) }
    var exportMessage by remember { mutableStateOf<String?>(null) }
    var confirmDeleteData by remember { mutableStateOf(false) }
    var isPreparingComparisonCsv by remember { mutableStateOf(false) }

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
                    KeyValueRow("Scenario", scenario.name)
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
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("share-pdf-report-button")
                    ) {
                        Icon(Icons.Filled.PictureAsPdf, contentDescription = null)
                        Text("Share PDF Report")
                    }
                    Button(
                        onClick = {
                            context.shareText(
                                title = "Retirement Readiness Lab report",
                                text = report
                            )
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("share-text-report-button")
                    ) {
                        Icon(Icons.Filled.Share, contentDescription = null)
                        Text("Share Report")
                    }
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
                    OutlinedButton(
                        onClick = {
                            val scenariosForExport = state.scenarios.toList()
                            isPreparingComparisonCsv = true
                            exportMessage = "Preparing comparison CSV..."
                            scope.launch {
                                val csv = runCatching {
                                    withContext(Dispatchers.Default) {
                                        val results = scenariosForExport.associate { exportScenario ->
                                            exportScenario.id to RetirementSimulator.run(exportScenario)
                                        }
                                        ReportBuilder.buildScenarioCsv(
                                            scenarios = scenariosForExport,
                                            results = results
                                        )
                                    }
                                }
                                csv.onSuccess { generatedCsv ->
                                    exportMessage = null
                                    context.shareText(
                                        title = "Retirement Readiness Lab scenario comparison CSV",
                                        text = generatedCsv
                                    )
                                }.onFailure {
                                    exportMessage = "Comparison CSV could not be created: ${it.message ?: "unknown error"}"
                                }
                                isPreparingComparisonCsv = false
                            }
                        },
                        enabled = !isPreparingComparisonCsv,
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("share-comparison-csv-button")
                    ) {
                        Icon(Icons.Filled.TableChart, contentDescription = null)
                        Text(if (isPreparingComparisonCsv) "Preparing CSV..." else "Share Comparison CSV")
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
