package com.retirementreadinesslab.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.text.KeyboardOptions
import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.LongTermCareAssumption
import com.retirementreadinesslab.model.MortgagePlan
import com.retirementreadinesslab.model.RothConversionStrategy
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCompactCurrency
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabMutedText
import kotlin.math.roundToInt

@Composable
fun OnboardingScreen(state: RetirementLabState) {
    val scenario = state.selectedScenario
    var retirementAge by remember(scenario.id) { mutableStateOf(scenario.household.retirementAge.toFloat()) }
    var spending by remember(scenario.id) { mutableStateOf(scenario.spending.annualBaseSpending.toFloat()) }
    var claimAge by remember(scenario.id) { mutableStateOf(scenario.socialSecurity.claimAge.toFloat()) }
    var rothEnabled by remember(scenario.id) { mutableStateOf(scenario.rothConversion.enabled) }
    var ltcEnabled by remember(scenario.id) { mutableStateOf(scenario.longTermCare.enabled) }
    var pretax by remember(scenario.id) { mutableStateOf(scenario.accounts.pretax.wholeDollarText()) }
    var roth by remember(scenario.id) { mutableStateOf(scenario.accounts.roth.wholeDollarText()) }
    var taxable by remember(scenario.id) { mutableStateOf(scenario.accounts.taxable.wholeDollarText()) }
    var cash by remember(scenario.id) { mutableStateOf(scenario.accounts.cash.wholeDollarText()) }
    var socialSecurityAt67 by remember(scenario.id) { mutableStateOf(scenario.socialSecurity.annualBenefitAt67.wholeDollarText()) }
    var healthcarePremium by remember(scenario.id) { mutableStateOf(scenario.healthcare.preMedicareMonthlyPremium.wholeDollarText()) }
    var mortgagePayment by remember(scenario.id) { mutableStateOf(scenario.mortgage.monthlyPayment.wholeDollarText()) }
    var mortgageYears by remember(scenario.id) { mutableStateOf(scenario.mortgage.yearsLeft.toString()) }
    var validationMessage by remember(scenario.id) { mutableStateOf<String?>(null) }
    val hasUnsavedChanges =
        retirementAge.toInt() != scenario.household.retirementAge ||
            spending.roundToInt().toDouble() != scenario.spending.annualBaseSpending ||
            claimAge.toInt() != scenario.socialSecurity.claimAge ||
            rothEnabled != scenario.rothConversion.enabled ||
            ltcEnabled != scenario.longTermCare.enabled ||
            pretax != scenario.accounts.pretax.wholeDollarText() ||
            roth != scenario.accounts.roth.wholeDollarText() ||
            taxable != scenario.accounts.taxable.wholeDollarText() ||
            cash != scenario.accounts.cash.wholeDollarText() ||
            socialSecurityAt67 != scenario.socialSecurity.annualBenefitAt67.wholeDollarText() ||
            healthcarePremium != scenario.healthcare.preMedicareMonthlyPremium.wholeDollarText() ||
            mortgagePayment != scenario.mortgage.monthlyPayment.wholeDollarText() ||
            mortgageYears != scenario.mortgage.yearsLeft.toString()

    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("setup-screen"),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp)
    ) {
        item {
            SectionHeader(
                title = "Guided setup",
                subtitle = "Change assumptions, then apply them to rerun the active scenario."
            )
        }

        if (state.isRunning) {
            item {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
        }

        item {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        text = if (hasUnsavedChanges) "Unsaved setup changes" else "Setup matches the current scenario",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = if (hasUnsavedChanges) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "Dashboard percentages update only after you tap Apply Changes And Run Stress Test.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = LabMutedText
                    )
                    state.lastRunMessage?.let { message ->
                        Text(
                            text = message,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }
        }

        item {
            SetupCard(title = "Retirement target", value = "Age ${retirementAge.toInt()}") {
                Slider(
                    value = retirementAge,
                    onValueChange = { retirementAge = it },
                    valueRange = scenario.household.currentAge.toFloat()..70f,
                    steps = 70 - scenario.household.currentAge - 1
                )
            }
        }

        item {
            SetupCard(title = "Annual spending", value = spending.toDouble().asCompactCurrency()) {
                Slider(
                    value = spending,
                    onValueChange = { spending = it },
                    valueRange = 35_000f..150_000f,
                    steps = 22
                )
            }
        }

        item {
            SetupCard(title = "Social Security", value = "Claim at ${claimAge.toInt()}") {
                Slider(
                    value = claimAge,
                    onValueChange = { claimAge = it },
                    valueRange = 62f..70f,
                    steps = 7
                )
            }
        }

        item {
            SetupCard(title = "Current balances", value = scenario.accounts.total.asCompactCurrency()) {
                MoneyField("Pre-tax accounts", pretax, onValueChange = { pretax = it })
                MoneyField("Roth accounts", roth, onValueChange = { roth = it })
                MoneyField("Taxable investments", taxable, onValueChange = { taxable = it })
                MoneyField("Cash reserve", cash, onValueChange = { cash = it })
            }
        }

        item {
            SetupCard(title = "Income and fixed costs", value = "Monthly inputs") {
                MoneyField("Social Security at 67", socialSecurityAt67, onValueChange = { socialSecurityAt67 = it })
                MoneyField("Pre-Medicare health premium", healthcarePremium, onValueChange = { healthcarePremium = it })
                MoneyField("Mortgage payment", mortgagePayment, onValueChange = { mortgagePayment = it })
                NumberField("Mortgage years left", mortgageYears, onValueChange = { mortgageYears = it })
            }
        }

        item {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(14.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp)
                ) {
                    Column(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        ToggleRow(
                            title = "Roth conversion lab",
                            subtitle = "Compare filling the 22% bracket after retirement.",
                            checked = rothEnabled,
                            onCheckedChange = { rothEnabled = it }
                        )
                        ToggleRow(
                            title = "Long-term care risk",
                            subtitle = "Stress-test late-life care costs.",
                            checked = ltcEnabled,
                            onCheckedChange = { ltcEnabled = it }
                        )
                    }
                }
            }
        }

        item {
            Button(
                onClick = {
                    val parsed = ParsedSetupInputs(
                        pretax = parseMoney(pretax),
                        roth = parseMoney(roth),
                        taxable = parseMoney(taxable),
                        cash = parseMoney(cash),
                        socialSecurityAt67 = parseMoney(socialSecurityAt67),
                        healthcarePremium = parseMoney(healthcarePremium),
                        mortgagePayment = parseMoney(mortgagePayment),
                        mortgageYears = mortgageYears.toIntOrNull()
                    )
                    if (!parsed.isValid) {
                        validationMessage = "Enter non-negative numbers for balances, income, healthcare, and mortgage fields."
                        return@Button
                    }
                    validationMessage = null
                    state.updateSelected {
                        it.copy(
                            household = it.household.copy(retirementAge = retirementAge.toInt()),
                            spending = it.spending.copy(annualBaseSpending = spending.roundToInt().toDouble()),
                            accounts = AccountBalances(
                                pretax = parsed.pretax!!,
                                roth = parsed.roth!!,
                                taxable = parsed.taxable!!,
                                cash = parsed.cash!!
                            ),
                            healthcare = HealthcarePlan(
                                preMedicareMonthlyPremium = parsed.healthcarePremium!!,
                                healthcareInflationMean = it.healthcare.healthcareInflationMean,
                                healthcareInflationStdDev = it.healthcare.healthcareInflationStdDev,
                                includeMedicarePremiums = it.healthcare.includeMedicarePremiums
                            ),
                            mortgage = MortgagePlan(
                                monthlyPayment = parsed.mortgagePayment!!,
                                yearsLeft = parsed.mortgageYears!!
                            ),
                            socialSecurity = SocialSecurityPlan(
                                annualBenefitAt67 = parsed.socialSecurityAt67!!,
                                claimAge = claimAge.toInt()
                            ),
                            rothConversion = RothConversionStrategy(enabled = rothEnabled, marginalRateCap = 0.22),
                            longTermCare = LongTermCareAssumption(enabled = ltcEnabled)
                        )
                    }
                },
                enabled = !state.isRunning,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    when {
                        state.isRunning -> "Running..."
                        hasUnsavedChanges -> "Apply Changes And Run Stress Test"
                        else -> "Run Current Scenario"
                    }
                )
            }
        }

        validationMessage?.let { message ->
            item {
                Text(message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            }
        }
    }
}

@Composable
private fun SetupCard(title: String, value: String, content: @Composable ColumnScope.() -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(value, style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
            }
            content()
        }
    }
}

@Composable
private fun MoneyField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
private fun NumberField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
private fun ToggleRow(
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = LabMutedText)
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

private data class ParsedSetupInputs(
    val pretax: Double?,
    val roth: Double?,
    val taxable: Double?,
    val cash: Double?,
    val socialSecurityAt67: Double?,
    val healthcarePremium: Double?,
    val mortgagePayment: Double?,
    val mortgageYears: Int?
) {
    val isValid: Boolean
        get() = listOf(
            pretax,
            roth,
            taxable,
            cash,
            socialSecurityAt67,
            healthcarePremium,
            mortgagePayment
        ).all { it != null && it >= 0.0 } && mortgageYears != null && mortgageYears >= 0
}

private fun parseMoney(value: String): Double? {
    return value
        .replace("$", "")
        .replace(",", "")
        .trim()
        .toDoubleOrNull()
}

private fun Double.wholeDollarText(): String = toLong().toString()
