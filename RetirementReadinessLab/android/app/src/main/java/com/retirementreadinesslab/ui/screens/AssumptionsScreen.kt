package com.retirementreadinesslab.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.retirementreadinesslab.compliance.ComplianceText
import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.FilingStatus
import com.retirementreadinesslab.model.Gender
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.LongTermCareAssumption
import com.retirementreadinesslab.model.MarketAssumptions
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.RothConversionStrategy
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPlan
import com.retirementreadinesslab.model.WithdrawalStrategy
import com.retirementreadinesslab.model.validate
import com.retirementreadinesslab.model.warnings
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCurrency
import com.retirementreadinesslab.ui.components.KeyValueRow
import com.retirementreadinesslab.ui.components.ScenarioWarningCard
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabMutedText
import java.util.Locale

@Composable
fun AssumptionsScreen(state: RetirementLabState) {
    val scenario = state.selectedScenario
    var form by remember(scenario) { mutableStateOf(EditableAssumptions.from(scenario)) }
    var validationMessage by remember(scenario) { mutableStateOf<String?>(null) }
    val savedForm = EditableAssumptions.from(scenario)
    val hasUnsavedChanges = form != savedForm
    val warnings = scenario.warnings()

    LazyColumn(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = PaddingValues(16.dp)
    ) {
        item {
            SectionHeader(
                title = "Assumptions",
                subtitle = "Review and tune the active scenario inputs."
            )
        }

        if (state.isRunning) {
            item {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
        }

        item {
            AssumptionCard("Scenario status") {
                Text(
                    text = if (hasUnsavedChanges) "Unsaved assumption changes" else "Assumptions match the current scenario",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = if (hasUnsavedChanges) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                )
                state.lastRunMessage?.let { message ->
                    Text(message, style = MaterialTheme.typography.bodySmall, color = LabMutedText)
                }
                KeyValueRow("Current scenario", scenario.name)
                KeyValueRow("Current account total", scenario.accounts.total.asCurrency())
            }
        }

        item {
            ScenarioWarningCard(title = "Assumption checks", warnings = warnings)
        }

        item {
            AssumptionCard("Household profile") {
                NumberField("Current age", form.currentAge) { form = form.copy(currentAge = it) }
                NumberField("Retirement age", form.retirementAge) { form = form.copy(retirementAge = it) }
                NumberField("Target end age", form.targetEndAge) { form = form.copy(targetEndAge = it) }
                ChoiceGroup(
                    title = "Filing status",
                    options = FilingStatus.entries.toList(),
                    selected = form.filingStatus,
                    label = { it.label },
                    onSelected = { form = form.copy(filingStatus = it) }
                )
                ChoiceGroup(
                    title = "Mortality table",
                    options = Gender.entries.toList(),
                    selected = form.gender,
                    label = { it.label },
                    onSelected = { form = form.copy(gender = it) }
                )
            }
        }

        item {
            AssumptionCard("Spending and Social Security") {
                MoneyField("Annual base spending", form.annualSpending) { form = form.copy(annualSpending = it) }
                PercentField("General inflation mean", form.generalInflationMean) {
                    form = form.copy(generalInflationMean = it)
                }
                PercentField("General inflation volatility", form.generalInflationStdDev) {
                    form = form.copy(generalInflationStdDev = it)
                }
                MoneyField("Social Security at 67", form.socialSecurityAt67) {
                    form = form.copy(socialSecurityAt67 = it)
                }
                NumberField("Social Security claim age", form.claimAge) { form = form.copy(claimAge = it) }
            }
        }

        item {
            AssumptionCard("Accounts") {
                MoneyField("Pre-tax accounts", form.pretax) { form = form.copy(pretax = it) }
                MoneyField("Roth accounts", form.roth) { form = form.copy(roth = it) }
                MoneyField("Taxable investments", form.taxable) { form = form.copy(taxable = it) }
                MoneyField("Cash reserve", form.cash) { form = form.copy(cash = it) }
            }
        }

        item {
            AssumptionCard("Healthcare and long-term care") {
                MoneyField("Pre-Medicare monthly premium", form.preMedicareMonthlyPremium) {
                    form = form.copy(preMedicareMonthlyPremium = it)
                }
                PercentField("Healthcare inflation mean", form.healthcareInflationMean) {
                    form = form.copy(healthcareInflationMean = it)
                }
                PercentField("Healthcare inflation volatility", form.healthcareInflationStdDev) {
                    form = form.copy(healthcareInflationStdDev = it)
                }
                AssumptionToggleRow(
                    title = "Medicare premiums",
                    checked = form.includeMedicarePremiums,
                    onCheckedChange = { form = form.copy(includeMedicarePremiums = it) }
                )
                AssumptionToggleRow(
                    title = "Long-term care shock",
                    checked = form.longTermCareEnabled,
                    onCheckedChange = { form = form.copy(longTermCareEnabled = it) }
                )
                MoneyField("Long-term care annual cost", form.longTermCareAnnualCost) {
                    form = form.copy(longTermCareAnnualCost = it)
                }
                NumberField("Long-term care duration years", form.longTermCareDurationYears) {
                    form = form.copy(longTermCareDurationYears = it)
                }
            }
        }

        item {
            AssumptionCard("Market returns") {
                PercentField("Pre-retirement return mean", form.preRetirementMeanReturn) {
                    form = form.copy(preRetirementMeanReturn = it)
                }
                PercentField("Pre-retirement return volatility", form.preRetirementStdDev) {
                    form = form.copy(preRetirementStdDev = it)
                }
                PercentField("Stock return mean", form.stockMeanReturn) {
                    form = form.copy(stockMeanReturn = it)
                }
                PercentField("Stock return volatility", form.stockStdDev) {
                    form = form.copy(stockStdDev = it)
                }
                PercentField("Bond return mean", form.bondMeanReturn) {
                    form = form.copy(bondMeanReturn = it)
                }
                PercentField("Bond return volatility", form.bondStdDev) {
                    form = form.copy(bondStdDev = it)
                }
            }
        }

        item {
            AssumptionCard("Tax and drawdown strategy") {
                AssumptionToggleRow(
                    title = "Roth conversion lab",
                    checked = form.rothConversionEnabled,
                    onCheckedChange = { form = form.copy(rothConversionEnabled = it) }
                )
                ChoiceGroup(
                    title = "Roth marginal bracket cap",
                    options = TAX_BRACKET_CAPS,
                    selected = form.rothBracketCap,
                    label = { it.percentOptionLabel() },
                    onSelected = { form = form.copy(rothBracketCap = it) }
                )
                AssumptionToggleRow(
                    title = "Use cash reserve during drawdowns",
                    checked = form.useCashReserveDuringDrawdowns,
                    onCheckedChange = { form = form.copy(useCashReserveDuringDrawdowns = it) }
                )
                PercentField("Drawdown trigger", form.drawdownTrigger) {
                    form = form.copy(drawdownTrigger = it)
                }
            }
        }

        item {
            AssumptionCard("Simulation settings") {
                NumberField("Number of simulations", form.numberOfSimulations) {
                    form = form.copy(numberOfSimulations = it)
                }
                NumberField("Random seed", form.seed) { form = form.copy(seed = it) }
                KeyValueRow("Federal tax table", "2024 brackets")
                KeyValueRow("Engine cadence", "Monthly cashflow model")
            }
        }

        item {
            Button(
                onClick = {
                    val parsed = form.toScenario(scenario)
                    if (parsed.error != null) {
                        validationMessage = parsed.error
                        return@Button
                    }
                    validationMessage = null
                    state.updateSelected { parsed.scenario!! }
                },
                enabled = !state.isRunning,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    when {
                        state.isRunning -> "Running..."
                        hasUnsavedChanges -> "Apply Assumptions And Run Stress Test"
                        else -> "Run Current Assumptions"
                    }
                )
            }
        }

        validationMessage?.let { message ->
            item {
                Text(message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
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

@Composable
private fun AssumptionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
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
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
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
private fun PercentField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text("$label (%)") },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text),
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
private fun AssumptionToggleRow(
    title: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(title, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun <T> ChoiceGroup(
    title: String,
    options: List<T>,
    selected: T,
    label: (T) -> String,
    onSelected: (T) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(title, style = MaterialTheme.typography.labelLarge, color = LabMutedText)
        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            options.forEach { option ->
                val text = label(option)
                if (option == selected) {
                    Button(onClick = { onSelected(option) }) {
                        Text(text)
                    }
                } else {
                    OutlinedButton(onClick = { onSelected(option) }) {
                        Text(text)
                    }
                }
            }
        }
    }
}

private data class EditableAssumptions(
    val currentAge: String,
    val retirementAge: String,
    val targetEndAge: String,
    val filingStatus: FilingStatus,
    val gender: Gender,
    val annualSpending: String,
    val generalInflationMean: String,
    val generalInflationStdDev: String,
    val pretax: String,
    val roth: String,
    val taxable: String,
    val cash: String,
    val preMedicareMonthlyPremium: String,
    val healthcareInflationMean: String,
    val healthcareInflationStdDev: String,
    val includeMedicarePremiums: Boolean,
    val socialSecurityAt67: String,
    val claimAge: String,
    val preRetirementMeanReturn: String,
    val preRetirementStdDev: String,
    val stockMeanReturn: String,
    val stockStdDev: String,
    val bondMeanReturn: String,
    val bondStdDev: String,
    val rothConversionEnabled: Boolean,
    val rothBracketCap: Double,
    val useCashReserveDuringDrawdowns: Boolean,
    val drawdownTrigger: String,
    val longTermCareEnabled: Boolean,
    val longTermCareAnnualCost: String,
    val longTermCareDurationYears: String,
    val numberOfSimulations: String,
    val seed: String
) {
    fun toScenario(base: RetirementScenario): ParsedAssumptions {
        val currentAge = currentAge.toIntOrNull()
        val retirementAge = retirementAge.toIntOrNull()
        val targetEndAge = targetEndAge.toIntOrNull()
        val annualSpending = parseMoney(annualSpending)
        val generalInflationMean = parsePercent(generalInflationMean)
        val generalInflationStdDev = parsePercent(generalInflationStdDev)
        val pretax = parseMoney(pretax)
        val roth = parseMoney(roth)
        val taxable = parseMoney(taxable)
        val cash = parseMoney(cash)
        val preMedicareMonthlyPremium = parseMoney(preMedicareMonthlyPremium)
        val healthcareInflationMean = parsePercent(healthcareInflationMean)
        val healthcareInflationStdDev = parsePercent(healthcareInflationStdDev)
        val socialSecurityAt67 = parseMoney(socialSecurityAt67)
        val claimAge = claimAge.toIntOrNull()
        val preRetirementMeanReturn = parsePercent(preRetirementMeanReturn)
        val preRetirementStdDev = parsePercent(preRetirementStdDev)
        val stockMeanReturn = parsePercent(stockMeanReturn)
        val stockStdDev = parsePercent(stockStdDev)
        val bondMeanReturn = parsePercent(bondMeanReturn)
        val bondStdDev = parsePercent(bondStdDev)
        val drawdownTrigger = parsePercent(drawdownTrigger)
        val longTermCareAnnualCost = parseMoney(longTermCareAnnualCost)
        val longTermCareDurationYears = longTermCareDurationYears.toIntOrNull()
        val numberOfSimulations = numberOfSimulations.toIntOrNull()
        val seed = seed.toLongOrNull()
        val firstError = listOfNotNull(
            requireInt("Current age", currentAge, 18, 90),
            requireInt("Retirement age", retirementAge, 18, 75),
            requireInt("Target end age", targetEndAge, 60, 115),
            requireMoney("Annual base spending", annualSpending),
            requirePercent("General inflation mean", generalInflationMean, -0.02, 0.15),
            requirePercent("General inflation volatility", generalInflationStdDev, 0.0, 0.30),
            requireMoney("Pre-tax accounts", pretax),
            requireMoney("Roth accounts", roth),
            requireMoney("Taxable investments", taxable),
            requireMoney("Cash reserve", cash),
            requireMoney("Pre-Medicare monthly premium", preMedicareMonthlyPremium),
            requirePercent("Healthcare inflation mean", healthcareInflationMean, 0.0, 0.20),
            requirePercent("Healthcare inflation volatility", healthcareInflationStdDev, 0.0, 0.30),
            requireMoney("Social Security at 67", socialSecurityAt67),
            requireInt("Social Security claim age", claimAge, 62, 70),
            requirePercent("Pre-retirement return mean", preRetirementMeanReturn, -0.20, 0.25),
            requirePercent("Pre-retirement return volatility", preRetirementStdDev, 0.0, 0.60),
            requirePercent("Stock return mean", stockMeanReturn, -0.20, 0.25),
            requirePercent("Stock return volatility", stockStdDev, 0.0, 0.60),
            requirePercent("Bond return mean", bondMeanReturn, -0.20, 0.20),
            requirePercent("Bond return volatility", bondStdDev, 0.0, 0.40),
            requirePercent("Drawdown trigger", drawdownTrigger, -0.50, 0.25),
            requireMoney("Long-term care annual cost", longTermCareAnnualCost),
            requireInt("Long-term care duration years", longTermCareDurationYears, 1, 10),
            requireInt("Number of simulations", numberOfSimulations, 100, 10_000),
            if (seed == null) "Random seed must be a whole number." else null
        ).firstOrNull()

        if (firstError != null) return ParsedAssumptions(error = firstError)

        val updated = base.copy(
            household = HouseholdProfile(
                currentAge = currentAge!!,
                retirementAge = retirementAge!!,
                targetEndAge = targetEndAge!!,
                filingStatus = filingStatus,
                gender = gender
            ),
            spending = SpendingPlan(
                annualBaseSpending = annualSpending!!,
                generalInflationMean = generalInflationMean!!,
                generalInflationStdDev = generalInflationStdDev!!
            ),
            accounts = AccountBalances(
                pretax = pretax!!,
                roth = roth!!,
                taxable = taxable!!,
                cash = cash!!
            ),
            healthcare = HealthcarePlan(
                preMedicareMonthlyPremium = preMedicareMonthlyPremium!!,
                healthcareInflationMean = healthcareInflationMean!!,
                healthcareInflationStdDev = healthcareInflationStdDev!!,
                includeMedicarePremiums = includeMedicarePremiums
            ),
            socialSecurity = SocialSecurityPlan(
                annualBenefitAt67 = socialSecurityAt67!!,
                claimAge = claimAge!!
            ),
            market = MarketAssumptions(
                preRetirementMeanReturn = preRetirementMeanReturn!!,
                preRetirementStdDev = preRetirementStdDev!!,
                stockMeanReturn = stockMeanReturn!!,
                stockStdDev = stockStdDev!!,
                bondMeanReturn = bondMeanReturn!!,
                bondStdDev = bondStdDev!!
            ),
            rothConversion = RothConversionStrategy(
                enabled = rothConversionEnabled,
                marginalRateCap = rothBracketCap
            ),
            withdrawalStrategy = WithdrawalStrategy(
                useCashReserveDuringDrawdowns = useCashReserveDuringDrawdowns,
                drawdownTrigger = drawdownTrigger!!
            ),
            longTermCare = LongTermCareAssumption(
                enabled = longTermCareEnabled,
                annualCost = longTermCareAnnualCost!!,
                averageDurationYears = longTermCareDurationYears!!
            ),
            numberOfSimulations = numberOfSimulations!!,
            seed = seed!!
        )

        val modelError = updated.validate().firstOrNull()
        return if (modelError != null) {
            ParsedAssumptions(error = modelError)
        } else {
            ParsedAssumptions(scenario = updated)
        }
    }

    companion object {
        fun from(scenario: RetirementScenario): EditableAssumptions {
            return EditableAssumptions(
                currentAge = scenario.household.currentAge.toString(),
                retirementAge = scenario.household.retirementAge.toString(),
                targetEndAge = scenario.household.targetEndAge.toString(),
                filingStatus = scenario.household.filingStatus,
                gender = scenario.household.gender,
                annualSpending = scenario.spending.annualBaseSpending.wholeDollarText(),
                generalInflationMean = scenario.spending.generalInflationMean.percentInputText(),
                generalInflationStdDev = scenario.spending.generalInflationStdDev.percentInputText(),
                pretax = scenario.accounts.pretax.wholeDollarText(),
                roth = scenario.accounts.roth.wholeDollarText(),
                taxable = scenario.accounts.taxable.wholeDollarText(),
                cash = scenario.accounts.cash.wholeDollarText(),
                preMedicareMonthlyPremium = scenario.healthcare.preMedicareMonthlyPremium.wholeDollarText(),
                healthcareInflationMean = scenario.healthcare.healthcareInflationMean.percentInputText(),
                healthcareInflationStdDev = scenario.healthcare.healthcareInflationStdDev.percentInputText(),
                includeMedicarePremiums = scenario.healthcare.includeMedicarePremiums,
                socialSecurityAt67 = scenario.socialSecurity.annualBenefitAt67.wholeDollarText(),
                claimAge = scenario.socialSecurity.claimAge.toString(),
                preRetirementMeanReturn = scenario.market.preRetirementMeanReturn.percentInputText(),
                preRetirementStdDev = scenario.market.preRetirementStdDev.percentInputText(),
                stockMeanReturn = scenario.market.stockMeanReturn.percentInputText(),
                stockStdDev = scenario.market.stockStdDev.percentInputText(),
                bondMeanReturn = scenario.market.bondMeanReturn.percentInputText(),
                bondStdDev = scenario.market.bondStdDev.percentInputText(),
                rothConversionEnabled = scenario.rothConversion.enabled,
                rothBracketCap = closestTaxCap(scenario.rothConversion.marginalRateCap),
                useCashReserveDuringDrawdowns = scenario.withdrawalStrategy.useCashReserveDuringDrawdowns,
                drawdownTrigger = scenario.withdrawalStrategy.drawdownTrigger.percentInputText(),
                longTermCareEnabled = scenario.longTermCare.enabled,
                longTermCareAnnualCost = scenario.longTermCare.annualCost.wholeDollarText(),
                longTermCareDurationYears = scenario.longTermCare.averageDurationYears.toString(),
                numberOfSimulations = scenario.numberOfSimulations.toString(),
                seed = scenario.seed.toString()
            )
        }
    }
}

private data class ParsedAssumptions(
    val scenario: RetirementScenario? = null,
    val error: String? = null
)

private val TAX_BRACKET_CAPS = listOf(0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37)

private fun closestTaxCap(value: Double): Double {
    return TAX_BRACKET_CAPS.minBy { kotlin.math.abs(it - value) }
}

private fun Double.percentOptionLabel(): String {
    return "${String.format(Locale.US, "%.0f", this * 100.0)}%"
}

private fun parseMoney(value: String): Double? {
    return value
        .replace("$", "")
        .replace(",", "")
        .trim()
        .toDoubleOrNull()
}

private fun parsePercent(value: String): Double? {
    return value
        .replace("%", "")
        .trim()
        .toDoubleOrNull()
        ?.div(100.0)
}

private fun requireMoney(label: String, value: Double?): String? {
    return if (value == null || value < 0.0) "$label must be a non-negative number." else null
}

private fun requireInt(label: String, value: Int?, min: Int, max: Int): String? {
    return if (value == null || value !in min..max) "$label must be between $min and $max." else null
}

private fun requirePercent(label: String, value: Double?, min: Double, max: Double): String? {
    if (value == null || value !in min..max) {
        return "$label must be between ${min.percentInputText()}% and ${max.percentInputText()}%."
    }
    return null
}

private fun Double.wholeDollarText(): String = toLong().toString()

private fun Double.percentInputText(): String {
    return String.format(Locale.US, "%.1f", this * 100.0)
        .trimEnd('0')
        .trimEnd('.')
}
