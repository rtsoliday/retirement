package com.retirementreadinesslab.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import com.retirementreadinesslab.model.BudgetLineItem
import com.retirementreadinesslab.model.BudgetProfile
import com.retirementreadinesslab.model.MonthlyBudget
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCurrency
import com.retirementreadinesslab.ui.components.KeyValueRow
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabMutedText
import java.time.YearMonth
import java.time.format.DateTimeFormatter
import java.time.format.DateTimeParseException
import java.util.Locale

@Composable
fun BudgetScreen(state: RetirementLabState) {
    val scenario = state.selectedScenario
    var form by remember(scenario) { mutableStateOf(EditableBudget.from(scenario.budget)) }
    var selectedMonth by remember(scenario.id) { mutableStateOf(currentMonthKey()) }
    var validationMessage by remember(scenario) { mutableStateOf<String?>(null) }
    var validationIsError by remember(scenario) { mutableStateOf(false) }
    val preview = form.previewBudget()
    val savedForm = EditableBudget.from(scenario.budget)
    val hasUnsavedChanges = form != savedForm
    val selectedMonthBudget = form.monthFor(selectedMonth)
    val canMoveForward = parseMonth(selectedMonth) < YearMonth.now()

    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("budget-screen"),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        contentPadding = PaddingValues(16.dp)
    ) {
        item {
            SectionHeader(
                title = "Budget",
                subtitle = "Estimate annual base spending without tracking every purchase."
            )
        }

        item {
            BudgetCard("Annual base spending estimate") {
                Text(
                    text = preview.annualBaseSpendingEstimate.asCurrency(),
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
                KeyValueRow("Current setup spending", scenario.spending.annualBaseSpending.asCurrency())
                KeyValueRow("Monthly records used", preview.monthsUsedForEstimate.size.toString())
                Text(
                    text = "Mortgage payments, rent, and healthcare premiums stay separate in Setup. Property taxes and insurance entered here are included in annual base spending.",
                    style = MaterialTheme.typography.bodySmall,
                    color = LabMutedText
                )
            }
        }

        item {
            BudgetCard("Annual amounts") {
                MoneyField("Yearly property taxes", form.annualPropertyTaxes) {
                    form = form.copy(annualPropertyTaxes = it)
                }
                MoneyField("Yearly home insurance", form.annualHomeInsurance) {
                    form = form.copy(annualHomeInsurance = it)
                }
                MoneyField("Yearly auto insurance", form.annualAutoInsurance) {
                    form = form.copy(annualAutoInsurance = it)
                }
            }
        }

        item {
            BudgetCard("Monthly spending") {
                MonthSelector(
                    month = selectedMonth,
                    canMoveForward = canMoveForward,
                    onPrevious = { selectedMonth = shiftMonth(selectedMonth, -1) },
                    onNext = { selectedMonth = shiftMonth(selectedMonth, 1) }
                )
                CategoryEditor(
                    title = "Monthly bills from checking or savings",
                    subtitle = "Exclude credit card, mortgage, and rent payments.",
                    items = selectedMonthBudget.checkingSavingsBills,
                    addButtonTag = "add-checking-bill-button",
                    onItemsChange = { updatedItems ->
                        form = form.updateMonth(selectedMonth) {
                            copy(checkingSavingsBills = updatedItems)
                        }
                    }
                )
                CategoryEditor(
                    title = "Monthly credit card bills",
                    subtitle = "Use the total paid to each card, not individual purchases.",
                    items = selectedMonthBudget.creditCardBills,
                    addButtonTag = "add-credit-card-bill-button",
                    onItemsChange = { updatedItems ->
                        form = form.updateMonth(selectedMonth) {
                            copy(creditCardBills = updatedItems)
                        }
                    }
                )
                MoneyField("Monthly cash spent and ATM withdrawals", selectedMonthBudget.cashAndAtmWithdrawals) {
                    form = form.updateMonth(selectedMonth) {
                        copy(cashAndAtmWithdrawals = it)
                    }
                }
            }
        }

        item {
            BudgetCard("Calculation") {
                KeyValueRow("Annual taxes and insurance", preview.annualFixedSpending.asCurrency())
                KeyValueRow("Average monthly spending", preview.averageMonthlySpending.asCurrency())
                KeyValueRow("Annualized monthly spending", preview.annualizedMonthlySpending.asCurrency())
                KeyValueRow("Estimated annual base spending", preview.annualBaseSpendingEstimate.asCurrency())
                Text(
                    text = "Annualized monthly spending is the average monthly record multiplied by 12. Estimated annual base spending adds annual property taxes, home insurance, and auto insurance. The estimate uses the latest 12 monthly records when more than 12 months are saved.",
                    style = MaterialTheme.typography.bodySmall,
                    color = LabMutedText
                )
            }
        }

        item {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(
                    onClick = {
                        val parsed = form.toBudget()
                        if (parsed.error != null) {
                            validationMessage = parsed.error
                            validationIsError = true
                        } else {
                            state.saveSelectedBudget(parsed.budget!!)
                            validationMessage = "Budget saved."
                            validationIsError = false
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("save-budget-button")
                ) {
                    Text(if (hasUnsavedChanges) "Save Budget" else "Budget Saved")
                }
                Button(
                    onClick = {
                        val parsed = form.toBudget()
                        if (parsed.error != null) {
                            validationMessage = parsed.error
                            validationIsError = true
                            return@Button
                        }
                        val error = state.applyBudgetToAnnualBaseSpending(parsed.budget!!)
                        validationMessage = error ?: "Budget estimate applied to annual base spending."
                        validationIsError = error != null
                    },
                    enabled = !state.isRunning && preview.annualBaseSpendingEstimate > 0.0,
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("apply-budget-spending-button")
                ) {
                    Text("Use Estimate For Annual Base Spending")
                }
            }
        }

        validationMessage?.let { message ->
            item {
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodySmall,
                    color = if (validationIsError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

@Composable
private fun BudgetCard(title: String, content: @Composable ColumnScope.() -> Unit) {
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
private fun MonthSelector(
    month: String,
    canMoveForward: Boolean,
    onPrevious: () -> Unit,
    onNext: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        OutlinedButton(onClick = onPrevious, modifier = Modifier.weight(1f)) {
            Text("Previous")
        }
        Text(
            text = monthLabel(month),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier
                .weight(1.3f)
                .testTag("budget-month-label")
        )
        OutlinedButton(
            onClick = onNext,
            enabled = canMoveForward,
            modifier = Modifier.weight(1f)
        ) {
            Text("Next")
        }
    }
}

@Composable
private fun CategoryEditor(
    title: String,
    subtitle: String,
    items: List<EditableBudgetLineItem>,
    addButtonTag: String,
    onItemsChange: (List<EditableBudgetLineItem>) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(title, style = MaterialTheme.typography.labelLarge)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = LabMutedText)
        }
        items.forEach { item ->
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedTextField(
                    value = item.name,
                    onValueChange = { newName ->
                        onItemsChange(items.map { if (it.id == item.id) it.copy(name = newName) else it })
                    },
                    label = { Text("Name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    MoneyField(
                        label = "Monthly amount",
                        value = item.monthlyAmount,
                        modifier = Modifier.weight(1f),
                        onValueChange = { newAmount ->
                            onItemsChange(items.map { if (it.id == item.id) it.copy(monthlyAmount = newAmount) else it })
                        }
                    )
                    IconButton(
                        onClick = { onItemsChange(items.filterNot { it.id == item.id }) },
                        modifier = Modifier.width(48.dp)
                    ) {
                        Icon(Icons.Filled.Delete, contentDescription = "Delete ${item.name.ifBlank { "budget item" }}")
                    }
                }
            }
        }
        OutlinedButton(
            onClick = {
                onItemsChange(items + EditableBudgetLineItem(
                    id = "budget-item-${System.currentTimeMillis()}-${items.size}",
                    name = "",
                    monthlyAmount = ""
                ))
            },
            modifier = Modifier
                .fillMaxWidth()
                .testTag(addButtonTag)
        ) {
            Icon(Icons.Filled.Add, contentDescription = null)
            Text("Add")
        }
    }
}

@Composable
private fun MoneyField(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    onValueChange: (String) -> Unit
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        modifier = modifier.fillMaxWidth()
    )
}

private data class EditableBudget(
    val annualPropertyTaxes: String,
    val annualHomeInsurance: String,
    val annualAutoInsurance: String,
    val monthlyBudgets: List<EditableMonthlyBudget>
) {
    fun monthFor(month: String): EditableMonthlyBudget {
        return monthlyBudgets.firstOrNull { it.month == month } ?: EditableMonthlyBudget(month = month)
    }

    fun updateMonth(month: String, transform: EditableMonthlyBudget.() -> EditableMonthlyBudget): EditableBudget {
        val current = monthFor(month)
        val updated = current.transform()
        val others = monthlyBudgets.filterNot { it.month == month }
        return copy(monthlyBudgets = (others + updated).sortedBy { it.month })
    }

    fun previewBudget(): BudgetProfile {
        return BudgetProfile(
            annualPropertyTaxes = parseMoneyOrZero(annualPropertyTaxes),
            annualHomeInsurance = parseMoneyOrZero(annualHomeInsurance),
            annualAutoInsurance = parseMoneyOrZero(annualAutoInsurance),
            monthlyBudgets = monthlyBudgets.mapNotNull { it.previewMonthlyBudget() }
        )
    }

    fun toBudget(): ParsedBudget {
        val annualPropertyTaxes = parseRequiredMoney("Yearly property taxes", annualPropertyTaxes)
        val annualHomeInsurance = parseRequiredMoney("Yearly home insurance", annualHomeInsurance)
        val annualAutoInsurance = parseRequiredMoney("Yearly auto insurance", annualAutoInsurance)
        val annualError = listOf(
            annualPropertyTaxes.error,
            annualHomeInsurance.error,
            annualAutoInsurance.error
        ).firstOrNull { it != null }
        if (annualError != null) return ParsedBudget(error = annualError)

        val parsedMonths = mutableListOf<MonthlyBudget>()
        monthlyBudgets.forEach { month ->
            val parsed = month.toMonthlyBudget()
            if (parsed.error != null) return ParsedBudget(error = parsed.error)
            if (parsed.monthlyBudget != null) parsedMonths += parsed.monthlyBudget
        }

        return ParsedBudget(
            budget = BudgetProfile(
                annualPropertyTaxes = annualPropertyTaxes.value,
                annualHomeInsurance = annualHomeInsurance.value,
                annualAutoInsurance = annualAutoInsurance.value,
                monthlyBudgets = parsedMonths.sortedBy { it.month }
            )
        )
    }

    companion object {
        fun from(budget: BudgetProfile): EditableBudget {
            return EditableBudget(
                annualPropertyTaxes = budget.annualPropertyTaxes.wholeDollarText(),
                annualHomeInsurance = budget.annualHomeInsurance.wholeDollarText(),
                annualAutoInsurance = budget.annualAutoInsurance.wholeDollarText(),
                monthlyBudgets = budget.monthlyBudgets
                    .sortedBy { it.month }
                    .map { EditableMonthlyBudget.from(it) }
            )
        }
    }
}

private data class EditableMonthlyBudget(
    val month: String,
    val checkingSavingsBills: List<EditableBudgetLineItem> = emptyList(),
    val creditCardBills: List<EditableBudgetLineItem> = emptyList(),
    val cashAndAtmWithdrawals: String = ""
) {
    fun previewMonthlyBudget(): MonthlyBudget? {
        val checking = checkingSavingsBills.mapNotNull { it.previewLineItem() }
        val creditCards = creditCardBills.mapNotNull { it.previewLineItem() }
        val cash = parseMoneyOrZero(cashAndAtmWithdrawals)
        if (checking.isEmpty() && creditCards.isEmpty() && cash == 0.0) return null
        return MonthlyBudget(
            month = month,
            checkingSavingsBills = checking,
            creditCardBills = creditCards,
            cashAndAtmWithdrawals = cash
        )
    }

    fun toMonthlyBudget(): ParsedMonthlyBudget {
        val checking = mutableListOf<BudgetLineItem>()
        checkingSavingsBills.forEachIndexed { index, item ->
            val parsed = item.toLineItem("Monthly bills from checking or savings", index)
            if (parsed.error != null) return ParsedMonthlyBudget(error = parsed.error)
            if (parsed.lineItem != null) checking += parsed.lineItem
        }

        val creditCards = mutableListOf<BudgetLineItem>()
        creditCardBills.forEachIndexed { index, item ->
            val parsed = item.toLineItem("Monthly credit card bills", index)
            if (parsed.error != null) return ParsedMonthlyBudget(error = parsed.error)
            if (parsed.lineItem != null) creditCards += parsed.lineItem
        }

        val cash = parseRequiredMoney("Monthly cash spent and ATM withdrawals for ${monthLabel(month)}", cashAndAtmWithdrawals)
        if (cash.error != null) return ParsedMonthlyBudget(error = cash.error)
        if (checking.isEmpty() && creditCards.isEmpty() && cash.value == 0.0) return ParsedMonthlyBudget()

        return ParsedMonthlyBudget(
            monthlyBudget = MonthlyBudget(
                month = month,
                checkingSavingsBills = checking,
                creditCardBills = creditCards,
                cashAndAtmWithdrawals = cash.value
            )
        )
    }

    companion object {
        fun from(monthlyBudget: MonthlyBudget): EditableMonthlyBudget {
            return EditableMonthlyBudget(
                month = monthlyBudget.month,
                checkingSavingsBills = monthlyBudget.checkingSavingsBills.map { EditableBudgetLineItem.from(it) },
                creditCardBills = monthlyBudget.creditCardBills.map { EditableBudgetLineItem.from(it) },
                cashAndAtmWithdrawals = monthlyBudget.cashAndAtmWithdrawals.wholeDollarText()
            )
        }
    }
}

private data class EditableBudgetLineItem(
    val id: String,
    val name: String,
    val monthlyAmount: String
) {
    fun previewLineItem(): BudgetLineItem? {
        val amount = parseMoneyOrZero(monthlyAmount)
        if (name.isBlank() && amount == 0.0) return null
        return BudgetLineItem(
            id = id,
            name = name.trim().ifBlank { "Budget item" },
            monthlyAmount = amount
        )
    }

    fun toLineItem(category: String, index: Int): ParsedBudgetLineItem {
        val amount = parseRequiredMoney("$category item ${index + 1}", monthlyAmount)
        if (amount.error != null) return ParsedBudgetLineItem(error = amount.error)
        if (name.isBlank() && amount.value == 0.0) return ParsedBudgetLineItem()
        return ParsedBudgetLineItem(
            lineItem = BudgetLineItem(
                id = id,
                name = name.trim().ifBlank { "Budget item ${index + 1}" },
                monthlyAmount = amount.value
            )
        )
    }

    companion object {
        fun from(item: BudgetLineItem): EditableBudgetLineItem {
            return EditableBudgetLineItem(
                id = item.id,
                name = item.name,
                monthlyAmount = item.monthlyAmount.wholeDollarText()
            )
        }
    }
}

private data class ParsedBudget(
    val budget: BudgetProfile? = null,
    val error: String? = null
)

private data class ParsedMonthlyBudget(
    val monthlyBudget: MonthlyBudget? = null,
    val error: String? = null
)

private data class ParsedBudgetLineItem(
    val lineItem: BudgetLineItem? = null,
    val error: String? = null
)

private data class ParsedMoney(
    val value: Double = 0.0,
    val error: String? = null
)

private fun parseRequiredMoney(label: String, value: String): ParsedMoney {
    val parsed = parseMoney(value)
    return if (parsed == null || parsed < 0.0) {
        ParsedMoney(error = "$label must be a non-negative number.")
    } else {
        ParsedMoney(value = parsed)
    }
}

private fun parseMoneyOrZero(value: String): Double {
    return parseMoney(value)?.coerceAtLeast(0.0) ?: 0.0
}

private fun parseMoney(value: String): Double? {
    val cleaned = value
        .replace("$", "")
        .replace(",", "")
        .trim()
    if (cleaned.isBlank()) return 0.0
    return cleaned.toDoubleOrNull()
}

private fun currentMonthKey(): String = YearMonth.now().toString()

private fun shiftMonth(month: String, delta: Long): String {
    return parseMonth(month).plusMonths(delta).coerceAtMost(YearMonth.now()).toString()
}

private fun parseMonth(month: String): YearMonth {
    return try {
        YearMonth.parse(month)
    } catch (_: DateTimeParseException) {
        YearMonth.now()
    }
}

private fun monthLabel(month: String): String {
    val formatter = DateTimeFormatter.ofPattern("MMMM yyyy", Locale.US)
    return parseMonth(month).format(formatter)
}

private fun Double.wholeDollarText(): String {
    return if (this == 0.0) "" else toLong().toString()
}
