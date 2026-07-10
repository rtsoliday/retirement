package com.retirementreadinesslab.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.LockOpen
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.retirementreadinesslab.compliance.ComplianceText
import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.DEFAULT_GENERAL_INFLATION_MEAN
import com.retirementreadinesslab.model.DEFAULT_GENERAL_INFLATION_STD_DEV
import com.retirementreadinesslab.model.DEFAULT_HEALTHCARE_INFLATION_MEAN
import com.retirementreadinesslab.model.DEFAULT_HEALTHCARE_INFLATION_STD_DEV
import com.retirementreadinesslab.model.DEFAULT_PRE_MEDICARE_MONTHLY_PREMIUM
import com.retirementreadinesslab.model.DEFAULT_PROJECTION_END_AGE
import com.retirementreadinesslab.model.FeatureAccess
import com.retirementreadinesslab.model.FilingStatus
import com.retirementreadinesslab.model.FREE_SIMULATION_LIMIT
import com.retirementreadinesslab.model.Gender
import com.retirementreadinesslab.model.GuaranteedIncomePlan
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HomePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.LongTermCareAssumption
import com.retirementreadinesslab.model.MarketAssumptions
import com.retirementreadinesslab.model.MortgagePlan
import com.retirementreadinesslab.model.PENALTY_FREE_WITHDRAWAL_AGE_MONTHS
import com.retirementreadinesslab.model.PostRetirementAllocationStrategy
import com.retirementreadinesslab.model.PostRetirementAllocationTier
import com.retirementreadinesslab.model.PRO_SIMULATION_LIMIT
import com.retirementreadinesslab.model.RentPlan
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.RothConversionStrategy
import com.retirementreadinesslab.model.RULE_OF_55_MINIMUM_RETIREMENT_AGE
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPlan
import com.retirementreadinesslab.model.SpendingPathModel
import com.retirementreadinesslab.model.WithdrawalStrategy
import com.retirementreadinesslab.model.forFeatureAccess
import com.retirementreadinesslab.model.validate
import com.retirementreadinesslab.model.warnings
import com.retirementreadinesslab.simulation.MedicarePremiums
import com.retirementreadinesslab.simulation.RetirementSimulator
import com.retirementreadinesslab.simulation.SocialSecurity
import com.retirementreadinesslab.state.RetirementLabState
import com.retirementreadinesslab.ui.asCurrency
import com.retirementreadinesslab.ui.findActivity
import com.retirementreadinesslab.ui.components.KeyValueRow
import com.retirementreadinesslab.ui.components.ProLockedInlineNotice
import com.retirementreadinesslab.ui.components.ScenarioWarningCard
import com.retirementreadinesslab.ui.components.SectionHeader
import com.retirementreadinesslab.ui.theme.LabMutedText
import java.util.Locale
import kotlin.math.roundToInt

private const val SHOW_CASH_RESERVE_DRAWDOWN_SETTINGS = false

@Composable
fun SetupScreen(
    state: RetirementLabState,
    onRunCurrentSetup: () -> Unit = {}
) {
    val activity = LocalContext.current.findActivity()
    val scenario = state.selectedScenario
    val featureAccess = state.featureAccess
    val visibleScenario = scenario.forFeatureAccess(featureAccess)
    val isProUnlocked = featureAccess.isProUnlocked
    var form by remember(scenario, featureAccess) { mutableStateOf(EditableAssumptions.from(visibleScenario)) }
    var validationMessage by remember(scenario, featureAccess) { mutableStateOf<String?>(null) }
    var promoCodeText by remember { mutableStateOf("") }
    val savedForm = EditableAssumptions.from(visibleScenario)
    val hasUnsavedChanges = form != savedForm
    val warnings = visibleScenario.warnings()

    val runCurrentSetup = {
        val parsed = form.toScenario(scenario, featureAccess)
        if (parsed.error != null) {
            validationMessage = parsed.error
        } else {
            validationMessage = null
            state.updateSelected { parsed.scenario!! }
            onRunCurrentSetup()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .testTag("setup-screen"),
            verticalArrangement = Arrangement.spacedBy(14.dp),
            contentPadding = PaddingValues(start = 16.dp, top = 16.dp, end = 16.dp, bottom = 128.dp)
        ) {
            item {
                SectionHeader(
                    title = "Setup",
                    subtitle = "Edit your scenario values, then click Run Current Scenario"
                )
            }

            if (state.isRunning) {
                item {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
            }

            if (!isProUnlocked && state.supportsUserPurchases) {
                item {
                    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.08f))) {
                        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            Text(
                                "Pro unlock",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Text(
                                "Unlock advanced assumptions, Scenario Lab, higher simulation counts, and report sharing.",
                                style = MaterialTheme.typography.bodyMedium,
                                color = LabMutedText
                            )
                            Text(
                                "For Google review access, enter an official Google Play promo code for Pro Unlock. After redeeming it in Google Play, return here and tap Restore Purchase.",
                                style = MaterialTheme.typography.bodySmall,
                                color = LabMutedText
                            )
                            Button(
                                onClick = { state.purchasePro(activity) },
                                enabled = !state.isPurchasingPro,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("setup-unlock-pro-button")
                            ) {
                                Icon(Icons.Filled.LockOpen, contentDescription = null)
                                Text(if (state.isPurchasingPro) "Opening Google Play..." else "Unlock Pro")
                            }
                            OutlinedButton(
                                onClick = state::restoreProPurchase,
                                enabled = !state.isRestoringPro,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("setup-restore-pro-purchase-button")
                            ) {
                                Icon(Icons.Filled.LockOpen, contentDescription = null)
                                Text(if (state.isRestoringPro) "Checking Google Play..." else "Restore Purchase")
                            }
                            OutlinedTextField(
                                value = promoCodeText,
                                onValueChange = { promoCodeText = it },
                                label = { Text("Google Play promo code") },
                                singleLine = true,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("setup-pro-promo-code-input")
                            )
                            OutlinedButton(
                                onClick = { state.unlockProWithPromoCode(activity, promoCodeText) },
                                enabled = promoCodeText.isNotBlank(),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("setup-unlock-pro-with-promo-code-button")
                            ) {
                                Icon(Icons.Filled.LockOpen, contentDescription = null)
                                Text("Redeem Google Play Promo Code")
                            }
                            state.storageMessage?.let { message ->
                                Text(
                                    message,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    }
                }
            }

            item {
                AssumptionCard("Scenario status") {
                    if (hasUnsavedChanges) {
                        Text(
                            text = "Unsaved setup changes",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                    state.lastRunMessage?.let { message ->
                        Text(message, style = MaterialTheme.typography.bodySmall, color = LabMutedText)
                    }
                    KeyValueRow("Current account total", scenario.accounts.total.asCurrency())
                }
            }

            item {
                ScenarioWarningCard(title = "Assumption checks", warnings = warnings)
            }

            item {
                AssumptionCard("Household profile") {
                    NumberField(
                        label = "Current age",
                        value = form.currentAge,
                        modifier = Modifier.testTag("current-age-input")
                    ) {
                        form = form.copy(currentAge = it)
                    }
                    NumberField("Retirement age", form.retirementAge) {
                        form = if (isProUnlocked) {
                            form.withRetirementAgeDefaults(it)
                        } else {
                            form.copy(retirementAge = it)
                        }
                    }
                    ChoiceGroup(
                        title = "Filing status",
                        options = FilingStatus.entries.toList(),
                        selected = form.filingStatus,
                        label = { it.label },
                        onSelected = {
                            form = form.copy(
                                filingStatus = it,
                                spouseCurrentAge = if (it == FilingStatus.Married && form.spouseCurrentAge.isBlank()) {
                                    form.currentAge
                                } else {
                                    form.spouseCurrentAge
                                }
                            )
                        }
                    )
                    if (form.filingStatus == FilingStatus.Married) {
                        NumberField("Spouse current age", form.spouseCurrentAge) {
                            form = form.copy(spouseCurrentAge = it)
                        }
                        Text(
                            text = "The spouse is modeled as the opposite gender with no separate earned or Social Security income. The simulation continues until both spouses have died or the household runs out of money.",
                            style = MaterialTheme.typography.bodySmall,
                            color = LabMutedText
                        )
                    }
                    ChoiceGroup(
                        title = "Sex for mortality calculations",
                        options = Gender.entries.toList(),
                        selected = form.gender,
                        label = { it.label },
                        onSelected = { form = form.copy(gender = it) }
                    )
                }
            }

            item {
                AssumptionCard("Spending") {
                    MoneyField("Annual base spending", form.annualSpending) { form = form.copy(annualSpending = it) }
                    Text(
                        text = "Use Budget to estimate this from regular bills, credit cards, cash spending, property taxes, and insurance. Mortgage, rent, and healthcare remain separate inputs.",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    PercentField(
                        label = "Max spending cut",
                        value = form.lowPortfolioSpendingReduction,
                        enabled = isProUnlocked
                    ) {
                        form = form.copy(lowPortfolioSpendingReduction = it)
                    }
                    Text(
                        text = "Applied if portfolio balance falls below 50% of the balance at retirement.",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    if (!isProUnlocked) {
                        ProLockedInlineNotice(
                            title = "Adaptive spending is Pro",
                            detail = "Free runs use fixed spending. Pro unlocks modeled spending cuts when the portfolio is under pressure."
                        )
                    }
                    ChoiceGroup(
                        title = "Spending path",
                        options = SpendingPathModel.entries.toList(),
                        selected = form.spendingPathModel,
                        label = { it.label },
                        onSelected = { form = form.copy(spendingPathModel = it) }
                    )
                    Text(
                        text = spendingPathHelpText(form.spendingPathModel),
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    RatePairFields(
                        title = "General inflation",
                        averageValue = form.generalInflationMean,
                        swingValue = form.generalInflationStdDev,
                        onAverageChange = { form = form.copy(generalInflationMean = it) },
                        onSwingChange = { form = form.copy(generalInflationStdDev = it) }
                    )
                    Text(
                        text = "Used for non-healthcare living costs. Healthcare inflation is modeled separately.",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    RestoreDefaultsButton(
                        testTag = "restore-general-inflation-defaults",
                        onClick = {
                            form = form.copy(
                                generalInflationMean = DEFAULT_GENERAL_INFLATION_MEAN.percentInputText(),
                                generalInflationStdDev = DEFAULT_GENERAL_INFLATION_STD_DEV.percentInputText()
                            )
                        }
                    )
                }
            }

            item {
                AssumptionCard("Housing") {
                    MoneyField("Monthly mortgage payment", form.mortgagePayment) {
                        form = form.copy(mortgagePayment = it)
                    }
                    Text(
                        text = "Enter principal and interest only. Do not include escrow for property taxes or homeowners insurance; enter those in the Budget page.",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    MortgageDurationFields(
                        yearsValue = form.mortgageYearsLeft,
                        monthsValue = form.mortgageMonthsLeft,
                        onYearsChange = { form = form.copy(mortgageYearsLeft = it) },
                        onMonthsChange = { form = form.copy(mortgageMonthsLeft = it) }
                    )
                    MoneyField("Current mortgage balance", form.mortgageBalance) {
                        form = form.copy(mortgageBalance = it)
                    }
                    MoneyField("Current home value", form.homeValue) {
                        form = form.copy(homeValue = it)
                    }
                    MoneyField("Monthly rent", form.monthlyRent) {
                        form = form.copy(monthlyRent = it)
                    }
                    Text(
                        text = "If home value is entered, the model assumes the home can be sold if portfolio assets run out. Sale proceeds use net equity after the remaining mortgage balance. Enter zero for costs or assets that do not apply.",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                }
            }

            item {
                AssumptionCard("Healthcare and long-term care") {
                    MoneyField("Pre-Medicare monthly premium", form.preMedicareMonthlyPremium) {
                        form = form.copy(preMedicareMonthlyPremium = it)
                    }
                    Text(
                        text = "Monthly premium per pre-Medicare adult for years before age 65. Default is based on a typical ACA Marketplace Silver plan; married households apply this amount to each spouse while that spouse is under 65.",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    RestoreDefaultsButton(
                        label = "Restore Default",
                        testTag = "restore-pre-medicare-premium-default",
                        onClick = {
                            form = form.copy(
                                preMedicareMonthlyPremium = DEFAULT_PRE_MEDICARE_MONTHLY_PREMIUM.wholeDollarText()
                            )
                        }
                    )
                    RatePairFields(
                        title = "Healthcare inflation",
                        averageValue = form.healthcareInflationMean,
                        swingValue = form.healthcareInflationStdDev,
                        enabled = isProUnlocked,
                        onAverageChange = { form = form.copy(healthcareInflationMean = it) },
                        onSwingChange = { form = form.copy(healthcareInflationStdDev = it) }
                    )
                    RestoreDefaultsButton(
                        testTag = "restore-healthcare-inflation-defaults",
                        enabled = isProUnlocked,
                        onClick = {
                            form = form.copy(
                                healthcareInflationMean = DEFAULT_HEALTHCARE_INFLATION_MEAN.percentInputText(),
                                healthcareInflationStdDev = DEFAULT_HEALTHCARE_INFLATION_STD_DEV.percentInputText()
                            )
                        }
                    )
                    if (!isProUnlocked) {
                        ProLockedInlineNotice(
                            title = "Healthcare stress assumptions are Pro",
                            detail = "Free runs use default healthcare inflation. Pro unlocks custom healthcare inflation and long-term care stress settings."
                        )
                    }
                    AssumptionToggleRow(
                        title = "Long-term care shock",
                        checked = form.longTermCareEnabled,
                        enabled = isProUnlocked,
                        onCheckedChange = { form = form.copy(longTermCareEnabled = it) }
                    )
                    MoneyField(
                        label = "Long-term care annual cost",
                        value = form.longTermCareAnnualCost,
                        enabled = isProUnlocked
                    ) {
                        form = form.copy(longTermCareAnnualCost = it)
                    }
                    NumberField(
                        label = "Long-term care duration years",
                        value = form.longTermCareDurationYears,
                        enabled = isProUnlocked
                    ) {
                        form = form.copy(longTermCareDurationYears = it)
                    }
                }
            }

            item {
                AssumptionCard("Accounts") {
                    MoneyField("Pre-tax accounts", form.pretax) { form = form.copy(pretax = it) }
                    AccountFieldHelp(
                        "Traditional 401(k), 403(b), 457(b), traditional IRA, rollover IRA, SEP/SIMPLE IRA, traditional TSP, and employer match or profit-sharing balances."
                    )
                    MoneyField("Roth accounts", form.roth) { form = form.copy(roth = it) }
                    AccountFieldHelp(
                        "Roth IRA, Roth 401(k), Roth 403(b), Roth TSP, and other retirement balances expected to be tax-free when qualified."
                    )
                    MoneyField("Taxable investments", form.taxable) { form = form.copy(taxable = it) }
                    AccountFieldHelp(
                        "Individual, joint, or trust brokerage investments held outside retirement accounts, including stocks, ETFs, mutual funds, bonds, vested company stock, and non-retirement Treasuries or brokered CDs if you want them modeled as investments."
                    )
                    MoneyField("Cash reserve", form.cash) { form = form.copy(cash = it) }
                    AccountFieldHelp(
                        "Checking, savings, money market, short-term CDs, Treasury bills, or other cash-like reserves intended for near-term spending or drawdowns."
                    )
                }
            }

            item {
                AssumptionCard("Market returns") {
                    RatePairFields(
                        title = "Current portfolio returns",
                        averageValue = form.preRetirementMeanReturn,
                        swingValue = form.preRetirementStdDev,
                        onAverageChange = { form = form.copy(preRetirementMeanReturn = it) },
                        onSwingChange = { form = form.copy(preRetirementStdDev = it) }
                    )
                    RatePairFields(
                        title = "Post-retirement stock returns",
                        averageValue = form.stockMeanReturn,
                        swingValue = form.stockStdDev,
                        onAverageChange = { form = form.copy(stockMeanReturn = it) },
                        onSwingChange = { form = form.copy(stockStdDev = it) }
                    )
                    Text(
                        text = STOCK_RETURN_DESCRIPTION,
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    RatePairFields(
                        title = "Post-retirement bond returns",
                        averageValue = form.bondMeanReturn,
                        swingValue = form.bondStdDev,
                        onAverageChange = { form = form.copy(bondMeanReturn = it) },
                        onSwingChange = { form = form.copy(bondStdDev = it) }
                    )
                    RestoreDefaultsButton(
                        testTag = "restore-market-returns-defaults",
                        onClick = {
                            val defaults = MarketAssumptions()
                            form = form.copy(
                                preRetirementMeanReturn = defaults.preRetirementMeanReturn.percentInputText(),
                                preRetirementStdDev = defaults.preRetirementStdDev.percentInputText(),
                                stockMeanReturn = defaults.stockMeanReturn.percentInputText(),
                                stockStdDev = defaults.stockStdDev.percentInputText(),
                                bondMeanReturn = defaults.bondMeanReturn.percentInputText(),
                                bondStdDev = defaults.bondStdDev.percentInputText()
                            )
                        }
                    )
                    PostRetirementAllocationSliders(
                        allocation = if (isProUnlocked) {
                            form.postRetirementAllocation
                        } else {
                            PostRetirementAllocationStrategy()
                        },
                        enabled = isProUnlocked,
                        onAllocationChange = { form = form.copy(postRetirementAllocation = it) }
                    )
                    if (!isProUnlocked) {
                        ProLockedInlineNotice(
                            title = "Investment ratio customization is Pro",
                            detail = "Free uses the default post-retirement stock and bond ratios."
                        )
                    }
                    RestoreDefaultsButton(
                        testTag = "restore-setup-post-retirement-allocation-defaults",
                        enabled = isProUnlocked,
                        onClick = {
                            form = form.copy(postRetirementAllocation = PostRetirementAllocationStrategy())
                        }
                    )
                }
            }

            item {
                AssumptionCard("Guaranteed income") {
                    MoneyField("Annual guaranteed income", form.guaranteedAnnualIncome) {
                        form = form.copy(guaranteedAnnualIncome = it)
                    }
                    NumberField("Income start age", form.guaranteedIncomeStartAge) {
                        form = form.copy(guaranteedIncomeStartAge = it)
                    }
                    PercentField("Annual income increase", form.guaranteedIncomeAnnualIncrease) {
                        form = form.copy(guaranteedIncomeAnnualIncrease = it)
                    }
                    PercentField("Survivor benefit", form.guaranteedIncomeSurvivorPercent) {
                        form = form.copy(guaranteedIncomeSurvivorPercent = it)
                    }
                    Text(
                        text = "Use this for pensions, annuities, and other guaranteed household income. Survivor benefit is the share that continues if the retiree dies before a spouse.",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                }
            }

            item {
                AssumptionCard("Social Security") {
                    MoneyField("Primary Social Security at FRA", form.socialSecurityAt67) {
                        form = form.copy(socialSecurityAt67 = it)
                    }
                    Text(
                        text = "Primary full retirement age: ${socialSecurityFullRetirementAgeText(form.currentAge, scenario.household.currentAge)}. Benefits use SSA early/delayed claiming factors and are indexed with general inflation.",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    NumberField("Primary Social Security claim age", form.claimAge) {
                        form = form.copy(claimAge = it)
                    }
                    if (form.filingStatus == FilingStatus.Married) {
                        NumberField("Spouse benefit claim age", form.spouseClaimAge) {
                            form = form.copy(spouseClaimAge = it)
                        }
                        Text(
                            text = "Spouse benefits are modeled from the primary worker record. While both are alive, the spouse may receive up to 50% of the primary FRA benefit. After the primary dies, survivor benefits can replace the spouse benefit.",
                            style = MaterialTheme.typography.bodySmall,
                            color = LabMutedText
                        )
                    }
                }
            }

            item {
                AssumptionCard("Tax and drawdown strategy") {
                    AssumptionToggleRow(
                        title = "Roth conversion lab",
                        checked = isProUnlocked && form.rothConversionEnabled,
                        enabled = isProUnlocked,
                        onCheckedChange = { form = form.copy(rothConversionEnabled = it) }
                    )
                    ChoiceGroup(
                        title = "Roth marginal bracket cap",
                        options = TAX_BRACKET_CAPS,
                        selected = form.rothBracketCap,
                        enabled = isProUnlocked,
                        label = { it.percentOptionLabel() },
                        onSelected = { form = form.copy(rothBracketCap = it) }
                    )
                    AssumptionToggleRow(
                        title = "Apply 10% early withdrawal tax",
                        checked = isProUnlocked && form.applyEarlyWithdrawalPenalty,
                        enabled = isProUnlocked,
                        onCheckedChange = { form = form.copy(applyEarlyWithdrawalPenalty = it) }
                    )
                    AssumptionToggleRow(
                        title = "Rule of 55 eligible",
                        checked = isProUnlocked && form.ruleOf55Eligible,
                        enabled = isProUnlocked,
                        onCheckedChange = { form = form.copy(ruleOf55Eligible = it) }
                    )
                    AssumptionToggleRow(
                        title = "72(t) / SEPP eligible",
                        checked = isProUnlocked && form.seppEligible,
                        enabled = isProUnlocked,
                        onCheckedChange = { form = form.copy(seppEligible = it) }
                    )
                    if (!isProUnlocked) {
                        KeyValueRow(
                            "Free early withdrawal tax",
                            automaticEarlyWithdrawalPenaltyLabel(form.retirementAge)
                        )
                    }
                    Text(
                        text = "The early withdrawal tax applies to modeled pre-tax withdrawals before age 59 1/2. Rule of 55 only applies to qualifying employer-plan distributions after separation in or after the year you turn 55. SEPP uses a fixed amortization payment from modeled pre-tax assets.",
                        style = MaterialTheme.typography.bodySmall,
                        color = LabMutedText
                    )
                    if (!isProUnlocked) {
                        ProLockedInlineNotice(
                            title = "Advanced tax strategy is Pro",
                            detail = "Free runs use basic early-withdrawal defaults. Pro unlocks Roth conversion, Rule of 55, and 72(t) / SEPP assumptions."
                        )
                    }
                    if (SHOW_CASH_RESERVE_DRAWDOWN_SETTINGS) {
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
            }

        item {
            AssumptionCard("Simulation settings") {
                NumberField("Number of simulations", form.numberOfSimulations) {
                    form = form.copy(numberOfSimulations = it)
                }
                Text(
                    text = if (isProUnlocked) {
                        "Pro supports up to $PRO_SIMULATION_LIMIT simulations."
                    } else {
                        "Free is limited to $FREE_SIMULATION_LIMIT simulations. Pro unlocks up to $PRO_SIMULATION_LIMIT simulations."
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = LabMutedText
                )
                NumberField("Random seed", form.seed) { form = form.copy(seed = it) }
                KeyValueRow("Federal tax table", "2024 brackets")
                KeyValueRow("Medicare model", MedicarePremiums.PREMIUM_TABLE_VERSION)
                KeyValueRow("Projection horizon", "Mortality cap age $DEFAULT_PROJECTION_END_AGE per person")
                KeyValueRow("Engine cadence", "Monthly cashflow model")
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

        Surface(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth(),
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 3.dp,
            shadowElevation = 6.dp
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                validationMessage?.let { message ->
                    Text(
                        text = message,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error
                    )
                }
                Button(
                    onClick = runCurrentSetup,
                    enabled = !state.isRunning,
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("apply-setup-button")
                ) {
                    Text(if (state.isRunning) "Running..." else "Run Current Scenario")
                }
            }
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
private fun MoneyField(
    label: String,
    value: String,
    enabled: Boolean = true,
    onValueChange: (String) -> Unit
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        enabled = enabled,
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
private fun NumberField(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    onValueChange: (String) -> Unit
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        enabled = enabled,
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = modifier.fillMaxWidth()
    )
}

@Composable
private fun PercentField(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    onValueChange: (String) -> Unit
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text("$label (%)") },
        enabled = enabled,
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text),
        modifier = modifier.fillMaxWidth()
    )
}

@Composable
private fun RatePairFields(
    title: String,
    averageValue: String,
    swingValue: String,
    enabled: Boolean = true,
    onAverageChange: (String) -> Unit,
    onSwingChange: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(title, style = MaterialTheme.typography.labelLarge, color = LabMutedText)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            PercentField(
                label = "Average",
                value = averageValue,
                modifier = Modifier.weight(1f),
                enabled = enabled,
                onValueChange = onAverageChange
            )
            PercentField(
                label = "Std Dev",
                value = swingValue,
                modifier = Modifier.weight(1f),
                enabled = enabled,
                onValueChange = onSwingChange
            )
        }
    }
}

@Composable
private fun RestoreDefaultsButton(
    label: String = "Restore Defaults",
    testTag: String,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.End
    ) {
        OutlinedButton(
            onClick = onClick,
            enabled = enabled,
            modifier = Modifier.testTag(testTag)
        ) {
            Text(label)
        }
    }
}

@Composable
private fun PostRetirementAllocationSliders(
    allocation: PostRetirementAllocationStrategy,
    enabled: Boolean = true,
    onAllocationChange: (PostRetirementAllocationStrategy) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            "Post-retirement investment ratios",
            style = MaterialTheme.typography.labelLarge,
            color = LabMutedText
        )
        Text(
            "Each slider controls the stock share; the remainder is bonds.",
            style = MaterialTheme.typography.bodySmall,
            color = LabMutedText
        )
        PostRetirementAllocationTier.entries.forEach { tier ->
            SetupAllocationSliderRow(
                tier = tier,
                stockAllocation = allocation.stockAllocationFor(tier),
                enabled = enabled,
                onStockAllocationChange = { value ->
                    onAllocationChange(allocation.withStockAllocation(tier, value))
                }
            )
        }
    }
}

@Composable
private fun SetupAllocationSliderRow(
    tier: PostRetirementAllocationTier,
    stockAllocation: Double,
    enabled: Boolean = true,
    onStockAllocationChange: (Double) -> Unit
) {
    val stockPercent = stockAllocation.coerceIn(0.0, 1.0)
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                tier.label,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.weight(1f)
            )
            Text(
                setupAllocationLabel(stockPercent),
                style = MaterialTheme.typography.labelMedium,
                color = LabMutedText
            )
        }
        Slider(
            value = stockPercent.toFloat(),
            onValueChange = { value ->
                onStockAllocationChange(((value * 20f).roundToInt() / 20.0).coerceIn(0.0, 1.0))
            },
            enabled = enabled,
            valueRange = 0f..1f,
            steps = 19,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("setup-allocation-slider-${tier.name.accessibilityTagSuffix()}")
        )
    }
}

@Composable
private fun MortgageDurationFields(
    yearsValue: String,
    monthsValue: String,
    onYearsChange: (String) -> Unit,
    onMonthsChange: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Mortgage time left", style = MaterialTheme.typography.labelLarge, color = LabMutedText)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            NumberField(
                label = "Years left",
                value = yearsValue,
                modifier = Modifier.weight(1f),
                onValueChange = onYearsChange
            )
            NumberField(
                label = "Months left",
                value = monthsValue,
                modifier = Modifier.weight(1f),
                onValueChange = onMonthsChange
            )
        }
    }
}

@Composable
private fun AccountFieldHelp(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.bodySmall,
        color = LabMutedText
    )
}

@Composable
private fun AssumptionToggleRow(
    title: String,
    checked: Boolean,
    enabled: Boolean = true,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(title, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
        Switch(checked = checked, enabled = enabled, onCheckedChange = onCheckedChange)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun <T> ChoiceGroup(
    title: String,
    options: List<T>,
    selected: T,
    enabled: Boolean = true,
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
                    Button(onClick = { onSelected(option) }, enabled = enabled) {
                        Text(text)
                    }
                } else {
                    OutlinedButton(onClick = { onSelected(option) }, enabled = enabled) {
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
    val filingStatus: FilingStatus,
    val gender: Gender,
    val spouseCurrentAge: String,
    val annualSpending: String,
    val generalInflationMean: String,
    val generalInflationStdDev: String,
    val spendingPathModel: SpendingPathModel,
    val lowPortfolioSpendingReduction: String,
    val pretax: String,
    val roth: String,
    val taxable: String,
    val cash: String,
    val preMedicareMonthlyPremium: String,
    val healthcareInflationMean: String,
    val healthcareInflationStdDev: String,
    val mortgagePayment: String,
    val mortgageYearsLeft: String,
    val mortgageMonthsLeft: String,
    val mortgageBalance: String,
    val homeValue: String,
    val monthlyRent: String,
    val socialSecurityAt67: String,
    val claimAge: String,
    val spouseClaimAge: String,
    val guaranteedAnnualIncome: String,
    val guaranteedIncomeStartAge: String,
    val guaranteedIncomeAnnualIncrease: String,
    val guaranteedIncomeSurvivorPercent: String,
    val preRetirementMeanReturn: String,
    val preRetirementStdDev: String,
    val stockMeanReturn: String,
    val stockStdDev: String,
    val bondMeanReturn: String,
    val bondStdDev: String,
    val postRetirementAllocation: PostRetirementAllocationStrategy,
    val rothConversionEnabled: Boolean,
    val rothBracketCap: Double,
    val applyEarlyWithdrawalPenalty: Boolean,
    val ruleOf55Eligible: Boolean,
    val seppEligible: Boolean,
    val useCashReserveDuringDrawdowns: Boolean,
    val drawdownTrigger: String,
    val longTermCareEnabled: Boolean,
    val longTermCareAnnualCost: String,
    val longTermCareDurationYears: String,
    val numberOfSimulations: String,
    val seed: String
) {
    fun toScenario(base: RetirementScenario, featureAccess: FeatureAccess): ParsedAssumptions {
        val currentAge = currentAge.toIntOrNull()
        val retirementAge = retirementAge.toIntOrNull()
        val spouseCurrentAge = spouseCurrentAge.toIntOrNull()
        val annualSpending = parseMoney(annualSpending)
        val generalInflationMean = parsePercent(generalInflationMean)
        val generalInflationStdDev = parsePercent(generalInflationStdDev)
        val lowPortfolioSpendingReduction = parsePercent(lowPortfolioSpendingReduction)
        val pretax = parseMoney(pretax)
        val roth = parseMoney(roth)
        val taxable = parseMoney(taxable)
        val cash = parseMoney(cash)
        val preMedicareMonthlyPremium = parseMoney(preMedicareMonthlyPremium)
        val healthcareInflationMean = parsePercent(healthcareInflationMean)
        val healthcareInflationStdDev = parsePercent(healthcareInflationStdDev)
        val mortgagePayment = parseMoney(mortgagePayment)
        val mortgageYearsLeft = mortgageYearsLeft.toIntOrNull()
        val mortgageMonthsLeft = mortgageMonthsLeft.toIntOrNull()
        val mortgageBalance = parseMoney(mortgageBalance)
        val homeValue = parseMoney(homeValue)
        val monthlyRent = parseMoney(monthlyRent)
        val socialSecurityAt67 = parseMoney(socialSecurityAt67)
        val claimAge = claimAge.toIntOrNull()
        val spouseClaimAge = spouseClaimAge.toIntOrNull()
        val guaranteedAnnualIncome = parseMoney(guaranteedAnnualIncome)
        val guaranteedIncomeStartAge = guaranteedIncomeStartAge.toIntOrNull()
        val guaranteedIncomeAnnualIncrease = parsePercent(guaranteedIncomeAnnualIncrease)
        val guaranteedIncomeSurvivorPercent = parsePercent(guaranteedIncomeSurvivorPercent)
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
            if (filingStatus == FilingStatus.Married) {
                requireInt("Spouse current age", spouseCurrentAge, 18, 100)
            } else {
                null
            },
            requireMoney("Annual base spending", annualSpending),
            requirePercent("General inflation average", generalInflationMean, -0.02, 0.15),
            requirePercent("General inflation Std Dev", generalInflationStdDev, 0.0, 0.30),
            requirePercent("Spending reduction below 50% portfolio", lowPortfolioSpendingReduction, 0.0, 1.0),
            requireMoney("Pre-tax accounts", pretax),
            requireMoney("Roth accounts", roth),
            requireMoney("Taxable investments", taxable),
            requireMoney("Cash reserve", cash),
            requireMoney("Pre-Medicare monthly premium", preMedicareMonthlyPremium),
            requirePercent("Healthcare inflation average", healthcareInflationMean, 0.0, 0.20),
            requirePercent("Healthcare inflation Std Dev", healthcareInflationStdDev, 0.0, 0.30),
            requireMoney("Mortgage payment", mortgagePayment),
            requireInt("Mortgage years left", mortgageYearsLeft, 0, 80),
            requireInt("Mortgage months left", mortgageMonthsLeft, 0, 11),
            requireMoney("Current mortgage balance", mortgageBalance),
            requireMoney("Current home value", homeValue),
            requireMoney("Monthly rent", monthlyRent),
            requireMoney("Primary Social Security at FRA", socialSecurityAt67),
            requireInt("Primary Social Security claim age", claimAge, 62, 70),
            requireInt("Spouse Social Security benefit claim age", spouseClaimAge, 60, 70),
            requireMoney("Annual guaranteed income", guaranteedAnnualIncome),
            requireInt("Guaranteed income start age", guaranteedIncomeStartAge, 18, DEFAULT_PROJECTION_END_AGE),
            requirePercent("Guaranteed income annual increase", guaranteedIncomeAnnualIncrease, -0.02, 0.15),
            requirePercent("Guaranteed income survivor benefit", guaranteedIncomeSurvivorPercent, 0.0, 1.0),
            requirePercent("Current portfolio returns average", preRetirementMeanReturn, -0.20, 0.25),
            requirePercent("Current portfolio returns Std Dev", preRetirementStdDev, 0.0, 0.60),
            requirePercent("Post-retirement stock returns average", stockMeanReturn, -0.20, 0.25),
            requirePercent("Post-retirement stock returns Std Dev", stockStdDev, 0.0, 0.60),
            requirePercent("Post-retirement bond returns average", bondMeanReturn, -0.20, 0.20),
            requirePercent("Post-retirement bond returns Std Dev", bondStdDev, 0.0, 0.40),
            requirePercent("Drawdown trigger", drawdownTrigger, -0.50, 0.25),
            requireMoney("Long-term care annual cost", longTermCareAnnualCost),
            requireInt("Long-term care duration years", longTermCareDurationYears, 1, 10),
            requireSimulationCount(numberOfSimulations, featureAccess),
            if (seed == null) "Random seed must be a whole number." else null
        ).firstOrNull()

        if (firstError != null) return ParsedAssumptions(error = firstError)

        val updated = base.copy(
            household = HouseholdProfile(
                currentAge = currentAge!!,
                retirementAge = retirementAge!!,
                targetEndAge = DEFAULT_PROJECTION_END_AGE,
                filingStatus = filingStatus,
                gender = gender,
                spouseCurrentAge = spouseCurrentAge ?: currentAge
            ),
            spending = SpendingPlan(
                annualBaseSpending = annualSpending!!,
                generalInflationMean = generalInflationMean!!,
                generalInflationStdDev = generalInflationStdDev!!,
                spendingPathModel = spendingPathModel,
                lowPortfolioSpendingReduction = lowPortfolioSpendingReduction!!
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
                includeMedicarePremiums = true
            ),
            mortgage = MortgagePlan(
                monthlyPayment = mortgagePayment!!,
                yearsLeft = mortgageYearsLeft!!,
                monthsLeft = mortgageMonthsLeft!!,
                currentBalance = mortgageBalance!!
            ),
            rent = RentPlan(monthlyRent = monthlyRent!!),
            home = HomePlan(currentValue = homeValue!!),
            socialSecurity = SocialSecurityPlan(
                annualBenefitAt67 = socialSecurityAt67!!,
                claimAge = claimAge!!,
                spouseClaimAge = spouseClaimAge!!
            ),
            guaranteedIncome = GuaranteedIncomePlan(
                annualIncome = guaranteedAnnualIncome!!,
                startAge = guaranteedIncomeStartAge!!,
                annualIncrease = guaranteedIncomeAnnualIncrease!!,
                survivorPercent = guaranteedIncomeSurvivorPercent!!
            ),
            market = MarketAssumptions(
                preRetirementMeanReturn = preRetirementMeanReturn!!,
                preRetirementStdDev = preRetirementStdDev!!,
                stockMeanReturn = stockMeanReturn!!,
                stockStdDev = stockStdDev!!,
                bondMeanReturn = bondMeanReturn!!,
                bondStdDev = bondStdDev!!
            ),
            postRetirementAllocation = postRetirementAllocation,
            rothConversion = RothConversionStrategy(
                enabled = rothConversionEnabled,
                marginalRateCap = rothBracketCap
            ),
            withdrawalStrategy = WithdrawalStrategy(
                useCashReserveDuringDrawdowns = SHOW_CASH_RESERVE_DRAWDOWN_SETTINGS && useCashReserveDuringDrawdowns,
                drawdownTrigger = drawdownTrigger!!,
                applyEarlyWithdrawalPenalty = applyEarlyWithdrawalPenalty,
                ruleOf55Eligible = ruleOf55Eligible,
                seppEligible = seppEligible
            ),
            longTermCare = LongTermCareAssumption(
                enabled = longTermCareEnabled,
                annualCost = longTermCareAnnualCost!!,
                averageDurationYears = longTermCareDurationYears!!
            ),
            numberOfSimulations = numberOfSimulations!!,
            seed = seed!!
        ).forFeatureAccess(featureAccess)

        val modelError = updated.validate().firstOrNull()
        return if (modelError != null) {
            ParsedAssumptions(error = modelError)
        } else {
            ParsedAssumptions(scenario = updated)
        }
    }

    fun withRetirementAgeDefaults(value: String): EditableAssumptions {
        val retirementAge = value.toIntOrNull() ?: return copy(retirementAge = value)
        val defaults = WithdrawalStrategy.defaultsForRetirementAge(retirementAge)
        return copy(
            retirementAge = value,
            applyEarlyWithdrawalPenalty = defaults.applyEarlyWithdrawalPenalty,
            ruleOf55Eligible = defaults.ruleOf55Eligible,
            seppEligible = defaultSeppEligible(retirementAge)
        )
    }

    private fun defaultSeppEligible(retirementAge: Int): Boolean {
        if (retirementAge >= RULE_OF_55_MINIMUM_RETIREMENT_AGE) return false
        if (retirementAge * 12 >= PENALTY_FREE_WITHDRAWAL_AGE_MONTHS) return false

        val pretaxBalance = parseMoney(pretax) ?: return false
        if (pretaxBalance <= 0.0) return false

        val annualBridgeNeed = estimatedAnnualBridgeNeed(retirementAge)
        if (annualBridgeNeed <= 0.0) return false

        val bridgeMonths = (PENALTY_FREE_WITHDRAWAL_AGE_MONTHS - retirementAge * 12).coerceAtLeast(0)
        val bridgeNeed = annualBridgeNeed * bridgeMonths.toDouble() / 12.0
        val nonPenaltyBridgeAssets =
            (parseMoney(cash) ?: 0.0) +
                (parseMoney(taxable) ?: 0.0) +
                (parseMoney(roth) ?: 0.0)
        if (bridgeNeed <= nonPenaltyBridgeAssets) return false

        return RetirementSimulator.seppFixedAmortizationAnnualPayment(
            accountBalance = pretaxBalance,
            age = retirementAge
        ) > 0.0
    }

    private fun estimatedAnnualBridgeNeed(retirementAge: Int): Double {
        val spending = parseMoney(annualSpending) ?: return 0.0
        val healthcare = (parseMoney(preMedicareMonthlyPremium) ?: 0.0) * 12.0
        val mortgage = (parseMoney(mortgagePayment) ?: 0.0) * 12.0
        val rent = (parseMoney(monthlyRent) ?: 0.0) * 12.0
        val guaranteedIncomeStart = guaranteedIncomeStartAge.toIntOrNull() ?: Int.MAX_VALUE
        val guaranteed = if (guaranteedIncomeStart <= retirementAge) {
            parseMoney(guaranteedAnnualIncome) ?: 0.0
        } else {
            0.0
        }
        return (spending + healthcare + mortgage + rent - guaranteed).coerceAtLeast(0.0)
    }

    companion object {
        fun from(scenario: RetirementScenario): EditableAssumptions {
            return EditableAssumptions(
                currentAge = scenario.household.currentAge.toString(),
                retirementAge = scenario.household.retirementAge.toString(),
                filingStatus = scenario.household.filingStatus,
                gender = scenario.household.gender,
                spouseCurrentAge = scenario.household.spouseCurrentAge.toString(),
                annualSpending = scenario.spending.annualBaseSpending.wholeDollarText(),
                generalInflationMean = scenario.spending.generalInflationMean.percentInputText(),
                generalInflationStdDev = scenario.spending.generalInflationStdDev.percentInputText(),
                spendingPathModel = scenario.spending.spendingPathModel,
                lowPortfolioSpendingReduction = scenario.spending.lowPortfolioSpendingReduction.percentInputText(),
                pretax = scenario.accounts.pretax.wholeDollarText(),
                roth = scenario.accounts.roth.wholeDollarText(),
                taxable = scenario.accounts.taxable.wholeDollarText(),
                cash = scenario.accounts.cash.wholeDollarText(),
                preMedicareMonthlyPremium = scenario.healthcare.preMedicareMonthlyPremium.wholeDollarText(),
                healthcareInflationMean = scenario.healthcare.healthcareInflationMean.percentInputText(),
                healthcareInflationStdDev = scenario.healthcare.healthcareInflationStdDev.percentInputText(),
                mortgagePayment = scenario.mortgage.monthlyPayment.wholeDollarText(),
                mortgageYearsLeft = scenario.mortgage.yearsLeft.toString(),
                mortgageMonthsLeft = scenario.mortgage.monthsLeft.toString(),
                mortgageBalance = scenario.mortgage.currentBalance.wholeDollarText(),
                homeValue = scenario.home.currentValue.wholeDollarText(),
                monthlyRent = scenario.rent.monthlyRent.wholeDollarText(),
                socialSecurityAt67 = scenario.socialSecurity.annualBenefitAt67.wholeDollarText(),
                claimAge = scenario.socialSecurity.claimAge.toString(),
                spouseClaimAge = scenario.socialSecurity.spouseClaimAge.toString(),
                guaranteedAnnualIncome = scenario.guaranteedIncome.annualIncome.wholeDollarText(),
                guaranteedIncomeStartAge = scenario.guaranteedIncome.startAge.toString(),
                guaranteedIncomeAnnualIncrease = scenario.guaranteedIncome.annualIncrease.percentInputText(),
                guaranteedIncomeSurvivorPercent = scenario.guaranteedIncome.survivorPercent.percentInputText(),
                preRetirementMeanReturn = scenario.market.preRetirementMeanReturn.percentInputText(),
                preRetirementStdDev = scenario.market.preRetirementStdDev.percentInputText(),
                stockMeanReturn = scenario.market.stockMeanReturn.percentInputText(),
                stockStdDev = scenario.market.stockStdDev.percentInputText(),
                bondMeanReturn = scenario.market.bondMeanReturn.percentInputText(),
                bondStdDev = scenario.market.bondStdDev.percentInputText(),
                postRetirementAllocation = scenario.postRetirementAllocation,
                rothConversionEnabled = scenario.rothConversion.enabled,
                rothBracketCap = closestTaxCap(scenario.rothConversion.marginalRateCap),
                applyEarlyWithdrawalPenalty = scenario.withdrawalStrategy.applyEarlyWithdrawalPenalty,
                ruleOf55Eligible = scenario.withdrawalStrategy.ruleOf55Eligible,
                seppEligible = scenario.withdrawalStrategy.seppEligible,
                useCashReserveDuringDrawdowns = SHOW_CASH_RESERVE_DRAWDOWN_SETTINGS &&
                    scenario.withdrawalStrategy.useCashReserveDuringDrawdowns,
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
private const val STOCK_RETURN_DESCRIPTION = "Default 13.3% ± 16.2% values based on the last 50 years of the S&P 500"

private fun closestTaxCap(value: Double): Double {
    return TAX_BRACKET_CAPS.minBy { kotlin.math.abs(it - value) }
}

private fun Double.percentOptionLabel(): String {
    return "${String.format(Locale.US, "%.0f", this * 100.0)}%"
}

private fun setupAllocationLabel(stockAllocation: Double): String {
    val stockPercent = stockAllocation.coerceIn(0.0, 1.0)
    val bondPercent = 1.0 - stockPercent
    return "${stockPercent.percentOptionLabel()} stocks / ${bondPercent.percentOptionLabel()} bonds"
}

private fun String.accessibilityTagSuffix(): String {
    return lowercase()
        .replace(Regex("[^a-z0-9]+"), "-")
        .trim('-')
}

private fun spendingPathHelpText(model: SpendingPathModel): String {
    return when (model) {
        SpendingPathModel.Flat -> "Base spending rises with general inflation and does not decline in real terms."
        SpendingPathModel.EmpiricalAgeDecline -> "Base spending rises with general inflation and follows an empirical age decline after 65, then flattens at 85. Mortgage, rent, healthcare, and long-term care remain separate."
    }
}

private fun socialSecurityFullRetirementAgeText(currentAgeText: String, fallbackCurrentAge: Int): String {
    val currentAge = currentAgeText.toIntOrNull() ?: fallbackCurrentAge
    val birthYear = SocialSecurity.primaryBirthYear(currentAge)
    return "${SocialSecurity.fullRetirementAgeText(birthYear)} for birth year $birthYear"
}

private fun automaticEarlyWithdrawalPenaltyLabel(retirementAgeText: String): String {
    val retirementAge = retirementAgeText.toIntOrNull()
        ?: return "Applied automatically before age 59 1/2"
    return if (retirementAge * 12 < PENALTY_FREE_WITHDRAWAL_AGE_MONTHS) {
        "Applied automatically before age 59 1/2"
    } else {
        "Not applied at this retirement age"
    }
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

private fun requireSimulationCount(value: Int?, featureAccess: FeatureAccess): String? {
    val max = featureAccess.maxSimulationCount
    if (value != null && value in 100..max) return null
    val base = "Number of simulations must be between 100 and $max."
    return if (featureAccess.isProUnlocked) {
        base
    } else {
        "$base Upgrade to Pro for up to $PRO_SIMULATION_LIMIT simulations."
    }
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
