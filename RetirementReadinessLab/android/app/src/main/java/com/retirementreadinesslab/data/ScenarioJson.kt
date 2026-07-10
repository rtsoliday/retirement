package com.retirementreadinesslab.data

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.BudgetLineItem
import com.retirementreadinesslab.model.BudgetProfile
import com.retirementreadinesslab.model.DEFAULT_GENERAL_INFLATION_MEAN
import com.retirementreadinesslab.model.DEFAULT_GENERAL_INFLATION_STD_DEV
import com.retirementreadinesslab.model.DEFAULT_HEALTHCARE_INFLATION_MEAN
import com.retirementreadinesslab.model.DEFAULT_HEALTHCARE_INFLATION_STD_DEV
import com.retirementreadinesslab.model.DEFAULT_PRE_MEDICARE_MONTHLY_PREMIUM
import com.retirementreadinesslab.model.DEFAULT_PROJECTION_END_AGE
import com.retirementreadinesslab.model.FilingStatus
import com.retirementreadinesslab.model.Gender
import com.retirementreadinesslab.model.GuaranteedIncomePlan
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HomePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.LongTermCareAssumption
import com.retirementreadinesslab.model.MarketAssumptions
import com.retirementreadinesslab.model.MonthlyBudget
import com.retirementreadinesslab.model.MortgagePlan
import com.retirementreadinesslab.model.PostRetirementAllocationStrategy
import com.retirementreadinesslab.model.RentPlan
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.RothConversionStrategy
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPlan
import com.retirementreadinesslab.model.SpendingPathModel
import com.retirementreadinesslab.model.WithdrawalStrategy
import org.json.JSONArray
import org.json.JSONObject

object ScenarioJson {
    fun encodeScenarios(scenarios: List<RetirementScenario>): String {
        val array = JSONArray()
        scenarios.forEach { array.put(encodeScenario(it)) }
        return array.toString()
    }

    fun decodeScenarios(raw: String): List<RetirementScenario> {
        if (raw.isBlank()) return emptyList()
        val array = JSONArray(raw)
        return buildList {
            for (index in 0 until array.length()) {
                add(decodeScenario(array.getJSONObject(index)))
            }
        }
    }

    private fun encodeScenario(scenario: RetirementScenario): JSONObject {
        return JSONObject()
            .put("id", scenario.id)
            .put("name", scenario.name)
            .put("household", JSONObject()
                .put("currentAge", scenario.household.currentAge)
                .put("retirementAge", scenario.household.retirementAge)
                .put("targetEndAge", scenario.household.targetEndAge)
                .put("filingStatus", scenario.household.filingStatus.name)
                .put("gender", scenario.household.gender.name)
                .put("spouseCurrentAge", scenario.household.spouseCurrentAge))
            .put("accounts", JSONObject()
                .put("pretax", scenario.accounts.pretax)
                .put("roth", scenario.accounts.roth)
                .put("taxable", scenario.accounts.taxable)
                .put("cash", scenario.accounts.cash))
            .put("spending", JSONObject()
                .put("annualBaseSpending", scenario.spending.annualBaseSpending)
                .put("generalInflationMean", scenario.spending.generalInflationMean)
                .put("generalInflationStdDev", scenario.spending.generalInflationStdDev)
                .put("spendingPathModel", scenario.spending.spendingPathModel.name)
                .put("lowPortfolioSpendingReduction", scenario.spending.lowPortfolioSpendingReduction))
            .put("budget", encodeBudget(scenario.budget))
            .put("mortgage", JSONObject()
                .put("monthlyPayment", scenario.mortgage.monthlyPayment)
                .put("yearsLeft", scenario.mortgage.yearsLeft)
                .put("monthsLeft", scenario.mortgage.monthsLeft)
                .put("currentBalance", scenario.mortgage.currentBalance))
            .put("rent", JSONObject()
                .put("monthlyRent", scenario.rent.monthlyRent))
            .put("home", JSONObject()
                .put("currentValue", scenario.home.currentValue))
            .put("healthcare", JSONObject()
                .put("preMedicareMonthlyPremium", scenario.healthcare.preMedicareMonthlyPremium)
                .put("healthcareInflationMean", scenario.healthcare.healthcareInflationMean)
                .put("healthcareInflationStdDev", scenario.healthcare.healthcareInflationStdDev)
                .put("includeMedicarePremiums", scenario.healthcare.includeMedicarePremiums))
            .put("socialSecurity", JSONObject()
                .put("annualBenefitAt67", scenario.socialSecurity.annualBenefitAt67)
                .put("claimAge", scenario.socialSecurity.claimAge)
                .put("spouseClaimAge", scenario.socialSecurity.spouseClaimAge))
            .put("guaranteedIncome", JSONObject()
                .put("annualIncome", scenario.guaranteedIncome.annualIncome)
                .put("startAge", scenario.guaranteedIncome.startAge)
                .put("annualIncrease", scenario.guaranteedIncome.annualIncrease)
                .put("survivorPercent", scenario.guaranteedIncome.survivorPercent))
            .put("market", JSONObject()
                .put("preRetirementMeanReturn", scenario.market.preRetirementMeanReturn)
                .put("preRetirementStdDev", scenario.market.preRetirementStdDev)
                .put("stockMeanReturn", scenario.market.stockMeanReturn)
                .put("stockStdDev", scenario.market.stockStdDev)
                .put("bondMeanReturn", scenario.market.bondMeanReturn)
                .put("bondStdDev", scenario.market.bondStdDev))
            .put("postRetirementAllocation", encodePostRetirementAllocation(scenario.postRetirementAllocation))
            .put("rothConversion", JSONObject()
                .put("enabled", scenario.rothConversion.enabled)
                .put("marginalRateCap", scenario.rothConversion.marginalRateCap))
            .put("withdrawalStrategy", JSONObject()
                .put("useCashReserveDuringDrawdowns", scenario.withdrawalStrategy.useCashReserveDuringDrawdowns)
                .put("drawdownTrigger", scenario.withdrawalStrategy.drawdownTrigger)
                .put("applyEarlyWithdrawalPenalty", scenario.withdrawalStrategy.applyEarlyWithdrawalPenalty)
                .put("ruleOf55Eligible", scenario.withdrawalStrategy.ruleOf55Eligible)
                .put("seppEligible", scenario.withdrawalStrategy.seppEligible))
            .put("longTermCare", JSONObject()
                .put("enabled", scenario.longTermCare.enabled)
                .put("annualCost", scenario.longTermCare.annualCost)
                .put("averageDurationYears", scenario.longTermCare.averageDurationYears))
            .put("numberOfSimulations", scenario.numberOfSimulations)
            .put("seed", scenario.seed)
    }

    private fun decodeScenario(json: JSONObject): RetirementScenario {
        val household = json.getJSONObject("household")
        val accounts = json.getJSONObject("accounts")
        val spending = json.getJSONObject("spending")
        val budget = json.optJSONObject("budget") ?: JSONObject()
        val mortgage = json.optJSONObject("mortgage") ?: JSONObject()
        val rent = json.optJSONObject("rent") ?: JSONObject()
        val home = json.optJSONObject("home") ?: JSONObject()
        val healthcare = json.getJSONObject("healthcare")
        val socialSecurity = json.getJSONObject("socialSecurity")
        val guaranteedIncome = json.optJSONObject("guaranteedIncome") ?: JSONObject()
        val market = json.optJSONObject("market") ?: JSONObject()
        val postRetirementAllocation = json.optJSONObject("postRetirementAllocation") ?: JSONObject()
        val rothConversion = json.optJSONObject("rothConversion") ?: JSONObject()
        val withdrawalStrategy = json.optJSONObject("withdrawalStrategy") ?: JSONObject()
        val longTermCare = json.optJSONObject("longTermCare") ?: JSONObject()

        val retirementAge = household.optInt("retirementAge", 60)
        val defaultWithdrawalStrategy = WithdrawalStrategy.defaultsForRetirementAge(retirementAge)

        return RetirementScenario(
            id = json.optString("id", "scenario-${System.currentTimeMillis()}"),
            name = json.optString("name", "Saved scenario"),
            household = HouseholdProfile(
                currentAge = household.optInt("currentAge", 50),
                retirementAge = retirementAge,
                targetEndAge = DEFAULT_PROJECTION_END_AGE,
                filingStatus = enumOrDefault(household.optString("filingStatus"), FilingStatus.Single),
                gender = enumOrDefault(household.optString("gender"), Gender.Male),
                spouseCurrentAge = household.optInt("spouseCurrentAge", household.optInt("currentAge", 50))
            ),
            accounts = AccountBalances(
                pretax = accounts.optDouble("pretax", 0.0),
                roth = accounts.optDouble("roth", 0.0),
                taxable = accounts.optDouble("taxable", 0.0),
                cash = accounts.optDouble("cash", 0.0)
            ),
            spending = SpendingPlan(
                annualBaseSpending = spending.optDouble("annualBaseSpending", 0.0),
                generalInflationMean = spending.optDouble("generalInflationMean", DEFAULT_GENERAL_INFLATION_MEAN),
                generalInflationStdDev = spending.optDouble("generalInflationStdDev", DEFAULT_GENERAL_INFLATION_STD_DEV),
                spendingPathModel = enumOrDefault(
                    spending.optString("spendingPathModel"),
                    SpendingPathModel.EmpiricalAgeDecline
                ),
                lowPortfolioSpendingReduction = spending.optDouble("lowPortfolioSpendingReduction", 0.10)
            ),
            budget = decodeBudget(budget),
            mortgage = MortgagePlan(
                monthlyPayment = mortgage.optDouble("monthlyPayment", 0.0),
                yearsLeft = mortgage.optInt("yearsLeft", 0),
                monthsLeft = mortgage.optInt("monthsLeft", 0),
                currentBalance = mortgage.optDouble("currentBalance", 0.0)
            ),
            rent = RentPlan(
                monthlyRent = rent.optDouble("monthlyRent", 0.0)
            ),
            home = HomePlan(
                currentValue = home.optDouble("currentValue", 0.0)
            ),
            healthcare = HealthcarePlan(
                preMedicareMonthlyPremium = healthcare.optDouble(
                    "preMedicareMonthlyPremium",
                    DEFAULT_PRE_MEDICARE_MONTHLY_PREMIUM
                ),
                healthcareInflationMean = healthcare.optDouble(
                    "healthcareInflationMean",
                    DEFAULT_HEALTHCARE_INFLATION_MEAN
                ),
                healthcareInflationStdDev = healthcare.optDouble(
                    "healthcareInflationStdDev",
                    DEFAULT_HEALTHCARE_INFLATION_STD_DEV
                ),
                includeMedicarePremiums = true
            ),
            socialSecurity = SocialSecurityPlan(
                annualBenefitAt67 = socialSecurity.optDouble("annualBenefitAt67", 0.0),
                claimAge = socialSecurity.optInt("claimAge", 67),
                spouseClaimAge = socialSecurity.optInt("spouseClaimAge", 67)
            ),
            guaranteedIncome = GuaranteedIncomePlan(
                annualIncome = guaranteedIncome.optDouble("annualIncome", 0.0),
                startAge = guaranteedIncome.optInt("startAge", 65),
                annualIncrease = guaranteedIncome.optDouble("annualIncrease", 0.0),
                survivorPercent = guaranteedIncome.optDouble("survivorPercent", 1.0)
            ),
            market = MarketAssumptions(
                preRetirementMeanReturn = market.optDouble("preRetirementMeanReturn", 0.133),
                preRetirementStdDev = market.optDouble("preRetirementStdDev", 0.162),
                stockMeanReturn = market.optDouble("stockMeanReturn", 0.133),
                stockStdDev = market.optDouble("stockStdDev", 0.162),
                bondMeanReturn = market.optDouble("bondMeanReturn", 0.03),
                bondStdDev = market.optDouble("bondStdDev", 0.06)
            ),
            postRetirementAllocation = decodePostRetirementAllocation(postRetirementAllocation),
            rothConversion = RothConversionStrategy(
                enabled = rothConversion.optBoolean("enabled", false),
                marginalRateCap = rothConversion.optDouble("marginalRateCap", 0.22)
            ),
            withdrawalStrategy = WithdrawalStrategy(
                useCashReserveDuringDrawdowns = false,
                drawdownTrigger = withdrawalStrategy.optDouble("drawdownTrigger", -0.01),
                applyEarlyWithdrawalPenalty = withdrawalStrategy.optBoolean(
                    "applyEarlyWithdrawalPenalty",
                    defaultWithdrawalStrategy.applyEarlyWithdrawalPenalty
                ),
                ruleOf55Eligible = withdrawalStrategy.optBoolean(
                    "ruleOf55Eligible",
                    defaultWithdrawalStrategy.ruleOf55Eligible
                ),
                seppEligible = withdrawalStrategy.optBoolean(
                    "seppEligible",
                    defaultWithdrawalStrategy.seppEligible
                )
            ),
            longTermCare = LongTermCareAssumption(
                enabled = longTermCare.optBoolean("enabled", false),
                annualCost = longTermCare.optDouble("annualCost", 100_000.0),
                averageDurationYears = longTermCare.optInt("averageDurationYears", 3)
            ),
            numberOfSimulations = json.optInt("numberOfSimulations", 1_500),
            seed = json.optLong("seed", 20260429L)
        )
    }

    private inline fun <reified T : Enum<T>> enumOrDefault(value: String, default: T): T {
        return enumValues<T>().firstOrNull { it.name == value } ?: default
    }

    private fun encodePostRetirementAllocation(allocation: PostRetirementAllocationStrategy): JSONObject {
        return JSONObject()
            .put("stockUnder30x", allocation.stockUnder30x)
            .put("stock30xTo35x", allocation.stock30xTo35x)
            .put("stock35xTo40x", allocation.stock35xTo40x)
            .put("stock40xTo45x", allocation.stock40xTo45x)
            .put("stock45xTo50x", allocation.stock45xTo50x)
            .put("stock50xOrMore", allocation.stock50xOrMore)
    }

    private fun decodePostRetirementAllocation(json: JSONObject): PostRetirementAllocationStrategy {
        val defaults = PostRetirementAllocationStrategy()
        return PostRetirementAllocationStrategy(
            stockUnder30x = json.optDouble("stockUnder30x", defaults.stockUnder30x),
            stock30xTo35x = json.optDouble("stock30xTo35x", defaults.stock30xTo35x),
            stock35xTo40x = json.optDouble("stock35xTo40x", defaults.stock35xTo40x),
            stock40xTo45x = json.optDouble("stock40xTo45x", defaults.stock40xTo45x),
            stock45xTo50x = json.optDouble("stock45xTo50x", defaults.stock45xTo50x),
            stock50xOrMore = json.optDouble("stock50xOrMore", defaults.stock50xOrMore)
        )
    }

    private fun encodeBudget(budget: BudgetProfile): JSONObject {
        return JSONObject()
            .put("annualPropertyTaxes", budget.annualPropertyTaxes)
            .put("annualHomeInsurance", budget.annualHomeInsurance)
            .put("annualAutoInsurance", budget.annualAutoInsurance)
            .put("monthlyBudgets", JSONArray().apply {
                budget.monthlyBudgets.forEach { month ->
                    put(JSONObject()
                        .put("month", month.month)
                        .put("checkingSavingsBills", encodeBudgetLineItems(month.checkingSavingsBills))
                        .put("creditCardBills", encodeBudgetLineItems(month.creditCardBills))
                        .put("cashAndAtmWithdrawals", month.cashAndAtmWithdrawals))
                }
            })
    }

    private fun encodeBudgetLineItems(items: List<BudgetLineItem>): JSONArray {
        return JSONArray().apply {
            items.forEach { item ->
                put(JSONObject()
                    .put("id", item.id)
                    .put("name", item.name)
                    .put("monthlyAmount", item.monthlyAmount))
            }
        }
    }

    private fun decodeBudget(json: JSONObject): BudgetProfile {
        return BudgetProfile(
            annualPropertyTaxes = json.optDouble("annualPropertyTaxes", 0.0),
            annualHomeInsurance = json.optDouble("annualHomeInsurance", 0.0),
            annualAutoInsurance = json.optDouble("annualAutoInsurance", 0.0),
            monthlyBudgets = decodeMonthlyBudgets(json.optJSONArray("monthlyBudgets") ?: JSONArray())
        )
    }

    private fun decodeMonthlyBudgets(array: JSONArray): List<MonthlyBudget> {
        return buildList {
            for (index in 0 until array.length()) {
                val month = array.getJSONObject(index)
                add(MonthlyBudget(
                    month = month.optString("month"),
                    checkingSavingsBills = decodeBudgetLineItems(month.optJSONArray("checkingSavingsBills") ?: JSONArray()),
                    creditCardBills = decodeBudgetLineItems(month.optJSONArray("creditCardBills") ?: JSONArray()),
                    cashAndAtmWithdrawals = month.optDouble("cashAndAtmWithdrawals", 0.0)
                ))
            }
        }
    }

    private fun decodeBudgetLineItems(array: JSONArray): List<BudgetLineItem> {
        return buildList {
            for (index in 0 until array.length()) {
                val item = array.getJSONObject(index)
                add(BudgetLineItem(
                    id = item.optString("id", "budget-item-$index"),
                    name = item.optString("name", "Budget item"),
                    monthlyAmount = item.optDouble("monthlyAmount", 0.0)
                ))
            }
        }
    }
}
