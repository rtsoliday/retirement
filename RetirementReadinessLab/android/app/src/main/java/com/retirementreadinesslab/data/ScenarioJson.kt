package com.retirementreadinesslab.data

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.FilingStatus
import com.retirementreadinesslab.model.Gender
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.LongTermCareAssumption
import com.retirementreadinesslab.model.MarketAssumptions
import com.retirementreadinesslab.model.MortgagePlan
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.RothConversionStrategy
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPlan
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
                .put("gender", scenario.household.gender.name))
            .put("accounts", JSONObject()
                .put("pretax", scenario.accounts.pretax)
                .put("roth", scenario.accounts.roth)
                .put("taxable", scenario.accounts.taxable)
                .put("cash", scenario.accounts.cash))
            .put("spending", JSONObject()
                .put("annualBaseSpending", scenario.spending.annualBaseSpending)
                .put("generalInflationMean", scenario.spending.generalInflationMean)
                .put("generalInflationStdDev", scenario.spending.generalInflationStdDev))
            .put("mortgage", JSONObject()
                .put("monthlyPayment", scenario.mortgage.monthlyPayment)
                .put("yearsLeft", scenario.mortgage.yearsLeft))
            .put("healthcare", JSONObject()
                .put("preMedicareMonthlyPremium", scenario.healthcare.preMedicareMonthlyPremium)
                .put("healthcareInflationMean", scenario.healthcare.healthcareInflationMean)
                .put("healthcareInflationStdDev", scenario.healthcare.healthcareInflationStdDev)
                .put("includeMedicarePremiums", scenario.healthcare.includeMedicarePremiums))
            .put("socialSecurity", JSONObject()
                .put("annualBenefitAt67", scenario.socialSecurity.annualBenefitAt67)
                .put("claimAge", scenario.socialSecurity.claimAge))
            .put("market", JSONObject()
                .put("preRetirementMeanReturn", scenario.market.preRetirementMeanReturn)
                .put("preRetirementStdDev", scenario.market.preRetirementStdDev)
                .put("stockMeanReturn", scenario.market.stockMeanReturn)
                .put("stockStdDev", scenario.market.stockStdDev)
                .put("bondMeanReturn", scenario.market.bondMeanReturn)
                .put("bondStdDev", scenario.market.bondStdDev))
            .put("rothConversion", JSONObject()
                .put("enabled", scenario.rothConversion.enabled)
                .put("marginalRateCap", scenario.rothConversion.marginalRateCap))
            .put("withdrawalStrategy", JSONObject()
                .put("useCashReserveDuringDrawdowns", scenario.withdrawalStrategy.useCashReserveDuringDrawdowns)
                .put("drawdownTrigger", scenario.withdrawalStrategy.drawdownTrigger))
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
        val mortgage = json.optJSONObject("mortgage") ?: JSONObject()
        val healthcare = json.getJSONObject("healthcare")
        val socialSecurity = json.getJSONObject("socialSecurity")
        val market = json.optJSONObject("market") ?: JSONObject()
        val rothConversion = json.optJSONObject("rothConversion") ?: JSONObject()
        val withdrawalStrategy = json.optJSONObject("withdrawalStrategy") ?: JSONObject()
        val longTermCare = json.optJSONObject("longTermCare") ?: JSONObject()

        return RetirementScenario(
            id = json.optString("id", "scenario-${System.currentTimeMillis()}"),
            name = json.optString("name", "Saved scenario"),
            household = HouseholdProfile(
                currentAge = household.optInt("currentAge", 50),
                retirementAge = household.optInt("retirementAge", 60),
                targetEndAge = household.optInt("targetEndAge", 95),
                filingStatus = enumOrDefault(household.optString("filingStatus"), FilingStatus.Single),
                gender = enumOrDefault(household.optString("gender"), Gender.Male)
            ),
            accounts = AccountBalances(
                pretax = accounts.optDouble("pretax", 0.0),
                roth = accounts.optDouble("roth", 0.0),
                taxable = accounts.optDouble("taxable", 0.0),
                cash = accounts.optDouble("cash", 0.0)
            ),
            spending = SpendingPlan(
                annualBaseSpending = spending.optDouble("annualBaseSpending", 0.0),
                generalInflationMean = spending.optDouble("generalInflationMean", 0.033),
                generalInflationStdDev = spending.optDouble("generalInflationStdDev", 0.04)
            ),
            mortgage = MortgagePlan(
                monthlyPayment = mortgage.optDouble("monthlyPayment", 0.0),
                yearsLeft = mortgage.optInt("yearsLeft", 0)
            ),
            healthcare = HealthcarePlan(
                preMedicareMonthlyPremium = healthcare.optDouble("preMedicareMonthlyPremium", 650.0),
                healthcareInflationMean = healthcare.optDouble("healthcareInflationMean", 0.055),
                healthcareInflationStdDev = healthcare.optDouble("healthcareInflationStdDev", 0.02),
                includeMedicarePremiums = healthcare.optBoolean("includeMedicarePremiums", true)
            ),
            socialSecurity = SocialSecurityPlan(
                annualBenefitAt67 = socialSecurity.optDouble("annualBenefitAt67", 0.0),
                claimAge = socialSecurity.optInt("claimAge", 67)
            ),
            market = MarketAssumptions(
                preRetirementMeanReturn = market.optDouble("preRetirementMeanReturn", 0.08),
                preRetirementStdDev = market.optDouble("preRetirementStdDev", 0.16),
                stockMeanReturn = market.optDouble("stockMeanReturn", 0.08),
                stockStdDev = market.optDouble("stockStdDev", 0.18),
                bondMeanReturn = market.optDouble("bondMeanReturn", 0.03),
                bondStdDev = market.optDouble("bondStdDev", 0.06)
            ),
            rothConversion = RothConversionStrategy(
                enabled = rothConversion.optBoolean("enabled", false),
                marginalRateCap = rothConversion.optDouble("marginalRateCap", 0.22)
            ),
            withdrawalStrategy = WithdrawalStrategy(
                useCashReserveDuringDrawdowns = withdrawalStrategy.optBoolean("useCashReserveDuringDrawdowns", true),
                drawdownTrigger = withdrawalStrategy.optDouble("drawdownTrigger", -0.01)
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
}
