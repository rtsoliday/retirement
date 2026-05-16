package com.retirementreadinesslab.data

import com.retirementreadinesslab.model.DEFAULT_PROJECTION_END_AGE
import com.retirementreadinesslab.model.BudgetLineItem
import com.retirementreadinesslab.model.BudgetProfile
import com.retirementreadinesslab.model.GuaranteedIncomePlan
import com.retirementreadinesslab.model.HomePlan
import com.retirementreadinesslab.model.MonthlyBudget
import com.retirementreadinesslab.model.RentPlan
import com.retirementreadinesslab.model.SpendingPathModel
import com.retirementreadinesslab.model.sampleBaseScenario
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScenarioJsonTest {
    @Test
    fun scenariosRoundTripThroughJsonBackup() {
        val scenario = sampleBaseScenario().copy(
            name = "Base, with comma",
            spending = sampleBaseScenario().spending.copy(lowPortfolioSpendingReduction = 0.25),
            household = sampleBaseScenario().household.copy(spouseCurrentAge = 49),
            guaranteedIncome = GuaranteedIncomePlan(
                annualIncome = 18_000.0,
                startAge = 62,
                annualIncrease = 0.02,
                survivorPercent = 0.75
            ),
            mortgage = sampleBaseScenario().mortgage.copy(currentBalance = 300_000.0),
            rent = RentPlan(monthlyRent = 1_850.0),
            home = HomePlan(currentValue = 425_000.0),
            budget = BudgetProfile(
                annualPropertyTaxes = 4_800.0,
                annualHomeInsurance = 2_400.0,
                annualAutoInsurance = 1_800.0,
                monthlyBudgets = listOf(
                    MonthlyBudget(
                        month = "2026-05",
                        checkingSavingsBills = listOf(BudgetLineItem("utilities", "Utilities", 325.0)),
                        creditCardBills = listOf(BudgetLineItem("visa", "Visa", 1_200.0)),
                        cashAndAtmWithdrawals = 300.0
                    )
                )
            )
        )

        val raw = ScenarioJson.encodeScenarios(listOf(scenario))
        val decoded = ScenarioJson.decodeScenarios(raw)

        assertEquals(1, decoded.size)
        assertEquals("Base, with comma", decoded.first().name)
        assertEquals(scenario.household.retirementAge, decoded.first().household.retirementAge)
        assertEquals(49, decoded.first().household.spouseCurrentAge)
        assertEquals(scenario.accounts.pretax, decoded.first().accounts.pretax, 0.01)
        assertEquals(SpendingPathModel.EmpiricalAgeDecline, decoded.first().spending.spendingPathModel)
        assertEquals(0.25, decoded.first().spending.lowPortfolioSpendingReduction, 0.01)
        assertEquals(300_000.0, decoded.first().mortgage.currentBalance, 0.01)
        assertEquals(1_850.0, decoded.first().rent.monthlyRent, 0.01)
        assertEquals(425_000.0, decoded.first().home.currentValue, 0.01)
        assertEquals(scenario.socialSecurity.claimAge, decoded.first().socialSecurity.claimAge)
        assertEquals(18_000.0, decoded.first().guaranteedIncome.annualIncome, 0.01)
        assertEquals(62, decoded.first().guaranteedIncome.startAge)
        assertEquals(0.02, decoded.first().guaranteedIncome.annualIncrease, 0.01)
        assertEquals(0.75, decoded.first().guaranteedIncome.survivorPercent, 0.01)
        assertEquals(4_800.0, decoded.first().budget.annualPropertyTaxes, 0.01)
        assertEquals("2026-05", decoded.first().budget.monthlyBudgets.first().month)
        assertEquals("Utilities", decoded.first().budget.monthlyBudgets.first().checkingSavingsBills.first().name)
        assertTrue(raw.startsWith("["))
    }

    @Test
    fun legacyTargetEndAgeImportsUseMortalityProjectionCap() {
        val scenario = sampleBaseScenario().copy(
            household = sampleBaseScenario().household.copy(targetEndAge = 95)
        )

        val decoded = ScenarioJson.decodeScenarios(ScenarioJson.encodeScenarios(listOf(scenario)))

        assertEquals(DEFAULT_PROJECTION_END_AGE, decoded.first().household.targetEndAge)
    }

    @Test
    fun legacyBackupsWithoutRentDefaultToZeroRent() {
        val raw = JSONArray(ScenarioJson.encodeScenarios(listOf(sampleBaseScenario()))).apply {
            getJSONObject(0).getJSONObject("spending").remove("lowPortfolioSpendingReduction")
            getJSONObject(0).getJSONObject("spending").remove("spendingPathModel")
            getJSONObject(0).getJSONObject("mortgage").remove("currentBalance")
            getJSONObject(0).getJSONObject("household").remove("spouseCurrentAge")
            getJSONObject(0).remove("guaranteedIncome")
            getJSONObject(0).remove("rent")
            getJSONObject(0).remove("home")
        }.toString()

        val decoded = ScenarioJson.decodeScenarios(raw)

        assertEquals(0.10, decoded.first().spending.lowPortfolioSpendingReduction, 0.01)
        assertEquals(SpendingPathModel.EmpiricalAgeDecline, decoded.first().spending.spendingPathModel)
        assertEquals(0.0, decoded.first().mortgage.currentBalance, 0.01)
        assertEquals(decoded.first().household.currentAge, decoded.first().household.spouseCurrentAge)
        assertEquals(0.0, decoded.first().guaranteedIncome.annualIncome, 0.01)
        assertEquals(65, decoded.first().guaranteedIncome.startAge)
        assertEquals(1.0, decoded.first().guaranteedIncome.survivorPercent, 0.01)
        assertEquals(0.0, decoded.first().rent.monthlyRent, 0.01)
        assertEquals(0.0, decoded.first().home.currentValue, 0.01)
    }
}
