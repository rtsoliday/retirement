package com.retirementreadinesslab.data

import com.retirementreadinesslab.model.DEFAULT_PROJECTION_END_AGE
import com.retirementreadinesslab.model.DEFAULT_HEALTHCARE_INFLATION_MEAN
import com.retirementreadinesslab.model.DEFAULT_HEALTHCARE_INFLATION_STD_DEV
import com.retirementreadinesslab.model.DEFAULT_PRE_MEDICARE_MONTHLY_PREMIUM
import com.retirementreadinesslab.model.BudgetLineItem
import com.retirementreadinesslab.model.BudgetProfile
import com.retirementreadinesslab.model.GuaranteedIncomePlan
import com.retirementreadinesslab.model.Gender
import com.retirementreadinesslab.model.HomePlan
import com.retirementreadinesslab.model.MonthlyBudget
import com.retirementreadinesslab.model.PostRetirementAllocationStrategy
import com.retirementreadinesslab.model.RentPlan
import com.retirementreadinesslab.model.SpendingPathModel
import com.retirementreadinesslab.model.WithdrawalStrategy
import com.retirementreadinesslab.model.sampleBaseScenario
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ScenarioJsonTest {
    @Test
    fun scenariosRoundTripThroughJsonBackup() {
        val scenario = sampleBaseScenario().copy(
            name = "Base, with comma",
            spending = sampleBaseScenario().spending.copy(lowPortfolioSpendingReduction = 0.25),
            household = sampleBaseScenario().household.copy(
                spouseCurrentAge = 49,
                spouseGender = Gender.Male
            ),
            socialSecurity = sampleBaseScenario().socialSecurity.copy(spouseClaimAge = 66),
            guaranteedIncome = GuaranteedIncomePlan(
                annualIncome = 18_000.0,
                startAge = 62,
                annualIncrease = 0.02,
                survivorPercent = 0.75
            ),
            mortgage = sampleBaseScenario().mortgage.copy(
                yearsLeft = 12,
                monthsLeft = 7,
                currentBalance = 300_000.0
            ),
            rent = RentPlan(monthlyRent = 1_850.0),
            home = HomePlan(currentValue = 425_000.0),
            postRetirementAllocation = PostRetirementAllocationStrategy(
                stockUnder30x = 0.95,
                stock30xTo35x = 0.85,
                stock35xTo40x = 0.75,
                stock40xTo45x = 0.65,
                stock45xTo50x = 0.55,
                stock50xOrMore = 0.45
            ),
            withdrawalStrategy = WithdrawalStrategy(
                useCashReserveDuringDrawdowns = false,
                drawdownTrigger = -0.02,
                applyEarlyWithdrawalPenalty = true,
                ruleOf55Eligible = true,
                seppEligible = true
            ),
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
        assertEquals(Gender.Male, decoded.first().household.spouseGender)
        assertEquals(scenario.accounts.pretax, decoded.first().accounts.pretax, 0.01)
        assertEquals(SpendingPathModel.EmpiricalAgeDecline, decoded.first().spending.spendingPathModel)
        assertEquals(0.25, decoded.first().spending.lowPortfolioSpendingReduction, 0.01)
        assertEquals(12, decoded.first().mortgage.yearsLeft)
        assertEquals(7, decoded.first().mortgage.monthsLeft)
        assertEquals(300_000.0, decoded.first().mortgage.currentBalance, 0.01)
        assertEquals(1_850.0, decoded.first().rent.monthlyRent, 0.01)
        assertEquals(425_000.0, decoded.first().home.currentValue, 0.01)
        assertEquals(0.95, decoded.first().postRetirementAllocation.stockUnder30x, 0.01)
        assertEquals(0.85, decoded.first().postRetirementAllocation.stock30xTo35x, 0.01)
        assertEquals(0.75, decoded.first().postRetirementAllocation.stock35xTo40x, 0.01)
        assertEquals(0.65, decoded.first().postRetirementAllocation.stock40xTo45x, 0.01)
        assertEquals(0.55, decoded.first().postRetirementAllocation.stock45xTo50x, 0.01)
        assertEquals(0.45, decoded.first().postRetirementAllocation.stock50xOrMore, 0.01)
        assertEquals(-0.02, decoded.first().withdrawalStrategy.drawdownTrigger, 0.001)
        assertTrue(decoded.first().withdrawalStrategy.applyEarlyWithdrawalPenalty)
        assertTrue(decoded.first().withdrawalStrategy.ruleOf55Eligible)
        assertTrue(decoded.first().withdrawalStrategy.seppEligible)
        assertEquals(scenario.socialSecurity.claimAge, decoded.first().socialSecurity.claimAge)
        assertEquals(66, decoded.first().socialSecurity.spouseClaimAge)
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
    fun importedScenariosAlwaysIncludeMedicarePremiums() {
        val raw = JSONArray(ScenarioJson.encodeScenarios(listOf(sampleBaseScenario()))).apply {
            getJSONObject(0).getJSONObject("healthcare").put("includeMedicarePremiums", false)
        }.toString()

        val decoded = ScenarioJson.decodeScenarios(raw)

        assertTrue(decoded.first().healthcare.includeMedicarePremiums)
    }

    @Test
    fun legacyBackupsWithoutRentDefaultToZeroRent() {
        val raw = JSONArray(ScenarioJson.encodeScenarios(listOf(sampleBaseScenario()))).apply {
            getJSONObject(0).getJSONObject("spending").remove("lowPortfolioSpendingReduction")
            getJSONObject(0).getJSONObject("spending").remove("spendingPathModel")
            getJSONObject(0).getJSONObject("mortgage").remove("currentBalance")
            getJSONObject(0).getJSONObject("mortgage").remove("monthsLeft")
            getJSONObject(0).getJSONObject("healthcare").remove("preMedicareMonthlyPremium")
            getJSONObject(0).getJSONObject("healthcare").remove("healthcareInflationMean")
            getJSONObject(0).getJSONObject("healthcare").remove("healthcareInflationStdDev")
            getJSONObject(0).getJSONObject("household").remove("spouseCurrentAge")
            getJSONObject(0).getJSONObject("household").remove("spouseGender")
            getJSONObject(0).getJSONObject("socialSecurity").remove("spouseClaimAge")
            getJSONObject(0).remove("guaranteedIncome")
            getJSONObject(0).remove("rent")
            getJSONObject(0).remove("home")
            getJSONObject(0).remove("postRetirementAllocation")
            getJSONObject(0).getJSONObject("withdrawalStrategy").remove("applyEarlyWithdrawalPenalty")
            getJSONObject(0).getJSONObject("withdrawalStrategy").remove("ruleOf55Eligible")
            getJSONObject(0).getJSONObject("withdrawalStrategy").remove("seppEligible")
        }.toString()

        val decoded = ScenarioJson.decodeScenarios(raw)

        assertEquals(0.10, decoded.first().spending.lowPortfolioSpendingReduction, 0.01)
        assertEquals(SpendingPathModel.EmpiricalAgeDecline, decoded.first().spending.spendingPathModel)
        assertEquals(0.0, decoded.first().mortgage.currentBalance, 0.01)
        assertEquals(0, decoded.first().mortgage.monthsLeft)
        assertEquals(DEFAULT_PRE_MEDICARE_MONTHLY_PREMIUM, decoded.first().healthcare.preMedicareMonthlyPremium, 0.01)
        assertEquals(DEFAULT_HEALTHCARE_INFLATION_MEAN, decoded.first().healthcare.healthcareInflationMean, 0.001)
        assertEquals(DEFAULT_HEALTHCARE_INFLATION_STD_DEV, decoded.first().healthcare.healthcareInflationStdDev, 0.001)
        assertEquals(decoded.first().household.currentAge, decoded.first().household.spouseCurrentAge)
        assertEquals(Gender.Female, decoded.first().household.spouseGender)
        assertEquals(67, decoded.first().socialSecurity.spouseClaimAge)
        assertEquals(0.0, decoded.first().guaranteedIncome.annualIncome, 0.01)
        assertEquals(65, decoded.first().guaranteedIncome.startAge)
        assertEquals(1.0, decoded.first().guaranteedIncome.survivorPercent, 0.01)
        assertEquals(0.0, decoded.first().rent.monthlyRent, 0.01)
        assertEquals(0.0, decoded.first().home.currentValue, 0.01)
        assertEquals(PostRetirementAllocationStrategy(), decoded.first().postRetirementAllocation)
        assertFalse(decoded.first().withdrawalStrategy.applyEarlyWithdrawalPenalty)
        assertFalse(decoded.first().withdrawalStrategy.ruleOf55Eligible)
        assertFalse(decoded.first().withdrawalStrategy.seppEligible)
    }
}
