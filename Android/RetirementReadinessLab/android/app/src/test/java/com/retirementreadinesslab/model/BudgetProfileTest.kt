package com.retirementreadinesslab.model

import org.junit.Assert.assertEquals
import org.junit.Test

class BudgetProfileTest {
    @Test
    fun annualBaseSpendingEstimateUsesOnlyLatestTwelveMonthlyRecords() {
        val monthlyBudgets = (1..13).map { index ->
            val year = if (index == 13) 2026 else 2025
            val month = if (index == 13) 1 else index
            MonthlyBudget(
                month = "%04d-%02d".format(year, month),
                checkingSavingsBills = listOf(
                    BudgetLineItem(
                        id = "bill-$index",
                        name = "Bill $index",
                        monthlyAmount = index * 100.0
                    )
                )
            )
        }

        val budget = BudgetProfile(
            annualPropertyTaxes = 3_000.0,
            annualHomeInsurance = 1_500.0,
            annualAutoInsurance = 1_500.0,
            monthlyBudgets = monthlyBudgets
        )

        assertEquals(12, budget.monthsUsedForEstimate.size)
        assertEquals("2025-02", budget.monthsUsedForEstimate.first().month)
        assertEquals(750.0, budget.averageMonthlySpending, 0.01)
        assertEquals(9_000.0, budget.annualizedMonthlySpending, 0.01)
        assertEquals(15_000.0, budget.annualBaseSpendingEstimate, 0.01)
    }
}
