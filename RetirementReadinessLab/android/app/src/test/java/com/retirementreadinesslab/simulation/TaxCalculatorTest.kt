package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.FilingStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TaxCalculatorTest {
    @Test
    fun singleFilerUsesProgressiveBrackets() {
        val tax = TaxCalculator.taxLiability(50_000.0, FilingStatus.Single)

        assertEquals(6_053.0, tax, 1.0)
    }

    @Test
    fun socialSecurityIsUntaxedBelowThreshold() {
        val taxable = TaxCalculator.taxableSocialSecurity(
            otherIncome = 10_000.0,
            annualSocialSecurity = 20_000.0,
            filingStatus = FilingStatus.Single
        )

        assertEquals(0.0, taxable, 0.01)
    }

    @Test
    fun grossWithdrawalExceedsNetNeedWhenTaxable() {
        val gross = TaxCalculator.grossWithdrawalForNetNeed(
            netNeed = 75_000.0,
            annualSocialSecurity = 0.0,
            filingStatus = FilingStatus.Single
        )

        assertTrue(gross > 75_000.0)
    }

    @Test
    fun otherTaxableIncomeCanCoverNetNeedBeforePortfolioWithdrawal() {
        val gross = TaxCalculator.grossWithdrawalForNetNeed(
            netNeed = 10_000.0,
            annualSocialSecurity = 0.0,
            filingStatus = FilingStatus.Single,
            annualOtherTaxableIncome = 12_000.0
        )

        assertEquals(0.0, gross, 0.01)
    }

    @Test
    fun rothConversionFillsOnlyRemainingBracketHeadroom() {
        val plan = TaxCalculator.rothConversionPlan(
            pretaxBalance = 100_000.0,
            currentTaxableIncome = 20_000.0,
            rateCap = 0.12,
            filingStatus = FilingStatus.Single
        )

        assertEquals(27_150.0, plan.conversionAmount, 0.01)
        assertEquals(
            TaxCalculator.taxLiability(47_150.0, FilingStatus.Single) -
                TaxCalculator.taxLiability(20_000.0, FilingStatus.Single),
            plan.additionalTax,
            0.01
        )
    }

    @Test
    fun topBracketRothConversionIsNotSilentlyDisabled() {
        val plan = TaxCalculator.rothConversionPlan(
            pretaxBalance = 80_000.0,
            currentTaxableIncome = 50_000.0,
            rateCap = 0.37,
            filingStatus = FilingStatus.Single
        )

        assertEquals(80_000.0, plan.conversionAmount, 0.01)
        assertTrue(plan.additionalTax > 0.0)
    }
}
