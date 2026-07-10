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
    fun grossWithdrawalIncludesAdditionalWithdrawalTaxRate() {
        val withoutPenalty = TaxCalculator.grossWithdrawalForNetNeed(
            netNeed = 75_000.0,
            annualSocialSecurity = 0.0,
            filingStatus = FilingStatus.Single
        )
        val withPenalty = TaxCalculator.grossWithdrawalForNetNeed(
            netNeed = 75_000.0,
            annualSocialSecurity = 0.0,
            filingStatus = FilingStatus.Single,
            additionalWithdrawalTaxRate = 0.10
        )

        assertTrue(withPenalty > withoutPenalty)
    }

    @Test
    fun additionalWithdrawalTaxCanBeLimitedToTaxableDistributionAmount() {
        val withoutPenalty = TaxCalculator.grossWithdrawalForNetNeed(
            netNeed = 75_000.0,
            annualSocialSecurity = 0.0,
            filingStatus = FilingStatus.Single
        )
        val withLimitedPenalty = TaxCalculator.grossWithdrawalForNetNeed(
            netNeed = 75_000.0,
            annualSocialSecurity = 0.0,
            filingStatus = FilingStatus.Single,
            additionalWithdrawalTaxRate = 0.10,
            additionalWithdrawalTaxableLimit = 1_000.0
        )
        val withFullPenalty = TaxCalculator.grossWithdrawalForNetNeed(
            netNeed = 75_000.0,
            annualSocialSecurity = 0.0,
            filingStatus = FilingStatus.Single,
            additionalWithdrawalTaxRate = 0.10
        )

        assertTrue(withLimitedPenalty > withoutPenalty)
        assertTrue(withLimitedPenalty < withFullPenalty)
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
}
