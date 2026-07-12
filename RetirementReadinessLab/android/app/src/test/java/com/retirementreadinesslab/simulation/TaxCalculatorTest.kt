package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.FilingStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TaxCalculatorTest {
    @Test
    fun singleFilerUsesProgressiveBrackets() {
        val tax = TaxCalculator.taxLiability(50_000.0, FilingStatus.Single)

        assertEquals(5_752.0, tax, 1.0)
    }

    @Test
    fun ordinaryIncomeAppliesBaseStandardDeduction() {
        assertEquals(
            0.0,
            TaxCalculator.ordinaryIncomeTaxLiability(16_100.0, FilingStatus.Single),
            0.01
        )
        assertEquals(
            1_000.0,
            TaxCalculator.ordinaryIncomeTaxLiability(26_100.0, FilingStatus.Single),
            0.01
        )
    }

    @Test
    fun seniorDeductionsApplyByAgeIncomeFilingStatusAndTaxYear() {
        assertEquals(
            585.0,
            TaxCalculator.ordinaryIncomeTaxLiability(
                ordinaryIncome = 30_000.0,
                filingStatus = FilingStatus.Single,
                age65OrOlderPeople = 1,
                taxYear = 2026
            ),
            0.01
        )

        val phasedSingleTaxableIncome = 100_000.0 - 16_100.0 - 2_050.0 - 4_500.0
        assertEquals(
            TaxCalculator.taxLiability(phasedSingleTaxableIncome, FilingStatus.Single),
            TaxCalculator.ordinaryIncomeTaxLiability(
                ordinaryIncome = 100_000.0,
                filingStatus = FilingStatus.Single,
                age65OrOlderPeople = 1,
                taxYear = 2026
            ),
            0.01
        )

        val marriedTaxableIncome = 100_000.0 - 32_200.0 - 2 * 1_650.0 - 2 * 6_000.0
        assertEquals(
            TaxCalculator.taxLiability(marriedTaxableIncome, FilingStatus.Married),
            TaxCalculator.ordinaryIncomeTaxLiability(
                ordinaryIncome = 100_000.0,
                filingStatus = FilingStatus.Married,
                age65OrOlderPeople = 2,
                taxYear = 2026
            ),
            0.01
        )

        val postExpirationTaxableIncome = 100_000.0 - 16_100.0 - 2_050.0
        assertEquals(
            TaxCalculator.taxLiability(postExpirationTaxableIncome, FilingStatus.Single),
            TaxCalculator.ordinaryIncomeTaxLiability(
                ordinaryIncome = 100_000.0,
                filingStatus = FilingStatus.Single,
                age65OrOlderPeople = 1,
                taxYear = 2029
            ),
            0.01
        )
    }

    @Test
    fun ordinaryIncomeIndexesBracketsAndStandardDeductionForInflation() {
        assertEquals(
            0.0,
            TaxCalculator.ordinaryIncomeTaxLiability(
                ordinaryIncome = 32_200.0,
                filingStatus = FilingStatus.Single,
                inflationMultiplier = 2.0
            ),
            0.01
        )
        assertEquals(
            2_000.0,
            TaxCalculator.ordinaryIncomeTaxLiability(
                ordinaryIncome = 52_200.0,
                filingStatus = FilingStatus.Single,
                inflationMultiplier = 2.0
            ),
            0.01
        )
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
    fun upperSocialSecurityTierUsesHalfBenefitWhenItIsBelowFixedBase() {
        val taxable = TaxCalculator.taxableSocialSecurity(
            otherIncome = 32_100.0,
            annualSocialSecurity = 4_000.0,
            filingStatus = FilingStatus.Single
        )

        assertEquals(2_085.0, taxable, 0.01)
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

    @Test
    fun rothConversionFillsOnlyUnusedBracketSpaceAndUsesIncrementalTax() {
        val plan = TaxCalculator.rothConversionPlan(
            pretaxBalance = 100_000.0,
            currentTaxableIncome = 80_000.0,
            rateCap = 0.22,
            filingStatus = FilingStatus.Single
        )

        assertEquals(41_800.0, plan.conversionAmount, 0.01)
        assertEquals(
            TaxCalculator.ordinaryIncomeTaxLiability(121_800.0, FilingStatus.Single) -
                TaxCalculator.ordinaryIncomeTaxLiability(80_000.0, FilingStatus.Single),
            plan.additionalTax,
            0.01
        )
    }

    @Test
    fun rothConversionAccountsForSeniorDeductionPhaseoutWhenFillingBracket() {
        val plan = TaxCalculator.rothConversionPlan(
            pretaxBalance = 100_000.0,
            currentTaxableIncome = 80_000.0,
            rateCap = 0.22,
            filingStatus = FilingStatus.Single,
            age65OrOlderPeople = 1,
            taxYear = 2026
        )

        assertEquals(46_745.28, plan.conversionAmount, 0.01)
        assertEquals(
            TaxCalculator.ordinaryIncomeTaxLiability(
                ordinaryIncome = 80_000.0 + plan.conversionAmount,
                filingStatus = FilingStatus.Single,
                age65OrOlderPeople = 1,
                taxYear = 2026
            ) - TaxCalculator.ordinaryIncomeTaxLiability(
                ordinaryIncome = 80_000.0,
                filingStatus = FilingStatus.Single,
                age65OrOlderPeople = 1,
                taxYear = 2026
            ),
            plan.additionalTax,
            0.01
        )
    }
}
