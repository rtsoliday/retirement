package com.retirementreadinesslab.simulation

import org.junit.Assert.assertEquals
import org.junit.Test

class SocialSecurityTest {
    @Test
    fun benefitAt67ReturnsFullBenefit() {
        assertEquals(30_000.0, SocialSecurity.annualBenefitAtClaimAge(30_000.0, 67), 0.01)
    }

    @Test
    fun claimingAt62ReducesBenefit() {
        val benefit = SocialSecurity.annualBenefitAtClaimAge(30_000.0, 62)

        assertEquals(21_000.0, benefit, 1.0)
    }

    @Test
    fun claimingAt70IncreasesBenefit() {
        val benefit = SocialSecurity.annualBenefitAtClaimAge(30_000.0, 70)

        assertEquals(37_200.0, benefit, 1.0)
    }
}
