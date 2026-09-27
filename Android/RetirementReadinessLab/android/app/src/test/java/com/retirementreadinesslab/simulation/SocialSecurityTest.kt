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

    @Test
    fun fullRetirementAgeUsesBirthYear() {
        assertEquals(66 * 12 + 8, SocialSecurity.fullRetirementAgeMonths(1958))
        assertEquals(67 * 12, SocialSecurity.fullRetirementAgeMonths(1960))
    }

    @Test
    fun survivorFullRetirementAgeUsesShiftedBirthYearSchedule() {
        assertEquals(66 * 12 + 4, SocialSecurity.survivorFullRetirementAgeMonths(1958))
    }

    @Test
    fun claimingAfterEarlierFraGetsDelayedCredit() {
        val benefit = SocialSecurity.annualPrimaryBenefitAtClaimAge(
            annualBenefitAtFullRetirementAge = 30_000.0,
            birthYear = 1958,
            claimAge = 67
        )

        assertEquals(30_800.0, benefit, 1.0)
    }

    @Test
    fun spousalBenefitUsesHalfPiaWithEarlyReduction() {
        val factor = SocialSecurity.spousalBenefitFactor(
            spouseBirthYear = 1960,
            spouseClaimAgeMonths = 62 * 12
        )

        assertEquals(0.325, factor, 0.001)
    }

    @Test
    fun survivorBenefitStartsAtSeventyOneAndHalfPercentAtAge60() {
        val factor = SocialSecurity.survivorBenefitFactor(
            spouseBirthYear = 1960,
            survivorClaimAgeMonths = 60 * 12
        )

        assertEquals(0.715, factor, 0.001)
    }

    @Test
    fun earlyWorkerClaimUsesWidowLimitAtSurvivorFra() {
        val factor = SocialSecurity.combinedSurvivorBenefitFactor(
            workerBirthYear = 1960,
            workerClaimAgeMonths = 62 * 12,
            workerDeathAgeMonths = 75 * 12,
            spouseBirthYear = 1960,
            survivorClaimAgeMonths = 67 * 12
        )

        assertEquals(0.825, factor, 0.001)
    }

    @Test
    fun earlySurvivorClaimIsNotReducedAgainByWorkerWidowLimit() {
        val factor = SocialSecurity.combinedSurvivorBenefitFactor(
            workerBirthYear = 1960,
            workerClaimAgeMonths = 62 * 12,
            workerDeathAgeMonths = 75 * 12,
            spouseBirthYear = 1960,
            survivorClaimAgeMonths = 60 * 12
        )

        assertEquals(0.715, factor, 0.001)
    }

    @Test
    fun workerDelayedCreditsEarnedBeforeDeathCarryToSurvivor() {
        val factor = SocialSecurity.combinedSurvivorBenefitFactor(
            workerBirthYear = 1960,
            workerClaimAgeMonths = 70 * 12,
            workerDeathAgeMonths = 69 * 12,
            spouseBirthYear = 1960,
            survivorClaimAgeMonths = 67 * 12
        )

        assertEquals(1.16, factor, 0.001)
    }

    @Test
    fun deathBeforeWorkerFraDoesNotApplyEarlyClaimReduction() {
        val factor = SocialSecurity.combinedSurvivorBenefitFactor(
            workerBirthYear = 1960,
            workerClaimAgeMonths = 70 * 12,
            workerDeathAgeMonths = 65 * 12,
            spouseBirthYear = 1960,
            survivorClaimAgeMonths = 67 * 12
        )

        assertEquals(1.0, factor, 0.001)
    }
}
