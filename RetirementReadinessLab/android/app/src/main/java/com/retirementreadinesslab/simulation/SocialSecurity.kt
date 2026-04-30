package com.retirementreadinesslab.simulation

object SocialSecurity {
    fun annualBenefitAtClaimAge(annualBenefitAt67: Double, claimAge: Int): Double {
        if (claimAge == 67) return annualBenefitAt67

        val monthsDifference = (claimAge - 67) * 12
        return if (claimAge < 67) {
            val monthsEarly = -monthsDifference
            val reduction = if (monthsEarly <= 36) {
                monthsEarly * (5.0 / 9.0) / 100.0
            } else {
                (36.0 * (5.0 / 9.0) + (monthsEarly - 36.0) * (5.0 / 12.0)) / 100.0
            }
            annualBenefitAt67 * (1.0 - reduction)
        } else {
            val monthsLate = monthsDifference.coerceAtMost(36)
            val increase = monthsLate * (2.0 / 3.0) / 100.0
            annualBenefitAt67 * (1.0 + increase)
        }
    }
}
