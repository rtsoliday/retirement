package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.FilingStatus

object TaxCalculator {
    private val rates = listOf(0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37)

    private val singleBrackets = listOf(0.0, 11_600.0, 47_150.0, 100_525.0, 191_950.0, 243_725.0, 609_350.0)
    private val marriedBrackets = listOf(0.0, 23_200.0, 94_300.0, 201_050.0, 383_900.0, 487_450.0, 731_200.0)
    private val headOfHouseholdBrackets = listOf(0.0, 16_550.0, 63_100.0, 100_500.0, 191_950.0, 243_700.0, 609_350.0)

    fun taxLiability(taxableIncome: Double, filingStatus: FilingStatus): Double {
        if (taxableIncome <= 0.0) return 0.0

        val brackets = bracketsFor(filingStatus)
        var tax = 0.0
        for (index in rates.indices) {
            val lower = brackets[index]
            val upper = brackets.getOrNull(index + 1) ?: Double.POSITIVE_INFINITY
            if (taxableIncome <= lower) break

            val taxableAtBracket = minOf(taxableIncome, upper) - lower
            tax += taxableAtBracket * rates[index]
            if (taxableIncome <= upper) break
        }
        return tax
    }

    fun taxableSocialSecurity(
        otherIncome: Double,
        annualSocialSecurity: Double,
        filingStatus: FilingStatus
    ): Double {
        if (annualSocialSecurity <= 0.0) return 0.0

        val (base, second, secondTierBase) = when (filingStatus) {
            FilingStatus.Married -> Triple(32_000.0, 44_000.0, 6_000.0)
            FilingStatus.Single,
            FilingStatus.HeadOfHousehold -> Triple(25_000.0, 34_000.0, 4_500.0)
        }

        val provisionalIncome = otherIncome + annualSocialSecurity * 0.5
        return when {
            provisionalIncome <= base -> 0.0
            provisionalIncome <= second -> {
                minOf(0.5 * (provisionalIncome - base), 0.5 * annualSocialSecurity)
            }
            else -> {
                val taxable = 0.85 * (provisionalIncome - second) + secondTierBase
                minOf(taxable, 0.85 * annualSocialSecurity)
            }
        }
    }

    fun grossWithdrawalForNetNeed(
        netNeed: Double,
        annualSocialSecurity: Double,
        filingStatus: FilingStatus,
        annualOtherTaxableIncome: Double = 0.0
    ): Double {
        fun netCashFrom(withdrawal: Double): Double {
            val taxableSocialSecurity = taxableSocialSecurity(
                otherIncome = annualOtherTaxableIncome + withdrawal,
                annualSocialSecurity = annualSocialSecurity,
                filingStatus = filingStatus
            )
            val tax = taxLiability(
                taxableIncome = annualOtherTaxableIncome + withdrawal + taxableSocialSecurity,
                filingStatus = filingStatus
            )
            return withdrawal + annualOtherTaxableIncome + annualSocialSecurity - tax
        }

        if (netCashFrom(0.0) >= netNeed) return 0.0

        val neededAfterGuaranteedIncome = (netNeed - annualSocialSecurity - annualOtherTaxableIncome).coerceAtLeast(0.0)
        var low = 0.0
        var high = neededAfterGuaranteedIncome * 1.8 + 10_000.0

        repeat(40) {
            val mid = (low + high) / 2.0
            if (netCashFrom(mid) >= netNeed) {
                high = mid
            } else {
                low = mid
            }
        }
        return high
    }

    fun upperBracketLimitForRate(rateCap: Double, filingStatus: FilingStatus): Double? {
        val index = rates.indexOfFirst { kotlin.math.abs(it - rateCap) < 0.0001 }
        if (index < 0) return null
        return bracketsFor(filingStatus).getOrNull(index + 1)
    }

    private fun bracketsFor(filingStatus: FilingStatus): List<Double> {
        return when (filingStatus) {
            FilingStatus.Single -> singleBrackets
            FilingStatus.Married -> marriedBrackets
            FilingStatus.HeadOfHousehold -> headOfHouseholdBrackets
        }
    }
}
