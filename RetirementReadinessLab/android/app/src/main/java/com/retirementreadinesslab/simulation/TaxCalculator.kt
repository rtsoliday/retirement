package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.FilingStatus

internal data class RothConversionPlan(
    val conversionAmount: Double,
    val additionalTax: Double
)

object TaxCalculator {
    private val rates = listOf(0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37)

    private val standardDeductions = mapOf(
        FilingStatus.Single to 16_100.0,
        FilingStatus.Married to 32_200.0,
        FilingStatus.HeadOfHousehold to 24_150.0
    )

    private val singleBrackets = listOf(0.0, 12_400.0, 50_400.0, 105_700.0, 201_775.0, 256_225.0, 640_600.0)
    private val marriedBrackets = listOf(0.0, 24_800.0, 100_800.0, 211_400.0, 403_550.0, 512_450.0, 768_700.0)
    private val headOfHouseholdBrackets = listOf(0.0, 17_700.0, 67_450.0, 105_700.0, 201_750.0, 256_200.0, 640_600.0)

    fun taxLiability(
        taxableIncome: Double,
        filingStatus: FilingStatus,
        inflationMultiplier: Double = 1.0
    ): Double {
        if (taxableIncome <= 0.0) return 0.0

        val multiplier = normalizedInflationMultiplier(inflationMultiplier)
        val brackets = bracketsFor(filingStatus)
        var tax = 0.0
        for (index in rates.indices) {
            val lower = brackets[index] * multiplier
            val upper = brackets.getOrNull(index + 1)
                ?.times(multiplier)
                ?: Double.POSITIVE_INFINITY
            if (taxableIncome <= lower) break

            val taxableAtBracket = minOf(taxableIncome, upper) - lower
            tax += taxableAtBracket * rates[index]
            if (taxableIncome <= upper) break
        }
        return tax
    }

    fun ordinaryIncomeTaxLiability(
        ordinaryIncome: Double,
        filingStatus: FilingStatus,
        inflationMultiplier: Double = 1.0
    ): Double {
        val multiplier = normalizedInflationMultiplier(inflationMultiplier)
        val taxableIncome = (
            ordinaryIncome.coerceAtLeast(0.0) -
                standardDeductions.getValue(filingStatus) * multiplier
            ).coerceAtLeast(0.0)
        return taxLiability(taxableIncome, filingStatus, multiplier)
    }

    fun taxableSocialSecurity(
        otherIncome: Double,
        annualSocialSecurity: Double,
        filingStatus: FilingStatus
    ): Double {
        if (annualSocialSecurity <= 0.0) return 0.0

        val (base, second, maximumSecondTierBase) = when (filingStatus) {
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
                val secondTierBase = minOf(maximumSecondTierBase, 0.5 * annualSocialSecurity)
                val taxable = 0.85 * (provisionalIncome - second) + secondTierBase
                minOf(taxable, 0.85 * annualSocialSecurity)
            }
        }
    }

    fun grossWithdrawalForNetNeed(
        netNeed: Double,
        annualSocialSecurity: Double,
        filingStatus: FilingStatus,
        annualOtherTaxableIncome: Double = 0.0,
        additionalWithdrawalTaxRate: Double = 0.0,
        additionalWithdrawalTaxableLimit: Double = Double.POSITIVE_INFINITY,
        inflationMultiplier: Double = 1.0
    ): Double {
        val additionalRate = additionalWithdrawalTaxRate.coerceAtLeast(0.0)
        val additionalTaxableLimit = additionalWithdrawalTaxableLimit.coerceAtLeast(0.0)

        fun netCashFrom(withdrawal: Double): Double {
            val taxableSocialSecurity = taxableSocialSecurity(
                otherIncome = annualOtherTaxableIncome + withdrawal,
                annualSocialSecurity = annualSocialSecurity,
                filingStatus = filingStatus
            )
            val tax = ordinaryIncomeTaxLiability(
                ordinaryIncome = annualOtherTaxableIncome + withdrawal + taxableSocialSecurity,
                filingStatus = filingStatus,
                inflationMultiplier = inflationMultiplier
            )
            val additionalWithdrawalTax = minOf(withdrawal, additionalTaxableLimit) * additionalRate
            return withdrawal + annualOtherTaxableIncome + annualSocialSecurity - tax - additionalWithdrawalTax
        }

        if (netCashFrom(0.0) >= netNeed) return 0.0

        val neededAfterGuaranteedIncome = (netNeed - annualSocialSecurity - annualOtherTaxableIncome).coerceAtLeast(0.0)
        var low = 0.0
        var high = neededAfterGuaranteedIncome * 1.8 + 10_000.0
        while (netCashFrom(high) < netNeed) {
            high *= 2.0
        }

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

    fun upperBracketLimitForRate(
        rateCap: Double,
        filingStatus: FilingStatus,
        inflationMultiplier: Double = 1.0
    ): Double? {
        val index = rates.indexOfFirst { kotlin.math.abs(it - rateCap) < 0.0001 }
        if (index < 0) return null
        return bracketsFor(filingStatus).getOrNull(index + 1)
            ?.times(normalizedInflationMultiplier(inflationMultiplier))
    }

    internal fun rothConversionPlan(
        pretaxBalance: Double,
        currentTaxableIncome: Double,
        rateCap: Double,
        filingStatus: FilingStatus,
        inflationMultiplier: Double = 1.0
    ): RothConversionPlan {
        val availablePretax = pretaxBalance.coerceAtLeast(0.0)
        if (availablePretax <= 0.0) return RothConversionPlan(0.0, 0.0)

        val rateIndex = rates.indexOfFirst { kotlin.math.abs(it - rateCap) < 0.0001 }
        if (rateIndex < 0) return RothConversionPlan(0.0, 0.0)

        val ordinaryIncome = currentTaxableIncome.coerceAtLeast(0.0)
        val multiplier = normalizedInflationMultiplier(inflationMultiplier)
        val upperLimit = bracketsFor(filingStatus).getOrNull(rateIndex + 1)?.times(multiplier)
        val conversion = if (upperLimit == null) {
            availablePretax
        } else {
            val bracketRoomBeforeDeduction = upperLimit + standardDeductions.getValue(filingStatus) * multiplier
            minOf(availablePretax, (bracketRoomBeforeDeduction - ordinaryIncome).coerceAtLeast(0.0))
        }
        if (conversion <= 0.0) return RothConversionPlan(0.0, 0.0)

        val additionalTax = (
            ordinaryIncomeTaxLiability(ordinaryIncome + conversion, filingStatus, multiplier) -
                ordinaryIncomeTaxLiability(ordinaryIncome, filingStatus, multiplier)
        ).coerceAtLeast(0.0)
        return RothConversionPlan(
            conversionAmount = conversion,
            additionalTax = additionalTax
        )
    }

    private fun bracketsFor(filingStatus: FilingStatus): List<Double> {
        return when (filingStatus) {
            FilingStatus.Single -> singleBrackets
            FilingStatus.Married -> marriedBrackets
            FilingStatus.HeadOfHousehold -> headOfHouseholdBrackets
        }
    }

    private fun normalizedInflationMultiplier(value: Double): Double {
        return value.takeIf { it.isFinite() }?.coerceAtLeast(0.0001) ?: 1.0
    }
}
