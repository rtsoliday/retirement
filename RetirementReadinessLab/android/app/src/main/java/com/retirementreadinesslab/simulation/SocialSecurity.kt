package com.retirementreadinesslab.simulation

import kotlin.math.max
import kotlin.math.min

object SocialSecurity {
    const val MODEL_VERSION = "SSA retirement, spousal, and survivor rules modeled from 2026 ages"

    private const val MODEL_YEAR = 2026
    private const val MONTHS_PER_YEAR = 12
    private const val MIN_RETIREMENT_CLAIM_AGE_MONTHS = 62 * MONTHS_PER_YEAR
    private const val MAX_RETIREMENT_CLAIM_AGE_MONTHS = 70 * MONTHS_PER_YEAR
    private const val MIN_SURVIVOR_CLAIM_AGE_MONTHS = 60 * MONTHS_PER_YEAR

    // SSA references:
    // - Full retirement age rises by birth year and reaches 67 for people born in 1960 or later.
    // - Worker early retirement reduction is 5/9 of 1% for the first 36 months and 5/12 of 1%
    //   for additional months.
    // - Spousal full benefit is 50% of the worker PIA, with early reduction of 25/36 of 1%
    //   for the first 36 months and 5/12 of 1% for additional months.
    // - Delayed retirement credits for 1943+ birth years are 8% annually, or 2/3 of 1% monthly.
    // - SSA OACT notes widow(er) survivor FRA uses the retirement-age table with the survivor's
    //   birth year shifted back two years.
    // See SSA retirement planner pages and SSA OACT early-retirement reduction documentation.
    fun annualBenefitAtClaimAge(annualBenefitAt67: Double, claimAge: Int): Double {
        return annualPrimaryBenefitAtClaimAge(
            annualBenefitAtFullRetirementAge = annualBenefitAt67,
            birthYear = 1960,
            claimAge = claimAge
        )
    }

    fun annualPrimaryBenefitAtClaimAge(
        annualBenefitAtFullRetirementAge: Double,
        birthYear: Int,
        claimAge: Int
    ): Double {
        return annualBenefitAtFullRetirementAge *
            retirementBenefitFactor(birthYear, claimAge * MONTHS_PER_YEAR)
    }

    fun primaryBirthYear(currentAge: Int): Int = MODEL_YEAR - currentAge

    fun fullRetirementAgeMonths(birthYear: Int): Int {
        return when {
            birthYear <= 1937 -> 65 * MONTHS_PER_YEAR
            birthYear == 1938 -> 65 * MONTHS_PER_YEAR + 2
            birthYear == 1939 -> 65 * MONTHS_PER_YEAR + 4
            birthYear == 1940 -> 65 * MONTHS_PER_YEAR + 6
            birthYear == 1941 -> 65 * MONTHS_PER_YEAR + 8
            birthYear == 1942 -> 65 * MONTHS_PER_YEAR + 10
            birthYear in 1943..1954 -> 66 * MONTHS_PER_YEAR
            birthYear == 1955 -> 66 * MONTHS_PER_YEAR + 2
            birthYear == 1956 -> 66 * MONTHS_PER_YEAR + 4
            birthYear == 1957 -> 66 * MONTHS_PER_YEAR + 6
            birthYear == 1958 -> 66 * MONTHS_PER_YEAR + 8
            birthYear == 1959 -> 66 * MONTHS_PER_YEAR + 10
            else -> 67 * MONTHS_PER_YEAR
        }
    }

    fun fullRetirementAgeText(birthYear: Int): String {
        val months = fullRetirementAgeMonths(birthYear)
        return ageMonthsText(months)
    }

    fun survivorFullRetirementAgeMonths(birthYear: Int): Int {
        return fullRetirementAgeMonths(birthYear - 2)
    }

    fun survivorFullRetirementAgeText(birthYear: Int): String {
        return ageMonthsText(survivorFullRetirementAgeMonths(birthYear))
    }

    private fun ageMonthsText(months: Int): String {
        val years = months / MONTHS_PER_YEAR
        val extraMonths = months % MONTHS_PER_YEAR
        return if (extraMonths == 0) {
            years.toString()
        } else {
            "$years years, $extraMonths months"
        }
    }

    fun retirementBenefitFactor(birthYear: Int, claimAgeMonths: Int): Double {
        val fullRetirementAgeMonths = fullRetirementAgeMonths(birthYear)
        val boundedClaimAge = claimAgeMonths.coerceIn(
            MIN_RETIREMENT_CLAIM_AGE_MONTHS,
            MAX_RETIREMENT_CLAIM_AGE_MONTHS
        )
        return if (boundedClaimAge < fullRetirementAgeMonths) {
            1.0 - earlyRetirementReduction(fullRetirementAgeMonths - boundedClaimAge)
        } else {
            val delayedMonths = min(
                boundedClaimAge,
                MAX_RETIREMENT_CLAIM_AGE_MONTHS
            ) - fullRetirementAgeMonths
            1.0 + delayedMonths * delayedRetirementMonthlyCredit(birthYear)
        }
    }

    fun spousalBenefitFactor(spouseBirthYear: Int, spouseClaimAgeMonths: Int): Double {
        val fullSpousalFactor = 0.50
        val fullRetirementAgeMonths = fullRetirementAgeMonths(spouseBirthYear)
        val boundedClaimAge = spouseClaimAgeMonths.coerceIn(
            MIN_RETIREMENT_CLAIM_AGE_MONTHS,
            MAX_RETIREMENT_CLAIM_AGE_MONTHS
        )
        if (boundedClaimAge >= fullRetirementAgeMonths) return fullSpousalFactor

        val monthsEarly = fullRetirementAgeMonths - boundedClaimAge
        val first36Months = min(monthsEarly, 36)
        val additionalMonths = max(0, monthsEarly - 36)
        val reduction = first36Months * (25.0 / 36.0) / 100.0 +
            additionalMonths * (5.0 / 12.0) / 100.0
        return fullSpousalFactor * (1.0 - reduction)
    }

    fun survivorBenefitFactor(spouseBirthYear: Int, survivorClaimAgeMonths: Int): Double {
        val fullRetirementAgeMonths = survivorFullRetirementAgeMonths(spouseBirthYear)
        val boundedClaimAge = survivorClaimAgeMonths.coerceIn(
            MIN_SURVIVOR_CLAIM_AGE_MONTHS,
            MAX_RETIREMENT_CLAIM_AGE_MONTHS
        )
        if (boundedClaimAge >= fullRetirementAgeMonths) return 1.0

        val progress = (boundedClaimAge - MIN_SURVIVOR_CLAIM_AGE_MONTHS).toDouble() /
            (fullRetirementAgeMonths - MIN_SURVIVOR_CLAIM_AGE_MONTHS).toDouble()
        return 0.715 + progress.coerceIn(0.0, 1.0) * (1.0 - 0.715)
    }

    private fun earlyRetirementReduction(monthsEarly: Int): Double {
        val first36Months = min(monthsEarly, 36)
        val additionalMonths = max(0, monthsEarly - 36)
        return first36Months * (5.0 / 9.0) / 100.0 +
            additionalMonths * (5.0 / 12.0) / 100.0
    }

    private fun delayedRetirementMonthlyCredit(birthYear: Int): Double {
        val annualCredit = when {
            birthYear in 1933..1934 -> 0.055
            birthYear in 1935..1936 -> 0.060
            birthYear in 1937..1938 -> 0.065
            birthYear in 1939..1940 -> 0.070
            birthYear in 1941..1942 -> 0.075
            else -> 0.080
        }
        return annualCredit / MONTHS_PER_YEAR.toDouble()
    }
}
