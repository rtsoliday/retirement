package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.FilingStatus

data class MedicarePremiumEstimate(
    val annualPremium: Double,
    val monthlyPremium: Double,
    val partBIrmaaMonthly: Double,
    val partDIrmaaMonthly: Double,
    val tierLabel: String
)

object MedicarePremiums {
    const val PREMIUM_TABLE_VERSION = "2026 modeled Medicare Parts B/D with indexed IRMAA"

    private const val BASE_PART_B_MONTHLY = 202.90
    private const val BASE_PART_D_MONTHLY = 38.99

    private val singleTiers = listOf(
        IrmaaTier(109_000.0, 0.0, 0.0, "Base premium"),
        IrmaaTier(137_000.0, 81.20, 14.50, "IRMAA tier 1"),
        IrmaaTier(171_000.0, 202.90, 37.50, "IRMAA tier 2"),
        IrmaaTier(205_000.0, 324.60, 60.40, "IRMAA tier 3"),
        IrmaaTier(500_000.0, 446.30, 83.30, "IRMAA tier 4"),
        IrmaaTier(Double.POSITIVE_INFINITY, 487.00, 91.00, "IRMAA tier 5")
    )

    private val marriedTiers = listOf(
        IrmaaTier(218_000.0, 0.0, 0.0, "Base premium"),
        IrmaaTier(274_000.0, 81.20, 14.50, "IRMAA tier 1"),
        IrmaaTier(342_000.0, 202.90, 37.50, "IRMAA tier 2"),
        IrmaaTier(410_000.0, 324.60, 60.40, "IRMAA tier 3"),
        IrmaaTier(750_000.0, 446.30, 83.30, "IRMAA tier 4"),
        IrmaaTier(Double.POSITIVE_INFINITY, 487.00, 91.00, "IRMAA tier 5")
    )

    fun estimateAnnualPremium(
        modifiedAdjustedGrossIncome: Double,
        filingStatus: FilingStatus,
        coveredPeople: Int = 1,
        inflationMultiplier: Double = 1.0,
        incomeThresholdInflationMultiplier: Double = inflationMultiplier
    ): MedicarePremiumEstimate {
        val people = coveredPeople.coerceAtLeast(0)
        val multiplier = inflationMultiplier.coerceAtLeast(0.0)
        val thresholdMultiplier = incomeThresholdInflationMultiplier.coerceAtLeast(0.0001)
        val tier = tiersFor(filingStatus)
            .first { modifiedAdjustedGrossIncome.coerceAtLeast(0.0) <= it.incomeCap * thresholdMultiplier }
        val monthlyPerPerson = (
            BASE_PART_B_MONTHLY +
                BASE_PART_D_MONTHLY +
                tier.partBIrmaaMonthly +
                tier.partDIrmaaMonthly
            ) * multiplier
        val monthlyPremium = monthlyPerPerson * people.toDouble()
        return MedicarePremiumEstimate(
            annualPremium = monthlyPremium * 12.0,
            monthlyPremium = monthlyPremium,
            partBIrmaaMonthly = tier.partBIrmaaMonthly * multiplier * people.toDouble(),
            partDIrmaaMonthly = tier.partDIrmaaMonthly * multiplier * people.toDouble(),
            tierLabel = tier.label
        )
    }

    private fun tiersFor(filingStatus: FilingStatus): List<IrmaaTier> {
        return when (filingStatus) {
            FilingStatus.Married -> marriedTiers
            FilingStatus.Single,
            FilingStatus.HeadOfHousehold -> singleTiers
        }
    }
}

private data class IrmaaTier(
    val incomeCap: Double,
    val partBIrmaaMonthly: Double,
    val partDIrmaaMonthly: Double,
    val label: String
)
