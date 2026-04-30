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
    const val PREMIUM_TABLE_VERSION = "2024 modeled Medicare Parts B/D with indexed IRMAA"

    private const val BASE_PART_B_MONTHLY = 174.70
    private const val BASE_PART_D_MONTHLY = 55.50

    private val singleTiers = listOf(
        IrmaaTier(103_000.0, 0.0, 0.0, "Base premium"),
        IrmaaTier(129_000.0, 69.90, 12.90, "IRMAA tier 1"),
        IrmaaTier(161_000.0, 174.70, 33.30, "IRMAA tier 2"),
        IrmaaTier(193_000.0, 279.50, 53.80, "IRMAA tier 3"),
        IrmaaTier(500_000.0, 384.30, 74.20, "IRMAA tier 4"),
        IrmaaTier(Double.POSITIVE_INFINITY, 419.30, 81.00, "IRMAA tier 5")
    )

    private val marriedTiers = listOf(
        IrmaaTier(206_000.0, 0.0, 0.0, "Base premium"),
        IrmaaTier(258_000.0, 69.90, 12.90, "IRMAA tier 1"),
        IrmaaTier(322_000.0, 174.70, 33.30, "IRMAA tier 2"),
        IrmaaTier(386_000.0, 279.50, 53.80, "IRMAA tier 3"),
        IrmaaTier(750_000.0, 384.30, 74.20, "IRMAA tier 4"),
        IrmaaTier(Double.POSITIVE_INFINITY, 419.30, 81.00, "IRMAA tier 5")
    )

    fun estimateAnnualPremium(
        modifiedAdjustedGrossIncome: Double,
        filingStatus: FilingStatus,
        coveredPeople: Int = 1,
        inflationMultiplier: Double = 1.0
    ): MedicarePremiumEstimate {
        val people = coveredPeople.coerceAtLeast(1)
        val multiplier = inflationMultiplier.coerceAtLeast(0.0)
        val tier = tiersFor(filingStatus)
            .first { modifiedAdjustedGrossIncome.coerceAtLeast(0.0) <= it.incomeCap * multiplier }
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
