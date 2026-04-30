package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.CalculationProvenance
import com.retirementreadinesslab.model.OutcomeBand
import com.retirementreadinesslab.model.FailureAgeBucket
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.RiskBreakdown
import com.retirementreadinesslab.model.RiskLevel
import com.retirementreadinesslab.model.SimulationResult
import com.retirementreadinesslab.model.validate
import java.util.Locale
import java.util.Random
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

object RetirementSimulator {
    private const val ENGINE_VERSION = "2026.04-medicare-parity"
    private const val ENGINE_CADENCE = "Monthly cashflow model with annual result bands"
    private const val TAX_TABLE_VERSION = "2024 federal brackets"
    private val MORTALITY_MODEL_VERSION = MortalityTables.TABLE_VERSION
    private const val MONTHS_PER_YEAR = 12
    private val MONTHLY_CASH_RETURN = monthlyEquivalent(0.02)

    fun run(scenario: RetirementScenario): SimulationResult {
        val validationErrors = scenario.validate()
        require(validationErrors.isEmpty()) { validationErrors.joinToString(" ") }

        val random = Random(scenario.seed)
        val yearlyBalances = MutableList(horizonYears(scenario) + 1) { mutableListOf<Double>() }
        val endingBalances = mutableListOf<Double>()
        val failureAges = mutableListOf<Int>()
        var successes = 0

        repeat(scenario.numberOfSimulations) {
            val path = runOneScenario(scenario, random)
            for ((index, balance) in path.yearEndBalances.withIndex()) {
                yearlyBalances[index] += balance
            }
            endingBalances += path.yearEndBalances.last()
            if (path.succeeded) {
                successes += 1
            } else if (path.failureAge != null) {
                failureAges += path.failureAge
            }
        }

        val successProbability = successes.toDouble() / scenario.numberOfSimulations.toDouble()
        val sortedEndings = endingBalances.sorted()
        val medianFailureAge = failureAges
            .takeIf { it.isNotEmpty() }
            ?.sorted()
            ?.let { percentileInt(it, 0.50) }
        val bands = yearlyBalances.mapIndexed { index, values ->
            val age = scenario.household.retirementAge + index
            val sortedValues = values.sorted()
            OutcomeBand(
                age = age,
                pessimistic = percentile(sortedValues, 0.10),
                median = percentile(sortedValues, 0.50),
                optimistic = percentile(sortedValues, 0.90)
            )
        }

        return SimulationResult(
            scenarioId = scenario.id,
            successProbability = successProbability,
            medianEndingBalance = percentile(sortedEndings, 0.50),
            pessimisticEndingBalance = percentile(sortedEndings, 0.10),
            optimisticEndingBalance = percentile(sortedEndings, 0.90),
            medianFailureAge = medianFailureAge,
            failureAgeBuckets = buildFailureBuckets(failureAges),
            balanceBands = bands,
            riskBreakdown = buildRiskBreakdown(scenario, successProbability),
            provenance = buildProvenance(scenario),
            generatedAtEpochMillis = System.currentTimeMillis()
        )
    }

    private fun buildProvenance(scenario: RetirementScenario): CalculationProvenance {
        return CalculationProvenance(
            engineVersion = ENGINE_VERSION,
            engineCadence = ENGINE_CADENCE,
            taxTableVersion = TAX_TABLE_VERSION,
            mortalityModelVersion = MORTALITY_MODEL_VERSION,
            randomSeed = scenario.seed,
            simulationCount = scenario.numberOfSimulations,
            assumptionFingerprint = assumptionFingerprint(scenario)
        )
    }

    private fun runOneScenario(scenario: RetirementScenario, random: Random): ScenarioPath {
        val balances = SimBalances(
            pretax = scenario.accounts.pretax,
            roth = scenario.accounts.roth,
            taxable = scenario.accounts.taxable,
            cash = scenario.accounts.cash
        )

        val yearsToRetirement = scenario.household.retirementAge - scenario.household.currentAge
        val preRetirementMonths = yearsToRetirement * MONTHS_PER_YEAR
        val preRetirementMonthlyMean = monthlyEquivalent(scenario.market.preRetirementMeanReturn)
        val preRetirementMonthlyStdDev = monthlyStdDev(scenario.market.preRetirementStdDev)
        repeat(preRetirementMonths) {
            val growth = samplePreRetirementMonthlyGrowth(random, preRetirementMonthlyMean, preRetirementMonthlyStdDev)
            balances.pretax *= 1.0 + growth
            balances.roth *= 1.0 + growth
            balances.taxable *= 1.0 + growth
            balances.cash *= 1.0 + MONTHLY_CASH_RETURN
        }

        val deathAge = sampleDeathAge(scenario, random)
        val ltcStartAge = sampleLongTermCareStartAge(scenario, deathAge, random)
        val claimBenefit = SocialSecurity.annualBenefitAtClaimAge(
            scenario.socialSecurity.annualBenefitAt67,
            scenario.socialSecurity.claimAge
        )
        val monthlyInflationMean = monthlyEquivalent(scenario.spending.generalInflationMean)
        val monthlyInflationStdDev = monthlyStdDev(scenario.spending.generalInflationStdDev)
        val monthlyHealthcareInflationMean = monthlyEquivalent(scenario.healthcare.healthcareInflationMean)
        val monthlyHealthcareInflationStdDev = monthlyStdDev(scenario.healthcare.healthcareInflationStdDev)
        var monthlySpending = scenario.spending.annualBaseSpending / MONTHS_PER_YEAR.toDouble() *
            compound(1.0 + monthlyInflationMean, preRetirementMonths)
        var monthlyHealthcare = scenario.healthcare.preMedicareMonthlyPremium *
            compound(1.0 + monthlyHealthcareInflationMean, preRetirementMonths)
        var medicareInflationMultiplier = compound(1.0 + monthlyHealthcareInflationMean, preRetirementMonths)
        val monthlyStockMean = monthlyEquivalent(scenario.market.stockMeanReturn)
        val monthlyStockStdDev = monthlyStdDev(scenario.market.stockStdDev)
        val monthlyBondMean = monthlyEquivalent(scenario.market.bondMeanReturn)
        val monthlyBondStdDev = monthlyStdDev(scenario.market.bondStdDev)

        val pathBalances = mutableListOf(balances.total)
        val annualMedicareIncomeHistory = mutableListOf<Double>()
        var currentAnnualMedicareIncome = 0.0
        var failureAge: Int? = null

        for (monthOffset in 0 until horizonMonths(scenario)) {
            val age = scenario.household.retirementAge + monthOffset / MONTHS_PER_YEAR
            val monthInYear = monthOffset % MONTHS_PER_YEAR
            if (age >= deathAge) {
                if (monthInYear == MONTHS_PER_YEAR - 1) {
                    pathBalances += balances.total
                }
                continue
            }

            val mortgageCost = if (monthOffset < mortgageMonthsInRetirement(scenario)) {
                scenario.mortgage.monthlyPayment
            } else {
                0.0
            }
            val inLongTermCare = ltcStartAge != null && age >= ltcStartAge
            val ltcCost = if (inLongTermCare) {
                scenario.longTermCare.annualCost / MONTHS_PER_YEAR.toDouble()
            } else {
                0.0
            }
            val socialSecurity = if (age >= scenario.socialSecurity.claimAge) {
                claimBenefit / MONTHS_PER_YEAR.toDouble()
            } else {
                0.0
            }
            val baseNeedBeforeHealthcare = monthlyRetirementNeed(
                monthlySpending = monthlySpending,
                mortgageCost = mortgageCost,
                healthcareCost = 0.0,
                longTermCareCost = ltcCost,
                inLongTermCare = inLongTermCare
            )
            val estimatedAnnualIncomeForIrmaa = baseNeedBeforeHealthcare
                .coerceAtLeast(0.0) * MONTHS_PER_YEAR.toDouble() +
                socialSecurity * MONTHS_PER_YEAR.toDouble()
            val healthcareCost = if (age < 65) {
                monthlyHealthcare
            } else if (scenario.healthcare.includeMedicarePremiums) {
                val irmaaIncome = medicareIncomeLookback(
                    annualMedicareIncomeHistory,
                    estimatedAnnualIncomeForIrmaa
                )
                MedicarePremiums
                    .estimateAnnualPremium(
                        modifiedAdjustedGrossIncome = irmaaIncome,
                        filingStatus = scenario.household.filingStatus,
                        inflationMultiplier = medicareInflationMultiplier
                    )
                    .monthlyPremium
            } else {
                0.0
            }
            val netNeed = monthlyRetirementNeed(
                monthlySpending = monthlySpending,
                mortgageCost = mortgageCost,
                healthcareCost = healthcareCost,
                longTermCareCost = ltcCost,
                inLongTermCare = inLongTermCare
            )

            val stockReturn = sampleNormal(random, monthlyStockMean, monthlyStockStdDev)
            val bondReturn = sampleNormal(random, monthlyBondMean, monthlyBondStdDev)
            val annualizedSpending = netNeed * MONTHS_PER_YEAR.toDouble()
            val stockAllocation = stockAllocation(balances.invested, annualizedSpending)
            val portfolioReturn = stockAllocation * stockReturn + (1.0 - stockAllocation) * bondReturn

            balances.pretax *= 1.0 + portfolioReturn
            balances.roth *= 1.0 + portfolioReturn
            balances.taxable *= 1.0 + portfolioReturn
            balances.cash *= 1.0 + MONTHLY_CASH_RETURN

            val grossWithdrawal = TaxCalculator.grossWithdrawalForNetNeed(
                netNeed = netNeed * MONTHS_PER_YEAR.toDouble(),
                annualSocialSecurity = socialSecurity * MONTHS_PER_YEAR.toDouble(),
                filingStatus = scenario.household.filingStatus
            ) / MONTHS_PER_YEAR.toDouble()

            if (scenario.withdrawalStrategy.useCashReserveDuringDrawdowns &&
                portfolioReturn < scenario.withdrawalStrategy.drawdownTrigger &&
                balances.cash > 0.0
            ) {
                val netAfterSocialSecurity = max(0.0, netNeed - socialSecurity)
                val cashDraw = minOf(balances.cash, netAfterSocialSecurity)
                balances.cash -= cashDraw
                withdrawStandard(grossWithdrawal - cashDraw, balances)
            } else {
                withdrawStandard(grossWithdrawal, balances)
            }

            val annualizedGrossWithdrawal = grossWithdrawal * MONTHS_PER_YEAR.toDouble()
            currentAnnualMedicareIncome += grossWithdrawal +
                TaxCalculator.taxableSocialSecurity(
                    otherIncome = annualizedGrossWithdrawal,
                    annualSocialSecurity = socialSecurity * MONTHS_PER_YEAR.toDouble(),
                    filingStatus = scenario.household.filingStatus
                ) / MONTHS_PER_YEAR.toDouble()

            if (scenario.rothConversion.enabled && monthInYear == 0) {
                val conversionLimit = TaxCalculator.upperBracketLimitForRate(
                    scenario.rothConversion.marginalRateCap,
                    scenario.household.filingStatus
                )
                if (conversionLimit != null && balances.pretax > 0.0) {
                    val conversion = minOf(balances.pretax, conversionLimit * 0.20)
                    val tax = TaxCalculator.taxLiability(conversion, scenario.household.filingStatus)
                    balances.pretax -= conversion
                    balances.roth += conversion
                    withdrawForConversionTax(tax, balances)
                    currentAnnualMedicareIncome += conversion
                }
            }

            val totalBalance = balances.total
            if (totalBalance < 0.0 && failureAge == null) {
                failureAge = age
                if (monthInYear != MONTHS_PER_YEAR - 1) {
                    pathBalances += totalBalance
                }
                break
            }
            if (monthInYear == MONTHS_PER_YEAR - 1) {
                pathBalances += totalBalance
            }

            val inflation = sampleNormal(
                random,
                monthlyInflationMean,
                monthlyInflationStdDev
            ).coerceAtLeast(monthlyEquivalent(-0.05))
            val healthcareInflation = sampleNormal(
                random,
                monthlyHealthcareInflationMean,
                monthlyHealthcareInflationStdDev
            ).coerceAtLeast(monthlyEquivalent(-0.02))

            monthlySpending *= 1.0 + inflation
            monthlyHealthcare *= 1.0 + healthcareInflation
            medicareInflationMultiplier *= 1.0 + healthcareInflation

            if (monthInYear == MONTHS_PER_YEAR - 1) {
                annualMedicareIncomeHistory += currentAnnualMedicareIncome
                currentAnnualMedicareIncome = 0.0
            }
        }

        while (pathBalances.size < horizonYears(scenario) + 1) {
            pathBalances += pathBalances.last()
        }

        return ScenarioPath(
            succeeded = failureAge == null,
            failureAge = failureAge,
            yearEndBalances = pathBalances
        )
    }

    private fun withdrawStandard(amount: Double, balances: SimBalances) {
        var remaining = amount.coerceAtLeast(0.0)
        remaining = drawFromPretax(balances, remaining)
        remaining = drawFromRoth(balances, remaining)
        remaining = drawFromTaxable(balances, remaining)
        balances.cash -= remaining
    }

    private fun withdrawForConversionTax(amount: Double, balances: SimBalances) {
        var remaining = amount.coerceAtLeast(0.0)
        remaining = drawFromCash(balances, remaining)
        remaining = drawFromRoth(balances, remaining)
        remaining = drawFromPretax(balances, remaining)
        balances.taxable -= remaining
    }

    private fun drawFromPretax(balances: SimBalances, requested: Double): Double {
        val draw = minOf(balances.pretax, requested)
        balances.pretax -= draw
        return requested - draw
    }

    private fun drawFromRoth(balances: SimBalances, requested: Double): Double {
        val draw = minOf(balances.roth, requested)
        balances.roth -= draw
        return requested - draw
    }

    private fun drawFromTaxable(balances: SimBalances, requested: Double): Double {
        val draw = minOf(balances.taxable, requested)
        balances.taxable -= draw
        return requested - draw
    }

    private fun drawFromCash(balances: SimBalances, requested: Double): Double {
        if (requested <= 0.0) return 0.0
        val draw = minOf(balances.cash, requested)
        balances.cash -= draw
        return requested - draw
    }

    private fun sampleNormal(random: Random, mean: Double, stdDev: Double): Double {
        return mean + random.nextGaussian() * stdDev
    }

    internal fun samplePreRetirementMonthlyGrowth(random: Random, mean: Double, stdDev: Double): Double {
        return sampleNormal(random, mean, stdDev)
    }

    internal fun monthlyRetirementNeed(
        monthlySpending: Double,
        mortgageCost: Double,
        healthcareCost: Double,
        longTermCareCost: Double,
        inLongTermCare: Boolean
    ): Double {
        return if (inLongTermCare) {
            longTermCareCost + healthcareCost
        } else {
            monthlySpending + mortgageCost + healthcareCost
        }
    }

    private fun stockAllocation(investedBalance: Double, annualSpending: Double): Double {
        if (annualSpending <= 0.0) return 0.50
        val ratio = investedBalance / annualSpending
        return when {
            ratio < 30.0 -> 1.00
            ratio < 35.0 -> 0.90
            ratio < 40.0 -> 0.80
            ratio < 45.0 -> 0.70
            ratio < 50.0 -> 0.60
            else -> 0.50
        }
    }

    private fun sampleDeathAge(scenario: RetirementScenario, random: Random): Int {
        var age = scenario.household.retirementAge
        while (age <= scenario.household.targetEndAge) {
            val deathProbability = MortalityTables.annualDeathProbability(scenario.household.gender, age)
            if (random.nextDouble() < deathProbability) return age
            age += 1
        }
        return scenario.household.targetEndAge + 1
    }

    private fun sampleLongTermCareStartAge(
        scenario: RetirementScenario,
        deathAge: Int,
        random: Random
    ): Int? {
        if (!scenario.longTermCare.enabled || deathAge < 65) return null
        val probability = when {
            deathAge < 75 -> 0.25
            deathAge < 85 -> 0.45
            deathAge < 95 -> 0.60
            else -> 0.70
        }
        if (random.nextDouble() > probability) return null
        return max(scenario.household.retirementAge, deathAge - scenario.longTermCare.averageDurationYears)
    }

    private fun mortgageMonthsInRetirement(scenario: RetirementScenario): Int {
        val yearsToRetirement = scenario.household.retirementAge - scenario.household.currentAge
        return max(0, scenario.mortgage.yearsLeft - yearsToRetirement) * MONTHS_PER_YEAR
    }

    private fun horizonYears(scenario: RetirementScenario): Int {
        return scenario.household.targetEndAge - scenario.household.retirementAge
    }

    private fun horizonMonths(scenario: RetirementScenario): Int {
        return horizonYears(scenario) * MONTHS_PER_YEAR
    }

    private fun compound(base: Double, years: Int): Double {
        var result = 1.0
        repeat(max(0, years)) {
            result *= base
        }
        return result
    }

    private fun monthlyEquivalent(annualRate: Double): Double {
        return (1.0 + annualRate).coerceAtLeast(0.0001).pow(1.0 / MONTHS_PER_YEAR.toDouble()) - 1.0
    }

    private fun monthlyStdDev(annualStdDev: Double): Double {
        return annualStdDev / sqrt(MONTHS_PER_YEAR.toDouble())
    }

    private fun medicareIncomeLookback(
        annualMedicareIncomeHistory: List<Double>,
        fallbackAnnualIncome: Double
    ): Double {
        return annualMedicareIncomeHistory
            .getOrNull(annualMedicareIncomeHistory.lastIndex - 1)
            ?: fallbackAnnualIncome
    }

    private fun percentile(sortedValues: List<Double>, percentile: Double): Double {
        if (sortedValues.isEmpty()) return 0.0
        val index = ((sortedValues.size - 1) * percentile).roundToInt()
        return sortedValues[index.coerceIn(0, sortedValues.lastIndex)]
    }

    private fun percentileInt(sortedValues: List<Int>, percentile: Double): Int {
        if (sortedValues.isEmpty()) return 0
        val index = ((sortedValues.size - 1) * percentile).roundToInt()
        return sortedValues[index.coerceIn(0, sortedValues.lastIndex)]
    }

    private fun buildFailureBuckets(failureAges: List<Int>): List<FailureAgeBucket> {
        if (failureAges.isEmpty()) return emptyList()
        val totalFailures = failureAges.size.toDouble()
        return failureAges
            .groupBy { age -> (age / 5) * 5 }
            .toSortedMap()
            .map { (bucketStart, ages) ->
                FailureAgeBucket(
                    label = "$bucketStart-${bucketStart + 4}",
                    count = ages.size,
                    shareOfFailures = ages.size / totalFailures
                )
            }
    }

    private fun buildRiskBreakdown(scenario: RetirementScenario, successProbability: Double): RiskBreakdown {
        val healthcareBurden = scenario.healthcare.preMedicareMonthlyPremium * 12.0 /
            scenario.spending.annualBaseSpending.coerceAtLeast(1.0)
        val taxBurden = scenario.accounts.pretax / scenario.accounts.total.coerceAtLeast(1.0)
        val spendingRatio = scenario.spending.annualBaseSpending / scenario.accounts.total.coerceAtLeast(1.0)
        val retirementAge = scenario.household.retirementAge

        val market = riskFrom(successProbability, 0.82, 0.65)
        val healthcare = when {
            healthcareBurden > 0.25 -> RiskLevel.AtRisk
            healthcareBurden > 0.18 || scenario.longTermCare.enabled -> RiskLevel.Watch
            else -> RiskLevel.Healthy
        }
        val taxes = when {
            taxBurden > 0.85 -> RiskLevel.AtRisk
            taxBurden > 0.60 -> RiskLevel.Watch
            else -> RiskLevel.Healthy
        }
        val spending = when {
            spendingRatio > 0.07 -> RiskLevel.AtRisk
            spendingRatio > 0.045 -> RiskLevel.Watch
            else -> RiskLevel.Healthy
        }
        val longevity = when {
            retirementAge < 55 -> RiskLevel.AtRisk
            retirementAge < 62 -> RiskLevel.Watch
            else -> RiskLevel.Healthy
        }

        val primaryRisk = listOf(
            "spending" to spending,
            "taxes" to taxes,
            "healthcare" to healthcare,
            "longevity" to longevity,
            "market sequence" to market
        ).firstOrNull { it.second == RiskLevel.AtRisk }?.first
            ?: listOf(
                "spending" to spending,
                "taxes" to taxes,
                "healthcare" to healthcare,
                "longevity" to longevity,
                "market sequence" to market
            ).firstOrNull { it.second == RiskLevel.Watch }?.first
            ?: "none"

        val nextTest = when (primaryRisk) {
            "spending" -> "Test a 5% lower spending scenario."
            "taxes" -> "Compare Roth conversions up to the 22% bracket."
            "healthcare" -> "Run the healthcare and long-term care stress test."
            "longevity" -> "Compare retiring two years later."
            "market sequence" -> "Test a larger cash reserve strategy."
            else -> "Compare Social Security claim ages."
        }

        return RiskBreakdown(
            market = market,
            longevity = longevity,
            healthcare = healthcare,
            taxes = taxes,
            spending = spending,
            primaryRisk = primaryRisk,
            recommendedNextTest = nextTest
        )
    }

    private fun riskFrom(value: Double, healthyAt: Double, watchAt: Double): RiskLevel {
        return when {
            value >= healthyAt -> RiskLevel.Healthy
            value >= watchAt -> RiskLevel.Watch
            else -> RiskLevel.AtRisk
        }
    }

    private fun assumptionFingerprint(scenario: RetirementScenario): String {
        val source = listOf(
            scenario.name,
            scenario.household.currentAge.toString(),
            scenario.household.retirementAge.toString(),
            scenario.household.targetEndAge.toString(),
            scenario.household.filingStatus.name,
            scenario.household.gender.name,
            scenario.accounts.pretax.fingerprintNumber(),
            scenario.accounts.roth.fingerprintNumber(),
            scenario.accounts.taxable.fingerprintNumber(),
            scenario.accounts.cash.fingerprintNumber(),
            scenario.spending.annualBaseSpending.fingerprintNumber(),
            scenario.spending.generalInflationMean.fingerprintNumber(),
            scenario.spending.generalInflationStdDev.fingerprintNumber(),
            scenario.mortgage.monthlyPayment.fingerprintNumber(),
            scenario.mortgage.yearsLeft.toString(),
            scenario.healthcare.preMedicareMonthlyPremium.fingerprintNumber(),
            scenario.healthcare.healthcareInflationMean.fingerprintNumber(),
            scenario.healthcare.healthcareInflationStdDev.fingerprintNumber(),
            scenario.healthcare.includeMedicarePremiums.toString(),
            scenario.socialSecurity.annualBenefitAt67.fingerprintNumber(),
            scenario.socialSecurity.claimAge.toString(),
            scenario.market.preRetirementMeanReturn.fingerprintNumber(),
            scenario.market.preRetirementStdDev.fingerprintNumber(),
            scenario.market.stockMeanReturn.fingerprintNumber(),
            scenario.market.stockStdDev.fingerprintNumber(),
            scenario.market.bondMeanReturn.fingerprintNumber(),
            scenario.market.bondStdDev.fingerprintNumber(),
            scenario.rothConversion.enabled.toString(),
            scenario.rothConversion.marginalRateCap.fingerprintNumber(),
            scenario.withdrawalStrategy.useCashReserveDuringDrawdowns.toString(),
            scenario.withdrawalStrategy.drawdownTrigger.fingerprintNumber(),
            scenario.longTermCare.enabled.toString(),
            scenario.longTermCare.annualCost.fingerprintNumber(),
            scenario.longTermCare.averageDurationYears.toString(),
            scenario.numberOfSimulations.toString(),
            scenario.seed.toString()
        ).joinToString("|")

        var hash = 1125899906842597L
        source.forEach { char ->
            hash = (hash * 31L) + char.code.toLong()
        }
        return java.lang.Long.toHexString(hash).uppercase(Locale.US).padStart(16, '0').takeLast(12)
    }

    private fun Double.fingerprintNumber(): String {
        return "%.6f".format(Locale.US, this)
    }

    private data class ScenarioPath(
        val succeeded: Boolean,
        val failureAge: Int?,
        val yearEndBalances: List<Double>
    )

    private data class SimBalances(
        var pretax: Double,
        var roth: Double,
        var taxable: Double,
        var cash: Double
    ) {
        val invested: Double
            get() = pretax + roth + taxable

        val total: Double
            get() = invested + cash
    }
}
