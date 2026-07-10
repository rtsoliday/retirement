package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.CalculationProvenance
import com.retirementreadinesslab.model.EARLY_WITHDRAWAL_PENALTY_RATE
import com.retirementreadinesslab.model.OutcomeBand
import com.retirementreadinesslab.model.FailureAgeBucket
import com.retirementreadinesslab.model.FilingStatus
import com.retirementreadinesslab.model.FundingThreshold
import com.retirementreadinesslab.model.Gender
import com.retirementreadinesslab.model.PENALTY_FREE_WITHDRAWAL_AGE_MONTHS
import com.retirementreadinesslab.model.PortfolioSurvivalPoint
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.RiskBreakdown
import com.retirementreadinesslab.model.RiskLevel
import com.retirementreadinesslab.model.RULE_OF_55_MINIMUM_RETIREMENT_AGE
import com.retirementreadinesslab.model.SENIOR_APARTMENT_MONTHLY_RENT_2026
import com.retirementreadinesslab.model.SEPP_DEFAULT_INTEREST_RATE
import com.retirementreadinesslab.model.SEPP_MINIMUM_PAYMENT_MONTHS
import com.retirementreadinesslab.model.SimulationMeanPoint
import com.retirementreadinesslab.model.SimulationPathPoint
import com.retirementreadinesslab.model.SimulationResult
import com.retirementreadinesslab.model.SpendingPathModel
import com.retirementreadinesslab.model.WithdrawalStrategy
import com.retirementreadinesslab.model.validate
import java.util.Locale
import java.util.Random
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

object RetirementSimulator {
    private const val ENGINE_VERSION = "2026.06-sepp"
    private const val ENGINE_CADENCE = "Monthly cashflow model with annual result bands"
    private const val TAX_TABLE_VERSION = "2024 federal brackets"
    private val MORTALITY_MODEL_VERSION = MortalityTables.TABLE_VERSION
    private const val MONTHS_PER_YEAR = 12
    private const val MAX_PATH_SCATTER_POINTS = 30_000
    private const val BALANCE_EPSILON = 0.01
    private const val FUNDING_THRESHOLD_TARGET_READINESS = 0.95
    private const val FUNDING_THRESHOLD_WINDOW_SIZE = 60
    private val MONTHLY_CASH_RETURN = monthlyEquivalent(0.02)

    // Hurd and Rohwedder (2023), "Spending trajectories after age 65: variation by initial wealth",
    // Journal of the Economics of Ageing 26, 100468, doi:10.1016/j.jeoa.2023.100468, estimates
    // HRS/CAMS 2005-2019 real spending declines of about 1.7%/yr for singles and 2.4%/yr for
    // coupled households after age 65.
    private const val EMPIRICAL_SPENDING_DECLINE_START_AGE = 65
    private const val EMPIRICAL_SPENDING_DECLINE_END_AGE = 85
    private const val EMPIRICAL_SINGLE_REAL_SPENDING_DECLINE = 0.017
    private const val EMPIRICAL_MARRIED_REAL_SPENDING_DECLINE = 0.024
    private const val EMPIRICAL_SINGLE_SPENDING_FLOOR = 0.65
    private const val EMPIRICAL_MARRIED_SPENDING_FLOOR = 0.60
    // Same Hurd and Rohwedder 2023 paper reports a 16% median household spending drop
    // when a coupled household transitions to widowhood between adjacent CAMS waves.
    private const val WIDOWHOOD_SPENDING_REDUCTION = 0.16

    fun run(scenario: RetirementScenario): SimulationResult {
        val validationErrors = scenario.validate()
        require(validationErrors.isEmpty()) { validationErrors.joinToString(" ") }

        val random = Random(scenario.seed)
        val yearlyBalances = MutableList(horizonYears(scenario) + 1) { mutableListOf<Double>() }
        val endingBalances = mutableListOf<Double>()
        val failureAges = mutableListOf<Int>()
        val paths = mutableListOf<ScenarioPath>()
        var successes = 0

        repeat(scenario.numberOfSimulations) {
            val path = runOneScenario(scenario, random)
            paths += path
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
            notFailedByAge = buildNotFailedByAge(scenario, paths),
            pathPoints = buildPathPoints(paths),
            meanPath = buildMeanPath(paths),
            fundingThreshold = buildFundingThreshold(paths),
            riskBreakdown = buildRiskBreakdown(scenario, successProbability),
            provenance = buildProvenance(scenario),
            generatedAtEpochMillis = System.currentTimeMillis()
        )
    }

    private fun buildNotFailedByAge(
        scenario: RetirementScenario,
        paths: List<ScenarioPath>
    ): List<PortfolioSurvivalPoint> {
        if (paths.isEmpty()) return emptyList()
        val simulationCount = paths.size.toDouble()
        val maxSurvivedAge = paths.maxOf { it.survivedThroughAge }
            .coerceAtLeast(scenario.household.retirementAge)
        return (scenario.household.retirementAge..maxSurvivedAge).map { age ->
            val notFailedCount = paths.count { path ->
                path.failureAge == null || path.failureAge > age
            }
            val aliveCount = paths.count { path ->
                path.survivedThroughAge >= age
            }
            PortfolioSurvivalPoint(
                age = age,
                notFailedShare = notFailedCount.toDouble() / simulationCount,
                aliveShare = aliveCount.toDouble() / simulationCount
            )
        }
    }

    private fun buildPathPoints(paths: List<ScenarioPath>): List<SimulationPathPoint> {
        if (paths.isEmpty()) return emptyList()
        val years = paths.maxOf { it.chartBalances.size }
        val failedMaxByYear = DoubleArray(years) { Double.NEGATIVE_INFINITY }
        val successMinByYear = DoubleArray(years) { Double.POSITIVE_INFINITY }
        var positivePointCount = 0

        paths.forEach { path ->
            path.chartBalances.forEachIndexed { year, balance ->
                if (balance > 0.0) {
                    positivePointCount += 1
                    if (path.succeeded) {
                        successMinByYear[year] = minOf(successMinByYear[year], balance)
                    } else {
                        failedMaxByYear[year] = maxOf(failedMaxByYear[year], balance)
                    }
                }
            }
        }

        if (positivePointCount == 0) return emptyList()
        val stride = max(1, ceil(positivePointCount.toDouble() / MAX_PATH_SCATTER_POINTS.toDouble()).toInt())
        val points = mutableListOf<SimulationPathPoint>()
        var positiveIndex = 0
        paths.forEach { path ->
            path.chartBalances.forEachIndexed { year, balance ->
                if (balance > 0.0) {
                    if (positiveIndex % stride == 0) {
                        points += SimulationPathPoint(
                            yearsInRetirement = year,
                            balance = balance,
                            successfulPath = path.succeeded,
                            separatedFromOppositeOutcome = if (path.succeeded) {
                                balance > failedMaxByYear[year]
                            } else {
                                balance < successMinByYear[year]
                            }
                        )
                    }
                    positiveIndex += 1
                }
            }
        }
        return points
    }

    private fun buildFundingThreshold(paths: List<ScenarioPath>): FundingThreshold? {
        return fundingThresholdFromCurrentRun(
            paths.mapNotNull { path ->
                val retirementStartBalance = path.chartBalances.firstOrNull() ?: return@mapNotNull null
                retirementStartBalance to path.succeeded
            }
        )
    }

    internal fun fundingThresholdFromCurrentRun(
        retirementStartOutcomes: List<Pair<Double, Boolean>>,
        targetReadiness: Double = FUNDING_THRESHOLD_TARGET_READINESS
    ): FundingThreshold? {
        if (targetReadiness <= 0.0 || targetReadiness > 1.0) return null
        val sortedOutcomes = retirementStartOutcomes
            .filter { (balance, _) -> balance.isFinite() && balance >= 0.0 }
            .sortedByDescending { (balance, _) -> balance }
        if (sortedOutcomes.isEmpty()) return null

        val windowSize = FUNDING_THRESHOLD_WINDOW_SIZE
        if (sortedOutcomes.size < windowSize) return null
        val failureLimit = fundingThresholdFailureLimit(windowSize, targetReadiness)
        val firstWindowSuccessCount = sortedOutcomes
            .take(windowSize)
            .count { (_, succeeded) -> succeeded }
        if (firstWindowSuccessCount < windowSize - failureLimit) return null

        val window = mutableListOf<Pair<Double, Boolean>>()
        var failureCount = 0

        sortedOutcomes.forEach { outcome ->
            window += outcome
            if (!outcome.second) failureCount += 1
            if (window.size > windowSize) {
                val removed = window.removeAt(0)
                if (!removed.second) failureCount -= 1
            }
            if (window.size == windowSize && fundingThresholdReached(failureCount, failureLimit)) {
                val medianFailureBalance = medianFailedRetirementStartBalance(window) ?: return null
                return FundingThreshold(
                    balance = medianFailureBalance,
                    targetReadiness = targetReadiness,
                    observedReadiness = (windowSize - failureCount).toDouble() / windowSize.toDouble(),
                    includedSimulationCount = windowSize,
                    totalSimulationCount = sortedOutcomes.size
                )
            }
        }
        return null
    }

    private fun fundingThresholdFailureLimit(windowSize: Int, targetReadiness: Double): Int {
        return (windowSize.toDouble() * (1.0 - targetReadiness))
            .roundToInt()
            .coerceIn(0, windowSize)
    }

    private fun fundingThresholdReached(failureCount: Int, failureLimit: Int): Boolean {
        return if (failureLimit == 0) {
            failureCount == 0
        } else {
            failureCount >= failureLimit
        }
    }

    private fun medianFailedRetirementStartBalance(window: List<Pair<Double, Boolean>>): Double? {
        val balances = window
            .filter { (_, succeeded) -> !succeeded }
            .map { (balance, _) -> balance }
            .sorted()
        if (balances.isEmpty()) return null
        val middle = balances.size / 2
        return if (balances.size % 2 == 0) {
            (balances[middle - 1] + balances[middle]) / 2.0
        } else {
            balances[middle]
        }
    }

    private fun buildMeanPath(paths: List<ScenarioPath>): List<SimulationMeanPoint> {
        if (paths.isEmpty()) return emptyList()
        val years = paths.maxOf { it.chartBalances.size }
        val sums = DoubleArray(years)
        val counts = IntArray(years)
        paths.forEach { path ->
            path.chartBalances.forEachIndexed { year, balance ->
                if (balance > 0.0) {
                    sums[year] += balance
                    counts[year] += 1
                }
            }
        }
        return sums.indices.mapNotNull { year ->
            if (counts[year] > 0) {
                SimulationMeanPoint(yearsInRetirement = year, balance = sums[year] / counts[year].toDouble())
            } else {
                null
            }
        }
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

        val married = scenario.household.filingStatus == FilingStatus.Married
        val spouseAgeAtRetirement = spouseAgeAtRetirement(scenario)
        val primaryDeathAge = sampleDeathAge(
            gender = scenario.household.gender,
            startAge = scenario.household.retirementAge,
            targetEndAge = scenario.household.targetEndAge,
            random = random
        )
        val spouseDeathAge = if (married) {
            sampleDeathAge(
                gender = scenario.household.gender.opposite(),
                startAge = spouseAgeAtRetirement,
                targetEndAge = scenario.household.targetEndAge,
                random = random
            )
        } else {
            primaryDeathAge
        }
        val spouseDeathPrimaryAge = scenario.household.retirementAge + (spouseDeathAge - spouseAgeAtRetirement)
        val householdDeathAge = if (married) {
            max(primaryDeathAge, spouseDeathPrimaryAge)
        } else {
            primaryDeathAge
        }
        val primaryLtcStartAge = sampleLongTermCareStartAge(
            scenario = scenario,
            startAge = scenario.household.retirementAge,
            deathAge = primaryDeathAge,
            random = random
        )
        val spouseLtcStartPrimaryAge = if (married) {
            sampleLongTermCareStartAge(
                scenario = scenario,
                startAge = spouseAgeAtRetirement,
                deathAge = spouseDeathAge,
                random = random
            )?.let { spouseLtcStartAge ->
                scenario.household.retirementAge + (spouseLtcStartAge - spouseAgeAtRetirement)
            }
        } else {
            null
        }
        val primaryBirthYear = SocialSecurity.primaryBirthYear(scenario.household.currentAge)
        val spouseBirthYear = SocialSecurity.primaryBirthYear(scenario.household.spouseCurrentAge)
        val primaryBenefitFactor = SocialSecurity.retirementBenefitFactor(
            primaryBirthYear,
            scenario.socialSecurity.claimAge * MONTHS_PER_YEAR
        )
        val primarySurvivorBaseFactor = if (primaryDeathAge < scenario.socialSecurity.claimAge) {
            1.0
        } else {
            primaryBenefitFactor
        }
        val monthlyInflationMean = monthlyEquivalent(scenario.spending.generalInflationMean)
        val monthlyInflationStdDev = monthlyStdDev(scenario.spending.generalInflationStdDev)
        val monthlyHealthcareInflationMean = monthlyEquivalent(scenario.healthcare.healthcareInflationMean)
        val monthlyHealthcareInflationStdDev = monthlyStdDev(scenario.healthcare.healthcareInflationStdDev)
        val monthlyGuaranteedIncomeIncrease = monthlyEquivalent(scenario.guaranteedIncome.annualIncrease)
        val initialRetirementBalance = balances.total
        val lowPortfolioThreshold = initialRetirementBalance * 0.50
        var spendingPathMultiplier = spendingPathMultiplierAtRetirementMonth(scenario, monthOffset = 0)
        var monthlySpending = scenario.spending.annualBaseSpending / MONTHS_PER_YEAR.toDouble() *
            compound(1.0 + monthlyInflationMean, preRetirementMonths) *
            spendingPathMultiplier
        var monthlyRent = scenario.rent.monthlyRent *
            compound(1.0 + monthlyInflationMean, preRetirementMonths)
        var monthlyHomeValue = scenario.home.currentValue *
            compound(1.0 + monthlyInflationMean, preRetirementMonths)
        var remainingMortgageBalance = mortgageBalanceAtRetirement(scenario)
        var remainingMortgageMonths = mortgageMonthsInRetirement(scenario)
        var monthlySeniorApartmentRent = SENIOR_APARTMENT_MONTHLY_RENT_2026 *
            compound(1.0 + monthlyInflationMean, preRetirementMonths)
        var monthlyGuaranteedIncome = scenario.guaranteedIncome.annualIncome / MONTHS_PER_YEAR.toDouble() *
            compound(1.0 + monthlyGuaranteedIncomeIncrease, preRetirementMonths)
        var monthlyHomeOwnershipCostInBase =
            (scenario.budget.annualPropertyTaxes + scenario.budget.annualHomeInsurance) /
                MONTHS_PER_YEAR.toDouble() *
                compound(1.0 + monthlyInflationMean, preRetirementMonths) *
                spendingPathMultiplier
        var monthlyHealthcare = scenario.healthcare.preMedicareMonthlyPremium *
            compound(1.0 + monthlyHealthcareInflationMean, preRetirementMonths)
        var medicareInflationMultiplier = compound(1.0 + monthlyHealthcareInflationMean, preRetirementMonths)
        var socialSecurityInflationMultiplier = compound(1.0 + monthlyInflationMean, preRetirementMonths)
        val monthlyStockMean = monthlyEquivalent(scenario.market.stockMeanReturn)
        val monthlyStockStdDev = monthlyStdDev(scenario.market.stockStdDev)
        val monthlyBondMean = monthlyEquivalent(scenario.market.bondMeanReturn)
        val monthlyBondStdDev = monthlyStdDev(scenario.market.bondStdDev)
        val seppEndAgeMonths = max(
            PENALTY_FREE_WITHDRAWAL_AGE_MONTHS,
            scenario.household.retirementAge * MONTHS_PER_YEAR + SEPP_MINIMUM_PAYMENT_MONTHS
        )
        val annualSeppPayment = if (scenario.withdrawalStrategy.seppEligible) {
            seppFixedAmortizationAnnualPayment(
                accountBalance = balances.pretax,
                age = scenario.household.retirementAge
            )
        } else {
            0.0
        }

        val pathBalances = mutableListOf(balances.total)
        val chartBalances = mutableListOf(balances.total)
        val annualMedicareIncomeHistory = mutableListOf<Double>()
        var currentAnnualMedicareIncome = 0.0
        var failureAge: Int? = null
        var homeSold = false

        for (monthOffset in 0 until horizonMonths(scenario)) {
            val age = scenario.household.retirementAge + monthOffset / MONTHS_PER_YEAR
            val spouseAge = spouseAgeAtRetirement + monthOffset / MONTHS_PER_YEAR
            val primaryAgeMonths = scenario.household.retirementAge * MONTHS_PER_YEAR + monthOffset
            val spouseAgeMonths = spouseAgeAtRetirement * MONTHS_PER_YEAR + monthOffset
            val monthInYear = monthOffset % MONTHS_PER_YEAR
            if (age >= householdDeathAge) {
                break
            }
            val primaryAlive = age < primaryDeathAge
            val spouseAlive = married && age < spouseDeathPrimaryAge
            val bothSpousesAlive = married && primaryAlive && spouseAlive
            val alivePeople = (if (primaryAlive) 1 else 0) + (if (spouseAlive) 1 else 0)
            val currentFilingStatus = if (married) {
                if (bothSpousesAlive) FilingStatus.Married else FilingStatus.Single
            } else {
                scenario.household.filingStatus
            }

            val primaryInLongTermCare = primaryAlive &&
                primaryLtcStartAge != null &&
                age >= primaryLtcStartAge
            val spouseInLongTermCare = spouseAlive &&
                spouseLtcStartPrimaryAge != null &&
                age >= spouseLtcStartPrimaryAge
            val longTermCarePeople = (if (primaryInLongTermCare) 1 else 0) +
                (if (spouseInLongTermCare) 1 else 0)
            val peopleOutsideLongTermCare = alivePeople - longTermCarePeople
            val replaceBaseSpendingWithLongTermCare = alivePeople > 0 && peopleOutsideLongTermCare == 0

            val mortgageCost = if (peopleOutsideLongTermCare > 0 && !homeSold && remainingMortgageMonths > 0) {
                scenario.mortgage.monthlyPayment
            } else {
                0.0
            }
            val rentCost = if (peopleOutsideLongTermCare <= 0) {
                0.0
            } else if (homeSold) {
                monthlySeniorApartmentRent
            } else {
                monthlyRent
            }
            val housingAdjustedSpending = if (homeSold) {
                (monthlySpending - monthlyHomeOwnershipCostInBase).coerceAtLeast(0.0)
            } else {
                monthlySpending
            }
            val spendingReduction = if (
                initialRetirementBalance > 0.0 &&
                balances.total < lowPortfolioThreshold
            ) {
                scenario.spending.lowPortfolioSpendingReduction
            } else {
                0.0
            }
            val widowhoodSpendingMultiplier = if (married && !bothSpousesAlive) {
                1.0 - WIDOWHOOD_SPENDING_REDUCTION
            } else {
                1.0
            }
            val baseSpending = housingAdjustedSpending *
                (1.0 - spendingReduction) *
                widowhoodSpendingMultiplier
            val ltcCost = if (longTermCarePeople > 0) {
                scenario.longTermCare.annualCost / MONTHS_PER_YEAR.toDouble() *
                    longTermCarePeople.toDouble()
            } else {
                0.0
            }
            val socialSecurity = monthlySocialSecurityBenefit(
                scenario = scenario,
                married = married,
                primaryAlive = primaryAlive,
                spouseAlive = spouseAlive,
                primaryAgeMonths = primaryAgeMonths,
                spouseAgeMonths = spouseAgeMonths,
                primaryBenefitFactor = primaryBenefitFactor,
                primarySurvivorBaseFactor = primarySurvivorBaseFactor,
                spouseBirthYear = spouseBirthYear,
                inflationMultiplier = socialSecurityInflationMultiplier
            )
            val guaranteedIncome = if (age >= scenario.guaranteedIncome.startAge && alivePeople > 0) {
                if (primaryAlive) {
                    monthlyGuaranteedIncome
                } else {
                    monthlyGuaranteedIncome * scenario.guaranteedIncome.survivorPercent
                }
            } else {
                0.0
            }
            val baseNeedBeforeHealthcare = monthlyRetirementNeed(
                monthlySpending = baseSpending,
                mortgageCost = mortgageCost,
                rentCost = rentCost,
                healthcareCost = 0.0,
                longTermCareCost = ltcCost,
                replaceBaseSpendingWithLongTermCare = replaceBaseSpendingWithLongTermCare
            )
            val estimatedAnnualIncomeForIrmaa = baseNeedBeforeHealthcare
                .coerceAtLeast(0.0) * MONTHS_PER_YEAR.toDouble() +
                socialSecurity * MONTHS_PER_YEAR.toDouble() +
                guaranteedIncome * MONTHS_PER_YEAR.toDouble()
            val preMedicarePeople = (if (primaryAlive && age < 65) 1 else 0) +
                (if (spouseAlive && spouseAge < 65) 1 else 0)
            val medicarePeople = (if (primaryAlive && age >= 65) 1 else 0) +
                (if (spouseAlive && spouseAge >= 65) 1 else 0)
            val preMedicareCost = monthlyHealthcare * preMedicarePeople.toDouble()
            val medicareCost = if (scenario.healthcare.includeMedicarePremiums && medicarePeople > 0) {
                val irmaaIncome = medicareIncomeLookback(
                    annualMedicareIncomeHistory,
                    estimatedAnnualIncomeForIrmaa
                )
                MedicarePremiums
                    .estimateAnnualPremium(
                        modifiedAdjustedGrossIncome = irmaaIncome,
                        filingStatus = currentFilingStatus,
                        coveredPeople = medicarePeople,
                        inflationMultiplier = medicareInflationMultiplier
                    )
                    .monthlyPremium
            } else {
                0.0
            }
            val healthcareCost = preMedicareCost + medicareCost
            val netNeed = monthlyRetirementNeed(
                monthlySpending = baseSpending,
                mortgageCost = mortgageCost,
                rentCost = rentCost,
                healthcareCost = healthcareCost,
                longTermCareCost = ltcCost,
                replaceBaseSpendingWithLongTermCare = replaceBaseSpendingWithLongTermCare
            )

            val stockReturn = sampleNormal(random, monthlyStockMean, monthlyStockStdDev)
            val bondReturn = sampleNormal(random, monthlyBondMean, monthlyBondStdDev)
            val annualizedSpending = netNeed * MONTHS_PER_YEAR.toDouble()
            val stockAllocation = scenario.postRetirementAllocation.stockAllocation(
                investedBalance = balances.invested,
                annualSpending = annualizedSpending
            )
            val portfolioReturn = stockAllocation * stockReturn + (1.0 - stockAllocation) * bondReturn

            balances.pretax *= 1.0 + portfolioReturn
            balances.roth *= 1.0 + portfolioReturn
            balances.taxable *= 1.0 + portfolioReturn
            balances.cash *= 1.0 + MONTHLY_CASH_RETURN

            val seppActive = scenario.withdrawalStrategy.seppEligible &&
                annualSeppPayment > 0.0 &&
                primaryAgeMonths < seppEndAgeMonths &&
                balances.pretax > 0.0
            val seppDistribution = if (seppActive) {
                val monthlySeppPayment = annualSeppPayment / MONTHS_PER_YEAR.toDouble()
                val distribution = minOf(balances.pretax, monthlySeppPayment)
                balances.pretax -= distribution
                distribution
            } else {
                0.0
            }
            val annualizedSeppDistribution = seppDistribution * MONTHS_PER_YEAR.toDouble()
            val annualizedGuaranteedIncome = guaranteedIncome * MONTHS_PER_YEAR.toDouble()

            val useCashReserveForDrawdown = scenario.withdrawalStrategy.useCashReserveDuringDrawdowns &&
                portfolioReturn < scenario.withdrawalStrategy.drawdownTrigger &&
                balances.cash > 0.0
            val withdrawalPlan = planPortfolioWithdrawalForNetNeed(
                netNeed = annualizedSpending,
                annualSocialSecurity = socialSecurity * MONTHS_PER_YEAR.toDouble(),
                filingStatus = currentFilingStatus,
                annualOtherTaxableIncome = annualizedGuaranteedIncome + annualizedSeppDistribution,
                balances = balances,
                useCashFirst = useCashReserveForDrawdown,
                additionalWithdrawalTaxRate = earlyWithdrawalPenaltyRate(
                    retirementAge = scenario.household.retirementAge,
                    primaryAgeMonths = primaryAgeMonths,
                    withdrawalStrategy = scenario.withdrawalStrategy
                )
            )
            val grossWithdrawal = withdrawalPlan.monthlyGrossWithdrawal

            val portfolioWithdrawal = if (useCashReserveForDrawdown) {
                val cashDraw = minOf(balances.cash, grossWithdrawal)
                balances.cash -= cashDraw
                grossWithdrawal - cashDraw
            } else {
                grossWithdrawal
            }.coerceAtLeast(0.0)
            withdrawStandard(portfolioWithdrawal, balances)
            val monthlyUnspentCash = ((withdrawalPlan.annualNetCash - annualizedSpending) /
                MONTHS_PER_YEAR.toDouble()).coerceAtLeast(0.0)
            if (monthlyUnspentCash > BALANCE_EPSILON) {
                balances.cash += monthlyUnspentCash
            }

            currentAnnualMedicareIncome += withdrawalPlan.monthlyTaxablePortfolioWithdrawal +
                seppDistribution +
                guaranteedIncome +
                withdrawalPlan.annualTaxableSocialSecurity / MONTHS_PER_YEAR.toDouble()

            if (scenario.rothConversion.enabled && !seppActive && monthInYear == 0) {
                val conversionLimit = TaxCalculator.upperBracketLimitForRate(
                    scenario.rothConversion.marginalRateCap,
                    currentFilingStatus
                )
                if (conversionLimit != null && balances.pretax > 0.0) {
                    val conversion = minOf(balances.pretax, conversionLimit * 0.20)
                    val tax = TaxCalculator.taxLiability(conversion, currentFilingStatus)
                    balances.pretax -= conversion
                    balances.roth += conversion
                    withdrawForConversionTax(tax, balances)
                    currentAnnualMedicareIncome += conversion
                }
            }

            if (!homeSold && remainingMortgageMonths > 0) {
                val principalReduction = remainingMortgageBalance / remainingMortgageMonths.toDouble()
                remainingMortgageBalance = (remainingMortgageBalance - principalReduction).coerceAtLeast(0.0)
                remainingMortgageMonths -= 1
            }

            normalizeNearZeroBalance(balances)
            if (balances.total < 0.0 && failureAge == null) {
                if (!homeSold && monthlyHomeValue > 0.0) {
                    val netHomeEquity = (monthlyHomeValue - remainingMortgageBalance).coerceAtLeast(0.0)
                    balances.cash += netHomeEquity
                    monthlyHomeValue = 0.0
                    remainingMortgageBalance = 0.0
                    remainingMortgageMonths = 0
                    homeSold = true
                }
                if (balances.total < 0.0) {
                    failureAge = age
                    pathBalances += balances.total
                    break
                }
            }
            val totalBalance = balances.total
            if (monthInYear == MONTHS_PER_YEAR - 1) {
                pathBalances += totalBalance
                chartBalances += totalBalance
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
            val nextSpendingPathMultiplier = spendingPathMultiplierAtRetirementMonth(
                scenario,
                monthOffset = monthOffset + 1
            )
            val spendingPathChange = nextSpendingPathMultiplier / spendingPathMultiplier.coerceAtLeast(0.0001)

            monthlySpending *= (1.0 + inflation) * spendingPathChange
            monthlyRent *= 1.0 + inflation
            if (!homeSold) {
                monthlyHomeValue *= 1.0 + inflation
            }
            monthlySeniorApartmentRent *= 1.0 + inflation
            monthlyGuaranteedIncome *= 1.0 + monthlyGuaranteedIncomeIncrease
            monthlyHomeOwnershipCostInBase *= (1.0 + inflation) * spendingPathChange
            monthlyHealthcare *= 1.0 + healthcareInflation
            medicareInflationMultiplier *= 1.0 + healthcareInflation
            socialSecurityInflationMultiplier *= 1.0 + inflation
            spendingPathMultiplier = nextSpendingPathMultiplier

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
            survivedThroughAge = minOf(
                householdDeathAge,
                scenario.household.retirementAge + horizonYears(scenario)
            ),
            yearEndBalances = pathBalances,
            chartBalances = chartBalances
        )
    }

    private fun normalizeNearZeroBalance(balances: SimBalances) {
        val total = balances.total
        if (total < 0.0 && total > -BALANCE_EPSILON) {
            balances.cash -= total
        }
    }

    private fun planPortfolioWithdrawalForNetNeed(
        netNeed: Double,
        annualSocialSecurity: Double,
        filingStatus: FilingStatus,
        annualOtherTaxableIncome: Double,
        balances: SimBalances,
        useCashFirst: Boolean,
        additionalWithdrawalTaxRate: Double
    ): PortfolioWithdrawalPlan {
        val additionalRate = additionalWithdrawalTaxRate.coerceAtLeast(0.0)

        fun estimate(annualGrossWithdrawal: Double): PortfolioWithdrawalPlan {
            val monthlyGrossWithdrawal = annualGrossWithdrawal / MONTHS_PER_YEAR.toDouble()
            val monthlyTaxablePortfolioWithdrawal = pretaxDrawForMonthlyWithdrawal(
                monthlyGrossWithdrawal,
                balances,
                useCashFirst
            )
            val annualTaxablePortfolioWithdrawal =
                monthlyTaxablePortfolioWithdrawal * MONTHS_PER_YEAR.toDouble()
            val annualTaxableSocialSecurity = TaxCalculator.taxableSocialSecurity(
                otherIncome = annualOtherTaxableIncome + annualTaxablePortfolioWithdrawal,
                annualSocialSecurity = annualSocialSecurity,
                filingStatus = filingStatus
            )
            val tax = TaxCalculator.taxLiability(
                taxableIncome = annualOtherTaxableIncome +
                    annualTaxablePortfolioWithdrawal +
                    annualTaxableSocialSecurity,
                filingStatus = filingStatus
            )
            val additionalWithdrawalTax = annualTaxablePortfolioWithdrawal * additionalRate
            val annualNetCash = annualSocialSecurity +
                annualOtherTaxableIncome +
                annualGrossWithdrawal -
                tax -
                additionalWithdrawalTax

            return PortfolioWithdrawalPlan(
                monthlyGrossWithdrawal = monthlyGrossWithdrawal,
                monthlyTaxablePortfolioWithdrawal = monthlyTaxablePortfolioWithdrawal,
                annualTaxableSocialSecurity = annualTaxableSocialSecurity,
                annualNetCash = annualNetCash
            )
        }

        if (estimate(0.0).annualNetCash >= netNeed) {
            return estimate(0.0)
        }

        val neededAfterIncome = (netNeed - annualSocialSecurity - annualOtherTaxableIncome)
            .coerceAtLeast(0.0)
        var low = 0.0
        var high = neededAfterIncome * 1.8 + 10_000.0
        while (estimate(high).annualNetCash < netNeed) {
            high *= 2.0
        }

        repeat(40) {
            val mid = (low + high) / 2.0
            if (estimate(mid).annualNetCash >= netNeed) {
                high = mid
            } else {
                low = mid
            }
        }

        return estimate(high)
    }

    private fun pretaxDrawForMonthlyWithdrawal(
        amount: Double,
        balances: SimBalances,
        useCashFirst: Boolean
    ): Double {
        val amountAfterCash = if (useCashFirst) {
            (amount - balances.cash).coerceAtLeast(0.0)
        } else {
            amount
        }
        return minOf(balances.pretax, amountAfterCash.coerceAtLeast(0.0))
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
        rentCost: Double,
        healthcareCost: Double,
        longTermCareCost: Double,
        replaceBaseSpendingWithLongTermCare: Boolean
    ): Double {
        val baseSpending = if (replaceBaseSpendingWithLongTermCare) 0.0 else monthlySpending
        return baseSpending + mortgageCost + rentCost + healthcareCost + longTermCareCost
    }

    internal fun earlyWithdrawalPenaltyRate(
        retirementAge: Int,
        primaryAgeMonths: Int,
        withdrawalStrategy: WithdrawalStrategy
    ): Double {
        if (!withdrawalStrategy.applyEarlyWithdrawalPenalty) return 0.0
        if (primaryAgeMonths >= PENALTY_FREE_WITHDRAWAL_AGE_MONTHS) return 0.0
        if (
            withdrawalStrategy.ruleOf55Eligible &&
            retirementAge >= RULE_OF_55_MINIMUM_RETIREMENT_AGE
        ) {
            return 0.0
        }
        return EARLY_WITHDRAWAL_PENALTY_RATE
    }

    internal fun seppFixedAmortizationAnnualPayment(
        accountBalance: Double,
        age: Int,
        interestRate: Double = SEPP_DEFAULT_INTEREST_RATE
    ): Double {
        if (accountBalance <= 0.0) return 0.0
        val years = singleLifeExpectancy(age) ?: return 0.0
        if (years <= 0.0) return 0.0
        val rate = interestRate.coerceAtLeast(0.0)
        val amortizationFactor = if (rate == 0.0) {
            years
        } else {
            (1.0 - (1.0 + rate).pow(-years)) / rate
        }
        return accountBalance / amortizationFactor
    }

    internal fun singleLifeExpectancy(age: Int): Double? {
        return SINGLE_LIFE_EXPECTANCY_TABLE.getOrNull(age.coerceAtLeast(0))
    }

    private fun monthlySocialSecurityBenefit(
        scenario: RetirementScenario,
        married: Boolean,
        primaryAlive: Boolean,
        spouseAlive: Boolean,
        primaryAgeMonths: Int,
        spouseAgeMonths: Int,
        primaryBenefitFactor: Double,
        primarySurvivorBaseFactor: Double,
        spouseBirthYear: Int,
        inflationMultiplier: Double
    ): Double {
        val primaryPiaMonthly = scenario.socialSecurity.annualBenefitAt67 /
            MONTHS_PER_YEAR.toDouble() *
            inflationMultiplier
        val primaryClaimAgeMonths = scenario.socialSecurity.claimAge * MONTHS_PER_YEAR
        val spouseClaimAgeMonths = scenario.socialSecurity.spouseClaimAge * MONTHS_PER_YEAR
        var monthlyBenefit = 0.0

        if (primaryAlive && primaryAgeMonths >= primaryClaimAgeMonths) {
            monthlyBenefit += primaryPiaMonthly * primaryBenefitFactor
        }

        if (!married || !spouseAlive) return monthlyBenefit

        if (primaryAlive) {
            val spouseSpousalStartAgeMonths = max(62 * MONTHS_PER_YEAR, spouseClaimAgeMonths)
            if (
                primaryAgeMonths >= primaryClaimAgeMonths &&
                spouseAgeMonths >= spouseSpousalStartAgeMonths
            ) {
                monthlyBenefit += primaryPiaMonthly *
                    SocialSecurity.spousalBenefitFactor(spouseBirthYear, spouseClaimAgeMonths)
            }
        } else {
            val spouseSurvivorStartAgeMonths = max(60 * MONTHS_PER_YEAR, spouseClaimAgeMonths)
            if (spouseAgeMonths >= spouseSurvivorStartAgeMonths) {
                monthlyBenefit += primaryPiaMonthly *
                    primarySurvivorBaseFactor *
                    SocialSecurity.survivorBenefitFactor(spouseBirthYear, spouseClaimAgeMonths)
            }
        }

        return monthlyBenefit
    }

    private fun sampleDeathAge(
        gender: Gender,
        startAge: Int,
        targetEndAge: Int,
        random: Random
    ): Int {
        var age = startAge
        while (age <= targetEndAge) {
            val deathProbability = MortalityTables.annualDeathProbability(gender, age)
            if (random.nextDouble() < deathProbability) return age
            age += 1
        }
        return targetEndAge + 1
    }

    private fun Gender.opposite(): Gender {
        return when (this) {
            Gender.Male -> Gender.Female
            Gender.Female -> Gender.Male
        }
    }

    private fun spouseAgeAtRetirement(scenario: RetirementScenario): Int {
        return scenario.household.spouseCurrentAge +
            (scenario.household.retirementAge - scenario.household.currentAge)
    }

    private fun sampleLongTermCareStartAge(
        scenario: RetirementScenario,
        startAge: Int,
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
        return max(startAge, deathAge - scenario.longTermCare.averageDurationYears)
    }

    private fun mortgageMonthsInRetirement(scenario: RetirementScenario): Int {
        val yearsToRetirement = scenario.household.retirementAge - scenario.household.currentAge
        val preRetirementMonths = yearsToRetirement * MONTHS_PER_YEAR
        return max(0, scenario.mortgage.totalMonthsLeft - preRetirementMonths)
    }

    private fun mortgageBalanceAtRetirement(scenario: RetirementScenario): Double {
        val totalMortgageMonths = scenario.mortgage.totalMonthsLeft
        if (totalMortgageMonths <= 0) return scenario.mortgage.currentBalance
        val yearsToRetirement = scenario.household.retirementAge - scenario.household.currentAge
        val preRetirementMonths = yearsToRetirement * MONTHS_PER_YEAR
        val remainingMonths = (totalMortgageMonths - preRetirementMonths).coerceAtLeast(0)
        return scenario.mortgage.currentBalance * remainingMonths.toDouble() / totalMortgageMonths.toDouble()
    }

    private fun horizonYears(scenario: RetirementScenario): Int {
        val primaryHorizon = scenario.household.targetEndAge - scenario.household.retirementAge
        if (scenario.household.filingStatus != FilingStatus.Married) return primaryHorizon

        val spouseHorizon = scenario.household.targetEndAge - spouseAgeAtRetirement(scenario)
        return max(primaryHorizon, spouseHorizon).coerceAtLeast(primaryHorizon)
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

    private fun spendingPathMultiplierAtRetirementMonth(
        scenario: RetirementScenario,
        monthOffset: Int
    ): Double {
        if (scenario.spending.spendingPathModel == SpendingPathModel.Flat) return 1.0

        val currentAgeMonths = scenario.household.retirementAge * MONTHS_PER_YEAR + monthOffset
        val baselineAgeMonths = max(
            EMPIRICAL_SPENDING_DECLINE_START_AGE,
            scenario.household.currentAge
        ) * MONTHS_PER_YEAR
        val cappedAgeMonths = minOf(
            currentAgeMonths,
            EMPIRICAL_SPENDING_DECLINE_END_AGE * MONTHS_PER_YEAR
        )
        val declineMonths = (cappedAgeMonths - baselineAgeMonths).coerceAtLeast(0)
        val yearsOfDecline = declineMonths.toDouble() / MONTHS_PER_YEAR.toDouble()
        val annualDecline = if (scenario.household.filingStatus == FilingStatus.Married) {
            EMPIRICAL_MARRIED_REAL_SPENDING_DECLINE
        } else {
            EMPIRICAL_SINGLE_REAL_SPENDING_DECLINE
        }
        val floor = if (scenario.household.filingStatus == FilingStatus.Married) {
            EMPIRICAL_MARRIED_SPENDING_FLOOR
        } else {
            EMPIRICAL_SINGLE_SPENDING_FLOOR
        }

        return (1.0 - annualDecline).pow(yearsOfDecline).coerceAtLeast(floor)
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
            scenario.household.currentAge.toString(),
            scenario.household.retirementAge.toString(),
            scenario.household.targetEndAge.toString(),
            scenario.household.filingStatus.name,
            scenario.household.gender.name,
            scenario.household.spouseCurrentAge.toString(),
            scenario.accounts.pretax.fingerprintNumber(),
            scenario.accounts.roth.fingerprintNumber(),
            scenario.accounts.taxable.fingerprintNumber(),
            scenario.accounts.cash.fingerprintNumber(),
            scenario.spending.annualBaseSpending.fingerprintNumber(),
            scenario.spending.generalInflationMean.fingerprintNumber(),
            scenario.spending.generalInflationStdDev.fingerprintNumber(),
            scenario.spending.spendingPathModel.name,
            scenario.spending.lowPortfolioSpendingReduction.fingerprintNumber(),
            scenario.mortgage.monthlyPayment.fingerprintNumber(),
            scenario.mortgage.yearsLeft.toString(),
            scenario.mortgage.monthsLeft.toString(),
            scenario.mortgage.currentBalance.fingerprintNumber(),
            scenario.rent.monthlyRent.fingerprintNumber(),
            scenario.home.currentValue.fingerprintNumber(),
            scenario.healthcare.preMedicareMonthlyPremium.fingerprintNumber(),
            scenario.healthcare.healthcareInflationMean.fingerprintNumber(),
            scenario.healthcare.healthcareInflationStdDev.fingerprintNumber(),
            scenario.healthcare.includeMedicarePremiums.toString(),
            scenario.socialSecurity.annualBenefitAt67.fingerprintNumber(),
            scenario.socialSecurity.claimAge.toString(),
            scenario.socialSecurity.spouseClaimAge.toString(),
            scenario.guaranteedIncome.annualIncome.fingerprintNumber(),
            scenario.guaranteedIncome.startAge.toString(),
            scenario.guaranteedIncome.annualIncrease.fingerprintNumber(),
            scenario.guaranteedIncome.survivorPercent.fingerprintNumber(),
            scenario.market.preRetirementMeanReturn.fingerprintNumber(),
            scenario.market.preRetirementStdDev.fingerprintNumber(),
            scenario.market.stockMeanReturn.fingerprintNumber(),
            scenario.market.stockStdDev.fingerprintNumber(),
            scenario.market.bondMeanReturn.fingerprintNumber(),
            scenario.market.bondStdDev.fingerprintNumber(),
            scenario.postRetirementAllocation.stockUnder30x.fingerprintNumber(),
            scenario.postRetirementAllocation.stock30xTo35x.fingerprintNumber(),
            scenario.postRetirementAllocation.stock35xTo40x.fingerprintNumber(),
            scenario.postRetirementAllocation.stock40xTo45x.fingerprintNumber(),
            scenario.postRetirementAllocation.stock45xTo50x.fingerprintNumber(),
            scenario.postRetirementAllocation.stock50xOrMore.fingerprintNumber(),
            scenario.rothConversion.enabled.toString(),
            scenario.rothConversion.marginalRateCap.fingerprintNumber(),
            scenario.withdrawalStrategy.useCashReserveDuringDrawdowns.toString(),
            scenario.withdrawalStrategy.drawdownTrigger.fingerprintNumber(),
            scenario.withdrawalStrategy.applyEarlyWithdrawalPenalty.toString(),
            scenario.withdrawalStrategy.ruleOf55Eligible.toString(),
            scenario.withdrawalStrategy.seppEligible.toString(),
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
        val survivedThroughAge: Int,
        val yearEndBalances: List<Double>,
        val chartBalances: List<Double>
    )

    private data class PortfolioWithdrawalPlan(
        val monthlyGrossWithdrawal: Double,
        val monthlyTaxablePortfolioWithdrawal: Double,
        val annualTaxableSocialSecurity: Double,
        val annualNetCash: Double
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

    private val SINGLE_LIFE_EXPECTANCY_TABLE = doubleArrayOf(
        84.6, 83.7, 82.8, 81.8, 80.8, 79.8, 78.8, 77.9, 76.9, 75.9,
        74.9, 73.9, 72.9, 71.9, 70.9, 69.9, 69.0, 68.0, 67.0, 66.0,
        65.0, 64.1, 63.1, 62.1, 61.1, 60.2, 59.2, 58.2, 57.3, 56.3,
        55.3, 54.4, 53.4, 52.5, 51.5, 50.5, 49.6, 48.6, 47.7, 46.7,
        45.7, 44.8, 43.8, 42.9, 41.9, 41.0, 40.0, 39.0, 38.1, 37.1,
        36.2, 35.3, 34.3, 33.4, 32.5, 31.6, 30.6, 29.8, 28.9, 28.0,
        27.1, 26.2, 25.4, 24.5, 23.7, 22.9, 22.0, 21.2, 20.4, 19.6,
        18.8, 18.0, 17.2, 16.4, 15.6, 14.8, 14.1, 13.3, 12.6, 11.9,
        11.2, 10.5, 9.9, 9.3, 8.7, 8.1, 7.6, 7.1, 6.6, 6.1,
        5.7, 5.3, 4.9, 4.6, 4.3, 4.0, 3.7, 3.4, 3.2, 3.0,
        2.8, 2.6, 2.5, 2.3, 2.2, 2.1, 2.1, 2.1, 2.0, 2.0,
        2.0, 2.0, 2.0, 1.9, 1.9, 1.8, 1.8, 1.6, 1.4, 1.1,
        1.0
    )
}
