package com.retirementreadinesslab.model

enum class FilingStatus(val label: String) {
    Single("Single"),
    Married("Married"),
    HeadOfHousehold("Head of household")
}

enum class Gender(val label: String) {
    Male("Male"),
    Female("Female")
}

enum class RiskLevel {
    Healthy,
    Watch,
    AtRisk
}

enum class ScenarioWarningSeverity {
    Note,
    Watch
}

enum class SpendingPathModel(val label: String) {
    Flat("Flat real spending"),
    EmpiricalAgeDecline("Empirical age decline")
}

const val DEFAULT_PROJECTION_END_AGE = 119
const val SENIOR_APARTMENT_MONTHLY_RENT_2026 = 3_000.0

data class ScenarioWarning(
    val title: String,
    val detail: String,
    val severity: ScenarioWarningSeverity = ScenarioWarningSeverity.Watch
)

data class HouseholdProfile(
    val currentAge: Int,
    val retirementAge: Int,
    val targetEndAge: Int = DEFAULT_PROJECTION_END_AGE,
    val filingStatus: FilingStatus = FilingStatus.Single,
    val gender: Gender = Gender.Male,
    val spouseGender: Gender = if (gender == Gender.Male) Gender.Female else Gender.Male,
    val spouseCurrentAge: Int = currentAge
)

data class AccountBalances(
    val pretax: Double,
    val roth: Double,
    val taxable: Double = 0.0,
    val cash: Double
) {
    val total: Double
        get() = pretax + roth + taxable + cash
}

/*
 * U.S. general household inflation assumption excluding healthcare.
 *
 * Use CPI-style inflation, not investment-return inflation.
 *
 * Recommended default:
 *   mean annual inflation      = 0.023   // 2.3%
 *   annual standard deviation  = 0.016   // 1.6%
 *
 * Reasoning:
 *   - The Federal Reserve's long-run inflation anchor is 2.0% PCE inflation.
 *   - CPI-style inflation used by households and retirement COLAs tends to be
 *     modeled a little above that. The 2025 Social Security Trustees Report
 *     uses 2.4% CPI as its intermediate long-run assumption from 2027 onward.
 *   - The 30-year TIPS breakeven inflation rate was about 2.25% in Apr 2026,
 *     which is a market-based long-run inflation signal.
 *   - Historical CPI-U volatility since the modern low-inflation regime began
 *     is about 1.5% per year, so 1.6% is a reasonable simulation default.
 *   - BLS CPI-U relative-importance data for Dec 2025 shows medical care
 *     at 8.423% of CPI-U and all-items-less-medical-care at 91.577%.
 *   - Because medical care is a relatively small CPI weight, removing it
 *     does not materially change the general inflation assumption.
 *   - Using a rough weighted-average adjustment:
 *
 *       non_healthcare =
 *           (overall_cpi - medical_weight * medical_inflation)
 *           / (1.0 - medical_weight)
 *
 *     If overall CPI = 2.4%, medical weight = 8.423%, and medical-care
 *     inflation is assumed around 3.5% to 4.0%, then non-healthcare
 *     inflation is about 2.25% to 2.30%.
 *
 *   - Historical BLS/FRED CPI-U "All Items Less Medical Care" data also
 *     supports this being a small adjustment. Using annual-average values
 *     from 1995-2024, all-items-less-medical-care inflation averaged about
 *     2.49% with an annual standard deviation of about 1.63%.
 */
const val DEFAULT_GENERAL_INFLATION_MEAN = 0.023
const val DEFAULT_GENERAL_INFLATION_STD_DEV = 0.016

data class SpendingPlan(
    val annualBaseSpending: Double,
    val generalInflationMean: Double = DEFAULT_GENERAL_INFLATION_MEAN,
    val generalInflationStdDev: Double = DEFAULT_GENERAL_INFLATION_STD_DEV,
    val spendingPathModel: SpendingPathModel = SpendingPathModel.EmpiricalAgeDecline,
    val lowPortfolioSpendingReduction: Double = 0.10
)

data class BudgetLineItem(
    val id: String,
    val name: String,
    val monthlyAmount: Double
)

data class MonthlyBudget(
    val month: String,
    val checkingSavingsBills: List<BudgetLineItem> = emptyList(),
    val creditCardBills: List<BudgetLineItem> = emptyList(),
    val cashAndAtmWithdrawals: Double = 0.0
) {
    val checkingSavingsTotal: Double
        get() = checkingSavingsBills.sumOf { it.monthlyAmount }

    val creditCardTotal: Double
        get() = creditCardBills.sumOf { it.monthlyAmount }

    val monthlyTotal: Double
        get() = checkingSavingsTotal + creditCardTotal + cashAndAtmWithdrawals
}

data class BudgetProfile(
    val annualPropertyTaxes: Double = 0.0,
    val annualHomeInsurance: Double = 0.0,
    val annualAutoInsurance: Double = 0.0,
    val monthlyBudgets: List<MonthlyBudget> = emptyList(),
    val isAppliedToAnnualBaseSpending: Boolean = false
) {
    val annualFixedSpending: Double
        get() = annualPropertyTaxes + annualHomeInsurance + annualAutoInsurance

    val monthsUsedForEstimate: List<MonthlyBudget>
        get() = monthlyBudgets
            .groupBy { it.month }
            .map { (_, entries) -> entries.last() }
            .sortedBy { it.month }
            .takeLast(MAX_BUDGET_MONTHS_FOR_ESTIMATE)

    val averageMonthlySpending: Double
        get() {
            val months = monthsUsedForEstimate
            if (months.isEmpty()) return 0.0
            return months.sumOf { it.monthlyTotal } / months.size.toDouble()
        }

    val annualizedMonthlySpending: Double
        get() = averageMonthlySpending * 12.0

    val annualBaseSpendingEstimate: Double
        get() = annualFixedSpending + annualizedMonthlySpending
}

const val MAX_BUDGET_MONTHS_FOR_ESTIMATE = 12

data class MortgagePlan(
    val monthlyPayment: Double = 0.0,
    val yearsLeft: Int = 0,
    val monthsLeft: Int = 0,
    val currentBalance: Double = 0.0
) {
    val totalMonthsLeft: Int
        get() = yearsLeft * 12 + monthsLeft
}

data class RentPlan(
    val monthlyRent: Double = 0.0
)

data class HomePlan(
    val currentValue: Double = 0.0
)

/*
 * U.S. household healthcare expense inflation assumption.
 *
 * Recommended default:
 *   healthcare_inflation_mean = 0.040   // 4.0% per year
 *   healthcare_inflation_sd   = 0.018   // 1.8% annual standard deviation
 *
 * This is intended for household healthcare expenses in retirement:
 * Medicare premiums, supplemental premiums, prescription drugs,
 * deductibles, copays, and other out-of-pocket medical costs.
 *
 * It is not a pure CPI medical-care price index. A pure medical-care CPI
 * assumption would be lower, around 3.2% to 3.4%. Household healthcare
 * expenses tend to grow faster because they include premium changes,
 * utilization, plan design, age-related usage, and cost shifting.
 *
 * Source logic:
 *   1. BLS/FRED CPI-U Medical Care is the historical price anchor.
 *      Series: CPIMEDSL, Consumer Price Index for All Urban Consumers:
 *      Medical Care in U.S. City Average.
 *
 *      Using Dec-to-Dec annual changes for 1995-2024:
 *        average annual medical CPI inflation ~= 3.3%
 *        annual standard deviation ~= 1.1%
 *
 *   2. CMS National Health Expenditure projections are the forward-looking
 *      healthcare-cost anchor. CMS projects national health expenditures
 *      to grow 5.8% per year over 2024-2033, faster than projected GDP
 *      growth of 4.3%. That total spending measure includes utilization,
 *      population, coverage, and intensity, so it is too high to use as
 *      pure household inflation.
 *
 *   3. CMS projects out-of-pocket spending growth to settle near 3.9%
 *      for 2027-2033. CMS also projects private health insurance spending
 *      growth around 4.3% for 2028-2033, hospital spending around 5.5%,
 *      physician/clinical services around 5.1%, and prescription drugs
 *      around 4.7%.
 *
 *   4. Therefore, a 4.0% household healthcare inflation mean is a compromise:
 *        - above pure medical-care CPI, because retirement healthcare
 *          expenses include premiums and utilization;
 *        - below total national health spending growth, because total NHE
 *          includes population/enrollment/intensity effects that should not
 *          all be applied to one household's expense line.
 *
 *   5. The standard deviation is set to 1.8%, higher than the 1.1% historical
 *      medical CPI volatility, because household healthcare expenses are
 *      more variable than the aggregate CPI medical-care price index.
 */
const val DEFAULT_HEALTHCARE_INFLATION_MEAN = 0.040
const val DEFAULT_HEALTHCARE_INFLATION_STD_DEV = 0.018

/*
 * Pre-Medicare health insurance premium assumption.
 *
 * Recommended default:
 *   pre_medicare_monthly_premium_per_adult = 1250.00
 *
 * Meaning:
 *   Gross monthly health insurance premium for one pre-Medicare adult,
 *   roughly age 60, buying individual ACA Marketplace coverage.
 *
 * This is intended to represent the premium only. It does not include:
 *   - deductibles
 *   - copays
 *   - coinsurance
 *   - prescription out-of-pocket costs
 *   - dental or vision premiums
 *   - HSA contributions
 *
 * Reasoning:
 *   - Pre-Medicare retirees generally need non-Medicare coverage until
 *     age 65, unless covered by an employer, spouse, retiree plan, COBRA,
 *     Medicaid, or another source.
 *
 *   - ACA Marketplace Silver coverage is a reasonable benchmark because
 *     Silver plans are the benchmark used for ACA premium tax credit
 *     calculations. KFF's 2026 Marketplace calculator states that the
 *     Silver premium is the second-lowest-cost Silver plan in the entered
 *     county and that premiums are based on actual 2026 exchange premiums
 *     from CMS, state exchanges, insurance departments, and KFF research.
 *
 *   - KFF gives a 2026 example of a 60-year-old earning $64,000, above the
 *     400% FPL subsidy cliff, paying an estimated $14,931/year for the
 *     annual premium with no tax credit.
 *
 *       $14,931 / 12 = $1,244.25/month
 *
 *     Rounded for calculator use:
 *
 *       pre_medicare_monthly_premium_per_adult = $1,250
 *
 *   - Use this as a gross premium before ACA premium tax credits. If the
 *     calculator models ACA subsidies, apply the subsidy after this premium
 *     estimate rather than lowering the default premium itself.
 *
 *   - For a two-adult household where both adults are under 65:
 *
 *       pre_medicare_monthly_premium_household =
 *           2 * pre_medicare_monthly_premium_per_adult
 *         = $2,500/month
 *
 *   - Premiums vary heavily by age, state, county, tobacco status, income,
 *     and subsidy eligibility. This default is intentionally a national
 *     planning approximation.
 */
const val DEFAULT_PRE_MEDICARE_MONTHLY_PREMIUM = 1250.0

data class HealthcarePlan(
    val preMedicareMonthlyPremium: Double = DEFAULT_PRE_MEDICARE_MONTHLY_PREMIUM,
    val healthcareInflationMean: Double = DEFAULT_HEALTHCARE_INFLATION_MEAN,
    val healthcareInflationStdDev: Double = DEFAULT_HEALTHCARE_INFLATION_STD_DEV,
    val includeMedicarePremiums: Boolean = true
)

data class SocialSecurityPlan(
    val annualBenefitAt67: Double,
    val claimAge: Int = 67,
    val spouseClaimAge: Int = 67
)

data class GuaranteedIncomePlan(
    val annualIncome: Double = 0.0,
    val startAge: Int = 65,
    val annualIncrease: Double = 0.0,
    val survivorPercent: Double = 1.0
)

data class MarketAssumptions(
    val preRetirementMeanReturn: Double = 0.133,
    val preRetirementStdDev: Double = 0.162,
    val stockMeanReturn: Double = 0.133,
    val stockStdDev: Double = 0.162,
    val bondMeanReturn: Double = 0.03,
    val bondStdDev: Double = 0.06
)

enum class PostRetirementAllocationTier(val label: String) {
    Under30x("Under 30x annual spending"),
    From30xTo35x("30x to <35x annual spending"),
    From35xTo40x("35x to <40x annual spending"),
    From40xTo45x("40x to <45x annual spending"),
    From45xTo50x("45x to <50x annual spending"),
    AtLeast50x("50x or more annual spending")
}

data class PostRetirementAllocationStrategy(
    val stockUnder30x: Double = 1.00,
    val stock30xTo35x: Double = 0.90,
    val stock35xTo40x: Double = 0.80,
    val stock40xTo45x: Double = 0.70,
    val stock45xTo50x: Double = 0.60,
    val stock50xOrMore: Double = 0.50
) {
    fun stockAllocationFor(tier: PostRetirementAllocationTier): Double {
        return when (tier) {
            PostRetirementAllocationTier.Under30x -> stockUnder30x
            PostRetirementAllocationTier.From30xTo35x -> stock30xTo35x
            PostRetirementAllocationTier.From35xTo40x -> stock35xTo40x
            PostRetirementAllocationTier.From40xTo45x -> stock40xTo45x
            PostRetirementAllocationTier.From45xTo50x -> stock45xTo50x
            PostRetirementAllocationTier.AtLeast50x -> stock50xOrMore
        }
    }

    fun withStockAllocation(
        tier: PostRetirementAllocationTier,
        stockAllocation: Double
    ): PostRetirementAllocationStrategy {
        val value = stockAllocation.coerceIn(0.0, 1.0)
        return when (tier) {
            PostRetirementAllocationTier.Under30x -> copy(stockUnder30x = value)
            PostRetirementAllocationTier.From30xTo35x -> copy(stock30xTo35x = value)
            PostRetirementAllocationTier.From35xTo40x -> copy(stock35xTo40x = value)
            PostRetirementAllocationTier.From40xTo45x -> copy(stock40xTo45x = value)
            PostRetirementAllocationTier.From45xTo50x -> copy(stock45xTo50x = value)
            PostRetirementAllocationTier.AtLeast50x -> copy(stock50xOrMore = value)
        }
    }

    fun stockAllocation(investedBalance: Double, annualSpending: Double): Double {
        if (annualSpending <= 0.0) return stock50xOrMore
        val ratio = investedBalance / annualSpending
        return when {
            ratio < 30.0 -> stockUnder30x
            ratio < 35.0 -> stock30xTo35x
            ratio < 40.0 -> stock35xTo40x
            ratio < 45.0 -> stock40xTo45x
            ratio < 50.0 -> stock45xTo50x
            else -> stock50xOrMore
        }
    }

    fun stockAllocations(): List<Double> {
        return PostRetirementAllocationTier.entries.map { stockAllocationFor(it) }
    }
}

data class RothConversionStrategy(
    val enabled: Boolean = false,
    val marginalRateCap: Double = 0.22
)

const val EARLY_WITHDRAWAL_PENALTY_RATE = 0.10
const val RULE_OF_55_MINIMUM_RETIREMENT_AGE = 55
const val PENALTY_FREE_WITHDRAWAL_AGE_MONTHS = 59 * 12 + 6
const val SEPP_MINIMUM_PAYMENT_MONTHS = 5 * 12
const val SEPP_DEFAULT_INTEREST_RATE = 0.05

data class WithdrawalStrategy(
    val useCashReserveDuringDrawdowns: Boolean = false,
    val drawdownTrigger: Double = -0.01,
    val applyEarlyWithdrawalPenalty: Boolean = false,
    val ruleOf55Eligible: Boolean = false,
    val seppEligible: Boolean = false
) {
    companion object {
        fun defaultsForRetirementAge(retirementAge: Int): WithdrawalStrategy {
            return WithdrawalStrategy(
                applyEarlyWithdrawalPenalty = retirementAge * 12 < PENALTY_FREE_WITHDRAWAL_AGE_MONTHS,
                // Age alone does not establish Rule-of-55 eligibility. The distribution must
                // come from a qualifying employer plan after a qualifying separation from service.
                ruleOf55Eligible = false,
                seppEligible = false
            )
        }
    }
}

data class LongTermCareAssumption(
    val enabled: Boolean = false,
    val annualCost: Double = 100_000.0,
    val averageDurationYears: Int = 3
)

data class RetirementScenario(
    val id: String,
    val name: String,
    val household: HouseholdProfile,
    val accounts: AccountBalances,
    val spending: SpendingPlan,
    val budget: BudgetProfile = BudgetProfile(),
    val mortgage: MortgagePlan = MortgagePlan(),
    val rent: RentPlan = RentPlan(),
    val home: HomePlan = HomePlan(),
    val healthcare: HealthcarePlan = HealthcarePlan(),
    val socialSecurity: SocialSecurityPlan,
    val guaranteedIncome: GuaranteedIncomePlan = GuaranteedIncomePlan(),
    val market: MarketAssumptions = MarketAssumptions(),
    val postRetirementAllocation: PostRetirementAllocationStrategy = PostRetirementAllocationStrategy(),
    val rothConversion: RothConversionStrategy = RothConversionStrategy(),
    val withdrawalStrategy: WithdrawalStrategy = WithdrawalStrategy(),
    val longTermCare: LongTermCareAssumption = LongTermCareAssumption(),
    val numberOfSimulations: Int = 1_500,
    val seed: Long = 20260429L
)

data class OutcomeBand(
    val age: Int,
    val pessimistic: Double,
    val median: Double,
    val optimistic: Double
)

data class SimulationPathPoint(
    val yearsInRetirement: Int,
    val balance: Double,
    val successfulPath: Boolean,
    val separatedFromOppositeOutcome: Boolean
)

data class SimulationMeanPoint(
    val yearsInRetirement: Int,
    val balance: Double
)

data class PortfolioSurvivalPoint(
    val age: Int,
    val notFailedShare: Double,
    val aliveShare: Double
)

data class FundingThreshold(
    val balance: Double,
    val targetReadiness: Double,
    val observedReadiness: Double,
    val includedSimulationCount: Int,
    val totalSimulationCount: Int
)

data class FailureAgeBucket(
    val label: String,
    val count: Int,
    val shareOfFailures: Double
)

data class RiskBreakdown(
    val market: RiskLevel,
    val longevity: RiskLevel,
    val healthcare: RiskLevel,
    val taxes: RiskLevel,
    val spending: RiskLevel,
    val primaryRisk: String,
    val recommendedNextTest: String
)

data class CalculationProvenance(
    val engineVersion: String,
    val engineCadence: String,
    val taxTableVersion: String,
    val mortalityModelVersion: String,
    val randomSeed: Long,
    val simulationCount: Int,
    val assumptionFingerprint: String
)

data class SimulationResult(
    val scenarioId: String,
    val successProbability: Double,
    val medianEndingBalance: Double,
    val pessimisticEndingBalance: Double,
    val optimisticEndingBalance: Double,
    val medianFailureAge: Int?,
    val failureAgeBuckets: List<FailureAgeBucket>,
    val balanceBands: List<OutcomeBand>,
    val notFailedByAge: List<PortfolioSurvivalPoint>,
    val pathPoints: List<SimulationPathPoint>,
    val meanPath: List<SimulationMeanPoint>,
    val fundingThreshold: FundingThreshold?,
    val riskBreakdown: RiskBreakdown,
    val provenance: CalculationProvenance,
    val generatedAtEpochMillis: Long
)

fun RetirementScenario.validate(): List<String> {
    val errors = mutableListOf<String>()
    val numericAssumptions = buildList {
        addAll(
            listOf(
                accounts.pretax,
                accounts.roth,
                accounts.taxable,
                accounts.cash,
                spending.annualBaseSpending,
                spending.generalInflationMean,
                spending.generalInflationStdDev,
                spending.lowPortfolioSpendingReduction,
                budget.annualPropertyTaxes,
                budget.annualHomeInsurance,
                budget.annualAutoInsurance,
                mortgage.monthlyPayment,
                mortgage.currentBalance,
                rent.monthlyRent,
                home.currentValue,
                healthcare.preMedicareMonthlyPremium,
                healthcare.healthcareInflationMean,
                healthcare.healthcareInflationStdDev,
                socialSecurity.annualBenefitAt67,
                guaranteedIncome.annualIncome,
                guaranteedIncome.annualIncrease,
                guaranteedIncome.survivorPercent,
                market.preRetirementMeanReturn,
                market.preRetirementStdDev,
                market.stockMeanReturn,
                market.stockStdDev,
                market.bondMeanReturn,
                market.bondStdDev,
                rothConversion.marginalRateCap,
                withdrawalStrategy.drawdownTrigger,
                longTermCare.annualCost
            )
        )
        addAll(postRetirementAllocation.stockAllocations())
        budget.monthlyBudgets.forEach { month ->
            add(month.cashAndAtmWithdrawals)
            addAll(month.checkingSavingsBills.map { it.monthlyAmount })
            addAll(month.creditCardBills.map { it.monthlyAmount })
        }
    }
    if (numericAssumptions.any { !it.isFinite() }) {
        errors += "Financial and percentage assumptions must be finite numbers."
    }
    if (household.currentAge <= 0) errors += "Current age must be positive."
    if (household.retirementAge < household.currentAge) {
        errors += "Retirement age must be greater than or equal to current age."
    }
    if (household.filingStatus == FilingStatus.Married) {
        if (household.spouseCurrentAge <= 0) errors += "Spouse age must be positive."
        val spouseAgeAtRetirement = household.spouseCurrentAge + (household.retirementAge - household.currentAge)
        if (spouseAgeAtRetirement >= household.targetEndAge) {
            errors += "Spouse age at retirement must be below the mortality projection cap."
        }
    }
    if (household.targetEndAge <= household.retirementAge) {
        errors += "Projection end age must be greater than retirement age."
    }
    if (socialSecurity.claimAge !in 62..70) {
        errors += "Social Security claim age must be between 62 and 70."
    }
    if (socialSecurity.spouseClaimAge !in 60..70) {
        errors += "Spouse Social Security benefit claim age must be between 60 and 70."
    }
    if (accounts.pretax < 0 || accounts.roth < 0 || accounts.taxable < 0 || accounts.cash < 0) {
        errors += "Account balances cannot be negative."
    }
    if (spending.annualBaseSpending < 0) errors += "Annual spending cannot be negative."
    if (spending.generalInflationStdDev < 0) errors += "General inflation std dev cannot be negative."
    if (spending.generalInflationMean !in -0.02..0.15 || spending.generalInflationStdDev !in 0.0..0.30) {
        errors += "General inflation assumptions are outside the supported range."
    }
    if (spending.lowPortfolioSpendingReduction !in 0.0..1.0) {
        errors += "Spending reduction must be between 0% and 100%."
    }
    if (
        budget.annualPropertyTaxes < 0.0 ||
        budget.annualHomeInsurance < 0.0 ||
        budget.annualAutoInsurance < 0.0 ||
        budget.monthlyBudgets.any { month ->
            month.cashAndAtmWithdrawals < 0.0 ||
                month.checkingSavingsBills.any { it.monthlyAmount < 0.0 } ||
                month.creditCardBills.any { it.monthlyAmount < 0.0 }
        }
    ) {
        errors += "Budget amounts cannot be negative."
    }
    if (
        mortgage.monthlyPayment < 0 ||
        mortgage.yearsLeft !in 0..80 ||
        mortgage.monthsLeft < 0 ||
        mortgage.currentBalance < 0
    ) {
        errors += "Mortgage payment and balance cannot be negative, and years left must be between 0 and 80."
    }
    if (mortgage.monthsLeft !in 0..11) {
        errors += "Mortgage months left must be between 0 and 11."
    }
    if (rent.monthlyRent < 0) errors += "Rent cannot be negative."
    if (home.currentValue < 0) errors += "Home value cannot be negative."
    if (healthcare.preMedicareMonthlyPremium < 0) errors += "Healthcare premium cannot be negative."
    if (healthcare.healthcareInflationStdDev < 0) errors += "Healthcare inflation std dev cannot be negative."
    if (healthcare.healthcareInflationMean !in 0.0..0.20 || healthcare.healthcareInflationStdDev !in 0.0..0.30) {
        errors += "Healthcare inflation assumptions are outside the supported range."
    }
    if (socialSecurity.annualBenefitAt67 < 0) errors += "Social Security estimate cannot be negative."
    if (guaranteedIncome.annualIncome < 0) errors += "Guaranteed income cannot be negative."
    if (guaranteedIncome.startAge < 0) errors += "Guaranteed income start age cannot be negative."
    if (guaranteedIncome.annualIncrease !in -0.02..0.15) {
        errors += "Guaranteed income annual increase is outside the supported range."
    }
    if (guaranteedIncome.survivorPercent !in 0.0..1.0) {
        errors += "Guaranteed income survivor benefit must be between 0% and 100%."
    }
    if (
        market.preRetirementStdDev < 0 ||
        market.stockStdDev < 0 ||
        market.bondStdDev < 0
    ) {
        errors += "Market return std dev cannot be negative."
    }
    if (
        market.preRetirementMeanReturn !in -0.20..0.25 ||
        market.preRetirementStdDev !in 0.0..0.60 ||
        market.stockMeanReturn !in -0.20..0.25 ||
        market.stockStdDev !in 0.0..0.60 ||
        market.bondMeanReturn !in -0.20..0.20 ||
        market.bondStdDev !in 0.0..0.40
    ) {
        errors += "Market return assumptions are outside the supported range."
    }
    if (postRetirementAllocation.stockAllocations().any { it !in 0.0..1.0 }) {
        errors += "Post-retirement investment ratios must be between 0% and 100% stocks."
    }
    if (rothConversion.enabled && !supportedRothBracketCaps.any { kotlin.math.abs(it - rothConversion.marginalRateCap) < 0.0001 }) {
        errors += "Roth conversion bracket cap must match a supported federal tax bracket."
    }
    if (longTermCare.annualCost < 0) errors += "Long-term care cost cannot be negative."
    if (longTermCare.averageDurationYears !in 1..10) {
        errors += "Long-term care duration must be between 1 and 10 years."
    }
    if (withdrawalStrategy.drawdownTrigger !in -0.50..0.25) {
        errors += "Cash reserve drawdown trigger is outside the supported range."
    }
    if (numberOfSimulations < 1) errors += "Simulation count must be positive."
    if (numberOfSimulations > PRO_SIMULATION_LIMIT) {
        errors += "Simulation count cannot exceed $PRO_SIMULATION_LIMIT."
    }
    return errors
}

fun RetirementScenario.warnings(): List<ScenarioWarning> {
    val warnings = mutableListOf<ScenarioWarning>()
    val yearsToRetirement = household.retirementAge - household.currentAge
    val retirementHorizon = household.targetEndAge - household.retirementAge
    val spendingRatio = spending.annualBaseSpending / accounts.total.coerceAtLeast(1.0)

    if (household.retirementAge < 50) {
        warnings += ScenarioWarning(
            title = "Very early retirement age",
            detail = "Retiring before 50 creates a long drawdown period and makes healthcare assumptions especially important."
        )
    }
    if (household.retirementAge < 55 && retirementHorizon > 50) {
        warnings += ScenarioWarning(
            title = "Long retirement horizon",
            detail = "The mortality-modeled projection includes a long potential drawdown period, so inflation and healthcare assumptions carry extra weight."
        )
    }
    if (spendingRatio > 0.07) {
        warnings += ScenarioWarning(
            title = "High spending draw",
            detail = "Annual spending is more than 7% of current assets before taxes, healthcare, mortgage, and rent costs."
        )
    }
    if (spending.generalInflationMean < 0.015) {
        warnings += ScenarioWarning(
            title = "Low inflation assumption",
            detail = "General inflation below 1.5% may understate long-term spending pressure.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if (market.stockMeanReturn > 0.145 || market.preRetirementMeanReturn > 0.145) {
        warnings += ScenarioWarning(
            title = "High return assumption",
            detail = "Expected stock or current-portfolio returns above 14.5% may make readiness look stronger than a conservative model."
        )
    }
    val spouseAgeAtRetirement = household.spouseCurrentAge + yearsToRetirement
    val hasPreMedicareAdult = household.retirementAge < 65 ||
        (household.filingStatus == FilingStatus.Married && spouseAgeAtRetirement < 65)
    if (hasPreMedicareAdult && healthcare.preMedicareMonthlyPremium <= 0.0) {
        warnings += ScenarioWarning(
            title = "Missing pre-Medicare healthcare premium",
            detail = "A household member is under 65 at retirement, so a zero healthcare premium can materially understate spending."
        )
    }
    if (!healthcare.includeMedicarePremiums) {
        warnings += ScenarioWarning(
            title = "Medicare premiums excluded",
            detail = "Medicare Parts B/D premiums are not included in this scenario.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if (!longTermCare.enabled) {
        warnings += ScenarioWarning(
            title = "Long-term care stress disabled",
            detail = "The plan does not include a late-life care shock.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if (socialSecurity.annualBenefitAt67 <= 0.0) {
        warnings += ScenarioWarning(
            title = "No Social Security estimate",
            detail = "A zero Social Security estimate may be intentional, but it materially changes bridge and withdrawal needs.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if (mortgage.totalMonthsLeft > 0 && mortgage.totalMonthsLeft > (yearsToRetirement + retirementHorizon) * 12) {
        warnings += ScenarioWarning(
            title = "Mortgage extends beyond horizon",
            detail = "Mortgage time left exceeds the internal mortality projection cap."
        )
    }
    if ((mortgage.monthlyPayment > 0.0 || mortgage.totalMonthsLeft > 0) && home.currentValue <= 0.0) {
        warnings += ScenarioWarning(
            title = "Home value not entered",
            detail = "Enter current home value if the home could be sold later to fund senior apartment rent.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if ((mortgage.monthlyPayment > 0.0 || mortgage.totalMonthsLeft > 0) && mortgage.currentBalance <= 0.0) {
        warnings += ScenarioWarning(
            title = "Mortgage balance not entered",
            detail = "Enter the current mortgage balance so a forced home sale uses net home equity instead of gross home value.",
            severity = ScenarioWarningSeverity.Note
        )
    }
    if (numberOfSimulations < 500) {
        warnings += ScenarioWarning(
            title = "Low simulation count",
            detail = "Use at least 500 simulations before relying on scenario comparisons.",
            severity = ScenarioWarningSeverity.Note
        )
    }

    return warnings
}

private val supportedRothBracketCaps = listOf(0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37)
