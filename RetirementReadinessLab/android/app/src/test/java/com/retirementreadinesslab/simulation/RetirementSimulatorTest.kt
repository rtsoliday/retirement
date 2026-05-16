package com.retirementreadinesslab.simulation

import com.retirementreadinesslab.model.AccountBalances
import com.retirementreadinesslab.model.DEFAULT_PROJECTION_END_AGE
import com.retirementreadinesslab.model.FilingStatus
import com.retirementreadinesslab.model.Gender
import com.retirementreadinesslab.model.GuaranteedIncomePlan
import com.retirementreadinesslab.model.HealthcarePlan
import com.retirementreadinesslab.model.HomePlan
import com.retirementreadinesslab.model.HouseholdProfile
import com.retirementreadinesslab.model.MarketAssumptions
import com.retirementreadinesslab.model.MortgagePlan
import com.retirementreadinesslab.model.RentPlan
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SocialSecurityPlan
import com.retirementreadinesslab.model.SpendingPlan
import com.retirementreadinesslab.model.SpendingPathModel
import com.retirementreadinesslab.model.WithdrawalStrategy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class RetirementSimulatorTest {
    @Test
    fun deterministicSeedProducesStableResult() {
        val scenario = sampleScenario()

        val first = RetirementSimulator.run(scenario)
        val second = RetirementSimulator.run(scenario)

        assertEquals(first.successProbability, second.successProbability, 0.0001)
        assertEquals(first.medianEndingBalance, second.medianEndingBalance, 0.01)
        assertEquals("2026.05-married-income", first.provenance.engineVersion)
        assertEquals("Monthly cashflow model with annual result bands", first.provenance.engineCadence)
        assertEquals("2024 federal brackets", first.provenance.taxTableVersion)
        assertEquals(
            "SSA Trustees Alt2 2025 annual death probabilities",
            first.provenance.mortalityModelVersion
        )
        assertEquals(scenario.seed, first.provenance.randomSeed)
        assertEquals(scenario.numberOfSimulations, first.provenance.simulationCount)
        assertEquals(first.provenance.assumptionFingerprint, second.provenance.assumptionFingerprint)
        assertTrue(first.provenance.assumptionFingerprint.length >= 8)
    }

    @Test
    fun noSpendingScenarioAlwaysSucceeds() {
        val result = RetirementSimulator.run(
            sampleScenario().copy(
                spending = SpendingPlan(annualBaseSpending = 0.0),
                healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false)
            )
        )

        assertEquals(1.0, result.successProbability, 0.0001)
    }

    @Test
    fun noAssetsWithSpendingUsuallyFails() {
        val result = RetirementSimulator.run(
            sampleScenario().copy(
                accounts = AccountBalances(pretax = 0.0, roth = 0.0, taxable = 0.0, cash = 0.0),
                socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67)
            )
        )

        assertTrue(result.successProbability < 0.10)
        assertTrue(result.failureAgeBuckets.isNotEmpty())
        assertTrue(result.failureAgeBuckets.sumOf { it.count } > 0)
        assertTrue(result.notFailedByAge.isNotEmpty())
        assertTrue(result.notFailedByAge.last().notFailedShare < 0.10)
    }

    @Test
    fun guaranteedIncomeOffsetsRetirementNeed() {
        val noIncome = sampleScenario().copy(
            household = HouseholdProfile(currentAge = 40, retirementAge = 40, targetEndAge = 45),
            accounts = AccountBalances(pretax = 0.0, roth = 0.0, taxable = 0.0, cash = 0.0),
            spending = SpendingPlan(
                annualBaseSpending = 10_000.0,
                generalInflationMean = 0.0,
                generalInflationStdDev = 0.0,
                spendingPathModel = SpendingPathModel.Flat,
                lowPortfolioSpendingReduction = 0.0
            ),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            market = MarketAssumptions(
                preRetirementMeanReturn = 0.0,
                preRetirementStdDev = 0.0,
                stockMeanReturn = 0.0,
                stockStdDev = 0.0,
                bondMeanReturn = 0.0,
                bondStdDev = 0.0
            ),
            withdrawalStrategy = WithdrawalStrategy(useCashReserveDuringDrawdowns = false),
            numberOfSimulations = 1
        )
        val withIncome = noIncome.copy(
            guaranteedIncome = GuaranteedIncomePlan(annualIncome = 12_000.0, startAge = 40)
        )

        assertEquals(0.0, RetirementSimulator.run(noIncome).successProbability, 0.0001)
        assertEquals(1.0, RetirementSimulator.run(withIncome).successProbability, 0.0001)
    }

    @Test
    fun marriedScenarioContinuesUntilBothSpousesHaveDied() {
        val scenario = sampleScenario().copy(
            household = HouseholdProfile(
                currentAge = 119,
                retirementAge = 119,
                targetEndAge = 120,
                filingStatus = FilingStatus.Married,
                gender = Gender.Male,
                spouseCurrentAge = 65
            ),
            spending = SpendingPlan(annualBaseSpending = 0.0),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            numberOfSimulations = 25
        )

        val result = RetirementSimulator.run(scenario)

        assertEquals(1.0, result.successProbability, 0.0001)
        assertTrue(result.notFailedByAge.last().age > scenario.household.targetEndAge)
    }

    @Test
    fun resultIncludesSimulationPathChartData() {
        val scenario = sampleScenario()
        val result = RetirementSimulator.run(scenario)

        assertTrue(result.pathPoints.isNotEmpty())
        assertTrue(result.meanPath.isNotEmpty())
        assertTrue(result.notFailedByAge.isNotEmpty())
        assertEquals(scenario.household.retirementAge, result.notFailedByAge.first().age)
        assertTrue(result.notFailedByAge.last().age <= scenario.household.targetEndAge)
        assertTrue(result.notFailedByAge.all { it.notFailedShare in 0.0..1.0 })
        assertTrue(result.notFailedByAge.all { it.aliveShare in 0.0..1.0 })
        assertEquals(1.0, result.notFailedByAge.first().aliveShare, 0.0001)
        assertTrue(result.pathPoints.all { it.balance > 0.0 })
        assertTrue(result.meanPath.all { it.balance > 0.0 })
    }

    @Test
    fun simulationPathChartStopsAfterPortfolioDepletion() {
        val scenario = sampleScenario().copy(
            household = HouseholdProfile(currentAge = 65, retirementAge = 65, targetEndAge = 75),
            accounts = AccountBalances(pretax = 0.0, roth = 0.0, taxable = 0.0, cash = 11_000.0),
            spending = SpendingPlan(
                annualBaseSpending = 12_000.0,
                generalInflationMean = 0.0,
                generalInflationStdDev = 0.0
            ),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            market = MarketAssumptions(
                preRetirementMeanReturn = 0.0,
                preRetirementStdDev = 0.0,
                stockMeanReturn = 0.0,
                stockStdDev = 0.0,
                bondMeanReturn = 0.0,
                bondStdDev = 0.0
            ),
            withdrawalStrategy = WithdrawalStrategy(useCashReserveDuringDrawdowns = false),
            numberOfSimulations = 1
        )

        val result = RetirementSimulator.run(scenario)

        assertEquals(0.0, result.successProbability, 0.0001)
        assertEquals(scenario.household.retirementAge, result.medianFailureAge)
        assertEquals(0.0, result.notFailedByAge.first().notFailedShare, 0.0001)
        assertEquals(1.0, result.notFailedByAge.first().aliveShare, 0.0001)
        assertTrue(result.notFailedByAge.all { it.notFailedShare == 0.0 })
        assertTrue(result.pathPoints.isNotEmpty())
        assertTrue(result.pathPoints.maxOf { it.yearsInRetirement } <= 0)
        assertTrue(result.meanPath.maxOf { it.yearsInRetirement } <= 0)
    }

    @Test
    fun deathWithoutDepletionRemainsFundedThroughChartHorizon() {
        val result = RetirementSimulator.run(
            sampleScenario().copy(
                household = HouseholdProfile(currentAge = 65, retirementAge = 65, targetEndAge = 90),
                spending = SpendingPlan(annualBaseSpending = 0.0),
                healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
                numberOfSimulations = 25
            )
        )

        assertEquals(1.0, result.successProbability, 0.0001)
        assertTrue(result.notFailedByAge.isNotEmpty())
        assertTrue(result.notFailedByAge.all { it.notFailedShare == 1.0 })
        assertTrue(result.notFailedByAge.all { it.aliveShare in 0.0..1.0 })
        assertTrue(result.notFailedByAge.last().aliveShare > 0.0)
    }

    @Test
    fun stillFundedChartStopsAtMaximumSimulatedSurvivalAge() {
        val scenario = sampleScenario().copy(
            household = HouseholdProfile(currentAge = 119, retirementAge = 119, targetEndAge = 130),
            spending = SpendingPlan(annualBaseSpending = 0.0),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            numberOfSimulations = 25
        )

        val result = RetirementSimulator.run(scenario)

        assertEquals(scenario.household.retirementAge, result.notFailedByAge.first().age)
        assertTrue(result.notFailedByAge.last().age <= 120)
        assertTrue(result.notFailedByAge.last().age < scenario.household.targetEndAge)
        assertTrue(result.notFailedByAge.last().aliveShare > 0.0)
    }

    @Test
    fun preRetirementGrowthDoesNotFloorMonthlyLosses() {
        val random = object : Random(0L) {
            override fun nextGaussian(): Double = -3.0
        }

        val growth = RetirementSimulator.samplePreRetirementMonthlyGrowth(
            random = random,
            mean = 0.0,
            stdDev = 1.0
        )

        assertEquals(-3.0, growth, 0.0001)
    }

    @Test
    fun longTermCareReplacesNormalSpendingWhileActive() {
        val normalNeed = RetirementSimulator.monthlyRetirementNeed(
            monthlySpending = 5_000.0,
            mortgageCost = 1_500.0,
            rentCost = 1_200.0,
            healthcareCost = 300.0,
            longTermCareCost = 8_000.0,
            inLongTermCare = false
        )
        val careNeed = RetirementSimulator.monthlyRetirementNeed(
            monthlySpending = 5_000.0,
            mortgageCost = 1_500.0,
            rentCost = 1_200.0,
            healthcareCost = 300.0,
            longTermCareCost = 8_000.0,
            inLongTermCare = true
        )

        assertEquals(8_000.0, normalNeed, 0.01)
        assertEquals(8_300.0, careNeed, 0.01)
    }

    @Test
    fun rentIsModeledAsSeparateRetirementHousingCost() {
        val withoutRent = sampleScenario().copy(
            household = HouseholdProfile(currentAge = 65, retirementAge = 65, targetEndAge = 80),
            accounts = AccountBalances(pretax = 0.0, roth = 0.0, taxable = 0.0, cash = 220_000.0),
            spending = SpendingPlan(
                annualBaseSpending = 0.0,
                generalInflationMean = 0.0,
                generalInflationStdDev = 0.0
            ),
            rent = RentPlan(monthlyRent = 0.0),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            market = MarketAssumptions(
                preRetirementMeanReturn = 0.0,
                preRetirementStdDev = 0.0,
                stockMeanReturn = 0.0,
                stockStdDev = 0.0,
                bondMeanReturn = 0.0,
                bondStdDev = 0.0
            ),
            withdrawalStrategy = WithdrawalStrategy(useCashReserveDuringDrawdowns = false),
            numberOfSimulations = 50
        )
        val withRent = withoutRent.copy(rent = RentPlan(monthlyRent = 2_000.0))

        assertEquals(1.0, RetirementSimulator.run(withoutRent).successProbability, 0.0001)
        assertTrue(RetirementSimulator.run(withRent).successProbability < 1.0)
    }

    @Test
    fun homeSaleCanRescuePathWhenPortfolioRunsOut() {
        val withoutHome = sampleScenario().copy(
            household = HouseholdProfile(currentAge = 65, retirementAge = 65, targetEndAge = 75),
            accounts = AccountBalances(pretax = 0.0, roth = 0.0, taxable = 0.0, cash = 10_000.0),
            spending = SpendingPlan(
                annualBaseSpending = 12_000.0,
                generalInflationMean = 0.0,
                generalInflationStdDev = 0.0
            ),
            home = HomePlan(currentValue = 0.0),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            market = MarketAssumptions(
                preRetirementMeanReturn = 0.0,
                preRetirementStdDev = 0.0,
                stockMeanReturn = 0.0,
                stockStdDev = 0.0,
                bondMeanReturn = 0.0,
                bondStdDev = 0.0
            ),
            withdrawalStrategy = WithdrawalStrategy(useCashReserveDuringDrawdowns = false),
            numberOfSimulations = 1
        )
        val withNoEquity = withoutHome.copy(
            home = HomePlan(currentValue = 1_000_000.0),
            mortgage = MortgagePlan(currentBalance = 1_000_000.0)
        )
        val withHome = withoutHome.copy(home = HomePlan(currentValue = 1_000_000.0))

        val withoutHomeResult = RetirementSimulator.run(withoutHome)
        val withNoEquityResult = RetirementSimulator.run(withNoEquity)
        val withHomeResult = RetirementSimulator.run(withHome)

        assertEquals(0.0, withoutHomeResult.successProbability, 0.0001)
        assertEquals(0.0, withNoEquityResult.successProbability, 0.0001)
        assertEquals(1.0, withHomeResult.successProbability, 0.0001)
        assertTrue(withHomeResult.medianEndingBalance > withoutHomeResult.medianEndingBalance)
    }

    @Test
    fun flexibleSpendingCutCanRescueLowPortfolioPath() {
        val noCut = sampleScenario().copy(
            household = HouseholdProfile(currentAge = 40, retirementAge = 40, targetEndAge = 45),
            accounts = AccountBalances(pretax = 0.0, roth = 0.0, taxable = 0.0, cash = 50_000.0),
            spending = SpendingPlan(
                annualBaseSpending = 12_000.0,
                generalInflationMean = 0.0,
                generalInflationStdDev = 0.0,
                lowPortfolioSpendingReduction = 0.0
            ),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            market = MarketAssumptions(
                preRetirementMeanReturn = 0.0,
                preRetirementStdDev = 0.0,
                stockMeanReturn = 0.0,
                stockStdDev = 0.0,
                bondMeanReturn = 0.0,
                bondStdDev = 0.0
            ),
            withdrawalStrategy = WithdrawalStrategy(useCashReserveDuringDrawdowns = false),
            numberOfSimulations = 1
        )
        val withCut = noCut.copy(
            spending = noCut.spending.copy(lowPortfolioSpendingReduction = 0.50)
        )

        assertEquals(0.0, RetirementSimulator.run(noCut).successProbability, 0.0001)
        assertEquals(1.0, RetirementSimulator.run(withCut).successProbability, 0.0001)
    }

    @Test
    fun empiricalSpendingPathReducesRealBaseSpendingAfterAge65() {
        val flat = sampleScenario().copy(
            household = HouseholdProfile(currentAge = 65, retirementAge = 65, targetEndAge = 75),
            accounts = AccountBalances(pretax = 0.0, roth = 0.0, taxable = 0.0, cash = 200_000.0),
            spending = SpendingPlan(
                annualBaseSpending = 12_000.0,
                generalInflationMean = 0.0,
                generalInflationStdDev = 0.0,
                spendingPathModel = SpendingPathModel.Flat,
                lowPortfolioSpendingReduction = 0.0
            ),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 0.0, includeMedicarePremiums = false),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 0.0, claimAge = 67),
            market = MarketAssumptions(
                preRetirementMeanReturn = 0.0,
                preRetirementStdDev = 0.0,
                stockMeanReturn = 0.0,
                stockStdDev = 0.0,
                bondMeanReturn = 0.0,
                bondStdDev = 0.0
            ),
            withdrawalStrategy = WithdrawalStrategy(useCashReserveDuringDrawdowns = false),
            numberOfSimulations = 1
        )
        val empirical = flat.copy(
            spending = flat.spending.copy(spendingPathModel = SpendingPathModel.EmpiricalAgeDecline)
        )
        val flatResult = RetirementSimulator.run(flat)
        val empiricalResult = RetirementSimulator.run(empirical)

        assertEquals(1.0, flatResult.successProbability, 0.0001)
        assertEquals(1.0, empiricalResult.successProbability, 0.0001)
        assertTrue(empiricalResult.medianEndingBalance > flatResult.medianEndingBalance)
    }

    private fun sampleScenario(): RetirementScenario {
        return RetirementScenario(
            id = "test",
            name = "Test",
            household = HouseholdProfile(currentAge = 50, retirementAge = 60, targetEndAge = DEFAULT_PROJECTION_END_AGE),
            accounts = AccountBalances(pretax = 700_000.0, roth = 100_000.0, taxable = 25_000.0, cash = 50_000.0),
            spending = SpendingPlan(annualBaseSpending = 60_000.0),
            healthcare = HealthcarePlan(preMedicareMonthlyPremium = 500.0),
            socialSecurity = SocialSecurityPlan(annualBenefitAt67 = 28_000.0, claimAge = 67),
            numberOfSimulations = 200,
            seed = 1234L
        )
    }
}
