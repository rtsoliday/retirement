export const ENGINE_VERSION = '2026.08-web-port';
export const DEFAULT_SEED = 20260429;
export const ALLOCATION_KEYS = ['stockUnder30x', 'stock30xTo35x', 'stock35xTo40x', 'stock40xTo45x', 'stock45xTo50x', 'stock50xOrMore'];

export function baseScenario() {
  return {
    id: 'base-plan', name: 'Base plan',
    household: {currentAge: 50, retirementAge: 67, targetEndAge: 119, filingStatus: 'Single', gender: 'Male', spouseGender: 'Female', spouseCurrentAge: 50},
    accounts: {pretax: 800000, roth: 100000, taxable: 0, cash: 50000},
    spending: {annualBaseSpending: 75000, generalInflationMean: .023, generalInflationStdDev: .016, spendingPathModel: 'EmpiricalAgeDecline', lowPortfolioSpendingReduction: .10},
    budget: {annualPropertyTaxes: 0, annualHomeInsurance: 0, annualAutoInsurance: 0, monthlyBudgets: [], isAppliedToAnnualBaseSpending: false},
    mortgage: {monthlyPayment: 0, yearsLeft: 0, monthsLeft: 0, currentBalance: 0},
    rent: {monthlyRent: 0}, home: {currentValue: 0},
    healthcare: {preMedicareMonthlyPremium: 1250, healthcareInflationMean: .04, healthcareInflationStdDev: .018, includeMedicarePremiums: true},
    socialSecurity: {annualBenefitAt67: 30000, claimAge: 67, spouseClaimAge: 67},
    guaranteedIncome: {annualIncome: 0, startAge: 65, annualIncrease: 0, survivorPercent: 1},
    market: {preRetirementMeanReturn: .133, preRetirementStdDev: .162, stockMeanReturn: .133, stockStdDev: .162, bondMeanReturn: .03, bondStdDev: .06},
    postRetirementAllocation: {stockUnder30x: 1, stock30xTo35x: .9, stock35xTo40x: .8, stock40xTo45x: .7, stock45xTo50x: .6, stock50xOrMore: .5},
    rothConversion: {enabled: false, marginalRateCap: .22},
    withdrawalStrategy: {useCashReserveDuringDrawdowns: false, drawdownTrigger: -.01, applyEarlyWithdrawalPenalty: false, ruleOf55Eligible: false, seppEligible: false},
    longTermCare: {enabled: true, annualCost: 100000, averageDurationYears: 3},
    numberOfSimulations: 1500, seed: DEFAULT_SEED
  };
}

export function sampleScenarios() {
  const base = baseScenario();
  const later = structuredClone(base);
  later.id = 'later-retirement'; later.name = 'Retire at 62'; later.household.retirementAge = 62;
  later.socialSecurity.claimAge = 70; later.rothConversion.enabled = true; later.seed = 20260430;
  const lean = structuredClone(base);
  lean.id = 'lean-plan'; lean.name = 'Lower spending'; lean.spending.annualBaseSpending = 68000; lean.seed = 20260431;
  return [base, later, lean];
}

export function normalizeScenario(raw) {
  if (raw?.currentAge !== undefined && !raw.household) {
    raw = {
      id: raw.id, name: raw.name,
      household: {currentAge: raw.currentAge, retirementAge: raw.retirementAge, spouseCurrentAge: raw.currentAge},
      accounts: {pretax: raw.pretaxBalance ?? 0, roth: raw.rothBalance ?? 0, cash: raw.cashBalance ?? 0},
      spending: {annualBaseSpending: raw.annualSpending ?? 0},
      socialSecurity: {annualBenefitAt67: raw.socialSecurityAt67 ?? 0, claimAge: raw.socialSecurityClaimAge ?? 67},
      healthcare: {preMedicareMonthlyPremium: raw.preMedicareMonthlyPremium ?? 1250},
      longTermCare: {enabled: !!raw.includeLongTermCare},
      rothConversion: {enabled: !!raw.enableRothConversion},
      withdrawalStrategy: {useCashReserveDuringDrawdowns: !!raw.enableCashReserveStrategy}
    };
  }
  const base = baseScenario();
  const result = structuredClone(base);
  for (const key of Object.keys(base)) {
    if (raw?.[key] === undefined) continue;
    if (base[key] && typeof base[key] === 'object' && !Array.isArray(base[key])) {
      result[key] = {...base[key], ...raw[key]};
    } else result[key] = raw[key];
  }
  return result;
}

export function validateScenario(s) {
  const errors = [];
  const h=s.household, a=s.accounts, sp=s.spending;
  const allNumbers = [];
  function collect(value) { if (typeof value === 'number') allNumbers.push(value); else if (value && typeof value === 'object') Object.values(value).forEach(collect); }
  collect(s);
  if (allNumbers.some(v=>!Number.isFinite(v))) errors.push('Financial and percentage assumptions must be finite numbers.');
  if (h.currentAge <= 0) errors.push('Current age must be positive.');
  if (h.retirementAge < h.currentAge) errors.push('Retirement age must be at least current age.');
  if (h.targetEndAge <= h.retirementAge || h.targetEndAge > 119) errors.push('Maximum modeling age must be after retirement and at most 119.');
  if (h.filingStatus === 'Married' && (h.spouseCurrentAge <= 0 || h.spouseCurrentAge + h.retirementAge - h.currentAge >= h.targetEndAge)) errors.push('Spouse age at retirement must be below the maximum modeling age.');
  if (s.socialSecurity.claimAge < 62 || s.socialSecurity.claimAge > 70) errors.push('Social Security claim age must be 62–70.');
  if (h.filingStatus === 'Married' && (s.socialSecurity.spouseClaimAge < 60 || s.socialSecurity.spouseClaimAge > 70)) errors.push('Spouse claim age must be 60–70.');
  if (Object.values(a).some(v=>v<0) || sp.annualBaseSpending<0) errors.push('Balances and spending cannot be negative.');
  if (sp.generalInflationMean < -.02 || sp.generalInflationMean > .15 || sp.generalInflationStdDev < 0 || sp.generalInflationStdDev > .3) errors.push('General inflation assumptions are outside the supported range.');
  if (sp.lowPortfolioSpendingReduction < 0 || sp.lowPortfolioSpendingReduction > 1) errors.push('Spending reduction must be between 0% and 100%.');
  if (s.healthcare.healthcareInflationMean < 0 || s.healthcare.healthcareInflationMean > .2 || s.healthcare.healthcareInflationStdDev < 0 || s.healthcare.healthcareInflationStdDev > .3) errors.push('Healthcare inflation assumptions are outside the supported range.');
  if (s.numberOfSimulations < 1 || s.numberOfSimulations > 10000 || !Number.isInteger(s.numberOfSimulations)) errors.push('Simulation count must be between 1 and 10,000.');
  if (!Number.isSafeInteger(s.seed)) errors.push('Seed must be an integer.');
  if (s.mortgage.yearsLeft < 0 || s.mortgage.yearsLeft > 80 || s.mortgage.monthsLeft < 0 || s.mortgage.monthsLeft > 11 || s.mortgage.monthlyPayment < 0 || s.mortgage.currentBalance < 0) errors.push('Mortgage payment, balance, or term is invalid.');
  if (s.rent.monthlyRent < 0 || s.home.currentValue < 0) errors.push('Housing amounts cannot be negative.');
  if (s.healthcare.preMedicareMonthlyPremium < 0) errors.push('Healthcare premium cannot be negative.');
  if (s.socialSecurity.annualBenefitAt67 < 0) errors.push('Social Security estimate cannot be negative.');
  if (s.guaranteedIncome.annualIncome < 0 || s.guaranteedIncome.startAge < 0 || s.guaranteedIncome.annualIncrease < -.02 || s.guaranteedIncome.annualIncrease > .15 || (h.filingStatus === 'Married' && (s.guaranteedIncome.survivorPercent < 0 || s.guaranteedIncome.survivorPercent > 1))) errors.push('Guaranteed income assumptions are outside the supported range.');
  const market=s.market;
  if (market.preRetirementMeanReturn < -.20 || market.preRetirementMeanReturn > .25 || market.stockMeanReturn < -.20 || market.stockMeanReturn > .25 || market.bondMeanReturn < -.20 || market.bondMeanReturn > .20 || market.preRetirementStdDev < 0 || market.preRetirementStdDev > .60 || market.stockStdDev < 0 || market.stockStdDev > .60 || market.bondStdDev < 0 || market.bondStdDev > .40) errors.push('Market return assumptions are outside the supported range.');
  if (ALLOCATION_KEYS.some(k=>s.postRetirementAllocation[k]<0 || s.postRetirementAllocation[k]>1)) errors.push('Stock allocation must be between 0% and 100%.');
  if (s.rothConversion.enabled && ![.10,.12,.22,.24,.32,.35,.37].some(x=>Math.abs(x-s.rothConversion.marginalRateCap)<.0001)) errors.push('Roth conversion cap must be a supported tax bracket.');
  if (s.longTermCare.annualCost < 0 || s.longTermCare.averageDurationYears < 1 || s.longTermCare.averageDurationYears > 10) errors.push('Long-term care cost or duration is invalid.');
  if (s.withdrawalStrategy.drawdownTrigger < -.50 || s.withdrawalStrategy.drawdownTrigger > .25) errors.push('Cash drawdown trigger is outside the supported range.');
  if (s.budget.annualPropertyTaxes < 0 || s.budget.annualHomeInsurance < 0 || s.budget.annualAutoInsurance < 0 || s.budget.monthlyBudgets.some(m=>m.cashAndAtmWithdrawals<0 || [...(m.checkingSavingsBills||[]),...(m.creditCardBills||[])].some(i=>i.monthlyAmount<0))) errors.push('Budget amounts cannot be negative.');
  return errors;
}

export function budgetEstimate(budget) {
  const byMonth = new Map((budget.monthlyBudgets||[]).map(m=>[m.month,m]));
  const months = [...byMonth.values()].sort((a,b)=>a.month.localeCompare(b.month)).slice(-12);
  const monthlyAverage = months.length ? months.reduce((sum,m)=>sum + Number(m.cashAndAtmWithdrawals||0) + [...(m.checkingSavingsBills||[]),...(m.creditCardBills||[])].reduce((x,i)=>x+Number(i.monthlyAmount||0),0),0)/months.length : 0;
  return monthlyAverage*12 + Number(budget.annualPropertyTaxes||0) + Number(budget.annualHomeInsurance||0) + Number(budget.annualAutoInsurance||0);
}

export function scenarioWarnings(s) {
  const notes=[],h=s.household,total=Object.values(s.accounts).reduce((a,b)=>a+b,0),years=h.retirementAge-h.currentAge;
  if(h.retirementAge<50)notes.push('Retiring before 50 creates a long drawdown period.');
  if(s.spending.annualBaseSpending/Math.max(1,total)>.07)notes.push('Annual base spending exceeds 7% of current assets.');
  if(s.spending.generalInflationMean<.015)notes.push('General inflation below 1.5% may understate spending pressure.');
  if(s.market.stockMeanReturn>.145||s.market.preRetirementMeanReturn>.145)notes.push('Expected return above 14.5% may make readiness look stronger.');
  const spouseAtRet=h.spouseCurrentAge+years;
  if((h.retirementAge<65||(h.filingStatus==='Married'&&spouseAtRet<65))&&s.healthcare.preMedicareMonthlyPremium<=0)notes.push('A pre-Medicare adult has no healthcare premium entered.');
  if(!s.healthcare.includeMedicarePremiums)notes.push('Medicare premiums are excluded.');
  if(!s.longTermCare.enabled)notes.push('Long-term care risk is excluded.');
  if(s.socialSecurity.annualBenefitAt67<=0)notes.push('No Social Security benefit is entered.');
  if(s.numberOfSimulations<500)notes.push('Use at least 500 simulations for final comparisons.');
  return notes;
}
