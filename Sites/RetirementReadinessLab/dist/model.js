export const ENGINE_VERSION = '2026.09-observed-balances';
export const ROTH_CONVERSION_RATES = [.10,.12,.22,.24,.32,.35,.37];
export const DEFAULT_SEED = 20260429;
export const FREE_SIMULATION_PATHS = 4;
export const MAX_SIMULATION_PATHS = 10000;
export const ALLOCATION_KEYS = ['stockUnder30x', 'stock30xTo35x', 'stock35xTo40x', 'stock40xTo45x', 'stock45xTo50x', 'stock50xOrMore'];

export function baseScenario() {
  return {
    id: 'base-plan', name: 'Base plan',
    household: {currentAge: 50, retirementAge: 67, targetEndAge: 119, filingStatus: 'Single', gender: 'Male', spouseGender: 'Female', spouseCurrentAge: 50},
    accounts: {pretax: 800000, roth: 100000, taxable: 0, cash: 50000},
    spending: {annualBaseSpending: 75000, generalInflationMean: .023, generalInflationStdDev: .016, spendingPathModel: 'EmpiricalAgeDecline', lowPortfolioSpendingReduction: .10},
    budget: {annualPropertyTaxes: 0, annualHomeInsurance: 0, annualAutoInsurance: 0, monthlyBudgets: [], retirementAnnualAdjustment: 0, estimateNeedsReview: false, isAppliedToAnnualBaseSpending: false},
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
    numberOfSimulations: FREE_SIMULATION_PATHS, simulationPathsCustomized: false, seed: DEFAULT_SEED
  };
}

export function sampleScenarios() {
  const base = baseScenario();
  const later = structuredClone(base);
  later.id = 'later-retirement'; later.name = 'Retire at 62'; later.household.retirementAge = 62;
  later.socialSecurity.claimAge = 70; later.rothConversion.enabled = true;
  const lean = structuredClone(base);
  lean.id = 'lean-plan'; lean.name = 'Lower spending'; lean.spending.annualBaseSpending = 68000;
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
  // Imported and previously saved scenarios cannot change the site's fixed seed.
  result.seed = DEFAULT_SEED;
  result.simulationPathsCustomized = raw?.simulationPathsCustomized === true;
  return result;
}

export function applyProSimulationDefault(s) {
  if (s.simulationPathsCustomized || s.numberOfSimulations !== FREE_SIMULATION_PATHS) return false;
  s.numberOfSimulations = MAX_SIMULATION_PATHS;
  return true;
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
  if (s.numberOfSimulations < 1 || s.numberOfSimulations > MAX_SIMULATION_PATHS || !Number.isInteger(s.numberOfSimulations)) errors.push('Simulation count must be between 1 and 10,000.');
  if (!Number.isSafeInteger(s.seed)) errors.push('Seed must be an integer.');
  if (s.mortgage.yearsLeft < 0 || s.mortgage.yearsLeft > 80 || s.mortgage.monthsLeft < 0 || s.mortgage.monthsLeft > 11 || s.mortgage.monthlyPayment < 0 || s.mortgage.currentBalance < 0) errors.push('Mortgage payment, balance, or term is invalid.');
  if (s.rent.monthlyRent < 0 || s.home.currentValue < 0) errors.push('Housing amounts cannot be negative.');
  if (s.healthcare.preMedicareMonthlyPremium < 0) errors.push('Healthcare premium cannot be negative.');
  if (s.socialSecurity.annualBenefitAt67 < 0) errors.push('Social Security estimate cannot be negative.');
  if (s.guaranteedIncome.annualIncome < 0 || s.guaranteedIncome.startAge < 0 || s.guaranteedIncome.annualIncrease < -.02 || s.guaranteedIncome.annualIncrease > .15 || (h.filingStatus === 'Married' && (s.guaranteedIncome.survivorPercent < 0 || s.guaranteedIncome.survivorPercent > 1))) errors.push('Guaranteed income assumptions are outside the supported range.');
  const market=s.market;
  if (market.preRetirementMeanReturn < -.20 || market.preRetirementMeanReturn > .25 || market.stockMeanReturn < -.20 || market.stockMeanReturn > .25 || market.bondMeanReturn < -.20 || market.bondMeanReturn > .20 || market.preRetirementStdDev < 0 || market.preRetirementStdDev > .60 || market.stockStdDev < 0 || market.stockStdDev > .60 || market.bondStdDev < 0 || market.bondStdDev > .40) errors.push('Market return assumptions are outside the supported range.');
  if (ALLOCATION_KEYS.some(k=>s.postRetirementAllocation[k]<0 || s.postRetirementAllocation[k]>1)) errors.push('Stock allocation must be between 0% and 100%.');
  if (s.rothConversion.enabled && !ROTH_CONVERSION_RATES.some(x=>Math.abs(x-s.rothConversion.marginalRateCap)<.0001)) errors.push('Roth conversion cap must be 10%, 12%, 22%, 24%, 32%, 35%, or 37%.');
  if (s.longTermCare.annualCost < 0 || s.longTermCare.averageDurationYears < 1 || s.longTermCare.averageDurationYears > 10) errors.push('Long-term care cost or duration is invalid.');
  if (s.withdrawalStrategy.drawdownTrigger < -.50 || s.withdrawalStrategy.drawdownTrigger > .25) errors.push('Cash drawdown trigger is outside the supported range.');
  // Draft budget errors are shown in the budget editor; only applied spending feeds the simulation.
  return errors;
}

export const ANNUAL_BILLS = [['Property taxes','annualPropertyTaxes','propertyTaxes'],['Home insurance','annualHomeInsurance','homeInsurance'],['Auto insurance','annualAutoInsurance','autoInsurance']];
export const SEPARATE_COSTS = [['Mortgage payments','mortgage'],['Rent','rent'],['Healthcare premiums','healthcare']];
const amount = v => Number(v ?? 0);
export function budgetMonthTotals(m) {
  const checking=(m.checkingSavingsBills||[]).reduce((sum,x)=>sum+amount(x.monthlyAmount),0);
  const credit=(m.creditCardBills||[]).reduce((sum,x)=>sum+amount(x.monthlyAmount),0);
  const gross=checking+credit+amount(m.cashAndAtmWithdrawals);
  const annualBills=ANNUAL_BILLS.reduce((sum,[,,key])=>sum+amount(m.adjustments?.[key]),0);
  const separateCosts=SEPARATE_COSTS.reduce((sum,[,key])=>sum+amount(m.adjustments?.[key]),0);
  return {checking,credit,gross,annualBills,separateCosts,adjusted:gross-annualBills-separateCosts};
}
export function budgetBreakdown(budget) {
  const months=[...(budget.monthlyBudgets||[])].sort((a,b)=>String(a.month).localeCompare(String(b.month))).slice(-12);
  const totals=months.reduce((sum,m)=>{const row=budgetMonthTotals(m);for(const key of Object.keys(sum))sum[key]+=row[key];return sum;},{gross:0,annualBills:0,separateCosts:0,adjusted:0});
  const count=months.length,average=key=>count?totals[key]/count:0;
  const annualBills=ANNUAL_BILLS.reduce((sum,[,key])=>sum+amount(budget[key]),0);
  const retirementAdjustment=amount(budget.retirementAnnualAdjustment);
  return {months,count,totals,grossAverage:average('gross'),annualBillsAverage:average('annualBills'),separateCostsAverage:average('separateCosts'),monthlyAverage:average('adjusted'),annualized:average('adjusted')*12,annualBills,retirementAdjustment,estimate:average('adjusted')*12+annualBills+retirementAdjustment};
}
export function validateBudget(budget,{requireMonths=false}={}) {
  const errors=[],months=budget.monthlyBudgets||[],seen=new Set();
  const nonnegative=v=>Number.isFinite(amount(v))&&amount(v)>=0;
  if(requireMonths&&!months.length)errors.push('Add at least one complete month of spending before using the estimate.');
  for(const [label,key] of ANNUAL_BILLS)if(!nonnegative(budget[key]))errors.push(`${label}: enter a finite amount of 0 or more.`);
  if(!Number.isFinite(amount(budget.retirementAnnualAdjustment)))errors.push('Enter a finite retirement adjustment.');
  for(const [i,m] of months.entries()){
    const label=/^\d{4}-(0[1-9]|1[0-2])$/.test(m.month)?m.month:`Month ${i+1}`;
    if(label!==m.month)errors.push(`${label}: choose a valid month.`);
    if(seen.has(m.month))errors.push(`${label}: this month is entered twice. Keep one combined entry per month.`);seen.add(m.month);
    if(!nonnegative(m.cashAndAtmWithdrawals)||!(m.checkingSavingsBills||[]).every(x=>nonnegative(x.monthlyAmount)))errors.push(`${label}: cash and bank spending must be 0 or more.`);
    if(!(m.creditCardBills||[]).every(x=>Number.isFinite(amount(x.monthlyAmount))))errors.push(`${label}: enter finite credit card spending. A negative net refund is allowed.`);
    if([...ANNUAL_BILLS.map(x=>x[2]),...SEPARATE_COSTS.map(x=>x[1])].some(k=>!nonnegative(m.adjustments?.[k])))errors.push(`${label}: amounts already counted must be 0 or more.`);
  }
  const d=budgetBreakdown(budget);
  // Refund-heavy months may be negative; evaluate deductions over the entire sample.
  if(d.totals.adjusted < -0.005)errors.push('Adjustments exceed spending across the selected months. Check for amounts deducted twice.');
  if(d.estimate < 0)errors.push('The retirement adjustment would make the annual estimate negative.');
  return errors;
}
export function budgetEstimate(budget) { return budgetBreakdown(budget).estimate; }
export function markBudgetEdited(budget) {
  if(budget.isAppliedToAnnualBaseSpending&&budget.appliedAnnualHomeCosts===undefined)budget.appliedAnnualHomeCosts=amount(budget.annualPropertyTaxes)+amount(budget.annualHomeInsurance);
  budget.estimateNeedsReview=true;
}
export function applyBudgetEstimate(scenario) {
  const errors=validateBudget(scenario.budget,{requireMonths:true});
  if(errors.length)throw new Error(errors.join(' '));
  const b=scenario.budget;
  scenario.spending.annualBaseSpending=budgetEstimate(b);
  b.appliedAnnualHomeCosts=amount(b.annualPropertyTaxes)+amount(b.annualHomeInsurance);
  b.isAppliedToAnnualBaseSpending=true;b.estimateNeedsReview=false;
  return scenario.spending.annualBaseSpending;
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
  if(s.numberOfSimulations===4)notes.push('Four paths are only a preview. Results move in 25% steps and are not reliable for decisions. Use many more paths for serious comparisons.');
  else if(s.numberOfSimulations<500)notes.push('Fewer than 500 paths can make comparisons unstable. Use more paths before relying on small differences.');
  return notes;
}
