import {savingsDefaults,hasFutureSavings} from './savings.js';
import {employerRothDefaults,employerRothTotal,hasEmployerRoth,validateEmployerRoth} from './employer-roth.js';
export {employerRothDefaults,employerRothTotal,hasEmployerRoth};
export const ENGINE_VERSION = '2026.09-calendar-dates';
export const scenarioEngineVersion=s=>{
  const version=s.household.alreadyRetired||s.household.separatePeople&&s.household.filingStatus==='Married'&&s.household.spouseAlreadyRetired?'2026.10-retired-forecast':s.household.separatePeople?'2026.10-separate-people':hasFutureSavings(s)?'2026.10-savings-contributions':ENGINE_VERSION;
  // Saved results predating these calculation corrections need a rerun.
  return version+(s.household.separatePeople?'-medicare-rmd-v2':'-survivor-pension-v2')+(hasEmployerRoth(s)?'-employer-roth-v1':'');
};
export const ROTH_CONVERSION_RATES = [.10,.12,.22,.24,.32,.35,.37];
// Retained across engine revisions for repeatable scenario comparisons.
// Android still uses 20260429.
export const DEFAULT_SEED = 20260766;
export const FREE_SIMULATION_PATHS = 100;
// Preserve small-run lifespan sampling independently of the free entitlement.
export const SMALL_SAMPLE_PATHS = 10;
// Keep the existing Pro selection range independent of the free allowance.
export const MIN_SIMULATION_PATHS = 4;
export const MAX_SIMULATION_PATHS = 10000;
export const MAX_DOLLAR_AMOUNT = Number.MAX_SAFE_INTEGER;
export const ALLOCATION_KEYS = ['stockUnder30x', 'stock30xTo35x', 'stock35xTo40x', 'stock40xTo45x', 'stock45xTo50x', 'stock50xOrMore'];
export const FILING_STATUSES = ['Single', 'Married', 'HeadOfHousehold'];
export const GENDERS = ['Male', 'Female'];
export const SPENDING_PATH_MODELS = ['EmpiricalAgeDecline', 'Flat'];

// Calendar values remain YYYY-MM-DD in storage; never parse them in local time.
export function localCalendarDate() {
  const now=new Date();
  return `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;
}
export function calendarDate(value) {
  if(typeof value!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(value))return null;
  const date=new Date(value+'T00:00:00Z');
  return Number.isFinite(date.getTime())&&date.toISOString().slice(0,10)===value?date:null;
}
export function addCalendarMonths(value,months) {
  const date=calendarDate(value);if(!date||!Number.isInteger(months))return '';
  const day=date.getUTCDate();date.setUTCDate(1);date.setUTCMonth(date.getUTCMonth()+months);
  const end=new Date(date.getTime());end.setUTCMonth(end.getUTCMonth()+1);end.setUTCDate(0);
  date.setUTCDate(Math.min(day,end.getUTCDate()));
  return Number.isFinite(date.getTime())?date.toISOString().slice(0,10):'';
}
// Completed calendar months, with month-end birthdays clamped to that month's last day.
export function calendarMonthsBetween(start,end) {
  const a=calendarDate(start),b=calendarDate(end);if(!a||!b)return NaN;
  const months=(b.getUTCFullYear()-a.getUTCFullYear())*12+b.getUTCMonth()-a.getUTCMonth();
  return months-Number(addCalendarMonths(start,months)>end);
}
export const usesCalendarDates = s => Boolean(s?.household?.birthday||s?.household?.retirementDate||s?.household?.alreadyRetired||s?.household?.separatePeople&&s?.household?.filingStatus==='Married'&&s?.household?.spouseAlreadyRetired);
// Current balances describe today, even when the recorded separation was years
// ago. Retain the actual date for eligibility checks; never replay past growth.
export function forecastRetirementDate(s,spouse=false,today=s.household.asOfDate||localCalendarDate()) {
  const h=s.household;
  return h[spouse?'spouseAlreadyRetired':'alreadyRetired']?today:h[spouse?'spouseRetirementDate':'retirementDate'];
}
export function scenarioTimeline(s,today=s.household.asOfDate||localCalendarDate()) {
  const h=s.household;
  if(!usesCalendarDates(s)){
    const retirementAge=h.retirementAge+(h.retirementAgeMonths??0)/12,preMonths=Math.round((retirementAge-h.currentAge)*12);
    return {currentAge:h.currentAge,retirementAge,preMonths,spouseAtRet:h.spouseCurrentAge+preMonths/12,birthYear:2026-h.currentAge,spouseBirthYear:2026-h.spouseCurrentAge,retirementYear:2026+Math.floor(preMonths/12)};
  }
  const ownDate=forecastRetirementDate(s,false,today),spouseDate=forecastRetirementDate(s,true,today);
  const startDate=h.separatePeople&&h.filingStatus==='Married'&&calendarDate(spouseDate)&&spouseDate<ownDate?spouseDate:ownDate;
  return {...(h.separatePeople?{startDate}:{}),currentAge:calendarMonthsBetween(h.birthday,today)/12,retirementAge:calendarMonthsBetween(h.birthday,startDate)/12,preMonths:calendarMonthsBetween(today,startDate),spouseAtRet:calendarMonthsBetween(h.spouseBirthday,startDate)/12,birthYear:calendarDate(h.birthday)?.getUTCFullYear(),spouseBirthYear:calendarDate(h.spouseBirthday)?.getUTCFullYear(),retirementYear:calendarDate(startDate)?.getUTCFullYear()};
}
export const retirementAge = s => scenarioTimeline(s).retirementAge;
export const primaryRetirementAge = s => usesCalendarDates(s)?calendarMonthsBetween(s.household.birthday,forecastRetirementDate(s))/12:retirementAge(s);
export function dateLabel(value) {
  const date=calendarDate(value);
  return date?new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}).format(date):'Choose a date';
}
// Legacy ages contain no birth day. Anchor their inferred birthdays to the day
// of migration, preserving the entered monthly distance to retirement.
export function prepareCalendarScenario(s,{today=localCalendarDate(),needsReview=true}={}) {
  const h=s.household;
  if(!usesCalendarDates(s)&&h.birthday===''&&h.retirementDate===''&&[h.currentAge,h.retirementAge,h.retirementAgeMonths].every(Number.isInteger)){
    h.birthday=addCalendarMonths(today,-h.currentAge*12);
    h.retirementDate=addCalendarMonths(h.birthday,h.retirementAge*12+h.retirementAgeMonths);
    h.spouseBirthday=Number.isInteger(h.spouseCurrentAge)&&h.spouseCurrentAge>0?addCalendarMonths(today,-h.spouseCurrentAge*12):'';
    h.datesNeedReview=needsReview;
  }
  return s;
}
export function setRetirementAge(s,age) {
  s.household.alreadyRetired=false;
  if(usesCalendarDates(s))s.household.retirementDate=addCalendarMonths(s.household.birthday,Math.round(age*12));
  s.household.retirementAge=Math.floor(age);s.household.retirementAgeMonths=Math.round(age*12)%12;
}
export function syncCalendarAges(s) {
  const h=s.household;h.asOfDate='';
  const t=scenarioTimeline(s),spouseMonths=calendarMonthsBetween(h.spouseBirthday,localCalendarDate());
  if(Number.isFinite(t.currentAge))h.currentAge=Math.floor(t.currentAge);
  const ownAge=primaryRetirementAge(s);if(Number.isFinite(ownAge)){h.retirementAge=Math.floor(ownAge);h.retirementAgeMonths=Math.round(ownAge*12)%12;}
  if(Number.isFinite(spouseMonths))h.spouseCurrentAge=Math.floor(spouseMonths/12);
}
export function delayRetirement(s,years) {
  if(usesCalendarDates(s)){s.household.retirementDate=addCalendarMonths(forecastRetirementDate(s),years*12);s.household.alreadyRetired=false;syncCalendarAges(s);}
  else setRetirementAge(s,retirementAge(s)+years);
}
// Separation must fall in or after the calendar year the person turns 55. With
// no birthday entered, a separation at 54 can still be in that year; earlier cannot.
const RULE_OF_55_EARLIEST_AGE = 54;
export const ruleOf55TimingMatches = s => usesCalendarDates(s)?calendarDate(s.household.retirementDate)?.getUTCFullYear()>=scenarioTimeline(s).birthYear+55:retirementAge(s)>=RULE_OF_55_EARLIEST_AGE;
export const ruleOf55Applies = s => s.withdrawalStrategy.ruleOf55Eligible === true && ruleOf55TimingMatches(s);
export function earlyWithdrawalContext(s) {
  const t=scenarioTimeline(s);
  return {
    earlyRetirement:Number.isFinite(t.retirementAge)&&t.retirementAge<59.5,
    youngerSpouse:s.household.filingStatus==='Married'&&Number.isFinite(t.spouseAtRet)&&t.spouseAtRet<59.5,
    ruleOf55Timing:Number.isFinite(t.retirementAge)&&t.retirementAge<59.5&&ruleOf55TimingMatches(s)
  };
}
export function ageLabel(value) {
  const months = Math.round(value * 12), years = Math.floor(months / 12), extra = months % 12;
  return extra ? `${years} years ${extra} ${extra === 1 ? 'month' : 'months'}` : String(years);
}

export function baseScenario() {
  return {
    id: 'base-plan', name: 'Base plan',
    household: {currentAge: 60, retirementAge: 67, retirementAgeMonths: 0, birthday: '', retirementDate: '', alreadyRetired: false, spouseBirthday: '', spouseRetirementDate: '', spouseAlreadyRetired: false, separatePeople: false, datesNeedReview: false, asOfDate: '', targetEndAge: 119, filingStatus: 'Single', gender: 'Male', spouseGender: 'Female', spouseCurrentAge: 60},
    accounts: {pretax: 500000, roth: 50000, taxable: 0, cash: 50000},
    contributions:savingsDefaults(), spouseContributions:savingsDefaults(),
    spouseAccounts:{pretax:0,roth:0},
    employerRothAccounts:[],
    spouseRothHistory:{contributionBasis:0,firstContributionYear:0,conversions:[],needsReview:false},
    spouseIncome:{annualBenefitAt67:0,annualPension:0,pensionStartAge:65,pensionStartAgeMonths:0,annualIncrease:0,survivorPercent:0},
    workingIncome:{primaryAnnualNet:0,spouseAnnualNet:0,annualIncrease:0},
    spouseWithdrawal:{ruleOf55Eligible:false,seppEligible:false},
    rothHistory: {contributionBasis: 50000, firstContributionYear: 2021, conversions: [], needsReview: false},
    spending: {annualBaseSpending: 75000, generalInflationMean: .023, generalInflationStdDev: .016, spendingPathModel: 'EmpiricalAgeDecline', lowPortfolioSpendingReduction: .10},
    budget: {annualPropertyTaxes: 0, annualHomeInsurance: 0, annualAutoInsurance: 0, monthlyBudgets: [], retirementAnnualAdjustment: 0, estimateNeedsReview: false, isAppliedToAnnualBaseSpending: false},
    mortgage: {monthlyPayment: 0, yearsLeft: 0, monthsLeft: 0, currentBalance: 0},
    rent: {monthlyRent: 0}, home: {currentValue: 0, annualTaxesAndInsurance: 0},
    healthcare: {preMedicareMonthlyPremium: 1250, healthcareInflationMean: .04, healthcareInflationStdDev: .018, includeMedicarePremiums: true},
    socialSecurity: {annualBenefitAt67: 30000, claimAge: 67, spouseClaimAge: 67},
    guaranteedIncome: {annualIncome: 0, startAge: 65, startAgeMonths: 0, annualIncrease: 0, survivorPercent: 1},
    market: {preRetirementMeanReturn: .133, preRetirementStdDev: .162, stockMeanReturn: .133, stockStdDev: .162, bondMeanReturn: .03, bondStdDev: .06},
    postRetirementAllocation: {stockUnder30x: 1, stock30xTo35x: .9, stock35xTo40x: .8, stock40xTo45x: .7, stock45xTo50x: .6, stock50xOrMore: .5},
    rothConversion: {enabled: false, marginalRateCap: .22},
    withdrawalStrategy: {useCashReserveDuringDrawdowns: false, drawdownTrigger: -.01, applyEarlyWithdrawalPenalty: true, ruleOf55Eligible: false, seppEligible: false},
    longTermCare: {enabled: true, annualCost: 100000, averageDurationYears: 3, averageDurationMonths: 0},
    numberOfSimulations: FREE_SIMULATION_PATHS, simulationPathsCustomized: false, seed: DEFAULT_SEED
  };
}

// Imported values must keep the primitive type of the matching default.
const TYPE_TEMPLATE = baseScenario();

export function sampleScenarios() {
  const base = baseScenario();
  // Illustrations include shortfalls in the fixed 100-path preview.
  // Keep these choices separate from defaults used to read older plans.
  base.accounts = {pretax: 175000, roth: 17500, taxable: 0, cash: 17500};
  base.rothHistory.contributionBasis = base.accounts.roth;
  const later = structuredClone(base);
  later.id = 'later-retirement'; later.name = 'Retire at 62'; later.household.retirementAge = 62;
  later.socialSecurity.claimAge = 70; later.rothConversion.enabled = true;
  const lean = structuredClone(base);
  lean.id = 'lean-plan'; lean.name = 'Lower spending'; lean.spending.annualBaseSpending = 68000;
  return [base, later, lean];
}

export function normalizeScenario(raw) {
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  if (!object(raw)) throw new Error('Each scenario must be an object.');
  // Reject corrupt sections before spreading defaults can hide their shape.
  for (const [key, value] of Object.entries(TYPE_TEMPLATE)) {
    if (object(value) && key in raw && !object(raw[key])) throw new Error(`Scenario ${key} must be an object.`);
  }
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
  if(raw.employerRothAccounts!==undefined){
    if(!Array.isArray(raw.employerRothAccounts))throw new Error('Employer Roth accounts must be an array.');
    result.employerRothAccounts=raw.employerRothAccounts.map(a=>{
      if(!object(a))throw new Error('Each employer Roth account must be an object.');
      return {...employerRothDefaults(),...a};
    });
  }
  // The new default applies to new plans. An older saved/imported plan with
  // no penalty field previously modeled none; retain that behavior too.
  if(raw.withdrawalStrategy?.applyEarlyWithdrawalPenalty===undefined)result.withdrawalStrategy.applyEarlyWithdrawalPenalty=false;
  // Preserve old balances and assumptions, but make the missing history visible.
  // Never replace an explicitly entered basis or conversion history.
  if(raw.rothHistory===undefined){
    result.rothHistory={contributionBasis:result.accounts.roth,firstContributionYear:result.accounts.roth>0?2021:0,conversions:[],needsReview:result.accounts.roth>0};
  }
  // Older plans kept home costs only while a budget was applied. Carry that
  // amount into the explicit field so their results do not change.
  const budget=result.budget;
  if(raw.home?.annualTaxesAndInsurance===undefined){
    const legacy=budget.isAppliedToAnnualBaseSpending===true?(budget.appliedAnnualHomeCosts??amount(budget.annualPropertyTaxes)+amount(budget.annualHomeInsurance)):0;
    result.home.annualTaxesAndInsurance=typeof legacy==='number'&&Number.isFinite(legacy)?legacy:0;
  }
  // The field replaces the old snapshot. A malformed snapshot stays so the
  // structure check still rejects it.
  if(typeof budget.appliedAnnualHomeCosts==='number'&&Number.isFinite(budget.appliedAnnualHomeCosts))delete budget.appliedAnnualHomeCosts;
  // Older backups used fractional pension ages and care durations. Split these
  // into years and months once; explicit month fields must already be valid.
  for (const [section, years, months] of [['guaranteedIncome','startAge','startAgeMonths'],['longTermCare','averageDurationYears','averageDurationMonths']]) {
    const value=result[section][years];
    if(raw[section]?.[months]===undefined&&typeof value==='number'&&Number.isFinite(value)&&!Number.isInteger(value)&&value>=(section==='longTermCare'?1:0)&&(section!=='longTermCare'||value<=10)) {
      const total=Math.round(value*12);
      result[section][years]=Math.floor(total/12);result[section][months]=total%12;
    }
  }
  // Imported and previously saved scenarios cannot change the site's fixed seed.
  result.seed = DEFAULT_SEED;
  result.simulationPathsCustomized = raw?.simulationPathsCustomized === true;
  return result;
}

// Scenario IDs key selection, deletion and results, so each must be a distinct string.
export function normalizeScenarios(list) {
  const scenarios=list.map(normalizeScenario);
  const ids=list.map(raw=>typeof raw.id==='string'||typeof raw.id==='number'?String(raw.id).trim():'');
  // A generated ID must not take a later plan's valid ID and redirect selection.
  const reserved=new Set(ids.filter(Boolean)),used=new Set();
  return scenarios.map((s, i) => {
    let id=ids[i];
    if (!id || used.has(id)) { let n = i + 1; do id = `plan-imported-${n++}`; while (reserved.has(id)||used.has(id)); }
    used.add(id); s.id = id;
    return s;
  });
}

export function applyProSimulationDefault(s) {
  if (s.simulationPathsCustomized || ![MIN_SIMULATION_PATHS,SMALL_SAMPLE_PATHS,FREE_SIMULATION_PATHS].includes(s.numberOfSimulations)) return false;
  s.numberOfSimulations = MAX_SIMULATION_PATHS;
  return true;
}

// Stored drafts may have unfinished assumptions, but must be safe to render.
export function validateScenarioStructure(s) {
  const wrongTypes=[];
  (function compare(expected,actual,path){for(const [key,value] of Object.entries(expected)){if(value===null)continue;const next=path?`${path}.${key}`:key;if(Array.isArray(value)){if(!Array.isArray(actual?.[key]))wrongTypes.push(next);}else if(typeof value==='object'){if(actual?.[key]&&typeof actual[key]==='object'&&!Array.isArray(actual[key]))compare(value,actual[key],next);else wrongTypes.push(next);}else if(typeof actual?.[key]!==typeof value)wrongTypes.push(next);}})(TYPE_TEMPLATE,s,'');
  if (wrongTypes.length) return [`These assumptions have the wrong type: ${wrongTypes.join(', ')}.`];
  return [...validateBudgetStructure(s.budget),...validateRothHistoryStructure(s.rothHistory),...validateRothHistoryStructure(s.spouseRothHistory),...validateEmployerRoth(s.employerRothAccounts)];
}

export function validateRothHistoryStructure(history) {
  if(!Array.isArray(history?.conversions))return ['Roth conversion history must be an array.'];
  const errors=[];
  for(const [i,lot] of history.conversions.entries()){
    if(!lot||typeof lot!=='object'||Array.isArray(lot)||!['taxYear','amount','taxableAmount'].every(key=>typeof lot[key]==='number'&&Number.isFinite(lot[key])))errors.push(`Roth conversion ${i+1} must have a finite tax year, remaining principal, and remaining taxable principal.`);
  }
  return errors;
}

// Backups can contain editable drafts that are not ready for calculation.
// Reject unsafe data and unsupported choices without requiring complete timing,
// account history or mutually consistent financial assumptions.
export function validateScenarioDraft(s) {
  const errors = [];
  const allNumbers = [];
  function collect(value) { if (typeof value === 'number') allNumbers.push(value); else if (value && typeof value === 'object') Object.values(value).forEach(collect); }
  collect(s);
  if (allNumbers.some(v=>!Number.isFinite(v))) errors.push('Financial and percentage assumptions must be finite numbers.');
  const structureErrors=validateScenarioStructure(s);
  if (structureErrors.length) return [...errors,...structureErrors];
  const h=s.household, sp=s.spending;
  if (!FILING_STATUSES.includes(h.filingStatus)) errors.push('Filing status must be Single, Married, or HeadOfHousehold.');
  if (!GENDERS.includes(h.gender) || (h.filingStatus === 'Married' && !GENDERS.includes(h.spouseGender))) errors.push('Longevity table must be Male or Female.');
  if (!SPENDING_PATH_MODELS.includes(sp.spendingPathModel)) errors.push('Spending path must be EmpiricalAgeDecline or Flat.');
  if(allNumbers.some(v=>Math.abs(v)>MAX_DOLLAR_AMOUNT))errors.push('Financial amounts exceed the supported dollar range.');
  const dateLabels={birthday:'Your birthday',retirementDate:h.alreadyRetired?'Actual retirement date':'Retirement date',spouseBirthday:'Spouse birthday',spouseRetirementDate:h.spouseAlreadyRetired?'Spouse actual retirement date':'Spouse retirement date',asOfDate:'Calculation date'};
  for(const [key,label] of Object.entries(dateLabels))if(h[key]&&!calendarDate(h[key]))errors.push(`${label} must be a valid calendar date or left blank.`);
  errors.push(...budgetDollarRangeErrors(s.budget));
  return errors;
}

export function validateScenario(s) {
  const errors=validateScenarioDraft(s);
  if(errors.length)return errors;
  const h=s.household, a=s.accounts, sp=s.spending,timeline=scenarioTimeline(s),primaryPension=s.guaranteedIncome.annualIncome>0;
  errors.push(...validateEmployerRoth(s.employerRothAccounts,{complete:true,today:h.asOfDate||localCalendarDate(),married:h.filingStatus==='Married'}));
  for(const a of s.employerRothAccounts){
    if(a.owner==='spouse'&&h.filingStatus!=='Married')continue;
    const spouse=a.owner==='spouse',date=forecastRetirementDate(s,spouse&&h.separatePeople),start=timeline.startDate||forecastRetirementDate(s),birthday=spouse?h.spouseBirthday:h.birthday,w=spouse&&h.separatePeople?s.spouseWithdrawal:s.withdrawalStrategy;
    if(a.rolloverDate&&!a.accessDate&&date&&a.rolloverDate<date)errors.push(a.name+': a pre-retirement rollover needs its permitted access date. Confirm an eligible in-service direct rollover with the plan.');
    if(a.separationDate&&birthday&&a.separationDate<birthday)errors.push(a.name+': employer separation cannot precede the owner’s birthday.');
    if(a.plannedConversionAmount===0)continue;
    if(start&&a.plannedConversionDate<start)errors.push(a.name+': planned in-plan conversion must be on or after the forecast starts.');
    if(w.seppEligible&&calendarMonthsBetween(birthday,date)/12<59.5&&a.plannedConversionDate>=date&&a.plannedConversionDate<addCalendarMonths(date,Math.max(60,Math.ceil(59.5*12-calendarMonthsBetween(birthday,date)))))errors.push(a.name+': an in-plan conversion cannot use SEPP-protected pre-tax savings.');
  }
  // Mortality and life-expectancy tables are indexed by whole years.
  if (![h.targetEndAge,...(usesCalendarDates(s)?[]:[h.currentAge,h.retirementAge,...(h.filingStatus==='Married'?[h.spouseCurrentAge]:[])])].every(Number.isInteger)) errors.push('Ages must be whole numbers.');
  if(usesCalendarDates(s)){
    const today=h.asOfDate||localCalendarDate();
    if(!calendarDate(today))errors.push('The calculation date must be a valid calendar date.');
    if(!calendarDate(h.birthday)||h.birthday>=today||timeline.currentAge<=0||timeline.currentAge>=119)errors.push('Your birthday must be a valid date before today and within the modeling age range.');
    if(h.alreadyRetired){
      if(h.retirementDate&&(!calendarDate(h.retirementDate)||h.retirementDate>today||h.retirementDate<h.birthday))errors.push('Actual retirement date must be between your birthday and today, or left blank.');
    }else if(!calendarDate(h.retirementDate)||h.retirementDate<today)errors.push('Retirement date must be a valid date today or later.');
    if(h.filingStatus==='Married'&&(!calendarDate(h.spouseBirthday)||h.spouseBirthday>=today||calendarMonthsBetween(h.spouseBirthday,today)<=0))errors.push('Spouse birthday must be a valid date before today.');
  }else if (h.currentAge <= 0) errors.push('Current age must be positive.');
  if (![h.retirementAgeMonths,...(primaryPension?[s.guaranteedIncome.startAgeMonths]:[]),...(s.longTermCare.enabled?[s.longTermCare.averageDurationMonths]:[])].every(v=>Number.isInteger(v)&&v>=0&&v<=11)) errors.push('Month fields must be whole numbers from 0 through 11.');
  if (![...(primaryPension?[s.guaranteedIncome.startAge]:[]),...(s.longTermCare.enabled?[s.longTermCare.averageDurationYears]:[])].every(Number.isInteger)) errors.push('Income start age and long-term care duration years must be whole numbers; use the month fields for extra months.');
  if (!usesCalendarDates(s)&&retirementAge(s) < h.currentAge) errors.push('Retirement age must be at least current age.');
  if (h.targetEndAge <= retirementAge(s) || h.targetEndAge > 119) errors.push('Maximum modeling age must be after retirement and at most 119.');
  if (h.filingStatus === 'Married' && ((!usesCalendarDates(s)&&h.spouseCurrentAge<=0) || timeline.spouseAtRet <= 0 || timeline.spouseAtRet >= h.targetEndAge)) errors.push('Spouse age at retirement must be below the maximum modeling age.');
  if (s.socialSecurity.claimAge < 62 || s.socialSecurity.claimAge > 70) errors.push('Social Security claim age must be 62–70.');
  if (h.filingStatus === 'Married' && (s.socialSecurity.spouseClaimAge < 60 || s.socialSecurity.spouseClaimAge > 70)) errors.push('Spouse claim age must be 60–70.');
  if (Object.values(a).some(v=>v<0) || sp.annualBaseSpending<0) errors.push('Balances and spending cannot be negative.');
  const dollars=[...Object.values(a),sp.annualBaseSpending,s.mortgage.monthlyPayment,s.mortgage.currentBalance,s.rent.monthlyRent,s.home.currentValue,s.home.annualTaxesAndInsurance,s.healthcare.preMedicareMonthlyPremium,s.socialSecurity.annualBenefitAt67,s.guaranteedIncome.annualIncome,s.longTermCare.annualCost];
  if (dollars.some(v=>Math.abs(v)>MAX_DOLLAR_AMOUNT)) errors.push('Financial amounts exceed the supported dollar range.');
  const rh=s.rothHistory;
  if(rh.contributionBasis<0||rh.contributionBasis>MAX_DOLLAR_AMOUNT)errors.push('Remaining Roth contributions must be a supported amount of 0 or more. Contributions may exceed the balance after investment losses.');
  if(!Number.isInteger(rh.firstContributionYear)||(rh.firstContributionYear!==0&&(rh.firstContributionYear<1998||rh.firstContributionYear>2026))||(rh.firstContributionYear===0&&(a.roth>0||rh.contributionBasis>0||rh.conversions.some(lot=>lot.amount>0))))errors.push('Enter the first Roth funding tax year from 1998 through 2026, or 0 if no Roth has been funded.');
  for(const [i,lot] of rh.conversions.entries()){
    if(!Number.isInteger(lot.taxYear)||lot.taxYear<1998||lot.taxYear>2026||lot.taxYear<rh.firstContributionYear||lot.amount<0||lot.amount>MAX_DOLLAR_AMOUNT||lot.taxableAmount<0||lot.taxableAmount>lot.amount)errors.push(`Roth conversion ${i+1}: enter a valid past tax year and remaining principal; taxable principal must be between 0 and the total principal.`);
  }
  if(rh.conversions.reduce((total,lot)=>total+lot.amount,rh.contributionBasis)>MAX_DOLLAR_AMOUNT)errors.push('Roth contribution and conversion principal exceed the supported dollar range.');
  if (sp.generalInflationMean < -.02 || sp.generalInflationMean > .15 || sp.generalInflationStdDev < 0 || sp.generalInflationStdDev > .3) errors.push('General inflation assumptions are outside the supported range.');
  if (sp.lowPortfolioSpendingReduction < 0 || sp.lowPortfolioSpendingReduction > 1) errors.push('Spending reduction must be between 0% and 100%.');
  if (s.healthcare.healthcareInflationMean < 0 || s.healthcare.healthcareInflationMean > .2 || s.healthcare.healthcareInflationStdDev < 0 || s.healthcare.healthcareInflationStdDev > .3) errors.push('Healthcare inflation assumptions are outside the supported range.');
  if (s.numberOfSimulations < 1 || s.numberOfSimulations > MAX_SIMULATION_PATHS || !Number.isInteger(s.numberOfSimulations)) errors.push('Simulation count must be between 1 and 10,000.');
  if (!Number.isSafeInteger(s.seed)) errors.push('Seed must be an integer.');
  if (![s.mortgage.yearsLeft,s.mortgage.monthsLeft].every(Number.isInteger) || s.mortgage.yearsLeft < 0 || s.mortgage.yearsLeft > 80 || s.mortgage.monthsLeft < 0 || s.mortgage.monthsLeft > 11 || s.mortgage.monthlyPayment < 0 || s.mortgage.currentBalance < 0) errors.push('Mortgage payment, balance, or term is invalid. Enter whole years and whole extra months (0–11).');
  if (s.mortgage.currentBalance > 0 && (s.mortgage.monthlyPayment <= 0 || s.mortgage.yearsLeft * 12 + s.mortgage.monthsLeft <= 0 || s.mortgage.monthlyPayment * (s.mortgage.yearsLeft * 12 + s.mortgage.monthsLeft) + .000001 < s.mortgage.currentBalance)) errors.push('Mortgage payments over the remaining term must cover the balance. Enter a positive payment and remaining term for an outstanding mortgage.');
  if (s.rent.monthlyRent < 0 || s.home.currentValue < 0 || s.home.annualTaxesAndInsurance < 0) errors.push('Housing amounts cannot be negative.');
  if (s.healthcare.preMedicareMonthlyPremium < 0) errors.push('Healthcare premium cannot be negative.');
  if (s.socialSecurity.annualBenefitAt67 < 0) errors.push('Social Security estimate cannot be negative.');
  if (s.guaranteedIncome.annualIncome < 0 || primaryPension&&(s.guaranteedIncome.startAge < 0 || s.guaranteedIncome.annualIncrease < -.02 || s.guaranteedIncome.annualIncrease > .15 || (h.filingStatus === 'Married' && (s.guaranteedIncome.survivorPercent < 0 || s.guaranteedIncome.survivorPercent > 1)))) errors.push('Guaranteed income assumptions are outside the supported range.');
  const market=s.market;
  if (market.preRetirementMeanReturn < -.20 || market.preRetirementMeanReturn > .25 || market.stockMeanReturn < -.20 || market.stockMeanReturn > .25 || market.bondMeanReturn < -.20 || market.bondMeanReturn > .20 || market.preRetirementStdDev < 0 || market.preRetirementStdDev > .60 || market.stockStdDev < 0 || market.stockStdDev > .60 || market.bondStdDev < 0 || market.bondStdDev > .40) errors.push('Market return assumptions are outside the supported range.');
  if (ALLOCATION_KEYS.some(k=>s.postRetirementAllocation[k]<0 || s.postRetirementAllocation[k]>1)) errors.push('Stock allocation must be between 0% and 100%.');
  if (s.rothConversion.enabled && !ROTH_CONVERSION_RATES.some(x=>Math.abs(x-s.rothConversion.marginalRateCap)<.0001)) errors.push('Roth conversion cap must be 10%, 12%, 22%, 24%, 32%, 35%, or 37%.');
  if (s.longTermCare.enabled && (s.longTermCare.annualCost < 0 || s.longTermCare.averageDurationYears < 1 || s.longTermCare.averageDurationYears + s.longTermCare.averageDurationMonths/12 > 10)) errors.push('Long-term care cost or duration is invalid.');
  if (s.withdrawalStrategy.drawdownTrigger < -.50 || s.withdrawalStrategy.drawdownTrigger > .25) errors.push('Cash drawdown trigger is outside the supported range.');
  // Keep inactive drafts, but validate only owners whose deposits are modeled.
  // Pooled couples share the primary retirement status.
  for(const [name,c,active] of [['You',s.contributions,!h.alreadyRetired],['Spouse',s.spouseContributions,h.filingStatus==='Married'&&!(h.separatePeople?h.spouseAlreadyRetired:h.alreadyRetired)]]){
    if(!active)continue;
    if(['pretax','roth','taxable','cash','employerPretax'].some(k=>c[k]<0||c[k]>MAX_DOLLAR_AMOUNT)||c.annualIncrease<-.02||c.annualIncrease>.15)errors.push(name+' savings contributions are outside the supported range.');
  }
  if(h.separatePeople){
    if(primaryRetirementAge(s)>=h.targetEndAge)errors.push('Your own retirement date must be before the maximum modeling age.');
    if(!usesCalendarDates(s))errors.push('Separate-person modeling requires calendar birthdays and retirement dates.');
    if(h.filingStatus==='Married'){
      const today=h.asOfDate||localCalendarDate();
      if(h.spouseAlreadyRetired){
        if(h.spouseRetirementDate&&(!calendarDate(h.spouseRetirementDate)||h.spouseRetirementDate>today||h.spouseRetirementDate<h.spouseBirthday))errors.push('Spouse actual retirement date must be between their birthday and today, or left blank.');
      }else if(!calendarDate(h.spouseRetirementDate)||h.spouseRetirementDate<today)errors.push('Choose a spouse retirement date today or later.');
      if(calendarMonthsBetween(h.spouseBirthday,forecastRetirementDate(s,true))/12>=h.targetEndAge)errors.push('Spouse own retirement date must be before the maximum modeling age.');
      if(Object.values(s.spouseAccounts).some(v=>v<0||v>MAX_DOLLAR_AMOUNT))errors.push('Spouse balances must be supported nonnegative amounts.');
      const si=s.spouseIncome;
      if(si.annualBenefitAt67<0||si.annualBenefitAt67>MAX_DOLLAR_AMOUNT||s.socialSecurity.spouseClaimAge<62||s.socialSecurity.spouseClaimAge>70)errors.push('Spouse own Social Security needs a nonnegative annual amount and claim age 62–70.');
      if(si.annualPension<0||si.annualPension>MAX_DOLLAR_AMOUNT||si.annualPension>0&&(!Number.isInteger(si.pensionStartAge)||si.pensionStartAge<0||!Number.isInteger(si.pensionStartAgeMonths)||si.pensionStartAgeMonths<0||si.pensionStartAgeMonths>11||si.annualIncrease<-.02||si.annualIncrease>.15||si.survivorPercent<0||si.survivorPercent>1))errors.push('Spouse pension assumptions are outside the supported range.');
      const rh=s.spouseRothHistory;
      if(rh.contributionBasis<0||rh.contributionBasis>MAX_DOLLAR_AMOUNT||!Number.isInteger(rh.firstContributionYear)||(rh.firstContributionYear!==0&&(rh.firstContributionYear<1998||rh.firstContributionYear>2026))||(rh.firstContributionYear===0&&(s.spouseAccounts.roth>0||rh.contributionBasis>0||rh.conversions.some(l=>l.amount>0))))errors.push('Review spouse Roth contribution basis and first funding year.');
      if(rh.conversions.reduce((n,l)=>n+l.amount,rh.contributionBasis)>MAX_DOLLAR_AMOUNT)errors.push('Spouse Roth principal exceeds the supported dollar range.');
      for(const lot of rh.conversions)if(!Number.isInteger(lot.taxYear)||lot.taxYear<1998||lot.taxYear>2026||lot.taxYear<rh.firstContributionYear||lot.amount<0||lot.amount>MAX_DOLLAR_AMOUNT||lot.taxableAmount<0||lot.taxableAmount>lot.amount)errors.push('Review spouse Roth conversion history.');
      // Individual plans retain these drafts, but have no working-person overlap.
      if(['primaryAnnualNet','spouseAnnualNet'].some(k=>s.workingIncome[k]<0||s.workingIncome[k]>MAX_DOLLAR_AMOUNT)||s.workingIncome.annualIncrease<-.02||s.workingIncome.annualIncrease>.15)errors.push('Take-home household support must be a supported nonnegative amount.');
    }
  }
  // Draft budget errors are shown in the budget editor; only applied spending feeds the simulation.
  return errors;
}

export const ANNUAL_BILLS = [['Property taxes','annualPropertyTaxes','propertyTaxes'],['Home insurance','annualHomeInsurance','homeInsurance'],['Auto insurance','annualAutoInsurance','autoInsurance']];
export const SEPARATE_COSTS = [['Mortgage payments','mortgage'],['Rent','rent'],['Healthcare premiums','healthcare']];
// Draft amounts can be unfinished, but imported structures must remain safe
// for the budget editor and reports before replacing any existing scenarios.
export function validateBudgetStructure(budget) {
  const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
  if(!object(budget)||!Array.isArray(budget.monthlyBudgets))return ['Budget monthlyBudgets must be an array.'];
  const errors=[];
  const number=(v,path)=>{if(v!==undefined&&(typeof v!=='number'||!Number.isFinite(v)))errors.push(`${path} must be a finite number.`);};
  number(budget.appliedAnnualHomeCosts,'Budget appliedAnnualHomeCosts');
  for(const [i,m] of budget.monthlyBudgets.entries()){
    const path=`Budget month ${i+1}`;
    if(!object(m)){errors.push(`${path} must be an object.`);continue;}
    if(typeof m.month!=='string')errors.push(`${path} must have a month string.`);
    if(m.includedPayments!==undefined&&(!object(m.includedPayments)||SEPARATE_COSTS.some(([,key])=>typeof m.includedPayments[key]!=='boolean')))errors.push(`${path} included payments must record a boolean for mortgage, rent and healthcare.`);
    number(m.cashAndAtmWithdrawals,`${path} cash withdrawals`);
    for(const key of ['checkingSavingsBills','creditCardBills']){
      if(m[key]===undefined)continue;
      if(!Array.isArray(m[key])){errors.push(`${path} ${key} must be an array.`);continue;}
      for(const bill of m[key]){
        if(!object(bill)){errors.push(`${path} ${key} entries must be objects.`);continue;}
        number(bill.monthlyAmount,`${path} ${key} monthlyAmount`);
      }
    }
    if(m.adjustments!==undefined){
      if(!object(m.adjustments))errors.push(`${path} adjustments must be an object.`);
      else for(const key of [...ANNUAL_BILLS.map(x=>x[2]),...SEPARATE_COSTS.map(x=>x[1])])number(m.adjustments[key],`${path} ${key}`);
    }
  }
  return errors;
}
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
  const structureErrors=validateBudgetStructure(budget);if(structureErrors.length)return structureErrors;
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
  errors.push(...budgetDollarRangeErrors(budget,d));
  // Refund-heavy months may be negative; evaluate deductions over the entire sample.
  if(d.totals.adjusted < -0.005)errors.push('Adjustments exceed spending across the selected months. Check for amounts deducted twice.');
  if(d.estimate < 0)errors.push('The retirement adjustment would make the annual estimate negative.');
  return errors;
}
function budgetDollarRangeErrors(budget,d=budgetBreakdown(budget)){
  const supported=v=>{const n=amount(v);return Number.isFinite(n)&&Math.abs(n)<=MAX_DOLLAR_AMOUNT;};
  const error=['Budget amounts and calculated totals exceed the supported dollar range.'];
  if([...ANNUAL_BILLS.map(([,key])=>budget[key]),budget.retirementAnnualAdjustment,budget.appliedAnnualHomeCosts].some(v=>!supported(v)))return error;
  const adjustments=[...ANNUAL_BILLS.map(x=>x[2]),...SEPARATE_COSTS.map(x=>x[1])];
  for(const m of budget.monthlyBudgets){
    if(!supported(m.cashAndAtmWithdrawals)||(m.checkingSavingsBills||[]).some(x=>!supported(x.monthlyAmount))||(m.creditCardBills||[]).some(x=>!supported(x.monthlyAmount))||adjustments.some(key=>!supported(m.adjustments?.[key]))||Object.values(budgetMonthTotals(m)).some(v=>!supported(v)))return error;
  }
  // Check intermediate sums as well as the estimate before applying anything.
  // Otherwise Infinity can become null when the scenario backup is serialized.
  return [...Object.values(d.totals),...Object.values(d).filter(v=>typeof v==='number')].some(v=>!supported(v))?error:[];
}
export function budgetEstimate(budget) { return budgetBreakdown(budget).estimate; }
// Draft edits leave the applied spending and the plan's home costs unchanged.
export function markBudgetEdited(budget) {
  budget.estimateNeedsReview=true;
}
export function applyBudgetEstimate(scenario,{replaceHomeCosts=scenario.budget.annualPropertyTaxes+scenario.budget.annualHomeInsurance>0}={}) {
  const errors=validateBudget(scenario.budget,{requireMonths:true});
  if(errors.length)throw new Error(errors.join(' '));
  const b=scenario.budget;
  scenario.spending.annualBaseSpending=budgetEstimate(b);
  // Empty optional bills do not erase separately entered home costs. An
  // explicit replacement can still record zero when those costs no longer apply.
  if(replaceHomeCosts)scenario.home.annualTaxesAndInsurance=amount(b.annualPropertyTaxes)+amount(b.annualHomeInsurance);
  delete b.appliedAnnualHomeCosts;
  b.isAppliedToAnnualBaseSpending=true;b.estimateNeedsReview=false;
  return scenario.spending.annualBaseSpending;
}

// Manually chosen spending replaces the applied worksheet. Home costs are a
// separate plan input, so the same spending gives the same result however it
// was entered, including in comparisons and spending-target candidates.
export function setAnnualBaseSpending(scenario,amount) {
  scenario.spending.annualBaseSpending=amount;
  scenario.budget.isAppliedToAnnualBaseSpending=false;
  scenario.budget.estimateNeedsReview=true;
}

export function scenarioWarnings(s) {
  const notes=[],h=s.household,total=Object.values(s.accounts).reduce((a,b)=>a+b,0)+employerRothTotal(s),years=retirementAge(s)-h.currentAge;
  if(retirementAge(s)<50)notes.push('Retiring before 50 creates a long drawdown period.');
  if(s.spending.annualBaseSpending/Math.max(1,total)>.07)notes.push('Annual base spending exceeds 7% of current assets.');
  if(s.spending.generalInflationMean<.015)notes.push('General inflation below 1.5% may understate spending pressure.');
  if(s.market.stockMeanReturn>.145||s.market.preRetirementMeanReturn>.145)notes.push('Expected return above 14.5% may make readiness look stronger.');
  const spouseAtRet=h.spouseCurrentAge+years;
  if((retirementAge(s)<65||(h.filingStatus==='Married'&&spouseAtRet<65))&&s.healthcare.preMedicareMonthlyPremium<=0)notes.push('A pre-Medicare adult has no healthcare premium entered.');
  if(!s.healthcare.includeMedicarePremiums)notes.push('Medicare premiums are excluded.');
  if(!s.longTermCare.enabled)notes.push('Long-term care risk is excluded.');
  if(s.rothHistory.needsReview)notes.push('Review Roth history: this older plan assumes its starting Roth value is remaining contributions, first funded in 2021, with no past conversions.');
  if(s.withdrawalStrategy.ruleOf55Eligible&&!ruleOf55Applies(s))notes.push('Rule of 55 is not applied: separation must be in or after the calendar year you turn 55. For a legacy age-only plan, it has no effect for a retirement age below 54.');
  if(s.household.datesNeedReview)notes.push('Dates were estimated from the saved ages. Check your birthday, retirement date, and spouse birthday in Household, then mark the dates reviewed.');
  if(s.home.annualTaxesAndInsurance>s.spending.annualBaseSpending)notes.push('Property tax and home insurance exceed annual base spending. Base spending should include them; the model removes that amount after a home sale.');
  if(s.socialSecurity.annualBenefitAt67<=0)notes.push('No Social Security benefit is entered.');
  if(s.numberOfSimulations<=FREE_SIMULATION_PATHS)notes.push('Small runs are only a preview. Use many more paths for serious comparisons and retirement decisions.');
  else if(s.numberOfSimulations<500)notes.push('Fewer than 500 paths can make comparisons unstable. Use more paths before relying on small differences.');
  return notes;
}
