import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,normalizeScenario,validateScenario,scenarioWarnings} from '../dist/model.js';
import {runOne,JavaRandom,medicarePremium} from '../dist/engine.js';
import {ordinaryIncomeTax,taxableSocialSecurity} from '../dist/tax.js';
import {RothConversionLedger} from '../dist/roth-conversions.js';
import {buildFundingSurvival} from '../dist/chart-data.js';

const fullLife={nextDouble:()=>.999999,normal:mean=>mean};
const near=(a,b)=>assert.ok(Math.abs(a-b)<.01,`${a} != ${b}`);
function flat(age,years=1){
  const s=baseScenario();Object.assign(s.household,{currentAge:age,retirementAge:age,targetEndAge:age+years});
  s.accounts={pretax:0,roth:100000,taxable:0,cash:0};
  s.rothHistory={contributionBasis:20000,firstContributionYear:2021,conversions:[],needsReview:false};
  Object.assign(s.spending,{annualBaseSpending:50000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  for(const key of Object.keys(s.market))s.market[key]=0;
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});
  s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;return s;
}
function path(s,rng=fullLife,options={}){assert.deepEqual(validateScenario(s),[]);return runOne(s,rng,{captureTaxDetails:true,captureMonthlyBalances:true,...options});}
// Independent annual net-cash equation: no return, other income, or conversions.
function annualGross(need,basis,penalty=0,seniors=0){
  let low=need,high=need*2;
  for(let i=0;i<60;i++){const mid=(low+high)/2,earnings=Math.max(0,mid-basis);if(mid-ordinaryIncomeTax(earnings,'Single',1,seniors,2026)-penalty*earnings>=need)high=mid;else low=mid;}
  return high;
}

test('Roth withdrawals include nonqualified earnings income tax and gross up the earnings penalty',()=>{
  const s=flat(55);s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  const p=path(s),gross=annualGross(50000,20000,.1);
  near(p.yearEnd.at(-1),100000-gross);near(p.taxYears[0].ordinaryIncome,gross-20000);
  near(p.taxYears[0].tax,ordinaryIncomeTax(gross-20000,'Single'));
  assert.equal(p.taxYears[0].pretaxDistributions,0);
});

test('turning off penalties still taxes nonqualified Roth earnings',()=>{
  const s=flat(55);s.withdrawalStrategy.applyEarlyWithdrawalPenalty=false;
  const p=path(s),gross=annualGross(50000,20000);
  near(p.yearEnd.at(-1),100000-gross);near(p.taxYears[0].ordinaryIncome,gross-20000);
});

test('the reviewed early Roth example cannot fund spending from the full untaxed balance',()=>{
  const s=flat(50,6);s.household.retirementAge=55;s.accounts.roth=50000;s.rothHistory.contributionBasis=50000;
  s.market.preRetirementMeanReturn=2**(1/5)-1;s.spending.annualBaseSpending=100000;
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  s.withdrawalStrategy.ruleOf55Eligible=true;
  const p=path(s);near(p.yearEnd[0],100000);assert.equal(p.success,false);assert.ok(p.failureAge<56);
  near(p.taxYears[0].ordinaryIncome,50000);near(p.taxYears[0].tax,3820);
});

test('after 59.5 a young Roth still owes earnings income tax but no age penalty',()=>{
  const s=flat(60);s.rothHistory.firstContributionYear=2024;s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  near(path(s).yearEnd.at(-1),100000-annualGross(50000,20000));
  s.rothHistory.firstContributionYear=2021;near(path(s).yearEnd.at(-1),50000);
});

test('regular contributions are available without waiting five years',()=>{
  const s=flat(55);s.rothHistory.firstContributionYear=2026;s.spending.annualBaseSpending=20000;
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;const p=path(s);
  near(p.yearEnd.at(-1),80000);near(p.taxYears[0].ordinaryIncome,0);
});

test('losses preserve contribution basis even when it exceeds the account value',()=>{
  const s=flat(55);s.accounts.roth=15000;s.rothHistory.contributionBasis=20000;s.spending.annualBaseSpending=15000;
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;const p=path(s);
  assert.equal(p.success,true);near(p.yearEnd.at(-1),0);near(p.taxYears[0].tax,0);
});

test('past taxable conversions have separate clocks and are not taxed as ordinary income twice',()=>{
  const s=flat(55);s.accounts.roth=30000;s.rothHistory.contributionBasis=0;s.spending.annualBaseSpending=18000;
  s.rothHistory.conversions=[{taxYear:2025,amount:30000,taxableAmount:30000}];
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  const p=path(s);near(p.yearEnd.at(-1),10000);near(p.taxYears[0].ordinaryIncome,0);
  s.rothHistory.conversions[0].taxYear=2021;near(path(s).yearEnd.at(-1),12000);
  s.household.currentAge=60;s.household.retirementAge=60;s.household.targetEndAge=61;
  s.rothHistory.conversions[0].taxYear=2025;near(path(s).yearEnd.at(-1),12000);
});

test('same-year conversion ordering groups taxable principal ahead of nontaxable principal',()=>{
  const ledger=new RothConversionLedger(1000,2020,[{taxYear:2025,amount:5000,taxableAmount:0},{taxYear:2025,amount:5000,taxableAmount:5000},{taxYear:2023,amount:2000,taxableAmount:1000}]);
  const first=ledger.distribution(4500,20000,2026,{consume:true});
  assert.deepEqual(first,{contributions:1000,conversions:3500,earnings:0,taxableEarnings:0,penaltyBase:2500});
  assert.equal(ledger.distribution(4000,15500,2026).penaltyBase,3500);
  assert.equal(ledger.distribution(20000,15500,2030).penaltyBase,7000);
});

test('a partially taxable past conversion charges recapture only on its remaining taxable principal',()=>{
  const s=flat(55);s.accounts.roth=30000;s.rothHistory.contributionBasis=0;s.spending.annualBaseSpending=27000;
  s.rothHistory.conversions=[{taxYear:2025,amount:30000,taxableAmount:20000}];
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  const p=path(s);near(p.yearEnd.at(-1),1000);near(p.taxYears[0].ordinaryIncome,0);
});

test('nonqualified Roth earnings enter Social Security taxation',()=>{
  const s=flat(62);s.accounts.roth=200000;s.rothHistory.firstContributionYear=2024;
  s.spending.annualBaseSpending=80000;Object.assign(s.socialSecurity,{annualBenefitAt67:30000,claimAge:62});
  const p=path(s),year=p.taxYears[0],taxableSS=taxableSocialSecurity(year.ordinaryIncome,year.socialSecurity,'Single');
  assert.ok(taxableSS>0);near(year.tax,ordinaryIncomeTax(year.ordinaryIncome+taxableSS,'Single'));
  near(p.yearEnd.at(-1),200000+year.socialSecurity-80000-year.tax);
});

test('nonqualified Roth earnings enter initial Medicare estimates and the two-year lookback',()=>{
  const s=flat(65,3);s.accounts.roth=1000000;s.rothHistory.contributionBasis=0;s.rothHistory.firstContributionYear=2026;
  s.spending.annualBaseSpending=130000;s.healthcare.includeMedicarePremiums=true;
  const p=path(s);assert.equal(p.success,true);
  for(const i of [0,2]){
    const year=p.taxYears[i],income=p.taxYears[0].ordinaryIncome;
    const premiums=12*medicarePremium(income,'Single',1,1,1,year.taxYear,1);
    assert.ok(premiums>12*241.89);
    near(year.ordinaryIncome-year.tax-130000,premiums);
  }
});

test('account qualification and conversion recapture use different five-year clocks',()=>{
  const ledger=new RothConversionLedger(0,2024,[{taxYear:2025,amount:10000,taxableAmount:10000}]);
  assert.equal(ledger.isQualified(2028,60),false);assert.equal(ledger.isQualified(2029,60),true);
  assert.equal(ledger.isQualified(2029,59.49),false);
  assert.equal(ledger.distribution(15000,20000,2029).penaltyBase,15000);
  assert.equal(ledger.distribution(15000,20000,2030).penaltyBase,5000);
  assert.equal(ledger.distribution(15000,20000,2029,{qualified:true}).taxableEarnings,0);
  const empty=new RothConversionLedger();empty.add(10000,2026);
  assert.equal(empty.firstContributionYear,2026);assert.equal(empty.isQualified(2030,65),false);assert.equal(empty.isQualified(2031,65),true);
});

test('a future conversion into a previously unfunded Roth starts the earnings clock',()=>{
  const s=flat(65,2);s.accounts={pretax:50000,roth:0,taxable:0,cash:0};
  s.rothHistory={contributionBasis:0,firstContributionYear:0,conversions:[],needsReview:false};
  s.rothConversion.enabled=true;s.rothConversion.marginalRateCap=.37;s.spending.annualBaseSpending=0;
  Object.assign(s.longTermCare,{enabled:true,annualCost:94000,averageDurationYears:1});
  s.market.stockStdDev=.1;for(const key of Object.keys(s.postRetirementAllocation))s.postRetirementAllocation[key]=1;
  const rng=()=>{let returns=0,deaths=0;return{nextDouble:()=>deaths++===0?.999999:0,normal:(mean,std)=>std>0?(returns++===12?Math.log(2):0):mean};};
  const p=path(s,rng());assert.equal(p.taxYears[0].conversions,50000);
  // The conversion leaves $47,146 principal after its senior-aware tax. A
  // doubling creates the same amount of earnings; tax makes care unaffordable.
  assert.equal(p.success,false);near(p.taxYears[1].ordinaryIncome,47146);
  near(p.taxYears[1].tax,ordinaryIncomeTax(47146,'Single',1,1,2027));
  s.rothHistory.firstContributionYear=2021;const mature=path(s,rng());
  assert.equal(mature.success,true);near(mature.yearEnd.at(-1),292);
});

test('annual senior eligibility prevents failure before a midyear 65th birthday',()=>{
  const s=flat(64,2);s.household.retirementAgeMonths=6;s.accounts={pretax:24500,roth:0,taxable:0,cash:0};
  s.spending.annualBaseSpending=48000;Object.assign(s.guaranteedIncome,{annualIncome:72000,startAge:65});
  const p=path(s);assert.equal(p.success,true);near(p.yearEnd.at(-1),19405.922967277584);
});

test('a spouse turning 65 within the year receives the full annual senior deduction',()=>{
  const s=flat(70);Object.assign(s.household,{filingStatus:'Married',retirementAgeMonths:6,spouseCurrentAge:64,targetEndAge:72});
  s.accounts={pretax:0,roth:200000,taxable:0,cash:0};Object.assign(s.guaranteedIncome,{annualIncome:60000,startAge:0});
  const p=path(s);near(p.taxYears[0].tax,ordinaryIncomeTax(p.taxYears[0].ordinaryIncome,'Married',1,2,2026));
});

test('a spouse dying before 65 does not receive the age deduction later in the year',()=>{
  const s=flat(70,2);Object.assign(s.household,{filingStatus:'Married',retirementAgeMonths:6,spouseCurrentAge:63});
  s.accounts.roth=200000;Object.assign(s.guaranteedIncome,{annualIncome:60000,startAge:0});
  let calls=0;const p=path(s,{nextDouble:()=>calls++===120-70?0:.999999,normal:mean=>mean});
  near(p.taxYears[0].tax,ordinaryIncomeTax(p.taxYears[0].ordinaryIncome,'Married',1,1,2026));
});

test('confirmed Rule of 55 eligibility applies before the birthday in the separation year',()=>{
  const s=flat(54,2);s.household.retirementAgeMonths=6;s.accounts={pretax:16500,roth:0,taxable:0,cash:0};
  s.spending.annualBaseSpending=30000;Object.assign(s.guaranteedIncome,{annualIncome:48000,startAge:55});
  Object.assign(s.withdrawalStrategy,{applyEarlyWithdrawalPenalty:true,ruleOf55Eligible:true});
  const eligible=path(s);assert.equal(eligible.success,true);near(eligible.yearEnd.at(-1),16325.924917288965);
  s.withdrawalStrategy.ruleOf55Eligible=false;const ineligible=path(s);assert.equal(ineligible.success,false);near(ineligible.failureAge,54+11/12);
});

test('a Rule of 55 declaration has no effect for a retirement before 54 and is flagged',()=>{
  const plan=(age,months=0,declared=false)=>{
    const s=flat(age,2);s.household.retirementAgeMonths=months;s.accounts={pretax:120000,roth:0,taxable:0,cash:0};
    s.rothHistory={contributionBasis:0,firstContributionYear:0,conversions:[],needsReview:false};s.spending.annualBaseSpending=30000;
    Object.assign(s.withdrawalStrategy,{applyEarlyWithdrawalPenalty:true,ruleOf55Eligible:declared});return s;
  };
  for(const [age,months,applies] of [[50,0,false],[53,11,false],[54,0,true],[56,0,true]]){
    const declared=plan(age,months,true),withRule=path(declared).yearEnd.at(-1),without=path(plan(age,months)).yearEnd.at(-1);
    if(applies)assert.ok(withRule>without+1000,`${age} years ${months} months`);else near(withRule,without);
    assert.equal(scenarioWarnings(declared).some(note=>/Rule of 55 is not applied/.test(note)),!applies);
  }
});

test('the modeling age limit does not move care or force a home sale before actual death',()=>{
  const s=flat(65,5);s.accounts.roth=250000;s.spending.annualBaseSpending=0;s.home.currentValue=200000;
  Object.assign(s.longTermCare,{enabled:true,annualCost:100000,averageDurationYears:3});
  Object.assign(s.guaranteedIncome,{annualIncome:140000,startAge:70});
  const rng=()=>{let calls=0;return{nextDouble:()=>calls++<9?.999999:0,normal:mean=>mean};};
  const short=path(s,rng());assert.equal(short.deathAge,75);assert.equal(short.observationEndAge,70);assert.equal(short.censored,true);
  assert.equal(short.success,true);near(short.yearEnd.at(-1),250000);
  s.household.targetEndAge=75;const long=path(s,rng());
  assert.equal(long.deathAge,75);assert.equal(long.censored,false);
  assert.deepEqual(short.monthlyBalances,long.monthlyBalances.slice(0,60));
  assert.deepEqual(short.taxYears,long.taxYears.slice(0,5));assert.equal(long.success,true);
  const curve=buildFundingSurvival([short],65);assert.equal(curve.at(-1).age,70);assert.equal(curve.at(-1).aliveShare,1);
});

test('seeded mortality is unchanged when the observation cutoff changes',()=>{
  const s=flat(65,5),short=path(s,new JavaRandom(12345));s.household.targetEndAge=90;
  const long=path(s,new JavaRandom(12345));assert.equal(short.deathAge,long.deathAge);
});

test('legacy Roth history migration preserves balances, marks assumptions, and round-trips',()=>{
  const old=flat(55);delete old.rothHistory;
  const migrated=normalizeScenario(old);assert.equal(migrated.accounts.roth,100000);
  assert.deepEqual(migrated.rothHistory,{contributionBasis:100000,firstContributionYear:2021,conversions:[],needsReview:true});
  assert.ok(scenarioWarnings(migrated).some(note=>note.startsWith('Review Roth history:')));
  const entered=flat(55);entered.rothHistory.conversions=[{taxYear:2022,amount:10000,taxableAmount:7000}];
  assert.deepEqual(normalizeScenario(JSON.parse(JSON.stringify(entered))),entered);
  const android=normalizeScenario({currentAge:55,retirementAge:60,rothBalance:12345});
  assert.equal(android.accounts.roth,12345);assert.equal(android.rothHistory.contributionBasis,12345);assert.equal(android.rothHistory.needsReview,true);
});

test('malformed Roth history and unsupported amounts are rejected before simulation',()=>{
  for(const edit of [s=>s.rothHistory=null,s=>s.rothHistory.conversions=null,s=>s.rothHistory.conversions=[null],s=>s.rothHistory.conversions=[{taxYear:2024,amount:'1000',taxableAmount:1000}]]){
    const s=flat(55);edit(s);assert.throws(()=>{const n=normalizeScenario(s);const errors=validateScenario(n);if(errors.length)throw Error(errors.join(' '));});
  }
  for(const edit of [s=>s.rothHistory.contributionBasis=-1,s=>s.rothHistory.contributionBasis=1e308,s=>s.rothHistory.firstContributionYear=0,s=>s.rothHistory.firstContributionYear=2027,s=>s.rothHistory.firstContributionYear=2024.5,s=>s.rothHistory.conversions=[{taxYear:2025,amount:1000,taxableAmount:1001}],s=>s.rothHistory.conversions=[{taxYear:2020,amount:1000,taxableAmount:1000}]]){
    const s=flat(55);edit(s);assert.ok(validateScenario(s).length);
  }
});
