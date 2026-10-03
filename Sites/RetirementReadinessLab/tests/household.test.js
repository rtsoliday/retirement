import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {baseScenario,normalizeScenario,validateScenario,scenarioTimeline,ruleOf55Applies} from '../dist/model.js';
import {runOne,runSimulation,runSteadySimulation} from '../dist/engine.js';
import {PersonAccounts} from '../dist/person-accounts.js';
import {personSocialSecurity,personPensions} from '../dist/person-income.js';
import {monthlySavings} from '../dist/savings.js';
import {unknownInputPaths} from '../dist/ux-guidance.js';
import {requiredMinimumDistribution} from '../dist/distributions.js';
import {ordinaryIncomeTax,taxableSocialSecurity} from '../dist/tax.js';

const rng=()=>({normal:mean=>mean,nextDouble:()=>.5});
const close=(a,b)=>assert.ok(Math.abs(a-b)<.001,`${a} != ${b}`);
function plan(married=false){
  const s=baseScenario();Object.assign(s.household,{separatePeople:true,birthday:'1966-10-01',retirementDate:'2026-10-01',spouseBirthday:'1966-10-01',spouseRetirementDate:'2027-10-01',asOfDate:'2026-10-01',filingStatus:married?'Married':'Single'});
  s.accounts={pretax:100000,roth:0,taxable:0,cash:0};s.rothHistory={contributionBasis:0,firstContributionYear:0,conversions:[],needsReview:false};
  Object.assign(s.spending,{annualBaseSpending:0,generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0,spendingPathModel:'Flat'});
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;
  for(const key of Object.keys(s.market))s.market[key]=0;
  return s;
}
const run=(s,primary=61,spouse=61,extra={})=>runOne(s,rng(),{captureMonthlyDetails:true,captureTaxDetails:true,fixedDeathAges:{primary,spouse},...extra});

test('annual savings are monthly external deposits, employee plus employer, with a distinct increase',()=>{
  assert.deepEqual(monthlySavings({pretax:12000,employerPretax:6000,roth:6000,taxable:2400,cash:1200,annualIncrease:.1},0),{pretax:1500,roth:500,taxable:200,cash:100});
  close(monthlySavings({pretax:12000,employerPretax:0,roth:0,taxable:0,cash:0,annualIncrease:.1},12).pretax,1100);
});
test('pre-retirement deposits grow after investment returns and Roth deposits increase basis',()=>{
  const s=plan();s.household.retirementDate='2027-10-01';Object.assign(s.contributions,{pretax:12000,employerPretax:6000,roth:6000,taxable:2400,cash:1200});
  const p=run(s,61.1).monthlyDetails[0];close(p.pretax,118000);close(p.roth,6000);close(p.taxable,2400);assert.ok(p.cash>1200);close(p.ownerAccounts[0].contributionBasis,6000);assert.equal(p.ownerAccounts[0].firstContributionYear,2026);
  const growth=plan();growth.household.retirementDate='2027-10-01';growth.market.preRetirementMeanReturn=.12;growth.contributions.pretax=12000;
  const monthly=Math.pow(1.12,1/12);let expected=100000;for(let i=0;i<12;i++)expected=expected*monthly+1000;
  close(run(growth,61.1).monthlyDetails[0].pretax,expected);
});
test('legacy direct deposits preserve the shared stopping date and pooled Roth basis',()=>{
  const s=plan(true);s.household.separatePeople=false;s.household.retirementDate='2027-10-01';s.contributions.pretax=12000;s.spouseContributions.roth=6000;
  const p=run(s,61.1,61.1).monthlyDetails[0];close(p.pretax,112000);close(p.roth,6000);
});
test('each owner contributes until their own retirement; shared cash and brokerage deposits are counted once',()=>{
  const s=plan(true);s.contributions.pretax=60000;Object.assign(s.spouseContributions,{pretax:12000,employerPretax:6000,roth:6000,taxable:2400,cash:1200});
  const p=run(s,62,62);close(p.monthlyDetails[12].ownerAccounts[0].pretax,100000);close(p.monthlyDetails[12].ownerAccounts[1].pretax,18000);close(p.monthlyDetails[12].ownerAccounts[1].roth,6000);close(p.monthlyDetails[12].cashFlow.savingsContributions,2300);close(p.monthlyDetails[13].cashFlow.savingsContributions,0);close(p.monthlyDetails[24].ownerAccounts[1].pretax,18000);
});
test('first retirement can belong to spouse; Rule of 55 uses your own separation date',()=>{
  const s=plan(true);s.household.birthday='1972-10-01';s.household.retirementDate='2027-10-01';s.household.spouseRetirementDate='2026-10-01';s.withdrawalStrategy.ruleOf55Eligible=true;s.contributions.pretax=12000;s.spouseContributions.pretax=50000;
  assert.equal(scenarioTimeline(s).retirementAge,54);assert.equal(scenarioTimeline(s).startDate,'2026-10-01');assert.equal(ruleOf55Applies(s),true);
  const p=run(s,56,62);close(p.monthlyDetails[12].ownerAccounts[0].pretax,112000);close(p.monthlyDetails[12].ownerAccounts[1].pretax,0);
});
test('deposits stop at death before the later retirement date',()=>{
  const s=plan(true);s.household.spouseRetirementDate='2030-10-01';s.spouseContributions.pretax=12000;
  const p=run(s,62,60.5);close(p.monthlyDetails[6].ownerAccounts[1].pretax,6000);close(p.monthlyDetails[7].cashFlow.savingsContributions,0);
  close(p.monthlyDetails[13].ownerAccounts[0].pretax,106000);close(p.monthlyDetails[13].ownerAccounts[1].pretax,0);
});
test('net household support funds costs while working and is not taxed or saved twice',()=>{
  const s=plan(true);s.spending.annualBaseSpending=12000;s.workingIncome.spouseAnnualNet=12000;
  const p=run(s,62,62);close(p.monthlyDetails[1].cashFlow.workingSupport,1000);close(p.monthlyDetails[1].cashFlow.incomeTax,0);close(p.monthlyDetails[12].pretax,100000);close(p.monthlyDetails[13].cashFlow.workingSupport,0);assert.ok(p.monthlyDetails[13].pretax<100000);close(p.taxYears[0].ordinaryIncome,0);
});
test('same-month savings deposit can fund a cost, and retirement stops it',()=>{
  const s=plan(true);s.accounts.pretax=0;s.spending.annualBaseSpending=12000;s.spouseContributions.pretax=12000;
  const p=run(s,62,62,{taxesEnabled:false});assert.equal(p.failureAge,61);close(p.monthlyDetails[12].cashFlow.savingsContributions,1000);
});
test('both own Social Security benefits, only excess spousal benefit, and survivor maximum',()=>{
  const s=plan(true),t=scenarioTimeline(s);s.socialSecurity.annualBenefitAt67=36000;s.spouseIncome.annualBenefitAt67=12000;
  close(personSocialSecurity(s,t,[67,67],[true,true],[95,95],1),4500); // 3000 + 1000 own + 500 excess.
  close(personSocialSecurity(s,t,[67,67],[false,true],[67,95],1),3000);
  close(personSocialSecurity(s,t,[67,67],[true,false],[95,67],1),3000);
  s.spouseIncome.annualBenefitAt67=36000;close(personSocialSecurity(s,t,[67,67],[true,true],[95,95],1),6000);
  s.socialSecurity.claimAge=62;s.socialSecurity.spouseClaimAge=62;s.spouseIncome.annualBenefitAt67=0;
  close(personSocialSecurity(s,t,[62,62],[true,true],[95,95],1),3075); // 70% own plus 32.5% of other PIA.
});
test('spousal supplement waits for the other worker to claim, with each cohort and age offset',()=>{
  const s=plan(true);s.socialSecurity.annualBenefitAt67=36000;s.spouseIncome.annualBenefitAt67=0;s.socialSecurity.claimAge=70;s.socialSecurity.spouseClaimAge=67;
  const t=scenarioTimeline(s);close(personSocialSecurity(s,t,[67,67],[true,true],[95,95],1),0);close(personSocialSecurity(s,t,[70,70],[true,true],[95,95],1),5220);
});
test('separate pensions have separate start months, growth and survivor elections; no pre-start survivor',()=>{
  const s=plan(true);Object.assign(s.guaranteedIncome,{annualIncome:12000,startAge:60,startAgeMonths:6,survivorPercent:.5});Object.assign(s.spouseIncome,{annualPension:24000,pensionStartAge:60,pensionStartAgeMonths:0,survivorPercent:.25,annualIncrease:.1});
  close(personPensions(s,[60,60],[true,true],[95,95],0),2000);close(personPensions(s,[60.5,60.5],[true,true],[95,95],0),3000);
  close(personPensions(s,[61,61],[true,false],[95,61],12),1550);close(personPensions(s,[61,61],[false,true],[60.1,95],0),2000);
});
test('owned RMDs use owner birth cohort; young spouse balance creates no older-owner RMD',()=>{
  const s=plan(true);s.household.birthday='1951-10-01';s.spouseAccounts.pretax=200000;
  const b={...s.accounts},p=new PersonAccounts(s,b);p.configure(0,2026,[20,30],[true,true],()=>0);
  close(p.people[0].rmd,requiredMinimumDistribution(100000,75,1951));close(p.people[1].rmd,0);close(p.scheduled().total,p.people[0].rmd/12);
});
test('initial Medicare estimates stop counting an RMD after its owner exhausts pretax savings',()=>{
  const s=plan();Object.assign(s.household,{birthday:'1953-10-03',retirementDate:'2026-10-03',asOfDate:'2026-10-03',targetEndAge:74});
  s.accounts.pretax=10000;s.accounts.roth=500000;s.rothHistory.firstContributionYear=2024;
  s.spending.annualBaseSpending=92000;s.healthcare.includeMedicarePremiums=true;
  const result=run(s,74),months=result.monthlyDetails.slice(1);
  assert.equal(months[1].ownerAccounts[0].pretax,0);
  // Nonqualified Roth earnings approach the first surcharge threshold. The
  // exhausted account cannot add its year-start RMD to the income estimate.
  for(const month of months)close(month.cashFlow.expenses,92000/12+241.89);
  close(result.taxYears[0].requiredMinimumDistribution,requiredMinimumDistribution(10000,73,1953));
  assert.ok(result.taxYears[0].ordinaryIncome<109000);
  const pooled=structuredClone(s);pooled.household.separatePeople=false;
  close(result.yearEnd.at(-1),run(pooled,74).yearEnd.at(-1));
});
test('SEPP protects only its owner and stops at death; the other owner remains accessible',()=>{
  const s=plan(true);s.household.birthday='1976-10-01';s.household.spouseBirthday='1976-10-01';s.household.spouseRetirementDate=s.household.retirementDate;s.withdrawalStrategy.seppEligible=true;s.spouseAccounts.pretax=20000;
  const p=new PersonAccounts(s,{...s.accounts});p.configure(0,2026,[0,20],[true,true],()=>6000);
  assert.equal(p.people[0].protected,true);const q=p.quote(10000);close(q.draws[0].pretax,0);close(q.draws[1].pretax,10000);close(q.penalties,1000);
  p.configure(6,2026,[0,20],[false,true],()=>6000);assert.equal(p.people[0].protected,false);close(p.quote(10000).penalties,0);
});
test('owner Roth qualification, basis and inherited histories remain distinct then merge after death year',()=>{
  const s=plan(true);s.household.birthday='1976-10-01';s.household.spouseBirthday='1976-10-01';s.accounts.pretax=0;s.accounts.roth=20000;s.rothHistory={contributionBasis:10000,firstContributionYear:2021,conversions:[{taxYear:2025,amount:5000,taxableAmount:5000}],needsReview:false};s.spouseAccounts.roth=20000;s.spouseRothHistory={contributionBasis:5000,firstContributionYear:2024,conversions:[],needsReview:false};
  const p=new PersonAccounts(s,{...s.accounts});p.configure(0,2026,[0,20],[true,true],()=>0);
  const q=p.quote(30000);close(q.rothTaxableEarnings,10000);close(q.penalties,1500); // recent conversion + 10k earnings, separate basis.
  p.configure(12,2027,[0,20],[false,true],()=>0);close(p.people[1].roth,40000);close(p.people[1].ledger.openingBalance,15000);assert.equal(p.people[1].ledger.firstContributionYear,2021);assert.equal(p.people[1].ledger.lots.length,1);
});
test('steady illustration uses the same contribution/ownership inputs with fixed deaths and no volatility',()=>{
  const s=plan(true);s.spouseContributions.pretax=12000;s.market.stockStdDev=.3;s.longTermCare.enabled=true;
  const p=runSteadySimulation(s);assert.equal(p.primaryDeathAge,95);assert.equal(p.spouseDeathAge,95);close(p.monthlyDetails[12].ownerAccounts[1].pretax,12000);close(p.monthlyDetails[13].cashFlow.savingsContributions,0);
});
test('owner conversions transfer savings once and fund the actual liability, including conversion-tax recapture',()=>{
  const s=plan(true);s.accounts.pretax=50000;s.spouseAccounts.pretax=50000;s.rothConversion.enabled=true;s.rothConversion.marginalRateCap=.22;
  const p=run(s,61,61),last=p.monthlyDetails.at(-1),f=last.cashFlow,y=p.taxYears.at(-1);
  assert.ok(f.conversionAmount>0);close(last.portfolio,100000-f.conversionTax);close(y.tax,ordinaryIncomeTax(y.ordinaryIncome+taxableSocialSecurity(y.ordinaryIncome,y.socialSecurity,y.status),y.status,1,0,2026));
  close(y.conversions,f.conversionAmount);close(last.roth,last.ownerAccounts[0].roth+last.ownerAccounts[1].roth);assert.ok(last.ownerAccounts.every(a=>a.firstContributionYear===2026));
});
test('staggered pre-Medicare premiums begin only at that person’s own retirement',()=>{
  const s=plan(true);s.healthcare.preMedicareMonthlyPremium=100;
  const p=run(s,62,62);close(p.monthlyDetails[1].cashFlow.expenses,100);close(p.monthlyDetails[13].cashFlow.expenses,200);
});
test('seeded expanded runs remain finite through death, care, penalty and conversion combinations',()=>{
  for(const spouseFirst of [false,true])for(const conversions of [false,true])for(const sepp of [false,true]){
    const s=plan(true);s.household.birthday='1971-10-01';s.household.spouseBirthday='1973-10-01';s.household.retirementDate=spouseFirst?'2028-10-01':'2026-10-01';s.household.spouseRetirementDate=spouseFirst?'2026-10-01':'2028-10-01';
    s.accounts.pretax=500000;s.accounts.cash=50000;s.spouseAccounts.pretax=200000;s.socialSecurity.annualBenefitAt67=24000;s.spouseIncome.annualBenefitAt67=12000;s.spending.annualBaseSpending=36000;s.longTermCare.enabled=true;
    s.market.stockMeanReturn=.06;s.market.stockStdDev=.15;s.market.bondMeanReturn=.03;s.market.bondStdDev=.05;s.contributions.roth=6000;s.spouseContributions.pretax=12000;s.rothConversion.enabled=conversions;s.withdrawalStrategy.seppEligible=sepp;s.workingIncome.spouseAnnualNet=12000;
    const r=runSimulation(s);assert.ok(Number.isFinite(r.medianEndingBalance));assert.equal(r.provenance.engineVersion,'2026.10-separate-people-medicare-rmd-v2');assert.ok(r.steadySimulation.monthlyDetails.every(p=>Number.isFinite(p.netAssets)&&p.ownerAccounts.every(a=>a.pretax>=-.001&&a.roth>=-.001)));
  }
});
test('new contribution/ownership fields validate amounts, dates, types and past Roth history',()=>{
  assert.deepEqual(validateScenario(plan(true)),[]);
  for(const edit of [s=>s.contributions.pretax=-1,s=>s.spouseContributions.roth='500',s=>s.household.spouseRetirementDate='',s=>s.household.retirementDate='2099-10-01',s=>{s.spouseIncome.annualPension=100;s.spouseIncome.pensionStartAgeMonths=12;},s=>s.spouseAccounts.roth=500,s=>s.spouseRothHistory.conversions=[null]]){
    const s=plan(true);edit(s);assert.ok(validateScenario(s).length);
  }
});
test('unknown active amounts block, zero is distinct, and inactive spouse inputs do not block individuals',()=>{
  const s=plan(true),sources={[s.id]:{'contributions.pretax':'Unknown','spouseAccounts.pretax':'Unknown','workingIncome.spouseAnnualNet':'Unknown'}};
  assert.equal(unknownInputPaths(s,sources).length,3);sources[s.id]['contributions.pretax']='Entered';s.contributions.pretax=0;assert.equal(unknownInputPaths(s,sources).length,2);
  s.household.filingStatus='Single';assert.deepEqual(unknownInputPaths(s,sources),[]);s.household.filingStatus='Married';s.household.separatePeople=false;assert.deepEqual(unknownInputPaths(s,sources),[]);
});
// The fixtures predate today's-dollar summaries, evenly spread preview
// lifespans and the pooled pension correction's cache version. These scenarios
// have no pre-start survivor pension, so their calculated results remain identical.
test('pre-change complete seeded results remain identical after normalization of old backups',()=>{
  const fixtures=JSON.parse(readFileSync(new URL('./fixtures/pre-household-results.json',import.meta.url)));
  for(const f of fixtures){const s=normalizeScenario(f.scenario);assert.equal(s.household.separatePeople,false);const r=runSimulation(s,undefined,{stratifyPreviewLifespans:false});delete r.generatedAtEpochMillis;delete r.todayDollars;
    assert.match(r.provenance.engineVersion,/-survivor-pension-v2$/);
    r.provenance.engineVersion=r.provenance.engineVersion.replace(/-survivor-pension-v2$/,'');
    assert.equal(createHash('sha256').update(JSON.stringify(r)).digest('hex'),f.sha256,f.name);
  }
});
