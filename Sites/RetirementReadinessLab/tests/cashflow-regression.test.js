import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,validateScenario} from '../dist/model.js';
import {runOne,runSimulation} from '../dist/engine.js';
import {ordinaryIncomeTax,taxableSocialSecurity} from '../dist/tax.js';
import {retirementBenefitFactor} from '../dist/social-security.js';

function flatPlan(age=62,years=1){
  const s=baseScenario();
  Object.assign(s.household,{currentAge:age,retirementAge:age,targetEndAge:age+years});
  s.accounts={pretax:0,roth:100000,taxable:0,cash:0};
  Object.assign(s.spending,{annualBaseSpending:30000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  Object.assign(s.market,{preRetirementMeanReturn:0,preRetirementStdDev:0,stockMeanReturn:0,stockStdDev:0,bondMeanReturn:0,bondStdDev:0});
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});
  s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;
  return s;
}
const fullLife={nextDouble:()=>.999999,normal:()=>0};
function ending(s,rng=fullLife){assert.deepEqual(validateScenario(s),[]);return runOne(s,rng).yearEnd.at(-1);}
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<.01,`${actual} != ${expected}`);

test('a pretax-to-Roth transition below the annual deduction stays funded',()=>{
  const s=flatPlan();s.accounts={pretax:16100,roth:14600,taxable:0,cash:0};
  const result=runSimulation(s);assert.equal(result.successProbability,1);near(result.medianEndingBalance,700);
});

test('partial-year pretax depletion charges exactly the annual tax on actual income',()=>{
  for(const pretax of [10000,16100,30000,50000]){
    const s=flatPlan();s.accounts.pretax=pretax;s.spending.annualBaseSpending=60000;
    near(ending(s),100000+pretax-60000-ordinaryIncomeTax(pretax,'Single'));
  }
});

test('annual tax tracking resets at the year boundary',()=>{
  const s=flatPlan(62,2);s.accounts={pretax:60000,roth:5000,taxable:0,cash:0};
  // $30,000 of annual ordinary income has $1,420 tax under this table.
  s.spending.annualBaseSpending=28580;
  const path=runOne(s,fullLife);assert.equal(path.success,true);
  near(path.yearEnd[1],35000);near(path.yearEnd[2],5000);
});

test('Social Security taxation uses actual annual pretax draws and pension income',()=>{
  const s=flatPlan(67);s.accounts.pretax=20000;s.socialSecurity.annualBenefitAt67=30000;
  s.guaranteedIncome.annualIncome=10000;s.spending.annualBaseSpending=80000;
  const ordinary=30000,tax=ordinaryIncomeTax(ordinary+taxableSocialSecurity(ordinary,30000,'Single'),'Single',1,1,2026);
  near(ending(s),120000-80000+40000-tax);
});

test('early withdrawal penalty applies only to actual pretax draws',()=>{
  const s=flatPlan(55);s.accounts.pretax=10000;s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  near(ending(s),110000-30000-1000);
});

test('year-end Roth conversion charges only the additional annual liability',()=>{
  const s=flatPlan();s.accounts={pretax:100000,roth:0,taxable:0,cash:0};
  Object.assign(s.rothConversion,{enabled:true,marginalRateCap:.12});
  // Filling the 12% bracket produces $66,500 total ordinary income and $5,800 tax.
  near(ending(s),100000-30000-5800);
});

test('SEPP distributions enter annual income once without an early penalty',()=>{
  const s=flatPlan(55);s.accounts.pretax=100000;s.withdrawalStrategy.seppEligible=true;
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  // Spending exceeds the mandatory distribution, so total pretax draws simply
  // cover net spending plus ordinary tax; the SEPP portion has no 10% penalty.
  const withoutPenalty=structuredClone(s);withoutPenalty.withdrawalStrategy.applyEarlyWithdrawalPenalty=false;
  assert.ok(ending(s)<ending(withoutPenalty));
  const noSepp=structuredClone(s);noSepp.withdrawalStrategy.seppEligible=false;
  assert.ok(ending(s)>ending(noSepp));
});

test('mortgage payments continue during full-household long-term care',()=>{
  const s=flatPlan(65);s.accounts.roth=200000;s.spending.annualBaseSpending=0;
  Object.assign(s.mortgage,{monthlyPayment:1000,yearsLeft:1,currentBalance:12000});
  s.longTermCare.enabled=true;
  near(ending(s,{nextDouble:()=>0,normal:()=>0}),88000);
});

test('mortgage payments stop at the end of the loan term during care',()=>{
  const s=flatPlan(65);s.accounts.roth=200000;s.spending.annualBaseSpending=0;
  Object.assign(s.mortgage,{monthlyPayment:1000,monthsLeft:6,currentBalance:6000});s.longTermCare.enabled=true;
  near(ending(s,{nextDouble:()=>0,normal:()=>0}),94000);
});

test('the entered age-67 benefit is preserved across birth cohorts',()=>{
  for(const age of [67,70,75,85]){
    const s=flatPlan(age);s.spending.annualBaseSpending=40000;s.socialSecurity.annualBenefitAt67=30000;
    near(ending(s),90000);
  }
});

test('early and delayed claims adjust relative to the entered age-67 amount',()=>{
  for(const claimAge of [62,66,70]){
    const s=flatPlan(70);s.spending.annualBaseSpending=50000;
    Object.assign(s.socialSecurity,{annualBenefitAt67:30000,claimAge});
    const benefit=30000*retirementBenefitFactor(1956,claimAge*12)/retirementBenefitFactor(1956,804);
    near(ending(s),50000+benefit);
  }
});

test('spousal and survivor payments share the corrected primary insurance amount',()=>{
  const s=flatPlan(70,4);Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:67});
  s.accounts.roth=500000;s.spending.annualBaseSpending=60000;s.socialSecurity.annualBenefitAt67=30000;
  let draws=0;const path=runOne(s,{nextDouble:()=>draws++===0?0:.999999,normal:()=>0});
  const spouse=30000/retirementBenefitFactor(1956,804)*.5;
  near(path.yearEnd[1],500000-60000+30000+spouse);
  // After the worker dies, the survivor receives the worker's age-67 benefit.
  near(path.yearEnd[1]-path.yearEnd[2],60000*.84-30000);
});

test('initial Medicare lookback does not count Roth spending as ordinary income',()=>{
  const s=flatPlan(65,3);s.accounts.roth=1e6;s.spending.annualBaseSpending=150000;s.healthcare.includeMedicarePremiums=true;
  const premium=(202.90+38.99)*12,path=runOne(s,fullLife);
  for(let year=1;year<=3;year++)near(path.yearEnd[year],1e6-year*(150000+premium));
});

test('initial Medicare estimate caps taxable income at available pretax savings',()=>{
  const s=flatPlan(65);s.accounts.pretax=10000;s.accounts.roth=1e6;
  s.spending.annualBaseSpending=150000;s.healthcare.includeMedicarePremiums=true;
  near(ending(s),1010000-150000-(202.90+38.99)*12);
});

test('Medicare still charges a surcharge for high taxable pension income',()=>{
  const s=flatPlan(65);s.guaranteedIncome.annualIncome=150000;s.spending.annualBaseSpending=200000;
  s.healthcare.includeMedicarePremiums=true;
  const premium=(202.90+38.99+202.90+37.50)*12;
  near(ending(s),100000+150000-200000-premium-ordinaryIncomeTax(150000,'Single',1,1,2026));
});
