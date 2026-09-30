import test from 'node:test';
import assert from 'node:assert/strict';
import {sampleDeathAge,runOne,JavaRandom} from '../dist/engine.js';
import {baseScenario,validateScenario} from '../dist/model.js';
import {maleMortality,femaleMortality} from '../dist/mortality.js';

test('mortality spreads deaths across months while preserving annual probabilities',()=>{
  // Stratified uniform draws make this distribution check deterministic. The
  // expected month probabilities come from the annual table's survival curve.
  const count=100000;
  for(const [gender,age,table] of [['Male',65,maleMortality],['Female',90,femaleMortality]]){
    const q=table[age],months=Array(12).fill(0);let deaths=0;
    for(let i=0;i<count;i++){
      let draws=0;const death=sampleDeathAge(gender,age,age+2,{nextDouble:()=>draws++===0?(i+.5)/count:1});
      const elapsed=Math.round((death-age)*12);
      assert.ok(elapsed>=1&&elapsed<=24);
      if(elapsed<=12){months[elapsed-1]++;deaths++;assert.equal(draws,1);}
      else assert.equal(draws,2);
    }
    assert.ok(Math.abs(deaths/count-q)<=1/count);
    for(let month=1;month<=12;month++){
      const expected=(1-q)**((month-1)/12)-(1-q)**(month/12);
      assert.ok(Math.abs(months[month-1]/count-expected)<=2/count,`${gender} age ${age}, month ${month}`);
    }
  }
});

test('partial-year mortality uses only the remaining months before the birthday',()=>{
  const start=65.5,q=maleMortality[65],probability=1-(1-q)**.5,count=100000,months=Array(6).fill(0);
  for(let i=0;i<count;i++){
    let draws=0;const death=sampleDeathAge('Male',start,67,{nextDouble:()=>draws++===0?(i+.5)/count:1});
    const elapsed=Math.round((death-start)*12);
    if(death<=66)months[elapsed-1]++;
    else assert.equal(death,67);
  }
  assert.ok(Math.abs(months.reduce((sum,n)=>sum+n,0)/count-probability)<=1/count);
  for(let month=1;month<=6;month++){
    const expected=(1-q)**((month-1)/12)-(1-q)**(month/12);
    assert.ok(Math.abs(months[month-1]/count-expected)<=2/count,`partial-year month ${month}`);
  }
  assert.equal(sampleDeathAge('Male',65+11/12,66,{nextDouble:()=>0}),66);
  assert.equal(sampleDeathAge('Male',120,120,{nextDouble:()=>{throw Error('No interval to sample');}}),120);
});

test('death-month sampling retains one random draw per selected age year',()=>{
  for(const gender of ['Male','Female'])for(const start of [54,65.5,85+11/12])for(let seed=0;seed<50;seed++){
    const source=new JavaRandom(seed);let draws=0;
    const death=sampleDeathAge(gender,start,120,{nextDouble:()=>{draws++;return source.nextDouble();}});
    const expected=Math.ceil(death)-Math.floor(start);
    assert.equal(draws,expected);
    const reference=new JavaRandom(seed);for(let i=0;i<expected;i++)reference.nextDouble();
    assert.equal(source.nextDouble(),reference.nextDouble());
  }
});

function flatPlan(age=65,years=3){
  const s=baseScenario();Object.assign(s.household,{currentAge:age,retirementAge:age,targetEndAge:age+years});
  s.accounts={pretax:0,roth:100000,taxable:0,cash:0};s.rothHistory={contributionBasis:100000,firstContributionYear:2021,conversions:[],needsReview:false};
  Object.assign(s.spending,{annualBaseSpending:12000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  for(const key of Object.keys(s.market))s.market[key]=0;
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});
  s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;
  return s;
}
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<.01,`${actual} != ${expected}`);

test('a death between birthdays stops cash flows and records the final partial year',()=>{
  const s=flatPlan();let draws=0;
  const path=runOne(s,{nextDouble:()=>draws++===0?maleMortality[65]/2:.999999,normal:()=>0},{captureMonthlyBalances:true,captureTaxDetails:true});
  assert.equal(path.deathAge,65.5);assert.equal(path.observationEndAge,65.5);assert.equal(path.censored,false);
  assert.equal(path.monthlyBalances.length,6);assert.equal(path.taxYears.length,1);
  assert.equal(path.yearEnd.length,2);near(path.yearEnd.at(-1),94000);
});

test('a monthly primary death changes survivor spending and the following tax year',()=>{
  const s=flatPlan(65,2);Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:65});let draws=0;
  const path=runOne(s,{nextDouble:()=>draws++===0?maleMortality[65]/2:.999999,normal:()=>0},{captureTaxDetails:true});
  assert.equal(path.censored,true);assert.equal(path.observationEndAge,67);
  assert.deepEqual(path.taxYears.map(year=>year.status),['Married','Single']);
  near(path.yearEnd[1],100000-6000-6000*.84);near(path.yearEnd[2],100000-6000-18000*.84);
});

test('monthly death timing moves care and home sale without depending on the observation cutoff',()=>{
  const s=flatPlan();s.home.currentValue=200000;Object.assign(s.longTermCare,{enabled:true,annualCost:24000,averageDurationYears:1});
  assert.deepEqual(validateScenario(s),[]);
  const rng=()=>{let draws=0;return {nextDouble:()=>[.999999,maleMortality[66]/2,0][draws++]??.999999,normal:()=>0};};
  const path=runOne(s,rng(),{captureMonthlyBalances:true});
  assert.equal(path.deathAge,66.5);assert.equal(path.observationEndAge,66.5);assert.equal(path.censored,false);
  near(path.monthlyBalances[6],94000);assert.ok(path.monthlyBalances[7]>290000);
  near(path.yearEnd.at(-1),100000-6000-24000+200000*1.02);
  const shorter=structuredClone(s);shorter.household.targetEndAge=66;
  const censored=runOne(shorter,rng(),{captureMonthlyBalances:true});
  assert.equal(censored.deathAge,66.5);assert.equal(censored.observationEndAge,66);assert.equal(censored.censored,true);
  assert.deepEqual(censored.monthlyBalances,path.monthlyBalances.slice(0,12));
});
