import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,validateScenario} from '../dist/model.js';
import {runOne} from '../dist/engine.js';

function plan(separatePeople=false){
  const s=baseScenario();
  Object.assign(s.household,{separatePeople,birthday:'1966-10-03',retirementDate:'2027-10-03',spouseBirthday:'1966-10-03',spouseRetirementDate:'2028-10-03',asOfDate:'2026-10-03',filingStatus:'Married'});
  s.spending.generalInflationStdDev=0;s.healthcare.healthcareInflationStdDev=0;s.longTermCare.enabled=false;
  for(const key of Object.keys(s.market))s.market[key]=0;
  return s;
}
const run=s=>runOne(s,{normal:mean=>mean,nextDouble:()=>.5},{captureMonthlyDetails:true,fixedDeathAges:{primary:64,spouse:64}});

test('inactive savings settings preserve values and have no effect on cash flows',()=>{
  for(const separate of [false,true])for(const owner of ['individual-spouse','retired-you','retired-spouse']){
    const s=plan(separate),prefix=owner==='retired-you'?'contributions':'spouseContributions';
    if(owner==='individual-spouse')s.household.filingStatus='Single';
    else if(owner==='retired-you'||!separate){s.household.alreadyRetired=true;s.household.retirementDate='';}
    else {s.household.spouseAlreadyRetired=true;s.household.spouseRetirementDate='';}
    s[prefix].pretax=-100;s[prefix].annualIncrease=-2;
    const before=structuredClone(s),baseline=structuredClone(s);baseline[prefix]=baseScenario()[prefix];
    assert.deepEqual(validateScenario(s),[],`${owner}, separate=${separate}`);
    assert.deepEqual(run(s),run(baseline));assert.deepEqual(s,before);
    if(owner==='individual-spouse')s.household.filingStatus='Married';
    else if(owner==='retired-you'||!separate){s.household.alreadyRetired=false;s.household.retirementDate='2027-10-03';}
    else {s.household.spouseAlreadyRetired=false;s.household.spouseRetirementDate='2028-10-03';}
    assert.match(validateScenario(s).join(' '),/savings contributions/);
  }
});

test('a pooled spouse follows the shared retirement status and working owners still need valid savings',()=>{
  const s=plan();s.household.spouseAlreadyRetired=true;s.spouseContributions.pretax=-100;
  assert.match(validateScenario(s).join(' '),/Spouse savings contributions/);
  for(const retired of ['you','spouse']){
    const s=plan(true);s.household[retired==='you'?'alreadyRetired':'spouseAlreadyRetired']=true;
    s[retired==='you'?'spouseContributions':'contributions'].pretax=-100;
    assert.match(validateScenario(s).join(' '),/savings contributions/);
  }
});

test('zero pensions ignore retained timing, growth and survivor choices until the pension is enabled',()=>{
  for(const separate of [false,true]){
    const s=plan(separate),baseline=structuredClone(s);
    Object.assign(s.guaranteedIncome,{startAge:-1.5,startAgeMonths:12,annualIncrease:-2,survivorPercent:2});
    Object.assign(s.spouseIncome,{pensionStartAge:-1.5,pensionStartAgeMonths:12,annualIncrease:-2,survivorPercent:2});
    const before=structuredClone(s);
    assert.deepEqual(validateScenario(s),[]);assert.deepEqual(run(s),run(baseline));assert.deepEqual(s,before);
    s.guaranteedIncome.annualIncome=100;
    assert.match(validateScenario(s).join(' '),/Month fields|Income start age|Guaranteed income/);
    s.guaranteedIncome.annualIncome=0;
    if(separate){s.spouseIncome.annualPension=100;assert.match(validateScenario(s).join(' '),/Spouse pension/);}
  }
  const s=plan();s.guaranteedIncome.annualIncrease=Number.MAX_SAFE_INTEGER;
  assert.deepEqual(validateScenario(s),[]);assert.ok(Number.isFinite(run(s).yearEnd.at(-1)));
});

test('disabled long-term care retains unused cost and duration drafts without changing cash flows',()=>{
  for(const separate of [false,true]){
    const s=plan(separate),baseline=structuredClone(s);
    Object.assign(s.longTermCare,{annualCost:-100,averageDurationYears:.5,averageDurationMonths:12});
    const before=structuredClone(s);
    assert.deepEqual(validateScenario(s),[]);
    assert.deepEqual(run(s),run(baseline));assert.deepEqual(s,before);
    s.longTermCare.enabled=true;
    assert.match(validateScenario(s).join(' '),/Month fields|duration years|Long-term care cost or duration/);
  }
});

test('individual plans retain unused household support drafts and couples validate them again',()=>{
  for(const filingStatus of ['Single','HeadOfHousehold']){
    const s=plan(true);s.household.filingStatus=filingStatus;
    const baseline=structuredClone(s);
    Object.assign(s.workingIncome,{primaryAnnualNet:-100,spouseAnnualNet:-200,annualIncrease:-2});
    const before=structuredClone(s);
    assert.deepEqual(validateScenario(s),[]);
    assert.deepEqual(run(s),run(baseline));assert.deepEqual(s,before);
    s.household.filingStatus='Married';
    assert.match(validateScenario(s).join(' '),/Take-home household support/);
  }
});

test('disabled care and individual support still reject malformed, nonfinite and unsafe values',()=>{
  for(const value of ['100',NaN,Infinity,Number.MAX_SAFE_INTEGER*2]){
    const s=plan(true);s.household.filingStatus='Single';
    s.longTermCare.annualCost=value;
    assert.ok(validateScenario(s).length,String(value));
    s.longTermCare.annualCost=0;s.workingIncome.spouseAnnualNet=value;
    assert.ok(validateScenario(s).length,String(value));
  }
});

test('inactive settings still reject malformed, nonfinite and unsafe data',()=>{
  for(const value of ['100',NaN,Infinity,Number.MAX_SAFE_INTEGER*2]){
    const s=plan(true);s.household.alreadyRetired=true;s.contributions.pretax=value;
    assert.ok(validateScenario(s).length,String(value));
    s.contributions.pretax=0;s.guaranteedIncome.annualIncrease=value;
    assert.ok(validateScenario(s).length,String(value));
  }
  for(const separate of [false,true]){
    const s=plan(separate);s.guaranteedIncome.annualIncome=-1;
    assert.match(validateScenario(s).join(' '),/Guaranteed income/);
    if(separate){s.guaranteedIncome.annualIncome=0;s.spouseIncome.annualPension=-1;assert.match(validateScenario(s).join(' '),/Spouse pension/);}
  }
});
