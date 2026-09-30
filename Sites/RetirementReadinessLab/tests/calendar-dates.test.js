import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,calendarDate,calendarMonthsBetween,addCalendarMonths,prepareCalendarScenario,scenarioTimeline,retirementAge,setRetirementAge,validateScenario,normalizeScenario,ruleOf55Applies,dateLabel,delayRetirement} from '../dist/model.js';
import {runSimulation,runOne,JavaRandom,decisionPlan,candidateReadiness} from '../dist/engine.js';

function calendarPlan(){
  const s=baseScenario();
  Object.assign(s.household,{birthday:'1966-10-15',spouseBirthday:'1968-01-31',retirementDate:'2033-04-14',asOfDate:'2026-09-30'});
  return s;
}
test('calendar parsing rejects rollover dates and uses calendar anniversaries at leap days and month ends',()=>{
  for(const value of ['2026-02-29','2026-04-31','09/30/2026','2026-9-30','invalid',42])assert.equal(calendarDate(value),null);
  assert.ok(calendarDate('2024-02-29'));
  assert.equal(addCalendarMonths('1964-02-29',61*12),'2025-02-28');
  assert.equal(calendarMonthsBetween('1964-02-29','2025-02-27'),61*12-1);
  assert.equal(calendarMonthsBetween('1964-02-29','2025-02-28'),61*12);
  assert.equal(calendarMonthsBetween('1960-01-31','2026-02-28'),66*12+1);
  assert.equal(dateLabel('2033-04-14'),'Apr 14, 2033');
  assert.equal(addCalendarMonths('1960-01-31',Number.MAX_VALUE),'');
});
test('dates determine completed months, true birth years, and the retirement year independently of legacy ages',()=>{
  const s=calendarPlan(),t=scenarioTimeline(s);
  assert.deepEqual(t,{currentAge:719/12,retirementAge:797/12,preMonths:78,spouseAtRet:782/12,birthYear:1966,spouseBirthYear:1968,retirementYear:2033});
  Object.assign(s.household,{currentAge:25,retirementAge:40,spouseCurrentAge:30});
  assert.deepEqual(scenarioTimeline(s),t);assert.equal(retirementAge(s),797/12);
  s.household.retirementDate='2033-04-15';assert.equal(retirementAge(s),798/12);
  assert.deepEqual(validateScenario(s),[]);
});
test('legacy plans retain balances and monthly retirement distance and inferred dates stay fixed on reload',()=>{
  const s=baseScenario();s.household.retirementAgeMonths=6;s.accounts.pretax=123456;
  prepareCalendarScenario(s,{today:'2026-09-30'});
  assert.equal(s.household.birthday,'1966-09-30');assert.equal(s.household.retirementDate,'2034-03-30');
  assert.equal(s.household.spouseBirthday,'1966-09-30');assert.equal(s.household.datesNeedReview,true);
  assert.equal(scenarioTimeline(s,'2026-09-30').preMonths,90);assert.equal(s.accounts.pretax,123456);
  const restored=prepareCalendarScenario(normalizeScenario(JSON.parse(JSON.stringify(s))),{today:'2026-10-01'});
  assert.equal(restored.household.retirementDate,s.household.retirementDate);
  assert.equal(restored.household.birthday,s.household.birthday);
});
test('invalid dates are rejected before simulation; married plans require a valid spouse birthday',()=>{
  for(const [key,value] of [['birthday','2026-02-29'],['birthday','2026-10-01'],['retirementDate','2026-09-29'],['retirementDate','2033-04-31'],['retirementDate','']]){
    const s=calendarPlan();s.household[key]=value;
    assert.ok(validateScenario(s).length);assert.throws(()=>runSimulation(s));
  }
  const s=calendarPlan();s.household.spouseBirthday='';assert.deepEqual(validateScenario(s),[]);
  s.household.filingStatus='Married';assert.match(validateScenario(s).join(' '),/Spouse birthday/);
  s.household.spouseBirthday='1900-01-01';assert.match(validateScenario(s).join(' '),/Spouse age/);
});
test('calendar simulations match equivalent monthly age scenarios and record the selected retirement year',()=>{
  const s=prepareCalendarScenario(baseScenario(),{today:'2026-09-30',needsReview:false});s.household.asOfDate='2026-09-30';
  const legacy=baseScenario();
  const options={includeRiskAnalysis:false};
  assert.deepEqual(runSimulation(s,()=>{},options).balanceBands,runSimulation(legacy,()=>{},options).balanceBands);
  s.household.retirementDate='2027-01-30';s.household.targetEndAge=62;
  const path=runOne(s,new JavaRandom(123n),{captureTaxDetails:true});
  assert.equal(path.taxYears[0].taxYear,2027);
});
test('Rule of 55 checks the real birth and retirement calendar years',()=>{
  const s=calendarPlan();s.withdrawalStrategy.ruleOf55Eligible=true;
  Object.assign(s.household,{birthday:'1972-12-31',retirementDate:'2026-12-31'});
  assert.equal(ruleOf55Applies(s),false);
  s.household.retirementDate='2027-01-01';assert.equal(ruleOf55Applies(s),true);
});
test('retirement variants and age searches change the calendar date and never test a past birthday',()=>{
  const s=calendarPlan();s.household.birthday='1966-09-29';
  const plan=decisionPlan(s,.8,50);assert.equal(plan.firstAge,61);assert.equal(plan.lastAge,70);
  setRetirementAge(s,68.5);assert.equal(s.household.retirementDate,'2035-03-29');assert.equal(retirementAge(s),68.5);
  assert.doesNotThrow(()=>candidateReadiness(s,{kind:'age',value:61,count:1,targetReadiness:0}));
  s.household.asOfDate='2026-09-29';assert.equal(decisionPlan(s,.8,50).firstAge,60);
});
test('later-retirement comparisons preserve the selected day of the month',()=>{
  const s=calendarPlan();delayRetirement(s,2);
  assert.equal(s.household.retirementDate,'2035-04-14');
});
