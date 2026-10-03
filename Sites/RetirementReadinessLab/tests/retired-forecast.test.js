import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,scenarioTimeline,forecastRetirementDate,validateScenario,normalizeScenario,delayRetirement,setRetirementAge,ruleOf55Applies} from '../dist/model.js';
import {runOne,JavaRandom} from '../dist/engine.js';
import {PersonAccounts} from '../dist/person-accounts.js';

function retiredPlan(){
  const s=baseScenario();
  Object.assign(s.household,{birthday:'1970-10-02',retirementDate:'2020-10-02',alreadyRetired:true,asOfDate:'2026-10-02',targetEndAge:60});
  s.longTermCare.enabled=false;s.socialSecurity.annualBenefitAt67=0;
  return s;
}

test('past retirement dates use today’s balances without replaying growth or future deposits',()=>{
  const s=retiredPlan();s.contributions.pretax=12000;
  assert.deepEqual(validateScenario(s),[]);
  assert.equal(forecastRetirementDate(s),'2026-10-02');
  assert.equal(scenarioTimeline(s).preMonths,0);
  const path=runOne(s,new JavaRandom(123n),{captureMonthlyDetails:true,fixedDeathAges:[58,null]});
  assert.equal(path.monthlyDetails[0].date,'2026-10-02');
  assert.equal(path.monthlyDetails[0].pretax,s.accounts.pretax);
  assert.equal(path.monthlyDetails[0].portfolio,Object.values(s.accounts).reduce((a,b)=>a+b,0));
  assert.equal(s.household.retirementDate,'2020-10-02');
});

test('retired status accepts an omitted separation date but still validates birthdays and actual dates',()=>{
  const s=retiredPlan();s.household.retirementDate='';assert.deepEqual(validateScenario(s),[]);
  for(const date of ['2026-10-03','1969-10-02','invalid']){s.household.retirementDate=date;assert.match(validateScenario(s).join(' '),/Actual retirement date/);}
  s.household.retirementDate='';s.household.birthday='';assert.match(validateScenario(s).join(' '),/birthday/);
  const normal=retiredPlan();normal.household.alreadyRetired=false;assert.match(validateScenario(normal).join(' '),/today or later/);
});

test('separate retired owners stop deposits while a working spouse continues savings and support',()=>{
  const s=retiredPlan();Object.assign(s.household,{separatePeople:true,filingStatus:'Married',spouseBirthday:'1972-10-02',spouseRetirementDate:'2028-10-02'});
  s.contributions.pretax=24000;s.spouseContributions.pretax=12000;s.workingIncome.spouseAnnualNet=24000;
  assert.deepEqual(validateScenario(s),[]);assert.equal(scenarioTimeline(s).startDate,'2026-10-02');
  const path=runOne(s,new JavaRandom(123n),{captureMonthlyDetails:true,taxesEnabled:false,fixedDeathAges:[58,58]});
  const first=path.monthlyDetails[1].cashFlow;
  assert.equal(first.savingsContributions,1000);assert.equal(first.workingSupport,2000);
  s.household.spouseAlreadyRetired=true;s.household.spouseRetirementDate='';
  const both=runOne(s,new JavaRandom(123n),{captureMonthlyDetails:true,taxesEnabled:false,fixedDeathAges:[58,58]});
  assert.equal(both.monthlyDetails[1].cashFlow.savingsContributions,0);
  assert.equal(both.monthlyDetails[1].cashFlow.workingSupport,0);
});

test('Rule of 55 uses actual separation, and never substitutes the forecast start when unknown',()=>{
  const s=retiredPlan();s.withdrawalStrategy.ruleOf55Eligible=true;
  assert.equal(ruleOf55Applies(s),false);
  s.household.separatePeople=true;
  assert.equal(new PersonAccounts(s,{...s.accounts}).people[0].rule55,false);
  s.household.retirementDate='2026-01-01';assert.equal(ruleOf55Applies(s),true);
  assert.equal(new PersonAccounts(s,{...s.accounts}).people[0].rule55,true);
  s.household.retirementDate='';assert.equal(ruleOf55Applies(s),false);
  assert.equal(new PersonAccounts(s,{...s.accounts}).people[0].rule55,false);
});

test('a spouse already retired starts household costs today while the primary owner keeps saving',()=>{
  const s=retiredPlan();Object.assign(s.household,{alreadyRetired:false,retirementDate:'2028-10-02',separatePeople:true,filingStatus:'Married',spouseBirthday:'1972-10-02',spouseAlreadyRetired:true,spouseRetirementDate:''});
  s.contributions.pretax=12000;s.spouseContributions.pretax=24000;s.workingIncome.primaryAnnualNet=36000;
  assert.deepEqual(validateScenario(s),[]);assert.equal(scenarioTimeline(s).startDate,'2026-10-02');assert.equal(scenarioTimeline(s).preMonths,0);
  const path=runOne(s,new JavaRandom(123n),{captureMonthlyDetails:true,taxesEnabled:false,fixedDeathAges:[59,57]});
  assert.equal(path.monthlyDetails[1].cashFlow.savingsContributions,1000);
  assert.equal(path.monthlyDetails[1].cashFlow.workingSupport,3000);
});

test('retired flags survive normalization and retirement comparisons explicitly return to future mode',()=>{
  const s=normalizeScenario(JSON.parse(JSON.stringify(retiredPlan())));assert.equal(s.household.alreadyRetired,true);
  delayRetirement(s,2);assert.equal(s.household.alreadyRetired,false);
  assert.equal(s.household.retirementDate,'2028-10-02');
  s.household.alreadyRetired=true;setRetirementAge(s,58);assert.equal(s.household.alreadyRetired,false);
  assert.equal(s.household.retirementDate,'2028-10-02');
});
