import test from 'node:test';
import assert from 'node:assert/strict';
import {accountCalendar,PersonAccounts} from '../dist/person-accounts.js';
import {JavaRandom,runOne,runSimulation} from '../dist/engine.js';
import {simulationScenarios} from './fixtures/simulation-scenarios.js';

test('account calendars retain month-end clamping, leap years and independent start/deposit days',()=>{
  const s=simulationScenarios().find(([name])=>name==='separate-couple')[1];
  Object.assign(s.household,{asOfDate:'2024-01-31',retirementDate:'2024-02-15',spouseRetirementDate:'2024-03-31'});
  const calendar=accountCalendar(s);
  assert.equal(calendar.todayDate(1),'2024-02-29');
  assert.equal(calendar.todayDate(2),'2024-03-31');
  assert.equal(calendar.todayDate(13),'2025-02-28');
  assert.equal(calendar.startDate(1),'2024-03-15');
  assert.equal(calendar.depositYear(11),2024);
  assert.equal(calendar.depositYear(12),2025);
  const first=new PersonAccounts(s,{...s.accounts},calendar),second=new PersonAccounts(s,{...s.accounts},calendar);
  first.people[0].pretax=0;
  assert.equal(second.people[0].pretax,s.accounts.pretax);
  s.household.asOfDate='2025-01-31';
  const next=accountCalendar(s);
  assert.equal(next.todayDate(1),'2025-02-28');
  assert.equal(calendar.todayDate(1),'2024-02-29');
});

for(const[name,s]of simulationScenarios())test(`calendar reuse preserves complete results and account/tax traces: ${name}`,()=>{
  // Evaluate the original date operations on this runtime. Cross-platform math
  // can differ in its final bits, so pinned JSON hashes would mask this check.
  const shared=accountCalendar(s),uncached=accountCalendar(s,{reuseDates:false});
  const result=runSimulation(s),reference=runSimulation(s,()=>{},{accountCalendar:uncached});
  delete result.generatedAtEpochMillis;delete reference.generatedAtEpochMillis;
  assert.deepEqual(result,reference);
  for(const i of [0,1,7]){
    if(!s.household.separatePeople)continue;
    const seed=BigInt(s.seed)+BigInt(i)*-7046029254386353131n,options={captureMonthlyBalances:true,captureMonthlyDetails:true,captureTaxDetails:true,captureTodayDollars:true};
    assert.deepEqual(runOne(s,new JavaRandom(seed),{...options,accountCalendar:shared}),runOne(s,new JavaRandom(seed),{...options,accountCalendar:uncached}));
  }
});
