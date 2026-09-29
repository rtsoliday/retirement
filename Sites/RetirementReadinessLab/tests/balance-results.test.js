import test from 'node:test';
import assert from 'node:assert/strict';
import {buildBalanceBands} from '../dist/chart-data.js';
import {baseScenario} from '../dist/model.js';
import {runSimulation,runOne,JavaRandom} from '../dist/engine.js';

// Seed of the reference snapshots below; the site default seed differs.
const REFERENCE_SEED=20260429;

test('bands exclude stopped paths, record failure as zero at its age, and show shrinking samples',()=>{
  const bands=buildBalanceBands([
    {chart:[100,90,80],failureAge:66,survivedThroughAge:100},
    {chart:[200,210,220,230,240],failureAge:null,survivedThroughAge:67},
    {chart:[300,310,320,330,340],failureAge:null,survivedThroughAge:68}
  ],65);
  assert.deepEqual(bands,[
    {age:65,pessimistic:100,median:200,optimistic:300,pathCount:3},
    {age:66,pessimistic:0,median:210,optimistic:310,pathCount:3},
    {age:67,pessimistic:220,median:320,optimistic:320,pathCount:2},
    {age:68,pessimistic:330,median:330,optimistic:330,pathCount:1}
  ]);
});
test('a shortfall in the first month ends at retirement age with zero downside and no extended bands',()=>{
  const s=baseScenario();s.household.currentAge=s.household.retirementAge=65;
  s.accounts={pretax:0,roth:0,taxable:0,cash:0};s.socialSecurity.annualBenefitAt67=0;s.numberOfSimulations=10;
  const r=runSimulation(s);assert.equal(r.successProbability,0);assert.equal(r.medianFailureAge,65);
  assert.equal(r.pessimisticEndingBalance,0);assert.equal(r.medianEndingBalance,0);assert.equal(r.optimisticEndingBalance,0);
  assert.deepEqual(r.balanceBands,[{age:65,pessimistic:0,median:0,optimistic:0,pathCount:10}]);
  const path=runOne(s,new JavaRandom(s.seed));assert.equal(path.yearEnd.at(-1),0);assert.equal(path.yearEnd.length,2);
});
test('successful lifetime balances stop at last living age rather than model cap',()=>{
  const s=baseScenario();s.seed=REFERENCE_SEED;s.numberOfSimulations=1;
  const path=runOne(s,new JavaRandom(s.seed)),r=runSimulation(s);
  assert.equal(path.success,true);assert.ok(path.survivedThroughAge<s.household.targetEndAge);
  assert.equal(r.balanceBands.at(-1).age,path.survivedThroughAge);
  assert.ok(path.yearEnd.length<s.household.targetEndAge-s.household.retirementAge+1);
  assert.equal(r.medianEndingBalance,path.yearEnd.at(-1));
});
test('mixed simulation bands are nonnegative, do not outlive sampled lifetimes, and retain original failure statistics',()=>{
  const s=baseScenario();s.household.currentAge=50;s.seed=REFERENCE_SEED;s.numberOfSimulations=50;
  const r=runSimulation(s);assert.equal(r.successProbability,.98);assert.equal(r.medianFailureAge,78);
  assert.ok(r.balanceBands.at(-1).age<=r.notFailedByAge.at(-1).age);
  assert.ok(r.balanceBands.at(-1).age<s.household.targetEndAge);
  for(const b of r.balanceBands){assert.ok(b.pessimistic>=0);assert.ok(b.pessimistic<=b.median&&b.median<=b.optimistic);assert.ok(b.pathCount>0&&b.pathCount<=s.numberOfSimulations);}
});
