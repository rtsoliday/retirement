import test from 'node:test';
import assert from 'node:assert/strict';
import {buildBalanceBands,buildFundingSurvival} from '../dist/chart-data.js';
import {baseScenario} from '../dist/model.js';
import {runSimulation,runOne,JavaRandom} from '../dist/engine.js';

// Seed of the reference snapshots below; the site default seed differs.
const REFERENCE_SEED=20260429;

test('ending balances, failure ages and bands average the middle pair for even samples',()=>{
  const middle=values=>{values.sort((a,b)=>a-b);return (values[Math.floor((values.length-1)/2)]+values[Math.floor(values.length/2)])/2;};
  for(const count of [1,3,4,50]){
    const s=baseScenario();s.numberOfSimulations=count;s.spending.annualBaseSpending=250000;
    const paths=Array.from({length:count},(_,i)=>runOne(s,new JavaRandom(BigInt(s.seed)+BigInt(i)*-7046029254386353131n)));
    const r=runSimulation(s),failures=paths.filter(p=>p.failureAge!==null).map(p=>p.failureAge);
    assert.equal(r.medianEndingBalance,middle(paths.map(p=>p.yearEnd.at(-1))));
    assert.equal(r.medianFailureAge,failures.length?middle(failures):null);
    assert.equal(r.balanceBands[0].median,middle(paths.map(p=>p.chart[0])));
  }
  const bands=buildBalanceBands([0,0,1000000,10000000].map(balance=>({chart:[balance],failureAge:null,survivedThroughAge:65})),65);
  assert.deepEqual(bands,[{age:65,pessimistic:0,median:500000,optimistic:10000000,pathCount:4}]);
});

test('funding and survival include the final partial-year shortfall without inventing death at the cutoff',()=>{
  const s=baseScenario();s.numberOfSimulations=10;Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:66});
  s.accounts={pretax:0,roth:11500,taxable:0,cash:0};
  Object.assign(s.spending,{annualBaseSpending:12000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  Object.assign(s.market,{stockMeanReturn:0,stockStdDev:0,bondMeanReturn:0,bondStdDev:0});
  s.socialSecurity.annualBenefitAt67=0;s.healthcare.includeMedicarePremiums=false;s.longTermCare.enabled=false;
  const r=runSimulation(s);assert.equal(r.successProbability,0);
  assert.deepEqual(r.notFailedByAge,[
    {age:65,notFailedShare:1,aliveShare:1},
    {age:791/12,notFailedShare:0,aliveShare:1},
    {age:66,notFailedShare:0,aliveShare:1}
  ]);
});

test('successful lifetimes remain funded at death and the final curve matches readiness',()=>{
  for(const extraMonths of [0,1,6,11]){
    const s=baseScenario();s.household.retirementAgeMonths=extraMonths;
    const r=runSimulation(s);
    assert.equal(r.notFailedByAge.at(-1).notFailedShare,r.successProbability);
    assert.equal(r.notFailedByAge.at(-1).aliveShare,0);
    for(let i=1;i<r.notFailedByAge.length;i++){
      assert.ok(r.notFailedByAge[i].age>r.notFailedByAge[i-1].age);
      assert.ok(r.notFailedByAge[i].notFailedShare<=r.notFailedByAge[i-1].notFailedShare);
      assert.ok(r.notFailedByAge[i].aliveShare<=r.notFailedByAge[i-1].aliveShare);
    }
  }
});

test('funding curves retain both death and failure transitions between retirement anniversaries',()=>{
  const curve=buildFundingSurvival([
    {failureAge:791/12,deathAge:66},
    {failureAge:null,deathAge:65.5},
    {failureAge:66.25,deathAge:67},
    {failureAge:null,deathAge:67.5}
  ],65.25);
  assert.deepEqual(curve,[
    {age:65.25,notFailedShare:1,aliveShare:1},
    {age:65.5,notFailedShare:1,aliveShare:.75},
    {age:791/12,notFailedShare:.75,aliveShare:.75},
    {age:66,notFailedShare:.75,aliveShare:.5},
    {age:66.25,notFailedShare:.5,aliveShare:.5},
    {age:67,notFailedShare:.5,aliveShare:.25},
    {age:67.25,notFailedShare:.5,aliveShare:.25},
    {age:67.5,notFailedShare:.5,aliveShare:0}
  ]);
});

test('monthly failure bands include other paths still alive and funded in that month',()=>{
  const bands=buildBalanceBands([
    {chart:[100],monthlyBalances:[100,90,80,0],failureAge:65.25,survivedThroughAge:67},
    {chart:[500,400],monthlyBalances:[500,490,480,470],failureAge:null,survivedThroughAge:66}
  ],65);
  assert.deepEqual(bands.find(b=>b.age===65.25),{age:65.25,pessimistic:0,median:235,optimistic:470,pathCount:2});
  assert.equal(bands.find(b=>b.age===65).pessimistic,100);
});

test('bands exclude stopped paths, record failure as zero at its age, and show shrinking samples',()=>{
  const bands=buildBalanceBands([
    {chart:[100,90,80],failureAge:66,survivedThroughAge:100},
    {chart:[200,210,220,230,240],failureAge:null,survivedThroughAge:67},
    {chart:[300,310,320,330,340],failureAge:null,survivedThroughAge:68}
  ],65);
  assert.deepEqual(bands,[
    {age:65,pessimistic:100,median:200,optimistic:300,pathCount:3},
    {age:66,pessimistic:0,median:210,optimistic:310,pathCount:3},
    {age:67,pessimistic:220,median:270,optimistic:320,pathCount:2},
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
  const path=runOne(s,new JavaRandom(s.seed)),r=runSimulation(s,undefined,{stratifyPreviewLifespans:false});
  assert.equal(path.success,true);assert.ok(path.survivedThroughAge<s.household.targetEndAge);
  assert.equal(r.balanceBands.at(-1).age,path.survivedThroughAge);
  assert.ok(path.yearEnd.length<s.household.targetEndAge-s.household.retirementAge+1);
  assert.equal(r.medianEndingBalance,path.yearEnd.at(-1));
});
test('mixed simulation bands are nonnegative, do not outlive sampled lifetimes, and retain readiness while reporting monthly failure ages',()=>{
  const s=baseScenario();s.accounts={pretax:800000,roth:100000,taxable:0,cash:50000};s.household.currentAge=50;s.seed=REFERENCE_SEED;s.numberOfSimulations=50;s.spending.annualBaseSpending=90000;
  const r=runSimulation(s);assert.equal(r.successProbability,.98);assert.equal(r.medianFailureAge,956/12);
  assert.ok(r.balanceBands.at(-1).age<=r.notFailedByAge.at(-1).age);
  assert.ok(r.balanceBands.at(-1).age<s.household.targetEndAge);
  for(const b of r.balanceBands){assert.ok(b.pessimistic>=0);assert.ok(b.pessimistic<=b.median&&b.median<=b.optimistic);assert.ok(b.pathCount>0&&b.pathCount<=s.numberOfSimulations);}
});
