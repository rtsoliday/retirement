import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,setRetirementAge,primaryRetirementAge,addCalendarMonths,prepareCalendarScenario} from '../dist/model.js';
import {annualPersonalSavings,setAnnualPersonalSavings} from '../dist/savings-targets.js';
import {searchSavingsFrontier} from '../dist/target-frontier.js';
import {candidateReadiness,runSimulation} from '../dist/engine.js';
import {frontierCard,frontierPlot,frontierChoices} from '../dist/frontier-view.js';

function shortPlan(){const s=baseScenario();Object.assign(s.household,{currentAge:60,retirementAge:63,targetEndAge:65});return s;}
test('savings allocation includes your employer Roth employee deposits, retains employer/spouse deposits and defaults to cash',()=>{
  const s=shortPlan();s.contributions.pretax=10000;s.contributions.roth=5000;s.contributions.employerPretax=4000;s.contributions.annualIncrease=.03;
  s.spouseContributions.cash=8000;s.employerRothAccounts=[{owner:'you',annualContribution:5000,annualEmployerContribution:2000,annualIncrease:.04},{owner:'spouse',annualContribution:3000}];
  const before=structuredClone(s);assert.equal(annualPersonalSavings(s),20000);setAnnualPersonalSavings(s,40000);
  assert.equal(s.contributions.pretax,20000);assert.equal(s.contributions.roth,10000);assert.equal(s.employerRothAccounts[0].annualContribution,10000);
  assert.equal(s.contributions.employerPretax,4000);assert.equal(s.contributions.annualIncrease,.03);assert.equal(s.employerRothAccounts[0].annualEmployerContribution,2000);assert.equal(s.employerRothAccounts[0].annualIncrease,.04);
  assert.deepEqual(s.spouseContributions,before.spouseContributions);assert.deepEqual(s.employerRothAccounts[1],before.employerRothAccounts[1]);
  const empty=shortPlan();setAnnualPersonalSavings(empty,25000);assert.equal(empty.contributions.cash,25000);assert.equal(empty.contributions.pretax,0);
});
test('savings search finds completed 80% pairs rounded up, tests zero and excludes failed points',async()=>{
  const s=shortPlan(),before=structuredClone(s),calls=[];
  const result=await searchSavingsFrontier(s,{evaluate:t=>{calls.push(t);return t.age===60?0:t.value>=10501-(t.age-61)*1000?.8:0;}});
  assert.deepEqual(result.points.map(p=>[p.age,p.annualSavings]),[[60,null],[61,11000],[62,10000],[63,9000],[64,8000]]);
  assert.equal(result.fixedAnnualSpending,s.spending.annualBaseSpending);assert.equal(result.savingsSearchLimit,1000000);
  assert.ok(frontierChoices(result).every(p=>calls.some(t=>t.age===p.age&&t.value===p.annualSavings&&t.count===200)));
  assert.deepEqual(s,before);const plot=frontierPlot(result),card=frontierCard(result);
  assert.equal((plot.match(/<circle/g)||[]).length,4);assert.doesNotMatch(plot,/<path/);assert.match(card,/Annual base spending stays at/);assert.match(card,/lower qualifying savings amounts may exist/);assert.match(card,/apply-savings-frontier-target/);
  const zero=await searchSavingsFrontier(s,{evaluate:()=>1});assert.ok(zero.points.every(p=>p.annualSavings===0&&p.evaluatedCandidates===1));
});
test('adaptive savings probes detect a lower qualifying island and reject untested chart points',async()=>{
  const s=shortPlan();s.contributions.cash=100000;
  const r=await searchSavingsFrontier(s,{evaluate:({value})=>value>=100000||value>=24000&&value<=27000?.85:0});
  assert.ok(r.points.every(p=>p.annualSavings<=27000));r.points[0].tested=false;r.points[1].simulationCount=180;
  assert.equal(frontierChoices(r).length,3);
});
test('future whole ages extend beyond planned retirement while respecting the horizon; retired plans offer no targets',async()=>{
  const s=prepareCalendarScenario(shortPlan(),{today:'2026-10-04',needsReview:false});s.household.birthday='1966-03-01';s.household.retirementDate='2029-08-01';
  const r=await searchSavingsFrontier(s,{evaluate:()=>1});assert.deepEqual(r.points.map(p=>p.age),[61,62,63,64]);assert.ok(r.points.some(p=>p.age>primaryRetirementAge(s)));assert.ok(r.points.every(p=>p.age<s.household.targetEndAge));
  s.household.alreadyRetired=true;s.household.retirementDate='2026-03-01';const empty=await searchSavingsFrontier(s,{evaluate:()=>assert.fail('Retired plan must not be converted to future retirement.')});
  assert.equal(empty.points.length,0);assert.match(frontierCard(empty),/already marks you retired/);assert.doesNotMatch(frontierCard(empty),/apply-savings-frontier-target/);
});
test('actual savings candidate matches all 200 simulation paths with the same spending and fixed contributions',()=>{
  const s=shortPlan();s.household.targetEndAge=64;s.accounts={pretax:0,roth:0,taxable:0,cash:400000};s.longTermCare.enabled=false;s.contributions.cash=5000;s.contributions.employerPretax=3000;
  const before=structuredClone(s),task={kind:'savings-frontier',age:62,value:20000,count:200,targetReadiness:.8},actual=candidateReadiness(s,task),variant=structuredClone(s);
  setRetirementAge(variant,62);setAnnualPersonalSavings(variant,20000);variant.numberOfSimulations=200;variant.seed+=20000;
  assert.ok(actual>=.8);assert.equal(actual,runSimulation(variant,undefined,{includeRiskAnalysis:false,includePathPoints:false}).successProbability);
  assert.deepEqual(variant.spending,before.spending);assert.deepEqual(s,before);
});

test('savings search includes later ages through 70 or five years after a later selected retirement',async()=>{
  const s=shortPlan();s.household.targetEndAge=95;const before=structuredClone(s);
  const r=await searchSavingsFrontier(s,{evaluate:()=>.8});assert.equal(r.searchEndAge,70);assert.equal(r.points.at(-1).age,70);assert.ok(r.points.some(p=>p.age>s.household.retirementAge));assert.deepEqual(s,before);
  s.household.currentAge=70;s.household.retirementAge=72;
  const later=await searchSavingsFrontier(s,{evaluate:()=>.8});assert.deepEqual(later.points.map(p=>p.age),[70,71,72,73,74,75,76,77]);
  assert.ok(later.points.every(p=>p.tested&&p.simulationCount===200));assert.match(frontierCard(later),/Retirement ages 70 through 77/);
  s.household.retirementAgeMonths=6;const fractional=await searchSavingsFrontier(s,{evaluate:()=>.8});assert.equal(fractional.searchEndAge,78);
  s.household.targetEndAge=75;const capped=await searchSavingsFrontier(s,{evaluate:()=>.8});assert.equal(capped.searchEndAge,74);
  const couple=shortPlan();couple.household.filingStatus='Married';couple.household.spouseCurrentAge=65;couple.household.targetEndAge=75;const spouseCapped=await searchSavingsFrontier(couple,{evaluate:()=>.8});assert.equal(spouseCapped.searchEndAge,69);
});
