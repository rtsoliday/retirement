import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,setRetirementAge,setAnnualBaseSpending} from '../dist/model.js';
import {candidateReadiness,runSimulation} from '../dist/engine.js';
import {searchReadinessFrontier} from '../dist/target-frontier.js';
import {frontierCard,frontierChoices,frontierPlot} from '../dist/frontier-view.js';

function shortPlan(){const s=baseScenario();Object.assign(s.household,{currentAge:60,retirementAge:61,targetEndAge:64});return s;}
test('adaptive targets reuse neighboring values, verify 200-path pairs and leave inputs intact',async()=>{
  const s=shortPlan(),before=structuredClone(s),calls=[],progress=[];
  const frontier=await searchReadinessFrontier(s,{evaluate:task=>{calls.push(task);return task.value<=40000+(task.age-60)*1000?.8:.7;},onProgress:p=>progress.push(p)});
  assert.deepEqual(frontier.points.map(point=>[point.age,point.annualSpending]),[[60,40000],[61,41000],[62,42000],[63,43000]]);
  assert.ok(calls.every(task=>task.kind==='frontier'&&task.count===200));
  assert.ok(frontier.points[1].evaluatedCandidates<frontier.points[0].evaluatedCandidates);
  assert.ok(frontier.checkedCandidates<4*501/4,'Far fewer evaluations than scanning every $500 amount at every age.');
  assert.equal(frontier.adaptiveSearch,true);assert.deepEqual(s,before);
  assert.equal(progress.at(-1).completedAges,4);assert.equal(progress.at(-1).totalFrontierAges,4);
});
test('wider probes find a higher qualifying island above a failed local boundary',async()=>{
  const s=shortPlan();s.household.targetEndAge=62;
  const frontier=await searchReadinessFrontier(s,{initialSpending:30000,evaluate:({age,value})=>age===60&&(value<=30000||value>=125000&&value<=160000)?.85:.1});
  assert.equal(frontier.points[0].annualSpending,160000);assert.equal(frontier.points[0].readiness,.85);
  assert.equal(frontier.points[1].annualSpending,null);assert.equal(frontierChoices(frontier).length,1);
  assert.match(frontierCard(frontier),/higher qualifying spending amounts may exist/);
});
test('frontier endpoints disclose a reached ceiling and exclude infeasible home-bill amounts',async()=>{
  const s=shortPlan();s.home.annualTaxesAndInsurance=30500.01;
  const frontier=await searchReadinessFrontier(s,{evaluate:task=>{assert.ok(task.value>=31000);return 1;}});
  assert.ok(frontier.points.every(point=>point.atSearchLimit&&point.annualSpending===250000));
  assert.match(frontierCard(frontier),/\$250,000/);
  s.home.annualTaxesAndInsurance=250000.01;
  const empty=await searchReadinessFrontier(s,{evaluate:()=>{assert.fail('No feasible amount should run.');}});
  assert.ok(empty.points.every(point=>point.annualSpending===null));assert.equal(empty.checkedCandidates,0);
  assert.doesNotMatch(frontierCard(empty),/apply-frontier-target|type="range"/);
});
test('joint age and spending candidates match the full simulation with the same paths and seed',()=>{
  const s=shortPlan();s.household.targetEndAge=62;s.accounts={pretax:0,roth:0,taxable:0,cash:200000};s.longTermCare.enabled=false;
  const task={kind:'frontier',age:61,value:75000,count:200,targetReadiness:.8},before=structuredClone(s);
  const readiness=candidateReadiness(s,task),variant=structuredClone(s);
  setRetirementAge(variant,task.age);setAnnualBaseSpending(variant,task.value);variant.numberOfSimulations=200;variant.seed+=20000;
  assert.ok(readiness>=.8);assert.equal(readiness,runSimulation(variant,undefined,{includeRiskAnalysis:false,includePathPoints:false}).successProbability);
  assert.deepEqual(s,before);
});
test('the chart shows only validated choices, leaves gaps and provides an accessible discrete slider',()=>{
  const frontier={targetReadiness:.8,simulationCount:200,searchStartAge:60,searchEndAge:63,spendingSearchLimit:250000,points:[
    {age:60,annualSpending:40000,readiness:.81,tested:true,simulationCount:200},{age:61,annualSpending:null,readiness:null},
    {age:62,annualSpending:35000,readiness:.8,tested:true,simulationCount:200},{age:63,annualSpending:50000,readiness:.79}
  ]};
  const plot=frontierPlot(frontier,1),card=frontierCard(frontier,1);
  assert.equal(frontierChoices(frontier).length,2);assert.equal((plot.match(/data-frontier-index=/g)||[]).length,2);
  assert.match(plot,/Age on the horizontal axis; annual base spending/);
  assert.doesNotMatch(plot,/<path|frontier-line/,'Untested intermediate pairs must not be connected.');
  assert.match(card,/id="frontier-age"[^>]*max="1"[^>]*step="1"[^>]*value="1"/);
  assert.match(card,/aria-valuetext="Age 62, \$35,000/);assert.match(card,/Use selected age &amp; spending/);
  assert.match(card,/tested together with 200 paths/);assert.match(card,/200-path sample/);
});

test('untested and incomplete pairs are never plotted or selectable even when they have plausible numbers',()=>{
  const frontier={targetReadiness:.8,simulationCount:200,searchStartAge:60,searchEndAge:64,points:[
    {age:60,annualSpending:50000,readiness:.9},
    {age:61,annualSpending:51000,readiness:.9,tested:false,simulationCount:200},
    {age:62,annualSpending:52000,readiness:.9,tested:true,simulationCount:180},
    {age:63,annualSpending:53000,readiness:.9,tested:true,simulationCount:200},
    {age:64,annualSpending:54000,readiness:.79,tested:true,simulationCount:200}
  ]};
  assert.deepEqual(frontierChoices(frontier).map(point=>point.age),[63]);
  const plot=frontierPlot(frontier);assert.equal((plot.match(/<circle/g)||[]).length,1);assert.doesNotMatch(plot,/<path/);
  assert.doesNotMatch(plot,/\$50,000|\$51,000|\$52,000|\$54,000/);
  assert.match(frontierCard(frontier),/no untested combinations are connected or selectable/);
});

test('claiming-age spending search tests future ages 62 through 70, reuses neighbors and records completed tests',async()=>{
  const s=shortPlan();s.household.targetEndAge=80;const before=structuredClone(s),calls=[];
  const result=await searchReadinessFrontier(s,{kind:'claiming',evaluate:task=>{calls.push(task);return task.value<=40000+(task.age-62)*1000?.8:0;}});
  assert.equal(result.kind,'claiming');assert.equal(result.searchStartAge,62);assert.equal(result.searchEndAge,70);
  assert.deepEqual(result.points.map(point=>point.annualSpending),[40000,41000,42000,43000,44000,45000,46000,47000,48000]);
  assert.ok(calls.every(task=>task.kind==='claim-frontier'&&task.count===200));
  assert.ok(result.points.every(point=>point.tested&&point.simulationCount===200&&calls.some(task=>task.age===point.age&&task.value===point.annualSpending)));
  assert.deepEqual(s,before);const card=frontierCard(result);
  assert.match(card,/Social Security claiming age/);assert.match(card,/id="claim-frontier-age"/);assert.match(card,/apply-claim-frontier-target/);
  s.household.currentAge=69;s.household.retirementAge=69;
  const late=await searchReadinessFrontier(s,{kind:'claiming',evaluate:()=>1});assert.deepEqual(late.points.map(point=>point.age),[69,70]);
  s.household.currentAge=71;s.household.retirementAge=71;
  const empty=await searchReadinessFrontier(s,{kind:'claiming',evaluate:()=>assert.fail('Past claiming ages must not be proposed.')});
  assert.equal(empty.points.length,0);assert.match(frontierCard(empty),/No future whole-year Social Security claiming ages/);
});

test('claiming age and spending candidates match an actual simulation without changing retirement or spouse choices',()=>{
  const s=shortPlan();s.household.targetEndAge=64;s.accounts={pretax:0,roth:0,taxable:0,cash:500000};s.longTermCare.enabled=false;
  const before=structuredClone(s),task={kind:'claim-frontier',age:62,value:75000,count:200,targetReadiness:.8};
  const readiness=candidateReadiness(s,task),variant=structuredClone(s);variant.socialSecurity.claimAge=62;setAnnualBaseSpending(variant,75000);variant.numberOfSimulations=200;variant.seed+=20000;
  assert.ok(readiness>=.8);assert.equal(readiness,runSimulation(variant,undefined,{includeRiskAnalysis:false,includePathPoints:false}).successProbability);
  assert.deepEqual(s,before);
});
