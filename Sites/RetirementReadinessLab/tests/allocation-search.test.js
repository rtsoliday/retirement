import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,ALLOCATION_KEYS} from '../dist/model.js';
import {allocationOutcome,runSimulation} from '../dist/engine.js';
import {searchAllocation,preferredAllocation,sameSchedule,ALLOCATION_SEARCH_PATHS,ALLOCATION_CHECK_PATHS} from '../dist/allocation-search.js';
import {allocationCard} from '../dist/plan-lab-view.js';
import {labLever,applyWhatIf,normalizeLabSets,whatIfErrors} from '../dist/plan-lab.js';

const schedule=(...v)=>Object.fromEntries(ALLOCATION_KEYS.map((k,i)=>[k,v[i]]));
const entry=(readiness,medianEndingBalance,allocation=schedule(1,.9,.8,.7,.6,.5))=>({readiness,medianEndingBalance,allocation});

test('preferred allocation takes the best readiness, then the larger median among schedules within one point',()=>{
  const a=entry(.90,100,schedule(1,1,1,1,1,1)),b=entry(.895,200,schedule(.5,.5,.5,.5,.5,.5)),c=entry(.88,900,schedule(0,0,0,0,0,0));
  assert.equal(preferredAllocation([a,b,c],a.allocation),b,'within a point, the larger median wins');
  assert.equal(preferredAllocation([a,c],a.allocation),a,'two points lower never wins on balance');
  // Identical outcomes keep the schedule closest to the anchor.
  const anchor=schedule(1,.9,.8,.7,.6,.5),same=entry(.9,100,schedule(1,.9,.8,.7,.6,0)),home=entry(.9,100,anchor);
  assert.equal(preferredAllocation([same,home],anchor),home);
  assert.equal(preferredAllocation([],anchor),null);
});

test('allocation search tunes bands with measured effects, retains other bands and reports fresh-path results',async()=>{
  const s=baseScenario(),before=structuredClone(s),calls=[],progress=[];
  // Readiness peaks at 60% stocks in the first band and 40% in the second;
  // the median favors more stock in the third band. Later bands are never reached.
  const evaluate=t=>{
    calls.push(t);const a=t.allocation;
    const readiness=.9-Math.abs(a.stockUnder30x-.6)/2-Math.abs(a.stock30xTo35x-.4)/2;
    const fresh=t.count===ALLOCATION_CHECK_PATHS?-.01:0;
    return Promise.resolve({readiness:readiness+fresh,medianEndingBalance:1000*a.stock35xTo40x});
  };
  const d=await searchAllocation(s,{evaluate,onProgress:p=>progress.push(p)});
  assert.deepEqual(d.bands.map(b=>Math.round(b.suggested*100)),[60,40,100,70,60,50]);
  assert.deepEqual(d.bands.map(b=>b.affectedResults),[true,true,true,false,false,false]);
  assert.equal(d.changed,true);assert.equal(d.improved,true);assert.equal(d.passes,2);
  assert.ok(calls.filter(t=>t.count===ALLOCATION_SEARCH_PATHS).every(t=>t.seed===s.seed+30000));
  assert.deepEqual(calls.filter(t=>t.count===ALLOCATION_CHECK_PATHS).map(t=>t.seed),[s.seed+40000,s.seed+40000]);
  assert.ok(Math.abs(d.suggested.readiness-.89)<1e-9,'reported readiness comes from the fresh paths');
  assert.equal(d.tested,new Set(calls.filter(t=>t.count===ALLOCATION_SEARCH_PATHS).map(t=>JSON.stringify(t.allocation))).size,'each schedule runs once');
  assert.equal(progress[0].phase,'search');assert.equal(progress.at(-1).phase,'check');
  assert.deepEqual(s,before,'the plan itself is never changed');
  const card=allocationCard(d);
  assert.match(card,/Use this allocation/);assert.match(card,/Add as a what-if/);assert.match(card,/no effect found/);assert.match(card,/2,000 fresh paths/);
});

test('allocation search keeps the current schedule when nothing beats it',async()=>{
  const s=baseScenario(),calls=[];
  const d=await searchAllocation(s,{evaluate:t=>{calls.push(t);return {readiness:.8,medianEndingBalance:5};}});
  assert.equal(d.changed,false);assert.equal(d.improved,false);assert.equal(d.passes,1);
  assert.ok(d.bands.every(b=>b.suggested===b.current&&!b.affectedResults));
  assert.equal(calls.filter(t=>t.count===ALLOCATION_CHECK_PATHS).length,1,'an unchanged schedule is checked once');
  const card=allocationCard(d);assert.match(card,/already the best one found/);assert.doesNotMatch(card,/Use this allocation/);
});

test('fractional current allocations are scored and retained without colliding with grid candidates',async()=>{
  const s=baseScenario();s.postRetirementAllocation.stockUnder30x=.5001;
  const calls=[],current=structuredClone(s.postRetirementAllocation);
  assert.equal(sameSchedule(current,{...current,stockUnder30x:.5}),false);
  const d=await searchAllocation(s,{evaluate:t=>{
    calls.push(t);
    return {readiness:.9,medianEndingBalance:t.allocation.stockUnder30x===.5001?1000:100};
  }});
  assert.equal(d.changed,false);
  assert.deepEqual(d.suggested.allocation,current);
  assert.deepEqual(d.searchResults.current,{readiness:.9,medianEndingBalance:1000});
  assert.ok(calls.some(t=>t.count===ALLOCATION_SEARCH_PATHS&&t.allocation.stockUnder30x===.5001));
  assert.match(allocationCard(d),/50\.01%/);
});

test('identical candidate outcomes are described as no measured effect rather than an unreached band',async()=>{
  const d=await searchAllocation(baseScenario(),{evaluate:()=>({readiness:.8,medianEndingBalance:0})});
  const html=allocationCard(d);
  assert.match(html,/no effect found/i);
  assert.doesNotMatch(html,/not reached|never came into play/);
  assert.match(html,/combinations/,'a coordinate search cannot exclude better combinations of grid values');
});

test('the allocation what-if editor preserves fractional percentages from an optimizer result',()=>{
  const s=baseScenario(),lever=labLever('stockAllocation'),allocation={...s.postRetirementAllocation,stockUnder30x:.5001,stock30xTo35x:.555,stock35xTo40x:1/3};
  const fields=lever.fields(s,allocation),draft=Object.fromEntries(fields.map(f=>[f.name,f.value]));
  assert.equal(draft.stockUnder30x,'50.01');assert.equal(draft.stock30xTo35x,'55.5');
  const read=lever.read(draft,s,allocation);assert.equal(read.error,undefined);
  assert.deepEqual(read.value,allocation);
});

test('allocation outcome is repeatable on the same seed and responds to the schedule',()=>{
  const s=baseScenario(),task=allocation=>({allocation,count:50,seed:s.seed+30000});
  const a=allocationOutcome(s,task(schedule(1,.9,.8,.7,.6,.5))),b=allocationOutcome(s,task(schedule(1,.9,.8,.7,.6,.5))),bonds=allocationOutcome(s,task(schedule(0,0,0,0,0,0)));
  assert.deepEqual(a,b);
  assert.ok(a.readiness>=0&&a.readiness<=1&&a.medianEndingBalance>=0);
  assert.notDeepEqual(a,bonds);
});

test('allocation scores match complete forecasts for pooled and separately owned plans',()=>{
  for(const separatePeople of [false,true]){
    const s=baseScenario();s.household.separatePeople=separatePeople;
    if(separatePeople)Object.assign(s.household,{birthday:'1966-10-01',retirementDate:'2033-10-01',spouseBirthday:'1968-10-01',spouseRetirementDate:'2035-10-01',asOfDate:'2026-10-01',filingStatus:'Married'});
    const allocation={...s.postRetirementAllocation,stockUnder30x:.5001},count=50,seed=s.seed+30000;
    const score=allocationOutcome(s,{allocation,count,seed});
    const full=runSimulation({...s,postRetirementAllocation:allocation,numberOfSimulations:count,seed},undefined,{includeRiskAnalysis:false,includePathPoints:false});
    assert.equal(score.readiness,full.successProbability);
    assert.equal(score.medianEndingBalance,full.todayDollars.medianEndingBalance);
  }
});

test('the stock allocation what-if stores a whole schedule and survives saving',()=>{
  const s=baseScenario(),lever=labLever('stockAllocation'),fields=lever.fields(s);
  assert.equal(fields.length,6);assert.deepEqual(fields.map(f=>f.value),['100','90','80','70','60','50']);
  const draft=Object.fromEntries(fields.map(f=>[f.name,f.value]));
  assert.deepEqual(lever.read({...draft,stock30xTo35x:'100'}).value,schedule(1,1,.8,.7,.6,.5));
  assert.match(lever.read({...draft,stock50xOrMore:'101'}).error,/50× or more/);
  assert.match(lever.read({...draft,stockUnder30x:'not a number'}).error,/percentage/);
  assert.match(lever.read({...draft,stockUnder30x:''}).error,/percentage/);
  const changes={stockAllocation:schedule(.6,.4,1,.7,.6,.5),stockShift:-10},next=applyWhatIf(s,changes);
  assert.deepEqual(next.postRetirementAllocation,schedule(.5,.3,.9,.6,.5,.4),'a mix change applies on top of the schedule');
  assert.deepEqual(whatIfErrors(s,{stockAllocation:changes.stockAllocation}),[]);
  assert.equal(lever.describe(changes.stockAllocation),'Stocks 60/40/100/70/60/50% by savings level');
  const saved={[s.id]:{activeSet:'set',sets:[{id:'set',name:'Set',whatIfs:[{id:'w1',name:'Good',changes:{stockAllocation:changes.stockAllocation}},{id:'w2',name:'Bad',changes:{stockAllocation:{...changes.stockAllocation,stockUnder30x:2}}}]}]}};
  const [kept,dropped]=normalizeLabSets(saved,[s])[s.id].sets[0].whatIfs;
  assert.deepEqual(kept.changes,{stockAllocation:changes.stockAllocation});assert.deepEqual(dropped.changes,{});
});
