import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,FREE_SIMULATION_PATHS} from '../dist/model.js';
import {runSimulation,runOne,JavaRandom,sampleDeathAge,deathAgeAtQuantile,previewLifespanQuantiles} from '../dist/engine.js';

const near=(a,b,tolerance=1e-6)=>assert.ok(Math.abs(a-b)<=tolerance*Math.max(1,Math.abs(b)),`${a} != ${b}`);

test('lifespan quantiles invert the monthly mortality distribution',()=>{
  const rng=new JavaRandom(7),draws=Array.from({length:20000},()=>sampleDeathAge('Male',65,120,rng));
  for(const u of [.1,.5,.9]){
    const age=deathAgeAtQuantile('Male',65,120,u),share=draws.filter(d=>d<=age).length/draws.length;
    assert.ok(Math.abs(share-u)<.015,`quantile ${u}: ${share}`);
    assert.equal(Math.round(age*12),age*12);
  }
  assert.ok(deathAgeAtQuantile('Female',65,120,.5)>deathAgeAtQuantile('Male',65,120,.5));
  assert.equal(deathAgeAtQuantile('Male',65,70,.999),70);
});

test('previews spread lifespans over equal-probability bands; larger runs stay random',()=>{
  assert.equal(previewLifespanQuantiles(FREE_SIMULATION_PATHS+1,1),null);
  const bands=previewLifespanQuantiles(10,20260766);
  assert.deepEqual(bands.map(b=>b.primary),[.05,.15,.25,.35,.45,.55,.65,.75,.85,.95]);
  assert.deepEqual(bands.map(b=>b.spouse).sort((a,b)=>a-b),bands.map(b=>b.primary));
  assert.deepEqual(previewLifespanQuantiles(10,20260766),bands);
  const s=baseScenario(),path=runOne(s,new JavaRandom(s.seed),{lifespanQuantiles:{primary:.5,spouse:.5}});
  assert.equal(path.deathAge,deathAgeAtQuantile('Male',67,120,.5));
});

test('a ten-path preview observes deaths across the whole table instead of by chance',()=>{
  const s=baseScenario();s.numberOfSimulations=10;
  const r=runSimulation(s),last=r.notFailedByAge.at(-1);
  // The latest band (95th percentile) outlives the median lifespan by years.
  assert.ok(last.age>=deathAgeAtQuantile('Male',67,120,.95)-1/12);
  const unbanded=runSimulation(s,undefined,{stratifyPreviewLifespans:false});
  assert.equal(unbanded.provenance.simulationCount,10);
});

test('today’s dollars deflate each path by its own price level and leave future dollars unchanged',()=>{
  const s=baseScenario();s.numberOfSimulations=12;
  const r=runSimulation(s),plain=runSimulation(s,undefined,{includeRiskAnalysis:false});
  assert.equal(r.medianEndingBalance,plain.medianEndingBalance);
  const t=r.todayDollars,startIndex=Math.pow(1+s.spending.generalInflationMean,7);
  // Pre-retirement prices follow the mean, so the first band deflates exactly.
  near(t.balanceBands[0].median,r.balanceBands[0].median/startIndex,1e-9);
  assert.ok(t.medianEndingBalance<r.medianEndingBalance);
  assert.equal(t.balanceBands.length,r.balanceBands.length);
  assert.equal(t.pathPoints.length,r.pathPoints.length);
  assert.equal(t.steadyPriceIndexes.length,r.steadySimulation.monthlyDetails.length);
  near(t.steadyPriceIndexes[0],startIndex,1e-9);
  near(t.steadyPriceIndexes[12],startIndex*(1+s.spending.generalInflationMean),1e-9);
});

test('with no inflation, today’s dollars equal future dollars',()=>{
  const s=baseScenario();s.numberOfSimulations=12;s.spending.generalInflationMean=0;s.spending.generalInflationStdDev=0;
  const r=runSimulation(s),t=r.todayDollars;
  near(t.medianEndingBalance,r.medianEndingBalance);
  r.balanceBands.forEach((band,i)=>{near(t.balanceBands[i].median,band.median);near(t.balanceBands[i].optimistic,band.optimistic);});
  t.steadyPriceIndexes.forEach(index=>near(index,1));
});
