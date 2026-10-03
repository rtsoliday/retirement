import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,prepareCalendarScenario,localCalendarDate} from '../dist/model.js';
import {runSimulation} from '../dist/engine.js';
import {resultFingerprint,matchingCachedResult,createResultCache} from '../dist/result-cache.js';

test('saved results match calculation inputs and date, while ignoring labels, draft budgets and next-run path counts',()=>{
  const s=prepareCalendarScenario(baseScenario(),{needsReview:false}),today=localCalendarDate();s.numberOfSimulations=10;
  const result=runSimulation(s),record={version:1,fingerprint:resultFingerprint(s,today),result};
  const copy=structuredClone(s);copy.name='Renamed';copy.budget.annualPropertyTaxes=12345;copy.numberOfSimulations=10000;copy.simulationPathsCustomized=true;
  assert.equal(matchingCachedResult(record,copy,today),result);
  for(const edit of [c=>c.accounts.cash++,c=>c.household.retirementDate='2035-01-01',c=>c.market.stockMeanReturn=.07,c=>c.household.gender='Female']){
    const changed=structuredClone(s);edit(changed);assert.equal(matchingCachedResult(record,changed,today),null);
  }
  assert.equal(matchingCachedResult(record,s,'2026-12-31'),null);
  const wrong=structuredClone(record);wrong.result.provenance.engineVersion='old-model';assert.equal(matchingCachedResult(wrong,s,today),null);
  wrong.result=result;wrong.version=2;assert.equal(matchingCachedResult(wrong,s,today),null);
  wrong.version=1;wrong.result={...result,steadySimulation:null};assert.equal(matchingCachedResult(wrong,s,today),null);
});

test('unavailable or blocked result storage fails without interrupting calculations or plan storage',async()=>{
  const unavailable=createResultCache(null);assert.equal(await unavailable.load('plan'),null);
  assert.equal(await unavailable.clear(),null);
  const blocked=createResultCache({open(){const request={};queueMicrotask(()=>request.onblocked());return request;}});
  assert.equal(await blocked.load('plan'),null);
});
