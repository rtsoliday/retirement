import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,prepareCalendarScenario,localCalendarDate,scenarioEngineVersion} from '../dist/model.js';
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

function cacheDatabase(records){
  return {closed:false,close(){this.closed=true;},transaction(){
    const tx={};
    const complete=result=>{const request={result};queueMicrotask(()=>tx.oncomplete());return request;};
    tx.objectStore=()=>({
      get:id=>complete(records.get(id)),
      put:record=>{records.set(record.id,structuredClone(record));return complete(record.id);}
    });
    return tx;
  }};
}

test('a temporary database opening error is retried and later results can be saved and loaded',async()=>{
  const records=new Map(),db=cacheDatabase(records);let opens=0;
  const cache=createResultCache({open(){
    const attempt=++opens,request={result:db,error:new Error('Temporary database failure')};
    queueMicrotask(()=>attempt===1?request.onerror():request.onsuccess());return request;
  }});
  assert.equal(await cache.load('base-plan'),null);
  const s=prepareCalendarScenario(baseScenario()),today=localCalendarDate(),result={scenarioId:s.id};
  assert.equal(await cache.save(s,result,today),s.id);
  assert.deepEqual((await cache.load(s.id)).result,result);
  assert.equal(opens,2);
});

test('a blocked database that opens late is closed and cannot replace a healthy retry connection',async()=>{
  const records=new Map([['base-plan',{id:'base-plan'}]]),late=cacheDatabase(records),healthy=cacheDatabase(records),requests=[];
  const cache=createResultCache({open(){
    const first=requests.length===0,request={result:first?late:healthy};requests.push(request);
    if(first)queueMicrotask(()=>request.onblocked());return request;
  }});
  assert.equal(await cache.load('base-plan'),null);
  const retry=cache.load('base-plan');
  assert.equal(requests.length,2);
  requests[0].onsuccess();assert.equal(late.closed,true);assert.equal(healthy.closed,false);
  requests[1].onsuccess();assert.deepEqual(await retry,{id:'base-plan'});
  assert.deepEqual(await cache.load('base-plan'),{id:'base-plan'});assert.equal(requests.length,2);
});

test('pooled results from before the pension correction cannot be restored',()=>{
  const s=prepareCalendarScenario(baseScenario()),today=localCalendarDate(),result=runSimulation(s,()=>{},{includeRiskAnalysis:false});
  result.riskBreakdown={};
  const currentVersion=scenarioEngineVersion(s),previousVersion=currentVersion.replace('-survivor-pension-v2','');
  assert.notEqual(currentVersion,previousVersion);
  const fingerprint=JSON.parse(resultFingerprint(s,today));fingerprint[0]=previousVersion;
  const record={version:1,fingerprint:JSON.stringify(fingerprint),result:{...result,provenance:{...result.provenance,engineVersion:previousVersion}}};
  assert.equal(matchingCachedResult(record,s,today),null);
  for(const change of [p=>p.contributions.pretax=12000,p=>p.household.alreadyRetired=true]){
    const plan=structuredClone(s);change(plan);assert.match(scenarioEngineVersion(plan),/-survivor-pension-v2$/);
  }
  s.household.separatePeople=true;assert.equal(scenarioEngineVersion(s),'2026.10-separate-people-medicare-rmd-v2');
});

test('separate-owner results from before the Medicare correction require a new run',()=>{
  const s=prepareCalendarScenario(baseScenario()),today=localCalendarDate();s.household.separatePeople=true;
  for(const retired of [false,true]){
    s.household.alreadyRetired=retired;
    if(retired)s.household.retirementDate='';
    const result=runSimulation(s,()=>{},{includeRiskAnalysis:false});result.riskBreakdown={};
    assert.equal(matchingCachedResult({version:1,fingerprint:resultFingerprint(s,today),result},s,today),result);
    const currentVersion=scenarioEngineVersion(s),previousVersion=retired?'2026.10-retired-forecast':'2026.10-separate-people';
    assert.notEqual(currentVersion,previousVersion);
    const fingerprint=JSON.parse(resultFingerprint(s,today));fingerprint[0]=previousVersion;
    const record={version:1,fingerprint:JSON.stringify(fingerprint),result:{...result,provenance:{...result.provenance,engineVersion:previousVersion}}};
    assert.equal(matchingCachedResult(record,s,today),null);
  }
});
