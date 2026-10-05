import {scenarioEngineVersion,syncCalendarAges,omitUnusedPlanLabInputs,DEFAULT_SEED} from './model.js';

// Match calculation inputs, not plan names, draft budgets, source labels or
// the next-run path count. A completed run always retains its actual count.
export function resultFingerprint(s,today){
  const copy=structuredClone(s);
  for(const key of ['id','name','budget','numberOfSimulations','simulationPathsCustomized'])delete copy[key];
  delete copy.household.datesNeedReview;
  copy.seed=DEFAULT_SEED;syncCalendarAges(copy);copy.household.asOfDate=today;omitUnusedPlanLabInputs(copy);
  const stable=value=>Array.isArray(value)?value.map(stable):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,stable(value[key])])):value;
  return JSON.stringify([scenarioEngineVersion(copy),stable(copy)]);
}

export function matchingCachedResult(record,s,today){
  const r=record?.result;
  if(record?.version!==1||record.fingerprint!==resultFingerprint(s,today)||!r||r.scenarioId!==s.id||r.provenance?.engineVersion!==scenarioEngineVersion(s))return null;
  if(!Number.isFinite(r.generatedAtEpochMillis)||!Number.isInteger(r.provenance.simulationCount)||r.provenance.simulationCount<4||r.provenance.simulationCount>10000||!Number.isFinite(r.successProbability)||r.successProbability<0||r.successProbability>1||!Number.isFinite(r.medianEndingBalance))return null;
  if(!['balanceBands','notFailedByAge','failureAgeBuckets','pathPoints'].every(key=>Array.isArray(r[key]))||!Array.isArray(r.steadySimulation?.monthlyDetails)||!r.riskBreakdown||!r.todayDollars)return null;
  return r;
}

// Results have their own local database so a large Pro run cannot use up the
// small localStorage allowance needed to save the user's plan inputs.
export function createResultCache(indexedDB=globalThis.indexedDB){
  let opening;
  const open=()=>{
    if(opening)return opening;
    const attempt=new Promise((resolve,reject)=>{
      if(!indexedDB){reject(new Error('Result storage unavailable'));return;}
      const request=indexedDB.open('retirement-forecast-results',1);let failed=false;
      const fail=error=>{failed=true;reject(error);};
      request.onupgradeneeded=()=>request.result.createObjectStore('results',{keyPath:'id'});
      request.onsuccess=()=>{
        const db=request.result;
        // A blocked request can finish after its callers have already retried.
        if(failed){db.close();return;}
        db.onversionchange=()=>{db.close();if(opening===attempt)opening=undefined;};
        resolve(db);
      };
      request.onerror=()=>fail(request.error);
      request.onblocked=()=>fail(new Error('Result storage is busy'));
    });
    opening=attempt;
    attempt.catch(()=>{if(opening===attempt)opening=undefined;});
    return attempt;
  };
  const transaction=async(mode,operation)=>{
    try{
      const db=await open();
      return await new Promise((resolve,reject)=>{
        const tx=db.transaction('results',mode),request=operation(tx.objectStore('results'));
        tx.oncomplete=()=>resolve(request.result??true);
        tx.onerror=tx.onabort=()=>reject(tx.error||request.error);
      });
    }catch{return null;}
  };
  return {
    load:id=>transaction('readonly',store=>store.get(id)),
    save:(s,result,today)=>transaction('readwrite',store=>store.put({id:s.id,version:1,fingerprint:resultFingerprint(s,today),result:structuredClone(result)})),
    remove:id=>transaction('readwrite',store=>store.delete(id)),
    clear:()=>transaction('readwrite',store=>store.clear())
  };
}
