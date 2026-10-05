import {ALLOCATION_KEYS} from './model.js';

// Stock allocation optimizer. Each portfolio-size band is tuned on its own in
// 10-point steps, starting from the plan's schedule and repeating the sweep
// until nothing moves. Every schedule runs on the same search paths. The best
// schedule has the highest readiness; among schedules within one point of it,
// the higher median ending balance wins. The winner and the current schedule
// are then rerun on fresh paths, so the reported numbers are not fitted to the
// paths the search tuned against.
export const ALLOCATION_SEARCH_PATHS=500,ALLOCATION_CHECK_PATHS=2000,ALLOCATION_NEAR_TIE=.01,ALLOCATION_MAX_PASSES=3;
export const ALLOCATION_STEPS=Array.from({length:11},(_,i)=>i/10);
export const ALLOCATION_BAND_LABELS=['Under 30×','30–35×','35–40×','40–45×','45–50×','50× or more'];
const SEARCH_SEED_OFFSET=30000,CHECK_SEED_OFFSET=40000;

// Keep imported/editor percentages distinct from neighboring grid values.
const keyOf=a=>JSON.stringify(ALLOCATION_KEYS.map(k=>a[k]));
const distance=(a,b)=>ALLOCATION_KEYS.reduce((n,k)=>n+Math.abs(a[k]-b[k]),0);
export const scheduleOf=s=>Object.fromEntries(ALLOCATION_KEYS.map(k=>[k,s.postRetirementAllocation[k]]));
export const sameSchedule=(a,b)=>keyOf(a)===keyOf(b);

// Highest readiness, then the higher median balance among near-ties, then the
// schedule closest to `anchor`, so bands with no measured effect stay as they are.
export function preferredAllocation(entries,anchor){
  if(!entries.length)return null;
  const top=Math.max(...entries.map(e=>e.readiness));
  return entries.filter(e=>e.readiness>=top-ALLOCATION_NEAR_TIE-1e-9).reduce((a,b)=>b.medianEndingBalance>a.medianEndingBalance||b.medianEndingBalance===a.medianEndingBalance&&distance(b.allocation,anchor)<distance(a.allocation,anchor)?b:a);
}

// `evaluate({kind:'allocation',allocation,count,seed})` resolves to
// {readiness,medianEndingBalance}; a caller can spread calls across workers.
export async function searchAllocation(s,{evaluate,onProgress,searchPaths=ALLOCATION_SEARCH_PATHS,checkPaths=ALLOCATION_CHECK_PATHS,maxPasses=ALLOCATION_MAX_PASSES}={}){
  const current=scheduleOf(s),cache=new Map(),bands=ALLOCATION_KEYS.length;
  const progress={phase:'search',pass:1,maxPasses,band:0,bands,tested:0,checkPaths};
  const report=()=>onProgress?.({...progress});
  const score=allocation=>{
    const key=keyOf(allocation);
    if(!cache.has(key))cache.set(key,Promise.resolve(evaluate({kind:'allocation',allocation,count:searchPaths,seed:s.seed+SEARCH_SEED_OFFSET})).then(r=>{progress.tested++;report();return {allocation,...r};}));
    return cache.get(key);
  };
  report();
  let schedule=current;const affectedResults={};
  for(let pass=1;pass<=maxPasses;pass++){
    progress.pass=pass;let moved=false;
    for(const [band,key] of ALLOCATION_KEYS.entries()){
      progress.band=band;progress.bandLabel=ALLOCATION_BAND_LABELS[band];report();
      const entries=await Promise.all([...new Set([...ALLOCATION_STEPS,schedule[key]])].map(v=>score({...schedule,[key]:v})));
      // Equal summary statistics do not prove the band was never reached.
      // Keep any measured effect, even if a later pass no longer sees one.
      affectedResults[key] ||= entries.some(e=>e.readiness!==entries[0].readiness||e.medianEndingBalance!==entries[0].medianEndingBalance);
      const pick=preferredAllocation(entries,schedule);
      if(pick.allocation[key]!==schedule[key]){schedule=pick.allocation;moved=true;}
    }
    if(!moved)break;
  }
  const searched=await Promise.all(cache.values()),best=preferredAllocation(searched,current),baseline=await score(current);
  progress.phase='check';report();
  const check=allocation=>Promise.resolve(evaluate({kind:'allocation',allocation,count:checkPaths,seed:s.seed+CHECK_SEED_OFFSET}));
  const changed=!sameSchedule(best.allocation,current);
  const [checkedCurrent,checkedBest]=await Promise.all(changed?[check(current),check(best.allocation)]:[check(current)]).then(r=>changed?r:[r[0],r[0]]);
  const suggested={allocation:best.allocation,...checkedBest},now={allocation:current,...checkedCurrent};
  return {searchPaths,checkPaths,nearTie:ALLOCATION_NEAR_TIE,tested:cache.size,passes:progress.pass,changed,
    improved:changed&&preferredAllocation([now,suggested],current)===suggested,
    current:now,suggested,searchResults:{current:{readiness:baseline.readiness,medianEndingBalance:baseline.medianEndingBalance},suggested:{readiness:best.readiness,medianEndingBalance:best.medianEndingBalance}},
    bands:ALLOCATION_KEYS.map((key,i)=>({key,label:ALLOCATION_BAND_LABELS[i],current:current[key],suggested:best.allocation[key],affectedResults:affectedResults[key]}))};
}
