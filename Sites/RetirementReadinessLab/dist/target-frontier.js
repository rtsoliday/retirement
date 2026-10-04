import {decisionPlan,candidateReadiness,TARGET_SIMULATION_PATHS} from './engine.js';
import {primaryRetirementAge} from './model.js';
import {annualPersonalSavings,savingsAllocation} from './savings-targets.js';

// Search for the lowest tested qualifying deposit amount. Readiness can change
// with taxes and allocation, so also probe below the local bracket and disclose
// the bounded adaptive search rather than claiming an exhaustive minimum.
export async function searchSavingsFrontier(s,{evaluate,targetReadiness=.8,simulationCount=TARGET_SIMULATION_PATHS,onProgress}={}){
  // Include later retirement alternatives without making the range unbounded.
  // decisionPlan also caps it at the household's modeling horizon.
  const lastRequestedAge=Math.max(70,Math.ceil(primaryRetirementAge(s))+5);
  const plan=decisionPlan(s,targetReadiness,simulationCount,lastRequestedAge),run=evaluate??(task=>candidateReadiness(s,task));
  if(s.household.alreadyRetired)plan.ages=[];
  const maximum=2000,points=[];let hint=Math.min(maximum,Math.ceil(annualPersonalSavings(s)/500)),checkedCandidates=0,age=null;
  const report=()=>onProgress?.({phase:'frontier',kind:'savings',completedAges:points.length,totalFrontierAges:plan.ages.length,age,checkedCandidates});report();
  for(age of plan.ages){
    const cache=new Map(),probe=index=>{if(!cache.has(index))cache.set(index,Promise.resolve().then(()=>run({kind:'savings-frontier',age,value:index*500,count:plan.count,targetReadiness})).then(readiness=>{checkedCandidates++;report();return readiness;}));return cache.get(index);};
    const qualifies=async index=>(await probe(index))>=targetReadiness;
    let best=null;
    if(await qualifies(0))best=0;
    else{
      let lower=0,upper=null;const start=Math.max(1,hint);
      if(await qualifies(start))upper=start;
      else{lower=start;for(let step=1;;step*=2){const index=Math.min(maximum,start+step);if(await qualifies(index)){upper=index;break;}lower=index;if(index===maximum)break;}}
      const refine=async(low,high)=>{while(high-low>1){const mid=Math.floor((low+high)/2);if(await qualifies(mid))high=mid;else low=mid;}return high;};
      if(upper!==null)best=await refine(lower,upper);
      const limit=best??maximum;
      await Promise.all([...new Set(Array.from({length:17},(_,i)=>Math.round(limit*i/16)))].map(probe));
      const results=await Promise.all([...cache].map(async([index,pending])=>[index,await pending]));
      const passing=results.filter(([,readiness])=>readiness>=targetReadiness).map(([index])=>index);
      if(passing.length){best=Math.min(...passing);const low=Math.max(0,...results.filter(([index,readiness])=>index<best&&readiness<targetReadiness).map(([index])=>index));best=await refine(low,best);
        await Promise.all([1,2,3,4].map(offset=>best-offset).filter(index=>index>=0).map(probe));
        for(const [index,pending] of cache)if(index<best&&(await pending)>=targetReadiness)best=index;
      }
    }
    points.push({age,annualSavings:best===null?null:best*500,readiness:best===null?null:await probe(best),tested:best!==null,simulationCount:best===null?0:plan.count,atSearchLimit:best===maximum,evaluatedCandidates:cache.size});if(best!==null)hint=best;report();
  }
  return {kind:'savings',targetReadiness:plan.targetReadiness,simulationCount:plan.count,searchStartAge:plan.firstAge,searchEndAge:plan.lastAge,fixedAnnualSpending:s.spending.annualBaseSpending,savingsSearchLimit:maximum*500,savingsIncrement:500,allocation:savingsAllocation(s).filter(x=>x.share>0),adaptiveSearch:true,alreadyRetired:s.household.alreadyRetired,checkedCandidates,points};
}

// A bounded adaptive search, not an exhaustive spending maximum. Neighboring
// ages supply a starting guess. Wider probes also look for qualifying amounts
// above local dips, because spending-dependent allocation is not monotonic.
export async function searchReadinessFrontier(s,{evaluate,targetReadiness=.8,simulationCount=TARGET_SIMULATION_PATHS,maxRetirementAge=70,initialSpending=s.spending.annualBaseSpending,onProgress,kind='retirement'}={}){
  const plan=decisionPlan(s,targetReadiness,simulationCount,maxRetirementAge),run=evaluate??(task=>candidateReadiness(s,task));
  if(kind==='claiming'){
    plan.firstAge=Math.max(62,plan.firstAge);plan.lastAge=Math.min(70,s.household.targetEndAge-1);plan.ages=[];
    for(let age=plan.firstAge;age<=plan.lastAge;age++)plan.ages.push(age);
  }
  const minimum=plan.amounts.at(-1)/500,maximum=plan.maximumCandidate/500,points=[];
  let hint=Math.round(initialSpending/500),checkedCandidates=0,age=null;
  const report=()=>onProgress?.({phase:'frontier',kind,completedAges:points.length,totalFrontierAges:plan.ages.length,age,checkedCandidates});
  report();
  for(age of plan.ages){
    report();const cache=new Map();
    const probe=index=>{
      if(!cache.has(index))cache.set(index,Promise.resolve().then(()=>run({kind:kind==='claiming'?'claim-frontier':'frontier',age,value:index*500,count:plan.count,targetReadiness})).then(readiness=>{checkedCandidates++;report();return readiness;}));
      return cache.get(index);
    };
    const qualifies=async index=>(await probe(index))>=targetReadiness;
    let best=null;
    if(plan.amounts.length&&await qualifies(maximum))best=maximum;
    else if(plan.amounts.length){
      const start=Math.max(minimum,Math.min(maximum,hint));
      let low=null,high=maximum;
      if(await qualifies(start)){
        low=start;
        for(let step=1;;step*=2){const index=Math.min(maximum,start+step);if(await qualifies(index))low=index;else{high=index;break;}}
      }else{
        high=start;
        for(let step=1;;step*=2){const index=Math.max(minimum,start-step);if(await qualifies(index)){low=index;break;}high=index;if(index===minimum)break;}
      }
      const refine=async(lower,upper)=>{
        while(upper-lower>1){const middle=Math.floor((lower+upper)/2);if(await qualifies(middle))lower=middle;else upper=middle;}
        return lower;
      };
      if(low!==null)low=await refine(low,high);
      // A failed local bracket does not rule out a higher qualifying island.
      const from=low===null?minimum:low+1;
      const wider=[...new Set(Array.from({length:17},(_,i)=>Math.round(from+(maximum-from)*i/16)))].filter(index=>index>=minimum&&index<=maximum);
      await Promise.all(wider.map(probe));
      const highestTested=async()=>{
        const results=await Promise.all([...cache].map(async([index,pending])=>[index,await pending]));
        const passing=results.filter(([,readiness])=>readiness>=targetReadiness).map(([index])=>index);
        if(!passing.length)return null;
        const lower=Math.max(...passing),upper=Math.min(...results.filter(([index,readiness])=>index>lower&&readiness<targetReadiness).map(([index])=>index));
        return Number.isFinite(upper)?refine(lower,upper):lower;
      };
      best=await highestTested();
      // Verify the immediate neighborhood too, including a small upward island
      // after one failed $500 amount. Never force the curve to increase with age.
      if(best!==null&&best<maximum){await Promise.all([1,2,3,4].map(offset=>best+offset).filter(index=>index<=maximum).map(probe));best=await highestTested();}
    }
    const point={age,annualSpending:best===null?null:best*500,readiness:best===null?null:await probe(best),tested:best!==null,simulationCount:best===null?0:plan.count,atSearchLimit:best===maximum,evaluatedCandidates:cache.size};
    points.push(point);if(best!==null)hint=best;report();
  }
  return {kind,targetReadiness:plan.targetReadiness,simulationCount:plan.count,searchStartAge:plan.firstAge,searchEndAge:plan.lastAge,spendingSearchLimit:plan.maximumCandidate,spendingIncrement:500,adaptiveSearch:true,checkedCandidates,points};
}
