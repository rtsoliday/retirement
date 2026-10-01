import {FREE_SIMULATION_PATHS} from './model.js';
// A small preview is a count of examples, not a readiness estimate.
export const PREVIEW_WARNING = 'Only a small sample of lifetimes is shown. This is too small a sample to estimate retirement readiness. Zero observed outcomes does not mean an outcome is impossible. If every sample ends without a shortfall, that still does not mean your plan is certain to succeed. Use a larger run and review the assumptions before drawing conclusions.';
export function isPreviewResult(result){return Boolean(result && result.provenance.simulationCount<=FREE_SIMULATION_PATHS);}
export function shareLabel(share,count){
  if(share===0)return `0 of ${count} observed`;
  if(count<=FREE_SIMULATION_PATHS)return `${Math.round(share*count)} of ${count}`;
  const rounded=(100*share).toFixed(1);
  return rounded==='0.0'?'<0.1%':rounded==='100.0'&&share<1?'>99.9%':`${rounded}%`;
}
export function readinessLabel(result){return shareLabel(result.successProbability,result.provenance.simulationCount);}

// Use one actual snapshot per whole-year age, keeping its statistics together.
// Averaging monthly percentiles or summing path counts would distort the result.
export function ageYearRows(points){
  const years=new Map();
  for(const point of points){
    const year=Math.floor(Math.round(point.age*12)/12),previous=years.get(year);
    if(!previous||point.age>previous.age)years.set(year,point);
  }
  return [...years].sort(([a],[b])=>a-b).map(([ageYear,point])=>({...point,ageYear}));
}

// A missing observed balance is not a zero-dollar outcome. Successful deaths
// stay in the all-path coverage curve, but cannot supply survivor finances.
export function hasLivingOutcomeAtAge(result,age){
  const points=result.notFailedByAge||[];
  let latest=null;for(const p of points){if(p.age>age)break;latest=p;}
  return latest===null||latest.aliveShare>0;
}
export function balanceDisplayRows(result){
  const bands=new Map(ageYearRows(result.balanceBands||[]).map(b=>[b.ageYear,b]));
  const lifespans=new Map(ageYearRows(result.notFailedByAge||[]).map(p=>[p.ageYear,p]));
  const years=new Set([...bands.keys(),...lifespans.keys()]);
  const steady=result.steadySimulation?.monthlyDetails||[];
  if(steady.length)for(let age=Math.floor(steady[0].age);age<=Math.floor(steady.at(-1).age);age++)years.add(age);
  return [...years].sort((a,b)=>a-b).map(ageYear=>{
    const band=bands.get(ageYear);
    // A final death observation can be later in the same year than the last
    // balance. Keep the annual tables consistent without carrying that earlier
    // balance past the end of every modeled lifetime.
    if(lifespans.get(ageYear)?.aliveShare===0)return {ageYear,pathCount:0,noOutcomes:true};
    return band?{...band,noOutcomes:!band.pathCount||!hasLivingOutcomeAtAge(result,band.age)}:{ageYear,pathCount:0,noOutcomes:true};
  });
}
