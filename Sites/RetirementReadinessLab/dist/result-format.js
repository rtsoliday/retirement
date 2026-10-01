import {FREE_SIMULATION_PATHS} from './model.js';
// A small preview is a count of examples, not a readiness estimate.
export const PREVIEW_WARNING = 'Only a small sample of lifetimes is shown. This is too small a sample to estimate retirement readiness. If every sample ends without a shortfall, that still does not mean your plan is certain to succeed. Use a larger run and review the assumptions before drawing conclusions.';
export function isPreviewResult(result){return Boolean(result && result.provenance.simulationCount<=FREE_SIMULATION_PATHS);}
export function shareLabel(share,count){return count<=FREE_SIMULATION_PATHS?`${Math.round(share*count)} of ${count}`:`${(100*share).toFixed(1)}%`;}
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
