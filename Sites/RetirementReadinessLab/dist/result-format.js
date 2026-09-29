// A four-path preview is a count of examples, not a readiness estimate.
export const PREVIEW_WARNING = 'Only four sample lifetimes are shown. This is too small a sample to estimate retirement readiness. Even 4 of 4 without a shortfall does not mean your plan is certain to succeed. Use a larger run and review the assumptions before drawing conclusions.';
export function isPreviewResult(result){return Boolean(result && result.provenance.simulationCount<=4);}
export function shareLabel(share,count){return count<=4?`${Math.round(share*count)} of ${count}`:`${(100*share).toFixed(1)}%`;}
export function readinessLabel(result){return shareLabel(result.successProbability,result.provenance.simulationCount);}
