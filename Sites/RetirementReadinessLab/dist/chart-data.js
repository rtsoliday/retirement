// Matches Android RetirementSimulator.buildPathPoints: classify against all paths,
// then sample deterministically without consuming the simulation random stream.
export function buildPathPoints(paths,maxPoints=30000){
  const failedMax=[],successMin=[];let count=0;
  for(const path of paths)path.chart.forEach((balance,year)=>{if(balance>0){count++;if(path.success)successMin[year]=Math.min(successMin[year]??Infinity,balance);else failedMax[year]=Math.max(failedMax[year]??-Infinity,balance);}});
  const stride=Math.max(1,Math.ceil(count/maxPoints)),points=[];let index=0;
  for(const path of paths)path.chart.forEach((balance,year)=>{if(balance>0){if(index%stride===0)points.push({yearsInRetirement:year,balance,successfulPath:path.success,separatedFromOppositeOutcome:path.success?balance>(failedMax[year]??-Infinity):balance<(successMin[year]??Infinity)});index++;}});
  return points;
}
export function pathBounds(points,mean){
  const values=[...points,...mean].map(p=>p.balance).filter(v=>v>0&&Number.isFinite(v));
  const smallest=values.reduce((a,b)=>Math.min(a,b),Infinity);
  const min=Math.max(1,Number.isFinite(smallest)?smallest:1);
  const max=Math.max(min*10,values.reduce((a,b)=>Math.max(a,b),0));
  return {min,max,maxYear:Math.max(1,...[...points,...mean].map(p=>p.yearsInRetirement)),logMin:Math.log(min),logMax:Math.log(max)};
}
export function valueToFraction(value,min,max,log=false){return log?(Math.log(Math.max(1,value))-Math.log(min))/(Math.log(max)-Math.log(min)):(value-min)/(max-min);}
export function fractionToValue(fraction,min,max,log=false){return log?Math.exp(Math.log(min)+fraction*(Math.log(max)-Math.log(min))):min+fraction*(max-min);}

// Annual observations only: no balances after a path's death or failure.
// A failure's zero endpoint replaces that age's opening balance in this annual view.
export function buildBalanceBands(paths,retirementAge){
  const byAge=new Map();
  for(const path of paths){
    const observations=new Map();
    path.chart.forEach((balance,year)=>{const age=retirementAge+year;if(age<=path.survivedThroughAge&&(path.failureAge===null||age<=path.failureAge))observations.set(age,Math.max(0,balance));});
    if(path.failureAge!==null)observations.set(path.failureAge,0);
    for(const [age,balance] of observations){if(!byAge.has(age))byAge.set(age,[]);byAge.get(age).push(balance);}
  }
  return [...byAge].sort(([a],[b])=>a-b).map(([age,values])=>{
    values.sort((a,b)=>a-b);const q=p=>values[Math.min(values.length-1,Math.round((values.length-1)*p))];
    return {age,pessimistic:q(.1),median:q(.5),optimistic:q(.9),pathCount:values.length};
  });
}
