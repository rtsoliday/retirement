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

// Callers sort once for their percentile summaries. Even-sized medians average
// the middle pair; the 10th/90th percentile selection rule stays unchanged.
export function medianOfSorted(values){
  if(!values.length)return 0;
  const mid=Math.floor(values.length/2);
  return values.length%2?values[mid]:values[mid-1]/2+values[mid]/2;
}

// Include annual observations and every monthly failure/death transition.
// Failed paths can still be alive; successful deaths remain funded. Sorting
// events once avoids rescanning thousands of paths at every monthly endpoint.
export function buildFundingSurvival(paths,retirementAge){
  if(!paths.length)return [];
  const deaths=paths.map(p=>Math.round(p.deathAge*12)).sort((a,b)=>a-b);
  const failures=paths.filter(p=>p.failureAge!==null).map(p=>Math.round(p.failureAge*12)).sort((a,b)=>a-b);
  const months=new Set([...deaths,...failures]);
  for(let month=Math.round(retirementAge*12);month<=deaths.at(-1);month+=12)months.add(month);
  let failed=0,died=0;
  return [...months].sort((a,b)=>a-b).map(month=>{
    while(failed<failures.length&&failures[failed]<=month)failed++;
    while(died<deaths.length&&deaths[died]<=month)died++;
    return {age:month/12,notFailedShare:(paths.length-failed)/paths.length,aliveShare:(paths.length-died)/paths.length};
  });
}

// Annual observations plus each failure's monthly zero endpoint. No balances
// are carried beyond a path's death or failure.
export function buildBalanceBands(paths,retirementAge){
  const byAge=new Map(),retirementMonths=Math.round(retirementAge*12);
  const failureMonths=new Set(paths.filter(path=>path.failureAge!==null).map(path=>Math.round(path.failureAge*12)));
  for(const path of paths){
    const observations=new Map();
    path.chart.forEach((balance,year)=>{const months=Math.round((retirementAge+year)*12),age=months/12;if(months<=Math.round(path.survivedThroughAge*12)&&(path.failureAge===null||months<=Math.round(path.failureAge*12)))observations.set(age,Math.max(0,balance));});
    // A monthly failure endpoint must also include every other path observed
    // in that month; otherwise its band would consist of failed paths alone.
    for(const months of failureMonths){const balance=path.monthlyBalances?.[months-retirementMonths];if(balance!==undefined)observations.set(months/12,Math.max(0,balance));}
    if(path.failureAge!==null)observations.set(path.failureAge,0);
    for(const [age,balance] of observations){if(!byAge.has(age))byAge.set(age,[]);byAge.get(age).push(balance);}
  }
  return [...byAge].sort(([a],[b])=>a-b).map(([age,values])=>{
    values.sort((a,b)=>a-b);const q=p=>values[Math.min(values.length-1,Math.round((values.length-1)*p))];
    return {age,pessimistic:q(.1),median:medianOfSorted(values),optimistic:q(.9),pathCount:values.length};
  });
}
