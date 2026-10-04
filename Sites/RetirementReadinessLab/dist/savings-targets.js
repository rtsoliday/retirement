// Employee-funded savings only. Employer deposits and the spouse's savings
// remain independent of the target. Amounts start now; existing growth rates stay.
export function savingsAllocation(s){
  const entries=['pretax','roth','taxable','cash'].map(key=>({path:'contributions.'+key,label:{pretax:'Pre-tax',roth:'Roth IRA',taxable:'Taxable investments',cash:'Cash'}[key],value:s.contributions[key]}));
  for(const [i,a] of (s.employerRothAccounts||[]).entries())if(a.owner==='you')entries.push({path:`employerRothAccounts.${i}.annualContribution`,label:a.name,value:a.annualContribution});
  const total=entries.reduce((sum,x)=>sum+x.value,0);
  return entries.map(x=>({...x,share:total>0?x.value/total:x.path==='contributions.cash'?1:0}));
}
export const annualPersonalSavings=s=>savingsAllocation(s).reduce((sum,x)=>sum+x.value,0);
export function setAnnualPersonalSavings(s,amount){
  for(const entry of savingsAllocation(s)){
    const keys=entry.path.split('.');let owner=s;for(const key of keys.slice(0,-1))owner=owner[key];
    owner[keys.at(-1)]=amount*entry.share;
  }
}
