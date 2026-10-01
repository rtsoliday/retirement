// Contributions are externally funded deposits, not investment returns.
export const SAVINGS_KEYS=['pretax','roth','taxable','cash'];
export const savingsDefaults=()=>({pretax:0,roth:0,taxable:0,cash:0,employerPretax:0,annualIncrease:0});
export function monthlySavings(settings,months){
  const factor=Math.pow(1+settings.annualIncrease,months/12)/12;
  return {pretax:(settings.pretax+settings.employerPretax)*factor,roth:settings.roth*factor,taxable:settings.taxable*factor,cash:settings.cash*factor};
}
export function depositSavings(b,ledger,settings,months,taxYear){
  const deposits=monthlySavings(settings,months);
  for(const key of SAVINGS_KEYS)b[key]+=deposits[key];
  if(deposits.roth){ledger.openingBalance+=deposits.roth;if(!ledger.firstContributionYear)ledger.firstContributionYear=taxYear;}
  return deposits;
}
export function hasFutureSavings(s){return [s.contributions,...(s.household.filingStatus==='Married'?[s.spouseContributions]:[])].some(x=>x&&['pretax','roth','taxable','cash','employerPretax'].some(k=>x[k]>0));}
