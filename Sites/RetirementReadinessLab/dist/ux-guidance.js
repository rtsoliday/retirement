// Presentation metadata lives alongside plans, never in calculation inputs.
export const INPUT_SOURCES=['Entered','Estimated','Unknown','Sample/default','Saved value; source not recorded'];
function scenarioValueExists(s,path){
  let value=s;
  for(const key of path.split('.')){if(!value||typeof value!=='object'||!Object.hasOwn(value,key))return false;value=value[key];}
  return ['number','string','boolean'].includes(typeof value);
}
export function normalizeInputSources(raw,scenarios,fallback='Saved value; source not recorded'){
  const result={};
  for(const s of scenarios){
    const notes=raw?.[s.id];result[s.id]={_origin:['Sample/default','Saved value; source not recorded'].includes(notes?._origin)?notes._origin:fallback};
    if(notes&&typeof notes==='object')for(const [path,source] of Object.entries(notes)){
      if(path!=='_origin'&&INPUT_SOURCES.includes(source)&&/^[a-zA-Z0-9.]+$/.test(path)&&scenarioValueExists(s,path))result[s.id][path]=source;
    }
  }
  return result;
}
export function inputSource(s,sources,path){return sources[s.id]?.[path]||sources[s.id]?._origin||'Sample/default';}
// The same four points appear as a list in setup and as one paragraph in review and reports.
export const EARNINGS_POINTS=[
  ['Salary is not an input.','Paychecks affect the plan only through the annual future savings deposits you enter (employee and employer), which stop when that person retires.'],
  ['Investment returns grow savings.','Returns apply to today’s balances and to new deposits. Cash earns 2% a year. Deposits are not returns, and returns are not deposits.'],
  ['In retirement, income pays first.','Social Security and pensions cover costs first; withdrawals from savings cover the rest, plus any taxes on those withdrawals.'],
  ['Still working after a spouse retires?','Couples with separate dates can enter take-home household support (after taxes and savings deposits) for the overlap.']
];
export const EARNINGS_EXPLANATION=EARNINGS_POINTS.map(([lead,text])=>lead+' '+text).join(' ');
// Short line shown under the input: who it covers, what to enter and where to find it.
// How the model uses the value belongs in the ? explanation (assumptionHelp in app.js).
export const FIELD_GUIDANCE={
  'household.filingStatus':'You · Choose Head of household only if you are unmarried and pay most household costs for a qualifying dependent.',
  'household.birthday':'You · Your date of birth, not a retirement age.',
  'household.retirementDate':'Household · One retirement date for both of you in this shared-date plan.',
  'household.spouseBirthday':'Spouse · Their date of birth.',
  'accounts.pretax':'Household total today · Add traditional 401(k), 403(b), IRA and similar pre-tax balances from statements.',
  'accounts.roth':'Household total today · All Roth IRA money: contributions, conversions and earnings.',
  'accounts.taxable':'Household total today · Brokerage investments outside retirement accounts, from statements.',
  'accounts.cash':'Household total today · Bank savings and cash outside investments. Avoid counting money twice.',
  'rothHistory.contributionBasis':'Regular contributions not yet withdrawn, from contribution records. Exclude conversions and growth.',
  'spending.annualBaseSpending':'Household · Yearly living costs in today’s dollars. For $4,000 per month, enter $48,000.',
  'socialSecurity.annualBenefitAt67':'You · Yearly benefit at 67 in today’s dollars. A $2,000 monthly estimate means $24,000 per year.',
  'socialSecurity.claimAge':'You · The age your own Social Security begins.',
  'socialSecurity.spouseClaimAge':'Spouse · Age they claim spousal benefits based on your record.',
  'guaranteedIncome.annualIncome':'You · Yearly payments in today’s dollars. $1,500 monthly means $18,000 yearly.',
  'guaranteedIncome.startAge':'You · Age payments start, from the benefit statement. Add extra months if needed.',
  'guaranteedIncome.annualIncrease':'Annual % · From the pension’s payment-increase terms; 0 if fixed.',
  'guaranteedIncome.survivorPercent':'Spouse · Share kept after your death, from your survivor election. 50 means half.',
  'mortgage.monthlyPayment':'Household · Principal and interest, excluding escrow. $1,200 each month stays $1,200 here.',
  'mortgage.currentBalance':'Household · Unpaid principal today, from your mortgage statement.',
  'rent.monthlyRent':'Household · Monthly rent today from your lease; enter 0 if none.',
  'home.currentValue':'Household · Estimated sale value today, before subtracting the mortgage.',
  'home.annualTaxesAndInsurance':'Household · Yearly property tax and home insurance, from bills. Keep them in base spending too; this amount only tells the model what stops if the home is sold.',
  'healthcare.preMedicareMonthlyPremium':'Each adult · Monthly premium today, from an insurance quote or bill.',
  'longTermCare.annualCost':'Each person · Yearly care cost today, from a care-provider estimate.',
  'market.preRetirementMeanReturn':'Annual % · Investment growth only, before inflation. New savings go under Future savings.',
  'market.stockMeanReturn':'Annual % · Average yearly stock return after retirement, before inflation and fees.',
  'market.bondMeanReturn':'Annual % · Average yearly bond return after retirement, before inflation.',
  'spending.generalInflationMean':'Annual % · Yearly price growth for living and housing costs. 2.3 means 2.3%.',
  'healthcare.healthcareInflationMean':'Annual % · Yearly price growth for healthcare premiums and care costs.',
  'household.spouseRetirementDate':'Spouse · Their own last working date. Their deposits and take-home support stop then.',
  'spouseAccounts.pretax':'Spouse today · Their traditional 401(k), 403(b), IRA and similar pre-tax balances. Exclude yours.',
  'spouseAccounts.roth':'Spouse today · Their Roth IRA contributions, conversions and earnings. Exclude yours.',
  'spouseRothHistory.contributionBasis':'Spouse · Regular contributions not yet withdrawn, from records. Exclude conversions and growth.',
  'spouseRothHistory.firstContributionYear':'Spouse · First Roth IRA funding tax year, from records. Enter 0 if never funded.',
  'spouseIncome.annualBenefitAt67':'Spouse · Their own yearly estimate at 67 in today’s dollars. $1,500 monthly means $18,000 yearly.',
  'spouseIncome.annualPension':'Spouse · Yearly pension income in today’s dollars. $1,000 monthly means $12,000 yearly.',
  'spouseIncome.pensionStartAge':'Spouse · Age their pension starts. Add extra months if needed.',
  'spouseIncome.annualIncrease':'Spouse pension · Annual payment increase; 0 if fixed.',
  'spouseIncome.survivorPercent':'You · Share of their pension you keep after their death. 50 means half.',
  'workingIncome.primaryAnnualNet':'You · Yearly take-home money for household costs while you work after your spouse retires.',
  'workingIncome.spouseAnnualNet':'Spouse · Yearly take-home money for household costs while they work after you retire.',
  'workingIncome.annualIncrease':'Annual % · Yearly growth of this take-home support from today.'
};
for(const prefix of ['contributions','spouseContributions'])for(const key of ['pretax','roth','taxable','cash','employerPretax','annualIncrease']){
  const who=prefix==='contributions'?'You':'Spouse',account={pretax:'employee pre-tax',roth:'Roth IRA',taxable:'shared taxable investment',cash:'shared cash',employerPretax:'employer pre-tax'}[key];
  FIELD_GUIDANCE[prefix+'.'+key]=key==='annualIncrease'?who+' · Yearly % increase in these deposits; 0 keeps them unchanged.':`${who} · Annual ${account} deposits. $500 monthly means $6,000 yearly.`;
}
export function unknownInputPaths(s,sources){
  return Object.entries(sources[s.id]||{}).filter(([path,source])=>{
    if(path==='_origin'||source!=='Unknown')return false;
    if(s.household.filingStatus!=='Married'&&(path.startsWith('spouse')||path.startsWith('workingIncome')||['household.spouseBirthday','household.spouseGender','household.spouseRetirementDate','socialSecurity.spouseClaimAge','guaranteedIncome.survivorPercent'].includes(path)))return false;
    if(!s.household.separatePeople&&(path.startsWith('spouseAccounts')||path.startsWith('spouseRothHistory')||path.startsWith('spouseIncome')||path.startsWith('spouseWithdrawal')||path.startsWith('workingIncome')||path==='household.spouseRetirementDate'))return false;
    return true;
  }).map(([path])=>path);
}
