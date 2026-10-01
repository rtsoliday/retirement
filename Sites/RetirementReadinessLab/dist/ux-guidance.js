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
export const FIELD_GUIDANCE={
  'household.filingStatus':'Household · Couples use Married filing status in this model. Single and Head of household model one person.',
  'household.birthday':'You · Use your date of birth, not a retirement age.',
  'household.retirementDate':'Household · In this saved shared-date plan, this one date starts retirement for both of you. Choose Use separate-person inputs to give your spouse their own date.',
  'household.spouseBirthday':'Spouse · Use their date of birth. Their ages for benefits, healthcare and longevity follow from it.',
  'accounts.pretax':'Household total today · Add traditional 401(k), 403(b), IRA and similar pre-tax balances from statements. No separate plan-specific rules or account owners are modeled.',
  'accounts.roth':'Household total today · Roth IRA balances only, including contributions and earnings. Employer Roth 401(k) rules are not modeled.',
  'accounts.taxable':'Household total today · Brokerage investments outside retirement accounts, from account statements. Capital-gains and dividend taxes are not modeled.',
  'accounts.cash':'Household total today · Bank savings and cash kept outside investments. Avoid counting the same money twice.',
  'rothHistory.contributionBasis':'Today’s dollars · Remaining regular Roth IRA contributions, from contribution and withdrawal records. Exclude conversions and investment growth; this is already part of your total Roth value.',
  'spending.annualBaseSpending':'Household · Annual living costs in today’s purchasing power. For $4,000 per month, enter $48,000. Include property tax and home insurance; exclude mortgage, rent and healthcare premiums entered separately. Use statements or the Budget builder.',
  'socialSecurity.annualBenefitAt67':'You · Annual benefit at age 67 in today’s purchasing power. Find your estimate at my Social Security. A $2,000 monthly estimate means $24,000 per year here.',
  'socialSecurity.claimAge':'You · The age when your own Social Security begins.',
  'socialSecurity.spouseClaimAge':'Spouse · In this saved shared-date plan, spousal/survivor benefits are derived from your benefit. Choose Use separate-person inputs to enter your spouse’s own earnings-based benefit.',
  'guaranteedIncome.annualIncome':'You · Annual pension or annuity payments in today’s purchasing power, from a benefit statement. $1,500 monthly means $18,000 yearly. Enter income, not the pension’s account or lump-sum value. One stream with your start age and survivor share is modeled here. Separate-person couples have an additional spouse pension input.',
  'guaranteedIncome.startAge':'You · Age when this pension or annuity starts paying. Add extra months if needed; use the benefit statement.',
  'guaranteedIncome.annualIncrease':'Annual % · From the pension’s payment-increase terms. This rate applies from today, including before payments begin. A future quoted payment needs adjustment to today’s dollars; do not also count it as today’s amount.',
  'guaranteedIncome.survivorPercent':'Spouse · Share of your pension retained after your death, from the survivor-benefit election. 50 means half; 0 means none.',
  'mortgage.monthlyPayment':'Household · Monthly principal and interest from your statement, excluding escrow. $1,200 each month stays $1,200 here.',
  'mortgage.currentBalance':'Household · Unpaid principal today from the mortgage statement.',
  'rent.monthlyRent':'Household · Monthly rent today from your lease; enter 0 if none.',
  'home.currentValue':'Household · Estimated sale value today, before the mortgage is deducted. It is separate from financial savings.',
  'home.annualTaxesAndInsurance':'Household · Annual property tax and home insurance today, from bills. Also include these in base spending; this identifies the portion that stops after a home sale.',
  'healthcare.preMedicareMonthlyPremium':'Each adult · Monthly premium today from your insurance quote or bill. Added for each retired adult under 65; use 0 only if no premium applies.',
  'longTermCare.annualCost':'Each person · Annual care cost today from a care-provider estimate, modeled only when the care-risk option is on.',
  'market.preRetirementMeanReturn':'Annual % · Investment growth only, before inflation. Enter new savings under Future savings, not here. The sample 13.3% is close to the S&P 500’s average yearly return with dividends over the last 50 years, so it assumes an all-stock portfolio. Balance growth on a statement also includes deposits; use the one-year statement check to separate them.',
  'market.stockMeanReturn':'Annual % · Average yearly stock return after retirement, before inflation and fees. The sample 13.3% (volatility 16.2%) is close to the S&P 500 with dividends over the last 50 years. Past returns are not a forecast; the full 1928–2025 average was 11.9%.',
  'market.bondMeanReturn':'Annual % · Assumed bond investment return after retirement, before inflation.',
  'spending.generalInflationMean':'Annual % · Estimated yearly price growth for living and housing costs. 2.3 means 2.3% per year.',
  'healthcare.healthcareInflationMean':'Annual % · Estimated yearly price growth for healthcare premiums and care costs.'
  ,'household.spouseRetirementDate':'Spouse · Their own last working date. Household costs start at the earlier retirement; their savings deposits and take-home support stop at their own retirement.',
  'spouseAccounts.pretax':'Spouse today · Their traditional 401(k), 403(b), IRA and similar pre-tax balances from statements. Exclude amounts entered under You. Non-Roth after-tax basis, employer-plan deferrals and special plan rules are not modeled.',
  'spouseAccounts.roth':'Spouse today · Their Roth IRA contributions plus conversions and earnings. Exclude your Roth balance. Employer Roth 401(k) rules are not modeled.',
  'spouseRothHistory.contributionBasis':'Spouse · Remaining regular Roth IRA contributions from records, excluding conversion principal and growth. Already included in their total Roth balance.',
  'spouseRothHistory.firstContributionYear':'Spouse · First Roth IRA funding tax year from contribution records. Enter 0 only if they have never funded a Roth IRA; new deposits start its five-tax-year clock.',
  'spouseIncome.annualBenefitAt67':'Spouse · Their own annual age-67 estimate in today’s purchasing power from my Social Security. $1,500 monthly means $18,000 yearly. Enter 0 if no own benefit; the model can still calculate an eligible spousal benefit.',
  'spouseIncome.annualPension':'Spouse · Annual pension or annuity income in today’s purchasing power from their benefit statement. $1,000 monthly means $12,000 yearly. This is income, not a pension balance; enter 0 if none.',
  'spouseIncome.pensionStartAge':'Spouse · Age when their pension starts. Use the extra-month field for partial years.',
  'spouseIncome.annualIncrease':'Spouse pension · Annual payment increase, applied from today including before payments start. Adjust a future quoted payment back to today’s dollars.',
  'spouseIncome.survivorPercent':'You · Share of the spouse pension retained after their death. 50 means half; 0 means none. No survivor pension is paid when the pension owner dies before its start age.',
  'workingIncome.primaryAnnualNet':'You · Annual take-home amount available for household costs while still working after the spouse retires. Use pay statements and your budget; subtract taxes and these savings deposits first. $2,000 monthly means $24,000 yearly. Enter 0 if savings should fund the gap.',
  'workingIncome.spouseAnnualNet':'Spouse · Annual take-home amount available for household costs while still working after you retire, after taxes and these savings deposits. $2,000 monthly means $24,000 yearly. Enter 0 if none.',
  'workingIncome.annualIncrease':'Annual % · Growth of take-home household support from today. This does not calculate salary or employment taxes.'

};
for(const prefix of ['contributions','spouseContributions'])for(const key of ['pretax','roth','taxable','cash','employerPretax','annualIncrease']){
  const who=prefix==='contributions'?'You':'Spouse',account={pretax:'employee pre-tax',roth:'Roth IRA',taxable:'shared taxable investment',cash:'shared cash',employerPretax:'employer pre-tax'}[key];
  FIELD_GUIDANCE[prefix+'.'+key]=key==='annualIncrease'?who+' · Annual % increase in these deposits from today; 0 keeps the nominal amount unchanged.':`${who} · Annual ${account} deposits from ${key==='roth'?'IRA contribution':'payroll or savings'} records. $500 monthly means $6,000 yearly. Enter 0 if none; choose Unknown if not known.`;
}
export function unknownInputPaths(s,sources){
  return Object.entries(sources[s.id]||{}).filter(([path,source])=>{
    if(path==='_origin'||source!=='Unknown')return false;
    if(s.household.filingStatus!=='Married'&&(path.startsWith('spouse')||path.startsWith('workingIncome')||['household.spouseBirthday','household.spouseGender','household.spouseRetirementDate','socialSecurity.spouseClaimAge','guaranteedIncome.survivorPercent'].includes(path)))return false;
    if(!s.household.separatePeople&&(path.startsWith('spouseAccounts')||path.startsWith('spouseRothHistory')||path.startsWith('spouseIncome')||path.startsWith('spouseWithdrawal')||path.startsWith('workingIncome')||path==='household.spouseRetirementDate'))return false;
    return true;
  }).map(([path])=>path);
}
