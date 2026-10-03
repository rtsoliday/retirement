import {budgetBreakdown,scenarioTimeline,forecastRetirementDate,localCalendarDate,calendarDate,calendarMonthsBetween,addCalendarMonths,MAX_DOLLAR_AMOUNT} from './model.js';
import {inputSource} from './ux-guidance.js';
import {parseMoneyInput} from './money-input.js';
import {retirementBenefitFactor} from './social-security.js';

// A draft calculator updates only base spending when explicitly applied. The
// deducted amounts remain separate retirement inputs, not automatic overrides.
export function calculateEverydaySpending(draft){
  const value=raw=>String(raw??'').trim()?parseMoneyInput(raw):NaN;
  const total=value(draft.total),deductions={};
  for(const key of ['mortgage','rent','healthcare'])deductions[key]=draft.included?.[key]?value(draft[key]):0;
  const amounts=[total,...Object.values(deductions)];
  if(amounts.some(v=>!Number.isFinite(v)||v<0||v>MAX_DOLLAR_AMOUNT/12))return {error:'Enter a monthly total and an amount of 0 or more for each included payment.'};
  const deducted=Object.values(deductions).reduce((sum,v)=>sum+v,0),difference=total-deducted;
  if(difference < -Number.EPSILON*Math.max(1,total,deducted)*2)return {error:'Included payments cannot exceed total monthly spending.'};
  const monthly=Math.max(0,difference);
  return {total,deductions,deducted,monthly,annual:monthly*12};
}

// Reuse a completed run's steady-growth illustration. Do not mix a median
// ending balance with cash flows from another sampled lifetime.
export function monthlyIncomeSummary(s,r,basis='today'){
  const row=r?.steadySimulation?.monthlyDetails?.find(p=>p.cashFlow);
  if(!row)return null;
  // Cash flows belong to the interval before this end-of-month balance row.
  const index=basis==='today'?r.todayDollars?.steadyPriceIndexes?.[row.month-1]:1;
  if(!Number.isFinite(index)||index<=0)return null;
  const c=row.cashFlow,divide=value=>(value||0)/index;
  const timeline=scenarioTimeline(s),start=timeline.startDate||forecastRetirementDate(s),bridges=[];
  const stream=(label,amount,birthday,age)=>{
    if(!(amount>0)||!calendarDate(birthday)||!calendarDate(start))return;
    const date=addCalendarMonths(birthday,Math.round(age*12)),months=Math.round(age*12)-calendarMonthsBetween(birthday,start);
    if(months>0)bridges.push({label,date,months});
  };
  stream('Your Social Security',s.socialSecurity.annualBenefitAt67,s.household.birthday,s.socialSecurity.claimAge);
  stream('Your pension',s.guaranteedIncome.annualIncome,s.household.birthday,s.guaranteedIncome.startAge+s.guaranteedIncome.startAgeMonths/12);
  if(s.household.separatePeople&&s.household.filingStatus==='Married'){
    stream('Spouse Social Security',s.spouseIncome.annualBenefitAt67,s.household.spouseBirthday,s.socialSecurity.spouseClaimAge);
    stream('Spouse pension',s.spouseIncome.annualPension,s.household.spouseBirthday,s.spouseIncome.pensionStartAge+s.spouseIncome.pensionStartAgeMonths/12);
  }
  if(s.household.filingStatus==='Married'&&calendarDate(start)){
    const separate=s.household.separatePeople,amounts=[s.socialSecurity.annualBenefitAt67,separate?s.spouseIncome.annualBenefitAt67:0],birthdays=[s.household.birthday,s.household.spouseBirthday],claims=[s.socialSecurity.claimAge,s.socialSecurity.spouseClaimAge],ages=[timeline.retirementAge,timeline.spouseAtRet],births=[timeline.birthYear,timeline.spouseBirthYear];
    const pia=amounts.map((amount,i)=>amount/retirementBenefitFactor(births[i],67*12));
    for(const i of separate?[0,1]:[1]){
      if(!calendarDate(birthdays[i])||!(pia[i]<pia[1-i]*.5))continue;
      const claim=Math.max(claims[i]*12,Math.round(ages[i]*12)+claims[1-i]*12-Math.round(ages[1-i]*12)),months=claim-Math.round(ages[i]*12);
      // An own benefit already listed at the same date also signals this start.
      if(months>0&&(!amounts[i]||claim>claims[i]*12))bridges.push({label:i===0?'Your Social Security from your spouse’s record':'Spouse Social Security from your record',date:addCalendarMonths(birthdays[i],claim),months});
    }
  }
  return {date:calendarDate(start)?start:row.date,expenses:divide(c.expenses),socialSecurity:divide(c.socialSecurity),pension:divide(c.guaranteedIncome),workingSupport:divide(c.workingSupport),income:divide(c.socialSecurity+c.guaranteedIncome+(c.workingSupport||0)),withdrawals:divide(c.additionalWithdrawal+c.seppDistribution+c.rmdDistribution),taxes:divide(c.incomeTax+c.earlyPenalty),unfunded:divide(row.unfundedAmount),bridges};
}

export const OPTIONAL_QUESTIONS={
  'your-pension':{label:'Your pension or annuity',step:2,required:['guaranteedIncome.annualIncome'],zero:['guaranteedIncome.annualIncome']},
  'spouse-pension':{label:'Spouse pension',step:2,required:['spouseIncome.annualPension'],zero:['spouseIncome.annualPension']},
  'home-inputs':{label:'Home ownership',step:3,required:['home.currentValue','home.annualTaxesAndInsurance'],zero:['home.currentValue','home.annualTaxesAndInsurance','mortgage.monthlyPayment','mortgage.currentBalance','mortgage.yearsLeft','mortgage.monthsLeft']},
  'mortgage-inputs':{label:'Mortgage',step:3,required:['mortgage.monthlyPayment','mortgage.currentBalance','mortgage.yearsLeft'],zero:['mortgage.monthlyPayment','mortgage.currentBalance','mortgage.yearsLeft','mortgage.monthsLeft']},
  'rent-inputs':{label:'Rent',step:3,required:['rent.monthlyRent'],zero:['rent.monthlyRent']}
};
export function normalizeOptionalAnswers(raw,scenarios){
  return Object.fromEntries(scenarios.map(s=>[s.id,Object.fromEntries(Object.entries(raw?.[s.id]||{}).filter(([id,value])=>Object.hasOwn(OPTIONAL_QUESTIONS,id)&&['yes','no','unsure'].includes(value)))]));
}
// These are explicit illustration assumptions, never a substitute for records.
// Keep entered conversions and any known contribution/funding-year answers.
export function temporaryRothValues(s,sources,prefix){
  const history=s[prefix],values={},latestYear=2026;
  if(inputSource(s,sources,prefix+'.contributionBasis')==='Unknown')values.contributionBasis=0;
  if(inputSource(s,sources,prefix+'.firstContributionYear')==='Unknown')values.firstContributionYear=Math.min(latestYear,...history.conversions.filter(l=>Number.isInteger(l.taxYear)&&l.taxYear>=1998&&l.taxYear<=latestYear).map(l=>l.taxYear));
  return values;
}

// A single route for progress counts, grouped review and direct edit links.
export function inputTask(path){
  if (/^(spouseAccounts|spouseRothHistory)\./.test(path)) return {key:'spouse-accounts',label:'Spouse’s accounts',step:1,task:0};
  if (/^(accounts\.(pretax|roth)|rothHistory\.)/.test(path)) return {key:'your-accounts',label:'Your accounts',step:1,task:0};
  if (path.startsWith('accounts.')) return {key:'shared-accounts',label:'Shared cash and investments',step:1,task:0};
  if (path.startsWith('spouseContributions.')) return {key:'spouse-savings',label:'Spouse’s future savings',step:1,task:2};
  if (path.startsWith('contributions.')) return {key:'your-savings',label:'Your future savings',step:1,task:2};
  if (path.startsWith('spending.')) return {key:'spending',label:'Everyday spending',step:1,task:1};
  if (path.startsWith('household.')) return {key:'household',label:'Household and retirement dates',step:0};
  if (/^(socialSecurity|spouseIncome|guaranteedIncome|workingIncome)\./.test(path)) return {key:'income',label:'Income and Social Security',step:2};
  if (/^(home|mortgage|rent)\./.test(path)) return {key:'housing',label:'Housing costs',step:3};
  if (/^(healthcare|longTermCare)\./.test(path)) return {key:'healthcare',label:'Healthcare costs',step:3};
  return {key:'strategy',label:'Market and strategy',step:4};
}

export function groupUnknownInputs(paths){
  const groups=new Map();
  for(const path of paths){const task=inputTask(path);if(!groups.has(task.key))groups.set(task.key,{...task,paths:[]});groups.get(task.key).paths.push(path);}
  return [...groups.values()].sort((a,b)=>a.step-b.step||(a.task??0)-(b.task??0));
}

// These checks connect an applied worksheet to its separate plan inputs. A
// typed zero explicitly records that the payment will not apply in retirement.
export function budgetPlanGaps(s,sources){
  if(!s.budget.isAppliedToAnnualBaseSpending||s.budget.estimateNeedsReview)return [];
  const months=budgetBreakdown(s.budget).months;
  return [
    ['mortgage','mortgage.monthlyPayment','mortgage payments'],
    ['rent','rent.monthlyRent','rent'],
    ['healthcare','healthcare.preMedicareMonthlyPremium','health insurance premiums']
  ].filter(([key,path])=>(key!=='healthcare'||spendingInputSummary(s,sources).preMedicareAdults>0)&&months.some(m=>m.adjustments?.[key]>0)&&!['Entered','Estimated'].includes(inputSource(s,sources,path)))
    .map(([key,path,label])=>({key,path,label}));
}

// Review entered costs at the first retirement, in today's dollars. Taxes,
// Medicare, inflation, age-based spending changes and care are computed per
// simulation, so they must not be presented as a fixed total here.
export function spendingInputSummary(s,sources){
  const t=scenarioTimeline(s,localCalendarDate()),couple=s.household.filingStatus==='Married';
  const preMedicareAdults=Number(t.retirementAge<65)+(couple?Number(t.spouseAtRet<65):0);
  const mortgageMonths=s.mortgage.yearsLeft*12+s.mortgage.monthsLeft;
  const mortgageActive=mortgageMonths>t.preMonths;
  const amount=(path,value)=>inputSource(s,sources,path)==='Unknown'?null:value;
  const base=amount('spending.annualBaseSpending',s.spending.annualBaseSpending/12);
  const mortgage=amount('mortgage.monthlyPayment',mortgageActive?s.mortgage.monthlyPayment:0);
  const rent=amount('rent.monthlyRent',s.rent.monthlyRent);
  const healthcare=preMedicareAdults?amount('healthcare.preMedicareMonthlyPremium',s.healthcare.preMedicareMonthlyPremium*preMedicareAdults):0;
  const values=[base,mortgage,rent,healthcare];
  return {base,mortgage,rent,healthcare,preMedicareAdults,medicareAdults:(couple?2:1)-preMedicareAdults,mortgageActive,total:values.includes(null)?null:values.reduce((sum,x)=>sum+x,0)};
}
