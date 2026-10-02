import {budgetBreakdown,scenarioTimeline,localCalendarDate} from './model.js';
import {inputSource} from './ux-guidance.js';

// These checks connect an applied worksheet to its separate plan inputs. A
// typed zero explicitly records that the payment will not apply in retirement.
export function budgetPlanGaps(s,sources){
  if(!s.budget.isAppliedToAnnualBaseSpending||s.budget.estimateNeedsReview)return [];
  const months=budgetBreakdown(s.budget).months;
  return [
    ['mortgage','mortgage.monthlyPayment','mortgage payments'],
    ['rent','rent.monthlyRent','rent'],
    ['healthcare','healthcare.preMedicareMonthlyPremium','health insurance premiums']
  ].filter(([key,path])=>months.some(m=>m.adjustments?.[key]>0)&&!['Entered','Estimated'].includes(inputSource(s,sources,path)))
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
