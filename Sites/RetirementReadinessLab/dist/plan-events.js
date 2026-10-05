import {partTimeIncomeActive,homePlanActive} from './model.js';

// Plan Lab inputs and stress tests shared by the pooled and separate-person
// engines. Both engines skip this entirely when nothing is in use, so existing
// plans keep their exact calculations and random-draw order.
export function planEvents(s,stress=null){
  const work=partTimeIncomeActive(s)?s.partTimeIncome:null,home=homePlanActive(s)?s.homePlan:null;
  const expenses=new Map();
  for(const item of s.oneTimeExpenses||[])if(item.amount>0)expenses.set(item.age*12,(expenses.get(item.age*12)||0)+item.amount);
  const shock=stress?.marketDrop>0?Math.min(.95,stress.marketDrop):0,inflation=stress?.inflationShock?.months>0?stress.inflationShock:null;
  if(!work&&!home&&!expenses.size&&!shock&&!inflation&&!stress?.forceCare&&!(stress?.minDeathAge>0))return null;
  return {
    forceCare:Boolean(stress?.forceCare),minDeathAge:stress?.minDeathAge>0?stress.minDeathAge:0,home,
    // Take-home pay is entered in today's dollars and rises with general prices.
    work:(ageMonths,working,priceIndex)=>work&&working&&ageMonths<work.endAge*12?work.annualNet/12*priceIndex:0,
    oneTime:(ageMonths,anyoneAlive,priceIndex)=>anyoneAlive?(expenses.get(ageMonths)||0)*priceIndex:0,
    homeSaleDue:ageMonths=>Boolean(home)&&ageMonths===home.saleAge*12,
    // A one-time fall in stock prices in the first modeled month of retirement.
    marketReturn:(month,portReturn,stockPart)=>shock&&month===0?(1+portReturn)*(1-shock*stockPart)-1:portReturn,
    inflation:(month,draw,monthlyRate)=>inflation&&month<inflation.months?monthlyRate(inflation.rate):draw
  };
}

// Recurring costs set the investment mix; a one-time cost should not move a
// whole month into the lowest-savings allocation band.
export function applyMonthlyEvents(recurringNeed,work,oneTime){
  const covered=Math.min(work,recurringNeed),recurring=recurringNeed-covered,leftover=work-covered,oneTimeCovered=Math.min(leftover,oneTime);
  return {need:recurring+oneTime-oneTimeCovered,allocationNeed:recurring,surplus:leftover-oneTimeCovered};
}

// Lifetime totals for comparisons, in each path's own today's dollars.
export function pathMetrics(){
  const m={tax:0,conversions:0,surchargeYears:0,taxByYear:[],surcharge:false};
  return {
    data:m,
    markSurcharge(medCost,healthIndex,people){if(medCost>(202.90+38.99)*Math.max(0,healthIndex)*people*(1+1e-9))m.surcharge=true;},
    close(tax,conversions,priceIndex){const real=tax/Math.max(.0001,priceIndex);m.tax+=real;m.taxByYear.push(real);m.conversions+=conversions/Math.max(.0001,priceIndex);if(m.surcharge)m.surchargeYears++;m.surcharge=false;},
    result(){return {tax:m.tax,conversions:m.conversions,surchargeYears:m.surchargeYears,taxByYear:m.taxByYear};}
  };
}
