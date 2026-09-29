import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,validateScenario,applyBudgetEstimate} from '../dist/model.js';
import {runOne,runSimulation} from '../dist/engine.js';
import {ordinaryIncomeTax,taxableSocialSecurity} from '../dist/tax.js';
import {retirementBenefitFactor} from '../dist/social-security.js';

function flatPlan(age=62,years=1){
  const s=baseScenario();
  Object.assign(s.household,{currentAge:age,retirementAge:age,targetEndAge:age+years});
  s.accounts={pretax:0,roth:100000,taxable:0,cash:0};
  Object.assign(s.spending,{annualBaseSpending:30000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  Object.assign(s.market,{preRetirementMeanReturn:0,preRetirementStdDev:0,stockMeanReturn:0,stockStdDev:0,bondMeanReturn:0,bondStdDev:0});
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});
  s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;
  return s;
}
const fullLife={nextDouble:()=>.999999,normal:()=>0};
function ending(s,rng=fullLife){assert.deepEqual(validateScenario(s),[]);return runOne(s,rng).yearEnd.at(-1);}
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<.01,`${actual} != ${expected}`);

test('a pretax-to-Roth transition below the annual deduction stays funded',()=>{
  const s=flatPlan();s.accounts={pretax:16100,roth:14600,taxable:0,cash:0};
  const result=runSimulation(s);assert.equal(result.successProbability,1);near(result.medianEndingBalance,700);
});

test('partial-year pretax depletion charges exactly the annual tax on actual income',()=>{
  for(const pretax of [10000,16100,30000,50000]){
    const s=flatPlan();s.accounts.pretax=pretax;s.spending.annualBaseSpending=60000;
    near(ending(s),100000+pretax-60000-ordinaryIncomeTax(pretax,'Single'));
  }
});

test('annual tax tracking resets at the year boundary',()=>{
  const s=flatPlan(62,2);s.accounts={pretax:60000,roth:5000,taxable:0,cash:0};
  // $30,000 of annual ordinary income has $1,420 tax under this table.
  s.spending.annualBaseSpending=28580;
  const path=runOne(s,fullLife);assert.equal(path.success,true);
  near(path.yearEnd[1],35000);near(path.yearEnd[2],5000);
});

test('Social Security taxation uses actual annual pretax draws and pension income',()=>{
  const s=flatPlan(67);s.accounts.pretax=20000;s.socialSecurity.annualBenefitAt67=30000;
  s.guaranteedIncome.annualIncome=10000;s.spending.annualBaseSpending=80000;
  const ordinary=30000,tax=ordinaryIncomeTax(ordinary+taxableSocialSecurity(ordinary,30000,'Single'),'Single',1,1,2026);
  near(ending(s),120000-80000+40000-tax);
});

test('early withdrawal penalty applies only to actual pretax draws',()=>{
  const s=flatPlan(55);s.accounts.pretax=10000;s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  near(ending(s),110000-30000-1000);
});

test('year-end Roth conversion charges only the additional annual liability',()=>{
  const s=flatPlan();s.accounts={pretax:100000,roth:0,taxable:0,cash:0};
  Object.assign(s.rothConversion,{enabled:true,marginalRateCap:.12});
  // Filling the 12% bracket produces $66,500 total ordinary income and $5,800 tax.
  near(ending(s),100000-30000-5800);
});

test('SEPP distributions enter annual income once without an early penalty',()=>{
  const s=flatPlan(55);s.accounts.pretax=100000;s.withdrawalStrategy.seppEligible=true;
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  // Spending exceeds the mandatory distribution, so total pretax draws simply
  // cover net spending plus ordinary tax; the SEPP portion has no 10% penalty.
  const withoutPenalty=structuredClone(s);withoutPenalty.withdrawalStrategy.applyEarlyWithdrawalPenalty=false;
  assert.ok(ending(s)<ending(withoutPenalty));
  const noSepp=structuredClone(s);noSepp.withdrawalStrategy.seppEligible=false;
  assert.ok(ending(s)>ending(noSepp));
});

test('mortgage payments continue during full-household long-term care',()=>{
  const s=flatPlan(65);s.accounts.roth=200000;s.spending.annualBaseSpending=0;
  Object.assign(s.mortgage,{monthlyPayment:1000,yearsLeft:1,currentBalance:12000});
  s.longTermCare.enabled=true;
  near(ending(s,{nextDouble:()=>0,normal:()=>0}),88000);
});

test('mortgage payments stop at the end of the loan term during care',()=>{
  const s=flatPlan(65);s.accounts.roth=200000;s.spending.annualBaseSpending=0;
  Object.assign(s.mortgage,{monthlyPayment:1000,monthsLeft:6,currentBalance:6000});s.longTermCare.enabled=true;
  near(ending(s,{nextDouble:()=>0,normal:()=>0}),94000);
});

test('full-household care sells the home before expenses, pays its mortgage and releases equity once',()=>{
  const s=flatPlan(65,2);s.accounts.roth=300000;s.home.currentValue=500000;
  s.budget.annualPropertyTaxes=10000;s.budget.annualHomeInsurance=2000;
  s.budget.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:[],creditCardBills:[]}];
  applyBudgetEstimate(s);
  Object.assign(s.mortgage,{monthlyPayment:2000,yearsLeft:20,currentBalance:100000});
  s.longTermCare.enabled=true;
  const path=runOne(s,{nextDouble:()=>.1,normal:()=>0});
  // Death is capped at 67 and care starts immediately. No mortgage, home bills,
  // or replacement rent remain; the $400,000 equity earns 2% in cash.
  near(path.yearEnd[1],200000+400000*1.02);
  near(path.yearEnd[2],100000+400000*1.02**2);
});

test('home equity can fund the first month of care even with no starting financial accounts',()=>{
  const s=flatPlan(65);s.accounts.roth=0;s.home.currentValue=200000;
  Object.assign(s.mortgage,{monthlyPayment:1000,yearsLeft:10,currentBalance:50000});
  s.longTermCare.enabled=true;
  let cash=150000;for(let m=0;m<12;m++)cash=cash*Math.pow(1.02,1/12)-100000/12;
  const path=runOne(s,{nextDouble:()=>0,normal:()=>0});
  assert.equal(path.success,true);near(path.yearEnd[1],cash);
});

test('two living spouses in care sell the home without adding replacement rent',()=>{
  const s=flatPlan(65);Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:65});
  s.accounts.roth=500000;s.home.currentValue=500000;
  Object.assign(s.mortgage,{monthlyPayment:2000,yearsLeft:20,currentBalance:100000});
  s.longTermCare.enabled=true;
  near(ending(s,{nextDouble:()=>0,normal:()=>0}),500000-200000+400000*1.02);
});

test('a spouse outside care keeps the home and its mortgage and budgeted carrying costs',()=>{
  const s=flatPlan(65);Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:65});
  s.accounts.roth=300000;s.spending.annualBaseSpending=12000;s.home.currentValue=500000;
  Object.assign(s.budget,{isAppliedToAnnualBaseSpending:true,appliedAnnualHomeCosts:12000});
  Object.assign(s.mortgage,{monthlyPayment:1000,yearsLeft:1,currentBalance:12000});
  s.longTermCare.enabled=true;
  const draws=[0,.999999,0,.999999];
  near(ending(s,{nextDouble:()=>draws.shift(),normal:()=>0}),176000);
});

test('the home sells when the last surviving spouse enters care, after being retained for that spouse',()=>{
  const s=flatPlan(65,3);Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:65});
  s.accounts.roth=500000;s.home.currentValue=500000;
  Object.assign(s.mortgage,{monthlyPayment:1000,yearsLeft:10,currentBalance:120000});
  Object.assign(s.longTermCare,{enabled:true,averageDurationYears:1});
  const draws=[.999999,0,.999999,.999999,0,0,0];
  const path=runOne(s,{nextDouble:()=>draws.shift(),normal:()=>0});
  // Primary dies at 67, spouse at 68. At 65 neither is in care; at 66 only
  // the primary is. At 67 the surviving spouse is in care and the house sells.
  near(path.yearEnd[1],500000-30000-12000);
  near(path.yearEnd[2],500000-2*(30000+12000)-100000);
  near(path.yearEnd[3],path.yearEnd[2]-100000+(500000-96000)*1.02);
});

test('the entered age-67 benefit is preserved across birth cohorts',()=>{
  for(const age of [67,70,75,85]){
    const s=flatPlan(age);s.spending.annualBaseSpending=40000;s.socialSecurity.annualBenefitAt67=30000;
    near(ending(s),90000);
  }
});

test('early and delayed claims adjust relative to the entered age-67 amount',()=>{
  for(const claimAge of [62,66,70]){
    const s=flatPlan(70);s.spending.annualBaseSpending=50000;
    Object.assign(s.socialSecurity,{annualBenefitAt67:30000,claimAge});
    const benefit=30000*retirementBenefitFactor(1956,claimAge*12)/retirementBenefitFactor(1956,804);
    near(ending(s),50000+benefit);
  }
});

test('spousal and survivor payments share the corrected primary insurance amount',()=>{
  const s=flatPlan(70,4);Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:67});
  s.accounts.roth=500000;s.spending.annualBaseSpending=60000;s.socialSecurity.annualBenefitAt67=30000;
  let draws=0;const path=runOne(s,{nextDouble:()=>draws++===0?0:.999999,normal:()=>0});
  const spouse=30000/retirementBenefitFactor(1956,804)*.5;
  near(path.yearEnd[1],500000-60000+30000+spouse);
  // After the worker dies, the survivor receives the worker's age-67 benefit.
  near(path.yearEnd[1]-path.yearEnd[2],60000*.84-30000);
});

test('initial Medicare lookback does not count Roth spending as ordinary income',()=>{
  const s=flatPlan(65,3);s.accounts.roth=1e6;s.spending.annualBaseSpending=150000;s.healthcare.includeMedicarePremiums=true;
  const premium=(202.90+38.99)*12,path=runOne(s,fullLife);
  for(let year=1;year<=3;year++)near(path.yearEnd[year],1e6-year*(150000+premium));
});

test('initial Medicare estimate caps taxable income at available pretax savings',()=>{
  const s=flatPlan(65);s.accounts.pretax=10000;s.accounts.roth=1e6;
  s.spending.annualBaseSpending=150000;s.healthcare.includeMedicarePremiums=true;
  near(ending(s),1010000-150000-(202.90+38.99)*12);
});

test('Medicare still charges a surcharge for high taxable pension income',()=>{
  const s=flatPlan(65);s.guaranteedIncome.annualIncome=150000;s.spending.annualBaseSpending=200000;
  s.healthcare.includeMedicarePremiums=true;
  const premium=(202.90+38.99+202.90+37.50)*12;
  near(ending(s),100000+150000-200000-premium-ordinaryIncomeTax(150000,'Single',1,1,2026));
});

test('Medicare retains the lookback filing status for two years after a spouse dies',()=>{
  const s=flatPlan(65,7);Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:65});
  s.accounts.roth=2000000;s.spending.annualBaseSpending=250000;
  Object.assign(s.guaranteedIncome,{annualIncome:200000,startAge:65,annualIncrease:0,survivorPercent:1});
  s.healthcare.includeMedicarePremiums=true;
  const rng=()=>{let calls=0;return {nextDouble:()=>++calls===3?0:.999999,normal:()=>0};};
  const withMedicare=runOne(s,rng());s.healthcare.includeMedicarePremiums=false;
  const withoutMedicare=runOne(s,rng());
  const charges=withMedicare.yearEnd.map((balance,i)=>withoutMedicare.yearEnd[i]-balance);
  const basic=(202.90+38.99)*12,singleHigh=(202.90+38.99+324.60+60.40)*12;
  for(let year=1;year<=7;year++)near(charges[year]-charges[year-1],year<=3?basic*2:year<=5?basic:singleHigh);
});

test('initial Medicare estimates follow cash-first triggers and cap spending covered by cash',()=>{
  for(const [cashFirst,cash,trigger,surcharge] of [[true,1000000,.01,0],[true,1000000,-.01,385],[false,1000000,.01,385],[true,10000,.01,385]]){
    const s=flatPlan(65,3);s.accounts={pretax:1000000,roth:0,taxable:0,cash};
    s.spending.annualBaseSpending=150000;s.healthcare.includeMedicarePremiums=true;
    Object.assign(s.withdrawalStrategy,{useCashReserveDuringDrawdowns:cashFirst,drawdownTrigger:trigger});
    const path=runOne(s,fullLife);
    if(cashFirst&&cash===1000000&&trigger>0){
      let expectedCash=cash;const growth=Math.pow(1.02,1/12),premium=(202.90+38.99)*12;
      for(let month=1;month<=36;month++){
        expectedCash=expectedCash*growth-(150000+premium)/12;
        if(month%12===0)near(path.yearEnd[month/12],1000000+expectedCash);
      }
    }else{
      // These cases really use pretax withdrawals: the initial estimate must
      // still charge the high-income tier rather than suppressing all IRMAA.
      const basic=structuredClone(s);basic.healthcare.includeMedicarePremiums=false;
      const noPremium=runOne(basic,fullLife);
      assert.ok(noPremium.yearEnd[1]-path.yearEnd[1]>(202.90+38.99+surcharge)*12);
    }
  }
});


test('an underwater care-triggered sale pays the full mortgage instead of forgiving debt',()=>{
  const s=flatPlan(65);s.spending.annualBaseSpending=0;s.accounts={pretax:0,roth:0,taxable:0,cash:200000};
  s.home.currentValue=100000;Object.assign(s.mortgage,{currentBalance:200000,monthlyPayment:1000,yearsLeft:10});
  Object.assign(s.longTermCare,{enabled:true,annualCost:0});
  near(ending(s,{nextDouble:()=>0,normal:()=>0}),102000);
});

test('an underwater sale funds its remaining payoff from pretax with ordinary tax',()=>{
  const s=flatPlan(65);s.spending.annualBaseSpending=0;s.accounts={pretax:200000,roth:0,taxable:0,cash:0};
  s.home.currentValue=100000;Object.assign(s.mortgage,{currentBalance:200000,monthlyPayment:1000,yearsLeft:10});
  Object.assign(s.longTermCare,{enabled:true,annualCost:0});
  const balance=ending(s,{nextDouble:()=>0,normal:()=>0}),draw=200000-balance;
  near(draw-ordinaryIncomeTax(draw,'Single',1,1,2026),100000);
});

test('an unaffordable underwater sale fails even with no other care expenses',()=>{
  const s=flatPlan(65);s.accounts.roth=0;s.spending.annualBaseSpending=0;s.home.currentValue=100000;
  Object.assign(s.mortgage,{currentBalance:200000,monthlyPayment:1000,yearsLeft:10});
  Object.assign(s.longTermCare,{enabled:true,annualCost:0});
  const path=runOne(s,{nextDouble:()=>0,normal:()=>0});assert.equal(path.success,false);assert.equal(path.failureAge,65);
});

test('a care duration of one year and six months charges eighteen months',()=>{
  const s=flatPlan(65,2);s.spending.annualBaseSpending=0;s.accounts.roth=500000;
  Object.assign(s.longTermCare,{enabled:true,annualCost:100000,averageDurationYears:1,averageDurationMonths:6});
  near(ending(s,{nextDouble:()=>.1,normal:()=>0}),350000);
});

test('pension starts at the selected month rather than the next birthday',()=>{
  const s=flatPlan(65,2);s.spending.annualBaseSpending=0;
  Object.assign(s.guaranteedIncome,{annualIncome:12000,startAge:65,startAgeMonths:6});
  let cash=0;for(let m=0;m<24;m++)cash=cash*Math.pow(1.02,1/12)+(m>=6?1000:0);
  near(ending(s),100000+cash);
});

test('retirement months grow accounts before retirement and retain the final partial-year balance',()=>{
  for(const months of [1,6,11]){
    const s=flatPlan(65);s.household.retirementAgeMonths=months;s.spending.annualBaseSpending=0;
    s.accounts={pretax:0,roth:0,taxable:0,cash:100000};
    const path=runOne(s,fullLife);near(path.yearEnd[0],100000*Math.pow(1.02,months/12));near(path.yearEnd.at(-1),102000);
    assert.equal(path.chart.length,1);assert.deepEqual(validateScenario(s),[]);
    const result=runSimulation(s);near(result.medianEndingBalance,102000);assert.equal(result.balanceBands[0].age,65+months/12);
  }
});

test('fractional retirement timing changes healthcare and penalties at birthdays and half-birthdays',()=>{
  const s=flatPlan(64,2);s.household.retirementAgeMonths=6;s.healthcare.preMedicareMonthlyPremium=1000;s.spending.annualBaseSpending=0;
  near(ending(s),94000); // Six pre-Medicare months, then no premiums.
  const early=flatPlan(59);early.household.retirementAgeMonths=6;early.accounts={pretax:10000,roth:0,taxable:0,cash:0};
  early.spending.annualBaseSpending=12000;early.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  near(ending(early),4000); // All six retirement months are after age 59.5.
});

test('a married plan with retirement months uses spouse mortality and benefit timing correctly',()=>{
  const s=flatPlan(65,2);s.household.retirementAgeMonths=6;Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:62});
  s.spending.annualBaseSpending=0;Object.assign(s.socialSecurity,{annualBenefitAt67:12000,claimAge:65,spouseClaimAge:62});
  // The primary claims at 65; a spouse aged 62.5 is immediately eligible.
  const path=runOne(s,fullLife);assert.equal(path.success,true);assert.ok(path.yearEnd[1]>100000+12000);
});
