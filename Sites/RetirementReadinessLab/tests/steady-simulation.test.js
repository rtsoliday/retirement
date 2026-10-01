import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,addCalendarMonths,retirementAge} from '../dist/model.js';
import {runSimulation,runOne,JavaRandom,runSteadySimulation} from '../dist/engine.js';
import {mortgageAtRetirement,payMortgage} from '../dist/mortgage.js';

const stride=-7046029254386353131n;
const fullLife={nextDouble:()=>.999999,normal:()=>0};
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<.01,`${actual} != ${expected}`);
function flatPlan(){
  const s=baseScenario();
  Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:67});
  s.accounts={pretax:0,roth:100000,taxable:0,cash:0};s.rothHistory.contributionBasis=100000;
  Object.assign(s.spending,{annualBaseSpending:12000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  Object.assign(s.market,{preRetirementMeanReturn:0,preRetirementStdDev:0,stockMeanReturn:0,stockStdDev:0,bondMeanReturn:0,bondStdDev:0});
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,includeMedicarePremiums:false,healthcareInflationMean:0,healthcareInflationStdDev:0});
  s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;
  return s;
}

test('fixed lifespans use current age at the 85-year boundary and extend beyond lower model limits',()=>{
  for(const [age,expected] of [[60,95],[85,95],[86,96],[90,100],[118,128]]){
    const s=flatPlan();Object.assign(s.household,{currentAge:age,retirementAge:age,targetEndAge:age+1});s.spending.annualBaseSpending=0;
    const result=runSteadySimulation(s);
    assert.equal(result.primaryDeathAge,expected);assert.equal(result.spouseDeathAge,null);
    assert.equal(result.monthlyDetails.at(-1).age,expected);
    assert.equal(result.monthlyDetails.length,(expected-age)*12+1);
    assert.equal(result.endReason,'lifespan');
  }
});

test('zero volatility retains average returns, inflation and pre-retirement growth independently of seed',()=>{
  const s=flatPlan();Object.assign(s.household,{currentAge:60,retirementAge:62,targetEndAge:63});s.spending.annualBaseSpending=0;s.home.currentValue=100000;
  Object.assign(s.market,{preRetirementMeanReturn:.05,preRetirementStdDev:.5,stockMeanReturn:.06,stockStdDev:.5,bondMeanReturn:.06,bondStdDev:.3});
  Object.assign(s.spending,{generalInflationMean:.03,generalInflationStdDev:.2});s.healthcare.healthcareInflationStdDev=.2;
  s.longTermCare.enabled=true;
  const original=structuredClone(s),first=runSteadySimulation(s);
  assert.deepEqual(s,original);s.seed=12345;
  assert.deepEqual(runSteadySimulation(s),first);
  near(first.monthlyDetails[0].roth,100000*1.05**2);
  near(first.monthlyDetails[12].roth,100000*1.05**2*1.06);
  near(first.monthlyDetails[0].home,100000*1.03**2);
  near(first.monthlyDetails[12].home,100000*1.03**3);
  assert.ok(first.monthlyDetails.every(row=>row.home>0),'No care-triggered home sale');
});

test('married lifespans are set independently and survivor cash flows continue to the last death',()=>{
  const s=flatPlan();Object.assign(s.household,{currentAge:90,retirementAge:90,spouseCurrentAge:80,filingStatus:'Married',targetEndAge:91});
  s.spending.annualBaseSpending=0;s.guaranteedIncome.annualIncome=12000;s.guaranteedIncome.annualIncrease=0;s.guaranteedIncome.survivorPercent=.5;
  s.longTermCare.enabled=true;s.home.currentValue=200000;
  const result=runSteadySimulation(s),rows=result.monthlyDetails,cashRate=1.02**(1/12)-1;
  assert.equal(result.primaryDeathAge,100);assert.equal(result.spouseDeathAge,95);
  assert.equal(rows.length,181);assert.equal(rows.at(-1).age,105);
  near(rows[120].cash-rows[119].cash*(1+cashRate),1000);
  near(rows[121].cash-rows[120].cash*(1+cashRate),500);
  assert.ok(rows.every(row=>row.home===200000));
});

test('a younger primary continues after an older spouse reaches their independent ten-year lifespan',()=>{
  const s=flatPlan();Object.assign(s.household,{currentAge:60,retirementAge:60,spouseCurrentAge:90,filingStatus:'Married',targetEndAge:119});s.spending.annualBaseSpending=0;
  const result=runSteadySimulation(s);
  assert.equal(result.primaryDeathAge,95);assert.equal(result.spouseDeathAge,100);
  assert.equal(result.monthlyDetails.at(-1).age,95);
});

test('calendar lifespan rules use current completed months rather than retirement ages',()=>{
  for(const [birthday,expected] of [['1941-10-01',95],['1941-09-01',95+1/12],['1936-04-01',100.5]]){
    const s=flatPlan();s.spending.annualBaseSpending=0;
    Object.assign(s.household,{birthday,asOfDate:'2026-10-01',retirementDate:'2027-10-01',targetEndAge:119});
    const result=runSteadySimulation(s);
    assert.equal(result.primaryDeathAge,expected);
    assert.equal(result.monthlyDetails.at(-1).age,expected);
  }
  const married=flatPlan();married.spending.annualBaseSpending=0;
  Object.assign(married.household,{filingStatus:'Married',birthday:'1966-10-01',spouseBirthday:'1936-09-30',asOfDate:'2026-10-01',retirementDate:'2027-10-31',targetEndAge:119});
  assert.equal(runSteadySimulation(married).spouseDeathAge,100,'Spouse current age comes from today, independently of retirement-month clamping');
});

test('retirement after the assumed household lifetime has no invented balances',()=>{
  const s=flatPlan();Object.assign(s.household,{currentAge:60,retirementAge:96,targetEndAge:119});
  const result=runSteadySimulation(s);
  assert.equal(result.endReason,'before-retirement');assert.equal(result.endingBalance,null);assert.deepEqual(result.monthlyDetails,[]);
  s.household.retirementAge=95;assert.deepEqual(runSteadySimulation(s).monthlyDetails,[]);
});

test('extra steady simulation preserves the plan and all sampled outcome counts and medians',()=>{
  for(const count of [1,3,4,20]){
    const s=baseScenario();s.numberOfSimulations=count;
    const original=structuredClone(s),paths=Array.from({length:count},(_,index)=>{
      const seed=BigInt(s.seed)+BigInt(index)*stride;
      const plain=runOne(s,new JavaRandom(seed),{captureMonthlyBalances:true,captureTaxDetails:true});
      const {monthlyDetails,...detailed}=runOne(s,new JavaRandom(seed),{captureMonthlyBalances:true,captureMonthlyDetails:true,captureTaxDetails:true});
      assert.deepEqual(detailed,plain);
      return plain;
    });
    const balances=paths.map(p=>Math.max(0,p.yearEnd.at(-1))).sort((a,b)=>a-b),result=runSimulation(s);
    assert.deepEqual(s,original);
    assert.equal(result.provenance.simulationCount,count);
    assert.equal(result.successProbability,paths.filter(p=>p.success).length/count);
    assert.equal(result.medianEndingBalance,(balances[Math.floor((count-1)/2)]+balances[Math.floor(count/2)])/2);
    assert.deepEqual(result.steadySimulation,runSteadySimulation(s));
    assert.equal(Object.hasOwn(result,'medianSimulation'),false);
    for(const row of result.steadySimulation.monthlyDetails){
      near(row.portfolio,row.pretax+row.roth+row.taxable+row.cash);
      near(row.netAssets,row.portfolio+row.home-row.mortgage);
    }
  }
});

test('monthly detail rows include opening and closing balances through a partial horizon',()=>{
  const s=flatPlan();s.household.retirementAgeMonths=6;
  const path=runOne(s,fullLife,{captureMonthlyDetails:true}),rows=path.monthlyDetails;
  assert.equal(rows.length,19);assert.equal(rows[0].age,65.5);assert.equal(rows.at(-1).age,67);
  for(let i=0;i<rows.length;i++){assert.equal(rows[i].month,i);near(rows[i].roth,100000-i*1000);}
  near(rows.at(-1).portfolio,path.yearEnd.at(-1));
});

test('mortgage snapshots retain pre-retirement amortization and each monthly payoff',()=>{
  const s=flatPlan();s.household.currentAge=64;s.spending.annualBaseSpending=0;s.home.currentValue=200000;
  Object.assign(s.mortgage,{currentBalance:100000,monthlyPayment:5000,yearsLeft:2});
  const schedule=mortgageAtRetirement(s.mortgage,12),rows=runOne(s,fullLife,{captureMonthlyDetails:true}).monthlyDetails;
  near(rows[0].mortgage,schedule.balance);
  let expected=schedule.balance;
  for(let i=1;i<rows.length;i++){
    if(i<=schedule.months)expected=payMortgage(expected,s.mortgage.monthlyPayment,schedule.rate);
    near(rows[i].mortgage,expected);near(rows[i].home,200000);
  }
  near(rows[12].mortgage,0);near(rows.at(-1).mortgage,0);
});

test('home-sale snapshots clear home and mortgage and transfer net equity only once',()=>{
  const s=flatPlan();s.household.targetEndAge=69;s.spending.annualBaseSpending=0;
  s.home.currentValue=200000;Object.assign(s.mortgage,{currentBalance:100000,monthlyPayment:1000,yearsLeft:10});
  Object.assign(s.longTermCare,{enabled:true,annualCost:0,averageDurationYears:2});
  let calls=0;const rng={nextDouble:()=>calls++<3?.999999:0,normal:()=>0};
  const path=runOne(s,rng,{captureMonthlyDetails:true}),rows=path.monthlyDetails;
  assert.equal(rows.length,49);assert.equal(rows[24].home,200000);
  const before=rows[24],after=rows[25],cashRate=1.02**(1/12)-1;
  assert.equal(after.home,0);assert.equal(after.mortgage,0);
  near(after.cash,(before.cash+before.home-before.mortgage)*(1+cashRate));
  for(const row of rows.slice(25)){assert.equal(row.home,0);assert.equal(row.mortgage,0);}
  near(rows[26].cash,after.cash*(1+cashRate));
  near(rows.at(-1).portfolio,path.yearEnd.at(-1));
});

test('failed paths show the last unmet monthly cost and stop without negative assets or padded rows',()=>{
  const s=flatPlan();s.accounts.roth=1500;
  const rows=runOne(s,fullLife,{captureMonthlyDetails:true}).monthlyDetails;
  assert.equal(rows.length,3);near(rows[1].roth,500);near(rows[2].portfolio,0);
  near(rows[2].unfundedAmount,500);assert.equal(rows[2].cash,0);
  const result=runSimulation(s);
  assert.equal(result.steadySimulation.endReason,'shortfall');
  assert.deepEqual(result.steadySimulation.monthlyDetails,rows);
});

test('calendar rows use the completed run dates and fixed lifespan, clamping month-end anniversaries',()=>{
  const s=flatPlan();s.spending.annualBaseSpending=0;Object.assign(s.household,{birthday:'1961-01-31',spouseBirthday:'1961-01-31',asOfDate:'2026-01-31',retirementDate:'2026-01-31',targetEndAge:66});
  const result=runSimulation(s),rows=result.steadySimulation.monthlyDetails;
  assert.equal(rows[0].date,'2026-01-31');assert.equal(rows[1].date,'2026-02-28');
  assert.equal(rows.at(-1).date,'2056-01-31');assert.equal(result.steadySimulation.endReason,'lifespan');assert.equal(rows.at(-1).age,95);
  for(const row of rows){assert.equal(row.date,addCalendarMonths(s.household.retirementDate,row.month));assert.equal(row.age,(Math.round(retirementAge(s)*12)+row.month)/12);}
});

test('monthly cash-flow capture reports actual spending and account draws without changing outcomes',()=>{
  const s=flatPlan(),plain=runOne(s,new JavaRandom(7n),{captureTaxDetails:true,fixedDeathAges:{primary:67}}),detailed=runOne(s,new JavaRandom(7n),{captureMonthlyDetails:true,captureTaxDetails:true,fixedDeathAges:{primary:67}});
  const {monthlyDetails,...outcome}=detailed;assert.deepEqual(outcome,plain);
  assert.equal(monthlyDetails[0].cashFlow,undefined);
  const f=monthlyDetails[1].cashFlow;
  near(f.expenses,1000);near(f.additionalWithdrawal,1000);near(f.accountWithdrawals.roth,1000);
  near(f.incomeTax,0);near(f.earlyPenalty,0);near(f.conversionAmount,0);
  for(const p of monthlyDetails.slice(1)){
    const f=p.cashFlow;
    near(f.socialSecurity+f.guaranteedIncome+f.seppDistribution+f.rmdDistribution+f.additionalWithdrawal,f.expenses+f.incomeTax+f.earlyPenalty+f.surplus);
  }
});

test('monthly examples distinguish taxes, early penalties, scheduled SEPP and cash-first draws',()=>{
  const s=flatPlan();Object.assign(s.household,{currentAge:50,retirementAge:50,targetEndAge:51});s.accounts={pretax:1000000,roth:100000,taxable:0,cash:20000};
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  let f=runOne(s,new JavaRandom(7n),{captureMonthlyDetails:true,fixedDeathAges:{primary:51}}).monthlyDetails[1].cashFlow;
  assert.ok(f.earlyPenalty>100);assert.ok(f.additionalWithdrawal>f.expenses);near(f.earlyPenalty,f.accountWithdrawals.pretax*.1);
  s.withdrawalStrategy.useCashReserveDuringDrawdowns=true;s.withdrawalStrategy.drawdownTrigger=.01;
  f=runOne(s,new JavaRandom(7n),{captureMonthlyDetails:true,fixedDeathAges:{primary:51}}).monthlyDetails[1].cashFlow;
  assert.equal(f.cashFirst,true);near(f.accountWithdrawals.cash,1000);near(f.accountWithdrawals.pretax,0);near(f.earlyPenalty,0);
  s.withdrawalStrategy.seppEligible=true;
  f=runOne(s,new JavaRandom(7n),{captureMonthlyDetails:true,fixedDeathAges:{primary:51}}).monthlyDetails[1].cashFlow;
  assert.equal(f.seppProtected,true);assert.ok(f.seppDistribution>0);near(f.accountWithdrawals.pretax,f.seppDistribution);near(f.earlyPenalty,0);
});

test('monthly examples show RMD surplus and separate conversion transfers and tax',()=>{
  const s=flatPlan();Object.assign(s.household,{currentAge:80,retirementAge:80,targetEndAge:81});s.accounts={pretax:1000000,roth:0,taxable:0,cash:100000};s.spending.annualBaseSpending=0;
  s.rothConversion.enabled=true;s.rothConversion.marginalRateCap=.22;
  const rows=runOne(s,new JavaRandom(7n),{captureMonthlyDetails:true,fixedDeathAges:{primary:81}}).monthlyDetails;
  const f=rows[1].cashFlow;assert.ok(f.rmdDistribution>0);assert.ok(f.surplus>0);near(f.additionalWithdrawal,0);near(f.accountWithdrawals.pretax,f.rmdDistribution);
  const last=rows.at(-1).cashFlow;assert.ok(last.conversionAmount>0);assert.ok(last.conversionTax>0);assert.equal(rows.slice(1,-1).some(p=>p.cashFlow.conversionAmount>0),false);
});
