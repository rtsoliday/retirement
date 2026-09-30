import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,validateScenario} from '../dist/model.js';
import {runOne,runSimulation,JavaRandom,medicarePremium} from '../dist/engine.js';
import {ordinaryIncomeTax,taxableSocialSecurity,taxableOrdinaryIncome} from '../dist/tax.js';
import {requiredMinimumDistribution,rmdStartAge} from '../dist/distributions.js';
import {monthlyRateDistribution,sampleMonthlyRate} from '../dist/annual-rates.js';

const fullLife={nextDouble:()=>.999999,normal:mean=>mean};
const near=(a,b,tolerance=.01)=>assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
function flatPlan(age=75,years=1){
  const s=baseScenario();
  Object.assign(s.household,{currentAge:age,retirementAge:age,targetEndAge:age+years});
  s.accounts={pretax:2000000,roth:0,taxable:0,cash:0};
  Object.assign(s.spending,{annualBaseSpending:10000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  for(const key of Object.keys(s.market))s.market[key]=0;
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});
  s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;
  assert.deepEqual(validateScenario(s),[]);
  return s;
}
const detail=s=>runOne(s,fullLife,{captureTaxDetails:true});

test('RMD cohorts and Uniform Lifetime factors match IRS tables',()=>{
  assert.equal(rmdStartAge(1959),73);assert.equal(rmdStartAge(1960),75);
  assert.equal(requiredMinimumDistribution(1000000,72,1953),0);
  near(requiredMinimumDistribution(1000000,73,1953),1000000/26.5);
  assert.equal(requiredMinimumDistribution(1000000,74,1960),0);
  near(requiredMinimumDistribution(1000000,75,1960),1000000/24.6);
  near(requiredMinimumDistribution(1000000,97,1929),1000000/7.8);
  near(requiredMinimumDistribution(1000000,119,1907),1000000/2.3);
  near(requiredMinimumDistribution(1000000,120,1906),500000);
  assert.equal(requiredMinimumDistribution(0,75,1951),0);
});

test('mandatory distributions create taxable income and unused proceeds remain in cash once',()=>{
  const s=flatPlan(),path=detail(s),year=path.taxYears[0],rmd=2000000/24.6;
  // Independent accounting: twelve equal RMD payments, spending and credited
  // bank interest, with the incremental annual tax paid from each surplus.
  let cash=0,income=0,taxPaid=0;const rate=1.02**(1/12)-1;
  for(let m=0;m<12;m++){
    const interest=cash*rate;income+=rmd/12+interest;
    const tax=ordinaryIncomeTax(income,'Single',1,1,2026);
    cash+=interest+rmd/12-10000/12-(tax-taxPaid);taxPaid=tax;
  }
  near(year.requiredMinimumDistribution,rmd);near(year.pretaxDistributions,rmd);
  near(year.accounts.pretax,2000000-rmd);near(year.accounts.cash,cash);
  near(year.ordinaryIncome,income);near(year.tax,taxPaid);
  near(path.yearEnd.at(-1),2000000-rmd+cash);
  assert.ok(year.tax>7368,'tax includes interest on the RMD surplus');
});

test('spending withdrawals satisfy the RMD without charging it again',()=>{
  const s=flatPlan();s.spending.annualBaseSpending=100000;
  let low=100000,high=200000;
  for(let i=0;i<80;i++){const mid=(low+high)/2;if(mid-ordinaryIncomeTax(mid,'Single',1,1,2026)>=100000)high=mid;else low=mid;}
  const path=detail(s),year=path.taxYears[0];
  near(year.pretaxDistributions,high);near(year.ordinaryIncome,high);
  near(path.yearEnd.at(-1),2000000-high);assert.ok(high>year.requiredMinimumDistribution);
});

test('RMDs use the preceding modeled year balance, including a final partial year',()=>{
  const s=flatPlan(75,2);s.household.retirementAgeMonths=6;
  const years=detail(s).taxYears;
  assert.equal(years.length,2);
  near(years[0].pretaxDistributions,2000000/24.6);
  near(years[1].requiredMinimumDistribution,years[0].accounts.pretax/23.7);
  near(years[1].pretaxDistributions,years[1].requiredMinimumDistribution);
});

test('RMD proceeds cannot be converted to Roth',()=>{
  const s=flatPlan();Object.assign(s.rothConversion,{enabled:true,marginalRateCap:.37});
  const year=detail(s).taxYears[0],rmd=2000000/24.6;
  near(year.pretaxDistributions,rmd);near(year.conversions,2000000-rmd);
  near(year.accounts.pretax,0);near(year.accounts.cash,0);
  assert.ok(year.accounts.roth<2000000-rmd,'conversion taxes are funded from cash and Roth');
});

test('the surviving spouse uses their own RMD age after the primary death year',()=>{
  const s=flatPlan(75,3);Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:70});
  let calls=0;const path=runOne(s,{nextDouble:()=>calls++===0?0:.999999,normal:mean=>mean},{captureTaxDetails:true});
  assert.ok(path.taxYears[0].requiredMinimumDistribution>0);
  assert.ok(path.taxYears[1].requiredMinimumDistribution>0,'remaining primary-owner death-year RMD');
  assert.equal(path.taxYears[2].requiredMinimumDistribution,0,'spouse age 72, born in 1956');
  near(path.taxYears[3].requiredMinimumDistribution,path.taxYears[2].accounts.pretax/26.5);
});

test('SEPP protects enrolled pretax assets while other accounts cover extra spending and taxes',()=>{
  const s=flatPlan(50);s.accounts={pretax:500000,roth:100000,taxable:0,cash:0};s.spending.annualBaseSpending=40000;
  Object.assign(s.withdrawalStrategy,{seppEligible:true,applyEarlyWithdrawalPenalty:true});s.rothConversion.enabled=true;
  const annualSepp=500000/((1-1.05**-36.2)/.05),path=detail(s),year=path.taxYears[0];
  assert.equal(path.success,true);near(year.pretaxDistributions,annualSepp);
  near(year.accounts.pretax,500000-annualSepp);near(year.ordinaryIncome,annualSepp);
  near(path.yearEnd.at(-1),600000-40000-ordinaryIncomeTax(annualSepp,'Single'));
  assert.equal(year.conversions,0);
});

test('a SEPP spending gap reports a shortfall instead of drawing extra protected principal',()=>{
  const s=flatPlan(50);s.accounts.pretax=500000;s.spending.annualBaseSpending=40000;
  Object.assign(s.withdrawalStrategy,{seppEligible:true,applyEarlyWithdrawalPenalty:true});
  const path=detail(s);assert.equal(path.success,false);assert.equal(path.failureAge,50);
  assert.ok(path.taxYears[0].accounts.pretax>490000);assert.equal(path.yearEnd.at(-1),0);
});

test('SEPP protection continues for five years when age 59.5 arrives earlier',()=>{
  const s=flatPlan(58,6);s.accounts={pretax:200000,roth:100000,taxable:0,cash:0};s.spending.annualBaseSpending=5000;
  s.withdrawalStrategy.seppEligible=true;Object.assign(s.rothConversion,{enabled:true,marginalRateCap:.37});
  const path=detail(s);assert.equal(path.success,true);
  assert.ok(path.taxYears.slice(0,5).every(y=>y.conversions===0));
  assert.ok(path.taxYears[5].conversions>0);
});

test('SEPP protection continues until 59.5 when the fifth anniversary arrives earlier',()=>{
  const s=flatPlan(50,11);s.accounts={pretax:100000,roth:100000,taxable:0,cash:0};s.spending.annualBaseSpending=0;
  s.withdrawalStrategy.seppEligible=true;Object.assign(s.rothConversion,{enabled:true,marginalRateCap:.37});
  const path=detail(s);assert.equal(path.success,true);
  assert.ok(path.taxYears.slice(0,9).every(y=>y.conversions===0));
  assert.ok(path.taxYears[9].conversions>0);
});

test('complete SEPP account depletion allows a final smaller payment without breaking the series',()=>{
  const s=flatPlan(50,11);s.accounts={pretax:500000,roth:2000000,taxable:0,cash:0};s.spending.annualBaseSpending=5000;
  Object.assign(s.market,{stockMeanReturn:-.2,bondMeanReturn:-.2});
  Object.assign(s.withdrawalStrategy,{seppEligible:true,applyEarlyWithdrawalPenalty:true});
  const path=detail(s),annualSepp=500000/((1-1.05**-36.2)/.05);
  assert.equal(path.success,true);
  const last=path.taxYears.find(y=>y.accounts.pretax===0);
  assert.ok(last);assert.ok(last.pretaxDistributions>0&&last.pretaxDistributions<annualSepp);
  assert.ok(path.taxYears.slice(0,9).every(y=>y.pretaxDistributions<=annualSepp+.01));
});

test('cash interest is taxed without adding the credited interest twice',()=>{
  const s=flatPlan(67);s.accounts={pretax:0,roth:0,taxable:0,cash:500000};s.spending.annualBaseSpending=0;
  Object.assign(s.guaranteedIncome,{annualIncome:100000,startAge:67});
  let cash=500000,income=0,taxPaid=0;const rate=1.02**(1/12)-1;
  for(let m=0;m<12;m++){
    const interest=cash*rate;income+=100000/12+interest;
    const tax=ordinaryIncomeTax(income,'Single',1,1,2026);
    cash+=interest+100000/12-(tax-taxPaid);taxPaid=tax;
  }
  const path=detail(s),year=path.taxYears[0];
  near(cash,596579.0430842103);near(path.yearEnd.at(-1),cash);
  near(year.ordinaryIncome,income);near(year.tax,taxPaid);
});

test('cash interest raises taxable Social Security and uses senior deduction phaseouts',()=>{
  const s=flatPlan(67);s.accounts={pretax:0,roth:0,taxable:0,cash:1500000};s.spending.annualBaseSpending=50000;
  s.socialSecurity.annualBenefitAt67=30000;Object.assign(s.guaranteedIncome,{annualIncome:40000,startAge:67});
  const year=detail(s).taxYears[0],taxableSS=taxableSocialSecurity(year.ordinaryIncome,30000,'Single');
  assert.ok(year.ordinaryIncome>69000);assert.ok(taxableSS>taxableSocialSecurity(40000,30000,'Single'));
  near(year.tax,ordinaryIncomeTax(year.ordinaryIncome+taxableSS,'Single',1,1,2026));
  const enhanced=Math.max(0,6000-(year.ordinaryIncome+taxableSS-75000)*.06);
  near(taxableOrdinaryIncome(year.ordinaryIncome+taxableSS,'Single',1,1,2026),year.ordinaryIncome+taxableSS-18150-enhanced);
});

test('cash interest fills part of the Roth conversion bracket',()=>{
  const s=flatPlan(67);s.accounts={pretax:100000,roth:0,taxable:0,cash:500000};s.spending.annualBaseSpending=0;
  Object.assign(s.rothConversion,{enabled:true,marginalRateCap:.12});
  const year=detail(s).taxYears[0];
  // $74,550 less the $24,150 senior deductions reaches the $50,400 cap.
  // Bank interest uses $10,000 of this room before the conversion.
  near(year.conversions,64550);near(year.ordinaryIncome,74550);near(year.tax,5800);
  near(year.accounts.pretax,35450);near(year.accounts.cash,504200);
});

test('cash interest and RMDs enter Medicare income estimates and the two-year lookback',()=>{
  for(const account of ['cash','pretax']){
    const s=flatPlan(75,3);s.accounts={pretax:0,roth:0,taxable:0,cash:0};s.accounts[account]=account==='cash'?6000000:3000000;
    s.spending.annualBaseSpending=0;s.healthcare.includeMedicarePremiums=true;
    const withMedicare=detail(s),without=structuredClone(s);without.healthcare.includeMedicarePremiums=false;
    const uncharged=detail(without);
    // Both sources exceed the base threshold without any spending withdrawal.
    for(const y of [0,2])assert.ok(uncharged.yearEnd[y+1]-withMedicare.yearEnd[y+1]>(y+1)*12*241.89);
  }
});

test('annual return and inflation distributions have the entered compounded moments',()=>{
  for(const [mean,std] of [[.133,.162],[.03,.06],[.023,.016],[.04,.018],[0,.2],[-.02,.3],[-.2,.6],[.2,0]]){
    const d=monthlyRateDistribution(mean,std),logMean=12*d.logMean,logVariance=12*d.logStdDev**2;
    const expectedFactor=Math.exp(logMean+logVariance/2);
    near(expectedFactor-1,mean,1e-12);
    near(expectedFactor*Math.sqrt(Math.expm1(logVariance)),std,1e-12);
  }
});

test('sampled annual inflation and returns match their means and volatility without clipping bias',()=>{
  const rng=new JavaRandom(582704),count=25000;
  for(const [mean,std,tolerance] of [[.023,.016,.0005],[.04,.018,.0006],[.133,.162,.003],[0,.2,.004]]){
    const d=monthlyRateDistribution(mean,std);let total=0,squares=0;
    for(let i=0;i<count;i++){
      let factor=1;for(let m=0;m<12;m++){const change=sampleMonthlyRate(d,rng);assert.ok(change>-1);factor*=1+change;}
      const change=factor-1;total+=change;squares+=change**2;
    }
    const average=total/count,volatility=Math.sqrt(squares/count-average**2);
    near(average,mean,tolerance);near(volatility,std,tolerance);
  }
});

test('zero volatility preserves exact annual growth and draws the same number of normals',()=>{
  const s=flatPlan(65,2);s.accounts={pretax:0,roth:100000,taxable:0,cash:0};s.spending.annualBaseSpending=0;
  s.market.stockMeanReturn=.133;s.postRetirementAllocation.stock50xOrMore=1;
  const path=runOne(s,fullLife);near(path.yearEnd[1],113300);near(path.yearEnd[2],128368.9);
  let calls=0;const d=monthlyRateDistribution(.04,0);let factor=1;
  for(let m=0;m<12;m++)factor*=1+sampleMonthlyRate(d,{normal:mean=>{calls++;return mean;}});
  near(factor,1.04,1e-12);assert.equal(calls,12);
});

test('cash-only inflation shortfalls are attributed by measured sensitivity, with no market effect',()=>{
  const s=flatPlan(65,45);s.accounts={pretax:0,roth:0,taxable:0,cash:1000000};s.spending.annualBaseSpending=40000;s.spending.generalInflationMean=.10;s.numberOfSimulations=100;
  const original=structuredClone(s),result=runSimulation(s),risk=result.riskBreakdown;
  assert.ok(result.successProbability<.8);assert.equal(risk.primaryRisk,'inflation');
  assert.equal(risk.checks.find(c=>c.key==='market').netReduction,0);
  assert.ok(risk.checks.find(c=>c.key==='inflation').netReduction>0);
  assert.match(risk.recommendedNextTest,/general inflation/i);assert.match(risk.summary,/paired paths/);
  assert.deepEqual(s,original);
});

test('sensitivity diagnostics disclose their bounded sample and do not assign a cause without an improvement',()=>{
  const s=flatPlan(65);s.accounts={pretax:0,roth:0,taxable:0,cash:1000000};s.numberOfSimulations=200;
  const r=runSimulation(s);assert.equal(r.riskBreakdown.simulationCount,128);assert.equal(r.riskBreakdown.primaryRisk,'none');
  for(const c of r.riskBreakdown.checks){assert.equal(c.netReduction,c.avoidedShortfalls-c.introducedShortfalls);assert.equal(c.variantFailures,r.riskBreakdown.baselineFailures-c.netReduction);}
  const compact=runSimulation(s,()=>{},{includeRiskAnalysis:false});assert.equal(compact.riskBreakdown,null);assert.equal(compact.successProbability,r.successProbability);assert.deepEqual(compact.balanceBands,r.balanceBands);
});

test('Medicare top tiers begin exactly at $500,000 single and $750,000 joint',()=>{
  for(const [status,threshold,people] of [['Single',500000,1],['HeadOfHousehold',500000,1],['Married',750000,2]]){
    near(medicarePremium(threshold-.01,status,people,1,1),771.49*people);
    near(medicarePremium(threshold,status,people,1,1),819.89*people);
    near(medicarePremium(threshold+.01,status,people,1,1),819.89*people);
    near(medicarePremium(threshold*1.1,status,people,1.2,1.1),819.89*people*1.2);
  }
  near(medicarePremium(109000,'Single',1,1,1),241.89);
  near(medicarePremium(109000.01,'Single',1,1,1),337.59);
  near(medicarePremium(137000,'Single',1,1,1),337.59);
});
