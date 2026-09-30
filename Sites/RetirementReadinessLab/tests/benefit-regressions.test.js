import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,validateScenario,ALLOCATION_KEYS} from '../dist/model.js';
import {runOne,medicarePremium} from '../dist/engine.js';

const near=(actual,expected,tolerance=.01)=>assert.ok(Math.abs(actual-expected)<tolerance,`${actual} != ${expected}`);
const fullLife={nextDouble:()=>.999999,normal:mean=>mean};
function flatPlan(age,years){
  const s=baseScenario();
  Object.assign(s.household,{currentAge:age,retirementAge:age,targetEndAge:age+years});
  s.accounts={pretax:0,roth:4000000,taxable:0,cash:0};
  Object.assign(s.spending,{annualBaseSpending:0,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  for(const key of Object.keys(s.market))s.market[key]=0;
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});
  s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;
  return s;
}
function conversionThenCare(factor,beforeRetirement=false){
  const s=flatPlan(beforeRetirement?49:50,18);
  Object.assign(s.household,{retirementAge:50,spouseCurrentAge:beforeRetirement?64:65,filingStatus:'Married',targetEndAge:67});
  s.accounts={pretax:10000,roth:10000,taxable:0,cash:0};s.rothHistory.contributionBasis=10000;
  s.market[beforeRetirement?'preRetirementStdDev':'stockStdDev']=.6;
  for(const key of ALLOCATION_KEYS)s.postRetirementAllocation[key]=1;
  Object.assign(s.longTermCare,{enabled:true,annualCost:factor<1?10000:20000,averageDurationYears:1});
  Object.assign(s.rothConversion,{enabled:true,marginalRateCap:.1});
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;
  assert.deepEqual(validateScenario(s),[]);
  // The primary lives to 52 and the spouse to 67 (primary age 52). Only
  // the spouse enters care, for the last year, after the first conversion.
  const deaths=[.999999,0,.999999,0,.5,0];let deathDraws=0,returnDraws=0;
  const rng={nextDouble:()=>deaths[deathDraws++]??.999999,normal:(mean,std)=>std>0?(returnDraws++===(beforeRetirement?0:12)?Math.log(factor):0):mean};
  return runOne(s,rng,{captureTaxDetails:true});
}

test('a market loss preserves opening Roth basis and cannot create a false care shortfall',()=>{
  const path=conversionThenCare(.5);
  assert.equal(path.success,true);assert.equal(path.failureAge,null);
  near(path.taxYears[0].conversions,10000);near(path.yearEnd.at(-1),0);
  near(path.taxYears[1].accounts.cash,0);
});

test('market gains cannot shelter a recent Roth conversion from recapture',()=>{
  const path=conversionThenCare(2);
  assert.equal(path.success,true);near(path.taxYears[0].conversions,10000);
  // $10,000 of fixed opening basis funds care first, then all $10,000 of
  // recent conversion principal. Funding its $1,000 recapture from earnings
  // also pays the earnings penalty; those earnings remain below the deduction.
  near(path.yearEnd.at(-1),40000-20000-1000/.9);
});

test('returns before retirement also leave the entered Roth basis unchanged',()=>{
  const loss=conversionThenCare(.5,true),gain=conversionThenCare(2,true);
  assert.equal(loss.success,true);near(loss.taxYears[0].conversions,5000);near(loss.yearEnd.at(-1),0);
  assert.equal(gain.success,true);near(gain.taxYears[0].conversions,20000);
  near(gain.yearEnd.at(-1),40000-10000-10000/.9);
});

function socialSecurityPath(factors,{married=false,beforeRetirement=false}={}){
  const s=flatPlan(beforeRetirement?66:67,factors.length+1+(beforeRetirement?1:0));
  Object.assign(s.household,{retirementAge:67,targetEndAge:67+factors.length+1,filingStatus:married?'Married':'Single',spouseCurrentAge:67});
  s.socialSecurity.annualBenefitAt67=30000;
  Object.assign(s.spending,{generalInflationMean:beforeRetirement?-.02:0,generalInflationStdDev:.1});
  assert.deepEqual(validateScenario(s),[]);
  let inflationDraws=0;
  const rng={nextDouble:()=>.999999,normal:(mean,std)=>std>0?Math.log(factors[Math.floor(inflationDraws++/12)]??1)/12:mean};
  const path=runOne(s,rng,{captureTaxDetails:true});
  assert.equal(inflationDraws,(factors.length+1)*12);
  return path.taxYears.map(year=>year.socialSecurity);
}

test('Social Security waits for full price recovery and then increases only above its previous reference',()=>{
  for(const married of [false,true]){
    const benefits=socialSecurityPath([.9,1.05,1.1,.9,1.12],{married});
    const expected=[30000,30000,30000,31185,31185,31434.48];
    // The 1959 birth cohort's age-67 primary amount includes two months of
    // delayed credits; the full spousal benefit is half the underlying PIA.
    const householdFactor=married?1+.5/(1+.08/6):1;
    benefits.forEach((amount,i)=>near(amount,expected[i]*householdFactor));
  }
});

test('Social Security remembers deflation before retirement when measuring recovery',()=>{
  const benefits=socialSecurityPath([1.01,1.02],{beforeRetirement:true});
  benefits.forEach((amount,i)=>near(amount,[30000,30000,30287.88][i]));
});

test('Social Security compounds ordinary positive COLAs without changing monthly draws',()=>{
  const benefits=socialSecurityPath([1.02,1.03]);
  benefits.forEach((amount,i)=>near(amount,[30000,30600,31518][i]));
});

test('the highest Medicare threshold stays frozen through 2027 while lower thresholds inflate',()=>{
  for(const status of ['Single','HeadOfHousehold','Married']){
    const threshold=status==='Married'?750000:500000;
    for(const year of [2026,2027])for(const inflation of [1.02,3]){
      near(medicarePremium(threshold,status,1,1,inflation,year,inflation),819.89);
      near(medicarePremium(threshold+.01,status,1,1,inflation,year,inflation),819.89);
    }
    near(medicarePremium(threshold-.01,status,1,1,1.02,2027,1.02),771.49);
    const firstThreshold=(status==='Married'?218000:109000)*1.02;
    near(medicarePremium(firstThreshold,status,1,1,1.02,2027,1.02),241.89);
    near(medicarePremium(firstThreshold+.01,status,1,1,1.02,2027,1.02),337.59);
  }
});

test('the highest Medicare threshold indexes from 2028 using the prior year and statutory rounding',()=>{
  for(const status of ['Single','HeadOfHousehold','Married']){
    const joint=status==='Married'?1.5:1;
    for(const [priorInflation,threshold] of [[1.02,510000],[1.017,509000],[.9,500000]]){
      near(medicarePremium(threshold*joint-.01,status,1,1,1.2,2028,priorInflation),771.49);
      near(medicarePremium(threshold*joint,status,1,1,1.2,2028,priorInflation),819.89);
    }
  }
});

test('Medicare uses the separate top threshold in both initial estimates and two-year lookbacks',()=>{
  const thresholdsByRetirement=[[500000,500000,510000,520000],[500000,510000,520000,531000],[510000,520000,531000,541000]];
  for(const status of ['Single','Married'])for(const yearsToRet of [0,1,2])for(const income of [500000,510000,520000]){
    const s=flatPlan(65,yearsToRet+4),joint=status==='Married'?1.5:1,people=status==='Married'?2:1;
    Object.assign(s.household,{retirementAge:65+yearsToRet,filingStatus:status,spouseCurrentAge:65});
    Object.assign(s.guaranteedIncome,{annualIncome:income*joint,startAge:65,annualIncrease:0});
    Object.assign(s.spending,{annualBaseSpending:income*joint,generalInflationMean:.02});
    assert.deepEqual(validateScenario(s),[]);
    const baseline=runOne(s,fullLife,{captureTaxDetails:true});
    s.healthcare.includeMedicarePremiums=true;
    const medicare=runOne(s,fullLife,{captureTaxDetails:true});
    assert.equal(baseline.success,true);assert.equal(medicare.success,true);
    let cumulative=0;
    medicare.taxYears.forEach((year,i)=>{
      cumulative+=(income>=thresholdsByRetirement[yearsToRet][i]?819.89:771.49)*12*people;
      near(baseline.yearEnd[i+1]-medicare.yearEnd[i+1],cumulative);
      near(year.ordinaryIncome,income*joint);
    });
  }
});
