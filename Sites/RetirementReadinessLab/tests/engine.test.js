import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,sampleScenarios,budgetEstimate,validateScenario,normalizeScenario,normalizeScenarios,ROTH_CONVERSION_RATES,DEFAULT_SEED} from '../dist/model.js';
import {runSimulation,estimateDecision,sampleDeathAge} from '../dist/engine.js';
import {taxableSocialSecurity,taxableOrdinaryIncome,ordinaryIncomeTax,rothConversionPlan} from '../dist/tax.js';
import {annualBenefitAtClaimAge} from '../dist/social-security.js';

// Seed of the Android reference snapshots below; the site default seed differs.
const REFERENCE_SEED=20260429;
// Keep the historical Android comparison inputs independent of site defaults.
function referenceScenario(){const s=baseScenario();Object.assign(s.household,{currentAge:50,spouseCurrentAge:50});return s;}

// Seeded web cashflow snapshots; starting balances retain the Android references.
// Corrected tax timing, death-year joint treatment, Medicare estimates and lookback filing status change endings.
// Independent annual accounting cases are in cashflow-regression.test.js.
// The web results now report failed endings as zero instead of the raw negative shortfall.
const cases=[
  ['base',s=>{},1,8824327.086129352,2576022.7361081364,null],
  ['zero',s=>{Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:67});Object.assign(s.accounts,{pretax:0,roth:0,taxable:0,cash:100000});Object.assign(s.spending,{annualBaseSpending:12000,generalInflationMean:0,generalInflationStdDev:0});Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});s.socialSecurity.annualBenefitAt67=0;Object.assign(s.market,{preRetirementMeanReturn:0,preRetirementStdDev:0,stockMeanReturn:0,stockStdDev:0,bondMeanReturn:0,bondStdDev:0});s.longTermCare.enabled=false;},1,79973.35589502829,100000,null],
  ['married',s=>Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:55}),1,1289628.2640126306,2576022.7361081364,null],
  ['early',s=>{s.household.retirementAge=55;s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;},1,1387594.4694501134,1309702.0532638722,null],
  ['roth',s=>{s.rothConversion.enabled=true;},1,8761885.136671377,2576022.7361081364,null],
  ['home',s=>{Object.assign(s.accounts,{pretax:100000,roth:0,cash:0});s.home.currentValue=500000;},0,-2341.10536489225,278445.629460946,961/12],
  // The two middle endings are 24355833.01249265 and 24464017.210152686;
  // the two middle starting balances are 5380044.310699925 and 6074093.048540814.
  ['fifty paths',s=>{s.numberOfSimulations=50;},.98,24409925.111322667,5727068.679620369,947/12]
];
for(const [name,edit,success,ending,starting,failure] of cases){test(`Seeded web regression: ${name}`,()=>{const s=referenceScenario();s.seed=REFERENCE_SEED;s.numberOfSimulations=1;edit(s);const r=runSimulation(s);assert.equal(r.successProbability,success);assert.ok(Math.abs(r.medianEndingBalance-Math.max(0,ending))<.01,`${r.medianEndingBalance} != ${ending}`);assert.ok(Math.abs(r.balanceBands[0].median-starting)<.01);assert.equal(r.medianFailureAge,failure);});}

test('budget estimate uses fixed costs and monthly spending',()=>{const b=baseScenario().budget;b.annualPropertyTaxes=4000;b.annualHomeInsurance=2000;b.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:[{monthlyAmount:1000}],creditCardBills:[{monthlyAmount:500}],cashAndAtmWithdrawals:100}];assert.equal(budgetEstimate(b),25200);});
test('invalid retirement age is rejected before simulation',()=>{const s=baseScenario();s.household.retirementAge=49;assert.match(validateScenario(s).join(' '),/Retirement age/);});
test('mortgage terms require whole years and whole extra months',()=>{
  for(const key of ['yearsLeft','monthsLeft'])for(const value of [.5,1.1]){
    const s=baseScenario();s.mortgage[key]=value;
    assert.match(validateScenario(s).join(' '),/Mortgage.*whole years.*whole extra months/);
    assert.throws(()=>runSimulation(s),/Mortgage/);
  }
  const s=baseScenario();s.mortgage.yearsLeft=1;s.mortgage.monthsLeft=11;
  assert.deepEqual(validateScenario(s),[]);
});
test('tax and benefit reference rules',()=>{assert.equal(taxableSocialSecurity(10000,30000,'Single'),0);assert.equal(ordinaryIncomeTax(16100,'Single',1,0,2026),0);assert.ok(Math.abs(annualBenefitAtClaimAge(30000,67)-30000)<.001);});

// Android forces the early-withdrawal penalty on for early ages; the web search keeps the plan's setting.
test('retirement and spending decision targets keep the penalty setting and flag the spending search limit',()=>{
  const s=referenceScenario();s.seed=REFERENCE_SEED;s.spending.annualBaseSpending=71000;const result=estimateDecision(s);
  assert.equal(result.earliestRetirementAge,55);assert.equal(result.earliestRetirementReadiness,0.8388888888888889);
  assert.equal(result.safeAnnualSpending,250000);assert.equal(result.safeSpendingAtSearchLimit,true);assert.equal(result.safeSpendingSearchLimit,250000);
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;const withPenalty=estimateDecision(s);
  assert.equal(withPenalty.earliestRetirementAge,55);assert.equal(withPenalty.earliestRetirementReadiness,0.8166666666666667);
});
test('safe spending below the search limit is not flagged as a lower bound',()=>{
  const s=baseScenario();s.household.currentAge=64;s.household.retirementAge=65;s.accounts={pretax:300000,roth:0,taxable:0,cash:0};
  const result=estimateDecision(s,.8,60);
  assert.ok(result.safeAnnualSpending!==null&&result.safeAnnualSpending<result.safeSpendingSearchLimit);assert.equal(result.safeSpendingAtSearchLimit,false);
});

test('ages must be whole numbers so mortality tables are never skipped',()=>{
  for(const edit of [s=>s.household.retirementAge=62.5,s=>s.household.currentAge=50.5,s=>s.household.targetEndAge=100.5,s=>{s.household.filingStatus='Married';s.household.spouseCurrentAge=48.5;}]){
    const s=baseScenario();edit(s);assert.match(validateScenario(s).join(' '),/Ages must be whole numbers/);assert.throws(()=>runSimulation(s),/Ages must be whole numbers/);
  }
  const single=baseScenario();single.household.spouseCurrentAge=48.5;assert.deepEqual(validateScenario(single),[]);
});

test('Roth conversions include Social Security that the conversion makes taxable',()=>{
  const ss=30000,other=20000,plan=rothConversionPlan(1e6,other,.12,'Single',1,1,2026,ss);
  const gross=x=>other+x+taxableSocialSecurity(other+x,ss,'Single');
  assert.ok(Math.abs(plan.tax-(ordinaryIncomeTax(gross(plan.amount),'Single',1,1,2026)-ordinaryIncomeTax(gross(0),'Single',1,1,2026)))<.01);
  assert.ok(plan.taxableSocialSecurityIncrease>0);assert.ok(Math.abs(plan.taxableSocialSecurityIncrease-(gross(plan.amount)-gross(0)-plan.amount))<.01);
  assert.ok(taxableOrdinaryIncome(gross(plan.amount),'Single',1,1,2026)<=50400+.01,'conversion stays within the 12% bracket');
  const noSS=rothConversionPlan(1e6,other,.12,'Single',1,1,2026);assert.equal(noSS.taxableSocialSecurityIncrease,0);assert.ok(noSS.amount>plan.amount);
});

test('imported scenarios need known choices, matching value types and distinct IDs',()=>{
  for(const [edit,pattern] of [[s=>s.household.filingStatus='MarriedFilingJointly',/Filing status/],[s=>s.household.gender='Other',/Longevity table/],[s=>s.spending.spendingPathModel='Declining',/Spending path/],[s=>s.accounts.pretax='800000',/wrong type: accounts\.pretax/],[s=>s.longTermCare.enabled='yes',/wrong type: longTermCare\.enabled/]]){
    const s=baseScenario();edit(s);assert.match(validateScenario(s).join(' '),pattern);
  }
  const single=baseScenario();single.household.spouseGender='Other';assert.deepEqual(validateScenario(single),[]);
  const plans=normalizeScenarios([{name:'A'},{name:'B'},{id:'keep',name:'C'},{id:'keep',name:'D'},{id:7,name:'E'}]);
  assert.deepEqual(plans.map(p=>p.id),['plan-imported-1','plan-imported-2','keep','plan-imported-4','7']);
  for(const p of plans)assert.deepEqual(validateScenario(p),[]);
});

test('Android and legacy JSON scenarios can be imported',()=>{const android=normalizeScenario({id:'android',household:{currentAge:60,retirementAge:65},accounts:{pretax:120000,roth:0,taxable:0,cash:0},spending:{annualBaseSpending:50000},socialSecurity:{annualBenefitAt67:30000}});assert.equal(android.accounts.pretax,120000);assert.equal(android.accounts.roth,0);const legacy=normalizeScenario({id:'legacy',currentAge:50,retirementAge:67,annualSpending:75000,pretaxBalance:800000,rothBalance:100000,cashBalance:50000,socialSecurityAt67:30000});assert.equal(legacy.spending.annualBaseSpending,75000);assert.equal(legacy.accounts.cash,50000);});

test('generated scenario IDs never displace an existing normalized ID',()=>{
  const input=[{}, {id:'plan-imported-1'}, {id:'duplicate'}, {id:'duplicate'}, {id:' plan-imported-4 '}, {id:7}];
  const plans=normalizeScenarios(input);
  assert.deepEqual(plans.map(s=>s.id),['plan-imported-2','plan-imported-1','duplicate','plan-imported-5','plan-imported-4','7']);
  assert.equal(new Set(plans.map(s=>s.id)).size,input.length);
  assert.deepEqual(normalizeScenarios(plans),plans);
});

test('single households do not require dormant spouse settings',()=>{const s=baseScenario();s.household.spouseCurrentAge=-1;s.socialSecurity.spouseClaimAge=99;s.guaranteedIncome.survivorPercent=2;assert.deepEqual(validateScenario(s),[]);s.household.filingStatus='Married';const errors=validateScenario(s).join(' ');assert.match(errors,/Spouse age/);assert.match(errors,/Spouse claim age/);assert.match(errors,/Guaranteed income/);});


test('Roth conversion cap accepts only supported brackets when enabled',()=>{
  const s=baseScenario();s.rothConversion.enabled=true;
  for(const rate of ROTH_CONVERSION_RATES){s.rothConversion.marginalRateCap=rate;assert.deepEqual(validateScenario(s),[]);}
  for(const rate of [.18,.23,0,.40]){s.rothConversion.marginalRateCap=rate;assert.match(validateScenario(s).join(' '),/Roth conversion cap/);}
  s.rothConversion.enabled=false;assert.deepEqual(validateScenario(s),[]);
});

test('every sample plan shows at least one shortfall in the four-path free preview',()=>{
  for(const s of sampleScenarios()){
    assert.equal(s.household.currentAge,60);assert.equal(s.household.spouseCurrentAge,60);
    assert.equal(s.seed,DEFAULT_SEED);assert.equal(s.numberOfSimulations,4);
    const r=runSimulation(s);assert.ok(r.successProbability<1,`${s.name}: ${r.successProbability*4} of 4 without a shortfall`);
  }
});


test('retirement month mortality uses the age table and never reads a fractional array index',()=>{
  assert.equal(sampleDeathAge('Male',65.5,70,{nextDouble:()=>0}),66);
  assert.equal(sampleDeathAge('Female',65+11/12,66,{nextDouble:()=>.999999}),66);
});

test('legacy fractional timing migrates to years and months without changing whole-year plans',()=>{
  const old=baseScenario();delete old.household.retirementAgeMonths;delete old.guaranteedIncome.startAgeMonths;delete old.longTermCare.averageDurationMonths;
  assert.deepEqual(normalizeScenario(old),baseScenario());
  old.guaranteedIncome.startAge=65.5;old.longTermCare.averageDurationYears=1.5;
  const s=normalizeScenario(old);assert.equal(s.guaranteedIncome.startAge,65);assert.equal(s.guaranteedIncome.startAgeMonths,6);
  assert.equal(s.longTermCare.averageDurationYears,1);assert.equal(s.longTermCare.averageDurationMonths,6);assert.deepEqual(validateScenario(s),[]);
  assert.deepEqual(normalizeScenario(JSON.parse(JSON.stringify(s))),s);
});

test('invalid months, fractional year fields and timing beyond the horizon are rejected',()=>{
  for(const [section,key] of [['household','retirementAgeMonths'],['guaranteedIncome','startAgeMonths'],['longTermCare','averageDurationMonths']])for(const value of [-1,12,.5]){
    const s=baseScenario();s[section][key]=value;assert.match(validateScenario(s).join(' '),/Month fields/);
  }
  for(const [section,key] of [['guaranteedIncome','startAge'],['longTermCare','averageDurationYears']]){
    const s=baseScenario();s[section][key]=1.5;assert.match(validateScenario(s).join(' '),/whole numbers/);
  }
  const s=baseScenario();s.longTermCare.averageDurationYears=10;s.longTermCare.averageDurationMonths=1;assert.match(validateScenario(s).join(' '),/duration/);
  Object.assign(s.household,{filingStatus:'Married',retirementAgeMonths:11,spouseCurrentAge:112});assert.match(validateScenario(s).join(' '),/Spouse age/);
});


test('legacy timing migration does not conceal out-of-range fractional inputs',()=>{
  for(const [section,key,value] of [['guaranteedIncome','startAge',-.001],['longTermCare','averageDurationYears',.9999],['longTermCare','averageDurationYears',10.0001]]){
    const old=baseScenario();delete old.guaranteedIncome.startAgeMonths;delete old.longTermCare.averageDurationMonths;old[section][key]=value;
    assert.ok(validateScenario(normalizeScenario(old)).length>0);
  }
});
