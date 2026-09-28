import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,budgetEstimate,validateScenario,normalizeScenario,ROTH_CONVERSION_RATES} from '../dist/model.js';
import {runSimulation,estimateDecision} from '../dist/engine.js';
import {taxableSocialSecurity,ordinaryIncomeTax} from '../dist/tax.js';
import {annualBenefitAtClaimAge} from '../dist/social-security.js';

// Android cashflow references with the same sample scenario and seed.
// The web results now report failed endings as zero instead of the raw negative shortfall.
const cases=[
  ['base',s=>{},1,8826526.467243172,2576022.7361081364,null],
  ['zero',s=>{Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:67});Object.assign(s.accounts,{pretax:0,roth:0,taxable:0,cash:100000});Object.assign(s.spending,{annualBaseSpending:12000,generalInflationMean:0,generalInflationStdDev:0});Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});s.socialSecurity.annualBenefitAt67=0;Object.assign(s.market,{preRetirementMeanReturn:0,preRetirementStdDev:0,stockMeanReturn:0,stockStdDev:0,bondMeanReturn:0,bondStdDev:0});s.longTermCare.enabled=false;},1,79973.35589502829,100000,null],
  ['married',s=>Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:55}),1,1138163.0530949987,2576022.7361081364,null],
  ['early',s=>{s.household.retirementAge=55;s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;},1,1370373.2101487578,1309702.0532638722,null],
  ['roth',s=>{s.rothConversion.enabled=true;},1,8750586.284404427,2576022.7361081364,null],
  ['home',s=>{Object.assign(s.accounts,{pretax:100000,roth:0,cash:0});s.home.currentValue=500000;},0,-2341.10536489225,278445.629460946,80],
  ['fifty paths',s=>{s.numberOfSimulations=50;},.98,24479787.206572603,6074093.048540814,78]
];
for(const [name,edit,success,ending,starting,failure] of cases){test(`Android parity: ${name}`,()=>{const s=baseScenario();s.numberOfSimulations=1;edit(s);const r=runSimulation(s);assert.equal(r.successProbability,success);assert.ok(Math.abs(r.medianEndingBalance-Math.max(0,ending))<.01,`${r.medianEndingBalance} != ${ending}`);assert.ok(Math.abs(r.balanceBands[0].median-starting)<.01);assert.equal(r.medianFailureAge,failure);});}

test('budget estimate uses fixed costs and monthly spending',()=>{const b=baseScenario().budget;b.annualPropertyTaxes=4000;b.annualHomeInsurance=2000;b.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:[{monthlyAmount:1000}],creditCardBills:[{monthlyAmount:500}],cashAndAtmWithdrawals:100}];assert.equal(budgetEstimate(b),25200);});
test('invalid retirement age is rejected before simulation',()=>{const s=baseScenario();s.household.retirementAge=49;assert.match(validateScenario(s).join(' '),/Retirement age/);});
test('tax and benefit reference rules',()=>{assert.equal(taxableSocialSecurity(10000,30000,'Single'),0);assert.equal(ordinaryIncomeTax(16100,'Single',1,0,2026),0);assert.ok(Math.abs(annualBenefitAtClaimAge(30000,67)-30000)<.001);});

test('Android parity: retirement and spending decision targets',()=>{const s=baseScenario();s.spending.annualBaseSpending=71000;const result=estimateDecision(s);assert.equal(result.earliestRetirementAge,55);assert.equal(result.safeAnnualSpending,250000);assert.equal(result.earliestRetirementReadiness,0.8055555555555556);});

test('Android and legacy JSON scenarios can be imported',()=>{const android=normalizeScenario({id:'android',household:{currentAge:60,retirementAge:65},accounts:{pretax:120000,roth:0,taxable:0,cash:0},spending:{annualBaseSpending:50000},socialSecurity:{annualBenefitAt67:30000}});assert.equal(android.accounts.pretax,120000);assert.equal(android.accounts.roth,0);const legacy=normalizeScenario({id:'legacy',currentAge:50,retirementAge:67,annualSpending:75000,pretaxBalance:800000,rothBalance:100000,cashBalance:50000,socialSecurityAt67:30000});assert.equal(legacy.spending.annualBaseSpending,75000);assert.equal(legacy.accounts.cash,50000);});

test('single households do not require dormant spouse settings',()=>{const s=baseScenario();s.household.spouseCurrentAge=-1;s.socialSecurity.spouseClaimAge=99;s.guaranteedIncome.survivorPercent=2;assert.deepEqual(validateScenario(s),[]);s.household.filingStatus='Married';const errors=validateScenario(s).join(' ');assert.match(errors,/Spouse age/);assert.match(errors,/Spouse claim age/);assert.match(errors,/Guaranteed income/);});


test('Roth conversion cap accepts only supported brackets when enabled',()=>{
  const s=baseScenario();s.rothConversion.enabled=true;
  for(const rate of ROTH_CONVERSION_RATES){s.rothConversion.marginalRateCap=rate;assert.deepEqual(validateScenario(s),[]);}
  for(const rate of [.18,.23,0,.40]){s.rothConversion.marginalRateCap=rate;assert.match(validateScenario(s).join(' '),/Roth conversion cap/);}
  s.rothConversion.enabled=false;assert.deepEqual(validateScenario(s),[]);
});
