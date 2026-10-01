import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,sampleScenarios,budgetEstimate,applyBudgetEstimate,validateScenario,normalizeScenario,normalizeScenarios,ROTH_CONVERSION_RATES,DEFAULT_SEED} from '../dist/model.js';
import {runSimulation,runOne,JavaRandom,estimateDecision,searchDecision,candidateReadiness,sampleDeathAge} from '../dist/engine.js';
import {taxableSocialSecurity,taxableOrdinaryIncome,ordinaryIncomeTax,rothConversionPlan} from '../dist/tax.js';
import {annualBenefitAtClaimAge} from '../dist/social-security.js';

// Historical comparison seed; the site default seed differs.
const REFERENCE_SEED=20260429;
// Keep the historical Android comparison inputs independent of site defaults.
function referenceScenario(){const s=baseScenario();Object.assign(s.household,{currentAge:50,spouseCurrentAge:50});s.accounts={pretax:800000,roth:100000,taxable:0,cash:50000};s.rothHistory.contributionBasis=100000;s.withdrawalStrategy.applyEarlyWithdrawalPenalty=false;return s;}

// Seeded web snapshots follow independent annual-moment, cash-flow, COLA,
// Roth, and death-month regressions. Monthly mortality preserves the annual
// probabilities and random draw counts, but changes death endpoints, care,
// and survivor timing. Endings differ from the former birthday-only model
// and Android; failed endings remain zero.
const cases=[
  ['base',s=>{},1,8729176.514236286,2963085.4111460363,null],
  ['zero',s=>{Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:67});Object.assign(s.accounts,{pretax:0,roth:0,taxable:0,cash:100000});Object.assign(s.spending,{annualBaseSpending:12000,generalInflationMean:0,generalInflationStdDev:0});Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});s.socialSecurity.annualBenefitAt67=0;Object.assign(s.market,{preRetirementMeanReturn:0,preRetirementStdDev:0,stockMeanReturn:0,stockStdDev:0,bondMeanReturn:0,bondStdDev:0});s.longTermCare.enabled=false;},1,79973.35771513257,100000,null],
  ['married',s=>Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:55}),1,3635039.4832704654,2963085.4111460363,null],
  ['early',s=>{s.household.retirementAge=55;s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;},1,2577942.2726794328,1377225.4206466398,null],
  ['roth',s=>{s.rothConversion.enabled=true;},1,8625526.407125238,2963085.4111460363,null],
  ['home',s=>{Object.assign(s.accounts,{pretax:100000,roth:0,cash:0});s.home.currentValue=500000;},0,0,321452.59335404605,82.5],
  ['fifty paths',s=>{s.numberOfSimulations=50;},1,20504199.304398973,5993512.06391697,null]
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
  assert.equal(result.earliestRetirementAge,53);assert.equal(result.earliestRetirementReadiness,0.8166666666666667);
  assert.equal(result.safeAnnualSpending,250000);assert.equal(result.safeSpendingAtSearchLimit,true);assert.equal(result.safeSpendingSearchLimit,250000);
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=true;const withPenalty=estimateDecision(s);
  assert.equal(withPenalty.earliestRetirementAge,55);assert.equal(withPenalty.earliestRetirementReadiness,0.8722222222222222);
});
test('safe spending below the search limit is not flagged as a lower bound',()=>{
  const s=baseScenario();s.household.currentAge=64;s.household.retirementAge=65;s.accounts={pretax:300000,roth:0,taxable:0,cash:0};
  const result=estimateDecision(s,.8,60);
  assert.ok(result.safeAnnualSpending!==null&&result.safeAnnualSpending<result.safeSpendingSearchLimit);assert.equal(result.safeSpendingAtSearchLimit,false);
});

test('spending targets do not recommend amounts below the retained home bills',async()=>{
  const s=baseScenario();Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:80});
  s.accounts={pretax:0,roth:10000,taxable:0,cash:0};s.rothHistory={contributionBasis:10000,firstContributionYear:2021,conversions:[],needsReview:false};
  Object.assign(s.spending,{annualBaseSpending:60000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  for(const key of Object.keys(s.market))s.market[key]=0;
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});
  s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;s.home.currentValue=300000;s.home.annualTaxesAndInsurance=30000;
  const original=structuredClone(s);
  assert.deepEqual(validateScenario(s),[]);
  // The old search claimed $1,500 at 100% readiness by omitting the $30,000
  // bills before the home sale. No feasible $500 candidate meets 80% here.
  const sequential=estimateDecision(s,.8,50);
  assert.equal(sequential.safeAnnualSpending,null);assert.equal(sequential.safeSpendingReadiness,null);
  assert.equal(sequential.safeSpendingAtSearchLimit,false);
  assert.deepEqual(await searchDecision(s,{simulationCount:50,concurrency:4}),sequential);
  assert.deepEqual(s,original);
});

test('spending targets round the home-bill minimum up to a feasible $500 amount',async()=>{
  for(const [bills,minimum] of [[0,0],[30000,30000],[30000.01,30500],[250000,250000]]){
    const s=baseScenario();s.home.annualTaxesAndInsurance=bills;
    const result=await searchDecision(s,{simulationCount:50,concurrency:4,evaluate:task=>{
      if(task.kind==='age')return .1;
      assert.ok(task.value>=bills,'Every tested spending amount must cover the home bills.');
      return task.value<=minimum?1:.1;
    }});
    assert.equal(result.safeAnnualSpending,minimum);assert.equal(result.safeSpendingReadiness,1);
    // If only amounts below the rounded minimum would qualify, there is no
    // feasible target, including when the entered bills contain cents.
    const exhausted=await searchDecision(s,{simulationCount:50,concurrency:4,evaluate:task=>task.kind==='spending'&&task.value<minimum?1:.1});
    assert.equal(exhausted.safeAnnualSpending,null);assert.equal(exhausted.safeSpendingReadiness,null);
  }
});

test('spending targets find no amount when home bills exceed the search ceiling',async()=>{
  for(const [bills,spending] of [[250000.01,60000],[1000000.01,1000000.01]]){
    const s=baseScenario();s.home.annualTaxesAndInsurance=bills;s.spending.annualBaseSpending=spending;
    let spendingChecks=0;
    const result=await searchDecision(s,{simulationCount:50,concurrency:4,evaluate:task=>{
      if(task.kind==='spending')spendingChecks++;
      return 1;
    }});
    assert.equal(result.safeAnnualSpending,null);assert.equal(result.safeSpendingReadiness,null);
    assert.equal(result.safeSpendingAtSearchLimit,false);assert.equal(spendingChecks,0);
  }
});

test('extreme spending is rejected before target search and valid large plans have a bounded search',()=>{
  const invalid=baseScenario();invalid.spending.annualBaseSpending=1e20;
  assert.match(validateScenario(invalid).join(' '),/supported dollar range/);
  assert.throws(()=>estimateDecision(invalid,.8,50,59),/supported dollar range/);
  const s=baseScenario();Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:66});
  s.accounts={pretax:0,roth:5e13,taxable:0,cash:0};s.spending.annualBaseSpending=1e12;
  s.longTermCare.enabled=false;const original=structuredClone(s);
  assert.deepEqual(validateScenario(s),[]);
  const decision=estimateDecision(s,.8,50,65);
  assert.equal(decision.safeSpendingSearchLimit,1000000);assert.equal(decision.safeAnnualSpending,1000000);
  assert.equal(decision.safeSpendingAtSearchLimit,true);assert.deepEqual(s,original);
});

test('all modeled dollar inputs reject magnitudes that can overflow calculations',()=>{
  const fields=['accounts.pretax','accounts.roth','accounts.taxable','accounts.cash','spending.annualBaseSpending','mortgage.monthlyPayment','mortgage.currentBalance','rent.monthlyRent','home.currentValue','home.annualTaxesAndInsurance','healthcare.preMedicareMonthlyPremium','socialSecurity.annualBenefitAt67','guaranteedIncome.annualIncome','longTermCare.annualCost','budget.annualPropertyTaxes','budget.annualHomeInsurance','budget.annualAutoInsurance','budget.retirementAnnualAdjustment'];
  for(const field of fields){
    const s=baseScenario(),[section,key]=field.split('.');s[section][key]=1e308;
    assert.match(validateScenario(s).join(' '),/supported dollar range/,field);
    assert.throws(()=>runSimulation(s),/supported dollar range/,field);
  }
  const applied=baseScenario();applied.budget.isAppliedToAnnualBaseSpending=true;applied.budget.appliedAnnualHomeCosts=1e308;
  assert.throws(()=>runSimulation(applied),/supported dollar range/);
  const imported=normalizeScenario({budget:{monthlyBudgets:[{month:'2026-01',checkingSavingsBills:[{monthlyAmount:1e308}]}]}});
  assert.match(validateScenario(imported).join(' '),/supported dollar range/);
});

test('a nonfinite path fails explicitly instead of counting as a successful retirement',()=>{
  for(const preRetirement of [true,false]){
    const s=baseScenario();if(!preRetirement)s.household.retirementAge=s.household.currentAge;
    const rng={normal:()=>1e6,nextDouble:()=>1};
    assert.throws(()=>runOne(s,rng),/finite|numeric range/);
  }
  const s=baseScenario();s.accounts.pretax=1e308;
  assert.throws(()=>runOne(s,new JavaRandom(s.seed)),/finite|numeric range/);
});

test('spending targets find qualifying amounts when zero spending changes allocation and fails',()=>{
  const s=baseScenario();Object.assign(s.household,{currentAge:60,retirementAge:60,targetEndAge:100});
  s.accounts={pretax:0,roth:100000,taxable:0,cash:0};
  Object.assign(s.spending,{annualBaseSpending:5000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  Object.assign(s.market,{preRetirementMeanReturn:0,preRetirementStdDev:0,stockMeanReturn:.1,stockStdDev:0,bondMeanReturn:-.02,bondStdDev:0});
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});
  s.socialSecurity.annualBenefitAt67=0;Object.assign(s.longTermCare,{enabled:true,annualCost:50000,averageDurationYears:3});
  for(const key of Object.keys(s.postRetirementAllocation))s.postRetirementAllocation[key]=1;s.postRetirementAllocation.stock50xOrMore=0;
  const screening=structuredClone(s);screening.seed+=20000;screening.numberOfSimulations=50;screening.spending.annualBaseSpending=0;
  assert.ok(runSimulation(screening).successProbability<.8);
  screening.spending.annualBaseSpending=5000;assert.ok(runSimulation(screening).successProbability>=.8);
  const result=estimateDecision(s,.8,50,60);
  assert.ok(result.safeAnnualSpending>=5000);assert.equal(result.safeAnnualSpending%500,0);assert.equal(result.safeSpendingAtSearchLimit,false);
  screening.spending.annualBaseSpending=result.safeAnnualSpending;
  assert.equal(result.safeSpendingReadiness,runSimulation(screening).successProbability);
  for(let spending=result.safeAnnualSpending+500;spending<=result.safeSpendingSearchLimit;spending+=500){screening.spending.annualBaseSpending=spending;assert.ok(runSimulation(screening,()=>{},{includePathPoints:false,includeRiskAnalysis:false}).successProbability<.8);}
});

test('spending targets remain attainable after an applied budget is replaced in the editor',()=>{
  const s=baseScenario();Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:75});
  s.accounts={pretax:0,roth:100000,taxable:0,cash:0};s.home.currentValue=300000;
  Object.assign(s.spending,{spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  for(const key of Object.keys(s.market))s.market[key]=0;
  Object.assign(s.healthcare,{preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0,includeMedicarePremiums:false});
  s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;
  s.budget.annualPropertyTaxes=20000;
  s.budget.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:[{monthlyAmount:2500}],creditCardBills:[]}];
  applyBudgetEstimate(s);assert.equal(s.spending.annualBaseSpending,50000);
  const original=structuredClone(s),decision=estimateDecision(s,.8,50,65);
  assert.notEqual(decision.safeAnnualSpending,null);
  const entered=structuredClone(s);entered.spending.annualBaseSpending=decision.safeAnnualSpending;
  entered.budget.isAppliedToAnnualBaseSpending=false;entered.budget.estimateNeedsReview=true;
  entered.seed+=20000;entered.numberOfSimulations=decision.simulationCount;
  const result=runSimulation(entered,()=>{},{includePathPoints:false});
  assert.ok(result.successProbability>=decision.targetReadiness);
  assert.equal(result.successProbability,decision.safeSpendingReadiness);
  entered.spending.annualBaseSpending+=500;
  assert.ok(runSimulation(entered,()=>{},{includePathPoints:false}).successProbability<decision.targetReadiness);
  assert.deepEqual(s,original,'The search must preserve the original applied budget');
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

test('sample plans retain their comparison seed and use ten-path previews',()=>{
  for(const s of sampleScenarios()){
    assert.equal(s.household.currentAge,60);assert.equal(s.household.spouseCurrentAge,60);
    assert.equal(s.accounts.pretax,500000);assert.equal(s.accounts.roth,50000);
    assert.equal(s.seed,DEFAULT_SEED);assert.equal(s.numberOfSimulations,10);
    const r=runSimulation(s);assert.equal(r.provenance.randomSeed,DEFAULT_SEED);assert.equal(r.provenance.simulationCount,10);assert.equal(r.riskBreakdown.simulationCount,10);assert.match(r.riskBreakdown.summary,/preview only/);
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

test('a 5% spending cut never adds shortfalls for a plan that relies on selling the home',()=>{
  // With a $30,000 property tax budget inside $60,000 of spending, the old
  // sensitivity check dropped the post-sale reduction and reported 26 added shortfalls.
  const s=baseScenario();Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:105});
  s.accounts={pretax:0,roth:0,taxable:0,cash:0};s.rothHistory={contributionBasis:0,firstContributionYear:0,conversions:[],needsReview:false};
  s.spending.spendingPathModel='Flat';s.home.currentValue=900000;s.socialSecurity.annualBenefitAt67=24000;s.longTermCare.enabled=false;s.numberOfSimulations=128;
  s.budget.annualPropertyTaxes=30000;s.budget.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:[{monthlyAmount:2500}],creditCardBills:[]}];applyBudgetEstimate(s);
  const check=runSimulation(s,()=>{},{includePathPoints:false}).riskBreakdown.checks.find(c=>c.key==='spending');
  assert.equal(check.introducedShortfalls,0);assert.ok(check.avoidedShortfalls>0);
});

// Readiness from a small table, delayed at random so completion order varies.
function syntheticSearch(s,readiness,{concurrency=4,seed=11}={}){
  const pause=()=>new Promise(resolve=>setTimeout(resolve,(seed=seed*16807%2147483647)%4));
  return searchDecision(s,{concurrency,simulationCount:50,evaluate:async task=>{await pause();return readiness(task);}});
}
test('parallel target searches keep the sequential answer and error order',async()=>{
  const s=baseScenario();Object.assign(s.household,{currentAge:60,retirementAge:62,targetEndAge:95});s.spending.annualBaseSpending=90100;
  const table=({kind,value})=>kind==='age'?(value===64||value===67?.85:.5):(value===120000||value===90000||value===270300?.9:.1);
  for(const concurrency of [1,3,8]){
    const found=await syntheticSearch(s,table,{concurrency});
    assert.equal(found.earliestRetirementAge,64);assert.equal(found.earliestRetirementReadiness,.85);
    assert.equal(found.safeAnnualSpending,120000);assert.equal(found.safeSpendingReadiness,.9);assert.equal(found.safeSpendingAtSearchLimit,false);
    assert.equal(found.retirementAgeSearchStart,60);assert.equal(found.retirementAgeSearchEnd,70);
  }
  // A candidate past the answer is never reached by a sequential scan, so its error is ignored.
  const skipped=task=>{if(task.kind==='spending'&&task.value<120000)throw new Error('not reached');return table(task);};
  assert.equal((await syntheticSearch(s,skipped)).safeAnnualSpending,120000);
  // An error before the answer is reported, as a sequential scan would report it.
  await assert.rejects(syntheticSearch(s,task=>{if(task.kind==='spending'&&task.value===150000)throw new Error('reached first');return table(task);}),/reached first/);
  // A qualifying top candidate checks the unrounded limit ($270,300) before claiming "At least".
  const top=await syntheticSearch(s,task=>task.kind==='spending'&&task.value>=270000?.9:table(task));
  assert.equal(top.safeAnnualSpending,270000);assert.equal(top.safeSpendingAtSearchLimit,true);assert.equal(top.safeSpendingSearchLimit,270300);
});
test('parallel target searches match the sequential engine search exactly',async()=>{
  const wealthy=baseScenario();Object.assign(wealthy.household,{currentAge:64,retirementAge:65,targetEndAge:80});
  wealthy.accounts={pretax:3000000,roth:0,taxable:0,cash:50000};wealthy.rothHistory={contributionBasis:0,firstContributionYear:0,conversions:[],needsReview:false};wealthy.longTermCare.enabled=false;
  const late=structuredClone(wealthy);Object.assign(late.household,{currentAge:72,retirementAge:72,targetEndAge:80});late.accounts.pretax=1000000;
  for(const s of [wealthy,late]){
    const sequential=estimateDecision(s,.8,50,70);let seed=5,updates=0;
    const pause=()=>new Promise(resolve=>setTimeout(resolve,(seed=seed*16807%2147483647)%3));
    const parallel=await searchDecision(s,{concurrency:4,simulationCount:50,evaluate:async task=>{await pause();return candidateReadiness(s,task);},onProgress:()=>updates++});
    assert.deepEqual(parallel,sequential);assert.ok(updates>1);
  }
  assert.equal(estimateDecision(late,.8,50,70).earliestRetirementAge,null);
});
