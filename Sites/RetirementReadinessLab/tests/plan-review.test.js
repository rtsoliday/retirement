import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,prepareCalendarScenario,applyBudgetEstimate,addCalendarMonths,localCalendarDate} from '../dist/model.js';
import {budgetPlanGaps,spendingInputSummary,needsPreMedicarePremium,inputTask,groupUnknownInputs,normalizeOptionalAnswers,temporaryRothValues,calculateEverydaySpending,monthlyIncomeSummary} from '../dist/plan-review.js';
import {runSimulation,runSteadySimulation} from '../dist/engine.js';

test('everyday spending subtracts only selected payments and rejects missing or excessive deductions',()=>{
  const draft={total:'6,000',mortgage:'1,500',rent:'bad',healthcare:'500',included:{mortgage:true,healthcare:true}};
  const d=calculateEverydaySpending(draft);assert.equal(d.monthly,4000);assert.equal(d.annual,48000);assert.equal(d.deductions.rent,0);
  for(const change of [{total:''},{mortgage:''},{healthcare:'-1'},{mortgage:'6000'},{total:'Infinity'}])assert.ok(calculateEverydaySpending({...draft,...change}).error);
  assert.equal(calculateEverydaySpending({total:'0'}).annual,0);
  assert.equal(calculateEverydaySpending({total:'0.30',mortgage:'0.10',healthcare:'0.20',included:{mortgage:true,healthcare:true}}).annual,0);
});

test('monthly picture reuses first cash flows and their starting price index with both income start dates',()=>{
  const s=baseScenario();Object.assign(s.household,{birthday:'1966-10-02',retirementDate:'2031-10-02',asOfDate:'2026-10-02',targetEndAge:70});
  Object.assign(s.guaranteedIncome,{annualIncome:12000,startAge:66,startAgeMonths:6});
  const r=runSimulation(s,()=>{},{includeRiskAnalysis:false}),c=r.steadySimulation.monthlyDetails[1].cashFlow;
  const future=monthlyIncomeSummary(s,r,'future'),today=monthlyIncomeSummary(s,r,'today'),index=r.todayDollars.steadyPriceIndexes[0];
  assert.equal(future.expenses,c.expenses);assert.equal(today.expenses,c.expenses/index);
  assert.equal(future.withdrawals,c.additionalWithdrawal+c.rmdDistribution+c.seppDistribution);
  assert.equal(future.income,0);assert.equal(future.date,'2031-10-02');
  assert.deepEqual(future.bridges,[{label:'Your Social Security',date:'2033-10-02',months:24},{label:'Your pension',date:'2033-04-02',months:18}]);
  assert.equal(monthlyIncomeSummary(s,{steadySimulation:r.steadySimulation},'today'),null);
});

test('income bridges include each spouse and exclude zero or already-started streams',()=>{
  const s=baseScenario();Object.assign(s.household,{birthday:'1960-10-02',retirementDate:'2026-10-02',spouseBirthday:'1962-10-02',spouseRetirementDate:'2027-10-02',asOfDate:'2026-10-02',separatePeople:true,filingStatus:'Married'});
  s.socialSecurity.claimAge=65;s.spouseIncome.annualBenefitAt67=24000;s.spouseIncome.annualPension=6000;s.spouseIncome.pensionStartAge=65;
  const r={steadySimulation:{monthlyDetails:[{month:1,cashFlow:{expenses:5000,socialSecurity:2000,guaranteedIncome:0,workingSupport:1000,additionalWithdrawal:2000,seppDistribution:0,rmdDistribution:0,incomeTax:0,earlyPenalty:0}}]}};
  const d=monthlyIncomeSummary(s,r,'future');assert.equal(d.income,3000);
  assert.deepEqual(d.bridges,[{label:'Spouse Social Security',date:'2029-10-02',months:36},{label:'Spouse pension',date:'2027-10-02',months:12}]);
});

test('monthly picture identifies an unfunded first month instead of implying savings cover it',()=>{
  const s=baseScenario();s.household.currentAge=s.household.retirementAge=67;
  s.accounts={pretax:0,roth:0,taxable:0,cash:0};s.socialSecurity.annualBenefitAt67=0;
  const r=runSimulation(s,()=>{},{includeRiskAnalysis:false}),d=monthlyIncomeSummary(s,r,'future');
  assert.ok(d.unfunded>0);assert.equal(d.income,0);
  assert.equal(d.unfunded,r.steadySimulation.monthlyDetails[1].unfundedAmount);
});

test('spousal-income bridges wait for both claims in pooled and separately owned plans',()=>{
  const s=baseScenario();Object.assign(s.household,{birthday:'1963-10-02',retirementDate:'2026-10-02',spouseBirthday:'1964-10-02',spouseRetirementDate:'2026-10-02',asOfDate:'2026-10-02',filingStatus:'Married'});
  s.socialSecurity.claimAge=67;s.socialSecurity.spouseClaimAge=65;
  const r={steadySimulation:{monthlyDetails:[{month:1,cashFlow:{}}]}};
  for(const separate of [false,true]){
    s.household.separatePeople=separate;
    assert.deepEqual(monthlyIncomeSummary(s,r,'future').bridges.find(b=>b.label==='Spouse Social Security from your record'),{label:'Spouse Social Security from your record',date:'2030-10-02',months:48});
  }
});

test('pooled spousal-income bridges match the simulated age-62 start for earlier claim choices',()=>{
  const s=baseScenario();Object.assign(s.household,{birthday:'1960-10-02',retirementDate:'2026-10-02',spouseBirthday:'1967-10-02',asOfDate:'2026-10-02',filingStatus:'Married'});
  s.socialSecurity.claimAge=67;
  s.spending.generalInflationMean=s.spending.generalInflationStdDev=0;
  for(const claimAge of [60,61,62]){
    s.socialSecurity.spouseClaimAge=claimAge;
    const steadySimulation=runSteadySimulation(s),r={steadySimulation};
    const bridge=monthlyIncomeSummary(s,r,'future').bridges.find(b=>b.label==='Spouse Social Security from your record');
    assert.deepEqual(bridge,{label:'Spouse Social Security from your record',date:'2029-10-02',months:36});
    const ownBenefit=s.socialSecurity.annualBenefitAt67/12;
    const firstSpousalPayment=steadySimulation.monthlyDetails.find(p=>p.cashFlow?.socialSecurity>ownBenefit);
    assert.equal(firstSpousalPayment.month-1,bridge.months,'The bridge ends when the simulation first pays the spousal benefit');
    assert.equal(addCalendarMonths(s.household.retirementDate,firstSpousalPayment.month-1),bridge.date);
  }
});

test('temporary Roth assumptions preserve known history and respect exhausted earlier conversions',()=>{
  const s=baseScenario(),sources={[s.id]:{_origin:'Entered','rothHistory.firstContributionYear':'Unknown'}};
  s.rothHistory.conversions=[{taxYear:2010,amount:0,taxableAmount:0},{taxYear:2020,amount:10000,taxableAmount:6000}];
  const before=structuredClone(s.rothHistory);
  assert.deepEqual(temporaryRothValues(s,sources,'rothHistory'),{firstContributionYear:2010});
  assert.deepEqual(s.rothHistory,before);
  sources[s.id]['rothHistory.contributionBasis']='Unknown';
  assert.deepEqual(temporaryRothValues(s,sources,'rothHistory'),{contributionBasis:0,firstContributionYear:2010});
  assert.deepEqual(temporaryRothValues(s,sources,'spouseRothHistory'),{});
});

test('optional answer backups retain recognized questions and choices only',()=>{
  const s=baseScenario(),raw={[s.id]:{'your-pension':'no','rent-inputs':'unsure','home-inputs':'maybe','unknown-question':'yes'},unused:{'rent-inputs':'yes'}};
  assert.deepEqual(normalizeOptionalAnswers(raw,[s]),{[s.id]:{'your-pension':'no','rent-inputs':'unsure'}});
  assert.deepEqual(normalizeOptionalAnswers(null,[s]),{[s.id]:{}});
});

test('unfinished inputs share the same task route for counts and exact edit links',()=>{
  const paths=['spouseAccounts.pretax','spouseAccounts.roth','spouseRothHistory.contributionBasis','spouseRothHistory.firstContributionYear','spouseIncome.annualBenefitAt67','spouseContributions.pretax','spouseContributions.employerPretax','spouseContributions.roth','spouseContributions.taxable','spouseContributions.cash'];
  const groups=groupUnknownInputs(paths);
  assert.equal(groups.length,3);assert.deepEqual(groups.map(g=>g.paths.length),[4,5,1]);
  assert.equal(groups.flatMap(g=>g.paths).length,10);
  assert.deepEqual(inputTask('spending.annualBaseSpending'),{key:'spending',label:'Everyday spending',step:1,task:1});
  assert.equal(inputTask('spouseRothHistory.firstContributionYear').task,0);
  assert.equal(inputTask('spouseContributions.cash').task,2);
});

test('budget reconciliation uses applied months and explicit entered zeros',()=>{
  const s=baseScenario();s.household.retirementAge=63;const sources={[s.id]:{_origin:'Sample/default'}};
  s.budget.monthlyBudgets=[{month:'2026-09',creditCardBills:[{monthlyAmount:4000}],adjustments:{mortgage:1000,rent:100,healthcare:100}}];
  assert.deepEqual(budgetPlanGaps(s,sources),[],'A draft must not imply those costs have been removed from the plan');
  applyBudgetEstimate(s);assert.deepEqual(budgetPlanGaps(s,sources).map(x=>x.key),['mortgage','rent','healthcare']);
  sources[s.id]['mortgage.monthlyPayment']='Entered';sources[s.id]['rent.monthlyRent']='Estimated';
  assert.deepEqual(budgetPlanGaps(s,sources).map(x=>x.key),['healthcare']);
  // The same latest-12-month window as the applied estimate.
  s.budget.monthlyBudgets.push(...Array.from({length:12},(_,i)=>({month:`2027-${String(i+1).padStart(2,'0')}`,adjustments:{}})));
  assert.deepEqual(budgetPlanGaps(s,sources),[]);
});

test('monthly review combines base, active mortgage, rent and age-appropriate premiums without double counting home bills',()=>{
  const s=prepareCalendarScenario(baseScenario());s.household.separatePeople=true;s.household.filingStatus='Married';
  s.household.retirementDate=addCalendarMonths(s.household.birthday,64*12);
  s.household.spouseBirthday=addCalendarMonths(s.household.birthday,-2*12);
  s.household.spouseRetirementDate=s.household.retirementDate;
  s.spending.annualBaseSpending=40800;s.mortgage.monthlyPayment=1200;s.mortgage.yearsLeft=10;s.rent.monthlyRent=200;
  s.healthcare.preMedicareMonthlyPremium=500;s.home.annualTaxesAndInsurance=6000;
  const sources={[s.id]:{_origin:'Entered'}},d=spendingInputSummary(s,sources);
  assert.equal(d.base,3400);assert.equal(d.mortgage,1200);assert.equal(d.rent,200);assert.equal(d.healthcare,500);
  assert.equal(d.preMedicareAdults,1);assert.equal(d.medicareAdults,1);assert.equal(d.total,5300);
  s.mortgage.yearsLeft=1;assert.equal(spendingInputSummary(s,sources).mortgage,0,'A paid-off mortgage is absent at retirement');
  sources[s.id]['spending.annualBaseSpending']='Unknown';assert.equal(spendingInputSummary(s,sources).total,null);
});

test('staggered healthcare review separates first-month premiums from later premium requirements',()=>{
  const today=localCalendarDate();
  for(const spouseFirst of [false,true]){
    const s=baseScenario();Object.assign(s.household,{separatePeople:true,filingStatus:'Married',birthday:addCalendarMonths(today,-(spouseFirst?60:67)*12),spouseBirthday:addCalendarMonths(today,-(spouseFirst?67:60)*12),retirementDate:spouseFirst?addCalendarMonths(today,60):today,spouseRetirementDate:spouseFirst?today:addCalendarMonths(today,60)});
    const sources={[s.id]:{'healthcare.preMedicareMonthlyPremium':'Unknown'}},d=spendingInputSummary(s,sources);
    assert.equal(d.preMedicareAdults,0);assert.equal(d.medicareAdults,1);assert.equal(d.healthcare,0);assert.equal(d.total,s.spending.annualBaseSpending/12);
    assert.equal(needsPreMedicarePremium(s),false,'Each person retires at 65 or later');
    s.household[spouseFirst?'retirementDate':'spouseRetirementDate']=addCalendarMonths(today,24);
    assert.equal(needsPreMedicarePremium(s),true,'The younger person will need premiums at their later retirement at 62');
    assert.equal(spendingInputSummary(s,sources).healthcare,0,'Those premiums are absent from the first month');
    s.budget.monthlyBudgets=[{month:'2026-09',creditCardBills:[{monthlyAmount:4000}],adjustments:{healthcare:500}}];applyBudgetEstimate(s);
    assert.deepEqual(budgetPlanGaps(s,sources).map(g=>g.key),['healthcare']);
    s.household[spouseFirst?'alreadyRetired':'spouseAlreadyRetired']=true;
    s.household[spouseFirst?'retirementDate':'spouseRetirementDate']='';
    assert.equal(spendingInputSummary(s,sources).preMedicareAdults,1,'Already-retired status uses age today');
    assert.equal(spendingInputSummary(s,sources).healthcare,null);
  }
});

test('pooled plans still include both people’s premiums at their shared retirement',()=>{
  const s=prepareCalendarScenario(baseScenario());s.household.filingStatus='Married';
  s.household.spouseBirthday=addCalendarMonths(s.household.birthday,5*12);
  const d=spendingInputSummary(s,{});assert.equal(d.preMedicareAdults,1);assert.equal(d.medicareAdults,1);
  assert.equal(d.healthcare,s.healthcare.preMedicareMonthlyPremium);assert.equal(needsPreMedicarePremium(s),true);
});
