import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import * as model from '../dist/model.js';
import * as format from '../dist/result-format.js';
import * as guidance from '../dist/ux-guidance.js';
import * as withdrawalsView from '../dist/withdrawals-view.js';
import * as growthHelper from '../dist/growth-helper.js';
import * as moneyInput from '../dist/money-input.js';
import * as planReview from '../dist/plan-review.js';
import * as resultCaching from '../dist/result-cache.js';
import * as frontierView from '../dist/frontier-view.js';
import * as savingsTargets from '../dist/savings-targets.js';
import * as planLab from '../dist/plan-lab.js';
import * as planLabView from '../dist/plan-lab-view.js';
import {runSimulation} from '../dist/engine.js';

test('employer Roth editor preserves records, monthly deposits and history through copies and backups',async()=>{
  const a=app();a.state.setupSection=1;a.state.view='setup';
  await a.click('account-answer',{answer:'yes'});
  await a.click('add-employer-roth');const s=a.current(),stem='employerRothAccounts.0.';
  assert.match(a.setup(),/data-field="employerRothAccounts.0.contributionBasis"/);
  assert.ok(guidance.unknownInputPaths(s,a.state.inputSources).includes(stem+'balance'));
  const count=a.workers.length;await a.run();assert.equal(a.workers.length,count);
  for(const [key,type,value] of [['name','text','My 403b'],['type','select','403b'],['balance','money','100000'],['contributionBasis','money','60000'],['firstContributionYear','number','2021']])await a.change('#main',{dataset:{field:stem+key,type},value});
  await a.change('#main',{dataset:{entryPeriod:stem+'annualContribution'},value:'month'});
  await a.change('#main',{dataset:{field:stem+'annualContribution',type:'money'},value:'500'});
  assert.equal(s.employerRothAccounts[0].annualContribution,6000);
  a.state.setupSection=5;assert.match(a.setup(),/Future savings \/ year[\s\S]*You \$6,000/);a.state.setupSection=1;
  await a.click('add-employer-conversion',{account:'0'});
  for(const [key,value] of [['taxYear','2025'],['amount','10000'],['taxableAmount','8000']])await a.change('#main',{dataset:{field:stem+'conversions.0.'+key,type:key==='taxYear'?'number':'money'},value});
  const report=a.reportText(s,null);assert.match(report,/My 403b · Roth 403\(b\) · You/);assert.match(report,/Employer Roth value today: \$100,000/);assert.match(report,/Remaining after-tax contributions and converted principal: \$60,000/);
  assert.doesNotMatch(report,/none have been entered/);
  const expected=structuredClone(s.employerRothAccounts);
  const restored=app(a.saved());assert.deepEqual(structuredClone(restored.current().employerRothAccounts),expected);assert.equal(restored.saved().entryPeriods[s.id][stem+'annualContribution'],'month');
  await a.click('new-scenario');assert.deepEqual(structuredClone(a.current().employerRothAccounts),expected);
  await a.click('export-backup');const backup=JSON.parse(await a.downloads.at(-1).blob.text()),imported=app();
  await imported.change('#import-file',{files:[{text:async()=>JSON.stringify(backup)}],value:'backup.json'});
  assert.deepEqual(structuredClone(imported.current().employerRothAccounts),expected);
});

test('removing employer records and conversion rows reindexes review metadata without stale Unknown blockers',async()=>{
  const a=app();a.state.setupSection=1;
  for(let i=0;i<2;i++)await a.click('add-employer-roth');const s=a.current();
  await a.change('#main',{dataset:{field:'employerRothAccounts.1.balance',type:'money'},value:'8000'});
  await a.click('add-employer-conversion',{account:'1'});await a.click('add-employer-conversion',{account:'1'});
  await a.change('#main',{dataset:{field:'employerRothAccounts.1.conversions.1.amount',type:'money'},value:''});
  await a.click('remove-employer-conversion',{account:'1',index:'0'});
  assert.equal(a.state.inputSources[s.id]['employerRothAccounts.1.conversions.0.amount'],'Unknown');
  await a.click('remove-employer-roth',{account:'0'});
  assert.equal(s.employerRothAccounts[0].balance,8000);assert.equal(a.state.inputSources[s.id]['employerRothAccounts.0.balance'],'Entered');
  assert.equal(a.state.inputSources[s.id]['employerRothAccounts.0.conversions.0.amount'],'Unknown');
  await a.click('remove-employer-roth',{account:'0'});assert.deepEqual(guidance.unknownInputPaths(s,a.state.inputSources),[]);
  assert.deepEqual(structuredClone(app(a.saved()).current().employerRothAccounts),[]);
});

test('employer optional dates can be cleared and a known zero settles initially unknown funding history',async()=>{
  const a=app();a.state.setupSection=1;await a.click('add-employer-roth');const s=a.current();
  await a.change('#main',{dataset:{field:'employerRothAccounts.0.accessDate',type:'date'},value:'2035-01-01'});
  await a.change('#main',{dataset:{field:'employerRothAccounts.0.accessDate',type:'date'},value:''});
  assert.equal(s.employerRothAccounts[0].accessDate,'');
  await a.change('#main',{dataset:{field:'employerRothAccounts.0.balance',type:'money'},value:'0'});
  assert.deepEqual(guidance.unknownInputPaths(s,a.state.inputSources),[]);
  assert.equal(a.state.inputSources[s.id]['employerRothAccounts.0.firstContributionYear'],'Entered');
});

test('previously applied budgets explain unfinished payment review without changing saved spending',async()=>{
  const s=model.baseScenario();s.budget.monthlyBudgets=[{month:'2026-09',creditCardBills:[{monthlyAmount:4000}]}];model.applyBudgetEstimate(s);
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='budget';
  const before=structuredClone(a.current()),stored=a.stored(),html=a.budget();
  assert.match(html,/>Review needed</);assert.doesNotMatch(html,/>Applied</);
  assert.match(html,/Previously applied\. Your plan still uses \$48,000\.00 \/ year/);
  assert.match(html,/Review each month’s housing and health payment choices below before applying again/);
  assert.match(html,/data-action="apply-budget" disabled/);
  await a.click('review-budget-deductions');assert.deepEqual(structuredClone(a.current()),before);assert.equal(a.stored(),stored);
  await confirmBudgetPaymentChoices(a);await a.click('apply-budget');
  assert.equal(a.current().spending.annualBaseSpending,48000);assert.match(a.budgetSummary(),/>Applied</);
  const restored=app(a.saved());restored.state.view='budget';const saved=restored.stored();
  assert.match(restored.budgetSummary(),/>Review needed</);
  assert.match(restored.budgetSummary(),/Confirm the housing and health deductions below before applying again/);
  await restored.change('#main',{dataset:{costConfirm:''},checked:true});
  assert.match(restored.budgetSummary(),/>Applied</);assert.equal(restored.stored(),saved);
});

test('an applied budget with a selected payment missing its amount cannot appear fully reviewed',async()=>{
  const s=model.baseScenario();s.budget.monthlyBudgets=[{month:'2026-09',creditCardBills:[{monthlyAmount:4000}],includedPayments:{mortgage:true,rent:false,healthcare:false}}];model.applyBudgetEstimate(s);
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='budget';
  await a.change('#main',{dataset:{costConfirm:''},checked:true});
  assert.match(a.budgetSummary(),/>Review needed</);assert.equal(a.budgetCostsReviewed(),false);
  assert.match(a.budget(),/data-action="apply-budget" disabled/);
  assert.equal(a.current().spending.annualBaseSpending,48000);
});

test('saved assumptions stay intact and restoring examples uses the new preview inputs',async()=>{
  const s=model.prepareCalendarScenario(model.baseScenario());s.name='My saved plan';
  const a=app({scenarios:[s],selectedId:s.id}),before=structuredClone(s);
  assert.equal(a.current().market.stockMeanReturn,.133);assert.equal(a.current().spending.annualBaseSpending,75000);
  assert.deepEqual(structuredClone(a.current()),before);
  await a.click('reset-assumptions');
  assert.equal(a.current().market.stockMeanReturn,.133);assert.equal(a.current().market.preRetirementMeanReturn,.133);
  assert.equal(a.current().spending.annualBaseSpending,75000);assert.equal(a.current().seed,model.DEFAULT_SEED);
  assert.deepEqual(a.current().accounts,{pretax:175000,roth:17500,taxable:0,cash:17500});
  assert.equal(a.current().rothHistory.contributionBasis,17500);
  assert.equal(a.current().name,'My saved plan');
});

test('Reset all plans replaces edited or deleted examples and custom plans with the original three',async()=>{
  const defaults=structuredClone(app().state.scenarios),custom=structuredClone(defaults[0]);custom.id='custom-plan';custom.name='My plan';
  for(const plans of [[custom],[...defaults.map(s=>({...structuredClone(s),name:'Edited example',accounts:{...s.accounts,pretax:900000}})),custom]]){
    const preferences=new Map([['retirement-setup-step',JSON.stringify({id:'base-plan',section:5})],['retirement-setup-modes',JSON.stringify({'base-plan':true,'custom-plan':true})],['retirement-dollar-basis','future']]);
    const a=app({scenarios:plans,selectedId:custom.id,inputSources:{'base-plan':{_origin:'Entered'},'custom-plan':{_origin:'Entered'}},optionalAnswers:{'base-plan':{'home-inputs':'yes'}},accountChecks:{'base-plan':'yes','custom-plan':'unsure'},entryPeriods:{'base-plan':{'spending.annualBaseSpending':'month'}}},{preferences});
    a.state.view='scenarios';seedExploration(a);await completeCachedRun(a);
    assert.match(a.scenarios(),/data-action="reset-plans"[^>]*>Reset all plans</);
    await a.click('reset-plans');
    assert.deepEqual(structuredClone(a.state.scenarios),defaults);assert.equal(a.current().id,'base-plan');
    assert.deepEqual([...a.state.exampleIds],defaults.map(s=>s.id));assert.equal(a.state.results.size,0);assert.equal(a.state.resultSaveStatus.size,0);assert.equal(a.state.restoredResults.size,0);assertCleared(a);
    assert.equal(a.state.hasStartedPlan,false);assert.equal(a.state.guided,false);assert.equal(a.state.setupSection,0);
    assert.deepEqual(a.saved().accountChecks,{});
    for(const s of defaults){assert.deepEqual(a.saved().inputSources[s.id],{_origin:'Sample/default'});assert.deepEqual(a.saved().optionalAnswers[s.id],{});assert.deepEqual(a.saved().entryPeriods[s.id],{});}
    assert.equal(preferences.has('retirement-setup-step'),false);assert.equal(preferences.has('retirement-setup-modes'),false);assert.equal(preferences.get('retirement-dollar-basis'),'future');
    const reloaded=app(a.saved(),{preferences});assert.deepEqual(structuredClone(reloaded.state.scenarios),defaults);assert.equal(reloaded.state.hasStartedPlan,false);assert.equal(reloaded.state.guided,false);
    await a.click('reset-plans');assert.deepEqual(structuredClone(a.state.scenarios),defaults);
  }
});

test('canceling Reset all plans preserves saved plans, results and setup preferences',async()=>{
  const {saved,resultRecords}=await twoCachedPlans(),preferences=new Map([['retirement-setup-modes',JSON.stringify({'base-plan':true})]]);
  let confirmation='';const a=app(saved,{resultRecords,preferences,confirm:message=>{confirmation=message;return false;}});await a.resultRestoreReady;
  const stored=a.stored(),records=structuredClone([...resultRecords]),modes=preferences.get('retirement-setup-modes');seedExploration(a);
  await a.click('reset-plans');
  assert.match(confirmation,/removes all saved plans and results.*3 original example plans/);
  assert.equal(a.stored(),stored);assert.deepEqual(structuredClone(a.state.scenarios),saved.scenarios);assert.deepEqual([...resultRecords],records);assert.equal(a.state.results.size,2);assert.ok(a.state.labResults);assert.equal(preferences.get('retirement-setup-modes'),modes);
});

test('Reset all plans retains Pro access and applies the normal Pro path default',async()=>{
  const a=app();a.state.access.tier='pro';await a.click('reset-plans');
  assert.equal(a.state.access.tier,'pro');assert.equal(a.state.scenarios.length,3);
  for(const s of a.state.scenarios){assert.equal(s.numberOfSimulations,10000);assert.equal(s.simulationPathsCustomized,false);}
});

test('Reset all plans preserves newer or unsavable stored plans and reports the failed save',async()=>{
  for(const fail of [true,false]){
    const storage={fail},a=app({scenarios:model.sampleScenarios(),selectedId:'base-plan'},{storage});
    if(!fail)storage.raw=JSON.stringify({scenarios:[model.baseScenario()],selectedId:'base-plan'});
    const before=a.stored();await a.click('reset-plans');
    assert.equal(a.stored(),before);assert.match(a.state.message,fail?/could not be saved/:/changed in another tab/);assert.doesNotMatch(a.state.message,/have been restored/);
  }
});

test('already-retired setup runs without a separation date or future savings and preserves both date modes',async()=>{
  const s=model.prepareCalendarScenario(model.baseScenario()),date=s.household.retirementDate;
  const a=app({scenarios:[s],selectedId:s.id,inputSources:{[s.id]:{_origin:'Entered','contributions.pretax':'Unknown'}}});
  await a.click('retirement-status',{owner:'you',mode:'retired'});
  assert.equal(a.current().household.retirementDate,'');assert.equal(a.current().household.alreadyRetired,true);
  assert.match(a.setup(),/Forecast starts today/);assert.match(a.reportSummaryText(a.current()),/Already retired/);
  a.state.setupSection=1;assert.match(a.setup(),/stops future deposits/);
  const r=await completeCachedRun(a);assert.ok(r);assert.match(a.results(),/Your monthly picture/);
  assert.equal(a.workers.at(-1).data.scenario.household.alreadyRetired,true);
  const restored=app(a.saved());assert.equal(restored.current().household.alreadyRetired,true);
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:'2020-01-01'});
  assert.equal(a.current().household.retirementDate,'2020-01-01');
  await a.click('retirement-status',{owner:'you',mode:'future'});assert.equal(a.current().household.retirementDate,date);
  await a.click('retirement-status',{owner:'you',mode:'retired'});assert.equal(a.current().household.retirementDate,'2020-01-01');
  await a.click('new-scenario');assert.equal(a.current().household.alreadyRetired,true);
  await a.click('export-backup');const backup=JSON.parse(await a.downloads.at(-1).blob.text());
  const imported=app();await imported.change('#import-file',{files:[{text:async()=>JSON.stringify(backup)}],value:'backup.json'});
  assert.equal(imported.state.scenarios.find(p=>p.id===s.id).household.alreadyRetired,true);
});

test('switching to Individual keeps invalid spouse savings editable without blocking a forecast',async()=>{
  const a=app();await a.click('household-choice',{kind:'couple'});
  await a.change('#main',{dataset:{field:'spouseContributions.pretax',type:'money'},value:'-100'});
  await a.click('household-choice',{kind:'individual'});a.state.setupSection=1;
  assert.doesNotMatch(a.setup(),/id="f-spouseContributions-pretax"/);
  assert.ok(await completeCachedRun(a));
  assert.equal(a.current().spouseContributions.pretax,-100);
  const restored=app(a.saved());assert.equal(restored.current().spouseContributions.pretax,-100);
  assert.deepEqual(model.validateScenario(restored.current()),[]);
  await a.click('household-choice',{kind:'couple'});const count=a.workers.length;await a.run();
  assert.equal(a.workers.length,count);assert.match(a.state.message,/Spouse savings contributions/);
});

test('Already retired preserves unused savings drafts and validates them again in future mode',async()=>{
  const a=app();await a.change('#main',{dataset:{field:'contributions.pretax',type:'money'},value:'-100'});
  await a.click('retirement-status',{owner:'you',mode:'retired'});a.state.setupSection=1;
  assert.doesNotMatch(a.setup(),/id="f-contributions-pretax"/);
  const r=await completeCachedRun(a);assert.ok(r);
  assert.equal(r.steadySimulation.monthlyDetails[1].cashFlow.savingsContributions,0);
  assert.equal(a.current().contributions.pretax,-100);
  assert.deepEqual(model.validateScenario(app(a.saved()).current()),[]);
  await a.click('retirement-status',{owner:'you',mode:'future'});const count=a.workers.length;await a.run();
  assert.equal(a.workers.length,count);assert.match(a.state.message,/You savings contributions/);
});

test('No pension ignores retained invalid details and a positive pension requires their correction',async()=>{
  const a=app();await a.click('optional-answer',{question:'your-pension',answer:'yes'});
  for(const [path,type,value] of [['guaranteedIncome.annualIncome','money','6000'],['guaranteedIncome.startAge','number','-1.5'],['guaranteedIncome.startAgeMonths','number','12'],['guaranteedIncome.annualIncrease','percent','-200']]){
    await a.change('#main',{dataset:{field:path,type},value});
  }
  await a.click('optional-answer',{question:'your-pension',answer:'no'});
  const before=structuredClone(a.current().guaranteedIncome),r=await completeCachedRun(a);assert.ok(r);
  assert.ok(r.steadySimulation.monthlyDetails.every(p=>!p.cashFlow||p.cashFlow.guaranteedIncome===0));
  assert.deepEqual(structuredClone(a.current().guaranteedIncome),before);
  assert.deepEqual(model.validateScenario(app(a.saved()).current()),[]);
  await a.click('optional-answer',{question:'your-pension',answer:'yes'});
  await a.change('#main',{dataset:{field:'guaranteedIncome.annualIncome',type:'money'},value:'6000'});
  const count=a.workers.length;await a.run();assert.equal(a.workers.length,count);
  assert.match(a.state.message,/Guaranteed income|Income start age|Month fields/);
});

for(const [prefix,amount,question,startAge] of [['guaranteedIncome','annualIncome','your-pension','startAge'],['spouseIncome','annualPension','spouse-pension','pensionStartAge']]){
  for(const method of ['typed zero','None'])test(`${question}: ${method} ignores Unknown pension details and positive income requires them`,async()=>{
    const a=app();await a.click('household-choice',{kind:'couple'});const s=a.current(),amountPath=prefix+'.'+amount;
    if(method==='typed zero')await a.click('optional-answer',{question,answer:'yes'});
    await a.change('#main',{dataset:{field:amountPath,type:'money'},value:'6000'});
    const details=[[startAge,'number'],['annualIncrease','percent'],['survivorPercent','percent']];
    for(const [key,type] of details)await a.change('#main',{dataset:{field:prefix+'.'+key,type},value:''});
    if(method==='typed zero')await a.change('#main',{dataset:{field:amountPath,type:'money'},value:'0'});
    else await a.click('input-none',{path:amountPath});
    assert.equal(a.state.optionalAnswers[s.id]?.[question],method==='typed zero'?'yes':undefined);
    assert.equal(s[prefix][amount],0);assert.equal(a.state.inputSources[s.id][amountPath],'Entered');
    const before=structuredClone(s[prefix]);a.state.guided=true;a.state.setupSection=5;
    assert.doesNotMatch(a.setup(),/data-action="run-plan" disabled/);
    assert.ok(await completeCachedRun(a));assert.deepEqual(structuredClone(s[prefix]),before);
    const restored=app(a.saved());restored.state.guided=true;restored.state.setupSection=5;
    assert.deepEqual(structuredClone(restored.current()[prefix]),before);
    for(const [key] of details)assert.equal(restored.state.inputSources[s.id][prefix+'.'+key],'Unknown');
    assert.doesNotMatch(restored.setup(),/data-action="run-plan" disabled/);
    await a.change('#main',{dataset:{field:amountPath,type:'money'},value:'6000'});
    assert.match(a.setup(),/data-action="run-plan" disabled/);
    const count=a.workers.length;await a.run();assert.equal(a.workers.length,count);assert.match(a.state.message,/Review Unknown inputs/);
  });
}

test('zero pension does not excuse an Unknown pension amount or spouse Social Security',async()=>{
  for(const [prefix,amount] of [['guaranteedIncome','annualIncome'],['spouseIncome','annualPension']]){
    const a=app();await a.click('household-choice',{kind:'couple'});const amountPath=prefix+'.'+amount;
    await a.change('#main',{dataset:{field:amountPath,type:'money'},value:'0'});
    await a.change('#main',{dataset:{field:amountPath,type:'money'},value:''});
    assert.equal(a.current()[prefix][amount],0);assert.equal(a.state.inputSources[a.current().id][amountPath],'Unknown');
    await a.run();assert.equal(a.workers.length,0);assert.match(a.state.message,/Review Unknown inputs/);
  }
  const a=app();await a.click('household-choice',{kind:'couple'});
  await a.click('input-none',{path:'spouseIncome.annualPension'});
  await a.change('#main',{dataset:{field:'spouseIncome.annualBenefitAt67',type:'money'},value:''});
  await a.run();assert.equal(a.workers.length,0);assert.match(a.state.message,/Review Unknown inputs/);
});

test('spending calculator applies a preview only and keeps housing, healthcare and budget entries',async()=>{
  const a=app(),s=a.current();a.state.view='setup';a.state.setupSection=1;s.mortgage.monthlyPayment=1800;s.healthcare.preMedicareMonthlyPremium=700;
  s.home.annualTaxesAndInsurance=4000;s.budget.monthlyBudgets=[{month:'2026-09',creditCardBills:[{monthlyAmount:5000}]}];
  const before=structuredClone(s),result={id:'finished'};a.state.results.set(s.id,result);
  for(const [key,value] of [['total','6000'],['mortgage','1500'],['healthcare','500']])await a.change('#main',{dataset:{spendingHelper:key},value});
  for(const key of ['mortgage','healthcare'])await a.change('#main',{dataset:{spendingIncluded:key},checked:true});
  assert.deepEqual(structuredClone(s),before);assert.equal(a.state.results.get(s.id),result);
  await a.click('apply-spending-calculator');assert.equal(s.spending.annualBaseSpending,48000);
  assert.equal(s.mortgage.monthlyPayment,1800);assert.equal(s.healthcare.preMedicareMonthlyPremium,700);
  assert.equal(s.home.annualTaxesAndInsurance,4000);assert.deepEqual(structuredClone(s.budget.monthlyBudgets),before.budget.monthlyBudgets);
  assert.equal(a.state.results.size,0);assert.equal(a.saved().inputSources[s.id]['spending.annualBaseSpending'],'Estimated');
  assert.equal(app(a.saved()).current().spending.annualBaseSpending,48000);
  await a.change('#main',{dataset:{spendingHelper:'mortgage'},value:'7000'});await a.click('apply-spending-calculator');
  assert.equal(s.spending.annualBaseSpending,48000);assert.match(a.state.message,/cannot exceed/);
});

test('inline plan naming preserves completed results and survives reload, copies and backups',async()=>{
  const resultRecords=new Map(),a=app(null,{resultRecords});await a.click('start-plan');
  assert.equal(a.current().name,'My retirement plan');
  const s=a.current();a.state.inputSources[s.id]._origin='Entered';for(const path of Object.keys(a.state.inputSources[s.id]))if(a.state.inputSources[s.id][path]==='Unknown')a.state.inputSources[s.id][path]='Entered';
  const r=await completeCachedRun(a);
  await a.change('#main',{dataset:{planName:''},value:'  My <retired> plan  '});
  assert.equal(a.current().name,'My <retired> plan');assert.equal(a.state.results.get(s.id),r);
  assert.match(a.scenarios(),/My &lt;retired&gt; plan/);assert.doesNotMatch(a.scenarios(),/My <retired> plan/);
  const restored=app(a.saved(),{resultRecords});await restored.resultRestoreReady;
  assert.equal(restored.current().name,'My <retired> plan');assert.equal(restored.state.results.size,1);
  await a.click('new-scenario');assert.match(a.current().name,/My <retired> plan/);
  await a.click('export-backup');const backup=JSON.parse(await a.downloads.at(-1).blob.text());assert.equal(backup.scenarios[0].name,'My <retired> plan');
});

test('budget application previews both inputs and preserves home costs unless explicitly replaced',async()=>{
  const s=model.baseScenario();s.home.annualTaxesAndInsurance=4000;
  s.budget.monthlyBudgets=[{month:'2026-09',creditCardBills:[{monthlyAmount:4000}]}];
  const a=app({scenarios:[s],selectedId:s.id,inputSources:{[s.id]:{'home.annualTaxesAndInsurance':'Entered'}}});
  a.state.view='budget';await confirmBudgetPaymentChoices(a);
  assert.match(a.budgetSummary(),/What will change in your plan/);
  assert.match(a.budgetSummary(),/\$4,000\.00 → \$4,000\.00 \/ year \(kept\)/);
  await a.click('apply-budget');assert.equal(a.current().home.annualTaxesAndInsurance,4000);
  assert.equal(a.saved().inputSources[s.id]['home.annualTaxesAndInsurance'],'Entered');
  const restored=app(a.saved());assert.equal(restored.current().home.annualTaxesAndInsurance,4000);
  await a.change('#main',{dataset:{replaceHomeCosts:''},checked:true});
  assert.match(a.budgetSummary(),/\$4,000\.00 → \$0\.00 \/ year/);
  await a.click('apply-budget');assert.equal(a.saved().scenarios[0].home.annualTaxesAndInsurance,0);
});

test('confirming an example preserves its value and completed calculation through reload, copies and backups',async()=>{
  const a=app(),s=a.current(),before=structuredClone(s),r=runSimulation(s);a.state.results.set(s.id,r);
  assert.match(a.setup(),/Use this value for Your sex/);
  await a.click('confirm-input',{path:'household.gender'});
  await a.click('confirm-input',{path:'socialSecurity.claimAge'});
  assert.deepEqual(structuredClone(s),before);assert.equal(a.state.results.get(s.id),r);
  assert.equal(a.saved().inputSources[s.id]['household.gender'],'Entered');
  assert.equal(app(a.saved()).state.inputSources[s.id]['socialSecurity.claimAge'],'Entered');
  await a.click('new-scenario');assert.equal(a.state.inputSources[a.current().id]['household.gender'],'Entered');
  await a.click('export-backup');const backup=JSON.parse(await a.downloads.at(-1).blob.text());
  const imported=app();await imported.change('#import-file',{files:[{text:async()=>JSON.stringify(backup)}],value:'backup.json'});
  assert.equal(imported.state.inputSources[s.id]['socialSecurity.claimAge'],'Entered');
  const unknown=app();await unknown.click('start-plan');await unknown.click('confirm-input',{path:'accounts.pretax'});
  assert.equal(unknown.state.inputSources[unknown.current().id]['accounts.pretax'],'Unknown');
});

test('unused pre-Medicare premiums are labeled Not needed and retained when retirement dates change',async()=>{
  const a=app();await a.click('start-plan');const s=a.current(),premium=s.healthcare.preMedicareMonthlyPremium;
  a.state.setupSection=3;
  assert.match(a.setup(),/Not needed for this plan/);assert.doesNotMatch(a.setup(),/id="f-healthcare-preMedicareMonthlyPremium"/);
  assert.match(a.reportText(s),/Pre-Medicare monthly premium: Not needed for this plan/);
  assert.doesNotMatch(a.reportText(s),/Pre-Medicare monthly premium: \$1,250/);
  const date=model.addCalendarMonths(s.household.birthday,63*12);
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:date});
  assert.match(a.element('#pre-medicare-input').innerHTML,/id="f-healthcare-preMedicareMonthlyPremium"/);
  assert.match(a.setup(),/id="f-healthcare-preMedicareMonthlyPremium"/);assert.equal(s.healthcare.preMedicareMonthlyPremium,premium);
  await a.run();assert.equal(a.workers.length,0);assert.match(a.state.message,/Unknown/);
  const couple=model.baseScenario();couple.household.filingStatus='Married';couple.household.spouseCurrentAge=50;
  const b=app({scenarios:[couple],selectedId:couple.id});b.state.setupSection=3;
  assert.match(b.setup(),/id="f-healthcare-preMedicareMonthlyPremium"/,'A younger spouse still needs a premium');
  await a.change('#main',{dataset:{field:'household.birthday',type:'date'},value:''});
  assert.match(a.element('#pre-medicare-input').innerHTML,/id="f-healthcare-preMedicareMonthlyPremium"/,'Unknown dates must not imply premiums are unnecessary');
});

async function completeCachedRun(a){
  const pending=a.run(),w=a.workers.at(-1);w.onmessage({data:{type:'result',result:runSimulation(w.data.scenario)}});
  await pending;await a.state.resultSavingPromise;return a.state.results.get(a.current().id);
}

async function twoCachedPlans(){
  const resultRecords=new Map(),a=app(null,{resultRecords});
  await completeCachedRun(a);await a.click('new-scenario');await completeCachedRun(a);
  return {saved:a.saved(),resultRecords};
}

test('disabled care keeps unused drafts through runs and reloads, then requires review when enabled',async()=>{
  const a=app();
  for(const [field,type,value] of [['longTermCare.annualCost','money','-100'],['longTermCare.averageDurationYears','number','0'],['longTermCare.averageDurationMonths','number','12']]){
    await a.change('#main',{dataset:{field,type},value});
  }
  await a.change('#main',{dataset:{field:'longTermCare.enabled',type:'checkbox'},checked:false});
  const before=structuredClone(a.current().longTermCare),r=await completeCachedRun(a);assert.ok(r);
  const restored=app(a.saved());
  assert.deepEqual(structuredClone(restored.current().longTermCare),before);
  assert.deepEqual(model.validateScenario(restored.current()),[]);
  await restored.change('#main',{dataset:{field:'longTermCare.enabled',type:'checkbox'},checked:true});
  await restored.run();assert.equal(restored.workers.length,0);
  assert.match(restored.state.message,/Month fields|Long-term care cost or duration/);
});

test('switching to Individual retains hidden support drafts without blocking runs or reloads',async()=>{
  const a=app();await a.click('household-choice',{kind:'couple'});
  await a.change('#main',{dataset:{field:'workingIncome.spouseAnnualNet',type:'money'},value:'-100'});
  await a.click('household-choice',{kind:'individual'});
  a.state.setupSection=2;assert.doesNotMatch(a.setup(),/id="f-workingIncome-spouseAnnualNet"/);
  assert.ok(await completeCachedRun(a));
  const restored=app(a.saved());assert.equal(restored.current().workingIncome.spouseAnnualNet,-100);
  assert.deepEqual(model.validateScenario(restored.current()),[]);
  await restored.click('household-choice',{kind:'couple'});await restored.run();
  assert.equal(restored.workers.length,0);assert.match(restored.state.message,/Take-home household support/);
});

test('completed results survive reload with their date and actual count, but changed inputs require a new run',async()=>{
  const resultRecords=new Map(),a=app(null,{resultRecords});a.state.view='results';
  const r=await completeCachedRun(a),s=a.current();assert.equal(resultRecords.size,1);
  const restored=app(a.saved(),{resultRecords,location:{search:'',pathname:'/',hash:'#/results'}});
  await restored.resultRestoreReady;
  assert.equal(restored.workers.length,0);assert.equal(restored.state.results.get(s.id).generatedAtEpochMillis,r.generatedAtEpochMillis);
  assert.deepEqual(structuredClone(restored.state.results.get(s.id)),structuredClone(r));
  assert.match(restored.results(),/Saved results · calculated/);assert.match(restored.results(),/Saved in this browser/);
  await restored.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'600000'});
  const changed=app(restored.saved(),{resultRecords});await changed.resultRestoreReady;
  assert.equal(changed.state.results.size,0);assert.match(changed.results(),/Your inputs are saved in this browser. Run again/);
});

test('switching plans while cached results load restores every unchanged plan',async()=>{
  const {saved,resultRecords}=await twoCachedPlans();
  let release;const gate=new Promise(resolve=>{release=resolve;});
  const a=app(saved,{resultRecords,resultStorage:{beforeLoad:()=>gate}});
  const otherId=[...resultRecords.keys()].find(id=>id!==saved.selectedId);
  await a.click('select-scenario',{id:otherId});release();await a.resultRestoreReady;
  assert.equal(a.current().id,otherId);assert.equal(a.state.results.size,2);
  for(const [id,record] of resultRecords)assert.deepEqual(structuredClone(a.state.results.get(id)),record.result);
  assert.equal(a.workers.length,0);
});

test('an edit during cache loading skips only the changed plan',async()=>{
  const {saved,resultRecords}=await twoCachedPlans();
  let release;const gate=new Promise(resolve=>{release=resolve;});
  const a=app(saved,{resultRecords,resultStorage:{beforeLoad:()=>gate}}),id=a.current().id;
  await a.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'600000'});
  release();await a.resultRestoreReady;
  assert.equal(a.state.results.has(id),false);assert.equal(a.state.results.size,1);
  assert.ok(a.state.results.has([...resultRecords.keys()].find(key=>key!==id)));
});

test('imports, resets and deletions during cache loading cannot resurrect discarded results',async()=>{
  for(const action of ['import','reset','reset-all','delete']){
    const {saved,resultRecords}=await twoCachedPlans();
    let release;const gate=new Promise(resolve=>{release=resolve;});
    const a=app(saved,{resultRecords,resultStorage:{beforeLoad:()=>gate}}),id=a.current().id;
    if(action==='import')await a.change('#import-file',{files:[{text:async()=>JSON.stringify(saved)}],value:'backup.json'});
    else if(action==='reset')await a.click('reset-assumptions');
    else if(action==='reset-all')await a.click('reset-plans');
    else await a.click('delete-scenario',{id});
    release();await a.resultRestoreReady;
    assert.equal(a.state.results.has(id),false,action);
    assert.equal(a.state.results.size,['import','reset-all'].includes(action)?0:1,action);
    if(action==='reset-all')assert.equal(resultRecords.size,0);
  }
});

test('result-storage failures retain live results and saved inputs, and importing clears older cached calculations',async()=>{
  const a=app(null,{resultStorage:{fail:true}});a.state.view='results';
  await completeCachedRun(a);assert.ok(a.saved());assert.ok(a.state.results.get(a.current().id));
  assert.match(a.results(),/run again after reloading to regenerate results/);
  const resultRecords=new Map(),b=app(null,{resultRecords});await completeCachedRun(b);
  await b.change('#import-file',{files:[{text:async()=>JSON.stringify(b.saved())}],value:'backup.json'});
  assert.equal(resultRecords.size,0);assert.equal(b.state.results.size,0);
});

test('switching plans during a result write keeps the completed result saved and restorable',async()=>{
  let release,started;
  const gate=new Promise(resolve=>{release=resolve;}),writing=new Promise(resolve=>{started=resolve;});
  const resultRecords=new Map(),a=app(null,{resultRecords,resultStorage:{beforeSave:async()=>{started();await gate;}}});
  const id=a.current().id,otherId=a.state.scenarios[1].id,pending=a.run(),worker=a.workers.at(-1);
  worker.onmessage({data:{type:'result',result:runSimulation(worker.data.scenario)}});
  await pending;await writing;
  const before=structuredClone(a.current());
  await a.click('select-scenario',{id:otherId});
  release();await a.state.resultSavingPromise;
  assert.deepEqual(structuredClone(a.state.scenarios.find(s=>s.id===id)),before);
  assert.equal(a.state.resultSaveStatus.get(id),true);assert.ok(resultRecords.has(id));
  const restored=app(a.saved(),{resultRecords});await restored.resultRestoreReady;
  assert.ok(restored.state.results.has(id));
});

test('an input edit during a result write prevents stale results from being restored',async()=>{
  let release,started;
  const gate=new Promise(resolve=>{release=resolve;}),writing=new Promise(resolve=>{started=resolve;});
  const resultRecords=new Map(),a=app(null,{resultRecords,resultStorage:{beforeSave:async()=>{started();await gate;}}});
  const id=a.current().id,pending=a.run(),worker=a.workers.at(-1);
  worker.onmessage({data:{type:'result',result:runSimulation(worker.data.scenario)}});
  await pending;await writing;
  await a.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'600000'});
  release();await a.state.resultSavingPromise;
  assert.equal(a.state.results.has(id),false);
  const restored=app(a.saved(),{resultRecords});await restored.resultRestoreReady;
  assert.equal(restored.state.results.has(id),false);
});

test('Reset all plans clears a result write still in flight for an original example',async()=>{
  let release,started;
  const gate=new Promise(resolve=>{release=resolve;}),writing=new Promise(resolve=>{started=resolve;});
  const resultRecords=new Map(),a=app(null,{resultRecords,resultStorage:{beforeSave:async()=>{started();await gate;}}});
  const pending=a.run(),worker=a.workers.at(-1);worker.onmessage({data:{type:'result',result:runSimulation(worker.data.scenario)}});await pending;await writing;
  const reset=a.click('reset-plans');release();await reset;
  assert.equal(resultRecords.size,0);assert.equal(a.state.results.size,0);assert.equal(a.state.resultSaveStatus.size,0);
  const restored=app(a.saved(),{resultRecords});await restored.resultRestoreReady;assert.equal(restored.state.results.size,0);
});

test('temporary Roth history explicitly resolves Unknown fields and survives backup, copy and reload',async()=>{
  const a=app(),before=structuredClone(a.current());
  await a.click('roth-records-unknown',{prefix:'rothHistory'});a.state.setupSection=1;
  assert.match(a.setup(),/Use temporary Roth assumptions/);
  assert.match(a.setup(),/<details id="roth-record-help-rothHistory"><summary>/,'The long record lookup stays collapsed so the alternatives remain easy to reach');
  await a.click('roth-use-estimates',{prefix:'rothHistory'});
  assert.equal(a.current().rothHistory.contributionBasis,0);
  assert.equal(a.current().rothHistory.firstContributionYear,2026);
  assert.equal(a.saved().inputSources[before.id]['rothHistory.contributionBasis'],'Estimated');
  assert.equal(a.saved().inputSources[before.id]['rothHistory.firstContributionYear'],'Estimated');
  assert.deepEqual(structuredClone(a.current().market),before.market);
  assert.deepEqual(structuredClone(a.current().accounts),before.accounts);
  assert.deepEqual(model.validateScenario(a.current()),[]);
  a.state.setupSection=5;assert.match(a.setup(),/Uses estimated Roth history/);
  assert.match(a.reportText(a.current(),null),/ROTH HISTORY ESTIMATES/);
  const restored=app(a.saved());restored.state.setupSection=5;assert.match(restored.setup(),/Uses estimated Roth history/);
  await a.click('new-scenario');assert.match(a.setup(),/Uses estimated Roth history/);
  await a.click('export-backup');const backup=JSON.parse(await a.downloads.at(-1).blob.text());
  const imported=app();await imported.change('#import-file',{files:[{text:async()=>JSON.stringify(backup)}],value:'backup.json'});
  assert.equal(imported.state.inputSources[before.id]['rothHistory.firstContributionYear'],'Estimated');
});

test('Roth illustration keeps entered records and conversions, and refuses an unknown balance',async()=>{
  const s=model.baseScenario();s.rothHistory.conversions=[{taxYear:2010,amount:10000,taxableAmount:6000}];
  const a=app({scenarios:[s],selectedId:s.id,inputSources:{[s.id]:{_origin:'Saved value; source not recorded','rothHistory.contributionBasis':'Entered','rothHistory.firstContributionYear':'Unknown'}}});
  await a.click('roth-records-unknown',{prefix:'rothHistory'});
  await a.click('roth-use-estimates',{prefix:'rothHistory'});
  assert.equal(a.current().rothHistory.contributionBasis,s.rothHistory.contributionBasis);
  assert.equal(a.current().rothHistory.firstContributionYear,2010);
  assert.deepEqual(structuredClone(a.current().rothHistory.conversions),s.rothHistory.conversions);
  assert.equal(a.saved().inputSources[s.id]['rothHistory.contributionBasis'],'Entered');
  const b=app();await b.click('start-plan');const before=structuredClone(b.current());
  await b.click('roth-use-estimates',{prefix:'rothHistory'});
  assert.deepEqual(structuredClone(b.current()),before);assert.match(b.state.message,/Enter the Roth IRA balance first/);
  await b.click('roth-finish-later');assert.equal(b.state.view,'dashboard');assert.equal(b.saved().inputSources[b.current().id]['accounts.roth'],'Unknown');
});

test('optional questions require an answer for new plans, preserve Not sure, and explicitly record No',async()=>{
  const a=app();await a.click('start-plan');const s=a.current();
  for(const path of ['guaranteedIncome.annualIncome','home.currentValue','mortgage.monthlyPayment','rent.monthlyRent'])assert.equal(a.state.inputSources[s.id][path],'Unknown',path);
  await a.click('optional-answer',{question:'rent-inputs',answer:'unsure'});
  assert.equal(a.saved().optionalAnswers[s.id]['rent-inputs'],'unsure');
  const restored=app(a.saved());restored.state.guided=true;restored.state.setupSection=3;
  assert.match(restored.setup(),/data-question="rent-inputs" data-answer="unsure" aria-pressed="true"/);
  await a.click('optional-answer',{question:'home-inputs',answer:'no'});
  for(const path of ['home.currentValue','home.annualTaxesAndInsurance','mortgage.monthlyPayment','mortgage.currentBalance','mortgage.yearsLeft'])assert.equal(a.saved().inputSources[s.id][path],'Entered');
  assert.equal(a.saved().optionalAnswers[s.id]['mortgage-inputs'],'no');
  await a.click('optional-answer',{question:'your-pension',answer:'yes'});
  assert.equal(a.saved().inputSources[s.id]['guaranteedIncome.annualIncome'],'Unknown');
  await a.click('optional-answer',{question:'your-pension',answer:'no'});
  assert.equal(a.current().guaranteedIncome.annualIncome,0);
  await a.click('export-backup');const backup=JSON.parse(await a.downloads.at(-1).blob.text());
  const imported=app();await imported.change('#import-file',{files:[{text:async()=>JSON.stringify(backup)}],value:'backup.json'});
  assert.equal(imported.state.optionalAnswers[s.id]['your-pension'],'no');
  await a.click('new-scenario');assert.equal(a.saved().optionalAnswers[a.current().id]['rent-inputs'],'unsure');
});

test('one-click lower-return screening adds a paired what-if, retains results and never raises a lower rate',async()=>{
  const a=app(),s=a.current();s.market.preRetirementMeanReturn=.04;s.market.stockMeanReturn=.133;
  const before=structuredClone(s),completed=runSimulation(s);completed.uxAssumptions=before;a.state.results.set(s.id,completed);
  assert.match(a.results(),/Compare with lower returns/);
  const pending=a.click('compare-lower-returns');
  for(let i=0;i<2;i++){
    const w=await nextWorker(a,i);assert.ok(w);
    assert.equal(w.data.scenario.market.preRetirementMeanReturn,.04);
    assert.equal(w.data.scenario.market.stockMeanReturn,i?.07:before.market.stockMeanReturn);
    assert.equal(w.data.scenario.market.stockStdDev,before.market.stockStdDev);
    assert.equal(w.data.options.captureMetrics,true);
    w.answered=true;w.onmessage({data:{type:'result',result:runSimulation(w.data.scenario)}});
  }
  await pending;assert.equal(a.state.view,'lab');
  const rows=labRows(a);assert.equal(rows.filter(r=>r.result).length,2);assert.equal(rows[1].label,'Lower returns · up to 7%');
  assert.deepEqual(structuredClone(a.current()),before);assert.equal(a.state.results.get(s.id),completed);
  assert.deepEqual(a.saved().labSets[s.id].sets[0].whatIfs[0].changes,{returnCap:.07});
  await a.click('lab-copy',{id:rows[1].id});assert.equal(a.current().market.stockMeanReturn,.07);assert.equal(a.current().market.preRetirementMeanReturn,.04);
  assert.equal(a.saved().scenarios.find(x=>x.id===before.id).market.stockMeanReturn,before.market.stockMeanReturn);
});

test('preview warning keeps its essential caution visible and expands the full explanation',()=>{
  const a=app(),s=a.current(),r=runSimulation(s);a.state.results.set(s.id,r);
  const html=a.results();
  assert.match(html,/100 lifetimes are too few to estimate readiness/);
  assert.match(html,/<details><summary>Why only 100 lifetimes\?<\/summary>/);
  assert.match(html,/Zero observed outcomes does not mean an outcome is impossible/);
  assert.match(a.reportText(s,r),/SAMPLE PREVIEW ONLY: Only a small sample/);
});

test('building a personal forecast blanks example balances and income, preserves saved plans and accepts explicit None',async()=>{
  const a=app(),before=structuredClone(a.current());await a.click('start-plan');a.state.setupSection=1;
  assert.match(a.setup(),/id="f-accounts-pretax"[^>]*value=""/);
  assert.equal(a.state.inputSources[a.current().id]['accounts.pretax'],'Unknown');
  await a.run();assert.equal(a.workers.length,0);
  await a.click('input-none',{path:'accounts.roth'});
  assert.equal(a.current().accounts.roth,0);assert.equal(a.saved().inputSources[a.current().id]['accounts.roth'],'Entered');
  assert.equal(a.current().market.stockMeanReturn,before.market.stockMeanReturn);
  const saved=app({scenarios:[before],selectedId:before.id});await saved.click('start-plan');
  assert.deepEqual(structuredClone(saved.current()),before);
  assert.notEqual(saved.state.inputSources[before.id]['accounts.pretax'],'Unknown');
  const restored=app(a.saved());assert.equal(restored.state.inputSources[a.current().id]['accounts.pretax'],'Unknown');
});

test('review names remaining examples, hides retained Unknown numbers and provides exact edit routes',async()=>{
  const a=app();await a.click('start-plan');a.state.setupSection=5;
  const html=a.setup();
  assert.match(html,/Stock return average %: 13\.3%[^]*data-action="edit-input" data-index="4" data-path="market.stockMeanReturn"/);
  assert.match(html,/<dt>Savings today<\/dt><dd>Unknown · enter your amounts/);
  assert.match(html,/Check your monthly spending/);
  const details={tagName:'DETAILS',open:false,hidden:true,classList:{contains:name=>name==='optional-inputs'}};
  const input=a.element('#f-mortgage-monthlyPayment');input.parentElement=details;
  await a.click('edit-input',{index:'3',path:'mortgage.monthlyPayment'});
  assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,3);
  assert.equal(details.hidden,false);assert.equal(details.open,true,'Review links expose the requested optional answer');
});

test('applied budget deductions remain visible at the housing handoff until the amount is reviewed',async()=>{
  const s=model.baseScenario();s.budget.monthlyBudgets=[{month:'2026-09',creditCardBills:[{monthlyAmount:4600}],adjustments:{mortgage:1200}}];model.applyBudgetEstimate(s);
  const a=app({scenarios:[s],selectedId:s.id,inputSources:{[s.id]:{_origin:'Sample/default'}}});a.state.guided=true;a.state.setupSection=3;
  assert.match(a.setup(),/You excluded mortgage payments from your budget/);
  assert.match(a.budgetSummary(),/Deducting a payment from the budget does not enter it in the plan/);
  a.state.setupSection=5;assert.match(a.setup(),/Review mortgage payments/);
  await a.click('input-none',{path:'mortgage.monthlyPayment'});
  assert.doesNotMatch(a.setup(),/You excluded mortgage payments from your budget/);
  assert.equal(a.saved().inputSources[s.id]['mortgage.monthlyPayment'],'Entered');
});

test('the what-if builder stacks date and spending changes without changing the base plan',async()=>{
  const a=app(),s=a.current(),before=structuredClone(s),date=model.addCalendarMonths(s.household.retirementDate,12);a.state.view='lab';
  assert.match(a.lab(),/Plan Lab/);assert.match(a.lab(),/data-action="lab-add"/);assert.doesNotMatch(a.lab(),/value="NaN"/);
  await a.click('lab-add');const w=a.labViewModel().editing;assert.ok(w);assert.match(a.lab(),/id="lab-builder"/);
  await setLever(a,'retirementDate',{date});await setLever(a,'annualBaseSpending',{annual:'48k'});
  assert.deepEqual(structuredClone(w.changes),{retirementDate:date,annualBaseSpending:48000});
  assert.equal(w.name,`Retire ${model.dateLabel(date)} · age ${model.ageLabel(model.calendarMonthsBetween(s.household.birthday,date)/12)} + Spend $48,000 a year`);
  assert.deepEqual(structuredClone(a.current()),before);
  const pending=a.runLab(),seen=[];await settle(a,pending,worker=>{seen.push(worker.data.scenario);return runSimulation(worker.data.scenario);});
  assert.equal(seen.length,2);assert.equal(seen[0].spending.annualBaseSpending,before.spending.annualBaseSpending);
  assert.equal(seen[1].household.retirementDate,date);assert.equal(seen[1].spending.annualBaseSpending,48000);assert.equal(seen[1].household.spouseRetirementDate,before.household.spouseRetirementDate);
  assert.deepEqual(structuredClone(a.current()),before);assert.equal(labRows(a)[1].result.provenance.simulationCount,100);
  await a.click('lab-copy',{id:w.id});
  assert.equal(a.current().spending.annualBaseSpending,48000);assert.equal(a.current().household.retirementDate,date);
  assert.equal(a.current().numberOfSimulations,before.numberOfSimulations);assert.equal(a.saved().inputSources[a.current().id]['spending.annualBaseSpending'],'Entered');
  assert.equal(a.saved().scenarios.find(x=>x.id===before.id).spending.annualBaseSpending,before.spending.annualBaseSpending);
});

test('lever editors reject incomplete or excessive amounts and keep the previous change',async()=>{
  const a=app();await a.click('lab-add');const w=a.labViewModel().editing;
  await setLever(a,'annualBaseSpending',{annual:'60000'});assert.equal(w.changes.annualBaseSpending,60000);
  for(const value of ['', '-1', '1e400', String(model.MAX_DOLLAR_AMOUNT+2)]){
    await setLever(a,'annualBaseSpending',{annual:value});
    assert.equal(w.changes.annualBaseSpending,60000,value);assert.ok(a.state.labUi.leverError,value);assert.match(a.lab(),/class="field-error"/);
    assert.match(a.lab(),new RegExp(`id="lab-annualBaseSpending-annual"[^>]*value="${value}"`),'the invalid value stays visible for correction');
    await a.click('lab-lever-cancel');
  }
  await setLever(a,'oneTimeExpenses',{label0:'Roof',age0:'2.5',amount0:'1000'});assert.equal(w.changes.oneTimeExpenses,undefined);assert.match(a.state.labUi.leverError,/Expense 1/);
  await a.click('lab-lever-cancel');assert.equal(a.workers.length,0);
  const pending=a.runLab(),first=await nextWorker(a,0);
  await setLever(a,'annualBaseSpending',{annual:'50000'});
  await settle(a,pending);
  const row=labRows(a)[1];assert.equal(row.stale,true,'An edited what-if no longer matches its completed run');assert.equal(a.state.busy,false);
  assert.ok(first);assert.equal(a.labViewModel().summary,null);
});

test('guided savings tasks and review links reveal the right fields without changing assumptions',async()=>{
  const a=app();await a.click('start-plan');a.state.setupSection=1;
  const before=structuredClone(a.current());
  assert.match(a.setup(),/id="savings-task-0" tabindex="-1"  aria-label="Account balances"/);
  assert.match(a.setup(),/id="savings-task-1" tabindex="-1" hidden/);
  await a.click('savings-task',{task:'2'});
  assert.match(a.setup(),/id="savings-task-2" tabindex="-1"  aria-label="Future savings"/);
  assert.deepEqual(structuredClone(a.current()),before);
  await a.click('edit-input',{index:'1',path:'spending.annualBaseSpending'});
  assert.match(a.setup(),/id="savings-task-1" tabindex="-1"  aria-label="Everyday spending"/);
  assert.match(a.setup(),/id="setup-count-1"> · \d+ left/);
  a.state.setupSection=5;assert.match(a.setup(),/missing-tasks/);assert.match(a.setup(),/See missing answers/);
  assert.match(a.setup(),/Annual base spending<[^]*?data-action="edit-input" data-index="1" data-path="spending.annualBaseSpending"/);
  await a.click('setup-section',{index:'1',path:'contributions.pretax',reviewEdit:'true'});
  assert.match(a.setup(),/id="savings-task-2" tabindex="-1"  aria-label="Future savings"/);
});

test('unknown Roth records save a draft and preserve retained values and all investment defaults',async()=>{
  const a=app(),before=structuredClone(a.current());
  await a.click('roth-records-unknown',{prefix:'rothHistory'});
  assert.deepEqual(structuredClone(a.current()),before);
  assert.equal(a.saved().inputSources[before.id]['rothHistory.contributionBasis'],'Unknown');
  assert.equal(a.saved().inputSources[before.id]['rothHistory.firstContributionYear'],'Unknown');
  await a.run();assert.equal(a.workers.length,0);
  const restored=app(a.saved());restored.state.setupSection=1;
  assert.match(restored.setup(),/a \$50,000 Roth IRA could contain \$30,000/);
  assert.match(restored.setup(),/Add each plan under Employer Roth accounts/);
  assert.equal(restored.current().market.stockMeanReturn,before.market.stockMeanReturn);
  await a.click('input-none',{path:'accounts.roth'});
  assert.equal(a.saved().inputSources[before.id]['rothHistory.firstContributionYear'],'Entered');
  await a.click('roth-records-unknown',{prefix:'rothHistory'});
  assert.equal(a.saved().inputSources[before.id]['rothHistory.firstContributionYear'],'Entered','A zero account never needs history');
});

test('Pro sets compare up to four stacked what-ifs, and Make this my plan updates the saved plan',async()=>{
  const a=app();a.state.access.tier='pro';const s=a.current(),before=structuredClone(s),date=model.addCalendarMonths(before.household.retirementDate,12);
  const set=a.labViewModel().set;assert.deepEqual(set.whatIfs.map(w=>w.name),['Retire 2 years later','Spend 5% less','Claim Social Security at 70']);
  await a.click('lab-add');const w=a.labViewModel().editing;
  await setLever(a,'retirementDate',{date});await setLever(a,'annualBaseSpending',{annual:'48000'});await setLever(a,'rothConversion',{cap:'0.24'});
  assert.match(a.lab(),/4 of 4 what-ifs/);await a.click('lab-add');assert.match(a.state.message,/up to 4 what-ifs/);
  const pending=a.runLab(),seen=[];await settle(a,pending,worker=>{if(worker.data.options?.captureMetrics)seen.push(worker.data.scenario);return runSimulation({...worker.data.scenario,numberOfSimulations:20});});
  assert.equal(seen.length,5);assert.ok(seen.every(x=>x.numberOfSimulations===100));
  const combined=seen[4];assert.equal(combined.household.retirementDate,date);assert.equal(combined.spending.annualBaseSpending,48000);assert.deepEqual(combined.rothConversion,{enabled:true,marginalRateCap:.24});
  assert.equal(combined.household.spouseRetirementDate,before.household.spouseRetirementDate);assert.deepEqual(structuredClone(a.current()),before);
  const html=a.lab();assert.match(html,/In short/);assert.match(html,/Every number, side by side/);assert.match(html,/Lifetime federal income tax/);
  await a.click('lab-apply',{id:w.id});
  assert.equal(a.current().id,before.id);assert.equal(a.current().spending.annualBaseSpending,48000);assert.equal(a.current().household.retirementDate,date);assert.equal(a.current().rothConversion.marginalRateCap,.24);
  assert.equal(a.saved().inputSources[before.id]['spending.annualBaseSpending'],'Entered');assert.equal(a.state.results.has(before.id),false);
  assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,5);assert.equal(a.state.labResults,null);
  assert.equal(a.saved().labSets[before.id].sets[0].whatIfs.length,4,'The what-if stays in the set');
});

test('recurring budget deductions copy only housing and health amounts and require fresh review',async()=>{
  const s=model.baseScenario();s.budget.monthlyBudgets=[{month:'2026-08',creditCardBills:[{monthlyAmount:5000}],adjustments:{rent:1200,healthcare:300,propertyTaxes:100}},{month:'2026-09',creditCardBills:[{monthlyAmount:5000}],adjustments:{propertyTaxes:200}}];
  model.applyBudgetEstimate(s);const applied=s.spending.annualBaseSpending;
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='budget';
  await a.click('copy-month-costs',{index:'1'});
  const m=a.current().budget.monthlyBudgets[1];
  assert.equal(m.adjustments.rent,1200);assert.equal(m.adjustments.healthcare,300);assert.equal(m.adjustments.propertyTaxes,200);assert.equal(m.creditCardBills[0].monthlyAmount,5000);
  assert.equal(a.current().spending.annualBaseSpending,applied);assert.equal(a.budgetCostsReviewed(),false);
  await a.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(a.budgetCostsReviewed(),true);
  await a.change('#main',{dataset:{monthCost:'mortgage',index:'1'},checked:true});
  await a.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(a.budgetCostsReviewed(),false,'An included payment needs its amount');
  await a.change('#main',{dataset:{monthCost:'mortgage',index:'1'},checked:false});
  await a.change('#main',{dataset:{monthCost:'rent',index:'1'},checked:false});assert.equal(m.adjustments.rent,0);assert.equal(a.budgetCostsReviewed(),false);
  assert.equal(a.saved().scenarios[0].budget.monthlyBudgets[1].adjustments.rent,0);
  assert.match(a.budget(),/Which payments are already in this month’s totals/);
});

// Execute the actual app and event handlers. Only browser IO is replaced;
// workers stay pending so changes during calculations can be reproduced.
function webLocks(){
  let queue=Promise.resolve();
  return {request(_name,callback){const next=queue.then(callback);queue=next.catch(()=>{});return next;}};
}
function app(saved=null,{fetch=async()=>{throw new Error('offline');},storage={fail:false},clock={now:Date.now()},session=new Map(),location={search:'',pathname:'/',hash:''},identity={accountKey:null},preferences=new Map(),params=URLSearchParams,confirm=()=>true,rawStorage,locks=webLocks(),socialActions={},resultRecords=new Map(),resultStorage={fail:false}}={}){
  const elements=new Map(),workers=[],timers=new Map(),downloadBlobs=new Map(),downloads=[];let nextTimer=0;
  let stored=rawStorage===undefined?(saved===null?null:JSON.stringify(saved)):rawStorage;
  const readStored=()=>Object.hasOwn(storage,'raw')?storage.raw:stored;
  const windowListeners={};
  function element(selector){
    if(['#advanced-model','#allocation-settings'].includes(selector))return null;
    if(!elements.has(selector))elements.set(selector,{innerHTML:'',textContent:'',dataset:{},listeners:{},classList:{toggle(){},remove(){}},addEventListener(name,fn){this.listeners[name]=fn;},querySelectorAll(){return [];},querySelector(child){return element(selector+' '+child);},setAttribute(){},focus(){},scrollIntoView(){},insertAdjacentHTML(){},remove(){elements.delete(selector);}});
    return elements.get(selector);
  }
  const document={activeElement:null,querySelector:element,querySelectorAll:()=>[],addEventListener(){},createElement(){return {click(){downloads.push({name:this.download,blob:downloadBlobs.get(this.href)});}};}};
  const context=vm.createContext({...model,...format,...guidance,...withdrawalsView,...growthHelper,...moneyInput,...planReview,...resultCaching,...frontierView,...savingsTargets,...planLab,...planLabView,createResultCache:()=>({
      async load(id){const record=resultStorage.fail?null:structuredClone(resultRecords.get(id)||null);await resultStorage.beforeLoad?.(id);return record;},
      async save(s,r,today){if(resultStorage.fail)return null;await resultStorage.beforeSave?.(s,r);resultRecords.set(s.id,{id:s.id,version:1,fingerprint:resultCaching.resultFingerprint(s,today),result:structuredClone(r)});return s.id;},
      async remove(id){resultRecords.delete(id);},async clear(){resultRecords.clear();}
    }),structuredClone,Intl,URLSearchParams:params,Blob,URL:class extends URL{static createObjectURL(blob){const url='blob:test-'+downloadBlobs.size;downloadBlobs.set(url,blob);return url;}static revokeObjectURL(url){downloadBlobs.delete(url);}},console,Date:class extends Date{static now(){return clock.now;}},
    location,history:{replaceState(_state,_title,url){const next=new URL(url,'https://example.test');location.search=next.search;location.hash=next.hash;}},confirm,
    setTimeout(fn,delay){const id=++nextTimer;timers.set(id,{fn,at:clock.now+delay});return id;},clearTimeout(id){timers.delete(id);},
    sessionStorage:{getItem:key=>session.get(key)||null,setItem:(key,value)=>session.set(key,value)},socialState:()=>({...identity}),
    navigator:{locks},localStorage:{getItem:key=>key==='retirement-readiness-lab-sites-v1'?readStored():preferences.get(key)||null,setItem(key,value){if(key!=='retirement-readiness-lab-sites-v1'){preferences.set(key,value);return;}if(storage.fail)throw new Error('QuotaExceededError');if(Object.hasOwn(storage,'raw'))storage.raw=value;else stored=value;},removeItem:key=>preferences.delete(key)},window:{addEventListener(name,handler){windowListeners[name]=handler;},scrollTo(){}},
    document,
    chartCard:()=>'',mountCharts(){},disposeCharts(){},initializeSocialAuth:()=>new Promise(()=>{}),fetch,authHeaders:async()=>({}),...socialActions,
    Worker:class{constructor(){workers.push(this);}postMessage(data){this.data=data;}terminate(){this.terminated=true;}},
  });
  const source=readFileSync(new URL('../dist/app.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replaceAll('import.meta.url',JSON.stringify(new URL('../dist/app.js',import.meta.url).href));
  vm.runInContext(source.replace('function render({preserveEditor=false}={}){','let renderCount=0;function render({preserveEditor=false}={}){renderCount++;'),context);
  const api=vm.runInContext('({state,resultRestoreReady,setup,run,runLab,runLabStress,runLabSensitivity,labViewModel,runDecision,results,withdrawals,dashboard,lab,budget,budgetView,budgetSummary,budgetCostCheck,budgetCostsReviewed,enterBudget,scenarios,render,billingView,accountCheck,reports,reportText,reportSummaryText,reportDetailsText,current,persist,loadAccess,isPro,effectivePaths,syncAuthState,linkAccounts,renders:()=>renderCount})',context);
  return {...api,workers,element,timers,document,downloads,stored:readStored,storageChanged:()=>windowListeners.storage({key:'retirement-readiness-lab-sites-v1'}),beforeUnload:event=>windowListeners.beforeunload(event),advanceTime(ms){clock.now+=ms;for(const [id,timer] of [...timers])if(timer.at<=clock.now){timers.delete(id);timer.fn();}},saved:()=>JSON.parse(readStored()),change:(selector,target)=>element(selector).listeners.change({target}),
    click:(action,extra={})=>{const el={dataset:{action,...extra}};return element('#main').listeners.click({target:{closest:selector=>selector==='[data-action]'?el:null}});}};
}
function seedExploration(a){a.state.labResults=[{label:'Old plan',result:null}];a.state.decision={targetReadiness:.8,simulationCount:200};}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
// Answers workers in creation order until the calculation settles. Plan Lab
// starts each comparison only after the previous one finishes.
async function settle(a,pending,respond=w=>runSimulation(w.data.scenario)){
  let done=false;pending.then(()=>{done=true;},()=>{done=true;});
  for(let i=0;i<400&&!done;i++){
    await tick();
    for(const w of a.workers)if(!w.answered&&!w.terminated&&w.data&&w.onmessage){w.answered=true;w.onmessage({data:{type:'result',result:respond(w)}});}
  }
  return pending;
}
async function nextWorker(a,index){for(let i=0;i<50&&!a.workers[index]?.data;i++)await tick();return a.workers[index];}
const labRows=a=>a.labViewModel().rows;
const leverInput=(a,lever,field,value)=>a.element('#main').listeners.input({target:{dataset:{labLever:lever,labField:field},value}});
async function setLever(a,lever,values){await a.click('lab-lever-edit',{lever});for(const [field,value] of Object.entries(values))leverInput(a,lever,field,value);await a.click('lab-lever-save',{lever});}
function assertCleared(a){assert.equal(a.state.labResults,null);assert.equal(a.state.decision,null);assert.equal(a.state.claimDecision,null);assert.equal(a.state.savingsDecision,null);assert.equal(a.state.allocationDecision,null);}

test('applying an age target saves its birthday date and preserves spouse timing and full-run paths',async()=>{
  const records=new Map(),a=app(null,{resultRecords:records});a.state.access.tier='pro';
  const s=a.current();s.numberOfSimulations=10000;s.household.separatePeople=true;s.household.filingStatus='Married';
  s.household.spouseRetirementDate=model.addCalendarMonths(s.household.birthday,66*12);
  s.household.alreadyRetired=true;s.household.retirementDate='';
  const before=structuredClone(s),id=s.id;
  a.state.results.set(id,{old:true});records.set(id,{old:true});seedExploration(a);
  a.state.decision={...a.state.decision,earliestRetirementAge:64,safeAnnualSpending:90000};
  assert.match(a.lab(),/Use this retirement age/);assert.match(a.lab(),/Use this spending amount/);
  assert.match(a.lab(),/tested separately with 200 paths/);
  await a.click('apply-age-target');
  assert.equal(s.household.retirementDate,model.addCalendarMonths(s.household.birthday,64*12));
  assert.equal(s.household.retirementAge,64);assert.equal(s.household.retirementAgeMonths,0);
  assert.equal(s.household.alreadyRetired,false);assert.equal(s.household.asOfDate,'');
  assert.equal(s.household.spouseRetirementDate,before.household.spouseRetirementDate);
  assert.equal(s.spending.annualBaseSpending,before.spending.annualBaseSpending);assert.equal(s.numberOfSimulations,10000);
  assert.equal(s.seed,before.seed);assert.equal(s.id,id);assert.equal(a.state.exampleIds.has(id),false);
  assert.equal(a.state.inputSources[id]['household.retirementDate'],'Estimated');
  assertCleared(a);assert.equal(a.state.results.has(id),false);assert.equal(records.has(id),false);
  assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,5);assert.match(a.state.message,/Run the full forecast/);
  const restored=app(a.saved());assert.equal(restored.current().household.retirementDate,s.household.retirementDate);
  assert.equal(restored.current().numberOfSimulations,10000);
});

test('applying a spending target keeps housing costs and dates, replaces an applied budget and saves Estimated',async()=>{
  const a=app();a.state.access.tier='pro';const s=a.current();s.numberOfSimulations=500;
  s.home.annualTaxesAndInsurance=30000;s.budget.isAppliedToAnnualBaseSpending=true;s.budget.estimateNeedsReview=false;
  const before=structuredClone(s);a.state.results.set(s.id,{old:true});seedExploration(a);
  a.state.decision={...a.state.decision,earliestRetirementAge:64,safeAnnualSpending:120000};
  await a.click('apply-spending-target');
  assert.equal(s.spending.annualBaseSpending,120000);assert.equal(s.home.annualTaxesAndInsurance,30000);
  assert.deepEqual(structuredClone(s.household),before.household);assert.equal(s.numberOfSimulations,500);
  assert.equal(s.budget.isAppliedToAnnualBaseSpending,false);assert.equal(s.budget.estimateNeedsReview,true);
  assert.deepEqual(structuredClone(s.accounts),before.accounts);assert.equal(s.seed,before.seed);
  assert.equal(a.saved().inputSources[s.id]['spending.annualBaseSpending'],'Estimated');
  assert.equal(a.saved().scenarios[0].spending.annualBaseSpending,120000);assertCleared(a);
  assert.equal(a.state.results.has(s.id),false);assert.equal(a.state.view,'setup');
});

test('target apply buttons require a found result and ignore clicks while busy or after invalidation',async()=>{
  const a=app();a.state.access.tier='pro';const before=structuredClone(a.current());
  a.state.decision={targetReadiness:.8,simulationCount:200,earliestRetirementAge:null,safeAnnualSpending:null};
  assert.doesNotMatch(a.lab(),/data-action="apply-(?:age|spending)-target"/);
  await a.click('apply-age-target');await a.click('apply-spending-target');
  a.state.decision={...a.state.decision,earliestRetirementAge:64,safeAnnualSpending:0};a.state.busy=true;
  assert.match(a.lab(),/data-action="apply-age-target" disabled/);
  await a.click('apply-age-target');await a.click('apply-spending-target');
  a.state.busy=false;a.state.decision=null;await a.click('apply-age-target');await a.click('apply-spending-target');
  assert.deepEqual(structuredClone(a.current()),before);
});

test('the frontier slider previews a tested pair and applying it saves both fields without changing full-run settings',async()=>{
  const records=new Map(),a=app(null,{resultRecords:records});a.state.access.tier='pro';
  const s=a.current();s.numberOfSimulations=500;s.simulationPathsCustomized=true;s.household.separatePeople=true;s.household.filingStatus='Married';
  s.household.spouseRetirementDate=model.addCalendarMonths(s.household.spouseBirthday,66*12);
  s.budget.isAppliedToAnnualBaseSpending=true;s.budget.estimateNeedsReview=false;
  const before=structuredClone(s),stored=a.stored();a.state.results.set(s.id,{old:true});records.set(s.id,{old:true});seedExploration(a);
  a.state.decision.frontier={targetReadiness:.8,simulationCount:200,searchStartAge:62,searchEndAge:64,spendingSearchLimit:250000,points:[
    {age:62,annualSpending:85000,readiness:.8,tested:true,simulationCount:200},{age:63,annualSpending:null,readiness:null},{age:64,annualSpending:120000,readiness:.803,tested:true,simulationCount:200}
  ]};
  a.element('#main').listeners.input({target:{dataset:{frontierSlider:''},value:'1'}});
  assert.equal(a.state.frontierSelection,1);assert.match(a.element('#frontier-selection').innerHTML,/120,000/);
  assert.match(a.element('#frontier-selection').innerHTML,/>64</);assert.deepEqual(structuredClone(s),before);assert.equal(a.stored(),stored);
  await a.click('apply-frontier-target');
  assert.equal(s.household.retirementDate,model.addCalendarMonths(s.household.birthday,64*12));assert.equal(s.spending.annualBaseSpending,120000);
  assert.equal(s.household.spouseRetirementDate,before.household.spouseRetirementDate);assert.equal(s.numberOfSimulations,500);
  assert.equal(s.simulationPathsCustomized,true);assert.equal(s.seed,before.seed);assert.deepEqual(structuredClone(s.accounts),before.accounts);
  assert.equal(s.budget.isAppliedToAnnualBaseSpending,false);assert.equal(s.budget.estimateNeedsReview,true);
  assert.equal(a.saved().inputSources[s.id]['household.retirementDate'],'Estimated');assert.equal(a.saved().inputSources[s.id]['spending.annualBaseSpending'],'Estimated');
  assertCleared(a);assert.equal(a.state.results.has(s.id),false);assert.equal(records.has(s.id),false);assert.equal(a.state.frontierSelection,0);
  assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,5);assert.match(a.state.message,/Retirement age and base spending target applied/);
  const restored=app(a.saved());assert.equal(restored.current().household.retirementAge,64);assert.equal(restored.current().spending.annualBaseSpending,120000);
  assert.equal(restored.current().numberOfSimulations,500);
});

test('frontier selection and applying ignore incomplete, failing, busy and invalidated results',async()=>{
  const a=app(),before=structuredClone(a.current());a.state.access.tier='pro';
  a.state.decision={targetReadiness:.8,simulationCount:200,earliestRetirementAge:null,safeAnnualSpending:null,frontier:{targetReadiness:.8,simulationCount:200,searchStartAge:64,searchEndAge:66,points:[{age:64,annualSpending:null,readiness:null},{age:65,annualSpending:100000,readiness:.79}]}};
  assert.doesNotMatch(a.lab(),/data-action="apply-frontier-target"/);await a.click('apply-frontier-target');
  a.state.decision.frontier.points.push({age:66,annualSpending:105000,readiness:.8,tested:true,simulationCount:200});a.state.busy=true;
  assert.match(a.lab(),/data-action="apply-frontier-target" disabled/);await a.click('apply-frontier-target');
  a.element('#main').listeners.input({target:{dataset:{frontierSlider:''},value:'0'}});assert.equal(a.state.frontierSelection,undefined);
  a.state.busy=false;a.state.decision=null;await a.click('apply-frontier-target');assert.deepEqual(structuredClone(a.current()),before);
});

test('new plans model early penalties while saved and imported choices retain their previous behavior',async()=>{
  const fresh=app();fresh.state.setupSection=4;
  assert.equal(fresh.current().withdrawalStrategy.applyEarlyWithdrawalPenalty,true);
  assert.equal(fresh.current().withdrawalStrategy.ruleOf55Eligible,false);
  assert.equal(fresh.current().withdrawalStrategy.seppEligible,false);
  assert.match(fresh.setup(),/id="f-withdrawalStrategy-applyEarlyWithdrawalPenalty"[^>]* checked/);
  assert.match(fresh.setup(),/Accessing retirement savings before 59½/);
  assert.match(fresh.setup(),/Use a 72\(t\)\/SEPP withdrawal plan/);
  for(const setting of [true,false,undefined]){
    const s=model.baseScenario();Object.assign(s.withdrawalStrategy,{applyEarlyWithdrawalPenalty:setting,ruleOf55Eligible:true,seppEligible:true});
    if(setting===undefined)delete s.withdrawalStrategy.applyEarlyWithdrawalPenalty;
    const restored=app({scenarios:[s],selectedId:s.id}),expected=setting===true;
    assert.equal(restored.current().withdrawalStrategy.applyEarlyWithdrawalPenalty,expected);
    assert.equal(restored.current().withdrawalStrategy.ruleOf55Eligible,true);
    assert.equal(restored.current().withdrawalStrategy.seppEligible,true);
    await restored.persist();assert.equal(app(restored.saved()).current().withdrawalStrategy.applyEarlyWithdrawalPenalty,expected);
    await fresh.change('#import-file',{files:[{text:async()=>JSON.stringify({scenarios:[s]})}],value:'plan.json'});
    assert.equal(fresh.current().withdrawalStrategy.applyEarlyWithdrawalPenalty,expected);
  }
  const legacy=model.baseScenario();delete legacy.withdrawalStrategy;
  assert.equal(model.normalizeScenario(legacy).withdrawalStrategy.applyEarlyWithdrawalPenalty,false);
  assert.equal(model.normalizeScenario({currentAge:50,retirementAge:55}).withdrawalStrategy.applyEarlyWithdrawalPenalty,false);
});

test('Rule of 55 prompts use the separation calendar year without declaring employer-plan eligibility',()=>{
  const year=Number(model.localCalendarDate().slice(0,4))+2;
  const s=model.baseScenario();Object.assign(s.household,{birthday:`${year-55}-12-31`,retirementDate:`${year}-01-01`});
  const a=app({scenarios:[s],selectedId:s.id});a.state.setupSection=4;
  assert.equal(model.retirementAge(a.current())<55,true,'Separation can precede the 55th birthday in that year');
  assert.match(a.setup(),/retirement date meets the Rule of 55 age requirement/);
  assert.match(a.setup(),/IRAs do not qualify/);assert.match(a.setup(),/SEPP withdrawal plan is optional/);
  assert.equal(a.current().withdrawalStrategy.ruleOf55Eligible,false);
  assert.equal(a.current().withdrawalStrategy.seppEligible,false);
  a.current().household.retirementDate=`${year-1}-12-31`;
  assert.doesNotMatch(a.setup(),/retirement date meets the Rule of 55 age requirement/);
  a.current().withdrawalStrategy.ruleOf55Eligible=true;
  assert.match(a.setup(),/Rule of 55 does not apply with this retirement date/);
  assert.equal(a.current().withdrawalStrategy.ruleOf55Eligible,true,'Changing dates must not overwrite a declaration');
});

test('date and withdrawal-choice edits refresh guidance without replacing inputs or overwriting saved choices',async()=>{
  const s=model.baseScenario();s.withdrawalStrategy.applyEarlyWithdrawalPenalty=false;
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='setup';a.state.setupSection=0;
  const today=model.localCalendarDate(),before=a.renders();
  await a.change('#main',{dataset:{field:'household.birthday',type:'date'},value:model.addCalendarMonths(today,-56*12)});
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:today});
  assert.equal(a.current().withdrawalStrategy.applyEarlyWithdrawalPenalty,false);
  assert.match(a.element('#early-withdrawal-review').innerHTML,/Review withdrawal choices/);
  a.state.setupSection=4;
  assert.match(a.setup(),/Penalty modeling is off in this plan/);
  assert.equal(a.renders(),before,'Guidance updates must preserve unfinished editors');
  a.state.results.set(a.current().id,{completed:true});seedExploration(a);
  await a.change('#main',{dataset:{field:'withdrawalStrategy.applyEarlyWithdrawalPenalty',type:'checkbox'},checked:true});
  assert.doesNotMatch(a.element('#early-withdrawal-guidance').innerHTML,/Penalty modeling is off/);
  assert.equal(a.state.results.size,0);assertCleared(a);
  for(const field of ['ruleOf55Eligible','seppEligible'])await a.change('#main',{dataset:{field:'withdrawalStrategy.'+field,type:'checkbox'},checked:true});
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:model.addCalendarMonths(today,60)});
  assert.match(a.element('#early-withdrawal-guidance').innerHTML,/will not start SEPP payments/);
  assert.deepEqual(JSON.parse(JSON.stringify(a.current().withdrawalStrategy)),a.saved().scenarios[0].withdrawalStrategy);
  assert.equal(a.current().withdrawalStrategy.ruleOf55Eligible,true);assert.equal(a.current().withdrawalStrategy.seppEligible,true);
});

test('younger spouse prompts and copied plans preserve explicit settings; restoring samples uses the new default',async()=>{
  const s=model.baseScenario();Object.assign(s.household,{filingStatus:'Married',spouseCurrentAge:50});
  s.withdrawalStrategy.applyEarlyWithdrawalPenalty=false;
  const a=app({scenarios:[s],selectedId:s.id});a.state.setupSection=4;
  assert.match(a.setup(),/Your spouse will be younger than 59½/);
  assert.doesNotMatch(a.setup(),/retirement date meets the Rule of 55 age requirement/);
  await a.click('new-scenario');assert.equal(a.current().withdrawalStrategy.applyEarlyWithdrawalPenalty,false);
  await a.click('reset-assumptions');assert.equal(a.current().withdrawalStrategy.applyEarlyWithdrawalPenalty,true);
  assert.equal(a.current().withdrawalStrategy.ruleOf55Eligible,false);assert.equal(a.current().withdrawalStrategy.seppEligible,false);
});

test('the first Budget month is visible without saving or treating an untouched entry as zero spending',async()=>{
  const a=app(),s=a.current();a.state.view='budget';const html=a.budget();
  const lastMonth=new Date();lastMonth.setDate(1);lastMonth.setMonth(lastMonth.getMonth()-1);
  const expected=`${lastMonth.getFullYear()}-${String(lastMonth.getMonth()+1).padStart(2,'0')}`;
  assert.match(html,/id="budget-month-0"[^>]* open/);
  assert.match(html,/id="month-0-credit"[^>]*value=""/);
  assert.equal(a.budgetView().pending.month,expected);
  assert.match(html,/data-action="apply-budget" disabled/);
  assert.doesNotMatch(html,/Add at least one complete month/,'An untouched worksheet needs guidance, not an error');
  assert.equal(s.budget.monthlyBudgets.length,0);assert.equal(s.spending.annualBaseSpending,75000);assert.equal(a.stored(),null);
  await a.change('#main',{dataset:{month:'0',part:'month'},value:'2026-01'});
  assert.equal(a.budgetView().pending.month,'2026-01');assert.equal(s.budget.monthlyBudgets.length,0);assert.equal(a.stored(),null);
  await a.change('#main',{dataset:{month:'0',part:'credit'},value:'',validity:{badInput:true}});
  assert.equal(s.budget.monthlyBudgets.length,0);assert.equal(a.stored(),null);
  await a.change('#main',{dataset:{month:'0',part:'credit'},value:'0'});
  assert.equal(s.budget.monthlyBudgets.length,1);assert.equal(s.budget.monthlyBudgets[0].month,'2026-01');
  assert.equal(a.saved().scenarios[0].budget.monthlyBudgets.length,1,'An explicitly entered zero is a real spending entry');
  assert.equal(s.spending.annualBaseSpending,75000);assert.equal(a.budgetView().pending,null);
});

test('adding an unentered month cannot reduce an applied estimate; committing it creates a draft',async()=>{
  const s=model.baseScenario();s.budget.monthlyBudgets=[{month:'2026-08',creditCardBills:[{monthlyAmount:4000}],includedPayments:{mortgage:false,rent:false,healthcare:false}}];model.applyBudgetEstimate(s);
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='budget';a.budget();await confirmBudgetPaymentChoices(a);const saved=a.stored();
  assert.match(a.budgetSummary(),/>Applied</);
  await a.click('add-month');assert.equal(a.stored(),saved);
  assert.equal(a.current().budget.monthlyBudgets.length,1);assert.equal(model.budgetEstimate(a.current().budget),48000);
  assert.notEqual(a.budgetView().pending.month,'2026-08');assert.match(a.budgetSummary(),/>Applied</);
  await a.click('add-month');assert.equal(a.current().budget.monthlyBudgets.length,1,'Only one unentered month is offered at a time');
  await a.click('remove-month',{index:'1'});assert.equal(a.budgetView().pending,null);assert.equal(a.stored(),saved);
  await a.click('add-month');
  await a.change('#main',{dataset:{month:'1',part:'checking'},value:'5000'});
  assert.equal(a.current().budget.monthlyBudgets.length,2);assert.equal(a.budgetView().pending,null);
  assert.equal(a.current().spending.annualBaseSpending,48000);assert.equal(model.budgetEstimate(a.current().budget),54000);
  assert.match(a.budgetSummary(),/>Draft</);assert.match(a.budgetSummary(),/plan still uses \$48,000.00/);
  await confirmBudgetPaymentChoices(a);await a.click('apply-budget');assert.equal(a.current().spending.annualBaseSpending,54000);assert.match(a.budgetSummary(),/>Applied</);
});

test('completed months collapse, optional sections summarize their values, and the final monthly amount includes all adjustments',async()=>{
  const s=model.baseScenario();s.budget.monthlyBudgets=Array.from({length:12},(_,i)=>({month:`2026-${String(i+1).padStart(2,'0')}`,creditCardBills:[{monthlyAmount:4000}]}));
  s.budget.annualPropertyTaxes=6000;s.budget.retirementAnnualAdjustment=-2400;
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='budget';const html=a.budget();
  const rows=[...html.matchAll(/<details class="spending-month"[^>]*>/g)].map(m=>m[0]);
  assert.equal(rows.length,12);assert.ok(rows.every(row=>!row.includes(' open')));
  assert.match(html,/January 2026/);assert.match(html,/data-budget-disclosure="annual-bills"><summary/);
  assert.match(html,/id="budget-annual-bills-summary">\$6,000.00 \/ year/);
  assert.match(html,/id="budget-retirement-summary">\$200.00 \/ month less/);
  assert.match(html,/Monthly equivalent<\/span><strong>\$4,300.00/);
  assert.match(html,/Annual spending<\/span><strong>\$51,600.00/);
  assert.match(html,/data-budget-disclosure="calculation"><summary>How this was calculated/);
  assert.match(html,/Deduct included mortgage, rent and health premiums/);assert.match(html,/Use this spending in my plan/);
  const original=JSON.stringify(a.current()),stored=a.stored();a.element('#budget-month-0').open=true;
  await a.click('finish-month',{index:'0'});assert.equal(a.element('#budget-month-0').open,false);
  assert.equal(JSON.stringify(a.current()),original);assert.equal(a.stored(),stored);
});

test('monthly retirement controls preserve legacy annual amounts exactly when changing direction',async()=>{
  for(const original of [-1000,1000]){
    const s=model.baseScenario();s.budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[{monthlyAmount:4000}]}];s.budget.retirementAnnualAdjustment=original;model.applyBudgetEstimate(s);
    const a=app({scenarios:[s],selectedId:s.id});a.state.view='budget';const applied=a.current().spending.annualBaseSpending;
    assert.match(a.budget(),/id="budget-retirement-monthly-adjustment"[^>]*value="83.33"/);
    assert.equal(a.current().budget.retirementAnnualAdjustment,original,'Displaying a rounded monthly value must not change a backup');
    await a.change('#main',{dataset:{budgetAdjustment:'direction'},value:original<0?'more':'less'});
    assert.equal(a.current().budget.retirementAnnualAdjustment,-original,'A sign change uses the precise annual value');
    assert.equal(a.current().spending.annualBaseSpending,applied);assert.match(a.budgetSummary(),/>Draft</);
    await a.change('#main',{dataset:{budgetAdjustment:'amount'},value:'125.50'});
    assert.equal(a.current().budget.retirementAnnualAdjustment,original<0?1506:-1506);
    const copy=app(a.saved());assert.equal(copy.current().budget.retirementAnnualAdjustment,original<0?1506:-1506);
    await a.click('export-backup');const backup=JSON.parse(await a.downloads[0].blob.text());
    assert.equal(backup.scenarios[0].budget.retirementAnnualAdjustment,original<0?1506:-1506);
    assert.equal(Object.hasOwn(backup.scenarios[0].budget,'pending'),false);
  }
  const a=app();a.state.view='budget';a.budget();await a.change('#main',{dataset:{budgetAdjustment:'direction'},value:'more'});
  await a.change('#main',{dataset:{budgetAdjustment:'amount'},value:'250'});assert.equal(a.current().budget.retirementAnnualAdjustment,3000);
});

test('invalid monthly retirement edits retain applied spending and reject overflow after annualizing',async()=>{
  const s=model.baseScenario();s.budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[{monthlyAmount:4000}]}];s.budget.retirementAnnualAdjustment=-6000;model.applyBudgetEstimate(s);
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='budget';a.budget();const before=JSON.stringify(a.current()),stored=a.stored();
  for(const invalid of [{value:'',validity:{badInput:true}},{value:'1e'},{value:'-50'},{value:'1e15'},{value:'1e308'}]){
    await a.change('#main',{dataset:{budgetAdjustment:'amount'},...invalid});
    assert.equal(JSON.stringify(a.current()),before);assert.equal(a.stored(),stored);assert.match(a.state.message,/^Error/);
  }
  await a.change('#main',{dataset:{budgetAdjustment:'amount'},value:'0'});
  assert.equal(a.current().budget.retirementAnnualAdjustment,0);assert.equal(a.current().spending.annualBaseSpending,42000);
});

test('Budget disclosures and unfinished monthly adjustment editors survive a background render',async()=>{
  for(const tagName of ['INPUT','SELECT']){
    const a=app(null,{fetch:async()=>Response.json({tier:'free',signedIn:false})});a.state.view='budget';a.budget();
    const layout=a.element('.budget-layout');layout.dataset.scenarioId=a.current().id;
    layout.querySelectorAll=()=>['help','annual-bills','retirement','month-0'].map(key=>({dataset:{budgetDisclosure:key},open:true}));
    const editor={tagName,type:tagName==='INPUT'?'number':'select-one',id:'live-adjustment',dataset:{budgetAdjustment:tagName==='INPUT'?'amount':'direction'},value:tagName==='INPUT'?'123.45':'more',focus(){a.document.activeElement=this;}};
    const replacement=a.element('#live-adjustment');replacement.type=editor.type;let retained;replacement.replaceWith=node=>{retained=node;};a.document.activeElement=editor;
    await a.loadAccess();assert.equal(retained,editor);assert.equal(a.document.activeElement,editor);
    assert.equal(editor.value,tagName==='INPUT'?'123.45':'more');
    const html=a.budget();for(const key of ['help','annual-bills','retirement','month-0'])assert.match(html,new RegExp('data-budget-disclosure="'+key+'" open'));
  }
});

test('Roth setup shows separate total and contribution inputs with useful five-year explanations',()=>{
  const a=app();a.state.setupSection=1;const html=a.setup();
  assert.match(html,/Total Roth IRA value/);assert.match(html,/Remaining regular contributions/);
  assert.match(html,/data-field="rothHistory.contributionBasis"/);
  assert.match(html,/First Roth funding tax year/);assert.match(html,/five.tax.year clock/);
  assert.match(html,/can exceed the current balance/);assert.doesNotMatch(html,/treats withdrawals from this balance as tax-free/);
});

test('Roth history edits save, invalidate results, survive reload, and appear in reports and backups',async()=>{
  const a=app(),s=a.current();a.state.results.set(s.id,{completed:true});seedExploration(a);
  await a.change('#main',{dataset:{field:'rothHistory.contributionBasis',type:'money'},value:'75000'});
  assert.equal(s.rothHistory.contributionBasis,75000);assert.equal(s.accounts.roth,17500);
  assert.equal(a.state.results.has(s.id),false);assertCleared(a);
  await a.change('#main',{dataset:{field:'rothHistory.firstContributionYear',type:'number'},value:'2024'});
  await a.click('add-roth-conversion');
  await a.change('#main',{dataset:{field:'rothHistory.conversions.0.taxYear',type:'number'},value:'2025'});
  await a.change('#main',{dataset:{field:'rothHistory.conversions.0.amount',type:'money'},value:'12000'});
  assert.equal(s.rothHistory.conversions[0].taxableAmount,12000);
  await a.change('#main',{dataset:{field:'rothHistory.conversions.0.taxableAmount',type:'money'},value:'8000'});
  await a.change('#main',{dataset:{field:'rothHistory.conversions.0.amount',type:'money'},value:'11000'});
  assert.equal(s.rothHistory.conversions[0].taxableAmount,8000,'An explicit partially taxable conversion is preserved');
  const restored=app(a.saved());assert.deepEqual(JSON.parse(JSON.stringify(restored.current().rothHistory)),JSON.parse(JSON.stringify(s.rothHistory)));
  const report=a.reportText(s,null);assert.match(report,/Remaining regular contributions: \$75,000/);assert.doesNotMatch(report,/Value source/);
  assert.match(report,/First Roth funding tax year: 2024/);assert.match(report,/Tax year 2025: remaining principal \$11,000.00; remaining taxable principal \$8,000.00/);
  await a.click('export-backup');const backup=JSON.parse(await a.downloads[0].blob.text());
  assert.deepEqual(backup.scenarios[0].rothHistory,JSON.parse(JSON.stringify(s.rothHistory)));
  await a.click('remove-roth-conversion',{index:'0'});assert.equal(s.rothHistory.conversions.length,0);
});

test('legacy Roth assumptions are visibly marked for review without changing saved balances',async()=>{
  const old=model.baseScenario();delete old.rothHistory;const a=app({scenarios:[old],selectedId:old.id});
  assert.equal(a.current().accounts.roth,50000);assert.equal(a.current().rothHistory.needsReview,true);
  a.state.setupSection=1;assert.match(a.setup(),/older plan had no Roth history/);assert.match(a.dashboard(),/Review Roth history/);
  assert.match(a.reportText(a.current(),null),/Roth history needs review: Yes/);
  await a.click('review-roth-history');assert.equal(a.current().rothHistory.needsReview,false);
  assert.equal(a.saved().scenarios[0].accounts.roth,50000);assert.doesNotMatch(a.setup(),/older plan had no Roth history/);
});

for(const prefix of ['rothHistory','spouseRothHistory']){
  const action=prefix==='rothHistory'?'remove-roth-conversion':'remove-spouse-conversion';
  const conversionPlan=()=>{
    const s=model.prepareCalendarScenario(model.baseScenario(),{needsReview:false});
    Object.assign(s.household,{separatePeople:true,filingStatus:'Married',spouseRetirementDate:s.household.retirementDate});
    s.spouseAccounts.roth=1000;s.spouseRothHistory.firstContributionYear=2021;
    return s;
  };
  test(prefix+' row removal clears an Unknown field without leaving a forecast blocker',async()=>{
    const s=conversionPlan();s[prefix].conversions=[{taxYear:2023,amount:100,taxableAmount:100}];
    const a=app({scenarios:[s],selectedId:s.id,inputSources:{[s.id]:{_origin:'Entered'}}});await a.resultRestoreReady;
    await a.change('#main',{dataset:{field:prefix+'.conversions.0.amount',type:'money'},value:''});
    await a.click(action,{index:'0'});
    assert.equal(a.current()[prefix].conversions.length,0);
    assert.deepEqual(guidance.unknownInputPaths(a.current(),a.state.inputSources),[]);
    const running=a.run();assert.equal(a.workers.length,1);
    await a.click('cancel-calculation');await running;
  });
  test(prefix+' row removal preserves source labels and Unknown status on remaining rows after reload',async()=>{
    const s=conversionPlan();s[prefix].conversions=[2022,2023,2024].map(taxYear=>({taxYear,amount:100,taxableAmount:100}));
    const notes={_origin:'Entered','accounts.cash':'Estimated',[prefix+'.conversions.0.taxYear']:'Estimated',[prefix+'.conversions.2.taxableAmount']:'Estimated'};
    const a=app({scenarios:[s],selectedId:s.id,inputSources:{[s.id]:notes}});await a.resultRestoreReady;
    await a.change('#main',{dataset:{field:prefix+'.conversions.1.amount',type:'money'},value:''});
    await a.click(action,{index:'0'});
    assert.equal(a.state.inputSources[s.id][prefix+'.conversions.0.amount'],'Unknown');
    assert.equal(a.state.inputSources[s.id][prefix+'.conversions.0.taxYear'],undefined);
    assert.equal(a.state.inputSources[s.id][prefix+'.conversions.1.taxableAmount'],'Estimated');
    assert.equal(a.state.inputSources[s.id][prefix+'.conversions.2.taxableAmount'],undefined);
    assert.equal(a.state.inputSources[s.id]['accounts.cash'],'Estimated');
    const restored=app(a.saved());await restored.resultRestoreReady;
    assert.deepEqual(guidance.unknownInputPaths(restored.current(),restored.state.inputSources),[prefix+'.conversions.0.amount']);
    await restored.run();assert.equal(restored.workers.length,0);
    assert.match(restored.state.message,/Review Unknown inputs/);
  });
}

test('tabbing from conversion principal keeps the visible taxable principal in sync without replacing the focused field',async()=>{
  const a=app();await a.click('add-roth-conversion');
  const taxable=a.element('#f-rothHistory-conversions-0-taxableAmount');
  Object.assign(taxable,{tagName:'INPUT',id:'f-rothHistory-conversions-0-taxableAmount',type:'number',dataset:{field:'rothHistory.conversions.0.taxableAmount',type:'money'},value:'0',replaceWith(){}});
  a.document.activeElement=taxable;
  const renders=a.renders();
  await a.change('#main',{dataset:{field:'rothHistory.conversions.0.amount',type:'money'},value:'20000'});
  assert.equal(a.current().rothHistory.conversions[0].taxableAmount,20000);
  assert.equal(taxable.value,'20,000','The next input must show the updated default when it receives focus');
  assert.equal(a.renders(),renders,'A principal edit should not replace the newly focused input');
  taxable.value='15000';await a.change('#main',taxable);
  await a.change('#main',{dataset:{field:'rothHistory.conversions.0.amount',type:'money'},value:'25000'});
  assert.equal(taxable.value,'15,000');
  assert.equal(a.current().rothHistory.conversions[0].taxableAmount,15000);
});

test('malformed Roth numeric edits and malformed history imports retain the saved plan',async()=>{
  const a=app();await a.click('add-roth-conversion');const before=a.stored(),history=JSON.stringify(a.current().rothHistory);
  for(const field of ['rothHistory.contributionBasis','rothHistory.conversions.0.amount']){
    await a.change('#main',{dataset:{field,type:'money'},value:'1e308'});
    assert.equal(a.stored(),before);assert.equal(JSON.stringify(a.current().rothHistory),history);
  }
  const bad=model.baseScenario();bad.rothHistory.conversions=[{taxYear:2024,amount:1000,taxableAmount:'1000'}];
  await a.change('#import-file',{files:[{text:async()=>JSON.stringify({scenarios:[bad]})}],value:'bad.json'});
  assert.equal(a.stored(),before);assert.equal(JSON.stringify(a.current().rothHistory),history);
  assert.match(a.state.message,/Roth conversion/);
});

test('malformed numeric changes keep saved assumptions, results and budget snapshots',async()=>{
  const targets=[
    {dataset:{field:'spending.annualBaseSpending',type:'money'}},
    {dataset:{field:'household.currentAge',type:'number'}},
    {dataset:{field:'market.stockMeanReturn',type:'percent'}},
    {dataset:{budget:'annualPropertyTaxes'}},
    ...['checking','credit','cashAndAtmWithdrawals','mortgage'].map(part=>({dataset:{month:'0',part}})),
  ];
  for(const target of targets)for(const invalid of [{value:'',validity:{badInput:true}},{value:'1e'},{value:'1e309'}]){
    const a=app(),s=a.current();
    s.budget.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:[{id:'original',monthlyAmount:4500}],creditCardBills:[{monthlyAmount:100}],cashAndAtmWithdrawals:50,adjustments:{mortgage:1000}}];
    model.applyBudgetEstimate(s);await a.persist();
    const original=JSON.stringify(s),saved=a.stored(),renders=a.renders(),result={completed:true};
    a.state.results.set(s.id,result);seedExploration(a);
    await a.change('#main',{...target,...invalid});
    assert.equal(JSON.stringify(s),original);assert.equal(a.stored(),saved);
    assert.equal(a.state.results.get(s.id),result);assert.ok(a.state.labResults);assert.ok(a.state.decision);
    assert.equal(a.renders(),renders);assert.match(a.state.message,/valid.*number.*previous value was kept/i);
  }
});

test('numeric editors reject unsupported dollar amounts and accept repaired entries and intentional zero',async()=>{
  for(const target of [{dataset:{field:'accounts.pretax',type:'money'}},{dataset:{budget:'annualPropertyTaxes'}},{dataset:{month:'0',part:'checking'}}]){
    const a=app(),s=a.current();s.budget.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:[{monthlyAmount:4500}]}];await a.persist();
    const original=JSON.stringify(s),saved=a.stored();
    await a.change('#main',{...target,value:'1e308',validity:{badInput:false}});
    assert.equal(JSON.stringify(s),original);assert.equal(a.stored(),saved);assert.match(a.state.message,/supported dollar range/i);
    await a.change('#main',{...target,value:'1250.50',validity:{badInput:false}});
    assert.notEqual(a.stored(),saved);assert.doesNotMatch(a.stored(),/:null/);
  }
  const a=app();
  await a.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:'0',validity:{badInput:false}});
  assert.equal(a.current().spending.annualBaseSpending,0);
  await a.change('#main',{tagName:'SELECT',dataset:{field:'rothConversion.marginalRateCap',type:'percent'},value:'24'});
  assert.equal(a.current().rothConversion.marginalRateCap,.24);
});

test('sensitivity views and reports disclose paired comparisons and their sample size',()=>{
  const a=app(),s=a.current(),r=runSimulation(s);a.state.results.set(s.id,r);
  const overview=a.dashboard(),results=a.results(),report=a.reportText(s,r);
  assert.match(overview,/Assumption sensitivity/);assert.match(overview,/General inflation/);
  assert.match(overview,/100 paired paths/);assert.match(results,/100 paired paths/);
  assert.match(report,/Most helpful sensitivity check:/);assert.match(report,/100 paired paths/);
  assert.doesNotMatch(report,/Primary risk:/);
  assert.match(overview,/Constant returns at the entered annual means/);
});

test('Results tables group monthly observations by whole-year age and keep the last snapshot together',()=>{
  const a=app();a.state.dollarBasis='future';const s=a.current();s.numberOfSimulations=4;const r=runSimulation(s);
  const ages=[65.5,65.75,66,66.25,66.5,66.75,67,67.25];
  const shares=[[1,1],[.75,1],[.75,.75],[.5,.75],[.5,.5],[.25,.5],[.25,.25],[.25,0]];
  r.notFailedByAge=ages.map((age,i)=>({age,notFailedShare:shares[i][0],aliveShare:shares[i][1]}));
  r.balanceBands=ages.map((age,i)=>({age,pessimistic:900-i*100,median:1000-i*100,optimistic:1100-i*100,pathCount:4-Math.floor(i/3)}));
  const monthlyData=JSON.stringify([r.notFailedByAge,r.balanceBands]);
  a.state.results.set(s.id,r);
  const html=a.results(),tables=[...html.matchAll(/<tbody>(.*?)<\/tbody>/gs)].map(table=>
    [...table[1].matchAll(/<tr>(.*?)<\/tr>/gs)].map(row=>[...row[1].matchAll(/<td>(.*?)<\/td>/gs)].map(cell=>cell[1])));
  assert.deepEqual([tables[0],tables[1].filter(row=>Number(row[0])<68)],[
    [['65','3 of 4','4 of 4'],['66','1 of 4','2 of 4'],['67','1 of 4','0 of 4 observed']],
    [['65','4','$800','$900','$1,000'],['66','3','$400','$500','$600'],['67','0','Not enough simulated outcomes','—','—']]
  ]);
  assert.match(html,/Each whole-year age shows its last modeled observation/);
  assert.equal(JSON.stringify([r.notFailedByAge,r.balanceBands]),monthlyData,'Table grouping preserves monthly chart data');
});

test('Results show yearly account and debt rows from the completed steady-growth run, collapsed by default',()=>{
  const a=app();a.state.dollarBasis='future';const s=a.current(),r=runSimulation(s);a.state.results.set(s.id,r);
  const details=JSON.stringify(r.steadySimulation),html=a.results();
  assert.match(html,/<details class="card steady-details"><summary><h2>Steady-growth illustration · yearly balances<\/h2>/);
  assert.match(html,/all volatility set to 0%/);
  assert.match(html,/long-term care risk turned off/);
  assert.match(html,/not a statistical median/);
  assert.match(html,/Assumed lifespan: you to age 95/);
  assert.doesNotMatch(html,/lower of the two middle paths/);
  for(const label of ['Pre-tax','Roth','Taxable','Cash','Home value','Mortgage debt','Portfolio total','Net assets'])assert.ok(html.includes(label),label);
  const monthly=html.match(/<div class="table-wrap monthly-balances".*?<tbody>(.*?)<\/tbody>/s)[1],rows=r.steadySimulation.monthlyDetails,last=rows.at(-1);
  const yearly=rows.filter((p,i)=>p.month%12===0||i===rows.length-1);
  assert.equal([...monthly.matchAll(/<tr>/g)].length,yearly.length);
  assert.ok(yearly.length<rows.length/10);
  assert.match(monthly,/Retirement start/);
  assert.ok(monthly.includes(last.month%12?'Final month':'Year '+last.month/12));
  const whole=new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0});
  assert.ok(monthly.includes(whole.format(rows[0].pretax)));
  assert.doesNotMatch(monthly,/\$[\d,]+\.\d\d/);
  s.accounts.pretax=123;s.numberOfSimulations=10000;
  assert.equal(a.results(),html,'Retained results use the completed balances and path count');
  assert.equal(JSON.stringify(r.steadySimulation),details);
});

test('Results disclose a steady-growth shortfall without labeling unmet costs as mortgage debt',()=>{
  const a=app(),s=a.current();s.accounts={pretax:0,roth:0,taxable:0,cash:0};s.socialSecurity.annualBenefitAt67=0;
  const r=runSimulation(s);a.state.results.set(s.id,r);
  const html=a.results();
  assert.match(html,/Stops in the month funds run short/);
  assert.match(html,/Unfunded amount/);assert.match(html,/final unmet cost, separate from mortgage debt/);
  assert.equal(r.steadySimulation.monthlyDetails.length,2);
});

test('Results explain retirement after the fixed lifespan without showing fictional monthly balances',()=>{
  const a=app(),s=a.current();Object.assign(s.household,{birthday:'1966-10-01',asOfDate:'2026-10-01',retirementDate:'2062-10-01',targetEndAge:119});
  const r=runSimulation(s);a.state.results.set(s.id,r);
  const html=a.results();
  assert.match(html,/Retirement begins on or after the end of the assumed household lifetime/);
  assert.doesNotMatch(html,/class="table-wrap monthly-balances"/);
  assert.deepEqual(r.steadySimulation.monthlyDetails,[]);
});

test('a stale tab cannot overwrite saved edits with an unrelated change or automatic Pro defaults',async()=>{
  const storage={raw:JSON.stringify({scenarios:[model.baseScenario()],selectedId:'base-plan'})},locks=webLocks();
  const first=app(null,{storage,locks}),second=app(null,{storage,locks,fetch:async()=>Response.json({tier:'pro',signedIn:true,accountKey:'owner'})});
  await first.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:'90000'});
  const saved=storage.raw;
  await second.change('#main',{dataset:{field:'accounts.cash',type:'money'},value:'60000'});
  assert.equal(storage.raw,saved);assert.equal(second.current().accounts.cash,60000);
  assert.match(second.dashboard(),/changed in another tab/);
  await second.loadAccess();assert.equal(storage.raw,saved);assert.match(second.dashboard(),/changed in another tab/);
  await second.click('export-backup');
  const backup=JSON.parse(await second.downloads[0].blob.text());assert.equal(backup.scenarios[0].accounts.cash,60000);
  const reloaded=app(null,{storage,locks});assert.equal(reloaded.current().spending.annualBaseSpending,90000);
  await reloaded.change('#main',{dataset:{field:'accounts.cash',type:'money'},value:'60000'});
  assert.equal(reloaded.saved().scenarios[0].spending.annualBaseSpending,90000);assert.equal(reloaded.saved().scenarios[0].accounts.cash,60000);
});

test('the storage lock serializes simultaneous saves from separate tabs',async()=>{
  const storage={raw:JSON.stringify({scenarios:[model.baseScenario()],selectedId:'base-plan'})},locks=webLocks();
  const first=app(null,{storage,locks}),second=app(null,{storage,locks});
  first.current().spending.annualBaseSpending=90000;second.current().accounts.cash=60000;
  const results=await Promise.all([first.persist(),second.persist()]);
  assert.deepEqual(results,[true,false]);
  assert.equal(first.saved().scenarios[0].spending.annualBaseSpending,90000);
  assert.equal(first.saved().scenarios[0].accounts.cash,50000);assert.match(second.state.storageError,/changed in another tab/);
});

test('one tab queues rapid edits without losing either change or reporting a conflict',async()=>{
  const a=app();
  await Promise.all([
    a.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:'90000'}),
    a.change('#main',{dataset:{field:'accounts.cash',type:'money'},value:'60000'}),
  ]);
  assert.equal(a.saved().scenarios[0].spending.annualBaseSpending,90000);assert.equal(a.saved().scenarios[0].accounts.cash,60000);
  assert.equal(a.state.storageError,'');
});

test('external storage updates warn without discarding local drafts or completed calculations',async()=>{
  const storage={raw:JSON.stringify({scenarios:[model.baseScenario()],selectedId:'base-plan'})},locks=webLocks();
  const first=app(null,{storage,locks}),second=app(null,{storage,locks});
  second.current().accounts.cash=60000;second.state.results.set(second.current().id,runSimulation(second.current()));
  await first.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:'90000'});
  const saved=storage.raw;second.storageChanged();
  assert.equal(second.current().accounts.cash,60000);assert.equal(second.state.results.size,1);assert.equal(storage.raw,saved);
  assert.match(second.dashboard(),/changed in another tab/);
});

test('explicit import and unreadable-backup replacement cannot bypass another tab update',async()=>{
  for(const unreadable of [false,true]){
    const storage={raw:JSON.stringify(unreadable?{scenarios:[null]}:{scenarios:[model.baseScenario()],selectedId:'base-plan'})},a=app(null,{storage});
    storage.raw=JSON.stringify({scenarios:[{...model.baseScenario(),name:'Newer plan'}],selectedId:'base-plan'});
    const newer=storage.raw;
    if(unreadable)await a.click('replace-unreadable-plans');
    else await a.change('#import-file',{files:[{text:async()=>JSON.stringify([model.baseScenario()])}],value:'backup.json'});
    assert.equal(storage.raw,newer);assert.match(a.state.storageError,/changed in another tab/);
  }
});

test('a browser without storage locks keeps drafts exportable and does not save unsafely',async()=>{
  const a=app(null,{locks:null});
  await a.change('#main',{dataset:{field:'accounts.cash',type:'money'},value:'60000'});
  assert.equal(a.saved(),null);assert.equal(a.current().accounts.cash,60000);assert.match(a.dashboard(),/Automatic saving is unavailable/);
  await a.click('export-backup');assert.equal(JSON.parse(await a.downloads[0].blob.text()).scenarios[0].accounts.cash,60000);
});

test('closing a tab with pending or blocked saves warns; a successful save clears the warning',async()=>{
  const locks=webLocks();let release;
  const held=locks.request('retirement-readiness-lab-sites-v1',()=>new Promise(resolve=>{release=resolve;}));
  await Promise.resolve();
  const a=app(null,{locks}),saving=a.persist();
  const leaving={prevented:false,preventDefault(){this.prevented=true;}};a.beforeUnload(leaving);assert.equal(leaving.prevented,true);
  release();await held;assert.equal(await saving,true);
  const saved={prevented:false,preventDefault(){this.prevented=true;}};a.beforeUnload(saved);assert.equal(saved.prevented,false);
  a.state.storageError='Error: Unsaved edits';const blocked={prevented:false,preventDefault(){this.prevented=true;}};a.beforeUnload(blocked);assert.equal(blocked.prevented,true);
});

test('recovery drafts warn on closing until an explicit replacement or valid import saves them',async()=>{
  for(const recover of ['replace','import']){
    const rawStorage='{broken-json',a=app(null,{rawStorage});
    const leaving=()=>({prevented:false,preventDefault(){this.prevented=true;}});
    let exit=leaving();a.beforeUnload(exit);assert.equal(exit.prevented,false);
    await a.persist(false);exit=leaving();a.beforeUnload(exit);assert.equal(exit.prevented,false,'Automatic defaults do not mark recovery edits');
    await a.change('#main',{dataset:{field:'accounts.cash',type:'money'},value:'123456'});
    assert.equal(a.stored(),rawStorage);exit=leaving();a.beforeUnload(exit);assert.equal(exit.prevented,true);
    await a.click('export-backup');assert.equal(JSON.parse(await a.downloads[0].blob.text()).scenarios[0].accounts.cash,123456);
    if(recover==='replace')await a.click('replace-unreadable-plans');
    else await a.change('#import-file',{files:[{text:async()=>JSON.stringify([a.current()])}],value:'backup.json'});
    assert.equal(a.saved().scenarios[0].accounts.cash,123456);exit=leaving();a.beforeUnload(exit);assert.equal(exit.prevented,false);
  }
});

test('edits blocked during an in-flight recovery replacement retain their closing warning',async()=>{
  const locks=webLocks();let release;
  const held=locks.request('retirement-readiness-lab-sites-v1',()=>new Promise(resolve=>{release=resolve;}));
  await Promise.resolve();
  const a=app(null,{rawStorage:'{broken-json',locks});
  await a.change('#main',{dataset:{field:'accounts.cash',type:'money'},value:'123456'});
  const replacement=a.click('replace-unreadable-plans');
  await a.change('#main',{dataset:{field:'accounts.cash',type:'money'},value:'234567'});
  release();await held;await replacement;
  assert.equal(a.saved().scenarios[0].accounts.cash,123456);assert.equal(a.current().accounts.cash,234567);
  const exit={prevented:false,preventDefault(){this.prevented=true;}};a.beforeUnload(exit);assert.equal(exit.prevented,true);
  await a.persist();assert.equal(a.saved().scenarios[0].accounts.cash,234567);
  exit.prevented=false;a.beforeUnload(exit);assert.equal(exit.prevented,false);
});

test('repairing a missing or duplicate scenario ID preserves later IDs and the selected plan',async()=>{
  for(const repeated of [false,true]){
    const missing=model.baseScenario(),selected=model.baseScenario();delete missing.id;
    missing.name='Needs an ID';selected.id='plan-imported-1';selected.name='Selected plan';
    const scenarios=repeated?[{...model.baseScenario(),id:'duplicate'}, {...missing,id:'duplicate'}, selected]:[missing,selected];
    const a=app({scenarios,selectedId:selected.id});
    assert.equal(a.current().name,'Selected plan');assert.equal(a.current().id,selected.id);
    assert.equal(new Set(a.state.scenarios.map(s=>s.id)).size,scenarios.length);
    await a.persist();const restored=app(a.saved());assert.equal(restored.current().name,'Selected plan');
  }
});

test('saved selections follow normalized numeric and trimmed scenario IDs',()=>{
  for(const id of [7,' selected ']){
    const selected={...model.baseScenario(),id,name:'Selected plan'};
    const a=app({scenarios:[model.baseScenario(),selected],selectedId:id});
    assert.equal(a.current().name,'Selected plan');assert.equal(a.state.selectedId,String(id).trim());
  }
});

test('owner billing shows existing portal access and explicitly labeled sandbox checkout',()=>{
  const a=app();
  a.state.access={tier:'pro',signedIn:true,ownerAccess:true,billingPortalAvailable:true,checkoutAvailable:true,testBilling:true};
  let html=a.billingView();
  assert.match(html,/data-action="billing-portal"/);assert.match(html,/data-action="checkout"/);assert.match(html,/do not charge real money/);
  a.state.access.testBilling=false;html=a.billingView();
  assert.match(html,/data-action="billing-portal"/);assert.doesNotMatch(html,/data-action="checkout"/);
  a.state.access.ownerAccess=false;a.state.access.testBilling=true;
  assert.doesNotMatch(a.billingView(),/data-action="checkout"/);
});

test('owner billing outages retain customer lookup hints and complimentary Pro',async()=>{
  const session=new Map([['retirement-billing-reference',JSON.stringify({customerId:'cus_owner'})]]);
  const a=app(null,{session,fetch:async()=>Response.json({tier:'pro',signedIn:true,ownerAccess:true,accountKey:'owner',billingPortalAvailable:true,billingLookupUnavailable:true,checkoutAvailable:false})});
  await a.loadAccess();
  assert.equal(a.isPro(),true);assert.equal(JSON.parse(session.get('retirement-billing-reference')).customerId,'cus_owner');
  assert.match(a.billingView(),/Billing lookup is temporarily unavailable/);assert.match(a.billingView(),/data-action="billing-portal"/);
});

test('malformed backup structures cannot replace saved scenarios',async()=>{
  const edits=[s=>s.budget.monthlyBudgets={},s=>s.budget.monthlyBudgets=null,s=>s.budget.monthlyBudgets=[null],s=>s.budget.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:{}}],s=>s.budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[null]}],s=>s.budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[{monthlyAmount:'100'}]}],s=>s.budget.monthlyBudgets=[{month:'2026-01',adjustments:[]}]];
  for(const edit of edits){
    const original={scenarios:model.sampleScenarios(),selectedId:'base-plan'},a=app(original),before=JSON.stringify(a.state.scenarios),bad=model.baseScenario();
    bad.id='malformed-import';edit(bad);
    await a.change('#import-file',{files:[{text:async()=>JSON.stringify([bad])}],value:'backup.json'});
    assert.match(a.state.message,/Error:/);assert.equal(JSON.stringify(a.state.scenarios),before);assert.deepEqual(a.saved(),original);
    assert.doesNotThrow(()=>a.budget());assert.doesNotThrow(()=>a.reportText(a.current()));
  }
});

test('malformed scenario entries and sections cannot be hidden by defaults during import',async()=>{
  const entries=[null,42,'invalid',true,[],{household:null},{accounts:[]},{budget:'invalid'},{household:null,currentAge:50}];
  for(const bad of entries){
    const original={scenarios:model.sampleScenarios(),selectedId:'base-plan'},a=app(original),before=JSON.stringify(a.state.scenarios);
    await a.change('#import-file',{files:[{text:async()=>JSON.stringify([model.baseScenario(),bad])}],value:'backup.json'});
    assert.match(a.state.message,/Error:/);assert.equal(JSON.stringify(a.state.scenarios),before);assert.deepEqual(a.saved(),original);
  }
});

test('corrupt saved scenario structures do not crash startup or get overwritten by Pro defaults',async()=>{
  const original={scenarios:[null],selectedId:'old'},a=app(original);
  assert.match(a.state.message,/Saved plans could not be loaded/);
  a.state.scenarios.forEach(model.applyProSimulationDefault);assert.equal(await a.persist(false),false);
  assert.deepEqual(a.saved(),original);
});

test('unreadable saved plans survive navigation and edits until replacement is explicitly confirmed',async()=>{
  const original={scenarios:[model.baseScenario(),null],selectedId:'base-plan'};
  let approved=false;
  const a=app(original,{confirm:()=>approved});
  await a.click('start-plan');
  await a.click('select-scenario',{id:'later-retirement'});
  await a.change('#scenario-select',{value:'base-plan'});
  await a.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'123456'});
  await a.change('#main',{dataset:{budget:'annualPropertyTaxes'},value:'1000'});
  assert.deepEqual(a.saved(),original);
  assert.match(a.dashboard(),/export-unreadable-backup/);
  assert.match(a.dashboard(),/replace-unreadable-plans/);
  await a.click('replace-unreadable-plans');
  assert.deepEqual(a.saved(),original);
  approved=true;await a.click('replace-unreadable-plans');
  assert.equal(a.saved().scenarios[0].accounts.pretax,123456);
  assert.equal(a.saved().scenarios[0].budget.annualPropertyTaxes,1000);
  assert.doesNotMatch(a.dashboard(),/Saved plans could not be loaded/);
});

test('failed recovery replacement preserves the unreadable backup and can be retried',async()=>{
  const original={scenarios:[null]},storage={fail:true},a=app(original,{storage});
  await a.click('replace-unreadable-plans');
  assert.deepEqual(a.saved(),original);
  assert.match(a.dashboard(),/could not be saved/);
  assert.match(a.dashboard(),/replace-unreadable-plans/);
  storage.fail=false;await a.click('replace-unreadable-plans');
  assert.equal(a.saved().scenarios.length,3);
  assert.doesNotMatch(a.dashboard(),/could not be saved|replace-unreadable-plans/);
});

test('recovery export preserves the exact original backup even when JSON parsing fails',async()=>{
  const rawStorage='  {"scenarios": [broken JSON\n',a=app(null,{rawStorage});
  await a.click('start-plan');await a.click('export-unreadable-backup');
  assert.equal(a.downloads.length,1);
  assert.equal(a.downloads[0].name,'retirement-unreadable-backup.json');
  assert.equal(await a.downloads[0].blob.text(),rawStorage);
  assert.equal(a.stored(),rawStorage);
});

test('a valid explicit import can replace unreadable saved plans',async()=>{
  const a=app({scenarios:[null]}),s=model.baseScenario();s.id='recovered';
  await a.change('#import-file',{files:[{text:async()=>JSON.stringify([s])}],value:'backup.json'});
  assert.equal(a.saved().scenarios[0].id,'recovered');
  assert.doesNotMatch(a.dashboard(),/Saved plans could not be loaded/);
});

test('stored budgets and primitive types are checked before any view can render them',async()=>{
  const edits=[s=>s.budget.monthlyBudgets=null,s=>s.budget.monthlyBudgets=[null],s=>s.budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[null]}],s=>s.budget.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:{}}],s=>s.accounts.pretax='800000',s=>s.spending.annualBaseSpending=null];
  for(const edit of edits){
    const bad=model.baseScenario();edit(bad);
    const original={scenarios:[model.baseScenario(),bad],selectedId:bad.id},a=app(original);
    assert.match(a.state.message,/Saved plans could not be loaded/);
    assert.doesNotThrow(()=>a.budget());assert.doesNotThrow(()=>a.reportText(a.current()));
    a.state.scenarios.forEach(model.applyProSimulationDefault);assert.equal(await a.persist(false),false);
    assert.deepEqual(a.saved(),original);
  }
  for(const original of [{scenarios:null},{scenarios:[]},{}]){
    const a=app(original);assert.match(a.state.message,/Saved plans could not be loaded/);
    assert.equal(await a.persist(false),false);assert.deepEqual(a.saved(),original);
  }
});

test('safe stored drafts can still be loaded and corrected without losing their edits',()=>{
  const s=model.baseScenario();s.household.retirementAge=59;
  s.budget.monthlyBudgets=[{month:'',creditCardBills:[{monthlyAmount:-100}],adjustments:{mortgage:200}}];
  const original={scenarios:[s],selectedId:s.id},a=app(original);
  assert.equal(a.state.message,'');assert.deepEqual(JSON.parse(JSON.stringify(a.current())),model.prepareCalendarScenario(structuredClone(s)));assert.deepEqual(a.saved(),original);
  assert.doesNotThrow(()=>a.budget());assert.doesNotThrow(()=>a.reportText(a.current()));
  assert.equal(a.current().household.retirementAge,59);
});

test('calculation updates preserve focused drafts through success and failure until change commits them',async()=>{
  for(const task of ['run','runDecision','runLab'])for(const outcome of ['result','error'])for(const dataset of [{field:'accounts.pretax',type:'money'},{budget:'annualPropertyTaxes'},{month:'0',part:'credit'}]){
    const a=app();a.state.access.tier='pro';a.current().numberOfSimulations=4;
    a.state.view=dataset.field?'setup':'budget';a.state.setupSection=1;
    a.current().budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[]}];
    const main=a.element('#main');let html=main.innerHTML;
    Object.defineProperty(main,'innerHTML',{get:()=>html,set(value){html=value;a.document.activeElement=null;}});
    const editor={tagName:'INPUT',type:'number',id:'live-editor',dataset,value:'123456',focus(){a.document.activeElement=this;}};
    const replacement=a.element('#live-editor');replacement.type='number';
    let retained;replacement.replaceWith=node=>{retained=node;};
    const pending=a[task]();a.document.activeElement=editor;
    const workerCount=task==='runLab'?4:1;
    for(let i=0;i<workerCount;i++){
      const worker=a.workers[i];assert.ok(worker);
      worker.onmessage({data:outcome==='error'?{type:'error',message:'Test calculation failure'}:{type:'result',result:task==='runDecision'?{targetReadiness:.8,simulationCount:200}:runSimulation(worker.data.scenario)}});
      await Promise.resolve();
      assert.equal(a.document.activeElement,editor);assert.equal(retained,editor);assert.equal(editor.value,'123456');
    }
    await pending;assert.equal(a.document.activeElement,editor);assert.equal(a.state.busy,false);
    assert.equal(a.current().accounts.pretax,175000);assert.equal(a.current().budget.annualPropertyTaxes,0);
    await a.change('#main',editor);
    const s=a.saved().scenarios[0];
    assert.equal(dataset.field?s.accounts.pretax:dataset.budget?s.budget.annualPropertyTaxes:s.budget.monthlyBudgets[0].creditCardBills[0].monthlyAmount,123456);
    if(dataset.field){assert.equal(a.state.results.size,0);assertCleared(a);}
  }
});

test('background billing renders retain the original live editor and its uncommitted value',async()=>{
  for(const access of [
    {tier:'pro',maxPaths:10000,signedIn:true,accountKey:'user'},
    {tier:'free',maxPaths:100,signedIn:false,checkoutAvailable:true},
    {error:'unauthorized'},
  ])for(const dataset of [{field:'accounts.pretax',type:'money'},{budget:'annualPropertyTaxes'},{month:'0',part:'credit'}]){
    let finish;const a=app(null,{fetch:()=>new Promise(resolve=>{finish=resolve;})});
    a.state.view=dataset.field?'setup':'budget';a.state.setupSection=1;
    a.current().budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[]}];
    const main=a.element('#main');let html=main.innerHTML;
    // Replacing the page disconnects its focused input, as in a browser.
    Object.defineProperty(main,'innerHTML',{get:()=>html,set(value){html=value;a.document.activeElement=null;}});
    const details={tagName:'DETAILS',open:false};
    const editor={tagName:'INPUT',type:'number',id:'live-editor',dataset,value:'123456',parentElement:details,focus(){a.document.activeElement=this;}};
    const replacement=a.element('#live-editor');replacement.type='number';
    let retained;replacement.replaceWith=node=>{retained=node;};
    const pending=a.loadAccess({force:true});
    while(!finish)await Promise.resolve();
    a.document.activeElement=editor;
    finish(Response.json(access,{status:access.error?401:200}));await pending;
    assert.equal(details.open,true);assert.equal(retained,editor);assert.equal(a.document.activeElement,editor);assert.equal(editor.value,'123456');
    await a.change('#main',editor);
    const s=a.saved().scenarios[0];
    assert.equal(dataset.field?s.accounts.pretax:dataset.budget?s.budget.annualPropertyTaxes:s.budget.monthlyBudgets[0].creditCardBills[0].monthlyAmount,123456);
  }
});

test('delayed billing and sign-in errors preserve unfinished edits after leaving the billing view',async()=>{
  for(const action of ['checkout','billing-portal','social-signin','social-link-provider'])for(const dataset of [{field:'household.currentAge',type:'number'},{budget:'annualPropertyTaxes'},{month:'0',part:'credit'}]){
    let finish;
    const deferred=()=>new Promise((resolve,reject)=>{finish={resolve,reject};});
    const a=app(null,{fetch:deferred,socialActions:{signInSocial:deferred,linkSocialProvider:deferred}});
    a.state.view='billing';
    const pending=a.click(action,{interval:'monthly',provider:'google'});
    while(!finish)await Promise.resolve();
    a.current().budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[]}];
    a.element('#navigation').listeners.click({target:{closest:()=>({dataset:{view:dataset.field?'setup':'budget'}})}});
    const main=a.element('#main');let html=main.innerHTML;
    Object.defineProperty(main,'innerHTML',{get:()=>html,set(value){html=value;a.document.activeElement=null;}});
    const editor={tagName:'INPUT',type:'number',id:'live-editor',dataset,value:dataset.field?'70':'123456',focus(){a.document.activeElement=this;}};
    const replacement=a.element('#live-editor');replacement.type='number';let retained;
    replacement.replaceWith=node=>{retained=node;};a.document.activeElement=editor;
    if(action.startsWith('social-'))finish.reject(new Error('Temporary sign-in outage'));
    else finish.resolve(Response.json({error:'Temporary billing outage'},{status:502}));
    await pending;
    assert.equal(a.document.activeElement,editor);assert.equal(retained,editor);
    assert.equal(editor.value,dataset.field?'70':'123456');assert.equal(a.current().household.currentAge,60);
    assert.equal(a.current().budget.annualPropertyTaxes,0);assert.match(a.state.message,/Temporary .* outage/);
    await a.change('#main',editor);
    const s=a.saved().scenarios[0];
    assert.equal(dataset.field?s.household.currentAge:dataset.budget?s.budget.annualPropertyTaxes:s.budget.monthlyBudgets[0].creditCardBills[0].monthlyAmount,dataset.field?70:123456);
  }
});

test('a delayed sign-out confirmation preserves a live setup editor',async()=>{
  let finish;
  const a=app(null,{fetch:async()=>Response.json({tier:'free',signedIn:false,accountKey:null}),socialActions:{signOutSocial:()=>new Promise(resolve=>{finish=resolve;})}});
  a.state.view='billing';const pending=a.click('social-signout');
  a.element('#navigation').listeners.click({target:{closest:()=>({dataset:{view:'setup'}})}});
  const main=a.element('#main');let html=main.innerHTML;
  Object.defineProperty(main,'innerHTML',{get:()=>html,set(value){html=value;a.document.activeElement=null;}});
  const editor={tagName:'INPUT',type:'number',id:'live-editor',dataset:{field:'household.currentAge'},value:'70',focus(){a.document.activeElement=this;}};
  const replacement=a.element('#live-editor');replacement.type='number';let retained;
  replacement.replaceWith=node=>{retained=node;};a.document.activeElement=editor;
  finish();await pending;
  assert.equal(retained,editor);assert.equal(a.document.activeElement,editor);assert.equal(editor.value,'70');
  assert.equal(a.current().household.currentAge,60);assert.match(a.state.message,/Signed out of Google/);
});

test('unfinished but structurally valid budget drafts remain importable',async()=>{
  const a=app(),draft=model.baseScenario();draft.budget.monthlyBudgets=[{month:'',checkingSavingsBills:[],creditCardBills:[],cashAndAtmWithdrawals:0}];
  await a.change('#import-file',{files:[{text:async()=>JSON.stringify([draft])}],value:'backup.json'});
  assert.match(a.state.message,/1 scenarios imported/);assert.doesNotThrow(()=>a.budget());assert.doesNotThrow(()=>a.reportText(a.current()));
});

test('backups restore incomplete retirement and financial drafts alongside completed plans',async()=>{
  const a=app();await a.click('retirement-status',{owner:'you',mode:'retired'});
  const reloaded=app(a.saved());await reloaded.click('retirement-status',{owner:'you',mode:'future'});
  const draft=reloaded.current(),other=reloaded.state.scenarios[1];
  assert.equal(draft.household.retirementDate,'');assert.equal(reloaded.state.inputSources[draft.id]['household.retirementDate'],'Unknown');
  other.mortgage.currentBalance=50000;other.mortgage.monthlyPayment=0;
  await reloaded.click('export-backup');const text=await reloaded.downloads.at(-1).blob.text();
  const imported=app();await imported.change('#import-file',{files:[{text:async()=>text}],value:'backup.json'});
  assert.match(imported.state.message,/3 scenarios imported/);
  assert.deepEqual(JSON.parse(JSON.stringify(imported.state.scenarios)),JSON.parse(text).scenarios.map(model.normalizeScenario));
  assert.equal(imported.state.inputSources[draft.id]['household.retirementDate'],'Unknown');
  assert.equal(imported.current().household.retirementDate,'');
  assert.doesNotThrow(()=>imported.setup());assert.doesNotThrow(()=>imported.reportText(imported.current()));
  const loaded=app(imported.saved());assert.equal(loaded.current().household.retirementDate,'');
  await loaded.run();assert.equal(loaded.workers.length,0);assert.match(loaded.state.message,/Retirement date/);
  await loaded.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:model.addCalendarMonths(model.localCalendarDate(),84)});
  const running=loaded.run();assert.equal(loaded.workers.length,1);await loaded.click('cancel-calculation');await running;
  await loaded.click('select-scenario',{id:other.id});await loaded.run();
  assert.equal(loaded.workers.length,1);assert.match(loaded.state.message,/Mortgage payments/);
});

test('draft backup restoration still rejects unsupported choices, malformed dates and unsafe numbers',async()=>{
  for(const edit of [s=>s.household.filingStatus='invalid',s=>s.household.gender='invalid',s=>s.spending.spendingPathModel='invalid',s=>s.household.retirementDate='not-a-date',s=>s.accounts.pretax=1e308,s=>s.accounts.pretax=null]){
    const a=app(),before=a.stored(),bad=model.prepareCalendarScenario(model.baseScenario());edit(bad);
    await a.change('#import-file',{files:[{text:async()=>JSON.stringify([bad])}],value:'backup.json'});
    assert.match(a.state.message,/Error:/);assert.equal(a.stored(),before);assert.equal(a.current().accounts.pretax,175000);
  }
});

test('failed assumption saves show an error and recover after a successful retry',async()=>{
  const storage={fail:true},a=app(null,{storage});
  await a.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'123456'});
  assert.equal(a.current().accounts.pretax,123456);assert.equal(a.saved(),null);
  assert.match(a.state.message,/Error: Changes could not be saved/);assert.doesNotMatch(a.state.message,/^Saved/);
  assert.equal(a.element('#main > .notice').className,'notice error');
  storage.fail=false;await a.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'234567'});
  assert.equal(a.saved().scenarios[0].accounts.pretax,234567);assert.match(a.state.message,/^Saved/);assert.equal(a.state.storageError,'');
});

test('failed budget saves display a warning without leaving the editor',async()=>{
  const a=app(null,{storage:{fail:true}});a.state.view='budget';
  await a.change('#main',{dataset:{budget:'annualPropertyTaxes'},value:'1000'});
  assert.match(a.element('#main > .notice').textContent,/could not be saved/);assert.equal(a.saved(),null);assert.equal(a.state.view,'budget');
});

test('copy, reset, apply and import cannot overwrite a failed-save warning',async()=>{
  for(const action of ['new-scenario','reset-assumptions','apply-budget']){
    const a=app(null,{storage:{fail:true}});a.current().budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[{monthlyAmount:1000}]}];
    await confirmBudgetPaymentChoices(a);await a.click(action);assert.match(a.state.message,/could not be saved/);assert.equal(a.saved(),null);
  }
  const a=app(null,{storage:{fail:true}});
  await a.change('#import-file',{files:[{text:async()=>JSON.stringify([model.baseScenario()])}],value:'backup.json'});
  assert.match(a.state.message,/could not be saved/);assert.equal(a.saved(),null);
});

test('billing portal remains visible for a free account with an existing customer',async()=>{
  const a=app(null,{fetch:async()=>Response.json({tier:'free',maxPaths:100,signedIn:true,checkoutAvailable:true,billingPortalAvailable:true})});
  await a.loadAccess();assert.match(a.billingView(),/data-action="billing-portal"/);
  a.state.access.billingPortalAvailable=false;assert.doesNotMatch(a.billingView(),/data-action="billing-portal"/);
});

test('first visit leads with starting actions and an explicitly illustrative chart',()=>{
  const a=app(),html=a.dashboard();
  assert.match(html,/Explore how long your retirement savings could last/);
  assert.match(html,/Build my forecast/);assert.match(html,/Explore a sample plan/);
  assert.match(html,/Illustrative paths only/);assert.match(html,/Your financial inputs stay in your browser/);
  assert.match(html,/Free preview · 100 simulated lifetimes/);assert.match(html,/Sample plan at a glance/);
  assert.ok(html.indexOf('Build my forecast')<html.indexOf('overview-upgrade-title'));
  assert.equal(a.saved(),null);
});
test('starting a plan opens assumptions and survives reload without replacing scenarios',async()=>{
  const a=app(),expected=structuredClone(a.state.scenarios);expected[0].name='My retirement plan';const before=JSON.stringify(expected);
  await a.click('start-plan');assert.equal(a.state.view,'setup');assert.equal(JSON.stringify(a.state.scenarios),before);
  const restored=app(a.saved());assert.match(restored.dashboard(),/Continue my plan/);
  assert.doesNotMatch(restored.dashboard(),/Explore a sample plan/);
  assert.equal(JSON.stringify(restored.state.scenarios),before);
});
test('existing backups resume the selected plan and escape its name',()=>{
  const plans=model.sampleScenarios();plans[1].name='<b>My plan</b>';
  const a=app({scenarios:plans,selectedId:plans[1].id});
  assert.equal(a.current().id,plans[1].id);assert.match(a.dashboard(),/Continue my plan/);
  assert.match(a.dashboard(),/&lt;b&gt;My plan&lt;\/b&gt;/);assert.doesNotMatch(a.dashboard(),/<b>My plan<\/b>/);
});
test('automatic Pro defaults preserve the first-visit state across reloads',async()=>{
  const a=app();a.state.access.tier='pro';a.state.scenarios.forEach(model.applyProSimulationDefault);await a.persist(false);
  const restored=app(a.saved());restored.state.access.tier='pro';const html=restored.dashboard();
  assert.match(html,/Build my forecast/);assert.match(html,/Pro · Up to 10,000/);
  assert.doesNotMatch(html,/overview-upgrade-title|Free preview · 100 simulated lifetimes/);
});

test('both scenario selectors discard prior comparisons and targets',async()=>{
  const a=app();seedExploration(a);await a.click('select-scenario',{id:'later-retirement'});assert.equal(a.current().id,'later-retirement');assertCleared(a);
  seedExploration(a);await a.change('#scenario-select',{value:'base-plan'});assert.equal(a.current().id,'base-plan');assertCleared(a);
});
test('reset, duplicate, delete and backup replacement invalidate exploration',async()=>{
  const a=app();for(const action of ['reset-assumptions','new-scenario','delete-scenario']){seedExploration(a);await a.click(action,{id:a.current().id});assertCleared(a);}
  seedExploration(a);await a.change('#import-file',{files:[{text:async()=>JSON.stringify([model.baseScenario()])}],value:'backup.json'});assertCleared(a);
});
test('switching plans while a comparison worker runs cannot restore old rows',async()=>{
  const a=app(),pending=a.runLab();assert.equal(a.workers.length,1);
  await a.click('select-scenario',{id:'later-retirement'});
  a.workers[0].onmessage({data:{type:'result',result:runSimulation(a.workers[0].data.scenario)}});
  await pending;assertCleared(a);assert.equal(a.workers.length,1);assert.equal(a.state.busy,false);assert.doesNotMatch(a.lab(),/Comparisons ready|Old plan/);
});
test('an input edit during simulation discards the obsolete result',async()=>{
  const a=app(),pending=a.run();await a.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:'90000'});
  a.workers[0].onmessage({data:{type:'result',result:runSimulation(a.workers[0].data.scenario)}});
  await pending;assert.equal(a.state.results.size,0);assert.equal(a.current().spending.annualBaseSpending,90000);assert.equal(a.state.busy,false);
});
test('resetting a plan while target search runs discards its old targets',async()=>{
  const a=app();a.state.access.tier='pro';const pending=a.runDecision();await a.click('reset-assumptions');
  a.workers[0].onmessage({data:{type:'result',result:{earliestRetirementAge:55}}});await pending;assertCleared(a);assert.equal(a.state.busy,false);
});
test('unchanged runs still publish results and complete all comparison rows',async()=>{
  const a=app(),pending=a.run();a.workers[0].onmessage({data:{type:'result',result:runSimulation(a.workers[0].data.scenario)}});await pending;assert.equal(a.state.results.size,1);
  await a.click('lab-add',{preset:'spend-less'});const lab=a.runLab();await settle(a,lab);
  assert.equal(labRows(a).filter(r=>r.result).length,2);assert.equal(a.workers.length,3);assert.equal(a.state.busy,false);assert.equal(a.state.results.size,1);
});
test('lower-spending comparisons use the same home-sale assumptions as an editor change',async()=>{
  const s=model.baseScenario();s.home.currentValue=300000;
  s.budget.annualPropertyTaxes=20000;s.budget.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:[{monthlyAmount:2500}]}];
  model.applyBudgetEstimate(s);
  const a=app({scenarios:[s],selectedId:s.id});await a.click('lab-add',{preset:'spend-less'});
  const pending=a.runLab();let spendingVariant;
  await settle(a,pending,worker=>{if(a.workers.indexOf(worker)===1)spendingVariant=structuredClone(worker.data.scenario);else assert.equal(worker.data.scenario.budget.isAppliedToAnnualBaseSpending,true);return runSimulation(worker.data.scenario);});
  assert.equal(a.current().budget.isAppliedToAnnualBaseSpending,true,'Comparison must preserve the original budget');
  const comparisonReadiness=labRows(a)[1].result.successProbability;
  await a.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:String(s.spending.annualBaseSpending*.95)});
  assert.deepEqual(spendingVariant.budget,structuredClone(a.current().budget));
  const entered=structuredClone(a.current());entered.numberOfSimulations=spendingVariant.numberOfSimulations;
  assert.equal(runSimulation(entered).successProbability,comparisonReadiness);
});
test('100-path outcomes use counts and a visible warning in free and Pro views and reports',async()=>{
  for(const tier of ['free','pro']){
    const a=app();a.state.access.tier=tier;const r=runSimulation(a.current());a.state.results.set(a.current().id,r);
    await settle(a,a.runLab(),w=>w===a.workers[0]?structuredClone(r):runSimulation(w.data.scenario));
    const label=format.readinessLabel(r),percent=`${(100*r.successProbability).toFixed(1)}%`;assert.match(label,/^\d+ of 100$/);
    for(const html of [a.results(),a.dashboard(),a.lab()]){assert.ok(html.includes(label),label);assert.match(html,/Sample preview only/);assert.ok(!html.includes(percent),percent);}
    const report=a.reportText(a.current(),r);assert.ok(report.includes(`Lifetimes without a portfolio shortfall: ${label}`));assert.match(report,/SAMPLE PREVIEW ONLY/);assert.doesNotMatch(report,/Modeled readiness.*100\.0%/);
  }
});
test('larger runs retain percentage summaries without the small-preview warning',()=>{
  const a=app();a.state.access.tier='pro';a.current().numberOfSimulations=101;const r=runSimulation(a.current());a.state.results.set(a.current().id,r);
  assert.match(a.results(),/Monte Carlo readiness/);assert.match(a.results(),/\d+\.\d%/);assert.doesNotMatch(a.results(),/Sample preview only/);
  assert.equal(format.shareLabel(.75,4),'3 of 4');assert.equal(format.shareLabel(.75,101),'75.0%');
});

test('retained reports and comparisons keep their actual counts across upgrades and expiration',async()=>{
  for(const [completedCount,nextTier,nextCount] of [[4,'pro',10000],[10,'pro',10000],[100,'pro',10000],[150,'free',100]]){
    const a=app(null,{fetch:async()=>Response.json({tier:nextTier,signedIn:true,accountKey:'user-a'})});
    a.state.access.tier=completedCount>100?'pro':'free';
    a.current().numberOfSimulations=completedCount;
    const r=runSimulation(a.current());a.state.results.set(a.current().id,r);
    await settle(a,a.runLab(),w=>runSimulation({...w.data.scenario,numberOfSimulations:completedCount}));
    const completedLab=labRows(a).find(row=>row.result);completedLab.result;
    await a.loadAccess();
    const report=a.reportText(a.current(),r);
    assert.ok(report.includes(`Simulation paths: ${completedCount}; fixed comparison sequence`));
    assert.ok(report.includes(`  Simulation paths: ${completedCount}\n`));
    assert.ok(report.includes(`Paths for next run: ${nextCount}`));
    const html=a.lab(),lab=labRows(a)[0];
    assert.equal(lab.result.provenance.simulationCount,completedCount,'A completed comparison keeps its actual count');
    assert.ok(html.includes(completedCount<=100?'Sample lifetimes without a shortfall':'Readiness'));
    assert.equal(html.includes('Sample preview only'),completedCount<=100);
    assert.equal(a.state.results.get(a.current().id),r);
  }
});

test('a failed access check on window focus keeps results when the tier is unchanged',async()=>{
  const a=app(),pending=a.run();a.workers[0].onmessage({data:{type:'result',result:runSimulation(a.workers[0].data.scenario)}});await pending;
  await a.loadAccess();assert.equal(a.state.results.size,1);assert.doesNotMatch(a.state.message,/Subscription status is unavailable/,'an anonymous visitor already uses the free preview');
  const renders=a.renders();await a.loadAccess();assert.equal(a.state.results.size,1);assert.equal(a.renders(),renders,'an unchanged check does not redraw open charts');
  const known=app(null,{session:new Map([['retirement-billing-reference',JSON.stringify({customerId:'cus_123'})]])});
  await known.loadAccess();assert.match(known.state.message,/Subscription status is unavailable/,'a known billing customer is told why runs use the preview');
});
test('an access check that changes the tier preserves completed results and updates future runs',async()=>{
  const a=app(null,{fetch:async()=>({ok:true,json:async()=>({tier:'pro',maxPaths:10000,signedIn:true,checkoutAvailable:true,accountProvider:'chatgpt'})})});
  const result=runSimulation(a.current()),pending=a.run();a.workers[0].onmessage({data:{type:'result',result}});await pending;
  await a.loadAccess();assert.equal(a.state.access.tier,'pro');assert.equal(a.state.results.size,1);
  const pro=a.run();assert.equal(a.workers[1].data.scenario.numberOfSimulations,10000);a.workers[1].onmessage({data:{type:'result',result}});await pro;
  const renders=a.renders();await a.loadAccess();assert.equal(a.state.results.size,1);assert.equal(a.renders(),renders);
});
test('spending targets at the search limit are shown as a lower bound',()=>{
  const a=app();a.state.access.tier='pro';
  a.state.decision={targetReadiness:.8,simulationCount:200,earliestRetirementAge:55,safeAnnualSpending:250000,safeSpendingAtSearchLimit:true,safeSpendingSearchLimit:250000};
  assert.match(a.lab(),/At least \$250,000/);assert.match(a.lab(),/search stops at \$250,000/);
  a.state.decision={...a.state.decision,safeAnnualSpending:90000,safeSpendingAtSearchLimit:false};
  assert.doesNotMatch(a.lab(),/At least|search stops/);assert.match(a.lab(),/\$90,000/);
});

test('transient billing failures retain verified Pro briefly and never discard results',async()=>{
  for(const failure of ['network','502']){
    const clock={now:1000};let failing=false;
    const a=app(null,{clock,fetch:async()=>{if(failing){if(failure==='network')throw Error('offline');return Response.json({accountKey:'user-a',error:'Temporary outage'},{status:502});}return Response.json({tier:'pro',signedIn:true,accountKey:'user-a',maxPaths:10000});}});
    await a.loadAccess();a.state.results.set(a.current().id,runSimulation({...a.current(),numberOfSimulations:4}));seedExploration(a);failing=true;
    await a.loadAccess();assert.equal(a.isPro(),true);assert.equal(a.state.results.size,1);assert.ok(a.state.decision);assert.match(a.state.message,/last verified Pro/);
    clock.now+=5*60*1000;assert.equal(a.isPro(),false);assert.equal(a.effectivePaths(),100);
    await a.loadAccess();assert.equal(a.state.access.tier,'free');assert.equal(a.state.results.size,1);assert.ok(a.state.decision);
    failing=false;await a.loadAccess();assert.equal(a.isPro(),true);assert.equal(a.state.message,'');
  }
});

test('confirmed expiration, sign-out and authentication failure end Pro without deleting results',async()=>{
  for(const reply of [{tier:'free',signedIn:true,accountKey:'user-a'},{tier:'free',signedIn:false},{error:'Sign in again',status:401}]){
    let next={tier:'pro',signedIn:true,accountKey:'user-a'};
    const a=app(null,{fetch:async()=>Response.json(next,{status:next.status||200})});await a.loadAccess();
    a.state.results.set(a.current().id,runSimulation({...a.current(),numberOfSimulations:4}));seedExploration(a);next=reply;await a.loadAccess();
    assert.equal(a.isPro(),false);assert.equal(a.state.results.size,1);assert.ok(a.state.decision);
  }
});

test('a different identity cannot inherit grace after its billing check fails',async()=>{
  let next={tier:'pro',signedIn:true,accountKey:'firebase:a'};
  const identity={accountKey:'firebase:a'},a=app(null,{identity,fetch:async()=>Response.json(next,{status:next.status||200})});
  a.syncAuthState();await a.loadAccess();identity.accountKey='firebase:b';a.syncAuthState();
  next={accountKey:'firebase:b',status:502};await a.loadAccess();assert.equal(a.isPro(),false);
});

test('server identity changes and stale overlapping responses cannot restore another account access',async()=>{
  let next={tier:'pro',signedIn:true,accountKey:'a',billingCustomerId:'cus_old'};const session=new Map();
  const a=app(null,{session,fetch:async()=>Response.json(next,{status:next.status||200})});await a.loadAccess();
  next={accountKey:'b',status:502};await a.loadAccess();assert.equal(a.isPro(),false);assert.equal(a.state.access.accountKey,'b');assert.equal(a.state.access.signedIn,true);
  assert.equal(session.get('retirement-billing-reference'),'{}');await a.loadAccess();assert.equal(a.state.access.accountKey,'b');assert.doesNotMatch(a.state.message,/Sign in again/);
  const resolvers=[],b=app(null,{fetch:()=>new Promise(resolve=>resolvers.push(resolve))});
  const old=b.loadAccess(),recent=b.loadAccess();await new Promise(setImmediate);
  resolvers[1](Response.json({tier:'free',signedIn:false}));await recent;
  resolvers[0](Response.json({tier:'pro',signedIn:true,accountKey:'old'}));await old;assert.equal(b.isPro(),false);
});

test('verified checkout customer reference survives URL cleanup and reload',async()=>{
  const session=new Map(),location={search:'?checkout=success&session_id=cs_live_12345678&keep=1',pathname:'/',hash:''},requests=[];
  const fetch=async(url,options)=>{requests.push({url,options});return Response.json({tier:'pro',signedIn:true,accountKey:'a',billingCustomerId:'cus_123',billingPortalAvailable:true});};
  const a=app(null,{session,location,fetch});await a.loadAccess();assert.match(requests[0].url,/session_id=cs_live_12345678/);assert.equal(location.search,'?keep=1');
  await a.loadAccess();assert.equal(requests[1].options.headers['x-retirement-customer'],'cus_123');assert.doesNotMatch(requests[1].url,/session_id/);
  const b=app(null,{session,fetch});await b.loadAccess();assert.equal(requests[2].options.headers['x-retirement-customer'],'cus_123');
});

test('account-link query cleanup works without URLSearchParams.size',async()=>{
  class OlderParams extends URLSearchParams{get size(){return undefined;}}
  const location={search:'?link=google&keep=1',pathname:'/',hash:'#billing'};
  const a=app(null,{location,params:OlderParams,fetch:async()=>Response.json({linked:true,billingCustomerId:'cus_linked'})});await a.linkAccounts();
  // The legacy #billing link opens billing and is kept as its route.
  assert.equal(location.search,'?keep=1');assert.equal(location.hash,'#/billing');assert.equal(a.state.view,'billing');
});

test('initial Stripe outages retain authenticated identity and the stored customer reference',async()=>{
  const session=new Map([['retirement-billing-reference',JSON.stringify({customerId:'cus_123'})]]),requests=[];
  const a=app(null,{session,fetch:async(_url,options)=>{requests.push(options);return Response.json({signedIn:true,accountKey:'a',accountProvider:'chatgpt',error:'Stripe unavailable'},{status:502});}});
  for(let i=0;i<3;i++){
    assert.equal(await a.loadAccess(),false);assert.equal(a.state.access.signedIn,true);assert.equal(a.state.access.accountKey,'a');
    assert.equal(JSON.parse(session.get('retirement-billing-reference')).customerId,'cus_123');
    assert.match(a.billingView(),/Signed in with ChatGPT/);assert.doesNotMatch(a.state.message,/Sign in again/);assert.equal(a.isPro(),false);
  }
  assert.ok(requests.every(r=>r.headers['x-retirement-customer']==='cus_123'));
});

test('identity-service 503 preserves verified Google access only for the remaining grace period',async()=>{
  let failing=false;const session=new Map(),clock={now:1000};
  const fetch=async()=>failing?Response.json({error:'Identity verification is temporarily unavailable. Please retry.'},{status:503}):Response.json({tier:'pro',signedIn:true,accountKey:'firebase:a',accountProvider:'google',billingCustomerId:'cus_paid'});
  const a=app(null,{fetch,session,clock});await a.loadAccess();
  const result=runSimulation({...a.current(),numberOfSimulations:4});a.state.results.set(a.current().id,result);
  failing=true;a.advanceTime(60000);
  for(let i=0;i<2;i++){
    await a.loadAccess();assert.equal(a.isPro(),true);assert.equal(a.state.access.signedIn,true);assert.equal(a.state.access.accountKey,'firebase:a');
    assert.doesNotMatch(a.state.message,/Sign in again/);assert.equal(JSON.parse(session.get('retirement-billing-reference')).customerId,'cus_paid');
  }
  a.advanceTime(240000);assert.equal(a.isPro(),false);assert.equal(a.state.results.get(a.current().id),result);
  await a.loadAccess();assert.equal(a.state.access.accountKey,'firebase:a');assert.equal(JSON.parse(session.get('retirement-billing-reference')).customerId,'cus_paid');
  failing=false;await a.loadAccess();assert.equal(a.isPro(),true);assert.equal(a.state.message,'');
  // A new page has no confirmed entitlement; the same outage cannot grant Pro.
  failing=true;const fresh=app(null,{fetch,session});await fresh.loadAccess();assert.equal(fresh.isPro(),false);
  assert.equal(fresh.state.access.accountKey,undefined);assert.equal(JSON.parse(session.get('retirement-billing-reference')).customerId,'cus_paid');
});

test('repeated outages preserve known identity for Free and expired Pro access',async()=>{
  for(const tier of ['free','pro'])for(const failure of ['network','502']){
    let failing=false;const clock={now:1000},session=new Map();
    const a=app(null,{clock,session,fetch:async()=>{
      if(!failing)return Response.json({tier,signedIn:true,accountKey:'a',accountProvider:'google',billingCustomerId:'cus_123'});
      if(failure==='network')throw Error('offline');
      return Response.json({signedIn:true,accountKey:'a',accountProvider:'google'},{status:502});
    }});
    await a.loadAccess();failing=true;clock.now+=300001;
    for(let i=0;i<3;i++){
      await a.loadAccess();assert.equal(a.state.access.accountKey,'a');assert.equal(a.state.access.signedIn,true);assert.equal(a.state.access.accountProvider,'google');
      assert.equal(a.isPro(),false);assert.doesNotMatch(a.state.message,/Sign in again/);assert.equal(JSON.parse(session.get('retirement-billing-reference')).customerId,'cus_123');
    }
  }
});

test('grace expiration redraws permissions without extending grace or interrupting results',async()=>{
  for(const running of [false,true])for(const ownerAccess of [false,true]){
    let failing=false;const clock={now:1000};
    const a=app(null,{clock,fetch:async()=>failing?Response.json({accountKey:'a'},{status:502}):Response.json({tier:'pro',signedIn:true,accountKey:'a',ownerAccess})});
    a.state.view='billing';await a.loadAccess();const result=runSimulation({...a.current(),numberOfSimulations:4});a.state.results.set(a.current().id,result);seedExploration(a);
    failing=true;await a.loadAccess();a.advanceTime(60000);await a.loadAccess();assert.equal(a.timers.size,1);
    const pending=running?a.run():null,renders=a.renders();a.advanceTime(240000);
    assert.equal(a.renders(),renders+1);assert.equal(a.timers.size,0);assert.equal(a.effectivePaths(),100);
    assert.doesNotMatch(a.element('#main').innerHTML,/Active on this account|Owner Pro access is active|last verified Pro access is available/);
    assert.match(a.state.message,running?/Calculating this plan/:/New runs use the 100-path/);assert.equal(a.state.results.get(a.current().id),result);assert.ok(a.state.decision);
    if(running){assert.equal(a.workers[0].terminated,undefined);a.workers[0].onmessage({data:{type:'result',result}});await pending;}
    assert.equal(a.state.results.get(a.current().id),result);
  }
});

test('recovery and authentication changes cancel an obsolete grace timer',async()=>{
  for(const next of [{tier:'pro',signedIn:true,accountKey:'a'},{tier:'free',signedIn:false},{status:401}]){
    let reply={tier:'pro',signedIn:true,accountKey:'a'};const a=app(null,{clock:{now:1000},fetch:async()=>reply.status===401?new Response('Unauthorized',{status:401}):Response.json(reply,{status:reply.status||200})});
    await a.loadAccess();reply={accountKey:'a',status:502};await a.loadAccess();assert.equal(a.timers.size,1);
    reply=next;await a.loadAccess();assert.equal(a.timers.size,0);const renders=a.renders();a.advanceTime(300001);
    assert.equal(a.renders(),renders);assert.equal(a.isPro(),next.tier==='pro');
  }
});

test('payment pending becomes confirmed after checkout parameters are removed',async()=>{
  let tier='free';const location={search:'?checkout=success&session_id=cs_live_12345678&keep=1',pathname:'/',hash:''};
  const a=app(null,{location,fetch:async()=>Response.json({tier,signedIn:true,accountKey:'a',billingCustomerId:'cus_123'})});
  await a.loadAccess();assert.equal(location.search,'?keep=1');assert.match(a.state.message,/Payment is being confirmed/);
  await a.loadAccess();assert.match(a.state.message,/Payment is being confirmed/);
  tier='pro';await a.loadAccess();assert.equal(a.isPro(),true);assert.equal(a.state.message,'Pro is active for this account.');
});

test('linking uses the chosen customer and supersedes checks started before linking',async()=>{
  let resolveOld,first=true;const requests=[],session=new Map(),identity={accountKey:'firebase:a'};
  const a=app(null,{identity,session,fetch:async(url,options)=>{
    requests.push({url,options});
    if(url==='/api/billing/link')return Response.json({linked:true,billingCustomerId:'cus_linked'});
    if(first){first=false;return new Promise(resolve=>{resolveOld=resolve;});}
    return Response.json({tier:'pro',signedIn:true,accountKey:'firebase:a',billingCustomerId:'cus_linked'});
  }});
  const old=a.loadAccess();await new Promise(setImmediate);await a.linkAccounts();a.syncAuthState();
  resolveOld(Response.json({tier:'free',signedIn:true,accountKey:'firebase:a',billingCustomerId:'cus_old'}));await old;
  assert.equal(JSON.parse(session.get('retirement-billing-reference')).customerId,'cus_linked');
  await a.loadAccess();assert.equal(requests.at(-1).options.headers['x-retirement-customer'],'cus_linked');assert.equal(a.isPro(),true);
});

test('link confirmation does not overwrite a subsequent billing outage notice',async()=>{
  const a=app(null,{fetch:async url=>url==='/api/billing/link'?Response.json({linked:true,billingCustomerId:'cus_linked'}):Response.json({signedIn:true,accountKey:'a'},{status:502})});
  await a.click('social-link-accounts');assert.match(a.state.message,/Subscription status is unavailable/);assert.doesNotMatch(a.state.message,/Sign-in methods linked/);
});

test('account changes during linking cannot install the earlier account customer reference',async()=>{
  let finish;const identity={accountKey:'firebase:a'},session=new Map();
  const a=app(null,{identity,session,fetch:()=>new Promise(resolve=>{finish=resolve;})});
  const pending=a.linkAccounts();await new Promise(setImmediate);identity.accountKey='firebase:b';a.syncAuthState();
  finish(Response.json({linked:true,billingCustomerId:'cus_old'}));await assert.rejects(pending,/signed-in account changed/);
  assert.equal(session.get('retirement-billing-reference'),'{}');assert.equal(a.isPro(),false);
});

test('welcome illustration is fixed decoration and never reflects the visitor plan',()=>{
  const poor=model.baseScenario();poor.accounts={pretax:1,roth:0,taxable:0,cash:0};
  const fan=html=>html.match(/<svg viewBox="0 0 420 248"[\s\S]*?<\/svg>/)[0];
  const sample=fan(app().dashboard()),own=fan(app({scenarios:[poor],selectedId:poor.id}).dashboard());
  assert.equal(own,sample);assert.equal((sample.match(/class="fan-path /g)||[]).length,30);
  assert.match(sample,/not a simulation or a forecast of your plan/);
});


test('pension and care editors retain 0-11 month selectors and save months in backups and reports',async()=>{
  const a=app();
  for(const [section,years,months] of [[2,'guaranteedIncome.startAge','guaranteedIncome.startAgeMonths'],[3,'longTermCare.averageDurationYears','longTermCare.averageDurationMonths']]){
    a.state.setupSection=section;const html=a.setup();
    assert.match(html,new RegExp(`data-field="${years}" data-type="number"`));
    assert.match(html,new RegExp(`<select[^>]+data-field="${months}" data-type="month">`));
    assert.match(html,/<option value="0" selected>0<\/option>/);assert.match(html,/<option value="11" >11<\/option>/);
    await a.change('#main',{dataset:{field:months,type:'month'},value:'6'});
  }
  const s=a.saved().scenarios[0];assert.equal(s.household.retirementAgeMonths,0);assert.equal(s.guaranteedIncome.startAgeMonths,6);assert.equal(s.longTermCare.averageDurationMonths,6);
  assert.match(a.reportText(a.current()),/Your pension start age extra months: 6/);assert.match(a.reportText(a.current()),/Long-term care duration \/ years extra months: 6/);
});

test('changing retirement date discards pending calculations and exploration results',async()=>{
  const a=app();seedExploration(a);const pending=a.run();
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:model.addCalendarMonths(a.current().household.retirementDate,6)});
  a.workers[0].onmessage({data:{type:'result',result:runSimulation(a.workers[0].data.scenario)}});await pending;
  assert.equal(a.state.results.size,0);assertCleared(a);
});

test('explicit linking retains paid-customer hints across Google sign-in and unpaid status checks',async()=>{
  const identity={accountKey:null},session=new Map(),requests=[];let reply={tier:'pro',signedIn:true,accountKey:'chatgpt-a',billingCustomerId:'cus_paid'};
  const fetch=async(url,options)=>{requests.push({url,options});return Response.json(url==='/api/billing/link'?{linked:true,billingCustomerId:'cus_paid'}:reply);};
  const a=app(null,{identity,session,fetch});a.syncAuthState();await a.loadAccess();
  identity.accountKey='firebase:a';a.syncAuthState();reply={tier:'free',signedIn:true,accountKey:'firebase:a',billingCustomerId:'cus_draft'};await a.loadAccess();
  await a.linkAccounts();const headers=requests.find(r=>r.url==='/api/billing/link').options.headers;
  assert.equal(headers['x-retirement-customer'],'cus_draft');assert.match(headers['x-retirement-link-customers'],/cus_paid/);
  // The lookup hints also survive a sign-in redirect/reload of the page.
  const b=app(null,{identity,session,fetch});await b.linkAccounts();assert.match(requests.at(-1).options.headers['x-retirement-link-customers'],/cus_paid/);
});

test('a running calculation can be canceled without losing completed results',async()=>{
  for(const task of ['run','runDecision','runLab']){
    const a=app();a.state.access.tier='pro';a.current().numberOfSimulations=4;
    const earlier=runSimulation({...a.current(),numberOfSimulations:4});a.state.results.set(a.current().id,earlier);
    const pending=a[task]();assert.equal(a.state.busy,true);
    assert.match(a.element('#main').innerHTML,/data-action="cancel-calculation"/);
    await a.click('cancel-calculation');await pending;
    assert.equal(a.state.busy,false);assert.equal(a.workers.at(-1).terminated,true);assert.match(a.state.message,/canceled/);
    assert.equal(a.state.results.get(a.current().id),earlier);assert.doesNotMatch(a.element('#main').innerHTML,/cancel-calculation/);
    // Nothing stays blocked: another calculation starts and finishes normally.
    const next=a.run();a.workers.at(-1).onmessage({data:{type:'result',result:earlier}});await next;assert.equal(a.state.busy,false);
  }
});

test('worker progress updates the running status without finishing the calculation',async()=>{
  const a=app(),pending=a.run(),worker=a.workers[0];
  worker.onmessage({data:{type:'progress',fraction:.5}});
  assert.equal(a.state.busy,true);assert.equal(a.element('#busy-progress').value,.5);
  assert.match(a.element('#busy-detail').textContent,/50 of 100 lifetimes simulated/);assert.match(a.element('#result-state').textContent,/50%/);
  worker.onmessage({data:{type:'result',result:runSimulation(worker.data.scenario)}});await pending;
  assert.equal(a.state.busy,false);assert.equal(a.state.progress,null);assert.equal(a.state.results.size,1);
  const b=app();b.state.access.tier='pro';const search=b.runDecision();
  b.workers[0].onmessage({data:{type:'progress',phase:'spending',checkedAges:3,totalAges:7,checkedAmounts:10,totalAmounts:501}});
  assert.match(b.element('#busy-detail').textContent,/10 of up to 501 spending amounts/);assert.equal(b.element('#busy-progress').value,17/508);
  b.workers[0].onmessage({data:{type:'result',result:{targetReadiness:.8,simulationCount:200}}});await search;assert.equal(b.state.busy,false);
});

test('a comparison that already matches the plan is reported instead of rerun',async()=>{
  const s=model.baseScenario();Object.assign(s.withdrawalStrategy,{useCashReserveDuringDrawdowns:true,drawdownTrigger:-.01});
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='lab';
  assert.doesNotMatch(a.lab(),/data-preset="cash"/,'A preset the plan already uses is not offered');
  await a.click('lab-add');await setLever(a,'cashFirst',{trigger:'-0.01'});
  const pending=a.runLab();await settle(a,pending);
  assert.equal(a.workers.length,1);const row=labRows(a)[1];assert.equal(row.label,'Cash first in months below −1%');assert.equal(row.result,null);
  assert.match(a.lab(),/Already matches your current plan/);
});

test('warnings and progress use the neutral notice style; completed actions use success styling',()=>{
  const a=app();
  for(const [message,className] of [
    ['Subscription status is unavailable. New runs use the 100-path free preview; your completed results are kept.','notice'],
    ['Sign in again to verify your plan. Your completed results are still available.','notice'],
    ['Planning targets require Pro because 100 paths are too coarse.','notice'],
    ['Calculation canceled. Completed results are unchanged.','notice'],
    ['Results updated for Base plan.','notice good'],
    ['Error: Something failed.','notice error'],
  ]){a.state.message=message;assert.match(a.dashboard(),new RegExp(`<div class="${className}" role="status">`),message);}
});

test('planning targets report the whole-year ages actually searched',()=>{
  const a=app();a.state.access.tier='pro';
  a.state.decision={targetReadiness:.8,simulationCount:200,earliestRetirementAge:null,safeAnnualSpending:null,safeSpendingAtSearchLimit:false,safeSpendingSearchLimit:250000,retirementAgeSearchStart:60,retirementAgeSearchEnd:67};
  assert.match(a.lab(),/ages 60 through 67 with 200 paths per age/);assert.doesNotMatch(a.lab(),/through 70/);
  a.state.decision={...a.state.decision,retirementAgeSearchStart:72,retirementAgeSearchEnd:70};
  assert.match(a.lab(),/No whole-year retirement age before the maximum modeling age/);
});

test('property tax and home insurance are a Housing input that typing spending never changes',async()=>{
  const a=app();a.state.view='setup';a.state.setupSection=3;
  const html=a.setup();assert.match(html,/data-field="home\.annualTaxesAndInsurance" data-type="money"/);
  assert.match(html,/data-field="rent\.monthlyRent"/);assert.match(html,/data-field="longTermCare\.annualCost"/);
  await a.change('#main',{dataset:{field:'home.annualTaxesAndInsurance',type:'money'},value:'8000'});
  await a.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:'70000'});
  const saved=a.saved().scenarios[0];assert.equal(saved.home.annualTaxesAndInsurance,8000);assert.equal(saved.spending.annualBaseSpending,70000);
  assert.match(a.reportText(a.current()),/Property tax & home insurance \/ year: \$8,000\.00/);
});


test('household setup uses calendar fields, preserves saved dates, and includes them in reports',async()=>{
  const a=app(),html=a.setup();
  assert.match(html,/type="date"[^>]+data-field="household.birthday"/);
  assert.match(html,/type="date"[^>]+data-field="household.retirementDate"/);
  assert.doesNotMatch(html,/data-field="household.(?:currentAge|retirementAge|retirementAgeMonths|spouseBirthday)"/);
  await a.change('#main',{dataset:{field:'household.birthday',type:'date'},value:'1965-01-31'});
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:'2033-07-20'});
  await a.change('#main',{dataset:{field:'household.filingStatus',type:'select'},value:'Married'});
  assert.match(a.setup(),/type="date"[^>]+data-field="household.spouseBirthday"/);
  await a.change('#main',{dataset:{field:'household.spouseBirthday',type:'date'},value:'1968-03-02'});
  const restored=app(a.saved());
  assert.equal(restored.current().household.birthday,'1965-01-31');
  assert.equal(restored.current().household.retirementDate,'2033-07-20');
  assert.equal(restored.current().household.spouseBirthday,'1968-03-02');
  assert.match(a.reportText(a.current()),/Retirement date: 2033-07-20/);
  assert.match(a.reportText(a.current()),/Spouse birthday: 1968-03-02/);
  assert.match(a.dashboard(),/Jul 20, 2033/);
});

test('incomplete or impossible calendar edits preserve the previous saved assumption',async()=>{
  const a=app(),before=a.current().household.birthday;
  for(const value of ['', '2026-02-29', '2999-01-01']){
    await a.change('#main',{dataset:{field:'household.birthday',type:'date'},value});
    assert.equal(a.current().household.birthday,before);assert.match(a.state.message,/Error:.*birthday/);
  }
  const retirement=a.current().household.retirementDate;
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:'2000-01-01'});
  assert.equal(a.current().household.retirementDate,retirement);
});

test('inferred legacy dates show a review notice until explicitly acknowledged',async()=>{
  const old=model.baseScenario(),a=app({scenarios:[old],selectedId:old.id});
  assert.match(a.setup(),/Dates were estimated from your saved ages/);
  assert.equal(a.current().accounts.pretax,old.accounts.pretax);
  await a.click('review-calendar-dates');
  assert.equal(a.saved().scenarios[0].household.datesNeedReview,false);
  assert.doesNotMatch(app(a.saved()).setup(),/Dates were estimated from your saved ages/);
});

test('correcting a partly completed date draft never saves nonfinite legacy age values',async()=>{
  const s=model.baseScenario();s.household.birthday='1965-01-31';s.household.retirementDate='';
  const a=app({scenarios:[s],selectedId:s.id});
  await a.change('#main',{dataset:{field:'household.birthday',type:'date'},value:'1965-02-01'});
  assert.equal(typeof a.saved().scenarios[0].household.retirementAge,'number');
  assert.doesNotMatch(app(a.saved()).state.message,/could not be loaded/);
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:'2033-07-20'});
  assert.deepEqual(model.validateScenario(a.current()),[]);
});

test('withdrawal tab explains current choices before a run and links directly to settings',async()=>{
  const a=app();a.element('#navigation').listeners.click({target:{closest:()=>({dataset:{view:'withdrawals'}})}});
  assert.equal(a.element('#page-location').textContent,'How withdrawals work');
  assert.match(a.element('#main').innerHTML,/Your withdrawal assumptions/);
  assert.match(a.withdrawals(),/Run the selected plan to see an actual month/);
  assert.match(a.withdrawals(),/Pre-tax → Roth → taxable → cash/);
  a.current().withdrawalStrategy.applyEarlyWithdrawalPenalty=false;
  a.current().withdrawalStrategy.useCashReserveDuringDrawdowns=true;
  a.current().withdrawalStrategy.drawdownTrigger=-.025;
  a.current().rothConversion.enabled=true;
  const html=a.withdrawals();assert.match(html,/penalties are omitted; federal income taxes still apply/);assert.match(html,/below -2.5%/);assert.match(html,/federal bracket cap/);
  await a.click('withdrawal-settings',{index:'4'});assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,4);assert.equal(a.stored(),null);
});

test('withdrawal examples follow the selected month and plan without changing saved inputs or results',async()=>{
  const a=app(),s=a.current(),r=runSimulation(s);a.state.results.set(s.id,r);a.state.view='withdrawals';
  const before=JSON.stringify({s,r});const html=a.withdrawals();
  assert.match(html,/id="withdrawal-example-month"/);assert.match(html,/Month 1 ·/);assert.match(html,/Additional withdrawal needed/);
  const p=r.steadySimulation.monthlyDetails[12];
  await a.change('#main',{id:'withdrawal-example-month',value:'12'});
  assert.match(a.element('#withdrawal-example').innerHTML,/Month 12 ·/);
  assert.ok(a.element('#withdrawal-example').innerHTML.includes(new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(p.cashFlow.expenses)));
  assert.equal(JSON.stringify({s,r}),before);assert.equal(a.stored(),null);
  a.state.selectedId=a.state.scenarios[1].id;assert.match(a.withdrawals(),/Run the selected plan to see an actual month/);
});

test('withdrawal page handles a shortfall and retirement beyond the assumed lifespan',()=>{
  const a=app(),s=a.current();s.accounts={pretax:0,roth:0,taxable:0,cash:0};s.home.currentValue=0;s.spending.annualBaseSpending=1e6;
  a.state.results.set(s.id,runSimulation(s));assert.match(a.withdrawals(),/ends with an unfunded amount/);
  s.household.retirementDate=model.addCalendarMonths(s.household.birthday,96*12);a.state.results.set(s.id,runSimulation(s));
  assert.match(a.withdrawals(),/no retirement month to illustrate/);assert.doesNotMatch(a.withdrawals(),/id="withdrawal-example-month"/);
});

test('free runs use 100 paths for legacy and imported plans while preserving saved choices',async()=>{
  for(const savedCount of [4,10,100,5000]){
    const s=model.baseScenario();s.numberOfSimulations=savedCount;s.simulationPathsCustomized=savedCount!==4;
    const a=app({scenarios:[s],selectedId:s.id});
    assert.equal(a.effectivePaths(),100);assert.match(a.billingView(),/>100 paths</);
    a.state.setupSection=4;assert.match(a.setup(),/100 paths · Free preview/);
    const saved=a.stored(),pending=a.run();assert.equal(a.workers[0].data.scenario.numberOfSimulations,100);
    const result=runSimulation(a.workers[0].data.scenario);assert.equal(result.provenance.simulationCount,100);
    a.workers[0].onmessage({data:{type:'result',result}});await pending;
    assert.equal(a.stored(),saved);assert.equal(a.current().numberOfSimulations,savedCount);
    await a.change('#main',{dataset:{field:'numberOfSimulations',type:'number'},value:'1000'});
    assert.match(a.state.message,/100 paths are available/);assert.equal(a.current().numberOfSimulations,savedCount);
    a.state.access.tier='pro';assert.equal(a.effectivePaths(),savedCount,'The free allowance must not override a Pro selection');
  }
});

test('100 paths remain a counted preview and 101 paths use percentage formatting',()=>{
  for(const count of [4,10,100,101]){
    const result={provenance:{simulationCount:count},successProbability:.8};
    assert.equal(format.isPreviewResult(result),count<=100);
    assert.equal(format.readinessLabel(result),count<=100?`${Math.round(.8*count)} of ${count}`:'80.0%');
  }
});

test('guided setup offers household choices, progress, a skippable editor and editable review without changing assumptions',async()=>{
  const a=app(),expected=structuredClone(a.current());expected.name='My retirement plan';const original=JSON.stringify(expected);
  await a.click('start-plan');assert.equal(a.state.guided,true);
  assert.match(a.setup(),/Step 1 of 6/);assert.match(a.setup(),/Individual · Just me/);assert.match(a.setup(),/Couple · Me and my spouse/);
  await a.click('setup-section',{index:'1'});assert.match(a.setup(),/guided-extra/);assert.match(a.setup(),/annual future savings deposits/);
  await a.click('setup-section',{index:'5'});assert.match(a.setup(),/Review before running/);assert.match(a.setup(),/Sample\/default/);assert.match(a.setup(),/Edit Household/);
  await a.click('toggle-guided');assert.equal(a.state.guided,false);assert.equal(a.state.setupSection,0);
  assert.equal(JSON.stringify(a.current()),original);
  await a.click('household-choice',{kind:'couple'});
  assert.equal(a.current().household.filingStatus,'Married');assert.match(a.setup(),/<h3>Spouse<\/h3>/);
  a.state.setupSection=2;assert.match(a.setup(),/Your pension or annuity \/ year/);assert.match(a.setup(),/One stream with your start age/);assert.match(a.setup(),/Enter their own age-67 benefit/);
  await a.click('household-choice',{kind:'individual'});a.state.setupSection=0;
  assert.equal(a.current().household.filingStatus,'Single');assert.doesNotMatch(a.setup(),/<h3>Spouse<\/h3>/);
});

test('guided setup puts each person’s inputs together and keeps optional detail out of the way',async()=>{
  const a=app();await a.click('start-plan');await a.click('household-choice',{kind:'couple'});const s=a.current();
  let html=a.setup();assert.doesNotMatch(html,/section-picker/);assert.match(html,/class="setup-steps"/);assert.doesNotMatch(html,/Separate-person model/);
  assert.ok(html.indexOf('household-choice')<html.indexOf('f-household-birthday'));
  assert.ok(html.indexOf('f-household-spouseRetirementDate')<html.indexOf('f-household-spouseGender'),'Spouse dates come before their longevity table');
  await a.click('setup-section',{index:'1'});html=a.setup();
  const firstExtra=html.indexOf('guided-extra');
  for(const id of ['f-accounts-pretax','f-accounts-roth','f-spouseAccounts-pretax','f-spouseAccounts-roth','f-accounts-taxable'])assert.ok(html.indexOf(id)>0&&html.indexOf(id)<firstExtra,id+' stays visible');
  assert.ok(html.indexOf('f-spouseRothHistory-contributionBasis')>firstExtra,'Roth records are optional detail');
  assert.ok(html.indexOf('How earnings, savings and growth fit together')<html.indexOf('Are you still adding to your savings?'));
  await a.click('setup-section',{index:'2'});html=a.setup();
  assert.ok(html.indexOf('f-spouseIncome-annualBenefitAt67')<html.indexOf('f-socialSecurity-spouseClaimAge'));
  assert.match(html,/Working household support · only if one of you works after the other retires/);
  s.household.spouseRetirementDate=model.addCalendarMonths(s.household.retirementDate,24);
  assert.doesNotMatch(a.setup(),/only if one of you works after the other retires/);
  await a.click('toggle-guided');assert.match(a.setup(),/section-picker/);
});

test('one-year statement check separates new savings, transfers and investment growth',()=>{
  const values={start:'400000',end:'470000',yours:'18000',employer:'6000',transfersIn:'0',out:'0'};
  const r=growthHelper.oneYearGrowth(values);
  assert.equal(r.savings,24000);assert.equal(r.growth,46000);assert.equal(r.rate,46000/412000);
  const moved=growthHelper.oneYearGrowth({...values,transfersIn:'50000',out:'10000'});
  assert.equal(moved.savings,24000,'Rollovers are not new savings');assert.equal(moved.net,64000);assert.equal(moved.growth,6000);
  assert.equal(growthHelper.oneYearGrowth({...values,out:''}).complete,false,'A missing cash flow is unknown, not zero');
  assert.match(growthHelper.oneYearGrowth({...values,yours:'-5'}).error,/\$0 or more/);
  const roth=growthHelper.oneYearGrowth({...values,employer:''},{employer:false});assert.equal(roth.complete,true);assert.equal(roth.savings,18000);
  assert.equal(growthHelper.oneYearGrowth({...values,start:'0',yours:'0',employer:'0',end:'0'}).rate,null);
});

test('statement check applies only contributions, marked Estimated, and never the return',async()=>{
  const a=app(),s=a.current(),returnBefore=s.market.preRetirementMeanReturn;a.state.setupSection=1;
  assert.match(a.setup(),/Check last year’s statements/);assert.doesNotMatch(a.setup(),/id="growth-owner"/,'Individuals have no owner choice');
  const type=(key,value)=>a.element('#main').listeners.input({target:{dataset:{growthHelper:key},value}});
  for(const [key,value] of Object.entries({start:'400000',end:'470000',yours:'18000',employer:'6000',transfersIn:'50000'}))type(key,value);
  assert.match(a.element('#growth-helper-result').innerHTML,/Enter every amount/);
  type('out','0');const html=a.element('#growth-helper-result').innerHTML;
  assert.match(html,/Investment growth after fees.*−\$4,000\.00/s);assert.match(html,/not applied to your plan/);
  await a.click('apply-growth-savings');
  assert.equal(s.contributions.pretax,18000);assert.equal(s.contributions.employerPretax,6000);assert.equal(s.market.preRetirementMeanReturn,returnBefore);
  assert.equal(a.state.inputSources[s.id]['contributions.pretax'],'Estimated');assert.equal(a.state.inputSources[s.id]['contributions.employerPretax'],'Estimated');
  await a.click('household-choice',{kind:'couple'});a.state.setupSection=1;assert.match(a.setup(),/id="growth-owner"/);
  type('owner','spouseContributions');type('account','roth');await a.click('apply-growth-savings');
  assert.equal(s.spouseContributions.roth,18000);assert.equal(s.spouseContributions.employerPretax,0,'Employer amounts apply only to pre-tax accounts');
  assert.doesNotMatch(a.stored(),/470000/,'Statement balances are not saved with the plan');
});

test('review leads with a short summary and names each Unknown input',async()=>{
  const a=app();await a.click('start-plan');
  await a.change('#main',{dataset:{field:'guaranteedIncome.annualIncome',type:'money'},value:''});
  await a.click('setup-section',{index:'5'});const html=a.setup();
  assert.match(html,/At a glance/);assert.match(html,/Savings today/);assert.match(html,/Includes sample values/);
  assert.match(html,/<li>Your pension or annuity \/ year · <button[^>]*data-index="2"/);
  assert.match(html,/<details class="card review-all">/);assert.match(html,/Past Roth conversions/);
  assert.doesNotMatch(html,/id="f-accounts-roth"/,'Review summarizes rather than embedding stray editors');
});

test('individual review ignores unused spouse sources and does not label household choice as a sample amount',()=>{
  for(const filingStatus of ['Single','HeadOfHousehold']){
    const a=app(),s=a.current();s.household.filingStatus=filingStatus;a.state.setupSection=5;
    a.state.inputSources[s.id]={_origin:'Entered','household.filingStatus':'Sample/default','household.spouseBirthday':'Sample/default','household.spouseGender':'Unknown','household.spouseRetirementDate':'Sample/default'};
    const html=a.setup();
    assert.match(html,/<dt>Household<\/dt><dd>Individual<small>Selected/);
    assert.doesNotMatch(html,/Sample values are still in this plan|Includes sample values|Spouse birthday|Spouse longevity table|Finish \d+ tasks? before running/);
    s.household.filingStatus='Married';
    const couple=a.setup();
    assert.match(couple,/Sample values are still in this plan/);
    assert.match(couple,/Spouse birthday/);
    assert.match(couple,/Finish 1 task before running/);
    assert.equal(a.state.inputSources[s.id]['household.spouseBirthday'],'Sample/default');
  }
});

test('blank assumption amounts remain Unknown across saves and backups; explicit zero is a real input',async()=>{
  const a=app(),s=a.current(),before=s.guaranteedIncome.annualIncome;s.guaranteedIncome.annualIncome=24000;
  await a.change('#main',{id:'f-guaranteedIncome-annualIncome',dataset:{field:'guaranteedIncome.annualIncome',type:'money'},value:''});
  await a.persist();assert.equal(s.guaranteedIncome.annualIncome,24000);assert.equal(a.saved().inputSources[s.id]['guaranteedIncome.annualIncome'],'Unknown');
  await a.run();await a.runLab();a.state.access.tier='pro';await a.runDecision();assert.equal(a.workers.length,0);
  const restored=app(a.saved());restored.state.setupSection=2;
  assert.match(restored.setup(),/id="f-guaranteedIncome-annualIncome"[^>]*value=""/);
  assert.match(restored.setup(),/aria-describedby="[^"]*f-guaranteedIncome-annualIncome-unknown"[\s\S]*<p class="field-unknown" id="f-guaranteedIncome-annualIncome-unknown">Unknown · enter a number; 0 means none\.<\/p>/);
  restored.state.setupSection=5;assert.match(restored.setup(),/Unknown · number needed/);assert.match(restored.setup(),/data-action="run-plan" disabled/);
  await a.click('export-backup');const backup=JSON.parse(await a.downloads[0].blob.text());
  await restored.change('#import-file',{files:[{text:async()=>JSON.stringify(backup)}],value:'backup.json'});
  assert.equal(restored.state.inputSources[s.id]['guaranteedIncome.annualIncome'],'Unknown');
  await restored.change('#main',{dataset:{field:'guaranteedIncome.annualIncome',type:'money'},value:'0'});
  assert.equal(restored.current().guaranteedIncome.annualIncome,before);assert.equal(restored.state.inputSources[s.id]['guaranteedIncome.annualIncome'],'Entered');
});

test('estimated inputs keep their numeric meaning, source and engine result through reload and copy',async()=>{
  const a=app(),s=a.current();s.household.asOfDate=model.localCalendarDate();
  await a.change('#main',{dataset:{field:'guaranteedIncome.annualIncome',type:'money'},value:'18000'});
  a.state.inputSources[s.id]['guaranteedIncome.annualIncome']='Estimated';await a.persist();
  const clean=r=>{const copy=structuredClone(r);delete copy.generatedAtEpochMillis;return copy;},before=clean(runSimulation(s));
  const restored=app(a.saved());assert.equal(restored.state.inputSources[s.id]['guaranteedIncome.annualIncome'],'Estimated');
  assert.deepEqual(clean(runSimulation(restored.current())),before);
  await restored.click('new-scenario');assert.equal(restored.state.inputSources[restored.current().id]['guaranteedIncome.annualIncome'],'Estimated');
  await restored.click('reset-assumptions');assert.equal(restored.state.inputSources[restored.current().id]._origin,'Sample/default');
});

test('cleared detailed assumptions also retain Unknown across reload',async()=>{
  const a=app(),s=a.current(),before=s.market.stockStdDev;
  await a.change('#main',{dataset:{field:'market.stockStdDev',type:'percent'},value:''});await a.persist();
  const restored=app(a.saved());assert.equal(restored.current().market.stockStdDev,before);
  assert.equal(restored.state.inputSources[s.id]['market.stockStdDev'],'Unknown');
  await restored.run();assert.equal(restored.workers.length,0);
  await restored.change('#main',{dataset:{field:'market.stockStdDev',type:'percent'},value:'0'});
  assert.equal(restored.current().market.stockStdDev,0);assert.equal(restored.state.inputSources[s.id]['market.stockStdDev'],'Entered');
});

test('legacy input provenance is not guessed from matching a sample balance',()=>{
  const s=model.baseScenario(),a=app({scenarios:[s],selectedId:s.id});a.state.setupSection=1;
  assert.match(a.setup(),/Check older saved values/);assert.doesNotMatch(a.setup(),/Value source|data-input-source/,'Sources are recorded, not picked');
  a.state.setupSection=5;assert.match(a.setup(),/Saved value; source not recorded/);assert.equal(a.current().accounts.pretax,500000);
  assert.equal(a.state.inputSources[s.id]['household.birthday'],'Estimated','Legacy birthday is explicitly inferred');
  const sources=guidance.normalizeInputSources({[s.id]:{_origin:'Unknown',fake:'Unknown','accounts.pretax':'Unknown'}},[s]);
  assert.equal(sources[s.id]._origin,'Saved value; source not recorded');assert.equal(sources[s.id].fake,undefined);assert.equal(sources[s.id]['accounts.pretax'],'Unknown');
});

test('amounts with a Monthly/Yearly control omit worked conversion examples; balances keep their guidance',()=>{
  const a=app();a.state.setupSection=1;assert.match(a.setup(),/traditional 401\(k\), 403\(b\), IRA/);
  assert.match(a.setup(),/Yearly living costs in today’s dollars\.<\/p>/);assert.match(a.setup(),/id="unit-f-spending-annualBaseSpending"/);assert.doesNotMatch(a.setup(),/\$4,000 per month, enter \$48,000/);
  a.state.setupSection=2;assert.doesNotMatch(a.setup(),/\$1,500 monthly means \$18,000 yearly|\$2,000 monthly estimate/);assert.match(a.setup(),/not the pension’s account or lump-sum value/);
  a.state.setupSection=3;assert.match(a.setup(),/Principal and interest, excluding escrow\./);assert.doesNotMatch(a.setup(),/\$1,200 each month stays \$1,200/);
});

test('results distinguish zero survivors from funding, missing financial outcomes and the steady illustration',()=>{
  const a=app();a.state.dollarBasis='future';const s=a.current(),r=runSimulation(s);r.notFailedByAge=[{age:67,notFailedShare:1,aliveShare:1},{age:68,notFailedShare:1,aliveShare:0}];
  r.balanceBands=[{age:67,median:100,pessimistic:100,optimistic:100,pathCount:100}];a.state.results.set(s.id,r);
  const html=a.results();assert.match(html,/A simulated death is not running out of money/);assert.match(html,/Not enough simulated outcomes/);assert.match(html,/0 of 100 observed/);
  assert.match(html,/typical Monte Carlo outcome or guaranteed balance/);assert.match(html,/healthcare inflation 4.0%/);assert.match(html,/future dollars/);
  assert.equal(r.successProbability>0,true);assert.match(html,/Zero observed survivors does not mean living longer is impossible/);
  assert.equal(format.shareLabel(0,10000),'0 of 10000 observed');
});

test('steady explanation retains completed return assumptions after a current-plan edit',async()=>{
  const a=app(),running=a.run(),worker=a.workers[0],r=runSimulation(worker.data.scenario);
  worker.onmessage({data:{type:'result',result:r}});await running;
  assert.equal(a.state.results.get(a.current().id).uxAssumptions.market.stockMeanReturn,worker.data.scenario.market.stockMeanReturn);
  const html=a.results();a.current().market.stockMeanReturn=.9;assert.equal(a.results(),html);
});

test('financial zero is retained for a depleted living path; absence is not filled with zero',()=>{
  const r={balanceBands:[{age:65,pathCount:1,pessimistic:0,median:0,optimistic:0}],notFailedByAge:[{age:65,notFailedShare:0,aliveShare:1},{age:66,notFailedShare:0,aliveShare:0}]};
  const rows=format.balanceDisplayRows(r);assert.equal(rows[0].noOutcomes,false);assert.equal(rows[0].median,0);assert.equal(rows[1].noOutcomes,true);assert.equal(rows[1].median,undefined);
  assert.equal(format.shareLabel(1/10000,10000),'<0.1%');assert.equal(format.shareLabel(9999/10000,10000),'>99.9%');
});

test('a final death later in the same year does not leave a survivor balance beside zero survivors',()=>{
  const r={balanceBands:[{age:95,pathCount:1,pessimistic:8000000,median:8000000,optimistic:8000000}],notFailedByAge:[{age:95,notFailedShare:1,aliveShare:.1},{age:95+1/12,notFailedShare:1,aliveShare:0}]};
  const original=JSON.stringify(r),rows=format.balanceDisplayRows(r);
  assert.deepEqual(rows,[{ageYear:95,pathCount:0,noOutcomes:true}]);assert.equal(JSON.stringify(r),original);
});

test('the guided header Run button opens the editable summary before a calculation',async()=>{
  const a=app();await a.click('start-plan');a.element('#run-button').listeners.click();
  assert.equal(a.state.setupSection,5);assert.equal(a.state.view,'setup');assert.equal(a.workers.length,0);assert.match(a.setup(),/Review before running/);
});


test('old saved plans never silently switch models, including a saved first-visit state',async()=>{
  for(const hasStartedPlan of [true,false]){
    const s=model.baseScenario();s.household.filingStatus='Married';const a=app({scenarios:[s],selectedId:s.id,hasStartedPlan});
    assert.equal(a.current().household.separatePeople,false);await a.click('start-plan');assert.equal(a.current().household.separatePeople,false);
    seedExploration(a);await a.click('enable-people');assert.equal(a.current().household.separatePeople,true);assertCleared(a);
    assert.equal(a.current().accounts.pretax,500000);assert.equal(a.current().spouseAccounts.pretax,0);assert.equal(guidance.unknownInputPaths(a.current(),a.state.inputSources).length,4);
    await a.run();assert.equal(a.workers.length,0);assert.match(a.state.message,/Unknown/);
    const restored=app(a.saved());assert.equal(restored.current().household.separatePeople,true);assert.equal(guidance.unknownInputPaths(restored.current(),restored.state.inputSources).length,4);
  }
});
test('new household inputs have ownership, annual examples and editable sources; single plans omit inactive fields',async()=>{
  const a=app();assert.equal(a.current().household.separatePeople,true);assert.deepEqual(structuredClone(a.current().contributions),model.baseScenario().contributions);
  a.state.setupSection=1;assert.match(a.setup(),/Your future savings/);assert.match(a.setup(),/Annual employee pre-tax deposits\./);assert.match(a.setup(),/id="unit-f-contributions-pretax"/);assert.doesNotMatch(a.setup(),/Spouse retirement accounts/);
  a.state.setupSection=2;assert.doesNotMatch(a.setup(),/Working household support/);
  await a.click('household-choice',{kind:'couple'});a.state.setupSection=0;assert.match(a.setup(),/Spouse retirement date/);
  a.state.setupSection=1;assert.match(a.setup(),/Spouse retirement accounts/);assert.match(a.setup(),/id="f-spouseAccounts-pretax"/);
  a.state.setupSection=2;assert.match(a.setup(),/Spouse · Pension/);assert.match(a.setup(),/id="f-spouseIncome-annualBenefitAt67"/);assert.match(a.setup(),/after taxes and these savings deposits/);
  seedExploration(a);await a.click('add-spouse-conversion');assertCleared(a);assert.equal(a.current().spouseRothHistory.conversions.length,1);
  seedExploration(a);await a.click('remove-spouse-conversion',{index:'0'});assertCleared(a);
});
test('future savings unknown versus zero, separate ownership, copy and JSON backup survive reload',async()=>{
  const a=app();await a.click('household-choice',{kind:'couple'});const s=a.current();s.spouseAccounts.pretax=90000;s.spouseIncome.annualBenefitAt67=18000;
  await a.change('#main',{dataset:{field:'contributions.pretax',type:'money'},value:''});await a.run();assert.equal(a.workers.length,0);
  await a.change('#main',{dataset:{field:'contributions.pretax',type:'money'},value:'0'});await a.change('#main',{dataset:{field:'spouseContributions.roth',type:'money'},value:'6000'});
  await a.click('new-scenario');const copy=a.current();assert.equal(copy.household.separatePeople,true);assert.equal(copy.spouseAccounts.pretax,90000);assert.equal(copy.spouseContributions.roth,6000);
  await a.click('export-backup');const backup=JSON.parse(await a.downloads.at(-1).blob.text());const restored=app(backup);
  assert.equal(restored.current().spouseIncome.annualBenefitAt67,18000);assert.equal(restored.current().spouseContributions.roth,6000);assert.equal(restored.state.inputSources[copy.id]['contributions.pretax'],'Entered');
  restored.state.setupSection=5;assert.match(restored.setup(),/Spouse retirement accounts/);assert.match(restored.setup(),/Planned|annual future savings deposits/);
});

test('Results switch between future and today’s dollars without changing counts',async()=>{
  const a=app(),s=a.current(),r=runSimulation(s);a.state.results.set(s.id,r);
  const whole=v=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(v);
  assert.equal(a.state.dollarBasis,'today');await a.change('#main',{name:'dollar-basis',value:'future',dataset:{}});
  const future=a.results();
  assert.match(future,/name="dollar-basis" value="future" checked/);
  assert.match(future,/Median ending balance · future dollars/);
  assert.ok(future.includes(whole(r.medianEndingBalance)));
  await a.change('#main',{name:'dollar-basis',value:'today',dataset:{}});
  assert.equal(a.state.dollarBasis,'today');
  const today=a.results(),steady=r.steadySimulation.monthlyDetails[0],index=r.todayDollars.steadyPriceIndexes[0];
  assert.match(today,/name="dollar-basis" value="today" checked/);
  assert.match(today,/Median ending balance · today’s dollars/);
  assert.ok(today.includes(whole(r.todayDollars.medianEndingBalance)));
  assert.ok(today.includes(whole(steady.pretax/index)),'steady rows use the run’s price level');
  assert.equal(today.match(/ of 10/g).length,future.match(/ of 10/g).length);
  assert.match(a.reportText(s,r),/Median ending balance \(today’s dollars, each path adjusted by its own inflation\)/);
  await a.change('#main',{name:'dollar-basis',value:'future',dataset:{}});
  assert.equal(a.results(),future);
});

test('Results open with a short verdict, state the preview caveat once and flag sample values',()=>{
  const a=app(),s=a.current(),r=runSimulation(s);a.state.results.set(s.id,r);
  const html=a.results();
  assert.ok(html.indexOf('In short')<html.indexOf('result-hero'));
  assert.match(html,/Am I on track\?/);assert.match(html,/too few to/);
  assert.match(html,/What should I try next\?/);
  assert.match(html,/What to do next/);
  assert.equal(html.split('too small a sample to estimate').length-1,1);
  assert.doesNotMatch(html,/lifespans run from short to long/);
  const legacy=runSimulation({...s,numberOfSimulations:10});a.state.results.set(s.id,legacy);
  assert.match(a.results(),/lifespans run from short to long/);
  a.state.results.set(s.id,r);
  assert.match(html,/Based partly on sample values/);
  a.state.inputSources[s.id]={_origin:'Entered'};
  assert.doesNotMatch(a.results(),/Based partly on sample values/);
});

test('Household setup confirms ages as dates change and uses plain longevity wording',async()=>{
  const today=model.localCalendarDate(),s=model.baseScenario();
  Object.assign(s.household,{birthday:model.addCalendarMonths(today,-51*12),retirementDate:model.addCalendarMonths(today,14*12+3)});
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='setup';a.state.setupSection=0;
  const html=a.setup();
  assert.match(html,/id="age-feedback"[^>]*>You are 51 today\. You would retire at 65 years 3 months on [^<]*, 14 years 3 months from now\.</);
  assert.match(html,/Your sex \(for life expectancy\)/);
  assert.match(html,/<option value="Male" selected>Male<\/option><option value="Female" >Female<\/option>/);
  assert.doesNotMatch(html,/mortality rates<\/option>/);
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:model.addCalendarMonths(today,16*12)});
  assert.match(a.element('#age-feedback').textContent,/^You are 51 today\. You would retire at 67 on .*, 16 years from now\.$/);
  a.current().household.filingStatus='Married';a.current().household.spouseBirthday=model.addCalendarMonths(today,-49*12);
  assert.match(a.setup(),/id="spouse-age-feedback"[^>]*>Your spouse is 49 today and would be 65 on the shared retirement date, 16 years from now\.</);
});

test('the save confirmation stays on the step where the edit happened',async()=>{
  const a=app();a.state.view='setup';a.state.guided=true;a.state.setupSection=1;
  await a.change('#main',{dataset:{field:'accounts.cash',type:'money'},value:'1000'});
  assert.equal(a.state.message,'Saved. Run the simulation to refresh results.');
  await a.click('setup-section',{index:'2'});
  assert.equal(a.state.setupSection,2);assert.equal(a.state.message,'');
  assert.doesNotMatch(a.setup(),/Saved\. Run the simulation/);
});

test('pension guidance mentions a spouse only for couples, and the home-cost hint matches the model',()=>{
  const a=app(),s=a.current();a.state.setupSection=2;
  const note=()=>a.setup().match(/You · Pension or annuity income<\/h3><p>(.*?)<\/p>/)[1];
  s.household.filingStatus='Single';assert.doesNotMatch(note(),/spouse/i);
  s.household.filingStatus='Married';assert.match(note(),/spouse/);
  const hint=guidance.FIELD_GUIDANCE['home.annualTaxesAndInsurance'];
  assert.doesNotMatch(hint,/Also include/);assert.match(hint,/Keep them in base spending too/);assert.match(hint,/stops if the home is sold/);
});

test('examples stay identifiable while edited and duplicated plans become personal',async()=>{
  const a=app(),base=a.current().id;
  assert.equal(a.state.exampleIds.size,3);
  assert.match(a.scenarios(),/Example/);
  await a.change('#main',{dataset:{field:'accounts.cash',type:'money'},value:'1,234.50'});
  assert.equal(a.state.exampleIds.has(base),false);
  assert.equal(a.state.exampleIds.size,2);
  await a.persist();assert.equal(app(a.saved()).state.exampleIds.has(base),false);
  await a.click('select-scenario',{id:'later-retirement'});
  a.state.entryPeriods[a.current().id]={'spending.annualBaseSpending':'month'};
  await a.click('new-scenario');const first=a.current().id;
  assert.equal(a.state.exampleIds.has(first),false);
  assert.equal(a.state.entryPeriods[first]['spending.annualBaseSpending'],'month');
  await a.click('new-scenario');assert.notEqual(a.current().id,first);
  const legacy=model.baseScenario();legacy.name='Base plan';
  assert.equal(app({scenarios:[legacy]}).state.exampleIds.size,0,'Names and balances cannot identify an example');
});

test('the basic guide discloses current assumptions without changing their values',async()=>{
  const s=model.baseScenario();s.market.stockMeanReturn=.084;s.spending.spendingPathModel='Flat';
  const a=app({scenarios:[s],selectedId:s.id});a.state.guided=true;
  const before=structuredClone(a.current());a.state.setupSection=4;
  assert.match(a.setup(),/Stocks after retirement[\s\S]*8\.4%/);
  assert.match(a.setup(),/<details class="guided-extra" id="basic-market-settings"><summary>Returns, investment mix/);
  await a.click('setup-detail',{mode:'advanced'});
  assert.doesNotMatch(a.setup(),/id="basic-market-settings"/);
  await a.click('setup-detail',{mode:'basic'});a.state.setupSection=1;
  assert.match(a.setup(),/Base spending stays level before inflation/);
  assert.match(a.setup(),/<summary>Advanced · Spending pattern and inflation/);
  assert.match(a.setup(),/class="field-source"/,'Sources identify example and saved inputs beside their fields');
  assert.deepEqual(structuredClone(a.current()),before);
  assert.equal(app(a.saved()).state.basicSetup,true);
  await a.change('#main',{dataset:{field:'spending.generalInflationMean',type:'percent'},value:'3'});
  assert.match(a.element('#basic-spending-summary').innerHTML,/3\.0% a year/);
  await a.change('#main',{dataset:{field:'market.stockMeanReturn',type:'percent'},value:'7.5'});
  assert.match(a.element('#basic-market-summary').innerHTML,/Stocks after retirement[\s\S]*7\.5%/);
});

test('monthly and yearly entry preserves canonical amounts and exact saved precision',async()=>{
  const s=model.baseScenario();s.spending.annualBaseSpending=100001.123456789;
  const a=app({scenarios:[s],selectedId:s.id});a.state.setupSection=1;
  const path='spending.annualBaseSpending',selector='#f-spending-annualBaseSpending';
  for(let i=0;i<4;i++){
    await a.change('#main',{dataset:{entryPeriod:path},value:i%2?'year':'month'});
    assert.equal(a.current().spending.annualBaseSpending,s.spending.annualBaseSpending);
  }
  await a.change('#main',{dataset:{entryPeriod:path},value:'month'});
  assert.match(a.setup(),/Monthly living costs in today’s dollars/);
  assert.doesNotMatch(a.setup(),/For \$4,000 per month, enter \$48,000/);
  const el=a.element(selector);Object.assign(el,{id:selector.slice(1),dataset:{field:path,type:'money'},value:moneyInput.moneyInputValue(s.spending.annualBaseSpending/12)});
  await a.change('#main',{dataset:{entryPeriod:path},value:'year'});
  assert.equal(a.current().spending.annualBaseSpending,s.spending.annualBaseSpending);
  await a.change('#main',{dataset:{entryPeriod:path},value:'month'});
  await a.change('#main',{id:selector.slice(1),dataset:{field:path,type:'money'},value:'4,000.25'});
  assert.equal(a.current().spending.annualBaseSpending,48003);
  assert.equal(a.saved().scenarios[0].spending.annualBaseSpending,48003);
  const restored=app(a.saved());restored.state.setupSection=1;
  assert.match(restored.setup(),/value="4,000\.25"/);
  assert.match(restored.setup(),/\$48,003\.00 \/ year/);
  const monthly='healthcare.preMedicareMonthlyPremium';
  await a.change('#main',{dataset:{entryPeriod:monthly},value:'year'});
  await a.change('#main',{dataset:{field:monthly,type:'money'},value:'12,000'});
  assert.equal(a.current().healthcare.preMedicareMonthlyPremium,1000);
});

test('grouped money entry rejects malformed and out-of-range conversions without changing saved values',async()=>{
  const a=app(),path='spending.annualBaseSpending';
  await a.change('#main',{dataset:{entryPeriod:path},value:'month'});
  const prior=a.current().spending.annualBaseSpending;
  for(const value of ['4,00','1,234,56','1e','$oops',String(model.MAX_DOLLAR_AMOUNT/2)]){
    await a.change('#main',{dataset:{field:path,type:'money'},value});
    assert.equal(a.current().spending.annualBaseSpending,prior,value);
    assert.equal(a.saved().scenarios[0].spending.annualBaseSpending,prior);
    assert.match(a.state.message,/previous value was kept/i);
  }
  await a.change('#main',{dataset:{field:path,type:'money'},value:'$4,200.50'});
  assert.equal(a.current().spending.annualBaseSpending,50406);
  await a.change('#main',{dataset:{field:path,type:'money'},value:''});
  assert.equal(a.state.inputSources[a.current().id][path],'Unknown');
  assert.equal(a.current().spending.annualBaseSpending,50406,'Clearing is not zero');
});

test('budget uses monthly payment choices and one final confirmation, which resets after edits',async()=>{
  const a=app();a.state.view='budget';await a.click('add-month');
  await a.change('#main',{dataset:{month:'0',part:'checking'},value:'5,000'});
  const prior=a.current().spending.annualBaseSpending;
  await a.click('apply-budget');assert.equal(a.current().spending.annualBaseSpending,prior);
  await a.change('#main',{dataset:{costConfirm:''},checked:true});
  assert.equal(a.budgetCostsReviewed(),false,'An unanswered month cannot be confirmed');
  assert.doesNotMatch(a.budgetCostCheck(),/data-cost-review|budget-cost-review/);
  await a.change('#main',{dataset:{monthCost:'mortgage',index:'0'},checked:true});
  await a.change('#main',{dataset:{costConfirm:''},checked:true});
  assert.equal(a.budgetCostsReviewed(),false,'An included payment needs its amount');
  await a.change('#main',{dataset:{month:'0',part:'mortgage'},value:'1,000'});
  assert.equal(a.budgetView().costConfirmed,false);
  assert.match(a.budgetCostCheck(),/Mortgage payments deducted/);assert.match(a.budgetCostCheck(),/\$1,000\.00 \/ month/);
  await a.click('review-budget-deductions');assert.equal(a.element('#budget-month-0').open,true);
  await a.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(a.budgetCostsReviewed(),true);
  await a.click('apply-budget');assert.equal(a.current().spending.annualBaseSpending,48000);
  await a.change('#main',{dataset:{month:'0',part:'mortgage'},value:'1,100'});
  assert.equal(a.budgetCostsReviewed(),false,'A changed deduction needs confirmation again');
  await a.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(a.budgetCostsReviewed(),true);
  await a.change('#main',{dataset:{month:'0',part:'checking'},value:'5,100'});
  assert.equal(a.budgetCostsReviewed(),false);assert.equal(a.current().budget.monthlyBudgets[0].includedPayments.mortgage,true);
  assert.equal(a.current().spending.annualBaseSpending,48000,'Draft edits do not change applied spending');
  await a.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(a.budgetCostsReviewed(),true);
  await a.click('month-costs-excluded',{index:'0'});assert.equal(a.current().budget.monthlyBudgets[0].adjustments.mortgage,0);
  assert.equal(a.budgetCostsReviewed(),false);await a.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(a.budgetCostsReviewed(),true);
});

test('budget can be applied with excluded separate costs and returns to the active guided step',async()=>{
  const a=app();a.state.view='setup';a.state.guided=true;a.state.setupSection=1;
  a.enterBudget();assert.equal(a.state.view,'budget');assert.match(a.budget(),/Return to setup/);
  await a.click('add-month');await a.change('#main',{dataset:{month:'0',part:'credit'},value:'3,500'});
  await a.click('month-costs-excluded',{index:'0'});await a.change('#main',{dataset:{costConfirm:''},checked:true});
  assert.equal(a.budgetCostsReviewed(),true);
  await a.click('apply-budget');assert.equal(a.current().spending.annualBaseSpending,42000);
  await a.click('return-setup');assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,1);assert.equal(a.state.guided,true);
  assert.match(a.setup(),/<select id="setup-guided-step">[\s\S]*<option value="1" selected>2\./);
  await a.change('#main',{id:'setup-guided-step',dataset:{},value:'4'});
  assert.equal(a.state.setupSection,4);
});

test('all comparison modes allow 1000 Pro paths, keep free at 100 and respect smaller Pro selections',async()=>{
  for(const [tier,selectedCount,expectedCount] of [['pro',10000,1000],['pro',380,380],['pro',4,4],['free',10000,100]]){
    const completed=runSimulation({...model.baseScenario(),numberOfSimulations:4});
    // Mock successful worker completion; this test verifies dispatch limits,
    // displayed counts and preservation of saved choices, not engine math.
    completed.provenance.simulationCount=expectedCount;
    for(const [mode,rowCount] of [['run',tier==='pro'?4:1],['lower',tier==='pro'?5:2],['stress',tier==='pro'?10:0]]){
      const s=model.baseScenario();s.numberOfSimulations=selectedCount;s.simulationPathsCustomized=true;
      const a=app({scenarios:[s],selectedId:s.id});a.state.access.tier=tier;
      const before=JSON.stringify(a.current()),counts=[];
      await settle(a,mode==='run'?a.runLab():mode==='lower'?a.click('compare-lower-returns'):a.runLabStress(),worker=>{counts.push(worker.data.scenario.numberOfSimulations);return structuredClone(completed);});
      assert.equal(counts.length,rowCount,`${tier}: ${mode}`);assert.ok(counts.every(n=>n===expectedCount),`${tier}: ${mode}`);
      assert.equal(JSON.stringify(a.current()),before);
      if(mode!=='stress')assert.ok(a.lab().includes(`${expectedCount.toLocaleString('en-US')} paths each`));
      assert.match(a.billingView(),/Comparisons use up to 1,000 paths per scenario/);
    }
  }
});

test('comparison copies retain personal balances, full-run count and all unchanged assumptions',async()=>{
  const s=model.baseScenario();s.accounts.pretax=456789;s.accounts.cash=54321;s.numberOfSimulations=380;s.simulationPathsCustomized=true;
  s.market.stockMeanReturn=.09;
  const a=app({scenarios:[s],selectedId:s.id});a.state.access.tier='pro';
  a.state.entryPeriods[s.id]={'spending.annualBaseSpending':'month'};
  a.state.inputSources[s.id]['accounts.cash']='Entered';const parent=structuredClone(a.current()),pending=a.runLab();
  await settle(a,pending,worker=>{assert.equal(worker.data.scenario.numberOfSimulations,380);return runSimulation({...worker.data.scenario,numberOfSimulations:40});});
  const row=labRows(a)[2];assert.equal(row.label,'Spend 5% less');assert.match(a.lab(),/data-action="lab-copy"/);
  const candidate=planLab.applyWhatIf(parent,row.changes);
  await a.click('lab-copy',{id:row.id});const copy=structuredClone(a.current());
  assert.notEqual(copy.id,parent.id);assert.equal(copy.numberOfSimulations,380);
  assert.equal(copy.spending.annualBaseSpending,parent.spending.annualBaseSpending*.95);
  candidate.id=copy.id;candidate.name=copy.name;
  assert.deepEqual(copy,candidate);
  assert.deepEqual(structuredClone(a.state.scenarios.find(x=>x.id===parent.id)),parent);
  assert.equal(a.state.inputSources[copy.id]['accounts.cash'],'Entered');
  assert.equal(a.state.inputSources[copy.id]['spending.annualBaseSpending'],'Entered');
  assert.equal(a.state.entryPeriods[copy.id]['spending.annualBaseSpending'],'month');
  assert.equal(a.state.exampleIds.has(copy.id),false);assert.equal(a.state.results.has(copy.id),false);
  assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,5);
});

test('a comparison copied from an earlier date uses today after saving and reopening',async()=>{
  const today=model.localCalendarDate(),s=model.baseScenario();
  Object.assign(s.household,{separatePeople:true,alreadyRetired:true,birthday:model.addCalendarMonths(today,-60*12),retirementDate:''});
  const a=app({scenarios:[s],selectedId:s.id}),source=a.current();
  await a.click('lab-add',{preset:'spend-less'});const w=a.labViewModel().set.whatIfs[0];
  source.household.asOfDate=model.addCalendarMonths(today,-1);
  await a.click('lab-copy',{id:w.id});
  const copy=a.current();assert.notEqual(copy.id,source.id);assert.equal(copy.household.asOfDate,'');assert.equal(source.household.asOfDate,model.addCalendarMonths(today,-1));
  const reopened=app(a.saved()),timeline=model.scenarioTimeline(reopened.current());
  assert.equal(timeline.startDate,today);assert.equal(timeline.currentAge,60);
  assert.match(reopened.dashboard(),/Current age 60/);assert.ok(reopened.dashboard().includes(model.dateLabel(today)));
  const running=reopened.run();assert.equal(reopened.workers.length,1);
  assert.equal(reopened.workers[0].data.scenario.household.asOfDate,timeline.startDate);
  await reopened.click('cancel-calculation');await running;
});

test('result details open collapsed and existing dollar display preferences remain respected',()=>{
  const a=app();a.state.results.set(a.current().id,runSimulation(a.current()));
  assert.equal(a.state.dollarBasis,'today');
  const html=a.results();assert.match(html,/<details class="card result-detail"><summary>How to read these results/);
  assert.match(html,/<details class="card result-detail"><summary>More charts/);
  assert.doesNotMatch(html,/<details class="card result-detail"[^>]*\bopen/);
  const restored=app(null,{preferences:new Map([['retirement-dollar-basis','future']])});
  assert.equal(restored.state.dollarBasis,'future');
});

test('JSON backups retain example markers and monthly or yearly entry choices',async()=>{
  const a=app();a.state.entryPeriods[a.current().id]={'socialSecurity.annualBenefitAt67':'month'};
  await a.click('export-backup');const text=await a.downloads[0].blob.text(),backup=JSON.parse(text);
  assert.equal(backup.exampleIds.length,3);
  const restored=app();await restored.change('#import-file',{files:[{text:async()=>text}],value:'backup.json'});
  assert.equal(restored.state.exampleIds.size,3);
  restored.state.setupSection=2;
  assert.match(restored.setup(),/Your Social Security \/ month at age 67/);
});


test('malformed entry preferences cannot prevent opening or editing a valid plan',async()=>{
  const s=model.baseScenario();s.id='__proto__';
  for(const choice of [true,42,'month',[],{'spending.annualBaseSpending':'invalid'}]){
    const saved={scenarios:[s],selectedId:s.id,entryPeriods:Object.fromEntries([[s.id,choice]])},a=app(saved);
    await a.change('#main',{dataset:{entryPeriod:'spending.annualBaseSpending'},value:'month'});
    assert.equal(a.state.entryPeriods[s.id]['spending.annualBaseSpending'],'month');
    assert.equal(a.current().spending.annualBaseSpending,s.spending.annualBaseSpending);
    await a.change('#import-file',{files:[{text:async()=>JSON.stringify(saved)}],value:'backup.json'});
    await a.change('#main',{dataset:{entryPeriod:'spending.annualBaseSpending'},value:'month'});
    assert.equal(a.state.entryPeriods[s.id]['spending.annualBaseSpending'],'month');
  }
});

test('basic pension questions reveal details on income entry and preserve unused timing and survivor values',async()=>{
  const a=app(),s=a.current();a.state.view='setup';a.state.guided=true;a.state.setupSection=2;
  Object.assign(s.guaranteedIncome,{annualIncome:0,startAge:66,startAgeMonths:7,annualIncrease:.02,survivorPercent:50});
  const before=structuredClone(s.guaranteedIncome);
  assert.match(a.setup(),/id="your-pension-question">Do you have a pension/);
  assert.match(a.setup(),/data-question="your-pension" data-answer="no"/);
  assert.match(a.setup(),/id="your-pension-details" hidden/);
  assert.deepEqual(structuredClone(s.guaranteedIncome),before);
  const renders=a.renders();
  await a.change('#main',{dataset:{field:'guaranteedIncome.annualIncome',type:'money'},value:'18000'});
  assert.equal(a.element('#your-pension-details').hidden,false);
  assert.match(a.element('#your-pension-status').textContent,/Amounts entered/);
  assert.equal(a.renders(),renders,'Revealing pension details must not replace an active input');
  await a.change('#main',{dataset:{field:'guaranteedIncome.annualIncome',type:'money'},value:'0'});
  assert.equal(a.element('#your-pension-details').hidden,true);
  assert.deepEqual(structuredClone(s.guaranteedIncome),before);
  assert.equal(a.saved().scenarios[0].guaranteedIncome.startAgeMonths,7);
  await a.change('#main',{dataset:{field:'guaranteedIncome.startAge',type:'number'},value:''});
  assert.equal(a.element('#your-pension-details').hidden,false,'An Unknown detail must remain reachable even with zero pension income');
  assert.match(a.setup(),/id="your-pension"[^>]* open/);
  assert.match(a.element('#your-pension-status').textContent,/Unknown/);
  assert.equal(s.guaranteedIncome.startAge,66);
  assert.ok(await completeCachedRun(a),'Unknown timing does not block a zero pension');
  await a.change('#main',{dataset:{field:'guaranteedIncome.annualIncome',type:'money'},value:'18000'});
  const count=a.workers.length;await a.run();assert.equal(a.workers.length,count,'Unknown timing blocks a positive pension');
  assert.match(a.state.message,/Review Unknown inputs/);
});

test('basic questions expose saved mortgage, rent, spouse pension and savings without changing assumptions',async()=>{
  const s=model.baseScenario();s.household.separatePeople=true;s.household.filingStatus='Married';
  s.mortgage.currentBalance=120000;s.rent.monthlyRent=500;s.spouseIncome.annualPension=9000;s.spouseContributions.cash=1200;
  const a=app({scenarios:[s],selectedId:s.id});a.state.guided=true;const before=structuredClone(a.current());
  a.state.setupSection=3;
  for(const id of ['home-inputs','mortgage-inputs','rent-inputs'])assert.match(a.setup(),new RegExp('id="'+id+'"[^>]* open'));
  a.state.setupSection=2;assert.match(a.setup(),/id="spouse-pension"[^>]* open/);assert.doesNotMatch(a.setup(),/id="spouse-pension-details" hidden/);
  a.state.setupSection=1;assert.match(a.setup(),/id="your-savings"[^>]*><summary>/);assert.match(a.setup(),/id="spouse-savings"[^>]* open/);
  assert.match(a.setup(),/Include in base spending/);assert.match(a.setup(),/Leave these out of base spending so they are counted once/);
  await a.click('setup-detail',{mode:'advanced'});a.state.setupSection=2;
  assert.doesNotMatch(a.setup(),/id="your-pension-details"/);
  assert.match(a.setup(),/id="f-guaranteedIncome-startAgeMonths"/);
  assert.deepEqual(structuredClone(a.current()),before);
});

test('investment review opens actual controls and labels edited rates without applying suggested defaults',async()=>{
  const a=app(),s=a.current();a.state.view='setup';a.state.guided=true;a.state.setupSection=4;
  const before=structuredClone(s),renders=a.renders();
  assert.match(a.setup(),/13\.3% \/ year · Example assumption/);
  assert.match(a.setup(),/Help me review these assumptions/);
  await a.click('review-investments');assert.equal(a.element('#basic-market-settings').open,true);
  assert.deepEqual(structuredClone(s),before);assert.equal(a.renders(),renders);
  await a.change('#main',{dataset:{field:'market.stockMeanReturn',type:'percent'},value:'8'});
  assert.match(a.element('#basic-market-summary').innerHTML,/8\.0% \/ year · Entered assumption/);
  assert.equal(s.market.preRetirementMeanReturn,before.market.preRetirementMeanReturn);
  assert.equal(s.market.stockStdDev,before.market.stockStdDev);
  assert.equal(a.saved().scenarios[0].market.stockMeanReturn,.08);
});

test('results explain zero shortfalls and route next steps to review, investments and comparisons',async()=>{
  const a=app(),s=a.current();s.market.stockMeanReturn=.133;const r=runSimulation(s);r.successProbability=1;r.riskBreakdown.primaryRisk='none';
  a.state.results.set(s.id,r);a.state.view='results';
  let html=a.results();assert.match(html,/All 100 preview lifetimes stayed funded/);
  assert.match(html,/assumes stocks grow 13\.3% a year[^<]*Try a lower return/);assert.doesNotMatch(html,/No sensitivity check reduced shortfalls/);
  const modest=structuredClone(s);Object.assign(modest.market,{preRetirementMeanReturn:.07,stockMeanReturn:.07});
  r.uxAssumptions=modest;r.todayDollars.medianEndingBalance=modest.spending.annualBaseSpending*25;assert.match(a.results(),/about 25 years of base spending left/);
  r.todayDollars.medianEndingBalance=modest.spending.annualBaseSpending*5;assert.match(a.results(),/Try higher healthcare costs or a different retirement date/);
  delete r.uxAssumptions;
  assert.match(html,/Review sample inputs/);
  a.state.inputSources[s.id]={_origin:'Entered'};assert.match(a.results(),/Review my inputs/);
  await a.click('setup-section',{index:'5',resultsLink:''});assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,5);
  a.state.view='results';await a.click('setup-section',{index:'4',resultsLink:''});assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,4);
  r.successProbability=.5;a.state.view='results';html=a.results();assert.match(html,/Start by checking spending and income/);
  r.riskBreakdown.primaryRisk='spending';r.riskBreakdown.recommendedNextTest='Test a 5% lower spending scenario.';
  assert.match(a.results(),/Test a 5% lower spending scenario/,'Helpful sensitivity results remain visible when shortfalls occurred');
});

test('money entry accepts k and M shorthand but still rejects malformed amounts',()=>{
  for(const [text,value] of [['40k',40000],['$40K',40000],['$1.2M',1200000],['-$2.5k',-2500],['40 k',40000],['1,500k',1500000]])assert.equal(moneyInput.parseMoneyInput(text),value,text);
  for(const text of ['k','40kk','1e3k','4,0k','$k'])assert.ok(Number.isNaN(moneyInput.parseMoneyInput(text)),text);
});

test('typing a date year digit by digit neither reports an error nor redraws the editor',async()=>{
  const a=app(),before=a.current().household.retirementDate,renders=a.renders(),year=Number(before.slice(0,4))+5;
  for(const partial of ['0002','0020','0204'])await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:`${partial}-06-01`});
  assert.equal(a.current().household.retirementDate,before);assert.doesNotMatch(a.state.message,/Error/);assert.equal(a.renders(),renders);
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:`${year}-06-01`});
  assert.equal(a.current().household.retirementDate,`${year}-06-01`);
  await a.change('#main',{dataset:{field:'household.retirementDate',type:'date'},value:'2000-01-01'});
  assert.match(a.state.message,/Error: Choose a valid retirement date/);assert.equal(a.renders(),renders,'an invalid date keeps the live editor in place');
  a.state.message='';a.element('#main').listeners.focusout({target:{dataset:{type:'date',field:'household.birthday'},value:'0197-04-15'}});
  assert.match(a.state.message,/Error: Choose a valid birthday/,'an unfinished year is reported when leaving the field');
});

test('a new personal plan asks about future savings and clears example premiums and Roth history',async()=>{
  const a=app();await a.click('start-plan');const s=a.current(),sources=()=>a.state.inputSources[s.id];
  for(const path of ['healthcare.preMedicareMonthlyPremium','rothHistory.contributionBasis','rothHistory.firstContributionYear','contributions.pretax','contributions.cash'])assert.equal(sources()[path],'Unknown',path);
  a.state.setupSection=1;let html=a.setup();
  assert.match(html,/Are you still adding to your savings\?[^]*data-action="savings-answer"[^>]*data-answer="yes"/);assert.doesNotMatch(html,/id="f-contributions-pretax"/);
  await a.click('savings-answer',{prefix:'contributions',answer:'yes'});html=a.setup();
  assert.match(html,/id="f-contributions-pretax"/);assert.match(html,/Set the 5 remaining amounts to none/);
  await a.change('#main',{dataset:{field:'contributions.pretax',type:'money'},value:'20000'});
  await a.click('savings-rest-none',{prefix:'contributions'});
  assert.equal(s.contributions.pretax,20000);assert.equal(s.contributions.cash,0);assert.equal(sources()['contributions.cash'],'Entered');
  const b=app();await b.click('start-plan');await b.click('savings-answer',{prefix:'contributions',answer:'no'});
  assert.ok(['pretax','employerPretax','roth','taxable','cash'].every(key=>b.current().contributions[key]===0&&b.state.inputSources[b.current().id]['contributions.'+key]==='Entered'));
  // No Roth balance means no Roth history to find.
  await b.click('input-none',{path:'accounts.roth'});
  assert.equal(b.state.inputSources[b.current().id]['rothHistory.contributionBasis'],'Entered');assert.equal(b.current().rothHistory.firstContributionYear,0);
});

test('an Unknown pre-Medicare premium blocks a run only when someone retires before 65',async()=>{
  const a=app();await a.click('start-plan');const s=a.current(),today=model.localCalendarDate();
  s.household.birthday=model.addCalendarMonths(today,-60*12);s.household.retirementDate=model.addCalendarMonths(today,7*12);model.syncCalendarAges(s);
  a.state.inputSources[s.id]={_origin:'Entered','healthcare.preMedicareMonthlyPremium':'Unknown'};
  const pending=a.run();assert.equal(a.workers.length,1,'retiring at 67 never uses the premium');
  a.workers[0].onmessage({data:{type:'result',result:runSimulation(a.workers[0].data.scenario)}});await pending;
  s.household.retirementDate=model.addCalendarMonths(today,2*12);model.syncCalendarAges(s);
  await a.run();assert.equal(a.workers.length,1,'retiring at 62 needs the premium');assert.match(a.state.message,/Unknown/);
});

test('a younger working spouse needs an Unknown premium only if their own retirement is before 65',async()=>{
  const today=model.localCalendarDate(),s=model.baseScenario();
  Object.assign(s.household,{separatePeople:true,filingStatus:'Married',birthday:model.addCalendarMonths(today,-67*12),retirementDate:today,spouseBirthday:model.addCalendarMonths(today,-60*12),spouseRetirementDate:model.addCalendarMonths(today,60)});
  const a=app({scenarios:[s],selectedId:s.id,inputSources:{[s.id]:{'healthcare.preMedicareMonthlyPremium':'Unknown'}}});
  a.state.setupSection=3;assert.match(a.setup(),/Not needed for this plan/);
  const running=a.run();assert.equal(a.workers.length,1);await a.click('cancel-calculation');await running;
  s.household.spouseRetirementDate=model.addCalendarMonths(today,24);
  a.current().household.spouseRetirementDate=s.household.spouseRetirementDate;
  a.state.setupSection=3;assert.match(a.setup(),/id="f-healthcare-preMedicareMonthlyPremium"/);
  await a.run();assert.equal(a.workers.length,1);assert.match(a.state.message,/Unknown/);
});

test('excluded care ignores Unknown details while preserving them for re-enabling and backups',async()=>{
  const s=model.prepareCalendarScenario(model.baseScenario()),a=app({scenarios:[s],selectedId:s.id}),before=structuredClone(s.longTermCare);
  for(const [field,type] of [['annualCost','money'],['averageDurationYears','number'],['averageDurationMonths','number']])await a.change('#main',{dataset:{field:'longTermCare.'+field,type},value:''});
  await a.change('#main',{dataset:{field:'longTermCare.enabled',type:'checkbox'},checked:false});
  assert.deepEqual(structuredClone(a.current().longTermCare),{...before,enabled:false});
  const running=a.run();assert.equal(a.workers.length,1);await a.click('cancel-calculation');await running;
  await a.click('export-backup');const text=await a.downloads.at(-1).blob.text(),imported=app();
  await imported.change('#import-file',{files:[{text:async()=>text}],value:'backup.json'});
  const restored=app(imported.saved());const resumed=restored.run();assert.equal(restored.workers.length,1);await restored.click('cancel-calculation');await resumed;
  await restored.change('#main',{dataset:{field:'longTermCare.enabled',type:'checkbox'},checked:true});
  await restored.run();assert.equal(restored.workers.length,1);assert.match(restored.state.message,/Unknown/);
  assert.equal(restored.current().longTermCare.annualCost,before.annualCost);
  assert.equal(restored.state.inputSources[s.id]['longTermCare.annualCost'],'Unknown');
});

for(const [path,enabled] of [['rothConversion.marginalRateCap','rothConversion.enabled'],['withdrawalStrategy.drawdownTrigger','withdrawalStrategy.useCashReserveDuringDrawdowns']]){
  test(`disabled ${enabled} ignores its Unknown setting until re-enabled`,async()=>{
    const s=model.prepareCalendarScenario(model.baseScenario()),a=app({scenarios:[s],selectedId:s.id}),[group,key]=path.split('.'),before=s[group][key];
    await a.change('#main',{dataset:{field:path,type:'percent'},value:''});
    await a.change('#main',{dataset:{field:enabled,type:'checkbox'},checked:false});
    assert.equal(a.current()[group][key],before);assert.equal(a.state.inputSources[s.id][path],'Unknown');
    a.state.setupSection=5;assert.doesNotMatch(a.setup(),/data-action="run-plan" disabled/);
    const running=a.run();assert.equal(a.workers.length,1);await a.click('cancel-calculation');await running;
    const restored=app(a.saved()),resumed=restored.run();
    assert.equal(restored.workers.length,1);await restored.click('cancel-calculation');await resumed;
    await restored.change('#main',{dataset:{field:enabled,type:'checkbox'},checked:true});
    restored.state.setupSection=5;assert.match(restored.setup(),/data-action="run-plan" disabled/);
    await restored.run();assert.equal(restored.workers.length,1);assert.match(restored.state.message,/Review Unknown inputs/);
    assert.equal(restored.current()[group][key],before);assert.equal(restored.state.inputSources[s.id][path],'Unknown');
  });
}

test('the sample notice separates personal examples from example model assumptions',async()=>{
  const a=app(),s=a.current();a.state.guided=true;
  a.state.inputSources[s.id]={_origin:'Sample/default'};for(const path of ['household.birthday','household.retirementDate','household.gender','accounts.pretax','accounts.roth','accounts.cash','spending.annualBaseSpending','socialSecurity.annualBenefitAt67','socialSecurity.claimAge','healthcare.preMedicareMonthlyPremium','rothHistory.contributionBasis','rothHistory.firstContributionYear'])a.state.inputSources[s.id][path]='Entered';
  const html=a.setup();
  assert.match(html,/<summary>Example model assumptions in use \(\d+\)<\/summary>/);assert.match(html,/Stock return average %: 13\.3%/);
  assert.doesNotMatch(html,/Your pension or annuity \/ year: \$0|Mortgage payment \/ month: \$0/,'zero amounts are not examples to replace');
  s.socialSecurity.claimAge=67;a.state.inputSources[s.id]['socialSecurity.claimAge']='Sample/default';
  assert.match(a.setup(),/Sample values are still in this plan \(1\)[^]*Claim age: 67[^]*<summary>Example model assumptions/);
  a.state.setupSection=5;assert.match(a.setup(),/<dt>Pension or annuity<\/dt><dd>You None<small>None by default/);
});

test('individuals choose only individual filing statuses; couples are married filing jointly',async()=>{
  const a=app();a.state.guided=true;let html=a.setup();
  assert.match(html,/<label for="f-household-filingStatus">Tax filing status<\/label>/);assert.match(html,/<option value="HeadOfHousehold"/);assert.doesNotMatch(html,/<option value="Married"/);
  await a.click('household-choice',{kind:'couple'});html=a.setup();
  assert.doesNotMatch(html,/id="f-household-filingStatus"/);assert.match(html,/married filing jointly/);
});

test('the URL hash restores the view and setup step, and Continue my plan returns to the last step',async()=>{
  const location={search:'',pathname:'/',hash:'#/forecast/3'},a=app(null,{location});
  assert.equal(a.state.view,'setup');assert.equal(a.state.guided,true);assert.equal(a.state.setupSection,2);
  const results=app(null,{location:{search:'',pathname:'/',hash:'#/results'}});assert.equal(results.state.view,'results');
  const skip=app(null,{location:{search:'',pathname:'/',hash:'#main'}});assert.equal(skip.state.view,'dashboard');
  const s=model.baseScenario(),preferences=new Map([['retirement-setup-step',JSON.stringify({id:s.id,section:3})]]);
  const saved=app({scenarios:[s],selectedId:s.id},{preferences});await saved.click('start-plan');
  assert.equal(saved.state.setupSection,3);assert.equal(saved.state.view,'setup');
  saved.state.setupSection=4;saved.render();assert.deepEqual(JSON.parse(preferences.get('retirement-setup-step')),{id:s.id,section:4});
});

test('result follow-up edits retain the guided mode after reload, while an explicit detailed choice stays detailed',async()=>{
  const preferences=new Map(),a=app(null,{preferences});await a.click('start-plan');
  await a.click('optional-answer',{question:'your-pension',answer:'no'});
  const saved=a.saved(),id=a.current().id;
  const restored=app(saved,{preferences,location:{search:'',pathname:'/',hash:'#/results'}});
  await restored.click('setup-section',{index:'5',resultsLink:''});
  await restored.click('edit-input',{index:'2',path:'socialSecurity.claimAge'});
  assert.equal(restored.state.guided,true);assert.match(restored.setup(),/Build your forecast/);
  assert.match(restored.setup(),/Do you have a pension or annuity/);
  assert.match(restored.setup(),/id="your-pension-details"[^>]* hidden/);
  await restored.click('toggle-guided');
  const detailed=app(saved,{preferences,location:{search:'',pathname:'/',hash:'#/results'}});
  await detailed.click('edit-input',{index:'2',path:'socialSecurity.claimAge'});
  assert.equal(detailed.state.guided,false);assert.match(detailed.setup(),/Set up your Monte Carlo model/);
  assert.equal(JSON.parse(preferences.get('retirement-setup-modes'))[id],false);
});

test('editor modes belong to each plan and guided choices in older saves are restored',async()=>{
  const preferences=new Map(),a=app(null,{preferences});await a.click('start-plan');const original=a.current().id;
  await a.click('new-scenario');const copy=a.current().id;assert.equal(a.state.guided,true);
  await a.click('toggle-guided');await a.click('select-scenario',{id:original});assert.equal(a.state.guided,true);
  await a.click('select-scenario',{id:copy});assert.equal(a.state.guided,false);
  const legacy=app(a.saved(),{preferences:new Map([['retirement-setup-step',JSON.stringify({id:copy,section:2})]])});
  assert.equal(legacy.state.guided,true);
});

test('personal setup identifies supported employer Roth accounts before changing financial inputs',async()=>{
  const a=app(),before=structuredClone(a.current());
  assert.match(a.dashboard(),/data-action="check-accounts">Build my forecast/);
  await a.click('check-accounts');assert.equal(a.state.view,'account-check');
  assert.deepEqual(structuredClone(a.current()),before);
  assert.match(a.accountCheck(),/data-action="start-plan" disabled/);
  for(const answer of ['yes','unsure','no']){
    await a.click('account-answer',{answer});assert.deepEqual(structuredClone(a.current()),before);
    assert.match(a.accountCheck(),new RegExp(`data-answer="${answer}" aria-pressed="true"`));
    if(answer==='yes')assert.match(a.accountCheck(),/accounts are supported/);
    else if(answer==='unsure')assert.match(a.accountCheck(),/Check your account statement/);
    else assert.match(a.accountCheck(),/Continue to household setup/);
  }
  await a.click('start-plan');assert.equal(a.state.view,'setup');assert.equal(a.state.guided,true);
  assert.equal(a.state.inputSources[a.current().id]['accounts.pretax'],'Unknown');
});

test('account checks survive reload, copies and backups, without changing the calculation',async()=>{
  const a=app(),before=structuredClone(a.current());await a.click('account-answer',{answer:'yes'});
  assert.deepEqual(structuredClone(a.current()),before);
  const restored=app(a.saved(),{location:{search:'',pathname:'/',hash:'#/account-check'}});
  assert.equal(restored.state.view,'account-check');assert.match(restored.accountCheck(),/data-answer="yes" aria-pressed="true"/);
  await a.click('new-scenario');assert.equal(a.state.accountChecks[a.current().id],'yes');
  await a.click('export-backup');const backup=JSON.parse(await a.downloads.at(-1).blob.text());
  const imported=app();await imported.change('#import-file',{files:[{text:async()=>JSON.stringify(backup)}],value:'backup.json'});
  assert.equal(imported.state.accountChecks[a.current().id],'yes');
  assert.match(imported.reportSummaryText(imported.current()),/ACCOUNT CHECK:.*none have been entered/);
  const old=app({scenarios:[model.baseScenario()],accountChecks:{'base-plan':'bad',other:'yes'}});
  assert.deepEqual(structuredClone(old.state.accountChecks),{});assert.doesNotMatch(old.reportSummaryText(old.current()),/ACCOUNT CHECK:/);
});

test('the result answer and its primary action precede notices, dollar controls and preview sales copy',async()=>{
  const a=app(),r=await completeCachedRun(a);await a.click('account-answer',{answer:'unsure'});
  const html=a.results(),answer=html.indexOf('Am I on track?'),action=html.indexOf('class="primary" data-action="compare-lower-returns"');
  assert.ok(answer>=0&&action>answer);
  for(const marker of ['Account check needs review','Calculated','Show amounts in','result-hero'])assert.ok(html.indexOf(marker)>action,marker);
  assert.match(html.slice(answer,action),/too few to/);assert.equal((html.match(/class="primary" data-action="compare-lower-returns"/g)||[]).length,1);
  assert.equal(a.state.results.get(a.current().id),r,'An account note does not invalidate saved results');
});

test('reports begin with a readable summary, keep exact assumptions in a collapsed appendix, and export either scope',async()=>{
  const a=app(),s=a.current(),r=await completeCachedRun(a);
  const html=a.reports();assert.ok(html.indexOf('Plan summary')<html.indexOf('Detailed assumptions and results'));
  assert.match(html,/<details class="card report-details"><summary>/);
  const summary=a.reportSummaryText(s,r),full=a.reportText(s,r);
  assert.match(summary,/Simulated lifetimes without a shortfall: .*preview only/);
  assert.match(summary,/Monthly costs before Medicare and taxes:/);assert.doesNotMatch(summary,/ALL ASSUMPTIONS|EmpiricalAgeDecline/);
  assert.match(full,/DETAILED ASSUMPTIONS AND RESULTS[^]*ALL ASSUMPTIONS/);
  assert.match(full,/Pre-retirement return average %: 13\.3%/);assert.doesNotMatch(full,/EmpiricalAgeDecline/);
  await a.click('download-summary');assert.equal(a.downloads.at(-1).name,'retirement-summary.txt');assert.equal(await a.downloads.at(-1).blob.text(),summary);
  await a.click('download-report');assert.equal(a.downloads.at(-1).name,'retirement-report.txt');assert.match(await a.downloads.at(-1).blob.text(),/ALL ASSUMPTIONS/);
});

test('temporary Roth histories require review consistently in the summary and appendix for both people',()=>{
  const s=model.baseScenario();s.household.filingStatus='Married';s.household.separatePeople=true;s.spouseAccounts.roth=10000;
  const sources={'rothHistory.contributionBasis':'Estimated','rothHistory.firstContributionYear':'Estimated','spouseRothHistory.contributionBasis':'Entered','spouseRothHistory.firstContributionYear':'Estimated'};
  const a=app({scenarios:[s],inputSources:{[s.id]:sources}}),plan=a.current();
  assert.match(a.reportSummaryText(plan),/Your Roth history needs review: Yes — temporary assumptions/);
  assert.match(a.reportSummaryText(plan),/Spouse Roth history needs review: Yes — temporary assumptions/);
  assert.match(a.reportDetailsText(plan),/Roth history needs review: Yes — temporary assumptions/);
  assert.match(a.reportDetailsText(plan),/Spouse Roth history needs review: Yes — temporary assumptions/);
  assert.match(a.reportSummaryText(plan),/Spouse retirement:/);
  for(const prefix of ['rothHistory','spouseRothHistory'])for(const key of ['contributionBasis','firstContributionYear'])a.state.inputSources[plan.id][prefix+'.'+key]='Entered';
  assert.match(a.reportSummaryText(plan),/Your Roth history needs review: No — contribution/);
  plan.rothHistory.needsReview=true;assert.match(a.reportDetailsText(plan),/Roth history needs review: Yes — migrated/);
  plan.accounts.roth=0;assert.match(a.reportSummaryText(plan),/Your Roth history needs review: No — no Roth/);
  plan.household.filingStatus='Single';assert.doesNotMatch(a.reportSummaryText(plan),/Spouse/);
});

test('unfinished summaries label unknown personal values and escape names in the readable report',async()=>{
  const a=app();await a.click('start-plan');const s=a.current();s.name='<img src=x onerror=alert(1)>';
  a.state.inputSources[s.id]['household.retirementDate']='Unknown';
  const summary=a.reportSummaryText(s);
  assert.match(summary,/Your retirement: Unknown/);assert.match(summary,/Savings today: Unknown/);
  assert.match(summary,/Monthly costs before Medicare and taxes: Unknown/);
  assert.match(summary,/Your Roth history needs review: Yes — Roth balance is unknown/);
  assert.doesNotMatch(a.reports(),/<img src=x/);assert.match(a.reports(),/&lt;img/);
});


async function confirmBudgetPaymentChoices(a){
  for(const [i,m] of a.current().budget.monthlyBudgets.entries())if(!m.includedPayments&&!model.SEPARATE_COSTS.some(([,key])=>m.adjustments?.[key]>0))await a.click('month-costs-excluded',{index:String(i)});
  await a.change('#main',{dataset:{costConfirm:''},checked:true});
}

test('individual guidance uses own dates and accounts, while couple guidance retains shared-date rules',()=>{
  for(const filingStatus of ['Single','HeadOfHousehold','Married'])for(const separatePeople of [false,true]){
    const s=model.prepareCalendarScenario(model.baseScenario());s.household.filingStatus=filingStatus;s.household.separatePeople=separatePeople;
    const a=app({scenarios:[s],selectedId:s.id});a.state.guided=true;a.state.basicSetup=true;a.state.setupSection=0;
    const household=a.setup();a.state.setupSection=1;const balances=a.setup();
    if(filingStatus==='Married'){
      assert.match(household,separatePeople?/earlier retirement date/:/One retirement date for both of you/);
      assert.match(balances,separatePeople?/Spouse retirement accounts/:/Combined household balances/);
    }else{
      assert.match(household,/Your retirement date\. Future savings deposits stop when you retire/);
      assert.doesNotMatch(household,/both of you|earlier retirement date/);
      assert.doesNotMatch(balances,/Enter spouse accounts|Enter your spouse’s Roth|Combined household balances|Household · Shared balances/);
      assert.match(balances,/Your (?:current account balances|taxable investments)/);
    }
  }
});

test('monthly payment answers survive reload and backup, including a selected payment with no amount',async()=>{
  const a=app();a.state.view='budget';await a.change('#main',{dataset:{month:'0',part:'checking'},value:'5000'});
  await a.click('month-costs-excluded',{index:'0'});await a.change('#main',{dataset:{costConfirm:''},checked:true});
  assert.equal(a.budgetCostsReviewed(),true);
  const restored=app(a.saved());restored.state.view='budget';assert.doesNotMatch(restored.budgetCostCheck(),/Choose the included payments or/);
  await restored.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(restored.budgetCostsReviewed(),true);
  await restored.change('#main',{dataset:{monthCost:'rent',index:'0'},checked:true});
  const unfinished=app(restored.saved());unfinished.state.view='budget';
  assert.match(unfinished.budgetCostCheck(),/amount greater than zero/);
  await unfinished.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(unfinished.budgetCostsReviewed(),false);
  await unfinished.change('#main',{dataset:{month:'0',part:'rent'},value:'1200'});
  await unfinished.click('export-backup');const backup=JSON.parse(await unfinished.downloads.at(-1).blob.text());
  const imported=app();await imported.change('#import-file',{files:[{text:async()=>JSON.stringify(backup)}],value:'backup.json'});imported.state.view='budget';
  await imported.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(imported.budgetCostsReviewed(),true);
  assert.equal(imported.current().budget.monthlyBudgets[0].includedPayments.rent,true);
  await imported.click('new-scenario');assert.equal(imported.current().budget.monthlyBudgets[0].includedPayments.rent,true);
});

test('deduction reconciliation supports mixed months and uses only the latest twelve',async()=>{
  const s=model.baseScenario();s.budget.monthlyBudgets=[{month:'2025-09',creditCardBills:[{monthlyAmount:9000}]}];
  for(let i=0;i<12;i++)s.budget.monthlyBudgets.push({month:`2025-${String(10+i).padStart(2,'0')}`,creditCardBills:[{monthlyAmount:5000}],adjustments:{rent:i?0:1200},includedPayments:{mortgage:false,rent:i===0,healthcare:false}});
  s.budget.monthlyBudgets.forEach((m,i)=>m.month=model.addCalendarMonths('2025-09-01',i).slice(0,7));
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='budget';
  assert.match(a.budgetCostCheck(),/the 12 months/);assert.match(a.budgetCostCheck(),/\$100\.00 \/ month/);
  await a.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(a.budgetCostsReviewed(),true,'An unused old month does not require another answer');
  await a.click('apply-budget');assert.equal(a.current().spending.annualBaseSpending,58800);
});

function claimingTargets(){return {frontier:{kind:'claiming',targetReadiness:.8,simulationCount:200,searchStartAge:62,searchEndAge:70,spendingSearchLimit:250000,points:Array.from({length:9},(_,i)=>({age:62+i,annualSpending:80000+i*5000,readiness:.8,tested:true,simulationCount:200}))}};}

test('claiming target search uses its own worker and slider, then applies and saves claiming age plus spending',async()=>{
  const records=new Map(),a=app(null,{resultRecords:records});a.state.access.tier='pro';const s=a.current();s.numberOfSimulations=500;s.simulationPathsCustomized=true;
  s.household.filingStatus='Married';s.household.separatePeople=true;s.household.spouseRetirementDate=model.addCalendarMonths(s.household.spouseBirthday,68*12);s.socialSecurity.spouseClaimAge=69;
  const before=structuredClone(s);seedExploration(a);
  assert.match(a.lab(),/Goal finder/);assert.match(a.lab(),/<option value="claiming" >Social Security claiming age &amp; spending/);
  const pending=a.runDecision(true),worker=a.workers.at(-1);assert.equal(worker.data.task,'claim-decision');assert.equal(worker.data.options,undefined,'The original 80% target is the default');
  worker.onmessage({data:{type:'result',result:claimingTargets()}});await pending;
  assert.ok(a.state.decision);assert.ok(a.state.claimDecision);assert.equal(a.state.claimFrontierSelection,5);
  assert.deepEqual(structuredClone(s),before);const stored=a.stored();
  a.element('#main').listeners.input({target:{dataset:{frontierSlider:'claiming'},value:'1'}});
  assert.match(a.element('#claim-frontier-selection').innerHTML,/\$85,000/);assert.equal(a.state.claimFrontierSelection,1);assert.equal(a.stored(),stored);
  a.state.results.set(s.id,{old:true});records.set(s.id,{old:true});await a.click('apply-claim-frontier-target');
  assert.equal(s.socialSecurity.claimAge,63);assert.equal(s.spending.annualBaseSpending,85000);
  assert.equal(s.socialSecurity.spouseClaimAge,69);assert.deepEqual(structuredClone(s.household),before.household);assert.deepEqual(structuredClone(s.accounts),before.accounts);
  assert.equal(s.numberOfSimulations,500);assert.equal(s.simulationPathsCustomized,true);assert.equal(s.seed,before.seed);
  assert.equal(a.saved().inputSources[s.id]['socialSecurity.claimAge'],'Estimated');assert.equal(a.saved().inputSources[s.id]['spending.annualBaseSpending'],'Estimated');
  assertCleared(a);assert.equal(a.state.results.has(s.id),false);assert.equal(records.has(s.id),false);assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,5);
  assert.match(a.state.message,/Social Security claiming age and base spending target applied/);
  const restored=app(a.saved());assert.equal(restored.current().socialSecurity.claimAge,63);assert.equal(restored.current().spending.annualBaseSpending,85000);assert.equal(restored.current().numberOfSimulations,500);
});

test('claiming searches require Pro and ignore results for plans edited during the calculation',async()=>{
  const a=app();assert.doesNotMatch(a.lab(),/data-action="run-lab-goal"/);await a.runDecision(true);assert.equal(a.workers.length,0);
  a.state.access.tier='pro';a.state.claimDecision=claimingTargets();seedExploration(a);
  const pending=a.runDecision(true),worker=a.workers.at(-1);
  await a.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'900000'});assertCleared(a);
  worker.onmessage({data:{type:'result',result:claimingTargets()}});await pending;
  assertCleared(a);assert.equal(a.current().accounts.pretax,900000);assert.equal(a.state.busy,false);
});

test('claiming apply rejects untested points and ignores actions while another calculation is running',async()=>{
  const a=app(),before=structuredClone(a.current());a.state.access.tier='pro';a.state.claimDecision=claimingTargets();a.state.labUi.goal.kind='claiming';
  for(const point of a.state.claimDecision.frontier.points)point.tested=false;
  assert.doesNotMatch(a.lab(),/data-action="apply-claim-frontier-target"/);await a.click('apply-claim-frontier-target');
  a.state.claimDecision=claimingTargets();a.state.busy=true;assert.match(a.lab(),/data-action="apply-claim-frontier-target" disabled/);
  await a.click('apply-claim-frontier-target');assert.deepEqual(structuredClone(a.current()),before);
});

function savingsTargetsResult(){return {frontier:{kind:'savings',targetReadiness:.8,simulationCount:200,searchStartAge:60,searchEndAge:67,fixedAnnualSpending:75000,savingsSearchLimit:1000000,allocation:[{label:'Cash',share:1}],points:[{age:60,annualSavings:40000,readiness:.81,tested:true,simulationCount:200},{age:62,annualSavings:30000,readiness:.8,tested:true,simulationCount:200}]}};}
test('savings targets use a separate worker and slider, apply savings plus age and retain spending and employer/spouse inputs',async()=>{
  const records=new Map(),a=app(null,{resultRecords:records});a.state.access.tier='pro';const s=a.current();s.numberOfSimulations=500;s.simulationPathsCustomized=true;
  s.contributions.pretax=12000;s.contributions.roth=6000;s.contributions.employerPretax=3000;s.contributions.annualIncrease=.03;s.spouseContributions.cash=5000;
  const before=structuredClone(s);seedExploration(a);a.state.claimDecision=claimingTargets();
  assert.match(a.lab(),/Retirement age &amp; annual savings/);
  const pending=a.runDecision('savings'),worker=a.workers.at(-1);assert.equal(worker.data.task,'savings-decision');
  worker.onmessage({data:{type:'result',result:savingsTargetsResult()}});await pending;
  assert.ok(a.state.decision);assert.ok(a.state.claimDecision);assert.ok(a.state.savingsDecision);assert.deepEqual(structuredClone(s),before);
  a.element('#main').listeners.input({target:{dataset:{frontierSlider:'savings'},value:'1'}});
  assert.equal(a.state.savingsFrontierSelection,1);assert.match(a.element('#savings-frontier-selection').innerHTML,/30,000/);
  a.state.results.set(s.id,{old:true});records.set(s.id,{old:true});await a.click('apply-savings-frontier-target');
  assert.equal(model.primaryRetirementAge(s),62);assert.equal(s.contributions.pretax,20000);assert.equal(s.contributions.roth,10000);
  assert.equal(s.contributions.employerPretax,3000);assert.equal(s.contributions.annualIncrease,.03);assert.deepEqual(structuredClone(s.spouseContributions),before.spouseContributions);
  assert.deepEqual(structuredClone(s.spending),before.spending);assert.deepEqual(structuredClone(s.budget),before.budget);assert.deepEqual(structuredClone(s.socialSecurity),before.socialSecurity);
  assert.equal(s.numberOfSimulations,500);assert.equal(s.seed,before.seed);assert.equal(a.saved().inputSources[s.id]['contributions.pretax'],'Estimated');assert.equal(a.saved().inputSources[s.id]['contributions.roth'],'Estimated');
  assertCleared(a);assert.equal(records.has(s.id),false);assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,5);
  const restored=app(a.saved());assert.equal(model.primaryRetirementAge(restored.current()),62);assert.equal(restored.current().contributions.pretax,20000);assert.equal(restored.current().spending.annualBaseSpending,before.spending.annualBaseSpending);
});
test('savings targets require Pro, ignore edited-plan results and refuse untested or busy apply',async()=>{
  const a=app();assert.doesNotMatch(a.lab(),/data-action="run-lab-goal"/);await a.runDecision('savings');assert.equal(a.workers.length,0);
  a.state.access.tier='pro';const pending=a.runDecision('savings'),worker=a.workers.at(-1);
  await a.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'900000'});
  worker.onmessage({data:{type:'result',result:savingsTargetsResult()}});await pending;assertCleared(a);
  const before=structuredClone(a.current());a.state.savingsDecision=savingsTargetsResult();a.state.savingsDecision.frontier.points.forEach(p=>p.tested=false);
  assert.doesNotMatch(a.lab(),/data-action="apply-savings-frontier-target"/);await a.click('apply-savings-frontier-target');
  a.state.savingsDecision=savingsTargetsResult();a.state.busy=true;await a.click('apply-savings-frontier-target');assert.deepEqual(structuredClone(a.current()),before);
});

test('applying savings marks only the changed employer Roth deposit field Estimated and retains separate spouse timing',async()=>{
  const a=app(),s=a.current();a.state.access.tier='pro';s.household.filingStatus='Married';s.household.separatePeople=true;s.household.spouseRetirementDate=model.addCalendarMonths(s.household.spouseBirthday,68*12);
  s.employerRothAccounts=[{...model.employerRothDefaults(),id:'target-roth',name:'Work Roth',owner:'you',annualContribution:10000,annualEmployerContribution:2000,annualIncrease:.04}];
  const before=structuredClone(s);a.state.savingsDecision=savingsTargetsResult();a.state.savingsFrontierSelection=1;
  await a.click('apply-savings-frontier-target');assert.equal(s.employerRothAccounts[0].annualContribution,30000);assert.equal(s.employerRothAccounts[0].annualEmployerContribution,2000);assert.equal(s.employerRothAccounts[0].annualIncrease,.04);
  assert.equal(a.saved().inputSources[s.id]['employerRothAccounts.0.annualContribution'],'Estimated');assert.notEqual(a.saved().inputSources[s.id]['employerRothAccounts'],'Estimated');
  assert.equal(s.household.spouseRetirementDate,before.household.spouseRetirementDate);assert.deepEqual(structuredClone(s.spouseContributions),before.spouseContributions);assert.deepEqual(structuredClone(s.spending),before.spending);
});

test('comparison sets are Pro, persist with the plan, survive reload, copies and backups, and reject malformed entries',async()=>{
  const free=app();free.state.view='lab';assert.match(free.lab(),/\+ New comparison set · Pro/);await free.click('lab-new-set');assert.match(free.state.message,/part of Pro/);
  await free.click('lab-add',{preset:'claim-70'});await free.click('lab-add',{preset:'spend-less'});assert.match(free.state.message,/Free accounts compare one what-if/);
  assert.equal(free.labViewModel().set.whatIfs.length,1);
  const a=app();a.state.access.tier='pro';const s=a.current();a.state.view='lab';
  await a.click('lab-new-set');const second=a.labViewModel().set;assert.equal(second.name,'Comparison set 2');assert.equal(second.whatIfs.length,0);
  await a.change('#main',{dataset:{labSetName:''},value:'  Tax strategy  '});assert.equal(second.name,'Tax strategy');
  await a.click('lab-add',{preset:'roth'});await a.change('#main',{dataset:{labWhatifName:''},value:'Convert to 22%'});
  const saved=a.saved().labSets[s.id];assert.equal(saved.sets.length,2);assert.equal(saved.activeSet,second.id);assert.equal(saved.sets[1].whatIfs[0].name,'Convert to 22%');
  const reloaded=app(a.saved());reloaded.state.access.tier='pro';assert.equal(reloaded.labViewModel().set.name,'Tax strategy');
  await a.click('export-backup');const backup=JSON.parse(await a.downloads.at(-1).blob.text());assert.deepEqual(backup.labSets[s.id],saved);
  const imported=app();await imported.change('#import-file',{files:[{text:async()=>JSON.stringify(backup)}],value:'backup.json'});assert.deepEqual(structuredClone(imported.state.labSets[s.id]),saved);
  await a.click('new-scenario');const copied=a.saved().labSets[a.current().id];assert.equal(copied.sets.length,2);assert.notEqual(copied.sets[0].id,saved.sets[0].id);assert.deepEqual(copied.sets[1].whatIfs[0].changes,saved.sets[1].whatIfs[0].changes);
  await a.click('select-scenario',{id:s.id});a.state.view='lab';await a.click('lab-select-set',{set:saved.sets[1].id});
  await a.click('lab-delete-set');assert.equal(a.saved().labSets[s.id].sets.length,1);
  const bad=app({scenarios:[model.baseScenario()],labSets:{'base-plan':{activeSet:'x',sets:[{id:'a',name:5,whatIfs:[{id:'w',name:'',changes:{annualBaseSpending:1000,unknownLever:1}},'bad']},'bad',{id:'a'}]},missing:{sets:[]}}});
  const clean=bad.state.labSets['base-plan'];assert.equal(clean.sets.length,1);assert.equal(clean.activeSet,'a');assert.deepEqual(structuredClone(clean.sets[0].whatIfs[0].changes),{annualBaseSpending:1000});assert.equal(clean.sets[0].name,'Comparison set');assert.equal(bad.state.labSets.missing,undefined);
});

test('stress tests and the sensitivity ranking run Pro comparisons on the same paths and report each result',async()=>{
  const a=app();a.state.access.tier='pro';a.current().numberOfSimulations=40;a.current().simulationPathsCustomized=true;a.state.view='lab';
  assert.match(a.lab(),/data-action="run-lab-stress"/);assert.match(a.lab(),/data-action="run-lab-sensitivity"/);
  await settle(a,a.runLab());const target=labRows(a).filter(r=>!r.baseline).sort((x,y)=>y.result.successProbability-x.result.successProbability)[0];
  const stressed=[];await settle(a,a.runLabStress(),w=>{stressed.push(w.data);return runSimulation(w.data.scenario,()=>{},w.data.options);});
  assert.equal(stressed.length,10);assert.deepEqual(stressed[0].options.stress,{marketDrop:.3});assert.equal(stressed[4].scenario.market.stockMeanReturn,.07);assert.equal(stressed[6].options.stress.forceCare,true);
  assert.deepEqual(stressed.filter((x,i)=>i%2===1).map(x=>x.scenario.socialSecurity.claimAge===70||x.scenario.household.retirementDate!==a.current().household.retirementDate||x.scenario.spending.annualBaseSpending!==a.current().spending.annualBaseSpending),[true,true,true,true,true]);
  const vm=a.labViewModel();assert.equal(vm.stress.target,target.id);assert.equal(vm.stress.stale,false);assert.ok(vm.stress.rows.every(r=>r.results.baseline&&r.results[target.id]));
  assert.match(a.lab(),/Stocks fall 30% when you retire/);
  const runs=[];await settle(a,a.runLabSensitivity(),w=>{runs.push(w.data.scenario);return runSimulation(w.data.scenario,()=>{},w.data.options);});
  assert.equal(runs.length,15);assert.ok(runs.every(x=>x.numberOfSimulations===40));
  const t=a.labViewModel().sensitivity;assert.equal(t.rows.length,7);assert.ok(t.rows.every(r=>r.low===null||Number.isFinite(r.low)));
  const spending=t.rows.find(r=>r.key==='spending');assert.ok(spending.high>=spending.low);
  const before=a.labViewModel().set.whatIfs.length;await a.click('lab-remove',{id:a.labViewModel().set.whatIfs[0].id});
  await a.click('lab-sensitivity-whatif',{input:'claim',side:'high'});assert.equal(a.labViewModel().set.whatIfs.length,before);assert.deepEqual(structuredClone(a.labViewModel().set.whatIfs.at(-1).changes),{claimAge:70});
  await a.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:'90000'});assert.equal(a.state.labResults,null,'Plan edits clear Plan Lab results');
});

test('free accounts see locked Pro tools; the goal finder passes its readiness target and adds a tested pair as a what-if',async()=>{
  const free=app();free.state.view='lab';const html=free.lab();
  for(const text of ['Rank which inputs change your readiness','Test a market crash','Find the retirement age, spending'])assert.ok(html.includes(text),text);
  await free.runLabStress();await free.runLabSensitivity();assert.equal(free.workers.length,0);
  const a=app();a.state.access.tier='pro';a.state.view='lab';
  a.element('#main').listeners.input({target:{dataset:{labGoal:'target'},value:'90'}});await a.change('#main',{dataset:{labGoal:'kind'},value:'claiming'});
  const pending=a.click('run-lab-goal'),worker=await nextWorker(a,0);assert.equal(worker.data.task,'claim-decision');assert.equal(worker.data.options.targetReadiness,.9);
  worker.onmessage({data:{type:'result',result:claimingTargets()}});await pending;
  assert.match(a.lab(),/data-action="lab-frontier-whatif" data-kind="claiming"/);
  const count=a.labViewModel().set.whatIfs.length;await a.click('lab-remove',{id:a.labViewModel().set.whatIfs[0].id});
  await a.click('lab-frontier-whatif',{kind:'claiming'});const w=a.labViewModel().set.whatIfs.at(-1);
  assert.equal(a.labViewModel().set.whatIfs.length,count);assert.equal(w.changes.claimAge,67);assert.equal(w.changes.annualBaseSpending,105000);
});

test('the goal finder optimizes stock allocation, shows its progress, adds it as a what-if and applies it',async()=>{
  const a=app();a.state.access.tier='pro';a.state.view='lab';
  await a.change('#main',{dataset:{labGoal:'kind'},value:'allocation'});
  const page=a.lab();assert.match(page,/Find the best allocation/);assert.doesNotMatch(page,/id="lab-goal-target"/,'The optimizer has no readiness target');
  const pending=a.click('run-lab-goal'),worker=await nextWorker(a,0);assert.equal(worker.data.task,'allocation-decision');assert.equal(worker.data.options,undefined);
  worker.onmessage({data:{type:'progress',phase:'search',pass:1,maxPasses:3,band:2,bands:6,bandLabel:'35–40×',tested:25,checkPaths:2000}});
  assert.equal(a.element('#busy-detail').textContent,'Pass 1 of up to 3 · 35–40× band · 25 schedules tested');
  const keys=['stockUnder30x','stock30xTo35x','stock35xTo40x','stock40xTo45x','stock45xTo50x','stock50xOrMore'],current=structuredClone(a.current().postRetirementAllocation),suggested={...current,stock30xTo35x:1,stock35xTo40x:1};
  const result={searchPaths:500,checkPaths:2000,nearTie:.01,tested:81,passes:2,changed:true,improved:true,current:{allocation:current,readiness:.98,medianEndingBalance:2600000},suggested:{allocation:suggested,readiness:.98,medianEndingBalance:2700000},searchResults:{},bands:keys.map((key,i)=>({key,label:key,current:current[key],suggested:suggested[key],affectedResults:i<3}))};
  worker.onmessage({data:{type:'result',result}});await pending;
  const html=a.lab();assert.match(html,/data-action="apply-allocation-target"/);assert.match(html,/keeps readiness within a point and leaves a larger median balance/);
  await a.click('lab-allocation-whatif');const w=a.labViewModel().set.whatIfs.at(-1);
  assert.equal(w.name,'Optimized stock allocation');assert.deepEqual(structuredClone(w.changes),{stockAllocation:suggested});
  await a.click('apply-allocation-target');const s=a.current();
  assert.deepEqual(structuredClone(s.postRetirementAllocation),suggested);
  assert.equal(a.state.inputSources[s.id]['postRetirementAllocation.stock30xTo35x'],'Estimated');assert.equal(a.state.inputSources[s.id]['postRetirementAllocation.stockUnder30x'],undefined);
  assertCleared(a);assert.equal(a.state.view,'setup');
});

test('allocation searches enforce Pro access and discard results after canceling or editing the plan',async()=>{
  const free=app();free.state.labUi.goal.kind='allocation';await free.click('run-lab-goal');
  assert.equal(free.workers.length,0);assert.match(free.state.message,/Pro/);
  for(const action of ['cancel','edit','switch']){
    const a=app();a.state.access.tier='pro';a.state.labUi.goal.kind='allocation';
    const pending=a.click('run-lab-goal'),worker=await nextWorker(a,0);
    if(action==='cancel'){await a.click('cancel-calculation');assert.equal(worker.terminated,true);}
    else{
      if(action==='edit')await a.change('#main',{dataset:{field:'market.stockMeanReturn',type:'percent'},value:'10'});
      else await a.click('select-scenario',{id:a.state.scenarios[1].id});
      worker.onmessage({data:{type:'result',result:{changed:true,suggested:{allocation:{stockUnder30x:0}}}}});
    }
    await pending;assert.equal(a.state.allocationDecision,null);assert.equal(a.state.busy,false);
  }
});

test('reopening an optimizer what-if retains every fractional band while another band is edited',async()=>{
  const a=app();a.state.access.tier='pro';a.state.view='lab';
  const allocation={...a.current().postRetirementAllocation,stockUnder30x:.5001,stock35xTo40x:1/3};
  a.state.allocationDecision={changed:true,suggested:{allocation}};
  await a.click('lab-allocation-whatif');const w=a.labViewModel().set.whatIfs.at(-1);
  await a.click('lab-edit',{id:w.id});
  await setLever(a,'stockAllocation',{stock30xTo35x:'55.5'});
  assert.deepEqual(structuredClone(w.changes.stockAllocation),{...allocation,stock30xTo35x:.555});
  assert.deepEqual(structuredClone(a.saved().labSets[a.current().id].sets[0].whatIfs.at(-1).changes.stockAllocation),{...allocation,stock30xTo35x:.555});
});

test('the live estimate runs separately for Pro edits and CSV and report downloads include every compared plan',async()=>{
  const a=app();a.state.access.tier='pro';a.current().numberOfSimulations=30;a.current().simulationPathsCustomized=true;a.state.view='lab';
  const id=a.labViewModel().set.whatIfs[1].id;await a.click('lab-edit',{id});
  const live=await nextWorker(a,0);assert.equal(live.data.scenario.numberOfSimulations,200);assert.equal(live.data.options.captureMetrics,undefined);assert.equal(a.state.busy,false);
  await settle(a,(async()=>{for(let i=0;i<50&&a.state.labLive.status!=='done';i++)await tick();})());
  assert.equal(a.labViewModel().live.status,'done');assert.match(a.lab(),/≈ /);
  await setLever(a,'claimAge',{age:'68'});assert.equal(a.labViewModel().live.status,'running','A new edit replaces the finished estimate');
  await settle(a,a.runLab(),w=>runSimulation(w.data.scenario,()=>{},w.data.options||{}));await a.click('lab-download-csv');
  const csv=await a.downloads.at(-1).blob.text();assert.equal(a.downloads.at(-1).name,'plan-lab-comparison.csv');
  assert.match(csv,/^Plan,Changes,Paths,Readiness/);assert.match(csv,/Current plan/);assert.match(csv,/Spend 5% less/);assert.match(csv,/\nAge,Current plan median balance/);
  await a.click('lab-download-report');const report=await a.downloads.at(-1).blob.text();assert.match(report,/PLAN LAB COMPARISON/);assert.match(report,/Median lifetime federal income tax/);
  const free=app();await free.click('lab-download-csv');assert.equal(free.downloads.length,0);
});
test('inputs added through Plan Lab appear in review and reports and can be removed there',async()=>{
  const a=app(),s=a.current(),end=Math.floor(model.primaryRetirementAge(s))+3;await a.click('lab-add');
  await setLever(a,'partTimeIncome',{annualNet:'24,000',endAge:String(end)});await setLever(a,'withdrawalOrder',{order:'TaxableFirst'});
  const w=a.labViewModel().editing;await a.click('lab-apply',{id:w.id});
  assert.deepEqual(structuredClone(s.partTimeIncome),{annualNet:24000,endAge:end});assert.equal(s.withdrawalStrategy.withdrawalOrder,'TaxableFirst');
  assert.match(a.setup(),/Added in Plan Lab/);assert.match(a.setup(),/Work part-time for \$24,000 a year until/);
  assert.match(a.reportSummaryText(s),/Added in Plan Lab/);assert.match(model.scenarioEngineVersion(s),/plan-lab-v2/);
  await a.click('remove-lab-input',{key:'partTimeIncome'});assert.deepEqual(structuredClone(s.partTimeIncome),{annualNet:0,endAge:0});
  await a.click('remove-lab-input',{key:'withdrawalOrder'});assert.equal(s.withdrawalStrategy.withdrawalOrder,'Standard');
  assert.doesNotMatch(a.setup(),/Added in Plan Lab/);assert.equal(model.usesPlanLabInputs(s),false);assert.equal(app(a.saved()).current().withdrawalStrategy.withdrawalOrder,'Standard');
});

test('all lever draft fields survive a validation error and live estimate completion',async()=>{
  const a=app();a.state.access.tier='pro';a.state.view='lab';
  await a.click('lab-edit',{id:a.labViewModel().set.whatIfs[1].id});
  const live=await nextWorker(a,0);assert.ok(live);
  await a.click('lab-lever-edit',{lever:'oneTimeExpenses'});
  const draft={label0:'Roof & gutters',age0:'2.5',amount0:'30,000',label1:'Car',age1:'80',amount1:'0',label2:'',age2:'',amount2:''};
  for(const [field,value] of Object.entries(draft))leverInput(a,'oneTimeExpenses',field,value);
  await a.click('lab-lever-save',{lever:'oneTimeExpenses'});assert.match(a.state.labUi.leverError,/Expense 1/);
  const assertDraft=()=>{
    const html=a.element('#main').innerHTML;
    for(const [field,value] of Object.entries(draft))assert.ok(html.includes(`id="lab-oneTimeExpenses-${field}"`)&&new RegExp(`id="lab-oneTimeExpenses-${field}"[^>]*value="${value.replace('&','&amp;')}"`).test(html),field);
  };
  assertDraft();
  await settle(a,(async()=>{for(let i=0;i<50&&a.state.labLive.status!=='done';i++)await tick();})(),w=>runSimulation({...w.data.scenario,numberOfSimulations:20}));
  assert.equal(a.state.labLive.status,'done');assertDraft();
  leverInput(a,'oneTimeExpenses','age0','75');await a.click('lab-lever-save',{lever:'oneTimeExpenses'});
  assert.deepEqual(structuredClone(a.labViewModel().editing.changes.oneTimeExpenses),[{label:'Roof & gutters',age:75,amount:30000},{label:'Car',age:80,amount:0}]);
});

test('malformed saved lever values do not block loading or importing plans and comparisons',async()=>{
  const s=model.prepareCalendarScenario(model.baseScenario(),{needsReview:false});
  const saved={scenarios:[s],selectedId:s.id,labSets:{[s.id]:{activeSet:'set',sets:[{id:'set',name:'Imported',whatIfs:[{id:'w',name:'Expenses',changes:{oneTimeExpenses:null,partTimeIncome:null,rothConversion:null,cashFirst:[],homePlan:'bad',annualBaseSpending:60000}}]}]}}};
  const restored=app(saved),imported=app();
  await imported.change('#import-file',{files:[{text:async()=>JSON.stringify(saved)}],value:'backup.json'});
  for(const a of [restored,imported]){
    a.state.view='lab';assert.match(a.lab(),/Imported/);assert.match(a.lab(),/Spend \$60,000 a year/);
    const w=a.labViewModel().set.whatIfs[0];assert.deepEqual(structuredClone(w.changes),{annualBaseSpending:60000});
    await a.click('lab-edit',{id:w.id});await a.click('lab-lever-edit',{lever:'oneTimeExpenses'});assert.match(a.lab(),/Expense 1 name/);
  }
});

test('hiding all chart plans keeps the legend usable for every measure and reselecting restores the chart',async()=>{
  const a=app();a.state.view='lab';await a.click('lab-add',{preset:'spend-less'});
  await settle(a,a.runLab(),w=>runSimulation({...w.data.scenario,numberOfSimulations:20},()=>{},w.data.options));
  const rows=labRows(a),chart=()=>a.lab().match(/<section class="card lab-chart"[\s\S]*?<\/section>/)[0];
  for(const row of rows)await a.change('#main',{dataset:{labSeries:row.id},checked:false});
  for(const measure of ['funded','median','tough','tax']){
    await a.click('lab-measure',{measure});const html=chart();
    assert.match(html,/Select at least one plan to show the chart/);assert.equal((html.match(/data-lab-series=/g)||[]).length,rows.length);
    assert.doesNotMatch(html,/NaN|Infinity|<svg|id="lab-chart-age"/);
    await a.change('#main',{dataset:{labSeries:'baseline'},checked:true});assert.match(chart(),/<svg/);assert.match(chart(),/id="lab-chart-age"/);
    await a.change('#main',{dataset:{labSeries:'baseline'},checked:false});
  }
  const vm=a.labViewModel();vm.hidden=new Set();vm.rows=vm.rows.map(row=>({...row,result:{...row.result,notFailedByAge:[]}}));vm.measure='funded';
  const noData=planLabView.labPage(vm,{head:'',notices:'',goal:''}).match(/<section class="card lab-chart"[\s\S]*?<\/section>/)[0];
  assert.match(noData,/No data is available for this measure/);assert.doesNotMatch(noData,/NaN|Infinity|<svg|id="lab-chart-age"/);
});
