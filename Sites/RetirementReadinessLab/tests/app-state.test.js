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
import {runSimulation} from '../dist/engine.js';

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

test('result-storage failures retain live results and saved inputs, and importing clears older cached calculations',async()=>{
  const a=app(null,{resultStorage:{fail:true}});a.state.view='results';
  await completeCachedRun(a);assert.ok(a.saved());assert.ok(a.state.results.get(a.current().id));
  assert.match(a.results(),/run again after reloading to regenerate results/);
  const resultRecords=new Map(),b=app(null,{resultRecords});await completeCachedRun(b);
  await b.change('#import-file',{files:[{text:async()=>JSON.stringify(b.saved())}],value:'backup.json'});
  assert.equal(resultRecords.size,0);assert.equal(b.state.results.size,0);
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

test('one-click lower-return screening uses paired scenarios, retains results and never raises a lower rate',async()=>{
  const a=app(),s=a.current();s.market.preRetirementMeanReturn=.04;s.market.stockMeanReturn=.133;
  const before=structuredClone(s),completed=runSimulation(s);completed.uxAssumptions=before;a.state.results.set(s.id,completed);
  assert.match(a.results(),/Compare with lower returns/);
  const pending=a.click('compare-lower-returns');
  for(let i=0;i<2;i++){
    const w=a.workers[i];assert.ok(w);
    assert.equal(w.data.scenario.market.preRetirementMeanReturn,.04);
    assert.equal(w.data.scenario.market.stockMeanReturn,i?.07:before.market.stockMeanReturn);
    assert.equal(w.data.scenario.market.stockStdDev,before.market.stockStdDev);
    w.onmessage({data:{type:'result',result:runSimulation(w.data.scenario)}});await Promise.resolve();
  }
  await pending;assert.equal(a.state.view,'lab');assert.equal(a.state.labResults.length,2);
  assert.deepEqual(structuredClone(a.current()),before);assert.equal(a.state.results.get(s.id),completed);
  await a.click('copy-comparison',{index:'1'});assert.equal(a.current().market.stockMeanReturn,.07);
  assert.equal(a.saved().scenarios.find(x=>x.id===before.id).market.stockMeanReturn,before.market.stockMeanReturn);
});

test('preview warning keeps its essential caution visible and expands the full explanation',()=>{
  const a=app(),s=a.current(),r=runSimulation(s);a.state.results.set(s.id,r);
  const html=a.results();
  assert.match(html,/10 lifetimes are too few to estimate readiness/);
  assert.match(html,/<details><summary>Why only 10 lifetimes\?<\/summary>/);
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

test('custom comparisons run independent date and spending changes without changing the base plan',async()=>{
  const a=app(),s=a.current(),before=structuredClone(s),date=model.addCalendarMonths(s.household.retirementDate,12);
  assert.match(a.lab(),/id="lab-monthly-spending"[^>]*value="6,250"/);assert.doesNotMatch(a.lab(),/value="NaN"/);
  const input=(key,value)=>a.element('#main').listeners.input({target:{dataset:{labInput:key},value}});
  input('retirementDate',date);input('monthlySpending','4000');
  const pending=a.runLab(true);
  for(let i=0;i<3;i++){
    const w=a.workers[i];assert.ok(w);
    if(i===1){assert.equal(w.data.scenario.household.retirementDate,date);assert.equal(w.data.scenario.spending.annualBaseSpending,before.spending.annualBaseSpending);}
    if(i===2){assert.equal(w.data.scenario.spending.annualBaseSpending,48000);assert.equal(w.data.scenario.household.retirementDate,before.household.retirementDate);}
    w.onmessage({data:{type:'result',result:runSimulation(w.data.scenario)}});await Promise.resolve();
  }
  await pending;assert.equal(a.state.labResults.length,3);assert.deepEqual(structuredClone(a.current()),before);
  await a.click('copy-comparison',{index:'2'});assert.equal(a.current().spending.annualBaseSpending,48000);
  assert.equal(a.saved().scenarios.find(x=>x.id===before.id).spending.annualBaseSpending,before.spending.annualBaseSpending);
});

test('custom comparisons reject incomplete or excessive amounts and edited drafts discard pending results',async()=>{
  for(const value of ['', '-1', '1e400', String(model.MAX_DOLLAR_AMOUNT)]){
    const a=app();a.element('#main').listeners.input({target:{dataset:{labInput:'monthlySpending'},value}});
    await a.runLab(true);assert.equal(a.workers.length,0,value);assert.match(a.state.message,/Error:/);
  }
  const a=app();a.element('#main').listeners.input({target:{dataset:{labInput:'monthlySpending'},value:'4000'}});
  const pending=a.runLab(true),w=a.workers[0];
  a.element('#main').listeners.input({target:{dataset:{labInput:'monthlySpending'},value:'5000'}});
  w.onmessage({data:{type:'result',result:runSimulation(w.data.scenario)}});await pending;
  assert.equal(a.state.labResults,null);assert.equal(a.state.busy,false);
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
  assert.match(restored.setup(),/Those employer Roth accounts are not supported/);
  assert.equal(restored.current().market.stockMeanReturn,before.market.stockMeanReturn);
  await a.click('input-none',{path:'accounts.roth'});
  assert.equal(a.saved().inputSources[before.id]['rothHistory.firstContributionYear'],'Entered');
  await a.click('roth-records-unknown',{prefix:'rothHistory'});
  assert.equal(a.saved().inputSources[before.id]['rothHistory.firstContributionYear'],'Entered','A zero account never needs history');
});

test('combined comparisons apply both changes, retain ownership and copies preserve source paths',async()=>{
  const a=app(),before=structuredClone(a.current()),date=model.addCalendarMonths(before.household.retirementDate,12);
  a.element('#main').listeners.input({target:{dataset:{labInput:'retirementDate'},value:date}});
  a.element('#main').listeners.input({target:{dataset:{labInput:'monthlySpending'},value:'4000'}});
  const pending=a.runLab(true,true);
  for(let i=0;i<4;i++){
    const w=a.workers[i];assert.ok(w);
    if(i===3){assert.equal(w.data.scenario.household.retirementDate,date);assert.equal(w.data.scenario.spending.annualBaseSpending,48000);assert.equal(w.data.scenario.household.spouseRetirementDate,before.household.spouseRetirementDate);}
    w.onmessage({data:{type:'result',result:runSimulation(w.data.scenario)}});await Promise.resolve();
  }
  await pending;assert.equal(a.state.labResults.length,4);assert.deepEqual(structuredClone(a.current()),before);
  await a.click('copy-comparison',{index:'3'});
  assert.equal(a.current().spending.annualBaseSpending,48000);assert.equal(a.current().household.retirementDate,date);
  assert.equal(a.current().numberOfSimulations,before.numberOfSimulations);
  assert.equal(a.saved().inputSources[a.current().id]['spending.annualBaseSpending'],'Entered');
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
  const context=vm.createContext({...model,...format,...guidance,...withdrawalsView,...growthHelper,...moneyInput,...planReview,...resultCaching,createResultCache:()=>({
      async load(id){return resultStorage.fail?null:structuredClone(resultRecords.get(id)||null);},
      async save(s,r,today){if(resultStorage.fail)return null;resultRecords.set(s.id,{id:s.id,version:1,fingerprint:resultCaching.resultFingerprint(s,today),result:structuredClone(r)});return s.id;},
      async remove(id){resultRecords.delete(id);},async clear(){resultRecords.clear();}
    }),structuredClone,Intl,URLSearchParams:params,Blob,URL:class extends URL{static createObjectURL(blob){const url='blob:test-'+downloadBlobs.size;downloadBlobs.set(url,blob);return url;}static revokeObjectURL(url){downloadBlobs.delete(url);}},console,Date:class extends Date{static now(){return clock.now;}},
    location,history:{replaceState(_state,_title,url){const next=new URL(url,'https://example.test');location.search=next.search;location.hash=next.hash;}},confirm,
    setTimeout(fn,delay){const id=++nextTimer;timers.set(id,{fn,at:clock.now+delay});return id;},clearTimeout(id){timers.delete(id);},
    sessionStorage:{getItem:key=>session.get(key)||null,setItem:(key,value)=>session.set(key,value)},socialState:()=>({...identity}),
    navigator:{locks},localStorage:{getItem:key=>key==='retirement-readiness-lab-sites-v1'?readStored():preferences.get(key)||null,setItem(key,value){if(key!=='retirement-readiness-lab-sites-v1'){preferences.set(key,value);return;}if(storage.fail)throw new Error('QuotaExceededError');if(Object.hasOwn(storage,'raw'))storage.raw=value;else stored=value;}},window:{addEventListener(name,handler){windowListeners[name]=handler;},scrollTo(){}},
    document,
    chartCard:()=>'',mountCharts(){},disposeCharts(){},initializeSocialAuth:()=>new Promise(()=>{}),fetch,authHeaders:async()=>({}),...socialActions,
    Worker:class{constructor(){workers.push(this);}postMessage(data){this.data=data;}terminate(){this.terminated=true;}},
  });
  const source=readFileSync(new URL('../dist/app.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replaceAll('import.meta.url',JSON.stringify(new URL('../dist/app.js',import.meta.url).href));
  vm.runInContext(source.replace('function render({preserveEditor=false}={}){','let renderCount=0;function render({preserveEditor=false}={}){renderCount++;'),context);
  const api=vm.runInContext('({state,resultRestoreReady,setup,run,runLab,runDecision,results,withdrawals,dashboard,lab,budget,budgetView,budgetSummary,budgetCostCheck,budgetCostsReviewed,enterBudget,scenarios,render,billingView,accountCheck,reports,reportText,reportSummaryText,reportDetailsText,current,persist,loadAccess,isPro,effectivePaths,syncAuthState,linkAccounts,renders:()=>renderCount})',context);
  return {...api,workers,element,timers,document,downloads,stored:readStored,storageChanged:()=>windowListeners.storage({key:'retirement-readiness-lab-sites-v1'}),beforeUnload:event=>windowListeners.beforeunload(event),advanceTime(ms){clock.now+=ms;for(const [id,timer] of [...timers])if(timer.at<=clock.now){timers.delete(id);timer.fn();}},saved:()=>JSON.parse(readStored()),change:(selector,target)=>element(selector).listeners.change({target}),
    click:(action,extra={})=>{const el={dataset:{action,...extra}};return element('#main').listeners.click({target:{closest:selector=>selector==='[data-action]'?el:null}});}};
}
function seedExploration(a){a.state.labResults=[{label:'Old plan',result:null}];a.state.decision={targetReadiness:.8,simulationCount:180};}
function assertCleared(a){assert.equal(a.state.labResults,null);assert.equal(a.state.decision,null);}

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
  assert.match(overview,/10 paired paths/);assert.match(results,/10 paired paths/);
  assert.match(report,/Most helpful sensitivity check:/);assert.match(report,/10 paired paths/);
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
    const workerCount=task==='runLab'?7:1;
    for(let i=0;i<workerCount;i++){
      const worker=a.workers[i];assert.ok(worker);
      worker.onmessage({data:outcome==='error'?{type:'error',message:'Test calculation failure'}:{type:'result',result:task==='runDecision'?{targetReadiness:.8,simulationCount:180}:runSimulation(worker.data.scenario)}});
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
    {tier:'free',maxPaths:10,signedIn:false,checkoutAvailable:true},
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
  const a=app(null,{fetch:async()=>Response.json({tier:'free',maxPaths:10,signedIn:true,checkoutAvailable:true,billingPortalAvailable:true})});
  await a.loadAccess();assert.match(a.billingView(),/data-action="billing-portal"/);
  a.state.access.billingPortalAvailable=false;assert.doesNotMatch(a.billingView(),/data-action="billing-portal"/);
});

test('first visit leads with starting actions and an explicitly illustrative chart',()=>{
  const a=app(),html=a.dashboard();
  assert.match(html,/Explore how long your retirement savings could last/);
  assert.match(html,/Build my forecast/);assert.match(html,/Explore a sample plan/);
  assert.match(html,/Illustrative paths only/);assert.match(html,/Your financial inputs stay in your browser/);
  assert.match(html,/Free preview · 10 simulated lifetimes/);assert.match(html,/Sample plan at a glance/);
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
  assert.doesNotMatch(html,/overview-upgrade-title|Free preview · 10 simulated lifetimes/);
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
  const lab=a.runLab();for(let i=1;i<=7;i++){const w=a.workers[i];assert.ok(w);w.onmessage({data:{type:'result',result:runSimulation(w.data.scenario)}});await Promise.resolve();}
  await lab;assert.equal(a.state.labResults.length,7);assert.equal(a.state.busy,false);
});
test('lower-spending comparisons use the same home-sale assumptions as an editor change',async()=>{
  const s=model.baseScenario();s.home.currentValue=300000;
  s.budget.annualPropertyTaxes=20000;s.budget.monthlyBudgets=[{month:'2026-01',checkingSavingsBills:[{monthlyAmount:2500}]}];
  model.applyBudgetEstimate(s);
  const a=app({scenarios:[s],selectedId:s.id}),pending=a.runLab();let spendingVariant;
  for(let i=0;i<7;i++){
    const worker=a.workers[i];assert.ok(worker);
    if(i===2)spendingVariant=structuredClone(worker.data.scenario);
    if(i===0||i===1)assert.equal(worker.data.scenario.budget.isAppliedToAnnualBaseSpending,true);
    worker.onmessage({data:{type:'result',result:runSimulation(worker.data.scenario)}});await Promise.resolve();
  }
  await pending;
  assert.equal(a.current().budget.isAppliedToAnnualBaseSpending,true,'Comparison must preserve the original budget');
  const comparisonReadiness=a.state.labResults[2].result.successProbability;
  await a.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:String(s.spending.annualBaseSpending*.95)});
  assert.deepEqual(spendingVariant.budget,structuredClone(a.current().budget));
  const entered=structuredClone(a.current());entered.numberOfSimulations=spendingVariant.numberOfSimulations;
  assert.equal(runSimulation(entered).successProbability,comparisonReadiness);
});
test('ten-path outcomes use counts and a visible warning in free and Pro views and reports',()=>{
  for(const tier of ['free','pro']){
    const a=app();a.state.access.tier=tier;const r=runSimulation(a.current());a.state.results.set(a.current().id,r);a.state.labResults=[{label:'Current plan',result:r}];
    const label=format.readinessLabel(r),percent=`${(100*r.successProbability).toFixed(1)}%`;assert.match(label,/^\d+ of 10$/);
    for(const html of [a.results(),a.dashboard(),a.lab()]){assert.ok(html.includes(label),label);assert.match(html,/Sample preview only/);assert.ok(!html.includes(percent),percent);}
    const report=a.reportText(a.current(),r);assert.ok(report.includes(`Lifetimes without a portfolio shortfall: ${label}`));assert.match(report,/SAMPLE PREVIEW ONLY/);assert.doesNotMatch(report,/Modeled readiness.*100\.0%/);
  }
});
test('larger runs retain percentage summaries without the small-preview warning',()=>{
  const a=app();a.state.access.tier='pro';a.current().numberOfSimulations=100;const r=runSimulation(a.current());a.state.results.set(a.current().id,r);
  assert.match(a.results(),/Monte Carlo readiness/);assert.match(a.results(),/\d+\.\d%/);assert.doesNotMatch(a.results(),/Sample preview only/);
  assert.equal(format.shareLabel(.75,4),'3 of 4');assert.equal(format.shareLabel(.75,100),'75.0%');
});

test('retained reports and comparisons keep their actual counts across upgrades and expiration',async()=>{
  for(const [completedCount,nextTier,nextCount] of [[4,'pro',10000],[10,'pro',10000],[150,'free',10]]){
    const a=app(null,{fetch:async()=>Response.json({tier:nextTier,signedIn:true,accountKey:'user-a'})});
    a.state.access.tier=completedCount>10?'pro':'free';
    a.current().numberOfSimulations=completedCount;
    const r=runSimulation(a.current());a.state.results.set(a.current().id,r);
    a.state.labResults=[{label:'Current plan',result:r}];
    await a.loadAccess();
    const report=a.reportText(a.current(),r);
    assert.ok(report.includes(`Simulation paths: ${completedCount}; fixed comparison sequence`));
    assert.ok(report.includes(`  Simulation paths: ${completedCount}\n`));
    assert.ok(report.includes(`Paths for next run: ${nextCount}`));
    const html=a.lab();
    assert.ok(html.includes(`Each comparison runs ${completedCount} Monte Carlo paths`));
    assert.ok(html.includes(completedCount<=10?'Samples without shortfall':'Modeled readiness'));
    assert.equal(html.includes('Sample preview only'),completedCount<=10);
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
  a.state.decision={targetReadiness:.8,simulationCount:180,earliestRetirementAge:55,safeAnnualSpending:250000,safeSpendingAtSearchLimit:true,safeSpendingSearchLimit:250000};
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
    clock.now+=5*60*1000;assert.equal(a.isPro(),false);assert.equal(a.effectivePaths(),10);
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
    assert.equal(a.renders(),renders+1);assert.equal(a.timers.size,0);assert.equal(a.effectivePaths(),10);
    assert.doesNotMatch(a.element('#main').innerHTML,/Active on this account|Owner Pro access is active|last verified Pro access is available/);
    assert.match(a.state.message,running?/Calculating this plan/:/New runs use the 10-path/);assert.equal(a.state.results.get(a.current().id),result);assert.ok(a.state.decision);
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
  assert.match(a.element('#busy-detail').textContent,/5 of 10 lifetimes simulated/);assert.match(a.element('#result-state').textContent,/50%/);
  worker.onmessage({data:{type:'result',result:runSimulation(worker.data.scenario)}});await pending;
  assert.equal(a.state.busy,false);assert.equal(a.state.progress,null);assert.equal(a.state.results.size,1);
  const b=app();b.state.access.tier='pro';const search=b.runDecision();
  b.workers[0].onmessage({data:{type:'progress',phase:'spending',checkedAges:3,totalAges:7,checkedAmounts:10,totalAmounts:501}});
  assert.match(b.element('#busy-detail').textContent,/10 of up to 501 spending amounts/);assert.equal(b.element('#busy-progress').value,17/508);
  b.workers[0].onmessage({data:{type:'result',result:{targetReadiness:.8,simulationCount:180}}});await search;assert.equal(b.state.busy,false);
});

test('a comparison that already matches the plan is reported instead of rerun',async()=>{
  const s=model.baseScenario();Object.assign(s.withdrawalStrategy,{useCashReserveDuringDrawdowns:true,drawdownTrigger:-.01});
  const a=app({scenarios:[s],selectedId:s.id}),pending=a.runLab();
  for(let i=0;i<6;i++){const w=a.workers[i];assert.ok(w);w.onmessage({data:{type:'result',result:runSimulation(w.data.scenario)}});await Promise.resolve();}
  await pending;assert.equal(a.workers.length,6);assert.equal(a.state.labResults.length,7);
  const row=a.state.labResults.at(-1);assert.equal(row.label,'Use cash first in months below −1%');assert.equal(row.result,null);
  assert.match(a.lab(),/Already matches the current plan/);assert.doesNotMatch(a.lab(),/Larger cash reserve/);
});

test('warnings and progress use the neutral notice style; completed actions use success styling',()=>{
  const a=app();
  for(const [message,className] of [
    ['Subscription status is unavailable. New runs use the 10-path free preview; your completed results are kept.','notice'],
    ['Sign in again to verify your plan. Your completed results are still available.','notice'],
    ['Planning targets require Pro because 10 paths are too coarse.','notice'],
    ['Calculation canceled. Completed results are unchanged.','notice'],
    ['Results updated for Base plan.','notice good'],
    ['Error: Something failed.','notice error'],
  ]){a.state.message=message;assert.match(a.dashboard(),new RegExp(`<div class="${className}" role="status">`),message);}
});

test('planning targets report the whole-year ages actually searched',()=>{
  const a=app();a.state.access.tier='pro';
  a.state.decision={targetReadiness:.8,simulationCount:180,earliestRetirementAge:null,safeAnnualSpending:null,safeSpendingAtSearchLimit:false,safeSpendingSearchLimit:250000,retirementAgeSearchStart:60,retirementAgeSearchEnd:67};
  assert.match(a.lab(),/ages 60 through 67 with 180 paths per age/);assert.doesNotMatch(a.lab(),/through 70/);
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

test('free runs use ten paths for legacy and imported plans while preserving saved choices',async()=>{
  for(const savedCount of [4,5000]){
    const s=model.baseScenario();s.numberOfSimulations=savedCount;s.simulationPathsCustomized=savedCount!==4;
    const a=app({scenarios:[s],selectedId:s.id});
    assert.equal(a.effectivePaths(),10);assert.match(a.billingView(),/>10 paths</);
    a.state.setupSection=4;assert.match(a.setup(),/10 paths · Free preview/);
    const saved=a.stored(),pending=a.run();assert.equal(a.workers[0].data.scenario.numberOfSimulations,10);
    const result=runSimulation(a.workers[0].data.scenario);assert.equal(result.provenance.simulationCount,10);
    a.workers[0].onmessage({data:{type:'result',result}});await pending;
    assert.equal(a.stored(),saved);assert.equal(a.current().numberOfSimulations,savedCount);
    await a.change('#main',{dataset:{field:'numberOfSimulations',type:'number'},value:'1000'});
    assert.match(a.state.message,/10 paths are available/);assert.equal(a.current().numberOfSimulations,savedCount);
    a.state.access.tier='pro';assert.equal(a.effectivePaths(),savedCount,'The free allowance must not override a Pro selection');
  }
});

test('ten paths remain a counted preview and eleven paths use percentage formatting',()=>{
  for(const count of [4,10,11]){
    const result={provenance:{simulationCount:count},successProbability:.8};
    assert.equal(format.isPreviewResult(result),count<=10);
    assert.equal(format.readinessLabel(result),count<=10?`${Math.round(.8*count)} of ${count}`:'80.0%');
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
  r.balanceBands=[{age:67,median:100,pessimistic:100,optimistic:100,pathCount:10}];a.state.results.set(s.id,r);
  const html=a.results();assert.match(html,/A simulated death is not running out of money/);assert.match(html,/Not enough simulated outcomes/);assert.match(html,/0 of 10 observed/);
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
  assert.match(html,/lifespans run from short to long/);
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

test('comparison copies retain personal balances, full-run count and all unchanged assumptions',async()=>{
  const s=model.baseScenario();s.accounts.pretax=456789;s.accounts.cash=54321;s.numberOfSimulations=380;s.simulationPathsCustomized=true;
  s.market.stockMeanReturn=.09;
  const a=app({scenarios:[s],selectedId:s.id});a.state.access.tier='pro';
  a.state.entryPeriods[s.id]={'spending.annualBaseSpending':'month'};
  a.state.inputSources[s.id]['accounts.cash']='Entered';const parent=structuredClone(a.current()),pending=a.runLab();
  for(let i=0;i<7;i++){
    const worker=a.workers[i];assert.ok(worker);assert.equal(worker.data.scenario.numberOfSimulations,150);
    worker.onmessage({data:{type:'result',result:runSimulation(worker.data.scenario)}});await Promise.resolve();
  }
  await pending;assert.match(a.lab(),/Create a plan with this change/);
  const candidate=structuredClone(a.state.labResults[2].scenario);
  await a.click('copy-comparison',{index:'2'});const copy=structuredClone(a.current());
  assert.notEqual(copy.id,parent.id);assert.equal(copy.numberOfSimulations,380);
  assert.equal(copy.spending.annualBaseSpending,parent.spending.annualBaseSpending*.95);
  candidate.id=copy.id;candidate.name=copy.name;candidate.numberOfSimulations=380;
  assert.deepEqual(copy,candidate);
  assert.deepEqual(structuredClone(a.state.scenarios.find(x=>x.id===parent.id)),parent);
  assert.equal(a.state.inputSources[copy.id]['accounts.cash'],'Entered');
  assert.equal(a.state.inputSources[copy.id]['spending.annualBaseSpending'],'Entered');
  assert.equal(a.state.entryPeriods[copy.id]['spending.annualBaseSpending'],'month');
  assert.equal(a.state.exampleIds.has(copy.id),false);assert.equal(a.state.results.has(copy.id),false);
  assert.equal(a.state.view,'setup');assert.equal(a.state.setupSection,5);
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
  await a.run();assert.equal(a.workers.length,0,'Unknown timing still blocks calculations');
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
  let html=a.results();assert.match(html,/All 10 preview lifetimes stayed funded/);
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

test('personal setup asks about unsupported accounts before changing any financial inputs',async()=>{
  const a=app(),before=structuredClone(a.current());
  assert.match(a.dashboard(),/data-action="check-accounts">Build my forecast/);
  await a.click('check-accounts');assert.equal(a.state.view,'account-check');
  assert.deepEqual(structuredClone(a.current()),before);
  assert.match(a.accountCheck(),/data-action="start-plan" disabled/);
  for(const answer of ['yes','unsure','no']){
    await a.click('account-answer',{answer});assert.deepEqual(structuredClone(a.current()),before);
    assert.match(a.accountCheck(),new RegExp(`data-answer="${answer}" aria-pressed="true"`));
    if(answer!=='no')assert.match(a.accountCheck(),/incomplete/);
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
  assert.match(imported.reportSummaryText(imported.current()),/ACCOUNT CHECK:.*incomplete/);
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
