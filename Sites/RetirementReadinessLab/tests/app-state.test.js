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
import {runSimulation} from '../dist/engine.js';

// Execute the actual app and event handlers. Only browser IO is replaced;
// workers stay pending so changes during calculations can be reproduced.
function webLocks(){
  let queue=Promise.resolve();
  return {request(_name,callback){const next=queue.then(callback);queue=next.catch(()=>{});return next;}};
}
function app(saved=null,{fetch=async()=>{throw new Error('offline');},storage={fail:false},clock={now:Date.now()},session=new Map(),location={search:'',pathname:'/',hash:''},identity={accountKey:null},preferences=new Map(),params=URLSearchParams,confirm=()=>true,rawStorage,locks=webLocks(),socialActions={}}={}){
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
  const context=vm.createContext({...model,...format,...guidance,...withdrawalsView,...growthHelper,...moneyInput,structuredClone,Intl,URLSearchParams:params,Blob,URL:class extends URL{static createObjectURL(blob){const url='blob:test-'+downloadBlobs.size;downloadBlobs.set(url,blob);return url;}static revokeObjectURL(url){downloadBlobs.delete(url);}},console,Date:class extends Date{static now(){return clock.now;}},
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
  const api=vm.runInContext('({state,setup,run,runLab,runDecision,results,withdrawals,dashboard,lab,budget,budgetView,budgetSummary,budgetCostCheck,budgetCostsReviewed,enterBudget,scenarios,render,billingView,reportText,current,persist,loadAccess,isPro,effectivePaths,syncAuthState,linkAccounts,renders:()=>renderCount})',context);
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
  const s=model.baseScenario();s.budget.monthlyBudgets=[{month:'2026-08',creditCardBills:[{monthlyAmount:4000}]}];model.applyBudgetEstimate(s);
  const a=app({scenarios:[s],selectedId:s.id});a.state.view='budget';a.budget();const saved=a.stored();
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
  a.budgetView().costReview='excluded';await a.click('apply-budget');assert.equal(a.current().spending.annualBaseSpending,54000);assert.match(a.budgetSummary(),/>Applied</);
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
  assert.equal(s.rothHistory.contributionBasis,75000);assert.equal(s.accounts.roth,50000);
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
    assert.equal(a.current().accounts.pretax,500000);assert.equal(a.current().budget.annualPropertyTaxes,0);
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
    a.budgetView().costReview='excluded';await a.click(action);assert.match(a.state.message,/could not be saved/);assert.equal(a.saved(),null);
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
  const a=app(),before=JSON.stringify(a.state.scenarios);
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
  await a.loadAccess();assert.equal(a.state.results.size,1);assert.match(a.state.message,/Subscription status is unavailable/);
  const renders=a.renders();await a.loadAccess();assert.equal(a.state.results.size,1);assert.equal(a.renders(),renders,'an unchanged check does not redraw open charts');
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
  assert.equal(location.search,'?keep=1');assert.equal(location.hash,'#billing');
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
  const a=app(),original=JSON.stringify(a.current());
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
  assert.ok(html.indexOf('How earnings, savings and growth fit together')<html.indexOf('Your future savings'));
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
    assert.doesNotMatch(html,/Sample values are still in this plan|Includes sample values|Spouse birthday|Spouse longevity table|Review Unknown inputs before running/);
    s.household.filingStatus='Married';
    const couple=a.setup();
    assert.match(couple,/Sample values are still in this plan/);
    assert.match(couple,/Spouse birthday/);
    assert.match(couple,/Review Unknown inputs before running/);
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

test('input examples distinguish monthly pension income, annual spending and account balances',()=>{
  const a=app();a.state.setupSection=1;assert.match(a.setup(),/traditional 401\(k\), 403\(b\), IRA/);assert.match(a.setup(),/\$4,000 per month, enter \$48,000/);
  a.state.setupSection=2;assert.match(a.setup(),/\$1,500 monthly means \$18,000 yearly/);assert.match(a.setup(),/not the pension’s account or lump-sum value/);
  a.state.setupSection=3;assert.match(a.setup(),/\$1,200 each month stays \$1,200/);
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
  a.state.setupSection=1;assert.match(a.setup(),/Your future savings/);assert.match(a.setup(),/\$500 monthly means \$6,000 yearly/);assert.doesNotMatch(a.setup(),/Spouse retirement accounts/);
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
  assert.ok(html.includes(r.riskBreakdown.recommendedNextTest));
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
  assert.match(a.setup(),/<details class="guided-extra" id="basic-market-settings"><summary>Advanced/);
  await a.click('setup-detail',{mode:'advanced'});
  assert.doesNotMatch(a.setup(),/id="basic-market-settings"/);
  await a.click('setup-detail',{mode:'basic'});a.state.setupSection=1;
  assert.match(a.setup(),/Base spending stays level before inflation/);
  assert.match(a.setup(),/<summary>Advanced · Spending pattern and inflation/);
  assert.doesNotMatch(a.setup(),/class="field-source"/,'No per-field sample labels are added');
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

test('budget application requires consistent housing and health review and resets after edits',async()=>{
  const a=app();a.state.view='budget';await a.click('add-month');
  await a.change('#main',{dataset:{month:'0',part:'checking'},value:'5,000'});
  const prior=a.current().spending.annualBaseSpending;
  await a.click('apply-budget');assert.equal(a.current().spending.annualBaseSpending,prior);
  await a.change('#main',{dataset:{costReview:''},value:'included'});
  await a.change('#main',{dataset:{costConfirm:''},checked:true});
  assert.equal(a.budgetCostsReviewed(),false,'A confirmation with no deductions is insufficient');
  await a.change('#main',{dataset:{month:'0',part:'mortgage'},value:'1,000'});
  assert.equal(a.budgetView().costConfirmed,false);
  await a.click('review-budget-deductions');assert.equal(a.element('#budget-month-0').open,true);assert.equal(a.element('[data-budget-disclosure="adjustments-0"]').open,true);
  await a.change('#main',{dataset:{costReview:''},value:'excluded'});
  assert.equal(a.budgetCostsReviewed(),false,'Excluded payments must not also be deducted');
  assert.match(a.budgetCostCheck(),/Your months still have housing or health deductions/);
  await a.change('#main',{dataset:{costReview:''},value:'included'});
  await a.change('#main',{dataset:{costConfirm:''},checked:true});
  assert.equal(a.budgetCostsReviewed(),true);
  await a.click('apply-budget');assert.equal(a.current().spending.annualBaseSpending,48000);
  await a.change('#main',{dataset:{month:'0',part:'mortgage'},value:'1,100'});
  assert.equal(a.budgetCostsReviewed(),false,'A changed deduction needs confirmation again');
  await a.change('#main',{dataset:{costConfirm:''},checked:true});assert.equal(a.budgetCostsReviewed(),true);
  await a.change('#main',{dataset:{month:'0',part:'checking'},value:'5,100'});
  assert.equal(a.budgetView().costReview,'');assert.equal(a.budgetCostsReviewed(),false);
  assert.equal(a.current().spending.annualBaseSpending,48000,'Draft edits do not change applied spending');
});

test('budget can be applied with excluded separate costs and returns to the active guided step',async()=>{
  const a=app();a.state.view='setup';a.state.guided=true;a.state.setupSection=1;
  a.enterBudget();assert.equal(a.state.view,'budget');assert.match(a.budget(),/Return to setup/);
  await a.click('add-month');await a.change('#main',{dataset:{month:'0',part:'credit'},value:'3,500'});
  await a.change('#main',{dataset:{costReview:''},value:'excluded'});
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
