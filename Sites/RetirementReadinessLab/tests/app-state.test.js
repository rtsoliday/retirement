import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import * as model from '../dist/model.js';
import * as format from '../dist/result-format.js';
import {runSimulation} from '../dist/engine.js';

// Execute the actual app and event handlers. Only browser IO is replaced;
// workers stay pending so changes during calculations can be reproduced.
function app(saved=null,{fetch=async()=>{throw new Error('offline');},storage={fail:false}}={}){
  const elements=new Map(),workers=[];
  let stored=saved===null?null:JSON.stringify(saved);
  function element(selector){
    if(['#advanced-model','#allocation-settings'].includes(selector))return null;
    if(!elements.has(selector))elements.set(selector,{innerHTML:'',textContent:'',dataset:{},listeners:{},classList:{toggle(){},remove(){}},addEventListener(name,fn){this.listeners[name]=fn;},querySelectorAll(){return [];},setAttribute(){},focus(){},scrollIntoView(){},insertAdjacentHTML(){},remove(){elements.delete(selector);}});
    return elements.get(selector);
  }
  const context=vm.createContext({...model,...format,structuredClone,Intl,URLSearchParams,URL,console,
    location:{search:'',pathname:'/',hash:''},history:{replaceState(){}},confirm:()=>true,
    localStorage:{getItem:()=>stored,setItem(key,value){if(storage.fail)throw new Error('QuotaExceededError');stored=value;}},window:{addEventListener(){},scrollTo(){}},
    document:{querySelector:element,querySelectorAll:()=>[],addEventListener(){}},
    chartCard:()=>'',mountCharts(){},disposeCharts(){},initializeSocialAuth:()=>new Promise(()=>{}),fetch,authHeaders:async()=>({}),
    Worker:class{constructor(){workers.push(this);}postMessage(data){this.data=data;}terminate(){this.terminated=true;}},
  });
  const source=readFileSync(new URL('../dist/app.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'');
  vm.runInContext(source.replace('function render(){','let renderCount=0;function render(){renderCount++;'),context);
  const api=vm.runInContext('({state,run,runLab,runDecision,results,dashboard,lab,budget,billingView,reportText,current,persist,loadAccess,renders:()=>renderCount})',context);
  return {...api,workers,element,saved:()=>JSON.parse(stored),change:(selector,target)=>element(selector).listeners.change({target}),
    click:(action,extra={})=>{const el={dataset:{action,...extra}};return element('#main').listeners.click({target:{closest:selector=>selector==='[data-action]'?el:null}});}};
}
function seedExploration(a){a.state.labResults=[{label:'Old plan',result:null}];a.state.decision={targetReadiness:.8,simulationCount:180};}
function assertCleared(a){assert.equal(a.state.labResults,null);assert.equal(a.state.decision,null);}

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

test('unfinished but structurally valid budget drafts remain importable',async()=>{
  const a=app(),draft=model.baseScenario();draft.budget.monthlyBudgets=[{month:'',checkingSavingsBills:[],creditCardBills:[],cashAndAtmWithdrawals:0}];
  await a.change('#import-file',{files:[{text:async()=>JSON.stringify([draft])}],value:'backup.json'});
  assert.match(a.state.message,/1 scenarios imported/);assert.doesNotThrow(()=>a.budget());assert.doesNotThrow(()=>a.reportText(a.current()));
});

test('failed assumption saves show an error and recover after a successful retry',()=>{
  const storage={fail:true},a=app(null,{storage});
  a.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'123456'});
  assert.equal(a.current().accounts.pretax,123456);assert.equal(a.saved(),null);
  assert.match(a.state.message,/Error: Changes could not be saved/);assert.doesNotMatch(a.state.message,/^Saved/);
  assert.equal(a.element('#main > .notice').className,'notice error');
  storage.fail=false;a.change('#main',{dataset:{field:'accounts.pretax',type:'money'},value:'234567'});
  assert.equal(a.saved().scenarios[0].accounts.pretax,234567);assert.match(a.state.message,/^Saved/);assert.equal(a.state.storageError,'');
});

test('failed budget saves display a warning without leaving the editor',()=>{
  const a=app(null,{storage:{fail:true}});a.state.view='budget';
  a.change('#main',{dataset:{budget:'annualPropertyTaxes'},value:'1000'});
  assert.match(a.element('#main > .notice').textContent,/could not be saved/);assert.equal(a.saved(),null);assert.equal(a.state.view,'budget');
});

test('copy, reset, apply and import cannot overwrite a failed-save warning',async()=>{
  for(const action of ['new-scenario','reset-assumptions','apply-budget']){
    const a=app(null,{storage:{fail:true}});a.current().budget.monthlyBudgets=[{month:'2026-01',creditCardBills:[{monthlyAmount:1000}]}];
    await a.click(action);assert.match(a.state.message,/could not be saved/);assert.equal(a.saved(),null);
  }
  const a=app(null,{storage:{fail:true}});
  await a.change('#import-file',{files:[{text:async()=>JSON.stringify([model.baseScenario()])}],value:'backup.json'});
  assert.match(a.state.message,/could not be saved/);assert.equal(a.saved(),null);
});

test('billing portal remains visible for a free account with an existing customer',async()=>{
  const a=app(null,{fetch:async()=>Response.json({tier:'free',maxPaths:4,signedIn:true,checkoutAvailable:true,billingPortalAvailable:true})});
  await a.loadAccess();assert.match(a.billingView(),/data-action="billing-portal"/);
  a.state.access.billingPortalAvailable=false;assert.doesNotMatch(a.billingView(),/data-action="billing-portal"/);
});

test('first visit leads with starting actions and an explicitly illustrative chart',()=>{
  const a=app(),html=a.dashboard();
  assert.match(html,/Explore how long your retirement savings could last/);
  assert.match(html,/Build my forecast/);assert.match(html,/Explore a sample plan/);
  assert.match(html,/Illustrative paths only/);assert.match(html,/Your financial inputs stay in your browser/);
  assert.match(html,/Free preview · 4 simulated lifetimes/);assert.match(html,/Sample plan at a glance/);
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
test('automatic Pro defaults preserve the first-visit state across reloads',()=>{
  const a=app();a.state.access.tier='pro';a.state.scenarios.forEach(model.applyProSimulationDefault);a.persist(false);
  const restored=app(a.saved());restored.state.access.tier='pro';const html=restored.dashboard();
  assert.match(html,/Build my forecast/);assert.match(html,/Pro · Up to 10,000/);
  assert.doesNotMatch(html,/overview-upgrade-title|Free preview · 4 simulated lifetimes/);
});

test('both scenario selectors discard prior comparisons and targets',async()=>{
  const a=app();seedExploration(a);await a.click('select-scenario',{id:'later-retirement'});assert.equal(a.current().id,'later-retirement');assertCleared(a);
  seedExploration(a);a.change('#scenario-select',{value:'base-plan'});assert.equal(a.current().id,'base-plan');assertCleared(a);
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
  const a=app(),pending=a.run();a.change('#main',{dataset:{field:'spending.annualBaseSpending',type:'money'},value:'90000'});
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
test('four-path outcomes use counts and a visible warning in free and Pro views and reports',()=>{
  for(const tier of ['free','pro']){
    const a=app();a.state.access.tier=tier;const r=runSimulation(a.current());a.state.results.set(a.current().id,r);a.state.labResults=[{label:'Current plan',result:r}];
    for(const html of [a.results(),a.dashboard(),a.lab()]){assert.match(html,/4 of 4/);assert.match(html,/Sample preview only/);assert.doesNotMatch(html,/100\.0%/);}
    const report=a.reportText(a.current(),r);assert.match(report,/4 of 4/);assert.match(report,/SAMPLE PREVIEW ONLY/);assert.doesNotMatch(report,/Modeled readiness.*100\.0%/);
  }
});
test('larger runs retain percentage summaries without the four-path warning',()=>{
  const a=app();a.state.access.tier='pro';a.current().numberOfSimulations=100;const r=runSimulation(a.current());a.state.results.set(a.current().id,r);
  assert.match(a.results(),/Monte Carlo readiness/);assert.match(a.results(),/\d+\.\d%/);assert.doesNotMatch(a.results(),/Sample preview only/);
  assert.equal(format.shareLabel(.75,4),'3 of 4');assert.equal(format.shareLabel(.75,100),'75.0%');
});

test('a failed access check on window focus keeps results when the tier is unchanged',async()=>{
  const a=app(),pending=a.run();a.workers[0].onmessage({data:{type:'result',result:runSimulation(a.workers[0].data.scenario)}});await pending;
  await a.loadAccess();assert.equal(a.state.results.size,1);assert.match(a.state.message,/Subscription status is unavailable/);
  const renders=a.renders();await a.loadAccess();assert.equal(a.state.results.size,1);assert.equal(a.renders(),renders,'an unchanged check does not redraw open charts');
});
test('an access check that changes the tier still clears results computed under the old tier',async()=>{
  const a=app(null,{fetch:async()=>({ok:true,json:async()=>({tier:'pro',maxPaths:10000,signedIn:true,checkoutAvailable:true,accountProvider:'chatgpt'})})});
  const result=runSimulation(a.current()),pending=a.run();a.workers[0].onmessage({data:{type:'result',result}});await pending;
  await a.loadAccess();assert.equal(a.state.access.tier,'pro');assert.equal(a.state.results.size,0);
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
