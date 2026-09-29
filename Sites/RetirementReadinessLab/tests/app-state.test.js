import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import * as model from '../dist/model.js';
import * as format from '../dist/result-format.js';
import {runSimulation} from '../dist/engine.js';

// Execute the actual app and event handlers. Only browser IO is replaced;
// workers stay pending so changes during calculations can be reproduced.
function app(saved=null,{fetch=async()=>{throw new Error('offline');},storage={fail:false},clock={now:Date.now()},session=new Map(),location={search:'',pathname:'/',hash:''},identity={accountKey:null},params=URLSearchParams}={}){
  const elements=new Map(),workers=[],timers=new Map();let nextTimer=0;
  let stored=saved===null?null:JSON.stringify(saved);
  function element(selector){
    if(['#advanced-model','#allocation-settings'].includes(selector))return null;
    if(!elements.has(selector))elements.set(selector,{innerHTML:'',textContent:'',dataset:{},listeners:{},classList:{toggle(){},remove(){}},addEventListener(name,fn){this.listeners[name]=fn;},querySelectorAll(){return [];},setAttribute(){},focus(){},scrollIntoView(){},insertAdjacentHTML(){},remove(){elements.delete(selector);}});
    return elements.get(selector);
  }
  const context=vm.createContext({...model,...format,structuredClone,Intl,URLSearchParams:params,URL,console,Date:class extends Date{static now(){return clock.now;}},
    location,history:{replaceState(_state,_title,url){const next=new URL(url,'https://example.test');location.search=next.search;location.hash=next.hash;}},confirm:()=>true,
    setTimeout(fn,delay){const id=++nextTimer;timers.set(id,{fn,at:clock.now+delay});return id;},clearTimeout(id){timers.delete(id);},
    sessionStorage:{getItem:key=>session.get(key)||null,setItem:(key,value)=>session.set(key,value)},socialState:()=>({...identity}),
    localStorage:{getItem:()=>stored,setItem(key,value){if(storage.fail)throw new Error('QuotaExceededError');stored=value;}},window:{addEventListener(){},scrollTo(){}},
    document:{querySelector:element,querySelectorAll:()=>[],addEventListener(){}},
    chartCard:()=>'',mountCharts(){},disposeCharts(){},initializeSocialAuth:()=>new Promise(()=>{}),fetch,authHeaders:async()=>({}),
    Worker:class{constructor(){workers.push(this);}postMessage(data){this.data=data;}terminate(){this.terminated=true;}},
  });
  const source=readFileSync(new URL('../dist/app.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replaceAll('import.meta.url',JSON.stringify(new URL('../dist/app.js',import.meta.url).href));
  vm.runInContext(source.replace('function render(){','let renderCount=0;function render(){renderCount++;'),context);
  const api=vm.runInContext('({state,run,runLab,runDecision,results,dashboard,lab,budget,billingView,reportText,current,persist,loadAccess,isPro,effectivePaths,syncAuthState,linkAccounts,renders:()=>renderCount})',context);
  return {...api,workers,element,timers,advanceTime(ms){clock.now+=ms;for(const [id,timer] of [...timers])if(timer.at<=clock.now){timers.delete(id);timer.fn();}},saved:()=>JSON.parse(stored),change:(selector,target)=>element(selector).listeners.change({target}),
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
    clock.now+=5*60*1000;assert.equal(a.isPro(),false);assert.equal(a.effectivePaths(),4);
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
    assert.equal(a.renders(),renders+1);assert.equal(a.timers.size,0);assert.equal(a.effectivePaths(),4);
    assert.doesNotMatch(a.element('#main').innerHTML,/Active on this account|Owner Pro access is active|last verified Pro access is available/);
    assert.match(a.state.message,running?/Calculating this plan/:/New runs use the four-path/);assert.equal(a.state.results.get(a.current().id),result);assert.ok(a.state.decision);
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
