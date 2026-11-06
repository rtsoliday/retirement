import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {VIEW_KEY,LANDING_KEY,sheetHref,plannerHref,parseSheetHash,rememberView,handOffPlan,takeHandedOffPlan,arrivalReferrer} from '../dist/view-switch.js';
import {SECTIONS,countsAsSimulationStart} from '../dist/sheet-model.js';
import {inputTask} from '../dist/plan-review.js';

const read=name=>readFileSync(new URL('../dist/'+name,import.meta.url),'utf8');
const memory=(entries=[])=>{const map=new Map(entries);return {map,getItem:key=>map.has(key)?map.get(key):null,setItem:(key,value)=>{map.set(key,String(value));},removeItem:key=>{map.delete(key);}};};
function withStorage(name,store,fn){
  const before=Object.getOwnPropertyDescriptor(globalThis,name);
  Object.defineProperty(globalThis,name,{value:store,configurable:true,writable:true});
  try{return fn();}
  finally{if(before)Object.defineProperty(globalThis,name,before);else delete globalThis[name];}
}

// Runs the home page's inline landing rule against a fake browser.
function landing({search='',hash='',preference='sheet',navigation='navigate',referrer='https://example.com/page',brokenStorage=false}={}){
  const script=read('index.html').match(/<script>([\s\S]*?)<\/script>/)[1],calls=[],session=memory();
  vm.runInNewContext(script,{
    location:{search,hash,replace:url=>calls.push(['replace',url])},
    localStorage:{getItem(key){if(brokenStorage)throw new Error('blocked');assert.equal(key,VIEW_KEY);return preference;}},
    sessionStorage:session,document:{referrer},window:{stop:()=>calls.push(['stop'])},
    performance:{getEntriesByType:()=>navigation===null?[]:[{type:navigation}]},
  });
  return {calls,session};
}

test('a visitor who chose the sheet lands on it, and the page stops loading first',()=>{
  const {calls,session}=landing();
  assert.deepEqual(calls,[['stop'],['replace','./sheet.html']]);
  assert.equal(session.getItem(LANDING_KEY),'https://example.com/page');
});

test('the landing rule leaves everyone else on the planner',()=>{
  for(const options of [
    {preference:null},{preference:'planner'},
    {hash:'#/budget'},{hash:'#/'},{hash:'#billing'},{search:'?checkout=success&session_id=x'},{search:'?link=1'},
    // Back and reload must not bounce someone off the page they returned to.
    {navigation:'back_forward'},{navigation:'reload'},{navigation:null},
    {brokenStorage:true},
  ])assert.deepEqual(landing(options).calls,[],JSON.stringify(options));
});

test('switching keeps your place in both directions',()=>{
  assert.equal(sheetHref('results'),'./sheet.html#forecast');
  assert.equal(sheetHref('withdrawals'),'./sheet.html#yearly');
  for(const section of [0,1,2,3,4]){
    const href=sheetHref('setup',section);
    assert.equal(parseSheetHash(href.slice(href.indexOf('#'))).step,section);
    assert.equal(plannerHref('plan',section),`./#/assumptions/${section+1}`);
  }
  // The review step and pages with no equivalent open the sheet at the top.
  for(const [view,section] of [['setup',5],['dashboard',0],['budget',0],['lab',0],['scenarios',0],['reports',0],['billing',0]])assert.equal(sheetHref(view,section),'./sheet.html');
  assert.deepEqual(parseSheetHash('#forecast'),{tab:'forecast'});
  assert.deepEqual(parseSheetHash('#yearly'),{tab:'yearly'});
  for(const other of ['','#','#plan','#step-0','#step-6','#step-1x'])assert.deepEqual(parseSheetHash(other),{},other);
  assert.equal(plannerHref('forecast'),'./#/results');
  assert.equal(plannerHref('yearly'),'./#/withdrawals');
  assert.equal(plannerHref('plan',null),'./#/');
});

test('every way back to the planner has a hash, so the landing rule cannot bounce it',()=>{
  for(const href of [plannerHref('plan'),plannerHref('plan',0),plannerHref('plan',4),plannerHref('forecast'),plannerHref('yearly')])assert.match(href,/^\.\/#\//);
  assert.match(read('sheet.html'),/id="to-planner" href="\.\/#\/"/);
  for(const [,href] of read('sheet.html').matchAll(/<a[^>]*href="(\.\/(?:#[^"]*)?)"/g))assert.match(href,/^\.\/#\//);
  assert.ok(!/<a[^>]*href="\.\/"/.test(read('sheet.html')),'a bare home link would land back on the sheet');
});

test('each planner setup step has rows to land on in the sheet',()=>{
  const steps=new Set(SECTIONS.flatMap(section=>section.rows.filter(row=>row.path).map(row=>inputTask(row.path).step)));
  assert.deepEqual([...steps].sort(),[0,1,2,3,4]);
  // Spot checks: a mapping change here would send people to the wrong page.
  const stepOf=path=>inputTask(path).step;
  assert.equal(stepOf('household.retirementDate'),0);
  assert.equal(stepOf('accounts.pretax'),1);
  assert.equal(stepOf('spending.annualBaseSpending'),1);
  assert.equal(stepOf('socialSecurity.claimAge'),2);
  assert.equal(stepOf('healthcare.preMedicareMonthlyPremium'),3);
});

test('both pages show the same two-way switch and mark where you are',()=>{
  const planner=read('index.html'),sheet=read('sheet.html');
  assert.match(planner,/<nav class="view-switch" aria-label="View"><span aria-current="page">Planner<\/span><a id="to-sheet" href="\.\/sheet\.html"[^>]*>Sheet<\/a><\/nav>/);
  assert.match(sheet,/<nav class="view-switch" aria-label="View"><a id="to-planner" href="\.\/#\/"[^>]*>Planner<\/a><span aria-current="page">Sheet<\/span><\/nav>/);
  assert.ok(!/classic/i.test(sheet),'the sheet should not call the planner classic');
  assert.ok(!/classic planner/i.test(read('sheet-model.js').replace(/^\s*\/\/.*$/gm,'')),'no user-visible classic wording');
  assert.ok(!/classic planner/i.test(read('sheet.js').replace(/^\s*\/\/.*$/gm,'')),'no user-visible classic wording');
});

test('the inline rule uses the same storage names as the module',()=>{
  const script=read('index.html').match(/<script>([\s\S]*?)<\/script>/)[1];
  assert.ok(script.includes(`'${VIEW_KEY}'`));assert.ok(script.includes(`'${LANDING_KEY}'`));
});

test('the plan in focus travels once and never enters a URL',()=>{
  withStorage('sessionStorage',memory(),()=>{
    assert.equal(takeHandedOffPlan(),null);
    handOffPlan('plan-7');
    assert.equal(takeHandedOffPlan(),'plan-7');
    assert.equal(takeHandedOffPlan(),null);
  });
  // Blocked storage is harmless: the views just open on the saved current plan.
  withStorage('sessionStorage',{getItem(){throw new Error('blocked');},setItem(){throw new Error('blocked');}},()=>{
    assert.doesNotThrow(()=>handOffPlan('plan-7'));
    assert.equal(takeHandedOffPlan(),null);
  });
  for(const href of [sheetHref('setup',1),sheetHref('results'),plannerHref('plan',1)])assert.doesNotMatch(href,/plan-|\?/);
});

test('remembering a view survives blocked storage',()=>{
  const store=memory();
  withStorage('localStorage',store,()=>rememberView('sheet'));
  assert.equal(store.getItem(VIEW_KEY),'sheet');
  withStorage('localStorage',{setItem(){throw new Error('blocked');}},()=>assert.doesNotThrow(()=>rememberView('sheet')));
});

test('landing on the sheet keeps the original referrer once, for arrival counts',()=>{
  const store=memory([[LANDING_KEY,'https://search.example/']]);
  withStorage('sessionStorage',store,()=>{
    assert.equal(arrivalReferrer('https://retirementforecast.us/'),'https://search.example/');
    assert.equal(arrivalReferrer('https://retirementforecast.us/'),'https://retirementforecast.us/');
  });
  // A typed address has an empty referrer. That must stay direct, not become internal.
  withStorage('sessionStorage',memory([[LANDING_KEY,'']]),()=>assert.equal(arrivalReferrer('https://retirementforecast.us/'),''));
  withStorage('sessionStorage',{getItem(){throw new Error('blocked');}},()=>assert.equal(arrivalReferrer('x'),'x'));
});

test('what counts as a simulation start in the sheet',()=>{
  const free={count:100,fullCount:100},proLive={count:200,fullCount:1000},proFull={count:1000,fullCount:1000};
  // Free tier: page-load estimates don't count, the first edit's recalculation does.
  assert.equal(countsAsSimulationStart({...free,explicit:false,editPending:false}),false);
  assert.equal(countsAsSimulationStart({...free,explicit:false,editPending:true}),true);
  // Pro: a live estimate is not the full run, however it was caused. F9 is.
  assert.equal(countsAsSimulationStart({...proLive,explicit:false,editPending:true}),false);
  assert.equal(countsAsSimulationStart({...proFull,explicit:true,editPending:false}),true);
  // F9 counts whatever else is going on.
  assert.equal(countsAsSimulationStart({...free,explicit:true,editPending:false}),true);
});

test('frozen result cells stay opaque so scrolled rows cannot show through them',()=>{
  // Grid cells share the calc class with the status bar label, so that rule must stay scoped.
  const css=read('sheet.css');
  assert.doesNotMatch(css,/(^|\n)\s*\.calc\s*\{[^}]*opacity/);
  assert.match(css,/\.status \.calc\{[^}]*opacity/);
});
