import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as sheetModel from '../dist/sheet-model.js';
import * as model from '../dist/model.js';
import * as planReview from '../dist/plan-review.js';
import {resultFingerprint} from '../dist/result-cache.js';
import {baseScenario,sampleScenarios,prepareCalendarScenario,validateScenario,scenarioWarnings,addCalendarMonths,localCalendarDate} from '../dist/model.js';
import {normalizeInputSources} from '../dist/ux-guidance.js';
import {runSimulation} from '../dist/engine.js';
import {SECTIONS,EDITABLE_ROWS,getPath,cellText,editText,parseCell,applyCell,clearCell,isUnknown,evaluateArithmetic,blockingUnknowns,planIssues,simulationScenario,shownResult,readinessDelta,forecastTable,yearlyTable,fanPaths,fanPeak,totalSavings,toCsv,toTsv,parseTsv} from '../dist/sheet-model.js';

const rowFor=path=>EDITABLE_ROWS.find(r=>r.path===path);
function plan(edit=()=>{}){
  const s=prepareCalendarScenario(sampleScenarios()[0],{needsReview:false});
  s.household.separatePeople=true;s.household.spouseRetirementDate=s.household.retirementDate;
  edit(s);
  return s;
}
const contextFor=s=>({sources:normalizeInputSources(null,[s],'Sample/default'),exampleIds:new Set([s.id]),drafts:new Map()});
function fill(s,count=60){return runSimulation(simulationScenario(s,count),()=>{},{includePathPoints:false,includeRiskAnalysis:false});}

test('every sheet row points at a real plan field and appears once',()=>{
  const base=baseScenario(),seen=new Set();
  for(const row of EDITABLE_ROWS){
    assert.notEqual(getPath(base,row.path),undefined,row.path);
    assert.ok(!seen.has(row.path),'duplicate '+row.path);seen.add(row.path);
    assert.ok(row.label&&row.note,row.path);
    // Terse by design: the note is one short line in the status bar.
    assert.ok(row.note.length<=100,`${row.path} note is ${row.note.length} characters`);
  }
  assert.ok(SECTIONS.flatMap(s=>s.rows).length>=90);
});

test('what a cell shows parses back to the same stored value',()=>{
  const s=plan(p=>{p.household.filingStatus='Married';p.market.stockMeanReturn=.133;p.spending.generalInflationMean=.023;p.withdrawalStrategy.drawdownTrigger=-.01;p.accounts.pretax=1234567.89;});
  for(const row of EDITABLE_ROWS){
    const value=getPath(s,row.path);
    if(row.type==='date'&&!value)continue;
    for(const text of [cellText(row,value),editText(row,value)]){
      const out=parseCell(row,text,s);
      assert.equal(out.error,undefined,`${row.path} rejected "${text}": ${out.error}`);
      assert.equal(out.value,value,`${row.path} via "${text}"`);
    }
  }
});

test('number entry takes shorthand and arithmetic but never runs code',()=>{
  const s=plan(),money=rowFor('accounts.pretax'),pct=rowFor('market.stockMeanReturn'),int=rowFor('socialSecurity.claimAge');
  for(const [text,expected] of [['40k',40000],['$1.2M',1200000],['1,250,000',1250000],['=3500*12',42000],['3500*12',42000],['=(1000+500)*2',3000],['=100k/2',50000],['-5',-5],['0',0]])assert.equal(parseCell(money,text,s).value,expected,text);
  assert.equal(parseCell(pct,'13.3',s).value,.133);
  assert.equal(parseCell(pct,'13.3%',s).value,.133);
  assert.equal(parseCell(pct,'-1',s).value,-.01);
  assert.equal(parseCell(int,'67',s).value,67);
  assert.ok(parseCell(int,'67.5',s).error);
  for(const text of ['abc','1,2,3','=C14*2','=alert(1)','=1/0','=(1+','=constructor','1e999','=Infinity'])assert.ok(parseCell(money,text,s).error,text);
  assert.match(evaluateArithmetic('=C14*2').error,/references/);
  assert.equal(parseCell(money,'9007199254740993',s).error,'That amount is too large.');
});

test('a blank is Unknown rather than zero, keeps the value and blocks the run',()=>{
  const s=plan(),ctx=contextFor(s),row=rowFor('accounts.taxable');
  const before=s.accounts.taxable;
  assert.deepEqual(parseCell(row,'  ',s),{clear:true});
  clearCell(ctx,s,row);
  assert.equal(isUnknown(ctx,s,row),true);
  assert.equal(s.accounts.taxable,before);
  assert.deepEqual(blockingUnknowns(s,ctx.sources),['accounts.taxable']);
  assert.equal(planIssues(s,ctx.sources)[0].level,'unknown');
  applyCell(ctx,s,row,0);
  assert.equal(isUnknown(ctx,s,row),false);
  assert.equal(s.accounts.taxable,0);
  assert.deepEqual(blockingUnknowns(s,ctx.sources),[]);
});

test('unused Unknowns do not block, matching the classic planner',()=>{
  const s=plan(p=>{p.longTermCare.enabled=false;p.rothConversion.enabled=false;p.guaranteedIncome.annualIncome=0;}),ctx=contextFor(s);
  for(const path of ['longTermCare.annualCost','rothConversion.marginalRateCap','guaranteedIncome.startAge','spouseAccounts.pretax'])clearCell(ctx,s,{path});
  assert.deepEqual(blockingUnknowns(s,ctx.sources),[]);
});

test('edits carry the same side effects as a classic field commit',()=>{
  const s=plan(),ctx=contextFor(s);
  s.budget.isAppliedToAnnualBaseSpending=true;
  applyCell(ctx,s,rowFor('spending.annualBaseSpending'),80000);
  assert.equal(s.spending.annualBaseSpending,80000);
  assert.equal(s.budget.isAppliedToAnnualBaseSpending,false);
  assert.equal(s.budget.estimateNeedsReview,true);
  assert.equal(ctx.exampleIds.has(s.id),false);
  assert.equal(ctx.sources[s.id]['spending.annualBaseSpending'],'Entered');

  const date=addCalendarMonths(localCalendarDate(),120);
  applyCell(ctx,s,rowFor('household.retirementDate'),date);
  assert.equal(s.household.retirementDate,date);
  assert.equal(validateScenario(simulationScenario(s,100)).length,0);
  assert.ok(Math.abs(s.household.retirementAge-(s.household.currentAge+10))<=1);

  clearCell(ctx,s,rowFor('rothHistory.contributionBasis'));
  applyCell(ctx,s,rowFor('accounts.roth'),0);
  assert.equal(s.rothHistory.contributionBasis,0);
  assert.equal(ctx.sources[s.id]['rothHistory.contributionBasis'],'Entered');
});

test('dates accept common formats and follow the retirement rules',()=>{
  const s=plan(),row=rowFor('household.retirementDate'),birthday=rowFor('household.birthday');
  assert.equal(parseCell(row,'10/1/2045',s).value,'2045-10-01');
  assert.equal(parseCell(row,'2045-10-01',s).value,'2045-10-01');
  assert.equal(parseCell(row,'10-1-45',s).value,'2045-10-01');
  assert.ok(parseCell(row,'13/1/2045',s).error);
  assert.ok(parseCell(row,'2/30/2045',s).error);
  assert.match(parseCell(row,'1/1/2001',s).error,/today or later/);
  assert.match(parseCell(birthday,'1/1/2099',s).error,/before today/);
  s.household.alreadyRetired=true;
  assert.equal(parseCell(row,'1/1/2021',s).value,'2021-01-01');
  assert.match(parseCell(row,'1/1/2099',s).error,/between/);
});

test('switching retired status swaps its date and back again',()=>{
  const s=plan(),ctx=contextFor(s),row=rowFor('household.alreadyRetired'),date=s.household.retirementDate;
  applyCell(ctx,s,row,true);
  assert.equal(s.household.alreadyRetired,true);
  assert.equal(s.household.retirementDate,'');
  assert.deepEqual(blockingUnknowns(s,ctx.sources),[]);
  applyCell(ctx,s,row,false);
  assert.equal(s.household.alreadyRetired,false);
  assert.equal(s.household.retirementDate,date);
});

test('turning off the diagnostics the sheet skips leaves results identical',()=>{
  const s=plan(),sim=simulationScenario(s,120);
  const full=runSimulation(structuredClone(sim));
  const sheet=runSimulation(structuredClone(sim),()=>{},{includePathPoints:false,includeRiskAnalysis:false});
  for(const key of ['successProbability','medianEndingBalance','pessimisticEndingBalance','optimisticEndingBalance','medianFailureAge','balanceBands','notFailedByAge','steadySimulation'])assert.deepEqual(sheet[key],full[key],key);
  assert.deepEqual(sheet.todayDollars.balanceBands,full.todayDollars.balanceBands);
  assert.equal(sheet.todayDollars.medianEndingBalance,full.todayDollars.medianEndingBalance);
});

test('the dollar basis switches amounts and leaves counts alone',()=>{
  const r=fill(plan()),today=shownResult(r,'today'),future=shownResult(r,'future');
  assert.equal(today.medianEndingBalance,r.todayDollars.medianEndingBalance);
  assert.equal(future.medianEndingBalance,r.medianEndingBalance);
  assert.equal(today.successProbability,future.successProbability);
  assert.equal(shownResult(null,'today'),null);
});

test('readiness differences use counts at preview size and percent above it',()=>{
  const make=(p,n)=>({successProbability:p,provenance:{simulationCount:n}});
  assert.equal(readinessDelta(make(.93,100),make(.9,100)),'+3');
  assert.equal(readinessDelta(make(.9,100),make(.93,100)),'−3');
  assert.equal(readinessDelta(make(.931,200),make(.9,200)),'+3.1');
  assert.equal(readinessDelta(make(.9,100),make(.9,100)),'=');
  assert.equal(readinessDelta(make(.9,100),make(.9,200)),'');
  const same=make(.9,100);assert.equal(readinessDelta(same,same),'');
});

test('the Forecast sheet reports each plan’s own bands by whole-year age',()=>{
  const a=shownResult(fill(plan()),'today'),b=shownResult(fill(plan(p=>{p.spending.annualBaseSpending=40000;})),'today');
  const table=forecastTable([a,b,null]);
  assert.ok(table.length>5);
  for(const [i,row] of table.entries()){
    if(i)assert.ok(row.age>table[i-1].age);
    assert.equal(row.plans[2],null);
  }
  const hit=table.find(row=>row.plans[0]),band=a.balanceBands.filter(x=>Math.floor(Math.round(x.age*12)/12)===hit.age).at(-1);
  assert.equal(hit.plans[0].median,band.median);
  assert.equal(hit.plans[0].p10,band.pessimistic);
  assert.equal(hit.plans[0].p90,band.optimistic);
  assert.ok(hit.plans[0].funded>=0&&hit.plans[0].funded<=1);
});

test('Yearly balances match the run and flows add up to its months',()=>{
  const married=plan(p=>{p.household.filingStatus='Married';p.spouseAccounts.pretax=100000;});
  for(const s of [plan(),married]){
    const r=fill(s),rows=yearlyTable(r,'future'),details=r.steadySimulation.monthlyDetails;
    assert.ok(rows.length>2);
    assert.equal(rows[0].month,0);
    assert.equal(rows[0].spending,0);
    for(const row of rows){
      const month=details.find(p=>p.month===row.month);
      assert.equal(row.portfolio,month.portfolio);
      assert.equal(row.netAssets,month.netAssets);
      assert.ok(Math.abs(row.pretax+row.roth+row.employerRoth+row.taxable+row.cash-row.portfolio)<.01,`accounts add to portfolio at month ${row.month}`);
    }
    const flow=key=>details.reduce((n,p)=>n+(p.cashFlow?.[key]||0),0),total=key=>rows.reduce((n,row)=>n+row[key],0);
    assert.ok(Math.abs(total('spending')-flow('expenses'))<.01);
    assert.ok(Math.abs(total('socialSecurity')-flow('socialSecurity'))<.01);
    assert.ok(Math.abs(total('withdrawals')-flow('additionalWithdrawal'))<.01);
    const today=yearlyTable(r,'today');
    assert.equal(today.length,rows.length);
    assert.ok(today.at(-1).spending<rows.at(-1).spending,'today’s dollars remove inflation');
  }
  assert.deepEqual(yearlyTable(null,'today'),[]);
});

test('fan geometry stays inside its box on a shared scale',()=>{
  const r=fill(plan()),peak=fanPeak([r,null]),paths=fanPaths(r.balanceBands,peak);
  for(const point of [...paths.band.split(' '),...paths.median.split(' ')]){
    const [x,y]=point.split(',').map(Number);
    assert.ok(x>=0&&x<=100&&y>=0&&y<=30,point);
  }
  assert.equal(fanPaths([],peak),null);
});

test('totals count the accounts a plan really uses',()=>{
  const s=plan(p=>{p.accounts={pretax:100,roth:20,taxable:3,cash:4};p.spouseAccounts={pretax:1000,roth:1000};});
  assert.equal(totalSavings(s),127);
  s.household.filingStatus='Married';
  assert.equal(totalSavings(s),2127);
});

test('clipboard text and CSV are safe and round-trip',()=>{
  const rows=[['Plan','A, B'],['Pre-tax',1250000],['Note','say "hi"'],['Bad','=HYPERLINK("x")'],['Neg','-5']];
  const csv=toCsv(rows);
  assert.match(csv,/"A, B"/);
  assert.match(csv,/"say ""hi"""/);
  assert.match(csv,/"'=HYPERLINK\(""x""\)"/);
  assert.match(csv,/Neg,-5/);
  assert.deepEqual(parseTsv(toTsv([['a','b'],['c','d']])),[['a','b'],['c','d']]);
  assert.deepEqual(parseTsv('1\t2\r\n3\t4\r\n'),[['1','2'],['3','4']]);
});

test('the sheet page uses the shared store and the release hash',()=>{
  const script=readFileSync(new URL('../dist/sheet.js',import.meta.url),'utf8'),app=readFileSync(new URL('../dist/app.js',import.meta.url),'utf8');
  assert.match(script,/retirement-readiness-lab-sites-v1/);
  assert.match(app,/retirement-readiness-lab-sites-v1/);
  assert.match(script,/navigator\.locks\.request\(STORE_KEY/);
  const build=readFileSync(new URL('../scripts/build-sites.mjs',import.meta.url),'utf8');
  assert.match(build,/sheet\.js/);
});

test('a queued plan runs on the inputs it was queued with, so an edit cannot file its result under the old key',()=>{
  const today=localCalendarDate(),fingerprint=s=>resultFingerprint(s,today);
  const source=readFileSync(new URL('../dist/sheet.js',import.meta.url),'utf8');
  const block=source.slice(source.indexOf('function cancelJob'),source.indexOf('let recalcTimer')).replaceAll('import.meta.url',JSON.stringify('https://example.test/sheet.js'));
  const workers=[],remembered=new Map();
  class Worker{constructor(){workers.push(this);}postMessage(message){this.posted=structuredClone(message);}terminate(){this.terminated=true;}}
  const running=plan(s=>{s.id='running';}),queued=plan(s=>{s.id='queued';s.spending.annualBaseSpending=60000;});
  const context=vm.createContext({app:{plans:[running,queued],jobs:new Map(),queue:[],results:new Map(),full:false,startState:'idle'},navigator:{hardwareConcurrency:2},Worker,URL,
    simulationScenario:sheetModel.simulationScenario,countsAsSimulationStart:sheetModel.countsAsSimulationStart,fingerprint,fullPaths:()=>100,OPTIONS:{},
    remember:(key,result)=>remembered.set(key,result),refresh(){},setNote(){},renderCalc(){},reportUsage(){}});
  vm.runInContext(block+';this.enqueue=enqueue;this.pump=pump;',context);
  context.enqueue(running,100);context.enqueue(queued,100);context.pump();
  assert.equal(workers.length,1,'one worker, so the second plan waits');
  const queuedKey=fingerprint(queued)+'|100';
  queued.spending.annualBaseSpending=5;
  workers[0].onmessage({data:{type:'result',result:{plan:'running'}}});
  assert.equal(workers.length,2,'the waiting plan starts when the worker frees up');
  assert.equal(workers[1].posted.scenario.spending.annualBaseSpending,60000,'it must run the inputs it was queued with');
  workers[1].onmessage({data:{type:'result',result:{plan:'queued'}}});
  assert.equal(remembered.get(queuedKey)?.plan,'queued');
  assert.equal(fingerprint(workers[1].posted.scenario)+'|100',queuedKey,'the key must describe exactly what ran');
});

test('the sheet repairs locked stores the same way the planner does',()=>{
  const script=readFileSync(new URL('../dist/sheet.js',import.meta.url),'utf8');
  assert.match(script,/repaired=repairScenarios\(plans\)/);
  assert.match(script,/markRepairedUnknown\(sources,repaired\)/);
});

// Runs the sheet's real store code against a fake browser, so these tests exercise what ships.
function storeHarness({raw=null,locks=true,quota=false}={}){
  const source=readFileSync(new URL('../dist/sheet.js',import.meta.url),'utf8');
  const stored={value:raw,full:quota,held:null};
  const storage={getItem:()=>stored.value,setItem(_key,value){if(stored.full)throw new Error('QuotaExceededError');stored.value=value;}};
  const locksApi={request:async(_name,fn)=>{if(stored.held)await stored.held;return fn();}};
  const notes=[],listeners={};
  const context=vm.createContext({
    ...model,normalizeInputSources,...planReview,structuredClone,JSON,Promise,Set,Map,String,Array,Object,Number,
    localStorage:storage,navigator:locks?{locks:locksApi}:{},
    window:{addEventListener:(name,fn)=>{listeners[name]=fn;}},
    setNote:(text,error)=>{notes.push({text,error});},refresh(){},renderNote(){},copyLabSets:x=>x,
  });
  const core=source.slice(source.indexOf('const STORE_KEY'),source.indexOf('const $=id'))+source.slice(source.indexOf('let lastSavedRaw'),source.indexOf('function snapshot'))+source.slice(source.indexOf("window.addEventListener('storage'"),source.indexOf('// Same access check'));
  vm.runInContext(core+';this.h={app,loadInto,persist,block:()=>storeBlocked,setBlocked:v=>{storeBlocked=v;}};',context);
  return {...context.h,stored,notes,fire:name=>listeners[name]};
}
const validStore=edit=>{const s=plan();edit?.(s);return JSON.stringify({scenarios:[s],selectedId:s.id,hasStartedPlan:true});};

test('a failed save stays visible and leaving warns, until a later save works',async()=>{
  const h=storeHarness({raw:validStore()});h.loadInto();
  const before=h.stored.value;h.stored.full=true;
  h.app.plans[0].accounts.pretax=2000;
  assert.equal(await h.persist(),false);
  assert.equal(h.stored.value,before,'the store keeps its last good copy');
  assert.match(h.app.saveFailure,/lost when you leave/);
  let warned=false;h.fire('beforeunload')({preventDefault(){warned=true;}});assert.ok(warned,'leaving must warn');
  h.stored.full=false;
  assert.equal(await h.persist(),true);
  assert.equal(h.app.saveFailure,'');
  warned=false;h.fire('beforeunload')({preventDefault(){warned=true;}});assert.equal(warned,false);
});

test('a browser without Web Locks is told plainly that edits here will not be kept',async()=>{
  const h=storeHarness({raw:validStore(),locks:false});h.loadInto();
  assert.equal(await h.persist(),false);
  assert.match(h.app.saveFailure,/lost when you leave/);
  assert.doesNotMatch(h.app.saveFailure,/back up/i,'the planner never sees these edits, so it cannot back them up');
});

test('repairing an unreadable store in another tab lets this tab save again',async()=>{
  const h=storeHarness({raw:'{bad'});
  assert.throws(()=>h.loadInto());h.setBlocked(true);
  assert.equal(await h.persist(),false,'blocked while unreadable');
  h.stored.value=validStore(s=>{s.name='Repaired';});
  h.fire('storage')({key:'retirement-readiness-lab-sites-v1'});
  assert.equal(h.block(),false,'a good reload must clear the block');
  assert.equal(h.app.plans[0].name,'Repaired');
  h.app.plans[0].accounts.pretax=123;
  assert.equal(await h.persist(),true);
  assert.equal(JSON.parse(h.stored.value).scenarios[0].accounts.pretax,123);
});

test('an unreadable store written by another tab is never overwritten, even with no storage event',async()=>{
  const h=storeHarness({raw:validStore()});h.loadInto();
  const unreadable='{"scenarios":"UNREADABLE-BACKUP"}';h.stored.value=unreadable;
  assert.equal(await h.persist(),false);
  assert.equal(h.stored.value,unreadable);
  assert.equal(h.block(),true,'the failed reload must block saving');
  assert.equal(await h.persist(),false);
  assert.equal(h.stored.value,unreadable,'a second edit must not overwrite it');
});

test('a save queued before a conflict cannot overwrite the other tab',async()=>{
  const h=storeHarness({raw:validStore()});h.loadInto();
  let release;h.stored.held=new Promise(resolve=>{release=resolve;});
  h.app.plans[0].accounts.pretax=200000;const first=h.persist();
  h.app.plans[0].accounts.pretax=250000;const second=h.persist();
  h.stored.value=validStore(s=>{s.name='Other tab edit';});const other=h.stored.value;
  release();h.stored.held=null;
  assert.equal(await first,false);assert.equal(await second,false);
  assert.equal(h.stored.value,other,'the other tab edit survives');
  assert.equal(h.app.plans[0].name,'Other tab edit');
});

// The display code, run with a stubbed plan state.
function displayHarness(state,extra={}){
  const source=readFileSync(new URL('../dist/sheet.js',import.meta.url),'utf8');
  const els={calc:{textContent:''},recalc:{classList:{toggle(){}},innerHTML:''},note:{classList:{toggle(_c,on){els.note.error=on;}},textContent:''}};
  const context=vm.createContext({...sheetModel,letter:i=>String(i),$:id=>els[id],basePlan:()=>extra.plan,activeCell:()=>null,LIVE_ESTIMATE_PATHS:200,planState:()=>state,
    storeBlocked:false,recalcPending:false,app:{plans:[extra.plan],basis:'today',yearlyId:extra.plan.id,jobs:new Map(),queue:[],sel:{r:0,c:0},noteOverride:'',saveFailure:'',calcFailure:'',...extra.app}});
  const pick=(from,to)=>source.slice(source.indexOf(from),source.indexOf(to));
  vm.runInContext(pick('function forecastModel','const buildModel')+pick('function renderNote','function renderAggregate')+pick('function renderCalc','function renderToolbar')+';this.d={forecastModel,yearlyModel,renderNote,renderCalc};',context);
  return {...context.d,els};
}
const staleCells=model=>model.rows.flatMap(row=>row.cells).filter(cell=>/\bstale\b/.test(cell.cls||'')).length;

test('old numbers are marked as old on Forecast and Yearly, and the status does not say Ready',()=>{
  const s=plan(),result=runSimulation(sheetModel.simulationScenario(s,100));
  const current=displayHarness({result,stale:false,blocked:false},{plan:s});
  assert.equal(staleCells(current.forecastModel()),0);assert.equal(staleCells(current.yearlyModel()),0);
  current.renderCalc();assert.equal(current.els.calc.textContent,'Ready');
  const old=displayHarness({result,stale:true,blocked:false},{plan:s});
  assert.ok(staleCells(old.forecastModel())>0,'Forecast must mark them');
  assert.ok(staleCells(old.yearlyModel())>0,'Yearly must mark them');
  old.renderCalc();assert.equal(old.els.calc.textContent,'Out of date');
});

test('a worker failure stays on screen and in the status',()=>{
  const s=plan(),failed=displayHarness({result:null,stale:false,blocked:false},{plan:s,app:{calcFailure:'The calculation couldn’t start.'}});
  failed.renderCalc();assert.equal(failed.els.calc.textContent,'Stopped');
  failed.renderNote();assert.match(failed.els.note.textContent,/couldn’t start/);assert.equal(failed.els.note.error,true);
});

test('a failed save outranks other notes in the status bar',()=>{
  const s=plan(),h=displayHarness({result:null,stale:false,blocked:false},{plan:s,app:{saveFailure:'lost when you leave',noteOverride:'something else'}});
  h.renderNote();assert.equal(h.els.note.textContent,'lost when you leave');assert.equal(h.els.note.error,true);
});

test('a downloaded CSV says so when its numbers are out of date',()=>{
  const source=readFileSync(new URL('../dist/sheet.js',import.meta.url),'utf8');
  assert.match(source.slice(source.indexOf('function downloadCsv'),source.indexOf('function setTab')),/Out of date/);
});

test('a full result cache never drops a result a plan on screen is still showing',()=>{
  const source=readFileSync(new URL('../dist/sheet.js',import.meta.url),'utf8');
  const code=source.slice(source.indexOf('function remember'),source.indexOf('function currentResult'));
  const today=localCalendarDate(),fingerprint=s=>resultFingerprint(s,today);
  const untouched=plan(s=>{s.id='untouched';}),edited=plan(s=>{s.id='edited';});
  const app={results:new Map(),plans:[untouched,edited]};
  const context=vm.createContext({app,fingerprint});
  vm.runInContext(code+';this.remember=remember;',context);
  const key=fingerprint(untouched)+'|100';
  context.remember(key,{plan:'untouched'});
  const firstEdit=()=>{edited.accounts.pretax=1000;return fingerprint(edited)+'|100';};
  const oldest=firstEdit();
  for(let i=0;i<120;i++){edited.accounts.pretax=1000+i;context.remember(fingerprint(edited)+'|100',{i});}
  assert.equal(app.results.get(key)?.plan,'untouched','a plan nobody edited keeps its result');
  assert.ok(app.results.has(fingerprint(edited)+'|100'),'and so does the plan being edited');
  assert.ok(app.results.size<=48,'the cache stays bounded');
  assert.equal(app.results.has(oldest),false,'superseded results are the ones dropped');
});

test('text that only looks like a number is refused, never misread',()=>{
  const s=plan(),money=rowFor('accounts.pretax'),pct=rowFor('market.stockMeanReturn'),int=rowFor('socialSecurity.claimAge'),year=rowFor('rothHistory.firstContributionYear');
  // A misaligned paste reaches these. Each used to become a different number with no complaint.
  for(const text of ['10/1/2029','2029-10-01','(1,234)','3500/12','100-20','2,5','=2,5*2','=12,34+1','0x10'])assert.ok(parseCell(money,text,s).error,`money ${text} -> ${JSON.stringify(parseCell(money,text,s))}`);
  for(const text of ['2,5','0x10','10/1','1-2'])assert.ok(parseCell(pct,text,s).error,`percent ${text}`);
  for(const text of ['6,7','0x10','1/2'])assert.ok(parseCell(int,text,s).error,`whole number ${text}`);
  for(const text of ['2018-10-1','2018/1'])assert.ok(parseCell(year,text,s).error,`year ${text}`);
  // A percent sign in a percent cell means that percent, not a percent of a percent.
  assert.equal(parseCell(pct,'=2.5%*2',s).value,.05);
  assert.equal(parseCell(pct,'=2.5*2',s).value,.05);
  // Everything the sheet documents still works, and = still opens every operator.
  for(const [text,expected] of [['40k',40000],['$1.2M',1200000],['1,250,000',1250000],['$1,000.50',1000.5],['=3500*12',42000],['3500*12',42000],['1000+500',1500],['=1,000*12',12000],['=3500/12*12',3500],['=100-20',80],['=(1000+500)*2',3000],['(1000+500)*2',3000],['-5',-5],['1e3',1000]])assert.equal(parseCell(money,text,s).value,expected,text);
  for(const [text,expected] of [['2.5',.025],['2.5%',.025],['-1',-.01],['=5/2',.025]])assert.equal(parseCell(pct,text,s).value,expected,text);
  assert.equal(parseCell(int,'1,000',s).value,1000);
  // Dates still belong in date cells.
  assert.equal(parseCell(rowFor('household.retirementDate'),'10/1/2039',s).value,'2039-10-01');
});

test('an impossible pension age is an error, and one past the plan’s own horizon is a warning',()=>{
  const startError=s=>validateScenario(s).filter(text=>/pension start age/i.test(text));
  const startWarning=s=>scenarioWarnings(s).filter(text=>/pension.*maximum modeling age/i.test(text));
  const pays=plan(s=>{s.guaranteedIncome.annualIncome=60000;s.guaranteedIncome.startAge=66;});
  assert.deepEqual(startError(pays),[]);assert.deepEqual(startWarning(pays),[]);
  const typo=plan(s=>{s.guaranteedIncome.annualIncome=60000;s.guaranteedIncome.startAge=660;});
  assert.equal(startError(typo).length,1,'660 for 66 must be caught');
  // The modeling age is a cutoff, not a lifespan, so these plans still run correctly.
  const pastHorizon=plan(s=>{s.guaranteedIncome.annualIncome=60000;s.guaranteedIncome.startAge=95;s.household.targetEndAge=90;});
  assert.deepEqual(startError(pastHorizon),[],'a plan that stops early is still valid');
  assert.equal(startWarning(pastHorizon).length,1,'but it should say the pension is not counted');
  const atHorizon=plan(s=>{s.guaranteedIncome.annualIncome=60000;s.guaranteedIncome.startAge=90;s.household.targetEndAge=90;});
  assert.deepEqual(startError(atHorizon),[]);assert.deepEqual(startWarning(atHorizon),[]);
  // A zero pension's other fields are unused, so they stay ignored.
  const unused=plan(s=>{s.guaranteedIncome.annualIncome=0;s.guaranteedIncome.startAge=660;});
  assert.deepEqual(startError(unused),[]);assert.deepEqual(startWarning(unused),[]);
  const spouse=edit=>plan(s=>{s.household.filingStatus='Married';s.household.spouseBirthday='1970-01-01';s.household.spouseRetirementDate=s.household.retirementDate;s.spouseIncome.annualPension=30000;edit(s);});
  assert.equal(startError(spouse(s=>{s.spouseIncome.pensionStartAge=660;})).length,1,'the spouse pension too');
  assert.equal(startWarning(spouse(s=>{s.spouseIncome.pensionStartAge=95;s.household.targetEndAge=90;})).length,1);
});

test('a change of access restarts running jobs, so no column is left waiting',async()=>{
  const source=readFileSync(new URL('../dist/sheet.js',import.meta.url),'utf8');
  const code=source.slice(source.indexOf('async function loadAccess'),source.indexOf('async function start'));
  const calls=[],app={tier:'free',plans:[plan()]};let tier='pro';
  const context=vm.createContext({app,JSON,sessionStorage:{getItem:()=>null},authHeaders:async()=>({}),applyProSimulationDefault(){},
    fetch:async()=>({ok:true,json:async()=>({tier})}),stopAll:()=>calls.push('stopAll'),refresh:()=>calls.push('refresh')});
  vm.runInContext(code+';this.loadAccess=loadAccess;',context);
  await context.loadAccess();
  assert.deepEqual(calls,['stopAll','refresh'],'jobs at the old path count must stop before the plans recalculate');
  calls.length=0;await context.loadAccess();
  assert.deepEqual(calls,[],'no change means running jobs carry on');
});
