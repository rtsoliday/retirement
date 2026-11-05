import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {baseScenario,sampleScenarios,prepareCalendarScenario,validateScenario,addCalendarMonths,localCalendarDate} from '../dist/model.js';
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
