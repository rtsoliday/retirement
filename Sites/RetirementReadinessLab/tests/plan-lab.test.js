import test from 'node:test';
import assert from 'node:assert/strict';
import {sampleScenarios,prepareCalendarScenario,validateScenario,addCalendarMonths,localCalendarDate} from '../dist/model.js';
import {LAB_LEVERS,LAB_PRESETS,applyWhatIf,whatIfErrors,autoName,defaultLabSets,normalizeLabSets,copyLabSets,readinessDelta,labSummary,labCsv,labMetrics,SENSITIVITY_INPUTS,STRESS_TESTS} from '../dist/plan-lab.js';

const plan=()=>{const s=prepareCalendarScenario(sampleScenarios()[0],{needsReview:false});s.home.currentValue=300000;s.accounts.taxable=20000;return s;};
const result=(p,n=1000,extra={})=>({successProbability:p,medianEndingBalance:p*1e6,pessimisticEndingBalance:p*1e5,medianFailureAge:88,provenance:{simulationCount:n},...extra});

test('every lever reads its own default fields, applies a valid change and leaves the base plan untouched',()=>{
  const s=plan(),before=structuredClone(s);
  for(const lever of LAB_LEVERS){
    assert.ok(lever.available(s),lever.key);
    const draft=Object.fromEntries(lever.fields(s).map(f=>[f.name,f.value]));
    if(lever.key==='oneTimeExpenses')Object.assign(draft,{label0:'Car',age0:String(Math.floor(s.household.retirementAge)+4),amount0:'30,000'});
    if(lever.key==='partTimeIncome')draft.annualNet='20k';
    const read=lever.read(draft,s);assert.equal(read.error,undefined,`${lever.key}: ${read.error}`);
    assert.deepEqual(whatIfErrors(s,{[lever.key]:read.value}),[],lever.key);
    assert.ok(lever.describe(read.value,s).length>0);assert.ok(lever.current(s).length>0);
  }
  assert.deepEqual(s,before);
});

test('what-ifs stack changes, name themselves and reject levers that do not apply',()=>{
  const s=plan(),date=addCalendarMonths(s.household.retirementDate,24);
  const changes={retirementDate:date,annualBaseSpending:68000,stockShift:-20,healthcareChange:.25,returnCap:.07};
  const next=applyWhatIf(s,changes);
  assert.equal(next.household.retirementDate,date);assert.equal(next.spending.annualBaseSpending,68000);assert.equal(next.market.stockMeanReturn,.07);
  assert.equal(next.postRetirementAllocation.stockUnder30x,.8);assert.equal(next.healthcare.preMedicareMonthlyPremium,s.healthcare.preMedicareMonthlyPremium*1.25);
  assert.match(autoName({annualBaseSpending:68000},s),/^Spend \$68,000 a year$/);
  const retired=plan();retired.household.alreadyRetired=true;retired.household.retirementDate='';assert.match(whatIfErrors(retired,{retirementDate:date}).join(' '),/does not apply/);
  const renter=plan();renter.home.currentValue=0;assert.equal(LAB_LEVERS.find(l=>l.key==='homePlan').available(renter),false);
  assert.ok(whatIfErrors(s,{annualBaseSpending:-1}).length);
});

test('presets skip changes the plan already has; Pro sets start with three and free sets start empty',()=>{
  const s=plan();s.socialSecurity.claimAge=70;s.rothConversion.enabled=true;
  const keys=LAB_PRESETS.filter(p=>p.changes(s)).map(p=>p.key);assert.ok(!keys.includes('claim-70'));assert.ok(!keys.includes('roth'));assert.ok(keys.includes('later'));
  const pro=defaultLabSets(plan(),true),free=defaultLabSets(plan(),false);
  assert.equal(pro.sets[0].whatIfs.length,3);assert.equal(free.sets[0].whatIfs.length,0);assert.equal(pro.activeSet,pro.sets[0].id);
  for(const w of pro.sets[0].whatIfs)assert.deepEqual(validateScenario(applyWhatIf(plan(),w.changes)),[]);
});

test('saved sets normalize, cap their sizes and copy with fresh IDs',()=>{
  const s=plan();s.id='p';const many={activeSet:'s3',sets:Array.from({length:12},(_,i)=>({id:'s'+i,name:'Set '+i,whatIfs:Array.from({length:6},(_,j)=>({id:`w${i}-${j}`,name:'W',changes:{claimAge:70,bogus:true}}))}))};
  const clean=normalizeLabSets({p:many,other:many},[s]).p;
  assert.equal(clean.sets.length,8);assert.equal(clean.sets[0].whatIfs.length,4);assert.deepEqual(clean.sets[0].whatIfs[0].changes,{claimAge:70});assert.equal(clean.activeSet,'s3');
  assert.deepEqual(normalizeLabSets({p:{sets:'bad'}},[s]),{});assert.deepEqual(normalizeLabSets(null,[s]),{});
  const copy=copyLabSets(clean);assert.notEqual(copy.sets[3].id,'s3');assert.equal(copy.activeSet,copy.sets[3].id);assert.deepEqual(copy.sets[0].whatIfs[0].changes,{claimAge:70});
});

test('readiness changes use points, or counts for a 100-path preview',()=>{
  assert.equal(readinessDelta(result(.89),result(.78)),'+11 pts');assert.equal(readinessDelta(result(.785),result(.78)),'+0.5 pts');
  assert.equal(readinessDelta(result(.7),result(.78)),'−8 pts');assert.equal(readinessDelta(result(.5),result(.5)),'Same');
  assert.equal(readinessDelta(result(.55,100),result(.5,100)),'+5 of 100');
});

test('saved comparisons reject malformed values for every lever without losing valid changes',()=>{
  const s=plan(),normalize=changes=>normalizeLabSets({[s.id]:{activeSet:'set',sets:[{id:'set',whatIfs:[{id:'w',changes}]}]}},[s])[s.id].sets[0].whatIfs[0].changes;
  const valid={};
  for(const lever of LAB_LEVERS){
    const draft=Object.fromEntries(lever.fields(s).map(f=>[f.name,f.value]));
    if(lever.key==='oneTimeExpenses')Object.assign(draft,{label0:'Car',age0:String(Math.floor(s.household.retirementAge)+4),amount0:'30,000'});
    if(lever.key==='partTimeIncome')draft.annualNet='20k';
    const read=lever.read(draft,s);assert.equal(read.error,undefined,lever.key);valid[lever.key]=read.value;
    for(const malformed of [null,undefined,'bad',['bad'],{},true,NaN,Infinity]){
      const retained=lever.key==='annualBaseSpending'?{claimAge:70}:{annualBaseSpending:60000},changes=normalize({[lever.key]:malformed,...retained});
      assert.deepEqual(changes,retained,`${lever.key}: ${String(malformed)}`);
    }
  }
  assert.deepEqual(normalize(valid),valid);assert.deepEqual(normalize({oneTimeExpenses:[]}),{oneTimeExpenses:[]});
  const malformed={claimAge:71,annualSavings:-1,stockShift:1,returnCap:.123,spendingPathModel:'Random',withdrawalOrder:'Random',healthcareChange:1,
    rothConversion:{enabled:'true',marginalRateCap:.22},cashFirst:{enabled:true,drawdownTrigger:'-0.01'},partTimeIncome:{annualNet:20000,endAge:70.5},
    homePlan:{saleAge:75,mode:'rent',downsizeShare:.5,monthlyRent:-1},oneTimeExpenses:[{age:75,amount:1000,label:{}}],retirementDate:'2026-02-30'};
  assert.deepEqual(normalize(malformed),{});
  // Keep well-formed changes that have become inapplicable as the base plan ages.
  const older={retirementDate:'2001-01-01',partTimeIncome:{annualNet:20000,endAge:18},oneTimeExpenses:[{age:18,amount:1000,label:'Car'}],homePlan:{saleAge:18,mode:'downsize',downsizeShare:.5,monthlyRent:0}};
  assert.deepEqual(normalize(older),older);assert.ok(whatIfErrors(s,older).length);
});

test('the summary names the best what-if, the biggest single change, the most resilient plan and the lowest tax',()=>{
  const lab=tax=>({planLab:{lifetimeTax:tax,surchargeYears:1,conversions:0}});
  const rows=[{id:'baseline',baseline:true,label:'Current plan',changes:{},result:result(.78,1000,lab(214000))},{id:'a',label:'Retire later',changes:{retirementDate:'x'},result:result(.89,1000,lab(236000))},{id:'b',label:'Claim at 70 + Roth',changes:{claimAge:70,rothConversion:{}},result:result(.84,1000,lab(175600))},{id:'c',label:'Both',changes:{retirementDate:'x',annualBaseSpending:1},result:result(.94,1000,lab(226000))}];
  const stress={rows:STRESS_TESTS.map(t=>({key:t.key,results:{baseline:result(.5),c:result(.8)}}))};
  const s=labSummary(rows,{stress});
  assert.equal(s.best.id,'c');assert.equal(s.delta,'+16 pts');assert.equal(s.single.row.id,'a');assert.equal(s.single.delta,'+11 pts');
  assert.equal(s.lowTax.row.id,'b');assert.equal(s.lowTax.saving,38400);assert.equal(s.resilient.row.id,'c');assert.match(s.headline,/Both gives the highest readiness/);
  assert.equal(labSummary(rows.slice(0,1)),null);
  const worse=labSummary([rows[0],{...rows[1],result:result(.5)}]);assert.equal(worse.improved,false);assert.match(worse.headline,/None of these what-ifs beats/);
});

test('CSV exports escape names and include year-by-year balances',()=>{
  const r=result(.8,1000,{planLab:{lifetimeTax:1000.4,surchargeYears:2,conversions:0}});
  const csv=labCsv([{label:'Plan, "A"',changesText:'Spend less',metrics:labMetrics(r,r)}],{yearly:[{age:70,values:[123456.7]}]});
  assert.match(csv,/^Plan,Changes,Paths/);assert.match(csv,/\n"Plan, ""A""",Spend less,1000,80\.0%,200,800000,80000,88\.0,1000,2,0\n/);assert.match(csv,/\nAge,"Plan, ""A"" median balance"\n70,123457\n$/);
});

test('sensitivity inputs move one assumption each way and skip a retirement date before today',()=>{
  const s=plan();
  for(const input of SENSITIVITY_INPUTS)for(const side of ['low','high']){const x=structuredClone(s);assert.equal(input[side](x),true,input.key+side);assert.notDeepEqual(x,s,input.key+side);}
  const soon=plan();soon.household.retirementDate=addCalendarMonths(localCalendarDate(),3);
  assert.equal(SENSITIVITY_INPUTS.find(i=>i.key==='retirement').low(structuredClone(soon)),false);
});
