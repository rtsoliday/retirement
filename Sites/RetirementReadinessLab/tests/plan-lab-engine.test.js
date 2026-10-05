import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,sampleScenarios,prepareCalendarScenario,validateScenario,normalizeScenario,scenarioEngineVersion,usesPlanLabInputs,primaryRetirementAge} from '../dist/model.js';
import {runSimulation,runOne,runSteadySimulation,JavaRandom} from '../dist/engine.js';
import {resultFingerprint} from '../dist/result-cache.js';

const TODAY='2026-10-04';
// A calendar plan with modest savings so readiness responds to each change.
function plan({separate=false,married=false}={}){
  const s=prepareCalendarScenario(sampleScenarios()[0],{today:TODAY,needsReview:false});
  s.household.asOfDate=TODAY;s.numberOfSimulations=200;
  if(married){s.household.filingStatus='Married';s.household.spouseBirthday='1966-06-01';}
  if(separate){s.household.separatePeople=true;s.household.spouseRetirementDate=s.household.retirementDate;}
  return s;
}
const age=s=>Math.floor(primaryRetirementAge(s));
const variants=[['pooled',{}],['separate couple',{separate:true,married:true}]];

test('new plan inputs default to unused and leave versions and result fingerprints unchanged',()=>{
  const s=plan();assert.equal(usesPlanLabInputs(s),false);assert.deepEqual(validateScenario(s),[]);
  const legacy=structuredClone(s);delete legacy.partTimeIncome;delete legacy.oneTimeExpenses;delete legacy.homePlan;delete legacy.withdrawalStrategy.withdrawalOrder;
  const normalized=normalizeScenario(legacy);
  assert.deepEqual(normalized.partTimeIncome,{annualNet:0,endAge:0});assert.deepEqual(normalized.oneTimeExpenses,[]);assert.equal(normalized.withdrawalStrategy.withdrawalOrder,'Standard');
  assert.equal(resultFingerprint(normalized,TODAY),resultFingerprint(legacy,TODAY));
  const changed=structuredClone(s);changed.partTimeIncome={annualNet:20000,endAge:age(s)+3};
  assert.match(scenarioEngineVersion(changed),/plan-lab-v2$/);assert.doesNotMatch(scenarioEngineVersion(s),/plan-lab/);
  const expense=structuredClone(s);expense.oneTimeExpenses=[{age:age(s)+3,amount:1000,label:'Car'}];
  assert.match(scenarioEngineVersion(expense),/plan-lab-v1$/,'other new inputs keep their existing result version');
  assert.notEqual(resultFingerprint(changed,TODAY),resultFingerprint(s,TODAY));
});

test('invalid part-time, one-time, home-sale and withdrawal-order inputs are rejected',()=>{
  const cases=[
    s=>{s.partTimeIncome={annualNet:10000,endAge:age(s)};},
    s=>{s.partTimeIncome={annualNet:-1,endAge:0};},
    s=>{s.oneTimeExpenses=[{age:age(s)+2.5,amount:1000,label:'Roof'}];},
    s=>{s.oneTimeExpenses=[{age:age(s)+2,amount:-5,label:''}];},
    s=>{s.homePlan={saleAge:age(s),mode:'downsize',downsizeShare:.5,monthlyRent:0};},
    s=>{s.homePlan={saleAge:age(s)+5,mode:'downsize',downsizeShare:.99,monthlyRent:0};},
    s=>{s.homePlan={saleAge:age(s)+5,mode:'boat',downsizeShare:.5,monthlyRent:0};},
    s=>{s.withdrawalStrategy.withdrawalOrder='Random';}
  ];
  for(const edit of cases){const s=plan();edit(s);assert.ok(validateScenario(s).length,JSON.stringify(s.partTimeIncome)+JSON.stringify(s.oneTimeExpenses)+JSON.stringify(s.homePlan));}
  const s=plan();s.oneTimeExpenses=[{age:1,amount:1,label:''},'bad'];assert.match(validateScenario(s).join(' '),/one-time expense/i);
});

for(const [name,options] of variants){
  test(`part-time take-home pay lowers withdrawals until its end age: ${name}`,()=>{
    const s=plan(options),start=age(s),work=structuredClone(s);work.partTimeIncome={annualNet:24000,endAge:start+4};
    const before=runSteadySimulation(s).monthlyDetails,after=runSteadySimulation(work).monthlyDetails;
    const first=after.find(p=>p.cashFlow&&p.age>=start+1),same=before.find(p=>p.month===first.month);
    assert.ok(first.cashFlow.expenses<same.cashFlow.expenses-1500,'take-home pay covers part of the monthly costs');
    const later=after.find(p=>p.cashFlow&&p.age>=start+6),laterBefore=before.find(p=>p.month===later.month);
    assert.ok(Math.abs(later.cashFlow.expenses-laterBefore.cashFlow.expenses)<laterBefore.cashFlow.expenses*.05,'no take-home pay after the end age');
    assert.ok(runSimulation(work).successProbability>=runSimulation(s).successProbability);
  });

  test(`one-time expenses are paid once, in today's dollars, at the chosen age: ${name}`,()=>{
    const s=plan(options),when=age(s)+3,cost=structuredClone(s);cost.oneTimeExpenses=[{age:when,amount:40000,label:'New car'}];
    const before=runSteadySimulation(s).monthlyDetails,after=runSteadySimulation(cost).monthlyDetails;
    // Later Medicare premiums can follow the higher income from paying it.
    const deltas=after.filter(p=>p.cashFlow).map(p=>p.cashFlow.expenses-before.find(q=>q.month===p.month).cashFlow.expenses);
    const paid=deltas.filter(d=>d>5000);
    assert.equal(paid.length,1);assert.ok(paid[0]>40000&&paid[0]<40000*1.5,'inflated from today');
    assert.ok(deltas.every(d=>d===paid[0]||Math.abs(d)<2000));
  });

  test(`downsizing and selling to rent change the home, cash and costs at the sale age: ${name}`,()=>{
    const s=plan(options);s.home.currentValue=400000;s.home.annualTaxesAndInsurance=6000;s.spending.annualBaseSpending=80000;
    const saleAge=age(s)+5,downsize=structuredClone(s),rent=structuredClone(s);
    downsize.homePlan={saleAge,mode:'downsize',downsizeShare:.5,monthlyRent:0};rent.homePlan={saleAge,mode:'rent',downsizeShare:.5,monthlyRent:2000};
    assert.deepEqual(validateScenario(downsize),[]);assert.deepEqual(validateScenario(rent),[]);
    const base=runSteadySimulation(s).monthlyDetails,small=runSteadySimulation(downsize).monthlyDetails,renter=runSteadySimulation(rent).monthlyDetails;
    const month=base.findIndex(p=>p.age>=saleAge);
    assert.ok(Math.abs(small[month+1].home-base[month+1].home/2)<base[month+1].home*.01,'half the home value remains');
    assert.ok(small[month+1].cash>base[month+1].cash+150000,'net proceeds arrive in cash');
    assert.equal(renter[month+1].home,0);assert.ok(renter[month+1].cash>base[month+1].cash+300000);
    const after=month+13;assert.ok(renter[after].cashFlow.expenses>base[after].cashFlow.expenses,'rent replaces owned-home costs');
  });

  test(`withdrawal order changes which accounts pay first: ${name}`,()=>{
    const s=plan(options);s.accounts.taxable=150000;
    const take=order=>{const x=structuredClone(s);x.withdrawalStrategy.withdrawalOrder=order;assert.deepEqual(validateScenario(x),[]);return runSteadySimulation(x).monthlyDetails.find(p=>p.cashFlow&&p.cashFlow.additionalWithdrawal>0).cashFlow.accountWithdrawals;};
    const standard=take('Standard'),taxable=take('TaxableFirst'),rothLast=take('RothLast');
    assert.ok(standard.pretax>0);assert.equal(standard.taxable,0);
    assert.ok(taxable.taxable>0);assert.equal(taxable.pretax,0);
    assert.ok(rothLast.pretax>0);assert.equal(rothLast.roth,0);
  });

  test(`stress tests keep the same paths and only add the bad event: ${name}`,()=>{
    const s=plan(options),base=runSimulation(s,()=>{},{includeRiskAnalysis:false});
    for(const stress of [{marketDrop:.3},{inflationShock:{rate:.05,months:120}},{forceCare:true}]){
      const r=runSimulation(s,()=>{},{includeRiskAnalysis:false,stress});
      assert.ok(r.successProbability<=base.successProbability,JSON.stringify(stress));assert.deepEqual(r.stress,stress);
    }
    // Longer lives need not lower readiness: two Social Security benefits can
    // outlast the survivor's lower spending. Every lifetime reaches 100.
    for(let i=0;i<20;i++){const path=runOne(s,new JavaRandom(BigInt(s.seed)+BigInt(i)),{stress:{minDeathAge:100}});assert.ok(path.deathAge>=100);}
    const unchanged=runSimulation(s,()=>{},{includeRiskAnalysis:false,stress:{marketDrop:0}});delete unchanged.generatedAtEpochMillis;delete unchanged.stress;
    const plain=structuredClone(base);delete plain.generatedAtEpochMillis;assert.deepEqual(unchanged,plain);
    const rng=()=>new JavaRandom(BigInt(s.seed)),drop=runOne(s,rng(),{stress:{marketDrop:.3}}),same=runOne(s,rng());
    assert.ok(drop.yearEnd[1]<same.yearEnd[1],'first-year balance falls');
  });

  test(`comparison totals report lifetime taxes, surcharge years and conversions: ${name}`,()=>{
    const s=plan(options);s.rothConversion.enabled=true;s.accounts.pretax=1500000;
    const r=runSimulation(s,()=>{},{includeRiskAnalysis:false,captureMetrics:true});
    assert.ok(r.planLab.lifetimeTax>0);assert.ok(r.planLab.conversions>0);assert.ok(r.planLab.surchargeYears>=0);
    assert.ok(r.planLab.taxByAge.length>10);assert.equal(r.planLab.taxByAge[0].count,200);
    assert.equal(runSimulation(s,()=>{},{includeRiskAnalysis:false}).planLab,undefined);
  });
}

test('legacy pooled age-based plans accept the new inputs too',()=>{
  const s=baseScenario();s.numberOfSimulations=50;s.partTimeIncome={annualNet:20000,endAge:70};s.oneTimeExpenses=[{age:75,amount:30000,label:'Roof'}];s.withdrawalStrategy.withdrawalOrder='TaxableFirst';
  assert.deepEqual(validateScenario(s),[]);
  const r=runSimulation(s,()=>{},{captureMetrics:true});assert.ok(Number.isFinite(r.successProbability));assert.ok(r.planLab.lifetimeTax>0);
});

for(const separate of [false,true])test(`initial Medicare estimates account for take-home work pay: ${separate?'separate people':'pooled'}`,()=>{
  const s=baseScenario();Object.assign(s.household,{currentAge:65,retirementAge:65,targetEndAge:70,separatePeople:separate});
  if(separate)Object.assign(s.household,{birthday:'1961-10-04',retirementDate:TODAY,asOfDate:TODAY});
  s.accounts={pretax:5000000,roth:0,taxable:0,cash:0};s.rothHistory.contributionBasis=0;s.socialSecurity.annualBenefitAt67=0;
  Object.assign(s.spending,{annualBaseSpending:150000,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0});
  Object.assign(s.healthcare,{healthcareInflationMean:0,healthcareInflationStdDev:0});s.longTermCare.enabled=false;
  s.market={preRetirementMeanReturn:0,preRetirementStdDev:0,stockMeanReturn:0,stockStdDev:0,bondMeanReturn:0,bondStdDev:0};
  s.partTimeIncome={annualNet:100000,endAge:69};
  const reduced=structuredClone(s);reduced.spending.annualBaseSpending=50000;reduced.partTimeIncome={annualNet:0,endAge:0};
  const options={fixedDeathAges:{primary:75,spouse:75},captureMonthlyDetails:true,captureTaxDetails:true};
  const run=x=>runOne(x,new JavaRandom(x.seed),options).monthlyDetails.filter(p=>p.cashFlow).slice(0,24);
  const worked=run(s),netCosts=run(reduced);assert.equal(worked.length,24);
  for(let i=0;i<24;i++)for(const key of ['expenses','additionalWithdrawal','incomeTax']){
    assert.ok(Math.abs(worked[i].cashFlow[key]-netCosts[i].cashFlow[key])<1e-6,`month ${i+1} ${key}: work pay must cover costs before taxable income is estimated`);
  }
  assert.notEqual(resultFingerprint(s,TODAY),resultFingerprint({...s,partTimeIncome:{annualNet:0,endAge:0}},TODAY));
});
