import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,employerRothDefaults,normalizeScenario,validateScenario,validateScenarioDraft,scenarioEngineVersion} from '../dist/model.js';
import {EmployerRothLedger} from '../dist/employer-roth.js';
import {RothConversionLedger} from '../dist/roth-conversions.js';
import {PersonAccounts} from '../dist/person-accounts.js';
import {runOne,runSimulation,runSteadySimulation} from '../dist/engine.js';
import {ordinaryIncomeTax} from '../dist/tax.js';
import {resultFingerprint} from '../dist/result-cache.js';
import {withdrawalContent} from '../dist/withdrawals-view.js';
import {monthlyIncomeSummary} from '../dist/plan-review.js';

const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<.002,`${actual} != ${expected}`);
const account=values=>({...employerRothDefaults(),balance:100000,contributionBasis:60000,firstContributionYear:2021,...values});
function plan(age=55,married=false){
  const s=baseScenario();Object.assign(s.household,{separatePeople:true,birthday:`${2026-age}-10-01`,retirementDate:'2026-10-01',asOfDate:'2026-10-01',filingStatus:married?'Married':'Single',spouseBirthday:`${2026-age}-10-01`,spouseRetirementDate:'2026-10-01'});
  s.accounts={pretax:0,roth:0,taxable:0,cash:0};s.rothHistory={contributionBasis:0,firstContributionYear:0,conversions:[],needsReview:false};
  Object.assign(s.spending,{annualBaseSpending:0,spendingPathModel:'Flat',generalInflationMean:0,generalInflationStdDev:0,lowPortfolioSpendingReduction:0});
  Object.assign(s.healthcare,{includeMedicarePremiums:false,preMedicareMonthlyPremium:0,healthcareInflationMean:0,healthcareInflationStdDev:0});s.socialSecurity.annualBenefitAt67=0;s.longTermCare.enabled=false;for(const k of Object.keys(s.market))s.market[k]=0;
  s.employerRothAccounts=[account()];return s;
}
const rng=()=>({normal:mean=>mean,nextDouble:()=>.5});
const run=(s,months=12)=>runOne(s,rng(),{captureMonthlyDetails:true,captureTaxDetails:true,fixedDeathAges:{primary:2026-Number(s.household.birthday.slice(0,4))+months/12,spouse:2026-Number(s.household.spouseBirthday.slice(0,4))+months/12}});
const pools=s=>new PersonAccounts(s,{...s.accounts});
function configure(p,month=0,alive=[true,false],deaths=[100,100]){p.configure(month,2026+Math.floor(month/12),deaths,alive,()=>0);return p;}

test('nonqualified employer Roth withdrawals recover proportional basis, including after losses',()=>{
  const l=new EmployerRothLedger(account()),q=l.distribution(10000,100000,2026,{consume:true});near(q.basis,6000);near(q.taxableEarnings,4000);near(q.penaltyBase,4000);near(l.basis,54000);
  const loss=new EmployerRothLedger(account({contributionBasis:120000}));const d=loss.distribution(10000,100000,2026,{consume:true});near(d.taxableEarnings,0);near(loss.basis,110000);
});
test('qualified distributions still reduce basis proportionally and exact qualification uses age and tax year',()=>{
  const l=new EmployerRothLedger(account());assert.equal(l.isQualified(2025,60),false);assert.equal(l.isQualified(2026,59.49),false);assert.equal(l.isQualified(2026,59.5),true);
  const d=l.distribution(10000,100000,2026,{qualified:true,consume:true});near(d.taxableEarnings,0);near(d.penaltyBase,0);near(l.basis,54000);
  assert.equal(l.isQualified(2026,50,true),true);assert.equal(l.isQualified(2025,50,false,true),false);assert.equal(l.isQualified(2026,50,false,true),true);
});
test('in-plan recapture allocates recovered basis after regular basis, then FIFO and taxable-first',()=>{
  const l=new EmployerRothLedger(account({contributionBasis:80000,conversions:[{taxYear:2025,amount:30000,taxableAmount:20000}]}));
  const first=l.distribution(50000,100000,2026,{consume:true});near(first.basis,40000);near(first.penaltyBase,10000);
  const second=l.distribution(25000,50000,2026,{consume:true});near(second.basis,20000);near(second.penaltyBase,15000);near(l.principal.lots[0].taxableAmount,10000);
  const separate=new EmployerRothLedger(account({conversionOnly:true,conversions:[{taxYear:2025,amount:30000,taxableAmount:20000}]}));near(separate.distribution(10000,100000,2026).penaltyBase,10000);
});
test('a qualified rollover adds the whole balance to IRA principal; a nonqualified rollover carries only basis',()=>{
  for(const qualified of [true,false]){const l=new EmployerRothLedger(account()),ira=new RothConversionLedger(5000,2020);l.rollover(100000,2026,qualified,ira);near(ira.openingBalance,qualified?105000:65000);near(l.basis,0);}
});
test('rollovers preserve recent in-plan recapture without taxing conversion principal again',()=>{
  const l=new EmployerRothLedger(account({contributionBasis:80000,conversions:[{taxYear:2025,amount:30000,taxableAmount:20000}]})),ira=new RothConversionLedger();l.rollover(100000,2026,false,ira);ira.firstContributionYear=2026;
  const q=ira.distribution(60000,100000,2026);near(q.taxableEarnings,0);near(q.penaltyBase,10000);assert.equal(ira.lots[0].taxYear,2025);
});
test('a full IRA rollover preserves basis above a loss-reduced balance and all conversion history',()=>{
  for(const qualified of [true,false]){
    const l=new EmployerRothLedger(account({balance:50000,contributionBasis:80000,conversions:[{taxYear:2025,amount:30000,taxableAmount:20000}]})),ira=new RothConversionLedger();
    l.rollover(50000,2026,qualified,ira);near(ira.openingBalance,50000);near(ira.lots[0].amount,30000);near(ira.lots[0].taxableAmount,20000);
    near(ira.distribution(50000,50000,2026).penaltyBase,0);
  }
});
test('calendar-year qualification and conversion recapture change at January rather than the forecast anniversary',()=>{
  const s=plan(60);s.household.retirementDate='2026-12-01';s.household.birthday='1966-12-01';s.employerRothAccounts=[account({firstContributionYear:2022})];const p=configure(pools(s));
  assert.equal(p.employer[0].qualified,false);configure(p,1);assert.equal(p.employer[0].qualified,true);
  const l=new EmployerRothLedger(account({contributionBasis:100000,conversions:[{taxYear:2022,amount:100000,taxableAmount:100000}]}));
  near(l.distribution(10000,100000,2026).penaltyBase,10000);near(l.distribution(10000,100000,2027).penaltyBase,0);
});
test('disability waives penalties but an unfunded five-year clock leaves earnings taxable; death stops deposits',()=>{
  const s=plan(55);s.employerRothAccounts=[account({disabled:true,firstContributionYear:2026,annualContribution:12000})];const p=configure(pools(s));
  const q=p.quote(10000);near(q.rothTaxableEarnings,4000);near(q.penalties,0);
  configure(p,0,[false]);near(p.deposits(0),0);near(p.quote(10000).penalties,0);
});
test('declared disability follows the owner across plans and IRA/pre-tax distributions after rollover',()=>{
  const s=plan(55);s.accounts.pretax=10000;s.accounts.roth=10000;s.rothHistory={contributionBasis:0,firstContributionYear:2021,conversions:[],needsReview:false};
  s.employerRothAccounts=[account({disabled:true,rolloverDate:'2026-10-01'}),account({name:'Other plan',firstContributionYear:2021,rolloverDate:'2026-10-01'})];const p=configure(pools(s));
  assert.equal(p.people[0].qualified,true);assert.equal(p.employer[1].qualified,true);near(p.quote(20000).penalties,0);p.events(0);near(p.people[0].ledger.openingBalance,200000);configure(p,1);assert.equal(p.people[0].qualified,true);near(p.quote(20000).penalties,0);
});
test('planned conversions reserve unmet RMDs and validation rejects protected or unavailable transfers',()=>{
  const s=plan(80);s.accounts.pretax=100000;Object.assign(s.employerRothAccounts[0],{plannedConversionAmount:100000,plannedConversionDate:'2026-10-01'});const p=configure(pools(s));
  assert.ok(p.people[0].rmd>0);near(p.events(0).conversion,100000-p.people[0].rmd);near(p.people[0].pretax,p.people[0].rmd);
  const early=plan(55);early.withdrawalStrategy.seppEligible=true;Object.assign(early.employerRothAccounts[0],{plannedConversionAmount:1000,plannedConversionDate:'2026-11-01'});
  assert.ok(validateScenario(early).some(e=>/SEPP-protected/.test(e)));
  const future=plan(60);future.household.retirementDate='2027-10-01';future.employerRothAccounts[0].rolloverDate='2026-11-01';
  assert.ok(validateScenario(future).some(e=>/permitted access date/.test(e)));future.employerRothAccounts[0].accessDate='2026-10-01';assert.deepEqual(validateScenario(future),[]);
});
test('Rule of 55 exempts only an eligible employer plan and does not erase earnings income',()=>{
  const s=plan();s.employerRothAccounts=[account({ruleOf55Eligible:true,separationDate:'2026-10-01'}),account({name:'Old employer',separationDate:'2020-10-01'})];const p=configure(pools(s));
  const q=p.quote(110000);near(q.rothTaxableEarnings,44000);near(q.penalties,400);p.consume(q);near(p.b.employerRoth,90000);near(p.b.roth,90000);
  s.employerRothAccounts[0].separationDate='2025-10-01';near(configure(pools(s)).quote(10000).penalties,400);
  s.accounts.roth=10000;s.rothHistory={contributionBasis:0,firstContributionYear:2021,conversions:[],needsReview:false};near(configure(pools(s)).quote(10000).penalties,1000);
});
test('access dates leave invested funds unavailable until permitted, with no owner RMD',()=>{
  const s=plan(80);s.employerRothAccounts[0].accessDate='2027-10-01';const p=configure(pools(s));near(p.quote(10000).shared.cash,10000);near(p.scheduled().rmd,0);near(configure(p,12).quote(10000).employerDraws[0],10000);
});
test('multiple owners retain independent histories and deposits stop at their own retirement',()=>{
  const s=plan(60,true);s.household.spouseRetirementDate='2027-10-01';s.employerRothAccounts=[account({owner:'spouse',annualContribution:12000,annualEmployerContribution:6000})];const p=configure(pools(s),0,[true,true]);near(p.deposits(0),1500);near(p.b.roth,101500);near(p.employer[0].ledger.basis,61500);configure(p,12,[true,true]);near(p.deposits(12),0);
});
test('a first employer Roth deposit starts its own clock and grows only after the deposit month',()=>{
  const s=plan(60);s.household.retirementDate='2027-10-01';s.employerRothAccounts=[account({balance:0,contributionBasis:0,firstContributionYear:0,annualContribution:12000})];const p=run(s,13).monthlyDetails[0];near(p.employerRoth,12000);near(p.ownerAccounts[0].employerRoth[0].contributionBasis,12000);assert.equal(p.ownerAccounts[0].employerRoth[0].firstContributionYear,2026);
});
test('rollover into a new IRA starts a new IRA clock; an existing earlier IRA clock is retained',()=>{
  for(const first of [0,2020]){const s=plan(60);s.rothHistory.firstContributionYear=first;s.employerRothAccounts[0].rolloverDate='2026-10-01';const p=configure(pools(s));const before=p.b.roth;p.events(0);near(p.b.roth,before);near(p.b.employerRoth,0);near(p.b.rothIRA,100000);assert.equal(p.people[0].ledger.firstContributionYear,first||2026);near(p.people[0].ledger.openingBalance,100000);}
});
test('pre-retirement full rollovers transfer once and are included in starting savings',()=>{
  const s=plan(60);s.household.retirementDate='2027-10-01';s.employerRothAccounts[0].accessDate='2026-10-01';s.employerRothAccounts[0].rolloverDate='2026-11-01';const row=run(s,13).monthlyDetails[0];near(row.rothIRA,100000);near(row.employerRoth,0);near(row.portfolio,100000);assert.equal(row.ownerAccounts[0].firstContributionYear,2026);
});
test('spouse inheritance preserves employer plans and disables original employer Rule of 55 eligibility',()=>{
  const s=plan(55,true);s.household.spouseBirthday='1976-10-01';s.employerRothAccounts[0].ruleOf55Eligible=true;s.employerRothAccounts[0].separationDate='2026-10-01';const p=configure(pools(s),0,[false,true],[0,100]);p.configure(12,2027,[0,100],[false,true],()=>0);assert.equal(p.employer[0].person,p.people[1]);assert.equal(p.employer[0].ruleOf55Eligible,false);near(p.quote(10000).penalties,400);
});
test('owner death cancels future transfers and inherited plans do not restart the deceased employee deposits',()=>{
  const s=plan(55,true);s.household.spouseRetirementDate='2028-10-01';s.accounts.pretax=10000;
  Object.assign(s.employerRothAccounts[0],{annualContribution:12000,annualEmployerContribution:6000,plannedConversionAmount:5000,plannedConversionDate:'2026-10-01',rolloverDate:'2026-11-01'});
  const p=configure(pools(s),0,[false,true],[0,100]);near(p.events(0).conversion,0);near(p.events(1).rollover,0);
  p.configure(12,2027,[0,100],[false,true],()=>0);near(p.deposits(12),0);near(p.b.employerRoth,100000);assert.equal(p.employer[0].plannedConversionAmount,0);
});
test('planned in-plan conversions transfer principal once and fund income tax from the portfolio',()=>{
  const s=plan(60);s.accounts.pretax=50000;s.accounts.cash=10000;Object.assign(s.employerRothAccounts[0],{balance:0,contributionBasis:0,firstContributionYear:0,plannedConversionAmount:30000,plannedConversionDate:'2026-10-01'});const r=run(s);near(r.taxYears[0].conversions,30000);near(r.taxYears[0].ordinaryIncome,30000+r.taxYears[0].accounts.cash-10000+r.taxYears[0].tax);assert.ok(r.taxYears[0].tax>0);near(r.monthlyDetails[1].cashFlow.inPlanConversion,30000);near(r.monthlyDetails[1].pretax,20000);near(r.monthlyDetails.at(-1).pretax,20000-r.taxYears[0].pretaxDistributions);near(r.monthlyDetails.at(-1).employerRoth,30000);
});
test('monthly explanations separate IRA and employer balances and include in-plan tax funding',()=>{
  const s=plan(60);s.accounts.pretax=50000;s.accounts.cash=10000;Object.assign(s.employerRothAccounts[0],{plannedConversionAmount:30000,plannedConversionDate:'2026-10-01'});
  const steady=run(s),r={steadySimulation:steady};const first=steady.monthlyDetails[1].cashFlow,d=monthlyIncomeSummary(s,r,'future');
  near(d.withdrawals,first.additionalWithdrawal+first.inPlanConversionTax);near(d.taxes,first.incomeTax+first.earlyPenalty+first.inPlanConversionTax);
  const html=withdrawalContent(s,r);assert.match(html,/Roth IRA → available employer Roth/);assert.match(html,/after the entered access date|entered access date/);assert.match(html,/moves from the owner’s pre-tax savings/);assert.match(html,/Roth IRA<\/dt>/);assert.match(html,/Employer Roth<\/dt>/);
  s.employerRothAccounts[0].plannedConversionAmount=0;s.employerRothAccounts[0].rolloverDate='2026-10-01';assert.match(withdrawalContent(s,{steadySimulation:run(s)}),/A full direct rollover transfers/);
});
test('employer earnings taxes and penalties are grossed up to cover spending and flow into annual income',()=>{
  const s=plan();s.spending.annualBaseSpending=30000;const r=run(s);const y=r.taxYears[0],gross=100000-y.accounts.employerRoth;near(y.ordinaryIncome,.4*gross);near(y.tax,ordinaryIncomeTax(y.ordinaryIncome,'Single',1,0,2026));near(gross-y.tax-.04*gross,30000);near(y.accounts.roth,y.accounts.employerRoth);
});
test('nonqualified earnings affect Social Security taxation and Medicare estimates/lookback',()=>{
  const s=plan(65);s.employerRothAccounts=[account({balance:1000000,contributionBasis:0,firstContributionYear:2026})];s.spending.annualBaseSpending=150000;s.socialSecurity.annualBenefitAt67=40000;s.socialSecurity.claimAge=65;s.healthcare.includeMedicarePremiums=true;
  const r=run(s,36);assert.ok(r.taxYears[0].ordinaryIncome>100000);assert.ok(r.taxYears[0].tax>ordinaryIncomeTax(r.taxYears[0].ordinaryIncome,'Single',1,1,2026));assert.ok(r.monthlyDetails[1].cashFlow.expenses>150000/12+241.89);assert.ok(r.monthlyDetails[25].cashFlow.expenses>150000/12+241.89);
});
test('validation rejects malformed imports, incomplete history and conflicting transactions',()=>{
  for(const edit of [s=>s.employerRothAccounts=null,s=>s.employerRothAccounts=[null],s=>s.employerRothAccounts[0].conversions=[null],s=>s.employerRothAccounts[0].balance='100',s=>s.employerRothAccounts[0].accessDate='2026-02-30']){const s=plan();edit(s);assert.ok(validateScenarioDraft(s).length);}
  for(const edit of [s=>s.employerRothAccounts[0].firstContributionYear=0,s=>s.employerRothAccounts[0].ruleOf55Eligible=true,s=>Object.assign(s.employerRothAccounts[0],{rolloverDate:'2026-10-01',accessDate:'2027-10-01'}),s=>Object.assign(s.employerRothAccounts[0],{plannedConversionAmount:1,plannedConversionDate:'2026-09-01'}),s=>s.employerRothAccounts[0].conversions=[{taxYear:2025,amount:70000,taxableAmount:70000}]]){const s=plan();edit(s);assert.ok(validateScenario(s).length);}
  const old=baseScenario();delete old.employerRothAccounts;assert.deepEqual(normalizeScenario(old).employerRothAccounts,[]);const s=plan();assert.deepEqual(normalizeScenario(JSON.parse(JSON.stringify(s))),s);
});
test('pooled and separate plans include employer balances in seeded and steady results; saved caches distinguish edits',()=>{
  for(const separatePeople of [false,true]){const s=plan(65);s.household.separatePeople=separatePeople;s.household.targetEndAge=66;s.numberOfSimulations=10;assert.equal(runSimulation(s).provenance.engineVersion,scenarioEngineVersion(s));near(runSteadySimulation(s).monthlyDetails[0].portfolio,100000);assert.match(scenarioEngineVersion(s),/employer-roth/);const f=resultFingerprint(s,'2026-10-01');s.employerRothAccounts[0].contributionBasis=1;assert.notEqual(resultFingerprint(s,'2026-10-01'),f);}
});
