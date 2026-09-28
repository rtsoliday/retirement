import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,budgetEstimate,budgetBreakdown,budgetMonthTotals,validateBudget,applyBudgetEstimate,markBudgetEdited,normalizeScenario} from '../dist/model.js';
import {runSimulation} from '../dist/engine.js';
const month=(date,spending=4500,adjustments={})=>({month:date,checkingSavingsBills:[{monthlyAmount:spending}],creditCardBills:[],cashAndAtmWithdrawals:0,adjustments});
test('quarter containing a $6000 tax payment adds tax exactly once per year',()=>{
  const b=baseScenario().budget;b.annualPropertyTaxes=6000;b.annualHomeInsurance=2000;b.annualAutoInsurance=1000;
  b.monthlyBudgets=[month('2026-01',10500,{propertyTaxes:6000}),month('2026-02'),month('2026-03')];
  const d=budgetBreakdown(b);assert.equal(d.grossAverage,6500);assert.equal(d.monthlyAverage,4500);assert.equal(d.annualized,54000);assert.equal(budgetEstimate(b),63000);assert.deepEqual(validateBudget(b),[]);
});
test('three sources are summed and separate modeled costs removed before annualizing',()=>{
  const b=baseScenario().budget;b.monthlyBudgets=[{...month('2026-01',2000,{mortgage:1000,rent:200,healthcare:300}),creditCardBills:[{monthlyAmount:1500},{monthlyAmount:500}],cashAndAtmWithdrawals:500}];
  assert.equal(budgetMonthTotals(b.monthlyBudgets[0]).gross,4500);assert.equal(budgetEstimate(b),36000);
  b.retirementAnnualAdjustment=-6000;assert.equal(budgetEstimate(b),30000);
});
test('a full year with annual bills already counted leaves total unchanged',()=>{
  const b=baseScenario().budget;b.annualPropertyTaxes=6000;
  b.monthlyBudgets=Array.from({length:12},(_,i)=>month(`2026-${String(i+1).padStart(2,'0')}`,i===0?10500:4500,i===0?{propertyTaxes:6000}:{}));
  assert.equal(budgetEstimate(b),60000);
});
test('old sample months and their deductions are both excluded from latest twelve',()=>{
  const b=baseScenario().budget;b.monthlyBudgets=[month('2025-12',100000,{propertyTaxes:90000}),...Array.from({length:12},(_,i)=>month(`2026-${String(i+1).padStart(2,'0')}`,1000))];
  const d=budgetBreakdown(b);assert.equal(d.count,12);assert.equal(d.totals.annualBills,0);assert.equal(d.estimate,12000);
});
test('invalid worksheets cannot be applied: missing months, duplicates, over-deductions and negative estimate',()=>{
  const s=baseScenario();assert.throws(()=>applyBudgetEstimate(s),/complete month/);
  s.budget.monthlyBudgets=[month('2026-01'),month('2026-01')];assert.throws(()=>applyBudgetEstimate(s),/twice/);
  s.budget.monthlyBudgets=[month('',100)];assert.match(validateBudget(s.budget).join(' '),/valid month/);
  s.budget.monthlyBudgets=[month('2026-01',100,{mortgage:101})];assert.throws(()=>applyBudgetEstimate(s),/exceed spending/);
  s.budget.monthlyBudgets=[month('2026-01',100)];s.budget.retirementAnnualAdjustment=-1201;assert.throws(()=>applyBudgetEstimate(s),/negative/);
  s.budget.retirementAnnualAdjustment=Infinity;assert.match(validateBudget(s.budget).join(' '),/finite/);
  assert.equal(s.spending.annualBaseSpending,75000);
});
test('refunds can reduce credit purchases including a negative month',()=>{
  const b=baseScenario().budget;b.monthlyBudgets=[{...month('2026-01',0),creditCardBills:[{monthlyAmount:-100}]},month('2026-02',2100)];
  assert.deepEqual(validateBudget(b),[]);assert.equal(budgetEstimate(b),12000);
});
test('budget edits leave the applied plan and home-sale calculation unchanged until reapplied',()=>{
  const s=baseScenario();s.numberOfSimulations=1;s.accounts.pretax=10000;s.accounts.roth=0;s.accounts.cash=0;s.home.currentValue=500000;
  s.budget.monthlyBudgets=[month('2026-01')];s.budget.annualPropertyTaxes=6000;applyBudgetEstimate(s);
  const before=runSimulation(s);markBudgetEdited(s.budget);s.budget.annualPropertyTaxes=15000;s.budget.monthlyBudgets[0].checkingSavingsBills[0].monthlyAmount=6000;
  assert.equal(s.spending.annualBaseSpending,60000);assert.equal(s.budget.appliedAnnualHomeCosts,6000);assert.equal(s.budget.estimateNeedsReview,true);
  const after=runSimulation(s);delete before.generatedAtEpochMillis;delete after.generatedAtEpochMillis;assert.deepEqual(after,before);applyBudgetEstimate(s);assert.equal(s.spending.annualBaseSpending,87000);assert.equal(s.budget.appliedAnnualHomeCosts,15000);assert.equal(s.budget.estimateNeedsReview,false);
});
test('legacy budgets keep estimates and new adjustments survive JSON backup',()=>{
  const s=normalizeScenario({budget:{annualPropertyTaxes:4000,monthlyBudgets:[month('2026-01',1000)]}});assert.equal(budgetEstimate(s.budget),16000);
  s.budget.monthlyBudgets[0].adjustments={propertyTaxes:500,healthcare:100};s.budget.retirementAnnualAdjustment=1000;applyBudgetEstimate(s);
  const copy=normalizeScenario(JSON.parse(JSON.stringify(s)));assert.deepEqual(copy,s);assert.equal(budgetEstimate(copy.budget),9800);
});
