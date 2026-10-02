import test from 'node:test';
import assert from 'node:assert/strict';
import {baseScenario,prepareCalendarScenario,applyBudgetEstimate,addCalendarMonths} from '../dist/model.js';
import {budgetPlanGaps,spendingInputSummary} from '../dist/plan-review.js';

test('budget reconciliation uses applied months and explicit entered zeros',()=>{
  const s=baseScenario(),sources={[s.id]:{_origin:'Sample/default'}};
  s.budget.monthlyBudgets=[{month:'2026-09',creditCardBills:[{monthlyAmount:4000}],adjustments:{mortgage:1000,rent:100,healthcare:100}}];
  assert.deepEqual(budgetPlanGaps(s,sources),[],'A draft must not imply those costs have been removed from the plan');
  applyBudgetEstimate(s);assert.deepEqual(budgetPlanGaps(s,sources).map(x=>x.key),['mortgage','rent','healthcare']);
  sources[s.id]['mortgage.monthlyPayment']='Entered';sources[s.id]['rent.monthlyRent']='Estimated';
  assert.deepEqual(budgetPlanGaps(s,sources).map(x=>x.key),['healthcare']);
  // The same latest-12-month window as the applied estimate.
  s.budget.monthlyBudgets.push(...Array.from({length:12},(_,i)=>({month:`2027-${String(i+1).padStart(2,'0')}`,adjustments:{}})));
  assert.deepEqual(budgetPlanGaps(s,sources),[]);
});

test('monthly review combines base, active mortgage, rent and age-appropriate premiums without double counting home bills',()=>{
  const s=prepareCalendarScenario(baseScenario());s.household.separatePeople=true;s.household.filingStatus='Married';
  s.household.retirementDate=addCalendarMonths(s.household.birthday,64*12);
  s.household.spouseBirthday=addCalendarMonths(s.household.birthday,-2*12);
  s.household.spouseRetirementDate=s.household.retirementDate;
  s.spending.annualBaseSpending=40800;s.mortgage.monthlyPayment=1200;s.mortgage.yearsLeft=10;s.rent.monthlyRent=200;
  s.healthcare.preMedicareMonthlyPremium=500;s.home.annualTaxesAndInsurance=6000;
  const sources={[s.id]:{_origin:'Entered'}},d=spendingInputSummary(s,sources);
  assert.equal(d.base,3400);assert.equal(d.mortgage,1200);assert.equal(d.rent,200);assert.equal(d.healthcare,500);
  assert.equal(d.preMedicareAdults,1);assert.equal(d.medicareAdults,1);assert.equal(d.total,5300);
  s.mortgage.yearsLeft=1;assert.equal(spendingInputSummary(s,sources).mortgage,0,'A paid-off mortgage is absent at retirement');
  sources[s.id]['spending.annualBaseSpending']='Unknown';assert.equal(spendingInputSummary(s,sources).total,null);
});
