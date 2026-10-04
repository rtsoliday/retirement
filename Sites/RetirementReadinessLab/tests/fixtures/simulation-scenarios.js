import {baseScenario,employerRothDefaults} from '../../dist/model.js';

// Fixed dates and seeds make performance measurements and full-result
// regressions reproducible, independent of the day the checks run.
export function simulationScenarios(){
  const pooled=baseScenario();pooled.numberOfSimulations=32;
  const shortfall=structuredClone(pooled);shortfall.numberOfSimulations=10;
  shortfall.accounts={pretax:175000,roth:17500,taxable:0,cash:17500};shortfall.rothHistory.contributionBasis=17500;
  const early=structuredClone(pooled);early.numberOfSimulations=10;
  Object.assign(early.household,{currentAge:50,retirementAge:55,filingStatus:'Married',spouseCurrentAge:48});
  early.accounts={pretax:800000,roth:180000,taxable:75000,cash:50000};
  Object.assign(early.rothHistory,{contributionBasis:80000,firstContributionYear:2023,conversions:[{taxYear:2025,amount:40000,taxableAmount:30000}]});
  early.withdrawalStrategy.useCashReserveDuringDrawdowns=true;early.rothConversion.enabled=true;
  const sepp=structuredClone(early);sepp.withdrawalStrategy.seppEligible=true;sepp.numberOfSimulations=32;
  const married=structuredClone(pooled);
  Object.assign(married.household,{separatePeople:true,birthday:'1966-10-01',retirementDate:'2033-10-01',spouseBirthday:'1968-10-01',spouseRetirementDate:'2035-10-01',asOfDate:'2026-10-01',filingStatus:'Married'});
  married.spouseAccounts={pretax:300000,roth:50000};
  Object.assign(married.spouseRothHistory,{contributionBasis:30000,firstContributionYear:2021});
  Object.assign(married.contributions,{pretax:12000,employerPretax:6000,roth:6000,annualIncrease:.02});
  Object.assign(married.spouseContributions,{pretax:12000,employerPretax:3000,roth:6000,taxable:2400});
  Object.assign(married.spouseIncome,{annualBenefitAt67:24000,annualPension:12000,survivorPercent:.5});
  married.workingIncome.spouseAnnualNet=36000;married.rothConversion.enabled=true;
  const retired=structuredClone(married);retired.numberOfSimulations=10;
  Object.assign(retired.household,{alreadyRetired:true,retirementDate:'2024-10-01',spouseRetirementDate:'2028-10-01'});
  retired.accounts={pretax:100000,roth:50000,taxable:25000,cash:20000};retired.spouseAccounts.pretax=75000;
  retired.mortgage={monthlyPayment:2000,yearsLeft:5,monthsLeft:0,currentBalance:100000};
  retired.home={currentValue:300000,annualTaxesAndInsurance:6000};
  const employer=structuredClone(married);employer.numberOfSimulations=10;
  Object.assign(employer.household,{birthday:'1971-10-01',retirementDate:'2029-10-01',spouseBirthday:'1973-10-01',spouseRetirementDate:'2031-10-01'});
  employer.employerRothAccounts=[
    {...employerRothDefaults(),name:'Primary 401k',balance:150000,contributionBasis:100000,firstContributionYear:2021,annualContribution:12000,annualEmployerContribution:3000,accessDate:'2029-10-01',separationDate:'2029-10-01',ruleOf55Eligible:true,plannedConversionDate:'2029-11-01',plannedConversionAmount:20000,rolloverDate:'2032-10-01'},
    {...employerRothDefaults(),name:'Spouse 403b',type:'403b',owner:'spouse',balance:75000,contributionBasis:50000,firstContributionYear:2025,conversions:[{taxYear:2025,amount:10000,taxableAmount:8000}],accessDate:'2031-10-01',separationDate:'2031-10-01',ruleOf55Eligible:true}
  ];
  const legacyEmployer=structuredClone(employer);legacyEmployer.household.separatePeople=false;
  return [
    ['pooled',pooled],['shortfall-preview',shortfall],['early-conversions',early],['sepp',sepp],
    ['separate-couple',married],['already-retired',retired],['employer-roth',employer],['pooled-employer-roth',legacyEmployer]
  ];
}
