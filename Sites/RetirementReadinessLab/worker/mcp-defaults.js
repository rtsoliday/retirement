import { baseScenario } from '../dist/model.js';

// Only these projections may use website assumptions. Never merge an entire
// sample scenario: balances, income, spending and account histories are personal.
const fields = [
  ['market.preRetirementMeanReturn', 'Pre-retirement return average', 'annual rate', 'Illustrative growth assumption before retirement.'],
  ['market.preRetirementStdDev', 'Pre-retirement return volatility', 'annual rate', 'Annual standard deviation.'],
  ['market.stockMeanReturn', 'Stock return average', 'annual rate', 'Illustrative stock return, not a prediction.'],
  ['market.stockStdDev', 'Stock return volatility', 'annual rate', 'Annual standard deviation.'],
  ['market.bondMeanReturn', 'Bond return average', 'annual rate', 'Illustrative bond return, not a prediction.'],
  ['market.bondStdDev', 'Bond return volatility', 'annual rate', 'Annual standard deviation.'],
  ['spending.generalInflationMean', 'General inflation average', 'annual rate', 'Applied to general costs.'],
  ['spending.generalInflationStdDev', 'General inflation volatility', 'annual rate', 'Annual standard deviation.'],
  ['healthcare.preMedicareMonthlyPremium', 'Pre-Medicare health premium', 'USD per adult per month', 'Illustrative premium for each retired adult under 65; replace with an insurance estimate when available.'],
  ['healthcare.healthcareInflationMean', 'Healthcare inflation average', 'annual rate', 'Applied to health premiums and long-term-care costs.'],
  ['healthcare.healthcareInflationStdDev', 'Healthcare inflation volatility', 'annual rate', 'Annual standard deviation.'],
  ['healthcare.includeMedicarePremiums', 'Include Medicare premiums', 'boolean', 'Premiums are calculated by the existing Medicare model.'],
  ['longTermCare.enabled', 'Model long-term-care risk', 'boolean', 'Models uncertain care episodes; does not assume everyone needs care.'],
  ['longTermCare.annualCost', 'Long-term-care cost', 'USD per person per year', 'Illustrative cost during a modeled care episode, before future healthcare inflation.'],
  ['longTermCare.averageDurationYears', 'Long-term-care average duration', 'years', 'Average modeled episode duration; not a personal care estimate.'],
  ['longTermCare.averageDurationMonths', 'Additional long-term-care duration', 'months', 'Added to the average duration in years.'],
  ['postRetirementAllocation.stockUnder30x', 'Stocks below 30 times annual spending', 'portfolio share', 'Remainder is bonds; the band uses invested assets divided by annual spending.'],
  ['postRetirementAllocation.stock30xTo35x', 'Stocks at 30–35 times annual spending', 'portfolio share', 'Remainder is bonds.'],
  ['postRetirementAllocation.stock35xTo40x', 'Stocks at 35–40 times annual spending', 'portfolio share', 'Remainder is bonds.'],
  ['postRetirementAllocation.stock40xTo45x', 'Stocks at 40–45 times annual spending', 'portfolio share', 'Remainder is bonds.'],
  ['postRetirementAllocation.stock45xTo50x', 'Stocks at 45–50 times annual spending', 'portfolio share', 'Remainder is bonds.'],
  ['postRetirementAllocation.stock50xOrMore', 'Stocks at 50 or more times annual spending', 'portfolio share', 'Remainder is bonds.'],
];
export const MODEL_DEFAULT_GROUPS = new Set(['market', 'healthcare', 'longTermCare', 'postRetirementAllocation']);
export function websiteModelDefaults() {
  const template = baseScenario();
  return fields.map(([path, label, units, note]) => {
    const [group, key] = path.split('.'), value = template[group][key];
    const displayValue = units === 'annual rate' || units === 'portfolio share' ? `${Number((value * 100).toFixed(2))}%`
      : units.startsWith('USD') ? `$${value.toLocaleString('en-US')}` : String(value);
    return { path, label, value, displayValue, units, note };
  });
}
export const MODEL_DEFAULT_FIELDS = new Map(websiteModelDefaults().map(field => [field.path, field]));
const unknown = value => value === undefined || value === null || value === 'unknown';
export function resolveModelDefaults(input) {
  const scenario = structuredClone(input), defaultsApplied = [];
  for (const group of MODEL_DEFAULT_GROUPS) if (unknown(scenario[group])) scenario[group] = {};
  for (const field of websiteModelDefaults()) {
    const [group, key] = field.path.split('.');
    if (scenario[group] && unknown(scenario[group][key])) {
      scenario[group][key] = field.value;
      defaultsApplied.push(field);
    }
  }
  return { scenario, defaultsApplied };
}
