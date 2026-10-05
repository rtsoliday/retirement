import * as z from 'zod/v4';
import { baseScenario, employerRothDefaults, DEFAULT_SEED } from '../dist/model.js';
import { MODEL_DEFAULT_FIELDS, MODEL_DEFAULT_GROUPS } from './mcp-defaults.js';

export const SCHEMA_VERSION = '1.0';
const finite = z.number().finite().min(-Number.MAX_SAFE_INTEGER).max(Number.MAX_SAFE_INTEGER);
const computed = z.number().finite();
const conversion = z.strictObject({ taxYear: finite, amount: finite, taxableAmount: finite });
const bill = z.strictObject({ name: z.string().max(120).optional(), label: z.string().max(120).optional(), monthlyAmount: finite });
const month = z.object({
  month: z.string().max(7), cashAndAtmWithdrawals: finite.optional(),
  checkingSavingsBills: z.array(bill).max(100).optional(), creditCardBills: z.array(bill).max(100).optional(),
  includedPayments: z.strictObject({ mortgage: z.boolean(), rent: z.boolean(), healthcare: z.boolean() }).optional(),
  adjustments: z.record(z.string().max(80), finite).optional(),
}).strict();
const arrays = {
  conversions: z.array(conversion).max(200),
  monthlyBudgets: z.array(month).max(120),
  oneTimeExpenses: z.array(z.strictObject({ label: z.string().max(120), age: finite, amount: finite })).max(12),
};
// Personal fields stay required. Only allowlisted projections can be unknown;
// their website values are resolved centrally and disclosed in the result.
function shape(template, prefix = '') {
  const fields = {};
  for (const [key, value] of Object.entries(template)) {
    const path = prefix ? `${prefix}.${key}` : key;
    fields[key] = Array.isArray(value) ? arrays[key] : value !== null && typeof value === 'object' ? shape(value, path)
      : typeof value === 'number' ? finite : typeof value === 'boolean' ? z.boolean() : z.string().max(120);
    if (!fields[key]) throw new Error(`Missing array schema: ${key}`);
    const modelDefault = MODEL_DEFAULT_FIELDS.get(path);
    if (modelDefault || MODEL_DEFAULT_GROUPS.has(path)) {
      fields[key] = z.union([fields[key], z.literal('unknown')]).nullable().optional().describe(modelDefault
        ? `${modelDefault.label}. Omit, null or "unknown" uses the website illustrative default ${modelDefault.displayValue} (${modelDefault.units}). Review this value with the user first. ${modelDefault.note}`
        : 'Omit, null or "unknown" uses this group of website illustrative defaults. See explain_forecast_methodology and review the values with the user first.');
    }
  }
  return z.strictObject(fields);
}
arrays.employerRothAccounts = z.array(shape(employerRothDefaults())).max(50);
const template = baseScenario();
export const scenarioSchema = shape(template).extend({
  id: z.string().max(120).optional(), name: z.string().max(120).optional(),
  seed: z.number().int().safe().optional(), numberOfSimulations: z.number().int().min(1).max(10000).optional(),
  simulationPathsCustomized: z.boolean().optional(),
  household: shape(template.household).extend({ asOfDate: z.string().max(10).optional(), datesNeedReview: z.boolean().optional() }),
  budget: shape(template.budget).extend({ appliedAnnualHomeCosts: finite.optional() }),
}).describe('Supply explicit personal inputs, including zero values and disabled features. Allowlisted market, inflation, healthcare, long-term-care and allocation projections may be omitted, null or "unknown" to use reviewed website illustrative defaults. Never substitute a guessed rate for an unknown projection. See explain_forecast_methodology for exact defaults.');
const common = {
  schemaVersion: z.literal(SCHEMA_VERSION),
  pathCount: z.number().int().min(1).max(1000).optional().describe('Defaults to 100 Free or 1000 Pro/owner; requests above your entitlement are rejected.'),
  forecastDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional().describe('YYYY-MM-DD; defaults to today UTC. Both comparison scenarios use this date.'),
  processingAcknowledged: z.literal(true).describe('Set only after the user reviews assumptions and acknowledges temporary processing by this hosted service and return of results to their assistant.'),
};
export const forecastInput = z.strictObject({ ...common, scenario: scenarioSchema });
export const compareInput = z.strictObject({ ...common, scenarios: z.tuple([scenarioSchema, scenarioSchema]) });
export const methodologyInput = z.strictObject({ topic: z.enum(['overview', 'inputs', 'defaults', 'privacy', 'limits', 'taxes', 'mortality', 'comparisons']).optional() });
export const modelDefaultOutput = z.strictObject({ path: z.string(), label: z.string(), value: z.union([finite, z.boolean(), z.string()]), displayValue: z.string(), units: z.string(), note: z.string() });
const band = z.strictObject({ age: finite, pessimistic: computed, median: computed, optimistic: computed, pathCount: z.number().int() });
const summary = z.strictObject({
  label: z.string(), successProbability: z.number().min(0).max(1), shortfallFrequency: z.number().min(0).max(1),
  medianEndingBalance: computed, p10EndingBalance: computed, p90EndingBalance: computed, medianFailureAge: finite.nullable(),
  failureAgeBuckets: z.array(z.strictObject({ label: z.string(), count: z.number().int(), shareOfFailures: finite })),
  annualBalanceRanges: z.array(band), fundedShareByAge: z.array(z.strictObject({ age: finite, notFailedShare: finite, aliveShare: finite })),
  provenance: z.strictObject({ engineVersion: z.string(), engineCadence: z.string(), taxTableVersion: z.string(), mortalityModelVersion: z.string(), randomSeed: z.number().int(), simulationCount: z.number().int() }),
  defaultsApplied: z.array(modelDefaultOutput).optional().describe('Website illustrative assumptions filled in for omitted, null or unknown projection fields. Disclose these to the user.'),
});
export const forecastOutput = z.strictObject({
  schemaVersion: z.literal(SCHEMA_VERSION), forecastDate: z.string(), dollarBasis: z.literal('forecast-date purchasing power'),
  tier: z.enum(['free', 'pro']), pathCount: z.number().int(), forecasts: z.array(summary).min(1).max(2),
  differences: z.strictObject({ shortfallFrequency: finite, medianEndingBalance: computed, p10EndingBalance: computed, p90EndingBalance: computed, medianFailureAge: finite.nullable() }).optional(),
  warnings: z.array(z.string()), links: z.strictObject({ planner: z.string(), methodology: z.string(), privacy: z.string() }),
});
export const MCP_SEED = DEFAULT_SEED;
