import test from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import { Client, StreamableHTTPClientTransport } from '@modelcontextprotocol/client';
import { CfWorkerJsonSchemaValidator } from '@modelcontextprotocol/server/validators/cf-worker';
import worker from '../worker/index.js';
import { calculateForecast, summarizeForecast } from '../worker/mcp.js';
import { forecastInput, compareInput, forecastOutput } from '../worker/mcp-schema.js';
import { acquireUsage, finishUsage, accountKey } from '../worker/mcp-storage.js';
import { forecastEntitlement } from '../worker/billing.js';
import { baseScenario, employerRothDefaults } from '../dist/model.js';
import { resolveModelDefaults, websiteModelDefaults, MODEL_DEFAULT_GROUPS } from '../worker/mcp-defaults.js';
import { runSimulation, runSimulationAsync } from '../dist/engine.js';

const origin = 'https://retirementforecast.us', owner = '028696a7-7846-4822-a4c1-67026aa2383f';
function database() {
  const db = new DatabaseSync(':memory:');
  for (const name of ['0000_opposite_morlocks', '0001_chilly_ultron', '0002_moaning_rhodey']) db.exec(readFileSync(new URL(`../drizzle/${name}.sql`, import.meta.url), 'utf8'));
  return { db, prepare(sql) { const statement = db.prepare(sql); let values = [];
    return { bind(...args) { values = args; return this; }, async all() { return { success: true, results: statement.all(...values) }; }, async first() { return statement.get(...values); }, run() { return { success: true, meta: statement.run(...values) }; } }; },
    async batch(statements) { db.exec('BEGIN'); try { const results = statements.map(s => s.run()); db.exec('COMMIT'); return results; } catch (e) { db.exec('ROLLBACK'); throw e; } } };
}
function req(body, user = owner, extra = {}) { return new Request(origin + '/mcp', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream', ...(user ? { 'oai-authenticated-user-id': user } : {}), ...extra }, body: JSON.stringify(body) }); }
const call = (name, args) => ({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name, arguments: args } });
const args = scenario => ({ schemaVersion: '1.0', processingAcknowledged: true, pathCount: 4, forecastDate: '2026-10-05', scenario });
function scenario() { const s = baseScenario(); delete s.numberOfSimulations; delete s.seed; s.household.asOfDate = ''; s.household.birthday = '1966-10-05'; s.household.retirementDate = '2033-10-05'; return s; }
async function body(response) { const text = await response.text(); return response.headers.get('content-type')?.includes('event-stream') ? JSON.parse(text.split('\n').find(line => line.startsWith('data: ')).slice(6)) : JSON.parse(text); }

test('strict schemas require personal amounts and consent, accept explicit unused values, reject impersonation', () => {
  const a = args(scenario()); assert.equal(forecastInput.safeParse(a).success, true);
  const missing = structuredClone(a); delete missing.scenario.accounts.pretax;
  assert.equal(forecastInput.safeParse(missing).success, false);
  assert.equal(forecastInput.safeParse({ ...a, processingAcknowledged: false }).success, false);
  assert.equal(forecastInput.safeParse({ ...a, accountId: 'other', tier: 'pro' }).success, false);
  assert.equal(compareInput.safeParse({ schemaVersion: '1.0', processingAcknowledged: true, scenarios: [scenario(), scenario(), scenario()] }).success, false);
  a.scenario.accounts.pretax = Infinity; assert.equal(forecastInput.safeParse(a).success, false);
});

test('only allowlisted projections accept omitted, null or unknown values; personal facts stay required', async () => {
  for (const { path } of websiteModelDefaults()) {
    const [group, key] = path.split('.');
    for (const marker of [undefined, null, 'unknown']) {
      const input = args(scenario());
      if (marker === undefined) delete input.scenario[group][key]; else input.scenario[group][key] = marker;
      assert.equal(forecastInput.safeParse(input).success, true, `${path}: ${marker}`);
    }
    const bad = args(scenario()); bad.scenario[group][key] = 'guess';
    assert.equal(forecastInput.safeParse(bad).success, false, path);
  }
  for (const group of MODEL_DEFAULT_GROUPS) {
    const input = args(scenario()); delete input.scenario[group];
    assert.equal(forecastInput.safeParse(input).success, true, group);
  }
  for (const path of ['accounts.pretax', 'spending.annualBaseSpending', 'socialSecurity.annualBenefitAt67', 'rothHistory.contributionBasis', 'mortgage.monthlyPayment']) {
    const [group, key] = path.split('.');
    for (const marker of [undefined, null, 'unknown']) {
      const input = args(scenario());
      if (marker === undefined) delete input.scenario[group][key]; else input.scenario[group][key] = marker;
      assert.equal(forecastInput.safeParse(input).success, false, `${path}: ${marker}`);
    }
  }
  const missingDate = args(scenario()); delete missingDate.scenario.household.birthday;
  assert.equal(forecastInput.safeParse(missingDate).success, false);
  missingDate.scenario.household.birthday = 'unknown';
  await assert.rejects(calculateForecast(missingDate, { tier: 'free', maxPaths: 100 }, req({})), e => e.code === 'invalid_scenario');
});

test('unknown projections resolve to current website values, disclose every default and match the browser', async () => {
  const explicit = scenario(), unknown = structuredClone(explicit);
  unknown.market = 'unknown'; delete unknown.healthcare; unknown.longTermCare = null; unknown.postRetirementAllocation = {};
  delete unknown.spending.generalInflationMean; unknown.spending.generalInflationStdDev = 'unknown';
  const original = structuredClone(unknown), resolved = resolveModelDefaults(unknown);
  assert.deepEqual(unknown, original);
  assert.deepEqual(resolved.scenario, explicit);
  assert.deepEqual(resolved.defaultsApplied, websiteModelDefaults());
  const input = forecastInput.parse(args(unknown)), entitlement = { tier: 'free', maxPaths: 100 };
  const result = await calculateForecast(input, entitlement, req({}));
  const reference = await calculateForecast(args(explicit), entitlement, req({}));
  assert.equal(forecastOutput.safeParse(result).success, true);
  assert.deepEqual(result.forecasts[0].defaultsApplied, websiteModelDefaults());
  const summary = { ...result.forecasts[0] }; delete summary.defaultsApplied;
  assert.deepEqual(summary, reference.forecasts[0]);
  assert.ok(result.warnings.some(w => w.includes('22 website illustrative defaults')));
  const browserPlan = { ...explicit, id: 'mcp-1', seed: result.forecasts[0].provenance.randomSeed, numberOfSimulations: 4, household: { ...explicit.household, asOfDate: '2026-10-05' } };
  assert.deepEqual(summary, summarizeForecast(runSimulation(browserPlan, () => {}, { includeRiskAnalysis: false, includePathPoints: false }), browserPlan, 0));
});

test('paired defaults are resolved independently and never override explicit rates, zeroes or disabled care', async () => {
  const a = scenario(), b = scenario(); a.market.stockMeanReturn = null;
  b.market.stockMeanReturn = .07; b.market.stockStdDev = 0; b.healthcare.preMedicareMonthlyPremium = 0;
  b.healthcare.includeMedicarePremiums = false; b.longTermCare.enabled = false; b.postRetirementAllocation.stockUnder30x = 0;
  const input = compareInput.parse({ schemaVersion: '1.0', processingAcknowledged: true, pathCount: 4, forecastDate: '2026-10-05', scenarios: [a, b] });
  const resolved = resolveModelDefaults(b);
  assert.deepEqual(resolved.scenario, b); assert.deepEqual(resolved.defaultsApplied, []);
  const result = await calculateForecast(input, { tier: 'free', maxPaths: 100 }, req({}));
  assert.deepEqual(result.forecasts[0].defaultsApplied.map(d => [d.path, d.value]), [['market.stockMeanReturn', .133]]);
  assert.equal(result.forecasts[1].defaultsApplied, undefined);
  assert.equal(forecastOutput.safeParse(result).success, true);
  a.market.stockMeanReturn = .133;
  const reference = await calculateForecast({ ...input, scenarios: [a, b] }, { tier: 'free', maxPaths: 100 }, req({}));
  const summary = { ...result.forecasts[0] }; delete summary.defaultsApplied;
  assert.deepEqual(summary, reference.forecasts[0]); assert.deepEqual(result.forecasts[1], reference.forecasts[1]);
  assert.deepEqual(result.differences, reference.differences);
});

for (const variant of ['single', 'married', 'retired', 'employer', 'contributions', 'expenses']) {
  test(`hosted summary and async engine match direct browser calculation: ${variant}`, async () => {
    const s = scenario();
    if (variant === 'married') { s.household.filingStatus = 'Married'; s.household.separatePeople = true; s.household.birthday = '1966-10-05'; s.household.retirementDate = '2033-10-05'; s.household.spouseBirthday = '1966-10-05'; s.household.spouseRetirementDate = '2033-10-05'; }
    if (variant === 'retired') { s.household.alreadyRetired = true; s.household.birthday = '1960-10-05'; s.household.retirementDate = '2025-10-05'; }
    if (variant === 'employer') s.employerRothAccounts = [{ ...employerRothDefaults(), balance: 5000, contributionBasis: 5000, firstContributionYear: 2020, accessDate: '2033-10-05' }];
    if (variant === 'contributions') s.contributions.cash = 12000;
    if (variant === 'expenses') s.oneTimeExpenses = [{ label: 'Example expense', age: 75, amount: 10000 }];
    const result = await calculateForecast(args(s), { tier: 'free', maxPaths: 100 }, req({}));
    const direct = structuredClone(s); direct.numberOfSimulations = 4; direct.seed = result.forecasts[0].provenance.randomSeed; direct.id = 'mcp-1'; direct.household.asOfDate = '2026-10-05';
    const browser = runSimulation(direct, () => {}, { includeRiskAnalysis: false, includePathPoints: false });
    assert.deepEqual(result.forecasts[0], summarizeForecast(browser, direct, 0));
    assert.equal(forecastOutput.safeParse(result).success, true);
    const asyncResult = await runSimulationAsync(direct, () => {}, { includeRiskAnalysis: false, includePathPoints: false });
    delete browser.generatedAtEpochMillis; delete asyncResult.generatedAtEpochMillis; assert.deepEqual(asyncResult, browser);
  });
}
test('paired comparisons repeat, use shared dates and reject execution conflicts or entitlement bypass', async () => {
  const a = scenario(), b = scenario(); b.spending.annualBaseSpending = 60000;
  const input = { ...args(a), scenarios: [a, b] }; delete input.scenario;
  const ent = { tier: 'free', maxPaths: 100 }, one = await calculateForecast(input, ent, req({})), two = await calculateForecast(input, ent, req({}));
  assert.deepEqual(one, two); assert.equal(one.forecasts[0].provenance.randomSeed, one.forecasts[1].provenance.randomSeed);
  assert.equal(one.differences.shortfallFrequency, one.forecasts[1].shortfallFrequency - one.forecasts[0].shortfallFrequency);
  await assert.rejects(calculateForecast({ ...args(a), pathCount: 101 }, ent, req({})), e => e.code === 'path_limit');
  a.numberOfSimulations = 10000; await assert.rejects(calculateForecast(args(a), ent, req({})), e => e.code === 'conflicting_execution_fields');
  delete a.numberOfSimulations; a.seed = 1; await assert.rejects(calculateForecast(args(a), ent, req({})), e => e.code === 'conflicting_execution_fields');
  delete a.seed; await assert.rejects(calculateForecast({ ...args(a), forecastDate: '2026-02-30' }, ent, req({})), e => e.code === 'invalid_date');
});
test('missing calendar assumptions, validation errors and omitted date/count have explicit behavior', async () => {
  const s = scenario(), request = req({}), ent = { tier: 'free', maxPaths: 100 };
  s.household.birthday = ''; await assert.rejects(calculateForecast(args(s), ent, request), e => e.code === 'invalid_scenario');
  s.household.birthday = '1966-10-05'; s.accounts.pretax = -1; await assert.rejects(calculateForecast(args(s), ent, request), e => e.code === 'invalid_scenario');
  s.accounts.pretax = 500000;
  const input = args(s); delete input.pathCount; delete input.forecastDate;
  let observed;
  const direct = runSimulation({ ...s, id: 'mcp-1', numberOfSimulations: 4, seed: 20260766 });
  const output = await calculateForecast(input, ent, request, async plan => { observed = plan; return { ...direct, provenance: { ...direct.provenance, simulationCount: plan.numberOfSimulations } }; });
  assert.equal(observed.numberOfSimulations, 100); assert.equal(output.pathCount, 100); assert.equal(observed.household.asOfDate, new Date().toISOString().slice(0, 10));
});
test('cooperative cancellation and deadline produce no partial result', async () => {
  const controller = new AbortController(); controller.abort();
  const request = new Request(origin + '/mcp', { signal: controller.signal });
  await assert.rejects(calculateForecast(args(scenario()), { tier: 'free', maxPaths: 100 }, request), e => e.code === 'cancelled');
  const realNow = Date.now; let t = 0;
  try { Date.now = () => (t += 20001); await assert.rejects(calculateForecast(args(scenario()), { tier: 'free', maxPaths: 100 }, req({})), e => e.code === 'computation_limit'); }
  finally { Date.now = realNow; }
});
test('atomic quotas, active leases, hour boundaries, stale expiry and privacy', async () => {
  const DB = database(), env = { DB }, now = 100 * 3600000 + 3599000;
  try {
    const outcomes = await Promise.allSettled([acquireUsage(env, 'same-account', false, now), acquireUsage(env, 'same-account', false, now)]);
    assert.equal(outcomes.filter(r => r.status === 'fulfilled').length, 1);
    await assert.rejects(acquireUsage(env, 'same-account', false, now + 2000), e => e.code === 'calculation_in_progress');
    await finishUsage(env, outcomes.find(r => r.status === 'fulfilled').value, 'create_retirement_forecast', 'completed', 500, now);
    for (let i = 1; i < 20; i++) { const usage = await acquireUsage(env, 'same-account', false, now); await finishUsage(env, usage, 'create_retirement_forecast', 'completed', 500, now); }
    await assert.rejects(acquireUsage(env, 'same-account', false, now), e => e.code === 'hourly_limit');
    const reset = await acquireUsage(env, 'same-account', false, now + 2000); const stale = await acquireUsage(env, 'same-account', false, now + 122001);
    await finishUsage(env, reset, 'compare_retirement_scenarios', 'deadline', 20000, now + 122002);
    assert.equal(DB.db.prepare('SELECT lease FROM mcp_usage').get().lease, stale.lease);
    assert.equal(DB.db.prepare('SELECT account_key FROM mcp_usage').get().account_key, await accountKey('same-account'));
    assert.equal(JSON.stringify(DB.db.prepare('SELECT * FROM mcp_usage').all()).includes('same-account'), false);
    await finishUsage(env, stale, 'compare_retirement_scenarios', 'completed', 1200, now + 172922001);
    assert.equal(DB.db.prepare('SELECT COUNT(*) AS n FROM mcp_usage').get().n, 0);
  } finally { DB.db.close(); }
});
test('entitlements reuse paid, cancelled, linked account and owner handling; Stripe outages fail closed', async () => {
  const request = req({}, 'linked-user'), base = { STRIPE_SECRET_KEY: 'sk_live_example', STRIPE_PRO_LIVE_MONTHLY_PRICE_ID: 'price_pro', STRIPE_PRO_LIVE_YEARLY_PRICE_ID: 'price_year' };
  for (const status of ['active', 'trialing', 'canceled']) {
    const env = { ...base, STRIPE_FETCH: async url => new Response(JSON.stringify(String(url).includes('/customers/search') ? { data: [{ id: 'cus_link', metadata: { retirement_site_user_id: 'linked-user', retirement_firebase_uid: 'google-user' } }], has_more: false } : { data: [{ status, items: { data: [{ price: { id: 'price_pro' } }] } }], has_more: false }), { status: 200 }) };
    assert.equal((await forecastEntitlement(request, env)).maxPaths, status === 'canceled' ? 100 : 1000);
  }
  await assert.rejects(forecastEntitlement(request, { ...base, STRIPE_FETCH: async () => { throw new Error('offline'); } }));
  assert.equal((await forecastEntitlement(req({}), base)).maxPaths, 1000);
  assert.equal((await forecastEntitlement(request, {})).maxPaths, 100);
});
test('Pro quotas allow exactly 60 calls, notifications work and missing consent never consumes quota', async () => {
  const DB = database(), env = { DB }, now = 100 * 3600000;
  try {
    for (let i = 0; i < 60; i++) { const use = await acquireUsage(env, 'paid-user', true, now); await finishUsage(env, use, 'create_retirement_forecast', 'completed', 100, now); }
    await assert.rejects(acquireUsage(env, 'paid-user', true, now), e => e.code === 'hourly_limit');
    const input = args(scenario()); delete input.processingAcknowledged;
    const rejected = await body(await worker.fetch(req(call('create_retirement_forecast', input)), env)); assert.ok(rejected.error || rejected.result?.isError);
    assert.equal(DB.db.prepare('SELECT COUNT(*) AS n FROM mcp_usage').get().n, 1);
    const notification = await worker.fetch(req({ jsonrpc: '2.0', method: 'notifications/initialized' }), env); assert.equal(notification.status, 202);
  } finally { DB.db.close(); }
});
test('assistant onboarding routes render status without leaking runtime settings', async () => {
  const html = readFileSync(new URL('../dist/ai-assistants.html', import.meta.url), 'utf8');
  for (const enabled of [undefined, 'true']) {
    const response = await worker.fetch(new Request(origin + '/ai-assistants'), { MCP_CALCULATIONS_ENABLED: enabled, ASSETS: { fetch: async () => new Response(html) } });
    const text = await response.text(); assert.ok(text.includes(enabled ? 'Hosted forecasts are available' : 'Hosted forecasts are in owner testing'));
    assert.equal(text.includes('MCP_CALCULATIONS_ENABLED'), false); assert.equal(response.headers.get('cache-control'), 'no-store');
  }
});
test('HTTP boundary enforces auth, gated access, payload/origin limits, malformed calls and no cache', async () => {
  assert.equal((await worker.fetch(req(call('create_retirement_forecast', args(scenario())), null), {})).status, 401);
  assert.equal((await worker.fetch(req(call('create_retirement_forecast', args(scenario())), 'user'), {})).status, 503);
  assert.equal((await worker.fetch(req({}, owner, { Origin: 'https://other.example' }), {})).status, 403);
  assert.equal((await worker.fetch(req({ huge: 'a'.repeat(128 * 1024) }), {})).status, 413);
  const bad = new Request(origin + '/mcp', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{' });
  assert.equal((await worker.fetch(bad, {})).status, 400);
  const missing = await body(await worker.fetch(req(call('unknown', {})), {})); assert.ok(missing.error || missing.result?.isError);
  const response = await worker.fetch(req(call('explain_forecast_methodology', {}), null), {});
  assert.equal(response.headers.get('cache-control'), 'no-store'); assert.equal((await body(response)).result.structuredContent.schemaVersion, '1.0');
});
for (const mode of ['legacy', 'auto']) test(`official MCP client connects, discovers and calls methodology in ${mode} era`, async () => {
  const client = new Client({ name: 'mcp-test', version: '1' }, { versionNegotiation: { mode }, jsonSchemaValidator: new CfWorkerJsonSchemaValidator() });
  const transport = new StreamableHTTPClientTransport(new URL(origin + '/mcp'), { fetch: (input, init) => worker.fetch(new Request(input, init), {}) });
  try { await client.connect(transport); const list = await client.listTools(); assert.equal(list.tools.length, 3);
    const result = await client.callTool({ name: 'explain_forecast_methodology', arguments: { topic: 'defaults' } }); assert.equal(result.isError, undefined); assert.equal(result.structuredContent.topic, 'defaults');
    assert.deepEqual(result.structuredContent.modelDefaults, websiteModelDefaults());
    assert.equal(client.getProtocolEra(), mode === 'auto' ? 'modern' : 'legacy');
  } finally { await client.close(); }
});
test('legacy hosted-client messages with the newest header negotiate and discover successfully', async () => {
  const headers = { 'mcp-protocol-version': '2026-07-28' };
  const initialize = await worker.fetch(req({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2026-07-28', capabilities: {}, clientInfo: { name: 'hosted-connection-test', version: '1' } } }, null, headers), {});
  assert.equal(initialize.status, 200);
  assert.equal((await body(initialize)).result.protocolVersion, '2025-11-25');
  const list = await worker.fetch(req({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} }, null, headers), {});
  assert.equal(list.status, 200); assert.equal((await body(list)).result.tools.length, 3);
  const notification = await worker.fetch(req({ jsonrpc: '2.0', method: 'notifications/initialized' }, null, headers), {});
  assert.equal(notification.status, 202);
  const methodology = await worker.fetch(req(call('explain_forecast_methodology', { topic: 'privacy' }), null, headers), {});
  assert.equal(methodology.status, 200); assert.equal((await body(methodology)).result.structuredContent.topic, 'privacy');
  const unauthorized = await worker.fetch(req(call('create_retirement_forecast', args(scenario())), null, headers), {});
  assert.equal(unauthorized.status, 401);
  const gated = await worker.fetch(req(call('create_retirement_forecast', args(scenario())), 'nonowner', headers), {});
  assert.equal(gated.status, 503);
});
test('modern envelope and header mismatches remain rejected without logging personal values', async () => {
  const warnings = [], warn = console.warn; console.warn = (...values) => warnings.push(values);
  try {
    const response = await worker.fetch(req({ jsonrpc: '2.0', id: 'private-request-id', method: 'tools/list', params: { _meta: { 'io.modelcontextprotocol/protocolVersion': '2026-07-28' }, privateNote: 'private-financial-value' } }, null, { 'mcp-protocol-version': '2026-07-28' }), {});
    assert.equal(response.status, 400);
    assert.deepEqual(warnings, [['mcp_transport_rejected', 'tools/list', 400]]);
    assert.equal(JSON.stringify(warnings).includes('private-'), false);
    const _meta = { 'io.modelcontextprotocol/protocolVersion': '2026-07-28', 'io.modelcontextprotocol/clientCapabilities': {} };
    const wrongMethod = await worker.fetch(req({ jsonrpc: '2.0', id: 2, method: 'server/discover', params: { _meta } }, null, { 'mcp-protocol-version': '2026-07-28', 'mcp-method': 'tools/list' }), {});
    assert.equal(wrongMethod.status, 400);
    const wrongName = await worker.fetch(req({ ...call('explain_forecast_methodology', {}), params: { name: 'explain_forecast_methodology', arguments: {}, _meta } }, null, { 'mcp-protocol-version': '2026-07-28', 'mcp-name': 'create_retirement_forecast' }), {});
    assert.equal(wrongName.status, 400);
  } finally { console.warn = warn; }
});
test('modern hosted-client discovery and tool calls derive absent routing headers from the validated body', async () => {
  const _meta = { 'io.modelcontextprotocol/protocolVersion': '2026-07-28', 'io.modelcontextprotocol/clientCapabilities': {}, 'io.modelcontextprotocol/clientInfo': { name: 'hosted-connection-test', version: '1' } };
  const headers = { 'mcp-protocol-version': '2026-07-28' };
  for (const method of ['server/discover', 'tools/list', 'tools/call']) {
    const params = { _meta, ...(method === 'tools/call' ? { name: 'explain_forecast_methodology', arguments: { topic: 'privacy' } } : {}) };
    const response = await worker.fetch(req({ jsonrpc: '2.0', id: 1, method, params }, null, headers), {});
    assert.equal(response.status, 200);
    const result = (await body(response)).result;
    if (method === 'server/discover') assert.deepEqual(result.supportedVersions, ['2026-07-28']);
    if (method === 'tools/list') assert.equal(result.tools.length, 3);
    if (method === 'tools/call') assert.equal(result.structuredContent.topic, 'privacy');
  }
});
test('owner calculation tool records only aggregate outcomes and releases its pseudonymous lease', async () => {
  const DB = database();
  const input = args(scenario()); input.scenario.market = 'unknown';
  try { const response = await worker.fetch(req(call('create_retirement_forecast', input)), { DB }); const result = (await body(response)).result;
    assert.equal(result.isError, undefined); assert.equal(result.structuredContent.pathCount, 4);
    assert.equal(result.structuredContent.forecasts[0].defaultsApplied.length, 6);
    assert.ok(result.content[0].text.includes('Stock return average: 13.3%'));
    assert.equal(DB.db.prepare('SELECT lease_until FROM mcp_usage').get().lease_until, 0);
    const aggregate = DB.db.prepare('SELECT * FROM mcp_daily').get(); assert.equal(aggregate.outcome, 'completed');
    assert.deepEqual(Object.keys(aggregate).sort(), ['count', 'date', 'duration_band', 'outcome', 'tool']);
    assert.equal((await worker.fetch(new Request(origin + '/api/admin/mcp'), { DB })).status, 403);
    const admin = await worker.fetch(new Request(origin + '/api/admin/mcp', { headers: { 'oai-authenticated-user-id': owner } }), { DB }); assert.equal((await admin.json()).completed, 1);
  } finally { DB.db.close(); }
});
