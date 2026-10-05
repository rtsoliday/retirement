import { createMcpHandler, McpServer } from '@modelcontextprotocol/server';
import { CfWorkerJsonSchemaValidator } from '@modelcontextprotocol/server/validators/cf-worker';
import * as z from 'zod/v4';
import { runSimulationAsync } from '../dist/engine.js';
import { validateScenario, calendarDate, retirementAge, scenarioWarnings } from '../dist/model.js';
import { forecastEntitlement } from './billing.js';
import { acquireUsage, finishUsage, UsageLimitError, startExecution, cancelExecution, validRequestId } from './mcp-storage.js';
import { calculationGate, hostedExecution } from './mcp-execution.js';
import { forecastInput, compareInput, methodologyInput, forecastOutput, modelDefaultOutput, SCHEMA_VERSION, MCP_SEED } from './mcp-schema.js';
import { resolveModelDefaults, websiteModelDefaults } from './mcp-defaults.js';

const CALCULATORS = new Set(['create_retirement_forecast', 'compare_retirement_scenarios']);
const LINKS = { planner: 'https://retirementforecast.us/', methodology: 'https://retirementforecast.us/methodology', privacy: 'https://retirementforecast.us/privacy' };
const MAX_BYTES = 128 * 1024;
const LEGACY_METHODS = new Set(['initialize', 'notifications/initialized', 'ping', 'tools/list', 'tools/call']);
const TOOL_NAMES = new Set([...CALCULATORS, 'explain_forecast_methodology']);
function transportRequest(request, message) {
  const meta = message?.params?._meta;
  const hasModernEnvelope = meta && typeof meta === 'object' && Object.keys(meta).some(key => key.startsWith('io.modelcontextprotocol/'));
  if (request.headers.get('mcp-protocol-version') !== '2026-07-28') return request;
  const headers = new Headers(request.headers);
  if (hasModernEnvelope) {
    // The Sites discovery client supplies the envelope but may omit the new
    // routing headers. Derive missing routing hints from known methods/names;
    // supplied mismatches and malformed envelopes remain SDK errors.
    if (!LEGACY_METHODS.has(message?.method) && message?.method !== 'server/discover') return request;
    if (!headers.has('mcp-method')) headers.set('mcp-method', message.method);
    if (message.method === 'tools/call' && TOOL_NAMES.has(message.params?.name) && !headers.has('mcp-name')) headers.set('mcp-name', message.params.name);
  } else {
    // Some hosted clients send legacy handshake/message shapes with the newest
    // version header. Negotiate the supported legacy revision for those shapes;
    // genuine modern envelopes still receive the SDK's strict validation.
    if (!LEGACY_METHODS.has(message?.method)) return request;
    headers.set('mcp-protocol-version', '2025-11-25');
  }
  return new Request(request.url, { method: request.method, headers, signal: request.signal });
}
class ForecastError extends Error { constructor(code, message) { super(message); this.code = code; } }
function errorResult(code, message) { return { isError: true, content: [{ type: 'text', text: JSON.stringify({ code, message }) }] }; }
function responseError(status, code, message, id = null) {
  return new Response(JSON.stringify({ jsonrpc: '2.0', id, error: { code: -32000, message, data: { code } } }), { status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } });
}
export function summarizeForecast(result, scenario, index) {
  const start = retirementAge(scenario), annual = age => Math.abs((age - start) - Math.round(age - start)) < 1e-7;
  return {
    label: scenario.name || `Scenario ${index + 1}`, successProbability: result.successProbability, shortfallFrequency: 1 - result.successProbability,
    medianEndingBalance: result.todayDollars.medianEndingBalance, p10EndingBalance: result.todayDollars.pessimisticEndingBalance, p90EndingBalance: result.todayDollars.optimisticEndingBalance,
    medianFailureAge: result.medianFailureAge, failureAgeBuckets: result.failureAgeBuckets,
    annualBalanceRanges: result.todayDollars.balanceBands.filter(row => annual(row.age)),
    fundedShareByAge: result.notFailedByAge.filter(row => annual(row.age)), provenance: result.provenance,
  };
}
export async function calculateForecast(args, entitlement, request, runner = runSimulationAsync, execution = {}) {
  const pathCount = args.pathCount ?? entitlement.maxPaths, forecastDate = args.forecastDate ?? new Date().toISOString().slice(0, 10);
  if (pathCount > entitlement.maxPaths) throw new ForecastError('path_limit', `Your ${entitlement.tier} access permits at most ${entitlement.maxPaths} paths per scenario.`);
  if (!calendarDate(forecastDate)) throw new ForecastError('invalid_date', 'Provide a valid forecastDate in YYYY-MM-DD format.');
  const appliedDefaults = [];
  const scenarios = (args.scenarios || [args.scenario]).map((input, i) => {
    if (input.numberOfSimulations !== undefined && input.numberOfSimulations !== pathCount) throw new ForecastError('conflicting_execution_fields', 'Remove scenario.numberOfSimulations or make it equal to pathCount.');
    if (input.seed !== undefined && input.seed !== MCP_SEED) throw new ForecastError('conflicting_execution_fields', `Remove scenario.seed or use the shared seed ${MCP_SEED}.`);
    if (input.household.asOfDate && input.household.asOfDate !== forecastDate) throw new ForecastError('conflicting_execution_fields', 'Remove household.asOfDate or make it equal to forecastDate.');
    const { scenario, defaultsApplied } = resolveModelDefaults(input);
    appliedDefaults.push(defaultsApplied);
    scenario.id = `mcp-${i + 1}`; scenario.numberOfSimulations = pathCount; scenario.seed = MCP_SEED; scenario.household.asOfDate = forecastDate;
    const h = scenario.household;
    if (!calendarDate(h.birthday) || !h.alreadyRetired && !calendarDate(h.retirementDate) ||
      h.filingStatus === 'Married' && (!calendarDate(h.spouseBirthday) || h.separatePeople && !h.spouseAlreadyRetired && !calendarDate(h.spouseRetirementDate))) {
      throw new ForecastError('invalid_scenario', `Scenario ${i + 1}: provide birthdays for active people and planned retirement dates for working people. Already-retired people may leave the actual separation date blank.`);
    }
    const errors = validateScenario(scenario);
    if (errors.length) throw new ForecastError('invalid_scenario', `Scenario ${i + 1}: ${errors.join(' ')}`);
    return scenario;
  });
  const start = Date.now(), deadlineMs = execution.deadlineMs ?? 20000, checkExecution = () => {
    if (request.signal.aborted || execution.signal?.aborted || execution.isCancelled?.() || execution.isActive && !execution.isActive()) throw new ForecastError('cancelled', 'Calculation was cancelled.');
    if (Date.now() - start >= deadlineMs) throw new ForecastError('computation_limit', `Calculation exceeded the ${deadlineMs / 1000}-second limit. No partial result was returned. Simplify the scenario or explicitly request fewer paths.`);
  };
  const forecasts = [];
  for (const [index, scenario] of scenarios.entries()) {
    checkExecution();
    // Retain only compact summaries between runs, never full path collections.
    const summary = summarizeForecast(await runner(scenario, () => {}, { includeRiskAnalysis: false, includePathPoints: false, checkExecution, ...(execution.yieldExecution ? { yieldExecution: execution.yieldExecution } : {}) }), scenario, index);
    if (appliedDefaults[index].length) summary.defaultsApplied = appliedDefaults[index];
    forecasts.push(summary);
  }
  checkExecution();
  const warnings = [
    'Hypothetical educational scenarios, not predictions, guarantees, or individualized financial advice.',
    'Balances use forecast-date purchasing power. Balance percentiles describe paths observed at each age; funded share includes successful paths after death and is not conditional on being alive.',
    ...scenarios.flatMap(s => scenarioWarnings(s)),
    ...forecasts.filter(f => f.defaultsApplied).map(f => `${f.label} uses ${f.defaultsApplied.length} website illustrative defaults. Review the listed assumptions; healthcare and care costs are examples, not personal estimates.`),
    ...(pathCount <= 100 ? ['Small 100-path-or-fewer runs are previews, too small to estimate retirement readiness. Describe frequencies as sample outcomes.'] : []),
  ];
  const output = { schemaVersion: SCHEMA_VERSION, forecastDate, dollarBasis: 'forecast-date purchasing power', tier: entitlement.tier, pathCount, forecasts, warnings: [...new Set(warnings)], links: LINKS };
  if (forecasts.length === 2) {
    const [a, b] = forecasts;
    output.differences = { shortfallFrequency: b.shortfallFrequency - a.shortfallFrequency, medianEndingBalance: b.medianEndingBalance - a.medianEndingBalance,
      p10EndingBalance: b.p10EndingBalance - a.p10EndingBalance, p90EndingBalance: b.p90EndingBalance - a.p90EndingBalance,
      medianFailureAge: a.medianFailureAge === null || b.medianFailureAge === null ? null : b.medianFailureAge - a.medianFailureAge };
    output.warnings.push('Differences are scenario 2 minus scenario 1. Shared random draws reduce comparison noise; they do not establish causation.');
  }
  return output;
}
export function methodology(topic = 'overview') {
  return { schemaVersion: SCHEMA_VERSION, topic, links: LINKS,
    model: 'U.S. monthly retirement cashflows, stochastic stock/bond returns and inflation, federal 2026 income taxes, Social Security, SSA Trustees Alt2 2025 mortality, account-specific withdrawal and Roth rules. State tax and individualized advice are outside scope.',
    inputs: 'Use the scenario schema from tools/list. Ask for missing personal amounts, retirement/birth dates, spouse assumptions, balances and Roth histories, contributions, income, ordinary spending and home costs. When the user answers unknown or leaves a projected market, inflation, healthcare, long-term-care or stock-allocation assumption unspecified, offer the exact website defaults listed in modelDefaults. After review, send those fields as null, "unknown" or omit them; the server fills them and reports defaultsApplied. Never choose your own replacement rate, convert an unknown personal amount to zero, or replace an explicit value. Rates are decimals (0.04 is 4%). Annual amounts are USD unless the field says monthly. Other inactive features need explicit zero/false values, empty arrays or blank dates. Planner-export execution fields must match pathCount, forecastDate and the fixed seed, or be removed.',
    review: 'Review all financial assumptions with the user, identifying any website defaults as illustrative assumptions, including healthcare costs per adult and long-term-care costs per person. Unknown is permission to offer these defaults, not to guess personal facts. Explain that their inputs go to the hosted service and results return to their assistant; obtain acknowledgment before setting processingAcknowledged=true. Disclose defaultsApplied when presenting results.',
    modelDefaults: websiteModelDefaults(),
    limits: 'Authenticated Free: up to 100 paths and 20 calculation calls/hour. Pro and owner: up to 1000 paths and 60 calls/hour. One active calculation/account; a comparison contains exactly two complete scenarios. Request maximum 128 KiB, cooperative calculation deadline 20 seconds. Checkout and linking sign-ins happen on the website. General access may be disabled during hosted verification.',
    comparisons: `Each scenario uses the same forecast date and fixed seed ${MCP_SEED}. Differences are second minus first. Path counts above entitlement are rejected. Expensive sensitivity analysis and individual path traces are omitted.`,
    privacy: 'No forecast inputs/results in application storage, caches, URLs or diagnostic logs. Temporary computation sends results to the assistant. Pseudonymous quota counters expire after 48 hours; hashed cancellation identifiers expire after two minutes or are removed on completion. Expired records are deleted on subsequent activity; daily tool/outcome/duration aggregates are retained. Hosting and assistant providers have separate data policies.',
    interpretation: '100-path previews illustrate sample outcomes. Larger samples reduce Monte Carlo noise but do not fix assumptions. Read methodology before interpreting success or shortfalls.',
  };
}
const methodologyOutput = z.strictObject({ schemaVersion: z.literal(SCHEMA_VERSION), topic: z.string(), links: z.strictObject({ planner: z.string(), methodology: z.string(), privacy: z.string() }), model: z.string(), inputs: z.string(), review: z.string(), modelDefaults: z.array(modelDefaultOutput), limits: z.string(), comparisons: z.string(), privacy: z.string(), interpretation: z.string() });
async function callCalculation(name, args, request, env, isOwner, ctx) {
  let usage, outcome = 'unavailable', claim;
  const start = Date.now();
  try {
    let entitlement;
    try { entitlement = await forecastEntitlement(request, env, isOwner); }
    catch { throw new ForecastError('subscription_unavailable', 'Subscription verification is unavailable. Retry later.'); }
    claim = calculationGate.claim();
    if (!claim) throw new ForecastError('server_busy', 'Another forecast is using this server instance. Retry later.');
    usage = await acquireUsage(env, entitlement.user.id, entitlement.tier === 'pro');
    await startExecution(env, usage, ctx?.mcpReq?.id);
    const execution = hostedExecution(env, isOwner, () => calculationGate.owns(claim), usage);
    execution.signal = ctx?.mcpReq?.signal;
    const output = await calculateForecast(args, entitlement, request, runSimulationAsync, execution);
    outcome = 'completed';
    return { structuredContent: output, content: [{ type: 'text', text: `${output.pathCount}-path ${output.pathCount <= 100 ? 'preview' : 'forecast'} as of ${output.forecastDate}. ${output.forecasts.map(f => `${f.label}: ${(f.shortfallFrequency * 100).toFixed(1)}% of sample paths had a shortfall; median ending balance $${Math.round(f.medianEndingBalance).toLocaleString('en-US')} in forecast-date dollars.${f.defaultsApplied ? ` Website illustrative defaults used: ${f.defaultsApplied.map(d => `${d.label}: ${d.displayValue} (${d.units})`).join('; ')}.` : ''}`).join(' ')} ${output.warnings.join(' ')}` }] };
  } catch (error) {
    const code = error instanceof ForecastError || error instanceof UsageLimitError ? error.code : 'unavailable';
    outcome = code === 'computation_limit' ? 'deadline' : code === 'invalid_scenario' || code === 'invalid_date' || code === 'conflicting_execution_fields' ? 'invalid' : code === 'path_limit' || error instanceof UsageLimitError ? 'limited' : code === 'cancelled' ? 'cancelled' : 'unavailable';
    return errorResult(code, error instanceof ForecastError ? error.message : error instanceof UsageLimitError ? code === 'hourly_limit' ? 'Hourly calculation allowance reached. Retry next UTC hour.' : 'A calculation is already active for this account. Wait for it to finish.' : 'Forecast service is temporarily unavailable. Retry later.');
  } finally {
    if (claim) calculationGate.release(claim);
    if (env.DB) {
      try { await finishUsage(env, usage, name, outcome, Date.now() - start); }
      catch { /* Never log request data. An unreleased lease expires safely. */ }
    }
  }
}
export async function mcp(request, env, isOwner) {
  if (request.method !== 'POST') return responseError(405, 'method_not_allowed', 'Use POST /mcp.');
  const origin = request.headers.get('origin');
  if (origin && origin !== new URL(request.url).origin) return responseError(403, 'origin_rejected', 'Cross-origin browser calls are not permitted.');
  if (!request.headers.get('content-type')?.toLowerCase().startsWith('application/json')) return responseError(415, 'content_type', 'Use application/json.');
  let parsedBody;
  try {
    const reader = request.body?.getReader(); if (!reader) return responseError(400, 'invalid_request', 'Provide a JSON-RPC request.');
    let size = 0; const chunks = [];
    for (;;) { const { value, done } = await reader.read(); if (done) break; size += value.byteLength;
      if (size > MAX_BYTES) { await reader.cancel(); return responseError(413, 'request_limit', 'Request exceeds 128 KiB.'); } chunks.push(value); }
    const bytes = new Uint8Array(size); let offset = 0; for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    parsedBody = JSON.parse(new TextDecoder().decode(bytes));
  } catch { return responseError(400, 'invalid_json', 'Provide valid JSON.'); }
  if (Array.isArray(parsedBody)) return responseError(400, 'batch_unsupported', 'Send one JSON-RPC request at a time.');
  // Serving is stateless and cancellation may reach another isolate. Record
  // only a hashed request key against this authenticated account's live lease.
  if (parsedBody?.method === 'notifications/cancelled') {
    const userId = request.headers.get('oai-authenticated-user-id');
    if (!userId) return responseError(401, 'authentication_required', 'Sign in to cancel a forecast.');
    if (!isOwner && env.MCP_CALCULATIONS_ENABLED !== 'true') return responseError(503, 'verification_pending', 'Hosted forecasts are undergoing runtime verification.');
    if (parsedBody.jsonrpc === '2.0' && parsedBody.id === undefined && validRequestId(parsedBody.params?.requestId)) {
      try { await cancelExecution(env, userId, parsedBody.params.requestId); }
      catch { return responseError(503, 'unavailable', 'Forecast service is temporarily unavailable. Retry later.'); }
    }
    return new Response(null, { status: 202, headers: { 'Cache-Control': 'no-store' } });
  }
  if (parsedBody?.method === 'tools/call' && CALCULATORS.has(parsedBody.params?.name)) {
    if (!request.headers.get('oai-authenticated-user-id')) return responseError(401, 'authentication_required', 'Connect Retirement Forecast through Sites to calculate.', parsedBody.id ?? null);
    if (!isOwner && env.MCP_CALCULATIONS_ENABLED !== 'true') return responseError(503, 'verification_pending', 'Hosted forecasts are undergoing runtime verification. Methodology and discovery remain available.', parsedBody.id ?? null);
  }
  const handler = createMcpHandler(() => {
    const server = new McpServer({ name: 'Retirement Forecast', version: '1.0.0' }, { jsonSchemaValidator: new CfWorkerJsonSchemaValidator() });
    const annotations = { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true };
    const instruction = 'Use for hypothetical U.S. retirement planning. For unknown or unspecified projections, call explain_forecast_methodology for modelDefaults and offer the website illustrative assumptions. After reviewing them with the user, omit those fields or send null/"unknown"; the server fills the exact website defaults and reports defaultsApplied. Never guess a replacement return or turn an unknown personal amount into zero. Explicit user values are retained. First review complete assumptions and get acknowledgment of temporary hosted processing. Never invent missing personal inputs or describe sample outcomes as guarantees. Updates quota and aggregate usage counters; retrieves subscription access from Stripe.';
    server.registerTool('create_retirement_forecast', { title: 'Create a retirement forecast', description: `Forecast retirement savings, spending and shortfalls across Monte Carlo paths. ${instruction}`, inputSchema: forecastInput, outputSchema: forecastOutput, annotations }, (args, ctx) => callCalculation('create_retirement_forecast', args, request, env, isOwner, ctx));
    server.registerTool('compare_retirement_scenarios', { title: 'Compare two retirement scenarios', description: `Compare two complete retirement plans on shared random draws, such as retiring later or changing spending. ${instruction}`, inputSchema: compareInput, outputSchema: forecastOutput, annotations }, (args, ctx) => callCalculation('compare_retirement_scenarios', args, request, env, isOwner, ctx));
    server.registerTool('explain_forecast_methodology', { title: 'Explain retirement forecast assumptions', description: 'Explain the forecast model, input requirements, exact website illustrative defaults for unknown projections, privacy, Free/Pro limits, taxes, mortality and paired comparisons. Use before collecting inputs, resolving unknown assumptions or interpreting results.', inputSchema: methodologyInput, outputSchema: methodologyOutput, annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false } }, ({ topic }) => { const output = methodology(topic); return { structuredContent: output, content: [{ type: 'text', text: JSON.stringify(output) }] }; });
    return server;
  }, { legacy: 'stateless', responseMode: 'auto', onerror: () => {}, maxSubscriptions: 0 });
  const response = await handler.fetch(transportRequest(request, parsedBody), { parsedBody });
  if (response.status >= 400) {
    // Protocol diagnostics only: never include arguments, results, identities,
    // request IDs, or SDK error text that could echo user-provided values.
    console.warn('mcp_transport_rejected', LEGACY_METHODS.has(parsedBody?.method) || parsedBody?.method === 'server/discover' ? parsedBody.method : 'other', response.status);
  }
  const headers = new Headers(response.headers); headers.set('Cache-Control', 'no-store');
  return new Response(response.body, { status: response.status, headers });
}
