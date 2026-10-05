import test from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import worker from '../worker/index.js';
import { referralSource } from '../worker/metrics.js';
import { billingMetrics } from '../worker/admin-billing.js';
import { reportUsage } from '../dist/usage-metrics.js';

const origin = 'https://retirementforecast.us';
const owner = '028696a7-7846-4822-a4c1-67026aa2383f';
function database() {
  const db = new DatabaseSync(':memory:');
  db.exec(readFileSync(new URL('../drizzle/0000_opposite_morlocks.sql', import.meta.url), 'utf8'));
  return {
    db,
    prepare(sql) {
      const stmt = db.prepare(sql);
      let values = [];
      return { bind(...args) { values = args; return this; }, run() { return stmt.run(...values); },
        async all() { return { success: true, results: stmt.all(...values) }; }, async first() { return stmt.get(...values); } };
    },
    async batch(statements) {
      db.exec('BEGIN');
      try { const results = statements.map(s => s.run()); db.exec('COMMIT'); return results; }
      catch (e) { db.exec('ROLLBACK'); throw e; }
    },
  };
}
function event(payload, headers = {}) {
  return new Request(origin + '/api/metrics/event', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(payload) });
}
const get = (path, user = owner) => new Request(origin + path, { headers: user ? { 'oai-authenticated-user-id': user } : {} });

test('events persist, deduplicate and aggregate daily counts without scenario data or internal referrals', async () => {
  const DB = database(), env = { DB };
  try {
    const start = { id: randomUUID(), event: 'simulation_start' };
    for (let i = 0; i < 3; i++) assert.equal((await worker.fetch(event(start), env)).status, 204);
    for (const referrer of ['google.com', 'google.com', '', 'www.retirementforecast.us']) {
      assert.equal((await worker.fetch(event({ id: randomUUID(), event: 'arrival', referrer }), env)).status, 204);
    }
    // Signed-in owner runs do not inflate product usage.
    assert.equal((await worker.fetch(event({ id: randomUUID(), event: 'simulation_start' }, { 'oai-authenticated-user-id': owner }), env)).status, 204);
    const response = await worker.fetch(get('/api/admin/usage?days=7'), env), data = await response.json();
    assert.equal(response.status, 200);
    assert.equal(data.simulationStarts, 1);
    assert.equal(data.series.length, 7);
    assert.deepEqual(data.referrals, [{ source: 'google.com', arrivals: 2 }, { source: 'Direct / unknown', arrivals: 1 }]);
    assert.equal(DB.db.prepare('SELECT COUNT(*) AS n FROM metric_receipts').get().n, 5);
    assert.equal(response.headers.get('cache-control'), 'no-store');
    assert.equal((await worker.fetch(get('/api/admin/usage?days=30'), env)).status, 200);
  } finally { DB.db.close(); }
});

test('measurement rejects cross-origin, financial payloads, bad events and fails safely without storage', async () => {
  const base = { id: randomUUID(), event: 'simulation_start' };
  for (const [request, expected] of [
    [event(base, { Origin: 'https://evil.test' }), 403],
    [event({ ...base, balance: 100000 }), 400],
    [event({ ...base, event: 'unknown' }), 400],
    [event({ ...base, id: 'bad' }), 400],
    [event(base), 503],
  ]) {
    assert.equal((await worker.fetch(request, { DB: expected === 503 ? null : {} })).status, expected);
  }
  assert.equal(referralSource('https://google.com/private?balance=100', 'retirementforecast.us'), 'google.com');
  assert.equal(referralSource('javascript:alert(1)', 'retirementforecast.us'), 'Direct / unknown');
});

test('both admin APIs fail closed before touching databases or Stripe', async () => {
  for (const path of ['/api/admin/usage', '/api/admin/billing']) {
    const env = { DB: { prepare() { throw Error('unauthorized access'); } }, STRIPE_FETCH() { throw Error('unauthorized access'); } };
    assert.equal((await worker.fetch(get(path, null), env)).status, 403);
    assert.equal((await worker.fetch(get(path, 'other'), env)).status, 403);
    assert.equal((await worker.fetch(get(path + '?days=1'), env)).status, 400);
  }
});

const price = (amount, interval = 'month') => ({ id: 'price_pro', currency: 'usd', unit_amount: amount, billing_scheme: 'per_unit', recurring: { interval, interval_count: 1, usage_type: 'licensed' } });
const sub = (id, amount, interval = 'month', status = 'active') => ({ id, status, customer: { id: 'cus_' + id, metadata: {} }, metadata: { retirement_plan: 'pro' }, items: { has_more: false, data: [{ quantity: 1, price: price(amount, interval) }] } });

test('Stripe statistics paginate, normalize annual prices, include ended subscription receipts and use payment dates', async () => {
  const window = { start: '2026-10-01', end: '2026-10-08' }, paidAt = Date.parse('2026-10-03') / 1000;
  const calls = [];
  const env = { STRIPE_SECRET_KEY: 'sk_test_example', STRIPE_FETCH: async input => {
    const url = new URL(input); calls.push(url);
    if (url.pathname.endsWith('/subscriptions')) return Response.json(url.searchParams.has('starting_after') ?
      { data: [sub('ended', 500, 'month', 'canceled'), sub('free', 0), sub('trial', 500, 'month', 'trialing'), { ...sub('unrelated', 999), metadata: {}, items: { data: [{ price: { id: 'other' } }] } }], has_more: false } :
      { data: [sub('monthly', 500), sub('annual', 2000, 'year')], has_more: true });
    const invoice = (id, subscription, amount, date) => ({ id, parent: { subscription_details: { subscription } }, amount_paid: amount, currency: 'usd', status_transitions: { paid_at: date }, created: Date.parse('2026-09-01') / 1000 });
    return Response.json(url.searchParams.has('starting_after') ? { data: [invoice('in_ended', 'ended', 500, paidAt), invoice('in_old', 'monthly', 500, Date.parse('2026-09-30') / 1000), invoice('in_other', 'unrelated', 999, paidAt)], has_more: false } :
      { data: [invoice('in_annual', 'annual', 2000, paidAt)], has_more: true });
  } };
  const data = await billingMetrics(env, window);
  assert.equal(data.mode, 'test');
  assert.equal(data.activeSubscriptions, 2);
  assert.equal(data.monthlyRecurringCents, 500 + 2000 / 12);
  assert.equal(data.grossCollectedCents, 2500);
  assert.equal(calls.length, 4);
  assert.equal(calls[1].searchParams.get('starting_after'), 'annual');
});

test('Stripe outages and broken pagination produce unavailable, never false zero totals', async () => {
  for (const STRIPE_FETCH of [async () => { throw Error('offline'); }, async () => Response.json({ data: [], has_more: true })]) {
    const response = await worker.fetch(get('/api/admin/billing'), { STRIPE_SECRET_KEY: 'sk_live_example', STRIPE_FETCH });
    assert.equal(response.status, 503);
    assert.equal((await response.json()).activeSubscriptions, undefined);
  }
});

test('browser usage sends only a hostname or marker and tolerates offline failures', async () => {
  const oldFetch = globalThis.fetch, payloads = [];
  globalThis.fetch = async (_url, options) => { payloads.push(JSON.parse(options.body)); throw Error('offline'); };
  try {
    reportUsage('arrival', 'https://google.com/search?q=private'); reportUsage('simulation_start');
    await new Promise(setImmediate);
    assert.equal(payloads[0].referrer, 'google.com');
    assert.deepEqual(Object.keys(payloads[0]).sort(), ['event', 'id', 'referrer']);
    assert.deepEqual(Object.keys(payloads[1]).sort(), ['event', 'id']);
  } finally { globalThis.fetch = oldFetch; }
});
