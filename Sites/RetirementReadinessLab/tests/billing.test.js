import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../worker/index.js';
import { access as billingAccess } from '../worker/billing.js';
import { baseScenario, normalizeScenario, applyProSimulationDefault, DEFAULT_SEED, FREE_SIMULATION_PATHS, MAX_SIMULATION_PATHS } from '../dist/model.js';
import { runSimulation } from '../dist/engine.js';

const assets = { fetch: async () => new Response('planner') };
const site = 'https://retirementforecast.us';
function request(path, { user, email, method = 'GET', origin, interval = 'monthly' } = {}) {
  const checkout = path === '/api/billing/checkout' && method === 'POST';
  return new Request(site + path, { method, headers: { ...(user ? { 'oai-authenticated-user-id': user } : {}), ...(email ? { 'oai-authenticated-user-email': email } : {}), ...(origin ? { Origin: origin } : {}), ...(checkout ? { 'Content-Type': 'application/json' } : {}) }, body: checkout ? JSON.stringify({ interval }) : undefined });
}
const customer = { id: 'cus_123', metadata: { retirement_site_user_id: 'user-123' } };
const subscription = { id: 'sub_123', status: 'active', items: { data: [{ price: { id: 'price_monthly' } }] } };
const calls = [];
const stripeEnv = { ASSETS: assets, STRIPE_SECRET_KEY: 'sk_live_fake', STRIPE_PRO_MONTHLY_PRICE_ID: 'price_test_monthly', STRIPE_PRO_YEARLY_PRICE_ID: 'price_test_yearly', STRIPE_PRO_LIVE_MONTHLY_PRICE_ID: 'price_monthly', STRIPE_PRO_LIVE_YEARLY_PRICE_ID: 'price_yearly', STRIPE_FETCH: async (url, init) => {
  const parsed = new URL(url); calls.push({ path: parsed.pathname, search: parsed.searchParams, body: init.body, method: init.method });
  if (parsed.pathname === '/v1/customers/search') return Response.json({ data: [customer], has_more: false });
  if (parsed.pathname === '/v1/subscriptions') return Response.json({ data: [subscription] });
  if (parsed.pathname === '/v1/checkout/sessions/cs_test_12345678') return Response.json({ client_reference_id: 'user-123', customer: customer.id, subscription: subscription.id });
  if (parsed.pathname === '/v1/subscriptions/sub_123') return Response.json(subscription);
  if (parsed.pathname === '/v1/checkout/sessions') return Response.json({ url: 'https://checkout.stripe.com/c/pay/example' });
  if (parsed.pathname === '/v1/billing_portal/sessions') return Response.json({ url: 'https://billing.stripe.com/p/session/example' });
  return Response.json({ error: 'unexpected' }, { status: 404 });
} };

test('free model defaults to four paths, up to ten thousand validates, and imported seed is reset', () => {
  const base = baseScenario();
  assert.equal(base.numberOfSimulations, FREE_SIMULATION_PATHS);
  assert.equal(MAX_SIMULATION_PATHS, 10000);
  const imported = normalizeScenario({ ...base, numberOfSimulations: 5000, seed: 9 });
  assert.equal(imported.seed, DEFAULT_SEED);
  assert.equal(imported.numberOfSimulations, 5000);
  assert.equal(runSimulation(base).provenance.simulationCount, 4);
});

test('Pro defaults to 10,000 paths and preserves a manually selected four paths', () => {
  const fresh = baseScenario();
  assert.equal(applyProSimulationDefault(fresh), true);
  assert.equal(fresh.numberOfSimulations, 10000);
  assert.equal(applyProSimulationDefault(fresh), false);
  const savedFreePlan = normalizeScenario({ ...baseScenario(), numberOfSimulations: 4 });
  assert.equal(applyProSimulationDefault(savedFreePlan), true);
  const chosenFour = normalizeScenario({ ...baseScenario(), numberOfSimulations: 4, simulationPathsCustomized: true });
  assert.equal(applyProSimulationDefault(chosenFour), false);
  assert.equal(chosenFour.numberOfSimulations, 4);
  const chosenFiveHundred = normalizeScenario({ ...baseScenario(), numberOfSimulations: 500 });
  assert.equal(applyProSimulationDefault(chosenFiveHundred), false);
  assert.equal(chosenFiveHundred.numberOfSimulations, 500);
});

test('anonymous and unconfigured accounts remain at four paths without Stripe calls', async () => {
  calls.length = 0;
  const anonymous = await (await worker.fetch(request('/api/billing/status'), stripeEnv)).json();
  assert.deepEqual({ tier: anonymous.tier, maxPaths: anonymous.maxPaths, checkoutAvailable: anonymous.checkoutAvailable }, { tier: 'free', maxPaths: 4, checkoutAvailable: true });
  const unconfigured = await (await worker.fetch(request('/api/billing/status', { user: 'user-123' }), { ASSETS: assets })).json();
  assert.equal(unconfigured.maxPaths, 4);
  assert.equal(unconfigured.checkoutAvailable, false);
  assert.equal(calls.length, 0);
});

test('Sites owner has complimentary Pro without Stripe; anonymous email does not grant it', async () => {
  calls.length = 0;
  const noStripe = { ASSETS: assets };
  const owner = await (await worker.fetch(request('/api/billing/status', { user: '028696a7-7846-4822-a4c1-67026aa2383f' }), noStripe)).json();
  assert.equal(owner.tier, 'pro'); assert.equal(owner.maxPaths, 10000);
  assert.equal(owner.ownerAccess, true); assert.equal(owner.checkoutAvailable, false);
  const emailOwner = await (await worker.fetch(request('/api/billing/status', { user: 'current-owner-id', email: 'rtsoliday@gmail.com' }), noStripe)).json();
  assert.equal(emailOwner.tier, 'pro'); assert.equal(emailOwner.maxPaths, 10000);
  const emailWithoutSignIn = await (await worker.fetch(request('/api/billing/status', { email: 'rtsoliday@gmail.com' }), noStripe)).json();
  assert.equal(emailWithoutSignIn.tier, 'free');
  const otherUser = await (await worker.fetch(request('/api/billing/status', { user: 'someone-else', email: 'other@example.com' }), noStripe)).json();
  assert.equal(otherUser.tier, 'free');
  const checkout = await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: '028696a7-7846-4822-a4c1-67026aa2383f' }), stripeEnv);
  assert.equal(checkout.status, 409);
  assert.equal(calls.length, 0);
});

test('active matching subscription grants Pro and canceled subscription does not', async () => {
  calls.length = 0;
  const active = await (await worker.fetch(request('/api/billing/status', { user: 'user-123' }), stripeEnv)).json();
  assert.equal(active.tier, 'pro'); assert.equal(active.maxPaths, 10000);
  assert.match(calls[0].search.get('query'), /retirement_site_user_id/);
  assert.equal(calls.find(call => call.path === '/v1/subscriptions').search.has('price'), false);
  const yearlySubscription = { ...subscription, items: { data: [{ price: { id: 'price_yearly' } }] } };
  const yearlyEnv = { ...stripeEnv, STRIPE_FETCH: async (url) => new URL(url).pathname === '/v1/customers/search' ? Response.json({ data: [customer] }) : Response.json({ data: [yearlySubscription] }) };
  assert.equal((await (await worker.fetch(request('/api/billing/status', { user: 'user-123' }), yearlyEnv)).json()).tier, 'pro');
  const canceledEnv = { ...stripeEnv, STRIPE_FETCH: async (url) => new URL(url).pathname === '/v1/customers/search' ? Response.json({ data: [customer] }) : Response.json({ data: [{ ...subscription, status: 'canceled' }] }) };
  const canceled = await (await worker.fetch(request('/api/billing/status', { user: 'user-123' }), canceledEnv)).json();
  assert.equal(canceled.tier, 'free'); assert.equal(canceled.maxPaths, 4);
});

test('return session is tied to the authenticated account before Pro is granted', async () => {
  const valid = await (await worker.fetch(request('/api/billing/status?session_id=cs_test_12345678', { user: 'user-123' }), stripeEnv)).json();
  assert.equal(valid.tier, 'pro');
  const other = await (await worker.fetch(request('/api/billing/status?session_id=cs_test_12345678', { user: 'other-user' }), stripeEnv)).json();
  assert.equal(other.tier, 'free');
});

test('Checkout and portal reject anonymous or cross-origin writes', async () => {
  calls.length = 0;
  for (const path of ['/api/billing/checkout', '/api/billing/portal']) {
    assert.equal((await worker.fetch(request(path, { method: 'POST', origin: site }), stripeEnv)).status, 401);
    assert.equal((await worker.fetch(request(path, { method: 'POST', origin: 'https://attacker.test', user: 'user-123' }), stripeEnv)).status, 403);
  }
  assert.equal(calls.length, 0);
});

test('an existing Pro customer is not offered a duplicate Checkout', async () => {
  const response = await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'user-123' }), stripeEnv);
  assert.equal(response.status, 409);
});

test('Checkout and billing portal return only expected Stripe destinations', async () => {
  const freeEnv = { ...stripeEnv, STRIPE_FETCH: async (url, init) => {
    const parsed = new URL(url);
    if (parsed.pathname === '/v1/customers/search') return Response.json({ data: [customer] });
    if (parsed.pathname === '/v1/subscriptions') return Response.json({ data: [] });
    return stripeEnv.STRIPE_FETCH(url, init);
  } };
  calls.length = 0;
  const checkout = await (await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'user-123' }), freeEnv)).json();
  assert.equal(new URL(checkout.url).hostname, 'checkout.stripe.com');
  assert.equal(new URLSearchParams(calls.find(call => call.path === '/v1/checkout/sessions').body).get('line_items[0][price]'), 'price_monthly');
  calls.length = 0;
  const yearly = await (await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'user-123', interval: 'yearly' }), freeEnv)).json();
  assert.equal(new URL(yearly.url).hostname, 'checkout.stripe.com');
  assert.equal(new URLSearchParams(calls.find(call => call.path === '/v1/checkout/sessions').body).get('line_items[0][price]'), 'price_yearly');
  calls.length = 0;
  assert.equal((await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'user-123', interval: 'weekly' }), freeEnv)).status, 400);
  assert.equal(calls.length, 0);
  const portal = await (await worker.fetch(request('/api/billing/portal', { method: 'POST', origin: site, user: 'user-123' }), freeEnv)).json();
  assert.equal(new URL(portal.url).hostname, 'billing.stripe.com');
});

test('sandbox visitors cannot check out and the signed-in owner has complimentary Pro', async () => {
  calls.length = 0;
  const testEnv = { ...stripeEnv, STRIPE_SECRET_KEY: 'sk_test_fake' };
  const visitor = await (await worker.fetch(request('/api/billing/status', { user: 'someone-else' }), testEnv)).json();
  assert.equal(visitor.checkoutAvailable, false);
  assert.equal(visitor.maxPaths, 4);
  assert.equal((await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'someone-else' }), testEnv)).status, 403);
  assert.equal(calls.length, 0);
  const owner = await (await worker.fetch(request('/api/billing/status', { user: 'site-owner-id', email: 'rtsoliday@gmail.com' }), testEnv)).json();
  assert.equal(owner.tier, 'pro'); assert.equal(owner.checkoutAvailable, false);
  assert.equal((await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'site-owner-id', email: 'rtsoliday@gmail.com' }), testEnv)).status, 409);
  assert.equal(calls.length, 0);
});

test('Stripe key mode selects matching prices and disables billing if live prices are missing', async () => {
  const testEnv = { ...stripeEnv, STRIPE_SECRET_KEY: 'sk_test_fake', STRIPE_FETCH: async (url) => {
    const path = new URL(url).pathname;
    if (path === '/v1/customers/search') return Response.json({ data: [customer] });
    if (path === '/v1/subscriptions') return Response.json({ data: [{ ...subscription, items: { data: [{ price: { id: 'price_test_yearly' } }] } }] });
    return Response.json({}, { status: 404 });
  } };
  assert.equal((await billingAccess(testEnv, 'user-123')).pro, true);
  const missingLive = { ...stripeEnv, STRIPE_PRO_LIVE_MONTHLY_PRICE_ID: undefined };
  const access = await (await worker.fetch(request('/api/billing/status', { user: 'user-123' }), missingLive)).json();
  assert.equal(access.checkoutAvailable, false);
  const unknownKey = { ...stripeEnv, STRIPE_SECRET_KEY: 'invalid_key' };
  const unknownAccess = await (await worker.fetch(request('/api/billing/status', { user: 'user-123' }), unknownKey)).json();
  assert.equal(unknownAccess.checkoutAvailable, false);
});
