import { stripe, isProSubscription } from './billing.js';

async function listAll(env, path, params) {
  const rows = [], seen = new Set();
  let cursor;
  for (let page = 0; page < 50; page++) {
    const result = await stripe(env, path, { limit: '100', ...params, ...(cursor ? { starting_after: cursor } : {}) });
    if (!Array.isArray(result.data) || typeof result.has_more !== 'boolean') throw new Error('Incomplete Stripe data');
    rows.push(...result.data);
    if (!result.has_more) return rows;
    cursor = result.data.at(-1)?.id;
    if (!cursor || seen.has(cursor)) throw new Error('Incomplete Stripe pagination');
    seen.add(cursor);
  }
  throw new Error('Stripe data exceeds dashboard lookup limit');
}

export function recurringMonthly(subscription) {
  if (subscription.items?.has_more || !Array.isArray(subscription.items?.data)) throw new Error('Incomplete Stripe items');
  return subscription.items.data.reduce((sum, item) => {
    const price = item.price, recurring = price?.recurring;
    if (!recurring) return sum;
    const amount = price.unit_amount ?? Number(price.unit_amount_decimal);
    if (price.currency !== 'usd' || price.billing_scheme !== 'per_unit' || recurring.usage_type === 'metered' ||
        !Number.isFinite(amount) || amount < 0 || !Number.isInteger(item.quantity) || item.quantity < 0 ||
        !Number.isInteger(recurring.interval_count) || recurring.interval_count < 1) throw new Error('Unsupported recurring price');
    const periods = { month: 1, year: 1 / 12 };
    if (!periods[recurring.interval]) throw new Error('Unsupported recurring interval');
    return sum + amount * item.quantity * periods[recurring.interval] / recurring.interval_count;
  }, 0);
}

export async function billingMetrics(env, window) {
  if (!env.STRIPE_SECRET_KEY) throw new Error('Stripe is not connected yet.');
  const subscriptions = await listAll(env, 'subscriptions', { status: 'all', 'expand[0]': 'data.customer' });
  const pro = subscriptions.filter(s => isProSubscription(s, env, typeof s.customer === 'object' ? s.customer.metadata : {}));
  const active = pro.filter(s => s.status === 'active' && recurringMonthly(s) > 0);
  const ids = new Set(pro.map(s => s.id));
  // Use payment date, not creation date: late payments count when collected.
  const invoices = await listAll(env, 'invoices', { status: 'paid' });
  let grossCollectedCents = 0;
  const start = Date.parse(window.start) / 1000, end = Date.parse(window.end) / 1000;
  for (const invoice of invoices) {
    const subscriptionId = invoice.subscription?.id || invoice.subscription ||
      invoice.parent?.subscription_details?.subscription?.id || invoice.parent?.subscription_details?.subscription;
    if (!ids.has(subscriptionId)) continue;
    const paidAt = invoice.status_transitions?.paid_at;
    if (!Number.isFinite(paidAt)) throw new Error('Missing Stripe payment date');
    if (paidAt < start || paidAt >= end) continue;
    if (invoice.currency !== 'usd' || !Number.isSafeInteger(invoice.amount_paid) || invoice.amount_paid < 0) throw new Error('Unsupported invoice amount');
    grossCollectedCents += invoice.amount_paid;
  }
  return { mode: /^(sk|rk)_test_/.test(env.STRIPE_SECRET_KEY) ? 'test' : 'live', currency: 'USD',
    activeSubscriptions: active.length,
    monthlyRecurringCents: active.reduce((sum, s) => sum + recurringMonthly(s), 0), grossCollectedCents };
}
