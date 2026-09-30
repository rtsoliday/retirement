// Stripe grants subscriber access; verified owner identities have complimentary Pro access.
// Financial scenarios never reach this Worker.
import { chatgptPrincipal, firebasePrincipal, principal as requestPrincipal, IdentityKeysUnavailableError, OWNER_CHATGPT_USER_ID, OWNER_GOOGLE_EMAIL, OWNER_FIREBASE_UID } from './auth.js';
const FREE_PATHS = 4;
const PRO_PATHS = 10000;
const ACTIVE_STATUSES = new Set(['active', 'trialing']);

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: {
    'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
  } });
}
function authenticationFailure(error, message) {
  // Do not return an identity or entitlement from an unverified token.
  return error instanceof IdentityKeysUnavailableError
    ? json({ error: 'Identity verification is temporarily unavailable. Please retry.' }, 503)
    : json({ error: message }, 401);
}
function priceIds(env) {
  const key = env.STRIPE_SECRET_KEY || '';
  const prefix = /^(sk|rk)_live_/.test(key) ? 'STRIPE_PRO_LIVE_' : /^(sk|rk)_test_/.test(key) ? 'STRIPE_PRO_' : null;
  if (!prefix) return { monthly: null, yearly: null };
  return { monthly: env[`${prefix}MONTHLY_PRICE_ID`], yearly: env[`${prefix}YEARLY_PRICE_ID`] };
}
function configured(env) { const prices = priceIds(env); return Boolean(env.STRIPE_SECRET_KEY && prices.monthly && prices.yearly); }
function testBilling(env) { return /^(sk|rk)_test_/.test(env.STRIPE_SECRET_KEY || ''); }
function hasProPrice(subscription, env, customerMetadata = {}) {
  const ids = new Set(Object.values(priceIds(env)).filter(id => typeof id === 'string' && id));
  if (subscription.items?.data?.some(item => ids.has(item.price?.id))) return true;
  // Checkout configuration can be incomplete while an existing plan is paid.
  // Only our server-written plan/identity metadata can identify a missing price;
  // an arbitrary subscription on the customer is not proof of Pro access.
  return !configured(env) && (subscription.metadata?.retirement_plan === 'pro' ||
    ['retirement_site_user_id', 'retirement_firebase_uid'].some(key =>
      typeof customerMetadata[key] === 'string' && customerMetadata[key] && subscription.metadata?.[key] === customerMetadata[key]));
}
function asPrincipal(value) { return typeof value === 'string' ? { kind: 'chatgpt', id: value, provider: 'chatgpt' } : value; }
function isOwnerAccount(user) {
  return (user?.kind === 'chatgpt' && user.id === OWNER_CHATGPT_USER_ID) ||
    (user?.kind === 'firebase' && user.provider === 'google' && (user.id === OWNER_FIREBASE_UID || user.verifiedEmail === OWNER_GOOGLE_EMAIL));
}
function metadataKey(value) { return value.kind === 'firebase' ? 'retirement_firebase_uid' : 'retirement_site_user_id'; }
function reference(value) { return value.kind === 'firebase' ? `firebase:${value.id}` : value.id; }
function allowedOrigin(request) { return request.headers.get('origin') === new URL(request.url).origin; }
function safeStripeUrl(value, host) {
  try { const url = new URL(value); return url.protocol === 'https:' && url.hostname === host ? url.href : null; } catch { return null; }
}
async function stripe(env, path, params = null, method = 'GET', idempotencyKey = null, allowMissing = false) {
  const url = new URL(`https://api.stripe.com/v1/${path}`);
  if (method === 'GET' && params) url.search = new URLSearchParams(params).toString();
  const headers = { Authorization: `Bearer ${env.STRIPE_SECRET_KEY}` };
  if (method !== 'GET') headers['Content-Type'] = 'application/x-www-form-urlencoded';
  if (idempotencyKey) headers['Idempotency-Key'] = idempotencyKey;
  const response = await (env.STRIPE_FETCH || fetch)(url, {
    method, headers, body: method === 'GET' ? undefined : new URLSearchParams(params).toString(),
    signal: AbortSignal.timeout(10000),
  });
  if (allowMissing && response.status === 404) return null;
  if (!response.ok) throw new Error('Stripe request failed');
  return response.json();
}
async function customers(env, value) {
  const user = asPrincipal(value);
  if (!user || !/^[A-Za-z0-9_-]{1,128}$/.test(user.id)) throw new Error('Invalid account ID');
  const key = metadataKey(user), query = `metadata['${key}']:'${user.id}'`;
  const found = [];
  let page;
  do {
    const result = await stripe(env, 'customers/search', { query, limit: '100', ...(page ? { page } : {}) });
    found.push(...(result.data || []).filter(c => c.metadata?.[key] === user.id && !c.deleted));
    page = result.has_more ? result.next_page : null;
  } while (page && found.length < 500);
  return found;
}
async function activeSubscription(env, customerId, customerMetadata = {}) {
  let cursor;
  const cursors=new Set();
  do {
    const result = await stripe(env, 'subscriptions', { customer: customerId, status: 'all', limit: '100', ...(cursor ? { starting_after: cursor } : {}) });
    if(!Array.isArray(result.data))throw new Error('Unexpected subscription response');
    const match=result.data.find(s => ACTIVE_STATUSES.has(s.status) && hasProPrice(s, env, customerMetadata));
    if(match)return match;
    cursor=result.has_more?result.data.at(-1)?.id:null;
    // An incomplete lookup must not be mistaken for permission to buy again.
    if(result.has_more&&(typeof cursor!=='string'||!cursor||cursors.has(cursor)))throw new Error('Unexpected subscription pagination');
    if(cursor)cursors.add(cursor);
  } while(cursor);
  return null;
}
async function access(env, value, sessionId = null, customerHint = null) {
  const user = asPrincipal(value);
  // A browser reference is a lookup hint, never proof of ownership or payment.
  async function verifiedCustomer(id) {
    if (typeof id !== 'string' || !/^cus_[A-Za-z0-9]{1,200}$/.test(id)) return null;
    const customer = await stripe(env, `customers/${id}`, null, 'GET', null, true);
    return customer && !customer.deleted && customer.metadata?.[metadataKey(user)] === user.id ? customer : null;
  }
  let knownCustomer = null;
  if (sessionId && /^cs_(?:test_|live_)?[A-Za-z0-9]{8,200}$/.test(sessionId)) {
    const session = await stripe(env, `checkout/sessions/${sessionId}`, null, 'GET', null, true);
    if (session?.client_reference_id === reference(user)) {
      knownCustomer = await verifiedCustomer(session.customer);
    }
  }
  knownCustomer ||= await verifiedCustomer(customerHint);
  if (knownCustomer && await activeSubscription(env, knownCustomer.id, knownCustomer.metadata)) return { pro: true, customerId: knownCustomer.id };
  const matches = await customers(env, user);
  for (const customer of matches) if (customer.id !== knownCustomer?.id && await activeSubscription(env, customer.id, customer.metadata)) return { pro: true, customerId: customer.id };
  const customer = knownCustomer || matches[0] || null;
  return { pro: false, customerId: customer?.id || null, customer };
}
async function status(request, env, user, isOwner = false) {
  if (request.method !== 'GET') return json({ error: 'Method not allowed' }, 405);
  if (isOwnerAccount(user) || (isOwner && user?.kind === 'chatgpt')) {
    const ownerAccess={ tier: 'pro', maxPaths: PRO_PATHS, signedIn: true, checkoutAvailable: false, accountKey: reference(user), accountProvider: user.provider, ownerAccess: true, testBilling: testBilling(env) };
    if (!env.STRIPE_SECRET_KEY) return json(ownerAccess);
    try {
      const result=await access(env,user,new URL(request.url).searchParams.get('session_id'),request.headers.get('x-retirement-customer'));
      return json({...ownerAccess,checkoutAvailable:testBilling(env)&&configured(env)&&!result.pro,billingPortalAvailable:Boolean(result.customerId),billingCustomerId:result.customerId});
    } catch {
      // Stripe cannot revoke complimentary access. Keep a portal retry action
      // and let that authenticated endpoint verify customer ownership afresh.
      return json({...ownerAccess,billingPortalAvailable:true,billingLookupUnavailable:true});
    }
  }
  const enabled = configured(env);
  if (!env.STRIPE_SECRET_KEY || !user) return json({ tier: 'free', maxPaths: FREE_PATHS, signedIn: Boolean(user), checkoutAvailable: enabled, accountKey: user ? reference(user) : null, accountProvider: user?.provider || null });
  try {
    const result = await access(env, user, new URL(request.url).searchParams.get('session_id'), request.headers.get('x-retirement-customer'));
    return json({ tier: result.pro ? 'pro' : 'free', maxPaths: result.pro ? PRO_PATHS : FREE_PATHS, signedIn: true, checkoutAvailable: enabled, billingPortalAvailable: Boolean(result.customerId), billingCustomerId: result.customerId, accountKey: reference(user), accountProvider: user.provider });
  } catch { return json({ signedIn: true, accountKey: reference(user), accountProvider: user.provider, error: 'Could not verify subscription. Please retry.' }, 502); }
}
async function checkoutHistory(env, customer, user) {
  const references = new Set([reference(user), customer.metadata?.retirement_site_user_id,
    customer.metadata?.retirement_firebase_uid ? `firebase:${customer.metadata.retirement_firebase_uid}` : null].filter(Boolean));
  let latest = null, cursor;
  const open = [];
  do {
    const page = await stripe(env, 'checkout/sessions', { customer: customer.id, limit: '100', ...(cursor ? { starting_after: cursor } : {}) });
    if (!Array.isArray(page.data)) throw new Error('Unexpected Checkout history');
    for (const session of page.data) {
      if (session.customer !== customer.id || session.mode !== 'subscription' ||
        !(references.has(session.client_reference_id) || session.metadata?.retirement_plan === 'pro')) continue;
      latest ||= session;
      if (session.status === 'open') open.push(session);
    }
    cursor = page.has_more ? page.data.at(-1)?.id : null;
    if (page.has_more && !cursor) throw new Error('Unexpected Checkout pagination');
  } while (cursor);
  return { latest, open };
}
async function checkoutPrice(env, session) {
  if (session.metadata?.retirement_price) return session.metadata.retirement_price;
  const items = await stripe(env, `checkout/sessions/${session.id}/line_items`, { limit: '1' });
  return items.data?.[0]?.price?.id;
}
async function checkout(request, env, user, isOwner = false) {
  if (request.method !== 'POST') return json({ error: 'Method not allowed' }, 405);
  if (!allowedOrigin(request)) return json({ error: 'Request origin rejected' }, 403);
  if (!user) return json({ error: 'Sign in before upgrading' }, 401);
  if (!testBilling(env) && (isOwnerAccount(user) || (isOwner && user.kind === 'chatgpt'))) return json({ error: 'This account already has Pro access' }, 409);
  if (!configured(env)) return json({ error: 'Paid access is not configured yet' }, 503);
  let interval;
  try { ({ interval } = await request.json()); } catch { return json({ error: 'Choose a billing interval' }, 400); }
  const priceId = priceIds(env)[interval];
  if (!priceId || !['monthly', 'yearly'].includes(interval)) return json({ error: 'Choose monthly or yearly billing' }, 400);
  try {
    const prior = await access(env, user, null, request.headers.get('x-retirement-customer'));
    if (prior.pro) return json({ error: 'This account already has Pro access' }, 409);
    const customer = prior.customer || await stripe(env, 'customers', { [`metadata[${metadataKey(user)}]`]: user.id }, 'POST', `retirement-${user.kind}-${user.id}`);
    const history = await checkoutHistory(env, customer, user);
    // Reuse a pending purchase; changing intervals expires the old payment link.
    // Fail closed if expiration races a payment already in progress.
    const reusable = history.open[0];
    if (reusable && await checkoutPrice(env, reusable) === priceId) {
      for (const extra of history.open.slice(1)) await stripe(env, `checkout/sessions/${extra.id}/expire`, {}, 'POST');
      const url = safeStripeUrl(reusable.url, 'checkout.stripe.com');
      if (!url) throw new Error('Unexpected Checkout URL');
      return json({ url });
    }
    for (const pending of history.open) await stripe(env, `checkout/sessions/${pending.id}/expire`, {}, 'POST');
    if (history.latest?.status === 'complete' && history.latest.payment_status === 'unpaid') {
      const id = history.latest.subscription;
      const subscription = typeof id === 'string' && /^sub_[A-Za-z0-9]+$/.test(id) ? await stripe(env, `subscriptions/${id}`) : null;
      if (!['canceled', 'incomplete_expired'].includes(subscription?.status)) {
        return json({ error: 'A payment is still being confirmed. Check your plan before starting another checkout.' }, 409);
      }
    }
    const origin = new URL(request.url).origin;
    // The generation comes from Stripe history, rather than a clock bucket or
    // browser hint. Concurrent servers use one key even for different intervals.
    const session = await stripe(env, 'checkout/sessions', {
      mode: 'subscription', customer: customer.id, client_reference_id: reference(user),
      'line_items[0][price]': priceId, 'line_items[0][quantity]': '1',
      [`subscription_data[metadata][${metadataKey(user)}]`]: user.id,
      'subscription_data[metadata][retirement_plan]': 'pro',
      'metadata[retirement_plan]': 'pro', 'metadata[retirement_price]': priceId,
      success_url: `${origin}/?checkout=success&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${origin}/?checkout=canceled`,
    }, 'POST', `retirement-checkout-${customer.id}-${history.latest?.id || 'initial'}`);
    const url = safeStripeUrl(session.url, 'checkout.stripe.com');
    if (!url) throw new Error('Unexpected Checkout URL');
    return json({ url });
  } catch { return json({ error: 'Could not start Stripe Checkout. Please try again.' }, 502); }
}
async function portal(request, env, user) {
  if (request.method !== 'POST') return json({ error: 'Method not allowed' }, 405);
  if (!allowedOrigin(request)) return json({ error: 'Request origin rejected' }, 403);
  if (!user) return json({ error: 'Sign in to manage billing' }, 401);
  if (!env.STRIPE_SECRET_KEY) return json({ error: 'Billing is not configured yet' }, 503);
  try {
    const account = await access(env, user, null, request.headers.get('x-retirement-customer'));
    if (!account.customerId) return json({ error: 'No billing account was found' }, 404);
    const session = await stripe(env, 'billing_portal/sessions', { customer: account.customerId, return_url: `${new URL(request.url).origin}/` }, 'POST');
    const url = safeStripeUrl(session.url, 'billing.stripe.com');
    if (!url) throw new Error('Unexpected portal URL');
    return json({ url });
  } catch { return json({ error: 'Could not open billing. Please try again.' }, 502); }
}
async function linkAccounts(request, env) {
  if (request.method !== 'POST') return json({ error: 'Method not allowed' }, 405);
  if (!allowedOrigin(request)) return json({ error: 'Request origin rejected' }, 403);
  if (!configured(env)) return json({ error: 'Billing is not configured yet' }, 503);
  const chatgpt = chatgptPrincipal(request);
  if (!chatgpt) return json({ error: 'Sign in with ChatGPT before linking accounts' }, 401);
  let firebase;
  try { firebase = await firebasePrincipal(request, env); } catch (error) { return authenticationFailure(error, 'Sign in with Google again'); }
  if (!firebase) return json({ error: 'Sign in with Google before linking accounts' }, 401);
  try {
    const [chatgptMatches, firebaseMatches] = await Promise.all([customers(env, chatgpt), customers(env, firebase)]);
    // Stripe Search may lag behind Checkout or a metadata change. Include
    // direct hints only after verifying ownership by either authenticated user.
    const hints=[request.headers.get('x-retirement-customer'),...(request.headers.get('x-retirement-link-customers')||'').split(',')];
    const direct=[];
    for(const id of [...new Set(hints)].filter(id=>typeof id==='string'&&/^cus_[A-Za-z0-9]{1,200}$/.test(id)).slice(0,5)){
      const customer=await stripe(env,`customers/${id}`,null,'GET',null,true);
      if(customer&&!customer.deleted&&(customer.metadata?.[metadataKey(chatgpt)]===chatgpt.id||customer.metadata?.[metadataKey(firebase)]===firebase.id))direct.push(customer);
    }
    const matches = [...new Map([...chatgptMatches, ...firebaseMatches,...direct].map(c => [c.id, c])).values()];
    const paid = [];
    // Search order does not identify the customer's subscription. Check every
    // record, including duplicates left behind by abandoned checkouts.
    for (const customer of matches) if (await activeSubscription(env, customer.id, customer.metadata)) paid.push(customer);
    if (paid.length > 1) return json({ error: 'Multiple billing accounts have subscriptions. Contact support before linking them.' }, 409);
    let chosen = paid[0] || matches[0];
    if (!chosen) {
      chosen = await stripe(env, 'customers', {
        'metadata[retirement_site_user_id]': chatgpt.id,
        'metadata[retirement_firebase_uid]': firebase.id,
      }, 'POST', `retirement-link-${chatgpt.id}-${firebase.id}`);
    } else {
      await stripe(env, `customers/${chosen.id}`, {
        'metadata[retirement_site_user_id]': chatgpt.id,
        'metadata[retirement_firebase_uid]': firebase.id,
      }, 'POST');
    }
    // Remove only these verified identities from the other, unpaid records.
    for (const customer of matches) {
      if (customer.id === chosen.id) continue;
      const params = {};
      if (customer.metadata?.retirement_site_user_id === chatgpt.id) params['metadata[retirement_site_user_id]'] = '';
      if (customer.metadata?.retirement_firebase_uid === firebase.id) params['metadata[retirement_firebase_uid]'] = '';
      if (Object.keys(params).length) await stripe(env, `customers/${customer.id}`, params, 'POST');
    }
    return json({ linked: true, billingCustomerId: chosen.id });
  } catch { return json({ error: 'Could not link accounts. Please try again.' }, 502); }
}
export async function billing(request, env, pathname, isOwner = false) {
  let user;
  try { user = await requestPrincipal(request, env); } catch (error) { return authenticationFailure(error, 'Sign in again to verify your account'); }
  if (/^(sk|rk)_test_/.test(env.STRIPE_SECRET_KEY || '') && !isOwnerAccount(user) && !(isOwner && user?.kind === 'chatgpt')) {
    if (pathname === '/api/billing/status') return json({ tier: 'free', maxPaths: FREE_PATHS, signedIn: Boolean(user), checkoutAvailable: false, accountProvider: user?.provider || null });
    return json({ error: 'Test billing is available only to the site owner' }, 403);
  }
  if (pathname === '/api/billing/link') return linkAccounts(request, env);
  if (pathname === '/api/billing/status') return status(request, env, user, isOwner);
  if (pathname === '/api/billing/checkout') return checkout(request, env, user, isOwner);
  if (pathname === '/api/billing/portal') return portal(request, env, user);
  return json({ error: 'Not found' }, 404);
}
export { access, activeSubscription, customers, safeStripeUrl };
