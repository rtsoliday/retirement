import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../worker/index.js';

const origin = 'https://retirementforecast.us';
const project = 'retirement-auth-test';
const firebaseEnv = {
  ASSETS: { fetch: async () => new Response('planner') },
  FIREBASE_API_KEY: 'test-public-key', FIREBASE_AUTH_DOMAIN: `${project}.firebaseapp.com`,
  FIREBASE_PROJECT_ID: project, FIREBASE_APP_ID: '1:123:web:test',
  FIREBASE_GOOGLE_ENABLED: 'true',
  STRIPE_SECRET_KEY: 'rk_live_test',
  STRIPE_PRO_LIVE_MONTHLY_PRICE_ID: 'price_monthly', STRIPE_PRO_LIVE_YEARLY_PRICE_ID: 'price_yearly',
};
const encoder = new TextEncoder();
const b64 = bytes => Buffer.from(bytes).toString('base64url');
const keyPair = await crypto.subtle.generateKey({ name: 'RSASSA-PKCS1-v1_5', modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' }, true, ['sign', 'verify']);
const jwk = { ...await crypto.subtle.exportKey('jwk', keyPair.publicKey), kid: 'test-kid', alg: 'RS256' };
firebaseEnv.FIREBASE_JWKS_FETCH = async () => Response.json({ keys: [jwk] });
async function token(overrides = {}) {
  const now = Math.floor(Date.now() / 1000);
  const header = b64(JSON.stringify({ alg: 'RS256', kid: 'test-kid', typ: 'JWT' }));
  const claims = b64(JSON.stringify({ aud: project, iss: `https://securetoken.google.com/${project}`, sub: 'firebase-user', exp: now + 3600, iat: now, auth_time: now, firebase: { sign_in_provider: 'google.com' }, ...overrides }));
  const signature = await crypto.subtle.sign('RSASSA-PKCS1-v1_5', keyPair.privateKey, encoder.encode(`${header}.${claims}`));
  return `${header}.${claims}.${b64(signature)}`;
}
function request(path, { user, bearer, method = 'GET', siteOrigin = origin, body } = {}) {
  return new Request(origin + path, { method, headers: { ...(user ? { 'oai-authenticated-user-id': user } : {}), ...(bearer ? { Authorization: `Bearer ${bearer}` } : {}), ...(method === 'POST' ? { Origin: siteOrigin } : {}), ...(body ? { 'Content-Type': 'application/json' } : {}) }, ...(body ? { body: JSON.stringify(body) } : {}) });
}
function stripeFixture(initial = []) {
  const customers = new Map(initial.map(c => [c.id, structuredClone(c)]));
  const writes = [];
  const fetch = async (url, init) => {
    const parsed = new URL(url), path = parsed.pathname, values = new URLSearchParams(init.body || '');
    writes.push({ path, method: init.method, values, search: parsed.searchParams });
    if (path === '/v1/customers/search') {
      const query = parsed.searchParams.get('query') || '';
      const match = /metadata\['([^']+)'\]:'([^']+)'/.exec(query);
      return Response.json({ data: [...customers.values()].filter(c => c.metadata?.[match?.[1]] === match?.[2]), has_more: false });
    }
    if (path === '/v1/subscriptions') {
      const customer = customers.get(parsed.searchParams.get('customer'));
      return Response.json({ data: customer?.paid ? [{ id: 'sub_paid', status: 'active', items: { data: [{ price: { id: 'price_monthly' } }] } }] : [] });
    }
    if (path === '/v1/customers' && init.method === 'POST') {
      const customer = { id: `cus_${customers.size + 1}`, metadata: { retirement_site_user_id: values.get('metadata[retirement_site_user_id]') || '', retirement_firebase_uid: values.get('metadata[retirement_firebase_uid]') || '' } };
      customers.set(customer.id, customer); return Response.json(customer);
    }
    if (path.startsWith('/v1/customers/') && init.method === 'POST') {
      const customer = customers.get(path.split('/').at(-1));
      if (!customer) return Response.json({}, { status: 404 });
      for (const [key, value] of values) { const m = /^metadata\[(.+)\]$/.exec(key); if (m) customer.metadata[m[1]] = value; }
      return Response.json(customer);
    }
    if (path.startsWith('/v1/customers/') && init.method === 'GET') {
      const customer=customers.get(path.split('/').at(-1));
      return Response.json(customer||{}, {status:customer?200:404});
    }
    if (path === '/v1/checkout/sessions' && init.method === 'GET') return Response.json({data:[],has_more:false});
    if (path === '/v1/checkout/sessions' && init.method === 'POST') return Response.json({ url: 'https://checkout.stripe.com/c/pay/example' });
    if (path === '/v1/billing_portal/sessions' && init.method === 'POST') return Response.json({ url: 'https://billing.stripe.com/p/session/example' });
    return Response.json({}, { status: 404 });
  };
  return { env: { ...firebaseEnv, STRIPE_FETCH: fetch }, customers, writes };
}

test('public sign-in config is only exposed when all Firebase settings exist', async () => {
  const disabled = await (await worker.fetch(request('/api/auth/config'), { ASSETS: firebaseEnv.ASSETS })).json();
  assert.equal(disabled.configured, false); assert.equal(disabled.firebase, null);
  const enabled = await (await worker.fetch(request('/api/auth/config', { user: 'chatgpt-user' }), firebaseEnv)).json();
  assert.equal(enabled.configured, true); assert.equal(enabled.chatgptSignedIn, true);
  assert.deepEqual(enabled.providers, { google: true });
  assert.equal(enabled.firebase.projectId, project); assert.equal(enabled.firebase.apiKey, 'test-public-key');
  assert.equal(JSON.stringify(enabled).includes('rk_live_'), false);
});

test('verified Google accounts can reach their own Pro subscription', async () => {
  const fixture = stripeFixture([{ id: 'cus_paid', metadata: { retirement_firebase_uid: 'firebase-user' }, paid: true }]);
  const response = await worker.fetch(request('/api/billing/status', { bearer: await token() }), fixture.env);
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.tier, 'pro'); assert.equal(result.maxPaths, 10000);
  assert.equal(result.accountProvider, 'google');
  assert.match(fixture.writes[0].search.get('query'), /retirement_firebase_uid/);
});

test('only the signed, verified owner Google identity receives complimentary Pro', async () => {
  const fixture = stripeFixture();
  const ownerToken = await token({ email: 'rtsoliday@gmail.com', email_verified: true });
  const owner = await (await worker.fetch(request('/api/billing/status', { bearer: ownerToken }), fixture.env)).json();
  assert.equal(owner.tier, 'pro'); assert.equal(owner.maxPaths, 10000);
  assert.equal(owner.ownerAccess, true); assert.equal(owner.checkoutAvailable, false);
  assert.equal(fixture.writes.filter(call=>call.method==='POST').length, 0);
  const ownerByUid = await (await worker.fetch(request('/api/billing/status', { bearer: await token({ sub: 'xYPnJEpGrHfTJmBtFUnXlAQSdzU2' }) }), fixture.env)).json();
  assert.equal(ownerByUid.tier, 'pro'); assert.equal(ownerByUid.ownerAccess, true);
  const testMode = await (await worker.fetch(request('/api/billing/status', { bearer: ownerToken }), { ...fixture.env, STRIPE_SECRET_KEY: 'rk_test_fake' })).json();
  assert.equal(testMode.tier, 'pro');
  const checkout = await worker.fetch(request('/api/billing/checkout', { bearer: ownerToken, method: 'POST', body: { interval: 'monthly' } }), fixture.env);
  assert.equal(checkout.status, 409);
  assert.equal(fixture.writes.filter(call=>call.method==='POST').length, 0);
  const otherGoogle = await (await worker.fetch(request('/api/billing/status', { user: '028696a7-7846-4822-a4c1-67026aa2383f', bearer: await token({ email: 'other@example.com', email_verified: true }) }), fixture.env)).json();
  assert.equal(otherGoogle.tier, 'free');
  for (const claims of [
    { email: 'rtsoliday@gmail.com', email_verified: false },
    { email: 'other@example.com', email_verified: true },
  ]) {
    const visitor = await (await worker.fetch(request('/api/billing/status', { bearer: await token(claims) }), fixture.env)).json();
    assert.equal(visitor.tier, 'free'); assert.equal(visitor.maxPaths, 4);
  }
});

test('verified Google owner can use sandbox Checkout but other Google identities cannot borrow ChatGPT owner access',async()=>{
  const fixture=stripeFixture([{id:'cus_owner',metadata:{retirement_firebase_uid:'xYPnJEpGrHfTJmBtFUnXlAQSdzU2'}}]);
  const env={...fixture.env,STRIPE_SECRET_KEY:'rk_test_fake',STRIPE_PRO_MONTHLY_PRICE_ID:'price_test_monthly',STRIPE_PRO_YEARLY_PRICE_ID:'price_test_yearly'};
  const bearer=await token({sub:'xYPnJEpGrHfTJmBtFUnXlAQSdzU2'});
  const access=await (await worker.fetch(request('/api/billing/status',{bearer}),env)).json();
  assert.equal(access.tier,'pro');assert.equal(access.checkoutAvailable,true);assert.equal(access.testBilling,true);
  assert.equal((await worker.fetch(request('/api/billing/checkout',{bearer,method:'POST',body:{interval:'monthly'}}),env)).status,200);
  assert.equal(fixture.writes.find(call=>call.path==='/v1/checkout/sessions' && call.method==='POST').values.get('line_items[0][price]'),'price_test_monthly');
  assert.equal((await worker.fetch(request('/api/billing/checkout',{user:'028696a7-7846-4822-a4c1-67026aa2383f',bearer:await token(),method:'POST',body:{interval:'monthly'}}),env)).status,403);
});

test('forged, expired, wrong-project and unsupported tokens never fall back to ChatGPT', async () => {
  const fixture = stripeFixture([{ id: 'cus_paid', metadata: { retirement_site_user_id: 'chatgpt-user' }, paid: true }]);
  const good = await token();
  const invalid = [good.slice(0, -5) + 'abcde', await token({ exp: Math.floor(Date.now() / 1000) - 1 }), await token({ aud: 'other-project' }), await token({ firebase: { sign_in_provider: 'anonymous' } }), await token({ firebase: { sign_in_provider: 'apple.com' } })];
  for (const bearer of invalid) {
    const response = await worker.fetch(request('/api/billing/status', { user: 'chatgpt-user', bearer }), fixture.env);
    assert.equal(response.status, 401);
  }
  assert.equal(fixture.writes.length, 0);
});

test('Google key-service failures return 503 without asserting identity or calling Stripe',async()=>{
  const bearer=await token();
  const failures=[
    async()=>{throw new TypeError('Network unreachable');},
    async()=>{throw new DOMException('Timed out','TimeoutError');},
    async()=>new Response('Unavailable',{status:503}),
    async()=>new Response('invalid JSON'),
    async()=>Response.json({keys:null}),
    async()=>Response.json({keys:[]}),
    async()=>Response.json({keys:[{kty:'RSA',alg:'RS256',kid:'test-kid'}]}),
  ];
  for(const unavailable of failures)for(const route of ['status','checkout','portal','link']){
    const fixture=stripeFixture();fixture.env.FIREBASE_JWKS_FETCH=unavailable;
    const response=await worker.fetch(request(`/api/billing/${route}`,{user:'chatgpt-user',bearer,method:route==='status'?'GET':'POST',...(route==='checkout'?{body:{interval:'monthly'}}:{})}),fixture.env);
    assert.equal(response.status,503,route);const data=await response.json();
    assert.deepEqual(Object.keys(data),['error']);assert.match(data.error,/temporarily unavailable/);assert.doesNotMatch(data.error,/Sign in again/);
    assert.equal(response.headers.get('cache-control'),'no-store');assert.equal(fixture.writes.length,0);
  }
});

test('a key-service outage during the second account-link verification also returns 503',async()=>{
  const fixture=stripeFixture();let checks=0;
  fixture.env.FIREBASE_JWKS_FETCH=async()=>++checks===1?Response.json({keys:[jwk]}):new Response('Unavailable',{status:503});
  const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'}),fixture.env);
  assert.equal(checks,2);assert.equal(response.status,503);assert.equal(fixture.writes.length,0);
});

test('invalid claims and unknown signing keys remain authentication failures',async()=>{
  const fixture=stripeFixture();let lookups=0;
  fixture.env.FIREBASE_JWKS_FETCH=async()=>{lookups++;throw Error('Unavailable');};
  for(const bearer of [await token({exp:1}),await token({aud:'wrong-project'}),'not.a.token']){
    const response=await worker.fetch(request('/api/billing/status',{user:'chatgpt-user',bearer}),fixture.env);assert.equal(response.status,401);
  }
  assert.equal(lookups,0);
  fixture.env.FIREBASE_JWKS_FETCH=async()=>Response.json({keys:[{...jwk,kid:'different-key'}]});
  const response=await worker.fetch(request('/api/billing/status',{user:'chatgpt-user',bearer:await token()}),fixture.env);
  assert.equal(response.status,401);assert.equal(fixture.writes.length,0);
});

test('ChatGPT-only billing continues without Google keys during a key-service outage',async()=>{
  const fixture=stripeFixture([{id:'cus_paid',metadata:{retirement_site_user_id:'chatgpt-user'},paid:true}]);let lookups=0;
  fixture.env.FIREBASE_JWKS_FETCH=async()=>{lookups++;throw Error('Unavailable');};
  const response=await worker.fetch(request('/api/billing/status',{user:'chatgpt-user'}),fixture.env);
  assert.equal(response.status,200);assert.equal((await response.json()).tier,'pro');assert.equal(lookups,0);
});

test('account linking requires both identities and same origin, then shares one paid customer', async () => {
  const fixture = stripeFixture([{ id: 'cus_paid', metadata: { retirement_site_user_id: 'chatgpt-user' }, paid: true }]);
  const bearer = await token();
  assert.equal((await worker.fetch(request('/api/billing/link', { bearer, method: 'POST' }), fixture.env)).status, 401);
  assert.equal((await worker.fetch(request('/api/billing/link', { user: 'chatgpt-user', method: 'POST' }), fixture.env)).status, 401);
  assert.equal((await worker.fetch(request('/api/billing/link', { user: 'chatgpt-user', bearer, method: 'POST', siteOrigin: 'https://evil.test' }), fixture.env)).status, 403);
  const response = await worker.fetch(request('/api/billing/link', { user: 'chatgpt-user', bearer, method: 'POST' }), fixture.env);
  assert.equal(response.status, 200); assert.equal(fixture.customers.get('cus_paid').metadata.retirement_firebase_uid, 'firebase-user');
  const firebaseAccess = await (await worker.fetch(request('/api/billing/status', { bearer }), fixture.env)).json();
  assert.equal(firebaseAccess.tier, 'pro');
});

test('two paid accounts are not silently combined', async () => {
  const fixture = stripeFixture([
    { id: 'cus_chatgpt', metadata: { retirement_site_user_id: 'chatgpt-user' }, paid: true },
    { id: 'cus_firebase', metadata: { retirement_firebase_uid: 'firebase-user' }, paid: true },
  ]);
  const response = await worker.fetch(request('/api/billing/link', { user: 'chatgpt-user', bearer: await token(), method: 'POST' }), fixture.env);
  assert.equal(response.status, 409);
  assert.equal(fixture.customers.get('cus_chatgpt').metadata.retirement_firebase_uid, undefined);
});

test('linking selects the paid customer behind abandoned checkouts and clears duplicate identity records',async()=>{
  for(const provider of ['chatgpt','firebase']){
    const fixture=stripeFixture([
      {id:'cus_draft1',metadata:{retirement_site_user_id:'chatgpt-user'},paid:false},
      {id:'cus_draft2',metadata:{retirement_site_user_id:'chatgpt-user',retirement_firebase_uid:'firebase-user'},paid:false},
      {id:'cus_draft3',metadata:{retirement_firebase_uid:'firebase-user'},paid:false},
      {id:'cus_paid',metadata:provider==='chatgpt'?{retirement_site_user_id:'chatgpt-user'}:{retirement_firebase_uid:'firebase-user'},paid:true},
    ]);
    const bearer=await token();
    const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'}),fixture.env);
    assert.equal(response.status,200);assert.equal((await response.json()).billingCustomerId,'cus_paid');
    for(const c of fixture.customers.values())if(c.id!=='cus_paid'){
      assert.notEqual(c.metadata.retirement_site_user_id,'chatgpt-user');
      assert.notEqual(c.metadata.retirement_firebase_uid,'firebase-user');
    }
    for(const identity of [{user:'chatgpt-user'},{bearer}]){
      const status=await (await worker.fetch(request('/api/billing/status',identity),fixture.env)).json();
      assert.equal(status.tier,'pro');assert.equal(status.billingCustomerId,'cus_paid');
    }
  }
});

test('linking rejects hidden paid-customer conflicts before modifying any metadata',async()=>{
  for(const secondMetadata of [{retirement_firebase_uid:'firebase-user'},{retirement_site_user_id:'chatgpt-user'}]){
    const fixture=stripeFixture([
      {id:'cus_draft',metadata:{retirement_site_user_id:'chatgpt-user',retirement_firebase_uid:'firebase-user'},paid:false},
      {id:'cus_paid1',metadata:{retirement_site_user_id:'chatgpt-user'},paid:true},
      {id:'cus_paid2',metadata:secondMetadata,paid:true},
    ]);
    const before=JSON.stringify([...fixture.customers.values()]);
    const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'}),fixture.env);
    assert.equal(response.status,409);assert.equal(JSON.stringify([...fixture.customers.values()]),before);
    assert.equal(fixture.writes.some(w=>w.method==='POST'),false);
  }
});

test('linking rejects paid-customer conflicts found on later subscription pages',async()=>{
  const fixture=stripeFixture([
    {id:'cus_paid1',metadata:{retirement_site_user_id:'chatgpt-user'},paid:true},
    {id:'cus_paid2',metadata:{retirement_firebase_uid:'firebase-user'},paid:true}
  ]);
  const upstream=fixture.env.STRIPE_FETCH;
  let laterPages=0;
  fixture.env.STRIPE_FETCH=async(url,init)=>{
    const parsed=new URL(url);
    if(parsed.pathname==='/v1/subscriptions'&&parsed.searchParams.get('customer')==='cus_paid2'){
      if(!parsed.searchParams.has('starting_after'))return Response.json({data:[{id:'sub_old',status:'canceled'}],has_more:true});
      assert.equal(parsed.searchParams.get('starting_after'),'sub_old');laterPages++;
    }
    return upstream(url,init);
  };
  const before=JSON.stringify([...fixture.customers.values()]);
  const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'}),fixture.env);
  assert.equal(response.status,409);assert.equal(laterPages,1);
  assert.equal(JSON.stringify([...fixture.customers.values()]),before);
  assert.equal(fixture.writes.some(w=>w.method==='POST'),false);
});

test('linked customer reference restores either sign-in without waiting for search updates',async()=>{
  for(const paid of ['chatgpt','firebase']){
    const fixture=stripeFixture([
      {id:'cus_chatgpt',metadata:{retirement_site_user_id:'chatgpt-user'},paid:paid==='chatgpt'},
      {id:'cus_firebase',metadata:{retirement_firebase_uid:'firebase-user'},paid:paid==='firebase'},
    ]);
    const bearer=await token();
    const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'}),fixture.env);
    assert.equal(response.status,200);const linked=await response.json();assert.equal(linked.billingCustomerId,`cus_${paid}`);
    const fetch=fixture.env.STRIPE_FETCH;
    fixture.env.STRIPE_FETCH=async(url,init)=>{assert.notEqual(new URL(url).pathname,'/v1/customers/search','must not search after linking');return fetch(url,init);};
    for(const identity of [{user:'chatgpt-user'},{bearer}]){
      const statusRequest=request('/api/billing/status',identity);statusRequest.headers.set('x-retirement-customer',linked.billingCustomerId);
      const status=await worker.fetch(statusRequest,fixture.env);assert.equal(status.status,200);assert.equal((await status.json()).tier,'pro');
    }
  }
});

test('Firebase checkout uses its verified UID, without borrowing ChatGPT entitlement', async () => {
  const fixture = stripeFixture();
  const response = await worker.fetch(request('/api/billing/checkout', { bearer: await token(), method: 'POST', body: { interval: 'yearly' } }), fixture.env);
  assert.equal(response.status, 200);
  const checkout = fixture.writes.find(w => w.path === '/v1/checkout/sessions' && w.method === 'POST');
  assert.equal(checkout.values.get('client_reference_id'), 'firebase:firebase-user');
  assert.equal(checkout.values.get('line_items[0][price]'), 'price_yearly');
  assert.equal(fixture.customers.get('cus_1').metadata.retirement_firebase_uid, 'firebase-user');
});


test('linking includes verified direct customer hints when Stripe Search has not indexed a paid customer',async()=>{
  for(const provider of ['chatgpt','firebase']){
    const fixture=stripeFixture([{id:'cus_paid',metadata:provider==='chatgpt'?{retirement_site_user_id:'chatgpt-user'}:{retirement_firebase_uid:'firebase-user'},paid:true}]);
    const upstream=fixture.env.STRIPE_FETCH;
    fixture.env.STRIPE_FETCH=async(url,init)=>new URL(url).pathname==='/v1/customers/search'?Response.json({data:[],has_more:false}):upstream(url,init);
    const bearer=await token(),req=request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'});
    req.headers.set(provider==='chatgpt'?'x-retirement-link-customers':'x-retirement-customer','cus_paid');
    const response=await worker.fetch(req,fixture.env);assert.equal(response.status,200);assert.equal((await response.json()).billingCustomerId,'cus_paid');
    assert.equal(fixture.customers.size,1);
    for(const identity of [{user:'chatgpt-user'},{bearer}]){
      const statusRequest=request('/api/billing/status',identity);statusRequest.headers.set('x-retirement-customer','cus_paid');
      assert.equal((await(await worker.fetch(statusRequest,fixture.env)).json()).tier,'pro');
    }
  }
});

test('linking ignores foreign customer hints and checks paid conflicts among owned hints',async()=>{
  const foreign=stripeFixture([{id:'cus_foreign',metadata:{retirement_site_user_id:'another-user'},paid:true}]);
  const before=JSON.stringify(foreign.customers.get('cus_foreign'));
  const req=request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'});req.headers.set('x-retirement-link-customers','cus_foreign');
  assert.equal((await worker.fetch(req,foreign.env)).status,200);assert.equal(JSON.stringify(foreign.customers.get('cus_foreign')),before);
  const fixture=stripeFixture([{id:'cus_one',metadata:{retirement_site_user_id:'chatgpt-user'},paid:true},{id:'cus_two',metadata:{retirement_firebase_uid:'firebase-user'},paid:true}]);
  const upstream=fixture.env.STRIPE_FETCH;
  fixture.env.STRIPE_FETCH=async(url,init)=>new URL(url).pathname==='/v1/customers/search'?Response.json({data:[],has_more:false}):upstream(url,init);
  const conflict=request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'});conflict.headers.set('x-retirement-link-customers','cus_one,cus_two');
  assert.equal((await worker.fetch(conflict,fixture.env)).status,409);assert.equal(fixture.writes.some(w=>w.method==='POST'),false);
});
