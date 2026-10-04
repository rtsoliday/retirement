import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../worker/index.js';
import { identityKeyCache, IdentityKeysUnavailableError } from '../worker/auth.js';

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
  const writes = [],keys=new Map();
  const handle = async (url, init) => {
    const parsed = new URL(url), path = parsed.pathname, values = new URLSearchParams(init.body || '');
    writes.push({ path, method: init.method, values, search: parsed.searchParams });
    if (path === '/v1/customers/search') {
      const query = parsed.searchParams.get('query') || '';
      const match = /metadata\['([^']+)'\]:'([^']+)'/.exec(query);
      return Response.json({ data: [...customers.values()].filter(c => c.metadata?.[match?.[1]] === match?.[2]), has_more: false });
    }
    if (path === '/v1/customers' && init.method === 'GET') return Response.json({ data: [...customers.values()], has_more: false });
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
  const fetch=async(url,init)=>{
    const key=init.method==='POST'?init.headers['Idempotency-Key']:null;
    if(!key)return handle(url,init);
    const path=new URL(url).pathname,previous=keys.get(key);
    if(previous){
      if(previous.body!==init.body||previous.path!==path)return Response.json({error:{type:'idempotency_error'}},{status:400});
      return (await previous.response).clone();
    }
    const entry={path,body:init.body};keys.set(key,entry);
    entry.response=handle(url,init);return (await entry.response).clone();
  };
  return { env: { ...firebaseEnv, STRIPE_FETCH: fetch }, customers, writes, keys };
}

function trackCheckouts(fixture,{beforeCreate=async()=>{}}={}){
  const upstream=fixture.env.STRIPE_FETCH,sessions=[],keys=new Map();let nextSession=0;
  fixture.env.STRIPE_FETCH=async(url,init)=>{
    const parsed=new URL(url),path=parsed.pathname,params=new URLSearchParams(init.body||'');
    if(path==='/v1/checkout/sessions'&&init.method==='GET')return Response.json({data:sessions.filter(session=>session.customer===parsed.searchParams.get('customer')).toReversed(),has_more:false});
    const expiring=/^\/v1\/checkout\/sessions\/([^/]+)\/expire$/.exec(path);
    if(path==='/v1/checkout/sessions'&&init.method==='POST'||expiring){
      const key=init.headers['Idempotency-Key'];assert.ok(key);const previous=keys.get(key);
      if(previous)return previous.body===init.body&&previous.path===path?(await previous.response).clone():Response.json({error:{type:'idempotency_error'}},{status:400});
      const response=(async()=>{
        if(expiring){
          const session=sessions.find(session=>session.id===expiring[1]);
          if(session?.status!=='open')return Response.json({error:'not open'},{status:400});
          session.status='expired';return Response.json(session);
        }
        const number=++nextSession;
        await beforeCreate(params,number);
        const session={id:'cs_race'+number,customer:params.get('customer'),mode:'subscription',status:'open',client_reference_id:params.get('client_reference_id'),metadata:{retirement_plan:'pro',retirement_price:params.get('metadata[retirement_price]')},url:'https://checkout.stripe.com/c/pay/race'+number};
        sessions.push(session);return Response.json(session);
      })();
      keys.set(key,{path,body:init.body,response});return (await response).clone();
    }
    return upstream(url,init);
  };
  return {sessions,keys};
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
    assert.equal(visitor.tier, 'free'); assert.equal(visitor.maxPaths, 100);
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

test('unknown signing-key IDs refetch Google keys at most once a minute',async()=>{
  let downloads=0,time=10000000,available=true;
  const keyFor=identityKeyCache(async()=>{downloads++;return available?new Response(JSON.stringify({keys:[jwk]}),{headers:{'Content-Type':'application/json','Cache-Control':'public, max-age=3600'}}):new Response('Unavailable',{status:503});},()=>time);
  assert.ok(await keyFor('test-kid'));assert.ok(await keyFor('test-kid'));assert.equal(downloads,1);
  // A forged key ID cannot force another download while the key set is fresh.
  await assert.rejects(keyFor('forged'),/Unrecognized identity key/);assert.equal(downloads,1);
  // After a minute, concurrent unknown IDs share one download for a possible key rotation.
  time+=60000;
  await assert.rejects(Promise.all([keyFor('forged-1'),keyFor('forged-2')]),/Unrecognized identity key/);assert.equal(downloads,2);
  await assert.rejects(keyFor('forged-3'),/Unrecognized identity key/);assert.equal(downloads,2);
  // An expired set reloads; a failed download is reported and retried on the next request.
  time+=3600000;available=false;
  await assert.rejects(keyFor('test-kid'),IdentityKeysUnavailableError);assert.equal(downloads,3);
  available=true;assert.ok(await keyFor('test-kid'));assert.equal(downloads,4);
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

test('linking selects the paid customer and keeps all verified records accessible to both identities',async()=>{
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
      assert.equal(c.metadata.retirement_site_user_id,'chatgpt-user');
      assert.equal(c.metadata.retirement_firebase_uid,'firebase-user');
    }
    for(const identity of [{user:'chatgpt-user'},{bearer}]){
      const status=await (await worker.fetch(request('/api/billing/status',identity),fixture.env)).json();
      assert.equal(status.tier,'pro');assert.equal(status.billingCustomerId,'cus_paid');
    }
  }
});

test('linking expires existing Checkouts while retaining their customer ownership',async()=>{
  const fixture=stripeFixture([{id:'cus_first',metadata:{retirement_site_user_id:'chatgpt-user'}},{id:'cus_pending',metadata:{retirement_firebase_uid:'firebase-user'}}]);
  const original=fixture.env.STRIPE_FETCH,pending={id:'cs_pending',customer:'cus_pending',mode:'subscription',status:'open',client_reference_id:'firebase:firebase-user'};
  const operations=[];
  fixture.env.STRIPE_FETCH=async(url,init)=>{
    const parsed=new URL(url),path=parsed.pathname;operations.push(init.method+' '+path);
    if(path==='/v1/checkout/sessions'&&init.method==='GET')return Response.json({data:parsed.searchParams.get('customer')==='cus_pending'?[pending]:[],has_more:false});
    if(path==='/v1/checkout/sessions/cs_pending/expire'){pending.status='expired';return Response.json(pending);}
    return original(url,init);
  };
  const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'}),fixture.env);
  assert.equal(response.status,200);assert.equal(pending.status,'expired');
  assert.equal(fixture.customers.get('cus_pending').metadata.retirement_firebase_uid,'firebase-user');
  assert.equal(fixture.customers.get('cus_pending').metadata.retirement_site_user_id,'chatgpt-user');
  const expiration=operations.indexOf('POST /v1/checkout/sessions/cs_pending/expire');
  assert.ok(expiration>=0&&expiration<operations.indexOf('POST /v1/customers/cus_pending'));
});

test('a payment racing Checkout expiration cannot detach either account',async()=>{
  const fixture=stripeFixture([{id:'cus_first',metadata:{retirement_site_user_id:'chatgpt-user'}},{id:'cus_pending',metadata:{retirement_firebase_uid:'firebase-user'}}]);
  const before=structuredClone([...fixture.customers.values()]),original=fixture.env.STRIPE_FETCH;
  fixture.env.STRIPE_FETCH=async(url,init)=>{
    const parsed=new URL(url),path=parsed.pathname;
    if(path==='/v1/checkout/sessions'&&init.method==='GET')return Response.json({data:parsed.searchParams.get('customer')==='cus_pending'?[{id:'cs_pending',customer:'cus_pending',mode:'subscription',status:'open',client_reference_id:'firebase:firebase-user'}]:[],has_more:false});
    if(path.endsWith('/expire'))return Response.json({error:'Checkout already completed'},{status:400});
    return original(url,init);
  };
  assert.equal((await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'}),fixture.env)).status,502);
  assert.deepEqual([...fixture.customers.values()],before);assert.equal(fixture.writes.some(w=>w.method==='POST'),false);
});

test('Checkout racing link reservations is retired, and a retry retains access through both sign-ins',{timeout:5000},async()=>{
  for(const finish of ['during','after']){
    const fixture=stripeFixture([{id:'cus_first',metadata:{retirement_site_user_id:'chatgpt-user'}},{id:'cus_pending',metadata:{retirement_firebase_uid:'firebase-user'}}]);
    const bearer=await token();
    let concurrent,started=false,creating,release;
    const creation=new Promise(resolve=>{creating=resolve;}),held=new Promise(resolve=>{release=resolve;});
    const {sessions}=trackCheckouts(fixture,{beforeCreate:async(_params,number)=>{if(number===1){creating();if(finish==='after')await held;}}});
    const upstream=fixture.env.STRIPE_FETCH;
    fixture.env.STRIPE_FETCH=async(url,init)=>{
      const path=new URL(url).pathname,params=new URLSearchParams(init.body||'');
      if(path==='/v1/customers/cus_first'&&init.method==='POST'&&params.has('metadata[retirement_link_reservation]')&&!started){
        started=true;
        concurrent=worker.fetch(request('/api/billing/checkout',{bearer,method:'POST',body:{interval:'monthly'}}),fixture.env);
        if(finish==='during')assert.equal((await concurrent).status,200);else await creation;
      }
      return upstream(url,init);
    };
    let linked;
    try{linked=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'}),fixture.env);}finally{release();}
    assert.equal(linked.status,200);assert.equal((await concurrent).status,finish==='during'?200:409);
    assert.equal(sessions.length,1);assert.equal(sessions[0].customer,'cus_pending');assert.equal(sessions[0].status,'expired');
    assert.equal((await worker.fetch(request('/api/billing/checkout',{bearer,method:'POST',body:{interval:'monthly'}}),fixture.env)).status,200);
    assert.equal(sessions.filter(session=>session.status==='open').length,1);
    const paid=fixture.customers.get(sessions.at(-1).customer);paid.paid=true;
    assert.equal(paid.metadata.retirement_firebase_uid,'firebase-user');assert.equal(paid.metadata.retirement_site_user_id,'chatgpt-user');
    // The directly owned customer remains usable even if search has not caught up.
    for(const identity of [{user:'chatgpt-user'},{bearer}]){
      const statusRequest=request('/api/billing/status',identity);statusRequest.headers.set('x-retirement-customer',paid.id);
      const access=await (await worker.fetch(statusRequest,fixture.env)).json();assert.equal(access.tier,'pro');assert.equal(access.billingCustomerId,paid.id);
      const portalRequest=request('/api/billing/portal',{...identity,method:'POST'});portalRequest.headers.set('x-retirement-customer',paid.id);
      assert.equal((await worker.fetch(portalRequest,fixture.env)).status,200);
    }
  }
});

test('a first-time Checkout finishing during or after linking expires before returning a payment link',{timeout:5000},async()=>{
  for(const provider of ['chatgpt','firebase'])for(const finish of ['during','after']){
    const fixture=stripeFixture(),bearer=await token();
    let reads=0,releaseReads,releaseCheckout,signalLink,linkStarted=false,checkout;
    const snapshots=new Promise(resolve=>{releaseReads=resolve;}),linkUpdating=new Promise(resolve=>{signalLink=resolve;}),held=new Promise(resolve=>{releaseCheckout=resolve;});
    const {sessions}=trackCheckouts(fixture,{beforeCreate:async(_params,number)=>{if(number===1){await linkUpdating;if(finish==='after')await held;}}});
    const upstream=fixture.env.STRIPE_FETCH;
    fixture.env.STRIPE_FETCH=async(url,init)=>{
      const parsed=new URL(url),path=parsed.pathname,params=new URLSearchParams(init.body||'');
      if(path==='/v1/customers'&&init.method==='GET'&&reads<2){
        const snapshot=await upstream(url,init);if(++reads===2)releaseReads();await snapshots;return snapshot;
      }
      if(path.startsWith('/v1/customers')&&init.method==='POST'&&params.get('metadata[retirement_site_user_id]')&&params.get('metadata[retirement_firebase_uid]')&&!linkStarted){
        linkStarted=true;signalLink();if(finish==='during')assert.equal((await checkout).status,409);
      }
      return upstream(url,init);
    };
    checkout=worker.fetch(request('/api/billing/checkout',{...(provider==='chatgpt'?{user:'chatgpt-user'}:{bearer}),method:'POST',body:{interval:'monthly'}}),fixture.env);
    let linked;
    try{linked=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'}),fixture.env);}finally{releaseCheckout();}
    assert.equal(linked.status,200);assert.equal((await checkout).status,409);assert.equal(sessions.length,1);assert.equal(sessions[0].status,'expired');
    assert.equal((await worker.fetch(request('/api/billing/checkout',{...(provider==='chatgpt'?{user:'chatgpt-user'}:{bearer}),method:'POST',body:{interval:'monthly'}}),fixture.env)).status,200);
    assert.equal(sessions.filter(session=>session.status==='open').length,1);
    const paid=fixture.customers.get(sessions.at(-1).customer);paid.paid=true;
    assert.equal(paid.metadata.retirement_site_user_id,'chatgpt-user');assert.equal(paid.metadata.retirement_firebase_uid,'firebase-user');
    for(const identity of [{user:'chatgpt-user'},{bearer}]){
      const status=await(await worker.fetch(request('/api/billing/status',identity),fixture.env)).json();
      assert.equal(status.tier,'pro');assert.equal(status.billingCustomerId,paid.id);
      assert.equal((await worker.fetch(request('/api/billing/portal',{...identity,method:'POST'}),fixture.env)).status,200);
      assert.equal((await worker.fetch(request('/api/billing/checkout',{...identity,method:'POST',body:{interval:'monthly'}}),fixture.env)).status,409);
    }
  }
});

test('linking and a later Checkout cannot leave an earlier in-flight session payable on another customer',{timeout:5000},async()=>{
  for(const interval of ['monthly','yearly']){
    const fixture=stripeFixture([{id:'cus_A',metadata:{retirement_site_user_id:'chatgpt-user'}},{id:'cus_Z',metadata:{retirement_firebase_uid:'firebase-user'}}]);
    const bearer=await token();let started,release;
    const creating=new Promise(resolve=>{started=resolve;}),held=new Promise(resolve=>{release=resolve;});
    const {sessions}=trackCheckouts(fixture,{beforeCreate:async(params,number)=>{if(number===1){assert.equal(params.get('customer'),'cus_Z');started();await held;}}});
    const first=worker.fetch(request('/api/billing/checkout',{bearer,method:'POST',body:{interval:'monthly'}}),fixture.env);
    await creating;
    let second;
    try{
      assert.equal((await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'}),{...fixture.env})).status,200);
      second=await worker.fetch(request('/api/billing/checkout',{user:'chatgpt-user',method:'POST',body:{interval}}),{...fixture.env});
      assert.equal(second.status,200);assert.equal(sessions.length,1);assert.equal(sessions[0].customer,'cus_A');
    }finally{release();}
    const late=await first;
    assert.equal(late.status,409);assert.equal((await late.json()).url,undefined);
    assert.equal(sessions.length,2);assert.equal(sessions.filter(session=>session.status==='open').length,1);
    assert.equal(sessions.find(session=>session.customer==='cus_Z').status,'expired');
    const retry=await worker.fetch(request('/api/billing/checkout',{bearer,method:'POST',body:{interval}}),fixture.env);
    assert.equal(retry.status,200);assert.deepEqual(await retry.json(),await second.json());
  }
});

test('Checkout cannot create a payment link while customer associations are being committed',{timeout:5000},async()=>{
  const fixture=stripeFixture([{id:'cus_A',metadata:{retirement_site_user_id:'chatgpt-user'}},{id:'cus_G',metadata:{retirement_firebase_uid:'firebase-user'}}]),bearer=await token();
  const {sessions}=trackCheckouts(fixture),upstream=fixture.env.STRIPE_FETCH;
  let started,release,heldOnce=false;
  const committing=new Promise(resolve=>{started=resolve;}),held=new Promise(resolve=>{release=resolve;});
  fixture.env.STRIPE_FETCH=async(url,init)=>{
    const params=new URLSearchParams(init.body||'');
    if(init.method==='POST'&&params.get('metadata[retirement_link_state]')==='linked'&&!heldOnce){heldOnce=true;started();await held;}
    return upstream(url,init);
  };
  const link=worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'}),fixture.env);
  await committing;
  try{
    for(const identity of [{user:'chatgpt-user'},{bearer}]){
      const response=await worker.fetch(request('/api/billing/checkout',{...identity,method:'POST',body:{interval:'monthly'}}),{...fixture.env});
      assert.equal(response.status,409);assert.match((await response.json()).error,/being linked/);
    }
    assert.equal(sessions.length,0);
  }finally{release();}
  assert.equal((await link).status,200);
  assert.equal((await worker.fetch(request('/api/billing/checkout',{bearer,method:'POST',body:{interval:'monthly'}}),fixture.env)).status,200);
  assert.equal(sessions.filter(session=>session.status==='open').length,1);
});

test('a losing simultaneous link never associates its paid customer and can link a different Google account',{timeout:5000},async()=>{
  const fixture=stripeFixture([{id:'cus_A',metadata:{retirement_site_user_id:'chatgpt-A'},paid:true},{id:'cus_B',metadata:{retirement_site_user_id:'chatgpt-B'},paid:true},{id:'cus_G',metadata:{retirement_firebase_uid:'firebase-user'}}]);
  const upstream=fixture.env.STRIPE_FETCH,bearer=await token();let lists=0,claims=0,releaseLists,releaseClaims;
  let snapshots=new Promise(resolve=>{releaseLists=resolve;});
  const claimed=new Promise(resolve=>{releaseClaims=resolve;});
  fixture.env.STRIPE_FETCH=async(url,init)=>{
    const path=new URL(url).pathname,params=new URLSearchParams(init.body||'');
    if(path==='/v1/customers'&&init.method==='GET'&&lists<4){
      const snapshot=await upstream(url,init),barrier=snapshots;
      if(++lists%2===0){releaseLists();snapshots=new Promise(resolve=>{releaseLists=resolve;});}
      await barrier;return snapshot;
    }
    if(['/v1/customers/cus_A','/v1/customers/cus_B'].includes(path)&&init.method==='POST'&&params.has('metadata[retirement_link_reservation]')&&claims<2){
      const response=await upstream(url,init);if(++claims===2)releaseClaims();await claimed;return response;
    }
    return upstream(url,init);
  };
  const users=['chatgpt-A','chatgpt-B'];
  const responses=await Promise.all(users.map(user=>worker.fetch(request('/api/billing/link',{user,bearer,method:'POST'}),{...fixture.env})));
  assert.deepEqual(responses.map(response=>response.status).sort(),[200,409]);
  const winner=responses.findIndex(response=>response.status===200),loser=1-winner;
  const winnerCustomer=winner===0?'cus_A':'cus_B',loserCustomer=loser===0?'cus_A':'cus_B';
  assert.equal(fixture.customers.get(winnerCustomer).metadata.retirement_firebase_uid,'firebase-user');
  assert.equal(fixture.customers.get(loserCustomer).metadata.retirement_firebase_uid,undefined);
  assert.equal(fixture.writes.some(write=>write.path==='/v1/customers/'+loserCustomer&&write.values.has('metadata[retirement_firebase_uid]')),false);
  const status=await (await worker.fetch(request('/api/billing/status',{bearer}),fixture.env)).json();
  assert.equal(status.tier,'pro');assert.equal(status.billingCustomerId,winnerCustomer);
  const portal=request('/api/billing/portal',{bearer,method:'POST'});portal.headers.set('x-retirement-customer',loserCustomer);
  assert.equal((await worker.fetch(portal,fixture.env)).status,200);
  assert.equal(fixture.writes.filter(write=>write.path==='/v1/billing_portal/sessions').at(-1).values.get('customer'),winnerCustomer);
  // A failed reservation must not prevent the rejected account from linking
  // an unrelated sign-in after the competing pair has committed.
  const otherBearer=await token({sub:'firebase-other'});
  assert.equal((await worker.fetch(request('/api/billing/link',{user:users[loser],bearer:otherBearer,method:'POST'}),fixture.env)).status,200);
  assert.equal(fixture.customers.get(loserCustomer).metadata.retirement_firebase_uid,'firebase-other');
  assert.notEqual(JSON.parse(fixture.customers.get(loserCustomer).metadata.retirement_link_reservation).generation,'initial');
  assert.equal((await (await worker.fetch(request('/api/billing/status',{bearer:otherBearer}),fixture.env)).json()).billingCustomerId,loserCustomer);
  fixture.keys.clear();
  assert.equal((await worker.fetch(request('/api/billing/link',{user:users[loser],bearer,method:'POST'}),fixture.env)).status,409);
  assert.equal(fixture.customers.get(loserCustomer).metadata.retirement_firebase_uid,'firebase-other');
});

test('an interrupted link resumes its reservations and completed retries cannot re-block Checkout',async()=>{
  const fixture=stripeFixture([{id:'cus_A',metadata:{retirement_site_user_id:'chatgpt-user'}},{id:'cus_G',metadata:{retirement_firebase_uid:'firebase-user'}}]);
  const bearer=await token(),upstream=fixture.env.STRIPE_FETCH;let fail=true;
  fixture.env.STRIPE_FETCH=async(url,init)=>{
    const params=new URLSearchParams(init.body||'');
    if(new URL(url).pathname==='/v1/customers/cus_G'&&init.method==='POST'&&params.has('metadata[retirement_link_reservation]')&&fail){fail=false;return Response.json({error:'offline'},{status:503});}
    return upstream(url,init);
  };
  const link=()=>worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'}),fixture.env);
  assert.equal((await link()).status,502);
  assert.equal(fixture.customers.get('cus_A').metadata.retirement_firebase_uid,undefined);
  assert.equal(fixture.customers.get('cus_G').metadata.retirement_site_user_id,undefined);
  assert.equal((await worker.fetch(request('/api/billing/checkout',{user:'chatgpt-user',method:'POST',body:{interval:'monthly'}}),fixture.env)).status,409);
  assert.equal((await link()).status,200);
  // Reservation keys can expire before later ownership-update keys. Replaying
  // a completed pair must leave its live state linked, even if commit is cached.
  for(const key of fixture.keys.keys())if(key.startsWith('retirement-link-reserve-'))fixture.keys.delete(key);
  assert.equal((await link()).status,200);
  for(const customer of fixture.customers.values())assert.equal(customer.metadata.retirement_link_state,'linked');
  assert.equal((await worker.fetch(request('/api/billing/checkout',{bearer,method:'POST',body:{interval:'monthly'}}),fixture.env)).status,200);
});

test('linking cannot claim a customer from a stale creation response after ownership changes',async()=>{
  const fixture=stripeFixture();
  assert.equal((await worker.fetch(request('/api/billing/checkout',{user:'chatgpt-user',method:'POST',body:{interval:'monthly'}}),fixture.env)).status,200);
  const customer=fixture.customers.get('cus_1');customer.metadata={retirement_site_user_id:'another-chatgpt',retirement_firebase_uid:'another-google'};
  const before=structuredClone(customer);
  const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'}),fixture.env);
  assert.equal(response.status,409);assert.deepEqual(customer,before);assert.equal(fixture.customers.size,1);
  assert.equal(fixture.writes.filter(w=>w.path==='/v1/customers/cus_1'&&w.method==='POST').length,0);
});

test('linking resolves older completed purchases behind newer abandoned Checkouts',async()=>{
  for(const status of ['active','incomplete','canceled'])for(const paginated of [false,true]){
    const fixture=stripeFixture([{id:'cus_first',metadata:{retirement_site_user_id:'chatgpt-user'}},{id:'cus_pending',metadata:{retirement_firebase_uid:'firebase-user'}}]);
    const before=structuredClone([...fixture.customers.values()]),upstream=fixture.env.STRIPE_FETCH;
    const latest={id:'cs_newer',customer:'cus_pending',mode:'subscription',status:'expired',client_reference_id:'firebase:firebase-user'};
    const older={...latest,id:'cs_older',status:'complete',payment_status:'paid',subscription:'sub_pending'};
    fixture.env.STRIPE_FETCH=async(url,init)=>{
      const parsed=new URL(url),path=parsed.pathname;
      if(path==='/v1/checkout/sessions'&&init.method==='GET'&&parsed.searchParams.get('customer')==='cus_pending'){
        if(!paginated)return Response.json({data:[latest,older],has_more:false});
        if(!parsed.searchParams.has('starting_after'))return Response.json({data:[latest],has_more:true});
        assert.equal(parsed.searchParams.get('starting_after'),latest.id);return Response.json({data:[older],has_more:false});
      }
      if(path==='/v1/subscriptions/sub_pending')return Response.json({status,items:{data:[{price:{id:'price_monthly'}}]}});
      return upstream(url,init);
    };
    const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'}),fixture.env);
    assert.equal(response.status,status==='incomplete'?409:200);
    if(status==='incomplete')assert.deepEqual([...fixture.customers.values()],before);
    else assert.equal((await response.json()).billingCustomerId,status==='active'?'cus_pending':'cus_first');
  }
});

test('linking cannot replace a third sign-in on a verified customer record',async()=>{
  for(const metadata of [{retirement_site_user_id:'chatgpt-user',retirement_firebase_uid:'another-google'},{retirement_site_user_id:'another-chatgpt',retirement_firebase_uid:'firebase-user'}]){
    const fixture=stripeFixture([{id:'cus_linked',metadata}]),before=structuredClone([...fixture.customers.values()]);
    const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'}),fixture.env);
    assert.equal(response.status,409);assert.deepEqual([...fixture.customers.values()],before);assert.equal(fixture.writes.some(w=>w.method==='POST'),false);
  }
});

test('conflicting simultaneous Google links cannot both replace the same paid customer',{timeout:5000},async()=>{
  const fixture=stripeFixture([{id:'cus_paid',metadata:{retirement_site_user_id:'chatgpt-user'},paid:true}]);
  const upstream=fixture.env.STRIPE_FETCH;let posts=0,release;
  const barrier=new Promise(resolve=>{release=resolve;});
  fixture.env.STRIPE_FETCH=async(url,init)=>{
    if(new URL(url).pathname==='/v1/customers/cus_paid'&&init.method==='POST'&&posts<2){if(++posts===2)release();await barrier;}
    return upstream(url,init);
  };
  const identities=['firebase-one','firebase-two'],bearers=await Promise.all(identities.map(sub=>token({sub})));
  const responses=await Promise.all(bearers.map(bearer=>worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'}),fixture.env)));
  assert.deepEqual(responses.map(r=>r.status).sort(),[200,409]);
  const winner=responses.findIndex(r=>r.status===200),loser=1-winner;
  assert.equal(fixture.customers.get('cus_paid').metadata.retirement_firebase_uid,identities[winner]);
  assert.equal(fixture.writes.filter(w=>w.path==='/v1/customers/cus_paid'&&w.method==='POST'&&w.values.has('metadata[retirement_firebase_uid]')).length,1);
  for(let i=0;i<bearers.length;i++)assert.equal((await(await worker.fetch(request('/api/billing/status',{bearer:bearers[i]}),fixture.env)).json()).tier,i===winner?'pro':'free');
  // Once idempotency entries expire, the current customer association still
  // rejects a different identity before any update is attempted.
  fixture.keys.clear();
  assert.equal((await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:bearers[loser],method:'POST'}),fixture.env)).status,409);
  assert.equal(fixture.customers.get('cus_paid').metadata.retirement_firebase_uid,identities[winner]);
});

test('concurrent new links provision each identity once and only the winning pair receives access',{timeout:5000},async()=>{
  for(const same of [false,true]){
    const fixture=stripeFixture(),upstream=fixture.env.STRIPE_FETCH;let reads=0,release;
    const barrier=new Promise(resolve=>{release=resolve;});
    fixture.env.STRIPE_FETCH=async(url,init)=>{
      if(new URL(url).pathname==='/v1/customers'&&init.method==='GET'&&reads<2){
        const snapshot=await upstream(url,init);if(++reads===2)release();await barrier;return snapshot;
      }
      return upstream(url,init);
    };
    const identities=['firebase-one',same?'firebase-one':'firebase-two'],bearers=await Promise.all(identities.map(sub=>token({sub})));
    const responses=await Promise.all(bearers.map(bearer=>worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer,method:'POST'}),fixture.env)));
    assert.deepEqual(responses.map(r=>r.status).sort(),same?[200,200]:[200,409]);
    const expectedCustomers=same?2:3;
    assert.equal(fixture.customers.size,expectedCustomers);assert.equal(fixture.writes.filter(w=>w.path==='/v1/customers'&&w.method==='POST').length,expectedCustomers);
    const winner=responses.findIndex(r=>r.status===200),paid=fixture.customers.get('cus_1');paid.paid=true;
    assert.equal(paid.metadata.retirement_site_user_id,'chatgpt-user');assert.equal(paid.metadata.retirement_firebase_uid,identities[winner]);
    for(let i=0;i<bearers.length;i++){
      const access=await(await worker.fetch(request('/api/billing/status',{bearer:bearers[i]}),fixture.env)).json();
      assert.equal(access.tier,same||i===winner?'pro':'free');
    }
    assert.equal((await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:bearers[winner],method:'POST'}),fixture.env)).status,200);
    assert.equal(fixture.customers.size,expectedCustomers);
  }
});

test('linking resolves completed purchases directly before selecting a customer',async()=>{
  for(const status of ['active','incomplete','canceled']){
    const fixture=stripeFixture([{id:'cus_first',metadata:{retirement_site_user_id:'chatgpt-user'}},{id:'cus_pending',metadata:{retirement_firebase_uid:'firebase-user'}}]);
    const before=structuredClone([...fixture.customers.values()]),original=fixture.env.STRIPE_FETCH;
    fixture.env.STRIPE_FETCH=async(url,init)=>{
      const parsed=new URL(url),path=parsed.pathname;
      if(path==='/v1/checkout/sessions'&&init.method==='GET')return Response.json({data:parsed.searchParams.get('customer')==='cus_pending'?[{id:'cs_completed',customer:'cus_pending',mode:'subscription',status:'complete',payment_status:status==='active'?'paid':'unpaid',subscription:'sub_pending',client_reference_id:'firebase:firebase-user'}]:[],has_more:false});
      if(path==='/v1/subscriptions/sub_pending')return Response.json({id:'sub_pending',status,items:{data:[{price:{id:'price_monthly'}}]}});
      return original(url,init);
    };
    const response=await worker.fetch(request('/api/billing/link',{user:'chatgpt-user',bearer:await token(),method:'POST'}),fixture.env);
    if(status==='incomplete'){assert.equal(response.status,409);assert.deepEqual([...fixture.customers.values()],before);}
    else {assert.equal(response.status,200);assert.equal((await response.json()).billingCustomerId,status==='active'?'cus_pending':'cus_first');}
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
    assert.equal(fixture.customers.size,2);
    for(const customer of fixture.customers.values()){
      assert.equal(customer.metadata.retirement_site_user_id,'chatgpt-user');assert.equal(customer.metadata.retirement_firebase_uid,'firebase-user');
    }
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
