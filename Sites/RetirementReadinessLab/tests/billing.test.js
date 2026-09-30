import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../worker/index.js';
import { access as billingAccess, activeSubscription } from '../worker/billing.js';
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
  if (parsed.pathname === '/v1/customers/cus_123') return Response.json(customer);
  if (parsed.pathname === '/v1/subscriptions') return Response.json({ data: [subscription] });
  if (parsed.pathname === '/v1/checkout/sessions/cs_test_12345678') return Response.json({ client_reference_id: 'user-123', customer: customer.id, subscription: subscription.id });
  if (parsed.pathname === '/v1/subscriptions/sub_123') return Response.json(subscription);
  if (parsed.pathname === '/v1/checkout/sessions') return init.method === 'GET' ? Response.json({data:[],has_more:false}) : Response.json({ url: 'https://checkout.stripe.com/c/pay/example' });
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

test('a matching subscription on a later page grants Pro and prevents duplicate Checkout',async()=>{
  const cursors=[];
  const env={...stripeEnv,STRIPE_FETCH:async(url,init)=>{
    const parsed=new URL(url);
    if(parsed.pathname==='/v1/subscriptions'){
      const cursor=parsed.searchParams.get('starting_after');cursors.push(cursor);
      assert.equal(parsed.searchParams.get('customer'),customer.id);
      assert.equal(parsed.searchParams.get('status'),'all');assert.equal(parsed.searchParams.get('limit'),'100');
      if(!cursor)return Response.json({data:Array.from({length:100},(_,i)=>({...subscription,id:'sub_old_'+i,status:'canceled'})),has_more:true});
      assert.equal(cursor,'sub_old_99');return Response.json({data:[subscription],has_more:false});
    }
    if(parsed.pathname==='/v1/checkout/sessions')assert.fail('A paid account must not create another Checkout');
    return stripeEnv.STRIPE_FETCH(url,init);
  }};
  const status=await (await worker.fetch(request('/api/billing/status',{user:'user-123'}),env)).json();
  assert.equal(status.tier,'pro');
  const checkout=await worker.fetch(request('/api/billing/checkout',{method:'POST',origin:site,user:'user-123'}),env);
  assert.equal(checkout.status,409);assert.deepEqual(cursors,[null,'sub_old_99',null,'sub_old_99']);
});

test('subscription pagination scans nonmatching pages until it finds Pro or reaches the end',async()=>{
  for(const paid of [false,true]){
    const cursors=[];
    const env={...stripeEnv,STRIPE_FETCH:async url=>{
      const cursor=new URL(url).searchParams.get('starting_after');cursors.push(cursor);
      if(!cursor)return Response.json({data:[{...subscription,id:'sub_other',items:{data:[{price:{id:'price_unrelated'}}]}}],has_more:true});
      if(cursor==='sub_other')return Response.json({data:[{...subscription,id:'sub_canceled',status:'canceled'}],has_more:true});
      assert.equal(cursor,'sub_canceled');return Response.json({data:paid?[subscription]:[],has_more:false});
    }};
    assert.deepEqual(await activeSubscription(env,customer.id),paid?subscription:null);
    assert.deepEqual(cursors,[null,'sub_other','sub_canceled']);
  }
});

test('unreadable, stalled or failed subscription pagination cannot authorize another purchase',async()=>{
  for(const next of [{data:[],has_more:true},{data:[{id:'sub_old',status:'canceled'}],has_more:true},{error:'offline'}, {data:null}]){
    let count=0;
    const env={...stripeEnv,STRIPE_FETCH:async(url,init)=>{
      if(new URL(url).pathname==='/v1/subscriptions'){
        if(count++===0)return Response.json({data:[{id:'sub_old',status:'canceled'}],has_more:true});
        return Response.json(next,{status:next.error?503:200});
      }
      if(new URL(url).pathname==='/v1/checkout/sessions')assert.fail('An incomplete subscription lookup must block Checkout');
      return stripeEnv.STRIPE_FETCH(url,init);
    }};
    const response=await worker.fetch(request('/api/billing/checkout',{method:'POST',origin:site,user:'user-123'}),env);
    assert.equal(response.status,502);
  }
});

test('an unpaid customer reference cannot hide a paid customer or permit another checkout', async () => {
  const oldCustomer={id:'cus_old',metadata:customer.metadata};
  const checked=[];
  const env={...stripeEnv,STRIPE_FETCH:async(url,init)=>{
    const parsed=new URL(url);
    if(parsed.pathname==='/v1/customers/cus_old')return Response.json(oldCustomer);
    if(parsed.pathname==='/v1/customers/search')return Response.json({data:[oldCustomer,customer]});
    if(parsed.pathname==='/v1/subscriptions'){
      const id=parsed.searchParams.get('customer');checked.push(id);
      return Response.json({data:id===customer.id?[subscription]:[]});
    }
    if(parsed.pathname==='/v1/checkout/sessions')assert.fail('A paying account must not create another checkout');
    return stripeEnv.STRIPE_FETCH(url,init);
  }};
  const headers={'oai-authenticated-user-id':'user-123','x-retirement-customer':'cus_old',Origin:site};
  const status=await (await worker.fetch(new Request(site+'/api/billing/status',{headers}),env)).json();
  assert.equal(status.tier,'pro');assert.equal(status.billingCustomerId,customer.id);
  const portal=await worker.fetch(new Request(site+'/api/billing/portal',{method:'POST',headers}),env);
  assert.equal(portal.status,200);
  const checkout=await worker.fetch(new Request(site+'/api/billing/checkout',{method:'POST',headers:{...headers,'Content-Type':'application/json'},body:JSON.stringify({interval:'monthly'})}),env);
  assert.equal(checkout.status,409);
  assert.equal(checked.filter(id=>id==='cus_old').length,3,'each lookup checks the old customer only once');
});

test('an unpaid verified reference retains portal access when search is delayed',async()=>{
  const env={...stripeEnv,STRIPE_FETCH:async(url,init)=>{
    const path=new URL(url).pathname;
    if(path==='/v1/customers/search'||path==='/v1/subscriptions')return Response.json({data:[]});
    return stripeEnv.STRIPE_FETCH(url,init);
  }};
  const account=await billingAccess(env,'user-123',null,customer.id);
  assert.equal(account.pro,false);assert.equal(account.customerId,customer.id);
});

test('past-due and canceled customers retain portal access without Pro entitlement', async () => {
  for(const status of ['past_due','unpaid','canceled']){
    const env={...stripeEnv,STRIPE_FETCH:async(url,init)=>new URL(url).pathname==='/v1/subscriptions'?Response.json({data:[{...subscription,status}]}):stripeEnv.STRIPE_FETCH(url,init)};
    const result=await (await worker.fetch(request('/api/billing/status',{user:'user-123'}),env)).json();
    assert.equal(result.tier,'free');assert.equal(result.billingPortalAvailable,true);
    const response=await worker.fetch(request('/api/billing/portal',{method:'POST',origin:site,user:'user-123'}),env);
    assert.equal(response.status,200);assert.equal(new URL((await response.json()).url).hostname,'billing.stripe.com');
  }
  const env={...stripeEnv,STRIPE_FETCH:async()=>Response.json({data:[]})};
  const result=await (await worker.fetch(request('/api/billing/status',{user:'new-user'}),env)).json();
  assert.equal(result.billingPortalAvailable,false);
});

test('checkout and direct customer references work while search has no matching record',async()=>{
  const env={...stripeEnv,STRIPE_FETCH:async(url,init)=>new URL(url).pathname==='/v1/customers/search'?Response.json({data:[]}):stripeEnv.STRIPE_FETCH(url,init)};
  const first=await (await worker.fetch(request('/api/billing/status?session_id=cs_live_12345678',{user:'user-123'}),{...env,STRIPE_FETCH:async(url,init)=>new URL(url).pathname.includes('/checkout/sessions/')?Response.json({client_reference_id:'user-123',customer:'cus_123',subscription:'sub_123'}):env.STRIPE_FETCH(url,init)})).json();
  assert.equal(first.tier,'pro');assert.equal(first.billingCustomerId,'cus_123');
  const headers={'oai-authenticated-user-id':'user-123','x-retirement-customer':first.billingCustomerId};
  const later=await (await worker.fetch(new Request(site+'/api/billing/status',{headers}),env)).json();assert.equal(later.tier,'pro');
  const portal=await worker.fetch(new Request(site+'/api/billing/portal',{method:'POST',headers:{...headers,Origin:site}}),env);assert.equal(portal.status,200);
  const checkout=await worker.fetch(new Request(site+'/api/billing/checkout',{method:'POST',headers:{...headers,Origin:site,'Content-Type':'application/json'},body:JSON.stringify({interval:'monthly'})}),env);assert.equal(checkout.status,409);
});

test('untrusted customer references cannot grant access or open someone else billing',async()=>{
  const env={...stripeEnv,STRIPE_FETCH:async(url,init)=>new URL(url).pathname==='/v1/customers/search'?Response.json({data:[]}):stripeEnv.STRIPE_FETCH(url,init)};
  const headers={'oai-authenticated-user-id':'other-user','x-retirement-customer':'cus_123',Origin:site};
  const status=await (await worker.fetch(new Request(site+'/api/billing/status',{headers}),env)).json();assert.equal(status.tier,'free');assert.equal(status.billingCustomerId,null);
  const portal=await worker.fetch(new Request(site+'/api/billing/portal',{method:'POST',headers}),env);assert.equal(portal.status,404);
  const expired={...env,STRIPE_FETCH:async(url,init)=>new URL(url).pathname==='/v1/subscriptions'?Response.json({data:[{...subscription,status:'canceled'}]}):env.STRIPE_FETCH(url,init)};
  const ended=await billingAccess(expired,'user-123',null,'cus_123');assert.equal(ended.pro,false);assert.equal(ended.customerId,'cus_123');
});

test('deleted, missing and reassigned customer references fall back without granting access',async()=>{
  for(const customerResponse of [Response.json({deleted:true,id:'cus_123'}),Response.json({error:'missing'},{status:404}),Response.json({...customer,metadata:{retirement_site_user_id:'new-owner'}})]){
    const env={...stripeEnv,STRIPE_FETCH:async(url)=>new URL(url).pathname==='/v1/customers/cus_123'?customerResponse:Response.json({data:[]})};
    assert.equal((await billingAccess(env,'user-123',null,'cus_123')).pro,false);
  }
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
  assert.equal(new URLSearchParams(calls.find(call => call.path === '/v1/checkout/sessions' && call.method === 'POST').body).get('line_items[0][price]'), 'price_monthly');
  calls.length = 0;
  const yearly = await (await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'user-123', interval: 'yearly' }), freeEnv)).json();
  assert.equal(new URL(yearly.url).hostname, 'checkout.stripe.com');
  assert.equal(new URLSearchParams(calls.find(call => call.path === '/v1/checkout/sessions' && call.method === 'POST').body).get('line_items[0][price]'), 'price_yearly');
  calls.length = 0;
  assert.equal((await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'user-123', interval: 'weekly' }), freeEnv)).status, 400);
  assert.equal(calls.length, 0);
  const portal = await (await worker.fetch(request('/api/billing/portal', { method: 'POST', origin: site, user: 'user-123' }), freeEnv)).json();
  assert.equal(new URL(portal.url).hostname, 'billing.stripe.com');
});

test('sandbox visitors cannot check out and the signed-in owner can test Checkout while keeping complimentary Pro', async () => {
  calls.length = 0;
  const testEnv = { ...stripeEnv, STRIPE_SECRET_KEY: 'sk_test_fake' };
  const visitor = await (await worker.fetch(request('/api/billing/status', { user: 'someone-else' }), testEnv)).json();
  assert.equal(visitor.checkoutAvailable, false);
  assert.equal(visitor.maxPaths, 4);
  assert.equal((await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'someone-else' }), testEnv)).status, 403);
  assert.equal(calls.length, 0);
  const ownerCustomer={id:'cus_owner',metadata:{retirement_site_user_id:'site-owner-id'}};
  const ownerEnv={...testEnv,STRIPE_FETCH:async(url,init)=>{
    const path=new URL(url).pathname;
    if(path==='/v1/customers/search')return Response.json({data:[ownerCustomer]});
    if(path==='/v1/subscriptions')return Response.json({data:[]});
    return stripeEnv.STRIPE_FETCH(url,init);
  }};
  const owner = await (await worker.fetch(request('/api/billing/status', { user: 'site-owner-id', email: 'rtsoliday@gmail.com' }), ownerEnv)).json();
  assert.equal(owner.tier, 'pro'); assert.equal(owner.checkoutAvailable, true);assert.equal(owner.testBilling,true);
  assert.equal((await worker.fetch(request('/api/billing/checkout', { method: 'POST', origin: site, user: 'site-owner-id', email: 'rtsoliday@gmail.com' }), ownerEnv)).status, 200);
  const session=calls.find(call=>call.path==='/v1/checkout/sessions' && call.method==='POST');
  assert.equal(new URLSearchParams(session.body).get('line_items[0][price]'),'price_test_monthly');
});

test('owner billing lookup retains portal access for paid and canceled subscriptions without affecting complimentary Pro',async()=>{
  const ownerId='028696a7-7846-4822-a4c1-67026aa2383f',ownerCustomer={...customer,metadata:{retirement_site_user_id:ownerId}};
  for(const status of ['active','canceled']){
    const env={...stripeEnv,STRIPE_FETCH:async(url,init)=>{
      const path=new URL(url).pathname;
      if(path==='/v1/customers/search')return Response.json({data:[ownerCustomer]});
      if(path==='/v1/customers/cus_123')return Response.json(ownerCustomer);
      if(path==='/v1/subscriptions')return Response.json({data:[{...subscription,status}]});
      return stripeEnv.STRIPE_FETCH(url,init);
    }};
    const headers={'oai-authenticated-user-id':ownerId,'x-retirement-customer':customer.id,Origin:site};
    const access=await (await worker.fetch(new Request(site+'/api/billing/status',{headers}),env)).json();
    assert.equal(access.tier,'pro');assert.equal(access.ownerAccess,true);assert.equal(access.billingPortalAvailable,true);assert.equal(access.billingCustomerId,customer.id);
    assert.equal((await worker.fetch(new Request(site+'/api/billing/portal',{method:'POST',headers}),env)).status,200);
    assert.equal((await worker.fetch(request('/api/billing/checkout',{method:'POST',origin:site,user:ownerId}),env)).status,409);
  }
});

test('Stripe outages preserve owner Pro and offer an ownership-checked portal retry',async()=>{
  const env={...stripeEnv,STRIPE_FETCH:async()=>{throw new Error('offline');}};
  const owner='028696a7-7846-4822-a4c1-67026aa2383f';
  const access=await (await worker.fetch(request('/api/billing/status',{user:owner}),env)).json();
  assert.equal(access.tier,'pro');assert.equal(access.billingLookupUnavailable,true);assert.equal(access.billingPortalAvailable,true);assert.equal(access.checkoutAvailable,false);
  assert.equal((await worker.fetch(request('/api/billing/portal',{method:'POST',origin:site,user:owner}),env)).status,502);
});

test('an active owner test subscription blocks duplicate sandbox checkout',async()=>{
  const user='028696a7-7846-4822-a4c1-67026aa2383f';
  const env={...stripeEnv,STRIPE_SECRET_KEY:'sk_test_fake',STRIPE_FETCH:async(url)=>{
    const path=new URL(url).pathname;
    if(path==='/v1/customers/search')return Response.json({data:[{id:'cus_owner',metadata:{retirement_site_user_id:user}}]});
    if(path==='/v1/subscriptions')return Response.json({data:[{...subscription,items:{data:[{price:{id:'price_test_monthly'}}]}}]});
    assert.fail('Duplicate checkout must not create a Stripe session');
  }};
  const access=await (await worker.fetch(request('/api/billing/status',{user}),env)).json();
  assert.equal(access.tier,'pro');assert.equal(access.checkoutAvailable,false);assert.equal(access.billingPortalAvailable,true);
  assert.equal((await worker.fetch(request('/api/billing/checkout',{method:'POST',origin:site,user}),env)).status,409);
});

test('owner can manage existing billing when Checkout prices are missing',async()=>{
  const user='028696a7-7846-4822-a4c1-67026aa2383f';
  const env={...stripeEnv,STRIPE_PRO_LIVE_MONTHLY_PRICE_ID:undefined,STRIPE_FETCH:async(url,init)=>{
    if(new URL(url).pathname==='/v1/customers/search')return Response.json({data:[{id:customer.id,metadata:{retirement_site_user_id:user}}]});
    return stripeEnv.STRIPE_FETCH(url,init);
  }};
  const access=await (await worker.fetch(request('/api/billing/status',{user}),env)).json();
  assert.equal(access.tier,'pro');assert.equal(access.checkoutAvailable,false);assert.equal(access.billingPortalAvailable,true);
  assert.equal((await worker.fetch(request('/api/billing/portal',{method:'POST',origin:site,user}),env)).status,200);
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

test('missing Checkout prices retain paid access and billing management for existing subscribers', async () => {
  for (const missing of ['STRIPE_PRO_LIVE_YEARLY_PRICE_ID','STRIPE_PRO_LIVE_MONTHLY_PRICE_ID','both']) {
    const paid = {...subscription, metadata:{retirement_site_user_id:'user-123'}};
    const env = {...stripeEnv, STRIPE_FETCH:async(url,init)=>new URL(url).pathname==='/v1/subscriptions'?Response.json({data:[paid]}):stripeEnv.STRIPE_FETCH(url,init)};
    if(missing==='both'){delete env.STRIPE_PRO_LIVE_MONTHLY_PRICE_ID;delete env.STRIPE_PRO_LIVE_YEARLY_PRICE_ID;}else delete env[missing];
    const result = await (await worker.fetch(request('/api/billing/status',{user:'user-123'}),env)).json();
    assert.equal(result.tier,'pro');assert.equal(result.checkoutAvailable,false);assert.equal(result.billingPortalAvailable,true);
    assert.equal((await worker.fetch(request('/api/billing/portal',{user:'user-123',method:'POST',origin:site}),env)).status,200);
    assert.equal((await worker.fetch(request('/api/billing/checkout',{user:'user-123',method:'POST',origin:site}),env)).status,503);
  }
});

test('incomplete prices do not treat unrelated or malformed subscriptions as Pro',async()=>{
  for(const unrelated of [{...subscription,items:{data:[{price:{id:'unrelated'}}]}},{...subscription,items:{data:[{}]}},{...subscription,metadata:{retirement_site_user_id:'another-user'},items:{data:[{price:{id:'unrelated'}}]}}]){
    const env={...stripeEnv,STRIPE_PRO_LIVE_MONTHLY_PRICE_ID:undefined,STRIPE_FETCH:async(url,init)=>new URL(url).pathname==='/v1/subscriptions'?Response.json({data:[unrelated]}):stripeEnv.STRIPE_FETCH(url,init)};
    const result=await (await worker.fetch(request('/api/billing/status',{user:'user-123'}),env)).json();
    assert.equal(result.tier,'free');assert.equal(result.billingPortalAvailable,true);assert.equal(result.checkoutAvailable,false);
  }
});

// Shared fake Stripe backend: separate Worker requests coordinate only through
// Stripe history and its idempotency store, rather than a process-local lock.
function checkoutFixture(initial=[]) {
  const sessions=structuredClone(initial),keys=new Map(),events=[];
  const env={...stripeEnv,STRIPE_FETCH:async(url,init)=>{
    const parsed=new URL(url),path=parsed.pathname,params=new URLSearchParams(init.body||'');
    if(path==='/v1/subscriptions')return Response.json({data:[]});
    if(path==='/v1/checkout/sessions'&&init.method==='GET')return Response.json({data:structuredClone(sessions),has_more:false});
    const expiring=/^\/v1\/checkout\/sessions\/([^/]+)\/expire$/.exec(path);
    if(expiring){const session=sessions.find(s=>s.id===expiring[1]);if(session?.status!=='open')return Response.json({error:'not open'},{status:400});session.status='expired';events.push('expire:'+session.id);return Response.json(session);}
    if(path==='/v1/checkout/sessions'&&init.method==='POST'){
      const key=init.headers['Idempotency-Key'];assert.ok(key);const previous=keys.get(key);
      if(previous)return previous.body===init.body?Response.json(previous.session):Response.json({error:'idempotency mismatch'},{status:400});
      const session={id:'cs_new'+(sessions.length+1),customer:params.get('customer'),mode:'subscription',status:'open',client_reference_id:params.get('client_reference_id'),metadata:{retirement_plan:params.get('metadata[retirement_plan]'),retirement_price:params.get('metadata[retirement_price]')},url:'https://checkout.stripe.com/c/pay/new'+(sessions.length+1)};
      keys.set(key,{body:init.body,session});sessions.unshift(session);events.push('create:'+session.id);return Response.json(session);
    }
    const items=/^\/v1\/checkout\/sessions\/([^/]+)\/line_items$/.exec(path);
    if(items)return Response.json({data:[{price:{id:sessions.find(s=>s.id===items[1]).price}}]});
    return stripeEnv.STRIPE_FETCH(url,init);
  }};
  return {env,sessions,keys,events};
}
const checkoutRequest=interval=>request('/api/billing/checkout',{user:'user-123',method:'POST',origin:site,interval});

test('concurrent identical checkouts create one session and return the same payment link',async()=>{
  const f=checkoutFixture();
  const responses=await Promise.all([worker.fetch(checkoutRequest('monthly'),f.env),worker.fetch(checkoutRequest('monthly'),{...f.env})]);
  assert.deepEqual(responses.map(r=>r.status),[200,200]);assert.deepEqual(await responses[0].json(),await responses[1].json());
  assert.equal(f.sessions.length,1);assert.equal(f.keys.size,1);
  assert.equal((await worker.fetch(checkoutRequest('monthly'),f.env)).status,200);assert.equal(f.sessions.length,1);
});

test('concurrent different intervals cannot create two payable sessions',async()=>{
  const f=checkoutFixture();const responses=await Promise.all([worker.fetch(checkoutRequest('monthly'),f.env),worker.fetch(checkoutRequest('yearly'),{...f.env})]);
  assert.equal(responses.filter(r=>r.status===200).length,1);assert.equal(f.sessions.filter(s=>s.status==='open').length,1);assert.equal(f.keys.size,1);
});

test('changing intervals expires the old link before creating a replacement; expired sessions can restart',async()=>{
  const f=checkoutFixture();await worker.fetch(checkoutRequest('monthly'),f.env);
  const old=f.sessions[0];assert.equal((await worker.fetch(checkoutRequest('yearly'),f.env)).status,200);
  assert.equal(old.status,'expired');assert.deepEqual(f.events.slice(1),['expire:'+old.id,'create:cs_new2']);assert.equal(f.sessions[0].metadata.retirement_price,'price_yearly');
  f.sessions[0].status='expired';assert.equal((await worker.fetch(checkoutRequest('monthly'),f.env)).status,200);
  assert.equal(f.sessions.filter(s=>s.status==='open').length,1);assert.equal(f.keys.size,3);
});

test('legacy pending sessions are reused and a completed pending payment blocks another purchase',async()=>{
  const legacy={id:'cs_legacy',customer:customer.id,mode:'subscription',status:'open',client_reference_id:'user-123',price:'price_monthly',url:'https://checkout.stripe.com/c/pay/legacy'};
  const f=checkoutFixture([legacy]);const response=await worker.fetch(checkoutRequest('monthly'),f.env);
  assert.deepEqual(await response.json(),{url:legacy.url});assert.equal(f.keys.size,0);
  f.sessions[0].status='complete';f.sessions[0].payment_status='unpaid';
  assert.equal((await worker.fetch(checkoutRequest('monthly'),f.env)).status,409);assert.equal(f.keys.size,0);
});

test('a confirmed canceled subscription permits retrying an unsuccessful completed Checkout',async()=>{
  const f=checkoutFixture([{id:'cs_failed',customer:customer.id,mode:'subscription',status:'complete',payment_status:'unpaid',subscription:'sub_failed',client_reference_id:'user-123'}]);
  for(const status of ['incomplete','canceled']){
    const env={...f.env,STRIPE_FETCH:async(url,init)=>new URL(url).pathname==='/v1/subscriptions/sub_failed'?Response.json({status}):f.env.STRIPE_FETCH(url,init)};
    assert.equal((await worker.fetch(checkoutRequest('monthly'),env)).status,status==='canceled'?200:409);
  }
  assert.equal(f.sessions.filter(s=>s.status==='open').length,1);
});

test('Checkout history and expiration failures cannot bypass purchase coordination',async()=>{
  for(const failure of ['history','expire']){
    const f=checkoutFixture([{id:'cs_legacy',customer:customer.id,mode:'subscription',status:'open',client_reference_id:'user-123',price:'price_monthly',url:'https://checkout.stripe.com/c/pay/legacy'}]);
    const env={...f.env,STRIPE_FETCH:async(url,init)=>{
      const path=new URL(url).pathname;
      if(failure==='history'&&path==='/v1/checkout/sessions'&&init.method==='GET'||failure==='expire'&&path.endsWith('/expire'))return Response.json({error:'offline'},{status:503});
      return f.env.STRIPE_FETCH(url,init);
    }};
    assert.equal((await worker.fetch(checkoutRequest('yearly'),env)).status,502);assert.equal(f.keys.size,0);
  }
});

test('pending Checkout lookup paginates and recognizes a previously linked Google identity',async()=>{
  const pending={id:'cs_google',customer:customer.id,mode:'subscription',status:'open',client_reference_id:'firebase:linked-user',price:'price_monthly',url:'https://checkout.stripe.com/c/pay/linked'};
  const f=checkoutFixture([pending]);let pages=0;
  const env={...f.env,STRIPE_FETCH:async(url,init)=>{
    const parsed=new URL(url);
    if(parsed.pathname==='/v1/customers/search')return Response.json({data:[{...customer,metadata:{...customer.metadata,retirement_firebase_uid:'linked-user'}}]});
    if(parsed.pathname==='/v1/checkout/sessions'&&init.method==='GET'){
      pages++;
      if(!parsed.searchParams.has('starting_after'))return Response.json({data:[{id:'cs_unrelated',customer:customer.id,mode:'payment',status:'complete'}],has_more:true});
      assert.equal(parsed.searchParams.get('starting_after'),'cs_unrelated');
    }
    return f.env.STRIPE_FETCH(url,init);
  }};
  const response=await worker.fetch(checkoutRequest('monthly'),env);
  assert.equal(response.status,200);assert.deepEqual(await response.json(),{url:pending.url});assert.equal(pages,2);assert.equal(f.keys.size,0);
});
