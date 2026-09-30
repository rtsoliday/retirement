import vm from 'node:vm';
import {readFileSync,existsSync} from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';
import worker, { analyticsQuery, normalizeDaily, utcWindow } from '../worker/index.js';

const owner = '028696a7-7846-4822-a4c1-67026aa2383f';
const ownerEmail = 'rtsoliday@gmail.com';
const assets = { fetch: async request => new Response(request.url.includes('admin.html') ? 'admin page' : 'planner', { headers: { 'Content-Type': 'text/html' } }) };
const request = (path, user, email) => new Request(`https://retirementforecast.us${path}`, { headers: { ...(user ? { 'oai-authenticated-user-id': user } : {}), ...(email ? { 'oai-authenticated-user-email': email } : {}) } });

test('admin route redirects anonymous visitors and rejects other signed-in users', async () => {
  const anonymous = await worker.fetch(request('/admin'), { ASSETS: assets });
  assert.equal(anonymous.status, 302);
  assert.equal(new URL(anonymous.headers.get('location')).pathname, '/signin-with-chatgpt');
  const other = await worker.fetch(request('/admin', 'someone-else'), { ASSETS: assets });
  assert.equal(other.status, 403);
  assert.equal((await worker.fetch(request('/admin', 'someone-else', 'other@example.com'), { ASSETS: assets })).status, 403);
  assert.equal((await worker.fetch(request('/admin', null, ownerEmail), { ASSETS: assets })).status, 302);
  const allowed = await worker.fetch(request('/admin', owner), { ASSETS: assets });
  assert.equal(allowed.status, 200);
  assert.equal(await allowed.text(), '__ADMIN_HTML__');
  assert.equal(allowed.headers.get('cache-control'), 'no-store');
  assert.equal((await worker.fetch(request('/admin', 'site-scoped-owner-id', ' RTSOLIDAY@gmail.com '), { ASSETS: assets })).status, 200);
});

test('admin assets and planner links resolve to existing files from every supported route',async()=>{
  const html=readFileSync(new URL('../dist/admin.html',import.meta.url),'utf8');
  const localLinks=[...html.matchAll(/(?:href|src)="([/.][^"]+)"/g)].map(match=>match[1]);
  assert.ok(localLinks.includes('/admin.js'));
  assert.ok(localLinks.includes('/admin.css'));
  assert.ok(localLinks.includes('/index.html'));
  for(const route of ['/admin','/admin/','/admin.html']){
    const response=await worker.fetch(request(route,owner),{ASSETS:assets});
    assert.equal(response.status,200);
    for(const link of localLinks){
      const url=new URL(link,request(route,owner).url);
      assert.ok(existsSync(new URL('../dist'+url.pathname,import.meta.url)),`${route}: missing ${url.pathname}`);
    }
  }
});

test('analytics API fails closed and never calls Cloudflare for unauthorized visitors', async () => {
  let called = false;
  const env = { ASSETS: assets, CLOUDFLARE_ANALYTICS_TOKEN: 'test-token', UPSTREAM_FETCH: async () => { called = true; throw new Error('not expected'); } };
  const response = await worker.fetch(request('/api/admin/traffic?days=7'), env);
  assert.equal(response.status, 403);
  assert.equal(called, false);
  assert.equal((await worker.fetch(request('/api/admin/traffic?days=7', 'other-user', 'other@example.com'), env)).status, 403);
  assert.equal(called, false);
  assert.equal((await worker.fetch(request('/api/admin/traffic?days=9', 'site-scoped-owner-id', ownerEmail), env)).status, 400);
  assert.equal((await worker.fetch(request('/api/admin/traffic?days=9', owner), env)).status, 400);
  assert.equal(called, false);
});

test('daily Cloudflare values are normalized without adding daily uniques into a false period total', async () => {
  const { start } = utcWindow(7);
  let sent;
  const env = {
    ASSETS: assets,
    CLOUDFLARE_ANALYTICS_TOKEN: 'test-token',
    UPSTREAM_FETCH: async (url, options) => {
      sent = { url, options };
      return Response.json({ data: { viewer: { zones: [{ daily: [{ dimensions: { date: start }, sum: { pageViews: 5 }, uniq: { uniques: 3 } }] }] } } });
    },
  };
  const response = await worker.fetch(request('/api/admin/traffic?days=7', owner), env);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.series.length, 7);
  assert.equal(data.pageViews, 5);
  assert.equal(data.peakDailyUniqueIps, 3);
  assert.equal(data.series[0].uniqueIps, 3);
  assert.equal(data.series[1].uniqueIps, 0);
  assert.equal(sent.url, 'https://api.cloudflare.com/client/v4/graphql');
  assert.equal(sent.options.headers.Authorization, 'Bearer test-token');
  assert.match(JSON.parse(sent.options.body).query, /httpRequests1dGroups/);
  assert.equal(response.headers.get('cache-control'), 'no-store');
});

test('missing token and malformed Cloudflare data produce clear errors', async () => {
  const missing = await worker.fetch(request('/api/admin/traffic', owner), { ASSETS: assets });
  assert.equal(missing.status, 503);
  const malformed = await worker.fetch(request('/api/admin/traffic', owner), {
    ASSETS: assets,
    CLOUDFLARE_ANALYTICS_TOKEN: 'test-token',
    UPSTREAM_FETCH: async () => Response.json({ data: { viewer: { zones: [{ daily: [{ dimensions: { date: 'bad' }, sum: { pageViews: 1 }, uniq: { uniques: 1 } }] }] } } }),
  });
  assert.equal(malformed.status, 502);
  assert.match((await malformed.json()).error, /unexpected/i);
});

test('query uses a fixed zone and server generated dates', () => {
  assert.deepEqual(utcWindow(7, new Date('2026-09-28T12:00:00Z')), { start: '2026-09-22', end: '2026-09-29' });
  assert.match(analyticsQuery('2026-09-22', '2026-09-29'), /19fc99ed5a9bc17308d434c3c7d959aa/);
  assert.equal(normalizeDaily([], 7, new Date('2026-09-28T12:00:00Z')).length, 7);
});


function adminBrowser(){
  const elements=new Map();
  const element=key=>{if(!elements.has(key))elements.set(key,{textContent:'',hidden:true,disabled:false,classList:{toggle(){}},listeners:{},setAttribute(){},addEventListener(event,fn){this.listeners[event]=fn;},replaceChildren(){},append(){},dataset:{}});return elements.get(key);};
  const days=[7,30].map(n=>({...element('days'+n),dataset:{days:String(n)}})),pending=[];
  const context=vm.createContext({Intl,Date,fetch:url=>new Promise((resolve,reject)=>pending.push({url,resolve,reject})),document:{querySelector:element,querySelectorAll:()=>days,createElement:()=>({dataset:{},style:{},setAttribute(){},append(){}})}});
  vm.runInContext(readFileSync(new URL('../dist/admin.js',import.meta.url),'utf8'),context);
  const data=n=>({series:[{date:'2026-09-29',pageViews:n,uniqueIps:n}],pageViews:n,peakDailyUniqueIps:n,start:'2026-09-29'});
  return {element,days,pending,data};
}

test('stale analytics successes, HTTP errors and network errors cannot replace the selected range',async()=>{
  for(const outcome of ['success','http','network']){
    const a=adminBrowser();a.days[1].listeners.click();a.pending[1].resolve(Response.json(a.data(30)));await new Promise(setImmediate);
    if(outcome==='network')a.pending[0].reject(Error('offline'));else a.pending[0].resolve(outcome==='success'?Response.json(a.data(7)):Response.json({error:'old failure'},{status:502}));
    await new Promise(setImmediate);assert.equal(a.element('#total-views').textContent,'30');assert.equal(a.element('#dashboard').hidden,false);assert.equal(a.element('#status').textContent,'Cloudflare traffic loaded.');
  }
});

test('a stale analytics completion cannot reenable refresh while the latest request is pending',async()=>{
  const a=adminBrowser();a.days[1].listeners.click();a.pending[0].resolve(Response.json(a.data(7)));await new Promise(setImmediate);
  assert.equal(a.element('#refresh').disabled,true);assert.equal(a.element('#total-views').textContent,'');
  a.pending[1].resolve(Response.json(a.data(30)));await new Promise(setImmediate);assert.equal(a.element('#refresh').disabled,false);
});
