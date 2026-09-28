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
