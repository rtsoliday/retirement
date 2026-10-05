// Only aggregate events and short-lived random replay receipts are stored.
// No account identity, IP, scenario, or full referral URL reaches this database.
export function referralSource(value, ownHost) {
  if (!value) return 'Direct / unknown';
  try {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol)) return 'Direct / unknown';
    const host = url.hostname.toLowerCase();
    if (host === ownHost || ['retirementforecast.us', 'www.retirementforecast.us'].includes(host)) return 'Internal navigation';
    return host.slice(0, 253);
  } catch { return 'Direct / unknown'; }
}

export async function recordMetric(request, env, isOwner, json) {
  if (request.method !== 'POST') return json({ error: 'Method not allowed' }, 405);
  const url = new URL(request.url);
  if (request.headers.get('origin') !== url.origin) return json({ error: 'Same-origin request required' }, 403);
  if (!env.DB) return json({ error: 'Usage measurement unavailable' }, 503);
  // Reject arbitrary bodies instead of accepting financial data accidentally.
  if (!request.headers.get('content-type')?.startsWith('application/json')) return json({ error: 'JSON required' }, 415);
  if (Number(request.headers.get('content-length') || 0) > 1024) return json({ error: 'Event too large' }, 413);
  let event;
  try {
    const body = await request.text();
    if (body.length > 1024) return json({ error: 'Event too large' }, 413);
    event = JSON.parse(body);
  } catch { return json({ error: 'Invalid event' }, 400); }
  if (!event || Object.keys(event).some(key => !['id', 'event', 'referrer'].includes(key)) ||
      !/^[a-f0-9-]{36}$/i.test(event.id || '') || !['arrival', 'simulation_start'].includes(event.event) ||
      (event.referrer !== undefined && (typeof event.referrer !== 'string' || event.referrer.length > 253)) ||
      (event.event === 'simulation_start' && event.referrer !== undefined)) return json({ error: 'Invalid event' }, 400);
  if (isOwner) return new Response(null, { status: 204, headers: { 'Cache-Control': 'no-store' } });
  const date = new Date().toISOString().slice(0, 10);
  const source = event.event === 'arrival' ? referralSource(event.referrer ? `https://${event.referrer}` : '', url.hostname) : '';
  try {
    // D1 batch is atomic; changes() ensures retries increment only once.
    await env.DB.batch([
      env.DB.prepare('INSERT OR IGNORE INTO metric_receipts (id, date) VALUES (?, ?)').bind(event.id, date),
      env.DB.prepare('INSERT INTO metric_daily (date, event, source, count) SELECT ?, ?, ?, 1 WHERE changes() = 1 ON CONFLICT(date, event, source) DO UPDATE SET count = count + 1').bind(date, event.event, source),
      env.DB.prepare("DELETE FROM metric_receipts WHERE date < date('now', '-35 days')"),
    ]);
    return new Response(null, { status: 204, headers: { 'Cache-Control': 'no-store' } });
  } catch { return json({ error: 'Usage measurement temporarily unavailable' }, 503); }
}

export async function usageMetrics(env, window, days) {
  if (!env.DB) throw new Error('Usage measurement is not connected yet.');
  const data = await env.DB.prepare('SELECT date, event, source, count FROM metric_daily WHERE date >= ? AND date < ? ORDER BY date').bind(window.start, window.end).all();
  if (!data.success || !Array.isArray(data.results)) throw new Error('Usage measurement temporarily unavailable.');
  const first = await env.DB.prepare('SELECT MIN(date) AS date FROM metric_daily').first();
  const byDate = new Map(), sources = new Map();
  for (const row of data.results) {
    if (row.event === 'simulation_start') byDate.set(row.date, (byDate.get(row.date) || 0) + row.count);
    if (row.event === 'arrival' && row.source !== 'Internal navigation') sources.set(row.source, (sources.get(row.source) || 0) + row.count);
  }
  const series = Array.from({ length: days }, (_, i) => {
    const date = new Date(Date.parse(window.start) + i * 86400000).toISOString().slice(0, 10);
    return { date, starts: byDate.get(date) || 0 };
  });
  return { series, simulationStarts: series.reduce((sum, row) => sum + row.starts, 0),
    firstRecordedDate: first?.date || null,
    referrals: [...sources].map(([source, arrivals]) => ({ source, arrivals })).sort((a, b) => b.arrivals - a.arrivals || a.source.localeCompare(b.source)) };
}
