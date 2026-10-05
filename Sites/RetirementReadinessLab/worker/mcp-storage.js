import { CALCULATION_LEASE_MS } from './mcp-execution.js';

export class UsageLimitError extends Error {
  constructor(code) { super(code); this.code = code; }
}
export async function accountKey(id) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode('retirement-mcp-v1:' + id));
  return Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, '0')).join('');
}
export function validRequestId(id) {
  return typeof id === 'string' && id.length > 0 && id.length <= 256 || typeof id === 'number' && Number.isSafeInteger(id);
}
async function requestKey(id) { return accountKey('request:' + JSON.stringify(id)); }
export async function startExecution(env, usage, id, now = Date.now()) {
  const key = validRequestId(id) ? await requestKey(id) : '';
  const result = await env.DB.prepare(`INSERT INTO mcp_execution (account_key, lease, request_key, cancelled, expires_at)
    VALUES (?, ?, ?, 0, ?) ON CONFLICT(account_key) DO UPDATE SET lease=excluded.lease,
    request_key=excluded.request_key, cancelled=0, expires_at=excluded.expires_at`).bind(usage.key, usage.lease, key, now + CALCULATION_LEASE_MS).run();
  if (!result.success) throw new Error('MCP storage unavailable');
}
export async function cancelExecution(env, userId, id, now = Date.now()) {
  if (!validRequestId(id)) return;
  const result = await env.DB.prepare('UPDATE mcp_execution SET cancelled = 1 WHERE account_key = ? AND request_key = ? AND expires_at > ?').bind(await accountKey(userId), await requestKey(id), now).run();
  if (!result.success) throw new Error('MCP storage unavailable');
}
export async function acquireUsage(env, id, pro, now = Date.now()) {
  if (!env.DB) throw new Error('MCP storage unavailable');
  const key = await accountKey(id), hour = Math.floor(now / 3600000), lease = crypto.randomUUID(), limit = pro ? 60 : 20;
  // One conditional UPSERT arbitrates both the hourly quota and active lease.
  // The lease survives hour boundaries and cannot be replaced until expiry.
  const result = await env.DB.prepare(`INSERT INTO mcp_usage (account_key, hour, calls, lease, lease_until, expires_at)
    VALUES (?, ?, 1, ?, ?, ?)
    ON CONFLICT(account_key) DO UPDATE SET hour=excluded.hour,
      calls=CASE WHEN mcp_usage.hour=excluded.hour THEN mcp_usage.calls+1 ELSE 1 END,
      lease=excluded.lease, lease_until=excluded.lease_until, expires_at=excluded.expires_at
    WHERE mcp_usage.lease_until <= ? AND (mcp_usage.hour != excluded.hour OR mcp_usage.calls < ?)
    RETURNING account_key`).bind(key, hour, lease, now + CALCULATION_LEASE_MS, now + 172800000, now, limit).all();
  if (!result.success) throw new Error('MCP storage unavailable');
  if (!result.results?.length) {
    const row = await env.DB.prepare('SELECT lease_until FROM mcp_usage WHERE account_key = ?').bind(key).first();
    throw new UsageLimitError(row?.lease_until > now ? 'calculation_in_progress' : 'hourly_limit');
  }
  return { key, lease };
}
export async function finishUsage(env, usage, tool, outcome, elapsedMs, now = Date.now()) {
  const band = elapsedMs < 1000 ? 'under_1s' : elapsedMs < 5000 ? '1_to_5s' : elapsedMs < 10000 ? '5_to_10s' : elapsedMs < 20000 ? '10_to_20s' : '20s_or_more';
  const statements = [];
  if (usage) statements.push(env.DB.prepare('UPDATE mcp_usage SET lease_until = 0, lease = ? WHERE account_key = ? AND lease = ?').bind('', usage.key, usage.lease));
  if (usage) statements.push(env.DB.prepare('DELETE FROM mcp_execution WHERE account_key = ? AND lease = ?').bind(usage.key, usage.lease));
  statements.push(env.DB.prepare(`INSERT INTO mcp_daily (date, tool, outcome, duration_band, count) VALUES (?, ?, ?, ?, 1)
    ON CONFLICT(date, tool, outcome, duration_band) DO UPDATE SET count=count+1`).bind(new Date(now).toISOString().slice(0, 10), tool, outcome, band));
  statements.push(env.DB.prepare('DELETE FROM mcp_usage WHERE expires_at <= ?').bind(now));
  statements.push(env.DB.prepare('DELETE FROM mcp_execution WHERE expires_at <= ?').bind(now));
  const results = await env.DB.batch(statements);
  if (results.some(r => !r.success)) throw new Error('MCP storage unavailable');
}
export async function mcpMetrics(env, window) {
  if (!env.DB) throw new Error('MCP storage unavailable');
  const result = await env.DB.prepare('SELECT tool, outcome, duration_band, SUM(count) AS count FROM mcp_daily WHERE date >= ? AND date < ? GROUP BY tool, outcome, duration_band ORDER BY tool, outcome, duration_band').bind(window.start, window.end).all();
  if (!result.success) throw new Error('MCP storage unavailable');
  const rows = result.results || [];
  return { calls: rows.reduce((n, r) => n + r.count, 0), completed: rows.filter(r => r.outcome === 'completed').reduce((n, r) => n + r.count, 0), rows, generalAccessEnabled: env.MCP_CALCULATIONS_ENABLED === 'true' };
}
