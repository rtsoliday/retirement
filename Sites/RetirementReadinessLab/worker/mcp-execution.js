// Match the account lease's fallback lifetime. If the host terminates JavaScript
// before finally runs, a later request can reclaim the isolate after expiry.
export const CALCULATION_LEASE_MS = 120000;
export class CalculationGate {
  constructor() { this.active = null; }
  claim(now = Date.now()) {
    if (this.active && this.active.expiresAt > now) return null;
    const claim = { expiresAt: now + CALCULATION_LEASE_MS };
    this.active = claim;
    return claim;
  }
  owns(claim) { return this.active === claim; }
  release(claim) { if (this.owns(claim)) this.active = null; }
}
export const calculationGate = new CalculationGate();
export const EXECUTION_CHECK_SQL = 'SELECT cancelled FROM mcp_execution WHERE account_key = ? AND lease = ?';

export function hostedExecution(env, isOwner, isActive, usage) {
  const requested = Number(env.MCP_VERIFICATION_DEADLINE_MS);
  // Owner-only verification may shorten the deadline while general access is
  // disabled. It can never raise the public 20-second computation limit.
  const deadlineMs = isOwner && env.MCP_CALCULATIONS_ENABLED !== 'true' &&
    Number.isInteger(requested) && requested >= 1 && requested <= 20000 ? requested : 20000;
  let batches = 0, cancelled = false;
  return {
    deadlineMs, isActive, isCancelled: () => cancelled,
    async yieldExecution({ done }) {
      // Date.now() only advances after real I/O on the host. Refresh it after
      // every 100 paths and after summaries, without sending scenario data.
      if (done || ++batches % 4 === 0) {
        if (usage) {
          const result = await env.DB.prepare(EXECUTION_CHECK_SQL).bind(usage.key, usage.lease).first();
          cancelled = !result || result.cancelled === 1;
        } else {
          const result = await env.DB.prepare('SELECT 1 AS ready').first();
          if (result?.ready !== 1) throw new Error('MCP clock checkpoint unavailable');
        }
      } else await new Promise(resolve => setTimeout(resolve, 0));
    },
  };
}
