// Best-effort measurement must never delay or interrupt a calculation.
export function reportUsage(event, referrer) {
  try {
    const payload = { id: crypto.randomUUID(), event };
    if (event === 'arrival') {
      try { payload.referrer = referrer ? new URL(referrer).hostname : ''; } catch { payload.referrer = ''; }
    }
    void fetch('/api/metrics/event', { method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      keepalive: true, signal: AbortSignal.timeout(5000) }).catch(() => {});
  } catch { /* Offline or unsupported browser: the planner still works. */ }
}
