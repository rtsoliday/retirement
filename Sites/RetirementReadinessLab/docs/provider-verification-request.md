# Hosted resource verification request

Prepared for the existing Retirement Forecast Site. This is a draft, not a sent support request.

Site: `appgprj_6ab98ae02cd481919b2a1d47edb8ee3c`  
Production URL: https://retirementforecast.us  
MCP: https://retirement-readiness-lab-web.rtsoliday123.chatgpt.site/mcp

Please provide the following evidence for the deployed Worker and identify the deployment, script version, effective runtime configuration and measurement method:

1. Confirm the effective CPU allowance is at least 30,000 ms. The packaged `dist/server/wrangler.json` requests `limits.cpu_ms: 30000`; we need confirmation that Sites applies it. Observing a CPU termination near 32,500 ms is not a configuration readback.
2. Measure peak isolate memory for each of the approved synthetic paired 1,000-path employer-Roth and separate-couple requests, including three repeated calls per case. The acceptance limit is strictly below 90 MiB (94,371,840 bytes). Report peak bytes and measurement scope; Node heap or local process RSS is insufficient.
3. In a controlled deployment of the same source and D1 schema, force a provider CPU termination before cleanup completes. Keep general calculation access disabled. Record the termination outcome and request ID, verify an immediate retry is rejected while the lease is active, then verify successful calculation after the two-minute lease expires without redeploying or restarting the Worker. Provide timestamps and version identifiers. Restore the normal CPU setting afterward.

Only use the reviewed synthetic payloads in `mcp-acceptance-requests.json`. Do not collect or include personal plans, financial request bodies, authorization headers, cookies, or account identifiers in the report. Provider wall time and CPU time should be supplied for the repeated runs alongside memory. The owner verification page tests overlap, explicit MCP cancellation, HTTP disconnect and simulated abandoned leases; that simulation does not establish actual CPU-termination recovery.

General MCP calculations remain disabled until these resource requirements and the available hosted acceptance checks pass.
