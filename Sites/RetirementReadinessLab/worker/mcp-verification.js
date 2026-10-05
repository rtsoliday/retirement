import { calculationGate, CALCULATION_LEASE_MS } from './mcp-execution.js';
import { acquireUsage, accountKey, startExecution, finishUsage } from './mcp-storage.js';
import { verificationScenarios } from './verification-scenarios.js';
import { mcp } from './mcp.js';

export const verificationEnabled = env => env.MCP_VERIFICATION_ENABLED === 'true' && env.MCP_CALCULATIONS_ENABLED !== 'true';
const json = (body, status = 200) => Response.json(body, {status, headers:{'Cache-Control':'no-store'}});
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])])) : value;
export async function verificationRpc(request,env,isOwner) {
  if (!isOwner || !request.headers.get('oai-authenticated-user-id') || !verificationEnabled(env)) return json({error:'Owner verification controls are disabled'},403);
  if (request.method !== 'POST') return json({error:'Use POST'},405);
  if (request.headers.get('origin') !== new URL(request.url).origin) return json({error:'Same-origin request required'},403);
  // The custom domain reserves /mcp for the provider. This private diagnostic
  // route exercises the identical handler and accepts only reviewed samples.
  let body;try { const text=await request.clone().text();if(text.length>131072)return json({error:'Request too large'},413);body=JSON.parse(text); } catch {return json({error:'Invalid JSON'},400);}
  if(body?.method !== 'notifications/cancelled') {
    const args=body?.params?.arguments;
    if(body?.method!=='tools/call'||body.params?.name!=='compare_retirement_scenarios'||![4,1000].includes(args?.pathCount)||!verificationScenarios.some(sample=>JSON.stringify(canonical({...args,pathCount:1000}))===JSON.stringify(canonical(sample.args))))return json({error:'Choose a reviewed synthetic comparison'},400);
  }
  return mcp(request,env,true);
}
export async function verificationStatus(request, env) {
  const key = await accountKey(request.headers.get('oai-authenticated-user-id'));
  const row = await env.DB.prepare('SELECT lease_until, lease, calls, hour FROM mcp_usage WHERE account_key = ?').bind(key).first();
  const execution = await env.DB.prepare('SELECT expires_at FROM mcp_execution WHERE account_key = ? AND lease = ?').bind(key,row?.lease || '').first();
  const counts = await env.DB.prepare('SELECT outcome, SUM(count) AS count FROM mcp_daily WHERE date = ? AND tool = ? GROUP BY outcome').bind(new Date().toISOString().slice(0,10),'compare_retirement_scenarios').all();
  if (!counts.success) throw new Error('Verification unavailable');
  return {enabled:verificationEnabled(env),active:row?.lease_until > Date.now(),executionReady:execution?.expires_at > Date.now(),leaseExpiresAt:row?.lease_until || 0,
    remainingMs:Math.max(0,(row?.lease_until || 0)-Date.now()),calls:row?.hour===Math.floor(Date.now()/3600000)?row.calls:0,
    hourlyLimit:60,outcomes:Object.fromEntries((counts.results || []).map(r=>[r.outcome,r.count]))};
}
export async function verificationApi(request, env, isOwner) {
  if (!isOwner || !request.headers.get('oai-authenticated-user-id')) return json({error:'Owner sign-in required'},403);
  try {
    if (request.method === 'GET') {
      const status=await verificationStatus(request,env);
      return json({...status,...(status.enabled?{scenarios:verificationScenarios}: {})});
    }
    if (request.method !== 'POST') return json({error:'Use GET or POST'},405);
    if (!verificationEnabled(env)) return json({error:'Owner verification controls are disabled'},403);
    if (request.headers.get('origin') !== new URL(request.url).origin || !request.headers.get('content-type')?.startsWith('application/json')) return json({error:'Same-origin JSON request required'},403);
    const body=await request.text();if(body.length>2048)return json({error:'Request too large'},413);
    let input;try{input=JSON.parse(body);}catch{return json({error:'Invalid JSON'},400);}
    if(input?.action!=='abandon' || Object.keys(input).length!==1)return json({error:'Choose the abandoned-lease test'},400);
    const claim=calculationGate.claim();if(!claim)return json({error:'A calculation is active'},409);
    let usage;
    try{
      usage=await acquireUsage(env,request.headers.get('oai-authenticated-user-id'),true);
      await startExecution(env,usage,'verification-abandoned');
    }catch(error){
      calculationGate.release(claim);
      if(usage)await finishUsage(env,usage,'owner_verification','unavailable',0);
      return json({error:error.code==='hourly_limit'?'Hourly allowance reached':'Verification unavailable'},409);
    }
    // Deliberately omit both releases to simulate termination before finally.
    // This only affects the authenticated owner during disabled general access.
    return json({simulated:true,leaseMs:CALCULATION_LEASE_MS,expiresAt:claim.expiresAt});
  }catch{return json({error:'Verification unavailable'},503);}
}
