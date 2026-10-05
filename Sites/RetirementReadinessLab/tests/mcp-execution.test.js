import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import worker from '../worker/index.js';
import {CalculationGate, calculationGate, hostedExecution, CALCULATION_LEASE_MS,EXECUTION_CHECK_SQL} from '../worker/mcp-execution.js';
import {acquireUsage,startExecution,cancelExecution,finishUsage} from '../worker/mcp-storage.js';
import {verificationScenarios} from '../worker/verification-scenarios.js';
import {baseScenario} from '../dist/model.js';

const origin='https://retirementforecast.us',owner='028696a7-7846-4822-a4c1-67026aa2383f';
function database(checkpoint=async()=>{}){
  const db=new DatabaseSync(':memory:');
  for(const name of ['0000_opposite_morlocks','0001_chilly_ultron','0002_moaning_rhodey'])db.exec(readFileSync(new URL(`../drizzle/${name}.sql`,import.meta.url),'utf8'));
  return {db,prepare(sql){const statement=db.prepare(sql);let values=[];return {bind(...args){values=args;return this;},async all(){return {success:true,results:statement.all(...values)};},async first(){if(sql==='SELECT 1 AS ready'||sql===EXECUTION_CHECK_SQL)await checkpoint();return statement.get(...values);},run(){return {success:true,meta:statement.run(...values)};}};},async batch(statements){db.exec('BEGIN');try{const result=statements.map(s=>s.run());db.exec('COMMIT');return result;}catch(error){db.exec('ROLLBACK');throw error;}}};
}
function input(paths=100,paired=false){
  const s=baseScenario();delete s.seed;delete s.numberOfSimulations;
  Object.assign(s.household,{asOfDate:'2026-10-05',birthday:'1966-10-05',retirementDate:'2033-10-05'});
  return {schemaVersion:'1.0',processingAcknowledged:true,pathCount:paths,forecastDate:'2026-10-05',...(paired?{scenarios:[s,structuredClone(s)]}:{scenario:s})};
}
function request(args,signal){return new Request(origin+'/mcp',{method:'POST',signal,headers:{'Content-Type':'application/json',Accept:'application/json, text/event-stream','oai-authenticated-user-id':owner},body:JSON.stringify({jsonrpc:'2.0',id:1,method:'tools/call',params:{name:args.scenarios?'compare_retirement_scenarios':'create_retirement_forecast',arguments:args}})});}
async function result(response){const text=await response.text();return JSON.parse(response.headers.get('content-type')?.includes('event-stream')?text.split('\n').find(line=>line.startsWith('data: ')).slice(6):text).result;}
const code=r=>JSON.parse(r.content[0].text).code;
const assertReleased=DB=>assert.equal(DB.db.prepare('SELECT lease_until FROM mcp_usage').get().lease_until,0);

test('isolate gate recovers after termination without finally and an old release cannot clear a new claim',()=>{
  const gate=new CalculationGate(),old=gate.claim(1000);
  assert.equal(gate.claim(1001),null);
  assert.equal(gate.claim(1000+CALCULATION_LEASE_MS-1),null);
  // Simulate host termination: deliberately omit release(old).
  const next=gate.claim(1000+CALCULATION_LEASE_MS);
  assert.ok(next);assert.equal(gate.owns(old),false);
  gate.release(old);assert.equal(gate.owns(next),true);
  gate.release(next);assert.ok(gate.claim(1000+CALCULATION_LEASE_MS));
});

test('verification deadline only shortens owner calculations with general access disabled',()=>{
  for(const value of [undefined,'bad','0','-1','20001','1.5'])assert.equal(hostedExecution({MCP_VERIFICATION_DEADLINE_MS:value},true).deadlineMs,20000);
  for(const value of ['1','1000','20000']){
    assert.equal(hostedExecution({MCP_VERIFICATION_DEADLINE_MS:value},true).deadlineMs,Number(value));
    assert.equal(hostedExecution({MCP_VERIFICATION_DEADLINE_MS:value},false).deadlineMs,20000);
    assert.equal(hostedExecution({MCP_VERIFICATION_DEADLINE_MS:value,MCP_CALCULATIONS_ENABLED:'true'},true).deadlineMs,20000);
  }
});

test('host checkpoints use read-only I/O every 100 paths and after summaries, with no scenario data',async()=>{
  const queries=[],execution=hostedExecution({DB:{prepare(sql){queries.push(sql);return {async first(){return {ready:1};}};}}},true);
  for(let i=0;i<8;i++)await execution.yieldExecution({done:false});
  await execution.yieldExecution({done:true});
  assert.deepEqual(queries,['SELECT 1 AS ready','SELECT 1 AS ready','SELECT 1 AS ready']);
  await assert.rejects(hostedExecution({DB:{prepare(){return {async first(){return null;}};}}},true).yieldExecution({done:true}),/checkpoint unavailable/);
});

test('frozen host clock advances at I/O, times out a comparison without leaking its completed first scenario, and recovers',async()=>{
  const realNow=Date.now;let now=Date.UTC(2026,9,5),checks=0,advance=true;
  const DB=database(async()=>{if(advance)now+=++checks<=2?5000:11000;});
  try{
    Date.now=()=>now;
    const timed=await result(await worker.fetch(request(input(100,true)),{DB}));
    assert.equal(code(timed),'computation_limit');assert.equal(timed.structuredContent,undefined);
    assert.ok(!timed.content[0].text.includes('forecasts'));assert.equal(checks,3);
    assertReleased(DB);assert.equal(DB.db.prepare('SELECT outcome FROM mcp_daily').get().outcome,'deadline');
    advance=false;
    const recovered=await result(await worker.fetch(request(input(4)),{DB}));
    assert.equal(recovered.isError,undefined);assert.equal(recovered.structuredContent.pathCount,4);assertReleased(DB);
  }finally{Date.now=realNow;DB.db.close();}
});

test('cancellation during an I/O checkpoint releases account and isolate gates before the next call',async()=>{
  const controller=new AbortController();let abort=true;
  const DB=database(async()=>{if(abort)controller.abort();});
  try{
    const cancelled=await worker.fetch(request(input(100),controller.signal),{DB});
    // The MCP transport closes its stream when the client disconnects. The
    // internal cancellation outcome and lease cleanup still must complete.
    assert.doesNotMatch(await cancelled.text(),/structuredContent|forecasts/);
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(DB.db.prepare('SELECT outcome FROM mcp_daily').get().outcome,'cancelled');assertReleased(DB);
    abort=false;const next=await result(await worker.fetch(request(input(4)),{DB}));
    assert.equal(next.isError,undefined);assertReleased(DB);
  }finally{DB.db.close();}
});

test('checkpoint failure releases both gates and never echoes infrastructure error text',async()=>{
  let fail=true;const DB=database(async()=>{if(fail)throw Error('private infrastructure detail');});
  try{
    const failed=await result(await worker.fetch(request(input(4)),{DB}));
    assert.equal(code(failed),'unavailable');assert.equal(failed.structuredContent,undefined);
    assert.ok(!failed.content[0].text.includes('private infrastructure detail'));assertReleased(DB);
    fail=false;assert.equal((await result(await worker.fetch(request(input(4)),{DB}))).isError,undefined);assertReleased(DB);
  }finally{DB.db.close();}
});

test('overlapping owner requests admit one calculation, charge one quota call, and permit another after completion',async()=>{
  let notify,release,paused=false;
  const entered=new Promise(resolve=>{notify=resolve;}),resume=new Promise(resolve=>{release=resolve;});
  const DB=database(async()=>{if(!paused){paused=true;notify();await resume;}});
  try{
    const first=worker.fetch(request(input(100)),{DB});await entered;
    const rejected=await result(await worker.fetch(request(input(4)),{DB}));
    assert.equal(code(rejected),'server_busy');assert.equal(DB.db.prepare('SELECT calls FROM mcp_usage').get().calls,1);
    release();assert.equal((await result(await first)).isError,undefined);assertReleased(DB);
    assert.equal((await result(await worker.fetch(request(input(4)),{DB}))).isError,undefined);
    assert.equal(DB.db.prepare('SELECT calls FROM mcp_usage').get().calls,2);assertReleased(DB);
  }finally{release();DB.db.close();}
});

test('stateless MCP cancellation reaches the active request, records cancellation and recovers',async()=>{
  let entered,release,paused=false;
  const ready=new Promise(resolve=>{entered=resolve;}),resume=new Promise(resolve=>{release=resolve;});
  const DB=database(async()=>{if(!paused){paused=true;entered();await resume;}});
  try{
    const first=worker.fetch(request(input(100,true)),{DB});await ready;
    const notification=new Request(origin+'/mcp',{method:'POST',headers:{'Content-Type':'application/json',Accept:'application/json, text/event-stream','oai-authenticated-user-id':owner},body:JSON.stringify({jsonrpc:'2.0',method:'notifications/cancelled',params:{requestId:1,reason:'private reason'}})});
    assert.equal((await worker.fetch(notification,{DB})).status,202);
    assert.equal(DB.db.prepare('SELECT cancelled FROM mcp_execution').get().cancelled,1);
    assert.ok(!JSON.stringify(DB.db.prepare('SELECT * FROM mcp_execution').all()).includes('private reason'));
    release();const cancelled=await result(await first);
    assert.equal(code(cancelled),'cancelled');assert.equal(cancelled.structuredContent,undefined);assertReleased(DB);
    assert.equal(DB.db.prepare('SELECT COUNT(*) AS n FROM mcp_execution').get().n,0);
    assert.equal(DB.db.prepare('SELECT outcome FROM mcp_daily').get().outcome,'cancelled');
    assert.equal((await result(await worker.fetch(request(input(4)),{DB}))).isError,undefined);
  }finally{release();DB.db.close();}
});

test('cancellation is account scoped, matches the request ID type, expires, and never clears a replacement lease',async()=>{
  const DB=database(),env={DB},now=1000000;
  try{
    const usage=await acquireUsage(env,'private-account',true,now);await startExecution(env,usage,'private-request',now);
    for(const [user,id,time] of [['other-account','private-request',now],['private-account','wrong-request',now],['private-account',1,now],['private-account',{},now],['private-account','private-request',now+CALCULATION_LEASE_MS]]){
      await cancelExecution(env,user,id,time);assert.equal(DB.db.prepare('SELECT cancelled FROM mcp_execution').get().cancelled,0);
    }
    assert.doesNotMatch(JSON.stringify(DB.db.prepare('SELECT * FROM mcp_execution').all()),/private-account|private-request/);
    await cancelExecution(env,'private-account','private-request',now+1);assert.equal(DB.db.prepare('SELECT cancelled FROM mcp_execution').get().cancelled,1);
    const replacement=await acquireUsage(env,'private-account',true,now+CALCULATION_LEASE_MS);await startExecution(env,replacement,2,now+CALCULATION_LEASE_MS);
    await finishUsage(env,usage,'create_retirement_forecast','cancelled',100,now+CALCULATION_LEASE_MS);
    assert.equal(DB.db.prepare('SELECT lease FROM mcp_execution').get().lease,replacement.lease);
    await cancelExecution(env,'private-account','2',now+CALCULATION_LEASE_MS+1);assert.equal(DB.db.prepare('SELECT cancelled FROM mcp_execution').get().cancelled,0);
    await cancelExecution(env,'private-account',2,now+CALCULATION_LEASE_MS+1);assert.equal(DB.db.prepare('SELECT cancelled FROM mcp_execution').get().cancelled,1);
    await finishUsage(env,replacement,'create_retirement_forecast','cancelled',100,now+CALCULATION_LEASE_MS+1);assert.equal(DB.db.prepare('SELECT COUNT(*) AS n FROM mcp_execution').get().n,0);
  }finally{DB.db.close();}
});

function verificationRequest(path='/api/admin/mcp-verification',user=owner,method='GET',body){return new Request(origin+path,{method,headers:{...(user?{'oai-authenticated-user-id':user}:{}),Origin:origin,'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});}
test('verification page and controls require the owner Site identity; controls fail closed under general access',async()=>{
  const DB=database();
  try{
    for(const path of ['/mcp-verification','/mcp-verification/','/mcp-verification.html','/api/admin/mcp-verification','/api/admin/mcp-verification/rpc'])for(const user of [null,'another-account'])assert.equal((await worker.fetch(verificationRequest(path,user),{DB,MCP_VERIFICATION_ENABLED:'true'})).status,403);
    const emailOnly=new Request(origin+'/mcp-verification',{headers:{'oai-authenticated-user-email':'rtsoliday@gmail.com'}});assert.equal((await worker.fetch(emailOnly,{DB})).status,403);
    const page=await worker.fetch(verificationRequest('/mcp-verification'),{DB});assert.equal(page.status,200);assert.equal(page.headers.get('cache-control'),'no-store');assert.match(page.headers.get('content-security-policy'),/frame-ancestors 'none'/);
    for(const extra of [{},{MCP_VERIFICATION_ENABLED:'true',MCP_CALCULATIONS_ENABLED:'true'}]){
      const status=await (await worker.fetch(verificationRequest(),{DB,...extra})).json();assert.equal(status.enabled,false);assert.equal(status.scenarios,undefined);
      assert.equal((await worker.fetch(verificationRequest(undefined,owner,'POST',{action:'abandon'}),{DB,...extra})).status,403);
    }
    const env={DB,MCP_VERIFICATION_ENABLED:'true'};
    const config=await (await worker.fetch(verificationRequest(),env)).json();assert.equal(config.scenarios.length,2);assert.equal(config.active,false);assert.equal(config.executionReady,false);
    for(const scenario of config.scenarios){assert.equal(scenario.args.pathCount,1000);assert.match(scenario.expectedHash,/^[a-f0-9]{64}$/);}
    const crossOrigin=verificationRequest(undefined,owner,'POST',{action:'abandon'});crossOrigin.headers.set('Origin','https://other.example');assert.equal((await worker.fetch(crossOrigin,env)).status,403);
    assert.equal((await worker.fetch(verificationRequest(undefined,owner,'POST',{action:'abandon',privatePlan:{}}),env)).status,400);
    assert.equal((await worker.fetch(verificationRequest(undefined,owner,'POST',{action:'abandon',extra:'a'.repeat(2048)}),env)).status,413);
    assert.equal(DB.db.prepare('SELECT COUNT(*) AS n FROM mcp_usage').get().n,0);
  }finally{DB.db.close();}
});

test('private diagnostic RPC runs the same MCP handler only for reviewed synthetic inputs',async()=>{
  const DB=database(),env={DB,MCP_VERIFICATION_ENABLED:'true'},path='/api/admin/mcp-verification/rpc';
  const args=structuredClone(verificationScenarios[1].args);args.pathCount=4;
  const body={jsonrpc:'2.0',id:3,method:'tools/call',params:{name:'compare_retirement_scenarios',arguments:args}};
  try{
    for(const extra of [{MCP_VERIFICATION_ENABLED:'false'},{MCP_CALCULATIONS_ENABLED:'true'}])assert.equal((await worker.fetch(verificationRequest(path,owner,'POST',body),{...env,...extra})).status,403);
    const bad=structuredClone(body);bad.params.arguments.scenarios[0].accounts.pretax++;assert.equal((await worker.fetch(verificationRequest(path,owner,'POST',bad),env)).status,400);
    const request=verificationRequest(path,owner,'POST',body);request.headers.set('Accept','application/json, text/event-stream');request.headers.set('mcp-protocol-version','2025-11-25');
    const output=await result(await worker.fetch(request,env));assert.equal(output.isError,undefined);assert.equal(output.structuredContent.pathCount,4);assertReleased(DB);
  }finally{DB.db.close();}
});

test('owner abandoned-lease control blocks until exactly two minutes and then calculates without redeployment',async()=>{
  const DB=database(),env={DB,MCP_VERIFICATION_ENABLED:'true'},realNow=Date.now;let now=Date.UTC(2026,9,5);
  try{
    Date.now=()=>now;const response=await worker.fetch(verificationRequest(undefined,owner,'POST',{action:'abandon'}),env),setup=await response.json();assert.equal(response.status,200);assert.equal(setup.simulated,true);assert.equal(setup.leaseMs,CALCULATION_LEASE_MS);
    const state=await (await worker.fetch(verificationRequest(),env)).json();assert.equal(state.active,true);assert.equal(state.executionReady,true);
    now+=CALCULATION_LEASE_MS-1;assert.equal(code(await result(await worker.fetch(request(input(4)),env))),'server_busy');
    assert.equal(DB.db.prepare('SELECT calls FROM mcp_usage').get().calls,1);
    now++;const recovered=await result(await worker.fetch(request(input(4)),env));assert.equal(recovered.isError,undefined);assertReleased(DB);
    assert.equal(DB.db.prepare('SELECT COUNT(*) AS n FROM mcp_execution').get().n,0);assert.equal(DB.db.prepare('SELECT calls FROM mcp_usage').get().calls,2);
  }finally{calculationGate.release(calculationGate.active);Date.now=realNow;DB.db.close();}
});
