import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {baseScenario} from '../dist/model.js';
import {searchDecision} from '../dist/engine.js';
import {searchReadinessFrontier,searchSavingsFrontier} from '../dist/target-frontier.js';

test('the actual target worker uses 200 paths even when the plan selects fewer paths',async()=>{
  const s=baseScenario();s.numberOfSimulations=4;
  const messages=[],candidates=[],self={navigator:{hardwareConcurrency:1},postMessage:message=>messages.push(message)};
  const source=readFileSync(new URL('../dist/worker.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replaceAll('import.meta.url',JSON.stringify(new URL('../dist/worker.js',import.meta.url).href));
  vm.runInNewContext(source,{self,searchDecision,searchReadinessFrontier,candidateReadiness:(_scenario,task)=>{candidates.push(task);return .8;}});
  await self.onmessage({data:{task:'decision',scenario:s}});
  const result=messages.find(message=>message.type==='result')?.result;
  assert.ok(result);assert.equal(result.simulationCount,200);assert.equal(result.targetReadiness,.8);
  assert.equal(result.frontier.simulationCount,200);
  assert.equal(candidates.length,2+result.frontier.points.length);assert.ok(candidates.every(task=>task.count===200));
  assert.ok(result.frontier.points.every(point=>point.readiness>=.8));
  assert.equal(s.numberOfSimulations,4);
});

test('200-path screening rejects the 41st failure and fully evaluates an exactly 80% candidate',()=>{
  // Exercise the real private screening function with known path outcomes.
  const source=readFileSync(new URL('../dist/engine.js',import.meta.url),'utf8');
  const start=source.indexOf('function screenedReadiness('),end=source.indexOf('// Keep the plan',start);
  for(const failures of [40,41]){
    let calls=0;
    const screen=vm.runInNewContext(source.slice(start,end)+';screenedReadiness',{
      validateScenario:()=>[],JavaRandom:class{},STRIDE:1n,runOne:()=>({success:++calls>failures})
    });
    assert.equal(screen({seed:0},200,.8),failures===40?.8:0);
    assert.equal(calls,failures===40?200:41);
  }
});

test('the claiming target worker tests only claiming/spending pairs with 200 paths and preserves the full-run settings',async()=>{
  const s=baseScenario();s.numberOfSimulations=10000;const before=structuredClone(s),messages=[],candidates=[];
  const self={navigator:{hardwareConcurrency:1},postMessage:message=>messages.push(message)};
  const source=readFileSync(new URL('../dist/worker.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replaceAll('import.meta.url',JSON.stringify(new URL('../dist/worker.js',import.meta.url).href));
  vm.runInNewContext(source,{self,searchDecision,searchReadinessFrontier,candidateReadiness:(_scenario,task)=>{candidates.push(task);return .8;}});
  await self.onmessage({data:{task:'claim-decision',scenario:s}});
  const frontier=messages.find(message=>message.type==='result')?.result.frontier;
  assert.ok(frontier);assert.equal(frontier.kind,'claiming');assert.equal(frontier.simulationCount,200);
  assert.deepEqual(frontier.points.map(point=>point.age),[62,63,64,65,66,67,68,69,70]);
  assert.ok(candidates.every(task=>task.kind==='claim-frontier'&&task.count===200));
  assert.ok(frontier.points.every(point=>point.tested&&point.simulationCount===200));assert.deepEqual(s,before);
});

test('savings target worker evaluates retirement and savings pairs with 200 paths, preserving full-run settings',async()=>{
  const s=baseScenario(),before=structuredClone(s),messages=[],candidates=[];const self={navigator:{hardwareConcurrency:1},postMessage:m=>messages.push(m)};
  const source=readFileSync(new URL('../dist/worker.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replaceAll('import.meta.url',JSON.stringify(new URL('../dist/worker.js',import.meta.url).href));
  vm.runInNewContext(source,{self,searchDecision,searchReadinessFrontier,searchSavingsFrontier,candidateReadiness:(_s,task)=>{candidates.push(task);return task.value>=10000?.8:0;}});
  await self.onmessage({data:{task:'savings-decision',scenario:s}});const frontier=messages.find(m=>m.type==='result')?.result.frontier;
  assert.ok(frontier);assert.equal(frontier.kind,'savings');assert.equal(frontier.simulationCount,200);assert.ok(frontier.points.some(p=>p.age>s.household.retirementAge));
  assert.ok(candidates.every(t=>t.kind==='savings-frontier'&&t.count===200));assert.ok(frontier.points.every(p=>p.annualSavings===10000&&p.tested));assert.deepEqual(s,before);
});
