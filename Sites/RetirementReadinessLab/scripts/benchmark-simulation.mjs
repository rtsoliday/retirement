import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {runSimulation} from '../dist/engine.js';
import {simulationScenarios} from '../tests/fixtures/simulation-scenarios.js';

const args=Object.fromEntries(process.argv.slice(2).map(arg=>arg.replace(/^--/,'').split('=')));
const paths=Number(args.paths??1000),repeats=Number(args.repeats??5);
if(!Number.isInteger(paths)||paths<4||paths>10000||!Number.isInteger(repeats)||repeats<1)throw Error('Use --paths=4..10000 and --repeats=1 or more.');
const baseline=args.baseline?(await import(pathToFileURL(args.baseline))).runSimulation:null;
const median=values=>[...values].sort((a,b)=>a-b)[Math.floor(values.length/2)];
const selected=simulationScenarios().filter(([name])=>args.scenario?name===args.scenario:['pooled','separate-couple','employer-roth'].includes(name));
if(!selected.length)throw Error('Unknown --scenario; use a name from tests/fixtures/simulation-scenarios.js.');
for(const [name,s] of selected){
  s.numberOfSimulations=paths;
  const runners=baseline?[['before',baseline],['after',runSimulation]]:[['current',runSimulation]];
  let expected;const timings=Object.fromEntries(runners.map(([label])=>[label,[]]));
  function measure(label,run,record){
    const start=performance.now(),result=run(s),elapsed=performance.now()-start;
    delete result.generatedAtEpochMillis;
    if(expected)assert.deepEqual(result,expected,`${name}: complete results changed`);else expected=result;
    if(record)timings[label].push(elapsed);
  }
  for(let warmup=0;warmup<2;warmup++)for(const [label,run] of runners)measure(label,run,false);
  for(let i=0;i<repeats;i++)for(const [label,run] of (i%2?[...runners].reverse():runners))measure(label,run,true);
  const milliseconds=Object.fromEntries(Object.entries(timings).map(([label,times])=>[label,Math.round(median(times))]));
  console.log(JSON.stringify({scenario:name,paths,repeats,medianMilliseconds:milliseconds,...(baseline?{percentLessTime:Math.round(100*(1-median(timings.after)/median(timings.before))),exactResults:true}:{})}));
}
