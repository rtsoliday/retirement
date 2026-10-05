import { calculateForecast } from '../worker/mcp.js';
import { simulationScenarios } from '../tests/fixtures/simulation-scenarios.js';
import { prepareCalendarScenario } from '../dist/model.js';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

// Synthetic fixtures only. Never print or save user inputs/results.
const options=Object.fromEntries(process.argv.slice(2).map(arg=>arg.replace(/^--/,'').split('=')));
const repeats=Number(options.repeats??1),warmups=Number(options.warmups??(options.baseline?1:0));
if(!Number.isInteger(repeats)||repeats<1||repeats>10||!Number.isInteger(warmups)||warmups<0||warmups>3)throw Error('Use --repeats=1..10 and --warmups=0..3.');
const baseline=options.baseline?(await import(pathToFileURL(options.baseline))).runSimulationAsync:undefined;
const selected = simulationScenarios().filter(([name]) => options.scenario?name===options.scenario:['pooled', 'separate-couple', 'employer-roth'].includes(name));
if(!selected.length)throw Error('Unknown --scenario; use a name from tests/fixtures/simulation-scenarios.js.');
const median=values=>[...values].sort((a,b)=>a-b)[Math.floor(values.length/2)];
const request = new Request('https://retirementforecast.us/mcp');
for (const [name, original] of selected) {
  const first = structuredClone(original), second = structuredClone(original);
  second.spending.annualBaseSpending *= .9;
  for (const s of [first, second]) { prepareCalendarScenario(s, { today: '2026-10-05', needsReview: false }); delete s.seed; delete s.numberOfSimulations; s.household.asOfDate = '2026-10-05'; }
  const runners=baseline?[['before',baseline],['after',undefined]]:[['current',undefined]],timings=Object.fromEntries(runners.map(([label])=>[label,[]]));
  const args={ schemaVersion: '1.0', processingAcknowledged: true, pathCount: 1000, forecastDate: '2026-10-05', scenarios: [first, second] };
  let peak=process.memoryUsage().heapUsed,expected,responseBytes=0;
  async function measure(label,runner,record){
    const timer=setInterval(()=>{peak=Math.max(peak,process.memoryUsage().heapUsed);},0),start=performance.now();
    try{
      const result=await calculateForecast(args,{tier:'pro',maxPaths:1000},request,runner),elapsed=performance.now()-start;
      if(expected)assert.deepEqual(result,expected,`${name}: complete forecast summaries changed`);else expected=result;
      responseBytes=new TextEncoder().encode(JSON.stringify(result)).length;
      if(record)timings[label].push(elapsed);
    }finally{clearInterval(timer);}
  }
  const start=performance.now();
  try {
    for(let i=0;i<warmups;i++)for(const[label,runner]of runners)await measure(label,runner,false);
    for(let i=0;i<repeats;i++)for(const[label,runner]of(i%2?[...runners].reverse():runners))await measure(label,runner,true);
    const milliseconds=Object.fromEntries(Object.entries(timings).map(([label,values])=>[label,Math.round(median(values))]));
    console.log(JSON.stringify({scenario:name,pathsPerScenario:1000,scenarios:2,repeats,warmups,elapsedMs:milliseconds.after??milliseconds.current,medianMilliseconds:milliseconds,...(baseline?{percentLessTime:Math.round(100*(1-median(timings.after)/median(timings.before))),exactResults:true}:{}),peakNodeHeapMiB:Math.round(peak/1048576),responseBytes,localOnly:true}));
  } catch (e) { console.log(JSON.stringify({ scenario: name, elapsedMs: Math.round(performance.now() - start), error: e.code || 'unavailable', peakNodeHeapMiB: Math.round(peak / 1048576), localOnly: true })); process.exitCode = 1; }
}
