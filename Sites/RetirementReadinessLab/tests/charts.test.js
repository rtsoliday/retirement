import test from 'node:test';
import assert from 'node:assert/strict';
import {buildPathPoints,pathBounds,valueToFraction,fractionToValue} from '../dist/chart-data.js';
import {baseScenario} from '../dist/model.js';
import {runSimulation} from '../dist/engine.js';
import {mountCharts,disposeCharts} from '../dist/charts.js';

test('Android scatter classification uses whole-run outcome and strict separation at each age',()=>{
  const paths=[{success:true,chart:[100,200,300]},{success:false,chart:[100,50,0]},{success:true,chart:[80,40,-1]}];
  assert.deepEqual(buildPathPoints(paths),[
    {yearsInRetirement:0,balance:100,successfulPath:true,separatedFromOppositeOutcome:false},
    {yearsInRetirement:1,balance:200,successfulPath:true,separatedFromOppositeOutcome:true},
    {yearsInRetirement:2,balance:300,successfulPath:true,separatedFromOppositeOutcome:true},
    {yearsInRetirement:0,balance:100,successfulPath:false,separatedFromOppositeOutcome:false},
    {yearsInRetirement:1,balance:50,successfulPath:false,separatedFromOppositeOutcome:false},
    {yearsInRetirement:0,balance:80,successfulPath:true,separatedFromOppositeOutcome:false},
    {yearsInRetirement:1,balance:40,successfulPath:true,separatedFromOppositeOutcome:false}
  ]);
  assert.equal(buildPathPoints([{success:false,chart:[10]},{success:true,chart:[20]}])[0].separatedFromOppositeOutcome,true);
});
test('scatter sampling is deterministic, capped and classified before downsampling',()=>{
  const paths=Array.from({length:1000},(_,i)=>({success:i%2===0,chart:Array.from({length:80},(_,y)=>i+y+1)}));
  const points=buildPathPoints(paths);assert.equal(points.length,26667);assert.deepEqual(points,buildPathPoints(paths));
  assert.equal(points[1].yearsInRetirement,3);
  const sampled=buildPathPoints([{success:true,chart:[100,100]},{success:false,chart:[200,50]}],1);
  assert.equal(sampled[0].separatedFromOppositeOutcome,false);
});
test('log bounds match Android for empty, flat, small and wide balances',()=>{
  assert.deepEqual(pathBounds([],[]),{min:1,max:10,maxYear:1,logMin:0,logMax:Math.log(10)});
  const b=pathBounds([{balance:100,yearsInRetirement:0}],[{balance:100,yearsInRetirement:12}]);assert.equal(b.min,100);assert.equal(b.max,1000);assert.equal(b.maxYear,12);
  assert.equal(pathBounds([{balance:.01,yearsInRetirement:1}],[]).min,1);
  for(const log of [false,true])for(const fraction of [0,.25,.5,.75,1])assert.ok(Math.abs(valueToFraction(fractionToValue(fraction,1,1e9,log),1,1e9,log)-fraction)<1e-10);
});
test('scatter output does not change simulation results and mean uses observed positive balances',()=>{
  const s=baseScenario();s.numberOfSimulations=50;
  const a=runSimulation(s),b=runSimulation(s,()=>{},{includePathPoints:false});
  assert.ok(a.pathPoints.some(p=>!p.successfulPath));assert.ok(a.pathPoints.every(p=>p.balance>0));
  for(const m of a.meanPath){const points=a.pathPoints.filter(p=>p.yearsInRetirement===m.yearsInRetirement);assert.ok(Math.abs(m.balance-points.reduce((sum,p)=>sum+p.balance,0)/points.length)<.001);}
  delete a.generatedAtEpochMillis;delete b.generatedAtEpochMillis;delete a.pathPoints;delete b.pathPoints;assert.deepEqual(a,b);
});

test('balance chart inspection can reach a failure between annual observations',()=>{
  const original=globalThis.ResizeObserver;
  globalThis.ResizeObserver=class{observe(){}disconnect(){}};
  try{
    const ctx=new Proxy({},{get:(target,key)=>target[key]??(()=>{})});
    const canvas={getContext:()=>ctx,setAttribute(){},getAttribute:()=>'',getBoundingClientRect:()=>({width:600,height:300,left:0,top:0})};
    const slider={},output={textContent:''},caption={textContent:''};
    const el={dataset:{plot:'bands'},querySelector:selector=>selector==='canvas'?canvas:selector==='input[type=range]'?slider:selector==='output'?output:caption};
    const root={querySelectorAll:selector=>selector==='[data-plot]'?[el]:[]};
    const result={provenance:{simulationCount:4},balanceBands:[{age:829/12,median:8500,pessimistic:8500,optimistic:8500,pathCount:4},{age:69.75,median:0,pessimistic:0,optimistic:0,pathCount:4}]};
    mountCharts(root,result,829/12);assert.equal(slider.step,'any');
    slider.value=69.72;slider.oninput();assert.equal(slider.value,69.75);assert.match(output.textContent,/69 years 9 months.*Median \$0/);
    slider.value=829/12;slider.oninput();assert.match(output.textContent,/69 years 1 months.*Median \$8,500/);
  }finally{disposeCharts();globalThis.ResizeObserver=original;}
});

test('funding chart inspection reaches monthly failures and final death endpoints',()=>{
  const original=globalThis.ResizeObserver;
  globalThis.ResizeObserver=class{observe(){}disconnect(){}};
  try{
    const ctx=new Proxy({},{get:(target,key)=>target[key]??(()=>{})});
    const canvas={getContext:()=>ctx,setAttribute(){},getAttribute:()=>'',getBoundingClientRect:()=>({width:600,height:300,left:0,top:0})};
    const slider={},output={textContent:''},caption={textContent:''};
    const el={dataset:{plot:'survival'},querySelector:selector=>selector==='canvas'?canvas:selector==='input[type=range]'?slider:selector==='output'?output:caption};
    const root={querySelectorAll:selector=>selector==='[data-plot]'?[el]:[]};
    const result={provenance:{simulationCount:4},notFailedByAge:[
      {age:65,notFailedShare:1,aliveShare:1},
      {age:791/12,notFailedShare:0,aliveShare:1},
      {age:66,notFailedShare:0,aliveShare:0}
    ]};
    mountCharts(root,result,65);assert.equal(slider.step,'any');assert.equal(slider.max,66);
    slider.value=791/12;slider.oninput();
    assert.match(output.textContent,/65 years 11 months.*Still funded 0 of 4.*Still alive 4 of 4/);
    slider.value=66;slider.oninput();assert.match(output.textContent,/Still funded 0 of 4.*Still alive 0 of 4/);
  }finally{disposeCharts();globalThis.ResizeObserver=original;}
});
