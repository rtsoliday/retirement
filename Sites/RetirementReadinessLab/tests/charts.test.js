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
  const s=baseScenario();s.numberOfSimulations=50;s.spending.annualBaseSpending=120000;
  const a=runSimulation(s),b=runSimulation(s,()=>{},{includePathPoints:false});
  assert.ok(a.pathPoints.some(p=>!p.successfulPath));assert.ok(a.pathPoints.every(p=>p.balance>0));
  for(const m of a.meanPath){const points=a.pathPoints.filter(p=>p.yearsInRetirement===m.yearsInRetirement);assert.ok(Math.abs(m.balance-points.reduce((sum,p)=>sum+p.balance,0)/points.length)<.001);}
  delete a.generatedAtEpochMillis;delete b.generatedAtEpochMillis;delete a.pathPoints;delete b.pathPoints;
  assert.ok(a.todayDollars.pathPoints.length);assert.deepEqual(b.todayDollars.pathPoints,[]);delete a.todayDollars.pathPoints;delete b.todayDollars.pathPoints;assert.deepEqual(a,b);
});

test('scatter dots at the final age stay visible while points outside a zoomed viewport stay clipped',()=>{
  const originalObserver=globalThis.ResizeObserver,originalWindow=globalThis.window,originalDocument=globalThis.document;
  globalThis.window={devicePixelRatio:1};
  globalThis.ResizeObserver=class{constructor(callback){this.callback=callback;}observe(){this.callback();}disconnect(){}};
  try{
    let shape=[],fills=[];
    const ctx=new Proxy({beginPath:()=>{shape=[];},arc:(x,y)=>shape.push({x,y,circle:true}),moveTo:(x,y)=>shape.push({x,y}),lineTo:(x,y)=>shape.push({x,y}),fill:()=>fills.push([...shape])},{get:(o,k)=>o[k]??(()=>{})});
    const canvas={getContext:()=>ctx,setAttribute(){},getAttribute:()=>'',classList:{add(){}},setPointerCapture(){},getBoundingClientRect:()=>({width:600,height:300,left:0,top:0})};
    const slider={},output={},caption={textContent:''},el={dataset:{plot:'paths'},querySelector:q=>q==='canvas'?canvas:q==='input[type=range]'?slider:q==='output'?output:caption};
    const result={provenance:{simulationCount:100},pathPoints:[{yearsInRetirement:0,balance:100,successfulPath:true},{yearsInRetirement:10,balance:200,successfulPath:true},{yearsInRetirement:10,balance:300,successfulPath:false}],meanPath:[]};
    const assertEndpoints=()=>{assert.equal(fills.length,3);for(const fill of fills)assert.ok(fill.every(p=>p.x>=68&&p.x<=584));assert.ok(fills[1][0].circle);assert.equal(fills[2].length,3);};
    mountCharts({querySelectorAll:q=>q==='[data-plot]'?[el]:[]},result,65);
    assertEndpoints();assert.equal(slider.max,75);assert.match(caption.textContent,/Showing 3 sampled points/);
    disposeCharts();fills=[];
    let expand;const controls=new Map(),button={dataset:{expandPlot:'paths'},addEventListener(_event,fn){expand=fn;},focus(){}};
    const dialog={showModal(){},close(){},remove(){},addEventListener(){},querySelector:q=>q==='[data-plot]'?el:q==='[data-expand-plot]'||q==='.plot-card .section-heading'?{remove(){}}:(controls.has(q)?controls.get(q):(controls.set(q,{}),controls.get(q)))};
    globalThis.document={createElement:()=>dialog,body:{append(){}}};
    mountCharts({querySelectorAll:q=>q==='[data-expand-plot]'?[button]:[]},result,65);expand();
    fills=[];controls.get('[data-reset]').onclick();assertEndpoints();
    fills=[];controls.get('[data-zoom=in]').onclick();assert.equal(fills.length,0,'Off-screen ages must not be clamped into view');
    fills=[];controls.get('[data-reset]').onclick();assertEndpoints();
  }finally{disposeCharts();globalThis.ResizeObserver=originalObserver;globalThis.window=originalWindow;globalThis.document=originalDocument;}
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
    slider.value=829/12;slider.oninput();assert.match(output.textContent,/69 years 1 month · .*Median \$8,500/);
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
    assert.match(output.textContent,/65 years 11 months.*No shortfall observed 0 of 4.*Household still alive 4 of 4/);
    slider.value=66;slider.oninput();assert.match(output.textContent,/No shortfall observed 0 of 4.*Household still alive 0 of 4/);
  }finally{disposeCharts();globalThis.ResizeObserver=original;}
});

test('funding and survival strokes and shading change only at event ages, including zoom and pan',()=>{
  const originalObserver=globalThis.ResizeObserver,originalWindow=globalThis.window,originalDocument=globalThis.document;
  globalThis.window={devicePixelRatio:1};
  globalThis.ResizeObserver=class{constructor(callback){this.callback=callback;}observe(){this.callback();}disconnect(){}};
  try{
    let shape=[],strokes=[],fills=[];
    const ctx=new Proxy({beginPath:()=>{shape=[];},moveTo:(x,y)=>shape.push([x,y]),lineTo:(x,y)=>shape.push([x,y]),stroke(){strokes.push({color:this.strokeStyle,points:[...shape]});},fill(){fills.push({color:this.fillStyle,points:[...shape]});}},{get:(target,key)=>target[key]??(()=>{})});
    const canvas={getContext:()=>ctx,setAttribute(){},getAttribute:()=>'',classList:{add(){}},setPointerCapture(){},getBoundingClientRect:()=>({width:600,height:300,left:0,top:0})};
    const slider={},output={},caption={textContent:''},el={dataset:{plot:'survival'},querySelector:q=>q==='canvas'?canvas:q==='input[type=range]'?slider:q==='output'?output:caption};
    const result={provenance:{simulationCount:4},notFailedByAge:[{age:65,notFailedShare:1,aliveShare:1},{age:65.5,notFailedShare:0,aliveShare:1},{age:66,notFailedShare:0,aliveShare:0}]};
    const funded=()=>strokes.filter(s=>s.color==='#176b5b').at(-1).points;
    const alive=()=>strokes.filter(s=>s.color==='#b27615').at(-1).points;
    const shaded=()=>fills.filter(s=>s.color==='#176b5b').at(-1).points;
    const assertSteps=()=>{
      for(const path of [funded(),alive(),shaded()])for(let i=1;i<path.length;i++)assert.ok(path[i][0]===path[i-1][0]||path[i][1]===path[i-1][1],'An event series cannot contain a diagonal segment');
      assert.equal(funded()[1][0],funded()[2][0]);assert.equal(alive()[3][0],alive()[4][0]);
    };
    mountCharts({querySelectorAll:q=>q==='[data-plot]'?[el]:[]},result,65);
    // 100% and 0% sit 10px inside the 22–256px frame, off its top and bottom borders.
    assert.deepEqual(funded(),[[68,32],[326,32],[326,246],[584,246],[584,246]]);
    assert.deepEqual(alive(),[[68,32],[326,32],[326,32],[584,32],[584,246]]);
    assert.deepEqual(shaded(),[[68,246],[68,32],[326,32],[326,246],[584,246],[584,246],[584,246]]);assertSteps();
    disposeCharts();
    let expand;const controls=new Map(),button={dataset:{expandPlot:'survival'},addEventListener(_event,fn){expand=fn;},focus(){}};
    const dialog={showModal(){},close(){},remove(){},addEventListener(){},querySelector:q=>q==='[data-plot]'?el:q==='[data-expand-plot]'||q==='.plot-card .section-heading'?{remove(){}}:(controls.has(q)?controls.get(q):(controls.set(q,{}),controls.get(q)))};
    globalThis.document={createElement:()=>dialog,body:{append(){}}};
    mountCharts({querySelectorAll:q=>q==='[data-expand-plot]'?[button]:[]},result,65);expand();
    controls.get('[data-zoom=in]').onclick();assertSteps();
    const before=funded()[2][0];
    canvas.onpointerdown({clientX:326,clientY:100,pointerId:1});canvas.onpointermove({clientX:366,clientY:100});canvas.onpointerup({clientX:366,clientY:100});
    assert.equal(funded()[2][0],before+40);assertSteps();
    controls.get('[data-reset]').onclick();assert.equal(funded()[2][0],326);assertSteps();
  }finally{disposeCharts();globalThis.ResizeObserver=originalObserver;globalThis.window=originalWindow;globalThis.document=originalDocument;}
});

test('chart drawing and pointer inspection agree for partial-year spans, including zoom and pan',()=>{
  const originalObserver=globalThis.ResizeObserver,originalWindow=globalThis.window;
  globalThis.window={devicePixelRatio:1};
  globalThis.ResizeObserver=class{constructor(callback){this.callback=callback;}observe(){this.callback();}disconnect(){}};
  try{
    for(const type of ['bands','survival']){
      const points=[];
      const ctx=new Proxy({lineTo:(x,y)=>points.push({x,y})},{get:(target,key)=>target[key]??(()=>{})});
      const canvas={getContext:()=>ctx,setAttribute(){},getAttribute:()=>'',classList:{add(){}},setPointerCapture(){},getBoundingClientRect:()=>({width:600,height:300,left:0,top:0})};
      const click=x=>{const event={clientX:x,clientY:100,pointerId:1};canvas.onpointerdown(event);canvas.onpointerup(event);};
      const slider={},output={textContent:''},caption={textContent:''};
      const el={dataset:{plot:type},querySelector:q=>q==='canvas'?canvas:q==='input[type=range]'?slider:q==='output'?output:caption};
      const controls=new Map(),dialog={querySelector:q=>{if(!controls.has(q))controls.set(q,{});return controls.get(q);}};
      const root={querySelectorAll:q=>q==='[data-plot]'?[el]:[]};
      const result={provenance:{simulationCount:4},balanceBands:[69.5,69.75,70].map((age,i)=>({age,median:100-i*50,pessimistic:100-i*50,optimistic:100-i*50,pathCount:4})),notFailedByAge:[69.5,69.75,70].map((age,i)=>({age,notFailedShare:1-i/2,aliveShare:type==='bands'?1:1-i/2}))};
      mountCharts(root,result,69.5);
      assert.ok(points.some(p=>p.x===326),'The middle observation is drawn at the axis midpoint');
      for(const [age,x] of [[69.5,68],[69.75,326],[70,584]]){
        click(x);assert.equal(slider.value,age);
      }
      disposeCharts();
      // Open the real expanded chart through mountCharts' registered handler.
      let expand;
      const button={dataset:{expandPlot:type},addEventListener(_event,fn){expand=fn;},focus(){}};
      const expanded={...dialog,className:'',innerHTML:'',showModal(){},close(){},remove(){},addEventListener(){},querySelector:q=>q==='[data-plot]'?el:q==='[data-expand-plot]'||q==='.plot-card .section-heading'?{remove(){}}:dialog.querySelector(q)};
      const originalDocument=globalThis.document;globalThis.document={createElement:()=>expanded,body:{append(){}}};
      try{
        mountCharts({querySelectorAll:q=>q==='[data-expand-plot]'?[button]:[]},result,69.5);expand();
        controls.get('[data-zoom=in]').onclick();
        click(326);assert.equal(slider.value,69.75);
        canvas.onpointerdown({clientX:326,clientY:100,pointerId:1});
        canvas.onpointermove({clientX:366,clientY:100});canvas.onpointerup({clientX:366,clientY:100});
        click(366);assert.equal(slider.value,69.75);
        controls.get('[data-reset]').onclick();click(584);assert.equal(slider.value,70);
      }finally{disposeCharts();globalThis.document=originalDocument;}
    }
  }finally{disposeCharts();globalThis.ResizeObserver=originalObserver;globalThis.window=originalWindow;}
});

test('balance charts label the selected dollar basis on the axis and in age inspection',()=>{
  const originalObserver=globalThis.ResizeObserver,originalWindow=globalThis.window;
  globalThis.window={devicePixelRatio:1};
  globalThis.ResizeObserver=class{constructor(callback){this.callback=callback;}observe(){this.callback();}disconnect(){}};
  try{
    for(const [basis,label] of [['today','Today’s dollars'],[undefined,'Future dollars']]){
      const texts=[],ctx=new Proxy({fillText:text=>texts.push(text)},{get:(target,key)=>target[key]??(()=>{})});
      const canvas={getContext:()=>ctx,setAttribute(){},getAttribute:()=>'',classList:{add(){}},getBoundingClientRect:()=>({width:600,height:300,left:0,top:0})};
      const slider={},output={},caption={textContent:''},el={dataset:{plot:'bands'},querySelector:q=>q==='canvas'?canvas:q==='input[type=range]'?slider:q==='output'?output:caption};
      const result={provenance:{simulationCount:4},dollarBasis:basis,notFailedByAge:[{age:65,notFailedShare:1,aliveShare:1},{age:70,notFailedShare:1,aliveShare:1}],balanceBands:[{age:65,pessimistic:1,median:2,optimistic:3,pathCount:4},{age:70,pessimistic:1,median:2,optimistic:3,pathCount:4}]};
      mountCharts({querySelectorAll:q=>q==='[data-plot]'?[el]:[]},result,65);
      assert.ok(texts.includes(`${label} · linear scale`));assert.match(output.textContent,new RegExp(label+'$'));
      assert.doesNotMatch(caption.textContent,/Sample preview only/);
      disposeCharts();
    }
  }finally{disposeCharts();globalThis.ResizeObserver=originalObserver;globalThis.window=originalWindow;}
});