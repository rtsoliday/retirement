import {isPreviewResult,shareLabel} from './result-format.js';
import {pathBounds,valueToFraction,fractionToValue} from './chart-data.js';
const colors={funded:'#176b5b',alive:'#b27615',success:'#288445',strong:'#38bd60',failure:'#7f1d1d',clear:'#e53935',mean:'#25333e',range:'#dceee6'};
const compact=v=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',notation:'compact',maximumFractionDigits:1}).format(v);
const money=v=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(v);
const titles={survival:'Funding and survival by age',paths:'Simulation paths',bands:'Projected balance range'};
const notes={survival:'Still funded includes lifetimes that ended without a shortfall. It is not the probability of being both alive and funded.',paths:'Each dot is a positive annual balance, colored by that simulation’s final outcome. The mean uses positive balances still observed at each age. Log scale shows equal proportional changes with equal spacing. Brighter colors mark separation from opposite outcomes in this run only; if none had the opposite outcome, all dots use the brighter color.',bands:'Median and 10th–90th percentile balances of paths still running at each age. Failures contribute $0 at their failure age, then stop. Balances are never carried forward after death or failure. Later ages have fewer observations; these ranges are not overall readiness estimates.'};
function legend(type){const entries=type==='survival'?[[colors.funded,'Still funded'],[colors.alive,'Still alive (dashed)']]:type==='paths'?[[colors.success,'Successful lifetime'],[colors.failure,'Failed lifetime (triangles)'],[colors.mean,'Mean'],[colors.strong,'Above all failed samples'],[colors.clear,'Below all successful samples']]:[[colors.funded,'Median'],[colors.range,'10th–90th percentile']];return entries.map(([color,label])=>`<span><i style="background:${color}"></i>${label}</span>`).join('');}
let id=0;
export function chartCard(type){const uid='plot-'+(++id);return `<section class="card plot-card" data-plot="${type}"><div class="section-heading"><h2>${titles[type]}</h2><button class="secondary" data-expand-plot="${type}" aria-label="Expand ${titles[type]}">Expand ↗</button></div><div class="plot-surface"><canvas role="img" aria-label="${titles[type]}. ${notes[type]}"></canvas></div><div class="plot-legend">${legend(type)}</div><p class="chart-caption">${notes[type]}</p>${type==='paths'?'<p class="chart-caption" data-point-count></p>':''}<div class="plot-inspector"><label for="${uid}">Inspect age</label><input id="${uid}" type="range" step="1" aria-label="Inspect age in ${titles[type]}"><output aria-live="polite"></output></div></section>`;}
let cleanups=[],activeDialog=null;
export function disposeCharts(){cleanups.forEach(fn=>fn());cleanups=[];if(activeDialog){activeDialog.close();activeDialog.remove();activeDialog=null;}}
export function mountCharts(root,result,retirementAge){
  if(!result)return;
  root.querySelectorAll('[data-plot]').forEach(el=>cleanups.push(mount(el,result,retirementAge)));
  root.querySelectorAll('[data-expand-plot]').forEach(button=>button.addEventListener('click',()=>{
    const type=button.dataset.expandPlot,dialog=document.createElement('dialog');dialog.className='plot-dialog';
    dialog.innerHTML=`<div class="plot-dialog-head"><h2>${titles[type]}</h2><button class="secondary" data-close>Close ✕</button></div><div class="plot-controls"><button class="secondary" data-zoom="in">Zoom in +</button><button class="secondary" data-zoom="out">Zoom out −</button><button class="secondary" data-reset>Reset view</button><span>Drag the plot to pan when zoomed. Arrow keys also pan; Escape closes.</span></div>${chartCard(type)}`;
    dialog.querySelector('[data-expand-plot]').remove();dialog.querySelector('.plot-card .section-heading').remove();document.body.append(dialog);activeDialog=dialog;dialog.showModal();
    const cleanup=mount(dialog.querySelector('[data-plot]'),result,retirementAge,dialog);
    dialog.querySelector('[data-close]').onclick=()=>dialog.close();dialog.addEventListener('close',()=>{cleanup();dialog.remove();if(activeDialog===dialog)activeDialog=null;button.focus();},{once:true});
  }));
}
function mount(el,result,age,dialog=null){
  const type=el.dataset.plot,canvas=el.querySelector('canvas'),ctx=canvas.getContext('2d'),slider=el.querySelector('input[type=range]'),output=el.querySelector('output');
  const preview=isPreviewResult(result),count=result.provenance.simulationCount;
  if(preview){el.querySelector('.chart-caption').textContent+=' Sample preview only: four lifetimes cannot estimate retirement readiness.';canvas.setAttribute('aria-label',canvas.getAttribute('aria-label')+' Sample preview only: four lifetimes.');}
  const survival=result.notFailedByAge||[],bands=result.balanceBands||[],points=result.pathPoints||[],mean=result.meanPath||[],bounds=pathBounds(points,mean),log=type==='paths';
  const start=age,end=type==='survival'?(survival.at(-1)?.age??age):type==='bands'?(bands.at(-1)?.age??age):age+bounds.maxYear;
  const min=log?bounds.min:type==='bands'?Math.min(0,...bands.map(p=>p.pessimistic)):0,max=log?bounds.max:type==='survival'?1:Math.max(1,...bands.map(p=>p.optimistic))*1.04;
  let width=0,height=0,scale=1,offsetX=0,offsetY=0,selected=start,drag=null,moved=false;
  if(type==='paths')el.querySelector('[data-point-count]').textContent=`Showing ${points.length.toLocaleString()} sampled points from ${result.provenance.simulationCount.toLocaleString()} simulated lifetimes.`;
  slider.min=start;slider.max=end;slider.value=start;canvas.tabIndex=0;
  const inset={left:68,right:16,top:22,bottom:44};
  const span=()=>({w:Math.max(1,width-inset.left-inset.right),h:Math.max(1,height-inset.top-inset.bottom)});
  function constrain(){const {w,h}=span();offsetX=Math.max(w*(1-scale),Math.min(0,offsetX));offsetY=Math.max(h*(1-scale),Math.min(0,offsetY));}
  function description(a){if(type==='survival'){const p=survival.find(p=>p.age===a);return p?`Age ${a} · Still funded ${shareLabel(p.notFailedShare,count)} · Still alive ${shareLabel(p.aliveShare,count)}`:'';}if(type==='bands'){const p=bands.find(p=>p.age===a);return p?`Age ${a} · ${p.pathCount} ${p.pathCount===1?'path':'paths'} · Median ${money(p.median)} · 10th–90th ${money(p.pessimistic)}–${money(p.optimistic)}`:'';}const p=mean.find(p=>p.yearsInRetirement===a-age);return `Age ${a} · ${p?'Mean '+money(p.balance):'No positive balances observed'}`;}
  function draw(){
    const {w,h}=span(),x=a=>inset.left+(a-start)/Math.max(1,end-start)*w*scale+offsetX,y=v=>inset.top+(1-valueToFraction(v,min,max,log))*h*scale+offsetY;
    ctx.clearRect(0,0,width,height);ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);ctx.font='11px system-ui';ctx.lineWidth=1;
    for(let i=0;i<5;i++){const py=inset.top+h-i*h/4,f=1-(py-inset.top-offsetY)/(h*scale),v=fractionToValue(f,min,max,log);ctx.strokeStyle='#e2e8e9';ctx.beginPath();ctx.moveTo(inset.left,py);ctx.lineTo(inset.left+w,py);ctx.stroke();ctx.fillStyle='#60757d';ctx.textAlign='right';ctx.fillText(type==='survival'?(preview?String(Math.round(v*count)):Math.round(v*100)+'%'):compact(v),inset.left-9,py+4);}
    const ageLow=start-offsetX/(w*scale)*(end-start),ageHigh=start+(w-offsetX)/(w*scale)*(end-start),interval=(ageHigh-ageLow)>35&&width<520?10:5;
    const ticks=[Math.ceil(ageLow),...Array.from({length:Math.max(0,Math.floor(ageHigh/interval)-Math.ceil(ageLow/interval)+1)},(_,i)=>(Math.ceil(ageLow/interval)+i)*interval),Math.floor(ageHigh)];let last=-Infinity;
    ctx.textAlign='center';for(const a of [...new Set(ticks)].sort((a,b)=>a-b)){const px=x(a);if(px-last<30)continue;last=px;ctx.fillStyle='#60757d';ctx.fillText(String(a),px,inset.top+h+19);}
    ctx.fillText('Age',inset.left+w/2,height-5);ctx.textAlign='left';ctx.fillText(type==='survival'?(preview?'Sample lifetimes (out of '+count+')':'Share of simulated lifetimes'):log?'Portfolio balance · log scale':'Portfolio balance · linear scale',inset.left,12);
    ctx.save();ctx.beginPath();ctx.rect(inset.left,inset.top,w,h);ctx.clip();
    function line(rows,xKey,yKey,color,dashed=false){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dashed?[6,4]:[]);ctx.beginPath();rows.forEach((p,i)=>{const px=x(xKey(p)),py=y(p[yKey]);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();ctx.setLineDash([]);if(rows.length===1){ctx.fillStyle=color;ctx.beginPath();ctx.arc(x(xKey(rows[0])),y(rows[0][yKey]),3,0,Math.PI*2);ctx.fill();}}
    if(type==='survival'){line(survival,p=>p.age,'notFailedShare',colors.funded);line(survival,p=>p.age,'aliveShare',colors.alive,true);}
    else if(type==='bands'){ctx.fillStyle=colors.range;ctx.beginPath();bands.forEach((p,i)=>i?ctx.lineTo(x(p.age),y(p.optimistic)):ctx.moveTo(x(p.age),y(p.optimistic)));[...bands].reverse().forEach(p=>ctx.lineTo(x(p.age),y(p.pessimistic)));ctx.closePath();ctx.fill();line(bands,p=>p.age,'median',colors.funded);}
    else{ctx.globalAlpha=.62;for(const p of points){const px=x(age+p.yearsInRetirement)+(p.successfulPath?1:3),py=y(p.balance);if(px<inset.left||px>inset.left+w||py<inset.top||py>inset.top+h)continue;ctx.fillStyle=p.successfulPath?(p.separatedFromOppositeOutcome?colors.strong:colors.success):(p.separatedFromOppositeOutcome?colors.clear:colors.failure);ctx.beginPath();if(p.successfulPath)ctx.arc(px,py,dialog?1.65:1.2,0,Math.PI*2);else{ctx.moveTo(px,py-2);ctx.lineTo(px-1.8,py+1.5);ctx.lineTo(px+1.8,py+1.5);ctx.closePath();}ctx.fill();}ctx.globalAlpha=1;line(mean,p=>age+p.yearsInRetirement,'balance',colors.mean);}
    ctx.strokeStyle='#748d98';ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(x(selected),inset.top);ctx.lineTo(x(selected),inset.top+h);ctx.stroke();ctx.setLineDash([]);ctx.restore();
    ctx.strokeStyle='#9bafb7';ctx.strokeRect(inset.left,inset.top,w,h);
    if(type==='paths'&&!points.length&&!mean.length){ctx.fillStyle='#60757d';ctx.textAlign='center';ctx.fillText('No positive balances to plot',inset.left+w/2,inset.top+h/2);}
  }
  function inspect(a){selected=Math.max(start,Math.min(end,Math.round(a)));slider.value=selected;output.textContent=description(selected);draw();}
  slider.oninput=()=>{const next=Number(slider.value),{w}=span();const px=(next-start)/Math.max(1,end-start)*w*scale+offsetX;if(px<0||px>w)offsetX=w/2-(next-start)/Math.max(1,end-start)*w*scale;constrain();inspect(next);};
  function resize(){const rect=canvas.getBoundingClientRect();width=rect.width;height=rect.height;const ratio=window.devicePixelRatio||1;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);constrain();draw();}
  const observer=new ResizeObserver(resize);observer.observe(canvas);output.textContent=description(start);
  function pos(e){const r=canvas.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top};}
  canvas.onpointerdown=e=>{const p=pos(e);drag={...p,ox:offsetX,oy:offsetY};moved=false;if(dialog)canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag||!dialog||scale===1)return;const p=pos(e);if(Math.abs(p.x-drag.x)+Math.abs(p.y-drag.y)>4)moved=true;offsetX=drag.ox+p.x-drag.x;offsetY=drag.oy+p.y-drag.y;constrain();draw();};
  canvas.onpointerup=e=>{if(!moved){const p=pos(e),{w}=span();inspect(start+(p.x-inset.left-offsetX)/(w*scale)*(end-start));}drag=null;};canvas.onpointercancel=()=>{drag=null;};
  if(dialog){canvas.classList.add('interactive');function zoom(factor){const {w,h}=span(),next=Math.max(1,Math.min(8,scale*factor)),ratio=next/scale;offsetX=w/2-(w/2-offsetX)*ratio;offsetY=h/2-(h/2-offsetY)*ratio;scale=next;constrain();draw();dialog.querySelector('[data-zoom=in]').disabled=scale>=8;dialog.querySelector('[data-zoom=out]').disabled=scale<=1;}
    dialog.querySelector('[data-zoom=in]').onclick=()=>zoom(1.5);dialog.querySelector('[data-zoom=out]').onclick=()=>zoom(1/1.5);dialog.querySelector('[data-reset]').onclick=()=>{scale=1;offsetX=offsetY=0;zoom(1);};zoom(1);
    canvas.onkeydown=e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();if(e.key==='ArrowLeft')offsetX+=40;if(e.key==='ArrowRight')offsetX-=40;if(e.key==='ArrowUp')offsetY+=40;if(e.key==='ArrowDown')offsetY-=40;constrain();draw();}};
  }
  return ()=>observer.disconnect();
}
