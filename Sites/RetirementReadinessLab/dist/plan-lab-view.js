import {LAB_LEVERS,LAB_LEVER_GROUPS,LAB_PRESETS,LAB_MAX_WHAT_IFS,LAB_FREE_WHAT_IFS,LAB_MAX_SETS,STRESS_TESTS,readinessDelta} from './plan-lab.js';
import {readinessLabel,ageYearRows} from './result-format.js';

const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=(v,d=0)=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:d}).format(v||0);
const compact=v=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',notation:'compact',maximumFractionDigits:1}).format(v||0);
// Plan colors differ in lightness as well as hue; the current plan is dashed grey.
export const LAB_COLORS=['#6b7a84','#186c65','#b86e0c','#3d6fd1','#8a4fc0'];
const swatch=(color,dashed=false)=>`<span class="lab-swatch${dashed?' dashed':''}" style="--swatch:${color}" aria-hidden="true"></span>`;
const proTag='<span class="lab-pro-tag">Pro</span>';
const lockedNote=text=>`<p class="lab-locked">${proTag} ${esc(text)} <button class="text-link" data-view="billing">Explore Pro</button></p>`;

export function labPage(vm,h){
  return `${h.head}${h.notices}${setTabs(vm)}${summaryHero(vm)}${planCards(vm)}${builder(vm)}${overlayChart(vm)}<div class="lab-pair">${sensitivityCard(vm)}${goalCard(vm,h.goal)}</div>${stressSection(vm)}${numbersTable(vm)}<p class="form-note lab-footnote">Every plan uses the same fixed sequence of ${vm.paths.toLocaleString('en-US')} market, inflation, lifespan and care paths, so differences come from the changes rather than luck. ${vm.pro?'':`Free accounts compare one what-if with ${vm.paths} paths.`} Results are hypothetical and in ${vm.basis==='today'?'today’s':'future'} dollars where noted.</p>`;
}

function setTabs(vm){
  const tabs=vm.sets.map(set=>`<button role="tab" class="lab-set${set.id===vm.set.id?' active':''}" aria-selected="${set.id===vm.set.id}" data-action="lab-select-set" data-set="${esc(set.id)}">${esc(set.name)}</button>`).join('');
  const add=vm.pro?(vm.sets.length<LAB_MAX_SETS?'<button class="lab-set add" data-action="lab-new-set">+ New comparison set</button>':''):`<button class="lab-set add" data-action="lab-new-set" disabled>+ New comparison set · Pro</button>`;
  return `<section class="lab-sets" aria-label="Comparison sets"><div class="lab-set-tabs" role="tablist" aria-label="Saved comparison sets">${tabs}${add}</div><div class="lab-set-tools"><label for="lab-set-name">Set name</label><input id="lab-set-name" type="text" maxlength="60" data-lab-set-name value="${esc(vm.set.name)}">${vm.sets.length>1?'<button class="text-link" data-action="lab-delete-set">Delete this set</button>':''}<span class="form-note">Saved with this plan and included in backups. ${vm.ranAt?`Last run ${esc(vm.ranAt)}.`:'Not run yet.'}</span></div></section>`;
}

function summaryHero(vm){
  const s=vm.summary;if(!s)return '';
  const label=r=>esc(r.label),value=r=>readinessLabel(r.result);
  const facts=[
    s.single?[`Biggest single change`,`${label(s.single.row)} <em>${esc(s.single.delta)}</em>`]:null,
    s.resilient?[`More resilient in the stress tests`,`${label(s.resilient.row)} <em>${(s.resilient.average*100).toFixed(0)}% average</em>`]:null,
    s.lowTax?[`Lowest lifetime federal income tax`,`${label(s.lowTax.row)} <em>−${money(s.lowTax.saving)}</em>`]:null
  ].filter(Boolean);
  const actions=s.improved?`<div class="actions"><button class="lab-gold" data-action="lab-apply" data-id="${esc(s.best.id)}" ${vm.busy?'disabled':''}>Make “${label(s.best)}” my plan</button><button class="lab-ghost" data-action="lab-copy" data-id="${esc(s.best.id)}" ${vm.busy?'disabled':''}>Save as a copy</button></div>`:'';
  const detail=s.improved?`It reaches ${value(s.best)} compared with ${value(s.base)} for your current plan (${esc(s.delta)}).`:`Your current plan reaches ${value(s.base)}. Try a combination of changes in the builder below.`;
  return `<section class="lab-hero" aria-labelledby="lab-hero-title"><div class="lab-hero-main"><span class="lab-eyebrow">In short</span><h2 id="lab-hero-title">${esc(s.headline)}</h2><p>${detail}</p>${actions}</div>${facts.length?`<dl class="lab-hero-facts">${facts.map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join('')}</dl>`:''}</section>`;
}

function card(row,vm,i){
  const done=row.result&&!row.stale,shown=row.shown;
  const status=row.error?`<p class="lab-row-error">${esc(row.error)}</p>`:row.note?`<p class="form-note">${esc(row.note)}</p>`:row.locked?lockedNote('Free accounts compare one what-if.'):row.stale?'<p class="form-note">Changed since the last run.</p>':'';
  const chips=row.baseline?row.changesText:row.changesText.length?row.changesText:['No changes yet'];
  const actions=row.baseline?`<button class="secondary" data-action="lab-open-plan">Open plan</button>`:`<button class="secondary" data-action="lab-edit" data-id="${esc(row.id)}" aria-pressed="${vm.editing?.id===row.id}">${vm.editing?.id===row.id?'Editing':'Edit'}</button><details class="lab-menu"><summary aria-label="More actions for ${esc(row.label)}">•••</summary><div class="lab-menu-items"><button class="text-link" data-action="lab-duplicate" data-id="${esc(row.id)}" ${vm.whatIfCount>=vm.maxWhatIfs?'disabled':''}>Duplicate</button><button class="text-link" data-action="lab-apply" data-id="${esc(row.id)}" ${!done||vm.busy?'disabled':''}>Make this my plan</button><button class="text-link" data-action="lab-copy" data-id="${esc(row.id)}" ${!done||vm.busy?'disabled':''}>Save as a copy</button><button class="text-link danger-text" data-action="lab-remove" data-id="${esc(row.id)}">Remove</button></div></details>`;
  return `<article class="lab-card${vm.editing?.id===row.id?' editing':''}${row.baseline?' baseline':''}" style="--plan:${LAB_COLORS[i%LAB_COLORS.length]}"><div class="lab-card-title">${swatch(LAB_COLORS[i%LAB_COLORS.length],row.baseline)}<h3>${esc(row.label)}</h3></div><ul class="lab-chips">${chips.map(c=>`<li>${esc(c)}</li>`).join('')}</ul><div class="lab-card-result"><span>${vm.preview?'Sample lifetimes without a shortfall':'Readiness'}</span><strong>${done?readinessLabel(row.result):'—'}</strong>${done?`<em class="${row.baseline?'':row.deltaTone}">${row.baseline?'Baseline':esc(row.delta)}</em>`:''}${done?`<div class="lab-meter" aria-hidden="true"><i style="--fill:${row.result.successProbability}"></i></div>`:''}</div>${done?`<dl class="lab-card-stats"><div><dt>Median left</dt><dd>${money(shown.medianEndingBalance)}</dd></div><div><dt>Shortfalls</dt><dd>${row.metrics.shortfalls.toLocaleString('en-US')} of ${row.metrics.paths.toLocaleString('en-US')}</dd></div></dl>`:''}${status}<div class="lab-card-actions">${actions}</div></article>`;
}

function planCards(vm){
  const canAdd=vm.whatIfCount<vm.maxWhatIfs;
  const presets=vm.presets.map(p=>`<button class="text-link" data-action="lab-add" data-preset="${p.key}" ${canAdd?'':'disabled'}>${esc(p.label)}</button>`).join('');
  const add=`<article class="lab-card lab-add-card"><h3>Add a what-if</h3>${canAdd?`<button class="primary" data-action="lab-add">Start from my plan</button><p class="form-note">Or start with a common change:</p><div class="lab-presets">${presets}</div>`:vm.pro?`<p class="form-note">This set has ${LAB_MAX_WHAT_IFS} what-ifs, the most it can compare. Remove one or start a new set.</p>`:lockedNote(`Compare up to ${LAB_MAX_WHAT_IFS} what-ifs side by side.`)}</article>`;
  return `<section class="lab-section" aria-labelledby="lab-plans-title"><div class="lab-section-head"><h2 id="lab-plans-title">Side by side</h2><span class="form-note">${vm.whatIfCount} of ${vm.maxWhatIfs} what-if${vm.maxWhatIfs===1?'':'s'} · ${vm.paths.toLocaleString('en-US')} paths each · amounts in ${vm.basis==='today'?'today’s':'future'} dollars</span></div><div class="lab-cards-scroll" role="region" aria-label="Plans side by side" tabindex="0"><div class="lab-cards">${vm.rows.map((row,i)=>card(row,vm,i)).join('')}${add}</div></div></section>`;
}

function field(lever,f){
  const id=`lab-${lever.key}-${f.name}`,attrs=`id="${id}" data-lab-lever="${lever.key}" data-lab-field="${f.name}"`;
  const input=f.type==='select'?`<select ${attrs}>${f.options.map(([v,t])=>`<option value="${esc(v)}" ${String(v)===String(f.value)?'selected':''}>${esc(t)}</option>`).join('')}</select>`:f.type==='money'?`<span class="money-entry"><span aria-hidden="true">$</span><input ${attrs} type="text" inputmode="decimal" value="${esc(f.value)}"></span>`:`<input ${attrs} type="${f.type==='number'?'text':f.type}" ${f.type==='number'?'inputmode="numeric"':''} ${f.min?`min="${f.min}"`:''} value="${esc(f.value)}">`;
  return `<div class="field"><label for="${id}">${esc(f.label)}</label>${input}</div>`;
}
function leverRow(lever,vm){
  const w=vm.editing,on=Object.hasOwn(w.changes,lever.key),editing=vm.editingLever===lever.key,available=lever.available(vm.plan);
  const head=`<div class="lab-lever-name"><strong>${esc(lever.label)}</strong>${lever.isNew?'<span class="lab-new">New</span>':''}</div>`;
  if(editing){const fields=lever.fields(vm.plan,vm.leverDraftValue).map(f=>Object.hasOwn(vm.leverDraft??{},f.name)?{...f,value:vm.leverDraft[f.name]}:f);return `<div class="lab-lever editing" id="lab-lever-${lever.key}">${head}<div class="lab-lever-form"><div class="fields">${fields.map(f=>field(lever,f)).join('')}</div>${vm.leverError?`<p class="field-error" role="alert">${esc(vm.leverError)}</p>`:''}<div class="actions"><button class="primary" data-action="lab-lever-save" data-lever="${lever.key}">Use this change</button><button class="secondary" data-action="lab-lever-cancel">Cancel</button></div></div></div>`;}
  if(on)return `<div class="lab-lever on">${head}<div class="lab-lever-value"><s>${esc(lever.current(vm.plan))}</s><span class="lab-value">${esc(lever.describe(w.changes[lever.key],vm.plan))}</span><button class="text-link" data-action="lab-lever-edit" data-lever="${lever.key}">Edit</button><button class="lab-icon-button" data-action="lab-lever-remove" data-lever="${lever.key}" aria-label="Remove ${esc(lever.label)} change">×</button></div></div>`;
  return `<div class="lab-lever">${head}<div class="lab-lever-value"><span class="muted">${available?esc(lever.current(vm.plan)):'Not available for this plan'}</span>${available?`<button class="secondary compact" data-action="lab-lever-edit" data-lever="${lever.key}">Change</button>`:''}</div></div>`;
}
function builder(vm){
  const w=vm.editing;if(!w)return '';
  const live=vm.live,base=vm.rows[0]?.result;
  const estimate=!vm.pro?lockedNote('A live estimate updates as you edit.'):live.status==='running'?'<p class="lab-live-value" aria-live="polite">Estimating…</p>':live.result?`<p class="lab-live-value" aria-live="polite">≈ ${readinessLabel(live.result)}</p><p class="form-note">${live.base?esc(readinessDelta(live.result,live.base))+' compared with your current plan at the same paths':''}</p>`:live.error?`<p class="lab-row-error">${esc(live.error)}</p>`:'<p class="form-note">Make a change to see a quick estimate.</p>';
  const groups=LAB_LEVER_GROUPS.map(g=>`<div class="lab-lever-group"><h3>${esc(g)}</h3>${LAB_LEVERS.filter(l=>l.group===g).map(l=>leverRow(l,vm)).join('')}</div>`).join('');
  return `<section class="lab-builder card" id="lab-builder" aria-labelledby="lab-builder-title"><div class="lab-builder-head"><div><h2 id="lab-builder-title">Edit what-if</h2><p class="form-note">Stack as many changes as you like. Your saved plan stays as it is until you choose to use this version.</p></div><div class="field lab-name-field"><label for="lab-whatif-name">What-if name</label><input id="lab-whatif-name" type="text" maxlength="80" data-lab-whatif-name value="${esc(w.name)}"></div></div><div class="lab-builder-body"><div class="lab-levers">${groups}</div><aside class="lab-live"><span class="lab-eyebrow dark">Live estimate</span><p class="form-note">Quick check with ${vm.livePaths} paths${base?'':' · run the comparison for the full answer'}.</p>${estimate}<p class="form-note">Changes: ${Object.keys(w.changes).length}</p><button class="primary" data-action="run-lab" ${vm.busy?'disabled':''}>Run full comparison</button><button class="secondary" data-action="lab-duplicate" data-id="${esc(w.id)}" ${vm.whatIfCount>=vm.maxWhatIfs?'disabled':''}>Duplicate as a new what-if</button><button class="text-link" data-action="lab-edit-close">Close editor</button></aside></div></section>`;
}

const MEASURES=[['funded','Funded share'],['median','Median balance'],['tough','Tough markets (10th pct.)'],['tax','Yearly taxes']];
function series(row,measure){
  if(measure==='funded')return ageYearRows(row.result.notFailedByAge||[]).map(p=>({age:p.ageYear,value:p.notFailedShare}));
  if(measure==='tax')return (row.result.planLab?.taxByAge||[]).map(p=>({age:Math.floor(p.age+1e-9),value:p.median}));
  return ageYearRows(row.shown.balanceBands||[]).filter(p=>p.pathCount>0).map(p=>({age:p.ageYear,value:measure==='median'?p.median:p.pessimistic}));
}
function overlayChart(vm){
  const rows=vm.rows.map((row,i)=>({row,color:LAB_COLORS[i%LAB_COLORS.length],i})).filter(x=>x.row.result&&!x.row.stale);
  const tabs=`<div class="lab-segments" role="group" aria-label="Chart measure">${MEASURES.map(([k,l])=>`<button data-action="lab-measure" data-measure="${k}" aria-pressed="${vm.measure===k}">${l}</button>`).join('')}</div>`;
  const head=`<div class="lab-section-head"><div><h2 id="lab-chart-title">${vm.measure==='funded'?'Still funded at each age':vm.measure==='median'?'Median balance by age':vm.measure==='tough'?'Balance in tough markets by age':'Median yearly federal income tax'}</h2><p class="form-note">${vm.measure==='funded'?'Share of modeled lifetimes with money left, by your age. Lifetimes that end without a shortfall count as funded.':vm.measure==='tax'?'Median federal income tax each year among lifetimes still running, in today’s dollars.':`Among lifetimes still running, in ${vm.basis==='today'?'today’s':'future'} dollars.`}</p></div>${tabs}</div>`;
  if(!rows.length)return `<section class="card lab-chart" aria-labelledby="lab-chart-title">${head}<p class="form-note">Run the comparison to draw every plan on one chart.</p></section>`;
  if(vm.measure==='tax'&&!rows.some(x=>x.row.result.planLab))return `<section class="card lab-chart" aria-labelledby="lab-chart-title">${head}<p class="form-note">Run the comparison again to add yearly taxes.</p></section>`;
  const legend=`<div class="lab-legend">${rows.map(x=>`<label><input type="checkbox" data-lab-series="${esc(x.row.id)}" ${vm.hidden.has(x.row.id)?'':'checked'}>${swatch(x.color,x.row.baseline)}${esc(x.row.label)}</label>`).join('')}</div>`;
  const visible=rows.filter(x=>!vm.hidden.has(x.row.id)),data=visible.map(x=>({...x,points:series(x.row,vm.measure).filter(p=>Number.isFinite(p.age)&&p.age<=105&&Number.isFinite(p.value))})).filter(x=>x.points.length);
  if(!data.length)return `<section class="card lab-chart" aria-labelledby="lab-chart-title">${head}${legend}<p class="form-note">${visible.length?'No data is available for this measure.':'Select at least one plan to show the chart.'}</p></section>`;
  const maxAge=Math.min(105,Math.max(...data.flatMap(x=>x.points).map(p=>p.age))),all=data.flatMap(x=>x.points).filter(p=>p.age<=maxAge),minAge=Math.min(...all.map(p=>p.age));
  const share=vm.measure==='funded',maxValue=share?1:Math.max(1,...all.map(p=>p.value))*1.08,minValue=share?Math.max(0,Math.floor(Math.min(...all.map(p=>p.value))*10)/10-.1):0;
  const W=720,H=300,L=58,R=18,T=14,B=38,x=a=>L+(a-minAge)/Math.max(1,maxAge-minAge)*(W-L-R),y=v=>T+(1-(v-minValue)/Math.max(1e-9,maxValue-minValue))*(H-T-B);
  const yTicks=Array.from({length:5},(_,i)=>minValue+(maxValue-minValue)*i/4),step=Math.max(1,Math.ceil((maxAge-minAge)/8/5)*5),xTicks=[];for(let a=Math.ceil(minAge/5)*5;a<=maxAge;a+=step)xTicks.push(a);
  const fmt=v=>share?`${Math.round(v*100)}%`:compact(v),n=data[0]?.row.result.provenance.simulationCount??0;
  // A 100-path preview reports counts, not percentages, outside the axis scale.
  const valueText=v=>share&&vm.preview?`${Math.round(v*n)} of ${n}`:fmt(v);
  const lines=data.map(d=>`<polyline points="${d.points.filter(p=>p.age<=maxAge).map(p=>`${x(p.age).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ')}" fill="none" stroke="${d.color}" stroke-width="${d.row.baseline?2.4:2.8}" ${d.row.baseline?'stroke-dasharray="6 4"':''} stroke-linejoin="round"/>`).join('');
  const age=Math.max(minAge,Math.min(maxAge,vm.chartAge??Math.min(maxAge,90))),marker=`<line x1="${x(age)}" y1="${T}" x2="${x(age)}" y2="${H-B}" class="lab-age-line"/>`;
  const svg=`<svg class="lab-chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(MEASURES.find(([k])=>k===vm.measure)[1])} by age for ${data.length} plans">${yTicks.map(v=>`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" class="lab-grid"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end">${fmt(v)}</text>`).join('')}${xTicks.map(a=>`<text x="${x(a)}" y="${H-14}" text-anchor="middle">${a}</text>`).join('')}${marker}${lines}</svg>`;
  const ages=[];for(let a=Math.ceil(minAge);a<=maxAge;a++)ages.push(a);
  const valueAt=d=>{let v=null;for(const p of d.points){if(p.age>age)break;v=p.value;}return v;};
  const inspector=`<div class="lab-inspector"><label for="lab-chart-age">Compare at age</label><select id="lab-chart-age" data-lab-chart-age>${ages.map(a=>`<option value="${a}" ${a===age?'selected':''}>${a}</option>`).join('')}</select><dl>${data.map(d=>`<div><dt>${swatch(d.color,d.row.baseline)}${esc(d.row.label)}</dt><dd>${valueAt(d)==null?'No lifetimes':valueText(valueAt(d))}</dd></div>`).join('')}</dl></div>`;
  return `<section class="card lab-chart" aria-labelledby="lab-chart-title">${head}${legend}<div class="lab-chart-body"><div class="lab-chart-plot">${svg}</div>${inspector}</div></section>`;
}

function sensitivityCard(vm){
  const t=vm.sensitivity,head=`<div class="lab-section-head"><div><h2 id="lab-sensitivity-title">What moves your result most</h2><p class="form-note">Each row changes one input both ways from your current plan, using ${vm.sensitivityPaths} paths.</p></div></div>`;
  if(!vm.pro)return `<section class="card lab-sensitivity" aria-labelledby="lab-sensitivity-title">${head}${lockedNote('Rank which inputs change your readiness the most.')}</section>`;
  const button=`<button class="${t?.rows?'secondary':'primary'}" data-action="run-lab-sensitivity" ${vm.busy?'disabled':''}>${t?.rows?'Run again':'Find what matters most'}</button>`;
  if(!t?.rows)return `<section class="card lab-sensitivity" aria-labelledby="lab-sensitivity-title">${head}<p class="form-note">Tests stock returns, retirement date, spending, inflation, claiming age, healthcare costs and stock allocation.</p>${button}</section>`;
  const max=Math.max(1,...t.rows.flatMap(r=>[Math.abs(r.low??0),Math.abs(r.high??0)]));
  const side=v=>v==null?'':`${v>0?'+':v<0?'−':''}${Math.abs(v).toFixed(1).replace(/\.0$/,'')}`;
  const bar=(v,dir)=>v==null?'<span class="lab-bar-empty">n/a</span>':`<i class="lab-bar ${v<0?'down':'up'}" style="width:${(Math.abs(v)/max*100).toFixed(1)}%"></i><b>${side(v)}</b>`;
  const rows=t.rows.map(r=>`<div class="lab-tornado-row"><div class="lab-tornado-label"><strong>${esc(r.label)}</strong><small>${esc(r.range)}</small></div><div class="lab-tornado-bars"><span class="neg">${bar(r.low,'low')}</span><span class="pos">${bar(r.high,'high')}</span></div>${r.control&&r.best?`<button class="text-link" data-action="lab-sensitivity-whatif" data-input="${r.key}" data-side="${r.best}">Add as a what-if</button>`:'<span></span>'}</div>`).join('');
  return `<section class="card lab-sensitivity" aria-labelledby="lab-sensitivity-title">${head}<div class="lab-tornado-key"><span>Lower readiness</span><span>Higher readiness · points</span></div><div class="lab-tornado">${rows}</div><p class="form-note">${t.stale?'Your plan changed since this ran. ':''}Baseline ${readinessLabel(t.base)} with ${t.base.provenance.simulationCount} paths. Returns, inflation and healthcare costs are assumptions you do not control; effects overlap.</p>${button}</section>`;
}

function goalCard(vm,frontierHtml){
  const g=vm.goal,head=`<div class="lab-section-head"><div><h2 id="lab-goal-title">Goal finder</h2><p class="form-note">Pick a readiness target and what you are willing to change.</p></div></div>`;
  if(!vm.pro)return `<section class="card lab-goal" aria-labelledby="lab-goal-title">${head}${lockedNote('Find the retirement age, spending, savings or claiming age that reaches your target.')}</section>`;
  const kinds=[['retirement','Retirement age & spending'],['claiming','Social Security claiming age & spending'],['savings','Retirement age & annual savings']];
  return `<section class="card lab-goal" aria-labelledby="lab-goal-title">${head}<div class="fields lab-goal-fields"><div class="field"><label for="lab-goal-target">Target readiness <output id="lab-goal-target-value">${Math.round(g.target*100)}%</output></label><input id="lab-goal-target" type="range" min="70" max="95" step="5" value="${Math.round(g.target*100)}" data-lab-goal="target"></div><div class="field"><label for="lab-goal-kind">Change</label><select id="lab-goal-kind" data-lab-goal="kind">${kinds.map(([k,l])=>`<option value="${k}" ${k===g.kind?'selected':''}>${esc(l)}</option>`).join('')}</select></div></div><button class="primary" data-action="run-lab-goal" ${vm.busy?'disabled':''}>Find options</button>${frontierHtml}</section>`;
}

function stressSection(vm){
  const st=vm.stress,head=`<div class="lab-section-head"><div><h2 id="lab-stress-title">Stress tests</h2><p class="form-note">How your plan and one what-if hold up when one thing goes wrong. Same paths, one forced bad event.</p></div>${vm.pro&&vm.stressChoices.length?`<div class="field lab-stress-pick"><label for="lab-stress-target">Compare with</label><select id="lab-stress-target" data-lab-stress-target>${vm.stressChoices.map(r=>`<option value="${esc(r.id)}" ${r.id===vm.stressTarget?'selected':''}>${esc(r.label)}</option>`).join('')}</select></div>`:''}</div>`;
  if(!vm.pro)return `<section class="lab-section" aria-labelledby="lab-stress-title">${head}${lockedNote('Test a market crash, high inflation, low returns, long-term care and long lives.')}</section>`;
  const target=vm.rows.find(r=>r.id===vm.stressTarget),results=new Map((st?.rows||[]).map(r=>[r.key,r]));
  const meter=(label,r,color)=>`<div class="lab-stress-line"><span>${esc(label)}</span><strong>${r?readinessLabel(r):'—'}</strong><div class="lab-meter" aria-hidden="true"><i style="--fill:${r?r.successProbability:0};--plan:${color}"></i></div></div>`;
  const cards=STRESS_TESTS.map(t=>{const r=results.get(t.key);return `<article class="lab-stress-card"><h3>${esc(t.label)}</h3>${meter('Current plan',r?.results.baseline,LAB_COLORS[0])}${target?meter(target.label,r?.results[target.id],LAB_COLORS[Math.max(1,vm.rows.indexOf(target))%LAB_COLORS.length]):''}<p class="form-note">${esc(t.note)}</p></article>`;}).join('');
  return `<section class="lab-section" aria-labelledby="lab-stress-title">${head}<div class="lab-stress-grid">${cards}</div><div class="actions"><button class="${st?.rows?'secondary':'primary'}" data-action="run-lab-stress" ${vm.busy?'disabled':''}>${st?.rows?'Run stress tests again':'Run stress tests'}</button>${st?.stale?'<span class="form-note">Plans changed since these ran.</span>':''}</div></section>`;
}

function numbersTable(vm){
  const rows=vm.rows.filter(r=>r.result&&!r.stale);
  const head=`<div class="lab-section-head"><div><h2 id="lab-table-title">Every number, side by side</h2><p class="form-note">Medians across modeled lifetimes. Balances in ${vm.basis==='today'?'today’s':'future'} dollars; taxes and conversions in today’s dollars. Changes are against your current plan.</p></div><div class="actions"><button class="secondary" data-action="lab-download-csv" ${rows.length&&vm.pro?'':'disabled'}>Download CSV${vm.pro?'':' · Pro'}</button><button class="secondary" data-action="lab-download-report" ${rows.length&&vm.pro?'':'disabled'}>Download comparison report${vm.pro?'':' · Pro'}</button></div></div>`;
  if(!rows.length)return `<section class="card lab-table" aria-labelledby="lab-table-title">${head}<p class="form-note">Run the comparison to fill this table.</p></section>`;
  const base=rows.find(r=>r.baseline)?.metrics;
  const delta=(v,b,fmt,good)=>{if(v==null||b==null||v===b)return '';const d=v-b,tone=good===0?'':(d>0)===(good>0)?'up':'down';return `<small class="${tone}">${d>0?'+':'−'}${fmt(Math.abs(d))}</small>`;};
  const lines=[
    ['Readiness',m=>readinessLabel({successProbability:m.readiness,provenance:{simulationCount:m.paths}}),null],
    ['Lifetimes with a shortfall',m=>m.shortfalls.toLocaleString('en-US'),['shortfalls',v=>v.toLocaleString('en-US'),-1]],
    ['Median left at end',m=>money(m.medianLeft),['medianLeft',money,1]],
    ['Left in tough markets (10th pct.)',m=>money(m.toughLeft),['toughLeft',money,1]],
    ['Median age when money runs out',m=>m.medianFailureAge==null?'None':m.medianFailureAge.toFixed(1),['medianFailureAge',v=>v.toFixed(1)+' yrs',1]],
    ['Lifetime federal income tax',m=>m.lifetimeTax==null?'Run again':money(m.lifetimeTax),['lifetimeTax',money,-1]],
    ['Years with a Medicare surcharge',m=>m.surchargeYears==null?'Run again':String(m.surchargeYears),['surchargeYears',v=>String(v),-1]],
    ['Roth conversions',m=>m.conversions==null?'Run again':money(m.conversions),['conversions',money,0]]
  ];
  const cells=(r,[label,fmt,cmp])=>`<td>${fmt(r.metrics)}${!r.baseline&&cmp&&base?delta(r.metrics[cmp[0]],base[cmp[0]],cmp[1],cmp[2]):''}${!r.baseline&&!cmp&&base?`<small>${esc(r.delta)}</small>`:''}</td>`;
  const table=`<div class="table-wrap" role="region" aria-labelledby="lab-table-title" tabindex="0"><table class="lab-numbers"><thead><tr><th scope="col">Measure</th>${rows.map(r=>`<th scope="col">${swatch(LAB_COLORS[vm.rows.indexOf(r)%LAB_COLORS.length],r.baseline)}${esc(r.label)}</th>`).join('')}</tr></thead><tbody>${lines.map(line=>`<tr><th scope="row">${line[0]}</th>${rows.map(r=>cells(r,line)).join('')}</tr>`).join('')}</tbody></table></div>`;
  const yearly=vm.yearly.length?`<details class="lab-yearly"><summary>Year-by-year median balances</summary><div class="table-wrap" tabindex="0"><table><thead><tr><th scope="col">Age</th>${rows.map(r=>`<th scope="col">${esc(r.label)}</th>`).join('')}</tr></thead><tbody>${vm.yearly.map(y=>`<tr><th scope="row">${y.age}</th>${y.values.map(v=>`<td>${v==null?'—':money(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></details>`:'';
  return `<section class="card lab-table" aria-labelledby="lab-table-title">${head}${table}${yearly}</section>`;
}
