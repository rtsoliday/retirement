// Spreadsheet view controller. Rows and rules live in sheet-model.js; the
// calculation is the same worker and engine the classic planner uses.
import {normalizeScenarios,validateScenarioStructure,repairScenarios,markRepairedUnknown,repairNotice,prepareCalendarScenario,sampleScenarios,applyProSimulationDefault,localCalendarDate,FREE_SIMULATION_PATHS,MIN_SIMULATION_PATHS,MAX_SIMULATION_PATHS} from './model.js';
import {normalizeInputSources} from './ux-guidance.js';
import {normalizeOptionalAnswers,OPTIONAL_QUESTIONS,inputTask} from './plan-review.js';
import {copyLabSets,LIVE_ESTIMATE_PATHS} from './plan-lab.js';
import {resultFingerprint} from './result-cache.js';
import {readinessLabel} from './result-format.js';
import {initializeSocialAuth,authHeaders} from './auth.js';
import {reportUsage} from './usage-metrics.js';
import {SECTIONS,getPath,cellText,editText,parseCell,applyCell,clearCell,isUnknown,planIssues,blockingUnknowns,rowForPath,simulationScenario,shownResult,readinessDelta,fanPeak,fanPaths,forecastTable,fundedLabel,yearlyTable,YEARLY_COLUMNS,moneyText,toTsv,toCsv,parseTsv,countsAsSimulationStart} from './sheet-model.js';
import {rememberView,handOffPlan,takeHandedOffPlan,arrivalReferrer,parseSheetHash,plannerHref} from './view-switch.js';

const STORE_KEY='retirement-readiness-lab-sites-v1',UI_KEY='retirement-sheet-ui',BASIS_KEY='retirement-dollar-basis';
const $=id=>document.getElementById(id);
const escapeHTML=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const letter=i=>{let s='';for(let n=i+1;n>0;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const NAME_ROW={path:'name',type:'text',label:'Plan',note:'Plan name. The planner uses it too.'};
const PERSONAL_PATHS=['household.birthday','household.retirementDate','accounts.pretax','accounts.roth','accounts.taxable','accounts.cash','spending.annualBaseSpending','socialSecurity.annualBenefitAt67','spouseAccounts.pretax','spouseAccounts.roth','spouseIncome.annualBenefitAt67','healthcare.preMedicareMonthlyPremium','rothHistory.contributionBasis','rothHistory.firstContributionYear','spouseRothHistory.contributionBasis','spouseRothHistory.firstContributionYear',...['contributions','spouseContributions'].flatMap(prefix=>['pretax','employerPretax','roth','taxable','cash'].map(key=>`${prefix}.${key}`))];

// ---------- Saved plans (same store and locking as the classic planner) ----------
let lastSavedRaw=null,saveQueue=Promise.resolve(),pendingSaves=0,storeBlocked=false;
function originalPlans(){return sampleScenarios().map(s=>{prepareCalendarScenario(s,{needsReview:false});s.household.separatePeople=true;s.household.spouseRetirementDate=s.household.retirementDate;return s;});}
function readStore(){
  const raw=localStorage.getItem(STORE_KEY);lastSavedRaw=raw;
  if(raw===null){
    const plans=originalPlans();
    return {saved:{},plans,sources:normalizeInputSources(null,plans,'Sample/default'),optional:normalizeOptionalAnswers(null,plans),exampleIds:new Set(plans.map(p=>p.id)),selectedId:'base-plan'};
  }
  const {scenarios,...saved}=JSON.parse(raw);
  if(!Array.isArray(scenarios)||!scenarios.length)throw new Error('No saved scenarios.');
  const plans=normalizeScenarios(scenarios),repaired=repairScenarios(plans);
  for(const s of plans){if(validateScenarioStructure(s).length)throw new Error('Invalid saved scenario structure.');prepareCalendarScenario(s);}
  const sources=normalizeInputSources(saved.inputSources,plans,'Saved value; source not recorded');
  markRepairedUnknown(sources,repaired);
  const exampleIds=Array.isArray(saved.exampleIds)?saved.exampleIds.filter(id=>plans.some(p=>p.id===id)):plans.filter(p=>sources[p.id]?._origin==='Sample/default'&&Object.values(sources[p.id]).every(v=>v==='Sample/default')).map(p=>p.id);
  return {saved,plans,sources,optional:normalizeOptionalAnswers(saved.optionalAnswers,plans),exampleIds:new Set(exampleIds),selectedId:String(saved.selectedId??''),repairNote:repaired.size?repairNotice(plans,repaired):''};
}
const app={
  plans:[],sources:{},optional:{},exampleIds:new Set(),drafts:new Map(),selectedId:'',saved:{},
  tab:'plan',sel:{r:1,c:2,ar:1,ac:2},edit:null,model:null,collapsed:new Set(),diffOnly:false,basis:'today',yearlyId:'',
  tier:'free',undo:[],redo:[],results:new Map(),lastShown:new Map(),jobs:new Map(),queue:[],full:false,noteOverride:'',
  startState:'idle', // idle → edit-pending → counted; see countsAsSimulationStart.
};
function loadInto(){
  const store=readStore();
  Object.assign(app,{saved:store.saved,plans:store.plans,sources:store.sources,optional:store.optional,exampleIds:store.exampleIds,repairNote:store.repairNote||''});
  app.selectedId=store.plans.some(p=>p.id===store.selectedId)?store.selectedId:store.plans[0].id;
  if(!app.plans.some(p=>p.id===app.yearlyId))app.yearlyId=app.selectedId;
  if(app.tier==='pro')for(const s of app.plans)applyProSimulationDefault(s);
}
function persist(){
  if(storeBlocked)return Promise.resolve(false);
  const payload=JSON.stringify({...app.saved,scenarios:app.plans,selectedId:app.selectedId,hasStartedPlan:true,inputSources:app.sources,optionalAnswers:app.optional,exampleIds:[...app.exampleIds],accountChecks:app.saved.accountChecks??{},entryPeriods:app.saved.entryPeriods??{},labSets:app.saved.labSets??{}});
  const save=async()=>{
    if(!navigator.locks){setNote('Automatic saving isn’t available in this browser. Use the planner to back up.',true);return false;}
    return navigator.locks.request(STORE_KEY,()=>{
      if(localStorage.getItem(STORE_KEY)!==lastSavedRaw){
        setNote('Saved plans changed in another tab, so this edit wasn’t saved. Loaded the newer plans.',true);
        loadInto();app.undo=[];app.redo=[];refresh();return false;
      }
      localStorage.setItem(STORE_KEY,payload);lastSavedRaw=payload;return true;
    }).catch(()=>{setNote('This change couldn’t be saved in browser storage.',true);return false;});
  };
  pendingSaves++;saveQueue=saveQueue.then(save,save).finally(()=>{pendingSaves--;});return saveQueue;
}
function snapshot(){return structuredClone({plans:app.plans,sources:app.sources,optional:app.optional,exampleIds:[...app.exampleIds],selectedId:app.selectedId,saved:app.saved});}
function pushUndo(){app.undo.push(snapshot());if(app.undo.length>100)app.undo.shift();app.redo=[];if(app.startState==='idle')app.startState='edit-pending';}
function restore(snap){Object.assign(app,{plans:snap.plans,sources:snap.sources,optional:snap.optional,exampleIds:new Set(snap.exampleIds),selectedId:snap.selectedId,saved:snap.saved});if(!app.plans.some(p=>p.id===app.yearlyId))app.yearlyId=app.selectedId;}
function undo(){if(!app.undo.length)return;app.redo.push(snapshot());restore(app.undo.pop());persist();refresh();}
function redo(){if(!app.redo.length)return;app.undo.push(snapshot());restore(app.redo.pop());persist();refresh();}
const context=()=>({sources:app.sources,exampleIds:app.exampleIds,drafts:app.drafts});
const basePlan=()=>app.plans.find(p=>p.id===app.selectedId)||app.plans[0];

// ---------- Calculation ----------
const isPro=()=>app.tier==='pro';
const fullPaths=s=>isPro()?Math.max(MIN_SIMULATION_PATHS,Math.min(MAX_SIMULATION_PATHS,Number(s.numberOfSimulations)||FREE_SIMULATION_PATHS)):FREE_SIMULATION_PATHS;
const livePaths=s=>Math.min(fullPaths(s),isPro()?LIVE_ESTIMATE_PATHS:FREE_SIMULATION_PATHS);
const fingerprint=s=>resultFingerprint(s,localCalendarDate());
const OPTIONS={includePathPoints:false,includeRiskAnalysis:false};
function remember(key,result){app.results.delete(key);app.results.set(key,result);while(app.results.size>48)app.results.delete(app.results.keys().next().value);}
function currentResult(s){
  const fp=fingerprint(s),result=app.results.get(fp+'|'+fullPaths(s))||app.results.get(fp+'|'+livePaths(s))||null;
  return {result,fresh:Boolean(result)};
}
function planState(s){
  const issues=planIssues(s,app.sources),blocked=issues.some(i=>i.level==='unknown'||i.level==='error');
  const {result,fresh}=blocked?{result:null,fresh:false}:currentResult(s);
  if(result)app.lastShown.set(s.id,result);
  const shown=blocked?null:(result||app.lastShown.get(s.id)||null);
  return {issues,blocked,result:shown,stale:Boolean(shown)&&!fresh};
}
function cancelJob(id){const job=app.jobs.get(id);if(job){job.worker?.terminate();app.jobs.delete(id);}app.queue=app.queue.filter(q=>q.id!==id);}
function stopAll(){for(const id of [...app.jobs.keys()])cancelJob(id);app.queue=[];app.full=false;renderCalc();}
const CONCURRENCY=Math.max(1,Math.min(4,(navigator.hardwareConcurrency||2)-1));
function pump(){
  while(app.queue.length&&[...app.jobs.values()].filter(j=>j.worker).length<CONCURRENCY){
    const next=app.queue.shift(),plan=app.plans.find(p=>p.id===next.id);
    if(!plan){app.jobs.delete(next.id);continue;}
    const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
    const job={worker,key:next.key,progress:0,count:next.count};app.jobs.set(next.id,job);
    worker.onmessage=e=>{
      if(e.data.type==='progress'){job.progress=e.data.fraction??job.progress;renderCalc();return;}
      worker.terminate();if(app.jobs.get(next.id)===job)app.jobs.delete(next.id);
      if(e.data.type==='result')remember(next.key,e.data.result);
      else setNote(`${plan.name}: ${e.data.message}`,true);
      if(!app.jobs.size&&!app.queue.length)app.full=false;
      refresh({keepCalc:true});pump();
    };
    worker.onerror=()=>{worker.terminate();if(app.jobs.get(next.id)===job)app.jobs.delete(next.id);setNote('The calculation couldn’t start. Reload the page.',true);renderCalc();pump();};
    worker.postMessage({scenario:next.scenario,task:'simulation',options:OPTIONS});
    if(countsAsSimulationStart({explicit:next.explicit,editPending:app.startState==='edit-pending',count:next.count,fullCount:fullPaths(plan)})){
      app.startState='counted';reportUsage('simulation_start');
    }
  }
  renderCalc();
}
function enqueue(plan,count,explicit=false){
  const key=fingerprint(plan)+'|'+count,job=app.jobs.get(plan.id);
  if(app.results.has(key)||job?.key===key)return;
  cancelJob(plan.id);
  app.jobs.set(plan.id,{key,count,progress:0,worker:null});
  // Snapshot now. The key describes these inputs, and the plan can be edited before a worker is free.
  app.queue.push({id:plan.id,key,count,explicit,scenario:simulationScenario(plan,count)});
}
let recalcTimer=0;
function scheduleRecalc(delay=180){
  clearTimeout(recalcTimer);
  recalcTimer=setTimeout(()=>{
    for(const plan of app.plans){
      const blocked=planIssues(plan,app.sources).some(i=>i.level==='unknown'||i.level==='error');
      if(blocked){cancelJob(plan.id);continue;}
      // Undoing an edit can make a running job pointless.
      const job=app.jobs.get(plan.id);
      if(job&&!job.key.startsWith(fingerprint(plan)+'|'))cancelJob(plan.id);
      if(!currentResult(plan).result&&!app.jobs.get(plan.id))enqueue(plan,livePaths(plan));
    }
    pump();
  },delay);
}
function fullRun(){
  if(app.jobs.size&&app.full){stopAll();setNote('Full run stopped. Completed columns are kept.');return;}
  let queued=0;
  for(const plan of app.plans){
    if(planIssues(plan,app.sources).some(i=>i.level==='unknown'||i.level==='error'))continue;
    const key=fingerprint(plan)+'|'+fullPaths(plan);
    if(!app.results.has(key)){enqueue(plan,fullPaths(plan),true);queued++;}
  }
  if(!queued){setNote(isPro()?'Every column is already at its full path count.':`Free runs use ${FREE_SIMULATION_PATHS} paths, so every column is already current.`);return;}
  app.full=true;pump();
}

// ---------- Plan actions ----------
function newPlanId(){let id;do{id='plan-'+Date.now()+'-'+Math.random().toString(36).slice(2,9);}while(app.plans.some(s=>s.id===id));return id;}
const selectedPlan=()=>app.model?.cols[app.sel.c]?.plan||basePlan();
function addCopy(){
  const source=selectedPlan(),copy=structuredClone(source);pushUndo();
  copy.id=newPlanId();copy.name=source.name+' copy';
  app.plans.splice(app.plans.indexOf(source)+1,0,copy);
  app.sources[copy.id]=structuredClone(app.sources[source.id]);
  app.optional[copy.id]=structuredClone(app.optional[source.id]||{});
  if(app.saved.entryPeriods?.[source.id])(app.saved.entryPeriods[copy.id]=structuredClone(app.saved.entryPeriods[source.id]));
  if(app.saved.accountChecks?.[source.id])app.saved.accountChecks[copy.id]=app.saved.accountChecks[source.id];
  if(app.saved.labSets?.[source.id])app.saved.labSets[copy.id]=copyLabSets(app.saved.labSets[source.id]);
  persist();selectAfterStructure(copy.id);
}
function addBlank(){
  pushUndo();
  const plan=prepareCalendarScenario(sampleScenarios()[0],{needsReview:false});
  plan.id=newPlanId();plan.name='New plan';plan.household.separatePeople=true;plan.household.spouseRetirementDate=plan.household.retirementDate;
  if(isPro())applyProSimulationDefault(plan);
  app.plans.push(plan);
  app.sources[plan.id]={_origin:'Sample/default'};app.optional[plan.id]={};
  for(const path of [...PERSONAL_PATHS,...Object.values(OPTIONAL_QUESTIONS).flatMap(q=>q.required)])app.sources[plan.id][path]='Unknown';
  persist();selectAfterStructure(plan.id,'accounts.pretax');
  setNote('Fill the ? cells. Results appear when none are left; enter 0 for none.');
}
function deletePlan(){
  const plan=selectedPlan();
  if(app.plans.length===1){setNote('A sheet needs at least one plan.',true);return;}
  if(!confirm(`Delete "${plan.name}"?`))return;
  pushUndo();cancelJob(plan.id);
  app.plans=app.plans.filter(p=>p!==plan);
  if(app.saved.labSets)delete app.saved.labSets[plan.id];
  if(app.selectedId===plan.id)app.selectedId=app.plans[0].id;
  if(app.yearlyId===plan.id)app.yearlyId=app.selectedId;
  persist();refresh();
}
function setBase(){const plan=selectedPlan();pushUndo();app.selectedId=plan.id;persist();refresh();setNote(`Comparing against ${plan.name}.`);}
function selectAfterStructure(planId,path){
  refresh();
  const c=app.model.cols.findIndex(col=>col.plan?.id===planId),r=path?app.model.rows.findIndex(x=>x.row?.path===path):app.sel.r;
  if(c>=0)moveTo(r>=0?r:app.sel.r,c);
}

// ---------- Table models ----------
const planRowVisible=row=>!row.show||app.plans.some(row.show);
function planModel(){
  const plans=app.plans,base=basePlan(),states=new Map(plans.map(p=>[p.id,planState(p)]));
  const shown=new Map(plans.map(p=>[p.id,shownResult(states.get(p.id).result,app.basis)]));
  const peak=fanPeak([...shown.values()]),dollars=app.basis==='today'?'today $':'future $';
  const cols=[{cls:'c-label',width:'var(--label)',letter:'A'},{cls:'c-unit',width:'var(--unit)',letter:'B'},...plans.map((plan,i)=>({plan,width:'var(--plan)',letter:letter(i+2),tags:[...(plan.id===base.id?['base']:[]),...(app.exampleIds.has(plan.id)?['sample']:[])]}))];
  const rows=[];
  const label=(text,unit='')=>({t:text,cls:'c-label',html:`${escapeHTML(text)}${unit?`<span class="label-unit">${escapeHTML(unit)}</span>`:''}`});
  const unitCell=unit=>({t:unit,cls:'c-unit'});

  rows.push({kind:'name',frozen:true,row:NAME_ROW,cells:[label('Plan'),unitCell(''),...plans.map(p=>({t:p.name,clip:p.name,cls:`input left`,edit:{plan:p,row:NAME_ROW},note:NAME_ROW.note}))]});
  const resultRow=(id,text,unit,note,cell,extra={})=>rows.push({kind:'result',id,note,cells:[label(text,unit),unitCell(unit),...plans.map(p=>{const st=states.get(p.id);return {...cell(p,st,shown.get(p.id)),note};})],...extra});
  const empty=(st,fallback='')=>st.blocked?{t:'',cls:'none'}:(st.result?null:{t:fallback,cls:'none'});
  resultRow('readiness','Readiness','','Lifetimes with no shortfall. At 100 paths or fewer it’s a count, too small to estimate readiness. +/− is against base.',(p,st,r)=>{
    if(st.blocked){const unknown=st.issues.filter(i=>i.level==='unknown').length,errors=st.issues.filter(i=>i.level==='error').length;return {t:unknown?`? ${unknown} blank`:`${errors} error${errors===1?'':'s'}`,cls:'bad calc'};}
    if(!r)return {t:'…',cls:'calc stale'};
    const delta=readinessDelta(st.result,states.get(base.id).result),up=delta.startsWith('+'),down=delta.startsWith('−'),shown=readinessLabel(r).replace(' observed','');
    return {t:shown,clip:shown,cls:`readiness calc${st.stale?' stale':''}`,style:`--share:${(r.successProbability*100).toFixed(1)}%`,html:`${delta&&delta!=='='?`<span class="delta ${up?'up':down?'down':''}">${delta}</span> `:''}${escapeHTML(shown)}`};
  });
  const moneyRow=(id,text,note,key)=>resultRow(id,text,dollars,note,(p,st,r)=>empty(st,'…')||{t:moneyText(Math.round(r[key])),clip:Math.round(r[key]),num:r[key],cls:`calc${st.stale?' stale':''}`});
  moneyRow('median','End balance, median','Balance when each lifetime ends. Shortfalls count as $0.','medianEndingBalance');
  moneyRow('p10','End balance, 10th pct','Nine in ten lifetimes end above this. Shortfalls count as $0.','pessimisticEndingBalance');
  moneyRow('p90','End balance, 90th pct','One in ten lifetimes ends above this.','optimisticEndingBalance');
  resultRow('shortfall','Shortfall age, median','age','Median age savings run out, among lifetimes that run short.',(p,st,r)=>empty(st,'…')||{t:r.medianFailureAge===null?'none':String(Math.round(r.medianFailureAge)),clip:r.medianFailureAge??'',cls:`calc${st.stale?' stale':''}`});
  resultRow('fan','Balance by age','fan','10th to 90th percentile band and median by age, on one square-root scale for all plans. See Forecast.',(p,st,r)=>{
    const paths=r&&fanPaths(r.balanceBands,peak);
    return paths?{t:'',clip:'',cls:st.stale?'stale':'',html:`<svg class="fan" viewBox="0 0 100 30" preserveAspectRatio="none" role="img" aria-label="Balance range by age"><polygon points="${paths.band}"/><polyline points="${paths.median}"/></svg>`}:{t:'',cls:'none'};
  },{tall:true});
  resultRow('checks','Checks','','Blanks and errors stop a plan. Notes do not. Select this row to see them.',(p,st)=>{
    const count=level=>st.issues.filter(i=>i.level===level).length,unknown=count('unknown'),errors=count('error'),notes=count('note');
    if(unknown||errors)return {t:unknown?`${unknown} blank`:`${errors} error${errors===1?'':'s'}`,cls:'bad calc',issues:st.issues};
    return notes?{t:`${notes} note${notes===1?'':'s'}`,cls:'warn calc',issues:st.issues}:{t:'OK',cls:'calc'};
  });
  resultRow('paths','Paths used','paths','Live estimates recalculate as you edit. F9 runs the full path count.',(p,st,r)=>{
    if(!r)return {t:'',cls:'none'};
    const count=r.provenance.simulationCount;
    return {t:`${count.toLocaleString('en-US')}${count<fullPaths(p)?' live':''}`,clip:count,cls:`calc${st.stale?' stale':''}`};
  });

  const frozenThrough=innerHeight>=760&&innerWidth>760?'fan':'median';
  let frozen=true;
  for(const r of rows){r.frozen=frozen;if(r.id===frozenThrough){r.lastFrozen=true;frozen=false;}}

  for(const section of SECTIONS){
    const visible=section.rows.filter(planRowVisible);
    if(!visible.length)continue;
    const isOpen=!app.collapsed.has(section.id);
    const sectionRow={kind:'section',section,cells:[{t:section.label,cls:'c-label',html:`<span class="toggle" data-toggle="${section.id}" aria-hidden="true">${isOpen?'▾':'▸'}</span>${escapeHTML(section.label)}`,expanded:isOpen},{t:'',cls:'c-unit'},...plans.map(p=>sectionTotal(section,p))]};
    const dataRows=[];
    if(isOpen)for(const row of visible){
      if(app.diffOnly&&plans.length>1&&plans.every(p=>sameCell(p,base,row)))continue;
      dataRows.push({kind:'data',row,note:row.note,cells:[label(row.label,row.unit),unitCell(row.unit),...plans.map(p=>planCell(p,base,row))]});
    }
    if(app.diffOnly&&plans.length>1&&!dataRows.length)continue;
    rows.push(sectionRow,...dataRows);
  }
  return {cols,rows};
}
// A subtotal can't be trusted while one of its amounts is blank.
function sectionTotal(section,plan){
  if(!section.total)return {t:''};
  if(blockingUnknowns(plan,app.sources).some(path=>section.totalOf.some(prefix=>path.startsWith(prefix))))return {t:'?',clip:'',cls:'calc dim'};
  const total=section.total(plan);
  return {t:moneyText(total),clip:total,num:total,cls:'calc'};
}
function sameCell(a,b,row){
  if(row.type==='calc')return String(row.value(a))===String(row.value(b));
  if(row.show&&row.show(a)!==row.show(b))return false;
  return getPath(a,row.path)===getPath(b,row.path)&&isUnknown(context(),a,row)===isUnknown(context(),b,row);
}
function planCell(plan,base,row){
  if(row.show&&!row.show(plan))return {t:'',cls:'none'};
  if(row.type==='calc'){
    if(row.needs?.some(path=>blockingUnknowns(plan,app.sources).includes(path)))return {t:'',cls:'none'};
    const value=row.value(plan);
    return {t:row.money?moneyText(value):String(value),clip:value,num:row.money?value:undefined,cls:'calc'};
  }
  const value=getPath(plan,row.path),ctx=context(),unknown=isUnknown(ctx,plan,row),inactive=row.active&&!row.active(plan);
  const proLocked=row.proOnly&&!isPro();
  if(proLocked)return {t:String(FREE_SIMULATION_PATHS),clip:FREE_SIMULATION_PATHS,cls:'calc',note:'Free runs use 100 paths. Pro allows up to 10,000.'};
  const diff=plan.id!==base.id&&!sameCell(plan,base,row);
  const text=unknown?'?':cellText(row,value);
  return {t:text,clip:unknown?'':(row.type==='money'?value:text),num:!unknown&&row.type==='money'?value:undefined,cls:`input${diff&&!unknown?' diff':''}${unknown?' unknown':''}${inactive?' dim':''}`,edit:{plan,row}};
}
function forecastModel(){
  const plans=app.plans,states=plans.map(planState),results=states.map(st=>shownResult(st.result,app.basis));
  const table=forecastTable(results),metrics=[['Funded','funded'],['10th pct','p10'],['Median','median'],['90th pct','p90']];
  const cols=[{cls:'c-sticky',width:'52px',letter:'A'}];
  plans.forEach(plan=>metrics.forEach((_,i)=>cols.push({plan,width:i===0?'88px':'92px',letter:letter(cols.length),edge:i===0})));
  const rows=[
    {kind:'head',frozen:true,cells:[{t:'',cls:'c-sticky'},...plans.flatMap(p=>metrics.map((_,i)=>({t:i===0?p.name:'',clip:i===0?p.name:'',cls:i===0?'group-name left':''})))]},
    {kind:'head',frozen:true,lastFrozen:true,cells:[{t:'Age',cls:'c-sticky'},...plans.flatMap(()=>metrics.map(([label])=>({t:label})))]}
  ];
  for(const {age,plans:cells} of table)rows.push({kind:'data',cells:[{t:String(age),clip:age,cls:'c-sticky calc'},...cells.flatMap(cell=>metrics.map(([,key])=>{
    if(!cell)return {t:'',cls:'none'};
    if(key==='funded')return {t:fundedLabel(cell.funded,cell.count),clip:fundedLabel(cell.funded,cell.count),cls:'funded calc',style:`--share:${((cell.funded??0)*100).toFixed(1)}%`};
    return {t:moneyText(Math.round(cell[key])),clip:Math.round(cell[key]),num:cell[key],cls:'calc'};
  }))]});
  if(!table.length)rows.push({kind:'data',cells:[{t:'',cls:'c-sticky'},{t:'No results yet',cls:'none left'}]});
  return {cols,rows};
}
function yearlyModel(){
  const plan=app.plans.find(p=>p.id===app.yearlyId)||basePlan(),st=planState(plan),r=st.result;
  const data=r?yearlyTable(r,app.basis):[],used=YEARLY_COLUMNS.filter(([key])=>data.some(row=>Math.abs(row[key])>=.5));
  const cols=[{cls:'c-sticky',width:'58px',letter:'A'},{width:'82px',letter:'B',plan},...used.map(([key],i)=>({width:'88px',letter:letter(i+2),plan,edge:key==='pretax'}))];
  const rows=[{kind:'head',frozen:true,lastFrozen:true,cells:[{t:'Age',cls:'c-sticky'},{t:'Date'},...used.map(([,label])=>({t:label}))]}];
  for(const row of data){
    const months=Math.round(row.age*12),age=months%12?`${Math.floor(months/12)}y ${months%12}m`:String(months/12);
    const date=row.date?cellText({type:'date'},row.date):'';
    rows.push({kind:'data',cells:[{t:age,clip:row.age,cls:'c-sticky calc'},{t:date,cls:'calc'},...used.map(([key])=>{const v=Math.round(row[key]);return {t:v?moneyText(v):'–',clip:v,num:row[key],cls:`calc${key==='unfunded'&&v>0?' bad':''}`};})]});
  }
  if(!data.length)rows.push({kind:'data',cells:[{t:'',cls:'c-sticky'},{t:st.blocked?'Fill the blank cells first':'Calculating…',cls:'none left'}]});
  return {cols,rows};
}
const buildModel=()=>app.tab==='forecast'?forecastModel():app.tab==='yearly'?yearlyModel():planModel();

// ---------- Rendering ----------
const grid=$('grid'),scroller=$('scroller');
const cssPixels=(name,fallback)=>parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name))||fallback;
function renderGrid(){
  const m=app.model=buildModel();
  const head=`<colgroup><col style="width:var(--rownum)">${m.cols.map(c=>`<col style="width:${c.width}">`).join('')}<col></colgroup><thead><tr><th class="corner" scope="col"></th>${m.cols.map((c,i)=>`<th scope="col" data-col="${i}" class="${c.cls||''}" aria-colindex="${i+2}">${c.letter}${(c.tags||[]).map(tag=>`<span class="tag ${tag}">${tag}</span>`).join('')}</th>`).join('')}<th class="fill" aria-hidden="true">${app.tab==='plan'?'<button class="add-column" tabindex="-1" data-add-column title="Add a column that copies the selected plan">+ Plan</button>':''}</th></tr></thead>`;
  const rowHeight=cssPixels('--row',24);
  let top=cssPixels('--head',22);
  const body=m.rows.map((row,r)=>{
    const classes=[row.kind,row.frozen?'frozen':'',row.lastFrozen?'last':'',row.tall?'tall':''].filter(Boolean).join(' ');
    const stickyTop=row.frozen?top:0,height=row.tall?38:rowHeight;
    if(row.frozen)top+=height;
    const style=row.frozen?` style="top:${stickyTop}px"`:'';
    const cells=row.cells.map((cell,c)=>{
      const cls=[cell.cls||'',m.cols[c]?.edge?'edge':''].filter(Boolean).join(' ');
      const readOnly=cell.edit?'':' aria-readonly="true"';
      const expanded=cell.expanded!==undefined?` aria-expanded="${cell.expanded}"`:'';
      const inline=[row.frozen?`top:${stickyTop}px`:'',cell.style||''].filter(Boolean).join(';');
      return `<td id="c${r}-${c}" role="gridcell" data-r="${r}" data-c="${c}" class="${cls}"${readOnly}${expanded}${inline?` style="${inline}"`:''}>${cell.html??escapeHTML(cell.t)}</td>`;
    }).join('');
    return `<tr class="${classes}" aria-rowindex="${r+2}"><th class="rownum" scope="row" data-row="${r}"${style}>${r+1}</th>${cells}<td class="fill" aria-hidden="true"${style}></td></tr>`;
  }).join('');
  grid.innerHTML=`${head}<tbody>${body}</tbody>`;
  // A fixed-layout table needs a definite width; the filler column takes any spare room.
  grid.style.width=`max(100%, calc(var(--rownum) + ${m.cols.map(c=>c.width).join(' + ')}))`;
  grid.setAttribute('aria-rowcount',m.rows.length+1);grid.setAttribute('aria-colcount',m.cols.length+1);
  clampSelection();paintSelection();
}
const cellEl=(r,c)=>grid.tBodies[0]?.rows[r]?.cells[c+1];
const activeCell=()=>app.model?.rows[app.sel.r]?.cells[app.sel.c];
function clampSelection(){
  const m=app.model,maxR=m.rows.length-1,maxC=m.cols.length-1,s=app.sel;
  s.r=Math.min(s.r,maxR);s.ar=Math.min(s.ar,maxR);s.c=Math.min(s.c,maxC);s.ac=Math.min(s.ac,maxC);
}
const rangeBounds=()=>{const {r,c,ar,ac}=app.sel;return {top:Math.min(r,ar),bottom:Math.max(r,ar),left:Math.min(c,ac),right:Math.max(c,ac)};};
function paintSelection(){
  grid.querySelectorAll('.sel,.in-range,.on').forEach(el=>el.classList.remove('sel','in-range','on'));
  grid.querySelectorAll('[aria-selected]').forEach(el=>el.removeAttribute('aria-selected'));
  const {top,bottom,left,right}=rangeBounds(),multi=top!==bottom||left!==right;
  const body=grid.tBodies[0];if(!body)return;
  for(let r=top;r<=bottom;r++){
    const tr=body.rows[r];if(!tr)continue;
    tr.cells[0].classList.add('on');
    if(multi)for(let c=left;c<=right;c++)tr.cells[c+1]?.classList.add('in-range');
  }
  const headCells=grid.tHead.rows[0].cells;
  for(let c=left;c<=right;c++)headCells[c+1]?.classList.add('on');
  const active=cellEl(app.sel.r,app.sel.c);
  if(active){active.classList.add('sel');active.setAttribute('aria-selected','true');grid.setAttribute('aria-activedescendant',active.id);}
  renderFormula();renderNote();renderAggregate();renderIssues();
}
function renderFormula(){
  if(app.edit)return;
  const cell=activeCell(),input=$('formula-input');
  $('cell-name').textContent=app.model?letter(app.sel.c)+(app.sel.r+1):'';
  const edit=cell?.edit;
  if(edit&&edit.row.path!=='name'&&isUnknown(context(),edit.plan,edit.row))input.value='';
  else input.value=edit?(edit.row.path==='name'?edit.plan.name:editText(edit.row,getPath(edit.plan,edit.row.path))):String(cell?.t??'');
  input.readOnly=!edit;
}
function setNote(text,error=false){app.noteOverride=text;app.noteError=error;renderNote();}
function renderNote(){
  const el=$('note'),cell=activeCell(),row=app.model?.rows[app.sel.r];
  el.classList.toggle('error',Boolean(app.noteOverride)&&Boolean(app.noteError));
  el.textContent=app.noteOverride||cell?.note||row?.note||(row?.section?`${row.section.label}: Space shows or hides this block.`:'');
}
function renderAggregate(){
  const {top,bottom,left,right}=rangeBounds(),values=[];
  if(top===bottom&&left===right){$('aggregate').textContent='';return;}
  let count=0;
  for(let r=top;r<=bottom;r++)for(let c=left;c<=right;c++){const cell=app.model.rows[r]?.cells[c];if(!cell)continue;if(String(cell.t??'')!=='')count++;if(typeof cell.num==='number')values.push(cell.num);}
  if(!values.length){$('aggregate').textContent=`Count ${count}`;return;}
  const total=values.reduce((a,b)=>a+b,0);
  $('aggregate').textContent=`Sum ${moneyText(Math.round(total))} · Avg ${moneyText(Math.round(total/values.length))} · Count ${count}`;
}
function renderIssues(){
  const cell=activeCell(),panel=$('issues'),issues=cell?.issues;
  const plan=app.model?.cols[app.sel.c]?.plan;
  const list=issues||(app.model?.rows[app.sel.r]?.id==='readiness'&&plan?planIssues(plan,app.sources).filter(i=>i.level!=='note'):null);
  if(app.tab!=='plan'||!list?.length){panel.hidden=true;panel.innerHTML='';return;}
  panel.hidden=false;
  panel.innerHTML=`<ul>${list.map(item=>`<li class="${item.level}">${item.path&&rowForPath(item.path)?`<button data-jump="${escapeHTML(item.path)}" data-plan="${escapeHTML(plan.id)}">${escapeHTML(item.text)}</button>`:escapeHTML(item.text)}</li>`).join('')}</ul>`;
}
function renderCalc(){
  const jobs=[...app.jobs.values()],running=jobs.length,button=$('recalc');
  if(running){
    const done=jobs.reduce((n,j)=>n+(j.progress||0),0),totalCount=jobs.reduce((n,j)=>n+(j.count||0),0);
    $('calc').textContent=`Calculating ${Math.round(done/running*100)}%${totalCount>running*LIVE_ESTIMATE_PATHS?` · ${totalCount.toLocaleString('en-US')} paths`:''}`;
  }else $('calc').textContent='Ready';
  const stopping=running&&app.full;
  button.classList.toggle('stopping',Boolean(stopping));
  button.innerHTML=stopping?'Stop':'Full run <kbd>F9</kbd>';
}
function renderToolbar(){
  $('undo').disabled=!app.undo.length;$('redo').disabled=!app.redo.length;
  $('delete-plan').disabled=app.plans.length===1;
  $('diff-only').setAttribute('aria-pressed',String(app.diffOnly));
  $('basis-today').setAttribute('aria-pressed',String(app.basis==='today'));
  $('basis-future').setAttribute('aria-pressed',String(app.basis==='future'));
  $('access').innerHTML=isPro()?'Pro':`${FREE_SIMULATION_PATHS}-path preview · <a href="./#/billing">Pro</a>`;
  for(const tab of document.querySelectorAll('[data-tab]')){if(tab.dataset.tab===app.tab)tab.setAttribute('aria-current','page');else tab.removeAttribute('aria-current');}
  const onPlans=app.tab==='plan';
  for(const id of ['add-copy','add-blank','set-base','delete-plan','diff-only'])$(id).hidden=!onPlans;
  const caption=$('caption');
  if(app.tab==='yearly'){
    caption.hidden=false;
    caption.innerHTML=`<label>Plan <select id="yearly-plan">${app.plans.map(p=>`<option value="${escapeHTML(p.id)}"${p.id===app.yearlyId?' selected':''}>${escapeHTML(p.name)}</option>`).join('')}</select></label><span>Steady-growth illustration: average returns every year, no downturns or care costs. Not a forecast or a median.</span>`;
  }else if(app.tab==='forecast'){
    caption.hidden=false;caption.innerHTML='<span>Monte Carlo by age. Balances cover lifetimes still observed at that age. Funded counts lifetimes with no shortfall so far.</span>';
  }else{caption.hidden=true;caption.innerHTML='';}
}
function refresh({keepCalc=false}={}){
  renderGrid();renderToolbar();
  if(!keepCalc)scheduleRecalc();
  renderCalc();
  if(app.edit)placeEditor();
}

// ---------- Selection and movement ----------
function frozenOffsets(){
  const head=grid.tHead.rows[0].getBoundingClientRect().height;
  let top=head;
  for(const tr of grid.tBodies[0].rows){if(!tr.classList.contains('frozen'))break;top+=tr.getBoundingClientRect().height;}
  let left=0;
  const firstRow=grid.tBodies[0].rows[0];
  if(firstRow)for(const cell of firstRow.cells)if(/\b(rownum|c-label|c-unit|c-sticky)\b/.test(cell.className))left+=cell.getBoundingClientRect().width;
  return {top,left};
}
function reveal(r,c){
  const cell=cellEl(r,c);if(!cell)return;
  const box=cell.getBoundingClientRect(),view=scroller.getBoundingClientRect(),{top,left}=frozenOffsets();
  const frozenRow=cell.parentElement.classList.contains('frozen'),frozenCol=/\bc-(label|unit|sticky)\b/.test(cell.className);
  if(!frozenRow){
    if(box.top<view.top+top)scroller.scrollTop-=view.top+top-box.top;
    else if(box.bottom>view.bottom)scroller.scrollTop+=box.bottom-view.bottom;
  }
  if(!frozenCol){
    if(box.left<view.left+left)scroller.scrollLeft-=view.left+left-box.left;
    else if(box.right>view.right)scroller.scrollLeft+=box.right-view.right;
  }
}
function moveTo(r,c,extend=false){
  const m=app.model;
  r=Math.max(0,Math.min(m.rows.length-1,r));c=Math.max(0,Math.min(m.cols.length-1,c));
  app.sel.r=r;app.sel.c=c;
  if(!extend){app.sel.ar=r;app.sel.ac=c;}
  app.noteOverride='';
  paintSelection();reveal(r,c);
}
function pageRows(){return Math.max(1,Math.floor((scroller.clientHeight-frozenOffsets().top)/cssPixels('--row',24))-1);}
function toggleSection(id){
  if(app.collapsed.has(id))app.collapsed.delete(id);else app.collapsed.add(id);
  try{localStorage.setItem(UI_KEY,JSON.stringify({collapsed:[...app.collapsed]}));}catch{}
  renderGrid();
}

// ---------- Editing ----------
const editor=$('cell-editor'),picker=$('cell-select'),formulaInput=$('formula-input');
function placeEditor(){
  const cell=cellEl(app.edit?.r,app.edit?.c);if(!cell)return;
  const box=cell.getBoundingClientRect(),target=app.edit.picker?picker:editor;
  target.style.left=box.left+'px';target.style.top=box.top+'px';target.style.height=box.height+'px';
  target.style.width=Math.max(box.width,app.edit.picker?150:90)+'px';
}
function beginEdit(initial=null,{fromBar=false}={}){
  const cell=activeCell(),edit=cell?.edit;
  if(!edit){setNote(cell?.expanded!==undefined?'Press Space to show or hide this block.':'This cell is calculated.');return;}
  const {plan,row}=edit;
  if(row.type==='bool'&&initial===null){setValues([[app.sel.r,app.sel.c,!getPath(plan,row.path)]]);return;}
  if(row.type==='enum'&&initial===null){openPicker(plan,row);return;}
  const unknown=isUnknown(context(),plan,row);
  const text=initial??(row.path==='name'?plan.name:unknown?'':editText(row,getPath(plan,row.path)));
  app.edit={r:app.sel.r,c:app.sel.c,plan,row,fromBar,opened:initial===null?text:null};
  document.body.classList.add('editing');
  editor.hidden=false;editor.value=text;formulaInput.value=text;formulaInput.readOnly=false;
  editor.style.textAlign=row.path==='name'?'left':'right';
  editor.removeAttribute('aria-invalid');
  reveal(app.edit.r,app.edit.c);placeEditor();
  if(fromBar){app.edit.opened=text;formulaInput.focus();}
  else{
    editor.focus();
    // On a touch screen the keyboard takes over, so typing should replace the value.
    if(initial===null&&lastPointer!=='mouse')editor.select();
    else editor.setSelectionRange(editor.value.length,editor.value.length);
  }
}
function openPicker(plan,row){
  app.edit={r:app.sel.r,c:app.sel.c,plan,row,picker:true};
  document.body.classList.add('editing');
  picker.innerHTML=row.options.map(([value,label],i)=>`<option value="${i}"${value===getPath(plan,row.path)?' selected':''}>${escapeHTML(label)}</option>`).join('');
  picker.hidden=false;reveal(app.edit.r,app.edit.c);placeEditor();picker.focus();
  try{picker.showPicker?.();}catch{}
}
function endEdit(){
  app.edit=null;editor.hidden=true;picker.hidden=true;editor.value='';
  document.body.classList.remove('editing');
  renderFormula();grid.focus({preventScroll:true});
}
function cancelEdit(){if(app.edit){endEdit();app.noteOverride='';renderNote();}}
function commitEdit(dr=0,dc=0){
  const edit=app.edit;if(!edit)return true;
  const text=(document.activeElement===formulaInput?formulaInput:editor).value;
  // Opening a cell and leaving it unchanged must never alter the stored value.
  if(text===edit.opened){endEdit();moveTo(edit.r+dr,edit.c+dc);return true;}
  const out=parseCell(edit.row,text,edit.plan);
  if(out.error){editor.setAttribute('aria-invalid','true');setNote(out.error,true);return false;}
  pushUndo();
  if(out.clear)clearCell(context(),edit.plan,edit.row);else applyCell(context(),edit.plan,edit.row,out.value);
  persist();endEdit();app.noteOverride='';
  refresh();moveTo(findRow(edit.row,edit.r)+dr,edit.c+dc);
  return true;
}
// Rows can shift when an edit shows or hides spouse rows, so find it again.
function findRow(row,fallback){const at=app.model.rows.findIndex(r=>r.row===row||r.cells[app.sel.c]?.edit?.row===row);return at>=0?at:fallback;}
// Writes several cells as one undo step. Skips calculated cells. Returns rejected count.
function setValues(list){
  const ctx=context();let changed=false,rejected=0;const pending=[];
  for(const [r,c,value,raw] of list){
    const edit=app.model.rows[r]?.cells[c]?.edit;if(!edit){if(raw!==undefined)rejected++;continue;}
    pending.push([edit,value,raw]);
  }
  if(!pending.length)return rejected;
  pushUndo();
  for(const [edit,value,raw] of pending){
    if(raw!==undefined){
      const out=parseCell(edit.row,raw,edit.plan);
      if(out.error){rejected++;continue;}
      if(out.clear)clearCell(ctx,edit.plan,edit.row);else applyCell(ctx,edit.plan,edit.row,out.value);
    }else if(value===UNKNOWN)clearCell(ctx,edit.plan,edit.row);
    else applyCell(ctx,edit.plan,edit.row,value);
    changed=true;
  }
  if(changed){persist();refresh();}else app.undo.pop();
  return rejected;
}
const UNKNOWN=Symbol('unknown');
function clearSelection(){
  const {top,bottom,left,right}=rangeBounds(),list=[];
  for(let r=top;r<=bottom;r++)for(let c=left;c<=right;c++){
    const edit=app.model.rows[r]?.cells[c]?.edit;
    if(edit&&edit.row.path!=='name'&&!['bool','enum'].includes(edit.row.type))list.push([r,c,UNKNOWN]);
  }
  if(list.length){setValues(list);setNote('Blank means Unknown, not zero. Enter 0 for none.');}
  else setNote('This cell can’t be blanked.');
}
function fillRight(){
  const {top,bottom,left,right}=rangeBounds(),from=left===right?left-1:left,to=right,list=[],ctx=context();
  for(let r=top;r<=bottom;r++){
    const source=app.model.rows[r]?.cells[from]?.edit;if(!source||source.row.path==='name')continue;
    for(let c=from+1;c<=to;c++){
      const target=app.model.rows[r]?.cells[c]?.edit;if(!target||target.row!==source.row)continue;
      list.push([r,c,isUnknown(ctx,source.plan,source.row)?UNKNOWN:structuredClone(getPath(source.plan,source.row.path))]);
    }
  }
  if(list.length)setValues(list);else setNote('Select a cell with a plan to its left.');
}
function selectionRows(){
  const {top,bottom,left,right}=rangeBounds(),rows=[];
  for(let r=top;r<=bottom;r++){const line=[];for(let c=left;c<=right;c++){const cell=app.model.rows[r]?.cells[c];line.push(cell?.clip??cell?.t??'');}rows.push(line);}
  return rows;
}
function pasteText(text){
  const data=parseTsv(text),{top,bottom,left,right}=rangeBounds(),list=[];
  const single=data.length===1&&data[0].length===1;
  const rows=single?bottom-top+1:data.length,cols=single?right-left+1:Math.max(...data.map(l=>l.length));
  for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){
    const raw=single?data[0][0]:data[i][j];if(raw===undefined)continue;
    list.push([top+i,left+j,null,raw]);
  }
  const rejected=setValues(list);
  setNote(rejected?`${rejected} cell${rejected===1?'':'s'} skipped: calculated, or the value doesn’t fit.`:`Pasted ${list.length} cell${list.length===1?'':'s'}.`,rejected>0);
}
function downloadCsv(){
  const m=app.model,name=app.tab==='yearly'?(app.plans.find(p=>p.id===app.yearlyId)?.name||'plan')+' yearly':app.tab==='forecast'?'forecast':'plans';
  const rows=m.rows.map(row=>row.cells.map(cell=>cell.clip??cell.t??''));
  const url=URL.createObjectURL(new Blob([toCsv(rows)],{type:'text/csv'})),a=document.createElement('a');
  a.href=url;a.download=`retirement-${name.replace(/[^a-z0-9]+/gi,'-').toLowerCase()}.csv`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function setTab(tab){
  if(app.edit&&!commitEdit())return;
  const [r,c]={plan:[1,2],forecast:[2,0],yearly:[1,0]}[tab];
  app.tab=tab;app.sel={r,c,ar:r,ac:c};
  history.replaceState(null,'',location.pathname+location.search+(tab==='plan'?'':'#'+tab));
  app.noteOverride='';scroller.scrollTop=0;scroller.scrollLeft=0;refresh({keepCalc:true});grid.focus({preventScroll:true});
}

// ---------- Switching to and from the planner ----------
const focusedPlan=()=>(app.tab==='yearly'&&app.plans.find(p=>p.id===app.yearlyId))||selectedPlan();
// The planner step that edits the selected row. Results and the plan name have none.
function selectedStep(){
  const at=app.model.rows[app.sel.r];
  if(!at||at.kind==='name'||at.kind==='result')return null;
  const section=at.section||SECTIONS.find(s=>s.rows.includes(at.row));
  const path=at.row?.path||section?.rows.find(row=>row.path)?.path;
  return path?inputTask(path).step:null;
}
function placeOnArrival(){
  const handedOff=takeHandedOffPlan(),plan=app.plans.find(p=>p.id===handedOff)||basePlan();
  const {tab,step}=parseSheetHash(location.hash);
  if(tab){app.yearlyId=plan.id;setTab(tab);return;}
  const c=app.model.cols.findIndex(col=>col.plan?.id===plan.id);
  if(step===undefined){moveTo(1,c);return;}
  const section=SECTIONS.find(s=>s.rows.some(row=>row.path&&inputTask(row.path).step===step));
  if(section&&app.collapsed.has(section.id))toggleSection(section.id);
  const r=app.model.rows.findIndex(row=>row.kind==='data'&&row.row.path&&inputTask(row.row.path).step===step);
  moveTo(r>=0?r:1,c);
  // Show the row's section right under the frozen results, not at the bottom edge.
  const heading=r>0&&grid.tBodies[0].rows[r-1];
  if(heading)scroller.scrollTop+=heading.getBoundingClientRect().top-scroller.getBoundingClientRect().top-frozenOffsets().top;
  history.replaceState(null,'',location.pathname+location.search);
}
// Every link into the planner saves first and opens on the plan in focus. Only the
// switch changes where this visitor lands next time.
document.addEventListener('click',async e=>{
  const link=e.target.closest?.('a[href^="./#/"]');
  if(!link||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;
  e.preventDefault();
  if(app.edit&&!commitEdit())return;
  const isSwitch=link.id==='to-planner',target=isSwitch?plannerHref(app.tab,selectedStep()):link.getAttribute('href'),plan=focusedPlan();
  await saveQueue;
  handOffPlan(plan.id);
  if(isSwitch)rememberView('planner');
  location.assign(target);
});

// ---------- Events ----------
grid.addEventListener('keydown',e=>{
  if(app.edit)return;
  const ctrl=e.ctrlKey||e.metaKey,{r,c}=app.sel,m=app.model,cell=activeCell(),edit=cell?.edit,key=e.key;
  const go=(dr,dc,extend=e.shiftKey)=>{e.preventDefault();moveTo(r+dr,c+dc,extend);};
  switch(key){
    case 'ArrowUp':return ctrl?go(-r,0):go(-1,0);
    case 'ArrowDown':return ctrl?go(m.rows.length-1-r,0):go(1,0);
    case 'ArrowLeft':return ctrl?go(0,-c):go(0,-1);
    case 'ArrowRight':return ctrl?go(0,m.cols.length-1-c):go(0,1);
    case 'Tab':{
      // At the edge of the sheet Tab leaves the grid, so keyboard users are never trapped.
      const next=c+(e.shiftKey?-1:1);
      if(next<0||next>=m.cols.length)return;
      e.preventDefault();return moveTo(r,next);
    }
    case 'Enter':e.preventDefault();return moveTo(r+(e.shiftKey?-1:1),c);
    case 'Home':return ctrl?go(-r,-c):go(0,-c);
    case 'End':return ctrl?go(m.rows.length-1-r,m.cols.length-1-c):go(0,m.cols.length-1-c);
    case 'PageUp':if(ctrl){e.preventDefault();return setTab(['plan','forecast','yearly'][(['plan','forecast','yearly'].indexOf(app.tab)+2)%3]);}return go(-pageRows(),0);
    case 'PageDown':if(ctrl){e.preventDefault();return setTab(['plan','forecast','yearly'][(['plan','forecast','yearly'].indexOf(app.tab)+1)%3]);}return go(pageRows(),0);
    case 'F2':e.preventDefault();return beginEdit();
    case 'F9':e.preventDefault();return fullRun();
    case 'Escape':app.noteOverride='';return renderNote();
    case 'Delete':case 'Backspace':e.preventDefault();return clearSelection();
    case ' ':
      e.preventDefault();
      if(cell?.expanded!==undefined||m.rows[r]?.kind==='section')return toggleSection(m.rows[r].section.id);
      if(edit?.row.type==='bool')return beginEdit();
      if(edit?.row.type==='enum'){const options=edit.row.options,at=options.findIndex(([v])=>v===getPath(edit.plan,edit.row.path));return void setValues([[r,c,options[(at+1)%options.length][0]]]);}
      return;
  }
  if(ctrl){
    const k=key.toLowerCase();
    if(k==='z'){e.preventDefault();return e.shiftKey?redo():undo();}
    if(k==='y'){e.preventDefault();return redo();}
    if(k==='r'){e.preventDefault();return fillRight();}
    if(k==='c'){navigator.clipboard?.writeText?.(toTsv(selectionRows())).catch(()=>{});return;}
    if(k==='v'&&app.tab==='plan'){pasteSeen=false;setTimeout(()=>{if(!pasteSeen)navigator.clipboard?.readText?.().then(pasteText).catch(()=>{});},40);return;}
    if(k==='a'){e.preventDefault();app.sel={r:0,c:0,ar:m.rows.length-1,ac:m.cols.length-1};return paintSelection();}
    return;
  }
  if(e.altKey&&key==='ArrowDown'&&edit?.row.type==='enum'){e.preventDefault();return beginEdit();}
  // A printable key starts typing over the cell, as in any spreadsheet.
  if(key.length===1&&!e.altKey){
    if(!edit){setNote(cell?'This cell is calculated.':'');return;}
    if(edit.row.type==='bool'||edit.row.type==='enum'){
      e.preventDefault();
      const out=parseCell(edit.row,key,edit.plan);
      if(out.error)setNote(out.error,true);else setValues([[r,c,out.value]]);
      return;
    }
    e.preventDefault();beginEdit(key);
  }
});
function editorKeys(e){
  if(e.key==='Enter'){e.preventDefault();commitEdit(e.shiftKey?-1:1,0);}
  else if(e.key==='Tab'){e.preventDefault();commitEdit(0,e.shiftKey?-1:1);}
  else if(e.key==='Escape'){e.preventDefault();cancelEdit();}
  else if(!e.isComposing&&(e.key==='ArrowUp'||e.key==='ArrowDown')&&document.activeElement===editor){e.preventDefault();commitEdit(e.key==='ArrowUp'?-1:1,0);}
}
editor.addEventListener('keydown',editorKeys);
formulaInput.addEventListener('keydown',e=>{if(app.edit)editorKeys(e);});
editor.addEventListener('input',()=>{formulaInput.value=editor.value;editor.removeAttribute('aria-invalid');if(app.noteError){app.noteOverride='';renderNote();}});
formulaInput.addEventListener('focus',()=>{if(!app.edit&&activeCell()?.edit)beginEdit(formulaInput.value,{fromBar:true});});
formulaInput.addEventListener('input',()=>{editor.value=formulaInput.value;});
// Leaving the editor saves, as in a spreadsheet. A bad value keeps the editor open.
editor.addEventListener('blur',e=>{if(e.relatedTarget===formulaInput||!app.edit||app.edit.picker)return;if(!commitEdit())editor.focus();});
formulaInput.addEventListener('blur',e=>{if(e.relatedTarget===editor||!app.edit||app.edit.picker)return;if(!commitEdit())formulaInput.focus();});
picker.addEventListener('change',()=>{const edit=app.edit;if(!edit?.picker)return;const value=edit.row.options[Number(picker.value)][0];endEdit();setValues([[app.sel.r,app.sel.c,value]]);});
picker.addEventListener('keydown',e=>{
  if(e.key==='Enter'){
    e.preventDefault();
    const value=app.edit.row.options[Number(picker.value)][0],{r,c}=app.sel;
    endEdit();setValues([[r,c,value]]);
  }else if(e.key==='Escape'){e.preventDefault();cancelEdit();}else if(e.key==='Tab'){e.preventDefault();const d=e.shiftKey?-1:1;cancelEdit();moveTo(app.sel.r,app.sel.c+d);}});
picker.addEventListener('blur',()=>{if(app.edit?.picker)cancelEdit();});
picker.addEventListener('cancel',cancelEdit);

let dragging=false,lastPointer='mouse',tappedActive=false,pasteSeen=false;
grid.addEventListener('pointerdown',e=>{lastPointer=e.pointerType||'mouse';});
grid.addEventListener('mousedown',e=>{
  const toggle=e.target.closest?.('[data-toggle]');
  if(toggle){e.preventDefault();toggleSection(toggle.dataset.toggle);grid.focus({preventScroll:true});return;}
  if(e.target.closest?.('[data-add-column]')){e.preventDefault();addCopy();grid.focus({preventScroll:true});return;}
  const head=e.target.closest?.('th');
  if(head){
    e.preventDefault();
    if(head.classList.contains('fill'))return;
    if(app.edit&&!commitEdit())return;
    if(head.dataset.col!==undefined){const c=Number(head.dataset.col);app.sel={r:0,c,ar:app.model.rows.length-1,ac:c};paintSelection();}
    else if(head.dataset.row!==undefined){const r=Number(head.dataset.row);app.sel={r,c:0,ar:r,ac:app.model.cols.length-1};paintSelection();}
    grid.focus({preventScroll:true});return;
  }
  const td=e.target.closest?.('td');if(!td||td.dataset.r===undefined)return;
  e.preventDefault();
  const r=Number(td.dataset.r),c=Number(td.dataset.c);
  if(app.edit){if(app.edit.r===r&&app.edit.c===c)return;if(!commitEdit())return;}
  tappedActive=app.sel.r===r&&app.sel.c===c&&app.sel.ar===r&&app.sel.ac===c;
  grid.focus({preventScroll:true});moveTo(r,c,e.shiftKey);dragging=!e.shiftKey&&lastPointer==='mouse';
  // A second tap edits, because touch screens have no double click or F2.
  if(tappedActive&&lastPointer!=='mouse')beginEdit();
});
grid.addEventListener('mouseover',e=>{
  if(!dragging||!(e.buttons&1))return;
  const td=e.target.closest?.('td');if(!td||td.dataset.r===undefined)return;
  moveTo(Number(td.dataset.r),Number(td.dataset.c),true);
});
window.addEventListener('mouseup',()=>{dragging=false;});
grid.addEventListener('dblclick',e=>{if(e.target.closest?.('td[data-r]')&&!e.target.closest('[data-toggle]')&&!app.edit)beginEdit();});
scroller.addEventListener('scroll',()=>{if(app.edit)placeEditor();},{passive:true});
window.addEventListener('resize',()=>{renderGrid();if(app.edit)placeEditor();});

document.addEventListener('copy',e=>{
  const active=document.activeElement;
  if(app.edit||active===formulaInput||active===editor||active===picker)return;
  e.clipboardData.setData('text/plain',toTsv(selectionRows()));e.preventDefault();
  const {top,bottom,left,right}=rangeBounds();setNote(`Copied ${(bottom-top+1)*(right-left+1)} cell${(bottom-top+1)*(right-left+1)===1?'':'s'}.`);
});
document.addEventListener('paste',e=>{
  const active=document.activeElement;
  if(app.edit||active===formulaInput||active===editor||active===picker||app.tab!=='plan')return;
  pasteSeen=true;
  const text=e.clipboardData.getData('text/plain');if(!text)return;
  e.preventDefault();pasteText(text);
});
// F9 and Ctrl+PgUp work from anywhere, not only with the grid focused.
document.addEventListener('keydown',e=>{
  if(e.key==='F9'&&e.target!==grid){e.preventDefault();fullRun();}
  const typing=e.target instanceof HTMLInputElement||e.target instanceof HTMLSelectElement;
  if((e.ctrlKey||e.metaKey)&&!typing&&e.target!==grid){
    const k=e.key.toLowerCase();
    if(k==='z'){e.preventDefault();e.shiftKey?redo():undo();}
    else if(k==='y'){e.preventDefault();redo();}
  }
  if(e.key==='Escape'&&!$('keys-panel').hidden){$('keys-panel').hidden=true;$('keys').setAttribute('aria-expanded','false');}
});

// Like a spreadsheet ribbon, toolbar actions hand the keyboard straight back to the sheet.
document.querySelector('.toolbar').addEventListener('click',e=>{
  if(e.target.closest('button')&&e.target.closest('#keys')===null)setTimeout(()=>grid.focus({preventScroll:true}));
});
$('add-copy').addEventListener('click',addCopy);
$('add-blank').addEventListener('click',addBlank);
$('delete-plan').addEventListener('click',deletePlan);
$('set-base').addEventListener('click',setBase);
$('undo').addEventListener('click',undo);
$('redo').addEventListener('click',redo);
$('recalc').addEventListener('click',fullRun);
$('csv').addEventListener('click',downloadCsv);
$('diff-only').addEventListener('click',()=>{app.diffOnly=!app.diffOnly;renderGrid();renderToolbar();});
$('keys').addEventListener('click',()=>{const panel=$('keys-panel');panel.hidden=!panel.hidden;$('keys').setAttribute('aria-expanded',String(!panel.hidden));});
for(const [id,basis] of [['basis-today','today'],['basis-future','future']])$(id).addEventListener('click',()=>{app.basis=basis;try{localStorage.setItem(BASIS_KEY,basis);}catch{}refresh({keepCalc:true});});
for(const tab of document.querySelectorAll('[data-tab]'))tab.addEventListener('click',()=>setTab(tab.dataset.tab));
$('caption').addEventListener('change',e=>{if(e.target.id==='yearly-plan'){app.yearlyId=e.target.value;refresh({keepCalc:true});}});
$('issues').addEventListener('click',e=>{
  const button=e.target.closest('[data-jump]');if(!button)return;
  const row=app.model.rows.findIndex(r=>r.row?.path===button.dataset.jump),c=app.model.cols.findIndex(col=>col.plan?.id===button.dataset.plan);
  if(row>=0&&c>=0){moveTo(row,c);grid.focus({preventScroll:true});return;}
  const section=SECTIONS.find(s=>s.rows.some(x=>x.path===button.dataset.jump));
  if(section&&app.collapsed.has(section.id)){toggleSection(section.id);const at=app.model.rows.findIndex(r=>r.row?.path===button.dataset.jump);if(at>=0&&c>=0){moveTo(at,c);grid.focus({preventScroll:true});}}
});
window.addEventListener('storage',event=>{
  if(event.key!==STORE_KEY&&event.key!==null)return;
  try{if(localStorage.getItem(STORE_KEY)===lastSavedRaw)return;}catch{return;}
  if(app.edit||pendingSaves)return;
  try{loadInto();app.undo=[];app.redo=[];refresh();setNote('Updated from another tab.');}catch{storeBlocked=true;setNote('Saved plans couldn’t be read. Open the planner to recover them.',true);}
});
window.addEventListener('beforeunload',e=>{if(pendingSaves){e.preventDefault();e.returnValue='';}});

// Same access check as the classic planner. Any failure means the free preview.
async function loadAccess(){
  try{
    let customer={};
    try{const saved=JSON.parse(sessionStorage.getItem('retirement-billing-reference'));if(typeof saved?.customerId==='string'&&/^cus_[A-Za-z0-9]{1,200}$/.test(saved.customerId))customer={'x-retirement-customer':saved.customerId};}catch{}
    const response=await fetch('/api/billing/status',{credentials:'same-origin',cache:'no-store',headers:{...await authHeaders(),...customer}});
    const access=response.ok?await response.json():null;
    const tier=access?.tier==='pro'?'pro':'free';
    if(tier===app.tier)return;
    app.tier=tier;
    if(tier==='pro')for(const s of app.plans)applyProSimulationDefault(s);
    refresh();
  }catch{/* The free preview keeps working. */}
}
async function start(){
  try{const ui=JSON.parse(localStorage.getItem(UI_KEY));if(Array.isArray(ui?.collapsed))app.collapsed=new Set(ui.collapsed);}catch{}
  try{const basis=localStorage.getItem(BASIS_KEY);if(basis==='today'||basis==='future')app.basis=basis;}catch{}
  try{loadInto();}
  catch{
    // Never overwrite plans this page can't read.
    storeBlocked=true;
    const plans=originalPlans();Object.assign(app,{plans,sources:normalizeInputSources(null,plans,'Sample/default'),optional:normalizeOptionalAnswers(null,plans),selectedId:plans[0].id,yearlyId:plans[0].id});
    setNote('Saved plans couldn’t be read, so these are samples and nothing is saved. Open the planner to recover them.',true);
  }
  refresh();
  if(!storeBlocked)placeOnArrival();
  if(app.repairNote)setNote(app.repairNote);
  grid.focus({preventScroll:true});
  reportUsage('arrival',arrivalReferrer(document.referrer));
  try{await initializeSocialAuth(loadAccess);}catch{}
  loadAccess();
}
start();
