import {INPUT_SOURCES,normalizeInputSources,inputSource,unknownInputPaths,FIELD_GUIDANCE,EARNINGS_EXPLANATION,EARNINGS_POINTS} from './ux-guidance.js';
import {oneYearGrowth} from './growth-helper.js';
import {withdrawalContent} from './withdrawals-view.js';
import {isPreviewResult,readinessLabel,shareLabel,ageYearRows,balanceDisplayRows,PREVIEW_WARNING} from './result-format.js';
import {chartCard,mountCharts,disposeCharts} from './charts.js';
import {initializeSocialAuth,socialState,authHeaders,signInSocial,linkSocialProvider,signOutSocial} from './auth.js';
import {baseScenario,retirementAge,primaryRetirementAge,ageLabel,calendarDate,localCalendarDate,prepareCalendarScenario,scenarioTimeline,dateLabel,syncCalendarAges,delayRetirement,sampleScenarios,normalizeScenarios,applyProSimulationDefault,validateScenario,validateScenarioStructure,budgetBreakdown,budgetMonthTotals,validateBudget,markBudgetEdited,applyBudgetEstimate,setAnnualBaseSpending,ruleOf55Applies,earlyWithdrawalContext,ANNUAL_BILLS,SEPARATE_COSTS,ROTH_CONVERSION_RATES,scenarioWarnings,ENGINE_VERSION,scenarioEngineVersion,DEFAULT_SEED,FREE_SIMULATION_PATHS,MIN_SIMULATION_PATHS,MAX_SIMULATION_PATHS,MAX_DOLLAR_AMOUNT} from './model.js';

const $=s=>document.querySelector(s),escapeHTML=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=(v,d=0)=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:d}).format(v||0);
const pct=v=>`${(100*(v||0)).toFixed(1)}%`;
const deep=structuredClone;
const storageKey='retirement-readiness-lab-sites-v1';
const STORAGE_CONFLICT='Error: Saved plans changed in another tab. This tab’s edits have not been saved. Export a backup of this tab, then reload to load the newer saved plans.';
const ACCESS_UNAVAILABLE='Subscription status is unavailable. New runs use the 10-path free preview; your completed results are kept.';
const ACCESS_RETRY='Subscription status is temporarily unavailable. Your last verified Pro access is available for up to five minutes; your completed results are kept.';
const PAYMENT_PENDING='Payment is being confirmed. Refresh your plan in a moment.';
const SIGN_IN_AGAIN='Sign in again to verify your plan. Your completed results are still available.';
const GOOGLE_UNAVAILABLE='Google sign-in is temporarily unavailable. ChatGPT sign-in still works.';
const TARGETS_NEED_PRO='Planning targets require Pro because 10 paths are too coarse.';
const FREE_PATHS_ONLY='10 paths are available in the free preview.';
const LINK_HINT='To share a subscription, use the account-linking button below after both sign-ins are active.';
const CHECKOUT_CANCELED='Checkout was canceled.';
const CALCULATION_CANCELED='Calculation canceled. Completed results are unchanged.';
const TARGETS_CANCELED='Target search canceled.';
const COMPARISONS_CANCELED='Comparisons canceled. Completed rows are kept.';
// Warnings and neutral information use the plain notice style, not success styling.
const NEUTRAL_MESSAGES=new Set([ACCESS_UNAVAILABLE,ACCESS_RETRY,PAYMENT_PENDING,SIGN_IN_AGAIN,GOOGLE_UNAVAILABLE,TARGETS_NEED_PRO,FREE_PATHS_ONLY,LINK_HINT,CHECKOUT_CANCELED,CALCULATION_CANCELED,TARGETS_CANCELED,COMPARISONS_CANCELED]);
function noticeClass(message){return 'notice'+(message.startsWith('Error')?' error':NEUTRAL_MESSAGES.has(message)||state.busy&&message===state.busyMessage?'':' good');}
let lastSavedRaw=null,saveQueue=Promise.resolve(),pendingSaves=0,unsavedRecoveryDraft=false,recoveryDraftRevision=0;
let saved,savedScenarios,savedStoredRaw=null,savedLoadError='';
try{
  const stored=localStorage.getItem(storageKey);savedStoredRaw=stored;lastSavedRaw=stored;
  if(stored!==null){
    saved=JSON.parse(stored);
    if(!Array.isArray(saved?.scenarios)||!saved.scenarios.length)throw new Error('No saved scenarios.');
    const normalized=normalizeScenarios(saved.scenarios);
    for(const s of normalized){if(validateScenarioStructure(s).length)throw new Error('Invalid saved scenario structure.');prepareCalendarScenario(s);}
    savedScenarios=normalized;
  }
}catch{savedLoadError='Error: Saved plans could not be loaded. Your stored backup has not been changed. Export it before replacing it. Edits stay in this page until you import a valid backup or choose to replace the unreadable plans.';}
const hasSavedScenarios=Boolean(savedScenarios);
const savedSelectedId=typeof saved?.selectedId==='string'||typeof saved?.selectedId==='number'?String(saved.selectedId).trim():'';
const state={scenarios:savedScenarios||sampleScenarios().map(s=>{prepareCalendarScenario(s,{needsReview:false});s.household.separatePeople=true;s.household.spouseRetirementDate=s.household.retirementDate;return s;}),selectedId:savedSelectedId||'base-plan',view:'dashboard',results:new Map(),labResults:null,decision:null,busy:false,message:savedLoadError,access:{tier:'free',maxPaths:FREE_SIMULATION_PATHS,signedIn:false,checkoutAvailable:false},auth:{configured:false,chatgptSignedIn:false,signedIn:false,linkedProviders:[],enabledProviders:{google:false}},advancedOpen:false,allocationOpen:false,setupSection:0};
if(!state.scenarios.some(s=>s.id===state.selectedId))state.selectedId=state.scenarios[0].id;
state.inputSources=normalizeInputSources(saved?.inputSources,state.scenarios,hasSavedScenarios?'Saved value; source not recorded':'Sample/default');
for(const s of state.scenarios)if(s.household.datesNeedReview)for(const path of ['household.birthday','household.retirementDate','household.spouseBirthday'])if(!state.inputSources[s.id][path])state.inputSources[s.id][path]='Estimated';
// Newly introduced zero defaults have no historical user source. Existing
// values and explicitly saved notes remain unchanged.
if(hasSavedScenarios)for(const s of state.scenarios){
  const raw=saved.scenarios.find(x=>String(x.id)===s.id);
  for(const section of ['contributions','spouseContributions','spouseAccounts','spouseRothHistory','spouseIncome','workingIncome','spouseWithdrawal'])if(raw?.[section]===undefined)for(const [key,value] of Object.entries(s[section]))if(typeof value==='number'||typeof value==='boolean')state.inputSources[s.id][section+'.'+key] ||= 'Sample/default';
}
state.guided=false;
state.hasStartedPlan=hasSavedScenarios&&(saved.hasStartedPlan??true);
function current(){return state.scenarios.find(s=>s.id===state.selectedId)||state.scenarios[0];}
const ACCESS_GRACE_MS=5*60*1000;
let accessConfirmedAt=0,accessRequest=0,accessGraceTimer=null,billingReference={};
try{const savedReference=JSON.parse(sessionStorage.getItem('retirement-billing-reference'));if(typeof savedReference?.customerId==='string'&&/^cus_[A-Za-z0-9]{1,200}$/.test(savedReference.customerId))billingReference={customerId:savedReference.customerId};}catch{}
// Keep a few lookup hints for explicit linking across sign-in changes. The
// server verifies ownership against both current identities before using them.
let linkingCustomerIds=[];
try{const ids=JSON.parse(sessionStorage.getItem('retirement-link-customers'));if(Array.isArray(ids))linkingCustomerIds=ids.filter(id=>typeof id==='string'&&/^cus_[A-Za-z0-9]{1,200}$/.test(id)).slice(-4);}catch{}
function rememberLinkCustomer(id){if(!id)return;linkingCustomerIds=[...linkingCustomerIds.filter(x=>x!==id),id].slice(-4);try{sessionStorage.setItem('retirement-link-customers',JSON.stringify(linkingCustomerIds));}catch{}}
rememberLinkCustomer(billingReference.customerId);
function saveBillingReference(){try{sessionStorage.setItem('retirement-billing-reference',JSON.stringify(billingReference));}catch{}}
function isPro(){return state.access.tier==='pro'&&(!state.accessStale||Date.now()-accessConfirmedAt<ACCESS_GRACE_MS);}
async function billingHeaders(){return {...await authHeaders(),...(billingReference.customerId?{'x-retirement-customer':billingReference.customerId}:{})};}
function clearAccessGraceTimer(){clearTimeout(accessGraceTimer);accessGraceTimer=null;}
function resetBillingIdentity(){accessRequest++;clearAccessGraceTimer();accessConfirmedAt=0;billingReference={};saveBillingReference();state.access={tier:'free',maxPaths:FREE_SIMULATION_PATHS,signedIn:false,checkoutAvailable:false};state.accessStale=false;}
function scheduleAccessGraceExpiry(){
  clearAccessGraceTimer();
  if(!state.accessStale||state.access.tier!=='pro')return;
  accessGraceTimer=setTimeout(()=>{
    accessGraceTimer=null;
    if(!state.accessStale||state.access.tier!=='pro')return;
    if(isPro()){scheduleAccessGraceExpiry();return;}
    state.access={...state.access,tier:'free',maxPaths:FREE_SIMULATION_PATHS,ownerAccess:false,checkoutAvailable:false};
    if(state.message===ACCESS_RETRY)state.message=ACCESS_UNAVAILABLE;
    render({preserveEditor:true});
  },Math.max(0,accessConfirmedAt+ACCESS_GRACE_MS-Date.now()));
}
function syncAuthState(){const next=socialState();if(state.auth.accountKey!==undefined&&state.auth.accountKey!==next.accountKey)resetBillingIdentity();state.auth=next;}
function effectivePaths(s=current()){return isPro()?Math.max(MIN_SIMULATION_PATHS,Math.min(MAX_SIMULATION_PATHS,Number(s.numberOfSimulations)||FREE_SIMULATION_PATHS)):FREE_SIMULATION_PATHS;}
function simulationScenario(s=current()){const copy=deep(s);copy.numberOfSimulations=effectivePaths(s);copy.seed=DEFAULT_SEED;copy.household.asOfDate=localCalendarDate();return copy;}
function persist(markStarted=true,replaceUnreadable=false){
  // Protect the original unreadable backup, but warn before losing edits to
  // the recovery draft. Automatic defaults do not count as user edits.
  if(savedLoadError&&!replaceUnreadable){if(markStarted){unsavedRecoveryDraft=true;recoveryDraftRevision++;}return Promise.resolve(false);}
  if(markStarted)state.hasStartedPlan=true;
  // Capture each edit before waiting for the lock. The queue preserves this
  // tab's save order; the origin-wide lock makes the revision check and write
  // indivisible with respect to other tabs running this app.
  const payload=JSON.stringify({scenarios:state.scenarios,selectedId:state.selectedId,hasStartedPlan:state.hasStartedPlan,inputSources:state.inputSources});
  // Edits made while an explicit recovery save waits for its lock must remain
  // marked unsaved if they were not included in this snapshot.
  const draftRevision=recoveryDraftRevision;
  const save=async()=>{
    try{
      if(!globalThis.navigator?.locks){
        state.storageError='Error: Automatic saving is unavailable in this browser. Export a backup to keep your scenarios before closing this page.';
        state.message=state.storageError;return false;
      }
      return await navigator.locks.request(storageKey,()=>{
        if(localStorage.getItem(storageKey)!==lastSavedRaw){state.storageError=STORAGE_CONFLICT;state.message=STORAGE_CONFLICT;return false;}
        localStorage.setItem(storageKey,payload);lastSavedRaw=payload;
        if(state.message===state.storageError||state.message===savedLoadError)state.message='';
        state.storageError='';savedLoadError='';savedStoredRaw=null;unsavedRecoveryDraft=recoveryDraftRevision!==draftRevision;return true;
      });
    }catch{
      state.storageError='Error: Changes could not be saved in browser storage. Export a backup to keep your scenarios before closing this page.';
      state.message=state.storageError;return false;
    }
  };
  pendingSaves++;
  saveQueue=saveQueue.then(save,save).finally(()=>{pendingSaves--;});return saveQueue;
}
let calculationRevision=0;
function invalidateExploration(){calculationRevision++;state.labResults=null;state.decision=null;state.message='';}
function selectScenario(id){invalidateExploration();state.selectedId=id;}
function result(){return state.results.get(current().id);}
function setMessage(message,options){state.message=state.storageError||message;render(options);}
function currentViewLabel(){return {dashboard:'Overview',setup:'Assumptions',budget:'Budget',withdrawals:'How withdrawals work',scenarios:'Scenarios',lab:'Scenario lab',results:'Results',reports:'Reports & backup',billing:'Plans & billing'}[state.view];}
function pageHead(title,detail='',actions=''){return `<div class="page-head"><div><p class="kicker">${escapeHTML(currentViewLabel())}</p><h1>${escapeHTML(title)}</h1><p>${escapeHTML(detail)}</p></div>${actions?`<div class="actions">${actions}</div>`:''}</div>`;}
function card(title,content,extra=''){return `<section class="card ${extra}"><h2>${escapeHTML(title)}</h2>${content}</section>`;}
function info(label,value){return `<div class="info-row"><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong></div>`;}
function sensitivityNote(r){return r.riskBreakdown.summary?`<p class="form-note">${escapeHTML(r.riskBreakdown.summary)}</p>`:'';}
function sensitivityBreakdown(r){const checks=r.riskBreakdown.checks??['market','spending','taxes','healthcare','longevity'].map(key=>({key,label:key[0].toUpperCase()+key.slice(1)}));return `<div class="info-list">${checks.map(c=>`<div class="info-row"><span>${escapeHTML(c.label)}${c.description?`<small class="muted"> · ${escapeHTML(c.description)}</small>`:''}</span>${tag(r.riskBreakdown[c.key])}</div>`).join('')}</div>${sensitivityNote(r)}`;}
function tag(value){const c=value==='AtRisk'?'risk':value==='Watch'?'watch':'';return `<span class="tag ${c}">${escapeHTML(value)}</span>`;}
function recoveryActions(){return savedLoadError?`<div class="actions">${savedStoredRaw===null?'':'<button class="secondary" data-action="export-unreadable-backup">Export unreadable backup</button>'}<button class="secondary" data-action="replace-unreadable-plans">Replace unreadable saved plans</button></div>`:'';}
function messageNotice(){const message=state.storageError||savedLoadError||state.message;return message?`<div class="${noticeClass(message)}" role="status">${escapeHTML(message)}${recoveryActions()}</div>`:'';}
// A running calculation shows its progress and a way to stop it on every view.
function busyPanel(){if(!state.busy)return '';const p=state.progress||{};return `<div class="busy-panel"><progress id="busy-progress" max="1" aria-label="Calculation progress"${p.fraction==null?'':` value="${p.fraction}"`}></progress><span id="busy-detail" aria-live="polite">${escapeHTML(p.detail||'')}</span><button class="secondary" data-action="cancel-calculation">Cancel calculation</button></div>`;}
function notice(){return messageNotice()+busyPanel();}
function refreshNotice(){const existing=$('#main > .notice'),message=state.storageError||savedLoadError||state.message;if(existing){if(!message){existing.remove();return;}existing.className=noticeClass(message);existing.textContent=message;existing.insertAdjacentHTML('beforeend',recoveryActions());}else if(message){const head=$('#main > .page-head');if(head)head.insertAdjacentHTML('afterend',messageNotice());else $('#main').insertAdjacentHTML('afterbegin',messageNotice());}}
function planningDisclosure(){return '<aside class="planning-disclosure" aria-label="Important model limitations"><strong>For U.S. retirement planning only</strong><p>These hypothetical results use U.S. federal tax, Social Security, and Medicare assumptions plus your inputs. State and local income taxes and laws outside the U.S. are not modeled. “Readiness” is the share of simulated lifetimes without a portfolio shortfall, not the probability of your actual outcome. Results are not predictions or guarantees and do not capture every cost or event. This is educational, not individualized financial, investment, tax, legal, or insurance advice. Verify your inputs and consult qualified professionals before acting.</p><a href="./methodology.html">Read model scope &amp; limitations →</a></aside>';}
const icons={lock:'<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',check:'<circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.8 2.8L16.5 9.5"/>',model:'<path d="M3 12h4l2.5-6 5 12 2.5-6h4"/>',play:'<path class="icon-fill" d="M8 5.5v13l10.5-6.5z"/>',market:'<path d="m3 17 6-6 4 4 8-8"/><path d="M15 7h6v6"/>',costs:'<path d="M12 3v18"/><path d="M16.5 7.5c0-1.9-2-3-4.5-3s-4.5 1.1-4.5 3 2 2.7 4.5 3.3 4.5 1.5 4.5 3.7-2 3-4.5 3-4.5-1.1-4.5-3"/>',tax:'<path d="M7 3h7l4 4v14H7z"/><path d="M14 3v4h4M10 12h5M10 16h5"/>',life:'<path d="M12 20s-7.5-4.6-7.5-10.3A4.2 4.2 0 0 1 12 7.2a4.2 4.2 0 0 1 7.5 2.5C19.5 15.4 12 20 12 20z"/>'};
function icon(name){return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${icons[name]}</svg>`;}
function monteCarloBanner(){const features=[['market','Market swings','Stock and bond returns change month by month.'],['costs','Rising costs','General and healthcare inflation follow their own uncertain paths.'],['tax','Taxes & benefits','Federal income tax, Social Security, and Medicare premiums are applied.'],['life','Lifespan','Each lifetime draws its length from SSA mortality tables.']];return `<section class="monte-banner" aria-labelledby="monte-title"><div class="monte-intro"><span class="monte-eyebrow">Monte Carlo retirement simulation</span><h2 id="monte-title">One plan. Many possible futures.</h2><p>Each path samples market returns, general and healthcare inflation, and lifespan. The model then follows monthly spending, income, taxes, and withdrawals through that lifetime.</p><a href="./methodology.html#monte-carlo">How Monte Carlo works →</a></div><ul class="monte-features">${features.map(([name,title,text])=>`<li><span class="feature-icon">${icon(name)}</span><strong>${title}</strong><span>${text}</span></li>`).join('')}</ul></section>`;}
function planNotice(){return isPro()?'<div class="plan-notice pro"><strong>Pro</strong> Up to 10,000 Monte Carlo paths, calculated on this device. <button class="text-link" data-view="billing">Manage plan</button></div>':'<div class="plan-notice"><strong>Free preview · 10 paths</strong> <button class="text-link" data-view="billing">Explore Pro</button></div>';}
function overviewUpgradeCard(){return `<section class="upgrade-card" aria-labelledby="overview-upgrade-title"><div class="upgrade-copy"><span class="upgrade-eyebrow">Free preview · 10 paths</span><h2 id="overview-upgrade-title">Get a steadier Monte Carlo estimate with Pro</h2><p>Pro lets you run up to 10,000 modeled lifetimes using the same plan, right on your device.</p></div><div class="upgrade-offer"><strong>Pro · up to 10,000 paths</strong><span>$9.99/month or $79/year</span><button class="primary" data-view="billing">Explore Pro plans</button><small>More paths reduce sampling noise; results remain hypothetical.</small></div></section>`;}
function readinessUpgrade(r){if(isPro())return '';const count=r.provenance.simulationCount,successes=Math.round(r.successProbability*count);return `<div class="readiness-upgrade"><strong>${successes} of ${count} paths ended without a shortfall</strong><p>Pro lets you run up to 10,000 paths for a steadier estimate.</p><button class="primary" data-view="billing">Explore Pro plans</button><small>$9.99/month or $79/year · More paths do not make this a prediction.</small></div>`;}
function monteRunSummary(count,completed){return `<div class="monte-run-summary"><span class="monte-run-label">Monte Carlo model</span><span><strong>${Number(count).toLocaleString('en-US')}</strong> ${completed?'simulated lifetimes in this run':'paths configured for the next run'}</span><a href="./methodology.html#monte-carlo">How the simulation works →</a></div>`;}
function scopeDisclosure(){return '<p class="scope-disclosure">For U.S. retirement planning only. The model uses U.S. federal rules; state and local income taxes and laws outside the U.S. are not modeled. <a href="./methodology.html">Model scope &amp; limitations</a></p>';}
function previewNotice(r){return isPreviewResult(r)?`<aside class="preview-warning"><strong>Sample preview only</strong><p>${PREVIEW_WARNING}</p></aside>`:'';}
function resultHero(r){const preview=isPreviewResult(r),count=r.provenance.simulationCount;return `<section class="result-hero" aria-label="Simulation outcome summary"><div class="result-hero-main"><span class="result-eyebrow">${preview?'Sample outcomes':'Monte Carlo readiness'}</span><div class="result-figure">${readinessLabel(r)}</div><p>${preview?'Sample lifetimes without a shortfall.':'Share of modeled lifetimes without a portfolio shortfall.'}</p>${preview?'':`<div class="bar" aria-hidden="true"><span style="width:${Math.round(100*r.successProbability)}%"></span></div>`}${previewNotice(r)}</div><dl class="result-stats"><div><dt>Median ending balance · future dollars</dt><dd>${money(r.medianEndingBalance)}</dd><small>At death, modeling limit or shortfall across all paths; failures count as $0</small></div><div><dt>Median failure age</dt><dd>${r.medianFailureAge===null?'—':ageLabel(r.medianFailureAge)}</dd><small>${r.medianFailureAge===null?'No shortfalls observed in this run':'Among paths that ran out of funds'}</small></div><div><dt>Simulated lifetimes</dt><dd>${count.toLocaleString('en-US')}</dd><small>Fixed comparison sequence · <a href="./methodology.html#monte-carlo">How the simulation works</a></small></div></dl></section>`;}
// Welcome illustration: fixed, seeded sample paths drawn for decoration only.
// Nothing here comes from the visitor's plan or the simulation engine.
function illustrationPaths(){
  let seed=483059;const rand=()=>(seed=seed*48271%2147483647)/2147483647,normal=()=>Math.sqrt(-2*Math.log(rand()))*Math.cos(2*Math.PI*rand());
  const paths=[];for(let p=0;p<30;p++){let v=1;const values=[v];for(let i=1;i<=24&&v>0;i++){v=v*Math.exp(.06+normal()*.13)-.05*Math.pow(1.02,i);values.push(Math.max(0,v));}paths.push(values);}
  return paths;
}
const illustration=illustrationPaths();
// One loop: about 2.9 s drawing in, 5 s holding the finished picture, then a 0.5 s fade.
// The CSS fade is the final 6% of the cycle; when it ends, the illustration redraws.
const ILLUSTRATION_CYCLE_MS=8400;
let illustrationStarted=0;
function forecastIllustration(){
  // Re-renders resume the current loop where it left off instead of restarting it.
  const now=Date.now();if(!illustrationStarted||now-illustrationStarted>=ILLUSTRATION_CYCLE_MS)illustrationStarted=now;
  const left=14,right=406,top=14,floor=222,x=i=>left+(right-left)*i/24,y=v=>floor-(floor-top)*(1-Math.exp(-v/1.8)),n=v=>Math.round(v*10)/10;
  const line=values=>values.map((v,i)=>`${i?'L':'M'}${n(x(i))} ${n(y(v))}`).join('');
  const quantile=(i,q)=>{const alive=illustration.map(p=>p[i]??0).sort((a,b)=>a-b);return alive[Math.round((alive.length-1)*q)];};
  const steps=Array.from({length:25},(_,i)=>i),band=`${steps.map(i=>`${i?'L':'M'}${n(x(i))} ${n(y(quantile(i,.9)))}`).join('')}${[...steps].reverse().map(i=>`L${n(x(i))} ${n(y(quantile(i,.1)))}`).join('')}Z`;
  const paths=illustration.map((values,i)=>`<path class="fan-path ${values.at(-1)>0?'funded':'short'}" pathLength="1" style="--i:${i}" d="${line(values)}"/>`).join('');
  const ends=illustration.filter(values=>values.at(-1)===0).map(values=>`<circle class="fan-end" cx="${n(x(values.length-1))}" cy="${floor}" r="4"/>`).join('');
  return `<figure class="forecast-illustration" style="--elapsed:${now-illustrationStarted}ms;--fan-cycle:${ILLUSTRATION_CYCLE_MS}ms"><div class="illustration-heading"><strong>Same savings, different lifetimes</strong><span>Illustration</span></div><svg viewBox="0 0 420 248" role="img" aria-labelledby="forecast-example-title forecast-example-desc"><title id="forecast-example-title">Illustrative retirement savings paths</title><desc id="forecast-example-desc">Thirty hypothetical savings paths start from the same balance and spread apart over time. Most stay funded, and a few reach zero. A white line marks the middle path. This drawing is not a simulation or a forecast of your plan.</desc><g class="fan-grid"><path d="M14 14H406M14 83H406M14 152H406"/><path class="fan-floor" d="M14 222H406"/></g><g class="fan-data"><path class="fan-band" d="${band}"/><g class="fan-paths">${paths}</g><path class="fan-median" pathLength="1" d="${line(steps.map(i=>quantile(i,.5)))}"/>${ends}</g><circle class="fan-origin" cx="${left}" cy="${n(y(1))}" r="5"/><text x="14" y="242">Retirement</text><text x="406" y="242" text-anchor="end">Later life →</text></svg><ul class="fan-legend"><li><i class="key funded"></i>Stays funded</li><li><i class="key short"></i>Runs short</li><li><i class="key median"></i>Middle path</li><li><i class="key band"></i>Middle 80% of paths</li></ul><figcaption>Illustrative paths only. Your results appear after you run a simulation.</figcaption></figure>`;
}
function welcomePanel(){const returning=state.hasStartedPlan,points=[['model','Markets, inflation, taxes, and lifespan in one model.'],['lock','Your financial inputs stay in your browser.'],...(isPro()?[]:[['check','No sign-up needed for the free preview.']])];return `<section class="forecast-welcome" aria-labelledby="welcome-title"><div class="welcome-copy"><p class="kicker">${returning?'Welcome back':'Your retirement, explored'}</p><h1 id="welcome-title">${returning?'Pick up where you left off.':'Explore how long your retirement savings could last.'}</h1><p class="welcome-description">${returning?`Continue with <strong>${escapeHTML(current().name)}</strong>. Review your assumptions or run your plan to explore the range of possible outcomes.`:'Bring your savings, spending, and retirement plans together. Explore how different markets, costs, and lifespans could shape your future.'}</p><div class="welcome-actions"><button class="primary" data-action="start-plan">${returning?'Continue my plan':'Build my forecast'} <span aria-hidden="true">→</span></button><button class="secondary" data-action="run-plan" ${state.busy?'disabled':''}>${icon('play')}${returning?'Run my plan':'Explore a sample plan'}</button></div><ul class="welcome-points">${points.map(([name,text])=>`<li>${icon(name)}<span>${text}</span></li>`).join('')}</ul><p class="welcome-limit">${isPro()?'Pro · Up to 10,000 simulated lifetimes on your device.':'Free preview · 10 simulated lifetimes. A first look, not a reliable readiness estimate.'}</p></div>${forecastIllustration()}</section>`;}
function dashboard(){
  const s=current(),r=result(),notes=scenarioWarnings(simulationScenario(s)),assets=totalSavings(s);
  const steps=`<div class="workflow"><button data-view="setup"><span class="step-number">1</span><span><strong>Review your assumptions</strong><small>Household, savings, income & costs</small></span><span aria-hidden="true">↗</span></button><button data-view="budget"><span class="step-number">2</span><span><strong>Check your spending</strong><small>Build a budget from monthly costs</small></span><span aria-hidden="true">↗</span></button><button data-view="${r?'lab':'results'}" ${r?'':'data-action="run-plan"'}><span class="step-number">3</span><span><strong>${r?'Explore a different outcome':'Run your plan'}</strong><small>${r?'Compare changes in the scenario lab':'See readiness and the range of outcomes'}</small></span><span aria-hidden="true">↗</span></button></div>`;
  return `${r?pageHead('Your retirement forecast','Explore your results, then test what could change them.'):''}${notice()}<div class="stack">${r?'':welcomePanel()}${steps}<div class="plan-summary-heading"><h2>${state.hasStartedPlan?'Current plan':'Sample plan'} at a glance</h2><span>${escapeHTML(s.name)}</span></div><section class="plan-strip" aria-label="Current plan summary"><div><span>Retirement date</span><strong>${dateLabel(scenarioTimeline(s).startDate||s.household.retirementDate)}</strong><small>Household starts · Age ${ageLabel(retirementAge(s))} · Current age ${Math.floor(scenarioTimeline(s).currentAge)}</small></div><div><span>Starting savings</span><strong>${money(assets)}</strong><small>Across all four account types</small></div><div><span>Annual base spending</span><strong>${money(s.spending.annualBaseSpending)}</strong><small>Before separate housing & health costs</small></div><div><span>Social Security claim age</span><strong>${s.socialSecurity.claimAge}</strong><small>${money(s.socialSecurity.annualBenefitAt67)} / year estimate at 67</small></div></section>${r?`<div class="split"><section class="card feature-card"><div><div class="label">${isPreviewResult(r)?'Sample outcomes':'Monte Carlo readiness estimate'}</div><div class="huge">${readinessLabel(r)}</div><p>${isPreviewResult(r)?'Sample lifetimes without a shortfall.':'Share of modeled lifetimes without a portfolio shortfall.'}</p>${previewNotice(r)}<button class="secondary" data-view="results">View full results →</button></div>${isPreviewResult(r)?'':`<div class="ring" style="--value:${Math.round(r.successProbability*100)}%"><span>Simulated lifetimes</span></div>`}</section>${card('Next useful test',`<p>${escapeHTML(r.riskBreakdown.recommendedNextTest)}</p><button class="secondary" data-view="lab">Explore in scenario lab →</button>`)}</div><div class="split">${chartCard('survival')}${card('Assumption sensitivity',sensitivityBreakdown(r))}</div>`:''}${r?chartCard('paths'):''}${monteCarloBanner()}${isPro()?planNotice():overviewUpgradeCard()}${notes.length?`<details class="card planning-notes"><summary>Planning notes <span class="tag">${notes.length}</span></summary><ul class="warning-list">${notes.map(n=>`<li>${escapeHTML(n)}</li>`).join('')}</ul></details>`:''}${planningDisclosure()}</div>`;
}
const monthFields={'guaranteedIncome.startAge':'guaranteedIncome.startAgeMonths','longTermCare.averageDurationYears':'longTermCare.averageDurationMonths','spouseIncome.pensionStartAge':'spouseIncome.pensionStartAgeMonths'};
const schema=[
  ['Household & retirement',[['Your birthday','household.birthday','date'],['Retirement date','household.retirementDate','date'],['Maximum modeling age','household.targetEndAge','number'],['Filing status','household.filingStatus','select','Single|Married|HeadOfHousehold'],['Your longevity table','household.gender','select','Male|Female'],['Spouse birthday','household.spouseBirthday','date'],['Spouse longevity table','household.spouseGender','select','Male|Female']]],
  ['Accounts & spending',[['Pre-tax / non-Roth retirement savings','accounts.pretax','money'],['Total Roth IRA value','accounts.roth','money'],['Taxable savings','accounts.taxable','money'],['Cash reserve','accounts.cash','money'],['Annual base spending','spending.annualBaseSpending','money'],['Spending path','spending.spendingPathModel','select','EmpiricalAgeDecline|Flat'],['General inflation average %','spending.generalInflationMean','percent'],['General inflation volatility %','spending.generalInflationStdDev','percent'],['Cut spending if portfolio halves %','spending.lowPortfolioSpendingReduction','percent']]],
  ['Income & Social Security',[['Your Social Security / year at age 67','socialSecurity.annualBenefitAt67','money'],['Claim age','socialSecurity.claimAge','number'],['Spouse claim age','socialSecurity.spouseClaimAge','number'],['Your pension or annuity / year','guaranteedIncome.annualIncome','money'],['Your pension start age','guaranteedIncome.startAge','number'],['Pension annual increase %','guaranteedIncome.annualIncrease','percent'],['Survivor benefit %','guaranteedIncome.survivorPercent','percent']]],
  ['Housing & healthcare',[['Mortgage payment / month','mortgage.monthlyPayment','money'],['Mortgage balance','mortgage.currentBalance','money'],['Mortgage years left','mortgage.yearsLeft','number'],['Mortgage months left','mortgage.monthsLeft','number'],['Home value','home.currentValue','money'],['Property tax & home insurance / year','home.annualTaxesAndInsurance','money'],['Rent / month','rent.monthlyRent','money'],['Pre-Medicare premium / adult / month','healthcare.preMedicareMonthlyPremium','money'],['Healthcare inflation average %','healthcare.healthcareInflationMean','percent'],['Healthcare inflation volatility %','healthcare.healthcareInflationStdDev','percent'],['Include Medicare premiums','healthcare.includeMedicarePremiums','checkbox'],['Long-term care risk','longTermCare.enabled','checkbox'],['Long-term care cost / year','longTermCare.annualCost','money'],['Long-term care duration / years','longTermCare.averageDurationYears','number']]],
  ['Market & withdrawal strategy',[['Pre-retirement return average %','market.preRetirementMeanReturn','percent'],['Pre-retirement return volatility %','market.preRetirementStdDev','percent'],['Stock return average %','market.stockMeanReturn','percent'],['Stock return volatility %','market.stockStdDev','percent'],['Bond return average %','market.bondMeanReturn','percent'],['Bond return volatility %','market.bondStdDev','percent'],['Roth conversions','rothConversion.enabled','checkbox'],['Conversion tax bracket cap %','rothConversion.marginalRateCap','percent'],['Use cash in market drawdowns','withdrawalStrategy.useCashReserveDuringDrawdowns','checkbox'],['Drawdown trigger %','withdrawalStrategy.drawdownTrigger','percent'],['Model early withdrawal penalties','withdrawalStrategy.applyEarlyWithdrawalPenalty','checkbox'],['My employer plan qualifies for Rule of 55','withdrawalStrategy.ruleOf55Eligible','checkbox'],['Use a 72(t)/SEPP withdrawal plan','withdrawalStrategy.seppEligible','checkbox'],['Simulation paths','numberOfSimulations','number']]],
  ['Stock allocation by portfolio size',[['Under 30× spending %','postRetirementAllocation.stockUnder30x','percent'],['30–35× spending %','postRetirementAllocation.stock30xTo35x','percent'],['35–40× spending %','postRetirementAllocation.stock35xTo40x','percent'],['40–45× spending %','postRetirementAllocation.stock40xTo45x','percent'],['45–50× spending %','postRetirementAllocation.stock45xTo50x','percent'],['50× or more %','postRetirementAllocation.stock50xOrMore','percent']]]
];
schema.push(['Roth IRA history',[['Remaining regular contributions','rothHistory.contributionBasis','money'],['First Roth funding tax year','rothHistory.firstContributionYear','number']]]);
const contributionFields=prefix=>[['Employee pre-tax savings / year',prefix+'.pretax','money'],['Employer pre-tax contribution / year',prefix+'.employerPretax','money'],['Roth IRA contributions / year',prefix+'.roth','money'],['Taxable investments added / year',prefix+'.taxable','money'],['Cash saved / year',prefix+'.cash','money'],['Contribution annual increase %',prefix+'.annualIncrease','percent']].map(([label,...rest])=>[(prefix==='contributions'?'Your ':'Spouse ')+(label.startsWith('Roth')?label:label.charAt(0).toLowerCase()+label.slice(1)),...rest]);
schema[0][1].push(['Spouse retirement date','household.spouseRetirementDate','date']);
schema.push(['Your future savings',contributionFields('contributions')],['Spouse future savings',contributionFields('spouseContributions')],
  ['Spouse retirement accounts',[['Spouse pre-tax / non-Roth balance','spouseAccounts.pretax','money'],['Spouse total Roth IRA value','spouseAccounts.roth','money'],['Spouse remaining Roth contributions','spouseRothHistory.contributionBasis','money'],['Spouse first Roth funding year','spouseRothHistory.firstContributionYear','number']]],
  ['Spouse own benefits & pension',[['Spouse own Social Security / year at 67','spouseIncome.annualBenefitAt67','money'],['Spouse pension / year','spouseIncome.annualPension','money'],['Spouse pension start age','spouseIncome.pensionStartAge','number'],['Spouse pension annual increase %','spouseIncome.annualIncrease','percent'],['Your share of spouse pension after death %','spouseIncome.survivorPercent','percent']]],
  ['Working household support',[['Your take-home household support / year','workingIncome.primaryAnnualNet','money'],['Spouse take-home household support / year','workingIncome.spouseAnnualNet','money'],['Support annual increase %','workingIncome.annualIncrease','percent']]],
  ['Spouse early-access choices',[['Spouse employer plan qualifies for Rule of 55','spouseWithdrawal.ruleOf55Eligible','checkbox'],['Spouse uses a 72(t)/SEPP plan','spouseWithdrawal.seppEligible','checkbox']]]);
function personModelNotice(s){return s.household.separatePeople?'<p class="form-note">Separate-person model: expenses start at the first retirement date. Each person’s savings deposits stop at their own retirement or death. Taxable investments and cash are shared; pre-tax and Roth balances have separate owners. Split any combined statement between owners; do not count the same savings twice.</p>':`<aside class="setup-notice"><strong>Saved plan · shared-date model</strong><p>This plan keeps its original pooled accounts, one pension and spouse benefits based on your record. Future savings contributions are available. <button class="text-link" data-action="enable-people">Use separate-person inputs</button></p><p>When switching, the existing combined pre-tax and Roth balances initially stay under You. Split them between owners before running; do not add spouse balances on top of an unchanged combined total.</p></aside>`;}
function spouseConversions(s){return `<details class="inset-details" ${s.spouseRothHistory.conversions.length?'open':''}><summary>Spouse past Roth conversions</summary><p class="form-note">Remaining conversion principal, not new savings. The taxable portion must be part of the total. Spouse history remains separate from yours.</p>${s.spouseRothHistory.conversions.map((lot,i)=>`<div class="fields">${fieldHTML(s,['Tax year',`spouseRothHistory.conversions.${i}.taxYear`,'number'])}${fieldHTML(s,['Remaining principal',`spouseRothHistory.conversions.${i}.amount`,'money'])}${fieldHTML(s,['Taxable principal',`spouseRothHistory.conversions.${i}.taxableAmount`,'money'])}<button class="secondary" data-action="remove-spouse-conversion" data-index="${i}">Remove conversion</button></div>`).join('')}<button class="secondary" data-action="add-spouse-conversion">Add spouse conversion</button></details>`;}
function totalSavings(s){return Object.values(s.accounts).reduce((a,b)=>a+b,0)+(s.household.separatePeople&&s.household.filingStatus==='Married'?s.spouseAccounts.pretax+s.spouseAccounts.roth:0);}

const assumptionHelp={
  'household.birthday':'Your date of birth. Your current age and age at retirement are calculated from this date. The simulation uses completed calendar months.',
  'household.retirementDate':'The day retirement begins, today or later. Pre-retirement growth uses completed calendar months from today; retirement cash flows begin at your age on this date. The model does not calculate daily cash flows.',
  'household.spouseBirthday':'Your spouse’s date of birth. The model calculates their age at your retirement date for benefits, healthcare, and longevity.',
  'accounts.pretax':'Traditional 401(k), IRA, and similar tax-deferred balances. Withdrawals and required minimum distributions are ordinary income. For RMDs, this pooled balance belongs to the primary person, then to the surviving spouse after the death year. See Model scope for timing and beneficiary assumptions.',
  'accounts.roth':'Total Roth IRA value today, including remaining contributions, conversion principal, and earnings. Enter remaining regular contributions separately. This model uses Roth IRA ordering; employer Roth 401(k) distributions can follow different rules.',
  'rothHistory.contributionBasis':'Regular Roth IRA contributions that have not already been withdrawn. Exclude conversions and investment earnings. Withdrawals use these contributions first without income tax or an early-withdrawal penalty. Investment losses do not reduce this dollar basis, so it can exceed the current balance.',
  'rothHistory.firstContributionYear':'The tax year of your first Roth IRA funding, including a conversion if it was first. Earnings generally become qualified after five tax years and age 59½. Enter 0 only if you have never funded a Roth IRA; a future modeled conversion then starts the clock.',
  'accounts.taxable':'Brokerage investments outside retirement accounts. The model includes them in invested savings and can draw on them for spending. It does not calculate taxes on brokerage gains, dividends, or withdrawals.',
  'accounts.cash':'Cash reserve kept apart from invested savings, earning a fixed 2% yearly yield. Credited interest is ordinary income during retirement and affects taxes, taxable Social Security, and Medicare surcharges. Cash withdrawals themselves are not income.',
  'spending.annualBaseSpending':'Your yearly living costs in today\'s dollars, including property tax and home insurance. Leave out mortgage or rent payments and healthcare premiums; you enter those separately under Housing & healthcare.',
  'spending.generalInflationMean':'The average yearly increase assumed for general living and housing costs.',
  'socialSecurity.claimAge':'Age when the model starts your Social Security benefit. It adjusts the age-67 estimate above for this claiming age.',
  'guaranteedIncome.startAge':'Age when the pension or annuity income begins in the simulation.',
  'mortgage.monthlyPayment':'Monthly principal-and-interest mortgage payment added separately to living costs while the loan remains. Leave out escrow for property tax and insurance; include those bills in annual base spending and in Property tax & home insurance instead. Enter 0 if there is no mortgage.',
  'mortgage.yearsLeft':'Full years remaining on the mortgage. Add any extra months in the next field.',
  'mortgage.monthsLeft':'Extra months remaining beyond the full years above, from 0 through 11.',
  'mortgage.currentBalance':'Estimated unpaid mortgage principal. The model infers a fixed interest rate from your balance, payment, and remaining term, then deducts the remaining loan at a home sale. The payment must cover the balance over that term; use principal and interest only, without escrow for taxes or insurance.',
  'rent.monthlyRent':'Monthly rent added separately to living costs during retirement. Leave at 0 if you are not renting.',
  'healthcare.healthcareInflationMean':'Average yearly growth assumed for healthcare premiums and long-term care costs.',
  'longTermCare.annualCost':'Annual cost per person during a modeled long-term care episode, before future healthcare inflation.',
  'longTermCare.averageDurationYears':'Whole years plus extra months a modeled care episode lasts before the person’s death when long-term care occurs.',
  'market.stockMeanReturn':'Average yearly return assumed for the stock share of invested savings after retirement.',
  'market.bondMeanReturn':'Average yearly return assumed for the bond share of invested savings after retirement.',
  'market.bondStdDev':'Standard deviation of compounded yearly bond returns after retirement. Monthly returns match this yearly volatility and the entered yearly average.',
  'household.gender':'Select the mortality rates used to draw your lifespan in each simulation. This does not change your filing status.',
  'household.spouseGender':'Select the mortality rates used to draw your spouse’s lifespan in married simulations.',
  'spending.spendingPathModel':'Age-based decline gradually lowers base spending from ages 65 to 85. Flat keeps base spending level before inflation.',
  'spending.generalInflationStdDev':'Standard deviation of the compounded yearly change in general prices. Monthly changes are calibrated to this yearly volatility and the entered yearly average; deflation is allowed.',
  'spending.lowPortfolioSpendingReduction':'The model reduces base spending by this percentage while total savings are below half their value at retirement.',
  'socialSecurity.annualBenefitAt67':'Enter an annual Social Security estimate at age 67 in today’s dollars. The model adjusts it for the claim age below and applies its own future inflation.',
  'socialSecurity.spouseClaimAge':'The spouse’s modeled claiming age. Spousal or survivor payments may begin later if other conditions are not met.',
  'guaranteedIncome.annualIncome':'Annual pension or annuity income in today’s dollars, separate from Social Security. The annual increase applies from today, including years before payments begin at the income start age.',
  'guaranteedIncome.annualIncrease':'Yearly growth rate for the pension or annuity amount entered above.',
  'guaranteedIncome.survivorPercent':'For a married plan, the share of guaranteed income retained by the spouse after the primary person dies.',
  'home.currentValue':'The home is not counted as spendable savings initially. The model sells it when the portfolio runs out or every surviving household member is in long-term care, pays the remaining mortgage, adds the net equity to cash, and stops the property tax and home insurance entered below. Replacement rent applies only while someone lives outside care.',
  'home.annualTaxesAndInsurance':'Yearly property tax and homeowners insurance in today\'s dollars. These bills belong in annual base spending too; this amount tells the model how much of that spending stops after the home is sold. Using the budget estimate fills it in. Enter 0 if you do not own a home.',
  'healthcare.preMedicareMonthlyPremium':'Monthly healthcare premium for each retired adult younger than 65. The model grows this cost with healthcare inflation.',
  'healthcare.healthcareInflationStdDev':'Standard deviation of the compounded yearly change in healthcare costs. Monthly changes match this yearly volatility and the entered yearly average; cost decreases are allowed.',
  'healthcare.includeMedicarePremiums':'Adds modeled Medicare premiums from age 65, including higher-income surcharges when applicable.',
  'longTermCare.enabled':'Adds a chance of long-term care costs near the end of each modeled life, using the cost and duration below.',
  'market.preRetirementMeanReturn':'Average annual investment return before retirement. This strongly affects the assets available at retirement.',
  'market.preRetirementStdDev':'Standard deviation of compounded yearly investment returns before retirement. Monthly returns match this yearly volatility and the entered yearly average.',
  'market.stockStdDev':'Standard deviation of compounded yearly stock returns after retirement. Monthly returns match this yearly volatility and the entered yearly average.',
  'rothConversion.enabled':'Moves some pre-tax savings into Roth savings at year end and pays the estimated conversion tax from the portfolio. With early-withdrawal penalties enabled, withdrawals of newly converted principal before age 59½ can incur a 10% penalty for five modeled tax years, including money used to pay conversion tax.',
  'rothConversion.marginalRateCap':'The highest federal income tax bracket the model fills with Roth conversions. For example, 22% fills available room through the 22% bracket; it is not a flat 22% tax on the whole conversion. Applies only when Roth conversions are on. The 37% bracket has no upper income limit, so the model may convert the entire remaining pre-tax balance.',
  'withdrawalStrategy.useCashReserveDuringDrawdowns':'Uses the cash reserve before invested accounts when the modeled monthly portfolio return falls below the trigger.',
  'withdrawalStrategy.drawdownTrigger':'The monthly portfolio return that activates cash-first withdrawals. For example, -1% means a month below -1%.',
  'withdrawalStrategy.applyEarlyWithdrawalPenalty':'On by default for new plans. Adds a modeled 10% penalty to applicable withdrawals before age 59½, including nonqualified Roth earnings and taxable conversion principal within its separate five-tax-year period. Regular Roth contributions are exempt. Roth earnings income tax is calculated even when this penalty option is off.',
  'withdrawalStrategy.ruleOf55Eligible':'Select only when the pre-tax savings modeled here are held in a qualifying employer plan, with separation from that employer during or after the calendar year you turn 55. The date alone cannot confirm eligibility. Separation can precede your birthday in that year. This declaration exempts primary-owned pretax withdrawals, not IRA withdrawals or Roth conversion recapture. The model checks the birth year against the selected retirement year.',
  'withdrawalStrategy.seppEligible':'An optional withdrawal strategy you choose, not eligibility inferred from your age. For retirement before 59½, starts fixed monthly payments from the entire pretax balance using 5% amortization and the IRS Single Life table. Until the later of five years or age 59½, extra withdrawals and Roth conversions from this balance are blocked. Other accounts must cover any spending gap or the model reports a shortfall. Verify actual eligibility and account setup separately.',
  'numberOfSimulations':'Number of simulated lifetimes. More paths make the readiness estimate steadier but take longer to run.',
};
function helpHTML(path,label){const tip=current().household.separatePeople&&['accounts.pretax','withdrawalStrategy.seppEligible'].includes(path)?path==='accounts.pretax'?'Your traditional 401(k), 403(b), IRA and similar pre-tax balances. Owner-specific age and RMD rules apply; employer-plan deferrals and after-tax basis are not modeled.':'Optional SEPP starts at your own retirement before 59½. Only your owned pre-tax balance is protected from extra withdrawals and conversions until the later of five years or age 59½. The spouse can select a separate SEPP.':assumptionHelp[path];if(!tip)return '';const id='help-'+path.replaceAll('.','-');return `<span class="help"><button type="button" class="help-trigger" aria-label="Explain ${escapeHTML(label)}" aria-describedby="${id}" aria-expanded="false">?</button><span class="help-popover" id="${id}" role="tooltip">${escapeHTML(tip)}</span></span>`;}
const fullRowFields=new Set(['longTermCare.enabled']);
const spouseFieldPaths=new Set(['household.spouseBirthday','household.spouseGender','socialSecurity.spouseClaimAge','guaranteedIncome.survivorPercent']);
function visibleFields(s,fields){return fields.filter(([,path])=>!(s.household.filingStatus!=='Married'&&(spouseFieldPaths.has(path)||path.startsWith('spouse')||path==='household.spouseRetirementDate'||path.startsWith('workingIncome')))&&!(!s.household.separatePeople&&(path.startsWith('spouseAccounts')||path.startsWith('spouseRothHistory')||path.startsWith('spouseIncome')||path.startsWith('spouseWithdrawal')||path.startsWith('workingIncome')||path==='household.spouseRetirementDate')));}
function getPath(obj,path){return path.split('.').reduce((v,k)=>v[k],obj);}
function setPath(obj,path,value){const keys=path.split('.');const last=keys.pop();keys.reduce((v,k)=>v[k],obj)[last]=value;}
function fieldHTML(s,[label,path,type,options]){
  const source=inputSource(s,state.inputSources,path),value=source==='Unknown'?'':getPath(s,path),id='f-'+path.replaceAll('.','-'),help=helpHTML(path,label);
  const guidance=(s.household.separatePeople&&path==='accounts.pretax'?'You · Your traditional 401(k), 403(b), IRA and similar pre-tax accounts today, from statements. Enter spouse accounts separately. Non-Roth after-tax basis and plan-specific exceptions are not modeled.':s.household.separatePeople&&path==='accounts.roth'?'You · Your Roth IRA total today. Your spouse’s Roth value and history are separate. Employer Roth rules are not modeled.':s.household.separatePeople&&path==='household.retirementDate'?'You · Your retirement date stops your future savings deposits. Household costs begin at the earlier retirement date.':s.household.separatePeople&&path==='socialSecurity.spouseClaimAge'?'Spouse · Their own Social Security claim age, from 62 to 70. Spousal and survivor supplements are evaluated separately.':FIELD_GUIDANCE[path])||assumptionHelp[path]||'Review this assumption before running. Enter 0 only when the actual amount is zero.';
  const sourceControl=FIELD_GUIDANCE[path]&&path!=='household.filingStatus'?`<div class="input-source"><label for="${id}-source">Value source</label><select id="${id}-source" data-input-source="${path}" aria-label="Source for ${escapeHTML(label)}">${(source==='Saved value; source not recorded'?INPUT_SOURCES:INPUT_SOURCES.filter(x=>x!=='Saved value; source not recorded')).map(x=>`<option ${x===source?'selected':''}>${escapeHTML(x)}</option>`).join('')}</select></div>`:`<small class="source-label">${escapeHTML(source)}</small>`;
  const note=`<p class="field-guidance" id="${id}-guide">${escapeHTML(guidance)}</p>${sourceControl}`;
  if(path==='numberOfSimulations'&&!isPro())return `<div class="field"><div class="field-label"><span class="field-title">Simulation paths</span>${help}</div><strong class="field-static">10 paths · Free preview</strong><p class="form-note">Upgrade to choose up to 10,000 paths. <button class="text-link" data-view="billing">View Pro</button></p></div>`;
  if(type==='checkbox')return `<div class="field checkbox${fullRowFields.has(path)?' full-row':''}"><input id="${id}" type="checkbox" data-field="${path}" data-type="${type}" ${value?'checked':''}><label for="${id}">${escapeHTML(label)}</label>${help}${note}</div>`;
  const heading=`<div class="field-label"><label for="${id}">${escapeHTML(label)}</label>${help}</div>`;
  if(path==='rothConversion.marginalRateCap'){
    const supported=ROTH_CONVERSION_RATES.some(rate=>Math.abs(rate-value)<.0001);
    return `<div class="field">${heading}<select id="${id}" data-field="${path}" data-type="percent">${supported?'':`<option selected disabled value="${escapeHTML(value*100)}">Unsupported value (${escapeHTML(value*100)}%) — choose a bracket</option>`}${ROTH_CONVERSION_RATES.map(rate=>`<option value="${Math.round(rate*100)}" ${Math.abs(rate-value)<.0001?'selected':''}>${Math.round(rate*100)}%</option>`).join('')}</select>${note}</div>`;
  }
  if(type==='date')return `<div class="field">${heading}<input id="${id}" type="date" required ${path.endsWith('RetirementDate')||path==='household.retirementDate'?`min="${localCalendarDate()}"`:`max="${localCalendarDate()}"`} data-field="${path}" data-type="date" value="${escapeHTML(value)}" aria-describedby="${id}-guide">${note}</div>`;
  if(type==='select')return `<div class="field">${heading}<select id="${id}" data-field="${path}" data-type="${type}">${options.split('|').map(opt=>`<option value="${opt}" ${opt===value?'selected':''}>${escapeHTML(path.endsWith('gender')||path.endsWith('Gender')?opt+' mortality rates':opt==='HeadOfHousehold'?'Head of household':opt==='EmpiricalAgeDecline'?'Empirical age decline':opt)}</option>`).join('')}</select>${note}</div>`;
  const monthPath=monthFields[path];
  if(monthPath){
    const monthId='f-'+monthPath.replaceAll('.','-'),monthValue=getPath(s,monthPath);
    return `<div class="field"><div class="timing-inputs">${heading}<label class="timing-months" for="${monthId}">Extra months</label><input id="${id}" type="number" inputmode="numeric" step="1" min="0" data-field="${path}" data-type="number" value="${escapeHTML(value)}"><select id="${monthId}" data-field="${monthPath}" data-type="month">${Array.from({length:12},(_,month)=>`<option value="${month}" ${month===monthValue?'selected':''}>${month}</option>`).join('')}</select></div>${note}</div>`;
  }
  const resource=['socialSecurity.annualBenefitAt67','spouseIncome.annualBenefitAt67'].includes(path)?`<div class="field-resource"><a href="https://www.ssa.gov/myaccount/" target="_blank" rel="noopener noreferrer">Find your estimate at my Social Security ↗</a><small>Choose today’s dollars if offered; multiply the age-67 monthly estimate by 12.</small></div>`:'';
  return `<div class="field">${heading}<input id="${id}" type="number" inputmode="decimal" step="${type==='number'?1:'any'}" ${path==='numberOfSimulations'?`min="${MIN_SIMULATION_PATHS}" max="${MAX_SIMULATION_PATHS}"`:''} aria-describedby="${id}-guide" data-field="${path}" data-type="${type}" value="${escapeHTML(type==='percent'&&value!==''?Number((value*100).toPrecision(10)):value)}">${resource}${note}</div>`;
}
const setupSections=[
  ['Household','Your timeline and household','Set your retirement timing and the household used in the model.'],
  ['Accounts & spending','Savings and everyday spending','Enter current balances and the living costs your savings will need to support.'],
  ['Income & Social Security','Income you can plan around','Add Social Security and any pension or annuity income.'],
  ['Housing & healthcare','Housing and health costs','Mortgage, rent, and healthcare premiums are modeled separately from annual base spending. Home details set what changes after a sale.'],
  ['Market & strategy','Investments and withdrawal strategy','Review return assumptions and how the plan uses your savings.']
];
// The Roth total sits with the other balances; this is the history behind it.
function rothInputs(s){
  const rh=s.rothHistory;
  const conversions=rh.conversions.map((lot,i)=>`<div class="inset-details"><h4>Past conversion ${i+1}</h4><div class="fields compact">${[['Conversion tax year','taxYear','number'],['Remaining conversion principal','amount','money'],['Remaining taxable principal','taxableAmount','money']].map(([label,key,type])=>fieldHTML(s,[label,`rothHistory.conversions.${i}.${key}`,type])).join('')}</div><button class="text-link" data-action="remove-roth-conversion" data-index="${i}">Remove conversion</button></div>`).join('');
  return `<div class="fields">${fieldHTML(s,schema[6][1][0])}</div>${rh.needsReview?'<div class="notice">This older plan had no Roth history. Its starting value was carried forward as contributions, with 2021 assumed as the first funding year and no past conversions. Review all Roth entries, then <button class="text-link" data-action="review-roth-history">mark the history reviewed</button>.</div>':''}<details id="roth-history" class="advanced-settings inset-details" ${state.rothHistoryOpen||rh.needsReview?'open':''}><summary>Roth funding year and past conversions</summary><div class="fields">${fieldHTML(s,schema[6][1][1])}</div><p class="form-note">List remaining principal from conversions made before today, excluding investment growth. Taxable principal is the portion taxed as income when converted that has not yet been withdrawn. Enter 0 for a wholly nontaxable conversion. Each conversion has its own five-tax-year clock. The simulation adds future conversions automatically.</p>${conversions}<button class="secondary" data-action="add-roth-conversion">Add past conversion</button></details>`;
}
function earlyWithdrawalGuidance(s){
  const context=earlyWithdrawalContext(s),w=s.withdrawalStrategy,notes=[];
  if((context.earlyRetirement||context.youngerSpouse)&&!w.applyEarlyWithdrawalPenalty)notes.push('Penalty modeling is off in this plan. Review this choice if retirement savings may be withdrawn before age 59½.');
  if(context.youngerSpouse)notes.push('Your spouse will be younger than 59½ when you retire. Early-withdrawal penalties may matter if they later own and withdraw retirement savings.');
  if(context.ruleOf55Timing&&s.accounts.pretax>0&&!w.ruleOf55Eligible)notes.push('Your retirement date meets the Rule of 55 age requirement. Select the exception only if your employer plan qualifies; IRAs do not qualify.');
  if(w.ruleOf55Eligible&&!ruleOf55Applies(s))notes.push('Rule of 55 does not apply with this retirement date. Your choice has been kept; review the date and employer-plan eligibility.');
  if(context.earlyRetirement&&s.accounts.pretax>0&&!w.seppEligible)notes.push('A SEPP withdrawal plan is optional. It restricts withdrawals for at least five years and until age 59½.');
  if(w.seppEligible&&!context.earlyRetirement)notes.push('The model will not start SEPP payments when retirement begins at 59½ or later. Your choice has been kept.');
  return notes.map(note=>`<p class="form-note">${note}</p>`).join('');
}
function earlyWithdrawalReview(s){
  const context=earlyWithdrawalContext(s),w=s.withdrawalStrategy;
  if(!context.earlyRetirement&&!context.youngerSpouse&&!w.ruleOf55Eligible&&!w.seppEligible)return '';
  return `<p class="form-note">Check your early-withdrawal choices when changing household dates. <button class="text-link" data-action="setup-section" data-index="4">Review withdrawal choices</button></p>`;
}
function earlyWithdrawalInputs(s){
  return `<div class="fields early-access-fields">${schema[4][1].slice(10,13).map(f=>fieldHTML(s,f)).join('')}</div><div id="early-withdrawal-guidance" class="early-access-guidance" aria-live="polite">${earlyWithdrawalGuidance(s)}</div>`;
}
function refreshEarlyWithdrawalGuidance(){
  if(state.view!=='setup')return;
  const guidance=$('#early-withdrawal-guidance'),review=$('#early-withdrawal-review');
  if(guidance)guidance.innerHTML=earlyWithdrawalGuidance(current());
  if(review)review.innerHTML=earlyWithdrawalReview(current());
}
function setup(){
  const s=current(),i=state.setupSection;
  if(i===5)return setupReview(s);
  const [label,title,description]=setupSections[i];
  const couple=s.household.filingStatus==='Married',separate=s.household.separatePeople,separateCouple=separate&&couple;
  const render=rows=>visibleFields(s,rows).filter(([,path])=>path!=='household.targetEndAge').map(f=>fieldHTML(s,f)).join('');
  const fields=(section,start,end,layout='fields')=>`<div class="${layout}">${render(schema[section][1].slice(start,end))}</div>`;
  // Picks fields across sections so each person's inputs share one grid in a sensible order.
  const pick=(...refs)=>`<div class="fields">${render(refs.map(([section,n])=>schema[section][1][n]))}</div>`;
  const group=(title,content,note='')=>`<section class="form-group"><div class="group-heading"><h3>${title}</h3>${note?`<p>${note}</p>`:''}</div>${content}</section>`;
  const savingsFields=index=>fields(index,0,3)+(state.guided?`<details class="guided-extra"><summary>Other savings and annual deposit increases</summary>${fields(index,3)}</details>`:fields(index,3));
  const allocation=`<details id="allocation-settings" class="advanced-settings inset-details" ${state.allocationOpen?'open':''}><summary>Stock allocation by portfolio size</summary><p class="form-note">The app compares invested savings with one year of modeled costs. A 30× balance means invested savings equal 30 times that yearly amount; it is not a guarantee of 30 years of funding. Each percentage is the stock share; the rest is bonds.</p>${fields(5,0,undefined,'fields compact')}</details>`;
  const advanced=`<details id="advanced-model" class="advanced-settings inset-details" ${state.advancedOpen?'open':''}><summary>Advanced model settings</summary><p class="form-note">The maximum modeling age caps the simulation. It is not a predicted lifespan; each modeled lifetime uses mortality rates.</p><div class="fields">${fieldHTML(s,schema[0][1][2])}${schema[4][1].slice(13).map(f=>fieldHTML(s,f)).join('')}</div></details>`;
  let content=[
    group('Household',householdChoice(s)+(separate?'':fields(0,1,2)),'Who is this plan for? Couples use Married filing status.')+(state.guided?sampleInputNotice(s):'')+group('You',pick([0,0],...(separate?[[0,1]]:[]),[0,4])+(s.household.datesNeedReview?'<div class="notice">Dates were estimated from your saved ages. Check all household dates, then <button class="text-link" data-action="review-calendar-dates">mark the dates reviewed</button>.</div>':''),'Select dates using the calendar or type them in your browser’s date format. The model uses monthly steps.')+(couple?group('Spouse',pick([0,5],[0,7],[0,6]),separate?'Enter your spouse’s own retirement date. Shared household costs start at whichever retirement comes first.':'In this saved plan your spouse shares the retirement date above. Use separate-person inputs to give them their own date.'):'')+`<div id="early-withdrawal-review" class="early-access-review" aria-live="polite">${earlyWithdrawalReview(s)}</div>`,
    (separate?group('Your retirement accounts',pick([1,0],[1,1]),'Your own traditional 401(k), 403(b), IRA and Roth IRA balances today, from statements.'+(couple?' Enter your spouse’s accounts separately below.':''))+(couple?group('Spouse retirement accounts',fields(9,0,2),'Your spouse’s own balances today. If a statement combines both of you, split it between the two owners so nothing is counted twice.'):'')+group('Household · Shared balances',fields(1,2,4),'Taxable investments and cash are household totals.'):group('Household · Current account balances',fields(1,0,4),'Enter combined household balances today. Pre-tax balances use you as the modeled account owner, then the surviving spouse after the death year. Separate owners, employer-plan exceptions and after-tax non-Roth basis are not modeled.'))+group('Roth IRA history',rothInputs(s)+(separateCouple?`<h4>Spouse</h4>${fields(9,2)}${spouseConversions(s)}`:''),'Contributions and converted principal are already part of the Roth totals above; they are not added to savings again. These records decide which Roth withdrawals are tax- and penalty-free.')+group('Living costs',fields(1,4,6),'<button class="text-link" data-view="budget">Use the budget builder to estimate annual spending →</button>')+group('How spending changes',fields(1,6),'Percent fields accept values such as 2.3 for 2.3%.')+earningsExplanation(s)+growthHelperHTML(s)+group('Your future savings',savingsFields(7),'Annual deposits funded by earnings outside the model, paid monthly after growth. Include employee and employer contributions once. They are additional to balances today, not investment return. Contribution limits and eligibility are not checked.')+(couple?group('Spouse future savings',savingsFields(8),separate?'Their deposits stop at their own retirement date or death.':'In this shared-date plan, their deposits stop at the shared retirement date.'):''),
    group('You · Social Security',fields(2,0,2))+(couple?group('Spouse · Social Security',separate?pick([10,0],[2,2]):fields(2,2,3),separate?'Enter their own age-67 benefit. The model adds only an eligible excess spousal amount; survivor benefits replace a lower own benefit.':'Only a spousal or survivor benefit based on your record is modeled; a separate benefit based on spouse earnings is not supported.'):'')+group('You · Pension or annuity income',fields(2,3),separate?'A pension pays income; it is not an account balance. Enter 0 if you have none. Enter your pension here, with your spouse’s share after your death.':'A pension pays income; it is not an account balance. Enter 0 if you have none. Only one income stream is supported, with a survivor share for your spouse.')+(separateCouple?group('Spouse · Pension',fields(10,1),'One pension for each person, with separate start timing and survivor share. Enter 0 if they have none.'):'')+(separateCouple?group('Working household support',fields(11,0),'Optional take-home money available for household costs after taxes and these savings deposits. Stops at that person’s retirement or death. Employment gross income, payroll taxes, earnings tests and employment effects on federal brackets/Medicare are not modeled. Zero means savings pay the gap.') :''),
    group('Home & mortgage',fields(3,0,7),'Leave amounts at 0 where they do not apply.')+group('Healthcare premiums',fields(3,7,11))+group('Long-term care',fields(3,11)),
    group('Investment returns',fields(4,0,6),'Annual averages and volatility, entered as percentages. <a href="./methodology.html#sample-returns">Why the sample uses 13.3%</a>')+group('Roth conversions',fields(4,6,8))+group('Cash reserve strategy',fields(4,8,10))+group('Accessing retirement savings before 59½',earlyWithdrawalInputs(s),'Penalties apply only to applicable withdrawals. Select an exception only when it applies to your plan.')+(s.household.separatePeople&&s.household.filingStatus==='Married'?group('Spouse early-access choices',fields(12,0),'Applies to spouse-owned pre-tax accounts, using their separation date and age. SEPP protects only that owner’s balance.'):'')+allocation+advanced
  ][i];
  if(state.guided){
    // Support only matters when one person keeps working after the other retires.
    const overlap=separateCouple&&s.household.spouseRetirementDate!==s.household.retirementDate;
    const optional=i===1?[['Roth IRA history','Roth IRA history · contribution records and past conversions'],['How spending changes','How spending changes · inflation and spending cuts']]:i===2&&!overlap?[['Working household support','Working household support · only if one of you works after the other retires']]:i===3?[['Long-term care','Long-term care · Review details']]:i===4?[['Roth conversions','Roth conversions · Review details'],['Cash reserve strategy','Cash reserve strategy · Review details'],['Accessing retirement savings before 59½','Accessing retirement savings before 59½ · Review details']]:[];
    for(const [name,summary] of optional)content=content.replace(new RegExp('<section class="form-group"><div class="group-heading"><h3>'+name+'</h3>[\\s\\S]*?</section>'),'<details class="guided-extra"'+(state.advancedOpen?' open':'')+'><summary>'+summary+'</summary>$&</details>');
  }
  return `${pageHead(state.guided?'Build your forecast':'Set up your Monte Carlo model',state.guided?'A short guided setup. You can skip to the detailed editor at any time.':'Work through one section at a time. These inputs shape every simulated lifetime.',`<button class="secondary" data-action="toggle-guided">${state.guided?'Skip guide · Detailed editor':'Start guided setup'}</button>`)}${setupProgress(i)}${state.guided?'':sampleInputNotice(s)}${couple&&(!separate||i===1)?personModelNotice(s):''}${notice()}${state.guided?'':scopeDisclosure()+planNotice()}<div class="setup-layout${state.guided?' guided':''}">${state.guided?'':`<aside class="section-picker"><p class="section-eyebrow">Plan sections</p><div class="section-links" role="navigation" aria-label="Assumption sections">${setupSections.map(([name,,desc],n)=>`<button data-action="setup-section" data-index="${n}" ${i===n?'aria-current="step"':''}><span class="section-number">${n+1}</span><span>${name}</span><span class="section-arrow" aria-hidden="true">›</span></button>`).join('')}</div><div class="mobile-section"><label for="setup-section">Plan section</label><select id="setup-section">${setupSections.map(([name],n)=>`<option value="${n}" ${i===n?'selected':''}>${n+1}. ${name}</option>`).join('')}</select></div><div class="section-tip"><strong>Need a hand?</strong><p>Select a ? beside a field for a plain-language explanation.</p><p>Run the simulation after making changes to update your results.</p><p><strong>Statement check:</strong> under Accounts &amp; spending, separate last year’s new savings from investment growth. <a href="./methodology.html#planned-balance-history">How it works</a></p></div></aside>`}<div class="setup-content"><section class="card form-panel"><header class="form-panel-head"><span class="kicker">${state.guided?'Step':'Section'} ${i+1} of ${state.guided?6:setupSections.length}</span><h2 id="section-title" tabindex="-1">${title}</h2><p>${description}</p></header>${content}<div class="section-footer">${i?`<button class="secondary" data-action="setup-section" data-index="${i-1}">← Previous</button>`:'<span></span>'}${i<4?`<button class="primary" data-action="setup-section" data-index="${i+1}">Next: ${setupSections[i+1][0]} →</button>`:`<button class="primary" data-action="setup-section" data-index="5">Review assumptions →</button>`}</div></section><div class="setup-bottom"><span>Inputs are U.S. dollars today; results show future dollars. Blank or Unknown is not zero.</span><button class="text-link" data-action="reset-assumptions">Restore sample values</button></div></div></div>`;
}
// Unentered months and disclosure choices belong to the editor, not the plan.
const budgetViews=new Map();
function blankBudgetMonth(b){
  const date=new Date();date.setDate(1);date.setMonth(date.getMonth()-1);
  const used=new Set(b.monthlyBudgets.map(m=>m.month));let month;
  do{month=`${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}`;date.setMonth(date.getMonth()-1);}while(used.has(month));
  return {month,checkingSavingsBills:[],creditCardBills:[],cashAndAtmWithdrawals:0,adjustments:{}};
}
function budgetView(){
  const s=current();let ui=budgetViews.get(s.id);
  if(!ui){ui={open:{},pending:null,direction:'less'};budgetViews.set(s.id,ui);}
  if(!s.budget.monthlyBudgets.length&&!ui.pending)ui.pending=blankBudgetMonth(s.budget);
  return ui;
}
function rememberBudgetDisclosures(){
  const layout=$('.budget-layout'),ui=budgetViews.get(layout?.dataset.scenarioId);
  if(ui)for(const el of layout.querySelectorAll('details[data-budget-disclosure]'))ui.open[el.dataset.budgetDisclosure]=el.open;
}
function budgetOpen(key,fallback=false){return (budgetView().open[key]??fallback)?' open':'';}
function budgetMonthLabel(m){
  const match=/^(\d{4})-(0[1-9]|1[0-2])$/.exec(m.month);
  return match?new Intl.DateTimeFormat('en-US',{month:'long',year:'numeric'}).format(new Date(Number(match[1]),Number(match[2])-1,1)):'Choose a month';
}
function retirementAdjustmentSummary(b){const amount=b.retirementAnnualAdjustment||0;return amount?`${money(Math.abs(amount)/12,2)} / month ${amount<0?'less':'more'}`:'No change';}
function budgetSummary(){
  const s=current(),b=s.budget,d=budgetBreakdown(b),errors=validateBudget(b,{requireMonths:true}),applied=b.isAppliedToAnnualBaseSpending&&!b.estimateNeedsReview;
  const visibleErrors=d.count?errors:validateBudget(b);
  return `<div class="budget-summary-heading"><h2>Your spending estimate</h2><span class="budget-status ${applied?'applied':'draft'}">${applied?'Applied':'Draft'}</span></div>
    ${visibleErrors.length?`<div class="budget-errors" role="status"><ul>${visibleErrors.map(e=>`<li>${escapeHTML(e)}</li>`).join('')}</ul></div>`:''}
    <div class="budget-result"><div><span>Monthly equivalent</span><strong>${errors.length?'—':money(d.estimate/12,2)}</strong><small>per month</small></div><div><span>Annual spending</span><strong>${errors.length?'—':money(d.estimate,2)}</strong><small>per year</small></div></div>
    <p class="form-note budget-exclusions">In today’s dollars. Excludes mortgage/rent and health premiums entered in Assumptions.</p>
    ${d.count?`<p class="form-note">Based on ${d.count} ${d.count===1?'month':'months'} of spending.</p>${d.count<12?'<p class="sample-note">Add more complete months for a fuller picture. Twelve months capture seasonal bills.</p>':''}${b.monthlyBudgets.length>12?'<p class="sample-note">Only the 12 most recent months are used.</p>':''}
      <details class="budget-calculation" data-budget-disclosure="calculation"${budgetOpen('calculation')}><summary>How this was calculated</summary><p class="form-note">Months used: ${d.months.map(m=>escapeHTML(m.month)).join(', ')}.</p><div class="info-list">${info('Average monthly spending',money(d.grossAverage,2))}${info('Annual bills already counted (avg.)','− '+money(d.annualBillsAverage,2))}${info('Separate plan costs (avg.)','− '+money(d.separateCostsAverage,2))}${info('Monthly spending after adjustments',money(d.monthlyAverage,2))}${info('Annualized spending (× 12)',money(d.annualized,2))}${info('Annual bills added once','+ '+money(d.annualBills,2))}${info('Retirement adjustment',(d.retirementAdjustment<0?'− ':'+ ')+money(Math.abs(d.retirementAdjustment),2))}</div></details>`:'<p class="form-note">Enter spending for your first month to see an estimate.</p>'}
    <p class="budget-plan-status">${applied?'Applied to your plan.':'Draft — your plan still uses '+money(s.spending.annualBaseSpending,2)+' / year.'}</p>
    <p class="form-note">${applied?'Run your plan to see the forecast.':'Use this spending when you’re ready to update your plan.'}</p>`;
}
function budgetMonthSummary(m){const t=budgetMonthTotals(m);return `<span>Total spending <strong>${money(t.gross,2)}</strong></span><span>Costs already counted <strong>− ${money(t.annualBills+t.separateCosts,2)}</strong></span><span>After adjustments <strong>${money(t.adjusted,2)}</strong></span>`;}
function budgetMonthEditor(m,i,pending){
  const t=budgetMonthTotals(m);
  const field=(key,label,value,allowNegative=false)=>`<div class="field"><label for="month-${i}-${key}">${label}</label><input id="month-${i}-${key}" type="number" ${allowNegative?'':'min="0"'} step="any" data-month="${i}" data-part="${key}" value="${escapeHTML(value)}"></div>`;
  return `<details class="spending-month" id="budget-month-${i}" data-budget-disclosure="month-${i}"${budgetOpen('month-'+i,pending)}>
    <summary><span id="month-${i}-label">${escapeHTML(budgetMonthLabel(m))}</span><strong id="month-${i}-amount">${pending?'New month':money(t.adjusted,2)}</strong><span class="month-edit-label">Edit</span><span class="month-open-label">Editing</span></summary>
    <div class="month-editor"><div class="month-heading"><label for="month-${i}-date">Month<input id="month-${i}-date" type="month" data-month="${i}" data-part="month" value="${escapeHTML(m.month)}"></label><button class="subtle" id="month-${i}-remove" data-action="remove-month" data-index="${i}" aria-label="Remove ${escapeHTML(budgetMonthLabel(m))}" ${pending&&!current().budget.monthlyBudgets.length?'hidden':''}>Remove</button></div>
    ${pending?'<p class="form-note" id="budget-new-month-note">Enter a total to include this month. Enter 0 if you had no spending.</p>':''}
    <div class="fields monthly-amounts">${field('credit','Credit card purchases',pending?'':t.credit,true)}${field('checking','Checking / savings spending',pending?'':t.checking)}${field('cashAndAtmWithdrawals','Cash / ATM withdrawals',pending?'':m.cashAndAtmWithdrawals||0)}</div>
    <details class="budget-adjustments" data-budget-disclosure="adjustments-${i}"${budgetOpen('adjustments-'+i)}><summary>Avoid counting costs twice</summary><p class="form-note">Only deduct payments already included in the totals above. If you left a bill out, leave its deduction at 0. Deduct each payment once.</p><h3>Annual bills paid this month</h3><p class="form-note">Deduct these payments here, then enter their yearly amounts under Annual bills below.</p><div class="fields compact">${ANNUAL_BILLS.map(([label,,key])=>field(key,label+' already counted',m.adjustments?.[key]||0)).join('')}</div><h3>Housing &amp; health premiums</h3><p class="form-note">Deduct included mortgage, rent and health premiums. Enter their retirement amounts in <button class="text-link" data-action="housing-assumptions">Housing &amp; healthcare</button>. Keep other medical spending here. Split any mortgage escrow into taxes and insurance; deduct each part once.</p><div class="fields compact">${SEPARATE_COSTS.map(([label,key])=>field(key,label+' already counted',m.adjustments?.[key]||0)).join('')}</div></details>
    <div class="month-totals" data-month-totals="${i}">${budgetMonthSummary(m)}</div><button class="secondary month-done" id="month-${i}-done" data-action="finish-month" data-index="${i}" ${pending?'disabled':''}>Done with this month</button></div></details>`;
}
function budget(){
  const s=current(),b=s.budget,ui=budgetView(),months=[...b.monthlyBudgets,...(ui.pending?[ui.pending]:[])];
  const annualBills=budgetBreakdown(b).annualBills,direction=b.retirementAnnualAdjustment?(b.retirementAnnualAdjustment<0?'less':'more'):ui.direction;
  return `${pageHead('Build an annual budget','Start with your monthly spending. Add optional details as you need them.')}${notice()}${scopeDisclosure()}<div class="budget-layout" data-scenario-id="${escapeHTML(s.id)}"><div class="stack">
    ${card('Monthly spending',`<p class="form-note budget-guidance">Combine all your accounts for each complete month. <strong>Don’t include credit card payments or transfers.</strong></p>
      <details class="budget-help" data-budget-disclosure="help"${budgetOpen('help')}><summary>Help with these totals</summary><div class="spending-guide"><div><strong>Credit card purchases</strong><p>Purchases minus refunds. Exclude card payments, balance transfers and cash advances. Net refunds can be negative.</p></div><div><strong>Checking / savings spending</strong><p>Direct bills, checks and debit purchases. Exclude credit card payments, ATM withdrawals and transfers between your own accounts.</p></div><div><strong>Cash / ATM withdrawals</strong><p>Cash withdrawn for spending. We assume it was spent; don’t also include the cash purchases in another total.</p></div></div></details>
      <div class="month-list">${months.map((m,i)=>budgetMonthEditor(m,i,m===ui.pending)).join('')}</div><button class="secondary" id="budget-add-month" data-action="add-month" ${ui.pending?'disabled':''}>+ Add another month</button>`)}
    <details class="card budget-option" data-budget-disclosure="annual-bills"${budgetOpen('annual-bills')}><summary><span><strong>Annual bills</strong><small>Optional</small></span><span class="budget-option-value" id="budget-annual-bills-summary">${annualBills?money(annualBills,2)+' / year':'Not added'}</span></summary><div class="budget-option-body"><p class="form-note">Enter yearly amounts in today’s dollars. If a payment is already in a month’s spending, deduct it under that month’s “Avoid counting costs twice.” Property tax and home insurance also become the plan’s home costs, which stop after a home sale.</p><div class="fields compact">${ANNUAL_BILLS.map(([label,key])=>`<div class="field"><label for="budget-${key}">${label} / year</label><input id="budget-${key}" type="number" min="0" step="any" data-budget="${key}" value="${escapeHTML(b[key])}"></div>`).join('')}</div></div></details>
    <details class="card budget-option" data-budget-disclosure="retirement"${budgetOpen('retirement')}><summary><span><strong>Adjust for retirement</strong><small>Optional</small></span><span class="budget-option-value" id="budget-retirement-summary">${retirementAdjustmentSummary(b)}</span></summary><div class="budget-option-body"><p class="form-note">About how much more or less will you spend each month in retirement? Use today’s dollars. Leave out housing and health premiums entered separately.</p><div class="fields retirement-adjustment-fields"><div class="field"><label for="budget-retirement-direction">In retirement, I expect to</label><select id="budget-retirement-direction" data-budget-adjustment="direction"><option value="less" ${direction==='less'?'selected':''}>Spend less</option><option value="more" ${direction==='more'?'selected':''}>Spend more</option></select></div><div class="field"><label for="budget-retirement-monthly-adjustment">Amount / month</label><input id="budget-retirement-monthly-adjustment" type="number" min="0" step="any" data-budget-adjustment="amount" value="${Number((Math.abs(b.retirementAnnualAdjustment||0)/12).toFixed(2))}"></div></div></div></details>
    </div><aside class="card budget-summary" id="budget-summary"><div id="budget-summary-body">${budgetSummary()}</div><button class="primary" data-action="apply-budget" ${validateBudget(b,{requireMonths:true}).length?'disabled':''}>Use this spending in my plan</button><p class="form-note budget-apply-note">Also updates your plan’s property tax and home insurance from Annual bills.</p></aside></div>`;
}
function refreshBudget(){
  if(state.view!=='budget')return;
  rememberBudgetDisclosures();refreshNotice();
  const b=current().budget,ui=budgetView();
  $('#budget-summary-body').innerHTML=budgetSummary();
  $('#budget-summary [data-action=apply-budget]').disabled=validateBudget(b,{requireMonths:true}).length>0;
  const annualBills=budgetBreakdown(b).annualBills;
  $('#budget-annual-bills-summary').textContent=annualBills?money(annualBills,2)+' / year':'Not added';
  $('#budget-retirement-summary').textContent=retirementAdjustmentSummary(b);
  $('#budget-add-month').disabled=Boolean(ui.pending);
  [...b.monthlyBudgets,...(ui.pending?[ui.pending]:[])].forEach((m,i)=>{
    const row=document.querySelector(`[data-month-totals="${i}"]`);if(row)row.innerHTML=budgetMonthSummary(m);
    $('#month-'+i+'-label').textContent=budgetMonthLabel(m);
    $('#month-'+i+'-amount').textContent=m===ui.pending?'New month':money(budgetMonthTotals(m).adjusted,2);
    $('#month-'+i+'-done').disabled=m===ui.pending;
    const remove=$('#month-'+i+'-remove');remove.hidden=m===ui.pending&&!b.monthlyBudgets.length;remove.setAttribute('aria-label','Remove '+budgetMonthLabel(m));
  });
  if(!ui.pending)$('#budget-new-month-note')?.remove();
}
function scenarios(){return `${pageHead('Saved scenarios','Keep each set of assumptions so you can compare its Monte Carlo results.',`<button class="primary" data-action="new-scenario">Duplicate current plan</button>`)}${notice()}${card('Plans',state.scenarios.map(s=>`<div class="scenario-row"><div><h3>${escapeHTML(s.name)} ${s.id===state.selectedId?'<span class="tag">Selected</span>':''}</h3><p>Retire ${dateLabel(s.household.retirementDate)} · Your age ${ageLabel(primaryRetirementAge(s))} · ${money(s.spending.annualBaseSpending)} annual spending · ${money(totalSavings(s))} assets</p></div><div class="actions"><button class="secondary" data-action="select-scenario" data-id="${escapeHTML(s.id)}">Open</button><button class="subtle" data-action="rename-scenario" data-id="${escapeHTML(s.id)}">Rename</button><button class="danger" data-action="delete-scenario" data-id="${escapeHTML(s.id)}" ${state.scenarios.length===1?'disabled':''}>Delete</button></div></div>`).join(''))}`;}
const labVariants=[['Your retirement 2 years later',s=>delayRetirement(s,2)],['Spend 5% less',s=>setAnnualBaseSpending(s,s.spending.annualBaseSpending*.95)],['Claim Social Security at 70',s=>s.socialSecurity.claimAge=70],['Higher healthcare costs',s=>{s.healthcare.preMedicareMonthlyPremium*=1.25;s.healthcare.healthcareInflationMean=Math.min(.20,s.healthcare.healthcareInflationMean+.01);}],['Use Roth conversions',s=>{s.rothConversion.enabled=true;s.rothConversion.marginalRateCap=.22;}],['Use cash first in months below −1%',s=>{s.withdrawalStrategy.useCashReserveDuringDrawdowns=true;s.withdrawalStrategy.drawdownTrigger=-.01;}]];
function ageSearchNote(d){
  const start=d.retirementAgeSearchStart,end=d.retirementAgeSearchEnd,paths=`${d.simulationCount} paths per age`;
  if(!Number.isInteger(start)||!Number.isInteger(end))return `Searches whole-year retirement ages with ${paths}.`;
  return end<start?'No whole-year retirement age before the maximum modeling age was available to test.':`Searches whole-year retirement ages ${start} through ${end} with ${paths}.`;
}
function householdChoice(s){
  const couple=s.household.filingStatus==='Married';
  return `<div class="household-choice" role="group" aria-label="Who is this plan for?"><button class="${couple?'secondary':'primary'}" data-action="household-choice" data-kind="individual" aria-pressed="${!couple}">Individual · Just me</button><button class="${couple?'primary':'secondary'}" data-action="household-choice" data-kind="couple" aria-pressed="${couple}">Couple · Me and my spouse</button></div>${fieldHTML(s,schema[0][1][3])}`;
}
function setupProgress(i){
  if(!state.guided)return '';
  const names=[...setupSections.map(([name])=>name),'Review'];
  return `<div class="setup-progress"><label for="setup-progress">Step ${i+1} of 6 · ${names[i]}</label><progress id="setup-progress" value="${i+1}" max="6"></progress><nav aria-label="Guided setup steps"><ol class="setup-steps">${names.map((name,n)=>`<li><button class="text-link" data-action="setup-section" data-index="${n}" ${n===i?'aria-current="step"':''}>${n+1}. ${escapeHTML(name)}</button></li>`).join('')}</ol></nav></div>`;
}
function earningsExplanation(s){
  const points=EARNINGS_POINTS.slice(0,s.household.separatePeople&&s.household.filingStatus==='Married'?4:3);
  return `<aside class="earnings-note" aria-labelledby="earnings-title"><strong id="earnings-title">How earnings, savings and growth fit together</strong><ul>${points.map(([lead,text])=>`<li><strong>${escapeHTML(lead)}</strong> ${escapeHTML(text)}</li>`).join('')}</ul><p>Don’t use an account’s total balance growth as your return assumption; it also includes deposits and transfers. The statement check below separates them.</p></aside>`;
}
// Statement-check values are a calculator, not plan assumptions; they stay in this tab.
const growthHelpers=new Map();
const GROWTH_ACCOUNTS=[['pretax','Pre-tax retirement: 401(k), 403(b), traditional IRA'],['roth','Roth IRA'],['taxable','Taxable brokerage'],['cash','Cash savings']];
function growthHelper(s){
  let h=growthHelpers.get(s.id);
  if(!h){h={open:false,owner:'contributions',account:'pretax',start:'',end:'',yours:'',employer:'',transfersIn:'',out:''};growthHelpers.set(s.id,h);}
  if(s.household.filingStatus!=='Married')h.owner='contributions';
  return h;
}
function growthHelperResult(s){
  const h=growthHelper(s),employer=h.account==='pretax',r=oneYearGrowth(h,{employer});
  if(!r.complete)return `<p class="form-note">${r.error||'Enter every amount above to see the split. Use 0 for none.'}</p>`;
  const owner=h.owner==='spouseContributions'?'your spouse’s':'your',account={pretax:'pre-tax retirement',roth:'Roth IRA',taxable:'taxable brokerage',cash:'cash'}[h.account];
  const rate=r.rate===null?'Not enough to estimate':`${r.rate<0?'−':''}${pct(Math.abs(r.rate))}`;
  return `<div class="info-list">${info('New savings (yours'+(employer?' + employer':'')+')',money(r.savings,2))}${info('Net money moved in (− if out)',`${r.net<0?'−':''}${money(Math.abs(r.net),2)}`)}${info('Investment growth after fees',`${r.growth<0?'−':''}${money(Math.abs(r.growth),2)}`)}${info('Approximate one-year return',rate)}</div><p class="form-note">The return is for information only and is not applied to your plan. One year is far too variable to use as a long-term average: S&amp;P 500 yearly returns ranged from −18% to +28% in 2021–2025. Your plan assumes ${pct(s.market.preRetirementMeanReturn)} a year before retirement. <a href="./methodology.html#planned-balance-history">How this is calculated</a></p><button class="secondary" data-action="apply-growth-savings">Use as ${owner} yearly ${escapeHTML(account)} savings</button><p class="form-note">Sets ${employer?`employee savings to ${money(r.yourSavings)} and employer savings to ${money(r.employerSavings)}`:`yearly savings to ${money(r.yourSavings)}`} a year, replacing the current amount, and marks it Estimated. Rollovers and withdrawals are not copied.</p>`;
}
function growthHelperHTML(s){
  const h=growthHelper(s),employer=h.account==='pretax',open=h.open;
  const input=(key,label,hint)=>`<div class="field"><label for="growth-${key}">${label}</label><input id="growth-${key}" type="number" inputmode="decimal" min="0" step="any" data-growth-helper="${key}" value="${escapeHTML(h[key])}" aria-describedby="growth-${key}-hint"><p class="field-guidance" id="growth-${key}-hint">${hint}</p></div>`;
  const select=(key,label,options)=>`<div class="field"><label for="growth-${key}">${label}</label><select id="growth-${key}" data-growth-helper="${key}">${options.map(([value,text])=>`<option value="${value}" ${value===h[key]?'selected':''}>${escapeHTML(text)}</option>`).join('')}</select></div>`;
  return `<details class="inset-details growth-helper" id="growth-helper" data-scenario-id="${escapeHTML(s.id)}" ${open?'open':''}><summary>Check last year’s statements: separate new savings from investment growth</summary><p class="form-note">For one account, compare a statement from about a year ago with today’s. Enter every amount, using 0 for none. These values stay in this browser tab and are not saved with your plan.</p><div class="fields">${s.household.filingStatus==='Married'?select('owner','Whose account?',[['contributions','You'],['spouseContributions','Spouse']]):''}${select('account','Account type',GROWTH_ACCOUNTS)}${input('start','Balance about one year ago','From the statement closest to a year ago.')}${input('end','Balance today','From your latest statement.')}${input('yours','Your contributions during the year','Payroll deferrals or deposits you made. Reinvested dividends are growth, not contributions.')}${employer?input('employer','Employer contributions during the year','Match or profit sharing, from the plan statement.'):''}${input('transfersIn','Other money moved in','Rollovers or transfers from your other accounts. These are not new savings.')}${input('out','Money taken out','Withdrawals, loans and transfers out.')}</div><div id="growth-helper-result" aria-live="polite">${growthHelperResult(s)}</div></details>`;
}
function sampleInputNotice(s,reviewLink=true){
  const count=schema.flatMap(([,fields])=>visibleFields(s,fields)).filter(([,path])=>path!=='household.filingStatus'&&inputSource(s,state.inputSources,path)==='Sample/default').length;
  return `<aside class="setup-notice"><strong>${count?'Sample values are still in this plan':'Check your input sources'}</strong><p>${count?'Replace sample values, or keep them for an illustration.':'Older saved values have no recorded source unless you identify one.'} Use Value source to mark estimates or Unknown amounts. Zero means none.${reviewLink?' <button class="text-link" data-action="setup-section" data-index="5">Review all assumptions</button>':''}</p></aside>`;
}
// Schema sections are grouped differently from the five setup steps.
const setupStepForSchema=index=>[6,7,8,9].includes(index)?1:[10,11].includes(index)?2:index===12?4:Math.min(index,4);
function reviewGlance(s){
  const couple=s.household.filingStatus==='Married',separateCouple=couple&&s.household.separatePeople;
  const deposits=c=>c.pretax+c.employerPretax+c.roth+c.taxable+c.cash;
  const tag=paths=>{if(paths.length===1&&paths[0]==='household.filingStatus')return 'Selected';const sources=paths.map(path=>inputSource(s,state.inputSources,path));return sources.includes('Unknown')?'Unknown':sources.includes('Sample/default')?'Includes sample values':sources.every(x=>x==='Entered')?'Entered':'Mixed or estimated';};
  const pension=(amount,age,months)=>amount>0?`${money(amount)} / year from age ${ageLabel(age+months/12)}`:'None';
  const rows=[
    ['Household',couple?(separateCouple?'Couple · each with their own retirement date':'Couple · one shared retirement date'):'Individual',['household.filingStatus'],0],
    ['Your retirement',`${dateLabel(s.household.retirementDate)} · age ${ageLabel(primaryRetirementAge(s))}`,['household.birthday','household.retirementDate'],0],
    ...(separateCouple?[['Spouse retirement',dateLabel(s.household.spouseRetirementDate),['household.spouseBirthday','household.spouseRetirementDate'],0]]:[]),
    ['Savings today',`${money(totalSavings(s))}${separateCouple?` · You ${money(s.accounts.pretax+s.accounts.roth)}, Spouse ${money(s.spouseAccounts.pretax+s.spouseAccounts.roth)}, shared ${money(s.accounts.taxable+s.accounts.cash)}`:''}`,['accounts.pretax','accounts.roth','accounts.taxable','accounts.cash',...(separateCouple?['spouseAccounts.pretax','spouseAccounts.roth']:[])],1],
    ['Future savings / year',`You ${money(deposits(s.contributions))}${couple?` · Spouse ${money(deposits(s.spouseContributions))}`:''}`,['contributions.pretax','contributions.employerPretax','contributions.roth',...(couple?['spouseContributions.pretax','spouseContributions.employerPretax','spouseContributions.roth']:[])],1],
    ['Annual base spending',money(s.spending.annualBaseSpending),['spending.annualBaseSpending'],1],
    ['Social Security at 67 / year',`You ${money(s.socialSecurity.annualBenefitAt67)}${couple?` · Spouse ${separateCouple?money(s.spouseIncome.annualBenefitAt67)+' own benefit':'spousal/survivor benefit from your record'}`:''}`,['socialSecurity.annualBenefitAt67',...(separateCouple?['spouseIncome.annualBenefitAt67']:[])],2],
    ['Pension or annuity',`You ${pension(s.guaranteedIncome.annualIncome,s.guaranteedIncome.startAge,s.guaranteedIncome.startAgeMonths)}${separateCouple?` · Spouse ${pension(s.spouseIncome.annualPension,s.spouseIncome.pensionStartAge,s.spouseIncome.pensionStartAgeMonths)}`:''}`,['guaranteedIncome.annualIncome',...(separateCouple?['spouseIncome.annualPension']:[])],2],
    ['Average returns and inflation',`Before retirement ${pct(s.market.preRetirementMeanReturn)} · stocks ${pct(s.market.stockMeanReturn)} · bonds ${pct(s.market.bondMeanReturn)} · inflation ${pct(s.spending.generalInflationMean)}`,['market.preRetirementMeanReturn','market.stockMeanReturn','market.bondMeanReturn','spending.generalInflationMean'],4]
  ];
  return `<section class="card"><h2>At a glance</h2><dl class="assumption-summary">${rows.map(([label,value,paths,step])=>`<div><dt>${escapeHTML(label)}</dt><dd>${escapeHTML(value)}<small>${escapeHTML(tag(paths))} · <button class="text-link" data-action="setup-section" data-index="${step}" data-review-edit="true">Edit</button></small></dd></div>`).join('')}</dl></section>`;
}
function unknownInputList(s,unknown){
  const labels=new Map(schema.flatMap(([,fields],index)=>fields.map(([label,path])=>[path,[label,setupStepForSchema(index)]])));
  return `<div class="notice error"><strong>Review Unknown inputs before running.</strong> No calculation will use an unknown as zero.<ul>${unknown.map(path=>{const [label,step]=labels.get(path)??[path,1];return `<li>${escapeHTML(label)} · <button class="text-link" data-action="setup-section" data-index="${step}" data-review-edit="true">Enter a number</button></li>`;}).join('')}</ul></div>`;
}
function conversionReview(s){
  const describe=lots=>lots.length?`${lots.length} ${lots.length===1?'entry':'entries'} · ${money(lots.reduce((n,lot)=>n+lot.amount,0),2)} remaining principal`:'None entered';
  return `<section class="card"><div class="section-heading"><h2>Past Roth conversions</h2><button class="secondary" data-action="setup-section" data-index="1" data-review-edit="true">Edit Past Roth conversions</button></div><dl class="assumption-summary"><div><dt>${s.household.separatePeople?'Yours':'Recorded conversions'}</dt><dd>${escapeHTML(describe(s.rothHistory.conversions))}</dd></div>${s.household.separatePeople&&s.household.filingStatus==='Married'?`<div><dt>Spouse</dt><dd>${escapeHTML(describe(s.spouseRothHistory.conversions))}</dd></div>`:''}</dl></section>`;
}
function setupReview(s){
  const unknown=unknownInputPaths(s,state.inputSources);
  const groups=schema.map(([title,fields],index)=>visibleFields(s,fields).length===0?'':`<section class="card"><div class="section-heading"><h2>${escapeHTML(title)}</h2><button class="secondary" data-action="setup-section" data-index="${setupStepForSchema(index)}" data-review-edit="true">Edit ${escapeHTML(title)}</button></div><dl class="assumption-summary">${visibleFields(s,fields).map(([label,path,type])=>{const value=path==='numberOfSimulations'?effectivePaths(s):getPath(s,path),source=path==='household.filingStatus'?'Selected':inputSource(s,state.inputSources,path);const owner=s.household.separatePeople?(path.startsWith('accounts.pretax')||path.startsWith('accounts.roth')?'You · ':path.startsWith('accounts.')?'Household · ':''):'';return `<div><dt>${escapeHTML(owner+label)}</dt><dd>${source==='Unknown'?'Unknown · number needed':escapeHTML(type==='money'?money(value,2):type==='percent'?pct(value):type==='date'?dateLabel(value):type==='select'?(value==='HeadOfHousehold'?'Head of household':value==='EmpiricalAgeDecline'?'Age-based spending decline':value):type==='checkbox'?(value?'On':'Off'):monthFields[path]?`${value} years ${getPath(s,monthFields[path])} extra months`:value)}<small>${escapeHTML(source)}</small></dd></div>`;}).join('')}</dl></section>`).join('');
  return `${pageHead('Review before running','All assumptions remain editable. Inputs are amounts today; modeled balances are future dollars.',`<button class="secondary" data-action="toggle-guided">${state.guided?'Skip guide · Detailed editor':'Back to detailed editor'}</button>`)}${setupProgress(5)}${notice()}${sampleInputNotice(s,false)}<h2 id="section-title" tabindex="-1">Your assumptions summary</h2><p class="form-note">${s.household.filingStatus==='Married'?s.household.separatePeople?'Couple: separate retirement dates, owned pre-tax and Roth accounts, each person’s own benefit and pension, and monthly savings until their retirement. Household costs start at the first retirement.':'Couple: one retirement date, combined balances, primary-owned retirement-account rules, one pension stream, and spousal/survivor Social Security based on your record.':'Individual: only your timeline and benefits are used.'} <a href="./methodology.html#household-model">Household model limits</a></p>${unknown.length?unknownInputList(s,unknown):''}<div class="review-actions"><button class="primary" data-action="run-plan" ${unknown.length||state.busy?'disabled':''}>Run ${effectivePaths()} simulated paths →</button><button class="secondary" data-action="setup-section" data-index="0">Edit household</button></div><div class="stack">${reviewGlance(s)}${earningsExplanation(s)}<details class="card review-all"><summary>All assumptions by section</summary><div class="stack">${groups}${conversionReview(s)}</div></details></div><p class="form-note">Simulation math and privacy are the same in guided setup and the detailed editor.</p>`;
}
function resultsExplanation(r){
  const n=r.provenance.simulationCount,successes=Math.round(r.successProbability*n),last=r.notFailedByAge.at(-1),alive=Math.round((last?.aliveShare??0)*n);
  return card('What these results mean',`<div class="result-explanations"><div><h3>Did savings cover the modeled lifetime?</h3><p><strong>${successes} of ${n} simulated paths</strong> had no portfolio shortfall through death or the modeling limit. A simulated death is not running out of money.</p></div><div><h3>Was someone still alive?</h3><p>The lifespan curve counts paths with at least one household member alive, out of the same ${n} paths. It includes people whose savings ran short. ${last?`At the last observation (your age ${ageLabel(last.age)}), ${alive} of ${n} paths had someone alive.`:''} Zero observed survivors does not mean living longer is impossible.</p></div><div><h3>What are the balances?</h3><p>All result balances are <strong>future dollars</strong>, not today’s purchasing power. Ending balances use different lifetime-end ages; age-based ranges use only observed balances and stop after death or shortfall.</p></div></div><h3>Why do these results differ?</h3><p>Monte Carlo paths vary markets, inflation and lifespans. The separate steady-growth illustration below fixes lifespans and removes volatility and long-term care. A large balance in that illustration can coexist with no survivors in this run; it is not a survival or funding probability.</p><div class="actions"><button class="text-link" data-action="setup-section" data-index="0" data-results-link>Review lifespan assumptions</button><button class="text-link" data-action="setup-section" data-index="1" data-results-link>Review inflation</button><button class="text-link" data-action="setup-section" data-index="4" data-results-link>Review returns</button><a href="./methodology.html#result-definitions">Result definitions</a></div>`);
}
function steadySimulationTable(r){
  const path=r.steadySimulation;
  if(!path)return '';
  const hasGap=path.monthlyDetails.some(p=>p.unfundedAmount>0);
  const lifespan=`Assumed lifespan: you to age ${ageLabel(path.primaryDeathAge)}${path.spouseDeathAge===null?'':`; spouse to age ${ageLabel(path.spouseDeathAge)}`}.`;
  const ending=path.endReason==='shortfall'?'Stops in the month funds run short.':'Runs until the end of the assumed household lifetime.';
  const s=r.uxAssumptions||current();
  const items=[
    `${lifespan} It follows these lifespans even if the plan’s maximum modeling age is lower, so it can show balances at ages no simulated path reached.`,
    `Returns every year: pre-retirement investments ${pct(s.market.preRetirementMeanReturn)}; stocks ${pct(s.market.stockMeanReturn)}; bonds ${pct(s.market.bondMeanReturn)}, mixed by the selected allocation; cash yield 2%. This one extra simulation runs with all volatility set to 0%, so there are no market downturns.`,
    `Inflation every year: general ${pct(s.spending.generalInflationMean)}; healthcare inflation ${pct(s.healthcare.healthcareInflationMean)}. Your pension growth ${pct(s.guaranteedIncome.annualIncrease)}.${s.household.separatePeople&&s.household.filingStatus==='Married'?` Spouse pension growth ${pct(s.spouseIncome.annualIncrease)}.`:''}`,
    'Other settings: long-term care risk turned off; taxes, Social Security, healthcare premiums, spending adjustments, withdrawal settings, savings deposits and take-home support still apply as selected.',
    'All balances below are future dollars, not today’s purchasing power.'
  ];
  const ownerNote=s.household.separatePeople?'<p class="form-note">Pre-tax and Roth columns show each owner; taxable investments and cash are shared. Deposits during an overlap continue only for the person still working.</p>':'';
  const assumptions=`<p><strong>An illustration, not a forecast.</strong> It is separate from the Monte Carlo results and is not a statistical median, typical Monte Carlo outcome or guaranteed balance. Because every year earns the average return with no downturns, its balances are often higher than the middle Monte Carlo path at the same age.</p><ul>${items.map(item=>`<li>${item}</li>`).join('')}</ul><button class="text-link" data-view="withdrawals">How withdrawals work →</button>`;
  if(path.endReason==='before-retirement')return card('Steady-growth illustration · monthly balances',`<div class="steady-explanation">${assumptions}</div><p class="form-note">Retirement begins on or after the end of the assumed household lifetime, so there are no retirement months to show.</p>`);
  const columns=[...(s.household.separatePeople?[['You · Pre-tax',p=>p.ownerAccounts?.[0]?.pretax],['You · Roth',p=>p.ownerAccounts?.[0]?.roth],...(s.household.filingStatus==='Married'?[['Spouse · Pre-tax',p=>p.ownerAccounts?.[1]?.pretax],['Spouse · Roth',p=>p.ownerAccounts?.[1]?.roth]]:[])]:[['Pre-tax','pretax'],['Roth','roth']]),['Taxable','taxable'],['Cash','cash'],['Home value','home'],['Mortgage debt','mortgage'],['Portfolio total','portfolio'],['Net assets','netAssets'],...(hasGap?[['Unfunded amount','unfundedAmount']]:[])];
  return card('Steady-growth illustration · monthly balances',`<div class="steady-explanation" id="steady-monthly-note">${assumptions}</div>${ownerNote}<p class="form-note">The first row is the retirement starting balance after pre-retirement growth, selected savings deposits and mortgage payments. Later rows show balances after each modeled month, including returns, cash flows, conversions, and home sales. ${ending} Net assets = portfolio + home value − mortgage debt.${hasGap?' The unfunded amount is the final unmet cost, separate from mortgage debt.':''}</p><div class="table-wrap monthly-balances" role="region" aria-label="Steady-growth simulation monthly balances" aria-describedby="steady-monthly-note" tabindex="0"><table><thead><tr><th scope="col">Month</th><th scope="col">Age</th>${columns.map(([label])=>`<th scope="col">${label}</th>`).join('')}</tr></thead><tbody>${path.monthlyDetails.map(p=>`<tr><th scope="row">${p.month===0?'Retirement start':`Month ${p.month}`}${p.date?`<small>${dateLabel(p.date)}</small>`:''}</th><td>${ageLabel(p.age)}</td>${columns.map(([,key])=>`<td>${money(typeof key==='function'?key(p):p[key],2)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
}
function lab(){const completed=state.labResults?.find(row=>row.result)?.result,comparisonCount=completed?.provenance.simulationCount??Math.min(effectivePaths(),150);return `${pageHead('Monte Carlo scenario lab','Change one assumption at a time, then compare modeled lifetimes against the same starting plan.',`<button class="primary" data-action="run-lab" ${state.busy?'disabled':''}>Run comparisons</button><button class="secondary" data-action="run-decision" ${state.busy||!isPro()?'disabled':''}>Find age & spending targets${isPro()?'':' · Pro'}</button>`)}${notice()}${planNotice()}${state.decision?card('Planning targets',`<div class="grid two"><div><span class="metric-label">Your earliest retirement age at ${pct(state.decision.targetReadiness)} readiness</span><div class="metric-value">${state.decision.earliestRetirementAge??'No age found'}</div><p class="form-note">${escapeHTML(ageSearchNote(state.decision))}${current().household.separatePeople&&current().household.filingStatus==='Married'?' Spouse retirement date remains fixed.':''}</p></div><div><span class="metric-label">Modeled annual spending at ${pct(state.decision.targetReadiness)} readiness</span><div class="metric-value">${state.decision.safeAnnualSpending===null?'No amount found':(state.decision.safeSpendingAtSearchLimit?'At least ':'')+money(state.decision.safeAnnualSpending)}</div><p class="form-note">${state.decision.safeSpendingAtSearchLimit?`The search stops at ${money(state.decision.safeSpendingSearchLimit)}; higher spending was not tested. `:''}Rounded down to $500; rerun the full plan before making decisions.</p></div></div>`):''}${state.labResults?card('Scenario comparison',`${previewNotice(completed)}<div class="comparison-row"><strong>Scenario</strong><strong>${comparisonCount<=FREE_SIMULATION_PATHS?'Samples without shortfall':'Modeled readiness'}</strong><strong>Median ending · future dollars</strong></div>${state.labResults.map(r=>`<div class="comparison-row"><div><strong>${escapeHTML(r.label)}</strong>${r.error||r.note?`<div class="muted">${escapeHTML(r.error||r.note)}</div>`:''}</div><strong>${r.result?readinessLabel(r.result):'—'}</strong><strong>${r.result?money(r.result.medianEndingBalance):'—'}</strong></div>`).join('')}<p class="form-note">Each comparison runs ${comparisonCount} Monte Carlo paths with the fixed comparison sequence for reproducible screening. Run the full plan for final results.</p>`):card('Quick comparisons',`<p class="form-note">The lab tests retirement timing, spending, claiming age, healthcare costs, Roth conversions, and cash use.</p><ul class="warning-list">${labVariants.map(x=>`<li>${escapeHTML(x[0])}</li>`).join('')}</ul>`)}`;}
function withdrawals(){return `${pageHead('How withdrawals work','See how this simulator uses your income and savings to pay retirement costs.')}${notice()}${withdrawalContent(current(),result(),state.withdrawalMonth??1)}`;}
function results(){const r=result();return `${pageHead('Monte Carlo simulation results','Explore the range of outcomes across many modeled lifetimes.')}${notice()}${planNotice()}${r?'':monteRunSummary(effectivePaths(),false)}${!r?(state.busy?'':'<div class="notice">Run the selected scenario to view its results.</div>'):`<div class="stack">${resultHero(r)}${resultsExplanation(r)}<div class="split">${chartCard('survival')}${card('Next useful test',`<p class="next-test">${escapeHTML(r.riskBreakdown.recommendedNextTest)}</p>${sensitivityNote(r)}<button class="secondary" data-view="lab">Explore in scenario lab →</button>${readinessUpgrade(r)}`)}</div>${chartCard('paths')}${chartCard('bands')}${card('Savings coverage and household lifespan by age',`<p class="form-note">Both columns use all ${r.provenance.simulationCount} paths as their denominator. No shortfall observed includes completed lives with savings left. Household still alive means at least one person; it is independent of financial shortfalls. Each whole-year age shows its last modeled observation.</p><div class="table-wrap"><table><thead><tr><th>Age</th><th>${'No shortfall observed'}</th><th>${'Household still alive'}</th></tr></thead><tbody>${ageYearRows(r.notFailedByAge).map(p=>`<tr><td>${p.ageYear}</td><td>${shareLabel(p.notFailedShare,r.provenance.simulationCount)}</td><td>${shareLabel(p.aliveShare,r.provenance.simulationCount)}</td></tr>`).join('')}</tbody></table></div>`)}${card('Balance bands by age',`<p class="form-note">Each whole-year age shows its last modeled observation. Only paths observed then contribute. Future dollars, not today’s purchasing power. A shortfall is recorded as $0 and the path stops; completed lifetimes are not extended. Later rows use fewer paths, so these ranges do not measure overall readiness. When there are no surviving or observed paths, amounts show “Not enough simulated outcomes” rather than $0. If the last lifespan observation in that year has no survivors, an earlier balance from that year is not carried forward. A row with one or two observations is also a very small sample.</p><div class="table-wrap"><table><thead><tr><th>Age</th><th>Paths at this age</th><th>10th percentile</th><th>Median</th><th>90th percentile</th></tr></thead><tbody>${balanceDisplayRows(r).map(b=>`<tr><td>${b.ageYear}</td><td>${b.pathCount}</td><td>${b.noOutcomes?'Not enough simulated outcomes':money(b.pessimistic)}</td><td>${b.noOutcomes?'—':money(b.median)}</td><td>${b.noOutcomes?'—':money(b.optimistic)}</td></tr>`).join('')}</tbody></table></div>`)}${steadySimulationTable(r)}${card('Failure ages',r.failureAgeBuckets.length?`<div class="info-list">${r.failureAgeBuckets.map(b=>info(`Ages ${b.label}`,isPreviewResult(r)?`${b.count} sample paths`:`${b.count} paths · ${pct(b.shareOfFailures)} of failures`)).join('')}</div>`:'<p class="muted">No simulated paths ran out of funds.</p>')}${planningDisclosure()}</div>`}`;}
function reportText(s,r){const count=r?.provenance.simulationCount??effectivePaths(s),precisePct=v=>`${Number((v*100).toPrecision(12))}%`;const lines=[`RETIREMENT FORECAST - MONTE CARLO SIMULATOR — ${s.name}`,`Generated ${new Date().toLocaleString()}`,`Calculation engine: ${r?.provenance.engineVersion||scenarioEngineVersion(s)}`,`Simulation paths: ${count}; fixed comparison sequence`,...(r?[`Paths for next run: ${effectivePaths(s)}`]:[]),'','U.S. MODEL SCOPE: U.S. federal tax, Social Security, and Medicare assumptions only. State and local income taxes and laws outside the U.S. are not modeled.',`Birthday: ${dateLabel(s.household.birthday)}; retirement date: ${dateLabel(s.household.retirementDate)}`,`Current age: ${Math.floor(scenarioTimeline(s).currentAge)}; your retirement age: ${ageLabel(primaryRetirementAge(s))}; household start age: ${ageLabel(retirementAge(s))}`,`Filing status: ${s.household.filingStatus}`,`Starting pre-tax: ${money(s.accounts.pretax,2)}; Roth: ${money(s.accounts.roth,2)}; taxable: ${money(s.accounts.taxable,2)}; cash: ${money(s.accounts.cash,2)}`,`Annual base spending: ${money(s.spending.annualBaseSpending,2)}`,`General inflation: ${precisePct(s.spending.generalInflationMean)} ± ${precisePct(s.spending.generalInflationStdDev)}`,`Social Security benefit at 67: ${money(s.socialSecurity.annualBenefitAt67,2)}; claim age: ${s.socialSecurity.claimAge}`,`Pre-Medicare monthly premium: ${money(s.healthcare.preMedicareMonthlyPremium,2)}`,`Long-term care: ${s.longTermCare.enabled?'included':'excluded'}`,`Roth conversions: ${s.rothConversion.enabled?'enabled':'disabled'}`,''];lines.push(`Household model: ${s.household.separatePeople?'Separate people':'Legacy pooled accounts'}`,`Spouse retirement date: ${s.household.separatePeople&&s.household.filingStatus==='Married'?dateLabel(s.household.spouseRetirementDate):'Shared / not applicable'}`,'INPUT AMOUNTS: Today’s dollars / purchasing power. RESULT BALANCES: Future dollars; not adjusted back to today.','EARNINGS: '+EARNINGS_EXPLANATION,'ALL ASSUMPTIONS');for(const [section,fields] of schema){lines.push(section+':');for(const [label,path,type] of visibleFields(s,fields)){const value=path==='numberOfSimulations'?count:getPath(s,path);lines.push(`  ${label}: ${inputSource(s,state.inputSources,path)==='Unknown'?'Unknown; prior value retained for editing':type==='percent'?precisePct(value):type==='money'?money(value,2):type==='checkbox'?(value?'Yes':'No'):value}`);lines.push(`  Value source: ${inputSource(s,state.inputSources,path)}`);if(monthFields[path])lines.push(`  ${label} extra months: ${getPath(s,monthFields[path])}`);}}lines.push('PAST ROTH CONVERSIONS',...(s.rothHistory.conversions.length?s.rothHistory.conversions.map(lot=>`Tax year ${lot.taxYear}: remaining principal ${money(lot.amount,2)}; remaining taxable principal ${money(lot.taxableAmount,2)}`):['None entered.']),`Roth history needs review: ${s.rothHistory.needsReview?'Yes - migrated assumptions; verify contribution basis, first funding year, and past conversions.':'No'}`,...(s.withdrawalStrategy.ruleOf55Eligible&&!ruleOf55Applies(s)?['Rule of 55: not applied because retirement is before the calendar year you turn 55.']:[]),'');lines.push(`Budget months: ${s.budget.monthlyBudgets.length}`,`Property taxes: ${money(s.budget.annualPropertyTaxes,2)}; home insurance: ${money(s.budget.annualHomeInsurance,2)}; auto insurance: ${money(s.budget.annualAutoInsurance,2)}`,'');const bd=budgetBreakdown(s.budget);lines.push('BUDGET WORKSHEET',`Months used: ${bd.months.map(m=>m.month).join(', ')||'none'}`,`Monthly average after adjustments: ${money(bd.monthlyAverage,2)}`,`Annual bills added once: ${money(bd.annualBills,2)}`,`Annual retirement adjustment: ${money(bd.retirementAdjustment,2)}`,`Draft annual estimate: ${money(bd.estimate,2)}`,`Applied and current: ${s.budget.isAppliedToAnnualBaseSpending&&!s.budget.estimateNeedsReview?'Yes':'No'}`,'');if(r)lines.push(`Lifetimes without a portfolio shortfall: ${readinessLabel(r)}`, ...(isPreviewResult(r)?[`SAMPLE PREVIEW ONLY: ${PREVIEW_WARNING}`]:[]),`Median ending balance (future dollars): ${money(r.medianEndingBalance)}`,`10th / 90th percentile of ending balances: ${money(r.pessimisticEndingBalance)} / ${money(r.optimisticEndingBalance)}`,`Ending balances are measured at death, the modeling age limit, or shortfall; failed endings count as $0. Paths reaching the limit can still be alive.`, `Balance bands use observed paths only, through age ${ageLabel(r.balanceBands.at(-1)?.age??retirementAge(s))}.`,`Median failure age: ${r.medianFailureAge===null?'none':ageLabel(r.medianFailureAge)}`,`Most helpful sensitivity check: ${r.riskBreakdown.primaryRisk}`, ...(r.riskBreakdown.summary?[r.riskBreakdown.summary]:[]),`Next useful test: ${r.riskBreakdown.recommendedNextTest}`);else lines.push('Run the scenario to include results.');if(s.household.separatePeople&&s.household.filingStatus==='Married')lines.push('SPOUSE PAST ROTH CONVERSIONS',...s.spouseRothHistory.conversions.map(l=>`Tax year ${l.taxYear}: principal ${money(l.amount,2)}; taxable principal ${money(l.taxableAmount,2)}`));lines.push('','RESULT LIMITATIONS: Hypothetical results depend on your inputs and model assumptions. Modeled readiness is not the probability your actual plan will succeed. Results are not predictions or guarantees and do not capture every cost or event.','USE: Educational estimate only, not individualized financial, investment, tax, legal, or insurance advice. Verify inputs and consult qualified professionals before acting.','DATA: Scenarios are stored locally in this browser; safeguard any exported file.');return lines.join('\n');}
function billingView(){
  const a=state.access,auth=state.auth;
  const chatgptLink='<a class="secondary" href="/signin-with-chatgpt?return_to=%2F%3Faccount%3D1">Continue with ChatGPT</a>';
  const socialButtons=auth.configured&&auth.enabledProviders.google?'<button class="secondary" data-action="social-signin" data-provider="google">Continue with Google</button>':'';
  let account;
  if(!a.signedIn){
    account=`<p>Sign in to subscribe or restore Pro access. Your scenarios remain on this device.</p><div class="actions account-actions">${chatgptLink}${socialButtons}</div>`;
  }else{
    const method=a.accountProvider==='google'?'Google':'ChatGPT';
    account=`<p><strong>Signed in with ${method}.</strong> Pro access follows this account.</p>`;
    if(a.ownerAccess){
      account+='<p class="form-note">Owner Pro access is active for your known ChatGPT account and verified Google account. Linking is not required.</p>';
      if(a.billingLookupUnavailable)account+='<p class="form-note">Billing lookup is temporarily unavailable. Owner Pro remains active; Manage billing can retry the lookup.</p>';
    }else if(auth.configured&&auth.chatgptSignedIn&&auth.signedIn){
      account+='<p class="form-note">Connect these two verified sign-ins to use one subscription with either method.</p><div class="actions account-actions"><button class="secondary" data-action="social-link-accounts">I agree to link these accounts</button></div>';
    }else if(auth.configured&&auth.chatgptSignedIn){
      account+=`<p class="form-note">You can connect a Google sign-in to this ChatGPT account and its subscription.</p><div class="actions account-actions">${auth.enabledProviders.google?'<button class="secondary" data-action="social-link-from-chatgpt" data-provider="google">Link Google</button>':''}</div>`;
    }else if(auth.configured&&auth.signedIn){
      const providers=new Set(auth.linkedProviders);
      account+='<div class="actions account-actions">'+(providers.has('google.com')||!auth.enabledProviders.google?'':'<button class="secondary" data-action="social-link-provider" data-provider="google">Link Google</button>')+'<a class="secondary" href="/signin-with-chatgpt?return_to=%2F%3Flink%3D1">Connect ChatGPT account</a></div>';
    }
    const signOutActions=[];
    if(auth.signedIn)signOutActions.push('<button class="subtle" data-action="social-signout">Sign out of Google</button>');
    if(auth.chatgptSignedIn)signOutActions.push('<a class="subtle" href="/signout-with-chatgpt?return_to=%2F%3Faccount%3D1">Sign out of ChatGPT</a>');
    if(signOutActions.length)account+=`<div class="actions account-actions">${signOutActions.join('')}</div>`;
  }
  const choices=`<div class="billing-options"><div class="billing-option"><strong>$9.99 <small>/ month</small></strong>${a.checkoutAvailable&&a.signedIn?'<button class="primary" data-action="checkout" data-interval="monthly">Choose monthly</button>':''}</div><div class="billing-option"><strong>$79 <small>/ year</small></strong>${a.checkoutAvailable&&a.signedIn?'<button class="primary" data-action="checkout" data-interval="yearly">Choose yearly</button>':''}</div></div>`;
  const portalAction=a.billingPortalAvailable?'<div class="actions"><button class="secondary" data-action="billing-portal">Manage billing</button></div>':'';
  const ownerTestCheckout=a.ownerAccess&&a.testBilling&&a.checkoutAvailable?`<p class="form-note">Stripe test mode: these checkouts do not charge real money. Owner Pro remains active.</p>${choices}`:'';
  const proActions=isPro()?`<span class="tag">${a.ownerAccess?'Owner access':'Active on this account'}</span><div class="field billing-paths"><label for="billing-path-count">Paths for the next full run</label><input id="billing-path-count" type="number" inputmode="numeric" min="${MIN_SIMULATION_PATHS}" max="${MAX_SIMULATION_PATHS}" step="1" data-field="numberOfSimulations" data-type="number" value="${escapeHTML(current().numberOfSimulations)}"></div>${portalAction}${ownerTestCheckout}`:`${portalAction}${choices}${!a.checkoutAvailable?'<p class="form-note">Paid access is not configured yet. Stripe confirms the final price before payment.</p>':''}`;
  return `${pageHead('Plans & billing','Choose the number of Monte Carlo paths for your plan.')}${notice()}${card('Your account',account,'account-card')}<div class="grid two">${isPro()?'':card('Free preview',`<div class="metric-value">10 paths</div><p>A quick look at possible lifetimes. The preview uses only 10 simulations.</p><p>Your scenarios and calculations stay on this device.</p>`)}${card('Pro',`<div class="metric-value">${isPro()?'10,000 paths by default':'Up to 10,000 paths'}</div><p>${isPro()?'Adjust the path count for a full run or use planning targets.':'Choose a higher path count for full runs and comparisons, and use planning targets.'} Calculations still run on your device.</p>${proActions}`)}</div><p class="billing-footnote">Subscriptions renew automatically until canceled. Manage cancellation in Plans &amp; billing → Manage billing; Stripe shows the effective date. <a href="./terms.html#subscriptions">Subscription terms</a> · <a href="./support.html#refunds">Refund requests</a> · <a href="./privacy.html">Privacy</a> · <a href="./support.html">Contact support</a></p><p class="billing-footnote">Subscriptions are linked to the account used to sign in. Link accounts explicitly to share a subscription. Stripe handles payment details; this Site does not receive card numbers or your retirement scenarios. Browser-side calculation limits can be bypassed by changing local code.</p>`;
}
function reports(){const text=reportText(current(),result());return `${pageHead('Monte Carlo reports & backup','Export the current plan, its simulation assumptions, and its results.',`<button class="secondary" data-action="print">Print / save PDF</button>`)}${notice()}<div class="grid two">${card('Current plan report',`<div class="actions"><button class="secondary" data-action="download-report">Download text</button><button class="secondary" data-action="copy-report">Copy report</button></div><pre class="report-text">${escapeHTML(text)}</pre>`)}${card('Scenario backup',`<p>Save all plans in one JSON file or restore a previous backup.</p><div class="actions"><button class="primary" data-action="export-backup">Export JSON</button><button class="secondary" data-action="import-backup">Import JSON</button></div><p class="form-note">Imports web backups or Android scenario JSON arrays. Import replaces the plans saved in this browser, so export a backup first if you want to keep them.</p>`)}</div>`;}
function render({preserveEditor=false}={}){
  const editor=preserveEditor?document.activeElement:null;
  const keepEditor=['INPUT','SELECT'].includes(editor?.tagName)&&editor.id&&(editor.dataset.field||editor.dataset.inputSource||editor.dataset.growthHelper||editor.dataset.budget||editor.dataset.budgetAdjustment||editor.dataset.month!==undefined);
  rememberBudgetDisclosures();
  disposeCharts();
  const details=$('#advanced-model');if(details)state.advancedOpen=details.open;
  const allocation=$('#allocation-settings');if(allocation)state.allocationOpen=allocation.open;
  const rothDetails=$('#roth-history');if(typeof rothDetails?.open==='boolean')state.rothHistoryOpen=rothDetails.open;
  const growth=$('#growth-helper');if(typeof growth?.open==='boolean'&&growthHelpers.has(growth.dataset.scenarioId))growthHelpers.get(growth.dataset.scenarioId).open=growth.open;
  const select=$('#scenario-select');
  select.innerHTML=state.scenarios.map(s=>`<option value="${escapeHTML(s.id)}" ${s.id===state.selectedId?'selected':''}>${escapeHTML(s.name)}</option>`).join('');
  $('#run-button').disabled=state.busy;$('#run-button').textContent=state.busy?'Calculating…':'Run simulation';
  document.querySelectorAll('#navigation button').forEach(b=>{const active=b.dataset.view===state.view;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  $('#page-location').textContent=currentViewLabel();
  $('#result-state').textContent=state.busy?busyStatus():result()?'Results up to date':unknownInputPaths(current(),state.inputSources).length?'Review Unknown inputs':'Ready to simulate';
  $('#main').innerHTML=({dashboard,setup,budget,withdrawals,scenarios,lab,results,reports,billing:billingView}[state.view]||dashboard)();
  // Reuse the live input so its draft, caret and pending change event survive
  // a background render. Explicit navigation/reset uses fresh inputs.
  if(keepEditor){
    const replacement=$('#'+editor.id);
    if(replacement&&replacement.type===editor.type){
      replacement.replaceWith(editor);
      for(let parent=editor.parentElement;parent;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;
      editor.focus({preventScroll:true});
    }
  }
  mountCharts($('#main'),result(),retirementAge(current()));
}
class CalculationCanceled extends Error{}
let cancelCalculation=null;
function runWorker(s,task='simulation',onProgress=null){return new Promise((resolve,reject)=>{
  const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
  const finish=()=>{worker.terminate();if(cancelCalculation===cancel)cancelCalculation=null;};
  const cancel=()=>{finish();reject(new CalculationCanceled('Calculation canceled.'));};
  cancelCalculation=cancel;
  worker.onmessage=e=>{if(e.data.type==='progress'){onProgress?.(e.data);return;}finish();e.data.type==='result'?resolve(e.data.result):reject(new Error(e.data.message));};
  worker.onerror=e=>{finish();reject(new Error((e.message||'Calculation failed')+'. Reload this page to load the latest calculator, then try again.'));};
  worker.postMessage({scenario:s,task});
});}
function startCalculation(message){state.busy=true;state.busyMessage=message;state.progress={fraction:0,detail:''};state.message=message;render({preserveEditor:true});}
function finishCalculation(){state.busy=false;state.progress=null;render({preserveEditor:true});}
function busyStatus(){const fraction=state.progress?.fraction;return fraction==null?'Simulation running':`Simulation running · ${Math.floor(fraction*100)}%`;}
// Progress updates touch only the progress controls, so editors and charts stay put.
function setProgress(fraction,detail){
  if(!state.busy)return;
  state.progress={fraction,detail};
  const bar=$('#busy-progress'),text=$('#busy-detail');
  if(bar&&fraction!=null)bar.value=fraction;
  if(text)text.textContent=detail;
  $('#result-state').textContent=busyStatus();
}
const countLabel=n=>Number(n).toLocaleString('en-US');
async function run(){
  if(state.busy)return;
  const s=simulationScenario(),unknown=unknownInputPaths(s,state.inputSources),errors=[...validateScenario(s),...(unknown.length?['Review Unknown inputs before running. Enter a number (0 means none), or explicitly use an estimate or sample value.']:[])];
  if(errors.length){state.message='Error: '+errors.join(' ');render({preserveEditor:true});return;}
  const revision=calculationRevision,total=s.numberOfSimulations;
  startCalculation('Calculating this plan…');
  try{const r=await runWorker(s,'simulation',p=>setProgress(p.fraction,p.fraction>=1?'Summarizing results and sensitivity checks…':`${countLabel(Math.round(p.fraction*total))} of ${countLabel(total)} lifetimes simulated`));if(revision!==calculationRevision)return;r.uxAssumptions=deep(s);state.results.set(s.id,r);state.message='Results updated for '+s.name+'.';}
  catch(e){if(revision===calculationRevision)state.message=e instanceof CalculationCanceled?CALCULATION_CANCELED:'Error: '+e.message;}
  finally{finishCalculation();}
}
// Each phase stops at its first qualifying candidate, so the totals are upper bounds.
function decisionProgress(p){const total=p.totalAges+p.totalAmounts;return total?(p.phase==='ages'?p.checkedAges:p.totalAges+p.checkedAmounts)/total:null;}
function decisionDetail(p){return p.phase==='ages'?`Checked ${p.checkedAges} of up to ${p.totalAges} retirement ages`:`Checked ${countLabel(p.checkedAmounts)} of up to ${countLabel(p.totalAmounts)} spending amounts`;}
async function runDecision(){
  if(state.busy)return;
  if(!isPro()){setMessage(TARGETS_NEED_PRO);return;}
  if(unknownInputPaths(current(),state.inputSources).length){setMessage('Error: Review Unknown inputs before running.');return;}
  const revision=calculationRevision,s=simulationScenario();
  startCalculation('Finding retirement-age and spending targets…');
  try{const decision=await runWorker(s,'decision',p=>setProgress(decisionProgress(p),decisionDetail(p)));if(revision!==calculationRevision)return;state.decision=decision;state.message='Planning targets ready.';}
  catch(e){if(revision===calculationRevision)state.message=e instanceof CalculationCanceled?TARGETS_CANCELED:'Error: '+e.message;}
  finally{finishCalculation();}
}
async function runLab(){
  if(state.busy)return;
  if(unknownInputPaths(current(),state.inputSources).length){setMessage('Error: Review Unknown inputs before running.');return;}
  const revision=calculationRevision,base=simulationScenario(),rows=[['Current plan',()=>{}],...labVariants];
  base.numberOfSimulations=Math.min(150,base.numberOfSimulations);
  const baseline=JSON.stringify(base);
  state.labResults=[];startCalculation('Running scenario comparisons…');
  try{
    for(const [index,[label,change]] of rows.entries()){
      const s=deep(base);change(s);const errors=validateScenario(s);let row;
      // A comparison identical to the plan would only repeat the first row.
      if(index&&JSON.stringify(s)===baseline)row={label,result:null,note:'Already matches the current plan.'};
      else try{row={label,result:errors.length?null:await runWorker(s,'simulation',p=>setProgress((index+p.fraction)/rows.length,`${label} · comparison ${index+1} of ${rows.length}`)),error:errors.join(' ')};}
      catch(e){if(e instanceof CalculationCanceled)throw e;row={label,result:null,error:e.message};}
      if(revision!==calculationRevision)return;
      state.labResults.push(row);setProgress((index+1)/rows.length,`${index+1} of ${rows.length} comparisons complete`);render({preserveEditor:true});
    }
    state.message='Comparisons ready.';
  }catch(e){if(!(e instanceof CalculationCanceled))throw e;if(revision===calculationRevision)state.message=COMPARISONS_CANCELED;}
  finally{finishCalculation();}
}
function download(name,text,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function closeHelp(except=null){for(const help of document.querySelectorAll('.help.open')){if(help===except)continue;help.classList.remove('open');help.querySelector('.help-trigger').setAttribute('aria-expanded','false');}}
document.addEventListener('click',e=>{const trigger=e.target instanceof Element?e.target.closest('.help-trigger'):null,help=trigger?.closest('.help'),wasOpen=help?.classList.contains('open');closeHelp();if(trigger&&!wasOpen){help.classList.add('open');trigger.setAttribute('aria-expanded','true');}});
// Replacing the SVG with a copy restarts its CSS animations for the next loop.
document.addEventListener('animationend',e=>{if(e.animationName!=='fan-cycle')return;const figure=e.target.closest('.forecast-illustration'),svg=figure?.querySelector('svg');if(!svg)return;illustrationStarted=Date.now();figure.style.setProperty('--elapsed','0ms');svg.replaceWith(svg.cloneNode(true));});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeHelp();if(document.activeElement?.classList.contains('help-trigger'))document.activeElement.blur();}});
$('#run-button').addEventListener('click',()=>{if(state.busy)return;if(state.guided&&state.setupSection<5){state.view='setup';state.setupSection=5;render();window.scrollTo(0,0);return;}state.view='results';window.scrollTo(0,0);run();});
$('#scenario-select').addEventListener('change',async e=>{selectScenario(e.target.value);await persist();render();});
$('#menu-toggle').addEventListener('click',()=>{const open=$('#menu-toggle').getAttribute('aria-expanded')!=='true';$('#menu-toggle').setAttribute('aria-expanded',String(open));$('.sidebar').classList.toggle('menu-open',open);});
$('#navigation').addEventListener('click',e=>{const button=e.target.closest('[data-view]');if(!button)return;state.view=button.dataset.view;$('#menu-toggle').setAttribute('aria-expanded','false');$('.sidebar').classList.remove('menu-open');state.message='';render();window.scrollTo(0,0);});
$('#main').addEventListener('click',async e=>{const runPlan=e.target.closest('[data-action=run-plan]');if(runPlan){if(state.guided&&state.setupSection<5){state.view='setup';state.setupSection=5;render();window.scrollTo(0,0);return;}state.view='results';render();window.scrollTo(0,0);await run();return;}const nav=e.target.closest('[data-view]');if(nav){state.view=nav.dataset.view;render();window.scrollTo(0,0);return;}const el=e.target.closest('[data-action]');if(!el)return;const a=el.dataset.action,s=current();
  if(a==='cancel-calculation'){cancelCalculation?.();return;}
  if(a==='export-unreadable-backup'){if(savedStoredRaw!==null)download('retirement-unreadable-backup.json',savedStoredRaw,'application/json');return;}
  if(a==='replace-unreadable-plans'){if(savedLoadError&&confirm('Replace the unreadable saved plans with the plans currently shown? Export the unreadable backup first if you want to keep it.')){if(await persist(true,true))setMessage('Replacement plans saved.');else render();}return;}
  if(a==='start-plan'){state.guided=true;state.setupSection=0;state.view='setup';await persist();render();window.scrollTo(0,0);return;}
  if(a.startsWith('social-')){
    el.disabled=true;
    try{
      if(a==='social-signin')await signInSocial(el.dataset.provider);
      else if(a==='social-link-from-chatgpt'){await signInSocial(el.dataset.provider);await linkAccounts();}
      else if(a==='social-link-provider')await linkSocialProvider(el.dataset.provider);
      else if(a==='social-link-accounts')await linkAccounts();
      else if(a==='social-signout')await signOutSocial();
      syncAuthState();
      const verified=await loadAccess({force:true});
      if(a.includes('link')&&verified)setMessage('Sign-in methods linked to the same subscription.',{preserveEditor:true});
      else if(a==='social-signout'&&verified)setMessage('Signed out of Google.',{preserveEditor:true});
    }catch(error){setMessage('Error: '+(error?.message||'Sign-in failed. Please try again.'),{preserveEditor:true});}
    finally{el.disabled=false;}
    return;
  }
  if(a==='withdrawal-settings'){state.view='setup';state.setupSection=Number(el.dataset.index);render();$('#section-title').focus();window.scrollTo(0,0);return;}
  if(a==='enable-people'){s.household.separatePeople=true;s.household.spouseRetirementDate ||= s.household.retirementDate;state.inputSources[s.id]['household.spouseRetirementDate']='Estimated';if(s.household.filingStatus==='Married')for(const path of ['spouseAccounts.pretax','spouseAccounts.roth','spouseIncome.annualBenefitAt67','spouseIncome.annualPension'])state.inputSources[s.id][path]='Unknown';state.results.delete(s.id);invalidateExploration();await persist();render();return;}
  if(a==='add-spouse-conversion'){s.spouseRothHistory.conversions.push({taxYear:2026,amount:0,taxableAmount:0});state.results.delete(s.id);invalidateExploration();await persist();render();return;}
  if(a==='remove-spouse-conversion'){s.spouseRothHistory.conversions.splice(Number(el.dataset.index),1);state.results.delete(s.id);invalidateExploration();await persist();render();return;}
  if(a==='toggle-guided'){state.guided=!state.guided;if(state.setupSection===5)state.setupSection=0;render();$('#section-title')?.focus();return;}
  if(a==='household-choice'){if(el.dataset.kind==='couple')s.household.filingStatus='Married';else if(s.household.filingStatus==='Married')s.household.filingStatus='Single';state.inputSources[s.id]['household.filingStatus']='Entered';state.results.delete(s.id);invalidateExploration();await persist();render();$('#section-title')?.focus();return;}
  if(a==='setup-section'){if(el.dataset.reviewEdit){state.advancedOpen=true;state.allocationOpen=true;state.rothHistoryOpen=true;}if(el.dataset.resultsLink!==undefined)state.view='setup';state.setupSection=Number(el.dataset.index);render();$('#section-title').focus();window.scrollTo(0,0);return;}
  if(a==='review-calendar-dates'){s.household.datesNeedReview=false;await persist();setMessage('Calendar dates reviewed.');return;}
  if(a==='reset-assumptions'){if(!confirm('Restore sample assumptions for this scenario?'))return;invalidateExploration();const fresh=prepareCalendarScenario(baseScenario(),{needsReview:false});fresh.household.separatePeople=s.household.separatePeople;fresh.household.spouseRetirementDate=s.household.separatePeople?fresh.household.retirementDate:'';if(isPro())applyProSimulationDefault(fresh);fresh.id=s.id;fresh.name=s.name;state.scenarios[state.scenarios.indexOf(s)]=fresh;budgetViews.delete(s.id);state.inputSources[s.id]={_origin:'Sample/default'};state.results.delete(s.id);await persist();setMessage('Sample assumptions restored.');}
  if(a==='new-scenario'){const copy=deep(s);copy.id='plan-'+Date.now();copy.name=s.name+' copy';state.scenarios.push(copy);state.inputSources[copy.id]=deep(state.inputSources[s.id]);selectScenario(copy.id);state.view='setup';await persist();setMessage('Scenario copied. Adjust its assumptions.');}
  if(a==='select-scenario'){selectScenario(el.dataset.id);state.view='dashboard';await persist();render();}
  if(a==='rename-scenario'){const target=state.scenarios.find(x=>x.id===el.dataset.id),name=prompt('Scenario name',target.name);if(name?.trim()){target.name=name.trim();await persist();render();}}
  if(a==='delete-scenario'){if(state.scenarios.length===1||!confirm('Delete this scenario?'))return;invalidateExploration();state.scenarios=state.scenarios.filter(x=>x.id!==el.dataset.id);state.results.delete(el.dataset.id);if(state.selectedId===el.dataset.id)state.selectedId=state.scenarios[0].id;await persist();render();}
  if(a==='add-month'){
    const ui=budgetView();if(ui.pending)return;
    ui.pending=blankBudgetMonth(s.budget);ui.open['month-'+s.budget.monthlyBudgets.length]=true;
    render();$('#month-'+s.budget.monthlyBudgets.length+'-credit')?.focus();return;
  }
  if(a==='finish-month'){
    const i=Number(el.dataset.index);if(!s.budget.monthlyBudgets[i])return;
    const details=$('#budget-month-'+i);details.open=false;budgetView().open['month-'+i]=false;
    details.querySelector('summary')?.focus();return;
  }
  if(a==='add-roth-conversion'||a==='remove-roth-conversion'||a==='review-roth-history'){
    if(a==='add-roth-conversion'){s.rothHistory.conversions.push({taxYear:2026,amount:0,taxableAmount:0});state.rothHistoryOpen=true;}
    if(a==='remove-roth-conversion')s.rothHistory.conversions.splice(Number(el.dataset.index),1);
    if(a==='review-roth-history')s.rothHistory.needsReview=false;
    state.results.delete(s.id);invalidateExploration();await persist();render();return;
  }
  if(a==='remove-month'){
    const i=Number(el.dataset.index),ui=budgetView();
    if(i===s.budget.monthlyBudgets.length&&ui.pending){ui.pending=null;render();return;}
    if(!s.budget.monthlyBudgets[i])return;
    markBudgetEdited(s.budget);s.budget.monthlyBudgets.splice(i,1);budgetViews.delete(s.id);await persist();render();return;
  }
  if(a==='housing-assumptions'){state.setupSection=3;state.view='setup';render();window.scrollTo(0,0);}
  if(a==='apply-growth-savings'){
    const h=growthHelper(s),employer=h.account==='pretax',r=oneYearGrowth(h,{employer});if(!r.complete)return;
    // Only new savings become deposits; transfers, withdrawals and the return are not copied.
    const paths=[[h.owner+'.'+h.account,r.yourSavings],...(employer?[[h.owner+'.employerPretax',r.employerSavings]]:[])];
    for(const [path,value] of paths){setPath(s,path,value);state.inputSources[s.id][path]='Estimated';}
    h.open=true;state.results.delete(s.id);invalidateExploration();await persist();setMessage('Yearly savings updated from your statements and marked Estimated. Run the plan to refresh results.');return;
  }
  if(a==='apply-budget'){try{applyBudgetEstimate(s);state.inputSources[s.id]['spending.annualBaseSpending']='Estimated';state.inputSources[s.id]['home.annualTaxesAndInsurance']='Estimated';state.results.delete(s.id);invalidateExploration();await persist();setMessage('Budget estimate applied. Run the plan to refresh results.');}catch(error){setMessage('Error: '+error.message);}}
  if(a==='run-lab')await runLab();
  if(a==='checkout'||a==='billing-portal'){el.disabled=true;try{const response=await fetch(a==='checkout'?'/api/billing/checkout':'/api/billing/portal',{method:'POST',credentials:'same-origin',headers:{...(await billingHeaders()),...(a==='checkout'?{'Content-Type':'application/json'}:{})},...(a==='checkout'?{body:JSON.stringify({interval:el.dataset.interval})}:{})});const data=await response.json();if(!response.ok)throw new Error(data.error||'Billing is unavailable.');const url=new URL(data.url);if(url.protocol!=='https:'||url.hostname!==(a==='checkout'?'checkout.stripe.com':'billing.stripe.com'))throw new Error('Unexpected billing link.');location.assign(url.href);}catch(error){el.disabled=false;setMessage('Error: '+error.message,{preserveEditor:true});}return;}
  if(a==='run-decision')await runDecision();
  if(a==='download-report')download('retirement-report.txt',reportText(s,result()),'text/plain');
  if(a==='copy-report'){try{await navigator.clipboard.writeText(reportText(s,result()));setMessage('Report copied.');}catch{setMessage('Error: Clipboard access is unavailable. Download the text report instead.');}}
  if(a==='print')window.print();
  if(a==='export-backup')download('retirement-scenarios.json',JSON.stringify({format:'retirement-readiness-lab-web-v1',inputSources:state.inputSources,scenarios:state.scenarios.map(({seed,...scenario})=>scenario)},null,2),'application/json');
  if(a==='import-backup')$('#import-file').click();
});
function numericEdit(el,{dollars=false,percent=false}={}){
  if(el.dataset.field&&!el.validity?.badInput&&el.value.trim()===''){state.inputSources[current().id][el.dataset.field]='Unknown';state.results.delete(current().id);invalidateExploration();persist();$('#result-state').textContent='Review Unknown inputs';state.message='Error: This input is Unknown, not zero. Enter a number or choose an estimate or sample value before running.';refreshNotice();const source=$('#'+el.id+'-source');if(source)source.value='Unknown';return null;}
  const value=Number(el.value)/(percent?100:1);
  // Browsers expose incomplete number input such as "1e" as an empty value.
  // Its badInput flag distinguishes it from intentionally marking an input Unknown.
  if(el.validity?.badInput||el.validity?.valueMissing||!Number.isFinite(value)){
    state.message='Error: Enter a valid, finite number. The previous value was kept.';refreshNotice();return null;
  }
  if(dollars&&Math.abs(value)>MAX_DOLLAR_AMOUNT){
    state.message='Error: The amount exceeds the supported dollar range. The previous value was kept.';refreshNotice();return null;
  }
  return value;
}
function awaitUnknownSave(){$('#result-state').textContent='Review Unknown inputs';persist();state.message='Error: This birthday or retirement date is Unknown. Choose a valid date before running.';refreshNotice();}
function dateEdit(el){
  if(el.value===''){state.inputSources[current().id][el.dataset.field]='Unknown';state.results.delete(current().id);invalidateExploration();awaitUnknownSave();return null;}
  const today=localCalendarDate(),retirement=el.dataset.field==='household.retirementDate'||el.dataset.field==='household.spouseRetirementDate';
  if(!calendarDate(el.value)||(retirement?el.value<today:el.value>=today)){
    setMessage('Error: '+(retirement?'Choose a valid retirement date today or later.':'Choose a valid birthday before today.'),{preserveEditor:true});return null;
  }
  return el.value;
}
$('#main').addEventListener('input',e=>{
  const el=e.target,key=el.dataset?.growthHelper;if(!key)return;
  const h=growthHelper(current());h[key]=el.value;h.open=true;
  // Changing the owner or account changes which fields are shown.
  if(key==='owner'||key==='account'){render({preserveEditor:true});return;}
  $('#growth-helper-result').innerHTML=growthHelperResult(current());
});
$('#main').addEventListener('change',async e=>{const el=e.target,s=current();if(el.dataset?.growthHelper)return;if(el.dataset?.inputSource){const path=el.dataset.inputSource;if(!INPUT_SOURCES.includes(el.value))return;state.inputSources[s.id][path]=el.value;state.results.delete(s.id);invalidateExploration();await persist();render();$('#f-'+path.replaceAll('.','-')+'-source')?.focus();return;}if(el.id==='withdrawal-example-month'){state.withdrawalMonth=Number(el.value);$('#withdrawal-example').innerHTML=withdrawalContent(current(),result(),state.withdrawalMonth,true);return;}if(el.id==='setup-section'){state.setupSection=Number(el.value);render();return;}if(el.dataset.field){if(el.dataset.field==='numberOfSimulations'&&!isPro()){setMessage(FREE_PATHS_ONLY);return;}const type=el.dataset.type,value=type==='checkbox'?el.checked:type==='select'?el.value:type==='date'?dateEdit(el):numericEdit(el,{dollars:type==='money',percent:type==='percent'});if(value===null)return;const conversionAmount=el.dataset.field.match(/^(rothHistory|spouseRothHistory)\.conversions\.(\d+)\.amount$/);if(conversionAmount){const lot=s[conversionAmount[1]].conversions[Number(conversionAmount[2])];if(lot.taxableAmount===lot.amount){lot.taxableAmount=value;const taxableInput=$('#f-'+conversionAmount[1]+'-conversions-'+conversionAmount[2]+'-taxableAmount');if(taxableInput)taxableInput.value=String(value);}}setPath(s,el.dataset.field,value);state.inputSources[s.id][el.dataset.field]='Entered';const source=$('#'+el.id+'-source');if(source)source.value='Entered';if(type==='date')syncCalendarAges(s);if(el.dataset.field==='numberOfSimulations')s.simulationPathsCustomized=true;if(el.dataset.field==='numberOfSimulations'&&(value<MIN_SIMULATION_PATHS||value>MAX_SIMULATION_PATHS||!Number.isInteger(value))){setPath(s,el.dataset.field,Math.max(MIN_SIMULATION_PATHS,Math.min(MAX_SIMULATION_PATHS,Math.round(value)||MIN_SIMULATION_PATHS)));render();}if(el.dataset.field==='spending.annualBaseSpending')setAnnualBaseSpending(s,value);state.results.delete(s.id);invalidateExploration();if(await persist())state.message='Saved. Run the simulation to refresh results.';if(el.dataset.field==='household.filingStatus'){render();}else{refreshEarlyWithdrawalGuidance();$('#result-state').textContent=unknownInputPaths(current(),state.inputSources).length?'Review Unknown inputs':'Ready to simulate';refreshNotice();}return;}
  if(el.dataset.budgetAdjustment){
    const b=s.budget,ui=budgetView();let annual;
    if(el.dataset.budgetAdjustment==='direction'){
      if(!['less','more'].includes(el.value))return;
      ui.direction=el.value;annual=Math.abs(b.retirementAnnualAdjustment||0)*(el.value==='less'?-1:1);
    }else{
      const monthly=numericEdit(el,{dollars:true});if(monthly===null)return;
      if(monthly<0){state.message='Error: Enter an amount of 0 or more and choose Spend less or Spend more.';refreshNotice();return;}
      annual=monthly*12;
      if(annual>MAX_DOLLAR_AMOUNT){state.message='Error: The annual spending change exceeds the supported dollar range. The previous value was kept.';refreshNotice();return;}
      const direction=b.retirementAnnualAdjustment?(b.retirementAnnualAdjustment<0?'less':'more'):ui.direction;
      if(direction==='less')annual=-annual;
    }
    if(annual===0)annual=0;
    if(annual!==b.retirementAnnualAdjustment){markBudgetEdited(b);b.retirementAnnualAdjustment=annual;await persist();}
    refreshBudget();return;
  }
  if(el.dataset.budget){const value=numericEdit(el,{dollars:true});if(value===null)return;markBudgetEdited(s.budget);s.budget[el.dataset.budget]=value;await persist();refreshBudget();return;}
  if(el.dataset.month!==undefined){
    const part=el.dataset.part,value=part==='month'?el.value:numericEdit(el,{dollars:true});if(value===null)return;
    const ui=budgetView(),i=Number(el.dataset.month),pending=i===s.budget.monthlyBudgets.length&&ui.pending,m=s.budget.monthlyBudgets[i]||pending;if(!m)return;
    if(part==='month')m.month=value;else if(part==='cashAndAtmWithdrawals')m.cashAndAtmWithdrawals=value;else if(part==='checking'||part==='credit')m[part==='checking'?'checkingSavingsBills':'creditCardBills']=[{id:part,name:part==='checking'?'Checking / savings spending':'Credit card purchases',monthlyAmount:value}];else{m.adjustments??={};m.adjustments[part]=value;}
    if(pending&&part==='month'){refreshBudget();return;}
    if(pending){s.budget.monthlyBudgets.push(m);ui.pending=null;ui.open['month-'+i]=true;}
    markBudgetEdited(s.budget);await persist();refreshBudget();
  }

});
$('#import-file').addEventListener('change',async e=>{const file=e.target.files?.[0];if(!file)return;try{const data=JSON.parse(await file.text()),scenarios=Array.isArray(data)?data:data.scenarios;if(!Array.isArray(scenarios)||!scenarios.length)throw new Error('No scenarios found in the file.');const normalized=normalizeScenarios(scenarios);for(const s of normalized){const errors=validateScenario(s);if(errors.length)throw new Error(`${s.name}: ${errors.join(' ')}`);prepareCalendarScenario(s);}state.scenarios=normalized;state.inputSources=normalizeInputSources(data.inputSources,normalized);budgetViews.clear();if(isPro())state.scenarios.forEach(applyProSimulationDefault);state.selectedId=normalized[0].id;state.results.clear();invalidateExploration();await persist(true,true);setMessage(`${normalized.length} scenarios imported.`);}catch(error){setMessage('Error: '+error.message);}e.target.value='';});
render();

async function linkAccounts(){
  syncAuthState();
  const linkingAccount=state.auth.accountKey;
  const response=await fetch('/api/billing/link',{method:'POST',credentials:'same-origin',headers:{...await billingHeaders(),...(linkingCustomerIds.length?{'x-retirement-link-customers':linkingCustomerIds.join(',')}:{})}});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||'Could not link accounts.');
  if(linkingAccount!==socialState().accountKey)throw new Error('The signed-in account changed. Check your current account before linking again.');
  if(data.linked!==true||typeof data.billingCustomerId!=='string'||!/^cus_[A-Za-z0-9]{1,200}$/.test(data.billingCustomerId))throw new Error('Could not verify the linked billing account. Please retry.');
  // Supersede checks started before linking and use the chosen customer directly.
  // The server still verifies ownership and current subscription on every use.
  accessRequest++;
  billingReference={customerId:data.billingCustomerId};saveBillingReference();rememberLinkCustomer(data.billingCustomerId);
  const query=new URLSearchParams(location.search);
  if(query.has('link')){query.delete('link');history.replaceState(null,'',location.pathname+(query.toString()?'?'+query:'')+location.hash);}
}
// Billing checks govern future runs. Completed results belong to the local plan
// and survive outages, upgrades, expiration and sign-out.
async function loadAccess({force=false}={}){
  const requestId=++accessRequest,before=JSON.stringify(state.access),query=new URLSearchParams(location.search);let message=null,rerender=force,verified=false;
  try{
    const sessionId=query.get('session_id');
    const response=await fetch('/api/billing/status'+(sessionId?'?session_id='+encodeURIComponent(sessionId):''),{credentials:'same-origin',cache:'no-store',headers:await billingHeaders()});
    if(requestId!==accessRequest)return false;
    // Authentication rejection is definitive even if the response is not JSON.
    if(response.status===401||response.status===403){
      resetBillingIdentity();
      setMessage(SIGN_IN_AGAIN,{preserveEditor:true});return false;
    }
    const access=await response.json();
    if(requestId!==accessRequest)return false;
    if(!response.ok){
      // Stripe can fail after authentication succeeds. An unknown prior identity
      // is not a change of user, and billing failures must retain known identity.
      if(typeof access?.accountKey==='string'&&access.accountKey){
        if(state.access.accountKey&&access.accountKey!==state.access.accountKey){
          resetBillingIdentity();
          state.access={...state.access,signedIn:true,accountKey:access.accountKey,accountProvider:access.accountProvider};
          state.accessStale=true;setMessage(ACCESS_UNAVAILABLE,{preserveEditor:true});return false;
        }
        state.access={...state.access,signedIn:true,accountKey:access.accountKey,accountProvider:access.accountProvider??state.access.accountProvider};
      }
      throw new Error(access?.error||'Subscription check failed');
    }
    if(!['free','pro'].includes(access.tier))throw new Error('Invalid subscription response');
    clearAccessGraceTimer();state.access=access;state.accessStale=false;accessConfirmedAt=Date.now();verified=true;
    if(!access.billingLookupUnavailable){billingReference=access.billingCustomerId?{customerId:access.billingCustomerId}:{};saveBillingReference();rememberLinkCustomer(access.billingCustomerId);}
    if(access.tier==='pro'){let changed=false;for(const scenario of state.scenarios)changed=applyProSimulationDefault(scenario)||changed;if(changed){await persist(false);rerender=true;}}
    if(requestId!==accessRequest)return false;
    if(query.get('checkout')==='success')message=access.tier==='pro'?'Pro is active for this account.':PAYMENT_PENDING;
    else if(query.get('checkout')==='canceled')message=CHECKOUT_CANCELED;
    else if(state.message===PAYMENT_PENDING&&access.tier==='pro')message='Pro is active for this account.';
    else if(state.message===ACCESS_UNAVAILABLE||state.message===ACCESS_RETRY)message='';
    // Retain an unverified checkout reference for another attempt. Once the
    // customer is verified, later requests use its direct, ownership-checked ID.
    if(!sessionId||access.billingCustomerId){query.delete('checkout');query.delete('session_id');const search=query.toString();history.replaceState(null,'',location.pathname+(search?'?'+search:'')+location.hash);}
  }catch{
    if(requestId!==accessRequest)return false;
    state.accessStale=true;
    if(isPro()){message=ACCESS_RETRY;scheduleAccessGraceExpiry();}
    else{clearAccessGraceTimer();state.access={...state.access,tier:'free',maxPaths:FREE_SIMULATION_PATHS,ownerAccess:false,checkoutAvailable:false};message=ACCESS_UNAVAILABLE;}
  }
  if(JSON.stringify(state.access)!==before)rerender=true;
  if(message!==null&&message!==state.message){state.message=message;rerender=true;}
  if(rerender)render({preserveEditor:true});
  return verified;
}
async function bootAuth(){
  try{await initializeSocialAuth(()=>{syncAuthState();loadAccess({force:true});});syncAuthState();}
  catch{state.auth=socialState();state.message=GOOGLE_UNAVAILABLE;}
  const accountQuery=new URLSearchParams(location.search);
  if(accountQuery.has('link')||accountQuery.has('account')||location.hash==='#billing'){
    state.view='billing';
    if(accountQuery.has('link'))state.message=LINK_HINT;
    accountQuery.delete('account');
    history.replaceState(null,'',location.pathname+(accountQuery.toString()?'?'+accountQuery:'')+location.hash);
  }
  await loadAccess({force:true});
}
bootAuth();
window.addEventListener('focus',()=>{if(!state.busy)loadAccess();});

window.addEventListener('storage',event=>{
  if(event.key!==storageKey&&event.key!==null)return;
  try{if(localStorage.getItem(storageKey)===lastSavedRaw)return;}catch{return;}
  state.storageError=STORAGE_CONFLICT;state.message=STORAGE_CONFLICT;render({preserveEditor:true});
});
window.addEventListener('beforeunload',event=>{
  if(!pendingSaves&&!state.storageError&&!unsavedRecoveryDraft)return;
  event.preventDefault();event.returnValue='';
});
