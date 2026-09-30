import {isPreviewResult,readinessLabel,shareLabel,PREVIEW_WARNING} from './result-format.js';
import {chartCard,mountCharts,disposeCharts} from './charts.js';
import {initializeSocialAuth,socialState,authHeaders,signInSocial,linkSocialProvider,signOutSocial} from './auth.js';
import {baseScenario,retirementAge,ageLabel,sampleScenarios,normalizeScenarios,applyProSimulationDefault,validateScenario,validateScenarioStructure,budgetBreakdown,budgetMonthTotals,validateBudget,markBudgetEdited,applyBudgetEstimate,ANNUAL_BILLS,SEPARATE_COSTS,ROTH_CONVERSION_RATES,scenarioWarnings,ENGINE_VERSION,DEFAULT_SEED,FREE_SIMULATION_PATHS,MAX_SIMULATION_PATHS} from './model.js';

const $=s=>document.querySelector(s),escapeHTML=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=(v,d=0)=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:d}).format(v||0);
const pct=v=>`${(100*(v||0)).toFixed(1)}%`;
const deep=structuredClone;
const storageKey='retirement-readiness-lab-sites-v1';
let saved,savedScenarios,savedStoredRaw=null,savedLoadError='';
try{
  const stored=localStorage.getItem(storageKey);savedStoredRaw=stored;
  if(stored!==null){
    saved=JSON.parse(stored);
    if(!Array.isArray(saved?.scenarios)||!saved.scenarios.length)throw new Error('No saved scenarios.');
    const normalized=normalizeScenarios(saved.scenarios);
    for(const s of normalized)if(validateScenarioStructure(s).length)throw new Error('Invalid saved scenario structure.');
    savedScenarios=normalized;
  }
}catch{savedLoadError='Error: Saved plans could not be loaded. Your stored backup has not been changed. Export it before replacing it. Edits stay in this page until you import a valid backup or choose to replace the unreadable plans.';}
const hasSavedScenarios=Boolean(savedScenarios);
const savedSelectedId=typeof saved?.selectedId==='string'||typeof saved?.selectedId==='number'?String(saved.selectedId).trim():'';
const state={scenarios:savedScenarios||sampleScenarios(),selectedId:savedSelectedId||'base-plan',view:'dashboard',results:new Map(),labResults:null,decision:null,busy:false,message:savedLoadError,access:{tier:'free',maxPaths:FREE_SIMULATION_PATHS,signedIn:false,checkoutAvailable:false},auth:{configured:false,chatgptSignedIn:false,signedIn:false,linkedProviders:[],enabledProviders:{google:false}},advancedOpen:false,allocationOpen:false,setupSection:0};
if(!state.scenarios.some(s=>s.id===state.selectedId))state.selectedId=state.scenarios[0].id;
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
function effectivePaths(s=current()){return isPro()?Math.max(FREE_SIMULATION_PATHS,Math.min(MAX_SIMULATION_PATHS,Number(s.numberOfSimulations)||FREE_SIMULATION_PATHS)):FREE_SIMULATION_PATHS;}
function simulationScenario(s=current()){const copy=deep(s);copy.numberOfSimulations=effectivePaths(s);copy.seed=DEFAULT_SEED;return copy;}
function persist(markStarted=true,replaceUnreadable=false){if(savedLoadError&&!replaceUnreadable)return false;if(markStarted)state.hasStartedPlan=true;try{localStorage.setItem(storageKey,JSON.stringify({scenarios:state.scenarios,selectedId:state.selectedId,hasStartedPlan:state.hasStartedPlan}));if(state.message===state.storageError||state.message===savedLoadError)state.message='';state.storageError='';savedLoadError='';savedStoredRaw=null;return true;}catch{state.storageError='Error: Changes could not be saved in browser storage. Export a backup to keep your scenarios before closing this page.';state.message=state.storageError;return false;}}
let calculationRevision=0;
function invalidateExploration(){calculationRevision++;state.labResults=null;state.decision=null;state.message='';}
function selectScenario(id){invalidateExploration();state.selectedId=id;}
function result(){return state.results.get(current().id);}
function setMessage(message,options){state.message=state.storageError||message;render(options);}
function currentViewLabel(){return {dashboard:'Overview',setup:'Assumptions',budget:'Budget',scenarios:'Scenarios',lab:'Scenario lab',results:'Results',reports:'Reports & backup',billing:'Plans & billing'}[state.view];}
function pageHead(title,detail='',actions=''){return `<div class="page-head"><div><p class="kicker">${escapeHTML(currentViewLabel())}</p><h1>${escapeHTML(title)}</h1><p>${escapeHTML(detail)}</p></div>${actions?`<div class="actions">${actions}</div>`:''}</div>`;}
function card(title,content,extra=''){return `<section class="card ${extra}"><h2>${escapeHTML(title)}</h2>${content}</section>`;}
function info(label,value){return `<div class="info-row"><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong></div>`;}
function tag(value){const c=value==='AtRisk'?'risk':value==='Watch'?'watch':'';return `<span class="tag ${c}">${escapeHTML(value)}</span>`;}
function recoveryActions(){return savedLoadError?`<div class="actions">${savedStoredRaw===null?'':'<button class="secondary" data-action="export-unreadable-backup">Export unreadable backup</button>'}<button class="secondary" data-action="replace-unreadable-plans">Replace unreadable saved plans</button></div>`:'';}
function notice(){const message=state.storageError||savedLoadError||state.message;return message?`<div class="notice ${message.startsWith('Error')?'error':'good'}" role="status">${escapeHTML(message)}${recoveryActions()}</div>`:'';}
function refreshNotice(){const existing=$('#main > .notice'),message=state.storageError||savedLoadError||state.message;if(existing){if(!message){existing.remove();return;}existing.className='notice '+(message.startsWith('Error')?'error':'good');existing.textContent=message;existing.insertAdjacentHTML('beforeend',recoveryActions());}else if(message){$('#main > .page-head').insertAdjacentHTML('afterend',notice());}}
function planningDisclosure(){return '<aside class="planning-disclosure" aria-label="Important model limitations"><strong>For U.S. retirement planning only</strong><p>These hypothetical results use U.S. federal tax, Social Security, and Medicare assumptions plus your inputs. State and local income taxes and laws outside the U.S. are not modeled. “Readiness” is the share of simulated lifetimes without a portfolio shortfall, not the probability of your actual outcome. Results are not predictions or guarantees and do not capture every cost or event. This is educational, not individualized financial, investment, tax, legal, or insurance advice. Verify your inputs and consult qualified professionals before acting.</p><a href="./methodology.html">Read model scope &amp; limitations →</a></aside>';}
const icons={lock:'<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',check:'<circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.8 2.8L16.5 9.5"/>',model:'<path d="M3 12h4l2.5-6 5 12 2.5-6h4"/>',play:'<path class="icon-fill" d="M8 5.5v13l10.5-6.5z"/>',market:'<path d="m3 17 6-6 4 4 8-8"/><path d="M15 7h6v6"/>',costs:'<path d="M12 3v18"/><path d="M16.5 7.5c0-1.9-2-3-4.5-3s-4.5 1.1-4.5 3 2 2.7 4.5 3.3 4.5 1.5 4.5 3.7-2 3-4.5 3-4.5-1.1-4.5-3"/>',tax:'<path d="M7 3h7l4 4v14H7z"/><path d="M14 3v4h4M10 12h5M10 16h5"/>',life:'<path d="M12 20s-7.5-4.6-7.5-10.3A4.2 4.2 0 0 1 12 7.2a4.2 4.2 0 0 1 7.5 2.5C19.5 15.4 12 20 12 20z"/>'};
function icon(name){return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${icons[name]}</svg>`;}
function monteCarloBanner(){const features=[['market','Market swings','Stock and bond returns change month by month.'],['costs','Rising costs','General and healthcare inflation follow their own uncertain paths.'],['tax','Taxes & benefits','Federal income tax, Social Security, and Medicare premiums are applied.'],['life','Lifespan','Each lifetime draws its length from SSA mortality tables.']];return `<section class="monte-banner" aria-labelledby="monte-title"><div class="monte-intro"><span class="monte-eyebrow">Monte Carlo retirement simulation</span><h2 id="monte-title">One plan. Many possible futures.</h2><p>Each path samples market returns, general and healthcare inflation, and lifespan. The model then follows monthly spending, income, taxes, and withdrawals through that lifetime.</p><a href="./methodology.html#monte-carlo">How Monte Carlo works →</a></div><ul class="monte-features">${features.map(([name,title,text])=>`<li><span class="feature-icon">${icon(name)}</span><strong>${title}</strong><span>${text}</span></li>`).join('')}</ul></section>`;}
function planNotice(){return isPro()?'<div class="plan-notice pro"><strong>Pro</strong> Up to 10,000 Monte Carlo paths, calculated on this device. <button class="text-link" data-view="billing">Manage plan</button></div>':'<div class="plan-notice"><strong>Free preview · 4 paths</strong> <button class="text-link" data-view="billing">Explore Pro</button></div>';}
function overviewUpgradeCard(){return `<section class="upgrade-card" aria-labelledby="overview-upgrade-title"><div class="upgrade-copy"><span class="upgrade-eyebrow">Free preview · 4 paths</span><h2 id="overview-upgrade-title">Get a steadier Monte Carlo estimate with Pro</h2><p>Pro lets you run up to 10,000 modeled lifetimes using the same plan, right on your device.</p></div><div class="upgrade-offer"><strong>Pro · up to 10,000 paths</strong><span>$9.99/month or $79/year</span><button class="primary" data-view="billing">Explore Pro plans</button><small>More paths reduce sampling noise; results remain hypothetical.</small></div></section>`;}
function readinessUpgrade(r){if(isPro())return '';const count=r.provenance.simulationCount,successes=Math.round(r.successProbability*count);return `<div class="readiness-upgrade"><strong>${successes} of ${count} paths ended without a shortfall</strong><p>Pro lets you run up to 10,000 paths for a steadier estimate.</p><button class="primary" data-view="billing">Explore Pro plans</button><small>$9.99/month or $79/year · More paths do not make this a prediction.</small></div>`;}
function monteRunSummary(count,completed){return `<div class="monte-run-summary"><span class="monte-run-label">Monte Carlo model</span><span><strong>${Number(count).toLocaleString('en-US')}</strong> ${completed?'simulated lifetimes in this run':'paths configured for the next run'}</span><a href="./methodology.html#monte-carlo">How the simulation works →</a></div>`;}
function scopeDisclosure(){return '<p class="scope-disclosure">For U.S. retirement planning only. The model uses U.S. federal rules; state and local income taxes and laws outside the U.S. are not modeled. <a href="./methodology.html">Model scope &amp; limitations</a></p>';}
function previewNotice(r){return isPreviewResult(r)?`<aside class="preview-warning"><strong>Sample preview only</strong><p>${PREVIEW_WARNING}</p></aside>`:'';}
function resultHero(r){const preview=isPreviewResult(r),count=r.provenance.simulationCount;return `<section class="result-hero" aria-label="Simulation outcome summary"><div class="result-hero-main"><span class="result-eyebrow">${preview?'Sample outcomes':'Monte Carlo readiness'}</span><div class="result-figure">${readinessLabel(r)}</div><p>${preview?'Sample lifetimes without a shortfall.':'Share of modeled lifetimes without a portfolio shortfall.'}</p>${preview?'':`<div class="bar" aria-hidden="true"><span style="width:${Math.round(100*r.successProbability)}%"></span></div>`}${previewNotice(r)}</div><dl class="result-stats"><div><dt>Median ending balance</dt><dd>${money(r.medianEndingBalance)}</dd><small>At each lifetime’s end or shortfall; failures count as $0</small></div><div><dt>Median failure age</dt><dd>${r.medianFailureAge===null?'—':ageLabel(r.medianFailureAge)}</dd><small>${r.medianFailureAge===null?'No simulated paths ran out of funds':'Among paths that ran out of funds'}</small></div><div><dt>Simulated lifetimes</dt><dd>${count.toLocaleString('en-US')}</dd><small>Fixed comparison sequence · <a href="./methodology.html#monte-carlo">How the simulation works</a></small></div></dl></section>`;}
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
function welcomePanel(){const returning=state.hasStartedPlan,points=[['model','Markets, inflation, taxes, and lifespan in one model.'],['lock','Your financial inputs stay in your browser.'],...(isPro()?[]:[['check','No sign-up needed for the free preview.']])];return `<section class="forecast-welcome" aria-labelledby="welcome-title"><div class="welcome-copy"><p class="kicker">${returning?'Welcome back':'Your retirement, explored'}</p><h1 id="welcome-title">${returning?'Pick up where you left off.':'Explore how long your retirement savings could last.'}</h1><p class="welcome-description">${returning?`Continue with <strong>${escapeHTML(current().name)}</strong>. Review your assumptions or run your plan to explore the range of possible outcomes.`:'Bring your savings, spending, and retirement plans together. Explore how different markets, costs, and lifespans could shape your future.'}</p><div class="welcome-actions"><button class="primary" data-action="start-plan">${returning?'Continue my plan':'Build my forecast'} <span aria-hidden="true">→</span></button><button class="secondary" data-action="run-plan" ${state.busy?'disabled':''}>${icon('play')}${returning?'Run my plan':'Explore a sample plan'}</button></div><ul class="welcome-points">${points.map(([name,text])=>`<li>${icon(name)}<span>${text}</span></li>`).join('')}</ul><p class="welcome-limit">${isPro()?'Pro · Up to 10,000 simulated lifetimes on your device.':'Free preview · 4 simulated lifetimes. A first look, not a reliable readiness estimate.'}</p></div>${forecastIllustration()}</section>`;}
function dashboard(){
  const s=current(),r=result(),notes=scenarioWarnings(simulationScenario(s)),assets=Object.values(s.accounts).reduce((a,b)=>a+b,0);
  const steps=`<div class="workflow"><button data-view="setup"><span class="step-number">1</span><span><strong>Review your assumptions</strong><small>Household, savings, income & costs</small></span><span aria-hidden="true">↗</span></button><button data-view="budget"><span class="step-number">2</span><span><strong>Check your spending</strong><small>Build a budget from monthly costs</small></span><span aria-hidden="true">↗</span></button><button data-view="${r?'lab':'results'}" ${r?'':'data-action="run-plan"'}><span class="step-number">3</span><span><strong>${r?'Explore a different outcome':'Run your plan'}</strong><small>${r?'Compare changes in the scenario lab':'See readiness and the range of outcomes'}</small></span><span aria-hidden="true">↗</span></button></div>`;
  return `${r?pageHead('Your retirement forecast','Explore your results, then test what could change them.'):''}${notice()}<div class="stack">${r?'':welcomePanel()}${steps}<div class="plan-summary-heading"><h2>${state.hasStartedPlan?'Current plan':'Sample plan'} at a glance</h2><span>${escapeHTML(s.name)}</span></div><section class="plan-strip" aria-label="Current plan summary"><div><span>Retirement age</span><strong>${ageLabel(retirementAge(s))}</strong><small>Current age ${s.household.currentAge}</small></div><div><span>Starting savings</span><strong>${money(assets)}</strong><small>Across all four account types</small></div><div><span>Annual base spending</span><strong>${money(s.spending.annualBaseSpending)}</strong><small>Before separate housing & health costs</small></div><div><span>Social Security claim age</span><strong>${s.socialSecurity.claimAge}</strong><small>${money(s.socialSecurity.annualBenefitAt67)} / year estimate at 67</small></div></section>${r?`<div class="split"><section class="card feature-card"><div><div class="label">${isPreviewResult(r)?'Sample outcomes':'Monte Carlo readiness estimate'}</div><div class="huge">${readinessLabel(r)}</div><p>${isPreviewResult(r)?'Sample lifetimes without a shortfall.':'Share of modeled lifetimes without a portfolio shortfall.'}</p>${previewNotice(r)}<button class="secondary" data-view="results">View full results →</button></div>${isPreviewResult(r)?'':`<div class="ring" style="--value:${Math.round(r.successProbability*100)}%"><span>Simulated lifetimes</span></div>`}</section>${card('Next useful test',`<p>${escapeHTML(r.riskBreakdown.recommendedNextTest)}</p><button class="secondary" data-view="lab">Explore in scenario lab →</button>`)}</div><div class="split">${chartCard('survival')}${card('What deserves attention',`<div class="info-list">${['market','spending','taxes','healthcare','longevity'].map(k=>`<div class="info-row"><span>${escapeHTML(k[0].toUpperCase()+k.slice(1))}</span>${tag(r.riskBreakdown[k])}</div>`).join('')}</div>`)}</div>`:''}${r?chartCard('paths'):''}${monteCarloBanner()}${isPro()?planNotice():overviewUpgradeCard()}${notes.length?`<details class="card planning-notes"><summary>Planning notes <span class="tag">${notes.length}</span></summary><ul class="warning-list">${notes.map(n=>`<li>${escapeHTML(n)}</li>`).join('')}</ul></details>`:''}${planningDisclosure()}</div>`;
}
const monthFields={'household.retirementAge':'household.retirementAgeMonths','guaranteedIncome.startAge':'guaranteedIncome.startAgeMonths','longTermCare.averageDurationYears':'longTermCare.averageDurationMonths'};
const schema=[
  ['Household & retirement',[['Current age','household.currentAge','number'],['Retirement age','household.retirementAge','number'],['Maximum modeling age','household.targetEndAge','number'],['Filing status','household.filingStatus','select','Single|Married|HeadOfHousehold'],['Your longevity table','household.gender','select','Male|Female'],['Spouse age','household.spouseCurrentAge','number'],['Spouse longevity table','household.spouseGender','select','Male|Female']]],
  ['Accounts & spending',[['Pre-tax savings','accounts.pretax','money'],['Roth savings','accounts.roth','money'],['Taxable savings','accounts.taxable','money'],['Cash reserve','accounts.cash','money'],['Annual base spending','spending.annualBaseSpending','money'],['Spending path','spending.spendingPathModel','select','EmpiricalAgeDecline|Flat'],['General inflation average %','spending.generalInflationMean','percent'],['General inflation volatility %','spending.generalInflationStdDev','percent'],['Cut spending if portfolio halves %','spending.lowPortfolioSpendingReduction','percent']]],
  ['Income & Social Security',[['Annual benefit at age 67','socialSecurity.annualBenefitAt67','money'],['Claim age','socialSecurity.claimAge','number'],['Spouse claim age','socialSecurity.spouseClaimAge','number'],['Guaranteed annual income','guaranteedIncome.annualIncome','money'],['Income start age','guaranteedIncome.startAge','number'],['Income annual increase %','guaranteedIncome.annualIncrease','percent'],['Survivor benefit %','guaranteedIncome.survivorPercent','percent']]],
  ['Housing & healthcare',[['Mortgage payment / month','mortgage.monthlyPayment','money'],['Mortgage years left','mortgage.yearsLeft','number'],['Mortgage months left','mortgage.monthsLeft','number'],['Mortgage balance','mortgage.currentBalance','money'],['Home value','home.currentValue','money'],['Rent / month','rent.monthlyRent','money'],['Pre-Medicare premium / adult / month','healthcare.preMedicareMonthlyPremium','money'],['Healthcare inflation average %','healthcare.healthcareInflationMean','percent'],['Healthcare inflation volatility %','healthcare.healthcareInflationStdDev','percent'],['Include Medicare premiums','healthcare.includeMedicarePremiums','checkbox'],['Long-term care risk','longTermCare.enabled','checkbox'],['Long-term care cost / year','longTermCare.annualCost','money'],['Long-term care duration / years','longTermCare.averageDurationYears','number']]],
  ['Market & withdrawal strategy',[['Pre-retirement return average %','market.preRetirementMeanReturn','percent'],['Pre-retirement return volatility %','market.preRetirementStdDev','percent'],['Stock return average %','market.stockMeanReturn','percent'],['Stock return volatility %','market.stockStdDev','percent'],['Bond return average %','market.bondMeanReturn','percent'],['Bond return volatility %','market.bondStdDev','percent'],['Roth conversions','rothConversion.enabled','checkbox'],['Conversion tax bracket cap %','rothConversion.marginalRateCap','percent'],['Use cash in market drawdowns','withdrawalStrategy.useCashReserveDuringDrawdowns','checkbox'],['Drawdown trigger %','withdrawalStrategy.drawdownTrigger','percent'],['Early withdrawal penalty','withdrawalStrategy.applyEarlyWithdrawalPenalty','checkbox'],['Rule of 55 eligible','withdrawalStrategy.ruleOf55Eligible','checkbox'],['72(t) / SEPP eligible','withdrawalStrategy.seppEligible','checkbox'],['Simulation paths','numberOfSimulations','number']]],
  ['Stock allocation by portfolio size',[['Under 30× spending %','postRetirementAllocation.stockUnder30x','percent'],['30–35× spending %','postRetirementAllocation.stock30xTo35x','percent'],['35–40× spending %','postRetirementAllocation.stock35xTo40x','percent'],['40–45× spending %','postRetirementAllocation.stock40xTo45x','percent'],['45–50× spending %','postRetirementAllocation.stock45xTo50x','percent'],['50× or more %','postRetirementAllocation.stock50xOrMore','percent']]]
];
const assumptionHelp={
  'accounts.pretax':'Traditional 401(k), IRA, and similar tax-deferred balances. The model treats withdrawals from this balance as taxable income.',
  'accounts.roth':'Roth retirement account balances. The model treats withdrawals from this balance as tax-free.',
  'accounts.taxable':'Brokerage investments outside retirement accounts. The model includes them in invested savings and can draw on them for spending. It does not calculate taxes on brokerage gains, dividends, or withdrawals.',
  'accounts.cash':'Cash reserve kept apart from invested savings. It can help cover withdrawals during modeled market declines.',
  'spending.annualBaseSpending':'Your yearly living costs in today\'s dollars excluding housing and healthcare which are entered separately below.',
  'spending.generalInflationMean':'The average yearly increase assumed for general living and housing costs.',
  'socialSecurity.claimAge':'Age when the model starts your Social Security benefit. It adjusts the age-67 estimate above for this claiming age.',
  'guaranteedIncome.startAge':'Age when the pension or annuity income begins in the simulation.',
  'mortgage.monthlyPayment':'Monthly principal-and-interest mortgage payment added separately to living costs while the loan remains. Enter taxes and insurance in the budget rather than including escrow here. Enter 0 if there is no mortgage.',
  'mortgage.yearsLeft':'Full years remaining on the mortgage. Add any extra months in the next field.',
  'mortgage.monthsLeft':'Extra months remaining beyond the full years above, from 0 through 11.',
  'mortgage.currentBalance':'Estimated unpaid mortgage principal. The model infers a fixed interest rate from your balance, payment, and remaining term, then deducts the remaining loan at a home sale. The payment must cover the balance over that term; use principal and interest only, with taxes and insurance entered in the budget.',
  'rent.monthlyRent':'Monthly rent added separately to living costs during retirement. Leave at 0 if you are not renting.',
  'healthcare.healthcareInflationMean':'Average yearly growth assumed for healthcare premiums and long-term care costs.',
  'longTermCare.annualCost':'Annual cost per person during a modeled long-term care episode, before future healthcare inflation.',
  'longTermCare.averageDurationYears':'Whole years plus extra months a modeled care episode lasts before the person’s death when long-term care occurs.',
  'market.stockMeanReturn':'Average yearly return assumed for the stock share of invested savings after retirement.',
  'market.bondMeanReturn':'Average yearly return assumed for the bond share of invested savings after retirement.',
  'market.bondStdDev':'How much modeled bond returns vary around their average after retirement.',
  'household.gender':'Select the mortality rates used to draw your lifespan in each simulation. This does not change your filing status.',
  'household.spouseGender':'Select the mortality rates used to draw your spouse’s lifespan in married simulations.',
  'spending.spendingPathModel':'Age-based decline gradually lowers base spending from ages 65 to 85. Flat keeps base spending level before inflation.',
  'spending.generalInflationStdDev':'Controls how much inflation varies around the average in simulated months. Higher values make future costs less predictable.',
  'spending.lowPortfolioSpendingReduction':'The model reduces base spending by this percentage while total savings are below half their value at retirement.',
  'socialSecurity.annualBenefitAt67':'Enter an annual Social Security estimate at age 67 in today’s dollars. The model adjusts it for the claim age below and applies its own future inflation.',
  'socialSecurity.spouseClaimAge':'The spouse’s modeled claiming age. Spousal or survivor payments may begin later if other conditions are not met.',
  'guaranteedIncome.annualIncome':'Annual pension or annuity income in today’s dollars, separate from Social Security. The annual increase applies from today, including years before payments begin at the income start age.',
  'guaranteedIncome.annualIncrease':'Yearly growth rate for the pension or annuity amount entered above.',
  'guaranteedIncome.survivorPercent':'For a married plan, the share of guaranteed income retained by the spouse after the primary person dies.',
  'home.currentValue':'The home is not counted as spendable savings initially. The model sells it when the portfolio runs out or every surviving household member is in long-term care, pays the remaining mortgage, and adds the net equity to cash. Replacement rent applies only while someone lives outside care.',
  'healthcare.preMedicareMonthlyPremium':'Monthly healthcare premium for each retired adult younger than 65. The model grows this cost with healthcare inflation.',
  'healthcare.healthcareInflationStdDev':'Controls how much healthcare cost growth varies around its average in simulated months.',
  'healthcare.includeMedicarePremiums':'Adds modeled Medicare premiums from age 65, including higher-income surcharges when applicable.',
  'longTermCare.enabled':'Adds a chance of long-term care costs near the end of each modeled life, using the cost and duration below.',
  'market.preRetirementMeanReturn':'Average annual investment return before retirement. This strongly affects the assets available at retirement.',
  'market.preRetirementStdDev':'How much pre-retirement investment returns vary around their average. Higher values widen the range of outcomes.',
  'market.stockStdDev':'How much modeled stock returns vary around their average after retirement.',
  'rothConversion.enabled':'Moves some pre-tax savings into Roth savings at year end and pays the estimated conversion tax from the portfolio. With early-withdrawal penalties enabled, withdrawals of newly converted principal before age 59½ can incur a 10% penalty for five modeled tax years, including money used to pay conversion tax.',
  'rothConversion.marginalRateCap':'The highest federal income tax bracket the model fills with Roth conversions. For example, 22% fills available room through the 22% bracket; it is not a flat 22% tax on the whole conversion. Applies only when Roth conversions are on. The 37% bracket has no upper income limit, so the model may convert the entire remaining pre-tax balance.',
  'withdrawalStrategy.useCashReserveDuringDrawdowns':'Uses the cash reserve before invested accounts when the modeled monthly portfolio return falls below the trigger.',
  'withdrawalStrategy.drawdownTrigger':'The monthly portfolio return that activates cash-first withdrawals. For example, -1% means a month below -1%.',
  'withdrawalStrategy.applyEarlyWithdrawalPenalty':'Adds a modeled 10% penalty to applicable withdrawals before age 59½. Rule of 55 can exempt pre-tax withdrawals in this model, but not distributions of recent Roth conversions. Opening Roth savings are assumed available without a penalty.',
  'withdrawalStrategy.ruleOf55Eligible':'In this model, removes that early withdrawal penalty when retirement age is at least 55. Verify your actual eligibility separately.',
  'withdrawalStrategy.seppEligible':'Models a series of pre-tax distributions using the app’s life-expectancy formula. Verify the actual 72(t) rules before relying on it.',
  'numberOfSimulations':'Number of simulated lifetimes. More paths make the readiness estimate steadier but take longer to run.',
};
function helpHTML(path,label){const tip=assumptionHelp[path];if(!tip)return '';const id='help-'+path.replaceAll('.','-');return `<span class="help"><button type="button" class="help-trigger" aria-label="Explain ${escapeHTML(label)}" aria-describedby="${id}" aria-expanded="false">?</button><span class="help-popover" id="${id}" role="tooltip">${escapeHTML(tip)}</span></span>`;}
const spouseFieldPaths=new Set(['household.spouseCurrentAge','household.spouseGender','socialSecurity.spouseClaimAge','guaranteedIncome.survivorPercent']);
function visibleFields(s,fields){return s.household.filingStatus==='Married'?fields:fields.filter(([,path])=>!spouseFieldPaths.has(path));}
function getPath(obj,path){return path.split('.').reduce((v,k)=>v[k],obj);}
function setPath(obj,path,value){const keys=path.split('.');const last=keys.pop();keys.reduce((v,k)=>v[k],obj)[last]=value;}
function fieldHTML(s,[label,path,type,options]){
  const value=getPath(s,path),id='f-'+path.replaceAll('.','-'),help=helpHTML(path,label);
  if(path==='numberOfSimulations'&&!isPro())return `<div class="field"><div class="field-label"><span>Simulation paths</span>${help}</div><strong>4 paths · Free preview</strong><p class="form-note">Upgrade to choose up to 10,000 paths. <button class="text-link" data-view="billing">View Pro</button></p></div>`;
  if(type==='checkbox')return `<div class="field checkbox"><input id="${id}" type="checkbox" data-field="${path}" data-type="${type}" ${value?'checked':''}><label for="${id}">${escapeHTML(label)}</label>${help}</div>`;
  const heading=`<div class="field-label"><label for="${id}">${escapeHTML(label)}</label>${help}</div>`;
  if(path==='rothConversion.marginalRateCap'){
    const supported=ROTH_CONVERSION_RATES.some(rate=>Math.abs(rate-value)<.0001);
    return `<div class="field">${heading}<select id="${id}" data-field="${path}" data-type="percent">${supported?'':`<option selected disabled value="${escapeHTML(value*100)}">Unsupported value (${escapeHTML(value*100)}%) — choose a bracket</option>`}${ROTH_CONVERSION_RATES.map(rate=>`<option value="${Math.round(rate*100)}" ${Math.abs(rate-value)<.0001?'selected':''}>${Math.round(rate*100)}%</option>`).join('')}</select></div>`;
  }
  if(type==='select')return `<div class="field">${heading}<select id="${id}" data-field="${path}" data-type="${type}">${options.split('|').map(opt=>`<option value="${opt}" ${opt===value?'selected':''}>${escapeHTML(path.endsWith('gender')||path.endsWith('Gender')?opt+' mortality rates':opt==='HeadOfHousehold'?'Head of household':opt==='EmpiricalAgeDecline'?'Empirical age decline':opt)}</option>`).join('')}</select></div>`;
  const monthPath=monthFields[path];
  if(monthPath){
    const monthId='f-'+monthPath.replaceAll('.','-'),monthValue=getPath(s,monthPath);
    return `<div class="field">${heading}<div class="timing-inputs"><input id="${id}" type="number" inputmode="numeric" step="1" min="0" data-field="${path}" data-type="number" value="${escapeHTML(value)}"><div class="timing-months"><label for="${monthId}">Extra months</label><select id="${monthId}" data-field="${monthPath}" data-type="month">${Array.from({length:12},(_,month)=>`<option value="${month}" ${month===monthValue?'selected':''}>${month}</option>`).join('')}</select></div></div></div>`;
  }
  const resource=path==='socialSecurity.annualBenefitAt67'?`<div class="field-resource"><a href="https://www.ssa.gov/myaccount/" target="_blank" rel="noopener noreferrer">Find your estimate at my Social Security ↗</a><small>Choose today’s dollars if offered; multiply the age-67 monthly estimate by 12.</small></div>`:'';
  return `<div class="field">${heading}<input id="${id}" type="number" inputmode="decimal" step="${type==='number'?1:'any'}" ${path==='numberOfSimulations'?`min="4" max="${MAX_SIMULATION_PATHS}"`:''} data-field="${path}" data-type="${type}" value="${escapeHTML(type==='percent'?Number((value*100).toPrecision(10)):value)}">${resource}</div>`;
}
const setupSections=[
  ['Household','Your timeline and household','Set your retirement timing and the household used in the model.'],
  ['Accounts & spending','Savings and everyday spending','Enter current balances and the living costs your savings will need to support.'],
  ['Income & Social Security','Income you can plan around','Add Social Security and any pension or annuity income.'],
  ['Housing & healthcare','Housing and health costs','These costs are modeled separately from annual base spending.'],
  ['Market & strategy','Investments and withdrawal strategy','Review return assumptions and how the plan uses your savings.']
];
function setup(){
  const s=current(),i=state.setupSection,[label,title,description]=setupSections[i];
  const fields=(section,start,end)=>`<div class="fields">${visibleFields(s,schema[section][1].slice(start,end)).filter(([,path])=>path!=='household.targetEndAge').map(f=>fieldHTML(s,f)).join('')}</div>`;
  const group=(title,content,note='')=>`<section class="form-group"><div class="group-heading"><h3>${title}</h3>${note?`<p>${note}</p>`:''}</div>${content}</section>`;
  const allocation=`<details id="allocation-settings" class="advanced-settings inset-details" ${state.allocationOpen?'open':''}><summary>Stock allocation by portfolio size</summary><p class="form-note">The app compares invested savings with one year of modeled costs. A 30× balance means invested savings equal 30 times that yearly amount; it is not a guarantee of 30 years of funding. Each percentage is the stock share; the rest is bonds.</p>${fields(5,0)}</details>`;
  const advanced=`<details id="advanced-model" class="advanced-settings inset-details" ${state.advancedOpen?'open':''}><summary>Advanced model settings</summary><p class="form-note">The maximum modeling age caps the simulation. It is not a predicted lifespan; each modeled lifetime uses mortality rates.</p><div class="fields">${fieldHTML(s,schema[0][1][2])}${schema[4][1].slice(13).map(f=>fieldHTML(s,f)).join('')}</div></details>`;
  const content=[
    group('Retirement timeline',fields(0,0,2))+group('Your household',fields(0,3),'Spouse settings appear when filing status is Married. Longevity tables guide mortality estimates.'),
    group('Current account balances',fields(1,0,4),'Enter today’s balances in dollars.')+group('Living costs',fields(1,4,6),'<button class="text-link" data-view="budget">Use the budget builder to estimate annual spending →</button>')+group('How spending changes',fields(1,6),'Percent fields accept values such as 2.3 for 2.3%.'),
    group('Social Security',fields(2,0,3))+group('Pension or annuity',fields(2,3),'Enter 0 for guaranteed annual income if this does not apply.'),
    group('Home & mortgage',fields(3,0,6),'Leave amounts at 0 where they do not apply.')+group('Healthcare premiums',fields(3,6,10))+group('Long-term care',fields(3,10)),
    group('Investment returns',fields(4,0,6),'Annual averages and volatility, entered as percentages.')+group('Roth conversions',fields(4,6,8))+group('Cash reserve strategy',fields(4,8,10))+group('Early withdrawals',fields(4,10,13))+allocation+advanced
  ][i];
  return `${pageHead('Set up your Monte Carlo model','Work through one section at a time. These inputs shape every simulated lifetime.')}${notice()}${scopeDisclosure()}${planNotice()}<div class="setup-layout"><aside class="section-picker"><p class="section-eyebrow">Plan sections</p><div class="section-links" role="navigation" aria-label="Assumption sections">${setupSections.map(([name,,desc],n)=>`<button data-action="setup-section" data-index="${n}" ${i===n?'aria-current="step"':''}><span class="section-number">${n+1}</span><span>${name}</span><span class="section-arrow" aria-hidden="true">›</span></button>`).join('')}</div><div class="mobile-section"><label for="setup-section">Plan section</label><select id="setup-section">${setupSections.map(([name],n)=>`<option value="${n}" ${i===n?'selected':''}>${n+1}. ${name}</option>`).join('')}</select></div><div class="section-tip"><strong>Need a hand?</strong><p>Select a ? beside a field for a plain-language explanation.</p><p>Run the simulation after making changes to update your results.</p></div></aside><div class="setup-content"><section class="card form-panel"><header class="form-panel-head"><span class="kicker">Section ${i+1} of ${setupSections.length}</span><h2 id="section-title" tabindex="-1">${title}</h2><p>${description}</p></header>${content}<div class="section-footer">${i?`<button class="secondary" data-action="setup-section" data-index="${i-1}">← Previous</button>`:'<span></span>'}${i<4?`<button class="primary" data-action="setup-section" data-index="${i+1}">Next: ${setupSections[i+1][0]} →</button>`:`<button class="primary" data-action="run-plan" ${state.busy?'disabled':''}>${state.busy?'Calculating…':'Run simulation →'}</button>`}</div></section><div class="setup-bottom"><span>All amounts in U.S. dollars.</span><button class="text-link" data-action="reset-assumptions">Restore sample values</button></div></div></div>`;
}
function budgetSummary(){
  const s=current(),b=s.budget,d=budgetBreakdown(b),errors=validateBudget(b,{requireMonths:true});
  return `<h2>Your spending estimate</h2>${errors.length?`<div class="budget-errors" role="status"><ul>${errors.map(e=>`<li>${escapeHTML(e)}</li>`).join('')}</ul></div>`:''}${d.count?`<p class="form-note">Using ${d.count} ${d.count===1?'month':'months'}: ${d.months.map(m=>escapeHTML(m.month)).join(', ')}.</p>${d.count<12?'<p class="sample-note">A shorter sample may miss seasonal spending. Use 12 complete months when available.</p>':''}${b.monthlyBudgets.length>12?'<p class="sample-note">Only the 12 most recent months are used. Adjustments from older months are excluded.</p>':''}<div class="info-list">${info('Average monthly spending',money(d.grossAverage,2))}${info('Annual bills already counted (avg.)','− '+money(d.annualBillsAverage,2))}${info('Separate plan costs (avg.)','− '+money(d.separateCostsAverage,2))}${info('Monthly spending after adjustments',money(d.monthlyAverage,2))}${info('Annualized spending (× 12)',money(d.annualized,2))}${info('Annual bills added once','+ '+money(d.annualBills,2))}${info('Retirement adjustment',(d.retirementAdjustment<0?'− ':'+ ')+money(Math.abs(d.retirementAdjustment),2))}</div>`:'<p class="form-note">Start with one complete month. Combine spending across accounts for that month.</p>'}<div class="budget-result"><span>Estimated annual base spending</span><strong>${errors.length?'—':money(d.estimate)}</strong></div><p class="form-note">Current plan: ${money(s.spending.annualBaseSpending)} / year.</p><p class="form-note">${b.isAppliedToAnnualBaseSpending&&!b.estimateNeedsReview?'This estimate is applied to the plan.':'Budget edits do not change the plan until you use this estimate.'}</p>`;
}
function budgetMonthSummary(m){const t=budgetMonthTotals(m);return `<span>Total spending <strong>${money(t.gross,2)}</strong></span><span>Already counted adjustments <strong>− ${money(t.annualBills+t.separateCosts,2)}</strong></span><span>After adjustments <strong>${money(t.adjusted,2)}</strong></span>`;}
function budget(){
  const b=current().budget;
  const monthlyField=(i,key,label,value,allowNegative=false)=>`<div class="field"><label for="month-${i}-${key}">${label}</label><input id="month-${i}-${key}" type="number" ${allowNegative?'':'min="0"'} step="any" data-month="${i}" data-part="${key}" value="${escapeHTML(value)}"></div>`;
  return `${pageHead('Build an annual budget','Use account statements to estimate everyday spending, then adjust for retirement.')}${notice()}${scopeDisclosure()}<div class="budget-layout"><div class="stack">${card('1. Monthly spending',`<p class="form-note">Enter complete months, once each. Combine all your accounts. Use purchases and spending, not changes in account balances.</p><div class="spending-guide"><div><strong>Credit card purchases</strong><p>Purchases minus refunds. Exclude card payments, balance transfers, and cash advances. Net refunds can be negative.</p></div><div><strong>Checking / savings spending</strong><p>Direct bills, checks, and debit purchases. Exclude credit card payments, ATM withdrawals, and transfers between your own accounts.</p></div><div><strong>Cash / ATM withdrawals</strong><p>Cash withdrawn for spending. We assume this cash was spent; do not also include cash purchases in another total.</p></div></div><div class="month-list">${b.monthlyBudgets.map((m,i)=>{const t=budgetMonthTotals(m);return `<section class="spending-month" aria-label="Spending entry ${i+1}"><div class="month-heading"><label for="month-${i}-date">Month<input id="month-${i}-date" type="month" data-month="${i}" data-part="month" value="${escapeHTML(m.month)}"></label><button class="subtle" data-action="remove-month" data-index="${i}" aria-label="Remove spending entry ${i+1}">Remove</button></div><div class="fields monthly-amounts">${monthlyField(i,'credit','Credit card purchases',t.credit,true)}${monthlyField(i,'checking','Checking / savings spending',t.checking)}${monthlyField(i,'cashAndAtmWithdrawals','Cash / ATM withdrawals',m.cashAndAtmWithdrawals||0)}</div><details class="budget-adjustments"><summary>Avoid counting costs twice</summary><p class="form-note">Enter only amounts already included in this month’s three totals. If you excluded a cost when entering those totals, leave its adjustment at 0. Never deduct the same payment in more than one field.</p><h3>Annual bills paid this month</h3><p class="form-note">Remove these payments from the monthly average, then enter their expected annual amounts in section 2.</p><div class="fields">${ANNUAL_BILLS.map(([label,,key])=>monthlyField(i,key,label+' already counted',m.adjustments?.[key]||0)).join('')}</div><h3>Costs modeled separately in the plan</h3><p class="form-note">Remove mortgage, rent, and healthcare premiums included in the totals. Enter the expected retirement costs in <button class="text-link" data-action="housing-assumptions">Housing & healthcare</button>. Keep other medical spending here. If a mortgage payment includes escrow for taxes or insurance, split it across the appropriate adjustment fields only once.</p><div class="fields">${SEPARATE_COSTS.map(([label,key])=>monthlyField(i,key,label+' already counted',m.adjustments?.[key]||0)).join('')}</div></details><div class="month-totals" data-month-totals="${i}">${budgetMonthSummary(m)}</div></section>`;}).join('')}</div><button class="secondary" data-action="add-month">+ Add month</button>`)}${card('2. Annual bills to spread across the year',`<p class="form-note">Optional. Enter expected yearly amounts in today’s dollars. They are added once per year. For any of these bills included in your monthly totals, enter the payment under that month’s “Avoid counting costs twice.”</p><div class="fields">${ANNUAL_BILLS.map(([label,key])=>`<div class="field"><label for="budget-${key}">${label} / year</label><input id="budget-${key}" type="number" min="0" step="any" data-budget="${key}" value="${escapeHTML(b[key])}"></div>`).join('')}</div>`)}${card('3. Adjust for retirement',`<p class="form-note">Optional. Add expected changes to everyday spending in today’s dollars: a positive amount for more spending, or a negative amount for less. Do not add housing or premiums modeled separately.</p><div class="field"><label for="budget-retirementAnnualAdjustment">Annual spending change (+ / −)</label><input id="budget-retirementAnnualAdjustment" type="number" step="any" data-budget="retirementAnnualAdjustment" value="${escapeHTML(b.retirementAnnualAdjustment||0)}"></div>`)}</div><aside class="card budget-summary" id="budget-summary"><div id="budget-summary-body">${budgetSummary()}</div><button class="primary" data-action="apply-budget" ${validateBudget(b,{requireMonths:true}).length?'disabled':''}>Use this estimate</button></aside></div>`;
}
function refreshBudget(){
  refreshNotice();
  $('#budget-summary-body').innerHTML=budgetSummary();
  $('#budget-summary [data-action=apply-budget]').disabled=validateBudget(current().budget,{requireMonths:true}).length>0;
  current().budget.monthlyBudgets.forEach((m,i)=>{const row=document.querySelector(`[data-month-totals="${i}"]`);if(row)row.innerHTML=budgetMonthSummary(m);});
}
function scenarios(){return `${pageHead('Saved scenarios','Keep each set of assumptions so you can compare its Monte Carlo results.',`<button class="primary" data-action="new-scenario">Duplicate current plan</button>`)}${notice()}${card('Plans',state.scenarios.map(s=>`<div class="scenario-row"><div><h3>${escapeHTML(s.name)} ${s.id===state.selectedId?'<span class="tag">Selected</span>':''}</h3><p>Retire at ${ageLabel(retirementAge(s))} · ${money(s.spending.annualBaseSpending)} annual spending · ${money(Object.values(s.accounts).reduce((a,b)=>a+b,0))} assets</p></div><div class="actions"><button class="secondary" data-action="select-scenario" data-id="${escapeHTML(s.id)}">Open</button><button class="subtle" data-action="rename-scenario" data-id="${escapeHTML(s.id)}">Rename</button><button class="danger" data-action="delete-scenario" data-id="${escapeHTML(s.id)}" ${state.scenarios.length===1?'disabled':''}>Delete</button></div></div>`).join(''))}`;}
const labVariants=[['Retire 2 years later',s=>s.household.retirementAge+=2],['Spend 5% less',s=>s.spending.annualBaseSpending*=.95],['Claim Social Security at 70',s=>s.socialSecurity.claimAge=70],['Higher healthcare costs',s=>{s.healthcare.preMedicareMonthlyPremium*=1.25;s.healthcare.healthcareInflationMean=Math.min(.20,s.healthcare.healthcareInflationMean+.01);}],['Use Roth conversions',s=>{s.rothConversion.enabled=true;s.rothConversion.marginalRateCap=.22;}],['Larger cash reserve strategy',s=>{s.withdrawalStrategy.useCashReserveDuringDrawdowns=true;s.withdrawalStrategy.drawdownTrigger=-.01;}]];
function lab(){const completed=state.labResults?.find(row=>row.result)?.result,comparisonCount=completed?.provenance.simulationCount??Math.min(effectivePaths(),150);return `${pageHead('Monte Carlo scenario lab','Change one assumption at a time, then compare modeled lifetimes against the same starting plan.',`<button class="primary" data-action="run-lab" ${state.busy?'disabled':''}>Run comparisons</button><button class="secondary" data-action="run-decision" ${state.busy||!isPro()?'disabled':''}>Find age & spending targets${isPro()?'':' · Pro'}</button>`)}${notice()}${planNotice()}${state.decision?card('Planning targets',`<div class="grid two"><div><span class="metric-label">Earliest age at ${pct(state.decision.targetReadiness)} readiness</span><div class="metric-value">${state.decision.earliestRetirementAge??'No age found'}</div><p class="form-note">Search whole-year ages through 70 with ${state.decision.simulationCount} paths per age.</p></div><div><span class="metric-label">Modeled annual spending at ${pct(state.decision.targetReadiness)} readiness</span><div class="metric-value">${state.decision.safeAnnualSpending===null?'No amount found':(state.decision.safeSpendingAtSearchLimit?'At least ':'')+money(state.decision.safeAnnualSpending)}</div><p class="form-note">${state.decision.safeSpendingAtSearchLimit?`The search stops at ${money(state.decision.safeSpendingSearchLimit)}; higher spending was not tested. `:''}Rounded down to $500; rerun the full plan before making decisions.</p></div></div>`):''}${state.labResults?card('Scenario comparison',`${previewNotice(completed)}<div class="comparison-row"><strong>Scenario</strong><strong>${comparisonCount<=FREE_SIMULATION_PATHS?'Samples without shortfall':'Modeled readiness'}</strong><strong>Median ending</strong></div>${state.labResults.map(r=>`<div class="comparison-row"><div><strong>${escapeHTML(r.label)}</strong>${r.error?`<div class="muted">${escapeHTML(r.error)}</div>`:''}</div><strong>${r.result?readinessLabel(r.result):'—'}</strong><strong>${r.result?money(r.result.medianEndingBalance):'—'}</strong></div>`).join('')}<p class="form-note">Each comparison runs ${comparisonCount} Monte Carlo paths with the fixed comparison sequence for reproducible screening. Run the full plan for final results.</p>`):card('Quick comparisons',`<p class="form-note">The lab tests retirement timing, spending, claiming age, healthcare costs, Roth conversions, and cash use.</p><ul class="warning-list">${labVariants.map(x=>`<li>${escapeHTML(x[0])}</li>`).join('')}</ul>`)}`;}
function results(){const r=result();return `${pageHead('Monte Carlo simulation results','Explore the range of outcomes across many modeled lifetimes.')}${notice()}${planNotice()}${r?'':monteRunSummary(effectivePaths(),false)}${!r?(state.busy?'':'<div class="notice">Run the selected scenario to view its results.</div>'):`<div class="stack">${resultHero(r)}<div class="split">${chartCard('survival')}${card('Next useful test',`<p class="next-test">${escapeHTML(r.riskBreakdown.recommendedNextTest)}</p><button class="secondary" data-view="lab">Explore in scenario lab →</button>${readinessUpgrade(r)}`)}</div>${chartCard('paths')}${chartCard('bands')}${card('Portfolio survival by age',`<div class="table-wrap"><table><thead><tr><th>Age</th><th>${isPreviewResult(r)?'Samples without shortfall':'Share without shortfall'}</th><th>${isPreviewResult(r)?'Samples alive':'Share alive'}</th></tr></thead><tbody>${r.notFailedByAge.filter((_,i)=>i%5===0||i===r.notFailedByAge.length-1).map(p=>`<tr><td>${ageLabel(p.age)}</td><td>${shareLabel(p.notFailedShare,r.provenance.simulationCount)}</td><td>${shareLabel(p.aliveShare,r.provenance.simulationCount)}</td></tr>`).join('')}</tbody></table></div>`)}${card('Balance bands by age',`<p class="form-note">Only paths still running at each age contribute. A failure contributes $0 at its failure age and stops. Completed lifetimes are not extended. Later rows use fewer paths, so these ranges do not measure overall readiness.</p><div class="table-wrap"><table><thead><tr><th>Age</th><th>Paths at this age</th><th>10th percentile</th><th>Median</th><th>90th percentile</th></tr></thead><tbody>${r.balanceBands.filter((_,i)=>i%5===0||i===r.balanceBands.length-1).map(b=>`<tr><td>${ageLabel(b.age)}</td><td>${b.pathCount}</td><td>${money(b.pessimistic)}</td><td>${money(b.median)}</td><td>${money(b.optimistic)}</td></tr>`).join('')}</tbody></table></div>`)}${card('Failure ages',r.failureAgeBuckets.length?`<div class="info-list">${r.failureAgeBuckets.map(b=>info(`Ages ${b.label}`,isPreviewResult(r)?`${b.count} sample paths`:`${b.count} paths · ${pct(b.shareOfFailures)} of failures`)).join('')}</div>`:'<p class="muted">No simulated paths ran out of funds.</p>')}${planningDisclosure()}</div>`}`;}
function reportText(s,r){const count=r?.provenance.simulationCount??effectivePaths(s),precisePct=v=>`${Number((v*100).toPrecision(12))}%`;const lines=[`RETIREMENT FORECAST - MONTE CARLO SIMULATOR — ${s.name}`,`Generated ${new Date().toLocaleString()}`,`Calculation engine: ${r?.provenance.engineVersion||ENGINE_VERSION}`,`Simulation paths: ${count}; fixed comparison sequence`,...(r?[`Paths for next run: ${effectivePaths(s)}`]:[]),'','U.S. MODEL SCOPE: U.S. federal tax, Social Security, and Medicare assumptions only. State and local income taxes and laws outside the U.S. are not modeled.',`Current age: ${s.household.currentAge}; retirement age: ${ageLabel(retirementAge(s))}`,`Filing status: ${s.household.filingStatus}`,`Starting pre-tax: ${money(s.accounts.pretax,2)}; Roth: ${money(s.accounts.roth,2)}; taxable: ${money(s.accounts.taxable,2)}; cash: ${money(s.accounts.cash,2)}`,`Annual base spending: ${money(s.spending.annualBaseSpending,2)}`,`General inflation: ${precisePct(s.spending.generalInflationMean)} ± ${precisePct(s.spending.generalInflationStdDev)}`,`Social Security benefit at 67: ${money(s.socialSecurity.annualBenefitAt67,2)}; claim age: ${s.socialSecurity.claimAge}`,`Pre-Medicare monthly premium: ${money(s.healthcare.preMedicareMonthlyPremium,2)}`,`Long-term care: ${s.longTermCare.enabled?'included':'excluded'}`,`Roth conversions: ${s.rothConversion.enabled?'enabled':'disabled'}`,''];lines.push('ALL ASSUMPTIONS');for(const [section,fields] of schema){lines.push(section+':');for(const [label,path,type] of visibleFields(s,fields)){const value=path==='numberOfSimulations'?count:getPath(s,path);lines.push(`  ${label}: ${type==='percent'?precisePct(value):type==='money'?money(value,2):type==='checkbox'?(value?'Yes':'No'):value}`);if(monthFields[path])lines.push(`  ${label} extra months: ${getPath(s,monthFields[path])}`);}}lines.push(`Budget months: ${s.budget.monthlyBudgets.length}`,`Property taxes: ${money(s.budget.annualPropertyTaxes,2)}; home insurance: ${money(s.budget.annualHomeInsurance,2)}; auto insurance: ${money(s.budget.annualAutoInsurance,2)}`,'');const bd=budgetBreakdown(s.budget);lines.push('BUDGET WORKSHEET',`Months used: ${bd.months.map(m=>m.month).join(', ')||'none'}`,`Monthly average after adjustments: ${money(bd.monthlyAverage,2)}`,`Annual bills added once: ${money(bd.annualBills,2)}`,`Annual retirement adjustment: ${money(bd.retirementAdjustment,2)}`,`Draft annual estimate: ${money(bd.estimate,2)}`,`Applied and current: ${s.budget.isAppliedToAnnualBaseSpending&&!s.budget.estimateNeedsReview?'Yes':'No'}`,'');if(r)lines.push(`Lifetimes without a portfolio shortfall: ${readinessLabel(r)}`, ...(isPreviewResult(r)?[`SAMPLE PREVIEW ONLY: ${PREVIEW_WARNING}`]:[]),`Median ending balance: ${money(r.medianEndingBalance)}`,`10th / 90th percentile of ending balances: ${money(r.pessimisticEndingBalance)} / ${money(r.optimisticEndingBalance)}`,`Ending balances are measured at each lifetime’s end or shortfall; failed endings count as $0.`, `Balance bands use observed paths only, through age ${ageLabel(r.balanceBands.at(-1)?.age??retirementAge(s))}.`,`Median failure age: ${r.medianFailureAge===null?'none':ageLabel(r.medianFailureAge)}`,`Primary risk: ${r.riskBreakdown.primaryRisk}`,`Next useful test: ${r.riskBreakdown.recommendedNextTest}`);else lines.push('Run the scenario to include results.');lines.push('','RESULT LIMITATIONS: Hypothetical results depend on your inputs and model assumptions. Modeled readiness is not the probability your actual plan will succeed. Results are not predictions or guarantees and do not capture every cost or event.','USE: Educational estimate only, not individualized financial, investment, tax, legal, or insurance advice. Verify inputs and consult qualified professionals before acting.','DATA: Scenarios are stored locally in this browser; safeguard any exported file.');return lines.join('\n');}
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
  const proActions=isPro()?`<span class="tag">${a.ownerAccess?'Owner access':'Active on this account'}</span><div class="field billing-paths"><label for="billing-path-count">Paths for the next full run</label><input id="billing-path-count" type="number" inputmode="numeric" min="4" max="${MAX_SIMULATION_PATHS}" step="1" data-field="numberOfSimulations" data-type="number" value="${escapeHTML(current().numberOfSimulations)}"></div>${portalAction}${ownerTestCheckout}`:`${portalAction}${choices}${!a.checkoutAvailable?'<p class="form-note">Paid access is not configured yet. Stripe confirms the final price before payment.</p>':''}`;
  return `${pageHead('Plans & billing','Choose the number of Monte Carlo paths for your plan.')}${notice()}${card('Your account',account,'account-card')}<div class="grid two">${isPro()?'':card('Free preview',`<div class="metric-value">4 paths</div><p>A quick look at possible lifetimes. The preview uses only four simulations.</p><p>Your scenarios and calculations stay on this device.</p>`)}${card('Pro',`<div class="metric-value">${isPro()?'10,000 paths by default':'Up to 10,000 paths'}</div><p>${isPro()?'Adjust the path count for a full run or use planning targets.':'Choose a higher path count for full runs and comparisons, and use planning targets.'} Calculations still run on your device.</p>${proActions}`)}</div><p class="billing-footnote">Subscriptions renew automatically until canceled. Manage cancellation in Plans &amp; billing → Manage billing; Stripe shows the effective date. <a href="./terms.html#subscriptions">Subscription terms</a> · <a href="./support.html#refunds">Refund requests</a> · <a href="./privacy.html">Privacy</a> · <a href="./support.html">Contact support</a></p><p class="billing-footnote">Subscriptions are linked to the account used to sign in. Link accounts explicitly to share a subscription. Stripe handles payment details; this Site does not receive card numbers or your retirement scenarios. Browser-side calculation limits can be bypassed by changing local code.</p>`;
}
function reports(){const text=reportText(current(),result());return `${pageHead('Monte Carlo reports & backup','Export the current plan, its simulation assumptions, and its results.',`<button class="secondary" data-action="print">Print / save PDF</button>`)}${notice()}<div class="grid two">${card('Current plan report',`<div class="actions"><button class="secondary" data-action="download-report">Download text</button><button class="secondary" data-action="copy-report">Copy report</button></div><pre class="report-text">${escapeHTML(text)}</pre>`)}${card('Scenario backup',`<p>Save all plans in one JSON file or restore a previous backup.</p><div class="actions"><button class="primary" data-action="export-backup">Export JSON</button><button class="secondary" data-action="import-backup">Import JSON</button></div><p class="form-note">Imports web backups or Android scenario JSON arrays. Import replaces the plans saved in this browser, so export a backup first if you want to keep them.</p>`)}</div>`;}
function render({preserveEditor=false}={}){
  const editor=preserveEditor?document.activeElement:null;
  const keepEditor=editor?.tagName==='INPUT'&&editor.id&&(editor.dataset.field||editor.dataset.budget||editor.dataset.month!==undefined);
  disposeCharts();
  const details=$('#advanced-model');if(details)state.advancedOpen=details.open;
  const allocation=$('#allocation-settings');if(allocation)state.allocationOpen=allocation.open;
  const select=$('#scenario-select');
  select.innerHTML=state.scenarios.map(s=>`<option value="${escapeHTML(s.id)}" ${s.id===state.selectedId?'selected':''}>${escapeHTML(s.name)}</option>`).join('');
  $('#run-button').disabled=state.busy;$('#run-button').textContent=state.busy?'Calculating…':'Run simulation';
  document.querySelectorAll('#navigation button').forEach(b=>{const active=b.dataset.view===state.view;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  $('#page-location').textContent=currentViewLabel();
  $('#result-state').textContent=state.busy?'Simulation running':result()?'Results up to date':'Ready to simulate';
  $('#main').innerHTML=({dashboard,setup,budget,scenarios,lab,results,reports,billing:billingView}[state.view]||dashboard)();
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
function runWorker(s,task='simulation'){return new Promise((resolve,reject)=>{const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});worker.onmessage=e=>{worker.terminate();e.data.type==='result'?resolve(e.data.result):reject(new Error(e.data.message));};worker.onerror=e=>{worker.terminate();reject(new Error((e.message||'Calculation failed')+'. Reload this page to load the latest calculator, then try again.'));};worker.postMessage({scenario:s,task});});}
async function run(){
  if(state.busy)return;
  const s=simulationScenario(),errors=validateScenario(s);
  if(errors.length){state.message='Error: '+errors.join(' ');render({preserveEditor:true});return;}
  const revision=calculationRevision;
  state.busy=true;state.message='Calculating this plan…';render({preserveEditor:true});
  try{const r=await runWorker(s);if(revision!==calculationRevision)return;state.results.set(s.id,r);state.message='Results updated for '+s.name+'.';}
  catch(e){if(revision===calculationRevision)state.message='Error: '+e.message;}
  finally{state.busy=false;render({preserveEditor:true});}
}
async function runDecision(){
  if(state.busy)return;
  if(!isPro()){setMessage('Planning targets require Pro because four paths are too coarse.');return;}
  const revision=calculationRevision,s=simulationScenario();
  state.busy=true;state.message='Finding retirement-age and spending targets…';render({preserveEditor:true});
  try{const decision=await runWorker(s,'decision');if(revision!==calculationRevision)return;state.decision=decision;state.message='Planning targets ready.';}
  catch(e){if(revision===calculationRevision)state.message='Error: '+e.message;}
  finally{state.busy=false;render({preserveEditor:true});}
}
async function runLab(){
  if(state.busy)return;
  const revision=calculationRevision,base=simulationScenario();base.numberOfSimulations=Math.min(150,base.numberOfSimulations);
  state.busy=true;state.labResults=[];state.message='Running scenario comparisons…';render({preserveEditor:true});
  try{
    for(const [label,change] of [['Current plan',()=>{}],...labVariants]){
      const s=deep(base);change(s);const errors=validateScenario(s);let row;
      try{row={label,result:errors.length?null:await runWorker(s),error:errors.join(' ')};}
      catch(e){row={label,result:null,error:e.message};}
      if(revision!==calculationRevision)return;
      state.labResults.push(row);render({preserveEditor:true});
    }
    state.message='Comparisons ready.';
  }finally{state.busy=false;render({preserveEditor:true});}
}
function download(name,text,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function closeHelp(except=null){for(const help of document.querySelectorAll('.help.open')){if(help===except)continue;help.classList.remove('open');help.querySelector('.help-trigger').setAttribute('aria-expanded','false');}}
document.addEventListener('click',e=>{const trigger=e.target instanceof Element?e.target.closest('.help-trigger'):null,help=trigger?.closest('.help'),wasOpen=help?.classList.contains('open');closeHelp();if(trigger&&!wasOpen){help.classList.add('open');trigger.setAttribute('aria-expanded','true');}});
// Replacing the SVG with a copy restarts its CSS animations for the next loop.
document.addEventListener('animationend',e=>{if(e.animationName!=='fan-cycle')return;const figure=e.target.closest('.forecast-illustration'),svg=figure?.querySelector('svg');if(!svg)return;illustrationStarted=Date.now();figure.style.setProperty('--elapsed','0ms');svg.replaceWith(svg.cloneNode(true));});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeHelp();if(document.activeElement?.classList.contains('help-trigger'))document.activeElement.blur();}});
$('#run-button').addEventListener('click',()=>{if(state.busy)return;state.view='results';window.scrollTo(0,0);run();});
$('#scenario-select').addEventListener('change',e=>{selectScenario(e.target.value);persist();render();});
$('#menu-toggle').addEventListener('click',()=>{const open=$('#menu-toggle').getAttribute('aria-expanded')!=='true';$('#menu-toggle').setAttribute('aria-expanded',String(open));$('.sidebar').classList.toggle('menu-open',open);});
$('#navigation').addEventListener('click',e=>{const button=e.target.closest('[data-view]');if(!button)return;state.view=button.dataset.view;$('#menu-toggle').setAttribute('aria-expanded','false');$('.sidebar').classList.remove('menu-open');state.message='';render();window.scrollTo(0,0);});
$('#main').addEventListener('click',async e=>{const runPlan=e.target.closest('[data-action=run-plan]');if(runPlan){state.view='results';render();window.scrollTo(0,0);await run();return;}const nav=e.target.closest('[data-view]');if(nav){state.view=nav.dataset.view;render();window.scrollTo(0,0);return;}const el=e.target.closest('[data-action]');if(!el)return;const a=el.dataset.action,s=current();
  if(a==='export-unreadable-backup'){if(savedStoredRaw!==null)download('retirement-unreadable-backup.json',savedStoredRaw,'application/json');return;}
  if(a==='replace-unreadable-plans'){if(savedLoadError&&confirm('Replace the unreadable saved plans with the plans currently shown? Export the unreadable backup first if you want to keep it.')){if(persist(true,true))setMessage('Replacement plans saved.');else render();}return;}
  if(a==='start-plan'){state.setupSection=0;state.view='setup';persist();render();window.scrollTo(0,0);return;}
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
      if(a.includes('link')&&verified)setMessage('Sign-in methods linked to the same subscription.');
      else if(a==='social-signout'&&verified)setMessage('Signed out of Google.');
    }catch(error){setMessage('Error: '+(error?.message||'Sign-in failed. Please try again.'));}
    finally{el.disabled=false;}
    return;
  }
  if(a==='setup-section'){state.setupSection=Number(el.dataset.index);render();$('#section-title').focus();$('#main').scrollIntoView({block:'start'});return;}
  if(a==='reset-assumptions'){if(!confirm('Restore sample assumptions for this scenario?'))return;invalidateExploration();const fresh=baseScenario();if(isPro())applyProSimulationDefault(fresh);fresh.id=s.id;fresh.name=s.name;state.scenarios[state.scenarios.indexOf(s)]=fresh;state.results.delete(s.id);persist();setMessage('Sample assumptions restored.');}
  if(a==='new-scenario'){const copy=deep(s);copy.id='plan-'+Date.now();copy.name=s.name+' copy';state.scenarios.push(copy);selectScenario(copy.id);state.view='setup';persist();setMessage('Scenario copied. Adjust its assumptions.');}
  if(a==='select-scenario'){selectScenario(el.dataset.id);state.view='dashboard';persist();render();}
  if(a==='rename-scenario'){const target=state.scenarios.find(x=>x.id===el.dataset.id),name=prompt('Scenario name',target.name);if(name?.trim()){target.name=name.trim();persist();render();}}
  if(a==='delete-scenario'){if(state.scenarios.length===1||!confirm('Delete this scenario?'))return;invalidateExploration();state.scenarios=state.scenarios.filter(x=>x.id!==el.dataset.id);state.results.delete(el.dataset.id);if(state.selectedId===el.dataset.id)state.selectedId=state.scenarios[0].id;persist();render();}
  if(a==='add-month'){markBudgetEdited(s.budget);const date=new Date();date.setDate(1);date.setMonth(date.getMonth()-1);const used=new Set(s.budget.monthlyBudgets.map(m=>m.month));let month;do{month=`${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}`;date.setMonth(date.getMonth()-1);}while(used.has(month));s.budget.monthlyBudgets.push({month,checkingSavingsBills:[],creditCardBills:[],cashAndAtmWithdrawals:0,adjustments:{}});persist();render();document.querySelector(`[data-month="${s.budget.monthlyBudgets.length-1}"][data-part=credit]`).focus();}
  if(a==='remove-month'){markBudgetEdited(s.budget);s.budget.monthlyBudgets.splice(Number(el.dataset.index),1);persist();render();}
  if(a==='housing-assumptions'){state.setupSection=3;state.view='setup';render();window.scrollTo(0,0);}
  if(a==='apply-budget'){try{applyBudgetEstimate(s);state.results.delete(s.id);invalidateExploration();persist();setMessage('Budget estimate applied. Run the plan to refresh results.');}catch(error){setMessage('Error: '+error.message);}}
  if(a==='run-lab')await runLab();
  if(a==='checkout'||a==='billing-portal'){el.disabled=true;try{const response=await fetch(a==='checkout'?'/api/billing/checkout':'/api/billing/portal',{method:'POST',credentials:'same-origin',headers:{...(await billingHeaders()),...(a==='checkout'?{'Content-Type':'application/json'}:{})},...(a==='checkout'?{body:JSON.stringify({interval:el.dataset.interval})}:{})});const data=await response.json();if(!response.ok)throw new Error(data.error||'Billing is unavailable.');const url=new URL(data.url);if(url.protocol!=='https:'||url.hostname!==(a==='checkout'?'checkout.stripe.com':'billing.stripe.com'))throw new Error('Unexpected billing link.');location.assign(url.href);}catch(error){el.disabled=false;setMessage('Error: '+error.message);}return;}
  if(a==='run-decision')await runDecision();
  if(a==='download-report')download('retirement-report.txt',reportText(s,result()),'text/plain');
  if(a==='copy-report'){try{await navigator.clipboard.writeText(reportText(s,result()));setMessage('Report copied.');}catch{setMessage('Error: Clipboard access is unavailable. Download the text report instead.');}}
  if(a==='print')window.print();
  if(a==='export-backup')download('retirement-scenarios.json',JSON.stringify({format:'retirement-readiness-lab-web-v1',scenarios:state.scenarios.map(({seed,...scenario})=>scenario)},null,2),'application/json');
  if(a==='import-backup')$('#import-file').click();
});
$('#main').addEventListener('change',e=>{const el=e.target,s=current();if(el.id==='setup-section'){state.setupSection=Number(el.value);render();return;}if(el.dataset.field){if(el.dataset.field==='numberOfSimulations'&&!isPro()){setMessage('Four paths are available in the free preview.');return;}const type=el.dataset.type,value=type==='checkbox'?el.checked:type==='select'?el.value:Number(el.value)/(type==='percent'?100:1);setPath(s,el.dataset.field,value);if(el.dataset.field==='numberOfSimulations')s.simulationPathsCustomized=true;if(el.dataset.field==='numberOfSimulations'&&(value<FREE_SIMULATION_PATHS||value>MAX_SIMULATION_PATHS||!Number.isInteger(value))){setPath(s,el.dataset.field,Math.max(FREE_SIMULATION_PATHS,Math.min(MAX_SIMULATION_PATHS,Math.round(value)||FREE_SIMULATION_PATHS)));render();}if(el.dataset.field==='spending.annualBaseSpending'){s.budget.isAppliedToAnnualBaseSpending=false;s.budget.estimateNeedsReview=true;}state.results.delete(s.id);invalidateExploration();if(persist())state.message='Saved. Run the simulation to refresh results.';if(el.dataset.field==='household.filingStatus'){render();}else{$('#result-state').textContent='Ready to simulate';refreshNotice();}return;}
  if(el.dataset.budget){markBudgetEdited(s.budget);s.budget[el.dataset.budget]=Number(el.value);persist();refreshBudget();return;}
  if(el.dataset.month!==undefined){markBudgetEdited(s.budget);const m=s.budget.monthlyBudgets[Number(el.dataset.month)],part=el.dataset.part;if(part==='month')m.month=el.value;else if(part==='cashAndAtmWithdrawals')m.cashAndAtmWithdrawals=Number(el.value);else if(part==='checking'||part==='credit')m[part==='checking'?'checkingSavingsBills':'creditCardBills']=[{id:part,name:part==='checking'?'Checking / savings spending':'Credit card purchases',monthlyAmount:Number(el.value)}];else{m.adjustments??={};m.adjustments[part]=Number(el.value);}persist();refreshBudget();}

});
$('#import-file').addEventListener('change',async e=>{const file=e.target.files?.[0];if(!file)return;try{const data=JSON.parse(await file.text()),scenarios=Array.isArray(data)?data:data.scenarios;if(!Array.isArray(scenarios)||!scenarios.length)throw new Error('No scenarios found in the file.');const normalized=normalizeScenarios(scenarios);for(const s of normalized){const errors=validateScenario(s);if(errors.length)throw new Error(`${s.name}: ${errors.join(' ')}`);}state.scenarios=normalized;if(isPro())state.scenarios.forEach(applyProSimulationDefault);state.selectedId=normalized[0].id;state.results.clear();invalidateExploration();persist(true,true);setMessage(`${normalized.length} scenarios imported.`);}catch(error){setMessage('Error: '+error.message);}e.target.value='';});
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
const ACCESS_UNAVAILABLE='Subscription status is unavailable. New runs use the four-path free preview; your completed results are kept.';
const PAYMENT_PENDING='Payment is being confirmed. Refresh your plan in a moment.';
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
      setMessage('Sign in again to verify your plan. Your completed results are still available.',{preserveEditor:true});return false;
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
    if(access.tier==='pro'){let changed=false;for(const scenario of state.scenarios)changed=applyProSimulationDefault(scenario)||changed;if(changed){persist(false);rerender=true;}}
    if(query.get('checkout')==='success')message=access.tier==='pro'?'Pro is active for this account.':PAYMENT_PENDING;
    else if(query.get('checkout')==='canceled')message='Checkout was canceled.';
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
const ACCESS_RETRY='Subscription status is temporarily unavailable. Your last verified Pro access is available for up to five minutes; your completed results are kept.';
async function bootAuth(){
  try{await initializeSocialAuth(()=>{syncAuthState();loadAccess({force:true});});syncAuthState();}
  catch{state.auth=socialState();state.message='Google sign-in is temporarily unavailable. ChatGPT sign-in still works.';}
  const accountQuery=new URLSearchParams(location.search);
  if(accountQuery.has('link')||accountQuery.has('account')||location.hash==='#billing'){
    state.view='billing';
    if(accountQuery.has('link'))state.message='To share a subscription, use the account-linking button below after both sign-ins are active.';
    accountQuery.delete('account');
    history.replaceState(null,'',location.pathname+(accountQuery.toString()?'?'+accountQuery:'')+location.hash);
  }
  await loadAccess({force:true});
}
bootAuth();
window.addEventListener('focus',()=>{if(!state.busy)loadAccess();});
