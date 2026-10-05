import {validateScenario,setAnnualBaseSpending,forecastRetirementDate,primaryRetirementAge,calendarDate,calendarMonthsBetween,addCalendarMonths,syncCalendarAges,dateLabel,ageLabel,localCalendarDate,ALLOCATION_KEYS,MAX_DOLLAR_AMOUNT,MAX_ONE_TIME_EXPENSES,FREE_SIMULATION_PATHS} from './model.js';
import {setAnnualPersonalSavings,annualPersonalSavings} from './savings-targets.js';
import {parseMoneyInput,moneyInputValue} from './money-input.js';

// Plan Lab: named what-if versions of a plan, compared on the same paths.
export const LAB_MAX_WHAT_IFS=4,LAB_FREE_WHAT_IFS=1,LAB_MAX_SETS=8,LIVE_ESTIMATE_PATHS=200,SENSITIVITY_PATHS=500;
const money=(v,d=0)=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:d}).format(v||0);
const pct=(v,d=0)=>`${v<0?'−':''}${(100*Math.abs(v||0)).toFixed(d)}%`;
const wholeAge=s=>Math.floor(primaryRetirementAge(s)+1e-9);
const int=v=>{const n=Number(String(v??'').trim());return String(v??'').trim()!==''&&Number.isInteger(n)?n:null;};
const amount=v=>{const n=parseMoneyInput(String(v??''));return String(v??'').trim()!==''&&Number.isFinite(n)&&n>=0&&n<=MAX_DOLLAR_AMOUNT?n:null;};
const choice=(options,v)=>options.find(([value])=>String(value)===String(v))?.[0];
const retired=s=>s.household.alreadyRetired;
// Presets follow the plan; dates explicitly selected in the editor stay fixed.
const retirementDelay=v=>v!==null&&typeof v==='object'&&!Array.isArray(v)&&Number.isInteger(v.delayMonths)&&v.delayMonths>0&&v.delayMonths<=1200;
const retirementDateValue=(s,v)=>retirementDelay(v)?addCalendarMonths(forecastRetirementDate(s),v.delayMonths):v;
const retirementDelayLabel=v=>v.delayMonths%12===0?`Retire ${v.delayMonths/12} year${v.delayMonths===12?'':'s'} later`:`Retire ${v.delayMonths} month${v.delayMonths===1?'':'s'} later`;

const ORDER_LABELS={Standard:'Pre-tax, then Roth, then taxable',TaxableFirst:'Taxable, then pre-tax, then Roth',RothLast:'Pre-tax, then taxable, Roth last'};
const ROTH_OPTIONS=[['off','Off'],['0.12','Up to the 12% bracket'],['0.22','Up to the 22% bracket'],['0.24','Up to the 24% bracket']];
const CASH_OPTIONS=[['off','Off'],['-0.01','Months below −1%'],['-0.05','Months below −5%'],['-0.1','Months below −10%']];
const MIX_OPTIONS=[-30,-20,-10,10,20,30].map(n=>[String(n),`${n>0?'+':'−'}${Math.abs(n)} points of stocks`]);
const CAP_OPTIONS=[.05,.06,.07,.08,.09,.10].map(n=>[String(n),`Stocks and pre-retirement growth up to ${pct(n)}`]);
const HEALTH_OPTIONS=[-.25,.25,.5].map(n=>[String(n),`${n>0?'+':'−'}${pct(Math.abs(n))} premiums and care costs`]);
const PATTERN_OPTIONS=[['Flat','Steady before inflation'],['EmpiricalAgeDecline','Gradual decline from 65 to 85']];
const HOME_OPTIONS=[['downsize','Downsize to a smaller home'],['rent','Sell and rent']];
const ALLOCATION_BAND_NAMES=['Under 30×','30–35×','35–40×','40–45×','45–50×','50× or more'];
const stockPercent=v=>String(Number((100*v).toPrecision(15)));
const schedulePercents=a=>ALLOCATION_KEYS.map(k=>stockPercent(a[k])).join('/')+'%';

// Each lever reads a draft of strings, writes one change and describes it.
// Applying a lever never edits the saved plan; it edits a comparison copy.
export const LAB_LEVERS=[
  {key:'retirementDate',group:'Timing',label:'Retirement date',available:s=>!retired(s),
    current:s=>`${dateLabel(forecastRetirementDate(s))} · age ${ageLabel(primaryRetirementAge(s))}`,
    fields:(s,v)=>[{name:'date',label:'Retirement date to test',type:'date',value:retirementDateValue(s,v??forecastRetirementDate(s)),min:localCalendarDate()}],
    read:(d,s,v)=>{const date=d.date??'';return calendarDate(date)&&date>=localCalendarDate()?{value:retirementDelay(v)&&date===retirementDateValue(s,v)?structuredClone(v):date}:{error:'Choose a retirement date today or later.'};},
    apply:(s,v)=>{s.household.retirementDate=retirementDateValue(s,v);s.household.alreadyRetired=false;syncCalendarAges(s);},
    describe:(v,s)=>{const date=retirementDateValue(s,v);return `Retire ${dateLabel(date)}${s.household.birthday?` · age ${ageLabel(calendarMonthsBetween(s.household.birthday,date)/12)}`:''}`;}},
  {key:'claimAge',group:'Timing',label:'Social Security claiming age',available:()=>true,
    current:s=>`Age ${s.socialSecurity.claimAge}`,
    fields:(s,v)=>[{name:'age',label:'Claiming age',type:'select',value:String(v??s.socialSecurity.claimAge),options:[62,63,64,65,66,67,68,69,70].map(a=>[String(a),'Age '+a])}],
    read:d=>{const age=int(d.age);return age>=62&&age<=70?{value:age}:{error:'Choose a claiming age from 62 to 70.'};},
    apply:(s,v)=>{s.socialSecurity.claimAge=v;},describe:v=>`Claim Social Security at ${v}`},
  {key:'partTimeIncome',group:'Timing',label:'Part-time work after retiring',isNew:true,available:()=>true,
    current:s=>s.partTimeIncome.annualNet>0&&s.partTimeIncome.endAge>0?`${money(s.partTimeIncome.annualNet)} a year until ${s.partTimeIncome.endAge}`:'None',
    fields:(s,v)=>[{name:'annualNet',label:'Take-home pay a year, today’s dollars',type:'money',value:v?moneyInputValue(v.annualNet):''},{name:'endAge',label:'Stop working at age',type:'number',value:v?String(v.endAge):String(Math.min(118,wholeAge(s)+3))}],
    read:(d,s)=>{const net=amount(d.annualNet),end=int(d.endAge);return net===null?{error:'Enter yearly take-home pay of 0 or more.'}:end===null||end<=primaryRetirementAge(s)||end>=s.household.targetEndAge?{error:'Enter a whole-year age after your retirement and before the maximum modeling age.'}:{value:{annualNet:net,endAge:end}};},
    apply:(s,v)=>{s.partTimeIncome={annualNet:v.annualNet,endAge:v.endAge};},describe:v=>v.annualNet>0?`Work part-time for ${money(v.annualNet)} a year until ${v.endAge}`:'No part-time work'},
  {key:'annualBaseSpending',group:'Spending',label:'Base spending',available:()=>true,
    current:s=>`${money(s.spending.annualBaseSpending)} a year`,
    fields:(s,v)=>[{name:'annual',label:'Annual base spending, today’s dollars',type:'money',value:moneyInputValue(v??s.spending.annualBaseSpending)}],
    read:d=>{const n=amount(d.annual);return n===null?{error:'Enter annual base spending of 0 or more.'}:{value:n};},
    apply:(s,v)=>setAnnualBaseSpending(s,v),describe:v=>`Spend ${money(v)} a year`},
  {key:'spendingPathModel',group:'Spending',label:'Spending pattern',available:()=>true,
    current:s=>PATTERN_OPTIONS.find(([k])=>k===s.spending.spendingPathModel)?.[1]??s.spending.spendingPathModel,
    fields:(s,v)=>[{name:'pattern',label:'Spending pattern',type:'select',value:v??s.spending.spendingPathModel,options:PATTERN_OPTIONS}],
    read:d=>{const v=choice(PATTERN_OPTIONS,d.pattern);return v?{value:v}:{error:'Choose a spending pattern.'};},
    apply:(s,v)=>{s.spending.spendingPathModel=v;},describe:v=>v==='Flat'?'Steady spending':'Spending eases from 65 to 85'},
  {key:'oneTimeExpenses',group:'Spending',label:'One-time expenses',isNew:true,available:()=>true,
    current:s=>s.oneTimeExpenses.length?`${s.oneTimeExpenses.length} planned · ${money(s.oneTimeExpenses.reduce((n,x)=>n+x.amount,0))}`:'None',
    fields:(s,v)=>{const rows=v??s.oneTimeExpenses,out=[];for(let i=0;i<3;i++){const r=rows[i];out.push({name:'label'+i,label:`Expense ${i+1} name`,type:'text',value:r?.label??'',row:i},{name:'age'+i,label:'At age',type:'number',value:r?String(r.age):'',row:i},{name:'amount'+i,label:'Amount, today’s dollars',type:'money',value:r?moneyInputValue(r.amount):'',row:i});}return out;},
    read:(d,s)=>{const rows=[];for(let i=0;i<3;i++){const label=String(d['label'+i]??'').trim().slice(0,60),a=String(d['age'+i]??'').trim(),m=String(d['amount'+i]??'').trim();if(!label&&!a&&!m)continue;const age=int(a),cost=amount(m);if(age===null||age<=primaryRetirementAge(s)||age>=s.household.targetEndAge||cost===null)return {error:`Expense ${i+1}: enter a whole-year age after your retirement and an amount of 0 or more.`};rows.push({label,age,amount:cost});}return rows.length?{value:rows.slice(0,MAX_ONE_TIME_EXPENSES)}:{error:'Enter at least one expense, or remove this change.'};},
    apply:(s,v)=>{s.oneTimeExpenses=v.map(x=>({...x}));},describe:v=>v.map(x=>`${x.label||'Expense'} ${money(x.amount)} at ${x.age}`).join(', ')},
  {key:'annualSavings',group:'Saving & investing',label:'Annual savings until retirement',available:s=>!retired(s),
    current:s=>`${money(annualPersonalSavings(s))} a year`,
    fields:(s,v)=>[{name:'annual',label:'Your yearly savings',type:'money',value:moneyInputValue(v??annualPersonalSavings(s))}],
    read:d=>{const n=amount(d.annual);return n===null?{error:'Enter yearly savings of 0 or more.'}:{value:n};},
    apply:(s,v)=>setAnnualPersonalSavings(s,v),describe:v=>`Save ${money(v)} a year`},
  // A whole schedule, such as one from the stock allocation optimizer. A stock
  // and bond mix change applies on top of it.
  {key:'stockAllocation',group:'Saving & investing',label:'Stock allocation by portfolio size',isNew:true,available:()=>true,
    current:s=>schedulePercents(s.postRetirementAllocation),
    fields:(s,v)=>ALLOCATION_KEYS.map((k,i)=>({name:k,label:`${ALLOCATION_BAND_NAMES[i]} spending, % stocks`,type:'number',inputmode:'decimal',value:stockPercent((v??s.postRetirementAllocation)[k])})),
    read:(d,s,v)=>{const out={},original=v??s?.postRetirementAllocation;for(const [i,k] of ALLOCATION_KEYS.entries()){const raw=String(d[k]??'').trim(),n=Number(raw);if(!raw||!Number.isFinite(n)||n<0||n>100)return {error:`${ALLOCATION_BAND_NAMES[i]} spending: enter a percentage from 0 to 100.`};out[k]=original&&raw===stockPercent(original[k])?original[k]:n/100;}return {value:out};},
    apply:(s,v)=>{for(const k of ALLOCATION_KEYS)s.postRetirementAllocation[k]=v[k];},
    describe:v=>`Stocks ${schedulePercents(v)} by savings level`},
  {key:'stockShift',group:'Saving & investing',label:'Stock and bond mix',available:()=>true,
    current:s=>`${pct(s.postRetirementAllocation.stockUnder30x)} stocks at the start of retirement`,
    fields:(s,v)=>[{name:'shift',label:'Change after retirement',type:'select',value:String(v??-10),options:MIX_OPTIONS}],
    read:d=>{const v=choice(MIX_OPTIONS,d.shift);return v?{value:Number(v)}:{error:'Choose a change in the stock share.'};},
    apply:(s,v)=>{for(const k of ALLOCATION_KEYS)s.postRetirementAllocation[k]=Math.min(1,Math.max(0,Math.round((s.postRetirementAllocation[k]+v/100)*100)/100));},
    describe:v=>`${v>0?'More':'Fewer'} stocks (${v>0?'+':'−'}${Math.abs(v)} points)`},
  {key:'returnCap',group:'Saving & investing',label:'Lower return assumption',available:()=>true,
    current:s=>`Stocks ${pct(s.market.stockMeanReturn,1)} · before retirement ${pct(s.market.preRetirementMeanReturn,1)}`,
    fields:(s,v)=>[{name:'cap',label:'Average returns up to',type:'select',value:String(v??.07),options:CAP_OPTIONS}],
    read:d=>{const v=choice(CAP_OPTIONS,d.cap);return v?{value:Number(v)}:{error:'Choose a return cap.'};},
    apply:(s,v)=>{s.market.preRetirementMeanReturn=Math.min(s.market.preRetirementMeanReturn,v);s.market.stockMeanReturn=Math.min(s.market.stockMeanReturn,v);},describe:v=>`Returns up to ${pct(v)}`},
  {key:'rothConversion',group:'Taxes & withdrawals',label:'Roth conversions',available:()=>true,
    current:s=>s.rothConversion.enabled?`Up to the ${pct(s.rothConversion.marginalRateCap)} bracket`:'Off',
    fields:(s,v)=>[{name:'cap',label:'Convert each year',type:'select',value:v?(v.enabled?String(v.marginalRateCap):'off'):(s.rothConversion.enabled?'off':'0.22'),options:ROTH_OPTIONS}],
    read:d=>{const v=choice(ROTH_OPTIONS,d.cap);return v===undefined?{error:'Choose a conversion setting.'}:{value:v==='off'?{enabled:false,marginalRateCap:.22}:{enabled:true,marginalRateCap:Number(v)}};},
    apply:(s,v)=>{s.rothConversion={enabled:v.enabled,marginalRateCap:v.marginalRateCap};},describe:v=>v.enabled?`Roth conversions to the ${pct(v.marginalRateCap)} bracket`:'No Roth conversions'},
  {key:'withdrawalOrder',group:'Taxes & withdrawals',label:'Withdrawal order',isNew:true,available:()=>true,
    current:s=>ORDER_LABELS[s.withdrawalStrategy.withdrawalOrder]??ORDER_LABELS.Standard,
    fields:(s,v)=>[{name:'order',label:'Take withdrawals from',type:'select',value:v??(s.withdrawalStrategy.withdrawalOrder==='TaxableFirst'?'RothLast':'TaxableFirst'),options:Object.entries(ORDER_LABELS)}],
    read:d=>{const v=choice(Object.entries(ORDER_LABELS),d.order);return v?{value:v}:{error:'Choose a withdrawal order.'};},
    apply:(s,v)=>{s.withdrawalStrategy.withdrawalOrder=v;},describe:v=>ORDER_LABELS[v]},
  {key:'cashFirst',group:'Taxes & withdrawals',label:'Use cash in down months',available:()=>true,
    current:s=>s.withdrawalStrategy.useCashReserveDuringDrawdowns?`Months below ${pct(s.withdrawalStrategy.drawdownTrigger)}`:'Off',
    fields:(s,v)=>[{name:'trigger',label:'Spend cash first in',type:'select',value:v?(v.enabled?String(v.drawdownTrigger):'off'):'-0.01',options:CASH_OPTIONS}],
    read:d=>{const v=choice(CASH_OPTIONS,d.trigger);return v===undefined?{error:'Choose when to use cash.'}:{value:v==='off'?{enabled:false,drawdownTrigger:-.01}:{enabled:true,drawdownTrigger:Number(v)}};},
    apply:(s,v)=>{s.withdrawalStrategy.useCashReserveDuringDrawdowns=v.enabled;s.withdrawalStrategy.drawdownTrigger=v.drawdownTrigger;},describe:v=>v.enabled?`Cash first in months below ${pct(v.drawdownTrigger)}`:'No cash-first rule'},
  {key:'homePlan',group:'Home & health',label:'Sell or downsize home',isNew:true,available:s=>s.home.currentValue>0,
    current:s=>s.homePlan.saleAge>0?(s.homePlan.mode==='rent'?`Sell at ${s.homePlan.saleAge} and rent`:`Downsize at ${s.homePlan.saleAge}`):'Keep home',
    fields:(s,v)=>[{name:'saleAge',label:'Sell at age',type:'number',value:String(v?.saleAge??Math.min(s.household.targetEndAge-1,Math.max(wholeAge(s)+1,75)))},{name:'mode',label:'Then',type:'select',value:v?.mode??'downsize',options:HOME_OPTIONS},{name:'share',label:'Smaller home price, % of sale',type:'number',value:String(Math.round(100*(v?.downsizeShare??.5)))},{name:'rent',label:'Monthly rent, today’s dollars',type:'money',value:moneyInputValue(v?.monthlyRent??0)}],
    read:(d,s)=>{const age=int(d.saleAge),mode=choice(HOME_OPTIONS,d.mode),share=Number(d.share),rent=amount(d.rent);
      if(age===null||age<=primaryRetirementAge(s)||age>=s.household.targetEndAge)return {error:'Enter a whole-year sale age after your retirement and before the maximum modeling age.'};
      if(!mode)return {error:'Choose downsize or rent.'};
      if(mode==='downsize'&&!(share>=0&&share<=95))return {error:'Enter a smaller home price from 0% to 95% of the sale price.'};
      if(mode==='rent'&&rent===null)return {error:'Enter monthly rent of 0 or more.'};
      return {value:{saleAge:age,mode,downsizeShare:mode==='downsize'?Math.round(share)/100:.5,monthlyRent:mode==='rent'?rent:0}};},
    apply:(s,v)=>{s.homePlan={...v};},describe:v=>v.mode==='rent'?`Sell home at ${v.saleAge} and rent for ${money(v.monthlyRent)}/mo`:`Downsize at ${v.saleAge} to a ${pct(v.downsizeShare)} home`},
  {key:'healthcareChange',group:'Home & health',label:'Healthcare costs',available:()=>true,
    current:s=>`${money(s.healthcare.preMedicareMonthlyPremium)}/mo before Medicare · care ${money(s.longTermCare.annualCost)}/yr`,
    fields:(s,v)=>[{name:'change',label:'Change',type:'select',value:String(v??.25),options:HEALTH_OPTIONS}],
    read:d=>{const v=choice(HEALTH_OPTIONS,d.change);return v?{value:Number(v)}:{error:'Choose a healthcare change.'};},
    apply:(s,v)=>{s.healthcare.preMedicareMonthlyPremium*=1+v;s.longTermCare.annualCost*=1+v;},describe:v=>`Healthcare costs ${v>0?'+':'−'}${pct(Math.abs(v))}`}
];
export const LAB_LEVER_GROUPS=['Timing','Spending','Saving & investing','Taxes & withdrawals','Home & health'];
const leverMap=new Map(LAB_LEVERS.map(l=>[l.key,l]));
export const labLever=key=>leverMap.get(key);

// One-click starting points; the first six match the earlier quick comparisons.
export const LAB_PRESETS=[
  {key:'later',label:'Retire 2 years later',changes:s=>retired(s)?null:{retirementDate:{delayMonths:24}}},
  {key:'spend-less',label:'Spend 5% less',changes:s=>({annualBaseSpending:Math.round(s.spending.annualBaseSpending*.95)})},
  {key:'claim-70',label:'Claim Social Security at 70',changes:s=>s.socialSecurity.claimAge===70?null:{claimAge:70}},
  {key:'health',label:'Higher healthcare costs',changes:()=>({healthcareChange:.25})},
  {key:'roth',label:'Use Roth conversions',changes:s=>s.rothConversion.enabled?null:{rothConversion:{enabled:true,marginalRateCap:.22}}},
  {key:'cash',label:'Use cash first in months below −1%',changes:s=>s.withdrawalStrategy.useCashReserveDuringDrawdowns?null:{cashFirst:{enabled:true,drawdownTrigger:-.01}}},
  {key:'lower-returns',label:'Lower returns · up to 7%',changes:s=>Math.max(s.market.stockMeanReturn,s.market.preRetirementMeanReturn)>.07?{returnCap:.07}:null},
  {key:'taxable-first',label:'Withdraw from taxable savings first',changes:s=>s.accounts.taxable>0&&s.withdrawalStrategy.withdrawalOrder!=='TaxableFirst'?{withdrawalOrder:'TaxableFirst'}:null}
];

export function applyWhatIf(base,changes={}){
  const s=structuredClone(base);
  for(const lever of LAB_LEVERS)if(Object.hasOwn(changes,lever.key))lever.apply(s,changes[lever.key]);
  return s;
}
export function whatIfErrors(base,changes){
  const unavailable=Object.keys(changes).filter(key=>!labLever(key)?.available(base)).map(key=>`${labLever(key)?.label??key} does not apply to this plan.`);
  return unavailable.length?unavailable:validateScenario(applyWhatIf(base,changes));
}
export const describeChanges=(changes,s)=>LAB_LEVERS.filter(l=>Object.hasOwn(changes,l.key)).map(l=>l.describe(changes[l.key],s));
export function autoName(changes,s){const parts=LAB_LEVERS.filter(l=>Object.hasOwn(changes,l.key)).map(l=>l.key==='retirementDate'&&retirementDelay(changes[l.key])?retirementDelayLabel(changes[l.key]):l.describe(changes[l.key],s));return parts.length?parts.join(' + ').slice(0,80):'New what-if';}

let idCounter=0;
export const labId=prefix=>`${prefix}-${Date.now().toString(36)}-${(++idCounter).toString(36)}${Math.random().toString(36).slice(2,6)}`;
export function newWhatIf(s,changes={},name=''){return {id:labId('whatif'),name:name||autoName(changes,s),changes:structuredClone(changes)};}
// Pro sets start with three common changes; the free set starts empty so its
// single what-if is the visitor's own choice.
export function defaultLabSets(s,pro){
  const presets=(pro?['later','spend-less','claim-70']:[]).map(key=>LAB_PRESETS.find(p=>p.key===key)).map(p=>({p,changes:p.changes(s)})).filter(x=>x.changes);
  const set={id:labId('set'),name:'My comparisons',whatIfs:presets.map(({p,changes})=>newWhatIf(s,changes,p.label))};
  return {activeSet:set.id,sets:[set]};
}

// Saved comparison sets are user data. Keep only well-formed entries with known
// levers; a malformed set never blocks loading the plans themselves.
const savedObject=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const savedAmount=v=>Number.isFinite(v)&&v>=0&&v<=MAX_DOLLAR_AMOUNT;
const savedAge=v=>Number.isInteger(v)&&v>=0&&v<120;
const savedChoice=(options,v)=>options.some(([value])=>Number(value)===v);
// Check storage shape and supported values independently of the current plan:
// a valid older date or expense still needs to be available for editing.
const CHANGE_VALIDATORS={
  retirementDate:v=>typeof v==='string'&&Boolean(calendarDate(v))||retirementDelay(v),
  claimAge:v=>Number.isInteger(v)&&v>=62&&v<=70,
  partTimeIncome:v=>savedObject(v)&&savedAmount(v.annualNet)&&savedAge(v.endAge),
  annualBaseSpending:savedAmount,
  spendingPathModel:v=>PATTERN_OPTIONS.some(([value])=>value===v),
  oneTimeExpenses:v=>Array.isArray(v)&&v.length<=MAX_ONE_TIME_EXPENSES&&v.every(x=>savedObject(x)&&savedAge(x.age)&&savedAmount(x.amount)&&typeof (x.label??'')==='string'),
  annualSavings:savedAmount,
  stockAllocation:v=>savedObject(v)&&ALLOCATION_KEYS.every(k=>Number.isFinite(v[k])&&v[k]>=0&&v[k]<=1),
  stockShift:v=>savedChoice(MIX_OPTIONS,v),
  returnCap:v=>savedChoice(CAP_OPTIONS,v),
  rothConversion:v=>savedObject(v)&&typeof v.enabled==='boolean'&&savedChoice(ROTH_OPTIONS.slice(1),v.marginalRateCap),
  withdrawalOrder:v=>typeof v==='string'&&Object.hasOwn(ORDER_LABELS,v),
  cashFirst:v=>savedObject(v)&&typeof v.enabled==='boolean'&&savedChoice(CASH_OPTIONS.slice(1),v.drawdownTrigger),
  homePlan:v=>savedObject(v)&&savedAge(v.saleAge)&&HOME_OPTIONS.some(([mode])=>mode===v.mode)&&Number.isFinite(v.downsizeShare)&&v.downsizeShare>=0&&v.downsizeShare<=.95&&savedAmount(v.monthlyRent),
  healthcareChange:v=>savedChoice(HEALTH_OPTIONS,v)
};
function cleanChanges(raw){
  if(!raw||typeof raw!=='object'||Array.isArray(raw))return {};
  const out={};for(const [key,value] of Object.entries(raw))if(Object.hasOwn(CHANGE_VALIDATORS,key)&&CHANGE_VALIDATORS[key](value))out[key]=structuredClone(value);
  return out;
}
function savedWhatIf(w){
  const name=typeof w.name==='string'&&w.name.trim()?w.name.trim().slice(0,80):'What-if',changes=cleanChanges(w.changes);
  // Older built-in later comparisons stored an absolute date while retaining
  // their relative name. Restore that preset's intended meaning on load.
  if(name==='Retire 2 years later'&&typeof changes.retirementDate==='string')changes.retirementDate={delayMonths:24};
  return {id:w.id,name,changes};
}
export function normalizeLabSets(raw,scenarios){
  return Object.fromEntries(scenarios.flatMap(s=>{
    const entry=raw&&typeof raw==='object'&&Object.hasOwn(raw,s.id)?raw[s.id]:null;
    if(!entry||!Array.isArray(entry.sets))return [];
    const ids=new Set();
    const sets=entry.sets.filter(x=>x&&typeof x==='object'&&typeof x.id==='string'&&!ids.has(x.id)&&ids.add(x.id)).slice(0,LAB_MAX_SETS).map(x=>{
      const whatIds=new Set();
      return {id:x.id,name:typeof x.name==='string'&&x.name.trim()?x.name.trim().slice(0,60):'Comparison set',whatIfs:(Array.isArray(x.whatIfs)?x.whatIfs:[]).filter(w=>w&&typeof w==='object'&&typeof w.id==='string'&&!whatIds.has(w.id)&&whatIds.add(w.id)).slice(0,LAB_MAX_WHAT_IFS).map(savedWhatIf)};
    });
    if(!sets.length)return [];
    return [[s.id,{activeSet:sets.some(x=>x.id===entry.activeSet)?entry.activeSet:sets[0].id,sets}]];
  }));
}
export function copyLabSets(entry){
  if(!entry)return null;const copy=structuredClone(entry);const map=new Map();
  for(const set of copy.sets){const id=labId('set');map.set(set.id,id);set.id=id;for(const w of set.whatIfs)w.id=labId('whatif');}
  copy.activeSet=map.get(copy.activeSet)??copy.sets[0]?.id;return copy;
}

// Rows compare readiness as points, or as counts for a 100-path preview.
export function readinessDelta(result,base){
  if(!result||!base)return '';
  const n=result.provenance.simulationCount;
  if(n<=FREE_SIMULATION_PATHS){const d=Math.round(result.successProbability*n)-Math.round(base.successProbability*n);return d===0?'Same':`${d>0?'+':'−'}${Math.abs(d)} of ${n}`;}
  const d=Math.round((result.successProbability-base.successProbability)*1000)/10;
  return d===0?'Same':`${d>0?'+':'−'}${Math.abs(d)%1?Math.abs(d).toFixed(1):Math.abs(d)} pts`;
}
export function labMetrics(result,shown){
  const n=result.provenance.simulationCount,lab=result.planLab;
  return {readiness:result.successProbability,shortfalls:Math.round((1-result.successProbability)*n),paths:n,medianLeft:shown.medianEndingBalance,toughLeft:shown.pessimisticEndingBalance,medianFailureAge:result.medianFailureAge,lifetimeTax:lab?.lifetimeTax??null,surchargeYears:lab?.surchargeYears??null,conversions:lab?.conversions??null};
}

// The one-paragraph answer at the top of the page.
export function labSummary(rows,{stress=null,label=r=>r.label}={}){
  const done=rows.filter(r=>r.result),base=done.find(r=>r.baseline),others=done.filter(r=>!r.baseline);
  if(!base||!others.length)return null;
  const p=r=>r.result.successProbability,better=(a,b)=>p(a)>p(b)||p(a)===p(b)&&a.result.medianEndingBalance>b.result.medianEndingBalance;
  const best=others.reduce((a,b)=>better(b,a)?b:a);
  const singles=others.filter(r=>Object.keys(r.changes||{}).length===1),single=singles.length?singles.reduce((a,b)=>better(b,a)?b:a):null;
  const taxes=done.filter(r=>r.result.planLab),lowTax=taxes.length>1?taxes.reduce((a,b)=>b.result.planLab.lifetimeTax<a.result.planLab.lifetimeTax?b:a):null;
  let resilient=null;
  if(stress?.rows?.length){const scores=new Map();for(const row of stress.rows)for(const [id,r] of Object.entries(row.results||{}))if(r)scores.set(id,(scores.get(id)||[]).concat(r.successProbability));const ranked=[...scores].map(([id,v])=>({id,avg:v.reduce((a,b)=>a+b,0)/v.length,n:v.length})).filter(x=>x.n===stress.rows.length).sort((a,b)=>b.avg-a.avg);if(ranked.length)resilient={row:done.find(r=>(r.id??'baseline')===ranked[0].id),average:ranked[0].avg};}
  const improved=p(best)>p(base);
  return {best,base,improved,delta:readinessDelta(best.result,base.result),single:single&&p(single)>p(base)?{row:single,delta:readinessDelta(single.result,base.result)}:null,
    lowTax:lowTax&&lowTax!==base&&base.result.planLab&&lowTax.result.planLab.lifetimeTax<base.result.planLab.lifetimeTax?{row:lowTax,saving:base.result.planLab.lifetimeTax-lowTax.result.planLab.lifetimeTax}:null,
    resilient:resilient?.row?resilient:null,headline:improved?`${label(best)} gives the highest readiness.`:'None of these what-ifs beats your current plan.'};
}

// Stress tests keep every path and add one forced event.
export const STRESS_TESTS=[
  {key:'market-drop',label:'Stocks fall 30% when you retire',note:'Sequence-of-returns risk in the first month of retirement',options:{stress:{marketDrop:.3}}},
  {key:'inflation',label:'Ten years of 5% inflation',note:'General inflation from the start of retirement',options:{stress:{inflationShock:{rate:.05,months:120}}}},
  {key:'low-returns',label:'Stocks average 7% a year',note:'Stock and pre-retirement averages capped at 7%',change:s=>{s.market.preRetirementMeanReturn=Math.min(s.market.preRetirementMeanReturn,.07);s.market.stockMeanReturn=Math.min(s.market.stockMeanReturn,.07);}},
  {key:'care',label:'Long-term care at the end of every life',note:'Uses your care cost and duration',options:{stress:{forceCare:true}}},
  {key:'long-life',label:'Everyone lives to at least 100',note:'Within your maximum modeling age',options:{stress:{minDeathAge:100}}}
];

// Each input moves both ways from the plan; rows rank by their total range.
const shiftDate=(s,months)=>{if(retired(s))return false;const date=addCalendarMonths(forecastRetirementDate(s),months);if(date<localCalendarDate())return false;s.household.retirementDate=date;s.household.alreadyRetired=false;syncCalendarAges(s);return true;};
const shiftMix=(s,points)=>{for(const k of ALLOCATION_KEYS)s.postRetirementAllocation[k]=Math.min(1,Math.max(0,s.postRetirementAllocation[k]+points));return true;};
export const SENSITIVITY_INPUTS=[
  {key:'returns',label:'Stock returns',range:'2 points lower / higher',control:false,low:s=>{s.market.stockMeanReturn=Math.max(-.2,s.market.stockMeanReturn-.02);s.market.preRetirementMeanReturn=Math.max(-.2,s.market.preRetirementMeanReturn-.02);return true;},high:s=>{s.market.stockMeanReturn=Math.min(.25,s.market.stockMeanReturn+.02);s.market.preRetirementMeanReturn=Math.min(.25,s.market.preRetirementMeanReturn+.02);return true;}},
  {key:'retirement',label:'Retirement date',range:'2 years earlier / later',control:true,low:s=>shiftDate(s,-24),high:s=>shiftDate(s,24),whatIf:(s,side)=>({retirementDate:addCalendarMonths(forecastRetirementDate(s),side==='high'?24:-24)})},
  {key:'spending',label:'Base spending',range:'10% more / less',control:true,low:s=>{setAnnualBaseSpending(s,Math.round(s.spending.annualBaseSpending*1.1));return true;},high:s=>{setAnnualBaseSpending(s,Math.round(s.spending.annualBaseSpending*.9));return true;},whatIf:(s,side)=>({annualBaseSpending:Math.round(s.spending.annualBaseSpending*(side==='high'?.9:1.1))})},
  {key:'inflation',label:'Inflation',range:'1 point higher / lower',control:false,low:s=>{s.spending.generalInflationMean=Math.min(.15,s.spending.generalInflationMean+.01);return true;},high:s=>{s.spending.generalInflationMean=Math.max(-.02,s.spending.generalInflationMean-.01);return true;}},
  {key:'claim',label:'Social Security claim',range:'Age 62 / age 70',control:true,low:s=>{s.socialSecurity.claimAge=62;return true;},high:s=>{s.socialSecurity.claimAge=70;return true;},whatIf:(s,side)=>({claimAge:side==='high'?70:62})},
  {key:'healthcare',label:'Healthcare costs',range:'25% higher / lower',control:false,low:s=>{s.healthcare.preMedicareMonthlyPremium*=1.25;s.longTermCare.annualCost*=1.25;return true;},high:s=>{s.healthcare.preMedicareMonthlyPremium*=.75;s.longTermCare.annualCost*=.75;return true;}},
  {key:'mix',label:'Stock allocation',range:'20 points less / more',control:true,low:s=>shiftMix(s,-.2),high:s=>shiftMix(s,.2),whatIf:(s,side)=>({stockShift:side==='high'?20:-20})}
];

// Plain-text and CSV exports keep every number visible outside the page.
const csvCell=v=>{const t=v==null?'':String(v);return /[",\n]/.test(t)?`"${t.replaceAll('"','""')}"`:t;};
export function labCsv(rows,{basis='today',yearly=[]}={}){
  const head=['Plan','Changes','Paths','Readiness','Lifetimes with a shortfall',`Median left at end (${basis==='today'?'today’s':'future'} dollars)`,'Left in tough markets (10th percentile)','Median age when money runs out','Median lifetime federal income tax (today’s dollars)','Median years with a Medicare surcharge','Median Roth conversions (today’s dollars)'];
  const lines=[head.map(csvCell).join(',')];
  for(const r of rows){if(!r.metrics)continue;const m=r.metrics;lines.push([r.label,r.changesText,m.paths,(m.readiness*100).toFixed(1)+'%',m.shortfalls,Math.round(m.medianLeft),Math.round(m.toughLeft),m.medianFailureAge==null?'':m.medianFailureAge.toFixed(1),m.lifetimeTax==null?'':Math.round(m.lifetimeTax),m.surchargeYears??'',m.conversions==null?'':Math.round(m.conversions)].map(csvCell).join(','));}
  if(yearly.length){lines.push('');lines.push(['Age',...rows.filter(r=>r.metrics).map(r=>`${r.label} median balance`)].map(csvCell).join(','));for(const y of yearly)lines.push([y.age,...y.values.map(v=>v==null?'':Math.round(v))].map(csvCell).join(','));}
  return lines.join('\n')+'\n';
}
