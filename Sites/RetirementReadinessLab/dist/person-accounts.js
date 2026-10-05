import {RothConversionLedger} from './roth-conversions.js';
import {requiredMinimumDistribution} from './distributions.js';
import {ordinaryIncomeTax,taxableSocialSecurity} from './tax.js';
import {calendarMonthsBetween,scenarioTimeline,forecastRetirementDate,localCalendarDate,addCalendarMonths,calendarDate} from './model.js';
import {monthlySavings} from './savings.js';
import {EmployerRothLedger} from './employer-roth.js';

// Shared account-group orders contain no balances or scenario data.
const QUOTE_ORDERS={Standard:[0,1,2,3],TaxableFirst:[3,0,1,2],RothLast:[0,3,1,2]},CONVERSION_ORDER=[4,3,1,2,0];

// Dates depend on the reviewed scenario, not on random draws or account state.
// Keep this memo private to one calculation; never retain it between requests.
export function accountCalendar(s,{reuseDates=true}={}){
  const timeline=scenarioTimeline(s),today=s.household.asOfDate||localCalendarDate(),start=timeline.startDate||s.household.retirementDate;
  const startDates=new Map(),todayDates=new Map(),depositYears=new Map();
  const date=(base,cache,month)=>{if(!reuseDates)return addCalendarMonths(base,month);if(!cache.has(month))cache.set(month,addCalendarMonths(base,month));return cache.get(month);};
  return {timeline,today,start,
    startDate:month=>date(start,startDates,month),
    todayDate:month=>date(today,todayDates,month),
    depositYear:month=>{if(!reuseDates)return calendarDate(date(today,todayDates,month)).getUTCFullYear();if(!depositYears.has(month))depositYears.set(month,calendarDate(date(today,todayDates,month)).getUTCFullYear());return depositYears.get(month);},
  };
}

// Keep separate ownership even though results summarize the household total.
export class PersonAccounts{
  constructor(s,b,calendar=accountCalendar(s)){
    this.s=s;this.b=b;this.calendar=calendar;const {timeline:t,today,start}=calendar;
    const make=(accounts,history,birthday,date,actualDate,savings,withdrawal,age,birth)=>({pretax:accounts.pretax,roth:accounts.roth,ledger:new RothConversionLedger(history.contributionBasis,history.firstContributionYear,history.conversions),retire:calendarMonthsBetween(start,date),retireAge:calendarMonthsBetween(birthday,date)/12,rule55:withdrawal.ruleOf55Eligible&&calendarDate(actualDate)?.getUTCFullYear()>=birth+55,seppSelected:withdrawal.seppEligible,savings,age,birth,rmd:0,paid:0,sepp:0,seppEnd:0});
    this.people=[make(s.accounts,s.rothHistory,s.household.birthday,forecastRetirementDate(s),s.household.retirementDate,s.contributions,s.withdrawalStrategy,t.retirementAge,t.birthYear)];
    if(s.household.filingStatus==='Married')this.people.push(make(s.spouseAccounts,s.spouseRothHistory,s.household.spouseBirthday,forecastRetirementDate(s,true),s.household.spouseRetirementDate,s.spouseContributions,s.spouseWithdrawal,t.spouseAtRet,t.spouseBirthYear));
    this.employer=(s.employerRothAccounts||[]).filter(a=>a.owner==='you'||this.people.length===2).map(a=>({...a,person:this.people[a.owner==='spouse'?1:0],ledger:new EmployerRothLedger(a),rolled:false,converted:false}));
    this.today=today;this.start=start;this.preMonths=t.preMonths;this.refresh();
  }
  refresh(){this.b.pretax=this.people.reduce((n,p)=>n+p.pretax,0);const ira=this.people.reduce((n,p)=>n+p.roth,0),employer=this.employer.reduce((n,a)=>n+a.balance,0);this.b.roth=ira+employer;if(this.employer.length){this.b.rothIRA=ira;this.b.employerRoth=employer;}}
  grow(rate,cashRate){for(const p of this.people){p.pretax*=1+rate;p.roth*=1+rate;}for(const a of this.employer)a.balance*=1+rate;this.b.taxable*=1+rate;this.b.cash*=1+cashRate;this.refresh();}
  deposits(month,pre=false){
    let total=0;const offset=pre?month:month+this.preMonths,year=this.calendar.depositYear(offset+1);
    for(const p of this.people){if(!pre&&(!p.alive||month>=p.retire))continue;const d=monthlySavings(p.savings,offset);p.pretax+=d.pretax;p.roth+=d.roth;this.b.taxable+=d.taxable;this.b.cash+=d.cash;total+=Object.values(d).reduce((a,v)=>a+v,0);if(d.roth){p.ledger.openingBalance+=d.roth;if(!p.ledger.firstContributionYear)p.ledger.firstContributionYear=year;}}
    for(const a of this.employer){const p=a.person;if(a.rolled||!pre&&(!p.alive||month>=p.retire))continue;const d=(a.annualContribution+a.annualEmployerContribution)/12*Math.pow(1+a.annualIncrease,offset/12);a.balance+=d;a.ledger.deposit(d,year);total+=d;}
    this.refresh();return total;
  }
  configure(month,taxYear,deathYears,alive,seppPayment){
    this.month=month;this.taxYear=taxYear;
    for(const [i,p] of this.people.entries()){p.alive=alive[i];p.ownerAge=p.age+month/12;p.afterDeath=!p.alive;}
    if(month%12===0){
      const survivor=this.people.find(p=>p.alive);
      if(survivor)for(const [i,p] of this.people.entries())if(!p.alive&&Math.floor(month/12)>deathYears[i]){
        survivor.pretax+=p.pretax;survivor.roth+=p.roth;survivor.ledger.openingBalance+=p.ledger.openingBalance;
        for(const lot of p.ledger.lots)survivor.ledger.add(lot.amount,lot.taxYear,lot.taxableAmount);
        const first=p.ledger.firstContributionYear;if(first&&(!survivor.ledger.firstContributionYear||first<survivor.ledger.firstContributionYear))survivor.ledger.firstContributionYear=first;
        p.pretax=p.roth=0;p.ledger=new RothConversionLedger();
        // Simplified spouse inheritance: keep each employer plan separate,
        // preserving its participation and conversion history. It becomes the
        // survivor's own plan after the modeled death year.
        for(const a of this.employer)if(a.person===p){a.person=survivor;a.ruleOf55Eligible=false;a.disabled=false;a.accessDate='';a.separationDate='';a.rolloverDate='';a.plannedConversionAmount=0;a.annualContribution=a.annualEmployerContribution=0;}
      }
      for(const p of this.people){p.rmd=requiredMinimumDistribution(p.pretax,Math.floor(p.ownerAge),p.birth);p.paid=0;}
    }
    for(const p of this.people){
      if(p.seppSelected&&p.alive&&month===p.retire&&p.retireAge<59.5){p.sepp=seppPayment(p.pretax,p.retireAge);p.seppEnd=Math.max(p.retire+60,Math.round((59.5-p.age)*12));}
      p.protected=p.alive&&p.sepp>0&&month<p.seppEnd;
      p.penalty=this.s.withdrawalStrategy.applyEarlyWithdrawalPenalty&&p.alive&&p.ownerAge<59.5?.1:0;
      p.pretaxPenalty=p.penalty&&!(p.rule55&&month>=p.retire)?p.penalty:0;
      p.qualified=p.ledger.isQualified(taxYear,p.ownerAge,p.afterDeath);
    }
    if(this.employer.length){
      this.date=this.calendar.startDate(month);this.calendarYear=Number(this.date.slice(0,4));this.taxYear=this.calendarYear;
      for(const p of this.people){
        p.disabled=this.employer.some(a=>a.person===p&&a.disabled);
        if(p.disabled)p.penalty=p.pretaxPenalty=0;
        p.qualified=p.ledger.isQualified(this.calendarYear,p.ownerAge,p.afterDeath||p.disabled);
      }
      for(const a of this.employer){const p=a.person;
        a.available=p.afterDeath||this.date>=(a.accessDate||this.calendar.startDate(p.retire));
        a.qualified=a.ledger.isQualified(this.calendarYear,p.ownerAge,p.afterDeath,p.disabled);
        const rule55=a.ruleOf55Eligible&&Number(a.separationDate.slice(0,4))>=p.birth+55&&this.date>=a.separationDate;
        a.penalty=p.penalty&&!(rule55||p.disabled)?p.penalty:0;
      }
    }
    this.refresh();
  }
  events(month,pre=false){
    if(!this.employer.length)return {conversion:0,rollover:0};
    const date=pre?this.calendar.todayDate(month):this.calendar.startDate(month),year=Number(date.slice(0,4));let conversion=0,rollover=0;
    for(const a of this.employer){const p=a.person,age=p.age+(pre?month-this.preMonths:month)/12;
      if(!pre&&!p.alive)continue;
      if(a.rolloverDate&&!a.rolled&&date>=a.rolloverDate){
        const first=p.ledger.firstContributionYear||year,qualified=a.ledger.isQualified(year,age,p.afterDeath,this.employer.some(other=>other.person===p&&other.disabled));
        a.ledger.rollover(a.balance,year,qualified,p.ledger);p.ledger.firstContributionYear=first;p.roth+=a.balance;rollover+=a.balance;a.balance=0;a.rolled=true;
      }
      if(!pre&&a.plannedConversionAmount>0&&!a.converted&&!a.rolled&&date>=a.plannedConversionDate){
        if(p.protected)throw new Error('An in-plan conversion cannot use an active SEPP-protected pre-tax account.');
        const d=Math.min(Math.max(0,p.pretax-Math.max(0,p.rmd-p.paid)),a.plannedConversionAmount);p.pretax-=d;a.balance+=d;a.ledger.convert(d,year);conversion+=d;a.converted=true;
      }
    }
    this.refresh();return {conversion,rollover};
  }
  scheduled(finalMonth=false){
    let sepp=0,rmd=0;
    for(const p of this.people){const a=p.protected?Math.min(p.pretax,p.sepp/12):0;p.pretax-=a;p.paid+=a;sepp+=a;
      const target=p.rmd*(finalMonth?1:(this.month%12+1)/12),d=Math.min(p.pretax,Math.max(0,target-p.paid));p.pretax-=d;p.paid+=d;rmd+=d;}
    this.refresh();return {sepp,rmd,total:sepp+rmd};
  }
  probe(gross,cashFirst,conversionTax){
    // The search only needs income and penalties. Avoid creating four draw
    // callbacks per probe, and stop visiting pools when the draw is funded.
    let remaining=Math.max(0,gross),taxableDraw=0,rothTaxableEarnings=0,penalties=0;
    if(cashFirst&&!conversionTax)remaining-=Math.min(Math.max(0,this.b.cash),remaining);
    const order=conversionTax?CONVERSION_ORDER:QUOTE_ORDERS[this.s.withdrawalStrategy.withdrawalOrder]??QUOTE_ORDERS.Standard;
    for(let group=0;group<order.length&&remaining>0;group++){
      const kind=order[group];
      if(kind===0){
        for(let i=0;i<this.people.length&&remaining>0;i++){const p=this.people[i];if(p.protected)continue;const d=Math.min(Math.max(0,p.pretax),remaining);remaining-=d;taxableDraw+=d;penalties+=d*p.pretaxPenalty;}
      }else if(kind===1){
        for(let i=0;i<this.people.length&&remaining>0;i++){const p=this.people[i],d=Math.min(Math.max(0,p.roth),remaining);if(d===0)continue;remaining-=d;if(p.qualified)continue;const r=p.ledger.distribution(d,p.roth,this.taxYear);rothTaxableEarnings+=r.taxableEarnings;penalties+=p.penalty*r.penaltyBase;}
      }else if(kind===2){
        for(let i=0;i<this.employer.length&&remaining>0;i++){const a=this.employer[i];if(!a.available)continue;const d=Math.min(Math.max(0,a.balance),remaining);if(d===0)continue;remaining-=d;if(a.qualified)continue;const r=a.ledger.distribution(d,a.balance,this.calendarYear);rothTaxableEarnings+=r.taxableEarnings;penalties+=a.penalty*r.penaltyBase;}
      }else{const key=kind===3?'taxable':'cash';remaining-=Math.min(Math.max(0,this.b[key]),remaining);}
    }
    return {taxableDraw,rothTaxableEarnings,penalties};
  }
  quote(gross,{cashFirst=false,conversionTax=false,capture=true}={}){
    if(!capture)return this.probe(gross,cashFirst,conversionTax);
    // Probes need only income and penalties; account draw records are needed
    // only for the final plan passed to consume(). Neither form mutates pools.
    let remaining=Math.max(0,gross),taxableDraw=0,rothTaxableEarnings=0,penalties=0;const draws=capture?this.people.map(()=>({pretax:0,roth:0})):null,employerDraws=capture?this.employer.map(()=>0):null,shared=capture?{cash:0,taxable:0}:null;
    const takeShared=key=>{const d=Math.min(Math.max(0,this.b[key]),remaining);if(capture)shared[key]+=d;remaining-=d;};
    const takePretax=()=>this.people.forEach((p,i)=>{if(p.protected)return;const d=Math.min(Math.max(0,p.pretax),remaining);if(capture)draws[i].pretax=d;remaining-=d;taxableDraw+=d;penalties+=d*p.pretaxPenalty;});
    const takeRoth=()=>this.people.forEach((p,i)=>{const d=Math.min(Math.max(0,p.roth),remaining);if(d===0)return;const r=p.ledger.distribution(d,p.roth,this.taxYear,{qualified:p.qualified});if(capture)draws[i].roth=d;remaining-=d;rothTaxableEarnings+=r.taxableEarnings;penalties+=p.penalty*r.penaltyBase;});
    const takeEmployer=()=>this.employer.forEach((a,i)=>{if(!a.available)return;const d=Math.min(Math.max(0,a.balance),remaining);if(d===0)return;const r=a.ledger.distribution(d,a.balance,this.calendarYear,{qualified:a.qualified});if(capture)employerDraws[i]=d;remaining-=d;rothTaxableEarnings+=r.taxableEarnings;penalties+=a.penalty*r.penaltyBase;});
    const order=this.s.withdrawalStrategy.withdrawalOrder??'Standard';
    if(conversionTax){takeShared('cash');takeShared('taxable');takeRoth();takeEmployer();takePretax();}
    else{
      if(cashFirst)takeShared('cash');
      if(order==='TaxableFirst'){takeShared('taxable');takePretax();takeRoth();takeEmployer();}
      else if(order==='RothLast'){takePretax();takeShared('taxable');takeRoth();takeEmployer();}
      else{takePretax();takeRoth();takeEmployer();takeShared('taxable');}
      if(!cashFirst)takeShared('cash');
    }
    // An unresolved gap is recorded as negative cash and causes a shortfall.
    if(capture)shared.cash+=remaining;
    return capture?{gross,taxableDraw,rothTaxableEarnings,penalties,draws,employerDraws,shared}:{taxableDraw,rothTaxableEarnings,penalties};
  }
  plan(need,ss,status,other,cashFirst,taxInflation,seniors,taxYear,ytd,incomeTax=ordinaryIncomeTax,conversionTax=false,netSupport=0){
    // Search probes only need net cash. Build the full plan once, retaining
    // the same quote, tax arithmetic and 32 search iterations.
    const estimate=(gross,capture=false)=>{const q=this.quote(gross,{cashFirst,conversionTax,capture});const ordinary=ytd.ordinary+other+q.taxableDraw+q.rothTaxableEarnings,social=ytd.social+ss,totalTax=incomeTax(ordinary+taxableSocialSecurity(ordinary,social,status),status,taxInflation,seniors,taxYear),net=ss+other+netSupport+gross-(totalTax-ytd.tax)-q.penalties;return capture?{...q,totalTax,net}:net;};
    if(estimate(0)>=need)return estimate(0,true);
    let low=0,high=Math.max(0,need-ss-other-netSupport)*1.8+10000;
    while(estimate(high)<need){high*=2;if(!Number.isFinite(high))throw Error('Simulation exceeded the finite numeric range.');}
    for(let i=0;i<32;i++){const mid=(low+high)/2;if(estimate(mid)>=need)high=mid;else low=mid;}return estimate(high,true);
  }
  consume(q){for(const [i,p] of this.people.entries()){const d=q.draws[i];p.pretax-=d.pretax;p.paid+=d.pretax;p.ledger.distribution(d.roth,p.roth,this.taxYear,{qualified:p.qualified,consume:true});p.roth-=d.roth;}for(const [i,a] of this.employer.entries()){const d=q.employerDraws[i];a.ledger.distribution(d,a.balance,this.calendarYear,{qualified:a.qualified,consume:true});a.balance-=d;}this.b.cash-=q.shared.cash;this.b.taxable-=q.shared.taxable;this.refresh();}
  convert(amount){let remaining=amount;for(const p of this.people){if(p.protected)continue;const d=Math.min(Math.max(0,p.pretax),remaining);p.pretax-=d;p.roth+=d;p.ledger.add(d,this.taxYear);remaining-=d;}this.refresh();return amount-remaining;}
  snapshot(){return this.people.map(p=>({pretax:p.pretax,roth:p.roth,contributionBasis:p.ledger.openingBalance,firstContributionYear:p.ledger.firstContributionYear,...(this.employer.length?{employerRoth:this.employer.filter(a=>a.person===p).map(a=>({name:a.name,type:a.type,balance:a.balance,contributionBasis:a.ledger.basis,firstContributionYear:a.ledger.firstContributionYear,rolled:a.rolled}))}:{})}));}
}
