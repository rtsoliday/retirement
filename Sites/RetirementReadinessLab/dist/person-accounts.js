import {RothConversionLedger} from './roth-conversions.js';
import {requiredMinimumDistribution} from './distributions.js';
import {ordinaryIncomeTax,taxableSocialSecurity} from './tax.js';
import {calendarMonthsBetween,scenarioTimeline,localCalendarDate,addCalendarMonths,calendarDate} from './model.js';
import {monthlySavings} from './savings.js';

// Keep separate ownership even though results summarize the household total.
export class PersonAccounts{
  constructor(s,b){
    this.s=s;this.b=b;const t=scenarioTimeline(s),today=s.household.asOfDate||localCalendarDate(),start=t.startDate||s.household.retirementDate;
    const make=(accounts,history,birthday,date,savings,withdrawal,age,birth)=>({pretax:accounts.pretax,roth:accounts.roth,ledger:new RothConversionLedger(history.contributionBasis,history.firstContributionYear,history.conversions),retire:calendarMonthsBetween(start,date),retireAge:calendarMonthsBetween(birthday,date)/12,rule55:withdrawal.ruleOf55Eligible&&calendarDate(date).getUTCFullYear()>=birth+55,seppSelected:withdrawal.seppEligible,savings,age,birth,rmd:0,paid:0,sepp:0,seppEnd:0});
    this.people=[make(s.accounts,s.rothHistory,s.household.birthday,s.household.retirementDate,s.contributions,s.withdrawalStrategy,t.retirementAge,t.birthYear)];
    if(s.household.filingStatus==='Married')this.people.push(make(s.spouseAccounts,s.spouseRothHistory,s.household.spouseBirthday,s.household.spouseRetirementDate,s.spouseContributions,s.spouseWithdrawal,t.spouseAtRet,t.spouseBirthYear));
    this.today=today;this.start=start;this.preMonths=t.preMonths;this.refresh();
  }
  refresh(){this.b.pretax=this.people.reduce((n,p)=>n+p.pretax,0);this.b.roth=this.people.reduce((n,p)=>n+p.roth,0);}
  grow(rate,cashRate){for(const p of this.people){p.pretax*=1+rate;p.roth*=1+rate;}this.b.taxable*=1+rate;this.b.cash*=1+cashRate;this.refresh();}
  deposits(month,pre=false){
    let total=0;const offset=pre?month:month+this.preMonths,date=addCalendarMonths(this.today,offset+1),year=calendarDate(date).getUTCFullYear();
    for(const p of this.people){if(!pre&&(!p.alive||month>=p.retire))continue;const d=monthlySavings(p.savings,offset);p.pretax+=d.pretax;p.roth+=d.roth;this.b.taxable+=d.taxable;this.b.cash+=d.cash;total+=Object.values(d).reduce((a,v)=>a+v,0);if(d.roth){p.ledger.openingBalance+=d.roth;if(!p.ledger.firstContributionYear)p.ledger.firstContributionYear=year;}}
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
    this.refresh();
  }
  scheduled(finalMonth=false){
    let sepp=0,rmd=0;
    for(const p of this.people){const a=p.protected?Math.min(p.pretax,p.sepp/12):0;p.pretax-=a;p.paid+=a;sepp+=a;
      const target=p.rmd*(finalMonth?1:(this.month%12+1)/12),d=Math.min(p.pretax,Math.max(0,target-p.paid));p.pretax-=d;p.paid+=d;rmd+=d;}
    this.refresh();return {sepp,rmd,total:sepp+rmd};
  }
  quote(gross,{cashFirst=false,conversionTax=false}={}){
    let remaining=Math.max(0,gross),taxableDraw=0,rothTaxableEarnings=0,penalties=0;const draws=this.people.map(()=>({pretax:0,roth:0})),shared={cash:0,taxable:0};
    const takeShared=key=>{const d=Math.min(Math.max(0,this.b[key]),remaining);shared[key]+=d;remaining-=d;};
    const takePretax=()=>this.people.forEach((p,i)=>{if(p.protected)return;const d=Math.min(Math.max(0,p.pretax),remaining);draws[i].pretax=d;remaining-=d;taxableDraw+=d;penalties+=d*p.pretaxPenalty;});
    const takeRoth=()=>this.people.forEach((p,i)=>{const d=Math.min(Math.max(0,p.roth),remaining),r=p.ledger.distribution(d,p.roth,this.taxYear,{qualified:p.qualified});draws[i].roth=d;remaining-=d;rothTaxableEarnings+=r.taxableEarnings;penalties+=p.penalty*r.penaltyBase;});
    if(conversionTax){takeShared('cash');takeShared('taxable');takeRoth();takePretax();}else{if(cashFirst)takeShared('cash');takePretax();takeRoth();takeShared('taxable');if(!cashFirst)takeShared('cash');}
    // An unresolved gap is recorded as negative cash and causes a shortfall.
    shared.cash+=remaining;
    return {gross,taxableDraw,rothTaxableEarnings,penalties,draws,shared};
  }
  plan(need,ss,status,other,cashFirst,taxInflation,seniors,taxYear,ytd,incomeTax=ordinaryIncomeTax,conversionTax=false,netSupport=0){
    const estimate=gross=>{const q=this.quote(gross,{cashFirst,conversionTax});const ordinary=ytd.ordinary+other+q.taxableDraw+q.rothTaxableEarnings,social=ytd.social+ss,totalTax=incomeTax(ordinary+taxableSocialSecurity(ordinary,social,status),status,taxInflation,seniors,taxYear),net=ss+other+netSupport+gross-(totalTax-ytd.tax)-q.penalties;return {...q,totalTax,net};};
    if(estimate(0).net>=need)return estimate(0);
    let low=0,high=Math.max(0,need-ss-other-netSupport)*1.8+10000;
    while(estimate(high).net<need){high*=2;if(!Number.isFinite(high))throw Error('Simulation exceeded the finite numeric range.');}
    for(let i=0;i<32;i++){const mid=(low+high)/2;if(estimate(mid).net>=need)high=mid;else low=mid;}return estimate(high);
  }
  consume(q){for(const [i,p] of this.people.entries()){const d=q.draws[i];p.pretax-=d.pretax;p.paid+=d.pretax;p.ledger.distribution(d.roth,p.roth,this.taxYear,{qualified:p.qualified,consume:true});p.roth-=d.roth;}this.b.cash-=q.shared.cash;this.b.taxable-=q.shared.taxable;this.refresh();}
  convert(amount){let remaining=amount;for(const p of this.people){if(p.protected)continue;const d=Math.min(Math.max(0,p.pretax),remaining);p.pretax-=d;p.roth+=d;p.ledger.add(d,this.taxYear);remaining-=d;}this.refresh();return amount-remaining;}
  snapshot(){return this.people.map(p=>({pretax:p.pretax,roth:p.roth,contributionBasis:p.ledger.openingBalance,firstContributionYear:p.ledger.firstContributionYear}));}
}
