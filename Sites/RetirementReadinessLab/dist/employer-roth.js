import {RothConversionLedger} from './roth-conversions.js';

export const employerRothDefaults=()=>({name:'Employer Roth',type:'401k',owner:'you',balance:0,contributionBasis:0,firstContributionYear:0,annualContribution:0,annualEmployerContribution:0,annualIncrease:0,accessDate:'',separationDate:'',ruleOf55Eligible:false,disabled:false,conversionOnly:false,conversions:[],rolloverDate:'',plannedConversionDate:'',plannedConversionAmount:0});
const validDate=value=>typeof value==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(value)&&Number.isFinite(Date.parse(value+'T00:00:00Z'))&&new Date(value+'T00:00:00Z').toISOString().slice(0,10)===value;
export const employerRothTotal=s=>(s.employerRothAccounts||[]).filter(a=>a.owner==='you'||s.household.filingStatus==='Married').reduce((n,a)=>n+a.balance,0);
export const hasEmployerRoth=s=>(s.employerRothAccounts||[]).some(a=>a.owner==='you'||s.household.filingStatus==='Married');
export function validateEmployerRoth(accounts,{complete=false,today='2026-12-31',married=true}={}){
  if(!Array.isArray(accounts))return ['Employer Roth accounts must be an array.'];
  const errors=[],defaults=employerRothDefaults(),year=Number(today.slice(0,4)),max=Number.MAX_SAFE_INTEGER;
  if(accounts.length>50)errors.push('Use at most 50 employer Roth accounts.');
  accounts.forEach((a,i)=>{
    const label=`Employer Roth account ${i+1}`;
    if(!a||typeof a!=='object'||Array.isArray(a)||Object.entries(defaults).some(([k,v])=>Array.isArray(v)?!Array.isArray(a[k]):typeof a[k]!==typeof v)){errors.push(label+': invalid account fields.');return;}
    if(!['401k','403b'].includes(a.type)||!['you','spouse'].includes(a.owner)||!a.name.trim()||a.name.length>120)errors.push(label+': choose a name, owner, and Roth 401(k) or 403(b).');
    for(const [k,v] of Object.entries(a))if(typeof v==='number'&&(!Number.isFinite(v)||Math.abs(v)>max))errors.push(label+': amounts must be finite and within the supported range.');
    for(const k of ['accessDate','separationDate','rolloverDate','plannedConversionDate'])if(a[k]&&!validDate(a[k]))errors.push(label+': '+k+' must be a valid date or blank.');
    if(a.conversions.some(l=>!l||typeof l!=='object'||Array.isArray(l)||!['taxYear','amount','taxableAmount'].every(k=>typeof l[k]==='number'&&Number.isFinite(l[k])&&Math.abs(l[k])<=max))){errors.push(label+': conversion history needs finite tax years and remaining principal.');return;}
    if(!complete||a.owner==='spouse'&&!married)return;
    if(['balance','contributionBasis','annualContribution','annualEmployerContribution','plannedConversionAmount'].some(k=>a[k]<0)||a.annualIncrease<-.02||a.annualIncrease>.15)errors.push(label+': balances and deposits must be nonnegative; deposit growth must be between -2% and 15%.');
    if(!Number.isInteger(a.firstContributionYear)||a.firstContributionYear!==0&&(a.firstContributionYear<2006||a.firstContributionYear>year)||!a.firstContributionYear&&(a.balance>0||a.contributionBasis>0))errors.push(label+': enter its first funding tax year from 2006 through '+year+', or 0 if never funded.');
    for(const l of a.conversions)if(!Number.isInteger(l.taxYear)||l.taxYear<2010||l.taxYear>year||l.taxYear<a.firstContributionYear||l.amount<0||l.taxableAmount<0||l.taxableAmount>l.amount)errors.push(label+': review past in-plan conversion years and remaining principal.');
    if(a.conversions.reduce((n,l)=>n+l.amount,0)>a.contributionBasis)errors.push(label+': remaining in-plan conversion principal cannot exceed total after-tax basis.');
    if(a.ruleOf55Eligible&&!a.separationDate)errors.push(label+': record this employer’s actual or planned separation date for Rule of 55.');
    if(a.rolloverDate&&a.rolloverDate<today)errors.push(label+': a planned rollover cannot precede today; record money already rolled over in the Roth IRA instead.');
    if(a.rolloverDate&&a.accessDate&&a.rolloverDate<a.accessDate)errors.push(label+': rollover must be on or after the permitted access date.');
    if(a.plannedConversionAmount>0&&(!a.plannedConversionDate||a.plannedConversionDate<today))errors.push(label+': choose a future or current date for the planned in-plan conversion.');
    if(a.rolloverDate&&a.plannedConversionAmount>0&&a.plannedConversionDate>=a.rolloverDate)errors.push(label+': planned conversion must precede the full rollover.');
  });
  return errors;
}

// Designated Roth distributions recover basis proportionally (IRC 72), while
// in-plan recapture allocates recovered basis after non-conversion basis, then
// FIFO/taxable-first (Notice 2010-84 Q13). This ledger does not tax principal twice.
export class EmployerRothLedger{
  constructor(account){
    this.basis=account.contributionBasis;this.firstContributionYear=account.firstContributionYear;
    this.principal=new RothConversionLedger(Math.max(0,this.basis-account.conversions.reduce((n,l)=>n+l.amount,0)),this.firstContributionYear,account.conversions);
    this.conversionOnly=account.conversionOnly;
  }
  deposit(amount,year){this.basis+=amount;this.principal.openingBalance+=amount;if(amount&&!this.firstContributionYear)this.firstContributionYear=year;}
  convert(amount,year){this.basis+=amount;this.principal.add(amount,year);if(amount&&!this.firstContributionYear)this.firstContributionYear=year;}
  isQualified(year,age,afterDeath=false,disabled=false){return this.firstContributionYear>0&&year-this.firstContributionYear>=5&&(age>=59.5||afterDeath||disabled);}
  distribution(amount,balance,year,{qualified=false,consume=false}={}){
    const gross=Math.min(Math.max(0,amount),Math.max(0,balance)),basis=gross*Math.min(1,Math.max(0,this.basis)/Math.max(balance,Number.MIN_VALUE)),earnings=gross-basis;
    const regular=this.conversionOnly?0:Math.min(basis,this.principal.openingBalance);
    let remaining=basis-regular,recapture=0;
    for(const l of this.principal.lots){const d=Math.min(remaining,l.amount),taxable=Math.min(d,l.taxableAmount);if(year-l.taxYear<5)recapture+=taxable;remaining-=d;if(consume){l.amount-=d;l.taxableAmount-=taxable;}if(remaining<=0)break;}
    if(consume){this.basis=Math.max(0,this.basis-basis);this.principal.openingBalance=Math.max(0,this.principal.openingBalance-regular-remaining);this.principal.lots=this.principal.lots.filter(l=>l.amount>0);}
    return {gross,basis,earnings,taxableEarnings:qualified?0:earnings,penaltyBase:qualified?0:earnings+recapture};
  }
  rollover(balance,year,qualified,ira){
    // A full direct rollover preserves unrecovered basis even after a loss
    // (§1.408A-10 Q3, §1.402A-1 Q6). Qualified earnings also become IRA basis.
    const basis=qualified?Math.max(balance,this.basis):this.basis,converted=Math.min(basis,this.principal.lots.reduce((n,l)=>n+l.amount,0));
    ira.openingBalance+=Math.max(0,basis-converted);
    let remaining=converted;
    for(const l of this.principal.lots){const amount=Math.min(l.amount,remaining);ira.add(amount,l.taxYear,Math.min(amount,l.taxableAmount));remaining-=amount;}
    // IRA qualification is independent of employer participation and recapture
    // years. Restore its actual first funding year after add() merges the lots.
    this.basis=0;this.principal=new RothConversionLedger();
    return basis;
  }
}
