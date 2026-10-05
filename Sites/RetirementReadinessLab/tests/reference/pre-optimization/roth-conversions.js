// Roth IRA ordering: remaining regular contributions, conversion principal
// (oldest tax year first, taxable principal first within that year), earnings.
// Dollar basis survives market losses and is never increased by returns.
export class RothConversionLedger {
  constructor(contributionBasis=0,firstContributionYear=0,conversions=[]) {
    this.openingBalance=contributionBasis;
    this.firstContributionYear=firstContributionYear;
    this.lots=[];
    for(const lot of conversions)this.add(lot.amount,lot.taxYear,lot.taxableAmount);
  }
  add(amount,taxYear,taxableAmount=amount) {
    if(amount<=0)return;
    const existing=this.lots.find(lot=>lot.taxYear===taxYear);
    if(existing){existing.amount+=amount;existing.taxableAmount+=taxableAmount;}
    else this.lots.push({amount,taxYear,taxableAmount});
    this.lots.sort((a,b)=>a.taxYear-b.taxYear);
    if(!this.firstContributionYear||taxYear<this.firstContributionYear)this.firstContributionYear=taxYear;
  }
  isQualified(taxYear,age,afterDeath=false) {
    return this.firstContributionYear>0&&taxYear-this.firstContributionYear>=5&&(age>=59.5||afterDeath);
  }
  distribution(amount,balance,taxYear,{qualified=false,consume=false}={}) {
    let remaining=Math.min(Math.max(0,amount),Math.max(0,balance)),recaptureBase=0;
    const contributions=Math.min(remaining,Math.max(0,this.openingBalance));
    remaining-=contributions;
    if(consume)this.openingBalance-=contributions;
    let conversions=0;
    for(const lot of this.lots){
      const take=Math.min(remaining,lot.amount),taxableTake=Math.min(take,lot.taxableAmount);
      if(taxYear-lot.taxYear<5)recaptureBase+=taxableTake;
      conversions+=take;remaining-=take;
      if(consume){lot.amount-=take;lot.taxableAmount-=taxableTake;}
      if(remaining<=0)break;
    }
    if(consume)this.lots=this.lots.filter(lot=>lot.amount>0);
    const earnings=remaining;
    return {contributions,conversions,earnings,taxableEarnings:qualified?0:earnings,penaltyBase:qualified?0:recaptureBase+earnings};
  }
  // Retain the conversion-recapture query used when funding conversion taxes.
  withdrawal(amount,balance,taxYear,consume=false) {
    const result=this.distribution(amount,balance,taxYear,{consume});
    return result.penaltyBase-result.earnings;
  }
}
