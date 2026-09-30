// Opening Roth savings retain the model's assumption of penalty-free access.
// Conversions made during a path have known dates and taxable principal. Draw
// that principal oldest first, before conversion earnings; market gains do not
// increase the principal subject to the conversion recapture penalty.
export class RothConversionLedger {
  constructor(openingBalance=0) { this.openingBalance=openingBalance; this.lots=[]; }
  grow(factor) { this.openingBalance*=Math.max(0,factor); }
  add(amount,taxYear) { this.lots.push({amount,taxYear}); }
  withdrawal(amount,balance,taxYear,consume=false) {
    let remaining=Math.min(Math.max(0,amount),Math.max(0,balance)),penaltyBase=0;
    const opening=Math.min(remaining,Math.max(0,this.openingBalance));
    remaining-=opening;
    if(consume)this.openingBalance-=opening;
    for(const lot of this.lots){
      const take=Math.min(remaining,lot.amount);
      if(taxYear-lot.taxYear<5)penaltyBase+=take;
      remaining-=take;
      if(consume)lot.amount-=take;
      if(remaining<=0)break;
    }
    if(consume)this.lots=this.lots.filter(lot=>lot.amount>0);
    return penaltyBase;
  }
}
