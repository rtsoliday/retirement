const rates=[.10,.12,.22,.24,.32,.35,.37];
const brackets={Single:[0,12400,50400,105700,201775,256225,640600],Married:[0,24800,100800,211400,403550,512450,768700],HeadOfHousehold:[0,17700,67450,105700,201750,256200,640600]};
const deductions={Single:16100,Married:32200,HeadOfHousehold:24150};
const seniorDeduction={Single:2050,Married:1650,HeadOfHousehold:2050};
const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));

export function taxableSocialSecurity(otherIncome, annualSocialSecurity, filingStatus) {
  if(annualSocialSecurity<=0)return 0;
  const married=filingStatus==='Married', base=married?32000:25000, second=married?44000:34000, maximum=married?6000:4500;
  const provisional=otherIncome+annualSocialSecurity*.5;
  if(provisional<=base)return 0;
  if(provisional<=second)return Math.min(.5*(provisional-base),.5*annualSocialSecurity);
  return Math.min(.85*(provisional-second)+Math.min(maximum,.5*annualSocialSecurity),.85*annualSocialSecurity);
}

export function taxableOrdinaryIncome(income,status='Single',inflation=1,seniors=0,taxYear=2026) {
  const multiplier=Number.isFinite(inflation)?Math.max(.0001,inflation):1;
  const seniorCount=clamp(seniors,0,status==='Married'?2:1);
  const ordinary=Math.max(0,income);
  const base=(deductions[status]+seniorDeduction[status]*seniorCount)*multiplier;
  const enhanced=seniorCount>0&&taxYear>=2025&&taxYear<=2028 ? Math.max(0,6000-Math.max(0,ordinary-(status==='Married'?150000:75000))*.06)*seniorCount : 0;
  return Math.max(0,ordinary-base-enhanced);
}

export function taxLiability(taxable,status='Single',inflation=1) {
  if(taxable<=0)return 0;
  const multiplier=Number.isFinite(inflation)?Math.max(.0001,inflation):1;
  const b=brackets[status]; let tax=0;
  for(let i=0;i<rates.length;i++){
    const low=b[i]*multiplier,high=i+1<b.length?b[i+1]*multiplier:Infinity;
    if(taxable<=low)break;
    tax+=(Math.min(taxable,high)-low)*rates[i];
    if(taxable<=high)break;
  }
  return tax;
}
export function ordinaryIncomeTax(income,status='Single',inflation=1,seniors=0,taxYear=2026){return taxLiability(taxableOrdinaryIncome(income,status,inflation,seniors,taxYear),status,inflation);}

// income excludes Social Security; converted dollars can make more of the benefit taxable.
export function rothConversionPlan(pretax,income,rateCap,status,inflation=1,seniors=0,taxYear=2026,annualSocialSecurity=0){
  const none={amount:0,tax:0,taxableSocialSecurityIncrease:0};
  if(pretax<=0)return none;
  const idx=rates.findIndex(x=>Math.abs(x-rateCap)<.0001);
  if(idx<0)return none;
  const limit=(brackets[status][idx+1]??Infinity)*Math.max(.0001,inflation);
  const gross=x=>income+x+taxableSocialSecurity(income+x,annualSocialSecurity,status);
  let low=0,high=pretax;
  if(taxableOrdinaryIncome(gross(high),status,inflation,seniors,taxYear)>limit){for(let i=0;i<40;i++){const mid=(low+high)/2;if(taxableOrdinaryIncome(gross(mid),status,inflation,seniors,taxYear)<=limit)low=mid;else high=mid;}high=low;}
  return {amount:high,tax:Math.max(0,ordinaryIncomeTax(gross(high),status,inflation,seniors,taxYear)-ordinaryIncomeTax(gross(0),status,inflation,seniors,taxYear)),taxableSocialSecurityIncrease:gross(high)-gross(0)-high};
}
