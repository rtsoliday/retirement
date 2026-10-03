import {retirementBenefitFactor,spousalBenefitFactor,combinedSurvivorBenefitFactor} from './social-security.js';
export function personSocialSecurity(s,t,ages,alive,deaths,index){
  const married=s.household.filingStatus==='Married';
  const birth=[t.birthYear,t.spouseBirthYear],claims=[s.socialSecurity.claimAge,s.socialSecurity.spouseClaimAge];
  const benefits=[s.socialSecurity.annualBenefitAt67,s.spouseIncome.annualBenefitAt67];
  const pia=benefits.map((v,i)=>v/retirementBenefitFactor(birth[i],67*12)/12*index);
  const own=benefits.map((_,i)=>alive[i]&&ages[i]>=claims[i]?pia[i]*retirementBenefitFactor(birth[i],claims[i]*12):0);
  if(!married)return own[0];
  return own.reduce((total,value,i)=>{
    if(!alive[i])return total;
    const j=1-i;
    if(alive[j]){
      // Own retirement benefit plus reduced excess spousal benefit, rather than
      // adding two full benefits or reducing the spouse's own benefit twice.
      const excessClaim=Math.max(claims[i],(i===0?t.retirementAge:t.spouseAtRet)+claims[j]-(j===0?t.retirementAge:t.spouseAtRet));
      const excess=ages[i]>=excessClaim&&ages[j]>=claims[j]?Math.max(0,.5*pia[j]-pia[i])*spousalBenefitFactor(birth[i],excessClaim*12)/.5:0;
      return total+value+excess;
    }
    const survivorClaim=Math.max(claims[i],60,(i===0?t.retirementAge:t.spouseAtRet)+deaths[j]-(j===0?t.retirementAge:t.spouseAtRet));
    const survivor=ages[i]>=survivorClaim?pia[j]*combinedSurvivorBenefitFactor(birth[j],claims[j]*12,deaths[j]*12,birth[i],survivorClaim*12):0;
    return total+Math.max(value,survivor);
  },0);
}
export function personPensions(s,ages,alive,deaths,months){
  const streams=[s.guaranteedIncome,{annualIncome:s.spouseIncome.annualPension,startAge:s.spouseIncome.pensionStartAge,startAgeMonths:s.spouseIncome.pensionStartAgeMonths,annualIncrease:s.spouseIncome.annualIncrease,survivorPercent:s.spouseIncome.survivorPercent}];
  let total=0;
  for(const [i,p] of streams.entries()){
    if(i===1&&s.household.filingStatus!=='Married')continue;
    if(p.annualIncome===0)continue;
    const start=p.startAge+p.startAgeMonths/12;if(ages[i]<start)continue;
    const factor=alive[i]?1:alive[1-i]&&deaths[i]>=start?p.survivorPercent:0;
    total+=p.annualIncome/12*Math.pow(1+p.annualIncrease,months/12)*factor;
  }
  return total;
}
