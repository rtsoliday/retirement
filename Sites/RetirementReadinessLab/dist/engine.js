import {runSeparatePeople} from './person-engine.js';
import {depositSavings,hasFutureSavings} from './savings.js';
import {buildPathPoints,buildBalanceBands,buildFundingSurvival,medianOfSorted} from './chart-data.js';
import {ENGINE_VERSION, scenarioEngineVersion, validateScenario, retirementAge, ruleOf55Applies, setAnnualBaseSpending, scenarioTimeline, forecastRetirementDate, usesCalendarDates, setRetirementAge, localCalendarDate, addCalendarMonths, calendarMonthsBetween, FREE_SIMULATION_PATHS} from './model.js';
import {maleMortality,femaleMortality} from './mortality.js';
import {taxableSocialSecurity,ordinaryIncomeTax,rothConversionPlan} from './tax.js';
import {RothConversionLedger} from './roth-conversions.js';
import {mortgageAtRetirement,payMortgage} from './mortgage.js';
import {retirementBenefitFactor,spousalBenefitFactor,combinedSurvivorBenefitFactor} from './social-security.js';
import {monthlyRateDistribution,sampleMonthlyRate} from './annual-rates.js';
import {requiredMinimumDistribution} from './distributions.js';

const STRIDE=-7046029254386353131n, MASK=(1n<<48n)-1n;
export const monthly=r=>Math.pow(Math.max(.0001,1+r),1/12)-1;
export function requireFinite(...values){if(values.some(v=>!Number.isFinite(v)))throw new Error('Simulation exceeded the finite numeric range. Reduce the financial amounts or rates and try again.');}
export function sum(b){const total=b.pretax+b.roth+b.taxable+b.cash;requireFinite(total);return total;}
const invested=b=>b.pretax+b.roth+b.taxable;
const clamp=(n,lo,hi)=>Math.min(hi,Math.max(lo,n));
function percentile(sorted,p){return sorted.length?sorted[Math.min(sorted.length-1,Math.round((sorted.length-1)*p))]:0;}

export class JavaRandom {
  constructor(seed){this.seed=(BigInt(seed)^0x5deece66dn)&MASK;this.gaussian=null;}
  next(bits){this.seed=(this.seed*0x5deece66dn+0xbn)&MASK;return Number(this.seed>>BigInt(48-bits));}
  nextDouble(){return (this.next(26)*134217728+this.next(27))/9007199254740992;}
  nextGaussian(){if(this.gaussian!==null){const x=this.gaussian;this.gaussian=null;return x;}let v1,v2,s;do{v1=2*this.nextDouble()-1;v2=2*this.nextDouble()-1;s=v1*v1+v2*v2;}while(s>=1||s===0);const m=Math.sqrt(-2*Math.log(s)/s);this.gaussian=v2*m;return v1*m;}
  normal(mean,std){return mean+this.nextGaussian()*std;}
}

export function sampleDeathAge(gender,start,end,rng){
  const table=gender==='Female'?femaleMortality:maleMortality,limit=Math.min(Math.round(end*12),120*12);let month=Math.round(start*12);
  while(month<limit){
    const age=Math.floor(month/12),next=Math.min((age+1)*12,limit),fraction=(next-month)/12,q=table[age]??1;
    const probability=fraction===1?q:1-Math.pow(1-q,fraction),draw=rng.nextDouble();
    if(draw<probability){
      // Constant hazard within the age year preserves its annual probability.
      // Reuse the annual draw for timing: lower draws select later months,
      // leaving the subsequent spouse, care and market draws unchanged.
      for(let deathMonth=month+1;deathMonth<next;deathMonth++){
        const laterDeathProbability=Math.pow(1-q,(deathMonth-month)/12)-(1-probability);
        if(draw>=laterDeathProbability)return deathMonth/12;
      }
      return next/12;
    }
    month=next;
  }
  return limit/12;
}
// The inverse of sampleDeathAge's distribution: the first month end by which the
// cumulative death probability reaches u, using the same constant monthly hazard.
export function deathAgeAtQuantile(gender,start,end,u){
  const table=gender==='Female'?femaleMortality:maleMortality,limit=Math.min(Math.round(end*12),120*12);let month=Math.round(start*12),survival=1;
  while(month<limit){
    const q=table[Math.floor(month/12)]??1;
    survival*=Math.pow(1-q,1/12);month++;
    if(survival<=1-u)return month/12;
  }
  return limit/12;
}
// A preview has too few paths to leave lifespans to chance. Give each path one
// equal-probability band of the mortality table instead. Spouse bands use a
// seeded shuffle so the two lifespans in a path are not paired by rank.
export function previewLifespanQuantiles(n,seed){
  if(n>FREE_SIMULATION_PATHS)return null;
  const rng=new JavaRandom(BigInt(seed)),spouse=Array.from({length:n},(_,i)=>i);
  for(let i=n-1;i>0;i--){const j=Math.floor(rng.nextDouble()*(i+1));[spouse[i],spouse[j]]=[spouse[j],spouse[i]];}
  return Array.from({length:n},(_,i)=>({primary:(i+.5)/n,spouse:(spouse[i]+.5)/n}));
}
// Draw the full lifetimes from the table. Preview bands replace the drawn ages,
// but the draws still run so care and market draws keep their order.
export function drawLifetimes(h,spouseAtRet,married,rng,fixedDeathAges,lifespanQuantiles){
  const death=fixedDeathAges?.primary??sampleDeathAge(h.gender,h.retirementAge,120,rng),spouseDeath=married?(fixedDeathAges?.spouse??sampleDeathAge(h.spouseGender,spouseAtRet,120,rng)):death;
  if(!lifespanQuantiles||fixedDeathAges)return [death,spouseDeath];
  const banded=deathAgeAtQuantile(h.gender,h.retirementAge,120,lifespanQuantiles.primary);
  return [banded,married?deathAgeAtQuantile(h.spouseGender,spouseAtRet,120,lifespanQuantiles.spouse):banded];
}
export function ltcStart(s,start,death,rng){const draw=rng.nextDouble();if(!s.longTermCare.enabled||death<65)return null;const p=death<75?.25:death<85?.45:death<95?.60:.70;return draw>p?null:Math.max(start,death-s.longTermCare.averageDurationYears-(s.longTermCare.averageDurationMonths??0)/12);}
export function allocation(s,b,annualSpending){if(annualSpending<=0)return s.postRetirementAllocation.stock50xOrMore;const ratio=invested(b)/annualSpending,a=s.postRetirementAllocation;return ratio<30?a.stockUnder30x:ratio<35?a.stock30xTo35x:ratio<40?a.stock35xTo40x:ratio<45?a.stock40xTo45x:ratio<50?a.stock45xTo50x:a.stock50xOrMore;}
export function spendingPath(s,offset,timeline=scenarioTimeline(s)){if(s.spending.spendingPathModel==='Flat')return 1;const h=s.household,months=Math.max(0,Math.min(Math.round(timeline.retirementAge*12)+offset,85*12)-Math.max(65,timeline.currentAge)*12),married=h.filingStatus==='Married';return Math.max(married?.60:.65,Math.pow(1-(married?.024:.017),months/12));}
export function medicarePremium(income,status,people,healthInflation,taxInflation,taxYear=2026,priorYearTaxInflation=1){
  const joint=status==='Married',limits=joint?[218000,274000,342000,410000]:[109000,137000,171000,205000];
  const b=[0,81.20,202.90,324.60,446.30,487],d=[0,14.50,37.50,60.40,83.30,91];
  // The top threshold is frozen through 2027. Starting in 2028 its separate
  // August 2026 base uses the preceding year's price level, rounded to $1,000;
  // joint filers use 150% of the indexed individual threshold.
  const indexedTop=500000*Math.max(1,priorYearTaxInflation);
  const top=(taxYear<2028?500000:Math.round((indexedTop+indexedTop*Number.EPSILON)/1000)*1000)*(joint?1.5:1);
  const scaledLimits=limits.map(cap=>cap*Math.max(.0001,taxInflation)),amount=Math.max(0,income);
  // Summing twelve monthly payments can land a few floating-point units below
  // an exact annual boundary. Preserve equality without moving a cent below it.
  const lowerTier=scaledLimits.findIndex(cap=>amount<=cap),idx=amount>=top-8*Number.EPSILON*top?5:lowerTier<0?4:lowerTier;
  return (202.90+38.99+b[idx]+d[idx])*Math.max(0,healthInflation)*people;
}
function draw(b,key,amount){const x=Math.min(b[key],amount);b[key]-=x;return amount-x;}
function withdrawAccounts(b,amount,order,ledger,taxYear){let n=Math.max(0,amount);for(const key of order){if(key==='roth')ledger.withdrawal(n,b.roth,taxYear,true);n=draw(b,key,n);}b.cash-=n;}
function withdrawConversionTax(b,amount,ledger,taxYear,rothPenalty){
  if(!rothPenalty){withdrawAccounts(b,amount,['cash','taxable','roth','pretax'],ledger,taxYear);return;}
  // Newly converted Roth funds can pay the tax, but the payment itself may
  // trigger recapture. Gross up once, including the funds used for that penalty.
  const net=gross=>gross-rothPenalty*ledger.withdrawal(Math.max(0,gross-b.cash-b.taxable),b.roth,taxYear);
  let low=amount,high=amount/(1-rothPenalty);
  for(let i=0;i<32;i++){const mid=(low+high)/2;if(net(mid)>=amount)high=mid;else low=mid;}
  withdrawAccounts(b,high,['cash','taxable','roth','pretax'],ledger,taxYear);
}
// Charge the increase in the actual year-to-date liability, using one set of
// brackets for the entire year. Never annualize a draw that can exhaust pretax.
function withdrawalPlan(need,ss,status,other,b,cashFirst,taxInflation,seniors,taxYear,penalty,ytd={ordinary:0,social:0,tax:0},ledger=null,rothPenalty=0,{pretaxAvailable=b.pretax,incomeTax=ordinaryIncomeTax,rothQualified=false}={}){
  requireFinite(need,ss,other,sum(b),pretaxAvailable,taxInflation,ytd.ordinary,ytd.social,ytd.tax);
  function estimate(gross){
    const taxableDraw=Math.min(Math.max(0,pretaxAvailable),Math.max(0,gross-(cashFirst?Math.max(0,b.cash):0)));
    const rothDraw=Math.max(0,gross-(cashFirst?Math.max(0,b.cash):0)-taxableDraw);
    const roth=ledger?.distribution(rothDraw,b.roth,taxYear,{qualified:rothQualified})??{taxableEarnings:0,penaltyBase:0};
    const ordinary=ytd.ordinary+other+taxableDraw+roth.taxableEarnings,social=ytd.social+ss;
    const taxableSS=taxableSocialSecurity(ordinary,social,status);
    const totalTax=incomeTax(ordinary+taxableSS,status,taxInflation,seniors,taxYear),tax=totalTax-ytd.tax;
    const recapture=rothPenalty*roth.penaltyBase;
    const net=ss+other+gross-tax-taxableDraw*Math.max(0,penalty)-recapture;
    requireFinite(gross,taxableDraw,totalTax,net);
    return {gross,taxableDraw,rothTaxableEarnings:roth.taxableEarnings,totalTax,net};
  }
  if(estimate(0).net>=need)return estimate(0);
  let low=0,high=Math.max(0,need-ss-other)*1.8+10000;
  while(estimate(high).net<need){high*=2;requireFinite(high);}
  for(let i=0;i<32;i++){const mid=(low+high)/2;if(estimate(mid).net>=need)high=mid;else low=mid;}
  return estimate(high);
}
const life=[84.6,83.7,82.8,81.8,80.8,79.8,78.8,77.9,76.9,75.9,74.9,73.9,72.9,71.9,70.9,69.9,69,68,67,66,65,64.1,63.1,62.1,61.1,60.2,59.2,58.2,57.3,56.3,55.3,54.4,53.4,52.5,51.5,50.5,49.6,48.6,47.7,46.7,45.7,44.8,43.8,42.9,41.9,41,40,39,38.1,37.1,36.2,35.3,34.3,33.4,32.5,31.6,30.6,29.8,28.9,28,27.1,26.2,25.4,24.5,23.7,22.9,22,21.2,20.4,19.6,18.8,18,17.2,16.4,15.6,14.8,14.1,13.3,12.6,11.9,11.2,10.5,9.9,9.3,8.7,8.1,7.6,7.1,6.6,6.1,5.7,5.3,4.9,4.6,4.3,4,3.7,3.4,3.2,3,2.8,2.6,2.5,2.3,2.2,2.1,2.1,2.1,2,2,2,2,2,1.9,1.9,1.8,1.8,1.6,1.4,1.1,1];
export function seppPayment(balance,age){const years=life[Math.max(0,Math.floor(age))];if(!years||balance<=0)return 0;return balance/((1-Math.pow(1.05,-years))/.05);}

export function runOne(s,rng,{captureMonthlyBalances=false,captureMonthlyDetails=false,captureTaxDetails=false,captureTodayDollars=false,taxesEnabled=true,horizonReductionYears=0,fixedDeathAges=null,lifespanQuantiles=null}={}){
  if(s.household.separatePeople)return runSeparatePeople(s,rng,{captureMonthlyBalances,captureMonthlyDetails,captureTaxDetails,captureTodayDollars,taxesEnabled,horizonReductionYears,fixedDeathAges,lifespanQuantiles});
  const timeline=scenarioTimeline(s),h={...s.household,currentAge:timeline.currentAge,retirementAge:timeline.retirementAge},b={...s.accounts},preMonths=timeline.preMonths;
  const preReturns=monthlyRateDistribution(s.market.preRetirementMeanReturn,s.market.preRetirementStdDev),cashGrowth=monthly(.02),incomeTax=taxesEnabled?ordinaryIncomeTax:()=>0;
  const rh=s.rothHistory,rothLedger=new RothConversionLedger(rh.contributionBasis,rh.firstContributionYear,rh.conversions),ruleOf55=ruleOf55Applies(s);
  for(let i=0;i<preMonths;i++){const growth=sampleMonthlyRate(preReturns,rng);b.pretax*=1+growth;b.roth*=1+growth;b.taxable*=1+growth;b.cash*=1+cashGrowth;if(hasFutureSavings(s)){const year=usesCalendarDates(s)?Number(addCalendarMonths(s.household.asOfDate||localCalendarDate(),i+1).slice(0,4)):2026+Math.floor(i/12);depositSavings(b,rothLedger,s.contributions,i,year);if(s.household.filingStatus==='Married')depositSavings(b,rothLedger,s.spouseContributions,i,year);}sum(b);}
  const married=h.filingStatus==='Married',spouseAtRet=timeline.spouseAtRet;
  // The reporting cutoff must not determine death or move terminal care sooner.
  // Draw the full lifetime from the table, then truncate cash flows separately.
  const [death,spouseDeath]=drawLifetimes(h,spouseAtRet,married,rng,fixedDeathAges,lifespanQuantiles);
  const spouseDeathPrimary=h.retirementAge+spouseDeath-spouseAtRet,houseDeath=married?Math.max(death,spouseDeathPrimary):death;
  const primaryDeathYear=Math.floor((Math.round(death*12)-Math.round(h.retirementAge*12))/12),spouseDeathYear=Math.floor((Math.round(spouseDeath*12)-Math.round(spouseAtRet*12))/12);
  const firstDeathYear=Math.min(primaryDeathYear,spouseDeathYear);
  const ltc=ltcStart(s,h.retirementAge,death,rng),spouseLtc=married?ltcStart(s,spouseAtRet,spouseDeath,rng):null,spouseLtcPrimary=spouseLtc===null?null:h.retirementAge+spouseLtc-spouseAtRet;
  const birth=timeline.birthYear,spouseBirth=timeline.spouseBirthYear,primaryFactor=retirementBenefitFactor(birth,s.socialSecurity.claimAge*12),age67Factor=retirementBenefitFactor(birth,67*12);
  const spousalClaim=Math.max(744,s.socialSecurity.spouseClaimAge*12,Math.round(spouseAtRet*12)+s.socialSecurity.claimAge*12-Math.round(h.retirementAge*12)),survivorClaim=Math.max(720,s.socialSecurity.spouseClaimAge*12,Math.round(spouseAtRet*12)+Math.round(death*12)-Math.round(h.retirementAge*12));
  const spouseFactor=spousalBenefitFactor(spouseBirth,spousalClaim),survivorFactor=combinedSurvivorBenefitFactor(birth,s.socialSecurity.claimAge*12,death*12,spouseBirth,survivorClaim);
  const infMean=monthly(s.spending.generalInflationMean),healthMean=monthly(s.healthcare.healthcareInflationMean),incomeGrowth=s.guaranteedIncome.annualIncome>0?monthly(s.guaranteedIncome.annualIncrease):0,inflation=monthlyRateDistribution(s.spending.generalInflationMean,s.spending.generalInflationStdDev),healthInflation=monthlyRateDistribution(s.healthcare.healthcareInflationMean,s.healthcare.healthcareInflationStdDev);
  const retireBalance=sum(b),lowThreshold=retireBalance*.5;let pathFactor=spendingPath(s,0,timeline),spending=s.spending.annualBaseSpending/12*Math.pow(1+infMean,preMonths)*pathFactor;
  let rent=s.rent.monthlyRent*Math.pow(1+infMean,preMonths),home=s.home.currentValue*Math.pow(1+infMean,preMonths),seniorRent=3000*Math.pow(1+infMean,preMonths);
  const mortgage=mortgageAtRetirement(s.mortgage,preMonths);let mortgageMonths=mortgage.months,mortgageBalance=mortgage.balance;
  // Property tax and home insurance are part of base spending and stop after a sale.
  let otherMonthly=s.guaranteedIncome.annualIncome/12*Math.pow(1+incomeGrowth,preMonths),homeCosts=s.home.annualTaxesAndInsurance/12*Math.pow(1+infMean,preMonths)*pathFactor;
  let preMedicare=s.healthcare.preMedicareMonthlyPremium*Math.pow(1+healthMean,preMonths),healthIndex=Math.pow(1+healthMean,preMonths),taxIndex=Math.pow(1+infMean,preMonths),ssIndex=Math.max(1,taxIndex);
  let priorYearTaxIndex=Math.pow(1+infMean,Math.max(0,preMonths-12));
  const stockReturns=monthlyRateDistribution(s.market.stockMeanReturn,s.market.stockStdDev),bondReturns=monthlyRateDistribution(s.market.bondMeanReturn,s.market.bondStdDev);
  const seppEnd=Math.max(714,Math.round(h.retirementAge*12)+60),annualSepp=s.withdrawalStrategy.seppEligible&&h.retirementAge<59.5?seppPayment(b.pretax,h.retirementAge):0;
  const primaryHorizon=h.targetEndAge-h.retirementAge,spouseHorizon=h.targetEndAge-spouseAtRet,horizon=fixedDeathAges?Math.max(0,houseDeath-h.retirementAge):Math.max(primaryHorizon,married?spouseHorizon:primaryHorizon);
  const yearEnd=[sum(b)],chart=[sum(b)],incomeHistory=[],monthlyBalances=captureMonthlyBalances?[]:null,taxYears=captureTaxDetails?[]:null;let annualOrdinaryIncome=0,annualSocialSecurity=0,annualTaxPaid=0,annualPretaxDistributions=0,annualRmd=0,annualConversions=0,yearTaxIndex=taxIndex,failureAge=null,homeSold=false;
  // Today's dollars divide each balance by this path's own general price level.
  const today=captureTodayDollars?{yearEnd:[sum(b)/taxIndex],chart:[sum(b)/taxIndex],...(monthlyBalances?{monthlyBalances:[]}:{}),...(captureMonthlyDetails?{monthlyPriceIndex:[]}:{})}:null;
  const stopAge=Math.max(h.retirementAge,Math.min(h.retirementAge+horizon,houseDeath-Math.max(0,horizonReductionYears)));
  function checkCosts(){requireFinite(spending,rent,home,seniorRent,otherMonthly,homeCosts,preMedicare,healthIndex,taxIndex,priorYearTaxIndex,ssIndex,annualSepp,mortgageBalance);}
  checkCosts();
  const monthlyDetails=captureMonthlyDetails?[]:null;let monthlyCashFlow=null;
  function recordMonth(month){
    if(!monthlyDetails)return;
    today?.monthlyPriceIndex.push(taxIndex);
    const accounts={...b,cash:Math.max(0,b.cash)},portfolio=accounts.pretax+accounts.roth+accounts.taxable+accounts.cash;
    const netAssets=portfolio+home-mortgageBalance;
    requireFinite(portfolio,netAssets);
    monthlyDetails.push({month,age:(Math.round(h.retirementAge*12)+month)/12,date:usesCalendarDates(s)?addCalendarMonths(forecastRetirementDate(s),month):null,...accounts,home,mortgage:mortgageBalance,portfolio,netAssets,unfundedAmount:Math.max(0,-b.cash),...(monthlyCashFlow?{cashFlow:monthlyCashFlow}:{})});
  }
  recordMonth(0);
  function recordTaxYear(taxYear,status){if(taxYears)taxYears.push({taxYear,status,ordinaryIncome:annualOrdinaryIncome,socialSecurity:annualSocialSecurity,tax:annualTaxPaid,pretaxDistributions:annualPretaxDistributions,requiredMinimumDistribution:annualRmd,conversions:annualConversions,accounts:{...b}});}
  function sellHome(){
    if(homeSold||home<=0)return;
    b.cash+=home-mortgageBalance;
    sum(b);
    home=0;mortgageBalance=0;mortgageMonths=0;homeSold=true;
  }
  let completedMonths=0;
  for(let m=0;m<Math.round(horizon*12);m++){
    const ageMonths=Math.round(h.retirementAge*12)+m,spouseMonths=Math.round(spouseAtRet*12)+m,monthInYear=m%12;
    if(ageMonths>=Math.round(stopAge*12))break;
    if(monthlyBalances){monthlyBalances[m]=sum(b);if(today)today.monthlyBalances[m]=monthlyBalances[m]/taxIndex;}
    const primaryAlive=ageMonths<Math.round(death*12),spouseAlive=married&&ageMonths<Math.round(spouseDeathPrimary*12),both=married&&primaryAlive&&spouseAlive,alive=Number(primaryAlive)+Number(spouseAlive),modelYear=Math.floor(m/12),taxYear=timeline.retirementYear+modelYear;
    // Joint tax treatment lasts through the modeled year of the first death;
    // monthly household costs and benefits still follow who is currently alive.
    const status=married?(modelYear<=firstDeathYear?'Married':'Single'):h.filingStatus;
    // Senior eligibility applies for the entire modeled tax year. A deceased
    // spouse qualifies in the death year only if they reached 65 before dying.
    const yearEndMonths=Math.round(h.retirementAge*12)+(modelYear+1)*12;
    const spouseYearEndMonths=Math.round(spouseAtRet*12)+(modelYear+1)*12;
    const seniors=Number((primaryAlive||modelYear===primaryDeathYear)&&Math.min(yearEndMonths,Math.round(death*12))>=780)+Number(married&&(spouseAlive||modelYear===spouseDeathYear)&&Math.min(spouseYearEndMonths,Math.round(spouseDeath*12))>=780);
    // With no separate spouse account input, treat the pool as primary-owned,
    // then as the surviving spouse's own account after the year of death.
    const spouseOwns=married&&modelYear>primaryDeathYear,ownerAge=spouseOwns?spouseMonths/12:ageMonths/12;
    const rothQualified=rothLedger.isQualified(taxYear,ownerAge,!primaryAlive&&!spouseOwns);
    if(monthInYear===0)annualRmd=requiredMinimumDistribution(b.pretax,Math.floor(ownerAge),spouseOwns?spouseBirth:birth);
    const seppProtected=annualSepp>0&&ageMonths<seppEnd&&primaryAlive;
    const inLtc=primaryAlive&&ltc!==null&&ageMonths>=Math.round(ltc*12),spouseInLtc=spouseAlive&&spouseLtcPrimary!==null&&ageMonths>=Math.round(spouseLtcPrimary*12),ltcPeople=Number(inLtc)+Number(spouseInLtc),outside=alive-ltcPeople,replaceSpending=alive>0&&outside===0;
    // Sell as soon as nobody alive remains at home, before this month's costs.
    if(replaceSpending)sellHome();
    // An underwater sale first uses cash; any remaining payoff needs a funded,
    // taxed withdrawal this month instead of silently cancelling the debt.
    const saleShortfall=homeSold?Math.max(0,-b.cash):0;
    if(saleShortfall)b.cash=0;
    const mortgageCost=!homeSold&&mortgageMonths>0?s.mortgage.monthlyPayment:0,rentCost=outside<=0?0:homeSold?seniorRent:rent;
    const reduction=retireBalance>0&&sum(b)<lowThreshold?s.spending.lowPortfolioSpendingReduction:0,baseSpending=Math.max(0,spending-(homeSold?homeCosts:0))*(1-reduction)*(married&&!both?.84:1),ltcCost=ltcPeople>0?Math.max(0,s.longTermCare.annualCost)/12*Math.max(0,healthIndex)*ltcPeople:0;
    const pia=s.socialSecurity.annualBenefitAt67/age67Factor/12*ssIndex,primaryClaim=s.socialSecurity.claimAge*12;
    let social=primaryAlive&&ageMonths>=primaryClaim?pia*primaryFactor:0;
    if(married&&spouseAlive){if(primaryAlive){if(ageMonths>=primaryClaim&&spouseMonths>=spousalClaim)social+=pia*spouseFactor;}else if(spouseMonths>=survivorClaim)social+=pia*survivorFactor;}
    const pensionStartMonths=s.guaranteedIncome.startAge*12+(s.guaranteedIncome.startAgeMonths??0);
    const pensionSurvivorEligible=Math.round(death*12)>=pensionStartMonths;
    const guaranteed=ageMonths>=pensionStartMonths&&alive>0?(primaryAlive?otherMonthly:pensionSurvivorEligible?otherMonthly*s.guaranteedIncome.survivorPercent:0):0;
    const prePeople=Number(primaryAlive&&ageMonths<780)+Number(spouseAlive&&spouseMonths<780),medPeople=Number(primaryAlive&&ageMonths>=780)+Number(spouseAlive&&spouseMonths>=780);
    const preCost=preMedicare*prePeople;
    // Draw returns once, in the same random-stream order, so the initial income
    // estimate can use the cash strategy that applies to this modeled month.
    const stock=sampleMonthlyRate(stockReturns,rng),bond=sampleMonthlyRate(bondReturns,rng);
    requireFinite(stock,bond);
    let medCost=0;
    if(s.healthcare.includeMedicarePremiums&&medPeople){
      const history=incomeHistory.length>=2?incomeHistory[incomeHistory.length-2]:null;
      medCost=medicarePremium(history?.income??0,history?.status??status,medPeople,healthIndex,yearTaxIndex,taxYear,priorYearTaxIndex);
      if(history===null){
        // Before a lookback exists, estimate annual income with the available
        // pretax balance and Roth history; nonqualified earnings create income.
        const annualSS=social*12,estimatedDistribution=seppProtected?Math.min(b.pretax,annualSepp):Math.min(b.pretax,annualRmd),annualOther=guaranteed*12+estimatedDistribution,estimatedInterest=Math.max(0,b.cash)*.02;
        const available={...b,pretax:Math.max(0,b.pretax-estimatedDistribution)};
        for(let tier=0;tier<6;tier++){
          const annualNeed=((replaceSpending?0:baseSpending)+mortgageCost+rentCost+ltcCost+preCost+medCost+saleShortfall)*12;
          const stockPart=allocation(s,b,annualNeed),portReturn=stockPart*stock+(1-stockPart)*bond;
          const cashFirst=s.withdrawalStrategy.useCashReserveDuringDrawdowns&&portReturn<s.withdrawalStrategy.drawdownTrigger&&b.cash>0;
          const estimate=withdrawalPlan(annualNeed,annualSS,status,annualOther,available,cashFirst,yearTaxIndex,seniors,taxYear,0,{ordinary:estimatedInterest,social:0,tax:0},rothLedger,0,{pretaxAvailable:seppProtected?0:available.pretax,incomeTax,rothQualified});
          const other=estimate.taxableDraw+estimate.rothTaxableEarnings+annualOther+estimatedInterest,income=other+taxableSocialSecurity(other,annualSS,status);
          const premium=medicarePremium(income,status,medPeople,healthIndex,yearTaxIndex,taxYear,priorYearTaxIndex);
          if(premium===medCost)break;
          medCost=premium;
        }
      }
    }
    const need=(replaceSpending?0:baseSpending)+mortgageCost+rentCost+preCost+medCost+ltcCost+saleShortfall;
    const annualNeed=need*12,stockPart=allocation(s,b,annualNeed),portReturn=stockPart*stock+(1-stockPart)*bond;
    const cashInterest=Math.max(0,b.cash)*cashGrowth;
    requireFinite(need,annualNeed,portReturn,cashInterest);
    b.pretax*=1+portReturn;b.roth*=1+portReturn;b.taxable*=1+portReturn;b.cash+=cashInterest;
    sum(b);
    const beforeWithdrawals=monthlyDetails?{...b}:null;
    const seppDistribution=seppProtected?Math.min(b.pretax,annualSepp/12):0;
    // Spending distributions already taken this year satisfy the RMD. Spread
    // the remaining minimum through the year; finish it in a final partial year.
    const finalMonth=ageMonths+1>=Math.round(stopAge*12),rmdTarget=annualRmd*(finalMonth?1:(monthInYear+1)/12);
    const rmdDistribution=Math.min(Math.max(0,b.pretax-seppDistribution),Math.max(0,rmdTarget-annualPretaxDistributions-seppDistribution));
    const scheduledDistribution=seppDistribution+rmdDistribution;b.pretax-=scheduledDistribution;
    const cashFirst=s.withdrawalStrategy.useCashReserveDuringDrawdowns&&portReturn<s.withdrawalStrategy.drawdownTrigger&&b.cash>0;
    // Eligibility is an explicit assertion about the employer plan and calendar
    // year of separation; the qualifying year can begin before the 55th birthday,
    // but never for a retirement before 54.
    const penalty=taxesEnabled&&s.withdrawalStrategy.applyEarlyWithdrawalPenalty&&ownerAge<59.5&&!(ruleOf55&&!spouseOwns)&&(primaryAlive||spouseOwns)?.10:0;
    const rothPenalty=taxesEnabled&&s.withdrawalStrategy.applyEarlyWithdrawalPenalty&&ownerAge<59.5&&(primaryAlive||spouseOwns)?.10:0;
    const plan=withdrawalPlan(need,social,status,guaranteed+scheduledDistribution,b,cashFirst,yearTaxIndex,seniors,taxYear,penalty,{ordinary:annualOrdinaryIncome+cashInterest,social:annualSocialSecurity,tax:annualTaxPaid},rothLedger,rothPenalty,{pretaxAvailable:seppProtected?0:b.pretax,incomeTax,rothQualified});
    let portfolioWithdrawal=plan.gross;if(cashFirst){const cashDraw=Math.min(b.cash,portfolioWithdrawal);b.cash-=cashDraw;portfolioWithdrawal-=cashDraw;}withdrawAccounts(b,portfolioWithdrawal,seppProtected?['roth','taxable']:['pretax','roth','taxable'],rothLedger,taxYear);
    if(monthlyDetails){
      // Capture the existing calculations without changing their timing or draws.
      monthlyCashFlow={expenses:need,socialSecurity:social,guaranteedIncome:guaranteed,seppDistribution,rmdDistribution,additionalWithdrawal:plan.gross,incomeTax:plan.totalTax-annualTaxPaid,earlyPenalty:Math.max(0,social+guaranteed+scheduledDistribution+plan.gross-plan.totalTax+annualTaxPaid-plan.net),surplus:Math.max(0,plan.net-need),cashFirst,seppProtected,conversionAmount:0,conversionTax:0,accountWithdrawals:Object.fromEntries(Object.keys(beforeWithdrawals).map(key=>[key,Math.max(0,Math.min(beforeWithdrawals[key],beforeWithdrawals[key]-b[key]))]))};
    }
    const surplus=Math.max(0,plan.net-need);if(surplus>.01)b.cash+=surplus;
    annualOrdinaryIncome+=plan.taxableDraw+plan.rothTaxableEarnings+scheduledDistribution+guaranteed+cashInterest;annualPretaxDistributions+=plan.taxableDraw+scheduledDistribution;annualSocialSecurity+=social;annualTaxPaid=plan.totalTax;
    if(taxesEnabled&&s.rothConversion.enabled&&!seppProtected&&monthInYear===11){const conversion=rothConversionPlan(b.pretax,annualOrdinaryIncome,s.rothConversion.marginalRateCap,status,yearTaxIndex,seniors,taxYear,annualSocialSecurity);if(conversion.amount>0){if(monthlyDetails){monthlyCashFlow.conversionAmount=conversion.amount;monthlyCashFlow.conversionTax=conversion.tax;}b.pretax-=conversion.amount;b.roth+=conversion.amount;rothLedger.add(conversion.amount,taxYear);withdrawConversionTax(b,conversion.tax,rothLedger,taxYear,rothPenalty);annualOrdinaryIncome+=conversion.amount;annualConversions+=conversion.amount;annualTaxPaid+=conversion.tax;}}
    if(!homeSold&&mortgageMonths>0){mortgageBalance=payMortgage(mortgageBalance,mortgageCost,mortgage.rate);mortgageMonths--;}
    sum(b);requireFinite(annualOrdinaryIncome,annualSocialSecurity,annualTaxPaid,annualPretaxDistributions,annualConversions,mortgageBalance);
    if(b.cash<0&&b.cash>-.01)b.cash=0;
    if(b.cash<0&&failureAge===null){sellHome();if(b.cash<0){failureAge=ageMonths/12;if(monthlyBalances){monthlyBalances[m]=0;if(today)today.monthlyBalances[m]=0;}yearEnd.push(0);today?.yearEnd.push(0);recordTaxYear(taxYear,status);recordMonth(m+1);break;}}
    completedMonths=m+1;
    if(monthInYear===11){yearEnd.push(sum(b));chart.push(sum(b));if(today){today.yearEnd.push(sum(b)/taxIndex);today.chart.push(sum(b)/taxIndex);}}
    const inf=sampleMonthlyRate(inflation,rng),healthInf=sampleMonthlyRate(healthInflation,rng),nextFactor=spendingPath(s,m+1,timeline),change=nextFactor/Math.max(.0001,pathFactor);
    spending*=(1+inf)*change;rent*=1+inf;if(!homeSold)home*=1+inf;seniorRent*=1+inf;otherMonthly*=1+incomeGrowth;homeCosts*=(1+inf)*change;preMedicare*=1+healthInf;healthIndex*=1+healthInf;taxIndex*=1+inf;pathFactor=nextFactor;
    checkCosts();
    recordMonth(m+1);
    if(monthInYear===11){
      // Keep the price level at the last COLA. A rebound following deflation
      // does not raise benefits until it exceeds that reference level.
      ssIndex=Math.max(ssIndex,taxIndex);
      incomeHistory.push({income:annualOrdinaryIncome+taxableSocialSecurity(annualOrdinaryIncome,annualSocialSecurity,status),status});recordTaxYear(taxYear,status);
      annualOrdinaryIncome=0;annualSocialSecurity=0;annualTaxPaid=0;annualPretaxDistributions=0;annualConversions=0;
      priorYearTaxIndex=yearTaxIndex;yearTaxIndex=taxIndex;
    }
  }
  // A retirement with extra months can end between annual observations.
  if(failureAge===null&&completedMonths%12!==0){yearEnd.push(sum(b));today?.yearEnd.push(sum(b)/taxIndex);const modelYear=Math.floor((completedMonths-1)/12);recordTaxYear(timeline.retirementYear+modelYear,married?(modelYear<=firstDeathYear?'Married':'Single'):h.filingStatus);}
  const censored=stopAge<houseDeath,survivedThroughAge=censored?stopAge:h.retirementAge+Math.max(0,Math.ceil(houseDeath-h.retirementAge)-1);
  return {success:failureAge===null,failureAge,yearEnd,chart,survivedThroughAge,deathAge:houseDeath,observationEndAge:stopAge,censored,...(monthlyBalances?{monthlyBalances}:{}),...(monthlyDetails?{monthlyDetails}:{}),...(taxYears?{taxYears}:{}),...(today?{today}:{})};
}

function riskBreakdown(s,paths,lifespanQuantiles=null){
  // Use a bounded, evenly spaced subset of completed paths and common random
  // numbers. These are sensitivity checks, not a decomposition of failure causes.
  const count=Math.min(paths.length,128),indices=Array.from({length:count},(_,i)=>Math.floor(i*paths.length/count));
  const baselineFailures=indices.filter(i=>!paths[i].success).length;
  const variants=[
    {key:'inflation',label:'General inflation',change:x=>{x.spending.generalInflationMean=0;x.spending.generalInflationStdDev=0;},description:'No general inflation',next:'Compare lower and higher general inflation assumptions.'},
    {key:'spending',label:'Spending',change:x=>setAnnualBaseSpending(x,x.spending.annualBaseSpending*.95),description:'5% lower annual base spending',next:'Test a 5% lower spending scenario.'},
    {key:'taxes',label:'Federal taxes',options:{taxesEnabled:false},description:'No federal income tax or early-withdrawal penalties',next:'Compare Roth conversions and withdrawal strategies with taxes included.'},
    {key:'healthcare',label:'Healthcare',change:x=>{x.healthcare.preMedicareMonthlyPremium=0;x.healthcare.includeMedicarePremiums=false;x.longTermCare.enabled=false;},description:'No modeled health premiums or long-term-care costs',next:'Compare health premiums and long-term-care costs.'},
    {key:'market',label:'Market variability',change:x=>{x.market.preRetirementStdDev=0;x.market.stockStdDev=0;x.market.bondStdDev=0;},description:'Constant returns at the entered annual means',next:'Compare market volatility and the cash reserve strategy.'},
    {key:'longevity',label:'Longevity',options:{horizonReductionYears:5},description:'Household lifetime ends five years earlier',next:'Compare a longer modeling horizon and later retirement.'}
  ];
  const checks=variants.map(check=>{
    const variant=structuredClone(s);check.change?.(variant);
    let avoidedShortfalls=0,introducedShortfalls=0;
    for(const i of indices){
      const result=runOne(variant,new JavaRandom(BigInt(s.seed)+BigInt(i)*STRIDE),{...check.options,lifespanQuantiles:lifespanQuantiles?.[i]??null});
      if(!paths[i].success&&result.success)avoidedShortfalls++;
      if(paths[i].success&&!result.success)introducedShortfalls++;
    }
    const netReduction=avoidedShortfalls-introducedShortfalls;
    return {key:check.key,label:check.label,description:check.description,avoidedShortfalls,introducedShortfalls,netReduction,variantFailures:baselineFailures-netReduction,recommendedNextTest:check.next};
  });
  const best=checks.reduce((best,check)=>check.netReduction>(best?.netReduction??0)?check:best,null);
  const values=Object.fromEntries(checks.map(c=>[c.key,c.netReduction>0?`${c.netReduction} fewer shortfalls`:c.netReduction<0?`${-c.netReduction} more shortfalls`:'No reduction']));
  const summary=`Sensitivity checks compare ${count} paired paths (${baselineFailures} original shortfalls). They change one assumption at a time; effects overlap and do not identify every cause. ${count<=FREE_SIMULATION_PATHS?'Small runs are a preview only.':''}`.trim();
  return {...values,checks,simulationCount:count,baselineFailures,method:'paired assumption sensitivity',summary,primaryRisk:best?.key??'none',recommendedNextTest:best?`${best.description} reduced shortfalls by ${best.netReduction} in the ${count}-path sensitivity check. ${best.recommendedNextTest}`:'No sensitivity check reduced shortfalls. Compare spending, income, retirement age, and combined assumptions.'};
}
// An additional illustration follows constant entered rates and fixed long
// lifespans. It is separate from all sampled outcomes and never changes inputs.
export function runSteadySimulation(s,{captureTodayDollars=false}={}){
  const plan=structuredClone(s);
  if(usesCalendarDates(plan))plan.household.asOfDate ||= localCalendarDate();
  const timeline=scenarioTimeline(plan),married=plan.household.filingStatus==='Married';
  const lifespan=age=>age<=85?95:age+10;
  const primaryDeathAge=lifespan(timeline.currentAge);
  const spouseCurrentAge=married?(usesCalendarDates(plan)?calendarMonthsBetween(plan.household.spouseBirthday,plan.household.asOfDate)/12:plan.household.spouseCurrentAge):null;
  const spouseDeathAge=married?lifespan(spouseCurrentAge):null;
  const houseDeath=married?Math.max(primaryDeathAge,timeline.retirementAge+spouseDeathAge-timeline.spouseAtRet):primaryDeathAge;
  // Do not grow accounts to a retirement that begins after everyone has died.
  if(Math.round(houseDeath*12)<=Math.round(timeline.retirementAge*12))return {primaryDeathAge,spouseDeathAge,endingBalance:null,success:null,endReason:'before-retirement',monthlyDetails:[],...(captureTodayDollars?{priceIndexes:[]}:{})};
  plan.market.preRetirementStdDev=0;plan.market.stockStdDev=0;plan.market.bondStdDev=0;
  plan.spending.generalInflationStdDev=0;plan.healthcare.healthcareInflationStdDev=0;
  plan.longTermCare.enabled=false;
  const path=runOne(plan,{normal:mean=>mean,nextDouble:()=>.5},{captureMonthlyDetails:true,captureTodayDollars,fixedDeathAges:{primary:primaryDeathAge,spouse:spouseDeathAge}});
  return {primaryDeathAge,spouseDeathAge,endingBalance:Math.max(0,path.yearEnd.at(-1)),success:path.success,endReason:path.success?'lifespan':'shortfall',monthlyDetails:path.monthlyDetails,...(captureTodayDollars?{priceIndexes:path.today.monthlyPriceIndex}:{})};
}
function meanBalancePath(paths){
  const maxYears=Math.max(...paths.map(x=>x.chart.length)),meanPath=[];
  for(let y=0;y<maxYears;y++){const positive=paths.map(x=>x.chart[y]).filter(x=>x>0);if(positive.length)meanPath.push({yearsInRetirement:y,balance:positive.reduce((a,b)=>a+b,0)/positive.length});}
  return meanPath;
}
// The same balance summaries, with each path deflated by its own inflation.
function todayDollarSummary(paths,age,steadyPriceIndexes,includePathPoints){
  const real=paths.map(p=>({...p,...p.today})),endings=real.map(p=>Math.max(0,p.yearEnd[p.yearEnd.length-1])).sort((a,b)=>a-b);
  return {medianEndingBalance:medianOfSorted(endings),pessimisticEndingBalance:percentile(endings,.1),optimisticEndingBalance:percentile(endings,.9),balanceBands:buildBalanceBands(real,age),meanPath:meanBalancePath(real),pathPoints:includePathPoints?buildPathPoints(real):[],steadyPriceIndexes};
}

export function runSimulation(s,onProgress=()=>{},options={}){
  if(usesCalendarDates(s)){s=structuredClone(s);s.household.asOfDate ||= localCalendarDate();}
  const errors=validateScenario(s);if(errors.length)throw new Error(errors.join(' '));
  const n=s.numberOfSimulations,paths=[],endings=[],failures=[];let successes=0;
  const lifespanQuantiles=options.stratifyPreviewLifespans===false?null:previewLifespanQuantiles(n,s.seed);
  for(let i=0;i<n;i++){const seed=BigInt(s.seed)+BigInt(i)*STRIDE,path=runOne(s,new JavaRandom(seed),{captureMonthlyBalances:true,captureTodayDollars:true,lifespanQuantiles:lifespanQuantiles?.[i]??null});paths.push(path);endings.push(Math.max(0,path.yearEnd[path.yearEnd.length-1]));if(path.success)successes++;else if(path.failureAge!==null)failures.push(path.failureAge);if(i%25===0)onProgress((i+1)/n);}
  // Every path is done; the summaries and sensitivity checks follow.
  onProgress(1);
  const {priceIndexes:steadyPriceIndexes,...steadySimulation}=runSteadySimulation(s,{captureTodayDollars:true});
  const p=successes/n,sorted=endings.sort((a,b)=>a-b),bands=buildBalanceBands(paths,retirementAge(s));
  failures.sort((a,b)=>a-b);const buckets=new Map();for(const age of failures){const start=Math.floor(age/5)*5;buckets.set(start,(buckets.get(start)||0)+1);}
  const failureAgeBuckets=[...buckets].map(([start,count])=>({label:`${start}-${start+4}`,count,shareOfFailures:count/failures.length}));
  const notFailedByAge=buildFundingSurvival(paths,retirementAge(s));
  const meanPath=meanBalancePath(paths),includePathPoints=options.includePathPoints!==false;
  return {scenarioId:s.id,successProbability:p,medianEndingBalance:medianOfSorted(sorted),steadySimulation,pessimisticEndingBalance:percentile(sorted,.1),optimisticEndingBalance:percentile(sorted,.9),medianFailureAge:failures.length?medianOfSorted(failures):null,failureAgeBuckets,balanceBands:bands,notFailedByAge,meanPath,pathPoints:includePathPoints?buildPathPoints(paths):[],riskBreakdown:options.includeRiskAnalysis===false?null:riskBreakdown(s,paths,lifespanQuantiles),todayDollars:todayDollarSummary(paths,retirementAge(s),steadyPriceIndexes,includePathPoints),provenance:{engineVersion:scenarioEngineVersion(s),engineCadence:'Monthly cashflow model with annual result bands',taxTableVersion:'2026 federal brackets with senior-aware deductions',mortalityModelVersion:'SSA Trustees Alt2 2025 annual death probabilities',randomSeed:s.seed,simulationCount:n},generatedAtEpochMillis:Date.now()};
}

// The candidates a planning-target search examines, in search order.
export function decisionPlan(s,targetReadiness=.80,simulationCount=180,maxRetirementAge=70){
  const errors=validateScenario(s);if(errors.length)throw new Error(errors.join(' '));
  if(targetReadiness<0||targetReadiness>1)throw new Error('Target readiness must be between 0% and 100%.');
  const count=clamp(simulationCount,50,10000),h=s.household;
  const timeline=scenarioTimeline(s),firstAge=Math.ceil(timeline.currentAge)+(usesCalendarDates(s)&&addCalendarMonths(h.birthday,Math.ceil(timeline.currentAge)*12)<(h.asOfDate||localCalendarDate())?1:0),lastAge=Math.floor(Math.min(maxRetirementAge,h.targetEndAge-1,h.filingStatus==='Married'?timeline.retirementAge+h.targetEndAge-timeline.spouseAtRet-1:Infinity));
  const ages=[];for(let age=firstAge;age<=lastAge;age++)ages.push(age);
  // Bound the screening work even for very large, otherwise valid plans.
  // Qualifying amounts at this ceiling remain explicitly reported as "At least".
  const safeSpendingSearchLimit=Math.min(1000000,Math.max(s.spending.annualBaseSpending*3,250000));
  // Allocation depends on spending, so readiness need not be monotonic.
  // Search every reported $500 candidate from the upper bound downward.
  // Base spending includes the retained home bills; a smaller total would omit
  // part of those costs before the home sale. Round the minimum upward.
  const maximumCandidate=Math.floor(safeSpendingSearchLimit/500)*500,minimumCandidate=Math.ceil(s.home.annualTaxesAndInsurance/500)*500,amounts=[];
  for(let step=maximumCandidate/500;step>=minimumCandidate/500;step--)amounts.push(step*500);
  return {targetReadiness,count,ages,amounts,firstAge,lastAge,safeSpendingSearchLimit,maximumCandidate};
}
// Candidates that cannot meet the target need no remaining paths and report 0.
// Qualifying candidates run every path, so their readiness is exact.
function screenedReadiness(variant,count,targetReadiness){
  const errors=validateScenario(variant);if(errors.length)throw new Error(errors.join(' '));
  let successes=0;
  for(let i=0;i<count;i++){
    if(runOne(variant,new JavaRandom(BigInt(variant.seed)+BigInt(i)*STRIDE)).success)successes++;
    if((successes+count-i-1)/count<targetReadiness)return 0;
  }
  return successes/count;
}
// Keep the plan's own early-withdrawal penalty setting so targets match a full run at that age.
export function retirementAgeReadiness(s,age,count,targetReadiness){
  const variant=structuredClone(s);setRetirementAge(variant,age);variant.numberOfSimulations=count;variant.seed=s.seed+10000;
  return screenedReadiness(variant,count,targetReadiness);
}
export function spendingReadiness(s,spending,count,targetReadiness){
  const variant=structuredClone(s);setAnnualBaseSpending(variant,spending);variant.numberOfSimulations=count;variant.seed=s.seed+20000;
  return screenedReadiness(variant,count,targetReadiness);
}
export function candidateReadiness(s,{kind,value,count,targetReadiness}){
  return kind==='age'?retirementAgeReadiness(s,value,count,targetReadiness):spendingReadiness(s,value,count,targetReadiness);
}
function decisionResult(plan,age,spending,safeSpendingAtSearchLimit){
  return {targetReadiness:plan.targetReadiness,simulationCount:plan.count,earliestRetirementAge:age?.value??null,earliestRetirementReadiness:age?.readiness??null,safeAnnualSpending:spending?.value??null,safeSpendingReadiness:spending?.readiness??null,safeSpendingAtSearchLimit,safeSpendingSearchLimit:plan.safeSpendingSearchLimit,retirementAgeSearchStart:plan.firstAge,retirementAgeSearchEnd:plan.lastAge};
}
export function estimateDecision(s,targetReadiness=.80,simulationCount=180,maxRetirementAge=70){
  const plan=decisionPlan(s,targetReadiness,simulationCount,maxRetirementAge);
  const task=(kind,value)=>({kind,value,count:plan.count,targetReadiness:plan.targetReadiness});
  const first=(kind,values)=>{for(const value of values){const readiness=candidateReadiness(s,task(kind,value));if(readiness>=plan.targetReadiness)return {value,readiness};}return null;};
  const age=first('age',plan.ages),spending=first('spending',plan.amounts);
  const atLimit=spending?.value===plan.maximumCandidate&&(plan.safeSpendingSearchLimit===plan.maximumCandidate||candidateReadiness(s,task('spending',plan.safeSpendingSearchLimit))>=plan.targetReadiness);
  return decisionResult(plan,age,spending,atLimit);
}
// Finds the first qualifying value in order while evaluating up to `concurrency`
// candidates at once. A later candidate never wins over an earlier one, and an
// error counts only where a sequential scan would have reached it. The answer
// settles once every earlier candidate is known; later speculative work is ignored.
function firstQualifying(values,evaluate,targetReadiness,concurrency,onChecked){
  return new Promise((resolve,reject)=>{
    const outcomes=new Map();let next=0,active=0,known=0,winner=Infinity,failed=Infinity,failure=null,done=false;
    const settle=()=>{
      const limit=Math.min(values.length,winner,failed);
      while(known<limit&&outcomes.has(known))known++;
      if(known<limit)return;
      done=true;
      if(failed<winner)reject(failure);else resolve(winner<values.length?{value:values[winner],readiness:outcomes.get(winner)}:null);
    };
    const launch=()=>{
      while(!done&&active<concurrency&&next<Math.min(values.length,winner,failed)){
        const index=next++;active++;
        Promise.resolve().then(()=>evaluate(values[index])).then(value=>{
          outcomes.set(index,value);if(value>=targetReadiness&&index<winner)winner=index;
        },error=>{outcomes.set(index,undefined);if(index<failed){failed=index;failure=error;}}).then(()=>{
          active--;
          if(done)return;
          try{onChecked?.();}catch{}
          settle();launch();
        });
      }
    };
    launch();settle();
  });
}
// The same search as estimateDecision, with candidates evaluated through an
// asynchronous `evaluate(task)` so a caller can spread them across workers.
export async function searchDecision(s,{evaluate,concurrency=1,onProgress,targetReadiness=.80,simulationCount=180,maxRetirementAge=70}={}){
  const plan=decisionPlan(s,targetReadiness,simulationCount,maxRetirementAge);
  const run=evaluate??(task=>candidateReadiness(s,task)),width=Math.max(1,Math.floor(concurrency)||1);
  const task=(kind,value)=>({kind,value,count:plan.count,targetReadiness:plan.targetReadiness});
  const progress={phase:'ages',checkedAges:0,totalAges:plan.ages.length,checkedAmounts:0,totalAmounts:plan.amounts.length};
  const report=()=>onProgress?.({...progress});
  report();
  const age=await firstQualifying(plan.ages,value=>run(task('age',value)),plan.targetReadiness,width,()=>{progress.checkedAges++;report();});
  progress.phase='spending';report();
  const spending=await firstQualifying(plan.amounts,value=>run(task('spending',value)),plan.targetReadiness,width,()=>{progress.checkedAmounts++;report();});
  let atLimit=false;
  if(spending?.value===plan.maximumCandidate)atLimit=plan.safeSpendingSearchLimit===plan.maximumCandidate||await run(task('spending',plan.safeSpendingSearchLimit))>=plan.targetReadiness;
  return decisionResult(plan,age,spending,atLimit);
}
