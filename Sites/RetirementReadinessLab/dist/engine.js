import {buildPathPoints,buildBalanceBands,buildFundingSurvival,medianOfSorted} from './chart-data.js';
import {ENGINE_VERSION, validateScenario, retirementAge} from './model.js';
import {maleMortality,femaleMortality} from './mortality.js';
import {taxableSocialSecurity,ordinaryIncomeTax,rothConversionPlan} from './tax.js';
import {RothConversionLedger} from './roth-conversions.js';
import {mortgageAtRetirement,payMortgage} from './mortgage.js';
import {retirementBenefitFactor,spousalBenefitFactor,combinedSurvivorBenefitFactor} from './social-security.js';

const STRIDE=-7046029254386353131n, MASK=(1n<<48n)-1n;
const monthly=r=>Math.pow(Math.max(.0001,1+r),1/12)-1;
const sd=s=>s/Math.sqrt(12);
const sum=b=>b.pretax+b.roth+b.taxable+b.cash;
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
  const table=gender==='Female'?femaleMortality:maleMortality;let age=start;
  while(age<end&&age<120){
    const next=Math.min(Math.floor(age)+1,end),fraction=next-age,q=table[Math.floor(age)]??1;
    if(rng.nextDouble()<(fraction===1?q:1-Math.pow(1-q,fraction)))return next;
    age=next;
  }
  return Math.min(age,end);
}
function ltcStart(s,start,death,rng){const draw=rng.nextDouble();if(!s.longTermCare.enabled||death<65)return null;const p=death<75?.25:death<85?.45:death<95?.60:.70;return draw>p?null:Math.max(start,death-s.longTermCare.averageDurationYears-(s.longTermCare.averageDurationMonths??0)/12);}
function allocation(s,b,annualSpending){if(annualSpending<=0)return s.postRetirementAllocation.stock50xOrMore;const ratio=invested(b)/annualSpending,a=s.postRetirementAllocation;return ratio<30?a.stockUnder30x:ratio<35?a.stock30xTo35x:ratio<40?a.stock35xTo40x:ratio<45?a.stock40xTo45x:ratio<50?a.stock45xTo50x:a.stock50xOrMore;}
function spendingPath(s,offset){if(s.spending.spendingPathModel==='Flat')return 1;const h=s.household,months=Math.max(0,Math.min(Math.round(retirementAge(s)*12)+offset,85*12)-Math.max(65,h.currentAge)*12),married=h.filingStatus==='Married';return Math.max(married?.60:.65,Math.pow(1-(married?.024:.017),months/12));}
function medicarePremium(income,status,people,healthInflation,taxInflation){const limits=status==='Married'?[218000,274000,342000,410000,750000,Infinity]:[109000,137000,171000,205000,500000,Infinity];const b=[0,81.20,202.90,324.60,446.30,487],d=[0,14.50,37.50,60.40,83.30,91];const idx=limits.findIndex(cap=>Math.max(0,income)<=cap*Math.max(.0001,taxInflation));return (202.90+38.99+b[idx]+d[idx])*Math.max(0,healthInflation)*people;}
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
function withdrawalPlan(need,ss,status,other,b,cashFirst,taxInflation,seniors,taxYear,penalty,ytd={ordinary:0,social:0,tax:0},ledger=null,rothPenalty=0){
  function estimate(gross){
    const taxableDraw=Math.min(Math.max(0,b.pretax),Math.max(0,gross-(cashFirst?Math.max(0,b.cash):0)));
    const ordinary=ytd.ordinary+other+taxableDraw,social=ytd.social+ss;
    const taxableSS=taxableSocialSecurity(ordinary,social,status);
    const totalTax=ordinaryIncomeTax(ordinary+taxableSS,status,taxInflation,seniors,taxYear),tax=totalTax-ytd.tax;
    const rothDraw=Math.max(0,gross-(cashFirst?Math.max(0,b.cash):0)-taxableDraw),recapture=rothPenalty*(ledger?.withdrawal(rothDraw,b.roth,taxYear)??0);
    return {gross,taxableDraw,totalTax,net:ss+other+gross-tax-taxableDraw*Math.max(0,penalty)-recapture};
  }
  if(estimate(0).net>=need)return estimate(0);
  let low=0,high=Math.max(0,need-ss-other)*1.8+10000;
  while(estimate(high).net<need)high*=2;
  for(let i=0;i<32;i++){const mid=(low+high)/2;if(estimate(mid).net>=need)high=mid;else low=mid;}
  return estimate(high);
}
const life=[84.6,83.7,82.8,81.8,80.8,79.8,78.8,77.9,76.9,75.9,74.9,73.9,72.9,71.9,70.9,69.9,69,68,67,66,65,64.1,63.1,62.1,61.1,60.2,59.2,58.2,57.3,56.3,55.3,54.4,53.4,52.5,51.5,50.5,49.6,48.6,47.7,46.7,45.7,44.8,43.8,42.9,41.9,41,40,39,38.1,37.1,36.2,35.3,34.3,33.4,32.5,31.6,30.6,29.8,28.9,28,27.1,26.2,25.4,24.5,23.7,22.9,22,21.2,20.4,19.6,18.8,18,17.2,16.4,15.6,14.8,14.1,13.3,12.6,11.9,11.2,10.5,9.9,9.3,8.7,8.1,7.6,7.1,6.6,6.1,5.7,5.3,4.9,4.6,4.3,4,3.7,3.4,3.2,3,2.8,2.6,2.5,2.3,2.2,2.1,2.1,2.1,2,2,2,2,2,1.9,1.9,1.8,1.8,1.6,1.4,1.1,1];
function seppPayment(balance,age){const years=life[Math.max(0,Math.floor(age))];if(!years||balance<=0)return 0;return balance/((1-Math.pow(1.05,-years))/.05);}

export function runOne(s,rng,{captureMonthlyBalances=false}={}){
  const h={...s.household,retirementAge:retirementAge(s)},b={...s.accounts},preMonths=Math.round((h.retirementAge-h.currentAge)*12),yearsToRet=preMonths/12;
  const preMean=monthly(s.market.preRetirementMeanReturn),preSd=sd(s.market.preRetirementStdDev),cashGrowth=monthly(.02);
  for(let i=0;i<preMonths;i++){const growth=rng.normal(preMean,preSd);b.pretax*=1+growth;b.roth*=1+growth;b.taxable*=1+growth;b.cash*=1+cashGrowth;}
  const rothLedger=new RothConversionLedger(b.roth);
  const married=h.filingStatus==='Married',spouseAtRet=h.spouseCurrentAge+yearsToRet;
  const death=sampleDeathAge(h.gender,h.retirementAge,h.targetEndAge,rng),spouseDeath=married?sampleDeathAge(h.spouseGender,spouseAtRet,h.targetEndAge,rng):death;
  const spouseDeathPrimary=h.retirementAge+spouseDeath-spouseAtRet,houseDeath=married?Math.max(death,spouseDeathPrimary):death;
  const primaryDeathYear=Math.floor((Math.round(death*12)-Math.round(h.retirementAge*12))/12),spouseDeathYear=Math.floor((Math.round(spouseDeath*12)-Math.round(spouseAtRet*12))/12);
  const firstDeathYear=Math.min(primaryDeathYear,spouseDeathYear);
  const ltc=ltcStart(s,h.retirementAge,death,rng),spouseLtc=married?ltcStart(s,spouseAtRet,spouseDeath,rng):null,spouseLtcPrimary=spouseLtc===null?null:h.retirementAge+spouseLtc-spouseAtRet;
  const birth=2026-h.currentAge,spouseBirth=2026-h.spouseCurrentAge,primaryFactor=retirementBenefitFactor(birth,s.socialSecurity.claimAge*12),age67Factor=retirementBenefitFactor(birth,67*12);
  const spousalClaim=Math.max(744,s.socialSecurity.spouseClaimAge*12,Math.round(spouseAtRet*12)+s.socialSecurity.claimAge*12-Math.round(h.retirementAge*12)),survivorClaim=Math.max(720,s.socialSecurity.spouseClaimAge*12,Math.round(spouseAtRet*12)+Math.round(death*12)-Math.round(h.retirementAge*12));
  const spouseFactor=spousalBenefitFactor(spouseBirth,spousalClaim),survivorFactor=combinedSurvivorBenefitFactor(birth,s.socialSecurity.claimAge*12,death*12,spouseBirth,survivorClaim);
  const infMean=monthly(s.spending.generalInflationMean),infSd=sd(s.spending.generalInflationStdDev),healthMean=monthly(s.healthcare.healthcareInflationMean),healthSd=sd(s.healthcare.healthcareInflationStdDev),incomeGrowth=monthly(s.guaranteedIncome.annualIncrease);
  const retireBalance=sum(b),lowThreshold=retireBalance*.5;let pathFactor=spendingPath(s,0),spending=s.spending.annualBaseSpending/12*Math.pow(1+infMean,preMonths)*pathFactor;
  let rent=s.rent.monthlyRent*Math.pow(1+infMean,preMonths),home=s.home.currentValue*Math.pow(1+infMean,preMonths),seniorRent=3000*Math.pow(1+infMean,preMonths);
  const mortgage=mortgageAtRetirement(s.mortgage,preMonths);let mortgageMonths=mortgage.months,mortgageBalance=mortgage.balance;
  let otherMonthly=s.guaranteedIncome.annualIncome/12*Math.pow(1+incomeGrowth,preMonths),homeCosts=s.budget.isAppliedToAnnualBaseSpending?(s.budget.appliedAnnualHomeCosts??(s.budget.annualPropertyTaxes+s.budget.annualHomeInsurance))/12*Math.pow(1+infMean,preMonths)*pathFactor:0;
  let preMedicare=s.healthcare.preMedicareMonthlyPremium*Math.pow(1+healthMean,preMonths),healthIndex=Math.pow(1+healthMean,preMonths),taxIndex=Math.pow(1+infMean,preMonths),ssIndex=Math.pow(1+Math.max(0,s.spending.generalInflationMean),yearsToRet),annualInf=1;
  const stockMean=monthly(s.market.stockMeanReturn),stockSd=sd(s.market.stockStdDev),bondMean=monthly(s.market.bondMeanReturn),bondSd=sd(s.market.bondStdDev);
  const seppEnd=Math.max(714,Math.round(h.retirementAge*12)+60),annualSepp=s.withdrawalStrategy.seppEligible?seppPayment(b.pretax,h.retirementAge):0;
  const primaryHorizon=h.targetEndAge-h.retirementAge,spouseHorizon=h.targetEndAge-spouseAtRet,horizon=Math.max(primaryHorizon,married?spouseHorizon:primaryHorizon);
  const yearEnd=[sum(b)],chart=[sum(b)],incomeHistory=[],monthlyBalances=captureMonthlyBalances?[]:null;let annualOrdinaryIncome=0,annualSocialSecurity=0,annualTaxPaid=0,yearTaxIndex=taxIndex,failureAge=null,homeSold=false;
  function sellHome(){
    if(homeSold||home<=0)return;
    b.cash+=home-mortgageBalance;
    home=0;mortgageBalance=0;mortgageMonths=0;homeSold=true;
  }
  let completedMonths=0;
  for(let m=0;m<Math.round(horizon*12);m++){
    const ageMonths=Math.round(h.retirementAge*12)+m,spouseMonths=Math.round(spouseAtRet*12)+m,monthInYear=m%12;
    if(ageMonths>=Math.round(houseDeath*12))break;
    if(monthlyBalances)monthlyBalances[m]=sum(b);
    const primaryAlive=ageMonths<Math.round(death*12),spouseAlive=married&&ageMonths<Math.round(spouseDeathPrimary*12),both=married&&primaryAlive&&spouseAlive,alive=Number(primaryAlive)+Number(spouseAlive),modelYear=Math.floor(m/12),taxYear=2026+Math.floor(yearsToRet)+modelYear;
    // Joint tax treatment lasts through the modeled year of the first death;
    // monthly household costs and benefits still follow who is currently alive.
    const status=married?(modelYear<=firstDeathYear?'Married':'Single'):h.filingStatus;
    // A deceased spouse retains age-based deductions in the year of death only,
    // and must have reached 65 before dying.
    const seniors=Number((primaryAlive||modelYear===primaryDeathYear)&&Math.min(ageMonths,Math.round(death*12))>=780)+Number(married&&(spouseAlive||modelYear===spouseDeathYear)&&Math.min(spouseMonths,Math.round(spouseDeath*12))>=780);
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
    const guaranteed=ageMonths>=s.guaranteedIncome.startAge*12+(s.guaranteedIncome.startAgeMonths??0)&&alive>0?(primaryAlive?otherMonthly:otherMonthly*s.guaranteedIncome.survivorPercent):0;
    const prePeople=Number(primaryAlive&&ageMonths<780)+Number(spouseAlive&&spouseMonths<780),medPeople=Number(primaryAlive&&ageMonths>=780)+Number(spouseAlive&&spouseMonths>=780);
    const preCost=preMedicare*prePeople;
    // Draw returns once, in the same random-stream order, so the initial income
    // estimate can use the cash strategy that applies to this modeled month.
    const stock=rng.normal(stockMean,stockSd),bond=rng.normal(bondMean,bondSd);
    let medCost=0;
    if(s.healthcare.includeMedicarePremiums&&medPeople){
      const history=incomeHistory.length>=2?incomeHistory[incomeHistory.length-2]:null;
      medCost=medicarePremium(history?.income??0,history?.status??status,medPeople,healthIndex,yearTaxIndex);
      if(history===null){
        // Before a lookback exists, estimate annual income with the available
        // pretax balance; Roth/taxable/cash spending does not create ordinary income.
        const annualSS=social*12,annualOther=guaranteed*12;
        for(let tier=0;tier<6;tier++){
          const annualNeed=((replaceSpending?0:baseSpending)+mortgageCost+rentCost+ltcCost+preCost+medCost+saleShortfall)*12;
          const stockPart=allocation(s,b,annualNeed),portReturn=stockPart*stock+(1-stockPart)*bond;
          const cashFirst=s.withdrawalStrategy.useCashReserveDuringDrawdowns&&portReturn<s.withdrawalStrategy.drawdownTrigger&&b.cash>0;
          const estimate=withdrawalPlan(annualNeed,annualSS,status,annualOther,b,cashFirst,yearTaxIndex,seniors,taxYear,0);
          const other=estimate.taxableDraw+annualOther,income=other+taxableSocialSecurity(other,annualSS,status);
          const premium=medicarePremium(income,status,medPeople,healthIndex,yearTaxIndex);
          if(premium===medCost)break;
          medCost=premium;
        }
      }
    }
    const need=(replaceSpending?0:baseSpending)+mortgageCost+rentCost+preCost+medCost+ltcCost+saleShortfall;
    const annualNeed=need*12,stockPart=allocation(s,b,annualNeed),portReturn=stockPart*stock+(1-stockPart)*bond;
    b.pretax*=1+portReturn;b.roth*=1+portReturn;b.taxable*=1+portReturn;b.cash*=1+cashGrowth;
    rothLedger.grow(1+portReturn);
    const seppActive=s.withdrawalStrategy.seppEligible&&annualSepp>0&&ageMonths<seppEnd&&b.pretax>0,seppDistribution=seppActive?Math.min(b.pretax,annualSepp/12):0;b.pretax-=seppDistribution;
    const cashFirst=s.withdrawalStrategy.useCashReserveDuringDrawdowns&&portReturn<s.withdrawalStrategy.drawdownTrigger&&b.cash>0;
    const penalty=s.withdrawalStrategy.applyEarlyWithdrawalPenalty&&ageMonths<714&&!(s.withdrawalStrategy.ruleOf55Eligible&&h.retirementAge>=55)?.10:0;
    const rothPenalty=s.withdrawalStrategy.applyEarlyWithdrawalPenalty&&ageMonths<714?.10:0;
    const plan=withdrawalPlan(need,social,status,guaranteed+seppDistribution,b,cashFirst,yearTaxIndex,seniors,taxYear,penalty,{ordinary:annualOrdinaryIncome,social:annualSocialSecurity,tax:annualTaxPaid},rothLedger,rothPenalty);
    let portfolioWithdrawal=plan.gross;if(cashFirst){const cashDraw=Math.min(b.cash,portfolioWithdrawal);b.cash-=cashDraw;portfolioWithdrawal-=cashDraw;}withdrawAccounts(b,portfolioWithdrawal,['pretax','roth','taxable'],rothLedger,taxYear);
    const surplus=Math.max(0,plan.net-need);if(surplus>.01)b.cash+=surplus;
    annualOrdinaryIncome+=plan.taxableDraw+seppDistribution+guaranteed;annualSocialSecurity+=social;annualTaxPaid=plan.totalTax;
    if(s.rothConversion.enabled&&!seppActive&&monthInYear===11){const conversion=rothConversionPlan(b.pretax,annualOrdinaryIncome,s.rothConversion.marginalRateCap,status,yearTaxIndex,seniors,taxYear,annualSocialSecurity);if(conversion.amount>0){b.pretax-=conversion.amount;b.roth+=conversion.amount;rothLedger.add(conversion.amount,taxYear);withdrawConversionTax(b,conversion.tax,rothLedger,taxYear,rothPenalty);annualOrdinaryIncome+=conversion.amount;}}
    if(!homeSold&&mortgageMonths>0){mortgageBalance=payMortgage(mortgageBalance,mortgageCost,mortgage.rate);mortgageMonths--;}
    if(sum(b)<0&&sum(b)>-.01)b.cash-=sum(b);
    if(sum(b)<0&&failureAge===null){sellHome();if(sum(b)<0){failureAge=ageMonths/12;if(monthlyBalances)monthlyBalances[m]=0;yearEnd.push(0);break;}}
    completedMonths=m+1;
    if(monthInYear===11){yearEnd.push(sum(b));chart.push(sum(b));}
    const inf=Math.max(monthly(-.05),rng.normal(infMean,infSd)),healthInf=Math.max(monthly(-.02),rng.normal(healthMean,healthSd)),nextFactor=spendingPath(s,m+1),change=nextFactor/Math.max(.0001,pathFactor);
    spending*=(1+inf)*change;rent*=1+inf;if(!homeSold)home*=1+inf;seniorRent*=1+inf;otherMonthly*=1+incomeGrowth;homeCosts*=(1+inf)*change;preMedicare*=1+healthInf;healthIndex*=1+healthInf;taxIndex*=1+inf;annualInf*=1+inf;pathFactor=nextFactor;
    if(monthInYear===11){ssIndex*=Math.max(1,annualInf);annualInf=1;incomeHistory.push({income:annualOrdinaryIncome+taxableSocialSecurity(annualOrdinaryIncome,annualSocialSecurity,status),status});annualOrdinaryIncome=0;annualSocialSecurity=0;annualTaxPaid=0;yearTaxIndex=taxIndex;}
  }
  // A retirement with extra months can end between annual observations.
  if(failureAge===null&&completedMonths%12!==0)yearEnd.push(sum(b));
  const survivedThroughAge=h.retirementAge+Math.max(0,Math.ceil(houseDeath-h.retirementAge)-1);
  return {success:failureAge===null,failureAge,yearEnd,chart,survivedThroughAge,deathAge:houseDeath,...(monthlyBalances?{monthlyBalances}:{})};
}

function riskBreakdown(s,p){const h={...s.household,retirementAge:retirementAge(s)},prePeople=Number(h.retirementAge<65)+Number(h.filingStatus==='Married'&&h.spouseCurrentAge+h.retirementAge-h.currentAge<65),healthBurden=s.healthcare.preMedicareMonthlyPremium*12*prePeople/Math.max(1,s.spending.annualBaseSpending),total=Object.values(s.accounts).reduce((a,b)=>a+b,0),taxBurden=s.accounts.pretax/Math.max(1,total),spendingRatio=s.spending.annualBaseSpending/Math.max(1,total);const market=p>=.82?'Healthy':p>=.65?'Watch':'AtRisk',healthcare=healthBurden>.25?'AtRisk':healthBurden>.18||s.longTermCare.enabled?'Watch':'Healthy',taxes=taxBurden>.85?'AtRisk':taxBurden>.60?'Watch':'Healthy',spending=spendingRatio>.07?'AtRisk':spendingRatio>.045?'Watch':'Healthy',longevity=h.retirementAge<55?'AtRisk':h.retirementAge<62?'Watch':'Healthy';const entries=[['spending',spending],['taxes',taxes],['healthcare',healthcare],['longevity',longevity],['market sequence',market]],primary=(entries.find(x=>x[1]==='AtRisk')||entries.find(x=>x[1]==='Watch')||['none'])[0],next={spending:'Test a 5% lower spending scenario.',taxes:'Compare Roth conversions up to the 22% bracket.',healthcare:'Run the healthcare and long-term care stress test.',longevity:'Compare retiring two years later.','market sequence':'Test a larger cash reserve strategy.',none:'Compare Social Security claim ages.'};return {market,healthcare,taxes,spending,longevity,primaryRisk:primary,recommendedNextTest:next[primary]};}
export function runSimulation(s,onProgress=()=>{},options={}){
  const errors=validateScenario(s);if(errors.length)throw new Error(errors.join(' '));
  const n=s.numberOfSimulations,paths=[],endings=[],failures=[];let successes=0;
  for(let i=0;i<n;i++){const seed=BigInt(s.seed)+BigInt(i)*STRIDE,path=runOne(s,new JavaRandom(seed),{captureMonthlyBalances:true});paths.push(path);endings.push(Math.max(0,path.yearEnd[path.yearEnd.length-1]));if(path.success)successes++;else if(path.failureAge!==null)failures.push(path.failureAge);if(i%25===0)onProgress((i+1)/n);}
  const p=successes/n,sorted=endings.sort((a,b)=>a-b),bands=buildBalanceBands(paths,retirementAge(s));
  failures.sort((a,b)=>a-b);const buckets=new Map();for(const age of failures){const start=Math.floor(age/5)*5;buckets.set(start,(buckets.get(start)||0)+1);}
  const failureAgeBuckets=[...buckets].map(([start,count])=>({label:`${start}-${start+4}`,count,shareOfFailures:count/failures.length}));
  const notFailedByAge=buildFundingSurvival(paths,retirementAge(s));
  const maxYears=Math.max(...paths.map(x=>x.chart.length)),meanPath=[];for(let y=0;y<maxYears;y++){const positive=paths.map(x=>x.chart[y]).filter(x=>x>0);if(positive.length)meanPath.push({yearsInRetirement:y,balance:positive.reduce((a,b)=>a+b,0)/positive.length});}
  return {scenarioId:s.id,successProbability:p,medianEndingBalance:medianOfSorted(sorted),pessimisticEndingBalance:percentile(sorted,.1),optimisticEndingBalance:percentile(sorted,.9),medianFailureAge:failures.length?medianOfSorted(failures):null,failureAgeBuckets,balanceBands:bands,notFailedByAge,meanPath,pathPoints:options.includePathPoints===false?[]:buildPathPoints(paths),riskBreakdown:riskBreakdown(s,p),provenance:{engineVersion:ENGINE_VERSION,engineCadence:'Monthly cashflow model with annual result bands',taxTableVersion:'2026 federal brackets with senior-aware deductions',mortalityModelVersion:'SSA Trustees Alt2 2025 annual death probabilities',randomSeed:s.seed,simulationCount:n},generatedAtEpochMillis:Date.now()};
}

export function estimateDecision(s,targetReadiness=.80,simulationCount=180,maxRetirementAge=70){
  if(targetReadiness<0||targetReadiness>1)throw new Error('Target readiness must be between 0% and 100%.');
  const count=clamp(simulationCount,50,10000),h=s.household;
  const last=Math.min(maxRetirementAge,h.targetEndAge-1,h.filingStatus==='Married'?h.currentAge+h.targetEndAge-h.spouseCurrentAge-1:Infinity);
  let earliestRetirementAge=null,earliestRetirementReadiness=null;
  for(let age=h.currentAge;age<=last;age++){
    // Keep the plan's own early-withdrawal penalty setting so targets match a full run at that age.
    const variant=structuredClone(s);variant.household.retirementAge=age;variant.household.retirementAgeMonths=0;variant.numberOfSimulations=count;variant.seed=s.seed+10000;
    const readiness=runSimulation(variant,()=>{},{includePathPoints:false}).successProbability;
    if(readiness>=targetReadiness){earliestRetirementAge=age;earliestRetirementReadiness=readiness;break;}
  }
  function readinessFor(spending){const variant=structuredClone(s);variant.spending.annualBaseSpending=spending;variant.numberOfSimulations=count;variant.seed=s.seed+20000;return runSimulation(variant,()=>{},{includePathPoints:false}).successProbability;}
  let safeAnnualSpending=null,safeSpendingReadiness=null,safeSpendingAtSearchLimit=false;
  const safeSpendingSearchLimit=Math.max(s.spending.annualBaseSpending*3,250000);
  if(readinessFor(0)>=targetReadiness){
    const maxSpend=safeSpendingSearchLimit;let low=0,high=Math.min(Math.max(s.spending.annualBaseSpending*1.5,40000),maxSpend),highReady=readinessFor(high);
    while(highReady>=targetReadiness&&high<maxSpend){low=high;high=Math.min(high*1.35,maxSpend);highReady=readinessFor(high);}
    // Reaching the limit is a lower bound, not the highest spending that meets the target.
    if(highReady>=targetReadiness){low=high;safeSpendingAtSearchLimit=true;}else for(let i=0;i<11;i++){const mid=(low+high)/2;if(readinessFor(mid)>=targetReadiness)low=mid;else high=mid;}
    let candidate=Math.floor(low/500)*500;
    while(candidate>0&&readinessFor(candidate)<targetReadiness)candidate=Math.max(0,candidate-500);
    safeAnnualSpending=candidate;safeSpendingReadiness=readinessFor(candidate);
    if(candidate<Math.floor(maxSpend/500)*500)safeSpendingAtSearchLimit=false;
  }
  return {targetReadiness,simulationCount:count,earliestRetirementAge,earliestRetirementReadiness,safeAnnualSpending,safeSpendingReadiness,safeSpendingAtSearchLimit,safeSpendingSearchLimit};
}
