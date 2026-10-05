import {scenarioTimeline,usesCalendarDates,addCalendarMonths} from './model.js';
import {monthlyRateDistribution,sampleMonthlyRate} from './annual-rates.js';
import {taxableSocialSecurity,ordinaryIncomeTax,rothConversionPlan} from './tax.js';
import {mortgageAtRetirement,payMortgage} from './mortgage.js';
import {PersonAccounts} from './person-accounts.js';
import {personSocialSecurity,personPensions} from './person-income.js';
import {monthly,requireFinite,sum,drawLifetimes,ltcStart,allocation,spendingPath,medicarePremium,seppPayment} from './engine.js';
import {planEvents,applyMonthlyEvents,pathMetrics} from './plan-events.js';

export function runSeparatePeople(s,rng,{captureMonthlyBalances=false,captureMonthlyDetails=false,captureTaxDetails=false,captureTodayDollars=false,taxesEnabled=true,horizonReductionYears=0,fixedDeathAges=null,lifespanQuantiles=null,stress=null,captureMetrics=false,accountCalendar}={}){
  const timeline=scenarioTimeline(s),h={...s.household,currentAge:timeline.currentAge,retirementAge:timeline.retirementAge},b={...s.accounts},preMonths=timeline.preMonths;
  const events=planEvents(s,stress),metrics=captureMetrics?pathMetrics():null;
  const preReturns=monthlyRateDistribution(s.market.preRetirementMeanReturn,s.market.preRetirementStdDev),cashGrowth=monthly(.02),incomeTax=taxesEnabled?ordinaryIncomeTax:()=>0;
  const pools=new PersonAccounts(s,b,accountCalendar);
  for(let i=0;i<preMonths;i++){pools.events(i,true);pools.grow(sampleMonthlyRate(preReturns,rng),cashGrowth);pools.deposits(i,true);sum(b);}
  const married=h.filingStatus==='Married',spouseAtRet=timeline.spouseAtRet;
  // The reporting cutoff must not determine death or move terminal care sooner.
  // Draw the full lifetime from the table, then truncate cash flows separately.
  let [death,spouseDeath]=drawLifetimes(h,spouseAtRet,married,rng,fixedDeathAges,lifespanQuantiles);
  if(events?.minDeathAge){death=Math.max(death,events.minDeathAge);spouseDeath=married?Math.max(spouseDeath,events.minDeathAge):death;}
  const spouseDeathPrimary=h.retirementAge+spouseDeath-spouseAtRet,houseDeath=married?Math.max(death,spouseDeathPrimary):death;
  const primaryDeathYear=Math.floor((Math.round(death*12)-Math.round(h.retirementAge*12))/12),spouseDeathYear=Math.floor((Math.round(spouseDeath*12)-Math.round(spouseAtRet*12))/12);
  const firstDeathYear=Math.min(primaryDeathYear,spouseDeathYear);
  const ltc=ltcStart(s,h.retirementAge,death,rng,events?.forceCare),spouseLtc=married?ltcStart(s,spouseAtRet,spouseDeath,rng,events?.forceCare):null,spouseLtcPrimary=spouseLtc===null?null:h.retirementAge+spouseLtc-spouseAtRet;
  const infMean=monthly(s.spending.generalInflationMean),healthMean=monthly(s.healthcare.healthcareInflationMean),inflation=monthlyRateDistribution(s.spending.generalInflationMean,s.spending.generalInflationStdDev),healthInflation=monthlyRateDistribution(s.healthcare.healthcareInflationMean,s.healthcare.healthcareInflationStdDev);
  const retireBalance=sum(b),lowThreshold=retireBalance*.5;let pathFactor=spendingPath(s,0,timeline),spending=s.spending.annualBaseSpending/12*Math.pow(1+infMean,preMonths)*pathFactor;
  let rent=s.rent.monthlyRent*Math.pow(1+infMean,preMonths),home=s.home.currentValue*Math.pow(1+infMean,preMonths),seniorRent=3000*Math.pow(1+infMean,preMonths);
  // Rent after a planned sale is entered in today's dollars.
  let saleRent=events?.home?.mode==='rent'?events.home.monthlyRent*Math.pow(1+infMean,preMonths):0,plannedSale=false,keptHomeShare=1;
  const mortgage=mortgageAtRetirement(s.mortgage,preMonths);let mortgageMonths=mortgage.months,mortgageBalance=mortgage.balance;
  // Property tax and home insurance are part of base spending and stop after a sale.
  let homeCosts=s.home.annualTaxesAndInsurance/12*Math.pow(1+infMean,preMonths)*pathFactor;
  let preMedicare=s.healthcare.preMedicareMonthlyPremium*Math.pow(1+healthMean,preMonths),healthIndex=Math.pow(1+healthMean,preMonths),taxIndex=Math.pow(1+infMean,preMonths),ssIndex=Math.max(1,taxIndex);
  let priorYearTaxIndex=Math.pow(1+infMean,Math.max(0,preMonths-12));
  const stockReturns=monthlyRateDistribution(s.market.stockMeanReturn,s.market.stockStdDev),bondReturns=monthlyRateDistribution(s.market.bondMeanReturn,s.market.bondStdDev);
  const primaryHorizon=h.targetEndAge-h.retirementAge,spouseHorizon=h.targetEndAge-spouseAtRet,horizon=fixedDeathAges?Math.max(0,houseDeath-h.retirementAge):Math.max(primaryHorizon,married?spouseHorizon:primaryHorizon);
  const yearEnd=[sum(b)],chart=[sum(b)],incomeHistory=[],monthlyBalances=captureMonthlyBalances?[]:null,taxYears=captureTaxDetails?[]:null;let annualOrdinaryIncome=0,annualSocialSecurity=0,annualTaxPaid=0,annualPretaxDistributions=0,annualRmd=0,annualConversions=0,yearTaxIndex=taxIndex,failureAge=null,homeSold=false;
  // Today's dollars divide each balance by this path's own general price level.
  const today=captureTodayDollars?{yearEnd:[sum(b)/taxIndex],chart:[sum(b)/taxIndex],...(monthlyBalances?{monthlyBalances:[]}:{}),...(captureMonthlyDetails?{monthlyPriceIndex:[]}:{})}:null;
  const stopAge=Math.max(h.retirementAge,Math.min(h.retirementAge+horizon,houseDeath-Math.max(0,horizonReductionYears)));
  function checkCosts(){requireFinite(spending,rent,home,seniorRent,homeCosts,preMedicare,healthIndex,taxIndex,priorYearTaxIndex,ssIndex,mortgageBalance);}
  checkCosts();
  const monthlyDetails=captureMonthlyDetails?[]:null;let monthlyCashFlow=null;
  function recordMonth(month){
    if(!monthlyDetails)return;
    today?.monthlyPriceIndex.push(taxIndex);
    const accounts={...b,cash:Math.max(0,b.cash)},portfolio=accounts.pretax+accounts.roth+accounts.taxable+accounts.cash;
    const netAssets=portfolio+home-mortgageBalance;
    requireFinite(portfolio,netAssets);
    monthlyDetails.push({month,age:(Math.round(h.retirementAge*12)+month)/12,date:usesCalendarDates(s)?addCalendarMonths(timeline.startDate,month):null,...accounts,ownerAccounts:pools.snapshot(),home,mortgage:mortgageBalance,portfolio,netAssets,unfundedAmount:Math.max(0,-b.cash),...(monthlyCashFlow?{cashFlow:monthlyCashFlow}:{})});
  }
  recordMonth(0);
  function recordTaxYear(taxYear,status){if(taxYears)taxYears.push({taxYear,status,ordinaryIncome:annualOrdinaryIncome,socialSecurity:annualSocialSecurity,tax:annualTaxPaid,pretaxDistributions:annualPretaxDistributions,requiredMinimumDistribution:annualRmd,conversions:annualConversions,accounts:{...b}});}
  function sellHome(){
    if(homeSold||home<=0)return;
    b.cash+=home-mortgageBalance;
    sum(b);
    home=0;mortgageBalance=0;mortgageMonths=0;homeSold=true;
  }
  // Selling to rent works like any sale. Downsizing keeps a smaller home with
  // proportionally lower property costs; an underwater payoff is funded this month.
  function plannedHomeSale(){
    if(events.home.mode==='rent'){sellHome();plannedSale=true;return 0;}
    const share=events.home.downsizeShare;b.cash+=home*(1-share)-mortgageBalance;home*=share;keptHomeShare=share;mortgageBalance=0;mortgageMonths=0;
    const shortfall=Math.max(0,-b.cash);if(shortfall)b.cash=0;sum(b);return shortfall;
  }
  function closeMetrics(){metrics?.close(annualTaxPaid,annualConversions,taxIndex);}
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
    // Owner-specific withdrawal rules; inherited accounts become the surviving
    // spouse’s own accounts after the modeled death year.
    pools.configure(m,taxYear,[primaryDeathYear,spouseDeathYear],[primaryAlive,spouseAlive],seppPayment);
    const transfers=pools.events(m);annualConversions+=transfers.conversion;
    if(!taxesEnabled){for(const p of pools.people)p.penalty=p.pretaxPenalty=0;for(const a of pools.employer)a.penalty=0;}
    annualRmd=pools.people.reduce((n,p)=>n+p.rmd,0);
    const seppProtected=pools.people.some(p=>p.protected);
    const inLtc=primaryAlive&&ltc!==null&&ageMonths>=Math.round(ltc*12),spouseInLtc=spouseAlive&&spouseLtcPrimary!==null&&ageMonths>=Math.round(spouseLtcPrimary*12),ltcPeople=Number(inLtc)+Number(spouseInLtc),outside=alive-ltcPeople,replaceSpending=alive>0&&outside===0;
    const downsizeShortfall=events?.homeSaleDue(ageMonths)&&alive>0&&!homeSold&&home>0?plannedHomeSale():0;
    // Sell as soon as nobody alive remains at home, before this month's costs.
    if(replaceSpending)sellHome();
    // An underwater sale first uses cash; any remaining payoff needs a funded,
    // taxed withdrawal this month instead of silently cancelling the debt.
    const saleShortfall=homeSold?Math.max(0,-b.cash):0;
    if(saleShortfall)b.cash=0;
    const mortgageCost=!homeSold&&mortgageMonths>0?s.mortgage.monthlyPayment:0,rentCost=outside<=0?0:homeSold?(plannedSale?saleRent:seniorRent):rent;
    const reduction=retireBalance>0&&sum(b)<lowThreshold?s.spending.lowPortfolioSpendingReduction:0,baseSpending=Math.max(0,spending-(homeSold?homeCosts:keptHomeShare<1?homeCosts*(1-keptHomeShare):0))*(1-reduction)*(married&&!both?.84:1),ltcCost=ltcPeople>0?Math.max(0,s.longTermCare.annualCost)/12*Math.max(0,healthIndex)*ltcPeople:0;
    const social=personSocialSecurity(s,timeline,[ageMonths/12,spouseMonths/12],[primaryAlive,spouseAlive],[death,spouseDeath],ssIndex);
    const guaranteed=personPensions(s,[ageMonths/12,spouseMonths/12],[primaryAlive,spouseAlive],[death,spouseDeath],preMonths+m);
    const support=pools.people.reduce((n,p,i)=>n+(p.alive&&m<p.retire?s.workingIncome[i===0?'primaryAnnualNet':'spouseAnnualNet']/12*Math.pow(1+s.workingIncome.annualIncrease,(preMonths+m)/12):0),0);
    const prePeople=Number(primaryAlive&&m>=pools.people[0].retire&&ageMonths<780)+Number(spouseAlive&&m>=pools.people[1].retire&&spouseMonths<780),medPeople=Number(primaryAlive&&ageMonths>=780)+Number(spouseAlive&&spouseMonths>=780);
    const preCost=preMedicare*prePeople;
    // Draw returns once, in the same random-stream order, so the initial income
    // estimate can use the cash strategy that applies to this modeled month.
    const stock=sampleMonthlyRate(stockReturns,rng),bond=sampleMonthlyRate(bondReturns,rng);
    requireFinite(stock,bond);
    const partTimeWork=events?events.work(ageMonths,primaryAlive&&m>=pools.people[0].retire,taxIndex):0;
    let medCost=0;
    if(s.healthcare.includeMedicarePremiums&&medPeople){
      const history=incomeHistory.length>=2?incomeHistory[incomeHistory.length-2]:null;
      medCost=medicarePremium(history?.income??0,history?.status??status,medPeople,healthIndex,yearTaxIndex,taxYear,priorYearTaxIndex);
      if(history===null){
        // Before a lookback exists, estimate annual income with the available
        // pretax balance and Roth history; nonqualified earnings create income.
        // Take-home work pay covers costs before estimating taxable withdrawals.
        const annualSS=social*12,estimatedDistribution=pools.people.filter(p=>p.protected).reduce((n,p)=>n+p.sepp,0),annualOther=guaranteed*12+estimatedDistribution+transfers.conversion,estimatedInterest=Math.max(0,b.cash)*.02;
        // A depleted owner's year-start RMD cannot be funded by the other
        // owner's savings or added again to nonqualified Roth earnings.
        const estimatedRmd=pools.people.reduce((n,p)=>n+Math.min(Math.max(0,p.pretax),p.rmd),0);
        for(let tier=0;tier<6;tier++){
          const annualNeed=Math.max(0,(replaceSpending?0:baseSpending)+mortgageCost+rentCost+ltcCost+preCost+medCost+saleShortfall-partTimeWork)*12;
          const stockPart=allocation(s,b,annualNeed),portReturn=stockPart*stock+(1-stockPart)*bond;
          const cashFirst=s.withdrawalStrategy.useCashReserveDuringDrawdowns&&portReturn<s.withdrawalStrategy.drawdownTrigger&&b.cash>0;
          const estimate=pools.plan(annualNeed,annualSS,status,annualOther,cashFirst,yearTaxIndex,seniors,taxYear,{ordinary:estimatedInterest,social:0,tax:0},incomeTax,false,support*12);
          const other=Math.max(estimate.taxableDraw+estimatedDistribution,estimatedRmd)+estimate.rothTaxableEarnings+guaranteed*12+estimatedInterest+transfers.conversion,income=other+taxableSocialSecurity(other,annualSS,status);
          const premium=medicarePremium(income,status,medPeople,healthIndex,yearTaxIndex,taxYear,priorYearTaxIndex);
          if(premium===medCost)break;
          medCost=premium;
        }
      }
      metrics?.markSurcharge(medCost,healthIndex,medPeople);
    }
    let need=(replaceSpending?0:baseSpending)+mortgageCost+rentCost+preCost+medCost+ltcCost+saleShortfall,allocationNeed=need,workSurplus=0;
    if(events){const e=applyMonthlyEvents(need,partTimeWork,events.oneTime(ageMonths,alive>0,taxIndex)+downsizeShortfall);need=e.need;allocationNeed=e.allocationNeed;workSurplus=e.surplus;}
    const annualNeed=allocationNeed*12,stockPart=allocation(s,b,annualNeed);let portReturn=stockPart*stock+(1-stockPart)*bond;
    if(events)portReturn=events.marketReturn(m,portReturn,stockPart);
    const cashInterest=Math.max(0,b.cash)*cashGrowth;
    requireFinite(need,annualNeed,portReturn,cashInterest);
    pools.grow(portReturn,cashGrowth);
    const deposited=pools.deposits(m);
    sum(b);
    const beforeWithdrawals=monthlyDetails?{...b}:null;
    const finalMonth=ageMonths+1>=Math.round(stopAge*12),scheduled=pools.scheduled(finalMonth),seppDistribution=scheduled.sepp,rmdDistribution=scheduled.rmd,scheduledDistribution=scheduled.total;
    const cashFirst=s.withdrawalStrategy.useCashReserveDuringDrawdowns&&portReturn<s.withdrawalStrategy.drawdownTrigger&&b.cash>0;
    const plan=pools.plan(need,social,status,guaranteed+scheduledDistribution,cashFirst,yearTaxIndex,seniors,taxYear,{ordinary:annualOrdinaryIncome+cashInterest,social:annualSocialSecurity,tax:annualTaxPaid},incomeTax,false,support);
    pools.consume(plan);
    if(monthlyDetails){
      // Capture the existing calculations without changing their timing or draws.
      monthlyCashFlow={expenses:need,socialSecurity:social,guaranteedIncome:guaranteed,seppDistribution,rmdDistribution,additionalWithdrawal:plan.gross,incomeTax:plan.totalTax-annualTaxPaid,earlyPenalty:plan.penalties,workingSupport:support,savingsContributions:deposited,surplus:Math.max(0,plan.net-need),cashFirst,seppProtected,conversionAmount:0,conversionTax:0,...(pools.employer.length?{inPlanConversion:transfers.conversion,employerRothRollover:transfers.rollover}:{}),accountWithdrawals:Object.fromEntries(Object.keys(beforeWithdrawals).map(key=>[key,Math.max(0,Math.min(beforeWithdrawals[key],beforeWithdrawals[key]-b[key]))]))};
    }
    const surplus=Math.max(0,plan.net-need);if(surplus>.01)b.cash+=surplus;
    if(workSurplus>0)b.cash+=workSurplus;
    annualOrdinaryIncome+=plan.taxableDraw+plan.rothTaxableEarnings+scheduledDistribution+guaranteed+cashInterest;annualPretaxDistributions+=plan.taxableDraw+scheduledDistribution;annualSocialSecurity+=social;annualTaxPaid=plan.totalTax;
    if(transfers.conversion>0){
      annualOrdinaryIncome+=transfers.conversion;
      const payment=pools.plan(0,0,status,0,false,yearTaxIndex,seniors,taxYear,{ordinary:annualOrdinaryIncome,social:annualSocialSecurity,tax:annualTaxPaid},incomeTax,true);
      pools.consume(payment);annualOrdinaryIncome+=payment.taxableDraw+payment.rothTaxableEarnings;annualPretaxDistributions+=payment.taxableDraw;annualTaxPaid=payment.totalTax;
      if(monthlyCashFlow)monthlyCashFlow.inPlanConversionTax=payment.gross;
    }
    if(taxesEnabled&&s.rothConversion.enabled&&monthInYear===11){
      const available=pools.people.filter(p=>!p.protected).reduce((n,p)=>n+p.pretax,0),conversion=rothConversionPlan(available,annualOrdinaryIncome,s.rothConversion.marginalRateCap,status,yearTaxIndex,seniors,taxYear,annualSocialSecurity);
      if(conversion.amount>0){const amount=pools.convert(conversion.amount);annualOrdinaryIncome+=amount;
        const payment=pools.plan(0,0,status,0,false,yearTaxIndex,seniors,taxYear,{ordinary:annualOrdinaryIncome,social:annualSocialSecurity,tax:annualTaxPaid},incomeTax,true);
        pools.consume(payment);annualOrdinaryIncome+=payment.taxableDraw+payment.rothTaxableEarnings;annualPretaxDistributions+=payment.taxableDraw;annualConversions+=amount;annualTaxPaid=payment.totalTax;
        if(monthlyDetails){monthlyCashFlow.conversionAmount=amount;monthlyCashFlow.conversionTax=payment.gross;}
      }
    }
    if(!homeSold&&mortgageMonths>0){mortgageBalance=payMortgage(mortgageBalance,mortgageCost,mortgage.rate);mortgageMonths--;}
    sum(b);requireFinite(annualOrdinaryIncome,annualSocialSecurity,annualTaxPaid,annualPretaxDistributions,annualConversions,mortgageBalance);
    if(b.cash<0&&b.cash>-.01)b.cash=0;
    if(b.cash<0&&failureAge===null){sellHome();if(b.cash<0){failureAge=ageMonths/12;if(monthlyBalances){monthlyBalances[m]=0;if(today)today.monthlyBalances[m]=0;}yearEnd.push(0);today?.yearEnd.push(0);recordTaxYear(taxYear,status);closeMetrics();recordMonth(m+1);break;}}
    completedMonths=m+1;
    if(monthInYear===11){yearEnd.push(sum(b));chart.push(sum(b));if(today){today.yearEnd.push(sum(b)/taxIndex);today.chart.push(sum(b)/taxIndex);}}
    let inf=sampleMonthlyRate(inflation,rng);const healthInf=sampleMonthlyRate(healthInflation,rng),nextFactor=spendingPath(s,m+1,timeline),change=nextFactor/Math.max(.0001,pathFactor);
    if(events){inf=events.inflation(m,inf,monthly);saleRent*=1+inf;}
    spending*=(1+inf)*change;rent*=1+inf;if(!homeSold)home*=1+inf;seniorRent*=1+inf;homeCosts*=(1+inf)*change;preMedicare*=1+healthInf;healthIndex*=1+healthInf;taxIndex*=1+inf;pathFactor=nextFactor;
    checkCosts();
    recordMonth(m+1);
    if(monthInYear===11){
      closeMetrics();
      // Keep the price level at the last COLA. A rebound following deflation
      // does not raise benefits until it exceeds that reference level.
      ssIndex=Math.max(ssIndex,taxIndex);
      incomeHistory.push({income:annualOrdinaryIncome+taxableSocialSecurity(annualOrdinaryIncome,annualSocialSecurity,status),status});recordTaxYear(taxYear,status);
      annualOrdinaryIncome=0;annualSocialSecurity=0;annualTaxPaid=0;annualPretaxDistributions=0;annualConversions=0;
      priorYearTaxIndex=yearTaxIndex;yearTaxIndex=taxIndex;
    }
  }
  // A retirement with extra months can end between annual observations.
  if(failureAge===null&&completedMonths%12!==0){yearEnd.push(sum(b));today?.yearEnd.push(sum(b)/taxIndex);const modelYear=Math.floor((completedMonths-1)/12);recordTaxYear(timeline.retirementYear+modelYear,married?(modelYear<=firstDeathYear?'Married':'Single'):h.filingStatus);closeMetrics();}
  const censored=stopAge<houseDeath,survivedThroughAge=censored?stopAge:h.retirementAge+Math.max(0,Math.ceil(houseDeath-h.retirementAge)-1);
  return {success:failureAge===null,failureAge,yearEnd,chart,survivedThroughAge,deathAge:houseDeath,observationEndAge:stopAge,censored,...(monthlyBalances?{monthlyBalances}:{}),...(monthlyDetails?{monthlyDetails}:{}),...(taxYears?{taxYears}:{}),...(today?{today}:{}),...(metrics?{metrics:metrics.result()}:{})};
}
