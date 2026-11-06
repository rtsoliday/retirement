// Spreadsheet view: the rows, cell parsing and table builders, with no DOM.
// It edits the same saved plans as the classic planner and runs the same engine.
import {validateScenario,scenarioWarnings,setAnnualBaseSpending,syncCalendarAges,calendarDate,localCalendarDate,scenarioTimeline,primaryRetirementAge,employerRothTotal,hasEmployerRoth,MAX_DOLLAR_AMOUNT,MIN_SIMULATION_PATHS,MAX_SIMULATION_PATHS,DEFAULT_SEED,ROTH_CONVERSION_RATES} from './model.js';
import {parseMoneyInput} from './money-input.js';
import {inputSource,unknownInputPaths} from './ux-guidance.js';
import {budgetPlanGaps,needsPreMedicarePremium} from './plan-review.js';
import {balanceDisplayRows,ageYearRows,shareLabel} from './result-format.js';

export const getPath=(o,path)=>path.split('.').reduce((v,k)=>v?.[k],o);
function setPath(o,path,value){const keys=path.split('.'),last=keys.pop();keys.reduce((v,k)=>v[k],o)[last]=value;}

const married=s=>s.household.filingStatus==='Married';
const separate=s=>married(s)&&s.household.separatePeople;
const working=s=>!s.household.alreadyRetired;
const spouseWorking=s=>s.household.separatePeople?!s.household.spouseAlreadyRetired:working(s);

const YES_NO='bool',MONEY='money',PCT='pct',INT='int',DATE='date',ENUM='enum';
const row=(label,unit,type,path,note,extra={})=>({id:path,label,unit,type,path,note,...extra});
const calc=(id,label,unit,value,note,extra={})=>({id,label,unit,type:'calc',value,note,...extra});
const shortAge=value=>{if(!Number.isFinite(value))return '';const months=Math.round(value*12),extra=months%12;return extra?`${Math.floor(months/12)}y ${extra}m`:String(months/12);};
const sum=(...values)=>values.reduce((a,b)=>a+(Number(b)||0),0);
export const totalSavings=s=>sum(...Object.values(s.accounts),employerRothTotal(s),...(separate(s)?[s.spouseAccounts.pretax,s.spouseAccounts.roth]:[]));
const yearlySavings=x=>sum(x.pretax,x.employerPretax,x.roth,x.taxable,x.cash);

const savingsRows=(prefix,who,show,active)=>[
  ['Employee pre-tax','pretax','Paycheck deposits to a 401(k) or similar.'],
  ['Employer pre-tax','employerPretax','Employer match and other pre-tax employer money.'],
  ['Roth IRA','roth','Yearly Roth IRA deposits.'],
  ['Taxable','taxable','Yearly deposits to shared taxable investments.'],
  ['Cash','cash','Yearly deposits to shared cash.']
].map(([label,key,note])=>row(who+label,'$/yr',MONEY,`${prefix}.${key}`,note+' Stops at retirement.',{show,active})).concat(
  row(who+'Yearly increase','%',PCT,`${prefix}.annualIncrease`,'Growth in these deposits each year. 0 keeps them flat.',{show,active}));

const pension=(prefix,amount,startAge,startMonths,who,show)=>{
  const active=s=>getPath(s,`${prefix}.${amount}`)>0;
  return [
    row(who+'Pension','$/yr',MONEY,`${prefix}.${amount}`,'Yearly pension or annuity in today’s dollars. 0 if none.',{show}),
    row(who+'Pension start','age',INT,`${prefix}.${startAge}`,'Whole-year age the payments start.',{show,active}),
    row(who+'Start + months','mo',INT,`${prefix}.${startMonths}`,'Extra months past that age, 0 to 11.',{show,active}),
    row(who+'Pension increase','%',PCT,`${prefix}.annualIncrease`,'Yearly payment increase. 0 if fixed.',{show,active})
  ];
};

// One section per block of the sheet. `show` hides a row when no plan uses it;
// `active` dims a value the plan keeps but does not currently use.
export const SECTIONS=[
  {id:'household',label:'Household',rows:[
    row('Filing','',ENUM,'household.filingStatus','Couples are modeled as married filing jointly.',{options:[['Single','Single'],['Married','Married'],['HeadOfHousehold','Head of household']]}),
    row('Birthday','date',DATE,'household.birthday','Your date of birth.'),
    row('Already retired','Y/N',YES_NO,'household.alreadyRetired','Yes starts the forecast today and stops your deposits.'),
    row('Retirement date','date',DATE,'household.retirementDate','Planned last working day, or an optional past date if already retired.'),
    calc('age-now','Age today','age',s=>shortAge(scenarioTimeline(s).currentAge),'From your birthday.',{needs:['household.birthday']}),
    calc('age-retire','Age at retirement','age',s=>shortAge(primaryRetirementAge(s)),'From your birthday and retirement date.',{needs:['household.birthday','household.retirementDate']}),
    row('Sex','',ENUM,'household.gender','Selects the SSA life-expectancy table.',{options:[['Male','Male'],['Female','Female']]}),
    row('Spouse birthday','date',DATE,'household.spouseBirthday','Spouse’s date of birth.',{show:married}),
    row('Spouse retired','Y/N',YES_NO,'household.spouseAlreadyRetired','Yes stops spouse deposits and support.',{show:separate}),
    row('Spouse retire date','date',DATE,'household.spouseRetirementDate','Spouse’s own last working day. Shared costs start at the first retirement.',{show:separate}),
    row('Spouse sex','',ENUM,'household.spouseGender','Selects the SSA life-expectancy table.',{options:[['Male','Male'],['Female','Female']],show:married}),
    row('Max model age','age',INT,'household.targetEndAge','Where the model stops. Not a lifespan prediction.')
  ]},
  {id:'accounts',label:'Balances today',total:s=>totalSavings(s),totalOf:['accounts.','spouseAccounts.'],rows:[
    row('Pre-tax','$',MONEY,'accounts.pretax','Traditional 401(k), 403(b), IRA and similar, from statements.'),
    row('Roth IRA','$',MONEY,'accounts.roth','All Roth IRA money: contributions, conversions and earnings.'),
    row('Taxable','$',MONEY,'accounts.taxable','Brokerage investments outside retirement accounts.'),
    row('Cash','$',MONEY,'accounts.cash','Bank savings and cash. Earns 2% a year.'),
    row('Spouse pre-tax','$',MONEY,'spouseAccounts.pretax','Spouse’s own pre-tax balances. Exclude yours.',{show:separate}),
    row('Spouse Roth IRA','$',MONEY,'spouseAccounts.roth','Spouse’s own Roth IRA. Exclude yours.',{show:separate}),
    calc('employer-roth','Employer Roth','$',s=>employerRothTotal(s),'Roth 401(k) and 403(b) records. Edit them in the planner.',{show:hasEmployerRoth,money:true})
  ]},
  {id:'roth',label:'Roth history',rows:[
    row('Regular basis','$',MONEY,'rothHistory.contributionBasis','Regular contributions not yet withdrawn. Excludes conversions and growth.'),
    row('First funded','year',INT,'rothHistory.firstContributionYear','Tax year of your first Roth IRA deposit. 0 if never funded.'),
    calc('roth-conversions','Past conversions','count',s=>s.rothHistory.conversions.length,'Past conversion records. Edit them in the planner.'),
    row('Spouse basis','$',MONEY,'spouseRothHistory.contributionBasis','Spouse’s regular contributions not yet withdrawn.',{show:separate}),
    row('Spouse first funded','year',INT,'spouseRothHistory.firstContributionYear','Spouse’s first Roth IRA tax year. 0 if never funded.',{show:separate})
  ]},
  {id:'savings',label:'Yearly savings',total:s=>yearlySavings(s.contributions)+(married(s)?yearlySavings(s.spouseContributions):0),totalOf:['contributions.','spouseContributions.'],rows:[
    ...savingsRows('contributions','',undefined,working),
    ...savingsRows('spouseContributions','Spouse ',married,spouseWorking),
    row('Your take-home support','$/yr',MONEY,'workingIncome.primaryAnnualNet','Take-home pay for household costs while you work after your spouse retires.',{show:separate}),
    row('Spouse take-home support','$/yr',MONEY,'workingIncome.spouseAnnualNet','Take-home pay for household costs while your spouse works after you retire.',{show:separate}),
    row('Support increase','%',PCT,'workingIncome.annualIncrease','Yearly growth of take-home support.',{show:separate})
  ]},
  {id:'spending',label:'Spending',rows:[
    row('Base spending','$/yr',MONEY,'spending.annualBaseSpending','Yearly living costs today, including property tax. Exclude mortgage, rent and health premiums.'),
    row('Pattern','',ENUM,'spending.spendingPathModel','Declining falls gradually from 65 to 85. Steady only follows inflation.',{options:[['EmpiricalAgeDecline','Declining'],['Flat','Steady']]}),
    row('Inflation','%',PCT,'spending.generalInflationMean','Average yearly price growth for living and housing costs.'),
    row('Inflation swing','%',PCT,'spending.generalInflationStdDev','Volatility of general inflation.'),
    row('Low-portfolio cut','%',PCT,'spending.lowPortfolioSpendingReduction','Spending cut applied if the portfolio halves.'),
    row('Part-time pay','$/yr',MONEY,'partTimeIncome.annualNet','Take-home pay after retiring, in today’s dollars. 0 if none.'),
    row('Part-time until','age',INT,'partTimeIncome.endAge','Stop working at this age. Must be after retirement.',{active:s=>s.partTimeIncome.annualNet>0}),
    calc('one-time','One-time costs','count',s=>s.oneTimeExpenses?.length??0,'One-time expense records. Edit them in the planner.')
  ]},
  {id:'income',label:'Income',rows:[
    row('Social Security at 67','$/yr',MONEY,'socialSecurity.annualBenefitAt67','Your yearly benefit at 67 in today’s dollars, from my Social Security.'),
    row('Claim age','age',INT,'socialSecurity.claimAge','Age your own benefit begins, 62 to 70.'),
    ...pension('guaranteedIncome','annualIncome','startAge','startAgeMonths','',undefined),
    row('Survivor share','%',PCT,'guaranteedIncome.survivorPercent','Share of your pension your spouse keeps after your death.',{show:married}),
    row('Spouse claim age','age',INT,'socialSecurity.spouseClaimAge','Age your spouse’s benefit begins.',{show:married}),
    row('Spouse SS at 67','$/yr',MONEY,'spouseIncome.annualBenefitAt67','Spouse’s own yearly benefit at 67 in today’s dollars.',{show:separate}),
    ...pension('spouseIncome','annualPension','pensionStartAge','pensionStartAgeMonths','Spouse ',separate),
    row('Spouse survivor share','%',PCT,'spouseIncome.survivorPercent','Share of your spouse’s pension you keep after their death.',{show:separate})
  ]},
  {id:'housing',label:'Housing',rows:[
    row('Mortgage payment','$/mo',MONEY,'mortgage.monthlyPayment','Principal and interest only, excluding escrow.'),
    row('Mortgage balance','$',MONEY,'mortgage.currentBalance','Unpaid principal today.'),
    row('Years left','yr',INT,'mortgage.yearsLeft','Whole years left on the mortgage.'),
    row('+ Months left','mo',INT,'mortgage.monthsLeft','Extra months left on the mortgage.'),
    row('Home value','$',MONEY,'home.currentValue','Estimated sale value today, before the mortgage.'),
    row('Tax + insurance','$/yr',MONEY,'home.annualTaxesAndInsurance','Keep these in base spending too. This only tells the model what stops after a sale.'),
    row('Rent','$/mo',MONEY,'rent.monthlyRent','Monthly rent today. 0 if none.'),
    row('Planned sale age','age',INT,'homePlan.saleAge','Age you plan to sell. 0 means no planned sale.'),
    row('Sale plan','',ENUM,'homePlan.mode','Downsize to a smaller home, or sell and rent.',{options:[['downsize','Downsize'],['rent','Sell and rent']],active:s=>s.homePlan.saleAge>0}),
    row('Downsize share','%',PCT,'homePlan.downsizeShare','New home’s cost as a share of the old home’s value.',{active:s=>s.homePlan.saleAge>0&&s.homePlan.mode==='downsize'}),
    row('Rent after sale','$/mo',MONEY,'homePlan.monthlyRent','Monthly rent after the sale, in today’s dollars.',{active:s=>s.homePlan.saleAge>0&&s.homePlan.mode==='rent'})
  ]},
  {id:'health',label:'Healthcare',rows:[
    row('Pre-Medicare premium','$/mo',MONEY,'healthcare.preMedicareMonthlyPremium','Monthly premium per adult before 65.',{active:needsPreMedicarePremium}),
    row('Health inflation','%',PCT,'healthcare.healthcareInflationMean','Average yearly growth in premiums and care costs.'),
    row('Health inflation swing','%',PCT,'healthcare.healthcareInflationStdDev','Volatility of healthcare inflation.'),
    row('Medicare premiums','Y/N',YES_NO,'healthcare.includeMedicarePremiums','Include Medicare premiums and income surcharges.'),
    row('Long-term care risk','Y/N',YES_NO,'longTermCare.enabled','Model a chance of paid care late in life.'),
    row('Care cost','$/yr',MONEY,'longTermCare.annualCost','Yearly care cost per person in today’s dollars.',{active:s=>s.longTermCare.enabled}),
    row('Care duration','yr',INT,'longTermCare.averageDurationYears','Average whole years of care.',{active:s=>s.longTermCare.enabled}),
    row('+ Care months','mo',INT,'longTermCare.averageDurationMonths','Extra months of care, 0 to 11.',{active:s=>s.longTermCare.enabled})
  ]},
  {id:'market',label:'Market',rows:[
    row('Pre-retirement return','%',PCT,'market.preRetirementMeanReturn','Average yearly growth before retirement, before inflation.'),
    row('Pre-retirement swing','%',PCT,'market.preRetirementStdDev','Yearly volatility before retirement.'),
    row('Stock return','%',PCT,'market.stockMeanReturn','Average yearly stock return after retirement, before inflation and fees.'),
    row('Stock swing','%',PCT,'market.stockStdDev','Yearly stock volatility.'),
    row('Bond return','%',PCT,'market.bondMeanReturn','Average yearly bond return after retirement.'),
    row('Bond swing','%',PCT,'market.bondStdDev','Yearly bond volatility.')
  ]},
  {id:'allocation',label:'Stocks by portfolio size',rows:[
    ['Under 30× spending','stockUnder30x'],['30–35× spending','stock30xTo35x'],['35–40× spending','stock35xTo40x'],['40–45× spending','stock40xTo45x'],['45–50× spending','stock45xTo50x'],['50× or more','stock50xOrMore']
  ].map(([label,key])=>row(label,'%',PCT,`postRetirementAllocation.${key}`,'Stock share in retirement while the portfolio is this multiple of spending. The rest is bonds.'))},
  {id:'strategy',label:'Withdrawals',rows:[
    row('Withdrawal order','',ENUM,'withdrawalStrategy.withdrawalOrder','Which accounts ordinary spending draws from first.',{options:[['Standard','Pre-tax, Roth, taxable'],['TaxableFirst','Taxable, pre-tax, Roth'],['RothLast','Pre-tax, taxable, Roth']]}),
    row('Roth conversions','Y/N',YES_NO,'rothConversion.enabled','Convert pre-tax money to Roth each year up to a bracket.'),
    row('Conversion bracket','%',ENUM,'rothConversion.marginalRateCap','Fill taxable income up to this federal bracket.',{options:ROTH_CONVERSION_RATES.map(rate=>[rate,Math.round(rate*100)+'%']),active:s=>s.rothConversion.enabled}),
    row('Cash in drawdowns','Y/N',YES_NO,'withdrawalStrategy.useCashReserveDuringDrawdowns','Spend cash first in months when returns fall below the trigger.'),
    row('Drawdown trigger','%',PCT,'withdrawalStrategy.drawdownTrigger','Monthly portfolio return that switches to cash first, such as −1%.',{active:s=>s.withdrawalStrategy.useCashReserveDuringDrawdowns}),
    row('Early penalties','Y/N',YES_NO,'withdrawalStrategy.applyEarlyWithdrawalPenalty','Apply the 10% penalty before 59½.'),
    row('Rule of 55','Y/N',YES_NO,'withdrawalStrategy.ruleOf55Eligible','Your employer plan allows penalty-free withdrawals after separating at 55 or later.'),
    row('72(t) / SEPP','Y/N',YES_NO,'withdrawalStrategy.seppEligible','Start a new series of substantially equal payments at the forecast start.'),
    row('Spouse Rule of 55','Y/N',YES_NO,'spouseWithdrawal.ruleOf55Eligible','Spouse’s employer plan allows penalty-free withdrawals after separating at 55 or later.',{show:separate}),
    row('Spouse 72(t) / SEPP','Y/N',YES_NO,'spouseWithdrawal.seppEligible','Start a new series for your spouse at the forecast start.',{show:separate}),
    row('Full-run paths','paths',INT,'numberOfSimulations','Paths used by F9. Pro allows 4 to 10,000; free stays at 100.',{proOnly:true})
  ]}
];
export const EDITABLE_ROWS=SECTIONS.flatMap(section=>section.rows).filter(r=>r.type!=='calc');

const numberFormat=new Intl.NumberFormat('en-US',{maximumFractionDigits:2});
export const moneyText=value=>numberFormat.format(value);
const percentNumber=value=>Number((value*100).toPrecision(12));
export const pctText=value=>percentNumber(value)+'%';
function dateText(value){const parts=/^(\d{4})-(\d{2})-(\d{2})$/.exec(value||'');return parts?`${Number(parts[2])}/${Number(parts[3])}/${parts[1]}`:(value||'');}

// What the cell shows, and what the clipboard and CSV carry.
export function cellText(row,value){
  switch(row.type){
    case MONEY:return moneyText(value);
    case PCT:return pctText(value);
    case DATE:return dateText(value);
    case YES_NO:return value?'Yes':'No';
    case ENUM:return row.options.find(([v])=>v===value)?.[1]??String(value);
    default:return String(value??'');
  }
}
// What the editor opens with. Amounts lose their grouping so they are easy to change.
export function editText(row,value){return row.type===MONEY?String(value):cellText(row,value);}

// Arithmetic only: =2500*12 is worked out once and the answer is stored. Parsed
// by hand so nothing a person pastes is ever evaluated as code.
const MIXED_PERCENT='Use % on every number being added, or on none, such as =2%+0.5%.';
const DIVIDE_BY_PERCENT='Dividing by a percent isn’t supported in this cell. Try a form such as =10%/2.';
export function evaluateArithmetic(text,{percentCell=false}={}){
  // A comma only counts as thousands grouping, so 2,5 can never be read as 25.
  // No lookbehind here: Safari below 16.4 can't parse it, which would stop the whole page.
  const ungrouped=text.replace(/(^|[^\d.])(\d{1,3}(?:,\d{3})+)(?!\d)/g,(_,before,group)=>before+group.replace(/,/g,''));
  if(ungrouped.includes(','))return {error:'That arithmetic can’t be worked out.'};
  const source=ungrouped.replace(/^=/,'').replace(/[\s$]/g,'');
  if(/[a-jl-zA-JL-Z]/.test(source.replace(/[kKmM](?![a-zA-Z0-9])/g,'')))return {error:'Cell references aren’t supported. Use numbers, such as =2500*12.'};
  let at=0,percents=0,refusal='';
  const peek=()=>source[at];
  function atom(){
    if(peek()==='-'){at++;return -atom();}
    if(peek()==='+'){at++;return atom();}
    if(peek()==='('){at++;const value=sumExpression();if(peek()!==')')throw new Error('bracket');at++;return value;}
    const match=/^(\d+(?:\.\d*)?|\.\d+)([kKmM%]?)/.exec(source.slice(at));
    if(!match)throw new Error('number');
    at+=match[0].length;
    if(match[2]==='%')percents++;
    const scale={k:1e3,K:1e3,m:1e6,M:1e6,'%':.01}[match[2]]??1;
    return Number(match[1])*scale;
  }
  function product(){
    let value=atom();
    while(peek()==='*'||peek()==='/'){
      const op=source[at++],before=percents,next=atom();
      // 1/5% is 2000% by the math, but someone typing it into a percent cell likely means 0.2%.
      if(percentCell&&op==='/'&&percents>before){refusal=DIVIDE_BY_PERCENT;throw new Error('refused');}
      value=op==='*'?value*next:value/next;
    }
    return value;
  }
  function term(){const before=percents,value=product();return [value,percents>before];}
  function sumExpression(){
    let [value,percent]=term();
    while(peek()==='+'||peek()==='-'){
      const op=source[at++],[next,nextPercent]=term();
      // A percent cell holds percent points, so 0.1+2% could mean 2.1% or 12%. Don't guess.
      if(percentCell&&percent!==nextPercent){refusal=MIXED_PERCENT;throw new Error('refused');}
      value=op==='+'?value+next:value-next;
    }
    return value;
  }
  try{
    const value=sumExpression();
    if(at!==source.length||!Number.isFinite(value))return {error:'That arithmetic can’t be worked out.'};
    return {value:Number(value.toPrecision(15))};
  }catch{return {error:refusal||'That arithmetic can’t be worked out.'};}
}
// Without a leading =, only * and + count as arithmetic. A / or - there is far more
// likely to be a date or a pasted accounting value, and reading it as arithmetic would
// quietly store a different number.
const looksLikeArithmetic=text=>text.startsWith('=')||/\d\s*[*+]/.test(text);
// parseMoneyInput's grouping rules, without its dollar sign and k/M shorthand.
const plainNumber=text=>/^[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$/.test(text)?Number(text.replace(/,/g,'')):NaN;

function parseNumberText(text,{money=false,percentCell=false}={}){
  if(looksLikeArithmetic(text)){
    const out=evaluateArithmetic(text,{percentCell});
    if(out.error)return out;
    // A percent cell holds percent points. With a % sign the arithmetic worked in
    // fractions (5%*10% is 0.005), so turn that back into points. Without one, =2.5*2 is 5.
    return {value:percentCell&&text.includes('%')?Number((out.value*100).toPrecision(15)):out.value};
  }
  const value=money?parseMoneyInput(text):plainNumber(text);
  return Number.isFinite(value)?{value}:{error:money?'Enter an amount such as 40000, 40k or =3500*12.':'Enter a number.'};
}
function parseDateText(text){
  const value=text.trim();let year,month,day;
  const us=/^(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})$/.exec(value),iso=/^(\d{4})-(\d{1,2})-(\d{1,2})$/.exec(value);
  if(us){[,month,day,year]=us;if(year.length===2)year=(Number(year)>50?'19':'20')+year;}
  else if(iso)[,year,month,day]=iso;
  else return null;
  const date=`${year}-${String(month).padStart(2,'0')}-${String(day).padStart(2,'0')}`;
  return calendarDate(date)?date:null;
}
function dateProblem(row,value,s){
  const today=localCalendarDate(),h=s.household;
  if(row.path==='household.retirementDate'||row.path==='household.spouseRetirementDate'){
    const spouse=row.path.endsWith('spouseRetirementDate'),retired=spouse?h.spouseAlreadyRetired:h.alreadyRetired,birth=spouse?h.spouseBirthday:h.birthday;
    if(retired)return value>today||value<birth?'A past retirement date must be between the birthday and today.':null;
    return value<today?'A planned retirement date must be today or later.':null;
  }
  return value>=today?'A birthday must be before today.':null;
}

// Blank is a real answer here: it means Unknown, never zero, and blocks a run.
export function parseCell(row,text,s){
  const value=String(text??'').trim();
  if(row.path==='name'){
    if(!value||value.length>120)return {error:'Enter a plan name from 1 to 120 characters.'};
    return {value};
  }
  if(value===''||value==='?'){
    if(row.type===YES_NO||row.type===ENUM)return {error:'Choose one of the listed values.'};
    return {clear:true};
  }
  switch(row.type){
    case MONEY:{const out=parseNumberText(value,{money:true});if(out.error)return out;return Math.abs(out.value)>MAX_DOLLAR_AMOUNT?{error:'That amount is too large.'}:out;}
    case PCT:{
      const out=parseNumberText(looksLikeArithmetic(value)?value:value.replace(/\s*%$/,''),{percentCell:true});
      if(out.error)return {error:[MIXED_PERCENT,DIVIDE_BY_PERCENT].includes(out.error)?out.error:'Enter a percent such as 2.5.'};
      const share=Number((out.value/100).toPrecision(15));
      return Math.abs(share)>MAX_DOLLAR_AMOUNT?{error:'That number is too large.'}:{value:share};
    }
    case INT:{
      const out=parseNumberText(value);
      if(out.error)return {error:'Enter a whole number.'};
      if(Math.abs(out.value)>MAX_DOLLAR_AMOUNT)return {error:'That number is too large.'};
      return Number.isInteger(out.value)?out:{error:'Enter a whole number.'};
    }
    case DATE:{const date=parseDateText(value);if(!date)return {error:'Enter a date such as 10/1/2029.'};const problem=dateProblem(row,date,s);return problem?{error:problem}:{value:date};}
    case YES_NO:{const v=value.toLowerCase();if(['y','yes','true','1','x'].includes(v))return {value:true};if(['n','no','false','0'].includes(v))return {value:false};return {error:'Enter Yes or No.'};}
    case ENUM:{
      const v=value.toLowerCase(),hit=row.options.find(([option,label])=>label.toLowerCase()===v||String(option).toLowerCase()===v)||row.options.find(([,label])=>label.toLowerCase().startsWith(v));
      return hit?{value:hit[0]}:{error:'Choose: '+row.options.map(([,label])=>label).join(', ')+'.'};
    }
    default:return {error:'This cell is calculated.'};
  }
}

// Mirrors the classic planner's field commit so both views save the same plan.
export function settleRothHistory(s,sources,path){
  const prefix={'accounts.roth':'rothHistory','spouseAccounts.roth':'spouseRothHistory'}[path];
  if(!prefix||getPath(s,path)!==0)return;
  for(const key of ['contributionBasis','firstContributionYear']){
    const history=`${prefix}.${key}`;
    if(inputSource(s,sources,history)==='Unknown'){setPath(s,history,0);sources[s.id][history]='Entered';}
  }
}
function setRetiredStatus(ctx,s,path,retired){
  const spouse=path.endsWith('spouseAlreadyRetired'),h=s.household,flag=spouse?'spouseAlreadyRetired':'alreadyRetired',field=spouse?'spouseRetirementDate':'retirementDate',datePath='household.'+field,key=s.id+':'+field;
  if(h[flag]===retired)return;
  // Keep each status's date, so switching back restores it.
  const sources=ctx.sources[s.id],old=h[field],source=inputSource(s,ctx.sources,datePath),draft=ctx.drafts.get(key)||{};
  if(source!=='Unknown'&&calendarDate(old))draft[h[flag]?'retired':'future']={date:old,source};
  ctx.drafts.set(key,draft);
  h[flag]=retired;
  const next=draft[retired?'retired':'future'];
  h[field]=next?.date||'';sources[datePath]=next?.source||'Unknown';
  sources[path]='Entered';syncCalendarAges(s);
}
export function applyCell(ctx,s,row,value){
  ctx.exampleIds.delete(s.id);
  const sources=ctx.sources[s.id];
  if(row.path==='name'){s.name=value;return;}
  if(row.path==='household.alreadyRetired'||row.path==='household.spouseAlreadyRetired'){setRetiredStatus(ctx,s,row.path,value);return;}
  if(row.path==='numberOfSimulations'){value=Math.max(MIN_SIMULATION_PATHS,Math.min(MAX_SIMULATION_PATHS,value));s.simulationPathsCustomized=true;}
  setPath(s,row.path,value);sources[row.path]='Entered';
  settleRothHistory(s,ctx.sources,row.path);
  if(row.type===DATE)syncCalendarAges(s);
  if(row.path==='spending.annualBaseSpending')setAnnualBaseSpending(s,value);
}
export function clearCell(ctx,s,row){
  ctx.exampleIds.delete(s.id);
  const h=s.household;
  // A past retirement date is optional, so blank is a value rather than Unknown.
  if(row.path==='household.retirementDate'&&h.alreadyRetired||row.path==='household.spouseRetirementDate'&&h.spouseAlreadyRetired){setPath(s,row.path,'');ctx.sources[s.id][row.path]='Entered';syncCalendarAges(s);return;}
  ctx.sources[s.id][row.path]='Unknown';
}
export const isUnknown=(ctx,s,row)=>Boolean(row.path)&&row.path!=='name'&&inputSource(s,ctx.sources,row.path)==='Unknown';

// What counts as a simulation start. F9 always does. Where a live estimate already
// uses every path, as on the free tier, there is no full run left to press, so the
// first recalculation a visitor's own edit causes counts instead. Recalculating saved
// plans on page load never does: the visitor didn't ask for that.
export function countsAsSimulationStart({explicit,editPending,count,fullCount}){
  return Boolean(explicit)||(Boolean(editPending)&&count===fullCount);
}

// Unknown values that stop this plan from running. Same rules as the classic planner.
export function blockingUnknowns(s,sources){
  return unknownInputPaths(s,sources).filter(path=>{
    if(path==='household.retirementDate'&&s.household.alreadyRetired||path==='household.spouseRetirementDate'&&s.household.spouseAlreadyRetired)return false;
    if(path.startsWith('contributions.')&&s.household.alreadyRetired||path.startsWith('spouseContributions.')&&(s.household.separatePeople?s.household.spouseAlreadyRetired:s.household.alreadyRetired))return false;
    if(path==='healthcare.preMedicareMonthlyPremium')return needsPreMedicarePremium(s);
    if(path.startsWith('longTermCare.')&&path!=='longTermCare.enabled'&&!s.longTermCare.enabled)return false;
    if(path==='rothConversion.marginalRateCap'&&!s.rothConversion.enabled)return false;
    if(path==='withdrawalStrategy.drawdownTrigger'&&!s.withdrawalStrategy.useCashReserveDuringDrawdowns)return false;
    for(const [prefix,amount] of [['guaranteedIncome','annualIncome'],['spouseIncome','annualPension']])if(path.startsWith(prefix+'.')&&path!==prefix+'.'+amount&&(prefix==='guaranteedIncome'||path!=='spouseIncome.annualBenefitAt67')&&inputSource(s,sources,prefix+'.'+amount)!=='Unknown'&&s[prefix][amount]===0)return false;
    return true;
  });
}
const ROW_BY_PATH=new Map(EDITABLE_ROWS.map(r=>[r.path,r]));
export const rowForPath=path=>ROW_BY_PATH.get(path);

export function simulationScenario(s,paths){
  const copy=structuredClone(s);
  copy.numberOfSimulations=paths;copy.seed=DEFAULT_SEED;copy.household.asOfDate=localCalendarDate();
  return copy;
}
// Everything the Checks cell reports, most serious first.
export function planIssues(s,sources){
  const unknown=blockingUnknowns(s,sources).map(path=>({level:'unknown',path,text:`${rowForPath(path)?.label??path} is blank. Enter a value; 0 means none.`}));
  const errors=unknown.length?[]:validateScenario(simulationScenario(s,s.numberOfSimulations)).map(text=>({level:'error',text}));
  const gaps=budgetPlanGaps(s,sources).map(gap=>({level:'note',path:gap.path,text:`Your budget deducted ${gap.label}. Review the retirement amount, or the forecast may overstate readiness.`}));
  const notes=scenarioWarnings(s).map(text=>({level:'note',text}));
  return [...unknown,...errors,...gaps,...notes];
}

// Same dollar-basis switch as the classic results page.
export function shownResult(r,basis){
  if(!r)return null;
  const t=basis==='today'&&r.todayDollars;
  return t?{...r,medianEndingBalance:t.medianEndingBalance,pessimisticEndingBalance:t.pessimisticEndingBalance,optimisticEndingBalance:t.optimisticEndingBalance,balanceBands:t.balanceBands,meanPath:t.meanPath,dollarBasis:'today'}:{...r,dollarBasis:'future'};
}
export function readinessDelta(r,base){
  if(!r||!base||r===base||r.provenance.simulationCount!==base.provenance.simulationCount)return '';
  const count=r.provenance.simulationCount,diff=r.successProbability-base.successProbability;
  if(Math.abs(diff)<1e-9)return '=';
  const sign=diff>0?'+':'−';
  return sign+(count<=100?Math.round(Math.abs(diff)*count):(Math.abs(diff)*100).toFixed(1));
}

// A mini fan chart. Every plan shares `peak`, so columns can be compared by eye.
const fanScale=value=>Math.sqrt(Math.max(0,value));
export const fanPeak=results=>Math.max(1,...results.flatMap(r=>(r?.balanceBands??[]).map(b=>b.optimistic)));
export function fanPaths(bands,peak,width=100,height=30){
  if(!bands?.length)return null;
  const first=bands[0].age,span=Math.max(1e-9,bands.at(-1).age-first),top=fanScale(peak);
  const x=age=>((age-first)/span*width).toFixed(1),y=value=>(height-1-(fanScale(value)/top)*(height-2)).toFixed(1);
  const upper=bands.map(b=>`${x(b.age)},${y(b.optimistic)}`),lower=bands.map(b=>`${x(b.age)},${y(b.pessimistic)}`).reverse();
  return {band:[...upper,...lower].join(' '),median:bands.map(b=>`${x(b.age)},${y(b.median)}`).join(' ')};
}

// The Forecast sheet: one row per age, four columns per plan.
export function forecastTable(results){
  const perPlan=results.map(r=>{
    if(!r)return new Map();
    const funding=new Map(ageYearRows(r.notFailedByAge||[]).map(p=>[p.ageYear,p]));
    return new Map(balanceDisplayRows(r).filter(b=>!b.noOutcomes).map(b=>[b.ageYear,{
      funded:funding.get(b.ageYear)?.notFailedShare,count:r.provenance.simulationCount,
      p10:b.pessimistic,median:b.median,p90:b.optimistic
    }]));
  });
  const ages=[...new Set(perPlan.flatMap(m=>[...m.keys()]))].sort((a,b)=>a-b);
  return ages.map(age=>({age,plans:perPlan.map(m=>m.get(age)||null)}));
}
export const fundedLabel=(share,count)=>share===undefined?'':shareLabel(share,count).replace(' observed','');

// The Yearly sheet: flows are added up month by month between retirement
// anniversaries, and each month is deflated by its own price level.
export const YEARLY_COLUMNS=[
  ['spending','Spending'],['support','Work pay'],['socialSecurity','Soc Sec'],['pension','Pension'],['scheduled','RMD/SEPP'],['withdrawals','Withdrawn'],['tax','Tax'],['penalty','Penalty'],['converted','Converted'],
  ['pretax','Pre-tax'],['roth','Roth'],['employerRoth','Empl Roth'],['taxable','Taxable'],['cash','Cash'],['portfolio','Portfolio'],['home','Home'],['mortgage','Mortgage'],['netAssets','Net assets'],['unfunded','Unfunded']
];
export function yearlyTable(r,basis){
  const details=r?.steadySimulation?.monthlyDetails??[],indexes=r?.todayDollars?.steadyPriceIndexes;
  const real=basis==='today'&&indexes?.length===details.length;
  const fresh=()=>({spending:0,support:0,socialSecurity:0,pension:0,scheduled:0,withdrawals:0,tax:0,penalty:0,converted:0});
  let flows=fresh();const rows=[];
  details.forEach((p,i)=>{
    const div=real?indexes[i]:1,f=p.cashFlow;
    if(f){
      flows.spending+=f.expenses/div;flows.support+=(f.workingSupport||0)/div;flows.socialSecurity+=f.socialSecurity/div;flows.pension+=f.guaranteedIncome/div;
      flows.scheduled+=(f.seppDistribution+f.rmdDistribution)/div;flows.withdrawals+=f.additionalWithdrawal/div;flows.tax+=(f.incomeTax+f.conversionTax)/div;flows.penalty+=f.earlyPenalty/div;flows.converted+=f.conversionAmount/div;
    }
    if(p.month%12&&i!==details.length-1)return;
    const owner=p.ownerAccounts??[],each=key=>owner.length?owner.reduce((n,a)=>n+(a[key]||0),0):p[key]||0;
    const employer=owner.length?owner.reduce((n,a)=>n+(a.employerRoth?.reduce((m,x)=>m+x.balance,0)||0),0):p.employerRoth||0;
    const roth=owner.length?each('roth'):(p.rothIRA??p.roth??0);
    rows.push({month:p.month,age:p.age,date:p.date,...flows,
      pretax:each('pretax')/div,roth:roth/div,employerRoth:employer/div,taxable:p.taxable/div,cash:p.cash/div,portfolio:p.portfolio/div,home:p.home/div,mortgage:p.mortgage/div,netAssets:p.netAssets/div,unfunded:p.unfundedAmount/div});
    flows=fresh();
  });
  return rows;
}

// Clipboard and CSV helpers.
export const toTsv=rows=>rows.map(cells=>cells.map(v=>String(v??'').replace(/[\t\r\n]+/g,' ')).join('\t')).join('\n');
export function toCsv(rows){
  return rows.map(cells=>cells.map(v=>{
    let text=String(v??'');
    // Plan names are free text; never let one run as a formula in another spreadsheet.
    if(/^[=+\-@\t\r]/.test(text)&&!/^-?[\d.,]+%?$/.test(text))text="'"+text;
    return /[",\n\r]/.test(text)?`"${text.replace(/"/g,'""')}"`:text;
  }).join(',')).join('\r\n')+'\r\n';
}
export const parseTsv=text=>text.replace(/\r\n?/g,'\n').replace(/\n$/,'').split('\n').map(line=>line.split('\t'));
