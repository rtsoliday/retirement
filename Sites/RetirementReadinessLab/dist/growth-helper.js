// One-year statement check: separates money added to an account from its
// investment growth. Cash flows are assumed to be spread evenly through the
// year (a Modified Dietz estimate), so the return is approximate.
export const GROWTH_HELPER_FIELDS=['start','end','yours','employer','transfersIn','out'];

// Every shown amount must be entered (0 for none): a missing cash flow stays
// unknown rather than becoming zero.
export function oneYearGrowth(values,{employer=true}={}){
  const keys=GROWTH_HELPER_FIELDS.filter(key=>employer||key!=='employer');
  if(keys.some(key=>values[key]===''||values[key]===undefined||values[key]===null))return {complete:false};
  const n=Object.fromEntries(keys.map(key=>[key,Number(values[key])]));
  if(keys.some(key=>!Number.isFinite(n[key])||n[key]<0))return {complete:false,error:'Enter amounts of $0 or more.'};
  const savings=n.yours+(employer?n.employer:0),net=savings+n.transfersIn-n.out,growth=n.end-n.start-net,base=n.start+net/2;
  return {complete:true,savings,employerSavings:employer?n.employer:0,yourSavings:n.yours,net,growth,rate:base>0?growth/base:null};
}
