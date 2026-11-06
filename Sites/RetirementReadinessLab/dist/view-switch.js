// The planner and the sheet are two views of one set of saved plans. Switching
// keeps your place. The plan travels in sessionStorage, so nothing about a plan
// enters the URL. index.html reads VIEW_KEY and LANDING_KEY in its inline script.
export const VIEW_KEY='retirement-preferred-view';
export const LANDING_KEY='retirement-landing-referrer';
const FOCUS_KEY='retirement-focus-plan';

export function rememberView(view){try{localStorage.setItem(VIEW_KEY,view);}catch{}}
export function handOffPlan(id){try{sessionStorage.setItem(FOCUS_KEY,String(id));}catch{}}
export function takeHandedOffPlan(){
  try{const id=sessionStorage.getItem(FOCUS_KEY);sessionStorage.removeItem(FOCUS_KEY);return id;}
  catch{return null;}
}
// Landing on the sheet from the home page should not hide where the visit came from.
export function arrivalReferrer(fallback){
  try{
    const landed=sessionStorage.getItem(LANDING_KEY);
    if(landed===null)return fallback;
    sessionStorage.removeItem(LANDING_KEY);return landed;
  }catch{return fallback;}
}

// Planner page → matching place in the sheet. setupSection 5 is the review step,
// which has no rows of its own.
export function sheetHref(view,setupSection=0){
  if(view==='results')return './sheet.html#forecast';
  if(view==='withdrawals')return './sheet.html#yearly';
  if(view==='setup'&&setupSection>=0&&setupSection<5)return `./sheet.html#step-${setupSection+1}`;
  return './sheet.html';
}
export function parseSheetHash(hash){
  if(hash==='#forecast'||hash==='#yearly')return {tab:hash.slice(1)};
  const step=/^#step-([1-5])$/.exec(hash);
  return step?{step:Number(step[1])-1}:{};
}
// Sheet place → matching planner page. Every target has a hash, so the home page's
// landing rule can never send a visitor who asked for the planner back to the sheet.
export function plannerHref(tab,step=null){
  if(tab==='forecast')return './#/results';
  if(tab==='yearly')return './#/withdrawals';
  return step===null?'./#/':`./#/assumptions/${step+1}`;
}
