// Presentation only: stored assumptions keep their original monthly/yearly units.
export function moneyInputValue(value) {
  return value === '' ? '' : new Intl.NumberFormat('en-US', {maximumFractionDigits:10}).format(value);
}

export function parseMoneyInput(text) {
  const value=String(text).trim().replace(/^(-?)\$\s*/, '$1');
  // Accept ordinary US amounts, including pasted dollar signs and grouping.
  // Reject malformed grouping, partial exponents and stray characters.
  if(!/^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?$/.test(value)&&!/^[-+]?\.\d+$/.test(value)&&!/^[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+$/.test(value))return NaN;
  return Number(value.replaceAll(',', ''));
}
