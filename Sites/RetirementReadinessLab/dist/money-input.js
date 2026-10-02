// Presentation only: stored assumptions keep their original monthly/yearly units.
export function moneyInputValue(value) {
  return value === '' ? '' : new Intl.NumberFormat('en-US', {maximumFractionDigits:10}).format(value);
}

export function parseMoneyInput(text) {
  let value=String(text).trim().replace(/^(-?)\$\s*/, '$1');
  // Shorthand such as 40k or $1.2M scales a plain amount; it never follows an exponent.
  const suffix=/^(.*\d)\s*([kKmM])$/.exec(value),scale=suffix?(/k/i.test(suffix[2])?1e3:1e6):1;
  if(suffix)value=suffix[1];
  // Accept ordinary US amounts, including pasted dollar signs and grouping.
  // Reject malformed grouping, partial exponents and stray characters.
  const plain=/^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?$/.test(value)||/^[-+]?\.\d+$/.test(value);
  if(!plain&&(suffix||!/^[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+$/.test(value)))return NaN;
  const amount=Number(value.replaceAll(',', ''));
  return scale===1?amount:Number((amount*scale).toPrecision(15));
}
