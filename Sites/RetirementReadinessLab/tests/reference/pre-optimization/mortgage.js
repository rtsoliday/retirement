// Infer a fixed monthly interest rate from the entered balance, payment and
// remaining term. A zero balance retains the legacy payment-only schedule.
export function mortgageAtRetirement(mortgage, elapsedMonths) {
  const totalMonths = mortgage.yearsLeft * 12 + mortgage.monthsLeft;
  const payment = mortgage.monthlyPayment;
  let balance = mortgage.currentBalance, rate = 0;
  if (balance > 0 && totalMonths > 0 && payment > balance / totalMonths) {
    let low = 0, high = payment / balance;
    for (let i = 0; i < 64; i++) {
      const mid = (low + high) / 2;
      const required = balance * mid / -Math.expm1(-totalMonths * Math.log1p(mid));
      if (required > payment) high = mid; else low = mid;
    }
    rate = (low + high) / 2;
  }
  for (let m = 0; m < Math.min(elapsedMonths, totalMonths); m++) balance = payMortgage(balance, payment, rate);
  return { balance, rate, months: Math.max(0, totalMonths - elapsedMonths) };
}

export function payMortgage(balance, payment, rate) {
  const remaining = Math.max(0, balance * (1 + rate) - payment);
  return remaining < .000001 ? 0 : remaining;
}
