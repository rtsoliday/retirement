// Independent monthly lognormal factors, calibrated to the entered arithmetic
// mean and standard deviation of the compounded twelve-month change.
// A positive factor avoids censoring negative inflation or investment draws.
export function monthlyRateDistribution(annualMean, annualStdDev) {
  const annualFactor = 1 + annualMean;
  const logVariance = Math.log1p((annualStdDev / annualFactor) ** 2) / 12;
  return {
    logMean: Math.log1p(annualMean) / 12 - logVariance / 2,
    logStdDev: Math.sqrt(logVariance)
  };
}

export function sampleMonthlyRate(distribution, rng) {
  // Consume the same Gaussian even at zero volatility so paired sensitivity
  // runs retain the original path's random draws in subsequent months.
  return Math.expm1(rng.normal(distribution.logMean, distribution.logStdDev));
}
