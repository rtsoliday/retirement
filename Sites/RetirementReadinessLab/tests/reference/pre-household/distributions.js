// IRS Publication 590-B, Appendix B, Table III (Uniform Lifetime).
// Account ownership and modeled-year timing assumptions are disclosed in the
// planner. The special table for a sole-beneficiary spouse >10 years younger
// cannot be selected without beneficiary information.
const uniformLifetime = [27.4,26.5,25.5,24.6,23.7,22.9,22,21.1,20.2,19.4,18.5,17.7,16.8,16,15.2,14.4,13.7,12.9,12.2,11.5,10.8,10.1,9.5,8.9,8.4,7.8,7.3,6.8,6.4,6,5.6,5.2,4.9,4.6,4.3,4.1,3.9,3.7,3.5,3.4,3.3,3.1,3,2.9,2.8,2.7,2.5,2.3,2];

export function rmdStartAge(birthYear) {
  if (birthYear >= 1960) return 75;
  if (birthYear >= 1951) return 73;
  // Only already-eligible older cohorts can enter this 2026-based model.
  return birthYear >= 1949 ? 72 : 70.5;
}

export function requiredMinimumDistribution(priorYearBalance, age, birthYear) {
  if (age < rmdStartAge(birthYear) || priorYearBalance <= 0) return 0;
  const denominator = uniformLifetime[Math.min(120, Math.max(72, Math.floor(age))) - 72];
  return priorYearBalance / denominator;
}
