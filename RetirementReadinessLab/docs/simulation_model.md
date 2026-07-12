# Simulation Model

## Current Android Engine

The current Android engine is a monthly cashflow Monte Carlo model with annual result bands for charts and reports. It provides real scenario feedback for the initial UI and will be expanded toward the existing Python model.

Current behavior:

- Monthly pre-retirement growth.
- Monthly retirement withdrawals and cashflows.
- Home-sale spending removes property tax and homeowners insurance only when the saved budget estimate was applied to annual base spending.
- Stock and bond returns sampled from normal distributions.
- Post-retirement stock/bond allocation selected from a tunable invested-assets-to-annual-spending ladder.
- Inflation sampled from normal distribution with a floor.
- Federal tax estimate using 2026 brackets, base and age-65 standard deductions, and the income-phased 2025-2028 enhanced senior deduction; permanent amounts are indexed with modeled general inflation.
- Optional 10% early withdrawal tax on modeled pre-tax withdrawals before age 59 1/2.
- Optional Rule of 55 assumption for qualifying employer-plan pre-tax withdrawals after retiring at 55 or later.
- Optional 72(t) / SEPP fixed amortization schedule from modeled pre-tax assets, using IRS single-life expectancy factors and a 5% interest assumption.
- Social Security claiming adjustment, including survivor delayed-retirement credits and the widow(er)'s benefit limit after an early worker claim.
- Annual Social Security COLA approximation with no nominal benefit reduction in deflation years.
- Taxable Social Security estimate.
- SSA Trustees Alt2 2025 annual death probabilities selected by gender, with an internal projection cap at age 119.
- Separate primary and spouse mortality-table selections for married households.
- Medicare Parts B/D premium estimate after age 65, with modeled 2026 IRMAA income tiers indexed separately from healthcare-cost inflation.
- Pre-Medicare healthcare estimate.
- Optional long-term care shock, indexed with healthcare inflation, that replaces normal spending while active.
- Optional annual Roth conversion that fills the unused room in a selected marginal bracket using modeled taxable income to date.
- Optional cash reserve use during market drawdowns.
- Paired Lab comparisons that reuse random paths so changes are not obscured by unrelated Monte Carlo noise.
- Success probability, ending balance percentiles, and failure age summary.
- Plain-language result interpretation for dashboard, detail view, and reports.

## Required Future Parity Work

To reach product-grade parity with the Python simulator, expand the Android engine to include:

- Correlated stock, bond, and inflation returns.
- Annual refresh process for mortality tables, tax brackets, Medicare premiums, and IRMAA tiers.
- More exact Roth conversion tax treatment.
- Cash reserve refill rules.
- Detailed path collection.
- Risk attribution based on scenario perturbation.
- Broader deterministic seed controls for report reproduction.

## Result Contract

Each simulation run should produce:

- Scenario ID and run timestamp.
- Calculation provenance: engine version, engine cadence, tax table version, mortality model version, random seed, simulation count, and assumption fingerprint.
- Success probability.
- Median ending balance.
- 10th percentile ending balance.
- 90th percentile ending balance.
- Median failure age when failures occur.
- Balance percentile path.
- Risk breakdown.
- Plain-language recommended next test.

## Validation Requirements

Hard validation:

- Current age must be positive.
- Retirement age must be greater than or equal to current age.
- The internal projection cap must be greater than retirement age.
- Social Security claim age must be 62-70.
- Balances and spending must be non-negative.
- Percentages must be in plausible bounded ranges.

Soft warnings:

- Very high return assumptions.
- Missing healthcare costs before 65.
- Long-term care disabled.
- Very low simulation count.
- Very early retirement with a long potential drawdown period.
