# Simulation Model

## Current Android Engine

The current Android engine is a monthly cashflow Monte Carlo model with annual result bands for charts and reports. It provides real scenario feedback for the initial UI and will be expanded toward the existing Python model.

Current behavior:

- Monthly pre-retirement growth.
- Monthly retirement withdrawals and cashflows.
- Stock and bond returns sampled from normal distributions.
- Inflation sampled from normal distribution with a floor.
- Federal tax estimate using 2024 brackets.
- Social Security claiming adjustment.
- Taxable Social Security estimate.
- SSA Trustees Alt2 2025 annual death probabilities selected by gender, with an internal projection cap at age 119.
- Medicare Parts B/D premium estimate after age 65, including modeled 2024 IRMAA tiers indexed by the Medicare inflation multiplier.
- Pre-Medicare healthcare estimate.
- Optional long-term care shock that replaces normal spending while active.
- Optional Roth conversion up to a selected marginal bracket.
- Optional cash reserve use during market drawdowns.
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
