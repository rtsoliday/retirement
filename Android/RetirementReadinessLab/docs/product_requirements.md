# Product Requirements

## Product Name

Retirement Readiness Lab

## Goal

Help users answer:

- Can I retire at my target age?
- What risks threaten the plan?
- Which decision improves the plan most?

## MVP User Story

As a late-career household planner, I want to enter my savings, spending, Social Security, healthcare, and retirement age assumptions so I can compare realistic retirement scenarios without building a spreadsheet.

## Core Workflows

1. Complete guided setup.
2. View readiness dashboard.
3. Adjust the active plan.
4. Run Lab stress tests.
5. Review assumptions.
6. Export a summary in Pro.

Deferred after first release:

- Visible scenario management and scenario comparison.

## MVP Functional Requirements

- Create one active plan locally.
- Keep scenario management hidden for the first release.
- Run Monte Carlo simulation on-device.
- Show success probability.
- Show median, pessimistic, and optimistic ending balances.
- Show failure age estimate when failures occur.
- Run quick Lab tests for retirement age, spending, Social Security, Roth conversion, healthcare, and long-term care in Pro.
- Show all assumptions behind the result.
- Avoid bank linking and cloud sync.

## First-Release Packaging

- Free download with a one-time Pro unlock.
- Free runs are capped at 500 simulations.
- Pro unlocks Scenario Lab, advanced setup controls, higher simulation counts, and PDF/text report sharing.
- Scenario management and comparison stay hidden for both Free and Pro in the first release.
- Cash reserve drawdown strategy controls stay hidden for both Free and Pro in the first release.
- JSON backup/import stays hidden for both Free and Pro in the first release.
- Other currently hidden feature flags should remain hidden for the first release.

## Non-Functional Requirements

- First useful result in under five minutes.
- Simulation should not block the UI in release builds.
- App must support light and dark mode.
- App must support dynamic font scaling.
- User financial data must remain local unless explicitly exported.
- Results must be framed as educational estimates, not financial advice.

## MVP Release Criteria

- Dashboard, Setup, Scenarios, Lab, Assumptions, and Reports screens exist.
- Scenario data persists locally.
- Simulation engine has unit coverage for tax, Social Security, healthcare, LTC, and withdrawals.
- App has a clear privacy policy and educational disclaimer.
- The app can be tested on a small phone, large phone, and tablet viewport.
