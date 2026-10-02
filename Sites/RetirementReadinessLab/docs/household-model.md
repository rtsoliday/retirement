# Contributions and separate-person households

Implemented locally following the user's decisions: do not publish yet; model future savings contributions; support separate couple timelines, benefits, pensions and retirement-account ownership; describe the balance-history helper as planned later. The preceding [UX-only report](first-time-ux.md) records the original reproduction, metric definitions, inflation verification and before/after evidence. This follow-up intentionally adds calculation capabilities; it does not treat the original usability reports as calculation bugs.

## What changed

- Annual employee pre-tax, employer pre-tax, Roth IRA, shared taxable and shared cash deposits, with an optional annual increase. Annual amounts are divided into monthly deposits after growth. Deposits increase balances once; Roth deposits also increase contribution basis and start an unfunded IRA's clock. They are funded outside the model. There is no unused salary field.
- New browser plans use separate people. Each has a calendar retirement date, own Social Security estimate, one pension, owned pre-tax and Roth balances, Roth history and early-access declarations. Shared spending starts at the first retirement; savings stop at each owner's retirement or death. Retirement-period household return assumptions apply to both owners after that first date.
- Optional take-home household support covers costs during the overlap when one person still works. Enter it after taxes and savings contributions. Its zero default means savings fund the gap; it is not taxed or deposited again.
- Separate owner ages govern RMDs, penalties, Roth qualification and declared Rule of 55 timing. Each selected SEPP starts at that owner's retirement and protects only their balance. Both own Social Security benefits are considered, with an eligible excess spousal supplement and a survivor maximum. Pensions have independent start months, growth and survivor shares.
- Setup groups You, Spouse and shared Household balances explicitly. Contribution labels identify the person even outside their group. Other deposit types and annual increases are collapsed in guided setup; all remain available in the detailed editor. Important new fields have annual/monthly examples and statement guidance. The app records each value's source automatically (sample, entered, estimated by a helper, or Unknown when cleared); there is no source picker. The review step shows the recorded source; reports flag only Unknown values. Review, reports, copies and JSON backups include the new assumptions. Steady monthly balances show each owner's pre-tax and Roth columns.
- A one-year statement check (`dist/growth-helper.js`) in the accounts step separates new savings, other money moved in, withdrawals and investment growth for one account. It can copy employee and employer contributions into that owner's yearly savings, marked Estimated. Its one-year return is shown for information only and is never applied. Its values are not saved. A multi-year balance history and statement import remain planned; no bank linking was added.

## Saved plans and calculations

`household.separatePeople` defaults to false when normalizing older plans. Missing contribution sections default to zero; missing new values are labeled Sample/default rather than attributed to the user. Old plans keep pooled ownership, their shared retirement date, one pension and derived spouse benefits until the user selects **Use separate-person inputs**. Merely opening guided setup does not switch them, including backups with `hasStartedPlan: false`.

Switching preserves all account dollars initially under You and starts spouse balances at zero. The screen instructs users to reallocate combined statements without counting them twice. Four important new spouse amounts become Unknown and must be reviewed before running. New inputs and source metadata retain the existing local-storage key and JSON format. Nothing in this work sends financial inputs to a server or changes authentication, billing or privacy behavior.

The legacy calculation path remains in `engine.js`; direct contributions enter its existing pre-retirement loop only when nonzero. The expanded path uses separate owner ledgers and the existing rate, mortality, tax, mortgage, allocation and aggregation functions. Result provenance distinguishes the original engine, direct-contribution engine and separate-person engine. No seed or return/inflation distribution was changed.

## Assumptions and remaining scope

This is a retirement model with externally funded deposits, not a full working-life cash-flow model. Before first retirement it does not simulate living expenses, employment taxes, annual income tax or RMDs. Afterward the net support input does not calculate gross wages, payroll tax, Social Security earnings tests or employment effects on federal brackets and Medicare. A full salary/budget model remains a separate possible phase.

Contribution limits, eligibility, non-Roth after-tax basis, employer Roth distributions, employer-plan RMD deferrals, separate owner investment allocations, plan-specific exceptions and multiple pensions per person remain unsupported. A pension survivor share applies only when the owner reached its start age before death. Independent survivor-claim strategies, disability and divorced-spouse Social Security rules are not modeled. Spouse inherited accounts become their own after the modeled death year; histories merge using the earliest Roth funding year. This is a simplified spouse-inheritance assumption, not a beneficiary election planner. Modeled tax years still follow retirement anniversaries rather than calendar years.

The [visible methodology](../dist/methodology.html#household-model) states these boundaries and links the [SSA own/spouse benefit explanation](https://www.ssa.gov/faqs/en/questions/KA-02011.html), [SSA claiming rules](https://www.ssa.gov/benefits/retirement/planner/claiming.html) and [IRS Publication 590-B](https://www.irs.gov/publications/p590b). The steady illustration uses the same deposits and ownership inputs with fixed lifespans, zero volatility and no long-term care risk; it remains separate from Monte Carlo funding and lifespan counts.

No genuine legacy calculation bug was established or corrected in this follow-up. New modeling can deliberately change results when enabled. Publishing is still deferred. The balance-history helper and a complete salary model need future scope decisions, not approval to complete this local implementation.

## Validation

`npm test --prefix Sites/RetirementReadinessLab`: **423 passed, zero failed**. The complete result objects for four pre-change, fixed-date, seeded scenarios are SHA-256 checked after normalization, excluding only generation time. These include individual, couple/pension, depleted and zero-income cases. Existing return/inflation, survivor, zero-survivor display, saved-plan, monthly/annual input and depleted-savings regressions still pass.

Added 21 calculation/compatibility regressions covering annual-to-monthly deposits, employer additions, growth timing, Roth basis/clocks, legacy shared stopping dates, either spouse retiring first, stopping at retirement/death, support without double taxation/counting, same-month funding, own/spousal/survivor benefits, independent pensions, RMD ownership, SEPP ownership, Roth inheritance, conversions and actual tax liability, staggered healthcare, invalid inputs, Unknown versus zero, steady assumptions and finite seeded care/penalty/conversion combinations. Three added app tests cover explicit migration, active/hidden ownership inputs, unknown blocking, copies and JSON round trips.

Browser checks at 1280×900, 390×844 and 320×740 exercised the saved-plan switch, independent dates, annual contributions, pensions, support, editable review and worker-run results. Individual mode hides spouse inputs; returning to couple preserves them. A source marked Unknown disables the review Run button; entering zero resolves it. Keyboard Tab reaches filing status and the next spouse contribution field. Mobile pages measured 375/375 and 305/305 pixels for content/viewport widths, with no page-wide overflow. Monthly result tables retain a keyboard-focusable horizontal scrolling region. No browser warnings or script errors were observed.

`npm run build:sites --prefix Sites/RetirementReadinessLab` completed. All **22** release modules match tracked source bytes and their relative imports resolve in the immutable release directory. Release asset ID: `eb4325f89aa5f7033516`. This is a local package, not a deployment. Existing staged work was preserved; no commit, push or publication was performed.

## Changed files

- Calculations/schema: `dist/model.js`, `dist/engine.js`; new `dist/savings.js`, `dist/person-accounts.js`, `dist/person-income.js`, `dist/person-engine.js`.
- User interface and explanations: `dist/app.js`, `dist/ux-guidance.js`, `dist/withdrawals-view.js`, `dist/methodology.html`, `dist/index.html`.
- Tests: `tests/app-state.test.js`; new `tests/household.test.js`, `tests/fixtures/pre-household-results.json`.
- Documentation: `README.md`, `docs/first-time-ux.md`, this report and `docs/ux-evidence/household-*.jpg`.

## Screenshots

Original live-site reproduction and UX-only before/after comparisons are in the [first-pass report](first-time-ux.md). The follow-up uses a synthetic example: combined pre-tax savings split into You $300,000 and Spouse $200,000; You $12,000 employee plus $6,000 employer yearly savings; Spouse $6,000 yearly savings, $18,000 annual own Social Security and $12,000 pension; retirement dates October 2033 and October 2035, with $36,000 annual spouse take-home support during the overlap. Other retained sample values remain visibly labeled. The new example's results are not a before/after calculation comparison.

| Flow | Before | After |
| --- | --- | --- |
| Household dates, desktop | [Original live setup](ux-evidence/before-desktop-household.jpg) | [Separate dates](ux-evidence/household-desktop-dates.jpg) |
| Future deposits, desktop | [UX-only accounts/setup](ux-evidence/after-desktop-household.jpg) | [Contribution inputs](ux-evidence/household-desktop-contributions.jpg) |
| Editable summary | [UX-only review](ux-evidence/after-mobile-review.jpg) | [Expanded desktop review](ux-evidence/household-desktop-review.jpg) |
| Owned accounts in illustration | [UX-only steady explanation](ux-evidence/after-desktop-steady.jpg) | [Separate owner columns](ux-evidence/household-desktop-steady.jpg) |
| Spouse on mobile | [Original accounts](ux-evidence/before-mobile-accounts.jpg) | [Dates](ux-evidence/household-mobile-dates.jpg), [contributions](ux-evidence/household-mobile-contributions.jpg), [pension at 320 pixels](ux-evidence/household-mobile-pension-320.jpg) |
