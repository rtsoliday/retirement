> This records the initial UX-only pass. The user subsequently approved direct savings contributions and separate-person household modeling; see [the follow-up implementation](household-model.md). Publication remains deferred.

# First-time UX implementation and evidence

Validated October 1, 2026. Changes are local and packaged; they have not been deployed to retirementforecast.us. No simulation mathematics, account-access service or network/privacy behavior changed.

## Reproduction before editing

Read the repository and Sites READMEs, then inspected the live site at 1280 × 900 and 390 × 844 using sample inputs. Followed Build my forecast, household selection, account/spending setup, income setup and the completed ten-path preview on both surfaces. Mobile uses the section selector instead of the desktop section sidebar.

The spouse inputs already existed but depended on discovering Married under Filing status. Pension income existed under Guaranteed annual income, while Pre-tax savings did not give familiar non-Roth account examples. Results placed financial coverage, household lifespan, observed balance ranges and a separate steady run together without an immediately visible explanation connecting their different populations and assumptions.

The tester’s exact 0% / approximately $8 million at age 95 cannot be reproduced without their assumptions and run. The ambiguous interpretation was reproducible. A local couple example with an $18,000/year pension also has zero survivors at its final observation while every path ends without a financial shortfall. Synthetic regressions reproduce zero survivors alongside a positive earlier balance and a continuing steady illustration. This is presentation evidence, not evidence of incorrect calculation.

## Supported, confusing, or requiring new calculations

| Report | Existing support / confusing presentation | Implemented UX | Separate modeling capability |
| --- | --- | --- | --- |
| Spouse hard to enter | Married reveals birthday, mortality table, claim age and survivor share | Obvious Individual/Couple controls; You, Spouse and Household groups | Separate retirement dates and independently owned accounts |
| Pension hard to find | One guaranteed income stream with start age, growth and survivor share | Your pension or annuity/year; explicitly income, not a balance | Separate spouse pensions, multiple streams, plan-specific terms |
| Non-Roth plans hard to identify | Pooled pre-tax balance | Pre-tax/non-Roth retirement savings; traditional 401(k), 403(b), IRA examples | After-tax non-Roth basis and separate employer-plan rules |
| Expected both salaries | Current balances grow; future contributions and salary are absent | Visible earnings → income → withdrawals explanation; no inactive salary fields | Salary, employee/employer contributions and working-period cash flows |
| Which numbers? | Defaults and explanation popovers; no source record | Visible statement sources, monthly/annual examples and explicit provenance | Optional future manual balance-history assistance |
| 0% vs large balance | Funding share, lifespan share, endpoint balances and steady run answer different questions | Counted ten-path preview, direct explanation, future-dollar labels, missing-outcome treatment and actual steady assumptions | No new modeling required for these explanations |

## Exact meanings of displayed results

Sources: `dist/engine.js` (`buildFundingSurvival`, `runSimulation`, `runOne`, `runSteadySimulation`, `riskBreakdown`), `dist/result-format.js`, `dist/charts.js`, `dist/app.js`. The public [methodology](../dist/methodology.html#result-definitions) contains matching definitions.

| Value | Numerator / population / denominator |
| --- | --- |
| Lifetime coverage/readiness | Paths without any portfolio shortfall / all completed paths. Coverage ends at death or the observation limit, not necessarily age 95. |
| No shortfall observed by age | Paths without failure at or before that monthly age / all paths. A successfully deceased path remains financially covered. |
| Household still alive by age | Paths with at least one person alive / all paths, including paths with financial shortfalls. Censored paths alive at the modeling limit remain alive. A couple’s axis uses the primary person’s age. |
| Median ending balance | All financial account totals at each path’s death, modeling limit or shortfall; failed endings count as zero. Home value is excluded. Median averages the middle pair for even counts. |
| Ending 10th/90th percentiles | Sorted endpoint balances at rounded index `(n − 1) × .1` or `.9`. These percentile ranks are not survival probabilities. |
| Balance bands and their path count | Actual observed balances at that snapshot, including zero at failure; no extension after failure/death. The annual table chooses the last balance snapshot in each whole-year age. If its final lifespan observation has no survivors, an earlier same-year balance is withheld, count is zero and amounts read Not enough simulated outcomes. |
| Scatter dots / point count | Positive annual observations; one lifetime supplies several dots. Display is a deterministic sample of at most 30,000 dots. |
| Scatter mean | All positive observed balances at that annual offset; excludes zeros and missing observations. It differs from the median and average ending balance. |
| Failure age / buckets | Median among actual failures; each five-year bucket count / all failures. No failures means no observed shortfalls, not impossibility. |
| Sensitivity counts | Change in shortfalls among up to 128 paired paths; original shortfalls and paired sample size are disclosed. Comparisons overlap and are not causal probabilities. |
| Scenario comparisons and target percentages | Same lifetime-coverage metric within the completed comparison sample. Target percentages are requested screening thresholds; target outputs are search results under those assumptions, not guarantees. |
| Steady monthly account balances | One extra run. Opening balances after pre-retirement growth and mortgage payments, then monthly closing balances. Portfolio sums four financial accounts; net assets adds home value and subtracts mortgage debt. A final unfunded amount is an unmet cost. |

Ten-path shares use counts. A zero share reads 0 of N observed rather than a claim of impossibility. Larger nonzero shares that would round to 0.0% or 100.0% instead read <0.1% or >99.9%. The results explanation and chart captions distinguish death from running out of savings. Financial amounts absent because there are no observed/surviving outcomes read Not enough simulated outcomes; a living depleted path’s actual zero is retained.

## Inflation and the separate illustration

Verified the current implementation before labeling money. Pre-retirement living/housing costs compound the monthly equivalent of the entered general mean; healthcare uses its separate mean. Pension income compounds its own entered increase from today, including before its payments begin. Retirement paths compound sampled general and healthcare factors separately; pension increases remain their own rate. Social Security’s COLA logic retains its cumulative price high-water mark. Existing tests cover compounded arithmetic means/volatilities, independent cash flows and deflation/COLA behavior.

Result dollars are future dollars. Inputs that represent balances/costs/benefit amounts are labeled as amounts today or today’s purchasing power. A future quoted pension payment needs adjustment before entry under the existing pension-growth convention. No purchasing-power conversion was added: converting an aggregate median with one average inflation rate would not represent the median of balances deflated by their own paths’ histories.

The steady illustration fixes each lifespan at 95 if current completed monthly age is at most 85; otherwise at current age plus ten years. It follows the final household lifespan even if the selected maximum modeling age is lower, and stops early on a shortfall. Pre-retirement, stock, bond, general-inflation and healthcare-inflation volatility are all zero; long-term-care risk is disabled. Entered average rates remain, allocation still follows the selected rules and cash earns 2%. Taxes, benefits, healthcare premiums, spending changes, conversions, withdrawals and home-sale rules remain selected. The actual completed-run rates and lifespans are visible directly beside the monthly table, along with the statement that this is not a median, typical Monte Carlo outcome or guaranteed balance.

## Guided setup, persistence and privacy

Five input steps plus editable review provide progress, previous/next navigation and an always available detailed-editor exit. Longer settings remain accessible in disclosures. Guided Run opens review before calculation. Important inputs show where to find the value, what it measures, who it belongs to and the time unit.

Provenance is stored in a top-level `inputSources` map alongside unchanged scenario objects: Entered, Estimated, Unknown, Sample/default, or Saved value; source not recorded. It survives reload, copies and web JSON export/import. Legacy sources are not guessed from a balance matching a sample. Dates inferred during existing age-only migration are marked Estimated. Reset restores Sample/default. Applied budget estimates mark spending and the associated home-cost snapshot Estimated.

Clearing a numeric/date assumption marks it Unknown while retaining its previous numeric value for editing. Unknown blocks simulations, comparisons and target searches; it never silently becomes zero. Explicit 0 is valid. Unknown spouse-only inputs do not block an individual scenario. Malformed input, invalid backup protection, stale-tab saving, focus preservation and existing migration behavior retain regression coverage. No source notes or scenario data are sent to a server. No bank linking or new service is added.

## Proposed phase: working household and separate spouse modeling

This phase needs a product decision before implementation. Recommended scope:

1. Model each person’s employment end date and their own salary/contributions. Decide between directly entered annual contributions and a full salary/take-home/spending model; salary alone is insufficient.
2. Define employee savings, employer match, contribution timing, escalation, account destination and limits. Avoid counting both a salary-derived contribution and an independently entered contribution.
3. Resolve working spouse income after the first retirement, taxes/health coverage, household spending and independent own-record Social Security benefits.
4. Add pension streams per person and account ownership/basis where required. Define survivor transitions, employer-plan exceptions and separate Roth histories.
5. Introduce explicit schema migration that reproduces current results when all new cash flows are zero and the shared-date assumptions are retained. Add independent monthly cash-flow and couple-transition tests.

No salary, spouse contribution, separate benefit or account-ownership calculation was added in this pass. No genuine calculation defect was identified for correction from the tester report.

## Separate proposal: historical-balance helper

Start with manual dated account balances and optional statement import, without bank linking. Require contributions, withdrawals, transfers, fees and their dates to be identified. Internal transfers should reconcile between accounts and cancel at household level. Missing cash flows must remain unknown; dividends/reinvestments and valuation changes must not be confused with new savings.

Display net deposits and residual change separately. Only estimate an investment return when data supports a stated method: time-weighted return needs valuations around cash flows; a money-weighted return needs dated flows and may have ambiguous/no solutions. Show method, completeness and uncertainty. Never copy total balance growth automatically into the model’s investment-return assumption. Applying any suggested rate must be an explicit editable choice. This helper remains a proposal, with no implementation or bank linking.

## Validation

- Before changes: all 388 existing Sites tests passed.
- After changes: all 399 Sites tests passed, with no failures/skips. New coverage includes individual/couple UI, pension and account examples, source persistence, missing versus zero, review-before-run, monthly versus annual units, zero survivors with positive earlier balances, living depleted paths, steady explanations and completed-run assumptions.
- Complete ten-path engine results for four fixed-date scenarios (individual, couple with pension and pre-tax accounts, depleted savings, zero income) are byte-identical before/after, excluding generation timestamp. Source hashes for `engine.js`, `model.js` and `chart-data.js` are unchanged.
- Production Sites packaging succeeds. All JavaScript modules, including the new guidance module, are in the same content-hashed release as the simulation worker. Source/packaged-module checks and whitespace checks pass.
- Browser checks: desktop 1280 × 900; mobile 390 × 844 and narrow 320 × 740; no page-level horizontal overflow in household, account, income or review screens. Household choices work with Enter; numeric keyboard edits distinguish blank/zero; source controls retain focus; expanded charts close with Escape; the monthly table can receive keyboard focus and scroll horizontally. Reload preserves the couple, balances and Estimated pension source. Duplicate input IDs were absent in the account screen; no browser console errors were observed.

These checks establish local behavior, not production deployment or exhaustive assistive-technology certification. Synthetic scenarios were used, with no real financial data.

## Screenshots

Before images are live-site captures taken before source edits. After images are local captures. Results screenshots use the original couple sample assumptions: both before and after show 9 of 10 lifetimes without shortfall, median ending balance $5,908,706 and median failure age 90 years 6 months. The separate pension-entry example uses $18,000/year and is marked Estimated; that income was restored to the original zero before capturing the matched results.

| Flow | Before | After |
| --- | --- | --- |
| Desktop household | [Before](ux-evidence/before-desktop-household.jpg) | [Guided household](ux-evidence/after-desktop-household.jpg) |
| Mobile accounts | [Before](ux-evidence/before-mobile-accounts.jpg) | [Full guided accounts](ux-evidence/after-mobile-accounts.jpg), [visible non-Roth guidance](ux-evidence/after-mobile-account-guidance.jpg) |
| Mobile pension | Existing income was under Guaranteed annual income | [Full income setup](ux-evidence/after-mobile-pension.jpg), [visible pension guidance](ux-evidence/after-mobile-pension-guidance.jpg) |
| Desktop results | [Before](ux-evidence/before-desktop-results.jpg) | [Result summary](ux-evidence/after-desktop-results.jpg), [direct explanation](ux-evidence/after-desktop-explanation.jpg), [steady assumptions and zero-survivor row](ux-evidence/after-desktop-steady.jpg) |
| Mobile results | [Before](ux-evidence/before-mobile-results.jpg) | [After](ux-evidence/after-mobile-results.jpg) |
| Mobile review | No guided review step | [Editable review](ux-evidence/after-mobile-review.jpg) |

## Follow-up review (October 1, 2026)

An independent check of both passes reran the engine before and after the change. The pre-change (`c9fb1c5`) and current engines produce byte-identical results for all four fixture scenarios. Under Node 16 the `couple-pension` fixture hash differs for both engines alike, so that mismatch is a runtime-version artifact, not a regression. No engine defect was found.

Usability problems remained in the guided flow, and these were fixed:

- On a 375-pixel phone, the Individual/Couple choice sat about 1,150 pixels down the page, below three notices, the section list and help text. Guided mode now uses a compact step list instead of the sidebar. The sample-values notice now follows the household choice, and the separate-person ownership note appears only for couples on the accounts step.
- In guided mode, your Roth IRA total was hidden inside a collapsed "Roth IRA savings" disclosure. Spouse balances also came after both savings-contribution groups, with spouse Roth records and conversions fully expanded. The accounts step is now ordered: your accounts, spouse accounts, shared balances, a collapsed Roth history for both people, living costs, the earnings explanation, then future savings.
- The spouse's own Social Security amount now comes before their claim age, matching your section. Spouse dates come before the longevity table. Working household support is collapsed unless the two retirement dates differ.
- The review listed about 90 rows, then ended with unlabeled, editable Roth fields. It now opens with an "At a glance" summary that shows sources and Edit links, and names each Unknown input with a link to fix it. The full grouped list sits in a disclosure, and past conversions are summarized read-only.
- The earnings explanation is a four-point list placed beside the future-savings inputs. It warns against using total balance growth as a return assumption.
- The steady-growth explanation leads with "An illustration, not a forecast" and lists its lifespan, return, inflation and other assumptions. It states that steady average returns often exceed the middle Monte Carlo path, and that it can show balances at ages no simulated path reached.
- Shared-date guidance for saved plans now points to **Use separate-person inputs** instead of saying spouse dates and benefits are unsupported. Roth IRA deposit guidance no longer mentions payroll. Ages read "1 month" rather than "1 months".

Still needs a product decision:

- Resolved: the 13.3% sample return stays as the default. It is close to the S&P 500's simple average yearly total return over the last 50 years: 13.2% for 1976–2025 and 13.6% for 1975–2024 (Damodaran data). The field guidance and the methodology section `#sample-returns` now explain this, along with its caveats: returns are before inflation and fees, the sample assumes an all-stock portfolio, and the window is a strong period. The 2.3% sample inflation is below the 1976–2025 CPI average of about 3.6%; it is disclosed and intentionally unchanged for now.
- Resolved: savings belong in the future-savings deposits, not in a higher pre-retirement return. The one-year statement check separates the two.
- Open: the steady-growth illustration compounds at the entered average (13.3%), while the middle Monte Carlo path compounds at about 12.2%. Using the median rate would make the illustration less optimistic, but it changes a calculation.
- Employer Roth 401(k) and 403(b) balances have no clear home. The Roth fields model Roth IRA ordering only.

## Changed files

- `dist/app.js`: guided setup, household grouping, labels/help, editable review, provenance persistence, missing-input behavior and result explanations.
- `dist/ux-guidance.js` (new): plain-language guidance and source metadata normalization.
- `dist/result-format.js`: counts, percentage rounding and missing-survivor display rows.
- `dist/charts.js`: funding/lifespan labels, denominators, future-dollar captions and absent-outcome descriptions.
- `dist/style.css`: responsive setup, sources, progress, focus and result explanations.
- `dist/index.html`: asset version references.
- `dist/methodology.html`: complete metric definitions, steady assumptions, inflation basis and household-model limitations.
- `tests/app-state.test.js`, `tests/charts.test.js`: focused regressions and expected wording updates.
- `README.md`: behavior and validation notes.
- `docs/first-time-ux.md`, `docs/ux-evidence/*.jpg`: findings, validation, proposals and before/after evidence.

## Follow-up from a first-time walkthrough (October 2, 2026)

- Basic Accounts & spending now has three tasks: Account balances, Everyday spending and Future savings. Previous/Next moves between them before continuing to Income. Remaining required answers update during entry, and review groups unfinished work by task with direct edit links. All settings and the detailed editor remain available.
- Roth IRA history includes a $50,000 balance / $30,000 contribution example and explains contribution, funding-year and conversion records. I don't know these details yet saves the draft with Unknown provenance and retains prior numeric values without using them in a calculation. A known zero Roth balance still settles history automatically. The account guide explicitly identifies Roth employer accounts as unsupported; it does not substitute Roth IRA rules.
- Monthly budget totals now ask whether mortgage, rent or healthcare payments are included. Selected costs expose their deductions. Copying from the preceding entered month changes only housing/health deductions and invalidates the final confirmation; statement totals, annual-bill deductions and applied spending stay intact until Apply. A selected payment without an amount blocks application.
- Scenario lab can test a retirement date and monthly spending together alongside the unchanged baseline and both independent comparisons. Each changed row can create a personal copy while preserving the source plan and its full simulation count.

This follow-up leaves the investment assumptions, allocation, withdrawal defaults and simulation model unchanged.

## Selected improvements from the October 2 walkthrough

Implemented suggestions 1, 2, 4 and 7 locally, with publication still deferred:

- Roth I don't know offers a saved-draft exit and an explicitly chosen illustration. Temporary inputs only replace Unknown fields, preserve entered records/conversions, use $0 remaining regular contributions and the latest supported funding year (2026, or an earlier known conversion year), and carry Estimated source notes through copies/reload/backups. Review, results, comparisons and reports disclose the estimate and its limitations. This may increase modeled taxes and penalties; unknown past conversions still need review. Distribution rules continue to follow the existing model and [IRS Publication 590-B](https://www.irs.gov/publications/p590b).
- Pension, home, mortgage and rent questions show Yes / No / Not sure. New personal plans require these answers; No records explicit zeros, and Not sure preserves Unknown values. Existing plans retain their amounts. Choice notes are stored outside the simulation assumptions and included in copies and backups.
- Compare with lower returns runs the current plan and a paired alternative with each of the pre-retirement and stock averages capped at 7%. It never raises a lower rate or changes volatility, bonds, allocation or any other input, and preserves the saved plan and completed result. The existing comparison flow offers a personal copy only when requested.
- Phone layouts use short visible preview/model cautions, expandable full explanations and full-width result actions. Comparison explanations start collapsed. Full warnings remain in exported reports.

Validation: all 479 web tests passed, JavaScript syntax and whitespace checks passed, and `npm run build:sites` prepared the Worker package. A local browser walkthrough at 390 × 844 and 1280 × 1000 verified saved Not sure answers, explicit No choices, home/mortgage dependencies, the Roth alternatives, Estimated disclosure after reload and on results, the paired lower-return comparison, preserved original results, and expandable preview explanations. Neither viewport had horizontal content overflow. Publication remains deferred.

Screenshots from illustrative local data: [phone answers](ux-evidence/selected-ux-phone-answers.jpg), [phone Roth alternatives](ux-evidence/selected-ux-phone-roth-options.jpg), [desktop results](ux-evidence/selected-ux-desktop-results.jpg).

## Four selected fixes from the next October 2 walkthrough

- Applying a budget previews the current and proposed annual base spending and property tax/home-insurance values. Untouched annual bills preserve existing home costs and their source; the replacement checkbox supports entered bills or an explicit zero.
- Example inputs offer Use this value to confirm the displayed value without changing it. Entered provenance survives reload, copies and backups.
- Pre-Medicare premiums read Not needed when known household dates put every adult at 65 or older when costs begin. Earlier retirement or younger spouses restore the saved input and its source. Review, reports and planning notes follow the same relevance rule.
- Completed results use a separate local IndexedDB cache and show their calculation time. Reload restores matching results; changed calculation inputs, a new calendar day or a changed engine require a new run. Draft budgets, plan names and the next-run path count do not invalidate a completed run. JSON backups contain inputs only. Storage failures keep the current result visible and explain regeneration after reload. No scenario or result data is uploaded.

Validation: all 487 web tests passed, the production Worker build succeeded, and whitespace checks passed. The packaged browser walkthrough verified preservation of $4,000 home costs, the explicit zero-replacement preview, confirmed values after reload, Not needed premiums at 65 and Unknown premiums after moving retirement to 63, unchanged dated results after reload, and stale-result regeneration guidance. The budget fit a 390 × 844 phone viewport without horizontal overflow. [Restored results screenshot](ux-evidence/four-fixes-saved-results.jpg) uses illustrative local data.

## Selected issues 2, 3, 4 and 6 from the latest walkthrough

- Each plan remembers its guided or detailed editing mode. A reload on Results followed by Review sample inputs and Edit claim age returns to the guided Income step and keeps an explicit No pension answer simplified. The legacy remembered guided step is recognized; explicitly skipping the guide and switching between plans retain the corresponding choices.
- The result verdict and its first action precede calculation timestamps, dollar-basis controls, Roth details and upgrade copy. The preview caution remains in the answer, with a brief estimated-Roth caution beside it. At a 390 × 844 viewport, the verdict began at 267 pixels and the primary comparison action ended at 523 pixels; content had no horizontal overflow.
- Build my forecast checks for Roth 401(k)/403(b) accounts before collecting personal setup amounts. Yes and Not sure explain unsupported or incomplete coverage and offer a supported-account illustration or sample plan. The account check survives reload, copies and JSON backups, and is flagged in results and reports without altering calculation inputs. Saved plans retain their existing values.
- Reports display a structured plan summary before the collapsed detailed appendix. Summary and full-report text downloads have distinct names; full reports retain precise assumptions and disclosures. Printing uses the summary by default and includes the appendix when it is opened. A shared status marks temporary, missing, example, unrecorded and migrated Roth histories as requiring review for each active person; a known zero balance needs no history. Spending-model names use the same readable labels as setup.

Validation: all 495 web tests passed, JavaScript syntax and whitespace checks passed, and the production Worker package built successfully. Eight added regressions cover guided/detailed reload choices, per-plan and legacy modes, the early compatibility flow, account-check copies/backups, result ordering, summary/full downloads, both people's Roth statuses and unfinished or escaped summary inputs. Local browser checks verified the existing illustrative saved result, its guided claim-age edit with pension fields hidden, consistent temporary-Roth report status, all three account answers, answer persistence after reload, and entry into household setup. Desktop report and phone result/account layouts were visually checked. Browser PDF printing was not exercised. Publication remains deferred.

Screenshots use illustrative local data: [phone result](ux-evidence/four-ux-phone-results.jpg), [early phone account check](ux-evidence/four-ux-phone-account-check.jpg), [desktop report](ux-evidence/four-ux-desktop-report.jpg).

## Selected issues 1, 4, 5 and 6 from the retired-user walkthrough

- Household setup asks whether each person is already retired. A retired person's effective forecast date is today, using current balances and costs without replaying past returns, deposits or withdrawals. Their actual separation date is optional and accepts past dates; Rule of 55 checks retain the actual year and do not assume eligibility when that date is missing. Future contributions stop for that person, while a separately modeled working spouse can continue savings and take-home support. Toggling status preserves the other date draft in the current tab. Status and the active date survive reloads, copies and JSON backups. Existing SEPP schedules remain unsupported; the option models a new series from the forecast date.
- Everyday spending includes an inline calculator: total monthly spending less selected mortgage principal/interest, rent and household health premiums. Property taxes, home insurance and medical costs other than premiums stay in base spending. The preview accepts explicit zero, rejects missing, invalid or excessive deductions, and handles decimal payments without a false negative balance from floating-point rounding. Applying updates base spending with Estimated provenance; separate housing, healthcare and budget records remain intact. Calculator drafts stay in the current tab, and reset/import clears them.
- Results show the first month's living costs, income, gross amount needed from savings and modeled taxes from the existing steady-growth illustration. An unfunded first month is flagged. Start dates and waiting periods cover pensions, own Social Security and modeled spousal benefits, including claims that must wait for the other person's claim. Amounts follow the selected future/today-dollar basis using the cash-flow interval's starting price index. The card labels this illustration and links back to income setup; it is not presented as a median sampled lifetime.
- Navigation uses My plan, Saved plans and Compare changes. Household setup offers Name your plan, and personal forecasts starting from a default example receive My retirement plan. Existing custom names are retained. Naming preserves completed results, and names survive copies, reloads and JSON backups.

Validation: 509 web tests pass, including 14 new regressions covering retired timelines, either spouse working, actual separation-year eligibility, normalization/comparisons, spending preservation and validation, monthly cash flows and income waits, result retention and backup behavior. The production Worker package builds, and JavaScript syntax and whitespace checks pass. Source and packaged JavaScript modules, CSS and methodology were compared. A local packaged browser walkthrough verified the retired plan, a saved January 2024 separation date, $6,000 minus $1,500 mortgage and $500 premiums producing $4,000 monthly/$48,000 annual base spending, preserved housing entries, named-plan reload, restored and newly calculated results, and the income-edit link. Desktop (1280 × 1000) and phone (390 × 844) layouts were visually checked; the phone setup, calculator and results had no horizontal overflow. Hosted billing/authentication were not exercised. Publication remains deferred.

Screenshots use illustrative local data: [desktop monthly picture](ux-evidence/retired-forecast-desktop-results.jpg), [phone monthly picture](ux-evidence/retired-forecast-phone-results.jpg), [phone retired setup](ux-evidence/retired-forecast-phone-setup.jpg), [phone spending calculator](ux-evidence/retired-forecast-phone-spending.jpg).

## Employer Roth support follow-up

The unsupported-account findings above describe the earlier release. The account check now directs Roth 401(k)/403(b) balances into separate employer Roth records in Account balances. Missing plans and uncertain account types remain flagged. See [the implemented behavior and remaining limits](employer-roth.md).
