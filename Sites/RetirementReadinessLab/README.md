# Retirement Readiness Lab for Sites

A browser port of the native Android app in `../../Android/RetirementReadinessLab/`. The web app runs as a static Sites page. Scenarios and calculations stay in the browser; no scenario data is sent to a server. Export JSON to retain plans outside browser storage.

## Included

- A focused five-section assumptions editor, grouped navigation, mobile menu, and overview with direct links to the planning workflow. Inputs save without interrupting keyboard entry; simulations open the results view.

- Editable household, accounts, spending, income, Social Security, housing, healthcare, market, Roth conversion, and withdrawal assumptions. Spouse settings appear for married households; the maximum modeling age is in advanced settings and is not a lifespan prediction. Explanations beside all accounts, spending, income, Social Security, housing, healthcare, market, and withdrawal inputs open on click or tap and close with Escape. Stock-allocation settings start collapsed, and the age-67 benefit input links to the official my Social Security account.
- Statement-based budget: monthly card purchases, direct bank spending, and cash withdrawals; per-month deductions for annual bills and housing/health premiums modeled separately; annual bills added once; optional retirement spending adjustment; calculation breakdown and explicit application to the plan. Duplicate months and excessive deductions are blocked. Uses the latest 12 months, with a short-sample warning. Draft edits preserve the applied spending and home-sale cost assumptions. Existing budgets and JSON backups remain compatible.
- Roth conversion caps are selected from the seven supported federal brackets; invalid saved values are shown explicitly until corrected.
- Scenario copies, scenario comparisons, retirement-age and safe-spending target searches.
- Seeded monthly Monte Carlo model with Android-matched 2026 tax rules, SSA mortality tables, Medicare premiums, long-term care, home equity, and survivor benefits.
- Android-style funding/survival curves and logarithmic simulation scatter plots (up to 30,000 deterministically sampled points), mean line, outcome colors, labeled axes, age inspection, and expanded zoom/pan views. The separate percentile balance plot and numerical tables remain available.
- Balance summaries count failed endings as $0. Age-based balance bands use only observed paths through death or failure (including a zero at the failure age), show sample counts, and do not pad to the maximum modeling age. This reporting behavior intentionally differs from Android’s carried-forward balance bands; readiness and failure ages keep the same cashflow calculations.
- Results charts and tables, text report, browser print/PDF, JSON backup and restore. Android scenario arrays can be imported.

The web app omits Android billing and platform-specific entitlement controls. Monthly budget editing uses category totals; imported line items remain in JSON until that category is edited. New budget adjustment fields are specific to the web app and may not be used by the Android app. The web result view currently omits Android's funding-threshold summary card. Results are recalculated after a page reload rather than stored.

## Verify

From the repository root:

```bash
npm test --prefix Sites/RetirementReadinessLab
python -m http.server 4173 --directory Sites/RetirementReadinessLab/dist
```

The Node tests include fixed outputs captured from the Android simulator and optimizer with identical scenarios and seeds. Because this is a buildless static Site, `dist/` is both tracked source and deployment output.
