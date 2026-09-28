# Retirement Readiness Lab for Sites

A browser port of the native Android app in `../../Android/RetirementReadinessLab/`. The web app runs as a static Sites page. Scenarios and calculations stay in the browser; no scenario data is sent to a server. Export JSON to retain plans outside browser storage.

## Included

- Editable household, accounts, spending, income, Social Security, housing, healthcare, market, Roth conversion, and withdrawal assumptions. Spouse settings appear for married households; the maximum modeling age is in advanced settings and is not a lifespan prediction. Explanations beside all accounts, spending, income, Social Security, housing, healthcare, market, and withdrawal inputs open on click or tap and close with Escape. Stock-allocation settings start collapsed, and the age-67 benefit input links to the official my Social Security account.
- Budget estimate, scenario copies, scenario comparisons, retirement-age and safe-spending target searches.
- Seeded monthly Monte Carlo model with Android-matched 2026 tax rules, SSA mortality tables, Medicare premiums, long-term care, home equity, and survivor benefits.
- Results charts and tables, text report, browser print/PDF, JSON backup and restore. Android scenario arrays can be imported.

The web app omits Android billing and platform-specific entitlement controls. Monthly budget editing uses category totals; imported line items remain in JSON until that month is edited. The web result view currently omits Android's path scatter plot and funding-threshold chart. Results are recalculated after a page reload rather than stored.

## Verify

From the repository root:

```bash
npm test --prefix Sites/RetirementReadinessLab
python -m http.server 4173 --directory Sites/RetirementReadinessLab/dist
```

The Node tests include fixed outputs captured from the Android simulator and optimizer with identical scenarios and seeds. Because this is a buildless static Site, `dist/` is both tracked source and deployment output.
