# Retirement Readiness Lab Implementation Plan

## 1. Product Vision

Retirement Readiness Lab is a polished Android application that helps people answer a practical question:

> Can I retire, what could go wrong, and which decision improves my odds the most?

The app should feel more like a guided retirement decision lab than a spreadsheet or technical simulator. The existing Python retirement simulator proves the core modeling concept. This new program should be treated as a separate product with a mobile-first interface, clearer workflows, stronger visual design, and a user experience built around scenario comparison.

## 2. Project Separation

Keep this program separate from the existing desktop/Python application.

Recommended future structure:

```text
RetirementReadinessLab/
  IMPLEMENTATION_PLAN.md
  README.md
  android/
    app/
    build.gradle.kts
    settings.gradle.kts
  docs/
    product_requirements.md
    design_system.md
    simulation_model.md
    privacy_and_compliance.md
  samples/
    example_scenarios.json
```

Initial work in this folder should be documentation-first. Actual Android source code can be added under `RetirementReadinessLab/android/` later.

## 3. Target Users

Primary users:

- People aged 45-65 deciding when they can retire.
- Early retirees and FIRE-adjacent planners.
- DIY investors who understand savings buckets but do not want to maintain a large spreadsheet.
- Households worried about healthcare, taxes, Social Security timing, market downturns, and long-term care.

Secondary users:

- Financial coaches who want an educational visual aid.
- People preparing for a meeting with a financial planner.
- Retirees testing whether spending changes are sustainable.

## 4. Product Positioning

The app should not be marketed as a generic calculator. Position it as a retirement stress-test and decision comparison tool.

Suggested positioning:

> A private, tax-aware retirement stress test for Android. Compare retirement ages, Social Security timing, Roth conversion strategies, healthcare costs, market downturns, and long-term care risk in minutes.

Core promise:

- Show the user's retirement readiness as a probability range, not a false guarantee.
- Explain the biggest risks in plain language.
- Show which levers improve the result most.
- Keep user data local unless the user explicitly exports it.

## 5. MVP Scope

The first release should be useful without becoming too broad.

MVP features:

- Guided onboarding for household basics.
- Manual entry of retirement accounts, savings, spending, mortgage, healthcare, and Social Security assumptions.
- Monte Carlo retirement simulation.
- Dashboard with readiness score and key risks.
- Scenario comparison with at least three saved alternatives.
- Social Security timing comparison.
- Roth conversion toggle with bracket cap selection.
- Healthcare and Medicare premium modeling.
- Long-term care stress test toggle.
- Dynamic withdrawal or cash reserve strategy toggle.
- Local-only persistence.
- Exportable summary report.
- Clear assumptions and disclaimer screen.

Out of scope for MVP:

- Bank or brokerage account linking.
- Personalized investment advice.
- State tax support.
- Estate planning.
- Spousal Social Security optimization beyond basic household support.
- Cloud sync.
- Advisor marketplace.

## 6. Differentiating Features

Focus on features that make the app feel deeper than a simple savings calculator.

High-value differentiators:

- "Earliest retirement age" finder.
- "Safe spending range" finder.
- Side-by-side scenario cards.
- Risk attribution: market sequence risk, longevity risk, healthcare risk, tax risk, and spending risk.
- Roth conversion scenario comparison.
- Social Security claim age comparison.
- Medicare/IRMAA estimate.
- Long-term care shock modeling.
- Cash reserve drawdown strategy.
- Local-first privacy message.

## 7. Recommended Technical Stack

Use a native Android stack for the marketable product.

Recommended stack:

- Language: Kotlin.
- UI: Jetpack Compose.
- Design foundation: Material 3 with custom theming.
- Architecture: MVVM or unidirectional state flow.
- Async: Kotlin coroutines and Flow.
- Persistence: Room for scenarios and simulation runs; DataStore for preferences.
- Charts: Compose-compatible chart library or custom Canvas charts.
- Reports: Android PDF generation or HTML-to-PDF export.
- Testing: JUnit, Kotlin test, Compose UI tests, screenshot tests.
- Build: Gradle Kotlin DSL.

Avoid:

- Porting the Tkinter UI.
- Building the Android app directly in Kivy unless speed is more important than long-term polish.
- Requiring users to create accounts for the MVP.
- Adding network calls for market data in the first release.

## 8. Licensing Decision

The current repository is GPLv3. Before commercializing the Android app, decide whether the new Android app will:

- Remain GPL-compatible and publish source for distributed builds.
- Reimplement the simulation logic from scratch under a different license, if legally available.
- Use the existing code and accept GPL obligations.
- Move the new app to a separate repository with a clear license boundary.

This decision should be made before porting `core.py` logic into Kotlin or sharing code between projects.

## 9. High-Level Architecture

Recommended modules:

```text
app
  Application entry point, navigation, dependency setup.

core-model
  Domain entities, simulation input models, result models, validation.

simulation-engine
  Monte Carlo engine, tax calculations, Social Security, healthcare, mortality,
  long-term care, Roth conversion logic, dynamic withdrawal logic.

scenario-store
  Room database, scenario persistence, import/export.

ui-design
  Theme, reusable components, charts, layout primitives.

feature-onboarding
  Guided setup screens.

feature-dashboard
  Readiness dashboard and risk summary.

feature-scenarios
  Scenario list, scenario editor, comparison screen.

feature-reports
  PDF/HTML export and share sheet.
```

Keep the simulation engine independent from Android UI APIs so it can be unit tested heavily.

## 10. Domain Model

Core entities:

- `HouseholdProfile`
- `PersonProfile`
- `AccountBalance`
- `SpendingPlan`
- `MortgagePlan`
- `HealthcarePlan`
- `SocialSecurityPlan`
- `TaxSettings`
- `MarketAssumptions`
- `WithdrawalStrategy`
- `RothConversionStrategy`
- `LongTermCareAssumption`
- `RetirementScenario`
- `SimulationResult`
- `RiskBreakdown`
- `ScenarioComparison`

Example scenario fields:

- Scenario name.
- Current age.
- Retirement age.
- Filing status.
- Current taxable/pre-tax/Roth/cash balances.
- Annual spending need.
- Mortgage payment and years left.
- Pre-Medicare healthcare premium.
- Social Security estimate at full retirement age.
- Social Security claim age.
- General inflation assumption.
- Healthcare inflation assumption.
- Stock and bond assumptions.
- Mortality table selection.
- Long-term care enabled flag.
- Roth conversion strategy.
- Dynamic withdrawal strategy.

## 11. Simulation Engine Plan

Port the existing model deliberately rather than line-by-line.

Phase 1 engine behavior:

- Monthly simulation steps.
- Pre-retirement portfolio growth.
- Retirement withdrawals.
- Tax-aware gross withdrawal estimate.
- Social Security payout adjustment by claim age.
- Social Security taxation.
- Federal tax brackets.
- Medicare Parts B/D premium estimates.
- IRMAA surcharge estimate.
- Pre-Medicare healthcare cost inflation.
- Mortgage phase-out.
- Long-term care event modeling.
- Mortality table sampling.
- Roth conversion up to selected tax bracket.
- Dynamic withdrawal using cash reserves.
- Success rate calculation.

Phase 2 engine behavior:

- Percentile paths.
- Median ending balance.
- Failure age distribution.
- Retirement age optimizer.
- Safe spending finder.
- Sensitivity analysis.
- Risk attribution.
- Deterministic random seed for repeatable reports.

Important implementation notes:

- Keep tax tables versioned by year.
- Store assumptions used for each simulation result.
- Return warnings when inputs are incomplete, unrealistic, or unsupported.
- Make simulation results reproducible for saved reports.
- Track calculation provenance so the assumptions screen can explain the result.

## 12. User Experience Principles

The app should feel calm, trustworthy, and decision-focused.

UX principles:

- Ask for only the minimum needed to produce a first result.
- Let advanced assumptions be edited later.
- Make every result explainable.
- Prefer side-by-side choices over dense forms.
- Show dollars in rounded, readable values by default.
- Use progressive disclosure for advanced tax and market assumptions.
- Show uncertainty visually without making the app feel alarming.
- Always make clear that results are educational estimates.

Avoid:

- A spreadsheet-like first screen.
- Dozens of exposed technical assumptions during onboarding.
- Overconfident language like "you are safe."
- Dense paragraphs explaining how to use the app.
- Dark, visually heavy finance-dashboard styling.

## 13. Visual Design Direction

The app should look premium but restrained.

Recommended visual style:

- Light-first theme with optional dark mode.
- Soft neutral background.
- Deep green or blue-green for confidence and progress.
- Amber for caution.
- Red only for serious risk states.
- Rounded corners no larger than 8dp for most cards and controls.
- High contrast body text.
- Large readable financial numbers.
- Charts that emphasize ranges and outcomes.
- Minimal decoration.

Suggested palette:

- Background: `#F7F8F5`
- Surface: `#FFFFFF`
- Primary: `#176B5B`
- Primary dark: `#0E463C`
- Accent: `#D6A23A`
- Success: `#2F7D4E`
- Caution: `#B7791F`
- Risk: `#B94A48`
- Text: `#17201C`
- Muted text: `#65706A`
- Divider: `#DDE3DE`

Typography:

- Use the platform default or a highly readable sans-serif.
- Reserve largest type for dashboard result numbers.
- Use compact section headings in forms and cards.
- Keep labels short.
- Avoid negative letter spacing.

Motion:

- Use subtle transitions between onboarding steps.
- Animate chart updates after a simulation completes.
- Do not animate large numbers continuously.
- Avoid decorative motion that makes the app feel less serious.

## 14. Main Navigation

Recommended top-level destinations:

- Dashboard.
- Scenarios.
- Lab.
- Assumptions.
- Reports.

Navigation behavior:

- First launch starts in onboarding.
- After onboarding, open to Dashboard.
- Primary action on Dashboard: "Run Stress Test" or "Update Plan".
- Scenario comparison should be one tap from Dashboard.
- Advanced assumptions should be available but not prominent.

## 15. Screen Plan

### 15.1 First Launch

Goal:

- Communicate privacy and purpose quickly.
- Start setup without marketing fluff.

Required elements:

- App name: Retirement Readiness Lab.
- Short subtitle: "Stress-test when you can retire."
- Local-first privacy statement.
- Continue button.
- Optional sample plan button.

### 15.2 Guided Setup

Suggested steps:

1. Household basics.
2. Retirement target.
3. Current balances.
4. Spending and mortgage.
5. Healthcare.
6. Social Security.
7. Strategy toggles.
8. Review assumptions.

Design notes:

- Use a progress indicator.
- Each step should contain 2-5 fields.
- Use steppers or sliders for ages.
- Use currency inputs for money.
- Use segmented controls for filing status and gender/mortality table.
- Use switches for Medicare, LTC, Roth conversion, and cash reserve strategies.
- Show small calculated hints, such as "8 years until retirement."

### 15.3 Dashboard

Primary dashboard sections:

- Readiness score: "82% of simulations lasted."
- Retirement target: "Retire at 58."
- Safe spending estimate.
- Earliest retirement age estimate.
- Biggest risk.
- Recommended scenario to test next.
- Mini chart of outcome distribution.

Dashboard cards:

- Confidence.
- Spending.
- Age.
- Healthcare.
- Taxes.
- Market downturn.

Design notes:

- The top section should be visually strong but not a marketing hero.
- Use one primary chart above the fold.
- Keep advanced details below the fold.
- Show the last simulation date and assumption snapshot.

### 15.4 Scenario List

Scenario list card content:

- Scenario name.
- Retirement age.
- Annual spending.
- Social Security claim age.
- Success probability.
- Key risk label.
- Last run timestamp.

Actions:

- Duplicate scenario.
- Edit scenario.
- Compare scenario.
- Archive/delete scenario.

### 15.5 Scenario Editor

Editor sections:

- Profile.
- Accounts.
- Spending.
- Healthcare.
- Social Security.
- Tax strategy.
- Market assumptions.
- Advanced risk assumptions.

Design notes:

- Use collapsible sections.
- Show validation inline.
- Show calculated derived values inside each section.
- Keep raw market assumptions in an advanced section.

### 15.6 Lab

The Lab is the app's most marketable feature.

Lab tools:

- Retirement age sweep.
- Spending sweep.
- Social Security claim age comparison.
- Roth conversion comparison.
- LTC shock test.
- Healthcare inflation stress test.
- Market downturn test.
- Mortgage payoff comparison.

Each tool should show:

- Short input control.
- Results chart.
- Best/worst case summary.
- Plain-language takeaway.

Example takeaway:

> Claiming Social Security at 70 improved the median ending balance but did not materially change the failure rate because healthcare spending before 65 is the largest pressure point.

### 15.7 Comparison Screen

Compare 2-4 scenarios.

Columns:

- Scenario name.
- Retirement age.
- Spending.
- Claim age.
- Roth conversion setting.
- Cash reserve setting.
- Success probability.
- Median ending balance.
- 10th percentile ending balance.
- Most common failure decade.

Visuals:

- Bar chart for success probability.
- Range chart for ending balance.
- Highlight changed assumptions.

### 15.8 Results Detail

Detailed results:

- Success rate.
- Failure distribution by age.
- Median path.
- 10th/25th/75th/90th percentile paths.
- Income sources over time.
- Taxable withdrawals over time.
- Healthcare costs over time.
- Medicare/IRMAA impacts.
- Roth conversion summary.
- Long-term care event impact.

### 15.9 Assumptions

Purpose:

- Build user trust by showing exactly what the app assumed.

Sections:

- Household.
- Spending.
- Accounts.
- Taxes.
- Social Security.
- Healthcare.
- Market returns.
- Inflation.
- Mortality.
- Long-term care.
- Simulation settings.

Include:

- Editable values.
- Version of tax tables.
- Version/date of mortality table.
- Simulation count.
- Random seed setting, if enabled.

### 15.10 Reports

Export options:

- PDF summary.
- JSON scenario backup.
- CSV result summary.

PDF sections:

- Cover summary.
- Scenario assumptions.
- Main result.
- Scenario comparison.
- Risk summary.
- Detailed assumptions.
- Educational disclaimer.

## 16. Data Visualization Plan

Required charts:

- Success/failure gauge or probability band.
- Portfolio percentile path chart.
- Ending balance distribution.
- Failure age histogram.
- Scenario comparison bar chart.
- Spending sensitivity curve.
- Retirement age sensitivity curve.

Chart design:

- Use clear labels and rounded dollar values.
- Keep axes readable on small screens.
- Avoid cluttered scatter plots for the primary mobile experience.
- Let users open detailed charts full-screen.
- Use consistent colors for success, caution, and risk.

## 17. Input Validation

Validation should prevent impossible inputs and warn about suspicious ones.

Hard errors:

- Retirement age is less than current age.
- Social Security claim age is below 62 or above 70.
- Negative balances.
- Negative spending.
- Invalid percentage values.
- Filing status missing.
- Simulation count less than 1.

Soft warnings:

- Very high expected stock returns.
- Very low inflation.
- Retirement age below 50.
- Spending much larger than assets.
- Healthcare premium missing before age 65.
- Long-term care disabled.
- No Social Security estimate entered.
- Mortgage years exceed plausible lifetime horizon.

## 18. Privacy And Compliance Plan

The app handles sensitive financial data even if it never connects to banks.

Privacy requirements:

- Store all data locally by default.
- Do not require account creation.
- Do not send financial data to analytics.
- Make backups/export explicit.
- Provide a privacy policy before release.
- Provide data deletion controls.
- Avoid collecting personally identifying data unless necessary.

Compliance and messaging:

- Add educational-use disclaimer.
- Do not call outputs financial advice.
- Avoid personalized investment recommendations.
- Avoid "guaranteed" language.
- Keep assumptions visible.
- Prepare required app store financial feature declarations before release.

## 19. Monetization Plan

Recommended model:

- Free version: one saved scenario, basic Monte Carlo result, basic assumptions.
- Pro version: unlimited scenarios, Lab tools, Roth conversion analysis, reports, advanced risk tests.

Possible pricing:

- One-time unlock for users who dislike subscriptions.
- Annual subscription if tax table updates, ongoing model improvements, and report features justify recurring value.

Avoid:

- Ads.
- Selling user data.
- Dark patterns around retirement anxiety.
- Locking basic data export behind a paywall.

## 20. Accessibility Requirements

Accessibility should be part of the initial design.

Requirements:

- Support dynamic font sizes.
- Minimum touch target of 48dp.
- High contrast text.
- Meaningful content descriptions for icons and charts.
- Do not rely only on color to communicate risk.
- Full keyboard and screen reader navigation for forms.
- Respect reduce-motion preferences where available.

## 21. Testing Plan

Unit tests:

- Tax bracket calculations.
- Social Security payout calculation.
- Social Security taxation.
- Medicare premium and IRMAA estimates.
- Roth conversion behavior.
- Dynamic withdrawal behavior.
- Long-term care event sampling.
- Mortality sampling.
- Scenario validation.
- Result summarization.

Simulation tests:

- Deterministic seed produces repeatable output.
- Zero-return scenarios behave predictably.
- No-spending scenario always succeeds.
- No-assets scenario fails when spending is positive.
- Medicare costs begin at age 65.
- Social Security begins at the selected claim age.
- Mortgage payments stop after the configured period.

UI tests:

- Onboarding can produce a runnable scenario.
- Scenario can be saved, edited, duplicated, and deleted.
- Comparison screen handles 2-4 scenarios.
- Validation messages appear for invalid inputs.
- Dashboard renders with no simulation result yet.
- Dashboard renders with completed result.

Visual tests:

- Light mode and dark mode screenshots.
- Small phone viewport.
- Large phone viewport.
- Tablet layout.
- Large font mode.
- Empty state screens.
- Error state screens.

Performance tests:

- 1,000 simulations complete quickly on mid-range devices.
- 10,000 simulations complete without UI blocking.
- Scenario list remains smooth with many saved scenarios.
- Reports export within a reasonable time.

## 22. Implementation Phases

### Phase 0: Product And Design Foundation

Deliverables:

- Product requirements document.
- Design system document.
- Screen wireframes.
- Simulation model specification.
- Licensing decision.
- Privacy/compliance notes.

Exit criteria:

- MVP scope is frozen.
- Core screens are designed.
- Simulation input/output contract is documented.

### Phase 1: Android Project Skeleton

Deliverables:

- New Android project under `RetirementReadinessLab/android/`.
- Compose navigation shell.
- Theme and reusable UI components.
- Local persistence foundation.
- Basic scenario model.

Exit criteria:

- App launches.
- Navigation works.
- Theme supports light and dark mode.
- Scenario can be created and saved locally.

### Phase 2: Simulation Engine Port

Deliverables:

- Kotlin simulation engine.
- Tax calculation module.
- Social Security module.
- Healthcare/Medicare module.
- Mortality and LTC module.
- Roth conversion module.
- Dynamic withdrawal module.
- Unit test suite.

Exit criteria:

- Engine produces stable, testable results.
- Known edge cases are covered.
- Simulation can run from a sample scenario.

### Phase 3: Guided Onboarding And Dashboard

Deliverables:

- First launch flow.
- Guided setup screens.
- Dashboard summary.
- Basic chart visualizations.
- Assumptions review screen.

Exit criteria:

- A new user can complete onboarding and get a result.
- Dashboard is visually polished.
- Invalid inputs are handled clearly.

### Phase 4: Scenario Lab

Deliverables:

- Scenario duplication.
- Scenario comparison.
- Retirement age sweep.
- Spending sweep.
- Social Security timing comparison.
- Roth conversion comparison.
- Healthcare/LTC stress tests.

Exit criteria:

- User can compare multiple decisions.
- Results communicate tradeoffs clearly.
- Lab tools are fast enough for interactive use.

### Phase 5: Reports And Export

Deliverables:

- PDF report export.
- JSON scenario backup.
- CSV summary export.
- Android share sheet integration.

Exit criteria:

- Reports include assumptions and disclaimers.
- Exported files can be shared and reopened.
- No sensitive data is exported without explicit user action.

### Phase 6: Polish, Compliance, And Release Prep

Deliverables:

- App icon.
- Store screenshots.
- Privacy policy.
- Educational disclaimer.
- Play Store listing copy.
- Closed testing build.
- Accessibility pass.
- Performance pass.

Exit criteria:

- App is stable on representative devices.
- Store assets are ready.
- Required policy declarations are complete.
- Beta users can complete core workflows.

## 23. Design Quality Checklist

Before considering the app release-ready:

- The first screen shows the actual app purpose immediately.
- A user can get a first result in under five minutes.
- Advanced assumptions do not overwhelm new users.
- Scenario comparison is easy to find.
- Dashboard results fit on small screens.
- Charts are readable without rotating the phone.
- Long numbers are formatted consistently.
- Every result has an assumptions link.
- Empty states are useful.
- Error states are calm and specific.
- Dark mode is not an afterthought.
- Accessibility text sizes do not break layouts.

## 24. Open Questions

Decide before implementation:

- Will the Android app be GPLv3, proprietary, or separately licensed?
- Should the MVP support couples or only one person?
- Should state taxes be deferred until after launch?
- Should market assumptions be editable in MVP or hidden behind advanced settings?
- Should reports be free or Pro-only?
- Should simulation run entirely on-device?
- Should a deterministic seed be exposed to users or kept internal?
- Which mortality table source should be used and how often should it be updated?

## 25. Initial Backlog

Product:

- Write product requirements document.
- Define MVP field list.
- Define Pro feature boundaries.
- Draft app store listing.
- Draft privacy policy and disclaimer.

Design:

- Create screen wireframes.
- Define typography and color tokens.
- Design dashboard cards.
- Design onboarding flow.
- Design chart components.
- Design empty, loading, and error states.

Engineering:

- Create Android project.
- Implement theme.
- Implement navigation.
- Implement scenario data model.
- Implement local persistence.
- Port simulation engine.
- Add deterministic tests.
- Implement onboarding.
- Implement dashboard.
- Implement scenario comparison.
- Implement reports.

Release:

- Create app icon.
- Create screenshots.
- Prepare closed testing release.
- Gather beta feedback.
- Fix onboarding confusion.
- Tune performance.
- Finalize store declarations.

## 26. Definition Of Done For MVP

The MVP is complete when:

- A user can create a retirement scenario from guided setup.
- The app can run a local Monte Carlo simulation.
- The dashboard shows a clear retirement readiness result.
- The user can create and compare at least three scenarios.
- The Lab includes at least retirement age, spending, Social Security, Roth conversion, healthcare, and LTC comparisons.
- All assumptions are visible and editable.
- Data persists locally.
- Basic export works.
- Unit tests cover core financial calculations.
- UI tests cover onboarding and scenario comparison.
- The app has a polished light theme and usable dark theme.
- Accessibility basics are verified.
- Privacy policy and educational disclaimer are ready for release.
