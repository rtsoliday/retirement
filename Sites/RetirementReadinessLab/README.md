# Retirement Forecast - Monte Carlo Simulator for Sites

A browser port of the native Android app in `../../Android/RetirementReadinessLab/`. The web app runs as a static Sites page. Scenarios and calculations stay in the browser; no scenario data is sent to a server. Export JSON to retain plans outside browser storage.

## Included

- A focused five-section assumptions editor, grouped navigation, mobile menu, and overview with direct links to the planning workflow. The welcome leads with Build my forecast and Explore a sample plan, a clearly labeled illustrative savings chart, local-input privacy, and the four-path preview limit. Visitors with saved plans see Continue my plan and their selected scenario; automatic Pro defaults do not mark a new visitor as returning. Free visitors see a priced Pro comparison below the introduction and planning workflow, plus a four-path explanation beside the readiness result; Pro visitors see neither upgrade prompt. Inputs save without interrupting keyboard entry; simulations open the results view.

- Editable household, accounts, spending, income, Social Security, housing, healthcare, market, Roth conversion, and withdrawal assumptions. Spouse settings appear for married households; the maximum modeling age is in advanced settings and is not a lifespan prediction. Explanations beside all accounts, spending, income, Social Security, housing, healthcare, market, and withdrawal inputs open on click or tap and close with Escape. Stock-allocation settings start collapsed, and the age-67 benefit input links to the official my Social Security account.
- Statement-based budget: monthly card purchases, direct bank spending, and cash withdrawals; per-month deductions for annual bills and housing/health premiums modeled separately; annual bills added once; optional retirement spending adjustment; calculation breakdown and explicit application to the plan. Duplicate months and excessive deductions are blocked. Uses the latest 12 months, with a short-sample warning. Draft edits preserve the applied spending and home-sale cost assumptions. Existing budgets and JSON backups remain compatible.
- Roth conversion caps are selected from the seven supported federal brackets; invalid saved values are shown explicitly until corrected. Conversion amounts and taxes include the extra Social Security that a conversion makes taxable, which Android does not.
- Scenario copies, scenario comparisons, retirement-age and modeled-spending target searches. Unlike Android, the retirement-age search keeps the plan's early-withdrawal penalty setting, and a spending target that reaches the search limit is shown as "At least" that amount.
- Ages must be whole numbers. Imported backups must use known filing, longevity, and spending-path values and the same value types as the defaults; scenarios with missing or repeated IDs receive new unique IDs.
- Seeded monthly Monte Carlo model with Android-matched 2026 tax rules, SSA mortality tables, Medicare premiums, long-term care, home equity, and survivor benefits.
- Android-style funding/survival curves and logarithmic simulation scatter plots (up to 30,000 deterministically sampled points), mean line, outcome colors, labeled axes, age inspection, and expanded zoom/pan views. The separate percentile balance plot and numerical tables remain available.
- Balance summaries count failed endings as $0. Age-based balance bands use only observed paths through death or failure (including a zero at the failure age), show sample counts, and do not pad to the maximum modeling age. This reporting behavior intentionally differs from Android’s carried-forward balance bands; readiness and failure ages keep the same cashflow calculations.
- Results charts and tables, text report, browser print/PDF, JSON backup and restore. Android scenario arrays can be imported.

The web app defaults to four local Monte Carlo paths for anonymous and free users. Signed-in Pro subscribers default to 10,000 local paths and can choose a smaller count. The numeric seed is fixed and hidden; imports normalize old seeds to the fixed value. Four-path results are for preview only. Their outcome summaries, comparison rows, chart inspection, and reports show counts rather than readiness percentages, with a small-sample warning even for Pro accounts running four paths. The browser code is public and can be modified by a determined visitor, so this is a product access control rather than tamper-proof metering. Monthly budget editing uses category totals; imported line items remain in JSON until that category is edited. New budget adjustment fields are specific to the web app and may not be used by the Android app. The web result view currently omits Android's funding-threshold summary card. Results are recalculated after a page reload rather than stored.

## Verify

From the repository root:

```bash
npm test --prefix Sites/RetirementReadinessLab
python -m http.server 4173 --directory Sites/RetirementReadinessLab/dist
```

The Node tests include fixed outputs captured from the Android simulator and optimizer with identical scenarios and seeds. The browser app remains buildless in `dist/`. `npm run build:sites` packages its files with a small Worker that protects `/admin` and proxies Cloudflare Analytics requests; publish `.sites-build/` as the Sites artifact.

## Public model disclosures

The overview shows an educational-use disclosure below planning notes, and results show it below failure ages. Assumptions and budget show a shorter U.S.-only scope note. The scenario lab and reports omit the on-page disclosure. dist/methodology.html explains the readiness metric, 2026 U.S. federal model basis, important exclusions, and local scenario storage. Exported text reports repeat the disclosure. The public dist/license.html page reproduces the repository’s proprietary notice and does not link to GitHub. It governs reuse of code and content; it is not visitor Terms of Use.

## Owner traffic dashboard

`/admin` is an owner-only page. The Worker checks the Sites authenticated user ID before serving the page or `/api/admin/traffic`; anonymous visitors are sent through Sign in with ChatGPT. It asks Cloudflare's zone GraphQL API for daily page views and unique IP counts for `retirementforecast.us`. It does not receive or send retirement scenario data. The daily unique figures cannot be added to produce a distinct total across the period, so the page shows them by day and reports only the daily peak. Cloudflare's zone metrics include traffic on the custom domain and its subdomains, not the separate `chatgpt.site` origin; internal planner tabs are not separate HTML page views.

To activate live figures, create a Cloudflare API token scoped to this zone with **Account Analytics: Read**, with Zone Resources limited to `retirementforecast.us`, then set `CLOUDFLARE_ANALYTICS_TOKEN` as a secret in the Sites environment and redeploy the current version. Never put the token in source files. Cloudflare may delay or limit the available history, especially on free plans.

## Pro billing setup

The Sites Worker grants complimentary Pro access to the verified site owner and associates paying subscribers with a Stripe Customer by verified ChatGPT or Firebase account ID. Google users can sign in independently through Firebase after the provider setup below, and visitors can explicitly link them with an existing ChatGPT account. All scenario data and calculations stay in the browser. Stripe receives the user ID and billing details, not financial assumptions. The worker checks subscriber status with Stripe on each access check; owner access does not require a Stripe call. No local entitlement database or webhook is required. Customer Search can be briefly delayed after checkout; the return session is verified directly to cover that interval.

To enable Checkout, create monthly and yearly recurring Prices for the same Pro product in Stripe, configure its billing portal, and set these Sites variables: `STRIPE_SECRET_KEY` (secret), `STRIPE_PRO_MONTHLY_PRICE_ID`, and `STRIPE_PRO_YEARLY_PRICE_ID` (test prices), plus `STRIPE_PRO_LIVE_MONTHLY_PRICE_ID` and `STRIPE_PRO_LIVE_YEARLY_PRICE_ID` (live prices). The Worker selects the matching price pair from the secret key's mode and disables Checkout if that pair is missing. The plan screen offers $9.99 per month and $79 per year; Stripe Checkout confirms the selected price before payment. While the key is in Stripe test mode, Checkout and billing are available only to the signed-in Site owner; other visitors remain on the free preview. With a live key, anonymous visitors can use the Sign in with ChatGPT link before upgrading. The signed-in account can manage its subscription through Stripe Billing Portal. Checkouts, billing status, and portal session creation have server-side authentication; no Stripe secret is included in browser files.

Owner access uses the same Sites-authenticated owner check as `/admin` (owner account ID or authenticated email `rtsoliday@gmail.com`) or a signed Firebase Google token with the known owner UID or that verified email. An anonymous email or unverified Firebase token is insufficient. The owner can use up to 10,000 paths without a Stripe subscription.

Because the Monte Carlo engine is delivered to every browser, a person can edit the JavaScript and run more paths without paying. Keeping all calculation on the user's computer makes that limitation unavoidable. Do not claim that Pro prevents deliberate bypass.

## Google sign-in

Google uses Firebase Authentication, separate from Sites' built-in ChatGPT sign-in. The browser receives only the public Firebase web-app config. The Worker verifies each Firebase ID token's signature, issuer, audience, expiry, and sign-in provider before looking up Stripe; it never trusts an email address as proof that two accounts belong to one person. A customer can explicitly link a ChatGPT sign-in to a Google sign-in in **Plans & billing**. Existing ChatGPT subscriptions remain accessible through ChatGPT until linked. If both separate accounts already have active subscriptions, linking stops rather than choosing one. The browser stores the Firebase sign-in session locally; it still stores scenarios locally and does not upload them to Firebase.

To activate Google, create a Firebase project and web app in the [Firebase console](https://console.firebase.google.com/), enable **Authentication → Sign-in method → Google**, and add `retirementforecast.us`, `www.retirementforecast.us`, and `retirement-readiness-lab-web.rtsoliday123.chatgpt.site` to Firebase Authentication's authorized domains. In the Sites environment set these **public** web-app config values from Firebase project settings: `FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN`, `FIREBASE_PROJECT_ID`, and `FIREBASE_APP_ID`. Set `FIREBASE_GOOGLE_ENABLED=true` only after the Google provider and authorized domains are ready. Redeploy the current Site version after changing environment variables.

Test Google sign-in on the custom domain and the `chatgpt.site` origin before relying on it: sign in, start and cancel Checkout, then use a test subscription to verify the 10,000-path entitlement and portal access. Test existing ChatGPT Pro subscribers before and after explicit linking. The Worker token tests use generated signing keys and fake Stripe responses; they do not replace a live provider and Stripe test. If Firebase config is absent, or the Google enabled flag is false, the Google button stays hidden and existing ChatGPT billing continues to work.

## Launch safeguards and customer help

Scenario selection, reset, import, and assumption changes clear comparison and target results. A calculation revision guard discards worker responses for plans changed during a run. UI regression tests exercise the real app handlers with a minimal DOM/worker harness.

Public `support.html`, `privacy.html`, and `terms.html` pages are linked from the footer and Plans & billing. Support is `rtsoliday@gmail.com`; refund requests are reviewed individually without an automatic refund promise. Cancellation instructions defer to the effective date shown in Stripe’s configured portal. The privacy notice describes local scenarios separately from authentication, billing, support email, and hosting traffic.
