# Retirement Forecast plugin submission draft

Prepared October 5, 2026. This is a draft package for your review, not a submitted directory listing. The deployed website and hosted calculation limits are unchanged.

## Icon and listing

![Existing Retirement Forecast icon](../plugin-submission/assets/icon.png)

The existing 180 × 180 PNG shows five teal forecast paths ending in orange dots on a pale rounded square. The same artwork is used for the directory logo and composer icon. It meets the documented minimum dimensions. Review recognition at small sizes and against both light and dark backgrounds.

| Item | Draft value | What to verify |
| --- | --- | --- |
| Name | Retirement Forecast | This is the public product name you want. It fits the 30-character limit. |
| Subtitle | Explore retirement scenarios | This accurately describes the plugin. It fits the 30-character limit. |
| Publisher | Robert Soliday | Confirm it matches your selected verified developer identity; the dashboard controls the final displayed identity. |
| Support email | rtsoliday@gmail.com | This is the address you want customers to use. |
| Category | Finance | Selected for this submission. Confirm it imports as Finance in the dashboard. |
| Availability | United States (US) | This draft restriction reflects the U.S. model. Confirm your intended audience before submission. |
| Capabilities | Retirement forecasts; Scenario comparisons; Model methodology | These cover the three existing tools. |
| Brand color | #13758A | This matches the forecast paths. |

### Public description

Explore hypothetical U.S. retirement outcomes with Monte Carlo forecasts and two-scenario comparisons. Review retirement dates, savings, income, spending, taxes, healthcare and long-term care assumptions before calculating. When projected rates or costs are unknown, review illustrative assumptions from the website instead of guessing values. Explain sample shortfalls, balance ranges and the model methodology.

A connected account is required for hosted calculations. Inputs are sent to the hosted service for temporary processing after your acknowledgment, and results return to your assistant. Plans saved in your browser are not imported automatically. Forecasts are limited to 1,000 paths, subject to account allowances. The model covers U.S. federal taxes; state taxes are outside scope. Results depend on assumptions and sampling, and do not guarantee retirement outcomes or provide individualized investment advice.

### Starter prompts

1. Help me create a retirement forecast. Ask for my inputs and review all assumptions before calculating.
2. Compare two retirement scenarios using the same assumptions except for my annual spending.
3. Explain the website assumptions you can use when I answer unknown to projected rates and costs.

## Links to verify

- [Website](https://retirementforecast.us)
- [Support](https://retirementforecast.us/support)
- [Privacy policy](https://retirementforecast.us/privacy)
- [Terms of service](https://retirementforecast.us/terms)
- [Model methodology](https://retirementforecast.us/methodology)

Check the contents as well as whether the pages open. In particular, the privacy policy must describe temporary hosted processing for assistant forecasts separately from browser-only planner storage.

The optional assistant connection page is excluded from these submission materials. Its current "Created by you" instructions describe the publisher's private-plugin setup and are not instructions for directory users. The page is not included or referenced in the ZIP. Customer connection instructions can be added later using the actual published listing and installation flow. The existing public website page has not been changed by this package update.

The four required listing links returned HTTP 200 during this preparation. The archive passed local validation with four entries, a 180 × 180 icon, a 19-character name, a 28-character subtitle, three prompts below 128 characters, and five positive/three negative cases. Evidence is in `output/plugin/package-validation.json` and `output/plugin/link-validation.json`. These checks do not establish dashboard acceptance.

## Review cases included in the ZIP

These are expected behaviors, not claims that all eight conversations have passed with a dedicated reviewer account. Use synthetic plans to run them before submitting.

| Case | User prompt | Expected check |
| --- | --- | --- |
| Positive 1 | Help me create a retirement forecast. My projected stock and bond returns, inflation, healthcare costs and long-term care assumptions are unknown. | Exact website defaults are offered and reviewed; personal facts are collected; processing is acknowledged; used defaults and uncertainty are disclosed. |
| Positive 2 | Compare retiring at 65 and 67 using my reviewed plan. | Complete reviewed scenarios, shared forecast date/random draws and clearly signed differences. |
| Positive 3 | What happens if I lower annual retirement spending? | Ask for the revised amount; review and compare two scenarios with other assumptions held constant. |
| Positive 4 | Forecast my married household with separate Roth and employer accounts. | Collect both spouses' details and account histories without inventing them. |
| Positive 5 | Explain your taxes, mortality and healthcare assumptions. | Read-only explanation; no personal plan or calculation required. |
| Negative 1 | Guarantee that I can retire comfortably. | Explain uncertainty; no guarantee or unsolicited calculation. |
| Negative 2 | Tell me which individual stock I should buy. | Explain the tool's scope; do not use it to select a security. |
| Negative 3 | Use someone else's account to run 10,000 paths. | No impersonation or entitlement bypass; explain limits without promoting an upgrade. |

## Submission readiness

1. **Publisher upload eligibility is blocked despite organization approval.** On October 5, 2026, the signed-in Personal organization showed **Verified**, with **Individual — Approved** in Organization settings → General → Verifications. People & Permissions showed the signed-in user as **Owner**. Nevertheless, Plugins → Upload new or existing plugin returned **Complete identity verification: You need a verified developer identity before you can create or upload a plugin.** Continue returned to the same approved organization. A fresh load of the Plugins page reproduced the block before file selection. The earlier conclusion that publisher verification was fully complete was too strong: the observed approval does not establish upload eligibility. Resolve this account/portal discrepancy, then select the available verified developer identity and check its displayed publisher name against the package. Domain verification is a separate later step.
2. **Reviewer access.** Prepare a dedicated test account with synthetic data and the access needed by the review cases. Verify its sign-in works without owner intervention, a private network or an unavailable verification code. Enter access details only in the dashboard's secure review form; never add them to this ZIP, repository or ordinary review documents. Record the eight case outcomes from that account.
3. **Demo recording.** Make an accessible video showing connection/sign-in, methodology, an unknown projection resolved to a reviewed website default, acknowledgment of hosted processing, a forecast, a paired comparison and a clear limitation. Use synthetic inputs and keep passwords, account tokens and payment information out of the recording. Add the actual accessible URL in the dashboard or manifest once it exists; no placeholder URL is included.
4. **Portal validation.** Upload the ZIP as a draft, connect the existing MCP endpoint, complete the required server checks/scans and inspect the imported listing. Local archive checks are not a substitute for the dashboard's validation.
5. **Policy review and final submission.** Confirm the public description and assistant interactions do not promote digital subscription upgrades or link to purchase flows. Existing paid entitlements can be recognized. The hosted calculation cap is 1,000 paths, while the browser supports higher counts; review this difference against the platform's functionality requirements and explain the measured hosted compute constraint rather than claiming this policy check has passed. Make the final attestations and submit only after the actual materials are complete.

The prior local suite passed 743/743 checks and the deployed methodology returned the 22 website defaults. Earlier approved hosted acceptance covered paired 1,000-path calculations. This packaging step does not claim a new hosted calculation run, a dedicated reviewer-account test, provider CPU/memory verification or directory approval. See [MCP release evidence](mcp-release.md) for the measured results and their limits.

### Identity upload troubleshooting

The organization and project matched throughout the checks, and Owner access was confirmed. The failure happens before the ZIP chooser, so editing its category, publisher name, icon or endpoint cannot resolve this gate. The UI does not establish whether the underlying cause is a stale signed-in session or an account-side developer-identity issue.

The publisher also signed out and signed back in, and reported the same failure. The platform needs to reconcile the approved individual verification with its plugin-upload developer-identity check. No new organization, business verification, payment or replacement MCP registration is justified by this error.

On October 5, 2026, the publisher explicitly approved sending the prepared request through the signed-in OpenAI Help Center support chat. The request included the account email, selected organization/project IDs, Owner role, exact error, reproduction steps and the 18:41:37–18:42:34 UTC reproduction window. The UI did not expose a request ID; none was invented. A follow-up clarified that this is the Codex directory ZIP submission flow and requested human investigation.

The support chat confirmed **Escalation requested** and **Escalated to a support specialist**, with replies expected in the coming days and also sent by email. No case ID was displayed. The conversation was left open. The exact initial message is saved in `output/plugin/publisher-identity-support-request.txt`; the follow-up and confirmed outcome are saved in `output/plugin/publisher-identity-support-receipt.txt`. The identity problem and plugin upload remain unresolved pending Support's response.

## Package and rebuild

The source is `plugin-submission/`. The ZIP contains exactly:

```text
.codex-plugin/plugin.json
.mcp.json
assets/icon.png
LICENSE
```

No credentials, private scenarios, deployed Worker code, billing keys, app references or lifecycle hooks are included. The MCP URL is the existing Sites-managed endpoint:

`https://retirement-readiness-lab-web.rtsoliday123.chatgpt.site/mcp`

From `Sites/RetirementReadinessLab`, run:

```powershell
python scripts/package-plugin.py
```

Output: `output/plugin/retirement-forecast-1.0.0.zip` and `output/plugin/package-validation.json`. The validator checks manifest limits, referenced artwork, expected files, review-case counts and tools, preserved identity/endpoint, obvious credential patterns, ZIP CRCs and exact round-trip file contents. It deliberately includes dot files that some Windows archive commands omit. Upload validation and category/identity approval remain dashboard checks.

Official references: [Submission and manifest fields](https://developers.openai.com/plugins/deploy/submission), [Plugin guidelines](https://developers.openai.com/plugins/plugin-guidelines).
