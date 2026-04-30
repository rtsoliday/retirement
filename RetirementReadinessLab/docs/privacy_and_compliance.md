# Privacy And Compliance

## Privacy Position

Retirement Readiness Lab should be local-first. The MVP should not require account creation, brokerage linking, cloud sync, or analytics that transmit financial inputs.

## Data Classes Handled

The app may store:

- Age and retirement age.
- Filing status.
- Account balances.
- Spending estimates.
- Mortgage assumptions.
- Healthcare assumptions.
- Social Security estimates.
- Scenario results.

These values are sensitive financial planning data even if they are manually entered.

## MVP Requirements

- Store scenario data locally.
- Provide delete controls.
- Export only after explicit user action.
- Do not transmit financial inputs.
- Provide a privacy policy before release.
- Provide an educational-use disclaimer.
- Avoid personalized investment recommendations.

## Implemented In-App Controls

- The Reports screen includes a Privacy and disclosures card.
- The Reports screen has explicit PDF, text, JSON backup, and CSV share actions.
- The Reports screen includes a two-step local data deletion flow.
- Generated text/PDF reports include a privacy note and educational-use disclaimer.
- Assumptions and Reports use the same centralized disclaimer text from the Android app source.

## Suggested Disclaimer

Retirement Readiness Lab provides educational estimates based on user-entered assumptions. It is not financial, tax, legal, or investment advice. Results are not guarantees and should be reviewed with qualified professionals before making retirement decisions.

## Release Checklist

- Privacy policy.
- Data safety declaration.
- Financial features declaration.
- App store listing language reviewed for advice/guarantee issues.
- Clear in-app data deletion path.
