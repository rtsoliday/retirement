# Closed Testing Plan

## Goal

Validate that beta users can complete the core retirement planning workflow without assistance:

1. Launch the app.
2. Complete setup or use sample plans.
3. Run a stress test.
4. Change at least one assumption.
5. Confirm Pro-locked Lab tools are visible in Free.
6. Use a Lab tool in a Pro-unlocked test build.
7. Review assumptions.
8. Confirm report sharing is Pro-locked in Free.
9. Delete local data or restore sample plans.

## Test Group

Target 8-15 testers for the first closed test:

- 3-5 people aged 45-65 who are actively thinking about retirement.
- 2-4 DIY investors comfortable with retirement accounts.
- 1-2 people who are less technical, to test onboarding clarity.
- 1-2 reviewers focused on accessibility or larger text settings.

## Devices

Cover at least:

- Small phone.
- Large phone.
- Tablet or foldable-width emulator.
- Android 12+ device for splash/icon behavior.
- One lower-end or mid-range device profile for performance.

## Required Test Scenarios

### First Run

- Fresh install opens to the welcome flow.
- Start Setup leads to editable assumptions.
- Use Sample Plans leads to Dashboard.

### Setup And Dashboard

- Change retirement age.
- Change annual spending.
- Change account balances.
- Apply changes and confirm Dashboard readiness changes.
- Confirm running state is visible.

### Lab

- In Free, confirm retirement age, spending, Social Security, and strategy tools are visible but locked.
- In a Pro-unlocked test build, review retirement age sweep.
- In a Pro-unlocked test build, review spending sweep.
- In a Pro-unlocked test build, review Social Security timing.
- In a Pro-unlocked test build, review strategy comparison cards.
- Confirm each takeaway is understandable.

### Assumptions

- Edit an advanced assumption.
- Trigger a validation error with an invalid value.
- Confirm assumption warnings are visible for suspicious inputs.
- Confirm calculation provenance is visible in Results.

### Reports

- In Free, confirm PDF and text report actions are visible but locked.
- In a Pro-unlocked test build, share PDF report.
- In a Pro-unlocked test build, share text report.
- Confirm JSON backup/import and CSV comparison export are not visible.
- Delete local data and confirm sample data reloads.

## Feedback Questions

Ask testers:

- What did you think the readiness percentage meant?
- Which screen made the app feel most useful?
- Which input was confusing or hard to estimate?
- Did any result feel overconfident?
- Did you understand that exports may contain sensitive assumptions?
- Did the app feel fast enough when running tests?
- Did text fit at your preferred font size?
- What would stop you from trusting the result?

## Exit Criteria

Closed testing is ready to expand when:

- No crash blocks the core workflow.
- New testers can get a first result in under five minutes.
- Free/Pro locks, report sharing, and delete-local-data controls work.
- No screen has obvious text overlap at large font settings.
- Reports include assumptions, provenance, privacy note, and disclaimer.
- Testers understand the app is educational and not advice.
