# Release Checklist

## Build And Packaging

- Confirm `applicationId`.
- Confirm version code and version name.
- Configure release signing outside source control.
- Confirm debug builds install as `com.retirementreadinesslab.debug`.
- Confirm release builds package as `com.retirementreadinesslab`.
- Run `./gradlew :app:assembleRelease`.
- Build a release candidate.
- Install the release candidate on a physical device or stable emulator.
- Confirm launcher icon, round icon, and Android 12+ splash icon.

## Functional Checks

- First launch flow works.
- Setup can produce a runnable scenario.
- Dashboard updates after assumptions change.
- Lab analysis runs without blocking the UI.
- Scenario duplicate, select, compare, delete, and restore controls work.
- Reports export PDF, text, CSV, and JSON backup.
- Import rejects invalid JSON with a calm error.
- Delete Local Data reloads sample plans.

## Simulation Checks

- Unit tests pass.
- 1,000-simulation performance profile passes.
- 10,000-simulation manual profile completes on representative hardware.
- Calculation provenance appears in Results and reports.
- Reports include assumption warnings.
- Monthly cashflow model docs match the app.

## Accessibility Checks

- Touch targets are usable.
- Text remains readable at large font settings.
- Charts have text alternatives or nearby readable summaries.
- Important actions are reachable with screen reader navigation.
- Risk states are not communicated by color alone.
- Light and dark mode are both reviewed.

## Store Assets

- App icon.
- Feature graphic.
- Phone screenshots.
- Tablet screenshots if targeting tablets.
- Short description.
- Full description.
- Privacy policy URL.
- Support contact.

## Policy And Compliance

- Privacy policy reviewed.
- Data safety answers match implementation.
- Financial feature declarations reviewed.
- Store copy avoids guarantee and advice language.
- In-app disclaimer is visible.
- Reports include disclaimer and privacy note.
- License decision documented before commercialization.

## Closed Testing Exit

- Beta users can complete the core workflow.
- No blocker crashes remain.
- High-priority feedback is triaged.
- Known limitations are documented.
- Version is ready for broader testing or public release.
