# Release Signing

Do not commit signing keys, keystores, or signing passwords.

## Current Build Setup

The Android project now has explicit `debug` and `release` build types:

- `debug` uses the application id suffix `.debug` and version suffix `-debug`.
- `release` disables debugging and points at `app/proguard-rules.pro`.
- Minification and resource shrinking are currently disabled for the MVP release candidate.

The release build is intentionally not wired to a committed signing key.

## Build Commands

From `RetirementReadinessLab/android`:

```bash
./gradlew :app:assembleDebug
./gradlew :app:assembleRelease
```

The debug build can be installed directly from Android Studio. The release APK produced by `assembleRelease` is suitable for compile/package verification, but Play distribution should use a properly signed release artifact.

## Signing Secrets

The project ignores these files:

- `*.jks`
- `*.keystore`
- `keystore.properties`
- `release-signing.properties`

Place real signing material outside source control. A future signing setup should read credentials from local Gradle properties, environment variables, or CI secrets.

## Before Closed Testing

- Decide whether to use Play App Signing.
- Create or select the upload key.
- Store the keystore outside the repository.
- Configure signing only through local/CI secrets.
- Build and install a release candidate.
- Confirm the package name, app icon, splash screen, and version number.
