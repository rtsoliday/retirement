# Release shrinker rules for Retirement Readiness Lab.
#
# Minification is intentionally disabled for the current MVP release candidate.
# Keep this file in place so release builds have an explicit, reviewed location
# for rules when minification/resource shrinking are enabled later.

# Report and backup exports rely on Android framework APIs and local model code.
# No custom keep rules are required at this stage.
