# Frozen calculation references

These modules are the complete transitive calculation dependencies of two original Git commits. Each manifest records the commit and SHA-256 of every unchanged source module. Preserve these files when changing the current engine; a reference must never import current calculation code. The integrity checks catch accidental edits.

`pre-optimization` comes from `c8c34b37b38f1ee3fab36f03e242172e4b97c797`. `pre-household` comes from `c9fb1c538f8c46abd1262b9b10e8e700c9fc7f88`. The fixed simulation inputs are in `fixtures/simulation-reference-inputs.json`; old backups remain in `fixtures/pre-household-results.json`.

Tests compare entire results and monthly account/tax traces exactly on the same supported Node runtime, excluding only the historical test's documented timestamp and newer compatibility fields. Assertion failures identify the first field and include Node, V8, OS and architecture. Random state transitions, draws, probabilities, balances and tax traces retain exact checks. The old output hashes remain as historical evidence, but are not portable assertions: the unmodified original engines also disagree with them on Node 24, while producing exactly the current engine's outputs. No financial-output snapshots were regenerated from the current engine.
