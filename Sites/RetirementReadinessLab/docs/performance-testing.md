# Simulation performance and result preservation

The October 3, 2026 optimization keeps financial calculations, the Java-compatible random stream, Gaussian caching, draw order, and all search iteration counts unchanged. It advances the 48-bit random state using two exact 24-bit integer words, builds withdrawal records only for the final search result, and skips Roth ledger queries when no Roth money is drawn. Intermediate random-state products and sums are below 2^49, within JavaScript's exact integer range.

The complete Node suite passes 601 tests. New checks compare one million random-state transitions plus mixed uniform/Gaussian/zero-volatility draws against the original BigInt implementation. Eight fixed scenarios pin full result objects and monthly balance, cash-flow, owner-account and tax traces to the original source commit recorded in `tests/fixtures/simulation-results-sha256.json`. Only the wall-clock generation timestamp is omitted. They cover pooled and separate accounts, preview mortality bands and random lifespans, shortfalls, early withdrawals, conversions, SEPP, working spouses, already-retired plans, and employer Roth conversions/rollovers. Additional before/after checks matched 9,000 account quotes and two retirement/spending target searches exactly.

## Repeated local measurements

Measured on macOS arm64 with Node.js 22.22.2 against unchanged commit `c8c34b37b38f1ee3fab36f03e242172e4b97c797`. Each case has two warmups per implementation followed by five alternating before/after measurements (three for the 10,000-path case). Times include the full result summary, six risk sensitivity checks, today's-dollar results, path points and steady illustration. Every warmup and measured result is compared in full, excluding the generation timestamp.

| Scenario | Paths | Before median | After median | Less time |
| --- | ---: | ---: | ---: | ---: |
| Pooled accounts | 1,000 | 1,021 ms | 809 ms | 21% |
| Separate-account couple | 1,000 | 5,925 ms | 4,301 ms | 27% |
| Employer Roth accounts | 1,000 | 14,167 ms | 9,794 ms | 31% |
| Pooled accounts | 10,000 | 7,242 ms | 5,616 ms | 22% |

These are local Node measurements. Browser and device timings may differ; no browser performance measurement or native Android/Python optimization is implied.

## Reproduce

From the repository root:

```sh
npm test --prefix Sites/RetirementReadinessLab
npm run benchmark:simulation --prefix Sites/RetirementReadinessLab -- --paths=1000
npm run benchmark:simulation --prefix Sites/RetirementReadinessLab -- --paths=10000 --repeats=3 --scenario=pooled
```

For a comparison, preserve the previous engine and all its sibling JavaScript modules in a separate directory before changing them. Pass `--baseline=/absolute/path/to/previous/dist/engine.js` to the benchmark. That directory must be an ES module package (its parent or the directory itself needs a `package.json` containing `"type": "module"`). The default benchmark measures pooled, separate-couple and employer-Roth cases; `--scenario` can select any name in `tests/fixtures/simulation-scenarios.js`.

The benchmark has no timing pass/fail threshold because machine load and runtime optimization affect timings. Result equality is mandatory. Intentional future financial model changes require reviewing and regenerating the pinned result hashes; performance-only changes must preserve them.
