# Module selection and migration ledger

## Select by dependency, not directory order

Build a lightweight dependency map from imports, call sites, test fixtures, configuration, and runtime initialization. Prefer a small leaf capability that:

- has a stable observable contract;
- has representative tests or can be characterized locally;
- unlocks downstream modules;
- does not require inventing several unfinished abstractions.

Qlib's top-level areas currently include `backtest`, `cli`, `contrib`, `data`, `model`, `rl`, `strategy`, `utils`, and `workflow`, plus root configuration/logging/type helpers. These names are inventory, not a mandated order. `contrib` and `rl` are large enough that they should normally be split into smaller slices.

Likely foundation candidates include constants and value types, error/config primitives, pure utilities, expression primitives, calendar/instrument abstractions, and storage interfaces. Confirm their actual dependency direction before choosing one.

## Slice definition

A slice entry must state:

- Python source files and relevant callers;
- target Rust crate/module and any binding surface;
- observable contract and exclusions;
- upstream, characterization, differential, and Rust tests;
- the tests that cover each success, boundary, failure, and conditional path;
- dependency blockers;
- numeric, ordering, concurrency, serialization, and performance expectations;
- selected third-party crates, required features, and any capability intentionally implemented locally;
- status: `planned`, `in_progress`, `blocked`, or `done`.

Use one `in_progress` slice at a time. Split a slice if its acceptance criteria cannot be reviewed or verified as a coherent unit.

## Ledger format

Maintain `docs/migration-status.md` in the Rust repository. Keep a summary table followed by evidence for active and completed slices:

```markdown
| Slice | Python surface | Rust target | Status | Evidence | Blockers |
|---|---|---|---|---|---|
| constants | qlib/constant.py | crates/core | done | cargo test -p core | - |
```

For each slice, record:

```markdown
## <slice name>

- Contract: ...
- In scope: ...
- Out of scope: ...
- Compatibility decisions: ...
- Dependencies: crate, features, purpose, and rejected alternatives where relevant
- Verification commands: ...
- Results: ...
- Coverage: exact line/function/region/branch percentages and report path
- Known gaps: ...
- Next candidate: ...
```

Commands must be reproducible from a stated working directory. Do not erase failed evidence; summarize the failure and its resolution when it affected design.

## Acceptance gate

A slice is `done` only when all applicable items hold:

- Rust code formats and compiles with warnings treated according to repository policy.
- Focused unit/integration tests pass.
- Project-owned production Rust code reaches exactly 100% per-file line, total line, function, region, and branch coverage with no uncovered items.
- Relevant upstream Python behavior is represented by tests or fixtures.
- Differential comparisons pass for normal, boundary, and failure cases.
- Public shape, ordering, dtype/schema, errors, and serialization are accounted for.
- Unsafe code is absent or narrowly documented with safety invariants and tests.
- General-purpose functionality uses suitable third-party crates; any local replacement is justified in the ledger.
- Dependency features are minimal, licenses and supported targets are acceptable, and `cargo tree` shows no unexplained duplicate major versions.
- Performance-sensitive code has no obvious regression; claimed improvements have a reproducible benchmark.
- Documentation and the migration ledger reflect limitations and deliberate incompatibilities.

If a check is not applicable, state why instead of silently omitting it.

## Suggested command families

Adapt these to the actual workspace and available tooling:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo tree --workspace --duplicates
cargo llvm-cov --workspace --all-features --fail-under-lines 100 --fail-under-file-lines 100 --fail-under-functions 100 --fail-under-regions 100 --show-missing-lines
cargo +nightly llvm-cov --workspace --all-features --branch --show-missing-lines
python -m pytest <focused upstream tests>
```

The stable coverage command is a hard machine-enforced gate. Because `cargo-llvm-cov` branch mode is unstable and does not provide an equivalent branch threshold flag, inspect or machine-parse its report in CI and fail when any branch is uncovered. Save a machine-readable and HTML report as CI artifacts when the project setup supports it.

Prefer focused checks during iteration, then run the workspace-level gate before closing a slice. Do not install dependencies, download large datasets, or invoke network services without checking existing project setup and scope.
