---
name: qlib-rust-refactor
description: Incrementally migrate the Python Microsoft Qlib codebase into this qlib-rs repository, one dependency-aware module at a time, while preserving observable behavior through characterization, differential, and Rust tests. Use for planning, implementing, reviewing, or continuing this migration; do not use for unrelated Rust rewrites or ordinary Qlib usage.
---

# Qlib Rust Refactor

Migrate Qlib as a sequence of small, independently verified vertical slices. Preserve behavior before optimizing it. Never claim whole-module parity from compilation alone.

## Repository defaults

- Rust target: the repository containing this `.agents/skills/qlib-rust-refactor` directory. Resolve its root before running commands.
- Python source: the sibling `qlib` repository (`../qlib` relative to the Rust repository root). Verify it exists; if absent, ask for the source checkout path rather than creating or guessing one.
- Treat paths supplied by the user as authoritative when they differ.
- Read repository instructions such as `AGENTS.md`, `CONTRIBUTING.md`, and build configuration before editing.
- Preserve unrelated user changes in both repositories. Do not modify the Python source except for explicitly authorized fixtures or compatibility harnesses.

## Package and directory naming

- The adopted naming convention is that all project-owned Rust package names and their corresponding directories omit the `qlib` prefix (including `qlib-` and `qlib_`), including future packages. Use `crates/<package-name>` and keep package names and directory names aligned.
- Current mapping: package `core` at `crates/core` uses library/import name `domain_core`; package `shmem` at `crates/shmem` uses `shmem`; package `locale` at `crates/locale` uses `locale`. Keep `[lib] name = "domain_core"` for `core` to distinguish it from Rust's standard library `core`; Cargo commands still select it with `-p core`.
- Keep workspace members, dependency names and paths, library imports, scripts, tests, and current documentation consistent with these names. Do not reintroduce the old prefixed names in new code or examples. Historical commands and coverage paths in the migration ledger may retain the names actually used by those runs.
- This convention applies to project-owned Rust packages and package directories, not the `qlib-rs` repository name, the `qlib-rust-refactor` skill name, upstream Python `qlib` imports, or third-party package names.

## Start or resume

1. Inspect both worktrees and `docs/migration-status.md` in the Rust repository.
2. If the Rust project or ledger does not exist, bootstrap the minimal Cargo workspace and ledger needed for the first slice; do not pre-create speculative crates.
3. Resume the sole `in_progress` slice. If none exists, select the next unblocked slice from the ledger. If no ledger exists, inventory dependencies and propose the smallest leaf slice.
4. Work on exactly one slice unless the user explicitly requests a broader batch. A slice may be smaller than a Python package when the package is large or tightly coupled.

Read [references/module-workflow.md](references/module-workflow.md) before selecting or implementing a slice. Read [references/rust-dependencies.md](references/rust-dependencies.md) when designing or reviewing a Rust implementation.

## Required migration loop

For the selected slice:

1. **Bound the contract.** Identify public imports, call signatures, accepted data shapes and dtypes, ordering, defaults, errors, serialization, side effects, and performance-sensitive behavior. Trace callers and dependencies with code search.
2. **Freeze current behavior.** Reuse upstream tests and add focused characterization fixtures or a differential harness for important cases, including empty, boundary, malformed, missing-value, timezone, and nondeterministic inputs when relevant. Record intentional incompatibilities before implementation.
3. **Design the Rust boundary.** Choose ownership, error, trait, concurrency, and Python-interoperability boundaries deliberately. Keep compatibility adapters at edges; avoid transliterating dynamic Python internals into unidiomatic Rust. Inventory mature third-party crates before proposing custom infrastructure.
4. **Implement the smallest complete slice.** Prefer well-maintained third-party crates for general-purpose capabilities and write Qlib-specific code only for domain behavior, compatibility adapters, or demonstrated gaps. Keep public Rust APIs narrow. Separate pure domain logic from Python bindings, storage, network, and process integration. Do not optimize until parity is measurable.
5. **Verify.** Run formatting, static checks, focused Rust tests, relevant upstream Python tests, differential tests, and the 100% coverage gate. Compare values, shapes, ordering, dtypes/schema, errors, and stable serialized forms. Benchmark only performance-critical paths, with a recorded command and baseline.
6. **Close the slice.** Update the ledger with evidence, remaining gaps, commands, and the next dependency-unblocked slice. Mark `done` only when every agreed acceptance criterion passes.

## Compatibility decisions

- Default to behavioral compatibility, not identical internal architecture.
- Preserve public Python compatibility only when the slice or migration plan requires it. Use PyO3/maturin or another binding layer only after identifying an actual Python-facing consumer.
- Match randomness through deterministic seeds and statistical invariants when bit-for-bit equality is unrealistic.
- For floating-point results, define justified absolute/relative tolerances and special-value handling; never silently use a broad tolerance.
- Preserve stable ordering explicitly; do not rely on hash-map iteration.
- Treat configuration, calendars, instruments, expressions, storage formats, and dataset schemas as contracts, not implementation details.
- Any deliberate behavior change must be listed in the ledger with rationale and user approval when it changes public behavior.

## Dependency policy

- Default to reuse: do not reimplement established functionality such as serialization, configuration parsing, logging, time zones, Arrow/Parquet I/O, DataFrame operations, async I/O, caches, database clients, Python bindings, plugin loading, numerical primitives, or ML runtimes when a suitable maintained crate exists.
- Prefer crates with active maintenance, compatible licenses, supported target platforms, acceptable MSRV and build footprint, documented safety posture, and evidence of ecosystem adoption. Verify current official documentation before committing to a dependency whose API or maintenance status may have changed.
- Keep Qlib domain contracts in project-owned crates. Do not expose volatile third-party types across stable plugin interfaces; use Arrow schemas, owned DTOs, or explicitly versioned wire types at boundaries.
- Centralize dependency versions and feature flags in the workspace manifest. Enable only required features and inspect duplicate or unexpectedly large dependencies with `cargo tree`.
- Wrap replaceable infrastructure behind narrow project traits when its semantics affect Qlib compatibility. A wrapper is not permission to duplicate the dependency's implementation.
- Implement functionality locally only when no suitable crate exists, upstream behavior requires a small compatibility-specific algorithm, or measurements show an unacceptable constraint. Record that decision and the rejected alternatives in the migration ledger.

## Test and coverage policy

- Every completed slice must have 100% coverage for project-owned production Rust code: lines, functions, regions, and branches. A slice below the threshold remains `in_progress` or `blocked`; never round a percentage up.
- Use `cargo-llvm-cov` as the coverage source. Enforce 100% total and per-file line coverage plus 100% function and region coverage. Run its branch mode on a compatible nightly toolchain and require zero uncovered branches; branch coverage is not optional merely because its tooling is unstable.
- Exclude only third-party dependencies and generated artifacts that are not maintained as project source. Do not exclude handwritten production modules, FFI adapters, error paths, platform branches, or difficult code. Record every coverage exclusion and its reason in the ledger.
- Do not use `cfg(coverage)`, coverage-off annotations, test-only behavior changes, dead-code removal solely to hide valid behavior, or assertion-free tests to manufacture 100%. Refactor hard-to-test code behind narrow interfaces and test the observable behavior.
- Coverage does not replace correctness. Each slice must still include meaningful unit/integration tests, upstream characterization or differential tests, boundary and failure cases, and property tests for invariant-rich code where applicable.
- Exercise both outcomes of conditions, every error variant, empty and boundary inputs, serialization round trips, deterministic randomness, platform-specific behavior on the relevant CI target, and plugin lifecycle failures when they exist.

## Stop conditions

Stop and report rather than guessing when source behavior is ambiguous in a way that changes the public contract, required test data or services are unavailable, or proceeding requires a new interoperability or architecture choice with broad downstream impact. Leave the slice `blocked` with exact reproduction steps and the smallest decision needed.

## Deliverable for each invocation

Report the slice, files changed, compatibility evidence, commands run and results, exact coverage percentages for lines/functions/regions/branches, known gaps, ledger status, and recommended next slice. If asked only to plan or review, do not mutate code.
