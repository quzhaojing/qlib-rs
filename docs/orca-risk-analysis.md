# Orca risk-analysis slice

## Scope

- Upstream surface: `qlib/contrib/evaluate.py::risk_analysis`, including its nested annualization-scaler closure.
- Rust target: `crates/core/src/risk_analysis.rs` and its public exports.
- In scope: sum and product accumulation, `N`/`freq` precedence, ordered warnings, the existing `Frequency` parser, empty-product positional failure, missing and non-finite floats, returns at and below `-1`, sample (`ddof=1`) deviation, ordered result fields, and floating-point precision.
- Out of scope: Pandas indexes and `DataFrame` storage, Python bindings, `indicator_analysis`, portfolio metrics, calendars, and any generic DataFrame replacement.

## Boundary and compatibility decisions

The Rust API accepts a float slice, where `NaN` represents a missing Pandas value. It returns a typed five-field result plus ordered warnings; `RISK_ANALYSIS_FIELDS` and `ordered_values` preserve the Pandas row order `mean`, `std`, `annualized_return`, `information_ratio`, `max_drawdown`. Product exponents use the full input length even when reductions skip `NaN`, matching the source. Product mode reads the final cumulative position, so a trailing `NaN` remains observable and an empty input returns the source `IndexError` message.

Frequency parsing reuses `domain_core::Frequency`; the compatibility layer only maps its error text to the source function's public message and applies the closure's scalers (57,120 minute periods, 238 days, 50 weeks, or 12 months divided by the parsed count). A supplied `N` takes precedence without parsing `freq` and emits `risk_analysis freq will be ignored`. No dependency, manifest change, production Python bridge, unsafe code, or coverage suppression was added.

## Source-pinned evidence

`crates/core/tests/fixtures/risk_analysis_contract.py` parses the checked-out upstream files and executes the actual `Freq` class and `risk_analysis` function AST nodes. Its matrix covers both modes, all four frequency units, explicit-scaler precedence over an invalid frequency, invalid/missing/zero frequency failures, invalid mode, empty sum and product inputs, singleton/all-missing series, a string-indexed product series (confirming positional final access), middle and trailing missing values, positive and mixed infinities, and returns equal to and below `-1`. The Rust differential compares schema and column names, values with an eight-ULP-scale bound (plus exact `NaN`/infinity classification), warning categories/order/text, and error types/messages.

## Verification record

- `python crates/core/tests/fixtures/risk_analysis_contract.py D:/code/github/qlib` — passed; generated 19 actual-source cases.
- `CARGO_TARGET_DIR=C:/Users/andy/.codex/builds/qlib-rs/risk-analysis-test CARGO_BUILD_JOBS=2 cargo test -p core --test risk_analysis` — passed, 3 tests.
- Focused strict Clippy reached this slice and identified three local findings; those were fixed. The same run was blocked by concurrent, out-of-scope findings in `alpha_returns.rs`, `data_normalization.rs`, `model_sequence.rs`, and `trading_indicator_analysis.rs`, so no strict-Clippy pass is claimed yet.
- Stable and nightly branch coverage remain pending the coordinator's shared executable-lock gate. Exact percentages must be added here only from the raw reports; earlier workspace coverage is not evidence for this slice.

## Remaining verification

Run strict focused Clippy again after concurrent owners settle, then run stable and nightly branch `cargo llvm-cov` for the unique `risk_analysis` integration target under coordinator approval. Inspect raw JSON totals for this production file and require exactly 100% lines, functions, regions, and branches with no exclusions. This function slice does not establish acceptance for the rest of `evaluate.py`.
