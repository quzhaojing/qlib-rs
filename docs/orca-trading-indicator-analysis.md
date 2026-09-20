# Orca worker report: trading indicator analysis

- Scope: `qlib/contrib/evaluate.py::indicator_analysis` only.
- Rust boundary: `TradingIndicatorTable` keeps an explicit datetime index aligned with an Arrow `RecordBatch`, accepting Float64 and Int64 columns with Arrow null masks; `IndicatorAnalysis` returns fixed `ffr`, `pa`, `pos` labels and one Float64 `value` column.
- Compatibility: all three weight columns are accessed before method validation; amount/value weights are absolute; sums skip NaN/null values; division retains IEEE NaN/infinity; `pos` always uses raw count weights; negative counts and input row/index order are preserved.
- Differential evidence: `tests/fixtures/trading_indicator_analysis_contract.py` extracts and executes the authoritative function AST and covers all methods, integer and nullable extension columns, cancellation order, negative/zero weights, NaN, infinity, empty input, output schema/order, and missing-column/error-order cases.
- Dependencies: existing Arrow, Chrono, Serde, and Thiserror only; no manifest change and no production Python bridge.
- Exclusions: `risk_analysis`, `report_indicator`, `order_indicator`, pandas object/string columns, numeric Arrow types other than Float64/Int64, arbitrary non-datetime pandas index types, and Python bindings. The datetime index is retained solely to prove row alignment and does not affect arithmetic.
- Verification from `D:\code\github\qlib-rs` with `CARGO_BUILD_JOBS=2` and C:-hosted target directories:
  - `cargo test -p core --test trading_indicator_analysis`: 3/3 tests pass; the differential test executes 21 actual-source cases.
  - `cargo clippy -p core --test trading_indicator_analysis -- -D warnings`: passed before a concurrent `alpha_returns.rs` edit; the latest retry is externally blocked by compile errors in that owner module, not this slice.
  - Stable and nightly branch coverage: approved after the parity review, pending a compilable shared library boundary.
