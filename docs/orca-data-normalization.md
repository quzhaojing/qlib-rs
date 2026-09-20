# Orca worker report: data normalization

## Scope

- Python source: `D:\code\github\qlib\qlib\utils\data.py`, SHA-256
  `3299196f937bf970b2d7dceef8813a7cfcbdbc90acf0667cf72dcdc4fd12a102` during characterization.
- Migrated symbols: `robust_zscore` and `zscore` only.
- Rust target: `crates/core/src/data_normalization.rs`.
- Explicitly excluded: `deepcopy_basic_type`, `update_config`, `guess_horizon`, and the unrelated
  dataframe-append implementation.

## Contract and design

`NumericSeries` and `NumericFrame` retain row labels, row-label names, Series names, duplicate
column labels, and column order. `NumericColumn` pairs Arrow storage with explicit native or
nullable Pandas dtype identity so that a masked integer cannot be confused with a NumPy integer
and a native `NaN` cannot be confused with an Arrow null.

The implementation accepts signed and unsigned 8/16/32/64-bit integers and native
`float16`/`float32`/`float64`; nullable extension inputs accept the same integers plus `Float32`
and `Float64`. Native integer results and nullable integer results promote to float64 and
nullable `Float64`, respectively. Native floating Series retain width. Native columns in a mixed
DataFrame use Pandas' common arithmetic width, while nullable extension columns retain their
independent extension identity.

Both operations are column-wise. Ordinary z-score uses missing-skipping mean and sample standard
deviation (`ddof=1`). Robust normalization subtracts the missing-skipping median, divides
sequentially by the median absolute deviation and `1.4826`, clips to `[-3, 3]`, and optionally
applies ordinary z-score afterward. Empty, singleton, constant, all-missing, native-NaN, positive
and negative infinity inputs preserve observed Pandas results. Inputs are borrowed and returned
labels/storage are independently owned or reference-counted, so normalization does not mutate the
input.

The reducers deliberately mirror the actual source stack rather than a textbook formula. Missing
positions are replaced with zero before NumPy-style pairwise mean summation, preserving their
position in reduction chunks. Variance follows pandas `nanvar`: a two-pass computation with a
float64 accumulator, `ddof=1`, followed by a cast back to the source floating precision. The
source-pinned differential includes 301-element cancellation vectors, long vectors with interleaved
NaNs, tiny and near-`float32`-limit magnitudes, mixed native DataFrame promotion, and nullable
missing values.

No dependency or manifest change and no production Python bridge were introduced. Existing Arrow,
`half`, and `thiserror` dependencies provide storage, float16 conversion, and typed failures.

## Evidence

The fixture `crates/core/tests/fixtures/data_normalization.py` extracts only the two authoritative
functions from the actual source AST and executes them with the upstream repository's environment.
It records values, dtypes, labels, input preservation, warnings, and representative source errors.
The integration test compares Rust results against that live output and separately exercises every
supported integer width and all typed validation failures.

Commands run from `D:\code\github\qlib-rs`, with `CARGO_BUILD_JOBS=2` and the isolated target
`C:\Users\andy\.codex\builds\qlib-rs-data-normalization`:

```text
cargo fmt -- crates/core/src/data_normalization.rs crates/core/tests/data_normalization.rs
cargo test -p core --test data_normalization
cargo clippy -p core --test data_normalization -- -D warnings
```

Latest test result: 6 passed, 0 failed. The initial five-case differential also passed before the
arithmetic review; the expanded six-test suite passes after the pairwise/two-pass correction.
Focused strict Clippy passed before a concurrent owner changed `alpha_returns.rs`. Its latest retry
is blocked only by compile errors in that coordinator-owned/other-worker file; no normalization
diagnostic remains, and this worker did not edit the unrelated file.

Stable and nightly branch coverage were approved but remain pending until the shared core crate
again compiles. Exact coverage percentages must not be inferred from the passing tests.

## Known gaps

- The fixture freezes NumPy/Pandas `RuntimeWarning` behavior for infinity reductions, but the Rust
  convenience API does not currently expose a warning sink. Values and dtypes match; warning parity
  remains unverified production behavior, not an approved incompatibility.
- Boolean, complex, decimal, object/string, datetime/timedelta, categorical, sparse, Arrow-backed,
  and third-party extension dtypes are outside the explicit accepted Rust boundary. Some are
  accepted by upstream Pandas and therefore remain migration gaps, not approved incompatibilities.
- Arbitrary non-string/MultiIndex label metadata is retained as Arrow storage, but this slice does
  not reconstruct every Pandas index subclass identity.
- The differential covers the actual upstream Windows environment and pinned source, not all
  historical or future NumPy/Pandas reducer versions.
- This function slice does not establish whole-file acceptance for `qlib/utils/data.py`.
