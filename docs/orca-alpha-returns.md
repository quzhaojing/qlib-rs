# Orca alpha-return slice

## Scope and source

- Python surface: `qlib/contrib/eva/alpha.py::calc_long_short_return` only.
- Rust surface: `crates/core/src/alpha_returns.rs`, exported by the coordinator from `domain_core`.
- Source evidence: `crates/core/tests/fixtures/alpha_returns_contract.py` parses the current upstream file, extracts the named `FunctionDef`, compiles that AST node, and executes every differential case against it. It does not import a copied implementation.
- Whole-file status: **not accepted**. `calc_long_short_precision`, `pred_autocorr`, `calc_ic`, `calc_all_ic`, and every other `alpha.py` surface remain outside this slice.

## Typed contract

`AlphaSeries` is an immutable two-level index plus an Arrow `Float64Array`. Index names and corresponding level dtypes must match between prediction and label. Supported level storage is Arrow UTF-8, `Int64`, `UInt64`, and all four Arrow timestamp resolutions, including retained timezone metadata. Null numeric values and IEEE `NaN` both take Pandas floating-missing semantics.

The adapter preserves the observed DataFrame-constructor alignment relevant to this function:

- equal indexes pair positionally, including equal duplicate indexes;
- unequal unique indexes form a sorted outer union;
- a unique series broadcasts onto a compatible duplicated index;
- incompatible non-unique indexes return `NonUniqueAlignment` before calculation.

Grouping drops missing date keys and sorts group keys. Stable ranking preserves aligned row order for ties, puts missing predictions after finite predictions, truncates `len(group) * quantile` toward zero, and treats a negative selection size as an empty selection as Pandas does. A zero-size selection returns `NaN`; means skip missing labels; `dropna=true` removes rows missing either value before group sizing. Output dates retain the selected level's Arrow dtype/timezone and the index name is `date_col`; the long-short series name is absent and the average series name is `label`, matching Pandas. Inputs are borrowed and their Arrow allocations are not changed.

## Differential matrix

The live-AST fixture currently covers 16 cases. Its SHA-256 is `cb8935ea9f6f34f2c4917ea3fa4364662f5835bd534024169eee0e57bb84e3e1`:

- unsorted groups, stable ties, duplicate labels, `NaN`, and input preservation;
- `dropna`, zero and negative selections, and quantiles beyond group length;
- unique outer alignment and duplicate-index broadcasting;
- missing date/instrument labels and incompatible duplicates;
- naive and UTC timestamp dates, signed and unsigned integer labels, and integer date groups;
- cancellation, positive/negative infinity aggregation, a wholly empty source input, non-finite quantile, and missing `date_col` failures.

Rust boundary tests additionally cover shape validation, duplicate/mismatched names, mismatched level dtypes, unsupported Arrow types, both duplicate-broadcast directions, extra-key duplicate failure, every timestamp resolution, null numeric storage, extreme finite quantiles, and immutable shared input buffers.

## Dependencies and design decision

No dependency was added. The implementation uses the workspace's existing Arrow arrays for typed storage, `thiserror` for diagnostics, and `num-traits` for checked numeric conversions. Existing DataFrame append machinery was inventoried, but it models append/constructor identity rather than grouped top/bottom selection; routing this small evaluator through it would require a speculative general DataFrame engine. The slice therefore keeps a local Qlib-specific alignment/ranking adapter and standard collections.

## Verification

Commands run from `D:\code\github\qlib-rs` with `CARGO_TARGET_DIR=C:\Users\andy\.codex\builds\qlib-rs` and `CARGO_BUILD_JOBS=2`:

- `cargo test -p core --test alpha_returns_contract` — passed all 6 focused public-API tests.
- `cargo clippy -p core --test alpha_returns_contract -- -D warnings` — passed with warnings denied.
- `rustfmt --edition 2024 crates/core/src/alpha_returns.rs crates/core/tests/alpha_returns_contract.rs` — passed for owned Rust files.
- `python crates/core/tests/fixtures/alpha_returns_contract.py D:\code\github\qlib\qlib\contrib\eva\alpha.py` — the widened 16-case actual-source fixture completes successfully.
- `cargo llvm-cov -p core --test alpha_returns_contract --json --output-path C:\Users\andy\.codex\builds\qlib-rs-alpha-returns-stable\alpha-returns-stable.json` — passed all 6 tests. Raw per-file audit for `alpha_returns.rs`: 385/385 lines, 50/50 functions, 574/574 regions, and 53/53 instantiations, each exactly 100%. Stable output contains no branch counters and is not branch evidence.
- `cargo +nightly llvm-cov --branch -p core --test alpha_returns_contract --json --output-path C:\Users\andy\.codex\builds\qlib-rs-alpha-returns-nightly\alpha-returns-nightly.json` — passed all 6 tests, but the generated JSON contains zero files. Manual `llvm-cov export` of the recorded executable/profile exposes dependency files only, not `alpha_returns.rs`. Therefore no branch percentage is available or claimed.

Focused stable and nightly branch coverage were authorized for unique C-drive targets. Stable line/function/region acceptance is exact. The mandatory branch gate remains **unaccepted** because the completed nightly tool run emitted no source counters; it is not normalized, inferred, or treated as 0/0. The slice therefore remains `in_progress` for strict acceptance even though its implementation, differential tests, focused tests, formatting, strict Clippy, and stable coverage pass.

## Explicit residual gaps

- Only two-level indexes are accepted. Arbitrary MultiIndex depth, single indexes, floating/object/categorical/period/interval index labels, Python subclasses, and mixed index dtypes are unsupported.
- Timezone metadata is retained, but cross-timezone or cross-resolution index coercion is not implemented; corresponding pred/label levels must already share an Arrow dtype.
- A nonempty input whose rows all have missing date keys (or are all removed by `dropna`) can make upstream return an inconsistent empty DataFrame/Series pair. The typed Rust result does not model that dynamic shape and this edge is not accepted as parity.
- The wholly empty upstream call currently raises a Pandas-version-specific conversion error; Rust returns stable `EmptyInput` instead of copying that incidental message.
- Exact Python exception classes/messages are recorded by the fixture, while the Rust API uses typed native errors.
- This is a native Rust API, not a production Python bridge, and it does not migrate unrelated `alpha.py` functions or establish whole-file acceptance.
