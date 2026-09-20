# `qlib/utils/time.py` whole-file closure audit

## Executive result

The source-pinned whole-file characterization and the focused Rust integration test pass for upstream `qlib/utils/time.py` at SHA-256 `af7ac3709cac0d2a11a15aac478c7ceb68579d492aed322695f5f25261a69699`. The audit accounts for all eleven module-level public constants/types/functions and all thirteen `Freq` attributes/methods, including defaults, return types, boundary errors, mutable-cache identity, call-shape cache keys, and caller-visible special cases. This is strong closure evidence, but it is **not whole-file acceptance**: exact Python mutable-list cache semantics, the mixed `str`/`Freq` return type of `Freq.get_recent_freq`, `NaT` and timezone-bearing timestamp boundaries, exact Python exception classes/messages, and a directly exported `concat_date_time` adapter remain non-isomorphic or absent at the current Rust boundary.

## Scope and inventory

- Upstream source: `D:/code/github/qlib/qlib/utils/time.py` (read-only).
- Owned Rust implementation reviewed: `frequency.rs`, `market_calendar.rs`, `minute_alignment.rs`, `single_value.rs`, and `epsilon.rs`.
- Existing supporting adapter reviewed but not modified: `intraday_index.rs`.
- Source module symbols: `CN_TIME`, `US_TIME`, `TW_TIME`, `get_min_cal`, `is_single_value`, `Freq`, `time_to_day_index`, `get_day_min_idx_range`, `concat_date_time`, `cal_sam_minute`, and `epsilon_change`.
- `Freq` surface: four normalization constants, `SUPPORT_CAL_LIST`, constructor, equality, string/repr, parse, timedelta, minimum-delta, and recent-frequency selection.
- Source callers audited under `qlib/`: configuration, backtest decisions/calendar utilities/high-performance quote storage, file calendar storage, RL order execution, high-frequency contributed operators, reporting, resampling, and strategy documentation.

## Evidence added

- `crates/core/tests/fixtures/time_whole_file_probe.py` executes AST-selected definitions from the exact source file. It records the source digest; signatures/defaults; constant, list, time, Timestamp, Timedelta, tuple, integer, `Freq`, and `NaTType` result types; exact ordinary and exceptional results; source cache mutation; and the fact that `functools.lru_cache` distinguishes default, positional, partial-keyword, and all-keyword call shapes.
- `crates/core/tests/time_whole_file.rs` pins the digest and symbol inventory, validates that complete Python snapshot, and checks the corresponding native frequency, calendars/cache identity, single-value, intraday-index, sampled-range, alignment, date/time composition, and epsilon behavior in one focused integration target.
- No production Rust file, shared manifest, lockfile, shared export file, ledger, inventory, upstream source, notebook, or example was changed.

## Verification performed

Working directory: `D:/code/github/qlib-rs`. Build output remained under `C:/Users/andy/.codex/builds/qlib-rs` with two jobs.

```text
python crates/core/tests/fixtures/time_whole_file_probe.py D:/code/github/qlib/qlib/utils/time.py
cargo test -p core --test time_whole_file
cargo clippy -p core --test time_whole_file -- -D warnings
```

Results: the Python characterization exited zero; the Rust target passed `1 passed, 0 failed`; focused warnings-denied Clippy exited zero. A first test run correctly exposed a mistaken local interpretation of Python `cache_info()` field order (`hits, misses, maxsize, currsize`); the expected assertion was corrected without changing the fixture or production behavior. No `--lib`, broad workspace test, coverage run, dependency installation, relinking, cleanup, commit, or push was performed.

## Compatibility findings and remaining gaps

1. Python `get_min_cal` returns a mutable cached list. Mutating one cached call changes later results for that exact cache key. Rust intentionally returns immutable `Arc<[NaiveTime]>`; it preserves same-key allocation identity and LRU eviction but cannot expose source mutation.
2. Python's decorator caches raw argument forms, so `get_min_cal()`, `get_min_cal(0, "cn")`, `get_min_cal(region="cn")`, and `get_min_cal(shift=0, region="cn")` occupy four distinct entries. Rust canonicalizes these to one `(shift, region)` key.
3. Python `Freq.get_recent_freq` returns `str` when its winning candidate is a string and `Freq` when its winning candidate is a `Freq`; Rust's typed API consistently returns `Option<Frequency>`.
4. Python timestamp APIs accept timezone-bearing `Timestamp` and `NaT`; `cal_sam_minute` deliberately strips timezone through `concat_date_time`, while `epsilon_change` preserves timezone and propagates `NaT`. The current native APIs accept `NaiveDateTime`, so they model ordinary/range-checked values but not those dynamic variants.
5. The native error enums preserve failure categories but do not reproduce Python exception classes and exact messages at a Python-facing boundary. Unsupported-region behavior is represented by the closed `Region` enum rather than a runtime `ValueError`/`NotImplementedError`.
6. Date-plus-time composition is implemented internally by normal Chrono composition and exercised in the test, but there is no separately exported native `concat_date_time` function. Adding one or adding a mutable exact-cache compatibility facade requires a shared `crates/core/src/lib.rs` export decision owned by the coordinator.

These gaps are explicit and prevent increasing the strict accepted whole-file count. The overall accepted baseline therefore remains **32/230 production files (13.91%)**, with no inferred completion.

## Proposed exact coverage gates

Run only after coordinator approval because coverage and shared core executable relinking are reserved. Use unique report/target locations outside the D: workspace and do not reuse historical measurements:

```text
$env:CARGO_TARGET_DIR='C:/Users/andy/.codex/builds/qlib-rs-time-whole-stable'
$env:CARGO_BUILD_JOBS='2'
cargo llvm-cov --locked -p core --test time_whole_file --json --output-path C:/Users/andy/.codex/builds/qlib-rs-time-whole-stable/time-whole-file.json --fail-under-lines 100 --fail-under-functions 100 --fail-under-regions 100

$env:CARGO_TARGET_DIR='C:/Users/andy/.codex/builds/qlib-rs-time-whole-nightly'
cargo +nightly llvm-cov --locked -p core --test time_whole_file --branch --json --output-path C:/Users/andy/.codex/builds/qlib-rs-time-whole-nightly/time-whole-file-branch.json
```

The stable report must be audited per owned production file for exact line/function/region 100%, and the Nightly JSON must show zero uncovered branches. Coverage is currently **not measured and not claimed**; even exact native coverage would not erase the semantic gaps above.
