# `get_min_cal` mutable-cache compatibility facade

## Result

`crates/core/src/time_calendar_cache.rs` adds a narrow, Python-runtime-free compatibility facade for the source-observable `functools.lru_cache(maxsize=240)` behavior of `qlib.utils.time.get_min_cal`. It leaves the existing immutable native `minute_calendar` API unchanged, reuses it for validated calendar generation, and adds raw positional/ordered-keyword keys, shared mutable list identity, exact LRU size/counters/clear behavior, failure non-caching, eviction with retained external values, and concurrent cold-call behavior. The source is pinned to SHA-256 `af7ac3709cac0d2a11a15aac478c7ceb68579d492aed322695f5f25261a69699`.

This closes the previously identified annotated integer-shift/string-region mutable-cache gap, but does **not** close all of `qlib/utils/time.py`. Dynamic Python/Pandas values outside the narrow boundary and the other whole-file gaps recorded in `docs/orca-time-closure.md` remain explicit.

## Native contract

- `TimeCalendarCall` owns positional arguments plus keyword entries in insertion order. Thus `get_min_cal()`, positional defaults, partial keyword calls, both keyword orders, and mixed positional/keyword calls remain distinct raw LRU keys exactly as source behavior requires.
- `TimeCalendarValue` deliberately supports only arbitrary-size integer shifts and region strings. `TimeCalendarKeyword` retains the raw keyword name and order.
- `TimeCalendarCache::get` increments misses before binding/execution, never inserts failures, returns the same `Arc<Mutex<Vec<NaiveTime>>>` on hits, and permits source-compatible mutation of a cached list.
- Capacity is exactly 240. Eviction drops only the cache's ownership; external `Arc` values remain alive and mutable. Repeating an evicted call creates a new identity.
- `cache_clear` removes entries and resets hits/misses while external values remain valid.
- Concurrent misses compute independently. The first published identity remains cached, while later overlapping miss callers receive their distinct computed identities, matching the pinned source probe rather than collapsing all callers onto the published value.
- Poisoned internal cache-state locking is an explicit Rust panic boundary. Individual calendar-list mutexes are caller-owned synchronization; poisoning one list does not poison cache metadata.

The coordinator added the shared `pub mod time_calendar_cache;` declaration in `crates/core/src/lib.rs`; this worker did not edit shared exports.

## Source characterization

`crates/core/tests/fixtures/time_calendar_cache_probe.py` executes the actual AST-selected source definitions and records:

- eight distinct valid raw call shapes and eight subsequent same-identity hits;
- mutation isolation across different raw keys;
- maxsize-240 LRU eviction, counter changes, and retained evicted values;
- unsupported-region retry, duplicate shift/region, excessive positional count, unexpected keyword, wrong-type shift/region, and the rule that failed calls increment misses but do not occupy cache entries;
- exact CN/TW positive and US negative Pandas Timedelta boundaries and exception classes/messages;
- two synchronized cold callers receiving distinct lists while one published list becomes the next hit;
- representative dynamic behavior: cold `False` fails, float `0.0` and later `False` collide under untyped `functools` key equality, `0.5` produces second-bearing clock values, `None` reaches the region error, and an unhashable list fails before misses increment.

The Rust integration target validates the supported native subset differentially and directly tests identity, mutation, clear, eviction, every native binding/error branch, numeric boundaries, and overlapping cold calls.

## Verification and coverage

Working directory: `D:/code/github/qlib-rs`. All build output used C: targets and `CARGO_BUILD_JOBS=2`.

```text
python crates/core/tests/fixtures/time_calendar_cache_probe.py D:/code/github/qlib/qlib/utils/time.py
cargo test -p core --test time_calendar_cache -- --nocapture

$env:CARGO_TARGET_DIR='C:/Users/andy/.codex/builds/qlib-rs-time-cache-stable'
cargo llvm-cov --locked -p core --test time_calendar_cache --no-clean --json --output-path C:/Users/andy/.codex/builds/qlib-rs-time-cache-stable/time-calendar-cache.json

$env:CARGO_TARGET_DIR='C:/Users/andy/.codex/builds/qlib-rs-time-cache-nightly'
cargo +nightly llvm-cov --locked -p core --test time_calendar_cache --no-clean --no-rustc-wrapper --branch --json --output-path C:/Users/andy/.codex/builds/qlib-rs-time-cache-nightly/time-calendar-cache-branch.json
```

Focused tests: 4 passed, 0 failed. Raw per-file audits for `time_calendar_cache.rs` are exact:

- lines: 136/136 (100%);
- functions: 14/14 (100%);
- regions: 148/148 (100%);
- branches: 10/10 (100%).

No exclusions, counter normalization, coverage-specific production behavior, assertion weakening, cleanup, dependency changes, upstream changes, commit, or push were used. The first stable audit measured 133/136 lines and 147/148 regions; the missing duplicate-region binding path was added to both the live source fixture and Rust test, after which the final stable and Nightly audits reached exact coverage.

Strict focused Clippy was requested with `cargo clippy -p core --test time_calendar_cache -- -D warnings`. The latest run reached no diagnostic in this module or integration target but was blocked by three concurrent worker-owned `alpha_returns.rs` diagnostics (two missing panic sections and one needless pass-by-value); a final clean strict result remains pending that owner correction and must not be inferred from absence of local diagnostics.

## Explicit dynamic and whole-file gaps

1. The native raw value enum does not accept Python floats, booleans, `None`, NumPy scalars, unhashable objects, user-defined hash/equality objects, or arbitrary region objects. Their representative actual-source behavior is characterized, including the surprising float/boolean untyped-key collision, but intentionally not reproduced by building a generic Python object/cache system.
2. The facade preserves exact annotated call forms, but it returns typed Rust errors rather than Python/Pandas exception objects. Source exception class/message evidence remains in the fixture.
3. Cache statistics are `u64`; behavior after counter overflow is not claimed equivalent to CPython's platform integer internals.
4. Process/thread scheduling is not deterministic. Both implementations preserve the documented concurrent-miss invariant, not a specific winner identity or completion order.
5. The prior whole-file gaps remain: mixed `str`/`Freq` result typing in `Freq.get_recent_freq`, timezone/`NaT` dynamic timestamp boundaries, a directly exported `concat_date_time` compatibility function, and exact Python exception surfaces outside this cache facade.

Therefore this work is a completed, exactly covered cache slice, not acceptance of `qlib/utils/time.py` as a whole and not a reason to increase the strict 32/230 accepted-file baseline.
