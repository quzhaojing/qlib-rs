# Third-party dependency guidance

Use this reference when selecting crates for a slice. Treat the list as preferred candidates, not permanent pins: verify current official documentation, maintenance status, platform support, licenses, and compatibility before adding one.

## Preferred capability map

| Capability | Preferred candidates |
|---|---|
| Serialization and configuration | `serde`, `serde_json`, `serde_yaml`, `toml` |
| Errors and diagnostics | `thiserror`, `anyhow`, `miette` |
| Logging and telemetry | `tracing`, `tracing-subscriber`, OpenTelemetry crates |
| Date, time, and time zones | `chrono`, `chrono-tz` |
| Stable ordering and identifiers | `indexmap`, `uuid` |
| Decimal and float wrappers | `rust_decimal`, `ordered-float` |
| Columnar schemas and interchange | Apache Arrow Rust crates |
| Parquet and Arrow IPC | Apache Arrow `parquet` and IPC crates |
| DataFrame and lazy query operations | `polars` |
| Dense arrays and numerical work | `ndarray`, `nalgebra`, `statrs` as appropriate |
| Parallel CPU work | `rayon` |
| Async I/O and orchestration | `tokio`, `tokio-util`, `futures` |
| HTTP and service middleware | `reqwest`, `tower`, `axum` when serving APIs |
| RPC and streaming data | `tonic`, `prost`, Arrow Flight |
| Object storage | `object_store` |
| Caching and concurrent maps | `moka`, `dashmap` |
| File mapping, locking, and hashing | `memmap2`, `fs2`, `blake3` |
| Redis and MongoDB | official or established `redis` and `mongodb` crates |
| SQL metadata | `sqlx` |
| CLI and progress | `clap`, `indicatif`, `clap_complete` |
| Randomness and reproducibility | `rand`, `rand_chacha` |
| Testing, coverage, and benchmarking | `cargo-llvm-cov`, `cargo-nextest`, `proptest`, `rstest`, `criterion`, `insta` where snapshots are meaningful |
| Python extension compatibility | `pyo3`, `maturin`, Arrow/PyArrow interchange |
| Compile-time component registry | `inventory` or `linkme` |
| Stable native plugin ABI | `abi_stable`; use raw `libloading` only behind a reviewed FFI layer |
| Sandboxed plugins | `wasmtime` Component Model |
| ONNX inference | `ort` |
| Rust-native ML/DL | evaluate `linfa`, `candle`, or `burn` per model |
| Existing native ML engines | maintained FFI crates or isolated process adapters for LightGBM, XGBoost, and libtorch |

## Selection rules

1. Search the existing workspace before adding a dependency or building an abstraction.
2. Match the crate to the required behavior, data layout, platform, and interoperability boundary; do not select solely by popularity.
3. Prefer one shared representation for tabular interchange: Arrow. Polars may be used internally but should not leak into stable plugin APIs.
4. Prefer batch APIs over per-row plugin or FFI calls.
5. Prefer safe APIs. Isolate and document unavoidable `unsafe` or native FFI in a dedicated adapter crate.
6. Avoid crates that only save a few obvious lines, duplicate standard-library behavior, are abandoned, impose incompatible licenses, or pull in a runtime-sized dependency for a trivial task.
7. Record dependency features and disable defaults when that materially reduces build time, binary size, native requirements, or attack surface.
8. Preserve upstream Qlib behavior even when a crate's default semantics differ; put explicit compatibility rules in a thin adapter and test them differentially.

## Local implementation exception

When local code is necessary, add a short ledger entry containing:

- required behavior;
- crates evaluated;
- the concrete incompatibility or missing capability;
- expected maintenance and test burden;
- a future replacement condition, if one is known.
