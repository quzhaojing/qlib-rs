# qlib.model.utils model-sequence migration

Status: native implementation and focused tests are complete, but whole-file acceptance remains blocked by unavailable real Torch interoperability evidence and Nightly coverage attribution.

## Contract

- Source: `D:/code/github/qlib/qlib/model/utils.py`, SHA-256 `4adc4cea4e29287c98967e143bc9a143da561d33d3539e774dff44636486949f`.
- `ConcatDataset` retains its input datasets in order, reports the minimum child length, delegates the original index object (including negative integers, slices, arbitrary-precision integers, or custom objects) to every child from left to right, stops at the first failure, and returns a fixed ordered tuple of the original item identities. With no datasets, indexing returns an empty tuple while the current Python 3.14 source run raises `ValueError: min() iterable argument is empty`.
- `IndexSampler` delegates length and indexing without normalization, returns the exact item identity paired with the original index object, and propagates source failures. Both source attributes remain replaceable after construction.
- Import surface is `Dataset`, `ConcatDataset`, and `IndexSampler`; the Python `ConcatDataset` directly inherits `torch.utils.data.Dataset` while `IndexSampler` directly inherits `object`.

## Rust boundary

- `ModelSequenceLength` and `ModelSequenceIndex<Index, Item, Error>` are process-local adapters. The index is borrowed unchanged rather than narrowed to a machine integer; identity-carrying index adapters use handles such as `Arc`.
- `ConcatDataset` accepts an arbitrary number of boxed adapters with a common item boundary. `DatasetTuple` is an immutable ordered tuple representation. Adapters use identity handles such as `Arc<Mutex<T>>` when mutation visibility matters.
- Child errors are returned unchanged. `EmptyConcatDatasetError` is the sole framework-created error and preserves the exact Python message.
- No dependency was added. Pulling in Torch or a Python runtime would be disproportionate for an inheritance marker and two sequence delegation contracts; physical Python/Torch interoperability remains an edge adapter concern.

## Evidence

- `model_sequence_contract.py` imports and executes the hash-pinned real source with a minimal `torch.utils.data.Dataset` module boundary, then records surface/inheritance, constructor identity, signed indexing, tuple order, mutation visibility, evaluation order, empty behavior, and exact source exceptions.
- Rust integration tests reproduce those cases with typed adapters and identity-preserving handles.
- Nightly branch instrumentation does not attribute cross-crate generic instantiations from the exported-API integration target. The approved supplemental `model_sequence_nightly` target path-includes the entire unchanged production module (not copied or selectively extracted code) and meaningfully exercises the same success, mutation, replacement, empty, short-circuit, arbitrary-index and failure contracts in the measured crate. The exported-API `model_sequence` target remains part of the same Nightly run.
- Because the approved path-included integration target still produced no generic Nightly attribution on Windows, the coordinator approved meaningful `cfg(test)` unit instantiations inside the owned production module and a filtered `--lib model_sequence` Nightly gate. The unit test exercises the unchanged generic API; it does not alter production behavior or disable coverage.

## Verification

- The coordinator exported `model_sequence` from the crate root. The final hash-pinned Python 3.14 characterization completed successfully against the real upstream source; its `torch.utils.data.Dataset` boundary is deliberately minimal because all installed Python 3.12, 3.13, and 3.14 interpreters lack Torch.
- Final stable command: `cargo llvm-cov -p core --test model_sequence --json --output-path C:/Users/andy/.codex/builds/qlib-rs-model-sequence-stable/model-sequence-stable-final.json`. It completed in 13m27s after the coordinator's Arrow feature change forced a rebuild; all 5 tests passed. Raw `model_sequence.rs` coverage is 80/80 lines, 23/23 functions, and 96/96 regions. Stable reports 0/0 branches, which is not treated as branch proof; generic instantiations are 32/56 and are not an acceptance metric.
- Earlier approved Nightly exported and path-included integration runs passed 5/5 and 1/1 tests but attributed only 3/80 lines, 1/23 functions, 5/96 regions, and 0/0 branches to this file. The first approved in-crate fallback omitted `--no-rustc-wrapper`; it completed in 17m31s and its one owned test passed with 410 unrelated tests filtered out, but cargo-llvm-cov 0.9.0 emitted `files=[]` and zero totals. Manual export of that emitted test executable with both newest profile files contained only `locale`/`path` maps and no `core/model_sequence` map; the failed JSON is preserved.
- The coordinator then approved one retry using the root-proven invocation: `cargo +nightly llvm-cov test --locked --offline -p core --lib model_sequence::tests --no-clean --no-rustc-wrapper --branch --json --output-path C:/Users/andy/.codex/builds/qlib-rs-model-sequence-nightly/model-sequence-nightly-unit-no-wrapper.json`. Cargo 1.100.0-nightly (`e8cb624d5`, 2026-08-22) with cargo-llvm-cov 0.9.0 completed in 16m59s; the owned test passed with 411 unrelated tests filtered out. The report has one current `model_sequence.rs` map (no duplicate filename): 149/149 lines, 29/29 functions, 31/31 instantiations, and 248/248 regions; it still reports 0/0 branches. Per the acceptance rule, 0/0 is recorded but not promoted to proven 100% branch coverage.
- Because the default report can discover stale binaries after shared feature changes, the coordinator's `scripts/export-coverage-binaries.ps1` re-exported the unchanged merged profile from the sole executable recorded by that cargo run. The raw report is `C:/Users/andy/.codex/builds/qlib-rs-model-sequence-nightly/model-sequence-nightly-exact-binary.json`, with input hashes and exact LLVM argv beside it in `.inputs.json`; it contains one `model_sequence.rs` map and reproduces 149/149 lines, 29/29 functions, 31/31 instantiations, 248/248 regions, and 0/0 branches. No source filter, counter normalization, or binary omission was used.
- Focused strict Clippy command `cargo clippy -p core --test model_sequence --test model_sequence_nightly -- -D warnings` reached the crate and reported no owned-file diagnostic. It stopped on four out-of-scope `time_compat.rs` diagnostics: missing `# Panics` at line 174 and `i64`-to-`u32` truncation warnings at lines 284-286. Those files were not edited.
- `rustfmt --edition 2021 --check` passes for all three owned Rust files. Read-only `python scripts/build-symbol-inventory.py --check` passes on the integrated workspace; no generated inventory was edited by this worker.

## Remaining blockers

- The mock boundary proves the two upstream classes' local method bodies, observable imports, and direct inheritance shape, but cannot prove behavior supplied by an actual installed `torch.utils.data.Dataset`. No Torch installation or production bridge was authorized, so the real interoperability boundary remains unverified.
- The final stable and corrected Nightly source metrics are exact for lines/functions/regions, but both report zero branch sites. A 0/0 branch report is not accepted as 100% branch evidence.
- Consequently `qlib/model/utils.py` must remain outside the whole-file accepted count. No whole-file completion or accepted-count increase is claimed in this report.
