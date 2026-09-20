# Migration checkpoint — 2026-09-21

This commit preserves the current Qlib-to-Rust migration work. It is a work-in-progress checkpoint, not a release or a claim of complete migration.

- Strict whole-production-file acceptance remains 32/230 (13.91%); 198 production files remain unaccepted. The slice ledger's completed count is not whole-project progress.
- D: previously filled completely. `crates/core/src/data_normalization.rs` was truncated by a failed write. Before this commit it was restored from two independently verified replays of the successful original patch events. Restored SHA-256: `837D5B49D3CB124BBA8F664838DC5B664B90C0E798E367E541B63F15D0FCEC4B` (932 lines, 29,099 bytes). The failed truncating patch was excluded.
- The latest existing multi-index test executable passed all 13 focused tests. These results predate some concurrent source changes and do not prove current integrated acceptance.
- Independently inspected historical focused production coverage: trading indicator analysis 183/183 lines, 26/26 functions, 325/325 regions, 28/28 branches; risk analysis 333/333 lines, 42/42 functions, 525/525 regions, 90/90 branches. Final integrated static checks and documentation reconciliation remain pending.
- Alpha returns integration tests still contain a custom unsafe Arrow Array implementation rejected by the workspace's `forbid(unsafe_code)` policy. Its pending removal and documentation corrections have not been applied in this checkpoint; error-path verification and fresh coverage remain open.
- Time compatibility still needs combined-error precedence fixes for invalid shift/zero sampling step with NaT, plus a corrected wrapper-free Nightly branch run.
- Normalization needs fresh post-restoration tests, static checks and coverage; its earlier report had two uncovered regions. Model sequence still lacks real Torch-boundary and accepted branch evidence.
- Existing detailed reports may contain earlier states. This checkpoint note records the above unresolved issues without overwriting historical evidence.
- Build caches, machine-local recovery logs, and C-drive temporary submission copies are not part of this commit. No full workspace build or full test suite was rerun for this backup commit.
