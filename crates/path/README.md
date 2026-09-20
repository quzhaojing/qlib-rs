# Native path queries

Package/directory/library name: `path`. Windows-only query primitives for the
remaining source-compatible configuration resolver; no `qlib` package prefix.

## Scope

- `final_path`: Python Windows `_getfinalpathname` open/query semantics, including
  desired access zero, share mode zero, directory backup semantics and DOS-volume
  verbatim results. It does **not** implement non-strict fallback or strip prefixes.
- `find_name`: Python `_findfirstfile` semantics, including native wildcard first
  match and finding the actual name without opening the file for reading.
- `read_link`: one-hop Python `nt.readlink` semantics for symlinks and junctions.
  Uses zero access/share, opens the reparse point itself, and queries its substitute
  name. Retains verbatim targets rather than converting them to display paths.
- `normalize_case`: Python `ntpath.normcase`, replacing `/` with `\` and delegating
  lowercase mapping to native `LCMapStringEx` with invariant locale. No dot folding,
  Unicode composition normalization or filesystem access. Empty input bypasses the
  native call; explicit lengths retain embedded NUL and unpaired UTF-16 units.
- `normalize`: lexical `ntpath.normpath` behavior before native resolution. Handles
  slash spelling, dot/parent reduction, drive-relative and UNC/device roots, retaining
  NUL and unpaired UTF-16 data. Does not inspect the filesystem or expand home paths.
- `split_root` and `split`: retain source drive/root/tail and directory/name spelling,
  including mixed slash types, trailing separators, malformed UNC/device roots,
  NULs and unpaired UTF-16. Root parts concatenate exactly to the input.
- `is_absolute`: Python 3.14 predicate for rooted drives or two leading separators;
  one leading separator alone is not absolute. Uses Unicode-character positions,
  not UTF-16-unit positions, for supplementary characters before a colon.
- `join`: source drive/root/tail composition over zero or more appended paths,
  preserving exact spelling, dots and trailing separators. Uses full Unicode-16
  string lowercase solely to compare unequal drives, distinct from native normcase.
- `read_link_deep`: source traversal through an explicit `LinkOperations` boundary,
  preserving case-key cycles, query order, relative-target rules and error policy.
  `NativeLinkOperations` supplies actual native queries and classification. This
  resolves link chains, not the full non-strict path-resolution/prefix policy.
- `symbolic_link_by_open`: the classification open/handle stage, using attribute-read
  access, no sharing, backup and open-reparse flags, then SDK attribute/tag queries.
  Tests the exact symlink tag (not all name surrogates), and retains open versus
  query errors. This does not implement by-name acquisition or stat fallback.
- `symbolic_link_by_name`: optional by-name classification stage. Dynamically loads
  the Windows API-set through `libloading`, retaining the library with its function
  pointer. Missing DLL/export returns error 50; actual query errors are retained.
  Successful queries test the exact symlink tag; this stage alone is not the full
  source predicate. `is_symbolic_link` provides the composed classification.
- `directory_attributes`: source directory-search fallback attributes/tag, without
  opening the file. Preserves native wildcard matching and trailing-name trimming;
  returns `None` when trimming leaves no queryable name, so callers retain their
  prior error. Non-reparse entries have tag zero even if reserved data is nonzero.
- `lstat_attributes`: the attribute/tag projection of source Windows lstat, with
  by-name fast-path, native retry/query gates and final close-error handling.
  Does not claim full stat timestamp/inode/mode representation.
- `is_symbolic_link`: complete native classification policy with distinct trusted
  by-name errors and stat fallback only for the source's selected open failures.
- Preserve Windows error codes and unpaired UTF-16 units. Native filesystem queries
  reject every input NUL before FFI; the deep traversal retains the current spelling
  on this error, as the source does.

No file creation, writing, deletion or permission change is performed by these
production functions. `real_path` now composes default non-strict resolution;
actual configuration wiring and strict/ALLOW_MISSING modes remain required.

## Dependencies and safety

Uses Microsoft's `windows-sys` 0.61.2 (MIT OR Apache-2.0, declared MSRV 1.71), already
in the workspace lockfile. Foundation, Security (CreateFileW signature),
Storage/FileSystem, Globalization, System/Environment, System/IO, Ioctl and SystemServices provide native bindings and
constants. Wdk/Storage/FileSystem supplies reparse structure field offsets, not a
kernel driver dependency. Existing `byteorder` 1.5.0 decodes checked byte slices;
`widestring` validates terminated UTF-16 and `thiserror` provides diagnostics.

Native calls are isolated in `src/windows.rs`; core continues to forbid unsafe code.
Each unsafe block documents pointer lifetime, buffer extent or handle ownership.
Bindings and structures come from windows-sys, not handwritten ABI declarations.
Every successful file/search acquisition creates one owner with the matching closer.
Error codes are captured before drop can overwrite the thread-local last error.
Buffer sizes returned by the query are retried; capacity allocation is fallible.
The private test seam must honor the same buffer/handle contracts as Windows.

Tests exercise native files/directories/missing paths/wildcards, real locked files,
unpaired-surrogate names and NULs against the actual Python `nt` primitives. Injected
tests assert exact open flags, repeated buffer growth, error-code retention,
allocation failure and exactly-once handle closure. Allocation capacity overflow is
tested without requesting a huge physical allocation.

Lexical normalization uses standard UTF-16 slices and a stack. Rust path components
do not implement Python's malformed UNC/device split-root and parent reduction
semantics; canonicalization libraries also perform filesystem operations or reject
NUL. This small compatibility policy adds no dependency and no unsafe code. Its
61,226-case live Python differential also checks normalization idempotence. It is a
resolver prerequisite, not a replacement for filesystem/link resolution.

Reparse parsing is safe Rust: verify returned extent, declared body length,
substitute-name range and UTF-16 alignment before reading. Unknown tags retain
Python's non-symbolic-link error category. Invalid/truncated native buffers return
Windows code 4392 instead of dereferencing unchecked native pointers. No typed cast
of the byte buffer or handwritten ABI structure is used. Substitute/print names are
not interchangeable; source-specific NT-prefix replacement ignores the relative
flag and retains names of four units or fewer unchanged. Tests cover both tags,
offsets, empty/short targets, NUL/unpaired UTF-16 and all validation failures.

The real-filesystem differential creates test-owned relative/absolute/directory
links, broken/chained/cyclic links and a junction, and compares 18 outcomes against
Python. Creating symlinks requires Windows developer mode or the appropriate
privilege; the test fails explicitly when that prerequisite is unavailable. No
production Python calls are introduced. Standard `read_link` was rejected after
inspecting its local Windows implementation: it has a different sharing policy
and may convert absolute substitute names to user-friendly paths.

Native case mapping uses a size query then a separate fallible output allocation,
preserves first/second-call native failures and rejects input lengths above INT_MAX
before FFI. Tests inject those errors and allocation failure without huge memory
requests. A 66,570-case actual-Python differential covers every UTF-16 unit,
supplementary Unicode characters in 1,024-character batches, empty/path/mixed-script
contexts and idempotence. This is native Windows lowercase, not Rust lowercase,
ASCII-only matching or generic Unicode case folding. It supplies the later resolver's
cycle-key semantics, consumed by `read_link_deep`.

Path parts are measured in 126,771 actual-Python cases, including all single UTF-16
units before `:/tail`, supplementary-prefix paths, empty inputs and partial UNC
roots. These functions use standard slices and the already tested source-specific
root policy, not Rust component iteration that would discard exact spelling. The
native OsString boundary represents UTF-16; it cannot distinguish a Python string
containing two explicit paired-surrogate code points from the equivalent single
supplementary character. `is_absolute` uses the latter Unicode interpretation.
This representation limitation is not a claim of arbitrary Python-string parity.

Joining uses pinned `icu_casemap` 2.0.0 with only `compiled_data`, plus pinned
`icu_locale_core` 2.1.1 for its static root-language identifier. Case data 2.0.0 and
property data 2.0.1 remain in the Unicode-16 series in Cargo.lock; locale/provider
2.1.1 supply infrastructure, not replacement case tables. Declared MSRV is 1.82
for casemap/data and 1.83 for locale-core, within this workspace's 1.85 minimum.
License: Unicode-3.0. Cargo selected 16 new compatible packages. The only duplicate
major in the path tree is build-time `syn` 2/3 from different derive dependencies,
not competing runtime case engines. No data generation/serde/network features are
enabled for ICU and no ICU types cross the public boundary.

All Unicode mapping is delegated to ICU's root-locale full-string lowercase,
including contextual final sigma and multi-character expansions. Standard UTF-16
decoding splits at unpaired surrogates; these uncased boundary units are preserved,
not replaced. 66,687 actual Python `.lower()` cases cover every UTF-16 unit,
supplementary characters and contextual/surrogate boundaries. Another 44,135
actual `ntpath.join` cases cover one, two and three path parts. Python's Unicode
version is asserted by the fixture so a future oracle upgrade is not silently
accepted. Std/intl Unicode-17 and Windows native case mapping remain deliberately
unused for this comparison. Full non-strict path resolution remains pending.

Deep-link policy is compared with the unchanged host Python `_readlink_deep` body
in 81 controlled cases, checking full query traces and results/errors. Another
18 physical cases compose real case mapping and reparse reads, including cycles,
missing targets, junctions, surrogate targets and trailing separators. Classification
in that physical test now uses `NativeLinkOperations`, replacing the earlier
test-owned metadata with the composed production classifier. The policy uses standard
HashSet/owned paths plus existing library-backed primitives; local code is limited
to source-specific control flow, not a replacement filesystem framework.

The classification open stage uses existing `windows-sys` bindings and
`FILE_ATTRIBUTE_TAG_INFO`; no new dependency or handwritten ABI. An 18-path
ctypes/native differential tests the exact acquisition stage, and successful
results also agree with actual `ntpath.islink`. Injected tests cover attribute/tag
combinations, native failure stages, NUL rejection and exactly-once cleanup.
The source's `GetFileType` disk-device value is unused for the symlink predicate;
this isolated stage reads only the attributes and tag that determine its result.
Do not substitute this stage for the full predicate by blindly mapping all errors
to false: upstream performs additional by-name/stat queries.

By-name loading reuses pinned `libloading` 0.8.9 (ISC, MSRV 1.71, no extra features),
already in the lockfile. Resolution is restricted to the Windows system directory;
this is an OS capability, not arbitrary plugin loading. `windows-sys` lacks the
required structure/function: the minimal `repr(C)` SDK declaration is therefore
an explicit compatibility exception, not generated metadata. A test-only C program
compiled with existing `cc` 1.4.4 (MIT/Apache-2.0, MSRV 1.65, no parallel feature)
independently verifies all 15 field offsets, size, alignment and enum width/value
against installed SDK headers. No production C code or new toolchain installation.
This test currently certifies the Windows x86_64 MSVC ABI; other target ABIs still
need their own certification. Replace the local declaration once maintained SDK
bindings expose it. `cc` is a dev dependency only; no runtime compiler dependency.
Failure tests cover missing module/export, unavailable capability, NUL rejection,
native error codes and attribute/tag combinations. The 18-path physical fixture
compares native by-name results/errors, with successful results additionally checked
against actual Python `ntpath.islink`. Stat fallback has separate evidence below.

Directory fallback reuses `windows-sys` FindFirstFileW/SDK data and existing
`widestring` validation/owned search handles. Its local trimming is source-specific:
the source tests the last remaining index, so `a/` also skips the query. No generic
glob engine, directory walker or new dependency. The copied trailing-name buffer
is fallible and maps allocation rejection to source error 8. Eighteen injected
name cases check exact query paths and handle closure, plus native search errors,
NUL/allocation failures, unpaired UTF-16 and reserved-tag clearing. A 34-path ctypes
native-stage oracle compares full attributes/tags/errors/skip decisions, including
wildcards, links, junctions, absent files and an exclusive-lock file. The source
trimming policy is implemented in that oracle; this is not a claim that Python's
private C helper can be called directly.

Stat attribute projection reuses the existing SDK bindings and typed query buffers,
without new dependencies. It retains the narrower stat fast-path error list, access/
share fallback, console open flags, recursive non-link reparse traversal, unhandled
tag behavior, required file/basic metadata gates, optional file-id query and final
close failures. Intermediate retry closes are intentionally ignored as in source;
final closes happen exactly once and can override the previous outcome. Non-disk
and unsupported-device outcomes have zero attribute/tag fields, matching the source
projection. No replacement generic stat framework or production Python bridge.

Injected tests assert complete query order and flags across success/failure gates,
including close failure; classification tests separately enforce both native error
lists and distinguish open failure from handle-query failure. Actual `os.lstat`
and `ntpath.islink` are compared on 29 paths, including locks, links, junctions,
devices, roots, wildcards, NUL and trailing separators. The 18 physical deep-link
comparisons now exercise all production native operations. Full stat representations,
actual configuration/storage wiring remain required. Outer default realpath is
implemented below; this is not certification of strict modes or pathlib construction.

`final_path_non_strict` now composes final-path, deep-link and real-name queries
through `FinalPathOperations`, with production `NativeFinalPathOperations`. It
preserves the source's 16 allowed native errors, narrower real-name retry list,
unresolved suffixes, raw spelling comparison and root concatenation. Native
link/search failures permit traversal; invalid-input/allocation/length failures
propagate. This is the inner fallback, not outer `realpath` or strict/ALLOW_MISSING.
The 151 controlled cases execute the unchanged host Python function and compare
entire query traces as well as outcomes. Another 27 actual filesystem cases cover
links, junctions, cycles, unpaired UTF-16, missing suffixes and exclusive locks.
No new dependency or unsafe code: existing SDK/widestring/ICU-backed primitives
perform native acquisition, validation and joining. `std::fs::canonicalize` alone
cannot retain missing suffixes or implement source-specific native error recovery;
this small orchestration is compatibility glue, not a replacement filesystem engine.

`real_path` implements outer `ntpath.realpath(strict=False)` and `real_path_with`
accepts a replaceable `RealPathOperations`; `NativeRealPathOperations` supplies all
real native queries. Preserves normalization before cwd lookup, unconditional cwd
lookup even for absolute/NUL paths, exact device special case, drive-relative
joining, embedded-NUL renormalization, inner fallback and verbatim-prefix recheck.
UNC prefix matching is case-sensitive and final names compare raw OS strings.
The 369 source-bytecode cases compare complete query traces and results; 37 real
filesystem cases compare native UTF-16 results to actual Python realpath.

Current directory uses existing windows-sys GetCurrentDirectoryW (only added its
System/Environment feature), a 256-unit source-sized stack buffer and existing
fallible allocation. Native errors are captured immediately. Buffer growth is
retried safely if a concurrent cwd change requires another resize; this avoids
reading beyond the allocation, rather than reproducing the source's unsafe second
growth behavior. No cwd mutation. Injected tests cover exact capacities, repeated
growth, query errors before/after growth, allocation failure and unpaired UTF-16.
No new dependency version, manual ABI, Python production bridge or filesystem engine.

```powershell
cargo test --locked -p path -- --test-threads=1
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
```

## Primary references

- [CPython 3.14 Windows query implementations](https://github.com/python/cpython/blob/v3.14.0/Modules/posixmodule.c)
- [GetFinalPathNameByHandleW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfinalpathnamebyhandlew)
- [GetFileInformationByHandleEx](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getfileinformationbyhandleex)
- [GetFileInformationByName](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getfileinformationbyname)
- [libloading](https://docs.rs/libloading/0.8.9/libloading/)
- [FILE_ATTRIBUTE_TAG_INFO](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_attribute_tag_info)
- [FindFirstFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-findfirstfilew)
- [DeviceIoControl](https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-deviceiocontrol)
- [Reparse data structure](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_reparse_data_buffer)
- [LCMapStringEx](https://learn.microsoft.com/en-us/windows/win32/api/winnls/nf-winnls-lcmapstringex)
- [CPython native string mapping](https://github.com/python/cpython/blob/v3.14.0/Modules/_winapi.c)
- [ICU4X 2.0 Unicode-16 release](https://github.com/unicode-org/icu4x/releases/tag/icu@2.0.0)
- [ICU4X case mapper implementation](https://github.com/unicode-org/icu4x/blob/icu@2.0.0/components/casemap/src/casemapper.rs)
- [Microsoft Rust for Windows](https://github.com/microsoft/windows-rs)

Do not infer that a normal read lock necessarily blocks metadata queries: the
source and native tests both allow the zero-access query while ordinary file reads
fail with sharing violation. Preserve measured behavior, not that assumption.
