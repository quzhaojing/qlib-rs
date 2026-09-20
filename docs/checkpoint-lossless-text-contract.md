# Checkpoint lossless string/path boundary

Status: additive public API approved and implemented; behavior differentials pass,
but exact coverage gates remain in progress. The existing UTF-8 renderer remains.

## Observed contract

Python strings can contain each code point from U+0000 through U+10FFFF, including
surrogates. `"\ud800\udc00"` contains two Python characters, while `"\U00010000"`
contains one. Both encode to the same two Windows UTF-16 path units. Preserve the
distinction until the OS path boundary; do not decode all input through UTF-16.

The existing String boundary is affected at more than the final file open:

| Boundary | Required behavior |
|---|---|
| Configured template and metric keys | Preserve literals, raw keyword names and nested specifications |
| Value/item/attribute protocols | Preserve code-point indexing and custom argument/result text |
| Numeric `c`, string padding and repr/ascii | Allow surrogate results/fills; repr/ascii escape them |
| Returned name and callback last-name state | Preserve raw name before graph collection and file I/O |
| Windows filesystem path | Encode valid scalars normally; preserve lone UTF-16 surrogate units |
| Unknown truthy latest mode | Even a surrogate string removes existing latest without replacement |

Live unchanged Qlib `_save_checkpoint` calls assign last name/iteration/time and
collect the graph before an invalid OS filename fails to write. Earlier rejection
by a UTF-8 conversion would change those observable partial effects.

## Candidate and rejected shortcuts

The already adopted widestring 1.2.1 `U32String` preserves code-point arrays and
Python character counts. Its checked `to_string()` correctly rejects surrogates,
but **its `to_os_string()` calls `to_string_lossy()`** in the installed source. That
shortcut mismatches 2,328 of the 3,336 test strings and must not be used here.

The isolated Rust candidate delegates scalar-to-UTF-16 conversion to
`char::encode_utf16`, preserves only surrogate code points as individual u16 units,
and uses standard Windows `OsString::from_wide`. This matches the Python
`surrogatepass` oracle and real Windows file operations. No replacement characters,
Unicode normalization, private-use placeholders, or general-purpose codec rewrite
is used. Production input must validate the Python range; U32String itself also
accepts values beyond U+10FFFF and therefore is not the entire domain contract.

JSON test transport uses integer code-point arrays. JSON strings are not a lossless
Python-string representation: readers may reject lone surrogates or combine an
escaped surrogate pair. Do not infer in-memory equality from equal Windows paths.

## Reproduction and evidence

From `D:\code\github\qlib-rs`, with the existing standalone probe target directory:

```powershell
$env:CARGO_TARGET_DIR='C:\Users\andy\.codex\tmp\qlib-checkpoint-format-probe'
cargo fmt --manifest-path scripts/checkpoint-format-probe/Cargo.toml -- --check
cargo clippy --manifest-path scripts/checkpoint-format-probe/Cargo.toml --all-targets -- -D warnings
python scripts/checkpoint-format-probe/surrogate_probe.py D:\code\github\qlib\qlib\rl\trainer\callbacks.py
```

Verified results:

- 3,336 lossless representation/UTF-16 comparisons: every one of the 2,048
  surrogates, eight scalar boundaries, 256 mixed cases and 1,024 seeded strings.
- 8 actual Windows write/read/listing cases, including lone surrogates, paired
  surrogates, non-BMP scalars, NUL failure and a missing subdirectory.
- Paired-surrogate versus non-BMP filenames alias on disk while remaining distinct
  Python/U32 strings before path conversion.
- 405 unchanged-Qlib filename calls characterize raw keyword/item/attribute names,
  custom format/str/repr effects, nested specs, padding, numeric `c` and errors.
- 24 unchanged-Qlib save calls verify retained state and actual filesystem effects
  for disabled/copy/unknown-surrogate latest modes, including write failures.

These groups originated as characterization. The current production lossless
renderer is compared against the 405 filename cases. The 24-save Rust test now
runs the unchanged Python Checkpoint methods via `--save-oracle-only` and compares
errors, retained callback state, actual saved payloads and latest-file contents
against the returned source results. A separate lossless callback harness compares
95 live-source scheduling/failure cases, including exact event and partial-file
ordering. It reuses deterministic boundary spies rather than claiming native codec
parity. The save fixture replaces Torch serialization with a fixed test payload;
it does not establish Torch checkpoint compatibility. Files live inside temporary
test directories that are automatically removed.

The 3,336 string cases now also compare production lossless `repr` against Python.
A regression with U+D800 followed by the literal characters `\ue000` demonstrated
that replacing marker escape strings after quoting corrupts user text. Repr now
escapes raw code points in place, sharing the existing RustPython quote/ASCII
escape logic and Unicode-version compatibility adapter with the UTF-8 renderer.
It does not use the surrogate-marker codec.

The mini-language adapter now also avoids an unused-character assumption. A
regression containing the entire old private-use candidate pool plus U+D800 proved
that the old codec panicked on valid text. Another regression proved that surrogate
padding could overwrite a literal private-use locale separator. The replacement
delegates text layout to the existing string engine using a sentinel distinct from
the fill, then restores original code points at content positions. Numeric surrogate
fills/`c` values use two distinct tag renderings: equal output positions preserve
literal data, and differing positions restore raw fill/character points. The second
render uses the first render's locale snapshot; the external provider is still
queried only once and only after normal validation. Tests assert provider errors,
query counts, and literal separators/decimal marks containing all four tags.

The new 9,427-case Python primitive-format differential covers text precision and
alignment, zero/Unicode fields, every surrogate `c` value, multiple fill characters,
numeric presentations and invalid specifications. A separate identity test retains
all 1,114,112 Python code points through text formatting. The third-party engines
still own syntax validation, rounding, truncation and padding; no new dependency
or public API is introduced. Full coverage acceptance is recorded in the ledger.

The unified parser and positional formatter passed the complete ordinary and
stable/nightly workspace suites on 2026-09-03. Both exact coverage checkers pass
for all 87 current production files: 15,619 lines, 1,972 functions, 20,357 regions
and 1,470 branches, each 100%. This accepts the additive text/path work, not the
unfinished concrete model-state migration or direct legacy Torch file support.

## Public API decision and acceptance

Approved direction: preserve existing UTF-8 Rust APIs and
add a lossless Checkpoint boundary using an owned project text type backed by the
third-party code-point container. Do not expose its third-party storage type as the
stable plugin contract. Existing UTF-8 callers should retain their behavior; the
lossless path must be wired through real callback state and storage, not just added
as an unused helper. No public Rust signature was changed in this characterization.

Under that approved direction, implementation must cover every boundary in the
table, validate out-of-range inputs, retain evaluation/failure ordering and code-point
indexing, and compare Rust against the frozen live-source cases. Preserve all existing
differentials and enforce the full exact total/per-file 100% lines/functions/regions/
branches gate without excluding newly added adapters or errors.

Primary references:

- [Rust Windows OsString lossless conversion](https://doc.rust-lang.org/std/os/windows/ffi/trait.OsStringExt.html)
- [Python filesystem encoding behavior](https://docs.python.org/3/library/os.html#file-names-command-line-arguments-and-environment-variables)
- Installed widestring 1.2.1 `src/ustr.rs`, U32Str::to_os_string and checked conversion.
- Local unchanged `D:\code\github\qlib\qlib\rl\trainer\callbacks.py`.
