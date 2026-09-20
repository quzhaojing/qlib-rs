# Checkpoint format dependency probe

This standalone, unpublished Cargo package compares candidate libraries against the
installed Python interpreter. It is test tooling, not the production filename adapter
and not a replacement for workspace acceptance or coverage.

From the repository root, in PowerShell:

```powershell
$env:CARGO_TARGET_DIR='C:\Users\andy\.codex\tmp\qlib-checkpoint-format-probe'
python scripts/checkpoint-format-probe/run.py
python scripts/checkpoint-format-probe/run.py --details
```

`run.py` obtains 508 scalar cases from the live Python oracle and runs the locked
candidate package. Success means exact output equality; failure comparison checks
error presence, not Python exception class/message equality. Panics are caught only
in this isolated evaluation executable and counted as incompatibilities. Production
formatting does not catch panics to hide dependency failures.

Measured baseline (2026-09-02):

| Candidate | Exact matches | Mismatches | Panics within mismatches |
|---|---:|---:|---:|
| rustpython-format 0.4.0 | 436/508 | 72 | 14 |
| pyformat-rs 0.1.0 | 429/508 | 79 | 0 |

RustPython mismatches include Unicode precision truncation and grouped exponent
panics, Boolean/text zero padding, `z`, and huge integer-to-float overflow. pyformat
mismatches include integers outside i128 and locale `n`. The two mismatch sets do
not overlap in this baseline; this alone does **not** establish that composing them
provides unrestricted Python compatibility.

The production `rl_checkpoint_format` module uses pyformat for text/floats and
RustPython for arbitrary-precision integer presentation. Small adapters validate
character precision, reject overflowing integer float conversions, handle Bool/None,
and translate floating `n` to `g` in the C numeric locale while rejecting grouping.
An independent cross-type matrix exposed the character-precision and `zn` gaps not
found by this baseline. It was expanded to 1,350 cases to cover invalid mini-language
conversion prefixes (RustPython accepts `!r`, unlike CPython) and valid `!` fills.
Character formatting now uses pyformat's checked codepoint conversion instead of
RustPython's surrogate panic path. Separate tests assert valid Unicode scalar
boundaries and explicit errors for unrepresentable lone surrogates. Both matrices
are production Rust differential tests:

```powershell
cargo test -p core --lib rl_checkpoint_format
```

The Python source filename oracle executes the unchanged Qlib Checkpoint method for
547 baseline and 141 extended cases, including nested fields, conversions,
item/attribute lookup and custom protocol side-effect ordering. The original
source-only characterization remains, and `rl_checkpoint_name` now additionally
compares the production replacement-field adapter to all 688 source calls. Real
Trainer/file integration also uses that production adapter. Non-C locales,
Python 3.14 fractional grouping and lone-surrogate strings
remain unresolved; do not mark unrestricted format compatibility complete. Unicode
repr-version behavior and ASCII precision beyond pyformat's 9,999 limit are now
checked separately as described below.

The replacement-field layer uses explicit model protocols and incremental framing
to preserve earlier effects on later syntax/lookup errors. `rustpython-literal`
provides primitive string repr. `intl` supplies Unicode numeric properties and
assignment age, restricted to Unicode 16 for the Python 3.14 oracle; its complete
decimal classification is checked across every Unicode scalar, not only ASCII.
These choices do not turn this isolated candidate benchmark into a production gate.

## Exhaustive Unicode repr evaluation

```powershell
cargo run --locked --manifest-path scripts/checkpoint-format-probe/Cargo.toml --bin unicode-repr
```

The isolated probe compares every valid Unicode scalar against Python 3.14 / UCD
16.0.0. On 2026-09-03, rustpython-literal 0.4.0 had 18,289 exact repr mismatches out
of 1,112,064 scalars; its old character table escapes characters that Python now
prints. intl's category plus assignment-age filter had zero printability mismatches.
The production adapter now retains RustPython quote selection/ASCII escaping while
using these versioned Unicode properties for non-ASCII representation. The new
production test compares all scalar repr outputs plus 1,136 mixed strings through
`!s`, `!r` and `!a`. It passes; lone surrogates remain outside Rust String.
The probe intentionally measures the uncorrected dependency, not the adapter.
Its `default-run` preserves the existing `run.py` scalar candidate command.

## Large-precision candidate comparison and production adapter

`python scripts/checkpoint-format-probe/run.py --large-precision` compares nine
floating values with nine formats using precision 10,000 or 20,000. On 2026-09-03,
RustPython matched 69/81 cases and pyformat matched 0/81, with no caught panics in
either candidate. RustPython still differs for the `z` flag and no-type formatting
of zero/negative zero/the largest finite float. This is candidate evidence, not a
production fix; do not route every large precision to RustPython based on the
partial pass count. The ordinary 508-case probe is unchanged.

The production precision adapter now delegates at precision 1,100 (beyond binary64's
exact decimal requirements), extends only mandatory trailing zeros and adjusts the
delegated width. It passes a separate 2,675-case matrix including precision 20,000,
all float presentations, 128 seeded binary64 bit patterns, grouping, signed zero,
overflow validation and adversarial Unicode/exponent/percent fill characters. This
does not change the unadapted candidate results above. Non-C locales and
lone-surrogate strings remain open; fractional grouping has a separate adapter below.

## Unicode numeric fields and fractional grouping

The production scalar adapter now normalizes only Unicode width/precision fields,
preserving literal fills and the distinction between Unicode zero and ASCII zero
padding. Its 30,872-case Python differential covers all 760 decimal digits with
eight value types, structural edges and overflow. Decimal classification is shared
with field indices through the existing intl-based Unicode 16 helper.

`python scripts/checkpoint-format-probe/run.py --fractional-grouping` compares 198
Python 3.14 cases. Both unadapted libraries match 70 and mismatch 128, with no caught
panics. These include integer/fraction separator combinations, omitted precision,
exponent/general/percent presentations, padding and invalid types. This standalone
probe preserves the existing default/large-precision modes and does not implement
production fractional grouping.

The production adapter now adds forward fractional grouping around the existing
engines, reducing delegated width by the separator count. It preserves omitted
precision, Boolean/None nonempty-format behavior, string semantics and `n` rejection.
Its separate `--fractional-grouping-probes` fixture passes 18,217 live Python cases,
including independent separators, all alignments, precision 10,000 and 1,024 seeded
binary64 cases with adversarial fills. Twelve added live Qlib filename cases exercise
nested specs and the custom protocol boundary. The 198-case unadapted benchmark is
unchanged; full acceptance evidence belongs in the migration ledger.

## Numeric locale candidates

`python scripts/checkpoint-format-probe/locale_probe.py --grouping` compares
thousands 0.2.0 with the installed Python stdlib grouping function across 960
nonempty integer-digit sequences, six group policies and four separators. All
match. Stop/repeat terminators are adapter responsibilities; they must not be
passed as zero-sized library groups. Without the switch, the script emits 1,040
real child-process LC_NUMERIC cases for C/en-US/de-DE/fr-FR/hi-IN on Windows.

Production now uses thousands for grouping via explicit owned locale snapshots,
with numeric conversion and padding still supplied by the existing engines.
A separate 16,340-case scalar fixture and 3,500 live Qlib calls exercise the
localized provider path; see the migration ledger for full acceptance status.
num-format's OS-default lookup is not a current-CRT locale getter, locale-settings
0.3.0 lacks Windows FFI, and intl's static CLDR defaults are not the current CRT
metadata. A native Windows provider is now implemented separately in locale;
its full acceptance status remains recorded in the migration ledger.

### Native UCRT experiment

Compile `native_locale_abi.c` with the host MSVC/SDK, then run
`python scripts/checkpoint-format-probe/native_locale_probe.py --abi-exe <absolute-exe>`.
The separate `locale-native` Rust binary verifies ten locale metadata records,
ten thread-local cases, a stale-cache counterexample, and 40,000 concurrent
snapshots during 3,000 global writes. Its private layout is independently compared
with the C SDK oracle. On the validated x64 host: size 152, alignment 8, grouping
offset 16, wide decimal offset 88 and wide separator offset 96. This experiment
is not a replacement for production tests or a proof of thread safety by stress.
The production adapter has its own failure/cleanup tests and 20,000-snapshot
subprocess contract. Wide decoding reuses widestring 1.2.1; UTF-16 errors are not
silently replaced. See `crates/locale/README.md` for the ownership argument.

### Lossless string and path experiment

`python scripts/checkpoint-format-probe/surrogate_probe.py D:\code\github\qlib\qlib\rl\trainer\callbacks.py`
compares 3,336 code-point/UTF-16 cases and eight real Windows filesystem cases.
It exposes 2,328 mismatches in widestring's lossy U32-to-OsString shortcut; the
candidate's scalar std encoding plus preserved surrogate units matches every case.
It also records 405 unchanged-Qlib filename calls and 24 real-path Qlib save calls,
which are characterization only, not a migrated Rust renderer. Test JSON uses
code-point arrays, preserving adjacent-surrogate versus single-non-BMP identity.
All test files are confined to an auto-cleaned temporary directory. See
`docs/checkpoint-lossless-text-contract.md` for ordering, API and acceptance scope.

Dependencies are MIT or MIT/Apache-2.0 licensed and centrally pinned in the main workspace. RustPython
uses num-bigint 0.4 while the project and Arrow use 0.5; decimal interchange at the
private adapter boundary preserves magnitude. RustPython also brings lexical parsing
0.8, whereas Arrow uses 1.x. Those transitive families cannot be unified through a
feature toggle. Its default malachite backend is disabled. pyformat has no runtime
dependencies. See the installed locked crate sources and the upstream projects:
[RustPython/Parser](https://github.com/RustPython/Parser),
[pyformat-rs](https://github.com/VoiceLessQ/pyformat-rs), and
[Python format specification](https://docs.python.org/3/library/string.html#format-specification-mini-language).
