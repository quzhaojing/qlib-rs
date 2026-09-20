# Calendar codec dependency probe

This is test-only dependency evaluation, not a Python production bridge or a
replacement mapping table. It compares `encoding_rs` 0.8.35 with the host Python
`cp936` codec over all 65,536 two-byte inputs. The production dependency version
and features live in the root workspace manifest; this isolated probe pins the
evaluated version for reproducibility.

From `D:\code\github\qlib-rs`:

```powershell
$env:CARGO_TARGET_DIR='D:\code\github\qlib-rs\target\candle-checkpoint-main'
cargo build --locked --manifest-path scripts/calendar-codec-probe/Cargo.toml
python scripts/calendar-codec-probe/compare.py target/candle-checkpoint-main/debug/calendar-codec-probe.exe
```

The observed Python 3.14.0 comparison found no missing or changed Python-valid
mapping, but 2,406 extra accepted two-byte inputs. Those include the web-standard
single-byte euro extension combined with other bytes and additional GB18030
double-byte mappings. `compare.py` prints compact exclusion ranges separately.

The production adapter keeps the crate's actual character mapping and restricts
input to Python's one-/two-byte codec. Its durable test recomputes single-byte,
double-byte and mixed-sequence results using Python, including first-error spans.
It must be rerun when updating the dependency or supported Python codec contract.

Sources:

- [GBK semantics](https://docs.rs/encoding_rs/latest/encoding_rs/static.GBK.html)
- [Pinned crate manifest, features, MSRV and license](https://raw.githubusercontent.com/hsivonen/encoding_rs/v0.8.35/Cargo.toml)
- [CPython 3.14 Chinese codec](https://raw.githubusercontent.com/python/cpython/v3.14.0/Modules/cjkcodecs/_codecs_cn.c)
