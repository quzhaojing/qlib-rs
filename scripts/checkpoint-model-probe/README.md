# Checkpoint model-state candidate

Independent evaluation of Candle CPU tensors and SafeTensors, plus a selectable
production-core adapter path; this is not a legacy Torch/pickle bridge. Its own
Cargo workspace/lockfile isolates candidate versions. `core` separately pins its
adopted 0.9.1 backend. The package/directory omit `qlib`.

The Python runner executes unchanged classes from Qlib's `network.py` and `policy.py`,
replacing only Qlib application imports. Torch, Gym and Tianshou are real dependencies;
network/policy bodies are not mocks. `weight_file=None`, so the removed Trainer import
is never used. The source hashes are included in the report.

## Run

From `D:\code\github\qlib-rs`, use the isolated Python oracle environment; do not
install these packages into the user's default interpreter:

```powershell
uv venv --python C:/Users/andy/AppData/Local/Programs/Python/Python312/python.exe C:/Users/andy/.codex/tmp/qlib-model-oracle
uv pip install --python C:/Users/andy/.codex/tmp/qlib-model-oracle/Scripts/python.exe torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python C:/Users/andy/.codex/tmp/qlib-model-oracle/Scripts/python.exe tianshou==0.4.10 'numpy<2' safetensors
$env:CARGO_TARGET_DIR='C:/Users/andy/.codex/tmp/qlib-model-candidate-target'
$env:CARGO_BUILD_JOBS='4'
cargo test --locked --manifest-path scripts/checkpoint-model-probe/Cargo.toml
cargo build --locked --manifest-path scripts/checkpoint-model-probe/Cargo.toml
& C:/Users/andy/.codex/tmp/qlib-model-oracle/Scripts/python.exe scripts/checkpoint-model-probe/run.py --binary C:/Users/andy/.codex/tmp/qlib-model-candidate-target/debug/checkpoint-model-probe.exe --report C:/Users/andy/.codex/tmp/qlib-model-state-probe.json
```

Do not recreate an existing environment to resume a download. Model data are generated
locally in a standard temporary directory; no pretrained weights or market data are
downloaded. Tensor fixture files are removed by that temporary-directory context.

## Evidence boundaries

- Check PPO and DQN with RNN/LSTM/GRU extractors, one/two layers, and DQN target copies.
  Materialize Adam moments before taking state snapshots. Compare every tensor's exact
  bytes, dtype and shape through the Rust candidate; restore a real Python policy and
  compare its forward outputs bit-for-bit. This is not Rust inference parity.
- Check shared parameter conflicts in registration order with reversed incoming mapping
  order. Safetensors alone does not preserve the model's sharing or `_metadata`;
  the fixture passes registration/sharing explicitly and restores metadata separately.
- Check F16/BF16/F32/F64 Attention snapshots; non-F32 forward is not claimed.
- Test detached/shallow versus owned snapshots and live-variable setter failures.
  Characterize actual Python partial mutation on shape/missing/unexpected-key errors,
  dtype conversion and strided inputs separately. The loader now accumulates errors
  and continues valid copies; full recursive/custom Python loader semantics remain open.
- Probe integer/bool buffers separately and retain unsupported or dtype-widening outcomes
  as limitations, not successful lossless migration.

Production acceptance remains open. Candle 0.11's source uses `usize::is_multiple_of`,
[stable since Rust 1.87](https://doc.rust-lang.org/std/primitive.usize.html#method.is_multiple_of),
above the production manifest's Rust 1.85 declaration. Evaluate a compatible dependency
version or resolve the toolchain contract before adopting it. The candidate runs on the
installed Rust 1.98; it does not raise the production MSRV.

The `msrv-candidate` feature selects Candle 0.9.1 instead, and has now passed a real
Rust 1.85.0 build/test. The `production` feature selects that backend and calls the
actual `core::rl_candle_checkpoint` adapter. Select these with `--no-default-features`;
pass matching `--runtime 0.9.1` or `--runtime production-0.9.1` to the Python runner.
The production path rejects unsupported initial model dtypes rather than creating
widened destinations. Incoming types are handled separately: BOOL/I8/I16/I32/U16
can now be converted into existing native targets without changing those targets'
dtypes. Both paths have passed the 28 model cases and five
partial-load comparisons, but the production module's full acceptance remains open.

Production now adds a 29th real Torch case with reserved/prefix-like/Unicode-NUL
parameter names. The adapter prefixes every stored key with `p` and marks its
SafeTensors metadata `core.candle_policy.tensor_names=prefix-p-v1`; the runner
decodes that marker before restoring Python models. Unmarked inputs stay literal.
This is a lossless native wire convention, not direct legacy-file interoperability.
Latest name-regression report: `target/model-state-names-production091.json`.

The production runner additionally compares 147 incoming dtype/shape cases against
real Torch module loading: seven source types × seven native target types × scalar,
vector and empty shapes. It compares target dtype, shape and exact bytes in both
snapshot and restored files. Report: `target/model-state-incoming-dtypes.json`.
Unsupported native target storage and still-unsupported source types (e.g. U64)
remain limitations; successful input conversion does not close those gaps.

The scalar-shape extension adds 231 real Torch comparisons covering one-dimensional
input into scalar parameters, singleton/long/empty vectors, ordinary rank mismatches,
and prior/later mutation order. Empty scalar indexing must abort immediately, while
ordinary copy errors allow later valid assignments. The matrix includes BOOL/I16/F64
input across seven native targets and empty U64 input before dtype rejection. It
checks both saved and restored files against actual Torch values, shapes, dtypes
and exact bytes. Report: `target/model-state-scalar-shapes.json`. These are loader
behavior checks, not legacy file-format interoperability or full ML implementation.
The corresponding root-lock scalar extension also passes complete workspace tests
and the stable/nightly exact total/per-file gates for 88 production files. See the
`scalar-shapes-unified.json` coverage reports and migration ledger for final counts.

The production probe also exercises the actual Qlib `set_weight` wrapper against
the new Rust retry protocol. It runs 58 weight-loading cases over real PPO/DQN
models plus collision and indexing edge cases. Compare the mutated input map
as well as the model snapshot/restored state: retry behavior is not just a success
flag. `weight_names` in a probe manifest explicitly carries input dictionary order;
tensor-file ordering is not a substitute. Report:
`target/model-state-policy-weights.json`. All earlier model/dtype/shape cases remain.
The root-lock implementation has also passed full workspace tests, strict Clippy
and exact stable/nightly total/per-file gates for 89 production files; reports are
`target/candle-checkpoint-stable/policy-weights.json` and
`target/candle-checkpoint-nightly-direct/policy-weights.json`. Complete model
inference/training and file extraction are not proven by these state-load tests.

The ordered-weight extension routes all 58 retry cases through production
`CandlePolicySnapshot::into_policy_weights`. The input SafeTensors file carries
the versioned `core.candle_policy.layout` order/whole-tensor-alias manifest;
`weight_names` is now only an expected-order assertion, not the loader's ordering
source. `weight-aliases.json` is compared with actual post-retry Python storage
aliases, and conversion back to Vars retains those aliases. Storage identity,
offset, shape, stride and dtype distinguish real aliases from equal or empty
independent tensors. Current report: `target/model-state-ordered-weights.json`.
All previous model/dtype/shape tests remain. This does not implement arbitrary
overlapping/strided storage relationships, unsupported native dtypes or training.
The root-lock extension also passes complete ordinary/stable/nightly suites and
the exact total/per-file gates for all 91 production files. Current coverage reports:
`target/candle-checkpoint-stable/ordered-weights.json` and
`target/candle-checkpoint-nightly-direct/ordered-weights.json`.

After C: ran out of build space, the three original candidate target directories were
moved intact under `D:\code\github\qlib-rs\target\model-probe-artifacts`. Their original
C: binary paths above are historical. Use current production commands:

```powershell
$env:CARGO_TARGET_DIR='D:/code/github/qlib-rs/target/candle-checkpoint-main'
cargo build --locked --manifest-path scripts/checkpoint-model-probe/Cargo.toml --no-default-features --features production
& C:/Users/andy/.codex/tmp/qlib-model-oracle/Scripts/python.exe scripts/checkpoint-model-probe/run.py --runtime production-0.9.1 --binary D:/code/github/qlib-rs/target/candle-checkpoint-main/debug/checkpoint-model-probe.exe --report C:/Users/andy/.codex/tmp/qlib-model-state-production091.json
```

The standalone probe has its own lockfile; unrelated core transitive versions can
differ from the root lock. Root-lock tests and coverage remain independently required.

Official APIs inspected: [Candle variables](https://docs.rs/candle-core/0.11.0/candle_core/struct.Var.html),
[Candle SafeTensors](https://docs.rs/candle-core/0.11.0/candle_core/safetensors/index.html),
and the downloaded crate's `variable.rs`/`safetensors.rs`. Shared identity, owned snapshots,
ordered loader behavior and an envelope for non-tensor state must compose with the real
Trainer driver before Checkpoint can be closed. Tests here are characterization tools,
not a newly accepted production module or an extension of its existing coverage claim.

## Verified result (2026-09-03)

The Rust unit test, build and strict Clippy pass. The live runner passes all 28 model
cases (2,136 named tensors, 24 Python forward comparisons and 12 shared-state cases),
plus seven dtype probes and five Python loader-effect probes. BOOL/I8 rejection and
U16-to-U32 widening are confirmed limitations. Report:
`C:\Users\andy\.codex\tmp\qlib-model-state-probe.json`.

The report includes Torch module metadata with an empty root key. In PowerShell use
`ConvertFrom-Json -AsHashtable`, not the default object conversion, to read it.
