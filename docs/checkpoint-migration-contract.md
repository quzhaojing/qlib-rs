# Checkpoint callback: implementation contract

Current workflow status: `in_progress`. The additive lossless public interface in
[the text/path contract](checkpoint-lossless-text-contract.md) was approved and is
implemented. Current verification and remaining gaps are recorded in
[the migration ledger](migration-status.md); historical evidence below is not a
current completion claim. The complete migration is not finished. Whether direct
legacy Python/Torch file interoperability is required remains a separate unanswered
choice; do not infer a Python runtime bridge from automatic continuation.

## Verified implementation snapshot before the decision stop

Status: `in_progress`. Production filename adapters pass 4,263 live Qlib filename differentials, including five numeric locales through explicit per-field providers and 63 shared-model cases. All 1,112,064 Unicode scalar repr outputs and 1,136 mixed strings match Python. Unicode width/precision, Python 3.14 fractional grouping and locale grouping/padding compose with existing format engines; 69,962 scalar cases include 18,217 fractional-grouping and 16,340 locale cases. Five native adapter tests and 32 Checkpoint-focused tests pass, as do formatting, strict Clippy, complete workspace tests and stable/nightly coverage. All 84 reported production files meet exact total/per-file 100% gates: 14,935 lines, 1,894 functions, 19,396 regions and 1,334 branches. These are the completed shared-value pipeline's counts, not a claim that missing functionality is tested. Native behavior is selected with CurrentCrtRlCheckpointLocale and LocalizedPythonRlCheckpointName, not by silently changing the C-locale unit adapter. Surrogate and model/Torch codec compatibility still prevent closing this migration slice; native acquisition on other platforms/static runtimes is not implemented or claimed.

## Authoritative local source

- `D:\code\github\qlib\qlib\rl\trainer\callbacks.py`: Callback and Checkpoint.
- `D:\code\github\qlib\qlib\rl\trainer\trainer.py`: state_dict, load_state_dict, fit, get_policy_state_dict.
- `D:\code\github\qlib\tests\rl\test_trainer.py::test_trainer_checkpoint`: two iterations produce 001.pth and 002.pth; latest points at 002.pth; loading 001 restores iteration 1 and episode 100.
- Existing Rust integration points: rl_trainer_driver, rl_trainer_checkpoint, training_vessel_state, rl_log_writer.

The scheduling/mutation-order notes below are now characterized by `rl_checkpoint_callback_contract.py`, which executes unchanged source methods against deterministic boundary spies. It does not validate Torch bytes or arbitrary Python filename formatting. Real Rust filesystem and Trainer graph integration tests provide separate evidence; keep these verification scopes distinct.

## Scheduling and state

Construction converts dirpath to Path but does not create it. Defaults are filename `{iter:03d}.pth`, save_latest `link`, absent iteration/time intervals, and save_on_fit_end true. Last name/iteration/time all start as None. Fit-start and checkpoint load are inherited no-ops: callback reuse does not reset these fields; inherited state_dict returns None and does not persist them.

Trainer increments current_iter and sets the maximum-iteration stop flag before IterEnd. Checkpoint nevertheless evaluates `(current_iter + 1) % every_n_iters == 0`. Do not substitute `current_iter % every_n_iters`. A zero interval fails during the hook, not construction; negative and arbitrary-sized integer intervals need characterization.

The time condition is evaluated after the iteration condition, even if the latter already requested a save. An absent last time makes a configured time interval immediately due without calling time.time in that condition. Otherwise use `time.time() - last_time >= interval`. Source wall-clock behavior includes backward jumps; replacing this with a monotonic clock changes the contract.

FitEnd ignores the intervals. It saves only if enabled and current_iter differs from last_checkpoint_iter. Consequently a previous failed save that already assigned the iteration may suppress the final save. Calling IterEnd twice at the same iteration can save twice; there is no general same-iteration deduplication there.

## Save order and partial failure

| Step | Observable operation | State retained if this step fails |
|---|---|---|
| 1 | mkdir(parents=True, exist_ok=True) | Previous callback state |
| 2 | Compute filename; then assign last name | Directory effects; previous last name if formatting fails |
| 3 | Read current iteration; assign last iteration | New last name |
| 4 | Read time.time; assign last time | New last name and iteration |
| 5 | Evaluate trainer.state_dict | All three new last fields; graph component side effects |
| 6 | torch.save graph to selected path | New last fields; actual codec/file partial effects |
| 7 | For truthy save_latest, check exists OR islink and unlink latest if found | Checkpoint file; new last fields; any unlink effect |
| 8 | Exact `link` creates symlink; exact `copy` copies checkpoint | Earlier effects; no rollback |

Do not open/truncate a destination before collecting the graph merely because a Rust serializer accepts a writer. Python evaluates trainer.state_dict before calling torch.save. No temporary-file replacement, rollback, retry, fsync guarantee, or retention cleanup exists in this callback.

The existing graph traversal must save the actual live vessel, every named callback and every named logger in source order. Saving only the vessel/model would not satisfy this callback. Include Checkpoint's own unit state. The current driver callback receives control and vessel, not a mutable reference to the entire callback list: resolve safe live-registry ownership explicitly, with integration tests. Do not manufacture copies of other callbacks to bypass borrowing, or hold a callback mutex while recursively trying to snapshot that same callback.

## Filename and latest-path behavior

- Formatting is Python str.format with explicit iter, local datetime.now formatted as `%Y%m%d%H%M%S`, and expanded metric keywords. The timestamp for the name and the numeric last-save timestamp are separate clock calls.
- Metric keys `iter` or `time` cause duplicate-keyword failure; they do not override reserved values. Missing keys, conversion flags, escaped braces, numeric width/precision, malformed formats and unsupported values require source cases.
- Inspect maintained formatting libraries before adding a Python-format compatibility adapter. A general Rust format string or template engine is not automatically equivalent to Python str.format. Do not silently reduce the supported filename language to literal substitution.
- Concrete CPython probes for the candidate-library test matrix: `format(True, '05')` is `00001`; `format('x', '05')` is `x0000`; `format(1.0, 'z.2f')` is `1.00`; `format(1.0, ',.2e')` is `1.00e+00`; `'{data[:]}'.format(data={':': 'colon-key'})` is `colon-key`. These were executed with the installed Python. Inspecting rustpython-format 0.4.0 found special bool/no-type handling, a grouping fallback that can panic, and colon-based field splitting; exercise these probes before adopting it, rather than assuming current CPython equivalence from its name.
- Only dirpath is created; a formatted filename containing a missing subdirectory can still fail. Source path joining also permits absolute paths. Any restriction or normalization is a compatibility change that must be explicit rather than disguised as parity.
- With save_latest disabled, existing latest files are untouched. A dangling latest symlink must be detected through islink when exists is false. A truthy unrecognized value removes an existing latest but creates no replacement in the unvalidated Python implementation; characterize before choosing a typed frontend policy.
- Symlink target is exactly dirpath / filename, not an automatically canonicalized absolute path. Relative dirpath can therefore yield a link resolving differently from an absolute path. Verify target text as well as dereferencing.
- If the selected filename is latest.pth, the unlink/copy/link sequence can remove or alias its own source. Freeze failure/effect order instead of adding an undocumented special case.
- Real Windows symlink privilege failures remain failures; do not silently replace link mode with copy. Test controlled lifecycle failures independently of machine privileges, plus the actual supported host filesystem behavior.

## Dependency and plugin work

Reuse existing BigInt for iteration arithmetic, Thiserror for typed failures, Serde and the existing checkpoint DTOs for Rust-owned state, and standard filesystem operations. Existing Bincode can encode those DTOs; it does not encode Torch pickle/ZIP files. Keep any Rust codec explicitly identified and test its matching reader. Do not label a Bincode file Torch-compatible because its suffix is .pth.

Before selecting filename/local-clock dependencies, inspect installed workspace capabilities and verify the selected library's official API and platform support. Separate replaceable filename formatting, clocks, graph collection and file/model codec boundaries only where contracts require them. These are linked Rust plugin interfaces; a stable dynamic-library ABI remains separate.

Model-owned tensor states need concrete, lossless snapshot representations and matching codecs, not just generic serde bounds. Direct interoperability with existing Torch files additionally needs a compatible codec/bridge if that consumer is required; it is not implied by saving a Rust checkpoint with a .pth suffix. File persistence must compose with the existing ordered state restore path after attachment and before FitStart. Keep Rust state completeness, legacy format compatibility and full ML implementation gaps distinct in the ledger. A Trainer state graph is not a serialized neural computation graph.

The concrete state inventory must follow the source, not a stronger invented resume
guarantee. `Trainer.state_dict` explicitly describes an iteration-boundary best-effort
snapshot and intentionally omits collector replay-buffer data. `TrainingVesselBase`
delegates only to `policy.state_dict`; it does not separately save optimizer, RNG or
collector state. Inspect the selected concrete policy's implementation before deciding
which of those objects are included. Preserve everything actually returned by the
policy, including tensor dtype/shape and state needed by its loader; do not claim
bit-for-bit continuation of arbitrary mid-collect training from the existing envelope.
The source PPO shares its extractor between actor and critic and deduplicates optimizer
parameters; DQN reuses the PPO actor architecture. Concrete adapters therefore need
state and shared-parameter tests for those real models, not just the current integer
vector persistence fixture. These model-specific requirements remain unverified.

Shared heterogeneous filename values can use `IndexMap<String, Arc<dyn RlCheckpointFormatValue>>`
through the existing callback and filename interfaces. The Arc forwarding implementation
preserves pointee identity and locale/custom protocol dispatch, without requiring the model
itself to implement Clone. Callback metric snapshots clone only these shared handles and
release the runtime lock before model calls. This does not make arbitrary trait objects
serializable or supply a Torch file codec. Model snapshots remain separate work;
the direct legacy-file compatibility requirement is awaiting the user's choice.

### Model codec candidate check (2026-09-02)

The source saves `trainer.state_dict()` and restores with
`torch.load(ckpt_path, weights_only=False)`, not just a tensor map. Its graph includes
the vessel, named callbacks/loggers, counters, stage, metrics and stop state.

- [tch VarStore](https://docs.rs/tch/latest/tch/nn/struct.VarStore.html#method.save)
  documents safetensors or libtorch C++ module output, not Python pickle output.
  Its loader dispatches pickle only for `.bin`/`.pt`; passing Qlib's `.pth` path is
  not automatically the right loader. This API alone cannot replace the full
  Qlib graph roundtrip.
- [Candle pickle](https://docs.rs/candle-core/latest/candle_core/pickle/index.html)
  exposes checkpoint tensor readers, a lazy tensor loader and a pickle stack reader.
  These are useful candidates for tensor import, but the documented API does not
  establish complete arbitrary callback/logger graph writing and restoration.

Neither crate was added during this check. A tensor import demonstration is not the
required checkpoint acceptance test. Keep the existing file-codec seam, select the
actual model payload representation, and verify Python-to-Rust and Rust-to-Python
whole-graph roundtrips before claiming Torch compatibility. An optional Python bridge
was raised as a user preference question; no bridge/runtime decision has been assumed.

## Required evidence before marking done

1. Execute the unchanged source AST with deterministic clocks, complete trainer graph spies, and injected failures at each ordered stage; compare snapshots, calls and errors with Rust.
2. Cover iteration/time combinations, first timed save, equality threshold, backward time, repeated hooks, fit reuse, failed-save FitEnd suppression, disabled and invalid latest modes, format/value failures and duplicate reserved metrics.
3. Exercise real directory/file effects, overwrites, copy bytes, symlink target/dangling-link behavior where supported, corrupt/truncated codec inputs and restoration failure order.
4. Run a real Trainer driver with live vessel/callback/logger state; save during IterEnd, modify all groups, read the file and restore before FitStart. Assert the source's two-iteration filenames and metadata. Do not replace training-state integration with a serialization-only round trip.
5. Run formatting, strict Clippy, focused differentials, the full workspace suite and stable/nightly exact per-file and total coverage gates. Require 100% production lines/functions/regions/branches without exclusions or weakened assertions.

Next action: implement concrete model-state adapters after evaluating their actual payload and loader behavior. Lossless text/path and native locale work now pass their full gates. Explicit locale snapshots and per-field providers compose with `thousands` grouping and the existing formatting engines. The native provider copies the calling thread's current dynamic MSVC UCRT locale with owned lifetime and mode restoration; it cannot observe another process or a separate static runtime. Ten real Python locales and a concurrent child-process contract verify acquisition. The unit filename adapter remains C-locale, while the localized adapter accepts the native provider explicitly. No production Python runtime bridge has been selected. The adapter uses incremental framing because candidate template engines fail required ordering and bracket-key cases. Primitive representation/Unicode properties reuse rustpython-literal and intl; numeric/string formatting reuses pinned pyformat-rs and rustpython-format. See [the formatting probe](../scripts/checkpoint-format-probe/README.md), [the model-state probe](../scripts/checkpoint-model-probe/README.md) and migration ledger for evidence and limitations. Bincode integration is still not Torch format compatibility.

### Concrete model-state evidence (2026-09-03)

An isolated Python 3.12.8 environment now runs real Torch 2.5.1 CPU, Tianshou 0.4.10,
Gym 0.26.2, NumPy 1.26.4 and SafeTensors 0.8.0. The unchanged Qlib PPO/DQN and
Recurrent/Attention bodies execute without importing the unrelated Qlib application
initialization graph. This is a test oracle, not a Python production dependency.

The Candle 0.11.0 CPU candidate passes 28 model-state cases covering 2,136 named
tensors: RNN/LSTM/GRU, one/two recurrent layers, PPO, DQN with/without target copies,
shared-name conflicts, and Attention's F16/BF16/F32/F64 states. Every roundtrip
preserves the tested tensor bits, dtype and shape; all 24 F32 policy cases reproduce
the original Python forward output after loading the Rust-written tensor file.
Twelve cases exercise shared tensors. Adam moments are initialized before snapshot;
the policy snapshot contains model tensors and module metadata, not the optimizer's
state object. `_metadata` and registration/sharing order are passed separately by
the fixture; SafeTensors alone is not a complete policy-state envelope.

Required adapter behavior established by executable probes:

- A clone/detached Candle tensor is not an owned early-stopping snapshot. Copy data
  for independent snapshots; keep shared live variables shared during restoration.
- Registration order wins when aliased names contain conflicting values, even if
  the incoming mapping has the opposite order. Do not sort loader assignments by
  tensor-file key order or deduplicate away observable assignments.
- PyTorch copies valid later parameters even when an earlier shape/missing-key
  error will make the load fail. Unexpected keys also fail after valid copies.
  Preserve these partial effects; the candidate's fail-fast `Var::set` loop is not
  sufficient as a production loader.
- PyTorch accepts F64 input into F32 parameters and strided tensor inputs in the
  tested cases. Candle `Var::set` alone rejects a mismatched dtype; an adapter needs
  explicit target-dtype conversion, not an unrequested change to the model dtype.
- Candidate native BOOL/I8 loading fails; U16 becomes U32. U8/I16/I32/I64 pass exact
  roundtrips. A generic state adapter cannot silently call these all lossless.
- Candle 0.11 source uses a standard-library API stabilized in Rust 1.87, so it has
  not been adopted under the current Rust 1.85 declaration. The probe uses the
  installed Rust 1.98. Evaluate a compatible version/toolchain contract before
  production dependency selection; do not silently raise the workspace MSRV.

Report: `C:\Users\andy\.codex\tmp\qlib-model-state-probe.json`. This establishes
model payload requirements and candidate capabilities, not Rust PPO/DQN inference,
training, all malformed-loader parity, arbitrary custom state, or legacy pickle
compatibility. Checkpoint and the full migration remain unfinished.

### Native adapter and tensor-name wire contract (2026-09-03)

The later production adapter uses pinned Candle 0.9.1 (CPU) and SafeTensors 0.4.5;
the earlier 0.11 candidate and MSRV comments above are historical. Its candidate
build/test passed with actual Rust 1.85; that is not a whole-workspace MSRV proof.
`CandlePolicyState` now implements the existing policy and checkpoint traits and
is exercised inside the actual Trainer graph/file save/restore integration.
The live fixture uses native tensor computation and shared variables, not vector
state, but its deterministic update is not a completed RL algorithm.

Tensor files written by this adapter carry SafeTensors string metadata
`core.candle_policy.tensor_names=prefix-p-v1`. Every tensor key is exactly `p`
followed by its original name, with no normalization. Decoding removes exactly
that first `p`, only when the marker is present. This preserves the valid Torch
parameter name `__metadata__` without colliding with SafeTensors' reserved field,
and remains injective for literal `p` prefixes, Unicode and NUL. Empty Rust keys
roundtrip as `p`; Python's own parameter-registration validation is separate.
Unmarked input files use literal names, including files with unrelated metadata.
Unknown marker values or any non-prefixed name in marked input are container
errors detected before copying parameters. Ordinary missing/copy/unexpected-key
failures still accumulate after other valid assignments in model registration order.

The wire marker is separate from the caller-owned module metadata in
`CandlePolicySnapshot<M>::metadata`; the native adapter does not yet execute
custom module-version hooks. A raw SafeTensors consumer must decode marked keys
before passing them to a model. No legacy Python checkpoint compatibility is implied.
Five focused Rust tests, 29 real model-state cases and two native live graph/file
integration tests pass; current full coverage results belong in the migration
ledger. Wider dtype support, custom state/hooks and complete model/training code
remain open and must not be hidden by these narrower results.

### Incoming dtype conversion versus native storage (2026-09-03)

Loading into an existing variable preserves that variable's dtype; it need not
reject every source dtype that the backend cannot store natively. The old adapter
incorrectly rejected I32/U16 input into F32 targets. The corrected loader preserves
BOOL/I8/I16/I32/U16 source values in a supported intermediate type and then uses
Candle's target-dtype conversion and assignment. I16 byte decoding reuses byteorder;
SafeTensors still validates the entire container before any model mutation.

A real Torch comparison covers 147 source/target/shape combinations, including
integer extrema, scalar and empty tensors. Snapshots contain destination dtypes,
not intermediate widening. This does not add native BOOL/I8/I16/I32/U16 variables,
nor support every remaining input type or arbitrary module hook. The probe checks
native initial types independently and retains those failures rather than calling
widened destinations compatible. The six focused tests and real model/cast matrix
pass; the latest full gates are recorded separately in the migration ledger.

### Scalar source-shape compatibility (2026-09-03)

For an existing scalar parameter, the policy loader selects the first element of
a one-dimensional input, even if that vector has multiple elements. The native
adapter now follows that rule through Candle tensor indexing before destination
dtype conversion. It does not broadcast scalars into vectors or flatten arbitrary
singleton matrices. Other shape mismatches remain ordinary accumulated failures.

An empty vector in this scalar path is different: Torch indexes before entering
its copy-error handler. The native adapter therefore returns an immediate `index`
error, retaining earlier assignments but skipping later ones and superseding any
previously accumulated missing/copy errors. This check precedes input dtype loading,
including for an empty unsupported U64 source. It does not add nonempty U64 support.

The production oracle compares 231 real Torch cases across shape, dtype, input
ordering and earlier/later failure effects. Eight focused native tests, strict
Clippy, ordinary workspace tests and full stable/nightly gates now pass. The
`scalar-shapes-unified.json` reports cover all 88 production files at exactly 100%:
15,755 lines, 1,997 functions, 20,567 regions and 1,488 branches. This accepts the
scalar-loading correction, not the unfinished policy consumers, wider native
dtypes, arbitrary custom state or full model/training migration.

### Policy weight retry consumer (2026-09-03)

Qlib PPO/DQN constructors invoke `policy.set_weight` after extracting a policy
state. The new runtime-independent `set_policy_weights` accepts an ordered
`PolicyWeights<Weight, Metadata>` map with shared Arc values and opaque metadata.
It loads once, retries exactly once only after `PolicyWeightLoadError::Runtime`,
and never interprets a diagnostic string to choose a retry. The conversion takes
the post-failure key list, then reads each value from the live map while inserting
`_actor_critic.` aliases. Existing names stay in place and prefix collisions can
affect values read later. A failed retry retains all preceding mutations.

The Candle backend shares the same parameter-copy implementation between tensor
file views and native input tensors. Additive `restore_typed` retains explicit
error categories while existing Trainer/string-error interfaces remain compatible.
The generic boundary does not require Clone for weights or metadata and is a
linked logical plugin seam, not a native dynamic ABI. A caller supplies logical
map order; arbitrary tensor-file order must not be substituted for it.

105 unchanged-source scenarios and 58 actual Qlib/Torch model cases pass, including
legacy PPO state names, failures on either load, alias identity and dictionary
collision effects. Current full gates are recorded in the migration ledger.
Both `policy-weights.json` reports now pass exact total/per-file checks for all
89 production files: 15,812 lines, 2,009 functions, 20,644 regions and 1,488
reported nightly branches are fully covered. Full workspace suites, strict root
and probe Clippy, formatting, and source/model comparisons pass. This accepts the
retry protocol and its native integration, not the entire Checkpoint migration.
Policy file extraction, complete constructors/inference/training, unsupported
native dtypes and arbitrary custom state remain separate unfinished requirements;
no legacy Torch file codec or production Python bridge is implied.

### Native policy file extraction (2026-09-03)

`rl_policy_checkpoint` composes the existing file/codec layer with an explicit
document projection. `DirectPolicyCheckpoint` moves a decoded policy unchanged;
`TrainerPolicyCheckpoint` moves `vessel.policy` from the complete decoded native
Trainer document. A missing vessel is an extraction error, while a present null
policy is a valid caller-owned payload. Read/decode failure prevents projection;
extraction itself neither restores the Trainer nor assigns model parameters.
Both codec and projection are replaceable logical plugins. No extra third-party
dependency or custom serialization implementation is needed.

The caller must configure the schema: the positional Bincode format cannot
reliably distinguish a pure policy from a full Trainer by trial decoding or file
suffix. A complete Trainer tail must decode before returning its policy. This
does not implement dynamic Python mapping auto-selection or a Torch/pickle codec.
The source fixture separately verifies the actual upstream decoder call with
`map_location="cpu"`, selection order, identity and failures. Native tensor DTOs
remain device-neutral until a concrete model adapter materializes them.

Focused tests and the actual Trainer-produced file integration pass. Complete
ordinary/stable/nightly workspace suites and exact total/per-file coverage gates
also pass: 90 production files, 15,841 lines, 2,014 functions, 20,671 regions and
1,488 nightly branches fully covered. Reports and commands are in the ledger. Native
ordered-weight materialization is still required before file extraction can feed
the retry consumer automatically: SafeTensors sorts physical keys, which cannot
recover the logical input order absent from older snapshot metadata. Full learned
models/training and the full migration remain unfinished.

### Ordered native weight payloads (2026-09-03)

New native snapshots supplement `prefix-p-v1` names with
`core.candle_policy.layout`, a JSON object containing `version: 1` and ordered
`entries: [[name, source_index], ...]`. A self index means an independent full
tensor; an earlier index denotes a full-tensor alias with identical dtype, shape
and raw bytes. Variable identity establishes aliases, never value equality. The
outer Bincode DTO stays unchanged. SafeTensors owns the actual tensor codec and
metadata serialization; metadata object-key order is not a canonical-byte promise.

`CandlePolicySnapshot::into_policy_weights` validates the entire layout, preserves
logical order, allocates CPU tensors and reuses Arc values for whole-tensor aliases.
It moves opaque module metadata unchanged. Earlier native snapshots may be used
with an explicitly supplied authoritative layout, but missing order or sharing
must never be guessed. An explicit layout cannot override conflicting stored data.
Existing direct parameter restoration remains compatible with older files.

The native full-file reader now composes with this method and `set_policy_weights`
in the real Trainer test. All 58 live Qlib/Torch retry comparisons also exercise
this production materialization step and compare final input alias groups as well
as order, errors and model bytes. Full coverage acceptance is recorded separately
in the migration ledger. Whole-variable sharing is not arbitrary overlapping or
strided-view parity; native unsupported source/destination dtypes and custom state
still need implementation. Existing cast-capable per-target restore behavior is
unchanged; the new Tensor-returning materializer does not silently widen inputs.

The ordered extension now passes full ordinary/stable/nightly workspace suites and
exact total/per-file gates: 91 executable production files, 15,944 lines, 2,025
functions, 20,837 regions and 1,510 branches, all covered. Its own 22 reported
branches are covered. This accepts the native whole-variable layout/materializer,
not the outstanding broader tensor representation or complete model/training work.
