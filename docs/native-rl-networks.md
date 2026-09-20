# Native order-execution networks

This is an implementation inside the ongoing Checkpoint migration slice, not a
claim that PPO/DQN training or the complete Qlib migration is finished.

## Implemented boundary

- `core::rl_candle_network` (Rust import `domain_core`) exposes
  `RecurrentFeatures`, `RecurrentConfig`, `RecurrentObservation` and the
  `CandleFeatureExtractor` trait. Configuration supports RNN/LSTM/GRU and multiple
  layers. Default hidden/output dimensions are 64/32, one GRU layer.
- `rl_candle_heads` supplies `PpoActor`, `PpoCritic`, `DqnModel = PpoActor` and
  unscaled `Attention`. DQN deliberately retains the source softmax output.
- Actor/critic share an `Arc` extractor and the same live parameter tensors;
  checkpoint registration retains both names without duplicating their variables.
  The unused `prev_rnn` bank remains registered because it belongs to source state.
- Feature extractors are replaceable in-process components. Candle tensors are a
  backend-facing API, not a stable dynamic plugin ABI. Existing Trainer checkpoint
  interfaces continue to exchange owned native snapshots and metadata.

## Third-party implementation and compatibility adapters

Workspace dependency `candle-nn = =0.9.1`, default features disabled, reuses the
  existing Candle 0.9.1 tensor runtime. License: MIT OR Apache-2.0. Native CPU is
  the tested backend; CUDA/Metal are not enabled or claimed. No production Python
  runtime bridge, Torch codec or hand-written numerical kernel was added.

The exact Candle source revision is
`cd96fa80da255e34f7b16b4ff98b6a31d557201b`. Its
[linear layers](https://raw.githubusercontent.com/huggingface/candle/cd96fa80da255e34f7b16b4ff98b6a31d557201b/candle-nn/src/linear.rs),
[recurrent cells](https://raw.githubusercontent.com/huggingface/candle/cd96fa80da255e34f7b16b4ff98b6a31d557201b/candle-nn/src/rnn.rs)
and native automatic differentiation do the main computation. Thin adapters handle:

- Qlib layer registration/order and multi-layer cell naming; state stacking uses
  `[batch, time, hidden]`, not the GRU helper's concatenated result.
- Torch linear initialization uses uniform bounds, not Candle's default normal
  initializer. Zero input dimension uses a zero initializer rather than an invalid
  zero-width random distribution. Cross-runtime RNG bitwise identity is not claimed.
- Recurrent feature padding, private time/position normalization, direction
  features, negative per-batch indexing and validation. Native tick/step inputs
  are explicitly I64 vectors. General Python object conversion is outside this API.
- Source actor/critic convert a Tianshou Batch using its deep-copying `to_torch`;
  native heads detach incoming observation tensors but keep model gradients.
- Empty matrices preserve zero gradients rather than silently missing gradients.
  Empty-slice reductions retain graph connections without evaluating NaN/Inf data.
- F16/BF16 linear and attention contractions accumulate in F32, then restore the
  model dtype. Native BF16 CPU GEMM is unavailable in this Candle release.
- ReLU preserves signed zero/NaN and Torch's zero derivative at zero, using native
  comparisons/selection. Candle's default ReLU has different boundary behavior.
- Reduced-precision softmax uses Candle forward operations plus a small backward
  hook. Torch saves rounded probabilities and rounds the gradient reduction to
  the source scalar type before subtraction. This behavior is explicit in the
  [Torch 2.5.1 CPU last-dimension kernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/cpu/SoftMaxKernel.cpp).
  The hook uses native Candle tensor operations, not a replacement arithmetic kernel.

Remove an adapter when a maintained third-party implementation demonstrably
provides the same behavior. Do not replace source behavior with library defaults.

## Numerical and state evidence

`scripts/checkpoint-model-probe/network_contract.py` executes unchanged upstream
class/function bodies after removing only app/relative import statements. It does
not modify the source checkout or deserialize external model files. The generated
contract is `crates/core/tests/fixtures/rl_candle_network.json`; ordinary Rust tests
consume it without requiring Torch at runtime. It records Torch version and source
SHA-256 hashes, all ordered weights/whole-tensor aliases, inputs, outputs and gradients.

Reproduce stdout from the repository root with the existing isolated oracle:

```powershell
& C:/Users/andy/.codex/tmp/qlib-model-oracle/Scripts/python.exe scripts/checkpoint-model-probe/network_contract.py
```

The 17 cases include six recurrent/actor/critic models (RNN/LSTM/GRU, one/two layers)
and eleven attention cases (F32/F64/F16/BF16, singleton broadcasting, zero query/key/
batch/output dimensions). Attention also checks intermediate values and gradients.
F32 uses absolute `1e-7` plus relative `2e-5`; F64 uses `1e-12` plus `1e-11`.
F16/BF16 values and gradients match exactly after decoding to their actual dtype,
including cancellation-sensitive key-bias gradients. These are fixture results,
not universal bitwise guarantees for all devices or tensor sizes.

Focused Rust tests also cover malformed configs/shapes/indices, missing parameters,
initialization, zero-dimension gradients, extractor failures and caller-state identity.
An actual shared GRU actor/critic is snapshotted; live parameters are changed and
both predictions change; ordered materialization plus `set_policy_weights` restores
the exact original predictions and shared parameter identity.

## Remaining acceptance and implementation

### Dense Adam update implementation (pending full acceptance)

`rl_candle_optimizer::CandleAdam` uses `candle-optimisers = =0.9.0`, with default
features disabled. Its MIT-licensed runtime reuses the pinned Candle 0.9.1 tree;
the selected dependency compiles on Rust 1.85. The published source revision is
`d9533bcd48f40c02a4c10221e4b2e7c0344e7793`.
The [published Adam implementation](https://raw.githubusercontent.com/KGrewal1/candle-optimisers/d9533bcd48f40c02a4c10221e4b2e7c0344e7793/src/adam.rs)
implements the moment/decay arithmetic. The project adapter:

- accepts the learning rate and coupled weight decay exposed by Qlib PPO/DQN;
- deduplicates live floating-point parameters by identity in first-seen order;
- creates one library optimizer lazily for each parameter receiving a gradient,
  and steps only present nonempty gradients, preserving independent update clocks;
- distinguishes absent gradients from zero/empty gradients when initializing state;
- updates rates for both already initialized and future parameter state;
- validates dense gradient shape/dtype/device before performing parameter updates.

The runtime rejects assigning an empty subtraction result which aliases its source
variable. An empty fixed-shape parameter has no values to update, so the adapter
retains its initialized state without making that invalid assignment. The private
third-party clock for empty variables is not advanced or exported; full optimizer
state serialization/introspection is not implemented. Do not claim full Torch
optimizer-state parity or bitwise optimizer resume from this update API.

Evidence is generated by `scripts/checkpoint-model-probe/adam_contract.py` and
`training_contract.py` using the existing real Torch oracle. The former checks
12 F32/F64 vector/scalar/empty cases over six updates, including missing gradients,
coupled decay and rate changes. The latter executes actual Qlib network bodies and
`chain_dedup`, then compares six shared RNN/LSTM/GRU models across four updates
(both heads, actor only, critic only, both heads again). Every named weight and
both predictions are compared, not just scalar loss or the number of parameters.

`crates/core/tests/rl_candle_training.rs` also exercises the public library API:
deterministically initialize a shared two-layer GRU, perform real backpropagation
and Adam updates, verify changed predictions, and restore exact policy parameters
from the native checkpoint. It deliberately does not reset or claim to restore
optimizer moments when restoring policy weights.

Full ordinary/stable/nightly workspace gates are required; see the current ledger
for terminal results and exact coverage counts. Previous 91-file coverage does not
cover these new modules. Do not mark this slice complete before its current gates.

### PPO L2 gradient clipping (pending full acceptance)

`CandleAdam::clip_grad_norm` implements the default L2 clipping operation used by
Tianshou PPO, before optimizer updates. Its input is the native gradient store and
its output is the total norm before clipping. It changes only present gradients,
counts shared parameters once and does not initialize Adam moments or change
parameters. Malformed gradient metadata is rejected before clipping starts.

The implementation composes existing Candle tensor operations because the pinned
Candle nn/optimizer APIs do not supply this Torch-compatible operation. It follows
the [Torch 2.5.1 source](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/nn/utils/clip_grad.py):
norm of individual norms, dtype promotion, epsilon `1e-6`, and an upper-only clamp.
F16/BF16 reductions accumulate in F32 and round each norm; mixed-precision scalar
multiplication preserves the coefficient in the gradient's arithmetic dtype until
the result is rounded. Negative limits and nonfinite values follow Torch defaults.
The operation itself accepts zero; the PPO learning loop must separately preserve
its source truthiness check (zero disables calling the operation).

`gradient_clip_contract.py` produces 37 finite and six nonfinite real Torch cases;
`training_contract.py --clip-gradients` produces six additional shared-model
trajectories (24 updates). Rust fixtures compare norms, clipped gradients, complete
weights and predictions. These do not yet constitute full PPO loss/trainer parity,
accelerator validation, arbitrary norm types or full optimizer-state serialization.

PPO/DQN policy construction, loss/update assembly and concrete default replay/
collectors are implemented in the sections below. Still required for the full
objective: full acceptance and remaining optimizer state/dtype behavior,
remaining model families and other Qlib modules. Broader
dtype conversion, arbitrary shared/strided state and custom module state/hooks also
remain. Legacy Torch file interoperability is a separate unresolved decision; it is
not inferred from implementing native differentiable networks.

## Native PPO prepared-minibatch loss (pending acceptance)

`rl_candle_ppo` supplies `PpoLossConfig`, `PpoLossInput` and `PpoLoss` for a native
discrete minibatch. Default configuration follows Qlib: epsilon .3, value clipping
and per-minibatch advantage normalization, value weight 1 and entropy weight .01.
Returns, advantages and previous predictions must come from the rollout-processing
stage; this function does not implement GAE, buffer management or the training loop.
The typed boundary accepts probabilities [batch, actions], integral actions [batch]
(integer or floating dtype, including the critic dtype used by source preprocessing),
and consistent floating [batch] fields for the remaining prepared data. Old values
are optional and inspected only when value clipping is enabled. Arbitrary Python
object conversion and complete Batch orchestration remain separate work.

The implementation reuses Candle tensor operations and differentiates the actual
policy/value/entropy loss. Adapters preserve Torch's probability renormalization,
dtype-epsilon clamping, F32 ratios, promotion against advantages, unbiased standard
deviation with no epsilon, scalar-clamp boundary gradients, min/max ties and NaN
gradients. Zero loss weights keep zero parameter gradients (not absent gradients),
which matters for Adam weight decay and state initialization.

Evidence: `ppo_loss_contract.py` invokes actual Tianshou `PPOPolicy.learn()` for
24 F32/F64 cases and records scalar components plus actor/critic gradients,
including missing/unused integer old values without clipping and floating actions.
`ppo_training_contract.py` invokes unchanged Qlib `PPO.learn()` for six shared
recurrent models and 24 updates. Tests compare all source state entries, aliases'
values, both predictions, losses, clipping norms and initialized state counts.
The public integration uses PPO loss before backward/clipping/Adam and checkpoint
restoration. Full reduced-precision loss parity and complete PPO lifecycle behavior
are not implied; current full coverage remains mandatory and unaccepted.

## Episodic returns and GAE (pending acceptance)

`rl_episodic_return` supplies native F64 arrays for returns and advantages from
rewards, current/next estimates and replay-buffer metadata. It reuses ndarray and
the existing NumPy is-close compatibility helper. Policy-specific recurrence is
local; reward-statistics normalization and final policy-dtype casting are separate.

Termination masks next-state bootstrap values; truncation ends the recurrence but
retains a valid next-state estimate. Unfinished buffer indices stop continuation
without changing input order. Missing current estimates use the source whole-array
roll; missing next estimates require lambda close to one. Multiplication by zero
retains source NaN/Inf behavior rather than acting as a conditional reset.

The real ReplayBuffer/Numba oracle has 48 cases, including ring buffers, repeated
indices, empty arrays and optional values. Recorded finite/infinite F64 bits and
NaN positions match. Public native integration now runs GAE → PPO loss → backward
→ clipping → Adam → policy checkpoint restore. This is not a complete collector,
replay buffer implementation or PPO lifecycle.

## Return normalization (pending acceptance)

`rl_candle_returns` prepares detached targets from evaluated F32/F64 critic
tensors and replay metadata, using Candle for tensor ownership/arithmetic/casts
and ndarray for arrays and two-pass batch statistics. `ReturnStatistics` holds
scalar return mean, variance and count. The small source-ordered parallel merge
is a compatibility adapter; no new general-purpose statistics engine is added.

The old variance plus `1e-8` is used to unnormalize estimates and normalize the
resulting returns; the mean is not subtracted. Only then are running statistics
updated from unnormalized F64 returns. Original critic values remain unchanged
for value clipping, and advantages are not divided by the running scale. F32
estimates round before the F64 GAE recurrence. NumPy 1.x promotes estimates to
F64 for finite scalar scales at or above `3.4e38`; Inf/NaN scales retain F32.

`policy_return_contract.py` invokes actual Tianshou `_compute_returns` and
`RunningMeanStd`, with a real ReplayBuffer and critic evaluation under no-grad.
It covers 22 scenarios, each with three sequential calls: F32/F64, normalization
on/off, ring/normal buffers, constant rewards, NaN/Inf, nonzero initial mean/count,
very large and infinite variance. Rust comparisons require exact F32 output bits,
matching NaN positions, and F64 error at most `1e-14 + 1e-13 * abs(expected)` for
different reduction orders. Tests also check detached targets, flattened views,
invalid input without statistics mutation, empty statistics and count overflow.
The public training test uses normalized GAE targets before PPO/backward/clipping/
Adam and verifies policy checkpoint restoration, not optimizer-state restoration.

Critic minibatch evaluation, old log-probability capture and repeated learning
are now assembled in `rl_candle_policy` (see below). This CPU F32/F64 stage does
not establish low-precision or accelerator parity; concrete collector assembly,
broader APIs and all coverage gates remain required.

## Categorical forward and minibatches (pending acceptance)

`rl_candle_categorical::CandleCategorical` supplies probability normalization,
dtype-clamped log probabilities, broadcasting integral action inputs, per-batch
entropy and sampling. `PpoActor::policy_forward` evaluates the existing head and
returns raw actor output (`logits` in source), I64 actions, the unchanged caller
state and its distribution. `PpoLossConfig` reuses this same distribution logic.

Normalization is checked after division, so all-negative input weights can yield
a valid distribution. Deterministic evaluation uses RAW actor argmax; this differs
from normalized probability argmax on those inputs. Training always samples, even
when the caller only needs log probabilities. Sampling uses the existing rand
0.9.2 `WeightedIndex` and caller-owned RNG, preserving zero-weight support and
native seeded reproducibility, not Torch/NumPy's exact random-number stream. The
existing Candle Gumbel-softmax sampler uses different logits/temperature/RNG
semantics, so it is not substituted for probability-weighted sampling.

A narrow native autograd hook preserves Torch's division-backward operation order
for normalization. Candle's generic denominator derivative rearranges arithmetic,
which produced a one-ULP F16 gradient difference. The hook composes existing Candle
operators and returns an already computed owned tensor in forward; it is not a
handwritten arithmetic kernel. Formula source: [Torch 2.5.1 division backward](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/csrc/autograd/FunctionsManual.cpp).
The adopted hook currently provides a CPU forward implementation; accelerator
support and full reduced-precision PPO training remain unverified.

`categorical_contract.py` invokes real Tianshou `PGPolicy.forward` (inherited by
PPO) and Torch Categorical. Twenty F16/BF16/F32/F64 cases compare distributions,
deterministic actions, log probabilities, entropy and gradients; six source failure
cases cover invalid inputs. Reduced-precision results are bit-exact; F32/F64 use
the existing strict network tolerances. Native tests check empty batches, illegal
actions, state identity, RNG consumption, deterministic replay, 30,000 samples
within six binomial standard deviations and exact zero-probability support.

`rl_policy_batch::minibatch_indices` reuses rand shuffling and standard slice
chunking, preserving one permutation per call and merging a short final batch.
The real `Batch.split` oracle covers 72 ordered cases; shuffled tests verify
permutation preservation, native seed repeatability and RNG state. Observation
`select_batch` uses Candle indexing consistently across all seven fields; tests
check order, duplicates, gradients, empty selections and each field's failure.
Public GRU integration now checks ordered minibatch predictions, policy forward,
shared distribution log probabilities, float actions, normalized GAE, PPO loss,
clipped Adam updates and exact policy checkpoint restoration. These building
blocks feed the native PPO implementation below. Concrete replay/collector
assembly and all acceptance gates remain required for the Checkpoint slice.

## Native PPO process, learn and update (pending acceptance)

`rl_candle_policy::CandlePpo` constructs the existing shared-extractor actor and
critic plus deduplicated `CandleAdam`. `CandlePpoConfig::new(lr)` follows Qlib's
wrapper defaults; it checks source gamma/lambda, dual clipping and reward/value
normalization constraints. This is a discrete Candle backend, not a production
Python bridge or stable native plugin ABI. No new dependency was added.

`process` evaluates critic current/next observations in interleaved ordered
minibatches, prepares detached normalized returns, converts actions to the critic
dtype/device, and captures detached old log probabilities in a second ordered
pass. Policy mode is preserved, including sampling in training mode. Running
statistics already updated before a later failure remain changed.

`learn` shuffles once per repeat, merges the short last minibatch, samples actor
forward, evaluates critic, runs the shared PPO loss/backward, optionally clips
gradients (None/zero disable it), and steps the third-party Adam adapter. It returns
all four source loss lists in order. Optional advantage recomputation refreshes
critic values/returns/advantages after the first repeat, retaining old log
probabilities; normalization statistics update on each recomputation. Zero repeats
return four empty lists without even validating the minibatch size.

`update` accepts in-process replay and optional scheduler adapters. Sampling
precedes `updating=true`; process, learn, optional unchanged replay-weight update,
and scheduler then run in source order. Success clears updating; later failure
leaves it true. Missing buffers return an empty mapping without touching existing
state. These failure semantics are intentional source compatibility, not an
automatic-reset bug. Ordinary replay adapters inherit a no-op priority hook.

Evidence: eight actual Qlib PPO + Tianshou process/learn fixtures (F32/F64 × return
normalization × advantage recomputation), each with three real dense Adam updates.
Tests compare all prepared targets, old log probabilities, running statistics,
loss lists and every parameter using existing F32/F64 tolerances. Numerical source
trajectories use full-batch updates to avoid claiming equivalent Torch/NumPy/rand
shuffle streams. Separate native tests check merged 2/3 minibatches, call order,
zero repeats, mode/RNG/state identity, typed errors and 14 actual BasePolicy.update
success/failure cases. A public two-layer GRU integration exercises the assembled
policy across two shuffled repeats and restores exact policy predictions.

The sections below add default replay/collector assembly, native constructor
weights/state registration and DQN/n-step updates. Still unfinished: stacked and
broader replay storage, complete file/Trainer extraction adapters, optimizer-state
restoration, broader dtype/device contracts and exact total/per-file coverage.
Restoring policy weights is not full optimizer
or random-state training resume. The single Checkpoint slice remains in progress.

## Training-vessel PPO adapter (pending acceptance)

`rl_candle_vessel::CandlePpoVessel<R>` owns a concrete PPO, caller-provided RNG and
optional scheduler. It implements the existing `TrainingRunPolicy<Buffer>` for
native replay adapters, so vessel updates now execute the real native optimizer.
Forward and update reuse the same model and RNG. Ordered loss lists are converted
to Arrow Float64 arrays (source Python floats), not pre-reduced means. Existing
vessel logging owns metric reduction. No dependency or stable plugin ABI changed.

The default `TrainingUpdateOptions` JSON adapter keeps source argument timing.
Duplicate `sample_size`/`buffer` keywords fail before sampling, including with a
None buffer. Other learn options are resolved after preprocessing through the
shared `update_with_options` lifecycle. Both learn arguments must be present even
for zero repeats, but zero/negative repeats do not inspect the batch-size value.
Booleans act as integers; floats are not silently coerced to integer loop counts.
Positive floating batch sizes consume one native shuffle before the source-like
integer error; invalid comparisons/nonpositive sizes fail before shuffling.
Unrelated kwargs are ignored. Integers beyond native indexing are explicit errors;
this JSON adapter does not represent arbitrary Python objects or custom __index__.
The existing generic vessel interface still permits separate opaque-value adapters.

`ppo_vessel_contract.py` records 56 actual Qlib/Tianshou update calls, testing all
those cases with and without buffers, final updating/statistics/optimizer state,
metric-list lengths, and whether NumPy permutation state advanced. Rust checks
native RNG state against real preprocessing plus the same native shuffle; it does
not equate Torch/NumPy and rand streams. Preserve raw oracle JSON numeric spelling:
rewriting `1.0` as `1` changes the keyword contract. Public GRU integration calls
the real TrainingRunPolicy update and checks finite ordered Arrow loss arrays.

Default collector factories/environment action routing are implemented below.
Stacked replay storage and the remaining checkpoint/model requirements are still
unfinished; this adapter alone is not full migration acceptance.

## Concrete single-stream replay (pending acceptance)

`rl_candle_replay::CandleReplayBuffer` stores detached owned Candle tensor rows
and implements `CandlePpoReplay`. It supports full-history recurrent observations,
separate next observations, scalar F64 rewards and separate terminated/truncated
flags. The current default is `stack_num=1` with stored next observations;
frame-stack storage, omitted-next reconstruction and arbitrary info/policy
metadata still need implementation. CPU F32/F64 observation and I64 index/action
contracts have real-source evidence; that does not establish every dtype/device.

`rl_replay_index::ReplayIndex` provides source physical-ring positions, episode
statistics, terminal-aware previous/next positions and frame-availability filtering.
It preserves chronological all-sample order and uses existing rand distributions
for caller-owned replacement sampling. ndarray owns done flags. Initialized
storage survives reset, including zero-valued unwritten slots and empty-read shape
metadata. Shared zero templates use copy-on-write rows, never whole-capacity copies
for each insertion. Tensor copying/broadcasting/batching uses Candle, not custom
numeric storage kernels or an extra replay dependency.

Insertion advances index/episode state before tensor assignment. Torch-backed
source storage rejects dtype mismatches rather than casting; native assignment
does too. Failures preserve preceding changes. Native field assignment order is
explicit (observation, next observation, action, reward, flags); Python's arbitrary
Batch top-level hash iteration order is not an established failure-order contract.

Real Tianshou fixtures cover 216 metadata transitions, 21 complete tensor storage
insertions and two dtype failures. Tests also verify detached snapshots, duplicate
reads, reset/empty shapes, replacement sampling and invalid inputs. Public GRU
integration now populates this buffer, then invokes actual TrainingRunPolicy PPO
updates over two shuffled repeats and checks retained replay rows and ordered
finite Arrow loss arrays. It no longer substitutes a one-shot test-only buffer.
Full coverage remains a required failing/unverified gate, not waived by these tests.

## Concrete vector replay (pending acceptance)

`rl_candle_replay::CandleVectorReplayBuffer` implements `CandlePpoReplay` for
multiple environments. `CandleReplayBatch` borrows tensor and scalar columns;
omitted environment IDs select every child in order. Rounded capacity gives each
child an equal-sized ring. Both single-stream and vector implementations reuse
the same private row storage, including one retained schema/zero template and
detached copy-on-write rows. Initializing any child initializes the schema for
unwritten children too, without allocating a separate incompatible tensor schema.

Batch insertion advances all selected child indices and episode statistics first,
including repeated IDs. It then broadcasts/validates and copies a complete tensor
field before committing its rows, in the documented native field order. Errors do
not roll back earlier index/field writes. Empty explicit IDs initialize the schema
before an index error, matching the source's empty non-integer index array. Default
storage is still one full-history observation and stored next observation, not a
frame-stack or prioritized implementation. Native IDs are nonnegative usize values.

All-data sampling concatenates each child's chronological contents. Positive
sampling uses rand WeightedIndex over active child lengths, then shared native
uniform replacement sampling within each child; returned rows remain grouped by
child. No cross-runtime bitwise NumPy/rand RNG equivalence is claimed. Global
previous/next queries wrap over capacity but never cross child episode boundaries.

`vector_replay_contract.py` runs actual Tianshou VectorReplayBuffer and records
21 batches / 41 insertions, whole physical tensor reads, reset statistics, global
navigation, four failures and the two-phase NumPy sampling call trace. Native tests
also check 30,000 statistical draws, caller RNG sequencing, scalar/batched tensor
broadcasting, detached snapshots and failures across all selected rows. Public GRU
integration now trains through both concrete single and two-environment vector
replays, including chronological regrouping and an unfinished child episode.

This buffer is not an optimizer-state checkpoint. The synchronous collector below
now supplies typed action routing and factory integration. Complete coverage and
broader replay/model requirements remain required; see the migration ledger for
exact current evidence.

## Synchronous native collection (pending acceptance)

`rl_candle_collector::CandleCollector` collects full-history observations from
`FiniteVectorEnv<RecurrentObservation, i64, f64, I>` into the concrete vector replay.
Its linked `CandleCollectionPolicy` trait adds action inference, optional exploration
and environment-only action mapping to the existing training policy interface.
`CandlePpoVessel` implements it with actual Candle forward/distribution sampling.
The same owned policy is borrowed for collection and learning; no downcast, copied
policy, Python production bridge or Torch whole-graph serialization is introduced.

`TrainingVesselRunner::with_policy` and the optional policy type parameter on
collector/factory interfaces retain this extended interface through orchestration.
The original `new` constructor retains its dynamic mode/update default. Implementors
of that default collector/factory interface now need an explicit `'static` trait-object
bound in their `&mut (dyn TrainingRunPolicy<...> + 'static)` signatures; the borrow
itself remains temporary. `TrainingCollectorBuildResult` names the returned boxed
collector result. These are linked Rust extension points, not a stable dynamic ABI.

No supplied buffer means a one-slot-per-environment vector replay, including
evaluation. Supplied replay data is retained. Collection writes original/noisy
actions to replay and sends mapped actions only to the environment. Finished
environments reset after replay insertion; episode collection trims surplus ready
rows after reset and performs one final all-environment reset before returning.
Step collection can exceed a non-multiple request and continues cached observations
on the next call. Environment exhaustion reaches the existing vessel guard, which
suppresses it and skips subsequent learning/logging. Other errors retain preceding
environment/replay effects; statistics update only after the collection loop.

`CandleTerminationInfo` supports Boolean JSON `TimeLimit.truncated`, unit info and
the existing log-only `RlLogInfo` (no top-level truncation flag). An adapter-owned
info type can implement it without JSON serialization. Arrow carries ordered
episode arrays and ndarray computes population mean/std; rand, Candle and chrono
provide sampling, tensor/model operations and the replaceable clock. No new crate
is needed. The native glue implements source ordering that these dependencies do
not provide, rather than replacing their general-purpose algorithms.

The actual Tianshou collector fixture covers 18 scenarios / 22 snapshots, including
continued collections, step overshoot, episode trimming, noise/map hooks and seven
failure stages. Additional Rust tests exercise real GRU training, train/validate/test
exhaustion with unchanged weights, replay and reward failures, empty episode metrics,
retained buffers, clock behavior and checked native counter errors.

Still unfinished: arbitrary recurrent hidden-state/policy metadata, preprocess/random/
render/async collector options, broader replay storage modes and priorities, collector
persistence, broader DQN contracts and other remaining models/modules, and exact coverage acceptance.
Policy-weight snapshots do not restore optimizer, replay or random state. This
implementation advances Qlib's default full-history path without declaring the
complete Tianshou collector surface or the full Qlib migration finished.

## PPO state registration and constructor weights (pending acceptance)

`CandlePpo::policy_state(metadata)` registers all parameters exposed by its native
actor/critic, in source state-dict order: actor, critic, then both repeated below
`_actor_critic`. Shared extractor and repeated module names retain the same live
variable IDs; owned snapshots preserve those aliases without relying on equal
values to infer sharing. Public real-GRU integration now uses this registration
instead of manually assembling an incomplete actor/critic-only dictionary.

`CandlePpo` implements the existing `PolicyWeightLoader`, and
`CandlePpo::new_with_weights` constructs the model and deduplicated optimizer before
applying caller-decoded weights through `set_policy_weights`. The one-runtime-error
retry appends the source legacy aliases to the caller's input map. Invalid shapes
and missing keys retain the source partial-copy and retry-map mutations. The model
state's metadata can differ from the input metadata type; `load_native_weights`
borrows and does not clone, coerce or interpret opaque input metadata.

The actual Qlib fixture `ppo_weight_contract.py` records complete, legacy, missing,
shape-invalid and conflicting-alias input cases, all final values/key order/shared
identities, and a constructor call that reads weights after creating Adam. Its file
reader is a recording boundary stub; it does not establish Torch-file decoding.
Native tests compare state and constructor loads, retry/error mutations and
snapshot materialization; a restored live prediction checks the computation path.

This covers parameters exposed by the current native feature-extractor interface,
not arbitrary Python module buffers or custom load hooks. File-format dispatch and
Trainer checkpoint extraction remain external adapters. Optimizer moments, return
normalization statistics and RNG state are not part of a policy state dict and are
not restored by these methods. Existing exact coverage gates remain mandatory.

## Native n-step targets (pending acceptance)

`rl_candle_nstep::{CandleNStepReplay, CandleNStepBatch, CandleNStepConfig,
prepare_nstep_returns}` implements the return-preparation dependency for DQN.
Both concrete single and vector replay implement the read-only interface. Scalar
columns read existing physical storage directly, without materializing observation
tensors; next-index navigation reuses each buffer's terminal-aware ring behavior.

Preparation reads rewards, walks next indices, evaluates detached target Q values,
applies the termination bootstrap mask, marks unfinished entries as reward-horizon
endpoints, and computes the reverse recurrence. ndarrays perform array operations
and Candle reshapes/casts the result and existing replay weights. Returns are
committed before weight conversion. True termination removes bootstrap; truncation
and unfinished endpoints retain it with a shortened discount horizon. Gamma powers
are multiplied sequentially; NaN/Inf arithmetic is not sanitized by zero masks.

Target outputs flatten to `[batch, remaining_elements]`, including rank-one input;
this follows observed source behavior rather than its same-shape docstring. F32,
F64 and I64 cases are verified. Source BF16 NumPy conversion and F16 Numba failures
are explicit dtype errors at the corresponding stage. Reward normalization is
unsupported as in source; the native positive-horizon API rejects zero explicitly.
The callback borrows replay read-only and may update its own policy state. This
does not promise callbacks which mutate replay through aliases or a stable ABI.

`nstep_contract.py` records 108 actual-source combinations across single/vector
ring overwrite, scalar/multi-axis Q shapes, dtypes, horizons and discounts. Tests
compare physical metadata, terminal callback indices, return values/dtypes/shapes
and converted priorities, plus adapter failure order, unchanged batch fields on
failure, malformed metadata and detachment. The public two-environment GRU test
also evaluates the actual softmax actor (the Qlib DqnModel network), computes
three-step targets and checks terminal/unfinished results before PPO updates.

This is target computation, not policy orchestration by itself. The numerical
dependencies and native policy/vessel assembly below now implement the default
DQN path. Existing full-workspace/per-file coverage requirements and the wider
compatibility limitations still apply; this is not full migration acceptance.

## Native DQN numerical and prepared-batch learning boundary (pending acceptance)

`rl_candle_dqn` provides masked forward selection with unchanged raw logits/state,
Double/Nature target selection, signed action indexing, weighted MSE or delta-one
Huber loss, one prepared-batch Adam update, and epsilon-greedy exploration. It
reuses Candle tensors, automatic differentiation and the existing `CandleAdam`,
plus ndarray/rand for exploration. No dependency or lockfile changes were needed.
Candle-nn 0.9.1 supplies unweighted MSE but no Huber function; source-specific
weight broadcasting and Huber use existing tensor operations. A small custom
autograd hook preserves finite saturated Huber derivatives at infinities and NaN
derivatives at NaN, which generic inactive squared branches otherwise corrupt.
This hook uses the current CPU backend; accelerator execution is not claimed.

Mask penalties use a global min/max, not per-row extrema. Numeric broadcast masks
are supported and all-masked rows still select an action. First maximum/first NaN
selection matches the source. Double DQN gathers target logits at online masked
actions; Nature DQN maximizes raw target logits regardless of the observation
mask. Targets are detached by n-step preparation, not by these numerical helpers.

Mask subtraction preserves the source NumPy 1.26 array dtype before conversion
to the logits dtype. U8/U32/I64 arrays use modular integer subtraction, while
F16/F32/F64 arrays retain their rounding boundary. Zero-dimensional NumPy masks
have different promotion rules: unsigned masks become signed and F16/F32 become
F64 before subtraction. BF16 uses Torch tensor semantics because NumPy has no
BF16 representation. The native Tensor mask argument models these default
NumPy semantics; it does not separately encode arbitrary Torch-versus-NumPy
provenance for zero-dimensional masks.

`dqn_mask_contract.py` / `rl_candle_dqn_masks.json` characterize 88 actual-source
cases across seven mask dtypes, four logit dtypes, scalar/row/column broadcasting
and exhaustive U8 values. Rust compares Q values bit-for-bit (including infinity
and signed zero), actions, unchanged raw logits/state and non-contiguous masks.
The small integer compatibility adapter uses std modular subtraction through
the existing num-traits crate: Candle 0.9.1's ordinary integer subtraction can
panic on overflow in debug, and conversion through F64 loses I64 bits. This
adapter materializes integer masks on the host and reconstructs their original
shape/device; floating masks stay in Candle operations. No new numerical engine,
dependency or production Python bridge is added. Accelerator performance and
full reduced-precision training remain outside the evidence of these tests.

Learning pops existing batch weights before model evaluation and signed action
indexing, reads returns afterward, computes loss, publishes signed TD errors, and
then performs backward/Adam. Errors retain those preceding mutations. Returns
flatten/cast to the selected Q tensor. Weighted MSE preserves broadcast shape:
column priorities produce an outer product, and scalar zero weights retain zero
gradients and Adam weight decay. Huber ignores weights even when malformed.
The helper does not itself synchronize targets or advance a policy iteration.

Exploration skips every draw when `abs(epsilon) <= 1e-8`; otherwise it draws a full
row-selection vector followed by the random score matrix and adds the mask. A
caller-owned rand generator avoids global RNG state. Exact NumPy seed-to-bitstream
identity is not claimed; actual-source draw injection verifies ordering, threshold
behavior and resulting actions without replacing the source policy methods.

`dqn_contract.py` and `rl_candle_dqn.json` capture 24 inference combinations, 28
three-step learning sequences (84 actual Adam updates), and 14 exploration cases
from Tianshou DQN. F32/F64 values, loss, signed priorities, gradients and every
updated logit parameter are compared. Whole-fixture live regeneration matches.
Native boundary tests cover empty/nonfinite inputs, signed-index extremes,
malformed masks/batches and mutation order. The public integration additionally
uses the real full-history GRU, concrete replay, three-step targets and Adam,
checks live predictions change and initializes optimizer states only for
parameters with present gradients. This is not a complete DQN policy or a claim
of reduced-precision learning parity, arbitrary model buffers, full-training
resume or exact coverage acceptance.

## Native DQN target model, policy and vessel (pending acceptance)

### Shared training/checkpoint ownership

`CandlePpo` and `CandleDqn`, and their respective vessel runtimes, implement
`TrainingPolicyState<CandlePolicySnapshot<()>>`. Registration uses live variable
handles, never mutable copies made from constant tensors. PPO registers current
actor/critic first, followed by the original `_actor_critic` container's handles.
As in Tianshou, replacing a public actor/critic does not rebind that container or
the existing optimizer. A snapshot can therefore contain distinct current and
original heads; restoring it visits both in registration order. Unsupported
constant replacement parameters fail before copying any checkpoint values, with
the offending registration name in the error. This is not a frozen-parameter
restriction: a native live Var remains writable regardless of which optimizer
currently references it. The default
native snapshot uses unit application metadata; callers needing custom metadata
can still use `policy_state(metadata)` directly. Custom Torch module-version
hooks are not supplied by this native default.

`TrainingVesselRunner::state_dict/load_state_dict` delegate to the same owned
policy used by collection and learning and preserve the existing `{policy: ...}`
envelope. The runner also implements `RlCheckpointState` for direct use in the
existing ordered Trainer graph traversal. These methods need no assigned Trainer
or active collector. Save/load failures retain their existing typed categories.
Loading is strict parameter restoration, not the constructor helper's legacy
weight-name retry; valid preceding copies on a later load failure remain applied.

Real GRU collector tests persist the native envelope with Bincode, perform actual
PPO/DQN learning, reject a damaged tensor container without changing parameters,
and restore the same live variables. PPO sharing and independent DQN targets are
retained. This is a native file round trip, not Torch pickle interoperability.
Separate post-learning tests retain optimizer initialization, caller RNG,
updating/action-count flags, DQN iteration/epsilon and PPO return statistics.
`policy_checkpoint_contract.py` independently verifies the corresponding source
scope, including unchanged actual Torch Adam state tensors after parameter load,
and retained original container/optimizer references after actor/critic replacement.
No complete optimizer/RNG/replay resume guarantee is added: Qlib's concrete policy
`state_dict` does not promise it. Full Trainer lifecycle integration and exact
coverage acceptance remain tracked separately.

`rl_candle_dqn_policy::CandleDqn` creates Qlib's softmax model and the existing
native Adam, validates gamma/horizon, and optionally constructs an independent
target. Positive target frequency enables it; zero/negative frequencies disable
it. Frequency two synchronizes before learn calls 0/2/4, after their n-step target
preparation but before loss evaluation. Successful optimizer updates advance the
counter. A native u64 overflow fails afterward, retaining preceding mutations.

`CandleFeatureExtractor::rebuild` is an explicit linked plugin capability.
`RecurrentFeatures` reconstructs its existing architecture around supplied copied
parameters. Unsupported custom extractors fail explicitly when targets are
enabled. `PpoActor::independent_copy` copies each live parameter identity once,
including aliases across extractor and head, and checks that reconstruction
retains those exact new handles. Existing variables are detached before Candle
Var construction, which copies storage; simply cloning an existing Var would
share it. Constants stay constants. This is parameter/model reconstruction, not
Torch whole-graph serialization. Whole-tensor aliases are supported, not arbitrary
views of shared storage or opaque Python module state.

The feature `set_mode` hook defaults to no-op for native recurrent layers (which
have no dropout/batch normalization state). Target reconstruction enters evaluation
mode. Later policy train/evaluation changes affect only the online extractor.
Custom rebuilders must independently copy any other mutable model-local state;
parameter-handle validation is not a verifier for arbitrary plugin internals.

The policy preserves first-forward action-count caching, online-before-target
inference, mask processing for each model, detached n-step targets, source
sync/priority/optimizer ordering, and epsilon draw behavior when action count has
not yet been discovered. State registration includes `model` then `model_old`
when present. Native constructor weights reuse the existing Qlib one-runtime-error
retry, including partial-copy and input-key mutations; they do not restore Adam,
epsilon, iteration, replay, normalization or RNG state or read Torch files.

`CandleDqnReplay` extends n-step metadata with sampling, next-observation access
and priority callbacks. Existing single/vector buffers implement it. Their default
storage has no masks or priorities; specialized linked adapters can supply those
fields, but this does not implement prioritized storage itself. `update` performs
sample -> updating -> process -> learn -> priorities -> scheduler, and only
successful completion clears updating. None replay returns an empty mapping
without clearing a previous failure state. Scheduler reuses the existing Adam
scheduler interface instead of duplicating it.

`rl_candle_dqn_vessel::CandleDqnVessel` supplies training-vessel and concrete
collector action/noise routing using the same policy/optimizer/RNG. DQN ignores
extra learn keywords and returns a scalar metric; it does not require PPO's
batch_size/repeat. Duplicate sample_size/buffer keywords still fail before update.
Current full-history collector observations do not carry action masks. The real
GRU integration collects five transitions across two environments, trains through
the vector replay/vessel, changes live online weights, and leaves the independent
target at its pre-update synchronized weights.

`dqn_policy_contract.py` / `rl_candle_dqn_policy.json` record 16 actual-Qlib
four-update trajectories (64 updates), six lifecycle failure/success cases and
six full/missing/shape weight-load cases. Tests compare every online/target state
value, target returns, TD priorities, loss, mode, counter and updating flag, plus
source retry mutations and target-gradient absence. Native tests additionally
cover plugin reconstruction, cross-extractor/head aliases, typed overflow,
unsupported normalization, scheduler/replay/keyword failures and real collector
assembly. Full/per-file exact coverage, broader storage/mask/hidden-state/device
contracts, remaining model families and the rest of Qlib still require work.

## Real policy Trainer/file lifecycle verification

`tests/support/rl_candle_trainer_checkpoint.rs` connects the real GRU/PPO runner
to `RlTrainerDriver`, an ordered observer and checkpoint callback, native Bincode
file storage and a shared `RlLogWriter`. The finite environment forwards actual
reset/step/reward/done events to the logger. Two training iterations each collect
five transitions/three episodes and perform two full-batch gradient updates.
The test checks `001.pth`, `002.pth` and latest state, then restores the first
file into the same live runner. Before resumed `FitStart`, all registered model
parameters, named callback state, logger history and trainer iteration match the
saved document. Resumed learning changes the original watched parameter without
replacing its identity or rewriting the first checkpoint file.

Full batches deliberately include both reward trajectories: source PPO divides
by minibatch advantage standard deviation without epsilon, so identical short
minibatches can produce nonfinite training state. This test does not alter that
source behavior or assert uninterrupted-run equivalence after a policy-only
restore (Adam and RNG state are not restored). Its seed/environment wrapper is
a test consumer of the linked traits, not a general production assembly factory
or an external plugin ABI. Native `.pth` test names do not imply Torch encoding.

Reserialized snapshots are compared by complete metadata-map contents and every
named tensor's dtype, shape and exact data bytes. SafeTensors metadata JSON key
order is not canonical; signed-zero bit changes, renamed tensors, dtype/shape
changes and changed alias-layout metadata must still fail comparison. The test
separately checks that the original file itself is not rewritten byte-for-byte.

`DataQueue<T>` now implements `RlTrainerSeedContext` directly. Entry activates the
existing producer; entry failures are reported at `seed_enter`. Exit cleans up
and returns false, so the original phase error is not suppressed. The real-policy
test consumes an actual queue seed to configure its environment, replacing the
former no-op context. General configured simulator/seed/environment assembly is
still unfinished; this adapter does not add a second queue implementation.

The same context protocol also supports `Arc<Mutex<DataQueue<T>>>` when multiple
owned in-process environments share one stream. Poisoned entry/exit locks are
reported at `seed_enter`/`seed_exit`, not silently recovered. The owner remains
responsible for entering before consumption and exiting after the phase.
`EnvironmentSeedSource::try_seeded` accepts standard iterators of fallible values:
ordinary read errors preserve the previous simulator/status and allow retry,
while end-of-stream or explicit `StopIteration` generates the sentinel and makes
the environment terminal. This is not subprocess queue transport; application
factories still connect their typed seed stream to their simulator constructors.

The collector now accepts `EnvironmentStepInfo<Observation, Action, Aux>` directly.
Like Qlib's wrapper, this type has no top-level time-limit flag; nested auxiliary
fields do not change termination into truncation. The finite log writer also
accepts this actual environment info type using `EnvironmentLogValue` as its
opaque payload: scalars remain numeric metrics, while debug observations/actions
retain their original types. Log order and levels are preserved, and the borrowed
step's auxiliary information is untouched. An integration test sends actual
`EnvironmentStepRunner` output through this adapter and compares its logs with
unchanged Python source. This does not yet establish the complete configured
simulator/GRU/Trainer assembly or serialization of arbitrary observation payloads.

For the default vessel's per-phase seed creation, use
`TrainingVesselSeeds::seed_queue_with_trainer(phase, &binding)`. It checks the
collection and logs its size before reading the current weakly bound Trainer's
fast-development setting, then reuses the existing subset and queue machinery.
Changes made by phase callbacks therefore affect the next queue. Missing
collections, logging errors and unavailable Trainer bindings stop in source
order; no producer starts before the owner enters the returned queue context.
This live entry point does not modify the standalone seed configuration or keep
the Trainer alive.

`RlTrainingVessel` is now the production bridge between `RlTrainerDriver`, those
seed collections and the existing `TrainingVesselRunner`. Its environment factory
receives the activated shared queue and current Trainer control. Train, validation
and test delegate to the same owned runner; checkpoint save/load delegate to that
runner's policy-only envelope. The bridge holds only weak Trainer bindings and
does not add optimizer, replay, seed or RNG state to the checkpoint.

Lifecycle tests cover live callback configuration, all phases, environment-factory
failure cleanup, missing seeds, policy-run errors and checkpoint errors. A real
GRU/PPO test trains through this production bridge, restores complete semantic
tensor snapshots into the original parameters and trains again. That test still
uses a controlled backend, not the complete configured simulator pipeline. The
application-owned environment factory must still assemble its simulator, independent
worker interpreters, observation space and loggers; arbitrary typed observation-log
persistence and the full configured simulator/GRU/Trainer graph remain unfinished.

`SaoeEnvironmentStateInterpreter` and `SaoeEnvironmentActionInterpreter` now connect
the existing stateless SAOE interpreter plugins to the environment interfaces.
The state adapter preserves the complete owned observation, including yesterday's
features and position; the action adapter delegates discrete/continuous actions
to the original execution-volume calculation. They preserve diagnostic text and
do not turn ordinary interpreter errors into seed exhaustion. These adapters do
not implement simulator construction, observation spaces or model projection.
Finite retirement must inspect the complete raw observation before any recurrent
model projection, and raw observations must remain available to logging.

The source-only `saoe_simulator_contract` integration test now freezes the outer
Qlib simulator constructor/reset/step behavior, cash-limit distinctions, repeated
action forwarding, final live adapter state and partial failures. It is an
executable contract for the pending owned native wrapper, not evidence that the
real simulator/Trainer pipeline is already connected.

`RecursiveStrategyDriver::new_owned` now retains an entire resumable executor
graph without a self-reference or copied executor state. The existing borrowed
constructor remains available, and both forms use the same action/decision loop.
This supplies ownership needed by a long-lived simulator, but does not yet supply
its configured construction, live SAOE adapter lookup or final report interface.

The existing `SaoeStateProvider` trait now also accepts `Arc<Mutex<P>>`, including
unsized provider objects. Strategy updates and simulator reads can share one live
adapter, preserving final metrics and fresh runtime state without modifying old
owned snapshots. Methods delegate under a short-lived lock; poisoned locks return
an explicit error and provider callbacks must not reenter the same mutex. The
complete simulator factory and graph assembly remain unfinished.
