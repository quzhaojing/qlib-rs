//! CPU Candle policy-state adapter. This is not a Torch/pickle file codec.
//!
//! The backend constructor accepts live Candle variables. Stable Trainer plugin
//! interfaces exchange owned `SafeTensors` bytes and caller-owned metadata only.
//! Ordinary native parameters are supported; custom Python module hooks, module
//! version upgrades and recursive unexpected-key diagnostics remain separate.

use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, Tensor, Var, safetensors::Load};
use indexmap::IndexMap;
use safetensors::{SafeTensorError, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::rl_policy_weight::{PolicyWeightLoadError, PolicyWeightLoader, PolicyWeights};
use crate::{RlCheckpointState, TrainingPolicyState};

const NAME_ENCODING_KEY: &str = "core.candle_policy.tensor_names";
const NAME_ENCODING_VERSION: &str = "prefix-p-v1";

#[path = "rl_candle_weights.rs"]
mod weights;
pub use weights::CandlePolicyLayout;

// One owned-error conversion for both reading and writing the tensor container.
fn container_result<T>(result: Result<T, SafeTensorError>) -> Result<T, String> {
    result.map_err(|error| error.to_string())
}

fn decode_tensors(bytes: &[u8]) -> Result<IndexMap<String, TensorView<'_>>, String> {
    // SafeTensors 0.4 exposes file metadata separately from tensor views. Treat
    // both reads as one fallible container boundary; never unwrap a validated read.
    let (metadata, tensors) =
        container_result(SafeTensors::read_metadata(bytes).and_then(|(_, metadata)| {
            SafeTensors::deserialize(bytes).map(|tensors| (metadata, tensors))
        }))?;
    let encoded = match metadata
        .metadata()
        .as_ref()
        .and_then(|m| m.get(NAME_ENCODING_KEY))
    {
        None => false,
        Some(version) if version == NAME_ENCODING_VERSION => true,
        Some(version) => return Err(format!("unsupported tensor-name encoding:{version}")),
    };
    let mut decoded = IndexMap::new();
    for (name, tensor) in tensors.iter() {
        let name = if encoded {
            name.strip_prefix('p')
                .ok_or_else(|| format!("invalid encoded tensor name:{name}"))?
        } else {
            name
        };
        decoded.insert(name.to_owned(), tensor);
    }
    Ok(decoded)
}

// Preserve source values in a supported intermediate dtype, then cast to the
// existing destination dtype. This is not creation of a new widened parameter.
fn load_parameter(source: &TensorView<'_>, device: &Device) -> candle_core::Result<Tensor> {
    match source.dtype() {
        safetensors::Dtype::BOOL => Tensor::from_vec(
            source
                .data()
                .iter()
                .map(|byte| u8::from(*byte != 0))
                .collect::<Vec<_>>(),
            source.shape(),
            device,
        ),
        safetensors::Dtype::I8 => Tensor::from_vec(
            source
                .data()
                .iter()
                .map(|byte| i64::from(i8::from_le_bytes([*byte])))
                .collect::<Vec<_>>(),
            source.shape(),
            device,
        ),
        safetensors::Dtype::I16 => {
            // SafeTensors validated shape/byte length before constructing this
            // view, so the bulk reader's exact-length precondition holds.
            let mut values = vec![0_i16; source.data().len() / 2];
            LittleEndian::read_i16_into(source.data(), &mut values);
            Tensor::from_vec(
                values.into_iter().map(i64::from).collect::<Vec<_>>(),
                source.shape(),
                device,
            )
        }
        // Candle already losslessly widens I32 to I64 and U16 to U32 on input.
        _ => source.load(device),
    }
}

trait ParameterSource {
    fn dimensions(&self) -> &[usize];
    fn load_on(&self, device: &Device) -> candle_core::Result<Tensor>;
}

impl ParameterSource for TensorView<'_> {
    fn dimensions(&self) -> &[usize] {
        self.shape()
    }

    fn load_on(&self, device: &Device) -> candle_core::Result<Tensor> {
        load_parameter(self, device)
    }
}

impl ParameterSource for Tensor {
    fn dimensions(&self) -> &[usize] {
        self.dims()
    }

    fn load_on(&self, device: &Device) -> candle_core::Result<Tensor> {
        // A supplied tensor can be the live target itself. An owned source copy
        // permits that no-op assignment without Candle's same-variable error.
        self.to_device(device).and_then(|tensor| tensor.copy())
    }
}

/// A value snapshot, independent of live weights. Metadata is retained without
/// imposing a JSON-only representation on the surrounding Trainer codec.
/// Tensor files written by this adapter mark `core.candle_policy.tensor_names`
/// as `prefix-p-v1` and prefix every tensor key with `p`. This bijection preserves
/// reserved keys (including `__metadata__`), Unicode and prefix lookalikes.
/// Unmarked input files retain literal keys; unknown encodings are rejected.
/// New snapshots also record logical key order and exact whole-variable aliases
/// in `core.candle_policy.layout`; the outer DTO and old parameter restore path
/// are unchanged. Additional metadata does not promise byte-canonical JSON order.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CandlePolicySnapshot<M> {
    pub tensors: Vec<u8>,
    pub metadata: M,
}

/// Registration order and live variable identity belong to the model, not the file.
/// Cloning a `Var` when registering shared names preserves that live identity.
pub struct CandlePolicyState<M> {
    variables: IndexMap<String, Var>,
    metadata: M,
}

impl<M> CandlePolicyState<M> {
    #[must_use]
    pub const fn new(variables: IndexMap<String, Var>, metadata: M) -> Self {
        Self {
            variables,
            metadata,
        }
    }

    /// Backend-facing access to the same live handles used by model computation.
    /// This is not part of the runtime-independent Trainer plugin interface.
    #[must_use]
    pub const fn variables(&self) -> &IndexMap<String, Var> {
        &self.variables
    }

    /// Restore native parameter values, retaining target dtype and sharing.
    ///
    /// Container corruption fails before assignments. Missing, incompatible and
    /// unexpected tensors are reported after all other valid assignments; earlier
    /// successful copies are not rolled back. Source registration order wins for
    /// conflicting shared names. Metadata remains model-owned: this native adapter
    /// does not interpret custom module-version migration hooks.
    /// A scalar destination accepts the first element of a one-dimensional
    /// source. An empty source in that case aborts immediately, retaining prior
    /// copies but preventing later ones, as in the source policy loader.
    ///
    /// # Errors
    /// Returns container errors, scalar-index errors or accumulated parameter failures.
    pub fn restore(&self, state: &CandlePolicySnapshot<M>) -> Result<(), String> {
        self.restore_typed(state).map_err(|error| error.to_string())
    }

    /// Restore a persisted snapshot with an explicit error category for policy callers.
    ///
    /// # Errors
    /// Container and scalar-index failures are non-retryable; accumulated parameter
    /// failures are runtime errors. Existing string-returning interfaces are unchanged.
    pub fn restore_typed(
        &self,
        state: &CandlePolicySnapshot<M>,
    ) -> Result<(), PolicyWeightLoadError> {
        let tensors = decode_tensors(&state.tensors).map_err(PolicyWeightLoadError::Other)?;
        let sources = tensors
            .iter()
            .map(|(name, view)| (name.clone(), view as &dyn ParameterSource))
            .collect();
        self.restore_parameters(&sources)
    }

    fn restore_parameters(
        &self,
        tensors: &IndexMap<String, &dyn ParameterSource>,
    ) -> Result<(), PolicyWeightLoadError> {
        let mut errors = Vec::new();
        for (name, target) in &self.variables {
            let Some(source) = tensors.get(name) else {
                errors.push(format!("missing:{name}"));
                continue;
            };
            // Torch's scalar compatibility indexing precedes dtype conversion
            // and its accumulated copy-error handler. Empty input is a fatal
            // indexing error even if earlier missing/copy errors were collected.
            let scalar_vector = target.rank() == 0 && source.dimensions().len() == 1;
            if scalar_vector && source.dimensions()[0] == 0 {
                return Err(PolicyWeightLoadError::Other(format!(
                    "index:{name}:index 0 is out of bounds for dimension 0 with size 0"
                )));
            }
            let result = source
                .load_on(target.device())
                .and_then(|source| {
                    if scalar_vector {
                        source.get(0)
                    } else {
                        Ok(source)
                    }
                })
                .and_then(|source| source.to_dtype(target.dtype()))
                .and_then(|source| target.set(&source));
            if let Err(error) = result {
                errors.push(format!("copy:{name}:{error}"));
            }
        }
        let mut unexpected = tensors
            .keys()
            .filter(|name| !self.variables.contains_key(*name))
            .collect::<Vec<_>>();
        unexpected.sort();
        errors.extend(
            unexpected
                .into_iter()
                .map(|name| format!("unexpected:{name}")),
        );
        if errors.is_empty() {
            Ok(())
        } else {
            Err(PolicyWeightLoadError::Runtime(errors.join("\n")))
        }
    }
}

impl<M> CandlePolicyState<M> {
    /// Copy decoded weights without cloning or interpreting their opaque metadata.
    /// The model's snapshot metadata and the input's metadata need not share a type.
    /// # Errors
    /// Returns the same ordered, partially mutating load errors as snapshot restore.
    pub fn load_native_weights<InputMetadata>(
        &self,
        state: &PolicyWeights<Tensor, InputMetadata>,
    ) -> Result<(), PolicyWeightLoadError> {
        let sources = state
            .weights
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.as_ref() as &dyn ParameterSource))
            .collect();
        self.restore_parameters(&sources)
    }
}

// Backend-facing implementation: the generic retry protocol has no Candle types
// in its own definition, while native callers may supply shared Candle tensors.
impl<M> PolicyWeightLoader<Tensor, M> for CandlePolicyState<M> {
    fn load_weights(
        &mut self,
        state: &mut PolicyWeights<Tensor, M>,
    ) -> Result<(), PolicyWeightLoadError> {
        self.load_native_weights(state)
    }
}

impl<M: Clone> CandlePolicyState<M> {
    /// Serialize values immediately, so mutations through shared live handles
    /// cannot change an early-stopping snapshot or an already collected graph.
    ///
    /// # Errors
    /// Returns a `SafeTensors` serialization error.
    pub fn snapshot(&self) -> Result<CandlePolicySnapshot<M>, String> {
        container_result(safetensors::serialize(
            self.variables
                .iter()
                .map(|(name, var)| (format!("p{name}"), var.as_tensor())),
            &Some(HashMap::from([
                (
                    NAME_ENCODING_KEY.to_owned(),
                    NAME_ENCODING_VERSION.to_owned(),
                ),
                (
                    weights::LAYOUT_KEY.to_owned(),
                    weights::encode_layout(&self.variables),
                ),
            ])),
        ))
        .map(|tensors| CandlePolicySnapshot {
            tensors,
            metadata: self.metadata.clone(),
        })
    }
}

impl<M: Clone + Send> TrainingPolicyState<CandlePolicySnapshot<M>> for CandlePolicyState<M> {
    fn state_dict(&mut self) -> Result<CandlePolicySnapshot<M>, String> {
        self.snapshot()
    }

    fn load_state_dict(&mut self, state: &CandlePolicySnapshot<M>) -> Result<(), String> {
        self.restore(state)
    }
}

impl<M: Clone> RlCheckpointState<CandlePolicySnapshot<M>> for CandlePolicyState<M> {
    fn save_checkpoint(&mut self) -> Result<CandlePolicySnapshot<M>, String> {
        self.snapshot()
    }

    fn load_checkpoint(&mut self, state: &CandlePolicySnapshot<M>) -> Result<(), String> {
        self.restore(state)
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_checkpoint.rs"]
mod tests;
