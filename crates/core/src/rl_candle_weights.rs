//! Native snapshot-to-weight materialization, preserving explicit order and aliases.

use super::{CandlePolicySnapshot, container_result, decode_tensors};
use candle_core::{DType, Device, Tensor, Var, safetensors::Load};
use indexmap::{IndexMap, IndexSet};
use safetensors::{SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::rl_policy_weight::PolicyWeights;

pub(super) const LAYOUT_KEY: &str = "core.candle_policy.layout";

/// A versioned logical input layout, independent of tensor-file iteration order.
/// Each entry is `(name, source_index)`: a unique tensor refers to its own index;
/// a whole-tensor alias refers to an earlier entry with identical dtype/shape/bits.
/// Equal values alone never establish aliasing. Partial/strided shared views are
/// not represented by this native whole-variable format.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CandlePolicyLayout {
    pub version: u32,
    pub entries: Vec<(String, usize)>,
}

pub(super) fn encode_layout(variables: &IndexMap<String, Var>) -> String {
    let mut sources = HashMap::new();
    let entries = variables
        .iter()
        .enumerate()
        .map(|(index, (name, var))| {
            let source = *sources.entry(var.id()).or_insert(index);
            serde_json::json!([name, source])
        })
        .collect::<Vec<_>>();
    // JSON values containing only strings and bounded integers serialize infallibly.
    serde_json::json!({"version": 1, "entries": entries}).to_string()
}

fn read_layout(
    bytes: &[u8],
    legacy: Option<&CandlePolicyLayout>,
) -> Result<CandlePolicyLayout, String> {
    let (_, metadata) = container_result(SafeTensors::read_metadata(bytes))?;
    match metadata.metadata().as_ref().and_then(|m| m.get(LAYOUT_KEY)) {
        Some(text) => {
            let layout: CandlePolicyLayout =
                serde_json::from_str(text).map_err(|e| e.to_string())?;
            if let Some(legacy) = legacy {
                if legacy != &layout {
                    return Err("explicit policy layout conflicts with stored layout".into());
                }
            }
            Ok(layout)
        }
        None => legacy.cloned().ok_or_else(|| {
            "missing policy layout: supply the original layout for this older snapshot".into()
        }),
    }
}

fn validate_layout<'a, 'data>(
    tensors: &'a IndexMap<String, TensorView<'data>>,
    layout: &CandlePolicyLayout,
) -> Result<Vec<&'a TensorView<'data>>, String> {
    if layout.version != 1 {
        return Err(format!(
            "unsupported policy layout version:{}",
            layout.version
        ));
    }
    if layout.entries.len() != tensors.len() {
        return Err("policy layout does not cover all tensors".into());
    }
    let mut names = IndexSet::new();
    let mut ordered: Vec<&TensorView<'data>> = Vec::new();
    for (index, (name, source)) in layout.entries.iter().enumerate() {
        if !names.insert(name) {
            return Err(format!("duplicate policy layout name:{name}"));
        }
        let view = tensors
            .get(name)
            .ok_or_else(|| format!("missing policy layout tensor:{name}"))?;
        if *source > index {
            return Err(format!("forward policy alias:{name}:{source}"));
        }
        if *source < index {
            let shared = ordered[*source];
            if view.dtype() != shared.dtype()
                || view.shape() != shared.shape()
                || view.data() != shared.data()
            {
                return Err(format!("inconsistent policy alias:{name}"));
            }
        }
        ordered.push(view);
    }
    Ok(ordered)
}

impl<M> CandlePolicySnapshot<M> {
    /// Move this snapshot into CPU weights for `set_policy_weights` without
    /// cloning opaque metadata or touching any existing model. Whole-tensor
    /// aliases reuse the same Arc; independent equal tensors remain independent.
    ///
    /// Supply `legacy_layout` only from an authoritative producer for older files
    /// that lack the layout. Stored and supplied layouts must agree when both exist.
    /// Source dtypes unsupported by Candle are rejected, never silently widened.
    /// Existing parameter restoration still supports its documented input casts.
    ///
    /// # Errors
    /// Returns corrupt/missing/conflicting layout, alias or native tensor load errors.
    pub fn into_policy_weights(
        self,
        legacy_layout: Option<&CandlePolicyLayout>,
    ) -> Result<PolicyWeights<Tensor, M>, String> {
        let tensors = decode_tensors(&self.tensors)?;
        let layout = read_layout(&self.tensors, legacy_layout)?;
        let ordered = validate_layout(&tensors, &layout)?;
        let mut values: Vec<Arc<Tensor>> = Vec::new();
        let mut weights = IndexMap::new();
        for (index, ((name, source), view)) in layout.entries.iter().zip(ordered).enumerate() {
            let tensor = if *source == index {
                Arc::new(
                    DType::try_from(view.dtype())
                        .and_then(|_| view.load(&Device::Cpu))
                        .map_err(|e| e.to_string())?,
                )
            } else {
                Arc::clone(&values[*source])
            };
            weights.insert(name.clone(), Arc::clone(&tensor));
            values.push(tensor);
        }
        Ok(PolicyWeights {
            weights,
            metadata: self.metadata,
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_candle_weights.rs"]
mod tests;
