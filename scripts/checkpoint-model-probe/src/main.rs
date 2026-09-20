//! Exercise real model payloads in a candidate runtime, without a Python bridge.
use candle_core::{Device, Tensor, Var, safetensors};
#[cfg(all(feature = "current", not(feature = "production")))]
use candle_current as candle_core;
#[cfg(all(
    feature = "msrv-candidate",
    any(feature = "production", not(feature = "current"))
))]
use candle_msrv as candle_core;
use serde::Deserialize;
#[cfg(not(feature = "production"))]
use std::collections::HashMap;
use std::{error::Error, fs, path::Path};

#[cfg(not(feature = "production"))]
fn restore(
    names: &[String],
    live: &HashMap<String, Var>,
    incoming: &HashMap<String, Tensor>,
) -> Vec<String> {
    let mut errors = Vec::new();
    for name in names {
        let Some(source) = incoming.get(name) else {
            errors.push(format!("missing:{name}"));
            continue;
        };
        let target = &live[name];
        // Torch retains the destination dtype. A failed parameter must not stop
        // later registered parameters from receiving their checkpoint values.
        let result = source
            .to_dtype(target.dtype())
            .and_then(|source| target.set(&source));
        if let Err(error) = result {
            errors.push(format!("copy:{name}:{error}"));
        }
    }
    // Error presence and partial effects are characterized here. Full recursive
    // PyTorch missing/unexpected-key diagnostics still need module-tree handling.
    let mut unexpected = incoming
        .keys()
        .filter(|name| !live.contains_key(*name))
        .collect::<Vec<_>>();
    unexpected.sort();
    errors.extend(
        unexpected
            .into_iter()
            .map(|name| format!("unexpected:{name}")),
    );
    errors
}

#[derive(Deserialize)]
struct Manifest {
    // Registration order, not the file's sorted key order.
    names: Vec<String>,
    alias_groups: Vec<Vec<String>>,
    #[cfg(feature = "production")]
    #[serde(default)]
    encoded_names: bool,
    #[cfg(feature = "production")]
    #[serde(default)]
    weight_names: Option<Vec<String>>,
}

#[cfg(not(feature = "production"))]
fn run(directory: &Path) -> Result<(), Box<dyn Error>> {
    let manifest: Manifest = serde_json::from_slice(&fs::read(directory.join("manifest.json"))?)?;
    let input = safetensors::load(directory.join("input.safetensors"), &Device::Cpu)?;
    let initial = safetensors::load(directory.join("initial.safetensors"), &Device::Cpu)?;
    let mut live = HashMap::<String, Var>::new();
    for name in &manifest.names {
        live.insert(name.clone(), Var::from_tensor(&initial[name])?);
    }
    // Existing model structure establishes sharing; a flat tensor file does not.
    for group in &manifest.alias_groups {
        let shared = live[&group[0]].clone();
        for name in group {
            live.insert(name.clone(), shared.clone());
        }
    }
    // Restore in model registration order, including repeated shared destinations.
    let errors = restore(&manifest.names, &live, &input);
    fs::write(
        directory.join("load-errors.json"),
        serde_json::to_vec(&errors)?,
    )?;
    let snapshot = manifest
        .names
        .iter()
        .map(|name| Ok((name.clone(), live[name].copy()?)))
        .collect::<candle_core::Result<HashMap<_, _>>>()?;
    for name in &manifest.names {
        let var = &live[name];
        var.set(&Tensor::zeros(var.shape(), var.dtype(), &Device::Cpu)?)?;
    }
    // Save after mutation to prove snapshots are independent of the live weights.
    safetensors::save(&snapshot, directory.join("snapshot.safetensors"))?;
    let decoded = safetensors::load(directory.join("snapshot.safetensors"), &Device::Cpu)?;
    assert!(restore(&manifest.names, &live, &decoded).is_empty());
    for group in &manifest.alias_groups {
        assert!(
            group
                .iter()
                .all(|name| live[name].id() == live[&group[0]].id())
        );
    }
    let restored = manifest
        .names
        .iter()
        .map(|name| (name.clone(), live[name].as_tensor().clone()))
        .collect::<HashMap<_, _>>();
    safetensors::save(&restored, directory.join("restored.safetensors"))?;
    println!(
        "{} tensors, {} shared groups restored",
        live.len(),
        manifest.alias_groups.len()
    );
    Ok(())
}

#[cfg(feature = "production")]
fn run(directory: &Path) -> Result<(), Box<dyn Error>> {
    use domain_core::rl_candle_checkpoint::{CandlePolicySnapshot, CandlePolicyState};
    use domain_core::rl_policy_weight::{PolicyWeightLoadError, set_policy_weights};
    use indexmap::IndexMap;
    let manifest: Manifest = serde_json::from_slice(&fs::read(directory.join("manifest.json"))?)?;
    // Initial state defines the live destination types. Never let a convenience
    // loader silently replace an unsupported model dtype with a wider one.
    let initial_bytes = fs::read(directory.join("initial.safetensors"))?;
    for (_, view) in safetensors_wire::SafeTensors::deserialize(&initial_bytes)?.iter() {
        candle_core::DType::try_from(view.dtype())?;
    }
    let initial = safetensors::load(directory.join("initial.safetensors"), &Device::Cpu)?;
    let mut live = manifest
        .names
        .iter()
        .map(|name| {
            let wire_name = if manifest.encoded_names {
                format!("p{name}")
            } else {
                name.clone()
            };
            Ok((name.clone(), Var::from_tensor(&initial[&wire_name])?))
        })
        .collect::<candle_core::Result<IndexMap<_, _>>>()?;
    for group in &manifest.alias_groups {
        let shared = live[&group[0]].clone();
        for name in group {
            live.insert(name.clone(), shared.clone());
        }
    }
    let mut model = CandlePolicyState::new(live, ());
    let input = CandlePolicySnapshot {
        tensors: fs::read(directory.join("input.safetensors"))?,
        metadata: (),
    };
    let errors = if let Some(names) = manifest.weight_names {
        let mut state = input.into_policy_weights(None)?;
        assert_eq!(names, state.weights.keys().cloned().collect::<Vec<_>>());
        let result = set_policy_weights(&mut model, &mut state);
        let error_kind = match &result {
            Ok(()) => None,
            Err(PolicyWeightLoadError::Runtime(_)) => Some("runtime"),
            Err(PolicyWeightLoadError::Other(_)) => Some("other"),
        };
        fs::write(
            directory.join("weight-error-kind.json"),
            serde_json::to_vec(&error_kind)?,
        )?;
        fs::write(
            directory.join("weight-names.json"),
            serde_json::to_vec(&state.weights.keys().collect::<Vec<_>>())?,
        )?;
        let mut input_aliases = IndexMap::<_, Vec<_>>::new();
        for (name, tensor) in &state.weights {
            input_aliases.entry(tensor.id()).or_default().push(name);
        }
        fs::write(
            directory.join("weight-aliases.json"),
            serde_json::to_vec(
                &input_aliases
                    .values()
                    .filter(|group| group.len() > 1)
                    .collect::<Vec<_>>(),
            )?,
        )?;
        // Use the established lossless wire encoding for the mutated input map.
        let mut converted_vars = IndexMap::new();
        let mut shared_vars = std::collections::HashMap::<_, Var>::new();
        for (name, tensor) in &state.weights {
            let variable = match shared_vars.entry(tensor.id()) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(Var::from_tensor(tensor)?).clone()
                }
            };
            converted_vars.insert(name.clone(), variable);
        }
        let converted = CandlePolicyState::new(converted_vars, ());
        fs::write(
            directory.join("converted.safetensors"),
            converted.snapshot()?.tensors,
        )?;
        result
            .err()
            .map(|error| error.to_string())
            .into_iter()
            .collect::<Vec<_>>()
    } else {
        model.restore(&input).err().into_iter().collect::<Vec<_>>()
    };
    fs::write(
        directory.join("load-errors.json"),
        serde_json::to_vec(&errors)?,
    )?;
    let snapshot = model.snapshot()?;
    for variable in model.variables().values() {
        variable.set(&Tensor::zeros(
            variable.shape(),
            variable.dtype(),
            &Device::Cpu,
        )?)?;
    }
    fs::write(directory.join("snapshot.safetensors"), &snapshot.tensors)?;
    model.restore(&snapshot)?;
    for group in &manifest.alias_groups {
        assert!(
            group
                .iter()
                .all(|name| model.variables()[name].id() == model.variables()[&group[0]].id())
        );
    }
    fs::write(
        directory.join("restored.safetensors"),
        &model.snapshot()?.tensors,
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let directory = std::env::args_os().nth(1).ok_or("missing case directory")?;
    run(Path::new(&directory))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shallow_handles_and_owned_snapshots_have_different_mutation_contracts()
    -> candle_core::Result<()> {
        let live = Var::new(&[1_f32, 2.], &Device::Cpu)?;
        let alias = live.clone();
        let detached = live.as_detached_tensor();
        let owned = live.copy()?;
        live.set(&Tensor::new(&[3_f32, 4.], &Device::Cpu)?)?;
        assert_eq!(alias.to_vec1::<f32>()?, [3., 4.]);
        assert_eq!(detached.to_vec1::<f32>()?, [3., 4.]);
        assert_eq!(owned.to_vec1::<f32>()?, [1., 2.]);
        assert!(live.set(live.as_tensor()).is_err());
        assert!(live.set(&Tensor::new(&[1_f32], &Device::Cpu)?).is_err());
        assert!(live.set(&Tensor::new(&[1_f64, 2.], &Device::Cpu)?).is_err());
        assert_eq!(live.to_vec1::<f32>()?, [3., 4.]);
        Ok(())
    }
}
