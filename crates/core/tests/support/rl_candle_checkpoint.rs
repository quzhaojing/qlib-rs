use super::*;
use crate::rl_policy_weight::{PolicyWeights, set_policy_weights};
use crate::{BincodeRlCheckpointCodec, RlCheckpointFileCodec, TrainingVesselState};
use candle_core::{Device, Tensor};
use std::sync::Arc;

fn var(values: &[f32]) -> Var {
    Var::from_slice(values, values.len(), &Device::Cpu).unwrap()
}

#[test]
fn snapshot_records_logical_order_and_whole_variable_aliases() {
    let shared = var(&[3., 4.]);
    let state = CandlePolicyState::new(
        IndexMap::from([
            ("z".into(), shared.clone()),
            ("a".into(), var(&[3., 4.])),
            ("模型\0".into(), shared),
        ]),
        1_u64,
    )
    .snapshot()
    .unwrap();
    let (_, metadata) = SafeTensors::read_metadata(&state.tensors).unwrap();
    let layout = metadata
        .metadata()
        .as_ref()
        .unwrap()
        .get("core.candle_policy.layout");
    assert!(
        layout.is_some(),
        "native snapshot lost its key order and aliases"
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(layout.unwrap()).unwrap(),
        serde_json::json!({"version": 1, "entries": [["z", 0], ["a", 1], ["模型\0", 0]]})
    );
}

fn payload(values: &[(&str, Tensor)]) -> CandlePolicySnapshot<u64> {
    CandlePolicySnapshot {
        tensors: safetensors::serialize(values.iter().map(|(n, v)| (*n, v)), &None).unwrap(),
        metadata: 1,
    }
}

#[test]
fn owned_snapshots_preserve_shared_live_parameters_and_real_computation() {
    let weight = var(&[2., 3.]);
    let variables = IndexMap::from([
        ("actor.weight".into(), weight.clone()),
        ("critic.weight".into(), weight.clone()),
    ]);
    let mut model = CandlePolicyState::new(variables, 1_u64);
    let before = weight
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let state = model.state_dict().unwrap();
    weight
        .set(&Tensor::new(&[9_f32, 9.], &Device::Cpu).unwrap())
        .unwrap();
    model.load_state_dict(&state).unwrap();
    assert_eq!(
        weight
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .to_bits(),
        before.to_bits()
    );
    assert_eq!(
        model.variables()["actor.weight"].id(),
        model.variables()["critic.weight"].id()
    );
    let conflict = payload(&[
        (
            "critic.weight",
            Tensor::new(&[7_f32, 8.], &Device::Cpu).unwrap(),
        ),
        (
            "actor.weight",
            Tensor::new(&[4_f32, 5.], &Device::Cpu).unwrap(),
        ),
    ]);
    model.restore(&conflict).unwrap();
    assert_eq!(weight.to_vec1::<f32>().unwrap(), [7., 8.]);
    let mut codec = BincodeRlCheckpointCodec;
    let mut bytes = Vec::new();
    codec.encode(&state, &mut bytes).unwrap();
    let decoded: CandlePolicySnapshot<u64> = codec.decode(&mut bytes.as_slice()).unwrap();
    assert_eq!(state, decoded);
    model.load_checkpoint(&decoded).unwrap();
    let saved_again = model.save_checkpoint().unwrap();
    assert_eq!(saved_again.metadata, state.metadata);
    assert_eq!(
        decode_tensors(&saved_again.tensors).unwrap(),
        decode_tensors(&state.tensors).unwrap()
    );
    // SafeTensors metadata is a HashMap; compare every decoded field, not JSON
    // object-key order. The exact original Bincode byte roundtrip above remains.
    assert_eq!(
        SafeTensors::read_metadata(&saved_again.tensors)
            .unwrap()
            .1
            .metadata(),
        SafeTensors::read_metadata(&state.tensors)
            .unwrap()
            .1
            .metadata()
    );
    let mut vessel = TrainingVesselState::new(Box::new(model));
    let state = vessel.state_dict().unwrap();
    weight
        .set(&Tensor::new(&[0_f32, 0.], &Device::Cpu).unwrap())
        .unwrap();
    vessel.load_state_dict(&state).unwrap();
    assert_eq!(weight.to_vec1::<f32>().unwrap(), [2., 3.]);
}

#[test]
fn errors_accumulate_without_preventing_later_valid_copies() {
    let first = var(&[1., 2.]);
    let last = var(&[3., 4.]);
    let model = CandlePolicyState::new(
        IndexMap::from([
            ("first".into(), first.clone()),
            ("last".into(), last.clone()),
        ]),
        1,
    );
    let state = payload(&[
        ("first", Tensor::new(&[0_f32], &Device::Cpu).unwrap()),
        ("last", Tensor::new(&[8_f64, 9.], &Device::Cpu).unwrap()),
        ("extra", Tensor::new(&[0_f32], &Device::Cpu).unwrap()),
    ]);
    let error = model.restore(&state).unwrap_err();
    assert!(error.contains("copy:first") && error.contains("unexpected:extra"));
    assert_eq!(first.to_vec1::<f32>().unwrap(), [1., 2.]);
    assert_eq!(last.to_vec1::<f32>().unwrap(), [8., 9.]);
    let missing = payload(&[("last", Tensor::new(&[5_f32, 6.], &Device::Cpu).unwrap())]);
    assert_eq!(model.restore(&missing).unwrap_err(), "missing:first");
    assert_eq!(last.to_vec1::<f32>().unwrap(), [5., 6.]);
    let corrupt = CandlePolicySnapshot {
        tensors: vec![0],
        metadata: 1,
    };
    assert!(model.restore(&corrupt).is_err());
    assert_eq!(last.to_vec1::<f32>().unwrap(), [5., 6.]);
}

#[test]
fn unsupported_wire_dtypes_fail_without_mutating_the_target() {
    let target = var(&[1., 2.]);
    let model = CandlePolicyState::new(IndexMap::from([("weight".into(), target.clone())]), 1);
    let raw = [0_u8; 16];
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::U64, vec![2], &raw).unwrap();
    let state = CandlePolicySnapshot {
        tensors: safetensors::serialize([("weight", view)], &None).unwrap(),
        metadata: 1,
    };
    assert!(
        model
            .restore(&state)
            .unwrap_err()
            .contains("unsupported safetensor dtype U64")
    );
    assert_eq!(target.to_vec1::<f32>().unwrap(), [1., 2.]);
    let empty = CandlePolicyState::new(IndexMap::new(), 1_u64);
    empty.restore(&empty.snapshot().unwrap()).unwrap();
}

#[test]
fn reserved_and_prefix_like_names_roundtrip_without_collisions() {
    let entries = [
        ("__metadata__", 2_f32),
        ("p__metadata__", 4.),
        ("p", 6.),
        ("", 8.),
        ("模型\0weight", 10.),
    ];
    let variables = entries
        .iter()
        .map(|(name, value)| ((*name).to_owned(), var(&[*value, *value + 1.])))
        .collect();
    let model = CandlePolicyState::new(variables, 1_u64);
    let state = model.snapshot().unwrap();
    for weight in model.variables().values() {
        weight
            .set(&Tensor::new(&[9_f32, 9.], &Device::Cpu).unwrap())
            .unwrap();
    }
    model.restore(&state).unwrap();
    for (name, value) in entries {
        assert_eq!(
            model.variables()[name].to_vec1::<f32>().unwrap(),
            [value, value + 1.]
        );
    }
}

#[test]
fn wire_schema_errors_fail_before_any_parameter_copy() {
    let target = var(&[1., 2.]);
    let model = CandlePolicyState::new(IndexMap::from([("weight".into(), target.clone())]), 1);
    let replacement = Tensor::new(&[8_f32, 9.], &Device::Cpu).unwrap();
    for (version, names, expected) in [
        (
            "future-v2",
            vec!["pweight"],
            "unsupported tensor-name encoding:future-v2",
        ),
        (
            NAME_ENCODING_VERSION,
            vec!["pweight", "bad"],
            "invalid encoded tensor name:bad",
        ),
    ] {
        let state = CandlePolicySnapshot {
            tensors: safetensors::serialize(
                names.iter().map(|name| (*name, &replacement)),
                &Some(HashMap::from([(
                    NAME_ENCODING_KEY.to_owned(),
                    version.to_owned(),
                )])),
            )
            .unwrap(),
            metadata: 1,
        };
        assert_eq!(model.restore(&state).unwrap_err(), expected);
        assert_eq!(target.to_vec1::<f32>().unwrap(), [1., 2.]);
    }
    // Foreign metadata without our encoding marker must not rename literal keys.
    let literal = CandlePolicySnapshot {
        tensors: safetensors::serialize(
            [("weight", &replacement)],
            &Some(HashMap::from([("format".to_owned(), "pt".to_owned())])),
        )
        .unwrap(),
        metadata: 1,
    };
    model.restore(&literal).unwrap();
    assert_eq!(target.to_vec1::<f32>().unwrap(), [8., 9.]);
    let extra = CandlePolicyState::new(
        IndexMap::from([
            ("__metadata__".into(), var(&[3., 4.])),
            ("pweight".into(), var(&[5., 6.])),
        ]),
        1,
    )
    .snapshot()
    .unwrap();
    assert_eq!(
        model.restore(&extra).unwrap_err(),
        "missing:weight\nunexpected:__metadata__\nunexpected:pweight"
    );
    assert_eq!(target.to_vec1::<f32>().unwrap(), [8., 9.]);
}

#[test]
fn integer_and_bool_sources_cast_into_the_existing_float_parameter() {
    use safetensors::{Dtype, tensor::TensorView};
    let cases = [
        (Dtype::BOOL, vec![0, 1, 2, 255], vec![0_f32, 1., 1., 1.]),
        (
            Dtype::I8,
            vec![128, 255, 0, 127],
            vec![-128., -1., 0., 127.],
        ),
        (
            Dtype::I16,
            [-32768_i16, -1, 0, 32767]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect(),
            vec![-32768., -1., 0., 32767.],
        ),
        (
            Dtype::I32,
            [i32::MIN, -1, 0, i32::MAX]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
            vec![-2_147_483_648., -1., 0., 2_147_483_648.],
        ),
        (
            Dtype::U16,
            [0_u16, 255, 32768, 65535]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            vec![0., 255., 32768., 65535.],
        ),
    ];
    for (dtype, bytes, expected) in cases {
        let weight = var(&[9., 9., 9., 9.]);
        let model = CandlePolicyState::new(IndexMap::from([("weight".into(), weight.clone())]), 1);
        let view = TensorView::new(dtype, vec![4], &bytes).unwrap();
        let state = CandlePolicySnapshot {
            tensors: safetensors::serialize([("weight", view)], &None).unwrap(),
            metadata: 1,
        };
        model.restore(&state).unwrap();
        assert_eq!(weight.dtype(), candle_core::DType::F32);
        assert_eq!(weight.to_vec1::<f32>().unwrap(), expected, "{dtype:?}");
        let snapshot = model.snapshot().unwrap();
        assert_eq!(
            decode_tensors(&snapshot.tensors).unwrap()["weight"].dtype(),
            Dtype::F32
        );
    }
}

#[test]
fn scalar_targets_take_the_first_vector_element_without_broadcasting_other_shapes() {
    for (incoming, expected, succeeds) in [
        (Tensor::new(&[3_f64], &Device::Cpu).unwrap(), 3_f32, true),
        (Tensor::new(&[4_f64, 8.], &Device::Cpu).unwrap(), 4., true),
        (Tensor::new(5_f64, &Device::Cpu).unwrap(), 5., true),
        (Tensor::new(&[[6_f64]], &Device::Cpu).unwrap(), 9., false),
    ] {
        let scalar = Var::new(9_f32, &Device::Cpu).unwrap();
        let later = var(&[0.]);
        let model = CandlePolicyState::new(
            IndexMap::from([
                ("value".into(), scalar.clone()),
                ("later".into(), later.clone()),
            ]),
            1,
        );
        let state = payload(&[
            ("later", Tensor::new(&[7_f32], &Device::Cpu).unwrap()),
            ("value", incoming),
        ]);
        let result = model.restore(&state);
        assert_eq!(result.is_ok(), succeeds, "{result:?}");
        if !succeeds {
            assert!(result.unwrap_err().contains("copy:value"));
        }
        assert_eq!(
            scalar.to_scalar::<f32>().unwrap().to_bits(),
            expected.to_bits()
        );
        assert_eq!(scalar.dtype(), candle_core::DType::F32);
        assert_eq!(later.to_vec1::<f32>().unwrap(), [7.]);
        let snapshot = model.snapshot().unwrap();
        assert!(
            decode_tensors(&snapshot.tensors).unwrap()["value"]
                .shape()
                .is_empty()
        );
        model.restore(&snapshot).unwrap();
    }
    let vector = var(&[9.]);
    let model = CandlePolicyState::new(IndexMap::from([("value".into(), vector.clone())]), 1);
    assert!(
        model
            .restore(&payload(&[(
                "value",
                Tensor::new(3_f32, &Device::Cpu).unwrap()
            )]))
            .is_err()
    );
    assert_eq!(vector.to_vec1::<f32>().unwrap(), [9.]);
}

#[test]
fn empty_scalar_source_aborts_at_the_index_without_rolling_back_prior_copies() {
    for dtype in [safetensors::Dtype::F64, safetensors::Dtype::U64] {
        for missing_before in [false, true] {
            let before = var(&[0.]);
            let scalar = Var::new(9_f32, &Device::Cpu).unwrap();
            let later = var(&[0.]);
            let model = CandlePolicyState::new(
                IndexMap::from([
                    ("before".into(), before.clone()),
                    ("value".into(), scalar.clone()),
                    ("later".into(), later.clone()),
                ]),
                1_u64,
            );
            let bytes = 7_f32.to_le_bytes();
            let mut values = vec![
                (
                    "later",
                    TensorView::new(safetensors::Dtype::F32, vec![1], &bytes).unwrap(),
                ),
                ("value", TensorView::new(dtype, vec![0], &[]).unwrap()),
            ];
            if !missing_before {
                values.push((
                    "before",
                    TensorView::new(safetensors::Dtype::F32, vec![1], &bytes).unwrap(),
                ));
            }
            let state = CandlePolicySnapshot {
                tensors: safetensors::serialize(values, &None).unwrap(),
                metadata: 1,
            };
            assert_eq!(
                model.restore(&state).unwrap_err(),
                "index:value:index 0 is out of bounds for dimension 0 with size 0"
            );
            assert_eq!(
                before.to_vec1::<f32>().unwrap(),
                [if missing_before { 0. } else { 7. }]
            );
            assert_eq!(
                scalar.to_scalar::<f32>().unwrap().to_bits(),
                9_f32.to_bits()
            );
            assert_eq!(later.to_vec1::<f32>().unwrap(), [0.]);
        }
    }
}

#[test]
fn native_weight_retry_fills_shared_legacy_names_and_preserves_input_identity() {
    let weight = var(&[0., 0.]);
    let mut model = CandlePolicyState::new(
        IndexMap::from([
            ("actor.weight".into(), weight.clone()),
            ("_actor_critic.actor.weight".into(), weight.clone()),
        ]),
        1_u64,
    );
    let input = Arc::new(Tensor::new(&[3_f64, 4.], &Device::Cpu).unwrap());
    let mut state = PolicyWeights {
        weights: IndexMap::from([("actor.weight".into(), Arc::clone(&input))]),
        metadata: 77_u64,
    };
    set_policy_weights(&mut model, &mut state).unwrap();
    assert_eq!(weight.to_vec1::<f32>().unwrap(), [3., 4.]);
    assert_eq!(state.metadata, 77);
    assert!(Arc::ptr_eq(
        &input,
        &state.weights["_actor_critic.actor.weight"]
    ));
    assert_eq!(
        state.weights.keys().map(String::as_str).collect::<Vec<_>>(),
        ["actor.weight", "_actor_critic.actor.weight"]
    );
    let snapshot = model.snapshot().unwrap();
    model.restore_typed(&snapshot).unwrap();
    assert_eq!(snapshot.metadata, 1);
    assert!(matches!(
        model.restore_typed(&CandlePolicySnapshot {
            tensors: vec![],
            metadata: 1
        }),
        Err(PolicyWeightLoadError::Other(_))
    ));

    // Self-copy is valid in Torch: do not turn Candle's same-variable check into a retry.
    let mut model =
        CandlePolicyState::new(IndexMap::from([("value".into(), weight.clone())]), 1_u64);
    let mut state = PolicyWeights {
        weights: IndexMap::from([("value".into(), Arc::new(weight.as_tensor().clone()))]),
        metadata: 77_u64,
    };
    set_policy_weights(&mut model, &mut state).unwrap();
    assert_eq!(state.weights.len(), 1);
    assert_eq!(weight.to_vec1::<f32>().unwrap(), [3., 4.]);
}

#[test]
fn native_weight_retry_preserves_second_failure_and_does_not_retry_index_errors() {
    let first = var(&[1., 2.]);
    let later = var(&[0.]);
    let mut model = CandlePolicyState::new(
        IndexMap::from([
            ("first".into(), first.clone()),
            ("later".into(), later.clone()),
        ]),
        1_u64,
    );
    let mut state = PolicyWeights {
        weights: IndexMap::from([
            (
                "later".into(),
                Arc::new(Tensor::new(&[7_f64], &Device::Cpu).unwrap()),
            ),
            (
                "first".into(),
                Arc::new(Tensor::new(&[9_f32], &Device::Cpu).unwrap()),
            ),
        ]),
        metadata: 77_u64,
    };
    let error = set_policy_weights(&mut model, &mut state).unwrap_err();
    assert!(matches!(error, PolicyWeightLoadError::Runtime(_)));
    assert!(error.to_string().contains("unexpected:_actor_critic.first"));
    assert_eq!(state.weights.len(), 4);
    assert!(Arc::ptr_eq(
        &state.weights["first"],
        &state.weights["_actor_critic.first"]
    ));
    assert_eq!(first.to_vec1::<f32>().unwrap(), [1., 2.]);
    assert_eq!(later.to_vec1::<f32>().unwrap(), [7.]);

    let scalar = Var::new(9_f32, &Device::Cpu).unwrap();
    let mut model = CandlePolicyState::new(
        IndexMap::from([
            ("value".into(), scalar.clone()),
            ("later".into(), later.clone()),
        ]),
        1_u64,
    );
    state.weights = IndexMap::from([
        (
            "value".into(),
            Arc::new(Tensor::new(&[] as &[f32], &Device::Cpu).unwrap()),
        ),
        (
            "later".into(),
            Arc::new(Tensor::new(&[8_f32], &Device::Cpu).unwrap()),
        ),
    ]);
    assert!(matches!(
        set_policy_weights(&mut model, &mut state),
        Err(PolicyWeightLoadError::Other(_))
    ));
    assert_eq!(state.weights.len(), 2);
    assert_eq!(
        scalar.to_scalar::<f32>().unwrap().to_bits(),
        9_f32.to_bits()
    );
    assert_eq!(later.to_vec1::<f32>().unwrap(), [7.]);
    state.weights.insert(
        "value".into(),
        Arc::new(Tensor::new(&[3_f64, 4.], &Device::Cpu).unwrap()),
    );
    set_policy_weights(&mut model, &mut state).unwrap();
    assert_eq!(
        scalar.to_scalar::<f32>().unwrap().to_bits(),
        3_f32.to_bits()
    );
    assert_eq!(later.to_vec1::<f32>().unwrap(), [8.]);
}
