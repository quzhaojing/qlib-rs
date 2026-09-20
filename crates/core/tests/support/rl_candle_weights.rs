use super::*;
use crate::rl_candle_checkpoint::CandlePolicyState;
use crate::rl_policy_weight::{PolicyWeightLoadError, PolicyWeightLoader, set_policy_weights};
use safetensors::Dtype;

// Deliberately not Clone, to test movement of model-owned metadata.
struct Opaque(Box<u64>);

fn layout(entries: &[(&str, usize)]) -> CandlePolicyLayout {
    CandlePolicyLayout {
        version: 1,
        entries: entries.iter().map(|(n, i)| ((*n).into(), *i)).collect(),
    }
}

fn snapshot(
    views: &[(&str, TensorView<'_>)],
    encoded_layout: Option<&str>,
) -> CandlePolicySnapshot<u64> {
    CandlePolicySnapshot {
        tensors: safetensors::serialize(
            views.iter().map(|(n, v)| (*n, v)),
            &encoded_layout.map(|text| HashMap::from([(LAYOUT_KEY.into(), text.into())])),
        )
        .unwrap(),
        metadata: 1,
    }
}

fn encoded(layout: &CandlePolicyLayout) -> String {
    serde_json::to_string(layout).unwrap()
}

#[test]
fn new_snapshots_materialize_order_aliases_and_opaque_metadata_without_clone() {
    let shared = Var::new(&[3_f32, 4.], &Device::Cpu).unwrap();
    let state = CandlePolicyState::new(
        IndexMap::from([
            ("z".into(), shared.clone()),
            ("a".into(), Var::new(&[3_f32, 4.], &Device::Cpu).unwrap()),
            ("模型\0".into(), shared.clone()),
        ]),
        1_u64,
    )
    .snapshot()
    .unwrap();
    // New reader also accepts the matching layout; a caller cannot override it.
    let expected_layout = layout(&[("z", 0), ("a", 1), ("模型\0", 0)]);
    let state: CandlePolicySnapshot<u64> =
        bincode::deserialize(&bincode::serialize(&state).unwrap()).unwrap();
    let weights = state
        .clone()
        .into_policy_weights(Some(&expected_layout))
        .unwrap();
    assert_eq!(
        weights
            .weights
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["z", "a", "模型\0"]
    );
    assert!(Arc::ptr_eq(
        &weights.weights["z"],
        &weights.weights["模型\0"]
    ));
    assert!(!Arc::ptr_eq(&weights.weights["z"], &weights.weights["a"]));
    assert_eq!(weights.weights["z"].id(), weights.weights["模型\0"].id());
    assert_ne!(weights.weights["z"].id(), weights.weights["a"].id());
    shared.set(&shared.affine(10., 0.).unwrap()).unwrap();
    assert_eq!(weights.weights["z"].to_vec1::<f32>().unwrap(), [3., 4.]);
    let independently_loaded = state.clone().into_policy_weights(None).unwrap();
    assert_ne!(
        independently_loaded.weights["z"].id(),
        weights.weights["z"].id()
    );
    let metadata = Opaque(Box::new(77));
    let pointer = std::ptr::from_ref(metadata.0.as_ref());
    let opaque = CandlePolicySnapshot {
        tensors: state.tensors,
        metadata,
    }
    .into_policy_weights(None)
    .unwrap();
    assert_eq!(std::ptr::from_ref(opaque.metadata.0.as_ref()), pointer);
    assert_eq!(*opaque.metadata.0, 77);
    let empty = CandlePolicyState::new(IndexMap::new(), 1_u64)
        .snapshot()
        .unwrap()
        .into_policy_weights(None)
        .unwrap();
    assert!(empty.weights.is_empty());
}

#[test]
fn legacy_layout_is_explicit_and_invalid_or_conflicting_manifests_fail() {
    let bytes = 3_f32.to_le_bytes();
    let view = TensorView::new(Dtype::F32, vec![1], &bytes).unwrap();
    let views = [("z", view.clone()), ("a", view)];
    let valid = layout(&[("z", 0), ("a", 1)]);
    assert!(
        snapshot(&views, None)
            .into_policy_weights(None)
            .err()
            .unwrap()
            .contains("missing policy layout")
    );
    let weights = snapshot(&views, None)
        .into_policy_weights(Some(&valid))
        .unwrap();
    assert_eq!(
        weights
            .weights
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    let stored = encoded(&valid);
    // An actual old native file has encoded names but no layout metadata.
    let old_native = CandlePolicySnapshot {
        tensors: safetensors::serialize(
            views.iter().map(|(n, v)| (format!("p{n}"), v)),
            &Some(HashMap::from([(
                super::super::NAME_ENCODING_KEY.into(),
                super::super::NAME_ENCODING_VERSION.into(),
            )])),
        )
        .unwrap(),
        metadata: 1_u64,
    };
    assert!(old_native.clone().into_policy_weights(None).is_err());
    let old_weights = old_native.into_policy_weights(Some(&valid)).unwrap();
    assert_eq!(old_weights.weights["z"].to_vec1::<f32>().unwrap(), [3.]);
    assert_eq!(
        old_weights
            .weights
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert!(
        snapshot(&views, Some(&stored))
            .into_policy_weights(Some(&layout(&[("a", 0), ("z", 1)])))
            .err()
            .unwrap()
            .contains("conflicts")
    );
    assert!(
        snapshot(&views, Some("{"))
            .into_policy_weights(Some(&valid))
            .is_err()
    );
    for (bad, message) in [
        (
            CandlePolicyLayout {
                version: 2,
                ..valid.clone()
            },
            "unsupported policy layout version:2",
        ),
        (
            layout(&[("z", 0)]),
            "policy layout does not cover all tensors",
        ),
        (
            layout(&[("z", 0), ("z", 0)]),
            "duplicate policy layout name:z",
        ),
        (
            layout(&[("z", 0), ("absent", 1)]),
            "missing policy layout tensor:absent",
        ),
        (layout(&[("z", 1), ("a", 1)]), "forward policy alias:z:1"),
        (
            layout(&[("z", 0), ("a", usize::MAX)]),
            &format!("forward policy alias:a:{}", usize::MAX),
        ),
    ] {
        assert_eq!(
            snapshot(&views, Some(&encoded(&bad)))
                .into_policy_weights(None)
                .err()
                .unwrap(),
            message
        );
    }
    assert!(
        CandlePolicySnapshot {
            tensors: vec![0],
            metadata: 1_u64
        }
        .into_policy_weights(None)
        .is_err()
    );
    assert!(read_layout(&[0], None).is_err());
}

#[test]
fn aliases_require_matching_dtype_shape_and_raw_bits_without_merging_equal_values() {
    let three = 3_f32.to_le_bytes();
    let four = 4_f32.to_le_bytes();
    let original = TensorView::new(Dtype::F32, vec![1], &three).unwrap();
    let manifest = encoded(&layout(&[("z", 0), ("a", 0)]));
    for different in [
        TensorView::new(Dtype::U32, vec![1], &three).unwrap(),
        TensorView::new(Dtype::F32, vec![], &three).unwrap(),
        TensorView::new(Dtype::F32, vec![1], &four).unwrap(),
    ] {
        assert_eq!(
            snapshot(
                &[("z", original.clone()), ("a", different)],
                Some(&manifest)
            )
            .into_policy_weights(None)
            .err()
            .unwrap(),
            "inconsistent policy alias:a"
        );
    }
    // A backward alias chain remains valid; every link must match exact bits.
    let manifest = encoded(&layout(&[("z", 0), ("a", 0), ("b", 1)]));
    let weights = snapshot(
        &[
            ("z", original.clone()),
            ("a", original.clone()),
            ("b", original),
        ],
        Some(&manifest),
    )
    .into_policy_weights(None)
    .unwrap();
    assert!(Arc::ptr_eq(&weights.weights["z"], &weights.weights["b"]));
}

#[test]
fn native_dtypes_materialize_exactly_and_unsupported_types_never_widen() {
    let manifest = encoded(&layout(&[("value", 0)]));
    for dtype in [
        DType::U8,
        DType::U32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let tensor = Tensor::arange(0_f32, 4., &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let bytes = safetensors::serialize(
            [("value", &tensor)],
            &Some(HashMap::from([(LAYOUT_KEY.into(), manifest.clone())])),
        )
        .unwrap();
        let weights = CandlePolicySnapshot {
            tensors: bytes.clone(),
            metadata: 1_u64,
        }
        .into_policy_weights(None)
        .unwrap();
        assert_eq!(weights.weights["value"].dtype(), dtype);
        assert!(weights.weights["value"].device().is_cpu());
        let reencoded =
            safetensors::serialize([("value", weights.weights["value"].as_ref())], &None).unwrap();
        assert_eq!(
            decode_tensors(&bytes).unwrap(),
            decode_tensors(&reencoded).unwrap()
        );
    }
    for (dtype, bytes) in [
        (Dtype::BOOL, vec![1]),
        (Dtype::I8, vec![1]),
        (Dtype::I16, vec![1, 0]),
        (Dtype::I32, vec![1, 0, 0, 0]),
        (Dtype::U16, vec![1, 0]),
        (Dtype::U64, vec![1, 0, 0, 0, 0, 0, 0, 0]),
    ] {
        let view = TensorView::new(dtype, vec![1], &bytes).unwrap();
        assert!(
            snapshot(&[("value", view)], Some(&manifest))
                .into_policy_weights(None)
                .is_err(),
            "{dtype:?} must not widen"
        );
    }
}

struct Retry {
    calls: usize,
}
impl PolicyWeightLoader<Tensor, u64> for Retry {
    fn load_weights(
        &mut self,
        _: &mut PolicyWeights<Tensor, u64>,
    ) -> Result<(), PolicyWeightLoadError> {
        self.calls += 1;
        if self.calls == 1 {
            Err(PolicyWeightLoadError::Runtime("legacy".into()))
        } else {
            Ok(())
        }
    }
}

#[test]
fn persisted_collision_order_drives_the_actual_retry_protocol() {
    for reverse in [false, true] {
        let mut entries = vec![
            ("a".to_owned(), Var::new(&[3_f32], &Device::Cpu).unwrap()),
            (
                "_actor_critic.a".to_owned(),
                Var::new(&[7_f32], &Device::Cpu).unwrap(),
            ),
        ];
        if reverse {
            entries.reverse();
        }
        let model = CandlePolicyState::new(entries.into_iter().collect(), 1_u64);
        let mut weights = model.snapshot().unwrap().into_policy_weights(None).unwrap();
        let original = Arc::clone(&weights.weights["a"]);
        let old_prefix = Arc::clone(&weights.weights["_actor_critic.a"]);
        let mut loader = Retry { calls: 0 };
        set_policy_weights(&mut loader, &mut weights).unwrap();
        assert_eq!(loader.calls, 2);
        assert!(Arc::ptr_eq(&weights.weights["_actor_critic.a"], &original));
        assert!(Arc::ptr_eq(
            &weights.weights["_actor_critic._actor_critic.a"],
            if reverse { &old_prefix } else { &original }
        ));
    }
}
