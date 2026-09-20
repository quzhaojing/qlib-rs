use std::{
    path::PathBuf,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use domain_core::{
    EnvironmentObservationSpace, EnvironmentPluginError, FiniteObservation, FiniteObservationError,
    SampledFiniteObservationSpace, check_nan_observation, fill_invalid, is_invalid,
};
use half::f16;
use indexmap::indexmap;
use ndarray::{ArrayD, IxDyn, array};
use serde_json::Value;

fn python_contract() -> Value {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/finite_observation_contract.py");
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/utils/finite_env.py");
    let output = Command::new("python")
        .arg(fixture)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn assert_pair(source: &FiniteObservation, dtype: &str, python: &Value) {
    assert!(!is_invalid(source).unwrap());
    let invalid = fill_invalid(source).unwrap();
    assert!(is_invalid(&invalid).unwrap());
    assert_eq!(python["arrays"][dtype]["invalid"], true);
    assert_eq!(python["arrays"][dtype]["source_invalid"], false);
    assert_eq!(
        python["arrays"][dtype]["filled"]["shape"],
        serde_json::json!([2, 2])
    );
}

#[test]
fn every_numpy_numeric_dtype_matches_live_python_sentinels() {
    let python = python_contract();
    assert_pair(
        &FiniteObservation::Float16(
            array![[0.0_f32, 1.0], [2.0, 3.0]]
                .mapv(f16::from_f32)
                .into_dyn(),
        ),
        "float16",
        &python,
    );
    assert_pair(
        &FiniteObservation::Float32(array![[0.0_f32, 1.0], [2.0, 3.0]].into_dyn()),
        "float32",
        &python,
    );
    assert_pair(
        &FiniteObservation::Float64(array![[0.0_f64, 1.0], [2.0, 3.0]].into_dyn()),
        "float64",
        &python,
    );
    assert_pair(
        &FiniteObservation::Int8(array![[0_i8, 1], [2, 3]].into_dyn()),
        "int8",
        &python,
    );
    assert_pair(
        &FiniteObservation::Int16(array![[0_i16, 1], [2, 3]].into_dyn()),
        "int16",
        &python,
    );
    assert_pair(
        &FiniteObservation::Int32(array![[0_i32, 1], [2, 3]].into_dyn()),
        "int32",
        &python,
    );
    assert_pair(
        &FiniteObservation::Int64(array![[0_i64, 1], [2, 3]].into_dyn()),
        "int64",
        &python,
    );
    assert_pair(
        &FiniteObservation::UInt8(array![[0_u8, 1], [2, 3]].into_dyn()),
        "uint8",
        &python,
    );
    assert_pair(
        &FiniteObservation::UInt16(array![[0_u16, 1], [2, 3]].into_dyn()),
        "uint16",
        &python,
    );
    assert_pair(
        &FiniteObservation::UInt32(array![[0_u32, 1], [2, 3]].into_dyn()),
        "uint32",
        &python,
    );
    assert_pair(
        &FiniteObservation::UInt64(array![[0_u64, 1], [2, 3]].into_dyn()),
        "uint64",
        &python,
    );

    assert_eq!(
        python["scalar_float"]["value"]["shape"],
        serde_json::json!([])
    );
    assert_eq!(
        python["scalar_int"]["value"]["shape"],
        serde_json::json!([])
    );
    let scalar = FiniteObservation::Float64(ArrayD::from_elem(IxDyn(&[]), 1.5));
    let FiniteObservation::Float64(scalar) = fill_invalid(&scalar).unwrap() else {
        panic!("float64 scalar dtype changed");
    };
    assert_eq!(scalar.ndim(), 0);
    assert!(scalar[IxDyn(&[])].is_nan());
}

#[test]
fn nested_container_order_shape_and_short_circuit_match_python() {
    let python = python_contract();
    let source = FiniteObservation::Map(indexmap! {
        "float".to_owned() => FiniteObservation::Float32(array![1.0_f32].into_dyn()),
        "children".to_owned() => FiniteObservation::List(vec![
            FiniteObservation::Int16(ArrayD::from_elem(IxDyn(&[]), 1_i16)),
            FiniteObservation::Tuple(vec![FiniteObservation::UInt8(array![2_u8].into_dyn())]),
        ]),
    });
    let invalid = fill_invalid(&source).unwrap();
    assert!(check_nan_observation(&invalid).unwrap());
    assert_eq!(python["nested_invalid"], true);
    assert_eq!(python["generated_check"], true);
    assert_eq!(python["nested"], python["generated"]);

    let FiniteObservation::Map(values) = invalid else {
        panic!("map container changed");
    };
    assert_eq!(
        values.keys().map(String::as_str).collect::<Vec<_>>(),
        ["float", "children"]
    );

    assert!(is_invalid(&FiniteObservation::Map(indexmap! {})).unwrap());
    assert!(is_invalid(&FiniteObservation::List(Vec::new())).unwrap());
    assert!(is_invalid(&FiniteObservation::Tuple(Vec::new())).unwrap());
    let short_circuit = FiniteObservation::List(vec![
        FiniteObservation::Float64(array![0.0_f64].into_dyn()),
        FiniteObservation::Bool(array![true].into_dyn()),
    ]);
    assert!(!is_invalid(&short_circuit).unwrap());
    assert_eq!(python["short_circuit"], false);

    let map_short_circuit = FiniteObservation::Map(indexmap! {
        "valid".to_owned() => FiniteObservation::Int8(array![0_i8].into_dyn()),
        "unsupported".to_owned() => FiniteObservation::Bool(array![true].into_dyn()),
    });
    assert!(!is_invalid(&map_short_circuit).unwrap());
}

#[test]
fn unsupported_numpy_and_opaque_edges_match_python_errors() {
    let python = python_contract();
    assert_eq!(python["scalar_bool"]["error"], "ValueError");
    assert_eq!(python["bool_array"]["error"], "ValueError");
    assert_eq!(python["complex_array"]["error"], "ValueError");
    assert_eq!(python["opaque_fill"]["error"], "ValueError");
    assert_eq!(python["opaque_invalid"], true);
    assert_eq!(python["empty_float_invalid"], true);
    assert_eq!(python["empty_dict_invalid"], true);

    let boolean = FiniteObservation::Bool(array![true].into_dyn());
    assert_eq!(
        fill_invalid(&boolean),
        Err(FiniteObservationError::UnsupportedDtype("bool".to_owned()))
    );
    assert_eq!(
        is_invalid(&boolean),
        Err(FiniteObservationError::UnsupportedDtype("bool".to_owned()))
    );
    let complex = FiniteObservation::UnsupportedArray {
        dtype: "complex64".to_owned(),
    };
    assert_eq!(
        fill_invalid(&complex),
        Err(FiniteObservationError::UnsupportedDtype(
            "complex64".to_owned()
        ))
    );
    assert_eq!(
        is_invalid(&complex),
        Err(FiniteObservationError::UnsupportedDtype(
            "complex64".to_owned()
        ))
    );
    let opaque = FiniteObservation::Opaque {
        description: "opaque".to_owned(),
    };
    assert_eq!(
        fill_invalid(&opaque),
        Err(FiniteObservationError::UnsupportedValue(
            "opaque".to_owned()
        ))
    );
    assert!(is_invalid(&opaque).unwrap());
    assert!(is_invalid(&FiniteObservation::Float64(ArrayD::zeros(IxDyn(&[0])))).unwrap());

    let nested_error = FiniteObservation::Tuple(vec![
        FiniteObservation::UInt8(array![1_u8].into_dyn()),
        boolean,
    ]);
    assert!(matches!(
        fill_invalid(&nested_error),
        Err(FiniteObservationError::UnsupportedDtype(dtype)) if dtype == "bool"
    ));

    let map_error = FiniteObservation::Map(indexmap! {
        "boolean".to_owned() => FiniteObservation::Bool(array![true].into_dyn()),
    });
    assert!(matches!(
        fill_invalid(&map_error),
        Err(FiniteObservationError::UnsupportedDtype(dtype)) if dtype == "bool"
    ));
    assert!(matches!(
        is_invalid(&map_error),
        Err(FiniteObservationError::UnsupportedDtype(dtype)) if dtype == "bool"
    ));
    assert!(matches!(
        is_invalid(&FiniteObservation::List(vec![FiniteObservation::Bool(
            array![true].into_dyn()
        )])),
        Err(FiniteObservationError::UnsupportedDtype(dtype)) if dtype == "bool"
    ));
}

#[test]
fn sampled_space_adapts_sampling_and_conversion_failures() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sampled_calls = Arc::clone(&calls);
    let mut space = SampledFiniteObservationSpace::new(move || {
        sampled_calls.fetch_add(1, Ordering::SeqCst);
        Ok(FiniteObservation::Int32(array![1_i32, 2].into_dyn()))
    });
    let observation = space.invalid_observation().unwrap();
    assert!(is_invalid(&observation).unwrap());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let mut sampling_failure =
        SampledFiniteObservationSpace::new(|| Err(EnvironmentPluginError::new("sample failed")));
    assert_eq!(
        sampling_failure.invalid_observation().unwrap_err().message,
        "sample failed"
    );

    let mut conversion_failure = SampledFiniteObservationSpace::new(|| {
        Ok(FiniteObservation::Opaque {
            description: "object".to_owned(),
        })
    });
    assert!(
        conversion_failure
            .invalid_observation()
            .unwrap_err()
            .message
            .contains("unsupported value")
    );
}
