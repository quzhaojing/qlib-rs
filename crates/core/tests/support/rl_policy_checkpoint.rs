use super::*;
use crate::BincodeRlCheckpointCodec;
use crate::rl_candle_checkpoint::{CandlePolicySnapshot, CandlePolicyState};
use candle_core::{Device, Var};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    cell::RefCell,
    fs,
    io::{Read, Write},
    path::PathBuf,
    process::Command,
    rc::Rc,
};

type Document<P> = RlTrainerCheckpoint<TrainingVesselCheckpoint<P>, (), ()>;

// Deliberately no Clone or serialization implementation: extraction must move it.
struct Opaque(Box<u64>);

#[test]
fn projections_preserve_owned_identity_and_present_null() {
    let policy = Opaque(Box::new(7));
    let pointer = std::ptr::from_ref(policy.0.as_ref());
    let direct = DirectPolicyCheckpoint.extract(policy).unwrap();
    assert_eq!(std::ptr::from_ref(direct.0.as_ref()), pointer);
    let document = Document {
        vessel: RlCheckpointField::Present(TrainingVesselCheckpoint { policy: direct }),
        ..Document::default()
    };
    let extracted = TrainerPolicyCheckpoint.extract(document).unwrap();
    assert_eq!(std::ptr::from_ref(extracted.0.as_ref()), pointer);
    let missing: Document<Opaque> = Document::default();
    assert_eq!(
        TrainerPolicyCheckpoint.extract(missing).err().unwrap(),
        "checkpoint field is missing: vessel"
    );
    let nullable: Document<Option<Opaque>> = Document {
        vessel: RlCheckpointField::Present(TrainingVesselCheckpoint { policy: None }),
        ..Document::default()
    };
    assert!(TrainerPolicyCheckpoint.extract(nullable).unwrap().is_none());
}

struct RecordingCodec {
    events: Rc<RefCell<Vec<&'static str>>>,
}
impl RlCheckpointFileCodec<String> for RecordingCodec {
    fn encode(&mut self, state: &String, output: &mut dyn Write) -> Result<(), String> {
        output
            .write_all(state.as_bytes())
            .map_err(|e| e.to_string())
    }
    fn decode(&mut self, input: &mut dyn Read) -> Result<String, String> {
        self.events.borrow_mut().push("decode");
        let mut text = String::new();
        input.read_to_string(&mut text).map_err(|e| e.to_string())?;
        if text == "broken" {
            Err("decode failure".into())
        } else {
            Ok(text)
        }
    }
}
struct RecordingProjection(Rc<RefCell<Vec<&'static str>>>);
impl PolicyCheckpointProjection<String> for RecordingProjection {
    type Policy = String;
    fn extract(&mut self, document: String) -> Result<String, String> {
        self.0.borrow_mut().push("extract");
        if document == "no-policy" {
            Err("policy absent".into())
        } else {
            Ok(document)
        }
    }
}

#[test]
fn reader_plugins_decode_before_projection_and_forward_failures() {
    let events = Rc::new(RefCell::new(vec![]));
    let mut reader = PolicyCheckpointFile::new(
        RecordingCodec {
            events: Rc::clone(&events),
        },
        RecordingProjection(Rc::clone(&events)),
    );
    let plugin: &mut dyn PolicyCheckpointReader<String> = &mut reader;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("arbitrary.extension");
    let error = plugin.read_policy(&path).unwrap_err();
    assert!(matches!(error, PolicyCheckpointError::Read(_)));
    assert!(
        error
            .to_string()
            .starts_with("policy checkpoint read failed:")
    );
    assert!(events.borrow().is_empty());
    for (content, expected, calls) in [
        (
            "broken",
            Err(PolicyCheckpointError::Read("decode failure".into())),
            vec!["decode"],
        ),
        (
            "no-policy",
            Err(PolicyCheckpointError::Extract("policy absent".into())),
            vec!["decode", "extract"],
        ),
        ("policy", Ok("policy".into()), vec!["decode", "extract"]),
    ] {
        events.borrow_mut().clear();
        fs::write(&path, content).unwrap();
        let result = plugin.read_policy(&path);
        assert_eq!(result, expected);
        assert_eq!(*events.borrow(), calls);
    }
    assert_eq!(
        PolicyCheckpointError::Extract("policy absent".into()).to_string(),
        "policy checkpoint extraction failed: policy absent"
    );
}

fn complete_document<P>(policy: P) -> Document<P> {
    use RlCheckpointField::Present;
    Document {
        vessel: Present(TrainingVesselCheckpoint { policy }),
        callbacks: Present(IndexMap::new()),
        loggers: Present(IndexMap::new()),
        should_stop: Present(false),
        current_iter: Present(2.into()),
        current_episode: Present(100.into()),
        current_stage: Present("train".into()),
        metrics: Present(IndexMap::from([("loss".into(), 0.25)])),
    }
}

#[test]
fn native_files_extract_exact_snapshot_and_reject_truncated_trainer_tail() {
    let weight = Var::new(&[3_f32, 4.], &Device::Cpu).unwrap();
    let policy = CandlePolicyState::new(IndexMap::from([("weight".into(), weight.clone())]), 7_u64);
    let snapshot = policy.snapshot().unwrap();
    let expected = snapshot.clone();
    let full = complete_document(snapshot);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("state.unrelated");
    let mut direct = PolicyCheckpointFile::<_, _, CandlePolicySnapshot<u64>>::new(
        BincodeRlCheckpointCodec,
        DirectPolicyCheckpoint,
    );
    let mut trainer = PolicyCheckpointFile::<_, _, Document<CandlePolicySnapshot<u64>>>::new(
        BincodeRlCheckpointCodec,
        TrainerPolicyCheckpoint,
    );
    fs::write(&path, bincode::serialize(&expected).unwrap()).unwrap();
    let loaded_direct = direct.read_policy(&path).unwrap();
    assert_eq!(loaded_direct, expected);
    let bytes = bincode::serialize(&full).unwrap();
    fs::write(&path, &bytes).unwrap();
    let loaded_trainer = trainer.read_policy(&path).unwrap();
    assert_eq!(loaded_trainer, expected);
    // Loading files does not assign any variable, even if the model changed meanwhile.
    weight.set(&weight.affine(10., 0.).unwrap()).unwrap();
    let another = trainer.read_policy(&path).unwrap();
    assert_eq!(weight.to_vec1::<f32>().unwrap(), vec![30., 40.]);
    let target = Var::new(&[0_f32, 0.], &Device::Cpu).unwrap();
    let restored =
        CandlePolicyState::new(IndexMap::from([("weight".into(), target.clone())]), 99_u64);
    for state in [loaded_direct, loaded_trainer, another] {
        restored.restore(&state).unwrap();
        assert_eq!(target.to_vec1::<f32>().unwrap(), vec![3., 4.]);
        assert_eq!(
            target
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .to_bits(),
            25_f32.to_bits()
        );
    }
    // The policy prefix is still intact; the configured full document must not be.
    assert!(bytes.len() > bincode::serialize(&expected).unwrap().len());
    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(matches!(
        trainer.read_policy(&path),
        Err(PolicyCheckpointError::Read(_))
    ));
    assert_eq!(target.to_vec1::<f32>().unwrap(), vec![3., 4.]);
}

#[derive(Deserialize, Serialize)]
struct SourceCase {
    document: Value,
    trainer: bool,
    expected: Value,
}

#[test]
fn explicit_projections_match_unchanged_qlib_successful_extraction() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_policy_checkpoint_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/trainer.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<SourceCase> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let result = if case.trainer {
            let document: Document<Value> = serde_json::from_value(case.document).unwrap();
            TrainerPolicyCheckpoint.extract(document).unwrap()
        } else {
            DirectPolicyCheckpoint.extract(case.document).unwrap()
        };
        assert_eq!(result, case.expected);
    }
}
