use super::*;
use crate::{TrainingTrainerField, TrainingTrainerView};
use serde_json::{Value, json};

type Calls = Arc<Mutex<Vec<String>>>;

struct Trainer {
    calls: Calls,
    size: Mutex<Result<Option<i64>, String>>,
}
impl TrainingTrainerView for Trainer {
    fn current_iteration(&self) -> Result<num_bigint::BigInt, String> {
        panic!("seed selection must not read the iteration")
    }
    fn fast_dev_run(&self) -> Result<Option<i64>, String> {
        self.calls.lock().unwrap().push("trainer".into());
        self.size.lock().unwrap().clone()
    }
}
struct SeedLogger(Calls, bool);
impl TrainingVesselSeedLogger for SeedLogger {
    fn collection_size(&mut self, _: TrainingSeedPhase, _: usize) -> Result<(), String> {
        self.0.lock().unwrap().push("log".into());
        if self.1 { Err("logger".into()) } else { Ok(()) }
    }
    fn fast_development_subset(
        &mut self,
        _: TrainingSeedPhase,
        _: usize,
        _: usize,
    ) -> Result<(), String> {
        Ok(())
    }
}
struct Sampler(Calls, bool);
impl DataQueueSampler for Sampler {
    fn indices(&mut self, len: usize, _: u64) -> Result<Vec<usize>, String> {
        self.0.lock().unwrap().push("subset".into());
        if self.1 {
            Err("subset".into())
        } else {
            Ok((0..len).collect())
        }
    }
}

fn source_cases() -> Vec<Value> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/training_vessel_live_seed_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/vessel.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn live_queue_construction_preserves_source_order_and_each_failure() {
    for case in source_cases() {
        let phase: TrainingSeedPhase = case["phase"].as_str().unwrap().parse().unwrap();
        let fail = case["case"].as_str().unwrap();
        let calls = Calls::default();
        let trainer: Arc<dyn TrainingTrainerView> = Arc::new(Trainer {
            calls: calls.clone(),
            size: Mutex::new(if fail == "trainer" {
                Err("trainer".into())
            } else {
                Ok(Some(2))
            }),
        });
        let mut binding = TrainingVesselBinding::default();
        binding.assign_trainer(&trainer);
        let values = (fail != "missing").then(|| Arc::new(vec![1_i32, 2, 3]));
        let mut seeds = TrainingVesselSeeds::with_plugins(
            values.clone(),
            values.clone(),
            values,
            Some(0),
            Box::new(Sampler(calls.clone(), fail == "subset")),
            Box::new(SeedLogger(calls.clone(), fail == "logger")),
        );
        let result = seeds.seed_queue_with_trainer(phase, &binding);
        assert_eq!(result.is_ok(), case["ok"].as_bool().unwrap());
        assert_eq!(json!(*calls.lock().unwrap()), case["events"]);
        match result {
            Ok(mut queue) => {
                assert!(!queue.is_activated());
                queue.activate().unwrap();
                let mut values = vec![queue.get().unwrap(), queue.get().unwrap()];
                values.sort_unstable();
                assert_eq!(json!(values), case["items"]);
                if phase == TrainingSeedPhase::Train {
                    assert!([1, 2].contains(&queue.get().unwrap()));
                } else {
                    assert!(matches!(queue.get(), Err(crate::DataQueueError::Exhausted)));
                }
                queue.cleanup();
            }
            Err(error) => {
                let expected = match fail {
                    "missing" => TrainingVesselSeedError::SeedIteratorNotAvailable { phase },
                    "logger" => TrainingVesselSeedError::Logger {
                        phase,
                        event: TrainingSeedLogEvent::CollectionSize,
                        message: "logger".into(),
                    },
                    "trainer" => TrainingVesselSeedError::Trainer {
                        phase,
                        source: TrainingVesselBindingError::Access {
                            field: TrainingTrainerField::FastDevRun,
                            message: "trainer".into(),
                        },
                    },
                    "subset" => TrainingVesselSeedError::Sampler {
                        phase,
                        message: "subset".into(),
                    },
                    other => panic!("unexpected error case {other}: {error}"),
                };
                assert_eq!(error, expected);
            }
        }
        assert_eq!(seeds.fast_dev_run(), Some(0));
    }
}

#[test]
fn live_settings_refresh_without_mutating_standalone_configuration_or_pinning_trainer() {
    let calls = Calls::default();
    let trainer = Arc::new(Trainer {
        calls: calls.clone(),
        size: Mutex::new(Ok(Some(1))),
    });
    let view: Arc<dyn TrainingTrainerView> = trainer.clone();
    let mut binding = TrainingVesselBinding::default();
    binding.assign_trainer(&view);
    drop(view);
    let mut seeds = TrainingVesselSeeds::with_plugins(
        None,
        Some(Arc::new(vec![1, 2, 3])),
        None,
        Some(0),
        Box::new(Sampler(calls.clone(), false)),
        Box::new(SeedLogger(calls.clone(), false)),
    );
    for (size, expected) in [
        (Some(1), vec![1]),
        (Some(2), vec![1, 2]),
        (None, vec![1, 2, 3]),
    ] {
        *trainer.size.lock().unwrap() = Ok(size);
        assert_eq!(
            consume(
                seeds
                    .seed_queue_with_trainer(TrainingSeedPhase::Val, &binding)
                    .unwrap()
            ),
            expected
        );
        assert_eq!(seeds.fast_dev_run(), Some(0));
    }
    assert!(consume(seeds.validation_seed_queue().unwrap()).is_empty());
    drop(trainer);
    for (binding, source) in [
        (&binding, TrainingVesselBindingError::Expired),
        (
            &TrainingVesselBinding::default(),
            TrainingVesselBindingError::Unassigned,
        ),
    ] {
        calls.lock().unwrap().clear();
        let error = seeds
            .seed_queue_with_trainer(TrainingSeedPhase::Val, binding)
            .err()
            .unwrap();
        assert_eq!(
            error,
            TrainingVesselSeedError::Trainer {
                phase: TrainingSeedPhase::Val,
                source
            }
        );
        assert_eq!(*calls.lock().unwrap(), ["log"]);
    }
}
