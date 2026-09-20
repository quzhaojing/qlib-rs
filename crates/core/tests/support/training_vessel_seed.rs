use std::{path::PathBuf, process::Command, sync::Mutex};

use super::*;

#[path = "training_vessel_seed_live.rs"]
mod live;

struct LogCapture(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for LogCapture {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        event.record(
            &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                self.0.lock().unwrap().push(format!("{field}={value:?}"));
            },
        );
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

#[test]
fn default_logger_delivers_structured_diagnostics_and_noop_is_optional() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let _subscriber = tracing::subscriber::set_default(LogCapture(Arc::clone(&events)));
    let mut seeds = TrainingVesselSeeds::new(None, Some(Arc::new(vec![3_i32])), None, Some(1));
    assert_eq!(consume(seeds.validation_seed_queue().unwrap()), vec![3]);
    let fields = events.lock().unwrap().clone();
    for expected in ["phase=val", "size=1", "original_size=1", "subset_size=1"] {
        assert!(fields.iter().any(|field| field == expected), "{fields:?}");
    }
    let mut noop = NoopTrainingVesselSeedLogger;
    noop.collection_size(TrainingSeedPhase::Train, 4).unwrap();
    noop.fast_development_subset(TrainingSeedPhase::Train, 4, 2)
        .unwrap();
    assert_eq!(*events.lock().unwrap(), fields);
}

struct FixedSampler(Result<Vec<usize>, String>);

impl DataQueueSampler for FixedSampler {
    fn indices(&mut self, _len: usize, epoch: u64) -> Result<Vec<usize>, String> {
        assert_eq!(epoch, 0);
        self.0.clone()
    }
}

type Events = Arc<Mutex<Vec<(TrainingSeedPhase, usize, Option<usize>)>>>;

struct Logger {
    events: Events,
    fail: Option<TrainingSeedLogEvent>,
}

impl TrainingVesselSeedLogger for Logger {
    fn collection_size(&mut self, phase: TrainingSeedPhase, size: usize) -> Result<(), String> {
        self.events.lock().unwrap().push((phase, size, None));
        if self.fail == Some(TrainingSeedLogEvent::CollectionSize) {
            Err("collection log".into())
        } else {
            Ok(())
        }
    }

    fn fast_development_subset(
        &mut self,
        phase: TrainingSeedPhase,
        original: usize,
        subset: usize,
    ) -> Result<(), String> {
        self.events
            .lock()
            .unwrap()
            .push((phase, original, Some(subset)));
        if self.fail == Some(TrainingSeedLogEvent::FastDevelopmentSubset) {
            Err("subset log".into())
        } else {
            Ok(())
        }
    }
}

fn vessel(
    size: Option<i64>,
    permutation: Result<Vec<usize>, String>,
    fail: Option<TrainingSeedLogEvent>,
) -> (TrainingVesselSeeds<i32>, Events) {
    let values = Arc::new(vec![10, 20, 30, 40]);
    let events = Arc::new(Mutex::new(Vec::new()));
    (
        TrainingVesselSeeds::with_plugins(
            Some(Arc::clone(&values)),
            Some(Arc::clone(&values)),
            Some(values),
            size,
            Box::new(FixedSampler(permutation)),
            Box::new(Logger {
                events: Arc::clone(&events),
                fail,
            }),
        ),
        events,
    )
}

fn consume(mut queue: DataQueue<i32>) -> Vec<i32> {
    assert!(!queue.is_activated());
    queue.activate().unwrap();
    let mut values = queue
        .consumer()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    values.sort_unstable();
    values
}

#[test]
fn phases_defaults_and_real_queues_preserve_repeat_policy() {
    for (phase, name) in [
        (TrainingSeedPhase::Train, "train"),
        (TrainingSeedPhase::Val, "val"),
        (TrainingSeedPhase::Test, "test"),
    ] {
        assert_eq!(phase.to_string(), name);
        assert_eq!(name.parse::<TrainingSeedPhase>().unwrap(), phase);
    }
    assert!("invalid".parse::<TrainingSeedPhase>().is_err());
    assert_eq!(
        TrainingSeedLogEvent::CollectionSize.to_string(),
        "collection_size"
    );
    assert_eq!(
        TrainingSeedLogEvent::FastDevelopmentSubset.to_string(),
        "fast_development_subset"
    );
    let values = Arc::new(vec![3]);
    let mut seeds = TrainingVesselSeeds::new(
        Some(Arc::clone(&values)),
        Some(Arc::clone(&values)),
        Some(values),
        None,
    );
    assert_eq!(seeds.fast_dev_run(), None);
    let mut train = seeds.train_seed_queue().unwrap();
    train.activate().unwrap();
    assert_eq!((train.get().unwrap(), train.get().unwrap()), (3, 3));
    train.cleanup();
    assert_eq!(consume(seeds.validation_seed_queue().unwrap()), vec![3]);
    assert_eq!(consume(seeds.test_seed_queue().unwrap()), vec![3]);
    seeds.set_fast_dev_run(Some(1));
    assert_eq!(seeds.fast_dev_run(), Some(1));
    assert_eq!(consume(seeds.test_seed_queue().unwrap()), vec![3]);
}

#[test]
fn python_slice_bounds_and_identity_are_preserved() {
    for (size, expected) in [
        (None, vec![10, 20, 30, 40]),
        (Some(0), vec![]),
        (Some(2), vec![40, 30]),
        (Some(-1), vec![40, 30, 20]),
        (Some(10), vec![40, 30, 20, 10]),
        (Some(-10), vec![]),
        (Some(i64::MIN), vec![]),
        (Some(i64::MAX), vec![40, 30, 20, 10]),
    ] {
        let (mut seeds, events) = vessel(size, Ok(vec![3, 2, 1, 0]), None);
        let original = seeds.collection(TrainingSeedPhase::Val).unwrap();
        let selected = seeds
            .random_subset(TrainingSeedPhase::Val, Arc::clone(&original), size)
            .unwrap();
        assert_eq!(*selected, expected);
        assert_eq!(Arc::ptr_eq(&original, &selected), size.is_none());
        assert_eq!(events.lock().unwrap().len(), usize::from(size.is_some()));
    }
    let mut empty =
        TrainingVesselSeeds::<i32>::new(None, Some(Arc::new(Vec::new())), None, Some(2));
    assert!(consume(empty.validation_seed_queue().unwrap()).is_empty());
}

#[test]
fn queue_construction_logs_before_and_after_selection() {
    let (mut seeds, events) = vessel(Some(2), Ok(vec![3, 2, 1, 0]), None);
    assert_eq!(
        consume(seeds.validation_seed_queue().unwrap()),
        vec![30, 40]
    );
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            (TrainingSeedPhase::Val, 4, None),
            (TrainingSeedPhase::Val, 4, Some(2))
        ]
    );
    let (mut seeds, events) = vessel(None, Err("must not sample".into()), None);
    assert_eq!(
        consume(seeds.test_seed_queue().unwrap()),
        vec![10, 20, 30, 40]
    );
    assert_eq!(
        *events.lock().unwrap(),
        vec![(TrainingSeedPhase::Test, 4, None)]
    );
}

#[test]
fn missing_collections_and_all_plugin_failures_are_typed() {
    let mut empty = TrainingVesselSeeds::<i32>::new(None, None, None, None);
    for phase in [
        TrainingSeedPhase::Train,
        TrainingSeedPhase::Val,
        TrainingSeedPhase::Test,
    ] {
        assert!(
            matches!(empty.seed_queue(phase), Err(TrainingVesselSeedError::SeedIteratorNotAvailable { phase: actual }) if actual == phase)
        );
    }
    for (permutation, expected) in [
        (
            Err("sampler".into()),
            TrainingVesselSeedError::Sampler {
                phase: TrainingSeedPhase::Val,
                message: "sampler".into(),
            },
        ),
        (
            Ok(vec![0]),
            TrainingVesselSeedError::InvalidPermutationLength {
                phase: TrainingSeedPhase::Val,
                expected: 4,
                actual: 1,
            },
        ),
        (
            Ok(vec![0, 1, 2, 4]),
            TrainingVesselSeedError::InvalidPermutationIndex {
                phase: TrainingSeedPhase::Val,
                index: 4,
                len: 4,
            },
        ),
        (
            Ok(vec![0, 1, 2, 2]),
            TrainingVesselSeedError::DuplicatePermutationIndex {
                phase: TrainingSeedPhase::Val,
                index: 2,
            },
        ),
    ] {
        let (mut seeds, events) = vessel(Some(2), permutation, None);
        match seeds.validation_seed_queue() {
            Err(error) => assert_eq!(error, expected),
            Ok(_) => panic!("expected failure"),
        }
        assert_eq!(events.lock().unwrap().len(), 1);
    }
    for event in [
        TrainingSeedLogEvent::CollectionSize,
        TrainingSeedLogEvent::FastDevelopmentSubset,
    ] {
        let (mut seeds, _) = vessel(Some(2), Ok(vec![3, 2, 1, 0]), Some(event));
        assert!(
            matches!(seeds.validation_seed_queue(), Err(TrainingVesselSeedError::Logger { phase: TrainingSeedPhase::Val, event: actual, .. }) if actual == event)
        );
    }
}

#[test]
fn subset_values_and_queue_defaults_match_live_python_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/training_vessel_seed_contract.py"),
            root.join("../../../qlib/qlib/rl/trainer/vessel.py"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    for case in result["subsets"].as_array().unwrap() {
        let size = case["size"].as_i64();
        let (mut seeds, _) = vessel(size, Ok(vec![3, 2, 1, 0]), None);
        let original = seeds.collection(TrainingSeedPhase::Val).unwrap();
        let selected = seeds
            .random_subset(TrainingSeedPhase::Val, Arc::clone(&original), size)
            .unwrap();
        assert_eq!(
            serde_json::to_value(selected.as_ref()).unwrap(),
            case["values"]
        );
        assert_eq!(
            Arc::ptr_eq(&original, &selected),
            case["same_object"].as_bool().unwrap()
        );
    }
    assert_eq!(
        result["queues"]["train_seed_iterator"],
        serde_json::json!({"repeat":-1,"shuffle":true})
    );
    assert_eq!(
        result["queues"]["val_seed_iterator"],
        serde_json::json!({"repeat":1})
    );
    assert_eq!(
        result["queues"]["test_seed_iterator"],
        serde_json::json!({"repeat":1})
    );
    assert_eq!(
        result["permutation_calls"],
        serde_json::json!([4, 4, 4, 4, 4])
    );
    assert_eq!(result["log_count"], 5);
    assert_eq!(
        result["missing"],
        serde_json::json!([
            "Seed iterator for training is not available.",
            "Seed iterator for validation is not available.",
            "Seed iterator for testing is not available."
        ])
    );
}
