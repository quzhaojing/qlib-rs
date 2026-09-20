use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use super::{
    DataQueue, DataQueueConfig, DataQueueDataset, DataQueueError, DataQueueProducerError,
    DataQueueSampler, RandomDataQueueSampler, SequentialDataQueueSampler,
};
use rand::seq::SliceRandom;
use serde_json::Value;

fn config(repeat: i64, shuffle: bool, workers: usize, capacity: usize) -> DataQueueConfig {
    DataQueueConfig {
        repeat,
        shuffle,
        producer_num_workers: workers,
        queue_maxsize: capacity,
    }
}

#[derive(Clone)]
struct DelayedDataset {
    values: Vec<i32>,
    fail: Option<usize>,
    panic: Option<usize>,
}

impl DataQueueDataset<i32> for DelayedDataset {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn get(&self, index: usize) -> Result<i32, String> {
        assert_ne!(self.panic, Some(index), "dataset panic");
        if self.fail == Some(index) {
            return Err("load failed".into());
        }
        thread::sleep(Duration::from_millis(
            u64::try_from(self.values.len() - index).unwrap(),
        ));
        Ok(self.values[index])
    }
}

struct ScriptedSampler {
    epochs: Arc<Mutex<Vec<u64>>>,
    result: Result<Vec<usize>, String>,
}

impl DataQueueSampler for ScriptedSampler {
    fn indices(&mut self, _dataset_len: usize, epoch: u64) -> Result<Vec<usize>, String> {
        self.epochs.lock().unwrap().push(epoch);
        self.result.clone()
    }
}

fn collect(queue: &mut DataQueue<i32>) -> Result<Vec<i32>, DataQueueError> {
    queue.consumer()?.collect()
}

#[test]
fn defaults_datasets_and_bundled_samplers_are_typed() {
    let defaults = DataQueueConfig::default();
    assert_eq!(defaults, config(1, true, 0, 0));
    assert!(defaults.effective_queue_maxsize() > 0);
    assert_eq!(config(1, false, 0, 7).effective_queue_maxsize(), 7);

    let values = vec![4, 5, 6];
    assert_eq!(DataQueueDataset::len(&values), 3);
    assert!(!DataQueueDataset::is_empty(&values));
    assert_eq!(DataQueueDataset::get(&values, 1).unwrap(), 5);
    assert!(DataQueueDataset::get(&values, 3).unwrap_err().contains('3'));
    let empty: Vec<i32> = Vec::new();
    assert!(DataQueueDataset::is_empty(&empty));

    let mut sequential = SequentialDataQueueSampler;
    assert_eq!(sequential.indices(3, 9).unwrap(), vec![0, 1, 2]);
    let mut random = RandomDataQueueSampler;
    let mut sampled = random.indices(8, 0).unwrap();
    sampled.sort_unstable();
    assert_eq!(sampled, (0..8).collect::<Vec<_>>());

    let mut direct = vec![1, 2, 3];
    direct.shuffle(&mut rand::rng());
    direct.sort_unstable();
    assert_eq!(direct, vec![1, 2, 3]);
}

#[test]
fn sequential_repeats_activation_iteration_and_manual_put_match_lifecycle() {
    let mut queue = DataQueue::new(Arc::new(vec![1, 2]), config(2, false, 0, 8));
    assert!(!queue.is_activated());
    assert!(matches!(
        queue.consumer(),
        Err(DataQueueError::NotActivated)
    ));
    queue.put(0).unwrap();
    let queue_address = std::ptr::from_ref(&queue);
    let activated_address = std::ptr::from_mut(queue.activate().unwrap()).cast_const();
    assert_eq!(activated_address, queue_address);
    assert!(queue.is_activated());
    assert_eq!(collect(&mut queue).unwrap(), vec![0, 1, 2, 1, 2]);
    assert!(queue.done());
    assert_eq!(queue.producer_error(), None);
    assert!(matches!(
        queue.activate(),
        Err(DataQueueError::AlreadyActivated)
    ));
    queue.cleanup();
    queue.cleanup();
    assert!(matches!(queue.put(9), Err(DataQueueError::Producer(_))));
}

#[test]
fn zero_negative_and_empty_repeats_exhaust_without_values() {
    for repeat in [0, -2] {
        let mut queue = DataQueue::new(Arc::new(vec![1]), config(repeat, false, 0, 1));
        queue.activate().unwrap();
        assert_eq!(collect(&mut queue).unwrap(), Vec::<i32>::new());
    }
    let mut empty = DataQueue::new(Arc::new(Vec::<i32>::new()), config(3, false, 2, 1));
    empty.activate().unwrap();
    assert!(matches!(empty.get(), Err(DataQueueError::Exhausted)));
}

#[test]
fn custom_sampler_and_parallel_workers_preserve_epoch_and_output_order() {
    let epochs = Arc::new(Mutex::new(Vec::new()));
    let sampler = ScriptedSampler {
        epochs: Arc::clone(&epochs),
        result: Ok(vec![3, 1, 2, 0]),
    };
    let dataset = DelayedDataset {
        values: vec![10, 11, 12, 13],
        fail: None,
        panic: None,
    };
    assert!(!DataQueueDataset::is_empty(&dataset));
    let mut queue =
        DataQueue::with_sampler(Arc::new(dataset), config(2, true, 3, 2), Box::new(sampler));
    queue.activate().unwrap();
    assert_eq!(
        collect(&mut queue).unwrap(),
        vec![13, 11, 12, 10, 13, 11, 12, 10]
    );
    assert_eq!(*epochs.lock().unwrap(), vec![0, 1]);
}

#[test]
fn infinite_production_honors_bounded_backpressure_and_cleanup() {
    let mut queue = DataQueue::new(Arc::new(vec![7, 8]), config(-1, false, 2, 1));
    queue.activate().unwrap();
    assert_eq!(queue.get().unwrap(), 7);
    assert_eq!(queue.get().unwrap(), 8);
    assert_eq!(queue.get().unwrap(), 7);
    queue.cleanup();
    assert!(queue.done());
    assert!(matches!(queue.get(), Err(DataQueueError::Exhausted)));
}

#[test]
fn random_construction_send_timeout_and_both_cancellation_modes_finish() {
    let mut shuffled = DataQueue::new(Arc::new(vec![1, 2, 3]), config(1, true, 0, 3));
    shuffled.activate().unwrap();
    let mut values = collect(&mut shuffled).unwrap();
    values.sort_unstable();
    assert_eq!(values, vec![1, 2, 3]);

    let slow = DelayedDataset {
        values: (0..20).collect(),
        fail: None,
        panic: None,
    };
    let mut sequential = DataQueue::new(Arc::new(slow.clone()), config(1, false, 0, 1));
    sequential.activate().unwrap();
    thread::sleep(Duration::from_millis(25));
    sequential.cleanup();

    let mut parallel = DataQueue::new(Arc::new(slow), config(1, false, 4, 1));
    parallel.activate().unwrap();
    thread::sleep(Duration::from_millis(25));
    parallel.cleanup();
    assert!(sequential.done() && parallel.done());
}

#[test]
fn cleanup_before_activation_closes_the_ephemeral_queue() {
    let mut queue = DataQueue::new(Arc::new(vec![1]), config(1, false, 0, 1));
    queue.cleanup();
    assert!(matches!(queue.activate(), Err(DataQueueError::Closed)));
}

#[test]
fn sampler_index_dataset_and_panic_failures_are_reported_once() {
    let cases: Vec<(
        Box<dyn DataQueueSampler>,
        DelayedDataset,
        DataQueueProducerError,
    )> = vec![
        (
            Box::new(ScriptedSampler {
                epochs: Arc::new(Mutex::new(Vec::new())),
                result: Err("sample failed".into()),
            }),
            DelayedDataset {
                values: vec![1],
                fail: None,
                panic: None,
            },
            DataQueueProducerError::Sampler {
                epoch: 0,
                message: "sample failed".into(),
            },
        ),
        (
            Box::new(ScriptedSampler {
                epochs: Arc::new(Mutex::new(Vec::new())),
                result: Ok(vec![1]),
            }),
            DelayedDataset {
                values: vec![1],
                fail: None,
                panic: None,
            },
            DataQueueProducerError::InvalidSampleIndex { index: 1, len: 1 },
        ),
        (
            Box::new(SequentialDataQueueSampler),
            DelayedDataset {
                values: vec![1],
                fail: Some(0),
                panic: None,
            },
            DataQueueProducerError::Dataset {
                index: 0,
                message: "load failed".into(),
            },
        ),
        (
            Box::new(SequentialDataQueueSampler),
            DelayedDataset {
                values: vec![1],
                fail: None,
                panic: Some(0),
            },
            DataQueueProducerError::Panicked,
        ),
    ];
    for (sampler, dataset, expected) in cases {
        let mut queue = DataQueue::with_sampler(Arc::new(dataset), config(1, false, 0, 1), sampler);
        queue.activate().unwrap();
        assert_eq!(queue.get(), Err(DataQueueError::Producer(expected.clone())));
        assert_eq!(queue.producer_error(), Some(expected));
        let mut iterator = queue.consumer().unwrap();
        assert!(iterator.next().unwrap().is_err());
        assert!(iterator.next().is_none());
    }
}

#[test]
fn parallel_dataset_failure_is_typed() {
    let dataset = DelayedDataset {
        values: vec![1, 2, 3],
        fail: Some(1),
        panic: None,
    };
    let mut queue = DataQueue::new(Arc::new(dataset), config(1, false, 2, 2));
    queue.activate().unwrap();
    assert_eq!(queue.get().unwrap(), 1);
    assert_eq!(
        queue.get(),
        Err(DataQueueError::Producer(DataQueueProducerError::Dataset {
            index: 1,
            message: "load failed".into(),
        }))
    );
}

#[test]
fn parallel_worker_panic_is_captured() {
    let dataset = DelayedDataset {
        values: vec![1, 2],
        fail: None,
        panic: Some(0),
    };
    let mut queue = DataQueue::new(Arc::new(dataset), config(1, false, 2, 1));
    queue.activate().unwrap();
    assert_eq!(
        queue.get(),
        Err(DataQueueError::Producer(DataQueueProducerError::Panicked))
    );
}

#[test]
fn live_python_source_contract_matches_the_rust_boundary() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/data_queue_contract.py");
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/rl/utils/data_queue.py");
    let output = Command::new("python")
        .args([fixture, source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["defaults"], serde_json::json!([1, true, 0, 0]));
    assert_eq!(value["timeouts"], serde_json::json!([5.0, 0.5]));
    assert_eq!(value["infinite_repeat_power"], true);
    assert_eq!(value["daemon_thread"], true);
    assert_eq!(value["loader_keywords"]["batch_size"], Value::Null);
    assert!(
        value["methods"]
            .as_array()
            .unwrap()
            .iter()
            .any(|name| name == "cleanup")
    );
}
