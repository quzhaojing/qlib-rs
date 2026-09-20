use super::*;
use crate::rl_candle_network_fixture::Record;
use candle_core::{DType, Device, Var};
use indexmap::IndexMap;
use num_traits::ToPrimitive;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

#[derive(Deserialize)]
struct Step {
    observation: IndexMap<String, Record>,
    next_observation: IndexMap<String, Record>,
    action: Record,
    reward: f64,
    terminated: bool,
    truncated: bool,
    added: Vec<f64>,
    indices: Vec<usize>,
    stored_observation: IndexMap<String, Record>,
    stored_next: IndexMap<String, Record>,
    stored_actions: Record,
    stored_rewards: Vec<f64>,
    stored_terminated: Vec<bool>,
    stored_truncated: Vec<bool>,
    unfinished: Vec<usize>,
}
#[derive(Deserialize)]
struct Case {
    capacity: usize,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Failure {
    field: String,
    size: usize,
    index: usize,
    last: usize,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    errors: Vec<Failure>,
}
fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/rl_candle_replay.json")).unwrap()
}

fn observation(records: &IndexMap<String, Record>) -> RecurrentObservation {
    RecurrentObservation {
        data_processed: records["data_processed"].tensor(),
        cur_tick: records["cur_tick"].tensor(),
        cur_step: records["cur_step"].tensor(),
        position_history: records["position_history"].tensor(),
        target: records["target"].tensor(),
        num_step: records["num_step"].tensor(),
        acquiring: records["acquiring"].tensor(),
    }
}
fn transition(step: &Step) -> CandleReplayTransition {
    CandleReplayTransition {
        observation: observation(&step.observation),
        next_observation: observation(&step.next_observation),
        action: step.action.tensor(),
        reward: step.reward,
        terminated: step.terminated,
        truncated: step.truncated,
    }
}
fn check_observation(expected: &IndexMap<String, Record>, actual: &RecurrentObservation) {
    for (name, tensor) in [
        "data_processed",
        "cur_tick",
        "cur_step",
        "position_history",
        "target",
        "num_step",
        "acquiring",
    ]
    .into_iter()
    .zip(fields(actual))
    {
        expected[name].compare(tensor, name);
    }
}

#[test]
fn physical_slots_owned_rows_and_source_dtypes_match_real_torch_replay() {
    let fixture = fixture();
    assert_eq!(fixture.cases.len(), 3);
    for case in fixture.cases {
        let mut buffer = CandleReplayBuffer::new(case.capacity);
        let mut rng = StdRng::seed_from_u64(31);
        for step in case.steps {
            let entry = transition(&step);
            let added = buffer.add(&entry).unwrap();
            assert_eq!(added.index, step.added[0].to_usize().unwrap());
            assert!((added.reward - step.added[1]).abs() < 1e-14);
            assert_eq!(added.length, step.added[2].to_usize().unwrap());
            assert_eq!(added.start, step.added[3].to_usize().unwrap());
            let stored = buffer.get(&(0..case.capacity).collect::<Vec<_>>()).unwrap();
            check_observation(&step.stored_observation, &stored.observations);
            check_observation(&step.stored_next, &stored.next_observations);
            step.stored_actions.compare(&stored.actions, "actions");
            assert_eq!(stored.rewards.to_vec(), step.stored_rewards);
            assert_eq!(stored.terminated.to_vec(), step.stored_terminated);
            assert_eq!(stored.truncated.to_vec(), step.stored_truncated);
            assert_eq!(
                stored.bootstrap_valid.to_vec(),
                step.stored_terminated
                    .iter()
                    .map(|value| !value)
                    .collect::<Vec<_>>()
            );
            let sampled = buffer.sample_batch(0, &mut rng).unwrap();
            assert_eq!(sampled.indices, step.indices);
            assert_eq!(sampled.unfinished_indices, step.unfinished);
            assert!(sampled.replay_weights.is_none());
        }
    }
}

#[test]
fn tensor_dtype_failures_retain_source_index_advancement() {
    let fixture = fixture();
    assert_eq!(fixture.errors.len(), 2);
    for failure in fixture.errors {
        let mut buffer = CandleReplayBuffer::new(3);
        let mut entry = transition(&fixture.cases[1].steps[0]);
        buffer.add(&entry).unwrap();
        if failure.field == "observation" {
            entry.observation.data_processed = entry
                .observation
                .data_processed
                .to_dtype(DType::F64)
                .unwrap();
        } else {
            entry.action = entry.action.to_dtype(DType::F32).unwrap();
        }
        assert!(matches!(
            buffer.add(&entry),
            Err(CandleReplayError::Tensor(_))
        ));
        assert_eq!(buffer.index().len(), failure.size);
        assert_eq!(buffer.index().next_write_index(), failure.index);
        assert_eq!(buffer.index().last_index(), failure.last);
    }
}

#[test]
fn snapshots_empty_reads_reset_and_native_failures_are_explicit() {
    let mut entry = transition(&fixture().cases[1].steps[0]);
    let variable = Var::from_tensor(&entry.observation.data_processed).unwrap();
    entry.observation.data_processed = variable.as_tensor().clone();
    let original = variable.to_vec3::<f32>().unwrap();
    let mut buffer = CandleReplayBuffer::new(3);
    let mut rng = StdRng::seed_from_u64(9);
    assert!(matches!(
        buffer.sample_batch(0, &mut rng),
        Err(CandleReplayError::MissingStorage)
    ));
    assert!(matches!(
        buffer.sample_batch(1, &mut rng),
        Err(CandleReplayError::Index(ReplayIndexError::EmptyPopulation))
    ));
    assert_eq!(
        buffer.sample(0, &mut rng).err().unwrap(),
        CandleReplayError::MissingStorage.to_string()
    );
    assert_eq!(
        buffer.sample(1, &mut rng).err().unwrap(),
        ReplayIndexError::EmptyPopulation.to_string()
    );
    buffer.add(&entry).unwrap();
    variable.set(&variable.ones_like().unwrap()).unwrap();
    assert_eq!(
        buffer
            .get(&[0])
            .unwrap()
            .observations
            .data_processed
            .to_vec3::<f32>()
            .unwrap(),
        original
    );
    assert_eq!(buffer.get(&[0, 0]).unwrap().actions.dims(), &[2]);
    assert!(matches!(
        buffer.get(&[3]),
        Err(CandleReplayError::Index(ReplayIndexError::Index))
    ));
    assert!(
        buffer
            .sample(u64::MAX, &mut rng)
            .err()
            .unwrap()
            .contains("signed index")
    );
    let draw = buffer.sample(10, &mut rng).unwrap();
    assert_eq!(draw.indices, vec![0; 10]);
    buffer.reset(true);
    let empty = buffer.sample_batch(0, &mut rng).unwrap();
    assert_eq!(empty.observations.data_processed.dims(), &[0, 2, 2]);
    assert_eq!(empty.actions.dims(), &[0]);
    assert!(empty.unfinished_indices.is_empty());
    assert_eq!(
        buffer
            .get(&[0])
            .unwrap()
            .observations
            .data_processed
            .to_vec3::<f32>()
            .unwrap(),
        original
    );
    entry.action = Tensor::new(0_i64, &Device::Cpu).unwrap();
    assert!(matches!(
        buffer.add(&entry),
        Err(CandleReplayError::BatchSize)
    ));
    entry.action = Tensor::new(&[0_i64], &Device::Cpu).unwrap();
    let mut zero = CandleReplayBuffer::new(0);
    assert!(matches!(
        zero.add(&entry),
        Err(CandleReplayError::Index(ReplayIndexError::ZeroCapacity))
    ));
    entry.observation.target = Tensor::ones((1, 2), DType::F32, &Device::Cpu).unwrap();
    assert!(matches!(
        buffer.add(&entry),
        Err(CandleReplayError::Tensor(_))
    ));
}

#[test]
fn each_observation_field_failure_preserves_exactly_the_preceding_native_writes() {
    let fixture = fixture();
    let original = transition(&fixture.cases[1].steps[1]);
    let values = |tensor: &Tensor| {
        tensor
            .to_dtype(DType::F64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()
    };
    for next in [false, true] {
        for failed_field in 0..7 {
            let mut buffer = CandleReplayBuffer::new(3);
            buffer.add(&transition(&fixture.cases[1].steps[0])).unwrap();
            let mut entry = original.clone();
            let obs = if next {
                &mut entry.next_observation
            } else {
                &mut entry.observation
            };
            let field = [
                &mut obs.data_processed,
                &mut obs.cur_tick,
                &mut obs.cur_step,
                &mut obs.position_history,
                &mut obs.target,
                &mut obs.num_step,
                &mut obs.acquiring,
            ]
            .into_iter()
            .nth(failed_field)
            .unwrap();
            *field = field.to_dtype(DType::F64).unwrap();
            assert!(matches!(
                buffer.add(&entry),
                Err(CandleReplayError::Tensor(_))
            ));
            assert_eq!(buffer.index().len(), 2);
            assert_eq!(buffer.index().next_write_index(), 2);
            let stored = buffer.get(&[1]).unwrap();
            for (is_next, actual, expected) in [
                (false, &stored.observations, &original.observation),
                (true, &stored.next_observations, &original.next_observation),
            ] {
                for (index, (actual, expected)) in
                    fields(actual).into_iter().zip(fields(expected)).enumerate()
                {
                    let assigned = if is_next == next {
                        index < failed_field
                    } else {
                        next
                    };
                    let expected = if assigned {
                        values(expected)
                    } else {
                        vec![0.; expected.elem_count()]
                    };
                    assert_eq!(values(actual), expected);
                }
            }
            assert_eq!(stored.actions.to_vec1::<i64>().unwrap(), [0]);
            assert_eq!(stored.rewards.to_vec(), [0.]);
            assert_eq!(stored.terminated.to_vec(), [false]);
            assert_eq!(stored.truncated.to_vec(), [false]);
        }
    }
}
