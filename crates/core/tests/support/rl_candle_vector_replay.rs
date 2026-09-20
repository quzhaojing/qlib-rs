use super::*;
use crate::rl_candle_network_fixture::Record;
use candle_core::{DType, Device, Var};
use indexmap::IndexMap;
use num_traits::ToPrimitive;
use rand::{RngCore, SeedableRng, rngs::StdRng};
use serde::Deserialize;

#[derive(Deserialize)]
struct Data {
    observation: IndexMap<String, Record>,
    next_observation: IndexMap<String, Record>,
    actions: Record,
    rewards: Vec<f64>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}
#[derive(Deserialize)]
struct State {
    lengths: Vec<usize>,
    last: Vec<usize>,
    next_write: Vec<usize>,
    unfinished: Option<Vec<usize>>,
    indices: Vec<usize>,
}
#[derive(Deserialize)]
struct Step {
    ids: Vec<usize>,
    reset: Option<bool>,
    input: Data,
    added: Vec<Vec<f64>>,
    state: State,
    stored: Data,
    query: Vec<usize>,
    previous: Vec<usize>,
    following: Vec<usize>,
}
#[derive(Deserialize)]
struct Case {
    total: usize,
    environments: usize,
    capacity: usize,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Failure {
    field: String,
    state: State,
}
#[derive(Deserialize)]
struct Sampling {
    calls: Vec<Call>,
    draws: Vec<usize>,
}
#[derive(Deserialize)]
struct Call {
    population: usize,
    size: usize,
    probabilities: Vec<f64>,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    failures: Vec<Failure>,
    sampling: Sampling,
}
fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/rl_candle_vector_replay.json")).unwrap()
}
fn observation(data: &IndexMap<String, Record>) -> RecurrentObservation {
    RecurrentObservation {
        data_processed: data["data_processed"].tensor(),
        cur_tick: data["cur_tick"].tensor(),
        cur_step: data["cur_step"].tensor(),
        position_history: data["position_history"].tensor(),
        target: data["target"].tensor(),
        num_step: data["num_step"].tensor(),
        acquiring: data["acquiring"].tensor(),
    }
}
fn compare_observation(expected: &IndexMap<String, Record>, actual: &RecurrentObservation) {
    for (key, tensor) in [
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
        expected[key].compare(tensor, key);
    }
}
impl Data {
    fn add(
        &self,
        buffer: &mut CandleVectorReplayBuffer,
        ids: Option<&[usize]>,
    ) -> Vec<ReplayEpisode> {
        buffer
            .add(
                &CandleReplayBatch {
                    observations: &observation(&self.observation),
                    next_observations: &observation(&self.next_observation),
                    actions: &self.actions.tensor(),
                    rewards: &self.rewards,
                    terminated: &self.terminated,
                    truncated: &self.truncated,
                },
                ids,
            )
            .unwrap()
    }
    fn compare(&self, actual: &CandlePpoRollout) {
        compare_observation(&self.observation, &actual.observations);
        compare_observation(&self.next_observation, &actual.next_observations);
        self.actions.compare(&actual.actions, "actions");
        assert_eq!(actual.rewards.to_vec(), self.rewards);
        assert_eq!(actual.terminated.to_vec(), self.terminated);
        assert_eq!(actual.truncated.to_vec(), self.truncated);
        assert_eq!(
            actual.bootstrap_valid.to_vec(),
            self.terminated
                .iter()
                .map(|value| !value)
                .collect::<Vec<_>>()
        );
    }
}
impl State {
    fn compare(&self, buffer: &CandleVectorReplayBuffer) {
        assert_eq!(
            buffer
                .children()
                .iter()
                .map(ReplayIndex::len)
                .collect::<Vec<_>>(),
            self.lengths
        );
        assert_eq!(buffer.len(), self.lengths.iter().sum::<usize>());
        assert_eq!(
            buffer
                .children()
                .iter()
                .enumerate()
                .map(|(id, child)| child.last_index() + id * buffer.child_capacity)
                .collect::<Vec<_>>(),
            self.last
        );
        assert_eq!(
            buffer
                .children()
                .iter()
                .map(ReplayIndex::next_write_index)
                .collect::<Vec<_>>(),
            self.next_write
        );
        match &self.unfinished {
            Some(expected) => assert_eq!(&buffer.unfinished_indices().unwrap(), expected),
            None => assert_eq!(
                buffer.unfinished_indices(),
                Err(ReplayIndexError::MissingDone)
            ),
        }
        let mut rng = StdRng::seed_from_u64(9);
        let mut untouched = rng.clone();
        assert_eq!(buffer.sample_indices(0, &mut rng).unwrap(), self.indices);
        assert_eq!(rng.next_u64(), untouched.next_u64());
    }
}

#[test]
fn shared_schema_ring_overwrite_and_reset_match_actual_vector_replay() {
    let fixture = fixture();
    assert_eq!(fixture.cases.len(), 3);
    for case in fixture.cases {
        let mut buffer = CandleVectorReplayBuffer::new(case.total, case.environments).unwrap();
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), case.capacity);
        for step in case.steps {
            if let Some(keep) = step.reset {
                buffer.reset(keep);
            }
            let added = step.input.add(&mut buffer, Some(&step.ids));
            for (row, episode) in added.iter().enumerate() {
                assert_eq!(episode.index, step.added[0][row].to_usize().unwrap());
                assert!((episode.reward - step.added[1][row]).abs() < 1e-13);
                assert_eq!(episode.length, step.added[2][row].to_usize().unwrap());
                assert_eq!(episode.start, step.added[3][row].to_usize().unwrap());
            }
            step.state.compare(&buffer);
            step.stored
                .compare(&buffer.get(&(0..case.capacity).collect::<Vec<_>>()).unwrap());
            assert_eq!(buffer.previous(&step.query).unwrap(), step.previous);
            assert_eq!(buffer.next(&step.query).unwrap(), step.following);
            let sampled = buffer.sample(0, &mut StdRng::seed_from_u64(2)).unwrap();
            assert_eq!(sampled.indices, step.state.indices);
        }
    }
}

#[test]
fn source_failure_boundaries_preserve_all_reached_child_states() {
    let fixture = fixture();
    assert_eq!(fixture.failures.len(), 4);
    for failure in fixture.failures {
        let mut buffer = CandleVectorReplayBuffer::new(5, 2).unwrap();
        if failure.field == "observation" || failure.field == "action" {
            fixture.cases[0].steps[0].input.add(&mut buffer, Some(&[0]));
        }
        let input = &fixture.cases[0].steps[1].input;
        let mut obs = observation(&input.observation);
        let mut next = observation(&input.next_observation);
        let mut actions = input.actions.tensor();
        let ids: &[usize] = match failure.field.as_str() {
            "observation" => {
                obs.data_processed = obs.data_processed.to_dtype(DType::F64).unwrap();
                &[0, 1]
            }
            "action" => {
                actions = actions.to_dtype(DType::F32).unwrap();
                &[0, 1]
            }
            "invalid_id" => &[0, 9],
            "empty" => {
                obs = super::super::map_fields(&obs, |value| value.narrow(0, 0, 0)).unwrap();
                next = super::super::map_fields(&next, |value| value.narrow(0, 0, 0)).unwrap();
                actions = actions.narrow(0, 0, 0).unwrap();
                &[]
            }
            other => panic!("unknown fixture {other}"),
        };
        let length = ids.len();
        let result = buffer.add(
            &CandleReplayBatch {
                observations: &obs,
                next_observations: &next,
                actions: &actions,
                rewards: &vec![1.; length],
                terminated: &vec![false; length],
                truncated: &vec![false; length],
            },
            Some(ids),
        );
        match failure.field.as_str() {
            "observation" | "action" => {
                assert!(matches!(result, Err(CandleVectorReplayError::Tensor(_))));
            }
            "invalid_id" => assert!(matches!(
                result,
                Err(CandleVectorReplayError::EnvironmentIndex)
            )),
            _ => assert!(matches!(result, Err(CandleVectorReplayError::EmptyIndices))),
        }
        failure.state.compare(&buffer);
    }
}

#[test]
fn two_stage_replacement_sampling_is_grouped_and_uses_the_caller_rng() {
    let fixture = fixture();
    let mut buffer = CandleVectorReplayBuffer::new(9, 3).unwrap();
    let input = &fixture.cases[0].steps[0].input;
    input.add(&mut buffer, Some(&[0]));
    input.add(&mut buffer, Some(&[0]));
    input.add(&mut buffer, Some(&[2]));
    let mut rng = StdRng::seed_from_u64(41);
    let mut expected_rng = rng.clone();
    let selected = buffer.sample_indices(30_000, &mut rng).unwrap();
    let distribution = WeightedIndex::new([2_usize, 0, 1]).unwrap();
    let mut counts = [0_i64; 3];
    for _ in 0..30_000 {
        counts[distribution.sample(&mut expected_rng)] += 1;
    }
    let mut expected = Vec::new();
    for (id, child) in buffer.children().iter().enumerate() {
        let count = if counts[id] == 0 { -1 } else { counts[id] };
        expected.extend(
            child
                .sample_indices(count, 1, false, &mut expected_rng)
                .unwrap()
                .into_iter()
                .map(|index| index + id * 3),
        );
    }
    assert_eq!(selected, expected);
    assert_eq!(rng.next_u64(), expected_rng.next_u64());
    assert!(selected.windows(2).all(|pair| pair[0] / 3 <= pair[1] / 3));
    let mut frequency = [0_i32; 9];
    for index in selected {
        frequency[index] += 1;
    }
    for index in [0, 1, 6] {
        assert!((frequency[index] - 10_000).abs() < 490);
    }
    assert!(buffer.sample_indices(-1, &mut rng).unwrap().is_empty());
    assert_eq!(rng.next_u64(), expected_rng.next_u64());
    let source = fixture.sampling;
    assert_eq!(source.calls[0].population, 3);
    assert_eq!(source.calls[0].size, 40);
    assert_eq!(source.calls[0].probabilities, [2. / 3., 0., 1. / 3.]);
    assert_eq!(source.calls[1].population, 2);
    assert_eq!(source.calls[2].population, 1);
    assert_eq!(source.calls[1].size + source.calls[2].size, 40);
    assert!(
        source.draws[..source.calls[1].size]
            .iter()
            .all(|&value| value < 2)
    );
    assert!(
        source.draws[source.calls[1].size..]
            .iter()
            .all(|&value| value == 6)
    );
}

#[test]
fn construction_default_ids_empty_reads_and_native_limits_are_explicit() {
    assert!(matches!(
        CandleVectorReplayBuffer::new(4, 0),
        Err(CandleVectorReplayError::EnvironmentCount)
    ));
    assert!(matches!(
        CandleVectorReplayBuffer::new(usize::MAX, 2),
        Err(CandleVectorReplayError::CapacityOverflow)
    ));
    let mut buffer = CandleVectorReplayBuffer::new(5, 2).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    assert_eq!(buffer.previous(&[]), Err(ReplayIndexError::MissingDone));
    assert_eq!(buffer.next(&[]), Err(ReplayIndexError::MissingDone));
    assert_eq!(
        buffer.sample_indices(1, &mut rng),
        Err(ReplayIndexError::EmptyPopulation)
    );
    assert!(
        buffer
            .sample(0, &mut rng)
            .err()
            .unwrap()
            .contains("not initialized")
    );
    assert_eq!(
        buffer.sample(1, &mut rng).err().unwrap(),
        ReplayIndexError::EmptyPopulation.to_string()
    );
    assert!(
        buffer
            .sample(u64::MAX, &mut rng)
            .err()
            .unwrap()
            .contains("signed index")
    );
    let fixture = fixture();
    fixture.cases[0].steps[1].input.add(&mut buffer, None);
    assert_eq!(
        buffer
            .children()
            .iter()
            .map(ReplayIndex::len)
            .collect::<Vec<_>>(),
        [1, 1]
    );
    assert_eq!(buffer.get(&[3, 0, 3]).unwrap().actions.dims(), [3]);
    assert!(buffer.get(&[6]).is_err());
    buffer.reset(false);
    let empty = buffer.sample_batch(0, &mut rng).unwrap();
    assert_eq!(empty.observations.data_processed.dims(), [0, 2, 2]);
    assert!(empty.unfinished_indices.is_empty());
    let mut zero = CandleVectorReplayBuffer::new(0, 2).unwrap();
    let input = &fixture.cases[0].steps[0].input;
    let obs = observation(&input.observation);
    let next = observation(&input.next_observation);
    let action = Tensor::new(&[0_i64], &Device::Cpu).unwrap();
    let mut batch = CandleReplayBatch {
        observations: &obs,
        next_observations: &next,
        actions: &action,
        rewards: &[1.],
        terminated: &[false],
        truncated: &[false],
    };
    assert!(matches!(
        zero.add(&batch, Some(&[0])),
        Err(CandleVectorReplayError::Index(
            ReplayIndexError::ZeroCapacity
        ))
    ));
    assert!(matches!(
        buffer.add(&batch, Some(&[0, 1])),
        Err(CandleVectorReplayError::BatchSize)
    ));
    assert_eq!(buffer.children()[0].len(), 1);
    batch.rewards = &[1., 2.];
    batch.terminated = &[false, false];
    batch.truncated = &[false, false];
    assert!(matches!(
        buffer.add(&batch, Some(&[0])),
        Err(CandleVectorReplayError::BatchSize)
    ));
    assert_eq!(buffer.children()[0].len(), 2);
    assert!(matches!(
        zero.add(&batch, Some(&[])),
        Err(CandleVectorReplayError::EmptyIndices)
    ));
    assert_eq!(zero.next(&[0]), Err(ReplayIndexError::ZeroCapacity));
    assert_eq!(zero.previous(&[]).unwrap(), Vec::<usize>::new());
}

#[test]
fn broadcasted_fields_are_owned_and_fail_atomically_across_selected_rows() {
    let fixture = fixture();
    let input = &fixture.cases[0].steps[0].input;
    let mut obs = observation(&input.observation);
    let next = observation(&input.next_observation);
    let original = obs.data_processed.to_vec3::<f32>().unwrap();
    let variable = Var::from_tensor(&obs.data_processed).unwrap();
    obs.data_processed = variable.as_tensor().clone();
    let action = Tensor::new(2_i64, &Device::Cpu).unwrap();
    let mut buffer = CandleVectorReplayBuffer::new(2, 2).unwrap();
    let batch = CandleReplayBatch {
        observations: &obs,
        next_observations: &next,
        actions: &action,
        rewards: &[1., 2.],
        terminated: &[false, true],
        truncated: &[false, false],
    };
    buffer.add(&batch, None).unwrap();
    variable.set(&variable.ones_like().unwrap()).unwrap();
    let stored = buffer.get(&[0, 1]).unwrap();
    assert_eq!(
        stored.observations.data_processed.to_vec3::<f32>().unwrap(),
        vec![original[0].clone(), original[0].clone()]
    );
    assert_eq!(stored.actions.to_vec1::<i64>().unwrap(), [2, 2]);
    assert_eq!(stored.rewards.to_vec(), [1., 2.]);
    let mut malformed = obs.clone();
    malformed.cur_tick = Tensor::new(&[[7_i64, 8], [9, 10]], &Device::Cpu).unwrap();
    let batch = CandleReplayBatch {
        observations: &malformed,
        ..batch
    };
    assert!(matches!(
        buffer.add(&batch, None),
        Err(CandleVectorReplayError::Tensor(_))
    ));
    let after = buffer.get(&[0, 1]).unwrap();
    assert_eq!(
        after.observations.data_processed.to_vec3::<f32>().unwrap(),
        vec![vec![vec![1.; 2]; 2]; 2]
    );
    assert_eq!(
        after.observations.cur_tick.to_vec1::<i64>().unwrap(),
        stored.observations.cur_tick.to_vec1::<i64>().unwrap()
    );
    assert_eq!(after.terminated, stored.terminated);
    assert_eq!(after.rewards, stored.rewards);
}

#[test]
fn malformed_flag_rows_fail_after_prior_child_advancement() {
    let fixture = fixture();
    let input = &fixture.cases[0].steps[0].input;
    let obs = observation(&input.observation);
    let next = observation(&input.next_observation);
    let action = input.actions.tensor();
    for missing_terminated in [false, true] {
        let mut buffer = CandleVectorReplayBuffer::new(5, 2).unwrap();
        let batch = CandleReplayBatch {
            observations: &obs,
            next_observations: &next,
            actions: &action,
            rewards: &[1., 2.],
            terminated: if missing_terminated {
                &[false]
            } else {
                &[false, false]
            },
            truncated: if missing_terminated {
                &[false, false]
            } else {
                &[false]
            },
        };
        assert!(matches!(
            buffer.add(&batch, None),
            Err(CandleVectorReplayError::BatchSize)
        ));
        assert_eq!(buffer.children()[0].len(), 1);
        assert_eq!(buffer.children()[1].len(), 0);
        assert_eq!(
            buffer.unfinished_indices(),
            Err(ReplayIndexError::MissingDone)
        );
    }
}

#[test]
fn surplus_flags_fail_after_tensor_writes_without_committing_scalars() {
    let fixture = fixture();
    let input = &fixture.cases[0].steps[0].input;
    let obs = observation(&input.observation);
    let next = observation(&input.next_observation);
    let action = Tensor::new(&[2_i64], &Device::Cpu).unwrap();
    for surplus_terminated in [false, true] {
        let mut buffer = CandleVectorReplayBuffer::new(3, 2).unwrap();
        let batch = CandleReplayBatch {
            observations: &obs,
            next_observations: &next,
            actions: &action,
            rewards: &[7.],
            terminated: if surplus_terminated {
                &[false, true]
            } else {
                &[false]
            },
            truncated: if surplus_terminated {
                &[false]
            } else {
                &[false, true]
            },
        };
        assert!(matches!(
            buffer.add(&batch, Some(&[0])),
            Err(CandleVectorReplayError::BatchSize)
        ));
        assert_eq!(buffer.children()[0].len(), 1);
        let stored = buffer.get(&[0]).unwrap();
        compare_observation(&input.observation, &stored.observations);
        assert_eq!(stored.actions.to_vec1::<i64>().unwrap(), [2]);
        assert_eq!(stored.rewards.to_vec(), [0.]);
        assert_eq!(stored.terminated.to_vec(), [false]);
        assert_eq!(stored.truncated.to_vec(), [false]);
        assert_eq!(stored.unfinished_indices, [0]);
    }
}
