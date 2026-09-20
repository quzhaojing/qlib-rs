use super::*;
use crate::rl_candle_network::RecurrentObservation;
use crate::rl_candle_network_fixture::Record;
use crate::rl_candle_replay::{
    CandleReplayBatch, CandleReplayBuffer, CandleReplayTransition, CandleVectorReplayBuffer,
};
use candle_core::Var;
use ndarray::array;
use num_traits::ToPrimitive;
use serde::Deserialize;
use std::cell::RefCell;

#[derive(Deserialize)]
struct Addition {
    id: usize,
    reward: f64,
    terminated: bool,
    truncated: bool,
}
#[derive(Deserialize)]
struct Case {
    steps: usize,
    gamma: f64,
    terminal: Vec<usize>,
    target: Record,
    returns: Record,
    weight: Record,
}
#[derive(Deserialize)]
struct BufferCase {
    count: usize,
    capacity: usize,
    additions: Vec<Addition>,
    indices: Vec<usize>,
    rewards: Vec<f64>,
    done: Vec<bool>,
    bootstrap: Vec<bool>,
    unfinished: Vec<usize>,
    cases: Vec<Case>,
}

fn observation() -> RecurrentObservation {
    let floats = Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap();
    let integers = Tensor::zeros(1, DType::I64, &Device::Cpu).unwrap();
    RecurrentObservation {
        data_processed: floats.reshape((1, 1, 1)).unwrap(),
        cur_tick: integers.clone(),
        cur_step: integers.clone(),
        position_history: floats.reshape((1, 1)).unwrap(),
        target: floats,
        num_step: integers.clone(),
        acquiring: integers,
    }
}
fn buffer(case: &BufferCase) -> Box<dyn CandleNStepReplay> {
    let mut single = CandleReplayBuffer::new(case.capacity);
    let mut vector = CandleVectorReplayBuffer::new(case.capacity, case.count).unwrap();
    for value in &case.additions {
        let row = CandleReplayTransition {
            observation: observation(),
            next_observation: observation(),
            action: Tensor::zeros(1, DType::I64, &Device::Cpu).unwrap(),
            reward: value.reward,
            terminated: value.terminated,
            truncated: value.truncated,
        };
        if case.count == 1 {
            single.add(&row).unwrap();
        } else {
            vector
                .add(
                    &CandleReplayBatch {
                        observations: &row.observation,
                        next_observations: &row.next_observation,
                        actions: &row.action,
                        rewards: &[row.reward],
                        terminated: &[row.terminated],
                        truncated: &[row.truncated],
                    },
                    Some(&[value.id]),
                )
                .unwrap();
        }
    }
    if case.count == 1 {
        Box::new(single)
    } else {
        Box::new(vector)
    }
}

#[test]
fn actual_source_single_vector_wraps_dtypes_shapes_and_horizons_match() {
    #[derive(Deserialize)]
    struct Fixture {
        buffers: Vec<BufferCase>,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/rl_candle_nstep.json")).unwrap();
    assert_eq!(fixture.buffers.len(), 2);
    for case in fixture.buffers {
        let buffer = buffer(&case);
        assert_eq!(buffer.rewards().unwrap().to_vec(), case.rewards);
        assert_eq!(buffer.end_flags().unwrap().to_vec(), case.done);
        assert_eq!(
            buffer
                .bootstrap_mask(&(0..case.rewards.len()).collect::<Vec<_>>())
                .unwrap()
                .to_vec(),
            case.bootstrap
        );
        assert_eq!(buffer.unfinished().unwrap(), case.unfinished);
        assert!(buffer.bootstrap_mask(&[case.rewards.len()]).is_err());
        assert_eq!(case.cases.len(), 54);
        for expected in case.cases {
            let mut batch = CandleNStepBatch {
                returns: None,
                weight: Some(
                    Tensor::new(
                        (0..case.indices.len())
                            .map(|i| i.to_f64().unwrap() + 0.25)
                            .collect::<Vec<_>>(),
                        &Device::Cpu,
                    )
                    .unwrap(),
                ),
            };
            prepare_nstep_returns(
                &mut batch,
                buffer.as_ref(),
                &case.indices,
                |_, terminal| {
                    assert_eq!(terminal, expected.terminal);
                    Ok(expected.target.tensor())
                },
                CandleNStepConfig {
                    gamma: expected.gamma,
                    steps: expected.steps,
                    reward_normalization: false,
                },
            )
            .unwrap();
            expected
                .returns
                .compare(batch.returns.as_ref().unwrap(), "returns");
            expected
                .weight
                .compare(batch.weight.as_ref().unwrap(), "weight");
        }
    }
}

struct Probe {
    events: RefCell<Vec<&'static str>>,
    fail: Option<&'static str>,
    malformed: Option<&'static str>,
}
impl Probe {
    fn new() -> Self {
        Self {
            events: RefCell::default(),
            fail: None,
            malformed: None,
        }
    }
    fn enter(&self, stage: &'static str) -> Result<(), String> {
        self.events.borrow_mut().push(stage);
        if self.fail == Some(stage) {
            Err(stage.into())
        } else {
            Ok(())
        }
    }
}
impl CandleNStepReplay for Probe {
    fn rewards(&self) -> Result<Array1<f64>, String> {
        self.enter("rewards")?;
        Ok(array![1., 2.])
    }
    fn next_indices(&self, indices: &[usize]) -> Result<Vec<usize>, String> {
        self.enter("next")?;
        Ok(match self.malformed {
            Some("next_length") => vec![0],
            Some("next_index") => vec![4, 4],
            _ => indices.to_vec(),
        })
    }
    fn bootstrap_mask(&self, _: &[usize]) -> Result<Array1<bool>, String> {
        self.enter("bootstrap")?;
        Ok(if self.malformed == Some("mask_length") {
            array![true]
        } else {
            array![true, false]
        })
    }
    fn end_flags(&self) -> Result<Array1<bool>, String> {
        self.enter("end flags")?;
        Ok(if self.malformed == Some("done_length") {
            array![true]
        } else {
            array![false, true]
        })
    }
    fn unfinished(&self) -> Result<Vec<usize>, String> {
        self.enter("unfinished")?;
        Ok(if self.malformed == Some("unfinished") {
            vec![99]
        } else {
            vec![0]
        })
    }
}
fn config() -> CandleNStepConfig {
    CandleNStepConfig {
        steps: 2,
        ..Default::default()
    }
}
fn target(probe: &Probe, _: &[usize]) -> Result<Tensor, String> {
    probe.enter("target Q")?;
    Ok(Tensor::new(&[3_f32, 4.], &Device::Cpu).unwrap())
}

#[test]
fn adapter_failures_stop_in_source_order_and_preserve_existing_batch_fields() {
    let stages = [
        "rewards",
        "next",
        "target Q",
        "bootstrap",
        "end flags",
        "unfinished",
    ];
    for (index, stage) in stages.iter().enumerate() {
        let mut probe = Probe::new();
        probe.fail = Some(stage);
        let marker = Tensor::new(&[9_f32], &Device::Cpu).unwrap();
        let mut batch = CandleNStepBatch {
            returns: Some(marker.clone()),
            weight: Some(marker.clone()),
        };
        let error =
            prepare_nstep_returns(&mut batch, &probe, &[0, 1], target, config()).unwrap_err();
        assert!(
            matches!(error, CandleNStepError::Adapter { stage: actual, .. } if actual == *stage)
        );
        assert_eq!(*probe.events.borrow(), stages[..=index]);
        assert_eq!(batch.returns.unwrap().id(), marker.id());
        assert_eq!(batch.weight.unwrap().id(), marker.id());
    }
}

#[test]
fn malformed_metadata_and_source_dtype_boundaries_are_explicit() {
    for malformed in [
        "mask_length",
        "done_length",
        "unfinished",
        "next_length",
        "next_index",
    ] {
        let mut probe = Probe::new();
        probe.malformed = Some(malformed);
        let mut batch = CandleNStepBatch::default();
        assert!(matches!(
            prepare_nstep_returns(&mut batch, &probe, &[0, 1], target, config()),
            Err(CandleNStepError::Metadata)
        ));
        assert!(batch.returns.is_none());
    }
    for dtype in [DType::F16, DType::BF16] {
        let probe = Probe::new();
        let result = prepare_nstep_returns(
            &mut CandleNStepBatch::default(),
            &probe,
            &[0, 1],
            |probe, ids| target(probe, ids).map(|t| t.to_dtype(dtype).unwrap()),
            config(),
        );
        assert!(matches!(result, Err(CandleNStepError::Dtype(actual)) if actual == dtype));
        assert_eq!(
            probe.events.borrow().len(),
            if dtype == DType::BF16 { 3 } else { 6 }
        );
    }
    let probe = Probe::new();
    assert!(matches!(
        prepare_nstep_returns(
            &mut CandleNStepBatch::default(),
            &probe,
            &[],
            target,
            config()
        ),
        Err(CandleNStepError::Empty)
    ));
    assert_eq!(*probe.events.borrow(), ["rewards", "next", "target Q"]);
    let probe = Probe::new();
    assert!(matches!(
        prepare_nstep_returns(
            &mut CandleNStepBatch::default(),
            &probe,
            &[0, 1],
            |_, _| Ok(Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap()),
            config()
        ),
        Err(CandleNStepError::Tensor(_))
    ));
    for invalid in [
        CandleNStepConfig {
            steps: 0,
            ..config()
        },
        CandleNStepConfig {
            reward_normalization: true,
            ..config()
        },
    ] {
        let probe = Probe::new();
        assert!(
            prepare_nstep_returns(
                &mut CandleNStepBatch::default(),
                &probe,
                &[0],
                target,
                invalid
            )
            .is_err()
        );
        assert!(probe.events.borrow().is_empty());
    }
}

#[test]
fn direct_recurrence_preserves_nonfinite_masking_and_target_is_detached() {
    assert!(matches!(
        nstep_returns(
            &array![1.].view(),
            &array![true].view(),
            &array![[2.]].view(),
            &[],
            1.
        ),
        Err(CandleNStepError::Horizon)
    ));
    let returns = nstep_returns(
        &array![1.].view(),
        &array![true].view(),
        &array![[f64::NAN]].view(),
        &[vec![0]],
        0.,
    )
    .unwrap();
    assert!(returns[(0, 0)].is_nan());
    let probe = Probe::new();
    let variable = Var::new(&[3_f64, 4.], &Device::Cpu).unwrap();
    let mut batch = CandleNStepBatch::default();
    prepare_nstep_returns(
        &mut batch,
        &probe,
        &[0, 1],
        |_, _| Ok(variable.as_tensor().clone()),
        config(),
    )
    .unwrap();
    let result = batch.returns.unwrap();
    assert_eq!(result.dims(), [2, 1]);
    assert!(!result.is_variable());
    assert!(
        result
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .get(&variable)
            .is_none()
    );
    for buffer in [
        Box::new(CandleReplayBuffer::new(2)) as Box<dyn CandleNStepReplay>,
        Box::new(CandleVectorReplayBuffer::new(2, 2).unwrap()),
    ] {
        assert!(buffer.rewards().is_err());
        assert!(buffer.end_flags().is_err());
        assert!(buffer.bootstrap_mask(&[0]).is_err());
        assert!(buffer.next_indices(&[0]).is_err());
        assert!(buffer.unfinished().is_err());
    }
}
