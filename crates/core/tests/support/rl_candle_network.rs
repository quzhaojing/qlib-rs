use super::*;
use crate::rl_candle_heads::{PpoActor, PpoCritic};
use std::sync::Arc;

use crate::rl_candle_network_fixture as fixture;

fn observation(inputs: &IndexMap<String, Tensor>) -> RecurrentObservation {
    RecurrentObservation {
        data_processed: inputs["data_processed"].clone(),
        cur_tick: inputs["cur_tick"].clone(),
        cur_step: inputs["cur_step"].clone(),
        position_history: inputs["position_history"].clone(),
        target: inputs["target"].clone(),
        num_step: inputs["num_step"].clone(),
        acquiring: inputs["acquiring"].clone(),
    }
}

fn fields(obs: &RecurrentObservation) -> [&Tensor; 7] {
    [
        &obs.data_processed,
        &obs.cur_tick,
        &obs.cur_step,
        &obs.position_history,
        &obs.target,
        &obs.num_step,
        &obs.acquiring,
    ]
}

#[test]
fn observation_minibatches_keep_field_alignment_duplicates_gradients_and_errors() {
    use candle_core::{Device, Var};
    let case = fixture::cases("recurrent").remove(0);
    let mut obs = observation(&case.inputs());
    let variable = Var::from_tensor(&obs.data_processed).unwrap();
    obs.data_processed = variable.as_tensor().clone();
    let indices = Tensor::new(&[1_i64, 0, 1], &Device::Cpu).unwrap();
    let selected = obs.select_batch(&indices).unwrap();
    for (source, actual) in fields(&obs).into_iter().zip(fields(&selected)) {
        assert_eq!(source.dtype(), actual.dtype());
        assert_eq!(actual.dims()[0], 3);
        assert_eq!(&source.dims()[1..], &actual.dims()[1..]);
        let values = source
            .to_dtype(DType::F64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let width = values.len() / 2;
        let expected: Vec<_> = [1, 0, 1]
            .into_iter()
            .flat_map(|index| values[index * width..(index + 1) * width].to_vec())
            .collect();
        assert_eq!(
            actual
                .to_dtype(DType::F64)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f64>()
                .unwrap(),
            expected
        );
    }
    let gradients = selected
        .data_processed
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    let width = obs.data_processed.elem_count() / 2;
    assert_eq!(
        gradients
            .get(&variable)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        [vec![1.; width], vec![2.; width]].concat()
    );
    let empty = obs
        .select_batch(&Tensor::zeros(0, DType::I64, &Device::Cpu).unwrap())
        .unwrap();
    assert!(fields(&empty).iter().all(|tensor| tensor.dims()[0] == 0));
    for invalid in [
        Tensor::new(&[-1_i64], &Device::Cpu).unwrap(),
        Tensor::new(&[2_i64], &Device::Cpu).unwrap(),
        Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
    ] {
        assert!(obs.select_batch(&invalid).is_err());
    }
    for index in 0..7 {
        let mut invalid = obs.detached();
        let fields = [
            &mut invalid.data_processed,
            &mut invalid.cur_tick,
            &mut invalid.cur_step,
            &mut invalid.position_history,
            &mut invalid.target,
            &mut invalid.num_step,
            &mut invalid.acquiring,
        ];
        *fields[index] = Tensor::zeros((), fields[index].dtype(), &Device::Cpu).unwrap();
        assert!(invalid.select_batch(&indices).is_err());
    }
}

#[test]
fn recurrent_heads_and_shared_gradients_match_real_torch() {
    let cases = fixture::cases("recurrent");
    assert_eq!(cases.len(), 6);
    for case in cases {
        let weights = case.weights();
        let builder = fixture::builder(&weights);
        let config: RecurrentConfig = serde_json::from_value(case.config.clone()).unwrap();
        let extractor =
            Arc::new(RecurrentFeatures::new(config, builder.pp("actor.extractor")).unwrap());
        let actor = PpoActor::new(extractor.clone(), 3, builder.pp("actor")).unwrap();
        let critic = PpoCritic::new(extractor.clone(), builder.pp("critic")).unwrap();
        let registered: IndexMap<_, _> = actor
            .parameters()
            .iter()
            .map(|(n, t)| (format!("actor.{n}"), t.clone()))
            .chain(
                critic
                    .parameters()
                    .iter()
                    .map(|(n, t)| (format!("critic.{n}"), t.clone())),
            )
            .collect();
        assert_eq!(
            registered.keys().collect::<Vec<_>>(),
            weights.keys().collect::<Vec<_>>()
        );
        for (name, tensor) in &registered {
            assert_eq!(tensor.id(), weights[name].id(), "{} {name}", case.name);
        }
        let inputs = case.inputs();
        let obs = observation(&inputs);
        let state = Box::new(String::from("opaque caller-owned state"));
        let state_ptr = std::ptr::from_ref(state.as_ref());
        let (actor_output, returned) = actor.forward(&obs, state).unwrap();
        assert_eq!(state_ptr, std::ptr::from_ref(returned.as_ref()));
        let critic_output = critic.forward(&obs).unwrap();
        let features = extractor.forward(&obs).unwrap();
        let (sources, public) = extractor.source_features(&obs).unwrap();
        for (name, tensor) in [
            ("actor", &actor_output),
            ("critic", &critic_output),
            ("features", &features),
            ("public", &public),
            ("public_slice", &sources[0]),
            ("private", &sources[1]),
            ("direction", &sources[2]),
        ] {
            case.outputs[name].compare(tensor, &format!("{} {name}", case.name));
        }
        let loss = (actor_output.sqr().unwrap().sum_all().unwrap()
            + critic_output.sqr().unwrap().sum_all().unwrap())
        .unwrap();
        let loss = (loss + features.sum_all().unwrap().affine(0.2, 0.).unwrap()).unwrap();
        let loss = (loss + public.sum_all().unwrap().affine(0.3, 0.).unwrap()).unwrap();
        case.check_gradients(&loss.backward().unwrap(), &weights, &inputs);
    }
}

#[test]
fn relu_boundary_matches_torch_values_and_gradients() {
    use candle_core::{Device, Var};
    let input = Var::new(
        &[f32::NEG_INFINITY, -1., -0., 0., 1., f32::INFINITY, f32::NAN],
        &Device::Cpu,
    )
    .unwrap();
    let output = relu(&input).unwrap();
    let values = output.to_vec1::<f32>().unwrap();
    for (value, expected) in values[..6]
        .iter()
        .zip([0_f32, 0., -0., 0., 1., f32::INFINITY])
    {
        assert_eq!(value.to_bits(), expected.to_bits());
    }
    assert!(values[6].is_nan());
    let gradients = output.sum_all().unwrap().backward().unwrap();
    assert_eq!(
        gradients.get(&input).unwrap().to_vec1::<f32>().unwrap(),
        [0., 0., 0., 0., 1., 1., 1.]
    );
}

#[test]
fn configuration_initialization_and_parameter_errors_are_explicit() {
    use candle_core::Device;
    use candle_nn::VarMap;
    let defaults = RecurrentConfig::new(5);
    assert_eq!(
        (
            defaults.data_dim,
            defaults.hidden_dim,
            defaults.output_dim,
            defaults.layers
        ),
        (5, 64, 32, 1)
    );
    assert_eq!(defaults.kind, RecurrentKind::Gru);
    let base = RecurrentConfig {
        data_dim: 2,
        hidden_dim: 2,
        output_dim: 3,
        kind: RecurrentKind::Rnn,
        layers: 1,
    };
    for (config, error) in [
        (
            RecurrentConfig {
                hidden_dim: 0,
                ..base
            },
            "positive",
        ),
        (RecurrentConfig { layers: 0, ..base }, "positive"),
        (
            RecurrentConfig {
                hidden_dim: usize::MAX,
                ..base
            },
            "source feature dimension overflow",
        ),
        (
            RecurrentConfig {
                hidden_dim: usize::MAX / 4 + 1,
                kind: RecurrentKind::Lstm,
                ..base
            },
            "recurrent gate dimension overflow",
        ),
    ] {
        assert!(
            RecurrentFeatures::new(config, VarBuilder::zeros(DType::F32, &Device::Cpu))
                .err()
                .unwrap()
                .to_string()
                .contains(error)
        );
    }
    assert!(
        RecurrentFeatures::new(
            base,
            VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu)
        )
        .is_err()
    );
    let wrong = HashMap::from([(
        "raw_rnn.weight_ih_l0".into(),
        Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap(),
    )]);
    assert!(
        RecurrentFeatures::new(
            base,
            VarBuilder::from_tensors(wrong, DType::F32, &Device::Cpu)
        )
        .is_err()
    );
    let map = VarMap::new();
    let model = RecurrentFeatures::new(
        base,
        VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu),
    )
    .unwrap();
    assert_eq!(model.parameters().len(), 24);
    for tensor in model.parameters().values() {
        assert!(tensor.is_variable());
        assert!(
            tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.abs() <= 1.)
        );
    }
    let mut parameters = Parameters::new(VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu));
    let layer = parameters.linear("zero-input", 0, 2).unwrap();
    assert_eq!(layer.bias().unwrap().to_vec1::<f32>().unwrap(), [0., 0.]);
}

#[test]
fn indexing_and_sequence_boundaries_preserve_errors() {
    use candle_core::Device;
    let sequence_tensor = Tensor::arange(0_f32, 12., &Device::Cpu)
        .unwrap()
        .reshape((2, 3, 2))
        .unwrap();
    for (indices, expected) in [
        ([0_i64, 2], vec![vec![0., 1.], vec![10., 11.]]),
        ([-3, -1], vec![vec![0., 1.], vec![10., 11.]]),
    ] {
        assert_eq!(
            select_step(
                &sequence_tensor,
                &Tensor::new(&indices, &Device::Cpu).unwrap()
            )
            .unwrap()
            .to_vec2::<f32>()
            .unwrap(),
            expected
        );
    }
    for indices in [vec![0_i64], vec![-4, 0], vec![0, 3], vec![i64::MIN, 0]] {
        assert!(
            select_step(
                &sequence_tensor,
                &Tensor::new(indices, &Device::Cpu).unwrap()
            )
            .is_err()
        );
    }
    assert!(
        select_step(
            &sequence_tensor,
            &Tensor::new(&[0_f32, 1.], &Device::Cpu).unwrap()
        )
        .is_err()
    );
    assert!(
        select_step(
            &sequence_tensor.flatten_all().unwrap(),
            &Tensor::new(&[0_i64], &Device::Cpu).unwrap()
        )
        .is_err()
    );
    assert!(
        sequence(
            &[],
            &Tensor::zeros((1, 0, 2), DType::F32, &Device::Cpu).unwrap()
        )
        .unwrap_err()
        .to_string()
        .contains("must not be empty")
    );
    let huge = Tensor::zeros((0, usize::MAX, 0), DType::F32, &Device::Cpu).unwrap();
    assert!(
        select_step(
            &huge,
            &Tensor::new(Vec::<i64>::new(), &Device::Cpu).unwrap()
        )
        .is_err()
    );
}

#[test]
fn native_checkpoint_restores_actual_shared_model_computation() {
    use crate::rl_candle_checkpoint::CandlePolicyState;
    use crate::rl_policy_weight::set_policy_weights;
    use candle_core::Var;
    let case = fixture::cases("recurrent").pop().unwrap();
    let weights = case.weights();
    let builder = fixture::builder(&weights);
    let config = serde_json::from_value(case.config.clone()).unwrap();
    let extractor =
        Arc::new(RecurrentFeatures::new(config, builder.pp("actor.extractor")).unwrap());
    let actor = PpoActor::new(extractor.clone(), 3, builder.pp("actor")).unwrap();
    let critic = PpoCritic::new(extractor, builder.pp("critic")).unwrap();
    let obs = observation(&case.inputs());
    let variables = weights
        .iter()
        .map(|(name, tensor)| (name.clone(), Var::from_tensor(tensor).unwrap()))
        .collect();
    let mut state = CandlePolicyState::new(variables, 17_u64);
    let snapshot = state.snapshot().unwrap();
    let actor_before = actor.forward(&obs, ()).unwrap().0.to_vec2::<f32>().unwrap();
    let critic_before = critic.forward(&obs).unwrap().to_vec1::<f32>().unwrap();
    for name in [
        "actor.extractor.fc.2.bias",
        "actor.layer_out.0.bias",
        "critic.value_out.bias",
    ] {
        let variable = &state.variables()[name];
        let changed = Tensor::arange(1_f32, 4., variable.device()).unwrap();
        let changed = changed.narrow(0, 0, variable.elem_count()).unwrap();
        variable.set(&changed).unwrap();
    }
    assert_ne!(
        actor_before,
        actor.forward(&obs, ()).unwrap().0.to_vec2::<f32>().unwrap()
    );
    assert_ne!(
        critic_before,
        critic.forward(&obs).unwrap().to_vec1::<f32>().unwrap()
    );
    let mut restored = snapshot.into_policy_weights(None).unwrap();
    set_policy_weights(&mut state, &mut restored).unwrap();
    assert_eq!(restored.metadata, 17);
    assert_eq!(
        actor_before,
        actor.forward(&obs, ()).unwrap().0.to_vec2::<f32>().unwrap()
    );
    assert_eq!(
        critic_before,
        critic.forward(&obs).unwrap().to_vec1::<f32>().unwrap()
    );
    for (name, tensor) in actor
        .parameters()
        .iter()
        .filter(|(name, _)| name.starts_with("extractor."))
    {
        assert_eq!(tensor.id(), critic.parameters()[name].id());
    }
}

#[test]
fn linear_empty_and_low_precision_paths_keep_values_and_zero_gradients() {
    use candle_core::{Device, Var};
    let device = &Device::Cpu;
    for (rows, channels, outputs) in [(0, 2, 3), (2, 0, 3), (2, 3, 0)] {
        for bias in [false, true] {
            let input = Var::ones((rows, channels), DType::F32, device).unwrap();
            let weight = Var::ones((outputs, channels), DType::F32, device).unwrap();
            let bias_var = Var::ones(outputs, DType::F32, device).unwrap();
            let layer = Linear::new(
                weight.as_tensor().clone(),
                bias.then(|| bias_var.as_tensor().clone()),
            );
            let output = linear_forward(&layer, &input).unwrap();
            assert_eq!(output.dims(), [rows, outputs]);
            assert!(
                output
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| v.to_bits() == f32::from(u8::from(bias)).to_bits())
            );
            let gradients = output.sum_all().unwrap().backward().unwrap();
            for variable in [&input, &weight] {
                assert!(
                    gradients
                        .get(variable)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .iter()
                        .all(|v| v.to_bits() == 0)
                );
            }
            if bias {
                assert!(gradients.get(&bias_var).is_some());
            }
        }
    }
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let input = Tensor::ones((2, 3), dtype, device).unwrap();
        let weight = Tensor::ones((1, 3), dtype, device).unwrap();
        let layer = Linear::new(weight, None);
        assert_eq!(
            linear_forward(&layer, &input)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![3.], vec![3.]]
        );
        assert!(linear_forward(&layer, &Tensor::zeros((), dtype, device).unwrap()).is_err());
        assert!(linear_forward(&layer, &Tensor::zeros((0, 4), dtype, device).unwrap()).is_err());
        let other_dtype = if dtype == DType::F64 {
            DType::F32
        } else {
            DType::F64
        };
        assert!(linear_forward(&layer, &input.to_dtype(other_dtype).unwrap()).is_err());
    }
}

#[test]
fn invalid_observations_fail_before_producing_predictions() {
    use candle_core::Device;
    let case = fixture::cases("recurrent").pop().unwrap();
    let weights = case.weights();
    let model = RecurrentFeatures::new(
        serde_json::from_value(case.config.clone()).unwrap(),
        fixture::builder(&weights).pp("actor.extractor"),
    )
    .unwrap();
    let inputs = case.inputs();
    let mut obs = observation(&inputs);
    obs.data_processed = Tensor::zeros((2, 3, 1), DType::F32, &Device::Cpu).unwrap();
    assert!(model.forward(&obs).is_err());
    let mut obs = observation(&inputs);
    obs.position_history = Tensor::zeros((2, 0), DType::F32, &Device::Cpu).unwrap();
    assert!(
        model
            .forward(&obs)
            .unwrap_err()
            .to_string()
            .contains("must not be empty")
    );
    let mut obs = observation(&inputs);
    obs.cur_tick = Tensor::new(&[4_i64, 0], &Device::Cpu).unwrap();
    assert!(
        model
            .forward(&obs)
            .unwrap_err()
            .to_string()
            .contains("out of range")
    );
    let mut obs = observation(&inputs);
    obs.cur_step = Tensor::new(&[0_i64, -4], &Device::Cpu).unwrap();
    assert!(
        model
            .forward(&obs)
            .unwrap_err()
            .to_string()
            .contains("out of range")
    );
    let mut obs = observation(&inputs);
    obs.target = Tensor::zeros((), DType::F32, &Device::Cpu).unwrap();
    assert!(model.forward(&obs).is_err());
}

#[test]
fn every_registered_parameter_is_required_by_the_model_constructor() {
    let case = fixture::cases("recurrent").pop().unwrap();
    let weights: IndexMap<_, _> = case
        .weights()
        .into_iter()
        .filter_map(|(name, tensor)| {
            name.strip_prefix("actor.extractor.")
                .map(|name| (name.to_owned(), tensor))
        })
        .collect();
    let config: RecurrentConfig = serde_json::from_value(case.config).unwrap();
    for missing in weights.keys() {
        let mut incomplete = weights.clone();
        incomplete.shift_remove(missing);
        let error = RecurrentFeatures::new(config, fixture::builder(&incomplete))
            .err()
            .unwrap();
        assert!(
            error.to_string().contains(missing),
            "missing {missing}: {error}"
        );
    }
}

#[derive(serde::Deserialize)]
struct TrainingStep {
    learning_rate: f64,
    ppo: Option<PpoTrainingBatch>,
    max_grad_norm: Option<f64>,
    gradient_norm: Option<fixture::Record>,
    initialized: usize,
    weights: IndexMap<String, fixture::Record>,
    actor: fixture::Record,
    critic: fixture::Record,
}
#[derive(serde::Deserialize)]
struct PpoTrainingBatch {
    inputs: IndexMap<String, fixture::Record>,
    metrics: IndexMap<String, f64>,
}
#[derive(serde::Deserialize)]
struct TrainingTrajectory {
    name: String,
    parameter_count: usize,
    steps: Vec<TrainingStep>,
}
#[derive(serde::Deserialize)]
struct TrainingFixture {
    trajectories: Vec<TrainingTrajectory>,
}

#[test]
fn shared_network_adam_trajectories_match_real_qlib_and_torch() {
    check_training_trajectories(include_str!("../fixtures/rl_candle_training.json"));
}

#[test]
fn shared_network_clipped_adam_trajectories_match_real_qlib_and_torch() {
    check_training_trajectories(include_str!("../fixtures/rl_candle_clipped_training.json"));
}

#[test]
fn real_qlib_ppo_learn_updates_match_native_shared_networks_loss_and_adam() {
    check_training_trajectories(include_str!("../fixtures/rl_candle_ppo_training.json"));
}

fn trajectory_loss(
    index: usize,
    expected: &TrainingStep,
    actor: &PpoActor,
    critic: &PpoCritic,
    extractor: &RecurrentFeatures,
    obs: &RecurrentObservation,
) -> Tensor {
    let probabilities = actor.forward(obs, ()).unwrap().0;
    let values = critic.forward(obs).unwrap();
    if let Some(prepared) = &expected.ppo {
        use crate::rl_candle_ppo::{PpoLossConfig, PpoLossInput};
        let tensors: IndexMap<_, _> = prepared
            .inputs
            .iter()
            .map(|(name, record)| (name.as_str(), record.tensor()))
            .collect();
        let loss = PpoLossConfig::default()
            .loss(&PpoLossInput {
                probabilities: &probabilities,
                values: &values,
                actions: &tensors["actions"],
                old_log_prob: &tensors["old_log_prob"],
                advantages: &tensors["advantages"],
                returns: &tensors["returns"],
                old_values: Some(&tensors["old_values"]),
            })
            .unwrap();
        for (name, tensor) in [
            ("loss", &loss.total),
            ("loss/clip", &loss.policy),
            ("loss/vf", &loss.value),
            ("loss/ent", &loss.entropy),
        ] {
            fixture::Record {
                shape: vec![],
                dtype: "float32".into(),
                values: vec![prepared.metrics[name]],
            }
            .compare(tensor, name);
        }
        return loss.total;
    }
    let actor_loss = probabilities.sqr().unwrap().sum_all().unwrap();
    let critic_loss = values.sqr().unwrap().sum_all().unwrap();
    match index {
        1 => actor_loss,
        2 => critic_loss,
        _ => (actor_loss + critic_loss)
            .unwrap()
            .add(
                &extractor
                    .forward(obs)
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .affine(0.2, 0.)
                    .unwrap(),
            )
            .unwrap(),
    }
}

fn check_training_trajectories(json: &str) {
    use crate::rl_candle_optimizer::CandleAdam;
    let trajectories: TrainingFixture = serde_json::from_str(json).unwrap();
    let cases = fixture::cases("recurrent");
    assert_eq!(trajectories.trajectories.len(), 6);
    for trajectory in trajectories.trajectories {
        let case = cases
            .iter()
            .find(|case| case.name == trajectory.name)
            .unwrap();
        let weights = case.weights();
        let builder = fixture::builder(&weights);
        let extractor = Arc::new(
            RecurrentFeatures::new(
                serde_json::from_value(case.config.clone()).unwrap(),
                builder.pp("actor.extractor"),
            )
            .unwrap(),
        );
        let actor = PpoActor::new(extractor.clone(), 3, builder.pp("actor")).unwrap();
        let critic = PpoCritic::new(extractor.clone(), builder.pp("critic")).unwrap();
        let parameters = actor
            .parameters()
            .values()
            .chain(critic.parameters().values())
            .cloned()
            .collect();
        let mut optimizer = CandleAdam::new(parameters, 0.003, 0.1).unwrap();
        assert_eq!(optimizer.parameters().len(), trajectory.parameter_count);
        let obs = observation(&case.inputs());
        for (index, expected) in trajectory.steps.iter().enumerate() {
            optimizer.set_learning_rate(expected.learning_rate).unwrap();
            let loss = trajectory_loss(index, expected, &actor, &critic, &extractor, &obs);
            let mut gradients = loss.backward().unwrap();
            if let Some(limit) = expected.max_grad_norm {
                let norm = optimizer.clip_grad_norm(&mut gradients, limit).unwrap();
                expected
                    .gradient_norm
                    .as_ref()
                    .unwrap()
                    .compare(&norm, "pre-clipping norm");
            } else {
                assert!(expected.gradient_norm.is_none());
            }
            optimizer.step(&gradients).unwrap();
            assert_eq!(
                optimizer.initialized_parameter_count(),
                expected.initialized
            );
            for (name, record) in &expected.weights {
                record.compare(
                    &weights[name.strip_prefix("_actor_critic.").unwrap_or(name)],
                    &format!("{} update {index} {name}", trajectory.name),
                );
            }
            expected.actor.compare(
                &actor.forward(&obs, ()).unwrap().0,
                &format!("{} update {index} actor", trajectory.name),
            );
            expected.critic.compare(
                &critic.forward(&obs).unwrap(),
                &format!("{} update {index} critic", trajectory.name),
            );
        }
    }
}
