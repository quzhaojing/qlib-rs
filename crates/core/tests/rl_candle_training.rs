use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use domain_core::{
    TrainingMetricValue, TrainingPolicyMode, TrainingRunPolicy,
    rl_candle_categorical::CandleCategorical,
    rl_candle_checkpoint::CandlePolicyState,
    rl_candle_heads::{PpoActor, PpoCritic},
    rl_candle_network::{
        CandleFeatureExtractor, RecurrentConfig, RecurrentFeatures, RecurrentKind,
        RecurrentObservation,
    },
    rl_candle_optimizer::CandleAdam,
    rl_candle_policy::{CandlePpo, CandlePpoConfig, CandlePpoRollout},
    rl_candle_ppo::{PpoLossConfig, PpoLossInput},
    rl_candle_replay::{
        CandleReplayBatch, CandleReplayBuffer, CandleReplayTransition, CandleVectorReplayBuffer,
    },
    rl_candle_returns::{CandleReturnInput, ReturnStatistics, prepare_returns},
    rl_candle_vessel::CandlePpoVessel,
    rl_policy_batch::minibatch_indices,
};
use indexmap::IndexMap;
use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use std::sync::Arc;

fn prepared_loss(
    probabilities: &Tensor,
    distribution: &CandleCategorical,
    values: &Tensor,
) -> Tensor {
    let device = probabilities.device();
    let actions = Tensor::new(&[0_i64, 2], device)
        .unwrap()
        .to_dtype(values.dtype())
        .unwrap();
    let old_log_prob = distribution.log_prob(&actions).unwrap().detach();
    let next_values = Tensor::cat(
        &[
            values.narrow(0, 1, 1).unwrap(),
            Tensor::zeros(1, values.dtype(), device).unwrap(),
        ],
        0,
    )
    .unwrap();
    let rewards = array![0.2, -0.4];
    let terminated = array![false, true];
    let truncated = array![false, false];
    let bootstrap_valid = array![true, false];
    let mut statistics = ReturnStatistics::default();
    let prepared = prepare_returns(
        &CandleReturnInput {
            rewards: rewards.view(),
            terminated: terminated.view(),
            truncated: truncated.view(),
            bootstrap_valid: bootstrap_valid.view(),
            indices: &[0, 1],
            unfinished_indices: &[],
            next_values: &next_values,
            values,
        },
        1.,
        1.,
        Some(&mut statistics),
    )
    .unwrap();
    assert_eq!(statistics.count, 2);
    assert!(statistics.variance > 0.);
    let loss = PpoLossConfig::default()
        .loss(&PpoLossInput {
            probabilities,
            values,
            actions: &actions,
            old_log_prob: &old_log_prob,
            advantages: &prepared.advantages,
            returns: &prepared.returns,
            old_values: Some(&prepared.old_values),
        })
        .unwrap();
    assert!(loss.total.to_scalar::<f32>().unwrap().is_finite());
    loss.total
}

#[test]
fn public_native_model_updates_and_checkpoint_restores_live_shared_parameters() {
    let device = &Device::Cpu;
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, device);
    let extractor = Arc::new(
        RecurrentFeatures::new(
            RecurrentConfig {
                data_dim: 2,
                hidden_dim: 2,
                output_dim: 3,
                kind: RecurrentKind::Gru,
                layers: 2,
            },
            builder.pp("extractor"),
        )
        .unwrap(),
    );
    let actor = PpoActor::new(extractor.clone(), 3, builder.pp("actor")).unwrap();
    let critic = PpoCritic::new(extractor.clone(), builder.pp("critic")).unwrap();
    let registered: IndexMap<_, _> =
        actor
            .parameters()
            .iter()
            .map(|(name, tensor)| (format!("actor.{name}"), Var::from_tensor(tensor).unwrap()))
            .chain(critic.parameters().iter().map(|(name, tensor)| {
                (format!("critic.{name}"), Var::from_tensor(tensor).unwrap())
            }))
            .collect();
    for variable in registered.values() {
        variable
            .set(
                &Tensor::ones(variable.shape(), variable.dtype(), device)
                    .unwrap()
                    .affine(0.1, 0.)
                    .unwrap(),
            )
            .unwrap();
    }
    registered["actor.layer_out.0.bias"]
        .set(&Tensor::new(&[0_f32, 0.2, -0.1], device).unwrap())
        .unwrap();
    let state = CandlePolicyState::new(registered, 1_u64);
    let mut optimizer = CandleAdam::new(
        actor
            .parameters()
            .values()
            .chain(critic.parameters().values())
            .cloned()
            .collect(),
        0.01,
        0.1,
    )
    .unwrap();
    assert_eq!(
        optimizer.parameters().len(),
        actor.parameters().len() + critic.parameters().len() - extractor.parameters().len()
    );
    let obs = observation(device);
    let policy_before = actor
        .policy_forward(
            &obs,
            (),
            TrainingPolicyMode::Train,
            true,
            &mut StdRng::seed_from_u64(41),
        )
        .unwrap();
    let actor_before = policy_before.logits;
    assert_eq!(policy_before.actions.dims(), &[2]);
    check_minibatches(&actor, &obs, &actor_before);
    let critic_before = critic.forward(&obs).unwrap();
    let snapshot = state.snapshot().unwrap();
    let loss = prepared_loss(&actor_before, &policy_before.distribution, &critic_before);
    let mut gradients = loss.backward().unwrap();
    let norm = optimizer.clip_grad_norm(&mut gradients, 0.1).unwrap();
    assert!(norm.to_scalar::<f32>().unwrap() > 0.1);
    assert_eq!(optimizer.initialized_parameter_count(), 0);
    optimizer.step(&gradients).unwrap();
    assert_ne!(
        actor_before.to_vec2::<f32>().unwrap(),
        actor.forward(&obs, ()).unwrap().0.to_vec2::<f32>().unwrap()
    );
    assert_ne!(
        critic_before.to_vec1::<f32>().unwrap(),
        critic.forward(&obs).unwrap().to_vec1::<f32>().unwrap()
    );
    assert!(optimizer.initialized_parameter_count() > 0);
    state.restore(&snapshot).unwrap();
    assert_eq!(
        actor_before.to_vec2::<f32>().unwrap(),
        actor.forward(&obs, ()).unwrap().0.to_vec2::<f32>().unwrap()
    );
    assert_eq!(
        critic_before.to_vec1::<f32>().unwrap(),
        critic.forward(&obs).unwrap().to_vec1::<f32>().unwrap()
    );
    // This restores policy parameters, not optimizer moments. Do not assert a
    // bitwise training-resume contract which source policy.state_dict does not offer.
}

fn observation(device: &Device) -> RecurrentObservation {
    RecurrentObservation {
        data_processed: Tensor::ones((2, 2, 2), DType::F32, device).unwrap(),
        cur_tick: Tensor::new(&[0_i64, 2], device).unwrap(),
        cur_step: Tensor::new(&[0_i64, 1], device).unwrap(),
        position_history: Tensor::ones((2, 2), DType::F32, device).unwrap(),
        target: Tensor::ones(2, DType::F32, device).unwrap(),
        num_step: Tensor::new(&[2_i64, 2], device).unwrap(),
        acquiring: Tensor::new(&[0_i64, 1], device).unwrap(),
    }
}

fn check_minibatches(actor: &PpoActor, obs: &RecurrentObservation, expected: &Tensor) {
    let mut rng = StdRng::seed_from_u64(7);
    let batches = minibatch_indices(2, 1, false, true, &mut rng).unwrap();
    let mut predictions = Vec::new();
    for positions in batches {
        let indices: Vec<_> = positions
            .into_iter()
            .map(|index| i64::try_from(index).unwrap())
            .collect();
        let obs = obs
            .select_batch(&Tensor::new(indices.as_slice(), &Device::Cpu).unwrap())
            .unwrap();
        let prediction = actor
            .policy_forward(&obs, (), TrainingPolicyMode::Evaluation, true, &mut rng)
            .unwrap();
        predictions.push(prediction.logits);
    }
    assert_eq!(
        Tensor::cat(&predictions, 0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap(),
        expected.to_vec2::<f32>().unwrap()
    );
}

#[test]
fn assembled_ppo_trains_a_real_gru_across_merged_minibatches() {
    let device = &Device::Cpu;
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, device);
    let extractor = Arc::new(
        RecurrentFeatures::new(
            RecurrentConfig {
                data_dim: 2,
                hidden_dim: 2,
                output_dim: 3,
                kind: RecurrentKind::Gru,
                layers: 2,
            },
            builder.pp("extractor"),
        )
        .unwrap(),
    );
    let config = CandlePpoConfig {
        max_batch_size: 2,
        ..CandlePpoConfig::new(0.003)
    };
    let mut policy = CandlePpo::new(extractor, 3, &builder, config).unwrap();
    let state = policy.policy_state(()).unwrap();
    let registered = state.variables();
    for variable in registered.values() {
        variable
            .set(
                &Tensor::ones(variable.shape(), variable.dtype(), device)
                    .unwrap()
                    .affine(0.1, 0.)
                    .unwrap(),
            )
            .unwrap();
    }
    registered["actor.layer_out.0.bias"]
        .set(&Tensor::new(&[0_f32, 0.2, -0.1], device).unwrap())
        .unwrap();
    let snapshot = state.snapshot().unwrap();
    let observations = observation(device)
        .select_batch(&Tensor::new(&[0_i64, 1, 0, 1, 0], device).unwrap())
        .unwrap();
    let rollout = CandlePpoRollout {
        next_observations: observations.detached(),
        observations,
        actions: Tensor::new(&[0_i64, 2, 1, 0, 2], device).unwrap(),
        rewards: array![0.2, -0.4, 0.1, 0.8, -0.2],
        terminated: array![false, true, false, false, true],
        truncated: array![false, false, false, false, false],
        bootstrap_valid: array![true, false, true, true, false],
        indices: vec![0, 1, 2, 3, 4],
        unfinished_indices: vec![],
        replay_weights: None,
    };
    let predict = |policy: &CandlePpo| {
        policy
            .actor
            .forward(&rollout.observations, ())
            .unwrap()
            .0
            .to_vec2::<f32>()
            .unwrap()
    };
    let before = predict(&policy);
    let mut rng = StdRng::seed_from_u64(41);
    let mut batch = policy.process(&rollout, &mut rng).unwrap();
    let metrics = policy.learn(&mut batch, 2, 2, &mut rng).unwrap();
    assert!(
        metrics
            .values()
            .all(|values| values.len() == 4 && values.iter().all(|value| value.is_finite()))
    );
    assert_eq!(policy.return_statistics.count, 5);
    // Source registers prev_rnn for state compatibility but never uses it in
    // forward. Absent gradients must not initialize its Adam state.
    let unused = policy
        .actor
        .parameters()
        .iter()
        .filter(|(name, _)| name.starts_with("extractor.prev_rnn."))
        .count();
    assert_eq!(
        policy.optimizer.initialized_parameter_count(),
        policy.optimizer.parameters().len() - unused
    );
    assert_ne!(before, predict(&policy));
    state.restore(&snapshot).unwrap();
    assert_eq!(before, predict(&policy));
    verify_vessel_bridge(policy, &rollout);
}

fn verify_vessel_bridge(policy: CandlePpo, rollout: &CandlePpoRollout) {
    let mut runtime = CandlePpoVessel::new(policy, StdRng::seed_from_u64(57));
    let mut buffer = CandleReplayBuffer::new(rollout.rewards.len());
    for index in 0..rollout.rewards.len() {
        let row = Tensor::new(&[i64::try_from(index).unwrap()], rollout.actions.device()).unwrap();
        buffer
            .add(&CandleReplayTransition {
                observation: rollout.observations.select_batch(&row).unwrap(),
                next_observation: rollout.next_observations.select_batch(&row).unwrap(),
                action: rollout.actions.narrow(0, index, 1).unwrap(),
                reward: rollout.rewards[index],
                terminated: rollout.terminated[index],
                truncated: rollout.truncated[index],
            })
            .unwrap();
    }
    let options = IndexMap::from([
        ("batch_size".into(), serde_json::json!(2)),
        ("repeat".into(), serde_json::json!(2)),
        ("extra".into(), serde_json::json!("ignored like source")),
    ]);
    let metrics = runtime.update(0, Some(&mut buffer), &options).unwrap();
    assert_eq!(
        metrics.keys().map(String::as_str).collect::<Vec<_>>(),
        ["loss", "loss/clip", "loss/vf", "loss/ent"]
    );
    for metric in metrics.values() {
        let TrainingMetricValue::Numeric(values) = metric else {
            panic!("expected numeric metric");
        };
        let values = values
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();
        assert_eq!(values.len(), 4);
        assert!(values.values().iter().all(|value| value.is_finite()));
    }
    assert_eq!(buffer.index().len(), rollout.rewards.len());
    assert_eq!(
        buffer.get(&rollout.indices).unwrap().rewards,
        rollout.rewards
    );
    assert!(!runtime.policy.is_updating());
    assert_eq!(runtime.policy.return_statistics.count, 10);
    verify_vector_bridge(&mut runtime, rollout, &options);
}

fn verify_vector_bridge(
    runtime: &mut CandlePpoVessel<StdRng>,
    rollout: &CandlePpoRollout,
    options: &IndexMap<String, serde_json::Value>,
) {
    let mut buffer = CandleVectorReplayBuffer::new(5, 2).unwrap();
    let added = buffer
        .add(
            &CandleReplayBatch {
                observations: &rollout.observations,
                next_observations: &rollout.next_observations,
                actions: &rollout.actions,
                rewards: rollout.rewards.as_slice().unwrap(),
                terminated: rollout.terminated.as_slice().unwrap(),
                truncated: rollout.truncated.as_slice().unwrap(),
            },
            Some(&[0, 1, 0, 1, 0]),
        )
        .unwrap();
    let indices: Vec<_> = added.iter().map(|episode| episode.index).collect();
    assert_eq!(indices, [0, 3, 1, 4, 2]);
    assert_eq!(buffer.get(&indices).unwrap().rewards, rollout.rewards);
    assert_eq!(buffer.unfinished_indices().unwrap(), [4]);
    verify_nstep_targets(runtime, &buffer, &indices);
    let predict = |runtime: &CandlePpoVessel<StdRng>| {
        runtime
            .policy
            .actor
            .forward(&rollout.observations, ())
            .unwrap()
            .0
            .to_vec2::<f32>()
            .unwrap()
    };
    let before = predict(runtime);
    let metrics = runtime.update(0, Some(&mut buffer), options).unwrap();
    assert_eq!(
        metrics.keys().map(String::as_str).collect::<Vec<_>>(),
        ["loss", "loss/clip", "loss/vf", "loss/ent"]
    );
    for metric in metrics.values() {
        let TrainingMetricValue::Numeric(values) = metric else {
            panic!("expected numeric metric");
        };
        let values = values
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();
        assert_eq!(values.len(), 4);
        assert!(values.values().iter().all(|value| value.is_finite()));
    }
    assert_ne!(predict(runtime), before);
    assert_eq!(runtime.policy.return_statistics.count, 15);
    assert!(!runtime.policy.is_updating());
    assert_eq!(buffer.get(&indices).unwrap().rewards, rollout.rewards);
}

fn verify_nstep_targets(
    runtime: &CandlePpoVessel<StdRng>,
    buffer: &CandleVectorReplayBuffer,
    indices: &[usize],
) {
    use domain_core::rl_candle_nstep::{
        CandleNStepBatch, CandleNStepConfig, prepare_nstep_returns,
    };
    let mut batch = CandleNStepBatch::default();
    let mut target_values = Vec::new();
    let statistics_before = runtime.policy.return_statistics.clone();
    prepare_nstep_returns(
        &mut batch,
        buffer,
        indices,
        |buffer, terminal| {
            assert_eq!(terminal, [2, 3, 2, 4, 2]);
            let rows = buffer.get(terminal).map_err(|error| error.to_string())?;
            // Qlib DqnModel reuses this same softmax actor and full-history GRU.
            let values = runtime
                .policy
                .actor
                .forward(&rows.next_observations, ())
                .and_then(|(logits, ())| logits.max(1))
                .map_err(|error| error.to_string())?;
            target_values = values.to_vec1::<f32>().unwrap();
            Ok(values)
        },
        CandleNStepConfig {
            gamma: 0.9,
            steps: 3,
            reward_normalization: false,
        },
    )
    .unwrap();
    let values = batch.returns.unwrap().to_vec2::<f32>().unwrap();
    let expected = [0.128, -0.4, -0.08, 0.8 + 0.9 * target_values[3], -0.2];
    assert_eq!(values.len(), expected.len());
    for (value, expected) in values.iter().zip(expected) {
        assert_eq!(value.len(), 1);
        assert!((value[0] - expected).abs() < 1e-6);
    }
    assert!(batch.weight.is_none());
    assert_eq!(runtime.policy.return_statistics, statistics_before);
}

fn dqn_model(device: &Device) -> PpoActor {
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, device);
    let extractor = Arc::new(
        RecurrentFeatures::new(
            RecurrentConfig {
                data_dim: 2,
                hidden_dim: 2,
                output_dim: 3,
                kind: RecurrentKind::Gru,
                layers: 2,
            },
            builder.pp("extractor"),
        )
        .unwrap(),
    );
    let model = PpoActor::new(extractor, 3, builder.pp("model")).unwrap();
    for parameter in model.parameters().values() {
        Var::from_tensor(parameter)
            .unwrap()
            .set(&parameter.ones_like().unwrap().affine(0.1, 0.).unwrap())
            .unwrap();
    }
    model
}

#[test]
fn public_dqn_gru_replay_nstep_and_adam_update_change_live_parameters() {
    use domain_core::rl_candle_dqn::{dqn_forward, dqn_loss, dqn_target_values, learn_dqn_batch};
    use domain_core::rl_candle_nstep::{
        CandleNStepBatch, CandleNStepConfig, prepare_nstep_returns,
    };
    let device = &Device::Cpu;
    let model = dqn_model(device);
    let mut optimizer =
        CandleAdam::new(model.parameters().values().cloned().collect(), 0.003, 0.1).unwrap();
    let obs = observation(device);
    let mut buffer = CandleReplayBuffer::new(4);
    for (index, reward, terminated) in [(0_i64, 0.2, false), (1, -0.4, true)] {
        let row = obs
            .select_batch(&Tensor::new(&[index], device).unwrap())
            .unwrap();
        buffer
            .add(&CandleReplayTransition {
                observation: row.clone(),
                next_observation: row,
                action: Tensor::new(&[index], device).unwrap(),
                reward,
                terminated,
                truncated: false,
            })
            .unwrap();
    }
    let mut batch = CandleNStepBatch::default();
    prepare_nstep_returns(
        &mut batch,
        &buffer,
        &[0, 1],
        |buffer, indices| {
            assert_eq!(indices, [1, 1]);
            let sampled = buffer.get(indices).map_err(|e| e.to_string())?;
            let logits = model
                .forward(&sampled.next_observations, ())
                .map_err(|e| e.to_string())?
                .0;
            let online = dqn_forward(logits, (), None).map_err(|e| e.to_string())?;
            dqn_target_values(&online, None, true).map_err(|e| e.to_string())
        },
        CandleNStepConfig {
            gamma: 0.9,
            steps: 3,
            reward_normalization: false,
        },
    )
    .unwrap();
    let returns = batch.returns.as_ref().unwrap().to_vec2::<f32>().unwrap();
    assert!((returns[0][0] + 0.16).abs() < 1e-6);
    assert!((returns[1][0] + 0.4).abs() < 1e-6);
    let before = model.forward(&obs, ()).unwrap().0.to_vec2::<f32>().unwrap();
    let sampled = buffer.get(&[0, 1]).unwrap();
    let gradients = dqn_loss(
        &model.forward(&sampled.observations, ()).unwrap().0,
        &sampled.actions,
        batch.returns.as_ref().unwrap(),
        None,
        false,
    )
    .unwrap()
    .loss
    .backward()
    .unwrap();
    let expected_initialized = optimizer
        .parameters()
        .iter()
        .filter(|p| gradients.get(p).is_some())
        .count();
    assert!(expected_initialized > 0);
    let loss = learn_dqn_batch(
        &mut batch,
        &sampled.actions,
        || {
            model
                .forward(&sampled.observations, ())
                .map(|(logits, ())| logits)
        },
        &mut optimizer,
        false,
    )
    .unwrap();
    assert!(loss.is_finite() && loss > 0.);
    let priorities = batch.weight.unwrap().to_vec1::<f32>().unwrap();
    for index in 0..2 {
        assert!((priorities[index] - (returns[index][0] - before[index][index])).abs() < 1e-6);
    }
    let after = model.forward(&obs, ()).unwrap().0.to_vec2::<f32>().unwrap();
    assert_ne!(before, after);
    assert!(after.iter().flatten().all(|v| v.is_finite()));
    // Short histories do not traverse every GRU branch. Absent gradients must
    // not initialize an Adam state or advance that parameter's clock.
    assert_eq!(
        optimizer.initialized_parameter_count(),
        expected_initialized
    );
    assert_eq!(buffer.index().len(), 2);
}
