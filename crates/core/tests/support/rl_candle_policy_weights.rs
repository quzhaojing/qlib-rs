use super::*;

#[derive(Deserialize)]
struct WeightCase {
    kind: String,
    initial: IndexMap<String, Record>,
    error: Option<String>,
    input_keys: Vec<String>,
    #[serde(rename = "final")]
    final_weights: IndexMap<String, Record>,
    aliases: Vec<usize>,
}

fn fixture() -> Vec<WeightCase> {
    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<WeightCase>,
    }
    serde_json::from_str::<Fixture>(include_str!("../fixtures/rl_candle_ppo_weights.json"))
        .unwrap()
        .cases
}

struct OpaqueMetadata(&'static str);

#[test]
fn trainable_replacement_models_preserve_source_container_and_optimizer_references() {
    for actor in [true, false] {
        let case = super::cases().remove(0);
        let (features, variables, config) = setup(&case);
        let mut policy = CandlePpo::new(features, 3, &builder(&variables), config).unwrap();
        let saved = policy.state_dict().unwrap();
        let name = if actor {
            "actor.layer_out.0.weight"
        } else {
            "critic.value_out.weight"
        };
        let old = variables[name].clone();
        let optimizer_ids: Vec<_> = policy
            .optimizer
            .parameters()
            .iter()
            .map(|value| value.id())
            .collect();
        let (features, replacements, _) = setup(&case);
        if actor {
            policy.actor = PpoActor::new(features, 3, builder(&replacements).pp("actor")).unwrap();
        } else {
            policy.critic = PpoCritic::new(features, builder(&replacements).pp("critic")).unwrap();
        }
        let current = replacements[name].clone();
        assert_ne!(current.id(), old.id());
        for (tensor, value) in [(&old, 7.), (&current, 9.)] {
            Var::from_tensor(tensor)
                .unwrap()
                .set(&tensor.ones_like().unwrap().affine(value, 0.).unwrap())
                .unwrap();
        }
        let registered = policy.policy_state(()).unwrap();
        assert_eq!(registered.variables()[name].id(), current.id());
        assert_eq!(
            registered.variables()[&format!("_actor_critic.{name}")].id(),
            old.id()
        );
        let changed = registered
            .snapshot()
            .unwrap()
            .into_policy_weights(None)
            .unwrap();
        let old_name = format!("_actor_critic.{name}");
        assert!(!Arc::ptr_eq(
            &changed.weights[name],
            &changed.weights[&old_name]
        ));
        assert_eq!(
            changed.weights[name].to_vec2::<f32>().unwrap(),
            current.to_vec2::<f32>().unwrap()
        );
        assert_eq!(
            changed.weights[&old_name].to_vec2::<f32>().unwrap(),
            old.to_vec2::<f32>().unwrap()
        );
        policy.load_state_dict(&saved).unwrap();
        case.initial[name].compare(&current, "restored current head");
        case.initial[name].compare(&old, "restored original container head");
        assert_eq!(
            policy
                .optimizer
                .parameters()
                .iter()
                .map(|value| value.id())
                .collect::<Vec<_>>(),
            optimizer_ids
        );
    }
}

#[test]
fn constant_replacement_models_never_register_detached_checkpoint_variables() {
    use crate::rl_candle_vessel::CandlePpoVessel;
    for actor in [true, false] {
        let (features, variables, config) = setup(&super::cases().remove(0));
        let mut policy = CandlePpo::new(features.clone(), 3, &builder(&variables), config).unwrap();
        let saved = policy.state_dict().unwrap();
        let unchanged = variables["actor.extractor.scale"].clone();
        let before = unchanged.to_vec1::<f32>().unwrap();
        let zeros = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let expected_name = if actor {
            policy.actor = PpoActor::new(features, 3, zeros).unwrap();
            "actor.layer_out.0.weight"
        } else {
            policy.critic = PpoCritic::new(features, zeros).unwrap();
            "critic.value_out.weight"
        };
        let Err(error) = policy.policy_state(()) else {
            panic!("constant model must not register a detached mutable copy")
        };
        assert!(error.to_string().contains(expected_name));
        assert!(policy.state_dict().is_err());
        assert!(policy.load_state_dict(&saved).is_err());
        let mut loaded = saved.clone().into_policy_weights(None).unwrap();
        let keys: Vec<_> = loaded.weights.keys().cloned().collect();
        let ids: Vec<_> = loaded.weights.values().map(|value| value.id()).collect();
        assert!(matches!(
            set_policy_weights(&mut policy, &mut loaded),
            Err(PolicyWeightLoadError::Other(_))
        ));
        assert_eq!(loaded.weights.keys().cloned().collect::<Vec<_>>(), keys);
        assert_eq!(
            loaded
                .weights
                .values()
                .map(|value| value.id())
                .collect::<Vec<_>>(),
            ids
        );
        let mut runtime = CandlePpoVessel::new(policy, StdRng::seed_from_u64(11));
        assert!(runtime.state_dict().is_err());
        assert!(runtime.load_state_dict(&saved).is_err());
        let mut runner = crate::TrainingVesselRunner::<_, _, _, (), _, _, _>::with_policy(
            Box::new(runtime),
            Box::new(crate::rl_candle_collector::CandleCollectorFactory),
            crate::TrainingVesselBinding::default(),
            crate::TrainingVesselRunConfig::default(),
            crate::TrainingVesselLog::default(),
        );
        assert!(matches!(
            runner.state_dict(),
            Err(crate::TrainingVesselStateError::Save(_))
        ));
        assert!(crate::RlCheckpointState::save_checkpoint(&mut runner).is_err());
        let envelope = crate::TrainingVesselCheckpoint { policy: saved };
        assert!(matches!(
            runner.load_state_dict(&envelope),
            Err(crate::TrainingVesselStateError::Load(_))
        ));
        assert!(crate::RlCheckpointState::load_checkpoint(&mut runner, &envelope).is_err());
        assert_eq!(unchanged.to_vec1::<f32>().unwrap(), before);
    }
}

fn input(case: &WeightCase) -> PolicyWeights<Tensor, OpaqueMetadata> {
    PolicyWeights {
        weights: case
            .initial
            .iter()
            .map(|(name, record)| (name.clone(), Arc::new(record.tensor())))
            .collect(),
        metadata: OpaqueMetadata("retained without clone"),
    }
}

fn check_state(policy: &CandlePpo, expected: &WeightCase) {
    let state = policy.policy_state(()).unwrap();
    let variables = state.variables();
    assert_eq!(
        variables.keys().collect::<Vec<_>>(),
        expected.final_weights.keys().collect::<Vec<_>>()
    );
    let values: Vec<_> = variables.values().collect();
    for ((name, variable), alias) in variables.iter().zip(&expected.aliases) {
        expected.final_weights[name].compare(variable, name);
        assert_eq!(variable.id(), values[*alias].id());
    }
    let snapshot = state.snapshot().unwrap().into_policy_weights(None).unwrap();
    assert_eq!(
        snapshot.weights.keys().collect::<Vec<_>>(),
        variables.keys().collect::<Vec<_>>()
    );
    let owned: Vec<_> = snapshot.weights.values().collect();
    for (index, &alias) in expected.aliases.iter().enumerate() {
        assert!(Arc::ptr_eq(owned[index], owned[alias]));
    }
}

#[test]
fn complete_and_legacy_weights_match_actual_source_order_and_failure_mutations() {
    let cases = fixture();
    assert_eq!(cases.len(), 5);
    for case in cases {
        for constructor in [false, true] {
            let (features, variables, config) = setup(&super::cases().remove(0));
            let mut policy =
                CandlePpo::new(features.clone(), 3, &builder(&variables), config).unwrap();
            for variable in policy.optimizer.parameters() {
                variable
                    .set(&variable.ones_like().unwrap().affine(0.1, 0.).unwrap())
                    .unwrap();
            }
            let mut weights = input(&case);
            let outcome = if constructor {
                CandlePpo::new_with_weights(features, 3, &builder(&variables), config, &mut weights)
                    .map(|created| {
                        policy = created;
                    })
            } else {
                set_policy_weights(&mut policy, &mut weights).map_err(CandlePpoError::from)
            };
            assert_eq!(outcome.is_err(), case.error.is_some(), "{}", case.kind);
            if let Err(error) = outcome {
                assert!(matches!(
                    error,
                    CandlePpoError::Weights(PolicyWeightLoadError::Runtime(_))
                ));
            }
            assert_eq!(
                weights.weights.keys().cloned().collect::<Vec<_>>(),
                case.input_keys
            );
            assert_eq!(weights.metadata.0, "retained without clone");
            assert_eq!(policy.optimizer.initialized_parameter_count(), 0);
            assert!(!policy.is_updating());
            check_state(&policy, &case);
        }
    }
}

#[test]
fn constructor_configuration_errors_do_not_attempt_weight_retry() {
    let (features, variables, mut config) = setup(&super::cases().remove(0));
    config.gamma = -1.;
    let mut weights = input(&fixture().remove(1));
    let before: Vec<_> = weights.weights.keys().cloned().collect();
    let error =
        CandlePpo::new_with_weights(features, 3, &builder(&variables), config, &mut weights);
    assert!(matches!(error, Err(CandlePpoError::Configuration(_))));
    assert_eq!(weights.weights.keys().cloned().collect::<Vec<_>>(), before);
}

#[test]
fn constructor_restores_source_legacy_snapshot_without_optimizer_resume() {
    #[derive(Deserialize)]
    struct Constructor {
        events: Vec<String>,
        keys: Vec<String>,
        initialized: usize,
        #[serde(rename = "final")]
        final_weights: IndexMap<String, Record>,
    }
    #[derive(Deserialize)]
    struct Fixture {
        constructor: Constructor,
    }
    let fixture =
        serde_json::from_str::<Fixture>(include_str!("../fixtures/rl_candle_ppo_weights.json"))
            .unwrap()
            .constructor;
    assert_eq!(fixture.events, ["optimizer", "weights.native"]);
    let mut weights = PolicyWeights {
        weights: fixture
            .final_weights
            .iter()
            .filter(|(name, _)| !name.starts_with("_actor_critic."))
            .map(|(name, record)| (name.clone(), Arc::new(record.tensor())))
            .collect(),
        metadata: OpaqueMetadata("constructor metadata"),
    };
    let (features, variables, config) = setup(&super::cases().remove(0));
    let mut policy =
        CandlePpo::new_with_weights(features, 3, &builder(&variables), config, &mut weights)
            .unwrap();
    assert_eq!(
        policy.optimizer.initialized_parameter_count(),
        fixture.initialized
    );
    assert_eq!(policy.return_statistics.count, 0);
    assert_eq!(weights.metadata.0, "constructor metadata");
    assert_eq!(
        weights.weights.keys().cloned().collect::<Vec<_>>(),
        fixture.keys
    );
    let state = policy.policy_state(()).unwrap();
    for (name, variable) in state.variables() {
        fixture.final_weights[name].compare(variable, name);
    }
    let snapshot = state.snapshot().unwrap();
    let predictions = policy
        .actor
        .forward(&rollout(&super::cases().remove(0)).observations, ())
        .unwrap()
        .0;
    state.variables()["actor.layer_out.0.bias"]
        .set(&Tensor::new(&[1_f32, 0., -1.], &Device::Cpu).unwrap())
        .unwrap();
    let mut decoded = snapshot.into_policy_weights(None).unwrap();
    set_policy_weights(&mut policy, &mut decoded).unwrap();
    let restored = policy
        .actor
        .forward(&rollout(&super::cases().remove(0)).observations, ())
        .unwrap()
        .0;
    assert_eq!(
        restored.to_vec2::<f32>().unwrap(),
        predictions.to_vec2::<f32>().unwrap()
    );
}
