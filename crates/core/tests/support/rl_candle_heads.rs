use super::*;

use crate::rl_candle_network_fixture as fixture;

#[test]
fn attention_forward_and_gradients_match_real_torch() {
    let cases = fixture::cases("attention");
    assert_eq!(cases.len(), 11);
    for case in cases {
        let weights = case.weights();
        let model = Attention::new(
            3,
            serde_json::from_value(case.config["output_dim"].clone()).unwrap(),
            fixture::builder(&weights),
        )
        .unwrap();
        assert_eq!(
            model.parameters().keys().collect::<Vec<_>>(),
            weights.keys().collect::<Vec<_>>()
        );
        let inputs = case.inputs();
        let q = linear_forward(&model.query, &inputs["q"]).unwrap();
        let k = linear_forward(&model.key, &inputs["k"]).unwrap();
        let v = linear_forward(&model.value, &inputs["v"]).unwrap();
        let scores = contract(&q, &k.transpose(1, 2).unwrap()).unwrap();
        let probabilities = softmax_last(&scores).unwrap();
        for stage in [
            "probabilities",
            "scores",
            "q_projected",
            "k_projected",
            "v_projected",
        ] {
            let leaf = candle_core::Var::from_tensor(&case.outputs[stage].tensor()).unwrap();
            let staged_v = if stage == "v_projected" {
                leaf.as_tensor().clone()
            } else {
                v.clone()
            };
            let staged_probabilities = match stage {
                "probabilities" => leaf.as_tensor().clone(),
                "scores" => softmax_last(&leaf).unwrap(),
                "q_projected" => {
                    softmax_last(&contract(&leaf, &k.transpose(1, 2).unwrap()).unwrap()).unwrap()
                }
                "k_projected" => {
                    softmax_last(&contract(&q, &leaf.transpose(1, 2).unwrap()).unwrap()).unwrap()
                }
                _ => probabilities.clone(),
            };
            let staged_output = contract(&staged_probabilities, &staged_v).unwrap();
            let gradients = staged_output
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .backward()
                .unwrap();
            case.intermediate_gradients[stage].compare(
                gradients.get(&leaf).unwrap(),
                &format!("{} intermediate gradient {stage}", case.name),
            );
        }
        for (name, tensor) in [
            ("q_projected", q),
            ("k_projected", k),
            ("v_projected", v),
            ("scores", scores),
            ("probabilities", probabilities),
        ] {
            case.outputs[name].compare(&tensor, &format!("{} {name}", case.name));
        }
        let output = model
            .forward(&inputs["q"], &inputs["k"], &inputs["v"])
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        case.outputs["attention"].compare(&output, &case.name);
        let gradients = output
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap_or_else(|error| panic!("{} backward: {error}", case.name));
        case.check_gradients(&gradients, &weights, &inputs);
    }
}

#[test]
fn attention_invalid_dimensions_and_parameter_sources_fail() {
    use candle_core::Device;
    use std::collections::HashMap;
    let device = &Device::Cpu;
    assert_eq!(RoundedSoftmax.name(), "source-rounded-softmax");
    assert!(
        Attention::new(
            3,
            2,
            VarBuilder::from_tensors(HashMap::new(), DType::F32, device)
        )
        .is_err()
    );
    let model = Attention::new(3, 2, VarBuilder::zeros(DType::F32, device)).unwrap();
    let good = Tensor::zeros((2, 4, 3), DType::F32, device).unwrap();
    for shape in [vec![3], vec![2, 3], vec![2, 4, 4], vec![3, 4, 3]] {
        let bad = Tensor::zeros(shape, DType::F32, device).unwrap();
        assert!(model.forward(&bad, &good, &good).is_err());
    }
    let incompatible = Tensor::zeros((2, 5, 3), DType::F32, device).unwrap();
    assert!(model.forward(&good, &good, &incompatible).is_err());
    assert!(
        contract(
            &Tensor::zeros((2, 3, 2), DType::F32, device).unwrap(),
            &Tensor::zeros((2, 2, 4), DType::F64, device).unwrap()
        )
        .is_err()
    );
    assert!(contract(&good, &Tensor::zeros((3, 4), DType::F32, device).unwrap()).is_err());
    assert_eq!(broadcast_extent(2, 2).unwrap(), 2);
    assert_eq!(broadcast_extent(2, 1).unwrap(), 2);
    assert_eq!(broadcast_extent(1, 2).unwrap(), 2);
    assert_eq!(broadcast_extent(0, 1).unwrap(), 0);
    assert_eq!(broadcast_extent(1, 0).unwrap(), 0);
    assert!(broadcast_extent(2, 3).is_err());
}

struct TestExtractor {
    parameters: IndexMap<String, Tensor>,
    fail: bool,
}
impl CandleFeatureExtractor for TestExtractor {
    fn output_dim(&self) -> usize {
        2
    }
    fn parameters(&self) -> &IndexMap<String, Tensor> {
        &self.parameters
    }
    fn forward(&self, obs: &RecurrentObservation) -> Result<Tensor> {
        if self.fail {
            Err(Error::Msg("extractor failure".into()))
        } else {
            Ok(obs.data_processed.clone())
        }
    }
}

#[test]
fn heads_preserve_plugin_failures_empty_actions_and_caller_state() {
    use candle_core::Device;
    use std::collections::HashMap;
    let device = &Device::Cpu;
    let empty = Tensor::zeros(0, DType::F32, device).unwrap();
    let obs = RecurrentObservation {
        data_processed: Tensor::ones((3, 2), DType::F32, device).unwrap(),
        cur_tick: empty.clone(),
        cur_step: empty.clone(),
        position_history: empty.clone(),
        target: empty.clone(),
        num_step: empty.clone(),
        acquiring: empty,
    };
    let missing = || VarBuilder::from_tensors(HashMap::new(), DType::F32, device);
    let extractor = Arc::new(TestExtractor {
        parameters: IndexMap::new(),
        fail: false,
    });
    assert!(PpoActor::new(extractor.clone(), 3, missing()).is_err());
    assert!(PpoCritic::new(extractor.clone(), missing()).is_err());
    let actor: DqnModel =
        PpoActor::new(extractor.clone(), 0, VarBuilder::zeros(DType::F32, device)).unwrap();
    let (output, state) = actor.forward(&obs, String::from("state")).unwrap();
    assert_eq!(output.dims(), [3, 0]);
    assert_eq!(state, "state");
    let critic = PpoCritic::new(extractor, VarBuilder::zeros(DType::F32, device)).unwrap();
    assert_eq!(critic.forward(&obs).unwrap().dims(), [3]);
    let failing = Arc::new(TestExtractor {
        parameters: IndexMap::new(),
        fail: true,
    });
    let actor = PpoActor::new(failing.clone(), 2, VarBuilder::zeros(DType::F32, device)).unwrap();
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
    assert!(
        actor
            .policy_forward(&obs, (), TrainingPolicyMode::Evaluation, true, &mut rng)
            .err()
            .unwrap()
            .to_string()
            .contains("extractor failure")
    );
    assert!(
        actor
            .forward(&obs, ())
            .unwrap_err()
            .to_string()
            .contains("extractor failure")
    );
    let critic = PpoCritic::new(failing, VarBuilder::zeros(DType::F32, device)).unwrap();
    assert!(
        critic
            .forward(&obs)
            .unwrap_err()
            .to_string()
            .contains("extractor failure")
    );
}

#[test]
fn each_attention_projection_requires_weight_and_bias() {
    let case = fixture::cases("attention").remove(0);
    let weights = case.weights();
    for missing in weights.keys() {
        let mut incomplete = weights.clone();
        incomplete.shift_remove(missing);
        let error = Attention::new(3, 2, fixture::builder(&incomplete))
            .err()
            .unwrap();
        assert!(
            error.to_string().contains(missing),
            "missing {missing}: {error}"
        );
    }
}
