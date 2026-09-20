use super::*;
use serde::Deserialize;
use std::{path::PathBuf, process::Command};

#[derive(Deserialize)]
struct Case {
    names: Vec<String>,
    outcomes: Vec<String>,
    hook: String,
    calls: Vec<Vec<(String, usize)>>,
    r#final: Vec<(String, usize)>,
    loads: usize,
    error: Option<String>,
}

// Intentionally not Clone: aliases must retain the actual value and metadata.
struct Weight(usize);
struct Metadata(usize);

struct Loader<'a> {
    case: &'a Case,
    calls: Vec<Vec<(String, usize)>>,
    hook_weight: Arc<Weight>,
}

fn entries(state: &PolicyWeights<Weight, Metadata>) -> Vec<(String, usize)> {
    state
        .weights
        .iter()
        .map(|(name, value)| (name.clone(), value.0))
        .collect()
}

impl PolicyWeightLoader<Weight, Metadata> for Loader<'_> {
    fn load_weights(
        &mut self,
        state: &mut PolicyWeights<Weight, Metadata>,
    ) -> Result<(), PolicyWeightLoadError> {
        self.calls.push(entries(state));
        state.metadata.0 += 1;
        if self.calls.len() == 1 {
            match self.case.hook.as_str() {
                "insert" => {
                    state
                        .weights
                        .insert("hook".into(), Arc::clone(&self.hook_weight));
                }
                "remove" => {
                    state.weights.shift_remove_index(0);
                }
                _ => {}
            }
        }
        match self.case.outcomes[self.calls.len() - 1].as_str() {
            "runtime" => Err(PolicyWeightLoadError::Runtime("runtime".into())),
            "other" => Err(PolicyWeightLoadError::Other("other".into())),
            _ => Ok(()),
        }
    }
}

#[test]
fn retry_order_identity_metadata_and_failures_match_unchanged_qlib() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("../../../qlib/qlib/rl/order_execution/policy.py");
    let output = Command::new("python")
        .args([
            root.join("tests/fixtures/rl_policy_weight_contract.py"),
            source,
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Case> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 105);
    for case in cases {
        let pool = (0..=case.names.len())
            .map(|id| Arc::new(Weight(id)))
            .collect::<Vec<_>>();
        let mut state = PolicyWeights {
            weights: case
                .names
                .iter()
                .enumerate()
                .map(|(id, name)| (name.clone(), Arc::clone(&pool[id])))
                .collect(),
            metadata: Metadata(0),
        };
        let mut loader = Loader {
            case: &case,
            calls: Vec::new(),
            hook_weight: Arc::clone(pool.last().unwrap()),
        };
        let result = set_policy_weights(&mut loader, &mut state);
        assert_eq!(result.err().map(|error| error.to_string()), case.error);
        assert_eq!(loader.calls, case.calls);
        assert_eq!(entries(&state), case.r#final);
        assert_eq!(state.metadata.0, case.loads);
        for value in state.weights.values() {
            assert!(Arc::ptr_eq(value, &pool[value.0]));
        }
    }
}
