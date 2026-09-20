use super::*;

struct Rig {
    mode: &'static str,
    names: Mutex<Vec<i64>>,
    events: Mutex<Vec<String>>,
}

fn name(id: i64) -> &'static str {
    match id {
        0 => "A",
        1 => "B",
        2 => "C",
        _ => panic!("unexpected state"),
    }
}

impl Rig {
    fn record(&self, event: String) -> Result<(), SaoeInterpreterError> {
        self.events.lock().unwrap().push(event.clone());
        if event == self.mode || (self.mode == "failure" && event == "observe:B") {
            return Err(SaoeInterpreterError::ProcessedDataPlugin(event));
        }
        Ok(())
    }
}

struct Observe(Arc<Rig>);
impl SaoeStateInterpreter for Observe {
    fn interpret(&self, state: &SaoeState) -> Result<SaoeObservation, SaoeInterpreterError> {
        let id = state.parts().cur_step;
        self.0.record(format!("observe:{}", name(id)))?;
        if self.0.mode == "delete" && id == 0 {
            self.0.names.try_lock().unwrap().remove(1);
        }
        Ok(SaoeObservation::Dummy {
            dummy: i32::try_from(id).unwrap(),
        })
    }
}

struct Policy(Arc<Rig>);
impl SaoePolicy for Policy {
    fn actions(
        &mut self,
        observations: &[SaoeObservation],
    ) -> Result<Vec<SaoePolicyAction>, SaoeInterpreterError> {
        let names: Vec<_> = observations
            .iter()
            .map(|obs| {
                let SaoeObservation::Dummy { dummy } = obs else {
                    panic!("unexpected observation")
                };
                name(i64::from(*dummy))
            })
            .collect();
        self.0.record(format!("policy:{}", names.join(",")))?;
        let count = match self.0.mode {
            "short" => observations.len().saturating_sub(1),
            "long" => observations.len() + 1,
            _ => observations.len(),
        };
        Ok((1..=count)
            .map(|value| SaoePolicyAction::Discrete(i64::try_from(value).unwrap()))
            .collect())
    }
}

struct Action(Arc<Rig>);
impl SaoeActionInterpreter for Action {
    fn action_space(&self) -> SaoeActionSpace {
        SaoeActionSpace::Discrete { size: 4 }
    }
    fn interpret(
        &self,
        state: &SaoeState,
        action: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError> {
        let SaoePolicyAction::Discrete(value) = action else {
            panic!("unexpected action")
        };
        self.0
            .record(format!("action:{}:{value}", name(state.parts().cur_step)))?;
        Ok(f64::from(i32::try_from(value).unwrap()))
    }
}

fn run(
    mode: &'static str,
) -> (
    Result<Vec<domain_core::SaoePolicyDecision>, SaoeInterpreterError>,
    Vec<String>,
) {
    let rig = Arc::new(Rig {
        mode,
        names: Mutex::new(match mode {
            "empty" => vec![],
            "delete" => vec![0, 1, 2],
            _ => vec![0, 1],
        }),
        events: Mutex::new(Vec::new()),
    });
    let mut pipeline = SaoePolicyPipeline::new(
        Box::new(Observe(Arc::clone(&rig))),
        Box::new(Policy(Arc::clone(&rig))),
        Box::new(Action(Arc::clone(&rig))),
    );
    let mut index = 0;
    let result = pipeline.decisions_from(|| {
        let Some(id) = rig.names.try_lock().unwrap().get(index).copied() else {
            return Ok(None);
        };
        index += 1;
        rig.record(format!("state:{}", name(id)))?;
        if mode == "append" && id == 0 {
            rig.names.try_lock().unwrap().push(2);
        }
        Ok(Some(state(OrderDir::Buy, id, 1.0, 3, 1)))
    });
    let events = rig.events.lock().unwrap().clone();
    (result, events)
}

#[test]
fn pull_pipeline_matches_actual_python_callback_order_and_zip_lengths() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_live_orders_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\strategy.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Value = serde_json::from_slice(&output.stdout).unwrap();
    for mode in ["normal", "append", "delete", "short", "long", "failure"] {
        let (result, events) = run(mode);
        let source = oracle[mode]["events"].as_array().unwrap();
        let expected: Vec<_> = source
            .iter()
            .skip(1)
            .take_while(|event| *event != "read")
            .cloned()
            .collect();
        assert_eq!(json!(events), json!(expected), "{mode}");
        if mode == "failure" {
            assert_eq!(
                result,
                Err(SaoeInterpreterError::ProcessedDataPlugin(
                    "observe:B".to_owned()
                ))
            );
        } else {
            let decisions = result.unwrap();
            let children = oracle[mode]["children"].as_array().unwrap();
            assert_eq!(decisions.len(), children.len());
            for (decision, child) in decisions.iter().zip(children) {
                assert_eq!(
                    decision.action,
                    SaoePolicyAction::Discrete(child[1].as_i64().unwrap())
                );
                exact(decision.execution_volume, child[1].as_f64().unwrap());
            }
        }
    }
}

#[test]
fn pull_pipeline_stops_at_each_failure_and_still_calls_policy_for_empty_input() {
    let (_, baseline) = run("normal");
    for failure in ["state:B", "observe:B", "policy:A,B", "action:B:2"] {
        let (result, events) = run(failure);
        assert_eq!(
            result,
            Err(SaoeInterpreterError::ProcessedDataPlugin(
                failure.to_owned()
            ))
        );
        let end = baseline.iter().position(|event| event == failure).unwrap() + 1;
        assert_eq!(events, baseline[..end]);
    }
    let (result, events) = run("empty");
    assert!(result.unwrap().is_empty());
    assert_eq!(events, ["policy:"]);
}
