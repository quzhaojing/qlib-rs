use super::*;
use domain_core::decision_construction::{
    DecisionOrderItem, SharedDecisionOrders, SharedOrderDecisionConstruction,
};
use domain_core::decision_update::{
    LiveDecisionHandle, SharedDecisionUpdateStrategy, SharedLiveDecision,
};
use domain_core::saoe_live_generation::LiveSaoeGeneration;
use domain_core::{SaoeObservation, SaoeStateInterpreter};
use std::sync::RwLock;

#[path = "saoe_live_generation_failures.rs"]
mod failures;

#[path = "saoe_live_initialization.rs"]
mod initialization;

struct Origin;
impl SharedDecisionUpdateStrategy<()> for Origin {
    fn update_trade_decision(
        &self,
        _: &SharedLiveDecision<Self, ()>,
        _: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<Option<SharedLiveDecision<Self, ()>>, domain_core::DecisionUpdateStrategyError>
    {
        Ok(None)
    }
}

fn orders(names: &[&str]) -> SharedDecisionOrders {
    Arc::new(RwLock::new(
        names
            .iter()
            .map(|name| {
                DecisionOrderItem::Order(Arc::new(RwLock::new(Order::new(
                    *name,
                    1.0,
                    OrderDir::Buy,
                    None,
                    None,
                ))))
            })
            .collect(),
    ))
}

struct Rig {
    mode: &'static str,
    outer: SharedLiveDecision<Origin, ()>,
    events: Mutex<Vec<String>>,
}
impl Rig {
    fn list(&self) -> SharedDecisionOrders {
        self.outer
            .try_read()
            .unwrap()
            .orders
            .as_ref()
            .unwrap()
            .clone()
    }
    fn replace(&self, names: &[&str]) {
        self.outer.try_write().unwrap().orders = Some(orders(names));
    }
    fn event(&self, value: String) {
        self.events.lock().unwrap().push(value);
    }
}
struct Plugin(Arc<Rig>);
impl SaoeStateInterpreter for Plugin {
    fn interpret(&self, state: &SaoeState) -> Result<SaoeObservation, SaoeInterpreterError> {
        let name = state.parts().order.stock_id();
        self.0.event(format!("observe:{name}"));
        if self.0.mode == "delete" && name == "A" {
            self.0.list().try_write().unwrap().remove(1);
        }
        if self.0.mode == "failure" && name == "B" {
            return Err(SaoeInterpreterError::ProcessedDataPlugin(
                "observation failed".to_owned(),
            ));
        }
        Ok(SaoeObservation::Dummy {
            dummy: match name {
                "A" => 0,
                "B" => 1,
                "C" => 2,
                _ => panic!("state"),
            },
        })
    }

    fn interpret_live(
        &self,
        state: &LiveSaoeState,
    ) -> Result<SaoeObservation, SaoeInterpreterError> {
        SaoeStateInterpreter::interpret(
            self,
            &state
                .snapshot()
                .map_err(|error| SaoeInterpreterError::LiveState(error.to_string()))?,
        )
    }
}
impl SaoePolicy for Plugin {
    fn actions(
        &mut self,
        observations: &[SaoeObservation],
    ) -> Result<Vec<SaoePolicyAction>, SaoeInterpreterError> {
        let names: Vec<_> = observations
            .iter()
            .map(|o| match o {
                SaoeObservation::Dummy { dummy } => {
                    ["A", "B", "C"][usize::try_from(*dummy).unwrap()]
                }
                _ => panic!("obs"),
            })
            .collect();
        self.0.event(format!("policy:{}", names.join(",")));
        if self.0.mode == "policy-failure" {
            return Err(SaoeInterpreterError::ProcessedDataPlugin(
                "policy failed".to_owned(),
            ));
        }
        if self.0.mode == "policy" {
            self.0.replace(&["X", "Y"]);
        }
        self.0.mutate_after_policy();
        let count = match self.0.mode {
            "short" => observations.len() - 1,
            "long" => observations.len() + 1,
            _ => observations.len(),
        };
        Ok((1..=count)
            .map(|i| {
                SaoePolicyAction::Continuous(
                    if self.0.mode == "zero" || (self.0.mode == "invalid-zero" && i == 1) {
                        0.0
                    } else {
                        f64::from(u32::try_from(i).unwrap())
                    },
                )
            })
            .collect())
    }
}
impl SaoeActionInterpreter for Plugin {
    fn action_space(&self) -> SaoeActionSpace {
        SaoeActionSpace::NonNegativeContinuous
    }
    fn interpret(
        &self,
        state: &SaoeState,
        action: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError> {
        let SaoePolicyAction::Continuous(value) = action else {
            panic!("action")
        };
        self.0
            .event(format!("action:{}:{value}", state.parts().order.stock_id()));
        if self.0.mode == "action-failure" {
            return Err(SaoeInterpreterError::ProcessedDataPlugin(
                "action failed".to_owned(),
            ));
        }
        Ok(value)
    }

    fn interpret_live(
        &self,
        state: &LiveSaoeState,
        action: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError> {
        SaoeActionInterpreter::interpret(
            self,
            &state
                .snapshot()
                .map_err(|error| SaoeInterpreterError::LiveState(error.to_string()))?,
            action,
        )
    }
}
impl SaoeOrderFactory for Plugin {
    fn create(
        &mut self,
        name: &str,
        amount: Option<f64>,
        direction: OrderDir,
    ) -> Result<Order, SaoePluginError> {
        self.0.event(format!("create:{name}"));
        if self.0.mode == "create-failure" {
            return Err(plugin_error("create"));
        }
        if self.0.mode == "factory" && name == "A" {
            self.0.replace(&["D", "E"]);
        }
        if name == "B" {
            self.0.mutate_after_children();
        }
        Ok(Order::new(name, amount.unwrap(), direction, None, None))
    }
}
impl SaoeCalendar for Plugin {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        Ok((0, 1))
    }
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        self.0.event("time".to_owned());
        if self.0.mode == "time-failure" {
            return Err(plugin_error("time"));
        }
        if self.0.mode == "details" {
            self.0.list().try_write().unwrap()[1] = orders(&["E"]).write().unwrap().remove(0);
        }
        Ok((time(0), time(1)))
    }
}
impl SaoeDecisionCalendar for Plugin {
    fn frequency(&self) -> Result<String, SaoePluginError> {
        self.0.event("freq".to_owned());
        if self.0.mode == "frequency-failure" {
            return Err(plugin_error("frequency"));
        }
        Ok("1min".to_owned())
    }
}

fn python_oracle() -> Value {
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
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn live_generation_matches_source_list_replacement_and_callback_mutations() {
    let oracle = python_oracle();
    for mode in [
        "normal", "append", "delete", "policy", "factory", "details", "short", "long", "failure",
    ] {
        let (rig, mut builder, handle) = failures::fixture(mode);
        let result = failures::generate(&rig, &mut builder, &handle);
        let expected: Vec<_> = oracle[mode]["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|event| *event != "read" && *event != "construct")
            .cloned()
            .collect();
        assert_eq!(
            json!(*rig.events.lock().unwrap()),
            json!(expected),
            "{mode}"
        );
        if mode == "failure" {
            assert!(
                result
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("observation failed")
            );
            continue;
        }
        let parts = result.unwrap();
        let children: Vec<_> = parts
            .orders
            .read()
            .unwrap()
            .iter()
            .map(|item| {
                let DecisionOrderItem::Order(order) = item else {
                    panic!("child")
                };
                let order = order.read().unwrap();
                assert!(order.start_time().is_none());
                json!([order.stock_id(), order.amount(), 1])
            })
            .collect();
        // JSON numeric representation distinguishes integers/floats; compare scalar values.
        let source_children = oracle[mode]["children"].as_array().unwrap();
        assert_eq!(children.len(), source_children.len());
        for (child, source) in children.iter().zip(source_children) {
            assert_eq!(child[0], source[0]);
            assert_eq!(child[1].as_f64(), source[1].as_f64());
            assert_eq!(child[2], source[2]);
        }
        assert_eq!(
            json!(
                parts
                    .details
                    .iter()
                    .map(|row| &row.instrument)
                    .collect::<Vec<_>>()
            ),
            oracle[mode]["details"]
        );
    }
}
