use super::*;
use domain_core::decision_construction::{
    ConstructedDecisionBase, DecisionOrderItem, SharedDecisionOrders,
    SharedOrderDecisionConstruction,
};
use domain_core::decision_update::{
    LiveDecisionHandle, SharedDecisionUpdateStrategy, SharedLiveDecision,
};
use domain_core::nested_executor::SharedNestedResult;
use domain_core::saoe_live_registry::{
    LiveSaoeAdapterFactory, LiveSaoeAdapterRegistry, LiveSaoeRegistryError as Error,
    LiveSaoeStateAdapter,
};
use domain_core::shared_simulator::SharedSimulatorExecution;
use domain_core::{IdxTradeRange, SharedTradeRange};
use std::sync::RwLock;

#[path = "saoe_live_registry_failures.rs"]
mod failures;

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

fn order(name: &str) -> Arc<RwLock<Order>> {
    Arc::new(RwLock::new(Order::new(
        name,
        1.0,
        OrderDir::Buy,
        Some(time(0)),
        Some(time(1)),
    )))
}
fn list(names: &[&str]) -> SharedDecisionOrders {
    Arc::new(RwLock::new(
        names
            .iter()
            .map(|name| DecisionOrderItem::Order(order(name)))
            .collect(),
    ))
}
struct Rig {
    mode: &'static str,
    outer: SharedLiveDecision<Origin, ()>,
    range: SharedTradeRange,
    events: Mutex<Vec<Value>>,
    count: Mutex<usize>,
    rows: SharedNestedResult,
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
    fn event(&self, event: Value) {
        self.events.lock().unwrap().push(event);
    }
}
struct Factory(Arc<Rig>);
struct Adapter {
    rig: Arc<Rig>,
    order: Arc<RwLock<Order>>,
    label: String,
}

impl LiveSaoeAdapterFactory for Factory {
    fn create(
        &mut self,
        item: &Arc<RwLock<Order>>,
        outer: &LiveDecisionHandle,
        range: &SharedTradeRange,
    ) -> Result<Box<dyn LiveSaoeStateAdapter>, SaoePluginError> {
        let original: LiveDecisionHandle = self.0.outer.clone();
        assert!(Arc::ptr_eq(outer, &original));
        assert!(Arc::ptr_eq(range, &self.0.range));
        assert!(item.try_write().is_ok());
        assert!(self.0.list().try_write().is_ok());
        let name = item.read().unwrap().stock_id().to_owned();
        let index = *self.0.count.lock().unwrap();
        let label = format!("{name}:{index}");
        self.0.event(json!(format!("create:{label}")));
        if self.0.mode == "factory-failure" && name == "B" {
            return Err(plugin_error("factory failed"));
        }
        *self.0.count.lock().unwrap() += 1;
        if self.0.mode == "key-failure" && name == "B" {
            *item.write().unwrap() = Order::new("B", 1.0, OrderDir::Buy, None, None);
        }
        if index == 0 {
            match self.0.mode {
                "late-key" => *item.write().unwrap() = order("X").read().unwrap().clone(),
                "append" => self
                    .0
                    .list()
                    .write()
                    .unwrap()
                    .push(DecisionOrderItem::Order(order("C"))),
                "delete" => {
                    self.0.list().write().unwrap().remove(1);
                }
                "replace" => self.0.outer.write().unwrap().orders = Some(list(&["D"])),
                "list-poison" => failures::poison(self.0.list()),
                "key-poison" => failures::poison(item.clone()),
                _ => (),
            }
            self.0
                .outer
                .try_write()
                .unwrap()
                .base
                .as_mut()
                .unwrap()
                .trade_range = Some(Arc::new(IdxTradeRange::new(8, 9)));
        }
        Ok(Box::new(Adapter {
            rig: self.0.clone(),
            order: item.clone(),
            label,
        }))
    }
}
impl LiveSaoeStateAdapter for Adapter {
    fn state(&self) -> Result<SaoeState, SaoePluginError> {
        assert!(self.order.try_write().is_ok());
        if self.rig.mode == "state-failure" {
            return Err(plugin_error("state failed"));
        }
        Ok(state_for(&self.order.read().unwrap()))
    }
    fn update(
        &mut self,
        rows: &[SharedOrderExecution],
        range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        assert_eq!(range, (2, 5));
        let original = self.rig.rows.try_lock().unwrap();
        for row in rows {
            assert!(original.iter().any(|item| Arc::ptr_eq(item, row)));
        }
        drop(original);
        let name = self.order.read().unwrap().stock_id().to_owned();
        let values: Vec<_> = rows
            .iter()
            .map(|row| json!([row.order.read().unwrap().stock_id(), row.trade_value]))
            .collect();
        self.rig.event(json!(["update", name, values]));
        if self.rig.mode == "mutate" && name == "A" {
            *self.rig.rows.lock().unwrap()[0].order.write().unwrap() =
                order("Z").read().unwrap().clone();
        }
        if self.rig.mode == "update-failure" && name == "B" {
            return Err(plugin_error("update failed"));
        }
        Ok(())
    }
    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        self.rig.event(json!(format!("final:{}", self.label)));
        if self.rig.mode == "final-failure" && self.label.starts_with('B') {
            return Err(plugin_error("final failed"));
        }
        Ok(())
    }
}

fn fixture(
    mode: &'static str,
    names: &[&str],
) -> (Arc<Rig>, LiveSaoeAdapterRegistry, LiveDecisionHandle) {
    let range: SharedTradeRange = Arc::new(IdxTradeRange::new(0, 9));
    let mut outer = SharedOrderDecisionConstruction::new(Arc::new(Origin));
    outer.orders = Some(list(names));
    outer.base = Some(ConstructedDecisionBase {
        start_time: time(0),
        end_time: time(1),
        trade_range: Some(range.clone()),
    });
    let rig = Arc::new(Rig {
        mode,
        outer: Arc::new(RwLock::new(outer)),
        range,
        events: Mutex::new(Vec::new()),
        count: Mutex::new(0),
        rows: Arc::new(Mutex::new(Vec::new())),
    });
    let handle: LiveDecisionHandle = rig.outer.clone();
    let registry = LiveSaoeAdapterRegistry::new(Box::new(Factory(rig.clone())));
    (rig, registry, handle)
}

fn oracle() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/saoe_lifecycle_contract.py"
            ),
            r"D:\code\github\qlib\qlib\rl\order_execution\strategy.py",
            r"D:\code\github\qlib\qlib\strategy\base.py",
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
fn live_registry_reset_matches_source_mutations_duplicates_and_partial_insertions() {
    let source = oracle();
    for mode in [
        "normal",
        "late-key",
        "append",
        "delete",
        "replace",
        "duplicate",
        "factory-failure",
        "key-failure",
    ] {
        let names = if mode == "duplicate" {
            vec!["A", "B", "A"]
        } else {
            vec!["A", "B"]
        };
        let (rig, mut registry, outer) = fixture(mode, &names);
        assert!(registry.is_empty());
        let result = registry.reset(Some(&outer));
        match mode {
            "factory-failure" => assert!(matches!(result, Err(Error::Factory(_)))),
            "key-failure" => assert!(matches!(result, Err(Error::Key(_)))),
            _ => result.unwrap(),
        }
        let expected = &source["reset"][mode];
        let creates: Vec<_> = expected["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|event| event.as_str().unwrap().starts_with("create:"))
            .cloned()
            .collect();
        assert_eq!(*rig.events.lock().unwrap(), creates, "{mode}");
        let entries = expected["entries"].as_array().unwrap();
        assert_eq!(registry.len(), entries.len());
        for entry in entries {
            let name = entry[0].as_str().unwrap();
            assert_eq!(
                registry
                    .state(&order(name))
                    .unwrap()
                    .parts()
                    .order
                    .stock_id(),
                name
            );
        }
        rig.events.lock().unwrap().clear();
        registry.finalize().unwrap();
        let finals: Vec<_> = entries
            .iter()
            .map(|entry| json!(format!("final:{}", entry[1].as_str().unwrap())))
            .collect();
        assert_eq!(*rig.events.lock().unwrap(), finals, "{mode}");
        registry.reset(None).unwrap();
        assert!(registry.is_empty());
    }
}

#[test]
fn live_registry_post_matches_source_grouping_aliases_and_fail_stop() {
    let source = oracle();
    for mode in [
        "normal",
        "none",
        "zero-empty",
        "zero-rows",
        "mutate",
        "update-failure",
        "final-failure",
    ] {
        let (rig, mut registry, outer) = fixture(mode, &["A", "B", "C"]);
        registry.reset(Some(&outer)).unwrap();
        if mode != "zero-empty" {
            for (name, value) in [("B", 11.0), ("A", 12.0), ("B", 13.0), ("X", 14.0)] {
                rig.rows
                    .lock()
                    .unwrap()
                    .push(Arc::new(SharedSimulatorExecution {
                        order: order(name),
                        trade_value: value,
                        trade_cost: 0.0,
                        trade_price: 1.0,
                    }));
            }
        }
        rig.events.lock().unwrap().clear();
        let rows = if mode == "none" {
            None
        } else {
            Some(&rig.rows)
        };
        let range = if mode.starts_with("zero") {
            (4, 4)
        } else {
            (2, 5)
        };
        let result = registry
            .update(rows, range)
            .and_then(|()| registry.finalize());
        match mode {
            "zero-rows" => assert!(matches!(result, Err(Error::UnexpectedExecutions))),
            "update-failure" => assert!(matches!(result, Err(Error::Update(_)))),
            "final-failure" => assert!(matches!(result, Err(Error::Finalize(_)))),
            _ => result.unwrap(),
        }
        let expected: Vec<_> = source["post"][mode]["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|event| !event.as_str().is_some_and(|s| s.starts_with("key:")))
            .cloned()
            .collect();
        let actual: Vec<_> = rig
            .events
            .lock()
            .unwrap()
            .iter()
            .map(|event| {
                if let Some(text) = event.as_str() {
                    json!(text.rsplit_once(':').unwrap().0)
                } else {
                    event.clone()
                }
            })
            .collect();
        // JSON numeric representation is not the contract: compare decoded numeric values.
        let normalize = |events: Vec<Value>| {
            events
                .into_iter()
                .map(|mut event| {
                    if let Some(items) = event.as_array_mut() {
                        for row in items[2].as_array_mut().unwrap() {
                            row[1] = json!(row[1].as_f64().unwrap());
                        }
                    }
                    event
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(normalize(actual), normalize(expected), "{mode}");
    }
}

#[test]
fn live_registry_empty_missing_range_and_state_lookup_boundaries() {
    let (rig, mut registry, outer) = fixture("normal", &["A"]);
    assert!(matches!(
        registry.state(&order("A")),
        Err(Error::MissingAdapter(..))
    ));
    assert!(matches!(
        registry.live_state(&order("A")),
        Err(Error::MissingAdapter(..))
    ));
    rig.outer
        .write()
        .unwrap()
        .base
        .as_mut()
        .unwrap()
        .trade_range = None;
    assert!(matches!(
        registry.reset(Some(&outer)),
        Err(Error::MissingRange)
    ));
    assert!(registry.is_empty());
    assert!(rig.events.lock().unwrap().is_empty());
    rig.outer.write().unwrap().orders = Some(list(&[]));
    registry.reset(Some(&outer)).unwrap();
    registry.update(None, (i64::MAX, i64::MIN)).unwrap();
    registry.finalize().unwrap();
    assert!(registry.is_empty());
    let (_, mut registry, outer) = fixture("state-failure", &["A"]);
    registry.reset(Some(&outer)).unwrap();
    assert!(matches!(registry.state(&order("A")), Err(Error::State(_))));
    let (_, mut registry, outer) = fixture("normal", &["A"]);
    registry.reset(Some(&outer)).unwrap();
    assert!(matches!(
        registry.live_state(&order("A")),
        Err(Error::State(error)) if error.message.contains("aliases are not supported")
    ));
    let poisoned = order("A");
    failures::poison(Arc::clone(&poisoned));
    assert!(matches!(
        registry.live_state(&poisoned),
        Err(Error::OrderPoisoned)
    ));
}
