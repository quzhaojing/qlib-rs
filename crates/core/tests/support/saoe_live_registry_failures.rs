use super::*;
use domain_core::decision_construction::DecisionAccessError;
use domain_core::decision_update::LiveDecisionAccessError;

struct FailingOrders {
    inner: LiveDecisionHandle,
    events: Mutex<Vec<&'static str>>,
}

impl domain_core::decision_update::LiveDecision for FailingOrders {
    fn total_step(
        &self,
    ) -> Result<domain_core::decision_construction::DecisionTotalStep, LiveDecisionAccessError>
    {
        self.inner.total_step()
    }
    fn base(&self) -> Result<ConstructedDecisionBase, LiveDecisionAccessError> {
        self.events.lock().unwrap().push("base");
        self.inner.base()
    }
    fn inherit_range(
        &self,
        range: Option<SharedTradeRange>,
    ) -> Result<(), LiveDecisionAccessError> {
        self.inner.inherit_range(range)
    }
    fn orders(&self) -> Result<SharedDecisionOrders, LiveDecisionAccessError> {
        self.events.lock().unwrap().push("orders");
        Err(DecisionAccessError::MissingOrders.into())
    }
    fn is_empty(&self) -> Result<bool, LiveDecisionAccessError> {
        self.events.lock().unwrap().push("empty");
        self.inner.is_empty()
    }
    fn update(
        self: Arc<Self>,
        calendar: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<Option<LiveDecisionHandle>, domain_core::decision_update::SharedDecisionUpdateError>
    {
        self.inner.clone().update(calendar)
    }
}

#[test]
fn registry_propagates_order_access_failure_after_successful_metadata_reads() {
    let (rig, mut registry, inner) = fixture("normal", &["A"]);
    let outer = Arc::new(FailingOrders {
        inner,
        events: Mutex::new(Vec::new()),
    });
    let handle: LiveDecisionHandle = outer.clone();
    assert!(matches!(
        registry.reset(Some(&handle)),
        Err(Error::Decision(LiveDecisionAccessError::Access(
            DecisionAccessError::MissingOrders
        )))
    ));
    assert_eq!(*outer.events.lock().unwrap(), ["empty", "base", "orders"]);
    assert!(registry.is_empty());
    assert!(rig.events.lock().unwrap().is_empty());
}

pub(super) fn poison<T: Send + Sync + 'static>(value: Arc<RwLock<T>>) {
    assert!(
        std::thread::spawn(move || {
            let _guard = value.try_write().unwrap();
            panic!("deliberate test poison");
        })
        .join()
        .is_err()
    );
}

#[test]
fn registry_reset_errors_keep_only_reached_insertions() {
    for mode in [
        "decision-poison",
        "missing-base",
        "missing-orders",
        "invalid",
        "list-poison",
        "key-poison",
    ] {
        let (rig, mut registry, outer) = fixture(mode, &["A", "B"]);
        match mode {
            "decision-poison" => poison(rig.outer.clone()),
            "missing-base" => rig.outer.write().unwrap().base = None,
            "missing-orders" => rig.outer.write().unwrap().orders = None,
            "invalid" => rig.list().write().unwrap()[1] = DecisionOrderItem::Other(Arc::new(7)),
            _ => (),
        }
        let error = registry.reset(Some(&outer)).unwrap_err();
        match mode {
            "decision-poison" => assert!(matches!(
                error,
                Error::Decision(LiveDecisionAccessError::DecisionPoisoned)
            )),
            "missing-base" => assert!(matches!(
                error,
                Error::Decision(LiveDecisionAccessError::MissingBase)
            )),
            "missing-orders" => assert!(matches!(
                error,
                Error::Decision(LiveDecisionAccessError::Access(
                    DecisionAccessError::MissingOrders
                ))
            )),
            "invalid" => assert!(matches!(
                error,
                Error::Access(DecisionAccessError::InvalidOrder(1))
            )),
            "list-poison" => assert!(matches!(
                error,
                Error::Access(DecisionAccessError::ListPoisoned)
            )),
            "key-poison" => assert!(matches!(error, Error::OrderPoisoned)),
            _ => unreachable!(),
        }
        let has_first = ["invalid", "list-poison"].contains(&mode);
        assert_eq!(registry.len(), usize::from(has_first), "{mode}");
        if has_first {
            assert_eq!(
                registry
                    .state(&order("A"))
                    .unwrap()
                    .parts()
                    .order
                    .stock_id(),
                "A"
            );
        }
        let expected = if ["invalid", "list-poison", "key-poison"].contains(&mode) {
            vec![json!("create:A:0")]
        } else {
            Vec::new()
        };
        assert_eq!(*rig.events.lock().unwrap(), expected, "{mode}");
    }
}

#[test]
fn registry_grouping_errors_precede_every_adapter_callback() {
    for mode in ["row-poison", "row-key", "rows-poison"] {
        let (rig, mut registry, outer) = fixture("normal", &["A", "B"]);
        registry.reset(Some(&outer)).unwrap();
        rig.events.lock().unwrap().clear();
        let item = order("B");
        match mode {
            "row-poison" => poison(item.clone()),
            "row-key" => *item.write().unwrap() = Order::new("B", 1.0, OrderDir::Buy, None, None),
            _ => (),
        }
        rig.rows
            .lock()
            .unwrap()
            .push(Arc::new(SharedSimulatorExecution {
                order: item,
                trade_value: 1.0,
                trade_cost: 0.0,
                trade_price: 1.0,
            }));
        if mode == "rows-poison" {
            let rows = rig.rows.clone();
            assert!(
                std::thread::spawn(move || {
                    let _guard = rows.try_lock().unwrap();
                    panic!("deliberate result-list poison");
                })
                .join()
                .is_err()
            );
            assert!(matches!(
                registry.update(Some(&rig.rows), (4, 4)),
                Err(Error::ExecutionsPoisoned)
            ));
        }
        let error = registry.update(Some(&rig.rows), (2, 5)).unwrap_err();
        match mode {
            "row-poison" => assert!(matches!(error, Error::OrderPoisoned)),
            "row-key" => assert!(matches!(error, Error::Key(_))),
            "rows-poison" => assert!(matches!(error, Error::ExecutionsPoisoned)),
            _ => unreachable!(),
        }
        assert!(rig.events.lock().unwrap().is_empty());
    }
}

#[test]
fn registry_state_uses_retained_adapter_order_not_query_order() {
    let (rig, mut registry, outer) = fixture("normal", &["A"]);
    let original = match &rig.list().read().unwrap()[0] {
        DecisionOrderItem::Order(item) => item.clone(),
        DecisionOrderItem::Other(_) => unreachable!(),
    };
    registry.reset(Some(&outer)).unwrap();
    *original.write().unwrap() = Order::new("A", 17.0, OrderDir::Buy, Some(time(0)), Some(time(1)));
    let query = order("A");
    assert_eq!(
        registry
            .state(&query)
            .unwrap()
            .parts()
            .order
            .amount()
            .to_bits(),
        17.0_f64.to_bits()
    );
    assert_eq!(query.read().unwrap().amount().to_bits(), 1.0_f64.to_bits());
    *query.write().unwrap() = Order::new("A", 1.0, OrderDir::Buy, None, None);
    assert!(matches!(registry.state(&query), Err(Error::Key(_))));
    poison(query.clone());
    assert!(matches!(registry.state(&query), Err(Error::OrderPoisoned)));
}
