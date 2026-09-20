use super::*;
use domain_core::decision_construction::DecisionAccessError;
use domain_core::decision_update::LiveDecisionAccessError;
use domain_core::saoe_live_generation::LiveSaoeGenerationError as Error;

fn poison<T: Send + Sync + 'static>(value: Arc<RwLock<T>>) {
    assert!(
        std::thread::spawn(move || {
            let _guard = value.try_write().unwrap();
            panic!("injected poison");
        })
        .join()
        .is_err()
    );
}

fn first_order(list: &SharedDecisionOrders) -> Arc<RwLock<Order>> {
    let items = list.read().unwrap();
    let DecisionOrderItem::Order(order) = &items[0] else {
        panic!("order")
    };
    Arc::clone(order)
}

impl Rig {
    pub(super) fn mutate_after_policy(&self) {
        match self.mode {
            "missing-child" => self.outer.write().unwrap().orders = None,
            "poison-decision" => poison(Arc::clone(&self.outer)),
            "poison-child-list" => poison(self.list()),
            "poison-child-order" => poison(first_order(&self.list())),
            "invalid-child" | "invalid-zero" => {
                self.list().write().unwrap()[0] = DecisionOrderItem::Other(Arc::new(7));
            }
            _ => (),
        }
    }
    pub(super) fn mutate_after_children(&self) {
        match self.mode {
            "missing-detail" => self.outer.write().unwrap().orders = None,
            "poison-detail-list" => {
                let list = orders(&["A", "B"]);
                poison(Arc::clone(&list));
                self.outer.write().unwrap().orders = Some(list);
            }
            "poison-detail-order" => poison(first_order(&self.list())),
            "invalid-detail" => {
                self.list().write().unwrap()[0] = DecisionOrderItem::Other(Arc::new(7));
            }
            "empty-detail" => self.replace(&[]),
            _ => (),
        }
    }
}

pub(super) fn fixture(mode: &'static str) -> (Arc<Rig>, LiveSaoeGeneration, LiveDecisionHandle) {
    let mut outer = SharedOrderDecisionConstruction::new(Arc::new(Origin));
    outer.orders = Some(orders(if mode == "delete" {
        &["A", "B", "C"]
    } else {
        &["A", "B"]
    }));
    let rig = Arc::new(Rig {
        mode,
        outer: Arc::new(RwLock::new(outer)),
        events: Mutex::new(Vec::new()),
    });
    let plugin = || Plugin(Arc::clone(&rig));
    let pipeline =
        SaoePolicyPipeline::new(Box::new(plugin()), Box::new(plugin()), Box::new(plugin()));
    let builder = LiveSaoeGeneration::new(pipeline, Box::new(plugin()), Arc::new(plugin()));
    let handle: LiveDecisionHandle = rig.outer.clone();
    (rig, builder, handle)
}

pub(super) fn generate(
    rig: &Rig,
    builder: &mut LiveSaoeGeneration,
    handle: &LiveDecisionHandle,
) -> Result<domain_core::saoe_live_generation::LiveSaoeDecisionParts, Error> {
    builder.generate(handle, |order| {
        let snapshot = order.try_read().unwrap().clone();
        rig.event(format!("state:{}", snapshot.stock_id()));
        assert!(order.try_write().is_ok());
        if rig.mode == "state-failure" {
            return Err(plugin_error("state"));
        }
        if rig.mode == "state-list-poison" {
            poison(rig.list());
        }
        if rig.mode == "append" && snapshot.stock_id() == "A" {
            rig.list()
                .try_write()
                .unwrap()
                .push(orders(&["C"]).write().unwrap().remove(0));
        }
        Ok(state_for(&snapshot))
    })
}

pub(super) fn generate_live(
    rig: &Rig,
    builder: &mut LiveSaoeGeneration,
    handle: &LiveDecisionHandle,
) -> Result<domain_core::saoe_live_generation::LiveSaoeDecisionParts, Error> {
    builder.generate_live(handle, |order| {
        let snapshot = order.try_read().unwrap().clone();
        rig.event(format!("state:{}", snapshot.stock_id()));
        assert!(order.try_write().is_ok());
        if rig.mode == "state-failure" {
            return Err(plugin_error("state"));
        }
        let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
        let backtest_data = domain_core::LiveSaoeBacktestData::from_owned(SaoeBacktestData {
            ticks_index: Vec::new(),
            ticks_for_order: Vec::new(),
            deal_prices: arr1(&[]),
            market_volumes: arr1(&[]),
            features: empty,
        });
        let ticks_index = Arc::clone(&backtest_data.ticks_index);
        let ticks_for_order = Arc::clone(&backtest_data.ticks_for_order);
        Ok(LiveSaoeState::new(LiveSaoeStateParts {
            order: Arc::clone(order),
            cur_time: time(0),
            cur_step: 0,
            position: snapshot.amount(),
            history_exec: Arc::new(RwLock::new(Vec::new())),
            history_steps: Arc::new(RwLock::new(Vec::new())),
            metrics: None,
            backtest_data: backtest_data.into_shared(),
            ticks_index,
            ticks_for_order,
            ticks_per_step: 1,
        }))
    })
}

#[test]
fn live_generation_accepts_original_aliases_and_maps_live_state_failures() {
    for mode in [
        "normal",
        "failure",
        "policy-failure",
        "action-failure",
        "state-failure",
    ] {
        let (rig, mut builder, handle) = fixture(mode);
        let result = generate_live(&rig, &mut builder, &handle);
        match mode {
            "normal" => {
                let parts = result.unwrap();
                assert_eq!(parts.orders.read().unwrap().len(), 2);
                assert_eq!(parts.details.len(), 2);
            }
            "failure" | "policy-failure" | "action-failure" => {
                assert!(matches!(result, Err(Error::Pipeline(_))));
            }
            "state-failure" => assert!(matches!(result, Err(Error::State(_)))),
            _ => unreachable!(),
        }
    }
}

#[test]
fn live_generation_maps_initial_decision_list_and_item_failures() {
    let (rig, mut builder, handle) = fixture("normal");
    rig.outer.write().unwrap().orders = None;
    assert!(matches!(
        generate_live(&rig, &mut builder, &handle),
        Err(Error::Decision(_))
    ));

    let (rig, mut builder, handle) = fixture("normal");
    poison(rig.list());
    assert!(matches!(
        generate_live(&rig, &mut builder, &handle),
        Err(Error::Orders(DecisionAccessError::ListPoisoned))
    ));

    let (rig, mut builder, handle) = fixture("normal");
    rig.list().write().unwrap()[0] = DecisionOrderItem::Other(Arc::new(7));
    assert!(matches!(
        generate_live(&rig, &mut builder, &handle),
        Err(Error::Orders(DecisionAccessError::InvalidOrder(0)))
    ));
}

#[test]
fn live_generation_failure_stages_preserve_prior_effects_and_release_guards() {
    for (mode, last) in [
        ("state-failure", "state:A"),
        ("missing-child", "action:B:2"),
        ("poison-decision", "action:B:2"),
        ("poison-child-list", "action:B:2"),
        ("poison-child-order", "action:B:2"),
        ("invalid-child", "action:B:2"),
        ("create-failure", "create:A"),
        ("missing-detail", "create:B"),
        ("poison-detail-list", "create:B"),
        ("poison-detail-order", "create:B"),
        ("invalid-detail", "create:B"),
        ("invalid-zero", "create:B"),
        ("time-failure", "time"),
        ("frequency-failure", "freq"),
    ] {
        let (rig, mut builder, handle) = fixture(mode);
        let error = generate(&rig, &mut builder, &handle).err().unwrap();
        match mode {
            "state-failure" => assert!(matches!(error, Error::State(_))),
            "missing-child" | "missing-detail" => assert!(matches!(
                error,
                Error::Decision(LiveDecisionAccessError::Access(
                    DecisionAccessError::MissingOrders
                ))
            )),
            "poison-decision" => assert!(matches!(
                error,
                Error::Decision(LiveDecisionAccessError::DecisionPoisoned)
            )),
            "poison-child-list" | "poison-detail-list" => assert!(matches!(
                error,
                Error::Orders(DecisionAccessError::ListPoisoned)
            )),
            "poison-child-order" | "poison-detail-order" => assert!(matches!(
                error,
                Error::Orders(DecisionAccessError::OrderPoisoned(0))
            )),
            "invalid-child" | "invalid-detail" | "invalid-zero" => assert!(matches!(
                error,
                Error::Orders(DecisionAccessError::InvalidOrder(0))
            )),
            "create-failure" => assert!(matches!(error, Error::Factory(_))),
            _ => assert!(matches!(error, Error::Calendar(_))),
        }
        assert!(!error.to_string().is_empty());
        let events = rig.events.lock().unwrap();
        assert_eq!(events.last().unwrap(), last, "{mode}");
        if mode == "invalid-zero" {
            assert!(!events.iter().any(|event| event == "create:A"));
        }
        if mode != "poison-decision" {
            assert!(rig.outer.try_write().is_ok());
        }
    }
}

#[test]
fn live_generation_handles_empty_zero_and_invalid_initial_lists() {
    for mode in [
        "empty",
        "zero",
        "empty-detail",
        "invalid",
        "missing",
        "poison-list",
        "state-list-poison",
    ] {
        let (rig, mut builder, handle) = fixture(mode);
        match mode {
            "empty" => rig.replace(&[]),
            "invalid" => rig.list().write().unwrap()[0] = DecisionOrderItem::Other(Arc::new(7)),
            "missing" => rig.outer.write().unwrap().orders = None,
            "poison-list" => poison(rig.list()),
            _ => (),
        }
        let result = generate(&rig, &mut builder, &handle);
        match mode {
            "empty" => {
                let result = result.unwrap();
                assert!(result.orders.read().unwrap().is_empty());
                assert!(result.details.is_empty());
                assert_eq!(*rig.events.lock().unwrap(), ["policy:"]);
            }
            "zero" => {
                let result = result.unwrap();
                assert!(result.orders.read().unwrap().is_empty());
                assert_eq!(result.details.len(), 2);
                assert!(result.details.iter().all(|row| row.execution_volume == 0.0));
            }
            "empty-detail" => {
                let result = result.unwrap();
                assert_eq!(result.orders.read().unwrap().len(), 2);
                assert!(result.details.is_empty());
            }
            "invalid" => assert!(matches!(
                result,
                Err(Error::Orders(DecisionAccessError::InvalidOrder(0)))
            )),
            "missing" => assert!(matches!(
                result,
                Err(Error::Decision(LiveDecisionAccessError::Access(
                    DecisionAccessError::MissingOrders
                )))
            )),
            _ => assert!(matches!(
                result,
                Err(Error::Orders(DecisionAccessError::ListPoisoned))
            )),
        }
    }
}
