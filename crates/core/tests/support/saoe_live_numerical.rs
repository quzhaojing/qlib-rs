use super::*;
use serde_json::json;

impl SaoeAdapterContext for Source {
    fn current_trade_step(&self) -> Result<i64, SaoePluginError> {
        self.events.lock().unwrap().push("step".to_owned());
        if self.mode == "state" {
            self.mutate("S", 80.0);
        }
        Ok(3)
    }
    fn latest_price_advantage(&self) -> Result<f64, SaoePluginError> {
        self.events.lock().unwrap().push("indicator".to_owned());
        match self.mode {
            "indicator" => self.mutate("Q", 50.0),
            "indicator-poison" => poison(self.order.clone()),
            "next-failure" => {
                *self.order.try_write().unwrap() =
                    Order::new("A", 10.0, OrderDir::Buy, Some(time(0)), None);
            }
            "indicator-failure" => {
                return Err(SaoePluginError {
                    message: "indicator failed".to_owned(),
                });
            }
            _ => (),
        }
        Ok(7.5)
    }
    fn warn_overfill(&self, _: f64, _: f64) -> Result<(), SaoePluginError> {
        panic!("the live numerical source fixture does not overfill")
    }
}

fn scalar(value: &SaoeNumeric) -> f64 {
    let SaoeNumeric::Scalar(value) = value else {
        panic!("expected a scalar metric");
    };
    *value
}

#[test]
fn bound_constructor_preserves_source_callback_and_order_observation_order() {
    let source = Arc::new(source("normal"));
    let start_source = Arc::clone(&source);
    let baseline_source = Arc::clone(&source);
    let adapter = ConcreteSaoeStateAdapter::new_live_bound(
        source.clone(),
        source.clone(),
        config(),
        &source.order,
        &mut move || {
            start_source.events.lock().unwrap().push("start".to_owned());
            *start_source.order.write().unwrap() =
                Order::new("I", 20.0, OrderDir::Sell, Some(time(1)), Some(time(5)));
            Ok(1)
        },
        &mut move || {
            baseline_source
                .events
                .lock()
                .unwrap()
                .push("baseline".to_owned());
            let order = baseline_source.order.read().unwrap();
            assert_eq!(order.stock_id(), "I");
            assert_eq!(order.amount().to_bits(), 20.0_f64.to_bits());
            Ok(arr1(&[10.0, 12.0]))
        },
    )
    .unwrap();
    assert_eq!(*source.events.lock().unwrap(), ["start", "baseline"]);
    assert_eq!(adapter.position().to_bits(), 10.0_f64.to_bits());
    assert_eq!(adapter.cur_time(), Some(time(1)));
    let state = adapter.state_snapshot().unwrap();
    assert_eq!(state.parts().order.stock_id(), "I");
    assert_eq!(state.parts().order.amount().to_bits(), 20.0_f64.to_bits());
    assert_eq!(state.parts().cur_step, 2);
    let expected = &live_contract::source()["init"];
    assert_eq!(
        &expected["events"].as_array().unwrap()[..2],
        &[json!("start"), json!("baseline")]
    );
    assert_eq!(
        json!({
            "position": adapter.position(),
            "time": adapter.cur_time().unwrap().to_string(),
            "amount": state.parts().order.amount(),
        }),
        expected["initial"]
    );
}

#[test]
fn bound_constructor_stops_at_each_failure_without_reordering_callbacks() {
    let failure = || SaoePluginError {
        message: "constructor failure".to_owned(),
    };
    let source = Arc::new(source("normal"));
    let events = Arc::new(Mutex::new(Vec::new()));
    let start_events = Arc::clone(&events);
    assert!(matches!(
        ConcreteSaoeStateAdapter::new_live_bound(
            source.clone(),
            source.clone(),
            config(),
            &source.order,
            &mut move || {
                start_events.lock().unwrap().push("start");
                Err(failure())
            },
            &mut || panic!("baseline must not follow a start failure"),
        ),
        Err(SaoeAdapterError::Plugin(_))
    ));
    assert_eq!(*events.lock().unwrap(), ["start"]);

    let events = Arc::new(Mutex::new(Vec::new()));
    let start_events = Arc::clone(&events);
    let baseline_events = Arc::clone(&events);
    assert!(matches!(
        ConcreteSaoeStateAdapter::new_live_bound(
            source.clone(),
            source.clone(),
            config(),
            &source.order,
            &mut move || {
                start_events.lock().unwrap().push("start");
                Ok(1)
            },
            &mut move || {
                baseline_events.lock().unwrap().push("baseline");
                Err(failure())
            },
        ),
        Err(SaoeAdapterError::Plugin(_))
    ));
    assert_eq!(*events.lock().unwrap(), ["start", "baseline"]);
}

#[test]
fn bound_constructor_reports_order_and_tick_failures_at_the_reached_stage() {
    let source = Arc::new(source("normal"));
    let missing_start = Arc::new(RwLock::new(Order::new(
        "A",
        10.0,
        OrderDir::Buy,
        None,
        Some(time(5)),
    )));
    assert!(matches!(
        ConcreteSaoeStateAdapter::new_live_bound(
            source.clone(),
            source.clone(),
            config(),
            &missing_start,
            &mut || Ok(1),
            &mut || Ok(arr1(&[10.0, 12.0])),
        ),
        Err(SaoeAdapterError::Order(
            domain_core::OrderError::MissingStartTime
        ))
    ));

    let poisoned = Arc::new(RwLock::new(order(OrderDir::Buy)));
    poison(Arc::clone(&poisoned));
    assert!(matches!(
        ConcreteSaoeStateAdapter::new_live_bound(
            source.clone(),
            source.clone(),
            config(),
            &poisoned,
            &mut || panic!("start callback must not follow initial order poison"),
            &mut || panic!("baseline callback must not follow initial order poison"),
        ),
        Err(SaoeAdapterError::AdapterOrderPoisoned)
    ));

    let between_poison = Arc::new(RwLock::new(order(OrderDir::Buy)));
    let poison_from_baseline = Arc::clone(&between_poison);
    assert!(matches!(
        ConcreteSaoeStateAdapter::new_live_bound(
            source.clone(),
            source.clone(),
            config(),
            &between_poison,
            &mut || Ok(1),
            &mut move || {
                poison(Arc::clone(&poison_from_baseline));
                Ok(arr1(&[10.0, 12.0]))
            },
        ),
        Err(SaoeAdapterError::AdapterOrderPoisoned)
    ));

    let mut empty_ticks = config();
    empty_ticks.backtest_data.ticks_for_order.clear();
    assert!(matches!(
        ConcreteSaoeStateAdapter::new_live_bound(
            source.clone(),
            source.clone(),
            empty_ticks,
            &source.order,
            &mut || Ok(1),
            &mut || Ok(arr1(&[10.0, 12.0])),
        ),
        Err(SaoeAdapterError::EmptyOrderTicks)
    ));

    let mut empty_index = config();
    empty_index.backtest_data.ticks_index.clear();
    let callbacks = Arc::new(Mutex::new(Vec::new()));
    let start_callbacks = Arc::clone(&callbacks);
    let baseline_callbacks = Arc::clone(&callbacks);
    assert!(matches!(
        ConcreteSaoeStateAdapter::new_live_bound(
            source.clone(),
            source.clone(),
            empty_index,
            &source.order,
            &mut move || {
                start_callbacks.lock().unwrap().push("start");
                Ok(1)
            },
            &mut move || {
                baseline_callbacks.lock().unwrap().push("baseline");
                Ok(arr1(&[10.0, 12.0]))
            },
        ),
        Err(SaoeAdapterError::EmptyTicks)
    ));
    assert_eq!(*callbacks.lock().unwrap(), ["start", "baseline"]);
}

#[test]
fn failed_shared_rebind_preserves_previous_session_history_and_position() {
    let source = Arc::new(source("normal"));
    let mut adapter =
        ConcreteSaoeStateAdapter::new_live(source.clone(), source.clone(), config()).unwrap();
    adapter.reset_shared_order(&source.order).unwrap();
    adapter
        .update_executions(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
        .unwrap();
    let invalid = Arc::new(RwLock::new(Order::new(
        "B",
        99.0,
        OrderDir::Sell,
        None,
        Some(time(5)),
    )));
    assert!(matches!(
        adapter.reset_shared_order(&invalid),
        Err(SaoeAdapterError::Order(
            domain_core::OrderError::MissingStartTime
        ))
    ));
    assert_eq!(adapter.position().to_bits(), 5.0_f64.to_bits());
    assert_eq!(adapter.cur_time(), Some(time(2)));
    assert_eq!(adapter.history_exec().unwrap().len(), 2);
    assert_eq!(adapter.history_steps().unwrap().len(), 1);
    assert_eq!(
        adapter.state_snapshot().unwrap().parts().order.stock_id(),
        "A"
    );
}

#[test]
fn poisoned_order_stops_at_post_indicator_or_pre_combined_market_boundary() {
    for mode in ["indicator-poison", "combined-poison"] {
        let source = Arc::new(source(mode));
        let mut adapter = if mode == "combined-poison" {
            // An unexpected market callback would panic because no response is supplied.
            ConcreteSaoeStateAdapter::new(Arc::new(Market::new([])), source.clone(), config())
                .unwrap()
        } else {
            ConcreteSaoeStateAdapter::new_live(source.clone(), source.clone(), config()).unwrap()
        };
        adapter.reset_shared_order(&source.order).unwrap();
        if mode == "combined-poison" {
            poison(source.order.clone());
        }
        assert!(matches!(
            adapter.update_executions(&[execution(0, 2.0)], (0, 1)),
            Err(SaoeAdapterError::AdapterOrderPoisoned)
        ));
        let events = if mode == "combined-poison" {
            vec![]
        } else {
            vec!["volume:A", "price:A:1", "indicator"]
        };
        assert_eq!(*source.events.lock().unwrap(), events);
        assert_eq!(adapter.position().to_bits(), 10.0_f64.to_bits());
        assert_eq!(adapter.cur_time(), Some(time(0)));
        assert!(adapter.history_exec().unwrap().is_empty());
        assert!(adapter.history_steps().unwrap().is_empty());
    }
}

#[test]
fn concrete_engine_implements_live_registry_adapter_success_and_error_paths() {
    for initialized in [false, true] {
        let source = Arc::new(source("normal"));
        let mut adapter =
            ConcreteSaoeStateAdapter::new_live(source.clone(), source.clone(), config()).unwrap();
        if initialized {
            adapter.reset_shared_order(&source.order).unwrap();
        }
        let mut plugin: Box<dyn domain_core::saoe_live_registry::LiveSaoeStateAdapter> =
            Box::new(adapter);
        if initialized {
            assert!(plugin.state().unwrap().parts().metrics.is_none());
            plugin
                .update(&[execution(0, 2.0), execution(1, 3.0)], (0, 1))
                .unwrap();
            plugin.finalize().unwrap();
            assert_eq!(
                scalar(
                    &plugin
                        .state()
                        .unwrap()
                        .parts()
                        .metrics
                        .as_ref()
                        .unwrap()
                        .ffr
                )
                .to_bits(),
                0.5_f64.to_bits()
            );
        } else {
            for error in [
                plugin.state().unwrap_err(),
                plugin.update(&[], (0, 1)).unwrap_err(),
                plugin.finalize().unwrap_err(),
            ] {
                assert!(error.message.contains("not been reset"), "{error}");
            }
        }
    }
}

#[test]
fn live_numerical_engine_matches_source_mutations_and_failure_commit_points() {
    let oracle = live_contract::source();
    for mode in [
        "normal",
        "volume",
        "price",
        "indicator",
        "state",
        "volume-failure",
        "price-failure",
        "indicator-failure",
        "next-failure",
    ] {
        let mut source = source(mode);
        source.mutate_failure = false;
        let source = Arc::new(source);
        let mut config = config();
        config.start_step = 1;
        config.deal_prices = arr1(&[10.0, 12.0]);
        let mut adapter =
            ConcreteSaoeStateAdapter::new_live(source.clone(), source.clone(), config).unwrap();
        adapter.reset_shared_order(&source.order).unwrap();
        let initial = json!({"position": adapter.position(), "time": adapter.cur_time().unwrap().to_string(),
            "amount": source.order.read().unwrap().amount()});
        let result = adapter.update_executions(&[execution(0, 2.0), execution(1, 3.0)], (0, 1));
        let error = classify_result(mode, result);
        let state = adapter.state_snapshot().unwrap();
        adapter.finalize_metrics().unwrap();
        // Explicit extra observation to retrieve final metrics; it invokes the calendar once more.
        let final_state = adapter.state_snapshot().unwrap();
        let metrics = final_state.parts().metrics.as_ref().unwrap();
        let actual = json!({"events": *source.events.lock().unwrap(), "initial": initial, "error": error,
            "position": adapter.position(), "time": adapter.cur_time().unwrap().to_string(),
            "state_stock": state.parts().order.stock_id(), "state_amount": state.parts().order.amount(),
            "state_step": state.parts().cur_step, "state_metrics_none": state.parts().metrics.is_none(),
            "exec_stock": adapter.history_exec().unwrap().iter().map(|row| row.stock_id.clone()).collect::<Vec<_>>(),
            "exec_ffr": adapter.history_exec().unwrap().iter().map(|row| row.ffr).collect::<Vec<_>>(),
            "step_count": adapter.history_steps().unwrap().len(), "metric_stock": metrics.stock_id,
            "metric_ffr": scalar(&metrics.ffr), "metric_position": scalar(&metrics.position)});
        let mut expected = oracle[mode].clone();
        let events = expected["events"].as_array_mut().unwrap();
        // Constructor loading is outside this preconfigured numerical-engine test.
        events.drain(..2);
        events.push(json!("step"));
        assert_eq!(actual, expected, "{mode}");
    }
}

fn classify_result(mode: &str, result: Result<(), SaoeAdapterError>) -> Option<&'static str> {
    match mode {
        "volume-failure" => {
            assert!(matches!(
                result,
                Err(SaoeAdapterError::LiveMarket(Error::Volume(_)))
            ));
            Some("RuntimeError")
        }
        "price-failure" => {
            assert!(matches!(
                result,
                Err(SaoeAdapterError::LiveMarket(Error::Price(_)))
            ));
            Some("RuntimeError")
        }
        "indicator-failure" => {
            assert!(matches!(result, Err(SaoeAdapterError::Plugin(_))));
            Some("RuntimeError")
        }
        "next-failure" => {
            assert!(matches!(result, Err(SaoeAdapterError::MissingEndTime)));
            Some("TypeError")
        }
        _ => {
            result.unwrap();
            None
        }
    }
}

#[test]
fn shared_binding_reads_external_changes_and_rejects_poison_without_resetting_state() {
    let source = Arc::new(source("normal"));
    let mut adapter =
        ConcreteSaoeStateAdapter::new_live(source.clone(), source.clone(), config()).unwrap();
    adapter.reset_shared_order(&source.order).unwrap();
    source.mutate("E", 20.0);
    assert_eq!(
        adapter.state_snapshot().unwrap().parts().order.stock_id(),
        "E"
    );
    assert_eq!(adapter.position().to_bits(), 10.0_f64.to_bits());
    poison(source.order.clone());
    assert!(matches!(
        adapter.reset_shared_order(&source.order),
        Err(SaoeAdapterError::AdapterOrderPoisoned)
    ));
    assert_eq!(adapter.position().to_bits(), 10.0_f64.to_bits());
    assert!(matches!(
        adapter.state_snapshot(),
        Err(SaoeAdapterError::AdapterOrderPoisoned)
    ));
    assert!(matches!(
        adapter.finalize_metrics(),
        Err(SaoeAdapterError::AdapterOrderPoisoned)
    ));
    assert!(matches!(
        adapter.update_executions(&[], (0, 1)),
        Err(SaoeAdapterError::LiveMarket(Error::VolumeOrderPoisoned))
    ));
    assert!(adapter.history_exec().unwrap().is_empty());
}
