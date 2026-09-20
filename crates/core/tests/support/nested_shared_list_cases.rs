use super::*;
use domain_core::nested_executor::SharedNestedResult;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[path = "shared_result_sink.rs"]
mod shared_result_sink;

#[test]
fn synchronous_return_sink_keeps_the_final_list_identity_and_failure_mutations() {
    for fail in [false, true] {
        let (state, _, mut binding, mut inner, mut strategy, mut outer) =
            harness(1, false, false, None, None);
        let retained = Arc::new(Mutex::new(None));
        let mut sink = shared_result_sink::ClearingResultSink {
            retained: Arc::clone(&retained),
            fail,
        };
        let mut account = LifecycleAccount {
            state: Arc::clone(&state),
            observed: Vec::new(),
        };
        let result = lifecycle(Arc::new(OuterCalendar(Arc::clone(&state))), false, "cash")
            .collect_data(NestedExecutorRun {
                level_binding: &mut binding,
                inner: &mut inner,
                strategy: &mut strategy,
                outer: &mut outer,
                account: &mut account,
                tracker: None,
                return_sink: Some(&mut sink),
                level: 0,
            });
        let handle = retained.lock().unwrap().as_ref().unwrap().clone();
        assert!(handle.lock().unwrap().is_empty());
        assert_eq!(account.observed, [(1, 1)]);
        assert!(
            state
                .lock()
                .unwrap()
                .events
                .iter()
                .any(|value| value == "commit")
        );
        if fail {
            assert!(matches!(
                result,
                Err(NestedExecutorLifecycleError::ReturnSink(_))
            ));
        } else {
            let collection = result.unwrap();
            assert!(Arc::ptr_eq(collection.executions(), &handle));
            assert_eq!(collection.decisions().len(), 1);
        }
    }
}

#[test]
fn legacy_shared_adapters_forward_successful_present_and_missing_previous_results() {
    let (_, _, _, _, mut strategy, _) = harness(2, false, false, None, None);
    let row = OwnedOrderExecution {
        order: Order::new("legacy", 1.0, OrderDir::Buy, None, None),
        trade_value: 10.0,
        trade_cost: 0.0,
        trade_price: 10.0,
    }
    .into_shared();
    let previous = Arc::new(Mutex::new(vec![Arc::clone(&row)]));
    assert!(
        strategy
            .generate_shared_trade_decision(Some(&previous))
            .is_ok()
    );
    assert!(matches!(
        strategy
            .begin_shared_trade_decision(Some(&previous))
            .unwrap(),
        domain_core::NestedStrategyProgress::Ready(_)
    ));
    assert!(matches!(
        strategy.begin_shared_trade_decision(None).unwrap(),
        domain_core::NestedStrategyProgress::Ready(_)
    ));
    strategy.post_shared_execute(&previous).unwrap();
    assert_eq!(
        strategy.previous,
        [
            Some(Arc::as_ptr(&row) as usize),
            Some(Arc::as_ptr(&row) as usize),
            None
        ]
    );
    assert!(Arc::ptr_eq(&previous.lock().unwrap()[0], &row));
}

struct MutatingStrategy {
    inner: Strategy,
    retained: Vec<SharedNestedResult>,
    poison: bool,
}

impl NestedStrategy for MutatingStrategy {
    fn reset(&mut self, outer: &dyn NestedOuterDecision) -> Result<(), NestedStrategyError> {
        self.inner.reset(outer)
    }
    fn alter_outer_decision(
        &mut self,
        outer: &mut dyn NestedOuterDecision,
    ) -> Result<(), NestedStrategyError> {
        self.inner.alter_outer_decision(outer)
    }
    fn generate_trade_decision(
        &mut self,
        previous: Option<&[SharedOrderExecution]>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        self.inner.generate_trade_decision(previous)
    }
    fn generate_shared_trade_decision(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        if let Some(previous) = previous {
            assert!(Arc::ptr_eq(previous, self.retained.last().unwrap()));
            let mut rows = previous.try_lock().expect("no framework guard across hook");
            let replacement = Arc::new(domain_core::shared_simulator::SharedSimulatorExecution {
                order: Arc::clone(&rows[0].order),
                trade_value: 99.0,
                trade_cost: 0.0,
                trade_price: 99.0,
            });
            rows.clear();
            rows.push(replacement);
        }
        self.inner.generate_trade_decision(None)
    }
    fn post_execute(
        &mut self,
        executions: &[SharedOrderExecution],
    ) -> Result<(), NestedStrategyError> {
        self.inner.post_execute(executions)
    }
    fn post_shared_execute(
        &mut self,
        executions: &SharedNestedResult,
    ) -> Result<(), NestedStrategyError> {
        {
            let mut rows = executions
                .try_lock()
                .expect("no framework guard across hook");
            if self.retained.is_empty() {
                let row = Arc::clone(&rows[0]);
                rows.push(row);
            }
        }
        self.retained.push(Arc::clone(executions));
        if self.poison {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    let _guard = executions.lock().unwrap();
                    panic!("poison list after hook mutation");
                }))
                .is_err()
            );
        }
        Ok(())
    }
    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        self.inner.post_upper_level()
    }
}

#[test]
fn actual_nested_loop_retains_previous_list_but_flattens_independent_membership() {
    let (_, calendar, mut binding, mut inner, strategy, mut outer) =
        harness(2, false, false, None, None);
    let mut strategy = MutatingStrategy {
        inner: strategy,
        retained: Vec::new(),
        poison: false,
    };
    let result = NestedExecutorCore::new(false, false)
        .collect_data(
            &calendar,
            &mut binding,
            &mut inner,
            &mut strategy,
            &mut outer,
            0,
        )
        .unwrap();
    let rows = result.executions().lock().unwrap();
    assert_eq!(
        rows.iter().map(|row| row.trade_value).collect::<Vec<_>>(),
        [10.0, 10.0, 11.0]
    );
    assert!(Arc::ptr_eq(&rows[0], &rows[1]));
    let retained_order = Arc::clone(&strategy.retained[0].lock().unwrap()[0].order);
    assert!(Arc::ptr_eq(&retained_order, &rows[0].order));
    retained_order.write().unwrap().set_deal_amount(8.0);
    for row in &rows[..2] {
        assert_eq!(
            row.order.read().unwrap().deal_amount().to_bits(),
            8.0_f64.to_bits()
        );
    }
    assert_eq!(
        strategy.retained[0].lock().unwrap()[0]
            .trade_value
            .to_bits(),
        99.0_f64.to_bits()
    );
    assert!(Arc::ptr_eq(
        &rows[2],
        &strategy.retained[1].lock().unwrap()[0]
    ));
    strategy.retained[1].lock().unwrap().clear();
    drop(rows);
    assert_eq!(result.executions().lock().unwrap().len(), 3);
}

#[test]
fn poisoned_shared_list_stops_flattening_before_indicator_or_upper_hook() {
    let (state, calendar, mut binding, mut inner, strategy, mut outer) =
        harness(2, false, false, None, None);
    let mut strategy = MutatingStrategy {
        inner: strategy,
        retained: Vec::new(),
        poison: true,
    };
    let error = NestedExecutorCore::new(false, false)
        .collect_data(
            &calendar,
            &mut binding,
            &mut inner,
            &mut strategy,
            &mut outer,
            0,
        )
        .err()
        .unwrap();
    assert_eq!(
        error,
        NestedExecutorError::Strategy(strategy_error("nested execution result list lock poisoned"))
    );
    assert_eq!(state.lock().unwrap().step, 1);
    assert!(
        !state
            .lock()
            .unwrap()
            .events
            .iter()
            .any(|event| event == "handle" || event == "upper")
    );
    assert_eq!(
        strategy.retained[0].lock().unwrap_err().into_inner().len(),
        2
    );
    let mut legacy = strategy.inner;
    assert!(
        legacy
            .generate_shared_trade_decision(Some(&strategy.retained[0]))
            .is_err()
    );
    assert!(
        legacy
            .begin_shared_trade_decision(Some(&strategy.retained[0]))
            .is_err()
    );
    assert!(legacy.post_shared_execute(&strategy.retained[0]).is_err());
}
