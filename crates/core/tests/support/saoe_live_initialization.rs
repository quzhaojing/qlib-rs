use super::*;
use domain_core::decision_construction::{
    DecisionConstructionError, DecisionConstructionStrategy, DecisionTotalStep,
};
use domain_core::saoe_live_generation::LiveSaoeDecisionParts;

struct RetainedOrigin {
    calls: Mutex<usize>,
    fail: Option<usize>,
    updates: Mutex<usize>,
}
impl DecisionConstructionStrategy for RetainedOrigin {
    type Error = SaoePluginError;
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), Self::Error> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        if self.fail == Some(*calls) {
            return Err(plugin_error("constructor calendar"));
        }
        let minute = i64::try_from(*calls).unwrap();
        Ok((time(minute), time(minute + 1)))
    }
}
impl SharedDecisionUpdateStrategy<Vec<domain_core::SaoeTradeDetail>> for RetainedOrigin {
    fn update_trade_decision(
        &self,
        decision: &SharedLiveDecision<Self, Vec<domain_core::SaoeTradeDetail>>,
        _: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<
        Option<SharedLiveDecision<Self, Vec<domain_core::SaoeTradeDetail>>>,
        domain_core::DecisionUpdateStrategyError,
    > {
        assert_eq!(
            Arc::as_ptr(&decision.try_read().unwrap().strategy),
            std::ptr::from_ref(self)
        );
        *self.updates.lock().unwrap() += 1;
        Ok(Some(Arc::clone(decision)))
    }
}
struct UpdateCalendar;
impl domain_core::DecisionUpdateCalendar for UpdateCalendar {
    fn trade_len(&self) -> Result<i64, domain_core::DecisionUpdateCalendarError> {
        Ok(7)
    }
}

fn origin(fail: Option<usize>) -> Arc<RetainedOrigin> {
    Arc::new(RetainedOrigin {
        calls: Mutex::new(0),
        fail,
        updates: Mutex::new(0),
    })
}

fn generated() -> LiveSaoeDecisionParts {
    let (rig, mut builder, handle) = failures::fixture("normal");
    failures::generate(&rig, &mut builder, &handle).unwrap()
}

#[test]
fn generated_decision_retains_original_strategy_orders_details_and_update_identity() {
    let parts = generated();
    let orders = Arc::clone(&parts.orders);
    let strategy = origin(None);
    let decision = parts.into_decision(Arc::clone(&strategy)).unwrap();
    let retained = decision.read().unwrap();
    assert!(Arc::ptr_eq(&retained.strategy, &strategy));
    assert!(Arc::ptr_eq(retained.orders.as_ref().unwrap(), &orders));
    assert_eq!(retained.total_step, DecisionTotalStep::Unset);
    assert_eq!(retained.base.as_ref().unwrap().start_time, time(1));
    assert_eq!(retained.base.as_ref().unwrap().end_time, time(2));
    assert!(retained.base.as_ref().unwrap().trade_range.is_none());
    assert_eq!(
        retained
            .details
            .as_ref()
            .unwrap()
            .iter()
            .map(|d| d.instrument.as_str())
            .collect::<Vec<_>>(),
        ["A", "B"]
    );
    for item in orders.read().unwrap().iter() {
        let DecisionOrderItem::Order(order) = item else {
            panic!("order")
        };
        assert_eq!(order.read().unwrap().start_time(), Some(time(2)));
        assert_eq!(order.read().unwrap().end_time(), Some(time(3)));
    }
    drop(retained);
    assert_eq!(*strategy.calls.lock().unwrap(), 2);
    let handle: LiveDecisionHandle = decision.clone();
    assert!(Arc::ptr_eq(&handle.orders().unwrap(), &orders));
    let updated = Arc::clone(&handle)
        .update(&UpdateCalendar)
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&updated, &handle));
    assert_eq!(handle.total_step().unwrap(), DecisionTotalStep::Value(7));
    assert_eq!(*strategy.updates.lock().unwrap(), 1);
}

#[test]
fn constructor_failures_do_not_publish_and_inspectable_state_keeps_reached_writes() {
    for fail in [Some(1), Some(2), None] {
        let parts = generated();
        let orders = Arc::clone(&parts.orders);
        if fail.is_none() {
            orders.write().unwrap()[1] = DecisionOrderItem::Other(Arc::new(7));
        }
        let strategy = origin(fail);
        let mut decision = SharedOrderDecisionConstruction::new(Arc::clone(&strategy));
        let error = parts.initialize_into(&mut decision).unwrap_err();
        match fail {
            Some(_) => assert!(matches!(error, DecisionConstructionError::Calendar(_))),
            None => assert!(matches!(error, DecisionConstructionError::InvalidOrder(1))),
        }
        assert!(Arc::ptr_eq(&strategy, &decision.strategy));
        assert_eq!(decision.base.is_some(), fail != Some(1));
        assert_eq!(decision.orders.is_some(), fail != Some(1));
        assert!(decision.details.is_none());
        let items = orders.read().unwrap();
        let DecisionOrderItem::Order(first) = &items[0] else {
            panic!("order")
        };
        assert_eq!(
            first.read().unwrap().start_time(),
            if fail.is_none() { Some(time(2)) } else { None }
        );
    }
    for fail in [1, 2] {
        let strategy = origin(Some(fail));
        assert!(matches!(
            generated().into_decision(Arc::clone(&strategy)),
            Err(DecisionConstructionError::Calendar(_))
        ));
        assert_eq!(*strategy.calls.lock().unwrap(), fail);
    }
}
