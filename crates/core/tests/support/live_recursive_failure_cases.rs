use super::*;
use std::sync::Weak;

struct FailingCleanup {
    inner: Box<dyn LiveNestedStrategy>,
    calls: Arc<Mutex<usize>>,
    _lifetime: Arc<()>,
}

impl LiveNestedStrategy for FailingCleanup {
    fn reset(&mut self, outer: &LiveDecisionHandle) -> Result<(), NestedStrategyError> {
        self.inner.reset(outer)
    }
    fn alter_outer_decision(
        &mut self,
        outer: LiveDecisionHandle,
    ) -> Result<LiveDecisionHandle, NestedStrategyError> {
        self.inner.alter_outer_decision(outer)
    }
    fn begin(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError> {
        self.inner.begin(previous)
    }
    fn resume(
        &mut self,
        volume: Option<f64>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError> {
        self.inner.resume(volume)
    }
    fn post_execute(&mut self, rows: &SharedNestedResult) -> Result<(), NestedStrategyError> {
        self.inner.post_execute(rows)
    }
    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        self.inner.post_upper_level()
    }
    fn close(&mut self) -> Result<(), NestedStrategyError> {
        self.inner.close()?;
        *self.calls.lock().unwrap() += 1;
        Err(NestedStrategyError {
            message: "cleanup failed after releasing generator".into(),
        })
    }
}

struct CleanupFactory {
    inner: GraphFactory,
    calls: Arc<Mutex<usize>>,
    lifetimes: Arc<Mutex<Vec<Weak<()>>>>,
}
impl LiveNestedChildAssemblyFactory for CleanupFactory {
    fn assemble(
        &mut self,
        decision: &LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedChildAssembly, domain_core::NestedInnerExecutorError> {
        let mut assembly = self.inner.assemble(decision, level)?;
        let lifetime = Arc::new(());
        self.lifetimes
            .lock()
            .unwrap()
            .push(Arc::downgrade(&lifetime));
        assembly.run.strategy = Box::new(FailingCleanup {
            inner: assembly.run.strategy,
            calls: self.calls.clone(),
            _lifetime: lifetime,
        });
        Ok(assembly)
    }
}

#[test]
fn native_configured_cleanup_failure_releases_graph_and_never_retries_close() {
    for explicit in [false, true] {
        let (factory, original) = factory(1, false);
        let calendar = factory.calendar.clone();
        let calls = Arc::new(Mutex::new(0));
        let lifetimes = Arc::new(Mutex::new(Vec::new()));
        let mut session = LiveConfiguredChildSession::new(Box::new(CleanupFactory {
            inner: factory,
            calls: calls.clone(),
            lifetimes: lifetimes.clone(),
        }));
        assert!(matches!(
            session.begin(original.clone(), 0).unwrap(),
            LiveNestedInnerProgress::Suspended(LiveNestedControlEvent::TrackedDecision(_))
        ));
        assert!(matches!(
            session.resume(NestedExecutorResume::Continue).unwrap(),
            LiveNestedInnerProgress::Suspended(LiveNestedControlEvent::StrategyPrompt(_))
        ));
        assert!(lifetimes.lock().unwrap()[0].upgrade().is_some());
        if explicit {
            assert!(
                session
                    .close()
                    .err()
                    .unwrap()
                    .message
                    .contains("cleanup failed")
            );
            assert!(lifetimes.lock().unwrap()[0].upgrade().is_none());
            session.close().unwrap();
            assert!(
                session
                    .resume(NestedExecutorResume::Continue)
                    .err()
                    .unwrap()
                    .message
                    .contains("not active")
            );
            // A failed cleanup does not prevent construction of an independent next graph.
            assert!(matches!(
                session.begin(original, 1).unwrap(),
                LiveNestedInnerProgress::Suspended(_)
            ));
        }
        drop(session);
        assert_eq!(*calls.lock().unwrap(), 1);
        assert!(
            lifetimes
                .lock()
                .unwrap()
                .iter()
                .all(|weak| weak.upgrade().is_none())
        );
        assert_eq!(NestedCalendar::trade_step(&*calendar).unwrap(), 0);
    }
}

#[test]
fn native_recursive_adapter_delegates_calendar_and_raw_indicator_without_legacy_execution() {
    let (factory, original) = factory(1, false);
    let calendar = factory.calendar.clone();
    let account = factory.account.clone();
    let (start, end) = NestedCalendar::step_time(&*calendar).unwrap();
    let lifecycle = OwnedAtomicNestedInnerAdapter::new(
        calendar.clone(),
        collector(calendar.clone(), true),
        account.clone(),
        Arc::new(Services),
        "None",
        IndicatorConfig::default(),
        false,
    );
    let expected = lifecycle.order_indicator_handle().unwrap();
    let mut inner = LiveRecursiveInnerAdapter::new(
        Box::new(lifecycle),
        LiveConfiguredChildSession::new(Box::new(factory)),
    );
    assert_eq!(inner.trade_len().unwrap(), 1);
    assert_eq!(inner.trade_step().unwrap(), 0);
    assert_eq!(inner.step_time().unwrap(), (start, end));
    assert!(!inner.finished().unwrap());
    inner.step().unwrap();
    assert!(inner.finished().unwrap());
    inner.reset_window(start, end).unwrap();
    assert!(!inner.finished().unwrap());
    assert!(Arc::ptr_eq(
        &inner.order_indicator_handle().unwrap(),
        &expected
    ));
    assert!(
        inner
            .order_indicator_snapshot()
            .unwrap()
            .metric("deal_amount")
            .is_none()
    );
    assert!(
        inner
            .collect_data(&mut decision(start, end), 0)
            .err()
            .unwrap()
            .message
            .contains("native resumable")
    );
    assert!(
        inner
            .collect_live_data(&original, 0)
            .err()
            .unwrap()
            .message
            .contains("does not support live")
    );
    calendar.0.lock().unwrap().fail_reset = true;
    assert!(inner.reset_window(start, end).is_err());
    calendar.0.lock().unwrap().fail_reset = false;
    let poison = account.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.lock().unwrap();
            panic!("poison account");
        })
        .join()
        .is_err()
    );
    assert!(
        inner
            .order_indicator_handle()
            .err()
            .unwrap()
            .message
            .contains("poisoned")
    );
    assert!(
        inner
            .order_indicator_snapshot()
            .err()
            .unwrap()
            .message
            .contains("poisoned")
    );
}
