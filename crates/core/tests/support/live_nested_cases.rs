use super::live_atomic_nested_cases::request;
use super::*;
use domain_core::decision_construction::{
    ConstructedDecisionBase, SharedOrderDecisionConstruction,
};
use domain_core::decision_update::{
    LiveDecisionHandle, SharedDecisionUpdateStrategy, SharedLiveDecision,
};
use domain_core::live_nested_executor::*;
use domain_core::nested_executor::{LiveNestedControlEvent, SharedNestedResult};
use domain_core::nested_executor_lifecycle::LiveNestedBarEnd;
use domain_core::{
    NestedBarEnd, NestedDecisionTracking, NestedExecutorAccount, NestedExecutorAccountError,
    NestedExecutorResume, NestedStrategyPrompt, ResumableNestedConfig, SharedNestedAccountAdapter,
    TradeCalendarRange, TradeCalendarRangeError,
};
use std::sync::RwLock;

#[path = "live_recursive_cases.rs"]
mod recursive;

impl TradeCalendarRange for SharedInnerCalendar {
    fn start_time(&self) -> Result<NaiveDateTime, TradeCalendarRangeError> {
        Ok(self.0.lock().unwrap().start)
    }
    fn get_range_idx(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<(i64, i64), TradeCalendarRangeError> {
        let state = self.0.lock().unwrap();
        assert!(start <= state.end && end >= state.start);
        Ok((0, 0))
    }
}

struct UpdatingOrigin;
impl SharedDecisionUpdateStrategy<()> for UpdatingOrigin {
    fn update_trade_decision(
        &self,
        decision: &SharedLiveDecision<Self, ()>,
        _: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<Option<SharedLiveDecision<Self, ()>>, domain_core::DecisionUpdateStrategyError>
    {
        assert!(decision.try_write().is_ok());
        Ok(Some(decision.clone()))
    }
}

struct Strategy {
    original: LiveDecisionHandle,
    replacement: LiveDecisionHandle,
    child: LiveDecisionHandle,
    prompts: bool,
    events: Arc<Mutex<Vec<&'static str>>>,
    posted: Arc<Mutex<Option<SharedNestedResult>>>,
}
impl LiveNestedStrategy for Strategy {
    fn reset(&mut self, outer: &LiveDecisionHandle) -> Result<(), NestedStrategyError> {
        assert!(Arc::ptr_eq(outer, &self.original));
        self.events.lock().unwrap().push("reset");
        Ok(())
    }
    fn alter_outer_decision(
        &mut self,
        outer: LiveDecisionHandle,
    ) -> Result<LiveDecisionHandle, NestedStrategyError> {
        assert!(Arc::ptr_eq(&outer, &self.original));
        self.events.lock().unwrap().push("alter");
        Ok(self.replacement.clone())
    }
    fn begin(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError> {
        assert!(previous.is_none());
        self.events.lock().unwrap().push("generate");
        if self.prompts {
            Ok(LiveNestedStrategyProgress::Suspended(
                NestedStrategyPrompt {
                    kind: "volume".into(),
                    schema_version: 1,
                    payload: vec![],
                },
            ))
        } else {
            Ok(LiveNestedStrategyProgress::Ready(self.child.clone()))
        }
    }
    fn resume(
        &mut self,
        volume: Option<f64>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError> {
        assert_eq!(volume, Some(42.0));
        self.events.lock().unwrap().push("action");
        Ok(LiveNestedStrategyProgress::Ready(self.child.clone()))
    }
    fn close(&mut self) -> Result<(), NestedStrategyError> {
        self.events.lock().unwrap().push("close");
        Ok(())
    }
    fn post_execute(&mut self, rows: &SharedNestedResult) -> Result<(), NestedStrategyError> {
        self.events.lock().unwrap().push("post");
        *self.posted.lock().unwrap() = Some(rows.clone());
        assert!(rows.try_lock().is_ok());
        Ok(())
    }
    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        self.events.lock().unwrap().push("upper");
        Ok(())
    }
}

struct CheckingAccount {
    actual: SharedNestedAccountAdapter,
    original: LiveDecisionHandle,
    child: LiveDecisionHandle,
    events: Arc<Mutex<Vec<&'static str>>>,
}
impl NestedExecutorAccount for CheckingAccount {
    fn settle_start(&mut self, value: &str) -> Result<(), NestedExecutorAccountError> {
        self.events.lock().unwrap().push("settle");
        self.actual.settle_start(value)
    }
    fn update_bar_end(&mut self, _: NestedBarEnd<'_>) -> Result<(), NestedExecutorAccountError> {
        panic!("legacy path must not run")
    }
    fn update_live_bar_end(
        &mut self,
        bar: LiveNestedBarEnd<'_>,
    ) -> Result<(), NestedExecutorAccountError> {
        assert!(Arc::ptr_eq(bar.outer_decision, &self.original));
        assert!(Arc::ptr_eq(&bar.steps[0].decision, &self.child));
        self.events.lock().unwrap().push("account");
        self.actual.update_live_bar_end(bar)
    }
    fn settle_commit(&mut self) -> Result<(), NestedExecutorAccountError> {
        self.events.lock().unwrap().push("commit");
        self.actual.settle_commit()
    }
}

struct Fixture {
    executor: LiveNestedExecutor,
    original: LiveDecisionHandle,
    child: LiveDecisionHandle,
    replacement: LiveDecisionHandle,
    events: Arc<Mutex<Vec<&'static str>>>,
    account: domain_core::SharedSaoeAccount,
    inner_calendar: Arc<SharedInnerCalendar>,
    outer_steps: Arc<Mutex<usize>>,
    posted: Arc<Mutex<Option<SharedNestedResult>>>,
}

fn fixture(delegated: bool, prompts: bool, fail: bool) -> Fixture {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let (template, _) = request(start, end);
    let mut original = SharedOrderDecisionConstruction::new(Arc::new(UpdatingOrigin));
    original.base = Some(ConstructedDecisionBase {
        start_time: start,
        end_time: end,
        trade_range: None,
    });
    original.orders = Some(template.orders().unwrap());
    original.details = Some(());
    let original: LiveDecisionHandle = Arc::new(RwLock::new(original));
    let (child, _) = request(start, end);
    let (replacement, _) = request(start, end);
    let inner_calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let inner_account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let inner = OwnedAtomicNestedInnerAdapter::new(
        inner_calendar.clone(),
        collector(inner_calendar.clone(), fail),
        inner_account,
        Arc::new(Services),
        "None",
        IndicatorConfig::default(),
        false,
    )
    .with_live_tracking(delegated);
    let events = Arc::new(Mutex::new(Vec::new()));
    let posted = Arc::new(Mutex::new(None));
    let strategy = Strategy {
        original: original.clone(),
        replacement: replacement.clone(),
        child: child.clone(),
        prompts,
        events: events.clone(),
        posted: posted.clone(),
    };
    let outer_steps = Arc::new(Mutex::new(0));
    let executor = LiveNestedExecutor::new(
        Box::new(OuterCalendar {
            start,
            end,
            steps: outer_steps.clone(),
        }),
        ResumableNestedConfig {
            skip_empty_decision: false,
            align_range_limit: false,
            decision_tracking: NestedDecisionTracking {
                outer: true,
                inner: true,
            },
            settle_type: "None".into(),
            indicator_config: IndicatorConfig::default(),
            aggregation_config: OrderIndicatorAggregationConfig::default(),
            level: 0,
        },
        LiveNestedRun {
            level_binding: Box::new(NoopBinding),
            inner: Box::new(inner),
            inner_range_calendar: inner_calendar.clone(),
            strategy: Box::new(strategy),
            outer: original.clone(),
            account: Box::new(CheckingAccount {
                actual: SharedNestedAccountAdapter::new(
                    account.clone(),
                    Arc::new(Services),
                    Arc::new(Services),
                    false,
                ),
                original: original.clone(),
                child: child.clone(),
                events: events.clone(),
            }),
            return_sink: None,
        },
    );
    Fixture {
        executor,
        original,
        child,
        replacement,
        events,
        account,
        inner_calendar,
        outer_steps,
        posted,
    }
}

#[test]
fn native_parent_checks_empty_even_when_skip_disabled_and_uses_time_calendar() {
    let mut f = fixture(false, false, false);
    let orders = f.replacement.orders().unwrap();
    assert!(
        std::thread::spawn(move || {
            let _guard = orders.write().unwrap();
            panic!("poison list to observe empty lookup");
        })
        .join()
        .is_err()
    );
    tracked(
        f.executor.resume(NestedExecutorResume::Continue).unwrap(),
        &f.original,
    );
    assert!(matches!(
        f.executor.resume(NestedExecutorResume::Continue),
        Err(LiveNestedError::Decision(_))
    ));
    assert_eq!(*f.events.lock().unwrap(), ["reset", "alter"]);

    let mut f = fixture(false, false, false);
    let range: domain_core::SharedTradeRange =
        Arc::new(domain_core::TradeRangeByTime::parse("09:30", "09:31").unwrap());
    f.replacement.inherit_range(Some(range.clone())).unwrap();
    tracked(
        f.executor.resume(NestedExecutorResume::Continue).unwrap(),
        &f.original,
    );
    tracked(
        f.executor.resume(NestedExecutorResume::Continue).unwrap(),
        &f.child,
    );
    assert!(Arc::ptr_eq(
        f.child.base().unwrap().trade_range.as_ref().unwrap(),
        &range
    ));
    let LiveNestedEvent::Complete(done) =
        f.executor.resume(NestedExecutorResume::Continue).unwrap()
    else {
        panic!("completion expected")
    };
    assert_eq!(done.decisions.len(), 1);
    assert_eq!(*f.outer_steps.lock().unwrap(), 1);
}

fn tracked(event: LiveNestedEvent, expected: &LiveDecisionHandle) {
    let LiveNestedEvent::Suspended(LiveNestedControlEvent::TrackedDecision(value)) = event else {
        panic!("tracking expected")
    };
    assert!(Arc::ptr_eq(&value, expected));
}

#[test]
fn native_parent_connects_strategy_child_live_records_and_real_account() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/live_nested_parent_contract.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    for delegated in [false, true] {
        let mut f = fixture(delegated, true, false);
        assert_eq!(
            f.executor
                .resume(NestedExecutorResume::Action(Some(1.0)))
                .err(),
            Some(LiveNestedError::InvalidInitialAction)
        );
        tracked(
            f.executor.resume(NestedExecutorResume::Continue).unwrap(),
            &f.original,
        );
        assert!(f.events.lock().unwrap().is_empty());
        assert!(matches!(
            f.executor
                .resume(NestedExecutorResume::Action(Some(99.0)))
                .unwrap(),
            LiveNestedEvent::Suspended(LiveNestedControlEvent::StrategyPrompt(_))
        ));
        tracked(
            f.executor
                .resume(NestedExecutorResume::Action(Some(42.0)))
                .unwrap(),
            &f.child,
        );
        assert!(f.account.try_lock().is_ok());
        assert_eq!(NestedCalendar::trade_step(&*f.inner_calendar).unwrap(), 0);
        let LiveNestedEvent::Complete(done) = f
            .executor
            .resume(NestedExecutorResume::Action(Some(999.0)))
            .unwrap()
        else {
            panic!("completion expected")
        };
        assert_eq!(
            *f.events.lock().unwrap(),
            [
                "reset", "alter", "generate", "action", "post", "upper", "account"
            ]
        );
        assert_eq!(*f.outer_steps.lock().unwrap(), 1);
        assert_eq!(json!(*f.events.lock().unwrap()), source["events"]);
        assert_eq!(json!(*f.outer_steps.lock().unwrap()), source["outer_steps"]);
        assert!(Arc::ptr_eq(&done.decisions[0].decision, &f.child));
        let posted = f.posted.lock().unwrap().clone().unwrap();
        assert!(!Arc::ptr_eq(&posted, &done.executions));
        assert!(Arc::ptr_eq(
            &posted.lock().unwrap()[0],
            &done.executions.lock().unwrap()[0]
        ));
        posted.lock().unwrap().clear();
        assert_eq!(done.executions.lock().unwrap().len(), 1);
        assert_eq!(
            f.executor.resume(NestedExecutorResume::Continue).err(),
            Some(LiveNestedError::Complete)
        );
        f.executor.close().unwrap();
        assert_eq!(
            f.executor.resume(NestedExecutorResume::Continue).err(),
            Some(LiveNestedError::Complete)
        );
    }
}

#[test]
fn native_parent_cancels_only_active_delegate_without_account_completion() {
    for boundary in ["start", "outer", "strategy", "framework", "child"] {
        let mut f = fixture(boundary == "child", boundary == "strategy", false);
        if boundary != "start" {
            tracked(
                f.executor.resume(NestedExecutorResume::Continue).unwrap(),
                &f.original,
            );
        }
        if ["strategy", "framework", "child"].contains(&boundary) {
            let _ = f.executor.resume(NestedExecutorResume::Continue).unwrap();
        }
        f.executor.close().unwrap();
        f.executor.close().unwrap();
        assert_eq!(
            f.executor.resume(NestedExecutorResume::Continue).err(),
            Some(LiveNestedError::Closed)
        );
        assert_eq!(NestedCalendar::trade_step(&*f.inner_calendar).unwrap(), 0);
        assert_eq!(*f.outer_steps.lock().unwrap(), 0);
        let events = f.events.lock().unwrap();
        assert!(!events.contains(&"account") && !events.contains(&"upper"));
        assert_eq!(
            events.iter().filter(|&&s| s == "close").count(),
            usize::from(boundary == "strategy")
        );
    }
}

#[test]
fn native_parent_child_failure_is_terminal_without_outer_finalization() {
    let mut f = fixture(true, false, true);
    tracked(
        f.executor.resume(NestedExecutorResume::Continue).unwrap(),
        &f.original,
    );
    tracked(
        f.executor.resume(NestedExecutorResume::Continue).unwrap(),
        &f.child,
    );
    assert!(matches!(
        f.executor.resume(NestedExecutorResume::Continue),
        Err(LiveNestedError::Inner(_))
    ));
    assert_eq!(
        f.executor.resume(NestedExecutorResume::Continue).err(),
        Some(LiveNestedError::Failed)
    );
    f.executor.close().unwrap();
    assert_eq!(*f.outer_steps.lock().unwrap(), 0);
    assert_eq!(*f.events.lock().unwrap(), ["reset", "alter", "generate"]);
}
