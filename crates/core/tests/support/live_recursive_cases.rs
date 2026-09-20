use super::*;
use domain_core::live_recursive_inner::*;
use domain_core::nested_executor::LiveNestedInnerProgress;

#[path = "live_recursive_failure_cases.rs"]
mod failure_cases;
#[path = "live_parent_failure_cases.rs"]
mod parent_failures;

struct CalendarView(Arc<SharedInnerCalendar>);
impl NestedCalendar for CalendarView {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        self.0.finished()
    }
    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        self.0.trade_len()
    }
    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        self.0.trade_step()
    }
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        NestedCalendar::step_time(&*self.0)
    }
    fn step(&self) -> Result<(), NestedCalendarError> {
        NestedCalendar::step(&*self.0)
    }
}

struct GraphFactory {
    calendar: Arc<SharedInnerCalendar>,
    account: domain_core::SharedSaoeAccount,
    decisions: Vec<LiveDecisionHandle>,
    levels: Arc<Mutex<Vec<usize>>>,
    events: Arc<Mutex<Vec<&'static str>>>,
    posted: Arc<Mutex<Option<SharedNestedResult>>>,
    fail_deal: bool,
    fail_assembly: bool,
}

impl LiveNestedChildAssemblyFactory for GraphFactory {
    fn assemble(
        &mut self,
        outer: &LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedChildAssembly, domain_core::NestedInnerExecutorError> {
        self.levels.lock().unwrap().push(level);
        if self.fail_assembly {
            return Err(domain_core::NestedInnerExecutorError {
                message: "assembly failed".into(),
            });
        }
        assert!(outer.orders().unwrap().try_write().is_ok());
        let (start, end) = NestedCalendar::step_time(&*self.calendar).unwrap();
        let inner_calendar = Arc::new(SharedInnerCalendar::new(start, end));
        let inner_account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
        let atomic = OwnedAtomicNestedInnerAdapter::new(
            inner_calendar.clone(),
            collector(
                inner_calendar.clone(),
                self.fail_deal || self.decisions.len() > 1,
            ),
            inner_account.clone(),
            Arc::new(Services),
            "None",
            IndicatorConfig::default(),
            false,
        )
        .with_live_tracking(true);
        let inner: Box<dyn NestedInnerExecutor> = if self.decisions.len() > 1 {
            Box::new(LiveRecursiveInnerAdapter::new(
                Box::new(atomic),
                LiveConfiguredChildSession::new(Box::new(Self {
                    calendar: inner_calendar.clone(),
                    account: inner_account,
                    decisions: self.decisions[1..].to_vec(),
                    levels: self.levels.clone(),
                    events: self.events.clone(),
                    posted: self.posted.clone(),
                    fail_deal: self.fail_deal,
                    fail_assembly: false,
                })),
            ))
        } else {
            Box::new(atomic)
        };
        let child = self.decisions[0].clone();
        Ok(LiveNestedChildAssembly {
            calendar: Box::new(CalendarView(self.calendar.clone())),
            config: ResumableNestedConfig {
                skip_empty_decision: false,
                align_range_limit: false,
                decision_tracking: NestedDecisionTracking {
                    outer: true,
                    inner: true,
                },
                settle_type: "None".into(),
                indicator_config: IndicatorConfig::default(),
                aggregation_config: OrderIndicatorAggregationConfig::default(),
                level,
            },
            run: LiveNestedRun {
                level_binding: Box::new(NoopBinding),
                inner,
                inner_range_calendar: inner_calendar,
                strategy: Box::new(Strategy {
                    original: outer.clone(),
                    replacement: outer.clone(),
                    child: child.clone(),
                    prompts: self.decisions.len() == 1,
                    events: self.events.clone(),
                    posted: self.posted.clone(),
                }),
                outer: outer.clone(),
                account: Box::new(CheckingAccount {
                    actual: SharedNestedAccountAdapter::new(
                        self.account.clone(),
                        Arc::new(Services),
                        Arc::new(Services),
                        false,
                    ),
                    original: outer.clone(),
                    child,
                    events: self.events.clone(),
                }),
                return_sink: None,
            },
        })
    }
}

fn factory(depth: usize, fail_deal: bool) -> (GraphFactory, LiveDecisionHandle) {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let (outer, _) = request(start, end);
    (
        GraphFactory {
            calendar: Arc::new(SharedInnerCalendar::new(start, end)),
            account: Arc::new(Mutex::new(Account::new(InfinitePosition, false))),
            decisions: (0..depth).map(|_| request(start, end).0).collect(),
            levels: Arc::default(),
            events: Arc::default(),
            posted: Arc::default(),
            fail_deal,
            fail_assembly: false,
        },
        outer,
    )
}

#[test]
fn native_recursive_driver_runs_two_nested_levels_and_retains_each_original_decision() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/recursive_nested_protocol.py"
            ),
            r"D:\code\github\qlib\qlib\backtest\executor.py",
            r"D:\code\github\qlib\qlib\rl\order_execution\simulator_qlib.py",
            "42",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    let (mut factory, original) = factory(2, false);
    let expected = [
        original.clone(),
        factory.decisions[0].clone(),
        factory.decisions[1].clone(),
    ];
    let levels = factory.levels.clone();
    let posted = factory.posted.clone();
    let calendar = factory.calendar.clone();
    let account = factory.account.clone();
    let assembly = factory.assemble(&original, 0).unwrap();
    let mut driver = LiveRecursiveStrategyDriver::new_owned(LiveNestedExecutor::new(
        assembly.calendar,
        assembly.config,
        assembly.run,
    ));
    assert!(matches!(
        driver.advance(None).unwrap(),
        LiveRecursiveStrategyEvent::StrategyPrompt(_)
    ));
    assert_eq!(driver.decisions().len(), 2);
    let names = ["outer", "middle", "atomic"];
    let mut first: Vec<_> = driver
        .decisions()
        .iter()
        .map(|seen| {
            names[expected
                .iter()
                .position(|actual| Arc::ptr_eq(seen, actual))
                .unwrap()]
        })
        .collect();
    first.push("Proxy");
    assert_eq!(json!(first), source["first"]);
    assert_eq!(source["received"], json!([42.0]));
    assert!(account.try_lock().is_ok());
    assert_eq!(NestedCalendar::trade_step(&*calendar).unwrap(), 0);
    // Whole graph can be moved while suspended, without borrowing a separate executor.
    let mut driver = std::thread::spawn(move || {
        let LiveRecursiveStrategyEvent::Complete(done) = driver.advance(Some(42.0)).unwrap() else {
            panic!("completion expected")
        };
        assert_eq!(done.executions.lock().unwrap().len(), 1);
        let child_rows = posted.lock().unwrap().clone().unwrap();
        assert!(!Arc::ptr_eq(&done.executions, &child_rows));
        assert!(Arc::ptr_eq(
            &done.executions.lock().unwrap()[0],
            &child_rows.lock().unwrap()[0]
        ));
        driver
    })
    .join()
    .unwrap();
    assert_eq!(*levels.lock().unwrap(), [0, 1]);
    assert_eq!(driver.decisions().len(), expected.len());
    let mut second: Vec<_> = driver
        .decisions()
        .iter()
        .skip(2)
        .map(|seen| {
            names[expected
                .iter()
                .position(|actual| Arc::ptr_eq(seen, actual))
                .unwrap()]
        })
        .collect();
    second.push("Complete");
    assert_eq!(json!(second), source["second"]);
    for (seen, actual) in driver.decisions().iter().zip(expected) {
        assert!(Arc::ptr_eq(seen, &actual));
    }
    assert_eq!(NestedCalendar::trade_step(&*calendar).unwrap(), 1);
    assert_eq!(driver.advance(None).err(), Some(LiveNestedError::Complete));
    driver.close().unwrap();
    assert_eq!(driver.decisions().len(), 3);
}

#[test]
fn native_configured_session_rejects_overlap_releases_close_and_rebuilds_after_failure() {
    let (factory, original) = factory(1, true);
    let events = factory.events.clone();
    let levels = factory.levels.clone();
    let mut child = LiveConfiguredChildSession::new(Box::new(factory));
    assert!(child.resume(NestedExecutorResume::Continue).is_err());
    assert!(matches!(
        child.begin(original.clone(), 7).unwrap(),
        LiveNestedInnerProgress::Suspended(_)
    ));
    assert!(
        child
            .begin(original.clone(), 8)
            .err()
            .unwrap()
            .message
            .contains("already active")
    );
    assert!(matches!(
        child.resume(NestedExecutorResume::Continue).unwrap(),
        LiveNestedInnerProgress::Suspended(LiveNestedControlEvent::StrategyPrompt(_))
    ));
    child.close().unwrap();
    child.close().unwrap();
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|&&s| s == "close")
            .count(),
        1
    );
    assert!(child.resume(NestedExecutorResume::Continue).is_err());
    let _ = child.begin(original.clone(), 9).unwrap();
    let _ = child.resume(NestedExecutorResume::Continue).unwrap();
    let _ = child
        .resume(NestedExecutorResume::Action(Some(42.0)))
        .unwrap();
    assert!(
        child
            .resume(NestedExecutorResume::Continue)
            .err()
            .unwrap()
            .message
            .contains("deal failed")
    );
    assert!(
        child
            .resume(NestedExecutorResume::Continue)
            .err()
            .unwrap()
            .message
            .contains("not active")
    );
    let _ = child.begin(original, 10).unwrap();
    child.close().unwrap();
    assert_eq!(*levels.lock().unwrap(), [7, 9, 10]);
}

#[test]
fn native_recursive_borrowed_driver_and_drop_close_only_the_suspended_strategy() {
    for explicit in [false, true] {
        let (mut factory, original) = factory(2, false);
        let events = factory.events.clone();
        let calendar = factory.calendar.clone();
        let assembly = factory.assemble(&original, 0).unwrap();
        let mut executor =
            LiveNestedExecutor::new(assembly.calendar, assembly.config, assembly.run);
        {
            let mut driver = LiveRecursiveStrategyDriver::new(&mut executor);
            assert!(matches!(
                driver.advance(None).unwrap(),
                LiveRecursiveStrategyEvent::StrategyPrompt(_)
            ));
            if explicit {
                driver.close().unwrap();
                driver.close().unwrap();
            }
        }
        drop(executor);
        assert_eq!(NestedCalendar::trade_step(&*calendar).unwrap(), 0);
        let events = events.lock().unwrap();
        assert_eq!(events.iter().filter(|&&s| s == "close").count(), 1);
        assert!(!events.contains(&"post") && !events.contains(&"account"));
    }
}

#[test]
fn native_child_assembly_failure_does_not_install_an_active_graph() {
    let (mut factory, original) = factory(1, false);
    factory.fail_assembly = true;
    let levels = factory.levels.clone();
    let mut child = LiveConfiguredChildSession::new(Box::new(factory));
    for level in [1, 2] {
        assert_eq!(
            child.begin(original.clone(), level).err().unwrap().message,
            "assembly failed"
        );
    }
    assert!(
        child
            .resume(NestedExecutorResume::Continue)
            .err()
            .unwrap()
            .message
            .contains("not active")
    );
    child.close().unwrap();
    assert_eq!(*levels.lock().unwrap(), [1, 2]);
}
