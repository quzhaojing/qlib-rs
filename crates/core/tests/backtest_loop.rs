use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    Account, BacktestLoop, BacktestLoopBackend, BacktestLoopError, BacktestLoopEvent,
    BacktestReports, InfinitePosition, NestedControlEvent, NestedExecutorResume,
    NestedInnerProgress, NestedStrategyPrompt, Order, OrderDecision, OrderDir, OwnedOrderExecution,
    SaoeCalendar, SaoeOrderFactory, SaoePluginError, SharedOrderExecution, SingleOrderStrategy,
    TrackedOrderDecision, collect_backtest_reports,
};
use serde_json::{Value, json};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
#[error("{0}")]
struct Failure(&'static str);

fn time() -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
}
struct Calendar;
impl SaoeCalendar for Calendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        panic!("not used")
    }
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        Ok((time(), time()))
    }
}
struct Orders;
impl SaoeOrderFactory for Orders {
    fn create(
        &mut self,
        id: &str,
        amount: Option<f64>,
        direction: OrderDir,
    ) -> Result<Order, SaoePluginError> {
        Ok(Order::new(id, amount.unwrap(), direction, None, None))
    }
}

struct Backend<'a> {
    events: Arc<Mutex<Vec<Value>>>,
    failures: Vec<&'static str>,
    steps: usize,
    step: usize,
    prompt: bool,
    immediate: bool,
    pending: Option<Box<dyn OrderDecision>>,
    previous: Option<SharedOrderExecution>,
    previous_list: Option<domain_core::nested_executor::SharedNestedResult>,
    strategy: SingleOrderStrategy,
    account: &'a Account,
}
impl<'a> Backend<'a> {
    fn new(account: &'a Account, failures: Vec<&'static str>, steps: usize) -> Self {
        Self {
            events: Arc::default(),
            failures,
            steps,
            step: 0,
            prompt: false,
            immediate: false,
            pending: None,
            previous: None,
            previous_list: None,
            account,
            strategy: SingleOrderStrategy::new(
                Order::new("A", 7.0, OrderDir::Buy, None, None),
                None,
            ),
        }
    }
    fn stage(&self, name: &'static str, values: Vec<Value>) -> Result<(), Failure> {
        let mut event = vec![json!(name)];
        event.extend(values);
        self.events.lock().unwrap().push(json!(event));
        if self.failures.contains(&name) {
            Err(Failure(name))
        } else {
            Ok(())
        }
    }
    fn complete(&mut self, value: f64) -> Result<NestedInnerProgress, Failure> {
        let decision = self.pending.take().unwrap();
        self.stage("resume", vec![json!(value)])?;
        self.step += 1;
        let execution = OwnedOrderExecution {
            order: decision.orders()[0].clone(),
            trade_value: value,
            trade_price: 1.0,
            trade_cost: 0.0,
        }
        .into_shared();
        self.previous = Some(Arc::clone(&execution));
        let executions = Arc::new(Mutex::new(vec![execution]));
        self.previous_list = Some(Arc::clone(&executions));
        Ok(NestedInnerProgress::Complete {
            decision,
            executions,
        })
    }
}
impl BacktestLoopBackend for Backend<'_> {
    type Error = Failure;
    type Reports = BacktestReports;
    fn reset_executor(&mut self, start: NaiveDateTime, end: NaiveDateTime) -> Result<(), Failure> {
        assert_eq!((start, end), (time(), time()));
        self.stage("reset", vec![])
    }
    fn reset_strategy(&mut self) -> Result<(), Failure> {
        self.stage("level", vec![])?;
        self.stage("strategy_reset", vec![])
    }
    fn trade_len(&mut self) -> Result<i64, Failure> {
        self.stage("trade_len", vec![])?;
        Ok(i64::try_from(self.steps).unwrap())
    }
    fn enter_progress(&mut self, total: i64) -> Result<(), Failure> {
        assert_eq!(usize::try_from(total).unwrap(), self.steps);
        self.stage("progress_enter", vec![])
    }
    fn finished(&mut self) -> Result<bool, Failure> {
        self.stage("finished", vec![])?;
        Ok(self.step == self.steps)
    }
    fn generate(
        &mut self,
        previous: Option<&domain_core::nested_executor::SharedNestedResult>,
    ) -> Result<Box<dyn OrderDecision>, Failure> {
        if let Some(previous) = previous {
            assert!(Arc::ptr_eq(previous, self.previous_list.as_ref().unwrap()));
        }
        let snapshot = previous.map(|rows| rows.lock().unwrap().clone());
        let previous = snapshot.as_deref();
        if let Some(previous) = previous {
            assert_eq!(previous.len(), 1);
            assert!(Arc::ptr_eq(
                previous.first().unwrap(),
                self.previous.as_ref().unwrap()
            ));
        }
        self.stage(
            "generate",
            vec![json!(previous.map(|p| {
                p.iter().map(|e| e.trade_value).collect::<Vec<_>>()
            }))],
        )?;
        Ok(Box::new(
            self.strategy
                .generate_trade_decision(previous, &mut Orders, &Calendar)
                .unwrap(),
        ))
    }
    fn begin_collection(
        &mut self,
        decision: Box<dyn OrderDecision>,
    ) -> Result<NestedInnerProgress, Failure> {
        self.stage("begin", vec![])?;
        assert_eq!(decision.orders()[0].amount().to_bits(), 7.0_f64.to_bits());
        let tracked = TrackedOrderDecision {
            orders: decision.orders().to_vec(),
            start_time: decision.start_time(),
            end_time: decision.end_time(),
            has_trade_range: false,
        };
        self.pending = Some(decision);
        self.prompt = false;
        if self.immediate {
            return self.complete(3.0);
        }
        Ok(NestedInnerProgress::Suspended(
            NestedControlEvent::TrackedDecision(tracked),
        ))
    }
    fn resume_collection(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, Failure> {
        if !self.prompt {
            self.prompt = true;
            return Ok(NestedInnerProgress::Suspended(
                NestedControlEvent::StrategyPrompt(NestedStrategyPrompt {
                    kind: "action".to_owned(),
                    schema_version: 1,
                    payload: vec![],
                }),
            ));
        }
        let NestedExecutorResume::Action(Some(value)) = input else {
            panic!("test needs an action")
        };
        self.complete(value)
    }
    fn close_collection(&mut self) -> Result<(), Failure> {
        self.pending.take();
        self.stage("child_close", vec![])
    }
    fn post_execute(
        &mut self,
        executions: &domain_core::nested_executor::SharedNestedResult,
    ) -> Result<(), Failure> {
        assert!(Arc::ptr_eq(
            executions,
            self.previous_list.as_ref().unwrap()
        ));
        let executions = executions.lock().unwrap();
        assert!(Arc::ptr_eq(&executions[0], self.previous.as_ref().unwrap()));
        self.stage(
            "post",
            vec![json!(
                executions.iter().map(|e| e.trade_value).collect::<Vec<_>>()
            )],
        )
    }
    fn update_progress(&mut self) -> Result<(), Failure> {
        self.stage("progress_update", vec![])
    }
    fn finalize_strategy(&mut self) -> Result<(), Failure> {
        self.stage("finalize", vec![])
    }
    fn close_progress(&mut self) -> Result<(), Failure> {
        self.stage("progress_close", vec![])
    }
    fn collect_reports(&mut self) -> Result<Self::Reports, Failure> {
        self.stage("reports", vec![])?;
        Ok(collect_backtest_reports([("day", self.account)]).unwrap())
    }
}

fn describe(error: BacktestLoopError<Failure>) -> (Option<String>, Option<String>) {
    assert!(!error.to_string().is_empty());
    match error {
        BacktestLoopError::Backend(e) => (Some(e.to_string()), None),
        BacktestLoopError::Cleanup { original, cleanup } => {
            (Some(cleanup.to_string()), Some(original.to_string()))
        }
        _ => panic!("unexpected protocol error"),
    }
}

#[test]
fn outer_generator_matches_source_across_rounds_failures_and_cleanup() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/backtest_loop_contract.py"
            ),
            r"D:\code\github\qlib\qlib\backtest\backtest.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let source: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        source["wrapper"],
        json!({"events":["start","resume"],"portfolio":{"day":1},"indicator":{"day":2}})
    );
    let stages = [
        "reset",
        "level",
        "strategy_reset",
        "trade_len",
        "progress_enter",
        "finished",
        "generate",
        "begin",
        "resume",
        "post",
        "progress_update",
        "finalize",
        "progress_close",
        "reports",
    ];
    let mut cases = vec![
        ("normal", vec![], 2, true, false),
        ("empty", vec![], 0, true, false),
        ("no_reports", vec![], 2, false, false),
        ("cancel", vec![], 2, true, true),
        (
            "double_failure",
            vec!["post", "progress_close"],
            2,
            true,
            false,
        ),
        (
            "cancel_failure",
            vec!["child_close", "progress_close"],
            2,
            true,
            true,
        ),
    ];
    cases.extend(
        stages
            .into_iter()
            .map(|stage| (stage, vec![stage], 2, true, false)),
    );
    for (name, failures, steps, reports, cancel) in cases {
        check_case(name, failures, steps, reports, cancel, &source);
    }
}

fn check_case(
    name: &str,
    failures: Vec<&'static str>,
    steps: usize,
    reports: bool,
    cancel: bool,
    source: &Value,
) {
    let account = Account::new(InfinitePosition, false);
    let backend = Backend::new(&account, failures, steps);
    let events = Arc::clone(&backend.events);
    let mut runner = BacktestLoop::new(backend, time(), time(), reports);
    assert!(events.lock().unwrap().is_empty());
    assert!(runner.reports().is_none());
    let mut yielded = Vec::new();
    let mut input = NestedExecutorResume::Continue;
    let mut action = 3.0;
    let mut failure = (None, None);
    loop {
        match runner.resume(input) {
            Ok(BacktestLoopEvent::Complete) => {
                assert!(matches!(
                    runner.resume(input),
                    Err(BacktestLoopError::AlreadyComplete)
                ));
                break;
            }
            Ok(BacktestLoopEvent::Suspended(event)) => {
                assert!(runner.reports().is_none());
                match event {
                    NestedControlEvent::TrackedDecision(_) => {
                        yielded.push("decision");
                        input = NestedExecutorResume::Action(Some(999.0));
                    }
                    NestedControlEvent::StrategyPrompt(_) => {
                        yielded.push("prompt");
                        input = NestedExecutorResume::Action(Some(action));
                        action += 1.0;
                    }
                }
                if cancel {
                    if let Err(error) = runner.close() {
                        failure = describe(error);
                    }
                    break;
                }
            }
            Err(error) => {
                failure = describe(error);
                assert!(matches!(
                    runner.resume(input),
                    Err(BacktestLoopError::Failed)
                ));
                break;
            }
        }
    }
    if let Some(reports) = runner.reports() {
        assert!(reports.portfolio.is_empty());
        assert!(Arc::ptr_eq(
            &reports.indicators["1day"].indicator,
            account.indicator()
        ));
    }
    assert_eq!(
        json!({"events": *events.lock().unwrap(), "yielded": yielded,
            "published": runner.reports().is_some(), "error": failure.0, "context": failure.1}),
        source[name],
        "{name}"
    );
    let before = events.lock().unwrap().clone();
    runner.close().unwrap();
    runner.close().unwrap();
    drop(runner);
    assert_eq!(*events.lock().unwrap(), before);
}

#[test]
fn immediate_results_initial_protocol_and_drop_cleanup_are_explicit() {
    let account = Account::new(InfinitePosition, false);
    let mut backend = Backend::new(&account, vec![], 2);
    backend.immediate = true;
    let mut runner = BacktestLoop::new(backend, time(), time(), true);
    assert!(matches!(
        runner.resume(NestedExecutorResume::Action(None)),
        Err(BacktestLoopError::InvalidInitialAction)
    ));
    assert!(matches!(
        runner.resume(NestedExecutorResume::Continue),
        Ok(BacktestLoopEvent::Complete)
    ));
    assert!(runner.reports().is_some());
    for failures in [
        vec![],
        vec!["child_close"],
        vec!["progress_close"],
        vec!["child_close", "progress_close"],
    ] {
        let backend = Backend::new(&account, failures.clone(), 2);
        let events = Arc::clone(&backend.events);
        let mut runner = BacktestLoop::new(backend, time(), time(), false);
        assert!(matches!(
            runner.resume(NestedExecutorResume::Continue),
            Ok(BacktestLoopEvent::Suspended(_))
        ));
        drop(runner);
        let events = events.lock().unwrap();
        assert_eq!(
            &events[events.len() - 2..],
            [json!(["child_close"]), json!(["progress_close"])]
        );
    }
    let backend = Backend::new(&account, vec![], 2);
    let events = Arc::clone(&backend.events);
    let mut runner = BacktestLoop::new(backend, time(), time(), true);
    runner.close().unwrap();
    assert!(matches!(
        runner.resume(NestedExecutorResume::Continue),
        Err(BacktestLoopError::Closed)
    ));
    assert!(events.lock().unwrap().is_empty());
}
