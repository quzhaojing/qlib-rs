use std::{
    cell::Cell,
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    ConfiguredNestedChildSession, IndicatorConfig, NestedBarEnd, NestedCalendar,
    NestedCalendarError, NestedChildAssembly, NestedChildAssemblyFactory, NestedChildSession,
    NestedCollection, NestedControlEvent, NestedDecisionTracking, NestedDecisionUpdate,
    NestedExecutorAccount, NestedExecutorAccountError, NestedExecutorResume,
    NestedExecutorReturnSink, NestedExecutorReturnSinkError, NestedInnerControlMode,
    NestedInnerExecutor, NestedInnerExecutorError, NestedInnerProgress, NestedLevelBinding,
    NestedLevelBindingError, NestedOuterDecision, NestedOuterDecisionError, NestedStrategy,
    NestedStrategyError, NestedStrategyProgress, NestedStrategyPrompt, NumpyOrderIndicator, Order,
    OrderDecision, OrderDir, OrderIndicatorAggregationConfig, OrderTradeDecision,
    OwnedOrderExecution, RecursiveNestedInnerAdapter, RecursiveStrategyDriver,
    RecursiveStrategyEvent, ResumableNestedConfig, ResumableNestedEvent, ResumableNestedExecutor,
    ResumableNestedExecutorError, ResumableNestedRun, SharedOrderExecution, TrackedOrderDecision,
};
use serde_json::{Value, json};

#[path = "support/resumable_shared_list_cases.rs"]
mod resumable_shared_list_cases;

fn time(minute: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::minutes(minute)
}

#[derive(Default)]
struct State {
    order: domain_core::SharedOrderIndicator<NumpyOrderIndicator>,
    events: Vec<String>,
    step: i64,
    outer_time_reads: usize,
    fail: Option<&'static str>,
    received: Vec<Option<f64>>,
    resumes: Vec<NestedExecutorResume>,
    sink_count: usize,
}

type SharedState = Arc<Mutex<State>>;

fn event(state: &SharedState, name: impl Into<String>) -> bool {
    let name = name.into();
    let mut state = state.lock().unwrap();
    state.events.push(name.clone());
    state.fail == Some(name.as_str())
}

fn calendar_error(message: &str) -> NestedCalendarError {
    NestedCalendarError {
        message: message.to_owned(),
    }
}

struct OuterCalendar(SharedState);

impl NestedCalendar for OuterCalendar {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        Ok(false)
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        Ok(1)
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        Ok(0)
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        let name = {
            let mut state = self.0.lock().unwrap();
            state.outer_time_reads += 1;
            if state.outer_time_reads == 1 {
                "outer_time"
            } else {
                "bar_time"
            }
        };
        if event(&self.0, name) {
            return Err(calendar_error(name));
        }
        Ok((time(0), time(59)))
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        if event(&self.0, "outer_step") {
            return Err(calendar_error("outer_step"));
        }
        Ok(())
    }
}

struct Inner {
    state: SharedState,
    len: i64,
}

impl NestedCalendar for Inner {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        if event(&self.state, "finished") {
            return Err(calendar_error("finished"));
        }
        Ok(self.state.lock().unwrap().step >= self.len)
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        if event(&self.state, "trade_len") {
            return Err(calendar_error("trade_len"));
        }
        Ok(self.len)
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        if event(&self.state, "trade_step") {
            return Err(calendar_error("trade_step"));
        }
        Ok(self.state.lock().unwrap().step)
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        if event(&self.state, "inner_time") {
            return Err(calendar_error("inner_time"));
        }
        let step = self.state.lock().unwrap().step;
        Ok((time(step), time(step + 1)))
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        if event(&self.state, "step") {
            return Err(calendar_error("step"));
        }
        self.state.lock().unwrap().step += 1;
        Ok(())
    }
}

impl NestedInnerExecutor for Inner {
    fn reset_window(
        &mut self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        if event(&self.state, "reset") {
            return Err(inner_error("reset"));
        }
        self.state.lock().unwrap().step = 0;
        Ok(())
    }

    fn collect_data(
        &mut self,
        decision: &mut dyn OrderDecision,
        level: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError> {
        if event(&self.state, format!("collect:{level}"))
            || self.state.lock().unwrap().fail == Some("collect")
        {
            return Err(inner_error("collect"));
        }
        let execution = OwnedOrderExecution {
            order: decision.orders()[0].clone(),
            trade_value: 20.0,
            trade_cost: 0.0,
            trade_price: 10.0,
        }
        .into_shared();
        self.state.lock().unwrap().step += 1;
        Ok(vec![execution])
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<domain_core::SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError>
    {
        if event(&self.state, "handle") {
            return Err(inner_error("handle"));
        }
        Ok(self.state.lock().unwrap().order.clone())
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        if event(&self.state, "snapshot") {
            return Err(inner_error("snapshot"));
        }
        Ok(NumpyOrderIndicator::default())
    }
}

fn tracked(decision: &dyn OrderDecision) -> TrackedOrderDecision {
    TrackedOrderDecision {
        orders: decision.orders().to_vec(),
        start_time: decision.start_time(),
        end_time: decision.end_time(),
        has_trade_range: decision.trade_range().is_some(),
    }
}

struct RecursiveChild {
    state: SharedState,
    stage: u8,
    decision: Option<Box<dyn OrderDecision>>,
}

impl NestedChildSession for RecursiveChild {
    fn close(&mut self) -> Result<(), NestedInnerExecutorError> {
        self.decision = None;
        self.stage = 0;
        if event(&self.state, "child_close") {
            return Err(inner_error("child_close"));
        }
        Ok(())
    }

    fn begin(
        &mut self,
        decision: Box<dyn OrderDecision>,
        level: usize,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        if event(&self.state, format!("child_begin:{level}"))
            || self.state.lock().unwrap().fail == Some("child_begin")
        {
            return Err(inner_error("child_begin"));
        }
        if self.stage != 0 {
            return Err(inner_error("child already active"));
        }
        let snapshot = tracked(&*decision);
        self.decision = Some(decision);
        self.stage = 1;
        Ok(NestedInnerProgress::Suspended(
            NestedControlEvent::TrackedDecision(snapshot),
        ))
    }

    fn resume(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        self.state.lock().unwrap().resumes.push(input);
        if event(&self.state, "child_resume") {
            return Err(inner_error("child_resume"));
        }
        match self.stage {
            1 => {
                self.stage = 2;
                Ok(NestedInnerProgress::Suspended(
                    NestedControlEvent::StrategyPrompt(prompt(0)),
                ))
            }
            2 => {
                let NestedExecutorResume::Action(volume) = input else {
                    return Err(inner_error("missing child action"));
                };
                self.state.lock().unwrap().received.push(volume);
                self.stage = 3;
                Ok(NestedInnerProgress::Suspended(
                    NestedControlEvent::TrackedDecision(tracked(&*order(7.0))),
                ))
            }
            3 => {
                self.stage = 4;
                self.state.lock().unwrap().step += 1;
                let decision = self
                    .decision
                    .take()
                    .ok_or_else(|| inner_error("missing child decision"))?;
                Ok(NestedInnerProgress::Complete {
                    decision,
                    executions: Arc::new(Mutex::new(vec![
                        OwnedOrderExecution {
                            order: order(7.0).orders()[0].clone(),
                            trade_value: 70.0,
                            trade_cost: 0.0,
                            trade_price: 10.0,
                        }
                        .into_shared(),
                    ])),
                })
            }
            _ => Err(inner_error("child is not suspended")),
        }
    }
}

struct FrameworkSuspendingInner {
    order: domain_core::SharedOrderIndicator<NumpyOrderIndicator>,
    finished: Cell<bool>,
    decision: Option<Box<dyn OrderDecision>>,
}

impl NestedCalendar for FrameworkSuspendingInner {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        Ok(self.finished.get())
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        Ok(1)
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        Ok(0)
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        Ok((time(0), time(1)))
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        self.finished.set(true);
        Ok(())
    }
}

impl NestedInnerExecutor for FrameworkSuspendingInner {
    fn reset_window(
        &mut self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        self.finished.set(false);
        Ok(())
    }

    fn collect_data(
        &mut self,
        _decision: &mut dyn OrderDecision,
        _level: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError> {
        Err(inner_error(
            "framework suspension requires owned collection",
        ))
    }

    fn begin_collect_data(
        &mut self,
        decision: Box<dyn OrderDecision>,
        _level: usize,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        self.decision = Some(decision);
        Ok(NestedInnerProgress::Suspended(
            NestedControlEvent::StrategyPrompt(prompt(0)),
        ))
    }

    fn resume_collect_data(
        &mut self,
        _input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        self.finished.set(true);
        Ok(NestedInnerProgress::Complete {
            decision: self
                .decision
                .take()
                .ok_or_else(|| inner_error("missing framework decision"))?,
            executions: Arc::default(),
        })
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<domain_core::SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError>
    {
        Ok(self.order.clone())
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        Ok(NumpyOrderIndicator::default())
    }
}

fn inner_error(message: &str) -> NestedInnerExecutorError {
    NestedInnerExecutorError {
        message: message.to_owned(),
    }
}

struct Binding(SharedState);

impl NestedLevelBinding for Binding {
    fn bind_inner(
        &mut self,
        _inner: &dyn NestedInnerExecutor,
    ) -> Result<(), NestedLevelBindingError> {
        if event(&self.0, "bind") {
            return Err(NestedLevelBindingError {
                message: "bind".to_owned(),
            });
        }
        Ok(())
    }
}

struct Outer {
    state: SharedState,
    decision: OrderTradeDecision,
    empty: bool,
    range: Option<(i64, i64)>,
    replace: bool,
}

impl NestedOuterDecision for Outer {
    fn order_decision(&self) -> &dyn OrderDecision {
        &self.decision
    }

    fn order_decision_mut(&mut self) -> &mut dyn OrderDecision {
        &mut self.decision
    }

    fn update(
        &mut self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<NestedDecisionUpdate, NestedOuterDecisionError> {
        if event(&self.state, "update") {
            return Err(outer_error("update"));
        }
        Ok(if self.replace {
            NestedDecisionUpdate::Replaced
        } else {
            NestedDecisionUpdate::Unchanged
        })
    }

    fn is_empty(&self) -> Result<bool, NestedOuterDecisionError> {
        if event(&self.state, "empty") {
            return Err(outer_error("empty"));
        }
        Ok(self.empty)
    }

    fn range_limit(
        &self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<Option<(i64, i64)>, NestedOuterDecisionError> {
        if event(&self.state, "range") {
            return Err(outer_error("range"));
        }
        Ok(self.range)
    }

    fn modify_inner_decision(
        &self,
        _decision: &mut dyn OrderDecision,
    ) -> Result<(), NestedOuterDecisionError> {
        if event(&self.state, "modify") {
            return Err(outer_error("modify"));
        }
        Ok(())
    }
}

fn outer_error(message: &str) -> NestedOuterDecisionError {
    NestedOuterDecisionError {
        message: message.to_owned(),
    }
}

fn order(amount: f64) -> Box<dyn OrderDecision> {
    Box::new(OrderTradeDecision::from_orders(
        vec![Order::new(
            "A",
            amount,
            OrderDir::Buy,
            Some(time(0)),
            Some(time(1)),
        )],
        time(0),
        time(1),
        None,
    ))
}

struct ImmediateStrategy(SharedState);

impl NestedStrategy for ImmediateStrategy {
    fn reset(&mut self, _outer: &dyn NestedOuterDecision) -> Result<(), NestedStrategyError> {
        fail_strategy(&self.0, "strategy_reset")
    }

    fn alter_outer_decision(
        &mut self,
        _outer: &mut dyn NestedOuterDecision,
    ) -> Result<(), NestedStrategyError> {
        fail_strategy(&self.0, "alter")
    }

    fn generate_trade_decision(
        &mut self,
        _previous: Option<&[SharedOrderExecution]>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        fail_strategy(&self.0, "begin")?;
        Ok(order(2.0))
    }

    fn post_execute(
        &mut self,
        _executions: &[SharedOrderExecution],
    ) -> Result<(), NestedStrategyError> {
        fail_strategy(&self.0, "post")
    }

    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        fail_strategy(&self.0, "upper")
    }
}

struct ProxyStrategy {
    state: SharedState,
    prompts: usize,
    resumed: usize,
}

impl NestedStrategy for ProxyStrategy {
    fn close_trade_decision(&mut self) -> Result<(), NestedStrategyError> {
        fail_strategy(&self.state, "strategy_close")
    }

    fn reset(&mut self, _outer: &dyn NestedOuterDecision) -> Result<(), NestedStrategyError> {
        fail_strategy(&self.state, "strategy_reset")
    }

    fn alter_outer_decision(
        &mut self,
        _outer: &mut dyn NestedOuterDecision,
    ) -> Result<(), NestedStrategyError> {
        fail_strategy(&self.state, "alter")
    }

    fn generate_trade_decision(
        &mut self,
        _previous: Option<&[SharedOrderExecution]>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        Err(strategy_error("immediate method must not run"))
    }

    fn begin_trade_decision(
        &mut self,
        _previous: Option<&[SharedOrderExecution]>,
    ) -> Result<NestedStrategyProgress, NestedStrategyError> {
        fail_strategy(&self.state, "begin")?;
        Ok(NestedStrategyProgress::Suspended(prompt(self.resumed)))
    }

    fn resume_trade_decision(
        &mut self,
        execution_volume: Option<f64>,
    ) -> Result<NestedStrategyProgress, NestedStrategyError> {
        fail_strategy(&self.state, "resume")?;
        self.state.lock().unwrap().received.push(execution_volume);
        self.resumed += 1;
        if self.resumed < self.prompts {
            Ok(NestedStrategyProgress::Suspended(prompt(self.resumed)))
        } else {
            Ok(NestedStrategyProgress::Ready(order(
                execution_volume.unwrap_or(0.0),
            )))
        }
    }

    fn post_execute(
        &mut self,
        _executions: &[SharedOrderExecution],
    ) -> Result<(), NestedStrategyError> {
        fail_strategy(&self.state, "post")
    }

    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        fail_strategy(&self.state, "upper")
    }
}

fn prompt(index: usize) -> NestedStrategyPrompt {
    NestedStrategyPrompt {
        kind: "proxy-saoe".to_owned(),
        schema_version: 1,
        payload: vec![u8::try_from(index).unwrap()],
    }
}

fn fail_strategy(shared: &SharedState, boundary: &str) -> Result<(), NestedStrategyError> {
    if event(shared, boundary) {
        return Err(strategy_error(boundary));
    }
    Ok(())
}

fn strategy_error(message: &str) -> NestedStrategyError {
    NestedStrategyError {
        message: message.to_owned(),
    }
}

struct Account(SharedState);

impl NestedExecutorAccount for Account {
    fn settle_start(&mut self, settle_type: &str) -> Result<(), NestedExecutorAccountError> {
        let stage = format!("settle_start:{settle_type}");
        if event(&self.0, &stage) || self.0.lock().unwrap().fail == Some("settle_start") {
            return Err(account_error("settle_start"));
        }
        Ok(())
    }

    fn update_bar_end(&mut self, bar: NestedBarEnd<'_>) -> Result<(), NestedExecutorAccountError> {
        assert_eq!(bar.inner_order_indicators.len(), bar.steps.len());
        if event(&self.0, "bar") {
            return Err(account_error("bar"));
        }
        Ok(())
    }

    fn settle_commit(&mut self) -> Result<(), NestedExecutorAccountError> {
        if event(&self.0, "commit") {
            return Err(account_error("commit"));
        }
        Ok(())
    }
}

fn account_error(message: &str) -> NestedExecutorAccountError {
    NestedExecutorAccountError {
        message: message.to_owned(),
    }
}

struct Sink {
    state: SharedState,
}

impl NestedExecutorReturnSink for Sink {
    fn store_execute_result(
        &mut self,
        executions: &domain_core::nested_executor::SharedNestedResult,
    ) -> Result<(), NestedExecutorReturnSinkError> {
        if event(&self.state, "sink") {
            return Err(NestedExecutorReturnSinkError {
                message: "sink".to_owned(),
            });
        }
        self.state.lock().unwrap().sink_count = executions.lock().unwrap().len();
        Ok(())
    }
}

fn config(track: bool, settlement: &str) -> ResumableNestedConfig {
    ResumableNestedConfig {
        skip_empty_decision: true,
        align_range_limit: true,
        decision_tracking: NestedDecisionTracking {
            outer: track,
            inner: track,
        },
        settle_type: settlement.to_owned(),
        indicator_config: IndicatorConfig::default(),
        aggregation_config: OrderIndicatorAggregationConfig::default(),
        level: 0,
    }
}

fn expect_error<T>(
    result: Result<T, ResumableNestedExecutorError>,
) -> ResumableNestedExecutorError {
    match result {
        Ok(_) => panic!("expected resumable executor error"),
        Err(error) => error,
    }
}

fn expect_inner_error<T>(result: Result<T, NestedInnerExecutorError>) -> NestedInnerExecutorError {
    match result {
        Ok(_) => panic!("expected nested inner error"),
        Err(error) => error,
    }
}

fn outer_decision() -> OrderTradeDecision {
    OrderTradeDecision::from_orders(
        vec![Order::new(
            "A",
            1.0,
            OrderDir::Buy,
            Some(time(0)),
            Some(time(59)),
        )],
        time(0),
        time(59),
        None,
    )
}

fn run_immediate_policy(
    shared: &SharedState,
    policy: ResumableNestedConfig,
    len: i64,
    empty: bool,
    range: Option<(i64, i64)>,
) -> NestedCollection {
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(shared)));
    let binding = Binding(Arc::clone(shared));
    let inner = Inner {
        state: Arc::clone(shared),
        len,
    };
    let strategy = ImmediateStrategy(Arc::clone(shared));
    let outer = Outer {
        state: Arc::clone(shared),
        decision: outer_decision(),
        empty,
        range,
        replace: false,
    };
    let account = Account(Arc::clone(shared));
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        policy,
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: None,
        },
    );
    match machine.resume(NestedExecutorResume::Continue).unwrap() {
        ResumableNestedEvent::Complete(collection) => collection,
        ResumableNestedEvent::TrackedDecision(_) | ResumableNestedEvent::StrategyPrompt(_) => {
            panic!("policy harness must not suspend")
        }
    }
}

#[test]
fn resumable_collection_keeps_repeated_raw_bindings_after_graph_drop() {
    let state = Arc::new(Mutex::new(State::default()));
    let raw = state.lock().unwrap().order.clone();
    let collection = run_immediate_policy(&state, config(false, "None"), 2, false, None);
    assert_eq!(collection.inner_order_indicators().len(), 2);
    assert!(
        collection
            .inner_order_indicators()
            .iter()
            .all(|handle| Arc::ptr_eq(handle, &raw))
    );
    state.lock().unwrap().order = Arc::default();
    drop(state);
    assert_eq!(Arc::strong_count(&raw), 3);
    assert!(raw.try_write().is_ok());
    drop(collection);
    assert_eq!(Arc::strong_count(&raw), 1);
}

fn close_machine(state: &SharedState, recursive: bool) -> ResumableNestedExecutor {
    let lifecycle = Box::new(Inner {
        state: Arc::clone(state),
        len: 1,
    });
    let inner: Box<dyn NestedInnerExecutor> = if recursive {
        Box::new(RecursiveNestedInnerAdapter::new(
            lifecycle,
            Box::new(RecursiveChild {
                state: Arc::clone(state),
                stage: 0,
                decision: None,
            }),
        ))
    } else {
        lifecycle
    };
    let strategy: Box<dyn NestedStrategy> = if recursive {
        Box::new(ImmediateStrategy(Arc::clone(state)))
    } else {
        Box::new(ProxyStrategy {
            state: Arc::clone(state),
            prompts: 1,
            resumed: 0,
        })
    };
    ResumableNestedExecutor::new(
        Box::new(OuterCalendar(Arc::clone(state))),
        config(true, "cash"),
        ResumableNestedRun {
            level_binding: Box::new(Binding(Arc::clone(state))),
            inner,
            strategy,
            outer: Box::new(Outer {
                state: Arc::clone(state),
                decision: outer_decision(),
                empty: false,
                range: None,
                replace: false,
            }),
            account: Box::new(Account(Arc::clone(state))),
            return_sink: Some(Box::new(Sink {
                state: Arc::clone(state),
            })),
        },
    )
}

#[test]
fn close_matches_live_python_active_delegate_and_cleanup_failure_contract() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/nested_close_contract.py"
            ),
            r"D:\code\github\qlib\qlib\backtest\executor.py",
            r"D:\code\github\qlib\qlib\rl\order_execution\simulator_qlib.py",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut actual = Vec::new();
    for mode in ["outer_track", "strategy", "inner"] {
        for fail in [false, true] {
            let state = Arc::new(Mutex::new(State::default()));
            let mut machine = close_machine(&state, mode == "inner");
            assert!(matches!(
                machine.resume(NestedExecutorResume::Continue).unwrap(),
                ResumableNestedEvent::TrackedDecision(_)
            ));
            if mode != "outer_track" {
                machine.resume(NestedExecutorResume::Continue).unwrap();
            }
            if mode == "inner" {
                assert!(matches!(
                    machine.resume(NestedExecutorResume::Continue).unwrap(),
                    ResumableNestedEvent::StrategyPrompt(_)
                ));
            }
            let boundary = if mode == "inner" {
                "child_close"
            } else {
                "strategy_close"
            };
            {
                let mut shared = state.lock().unwrap();
                shared.events.clear();
                shared.fail = fail.then_some(boundary);
            }
            let error = machine.close().err();
            if fail && mode != "outer_track" {
                let expected = if mode == "inner" {
                    domain_core::NestedExecutorError::from(inner_error(boundary))
                } else {
                    domain_core::NestedExecutorError::from(strategy_error(boundary))
                };
                assert_eq!(
                    error,
                    Some(ResumableNestedExecutorError::Collection(expected))
                );
            } else {
                assert_eq!(error, None);
            }
            machine.close().unwrap();
            assert_eq!(
                expect_error(machine.resume(NestedExecutorResume::Continue)),
                ResumableNestedExecutorError::Closed
            );
            assert_eq!(
                expect_error(machine.resume(NestedExecutorResume::Continue)),
                ResumableNestedExecutorError::Closed
            );
            drop(machine);
            let shared = state.lock().unwrap();
            assert_eq!(shared.step, 0);
            assert_eq!(shared.sink_count, 0);
            let events: Vec<_> = shared
                .events
                .iter()
                .map(|name| name.replace("child_close", "inner_close"))
                .collect();
            actual.push(json!({"mode": mode, "fail": fail, "events": events, "error": error.map(|_| if mode == "inner" { "inner_close" } else { "strategy_close" })}));
        }
    }
    assert_eq!(Value::Array(actual), oracle);
}

#[test]
fn close_before_start_after_inner_tracking_and_drop_are_not_completion() {
    for advances in 0..=3 {
        for fail in [false, true] {
            let state = Arc::new(Mutex::new(State::default()));
            let mut machine = close_machine(&state, false);
            for _ in 0..advances {
                machine.resume(NestedExecutorResume::Continue).unwrap();
            }
            {
                let mut shared = state.lock().unwrap();
                shared.events.clear();
                shared.fail = fail.then_some("strategy_close");
            }
            drop(machine);
            let shared = state.lock().unwrap();
            assert_eq!(
                shared.events,
                if advances == 2 {
                    vec!["strategy_close"]
                } else {
                    vec![]
                }
            );
            assert_eq!(shared.step, 0);
            assert_eq!(shared.sink_count, 0);
            assert_eq!(Arc::strong_count(&state), 1);
        }
    }
    let state = Arc::new(Mutex::new(State::default()));
    let mut machine = close_machine(&state, false);
    machine.close().unwrap();
    assert_eq!(
        expect_error(machine.resume(NestedExecutorResume::Action(None))),
        ResumableNestedExecutorError::Closed
    );
    assert_eq!(
        ResumableNestedExecutorError::Closed.to_string(),
        "resumable nested executor has been closed"
    );
    let mut immediate = ImmediateStrategy(Arc::clone(&state));
    immediate.close_trade_decision().unwrap();
    let mut inner = Inner { state, len: 1 };
    inner.close_collect_data().unwrap();
}

#[test]
fn configured_child_close_releases_graph_and_allows_a_new_collection() {
    let observations = Arc::new(Mutex::new(AssemblyObservations::default()));
    let mut session = ConfiguredNestedChildSession::new(Box::new(AssemblyFactory {
        observations: Arc::clone(&observations),
        machine_failure: false,
    }));
    session.close().unwrap();
    for _ in 0..2 {
        session.begin(order(1.0), 0).unwrap();
        session.resume(NestedExecutorResume::Continue).unwrap();
        session.close().unwrap();
        session.close().unwrap();
        assert!(
            expect_inner_error(session.resume(NestedExecutorResume::Continue))
                .message
                .contains("not active")
        );
    }
    assert_eq!(observations.lock().unwrap().inputs.len(), 2);
}

#[test]
fn owned_and_borrowed_drivers_close_the_same_active_child_and_keep_tracking() {
    for owned in [false, true] {
        for fail in [false, true] {
            let state = Arc::new(Mutex::new(State::default()));
            let mut machine = close_machine(&state, true);
            let mut driver = if owned {
                RecursiveStrategyDriver::new_owned(machine)
            } else {
                RecursiveStrategyDriver::new(&mut machine)
            };
            assert!(matches!(
                driver.advance(None).unwrap(),
                RecursiveStrategyEvent::StrategyPrompt(_)
            ));
            let decisions = driver.decisions().to_vec();
            assert_eq!(decisions.len(), 2);
            {
                let mut shared = state.lock().unwrap();
                shared.events.clear();
                shared.fail = fail.then_some("child_close");
            }
            assert_eq!(driver.close().is_err(), fail);
            driver.close().unwrap();
            assert_eq!(driver.decisions(), decisions);
            assert_eq!(
                expect_error(driver.advance(None)),
                ResumableNestedExecutorError::Closed
            );
            drop(driver);
            assert_eq!(state.lock().unwrap().events, ["child_close"]);
        }
    }
}

struct CleanupAssemblyFactory(SharedState);

impl NestedChildAssemblyFactory for CleanupAssemblyFactory {
    fn assemble(
        &mut self,
        decision: &dyn OrderDecision,
        level: usize,
    ) -> Result<NestedChildAssembly, NestedInnerExecutorError> {
        let mut assembly = AssemblyFactory {
            observations: Arc::new(Mutex::new(AssemblyObservations::default())),
            machine_failure: false,
        }
        .assemble(decision, level)?;
        assembly.run.strategy = Box::new(ProxyStrategy {
            state: Arc::clone(&self.0),
            prompts: 1,
            resumed: 0,
        });
        Ok(assembly)
    }
}

#[test]
fn configured_child_cleanup_failure_releases_the_active_graph_without_retry() {
    let state = Arc::new(Mutex::new(State::default()));
    let mut session =
        ConfiguredNestedChildSession::new(Box::new(CleanupAssemblyFactory(Arc::clone(&state))));
    for fail in [true, false] {
        session.begin(order(1.0), 0).unwrap();
        session.resume(NestedExecutorResume::Continue).unwrap();
        {
            let mut shared = state.lock().unwrap();
            shared.events.clear();
            shared.fail = fail.then_some("strategy_close");
        }
        assert_eq!(Arc::strong_count(&state), 3);
        let result = session.close();
        if fail {
            assert!(
                expect_inner_error(result)
                    .message
                    .contains("strategy_close")
            );
        } else {
            result.unwrap();
        }
        session.close().unwrap();
        assert_eq!(state.lock().unwrap().events, ["strategy_close"]);
        assert_eq!(Arc::strong_count(&state), 2);
    }
    drop(session);
    assert_eq!(Arc::strong_count(&state), 1);
}

fn python_protocol() -> Value {
    let script = r"
import ast,json,sys
from types import GeneratorType
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());b=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='BaseExecutor');bc=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='collect_data');n=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='NestedExecutor');nc=next(x for x in n.body if isinstance(x,ast.FunctionDef) and x.name=='_collect_data')
for f in (bc,nc):
 f.returns=None
 for a in f.args.args:a.annotation=None
class NestedExecutor:pass
class BasePosition:ST_NO='None'
ns={'GeneratorType':GeneratorType,'NestedExecutor':NestedExecutor,'BasePosition':BasePosition,'get_start_end_idx':lambda c,d:(0,0)};exec(compile(ast.fix_missing_locations(ast.Module(body=[bc,nc],type_ignores=[])),p,'exec'),ns);e=[]
class D:
 def __init__(self,n):self.name=n
 def get_range_limit(self,default_value=None):return None
 def update(self,c):return None
 def empty(self):return False
 def mod_inner_decision(self,d):pass
class P:
 def settle_start(self,s):pass
 def settle_commit(self):pass
class I:
 def get_order_indicator(self,raw=True):return 'I'
class A:
 current_position=P()
 def update_bar_end(self,*a,**k):pass
 def get_trade_indicator(self):return I()
class C:
 def __init__(self):self.i=0
 def get_step_time(self):return (9,16)
 def get_trade_step(self):return self.i
 def step(self):self.i+=1
class X:
 collect_data=ns['collect_data'];track_data=True;_settle_type='None';trade_account=A();trade_calendar=C();trade_exchange=object();indicator_config={}
 def reset(self,**k):self.trade_calendar.i=0
 def get_level_infra(self):return 0
 def finished(self):return self.trade_calendar.i>=1
 def _collect_data(self,trade_decision,level=0):return ['R'],{'trade_info':[]}
class S:
 def reset(self,*a,**k):pass
 def alter_outer_trade_decision(self,d):return d
 def generate_trade_decision(self,p):v=yield self;e.append(v);return D('inner')
 def post_exe_step(self,r):pass
 def post_upper_level_exe_step(self):pass
class L:
 def set_sub_level_infra(self,i):pass
class E(NestedExecutor):
 collect_data=ns['collect_data'];_collect_data=ns['_collect_data'];track_data=True;_settle_type='None';trade_account=A();trade_calendar=C();trade_exchange=object();indicator_config={};inner_executor=X();inner_strategy=S();level_infra=L();_skip_empty_decision=True;_align_range_limit=True
 def _init_sub_trading(self,d):s,z=self.trade_calendar.get_step_time();self.inner_executor.reset(start_time=s,end_time=z);i=self.inner_executor.get_level_infra();self.level_infra.set_sub_level_infra(i);self.inner_strategy.reset(i,d)
 def _update_trade_decision(self,d):u=d.update(self.inner_executor.trade_calendar);return d if u is None else u
 def post_inner_exe_step(self,r):self.inner_strategy.post_exe_step(r)
g=E().collect_data(D('outer'));o=[['decision',next(g).name],['prompt',type(next(g)).__name__],['decision',g.send(7).name]]
try:g.send(999)
except StopIteration as x:o.append(['complete',len(x.value),e])
print(json.dumps(o,separators=(',',':')))
";
    let output = Command::new("python")
        .args([
            "-c",
            script,
            r"D:\code\github\qlib\qlib\backtest\executor.py",
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

fn python_recursive_protocol() -> Value {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/recursive_nested_protocol.py"
            ),
            r"D:\code\github\qlib\qlib\backtest\executor.py",
            r"D:\code\github\qlib\qlib\rl\order_execution\simulator_qlib.py",
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

fn tracked_name(decision: &TrackedOrderDecision) -> &'static str {
    match decision.orders[0].amount().to_bits() {
        bits if bits == 1.0_f64.to_bits() => "outer",
        bits if bits == 2.0_f64.to_bits() => "middle",
        bits if bits == 7.0_f64.to_bits() => "atomic",
        _ => "unknown",
    }
}

#[derive(Default)]
struct AssemblyObservations {
    inputs: Vec<(f64, usize)>,
}

struct AssemblyFactory {
    observations: Arc<Mutex<AssemblyObservations>>,
    machine_failure: bool,
}

impl NestedChildAssemblyFactory for AssemblyFactory {
    fn assemble(
        &mut self,
        decision: &dyn OrderDecision,
        level: usize,
    ) -> Result<NestedChildAssembly, NestedInnerExecutorError> {
        let amount = decision.orders()[0].amount();
        self.observations
            .lock()
            .unwrap()
            .inputs
            .push((amount, level));
        let state = Arc::new(Mutex::new(State {
            fail: self.machine_failure.then_some("strategy_reset"),
            ..State::default()
        }));
        let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&state)));
        let mut child_config = config(true, "None");
        child_config.level = level;
        Ok(NestedChildAssembly {
            calendar,
            config: child_config,
            run: ResumableNestedRun {
                level_binding: Box::new(Binding(Arc::clone(&state))),
                inner: Box::new(Inner {
                    state: Arc::clone(&state),
                    len: 1,
                }),
                strategy: Box::new(ProxyStrategy {
                    state: Arc::clone(&state),
                    prompts: 1,
                    resumed: 0,
                }),
                outer: Box::new(Outer {
                    state: Arc::clone(&state),
                    decision: OrderTradeDecision::from_orders(
                        decision.orders().to_vec(),
                        decision.start_time(),
                        decision.end_time(),
                        None,
                    ),
                    empty: false,
                    range: Some((0, 0)),
                    replace: false,
                }),
                account: Box::new(Account(state)),
                return_sink: None,
            },
        })
    }
}

struct RejectingAssemblyFactory;

impl NestedChildAssemblyFactory for RejectingAssemblyFactory {
    fn assemble(
        &mut self,
        _decision: &dyn OrderDecision,
        _level: usize,
    ) -> Result<NestedChildAssembly, NestedInnerExecutorError> {
        Err(inner_error("assembly"))
    }
}

#[test]
fn configured_child_session_owns_rebuilds_and_routes_each_executor_graph() {
    let observations = Arc::new(Mutex::new(AssemblyObservations::default()));
    let mut session = ConfiguredNestedChildSession::new(Box::new(AssemblyFactory {
        observations: Arc::clone(&observations),
        machine_failure: false,
    }));
    assert_eq!(
        expect_inner_error(session.resume(NestedExecutorResume::Continue)).message,
        "configured child session is not active"
    );
    let NestedInnerProgress::Suspended(NestedControlEvent::TrackedDecision(tracked)) =
        session.begin(order(5.0), 3).unwrap()
    else {
        panic!("expected owned outer tracking event")
    };
    assert_eq!(tracked.orders[0].amount().to_bits(), 5.0_f64.to_bits());
    assert_eq!(
        expect_inner_error(session.begin(order(99.0), 9)).message,
        "configured child session is already active"
    );
    let NestedInnerProgress::Suspended(NestedControlEvent::StrategyPrompt(first_prompt)) =
        session.resume(NestedExecutorResume::Continue).unwrap()
    else {
        panic!("expected configured SAOE prompt")
    };
    assert_eq!(first_prompt.kind, "proxy-saoe");
    let NestedInnerProgress::Suspended(NestedControlEvent::TrackedDecision(inner)) = session
        .resume(NestedExecutorResume::Action(Some(4.0)))
        .unwrap()
    else {
        panic!("expected owned inner tracking event")
    };
    assert_eq!(inner.orders[0].amount().to_bits(), 4.0_f64.to_bits());
    let NestedInnerProgress::Complete {
        decision,
        executions,
    } = session
        .resume(NestedExecutorResume::Action(Some(4.0)))
        .unwrap()
    else {
        panic!("expected configured child completion")
    };
    assert_eq!(decision.orders()[0].amount().to_bits(), 5.0_f64.to_bits());
    assert_eq!(
        executions.lock().unwrap()[0]
            .order
            .read()
            .unwrap()
            .amount()
            .to_bits(),
        4.0_f64.to_bits()
    );

    assert!(matches!(
        session.begin(order(6.0), 4).unwrap(),
        NestedInnerProgress::Suspended(NestedControlEvent::TrackedDecision(_))
    ));
    assert_eq!(observations.lock().unwrap().inputs, [(5.0, 3), (6.0, 4)]);

    let mut rejected = ConfiguredNestedChildSession::new(Box::new(RejectingAssemblyFactory));
    assert_eq!(
        expect_inner_error(rejected.begin(order(1.0), 0)).message,
        "assembly"
    );

    let mut failed = ConfiguredNestedChildSession::new(Box::new(AssemblyFactory {
        observations,
        machine_failure: true,
    }));
    assert!(failed.begin(order(2.0), 0).is_ok());
    assert!(failed.resume(NestedExecutorResume::Continue).is_err());
    assert!(failed.resume(NestedExecutorResume::Continue).is_err());
}

#[test]
fn recursive_raw_handle_delegation_preserves_identity_and_failures() {
    let state = Arc::new(Mutex::new(State::default()));
    let raw = state.lock().unwrap().order.clone();
    let lifecycle = Inner {
        state: state.clone(),
        len: 1,
    };
    let child = RecursiveChild {
        state: state.clone(),
        stage: 0,
        decision: None,
    };
    let inner = RecursiveNestedInnerAdapter::new(Box::new(lifecycle), Box::new(child));
    {
        let _guard = raw.write().unwrap();
        assert!(Arc::ptr_eq(&raw, &inner.order_indicator_handle().unwrap()));
        assert!(Arc::ptr_eq(&raw, &inner.order_indicator_handle().unwrap()));
    }
    state.lock().unwrap().fail = Some("handle");
    assert_eq!(
        inner.order_indicator_handle().unwrap_err(),
        inner_error("handle")
    );
    state.lock().unwrap().fail = None;
    state.lock().unwrap().order = Arc::default();
    assert!(!Arc::ptr_eq(&raw, &inner.order_indicator_handle().unwrap()));
    drop(inner);
    drop(state);
    assert_eq!(Arc::strong_count(&raw), 1);
    assert!(raw.try_write().is_ok());
}

fn recursive_failure(
    failure: &'static str,
    owned: bool,
) -> (ResumableNestedExecutorError, ResumableNestedExecutorError) {
    let state = Arc::new(Mutex::new(State {
        fail: Some(failure),
        ..State::default()
    }));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&state)));
    let binding = Binding(Arc::clone(&state));
    let lifecycle = Inner {
        state: Arc::clone(&state),
        len: 1,
    };
    let child = RecursiveChild {
        state: Arc::clone(&state),
        stage: 0,
        decision: None,
    };
    let inner = RecursiveNestedInnerAdapter::new(Box::new(lifecycle), Box::new(child));
    let strategy = ImmediateStrategy(Arc::clone(&state));
    let outer = Outer {
        state: Arc::clone(&state),
        decision: outer_decision(),
        empty: false,
        range: Some((0, 0)),
        replace: false,
    };
    let account = Account(Arc::clone(&state));
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        config(false, "None"),
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: None,
        },
    );
    let mut driver = if owned {
        RecursiveStrategyDriver::new_owned(machine)
    } else {
        RecursiveStrategyDriver::new(&mut machine)
    };
    let Err(first) = driver.advance(None) else {
        panic!("{failure} unexpectedly succeeded")
    };
    let Err(terminal) = driver.advance(None) else {
        panic!("{failure} unexpectedly resumed")
    };
    drop(driver);
    if owned {
        assert_eq!(Arc::strong_count(&state), 1, "owned failed graph leaked");
    }
    (first, terminal)
}

#[test]
fn recursive_driver_matches_multilevel_python_send_protocol() {
    check_recursive_driver(false);
}

#[test]
fn owned_recursive_driver_preserves_source_protocol_and_releases_completed_graph() {
    check_recursive_driver(true);
}

fn check_recursive_driver(owned: bool) {
    let state = Arc::new(Mutex::new(State::default()));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&state)));
    let binding = Binding(Arc::clone(&state));
    let lifecycle = Inner {
        state: Arc::clone(&state),
        len: 1,
    };
    let child = RecursiveChild {
        state: Arc::clone(&state),
        stage: 0,
        decision: None,
    };
    let inner = RecursiveNestedInnerAdapter::new(Box::new(lifecycle), Box::new(child));
    let strategy = ImmediateStrategy(Arc::clone(&state));
    let outer = Outer {
        state: Arc::clone(&state),
        decision: outer_decision(),
        empty: false,
        range: Some((0, 0)),
        replace: false,
    };
    let account = Account(Arc::clone(&state));
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        config(true, "None"),
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: None,
        },
    );
    let mut driver = if owned {
        RecursiveStrategyDriver::new_owned(machine)
    } else {
        RecursiveStrategyDriver::new(&mut machine)
    };
    assert_eq!(
        match driver.advance(Some(1.0)) {
            Ok(_) => panic!("initial action unexpectedly succeeded"),
            Err(error) => error,
        },
        ResumableNestedExecutorError::InvalidInitialAction
    );
    let RecursiveStrategyEvent::StrategyPrompt(prompt) = driver.advance(None).unwrap() else {
        panic!("expected deepest child prompt")
    };
    assert_eq!(prompt.kind, "proxy-saoe");
    let first = driver
        .decisions()
        .iter()
        .map(tracked_name)
        .chain(["Proxy"])
        .collect::<Vec<_>>();
    let RecursiveStrategyEvent::Complete(collection) = driver.advance(Some(7.0)).unwrap() else {
        panic!("expected recursive completion")
    };
    assert_eq!(collection.executions().lock().unwrap().len(), 1);
    let second = driver.decisions()[2..]
        .iter()
        .map(tracked_name)
        .chain(["Complete"])
        .collect::<Vec<_>>();
    let observed = json!({
        "first": first,
        "second": second,
        "received": state.lock().unwrap().received,
    });
    let python = python_recursive_protocol();
    assert_eq!(observed["first"], python["first"]);
    assert_eq!(observed["second"], python["second"]);
    assert_eq!(observed["received"], python["received"]);
    assert_eq!(python["steps"], json!(["atomic", "child", "top"]));
    assert_eq!(
        state.lock().unwrap().resumes,
        [
            NestedExecutorResume::Continue,
            NestedExecutorResume::Action(Some(7.0)),
            NestedExecutorResume::Action(Some(7.0)),
        ]
    );
    assert!(matches!(
        driver.advance(None),
        Err(ResumableNestedExecutorError::AlreadyComplete)
    ));
    assert_eq!(driver.decisions().len(), 3);
    drop(driver);
    if owned {
        assert_eq!(Arc::strong_count(&state), 1, "owned completed graph leaked");
    }
}

#[test]
fn recursive_child_begin_and_resume_failures_terminate_the_parent_driver() {
    for failure in ["child_begin", "child_resume"] {
        for owned in [false, true] {
            let (first, terminal) = recursive_failure(failure, owned);
            assert!(matches!(first, ResumableNestedExecutorError::Collection(_)));
            assert_eq!(terminal, ResumableNestedExecutorError::Failed);
        }
    }
}

#[test]
fn synchronous_inner_defaults_return_owned_completion_and_reject_resume() {
    let state = Arc::new(Mutex::new(State::default()));
    let mut inner = Inner { state, len: 1 };
    assert_eq!(inner.control_mode(), NestedInnerControlMode::Framework);
    let NestedInnerProgress::Complete {
        decision,
        executions,
    } = inner.begin_collect_data(order(4.0), 3).unwrap()
    else {
        panic!("synchronous inner unexpectedly suspended")
    };
    assert_eq!(decision.orders()[0].amount().to_bits(), 4.0_f64.to_bits());
    assert_eq!(executions.lock().unwrap().len(), 1);
    let Err(error) = inner.resume_collect_data(NestedExecutorResume::Continue) else {
        panic!("synchronous inner unexpectedly resumed")
    };
    assert_eq!(error.message, "inner executor is not suspended");
}

#[test]
fn recursive_adapter_delegates_calendar_lifecycle_and_child_protocol() {
    let state = Arc::new(Mutex::new(State::default()));
    let lifecycle = Inner {
        state: Arc::clone(&state),
        len: 2,
    };
    let child = RecursiveChild {
        state: Arc::clone(&state),
        stage: 0,
        decision: None,
    };
    let mut adapter = RecursiveNestedInnerAdapter::new(Box::new(lifecycle), Box::new(child));
    assert!(!adapter.finished().unwrap());
    assert_eq!(adapter.trade_len().unwrap(), 2);
    assert_eq!(adapter.trade_step().unwrap(), 0);
    assert_eq!(adapter.step_time().unwrap(), (time(0), time(1)));
    adapter.reset_window(time(0), time(1)).unwrap();
    assert!(
        adapter
            .order_indicator_snapshot()
            .unwrap()
            .to_series()
            .is_empty()
    );
    assert_eq!(adapter.control_mode(), NestedInnerControlMode::Delegated);
    let mut direct = order(1.0);
    assert_eq!(
        adapter.collect_data(&mut *direct, 0).unwrap_err().message,
        "recursive child collection requires the resumable protocol"
    );
    assert!(matches!(
        adapter.begin_collect_data(order(2.0), 1).unwrap(),
        NestedInnerProgress::Suspended(NestedControlEvent::TrackedDecision(_))
    ));
    assert!(matches!(
        adapter
            .resume_collect_data(NestedExecutorResume::Continue)
            .unwrap(),
        NestedInnerProgress::Suspended(NestedControlEvent::StrategyPrompt(_))
    ));
    assert!(matches!(
        adapter
            .resume_collect_data(NestedExecutorResume::Action(Some(5.0)))
            .unwrap(),
        NestedInnerProgress::Suspended(NestedControlEvent::TrackedDecision(_))
    ));
    assert!(matches!(
        adapter
            .resume_collect_data(NestedExecutorResume::Action(Some(5.0)))
            .unwrap(),
        NestedInnerProgress::Complete { .. }
    ));
    adapter.step().unwrap();
    assert!(adapter.finished().unwrap());
}

#[test]
fn framework_tracked_decision_can_precede_an_inner_suspension() {
    let state = Arc::new(Mutex::new(State::default()));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&state)));
    let binding = Binding(Arc::clone(&state));
    let inner = FrameworkSuspendingInner {
        order: Arc::default(),
        finished: Cell::new(false),
        decision: None,
    };
    let strategy = ImmediateStrategy(Arc::clone(&state));
    let outer = Outer {
        state: Arc::clone(&state),
        decision: outer_decision(),
        empty: false,
        range: Some((0, 0)),
        replace: false,
    };
    let account = Account(Arc::clone(&state));
    let mut policy = config(true, "None");
    policy.decision_tracking.outer = false;
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        policy,
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: None,
        },
    );
    assert!(matches!(
        machine.resume(NestedExecutorResume::Continue),
        Ok(ResumableNestedEvent::TrackedDecision(_))
    ));
    assert!(matches!(
        machine.resume(NestedExecutorResume::Continue),
        Ok(ResumableNestedEvent::StrategyPrompt(_))
    ));
    assert!(matches!(
        machine.resume(NestedExecutorResume::Action(Some(2.0))),
        Ok(ResumableNestedEvent::Complete(_))
    ));
}

#[test]
fn resumable_protocol_matches_python_yield_send_and_ignored_decision_actions() {
    let state = Arc::new(Mutex::new(State::default()));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&state)));
    let binding = Binding(Arc::clone(&state));
    let inner = Inner {
        state: Arc::clone(&state),
        len: 1,
    };
    let strategy = ProxyStrategy {
        state: Arc::clone(&state),
        prompts: 1,
        resumed: 0,
    };
    let outer = Outer {
        state: Arc::clone(&state),
        decision: outer_decision(),
        empty: false,
        range: None,
        replace: false,
    };
    let account = Account(Arc::clone(&state));
    let sink = Sink {
        state: Arc::clone(&state),
    };
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        config(true, "cash"),
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: Some(Box::new(sink)),
        },
    );

    assert_eq!(
        expect_error(machine.resume(NestedExecutorResume::Action(Some(1.0)))),
        ResumableNestedExecutorError::InvalidInitialAction
    );
    let ResumableNestedEvent::TrackedDecision(first) =
        machine.resume(NestedExecutorResume::Continue).unwrap()
    else {
        panic!("expected outer decision")
    };
    assert_eq!(first.orders[0].amount().to_bits(), 1.0_f64.to_bits());
    assert!(!first.has_trade_range);
    assert!(state.lock().unwrap().events.is_empty());

    let ResumableNestedEvent::StrategyPrompt(prompt) = machine
        .resume(NestedExecutorResume::Action(Some(123.0)))
        .unwrap()
    else {
        panic!("expected strategy prompt")
    };
    assert_eq!(serde_json::to_value(&prompt).unwrap()["kind"], "proxy-saoe");
    assert_eq!(
        serde_json::from_value::<NestedExecutorResume>(
            serde_json::to_value(NestedExecutorResume::Action(Some(7.0))).unwrap(),
        )
        .unwrap(),
        NestedExecutorResume::Action(Some(7.0))
    );

    let ResumableNestedEvent::TrackedDecision(inner_decision) = machine
        .resume(NestedExecutorResume::Action(Some(7.0)))
        .unwrap()
    else {
        panic!("expected inner decision")
    };
    assert_eq!(
        inner_decision.orders[0].amount().to_bits(),
        7.0_f64.to_bits()
    );
    let ResumableNestedEvent::Complete(collection) = machine
        .resume(NestedExecutorResume::Action(Some(999.0)))
        .unwrap()
    else {
        panic!("expected completion")
    };
    assert_eq!(collection.executions().lock().unwrap().len(), 1);
    assert_eq!(collection.decisions().len(), 1);
    assert_eq!(state.lock().unwrap().received, [Some(7.0)]);
    assert_eq!(
        expect_error(machine.resume(NestedExecutorResume::Continue)),
        ResumableNestedExecutorError::AlreadyComplete
    );
    drop(machine);
    assert_eq!(state.lock().unwrap().sink_count, 1);
    assert_eq!(
        python_protocol(),
        json!([
            ["decision", "outer"],
            ["prompt", "S"],
            ["decision", "inner"],
            ["complete", 1, [7]],
        ])
    );
}

#[test]
fn immediate_strategy_tracks_both_decisions_aligns_range_and_skips_settlement() {
    let state = Arc::new(Mutex::new(State::default()));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&state)));
    let binding = Binding(Arc::clone(&state));
    let inner = Inner {
        state: Arc::clone(&state),
        len: 2,
    };
    let mut strategy = ImmediateStrategy(Arc::clone(&state));
    assert_eq!(
        match strategy.resume_trade_decision(None) {
            Ok(_) => panic!("expected an unsuspended strategy error"),
            Err(error) => error.message,
        },
        "strategy is not suspended"
    );
    let outer = Outer {
        state: Arc::clone(&state),
        decision: outer_decision(),
        empty: false,
        range: Some((1, 1)),
        replace: true,
    };
    let account = Account(Arc::clone(&state));
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        config(true, "None"),
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: None,
        },
    );
    assert!(matches!(
        machine.resume(NestedExecutorResume::Continue).unwrap(),
        ResumableNestedEvent::TrackedDecision(_)
    ));
    assert!(matches!(
        machine.resume(NestedExecutorResume::Continue).unwrap(),
        ResumableNestedEvent::TrackedDecision(_)
    ));
    let ResumableNestedEvent::Complete(collection) =
        machine.resume(NestedExecutorResume::Continue).unwrap()
    else {
        panic!("expected immediate completion")
    };
    assert_eq!(collection.executions().lock().unwrap().len(), 1);
    assert!(state.lock().unwrap().events.contains(&"step".to_owned()));
    assert!(state.lock().unwrap().events.contains(&"alter".to_owned()));
}

#[test]
fn repeated_strategy_prompts_forward_each_resume_value() {
    let state = Arc::new(Mutex::new(State::default()));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&state)));
    let binding = Binding(Arc::clone(&state));
    let inner = Inner {
        state: Arc::clone(&state),
        len: 1,
    };
    let strategy = ProxyStrategy {
        state: Arc::clone(&state),
        prompts: 2,
        resumed: 0,
    };
    let outer = Outer {
        state: Arc::clone(&state),
        decision: outer_decision(),
        empty: false,
        range: Some((0, 0)),
        replace: false,
    };
    let account = Account(Arc::clone(&state));
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        config(false, "None"),
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: None,
        },
    );
    assert!(matches!(
        machine.resume(NestedExecutorResume::Continue).unwrap(),
        ResumableNestedEvent::StrategyPrompt(_)
    ));
    let ResumableNestedEvent::StrategyPrompt(second) =
        machine.resume(NestedExecutorResume::Continue).unwrap()
    else {
        panic!("expected second prompt")
    };
    assert_eq!(second.payload, [1]);
    assert!(matches!(
        machine
            .resume(NestedExecutorResume::Action(Some(3.0)))
            .unwrap(),
        ResumableNestedEvent::Complete(_)
    ));
    assert_eq!(state.lock().unwrap().received, [None, Some(3.0)]);
}

#[test]
fn empty_decisions_complete_without_running_the_inner_executor() {
    let state = Arc::new(Mutex::new(State::default()));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&state)));
    let binding = Binding(Arc::clone(&state));
    let inner = Inner {
        state: Arc::clone(&state),
        len: 1,
    };
    let strategy = ImmediateStrategy(Arc::clone(&state));
    let outer = Outer {
        state: Arc::clone(&state),
        decision: outer_decision(),
        empty: true,
        range: None,
        replace: false,
    };
    let account = Account(Arc::clone(&state));
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        config(false, "None"),
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: None,
        },
    );
    let ResumableNestedEvent::Complete(collection) =
        machine.resume(NestedExecutorResume::Continue).unwrap()
    else {
        panic!("expected empty completion")
    };
    assert!(collection.executions().lock().unwrap().is_empty());
}

#[test]
fn disabled_empty_and_alignment_policies_short_circuit_their_checks() {
    let shared = Arc::new(Mutex::new(State::default()));
    let mut policy = config(false, "None");
    policy.skip_empty_decision = false;
    policy.align_range_limit = false;
    let collection = run_immediate_policy(&shared, policy, 2, true, Some((1, 1)));
    assert_eq!(collection.executions().lock().unwrap().len(), 2);
    let events = &shared.lock().unwrap().events;
    assert!(!events.contains(&"empty".to_owned()));
    assert!(!events.contains(&"step".to_owned()));
}

#[test]
fn alignment_skips_steps_above_the_range_limit() {
    let shared = Arc::new(Mutex::new(State::default()));
    let collection = run_immediate_policy(&shared, config(false, "None"), 2, false, Some((0, 0)));
    assert_eq!(collection.executions().lock().unwrap().len(), 1);
    assert!(shared.lock().unwrap().events.contains(&"step".to_owned()));
}

#[test]
fn suspended_ready_decision_preparation_failure_is_terminal() {
    let shared = Arc::new(Mutex::new(State {
        fail: Some("modify"),
        ..State::default()
    }));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&shared)));
    let binding = Binding(Arc::clone(&shared));
    let inner = Inner {
        state: Arc::clone(&shared),
        len: 1,
    };
    let strategy = ProxyStrategy {
        state: Arc::clone(&shared),
        prompts: 1,
        resumed: 0,
    };
    let outer = Outer {
        state: Arc::clone(&shared),
        decision: outer_decision(),
        empty: false,
        range: Some((0, 0)),
        replace: false,
    };
    let account = Account(Arc::clone(&shared));
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        config(false, "None"),
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: None,
        },
    );
    assert!(matches!(
        machine.resume(NestedExecutorResume::Continue),
        Ok(ResumableNestedEvent::StrategyPrompt(_))
    ));
    assert!(matches!(
        machine.resume(NestedExecutorResume::Action(Some(1.0))),
        Err(ResumableNestedExecutorError::Collection(_))
    ));
    assert_eq!(
        expect_error(machine.resume(NestedExecutorResume::Continue)),
        ResumableNestedExecutorError::Failed
    );
}

#[test]
fn tracked_inner_decision_execution_failure_is_terminal() {
    let shared = Arc::new(Mutex::new(State {
        fail: Some("collect"),
        ..State::default()
    }));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&shared)));
    let binding = Binding(Arc::clone(&shared));
    let inner = Inner {
        state: Arc::clone(&shared),
        len: 1,
    };
    let strategy = ImmediateStrategy(Arc::clone(&shared));
    let outer = Outer {
        state: Arc::clone(&shared),
        decision: outer_decision(),
        empty: false,
        range: Some((0, 0)),
        replace: false,
    };
    let account = Account(Arc::clone(&shared));
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        config(true, "None"),
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy: Box::new(strategy),
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: None,
        },
    );
    for _ in 0..2 {
        assert!(matches!(
            machine.resume(NestedExecutorResume::Continue),
            Ok(ResumableNestedEvent::TrackedDecision(_))
        ));
    }
    assert!(matches!(
        machine.resume(NestedExecutorResume::Continue),
        Err(ResumableNestedExecutorError::Collection(_))
    ));
    assert_eq!(
        expect_error(machine.resume(NestedExecutorResume::Continue)),
        ResumableNestedExecutorError::Failed
    );
}

fn assert_terminal_failure(boundary: &'static str) {
    let state = Arc::new(Mutex::new(State {
        fail: Some(boundary),
        ..State::default()
    }));
    let calendar: Box<dyn NestedCalendar> = Box::new(OuterCalendar(Arc::clone(&state)));
    let binding = Binding(Arc::clone(&state));
    let inner = Inner {
        state: Arc::clone(&state),
        len: 1,
    };
    let suspended = boundary == "resume";
    let immediate = ImmediateStrategy(Arc::clone(&state));
    let proxy = ProxyStrategy {
        state: Arc::clone(&state),
        prompts: 1,
        resumed: 0,
    };
    let strategy: Box<dyn NestedStrategy> = if suspended {
        Box::new(proxy)
    } else {
        Box::new(immediate)
    };
    let range = match boundary {
        "trade_len" => None,
        "step" => Some((1, 1)),
        _ => Some((0, 0)),
    };
    let outer = Outer {
        state: Arc::clone(&state),
        decision: outer_decision(),
        empty: false,
        range,
        replace: boundary == "alter",
    };
    let account = Account(Arc::clone(&state));
    let sink = Sink {
        state: Arc::clone(&state),
    };
    let mut machine = ResumableNestedExecutor::new(
        calendar,
        config(false, "cash"),
        ResumableNestedRun {
            level_binding: Box::new(binding),
            inner: Box::new(inner),
            strategy,
            outer: Box::new(outer),
            account: Box::new(account),
            return_sink: Some(Box::new(sink)),
        },
    );
    let mut result = machine.resume(NestedExecutorResume::Continue);
    if matches!(result, Ok(ResumableNestedEvent::StrategyPrompt(_))) {
        result = machine.resume(NestedExecutorResume::Action(Some(1.0)));
    }
    let Err(error) = result else {
        panic!("{boundary} unexpectedly succeeded")
    };
    assert_ne!(error, ResumableNestedExecutorError::InvalidInitialAction);
    assert_eq!(
        expect_error(machine.resume(NestedExecutorResume::Continue)),
        ResumableNestedExecutorError::Failed
    );
    assert!(
        state.lock().unwrap().events.iter().any(|item| {
            item == boundary
                || (boundary == "settle_start" && item == "settle_start:cash")
                || (boundary == "collect" && item.starts_with("collect:"))
        }),
        "{boundary}: {:?}",
        state.lock().unwrap().events
    );
}

#[test]
fn every_resumable_failure_is_terminal_at_its_boundary() {
    for stage in [
        "settle_start",
        "outer_time",
        "reset",
        "bind",
        "strategy_reset",
        "finished",
        "update",
        "alter",
        "empty",
        "range",
        "trade_len",
        "trade_step",
        "step",
        "begin",
        "resume",
        "modify",
        "inner_time",
        "collect",
        "post",
        "handle",
        "upper",
        "bar_time",
        "bar",
        "outer_step",
        "commit",
        "sink",
    ] {
        assert_terminal_failure(stage);
    }
}
