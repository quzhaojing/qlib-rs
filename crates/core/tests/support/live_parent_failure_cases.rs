use super::*;
use domain_core::decision_update::{
    LiveDecision, LiveDecisionAccessError, SharedDecisionUpdateError,
};
use domain_core::{
    NestedExecutorReturnSink, NestedExecutorReturnSinkError, NestedInnerExecutorError,
    NumpyOrderIndicator, SharedOrderIndicator,
};

#[derive(Clone)]
struct Probe {
    events: Arc<Mutex<Vec<&'static str>>>,
    fail: &'static str,
}
impl Probe {
    fn hit(&self, name: &'static str) -> Result<(), String> {
        self.events.lock().unwrap().push(name);
        if self.fail == name {
            Err(name.into())
        } else {
            Ok(())
        }
    }
}

struct OuterCalendarProbe {
    inner: Box<dyn NestedCalendar>,
    probe: Probe,
    times: Mutex<usize>,
}
impl NestedCalendar for OuterCalendarProbe {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        self.inner.finished()
    }
    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        self.inner.trade_len()
    }
    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        self.inner.trade_step()
    }
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        let mut count = self.times.lock().unwrap();
        *count += 1;
        self.probe
            .hit(if *count == 1 {
                "outer.init_time"
            } else {
                "outer.final_time"
            })
            .map_err(|message| NestedCalendarError { message })?;
        self.inner.step_time()
    }
    fn step(&self) -> Result<(), NestedCalendarError> {
        self.probe
            .hit("outer.step")
            .map_err(|message| NestedCalendarError { message })?;
        self.inner.step()
    }
}

struct InnerProbe {
    inner: Box<dyn NestedInnerExecutor>,
    probe: Probe,
    framework: bool,
}
impl NestedCalendar for InnerProbe {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        self.probe
            .hit("inner.finished")
            .map_err(|message| NestedCalendarError { message })?;
        self.inner.finished()
    }
    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        self.probe
            .hit("inner.len")
            .map_err(|message| NestedCalendarError { message })?;
        self.inner.trade_len()
    }
    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        self.probe
            .hit("inner.step_index")
            .map_err(|message| NestedCalendarError { message })?;
        self.inner.trade_step()
    }
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        self.probe
            .hit("inner.time")
            .map_err(|message| NestedCalendarError { message })?;
        self.inner.step_time()
    }
    fn step(&self) -> Result<(), NestedCalendarError> {
        self.probe
            .hit("inner.step")
            .map_err(|message| NestedCalendarError { message })?;
        self.inner.step()
    }
}
impl NestedInnerExecutor for InnerProbe {
    fn reset_window(
        &mut self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        self.probe
            .hit("inner.reset")
            .map_err(|message| NestedInnerExecutorError { message })?;
        self.inner.reset_window(start, end)
    }
    fn collect_data(
        &mut self,
        _: &mut dyn OrderDecision,
        _: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError> {
        panic!("no legacy collection")
    }
    fn live_control_mode(&self) -> domain_core::NestedInnerControlMode {
        if self.framework {
            domain_core::NestedInnerControlMode::Framework
        } else {
            self.inner.live_control_mode()
        }
    }
    fn begin_live_collect_data(
        &mut self,
        decision: LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        self.probe
            .hit("inner.begin")
            .map_err(|message| NestedInnerExecutorError { message })?;
        self.inner.begin_live_collect_data(decision, level)
    }
    fn resume_live_collect_data(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<LiveNestedInnerProgress, NestedInnerExecutorError> {
        self.probe
            .hit("inner.resume")
            .map_err(|message| NestedInnerExecutorError { message })?;
        self.inner.resume_live_collect_data(input)
    }
    fn close_live_collect_data(&mut self) -> Result<(), NestedInnerExecutorError> {
        self.inner.close_live_collect_data()?;
        self.probe
            .hit("inner.close")
            .map_err(|message| NestedInnerExecutorError { message })
    }
    fn order_indicator_handle(
        &self,
    ) -> Result<SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError> {
        self.probe
            .hit("inner.indicator")
            .map_err(|message| NestedInnerExecutorError { message })?;
        self.inner.order_indicator_handle()
    }
    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        self.inner.order_indicator_snapshot()
    }
}

struct DecisionProbe {
    inner: LiveDecisionHandle,
    probe: Probe,
}
impl LiveDecision for DecisionProbe {
    fn total_step(
        &self,
    ) -> Result<domain_core::decision_construction::DecisionTotalStep, LiveDecisionAccessError>
    {
        self.inner.total_step()
    }
    fn base(&self) -> Result<ConstructedDecisionBase, LiveDecisionAccessError> {
        self.inner.base()
    }
    fn inherit_range(
        &self,
        range: Option<domain_core::SharedTradeRange>,
    ) -> Result<(), LiveDecisionAccessError> {
        self.inner.inherit_range(range)
    }
    fn orders(
        &self,
    ) -> Result<domain_core::decision_construction::SharedDecisionOrders, LiveDecisionAccessError>
    {
        self.inner.orders()
    }
    fn is_empty(&self) -> Result<bool, LiveDecisionAccessError> {
        self.probe
            .hit("decision.empty")
            .map_err(|_| LiveDecisionAccessError::DecisionPoisoned)?;
        self.inner.is_empty()
    }
    fn range_limit(
        &self,
        calendar: Option<&dyn TradeCalendarRange>,
        default: domain_core::RangeLimitDefault,
    ) -> Result<Option<(i64, i64)>, LiveDecisionAccessError> {
        self.probe
            .hit("decision.range")
            .map_err(|_| LiveDecisionAccessError::DecisionPoisoned)?;
        self.inner.range_limit(calendar, default)
    }
    fn modify_inner_decision(
        &self,
        inner: &LiveDecisionHandle,
    ) -> Result<(), LiveDecisionAccessError> {
        self.probe
            .hit("decision.propagate")
            .map_err(|_| LiveDecisionAccessError::DecisionPoisoned)?;
        self.inner.modify_inner_decision(inner)
    }
    fn update(
        self: Arc<Self>,
        _: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<Option<LiveDecisionHandle>, SharedDecisionUpdateError> {
        self.probe.hit("decision.update").map_err(|message| {
            SharedDecisionUpdateError::Strategy(domain_core::DecisionUpdateStrategyError {
                message,
            })
        })?;
        Ok(Some(self))
    }
}

struct StrategyProbe {
    inner: Box<dyn LiveNestedStrategy>,
    probe: Probe,
}
impl LiveNestedStrategy for StrategyProbe {
    fn reset(&mut self, outer: &LiveDecisionHandle) -> Result<(), NestedStrategyError> {
        self.probe
            .hit("strategy.reset")
            .map_err(|message| NestedStrategyError { message })?;
        self.inner.reset(outer)
    }
    fn alter_outer_decision(
        &mut self,
        outer: LiveDecisionHandle,
    ) -> Result<LiveDecisionHandle, NestedStrategyError> {
        self.probe
            .hit("strategy.alter")
            .map_err(|message| NestedStrategyError { message })?;
        self.inner.alter_outer_decision(outer)
    }
    fn begin(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError> {
        self.probe
            .hit("strategy.begin")
            .map_err(|message| NestedStrategyError { message })?;
        self.inner.begin(previous)
    }
    fn resume(
        &mut self,
        volume: Option<f64>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError> {
        self.probe
            .hit("strategy.resume")
            .map_err(|message| NestedStrategyError { message })?;
        self.inner.resume(volume)
    }
    fn close(&mut self) -> Result<(), NestedStrategyError> {
        self.inner.close()
    }
    fn post_execute(&mut self, rows: &SharedNestedResult) -> Result<(), NestedStrategyError> {
        self.probe
            .hit("strategy.post")
            .map_err(|message| NestedStrategyError { message })?;
        self.inner.post_execute(rows)?;
        if self.probe.fail == "result.poison" {
            let rows = rows.clone();
            assert!(
                std::thread::spawn(move || {
                    let _guard = rows.lock().unwrap();
                    panic!("poison result after post");
                })
                .join()
                .is_err()
            );
        }
        Ok(())
    }
    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        self.probe
            .hit("strategy.upper")
            .map_err(|message| NestedStrategyError { message })?;
        self.inner.post_upper_level()
    }
}

struct BindingProbe(Probe);
impl NestedLevelBinding for BindingProbe {
    fn bind_inner(&mut self, _: &dyn NestedInnerExecutor) -> Result<(), NestedLevelBindingError> {
        self.0
            .hit("binding")
            .map_err(|message| NestedLevelBindingError { message })
    }
}
struct AccountProbe(Probe);
impl NestedExecutorAccount for AccountProbe {
    fn settle_start(&mut self, _: &str) -> Result<(), NestedExecutorAccountError> {
        self.0
            .hit("settle")
            .map_err(|message| NestedExecutorAccountError { message })
    }
    fn settle_commit(&mut self) -> Result<(), NestedExecutorAccountError> {
        self.0
            .hit("commit")
            .map_err(|message| NestedExecutorAccountError { message })
    }
    fn update_bar_end(&mut self, _: NestedBarEnd<'_>) -> Result<(), NestedExecutorAccountError> {
        panic!("no legacy account")
    }
    fn update_live_bar_end(
        &mut self,
        _: LiveNestedBarEnd<'_>,
    ) -> Result<(), NestedExecutorAccountError> {
        self.0
            .hit("account")
            .map_err(|message| NestedExecutorAccountError { message })
    }
}
struct SinkProbe {
    probe: Probe,
    retained: Arc<Mutex<Option<SharedNestedResult>>>,
}
impl NestedExecutorReturnSink for SinkProbe {
    fn store_execute_result(
        &mut self,
        rows: &SharedNestedResult,
    ) -> Result<(), NestedExecutorReturnSinkError> {
        *self.retained.lock().unwrap() = Some(rows.clone());
        rows.lock().unwrap().clear();
        self.probe
            .hit("sink")
            .map_err(|message| NestedExecutorReturnSinkError { message })
    }
}

struct Harness {
    executor: LiveNestedExecutor,
    events: Arc<Mutex<Vec<&'static str>>>,
    calendar: Arc<SharedInnerCalendar>,
    retained: Arc<Mutex<Option<SharedNestedResult>>>,
}

fn harness(fail: &'static str, policy: &str) -> Harness {
    let (mut factory, original) = factory(1, false);
    let events = Arc::new(Mutex::new(Vec::new()));
    let probe = Probe {
        events: events.clone(),
        fail,
    };
    let original: LiveDecisionHandle = Arc::new(DecisionProbe {
        inner: original,
        probe: probe.clone(),
    });
    if policy == "empty" || policy == "empty_continue" {
        original.orders().unwrap().write().unwrap().clear();
    }
    if policy == "skip" || policy == "above" {
        let index = if policy == "skip" { 1 } else { -1 };
        original
            .inherit_range(Some(Arc::new(domain_core::IdxTradeRange::new(
                index, index,
            ))))
            .unwrap();
    }
    let mut assembly = factory.assemble(&original, 0).unwrap();
    if policy == "immediate" || policy == "framework" {
        assembly.run.strategy = Box::new(Strategy {
            original: original.clone(),
            replacement: original.clone(),
            child: factory.decisions[0].clone(),
            prompts: false,
            events: factory.events.clone(),
            posted: factory.posted.clone(),
        });
    }
    let calendar = factory.calendar.clone();
    assembly.calendar = Box::new(OuterCalendarProbe {
        inner: assembly.calendar,
        probe: probe.clone(),
        times: Mutex::new(0),
    });
    assembly.config.decision_tracking.outer = false;
    assembly.config.decision_tracking.inner = policy == "framework";
    assembly.config.skip_empty_decision = policy != "empty_continue";
    assembly.config.align_range_limit = true;
    assembly.config.settle_type = "cash".into();
    assembly.run.inner = Box::new(InnerProbe {
        inner: assembly.run.inner,
        probe: probe.clone(),
        framework: policy == "framework",
    });
    assembly.run.strategy = Box::new(StrategyProbe {
        inner: assembly.run.strategy,
        probe: probe.clone(),
    });
    assembly.run.level_binding = Box::new(BindingProbe(probe.clone()));
    assembly.run.account = Box::new(AccountProbe(probe.clone()));
    let retained = Arc::new(Mutex::new(None));
    assembly.run.return_sink = Some(Box::new(SinkProbe {
        probe,
        retained: retained.clone(),
    }));
    Harness {
        executor: LiveNestedExecutor::new(assembly.calendar, assembly.config, assembly.run),
        events,
        calendar,
        retained,
    }
}

#[test]
fn native_parent_immediate_and_framework_tracked_start_failures_preserve_terminal_state() {
    for policy in ["immediate", "framework"] {
        let mut run = harness("inner.begin", policy);
        if policy == "framework" {
            assert!(matches!(
                run.executor.resume(NestedExecutorResume::Continue).unwrap(),
                LiveNestedEvent::Suspended(LiveNestedControlEvent::TrackedDecision(_))
            ));
            assert!(!run.events.lock().unwrap().contains(&"inner.begin"));
        }
        assert!(matches!(
            complete(&mut run.executor),
            Err(LiveNestedError::Inner(_))
        ));
        assert_eq!(run.events.lock().unwrap().last(), Some(&"inner.begin"));
        assert_eq!(
            run.executor.resume(NestedExecutorResume::Continue).err(),
            Some(LiveNestedError::Failed)
        );
        assert_eq!(NestedCalendar::trade_step(&*run.calendar).unwrap(), 0);
    }
    let mut run = harness("", "empty_continue");
    let done = complete(&mut run.executor).unwrap();
    assert_eq!(done.decisions.len(), 1);
    assert!(run.events.lock().unwrap().contains(&"decision.empty"));
    assert!(run.events.lock().unwrap().contains(&"strategy.post"));
    assert_eq!(NestedCalendar::trade_step(&*run.calendar).unwrap(), 1);
}

struct DefaultHooks;
impl LiveNestedStrategy for DefaultHooks {
    fn reset(&mut self, _: &LiveDecisionHandle) -> Result<(), NestedStrategyError> {
        Ok(())
    }
    fn alter_outer_decision(
        &mut self,
        decision: LiveDecisionHandle,
    ) -> Result<LiveDecisionHandle, NestedStrategyError> {
        Ok(decision)
    }
    fn begin(
        &mut self,
        _: Option<&SharedNestedResult>,
    ) -> Result<LiveNestedStrategyProgress, NestedStrategyError> {
        Ok(LiveNestedStrategyProgress::Suspended(
            NestedStrategyPrompt {
                kind: "unsupported".into(),
                schema_version: 1,
                payload: vec![],
            },
        ))
    }
    fn post_execute(&mut self, _: &SharedNestedResult) -> Result<(), NestedStrategyError> {
        panic!("unreachable after unsupported resume")
    }
    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        panic!("unreachable after unsupported resume")
    }
}

#[test]
fn native_parent_default_strategy_close_is_inert_and_none_resume_fails_terminally() {
    for close in [false, true] {
        let (mut factory, original) = factory(1, false);
        let mut assembly = factory.assemble(&original, 0).unwrap();
        assembly.config.decision_tracking.outer = false;
        assembly.run.strategy = Box::new(DefaultHooks);
        let mut executor =
            LiveNestedExecutor::new(assembly.calendar, assembly.config, assembly.run);
        assert!(matches!(
            executor.resume(NestedExecutorResume::Continue).unwrap(),
            LiveNestedEvent::Suspended(LiveNestedControlEvent::StrategyPrompt(_))
        ));
        if close {
            executor.close().unwrap();
            assert_eq!(
                executor.resume(NestedExecutorResume::Continue).err(),
                Some(LiveNestedError::Closed)
            );
        } else {
            let failure = executor
                .resume(NestedExecutorResume::Continue)
                .err()
                .unwrap();
            assert_eq!(
                failure,
                LiveNestedError::Strategy(NestedStrategyError {
                    message: "live strategy is not suspended".into()
                })
            );
            assert_eq!(
                executor.resume(NestedExecutorResume::Continue).err(),
                Some(LiveNestedError::Failed)
            );
        }
        assert_eq!(NestedCalendar::trade_step(&*factory.calendar).unwrap(), 0);
        assert!(factory.posted.lock().unwrap().is_none());
    }
}

fn complete(executor: &mut LiveNestedExecutor) -> Result<LiveNestedCollection, LiveNestedError> {
    let mut input = NestedExecutorResume::Continue;
    loop {
        match executor.resume(input)? {
            LiveNestedEvent::Complete(done) => return Ok(done),
            LiveNestedEvent::Suspended(LiveNestedControlEvent::StrategyPrompt(_)) => {
                input = NestedExecutorResume::Action(Some(42.0));
            }
            LiveNestedEvent::Suspended(LiveNestedControlEvent::TrackedDecision(_)) => {
                input = NestedExecutorResume::Continue;
            }
        }
    }
}

#[test]
fn native_parent_all_reached_failure_stages_terminate_without_later_callbacks() {
    let stages = [
        "settle",
        "outer.init_time",
        "inner.reset",
        "binding",
        "strategy.reset",
        "inner.finished",
        "decision.update",
        "strategy.alter",
        "decision.empty",
        "decision.range",
        "inner.len",
        "inner.step_index",
        "strategy.begin",
        "strategy.resume",
        "decision.propagate",
        "inner.time",
        "inner.begin",
        "inner.resume",
        "strategy.post",
        "inner.indicator",
        "strategy.upper",
        "outer.final_time",
        "account",
        "outer.step",
        "commit",
        "sink",
    ];
    let mut baseline = harness("", "normal");
    let done = complete(&mut baseline.executor).unwrap();
    let expected = baseline.events.lock().unwrap().clone();
    let sink = baseline.retained.lock().unwrap().clone().unwrap();
    assert!(Arc::ptr_eq(&sink, &done.executions));
    assert!(done.executions.lock().unwrap().is_empty());
    for stage in stages {
        let mut run = harness(stage, "normal");
        let failure = complete(&mut run.executor).err().unwrap();
        assert!(
            !matches!(
                failure,
                LiveNestedError::Complete | LiveNestedError::Closed | LiveNestedError::Failed
            ),
            "{stage}"
        );
        let index = expected.iter().position(|&entry| entry == stage).unwrap();
        assert_eq!(*run.events.lock().unwrap(), expected[..=index], "{stage}");
        assert_eq!(
            run.executor.resume(NestedExecutorResume::Continue).err(),
            Some(LiveNestedError::Failed)
        );
        run.executor.close().unwrap();
        assert_eq!(
            *run.events.lock().unwrap(),
            expected[..=index],
            "no retry after {stage}"
        );
        assert_eq!(
            NestedCalendar::trade_step(&*run.calendar).unwrap(),
            i64::from(matches!(stage, "commit" | "sink"))
        );
        if stage == "sink" {
            assert!(
                run.retained
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .is_empty()
            );
        }
    }
}

#[test]
fn native_parent_skip_policies_and_post_poison_preserve_source_effect_boundaries() {
    for policy in ["empty", "skip", "above"] {
        let mut run = harness("", policy);
        let done = complete(&mut run.executor).unwrap();
        assert!(done.decisions.is_empty());
        assert!(done.inner_order_indicators.is_empty());
        let events = run.events.lock().unwrap();
        assert!(!events.contains(&"strategy.begin"));
        assert_eq!(events.contains(&"decision.range"), policy != "empty");
        assert_eq!(events.contains(&"inner.step"), policy != "empty");
        assert_eq!(
            &events[events.len() - 6..],
            [
                "strategy.upper",
                "outer.final_time",
                "account",
                "outer.step",
                "commit",
                "sink"
            ]
        );
    }
    let mut run = harness("inner.step", "skip");
    assert!(matches!(
        complete(&mut run.executor),
        Err(LiveNestedError::Calendar(_))
    ));
    assert!(!run.events.lock().unwrap().contains(&"strategy.upper"));
    let mut run = harness("result.poison", "normal");
    assert!(matches!(
        complete(&mut run.executor),
        Err(LiveNestedError::Strategy(_))
    ));
    assert_eq!(run.events.lock().unwrap().last(), Some(&"strategy.post"));
    assert_eq!(NestedCalendar::trade_step(&*run.calendar).unwrap(), 0);
}
