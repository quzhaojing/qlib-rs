use super::*;
use domain_core::nested_executor::SharedNestedResult;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[path = "shared_result_sink.rs"]
mod shared_result_sink;

struct RetainingInner {
    inner: Inner,
    lists: Arc<Mutex<Vec<SharedNestedResult>>>,
}

impl NestedCalendar for RetainingInner {
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
        self.inner.step_time()
    }
    fn step(&self) -> Result<(), NestedCalendarError> {
        self.inner.step()
    }
}

impl NestedInnerExecutor for RetainingInner {
    fn reset_window(
        &mut self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        self.inner.reset_window(start, end)
    }
    fn collect_data(
        &mut self,
        _decision: &mut dyn OrderDecision,
        _level: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError> {
        panic!("must dispatch shared collection")
    }
    fn collect_shared_data(
        &mut self,
        decision: &mut dyn OrderDecision,
        level: usize,
    ) -> Result<SharedNestedResult, NestedInnerExecutorError> {
        let rows = Arc::new(Mutex::new(self.inner.collect_data(decision, level)?));
        self.lists.lock().unwrap().push(Arc::clone(&rows));
        Ok(rows)
    }
    fn order_indicator_handle(
        &self,
    ) -> Result<domain_core::SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError>
    {
        self.inner.order_indicator_handle()
    }
    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        self.inner.order_indicator_snapshot()
    }
}

struct RetainingSession {
    inner: RetainingInner,
    decision: Option<Box<dyn OrderDecision>>,
}

impl NestedChildSession for RetainingSession {
    fn begin(
        &mut self,
        decision: Box<dyn OrderDecision>,
        _level: usize,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        self.decision = Some(decision);
        Ok(NestedInnerProgress::Suspended(
            NestedControlEvent::StrategyPrompt(prompt(0)),
        ))
    }
    fn resume(
        &mut self,
        _input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        let mut decision = self.decision.take().unwrap();
        let executions = self.inner.collect_shared_data(&mut *decision, 1)?;
        Ok(NestedInnerProgress::Complete {
            decision,
            executions,
        })
    }
    fn close(&mut self) -> Result<(), NestedInnerExecutorError> {
        self.decision = None;
        Ok(())
    }
}

struct RetainingProxy {
    previous: Option<SharedNestedResult>,
    retained: Arc<Mutex<Vec<SharedNestedResult>>>,
    child_lists: Arc<Mutex<Vec<SharedNestedResult>>>,
    poison: bool,
}

impl NestedStrategy for RetainingProxy {
    fn reset(&mut self, _outer: &dyn NestedOuterDecision) -> Result<(), NestedStrategyError> {
        Ok(())
    }
    fn alter_outer_decision(
        &mut self,
        _outer: &mut dyn NestedOuterDecision,
    ) -> Result<(), NestedStrategyError> {
        Ok(())
    }
    fn generate_trade_decision(
        &mut self,
        _previous: Option<&[SharedOrderExecution]>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        panic!("must use shared begin")
    }
    fn begin_shared_trade_decision(
        &mut self,
        previous: Option<&SharedNestedResult>,
    ) -> Result<NestedStrategyProgress, NestedStrategyError> {
        self.previous = previous.cloned();
        if let Some(previous) = &self.previous {
            assert!(Arc::ptr_eq(previous, &self.retained.lock().unwrap()[0]));
            assert_eq!(previous.try_lock().unwrap().len(), 2);
        }
        Ok(NestedStrategyProgress::Suspended(prompt(0)))
    }
    fn resume_trade_decision(
        &mut self,
        volume: Option<f64>,
    ) -> Result<NestedStrategyProgress, NestedStrategyError> {
        if let Some(previous) = &self.previous {
            previous.try_lock().unwrap().clear();
        }
        Ok(NestedStrategyProgress::Ready(order(volume.unwrap())))
    }
    fn post_execute(
        &mut self,
        _executions: &[SharedOrderExecution],
    ) -> Result<(), NestedStrategyError> {
        panic!("must use shared post")
    }
    fn post_shared_execute(
        &mut self,
        executions: &SharedNestedResult,
    ) -> Result<(), NestedStrategyError> {
        assert!(Arc::ptr_eq(
            executions,
            self.child_lists.lock().unwrap().last().unwrap()
        ));
        let mut retained = self.retained.lock().unwrap();
        if retained.is_empty() {
            let mut rows = executions.try_lock().unwrap();
            let row = Arc::clone(&rows[0]);
            rows.push(row);
        }
        retained.push(Arc::clone(executions));
        if self.poison {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    let _guard = executions.lock().unwrap();
                    panic!("poison retained result");
                }))
                .is_err()
            );
        }
        Ok(())
    }
    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        Ok(())
    }
}

fn machine(
    state: &SharedState,
    retained: &Arc<Mutex<Vec<SharedNestedResult>>>,
    poison: bool,
    recursive: bool,
) -> ResumableNestedExecutor {
    machine_with_sink(state, retained, poison, recursive, None)
}

fn machine_with_sink(
    state: &SharedState,
    retained: &Arc<Mutex<Vec<SharedNestedResult>>>,
    poison: bool,
    recursive: bool,
    return_sink: Option<Box<dyn NestedExecutorReturnSink>>,
) -> ResumableNestedExecutor {
    let child_lists = Arc::new(Mutex::new(Vec::new()));
    let child = RetainingInner {
        inner: Inner {
            state: Arc::clone(state),
            len: 2,
        },
        lists: Arc::clone(&child_lists),
    };
    let inner: Box<dyn NestedInnerExecutor> = if recursive {
        Box::new(RecursiveNestedInnerAdapter::new(
            Box::new(Inner {
                state: Arc::clone(state),
                len: 2,
            }),
            Box::new(RetainingSession {
                inner: child,
                decision: None,
            }),
        ))
    } else {
        Box::new(child)
    };
    ResumableNestedExecutor::new(
        Box::new(OuterCalendar(Arc::clone(state))),
        config(false, "None"),
        ResumableNestedRun {
            level_binding: Box::new(Binding(Arc::clone(state))),
            inner,
            strategy: Box::new(RetainingProxy {
                previous: None,
                retained: Arc::clone(retained),
                child_lists,
                poison,
            }),
            outer: Box::new(Outer {
                state: Arc::clone(state),
                decision: outer_decision(),
                empty: false,
                range: None,
                replace: false,
            }),
            account: Box::new(Account(Arc::clone(state))),
            return_sink,
        },
    )
}

#[test]
fn previous_list_survives_real_strategy_suspension_and_mutates_without_flattened_aliasing() {
    let state = Arc::new(Mutex::new(State::default()));
    let retained = Arc::new(Mutex::new(Vec::new()));
    let mut machine = machine(&state, &retained, false, false);
    assert!(matches!(
        machine.resume(NestedExecutorResume::Continue).unwrap(),
        ResumableNestedEvent::StrategyPrompt(_)
    ));
    assert!(matches!(
        machine
            .resume(NestedExecutorResume::Action(Some(1.0)))
            .unwrap(),
        ResumableNestedEvent::StrategyPrompt(_)
    ));
    let first = Arc::clone(&retained.lock().unwrap()[0]);
    assert_eq!(first.try_lock().unwrap().len(), 2);
    let ResumableNestedEvent::Complete(result) = machine
        .resume(NestedExecutorResume::Action(Some(2.0)))
        .unwrap()
    else {
        panic!("must complete")
    };
    assert!(first.try_lock().unwrap().is_empty());
    assert_eq!(result.executions().lock().unwrap().len(), 3);
    {
        let rows = result.executions().lock().unwrap();
        assert!(Arc::ptr_eq(&rows[0], &rows[1]));
    }
    assert!(Arc::ptr_eq(
        &result.executions().lock().unwrap()[2],
        &retained.lock().unwrap()[1].lock().unwrap()[0]
    ));
    drop(machine);
    assert!(first.lock().unwrap().is_empty());
    assert_eq!(retained.lock().unwrap().len(), 2);
}

#[test]
fn poisoned_post_result_is_terminal_after_resumption_and_preserves_hook_mutation() {
    let state = Arc::new(Mutex::new(State::default()));
    let retained = Arc::new(Mutex::new(Vec::new()));
    let mut machine = machine(&state, &retained, true, false);
    assert!(matches!(
        machine.resume(NestedExecutorResume::Continue).unwrap(),
        ResumableNestedEvent::StrategyPrompt(_)
    ));
    let error = machine
        .resume(NestedExecutorResume::Action(Some(1.0)))
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("nested execution result list lock poisoned")
    );
    assert_eq!(state.lock().unwrap().step, 1);
    assert!(
        !state
            .lock()
            .unwrap()
            .events
            .iter()
            .any(|value| value == "handle" || value == "upper")
    );
    assert_eq!(
        retained.lock().unwrap()[0]
            .lock()
            .unwrap_err()
            .into_inner()
            .len(),
        2
    );
    assert!(machine.resume(NestedExecutorResume::Continue).is_err());
}

#[test]
fn recursive_child_completion_preserves_the_original_list_through_parent_hooks() {
    let state = Arc::new(Mutex::new(State::default()));
    let retained = Arc::new(Mutex::new(Vec::new()));
    let mut machine = machine(&state, &retained, false, true);
    let mut input = NestedExecutorResume::Continue;
    let mut prompts = 0;
    let result = loop {
        match machine.resume(input).unwrap() {
            ResumableNestedEvent::StrategyPrompt(_) => {
                prompts += 1;
                input = NestedExecutorResume::Action(Some(1.0));
            }
            ResumableNestedEvent::Complete(result) => break result,
            ResumableNestedEvent::TrackedDecision(_) => panic!("tracking disabled"),
        }
    };
    assert_eq!(prompts, 4);
    assert_eq!(result.executions().lock().unwrap().len(), 3);
    assert!(retained.lock().unwrap()[0].lock().unwrap().is_empty());
    {
        let rows = result.executions().lock().unwrap();
        assert!(Arc::ptr_eq(&rows[0], &rows[1]));
    }
    assert!(Arc::ptr_eq(
        &result.executions().lock().unwrap()[2],
        &retained.lock().unwrap()[1].lock().unwrap()[0]
    ));
}

#[test]
fn suspended_return_sink_mutates_the_returned_list_and_keeps_failure_side_effects() {
    for fail in [false, true] {
        let state = Arc::new(Mutex::new(State::default()));
        let child_lists = Arc::new(Mutex::new(Vec::new()));
        let retained = Arc::new(Mutex::new(None));
        let sink = shared_result_sink::ClearingResultSink {
            retained: Arc::clone(&retained),
            fail,
        };
        let mut machine =
            machine_with_sink(&state, &child_lists, false, true, Some(Box::new(sink)));
        let mut input = NestedExecutorResume::Continue;
        let result = loop {
            match machine.resume(input) {
                Ok(ResumableNestedEvent::StrategyPrompt(_)) => {
                    input = NestedExecutorResume::Action(Some(1.0));
                }
                Ok(ResumableNestedEvent::Complete(collection)) => break Ok(collection),
                Ok(ResumableNestedEvent::TrackedDecision(_)) => panic!("tracking disabled"),
                Err(error) => break Err(error),
            }
        };
        let handle = retained.lock().unwrap().as_ref().unwrap().clone();
        assert!(handle.lock().unwrap().is_empty());
        assert_eq!(child_lists.lock().unwrap()[1].lock().unwrap().len(), 1);
        if fail {
            assert!(
                result
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("sink after clear")
            );
            assert!(machine.resume(NestedExecutorResume::Continue).is_err());
        } else {
            assert!(Arc::ptr_eq(result.unwrap().executions(), &handle));
        }
    }
}
