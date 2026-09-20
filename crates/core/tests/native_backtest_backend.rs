use std::sync::{Arc, Mutex, RwLock};

use chrono::NaiveDateTime;
use domain_core::{
    Account, BacktestLoop, BacktestLoopBackend, BacktestLoopEvent, BacktestProgress,
    IndicatifBacktestProgress, InfinitePosition, NativeBacktestBackend, NativeBacktestBackendError,
    NestedCalendar, NestedCalendarError, NestedControlEvent, NestedExecutorResume,
    NestedInnerExecutor, NestedInnerExecutorError, NestedInnerProgress, NumpyOrderIndicator, Order,
    OrderDecision, OrderDir, OuterBacktestStrategy, OuterStrategyInfrastructure,
    OwnedOrderExecution, SaoeCalendar, SaoeOrderFactory, SaoePluginError,
    SharedAccountReportSource, SharedOrderExecution, SharedOrderFactory, SingleOrderOuterStrategy,
    SingleOrderStrategy, run_backtest_loop,
};

fn at() -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
}

struct Calendar;

impl SaoeCalendar for Calendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        Ok((0, 0))
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        Ok((at(), at()))
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

struct FailingOrders;

impl SaoeOrderFactory for FailingOrders {
    fn create(
        &mut self,
        _id: &str,
        _amount: Option<f64>,
        _direction: OrderDir,
    ) -> Result<Order, SaoePluginError> {
        Err(SaoePluginError {
            message: "order construction".to_owned(),
        })
    }
}

#[derive(Default)]
struct ExecutorState {
    events: Vec<&'static str>,
    step: usize,
}

struct Executor {
    state: Arc<Mutex<ExecutorState>>,
    indicator: Arc<RwLock<NumpyOrderIndicator>>,
    pending: Option<Box<dyn OrderDecision>>,
    suspended: bool,
    fail: Option<&'static str>,
}

impl Executor {
    fn new(suspended: bool, fail: Option<&'static str>) -> Self {
        Self {
            state: Arc::default(),
            indicator: Arc::default(),
            pending: None,
            suspended,
            fail,
        }
    }

    fn stage(&self, stage: &'static str) -> Result<(), NestedInnerExecutorError> {
        self.state.lock().unwrap().events.push(stage);
        if self.fail == Some(stage) {
            Err(NestedInnerExecutorError {
                message: stage.to_owned(),
            })
        } else {
            Ok(())
        }
    }

    fn finish(
        &mut self,
        decision: Box<dyn OrderDecision>,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        self.stage("finish")?;
        self.state.lock().unwrap().step += 1;
        let execution = OwnedOrderExecution {
            order: decision.orders()[0].clone(),
            trade_value: 7.0,
            trade_cost: 0.0,
            trade_price: 1.0,
        }
        .into_shared();
        Ok(NestedInnerProgress::Complete {
            decision,
            executions: Arc::new(Mutex::new(vec![execution])),
        })
    }
}

impl NestedCalendar for Executor {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        self.stage("finished")
            .map_err(|error| NestedCalendarError {
                message: error.message,
            })?;
        Ok(self.state.lock().unwrap().step == 1)
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        self.stage("trade_len")
            .map_err(|error| NestedCalendarError {
                message: error.message,
            })?;
        Ok(1)
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        Ok(i64::try_from(self.state.lock().unwrap().step).unwrap())
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        Ok((at(), at()))
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        Ok(())
    }
}

impl NestedInnerExecutor for Executor {
    fn reset_window(
        &mut self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        self.stage("reset")
    }

    fn collect_data(
        &mut self,
        _decision: &mut dyn OrderDecision,
        _level: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError> {
        panic!("the owned begin protocol is used")
    }

    fn begin_collect_data(
        &mut self,
        decision: Box<dyn OrderDecision>,
        level: usize,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        assert_eq!(level, 0);
        self.stage("begin")?;
        if self.suspended {
            let event = NestedControlEvent::TrackedDecision(domain_core::TrackedOrderDecision {
                orders: decision.orders().to_vec(),
                start_time: decision.start_time(),
                end_time: decision.end_time(),
                has_trade_range: false,
            });
            self.pending = Some(decision);
            Ok(NestedInnerProgress::Suspended(event))
        } else {
            self.finish(decision)
        }
    }

    fn resume_collect_data(
        &mut self,
        input: NestedExecutorResume,
    ) -> Result<NestedInnerProgress, NestedInnerExecutorError> {
        assert!(matches!(
            input,
            NestedExecutorResume::Continue | NestedExecutorResume::Action(Some(4.0))
        ));
        self.stage("resume")?;
        let decision = self.pending.take().unwrap();
        self.finish(decision)
    }

    fn close_collect_data(&mut self) -> Result<(), NestedInnerExecutorError> {
        self.stage("close")?;
        self.pending = None;
        Ok(())
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<Arc<RwLock<NumpyOrderIndicator>>, NestedInnerExecutorError> {
        Ok(Arc::clone(&self.indicator))
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        Ok(self.indicator.read().unwrap().clone())
    }
}

#[derive(Default)]
struct Progress {
    events: Arc<Mutex<Vec<String>>>,
}

impl BacktestProgress for Progress {
    fn enter(&mut self, total: i64) -> Result<(), NativeBacktestBackendError> {
        self.events.lock().unwrap().push(format!("enter:{total}"));
        Ok(())
    }

    fn increment(&mut self) -> Result<(), NativeBacktestBackendError> {
        self.events.lock().unwrap().push("increment".to_owned());
        Ok(())
    }

    fn close(&mut self) -> Result<(), NativeBacktestBackendError> {
        self.events.lock().unwrap().push("close".to_owned());
        Ok(())
    }
}

fn infrastructure() -> OuterStrategyInfrastructure {
    let calendar: Arc<dyn SaoeCalendar> = Arc::new(Calendar);
    let orders: SharedOrderFactory = Arc::new(Mutex::new(Box::new(Orders)));
    OuterStrategyInfrastructure::new(calendar, orders)
}

fn backend(
    executor: Executor,
    progress: Progress,
    account: Arc<Mutex<Account>>,
) -> NativeBacktestBackend {
    NativeBacktestBackend::with_progress(
        Box::new(executor),
        Box::new(SingleOrderOuterStrategy::new(SingleOrderStrategy::new(
            Order::new("A", 7.0, OrderDir::Buy, None, None),
            None,
        ))),
        infrastructure(),
        Box::new(progress),
        Box::new(SharedAccountReportSource::new(vec![(
            "day".to_owned(),
            account,
        )])),
    )
}

#[test]
fn native_backend_runs_immediate_and_suspended_outer_graphs() {
    for suspended in [false, true] {
        let executor = Executor::new(suspended, None);
        let executor_state = Arc::clone(&executor.state);
        let progress = Progress::default();
        let progress_events = Arc::clone(&progress.events);
        let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
        let mut loop_ = BacktestLoop::new(backend(executor, progress, account), at(), at(), true);

        let first = loop_.resume(NestedExecutorResume::Continue).unwrap();
        if suspended {
            assert!(matches!(first, BacktestLoopEvent::Suspended(_)));
            assert!(matches!(
                loop_
                    .resume(NestedExecutorResume::Action(Some(4.0)))
                    .unwrap(),
                BacktestLoopEvent::Complete
            ));
        } else {
            assert!(matches!(first, BacktestLoopEvent::Complete));
        }
        assert_eq!(
            *progress_events.lock().unwrap(),
            ["enter:1", "increment", "close"]
        );
        assert_eq!(
            *executor_state.lock().unwrap().events,
            if suspended {
                vec![
                    "reset",
                    "trade_len",
                    "finished",
                    "begin",
                    "resume",
                    "finish",
                    "finished",
                ]
            } else {
                vec![
                    "reset",
                    "trade_len",
                    "finished",
                    "begin",
                    "finish",
                    "finished",
                ]
            }
        );
        let reports = loop_.reports().unwrap();
        assert!(reports.portfolio.is_empty());
        assert_eq!(
            reports
                .indicators
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["1day"]
        );
    }
}

#[test]
fn noninteractive_wrapper_ignores_control_yields_and_returns_owned_reports() {
    let executor = Executor::new(true, None);
    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let reports =
        run_backtest_loop(backend(executor, Progress::default(), account), at(), at()).unwrap();
    assert!(reports.portfolio.is_empty());
    assert_eq!(
        reports
            .indicators
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["1day"]
    );

    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let error = run_backtest_loop(
        backend(
            Executor::new(false, Some("reset")),
            Progress::default(),
            account,
        ),
        at(),
        at(),
    )
    .err()
    .unwrap();
    assert_eq!(error.to_string(), "nested inner executor error: reset");
}

#[test]
fn concrete_strategy_progress_and_cleanup_failures_are_explicit() {
    let mut strategy = SingleOrderOuterStrategy::new(SingleOrderStrategy::new(
        Order::new("A", 1.0, OrderDir::Sell, None, None),
        None,
    ));
    assert_eq!(
        strategy.generate(None).err().unwrap().to_string(),
        "outer backtest strategy error: outer strategy infrastructure is not initialized"
    );
    strategy.reset(infrastructure()).unwrap();
    let decision = strategy.generate(None).unwrap();
    assert_eq!(decision.orders()[0].direction(), OrderDir::Sell);
    strategy.post_execute(&Arc::default()).unwrap();
    strategy.post_upper_level().unwrap();

    let poisoned_orders: SharedOrderFactory = Arc::new(Mutex::new(Box::new(Orders)));
    let poison = Arc::clone(&poisoned_orders);
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.lock().unwrap();
            panic!("poison order factory")
        })
        .join()
        .is_err()
    );
    let mut poisoned_strategy = SingleOrderOuterStrategy::new(SingleOrderStrategy::new(
        Order::new("A", 1.0, OrderDir::Buy, None, None),
        None,
    ));
    poisoned_strategy
        .reset(OuterStrategyInfrastructure::new(
            Arc::new(Calendar),
            poisoned_orders,
        ))
        .unwrap();
    assert_eq!(
        poisoned_strategy.generate(None).err().unwrap().to_string(),
        "outer backtest strategy error: outer strategy order factory lock poisoned"
    );
    let failing_orders: SharedOrderFactory = Arc::new(Mutex::new(Box::new(FailingOrders)));
    let mut failing_strategy = SingleOrderOuterStrategy::new(SingleOrderStrategy::new(
        Order::new("A", 1.0, OrderDir::Buy, None, None),
        None,
    ));
    failing_strategy
        .reset(OuterStrategyInfrastructure::new(
            Arc::new(Calendar),
            failing_orders,
        ))
        .unwrap();
    assert_eq!(
        failing_strategy.generate(None).err().unwrap().to_string(),
        "outer backtest strategy error: SAOE plugin error: order construction"
    );

    let mut progress = IndicatifBacktestProgress::default();
    assert_eq!(
        progress.enter(-1).unwrap_err().to_string(),
        "outer backtest progress error: backtest progress total cannot be negative: -1"
    );
    progress.enter(1).unwrap();
    assert_eq!(
        progress.enter(1).unwrap_err().to_string(),
        "outer backtest progress error: backtest progress is already active"
    );
    progress.increment().unwrap();
    progress.close().unwrap();
    progress.close().unwrap();
    assert_eq!(
        progress.increment().unwrap_err().to_string(),
        "outer backtest progress error: backtest progress is not active"
    );

    let executor = Executor::new(true, None);
    let state = Arc::clone(&executor.state);
    let progress = Progress::default();
    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let mut backend = backend(executor, progress, account);
    let decision = strategy.generate(None).unwrap();
    assert!(matches!(
        backend.begin_collection(decision).unwrap(),
        NestedInnerProgress::Suspended(_)
    ));
    backend.close_collection().unwrap();
    assert_eq!(state.lock().unwrap().events, ["begin", "close"]);

    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let mut default_progress_backend = NativeBacktestBackend::new(
        Box::new(Executor::new(false, None)),
        Box::new(SingleOrderOuterStrategy::new(SingleOrderStrategy::new(
            Order::new("A", 1.0, OrderDir::Buy, None, None),
            None,
        ))),
        infrastructure(),
        Box::new(SharedAccountReportSource::new(vec![(
            "day".to_owned(),
            account,
        )])),
    );
    default_progress_backend.enter_progress(0).unwrap();
    default_progress_backend.close_progress().unwrap();
}

#[test]
fn executor_and_shared_report_failures_keep_component_diagnostics() {
    for stage in ["reset", "trade_len", "finished", "begin", "resume", "close"] {
        let mut executor = Executor::new(stage == "resume" || stage == "close", Some(stage));
        if stage == "resume" {
            executor.pending = Some(strategy_decision());
        }
        let mut backend = backend(
            executor,
            Progress::default(),
            Arc::new(Mutex::new(Account::new(InfinitePosition, false))),
        );
        let result = match stage {
            "reset" => backend.reset_executor(at(), at()),
            "trade_len" => backend.trade_len().map(|_| ()),
            "finished" => backend.finished().map(|_| ()),
            "begin" => backend.begin_collection(strategy_decision()).map(|_| ()),
            "resume" => backend
                .resume_collection(NestedExecutorResume::Action(Some(4.0)))
                .map(|_| ()),
            "close" => backend.close_collection(),
            _ => unreachable!(),
        };
        let prefix = if matches!(stage, "trade_len" | "finished") {
            "nested calendar error"
        } else {
            "nested inner executor error"
        };
        assert_eq!(
            result.unwrap_err().to_string(),
            format!("{prefix}: {stage}")
        );
    }

    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let poison = Arc::clone(&account);
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.lock().unwrap();
            panic!("poison report account")
        })
        .join()
        .is_err()
    );
    let mut source = SharedAccountReportSource::new(vec![("day".to_owned(), account)]);
    assert_eq!(
        domain_core::BacktestReportSource::collect(&mut source)
            .err()
            .unwrap()
            .to_string(),
        "shared report account lock poisoned at frequency day"
    );

    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let mut source = SharedAccountReportSource::new(vec![("bad".to_owned(), account)]);
    assert!(matches!(
        domain_core::BacktestReportSource::collect(&mut source),
        Err(NativeBacktestBackendError::Report(
            domain_core::BacktestReportError::Frequency(_)
        ))
    ));
}

fn strategy_decision() -> Box<dyn OrderDecision> {
    let mut strategy = SingleOrderOuterStrategy::new(SingleOrderStrategy::new(
        Order::new("A", 1.0, OrderDir::Buy, None, None),
        None,
    ));
    strategy.reset(infrastructure()).unwrap();
    strategy.generate(None).unwrap()
}
