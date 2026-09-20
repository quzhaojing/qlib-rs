use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    AtomicBarEnd, AtomicExecutorAccount, AtomicExecutorAccountError, AtomicExecutorLifecycle,
    AtomicExecutorLifecycleError, ExecutionPosition, ExecutionPositionError, ExecutionTarget,
    ExecutionTargetError, ExecutorDecisionTracker, ExecutorDecisionTrackerError,
    ExecutorLifecycleCalendar, ExecutorLifecycleCalendarError, ExecutorReturnSink,
    ExecutorReturnSinkError, IdxTradeRange, IndicatorConfig, IndicatorWeightMethod, Order,
    OrderDealResult, OrderDecision, OrderDir, OrderTradeDecision, SharedTradeRange,
    SimulatorCalendar, SimulatorCalendarError, SimulatorCollector, SimulatorDealProvider,
    SimulatorExecutionLog, SimulatorExecutionReporter, SimulatorReporterError, TradeRange,
    TradeRangeByTime, TradeRangeError,
};
use serde_json::{Value, json};

#[path = "support/position_cases.rs"]
mod position_cases;
#[path = "support/position_cash_cases.rs"]
mod position_cash_cases;
#[path = "support/shared_executor_cases.rs"]
mod shared_executor_cases;

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

fn decision(range: Option<SharedTradeRange>) -> OrderTradeDecision {
    let start = timestamp("2024-01-02 09:30:00");
    let end = timestamp("2024-01-02 10:00:00");
    OrderTradeDecision::from_orders(
        vec![Order::new("A", 2.0, OrderDir::Buy, Some(start), Some(end))],
        start,
        end,
        range,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Start,
    Target,
    Bar,
    Commit,
}

#[derive(Clone, Copy)]
enum Failure {
    Account(Stage),
    SimulatorTime,
    LifecycleTime,
    LifecycleStep,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Track,
    SettleStart(String),
    Target,
    SimulatorTime,
    Deal(Option<u64>),
    LifecycleTime,
    Bar,
    LifecycleStep,
    SettleCommit,
    Return(usize),
}

struct QueueSimulatorCalendar {
    values: Mutex<VecDeque<Result<(NaiveDateTime, NaiveDateTime), SimulatorCalendarError>>>,
    events: Arc<Mutex<Vec<Event>>>,
}

impl SimulatorCalendar for QueueSimulatorCalendar {
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SimulatorCalendarError> {
        self.events.lock().unwrap().push(Event::SimulatorTime);
        self.values.lock().unwrap().pop_front().unwrap()
    }
}

struct LifecycleCalendar {
    events: Arc<Mutex<Vec<Event>>>,
    fail_time: bool,
    fail_step: bool,
}

impl ExecutorLifecycleCalendar for LifecycleCalendar {
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), ExecutorLifecycleCalendarError> {
        self.events.lock().unwrap().push(Event::LifecycleTime);
        if self.fail_time {
            return Err(ExecutorLifecycleCalendarError {
                message: "time".to_owned(),
            });
        }
        Ok((
            timestamp("2024-01-02 09:30:00"),
            timestamp("2024-01-02 10:00:00"),
        ))
    }

    fn step(&self) -> Result<(), ExecutorLifecycleCalendarError> {
        self.events.lock().unwrap().push(Event::LifecycleStep);
        if self.fail_step {
            return Err(ExecutorLifecycleCalendarError {
                message: "step".to_owned(),
            });
        }
        Ok(())
    }
}

struct Dealer {
    events: Arc<Mutex<Vec<Event>>>,
}

impl SimulatorDealProvider for Dealer {
    fn deal_order(
        &self,
        order: &mut Order,
        _account: &mut dyn ExecutionTarget,
        _dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, domain_core::ExchangeDealError> {
        self.events
            .lock()
            .unwrap()
            .push(Event::Deal(order.factor().map(f64::to_bits)));
        order.set_deal_amount(1.0);
        order.set_factor(Some(2.0));
        Ok(OrderDealResult {
            trade_value: 10.0,
            trade_cost: 1.0,
            trade_price: 10.0,
        })
    }
}

struct SilentReporter;

impl SimulatorExecutionReporter for SilentReporter {
    fn report(&self, _log: SimulatorExecutionLog<'_>) -> Result<(), SimulatorReporterError> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct BarSnapshot {
    start: NaiveDateTime,
    end: NaiveDateTime,
    atomic: bool,
    decision_start: NaiveDateTime,
    decision_end: NaiveDateTime,
    order_count: usize,
    has_range: bool,
    trade_values: Vec<f64>,
    indicator_config: IndicatorConfig,
}

struct Account {
    shared_bars: Vec<domain_core::shared_executor_lifecycle::SharedAtomicResult>,
    events: Arc<Mutex<Vec<Event>>>,
    fail: Option<Stage>,
    bars: Vec<BarSnapshot>,
    order: domain_core::SharedOrderIndicator<domain_core::NumpyOrderIndicator>,
}

impl ExecutionPosition for Account {
    fn check_stock(&self, _stock: &str) -> Result<bool, ExecutionPositionError> {
        Ok(true)
    }

    fn stock_amount(&self, _stock: &str) -> Result<f64, ExecutionPositionError> {
        Ok(100.0)
    }

    fn cash(&self) -> Result<f64, ExecutionPositionError> {
        Ok(100.0)
    }
}

impl ExecutionTarget for Account {
    fn position(&self) -> Result<&dyn ExecutionPosition, ExecutionTargetError> {
        Ok(self)
    }

    fn update_order(
        &mut self,
        _order: &Order,
        _trade_value: f64,
        _trade_cost: f64,
        _trade_price: f64,
    ) -> Result<(), ExecutionTargetError> {
        Ok(())
    }
}

impl AtomicExecutorAccount for Account {
    fn execution_target(&mut self) -> Result<&mut dyn ExecutionTarget, AtomicExecutorAccountError> {
        self.events.lock().unwrap().push(Event::Target);
        fail_account(self.fail, Stage::Target)?;
        Ok(self)
    }

    fn settle_start(&mut self, settle_type: &str) -> Result<(), AtomicExecutorAccountError> {
        self.events
            .lock()
            .unwrap()
            .push(Event::SettleStart(settle_type.to_owned()));
        fail_account(self.fail, Stage::Start)
    }

    fn update_bar_end(&mut self, bar: AtomicBarEnd<'_>) -> Result<(), AtomicExecutorAccountError> {
        self.events.lock().unwrap().push(Event::Bar);
        self.bars.push(BarSnapshot {
            start: bar.trade_start_time,
            end: bar.trade_end_time,
            atomic: bar.atomic,
            decision_start: bar.outer_decision.start_time,
            decision_end: bar.outer_decision.end_time,
            order_count: bar.outer_decision.order_count,
            has_range: bar.outer_decision.has_trade_range,
            trade_values: bar
                .trade_info
                .iter()
                .map(|execution| execution.trade_value)
                .collect(),
            indicator_config: bar.indicator_config,
        });
        fail_account(self.fail, Stage::Bar)
    }

    fn settle_commit(&mut self) -> Result<(), AtomicExecutorAccountError> {
        self.events.lock().unwrap().push(Event::SettleCommit);
        fail_account(self.fail, Stage::Commit)
    }

    fn order_indicator_snapshot(
        &self,
    ) -> Result<domain_core::NumpyOrderIndicator, AtomicExecutorAccountError> {
        Ok(domain_core::NumpyOrderIndicator::default())
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<
        domain_core::SharedOrderIndicator<domain_core::NumpyOrderIndicator>,
        AtomicExecutorAccountError,
    > {
        Ok(self.order.clone())
    }
}

fn fail_account(fail: Option<Stage>, current: Stage) -> Result<(), AtomicExecutorAccountError> {
    if fail == Some(current) {
        return Err(AtomicExecutorAccountError {
            message: format!("{current:?}"),
        });
    }
    Ok(())
}

struct Tracker {
    events: Arc<Mutex<Vec<Event>>>,
    fail: bool,
}

impl ExecutorDecisionTracker for Tracker {
    fn track(&self, decision: &mut dyn OrderDecision) -> Result<(), ExecutorDecisionTrackerError> {
        self.events.lock().unwrap().push(Event::Track);
        decision.orders_mut()[0].set_factor(Some(7.0));
        if self.fail {
            return Err(ExecutorDecisionTrackerError {
                message: "track".to_owned(),
            });
        }
        Ok(())
    }
}

struct ReturnSink {
    events: Arc<Mutex<Vec<Event>>>,
    fail: bool,
}

impl ExecutorReturnSink for ReturnSink {
    fn store_execute_result(
        &mut self,
        executions: &[domain_core::OrderExecution<'_>],
    ) -> Result<(), ExecutorReturnSinkError> {
        self.events
            .lock()
            .unwrap()
            .push(Event::Return(executions.len()));
        if self.fail {
            return Err(ExecutorReturnSinkError {
                message: "return".to_owned(),
            });
        }
        Ok(())
    }
}

struct Harness {
    lifecycle: AtomicExecutorLifecycle,
    collector: SimulatorCollector,
    account: Account,
    events: Arc<Mutex<Vec<Event>>>,
}

fn harness(track_data: bool, settle_type: &str, failure: Option<Failure>) -> Harness {
    let events = Arc::new(Mutex::new(Vec::new()));
    let time = timestamp("2024-01-02 09:30:00");
    let first = if matches!(failure, Some(Failure::SimulatorTime)) {
        Err(SimulatorCalendarError {
            message: "simulator time".to_owned(),
        })
    } else {
        Ok((time, time))
    };
    let simulator_calendar = QueueSimulatorCalendar {
        values: Mutex::new(VecDeque::from([first, Ok((time, time))])),
        events: Arc::clone(&events),
    };
    let lifecycle_calendar = LifecycleCalendar {
        events: Arc::clone(&events),
        fail_time: matches!(failure, Some(Failure::LifecycleTime)),
        fail_step: matches!(failure, Some(Failure::LifecycleStep)),
    };
    let indicator_config = IndicatorConfig {
        fulfill_rate: IndicatorWeightMethod::AmountWeighted,
        price_advantage: IndicatorWeightMethod::ValueWeighted,
    };
    Harness {
        lifecycle: AtomicExecutorLifecycle::new(
            Arc::new(lifecycle_calendar),
            track_data,
            settle_type,
            indicator_config,
        ),
        collector: SimulatorCollector::new(
            "serial",
            Arc::new(simulator_calendar),
            Arc::new(Dealer {
                events: Arc::clone(&events),
            }),
            Arc::new(SilentReporter),
            false,
        ),
        account: Account {
            shared_bars: Vec::new(),
            events: Arc::clone(&events),
            fail: match failure {
                Some(Failure::Account(stage)) => Some(stage),
                _ => None,
            },
            bars: Vec::new(),
            order: Arc::default(),
        },
        events,
    }
}

#[test]
fn plain_atomic_step_forwards_bar_data_and_returns_shared_execution() {
    let mut harness = harness(false, "None", None);
    let tracker = Tracker {
        events: Arc::clone(&harness.events),
        fail: false,
    };
    let mut input = decision(None);
    let collection = harness
        .lifecycle
        .collect_data(
            &mut harness.collector,
            &mut input,
            &mut harness.account,
            Some(&tracker),
            None,
            4,
        )
        .unwrap();
    assert_eq!(collection.execution_result().len(), 1);
    assert_eq!(
        collection.execution_result().as_ptr(),
        collection.trade_info().as_ptr()
    );
    assert_eq!(
        harness.events.lock().unwrap().as_slice(),
        [
            Event::Target,
            Event::SimulatorTime,
            Event::SimulatorTime,
            Event::Deal(None),
            Event::LifecycleTime,
            Event::Bar,
            Event::LifecycleStep,
        ]
    );
    assert_eq!(harness.account.bars.len(), 1);
    assert_eq!(
        harness.account.bars[0],
        BarSnapshot {
            start: timestamp("2024-01-02 09:30:00"),
            end: timestamp("2024-01-02 10:00:00"),
            atomic: true,
            decision_start: timestamp("2024-01-02 09:30:00"),
            decision_end: timestamp("2024-01-02 10:00:00"),
            order_count: 1,
            has_range: false,
            trade_values: vec![10.0],
            indicator_config: IndicatorConfig {
                fulfill_rate: IndicatorWeightMethod::AmountWeighted,
                price_advantage: IndicatorWeightMethod::ValueWeighted,
            },
        }
    );
}

#[test]
fn tracked_settlement_and_return_sink_preserve_python_order() {
    let mut harness = harness(true, "cash", None);
    let tracker = Tracker {
        events: Arc::clone(&harness.events),
        fail: false,
    };
    let mut sink = ReturnSink {
        events: Arc::clone(&harness.events),
        fail: false,
    };
    let mut input = decision(None);
    harness
        .lifecycle
        .collect_data(
            &mut harness.collector,
            &mut input,
            &mut harness.account,
            Some(&tracker),
            Some(&mut sink),
            3,
        )
        .unwrap();
    assert_eq!(
        harness.events.lock().unwrap().as_slice(),
        [
            Event::Track,
            Event::SettleStart("cash".to_owned()),
            Event::Target,
            Event::SimulatorTime,
            Event::SimulatorTime,
            Event::Deal(Some(7.0_f64.to_bits())),
            Event::LifecycleTime,
            Event::Bar,
            Event::LifecycleStep,
            Event::SettleCommit,
            Event::Return(1),
        ]
    );
}

#[test]
fn atomic_range_check_rejects_indices_accepts_missing_calendar_and_propagates_failure() {
    let mut indexed = harness(false, "None", None);
    let mut input = decision(Some(Arc::new(IdxTradeRange::new(-2, 9))));
    assert_eq!(
        indexed
            .lifecycle
            .collect_data(
                &mut indexed.collector,
                &mut input,
                &mut indexed.account,
                None,
                None,
                0,
            )
            .unwrap_err(),
        AtomicExecutorLifecycleError::UnsupportedRange {
            start_idx: -2,
            end_idx: 9,
        }
    );
    assert!(indexed.events.lock().unwrap().is_empty());

    let mut timed = harness(true, "None", None);
    let timed_range = TradeRangeByTime::parse("09:30", "10:30").unwrap();
    let mut input = decision(Some(Arc::new(timed_range)));
    timed
        .lifecycle
        .collect_data(
            &mut timed.collector,
            &mut input,
            &mut timed.account,
            None,
            None,
            0,
        )
        .unwrap();
    assert!(timed.account.bars[0].has_range);
    assert!(!timed.events.lock().unwrap().contains(&Event::Track));

    let mut failed = harness(false, "None", None);
    let mut input = decision(Some(Arc::new(FailingRange)));
    assert!(matches!(
        failed.lifecycle.collect_data(
            &mut failed.collector,
            &mut input,
            &mut failed.account,
            None,
            None,
            0,
        ),
        Err(AtomicExecutorLifecycleError::Decision(_))
    ));
    assert!(failed.events.lock().unwrap().is_empty());
}

struct FailingRange;

impl TradeRange for FailingRange {
    fn range_indices(
        &self,
        _calendar: Option<&dyn domain_core::TradeCalendarRange>,
    ) -> Result<(i64, i64), TradeRangeError> {
        Err(TradeRangeError::IndexTimeClippingUnsupported)
    }

    fn clip_time_range(
        &self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(NaiveDateTime, NaiveDateTime), TradeRangeError> {
        unreachable!()
    }
}

#[test]
fn every_lifecycle_failure_stops_before_later_side_effects() {
    let tracker_events = Arc::new(Mutex::new(Vec::new()));
    let failing_tracker = Tracker {
        events: Arc::clone(&tracker_events),
        fail: true,
    };
    let mut tracked = harness(true, "cash", None);
    let mut input = decision(None);
    assert!(matches!(
        tracked.lifecycle.collect_data(
            &mut tracked.collector,
            &mut input,
            &mut tracked.account,
            Some(&failing_tracker),
            None,
            0,
        ),
        Err(AtomicExecutorLifecycleError::Tracker(_))
    ));
    assert_eq!(tracker_events.lock().unwrap().as_slice(), [Event::Track]);
    assert!(tracked.events.lock().unwrap().is_empty());

    assert_failure(Failure::Account(Stage::Start), "account");
    assert_failure(Failure::Account(Stage::Target), "account");
    assert_failure(Failure::SimulatorTime, "collection");
    assert_failure(Failure::LifecycleTime, "calendar");
    assert_failure(Failure::Account(Stage::Bar), "account");
    assert_failure(Failure::LifecycleStep, "calendar");
    assert_failure(Failure::Account(Stage::Commit), "account");

    let mut returned = harness(false, "cash", None);
    let mut sink = ReturnSink {
        events: Arc::clone(&returned.events),
        fail: true,
    };
    let mut input = decision(None);
    assert!(matches!(
        returned.lifecycle.collect_data(
            &mut returned.collector,
            &mut input,
            &mut returned.account,
            None,
            Some(&mut sink),
            0,
        ),
        Err(AtomicExecutorLifecycleError::ReturnSink(_))
    ));
    assert!(matches!(
        returned.events.lock().unwrap().as_slice(),
        [.., Event::SettleCommit, Event::Return(1)]
    ));
}

fn assert_failure(failure: Failure, expected: &str) {
    let mut harness = harness(false, "cash", Some(failure));
    let mut input = decision(None);
    let error = harness
        .lifecycle
        .collect_data(
            &mut harness.collector,
            &mut input,
            &mut harness.account,
            None,
            None,
            0,
        )
        .unwrap_err();
    assert!(matches!(
        (expected, error),
        ("account", AtomicExecutorLifecycleError::Account(_))
            | ("collection", AtomicExecutorLifecycleError::Collection(_))
            | ("calendar", AtomicExecutorLifecycleError::Calendar(_))
    ));
}

fn live_python_snapshot() -> Value {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/executor.py");
    let script = r"
import ast,json,sys
from types import GeneratorType
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='BaseExecutor');n=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='collect_data');n.returns=None
for a in n.args.args:a.annotation=None
class NestedExecutor:pass
class BasePosition:ST_NO='None'
ns={'GeneratorType':GeneratorType,'NestedExecutor':NestedExecutor,'BasePosition':BasePosition};exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),p,'exec'),ns)
e=[]
class D:
 def get_range_limit(self,**kw):return None
class P:
 def settle_start(self,x):e.append('start:'+x)
 def settle_commit(self):e.append('commit')
class A:
 current_position=P()
 def update_bar_end(self,*a,**kw):e.append('bar:'+str(kw['atomic'])+':'+str(len(kw['trade_info'])))
class C:
 def get_step_time(self):e.append('time');return 1,2
 def step(self):e.append('step')
class R(dict):
 def update(self,v):e.append('return:'+str(len(v['execute_result'])));super().update(v)
class X:
 collect_data=ns['collect_data'];track_data=True;_settle_type='cash';trade_account=A();trade_calendar=C();trade_exchange=object();indicator_config={}
 def _collect_data(self,trade_decision,level=0):e.append('collect:'+str(level));return ['R'],{'trade_info':['T']}
x=X();d=D();r=R();g=x.collect_data(d,return_value=r,level=3);assert next(g) is d;e.append('track')
try:next(g)
except StopIteration as z:ret=z.value
print(json.dumps({'events':e,'ret':ret,'stored':r['execute_result']},separators=(',',':')))
";
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn atomic_lifecycle_matches_live_python_source() {
    let python = live_python_snapshot();
    let mut harness = harness(true, "cash", None);
    let tracker = Tracker {
        events: Arc::clone(&harness.events),
        fail: false,
    };
    let mut sink = ReturnSink {
        events: Arc::clone(&harness.events),
        fail: false,
    };
    let mut input = decision(None);
    let result = harness
        .lifecycle
        .collect_data(
            &mut harness.collector,
            &mut input,
            &mut harness.account,
            Some(&tracker),
            Some(&mut sink),
            3,
        )
        .unwrap();
    let events: Vec<String> = harness
        .events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|event| match event {
            Event::Track => Some("track".to_owned()),
            Event::SettleStart(value) => Some(format!("start:{value}")),
            Event::Deal(_) => Some("collect:3".to_owned()),
            Event::LifecycleTime => Some("time".to_owned()),
            Event::Bar => Some("bar:True:1".to_owned()),
            Event::LifecycleStep => Some("step".to_owned()),
            Event::SettleCommit => Some("commit".to_owned()),
            Event::Return(count) => Some(format!("return:{count}")),
            Event::Target | Event::SimulatorTime => None,
        })
        .collect();
    let rust = json!({
        "events": events,
        "ret": ["R"],
        "stored": if result.execution_result().len() == 1 { json!(["R"]) } else { json!([]) },
    });
    assert_eq!(rust, python);
}
