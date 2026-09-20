use std::{
    collections::HashMap,
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::NaiveDateTime;
use domain_core::{
    Account, AccountBarMarket, AccountBarMarketError, AtomicAccountAdapter, AtomicBarEnd,
    AtomicExecutorAccount, AtomicExecutorAccountError, AtomicNestedInnerAdapter,
    BasePriceDataProvider, BasePriceProviderError, DenseOrderIndicator, ExecutionTarget,
    ExecutorLifecycleCalendar, ExecutorLifecycleCalendarError, IndicatorConfig, InfinitePosition,
    MarketDataValue, NestedAccountAdapter, NestedCalendar, NestedCalendarError,
    NestedDecisionUpdate, NestedExecutorCore, NestedExecutorLifecycle, NestedExecutorRun,
    NestedInnerExecutor, NestedLevelBinding, NestedLevelBindingError, NestedOuterDecision,
    NestedOuterDecisionError, NestedStrategy, NestedStrategyError, Order, OrderDealResult,
    OrderDecision, OrderDir, OrderIndicatorAggregationConfig, OrderTradeDecision,
    OwnedAtomicNestedInnerAdapter, ResettableNestedCalendar, SharedOrderExecution,
    SimulatorCalendar, SimulatorCalendarError, SimulatorCollector, SimulatorDealProvider,
    SimulatorExecutionLog, SimulatorExecutionReporter, SimulatorReporterError, TimeRange,
};
use serde_json::{Value, json};

#[path = "support/live_atomic_nested_cases.rs"]
mod live_atomic_nested_cases;
#[path = "support/live_nested_cases.rs"]
mod live_nested_cases;

fn time(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

#[derive(Debug)]
struct CalendarState {
    start: NaiveDateTime,
    end: NaiveDateTime,
    step: i64,
    reset_count: usize,
    fail_reset: bool,
}

struct SharedInnerCalendar(Mutex<CalendarState>);

impl SharedInnerCalendar {
    fn new(start: NaiveDateTime, end: NaiveDateTime) -> Self {
        Self(Mutex::new(CalendarState {
            start,
            end,
            step: 0,
            reset_count: 0,
            fail_reset: false,
        }))
    }
}

impl NestedCalendar for SharedInnerCalendar {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        Ok(self.0.lock().unwrap().step >= 1)
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        Ok(1)
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        Ok(self.0.lock().unwrap().step)
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        let state = self.0.lock().unwrap();
        Ok((state.start, state.end))
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        self.0.lock().unwrap().step += 1;
        Ok(())
    }
}

impl ResettableNestedCalendar for SharedInnerCalendar {
    fn reset_window(
        &self,
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(), NestedCalendarError> {
        let mut state = self.0.lock().unwrap();
        if state.fail_reset {
            return Err(NestedCalendarError {
                message: "reset failed".to_owned(),
            });
        }
        state.start = start_time;
        state.end = end_time;
        state.step = 0;
        state.reset_count += 1;
        Ok(())
    }
}

impl ExecutorLifecycleCalendar for SharedInnerCalendar {
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), ExecutorLifecycleCalendarError> {
        let state = self.0.lock().unwrap();
        Ok((state.start, state.end))
    }

    fn step(&self) -> Result<(), ExecutorLifecycleCalendarError> {
        self.0.lock().unwrap().step += 1;
        Ok(())
    }
}

impl SimulatorCalendar for SharedInnerCalendar {
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SimulatorCalendarError> {
        let state = self.0.lock().unwrap();
        Ok((state.start, state.end))
    }
}

struct ApplyingDealer {
    fail: bool,
}

impl SimulatorDealProvider for ApplyingDealer {
    fn deal_order(
        &self,
        order: &mut Order,
        account: &mut dyn ExecutionTarget,
        _dealt_order_amount: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, domain_core::ExchangeDealError> {
        if self.fail {
            return Err(domain_core::ExchangeDealError::Target(
                domain_core::ExecutionTargetError {
                    message: "deal failed".to_owned(),
                },
            ));
        }
        order.set_deal_amount(order.amount());
        let trade_price = 10.0;
        let trade_value = order.amount() * trade_price;
        account.update_order(order, trade_value, 0.0, trade_price)?;
        Ok(OrderDealResult {
            trade_value,
            trade_cost: 0.0,
            trade_price,
        })
    }
}

struct SilentReporter;

impl SimulatorExecutionReporter for SilentReporter {
    fn report(&self, _log: SimulatorExecutionLog<'_>) -> Result<(), SimulatorReporterError> {
        Ok(())
    }
}

struct Services;

impl AccountBarMarket for Services {
    fn is_suspended(&self, _stock: &str, _range: TimeRange) -> Result<bool, AccountBarMarketError> {
        Ok(false)
    }

    fn close(&self, _stock: &str, _range: TimeRange) -> Result<f64, AccountBarMarketError> {
        Ok(10.0)
    }
}

impl BasePriceDataProvider for Services {
    fn deal_price(
        &self,
        _stock: &str,
        _range: TimeRange,
        _direction: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        Ok(Some(MarketDataValue::Scalar(10.0)))
    }

    fn volume(
        &self,
        _stock: &str,
        _range: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        Ok(Some(MarketDataValue::Scalar(1.0)))
    }
}

fn collector(calendar: Arc<SharedInnerCalendar>, fail: bool) -> SimulatorCollector {
    SimulatorCollector::new(
        "serial",
        calendar,
        Arc::new(ApplyingDealer { fail }),
        Arc::new(SilentReporter),
        false,
    )
}

fn decision(start: NaiveDateTime, end: NaiveDateTime) -> OrderTradeDecision {
    OrderTradeDecision::from_orders(
        vec![Order::new("A", 2.0, OrderDir::Buy, Some(start), Some(end))],
        start,
        end,
        None,
    )
}

#[test]
fn atomic_inner_adapter_runs_real_lifecycle_and_owns_executions() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let mut collector = collector(Arc::clone(&calendar), false);
    let services = Services;
    let mut account = Account::new(InfinitePosition, false);
    let mut account_adapter = AtomicAccountAdapter::new(&mut account, &services, false);
    let mut inner = AtomicNestedInnerAdapter::new(
        calendar.clone(),
        &mut collector,
        &mut account_adapter,
        "None",
        IndicatorConfig::default(),
    );

    assert_eq!(NestedCalendar::trade_len(&inner).unwrap(), 1);
    assert_eq!(NestedCalendar::trade_step(&inner).unwrap(), 0);
    assert_eq!(NestedCalendar::step_time(&inner).unwrap(), (start, end));
    assert!(!NestedCalendar::finished(&inner).unwrap());
    NestedCalendar::step(&inner).unwrap();
    assert!(NestedCalendar::finished(&inner).unwrap());
    inner.reset_window(start, end).unwrap();

    let mut trade_decision = decision(start, end);
    let executions = inner.collect_data(&mut trade_decision, 3).unwrap();
    assert_eq!(executions.len(), 1);
    assert_eq!(
        executions[0].order.read().unwrap().deal_amount().to_bits(),
        2.0_f64.to_bits()
    );
    trade_decision.orders_mut()[0].set_deal_amount(0.5);
    assert_eq!(
        executions[0].order.read().unwrap().deal_amount().to_bits(),
        2.0_f64.to_bits()
    );
    assert!(NestedCalendar::finished(&inner).unwrap());
    let snapshot = inner.order_indicator_snapshot().unwrap();
    let raw = inner.order_indicator_handle().unwrap();
    {
        let _guard = raw.write().unwrap();
        assert!(Arc::ptr_eq(&raw, &inner.order_indicator_handle().unwrap()));
    }
    assert_eq!(
        snapshot.metric("deal_amount").unwrap().values()[0].to_bits(),
        2.0_f64.to_bits()
    );
    assert_eq!(calendar.0.lock().unwrap().reset_count, 1);
}

struct SnapshotFailAccount<'a> {
    account: &'a mut Account,
}

impl AtomicExecutorAccount for SnapshotFailAccount<'_> {
    fn order_indicator_handle(
        &self,
    ) -> Result<
        domain_core::SharedOrderIndicator<domain_core::NumpyOrderIndicator>,
        AtomicExecutorAccountError,
    > {
        Err(AtomicExecutorAccountError {
            message: "handle failed".into(),
        })
    }
    fn execution_target(&mut self) -> Result<&mut dyn ExecutionTarget, AtomicExecutorAccountError> {
        Ok(self.account)
    }

    fn settle_start(&mut self, _settle_type: &str) -> Result<(), AtomicExecutorAccountError> {
        Ok(())
    }

    fn update_bar_end(&mut self, _bar: AtomicBarEnd<'_>) -> Result<(), AtomicExecutorAccountError> {
        Ok(())
    }

    fn settle_commit(&mut self) -> Result<(), AtomicExecutorAccountError> {
        Ok(())
    }

    fn order_indicator_snapshot(
        &self,
    ) -> Result<domain_core::NumpyOrderIndicator, AtomicExecutorAccountError> {
        Err(AtomicExecutorAccountError {
            message: "snapshot failed".to_owned(),
        })
    }
}

#[test]
fn atomic_inner_adapter_maps_reset_collection_and_snapshot_failures() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let services = Services;

    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    calendar.0.lock().unwrap().fail_reset = true;
    let mut failing_collector = collector(Arc::clone(&calendar), true);
    let mut account = Account::new(InfinitePosition, false);
    let mut account_adapter = AtomicAccountAdapter::new(&mut account, &services, false);
    let mut inner = AtomicNestedInnerAdapter::new(
        calendar.clone(),
        &mut failing_collector,
        &mut account_adapter,
        "None",
        IndicatorConfig::default(),
    );
    assert_eq!(
        inner.reset_window(start, end).unwrap_err().message,
        "nested calendar error: reset failed"
    );
    assert!(
        inner
            .collect_data(&mut decision(start, end), 0)
            .unwrap_err()
            .message
            .contains("deal failed")
    );

    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let mut collector = collector(Arc::clone(&calendar), false);
    let mut account = Account::new(InfinitePosition, false);
    let mut failing_account = SnapshotFailAccount {
        account: &mut account,
    };
    let inner = AtomicNestedInnerAdapter::new(
        calendar,
        &mut collector,
        &mut failing_account,
        "None",
        IndicatorConfig::default(),
    );
    assert_eq!(
        inner.order_indicator_snapshot().unwrap_err().message,
        "atomic executor account error: snapshot failed"
    );
    assert_eq!(
        inner.order_indicator_handle().unwrap_err().message,
        "atomic executor account error: handle failed"
    );
}

struct OuterCalendar {
    start: NaiveDateTime,
    end: NaiveDateTime,
    steps: Arc<Mutex<usize>>,
}

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
        Ok((self.start, self.end))
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        *self.steps.lock().unwrap() += 1;
        Ok(())
    }
}

struct OuterDecision(OrderTradeDecision);

impl NestedOuterDecision for OuterDecision {
    fn order_decision(&self) -> &dyn OrderDecision {
        &self.0
    }

    fn order_decision_mut(&mut self) -> &mut dyn OrderDecision {
        &mut self.0
    }

    fn update(
        &mut self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<NestedDecisionUpdate, NestedOuterDecisionError> {
        Ok(NestedDecisionUpdate::Unchanged)
    }

    fn is_empty(&self) -> Result<bool, NestedOuterDecisionError> {
        Ok(false)
    }

    fn range_limit(
        &self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<Option<(i64, i64)>, NestedOuterDecisionError> {
        Ok(None)
    }

    fn modify_inner_decision(
        &self,
        _decision: &mut dyn OrderDecision,
    ) -> Result<(), NestedOuterDecisionError> {
        Ok(())
    }
}

struct OneDecisionStrategy {
    start: NaiveDateTime,
    end: NaiveDateTime,
}

impl NestedStrategy for OneDecisionStrategy {
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
        previous: Option<&[SharedOrderExecution]>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        assert!(previous.is_none());
        Ok(Box::new(decision(self.start, self.end)))
    }

    fn post_execute(
        &mut self,
        executions: &[SharedOrderExecution],
    ) -> Result<(), NestedStrategyError> {
        assert_eq!(executions.len(), 1);
        Ok(())
    }

    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        Ok(())
    }
}

struct NoopBinding;

impl NestedLevelBinding for NoopBinding {
    fn bind_inner(
        &mut self,
        _inner: &dyn NestedInnerExecutor,
    ) -> Result<(), NestedLevelBindingError> {
        Ok(())
    }
}

fn python_composed_lifecycle() -> Value {
    let script = r"
import ast,json,sys
from types import GeneratorType
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read())
b=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='BaseExecutor')
bc=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='collect_data');bc.returns=None
n=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='NestedExecutor')
nc=next(x for x in n.body if isinstance(x,ast.FunctionDef) and x.name=='_collect_data');nc.returns=None
for f in (bc,nc):
 for a in f.args.args:a.annotation=None
class NestedExecutor:pass
class BasePosition:ST_NO='None'
ns={'GeneratorType':GeneratorType,'NestedExecutor':NestedExecutor,'BasePosition':BasePosition,'get_start_end_idx':lambda c,d:(0,0)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[bc,nc],type_ignores=[])),p,'exec'),ns)
e=[]
class D:
 def get_range_limit(self,default_value=None):return None
 def update(self,c):return None
 def empty(self):return False
 def mod_inner_decision(self,d):pass
class P:
 def settle_start(self,s):e.append('settle-start')
 def settle_commit(self):e.append('settle-commit')
class I:
 def get_order_indicator(self,raw=True):return 'indicator'
class A:
 def __init__(self,name):self.name=name;self.current_position=P()
 def update_bar_end(self,start,end,exchange,**kw):e.append(self.name+'-bar:'+str(kw['atomic']).lower())
 def get_trade_indicator(self):return I()
class C:
 def __init__(self):self.i=0
 def get_step_time(self):return (9,16)
 def get_trade_step(self):return self.i
 def step(self):self.i+=1;e.append('step')
class X:
 collect_data=ns['collect_data'];track_data=False;_settle_type='None';trade_account=A('inner');trade_calendar=C();trade_exchange=object();indicator_config={}
 def reset(self,start_time,end_time):self.trade_calendar.i=0;e.append('reset')
 def get_level_infra(self):return object()
 def finished(self):return self.trade_calendar.i>=1
 def _collect_data(self,trade_decision,level=0):e.append('atomic-collect:'+str(level));return ['execution'],{}
class S:
 def reset(self,level_infra,outer_trade_decision):pass
 def alter_outer_trade_decision(self,d):return d
 def generate_trade_decision(self,previous):return D()
 def post_exe_step(self,r):pass
 def post_upper_level_exe_step(self):pass
class L:
 def set_sub_level_infra(self,i):pass
class E(NestedExecutor):
 collect_data=ns['collect_data'];_collect_data=ns['_collect_data'];track_data=False;_settle_type='None';trade_account=A('outer');trade_calendar=C();trade_exchange=object();indicator_config={};inner_executor=X();inner_strategy=S();level_infra=L();_skip_empty_decision=True;_align_range_limit=True
 def _init_sub_trading(self,d):
  s,e=self.trade_calendar.get_step_time();self.inner_executor.reset(s,e);i=self.inner_executor.get_level_infra();self.level_infra.set_sub_level_infra(i);self.inner_strategy.reset(i,d)
 def _update_trade_decision(self,d):
  u=d.update(self.inner_executor.trade_calendar)
  return self.inner_strategy.alter_outer_trade_decision(u) if u is not None else d
 def post_inner_exe_step(self,r):self.inner_strategy.post_exe_step(r)
g=E().collect_data(D(),level=0)
try:
 while True:next(g)
except StopIteration as x:r=x.value
print(json.dumps({'executions':len(r),'inner_bars':e.count('inner-bar:true'),'outer_bars':e.count('outer-bar:false'),'steps':e.count('step'),'reset':e.count('reset')},separators=(',',':')))
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

#[test]
fn real_nested_lifecycle_drives_real_atomic_inner_and_outer_accounts() {
    run_composed_lifecycle(false);
    run_composed_lifecycle(true);
}

fn run_composed_lifecycle(owned: bool) {
    let outer_start = time("2024-01-02 09:30:00");
    let outer_end = time("2024-01-02 09:30:59");
    let inner_calendar = Arc::new(SharedInnerCalendar::new(outer_start, outer_end));
    let outer_steps = Arc::new(Mutex::new(0));
    let outer_calendar: Arc<dyn NestedCalendar> = Arc::new(OuterCalendar {
        start: outer_start,
        end: outer_end,
        steps: Arc::clone(&outer_steps),
    });
    let lifecycle = NestedExecutorLifecycle::new(
        NestedExecutorCore::new(true, true),
        outer_calendar,
        false,
        "None",
        IndicatorConfig::default(),
        OrderIndicatorAggregationConfig::default(),
    );
    let services = Services;
    let mut collector = collector(Arc::clone(&inner_calendar), false);
    let inner_account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let mut outer_account = Account::new(InfinitePosition, false);
    let mut strategy = OneDecisionStrategy {
        start: outer_start,
        end: outer_end,
    };
    let mut outer = OuterDecision(decision(outer_start, outer_end));
    let mut binding = NoopBinding;

    let collection = {
        let mut borrowed = if owned {
            None
        } else {
            Some(inner_account.lock().unwrap())
        };
        let mut atomic_account = borrowed
            .as_mut()
            .map(|account| AtomicAccountAdapter::new(account, &services, false));
        let mut inner: Box<dyn NestedInnerExecutor + '_> = if owned {
            Box::new(OwnedAtomicNestedInnerAdapter::new(
                inner_calendar.clone(),
                collector,
                inner_account.clone(),
                Arc::new(Services),
                "None",
                IndicatorConfig::default(),
                false,
            ))
        } else {
            Box::new(AtomicNestedInnerAdapter::new(
                inner_calendar.clone(),
                &mut collector,
                atomic_account.as_mut().unwrap(),
                "None",
                IndicatorConfig::default(),
            ))
        };
        let mut nested_account =
            NestedAccountAdapter::new(&mut outer_account, &services, &services, false);
        lifecycle
            .collect_data(NestedExecutorRun {
                level_binding: &mut binding,
                inner: &mut *inner,
                strategy: &mut strategy,
                outer: &mut outer,
                account: &mut nested_account,
                tracker: None,
                return_sink: None,
                level: 0,
            })
            .unwrap()
    };

    assert_eq!(collection.executions().lock().unwrap().len(), 1);
    assert_eq!(collection.inner_order_indicators().len(), 1);
    assert_eq!(collection.decisions().len(), 1);
    assert_eq!(*outer_steps.lock().unwrap(), 1);
    assert_eq!(inner_calendar.0.lock().unwrap().step, 1);
    let inner_account = inner_account.lock().unwrap();
    assert_composed_accounts(
        &collection,
        &inner_account,
        &outer_account,
        &inner_calendar,
        outer_start,
    );
}

fn assert_composed_accounts(
    collection: &domain_core::NestedCollection,
    inner_account: &Account,
    outer_account: &Account,
    inner_calendar: &SharedInnerCalendar,
    outer_start: NaiveDateTime,
) {
    let retained = &collection.inner_order_indicators()[0];
    assert!(Arc::ptr_eq(
        retained,
        &inner_account.order_indicator_handle().unwrap()
    ));
    let execution = Arc::clone(&collection.executions().lock().unwrap()[0]);
    assert_eq!(
        retained
            .read()
            .unwrap()
            .metric("trade_price")
            .unwrap()
            .values()[0]
            .to_bits(),
        (execution.trade_price * execution.order.read().unwrap().deal_amount()).to_bits(),
    );
    assert!(retained.try_write().is_ok());
    assert!(
        inner_account
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(outer_start)
            .is_some()
    );
    assert!(
        outer_account
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(outer_start)
            .is_some()
    );

    let python = python_composed_lifecycle();
    assert_eq!(
        python,
        json!({
            "executions": collection.executions().lock().unwrap().len(),
            "inner_bars": usize::from(
                inner_account
                    .indicator().read().unwrap()
                    .recorded_trade_indicator(outer_start)
                    .is_some()
            ),
            "outer_bars": usize::from(
                outer_account
                    .indicator().read().unwrap()
                    .recorded_trade_indicator(outer_start)
                    .is_some()
            ),
            "steps": 2,
            "reset": inner_calendar.0.lock().unwrap().reset_count,
        })
    );
}

struct RememberingDealer(Arc<Mutex<Vec<Option<f64>>>>);

#[test]
fn owned_nested_graph_releases_shared_accounts_at_each_suspension() {
    use domain_core::{
        NestedDecisionTracking, NestedExecutorResume, ResumableNestedConfig, ResumableNestedEvent,
        ResumableNestedExecutor, ResumableNestedRun, SharedNestedAccountAdapter,
    };
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let inner_calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let inner_account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let outer_account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let outer_steps = Arc::new(Mutex::new(0));
    let mut executor = ResumableNestedExecutor::new(
        Box::new(OuterCalendar {
            start,
            end,
            steps: outer_steps.clone(),
        }),
        ResumableNestedConfig {
            skip_empty_decision: true,
            align_range_limit: true,
            decision_tracking: NestedDecisionTracking {
                outer: true,
                inner: true,
            },
            settle_type: "cash".into(),
            indicator_config: IndicatorConfig::default(),
            aggregation_config: OrderIndicatorAggregationConfig::default(),
            level: 0,
        },
        ResumableNestedRun {
            level_binding: Box::new(NoopBinding),
            inner: Box::new(OwnedAtomicNestedInnerAdapter::new(
                inner_calendar.clone(),
                collector(inner_calendar.clone(), false),
                inner_account.clone(),
                Arc::new(Services),
                "None",
                IndicatorConfig::default(),
                false,
            )),
            strategy: Box::new(OneDecisionStrategy { start, end }),
            outer: Box::new(OuterDecision(decision(start, end))),
            account: Box::new(SharedNestedAccountAdapter::new(
                outer_account.clone(),
                Arc::new(Services),
                Arc::new(Services),
                false,
            )),
            return_sink: None,
        },
    );
    for _ in 0..2 {
        assert!(matches!(
            executor.resume(NestedExecutorResume::Continue).unwrap(),
            ResumableNestedEvent::TrackedDecision(_)
        ));
        for account in [&inner_account, &outer_account] {
            assert!(
                account
                    .try_lock()
                    .unwrap()
                    .indicator()
                    .read()
                    .unwrap()
                    .recorded_trade_indicator(start)
                    .is_none()
            );
        }
    }
    // The graph owns all collaborators and can move while suspended, without guard lifetimes.
    let (executor, result) = std::thread::spawn(move || {
        let result = executor.resume(NestedExecutorResume::Continue).unwrap();
        (executor, result)
    })
    .join()
    .unwrap();
    let ResumableNestedEvent::Complete(collection) = result else {
        panic!("expected completion")
    };
    assert_eq!(*outer_steps.lock().unwrap(), 1);
    assert_eq!(inner_calendar.0.lock().unwrap().step, 1);
    assert_composed_accounts(
        &collection,
        &inner_account.try_lock().unwrap(),
        &outer_account.try_lock().unwrap(),
        &inner_calendar,
        start,
    );
    drop(executor);
    assert_retained_collection_survives_reset_and_drop(&collection, inner_account);
    assert!(
        outer_account
            .lock()
            .unwrap()
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(start)
            .is_some()
    );
}

fn assert_retained_collection_survives_reset_and_drop(
    collection: &domain_core::NestedCollection,
    account: domain_core::SharedSaoeAccount,
) {
    let old = collection.inner_order_indicators()[0].clone();
    {
        let account = account.lock().unwrap();
        account.indicator().write().unwrap().reset().unwrap();
        assert!(!Arc::ptr_eq(
            &old,
            &account.order_indicator_handle().unwrap()
        ));
    }
    drop(account);
    assert!(old.try_write().is_ok());
}

impl SimulatorDealProvider for RememberingDealer {
    fn deal_order(
        &self,
        order: &mut Order,
        account: &mut dyn ExecutionTarget,
        dealt: &HashMap<String, f64>,
    ) -> Result<OrderDealResult, domain_core::ExchangeDealError> {
        self.0.lock().unwrap().push(dealt.get("A").copied());
        ApplyingDealer { fail: false }.deal_order(order, account, dealt)
    }
}

#[test]
fn owned_atomic_child_survives_moves_and_retains_account_and_intraday_state() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let account = Arc::new(Mutex::new(Account::new(
        domain_core::Position::from_initial(100.0, indexmap::IndexMap::new()),
        false,
    )));
    let prior = Arc::new(Mutex::new(Vec::new()));
    let collector = SimulatorCollector::new(
        "serial",
        calendar.clone(),
        Arc::new(RememberingDealer(prior.clone())),
        Arc::new(SilentReporter),
        false,
    );
    // The 'static trait object proves no stack-local collector or account borrow escapes.
    let mut inner: Box<dyn NestedInnerExecutor + 'static> =
        Box::new(OwnedAtomicNestedInnerAdapter::new(
            calendar.clone(),
            collector,
            account.clone(),
            Arc::new(Services),
            "cash",
            IndicatorConfig::default(),
            false,
        ));
    assert_eq!(inner.trade_len().unwrap(), 1);
    assert_eq!(inner.trade_step().unwrap(), 0);
    let raw = inner.order_indicator_handle().unwrap();
    {
        let _guard = raw.write().unwrap();
        assert!(Arc::ptr_eq(&raw, &inner.order_indicator_handle().unwrap()));
        let live = account.try_lock().unwrap();
        assert!(Arc::ptr_eq(&raw, &live.order_indicator_handle().unwrap()));
        assert!(live.indicator().try_write().is_ok());
    }
    assert_eq!(inner.step_time().unwrap(), (start, end));
    assert!(!inner.finished().unwrap());
    inner.step().unwrap();
    assert!(inner.finished().unwrap());
    inner.reset_window(start, end).unwrap();
    let mut request = decision(start, end);
    let first = inner.collect_data(&mut request, 0).unwrap();
    request.orders_mut()[0].set_deal_amount(0.5);
    let retained_amount = first[0].order.read().unwrap().deal_amount();
    assert_eq!(retained_amount.to_bits(), 2.0_f64.to_bits());
    assert_eq!(
        inner
            .order_indicator_snapshot()
            .unwrap()
            .metric("deal_amount")
            .unwrap()
            .values()[0]
            .to_bits(),
        2.0_f64.to_bits()
    );
    assert!(
        account
            .try_lock()
            .unwrap()
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(start)
            .is_some()
    );
    // Move the owned graph to another thread; reset must preserve the collector's day budget.
    let mut inner = std::thread::spawn(move || {
        inner.reset_window(start, end).unwrap();
        inner.collect_data(&mut decision(start, end), 1).unwrap();
        inner
    })
    .join()
    .unwrap();
    assert_eq!(*prior.lock().unwrap(), vec![None, Some(2.0)]);
    {
        let account = account.try_lock().unwrap();
        let position = account.execution_position().unwrap();
        assert_eq!(position.cash().unwrap().to_bits(), 60.0_f64.to_bits());
        assert_eq!(
            position.stock_amount("A").unwrap().to_bits(),
            4.0_f64.to_bits()
        );
    }
    calendar.0.lock().unwrap().fail_reset = true;
    assert_eq!(
        inner.reset_window(start, end).unwrap_err().message,
        "nested calendar error: reset failed"
    );
    assert!(account.try_lock().is_ok());
    drop(inner);
    assert!(
        account
            .lock()
            .unwrap()
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(start)
            .is_some()
    );
}

#[derive(Default)]
struct SnapshotErrorIndicator {
    values: domain_core::SharedTradeIndicator,
}
impl domain_core::AccountIndicator for SnapshotErrorIndicator {
    fn order_indicator_handle(
        &self,
    ) -> Result<
        domain_core::SharedOrderIndicator<domain_core::NumpyOrderIndicator>,
        domain_core::AccountIndicatorError,
    > {
        Err(domain_core::AccountIndicatorError {
            message: "handle failed".into(),
        })
    }
    fn reset(&mut self) -> Result<(), domain_core::AccountIndicatorError> {
        unreachable!()
    }
    fn update_atomic(
        &mut self,
        _: &[domain_core::OrderExecution<'_>],
    ) -> Result<(), domain_core::AccountIndicatorError> {
        unreachable!()
    }
    fn update_nested(
        &mut self,
        _: domain_core::NestedAccountIndicatorUpdate<'_>,
    ) -> Result<(), domain_core::AccountIndicatorError> {
        unreachable!()
    }
    fn calculate(&mut self, _: IndicatorConfig) -> Result<(), domain_core::AccountIndicatorError> {
        unreachable!()
    }
    fn record(&mut self, _: NaiveDateTime) -> Result<(), domain_core::AccountIndicatorError> {
        unreachable!()
    }
    fn trade_indicator(&self) -> &domain_core::SharedTradeIndicator {
        &self.values
    }
    fn order_indicator_snapshot(
        &self,
    ) -> Result<domain_core::NumpyOrderIndicator, domain_core::AccountIndicatorError> {
        Err(domain_core::AccountIndicatorError {
            message: "snapshot failed".into(),
        })
    }
    fn recorded_trade_indicator(
        &self,
        _: NaiveDateTime,
    ) -> Option<&domain_core::SharedTradeIndicator> {
        unreachable!()
    }
    fn trade_indicator_report(
        &self,
    ) -> Result<domain_core::TradeIndicatorReport, domain_core::AccountIndicatorError> {
        unreachable!()
    }
}

#[test]
fn owned_atomic_failure_releases_guards_and_poison_is_not_recovered() {
    let start = time("2024-01-02 09:30:00");
    let end = time("2024-01-02 09:30:59");
    let calendar = Arc::new(SharedInnerCalendar::new(start, end));
    let account = Arc::new(Mutex::new(Account::new(InfinitePosition, false)));
    let mut inner = OwnedAtomicNestedInnerAdapter::new(
        calendar.clone(),
        collector(calendar.clone(), true),
        account.clone(),
        Arc::new(Services),
        "None",
        IndicatorConfig::default(),
        false,
    );
    assert!(
        inner
            .collect_data(&mut decision(start, end), 0)
            .unwrap_err()
            .message
            .contains("deal failed")
    );
    assert_eq!(inner.trade_step().unwrap(), 0);
    account
        .try_lock()
        .unwrap()
        .replace_indicator(Box::new(SnapshotErrorIndicator::default()));
    assert_eq!(
        inner.order_indicator_snapshot().unwrap_err().message,
        "atomic executor account error: account indicator plugin error: snapshot failed"
    );
    assert_eq!(
        inner.order_indicator_handle().unwrap_err().message,
        "atomic executor account error: account indicator plugin error: handle failed"
    );
    assert!(account.try_lock().is_ok());
    let poison = account.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.lock().unwrap();
            panic!("intentional poison");
        })
        .join()
        .is_err()
    );
    assert_eq!(
        inner.order_indicator_snapshot().unwrap_err().message,
        "owned atomic account lock poisoned"
    );
    assert_eq!(
        inner.order_indicator_handle().unwrap_err().message,
        "owned atomic account lock poisoned"
    );
    assert_eq!(
        inner
            .collect_data(&mut decision(start, end), 0)
            .unwrap_err()
            .message,
        "owned atomic account lock poisoned"
    );
    assert_eq!(inner.trade_step().unwrap(), 0);
}
