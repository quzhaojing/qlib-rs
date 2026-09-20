use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::{NaiveDate, NaiveDateTime};
use domain_core::{
    Account, AccountBarMarket, AccountBarMarketError, BasePriceDataProvider,
    BasePriceProviderError, ExecutorDecisionTracker, ExecutorDecisionTrackerError, IndicatorConfig,
    InfinitePosition, InitialPositionValue, MarketDataValue, NestedAccountAdapter, NestedBarEnd,
    NestedCalendar, NestedCalendarError, NestedDecisionUpdate, NestedExecutorAccount,
    NestedExecutorAccountError, NestedExecutorCore, NestedExecutorError, NestedExecutorLifecycle,
    NestedExecutorLifecycleError, NestedExecutorReturnSink, NestedExecutorReturnSinkError,
    NestedExecutorRun, NestedInnerExecutor, NestedInnerExecutorError, NestedLevelBinding,
    NestedLevelBindingError, NestedOuterDecision, NestedOuterDecisionError, NestedStrategy,
    NestedStrategyError, NumpyOrderIndicator, Order, OrderDecision, OrderDir, OrderExecution,
    OrderIndicatorAggregationConfig, OrderTradeDecision, OwnedOrderExecution, Position,
    PositionHolding, SharedOrderExecution, TimeRange,
};
use indexmap::IndexMap;
use serde_json::{Value, json};

#[path = "support/nested_shared_list_cases.rs"]
mod nested_shared_list_cases;

fn time(hour: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, 2)
        .unwrap()
        .and_hms_opt(hour, 0, 0)
        .unwrap()
}

#[derive(Default)]
struct State {
    order: domain_core::SharedOrderIndicator<NumpyOrderIndicator>,
    events: Vec<String>,
    step: i64,
    outer_times: usize,
    fail: Option<&'static str>,
}

type SharedState = Arc<Mutex<State>>;

fn event(state: &SharedState, name: impl Into<String>) -> bool {
    let name = name.into();
    let mut state = state.lock().unwrap();
    state.events.push(name.clone());
    state.fail == Some(name.as_str())
}

struct OuterCalendar(SharedState);

impl NestedCalendar for OuterCalendar {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        unreachable!()
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        unreachable!()
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        unreachable!()
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        let failed = {
            let mut state = self.0.lock().unwrap();
            state.events.push("outer_time".to_owned());
            state.outer_times += 1;
            state.fail == Some("outer_time")
                || (state.fail == Some("bar_time") && state.outer_times == 2)
        };
        if failed {
            return Err(calendar_error("outer_time"));
        }
        Ok((time(9), time(16)))
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
        let step = u32::try_from(self.state.lock().unwrap().step).unwrap();
        Ok((time(10 + step), time(11 + step)))
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
        start_time: NaiveDateTime,
        end_time: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        if event(&self.state, format!("reset:{start_time}:{end_time}"))
            || self.state.lock().unwrap().fail == Some("reset")
        {
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
        let step = self.state.lock().unwrap().step;
        let step_value = f64::from(i32::try_from(step).unwrap());
        let execution = OwnedOrderExecution {
            order: decision.orders()[0].clone(),
            trade_value: step_value + 10.0,
            trade_cost: 0.1,
            trade_price: 2.0,
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
    replace_first: bool,
    range: Option<(i64, i64)>,
    update_count: usize,
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
        self.update_count += 1;
        Ok(if self.replace_first && self.update_count == 1 {
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

struct Strategy {
    state: SharedState,
    previous: Vec<Option<usize>>,
}

impl NestedStrategy for Strategy {
    fn reset(&mut self, _outer: &dyn NestedOuterDecision) -> Result<(), NestedStrategyError> {
        if event(&self.state, "strategy_reset") {
            return Err(strategy_error("strategy_reset"));
        }
        Ok(())
    }

    fn alter_outer_decision(
        &mut self,
        _outer: &mut dyn NestedOuterDecision,
    ) -> Result<(), NestedStrategyError> {
        if event(&self.state, "alter") {
            return Err(strategy_error("alter"));
        }
        Ok(())
    }

    fn generate_trade_decision(
        &mut self,
        previous: Option<&[SharedOrderExecution]>,
    ) -> Result<Box<dyn OrderDecision>, NestedStrategyError> {
        if event(&self.state, "generate") {
            return Err(strategy_error("generate"));
        }
        self.previous
            .push(previous.map(|items| Arc::as_ptr(&items[0]) as usize));
        let step = self.state.lock().unwrap().step;
        Ok(Box::new(OrderTradeDecision::from_orders(
            vec![Order::new(
                format!("S{step}"),
                1.0,
                OrderDir::Buy,
                None,
                None,
            )],
            time(10),
            time(15),
            None,
        )))
    }

    fn post_execute(
        &mut self,
        _executions: &[SharedOrderExecution],
    ) -> Result<(), NestedStrategyError> {
        if event(&self.state, "post") {
            return Err(strategy_error("post"));
        }
        Ok(())
    }

    fn post_upper_level(&mut self) -> Result<(), NestedStrategyError> {
        if event(&self.state, "upper") {
            return Err(strategy_error("upper"));
        }
        Ok(())
    }
}

fn calendar_error(message: &str) -> NestedCalendarError {
    NestedCalendarError {
        message: message.to_owned(),
    }
}

fn inner_error(message: &str) -> NestedInnerExecutorError {
    NestedInnerExecutorError {
        message: message.to_owned(),
    }
}

fn outer_error(message: &str) -> NestedOuterDecisionError {
    NestedOuterDecisionError {
        message: message.to_owned(),
    }
}

fn strategy_error(message: &str) -> NestedStrategyError {
    NestedStrategyError {
        message: message.to_owned(),
    }
}

fn harness(
    len: i64,
    empty: bool,
    replace_first: bool,
    range: Option<(i64, i64)>,
    fail: Option<&'static str>,
) -> (SharedState, OuterCalendar, Binding, Inner, Strategy, Outer) {
    let state = Arc::new(Mutex::new(State {
        fail,
        ..State::default()
    }));
    (
        Arc::clone(&state),
        OuterCalendar(Arc::clone(&state)),
        Binding(Arc::clone(&state)),
        Inner {
            state: Arc::clone(&state),
            len,
        },
        Strategy {
            state: Arc::clone(&state),
            previous: Vec::new(),
        },
        Outer {
            state,
            decision: OrderTradeDecision::from_orders(Vec::new(), time(9), time(16), None),
            empty,
            replace_first,
            range,
            update_count: 0,
        },
    )
}

#[test]
fn nested_loop_preserves_update_timing_identity_and_owned_outputs() {
    let (state, calendar, mut binding, mut inner, mut strategy, mut outer) =
        harness(3, false, true, None, None);
    let mut result = NestedExecutorCore::new(true, true)
        .collect_data(
            &calendar,
            &mut binding,
            &mut inner,
            &mut strategy,
            &mut outer,
            4,
        )
        .unwrap();

    assert_eq!(result.executions().lock().unwrap().len(), 3);
    assert_eq!(result.inner_order_indicators().len(), 3);
    assert_eq!(result.inner_order_indicators_mut().len(), 3);
    let raw = state.lock().unwrap().order.clone();
    assert!(
        result
            .inner_order_indicators()
            .iter()
            .all(|handle| Arc::ptr_eq(handle, &raw))
    );
    state.lock().unwrap().order = Arc::default();
    assert!(!Arc::ptr_eq(&state.lock().unwrap().order, &raw));
    assert!(
        result
            .inner_order_indicators()
            .iter()
            .all(|handle| Arc::ptr_eq(handle, &raw))
    );
    assert_eq!(result.decisions().len(), 3);
    assert_eq!(
        result.executions().lock().unwrap()[0]
            .order
            .read()
            .unwrap()
            .stock_id(),
        "S0"
    );
    assert_eq!(
        result.executions().lock().unwrap()[2].trade_value.to_bits(),
        12.0_f64.to_bits()
    );
    assert_eq!(strategy.previous[0], None);
    assert_eq!(
        strategy.previous[1],
        Some(Arc::as_ptr(&result.executions().lock().unwrap()[0]) as usize)
    );
    assert_eq!(
        strategy.previous[2],
        Some(Arc::as_ptr(&result.executions().lock().unwrap()[1]) as usize)
    );
    assert_eq!(result.decisions()[0].start_time(), time(10));
    assert_eq!(result.decisions()[2].end_time(), time(13));
    assert_eq!(
        result.decisions()[0].decision().orders()[0].stock_id(),
        "S0"
    );
    let steps = result.base_price_steps();
    assert_eq!(steps[1].start_time, time(11));
    assert!(steps[1].trade_range.is_none());
    let (indicators, aggregation_steps) = result.aggregation_inputs();
    assert_eq!(indicators.len(), 3);
    assert_eq!(aggregation_steps.len(), 3);

    let events = &state.lock().unwrap().events;
    assert_eq!(
        &events[..4],
        [
            "outer_time",
            "reset:2024-01-02 09:00:00:2024-01-02 16:00:00",
            "bind",
            "strategy_reset"
        ]
    );
    assert_eq!(events.iter().filter(|event| *event == "alter").count(), 1);
    assert_eq!(events.last().unwrap(), "upper");
}

#[test]
fn empty_and_range_alignment_preserve_break_skip_and_unaligned_paths() {
    let (state, calendar, mut binding, mut inner, mut strategy, mut outer) =
        harness(2, true, false, Some((0, 1)), None);
    let empty = NestedExecutorCore::new(true, true)
        .collect_data(
            &calendar,
            &mut binding,
            &mut inner,
            &mut strategy,
            &mut outer,
            0,
        )
        .unwrap();
    assert!(empty.executions().lock().unwrap().is_empty());
    assert!(!state.lock().unwrap().events.contains(&"range".to_owned()));
    assert_eq!(state.lock().unwrap().events.last().unwrap(), "upper");

    let (state, calendar, mut binding, mut inner, mut strategy, mut outer) =
        harness(3, false, false, Some((1, 1)), None);
    let aligned = NestedExecutorCore::new(false, true)
        .collect_data(
            &calendar,
            &mut binding,
            &mut inner,
            &mut strategy,
            &mut outer,
            0,
        )
        .unwrap();
    assert_eq!(aligned.executions().lock().unwrap().len(), 1);
    assert_eq!(
        aligned.executions().lock().unwrap()[0]
            .order
            .read()
            .unwrap()
            .stock_id(),
        "S1"
    );
    assert_eq!(
        state
            .lock()
            .unwrap()
            .events
            .iter()
            .filter(|event| *event == "step")
            .count(),
        2
    );

    let (_, calendar, mut binding, mut inner, mut strategy, mut outer) =
        harness(2, false, false, Some((9, 9)), None);
    let unaligned = NestedExecutorCore::new(false, false)
        .collect_data(
            &calendar,
            &mut binding,
            &mut inner,
            &mut strategy,
            &mut outer,
            0,
        )
        .unwrap();
    assert_eq!(unaligned.executions().lock().unwrap().len(), 2);
}

#[test]
fn every_nested_plugin_failure_stops_at_its_exact_boundary() {
    let stages = [
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
        "generate",
        "modify",
        "inner_time",
        "collect",
        "post",
        "handle",
        "upper",
    ];
    for stage in stages {
        let range = match stage {
            "trade_len" => None,
            "step" => Some((1, 1)),
            _ => Some((0, 0)),
        };
        let replace = stage == "alter";
        let (state, calendar, mut binding, mut inner, mut strategy, mut outer) =
            harness(1, false, replace, range, Some(stage));
        let result = NestedExecutorCore::new(true, true).collect_data(
            &calendar,
            &mut binding,
            &mut inner,
            &mut strategy,
            &mut outer,
            2,
        );
        let Err(error) = result else {
            panic!("{stage} unexpectedly succeeded");
        };
        let expected_domain = match stage {
            "outer_time" | "finished" | "trade_len" | "trade_step" | "step" | "inner_time" => {
                "nested calendar error"
            }
            "reset" | "collect" | "handle" => "nested inner executor error",
            "bind" => "nested level binding error",
            "update" | "empty" | "range" | "modify" => "nested outer decision error",
            "strategy_reset" | "alter" | "generate" | "post" | "upper" => "nested strategy error",
            _ => unreachable!(),
        };
        assert!(
            error.to_string().starts_with(expected_domain),
            "{stage}: {error}"
        );
        let events = &state.lock().unwrap().events;
        assert!(
            events
                .iter()
                .any(|event| event == stage || event.starts_with("reset:")),
            "{stage}: {events:?}"
        );
    }

    assert!(matches!(
        NestedExecutorError::from(calendar_error("x")),
        NestedExecutorError::Calendar(_)
    ));
}

#[test]
fn borrowed_atomic_execution_converts_to_an_independent_owned_value() {
    let mut order = Order::new("A", 3.0, OrderDir::Sell, Some(time(9)), Some(time(10)));
    order.set_deal_amount(2.0);
    let owned = OwnedOrderExecution::from_execution(OrderExecution {
        order: &order,
        trade_value: 8.0,
        trade_cost: 0.2,
        trade_price: 4.0,
    });
    order.set_deal_amount(1.0);
    assert_eq!(owned.order.deal_amount().to_bits(), 2.0_f64.to_bits());
    assert_eq!(
        (owned.trade_value, owned.trade_cost, owned.trade_price),
        (8.0, 0.2, 4.0)
    );
}

#[test]
fn synchronous_nested_loop_matches_live_python_source() {
    let script = r"
import ast,json,sys
from types import GeneratorType
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='NestedExecutor');n=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_collect_data');n.returns=None
for a in n.args.args:a.annotation=None
ns={'GeneratorType':GeneratorType,'get_start_end_idx':lambda cal,d:(0,1) if d.range is None else d.range}
exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),p,'exec'),ns)
e=[]
class D:
 def __init__(self,name='outer'):self.name=name;self.range=None
 def update(self,c):e.append('update');return None
 def empty(self):e.append('empty');return False
 def mod_inner_decision(self,d):e.append('modify')
class C:
 def __init__(self):self.i=0
 def get_trade_step(self):e.append('trade_step');return self.i
 def get_step_time(self):e.append('time:'+str(self.i));return (10+self.i,11+self.i)
 def step(self):e.append('step');self.i+=1
class A:
 class I:
  def get_order_indicator(self,raw=True):e.append('handle');return 'I'
 def get_trade_indicator(self):return self.I()
class X:
 def __init__(self):self.trade_calendar=C();self.trade_account=A()
 def reset(self,start_time,end_time):e.append('reset')
 def get_level_infra(self):return 'infra'
 def finished(self):e.append('finished');return self.trade_calendar.i>=2
 def collect_data(self,trade_decision,level):
  e.append('collect:'+str(level));self.trade_calendar.i+=1
  if False:yield None
  return ['R'+str(self.trade_calendar.i)]
class S:
 def reset(self,level_infra,outer_trade_decision):e.append('strategy_reset')
 def generate_trade_decision(self,prev):e.append('generate:'+('none' if prev is None else prev[0]));return D('inner')
 def alter_outer_trade_decision(self,d):e.append('alter');return d
 def post_exe_step(self,r):e.append('post:'+r[0])
 def post_upper_level_exe_step(self):e.append('upper')
class L:
 def set_sub_level_infra(self,i):e.append('bind')
class E:
 _collect_data=ns['_collect_data'];_skip_empty_decision=True;_align_range_limit=True
 def __init__(self):self.trade_calendar=type('O',(),{'get_step_time':lambda s:(9,16)})();self.inner_executor=X();self.inner_strategy=S();self.level_infra=L()
 def _init_sub_trading(self,d):
  start,end=self.trade_calendar.get_step_time();self.inner_executor.reset(start_time=start,end_time=end);infra=self.inner_executor.get_level_infra();self.level_infra.set_sub_level_infra(infra);self.inner_strategy.reset(level_infra=infra,outer_trade_decision=d)
 def _update_trade_decision(self,d):
  updated=d.update(self.inner_executor.trade_calendar)
  if updated is not None:d=self.inner_strategy.alter_outer_trade_decision(updated)
  return d
 def post_inner_exe_step(self,r):self.inner_strategy.post_exe_step(r)
g=E()._collect_data(D(),level=4)
try:
 while True:next(g)
except StopIteration as x:r=x.value
print(json.dumps({'events':e,'executions':r[0],'indicators':r[1]['inner_order_indicators'],'decisions':len(r[1]['decision_list'])},separators=(',',':')))
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
    let python: Value = serde_json::from_slice(&output.stdout).unwrap();

    let (_, calendar, mut binding, mut inner, mut strategy, mut outer) =
        harness(2, false, false, None, None);
    let rust = NestedExecutorCore::new(true, true)
        .collect_data(
            &calendar,
            &mut binding,
            &mut inner,
            &mut strategy,
            &mut outer,
            4,
        )
        .unwrap();
    assert_eq!(
        json!({
            "executions": rust.executions().lock().unwrap().iter().enumerate().map(|(index, _)| format!("R{}", index + 1)).collect::<Vec<_>>(),
            "indicators": vec!["I"; rust.inner_order_indicators().len()],
            "decisions": rust.decisions().len(),
        }),
        json!({
            "executions": python["executions"],
            "indicators": python["indicators"],
            "decisions": python["decisions"],
        })
    );
    assert_eq!(python["events"][0], "reset");
    assert_eq!(
        python["events"].as_array().unwrap().last().unwrap(),
        "upper"
    );
}

struct LifecycleAccount {
    state: SharedState,
    observed: Vec<(usize, usize)>,
}

impl NestedExecutorAccount for LifecycleAccount {
    fn settle_start(&mut self, settle_type: &str) -> Result<(), NestedExecutorAccountError> {
        let name = format!("settle_start:{settle_type}");
        if event(&self.state, &name) || self.state.lock().unwrap().fail == Some("settle_start") {
            return Err(account_error("settle_start"));
        }
        Ok(())
    }

    fn update_bar_end(&mut self, bar: NestedBarEnd<'_>) -> Result<(), NestedExecutorAccountError> {
        if event(&self.state, "bar") {
            return Err(account_error("bar"));
        }
        self.observed
            .push((bar.inner_order_indicators.len(), bar.steps.len()));
        assert_eq!(bar.trade_start_time, time(9));
        assert_eq!(bar.trade_end_time, time(16));
        assert!(bar.outer_decision.is_empty());
        Ok(())
    }

    fn settle_commit(&mut self) -> Result<(), NestedExecutorAccountError> {
        if event(&self.state, "commit") {
            return Err(account_error("commit"));
        }
        Ok(())
    }
}

struct Tracker(SharedState);

impl ExecutorDecisionTracker for Tracker {
    fn track(&self, decision: &mut dyn OrderDecision) -> Result<(), ExecutorDecisionTrackerError> {
        assert!(decision.is_empty());
        if event(&self.0, "track") {
            return Err(ExecutorDecisionTrackerError {
                message: "track".to_owned(),
            });
        }
        Ok(())
    }
}

struct Sink {
    state: SharedState,
    count: usize,
}

impl NestedExecutorReturnSink for Sink {
    fn store_execute_result(
        &mut self,
        executions: &domain_core::nested_executor::SharedNestedResult,
    ) -> Result<(), NestedExecutorReturnSinkError> {
        if event(&self.state, "return") {
            return Err(NestedExecutorReturnSinkError {
                message: "return".to_owned(),
            });
        }
        self.count = executions.lock().unwrap().len();
        Ok(())
    }
}

fn account_error(message: &str) -> NestedExecutorAccountError {
    NestedExecutorAccountError {
        message: message.to_owned(),
    }
}

fn lifecycle(
    calendar: Arc<dyn NestedCalendar>,
    track_data: bool,
    settlement: &str,
) -> NestedExecutorLifecycle {
    NestedExecutorLifecycle::new(
        NestedExecutorCore::new(true, true),
        calendar,
        track_data,
        settlement,
        IndicatorConfig::default(),
        OrderIndicatorAggregationConfig::default(),
    )
}

#[test]
fn nested_base_lifecycle_preserves_tracking_settlement_bar_and_return_order() {
    let (state, _, mut binding, mut inner, mut strategy, mut outer) =
        harness(1, false, false, None, None);
    let calendar: Arc<dyn NestedCalendar> = Arc::new(OuterCalendar(Arc::clone(&state)));
    let tracker = Tracker(Arc::clone(&state));
    let mut sink = Sink {
        state: Arc::clone(&state),
        count: 0,
    };
    let mut account = LifecycleAccount {
        state: Arc::clone(&state),
        observed: Vec::new(),
    };
    let result = lifecycle(calendar, true, "cash")
        .collect_data(NestedExecutorRun {
            level_binding: &mut binding,
            inner: &mut inner,
            strategy: &mut strategy,
            outer: &mut outer,
            account: &mut account,
            tracker: Some(&tracker),
            return_sink: Some(&mut sink),
            level: 5,
        })
        .unwrap();
    assert_eq!(result.executions().lock().unwrap().len(), 1);
    assert_eq!(account.observed, [(1, 1)]);
    assert_eq!(sink.count, 1);
    let events = &state.lock().unwrap().events;
    assert!(
        events.iter().position(|item| item == "track").unwrap()
            < events
                .iter()
                .position(|item| item == "settle_start:cash")
                .unwrap()
    );
    assert!(
        events.iter().position(|item| item == "bar").unwrap()
            < events.iter().position(|item| item == "outer_step").unwrap()
    );
    assert!(
        events.iter().position(|item| item == "outer_step").unwrap()
            < events.iter().position(|item| item == "commit").unwrap()
    );
    assert_eq!(events.last().unwrap(), "return");
}

#[test]
fn nested_lifecycle_preserves_optional_shortcuts_and_every_outer_failure() {
    for stage in [
        "track",
        "settle_start",
        "finished",
        "bar_time",
        "bar",
        "outer_step",
        "commit",
        "return",
    ] {
        let (state, _, mut binding, mut inner, mut strategy, mut outer) =
            harness(0, false, false, None, Some(stage));
        let calendar: Arc<dyn NestedCalendar> = Arc::new(OuterCalendar(Arc::clone(&state)));
        let tracker = Tracker(Arc::clone(&state));
        let mut sink = Sink {
            state: Arc::clone(&state),
            count: 0,
        };
        let mut account = LifecycleAccount {
            state,
            observed: Vec::new(),
        };
        let result = lifecycle(calendar, true, "cash").collect_data(NestedExecutorRun {
            level_binding: &mut binding,
            inner: &mut inner,
            strategy: &mut strategy,
            outer: &mut outer,
            account: &mut account,
            tracker: Some(&tracker),
            return_sink: Some(&mut sink),
            level: 0,
        });
        let Err(error) = result else {
            panic!("{stage} unexpectedly succeeded");
        };
        match stage {
            "track" => assert!(matches!(error, NestedExecutorLifecycleError::Tracker(_))),
            "settle_start" | "bar" | "commit" => {
                assert!(matches!(error, NestedExecutorLifecycleError::Account(_)));
            }
            "finished" => {
                assert!(matches!(error, NestedExecutorLifecycleError::Collection(_)));
            }
            "bar_time" | "outer_step" => {
                assert!(matches!(error, NestedExecutorLifecycleError::Calendar(_)));
            }
            "return" => assert!(matches!(error, NestedExecutorLifecycleError::ReturnSink(_))),
            _ => unreachable!(),
        }
    }

    let (state, _, mut binding, mut inner, mut strategy, mut outer) =
        harness(0, false, false, None, None);
    let calendar: Arc<dyn NestedCalendar> = Arc::new(OuterCalendar(Arc::clone(&state)));
    let mut account = LifecycleAccount {
        state: Arc::clone(&state),
        observed: Vec::new(),
    };
    lifecycle(calendar, false, "None")
        .collect_data(NestedExecutorRun {
            level_binding: &mut binding,
            inner: &mut inner,
            strategy: &mut strategy,
            outer: &mut outer,
            account: &mut account,
            tracker: None,
            return_sink: None,
            level: 0,
        })
        .unwrap();
    let events = &state.lock().unwrap().events;
    assert!(!events.iter().any(|item| item.starts_with("settle_")));
    assert!(!events.contains(&"track".to_owned()));
    assert!(!events.contains(&"return".to_owned()));

    let (state, _, mut binding, mut inner, mut strategy, mut outer) =
        harness(0, false, false, None, None);
    let calendar: Arc<dyn NestedCalendar> = Arc::new(OuterCalendar(Arc::clone(&state)));
    let mut account = LifecycleAccount {
        state,
        observed: Vec::new(),
    };
    lifecycle(calendar, true, "None")
        .collect_data(NestedExecutorRun {
            level_binding: &mut binding,
            inner: &mut inner,
            strategy: &mut strategy,
            outer: &mut outer,
            account: &mut account,
            tracker: None,
            return_sink: None,
            level: 0,
        })
        .unwrap();
}

#[test]
fn nested_base_lifecycle_matches_live_python_source() {
    let script = r"
import ast,json,sys
from types import GeneratorType
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='BaseExecutor');n=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='collect_data');n.returns=None
for a in n.args.args:a.annotation=None
class NestedExecutor:pass
class BasePosition:ST_NO='None'
ns={'GeneratorType':GeneratorType,'NestedExecutor':NestedExecutor,'BasePosition':BasePosition}
exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),p,'exec'),ns)
e=[]
class P:
 def settle_start(self,s):e.append('settle_start:'+s)
 def settle_commit(self):e.append('commit')
class A:
 current_position=P()
 def update_bar_end(self,start,end,exchange,**kw):e.append('bar:'+str(kw['atomic']).lower()+':'+str(len(kw['inner_order_indicators']))+':'+str(len(kw['decision_list'])))
class C:
 def get_step_time(self):return (9,16)
 def step(self):e.append('outer_step')
class R(dict):
 def update(self,v):e.append('return:'+str(len(v['execute_result'])));super().update(v)
class E(NestedExecutor):
 collect_data=ns['collect_data'];track_data=True;_settle_type='cash';trade_account=A();trade_calendar=C();trade_exchange=object();indicator_config={}
 def _collect_data(self,trade_decision,level=0):
  if False:yield None
  return ['R'],{'inner_order_indicators':['I'],'decision_list':['D']}
d=object();ret=R();g=E().collect_data(d,return_value=ret,level=5);assert next(g) is d;e.append('track')
try:next(g)
except StopIteration as x:result=x.value
print(json.dumps({'events':e,'result':result,'stored':ret['execute_result']},separators=(',',':')))
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
    let python: Value = serde_json::from_slice(&output.stdout).unwrap();

    let (state, _, mut binding, mut inner, mut strategy, mut outer) =
        harness(1, false, false, None, None);
    let calendar: Arc<dyn NestedCalendar> = Arc::new(OuterCalendar(Arc::clone(&state)));
    let tracker = Tracker(Arc::clone(&state));
    let mut sink = Sink {
        state: Arc::clone(&state),
        count: 0,
    };
    let mut account = LifecycleAccount {
        state: Arc::clone(&state),
        observed: Vec::new(),
    };
    let rust = lifecycle(calendar, true, "cash")
        .collect_data(NestedExecutorRun {
            level_binding: &mut binding,
            inner: &mut inner,
            strategy: &mut strategy,
            outer: &mut outer,
            account: &mut account,
            tracker: Some(&tracker),
            return_sink: Some(&mut sink),
            level: 5,
        })
        .unwrap();
    let rust_events = state
        .lock()
        .unwrap()
        .events
        .iter()
        .filter_map(|item| match item.as_str() {
            "track" | "settle_start:cash" | "outer_step" | "commit" => Some(item.clone()),
            "return" => Some("return:1".to_owned()),
            "bar" => Some("bar:false:1:1".to_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(python["events"], json!(rust_events));
    assert_eq!(python["result"], json!(["R"]));
    assert_eq!(python["stored"], json!(["R"]));
    assert_eq!(rust.executions().lock().unwrap().len(), 1);
    assert_eq!(sink.count, 1);
}

struct NullServices;

fn shared_empty_bar<'a>(
    outer: &'a OrderTradeDecision,
    indicators: &'a [domain_core::SharedOrderIndicator<NumpyOrderIndicator>],
) -> NestedBarEnd<'a> {
    NestedBarEnd {
        trade_start_time: time(9),
        trade_end_time: time(16),
        outer_decision: outer,
        inner_order_indicators: indicators,
        steps: &[],
        indicator_config: IndicatorConfig::default(),
        aggregation_config: OrderIndicatorAggregationConfig::default(),
    }
}

#[test]
fn shared_nested_account_preserves_settlement_and_releases_guard_on_failure() {
    let account = Arc::new(Mutex::new(Account::new(
        Position::from_initial(100.0, IndexMap::new()),
        false,
    )));
    let mut adapter = domain_core::SharedNestedAccountAdapter::new(
        account.clone(),
        Arc::new(NullServices),
        Arc::new(NullServices),
        false,
    );
    adapter.settle_start("cash").unwrap();
    assert!(account.try_lock().is_ok());
    assert!(
        adapter
            .settle_start("cash")
            .unwrap_err()
            .to_string()
            .contains("settlement cannot be nested")
    );
    let outer = OrderTradeDecision::from_orders(Vec::new(), time(9), time(16), None);
    adapter
        .update_bar_end(shared_empty_bar(&outer, &[]))
        .unwrap();
    assert!(
        account
            .try_lock()
            .unwrap()
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(time(9))
            .is_some()
    );
    adapter.settle_commit().unwrap();
    adapter.settle_start("other").unwrap();
    assert!(
        adapter
            .settle_commit()
            .unwrap_err()
            .to_string()
            .contains("unsupported settlement type")
    );
    assert!(account.try_lock().is_ok());
    drop(adapter);
    assert!(
        account
            .lock()
            .unwrap()
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(time(9))
            .is_some()
    );
}

#[test]
fn shared_nested_bar_failure_and_poison_keep_the_original_account() {
    let initial = IndexMap::from([(
        "A".into(),
        InitialPositionValue::Holding(PositionHolding::restored(1.0, Some(1.0), Some(1.0))),
    )]);
    let account = Arc::new(Mutex::new(Account::new(
        Position::from_initial(100.0, initial),
        false,
    )));
    let mut adapter = domain_core::SharedNestedAccountAdapter::new(
        account.clone(),
        Arc::new(NullServices),
        Arc::new(NullServices),
        false,
    );
    let outer = OrderTradeDecision::from_orders(Vec::new(), time(9), time(16), None);
    assert!(
        adapter
            .update_bar_end(shared_empty_bar(&outer, &[]))
            .unwrap_err()
            .to_string()
            .contains("unexpected close")
    );
    assert!(
        account
            .try_lock()
            .unwrap()
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(time(9))
            .is_none()
    );
    let poison = account.clone();
    assert!(
        std::thread::spawn(move || {
            let _guard = poison.lock().unwrap();
            panic!("intentional poison");
        })
        .join()
        .is_err()
    );
    for error in [
        adapter.settle_start("cash").unwrap_err(),
        adapter
            .update_bar_end(shared_empty_bar(&outer, &[]))
            .unwrap_err(),
        adapter.settle_commit().unwrap_err(),
    ] {
        assert_eq!(error.message, "shared nested account lock poisoned");
    }
}

impl AccountBarMarket for NullServices {
    fn is_suspended(&self, _stock: &str, _range: TimeRange) -> Result<bool, AccountBarMarketError> {
        Ok(false)
    }

    fn close(&self, _stock: &str, _range: TimeRange) -> Result<f64, AccountBarMarketError> {
        Err(AccountBarMarketError {
            message: "unexpected close".to_owned(),
        })
    }
}

impl BasePriceDataProvider for NullServices {
    fn deal_price(
        &self,
        _stock: &str,
        _range: TimeRange,
        _direction: OrderDir,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        Ok(None)
    }

    fn volume(
        &self,
        _stock: &str,
        _range: TimeRange,
    ) -> Result<Option<MarketDataValue>, BasePriceProviderError> {
        Ok(None)
    }
}

#[test]
fn real_nested_account_adapter_completes_empty_nested_bar_and_maps_settlement_errors() {
    let services = NullServices;
    let mut account = Account::new(InfinitePosition, false);
    let outer = OrderTradeDecision::from_orders(Vec::new(), time(9), time(16), None);
    let mut indicators = Vec::new();
    let steps = Vec::new();
    {
        let mut adapter = NestedAccountAdapter::new(&mut account, &services, &services, false);
        adapter.settle_start("cash").unwrap();
        adapter
            .update_bar_end(NestedBarEnd {
                trade_start_time: time(9),
                trade_end_time: time(16),
                outer_decision: &outer,
                inner_order_indicators: &mut indicators,
                steps: &steps,
                indicator_config: IndicatorConfig::default(),
                aggregation_config: OrderIndicatorAggregationConfig::default(),
            })
            .unwrap();
        adapter.settle_commit().unwrap();
    }
    assert!(
        account
            .indicator()
            .read()
            .unwrap()
            .recorded_trade_indicator(time(9))
            .is_some()
    );

    let initial: IndexMap<String, InitialPositionValue> = IndexMap::new();
    let mut finite_account = Account::new(Position::from_initial(100.0, initial), false);
    let mut adapter = NestedAccountAdapter::new(&mut finite_account, &services, &services, false);
    adapter.settle_start("cash").unwrap();
    let error = adapter.settle_start("cash").unwrap_err();
    assert!(error.to_string().contains("settlement cannot be nested"));
    adapter.settle_commit().unwrap();
    adapter.settle_start("other").unwrap();
    let error = adapter.settle_commit().unwrap_err();
    assert!(error.to_string().contains("unsupported settlement type"));

    let mut initial = IndexMap::new();
    initial.insert(
        "A".to_owned(),
        InitialPositionValue::Holding(PositionHolding::restored(1.0, Some(1.0), Some(1.0))),
    );
    let mut marked_account = Account::new(Position::from_initial(100.0, initial), false);
    let mut adapter = NestedAccountAdapter::new(&mut marked_account, &services, &services, false);
    let error = adapter
        .update_bar_end(NestedBarEnd {
            trade_start_time: time(9),
            trade_end_time: time(16),
            outer_decision: &outer,
            inner_order_indicators: &mut indicators,
            steps: &steps,
            indicator_config: IndicatorConfig::default(),
            aggregation_config: OrderIndicatorAggregationConfig::default(),
        })
        .unwrap_err();
    assert!(error.to_string().contains("unexpected close"));
}
