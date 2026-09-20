use std::sync::{Arc, Mutex, RwLock};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::{NaiveDateTime, TimeDelta};
use domain_core::decision_construction::{
    ConstructedDecisionBase, DecisionOrderItem, DecisionTotalStep, SharedDecisionOrders,
    SharedOrderDecisionConstruction,
};
use domain_core::decision_update::{
    DecisionUpdateStrategyError, LiveDecisionHandle, SharedDecisionUpdateStrategy,
    SharedLiveDecision,
};
use domain_core::live_recursive_inner::LiveNestedChildAssemblyFactory;
use domain_core::saoe_live_registry::{LiveSaoeAdapterFactory, LiveSaoeStateAdapter};
use domain_core::{
    ConfiguredLiveSaoeChildAssemblyFactory, ConfiguredNestedChildSession,
    ConfiguredSaoeChildAssemblyFactory, DummyStateInterpreter, IndicatorConfig,
    LiveSaoeChildComponents, LiveSaoeChildComponentsFactory, NestedBarEnd, NestedCalendar,
    NestedCalendarError, NestedChildAssemblyFactory, NestedChildSession, NestedDecisionTracking,
    NestedDecisionUpdate, NestedExecutorAccount, NestedExecutorAccountError,
    NestedExecutorReturnSink, NestedExecutorReturnSinkError, NestedInnerExecutor,
    NestedInnerExecutorError, NestedInnerProgress, NestedLevelBinding, NestedLevelBindingError,
    NestedOuterDecision, NestedOuterDecisionError, NumpyOrderIndicator, Order, OrderDecision,
    OrderDir, OrderIndicatorAggregationConfig, OrderTradeDecision, OwnedOrderExecution,
    ResumableNestedConfig, SaoeActionInterpreter, SaoeActionSpace, SaoeBacktestData, SaoeCalendar,
    SaoeChildComponents, SaoeChildComponentsError, SaoeChildComponentsFactory,
    SaoeDecisionCalendar, SaoeIntStateProvider, SaoeInterpreterError, SaoeObservation,
    SaoeOrderFactory, SaoePluginError, SaoePolicy, SaoePolicyAction, SaoePolicyPipeline, SaoeState,
    SaoeStateParts, SharedOrderExecution, SharedTradeRange, TradeCalendarRange,
    TradeCalendarRangeError, TradeRangeByTime,
};
use ndarray::arr1;

fn time(minute: i64) -> NaiveDateTime {
    NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap()
        + TimeDelta::minutes(minute)
}

fn order(amount: f64) -> Order {
    Order::new("A", amount, OrderDir::Buy, Some(time(0)), Some(time(9)))
}

fn decision(amount: f64) -> Box<dyn OrderDecision> {
    Box::new(OrderTradeDecision::from_orders(
        vec![order(amount)],
        time(0),
        time(9),
        None,
    ))
}

fn state_for(order: &Order) -> SaoeState {
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    SaoeState::new(SaoeStateParts {
        order: order.clone(),
        cur_time: time(0),
        cur_step: 0,
        position: order.amount(),
        history_exec: empty.clone(),
        history_steps: empty.clone(),
        metrics: None,
        backtest_data: SaoeBacktestData {
            ticks_index: Vec::new(),
            ticks_for_order: Vec::new(),
            deal_prices: arr1(&[]),
            market_volumes: arr1(&[]),
            features: empty,
        },
        ticks_per_step: 1,
        ticks_index: Vec::new(),
        ticks_for_order: Vec::new(),
    })
}

#[derive(Default)]
struct GraphState {
    order: domain_core::SharedOrderIndicator<NumpyOrderIndicator>,
    id: u32,
    finished: bool,
    events: Vec<String>,
}

fn event(state: &Arc<Mutex<GraphState>>, value: impl Into<String>) {
    state.lock().unwrap().events.push(value.into());
}

struct OuterCalendar(Arc<Mutex<GraphState>>);

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
        event(&self.0, "outer-time");
        Ok((time(0), time(9)))
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        event(&self.0, "outer-step");
        Ok(())
    }
}

struct StrategyCalendar(Arc<Mutex<GraphState>>);

impl SaoeCalendar for StrategyCalendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        event(&self.0, "strategy-range");
        Ok((1, 2))
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        event(&self.0, "strategy-time");
        Ok((time(0), time(1)))
    }
}

impl SaoeDecisionCalendar for StrategyCalendar {
    fn frequency(&self) -> Result<String, SaoePluginError> {
        event(&self.0, "strategy-frequency");
        Ok("1min".to_owned())
    }
}

impl TradeCalendarRange for StrategyCalendar {
    fn start_time(&self) -> Result<NaiveDateTime, TradeCalendarRangeError> {
        Ok(time(0))
    }

    fn get_range_idx(
        &self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(i64, i64), TradeCalendarRangeError> {
        Ok((0, 0))
    }
}

struct Inner(Arc<Mutex<GraphState>>);

impl NestedCalendar for Inner {
    fn finished(&self) -> Result<bool, NestedCalendarError> {
        Ok(self.0.lock().unwrap().finished)
    }

    fn trade_len(&self) -> Result<i64, NestedCalendarError> {
        Ok(1)
    }

    fn trade_step(&self) -> Result<i64, NestedCalendarError> {
        Ok(0)
    }

    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), NestedCalendarError> {
        event(&self.0, "inner-time");
        Ok((time(0), time(1)))
    }

    fn step(&self) -> Result<(), NestedCalendarError> {
        self.0.lock().unwrap().finished = true;
        Ok(())
    }
}

impl NestedInnerExecutor for Inner {
    fn reset_window(
        &mut self,
        _start_time: NaiveDateTime,
        _end_time: NaiveDateTime,
    ) -> Result<(), NestedInnerExecutorError> {
        let mut state = self.0.lock().unwrap();
        state.finished = false;
        state.events.push("inner-reset".to_owned());
        Ok(())
    }

    fn collect_data(
        &mut self,
        decision: &mut dyn OrderDecision,
        level: usize,
    ) -> Result<Vec<SharedOrderExecution>, NestedInnerExecutorError> {
        event(&self.0, format!("inner-collect:{level}"));
        self.0.lock().unwrap().finished = true;
        Ok(vec![
            OwnedOrderExecution {
                order: decision.orders()[0].clone(),
                trade_value: 0.0,
                trade_cost: 0.0,
                trade_price: 0.0,
            }
            .into_shared(),
        ])
    }

    fn order_indicator_handle(
        &self,
    ) -> Result<domain_core::SharedOrderIndicator<NumpyOrderIndicator>, NestedInnerExecutorError>
    {
        Ok(self.0.lock().unwrap().order.clone())
    }

    fn order_indicator_snapshot(&self) -> Result<NumpyOrderIndicator, NestedInnerExecutorError> {
        event(&self.0, "inner-snapshot");
        Ok(NumpyOrderIndicator::new())
    }
}

struct Binding(Arc<Mutex<GraphState>>);

impl NestedLevelBinding for Binding {
    fn bind_inner(
        &mut self,
        _inner: &dyn NestedInnerExecutor,
    ) -> Result<(), NestedLevelBindingError> {
        event(&self.0, "bind");
        Ok(())
    }
}

struct Outer {
    state: Arc<Mutex<GraphState>>,
    decision: OrderTradeDecision,
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
        event(&self.state, "outer-update");
        Ok(NestedDecisionUpdate::Unchanged)
    }

    fn is_empty(&self) -> Result<bool, NestedOuterDecisionError> {
        Ok(false)
    }

    fn range_limit(
        &self,
        _calendar: &dyn NestedCalendar,
    ) -> Result<Option<(i64, i64)>, NestedOuterDecisionError> {
        Ok(Some((0, 0)))
    }

    fn modify_inner_decision(
        &self,
        _decision: &mut dyn OrderDecision,
    ) -> Result<(), NestedOuterDecisionError> {
        event(&self.state, "outer-modify");
        Ok(())
    }
}

struct Account(Arc<Mutex<GraphState>>);

impl NestedExecutorAccount for Account {
    fn settle_start(&mut self, _settle_type: &str) -> Result<(), NestedExecutorAccountError> {
        event(&self.0, "settle-start");
        Ok(())
    }

    fn update_bar_end(&mut self, _bar: NestedBarEnd<'_>) -> Result<(), NestedExecutorAccountError> {
        event(&self.0, "account-bar");
        Ok(())
    }

    fn settle_commit(&mut self) -> Result<(), NestedExecutorAccountError> {
        event(&self.0, "settle-commit");
        Ok(())
    }
}

struct Sink(Arc<Mutex<GraphState>>);

impl NestedExecutorReturnSink for Sink {
    fn store_execute_result(
        &mut self,
        executions: &domain_core::nested_executor::SharedNestedResult,
    ) -> Result<(), NestedExecutorReturnSinkError> {
        event(
            &self.0,
            format!("sink:{}", executions.lock().unwrap().len()),
        );
        Ok(())
    }
}

struct States(Arc<Mutex<GraphState>>);

impl SaoeIntStateProvider for States {
    fn reset(&mut self, outer: &dyn NestedOuterDecision) -> Result<(), SaoePluginError> {
        event(
            &self.0,
            format!("states-reset:{}", outer.order_decision().orders().len()),
        );
        Ok(())
    }

    fn state(&self, order: &Order) -> Result<SaoeState, SaoePluginError> {
        event(&self.0, "states-state");
        Ok(state_for(order))
    }

    fn update(
        &mut self,
        executions: &[SharedOrderExecution],
        step_range: (i64, i64),
    ) -> Result<(), SaoePluginError> {
        event(
            &self.0,
            format!(
                "states-update:{}:{}-{}",
                executions.len(),
                step_range.0,
                step_range.1
            ),
        );
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), SaoePluginError> {
        event(&self.0, "states-finalize");
        Ok(())
    }
}

struct Policy(Arc<Mutex<GraphState>>);

impl SaoePolicy for Policy {
    fn actions(
        &mut self,
        observations: &[SaoeObservation],
    ) -> Result<Vec<SaoePolicyAction>, SaoeInterpreterError> {
        let id = self.0.lock().unwrap().id;
        event(&self.0, format!("policy:{}", observations.len()));
        Ok(vec![
            SaoePolicyAction::Continuous(f64::from(id));
            observations.len()
        ])
    }
}

struct Action(Arc<Mutex<GraphState>>);

impl SaoeActionInterpreter for Action {
    fn action_space(&self) -> SaoeActionSpace {
        SaoeActionSpace::NonNegativeContinuous
    }

    fn interpret(
        &self,
        _state: &SaoeState,
        action: SaoePolicyAction,
    ) -> Result<f64, SaoeInterpreterError> {
        event(&self.0, "action");
        let SaoePolicyAction::Continuous(value) = action else {
            return Err(SaoeInterpreterError::ExpectedContinuous);
        };
        Ok(value)
    }
}

struct Orders(Arc<Mutex<GraphState>>);

impl SaoeOrderFactory for Orders {
    fn create(
        &mut self,
        stock_id: &str,
        amount: Option<f64>,
        direction: OrderDir,
    ) -> Result<Order, SaoePluginError> {
        event(&self.0, format!("order:{}", amount.unwrap()));
        Ok(Order::new(stock_id, amount.unwrap(), direction, None, None))
    }
}

struct ComponentsFactory {
    calls: Arc<Mutex<Vec<(f64, usize, u32)>>>,
    graphs: Arc<Mutex<Vec<Arc<Mutex<GraphState>>>>>,
    next_id: u32,
    fail: bool,
}

impl SaoeChildComponentsFactory for ComponentsFactory {
    fn create(
        &mut self,
        decision: &dyn OrderDecision,
        level: usize,
    ) -> Result<SaoeChildComponents, SaoeChildComponentsError> {
        if self.fail {
            return Err(SaoeChildComponentsError {
                message: "components".to_owned(),
            });
        }
        self.next_id += 1;
        let id = self.next_id;
        self.calls
            .lock()
            .unwrap()
            .push((decision.orders()[0].amount(), level, id));
        let state = Arc::new(Mutex::new(GraphState {
            id,
            ..GraphState::default()
        }));
        self.graphs.lock().unwrap().push(Arc::clone(&state));
        let strategy_calendar: Arc<dyn SaoeDecisionCalendar> =
            Arc::new(StrategyCalendar(Arc::clone(&state)));
        Ok(SaoeChildComponents {
            calendar: Box::new(OuterCalendar(Arc::clone(&state))),
            strategy_calendar,
            level_binding: Box::new(Binding(Arc::clone(&state))),
            inner: Box::new(Inner(Arc::clone(&state))),
            states: Box::new(States(Arc::clone(&state))),
            pipeline: SaoePolicyPipeline::new(
                Box::new(DummyStateInterpreter),
                Box::new(Policy(Arc::clone(&state))),
                Box::new(Action(Arc::clone(&state))),
            ),
            orders: Box::new(Orders(Arc::clone(&state))),
            outer: Box::new(Outer {
                state: Arc::clone(&state),
                decision: OrderTradeDecision::from_orders(
                    decision.orders().to_vec(),
                    decision.start_time(),
                    decision.end_time(),
                    None,
                ),
            }),
            account: Box::new(Account(Arc::clone(&state))),
            return_sink: (id == 1)
                .then(|| Box::new(Sink(Arc::clone(&state))) as Box<dyn NestedExecutorReturnSink>),
        })
    }
}

struct LiveOrigin;

impl SharedDecisionUpdateStrategy<()> for LiveOrigin {
    fn update_trade_decision(
        &self,
        _decision: &SharedLiveDecision<Self, ()>,
        _calendar: &dyn domain_core::DecisionUpdateCalendar,
    ) -> Result<Option<SharedLiveDecision<Self, ()>>, DecisionUpdateStrategyError> {
        Ok(None)
    }
}

fn live_decision(amount: f64) -> LiveDecisionHandle {
    let orders: SharedDecisionOrders = Arc::new(RwLock::new(vec![DecisionOrderItem::Order(
        Arc::new(RwLock::new(order(amount))),
    )]));
    let range: SharedTradeRange = Arc::new(TradeRangeByTime::parse("09:30", "09:39").unwrap());
    Arc::new(RwLock::new(SharedOrderDecisionConstruction {
        strategy: Arc::new(LiveOrigin),
        base: Some(ConstructedDecisionBase {
            start_time: time(0),
            end_time: time(9),
            trade_range: Some(range),
        }),
        total_step: DecisionTotalStep::Value(10),
        orders: Some(orders),
        details: Some(()),
    }))
}

struct UnusedLiveAdapters;

impl LiveSaoeAdapterFactory for UnusedLiveAdapters {
    fn create(
        &mut self,
        _order: &Arc<RwLock<Order>>,
        _outer: &LiveDecisionHandle,
        _range: &SharedTradeRange,
    ) -> Result<Box<dyn LiveSaoeStateAdapter>, SaoePluginError> {
        Err(SaoePluginError {
            message: "unused live adapter".to_owned(),
        })
    }
}

struct LiveComponentsFactory {
    calls: Arc<Mutex<Vec<(usize, usize)>>>,
    state: Arc<Mutex<GraphState>>,
    fail: bool,
}

impl LiveSaoeChildComponentsFactory for LiveComponentsFactory {
    fn create(
        &mut self,
        decision: &LiveDecisionHandle,
        level: usize,
    ) -> Result<LiveSaoeChildComponents, SaoeChildComponentsError> {
        if self.fail {
            return Err(SaoeChildComponentsError {
                message: "live-components".to_owned(),
            });
        }
        self.calls
            .lock()
            .unwrap()
            .push((Arc::strong_count(decision), level));
        let strategy_calendar = Arc::new(StrategyCalendar(Arc::clone(&self.state)));
        Ok(LiveSaoeChildComponents {
            calendar: Box::new(OuterCalendar(Arc::clone(&self.state))),
            strategy_calendar: Arc::clone(&strategy_calendar) as Arc<dyn SaoeDecisionCalendar>,
            range_calendar: strategy_calendar as Arc<dyn TradeCalendarRange>,
            level_binding: Box::new(Binding(Arc::clone(&self.state))),
            inner: Box::new(Inner(Arc::clone(&self.state))),
            adapters: Box::new(UnusedLiveAdapters),
            pipeline: SaoePolicyPipeline::new(
                Box::new(DummyStateInterpreter),
                Box::new(Policy(Arc::clone(&self.state))),
                Box::new(Action(Arc::clone(&self.state))),
            ),
            orders: Box::new(Orders(Arc::clone(&self.state))),
            account: Box::new(Account(Arc::clone(&self.state))),
            return_sink: None,
        })
    }
}

fn config() -> ResumableNestedConfig {
    ResumableNestedConfig {
        skip_empty_decision: false,
        align_range_limit: false,
        decision_tracking: NestedDecisionTracking::default(),
        settle_type: "None".to_owned(),
        indicator_config: IndicatorConfig::default(),
        aggregation_config: OrderIndicatorAggregationConfig::default(),
        level: 999,
    }
}

#[test]
fn configured_factory_builds_and_runs_a_fresh_immediate_saoe_graph_per_child() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let graphs = Arc::new(Mutex::new(Vec::new()));
    let factory = ConfiguredSaoeChildAssemblyFactory::new(
        config(),
        Box::new(ComponentsFactory {
            calls: Arc::clone(&calls),
            graphs: Arc::clone(&graphs),
            next_id: 0,
            fail: false,
        }),
    );
    let mut session = ConfiguredNestedChildSession::new(Box::new(factory));

    let NestedInnerProgress::Complete {
        decision: first_outer,
        executions: first,
    } = session.begin(decision(10.0), 3).unwrap()
    else {
        panic!("immediate SAOE child should complete synchronously")
    };
    assert_eq!(
        first_outer.orders()[0].amount().to_bits(),
        10.0_f64.to_bits()
    );
    assert_eq!(
        first.lock().unwrap()[0]
            .order
            .read()
            .unwrap()
            .amount()
            .to_bits(),
        1.0_f64.to_bits()
    );

    let NestedInnerProgress::Complete {
        decision: second_outer,
        executions: second,
    } = session.begin(decision(20.0), 7).unwrap()
    else {
        panic!("second immediate SAOE child should also complete synchronously")
    };
    assert_eq!(
        second_outer.orders()[0].amount().to_bits(),
        20.0_f64.to_bits()
    );
    assert_eq!(
        second.lock().unwrap()[0]
            .order
            .read()
            .unwrap()
            .amount()
            .to_bits(),
        2.0_f64.to_bits()
    );
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        [(10.0, 3, 1), (20.0, 7, 2)]
    );
    let graphs = graphs.lock().unwrap();
    assert_eq!(graphs.len(), 2);
    assert!(
        graphs[0]
            .lock()
            .unwrap()
            .events
            .contains(&"inner-collect:4".to_owned())
    );
    assert!(
        graphs[0]
            .lock()
            .unwrap()
            .events
            .contains(&"sink:1".to_owned())
    );
    assert!(
        graphs[1]
            .lock()
            .unwrap()
            .events
            .contains(&"inner-collect:8".to_owned())
    );
    assert!(
        !graphs[1]
            .lock()
            .unwrap()
            .events
            .contains(&"sink:1".to_owned())
    );
}

#[test]
fn configured_factory_maps_component_failure_and_overrides_template_level() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let graphs = Arc::new(Mutex::new(Vec::new()));
    let mut factory = ConfiguredSaoeChildAssemblyFactory::new(
        config(),
        Box::new(ComponentsFactory {
            calls: Arc::clone(&calls),
            graphs: Arc::clone(&graphs),
            next_id: 0,
            fail: false,
        }),
    );
    let assembly = factory.assemble(&*decision(3.0), 4).unwrap();
    assert_eq!(assembly.config.level, 4);
    assert_eq!(
        assembly.run.outer.order_decision().orders()[0]
            .amount()
            .to_bits(),
        3.0_f64.to_bits()
    );
    assert!(assembly.run.return_sink.is_some());

    let mut failing = ConfiguredSaoeChildAssemblyFactory::new(
        config(),
        Box::new(ComponentsFactory {
            calls,
            graphs,
            next_id: 0,
            fail: true,
        }),
    );
    assert_eq!(
        failing.assemble(&*decision(1.0), 0).err().unwrap().message,
        "SAOE child component assembly failed: components"
    );
    assert_eq!(
        SaoeChildComponentsError {
            message: "typed".to_owned()
        }
        .to_string(),
        "SAOE child component assembly failed: typed"
    );
}

#[test]
fn configured_live_factory_preserves_incoming_decision_and_builds_shared_strategy_graph() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let state = Arc::new(Mutex::new(GraphState::default()));
    let decision = live_decision(8.0);
    let initial_count = Arc::strong_count(&decision);
    let mut factory = ConfiguredLiveSaoeChildAssemblyFactory::new(
        config(),
        Box::new(LiveComponentsFactory {
            calls: Arc::clone(&calls),
            state: Arc::clone(&state),
            fail: false,
        }),
    );
    let assembly = factory.assemble(&decision, 6).unwrap();
    assert_eq!(assembly.config.level, 6);
    assert!(Arc::ptr_eq(&assembly.run.outer, &decision));
    assert_eq!(calls.lock().unwrap().as_slice(), [(initial_count, 6)]);

    let mut failing = ConfiguredLiveSaoeChildAssemblyFactory::new(
        config(),
        Box::new(LiveComponentsFactory {
            calls,
            state,
            fail: true,
        }),
    );
    assert_eq!(
        failing.assemble(&decision, 0).err().unwrap().message,
        "SAOE child component assembly failed: live-components"
    );
}
