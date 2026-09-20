use std::{
    process::Command,
    sync::{Arc, Mutex},
};

use chrono::{NaiveDateTime, TimeDelta};
use domain_core::{
    IdxTradeRange, Order, OrderDir, SaoeCalendar, SaoeOrderFactory, SaoePluginError,
    SharedOrderExecution, SharedTradeRange, SingleOrderStrategy,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, PartialEq)]
struct ModuleSnapshot {
    source_sha256: String,
    module_doc: Option<String>,
    module_body: Vec<String>,
    imports: Vec<String>,
    class_bases: Vec<String>,
    class_doc: String,
    constructor_signature: String,
    constructor_body: Vec<String>,
    generator_signature: String,
    generator_body: Vec<String>,
    base_events: Vec<String>,
    facts: Vec<String>,
}

fn time(value: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").unwrap()
}

type Events = Arc<Mutex<Vec<Value>>>;

const WHOLE_SOURCE_SCRIPT: &str = r#"
import ast
import hashlib
import inspect
import json
import sys
import types
from pathlib import Path

source_path = Path(sys.argv[1])
raw = source_path.read_bytes()
tree = ast.parse(raw)

imports = []
for node in tree.body:
    if isinstance(node, ast.ImportFrom):
        prefix = "." * node.level + (node.module or "")
        imports.append(prefix + ":" + ",".join(alias.name for alias in node.names))

strategy_node = next(
    node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SingleOrderStrategy"
)
constructor_node = next(
    node for node in strategy_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
)
generator_node = next(
    node
    for node in strategy_node.body
    if isinstance(node, ast.FunctionDef) and node.name == "generate_trade_decision"
)

events = []
class BaseStrategy:
    def __init__(self):
        assert not hasattr(self, "_order")
        assert not hasattr(self, "_trade_range")
        events.append("base_init")

class TradeDecisionWO:
    def __init__(self, order_list, strategy, trade_range):
        self.order_list = order_list
        self.strategy = strategy
        self.trade_range = trade_range

class Order:
    pass

class OrderHelper:
    pass

class TradeRange:
    pass

qlib = types.ModuleType("qlib")
backtest = types.ModuleType("qlib.backtest")
backtest.Order = Order
decision = types.ModuleType("qlib.backtest.decision")
decision.OrderHelper = OrderHelper
decision.TradeDecisionWO = TradeDecisionWO
decision.TradeRange = TradeRange
strategy_package = types.ModuleType("qlib.strategy")
base = types.ModuleType("qlib.strategy.base")
base.BaseStrategy = BaseStrategy
sys.modules.update({
    "qlib": qlib,
    "qlib.backtest": backtest,
    "qlib.backtest.decision": decision,
    "qlib.strategy": strategy_package,
    "qlib.strategy.base": base,
})

module = types.ModuleType("qlib.rl.strategy.single_order")
module.__package__ = "qlib.rl.strategy"
exec(compile(raw, str(source_path), "exec"), module.__dict__)
strategy_type = module.SingleOrderStrategy

order = types.SimpleNamespace(stock_id="A", amount=7.0, direction="sell")
trade_range = object()
strategy = strategy_type(order, trade_range)
retained_order_identity = strategy._order is order
retained_range_identity = strategy._trade_range is trade_range
order.stock_id = "B"
order.amount = 8.0
order.direction = "buy"
create_calls = []
created_order = object()
class Helper:
    def create(self, **kwargs):
        create_calls.append(kwargs)
        return created_order
helper = Helper()
strategy.common_infra = {
    "trade_exchange": types.SimpleNamespace(get_order_helper=lambda: helper)
}
previous = object()
decision_value = strategy.generate_trade_decision(previous)

snapshot = {
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": ast.get_docstring(tree, clean=False),
    "module_body": [type(node).__name__ for node in tree.body],
    "imports": imports,
    "class_bases": [ast.unparse(node) for node in strategy_node.bases],
    "class_doc": ast.get_docstring(strategy_node, clean=False),
    "constructor_signature": str(inspect.signature(strategy_type.__init__)),
    "constructor_body": [ast.unparse(node) for node in constructor_node.body],
    "generator_signature": str(inspect.signature(strategy_type.generate_trade_decision)),
    "generator_body": [ast.unparse(node) for node in generator_node.body],
    "base_events": events,
    "facts": [name for name, value in {
        "retained_order_identity": retained_order_identity,
        "retained_range_identity": retained_range_identity,
        "generated_from_mutated_order": create_calls == [{
            "code": "B", "amount": 8.0, "direction": "buy"
        }],
        "previous_result_ignored": previous not in create_calls and len(create_calls) == 1,
        "decision_identity": (
            decision_value.order_list == [created_order]
            and decision_value.strategy is strategy
            and decision_value.trade_range is trade_range
        ),
    }.items() if value],
}
print(json.dumps(snapshot))
"#;

#[test]
fn whole_source_surface_initialization_and_retained_inputs_are_frozen() {
    let source = r"D:\code\github\qlib\qlib\rl\strategy\single_order.py";
    let output = Command::new("python")
        .args(["-c", WHOLE_SOURCE_SCRIPT, source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: ModuleSnapshot = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        actual,
        ModuleSnapshot {
            source_sha256:
                "904013072a947b7297005ccd24b3f699d8f1c16b95e472598609a062aae56675"
                    .to_owned(),
            module_doc: None,
            module_body: vec![
                "ImportFrom".to_owned(),
                "ImportFrom".to_owned(),
                "ImportFrom".to_owned(),
                "ImportFrom".to_owned(),
                "ClassDef".to_owned(),
            ],
            imports: vec![
                "__future__:annotations".to_owned(),
                "qlib.backtest:Order".to_owned(),
                "qlib.backtest.decision:OrderHelper,TradeDecisionWO,TradeRange".to_owned(),
                "qlib.strategy.base:BaseStrategy".to_owned(),
            ],
            class_bases: vec!["BaseStrategy".to_owned()],
            class_doc: "Strategy used to generate a trade decision with exactly one order."
                .to_owned(),
            constructor_signature:
                "(self, order: 'Order', trade_range: 'TradeRange | None' = None) -> 'None'"
                    .to_owned(),
            constructor_body: vec![
                "super().__init__()".to_owned(),
                "self._order = order".to_owned(),
                "self._trade_range = trade_range".to_owned(),
            ],
            generator_signature:
                "(self, execute_result: 'list | None' = None) -> 'TradeDecisionWO'".to_owned(),
            generator_body: vec![
                "oh: OrderHelper = self.common_infra.get('trade_exchange').get_order_helper()"
                    .to_owned(),
                "order_list = [oh.create(code=self._order.stock_id, amount=self._order.amount, direction=self._order.direction)]"
                    .to_owned(),
                "return TradeDecisionWO(order_list, self, self._trade_range)".to_owned(),
            ],
            base_events: vec!["base_init".to_owned()],
            facts: [
                "retained_order_identity",
                "retained_range_identity",
                "generated_from_mutated_order",
                "previous_result_ignored",
                "decision_identity",
            ]
            .map(str::to_owned)
            .to_vec(),
        }
    );
}

struct Orders {
    events: Events,
    fail: bool,
    custom: bool,
}

impl SaoeOrderFactory for Orders {
    fn create(
        &mut self,
        stock_id: &str,
        amount: Option<f64>,
        direction: OrderDir,
    ) -> Result<Order, SaoePluginError> {
        self.events
            .lock()
            .unwrap()
            .push(json!(["create", stock_id, amount, direction.value()]));
        if self.fail {
            return Err(SaoePluginError {
                message: "create".to_owned(),
            });
        }
        Ok(Order::new(
            stock_id,
            amount.unwrap(),
            direction,
            self.custom.then(|| time("2022-01-01 00:00:00")),
            None,
        ))
    }
}

struct Calendar {
    events: Events,
    count: Mutex<i64>,
    fail: Option<i64>,
}

impl SaoeCalendar for Calendar {
    fn available_step_range(&self) -> Result<(i64, i64), SaoePluginError> {
        panic!("single-order generation must not query the available range")
    }
    fn step_time(&self) -> Result<(NaiveDateTime, NaiveDateTime), SaoePluginError> {
        let mut count = self.count.lock().unwrap();
        *count += 1;
        self.events
            .lock()
            .unwrap()
            .push(json!(["calendar", *count]));
        if self.fail == Some(*count) {
            return Err(SaoePluginError {
                message: format!("calendar{count}"),
            });
        }
        let start = time("2024-01-01 00:00:00") + TimeDelta::days(*count);
        Ok((start, start + TimeDelta::hours(1)))
    }
}

#[test]
fn single_order_generation_matches_real_source_helpers_and_decision_constructors() {
    let output = Command::new("python")
        .args([
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/single_order_strategy_contract.py"
            ),
            r"D:\code\github\qlib\qlib",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    for (name, fail_create, fail_calendar, custom) in [
        ("normal", false, None, false),
        ("custom", false, None, true),
        ("create", true, None, false),
        ("calendar1", false, Some(1), false),
        ("calendar2", false, Some(2), false),
    ] {
        let events = Events::default();
        let mut orders = Orders {
            events: Arc::clone(&events),
            fail: fail_create,
            custom,
        };
        let calendar = Calendar {
            events: Arc::clone(&events),
            count: Mutex::new(0),
            fail: fail_calendar,
        };
        let mut request = Order::new(
            "A",
            7.0,
            OrderDir::Sell,
            Some(time("2020-01-01 00:00:00")),
            Some(time("2020-01-02 00:00:00")),
        );
        request.set_deal_amount(4.0);
        request.set_factor(Some(2.0));
        let range: Option<SharedTradeRange> =
            (!custom).then(|| Arc::new(IdxTradeRange::new(1, 4)) as SharedTradeRange);
        let strategy = SingleOrderStrategy::new(request, range.clone());
        let mut decisions = Vec::new();
        let mut error = None;
        let previous: Vec<SharedOrderExecution> = vec![
            domain_core::OwnedOrderExecution {
                order: Order::new("unrelated", 500.0, OrderDir::Buy, None, None),
                trade_value: 1000.0,
                trade_cost: 10.0,
                trade_price: 2.0,
            }
            .into_shared(),
        ];
        for previous in [None, Some(previous.as_slice())] {
            match strategy.generate_trade_decision(previous, &mut orders, &calendar) {
                Ok(mut decision) => {
                    match (&range, decision.shared_trade_range()) {
                        (Some(expected), Some(actual)) => assert!(Arc::ptr_eq(expected, actual)),
                        (None, None) => {}
                        _ => panic!("trade range changed"),
                    }
                    assert_eq!(decision.orders().len(), 1);
                    let order = &decision.orders()[0];
                    assert_eq!(order.stock_id(), "A");
                    assert_eq!(order.direction(), OrderDir::Sell);
                    decisions.push(json!({"decision": [decision.start_time().to_string(), decision.end_time().to_string()],
                        "order": [order.start_time().unwrap().to_string(), order.end_time().unwrap().to_string()],
                        "amount": order.amount(), "deal_amount": order.deal_amount(), "factor": order.factor()}));
                    decision.orders_mut()[0].set_deal_amount(99.0);
                }
                Err(caught) => {
                    error = Some(caught.message);
                    break;
                }
            }
        }
        assert_eq!(
            json!({"events": *events.lock().unwrap(), "decisions": decisions, "error": error}),
            expected[name]
        );
    }
}

#[test]
fn helper_rebinding_retry_and_ieee_request_values_are_preserved() {
    for amount in [-0.0_f64, -2.5, f64::INFINITY, f64::NAN] {
        let events = Events::default();
        let calendar = Calendar {
            events: Arc::clone(&events),
            count: Mutex::new(0),
            fail: None,
        };
        let strategy =
            SingleOrderStrategy::new(Order::new("B", amount, OrderDir::Buy, None, None), None);
        let mut failing = Orders {
            events: Arc::clone(&events),
            fail: true,
            custom: false,
        };
        let error = strategy
            .generate_trade_decision(None, &mut failing, &calendar)
            .err()
            .unwrap();
        assert_eq!(
            error,
            SaoePluginError {
                message: "create".to_owned()
            }
        );
        assert_eq!(*calendar.count.lock().unwrap(), 0);
        let mut replacement = Orders {
            events: Arc::clone(&events),
            fail: false,
            custom: true,
        };
        let mut result = strategy
            .generate_trade_decision(Some(&[]), &mut replacement, &calendar)
            .unwrap();
        assert_eq!(result.orders()[0].amount().to_bits(), amount.to_bits());
        assert_eq!(result.orders()[0].direction(), OrderDir::Buy);
        assert_eq!(result.orders()[0].stock_id(), "B");
        assert_eq!(
            result.orders()[0].start_time(),
            Some(time("2022-01-01 00:00:00"))
        );
        result.orders_mut()[0].set_deal_amount(999.0);
        let next_calendar = Calendar {
            events: Arc::clone(&events),
            count: Mutex::new(10),
            fail: None,
        };
        let next = strategy
            .generate_trade_decision(None, &mut replacement, &next_calendar)
            .unwrap();
        assert_eq!(next.orders()[0].deal_amount().to_bits(), 0.0_f64.to_bits());
        assert_eq!(next.start_time(), time("2024-01-12 00:00:00"));
        assert_eq!(
            next.orders()[0].end_time(),
            Some(time("2024-01-13 01:00:00"))
        );
        assert_eq!(*calendar.count.lock().unwrap(), 2);
    }
}
