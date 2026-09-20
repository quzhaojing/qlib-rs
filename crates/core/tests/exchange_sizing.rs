use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::Schema;
use chrono::NaiveDateTime;
use domain_core::{
    DealPriceFields, ExchangeExecutionSizer, ExchangeQuoteProvider, ExchangeSizingError,
    FactorInput, FactorProvider, FactorProviderError, Quote, QuoteData, QuoteError, QuoteMethod,
    TimeRange,
};
use serde_json::{Value, json};

fn timestamp(text: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
}

fn direct(factor: f64) -> FactorInput<'static> {
    FactorInput {
        factor: Some(factor),
        stock: None,
        start_time: None,
        end_time: None,
    }
}

fn market() -> FactorInput<'static> {
    FactorInput {
        factor: None,
        stock: Some("A"),
        start_time: Some(timestamp("2024-01-02 09:30:00")),
        end_time: Some(timestamp("2024-01-02 10:00:00")),
    }
}

#[derive(Clone)]
struct TestFactorProvider {
    result: Result<Option<f64>, FactorProviderError>,
    calls: Arc<Mutex<Vec<(String, TimeRange)>>>,
}

impl FactorProvider for TestFactorProvider {
    fn factor(&self, stock: &str, range: TimeRange) -> Result<Option<f64>, FactorProviderError> {
        self.calls.lock().unwrap().push((stock.to_owned(), range));
        self.result.clone()
    }
}

#[test]
fn configuration_short_circuits_and_factor_resolution_match_exchange() {
    let missing = FactorInput {
        factor: None,
        stock: None,
        start_time: None,
        end_time: None,
    };
    let adjusted = ExchangeExecutionSizer::new(true, Some(100.0), 5.0, None);
    assert!(adjusted.trade_with_adjusted_price());
    assert_eq!(adjusted.trade_unit(), Some(100.0));
    assert_eq!(adjusted.min_cost().to_bits(), 5.0_f64.to_bits());
    assert_eq!(adjusted.amount_of_trade_unit(missing), Ok(None));
    assert_eq!(
        adjusted.round_amount_by_trade_unit(321.0, missing),
        Ok(321.0)
    );

    let disabled = ExchangeExecutionSizer::new(false, None, 5.0, None);
    assert!(!disabled.trade_with_adjusted_price());
    assert_eq!(disabled.trade_unit(), None);
    assert_eq!(disabled.amount_of_trade_unit(missing), Ok(None));
    assert_eq!(
        disabled.round_amount_by_trade_unit(321.0, missing),
        Ok(321.0)
    );

    let calls = Arc::new(Mutex::new(Vec::new()));
    let provider: Arc<dyn FactorProvider> = Arc::new(TestFactorProvider {
        result: Ok(Some(4.0)),
        calls: Arc::clone(&calls),
    });
    let sizer = ExchangeExecutionSizer::new(false, Some(100.0), 5.0, Some(provider));
    assert_eq!(sizer.amount_of_trade_unit(direct(2.0)), Ok(Some(50.0)));
    assert_eq!(
        sizer.round_amount_by_trade_unit(321.0, direct(2.0)),
        Ok(300.0)
    );
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(sizer.amount_of_trade_unit(market()), Ok(Some(25.0)));
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        [(
            "A".to_owned(),
            TimeRange {
                start: Some(timestamp("2024-01-02 09:30:00")),
                end: Some(timestamp("2024-01-02 10:00:00"))
            }
        )]
    );

    let no_provider = ExchangeExecutionSizer::new(false, Some(100.0), 5.0, None);
    assert_eq!(
        no_provider.amount_of_trade_unit(missing),
        Err(ExchangeSizingError::MissingFactorInput)
    );
    assert_eq!(
        no_provider.round_amount_by_trade_unit(321.0, missing),
        Err(ExchangeSizingError::MissingFactorInput)
    );
    assert_eq!(
        no_provider.amount_of_trade_unit(FactorInput {
            factor: None,
            stock: Some("A"),
            start_time: Some(timestamp("2024-01-02 09:30:00")),
            end_time: None,
        }),
        Err(ExchangeSizingError::MissingFactorInput)
    );
    assert_eq!(
        no_provider.amount_of_trade_unit(market()),
        Err(ExchangeSizingError::MissingFactorProvider)
    );

    for (result, expected) in [
        (Ok(None), ExchangeSizingError::MissingFactor),
        (
            Err(FactorProviderError {
                message: "offline".to_owned(),
            }),
            ExchangeSizingError::FactorProvider(FactorProviderError {
                message: "offline".to_owned(),
            }),
        ),
    ] {
        let provider: Arc<dyn FactorProvider> = Arc::new(TestFactorProvider {
            result,
            calls: Arc::new(Mutex::new(Vec::new())),
        });
        let failed = ExchangeExecutionSizer::new(false, Some(100.0), 5.0, Some(provider));
        assert_eq!(failed.amount_of_trade_unit(market()), Err(expected));
    }
}

#[test]
fn trade_unit_amounts_and_rounding_preserve_python_float_semantics() {
    let sizer = ExchangeExecutionSizer::new(false, Some(100.0), 5.0, None);
    for (factor, expected) in [
        (2.0, 50.0),
        (0.5, 200.0),
        (-2.0, -50.0),
        (f64::INFINITY, 0.0),
        (f64::NEG_INFINITY, -0.0),
    ] as [(f64, f64); 5]
    {
        assert_eq!(
            sizer
                .amount_of_trade_unit(direct(factor))
                .unwrap()
                .unwrap()
                .to_bits(),
            expected.to_bits()
        );
    }
    assert!(
        sizer
            .amount_of_trade_unit(direct(f64::NAN))
            .unwrap()
            .unwrap()
            .is_nan()
    );
    assert_eq!(
        sizer.amount_of_trade_unit(direct(0.0)),
        Err(ExchangeSizingError::ZeroFactor)
    );
    assert_eq!(
        sizer.amount_of_trade_unit(direct(-0.0)),
        Err(ExchangeSizingError::ZeroFactor)
    );

    for (unit, factor, amount, expected) in [
        (100.0, 2.0, 321.0, 300.0),
        (100.0, 2.0, 299.95, 300.0),
        (100.0, 2.0, -321.0, -350.0),
        (100.0, -2.0, 321.0, 350.0),
        (100.0, -2.0, 0.0, -0.0),
        (-100.0, 2.0, 0.0, 50.0),
        (f64::INFINITY, 2.0, -321.0, f64::NEG_INFINITY),
        (f64::INFINITY, -2.0, 321.0, f64::INFINITY),
        (f64::NEG_INFINITY, 2.0, 0.0, f64::INFINITY),
    ] {
        let candidate = ExchangeExecutionSizer::new(false, Some(unit), 5.0, None);
        assert_eq!(
            candidate
                .round_amount_by_trade_unit(amount, direct(factor))
                .unwrap()
                .to_bits(),
            expected.to_bits()
        );
    }
    for (factor, amount) in [
        (2.0, f64::NAN),
        (2.0, f64::INFINITY),
        (f64::NAN, 321.0),
        (f64::INFINITY, 321.0),
    ] {
        assert!(
            sizer
                .round_amount_by_trade_unit(amount, direct(factor))
                .unwrap()
                .is_nan()
        );
    }
    let rounding_correction =
        ExchangeExecutionSizer::new(false, Some(3.717_755_350_211_595_3e-16), 5.0, None);
    assert_eq!(
        rounding_correction
            .round_amount_by_trade_unit(
                -9.729_691_085_424_728e-91,
                direct(-7.530_717_256_364_051e-91),
            )
            .unwrap()
            .to_bits(),
        15_163_813_606_985_548_625
    );
    assert_eq!(
        sizer.round_amount_by_trade_unit(321.0, direct(0.0)),
        Err(ExchangeSizingError::ZeroFactor)
    );
    let zero_unit = ExchangeExecutionSizer::new(false, Some(0.0), 5.0, None);
    assert_eq!(
        zero_unit.round_amount_by_trade_unit(321.0, direct(0.0)),
        Err(ExchangeSizingError::ZeroTradeUnit)
    );
}

#[test]
fn cash_limited_buy_amounts_preserve_branch_and_division_order() {
    let sizer = ExchangeExecutionSizer::new(false, Some(100.0), 5.0, None);
    for (price, cash, ratio, expected) in [
        (100.0, 4.0, 0.01, 0.0),
        (100.0, 5.0, 0.01, 0.0),
        (100.0, 100.0, 0.1, 0.909_090_909_090_909_1),
        (100.0, 55.0, 0.1, 0.499_999_999_999_999_94),
        (100.0, f64::NAN, 0.1, 0.0),
        (f64::INFINITY, 55.0, 0.1, 0.0),
        (100.0, f64::INFINITY, 0.1, f64::INFINITY),
        (100.0, 10.0, f64::INFINITY, 0.0),
        (-100.0, 55.0, 0.1, -0.499_999_999_999_999_94),
    ] {
        assert_eq!(
            sizer
                .buy_amount_by_cash_limit(price, cash, ratio)
                .unwrap()
                .to_bits(),
            expected.to_bits()
        );
    }
    assert!(
        sizer
            .buy_amount_by_cash_limit(f64::NAN, 10.0, 0.1)
            .unwrap()
            .is_nan()
    );
    assert_eq!(sizer.buy_amount_by_cash_limit(100.0, 4.0, 0.0), Ok(0.0));
    assert_eq!(
        sizer.buy_amount_by_cash_limit(100.0, 5.0, 0.0),
        Err(ExchangeSizingError::ZeroCostRatio)
    );
    assert_eq!(
        sizer.buy_amount_by_cash_limit(100.0, 5.0, -1.0),
        Err(ExchangeSizingError::ZeroCostDenominator)
    );
    assert_eq!(
        sizer.buy_amount_by_cash_limit(0.0, 5.0, 0.1),
        Err(ExchangeSizingError::ZeroTradePrice)
    );
    assert_eq!(
        sizer.buy_amount_by_cash_limit(0.0, 55.0, 0.1),
        Err(ExchangeSizingError::ZeroTradePrice)
    );
    let nan_min = ExchangeExecutionSizer::new(false, Some(100.0), f64::NAN, None);
    assert_eq!(nan_min.buy_amount_by_cash_limit(100.0, 10.0, 0.1), Ok(0.0));
    assert_eq!(
        sizer
            .buy_amount_by_cash_limit(100.0, 10.0, f64::NAN)
            .unwrap()
            .to_bits(),
        0.05_f64.to_bits()
    );
    let negative_min = ExchangeExecutionSizer::new(false, Some(100.0), -5.0, None);
    assert_eq!(
        negative_min
            .buy_amount_by_cash_limit(100.0, -1.0, 0.1)
            .unwrap()
            .to_bits(),
        (-0.009_090_909_090_909_09_f64).to_bits()
    );
}

enum QuoteScenario {
    Scalar(f64),
    Missing,
    Series,
    Failure,
}

type QuoteCall = (String, TimeRange, String, QuoteMethod);

struct FactorQuote {
    scenario: QuoteScenario,
    calls: Arc<Mutex<Vec<QuoteCall>>>,
}

impl Quote for FactorQuote {
    fn get_all_stock(&self) -> Vec<String> {
        vec!["A".to_owned()]
    }

    fn get_data(
        &self,
        stock: &str,
        range: TimeRange,
        field: &str,
        method: QuoteMethod,
    ) -> Result<Option<QuoteData>, QuoteError> {
        self.calls
            .lock()
            .unwrap()
            .push((stock.to_owned(), range, field.to_owned(), method));
        match self.scenario {
            QuoteScenario::Scalar(value) => {
                Ok(Some(QuoteData::Scalar(Arc::new(Float64Array::from(vec![
                    value,
                ])))))
            }
            QuoteScenario::Missing => Ok(None),
            QuoteScenario::Series => Ok(Some(QuoteData::Series(RecordBatch::new_empty(Arc::new(
                Schema::empty(),
            ))))),
            QuoteScenario::Failure => Err(QuoteError::MissingStock {
                stock: stock.to_owned(),
            }),
        }
    }
}

#[test]
fn exchange_quote_is_a_factor_provider_with_typed_adapter_failures() {
    for (scenario, expected) in [
        (QuoteScenario::Scalar(2.0), Ok(Some(2.0))),
        (QuoteScenario::Missing, Ok(None)),
    ] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let quote: Arc<dyn Quote> = Arc::new(FactorQuote {
            scenario,
            calls: Arc::clone(&calls),
        });
        let exchange = ExchangeQuoteProvider::new(quote, DealPriceFields::shared("close").unwrap());
        assert_eq!(
            FactorProvider::factor(&exchange, "A", TimeRange::default()),
            expected
        );
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            [(
                "A".to_owned(),
                TimeRange::default(),
                "$factor".to_owned(),
                QuoteMethod::LastValid
            )]
        );
    }

    for (scenario, expected_message) in [
        (
            QuoteScenario::Series,
            "aggregated quote result must be scalar, not a series",
        ),
        (
            QuoteScenario::Failure,
            "stock A is not present in the quote",
        ),
    ] {
        let exchange = ExchangeQuoteProvider::new(
            Arc::new(FactorQuote {
                scenario,
                calls: Arc::new(Mutex::new(Vec::new())),
            }),
            DealPriceFields::shared("close").unwrap(),
        );
        assert_eq!(
            FactorProvider::factor(&exchange, "A", TimeRange::default()),
            Err(FactorProviderError {
                message: expected_message.to_owned()
            })
        );
    }
}

#[test]
fn execution_sizing_matches_live_python_exchange_methods() {
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib/qlib/backtest/exchange.py");
    let script = r"
import ast,json,math,sys
p=sys.argv[1];t=ast.parse(open(p,encoding='utf-8').read(),filename=p);c=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='Exchange');names=['_get_factor_or_raise_error','get_amount_of_trade_unit','round_amount_by_trade_unit','_get_buy_amount_by_cash_limit'];nodes=[next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==n) for n in names];ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),p,'exec'),globals(),ns)
class X:
 _get_factor_or_raise_error=ns['_get_factor_or_raise_error'];get_amount_of_trade_unit=ns['get_amount_of_trade_unit'];round_amount_by_trade_unit=ns['round_amount_by_trade_unit'];_get_buy_amount_by_cash_limit=ns['_get_buy_amount_by_cash_limit']
 def __init__(self,adj=False,unit=100,min_cost=5,result=4,error=None):self.trade_w_adj_price=adj;self.trade_unit=unit;self.min_cost=min_cost;self.result=result;self.error=error;self.calls=0
 def get_factor(self,**kwargs):self.calls+=1;(_ for _ in ()).throw(RuntimeError(self.error)) if self.error else None;return self.result
def norm(v):
 if isinstance(v,float) and math.isnan(v):return 'nan'
 if isinstance(v,float) and math.isinf(v):return '+inf' if v>0 else '-inf'
 if isinstance(v,float) and v==0:return '-0' if math.copysign(1,v)<0 else 0
 return v
def out(f):
 try:return ['ok',norm(f())]
 except BaseException as e:return [type(e).__name__,str(e)]
r={};x=X(adj=True);r['adjusted']=[out(lambda:x.get_amount_of_trade_unit()),out(lambda:x.round_amount_by_trade_unit(321)),x.calls];x=X(unit=None);r['disabled']=[out(lambda:x.get_amount_of_trade_unit()),out(lambda:x.round_amount_by_trade_unit(321)),x.calls];x=X();r['direct']=[out(lambda:x.get_amount_of_trade_unit(factor=2)),out(lambda:x.round_amount_by_trade_unit(321,factor=2)),x.calls];x=X();r['provider']=[out(lambda:x.get_amount_of_trade_unit(stock_id='A',start_time=1,end_time=2)),x.calls];r['missing']=out(lambda:X().get_amount_of_trade_unit());r['provider_none']=out(lambda:X(result=None).get_amount_of_trade_unit(stock_id='A',start_time=1,end_time=2));r['provider_error']=out(lambda:X(error='offline').get_amount_of_trade_unit(stock_id='A',start_time=1,end_time=2))
r['units']={str(f):out(lambda f=f:X().get_amount_of_trade_unit(factor=f)) for f in [0.0,-2.0,float('nan'),float('inf'),float('-inf')]};r['round']={k:out(f) for k,f in {'normal':lambda:X().round_amount_by_trade_unit(321,2),'precision':lambda:X().round_amount_by_trade_unit(299.95,2),'negative':lambda:X().round_amount_by_trade_unit(-321,2),'negative_factor':lambda:X().round_amount_by_trade_unit(321,-2),'zero_factor':lambda:X().round_amount_by_trade_unit(321,0),'zero_unit':lambda:X(unit=0).round_amount_by_trade_unit(321,2),'nan':lambda:X().round_amount_by_trade_unit(float('nan'),2),'inf':lambda:X().round_amount_by_trade_unit(float('inf'),2)}.items()};r['cash']={k:out(f) for k,f in {'below':lambda:X()._get_buy_amount_by_cash_limit(100,4,.01),'minimum':lambda:X()._get_buy_amount_by_cash_limit(100,5,.01),'fixed':lambda:X()._get_buy_amount_by_cash_limit(100,100,.1),'rate':lambda:X()._get_buy_amount_by_cash_limit(100,55,.1),'zero_ratio_below':lambda:X()._get_buy_amount_by_cash_limit(100,4,0),'zero_ratio':lambda:X()._get_buy_amount_by_cash_limit(100,5,0),'minus_one':lambda:X()._get_buy_amount_by_cash_limit(100,5,-1),'zero_price':lambda:X()._get_buy_amount_by_cash_limit(0,5,.1),'nan_cash':lambda:X()._get_buy_amount_by_cash_limit(100,float('nan'),.1),'nan_ratio':lambda:X()._get_buy_amount_by_cash_limit(100,10,float('nan')),'inf_cash':lambda:X()._get_buy_amount_by_cash_limit(100,float('inf'),.1)}.items()};print(json.dumps(r,allow_nan=True))
";
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
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
    let actual: Value = serde_json::from_slice(&output.stdout).unwrap();
    let expected = json!({
        "adjusted": [["ok", null], ["ok", 321], 0],
        "disabled": [["ok", null], ["ok", 321], 0],
        "direct": [["ok", 50.0], ["ok", 300.0], 0],
        "provider": [["ok", 25.0], 1],
        "missing": ["ValueError", "`factor` and (`stock_id`, `start_time`, `end_time`) can't both be None"],
        "provider_none": ["AssertionError", ""],
        "provider_error": ["RuntimeError", "offline"],
        "units": {
            "0.0": ["ZeroDivisionError", "division by zero"], "-2.0": ["ok", -50.0],
            "nan": ["ok", "nan"], "inf": ["ok", 0], "-inf": ["ok", "-0"]
        },
        "round": {
            "normal": ["ok", 300.0], "precision": ["ok", 300.0], "negative": ["ok", -350.0],
            "negative_factor": ["ok", 350.0], "zero_factor": ["ZeroDivisionError", "division by zero"],
            "zero_unit": ["ZeroDivisionError", "division by zero"], "nan": ["ok", "nan"], "inf": ["ok", "nan"]
        },
        "cash": {
            "below": ["ok", 0], "minimum": ["ok", 0], "fixed": ["ok", 0.909_090_909_090_909_2],
            "rate": ["ok", 0.499_999_999_999_999_94], "zero_ratio_below": ["ok", 0],
            "zero_ratio": ["ZeroDivisionError", "division by zero"], "minus_one": ["ZeroDivisionError", "division by zero"],
            "zero_price": ["ZeroDivisionError", "division by zero"], "nan_cash": ["ok", 0],
            "nan_ratio": ["ok", 0.05], "inf_cash": ["ok", "+inf"]
        }
    });
    assert_eq!(actual, expected);
}
