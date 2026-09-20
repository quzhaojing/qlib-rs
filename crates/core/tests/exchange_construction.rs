use std::{
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

use domain_core::{
    ExchangeCodes, ExchangeConfiguration, ExchangeConstructionPluginError,
    ExchangeConstructionRequest, ExchangeDealPriceInput, ExchangeDefaults, ExchangeFactory,
    ExchangeInstance, ExchangeLimitThreshold, ExchangeSource, ExchangeTimeInput, GetExchangeError,
    Region, RegionExchangeDefaults, get_exchange,
};
use indexmap::IndexMap;
use serde_json::{Value, json};

#[derive(Debug)]
struct Marker;

struct Defaults {
    events: Arc<Mutex<Vec<Value>>>,
    value: Result<Option<ExchangeLimitThreshold>, ExchangeConstructionPluginError>,
}

impl ExchangeDefaults for Defaults {
    fn limit_threshold(
        &self,
    ) -> Result<Option<ExchangeLimitThreshold>, ExchangeConstructionPluginError> {
        self.events.lock().unwrap().push(json!(["default"]));
        self.value.clone()
    }
}

struct Factory {
    events: Arc<Mutex<Vec<Value>>>,
    marker: Arc<dyn ExchangeInstance>,
    fail_create: bool,
    fail_resolve: bool,
}

fn threshold_value(value: Option<&ExchangeLimitThreshold>) -> Value {
    match value {
        None => Value::Null,
        Some(ExchangeLimitThreshold::Rate(value)) => json!(["rate", value]),
        Some(ExchangeLimitThreshold::Expressions { buy, sell }) => {
            json!(["expressions", buy, sell])
        }
    }
}

fn time_value(value: Option<ExchangeTimeInput>) -> Value {
    match value {
        None => Value::Null,
        Some(ExchangeTimeInput::Timestamp(value)) => json!(["timestamp", value.to_string()]),
        Some(ExchangeTimeInput::Text(value)) => json!(["text", value]),
    }
}

fn codes_value(value: ExchangeCodes) -> Value {
    match value {
        ExchangeCodes::Universe(value) => json!(["universe", value]),
        ExchangeCodes::Instruments(value) => json!(["instruments", value]),
    }
}

fn deal_price_value(value: Option<ExchangeDealPriceInput>) -> Value {
    match value {
        None => Value::Null,
        Some(ExchangeDealPriceInput::Shared(value)) => json!(["shared", value]),
        Some(ExchangeDealPriceInput::Sequence(value)) => json!(["sequence", value]),
    }
}

impl ExchangeFactory for Factory {
    fn create(
        &self,
        request: ExchangeConstructionRequest,
    ) -> Result<Arc<dyn ExchangeInstance>, ExchangeConstructionPluginError> {
        let ExchangeConstructionRequest {
            frequency,
            start_time,
            end_time,
            codes,
            subscribe_fields,
            open_cost,
            close_cost,
            min_cost,
            limit_threshold,
            deal_price,
            extra_arguments,
        } = request;
        self.events.lock().unwrap().push(json!([
            "create",
            {
                "frequency": frequency,
                "start_time": time_value(start_time),
                "end_time": time_value(end_time),
                "codes": codes_value(codes),
                "subscribe_fields": subscribe_fields,
                "open_cost": open_cost,
                "close_cost": close_cost,
                "min_cost": min_cost,
                "limit_threshold": threshold_value(limit_threshold.as_ref()),
                "deal_price": deal_price_value(deal_price),
                "extra_arguments": extra_arguments,
            },
        ]));
        if self.fail_create {
            Err(ExchangeConstructionPluginError {
                message: "create".to_owned(),
            })
        } else {
            Ok(Arc::clone(&self.marker))
        }
    }

    fn resolve(
        &self,
        source: ExchangeSource,
    ) -> Result<Arc<dyn ExchangeInstance>, ExchangeConstructionPluginError> {
        let (event, result) = match source {
            ExchangeSource::New => panic!("new sources use create"),
            ExchangeSource::Existing(existing) => (json!(["resolve", "existing"]), existing),
            ExchangeSource::Configuration(ExchangeConfiguration::Name(name)) => {
                (json!(["resolve", "name", name]), Arc::clone(&self.marker))
            }
            ExchangeSource::Configuration(ExchangeConfiguration::Mapping(mapping)) => (
                json!(["resolve", "mapping", mapping]),
                Arc::clone(&self.marker),
            ),
            ExchangeSource::Configuration(ExchangeConfiguration::Path(path)) => (
                json!(["resolve", "path", path.to_string_lossy()]),
                Arc::clone(&self.marker),
            ),
        };
        self.events.lock().unwrap().push(event);
        if self.fail_resolve {
            Err(ExchangeConstructionPluginError {
                message: "resolve".to_owned(),
            })
        } else {
            Ok(result)
        }
    }
}

fn request(threshold: Option<ExchangeLimitThreshold>) -> ExchangeConstructionRequest {
    ExchangeConstructionRequest {
        frequency: "1min".to_owned(),
        start_time: Some(ExchangeTimeInput::Text("2024-01-02".to_owned())),
        end_time: Some(ExchangeTimeInput::Timestamp(
            chrono::NaiveDate::from_ymd_opt(2024, 1, 31)
                .unwrap()
                .and_hms_opt(16, 0, 0)
                .unwrap(),
        )),
        codes: ExchangeCodes::Universe("all".to_owned()),
        subscribe_fields: vec!["$vwap".to_owned()],
        open_cost: 0.1,
        close_cost: 0.2,
        min_cost: 3.0,
        limit_threshold: threshold,
        deal_price: Some(ExchangeDealPriceInput::Sequence(vec![
            "$ask".to_owned(),
            "$bid".to_owned(),
        ])),
        extra_arguments: IndexMap::from([("extra".to_owned(), json!(7))]),
    }
}

fn error(result: Result<Arc<dyn ExchangeInstance>, GetExchangeError>) -> GetExchangeError {
    match result {
        Ok(_) => panic!("operation should fail"),
        Err(error) => error,
    }
}

#[test]
fn unchanged_source_dispatch_order_arguments_identity_and_failures_are_frozen() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/get_exchange_contract.py"
        ))
        .output()
        .expect("Python characterization fixture runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&output.stdout).expect("fixture emits JSON");
    let cases = actual.as_array().expect("fixture emits cases");
    assert_eq!(cases.len(), 7);
    assert_eq!(cases[0]["events"][0], json!(["default"]));
    assert_eq!(cases[0]["events"][1], json!(["log", "Create new exchange"]));
    assert_eq!(cases[0]["events"][2][0], "create");
    assert_eq!(cases[0]["events"][2][1]["limit_threshold"], 0.095);
    assert_eq!(cases[0]["returned"], "new");
    assert_eq!(cases[1]["events"][0], json!(["log", "Create new exchange"]));
    assert_eq!(
        cases[1]["events"][1][1]["limit_threshold"],
        json!(["up", "down"])
    );
    assert_eq!(
        cases[2]["events"],
        json!([["resolve", {"class":"Configured"}, true]])
    );
    assert_eq!(cases[2]["returned"], "marker");
    assert_eq!(
        cases[3]["events"],
        json!([["default"], ["resolve", "existing", true]])
    );
    assert_eq!(cases[3]["returned"], "existing");
    assert_eq!(cases[4]["events"], json!([["default"]]));
    assert_eq!(cases[4]["error"], "RuntimeError:default");
    assert_eq!(cases[5]["events"][0], json!(["log", "Create new exchange"]));
    assert_eq!(cases[5]["error"], "RuntimeError:create");
    assert_eq!(cases[6]["events"], json!([["resolve", "fail", true]]));
    assert_eq!(cases[6]["error"], "RuntimeError:resolve");
}

#[test]
fn new_exchange_reads_live_default_before_factory_and_forwards_all_owned_values() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let marker: Arc<dyn ExchangeInstance> = Arc::new(Marker);
    let defaults = Defaults {
        events: Arc::clone(&events),
        value: Ok(Some(ExchangeLimitThreshold::Rate(0.095))),
    };
    let factory = Factory {
        events: Arc::clone(&events),
        marker: Arc::clone(&marker),
        fail_create: false,
        fail_resolve: false,
    };
    let result = get_exchange(ExchangeSource::New, request(None), &defaults, &factory).unwrap();
    assert!(Arc::ptr_eq(&result, &marker));
    assert_eq!(
        events.lock().unwrap().as_slice(),
        &[
            json!(["default"]),
            json!(["create", {
                "frequency": "1min",
                "start_time": ["text", "2024-01-02"],
                "end_time": ["timestamp", "2024-01-31 16:00:00"],
                "codes": ["universe", "all"],
                "subscribe_fields": ["$vwap"],
                "open_cost": 0.1,
                "close_cost": 0.2,
                "min_cost": 3.0,
                "limit_threshold": ["rate", 0.095],
                "deal_price": ["sequence", ["$ask", "$bid"]],
                "extra_arguments": {"extra":7},
            }]),
        ]
    );

    events.lock().unwrap().clear();
    let result = get_exchange(
        ExchangeSource::New,
        request(Some(ExchangeLimitThreshold::Expressions {
            buy: "up".to_owned(),
            sell: "down".to_owned(),
        })),
        &defaults,
        &factory,
    )
    .unwrap();
    assert!(Arc::ptr_eq(&result, &marker));
    assert_eq!(
        events.lock().unwrap().as_slice(),
        &[json!(["create", {
            "frequency": "1min",
            "start_time": ["text", "2024-01-02"],
            "end_time": ["timestamp", "2024-01-31 16:00:00"],
            "codes": ["universe", "all"],
            "subscribe_fields": ["$vwap"],
            "open_cost": 0.1,
            "close_cost": 0.2,
            "min_cost": 3.0,
            "limit_threshold": ["expressions", "up", "down"],
            "deal_price": ["sequence", ["$ask", "$bid"]],
            "extra_arguments": {"extra":7},
        }])]
    );
}

#[test]
fn configured_sources_resolve_after_optional_default_lookup_and_keep_identity() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let marker: Arc<dyn ExchangeInstance> = Arc::new(Marker);
    let defaults = Defaults {
        events: Arc::clone(&events),
        value: Ok(None),
    };
    let factory = Factory {
        events: Arc::clone(&events),
        marker: Arc::clone(&marker),
        fail_create: false,
        fail_resolve: false,
    };
    let existing: Arc<dyn ExchangeInstance> = Arc::new(7_u8);
    let result = get_exchange(
        ExchangeSource::Existing(Arc::clone(&existing)),
        request(None),
        &defaults,
        &factory,
    )
    .unwrap();
    assert!(Arc::ptr_eq(&result, &existing));
    assert_eq!(
        events.lock().unwrap().as_slice(),
        &[json!(["default"]), json!(["resolve", "existing"])]
    );

    for (configuration, expected) in [
        (
            ExchangeConfiguration::Name("named".to_owned()),
            json!(["resolve", "name", "named"]),
        ),
        (
            ExchangeConfiguration::Mapping(IndexMap::from([("class".to_owned(), json!("C"))])),
            json!(["resolve", "mapping", {"class":"C"}]),
        ),
        (
            ExchangeConfiguration::Path(PathBuf::from("x.pkl")),
            json!(["resolve", "path", "x.pkl"]),
        ),
    ] {
        events.lock().unwrap().clear();
        let result = get_exchange(
            ExchangeSource::Configuration(configuration),
            request(Some(ExchangeLimitThreshold::Rate(0.2))),
            &defaults,
            &factory,
        )
        .unwrap();
        assert!(Arc::ptr_eq(&result, &marker));
        assert_eq!(events.lock().unwrap().as_slice(), &[expected]);
    }
}

#[test]
fn default_create_and_resolve_failures_keep_their_stage() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let marker: Arc<dyn ExchangeInstance> = Arc::new(Marker);
    let failing_defaults = Defaults {
        events: Arc::clone(&events),
        value: Err(ExchangeConstructionPluginError {
            message: "default".to_owned(),
        }),
    };
    let mut factory = Factory {
        events: Arc::clone(&events),
        marker,
        fail_create: false,
        fail_resolve: false,
    };
    assert_eq!(
        error(get_exchange(
            ExchangeSource::New,
            request(None),
            &failing_defaults,
            &factory,
        )),
        GetExchangeError::Defaults(ExchangeConstructionPluginError {
            message: "default".to_owned()
        })
    );
    assert_eq!(events.lock().unwrap().as_slice(), &[json!(["default"])]);

    events.lock().unwrap().clear();
    let defaults = Defaults {
        events: Arc::clone(&events),
        value: Ok(None),
    };
    factory.fail_create = true;
    assert_eq!(
        error(get_exchange(
            ExchangeSource::New,
            request(Some(ExchangeLimitThreshold::Rate(0.1))),
            &defaults,
            &factory,
        )),
        GetExchangeError::Factory(ExchangeConstructionPluginError {
            message: "create".to_owned()
        })
    );
    factory.fail_create = false;
    factory.fail_resolve = true;
    assert_eq!(
        error(get_exchange(
            ExchangeSource::Configuration(ExchangeConfiguration::Name("x".to_owned())),
            request(Some(ExchangeLimitThreshold::Rate(0.1))),
            &defaults,
            &factory,
        )),
        GetExchangeError::Factory(ExchangeConstructionPluginError {
            message: "resolve".to_owned()
        })
    );
}

#[test]
fn region_defaults_cover_each_current_market_policy() {
    for (region, expected) in [
        (Region::Cn, Some(0.095_f64)),
        (Region::Us, None),
        (Region::Tw, Some(0.1_f64)),
    ] {
        let actual = RegionExchangeDefaults { region }.limit_threshold().unwrap();
        let actual = actual.map(|threshold| match threshold {
            ExchangeLimitThreshold::Rate(value) => value,
            ExchangeLimitThreshold::Expressions { .. } => panic!("region default is a rate"),
        });
        assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
    }
}
