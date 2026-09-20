use super::*;
use domain_core::SaoeBacktestDataSource;
use domain_core::saoe_live_market::{LiveSaoeMarketError as Error, read_live_saoe_market};
use std::sync::RwLock;

#[path = "saoe_live_numerical.rs"]
mod numerical;

struct Source {
    mode: &'static str,
    order: Arc<RwLock<Order>>,
    events: Mutex<Vec<String>>,
    mutate_failure: bool,
}

fn poison(order: Arc<RwLock<Order>>) {
    assert!(
        std::thread::spawn(move || {
            let _guard = order.try_write().unwrap();
            panic!("deliberate order poison");
        })
        .join()
        .is_err()
    );
}

impl Source {
    fn mutate(&self, name: &str, amount: f64) {
        *self.order.try_write().unwrap() =
            Order::new(name, amount, OrderDir::Sell, Some(time(0)), Some(time(1)));
    }
}

impl SaoeBacktestDataSource for Source {
    fn quote_timestamps(&self) -> Result<Vec<NaiveDateTime>, SaoePluginError> {
        panic!("per-step market reads must not query the backtest timeline")
    }

    fn market_volumes(
        &self,
        stock: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> Result<ndarray::Array1<f64>, SaoePluginError> {
        assert_eq!((start, end), (time(0), time(1)));
        assert!(self.order.try_write().is_ok());
        self.events.lock().unwrap().push(format!("volume:{stock}"));
        match self.mode {
            "volume" => self.mutate("V", 20.0),
            "volume-failure" => {
                if self.mutate_failure {
                    self.mutate("F", 20.0);
                }
                return Err(SaoePluginError {
                    message: "volume failed".to_owned(),
                });
            }
            "poison-between" => poison(self.order.clone()),
            _ => (),
        }
        Ok(arr1(&[100.0, 200.0]))
    }

    fn deal_prices(
        &self,
        stock: &str,
        start: NaiveDateTime,
        end: NaiveDateTime,
        direction: OrderDir,
    ) -> Result<ndarray::Array1<f64>, SaoePluginError> {
        assert_eq!((start, end), (time(0), time(1)));
        assert!(self.order.try_write().is_ok());
        self.events
            .lock()
            .unwrap()
            .push(format!("price:{stock}:{}", direction.value()));
        match self.mode {
            "price" => self.mutate("P", 40.0),
            "price-failure" => {
                if self.mutate_failure {
                    self.mutate("F", 40.0);
                }
                return Err(SaoePluginError {
                    message: "price failed".to_owned(),
                });
            }
            // Query success must not perform a spurious post-callback order read.
            "poison-after" => poison(self.order.clone()),
            _ => (),
        }
        Ok(if self.mode == "raw" {
            arr1(&[f64::NAN])
        } else {
            arr1(&[10.0, 12.0])
        })
    }
}

fn source(mode: &'static str) -> Source {
    Source {
        mode,
        order: Arc::new(RwLock::new(order(OrderDir::Buy))),
        events: Mutex::new(Vec::new()),
        mutate_failure: true,
    }
}

#[test]
fn live_market_matches_actual_source_callbacks_and_releases_order_guards() {
    let expected = live_contract::source();
    for mode in [
        "normal",
        "volume",
        "price",
        "volume-failure",
        "price-failure",
    ] {
        let source = source(mode);
        let result = read_live_saoe_market(&source, &source.order, time(0), time(1));
        match mode {
            "volume-failure" => assert!(matches!(result, Err(Error::Volume(_)))),
            "price-failure" => assert!(matches!(result, Err(Error::Price(_)))),
            _ => {
                let result = result.unwrap();
                assert_eq!(result.volume, arr1(&[100.0, 200.0]));
                assert_eq!(result.price, arr1(&[10.0, 12.0]));
            }
        }
        let events: Vec<_> = expected[mode]["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .filter(|event| event.starts_with("volume:") || event.starts_with("price:"))
            .map(str::to_owned)
            .collect();
        assert_eq!(*source.events.lock().unwrap(), events, "{mode}");
        let name = match mode {
            "volume" => "V",
            "price" => "P",
            "volume-failure" | "price-failure" => "F",
            _ => "A",
        };
        assert_eq!(source.order.read().unwrap().stock_id(), name);
    }
}

#[test]
fn live_market_poison_and_raw_arrays_preserve_exact_stage_boundaries() {
    for mode in ["poison-before", "poison-between", "poison-after", "raw"] {
        let source = source(mode);
        if mode == "poison-before" {
            poison(source.order.clone());
        }
        let result = read_live_saoe_market(&source, &source.order, time(0), time(1));
        match mode {
            "poison-before" => assert!(matches!(result, Err(Error::VolumeOrderPoisoned))),
            "poison-between" => assert!(matches!(result, Err(Error::PriceOrderPoisoned))),
            "poison-after" => assert_eq!(result.unwrap().price, arr1(&[10.0, 12.0])),
            "raw" => {
                let result = result.unwrap();
                assert_eq!(result.volume.len(), 2);
                assert_eq!(result.price.len(), 1);
                assert!(result.price[0].is_nan());
            }
            _ => unreachable!(),
        }
        let expected = match mode {
            "poison-before" => vec![],
            "poison-between" => vec!["volume:A"],
            _ => vec!["volume:A", "price:A:1"],
        };
        assert_eq!(*source.events.lock().unwrap(), expected, "{mode}");
    }
}
