//! Test-only candidate evaluation. Never used by the production filename adapter.
use rustpython_format::{CharLen, FormatSpec};
use serde_json::{Value, json};
use std::{
    io::{self, Read},
    ops::Deref,
};

struct Text<'a>(&'a str);
impl Deref for Text<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        self.0
    }
}
impl CharLen for Text<'_> {
    fn char_len(&self) -> usize {
        self.0.chars().count()
    }
}

fn rustpython(case: &Value) -> Result<String, String> {
    let spec = FormatSpec::parse(case["format"].as_str().unwrap()).map_err(|e| format!("{e:?}"))?;
    let kind = case["value"][0].as_str().unwrap();
    let value = case["value"][1].as_str().unwrap();
    match kind {
        "int" => spec.format_int(&value.parse().unwrap()),
        "float" => spec.format_float(value.parse().unwrap()),
        "text" => spec.format_string(&Text(value)),
        "bool" => spec.format_bool(value == "true"),
        "null" => return Err("no None formatter".into()),
        _ => unreachable!(),
    }
    .map_err(|e| format!("{e:?}"))
}
fn pyformat(case: &Value) -> Result<String, String> {
    let spec = case["format"].as_str().unwrap();
    let kind = case["value"][0].as_str().unwrap();
    let value = case["value"][1].as_str().unwrap();
    let value = match kind {
        "int" => pyformat_rs::Value::Int(value.parse().map_err(|_| "outside i128")?),
        "float" => pyformat_rs::Value::Float(value.parse().unwrap()),
        "text" => pyformat_rs::Value::Str(value.into()),
        "bool" => pyformat_rs::Value::Bool(value == "true"),
        "null" => pyformat_rs::Value::None,
        _ => unreachable!(),
    };
    pyformat_rs::str_format(&format!("{{0:{spec}}}"), &[value], &[]).map_err(|e| e.to_string())
}
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&input).unwrap();
    // Panics are evidence against a candidate, not a production recovery mechanism.
    std::panic::set_hook(Box::new(|_| {}));
    let mut mismatches = vec![];
    for (name, candidate) in [
        (
            "rustpython-format",
            rustpython as fn(&Value) -> Result<String, String>,
        ),
        ("pyformat-rs", pyformat),
    ] {
        let mut matched = 0;
        let mut panics = 0;
        let mut errors = vec![];
        for case in &cases {
            let result = std::panic::catch_unwind(|| candidate(case));
            let actual = match result {
                Ok(Ok(value)) => json!({"output":value,"error":false}),
                Ok(Err(error)) => json!({"output":null,"error":true,"detail":error}),
                Err(_) => {
                    panics += 1;
                    json!({"panic":true})
                }
            };
            let correct = actual.get("panic").is_none()
                && if case["error"].is_null() {
                    actual["output"] == case["output"] && actual["error"] == false
                } else {
                    actual["error"] == true
                };
            if correct {
                matched += 1;
            } else {
                errors.push(json!({"case":case,"actual":actual}));
            }
        }
        mismatches.push(json!({"candidate":name,"cases":cases.len(),"matched":matched,"panics":panics,"mismatches":errors}));
    }
    println!("{}", json!(mismatches));
}
