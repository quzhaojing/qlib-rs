//! Isolated locale-grouping candidate probe; not a production formatter.
use serde_json::{Value, json};
use std::io::{self, Read};
use thousands::{Separable, SeparatorPolicy, digits};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&input).unwrap();
    let mut differences = Vec::new();
    for case in &cases {
        let groups: Vec<u8> = case["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        let output = case["digits"]
            .as_str()
            .unwrap()
            .separate_by_policy(SeparatorPolicy {
                groups: &groups,
                separator: case["separator"].as_str().unwrap(),
                digits: digits::ASCII_DECIMAL,
            });
        if output != case["output"].as_str().unwrap() {
            differences.push(json!({"case":case,"actual":output}));
        }
    }
    println!("{}", json!({"cases":cases.len(),"mismatches":differences}));
}
