//! Exhaustive dependency evaluation, separate from production acceptance.
use intl::unicode::{Group, age, general_category};
use rustpython_literal::escape::UnicodeEscape;
use serde_json::{Value, json};
use std::process::Command;

fn main() {
    let output = Command::new("python")
        .args([
            "-c",
            "import json,sys,unicodedata; assert unicodedata.unidata_version == '16.0.0'; json.dump([[repr(chr(cp)),chr(cp).isprintable()] for cp in range(0x110000) if not 0xd800 <= cp <= 0xdfff],sys.stdout)",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<(String, bool)> = serde_json::from_slice(&output.stdout).unwrap();
    let mut repr_mismatches = 0;
    let mut property_mismatches = 0;
    let mut samples: Vec<Value> = Vec::new();
    let mut property_samples: Vec<Value> = Vec::new();
    let mut checked = 0;
    for (ch, (repr, printable)) in (0..=0x0010_ffff)
        .filter_map(char::from_u32)
        .zip(expected.iter())
    {
        checked += 1;
        let text = ch.to_string();
        let actual = UnicodeEscape::new_repr(&text)
            .str_repr()
            .to_string()
            .unwrap();
        if actual != *repr {
            repr_mismatches += 1;
            if samples.len() < 20 {
                samples.push(
                    json!({"cp":format!("U+{:04X}",u32::from(ch)),"actual":actual,"expected":repr}),
                );
            }
        }
        let property = ch == ' '
            || (age(ch).is_some_and(|version| version <= (16, 0))
                && !matches!(
                    general_category(ch).group(),
                    Group::Other | Group::Separator
                ));
        if property != *printable {
            property_mismatches += 1;
            if property_samples.len() < 20 {
                property_samples.push(json!({"cp":format!("U+{:04X}",u32::from(ch)),"actual":property,"expected":printable}));
            }
        }
    }
    assert_eq!(checked, 0x0011_0000 - 0x800);
    assert_eq!(checked, expected.len());
    println!(
        "{}",
        json!({"checked":checked,"rustpython_repr_mismatches":repr_mismatches,"samples":samples,"age_filtered_intl_printable_mismatches":property_mismatches,"property_samples":property_samples})
    );
}
