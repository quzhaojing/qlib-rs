#![cfg(windows)]

use domain_core::windows_home::{HomeExpansionError, WindowsHomeEnvironment};
use serde_json::Value;
use std::{ffi::OsString, os::windows::ffi::OsStringExt, path::Path, process::Command};

#[test]
fn pathlib_construction_preserves_devices_and_malformed_prefixes() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/windows_path_objects.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 240);
    for case in cases {
        let input = native(&case[0]);
        let actual = WindowsHomeEnvironment::default()
            .expand(Path::new(&input))
            .unwrap();
        assert_eq!(actual.as_os_str(), native(&case[1]), "{input:?}");
    }
}

fn native(value: &Value) -> OsString {
    OsString::from_wide(&serde_json::from_value::<Vec<u16>>(value.clone()).unwrap())
}

#[test]
fn windows_home_expansion_matches_actual_pathlib_environment_matrix() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/windows_home_contract.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.as_array().unwrap().len(), 289);
    for case in cases.as_array().unwrap() {
        let field = |index: usize| {
            (!case["environment"][index].is_null()).then(|| native(&case["environment"][index]))
        };
        let environment = WindowsHomeEnvironment {
            user_profile: field(0),
            home_drive: field(1),
            home_path: field(2),
            user_name: field(3),
        };
        let input = native(&case["input"]);
        match environment.expand(Path::new(&input)) {
            Ok(path) => {
                assert!(case["error"].is_null(), "{case}");
                assert_eq!(path.as_os_str(), native(&case["result"]), "{case}");
            }
            Err(error) => {
                assert!(case["result"].is_null(), "{case}");
                assert_eq!(error.to_string(), case["error"].as_str().unwrap());
                assert_eq!(error, HomeExpansionError);
            }
        }
    }
}

#[test]
fn capture_reads_the_current_environment_without_changing_it() {
    let environment = WindowsHomeEnvironment::capture();
    assert_eq!(environment.user_profile, std::env::var_os("USERPROFILE"));
    assert_eq!(environment.home_drive, std::env::var_os("HOMEDRIVE"));
    assert_eq!(environment.home_path, std::env::var_os("HOMEPATH"));
    assert_eq!(environment.user_name, std::env::var_os("USERNAME"));
    let absent = WindowsHomeEnvironment::default();
    assert_eq!(absent.expand(Path::new("~")), Err(HomeExpansionError));
    assert_eq!(
        absent.expand(Path::new("ordinary")).unwrap(),
        Path::new("ordinary")
    );
}
