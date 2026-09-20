#![cfg(windows)]
use path::{PathQueryError, final_path, find_name};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    os::windows::{
        ffi::{OsStrExt, OsStringExt},
        fs::OpenOptionsExt,
    },
    path::{Path, PathBuf},
    process::Command,
};

fn result(value: Result<OsString, PathQueryError>) -> Value {
    match value {
        Ok(value) => json!({"value": value.encode_wide().collect::<Vec<_>>()}),
        Err(PathQueryError::EmbeddedNul) => json!({"error": "ValueError"}),
        Err(PathQueryError::Windows { code, .. }) => json!({"error": code}),
        Err(error) => panic!("unexpected allocation error: {error}"),
    }
}

#[test]
fn real_queries_match_python_including_no_share_open_and_surrogates() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("folder")).unwrap();
    std::fs::write(directory.path().join("MiXeD.txt"), "content").unwrap();
    let surrogate = OsString::from_wide(&[0xd800, 46, 116, 120, 116]);
    std::fs::write(directory.path().join(surrogate), "content").unwrap();
    let _locked = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .share_mode(0)
        .open(directory.path().join("locked.txt"))
        .unwrap();
    let script = r"
import json, nt, sys
from pathlib import Path
def units(text):
    b = text.encode('utf-16-le', errors='surrogatepass')
    return [int.from_bytes(b[i:i+2], 'little') for i in range(0,len(b),2)]
def call(function, text):
    try: return {'value': units(function(text))}
    except ValueError: return {'error': 'ValueError'}
    except OSError as e: return {'error': e.winerror}
base = Path(sys.argv[1])
inputs = [str(base / item) for item in ['MiXeD.txt','mixed.txt','folder','missing','missing/child','locked.txt','\ud800.txt','bad\0name','MiXeD*']]
inputs += ['', str(base), str(base / 'folder') + '\\']
print(json.dumps([[units(s),call(nt._getfinalpathname,s),call(nt._findfirstfile,s)] for s in inputs]))
";
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
    for case in expected.as_array().unwrap() {
        let input =
            OsString::from_wide(&serde_json::from_value::<Vec<u16>>(case[0].clone()).unwrap());
        assert_eq!(
            result(final_path(Path::new(&input)).map(PathBuf::into_os_string)),
            case[1],
            "{input:?}"
        );
        assert_eq!(result(find_name(Path::new(&input))), case[2], "{input:?}");
    }
    assert_eq!(
        std::fs::File::open(directory.path().join("locked.txt"))
            .unwrap_err()
            .raw_os_error(),
        Some(32)
    );
    assert!(
        final_path(&directory.path().join("locked.txt"))
            .unwrap()
            .ends_with("locked.txt")
    );
    assert_eq!(
        find_name(&directory.path().join("locked.txt")).unwrap(),
        "locked.txt"
    );
}
