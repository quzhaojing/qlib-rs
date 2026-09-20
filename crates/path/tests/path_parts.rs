#![cfg(windows)]
use std::{
    ffi::OsString,
    os::windows::ffi::{OsStrExt, OsStringExt},
    path::Path,
    process::Command,
};

type Case = (Vec<u16>, [Vec<u16>; 3], [Vec<u16>; 2], bool);

#[test]
fn windows_path_parts_match_actual_python() {
    let output = Command::new("python")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/path_parts.py"
        ))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<Case> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(cases.len(), 126_771);
    for (input, root_parts, parts, absolute) in cases {
        let spelling = OsString::from_wide(&input);
        let input_path = Path::new(&spelling);
        let (drive, root, tail) = path::split_root(input_path);
        assert_eq!(
            [
                drive.encode_wide().collect::<Vec<_>>(),
                root.encode_wide().collect(),
                tail.encode_wide().collect()
            ],
            root_parts,
            "root {input:?}"
        );
        let mut reconstructed = drive;
        reconstructed.push(root);
        reconstructed.push(tail);
        assert_eq!(reconstructed, spelling);
        let (directory, name) = path::split(input_path);
        assert_eq!(
            [
                directory.as_os_str().encode_wide().collect::<Vec<_>>(),
                name.encode_wide().collect()
            ],
            parts,
            "split {input:?}"
        );
        assert_eq!(
            path::is_absolute(input_path),
            absolute,
            "absolute {input:?}"
        );
    }
}
