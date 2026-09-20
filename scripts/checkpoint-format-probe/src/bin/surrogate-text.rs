//! Lossless code-point/path candidate probe, not the migrated filename renderer.
use serde_json::{Value, json};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::{
    ffi::OsString,
    io::{self, Read},
    path::Path,
};
use widestring::U32String;

fn points(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|n| u32::try_from(n.as_u64().unwrap()).unwrap())
        .collect()
}

fn wide(text: &U32String) -> Vec<u16> {
    let mut units = Vec::new();
    for &point in text.as_slice() {
        // Delegate ordinary scalar encoding to std. Preserve just the otherwise
        // unrepresentable surrogate code point; don't implement a new codec.
        if let Some(ch) = char::from_u32(point) {
            let mut buffer = [0; 2];
            units.extend_from_slice(ch.encode_utf16(&mut buffer));
        } else {
            assert!((0xd800..=0xdfff).contains(&point));
            units.push(u16::try_from(point).unwrap());
        }
    }
    units
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let request: Value = serde_json::from_str(&input).unwrap();
    let strings = request["strings"].as_array().unwrap();
    let mut lossy_paths = 0;
    for case in strings {
        let raw = points(&case["points"]);
        let text = U32String::from_vec(raw.clone());
        assert_eq!(text.as_slice(), raw);
        assert_eq!(
            text.len(),
            usize::try_from(case["length"].as_u64().unwrap()).unwrap()
        );
        assert_eq!(text.to_string().is_ok(), case["scalar"].as_bool().unwrap());
        let expected: Vec<u16> = points(&case["utf16"])
            .into_iter()
            .map(|n| u16::try_from(n).unwrap())
            .collect();
        let units = wide(&text);
        assert_eq!(units, expected);
        let path = OsString::from_wide(&units);
        assert_eq!(path.encode_wide().collect::<Vec<_>>(), expected);
        // Expose, rather than hide, the library's lossy U32 -> OsString shortcut.
        if text.to_os_string().encode_wide().collect::<Vec<_>>() != expected {
            lossy_paths += 1;
        }
    }
    assert!(lossy_paths > 0);
    let root = Path::new(request["directory"].as_str().unwrap());
    let disk = request["disk"].as_array().unwrap();
    for (index, case) in disk.iter().enumerate() {
        let folder = root.join(index.to_string());
        std::fs::create_dir_all(&folder).unwrap();
        let name = OsString::from_wide(&wide(&U32String::from_vec(points(&case["points"]))));
        let path = folder.join(name);
        let result = std::fs::write(&path, b"checkpoint-payload");
        assert_eq!(result.is_err(), !case["error"].is_null());
        if result.is_ok() {
            assert_eq!(std::fs::read(&path).unwrap(), b"checkpoint-payload");
            let entries: Vec<_> = std::fs::read_dir(&folder)
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap();
            assert_eq!(entries.len(), 1);
            let listed: Vec<u32> = entries[0]
                .file_name()
                .encode_wide()
                .map(u32::from)
                .collect();
            assert_eq!(listed, points(&case["listed_utf16"]));
        }
    }
    let pair = U32String::from_vec(vec![0xd800, 0xdc00]);
    let scalar = U32String::from_vec(vec![0x10000]);
    assert_ne!(pair, scalar);
    assert_ne!(pair.len(), scalar.len());
    assert_eq!(wide(&pair), wide(&scalar));
    let folder = root.join("alias");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(
        folder.join(OsString::from_wide(&wide(&pair))),
        b"same-wide-path",
    )
    .unwrap();
    assert_eq!(
        std::fs::read(folder.join(OsString::from_wide(&wide(&scalar)))).unwrap(),
        b"same-wide-path"
    );
    println!(
        "{}",
        json!({"lossless_string_cases": strings.len(), "lossy_shortcut_mismatches": lossy_paths,
        "filesystem_cases": disk.len(), "windows_alias_verified": true,
        "qlib_filename_cases_characterized_not_yet_implemented": request["filenames"].as_array().unwrap().len(),
        "qlib_save_cases_characterized_not_yet_implemented": request["saves"].as_array().unwrap().len()})
    );
}
