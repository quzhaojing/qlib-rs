use super::*;
use std::{
    cell::Cell,
    mem::{align_of, offset_of},
    process::Command,
};
use windows_sys::Win32::Foundation::SetLastError;

thread_local! { static RESULT: Cell<(u32, u32, u32)> = const { Cell::new((0, 0, 0)) }; }

unsafe extern "system" fn query(path: PCWSTR, class: i32, data: *mut c_void, size: u32) -> BOOL {
    assert_eq!(class, BASIC_BY_NAME);
    assert_eq!(size as usize, size_of::<BasicInformation>());
    // SAFETY: the adapter supplies live terminated input and exclusive SDK storage.
    unsafe {
        assert_eq!(*path, u16::from(b'x'));
    }
    RESULT.with(|result| {
        let (attributes, tag, error) = result.get();
        if error != 0 {
            // SAFETY: setting this thread's native error for the injected failure.
            unsafe {
                SetLastError(error);
            }
            return 0;
        }
        // SAFETY: fully initialized, aligned output allocated by the adapter.
        unsafe {
            (*data.cast::<BasicInformation>()).attributes = attributes;
            (*data.cast::<BasicInformation>()).tag = tag;
        }
        1
    })
}

#[test]
fn optional_api_and_query_outcomes_preserve_native_contract() {
    assert!(load_query("missing-path-test-module-843c.dll", SYMBOL).is_none());
    assert!(load_query("kernel32.dll", b"MissingPathTestExport843c\0").is_none());
    assert!(
        load_query(MODULE, SYMBOL).is_some(),
        "host must expose the oracle API"
    );
    assert_eq!(
        query_with(Path::new("x"), None),
        Err(PathQueryError::Windows {
            operation: "GetFileInformationByName",
            code: 50,
        })
    );
    for text in ["x\0", "x\0y"] {
        assert_eq!(
            query_with(Path::new(text), Some(query)),
            Err(PathQueryError::EmbeddedNul)
        );
    }
    for (attributes, tag, expected) in [
        (0, IO_REPARSE_TAG_SYMLINK, false),
        (FILE_ATTRIBUTE_REPARSE_POINT, 0, false),
        (FILE_ATTRIBUTE_REPARSE_POINT, 0xa000_0003, false),
        (FILE_ATTRIBUTE_REPARSE_POINT, IO_REPARSE_TAG_SYMLINK, true),
    ] {
        RESULT.set((attributes, tag, 0));
        let actual = query_with(Path::new("x"), Some(query)).unwrap();
        assert_eq!(
            actual,
            DirectoryAttributes {
                attributes,
                reparse_tag: tag
            }
        );
        assert_eq!(
            actual.attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0
                && actual.reparse_tag == IO_REPARSE_TAG_SYMLINK,
            expected
        );
    }
    for code in [
        1, 2, 3, 5, 21, 32, 50, 53, 67, 87, 123, 161, 206, 1920, 9999,
    ] {
        RESULT.set((0, 0, code));
        assert_eq!(
            query_with(Path::new("x"), Some(query)),
            Err(PathQueryError::Windows {
                operation: "GetFileInformationByName",
                code,
            })
        );
    }
    assert_eq!(
        symbolic_link_by_name(Path::new("x\0")),
        Err(PathQueryError::EmbeddedNul)
    );
}

#[test]
fn complete_rust_layout_matches_independently_compiled_windows_sdk() {
    let directory = tempfile::tempdir().unwrap();
    let target = "x86_64-pc-windows-msvc";
    let compiler = cc::Build::new()
        .cargo_metadata(false)
        .target(target)
        .host(target)
        .out_dir(directory.path())
        .opt_level(0)
        .get_compiler();
    assert!(compiler.is_like_msvc());
    let binary = directory.path().join("layout.exe");
    let object = directory.path().join("layout.obj");
    let output = compiler
        .to_command()
        .current_dir(directory.path())
        .args(["/nologo", "/WX"])
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/by_name_layout.c"
        ))
        .arg(format!("/Fe:{}", binary.display()))
        .arg(format!("/Fo:{}", object.display()))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(binary).output().unwrap();
    assert!(output.status.success());
    let actual: Vec<usize> = std::str::from_utf8(&output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse().unwrap())
        .collect();
    assert_eq!(
        actual,
        [
            size_of::<BasicInformation>(),
            align_of::<BasicInformation>(),
            usize::try_from(BASIC_BY_NAME).unwrap(),
            size_of::<i32>(),
            offset_of!(BasicInformation, file_id),
            offset_of!(BasicInformation, creation_time),
            offset_of!(BasicInformation, last_access_time),
            offset_of!(BasicInformation, last_write_time),
            offset_of!(BasicInformation, change_time),
            offset_of!(BasicInformation, allocation_size),
            offset_of!(BasicInformation, end_of_file),
            offset_of!(BasicInformation, attributes),
            offset_of!(BasicInformation, tag),
            offset_of!(BasicInformation, number_of_links),
            offset_of!(BasicInformation, device_type),
            offset_of!(BasicInformation, device_characteristics),
            offset_of!(BasicInformation, reserved),
            offset_of!(BasicInformation, volume_serial_number),
            offset_of!(BasicInformation, file_id_128)
        ]
    );
}
