use super::*;
use std::ffi::c_void;
use std::{cell::RefCell, collections::VecDeque};
use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;

type Response = Result<Vec<u32>, u32>;
type Step = (&'static str, Response);
#[derive(Default)]
struct State {
    steps: VecDeque<Step>,
    code: u32,
}
thread_local! { static STATE: RefCell<State> = RefCell::default(); }

fn invoke(name: &str) -> Response {
    STATE.with_borrow_mut(|state| {
        let (expected, result) = state.steps.pop_front().expect("unexpected query");
        assert_eq!(name, expected);
        if let Err(code) = result {
            state.code = code;
        }
        result
    })
}
fn by_name(_: &Path) -> Result<DirectoryAttributes, PathQueryError> {
    invoke("name")
        .map(|v| DirectoryAttributes {
            attributes: v[0],
            reparse_tag: v[1],
        })
        .map_err(|code| PathQueryError::Windows {
            operation: "name",
            code,
        })
}
fn directory(_: &Path) -> Result<Option<DirectoryAttributes>, PathQueryError> {
    invoke("directory")
        .map(|v| {
            if v.is_empty() {
                None
            } else {
                Some(DirectoryAttributes {
                    attributes: v[0],
                    reparse_tag: v[1],
                })
            }
        })
        .map_err(|code| PathQueryError::Windows {
            operation: "directory",
            code,
        })
}
unsafe extern "system" fn mock_open(
    name: PCWSTR,
    access: u32,
    share: u32,
    security: *const SECURITY_ATTRIBUTES,
    creation: u32,
    flags: u32,
    template: HANDLE,
) -> HANDLE {
    assert!(security.is_null() && template.is_null());
    assert_eq!(creation, OPEN_EXISTING);
    // SAFETY: production supplies a live terminated path.
    unsafe {
        assert_eq!(*name, u16::from(b'x'));
    }
    let follow = flags & FILE_FLAG_OPEN_REPARSE_POINT == 0;
    let console = access & GENERIC_READ != 0;
    assert_eq!(
        flags,
        FILE_FLAG_BACKUP_SEMANTICS
            | if follow {
                0
            } else {
                FILE_FLAG_OPEN_REPARSE_POINT
            }
    );
    assert_eq!(
        access,
        FILE_READ_ATTRIBUTES | if console { GENERIC_READ } else { 0 }
    );
    assert_eq!(
        share,
        if console {
            FILE_SHARE_READ | FILE_SHARE_WRITE
        } else {
            0
        }
    );
    match invoke(&format!("open:{follow}:{console}")) {
        Ok(_) => ptr::without_provenance_mut(1),
        Err(_) => INVALID_HANDLE_VALUE,
    }
}
unsafe extern "system" fn mock_close(handle: HANDLE) -> BOOL {
    assert_eq!(handle, ptr::without_provenance_mut(1));
    i32::from(invoke("close").is_ok())
}
unsafe extern "system" fn mock_last() -> u32 {
    STATE.with_borrow(|state| state.code)
}
unsafe extern "system" fn file_type(_: HANDLE) -> u32 {
    match invoke("type") {
        Ok(v) => {
            STATE.with_borrow_mut(|state| state.code = 0);
            v[0]
        }
        Err(_) => FILE_TYPE_UNKNOWN,
    }
}
unsafe extern "system" fn attributes(_: PCWSTR) -> u32 {
    invoke("attributes").map_or(u32::MAX, |v| v[0])
}
unsafe extern "system" fn information(_: HANDLE, output: *mut BY_HANDLE_FILE_INFORMATION) -> BOOL {
    match invoke("information") {
        Ok(v) => {
            // SAFETY: caller supplies aligned exclusive initialized SDK storage.
            unsafe {
                (*output).dwFileAttributes = v[0];
            }
            1
        }
        Err(_) => 0,
    }
}
unsafe extern "system" fn query(
    _: HANDLE,
    class: FILE_INFO_BY_HANDLE_CLASS,
    output: *mut c_void,
    size: u32,
) -> BOOL {
    let (name, expected_size) = match class {
        value if value == FileAttributeTagInfo => ("tag", size_of::<FILE_ATTRIBUTE_TAG_INFO>()),
        value if value == FileBasicInfo => ("basic", size_of::<FILE_BASIC_INFO>()),
        value if value == FileIdInfo => ("id", size_of::<FILE_ID_INFO>()),
        _ => panic!("unexpected SDK class"),
    };
    assert_eq!(size as usize, expected_size);
    match invoke(name) {
        Ok(v) => {
            if class == FileAttributeTagInfo {
                // SAFETY: class and extent match exclusive SDK output storage.
                unsafe {
                    *output.cast::<FILE_ATTRIBUTE_TAG_INFO>() = FILE_ATTRIBUTE_TAG_INFO {
                        FileAttributes: v[0],
                        ReparseTag: v[1],
                    };
                }
            }
            1
        }
        Err(_) => 0,
    }
}
const BASE: Api = Api {
    open: mock_open,
    close_file: mock_close,
    last_error: mock_last,
    ..WINDOWS
};
const MOCK: StatApi = StatApi {
    base: &BASE,
    query,
    information,
    file_type,
    attributes,
    by_name,
    directory,
};

fn yes(name: &'static str, values: &[u32]) -> Step {
    (name, Ok(values.to_vec()))
}
fn no(name: &'static str, code: u32) -> Step {
    (name, Err(code))
}
fn run(steps: Vec<Step>, expected: Result<[u32; 2], u32>) {
    STATE.set(State {
        steps: steps.into(),
        code: 0,
    });
    let result = stat_with(Path::new("x"), &MOCK)
        .map(|value| [value.attributes, value.reparse_tag])
        .map_err(|error| match error {
            PathQueryError::Windows { code, .. } => code,
            other => panic!("{other}"),
        });
    assert_eq!(result, expected);
    STATE.with_borrow(|state| assert!(state.steps.is_empty(), "remaining {:?}", state.steps));
}
fn prefix() -> Vec<Step> {
    vec![
        no("name", 50),
        yes("open:false:false", &[]),
        yes("type", &[FILE_TYPE_DISK]),
    ]
}
fn retry_prefix() -> Vec<Step> {
    let mut steps = prefix();
    steps.extend([
        yes("tag", &[FILE_ATTRIBUTE_REPARSE_POINT, 16]),
        no("close", 999),
    ]);
    steps
}

#[test]
fn fast_paths_open_retries_and_directory_failures_match_source_order() {
    for value in [
        [0, 0],
        [16, 0],
        [FILE_ATTRIBUTE_REPARSE_POINT, 0xa000_000c],
        [FILE_ATTRIBUTE_REPARSE_POINT, 0xa000_0003],
    ] {
        run(vec![yes("name", &value)], Ok(value));
    }
    for code in [2, 3, 21, 67] {
        run(vec![no("name", code)], Err(code));
    }
    for code in [
        0, 1, 4, 5, 20, 22, 32, 50, 53, 66, 68, 87, 123, 161, 206, 1920, 9999,
    ] {
        run(
            vec![no("name", code), no("open:false:false", 999)],
            Err(999),
        );
    }
    run(
        vec![
            yes("name", &[FILE_ATTRIBUTE_REPARSE_POINT, 16]),
            no("open:false:false", 1920),
        ],
        Err(1920),
    );
    for code in [5, 32] {
        for result in [
            Ok(vec![]),
            Ok(vec![FILE_ATTRIBUTE_REPARSE_POINT, 16]),
            Err(5),
            Err(8),
            Err(50),
            Err(123),
        ] {
            run(
                vec![
                    no("name", 50),
                    no("open:false:false", code),
                    ("directory", result),
                ],
                Err(code),
            );
        }
        for failure in [2, 3, 21, 67] {
            run(
                vec![
                    no("name", 50),
                    no("open:false:false", code),
                    no("directory", failure),
                ],
                Err(failure),
            );
        }
        for value in [[0, 0], [16, 0], [FILE_ATTRIBUTE_REPARSE_POINT, 0xa000_000c]] {
            run(
                vec![
                    no("name", 50),
                    no("open:false:false", code),
                    yes("directory", &value),
                ],
                Ok(value),
            );
        }
    }
    run(
        vec![
            no("name", 50),
            no("open:false:false", 87),
            no("open:false:true", 5),
        ],
        Err(87),
    );
    run(
        vec![
            no("name", 50),
            no("open:false:false", 87),
            yes("open:false:true", &[]),
            yes("type", &[2]),
            no("attributes", 3),
            yes("close", &[]),
        ],
        Ok([0, 0]),
    );
}

#[test]
fn invalid_input_does_not_reach_stat_queries() {
    fn invalid(_: &Path) -> Result<DirectoryAttributes, PathQueryError> {
        Err(PathQueryError::EmbeddedNul)
    }
    STATE.set(State::default());
    assert_eq!(
        slow(Path::new("x\0"), false, &MOCK),
        Err(PathQueryError::EmbeddedNul)
    );
    assert_eq!(
        stat_with(
            Path::new("x"),
            &StatApi {
                by_name: invalid,
                ..MOCK
            }
        ),
        Err(PathQueryError::EmbeddedNul)
    );
}

#[test]
fn handle_queries_metadata_gates_and_final_close_match_source_order() {
    for file_type in [0, 2, 3] {
        for close in [yes("close", &[]), no("close", 6)] {
            let expected = if close.1.is_ok() { Ok([0, 0]) } else { Err(6) };
            run(
                vec![
                    no("name", 50),
                    yes("open:false:false", &[]),
                    yes("type", &[file_type]),
                    yes("attributes", &[16]),
                    close,
                ],
                expected,
            );
        }
    }
    run(
        vec![
            no("name", 50),
            yes("open:false:false", &[]),
            no("type", 123),
            yes("close", &[]),
        ],
        Err(123),
    );
    run(
        vec![
            no("name", 50),
            yes("open:false:false", &[]),
            no("type", 123),
            no("close", 6),
        ],
        Err(6),
    );
    for tag in [
        yes("tag", &[0, 0]),
        yes("tag", &[FILE_ATTRIBUTE_REPARSE_POINT, 0xa000_000c]),
        no("tag", 1),
        no("tag", 50),
        no("tag", 87),
    ] {
        let expected_tag = if let Ok(ref value) = tag.1 {
            value[1]
        } else {
            0
        };
        for id in [yes("id", &[]), no("id", 5)] {
            let mut steps = prefix();
            steps.extend([
                tag.clone(),
                yes("information", &[1024]),
                yes("basic", &[]),
                id,
                yes("close", &[]),
            ]);
            run(steps, Ok([1024, expected_tag]));
        }
    }
    let mut steps = prefix();
    steps.extend([no("tag", 5), yes("close", &[])]);
    run(steps, Err(5));
    for failure in [1, 50, 87, 5] {
        for stage in ["information", "basic"] {
            let mut steps = prefix();
            steps.push(yes("tag", &[0, 0]));
            if stage == "basic" {
                steps.push(yes("information", &[16]));
            }
            steps.extend([no(stage, failure), yes("close", &[])]);
            run(steps, if failure == 5 { Err(5) } else { Ok([0, 0]) });
        }
    }
    let mut steps = prefix();
    steps.extend([
        yes("tag", &[0, 0]),
        yes("information", &[16]),
        yes("basic", &[]),
        yes("id", &[]),
        no("close", 6),
    ]);
    run(steps, Err(6));
}

#[test]
fn non_link_reparse_retry_and_unhandled_tag_rules_match_source_order() {
    let mut steps = retry_prefix();
    steps.extend([
        yes("open:true:false", &[]),
        yes("type", &[FILE_TYPE_DISK]),
        yes("information", &[16]),
        yes("basic", &[]),
        yes("id", &[]),
        yes("close", &[]),
    ]);
    run(steps, Ok([16, 0]));
    for code in [5, 32] {
        let mut steps = retry_prefix();
        steps.extend([
            no("open:true:false", code),
            yes("directory", &[FILE_ATTRIBUTE_REPARSE_POINT, 0xa000_000c]),
        ]);
        run(steps, Err(code));
    }
    let mut steps = retry_prefix();
    steps.extend([
        no("open:true:false", 87),
        yes("open:true:true", &[]),
        yes("type", &[2]),
        yes("attributes", &[0]),
        yes("close", &[]),
    ]);
    run(steps, Ok([0, 0]));
    let mut steps = retry_prefix();
    steps.extend([no("open:true:false", 1920), no("open:false:false", 5)]);
    run(steps, Err(1920));
    for tag in [0xa000_000c, 16] {
        let mut steps = retry_prefix();
        steps.extend([
            no("open:true:false", 1920),
            yes("open:false:false", &[]),
            yes("type", &[FILE_TYPE_DISK]),
            yes("tag", &[FILE_ATTRIBUTE_REPARSE_POINT, tag]),
        ]);
        if tag == 16 {
            steps.extend([
                yes("information", &[1024]),
                yes("basic", &[]),
                yes("id", &[]),
            ]);
        }
        steps.push(yes("close", &[]));
        run(
            steps,
            if tag == 16 {
                Ok([1024, tag])
            } else {
                Err(1920)
            },
        );
    }
}
