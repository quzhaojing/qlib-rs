use super::*;
use std::{cell::RefCell, os::windows::ffi::OsStringExt};

#[derive(Default)]
struct State {
    open_fails: bool,
    find_fails: bool,
    lengths: Vec<u32>,
    capacities: Vec<u32>,
    closes: [usize; 2],
    code: u32,
    open_parameters: Vec<u32>,
    ioctl_fails: bool,
    returned: u32,
    map_results: Vec<i32>,
    map_inputs: Vec<Vec<u16>>,
    tag: Option<[u32; 2]>,
    tag_calls: usize,
    find_inputs: Vec<Vec<u16>>,
    directory_tag: [u32; 2],
}

unsafe extern "system" fn tag_query(
    handle: HANDLE,
    class: FILE_INFO_BY_HANDLE_CLASS,
    buffer: *mut c_void,
    size: u32,
) -> BOOL {
    assert_eq!(handle, ptr::without_provenance_mut(1));
    assert_eq!(class, FileAttributeTagInfo);
    assert_eq!(
        usize::try_from(size).unwrap(),
        size_of::<FILE_ATTRIBUTE_TAG_INFO>()
    );
    STATE.with_borrow_mut(|state| {
        state.tag_calls += 1;
        let Some(tag) = state.tag else {
            return 0;
        };
        // SAFETY: the production adapter supplies exclusive aligned SDK storage.
        unsafe {
            *buffer.cast::<FILE_ATTRIBUTE_TAG_INFO>() = FILE_ATTRIBUTE_TAG_INFO {
                FileAttributes: tag[0],
                ReparseTag: tag[1],
            };
        }
        1
    })
}

#[test]
fn symbolic_tag_probe_matches_open_policy_and_preserves_failure_stage() {
    for (attributes, tag, expected) in [
        (0, IO_REPARSE_TAG_SYMLINK, false),
        (FILE_ATTRIBUTE_REPARSE_POINT, 0, false),
        (FILE_ATTRIBUTE_REPARSE_POINT, 0xa000_0003, false),
        (FILE_ATTRIBUTE_REPARSE_POINT, 0xa000_001d, false),
        (FILE_ATTRIBUTE_REPARSE_POINT, IO_REPARSE_TAG_SYMLINK, true),
        (
            FILE_ATTRIBUTE_REPARSE_POINT | 16,
            IO_REPARSE_TAG_SYMLINK,
            true,
        ),
    ] {
        STATE.set(State {
            tag: Some([attributes, tag]),
            ..State::default()
        });
        assert_eq!(
            symbolic_link_by_open_with(Path::new("link"), &MOCK, tag_query),
            Ok(expected)
        );
        STATE.with_borrow(|state| {
            assert_eq!(
                state.open_parameters,
                [
                    FILE_READ_ATTRIBUTES,
                    0,
                    OPEN_EXISTING,
                    FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT
                ]
            );
            assert_eq!(state.closes, [1, 0]);
            assert_eq!(state.tag_calls, 1);
        });
    }
    for code in [2, 3, 5, 32, 87, 1920, 9999] {
        for open_fails in [false, true] {
            STATE.set(State {
                open_fails,
                code,
                ..State::default()
            });
            assert_eq!(
                symbolic_link_by_open_with(Path::new("link"), &MOCK, tag_query),
                Err(PathQueryError::Windows {
                    operation: if open_fails {
                        "CreateFileW"
                    } else {
                        "GetFileInformationByHandleEx"
                    },
                    code
                })
            );
            STATE.with_borrow(|state| {
                assert_eq!(state.closes, [usize::from(!open_fails), 0]);
                assert_eq!(state.tag_calls, usize::from(!open_fails));
            });
        }
    }
    STATE.set(State::default());
    for invalid in ["a\0b", "a\0"] {
        assert_eq!(
            symbolic_link_by_open_with(Path::new(invalid), &MOCK, tag_query),
            Err(PathQueryError::EmbeddedNul)
        );
    }
    STATE.with_borrow(|state| {
        assert!(state.open_parameters.is_empty());
        assert_eq!(state.closes, [0, 0]);
        assert_eq!(state.tag_calls, 0);
    });
}

thread_local! { static STATE: RefCell<State> = RefCell::default(); }

unsafe extern "system" fn open(
    _: PCWSTR,
    access: u32,
    share: u32,
    _: *const SECURITY_ATTRIBUTES,
    creation: u32,
    flags: u32,
    _: HANDLE,
) -> HANDLE {
    STATE.with_borrow_mut(|state| {
        state.open_parameters = vec![access, share, creation, flags];
        if state.open_fails {
            INVALID_HANDLE_VALUE
        } else {
            ptr::without_provenance_mut(1)
        }
    })
}

unsafe extern "system" fn query(_: HANDLE, buffer: PWSTR, capacity: u32, _: u32) -> u32 {
    STATE.with_borrow_mut(|state| {
        let index = state.capacities.len();
        state.capacities.push(capacity);
        let length = state.lengths[index];
        if length == 3 && capacity > 3 {
            // SAFETY: The adapter supplies at least capacity initialized units;
            // this successful mock response writes exactly three units.
            unsafe { ptr::copy_nonoverlapping([67, 58, 0xd800].as_ptr(), buffer, 3) };
        }
        length
    })
}

unsafe extern "system" fn find(name: PCWSTR, data: *mut WIN32_FIND_DATAW) -> HANDLE {
    STATE.with_borrow_mut(|state| {
        // SAFETY: all native query adapters supply a live terminated path buffer.
        state.find_inputs.push(
            unsafe { widestring::U16CStr::from_ptr_str(name) }
                .as_slice()
                .to_vec(),
        );
        if state.find_fails {
            INVALID_HANDLE_VALUE
        } else {
            // SAFETY: The caller passes exclusive, initialized SDK data storage.
            unsafe {
                (&mut (*data).cFileName)[..3].copy_from_slice(&[65, 0xd800, 0]);
                (*data).dwFileAttributes = state.directory_tag[0];
                (*data).dwReserved0 = state.directory_tag[1];
            }
            ptr::without_provenance_mut(2)
        }
    })
}

unsafe extern "system" fn close_file(_: HANDLE) -> BOOL {
    STATE.with_borrow_mut(|state| {
        state.closes[0] += 1;
        state.code = 999;
    });
    1
}

unsafe extern "system" fn close_search(_: HANDLE) -> BOOL {
    STATE.with_borrow_mut(|state| {
        state.closes[1] += 1;
        state.code = 999;
    });
    1
}

unsafe extern "system" fn last() -> u32 {
    STATE.with_borrow(|state| state.code)
}

fn fail_allocation(_: usize) -> Result<Vec<u16>, PathQueryError> {
    Err(PathQueryError::Allocation)
}

const MOCK: Api = Api {
    open,
    query,
    find,
    close_file,
    close_search,
    last_error: last,
    allocate,
    native_length,
};

#[test]
fn directory_fallback_preserves_trim_search_error_and_tag_contract() {
    for (input, expected) in [
        ("", Some("")),
        ("a", Some("a")),
        ("a/", None),
        ("a\\\\", None),
        ("/", None),
        ("\\", None),
        ("//", None),
        ("/\\/", None),
        ("C:/", None),
        ("C:\\\\", None),
        ("1:/", None),
        ("ab/", Some("ab")),
        ("ab\\/\\", Some("ab")),
        ("C:", Some("C:")),
        ("/a/", Some("/a")),
        ("//server/share/", Some("//server/share")),
        ("C:/dir/*.txt", Some("C:/dir/*.txt")),
        ("C:/dir/?/", Some("C:/dir/?")),
    ] {
        STATE.set(State {
            directory_tag: [FILE_ATTRIBUTE_REPARSE_POINT, IO_REPARSE_TAG_SYMLINK],
            ..State::default()
        });
        let actual = directory_attributes_with(Path::new(input), &MOCK).unwrap();
        assert_eq!(
            actual,
            expected.map(|_| DirectoryAttributes {
                attributes: FILE_ATTRIBUTE_REPARSE_POINT,
                reparse_tag: IO_REPARSE_TAG_SYMLINK,
            }),
            "{input:?}"
        );
        STATE.with_borrow(|state| {
            let expected_inputs: Vec<Vec<u16>> = expected
                .into_iter()
                .map(|s| s.encode_utf16().collect())
                .collect();
            assert_eq!(state.find_inputs, expected_inputs);
            assert_eq!(state.closes, [0, usize::from(expected.is_some())]);
        });
    }
    STATE.set(State {
        directory_tag: [16, IO_REPARSE_TAG_SYMLINK],
        ..State::default()
    });
    assert_eq!(
        directory_attributes_with(Path::new("dir"), &MOCK),
        Ok(Some(DirectoryAttributes {
            attributes: 16,
            reparse_tag: 0,
        }))
    );
    let input = OsString::from_wide(&[97, 0xd800, 92]);
    assert!(
        directory_attributes_with(Path::new(&input), &MOCK)
            .unwrap()
            .is_some()
    );
    STATE.with_borrow(|state| assert_eq!(state.find_inputs.last().unwrap(), &[97, 0xd800]));
    for code in [2, 3, 5, 21, 32, 67, 123, 9999] {
        STATE.set(State {
            find_fails: true,
            code,
            ..State::default()
        });
        assert_eq!(
            directory_attributes_with(Path::new("missing/"), &MOCK),
            Err(PathQueryError::Windows {
                operation: "FindFirstFileW",
                code,
            })
        );
        STATE.with_borrow(|state| assert_eq!(state.closes, [0, 0]));
    }
    STATE.set(State::default());
    assert_eq!(
        directory_attributes_with(Path::new("a\0/"), &MOCK),
        Err(PathQueryError::EmbeddedNul)
    );
    let failing = Api {
        allocate: fail_allocation,
        ..MOCK
    };
    assert_eq!(
        directory_attributes_with(Path::new("ab/"), &failing),
        Err(PathQueryError::Windows {
            operation: "attributes_from_dir",
            code: 8,
        })
    );
    STATE.with_borrow(|state| assert!(state.find_inputs.is_empty()));
}

#[test]
fn query_retries_and_all_acquired_handles_close_once_even_on_failures() {
    STATE.set(State {
        lengths: vec![300, 400, 3],
        ..State::default()
    });
    let result = final_path_with(Path::new("x"), &MOCK).unwrap();
    assert_eq!(result.as_os_str(), OsString::from_wide(&[67, 58, 0xd800]));
    STATE.with_borrow(|state| {
        assert_eq!(state.capacities, [260, 300, 400]);
        assert_eq!(state.closes, [1, 0]);
        assert_eq!(
            state.open_parameters,
            [0, 0, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS]
        );
    });
    for (open_fails, expected_operation, closed) in [
        (true, "CreateFileW", 0),
        (false, "GetFinalPathNameByHandleW", 1),
    ] {
        STATE.set(State {
            open_fails,
            lengths: vec![0],
            code: 32,
            ..State::default()
        });
        assert_eq!(
            final_path_with(Path::new("x"), &MOCK),
            Err(PathQueryError::Windows {
                operation: expected_operation,
                code: 32
            })
        );
        STATE.with_borrow(|state| assert_eq!(state.closes, [closed, 0]));
    }
    STATE.set(State::default());
    let failure = Api {
        allocate: fail_allocation,
        ..MOCK
    };
    assert_eq!(
        final_path_with(Path::new("x"), &failure),
        Err(PathQueryError::Allocation)
    );
    STATE.with_borrow(|state| assert_eq!(state.closes, [1, 0]));
    assert_eq!(allocate(usize::MAX), Err(PathQueryError::Allocation));
    assert_eq!(allocate(0).unwrap(), [0_u16; 0]);
}

#[test]
fn search_closes_only_acquired_handles_and_nul_never_reaches_native_api() {
    STATE.set(State::default());
    assert_eq!(
        find_name_with(Path::new("x"), &MOCK).unwrap(),
        OsString::from_wide(&[65, 0xd800])
    );
    STATE.with_borrow(|state| assert_eq!(state.closes, [0, 1]));
    STATE.set(State {
        find_fails: true,
        code: 2,
        ..State::default()
    });
    assert_eq!(
        find_name_with(Path::new("x"), &MOCK),
        Err(PathQueryError::Windows {
            operation: "FindFirstFileW",
            code: 2
        })
    );
    STATE.with_borrow(|state| assert_eq!(state.closes, [0, 0]));
    assert_eq!(
        find_name_with(Path::new("x\0y"), &MOCK),
        Err(PathQueryError::EmbeddedNul)
    );
    assert_eq!(
        final_path_with(Path::new("x\0y"), &MOCK),
        Err(PathQueryError::EmbeddedNul)
    );
}

#[test]
fn real_public_queries_preserve_file_names_and_missing_file_errors() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("MiXeD.txt");
    std::fs::write(&file, "content").unwrap();
    assert_eq!(find_name(&file).unwrap(), "MiXeD.txt");
    assert_eq!(
        final_path(&file).unwrap(),
        std::fs::canonicalize(&file).unwrap()
    );
    assert!(matches!(
        final_path(&directory.path().join("missing")),
        Err(PathQueryError::Windows { code: 2, .. })
    ));
    assert!(matches!(
        find_name(&directory.path().join("missing")),
        Err(PathQueryError::Windows { code: 2, .. })
    ));
    assert_eq!(
        final_path(Path::new("\0")),
        Err(PathQueryError::EmbeddedNul)
    );
    assert_eq!(find_name(Path::new("\0")), Err(PathQueryError::EmbeddedNul));
    assert_eq!(
        PathQueryError::EmbeddedNul.to_string(),
        "embedded null character"
    );
    assert_eq!(
        PathQueryError::Allocation.to_string(),
        "could not allocate the native path buffer"
    );
    assert_eq!(
        PathQueryError::Windows {
            operation: "CreateFileW",
            code: 32
        }
        .to_string(),
        "CreateFileW failed with Windows error 32"
    );
}

unsafe extern "system" fn ioctl(
    _: HANDLE,
    code: u32,
    input: *const c_void,
    input_size: u32,
    output: *mut c_void,
    output_size: u32,
    returned: *mut u32,
    overlapped: *mut OVERLAPPED,
) -> BOOL {
    assert_eq!(code, FSCTL_GET_REPARSE_POINT);
    assert!(input.is_null());
    assert_eq!(input_size, 0);
    assert_eq!(output_size, MAXIMUM_REPARSE_DATA_BUFFER_SIZE);
    assert!(overlapped.is_null());
    STATE.with_borrow(|state| {
        // SAFETY: Synchronous caller supplies writable returned-count storage and
        // a sufficiently large byte buffer. No uninitialized memory is read.
        unsafe {
            *returned = state.returned;
            if state.returned == 22 {
                let packet = [
                    12, 0, 0, 160, 14, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 0,
                ];
                ptr::copy_nonoverlapping(packet.as_ptr(), output.cast::<u8>(), packet.len());
            }
        }
        i32::from(!state.ioctl_fails)
    })
}

#[test]
fn readlink_preserves_native_policy_and_closes_on_every_exit() {
    for (open_fails, ioctl_fails, returned, expected, closed) in [
        (false, false, 22, Ok(PathBuf::from("A")), 1),
        (
            true,
            false,
            0,
            Err(PathQueryError::Windows {
                operation: "CreateFileW",
                code: 32,
            }),
            0,
        ),
        (
            false,
            true,
            0,
            Err(PathQueryError::Windows {
                operation: "DeviceIoControl",
                code: 32,
            }),
            1,
        ),
        (false, false, 0, Err(crate::reparse::INVALID_DATA), 1),
        (
            false,
            false,
            MAXIMUM_REPARSE_DATA_BUFFER_SIZE + 1,
            Err(crate::reparse::INVALID_DATA),
            1,
        ),
        (false, false, 8, Err(PathQueryError::NotSymbolicLink), 1),
    ] {
        STATE.set(State {
            open_fails,
            ioctl_fails,
            returned,
            code: 32,
            ..State::default()
        });
        assert_eq!(read_link_with(Path::new("link"), &MOCK, ioctl), expected);
        STATE.with_borrow(|state| {
            assert_eq!(state.closes, [closed, 0]);
            assert_eq!(
                state.open_parameters,
                [
                    0,
                    0,
                    OPEN_EXISTING,
                    FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT
                ]
            );
        });
    }
    STATE.set(State::default());
    assert_eq!(
        read_link_with(Path::new("link\0"), &MOCK, ioctl),
        Err(PathQueryError::EmbeddedNul)
    );
    STATE.with_borrow(|state| {
        assert_eq!(state.closes, [0, 0]);
        assert!(state.open_parameters.is_empty());
    });
}

unsafe extern "system" fn map_case(
    locale: PCWSTR,
    flags: u32,
    input: PCWSTR,
    length: i32,
    output: PWSTR,
    capacity: i32,
    version: *const NLSVERSIONINFO,
    reserved: *const c_void,
    sort_handle: isize,
) -> i32 {
    assert_eq!(flags, LCMAP_LOWERCASE);
    assert!(version.is_null());
    assert!(reserved.is_null());
    assert_eq!(sort_handle, 0);
    // SAFETY: The caller supplies a terminated locale and explicit input length.
    assert_eq!(unsafe { *locale }, 0);
    let input = unsafe { std::slice::from_raw_parts(input, usize::try_from(length).unwrap()) };
    STATE.with_borrow_mut(|state| {
        let index = state.map_inputs.len();
        state.map_inputs.push(input.to_vec());
        let result = state.map_results[index];
        if output.is_null() {
            assert_eq!(capacity, 0);
        } else if result > 0 {
            assert!(capacity >= result);
            assert_eq!(result, 3);
            // SAFETY: Output has at least three writable units per checked capacity.
            unsafe { ptr::copy_nonoverlapping([97, 0, 0xd800].as_ptr(), output, 3) };
        }
        result
    })
}

#[test]
fn case_mapping_preserves_explicit_lengths_and_all_errors() {
    assert_eq!(native_length(0), Ok(0));
    assert_eq!(native_length(i32::MAX as usize), Ok(i32::MAX));
    assert_eq!(
        native_length(i32::MAX as usize + 1),
        Err(PathQueryError::InputTooLong)
    );
    assert_eq!(
        PathQueryError::InputTooLong.to_string(),
        "input string is too long"
    );
    STATE.set(State::default());
    assert_eq!(
        normalize_case_with(Path::new(""), &MOCK, map_case).unwrap(),
        ""
    );
    STATE.with_borrow(|state| assert!(state.map_inputs.is_empty()));
    for (results, expected) in [
        (vec![5, 3], Ok(OsString::from_wide(&[97, 0, 0xd800]))),
        (
            vec![0],
            Err(PathQueryError::Windows {
                operation: "LCMapStringEx",
                code: 87,
            }),
        ),
        (
            vec![3, 0],
            Err(PathQueryError::Windows {
                operation: "LCMapStringEx",
                code: 87,
            }),
        ),
    ] {
        let calls = results.len();
        STATE.set(State {
            map_results: results,
            code: 87,
            ..State::default()
        });
        assert_eq!(
            normalize_case_with(Path::new("A/\0"), &MOCK, map_case),
            expected
        );
        STATE.with_borrow(|state| assert_eq!(state.map_inputs, vec![vec![65, 92, 0]; calls]));
    }
    STATE.set(State {
        map_results: vec![3],
        ..State::default()
    });
    let failure = Api {
        allocate: fail_allocation,
        ..MOCK
    };
    assert_eq!(
        normalize_case_with(Path::new("A"), &failure, map_case),
        Err(PathQueryError::Allocation)
    );
    STATE.with_borrow(|state| assert_eq!(state.map_inputs.len(), 1));
    STATE.set(State::default());
    let reject_length = Api {
        native_length: |_| Err(PathQueryError::InputTooLong),
        ..MOCK
    };
    assert_eq!(
        normalize_case_with(Path::new("A"), &reject_length, map_case),
        Err(PathQueryError::InputTooLong)
    );
    STATE.with_borrow(|state| assert!(state.map_inputs.is_empty()));
}
