use super::*;
use std::cell::RefCell;
type BoolResult = Result<bool, PathQueryError>;
type AttributesResult = Result<DirectoryAttributes, PathQueryError>;
struct State {
    name: BoolResult,
    open: BoolResult,
    stat: AttributesResult,
    calls: Vec<&'static str>,
}
thread_local! { static STATE: RefCell<Option<State>> = const { RefCell::new(None) }; }
fn name(_: &Path) -> BoolResult {
    STATE.with_borrow_mut(|s| {
        let s = s.as_mut().unwrap();
        s.calls.push("name");
        s.name.clone()
    })
}
fn open(_: &Path) -> BoolResult {
    STATE.with_borrow_mut(|s| {
        let s = s.as_mut().unwrap();
        s.calls.push("open");
        s.open.clone()
    })
}
fn stat(_: &Path) -> AttributesResult {
    STATE.with_borrow_mut(|s| {
        let s = s.as_mut().unwrap();
        s.calls.push("stat");
        s.stat.clone()
    })
}
const MOCK: Queries = Queries {
    by_name: name,
    by_open: open,
    lstat: stat,
};
fn error(operation: &'static str, code: u32) -> PathQueryError {
    PathQueryError::Windows { operation, code }
}
fn run(name: BoolResult, open: BoolResult, stat: AttributesResult, expected: bool, calls: &[&str]) {
    STATE.set(Some(State {
        name,
        open,
        stat,
        calls: vec![],
    }));
    assert_eq!(classify(Path::new("input"), &MOCK), expected);
    STATE.with_borrow(|s| assert_eq!(s.as_ref().unwrap().calls, calls));
}

#[test]
fn classification_uses_exact_error_lists_and_failure_stages() {
    let unused = Err(error("unused", 9999));
    for value in [false, true] {
        run(
            Ok(value),
            unused.clone(),
            Err(error("unused", 9999)),
            value,
            &["name"],
        );
    }
    for code in [2, 3, 21, 53, 67, 123, 161, 206] {
        run(
            Err(error("name", code)),
            unused.clone(),
            Err(error("unused", 9999)),
            false,
            &["name"],
        );
    }
    for invalid in [
        PathQueryError::EmbeddedNul,
        PathQueryError::Allocation,
        PathQueryError::InputTooLong,
        PathQueryError::NotSymbolicLink,
    ] {
        run(
            Err(invalid.clone()),
            unused.clone(),
            Err(error("unused", 9999)),
            false,
            &["name"],
        );
        run(
            Err(error("name", 50)),
            Err(invalid),
            Err(error("unused", 9999)),
            false,
            &["name", "open"],
        );
    }
    for code in [
        0, 1, 4, 5, 20, 22, 32, 50, 65, 66, 68, 87, 1005, 1920, 1921, 4390, 9999,
    ] {
        for value in [false, true] {
            run(
                Err(error("name", code)),
                Ok(value),
                Err(error("unused", 9999)),
                value,
                &["name", "open"],
            );
        }
    }
    for code in [5, 32, 87, 1920] {
        for (attributes, tag, expected) in [
            (0, IO_REPARSE_TAG_SYMLINK, false),
            (FILE_ATTRIBUTE_REPARSE_POINT, 0, false),
            (FILE_ATTRIBUTE_REPARSE_POINT, 0xa000_0003, false),
            (FILE_ATTRIBUTE_REPARSE_POINT, IO_REPARSE_TAG_SYMLINK, true),
        ] {
            run(
                Err(error("name", 50)),
                Err(error("CreateFileW", code)),
                Ok(DirectoryAttributes {
                    attributes,
                    reparse_tag: tag,
                }),
                expected,
                &["name", "open", "stat"],
            );
        }
        run(
            Err(error("name", 50)),
            Err(error("CreateFileW", code)),
            Err(error("stat", 2)),
            false,
            &["name", "open", "stat"],
        );
        run(
            Err(error("name", 50)),
            Err(error("GetFileInformationByHandleEx", code)),
            Err(error("unused", 9999)),
            false,
            &["name", "open"],
        );
    }
    for code in [0, 1, 2, 3, 21, 50, 123, 1921, 9999] {
        run(
            Err(error("name", 50)),
            Err(error("CreateFileW", code)),
            Err(error("unused", 9999)),
            false,
            &["name", "open"],
        );
    }
}
