use super::*;
use std::{cell::RefCell, collections::VecDeque};

thread_local! {
    static STEPS: RefCell<VecDeque<(u32,u32)>> = const { RefCell::new(VecDeque::new()) };
}
unsafe extern "system" fn query(capacity: u32, pointer: PWSTR) -> u32 {
    let (expected, result) = STEPS.with_borrow_mut(|s| s.pop_front().unwrap());
    assert_eq!(capacity, expected);
    if result != 0 && result < capacity {
        // SAFETY: test query obeys caller capacity, preserving unpaired UTF-16.
        unsafe { std::slice::from_raw_parts_mut(pointer, result as usize).fill(0xd800) };
    }
    result
}
unsafe extern "system" fn error() -> u32 {
    5
}
fn fail_allocation(_: usize) -> Result<Vec<u16>, PathQueryError> {
    Err(PathQueryError::Allocation)
}

#[test]
fn growth_errors_and_utf16_are_preserved() {
    let api = Api {
        last_error: error,
        ..WINDOWS
    };
    for steps in [vec![(256, 3)], vec![(256, 256), (256, 2048), (2048, 3)]] {
        STEPS.with_borrow_mut(|s| *s = steps.into());
        assert_eq!(
            query_with(&api, query).unwrap().into_os_string(),
            OsString::from_wide(&[0xd800; 3])
        );
        STEPS.with_borrow(|s| assert!(s.is_empty()));
    }
    for steps in [vec![(256, 0)], vec![(256, 2048), (2048, 0)]] {
        STEPS.with_borrow_mut(|s| *s = steps.into());
        assert_eq!(
            query_with(&api, query),
            Err(PathQueryError::Windows {
                operation: "GetCurrentDirectoryW",
                code: 5
            })
        );
        STEPS.with_borrow(|s| assert!(s.is_empty()));
    }
    STEPS.with_borrow_mut(|s| *s = vec![(256, 2048)].into());
    let failed = Api {
        allocate: fail_allocation,
        ..api
    };
    assert_eq!(query_with(&failed, query), Err(PathQueryError::Allocation));
    STEPS.with_borrow(|s| assert!(s.is_empty()));
}
