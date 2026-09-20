use super::*;
use serde_json::Value;
use std::{
    cell::RefCell,
    error::Error,
    ffi::CString,
    path::PathBuf,
    process::Command,
    sync::{Arc, Barrier},
};

#[derive(Default)]
struct FakeState {
    mode: c_int,
    fail_mode: bool,
    fail_acquire: bool,
    null_lookup: usize,
    lookups: usize,
    invalid_wide: usize,
    events: Vec<String>,
}

thread_local! {
    static STATE: RefCell<FakeState> = RefCell::new(FakeState { mode: 2, ..FakeState::default() });
    static GOOD: Lconv = const { raw(&[46, 0], &[44, 0]) };
    static BAD_DECIMAL: Lconv = const { raw(&[0xd800, 0], &[44, 0]) };
    static BAD_SEPARATOR: Lconv = const { raw(&[46, 0], &[0xd800, 0]) };
}

const fn raw(decimal: &'static [u16], separator: &'static [u16]) -> Lconv {
    let mut result = Lconv {
        narrow: [c"".as_ptr(); 10],
        flags: [0; 8],
        wide: [[0_u16].as_ptr(); 8],
    };
    result.narrow[2] = c"\x03\x02".as_ptr();
    result.wide[0] = decimal.as_ptr();
    result.wide[1] = separator.as_ptr();
    result
}

unsafe extern "C" fn fake_locale() -> *const Lconv {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.lookups += 1;
        state.events.push("lookup".into());
        if state.null_lookup == state.lookups {
            return std::ptr::null();
        }
        match state.invalid_wide {
            1 => BAD_DECIMAL.with(std::ptr::from_ref),
            2 => BAD_SEPARATOR.with(std::ptr::from_ref),
            _ => GOOD.with(std::ptr::from_ref),
        }
    })
}

unsafe extern "C" fn fake_config(flag: c_int) -> c_int {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.events.push(format!("config:{flag}"));
        if state.fail_mode {
            return -1;
        }
        let previous = state.mode;
        state.mode = flag;
        previous
    })
}

unsafe extern "C" fn fake_acquire() -> *mut c_void {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.events.push("acquire".into());
        if state.fail_acquire {
            std::ptr::null_mut()
        } else {
            NonNull::<c_void>::dangling().as_ptr()
        }
    })
}

unsafe extern "C" fn fake_free(pointer: *mut c_void) {
    assert_eq!(pointer, NonNull::<c_void>::dangling().as_ptr());
    STATE.with(|state| state.borrow_mut().events.push("free".into()));
}

const FAKE: Crt = Crt {
    locale: fake_locale,
    config: fake_config,
    acquire: fake_acquire,
    free: fake_free,
};

fn run_fake(
    configure: impl FnOnce(&mut FakeState),
) -> (Result<NumericLocale, LocaleError>, Vec<String>, c_int) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        *state = FakeState {
            mode: 2,
            ..FakeState::default()
        };
        configure(&mut state);
    });
    let result = snapshot(&FAKE);
    STATE.with(|state| {
        let state = state.borrow();
        (result, state.events.clone(), state.mode)
    })
}

#[test]
fn owned_snapshots_restore_both_thread_modes_after_copy() {
    for initial in [1, 2] {
        let (result, events, mode) = run_fake(|state| state.mode = initial);
        assert_eq!(
            result.unwrap(),
            NumericLocale {
                decimal: ".".into(),
                separator: ",".into(),
                grouping: vec![3, 2, 0],
            }
        );
        assert_eq!(mode, initial);
        assert_eq!(
            events,
            [
                "lookup",
                "config:1",
                "acquire",
                "lookup",
                "free",
                &format!("config:{initial}")
            ]
        );
    }
}

#[test]
fn acquisition_failures_preserve_cleanup_and_do_not_read_invalid_pointers() {
    for (stage, expected) in [
        (1, vec!["lookup"]),
        (2, vec!["lookup", "config:1"]),
        (3, vec!["lookup", "config:1", "acquire", "config:2"]),
        (
            4,
            vec![
                "lookup", "config:1", "acquire", "lookup", "free", "config:2",
            ],
        ),
    ] {
        let (result, events, mode) = run_fake(|state| match stage {
            1 => state.null_lookup = 1,
            2 => state.fail_mode = true,
            3 => state.fail_acquire = true,
            _ => state.null_lookup = 2,
        });
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            match stage {
                1 => "UCRT locale data unavailable at refresh",
                2 => "UCRT returned invalid thread locale mode -1",
                3 => "UCRT locale data unavailable at acquire",
                _ => "UCRT locale data unavailable at snapshot",
            }
        );
        assert!(error.source().is_none());
        assert_eq!(mode, 2);
        assert_eq!(events, expected);
    }
}

#[test]
fn invalid_utf16_is_reported_without_lossy_replacement_and_releases_both_guards() {
    for (index, field) in [(1, "decimal"), (2, "separator")] {
        let (result, events, mode) = run_fake(|state| state.invalid_wide = index);
        let error = result.unwrap_err();
        assert!(
            matches!(&error, LocaleError::InvalidUtf16 { field: actual, .. } if *actual == field)
        );
        assert!(
            error
                .to_string()
                .starts_with(&format!("UCRT {field} is not valid UTF-16:"))
        );
        assert!(error.source().is_some());
        assert_eq!(mode, 2);
        assert_eq!(
            events,
            [
                "lookup", "config:1", "acquire", "lookup", "free", "config:2"
            ]
        );
    }
}

#[test]
fn sdk_layout_is_explicit_on_the_supported_host() {
    assert_eq!(size_of::<Lconv>(), 18 * size_of::<usize>() + 8);
    assert_eq!(align_of::<Lconv>(), align_of::<usize>());
    assert_eq!(
        std::mem::offset_of!(Lconv, wide),
        10 * size_of::<usize>() + 8
    );
}

unsafe extern "C" {
    fn setlocale(category: c_int, name: *const c_char) -> *const c_char;
}

fn set_numeric(name: &str) {
    let name = CString::new(name).unwrap();
    // SAFETY: LC_NUMERIC is 4 and the C string lives through the call. Only
    // native_process_contract's isolated child process invokes this helper.
    assert!(!unsafe { setlocale(4, name.as_ptr()) }.is_null());
}

fn thread_mode() -> c_int {
    // SAFETY: Zero is the documented query-only flag.
    unsafe { _configthreadlocale(0) }
}

#[test]
fn native_process_contract() {
    if std::env::var_os("QLIB_LOCALE_TEST_CHILD").is_none() {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "windows::tests::native_process_contract",
                "--nocapture",
            ])
            .env("QLIB_LOCALE_TEST_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    let oracle = Command::new("python")
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/numeric_oracle.py"))
        .output()
        .unwrap();
    assert!(
        oracle.status.success(),
        "{}",
        String::from_utf8_lossy(&oracle.stderr)
    );
    let oracle: Vec<Value> = serde_json::from_slice(&oracle.stdout).unwrap();
    assert_eq!(oracle.len(), 10);
    let cases: Vec<(String, NumericLocale)> = oracle
        .into_iter()
        .map(|case| {
            let mut grouping: Vec<u8> = case["grouping"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| u8::try_from(n.as_u64().unwrap()).unwrap())
                .collect();
            if grouping.last() != Some(&0) {
                grouping.push(0);
            }
            (
                case["name"].as_str().unwrap().into(),
                NumericLocale {
                    decimal: case["decimal"].as_str().unwrap().into(),
                    separator: case["separator"].as_str().unwrap().into(),
                    grouping,
                },
            )
        })
        .collect();
    assert_eq!(thread_mode(), 2);
    for (name, expected) in &cases {
        set_numeric(name);
        assert_eq!(&current_numeric_locale().unwrap(), expected);
        assert_eq!(thread_mode(), 2);
    }
    // A previously warm thread must see another thread's later global update.
    set_numeric("en-US");
    assert_eq!(current_numeric_locale().unwrap().decimal, ".");
    std::thread::spawn(|| set_numeric("de-DE")).join().unwrap();
    assert_eq!(current_numeric_locale().unwrap().decimal, ",");
    for (name, expected) in cases.clone() {
        std::thread::spawn(move || {
            // SAFETY: The documented flag affects only this isolated worker.
            let previous = unsafe { _configthreadlocale(1) };
            set_numeric(&name);
            std::thread::spawn(|| set_numeric("C")).join().unwrap();
            assert_eq!(current_numeric_locale().unwrap(), expected);
            assert_eq!(thread_mode(), 1);
            // SAFETY: Restore the prior valid setting on the same thread.
            unsafe { _configthreadlocale(previous) };
            assert_eq!(current_numeric_locale().unwrap().decimal, ".");
        })
        .join()
        .unwrap();
    }
    let cases = Arc::new(cases);
    let barrier = Arc::new(Barrier::new(5));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let cases = cases.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..5_000 {
                    let actual = current_numeric_locale().unwrap();
                    assert!(cases.iter().any(|(_, expected)| expected == &actual));
                    assert_eq!(thread_mode(), 2);
                }
            })
        })
        .collect();
    barrier.wait();
    for index in 0..1_000 {
        set_numeric(&cases[index % cases.len()].0);
    }
    for reader in readers {
        reader.join().unwrap();
    }
}
