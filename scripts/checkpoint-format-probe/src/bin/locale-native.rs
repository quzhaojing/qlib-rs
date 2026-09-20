//! Isolated Windows UCRT acquisition experiment, not a production API.
//! All locale mutation is confined to this subprocess. See native_locale_probe.py.

use serde_json::{Value, json};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::io::{self, Read};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, Barrier};
use widestring::U16CStr;

// Exact SDK locale.h member order, grouped only where consecutive C members
// have the same type. The companion C probe verifies size/alignment/offsets.
#[repr(C)]
struct Lconv {
    narrow: [*const c_char; 10],
    flags: [c_char; 8],
    wide: [*const u16; 8],
}

#[link(name = "ucrt")]
unsafe extern "C" {
    fn localeconv() -> *const Lconv;
    fn _configthreadlocale(flag: c_int) -> c_int;
    fn setlocale(category: c_int, name: *const c_char) -> *const c_char;
    fn _get_current_locale() -> *mut c_void;
    fn _free_locale(locale: *mut c_void);
}

struct OwnedLocale(*mut c_void);

impl Drop for OwnedLocale {
    fn drop(&mut self) {
        // SAFETY: This unique handle is the successful result of UCRT acquisition.
        unsafe { _free_locale(self.0) };
    }
}

struct ThreadLocale {
    previous: c_int,
    // Restoring a thread setting on another thread would violate the contract.
    _same_thread: PhantomData<Rc<()>>,
}

impl ThreadLocale {
    fn freeze(refresh: bool) -> Self {
        // SAFETY: Public no-argument UCRT lookup refreshes this thread's retained
        // locale reference before freezing. No returned pointer is retained here.
        if refresh {
            assert!(!unsafe { localeconv() }.is_null());
        }
        // SAFETY: 1 is the SDK's documented enable-thread-locale flag.
        let previous = unsafe { _configthreadlocale(1) };
        assert!(matches!(previous, 1 | 2));
        Self {
            previous,
            _same_thread: PhantomData,
        }
    }
}

impl Drop for ThreadLocale {
    fn drop(&mut self) {
        // SAFETY: The value came from UCRT, and this guard cannot change threads.
        assert_eq!(unsafe { _configthreadlocale(self.previous) }, 1);
    }
}

fn mode() -> c_int {
    // SAFETY: Zero queries without changing the setting.
    unsafe { _configthreadlocale(0) }
}

fn set_numeric(name: &str) {
    let name = CString::new(name).unwrap();
    // SAFETY: LC_NUMERIC = 4; CString is valid throughout the call. Test-only
    // mutation deliberately affects either this thread or this child process.
    assert!(!unsafe { setlocale(4, name.as_ptr()) }.is_null());
}

fn snapshot(refresh: bool) -> Value {
    let before = mode();
    let guard = ThreadLocale::freeze(refresh);
    // SAFETY: Capture an owning reference after freezing the thread, so allocations
    // during copying cannot invalidate the data by reentering locale-dependent CRT.
    let owned = OwnedLocale(unsafe { _get_current_locale() });
    assert!(!owned.0.is_null());
    // SAFETY: The retained UCRT thread reference remains frozen until guard
    // drops. No callback, setlocale, or async operation runs during this copy.
    // Concurrent global setters replace their references, not this thread's.
    let value = unsafe {
        let raw = localeconv().as_ref().unwrap();
        let decimal = U16CStr::from_ptr_str(raw.wide[0]).to_string().unwrap();
        let separator = U16CStr::from_ptr_str(raw.wide[1]).to_string().unwrap();
        let mut grouping = CStr::from_ptr(raw.narrow[2]).to_bytes().to_vec();
        if !grouping.is_empty() && grouping.last() != Some(&127) {
            grouping.push(0);
        }
        json!({"decimal": decimal, "separator": separator, "grouping": grouping})
    };
    drop(owned);
    drop(guard);
    assert_eq!(mode(), before);
    value
}

fn main() {
    const {
        assert!(cfg!(all(
            windows,
            target_env = "msvc",
            not(target_feature = "crt-static")
        )));
    }
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let oracle: Vec<Value> = serde_json::from_str(&input).unwrap();
    // All symbols resolve through the dynamically linked Universal CRT.
    assert_eq!(mode(), 2);
    for case in &oracle {
        set_numeric(case["name"].as_str().unwrap());
        assert_eq!(snapshot(true), case["metadata"]);
    }

    // Reproduce the tempting but stale sequence: enable before refreshing.
    set_numeric("en-US");
    let english = snapshot(true);
    std::thread::spawn(|| set_numeric("de-DE")).join().unwrap();
    let stale = snapshot(false);
    let fresh = snapshot(true);
    assert_eq!(stale, english);
    assert_eq!(fresh["decimal"], ",");
    assert_ne!(stale, fresh);

    // Already-thread-local callers retain their own locale even when another
    // thread changes the global locale; nested guards restore the prior mode.
    let mut thread_checks = 0;
    for case in &oracle {
        let case = case.clone();
        std::thread::spawn(move || {
            let guard = ThreadLocale::freeze(true);
            set_numeric(case["name"].as_str().unwrap());
            std::thread::spawn(|| set_numeric("C")).join().unwrap();
            assert_eq!(snapshot(true), case["metadata"]);
            assert_eq!(mode(), 1);
            drop(guard);
            assert_eq!(mode(), 2);
            assert_eq!(snapshot(true)["decimal"], ".");
        })
        .join()
        .unwrap();
        thread_checks += 1;
    }

    // Defined stress: readers freeze their retained reference during copying;
    // one global writer replaces global references. A coherent snapshot must
    // be one whole catalog entry, never a mixture of two locales.
    let catalog: Arc<Vec<Value>> = Arc::new(oracle.iter().map(|v| v["metadata"].clone()).collect());
    let names: Vec<String> = oracle
        .iter()
        .map(|v| v["name"].as_str().unwrap().into())
        .collect();
    let barrier = Arc::new(Barrier::new(5));
    let mut readers = Vec::new();
    for _ in 0..4 {
        let barrier = barrier.clone();
        let catalog = catalog.clone();
        readers.push(std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..10_000 {
                assert!(catalog.contains(&snapshot(true)));
            }
        }));
    }
    barrier.wait();
    for index in 0..3_000 {
        set_numeric(&names[index % names.len()]);
    }
    for reader in readers {
        reader.join().unwrap();
    }
    println!(
        "{}",
        json!({
            "metadata_cases": oracle.len(), "thread_local_cases": thread_checks,
            "stale_sequence_reproduced": true, "concurrent_snapshots": 40_000,
            "global_writes": 3_000,
            "abi": {"size": size_of::<Lconv>(), "alignment": align_of::<Lconv>(),
                    "grouping": 2 * size_of::<*const c_char>(),
                    "wide_decimal": std::mem::offset_of!(Lconv, wide),
                    "wide_separator": std::mem::offset_of!(Lconv, wide) + size_of::<*const u16>()}
        })
    );
}
