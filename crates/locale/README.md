# Current UCRT locale adapter

`current_numeric_locale()` copies the **calling thread's current dynamically
linked MSVC UCRT** numeric locale. It does not read the Windows default language,
inspect another process, embed Python, or read a separately statically linked CRT.
The implementation is available only for Windows MSVC without `crt-static`.
Other platforms can use core's explicit locale snapshot/provider interface.

## Ownership and safety

1. `localeconv()` refreshes the thread's retained reference before freezing it.
2. `_configthreadlocale(1)` disables global-to-thread resynchronization locally.
3. `_get_current_locale()` pins that exact locale with an owning reference.
4. A second `localeconv()` obtains its numeric fields. `widestring` copies the
   wide decimal/separator fields, and `CStr` copies grouping including NUL.
5. RAII frees the handle and restores the previous thread setting, on success
   and on every represented error. No pointer or borrow escapes this scope.

The thread guard is not Send/Sync. No callback, asynchronous suspension or locale
setter is invoked while copying. The owning handle additionally protects storage
against locale-dependent reentry during allocations. This API is not suitable for
asynchronous signal handlers; UCRT locale setters are not signal-handler-safe.
External code must not mutate the contents of CRT-owned `lconv` pointers.

The implementation uses public CRT calls, **not** `_lock_locales`, opaque locale
structure dereferences, or undocumented CRT-internal exports. Microsoft ships the
supporting source with SDK 10.0.26100.0: `localeconv.cpp`, `wsetlocale.cpp`,
`locale_refcounting.cpp`, and `locale_update.cpp`. Those sources establish retained
thread references, copy-on-update ownership, and the lack of refresh in
`_configthreadlocale`. The standalone probe reproduces the stale result obtained
by enabling thread locale before refreshing; it is not merely a hypothetical race.

## Dependency decision and ABI

`widestring = 1.2.1` supplies checked UTF-16 conversion (MIT/Apache-2.0, MSRV 1.58);
`thiserror` supplies errors. Invalid UTF-16 is reported, never replaced lossily.
Existing libc 0.2.189 has no Windows `lconv`/localeconv binding, installed windows
bindings do not expose this CRT surface, and locale-settings 0.3.0 lacks Windows.
The narrow local FFI is therefore an explicit compatibility exception. Replace it
when a maintained binding crate supplies this exact UCRT surface.

The private `repr(C)` layout groups only adjacent identically typed members from
the public SDK header: ten char pointers, eight chars, eight wchar_t pointers.
No pointer to the opaque owned locale is dereferenced. The independent C program
`scripts/checkpoint-format-probe/native_locale_abi.c` checks SDK size, alignment,
grouping and wide-field offsets. On the validated x64 host these are respectively
152, 8, 16, 88 and 96 bytes. Other architectures require corresponding ABI and
native contract verification; x64 evidence is not a cross-platform safety claim.

## Verification scope

Unit tests inject refresh/config/acquire/lookup/UTF-16 failures and assert exact
free/restore ordering. A separate test subprocess compares ten real Python locale
metadata records, checks global updates and existing thread-local settings, and
copies 20,000 snapshots while another thread performs 1,000 global locale writes.
No shared workspace test process has its global locale changed. These stress cases
supplement ownership reasoning; they do not prove the absence of every possible
race. Full exact Rust coverage includes this handwritten adapter without exclusions.

Authoritative references:

- [Microsoft thread locale modes](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/configthreadlocale?view=msvc-170)
- [Microsoft current locale ownership](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/get-current-locale?view=msvc-170)
- [Microsoft locale release](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/free-locale?view=msvc-170)
- [CPython Windows wide-field conversion](https://raw.githubusercontent.com/python/cpython/v3.14.0/Python/fileutils.c)
- [widestring pointer and lifetime contract](https://docs.rs/widestring/1.2.1/widestring/ucstr/struct.U16CStr.html#method.from_ptr_str)
