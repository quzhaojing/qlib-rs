"""Native directory-search oracle with the source trailing-name boundary policy."""
import contextlib
import ctypes
from ctypes import wintypes
import io
import json
import runpy
from pathlib import Path

with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(Path(__file__).with_name("read_link.py")))
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.FindFirstFileW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.WIN32_FIND_DATAW)]
kernel.FindFirstFileW.restype = wintypes.HANDLE
kernel.FindClose.argtypes = [wintypes.HANDLE]
kernel.FindClose.restype = wintypes.BOOL

def probe(value):
    if "\0" in value:
        return {"error": "nul"}
    if value.endswith(("/", "\\")):
        value = value.rstrip("/\\")
        # Source checks the final remaining index (length minus one).
        if len(value) <= 1 or (len(value) == 2 and value[1] == ":"):
            return {"skip": True}
    info = wintypes.WIN32_FIND_DATAW()
    handle = kernel.FindFirstFileW(value, ctypes.byref(info))
    if handle == wintypes.HANDLE(-1).value:
        return {"error": ctypes.get_last_error()}
    kernel.FindClose(handle)
    return {"attributes": info.dwFileAttributes,
        "tag": info.dwReserved0 if info.dwFileAttributes & 0x400 else 0}

base = fixture["base"]
inputs = fixture["inputs"] + [str(base) + "\\" + value for value in ["*.txt", "?elative", "*",
    "dir-link/\\", "junction///", "absent///"]] + [
    "a/", "a\\", "ab/", "/", "\\", "//", "C:/", "1:/", "C:", "C://"]
print(json.dumps([[fixture["units"](value), probe(value)] for value in inputs]))
