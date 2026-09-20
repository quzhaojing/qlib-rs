"""Measure the native open/tag stage separately from the complete Python predicate."""
import contextlib
import ctypes
from ctypes import wintypes
import io
import json
import ntpath
from pathlib import Path
import runpy

with contextlib.redirect_stdout(io.StringIO()):
    fixture = runpy.run_path(str(Path(__file__).with_name("read_link.py")))

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
kernel.CreateFileW.restype = wintypes.HANDLE
kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
    wintypes.LPVOID, wintypes.DWORD]
kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL

class TagInfo(ctypes.Structure):
    _fields_ = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]

def probe(value):
    if "\0" in value:
        return {"error": "nul"}
    handle = kernel.CreateFileW(value, 128, 0, None, 3, 0x02200000, None)
    if handle == wintypes.HANDLE(-1).value:
        return {"error": ctypes.get_last_error(), "operation": "CreateFileW"}
    try:
        info = TagInfo()
        if not kernel.GetFileInformationByHandleEx(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return {"error": ctypes.get_last_error(), "operation": "GetFileInformationByHandleEx"}
        result = bool(info.attributes & 0x400 and info.tag == 0xa000000c)
        assert result == ntpath.islink(value), value
        return {"value": result}
    finally:
        assert kernel.CloseHandle(handle)

print(json.dumps([[fixture["units"](value), probe(value)] for value in fixture["inputs"]]))
