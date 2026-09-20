"""Independent native by-name acquisition; successful results also match Python."""
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

class BasicInformation(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ["file_id", "creation_time", "last_access_time",
        "last_write_time", "change_time", "allocation_size", "end_of_file"]] + [
        (name, wintypes.DWORD) for name in ["attributes", "tag", "links", "device_type",
        "device_characteristics", "reserved"]] + [
        ("volume_serial", ctypes.c_int64), ("id128", ctypes.c_ubyte * 16)]

library = ctypes.WinDLL("api-ms-win-core-file-l2-1-4.dll", use_last_error=True)
query = library.GetFileInformationByName
query.argtypes = [wintypes.LPCWSTR, ctypes.c_int, wintypes.LPVOID, wintypes.ULONG]
query.restype = wintypes.BOOL

def probe(value):
    if "\0" in value:
        return {"error": "nul"}
    info = BasicInformation()
    if not query(value, 3, ctypes.byref(info), ctypes.sizeof(info)):
        return {"error": ctypes.get_last_error()}
    result = bool(info.attributes & 0x400 and info.tag == 0xa000000c)
    assert result == ntpath.islink(value), value
    return {"value": result}

print(json.dumps([[fixture["units"](value), probe(value)] for value in fixture["inputs"]]))
