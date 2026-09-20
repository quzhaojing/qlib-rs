"""Actual ntpath.join over zero, one and multiple appended paths."""
import itertools
import json
import ntpath


def units(text):
    data = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(data[i:i+2], "little") for i in range(0, len(data), 2)]


paths = ["", "a", "a/", "a\\", ".", "..", "C:", "c:child", "C:/a", "D:child", "D:\\a",
         "\\root", "/root", "//server/share", "//server/share/", "//SERVER/SHARE/dir", "\\\\?\\C:\\dir",
         "\\\\?\\UNC\\server\\share", "///", "\\\\", "\\\\?\\", "A\0B", "\ud800:", "\udc00:name",
         "\u0130:a", "i\u0307:b", "\u03a3:a", "\u03c3:b", "\u03c2:c", "\u1c89:a", "\u1c8a:b",
         "\ua7ce:a", "\ua7cf:b", "//\u039f\u03a3/share", "//\u03bf\u03c2/SHARE"]
cases = []
for count in [1, 2, 3]:
    for parts in itertools.product(paths, repeat=count):
        cases.append(([units(part) for part in parts], units(ntpath.join(*parts))))
print(json.dumps(cases))
