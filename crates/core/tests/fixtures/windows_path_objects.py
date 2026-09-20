"""Compare real pathlib construction with home expansion disabled by input shape."""
import itertools
import json
from pathlib import Path

def units(value):
    data = value.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(data[i:i+2], "little") for i in range(0,len(data),2)]

prefixes = ["", "./", "../", "C:", "C:/", "1:", "/", "//", "///", "//server/share",
            "//?/C:", "//?/UNC/server/share", "//./NUL", "//./", "//?./share",
            "//?/unc/server/share", "//server/", "//?/", "//./UNC/server/share", "//?/x/y/z"]
tails = ["", "/", "\\", "/.", "/..", "/a/./b/", "/a//b", "a:b", "a", "./a:b", "\0", "\ud800"]
cases = []
for prefix, tail in itertools.product(prefixes,tails):
    value = prefix + tail
    cases.append([units(value), units(str(Path(value)))])
print(json.dumps(cases))
