"""Actual Windows ntpath normalization; UTF-16 JSON preserves native path data."""
import itertools
import json
import ntpath


def units(text):
    data = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]


prefixes = ["", "\\", "/", "C:", "C:\\", "1:", "?:", "\\\\", "\\\\server",
            "\\\\server\\share", "\\\\server\\share\\", "\\\\?\\C:\\",
            "\\\\?\\UNC\\server\\share\\", "\\\\?\\unc\\server\\share\\",
            "\\\\.\\device\\", "///", "\\\\?\\", "\\\\?\\UNC\\", "\\\\?\\UNC\\server",
            "\\\\?\\UnC\\server\\share\\", "\\\\_\\UNC\\server\\share\\", "\U0001f600:", "\U0001f600:\\"]
parts = ["", ".", "..", "a", "b", "...", "\ud800", "\udc00", "\x00", "a.", "a "]
cases = []
for prefix, components, separator in itertools.product(prefixes, itertools.product(parts, repeat=3), ["/", "\\"]):
    value = prefix + separator.join(components)
    cases.append((units(value), units(ntpath.normpath(value))))
print(json.dumps(cases))
