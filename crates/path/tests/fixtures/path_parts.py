"""Oracle uses actual native splitroot plus ntpath.split/isabs."""
import itertools
import json
import ntpath


def units(text):
    data = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(data[i:i+2], "little") for i in range(0, len(data), 2)]


prefixes = ["", "\\", "/", "C:", "C:\\", "1:", "?:", "\\\\", "\\\\server",
            "\\\\server\\share", "\\\\server\\share\\", "\\\\?\\C:\\",
            "\\\\?\\UNC\\server\\share\\", "\\\\?\\unc\\server\\share\\",
            "\\\\.\\device\\", "///", "\\\\?\\", "\\\\?\\UNC\\", "\\\\?\\UNC\\server",
            "\\\\?\\UnC\\server\\share\\", "\\\\_\\UNC\\server\\share\\", "\U0001f600:", "\U0001f600:\\"]
parts = ["", ".", "..", "a", "b", "...", "\ud800", "\udc00", "\x00", "a.", "a "]
inputs = [prefix + separator.join(components)
          for prefix, components, separator in itertools.product(prefixes, itertools.product(parts, repeat=3), ["/", "\\"])]
inputs += [chr(code) + ":/tail" for code in range(0x10000)]
inputs += ["", "a", "/", "\\", "C:", "C:/", "C:relative", "\\\\server", "\\\\server\\share"]
print(json.dumps([[units(value), [units(part) for part in ntpath.splitroot(value)],
                   [units(part) for part in ntpath.split(value)], ntpath.isabs(value)] for value in inputs]))
