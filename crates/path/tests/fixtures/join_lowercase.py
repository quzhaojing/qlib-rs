"""Pin the source Unicode version and compare full, contextual string lowercase."""
import itertools
import json
import unicodedata
assert unicodedata.unidata_version == "16.0.0"


def units(text):
    data = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(data[i:i+2], "little") for i in range(0, len(data), 2)]


inputs = [chr(code) for code in range(0x10000)]
inputs += ["".join(chr(code) for code in range(start, min(start + 1024, 0x110000)))
           for start in range(0x10000, 0x110000, 1024)]
neighbors = ["", "A", "a", "\u0301", "'", "\u200d", "\ud800", "\udc00", "\0", " ", "\U0001c89a"]
inputs += [before + "\u03a3" + after for before, after in itertools.product(neighbors, repeat=2)]
inputs += ["", "\ud800\ud800", "\udc00\udc00", "A\u03a3\ud800B", "\u0130", "A/../B"]
print(json.dumps([[units(value), units(value.lower())] for value in inputs]))
