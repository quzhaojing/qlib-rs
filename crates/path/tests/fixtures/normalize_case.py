"""Oracle: actual ntpath.normcase, not Python str.lower or a mapping table."""
import json
import ntpath


def units(text):
    data = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(data[i:i+2], "little") for i in range(0, len(data), 2)]


inputs = [chr(code) for code in range(0x10000)]
inputs += ["".join(chr(code) for code in range(start, min(start + 1024, 0x110000)))
           for start in range(0x10000, 0x110000, 1024)]
inputs += ["", "C:/A/../B", "\\\\?\\UNC/Server/SHARE", "A\0B/\ud800", "\u0130/I/\u0131",
           "\u039f\u03a3/\u03a3\u039f", "\u1e9e/\u00df/SS", "\u212a/K", "A\u030a/\u00c5", "./../"]
print(json.dumps([[units(value), units(ntpath.normcase(value))] for value in inputs]))
