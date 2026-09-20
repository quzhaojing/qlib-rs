"""Strict host-Python codecs: exhaustive byte pairs and mixed-sequence failures."""
import json
import random

def decode(data, encoding):
    try:
        return [data.decode(encoding), None]
    except UnicodeDecodeError as failure:
        return [None, [failure.start, failure.end, failure.reason]]

rng = random.Random(492)
mixed = [b"", b"\x81\x30\x81\x30", b"\x90\x30\x81\x30", "中文".encode("gbk") + b"\x80", b"ok\xff", b"ok\xffx"]
mixed += [rng.randbytes(length) for length in range(48) for _ in range(12)]
utf8 = [b"", b"ascii\r\n", "中文\ufeff\U0001f642".encode(), b"\xef\xbb\xbf2024-01-02", b"ok\xff", b"ok\xe4\xb8", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2\x82A"]
print(json.dumps({
    "single": [decode(bytes([value]), "cp936") for value in range(256)],
    "pairs": [decode(value.to_bytes(2, "big"), "cp936") for value in range(65536)],
    "mixed": [[list(data), decode(data, "cp936")] for data in mixed],
    "utf8": [[list(data), decode(data, "utf-8")] for data in utf8],
}))
