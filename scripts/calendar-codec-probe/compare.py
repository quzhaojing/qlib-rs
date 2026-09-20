"""Compare the complete byte-pair space against the host Python CP936 codec."""
import json
import subprocess
import sys

native = json.loads(subprocess.check_output([sys.argv[1]]))
assert len(native) == 65536
extra, missing, changed = [], [], []
for value, got in enumerate(native):
    try:
        expected = list(map(ord, value.to_bytes(2, "big").decode("cp936")))
    except UnicodeDecodeError:
        expected = None
    if expected == got:
        continue
    row = [f"{value:04x}", got, expected]
    if expected is None:
        extra.append(row)
    elif got is None:
        missing.append(row)
    else:
        changed.append(row)
non_private_extra = [row for row in extra if not (len(row[1]) == 1 and 0xE000 <= row[1][0] <= 0xF8FF)]
keys = [int(row[0], 16) for row in extra if int(row[0], 16) >= 0x8100]
ranges = []
for value in keys:
    if ranges and value == ranges[-1][1] + 1:
        ranges[-1][1] = value
    else:
        ranges.append([value, value])
print(json.dumps(dict(extra_count=len(extra), missing=missing, changed=changed,
                     extra_double_byte_ranges=[[f"{a:04x}", f"{b:04x}"] for a,b in ranges],
                     non_private_double_byte_extra=[row for row in non_private_extra if int(row[0],16)>=0x8100])))
