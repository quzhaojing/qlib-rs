import struct
import sys
import time


def read_frame():
    header = sys.stdin.buffer.read(4)
    if len(header) != 4:
        return None
    length = struct.unpack(">I", header)[0]
    payload = sys.stdin.buffer.read(length)
    if len(payload) != length:
        return None
    return payload


mode = sys.argv[1]
if mode == "exit_before_read":
    raise SystemExit(0)
if mode == "hang_before_read":
    time.sleep(float(sys.argv[2]))
    raise SystemExit(0)
if mode == "replies":
    for reply in sys.argv[2:]:
        frame = read_frame()
        if frame is None:
            raise SystemExit(2)
        payload = bytes.fromhex(reply)
        sys.stdout.buffer.write(struct.pack(">I", len(payload)) + payload)
        sys.stdout.buffer.flush()
    raise SystemExit(0)
if mode == "shmem_replies":
    import mmap
    import os
    import zlib

    path = os.environ["QLIB_FINITE_SHMEM_PATH"]
    capacity = int(os.environ["QLIB_FINITE_SHMEM_CAPACITY"])
    with open(path, "r+b") as shared_file:
        with mmap.mmap(shared_file.fileno(), 32 + capacity) as shared:
            for item in sys.argv[2:]:
                observation, reply = item.split(":", 1)
                request = read_frame()
                if request is None:
                    raise SystemExit(2)
                request_id = struct.unpack("<Q", request[2:10])[0]
                payload = struct.pack("<q", int(observation))
                header = bytearray(32)
                header[:8] = b"QLIBSHM1"
                header[8:10] = struct.pack("<H", 1)
                header[12:20] = struct.pack("<Q", request_id)
                header[20:28] = struct.pack("<Q", len(payload))
                header[28:32] = struct.pack("<I", zlib.crc32(payload))
                shared[32 : 32 + len(payload)] = payload
                shared[:32] = header
                shared.flush()
                response = bytes.fromhex(reply)
                sys.stdout.buffer.write(struct.pack(">I", len(response)) + response)
                sys.stdout.buffer.flush()
    raise SystemExit(0)

frame = read_frame()

if mode == "reply":
    payload = bytes.fromhex(sys.argv[2])
elif mode == "env_reply":
    import os

    payload = bytes.fromhex(os.environ[sys.argv[2]])
    sys.stdout.buffer.write(struct.pack(">I", len(payload)) + payload)
    sys.stdout.buffer.flush()
    raise SystemExit(0)
if mode == "reply":
    sys.stdout.buffer.write(struct.pack(">I", len(payload)) + payload)
    sys.stdout.buffer.flush()
elif mode == "echo":
    sys.stdout.buffer.write(struct.pack(">I", len(frame)) + frame)
    sys.stdout.buffer.flush()
elif mode == "oversized":
    sys.stdout.buffer.write(struct.pack(">I", int(sys.argv[2])))
    sys.stdout.buffer.flush()
elif mode == "partial":
    sys.stdout.buffer.write(struct.pack(">I", 8) + b"x")
    sys.stdout.buffer.flush()
elif mode == "hang":
    time.sleep(float(sys.argv[2]))
elif mode != "eof":
    raise RuntimeError(f"unknown mode: {mode}")
