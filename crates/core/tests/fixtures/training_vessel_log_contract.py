"""Run the original Qlib logger methods with NumPy and observable collaborators."""
import ast
import json
import random
import struct
import sys
import warnings
from types import SimpleNamespace

import numpy as np

source = sys.argv[1]
with open(source, encoding="utf-8") as handle:
    tree = ast.parse(handle.read(), filename=source)
base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainingVesselBase")
base.bases = []
base.body = [node for node in base.body if isinstance(node, ast.FunctionDef) and node.name in {"log", "log_dict"}]
messages = []
events = []
namespace = {"np": np, "_logger": SimpleNamespace(info=lambda message: messages.append(message))}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), base], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), source, "exec"), namespace)
vessel = namespace["TrainingVesselBase"]()

class Trainer:
    counter = 0
    @property
    def current_iter(self):
        result = self.counter
        self.counter += 1
        return result

vessel.trainer = Trainer()
inputs = []
values = {}
for dtype in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "bool"]:
    raw = [1, 2, 3, 4] if dtype != "bool" else [True, False, True, False]
    inputs.append({"name": dtype, "dtype": dtype, "values": raw})
    values[dtype] = np.array(raw, dtype=dtype).reshape(2, 2)
for name, dtype, raw in [
    ("f16_fraction", "float16", [.1, .2]),
    ("f32_fraction", "float32", [.1, .2]),
    ("empty", "float64", []),
    ("nan", "float64", [float("nan"), 1]),
    ("inf", "float64", [float("inf"), 1]),
    ("opposite_inf", "float64", [float("inf"), -float("inf")]),
]:
    inputs.append({"name": name, "dtype": dtype, "values": [str(v) if not np.isfinite(v) else v for v in raw]})
    values[name] = np.array(raw, dtype=dtype)
values.update(list_value=[1, 2.5], nested_list=[[1, 2], [3, 4]], tuple_value=(1, 2), text="raw text", boolean=True, null=None, integer=10**30, signed_zero=-0.0, scientific=1e16, tiny=1e-5)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vessel.log_dict(values)
normal = list(messages)
reads = vessel.trainer.counter

class ObservableTrainer:
    @property
    def current_iter(self):
        events.append("iteration")
        return 5

class Formatted:
    def __format__(self, spec):
        events.append("format")
        raise ValueError("format failed")

vessel.trainer = ObservableTrainer()
namespace["_logger"].info = lambda message: events.append("sink")
try:
    vessel.log("bad_array", ["a", "b"])
except TypeError:
    reduction_events = list(events)
try:
    vessel.log("custom", Formatted())
except ValueError:
    format_events = list(events)
events.clear()
def failed_sink(message):
    events.append(message)
    raise RuntimeError("sink failed")
namespace["_logger"].info = failed_sink
try:
    vessel.log_dict({"first": 1, "second": 2})
except RuntimeError:
    sink_events = list(events)
rng = random.Random(42)
float_bits = [0, 1, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000]
float_bits += [rng.getrandbits(64) for _ in range(256)]
float_texts = [{"bits": str(bits), "text": f"{struct.unpack('>d', bits.to_bytes(8, 'big'))[0]}"} for bits in float_bits]
reduction_inputs = [(index % 17 + 1) / 7 for index in range(1000)]
means = {dtype: float(np.mean(np.array(reduction_inputs, dtype=dtype))) for dtype in ["float16", "float32", "float64"]}
print(json.dumps({"inputs": inputs, "messages": normal, "reads": reads, "reduction_events": reduction_events, "format_events": format_events, "sink_events": sink_events, "float_texts": float_texts, "reduction_inputs": reduction_inputs, "means": means}))
