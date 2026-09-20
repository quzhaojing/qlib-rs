"""Real Qlib MetricsWriter and pandas CSV characterization, without Torch imports."""
import ast
import itertools
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

with open(sys.argv[1], encoding="utf-8") as source:
    tree = ast.parse(source.read())
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {"Callback", "MetricsWriter"}]
ns = dict(pd=pd)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes], type_ignores=[])), sys.argv[1], "exec"), ns)

class Custom:
    def __init__(self, text): self.text = text
    def __str__(self):
        if self.text == "FAIL": raise ValueError("custom format failure")
        return self.text

def scalar(tag):
    kind, *value = tag
    if kind == "null": return None
    if kind == "int": return int(value[0])
    if kind == "float": return float(value[0])
    if kind == "custom": return Custom(value[0])
    return value[0]

def run(spec):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "metrics"
        if spec.get("directory_file"): directory.write_text("existing", encoding="utf-8")
        outputs = []
        try:
            writer = ns["MetricsWriter"](directory)
        except Exception:
            return dict(spec=spec, init_error=True, outputs=[])
        for step in spec["steps"]:
            phase = step["phase"]
            target = directory / ("train_result.csv" if phase == "train" else "validation_result.csv")
            if step.get("blocked"): target.mkdir()
            trainer = SimpleNamespace(metrics=dict((k, scalar(v)) for k, v in step.get("metrics", [])))
            if step.get("absent"): del trainer.metrics
            error = False
            try:
                getattr(writer, "on_" + ("validate_end" if phase == "val" else "train_end" if phase == "train" else phase))(trainer, None)
            except Exception: error = True
            outputs.append(dict(error=error, train=len(writer.train_records), val=len(writer.valid_records),
                files={name: (directory / name).read_bytes().decode("utf-8") for name in ("train_result.csv", "validation_result.csv") if (directory / name).is_file()}))
        # MetricsWriter inherits Callback's no-op checkpoint methods.
        assert writer.state_dict() is None
        writer.load_state_dict({"ignored": True})
        return dict(spec=spec, init_error=False, outputs=outputs)

def step(phase, pairs): return dict(phase=phase, metrics=pairs)

specs = [dict(steps=[]), dict(directory_file=True, steps=[]), dict(steps=[step("train", []), step("train", []), step("train", [["a", ["int", "1"]]]), step("val", [])]),
    dict(steps=[step("train", [["z", ["float", "1"]], ["val/z", ["float", "2"]], ["a", ["int", "3"]]]), step("val", [["val/z", ["float", "2"]], ["z", ["float", "1"]]]), step("train", [["b", ["float", "nan"]], ["z", ["float", "-0.0"]]]), step("fit_start", []), step("test_end", [])]),
    dict(steps=[step("train", [["quotes,\"\r\n中文", ["text", "a,\"b\r\n中文"]], ["", ["text", ""]], ["val", ["bool", True]], ["Val/x", ["null"]]])]),
    dict(steps=[dict(phase="train", absent=True)]), dict(steps=[dict(phase="val", absent=True)]),
    dict(steps=[dict(phase="train", blocked=True, metrics=[["a", ["float", "1"]]])]),
    dict(steps=[dict(phase="val", blocked=True, metrics=[["val/a", ["float", "1"]]])]),
    dict(steps=[step("train", [["a", ["custom", "formatted"]]]), step("train", [["a", ["custom", "FAIL"]]])])]
tags = [["int", "1"], ["int", "-1"], ["int", str(2**53+1)], ["int", str(2**63)], ["int", str(2**64)], ["float", "1.5"], ["float", "nan"], ["bool", True], ["text", "x"], ["null"], None]
for left, right in itertools.product(tags, repeat=2):
    specs.append(dict(steps=[step("train", [] if value is None else [["x", value]]) for value in (left, right)]))
for values in ([['int','1'], ['int',str(10**1000)]], [['text','x'], ['int',str(10**1000)]], [['bool',True], ['int',str(10**1000)]], [['int','-1'], ['null'], ['int',str(2**64)]], [['int',str(2**63)], ['int','-1'], ['null']]):
    specs.append(dict(steps=[step("train", [["x", value]]) for value in values]))
for values in (["-0.0", "0.0", "inf", "-inf", "nan"], ["1e-7", "1e-5", "1e-4", "1e15", "1e16", "1e20", "1.2345678901234567", "5e-324", "1.7976931348623157e308"]):
    specs.append(dict(steps=[step("val", [["val/float", ["float", value]]]) for value in values]))
print(json.dumps([run(spec) for spec in specs], ensure_ascii=True, allow_nan=False))
