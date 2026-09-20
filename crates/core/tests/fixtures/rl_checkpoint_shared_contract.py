"""Unchanged Qlib names with aliased heterogeneous model values and ordered effects."""
import ast
import json
import locale
import sys
from types import SimpleNamespace

with open(sys.argv[1], encoding="utf-8") as source:
    tree = ast.parse(source.read())
classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
           and node.name in {"Callback", "Checkpoint"}]
module = ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    *classes], type_ignores=[]))
events = []


class Clock:
    @staticmethod
    def now():
        events.append(["clock"])
        return SimpleNamespace(strftime=lambda _: "20260902123456")


class Model:
    def __init__(self):
        self.calls = 0

    def __format__(self, spec):
        events.append(["format", spec])
        self.calls += 1
        if spec == "fail":
            raise ValueError("model format failed")
        return f"model-{self.calls}"

    def __str__(self):
        events.append(["str"])
        return "model中文"

    def __repr__(self):
        events.append(["repr"])
        return "Model中文"

    def __getattr__(self, name):
        events.append(["attribute", name])
        if name == "leaf":
            return 1234.5
        raise AttributeError("model attribute failed")

    def __getitem__(self, key):
        events.append(["item", key])
        if key in (0, "leaf"):
            return 1234.5
        raise KeyError("model item failed")


ns = dict(datetime=Clock)
exec(compile(module, sys.argv[1], "exec"), ns)
callback = ns["Checkpoint"].__new__(ns["Checkpoint"])
templates = [
    "{model}-{alias}", "{model.leaf:.1f}-{model[0]:.1f}-{model[leaf]:.1f}",
    "{model!r}-{alias!s}", "{model!a}", "{model:fail}{alias}",
    "{model}{missing}", "{model.missing}{alias}", "{model[missing]}-{alias}",
    "{model!q}", "{model:{alias}}", "{model!r:{alias}}", "{number!s}-{text!r}-{text[0]}",
    "{number:n}-{model.leaf:.10n}", "{number.real:.10n}-{model[0]:.10n}",
    "{alias[1]}", "{model.leaf!a}", "{model!s:>20}", "{model} }",
    "{model:{missing}}", "{model.leaf:badn}", "{text[999]}-{model}",
]
cases = []
for name in ["C", "de-DE", "hi-IN"]:
    locale.setlocale(locale.LC_NUMERIC, name)
    conv = locale.localeconv()
    metadata = dict(decimal=conv["decimal_point"], separator=conv["thousands_sep"], grouping=conv["grouping"])
    for template in templates:
        events.clear()
        model = Model()
        callback.filename = template
        trainer = SimpleNamespace(current_iter=7, metrics=dict(model=model, alias=model, number=1234.5, text="中文"))
        try:
            output, error = callback._new_checkpoint_name(trainer), None
        except Exception as exc:
            output, error = None, type(exc).__name__
        cases.append(dict(template=template, locale=metadata, output=output, error=error,
                          calls=model.calls, events=events.copy()))
print(json.dumps(cases, ensure_ascii=True))
