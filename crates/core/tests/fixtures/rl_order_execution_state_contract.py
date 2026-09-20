"""Snapshot the complete unchanged qlib.rl.order_execution.state module surface."""

import ast
import hashlib
import inspect
import json
import sys
import types
import typing
from pathlib import Path


source_path = Path(sys.argv[1])
raw = source_path.read_bytes()
tree = ast.parse(raw)


def module(name: str, **members: object) -> types.ModuleType:
    value = types.ModuleType(name)
    value.__dict__.update(members)
    return value


class Order:
    pass


class Timestamp:
    pass


class DatetimeIndex:
    pass


class DataFrame:
    pass


class Ndarray:
    pass


sys.modules.update(
    {
        "qlib": module("qlib"),
        "qlib.backtest": module("qlib.backtest", Order=Order),
        "qlib.typehint": module("qlib.typehint", TypedDict=typing.TypedDict),
        "numpy": module("numpy", ndarray=Ndarray),
        "pandas": module(
            "pandas",
            Timestamp=Timestamp,
            DatetimeIndex=DatetimeIndex,
            DataFrame=DataFrame,
        ),
    }
)

loaded = module("qlib.rl.order_execution.state")
loaded.__package__ = "qlib.rl.order_execution"
exec(compile(raw, str(source_path), "exec"), loaded.__dict__)
metrics_type = loaded.SAOEMetrics
state_type = loaded.SAOEState


def annotation_text(annotation: object) -> str:
    return getattr(annotation, "__forward_arg__", str(annotation))


def annotations(value: type) -> list[list[str]]:
    return [[name, annotation_text(annotation)] for name, annotation in value.__annotations__.items()]


imports = []
type_checking_imports = []
for node in tree.body:
    if isinstance(node, ast.Import):
        imports.append(
            ",".join(
                alias.name + (f" as {alias.asname}" if alias.asname else "") for alias in node.names
            )
        )
    elif isinstance(node, ast.ImportFrom):
        prefix = "." * node.level + (node.module or "")
        imports.append(prefix + ":" + ",".join(alias.name for alias in node.names))
    elif isinstance(node, ast.If):
        type_checking_imports.extend(
            (child.module or "") + ":" + ",".join(alias.name for alias in child.names)
            for child in node.body
            if isinstance(child, ast.ImportFrom)
        )

metric_values = [object() for _ in metrics_type.__annotations__]
metrics = metrics_type(**dict(zip(metrics_type.__annotations__, metric_values, strict=True)))
partial_metrics = metrics_type(stock_id=metric_values[0])
metrics["extra"] = metric_values[1]
state_values = [object() for _ in state_type._fields]
state = state_type(*state_values)
replacement = object()
replaced = state._replace(cur_time=replacement)
made = state_type._make(state_values)
as_dict = state._asdict()

facts = {
    "metrics_construct_plain_dict": type(metrics) is dict,
    "metrics_all_values_retain_identity": all(
        metrics[name] is value for name, value in zip(metrics_type.__annotations__, metric_values, strict=True)
    ),
    "metrics_partial_runtime_construction": partial_metrics == {"stock_id": metric_values[0]},
    "metrics_extra_runtime_key": metrics["extra"] is metric_values[1],
    "state_is_tuple": isinstance(state, tuple),
    "state_values_retain_identity": all(
        state[index] is value for index, value in enumerate(state_values)
    ),
    "state_asdict_order_and_identity": list(as_dict) == list(state_type._fields)
    and all(as_dict[name] is value for name, value in zip(state_type._fields, state_values, strict=True)),
    "state_replace_is_nonmutating": replaced.cur_time is replacement
    and state.cur_time is state_values[1]
    and all(
        replaced[index] is value
        for index, value in enumerate(state_values)
        if index != state_type._fields.index("cur_time")
    ),
    "state_make_preserves_identity": all(
        made[index] is value for index, value in enumerate(state_values)
    ),
    "type_checking_import_absent": "BaseIntradayBacktestData" not in loaded.__dict__,
    "no_explicit_all": "__all__" not in loaded.__dict__,
    "metrics_doc_present": inspect.getdoc(metrics_type) is not None,
    "state_doc_present": inspect.getdoc(state_type) is not None,
}

snapshot = {
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": ast.get_docstring(tree, clean=False),
    "module_body": [type(node).__name__ for node in tree.body],
    "imports": imports,
    "type_checking_imports": type_checking_imports,
    "public_names": [name for name in loaded.__dict__ if not name.startswith("_")],
    "metrics_annotations": annotations(metrics_type),
    "metrics_required_keys": sorted(metrics_type.__required_keys__),
    "metrics_optional_keys": sorted(metrics_type.__optional_keys__),
    "metrics_total": metrics_type.__total__,
    "state_annotations": annotations(state_type),
    "state_fields": list(state_type._fields),
    "state_signature": str(inspect.signature(state_type)),
    "state_defaults": state_type.__new__.__defaults__,
    "facts": [name for name, present in facts.items() if present],
}
print(json.dumps(snapshot, separators=(",", ":")))
