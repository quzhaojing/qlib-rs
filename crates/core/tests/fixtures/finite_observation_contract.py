import ast
import copy
import json
import math
import sys

import numpy as np


def load_functions(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    names = {"fill_invalid", "is_invalid", "generate_nan_observation", "check_nan_observation"}
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            node.returns = None
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                argument.annotation = None
            functions.append(node)
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {"np": np, "cast": lambda _kind, value: value, "bool": bool}
    exec(compile(module, path, "exec"), namespace)
    return namespace


contract = load_functions(sys.argv[1])
fill_invalid = contract["fill_invalid"]
is_invalid = contract["is_invalid"]
generate_nan_observation = contract["generate_nan_observation"]
check_nan_observation = contract["check_nan_observation"]


def describe(value):
    if isinstance(value, np.ndarray):
        values = []
        for item in value.reshape(-1).tolist():
            values.append("nan" if isinstance(item, float) and math.isnan(item) else item)
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": values}
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [describe(item) for item in value]
    if isinstance(value, tuple):
        return {"tuple": [describe(item) for item in value]}
    return value


def attempt(callback):
    try:
        return {"ok": True, "value": describe(callback())}
    except Exception as error:
        return {"ok": False, "error": type(error).__name__}


dtypes = ["float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
arrays = {}
for dtype in dtypes:
    source = np.array([[0, 1], [2, 3]], dtype=dtype)
    invalid = fill_invalid(source)
    arrays[dtype] = {
        "filled": describe(invalid),
        "invalid": bool(is_invalid(invalid)),
        "source_invalid": bool(is_invalid(source)),
    }

nested = {
    "float": np.array([1.0], dtype=np.float32),
    "children": [np.array(1, dtype=np.int16), (np.array([2], dtype=np.uint8),)],
}
nested_invalid = fill_invalid(nested)


class Space:
    def sample(self):
        return nested


print(json.dumps({
    "arrays": arrays,
    "nested": describe(nested_invalid),
    "nested_invalid": bool(is_invalid(nested_invalid)),
    "generated": describe(generate_nan_observation(Space())),
    "generated_check": bool(check_nan_observation(generate_nan_observation(Space()))),
    "scalar_float": attempt(lambda: fill_invalid(1.5)),
    "scalar_int": attempt(lambda: fill_invalid(2)),
    "scalar_bool": attempt(lambda: fill_invalid(True)),
    "bool_array": attempt(lambda: fill_invalid(np.array([True], dtype=bool))),
    "complex_array": attempt(lambda: fill_invalid(np.array([1j], dtype=np.complex64))),
    "opaque_fill": attempt(lambda: fill_invalid("opaque")),
    "opaque_invalid": bool(is_invalid("opaque")),
    "empty_float_invalid": bool(is_invalid(np.array([], dtype=np.float64))),
    "empty_dict_invalid": bool(is_invalid({})),
    "short_circuit": bool(is_invalid([np.array([0.0]), np.array([True], dtype=bool)])),
}))
