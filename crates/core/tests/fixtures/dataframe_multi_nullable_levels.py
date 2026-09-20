"""Masked MultiIndex levels: validation, exact scalar identity and ignored append."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_nullable_append.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "samples" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
fixture = Path(__file__).with_name("dataframe_multi_levels.py")
tree = ast.parse(fixture.read_bytes())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name == "run"], type_ignores=[]), str(fixture), "exec"))
constructor_run = run
fixture = Path(__file__).with_name("dataframe_multi_import.py")
tree = ast.parse(fixture.read_bytes())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name == "run"], type_ignores=[]), str(fixture), "exec"))

levels = {}
for dtype in ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "boolean"]:
    native = pd.api.types.pandas_dtype(dtype).numpy_dtype
    if dtype == "boolean":
        values = [False, True]
    elif native.kind in "iu":
        limits = np.iinfo(native)
        values = [int(limits.min), int(limits.max)]
    else:
        values = [-0., 1.25]
    for state, data in dict(values=values, missing=values+[None], empty=[], duplicate=[values[0]]*2,
                            duplicate_missing=[None, None]).items():
        levels[f"{dtype}_{state}"] = pd.Index(pd.array(data, dtype=dtype))
for dtype in ["float32", "float64"]:
    for state, data, mask in [("nan", [np.nan, 1., 2.], [False, False, True]),
                               ("nan_duplicate", [np.nan, np.nan], [False, False]),
                               ("zeros", [-0., 0.], [False, False])]:
        levels[f"{dtype}_{state}"] = pd.Index(pd.arrays.FloatingArray(
            np.array(data, dtype=dtype), np.array(mask)))

patterns = [[], [-1], [0], [0, 1, 2, -1], [1, 0], [0, 0], [-2], [99]]
cases = [dict(name=name, **run([level], [codes], [("masked", 7)], order))
         for (name, level), codes, order in itertools.product(levels.items(), patterns, [None, 0, 1])]
category = pd.CategoricalIndex(pd.Categorical.from_codes([1, 0, -1], categories=["b", "a", "unused"], ordered=True))
ordinary = pd.Index([0, 2**64-1], dtype="uint64")
for name, level in levels.items():
    if not name.endswith("_missing") or name.endswith("duplicate_missing"):
        continue
    for other in [category, ordinary]:
        for selected in [[level, other], [other, level]]:
            for left, right, order in itertools.product([[], [0], [0, 1, -1]], [[], [0], [0, 1, -1]], [None, 0, 1, 2]):
                cases.append(dict(name=name, **run(selected, [left, right], ["first", "second"], order)))
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
