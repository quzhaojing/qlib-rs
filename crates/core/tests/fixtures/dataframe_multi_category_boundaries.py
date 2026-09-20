"""Categorical MultiIndex inner/outer missing codes and numeric precision."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_multi_nullable_levels.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "levels" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

categories = {}
for dtype in ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "boolean"]:
    native = pd.api.types.pandas_dtype(dtype).numpy_dtype
    if native.kind in "iu":
        limits = np.iinfo(native)
        values = list(dict.fromkeys([int(limits.min), int(limits.max)] +
                      ([2**53+1, 2**53] if native.itemsize == 8 else [])))
    else:
        values = [False, True] if dtype == "boolean" else [-0., 1.25]
    categories[dtype] = pd.Index(pd.array(values, dtype=dtype))
    categories[str(native)] = pd.Index(np.array(values, dtype=native))
for dtype in ["float32", "float64"]:
    categories[f"{dtype}_valid_nan"] = pd.Index(pd.arrays.FloatingArray(
        np.array([np.nan, 1.], dtype=dtype), np.array([False, False])))

patterns = [[], [-1], [0], [0, -1], [1, -1], [0, 1, -1], [0, 1, 2, -1], [1, 0], [-2], [99]]
cases = []
for name, values in categories.items():
    for ordered, inner in itertools.product([False, True], [[], [-1], [0], [0, -1], [0, 1], [1, 0, -1], [0, 0], [-1, -1]]):
        level = pd.CategoricalIndex(pd.Categorical.from_codes(inner, categories=values, ordered=ordered))
        for outer, order in itertools.product(patterns, [None, 0, 1]):
            cases.append(dict(name=name, **run([level], [outer], [("category", 7)], order)))
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
