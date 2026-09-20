"""Directional Qlib appends between masked indexes and categorical indexes."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_nullable_append.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
masked = {k: v for k, v in samples.items() if "masked" in index_snapshot(v)}
categories = {}
types = {name: pd.Index(np.asarray([0, 1, 3], dtype=name))
         for name in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"]}
types["bool"] = pd.Index([False, True])
types["text"] = pd.Index(["a", "b"], dtype=object)
types["object_bool"] = pd.Index([False, True], dtype=object)
types["object_numeric"] = pd.Index([0, 1, 3], dtype=object)
types["object_mixed"] = pd.Index([True, "a"], dtype=object)
types["timestamp"] = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
types["duration"] = pd.TimedeltaIndex([0, 1], unit="s")
for (name, values), ordered, (state, codes) in itertools.product(
        types.items(), [False, True], dict(empty=[], values=[0, 1], missing=[0, -1]).items()):
    categories[f"category_{name}_{ordered}_{state}"] = pd.CategoricalIndex(
        pd.Categorical.from_codes(codes, categories=values, ordered=ordered), name="datetime")
for name in ["int64", "bool"]:
    for state, codes in dict(empty=[], missing=[-1]).items():
        categories[f"category_empty_{name}_{state}"] = pd.CategoricalIndex(
            pd.Categorical.from_codes(codes, categories=types[name][:0]), name="datetime")
inputs = {**masked, **categories}
pairs = []
for (mn, mask), (cn, cat), reverse, lc, rc in itertools.product(
        masked.items(), categories.items(), [False, True], [False, True], [False, True]):
    ln, left, rn, right = (cn, cat, mn, mask) if reverse else (mn, mask, cn, cat)
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))
contract = dict(inputs={k: index_snapshot(v) for k, v in inputs.items()}, pairs=pairs)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
