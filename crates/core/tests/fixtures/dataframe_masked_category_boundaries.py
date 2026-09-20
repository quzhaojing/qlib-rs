"""Qlib masked categories at numeric lookup, narrowing and precision boundaries."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_nullable_boundaries.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
focus = []
for dtype in ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "boolean"]:
    categories = pd.Index(pd.array(samples[f"category_{dtype}_values"].categories, dtype=dtype))
    for state, codes in dict(values=list(range(len(categories))),
                             missing=list(range(len(categories))) + [-1], empty=[]).items():
        label = f"masked_category_{dtype}_{state}"
        samples[label] = pd.CategoricalIndex(pd.Categorical.from_codes(codes, categories=categories), name="datetime")
        focus.append(label)
for dtype, values in [("Int64", [2**53+1]), ("UInt64", [2**64-1]),
                       ("UInt8", [0, 1]), ("UInt16", [0, 1])]:
    categories = pd.Index(pd.array(values, dtype=dtype))
    label = f"masked_category_{dtype}_lookup"
    samples[label] = pd.CategoricalIndex(pd.Categorical.from_codes(
        list(range(len(categories))) + [-1], categories=categories), name="datetime")
    focus.append(label)
for dtype in ["float32", "float64"]:
    categories = pd.Index(pd.arrays.FloatingArray(np.array([np.nan, 1.], dtype=dtype), np.array([False, False])))
    label = f"masked_category_{dtype}_valid_nan"
    samples[label] = pd.CategoricalIndex(pd.Categorical.from_codes([0, 1, -1], categories=categories), name="datetime")
    focus.append(label)
for value in [257, 2**24+1, 2**53+1, 2**63-1, 2**64-1]:
    for dtype in ["float32", "float64"]:
        values = np.array([value], dtype=dtype)
        samples[f"lookup_{value}_{dtype}"] = pd.Index(values, name="datetime")
        samples[f"lookup_{value}_masked_{dtype}"] = pd.Index(pd.arrays.FloatingArray(
            np.append(values, values), np.array([False, True])), name="datetime")
    if value <= 2**63-1:
        samples[f"lookup_{value}_int64"] = pd.Index([value], dtype="int64", name="datetime")
        samples[f"lookup_{value}_masked_Int64"] = pd.Index(pd.array([value, None], dtype="Int64"), name="datetime")
pair_names = set()
for left, right in itertools.product(focus, samples):
    pair_names.add((left, right))
    pair_names.add((right, left))
pairs = []
for (ln, rn), lc, rc in itertools.product(sorted(pair_names), [False, True], [False, True]):
    left, right = samples[ln], samples[rn]
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))
contract = dict(inputs={k: index_snapshot(v) for k, v in samples.items()}, pairs=pairs)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
