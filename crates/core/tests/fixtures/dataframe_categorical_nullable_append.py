"""Actual Qlib nonempty/empty-manager append with explicitly masked categories."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_nullable_append.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "samples" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
samples = {}
focus = []
category_sets = {}
dtypes = ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "boolean"]
for dtype in dtypes:
    values = [False, True] if dtype == "boolean" else [0, 1]
    categories = pd.Index(pd.array(values, dtype=dtype))
    category_sets[dtype] = categories
    samples[f"native_category_{dtype}"] = pd.CategoricalIndex(pd.Categorical.from_codes(
        [0, 1, -1], categories=categories.to_numpy(dtype=categories.dtype.numpy_dtype)), name="datetime")
    samples[f"masked_{dtype}"] = pd.Index(pd.array(values+[None], dtype=dtype), name="datetime")
    samples[f"masked_nonmissing_{dtype}"] = pd.Index(pd.array(values, dtype=dtype), name="datetime")
    samples[f"native_{dtype}"] = pd.Index(categories.to_numpy(dtype=categories.dtype.numpy_dtype), name="datetime")
for dtype in ["float32", "float64"]:
    category_sets[f"{dtype}_nan"] = pd.Index(pd.arrays.FloatingArray(
        np.array([np.nan, 1.], dtype=dtype), np.array([False, False])))
    samples[f"native_nan_{dtype}"] = pd.Index(np.array([np.nan, 1.], dtype=dtype), name="datetime")
    samples[f"masked_valid_nan_{dtype}"] = pd.Index(pd.arrays.FloatingArray(
        np.array([np.nan, 1., np.nan], dtype=dtype), np.array([False, False, True])), name="datetime")
for dtype in ["Int64", "UInt64"]:
    for maximum in [2**53+3, 2**53+4]:
        category_sets[f"{dtype}_{maximum}"] = pd.Index(pd.array([2**53+1, maximum], dtype=dtype))
for name, categories in category_sets.items():
    for state, codes in [("rows", [0, 1, -1]), ("empty", [])]:
        label = f"category_{name}_{state}"
        samples[label] = pd.CategoricalIndex(pd.Categorical.from_codes(codes, categories=categories), name="datetime")
        focus.append(label)
samples["object_none"] = pd.Index([False, None], dtype=object, name="datetime")
samples["object_na"] = pd.Index([False, pd.NA], dtype=object, name="datetime")
samples["object_empty"] = pd.Index([], dtype=object, name="datetime")
samples["object_nan"] = pd.Index([np.nan, 1], dtype=object, name="datetime")
samples["object_nat"] = pd.Index([pd.NaT, 1], dtype=object, name="datetime")
pair_names = set()
for left, right in itertools.product(focus, samples):
    pair_names.add((left, right))
    pair_names.add((right, left))
for dtype in dtypes:
    categories = category_sets[dtype]
    ordered = f"ordered_{dtype}"
    reordered = f"reordered_{dtype}"
    native = f"ordered_native_{dtype}"
    samples[ordered] = pd.CategoricalIndex(pd.Categorical.from_codes([0, 1, -1], categories=categories, ordered=True), name="datetime")
    samples[reordered] = pd.CategoricalIndex(pd.Categorical.from_codes([1, 0, -1], categories=categories[::-1], ordered=True), name="datetime")
    samples[native] = pd.CategoricalIndex(pd.Categorical.from_codes([0, 1, -1], categories=categories.to_numpy(dtype=categories.dtype.numpy_dtype), ordered=True), name="datetime")
    for left, right in itertools.product([ordered, reordered, native, f"category_{dtype}_rows"], repeat=2):
        pair_names.add((left, right))
ordered_labels = [name for name in samples if name.startswith(("ordered_", "reordered_"))]
pair_names.update(itertools.product(ordered_labels, repeat=2))
pairs = []
for (ln, rn), lc, rc in itertools.product(sorted(pair_names), [False, True], [False, True]):
    left, right = samples[ln], samples[rn]
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))
contract = dict(inputs={k:index_snapshot(v) for k,v in samples.items()}, pairs=pairs)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
