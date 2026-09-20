"""Qlib append at numeric width, precision and masked-value boundaries."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_nullable_append.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
samples = {}
for dtype in ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "boolean"]:
    native = pd.api.types.pandas_dtype(dtype).numpy_dtype
    if native.kind in "iu":
        limits = np.iinfo(native)
        values = [int(limits.min), int(limits.max), 1]
        if dtype == "Int64":
            values += [2**53 + 1]
        if dtype == "UInt64":
            values += [2**53 + 1, 2**63 + 1]
        data = np.asarray(values, dtype=native)
        make = pd.arrays.IntegerArray
    elif native.kind == "f":
        limit = np.finfo(native).max
        data = np.asarray([-limit, -0., 1.25, limit, np.inf, np.nan], dtype=native)
        make = pd.arrays.FloatingArray
    else:
        data = np.asarray([False, True], dtype=native)
        make = pd.arrays.BooleanArray
    mask = np.zeros(len(data), dtype=bool)
    for state in ["values", "missing", "empty"]:
        current = mask.copy()
        if state == "missing":
            current[0] = True
        array = make(data.copy(), current)
        if state == "empty":
            array = array[:0]
        samples[f"masked_{dtype}_{state}"] = pd.Index(array, name="datetime")
    samples[f"native_{dtype}"] = pd.Index(data.copy(), name="datetime")
    samples[f"native_{dtype}_empty"] = pd.Index(data[:0], name="datetime")
    categories = pd.Index(data[~pd.isna(data)]).drop_duplicates()
    for state, codes in dict(values=list(range(len(categories))),
                             missing=list(range(len(categories))) + [-1], empty=[]).items():
        samples[f"category_{dtype}_{state}"] = pd.CategoricalIndex(
            pd.Categorical.from_codes(codes, categories=categories), name="datetime")
pairs = []
for (ln, left), (rn, right), lc, rc in itertools.product(samples.items(), samples.items(), [False, True], [False, True]):
    if not (ln.startswith("masked_") or rn.startswith("masked_")):
        continue
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))
contract = dict(inputs={k: index_snapshot(v) for k, v in samples.items()}, pairs=pairs)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
