"""Actual Qlib StringDtype directional append with native/masked/object indexes."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_nullable_append.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
masked_snapshot = index_snapshot

def index_snapshot(index):
    result = masked_snapshot(index)
    if isinstance(index.dtype, pd.StringDtype):
        result["string"] = dict(storage=index.dtype.storage,
                                missing="NA" if index.dtype.na_value is pd.NA else "nan")
    return result

for storage, missing in itertools.product(["python", "pyarrow"], ["NA", "nan"]):
    dtype = pd.StringDtype(storage=storage, na_value=pd.NA if missing == "NA" else np.nan)
    for state, values in dict(empty=[], values=["a", "b"], missing=["a", None],
                              all_missing=[None], unicode=["中文", "😀"]).items():
        samples[f"{storage}_{missing}_{state}"] = pd.Index(values, dtype=dtype, name="datetime")
    if storage == "python":
        samples[f"{storage}_{missing}_surrogate"] = pd.Index([chr(0xd800), None], dtype=dtype, name="datetime")
for name, values in dict(text=["a", "b"], missing=["a", None], all_missing=[None],
                          nan=["a", np.nan], na=["a", pd.NA], nat=["a", pd.NaT]).items():
    samples[f"object_{name}"] = pd.Index(values, dtype=object, name="datetime")
for name, categories in dict(text=["a", "b", "unused"], numeric=[1, 2, 99]).items():
    for empty in [False, True]:
        samples[f"category_{name}_{empty}"] = pd.CategoricalIndex(pd.Categorical.from_codes(
            [] if empty else [0, -1], categories=categories), name="datetime")
pairs = []
for (ln, left), (rn, right), lc, rc in itertools.product(samples.items(), samples.items(), [False, True], [False, True]):
    if not isinstance(left.dtype, pd.StringDtype) and not isinstance(right.dtype, pd.StringDtype):
        continue
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))
contract = dict(inputs={k:index_snapshot(v) for k,v in samples.items()}, pairs=pairs)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
