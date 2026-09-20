"""Actual explicit MultiIndex StringDtype levels and ignored-empty Qlib appends."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_multi_levels.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "patterns" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
ordinary_snapshot = index_snapshot


def index_snapshot(index):
    result = ordinary_snapshot(index)
    if isinstance(index.dtype, pd.StringDtype):
        result["string"] = dict(storage=index.dtype.storage,
                                missing="NA" if index.dtype.na_value is pd.NA else "nan")
    return result


levels = {}
for storage, missing in itertools.product(["python", "pyarrow"], ["NA", "nan"]):
    dtype = pd.StringDtype(storage=storage, na_value=pd.NA if missing == "NA" else np.nan)
    samples = dict(empty=[], ordinary=["b", "a", "unused"], missing=["a", None, "b"],
                   all_missing=[None], duplicate=["a", "a"], unicode=["中文", "😀"])
    if storage == "python":
        samples["surrogate"] = [chr(0xd800), "a", None]
    for name, values in samples.items():
        levels[f"{storage}_{missing}_{name}"] = pd.Index(values, dtype=dtype)

patterns = [[], [-1], [0], [1], [0, 1, -1], [1, 0, -1], [0, 0, 1], [-2], [99]]
cases = []
for (name, level), codes, order in itertools.product(levels.items(), patterns, [None, -1, 0, 1, 2]):
    case = dict(name=name, **run([level], [codes], [("level", 7)], order))
    if "output" in case:
        index = pd.MultiIndex(levels=[level], codes=[codes], names=[("level", 7)], sortorder=order)
        frame = pd.DataFrame(dict(x=np.arange(len(index), dtype="float64")), index=index)
        current, case["empty_append"] = execute(frame, pd.Index([], dtype=object), False)
        assert current is not None
        _, case["second_empty_append"] = execute(current, pd.Index([], dtype=object), False)
    cases.append(case)
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
