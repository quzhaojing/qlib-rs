"""Typed StringDtype identity, Unicode/missing values and Arrow-storage limits."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "samples" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

inputs = [[], [None], [None, None], [""], ["a", "a", None],
          ["a", None, "", "b"], ["\x00", "a\x00b", "\n", "\\"],
          ["é", "e\u0301", "中文", "😀", chr(0x10ffff)],
          [chr(0xd800)], ["a"+chr(0xdc00)+"b"],
          [chr(0xd800)+chr(0xdc00), None], [None, "x"*10000]]
cases = []
for storage, missing, values in itertools.product(["python", "pyarrow"], ["NA", "nan"], inputs):
    dtype = pd.StringDtype(storage=storage, na_value=pd.NA if missing == "NA" else np.nan)
    case = dict(storage=storage, missing=missing, input=[scalar(v) for v in values])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            index = pd.Index(pd.array(values, dtype=dtype), name="datetime")
            case.update(output=index_snapshot(index), mask=index.isna().tolist(),
                        reversed=index_snapshot(index.take(list(reversed(range(len(index)))))),
                        slices=[index_snapshot(index[i:]) for i in range(len(index)+1)])
        except Exception as error:
            case.update(error=type(error).__name__, message=str(error))
    case["warnings"] = [[type(w.message).__name__, str(w.message)] for w in captured]
    cases.append(case)
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
