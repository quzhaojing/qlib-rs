"""Two actual Qlib append calls retaining the first call's index identity."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_string_append.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
firsts = ["python_NA_empty", "python_NA_missing", "pyarrow_nan_missing", "pyarrow_NA_values",
          "object_missing", "object_all_missing", "Int64_missing", "category_text_False", "multi_values", "native_nan"]
seconds = ["python_nan_values", "python_NA_all_missing", "pyarrow_NA_values",
           "object_empty", "category_numeric_False", "multi_missing"]
cases = []
for (name, index), first_name, second_name, columns in itertools.product(
        [(k,v) for k,v in samples.items() if isinstance(v.dtype, pd.StringDtype)], firsts, seconds, [False, True]):
    frame = pd.DataFrame(dict(x=list(range(len(index)))) if columns else {}, index=index)
    current, first = execute(frame, samples[first_name], columns)
    second = None
    if current is not None:
        _, second = execute(current, samples[second_name], True)
    cases.append(dict(left=name, first_name=first_name, second_name=second_name, columns=columns, first=first, second=second))
contract = dict(inputs={k:index_snapshot(v) for k,v in samples.items()}, cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
