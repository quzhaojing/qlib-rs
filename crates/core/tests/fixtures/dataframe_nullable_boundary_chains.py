"""Continuous Qlib append histories retain numeric boundary outcomes."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_nullable_boundaries.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
starts = ["masked_Int64_values", "masked_Int64_missing", "masked_UInt64_values", "masked_UInt64_missing",
          "masked_Float32_values", "masked_Float32_missing", "masked_Float64_values", "masked_Float64_missing",
          "category_Int64_values", "category_UInt64_values", "category_Float64_missing"]
firsts = ["masked_Int8_missing", "masked_UInt64_values", "masked_Float64_missing", "category_Int64_missing",
          "category_UInt64_missing", "category_Float64_values", "native_boolean", "masked_Int16_empty"]
seconds = ["masked_Int64_missing", "masked_Float32_values", "category_Int64_values", "category_boolean_missing", "masked_Int16_empty"]
chains = []
for name, first, second, columns in itertools.product(starts, firsts, seconds, [False, True]):
    current = pd.DataFrame(dict(x=list(range(len(samples[name])))) if columns else {}, index=samples[name])
    steps = []
    for right, right_columns in [(first, not columns), (second, True)]:
        current, outcome = execute(current, samples[right], right_columns)
        steps.append(dict(right=right, right_columns=right_columns, **outcome))
        if current is None:
            break
    chains.append(dict(left=name, left_columns=columns, steps=steps))
contract = dict(inputs={k: index_snapshot(v) for k, v in samples.items()}, chains=chains)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
