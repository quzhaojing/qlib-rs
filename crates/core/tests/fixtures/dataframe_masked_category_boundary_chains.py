"""Continuous Qlib histories for masked category numeric lookup boundaries."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_masked_category_boundaries.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pair_names" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
starts = ["masked_category_Int64_values", "masked_category_Int64_missing", "masked_category_Int64_lookup",
          "masked_category_UInt64_values", "masked_category_UInt64_missing", "masked_category_UInt64_lookup",
          "masked_category_UInt8_lookup", "masked_category_UInt16_lookup", "masked_category_Float32_missing",
          "masked_category_float64_valid_nan", "masked_category_boolean_missing"]
firsts = ["lookup_9007199254740993_float64", "lookup_257_masked_Int64", "lookup_9223372036854775807_masked_Int64",
          "masked_Int64_missing", "masked_UInt64_missing", "masked_Float32_values",
          "masked_category_Int64_missing", "native_boolean", "masked_Int16_empty"]
seconds = ["masked_category_UInt8_lookup", "masked_category_UInt64_lookup", "lookup_9007199254740993_float32",
           "masked_Int64_missing", "masked_Int16_empty"]
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
