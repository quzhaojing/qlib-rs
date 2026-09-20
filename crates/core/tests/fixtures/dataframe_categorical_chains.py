"""Source categorical history through two successive actual Qlib appends."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_categorical_append.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

chains = []
for (name, first), next_name in itertools.product(samples.items(),
        ["int_plain", "bool_False_permuted", "text_True_values", "float_True_values", "utc_False_values"]):
    initial = pd.DataFrame(columns=["x"], index=pd.Index([], dtype=object, name="datetime"))
    current, one = execute(initial, first, True)
    assert current is not None
    _, two = execute(current, samples[next_name], True)
    chains.append(dict(first=name, second=next_name, first_output=one, second_output=two))
contract = dict(inputs={name: index_snapshot(index) for name, index in samples.items()}, chains=chains)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
