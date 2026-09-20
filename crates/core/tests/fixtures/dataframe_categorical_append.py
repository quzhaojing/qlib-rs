"""Actual Qlib categorical concatenation, directionality and empty managers."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

samples = {}
for label, values in dict(text=pd.Index(["b", "a", "unused"]),
                          int=pd.Index([2, 1, 99]),
                          float=pd.Index([2., 1., 99.]),
                          bool=pd.Index([True, False]),
                          tuple=pd.Index([("b", 2), ("a", 1)], tupleize_cols=False),
                          datetime=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
                          utc=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
                          duration=pd.TimedeltaIndex(["1h", "2h"])).items():
    for ordered in [False, True]:
        for state, cats, codes in [("values", values, [0, -1, 1]),
                                   ("permuted", values[::-1], [0, 1]),
                                   ("empty", values, []),
                                   ("subset", values[:1], [0])]:
            samples[f"{label}_{ordered}_{state}"] = pd.CategoricalIndex(
                pd.Categorical.from_codes(codes, categories=cats, ordered=ordered), name="datetime")
    samples[f"{label}_plain"] = values.rename("datetime")
samples["object_empty"] = pd.Index([], dtype=object, name="datetime")
samples["int_empty"] = pd.Index([], dtype="int64", name="datetime")

pairs = []
for (ln, left), (rn, right), lc, rc in itertools.product(samples.items(), samples.items(), [False, True], [False, True]):
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))
contract = dict(inputs={name: index_snapshot(index) for name, index in samples.items()}, pairs=pairs)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
