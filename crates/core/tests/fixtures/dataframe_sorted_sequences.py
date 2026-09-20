"""Long object-factorization sequences around NumPy's partition threshold."""
import ast
import random
from pathlib import Path

sequence_fixture = Path(__file__).with_name("dataframe_sorted_factorization.py")
sequence_tree = ast.parse(sequence_fixture.read_bytes())
sequence_split = next(i for i, node in enumerate(sequence_tree.body) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "sort_pairs" for t in node.targets))
exec(compile(ast.Module(body=sequence_tree.body[:sequence_split], type_ignores=[]), str(sequence_fixture), "exec"))

pools = dict(
    numeric=values[3:12] + values[61:73] + list(range(-40, 40)),
    numeric_text=[-2, 0, 2**64-1, -.5, float("inf"), "", "a", "z", None],
    timestamps=[v for v in values if isinstance(v, pd.Timestamp) and v.tz is None],
    aware=[v for v in values if isinstance(v, pd.Timestamp) and v.tz is not None],
    duration=[v for v in values if isinstance(v, pd.Timedelta)],
    tuples=[(n, s) for n in range(6) for s in ["b", "a", None]],
    tuple_nan=[(n,) for n in range(24)] + [(np.nan,), (np.nan, 2), (np.nan, 1)],
    incompatible=[1, "b", pd.Timestamp("2024-01-01"), pd.Timedelta("1D"), (1,), None],
)
cases = []
for name, pool in pools.items():
    for length in [0, 1, 2, 15, 16, 17, 31, 64, 129]:
        for seed in range(6):
            rng = random.Random(seed)
            row = [rng.choice(pool) for _ in range(length)]
            outcome = sorted_description(row)
            codes, uniques = pd.factorize(pd.Index(row, dtype=object, tupleize_cols=False), sort=False)
            outcome["fallback"] = dict(codes=codes.tolist(), uniques=[scalar(v) for v in uniques])
            cases.append(dict(pool=name, seed=seed, values=[scalar(v) for v in row], **outcome))
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
