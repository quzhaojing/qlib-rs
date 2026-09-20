"""Categorical dtype equality differs from temporal value equality across units."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

samples = {}
for kind, base in dict(naive=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
        utc=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
        duration=pd.TimedeltaIndex(["1h", "2h"])).items():
    categories = {unit:base.as_unit(unit) for unit in ["s","ms","us","ns"]}
    categories["object"] = pd.Index(list(base),dtype=object)
    for (unit,cats), ordered in itertools.product(categories.items(),[False,True]):
        samples[f"{kind}_{unit}_{ordered}"] = pd.CategoricalIndex(pd.Categorical.from_codes([0,-1,1],categories=cats,ordered=ordered),name="datetime")
pairs = []
for (ln,left), (rn,right), lc, rc in itertools.product(samples.items(),samples.items(),[False,True],[False,True]):
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {},index=left)
    _, outcome = execute(frame,right,rc)
    pairs.append(dict(left=ln,right=rn,left_columns=lc,right_columns=rc,**outcome))
contract = dict(inputs={name:index_snapshot(index) for name,index in samples.items()},pairs=pairs)
digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__,numpy=np.__version__,digest=digest,**contract)))
