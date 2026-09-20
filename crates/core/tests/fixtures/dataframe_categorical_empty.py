"""Empty category sets with missing codes and typed append inputs."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

lefts = {}
for dtype, ordered, codes in itertools.product(
        ["object", "int64", "bool", "datetime64[ns]"], [False, True], [[], [-1]]):
    lefts[f"{dtype}_{ordered}_{len(codes)}"] = pd.CategoricalIndex(
        pd.Categorical.from_codes(codes, categories=pd.Index([], dtype=dtype),
                                  ordered=ordered), name="datetime")
rights = {dtype: pd.Index([1], dtype=dtype, name="datetime")
          for dtype in ["int64", "float64", "bool", "object"]}
inputs = {**{"left_"+k:v for k,v in lefts.items()},
          **{"right_"+k:v for k,v in rights.items()}}
pairs = []
for (ln,left), (rn,right), lc, rc in itertools.product(
        lefts.items(), rights.items(), [False,True], [False,True]):
    for a, b in [("left_"+ln, "right_"+rn), ("right_"+rn, "left_"+ln)]:
        index = inputs[a]
        frame = pd.DataFrame(dict(x=list(range(len(index)))) if lc else {}, index=index)
        _, outcome = execute(frame, inputs[b], rc)
        pairs.append(dict(left=a, right=b, left_columns=lc, right_columns=rc, **outcome))
contract = dict(inputs={name:index_snapshot(index) for name,index in inputs.items()}, pairs=pairs)
digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__,numpy=np.__version__,digest=digest,**contract)))
