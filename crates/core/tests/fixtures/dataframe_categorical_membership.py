"""Dtype-aware membership, mixed object labels, and distinct missing markers."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

lefts = {}
for name, cats in dict(int=pd.Index([1,0]), float=pd.Index([1.,0.]),
        bool=pd.Index([True,False]), object_int=pd.Index([1,0],dtype=object),
        object_bool=pd.Index([True,False],dtype=object)).items():
    for ordered, codes in itertools.product([False,True], [[], [0,-1]]):
        lefts[f"{name}_{ordered}_{len(codes)}"] = pd.CategoricalIndex(pd.Categorical.from_codes(codes,categories=cats,ordered=ordered),name="datetime")
rights = {dtype: pd.Index([True,False],dtype=dtype,name="datetime") for dtype in ["int64","float64","bool","object"]}
for name, values in dict(mixed=[True,1], mixed_float=[1.,False], mixed_none=[True,None],
        numeric_none=[1,None], outside=[True,"x"], none=[None], nan=[np.nan], na=[pd.NA], nat=[pd.NaT], empty=[]).items():
    rights[name] = pd.Index(values,dtype=object,name="datetime")
inputs = {**{"left_"+k:v for k,v in lefts.items()}, **{"right_"+k:v for k,v in rights.items()}}
pairs = []
for (ln,left), (rn,right), lc, rc in itertools.product(lefts.items(), rights.items(), [False,True], [False,True]):
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {},index=left)
    _, outcome = execute(frame,right,rc)
    pairs.append(dict(left="left_"+ln,right="right_"+rn,left_columns=lc,right_columns=rc,**outcome))
contract = dict(inputs={name:index_snapshot(index) for name,index in inputs.items()},pairs=pairs)
digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__,numpy=np.__version__,digest=digest,**contract)))
