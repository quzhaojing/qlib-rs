"""Actual Qlib append of masked numeric/bool indexes against native and object indexes."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop],type_ignores=[]),str(fixture),"exec"))
base_snapshot = index_snapshot

def index_snapshot(index):
    result = base_snapshot(index)
    if isinstance(index.dtype, (pd.Int8Dtype,pd.Int16Dtype,pd.Int32Dtype,pd.Int64Dtype,
                               pd.UInt8Dtype,pd.UInt16Dtype,pd.UInt32Dtype,pd.UInt64Dtype,
                               pd.Float32Dtype,pd.Float64Dtype,pd.BooleanDtype)):
        result["masked"] = dict(dtype=str(index.dtype.numpy_dtype),
                                physical=[scalar(v) for v in index._values._data],
                                mask=index._values._mask.tolist())
    return result

samples = {}
for dtype in ["Int8","Int16","Int32","Int64","UInt8","UInt16","UInt32","UInt64","Float32","Float64","boolean"]:
    for state, values in dict(empty=[],values=[0,1],missing=[0,None],all_missing=[None]).items():
        samples[f"{dtype}_{state}"] = pd.Index(pd.array(values,dtype=dtype),name="datetime")
    native = pd.api.types.pandas_dtype(dtype).numpy_dtype
    for state, values in dict(empty=[],values=[0,1]).items():
        samples[f"native_{dtype}_{state}"] = pd.Index(np.asarray(values,dtype=native),name="datetime")
samples["object_empty"] = pd.Index([],dtype=object,name="datetime")
samples["object_values"] = pd.Index([True,None],dtype=object,name="datetime")
samples["native_nan"] = pd.Index([np.nan,1.5],name="datetime")
for dtype in ["float32","float64"]:
    samples[f"{dtype}_valid_nan"] = pd.Index(pd.arrays.FloatingArray(
        np.asarray([np.nan,-0.,1.],dtype=dtype),np.asarray([False,True,False])),name="datetime")
samples["multi_values"] = pd.MultiIndex.from_tuples([("a",1),("b",2)])
samples["multi_missing"] = pd.MultiIndex.from_tuples([("a",None)])
samples["multi_empty"] = pd.MultiIndex(levels=[["unused"],[99]],codes=[[],[]])
pairs = []
for (ln,left),(rn,right),lc,rc in itertools.product(samples.items(),samples.items(),[False,True],[False,True]):
    if "masked" not in index_snapshot(left) and "masked" not in index_snapshot(right):
        continue
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {},index=left)
    _,outcome = execute(frame,right,rc)
    pairs.append(dict(left=ln,right=rn,left_columns=lc,right_columns=rc,**outcome))
contract = dict(inputs={k:index_snapshot(v) for k,v in samples.items()},pairs=pairs)
digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__,numpy=np.__version__,digest=digest,**contract)))
