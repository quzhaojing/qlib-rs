"""Pandas masked numeric/bool indexes: physical values and missing masks are distinct."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop],type_ignores=[]),str(fixture),"exec"))
cases = []
for dtype in ["int8","int16","int32","int64","uint8","uint16","uint32","uint64","float32","float64","bool"]:
    if dtype == "bool":
        values = [False,True]
        cls = pd.arrays.BooleanArray
    elif dtype.startswith("float"):
        values = [-np.inf,-0.,0.,np.inf,np.nan,1.25]
        cls = pd.arrays.FloatingArray
    else:
        info = np.iinfo(dtype)
        values = [info.min,0,1,info.max]
        cls = pd.arrays.IntegerArray
    data = np.asarray(values,dtype=dtype)
    for mask in [list(v) for v in itertools.product([False,True],repeat=len(data))]:
        array = cls(data.copy(),np.asarray(mask,dtype=bool))
        index = pd.Index(array,name="datetime")
        cases.append(dict(dtype=dtype,physical=[scalar(v) for v in data],mask=mask,
                          output=index_snapshot(index),missing=index.isna().tolist()))
    index = pd.Index(cls(np.asarray([],dtype=dtype),np.asarray([],dtype=bool)),name="datetime")
    cases.append(dict(dtype=dtype,physical=[],mask=[],output=index_snapshot(index),missing=[]))
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__,numpy=np.__version__,digest=digest,**contract)))
