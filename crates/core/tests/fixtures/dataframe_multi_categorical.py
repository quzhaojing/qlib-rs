"""Actual Qlib MultiIndex append with categorical NumPy boxing."""
import ast
import datetime
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_categorical_append.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
original_scalar = scalar

def scalar(value):
    if type(value) is datetime.datetime:
        return ["datetime", value.year, value.month, value.day, value.hour, value.minute,
                value.second, value.microsecond, value.fold, str(value.tzinfo)]
    if type(value) is datetime.timedelta:
        return ["timedelta", value.days, value.seconds, value.microseconds]
    return original_scalar(value)

samples = {k:v for k,v in samples.items() if isinstance(v, pd.CategoricalIndex)}
for kind, base in dict(datetime=pd.DatetimeIndex(["2024-01-01"]),
                       duration=pd.TimedeltaIndex(["1h"])).items():
    for unit in ["s", "ms", "us"]:
        samples[f"{kind}_{unit}"] = pd.CategoricalIndex(pd.Categorical.from_codes(
            [0,-1], categories=base.as_unit(unit)), name="datetime")
for dtype in ["int64", "bool", "object", "datetime64[ns]"]:
    samples[f"empty_{dtype}"] = pd.CategoricalIndex(pd.Categorical.from_codes(
        [-1], categories=pd.Index([],dtype=dtype)), name="datetime")
lefts = [pd.MultiIndex.from_tuples([("a",1)]),
         pd.MultiIndex(levels=[["unused"],[99]],codes=[[],[]]),
         pd.MultiIndex.from_tuples([(1,2),(3,4)])]
cases = []
for left, (name,right), lc, rc in itertools.product(lefts, samples.items(), [False,True], [False,True]):
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame,right,rc)
    cases.append(dict(left=frame_snapshot(frame),input=index_snapshot(right),
                      right_columns=rc,label=name,**outcome))
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__,numpy=np.__version__,digest=digest,**contract)))
