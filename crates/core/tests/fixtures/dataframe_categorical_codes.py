"""Categorical category validation, explicit codes, and scalar identity."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

categories = dict(
    empty=pd.Index([], dtype=object), empty_int=pd.Index([], dtype="int64"),
    int=pd.Index([2, 1, 99], dtype="int64"), uint=pd.Index([0, 2**64-1], dtype="uint64"),
    bool=pd.Index([True, False]), float=pd.Index([1.5, -0.0]),
    float32=pd.Index(np.array([1.25, -0.0], dtype="float32")),
    text=pd.Index(["b", "a", "unused"], dtype=object),
    tuple=pd.Index([("a", None), (1, 2), ()], dtype=object, tupleize_cols=False),
    duplicate=pd.Index([True, 1, 1.0], dtype=object),
    null=pd.Index([None, "a"], dtype=object), nan=pd.Index([np.nan, 1.0]),
    na=pd.Index([pd.NA, "a"], dtype=object), nat=pd.Index([pd.NaT, "a"], dtype=object),
    duplicate_null=pd.Index([None, None, 1, 1], dtype=object),
)
for unit in ["s", "ms", "us", "ns"]:
    for zone in [None, "UTC"]:
        categories[f"datetime_{unit}_{zone}"] = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz=zone).as_unit(unit)
    categories[f"duration_{unit}"] = pd.TimedeltaIndex(["1h", "2h"]).as_unit(unit)
    categories[f"missing_datetime_{unit}"] = pd.DatetimeIndex(["2024-01-01", None]).as_unit(unit)

cases = []
patterns = [[], [-1], [0], [1], [0, 1, -1], [1, 0, -1, 1], [-2], [99], [99, -2], [-1, -1]]
for name, values in categories.items():
    for codes in patterns:
        for ordered in [False, True]:
            case = dict(name=name, categories=index_snapshot(values), codes=codes, ordered=ordered)
            before = index_snapshot(values)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    result = pd.CategoricalIndex(pd.Categorical.from_codes(codes, categories=values, ordered=ordered), name="datetime")
                except Exception as error:
                    case.update(error=type(error).__name__, message=str(error))
                else:
                    case["output"] = index_snapshot(result)
            assert before == index_snapshot(values)
            case["warnings"] = [[type(w.message).__name__, str(w.message)] for w in captured]
            cases.append(case)
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
