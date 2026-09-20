"""Continuous Qlib append through categorical MultiIndex numeric boundaries."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_multi_category_boundaries.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "patterns" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
rights = dict(tuple_one=pd.Index([(2**53+1,)], tupleize_cols=False),
              tuple_two=pd.Index([("z", 9), ("a", 3)], tupleize_cols=False),
              scalar=pd.Index([7]),
              category_int=pd.CategoricalIndex(pd.Categorical.from_codes([0, -1], categories=[2**53+1, 1])),
              empty=pd.Index([], dtype=object))
cases = []
for name, values in categories.items():
    for ordered, inner, mixed in itertools.product([False, True], [[0], [0, -1], [0, 1]], [False, True]):
        category = pd.CategoricalIndex(pd.Categorical.from_codes(inner, categories=values, ordered=ordered))
        selected = [category, pd.Index([0, 2**64-1], dtype="uint64")] if mixed else [category]
        codes = list(range(len(category)))+[-1]
        row_codes = [codes, [1]*(len(codes)-1)+[-1]] if mixed else [codes]
        index = pd.MultiIndex(levels=selected, codes=row_codes, names=["category", "number"] if mixed else ["category"])
        for (label, right), lc, rc in itertools.product(rights.items(), [False, True], [False, True]):
            frame = pd.DataFrame(dict(x=list(range(len(index)))) if lc else {}, index=index)
            current, first = execute(frame, right, rc)
            assert current is not None
            _, second = execute(current, rights["tuple_two"], True)
            cases.append(dict(label=name, right=label, left=frame_snapshot(frame), right_columns=rc, first=first, second=second))
contract = dict(inputs={k:index_snapshot(v) for k,v in rights.items()}, cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
