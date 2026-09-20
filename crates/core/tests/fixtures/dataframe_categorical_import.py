"""Qlib ignored-empty append preserves explicit categorical index metadata."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_categorical_codes.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "cases" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))

cases = []
for categories_name, values in categories.items():
    for codes in [[], [-1], [0], [1], [0, 1, -1], [1, 0, -1, 1], [-1, -1]]:
        for ordered in [False, True]:
            for name in [None, "history"]:
                try:
                    index = pd.CategoricalIndex(pd.Categorical.from_codes(codes, categories=values, ordered=ordered), name=name)
                except ValueError:
                    continue
                current = pd.DataFrame(dict(x=list(range(len(index)))), index=index)
                case = dict(categories_name=categories_name, initial=frame_snapshot(current), name=name)
                current, case["first"] = execute(current, pd.Index([], dtype=object), False)
                assert current is not None
                _, case["second"] = execute(current, pd.Index([], dtype=object), False)
                case["right_import"] = []
                for columns in [False, True]:
                    if len(index) == 0 and not columns:
                        continue  # Both 0x0 managers require actual index concatenation.
                    _, outcome = execute(pd.DataFrame(), index, columns)
                    case["right_import"].append(dict(columns=columns, **outcome))
                cases.append(case)
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
