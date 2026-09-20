"""Masked categorical categories, scalar identity and actual Qlib import contracts."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_multi_nullable_levels.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "patterns" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
categories = levels
fixture = Path(__file__).with_name("dataframe_categorical_codes.py")
tree = ast.parse(fixture.read_bytes())
start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "cases" for t in node.targets))
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "contract" for t in node.targets))
exec(compile(ast.Module(body=tree.body[start:stop], type_ignores=[]), str(fixture), "exec"))
for case in cases:
    if "output" not in case:
        continue
    index = pd.CategoricalIndex(pd.Categorical.from_codes(case["codes"], categories=categories[case["name"]], ordered=case["ordered"]), name="datetime")
    current = pd.DataFrame(dict(x=list(range(len(index)))), index=index)
    current, case["first"] = execute(current, pd.Index([], dtype=object), False)
    _, case["second"] = execute(current, pd.Index([], dtype=object), False)
    case["right_import"] = []
    for columns in [False, True]:
        if len(index) == 0 and not columns:
            continue
        _, outcome = execute(pd.DataFrame(), index, columns)
        case["right_import"].append(dict(columns=columns, **outcome))
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
