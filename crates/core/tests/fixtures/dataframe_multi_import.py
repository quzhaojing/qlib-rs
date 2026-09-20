"""Qlib ignored-empty append of every valid explicit MultiIndex descriptor."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_multi_levels.py")
import_tree = ast.parse(fixture.read_bytes())
import_split = next(i for i, node in enumerate(import_tree.body) if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "patterns" for t in node.targets))
exec(compile(ast.Module(body=import_tree.body[:import_split], type_ignores=[]), str(fixture), "exec"))
constructor_run = run


def run(selected, codes, names, sortorder):
    result = constructor_run(selected, codes, names, sortorder)
    if "output" in result:
        index = pd.MultiIndex(levels=selected, codes=codes, names=names, sortorder=sortorder)
        frame = pd.DataFrame(dict(x=np.arange(len(index), dtype="float64")), index=index)
        current, result["empty_append"] = execute(frame, pd.Index([], dtype=object), False)
        assert current is not None
        _, result["second_empty_append"] = execute(current, pd.Index([], dtype=object), False)
    return result


exec(compile(ast.Module(body=import_tree.body[import_split:], type_ignores=[]), str(fixture), "exec"))
