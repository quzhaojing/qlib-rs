"""Two-level StringDtype combinations: construction and ignored-empty appends."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_multi_string_levels.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "patterns" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
selected = {name: value for name, value in levels.items()
            if name.endswith(("ordinary", "missing", "surrogate"))}
selected["integer"] = pd.Index([2**63-1, 0, 99], dtype="int64")
selected["object"] = pd.Index([None, "a", "b"], dtype=object)
patterns = [([], []), ([0, -1], [1, 0]), ([0, 1, -1], [-1, 0, 1]), ([0], [])]
cases = []
for (left_name, left), (right_name, right), (lc, rc), order in itertools.product(
        selected.items(), selected.items(), patterns, [None, 0, 2]):
    if not isinstance(left.dtype, pd.StringDtype) and not isinstance(right.dtype, pd.StringDtype):
        continue
    names = [("left", 7), None]
    case = dict(name=f"{left_name}/{right_name}", **run([left, right], [lc, rc], names, order))
    if "output" in case:
        index = pd.MultiIndex(levels=[left, right], codes=[lc, rc], names=names, sortorder=order)
        frame = pd.DataFrame(dict(x=np.arange(len(index), dtype="float64")), index=index)
        current, case["empty_append"] = execute(frame, pd.Index([], dtype=object), False)
        assert current is not None
        _, case["second_empty_append"] = execute(current, pd.Index([], dtype=object), False)
    cases.append(case)
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
