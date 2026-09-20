"""Source category hashing: column order and scalar string character expansion."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_string_nested_categories.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))
categories = {}
for name, values in dict(
        column_first=[("a", chr(0xd800)), (chr(0xd801), "b")],
        scalar_character=[(1, 2), "ab" + chr(0xd800)],
        same_column=[("x" + chr(0xd801),), (chr(0xd800),)],
).items():
    for reverse, ordered in itertools.product([False, True], [False, True]):
        dictionary = pd.Index(values[::-1] if reverse else values,
                              dtype=object, tupleize_cols=False)
        for state, codes in dict(empty=[], missing=[-1], first=[0, -1], last=[1]).items():
            categories[f"category_{name}_{reverse}_{ordered}_{state}"] = pd.CategoricalIndex(
                pd.Categorical.from_codes(codes, categories=dictionary, ordered=ordered), name="datetime")
samples = dict(strings, **categories)
pairs = []
for (cn, category), (sn, string), reverse, lc, rc in itertools.product(
        categories.items(), strings.items(), [False, True], [False, True], [False, True]):
    ln, left, rn, right = (sn, string, cn, category) if reverse else (cn, category, sn, string)
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))
contract = dict(inputs={k: index_snapshot(v) for k, v in samples.items()}, pairs=pairs)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
