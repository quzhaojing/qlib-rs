"""Categorical level metadata, validation and ignored Qlib append contracts."""
import ast
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_multi_import.py")
tree = ast.parse(fixture.read_bytes())
# Reuse the unchanged constructor and Qlib append harness, not its case list.
exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), str(fixture), "exec"))
cases = []
for name, values in levels.items():
    if name in ["duplicates", "mixed_missing"]:
        continue
    categories = values[~values.isna()]
    for ordered in [False, True]:
        for category_codes in [[], [-1], [0], [0,-1], [0,0], [-1,-1]]:
            if 0 in category_codes and not len(categories):
                continue
            level = pd.CategoricalIndex(pd.Categorical.from_codes(
                category_codes, categories=categories, ordered=ordered))
            for row_codes in [[], [-1], [0], [0,1,-1], [1,0], [-2], [99]]:
                for order in [None, 0, 1]:
                    cases.append(run([level], [row_codes], ["category"], order))
category = pd.CategoricalIndex(pd.Categorical.from_codes([1,0,-1], categories=["b","a","unused"], ordered=True))
for selected in [[category, levels["unsigned"]], [levels["unsigned"], category]]:
    for a, b, order in itertools.product([[],[0],[0,1,-1]], [[],[0],[0,1,-1]], [None,0,1,2]):
        cases.append(run(selected, [a,b], ["first", "second"], order))
contract = dict(cases=cases)
digest = hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__,numpy=np.__version__,digest=digest,**contract)))
