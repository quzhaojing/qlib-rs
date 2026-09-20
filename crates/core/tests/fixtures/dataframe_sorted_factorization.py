"""Sorted level factorization and categorical fallback required by MultiIndex append."""
import ast
import hashlib
import itertools
import json
from pathlib import Path

sort_fixture = Path(__file__).with_name("dataframe_object_factorization.py")
sort_tree = ast.parse(sort_fixture.read_bytes())
sort_split = next(i for i, node in enumerate(sort_tree.body) if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=sort_tree.body[:sort_split], type_ignores=[]), str(sort_fixture), "exec"))


def sorted_description(items):
    index = pd.Index(items, dtype=object, tupleize_cols=False)
    before = [scalar(v) for v in index]
    result = {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            codes, uniques = pd.factorize(index, sort=True)
        except Exception as error:
            result["sorted"] = dict(error=type(error).__name__, message=str(error))
        else:
            result["sorted"] = dict(codes=codes.tolist(), dtype=str(uniques.dtype),
                                    uniques=[scalar(v) for v in uniques])
        try:
            multi = pd.MultiIndex.from_arrays([index])
        except Exception as error:
            result["multi"] = dict(error=type(error).__name__, message=str(error))
        else:
            result["multi"] = dict(codes=multi.codes[0].tolist(), dtype=str(multi.levels[0].dtype),
                                   uniques=[scalar(v) for v in multi.levels[0]],
                                   values=[scalar(v) for v in multi], sortorder=multi.sortorder)
    result["warnings"] = [[type(w.message).__name__, str(w.message)] for w in captured]
    assert before == [scalar(v) for v in index]
    return result


sort_pairs = [dict(left=i, right=j, **sorted_description([left, right]))
              for (i, left), (j, right) in itertools.product(enumerate(values), repeat=2)]
sort_histories = [[], values, list(reversed(values)), values + values,
                  [True, 1, 1., False, 0, -0., None, pd.NA, pd.NaT, np.nan],
                  [(1, None), (True, None), (1., None), (1, np.nan)],
                  [("b", 2), ("a", 1), (None, 0)],
                  [2, "b", -1, "a", None],
                  [("b", 2), ("a",), (), ("a", 3)]]
contract = dict(values=[scalar(v) for v in values], pairs=sort_pairs,
                histories=[dict(values=[scalar(v) for v in row], **sorted_description(row))
                           for row in sort_histories])
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
