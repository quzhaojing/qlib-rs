"""Execute the unchanged upstream function with real Pandas indexed frames."""
import ast
import hashlib
import itertools
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

source = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == "89267f5cfc9e38751cb2c3a37c74ca712e8c395f492d53ed02f029a66272074f"
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dataframe_append")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "source", "exec"))


def frame(columns, rows, explicit_empty_columns=False):
    if not columns:
        return pd.DataFrame(index=range(rows), **({"columns": []} if explicit_empty_columns else {}))
    result = pd.concat([pd.Series([i + 1 + 10 * pos for i in range(rows)], dtype=dtype)
                        for pos, (_, dtype) in enumerate(columns)], axis=1)
    result.columns = [name for name, _ in columns]
    return result


def values(series):
    return [None if pd.isna(v) else v.value if isinstance(v, pd.Timestamp) else v for v in series.tolist()]


schemas = [[], [("a", "int64")], [("a", "float64")], [("b", "int64")],
           [("a", "int64"), ("b", "float64")], [("b", "float64"), ("a", "int64")],
           [("a", "int64"), ("a", "float64")],
           [("a", "int64"), ("a", "float64"), ("b", "int64")],
           [("a", "int64"), ("b", "float64"), ("a", "int64")],
           [("b", "int64"), ("a", "float64"), ("a", "int64")],
           [("datetime", "int64")]]
cases = []
left_schemas = [(schema, False) for schema in schemas] + [([], True)]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    for (left, explicit_empty_columns), right, n, m, name, index_kind in itertools.product(left_schemas, schemas[:-1], range(3), range(3), [None, "datetime", "old"], ["int64", "datetime64[ns]", "datetime64[ns, UTC]"]):
        df = frame(left, n, explicit_empty_columns)
        df.index = pd.Index([3 - i for i in range(n)], name=name, dtype=index_kind)
        other = frame(right, m)
        other.insert(m % (len(right) + 1), "datetime", pd.Series([3 for _ in range(m)], dtype=index_kind))
        case = dict(left=left, right=right, n=n, m=m, name=name, index_kind=index_kind, explicit_empty_columns=explicit_empty_columns)
        try:
            output = dataframe_append(df, other)
            case["output"] = dict(index=values(output.index), index_kind=str(output.index.dtype), name=output.index.name,
                                  column_axis="RangeIndex" if isinstance(output.columns, pd.RangeIndex) else "ObjectIndex",
                                  columns=[[str(c), str(output.iloc[:, i].dtype), values(output.iloc[:, i])] for i, c in enumerate(output.columns)])
        except pd.errors.InvalidIndexError as error:
            case["error"] = str(error)
        cases.append(case)
print(json.dumps(cases, allow_nan=False))
