"""Execute the actual upstream export method on caller-supplied numeric histories."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

path = Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indicator")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "generate_trade_indicators_dataframe")
namespace = {"pd": pd}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
             str(path), "exec"), namespace)
results = []
for case in json.loads(sys.argv[2]):
    history = {pd.Timestamp(time): dict(row) for time, row in case}
    frame = namespace[method.name](SimpleNamespace(trade_indicator_his=history))
    results.append({"columns": list(frame.columns),
                    "index": [str(t) for t in frame.index],
                    "data": json.loads(frame.to_json(orient="values")),
                    "dtypes": [str(t) for t in frame.dtypes],
                    "index_name": frame.index.name})
print(json.dumps(results, allow_nan=False))
