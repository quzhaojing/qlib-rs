"""Actual-source built-in record assembly, omission and column discovery contracts."""
import ast
import hashlib
import itertools
import json
import math
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

source = Path(sys.argv[1]).read_bytes()
source_hash = hashlib.sha256(source).hexdigest()
assert source_hash == "89267f5cfc9e38751cb2c3a37c74ca712e8c395f492d53ed02f029a66272074f"
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "dataframe_append")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "source", "exec"))
helpers = ast.parse(Path(__file__).with_name("dataframe_constructor_contract.py").read_bytes())
helpers = [n for n in helpers.body if isinstance(n, ast.FunctionDef) and n.name in {"scalar", "snapshot"}]
assert len(helpers) == 2
exec(compile(ast.Module(body=helpers, type_ignores=[]), "snapshot_helpers", "exec"))

atoms = [None, pd.NA, pd.NaT, False, True, -(2**63), -1, 0, 2**63-1, 2**63, 2**64-1,
         0.5, -0., np.nan, np.inf, -np.inf, "", "中\ud800"]
cases = []


def encode(records):
    return [[[key, scalar(value)] for key, value in row.items()] for row in records]


def execute(records):
    before = encode(records)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        constructed = snapshot(pd.DataFrame(records))
    assert not captured
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            outcome = dict(output=snapshot(dataframe_append(pd.DataFrame(), records)))
        except Exception as error:
            outcome = dict(error=type(error).__name__, message=str(error))
    assert encode(records) == before
    cases.append(dict(records=before, construction=constructed,
                      warnings=[[type(w.message).__name__, str(w.message)] for w in captured], **outcome))


for left, right, layout in itertools.product(range(19), range(19), range(3)):
    first = {} if left == 18 else dict(x=atoms[left])
    first["datetime"] = 9
    second = dict(y="y", datetime=3)
    if right != 18:
        second["x"] = atoms[right]
    records = [first, second]
    if layout == 1:
        records.reverse()
    elif layout == 2:
        records.insert(1, {})
    execute(records)
for atom in atoms:
    execute([dict(x=atom, datetime=1)])
for records in [[], [{}], [{}, {}], [dict(x=1)], [{}, dict(datetime=1, x=2), {}]]:
    execute(records)

digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, cases=cases)))
