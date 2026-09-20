"""Actual-source inference for owned built-in and temporal cells in lists and records."""
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
assert hashlib.sha256(source).hexdigest() == "89267f5cfc9e38751cb2c3a37c74ca712e8c395f492d53ed02f029a66272074f"
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "dataframe_append")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "source", "exec"))
helpers = ast.parse(Path(__file__).with_name("dataframe_constructor_contract.py").read_bytes())
helpers = [n for n in helpers.body if isinstance(n, ast.FunctionDef) and n.name in {"scalar", "snapshot"}]
exec(compile(ast.Module(body=helpers, type_ignores=[]), "snapshot_helpers", "exec"))

atoms = [None, pd.NA, pd.NaT, False, True, -(2**63), -1, 0, 2**63-1, 2**63, 2**64-1,
         0.5, -0., np.nan, np.inf, -np.inf, "", "中\ud800"]
for unit in ["s", "ms", "us", "ns"]:
    for zone in [None, "UTC", "Asia/Shanghai"]:
        atoms.append(pd.Timestamp("2024-01-02", tz=zone).as_unit(unit))
    atoms.append(pd.Timedelta("1h").as_unit(unit))
atoms += [pd.Timestamp("2500-01-01"), pd.Timestamp("2500-01-01", tz="UTC"),
          pd.Timedelta(np.timedelta64(10**12, "s")), pd.Timedelta(np.timedelta64(-10**12, "s")),
          pd.Timestamp.min, pd.Timestamp.max, pd.Timedelta.min, pd.Timedelta.max]
cases = []
ids_list = [()] + [(i,) for i in range(len(atoms))] + list(itertools.product(range(len(atoms)), repeat=2))
for i in range(18, len(atoms)):
    for missing in [0, 1, 2, 13]:
        ids_list += [(missing, i, missing), (i, missing, i)]
for ids in ids_list:
    values = [atoms[i] for i in ids]
    before = [scalar(v) for v in values]
    columns = dict(datetime=list(range(len(values))), x=values)
    records = [dict(datetime=i, x=v) for i, v in enumerate(values)]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        built = pd.DataFrame(columns)
        constructed = snapshot(built)
        record_frame = snapshot(pd.DataFrame(records))
        if values:
            assert record_frame == constructed
        result = snapshot(dataframe_append(pd.DataFrame(), columns))
        if values:
            assert snapshot(dataframe_append(pd.DataFrame(), records)) == result
    assert [scalar(v) for v in values] == before
    cases.append(dict(ids=ids, construction=constructed, records=record_frame,
                      output=result, warnings=[[type(w.message).__name__, str(w.message)] for w in captured]))
digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
omitted = []
for value in atoms[18:]:
    for reverse in [False, True]:
        for empty in [False, True]:
            first = dict(label="x", x=value, datetime=1) if reverse else dict(x=value, datetime=1, label="x")
            records = [first, {} if empty else dict(datetime=2)]
            if reverse:
                records.reverse()
            before = [[[k, scalar(v)] for k, v in row.items()] for row in records]
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                constructed = snapshot(pd.DataFrame(records))
                result = snapshot(dataframe_append(pd.DataFrame(), records))
            assert [[[k, scalar(v)] for k, v in row.items()] for row in records] == before
            omitted.append(dict(records=before,
                                construction=constructed, output=result,
                                warnings=[[type(w.message).__name__, str(w.message)] for w in captured]))
omitted_digest = hashlib.sha256(json.dumps(omitted, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, atoms=[scalar(v) for v in atoms], cases=cases, digest=digest,
                      omitted=omitted, omitted_digest=omitted_digest)))
