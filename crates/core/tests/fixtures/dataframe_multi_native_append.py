"""Actual append matrix for currently representable ordinary/MultiIndex levels.

The complete family fixture remains unchanged. Categorical and nullable level
metadata need their own import adapters and are not represented by this matrix.
"""
import ast
from pathlib import Path

native_fixture = Path(__file__).with_name("dataframe_multi_index_contract.py")
native_tree = ast.parse(native_fixture.read_bytes())
native_start = next(i for i, node in enumerate(native_tree.body) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "descriptors" for t in node.targets))
native_pairs = next(i for i, node in enumerate(native_tree.body) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=native_tree.body[:native_start], type_ignores=[]), str(native_fixture), "exec"))
samples.pop("multi_category")
samples.pop("multi_nullable")
samples.update(
    scalar_datetime=pd.date_range("2024-01-01", periods=2, name="datetime").as_unit("ns"),
    scalar_utc=pd.date_range("2024-01-01", periods=2, tz="UTC", name="datetime").as_unit("ns"),
    scalar_timedelta=pd.timedelta_range("1h", periods=2, freq="h", name="datetime").as_unit("ns"),
    scalar_datetime_missing=pd.DatetimeIndex(["2024-01-01", None], name="datetime").as_unit("ns"),
    scalar_utc_missing=pd.DatetimeIndex(["2024-01-01", None], tz="UTC", name="datetime").as_unit("ns"),
    scalar_timedelta_missing=pd.TimedeltaIndex(["1h", None], name="datetime").as_unit("ns"),
)
constructors = []
exec(compile(ast.Module(body=native_tree.body[native_pairs:], type_ignores=[]), str(native_fixture), "exec"))
