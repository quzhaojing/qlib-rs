"""Reproduce candidate-library evaluation; this is NOT a production acceptance gate."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--details", action="store_true", help="include every mismatch")
modes = parser.add_mutually_exclusive_group()
modes.add_argument("--large-precision", action="store_true", help="probe the remaining precision-above-9999 boundary")
modes.add_argument("--fractional-grouping", action="store_true", help="probe Python 3.14 fractional grouping")
args = parser.parse_args()
directory = Path(__file__).resolve().parent
root = directory.parent.parent
oracle = root / "crates/core/tests/fixtures/rl_checkpoint_filename_contract.py"
cases = subprocess.check_output([sys.executable, str(oracle), "--scalar-probes"])
if args.large_precision:
    probes = []
    values = [0.0, -0.0, 1.5, 2.675, 5e-324, 1e-308, 1.7976931348623157e308, float("inf"), float("nan")]
    specs = [".10000f", ".10000e", ".10000g", ".20000f", ".10000%", ",.10000f", "+020005.10000f", ".10000", "z.10000f"]
    for value in values:
        for spec in specs:
            try:
                output, error = format(value, spec), None
            except Exception as exc:
                output, error = None, type(exc).__name__
            probes.append(dict(value=["float", str(value)], format=spec, output=output, error=error))
    cases = json.dumps(probes).encode("utf-8")
if args.fractional_grouping:
    probes = []
    values = [("float", value) for value in [1.23456789, 12345.678901, -0.0, 1.23456789e-7, float("inf")]]
    values += [("int", 1234), ("bool", True), ("text", "é😀"), ("null", None)]
    specs = [".6_f", ".6,f", "_.6_f", ",.6_f", "_.6,f", ",.6,f", ".4_g", ".9_e", ".6_%", "20._f", ".3_s", "._s", "._d", "._x", ".2_f", ".6_n", "._", ".,", "20.,", "020,.6_f", "e<25.6_f", "😀^25.6_f"]
    for kind, value in values:
        text = str(value).lower() if kind == "bool" else str(value)
        for spec in specs:
            try:
                output, error = format(value, spec), None
            except Exception as exc:
                output, error = None, type(exc).__name__
            probes.append(dict(value=[kind, text], format=spec, output=output, error=error))
    cases = json.dumps(probes).encode("utf-8")
result = subprocess.run(
    ["cargo", "run", "--quiet", "--locked", "--manifest-path", str(directory / "Cargo.toml")],
    input=cases, capture_output=True, check=True,
)
reports = json.loads(result.stdout)
if args.details:
    print(json.dumps(reports, ensure_ascii=True))
else:
    print(json.dumps([dict(candidate=r["candidate"], cases=r["cases"], matched=r["matched"],
                           mismatches=len(r["mismatches"]), panics=r["panics"])
                      for r in reports], ensure_ascii=True))
