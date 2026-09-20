"""Real, child-process LC_NUMERIC oracles for Checkpoint formatting."""
import itertools
import json
import locale
import pathlib
import runpy
import sys

original = runpy.run_path(str(pathlib.Path(__file__).with_name("rl_checkpoint_filename_contract.py")))


def cases():
    tags = [["int", str(v)] for v in [0, 1, -1, 123456789, -(2**300), 10**1000]]
    tags += [["float", v] for v in ["0", "-0", "12345.6789", "1.23456789e-7", "5e-324", "1.7976931348623157e308", "inf", "-inf", "nan"]]
    tags += [["bool", "true"], ["bool", "false"], ["text", "123.456"], ["null", ""]]
    specs = ["n", "+n", " n", "#n", "zn", ".0n", ".1n", ".10n", ".10000n", ",n", "_n", "._n", ".6_n", "!rn", "s", ".6_f",
             "٠٢٥.١٠n", "٠>٠٢٥.١٠n", ".2147483648n", ".9223372036854775808n", "9223372036854775808n", ".n"]
    specs += [f"{prefix}{width}{precision}n" for prefix, width, precision in itertools.product(
        ["", "0", "+0", "0=+", "😀=+", "^", "<", "0^", "0<", "😀>", "e<", ",^", "_^", ".^", "+z#0"],
        ["0", "3", "8", "12", "25"], ["", ".10"])]
    output = []
    for name in ["C", "en-US", "de-DE", "fr-FR", "hi-IN"]:
        locale.setlocale(locale.LC_NUMERIC, name)
        data = locale.localeconv()
        metadata = dict(decimal=data["decimal_point"], separator=data["thousands_sep"], grouping=data["grouping"])
        if len(sys.argv) > 1 and sys.argv[1] == "--filenames":
            snapshots = original["source_cases"](sys.argv[2]) + original["source_cases"](sys.argv[2], extended=True)
            output.append(dict(locale=name, metadata=metadata, cases=snapshots))
            continue
        for tag, spec in itertools.product(tags, specs):
            try:
                text, error = format(original["scalar"](tag), spec), None
            except Exception as exc:
                text, error = None, type(exc).__name__
            output.append(dict(locale=name, metadata=metadata, value=tag, format=spec, output=text, error=error))
    return output


if __name__ == "__main__":
    print(json.dumps(cases(), ensure_ascii=True, allow_nan=False))
