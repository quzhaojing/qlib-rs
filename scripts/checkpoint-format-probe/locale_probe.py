"""Run real Python LC_NUMERIC characterization or the isolated grouping candidate."""
import argparse
import itertools
import json
import locale
import pathlib
import subprocess


def grouping_cases():
    cases = []
    for groups, separator, length in itertools.product(
        [[], [1], [3], [3, 2], [4], [1, 2, 3, 4, 5]],
        [",", ".", "\u202f", "😀::"],
        range(1, 41),
    ):
        digits = "1234567890" * (length // 10) + "1234567890"[:length % 10]
        # The installed stdlib grouping routine reads this metadata via localeconv.
        original = locale._override_localeconv
        locale._override_localeconv = dict(grouping=groups + [0] if groups else [], thousands_sep=separator)
        try:
            expected = locale._group(digits)[0]
        finally:
            locale._override_localeconv = original
        cases.append(dict(groups=groups, separator=separator, digits=digits, output=expected))
    return cases


def numeric_cases():
    records = []
    for name in ["C", "en-US", "de-DE", "fr-FR", "hi-IN"]:
        locale.setlocale(locale.LC_NUMERIC, name)
        metadata = locale.localeconv()
        for value, spec in itertools.product(
            [0, -123456789, 10**100, True, False, 0.0, -0.0, 12345.6789,
             1.23456789e-7, float("inf"), float("nan"), "12345", None],
            ["n", ".0n", ".10n", "+020n", "020n", "0=+25.10n", "😀^30.10n", "0^30.10n",
             ",n", "_n", "._n", "z.10n", "#.10n", "._f", "_>25n", ",<25n"],
        ):
            try:
                output, error = format(value, spec), None
            except Exception as exc:
                output, error = None, type(exc).__name__
            records.append(dict(locale=name, decimal=metadata["decimal_point"],
                                separator=metadata["thousands_sep"], grouping=metadata["grouping"],
                                value=repr(value), type=type(value).__name__, format=spec,
                                output=output, error=error))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouping", action="store_true")
    args = parser.parse_args()
    if args.grouping:
        root = pathlib.Path(__file__).resolve().parent
        completed = subprocess.run(
            ["cargo", "run", "--quiet", "--manifest-path", str(root / "Cargo.toml"), "--bin", "locale-grouping"],
            input=json.dumps(grouping_cases()), text=True, capture_output=True, check=True,
        )
        print(completed.stdout, end="")
    else:
        print(json.dumps(numeric_cases(), ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
