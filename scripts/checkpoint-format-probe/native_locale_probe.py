"""Compare an isolated Rust UCRT snapshot experiment with real Python locales.

Run with --abi-exe pointing to native_locale_abi.c compiled against the host SDK.
Neither the Rust experiment nor this script is a production locale provider.
"""
import argparse
import json
import locale
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abi-exe", required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    original = locale.setlocale(locale.LC_NUMERIC)
    cases = []
    try:
        for name in ["C", "en-US", "de-DE", "fr-FR", "hi-IN",
                     "ar-SA", "fa-IR", "ru-RU", "bn-IN", "en-US.UTF-8"]:
            locale.setlocale(locale.LC_NUMERIC, name)
            fields = locale.localeconv()
            cases.append(dict(name=name, metadata=dict(
                decimal=fields["decimal_point"], separator=fields["thousands_sep"],
                grouping=fields["grouping"])))
    finally:
        locale.setlocale(locale.LC_NUMERIC, original)
    abi = json.loads(subprocess.run([args.abi_exe], text=True, capture_output=True,
                                    check=True, timeout=30).stdout)
    result = subprocess.run(
        ["cargo", "run", "--quiet", "--manifest-path", str(root / "Cargo.toml"),
         "--bin", "locale-native"], input=json.dumps(cases), text=True,
        capture_output=True, check=True, timeout=120,
    )
    summary = json.loads(result.stdout)
    assert summary["abi"] == abi, (summary["abi"], abi)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
