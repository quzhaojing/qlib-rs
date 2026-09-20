"""Python primitive __format__ behavior used by Qlib checkpoint fields.

Transport possibly non-scalar text as code-point arrays. No runtime bridge,
source edits, locale changes, filesystem writes or third-party Python packages.
"""
import itertools
import json


def case(value, spec):
    if isinstance(value, str):
        kind, encoded = "text", list(map(ord, value))
    elif isinstance(value, bool):
        kind, encoded = "bool", value
    elif isinstance(value, int):
        kind, encoded = "int", str(value)
    elif isinstance(value, float):
        kind, encoded = "float", repr(value)
    else:
        kind, encoded = "none", None
    try:
        output, error = list(map(ord, format(value, spec))), None
    except (ValueError, TypeError, OverflowError) as exc:
        output, error = None, type(exc).__name__
    return dict(kind=kind, value=encoded, spec=list(map(ord, spec)), output=output, error=error)


fills = ["x", "y", "0", "{", "\0", "é", "😀", "\ue000", "\ue001", "\ue002", "\ud800", "\udfff"]
strings = ["", "x", "yxy", "\0xy", "\ud800", "\udfff", "\ud800\udc00", "\U00010000", "x\ue000\ue001\ue002\ue003y\ud800😀"]
text_specs = ["", "s", "0", ".0s", "٠٥", "１２.２s", "n", "!r", "\ud800", ".2\ud800", "\ud800\udfff"]
text_specs += [fill + align + tail for fill, align, tail in itertools.product(fills, "<>^=", ["13s", "13.0s", "13.3s"])]
numeric_values = [-1, 0, 1234, 0xd800, 0xdfff, 0x10000, 0x110000, 10**100, True, False, None, -0.0, 1234.5, -1.25, float("inf"), float("nan")]
numeric_specs = ["", "c", "n", "\ud800", ".2\ud800", "\ud800\udfff"]
numeric_specs += [fill + align + tail for fill, align, tail in itertools.product(fills, "<>^=", ["13c", "+13n", "13.2f", "13d", "13,.3_f"])]
cases = [case(value, spec) for value, spec in itertools.product(strings, text_specs)]
cases += [case(value, spec) for value, spec in itertools.product(numeric_values, numeric_specs)]
cases += [case(point, spec) for point, spec in itertools.product(range(0xd800, 0xe000), ["c", "\udfff>3c"])]
print(json.dumps(cases, ensure_ascii=True))
