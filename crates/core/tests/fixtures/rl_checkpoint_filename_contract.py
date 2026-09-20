"""Live Qlib filename characterization; no Torch/Qlib runtime imports are required.

Standalone stdout is a JSON case matrix. The optional --scalar-probes switch emits
low-level format(value, spec) expectations for testing candidate Rust dependencies.
"""
import ast
import itertools
import json
import sys
import unicodedata
from types import SimpleNamespace


def scalar(tag):
    kind, value = tag
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "bool":
        return value == "true"
    if kind == "null":
        return None
    return value


def probes():
    integer_specs = ["", "d", "03d", "+08d", "#08x", "#b", "_", ",", "08,", "^20d", "c", ".2f", ".2e", ".2g", "%", "05", "n", "z", "s", ".2"]
    float_specs = ["", ".2f", ".0f", ".3e", ",.2e", ".4g", ".3", "%", "z.2f", "+08.2f", "#g", "_", ",", "05", "n", "d", "s", "f!", "010,.2f"]
    string_specs = ["", "s", "05", "<8", ">8", "^8", ".1", ".2", ">8.2", "+", "#", "=8", "d", ","]
    groups = [
        ([['int', str(v)] for v in [0, 1, -1, 1234, 2**127-1, -(2**127), 2**127, 2**300, -(2**300), 10**1000]], integer_specs),
        ([['float', v] for v in ["0.0", "-0.0", "1.005", "2.675", "1e-7", "1e20", "1.2345678901234567", "nan", "inf", "-inf", "5e-324"]], float_specs),
        ([['text', v] for v in ["", "abc", "中文", "é😀"]], string_specs),
        ([['bool', v] for v in ["true", "false"]], integer_specs),
        ([["null", ""]], ["", "s", "05"]),
    ]
    cases = []
    for tags, formats in groups:
        for tag, spec in itertools.product(tags, formats):
            try:
                output, error = format(scalar(tag), spec), None
            except Exception as exc:
                output, error = None, type(exc).__name__
            cases.append(dict(value=tag, format=spec, output=output, error=error))
    return cases


def extended_probes():
    """Cross type/spec boundaries, separately from the frozen candidate baseline."""
    tags = [["int", str(v)] for v in [0, -1, 42, 1234567, 2**128, -(2**128), 10**1000]]
    tags += [["float", v] for v in ["0", "-0", "1.25", "12345.75", "nan", "inf", "-inf"]]
    tags += [["bool", "true"], ["bool", "false"], ["text", "é😀"], ["null", ""]]
    specs = ["badf", "badn", "z.2f", "z.2n", ",n", "_n", ".2n", "+n", "#n", "010n",
             "#020_x", "#020_b", "#020_o", "#020_X", "020,d", "020_d", "+020,d", "=+020,d",
             "😀^16", "中>16", "😀^16.2", "0=10", "*>+16.2f", "<#12.3g", ".0", "#.0",
             "c", "+c", "#c", "=8c", "08c", "_c", ",c", ".1c", "z", "F", "G", "E",
             "+010,.2e", "#010_.0f", "z010.0f", "%", "#", "+", " ", "_", ",", "05", "05s",
             "<", ">", "=", "^", "0", "00", ".", "..2f", "1.2.3f", "!", "{}", "q",
             "_>12n", ",^12n", "中<12n", "_>12_n", ">12n", "n",
             "!r", "!s", "!a", "!b", "!r03d", "!>10", "!^10", "!<10c"]
    cases = []
    for tag, spec in itertools.product(tags, specs):
        try:
            output, error = format(scalar(tag), spec), None
        except Exception as exc:
            output, error = None, type(exc).__name__
        cases.append(dict(value=tag, format=spec, output=output, error=error))
    return cases


def large_precision_probes():
    tags = [["float", value] for value in ["0", "-0", "1.5", "2.675", "1e-5", "1e-4", "5e-324", "-5e-324", "1e-308", "1.7976931348623157e308", "nan", "inf", "-inf", "-nan"]]
    tags += [["int", str(value)] for value in [0, -1, 42, 2**300, 10**1000]]
    tags += [["bool", "true"], ["bool", "false"], ["text", "é😀"], ["null", ""]]
    specs = [f".{precision}{kind}" for precision in [9999, 10000, 10001, 20000] for kind in ["f", "F", "e", "E", "g", "G", "%", "", "n"]]
    specs += [f"{prefix}.10000{kind}" for prefix in ["#", "+z#", "020100,", "😀<20100", "😀^20101", "😀>20100", "0=+20100,", "^020101_"] for kind in ["f", "e", "g", "", "%", "n"]]
    specs += [".2147483648f", ".999999999999999999999999f", "999999999999999999999999.10000f", "0.10000q", "z#.10000d", "!r.10000f", "+.10000s", "9223372036854775808.10000f", ".9223372036854775808f"]
    cases = []
    for tag, spec in itertools.product(tags, specs):
        try:
            output, error = format(scalar(tag), spec), None
        except Exception as exc:
            output, error = None, type(exc).__name__
        cases.append(dict(value=tag, format=spec, output=output, error=error))
    import random
    import struct

    rng = random.Random(0x514c4942)
    for _ in range(128):
        number = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
        for spec in [".10000f", "+z#020100.10001g", "😀^20100.10000e", ".20000"]:
            cases.append(dict(value=["float", repr(number)], format=spec, output=format(number, spec), error=None))
    for number in [1.5, -5e-324, float("inf")]:
        for fill in ["e", "%", "0", "\n", ".", "z", "#", "9"]:
            spec = f"{fill}^20100.10000e"
            cases.append(dict(value=["float", repr(number)], format=spec, output=format(number, spec), error=None))
    return cases


def unicode_numeric_probes():
    assert unicodedata.unidata_version == "16.0.0"
    tags = [["int", "1234"], ["int", str(-(2**100))], ["float", "1.25"], ["float", "-0"], ["float", "inf"], ["bool", "true"], ["text", "é😀"], ["null", ""]]
    specs = []
    for cp in range(0x110000):
        ch = chr(cp)
        if ch.isdecimal():
            specs.extend([ch, "0" + ch, "." + ch, ch + "<8", "٠" + ch])
    specs += ["٠٥", "０５", "0٠5", "+٠٥", "z٠٥", "#٠٥", "٠", "٠٠", "٠s", "٠c",
              "٠<٠٥", "٠>٠٥", ">٠٥", "^٠٥", "=٠٥", "😀^٠١٢.٢f", "x>0٥.٢f", " ٠١٢.٢f",
              "z#٠١٢.٢f", "+z#0１２.٢f", "٠١٢,.٢f", "٠١٢_.٢f", "٠١٢,,.٢f", "٠١٢,_.٢f",
              ".١٠٠٠٠f", ".１００００g", ".００００００１００００e", "０２０１００.１００００f",
              ".٢١٤٧٤٨٣٦٤٨f", ".９２２３３７２０３６８５４７７５８０８f", "９２２３３７２０３６８５４７７５８０８",
              "９２２３３７２０３６８５４７７５８０７0", "１８４４６７４４０７３７０９５５１６１９",
              "٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩x", ".٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩x", ".é", "é", "é<", "٠..٢f",
              "٠.²f", "²", "٠.٢f٣", "!r٠٥", "z+٠٥", "٠z", "٠#", "٠e", "٠E", "٠%", "٠n", "٠d",
              "\n^٠٥", "\x00>٠٥", ".٩٢٢٣٣٧٢٠٣٦٨٥٤٧٧٥٨٠٧s", ".١٠٠٠٠f!r", "٠,.٢f", "٠_.٢f"]
    specs += [chr(cp) for cp in [0x11de0, 0x1e5f1]]
    cases = []
    for tag, spec in itertools.product(tags, specs):
        try:
            output, error = format(scalar(tag), spec), None
        except Exception as exc:
            output, error = None, type(exc).__name__
        cases.append(dict(value=tag, format=spec, output=output, error=error))
    return cases


def fractional_grouping_probes():
    assert sys.version_info[:2] == (3, 14)
    tags = [["float", value] for value in ["0", "-0", "1.23456789", "12345.678901", "-12345.678901", "1.23456789e-7", "5e-324", "1.7976931348623157e308", "nan", "inf", "-inf"]]
    tags += [["int", str(value)] for value in [0, -1, 1234, 2**300, 10**1000]]
    tags += [["bool", "true"], ["bool", "false"], ["text", "123.456789"], ["text", "é😀"], ["null", ""]]
    specs = [f"{group}.{precision}{separator}{kind}"
             for group, precision, separator, kind in itertools.product(
                 ["", ",", "_"], ["", "0", "2", "3", "4", "6", "9"], [",", "_"],
                 ["", "e", "E", "f", "F", "g", "G", "%", "d", "x", "b", "o", "X", "c", "s", "n"])]
    specs += [f"{prefix}.6{separator}{kind}"
              for prefix, separator, kind in itertools.product(
                  ["20", "020,", "0=+20,", "😀^25", "e<25", "%>25", ".=+25", "0^25,", "^025_", "<025,", "+z#25,", " #25_"],
                  [",", "_"], ["", "f", "e", "%", "g"])]
    specs += ["._,", ".,_", "._,f", ".,,f", ".__f", ".6_,f", "._q", ".6_f!", "!r.6_f",
              ".2147483648_f", ".9223372036854775808_f", "9223372036854775808._f", ".６_f", "٠٢٥.٦_f",
              "٠<٠٢٥.٦_f", "0٠٢٥,.٦_f", "._s", ".3_s", "._", ".,", "20.,"]
    cases = []
    for tag, spec in itertools.product(tags, specs):
        try:
            output, error = format(scalar(tag), spec), None
        except Exception as exc:
            output, error = None, type(exc).__name__
        cases.append(dict(value=tag, format=spec, output=output, error=error))
    # Cross the existing large-precision adapter and adversarial fill boundary.
    for tag, prefix, kind in itertools.product(
        [["float", "1.5"], ["float", "-5e-324"], ["float", "inf"], ["int", "1234"]],
        ["", "e<20100", "😀^20101", "0=+20100,", "%>20100", ".^20101"],
        ["f", "e", "g", "%", ""]):
        spec = f"{prefix}.10000_{kind}"
        try:
            output, error = format(scalar(tag), spec), None
        except Exception as exc:
            output, error = None, type(exc).__name__
        cases.append(dict(value=tag, format=spec, output=output, error=error))
    import random
    import struct

    rng = random.Random(0x514c4942)
    for _ in range(128):
        number = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
        for kind in ["", "f", "F", "e", "E", "g", "G", "%"]:
            fill = rng.choice(["😀", "e", "E", "%", ".", "0", "9", "\n", "\0", ",", "_"])
            align = rng.choice(["<", ">", "=", "^"])
            flags = rng.choice(["", "+", " ", "z", "+z#", "#"])
            spec = f"{fill}{align}{flags}{rng.randrange(45)}{rng.choice(['', ',', '_'])}.{rng.randrange(20)}{rng.choice([',', '_'])}{kind}"
            cases.append(dict(value=["float", repr(number)], format=spec, output=format(number, spec), error=None))
    return cases


def source_cases(source_path, extended=False):
    with open(source_path, encoding="utf-8") as source:
        tree = ast.parse(source.read())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"Callback", "Checkpoint"}]
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes], type_ignores=[]))
    events = []

    class Clock:
        @staticmethod
        def now():
            events.append(["clock"])
            return SimpleNamespace(strftime=lambda pattern: "20260902123456")

    class Custom:
        def __str__(self):
            events.append(["str"])
            return "value中文"

        def __repr__(self):
            events.append(["repr"])
            return "Custom中文"

        def __format__(self, spec):
            events.append(["format", spec])
            if spec == "fail":
                raise ValueError("requested format failure")
            return "custom<" + spec + ">"

        def __getitem__(self, key):
            events.append(["item", key])
            return {"x": 7, ":": 8, "!": 9}[key]

        @property
        def real(self):
            events.append(["attribute", "real"])
            return 12

    namespace = dict(datetime=Clock)
    exec(compile(module, source_path, "exec"), namespace)
    callback = namespace["Checkpoint"].__new__(namespace["Checkpoint"])
    templates = ["{iter:03d}-{reward:.2f}-{time}.pth", "{{{iter}}}", "{reward!s}", "{reward!r}", "{reward!a}", "{reward:0{width}.{precision}f}", "{time[0]}", "{data[:]}-{data[!]}-{data[name]}", "{custom[x]:03d}", "{custom[:]}-{custom[!]}", "{custom.real:03d}", "{custom}", "{custom!s}", "{custom!r}", "{custom!a}", "{custom:fail}", "{custom:{width}}", "{custom} {", "{custom} {missing}", "{custom!q}", "{custom:{missing}}", "{iter.real}", "{}", "{0}", "{missing}", "}", "{", "{iter:bogus}", "{reward:{width:{precision}}}", "{iter!s:05}", "{time:.2}", "{data[name]:>8}", "literal.pth", "{val/reward:.2f}", "{中文}"]
    cases = []
    scalar_cases = probes()
    for probe in scalar_cases:
        tag = probe["value"]
        templates_with_values = [("{reward:" + probe["format"] + "}", tag)]
        for template, tag in templates_with_values:
            templates_input = dict(template=template, iteration="7", value=tag, probe=True)
            cases.append(templates_input)
    cases.extend(dict(template=t, iteration="7", value=["float", "2.675"], probe=False) for t in templates)
    cases.extend(dict(template="{iter:03d}", iteration=str(i), value=["float", "1"], probe=False) for i in [-(2**300), 2**300])
    cases.extend(dict(template="literal", iteration="7", value=["float", "1"], probe=False, reserved=key) for key in ["iter", "time"])
    if extended:
        cases = []
        extra_templates = ["", "中文", "{{", "}}", "{{}}", "{{{{x}}}}", "{custom} }", "{custom} {x{y}",
            "{custom!}", "{custom!", "{custom!rr}", "{custom!r!s}", "{custom!r", "{custom:",
            "{custom!r:{missing}}", "{custom!s:{width}}", "{custom!a:>20}", "{custom!q:{missing}}",
            "{custom.real.real}", "{custom[x].real}", "{custom.real[]}", "{custom[x]oops}",
            "{custom[x].}", "{custom[x][}", "{custom[missing]}", "{custom.missing}",
            "{custom[]}", "{custom.}", "{custom[x]:03d} {", "{custom!\0}", "{custom:{{}}}",
            "{custom:{custom}}", "{custom:{custom!s}}", "{custom:{custom:{width}}}",
            "{custom:{custom!r:{missing}}}", "{time[999]}", "{time[-1]}", "{time[]}", "{time[x]}",
            "{time[999999999999999999999999]}", "{time[00]}", "{iter[0]}", "{iter.numerator}",
            "{iter.denominator}", "{iter.imag}", "{reward.real}", "{reward.imag}", "{reward.missing}",
            "{time!s}", "{time!r}", "{time!a}", "{data[missing]}", "{data[name]!r}",
            "{missing[x]}", "{.real}", "{[x]}", "{00}", "{data[name]:{{<8}", "{custom:}}}",
            "{time[٠١]}", "{time[𝟚]}", "{custom[１]}", "{١}", "{data[１]}", "{time[²]}",
            "{999999999999999999999999x}", "{time[999999999999999999999x]}",
            "{reward:٠٥}", "{reward:٠١٢.٢f}", "{reward:.١٠٠٠٠f}", "{iter:٠٥d}", "{iter:0٥d}",
            "{time:٠٢٠}", "{time:.٠}", "{time:٠}", "{reward:٠>٠١٢.٢f}", "{reward:>{width}.٢f}",
            "{reward:_.6_f}", "{reward:,.6,f}", "{reward:.9_e}", "{reward:😀^25.6_f}",
            "{reward:020,.6_f}", "{reward:>{width}.{precision}_f}", "{reward:.１００００_f}",
            "{reward:._n}", "{iter:._d}", "{time:.3_s}", "{custom:.6_f}", "{reward!s:.6_s}"]
        cases.extend(dict(template=t, iteration="7", value=["float", "2.675"], probe=False) for t in extra_templates)
        for tag in [["text", "a'b"], ["text", 'a"b'], ["text", "é😀\n\u00a0\u200b"],
                    ["int", str(2**300)], ["bool", "true"], ["bool", "false"], ["null", ""], ["float", "-0.0"]]:
            cases.extend(dict(template=t, iteration="7", value=tag, probe=False)
                         for t in ["{reward!s}", "{reward!r}", "{reward!a}", "{reward.real}", "{reward.imag}",
                                   "{reward.numerator}", "{reward.denominator}", "{reward[0]}"])
    output = []
    for case in cases:
        events.clear()
        callback.filename = case["template"]
        metrics = dict(reward=scalar(case["value"]), width=8, precision=2, data={":": 11, "!": 13, "name": "中文"}, custom=Custom(), **{"val/reward": 1.25, "中文": "value"})
        if "reserved" in case:
            metrics[case["reserved"]] = 999
        trainer = SimpleNamespace(current_iter=int(case["iteration"]), metrics=metrics)
        try:
            result, error = callback._new_checkpoint_name(trainer), None
        except Exception as exc:
            result, error = None, type(exc).__name__
        output.append(dict(spec=case, output=result, error=error, events=events.copy()))
    # A source method change must not silently invalidate the lower-level candidate inputs.
    if not extended:
        for actual, expected in zip(output, scalar_cases):
            assert (actual["output"], actual["error"]) == (expected["output"], expected["error"])
    return output


if __name__ == "__main__":
    if sys.argv[1] == "--repr-probes":
        import itertools

        assert unicodedata.unidata_version == "16.0.0"
        characters = "".join(chr(cp) for cp in range(0x110000) if not 0xd800 <= cp <= 0xdfff)
        symbols = ["", "'", '"', "\\", "\n", "é", "\u0560", "\U0001fae9", "\u0378", "\u00a0"]
        mixed = ["".join(parts) for parts in itertools.product(symbols, repeat=3)]
        mixed.extend(characters[start:start + 8192] for start in range(0, len(characters), 8192))
        cases = dict(single=[repr(ch) for ch in characters], mixed=[[text, repr(text), ascii(text)] for text in mixed])
    elif sys.argv[1] == "--decimal-probes":
        assert unicodedata.unidata_version == "16.0.0"
        cases = [[cp, unicodedata.decimal(chr(cp))] for cp in range(0x110000) if chr(cp).isdecimal()]
    elif sys.argv[1] == "--unicode-numeric-probes":
        cases = unicode_numeric_probes()
    elif sys.argv[1] == "--fractional-grouping-probes":
        cases = fractional_grouping_probes()
    elif sys.argv[1] == "--large-scalar-probes":
        cases = large_precision_probes()
    elif sys.argv[1] == "--extended-scalar-probes":
        cases = extended_probes()
    elif sys.argv[1] == "--extended-template-probes":
        cases = source_cases(sys.argv[2], extended=True)
    else:
        cases = probes() if sys.argv[1] == "--scalar-probes" else source_cases(sys.argv[1])
    print(json.dumps(cases, ensure_ascii=True, allow_nan=False))
