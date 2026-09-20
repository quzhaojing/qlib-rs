"""Characterize lossless checkpoint strings and Windows paths before API migration.

All text that may contain a surrogate crosses JSON as code-point arrays, not JSON
strings: some JSON readers reject lone surrogates or merge adjacent surrogate pairs.
All filesystem mutations are confined to one auto-cleaned temporary directory.
"""
import argparse
import ast
import itertools
import json
import pathlib
import random
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace


def points(text):
    return list(map(ord, text))


def utf16(text):
    data = text.encode("utf-16-le", errors="surrogatepass")
    return [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]


def strings():
    texts = [chr(cp) for cp in range(0xd800, 0xe000)]
    texts += [chr(cp) for cp in [0, 0x7f, 0x80, 0xd7ff, 0xe000, 0xffff, 0x10000, 0x10ffff]]
    edges = ["", "a", "'", '"', "\\", "\0", "😀", "é"]
    for before, middle, after in itertools.product(
        edges, ["\ud800", "\udfff", "\ud800\udc00", "\udc00\ud800"], edges
    ):
        texts.append(before + middle + after)
    rng = random.Random(0x514c4942)
    for _ in range(1024):
        texts.append("".join(chr(rng.randrange(0x110000)) for _ in range(12)))
    return [dict(points=points(text), utf16=utf16(text), length=len(text), repr=points(repr(text)),
                 scalar=not any(0xd800 <= ord(ch) <= 0xdfff for ch in text)) for text in texts]


def filename_cases(source_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"Callback", "Checkpoint"}]
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *classes], type_ignores=[]))
    events = []

    class Clock:
        @staticmethod
        def now():
            events.append(["clock"])
            return SimpleNamespace(strftime=lambda _: "20260903123456")

    class Model:
        def __format__(self, spec):
            events.append(["format", points(spec)])
            return "\ud800<" + spec + ">"

        def __str__(self):
            events.append(["str"])
            return "\udfff"

        def __repr__(self):
            events.append(["repr"])
            return "\ud800"

        def __getattr__(self, name):
            events.append(["attribute", points(name)])
            return "\udfff"

        def __getitem__(self, key):
            events.append(["item", key if isinstance(key, int) else points(key)])
            return "\ud800\udc00"

    ns = dict(datetime=Clock)
    exec(compile(module, str(source_path), "exec"), ns)
    callback = ns["Checkpoint"].__new__(ns["Checkpoint"])
    templates = [
        "literal\ud800.pth", "{text}", "{text!s}", "{text!r}", "{text!a}",
        "{text[0]}", "{text[1]}", "{text:.1s}", "{text:\ud800^9.1s}",
        "{\ud800}", "{model.\udfff}", "{model[\ud800]}", "{model:\ud800>8}",
        "{model!s:\ud800^9}", "{model!r}", "{model!a}", "{model!\ud800}",
        "{model}{missing}", "{model}}", "{model:{spec}}", "{number:{spec}}",
        "{number:c}", "{number:\udfff>8c}", "{number:\ud800>8d}",
        "{text[999]}", "{text[\ud800]}", "{model[\ud800].\udfff}",
    ]
    result = []
    for text, number, template in itertools.product(
        ["\ud800", "\udfff", "\ud800\udc00", "\U00010000", "a\ud800😀"],
        [0xd800, 0xdfff, 0x10000], templates,
    ):
        events.clear()
        callback.filename = template
        metrics = {"text": text, "number": number, "model": Model(),
                   "spec": "\ud800>8c", "\ud800": "key\udfff"}
        trainer = SimpleNamespace(current_iter=7, metrics=metrics)
        try:
            output = points(callback._new_checkpoint_name(trainer))
            error = None
        except Exception as exc:
            output, error = None, type(exc).__name__
        result.append(dict(template=points(template), text=points(text), number=number,
                           output=output, error=error, events=events.copy()))
    return result


def disk_cases(root):
    names = ["\ud800.pth", "\udfff.pth", "\ud800\udc00.pth", "\U00010000.pth",
             "a\ud800😀.pth", "\udc00\ud800.pth", "bad\0.pth", "sub/\ud800.pth"]
    cases = []
    for index, name in enumerate(names):
        folder = root / str(index)
        folder.mkdir(parents=True)
        path = folder / name
        try:
            path.write_bytes(b"checkpoint-payload")
            assert path.read_bytes() == b"checkpoint-payload"
            listed = list(folder.iterdir())
            assert len(listed) == 1
            output, error = utf16(listed[0].name), None
        except Exception as exc:
            output, error = None, type(exc).__name__
        cases.append(dict(points=points(name), listed_utf16=output, error=error))
    alias = root / "alias"
    alias.mkdir()
    (alias / "\ud800\udc00.pth").write_bytes(b"same-wide-path")
    assert (alias / "\U00010000.pth").read_bytes() == b"same-wide-path"
    return cases


def save_cases(source_path, root, names):
    """Real Qlib ordering/paths with a non-Torch payload, not codec parity."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"Callback", "Checkpoint"}]
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *classes], type_ignores=[]))
    results = []
    for index, (case, latest_mode) in enumerate(itertools.product(names, [None, "copy", "\ud800"])):
        folder = root / str(index)
        folder.mkdir(parents=True)
        latest = folder / "latest.pth"
        latest.write_bytes(b"old")
        events = []

        def save(graph, path):
            assert graph == "live-graph"
            events.append(["save", points(str(path.relative_to(folder)))])
            path.write_bytes(b"checkpoint-payload")

        def graph():
            events.append(["graph"])
            return "live-graph"

        ns = dict(Path=pathlib.Path, os=os, shutil=shutil, torch=SimpleNamespace(save=save),
                  time=SimpleNamespace(time=lambda: 123.0),
                  datetime=SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda _: "20260903123456")))
        exec(compile(module, str(source_path), "exec"), ns)
        name = "".join(map(chr, case["points"]))
        callback = ns["Checkpoint"](folder, filename=name, save_latest=latest_mode)
        trainer = SimpleNamespace(current_iter=7, metrics={}, state_dict=graph)
        try:
            callback._save_checkpoint(trainer)
            error = None
        except Exception as exc:
            error = type(exc).__name__
        # Invalid OS paths are assigned to last_name before attempting the write.
        assert callback._last_checkpoint_name == name
        assert callback._last_checkpoint_iter == 7 and callback._last_checkpoint_time == 123.0
        assert events[0] == ["graph"] and events[1][0] == "save"
        expected_latest = (b"old" if error or latest_mode is None
                           else b"checkpoint-payload" if latest_mode == "copy" else None)
        actual_latest = latest.read_bytes() if latest.exists() else None
        assert actual_latest == expected_latest
        results.append(dict(name=points(name), latest_mode=None if latest_mode is None else points(latest_mode),
                            last_name=points(callback._last_checkpoint_name), last_iter=7, last_time=123.0,
                            error=error, events=events,
                            latest=None if actual_latest is None else actual_latest.decode("ascii")))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--oracle-only", action="store_true")
    modes.add_argument("--save-oracle-only", action="store_true")
    args = parser.parse_args()
    assert os.name == "nt", "This filesystem contract targets Windows"
    if args.save_oracle_only:
        with tempfile.TemporaryDirectory(prefix="qlib-checkpoint-save-oracle-") as folder:
            folder = pathlib.Path(folder)
            disk = disk_cases(folder / "oracle")
            print(json.dumps(save_cases(args.source, folder / "qlib", disk), ensure_ascii=True))
        return
    cases, filenames = strings(), filename_cases(args.source)
    if args.oracle_only:
        print(json.dumps(dict(strings=cases, filenames=filenames), ensure_ascii=True))
        return
    root = pathlib.Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="qlib-checkpoint-surrogate-") as folder:
        folder = pathlib.Path(folder)
        disk = disk_cases(folder / "oracle")
        saves = save_cases(args.source, folder / "qlib", disk)
        request = dict(strings=cases, filenames=filenames, disk=disk, saves=saves,
                       directory=str(folder / "rust"))
        run = subprocess.run(
            ["cargo", "run", "--quiet", "--manifest-path", str(root / "Cargo.toml"),
             "--bin", "surrogate-text"], input=json.dumps(request, ensure_ascii=True),
            text=True, capture_output=True, check=True, timeout=120,
        )
        print(run.stdout, end="")


if __name__ == "__main__":
    main()
