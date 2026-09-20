"""Execute unchanged Qlib DataQueue context methods without starting a producer."""
import ast
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
queue = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "DataQueue")
queue.bases = []
queue.decorator_list = []
queue.body = [node for node in queue.body if isinstance(node, ast.FunctionDef) and node.name in ("__enter__", "__exit__")]
assert {node.name for node in queue.body} == {"__enter__", "__exit__"}
module = ast.Module(body=ast.parse("from __future__ import annotations").body + [queue], type_ignores=[])
namespace = {}
exec(compile(ast.fix_missing_locations(module), sys.argv[1], "exec"), namespace)


def case(fail_enter, fail_body):
    events = []
    instance = namespace["DataQueue"]()

    def activate():
        events.append("activate")
        if fail_enter:
            raise RuntimeError("entry")

    def cleanup():
        events.append("cleanup")

    instance.activate = activate
    instance.cleanup = cleanup
    identity = None
    failure = None
    try:
        with instance as entered:
            identity = entered is instance
            events.append("body")
            if fail_body:
                raise RuntimeError("phase")
    except RuntimeError as error:
        failure = str(error)
    return {"events": events, "identity": identity, "failure": failure}


print(json.dumps([case(False, False), case(False, True), case(True, False)]))
