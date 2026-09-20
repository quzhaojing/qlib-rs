"""Characterize the unchanged Qlib selector; no Torch codec is emulated here."""
import ast
from collections import OrderedDict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

with open(sys.argv[1], encoding="utf-8") as source:
    tree = ast.parse(source.read())
trainer = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "Trainer")
function = next(node for node in trainer.body
                if isinstance(node, ast.FunctionDef) and node.name == "get_policy_state_dict")
# Remove only the class descriptor wrapper; execute the original function body.
function.decorator_list = []
namespace = {"Path": Path, "OrderedDict": OrderedDict}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
             sys.argv[1], "exec"), namespace)


def run(document, wrapped, failure=None):
    events = []
    path = Path("explicit-native-schema.not-a-codec")

    class Traced(OrderedDict):
        def __contains__(self, key):
            events.append(["contains", key])
            return super().__contains__(key)

        def __getitem__(self, key):
            events.append(["get", key])
            return super().__getitem__(key)

    state = Traced(document)
    if wrapped and isinstance(document["vessel"], dict):
        state["vessel"] = Traced(document["vessel"])
    expected = document["vessel"].get("policy") if wrapped and isinstance(document["vessel"], dict) else state

    def load(actual_path, *, map_location):
        assert actual_path is path
        assert map_location == "cpu"
        events.append(["load", "cpu"])
        if failure is OSError:
            raise OSError("decode failed before selection")
        return state

    namespace["torch"] = SimpleNamespace(load=load)
    try:
        result = namespace["get_policy_state_dict"](path)
    except (KeyError, TypeError, OSError) as error:
        assert failure is type(error), (failure, error)
    else:
        assert failure is None
        assert result is expected  # No copying or truth-value fallback.
    expected_events = [["load", "cpu"]]
    if failure is not OSError:
        expected_events.append(["contains", "vessel"])
        if wrapped:
            expected_events.append(["get", "vessel"])
            if isinstance(document["vessel"], dict):
                expected_events.append(["get", "policy"])
    assert events == expected_events, events
    if failure is None:
        return {"document": document, "trainer": wrapped, "expected": result}


cases = [run(document, False) for document in ({}, {"weight": [1, 2]},
          {"policy": None}, {"vessel.weight": [3]})]
cases += [run({"vessel": {"policy": payload}}, True)
          for payload in (None, {}, [1, 2], {"模型\0weight": [3], "_metadata": {"version": 1}})]
run({"vessel": {}}, True, KeyError)
run({"vessel": None}, True, TypeError)
run({"vessel": {"policy": [1]}}, True, OSError)
print(json.dumps(cases))
