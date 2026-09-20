"""Execute unchanged Qlib set_weight with ordered, identity-aware loader spies."""
import ast
from collections import OrderedDict
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    tree = ast.parse(source.read())
function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "set_weight")
namespace = {}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), sys.argv[1], "exec"), namespace)


def run(names, outcomes, hook):
    pool = [{"id": i} for i in range(len(names) + 1)]
    weights = OrderedDict(zip(names, pool))
    weights._metadata = {"loads": 0}
    metadata = weights._metadata
    calls = []

    def entries():
        return [[name, value["id"]] for name, value in weights.items()]

    class Loader:
        def load_state_dict(self, state):
            assert state is weights and state._metadata is metadata
            calls.append(entries())
            state._metadata["loads"] += 1
            if len(calls) == 1:
                if hook == "insert":
                    state["hook"] = pool[-1]
                elif hook == "remove" and state:
                    del state[next(iter(state))]
            outcome = outcomes[len(calls) - 1]
            if outcome == "runtime":
                raise RuntimeError("runtime")
            if outcome == "other":
                raise IndexError("other")

    error = None
    try:
        namespace["set_weight"](Loader(), weights)
    except (RuntimeError, IndexError) as failure:
        error = "runtime" if isinstance(failure, RuntimeError) else "other"
    assert all(value is pool[value["id"]] for value in weights.values())
    assert weights._metadata is metadata
    return {"names": names, "outcomes": outcomes, "hook": hook,
            "calls": calls, "final": entries(), "loads": metadata["loads"], "error": error}


name_cases = [[], ["a"], ["a", "b"], ["a", "_actor_critic.a"],
              ["_actor_critic.a", "a"],
              ["a", "_actor_critic.a", "_actor_critic._actor_critic.a"],
              ["", "模型\0weight", "__metadata__"]]
outcome_cases = [["ok"], ["runtime", "ok"], ["runtime", "runtime"],
                 ["other"], ["runtime", "other"]]
print(json.dumps([run(names, outcomes, hook) for names in name_cases
                  for outcomes in outcome_cases for hook in ("none", "insert", "remove")]))
