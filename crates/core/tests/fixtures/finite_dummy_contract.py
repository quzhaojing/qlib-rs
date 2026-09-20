import ast
import json
import sys


source_path = sys.argv[1]
tree = ast.parse(open(source_path, encoding="utf-8").read(), filename=source_path)

dummy = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "FiniteDummyVectorEnv"
)
vectorize = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "vectorize_env"
)


class MockVector:
    kind = "base"

    def __init__(self, logger, env_fns):
        self.logger = logger
        self.env_fns = env_fns
        self.environments = [factory() for factory in env_fns]


class FiniteDummyVectorEnv(MockVector):
    kind = "dummy"


class FiniteSubprocVectorEnv(MockVector):
    kind = "subproc"


class FiniteShmemVectorEnv(MockVector):
    kind = "shmem"


module = ast.Module(
    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), vectorize],
    type_ignores=[],
)
namespace = {
    "Dict": dict,
    "FiniteDummyVectorEnv": FiniteDummyVectorEnv,
    "FiniteSubprocVectorEnv": FiniteSubprocVectorEnv,
    "FiniteShmemVectorEnv": FiniteShmemVectorEnv,
}
exec(compile(ast.fix_missing_locations(module), source_path, "exec"), namespace)

logger = object()
factory_calls = []
factory_ids = []


def factory():
    factory_calls.append(len(factory_calls))
    factory_ids.append(id(factory))
    return object()


selected = []
call_counts = []
logger_ids = []
for kind in ("dummy", "subproc", "shmem"):
    before = len(factory_calls)
    result = namespace["vectorize_env"](factory, kind, 3, logger)
    selected.append(result.kind)
    call_counts.append(len(factory_calls) - before)
    logger_ids.append(id(result.logger))

try:
    namespace["vectorize_env"](factory, "invalid", 1, logger)
except KeyError as error:
    invalid_key = error.args[0]

print(
    json.dumps(
        {
            "dummy_bases": [base.id for base in dummy.bases],
            "dummy_body": [type(node).__name__ for node in dummy.body],
            "selected": selected,
            "factory_calls": call_counts,
            "same_factory_reference": len(set(factory_ids)) == 1,
            "same_logger_reference": len(set(logger_ids)) == 1 and logger_ids[0] == id(logger),
            "invalid_key": invalid_key,
        }
    )
)
