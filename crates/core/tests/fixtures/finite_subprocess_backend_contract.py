import ast
import json
import sys


# Batch behavior frozen from the official Tianshou v0.4.10 tagged source:
# https://github.com/thu-ml/tianshou/blob/v0.4.10/tianshou/env/venvs.py
# https://github.com/thu-ml/tianshou/blob/v0.4.10/tianshou/env/worker/subproc.py


class Worker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.pending = []
        self.events = []
        self.closed = False

    def send(self, action):
        self.events.append(["send", self.worker_id, action])
        self.pending.append(
            f"reset-{self.worker_id}-{len(self.pending)}"
            if action is None
            else f"step-{self.worker_id}-{action}"
        )

    def recv(self):
        self.events.append(["recv", self.worker_id])
        return self.pending.pop(0)

    def close(self):
        self.events.append(["close-send", self.worker_id])
        self.events.append(["close-recv", self.worker_id])
        self.events.append(["join", self.worker_id])
        self.closed = True


def reset(workers, ids):
    for worker_id in ids:
        workers[worker_id].send(None)
    return [workers[worker_id].recv() for worker_id in ids]


def step(workers, actions, ids):
    assert len(actions) == len(ids)
    for index, worker_id in enumerate(ids):
        workers[worker_id].send(actions[index])
    return [workers[worker_id].recv() for worker_id in ids]


def close(workers):
    for worker in workers:
        worker.close()


source_path = sys.argv[1]
tree = ast.parse(open(source_path, encoding="utf-8").read(), filename=source_path)
subproc = next(
    item
    for item in tree.body
    if isinstance(item, ast.ClassDef) and item.name == "FiniteSubprocVectorEnv"
)

workers = [Worker(0), Worker(1)]
reset_result = reset(workers, [1, 0, 0])
reset_events = workers[0].events + workers[1].events

workers = [Worker(0), Worker(1)]
step_result = step(workers, [7, 8, 9], [1, 0, 1])
step_events = workers[0].events + workers[1].events

workers = [Worker(0), Worker(1)]
try:
    step(workers, [1], [0, 1])
except Exception as error:
    action_mismatch = type(error).__name__

close(workers)
close_events = workers[0].events + workers[1].events

print(
    json.dumps(
        {
            "bases": [base.id for base in subproc.bases],
            "body": [type(item).__name__ for item in subproc.body],
            "reset_result": reset_result,
            "reset_events": reset_events,
            "step_result": step_result,
            "step_events": step_events,
            "action_mismatch": action_mismatch,
            "close_events": close_events,
        }
    )
)
