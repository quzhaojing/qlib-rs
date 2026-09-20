import ast
import copy
import json
import sys


class LogLevel:
    DEBUG = 10
    PERIODIC = 20


def load_step(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EnvWrapper")
    method = copy.deepcopy(next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "step"))
    method.returns = None
    for argument in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
        argument.annotation = None
    if method.args.vararg is not None:
        method.args.vararg.annotation = None
    if method.args.kwarg is not None:
        method.args.kwarg.annotation = None
    shell = ast.ClassDef(
        name="ExtractedEnv",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    namespace = {
        "InfoDict": lambda **values: values,
        "LogLevel": LogLevel,
        "RuntimeError": RuntimeError,
    }
    exec(compile(module, path, "exec"), namespace)
    return namespace["ExtractedEnv"]


ExtractedEnv = load_step(sys.argv[1])


class Logger:
    def __init__(self, events, minimum=LogLevel.DEBUG):
        self.events = events
        self.minimum = minimum
        self.logged = {}

    def reset(self):
        self.events.append("reset")
        self.logged = {}

    def _add(self, name, value, level):
        self.events.append("log:" + name)
        if level < self.minimum:
            return
        if name in self.logged:
            raise ValueError(name)
        self.logged[name] = [int(level), value]

    def add_scalar(self, name, value, loglevel=LogLevel.PERIODIC):
        self._add(name, float(value), loglevel)

    def add_any(self, name, value, loglevel=LogLevel.PERIODIC):
        self._add(name, value, loglevel)

    def logs(self):
        self.events.append("snapshot")
        return copy.deepcopy(self.logged)


class Simulator:
    def __init__(self, env, events, done, failure=None, conflict=None):
        self.env = env
        self.events = events
        self.done_value = done
        self.failure = failure
        self.conflict = conflict
        self.state_calls = 0
        self.value = 10

    def get_state(self):
        self.state_calls += 1
        stage = "pre_state" if self.state_calls == 1 else "post_state"
        self.events.append(stage)
        if self.failure == stage:
            raise RuntimeError(stage)
        return self.value

    def step(self, action):
        self.events.append("sim_step")
        if self.failure == "sim_step":
            raise RuntimeError("sim_step")
        self.value += action
        self.env.logger.add_scalar(self.conflict or "sim_metric", 101.0)

    def done(self):
        self.events.append("done")
        if self.failure == "done":
            raise RuntimeError("done")
        return self.done_value


class ActionInterpreter:
    def __init__(self, events, failure=None):
        self.events = events
        self.failure = failure

    def __call__(self, state, action):
        self.events.append("action")
        if self.failure == "action":
            raise RuntimeError("action")
        return action - 1


class StateInterpreter:
    def __init__(self, events, failure=None):
        self.events = events
        self.failure = failure

    def __call__(self, state):
        self.events.append("state_interp")
        if self.failure == "state_interp":
            raise RuntimeError("state_interp")
        return state


class Reward:
    def __init__(self, env, events, failure=None):
        self.env = env
        self.events = events
        self.failure = failure

    def __call__(self, state):
        self.events.append("reward_done:" + str(self.env.status["done"]).lower())
        if self.failure == "reward":
            raise RuntimeError("reward")
        self.env.logger.add_scalar("reward_metric", 7.0)
        return 7.0


class Auxiliary:
    def __init__(self, env, events, failure=None):
        self.env = env
        self.events = events
        self.failure = failure

    def __call__(self, state):
        self.events.append("aux_rewards:" + str(len(self.env.status["reward_history"])))
        if self.failure == "aux":
            raise RuntimeError("aux")
        return {"seen_step": self.env.status["cur_step"]}


def run(done=True, reward=True, auxiliary=True, minimum=LogLevel.DEBUG, failure=None, conflict=None, dead=False):
    env = ExtractedEnv()
    events = []
    env.seed_iterator = None if dead else object()
    env.status = {
        "cur_step": 0,
        "done": False,
        "initial_state": 5,
        "obs_history": [10],
        "action_history": [],
        "reward_history": [],
    }
    env.logger = Logger(events, minimum)
    env.simulator = Simulator(env, events, done, failure, conflict)
    env.action_interpreter = ActionInterpreter(events, failure)
    env.state_interpreter = StateInterpreter(events, failure)
    env.reward_fn = Reward(env, events, failure) if reward else None
    env.aux_info_collector = Auxiliary(env, events, failure) if auxiliary else None
    try:
        observation, result_reward, result_done, info = env.step(3)
        result = {
            "ok": True,
            "output": [observation, result_reward, result_done, info],
        }
    except Exception as error:
        result = {"ok": False, "error": type(error).__name__}
    result["events"] = events
    result["status"] = env.status
    result["retained_logs"] = copy.deepcopy(env.logger.logged)
    return result


failures = {}
for stage in ["pre_state", "action", "sim_step", "done", "post_state", "state_interp", "reward", "aux"]:
    failures[stage] = run(failure=stage)
for stage, name in [
    ("log_steps", "steps_per_episode"),
    ("log_reward", "reward"),
    ("log_obs", "obs"),
    ("log_action", "policy_act"),
]:
    failures[stage] = run(conflict=name)
failures["dead"] = run(dead=True)

print(
    json.dumps(
        {
            "success": run(),
            "fallback": run(done=False, reward=False, auxiliary=False, minimum=LogLevel.PERIODIC),
            "failures": failures,
        },
        allow_nan=True,
    )
)
