import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path


EXPECTED_SHA256 = "4adc4cea4e29287c98967e143bc9a143da561d33d3539e774dff44636486949f"


def load_source(path):
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(f"unexpected qlib.model.utils source hash: {digest}")

    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    torch_data.Dataset = Dataset
    torch_utils.data = torch_data
    torch.utils = torch_utils
    previous = {name: sys.modules.get(name) for name in ("torch", "torch.utils", "torch.utils.data")}
    sys.modules.update({"torch": torch, "torch.utils": torch_utils, "torch.utils.data": torch_data})
    try:
        spec = importlib.util.spec_from_file_location("qlib_model_utils_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                del sys.modules[name]
            else:
                sys.modules[name] = value
    return module, Dataset, digest


class ProbeError(RuntimeError):
    pass


class ProbeDataset:
    def __init__(self, name, values, events, len_error=None, get_error=None):
        self.name = name
        self.values = values
        self.events = events
        self.len_error = len_error
        self.get_error = get_error

    def __len__(self):
        self.events.append(f"len:{self.name}")
        if self.len_error is not None:
            raise ProbeError(self.len_error)
        return len(self.values)

    def __getitem__(self, index):
        self.events.append(f"get:{self.name}:{index}")
        if self.get_error == index:
            raise ProbeError(f"get failure {self.name} {index}")
        return self.values[index]


class EchoDataset:
    def __len__(self):
        return 7

    def __getitem__(self, index):
        return index


def failure(call):
    try:
        call()
    except Exception as error:
        return {"type": type(error).__name__, "message": str(error)}
    raise AssertionError("expected failure")


def main():
    source = Path(sys.argv[1])
    module, dataset_base, digest = load_source(source)

    events = []
    first_item = {"value": "a1"}
    second_item = {"value": "b1"}
    first = ProbeDataset("a", [{"value": "a0"}, first_item, {"value": "a2"}], events)
    second = ProbeDataset("b", [{"value": "b0"}, second_item], events)
    concat = module.ConcatDataset(first, second)
    normal = concat[1]
    identity = [normal[0] is first_item, normal[1] is second_item]
    first_item["value"] = "a1-mutated"
    normal_after_mutation = [item["value"] for item in normal]
    normal_events = list(events)

    events.clear()
    negative = [item["value"] for item in concat[-1]]
    negative_events = list(events)

    events.clear()
    length = len(concat)
    length_events = list(events)

    events.clear()
    get_failure = failure(lambda: module.ConcatDataset(first, ProbeDataset("bad", [], events, get_error=1))[1])
    get_failure_events = list(events)

    events.clear()
    length_failure = failure(lambda: len(module.ConcatDataset(first, ProbeDataset("bad", [], events, len_error="len failure bad"))))
    length_failure_events = list(events)

    events.clear()
    out_of_range = failure(lambda: concat[9])
    out_of_range_events = list(events)

    empty = module.ConcatDataset()
    empty_get = empty[123]
    empty_len = failure(lambda: len(empty))

    events.clear()
    sampler = module.IndexSampler(first)
    sampler_item, sampler_index = sampler[-1]
    sampler_events = list(events)
    events.clear()
    sampler_failure = failure(lambda: sampler[9])
    sampler_failure_events = list(events)
    events.clear()
    sampler_length = len(sampler)
    sampler_length_events = list(events)

    opaque_index = object()
    slice_index = slice(None, None, -1)
    echo_concat = module.ConcatDataset(EchoDataset(), EchoDataset())
    opaque_result = echo_concat[opaque_index]
    slice_result = echo_concat[slice_index]
    echo_sampler = module.IndexSampler(EchoDataset())
    echo_item, echo_index = echo_sampler[opaque_index]

    replacement = module.ConcatDataset(first)
    replacement.datasets = (second, first)
    replacement_values = [item["value"] for item in replacement[0]]
    replacement_length = len(replacement)
    replacement_sampler = module.IndexSampler(first)
    replacement_sampler.sampler = second
    replacement_sampler_value, replacement_sampler_index = replacement_sampler[0]

    print(json.dumps({
        "source_sha256": digest,
        "surface": sorted(name for name in module.__dict__ if not name.startswith("__")),
        "concat_base_is_dataset": module.ConcatDataset.__bases__ == (dataset_base,),
        "index_sampler_base": module.IndexSampler.__bases__[0].__name__,
        "constructor_tuple": type(concat.datasets).__name__,
        "constructor_identity": [concat.datasets[0] is first, concat.datasets[1] is second],
        "normal_tuple": type(normal).__name__,
        "normal_identity": identity,
        "normal_after_mutation": normal_after_mutation,
        "normal_events": normal_events,
        "negative": negative,
        "negative_events": negative_events,
        "length": length,
        "length_events": length_events,
        "get_failure": get_failure,
        "get_failure_events": get_failure_events,
        "length_failure": length_failure,
        "length_failure_events": length_failure_events,
        "out_of_range": out_of_range,
        "out_of_range_events": out_of_range_events,
        "empty_get_type": type(empty_get).__name__,
        "empty_get": list(empty_get),
        "empty_len": empty_len,
        "sampler_identity": sampler_item is first.values[-1],
        "sampler_value": sampler_item["value"],
        "sampler_index": sampler_index,
        "sampler_events": sampler_events,
        "sampler_failure": sampler_failure,
        "sampler_failure_events": sampler_failure_events,
        "sampler_length": sampler_length,
        "sampler_length_events": sampler_length_events,
        "opaque_index_identity": [item is opaque_index for item in opaque_result],
        "slice_index_identity": [item is slice_index for item in slice_result],
        "sampler_index_identity": [echo_item is opaque_index, echo_index is opaque_index],
        "replacement_values": replacement_values,
        "replacement_length": replacement_length,
        "replacement_sampler_value": replacement_sampler_value["value"],
        "replacement_sampler_index": replacement_sampler_index,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
