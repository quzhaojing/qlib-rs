"""Actual-source MultiIndex factoring, directional append and invalid descriptors."""
import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path

fixture = Path(__file__).with_name("dataframe_index_families.py")
tree = ast.parse(fixture.read_bytes())
stop = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pairs" for t in node.targets))
exec(compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(fixture), "exec"))


def object_index(values):
    return pd.Index(values, dtype=object, tupleize_cols=False, name="datetime")


samples = dict(
    multi_one=pd.MultiIndex.from_tuples([("b",), ("a",)], names=["asset"]),
    multi_two=pd.MultiIndex.from_tuples([("b", 2), ("a", 1)], names=["asset", "time"]),
    multi_three=pd.MultiIndex.from_tuples([("b", 2, False), ("a", 1, True)]),
    multi_duplicate=pd.MultiIndex.from_tuples([("a", 1), ("a", 1)]),
    multi_missing=pd.MultiIndex(levels=[["b", "a", "unused"], [2, 1, 99]],
                               codes=[[0, -1, 1], [-1, 1, 0]], names=["asset", "time"]),
    multi_all_missing=pd.MultiIndex(levels=[[], []], codes=[[-1, -1], [-1, -1]]),
    multi_empty=pd.MultiIndex(levels=[["unused"], [99]], codes=[[], []]),
    multi_empty_one=pd.MultiIndex(levels=[["unused"]], codes=[[]]),
    multi_empty_three=pd.MultiIndex(levels=[["unused"], [99], [True]], codes=[[], [], []]),
    multi_sorted=pd.MultiIndex(levels=[["a", "b"], [1, 2]], codes=[[0, 1], [0, 1]], sortorder=2),
    multi_tuple_names=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=[("asset", 1), 7]),
    multi_nested=pd.MultiIndex.from_tuples([(("a", 1), 2), (("b", 2), 1)]),
    multi_mixed=pd.MultiIndex.from_tuples([("a", 1), (2, "b"), (True, None)]),
    multi_datetime=pd.MultiIndex.from_arrays([pd.date_range("2024-01-01", periods=2, tz="UTC"), [1, 2]]),
    multi_category=pd.MultiIndex.from_arrays([pd.Categorical(["b", "a"], categories=["unused", "a", "b"], ordered=True), [1, 2]]),
    multi_nullable=pd.MultiIndex.from_arrays([pd.array([1, None], dtype="Int64"), [1, 2]]),
    tuple_one=object_index([("b",), ("a",)]),
    tuple_two=object_index([("b", 2), ("a", 1)]),
    tuple_three=object_index([("b", 2, False), ("a", 1, True)]),
    tuple_empty=object_index([(), ()]),
    tuple_ragged=object_index([("a",), ("b", 2, False)]),
    tuple_missing=object_index([("a", None), (None, 1), (pd.NA, pd.NaT)]),
    tuple_nested=object_index([(("a", 1), 2), (("b", 2), 1)]),
    tuple_scalar_mix=object_index([("a", 1), "b"]),
    tuple_null_mix=object_index([("a", 1), None]),
    scalar_text=object_index(["a", "b"]),
    scalar_int=pd.Index([1, 2], name="datetime"),
    object_empty=object_index([]),
)

# Check malformed imported level/code metadata separately from append: a native
# boundary must reject these without publishing a partly constructed index.
descriptors = dict(
    no_levels=dict(levels=[], codes=[]),
    no_codes=dict(levels=[[1]], codes=[]),
    level_count=dict(levels=[[1], [2]], codes=[[0]]),
    unequal_lengths=dict(levels=[[1], [2]], codes=[[0], [0, 0]]),
    code_too_large=dict(levels=[[1]], codes=[[1]]),
    code_too_negative=dict(levels=[[1]], codes=[[-2]]),
    duplicate_level=dict(levels=[[1, 1]], codes=[[0]]),
    wrong_names=dict(levels=[[1], [2]], codes=[[0], [0]], names=["only"]),
    unhashable_name=dict(levels=[[1]], codes=[[0]], names=[["bad"]]),
    invalid_sortorder=dict(levels=[[1, 2]], codes=[[1, 0]], sortorder=1),
    negative_sortorder=dict(levels=[[1, 2]], codes=[[1, 0]], sortorder=-1),
    float_codes=dict(levels=[[1, 2]], codes=[[0.9, 1.9]]),
    missing_in_levels=dict(levels=[[None, "a"]], codes=[[0, 1]]),
    empty_level_missing=dict(levels=[[]], codes=[[-1]]),
)
constructors = []
for name, descriptor in descriptors.items():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            index = pd.MultiIndex(**descriptor)
        except Exception as error:
            outcome = dict(error=type(error).__name__, message=str(error))
        else:
            outcome = dict(output=index_snapshot(index))
    constructors.append(dict(name=name, warnings=[[type(w.message).__name__, str(w.message)] for w in captured], **outcome))

pairs = []
for (ln, left), (rn, right), lc, rc in itertools.product(samples.items(), samples.items(), [False, True], [False, True]):
    frame = pd.DataFrame(dict(x=list(range(len(left)))) if lc else {}, index=left)
    _, outcome = execute(frame, right, rc)
    pairs.append(dict(left=ln, right=rn, left_columns=lc, right_columns=rc, **outcome))

chains = []
for (name, index), next_name in itertools.product(samples.items(), ["tuple_two", "tuple_ragged", "scalar_text", "object_empty"]):
    initial = pd.DataFrame(columns=["x"], index=index[:0])
    current, first = execute(initial, index, True)
    second = None
    if current is not None:
        _, second = execute(current, samples[next_name], True)
    chains.append(dict(first=name, second=next_name, initial=index_snapshot(initial.index), first_output=first, second_output=second))

contract = dict(inputs={name:index_snapshot(value) for name,value in samples.items()},
                constructors=constructors, pairs=pairs, chains=chains)
digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(dict(pandas=pd.__version__, numpy=np.__version__, digest=digest, **contract)))
