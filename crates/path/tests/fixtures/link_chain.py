"""Execute unchanged host _readlink_deep with observable controlled query operations."""
import json
import ntpath
import types


def run(case):
    trace = []
    key_calls = 0
    def key(value):
        nonlocal key_calls
        trace.append(["key", value])
        key_calls += 1
        if key_calls == case.get("key_fail", 0):
            raise MemoryError("mapping allocation")
        return ntpath.normcase(value)
    def read(value):
        trace.append(["read", value])
        step = case["steps"].get(value, {"error": 2})
        if "value" in step:
            return step["value"]
        failure = step["error"]
        if failure == "nul" or failure == "notlink":
            raise ValueError(failure)
        if failure == "allocation":
            raise MemoryError("read allocation")
        if failure == "length":
            raise OverflowError("length")
        error = OSError("native error")
        error.winerror = failure
        raise error
    def islink(value):
        trace.append(["islink", value])
        return case["steps"].get(value, {}).get("symbolic", False)
    namespace = dict(ntpath.__dict__)
    namespace.update(normcase=key, _nt_readlink=read, islink=islink)
    function = types.FunctionType(ntpath._readlink_deep.__code__, namespace, argdefs=ntpath._readlink_deep.__defaults__)
    try:
        result = {"value": function(case["input"])}
    except MemoryError:
        result = {"error": "allocation"}
    except OverflowError:
        result = {"error": "length"}
    except OSError as error:
        result = {"error": error.winerror}
    return {"case": case, "result": result, "trace": trace}


cases = [
    {"input": "C:/A", "steps": {"C:/A": {"value": "c:\\a"}}},
    {"input": "C:/A", "steps": {"C:/A": {"value": "D:/B"}, "D:/B": {"value": "C:/A"}}},
    {"input": "C:/A", "steps": {"C:/A": {"value": "../missing", "symbolic": True}}},
    {"input": "C:/A", "steps": {"C:/A": {"value": "relative", "symbolic": False}}},
    {"input": "", "steps": {}},
    {"input": "C:/A/./B", "steps": {"C:/A/./B": {"value": "C:/A/B"}}},
    {"input": "C:/\u0130", "steps": {"C:/\u0130": {"value": "c:/i"}}},
]
for error in [1,2,3,5,21,32,50,67,87,4390,4392,4393,0,4,53,65,123,161,1005,1920,1921,9999,"nul","notlink","allocation","length"]:
    for reached in [False, True]:
        steps = {"C:/B": {"error": error}}
        if reached:
            steps["C:/A"] = {"value": "C:/B"}
        cases.append({"input": "C:/A" if reached else "C:/B", "steps": steps})
for position in [1,2,3,4]:
    cases.append({"input": "C:/A", "steps": {"C:/A": {"value": "C:/B"}}, "key_fail": position})
for value in ["", ".", "..", "C:tail", "D:tail", "/root", "a/../b", "//server/share/path", "\\\\?\\C:\\other"]:
    for symbolic in [False, True]:
        cases.append({"input": "C:/base/link", "steps": {"C:/base/link": {"value": value, "symbolic": symbolic}}})
print(json.dumps([run(case) for case in cases]))
