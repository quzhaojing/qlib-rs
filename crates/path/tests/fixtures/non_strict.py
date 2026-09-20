"""Execute the unchanged host fallback with controlled observable native queries."""
import json
import ntpath
import types


def run(case):
    trace = []
    def query(stage, value):
        trace.append([stage, value])
        step = case.get(stage, {}).get(value)
        if step is None:
            step = {"value": value} if stage == "deep" else {"error": 2}
        if "value" in step:
            return step["value"]
        failure = step["error"]
        if failure in ("nul", "notlink"):
            raise ValueError(failure)
        if failure == "allocation":
            raise MemoryError()
        if failure == "length":
            raise OverflowError()
        error = OSError("native error")
        error.winerror = failure
        raise error
    namespace = dict(ntpath.__dict__)
    namespace.update(_getfinalpathname=lambda p: query("final", p),
                     _readlink_deep=lambda p, **kw: query("deep", p),
                     _findfirstfile=lambda p: query("find", p))
    function = types.FunctionType(ntpath._getfinalpathname_nonstrict.__code__, namespace,
                                  argdefs=ntpath._getfinalpathname_nonstrict.__defaults__)
    try:
        result = {"value": function(case["input"])}
    except ValueError as error:
        result = {"error": str(error)}
    except MemoryError:
        result = {"error": "allocation"}
    except OverflowError:
        result = {"error": "length"}
    except OSError as error:
        result = {"error": error.winerror}
    return {"case": case, "result": result, "trace": trace}


cases = []
for value in ["", "a", "a/b/c", "C:/", "C:", "/", "//host/share/", "C:/a/./b", "C:/a/", "a//", "C:child"]:
    cases.append({"input": value})
for error in [1,2,3,5,21,32,50,53,65,67,87,123,161,1005,1920,1921,
              0,4,6,8,80,206,4390,9999,"nul","notlink","allocation","length"]:
    for suffix in [False, True]:
        start = "C:/base/leaf" if suffix else "C:/base"
        cases.append({"input": start, "final": {"C:/base": {"error": error}},
                      "find": {"C:/base": {"value": "RealName"}}})
for stage in ["deep", "find"]:
    for error in [1,2,3,5,32,87,1920,1921,9999,"nul","notlink","allocation","length"]:
        for suffix in [False, True]:
            cases.append({"input": "C:/base/leaf" if suffix else "C:/base",
                          "final": {"C:/base": {"error": 5}}, stage: {"C:/base": {"error": error}}})
for stage in ["final", "deep"]:
    for replacement in ["", "D:/target", "C:/base/", "C:\\base", "C:/base/.", "\\\\?\\C:\\base"]:
        for suffix in [False, True]:
            cases.append({"input": "C:/base/leaf" if suffix else "C:/base",
                          stage: {"C:/base": {"value": replacement}}})
for value in ["", "Actual", "D:Other", "\\Root"]:
    for suffix in [False, True]:
        cases.append({"input": "C:/base/leaf" if suffix else "C:/base",
                      "final": {"C:/base": {"error": 5}}, "find": {"C:/base": {"value": value}}})
print(json.dumps([run(case) for case in cases]))
