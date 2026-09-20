"""Run unchanged actual ntpath.realpath with observable query callbacks."""
import json
import ntpath
import types


def run(case):
    trace = []
    counts = {}
    def query(stage, value=None):
        trace.append([stage] if value is None else [stage, value])
        position = counts.get(stage, 0)
        counts[stage] = position + 1
        steps = case.get(stage, [])
        step = steps[position] if position < len(steps) else None
        if step is None:
            if stage == "cwd":
                return "C:/cwd"
            if stage == "key":
                return ntpath.normcase(value)
            return value
        if "value" in step:
            return step["value"]
        error = step["error"]
        if error in ("nul", "notlink"):
            raise ValueError(error)
        if error == "allocation":
            raise MemoryError()
        if error == "length":
            raise OverflowError()
        failure = OSError("query failed")
        failure.winerror = error
        raise failure
    namespace = dict(ntpath.__dict__)
    namespace.update(os=types.SimpleNamespace(getcwd=lambda: query("cwd")),
                     normcase=lambda p: query("key", p),
                     _getfinalpathname=lambda p: query("final", p),
                     _getfinalpathname_nonstrict=lambda p, **kw: query("fallback", p))
    function = types.FunctionType(ntpath.realpath.__code__, namespace)
    function.__kwdefaults__ = ntpath.realpath.__kwdefaults__
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


cases = [{"input": value} for value in ["", ".", "..", "a/../b", "C:leaf", "D:leaf", "/root", "C:/root/",
         "//host/share/a", "nul", "NUL", "./nul", "a/../nul", "nul/child", "\\\\?\\C:\\a", "\\\\?\\UNC\\host\\share\\a"]]
for stage in ["cwd", "key", "final", "fallback"]:
    for failure in [0,2,3,5,32,87,1920,1921,9999,"nul","notlink","allocation","length"]:
        for value in ["relative", "C:/absolute", "nul", "\\\\?\\C:\\absolute"]:
            case = {"input": value, stage: [{"error": failure}]}
            if stage == "fallback":
                case["final"] = [{"error":2}]
            cases.append(case)
for target in ["\\\\?\\C:\\target", "\\\\?\\UNC\\host\\share\\file", "\\\\?\\unc\\host\\share\\file"]:
    for initial in [None, 0, 2, 5]:
        for check in [{"value": target}, {"value": target + "\\"}, {"value":"different"},
                      *[{"error": e} for e in [0,2,3,5,9999,"nul","notlink","allocation","length"]]]:
            case = {"input":"C:/start"}
            if initial is None:
                case["final"] = [{"value":target}, check]
            else:
                case["final"] = [{"error":initial}, check]
                case["fallback"] = [{"value":target}]
            cases.append(case)
# Embedded NUL with a verbatim cwd still reaches prefix verification; native
# query validation must reject the unchanged NUL again, never fabricate an OS error.
cases.append({"input":"a\0b", "cwd":[{"value":"\\\\?\\C:\\cwd"}],
              "final":[{"error":"nul"},{"error":"nul"}]})
print(json.dumps([run(case) for case in cases]))
