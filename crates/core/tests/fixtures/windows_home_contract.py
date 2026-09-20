"""Exercise actual host pathlib.expanduser with isolated subprocess environment."""
import json
import os
from pathlib import Path

def units(value):
    if value is None:
        return None
    data = value.encode("utf-16-le", errors="surrogatepass")
    return [int.from_bytes(data[i:i+2], "little") for i in range(0, len(data), 2)]

keys = ["USERPROFILE", "HOMEDRIVE", "HOMEPATH", "USERNAME"]
environments = [
    {}, {"USERPROFILE": "C:/Users/andy", "USERNAME": "andy"},
    {"USERPROFILE": "C:/Users/andy/", "USERNAME": "andy"},
    {"USERPROFILE": "C:/Users/custom", "USERNAME": "andy"},
    {"USERPROFILE": "", "USERNAME": "andy"},
    {"HOMEDRIVE": "D:", "HOMEPATH": "\\Users\\andy", "USERNAME": "andy"},
    {"HOMEPATH": "relative/andy", "USERNAME": "andy"},
    {"HOMEDRIVE": "C:", "HOMEPATH": "", "USERNAME": ""},
    {"USERPROFILE": "andy", "USERNAME": "andy"},
    {"USERPROFILE": "C:", "USERNAME": ""},
    {"USERPROFILE": "C:/Users/andy/", "USERNAME": ""},
    {"USERPROFILE": "~bad", "USERNAME": "bad"},
    {"USERPROFILE": "C:/Users/andy"},
    {"USERPROFILE": "C:/Users/Andy", "USERNAME": "andy"},
    {"USERPROFILE": "//server/share/andy", "USERNAME": "andy"},
    {"USERPROFILE": "//server/share", "USERNAME": ""},
    {"USERPROFILE": "C:/Users/\ud800", "USERNAME": "\ud800"},
]
paths = ["", ".", "./", "relative/../file", "C:/~andy/data", "\\~andy/data", "C:~andy", "~",
         "~/", "~\\", "~/data/../file", "./~/data", "~andy", "~bob/data", "~Andy/x", "~\ud800/file", "~+/x"]
cases = []
for environment in environments:
    for key in keys:
        os.environ.pop(key, None)
    os.environ.update(environment)
    for value in paths:
        try:
            result, error = str(Path(value).expanduser()), None
        except Exception as failure:
            result, error = None, str(failure)
        cases.append({"environment": [units(environment.get(key)) for key in keys],
                      "input": units(value), "result": units(result), "error": error})
print(json.dumps(cases))
