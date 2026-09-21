"""Pinned characterization for remaining public qlib.utils.time boundaries."""

import ast
import bisect
from datetime import date, datetime, time, timedelta, timezone
import functools
import hashlib
import itertools
import json
import re
import sys
import warnings
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import pytz


source_path = sys.argv[1]
source_bytes = open(source_path, "rb").read()
tree = ast.parse(source_bytes.decode("utf-8"), filename=source_path)
names = {
    "CN_TIME", "US_TIME", "TW_TIME", "get_min_cal", "Freq", "concat_date_time",
    "cal_sam_minute", "epsilon_change", "get_day_min_idx_range", "is_single_value", "time_to_day_index",
}
body = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in names for target in node.targets
    ):
        body.append(node)
    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
        body.append(node)
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {
    "bisect": bisect, "date": date, "datetime": datetime, "time": time,
    "timedelta": timedelta, "functools": functools, "pd": pd, "re": re,
    "C": SimpleNamespace(min_data_shift=0), "REG_CN": "cn", "REG_US": "us",
    "REG_TW": "tw", "Tuple": Tuple,
}
exec(compile(module, source_path, "exec"), namespace)
Freq = namespace["Freq"]


def captured(call):
    try:
        return {"ok": describe(call())}
    except Exception as error:
        return {"error": type(error).__name__, "message": str(error)}


def describe(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, int):
        return value
    if isinstance(value, pd.Timedelta):
        return {"kind": "Timedelta", "nanoseconds": value.value}
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return None
    if isinstance(value, Freq):
        return {"kind": "Freq", "text": str(value), "count": value.count, "base": value.base}
    if isinstance(value, str):
        return {"kind": "str", "text": value}
    if value is pd.NaT:
        return {"kind": "NaT", "text": "NaT"}
    if isinstance(value, pd.Timestamp):
        return {
            "kind": "Timestamp", "ticks": int(value.asm8.view("i8")), "unit": value.unit,
            "timezone": None if value.tz is None else str(value.tz), "text": str(value),
        }
    raise TypeError(type(value).__name__)


if "--ordinal-month" in sys.argv[2:]:
    from dateutil.parser import DEFAULTPARSER
    texts = []
    numbers = ["0", "1", "2", "32", "100", "0001", "2021"]
    months = ["Jan", "February"]
    separators = [" ", "-", "/", ".", ","]
    clocks = ["", " 9:30", " 13AM"]
    for first, last, suffixes, month, position, sep1, sep2, clock in itertools.product(
            numbers, ["0", "2", "0001", "2021"],
            [("st", ""), ("", "TH"), ("nd", "rd"), ("th", "st")],
            months, [0, 1, 2], separators, separators, clocks):
        first += suffixes[0]
        last += suffixes[1]
        tokens = ([month, first, last] if position == 0 else
                  [first, month, last] if position == 1 else [first, last, month])
        texts.append(tokens[0] + sep1 + tokens[1] + sep2 + tokens[2] + clock)
    for number, suffix, month, sep, clock in itertools.product(
            numbers, ["st", "ND", "rd", "Th"], months, separators, clocks):
        texts.extend([month + sep + number + suffix + clock,
                      number + suffix + sep + month + clock])
    texts.extend(prefix + clock for prefix, clock in itertools.product(
        ["Jan 1st, 2021", "Jan 2ND, 0001", "Jan 0001rd, 2", "Jan of 0001st 2"], clocks))
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if "--decimal-month-following" in sys.argv[2:]:
    from dateutil.parser import DEFAULTPARSER
    texts = []
    for token, month, position, number, clock in itertools.product(
            ["01.2", "01,2", "0.2", "29.2", "32.2", "99.2", "100.0", "100.2", "2020.22", "0001.22"],
            ["Jan", "February", "Sept"], ["first", "last", "of"],
            ["0", "01", "29", "32", "99", "100", "101", "0001", "0930", "2021"],
            ["", " 9:30", " 13AM", " 0930", " 9:30+8", " 9:30:00.0000001"]):
        prefix = (month + " " + token if position == "first" else
                  token + " " + month if position == "last" else month + " of " + token)
        texts.append(prefix + " " + number + clock)
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if "--explicit-month-day" in sys.argv[2:]:
    from dateutil.parser import DEFAULTPARSER
    texts = [month + marker + year + separator + day + clock
             for month, marker, year, separator, day, clock in itertools.product(
                 ["Jan", "February", "Sept"], [" of ", " OF "],
                 ["0", "1", "76", "100", "0000", "0001", "2020", "2021"],
                 [" ", "-", "/", ".", ","],
                 ["0", "01", "29", "32", "76", "99", "100", "101", "0001", "0930", "9999"],
                 ["", " 9:30", " 13AM", " 0930", " 123456.1230001", " 9:30+8", " 09", " 32", " 99"])]
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if "--month-compact-clock" in sys.argv[2:]:
    from dateutil.parser import DEFAULTPARSER
    texts = [date_text + " " + clock + period + zone
             for date_text, clock, period, zone in itertools.product(
                 ["Jan 1 2021", "1 Jan 2021", "2021 1 Jan", "Jan 32 2021",
                  "Feb 29 2021", "Jan-1-0000", "Jan 1 0001"],
                 ["00", "09", "24", "0930", "1260", "1330", "123456", "240000",
                  "123460", "123456.1", "123456.1234567", "123456.1230001",
                  "123456.0000001", "0930.5", "09.5", "123456.", "123456.000"],
                 ["", "PM", " AM"], ["", "+8", "+999"])]
    texts.extend(date_text + " " + clock + period + zone
                 for date_text, clock, period, zone in itertools.product(
                     ["Jan", "Jan 1", "Jan of 2021"],
                     ["123456", "240000", "123456.1", "123456.1234567"],
                     ["", "PM", " AM"], ["", "+8", "+999"]))
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if "--partial-month" in sys.argv[2:]:
    from dateutil.parser import DEFAULTPARSER
    months = ["Jan", "February", "Sept"]
    numbers = ["0", "1", "2", "29", "31", "32", "76", "99", "100", "101",
               "0000", "0001", "2021", "9999"]
    clocks = ["", " 9:30", " 13AM", " 9:30+8", " 9:30:00.0000001", " 24:00"]
    texts = [month + clock for month, clock in itertools.product(months, clocks)]
    for month, number, sep, clock in itertools.product(months, numbers, [" ", "-", "/", ".", ","], clocks):
        texts.extend([month + sep + number + clock, number + sep + month + clock])
    texts.extend(month + separator + number + clock
                 for month, number, clock, separator in itertools.product(
                     months, numbers, clocks, [" of ", " OF ", " Of "]))
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if any(mode in sys.argv[2:] for mode in ["--middle-month", "--edge-month"]):
    from dateutil.parser import DEFAULTPARSER
    positions = [0, 2] if "--edge-month" in sys.argv[2:] else [1]
    separators = [" ", "-", "/", ".", ","]
    first_numbers = ["0", "1", "29", "32", "100", "0001", "2020", "2021"]
    if "--edge-month" in sys.argv[2:]:
        first_numbers += ["01"]
    texts = []
    for first, month, last, sep1, sep2, clock, position in itertools.product(
                 first_numbers,
                 ["Jan", "February", "Sept"],
                 ["0", "2", "29", "32", "100", "0001", "2021"],
                 separators, separators,
                 ["", " 9:30", " 13AM", " 9:30+8", " 9:30:00.0000001"], positions):
        tokens = ([month, first, last] if position == 0 else
                  [first, month, last] if position == 1 else [first, last, month])
        texts.append(tokens[0] + sep1 + tokens[1] + sep2 + tokens[2] + clock)
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if "--decimal-day-period" in sys.argv[2:]:
    from dateutil.parser import DEFAULTPARSER
    texts = [token + "." + fraction + separator + following + " " + day + gap + period + zone
             for token, fraction, separator, following, day, gap, period, zone in itertools.product(
                 ["000001", "090102", "120000", "210102", "009999"], ["01", "1"],
                 ["-", "/", " "], ["00", "02", "13", "32", "99"],
                 ["23", "24", "31", "32", "99"], ["", " "], ["AM", "pm"], ["", "+8", "+999"])]
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if "--hour-period-zone" in sys.argv[2:]:
    from dateutil.parser import DEFAULTPARSER
    texts = [day + hour + gap + period + zone + suffix
             for day, hour, gap, period, zone, suffix in itertools.product(
                 ["", "2021-01 ", "2021-01-02 ", "01/02/2021 ", "Jan 2, 2021 ", "0000-01 ", "9999-12-31 ",
                  "01.02.2021 ", "01.02 2021 ", "01 02.2021 "],
                 ["0", "9", "12", "13", "23", "24", "99"], ["", " "], ["AM", "pm"],
                 ["+8", "-0230", "+8:99", "+25", "+999", " UTC", " GMT+8", " Z", " UTC-25"],
                 ["", " "])]
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if any(mode in sys.argv[2:] for mode in ["--year-month", "--wide-year-month", "--compact-pair"]):
    from dateutil.parser import DEFAULTPARSER
    years = ["0000", "0001", "0013", "0032", "0076", "0100", "2021", "9999"]
    if "--wide-year-month" in sys.argv[2:]:
        years = ["00000", "00001", "00013", "00032", "02021", "10000", "0000001",
                 "000002021", "2147483647", "2147483648", "9999999999999999999999"]
    if "--compact-pair" in sys.argv[2:]:
        years = ["000001", "210102", "991231", "20210102", "20210229",
                 "00000102", "202101020930", "202101022460", "20210102093045"]
    texts = [prefix + year + separator + month + clock + suffix
             for year, separator, month, clock, prefix, suffix in itertools.product(
                 years,
                 ["-", "/", ".", " ", "\\"], ["0", "1", "02", "13", "99"],
                 ["", " 9:30", "T09:30:00.123456789", " 9:30+8", " 24:60", " 9bad"],
                 ["", "\t"], ["", " "])]
    if "--compact-pair" in sys.argv[2:]:
        texts += [date_text + " " + clock for date_text, clock in itertools.product(
            ["20210102-02", "202101020930-02", "210102-02"],
            ["09+8", "0930+8", "123456+8"])]
        texts += [token + separator + "02" + clock for token, separator, clock in itertools.product(
            years, ["-", "/", ".", " ", "\\"],
            [" 9AM", " 13AM", " 24AM", " 24 AM", " 9AM+8", " 9:30:00", " 9:30+999"])]
        texts += ["210102.02-02" + clock for clock in [" 9AM", " 13AM", " 9AM+8"]]
        texts += ["000001.02 24AM+8", "000001.02 24AM+999",
                  "090102.02 31PM-25", "000001.02 99AM"]
    texts += ["2021-01 " + suffix for suffix in
              ["0930", "9", "9AM", "09:30 UTC", "02", "02 9:30"]]
    texts += [year + separator + month + " " + hour + gap + period
              for year, separator, month, hour, gap, period in itertools.product(
                  ["0000", "0001", "0013", "0032", "0076", "0100", "2021", "9999"],
                  ["-", "/", " "], ["0", "02", "13"],
                  ["0", "9", "12", "13", "23", "24", "99"], ["", " "], ["AM", "pm"])]
    texts += [year + separator + month + " " + day + clock
              for year, separator, month, day, clock in itertools.product(
                  ["0000", "0001", "0013", "0032", "0076", "0100", "2021", "9999"],
                  ["-", "/"], ["0", "02", "13"], ["0", "2", "31", "99"],
                  ["", " 9:30", " 9:30+8"])]
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if "--parser-context" in sys.argv[2:]:
    # Controlled parser initialization clock; actual Pandas and Qlib code remain
    # unchanged. Restoring info also prevents this fixture leaking process state.
    import time as system_time
    from unittest.mock import patch
    from dateutil.parser import _parser

    cases = []
    original_info = _parser.DEFAULTPARSER.info
    try:
        for year in [1, 1949, 1950, 1999, 2000, 2025, 2026, 2027, 2099, 2100, 9950, 9999]:
            with patch.object(_parser.time, "localtime", return_value=system_time.struct_time(
                    (year, 1, 1, 0, 0, 0, 0, 1, -1))):
                _parser.DEFAULTPARSER.info = _parser.parserinfo()
            assert _parser.DEFAULTPARSER.info._year == year
            for token, prefix, period in itertools.product(
                    ["1", "31", "32", "49", "50", "68", "69", "75", "76", "77",
                     "98", "99", "100", "1234", "9999", "10000", "2147483648"],
                    ["", " "], ["", " AM", "pm", " PM"]):
                text = prefix + "9:30:0," + token + period
                while True:
                    reference = date.today()
                    result = captured(lambda: pd.Timestamp(text))
                    if "ok" in result:
                        value = pd.Timestamp(text)
                        result["ok"]["components"] = [value.year, value.month, value.day,
                            value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
                    query = captured(lambda: namespace["get_day_min_idx_range"](
                        text, "2021-01-01 14:59", "1min", "cn"))
                    if date.today() == reference:
                        break
                cases.append({"text": text, "reference_date": reference.isoformat(),
                              "parser_year": year, "result": result, "range": query})
    finally:
        _parser.DEFAULTPARSER.info = original_info
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "cases": cases}))
    sys.exit(0)

if any(mode in sys.argv[2:] for mode in ("--year-first", "--wide-year", "--compact-following", "--compact-mixed")):
    from dateutil.parser import DEFAULTPARSER
    years = ["0000", "0001", "0013", "0076", "0100", "2020", "2021", "2263", "9999"]
    if "--wide-year" in sys.argv[2:]:
        years = ["00000", "00001", "00013", "00076", "02021", "10000",
                 "2147483647", "2147483648", "9999999999999999999999"]
    if "--compact-following" in sys.argv[2:] or "--compact-mixed" in sys.argv[2:]:
        years = ["000001", "210102", "991231", "20210102", "20210229",
                 "00000102", "202101020930", "202101022460", "20210102093045"]
    texts = [prefix + separator.join([year, month, day]) + clock + suffix
             for year, (month, day), separator, clock, prefix, suffix in itertools.product(
                 years,
                 [("1", "2"), ("01", "02"), ("02", "29"), ("00", "01"), ("13", "01"), ("01", "00")],
                 ["-", "/", ".", " ", "\\"],
                 ["", " 9:3", " 09:30:0.1234567", " 09:30:00,1", " 13:30 PM",
                  " 24:60:60", " 9:30:0,2", " 09:30:00.0000001"], ["", "\t"], ["", " "])]
    texts += [day + separator + clock for day, separator, clock in itertools.product(
        ["2021-1-2", "0000/01/02", "9999.1.2"], [" ", "T"],
        ["09", "0930", "093000", "093000.1234567", "9:3:0", "9:3:0.0000001"])]
    if "--compact-following" in sys.argv[2:]:
        texts += [token + "-01-02" + clock for token, clock in itertools.product(
            years, [" 9:30+8", " 9:30+999", " 9:30+25", " 9bad+8", " 9bad"])]
    if "--compact-mixed" in sys.argv[2:]:
        texts = [prefix + token + first + middle + second + last + clock + suffix
                 for token, first, second, (middle, last), clock, prefix, suffix in itertools.product(
                     years, ["-", "/", ".", " ", "\\"], ["-", "/", ".", " ", "\\"],
                     [("01", "02"), ("1", "2"), ("99", "00"), ("00", "32")],
                     ["", " 9:30", " 9:30:00.1234567", " 9:30+8", " 24:60"],
                     ["", "\t"], ["", " "])
                 if first != second]
        texts += [token + ".01 02" + clock for token, clock in itertools.product(
            ["00000102", "20210102", "202101020930"], [" 9bad", " 9bad+8", " 9:30+999"])]
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if "--month-date" in sys.argv[2:]:
    from dateutil.parser import DEFAULTPARSER
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
              "January", "February", "March", "April", "June", "July", "August",
              "September", "October", "November", "December", "Sept", "SEPT", "january"]
    texts = [prefix + month + " " + day + comma + " " + year + clock
             for month, day, year, clock, comma, prefix in itertools.product(
                 months, ["0", "1", "29", "32", "99"],
                 ["0000", "0001", "0076", "0100", "0101", "2020", "2021", "9999"],
                 ["", " 9:30", " 13:30 PM", " 9:30:00.0000001"], ["", ","], ["", " "])]
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "cases": cases}))
    sys.exit(0)

if any(mode in sys.argv[2:] for mode in ["--date-clock", "--general-delimited", "--decimal-date-token"]):
    from dateutil.parser import DEFAULTPARSER
    dates = ["01/02/2021", "13/02/2021", "02/29/2020", "02/29/2021",
             "00/01/2021", "31/13/2021", "01/02/0000", "01/02/0001",
             "01/02/0076", "01/02/1000", "01/02/2263", "12/31/9999",
             "32/01/2021", "99/01/2021", "13/00/2021", "00/13/2021"]
    clocks = ["0:00", "9:30", "9:3", "12:30", "13:30", "24:60:60", "9:60",
              "9:30:60", "9:30:0.1", "9:3:0.1230001", "9:30:00.0000001",
              "9:30:00.1234567", "9:30:00,123000", "9:30:0,"]
    texts = [prefix + day + separator + clock + period + suffix
             for day, clock, period, prefix, separator, suffix in itertools.product(
                 dates, clocks, ["", " AM", "pm", " PM"], ["", " "], [" ", "T"], ["", "\t"])]
    if "--general-delimited" in sys.argv[2:]:
        texts = [prefix + first + sep1 + second + sep2 + year + clock + suffix
                 for (first, second, year), sep1, sep2, clock, prefix, suffix in itertools.product(
                     [("01", "02", "2021"), ("13", "02", "2021"), ("02", "29", "2021"),
                      ("01", "02", "0000"), ("01", "02", "0001"), ("32", "01", "2021"),
                      ("00", "13", "2021"), ("01", "02", "0100")],
                     " /-.", " /-.",
                     ["", " 9:30", "T9:3", " 13:30 PM", " 24:60:60",
                      " 09:30:00.0000001", " 9:30+8", " 9:30+25", " 9:30:1,2"],
                     ["", " "], ["", "\t"])]
        texts += ["01 32.2021 9:30", "32.01 0100 9:30", "32.01 2021 9:30",
                  "01 02.2021 9:30:1,2020", "01 02.2021 9:30:1,0100",
                  "01 02.2021 9:30:0,", "01 02.2021 9:30:00,1"]
    if "--decimal-date-token" in sys.argv[2:]:
        texts = [day + " " + clock + token + period
                 for day, clock, token, period in itertools.product(
                     ["01 02.2021", "13 02.2021", "32 01.2021", "01.02 2021", "00.13 2021"],
                     ["9:30:1,", "13:30:1,", "24:60:1,", "9:3:0,"],
                     ["0", "2", "12", "23", "24", "31", "32", "100", "2020", "12345",
                      "123456", "235959", "246060", "20210101", "202101010102",
                      "20210101010203", "2147483648", "999999999999999999999999"],
                     ["", " AM", "pm", " PM", "am"])]
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__,
                     "parser_year": DEFAULTPARSER.info._year, "cases": cases}))
    sys.exit(0)

if "--delimited-date" in sys.argv[2:]:
    fields = ["0", "00", "1", "01", "2", "02", "12", "13", "29", "30", "31", "32"]
    years = ["1000", "2020", "2021", "9999"]
    texts = [month + sep1 + day + sep2 + year
             for month, day, year, sep1, sep2 in itertools.product(fields, fields, years, " /-.", " /-.")]
    texts += [month + sep + year for month, year, sep in itertools.product(
        ["00", "01", "02", "12", "13", "31", "32"], years, " /-")]
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "cases": cases}))
    sys.exit(0)

if "--numeric-date" in sys.argv[2:]:
    reference = date.today()
    tokens = ["010203", "130101", "311299", "990131", "000101", "000000", "999999",
              "20210101", "00000101", "00010101", "00000000", "20210229", "20210001", "20210100",
              "202101010102", "202101012460", "202101011360", "202101011259", "000001010102",
              "20210101010203", "20210101123060", "20210101246060", "20200229235959", "00000101010203"]
    spaces = ["", " ", "\t", "\n", "\r\n"]
    texts = [prefix + token + period + suffix
             for token, period, prefix, suffix in itertools.product(
                 tokens, ["", " AM", "pm"], spaces, spaces)]
    texts += [prefix + year + suffix for year, prefix, suffix in itertools.product(
        ["0000", "0001", "0012", "0031", "0032", "0076", "0099", "2021", "9999"], spaces, spaces)]
    cases = []
    for text in texts:
        result = captured(lambda: pd.Timestamp(text))
        if "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text, "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "reference_date": reference.isoformat(),
                     "cases": cases}))
    sys.exit(0)

if any(mode in sys.argv[2:] for mode in ["--clock-text", "--clock-errors", "--date-token", "--compact-token"]):
    cases = []
    clock_seconds = ["", ":1", ":01", ":1,", ":0.1", ":0.1230001", ":00.0", ":00.0000",
        ":00.1000", ":00.0000001", ":00.0000010", ":00.1230000", ":00.123456789",
        ":00.", ":00.123000123456789012345"]
    clock_seconds += [value.replace(".", ",") for value in clock_seconds if "." in value]
    clock_cases = itertools.product(
        ["0", "00", "9", "09", "12", "13", "23"], ["3", "03", "30"],
        clock_seconds,
        ["", " AM", "pm", " PM"], ["", " ", "\t"],
    )
    if "--date-token" in sys.argv[2:]:
        tokens = ["0", "00", "000", "1", "01", "001", "12", "13", "23", "24",
                  "28", "29", "30", "31", "32", "49", "50", "68", "69", "75",
                  "76", "99", "100", "1234", "9999", "10000", "2147483647",
                  "2147483648", "99999999999999999999999999999999999999"]
        clock_cases = itertools.product(["0", "9", "12", "13", "23", "24"],
            ["3", "30", "60"], [":0," + token for token in tokens],
            ["", " AM", "pm", " PM"], ["", " "])
    if "--compact-token" in sys.argv[2:]:
        tokens = ["010203", "130101", "311299", "990131", "000101", "999999",
                  "20210101", "00000101", "00010101", "20210229", "20210001", "20210100",
                  "202101010102", "202101012460", "202101011360", "202101011259",
                  "20210101010203", "20210101123060", "20210101246060", "20200229235959"]
        clock_cases = itertools.product(["0", "9", "12", "13", "23", "24"],
            ["3", "30", "60"], [":0," + token for token in tokens],
            ["", " AM", "pm", " PM"], ["", " "])
    if "--clock-errors" in sys.argv[2:]:
        # Include valid controls and date-token overrides: lexical AM/PM and
        # comma-date failures can precede otherwise invalid clock components.
        clock_cases = itertools.product(
            ["0", "12", "13", "23", "24", "99"], ["3", "30", "60", "99"],
            ["", ":0", ":60", ":99", ":0.1", ":00,1", ":0,", ":00.0000001",
             ":0,2", ":0,202101010102", ":0,999999999999999999999999"],
            ["", " AM", "pm", " PM"], ["", " ", "\t"],
        )
    for hour, minute, second, period, prefix in clock_cases:
        if period and int(hour) > 12 and "--clock-text" in sys.argv[2:]:
            continue
        text = prefix + hour + ":" + minute + second + period
        # Associate the source's local current date with this row, including the
        # rare midnight rollover without substituting a parser or changing source.
        while True:
            reference = date.today()
            result = captured(lambda: pd.Timestamp(text))
            if "ok" in result:
                value = pd.Timestamp(text)
                result["ok"]["components"] = [value.year, value.month, value.day,
                    value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
            query = captured(lambda: namespace["get_day_min_idx_range"](
                text, "2021-01-01 14:59", "1min", "cn"))
            if date.today() == reference:
                break
        cases.append({"text": text, "reference_date": reference.isoformat(),
                      "result": result, "range": query,
                      "general_date_token": "," in second and len(second.split(",")[0]) == 2
                          and bool(second.split(",")[1])})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "cases": cases}))
    sys.exit(0)

if any(mode in sys.argv[2:] for mode in ["--zone-formats", "--zoned-date-token", "--utc-alias", "--named-zone"]):
    from dateutil.parser import _parser

    texts = [prefix + day + clock + space + zone + suffix
             for day, clock, zone, prefix, space, suffix in itertools.product(
                 ["", "01/02/2021 ", "02/29/2021 ", "01/02/0000 ",
                  "Jan 2, 2021 ", "Jan 2, 0001 ", "Feb 29, 2021 "],
                 ["9:30", "9:3", "12:30 PM", "13:30 PM", "24:60:60", "9:30:60",
                  "09:30:00.1234567", "09:30:00.0000001", "09:30:00.1"],
                 ["+8", "-8", "+0", "+25", "+00:60", "+830"],
                 ["", " ", "\t"], ["", " ", "\r\n"], ["", " "])]
    texts += ["01/02/2021 9:30:1,2", "Jan 2, 2021 9:30:1,2"]
    if "--utc-alias" in sys.argv[2:]:
        texts = [prefix + day + clock + space + zone
                 for day, clock, zone, space, prefix in itertools.product(
                     ["", "2021-01-01 ", "0001-01-01 ", "2021-02-29 ", "Jan 2, 2021 "],
                     ["9:30", "9:3", "13:30 PM", "24:60:60", "09:30:00.0000001", "09:30:00.1234567"],
                     ["UTC", "GMT", "Z", "z", "UTC+8", "GMT-8", "UTC+25", "GMT+830"],
                     ["", " ", "\t"], ["", " "])]
    if "--named-zone" in sys.argv[2:]:
        texts = [prefix + day + clock + space + name + offset
                 for day, clock, name, offset, space, prefix in itertools.product(
                     ["", "2021-01-01 ", "0001-01-01 ", "2021-02-29 ", "Jan 2, 2021 "],
                     ["9:30", "9:3", "13:30 PM", "24:60:60", "09:30:00.0000001", "09:30:00.1234567"],
                     ["XYZ", "EST", "X", "PMUTC", "ABCDEF", "abc"],
                     ["", "+8", "+0", "+25", "+830"], ["", " "], ["", " "])]
        texts += ["1677-09-21 01:00:00.0000001 XYZ-1",
                  "2262-04-11 23:47:00.0000001 XYZ+1", "9:30:0,2147483648 XYZ"]
    if "--zoned-date-token" in sys.argv[2:]:
        texts = []
        for token, clock, period, space, zone, prefix in itertools.product(
                ["2", "32", "2020", "0001", "010203", "20210101", "202101012460",
                 "202101011259", "20210101123060", "2147483648", "999999999999999999999999", "0", "12345", "2147483647"],
                ["9:30:0,", "13:30:1,", "24:60:0,"], ["", "pm", " PM"],
                ["", " "], ["+8", "-8", "+0", "+25", "+00:60", "+830", "-32", "-2020", "-13", "-8:3"], ["", " "]):
            text = prefix + clock + token + period + space + zone
            texts.append(text)
        texts += ["9:30:00,1+8", "9:30:0,+8", "9:30:00,0000001+8"]
    cases = []
    for text in texts:
        while True:
            reference = date.today()
            with warnings.catch_warnings(record=True) as constructor_warnings:
                warnings.simplefilter("always")
                try:
                    value = pd.Timestamp(text)
                    result = {"ok": describe(value)}
                    result["ok"]["components"] = [value.year, value.month, value.day,
                        value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
                except Exception as error:
                    result = {"error": type(error).__name__, "message": str(error)}
            with warnings.catch_warnings(record=True) as range_warnings:
                warnings.simplefilter("always")
                query = captured(lambda: namespace["get_day_min_idx_range"](
                    text, "2021-01-01 14:59", "1min", "cn"))
            if date.today() == reference:
                break
        cases.append({"text": text, "reference_date": reference.isoformat(),
                      "result": result, "range": query,
                      "constructor_warnings": [{"category": warning.category.__name__, "message": str(warning.message)} for warning in constructor_warnings],
                      "range_warnings": [{"category": warning.category.__name__, "message": str(warning.message)} for warning in range_warnings],
                      "whole_zone_name_required": bool(re.search(r"[A-Za-z](?:UTC|GMT|Z|z)", text))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__,
                     "parser_year": _parser.DEFAULTPARSER.info._year,
                     "local_timezone_names": list(__import__("time").tzname), "cases": cases}))
    sys.exit(0)

if any(mode in sys.argv[2:] for mode in ["--iso-text", "--iso-offset", "--general-zone"]):
    dates = ["0000-01-01", "0001-01-01", "1000-01-01", "1677-09-21",
             "1970-01-01", "2021-01-01", "2262-04-11", "9999-12-31"]
    texts = [date + separator + clock + fraction + zone
             for date, separator, clock, fraction, zone in itertools.product(
                 dates, ["T", " "], ["00:12:43", "09:30:00", "23:47:16"],
                 ["", ".", ".0", ".001", ".1234", ".123456", ".1234567",
                  ".145224192", ".854775807", ".123456789012345678"],
                 ["", "Z", "+08:00", "-0530", "+23:59", "-00:00", "+01"])]
    texts += [date + clock for date, clock in itertools.product(dates, ["", " 09", " 09:30"])]
    texts += [
        "1677-09-21T01:12:43.145224192+01:00",
        "1677-09-21T01:12:43.145224193+01:00",
        "1677-09-21T00:12:43.145224193+01:00",
        "2262-04-11T23:47:16.854775807-01:00",
        "2262-04-11T23:47:16.854775808",
        "2262-04-11T22:47:16.854775808-01:00",
        " 2021-01-01T09:30:00 ", "\t2021-01-01T09:30:00 \r\n",
    ]
    if "--iso-offset" in sys.argv[2:]:
        texts = [day + "T" + clock + space + zone + suffix
                 for day, clock, space, zone, suffix in itertools.product(
                     dates,
                     ["09", "9:3", "0930", "09:30:00", "09:30:00.1", "09:30:00.1234",
                      "09:30:00.1234567", "09:30:00.0000001"],
                     ["", " ", "\t", "\r\n", "\v\f"],
                     ["+8", "-8", "+0", "-0", "+8:3", "-8:3", "+08:3", "-08:3",
                      "+8:03", "-8:03", "+083", "-083", "+003", "-003", "+0:0",
                      "-0:00", "+00:0", "Z", "+23:59", "-23:59"], ["", "\t "])]
        texts += ["1677-09-21T00:15:43.145224192+003",
                  "1677-09-21T00:15:43.145224191+0:3",
                  "2262-04-11T23:47:16.854775807-0:3"]
    if "--general-zone" in sys.argv[2:]:
        texts = [day + " " + clock + zone + " " for day, clock, zone in itertools.product(
            ["2021-01-01", "0000-01-01", "0001-01-01", "2021-02-29"],
            ["09", "0930", "093000", "09:30", "09:30:00.1234567", "13:30 PM",
             "24:60:60", "09:30:60", "09:30:00.0000001"],
            ["+8", "-8", "+0", "-0", "+25", "-25", "+00:60", "+23:60", "+9999", "+830", "+0:0"])]
        texts += ["1677-09-21 01:00:00.0000001+1 ", "2262-04-11 23:47:00.0000001-1 "]
    cases = []
    for text in texts:
        general = "--general-zone" in sys.argv[2:] or bool(re.search(r"[+-][0-9]{1,2}[\t ]+$", text))
        result = captured(lambda: pd.Timestamp(text))
        if general and "ok" in result:
            value = pd.Timestamp(text)
            result["ok"]["components"] = [value.year, value.month, value.day,
                value.hour, value.minute, value.second, value.microsecond, value.nanosecond]
        cases.append({"text": text,
                      "general_timezone_required": general,
                      "result": result,
                      "range": captured(lambda: namespace["get_day_min_idx_range"](
                          text, "2021-01-01 14:59", "1min", "cn"))})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "cases": cases}))
    sys.exit(0)

if "--named-zone-range" in sys.argv[2:]:
    from dateutil.parser import DEFAULTPARSER
    texts = ["2021-01-01 9:30 XYZ", "2021-02-29 9:30 XYZ",
             "0001-01-01 9:30:00.0000001 XYZ", "NaT", "2021-01-01 14:59"]
    cases = []
    for start, end, frequency, region in itertools.product(texts, texts, ["1min", "bad"], ["cn", "CN"]):
        namespace["get_min_cal"].cache_clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = captured(lambda: namespace["get_day_min_idx_range"](start, end, frequency, region))
        cases.append({"start": start, "end": end, "frequency": frequency, "region": region,
                      "result": result, "warnings": [str(warning.message) for warning in caught]})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "parser_year": DEFAULTPARSER.info._year,
                     "reference_date": date.today().isoformat(),
                     "local_timezone_names": list(__import__("time").tzname), "cases": cases}))
    sys.exit(0)

if "--range-text" in sys.argv[2:]:
    # Freeze the actual constructor surface before implementing a native parser.
    # Clock-only strings use today's date; range queries intentionally discard it.
    texts = [
        "9:30", "09:30:00.000000001", "9:30 PM", "14:59:59.999999999",
        "2021-01-01T09:30:00+08:00", "20210101", "Jan 1, 2021 9:30",
        "01/02/2021 9:30", "0000-01-01 09:30", "2021-01-01 09:30 XYZ",
        "NaT", "nat", "NAT", "", " ", "None", "2021-02-29", "24:00",
        "09:60", "2021-001", "2021-W01-1", "10000-01-01 09:30",
    ]
    inputs = []
    for text in texts:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = captured(lambda: pd.Timestamp(text))
        inputs.append({"text": text, "result": result,
                       "warnings": [str(w.message) for w in caught]})
    cases = []
    for start, end, frequency, region in itertools.product(
        range(len(texts)), range(len(texts)), ["1min", "5day", "0min", "bad"],
        ["cn", "tw", "us", "CN"],
    ):
        namespace["get_min_cal"].cache_clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = captured(lambda: namespace["get_day_min_idx_range"](
                texts[start], texts[end], frequency, region))
        cases.append({"start": start, "end": end, "frequency": frequency,
                      "region": region, "result": result,
                      "warnings": [str(w.message) for w in caught],
                      "cache_misses": namespace["get_min_cal"].cache_info().misses})
    markers = [{"text": text, "result": captured(lambda: pd.Timestamp(text))}
               for text in ["", "NaT", "nat", "NAT", "nan", "NaN", "NAN",
                            "Nat", "NAt", "NaT ", " nan", "NULL"]]
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                     "pandas_version": pd.__version__, "inputs": inputs, "cases": cases,
                     "markers": markers}))
    sys.exit(0)

single_value_inputs = {
    "nat": pd.NaT,
    "ordinary": pd.Timestamp("2021-01-01 10:00:00"),
    "cn_close": pd.Timestamp("2021-01-01 11:29:00.000000001"),
    "tw_close": pd.Timestamp("2021-01-01 13:59:00"),
    "us_close": pd.Timestamp("2021-01-01 15:59:00", tz="America/New_York"),
    "utc_same": pd.Timestamp("2021-01-01 20:59:00", tz="UTC"),
    "utc_later": pd.Timestamp("2021-01-01 21:00:00", tz="UTC"),
    "minimum": pd.Timestamp.min,
    "maximum": pd.Timestamp.max,
    "epoch_plus_ns": pd.Timestamp(1),
    "epoch_minus_ns": pd.Timestamp(-1),
    "year1000_s": pd.Timestamp(np.datetime64("1000-01-01", "s")),
    "year3000_s": pd.Timestamp(np.datetime64("3000-01-01", "s")),
    "ordinary_ms": pd.Timestamp(np.datetime64("2021-01-01T10:00:00.001", "ms")),
    "ordinary_us": pd.Timestamp(np.datetime64("2021-01-01T10:00:00.000001", "us")),
}
single_value_cases = []
for region, (start_name, start), (end_name, end), (freq_name, freq) in itertools.product(
    ["cn", "tw", "us", "CN", ""],
    single_value_inputs.items(), single_value_inputs.items(),
    [("nat", pd.NaT), ("negative", pd.Timedelta(-1, "ns")),
     ("zero", pd.Timedelta(0)), ("minute", pd.Timedelta(1, "min"))],
):
    single_value_cases.append({
        "region": region, "start_name": start_name, "end_name": end_name,
        "frequency_name": freq_name, "start": describe(start), "end": describe(end),
        "frequency": describe(freq),
        "result": captured(lambda: namespace["is_single_value"](start, end, freq, region)),
    })

for zone, fraction, reverse, region, freq in itertools.product(
    [None, "UTC", "Asia/Shanghai", "America/New_York", "Etc/GMT+5"],
    ["", ".120000", ".123456"], [False, True], ["cn", "tw", "us", "CN"],
    [pd.NaT, pd.Timedelta(1, "min")],
):
    wide = pd.Timestamp("3000-01-01 01:02:03" + fraction, tz="UTC")
    ordinary = pd.Timestamp("2021-01-01", tz="UTC").as_unit("ns")
    wide = wide.tz_localize(None) if zone is None else wide.tz_convert(zone)
    ordinary = ordinary.tz_localize(None) if zone is None else ordinary
    start, end = (ordinary, wide) if reverse else (wide, ordinary)
    single_value_cases.append({
        "region": region, "start": describe(start), "end": describe(end),
        "frequency": describe(freq),
        "result": captured(lambda: namespace["is_single_value"](start, end, freq, region)),
    })

for unit, bound, zone, region, freq, reverse, end_unit in itertools.product(
    ["s", "ms", "us"], ["minimum", "maximum", "closing"],
    [None, "UTC", "+08:00", "-05:00", "America/New_York", "Europe/London", "Asia/Shanghai", "Australia/Sydney"],
    ["cn", "tw", "us"], [pd.NaT, pd.Timedelta(1, "min")], [False, True],
    ["s", "ms", "us", "ns", None],
):
    scale = {"s": 1, "ms": 1000, "us": 1000000}[unit]
    ticks = -(2**63) + 1 if bound == "minimum" else 2**63 - 1
    if bound == "closing":
        # Complete day before the upper limit; retains CN's 11:29 local close.
        ticks = (ticks // (86400 * scale) - 1) * 86400 * scale + (11 * 3600 + 29 * 60) * scale
    if zone not in [None, "UTC"]:
        # Source construction rejects wall-clock overflow at the exact limit.
        # Keep valid wide inputs two days inside that constructor boundary.
        ticks += (1 if bound == "minimum" else -1) * 2 * 86400 * scale
        if bound == "closing":
            last_offset = pd.Timestamp(2**31 - 1, unit="s", tz="UTC").tz_convert(zone).utcoffset()
            ticks -= int(last_offset.total_seconds()) * scale
    wide = pd.Timestamp(ticks, unit=unit, tz=zone)
    other = pd.NaT if end_unit is None else pd.Timestamp(0, unit=end_unit, tz=zone)
    start, end = (other, wide) if reverse else (wide, other)
    single_value_cases.append({
        "region": region, "start": describe(start), "end": describe(end),
        "frequency": describe(freq),
        "result": captured(lambda: namespace["is_single_value"](start, end, freq, region)),
    })

alignment_timezones = []
for zone, year, month, unit, region, second in itertools.product(
    ["America/New_York", "Europe/London", "Asia/Shanghai", "Australia/Sydney", "Europe/Amsterdam", "Europe/Paris"],
    [1700, 1800, 1880, 1900, 1902, 1920, 1970, 2037, 2038, 2100],
    [1, 7], ["s", "ms", "us", "ns"], ["cn", "tw", "us"], [0, 1],
):
    hour, minute = {"cn": (11, 29), "tw": (13, 25), "us": (15, 59)}[region]
    start = pd.Timestamp(year=year, month=month, day=1, hour=hour, minute=minute,
                         second=second, tz=zone).as_unit(unit)
    single_value_cases.append({
        "region": region, "start": describe(start), "end": describe(pd.NaT),
        "frequency": describe(pd.NaT),
        "result": captured(lambda: namespace["is_single_value"](start, pd.NaT, pd.NaT, region)),
    })
    alignment_timezones.append({
        "region": region, "input": describe(start),
        "result": captured(lambda: namespace["cal_sam_minute"](start, 1, region)),
    })

for year, zone, region in itertools.product(
    [-300000, 300000], [None, "UTC", "+08:00", "America/New_York"], ["cn", "tw", "us"],
):
    start = pd.Timestamp(np.datetime64(f"{year}-01-01T12:00:00", "s"))
    if zone is not None:
        start = start.tz_localize("UTC").tz_convert(zone)
    alignment_timezones.append({
        "region": region, "input": describe(start),
        "result": captured(lambda: namespace["cal_sam_minute"](start, 1, region)),
    })

intraday_timestamp_inputs = list(single_value_inputs.values())
for clock, unit, zone in itertools.product(
    ["09:00:00.000000001", "09:29:59.999999999", "09:30:00", "11:29:59.999999999",
     "11:30:00", "13:00:00", "13:59:59.999999999", "14:00:00", "15:00:00", "16:00:00"],
    ["s", "ms", "us", "ns"], [None, "UTC", "America/New_York"],
):
    intraday_timestamp_inputs.append(pd.Timestamp("1900-01-01 " + clock, tz=zone).as_unit(unit))
for year, zone in itertools.product([-300000, 0, 1, 9999, 10000, 300000], [None, "UTC"]):
    intraday_timestamp_inputs.append(pd.Timestamp(np.datetime64(f"{year:04d}-01-01", "s")).tz_localize(zone))
for ticks, unit in itertools.product([-(2**63) + 1, 2**63 - 1], ["s", "ms", "us", "ns"]):
    intraday_timestamp_inputs.append(pd.Timestamp(ticks, unit=unit))
intraday_timestamp_cases = []
for value, region in itertools.product(intraday_timestamp_inputs, ["cn", "us", "tw", "CN", ""]):
    intraday_timestamp_cases.append({
        "input": describe(value), "region": region,
        "result": captured(lambda: namespace["time_to_day_index"](value, region)),
    })

range_timestamp_inputs = [single_value_inputs[key] for key in ["nat", "ordinary", "cn_close", "us_close"]] + [
    pd.Timestamp("1900-01-01 09:30:00.000000001"),
    pd.Timestamp("2021-01-01 15:00:00.000000001"),
    pd.Timestamp("2021-01-01 09:30:00", tz="UTC+08:00"),
    pd.Timestamp(np.datetime64("0000-01-01", "s")),
    pd.Timestamp(np.datetime64("300000-01-01", "s")),
    pd.Timestamp(2**63 - 1, unit="s"),
    pd.Timestamp(-(2**63) + 1, unit="ms"),
    pd.Timestamp("2021-01-01 09:30:00.000001"),
]
range_timestamp_cases = []
for start, end, frequency, region in itertools.product(
    range_timestamp_inputs, range_timestamp_inputs,
    ["1min", "2day", "0min", "bad", "-1min"], ["cn", "us", "tw", "CN", ""],
):
    range_timestamp_cases.append({
        "start": describe(start), "end": describe(end), "frequency": frequency, "region": region,
        "result": captured(lambda: namespace["get_day_min_idx_range"](start, end, frequency, region)),
    })

intraday_datetime_cases = []
for day, clock, aware, region in itertools.product(
    ["1899-12-31", "1900-01-01", "2021-01-01"],
    ["09:00:00.000001", "09:29:59.999999", "09:30:00", "11:29:59.999999",
     "11:30:00", "13:00:00", "13:59:59.999999", "14:00:00", "15:00:00", "16:00:00"],
    [False, True], ["cn", "us", "tw", "CN", ""],
):
    local = datetime.fromisoformat(day + "T" + clock)
    value = local.replace(tzinfo=timezone.utc) if aware else local
    intraday_datetime_cases.append({
        "local": local.isoformat(), "aware": aware, "region": region,
        "result": captured(lambda: namespace["time_to_day_index"](value, region)),
    })

market_clock_cases = []
for hour, minute in itertools.product(
    ["0", "00", "1", "01", "2", "09", "9", "10", "19", "20", "23", "24",
     " 9", "\t9", "009", "٩", "١", "1٩", "2٣", "2３", "٢3", "９", "１9", " ٩", "9 ", "\n9", "", "x", "1x", " x"],
    ["0", "00", "1", "01", "3", "03", "30", "59", "60", "003", "٣", "٣٠",
     "3٠", "５", "5９", "５9", " 3", "30 ", "30\n", "30:00", "", "\t30"],
):
    text = hour + ":" + minute
    market_clock_cases.append({
        "input": text,
        "parse": captured(lambda: pd.Timestamp(datetime.strptime(text, "%H:%M"))),
        "index": captured(lambda: namespace["time_to_day_index"](text, "cn")),
    })

intraday_string_cases = []
for text, region in itertools.product(
    [case["input"] for case in market_clock_cases] +
    ["", "930", "9:", ":30", "9:30x", "9:30\0", "'", '"', "'\"", "é\n", "🦀", "9:30:00"],
    ["cn", "us", "tw", "CN", ""],
):
    intraday_string_cases.append({
        "input": text, "region": region,
        "result": captured(lambda: namespace["time_to_day_index"](text, region)),
    })

frequency_cases = {
    "first_text": ("day", ["01MIN"]),
    "first_freq": ("day", [Freq("01MIN")]),
    "later_text": ("day", ["1min", "02MIN"]),
    "later_freq": ("day", ["1min", Freq("02MIN")]),
    "tie": ("day", ["2min", "02MIN"]),
    "none": ("1min", ["day", "week"]),
    "huge": ("9" * 80 + "min", ["1min", "2min"]),
    "invalid_after_eligible": ("day", ["1min", "bad"]),
}
frequency = {
    name: captured(lambda base=base, values=values: Freq.get_recent_freq(base, values))
    for name, (base, values) in frequency_cases.items()
}
frequency_newlines = [
    {"input": value, "result": captured(lambda: Freq(value))}
    for value in ["day\n", "1MIN\n", "02W\n", "day\r\n", "day\n\n", "day \n", "\nday", "\n"]
]

concat = {
    "minimum": describe(namespace["concat_date_time"](date(1, 1, 1), time(1, 2, 3, 456789))),
    "ordinary": describe(namespace["concat_date_time"](date(2020, 2, 29), time(1, 2, 3, 456789))),
    "maximum": describe(namespace["concat_date_time"](date(9999, 12, 31), time(23, 59, 59, 999999))),
}

epsilon_inputs = {
    "seconds": pd.Timestamp(np.datetime64("2021-01-01T00:00:00", "s")),
    "microseconds": pd.Timestamp(np.datetime64("2021-01-01T00:00:00.123456", "us")),
    "nanoseconds": pd.Timestamp("2021-01-01 00:00:00.123456789"),
    "iana": pd.Timestamp("2021-01-01 09:30:45.123456789", tz="Asia/Shanghai"),
    "nat": pd.NaT,
}
epsilon = {}
for name, value in epsilon_inputs.items():
    for direction in ["backward", "forward"]:
        epsilon[f"{name}_{direction}"] = captured(
            lambda value=value, direction=direction: namespace["epsilon_change"](value, direction)
        )
epsilon["nat_invalid_direction"] = captured(
    lambda: namespace["epsilon_change"](pd.NaT, "Backward")
)
epsilon["minimum_backward"] = captured(
    lambda: namespace["epsilon_change"](pd.Timestamp.min, "backward")
)
epsilon["maximum_forward"] = captured(
    lambda: namespace["epsilon_change"](pd.Timestamp.max, "forward")
)

alignment_inputs = {
    "naive": pd.Timestamp("2021-01-01 10:38:45.123456789"),
    "utc": pd.Timestamp("2021-01-01 10:38:45.123456789", tz="UTC"),
    "fixed": pd.Timestamp("2021-01-01 10:38:45.123456789+08:00"),
    "new_york_summer": pd.Timestamp("2021-07-01 14:38:45.123456789Z").tz_convert("America/New_York"),
    "new_york_winter": pd.Timestamp("2021-01-01 15:38:45.123456789Z").tz_convert("America/New_York"),
}
alignment = {
    name: captured(lambda value=value: namespace["cal_sam_minute"](value, 5))
    for name, value in alignment_inputs.items()
}
alignment["nat"] = captured(lambda: namespace["cal_sam_minute"](pd.NaT, 5))
alignment["zero_step"] = captured(
    lambda: namespace["cal_sam_minute"](pd.Timestamp("2021-01-01 10:38"), 0)
)

# Combined invalid inputs expose source evaluation order, not just isolated errors.
alignment_precedence = []
for region in ["cn", "us", "tw"]:
    for shift in [0, 10**100, -(10**100)]:
        namespace["C"].min_data_shift = shift
        for step in [0, 1, -1]:
            for is_nat in [False, True]:
                value = pd.NaT if is_nat else pd.Timestamp("2021-01-01 10:38")
                alignment_precedence.append({
                    "region": region, "shift": str(shift), "step": step, "is_nat": is_nat,
                    "result": captured(lambda: namespace["cal_sam_minute"](value, step, region)),
                })
namespace["C"].min_data_shift = 0

calendar = namespace["get_min_cal"]
calendar.cache_clear()
cache_input = pd.Timestamp("2021-01-01 10:38")
downstream_cache = {
    "before": captured(lambda: namespace["cal_sam_minute"](cache_input, 1)),
}
calendar(0, "cn")[:] = [time(9, 0)]
downstream_cache["mutated"] = captured(lambda: namespace["cal_sam_minute"](cache_input, 1))
calendar(0, "cn").clear()
downstream_cache["empty"] = captured(lambda: namespace["cal_sam_minute"](cache_input, 1))
downstream_cache["empty_nat"] = captured(lambda: namespace["cal_sam_minute"](pd.NaT, 1))
downstream_cache["empty_zero"] = captured(lambda: namespace["cal_sam_minute"](pd.NaT, 0))
calendar.cache_clear()

range_cache = []
for region in ["cn", "us", "tw"]:
    for minutes in [[], [540], [540, 570, 570, 600], [600, 540, 660, 570]]:
        for step in [0, 1, 2, 10**100]:
            for start, end in [(540, 570), (600, 540), (0, 1439)]:
                calendar.cache_clear()
                calendar(region=region)[:] = [time(m // 60, m % 60) for m in minutes]
                # A different positional call must not overwrite the keyword-only entry.
                calendar(0, region)[:] = [time(23, 59)]
                range_cache.append({
                    "region": region, "minutes": minutes, "step": str(step),
                    "start": start, "end": end,
                    "result": captured(lambda: namespace["get_day_min_idx_range"](
                        f"2021-01-01 {start // 60:02}:{start % 60:02}",
                        f"2021-01-01 {end // 60:02}:{end % 60:02}", f"{step}min", region)),
                })
calendar.cache_clear()

range_errors = [
    {"region": region, "step": step,
     "result": captured(lambda: namespace["get_day_min_idx_range"](
         "2021-01-01 09:30", "2021-01-01 10:00", f"{step}min", region))}
    for region in ["xx", "CN", "", "cn "] for step in [0, 1]
]

date_precedence = []
for year in [0, 10000]:
    for empty in [False, True]:
        for step in [0, 1]:
            calendar.cache_clear()
            if empty:
                calendar(0, "cn").clear()
            value = pd.Timestamp(np.datetime64(f"{year:04d}-01-01", "s"))
            date_precedence.append({
                "ticks": int(value.asm8.view("i8")), "year": year,
                "empty": empty, "step": step,
                "result": captured(lambda: namespace["cal_sam_minute"](value, step)),
            })
calendar.cache_clear()

duration_cases = []
with warnings.catch_warnings(record=True) as duration_warnings:
    warnings.simplefilter("always", FutureWarning)
    for unit in ["w", "W", "day", "days", "D", "h", "H", "hr", "hour", "hours",
                 "m", "min", "MIN", "minute", "minutes", "T", "s", "sec", "second", "seconds",
                 "ms", "MS", "millisecond", "milliseconds", "L", "us", "US", "µs", "microsecond",
                 "microseconds", "U", "ns", "NS", "nanosecond", "nanoseconds", "N",
                 "M", "Y", "y", "week", "month", "mon", "μs", "bad",
                 "", "S", "t", "l", "u", "n", "milli", "millis", "micro", "micros", "nano", "nanos"]:
        for count in [-106752, -2, 0, 1, 106751, 106752, 2**53 + 1, 10**100, 10**400]:
            duration_warnings.clear()
            result = captured(lambda: Freq.get_timedelta(count, unit))
            duration_cases.append({"count": str(count), "unit": unit,
                "result": result, "warnings": [
                    {"category": warning.category.__name__, "message": str(warning.message)}
                    for warning in duration_warnings]})

compound_duration_cases = []
for unit in [" day 2h", "h30m", "1h30m", "h 1H 2T", "h+1m", " h ",
             "h-1m", "h1", "H1bad", "h1M", "h 1H 2T 3H", "milli2micro3nano",
             "", "123", "h,2m", "1 2h", "\th\n", "-h", "h0m",
             "ns9223372036854774784ns1023ns", "ns9223372036854774784ns1024ns",
             "ns9223372036854774784ns1025ns", "ns9000000000000000000ns9000000000000000000ns",
             "ns9223372036854775808ns", "H9223372036854774784ns1N",
             "ns9000000000000000000ns9000000000000000000ns1bad",
             ":02:03", ":99:99", " days, 01:02:03.4", "h 01:02:03",
             "H 01:02:03.1234567899", ":02", "::03", ":02.3",
             ":02:03:04", ":02:03.000000001", ":02:03.123", ":02:03.123456",
             ":02:", ":02:03.", ":02:.1", ":02:03..4", ":9223372036854775808:03",
             ".5h", ".25days", ".1s", ".000000001s", ".0000000005s", ".1234567895s",
             ".5ns", ".9999999999us", ".1ms", ".25w", ".5H2.25T", "h0.5m",
             ".5day 01:02:03.4", ".5", ".5bad", ".5M",
             "h.5m", "h.5", "h.", ".5h.2m", ".5h2.3m", ":2:3.4h",
             ":2:3.4h5m", ":2:3h", ":2:3h4m", ":2:3.4.5", ":.1",
             ":2:.1h", ".5:2:3", ".5:2:3.4", ":2:3.4h.", ".h", ":2:3.-4",
             ":2:3.h4", ":2:3.h4m", ":2:3.h.", ":2:9223372036854775808.1"]:
    for count in [-2, -1, 0, 1, 2]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = captured(lambda: Freq.get_timedelta(count, unit))
            compound_duration_cases.append({"count": str(count), "unit": unit,
                "result": result, "warnings": [str(w.message) for w in caught]})

decimal_duration_cases = []
for unit in ["w", "day", "h", "min", "s", "ms", "us", "ns", "H", "T", "N"]:
    for fraction in ["1", "5", "0000000005", "1234567895", "9999999999", "0000015", "000000001", ""]:
        for count in [-106752, -2, 0, 1, 106751, 2**53 + 1, 10**100, 10**400]:
            suffix = "." + fraction + unit
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = captured(lambda: Freq.get_timedelta(count, suffix))
                decimal_duration_cases.append({"count": str(count), "unit": suffix,
                    "result": result, "warnings": [str(w.message) for w in caught]})

if "--generated-duration" in sys.argv[2:]:
    cases = []
    for length in range(5):
        for chars in itertools.product("01hH.:-+ ", repeat=length):
            suffix = "".join(chars)
            for count in [-1, 0, 1]:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = captured(lambda: Freq.get_timedelta(count, suffix))
                    cases.append({"count": str(count), "unit": suffix, "result": result,
                        "warnings": [str(w.message) for w in caught]})
    print(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(), "cases": cases}))
    sys.exit(0)

print(json.dumps({
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "pandas_version": pd.__version__,
    "pytz_version": pytz.__version__,
    "numpy_version": np.__version__,
    "frequency": frequency,
    "single_value_cases": single_value_cases,
    "frequency_newlines": frequency_newlines,
    "concat": concat,
    "epsilon": epsilon,
    "alignment": alignment,
    "alignment_timezones": alignment_timezones,
    "market_clock_cases": market_clock_cases,
    "intraday_datetime_cases": intraday_datetime_cases,
    "intraday_string_cases": intraday_string_cases,
    "intraday_timestamp_cases": intraday_timestamp_cases,
    "range_timestamp_cases": range_timestamp_cases,
    "alignment_precedence": alignment_precedence,
    "downstream_cache": downstream_cache,
    "range_cache": range_cache,
    "range_errors": range_errors,
    "date_precedence": date_precedence,
    "duration_cases": duration_cases,
    "compound_duration_cases": compound_duration_cases,
    "decimal_duration_cases": decimal_duration_cases,
}, sort_keys=True))
