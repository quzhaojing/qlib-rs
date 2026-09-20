#!/usr/bin/env python3
"""Build the auditable Python-to-Rust migration symbol inventory.

The generator deliberately separates proven acceptance from name-based evidence
linking.  A ledger row can make a symbol an evidence candidate, but only the
whole-file acceptance registry can mark every symbol in a file accepted.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable


SCHEMA_VERSION = 1
SOURCE_SUFFIXES = {".py", ".pyx"}
PRODUCTION_EXCLUDES = ("qlib/tests/",)
FLOW_NAMES = {
    "backtest": "backtest-execution",
    "cli": "command-line",
    "contrib": "contributed-models-workflows",
    "data": "data-provider-storage-expression",
    "model": "model-training-inference",
    "rl": "reinforcement-learning",
    "strategy": "strategy-signal-portfolio",
    "utils": "shared-utilities",
    "workflow": "experiment-workflow-recording",
}
LEDGER_HEADER = "| Slice | Python surface | Rust target | Status | Evidence | Blockers |"
INVENTORY_COLUMNS = (
    "schema_version",
    "source_file",
    "source_sha256",
    "module",
    "flow",
    "symbol_id",
    "parent_symbol_id",
    "qualname",
    "name",
    "kind",
    "line_start",
    "line_end",
    "visibility",
    "decorators",
    "status",
    "whole_file_evidence",
    "ledger_slice_ids",
    "rust_targets",
    "rust_locations",
    "acceptance_evidence",
    "blockers",
)
EVIDENCE_COLUMNS = (
    "schema_version",
    "slice_id",
    "slice",
    "python_surface",
    "rust_target",
    "status",
    "evidence",
    "blockers",
    "matched_symbol_count",
    "matched_symbol_ids",
    "rust_locations",
)


@dataclass
class Symbol:
    source_file: str
    source_sha256: str
    module: str
    flow: str
    qualname: str
    name: str
    kind: str
    line_start: int
    line_end: int
    parent_symbol_id: str = ""
    decorators: tuple[str, ...] = ()
    whole_file_evidence: str = ""
    ledger_slice_ids: set[str] = field(default_factory=set)
    rust_targets: set[str] = field(default_factory=set)
    rust_locations: set[str] = field(default_factory=set)
    acceptance_evidence: set[str] = field(default_factory=set)
    blockers: set[str] = field(default_factory=set)

    @property
    def symbol_id(self) -> str:
        return f"{self.source_file}::{self.qualname}@{self.line_start}"

    @property
    def visibility(self) -> str:
        return "private" if self.name.startswith("_") and not self.name.startswith("__") else "public"

    @property
    def status(self) -> str:
        if self.whole_file_evidence:
            return "whole_file_accepted" if self.kind == "module" else "accepted_by_whole_file"
        if self.ledger_slice_ids:
            return "ledger_evidence_candidate"
        return "not_started"

    def csv_row(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "module": self.module,
            "flow": self.flow,
            "symbol_id": self.symbol_id,
            "parent_symbol_id": self.parent_symbol_id,
            "qualname": self.qualname,
            "name": self.name,
            "kind": self.kind,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "visibility": self.visibility,
            "decorators": json.dumps(self.decorators, ensure_ascii=False),
            "status": self.status,
            "whole_file_evidence": self.whole_file_evidence,
            "ledger_slice_ids": json.dumps(sorted(self.ledger_slice_ids), ensure_ascii=False),
            "rust_targets": json.dumps(sorted(self.rust_targets), ensure_ascii=False),
            "rust_locations": json.dumps(sorted(self.rust_locations), ensure_ascii=False),
            "acceptance_evidence": json.dumps(sorted(self.acceptance_evidence), ensure_ascii=False),
            "blockers": json.dumps(sorted(self.blockers), ensure_ascii=False),
        }


@dataclass(frozen=True)
class LedgerSlice:
    slice_id: str
    name: str
    python_surface: str
    rust_target: str
    status: str
    evidence: str
    blockers: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"D:\code\github\qlib"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="fail if committed/generated outputs differ")
    return parser.parse_args()


def git_production_files(source: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "--", "qlib"],
        text=True,
        encoding="utf-8",
    )
    files = []
    for value in output.splitlines():
        path = PurePosixPath(value)
        if path.suffix not in SOURCE_SUFFIXES or value.startswith(PRODUCTION_EXCLUDES):
            continue
        files.append(value)
    return sorted(files)


def module_name(source_file: str) -> str:
    path = PurePosixPath(source_file)
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def flow_name(source_file: str) -> str:
    parts = PurePosixPath(source_file).parts
    area = parts[1] if len(parts) > 2 else "root"
    return FLOW_NAMES.get(area, "configuration-bootstrap" if area == "root" else area)


def decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = decorator_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return decorator_name(node.func)
    return ast.dump(node, include_attributes=False)


def assigned_names(node: ast.Assign | ast.AnnAssign) -> Iterable[tuple[str, int, int]]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        candidates = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
        for candidate in candidates:
            if isinstance(candidate, ast.Name):
                yield candidate.id, node.lineno, getattr(node, "end_lineno", node.lineno)


class PythonSymbolVisitor(ast.NodeVisitor):
    def __init__(self, source_file: str, digest: str):
        self.source_file = source_file
        self.digest = digest
        self.module = module_name(source_file)
        self.flow = flow_name(source_file)
        self.symbols: list[Symbol] = []
        self.parents: list[Symbol] = []
        self.function_depth = 0

    def add(self, name: str, kind: str, start: int, end: int, decorators: tuple[str, ...] = ()) -> Symbol:
        qualname = ".".join([*(parent.name for parent in self.parents), name])
        parent_id = self.parents[-1].symbol_id if self.parents else f"{self.source_file}::<module>@1"
        symbol = Symbol(
            self.source_file,
            self.digest,
            self.module,
            self.flow,
            qualname,
            name,
            kind,
            start,
            end,
            parent_id,
            decorators,
        )
        self.symbols.append(symbol)
        return symbol

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        symbol = self.add(
            node.name,
            "nested_class" if self.function_depth else "class",
            node.lineno,
            getattr(node, "end_lineno", node.lineno),
            tuple(decorator_name(item) for item in node.decorator_list),
        )
        self.parents.append(symbol)
        self.generic_visit(node)
        self.parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        inside_class = bool(self.parents and "class" in self.parents[-1].kind)
        if self.function_depth:
            kind = "nested_async_function" if is_async else "nested_function"
        elif inside_class:
            kind = "async_method" if is_async else "method"
        else:
            kind = "async_function" if is_async else "function"
        symbol = self.add(
            node.name,
            kind,
            node.lineno,
            getattr(node, "end_lineno", node.lineno),
            tuple(decorator_name(item) for item in node.decorator_list),
        )
        self.parents.append(symbol)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1
        self.parents.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        self._visit_assignment(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_assignment(node)
        self.generic_visit(node)

    def _visit_assignment(self, node: ast.Assign | ast.AnnAssign) -> None:
        if self.function_depth:
            return
        kind = "class_attribute" if self.parents and "class" in self.parents[-1].kind else "module_variable"
        for name, start, end in assigned_names(node):
            self.add(name, kind, start, end)


CYTHON_CLASS = re.compile(r"^(?P<indent>\s*)(?:cdef\s+)?class\s+(?P<name>[A-Za-z_]\w*)")
CYTHON_PY_FUNCTION = re.compile(r"^(?P<indent>\s*)def\s+(?P<name>[A-Za-z_]\w*)\s*\(")
CYTHON_C_FUNCTION = re.compile(r"^(?P<indent>\s*)(?:cdef|cpdef)\s+(?P<body>.+)$")
CYTHON_ATTRIBUTE = re.compile(
    r"^(?P<indent>\s*)cdef\s+(?!class\b)(?!.*\()(?P<type>.+\s+)(?P<name>[A-Za-z_]\w*)(?:\s*=.*)?$"
)


def cython_block_end(lines: list[str], start: int, indent: int) -> int:
    end = start
    for line_number in range(start + 1, len(lines) + 1):
        line = lines[line_number - 1]
        if not line.strip() or line.lstrip().startswith("#"):
            end = line_number
            continue
        prefix_length = len(line) - len(line.lstrip(" \t"))
        current_indent = len(line[:prefix_length].expandtabs(8))
        if current_indent <= indent:
            break
        end = line_number
    return end


def cython_function_parts(line: str) -> tuple[str, str] | None:
    python_match = CYTHON_PY_FUNCTION.match(line)
    if python_match:
        return python_match.group("indent"), python_match.group("name")
    c_match = CYTHON_C_FUNCTION.match(line)
    if c_match is None:
        return None
    body = c_match.group("body")
    square_depth = 0
    opening = None
    for index, character in enumerate(body):
        if character == "[":
            square_depth += 1
        elif character == "]":
            square_depth = max(0, square_depth - 1)
        elif character == "=" and square_depth == 0:
            return None
        elif character == "(" and square_depth == 0:
            opening = index
            break
    if opening is None:
        return None
    names = re.findall(r"[A-Za-z_]\w*", body[:opening])
    return (c_match.group("indent"), names[-1]) if names else None


def cython_symbols(source_file: str, text: str, digest: str) -> list[Symbol]:
    module = module_name(source_file)
    flow = flow_name(source_file)
    symbols: list[Symbol] = []
    classes: list[tuple[int, Symbol]] = []
    lines = text.splitlines()
    for line_number, line in enumerate(lines, 1):
        class_match = CYTHON_CLASS.match(line)
        function_parts = cython_function_parts(line)
        attribute_match = CYTHON_ATTRIBUTE.match(line)
        if class_match is None and function_parts is None and attribute_match is None:
            continue
        indent_text = (
            class_match.group("indent")
            if class_match
            else function_parts[0]
            if function_parts
            else attribute_match.group("indent")
        )
        indent = len(indent_text.expandtabs(8))
        while classes and classes[-1][0] >= indent:
            classes.pop()
        name = (
            class_match.group("name")
            if class_match
            else function_parts[1]
            if function_parts
            else attribute_match.group("name")
        )
        parent = classes[-1][1] if classes else None
        if attribute_match:
            if parent is None or indent != classes[-1][0] + 4:
                continue
            kind = "class_attribute"
        else:
            kind = "class" if class_match else ("method" if parent else "function")
        qualname = f"{parent.qualname}.{name}" if parent else name
        symbol = Symbol(
            source_file,
            digest,
            module,
            flow,
            qualname,
            name,
            kind,
            line_number,
            line_number if attribute_match else cython_block_end(lines, line_number, indent),
            parent.symbol_id if parent else f"{source_file}::<module>@1",
        )
        symbols.append(symbol)
        if class_match:
            classes.append((indent, symbol))
    return symbols


def collect_symbols(source: Path, source_files: list[str]) -> list[Symbol]:
    result: list[Symbol] = []
    for source_file in source_files:
        data = (source / source_file).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8")
        line_count = max(1, len(text.splitlines()))
        result.append(
            Symbol(source_file, digest, module_name(source_file), flow_name(source_file), "<module>", "<module>", "module", 1, line_count)
        )
        if source_file.endswith(".py"):
            visitor = PythonSymbolVisitor(source_file, digest)
            visitor.visit(ast.parse(text, filename=source_file))
            result.extend(visitor.symbols)
        else:
            result.extend(cython_symbols(source_file, text, digest))
    return result


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip().replace("\\|", "|") for cell in line.strip().strip("|").split("|")]


def ledger_slices(status_path: Path) -> list[LedgerSlice]:
    lines = status_path.read_text(encoding="utf-8").splitlines()
    start = lines.index(LEDGER_HEADER) + 2
    result = []
    for index, line in enumerate(lines[start:], 1):
        if not line.startswith("|"):
            break
        cells = split_markdown_row(line)
        if len(cells) != 6:
            raise ValueError(f"unexpected ledger row: {line}")
        result.append(LedgerSlice(f"ledger-{index:03d}", *cells))
    return result


def whole_file_acceptance(inventory_path: Path) -> dict[str, str]:
    result = {}
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `qlib/") or "已验收" not in line:
            continue
        cells = split_markdown_row(line)
        result[cells[0].strip("`")] = cells[2]
    return result


def source_file_registry(inventory_path: Path) -> dict[str, tuple[str, str]]:
    result = {}
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `qlib/"):
            continue
        cells = split_markdown_row(line)
        if len(cells) != 4 or cells[1] != "package":
            continue
        result[cells[0].strip("`")] = (cells[2], cells[3].strip("`"))
    return result


def normalized_surface(value: str) -> str:
    return value.replace("`", "").replace("{", "").replace("}", "")


RUST_DEFINITION = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
    r"(?:struct|enum|trait|type|const|static|fn|mod)\s+(?P<name>[A-Za-z_]\w*)\b"
)


def rust_definition_index(repo: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for path in sorted((repo / "crates").glob("*/src/**/*.rs")):
        relative = path.relative_to(repo).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = RUST_DEFINITION.match(line)
            if match:
                name = match.group("name")
                result[name].append(f"{relative}:{line_number}:{name}")
    return result


def rust_target_locations(target: str, index: dict[str, list[str]]) -> list[str]:
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", target.replace("`", "")))
    return sorted({location for token in tokens for location in index.get(token, ())})


def link_ledger(
    symbols: list[Symbol],
    slices: list[LedgerSlice],
    rust_index: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    by_file: dict[str, list[Symbol]] = defaultdict(list)
    by_name: dict[str, list[Symbol]] = defaultdict(list)
    for symbol in symbols:
        by_file[symbol.source_file].append(symbol)
        if symbol.kind != "module":
            by_name[symbol.name].append(symbol)

    links: dict[str, list[str]] = {}
    rust_links: dict[str, list[str]] = {}
    for item in slices:
        surface = normalized_surface(item.python_surface)
        named_source_files = set(re.findall(r"qlib/[A-Za-z0-9_./-]+\.(?:py|pyx)", surface))
        explicit_file_hints = {
            symbol.source_file
            for symbol in symbols
            if symbol.kind == "module" and symbol.source_file in named_source_files
        }
        mentioned_modules = (
            []
            if explicit_file_hints
            else [
                symbol
                for symbol in symbols
                if symbol.kind == "module" and symbol.module in surface
            ]
        )
        maximal_modules = [
            symbol
            for symbol in mentioned_modules
            if not any(
                other.module.startswith(symbol.module + ".")
                for other in mentioned_modules
                if other is not symbol
            )
        ]
        file_hints = explicit_file_hints | {symbol.source_file for symbol in maximal_modules}
        candidates: dict[str, Symbol] = {}
        for source_file in file_hints:
            candidates[by_file[source_file][0].symbol_id] = by_file[source_file][0]

        # A method token in ``Class.method`` is not an isolated name hint. Treating it as one
        # links every common method (notably ``__init__``) in an otherwise correctly narrowed
        # source file. The qualified-name pass below owns those exact matches.
        tokens = set(re.findall(r"(?<!\.)\b[A-Za-z_][A-Za-z0-9_]*\b", surface))
        for name in tokens:
            named = by_name.get(name, [])
            if file_hints:
                named = [symbol for symbol in named if symbol.source_file in file_hints]
            elif len({symbol.source_file for symbol in named}) > 1:
                continue
            for symbol in named:
                candidates[symbol.symbol_id] = symbol

        # A unique Class.method spelling is stronger than an isolated common method name.
        for symbol in symbols:
            if symbol.kind == "module" or "." not in symbol.qualname:
                continue
            if symbol.qualname in surface and (not file_hints or symbol.source_file in file_hints):
                candidates[symbol.symbol_id] = symbol

        matched = sorted(candidates)
        links[item.slice_id] = matched
        rust_links[item.slice_id] = (
            []
            if item.rust_target == "N/A (source-only absence contract)"
            else rust_target_locations(item.rust_target, rust_index)
        )
        for symbol_id in matched:
            symbol = candidates[symbol_id]
            symbol.ledger_slice_ids.add(item.slice_id)
            symbol.rust_targets.add(item.rust_target)
            symbol.rust_locations.update(rust_links[item.slice_id])
            symbol.acceptance_evidence.add(item.evidence)
            if item.blockers and item.blockers != "None":
                symbol.blockers.add(item.blockers)
    return links, rust_links


def validate_inventory(
    symbols: list[Symbol],
    source_files: list[str],
    source_registry: dict[str, tuple[str, str]],
    slices: list[LedgerSlice],
    links: dict[str, list[str]],
    rust_links: dict[str, list[str]],
) -> None:
    if set(source_files) != set(source_registry):
        missing = sorted(set(source_files).difference(source_registry))
        obsolete = sorted(set(source_registry).difference(source_files))
        raise RuntimeError(f"source baseline mismatch; missing={missing}, obsolete={obsolete}")

    modules = [symbol for symbol in symbols if symbol.kind == "module"]
    if len(modules) != len(source_files) or {symbol.source_file for symbol in modules} != set(source_files):
        raise RuntimeError("every source file must have exactly one module-flow record")

    symbol_ids = [symbol.symbol_id for symbol in symbols]
    if len(symbol_ids) != len(set(symbol_ids)):
        duplicates = [name for name, count in Counter(symbol_ids).items() if count > 1]
        raise RuntimeError(f"duplicate symbol ids: {duplicates}")
    known_ids = set(symbol_ids)
    broken_parents = sorted(
        symbol.symbol_id
        for symbol in symbols
        if symbol.kind != "module" and symbol.parent_symbol_id not in known_ids
    )
    if broken_parents:
        raise RuntimeError(f"symbols with missing parents: {broken_parents}")

    for module in modules:
        expected_digest = source_registry[module.source_file][1]
        if module.source_sha256 != expected_digest:
            raise RuntimeError(
                f"source hash differs from baseline for {module.source_file}: "
                f"{module.source_sha256} != {expected_digest}"
            )

    unlinked = [item.slice_id for item in slices if not links[item.slice_id]]
    if unlinked:
        raise RuntimeError(f"ledger slices without source candidates: {unlinked}")
    rust_unlinked = [
        item.slice_id
        for item in slices
        if not rust_links[item.slice_id] and item.rust_target != "N/A (source-only absence contract)"
    ]
    if rust_unlinked:
        raise RuntimeError(f"ledger slices without Rust definition candidates: {rust_unlinked}")


def csv_text(columns: tuple[str, ...], rows: Iterable[dict[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def markdown_summary(
    source: Path,
    source_revision: str,
    symbols: list[Symbol],
    slices: list[LedgerSlice],
    links: dict[str, list[str]],
    rust_links: dict[str, list[str]],
    accepted_files: dict[str, str],
) -> str:
    files = {symbol.source_file for symbol in symbols}
    status_counts = Counter(symbol.status for symbol in symbols)
    kind_counts = Counter(symbol.kind for symbol in symbols)
    flow_counts = Counter(symbol.flow for symbol in symbols)
    linked_slices = sum(bool(links[item.slice_id]) for item in slices)
    rust_applicable_slices = [item for item in slices if item.rust_target != "N/A (source-only absence contract)"]
    rust_linked_slices = sum(bool(rust_links[item.slice_id]) for item in rust_applicable_slices)
    accepted_symbols = status_counts["whole_file_accepted"] + status_counts["accepted_by_whole_file"]
    completion = accepted_symbols / len(symbols) * 100
    lines = [
        "# Python → Rust 符号级迁移清单",
        "",
        f"来源：`{source.as_posix()}`，Git HEAD `{source_revision}`。清单 schema v{SCHEMA_VERSION}；由 `scripts/build-symbol-inventory.py` 生成。",
        "",
        "本清单完整枚举包内生产 Python/Cython 文件及其模块、类、函数、方法、嵌套函数和静态属性。台账名称匹配只标记为 `ledger_evidence_candidate`，不自动计为完成；只有整文件验收记录当前能把对应符号标记为已验收。这样避免把局部切片、同名方法或仅存在的 Rust 文件误算为完整迁移。",
        "",
        "## 当前摘要",
        "",
        "| 指标 | 当前值 |",
        "|---|---:|",
        f"| 包内生产源文件 | {len(files)} |",
        f"| Python/Cython 符号记录（含每文件模块流程记录） | {len(symbols)} |",
        f"| 整文件已验收 | {len(accepted_files)} / {len(files)} ({len(accepted_files) / len(files) * 100:.2f}%) |",
        f"| 因整文件验收而严格完成的符号 | {accepted_symbols} / {len(symbols)} ({completion:.2f}%) |",
        f"| 台账功能切片 | {len(slices)}（{sum(item.status == 'done' for item in slices)} done，{sum(item.status == 'in_progress' for item in slices)} in_progress） |",
        f"| 已链接到至少一个候选符号的台账切片 | {linked_slices} / {len(slices)} |",
        f"| 需要 Rust 定义且已链接的台账切片 | {rust_linked_slices} / {len(rust_applicable_slices)} |",
        "",
        "符号完成率同样是严格下界：台账中的局部功能切片仍需逐项人工确认其精确符号边界，候选链接本身不进入完成分子。",
        "",
        "## 状态分布",
        "",
        "| 状态 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in sorted(status_counts.items()))
    lines.extend(["", "## 符号类型", "", "| 类型 | 数量 |", "|---|---:|"])
    lines.extend(f"| `{name}` | {count} |" for name, count in sorted(kind_counts.items()))
    lines.extend(
        [
            "",
            "## 业务流程",
            "",
            "| 流程 | 文件数 | 符号数 | 严格完成 | 台账候选 | 未开始 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for flow, count in sorted(flow_counts.items()):
        flow_symbols = [symbol for symbol in symbols if symbol.flow == flow]
        flow_files = len({symbol.source_file for symbol in flow_symbols})
        strict = sum(symbol.status in {"whole_file_accepted", "accepted_by_whole_file"} for symbol in flow_symbols)
        candidates = sum(symbol.status == "ledger_evidence_candidate" for symbol in flow_symbols)
        not_started = sum(symbol.status == "not_started" for symbol in flow_symbols)
        lines.append(f"| `{flow}` | {flow_files} | {count} | {strict} | {candidates} | {not_started} |")
    lines.extend(
        [
            "",
            "## 机器可读清单",
            "",
            "- `migration-symbol-inventory.csv`：每行一个源符号，包含源哈希、流程、父子关系、状态、Rust 目标、Rust 文件/行候选、验收候选证据和阻塞项。",
            "- `migration-symbol-evidence.csv`：每行一个迁移台账切片，保留 Python 表面、Rust 目标、Rust 文件/行候选、测试/覆盖率证据及所有候选符号 ID。",
            "",
            "重新生成：",
            "",
            "```powershell",
            "python scripts/build-symbol-inventory.py",
            "python scripts/build-symbol-inventory.py --check",
            "```",
            "",
            "`--check` 会重新读取 Git 跟踪的上游生产文件并逐字比较三个生成物，可用于防止源文件或台账变化后清单失效。",
            "",
        ]
    )
    return "\n".join(lines)


def update_output(path: Path, expected: str, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == expected:
        return False
    if check:
        print(f"out of date: {path}", file=sys.stderr)
        return True
    path.write_text(expected, encoding="utf-8", newline="")
    print(f"wrote {path}")
    return False


def main() -> int:
    args = parse_args()
    source_files = git_production_files(args.source)
    if len(source_files) != 230:
        raise RuntimeError(f"expected 230 package production files, found {len(source_files)}")

    symbols = collect_symbols(args.source, source_files)
    source_inventory_path = args.repo / "docs" / "migration-source-inventory.md"
    source_registry = source_file_registry(source_inventory_path)
    accepted_files = whole_file_acceptance(source_inventory_path)
    unknown_accepted = set(accepted_files).difference(source_files)
    if unknown_accepted:
        raise RuntimeError(f"accepted files absent from source inventory: {sorted(unknown_accepted)}")
    for symbol in symbols:
        symbol.whole_file_evidence = accepted_files.get(symbol.source_file, "")

    slices = ledger_slices(args.repo / "docs" / "migration-status.md")
    rust_index = rust_definition_index(args.repo)
    links, rust_links = link_ledger(symbols, slices, rust_index)
    validate_inventory(symbols, source_files, source_registry, slices, links, rust_links)
    source_revision = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    symbol_csv = csv_text(INVENTORY_COLUMNS, (symbol.csv_row() for symbol in symbols))
    evidence_csv = csv_text(
        EVIDENCE_COLUMNS,
        (
            {
                "schema_version": SCHEMA_VERSION,
                "slice_id": item.slice_id,
                "slice": item.name,
                "python_surface": item.python_surface,
                "rust_target": item.rust_target,
                "status": item.status,
                "evidence": item.evidence,
                "blockers": item.blockers,
                "matched_symbol_count": len(links[item.slice_id]),
                "matched_symbol_ids": json.dumps(links[item.slice_id], ensure_ascii=False),
                "rust_locations": json.dumps(rust_links[item.slice_id], ensure_ascii=False),
            }
            for item in slices
        ),
    )
    summary = markdown_summary(
        args.source,
        source_revision,
        symbols,
        slices,
        links,
        rust_links,
        accepted_files,
    )

    outputs = {
        args.repo / "docs" / "migration-symbol-inventory.csv": symbol_csv,
        args.repo / "docs" / "migration-symbol-evidence.csv": evidence_csv,
        args.repo / "docs" / "migration-symbol-inventory.md": summary,
    }
    stale = False
    for path, expected in outputs.items():
        stale |= update_output(path, expected, args.check)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
