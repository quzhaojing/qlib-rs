# Python → Rust 符号级迁移清单

来源：`D:/code/github/qlib`，Git HEAD `79633dd9506ea689e5400dea0197717b5b3d74b7`。清单 schema v1；由 `scripts/build-symbol-inventory.py` 生成。

本清单完整枚举包内生产 Python/Cython 文件及其模块、类、函数、方法、嵌套函数和静态属性。台账名称匹配只标记为 `ledger_evidence_candidate`，不自动计为完成；只有整文件验收记录当前能把对应符号标记为已验收。这样避免把局部切片、同名方法或仅存在的 Rust 文件误算为完整迁移。

## 当前摘要

| 指标 | 当前值 |
|---|---:|
| 包内生产源文件 | 230 |
| Python/Cython 符号记录（含每文件模块流程记录） | 3872 |
| 整文件已验收 | 32 / 230 (13.91%) |
| 因整文件验收而严格完成的符号 | 131 / 3872 (3.38%) |
| 台账功能切片 | 112（111 done，1 in_progress） |
| 已链接到至少一个候选符号的台账切片 | 112 / 112 |
| 需要 Rust 定义且已链接的台账切片 | 111 / 111 |

符号完成率同样是严格下界：台账中的局部功能切片仍需逐项人工确认其精确符号边界，候选链接本身不进入完成分子。

## 状态分布

| 状态 | 数量 |
|---|---:|
| `accepted_by_whole_file` | 99 |
| `ledger_evidence_candidate` | 129 |
| `not_started` | 3612 |
| `whole_file_accepted` | 32 |

## 符号类型

| 类型 | 数量 |
|---|---:|
| `class` | 539 |
| `class_attribute` | 288 |
| `function` | 289 |
| `method` | 2276 |
| `module` | 230 |
| `module_variable` | 179 |
| `nested_class` | 1 |
| `nested_function` | 70 |

## 业务流程

| 流程 | 文件数 | 符号数 | 严格完成 | 台账候选 | 未开始 |
|---|---:|---:|---:|---:|---:|
| `backtest-execution` | 12 | 402 | 14 | 42 | 346 |
| `command-line` | 3 | 9 | 1 | 0 | 8 |
| `configuration-bootstrap` | 5 | 109 | 10 | 5 | 94 |
| `contributed-models-workflows` | 102 | 1210 | 12 | 0 | 1198 |
| `data-provider-storage-expression` | 22 | 813 | 1 | 2 | 810 |
| `experiment-workflow-recording` | 16 | 346 | 2 | 0 | 344 |
| `model-training-inference` | 18 | 155 | 2 | 3 | 150 |
| `reinforcement-learning` | 38 | 551 | 83 | 65 | 403 |
| `shared-utilities` | 12 | 253 | 5 | 11 | 237 |
| `strategy-signal-portfolio` | 2 | 24 | 1 | 1 | 22 |

## 机器可读清单

- `migration-symbol-inventory.csv`：每行一个源符号，包含源哈希、流程、父子关系、状态、Rust 目标、Rust 文件/行候选、验收候选证据和阻塞项。
- `migration-symbol-evidence.csv`：每行一个迁移台账切片，保留 Python 表面、Rust 目标、Rust 文件/行候选、测试/覆盖率证据及所有候选符号 ID。

重新生成：

```powershell
python scripts/build-symbol-inventory.py
python scripts/build-symbol-inventory.py --check
```

`--check` 会重新读取 Git 跟踪的上游生产文件并逐字比较三个生成物，可用于防止源文件或台账变化后清单失效。
