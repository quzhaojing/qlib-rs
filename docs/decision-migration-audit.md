# decision.py 整文件迁移审计

审计日期：2026-09-16。结论：**尚不能整文件验收**。此结论取代此前“只剩最终表面审计”的推断；既有局部测试证明的范围仍然有效。

最新父级接入（同日）：`live_nested_executor.rs` 已将原决策的策略更新、范围传播、跟踪暂停、原子子执行、实时记录和共享账户报告连接为一个原生父级生成器；实际源码探针验证本地替换决策不会改变外层账户收到的初始决策。65个相关测试与严格Clippy通过。该新入口尚未接入递归/配置/SAOE工厂和实际策略，生产范围日历绑定、完整失败矩阵及四项100%覆盖率仍待验收；下表记录的其他整文件缺口未因此关闭。

后续实现（同日）：`decision_construction.rs` 新增真实共享列表/订单句柄和可观察的构造状态，已用本页源探针比较两次日历读取、原列表身份及各阶段失败。`SaoeDecisionOrigin` 保留具体策略借用并直接委托其提供的现有 SAOE 日历。该构造入口尚未替换执行器使用的 owned `OrderDecision` 通路，下面有关生产闭环的缺口仍成立。补充核查发现 `single_order_strategy.rs` 本来已经单独执行两次读取；“只接收预先计算的时间”描述的是通用 `OrderTradeDecision::from_orders`，不应外推为所有调用方都漏读。

后续执行接入（同日）：`SimulatorCollector::collect_shared_data` 现在可以直接消费构造器保存的列表，复用原 collector 的成交、日内累计和日志实现。共享提取器先检查列表元素、再校验模式，并稳定排序独立的句柄列表。新增源差分验证 serial/parallel、重复对象、执行中清空原列表、第二次成交失败以及结果/trade_info 同一列表语义。结果中的订单是原始实时句柄，数值交易结果是对应成交时的值。原子生命周期、账户报告和嵌套结果运输尚未改用此结果类型；旧 borrowed 插件回调仍在订单锁内运行，同一订单的可重入回调需要后续句柄接口。因此仍不能整文件封账。

后续原子生命周期（同日）：`SharedAtomicExecutorLifecycle` 直接调用共享 collector，账户回调收到原决策的可变借用，账户、返回值接收器和最终调用者通过 `SharedAtomicResult` 保留同一个可修改结果列表。回调前不持有结果/订单锁。已测试追踪、范围校验、结算、收集、账户回调、日历推进、提交和接收器失败的顺序；真实 `BaseExecutor.collect_data` 探针验证接收器清空列表后成功或失败时的共享可见性。这里完成的是共享生命周期及插件边界，`SharedAtomicExecutorAccount` 尚无现有生产 `Account` 适配实现，不能当作账户指标或嵌套回测已接通。

后续生产账户接入（同日）：`AtomicAccountAdapter` 已实现共享账户接口，新增 `SharedAtomic` bar-end/indicator 模式沿现有生产 `Account` 更新流程传递实时结果。读取订单推迟到仓位/历史更新及指标 reset 之后；原生 Indicator 按源字典规则提取数值行，再释放结果/订单锁后写入指标存储。两个数值后端复用同一赋值实现，没有复制 Order 作为运输对象。真实源探针与生产集成测试验证 market 修改成交量为2、reset修改为3后，指标读取3且账户成交本身仍只执行一次。默认未适配指标插件明确拒绝共享输入。完整嵌套运输和成交插件同订单重入仍未闭环。

源文件：`D:/code/github/qlib/qlib/backtest/decision.py`，SHA-256 `a6866d15bc8f3ad1c75bfc3856ccde5245d43f0a2de07b1ad565d6e7e20d8251`。下表覆盖符号清单中该文件全部 55 条记录；一行可合并多个明确列出的符号。Rust 路径均相对于 `crates/core/src/`，测试路径相对于 `crates/core/tests/`。映射存在不代表整符号行为全部验收。

## 符号与证据

| Python 符号 | Rust 实现及现有测试 | 审计结论 / 剩余工作 |
|---|---|---|
| `<module>`, `DecisionType` | `lib.rs` 导出、`TradeDecision<T>` 泛型 | 类型层有映射；无 Python 导入入口，运行时导入的 Cal/logger/time 行为由下列适配器分别负责，不能把泛型等同于整个模块初始化验收 |
| `OrderDir`, `OrderDir.SELL`, `OrderDir.BUY` | `order.rs::OrderDir`；`order.rs` 测试 | 0/1 核心枚举有实现；Python IntEnum 数值互操作需按调用边界核验 |
| `Order`, `Order.stock_id`, `Order.amount`, `Order.direction`, `Order.start_time`, `Order.end_time`, `Order.deal_amount`, `Order.factor`, `Order.SELL`, `Order.BUY` | `order.rs::Order` 字段、访问器和常量；`order.rs` 测试 | owned Order 与 Python 可别名对象不同；时间为 NaiveDateTime，不能表示带时区时间；dataclass 自动生成行为不能由命名符号清单自动证明 |
| `Order.__post_init__` | `Order::new`, `try_new`, `reset_results` | 重置 deal_amount/factor 已有实现；动态输入错误分类与 dataclass 构造参数边界仍需审计 |
| `Order.amount_delta`, `Order.deal_amount_delta`, `Order.sign` | 同名 Rust 方法；`order.rs` 测试 | 对 typed Order 的数值核心已有证据 |
| `Order.parse_dir` | `OrderDir::parse_text/parse_number/parse_values`, `ParseOrderDirectionTransform`；`order.rs`, `numpy_order_indicator.rs` | 当前核心以 f64 向量为主；源 ndarray 任意维度/dtype、整数精度、Python strip 字符集须补完整验收 |
| `Order.key_by_day`, `Order.key`, `Order.date` | `key_by_day/key/day_timestamp`；`order.rs` 测试 | 无时区路径已有实现；带时区与 Python 可变字段路径仍未覆盖完整合同 |
| `OrderHelper`, `OrderHelper.__init__`, `OrderHelper.create` | `order_helper.rs`；`order_helper.rs` 测试、`fixtures/order_helper_contract.py` | 保存 Exchange 及开始/结束解析顺序已测试；生产目录没有 `OrderTimestampParser` 实现，接口本身不能提供 Pandas 时间解析 |
| `TradeRange`, `TradeRange.__call__`, `TradeRange.clip_time_range` | `trade_range.rs::TradeRange` | Rust 要求实现的方法替代 Python 默认 NotImplementedError；自定义插件的未实现错误/回退合同仍需补齐 |
| `IdxTradeRange`, `IdxTradeRange.__init__`, `IdxTradeRange.__call__`, `IdxTradeRange.clip_time_range` | `IdxTradeRange`；`trade_range.rs` 测试 | i64 索引与拒绝时间裁剪已有实现；Python 任意精度整数不等同于 i64 |
| `TradeRangeByTime`, `TradeRangeByTime.__init__`, `TradeRangeByTime.__call__`, `TradeRangeByTime.clip_time_range` | `TradeRangeByTime`；`trade_range.rs` 测试 | chrono 常见文本格式与闭区间已实现；有限格式 parser 不等同于完整 pd.Timestamp 接受域，时区语义仍缺证据 |
| `BaseTradeDecision`, `BaseTradeDecision.__init__` | `trade_decision.rs::TradeDecision::from_items`；`decision_construction.rs::SharedOrderDecisionConstruction` | 共享构造入口已保留策略身份、首轮日历调用顺序和失败状态；旧 owned 入口仍只接收时间，嵌套执行及策略 update/repr 尚未统一到共享入口 |
| `BaseTradeDecision.get_decision` | `TradeDecision::items` | typed 容器返回切片；源基类自身会报错，子类可动态提供任意列表；尚非全部泛型行为 |
| `BaseTradeDecision.update` | `decision_update.rs`；`decision_update.rs` 测试及对应 fixture | 局部顺序已验证；生产目录没有 `DecisionUpdateStrategy` 实现，函数从外部接收策略而非证明是创建该决策的策略 |
| `BaseTradeDecision._get_range_limit`, `BaseTradeDecision.get_range_limit` | `TradeDecision::range_limit`；`trade_decision.rs` 测试 | 部分范围/回退/裁剪已实现；源捕获任意 NotImplementedError，Rust 仅 MissingCalendar 特判；日志文本不同；`total_step - 1` 在 i64::MIN 有溢出风险 |
| `BaseTradeDecision.get_data_cal_range_limit` | `decision_data_range.rs`；`decision_data_range.rs` 测试及对应 fixture | 日历调用/裁剪算法已验证；策略 Exchange.freq 来源、时区与原对象生命周期仍需调用链验收 |
| `BaseTradeDecision.empty` | `OrderTradeDecision::is_empty`, `OrderDecision::is_empty` | **仅订单路径**；源混合列表遇到首个非 Order 立即返回 true，泛型 `TradeDecision<T>` 没有相同行为接口 |
| `BaseTradeDecision.mod_inner_decision` | `propagate_trade_range_to`；`trade_decision.rs` 测试 | Arc 共享范围、只填缺失值已有证据；完整决策身份依赖构造/运输闭环 |
| `EmptyTradeDecision`, `EmptyTradeDecision.get_decision`, `EmptyTradeDecision.empty` | `EmptyTradeDecision`；`trade_decision.rs` 测试 | 空列表和 true 已实现；继承的策略构造流程仍缺 |
| `TradeDecisionWO`, `TradeDecisionWO.__init__` | `OrderTradeDecision::from_orders`；`SharedOrderDecisionConstruction::initialize` 及真实源差分 | 共享入口已验证两次日历读取、调用方列表/订单别名和部分失败状态，并接入原子 collector/lifecycle/Account；旧 owned 嵌套通路尚未替换 |
| `TradeDecisionWO.get_decision` | `orders/orders_mut`；共享构造对象的 `orders` 句柄 | 共享入口保留原列表及订单身份；旧容器切片仍非原列表，嵌套调用链及动态子类访问合同尚未完整验收 |
| `TradeDecisionWO.__repr__` | `trade_decision_repr.rs`；`trade_decision_repr.rs` 测试及对应 fixture | 格式器和首错截断已验证；生产目录没有 `TradeDecisionReprContext` 实现，仍需接入真实决策/策略/范围 |
| `TradeDecisionWithDetails`, `TradeDecisionWithDetails.__init__` | `trade_decision_details.rs`；`SharedOrderDecisionConstruction::initialize`；构造差分及 SAOE 泛型容器 | 共享构造差分已验证父构造成功才赋 details、失败保留外部订单修改及旧 details；SAOE/嵌套的旧 owned 通路仍须统一，不能据局部入口封账 |

## 本轮新取得的可执行证据

执行 `python crates/core/tests/fixtures/decision_surface_audit.py`，源哈希固定，直接执行原始 Order dataclass、BaseTradeDecision、TradeDecisionWO、TradeDecisionWithDetails 等 8 个类；没有替换父构造函数。外部日历使用可记录次数、可失败的测试对象，时间使用标记值以隔离调用顺序，并非时间解析验收。

断言全部通过：

1. 正常构造调用日历两次，决策 start_time 为 `start-1`，订单缺失 start_time 填 `start-2`，已有 end_time 不变，原列表/策略身份保留。
2. 第一次日历失败时 strategy 已保存，total_step/order_list/details 均未设置。
3. 第二次日历失败时决策时间和 order_list 已保存，订单未填充，details 未设置。
4. 第二个订单非法时第一个订单已被填充，外部原列表能观察该修改，details 未设置。
5. 混合列表 `[非订单, 正金额订单]`、`[正金额订单, 非订单]`、`[零订单, 非订单, 正金额订单]`、空列表的 empty 依次为 `true/false/true/true`。

这些结果是 Python 行为证据，**不是 Rust 差分通过**。旧 details fixture 使用替代父类，仅验证委托及 details 写入时机；它没有验证真实父构造的双日历读取和失败副作用。

## 后续实现与验收顺序

当前嵌套接入的具体阻塞点（生产源码复核）：`atomic_nested_inner.rs::collect_data` 调用旧 borrowed 生命周期后，通过 `OwnedOrderExecution::from_execution` 复制订单；`owned_atomic_inner.rs` 委托该路径。`nested_executor.rs::SharedOrderExecution = Arc<OwnedOrderExecution>` 只共享这个副本，不能观察原订单后续修改。同步与可恢复嵌套收集器的 previous-result 钩子、扁平结果和递归适配器都依赖此类型。因此后续必须贯通这些真实调用方，并验证上一轮结果中订单的可见修改与列表身份，不能仅把别名改名或新增无人调用的 trait 当作完成。

先迁移真实构造流程：保留创建策略的身份，分别读取两次日历；复用已有共享订单运输能力，验证原列表/订单可见修改及各阶段失败；让 details 构造使用同一入口。随后接入 update/repr/时间 parser 的生产实现，再补泛型 empty、自定义范围回退、时间/数组数据边界。新增通用能力优先复用已有第三方库，不能为通过当前样例缩窄源接受域。

关闭文件前必须重跑相关差分、实际调用链及项目生产代码精确行/函数/区域/分支 100% 覆盖率。此次只添加源行为审计，生产 Rust 未修改，没有新增覆盖率测量；既有局部 100% 不能外推为整文件或整个工作区 100%。整体严格验收仍为 2/230 生产文件、2/344 全量代码/Notebook 文件、14/3872 符号；总目标与唯一 Checkpoint 保持 in_progress。
