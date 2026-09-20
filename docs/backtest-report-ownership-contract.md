# 回测报告所有权契约

来源：`qlib/backtest/backtest.py::collect_data_loop`、`account.py::Account.reset/reset_report/get_portfolio_metrics/get_trade_indicator`、`report.py::Indicator.record/generate_trade_indicators_dataframe` 与 `PortfolioMetrics.generate_portfolio_metrics_dataframe`。

## 不能用账户句柄替代报告对象句柄

返回的数值表是导出时快照；历史持仓字典和指标对象则是当时的对象引用。同一对象的后续修改可通过旧报告引用观察到，但不修改旧数值表。账户重置后，旧报告并不会改为引用新对象。账户不再被持有时，报告仍使原对象存活。

所以持有型报告不能仅保存 `Arc<Mutex<Account>>`，然后每次调用 `account.indicator()` 或 `historical_positions()`；这会在重置后错误地切换对象。复制整个账户或长期锁住账户也不能实现上述行为。

## 重置顺序和部分失败

实现更新（2026-09-06）：下述六种重置/失败状态机现已在 Rust `Account::reset_with_factory` 中实现，并通过真实上游源码与原生逐字段差分。`AccountReportFactory` 和 `InitialStockPriceProvider` 是显式插件边界；有限持仓仅查询缺价股票的30日最新收盘价，完整校验后才原子写入价格和账户价值。最终源码验证中，`account.rs` 由全工作区报告证明四项精确100%，`position.rs` 由稳定版/nightly专项报告证明413/413行、82/82函数、489/489区域、18/18分支。该结论只关闭重置组件，不代表 `account.py` 整文件或生产外层回测图已最终验收。

`reset` 先写入已提供的频率、基准配置和启用标志，再调用 `reset_report`。有效启用组合指标时，后者依次：构造并赋值组合指标，替换历史持仓字典，按配置填充持仓价格，构造并赋值交易指标。禁用组合指标或持仓声明跳过更新时，仅替换交易指标。

| 情形 | 原组合指标仍绑定账户 | 原持仓历史仍绑定账户 | 原交易指标仍绑定账户 |
|---|---|---|---|
| 正常启用重置 | 否 | 否 | 否 |
| 禁用或跳过更新 | 是 | 是 | 否 |
| 组合指标构造失败 | 是 | 是 | 是 |
| 填充持仓价格失败 | 否 | 否 | 是 |
| 交易指标构造失败 | 否 | 否 | 是 |

所有情形都保留当前持仓与累计信息对象；失败不会回滚已写入的频率或已替换的对象。旧报告继续保留旧对象，独立于表中“是否仍绑定账户”。

另一个已测到的别名契约：`Indicator.record` 将当前交易指标映射的引用放入历史，而非复制；未调用 `reset` 就修改同一映射时，已有历史行随之变化。已导出的 DataFrame 不随之改变。此项必须在原生指标历史表示中单独审计。

当前交易指标映射已改为 `SharedTradeIndicator = Arc<RwLock<IndexMap<String, f64>>>`：record 克隆共享句柄而非字典，reset 更换当前字典而不清空历史引用，重算以 extend 原地更新固定字段并保留额外字段和既有键顺序。

订单指标现也改为 `SharedOrderIndicator<S> = Arc<RwLock<S>>`，record 保存同一存储身份，reset 创建新存储。NumPy/Pandas 测试覆盖原地修改可见、旧对象在重置/释放后存活、相同时间键替换保序、快照不随对象修改、锁中毒与重置恢复。原始更新/目标金额写入 API 现返回 Result，内部读写及账户调用传播错误；基准价格访问若订单锁中毒则在调用提供者前失败。严格 Clippy 与101个相关测试通过，覆盖率15354/95731已结束：账户四项100%，指标文件仍缺2行/10区域（函数/分支100%），本次改动尚未通过精确100%验收。需继续审查重复锁获取的操作级范围和相应错误传播测试。

仍需区分内部历史与跨执行层传输：上游 executor.py:473–475 保存 inner account 的 `get_order_indicator(raw=True)` 原始引用；原生 `NestedInnerExecutor::order_indicator_snapshot` 及适配器目前仍返回独立 NumpyOrderIndicator。内部 record 改为共享对象并不能自动修复这条跨层别名链，后续必须核验外层聚合对内层原指标的修改可见性，不得以数值相等的复制品替代源引用契约。

交易行读取须取得短期锁；`AccountIndicator` 当前与历史交易行访问返回共享句柄引用，实际消费者/测试插件已同步接入。`TradeIndicatorReport::from_shared_history` 仅在数值表导出时复制各行值，再复用原有 Arrow 导出，保留冻结表与实时字典的区别；中毒历史行使整个导出失败，不发布半成品。`Indicator::trade_indicator_report` 因此返回 Result。计算先完成指标求值再取得行写锁；中毒按显式错误传播，record 仅发布身份不读取行。账户可选输出及 SAOE 查询新增独立行锁错误路径。读取方调用计算/导出前须释放涉及的行锁，回调不得重入相同行锁。

NumPy/Pandas 后端测试验证同代多行共享、原地重算、自定义字段、失败保留映射、冻结数值表、重置分代、覆盖时间键保序、空行表行为和对象释放后旧字典存活。锁中毒测试覆盖计算/导出、保留旧代、覆盖历史键恢复导出、账户输出前停止及 SAOE 错误传播。严格 Clippy 和87个相关测试通过（1846d1）。首次测量未运行既有 base_price 测试导致文件级覆盖不足；加入该组后，稳定版65043/nightly52302均已结束，原始审计1c5ff6确认指标文件810/810行、111/111函数、1262/1262区域、62/62分支，以及账户/SAOE改动文件四项均精确100%。完整 Account.reset、订单历史别名及生产外层后端仍不在本次完成证据中。

## 可重复证据与当前缺口

`crates/core/tests/fixtures/backtest_report_ownership_contract.py` 执行未改写的上述源类/方法，覆盖六种重置情形、对象别名、旧表快照及账户释放。位置构造、原始订单指标创建、进度界面和基准获取为明确的替代边界，不对这些边界宣称差分完成。

`source_report_ownership_contract_retains_old_objects_across_reset_and_failures` 验证该源契约输出。这是源行为特征测试，不是 Rust 持有型报告已经实现的证明。

历史字典现已调整为 `HistoricalPositions = Arc<RwLock<IndexMap<...>>>`。`AccountPortfolioReport` 独立持有该字典，不再借用账户；`replace_historical_positions` 替换绑定而不清空旧对象。测试验证同对象修改、替换后的旧对象存活、账户释放、冻结表以及中毒锁错误。读取方按需取得短期读锁，报告不持续锁住账户；调用历史更新前须释放同一字典的读写锁。

指标现已调整为 `SharedAccountIndicator = Arc<RwLock<Box<dyn AccountIndicator>>>`，`BacktestIndicatorReport` 与 `BacktestReports` 不再带账户借用生命周期。账户替换指标返回旧对象的共享句柄；总报告保留原对象及冻结的数值表，释放账户不会使其失效。生产 SAOE 读取、订单指标快照与报告导出使用短期读锁；账户指标更新在写锁内保持 reset → update → calculate → output → record 的原有顺序。指标及输出回调不得重入同一指标锁，外部调用账户更新前须释放该指标的读写锁。锁中毒按插件错误显式传播，不静默恢复；替换新对象不修复旧报告的中毒对象。

新增真实内置指标测试验证跨账户生命周期、替换后新旧对象独立、表快照及跨线程总报告；新增锁中毒测试覆盖快照、更新、报告导出及 SAOE 读取/替换恢复。严格 Clippy 与六组共66个相关测试通过（74f152）。此段当时记录的覆盖率待测和完整重置缺口已由2026-09-06实现关闭；指标内部历史行别名的后续记录也已关闭相应组件，但生产外层后端和整文件验收仍未完成。

## 指标改造的调用边界审计（2026-09-04）

最终验证更新：2026-09-05稳定版/nightly全工作区测试均通过，native-connected-shared-full-workspace.json原始审计确认本轮五个生产文件四项覆盖率均精确100%。全局仍存在Candle/Trainer覆盖缺口，完整迁移仍在进行。以下“全工作区测量正在运行”为此最终结果之前的过程记录。

2026-09-05连接更新：共享完整聚合和基准价格入口已经实现，账户嵌套输入、NestedCollection、同步/可恢复收集器均已切换到原始句柄。上文和下文较早的“收集器仍复制/聚合尚待接入”描述由此更新。借用/共享基准价格共用逐步骤计算，输入锁不跨行情回调，后续输入能观察前一步回调写入；两个后端实际源码差分验证重复与输出自引用。真实嵌套账户组合验证成交额写回和结果寿命，数值快照接口继续独立。阶段间输出锁失败由 finish_shared_order_indicators 边界上的确定性中毒测试验证。五个改动文件的行/函数/区域/分支均已精确100%，全工作区测量正在运行；详细命令、失败和计数见台账。完整 Account.reset 状态机及生产外层执行图仍待完成。

最新接入：NestedInnerExecutor 现在也要求 order_indicator_handle；借用和持有型原子适配器读取真实账户，递归适配器读取自身 lifecycle。持有型账户锁仅覆盖一次同步获取，不随返回句柄逃出。新测试验证原始存储被独立写锁占用时仍可发布相同身份、返回后账户/引擎锁可取得，以及插件失败和账户中毒的错误传播。递归替换只切换新绑定，旧句柄在图释放后仍存活。41个相关测试及局部原始覆盖审计已通过，详细计数见台账。下文较早记录的“嵌套接口尚待接入”已由此更新；NestedCollection、可恢复收集状态及完整账户聚合仍使用独立数值存储，不能据此宣称端到端原始引用传递完成。

下一连接点：共享基准价格聚合不能同时锁住所有输入，也不能把读锁跨越行情回调。应复用现有逐步骤计算，让每个输入的 base_price/base_volume 获取完毕即释放，然后执行该步骤行情查询；不要预先复制所有步骤，否则后续输入可观察的修改时序会改变。保持 zip 截短、空方向短路、缺值回填和最后 base_volume→base_price 发布顺序。随后接通共享完整聚合和账户/收集器类型，测试重复引用、输出自引用及历史对象写回。

原始身份获取现已接到 AccountIndicator、Account 和 AtomicExecutorAccount/AtomicAccountAdapter 的 order_indicator_handle。它与 order_indicator_snapshot 分开，后者继续返回独立数值副本。新插件方法是必需接口，不用“复制快照再包装Arc”作为默认替代。内置指标获取句柄不读存储；账户仅在获取时短期读取指标引擎，因此句柄持有者不会占住账户/引擎锁。存储锁中毒时仍能发布该身份，数值读取失败；指标引擎锁中毒或插件获取失败则无法取得句柄。reset、替换及账户释放不重绑定旧句柄。真实账户/原子适配器测试验证这些生命周期及独立快照；嵌套执行器/递归适配器和收集结果仍需完成接口接入。

共享成交聚合现有原生入口 aggregate_shared_order_trade_info(&[SharedOrderIndicator<S>])。列表保留原句柄，每次出现都执行一次带符号成交额转换；无去重、无整存储复制。每次转换/读取/赋值只持有一个短期锁，读完释放后才锁输出，因而支持输出自引用及重复输入，不依赖多锁排序。该访问模型允许不同操作在访问间交错，不声称为整批跨对象原子事务；插件回调不得重入自己当前被访问的存储。现有独立存储入口保留；两入口共用 transform_trade_notional、aggregate_columns 和既有对齐/数值计算。

新增两后端差分比较原始重复引用、自引用、reset旧历史、冻结数值、父子对象释放及早/晚缺列状态；锁中毒测试验证初始输出失败、后续输入写失败、转换后输入读失败、发布失败，均保留已到达的修改。内部逐列写入边界另以单元故障注入覆盖8次赋值的每个失败位置，不依赖线程竞争概率。该入口仍需接入 AccountIndicator、AtomicExecutorAccount、NestedInnerExecutor、NestedCollection 和可恢复执行器；当前入口通过不代表生产引用运输已完成。

新增 inner_indicator_alias_contract.py 源契约：两后端均要求同一个原始指标在输入列表重复两次时连续转换两次。例如成交量2、价格3先变6再变12，最终父层汇总价格6；不能按地址去重，也不能先复制两个独立的3。child.reset后，列表与旧历史仍引用原对象；冻结数值快照保持3。输出存储也出现在输入列表中时，上游接受该自引用并完成计算，不能把该合法情况改成错误或互锁死锁。

本次同时发现并修正已有独立存储聚合的失败状态：先转换全部内层价格，再按 inner_amount、deal_amount、trade_price、trade_value、trade_cost、trade_dir 逐列聚合并立即赋值。NumPy在写列前固定inner_amount股票全集，Pandas逐列对齐。缺失后续列保留前面的未归一化聚合值；例如缺trade_cost时，输出trade_price仍是成交额6，而不是最终价格3或旧值9。全部列成功后才归一化价格与方向。新测试逐项比较源/原生输出、错误名称与输入变动；源引用/自引用测试尚不等价于原生运输链已完成。

操作级访问补充：完成率和价格优势的输入读取、计算与赋值使用单一订单写锁，避免混合并发修改前后的指标。外层聚合的成交聚合→目标量→完成率共用输出写锁，随后释放，再查询外部行情。复用现有 store_metric、aggregate_stores 和计算函数；独立 update_trade_amount 与外层路径共用目标量快照构建。此访问范围不是回滚事务：首个领域错误之前的内部订单修改仍保留，后续行情失败不清除已完成的前序阶段。源 transfer 按参数顺序取值，原生缺失指标错误顺序与空价格短路保持不变。测试以自定义存储观察独占访问，并以 try_read/try_write 验证正常/失败返回不保留锁，以及行情回调可访问已完成的前序状态。

跨层原始对象传递仍是独立缺口：源 executor.py 在 collect_data 完成后 append(get_order_indicator(raw=True))；原生 NestedInnerExecutor/AtomicExecutorAccount/AccountIndicator 的 order_indicator_snapshot 仍返回 NumpyOrderIndicator 数值副本，NestedCollection 与 ResumableNestedExecutor 也保存 Vec<NumpyOrderIndicator>。只替换最后一层的集合类型不能恢复身份；必须贯通插件、原子适配器、收集器、账户与外层聚合，并测试同一对象重复出现、重置后旧代存活及外层带符号成交额写回。引用运输过程中不能长期持有账户/指标锁；重复句柄和输出自别名不能通过同时获取多个写锁造成死锁。现有独立快照 API 的合法消费者应保留数值快照语义，不能暗中混用。

当前补充：基准价格查询只持有已取得的方向数值快照，不跨行情回调持有订单锁；回调可通过独立句柄访问订单存储。查询完成后用同一写锁依次写入 base_volume、base_price，其他读者不会观察到这两次正常赋值之间的状态。若回调期间锁中毒，发布失败，两项旧值及回调已经写入的其他字段保留；不会回滚已有副作用或自动恢复锁。自定义 IndicatorStore 的赋值回调不得重入同一个订单锁，此同步边界不承诺任意 Python 并发修改的相同调度。新增测试覆盖正常回调和发布前中毒，完整账户重置及跨层原始订单对象身份仍未完成。

以下为指标改造前、全工作区覆盖率冻结期间的调用边界审计；独立指标句柄改动现状见上节，内部历史行改造仍待进行：

- `Account::indicator` 当前返回借用的 `&dyn AccountIndicator`；直接生产消费者为报告聚合和 `saoe_infrastructure`。改为独立共享对象后，这些消费者须在短期访问范围内读取，不能让引用逃出锁的生命周期。
- `Account::update_indicator` 当前严格执行 reset → 原子/嵌套更新 → calculate → 可选输出 → record。新访问方式必须保持此顺序、首个错误和已发生的状态修改；输出失败不能提前记录历史。输出插件回调的重入/锁边界也须明确验证。
- `AccountIndicator::trade_indicator` 与 `recorded_trade_indicator` 返回映射引用；若内部历史行改为共享对象，不能通过复制映射或不安全延长引用寿命来维持旧签名。需要同步调整访问契约、真实消费者与插件测试。
- `Indicator::order_indicator_mut` 是公开的可观察路径：record 后继续修改当前订单指标，应能通过已记录历史观察到。只处理交易指标映射而继续克隆订单指标，同样不能关闭源行为缺口。
- 独立指标对象的存活、内部历史行的别名、完整 Account.reset 的部分失败是三个不同验收项；其中任何一项的通过都不能替代另两项。

以上是下一步实现约束，不是上述缺口已经修复的证据；当前完整迁移与 100% 全工作区覆盖率验收仍未完成。

补充只读源探针（662919）直接加载未修改的 `Indicator`，仅将原始订单指标工厂替换为 dict：同一代对象连续 record 两个时间键后，两行必须引用同一订单/交易指标；reset 更换当前对象，但旧代仍可修改并由历史读取；对旧时间键重新 record 只更换该行绑定，键顺序不变，其他旧代行仍保留原对象。四项断言均通过。此探针没有改动冻结中的 fixture 或生产代码，也不证明真实订单指标工厂或原生 Rust 的对应行为已经通过。

进一步的可重复测试 `indicator_history_identity_contract.py` 使用真实 `NumpyOrderIndicator` 与 `SingleData`，验证上述别名/重置/覆盖键规则，并加入原地重算保留自定义字段、字段顺序及错误不改已有映射。`source_history_identity_covers_recalculation_reset_failure_and_retained_rows` 对输出逐项断言；当时15个指标测试与严格Clippy通过（5b680b），发现原生计算替换字典的问题。该交易字典问题现已随共享交易历史行改造，现状与覆盖率见前文；订单指标历史的别名问题仍未关闭。

独立指标对象生产改动的聚焦测量现已结束（stable23665/nightly86446），原始审计8da3ee确认账户、报告聚合、SAOE接入三个改动文件行/函数/区域/分支均精确100%；不代表内部历史行契约已经实现或全工作区达到100%。
