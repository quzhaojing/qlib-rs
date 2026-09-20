# 日历持久存储迁移契约（待实现/验收）

写入前置组件已实现：`crates/core/src/calendar_write.rs`，复用 ndarray，提供原生二进制覆写/追加和 CalendarFileValues 编码接口。96 个真实上游差分场景通过，稳定版/nightly 聚焦测量为54/54行、10/10函数、89/89区域、4/4分支。新增证据还确认 NumPy Unicode 数组值末尾的 NUL 填充会在标量提取时移除，内部 NUL 保留。完整持久 storage 对象及以下索引/切片契约仍待实现和验收，不以写入组件代替全部生命周期。

持久请求状态现已接入 `file_calendar_storage.rs`：独立缓存发现频率与选定频率，仅成功时发布；原生根目录和 future 标志动态读取；data、直接读取、覆写/追加/清空及长度操作共用现有实现。新增完整原类 AST fixture 比较12组状态序列、118步操作及每步文件字节，含16个Value和6个Other错误。稳定版/nightly各272个相关测试通过；该组件行65/65、函数12/12、区域119/119、分支2/2，均精确100%。这尚不代表全部 storage 类完成；索引、插入、删除、切片、存储频率枚举及全局/override配置构造仍需继续。

2026-09-04。继续当前 Checkpoint 依赖链，不能以现有只读 FileCalendarBackend 替代完整 FileCalendarStorage。来源为上游 `qlib/data/storage/file_storage.py` 和 `storage.py`；以下实际探针执行原方法 AST，使用真实临时文件及已安装 NumPy/pandas。没有修改上游或生产文件。

## 已确认的边界

- 持久 storage 对象缓存 `_support_freq` 和 `_freq_file_cache`；当前 FileCalendarBackend 每个 data 调用建立新请求，不能直接当作有状态 storage 的全部语义。
- provider_override 不是 None 时先调用 format_provider_uri（空字典也进入归一化）；只有 None 从当前 C 读取路径配置。region 构造时捕获，缓存开关默认 true；freq/future/kwargs 来自父类。
- `_read_calendar` 对缺失文件创建空文件，不创建父目录；公开 data/index/get/delete/remove 先 check，不能统一成“读取即创建”。
- `_write_calendar` 先以 `wb` 或 `ab` 打开，再调用 `np.savetxt(..., fmt="%s", encoding="utf-8")`。写入必须记录打开、截断、转换失败、部分写入的顺序；不能擅自改成事务性替换。
- data 的共享原始文本键是 `orig_file + str(uri)`；文件存在检查先于缓存命中，重采样后于原始缓存。写操作没有自动失效缓存，len 通过 data 也可能看到旧值。
- index/get/set 用 Python list 规则；insert/delete 用 NumPy 规则，不能用同一套 Rust Vec 插入/切片近似所有方法。
- `_get_storage_freq` 对 uri.parent 下 txt 文件 stem 取第一个下划线之前部分，再去重排序；与 support_freq 的过滤、排序和缓存策略不同。

## 实际文件探针结果

每例独立文件，先调用 data 填充缓存，再执行操作；缓存始终启用。表中“文件后值”是直接 `_read_calendar()` 的结果，data 仍为操作前缓存。

| 初始行 | 操作 | 结果 | 文件后值 |
|---|---|---|---|
| a, bb | insert(1, LONG) | 成功，NumPy 固定宽度 Unicode 截断 | a, LO, bb |
| 空 | insert(0, x) | ValueError：空数组推断 float，字符串转换失败 | 空 |
| a, b | insert(5, x) | IndexError，不按 list.insert 规则夹取 | a, b |
| aa, bb | set slice(0,1) 为字符串 XYZ | 字符串作为字符序列展开 | X, Y, Z, bb |
| aa, bb | set 第 0 项为列表 [x,y] | 不规则数组 ValueError；打开 wb 已截断文件 | 空 |
| a | extend(generator(b,c)) | 0D 数组 ValueError，ab 未清空旧文件 | a |
| a, b | clear() | 文件清空，但 data/len 仍可读旧缓存 | 空 |
| a, b | extend([c]) | 追加 LF 分隔 UTF-8，保留旧缓存 | a, b, c |

探针证据：844358，退出码 0；8 例均读取并比较操作前缓存、操作错误、文件字节、操作后缓存及直接读值。此为行为发现，不是 Rust 差分验收或覆盖率结果。覆盖测量冻结解除后，将这些场景固化为正式差分 fixture，并扩展负索引、切片步长、空值、重复值、非 BMP 字符、文件缺失、错误传播和同对象频率缓存等场景。

## 实现路线与依赖

复用已有 Frequency、原生 DataPathManager、CalendarCache、CalendarTextDecoder 和日历重采样组件；保持文件读写与编码/时间解析可替换。使用标准文件 I/O 和现有第三方数值/编码组件，不另造通用文件系统或缓存。选择能够精确保留 Python/NumPy 字符宽度、索引与错误顺序的边界后实现；不能把 ndarray 的默认行为当成已经证明兼容。未增加依赖，正式设计前仍需按 Skill 评估适用库。

宿主默认编码、全局配置及 provider override 完整接入、完整 instrument/feature 存储和全部原目标仍在范围内。本契约并不将迁移目标缩小到日历文件。

## 插入与频率枚举的后续证据（2026-09-04）

插入实现已添加为 `calendar_insert.rs`：原始读取/缺失文件创建先于索引检查，索引检查先于转换，转换先于写打开；空数组使用既有 RustPython 浮点解析/格式化并做 Unicode16 十进制数字归一化，非空数组按已有最长 Unicode 标量宽度截断。2,484个真实上游差分场景通过，并验证只读文件上转换错误优先于写入权限错误、读取插件错误保持分类。全工作区原始稳定版2065/nightly28274均已结束，native-calendar-insert-full.json报告已审计：插入组件行41/41、函数9/9、区域76/76、分支2/2均为100%，全工作区仍存在其他模块覆盖缺口。冻结已解除，不能据此宣布完整存储类或整体迁移完成。

`_get_storage_freq` 不能复用 `supported_frequencies` 的解析/过滤/缓存规则：它对当前 uri 父目录的匹配文件名取 stem、按首个下划线切分、去重并以 Python 字符串顺序排序，不解析成 Frequency，也不剔除 future。实际宿主 PureWindowsPath 探针 fc7d0c 确认 `.txt` 的 stem 是 `.txt`（不是空串），`_future.txt` 产生空频率，`DAY.TXT` 产生 `DAY`，`day.extra.txt` 产生 `day.extra`。

Windows实现为 `calendar_storage_frequencies.rs`：用原生OsString保留文件名，标准UTF-16解码保留孤立代理字符并合并有效代理对，BTreeMap按Python码点去重排序。扫描先完整收集，打开或中途失败均不返回部分结果；目录和文件都参与匹配。真实差分发现Rust file_stem与Python3.14对`...txt`不同，依据实际PurePath.stem源码改为“移除后只剩点号则保留原名”的小型兼容层，没有引入新的通用glob库。20组40次上游调用、扫描错误测试及严格Clippy通过；稳定版/nightly各269个相关测试通过，两组原始测量均结束并审计，行56/56、函数12/12、区域93/93、分支8/8均为精确100%。该证据限定Windows组件，非完整配置或全工作区验收。

## 实时配置路径接入（2026-09-04，验收中）

上游FileStorageMixin在每次provider/dpm访问时选择实例override或当前全局配置，override仅替代provider，mount始终取当前全局值。新增CalendarPathProvider接口、LiveCalendarPaths共享映射实现及默认保留DataPathManager的泛型后端/存储；所有存储操作和新建后端请求都通过该路径策略。调用间可观察外部map替换和修改，但已有频率缓存不会自动清空。标准Arc/RwLock和既有IndexMap负责共享与同步；不新建全局单例，也不将路径策略当作配置快照。

3组108步真实上游调用比较结果、错误分类和全部目录文件字节通过；另验证共享别名、所有存储操作、新建请求与缓存实例的区别及锁poison传播。首次聚焦测量新策略行40/40、函数11/11、区域57/57，分支0/0无显式记录；共享后端区域215/216未通过。保留失败报告，统一共享发现实现后Clippy与相关9测试通过；完整工作区测量原始31547/30747仍在运行。归一化输入、构造时region捕获、完整全局配置和其他原目标仍需实现，不能将该路径策略当成整个配置初始化完成。

## 构造与运行时配置的新证据（2026-09-04）

覆盖测量冻结期间仅做只读源码检查和独立进程内行为探针，没有修改生产/测试/fixture。实际构造探针d36188执行原始FileCalendarStorage.__init__，确认：

- `freq`不在构造时解析；`bad-frequency`可以成功构造，验证应延迟到使用。
- `kwargs`中的enable_read_cache=False和region="us"只是保存在kwargs；实例enable_read_cache仍强制True，region仍读取C["region"]。
- provider=None直接使用全局配置、不调用format_provider_uri；非None先归一化，再读取region。字典保持同一对象；某一项归一化失败时，更早项目的规范化修改保留。归一化成功但region缺失时，字典修改也保留，构造失败不能回滚输入。
- 未知region字符串在构造阶段被原样接受，并不立即转成受限Region枚举。resam_calendar源代码又表明region=None会在重采样入口重新读取C["region"]，早于空数组返回；非空分钟采样才经get_min_cal拒绝未知region。日/周/月分支不验证region，不能提前拒绝。
- cal_sam_minute在每个时间点调用get_min_cal(C.min_data_shift, region)，而不是从FileCalendarStorage构造参数捕获shift。当前原生backend的显式Region/minute_shift策略并不等于完整的这个运行时配置契约。

因此完整构造接入还必须保留未解析频率、原始/None region的延迟消费与全局shift读取时机。只将初始化时的枚举和shift快照塞入现有后端不能声明源行为等价。该探针属于下一步契约证据，不是新Rust实现或验收结论。

## 提供者构造模板与持久实例的隔离边界（2026-09-04）

实际 `ProviderBackendMixin.backend_obj` 会先 `copy.deepcopy(backend)`，然后以本次freq/future覆盖模板kwargs，再构造存储。实际源代码探针d84a6c确认：连续两次构造的override映射互不相同，原模板中的相对路径不会被构造归一化修改；失败前的部分归一化也只影响该次副本，不修改模板。直接调用FileCalendarStorage构造仍保留调用者映射身份，两种边界不能混为一谈。

后续原生提供者接入必须把“复制模板→覆盖请求参数→归一化override→捕获region→获取data”留在LocalCalendarLoader的后端获取阶段。future的Value错误重试必须重新走完整构造过程，不能复用第一次实例或region快照；后端构造Other错误不应触发回退。现有CalendarStorageFactory是直接构造入口，不等于ProviderBackendMixin的模板隔离与动态类选择已经迁移。

该探针执行真实Mixin与存储构造函数，但注入了类分派边界；不是任意Python动态导入或deepcopy协议的完成证据。代码仍在全工作区覆盖率测量冻结期，本次只更新契约。

补充组合探针9e2c9c还执行实际LocalCalendarProvider.load_calendar和真实文件：future缺失触发的warning回调把全局region从cn改为us，第二次构造必须重新捕获us，14:59按60min采样得到14:30而不是cn的14:00。全局region缺失且请求频率非法时，构造KeyError先于频率校验，并且不触发future回退或警告。后续原生适配器必须以这组真实行为验证重试实例边界。

## 索引与切片差分证据（历史）

`calendar_sequence.rs` 使用现有 `num-bigint` 表达任意大小的整数索引和切片字段，使用现有 `ndarray` 按已验证索引选取行。边界夹取是小型 Python 兼容层：`ndarray` 的机器字长切片不能直接表达省略负向终点与显式 -1 的区别，不能直接替代。读写继续使用已有编码插件、文件写入层和缓存，不增加新的依赖。

真实上游类差分矩阵有11,940例：5种初始文件状态、整数/切片、正负与零步长、超大整数、文本/序列赋值、首次匹配查找/移除。对比返回形状和值、现有 Value/Other 错误分类、立即读取的文件字节及后续 data 缓存。另有脚本化解码插件测试，验证读取错误传播、remove 第二次读取失败/缩短和配置错误发生在解码之前。后者证明插件调用顺序，不宣称完整并发事务隔离。

当前类型边界为 Rust String 和字符串序列，不宣称覆盖任意 Python 对象、孤立代理字符或全部 Python 异常子类/消息。insert 的 NumPy 空数组浮点转换及固定宽度 Unicode 截断、存储频率枚举和完整配置仍需继续实现；不能将该差分矩阵当作整个存储类完成证据。初次测量缺7个区域；补充故障测试后，稳定版/nightly各265个测试通过，新增文件行189/189、函数24/24、区域332/332、分支22/22，精确100%。两组原始进程均已结束，报告已审计，详细原始证据见台账；全工作区门槛仍需另外验证。
