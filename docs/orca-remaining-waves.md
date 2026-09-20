# Remaining Qlib migration: audited residual and parallel waves

Snapshot date: 2026-09-20. This is a read-only planning audit; it does not accept or implement any source file.

## Executive result

- Authoritative whole-file registry: **32/230 accepted; 198 residual**. The residual count is exactly the requested 198.
- Strict symbol inventory: **3872 total; 131 accepted by whole-file closure; 3741 residual symbols**. Current ledger: **112 slices (111 done, 1 in progress)**.
- Existing Rust links are only candidate support: 28 residual files have candidate-linked symbols (129 candidate symbols); 170 have none. Candidate support never increases the 32-file numerator.
- Residual character: **159 runtime files, 18 frontend/plumbing files, 3 interface-contract files, 18 source-shape-only files**. Source shape still requires import/export/side-effect evidence.
- Active ownership stays reserved for qlib/utils/time.py, qlib/contrib/eva/alpha.py, and qlib/rl/order_execution/utils.py; first-wave ownership does not touch them.

## Audit provenance and cautions

- Upstream: D:/code/github/qlib at Git 79633dd9506ea689e5400dea0197717b5b3d74b7.
- Registry SHA-256: ab0807e41a9687a0ca60b63686fbedc3ddab0d54628221a22df41bab3a02ef8f; ledger SHA-256: 6c8aed8d6da8e3a9c8623a97011cb8f08e9072148ce503717850db145841607f.
- Read-only generator check: python scripts/build-symbol-inventory.py --check returned nonzero because migration-symbol-inventory.csv and migration-symbol-evidence.csv are out of date. No regeneration was performed.
- A separate in-memory current-generator pass validated 230 production paths, all registry hashes, 3,872 symbols, 112 ledger rows, and every ledger/Rust-definition link. The accepted/residual set therefore comes from migration-source-inventory.md; current source/ledger is used only to describe candidate support. Stale CSVs are not acceptance authority.
- Appendix dependencies are complete direct static Qlib imports for this snapshot. Dynamic config strings, plugin discovery, optional imports, data/services, and third-party packages remain per-slice work.

Legend: R runtime semantics; P frontend/plumbing; I abstract/interface contract; S source shape only. Every file also requires its module initialization, public names, import failures, ordering, serialization/errors and side effects even when only top-level definitions are listed.

## Conflict-free implementation waves

| Wave | Exclusive owners | Dependency reason and scope |
|---|---|---|
| 0 active | ACTIVE | Finish time utilities, alpha returns and dataframe append/string work; files stay frozen to active workers. |
| 1 immediate | L1-A through L1-D | Four independent files with no direct Qlib imports; reusable processor, reweighting, sequence and intraday-data contracts. |
| 1S shape | INIT, SHAPE | Hash-pinned import/export/absence/type contracts. Parallelizable, but never presented as runtime-semantic progress. |
| 2 foundations | CFG, UTIL, DATA-STORAGE, DATASET, DATA-ENGINE, CLI | Config/logging and utilities first; storage/providers/expression/dataset next. Storage follows active time because frequency/calendar semantics are inputs; CLI follows factories. |
| 3 trading | BT-DECISION, BT-ACCOUNT, BT-MARKET, BT-EXEC | Decisions and account/portfolio semantics feed quote/exchange/reporting and executor lifecycle. Partial adapters must be reconciled to whole files. |
| 4 RL | RL-DATA, RL-OE, RL-ENV, RL-TRAIN | Data/adapters and trading contracts precede order execution. Environment/trainer proceed on stable traits; RL-OE waits for active dataframe utility closure. |
| 5 model/workflow | MODEL-BASE, MODEL-ENS, MODEL-RISK, WF-CORE, WF-TASK, WF-ONLINE | Dataset/model and recorder/experiment persistence precede ensemble, risk, task and online orchestration. |
| 6 contrib | C-DATA, C-META, C-MODEL, C-TORCH-A, C-TORCH-B, C-ONLINE, C-REPORT, C-STRAT, C-TUNER, C-ROLL, C-WF, C-OPS, C-ANALYTICS | Start after core dependencies. Torch files are disjoint but use one coordinator-owned tensor/checkpoint interface. |

Appendix owner codes assign every residual file exactly once. Counts: ACTIVE=3, BT-ACCOUNT=4, BT-DECISION=2, BT-EXEC=1, BT-MARKET=4, C-ANALYTICS=2, C-DATA=9, C-META=4, C-MODEL=8, C-ONLINE=5, C-OPS=1, C-REPORT=11, C-ROLL=3, C-STRAT=6, C-TORCH-A=12, C-TORCH-B=15, C-TUNER=5, C-WF=1, CFG=3, CLI=2, DATA-ENGINE=9, DATA-STORAGE=2, DATASET=6, INIT=17, L1-A=1, L1-B=1, L1-C=1, L1-D=1, MODEL-BASE=6, MODEL-ENS=2, MODEL-RISK=4, RL-DATA=7, RL-ENV=4, RL-OE=7, RL-TRAIN=4, SHAPE=1, UTIL=10, WF-CORE=6, WF-ONLINE=4, WF-TASK=4.

## Actionable first wave beyond active work

| Owner | Exact source ownership | Concrete contract | Isolated implementation/evidence boundary |
|---|---|---|---|
| L1-A | qlib/data/inst_processor.py only | InstProcessor.__call__ abstract callable shape; __str__ exact class-name/JSON output, sorted keys, default=str, mutation visibility and serialization failure. | New core module and unique inst_processor_whole_file test/fixture/report; coordinator owns exports and ledgers. |
| L1-B | qlib/data/dataset/weight.py only | Reweighter.__init__ unconditional NotImplementedError; reweight exact message and arbitrary-input non-mutation before failure. | New module and reweighter_whole_file evidence; no dataset loader/processor overlap. |
| L1-C | qlib/model/utils.py only | ConcatDataset init/getitem/len tuple order, negative/out-of-range propagation and empty min failure; IndexSampler indexing/length/identity. | New module and source-pinned sequence tests; no production Torch dependency unless inheritance is observably required. |
| L1-D | qlib/rl/data/base.py only | BaseIntradayBacktestData abstract failures/signatures; BaseIntradayProcessedData shape; ProcessedDataProvider.get_data exact failure and arguments. | New interface module and rl_data_base_whole_file import/signature evidence; reuse owned DTO traits, no production Python bridge. |

These are semantic/interface closures, not passive initializer padding. They share no Python source, proposed Rust module, fixture/test target or report. Workers ask the coordinator to integrate Cargo.toml, Cargo.lock, crates/core/src/lib.rs, ledgers, registry and generated inventories.

## Acceptance gate

1. Hash-pin and execute the authoritative Python file; characterize imports, initialization, signatures/defaults, mutation/identity, ordering, dtype/schema, warnings/errors, serialization and boundaries.
2. Map every symbol to native Rust, an explicit source-shape/absence contract, or a blocker. Frontend/config wiring does not prove an underlying implementation.
3. Run focused Rust tests and genuine source differentials. Compilation, AST-only checks for runtime code, mocks bypassing source behavior and partial adapters cannot prove whole-file parity.
4. Require exact 100% lines/functions/regions/branches per project-owned production file with raw audit; no exclusions, counter normalization, assertion weakening, cfg(coverage), or inherited metrics.
5. Verify owned-file formatting, strict focused/static checks, dependency features/licenses/duplicates, failures, input preservation and stable serialization. Shared/broad gates require coordinator approval.
6. Coordinator integrates shared files and updates whole-file registry/generated inventories only after the entire source file, including imports/exports/side effects, passes.

## Dependency-linked sequencing notes

- CFG unlocks providers, workflow, CLI and factories; mutable global state, path resolution, logging and registration cannot be reduced to DTO parsing.
- DATA-STORAGE must reconcile current calendar-storage Rust work with sequence/mutation/cache/path/encoding/insertion/removal/slicing/provider contracts before file_storage.py closes.
- DATA-ENGINE is the provider/expression/cache backbone. Close value/expression primitives and Cython rolling/expanding semantics before datasets and contributed high-frequency code.
- BT-DECISION and BT-ACCOUNT feed BT-EXEC and RL-OE; current decision/account/executor adapters are useful but partial.
- RL-TRAIN has extensive partial support, but Python files still own configuration assembly, callbacks/checkpoint codecs, lifecycle and frontend behavior.
- Contributed models are implementations: defaults, device/seed behavior, tensor shapes, optimization, early stopping, persistence and prediction index reconstruction need runtime differentials.


## Complete 198-file residual audit

Support is current candidate Rust evidence, never acceptance. Deps lists every direct static Qlib module import found. Missing contracts lists every top-level class/function; all nested methods and the module surface are included in the symbol count.

### configuration-bootstrap (4)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/__init__.py | R / CFG | candidate 1/8; core/saoe_adapter_factory.rs, core/saoe_backtest_data.rs | qlib._version, qlib.config, qlib.data.cache, qlib.log | init, _mount_nfs_uri, init_from_yaml_conf, get_project_path, auto_init (8) |
| qlib/config.py | R / CFG | candidate 4/57; core/config.rs, core/constants.rs | qlib, qlib.constant, qlib.data.data, qlib.data.ops, qlib.utils, qlib.utils.time, qlib.workflow, qlib.workflow.utils | MLflowSettings, QSettings, Config, QlibConfig (57) |
| qlib/log.py | R / CFG | none | qlib.config | MetaLogger, QlibLogger, _QLibLoggerManager, TimeInspector, set_log_with_config, LogFilter, set_global_logger_level, set_global_logger_level_cm (28) |
| qlib/typehint.py | S / SHAPE | none | - | InstDictConf (6) |

### shared-utilities (11)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/utils/__init__.py | R / UTIL | none | qlib.config, qlib.data, qlib.log, qlib.utils.file, qlib.utils.mod | get_redis_connection, read_bin, get_period_list, get_period_offset, read_period_data, np_ffill, lower_bound, upper_bound, requests_with_retry, parse_config, drop_nan_by_y_index, hash_args, parse_field, compare_dict_value, remove_repeat_field, remove_fields_space, normalize_cache_fields, normalize_cache_instruments, is_tradable_date, get_date_range, get_date_by_shift, get_next_trading_date, get_pre_trading_date, transform_end_date, get_date_in_file_name, split_pred, time_to_slc_point, can_use_cache, exists_qlib_data, check_qlib_data, lazy_sort_index, flatten_dict, get_item_from_obj, fill_placeholder, auto_filter_kwargs, Wrapper, register_wrapper, load_dataset, code_to_fname, fname_to_code (53) |
| qlib/utils/data.py | R / UTIL | none | qlib.data.data | robust_zscore, zscore, deepcopy_basic_type, update_config, guess_horizon (7) |
| qlib/utils/file.py | R / UTIL | none | qlib.log | get_or_create_path, save_multiple_parts_file, unpack_archive_with_buffer, get_tmp_file_with_buffer, get_io_object (7) |
| qlib/utils/index_data.py | R / UTIL | candidate 1/59; core/numpy_order_indicator.rs, core/single_data.rs | - | concat, sum_by_index, Index, LocIndexer, BinaryOps, index_data_ops_creator, IndexData, SingleData, MultiData (59) |
| qlib/utils/mod.py | R / UTIL | none | qlib.typehint, qlib.utils.pickle_utils | get_module_by_module_path, split_module_path, get_callable_kwargs, init_instance_by_config, class_casting, find_all_classes (9) |
| qlib/utils/objm.py | R / UTIL | none | qlib.config, qlib.utils.pickle_utils | ObjManager, FileManager (17) |
| qlib/utils/paral.py | R / UTIL | none | qlib.config | ParallelExt, datetime_groupby_apply, AsyncCaller, DelayedTask, DelayedTuple, DelayedDict, is_delayed_tuple, _replace_and_get_dt, _recover_dt, complex_parallel, call_in_subproc (35) |
| qlib/utils/pickle_utils.py | R / UTIL | none | - | RestrictedUnpickler, restricted_pickle_load, restricted_pickle_loads, add_safe_class, get_safe_classes (9) |
| qlib/utils/resam.py | R / UTIL | candidate 4/8; core/calendar_resample.rs, core/feature_provider.rs, core/lib.rs, core/time_series_aggregation.rs +4 | qlib.config, qlib.data.data, qlib.data.dataset.utils, qlib.utils, qlib.utils.time | resam_calendar, get_higher_eq_freq_feature, resam_ts_data, get_valid_value, _ts_data_valid (8) |
| qlib/utils/serial.py | R / UTIL | none | qlib.config | Serializable (19) |
| qlib/utils/time.py | R / ACTIVE | candidate 6/25; core/constants.rs, core/epsilon.rs, core/frequency.rs, core/intraday_index.rs +3 | qlib.config, qlib.constant | get_min_cal, is_single_value, Freq, time_to_day_index, get_day_min_idx_range, concat_date_time, cal_sam_minute, epsilon_change (25) |

### data-provider-storage-expression (21)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/data/__init__.py | S / INIT | none | qlib.data.cache, qlib.data.data | module import/export/side-effect or absence surface (2) |
| qlib/data/_libs/expanding.pyx | R / DATA-ENGINE | none | - | Expanding, Mean, Slope, Resi, Rsquare, expanding, expanding_mean, expanding_slope, expanding_rsquare, expanding_resi (37) |
| qlib/data/_libs/rolling.pyx | R / DATA-ENGINE | none | - | Rolling, Mean, Slope, Resi, Rsquare, rolling, rolling_mean, rolling_slope, rolling_rsquare, rolling_resi (41) |
| qlib/data/base.py | R / DATA-ENGINE | none | qlib.data.cache, qlib.data.data, qlib.data.ops, qlib.log | Expression, Feature, PFeature, ExpressionOps (40) |
| qlib/data/cache.py | R / DATA-ENGINE | none | qlib.config, qlib.data.base, qlib.data.data, qlib.data.ops, qlib.log, qlib.utils, qlib.utils.pickle_utils | QlibCacheException, MemCacheUnit, MemCacheLengthUnit, MemCacheSizeofUnit, MemCache, MemCacheExpire, CacheUtils, BaseProviderCache, ExpressionCache, DatasetCache, DiskExpressionCache, DiskDatasetCache, SimpleDatasetCache, DatasetURICache, CalendarCache, MemoryCalendarCache (95) |
| qlib/data/client.py | R / DATA-ENGINE | none | qlib, qlib.log | Client (7) |
| qlib/data/data.py | R / DATA-ENGINE | none | qlib.config, qlib.data, qlib.data.cache, qlib.data.client, qlib.data.filter, qlib.data.inst_processor, qlib.data.ops, qlib.log, qlib.utils, qlib.utils.paral | ProviderBackendMixin, CalendarProvider, InstrumentProvider, FeatureProvider, PITProvider, ExpressionProvider, DatasetProvider, LocalCalendarProvider, LocalInstrumentProvider, LocalFeatureProvider, LocalPITProvider, LocalExpressionProvider, LocalDatasetProvider, ClientCalendarProvider, ClientInstrumentProvider, ClientDatasetProvider, BaseProvider, LocalProvider, ClientProvider, register_all_wrappers (100) |
| qlib/data/dataset/__init__.py | R / DATASET | none | qlib.data.dataset.handler, qlib.data.dataset.utils, qlib.log, qlib.utils, qlib.utils.serial | Dataset, DatasetH, TSDataSampler, TSDatasetH (41) |
| qlib/data/dataset/handler.py | R / DATASET | none | qlib.data.dataset, qlib.data.dataset.loader, qlib.data.dataset.storage, qlib.data.dataset.utils, qlib.log, qlib.typehint, qlib.utils, qlib.utils.serial | DataHandlerABC, DataHandler, DataHandlerLP (43) |
| qlib/data/dataset/loader.py | R / DATASET | none | qlib.data, qlib.data.dataset.handler, qlib.log, qlib.utils, qlib.utils.pickle_utils, qlib.utils.serial | DataLoader, DLWParser, QlibDataLoader, StaticDataLoader, NestedDataLoader, DataLoaderDH (23) |
| qlib/data/dataset/processor.py | R / DATASET | none | qlib.constant, qlib.data, qlib.data.dataset.storage, qlib.data.dataset.utils, qlib.data.inst_processor, qlib.utils.data, qlib.utils.paral, qlib.utils.serial | get_group_columns, Processor, DropnaProcessor, DropnaLabel, DropCol, FilterCol, TanhProcess, ProcessInf, Fillna, MinMaxNorm, ZScoreNorm, RobustZScoreNorm, CSZScoreNorm, CSRankNorm, CSZFillna, HashStockFormat, TimeRangeFlt (61) |
| qlib/data/dataset/storage.py | R / DATASET | none | qlib.data.dataset.handler, qlib.data.dataset.utils, qlib.log | BaseHandlerStorage, NaiveDFStorage, HashingStockStorage (11) |
| qlib/data/dataset/utils.py | R / DATASET | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.utils | get_level_index, fetch_df_by_index, fetch_df_by_col, convert_index_format, init_task_handler (6) |
| qlib/data/dataset/weight.py | I / L1-B | none | - | Reweighter (4) |
| qlib/data/filter.py | R / DATA-ENGINE | none | qlib.data.data | BaseDFilter, SeriesDFilter, NameDFilter, ExpressionDFilter (24) |
| qlib/data/inst_processor.py | I / L1-A | none | - | InstProcessor (4) |
| qlib/data/ops.py | R / DATA-ENGINE | none | qlib.data._libs.expanding, qlib.data._libs.rolling, qlib.data.base, qlib.data.pit, qlib.log, qlib.utils | ElemOperator, ChangeInstrument, NpElemOperator, Abs, Sign, Log, Mask, Not, PairOperator, NpPairOperator, Power, Add, Sub, Mul, Div, Greater, Less, Gt, Ge, Lt, Le, Eq, Ne, And, Or, If, Rolling, Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, IdxMax, Min, IdxMin, Quantile, Med, Mad, Rank, Count, Delta, Slope, Rsquare, Resi, WMA, EMA, PairRolling, Corr, Cov, TResample, OpsWrapper, register_all_ops (164) |
| qlib/data/pit.py | R / DATA-ENGINE | none | qlib.data.data, qlib.data.ops, qlib.log | P, PRef (10) |
| qlib/data/storage/__init__.py | S / INIT | none | qlib.data.storage.storage | module import/export/side-effect or absence surface (2) |
| qlib/data/storage/file_storage.py | R / DATA-STORAGE | candidate 2/50; core/calendar_write.rs, core/exchange_quote.rs, core/file_calendar_backend.rs, core/file_calendar_storage.rs +1 | qlib.config, qlib.data.cache, qlib.data.storage, qlib.log, qlib.utils.resam, qlib.utils.time | FileStorageMixin, FileCalendarStorage, FileInstrumentStorage, FileFeatureStorage (50) |
| qlib/data/storage/storage.py | R / DATA-STORAGE | none | qlib.log | BaseStorage, CalendarStorage, InstrumentStorage, FeatureStorage (47) |

### backtest-execution (10)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/backtest/account.py | R / BT-ACCOUNT | candidate 6/27; core/account.rs, core/account_executor.rs, core/atomic_nested_inner.rs, core/executor_lifecycle.rs +11 | qlib.backtest.decision, qlib.backtest.exchange, qlib.backtest.high_performance_ds, qlib.backtest.position, qlib.backtest.report, qlib.utils | AccumulatedInfo, Account (27) |
| qlib/backtest/decision.py | R / BT-DECISION | candidate 12/55; core/decision_construction.rs, core/order.rs, core/trade_decision.rs, core/trade_decision_details.rs +2 | qlib.backtest.exchange, qlib.backtest.utils, qlib.data.data, qlib.log, qlib.strategy.base, qlib.utils.time | OrderDir, Order, OrderHelper, TradeRange, IdxTradeRange, TradeRangeByTime, BaseTradeDecision, EmptyTradeDecision, TradeDecisionWO, TradeDecisionWithDetails (55) |
| qlib/backtest/exchange.py | R / BT-MARKET | candidate 7/32; core/exchange_quote.rs, core/exchange_sizing.rs, core/exchange_volume.rs, core/saoe_infrastructure.rs | qlib.backtest.account, qlib.backtest.decision, qlib.backtest.high_performance_ds, qlib.backtest.position, qlib.config, qlib.constant, qlib.data.data, qlib.log, qlib.utils.index_data | Exchange (32) |
| qlib/backtest/executor.py | R / BT-EXEC | candidate 8/28; core/account.rs, core/account_executor.rs, core/atomic_nested_inner.rs, core/calendar_provider.rs +15 | qlib.backtest.account, qlib.backtest.decision, qlib.backtest.exchange, qlib.backtest.position, qlib.backtest.utils, qlib.log, qlib.strategy.base, qlib.utils | BaseExecutor, NestedExecutor, _retrieve_orders_from_decision, SimulatorExecutor (28) |
| qlib/backtest/high_performance_ds.py | R / BT-MARKET | candidate 4/82; core/numpy_order_indicator.rs, core/numpy_quote.rs, core/order_indicator.rs, core/quote.rs +2 | qlib.log, qlib.utils.index_data, qlib.utils.resam, qlib.utils.time | BaseQuote, PandasQuote, NumpyQuote, BaseSingleMetric, BaseOrderIndicator, SingleMetric, PandasSingleMetric, PandasOrderIndicator, NumpyOrderIndicator (82) |
| qlib/backtest/position.py | R / BT-ACCOUNT | candidate 1/67; core/account.rs, core/account_executor.rs, core/shared_executor_lifecycle.rs | qlib.backtest.decision, qlib.data.data | BasePosition, Position, InfPosition (67) |
| qlib/backtest/profit_attribution.py | R / BT-ACCOUNT | none | qlib.backtest.position, qlib.config, qlib.data | get_benchmark_weight, get_stock_weight_df, decompose_portofolio_weight, decompose_portofolio, get_daily_bin_group, get_stock_group, brinson_pa (8) |
| qlib/backtest/report.py | R / BT-ACCOUNT | candidate 4/50; core/base_price.rs, core/lib.rs, core/report_indicator.rs | qlib.backtest.decision, qlib.backtest.exchange, qlib.backtest.high_performance_ds, qlib.tests.config, qlib.utils.index_data, qlib.utils.resam | PortfolioMetrics, Indicator (50) |
| qlib/backtest/signal.py | R / BT-MARKET | none | qlib.data.dataset, qlib.data.dataset.utils, qlib.model.base, qlib.utils, qlib.utils.resam | Signal, SignalWCache, ModelSignal, create_signal_from (10) |
| qlib/backtest/utils.py | R / BT-MARKET | none | qlib.backtest.decision, qlib.data.data, qlib.utils.time | TradeCalendarManager, BaseInfrastructure, CommonInfrastructure, LevelInfrastructure, get_start_end_idx (29) |

### strategy-signal-portfolio (1)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/strategy/base.py | R / BT-DECISION | candidate 1/23; core/saoe_child_assembly.rs, core/saoe_infrastructure.rs | qlib.backtest.decision, qlib.backtest.exchange, qlib.backtest.executor, qlib.backtest.position, qlib.backtest.utils, qlib.rl.interpreter, qlib.utils | BaseStrategy, RLStrategy, RLIntStrategy (23) |

### reinforcement-learning (27)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/rl/contrib/backtest.py | R / RL-DATA | none | qlib.backtest, qlib.backtest.decision, qlib.backtest.executor, qlib.backtest.high_performance_ds, qlib.rl.contrib.naive_config_parser, qlib.rl.contrib.utils, qlib.rl.data.integration, qlib.rl.order_execution.simulator_qlib, qlib.typehint | _get_multi_level_executor_config, _convert_indicator_to_dataframe, _generate_report, single_with_simulator, single_with_collect_data_loop, backtest (10) |
| qlib/rl/contrib/naive_config_parser.py | R / RL-DATA | none | - | merge_a_into_b, check_file_exist, parse_backtest_config, _convert_all_list_to_tuple, get_backtest_config_fromfile (7) |
| qlib/rl/contrib/train_onpolicy.py | R / RL-DATA | none | qlib.backtest, qlib.backtest.decision, qlib.constant, qlib.rl.data.native, qlib.rl.interpreter, qlib.rl.order_execution, qlib.rl.reward, qlib.rl.trainer, qlib.rl.trainer.callbacks, qlib.rl.utils.log, qlib.utils | seed_everything, _read_orders, LazyLoadDataset, train_and_test, main (14) |
| qlib/rl/contrib/utils.py | R / RL-DATA | none | - | read_order_file (2) |
| qlib/rl/data/base.py | I / L1-D | none | - | BaseIntradayBacktestData, BaseIntradayProcessedData, ProcessedDataProvider (12) |
| qlib/rl/data/integration.py | R / RL-DATA | none | qlib, qlib.constant, qlib.contrib.ops.high_freq | init_qlib (3) |
| qlib/rl/data/native.py | R / RL-DATA | candidate 3/25; core/alpha_returns.rs, core/exchange_quote.rs, core/rl_checkpoint_text.rs, core/saoe.rs +2 | qlib.backtest, qlib.backtest.decision, qlib.constant, qlib.rl.data.base, qlib.utils.pickle_utils | get_ticks_slice, IntradayBacktestData, DataframeIntradayBacktestData, load_backtest_data, HandlerIntradayProcessedData, load_handler_intraday_processed_data, HandlerProcessedDataProvider (25) |
| qlib/rl/data/pickle_styled.py | R / RL-DATA | none | qlib.backtest.decision, qlib.rl.data.base, qlib.typehint | _infer_processed_data_column_names, _find_pickle, _read_pickle, SimpleIntradayBacktestData, PickleIntradayProcessedData, load_simple_intraday_backtest_data, load_pickle_intraday_processed_data, PickleProcessedDataProvider, load_orders (21) |
| qlib/rl/order_execution/__init__.py | S / INIT | none | qlib.rl.order_execution.interpreter, qlib.rl.order_execution.network, qlib.rl.order_execution.policy, qlib.rl.order_execution.reward, qlib.rl.order_execution.simulator_simple, qlib.rl.order_execution.state, qlib.rl.order_execution.strategy | module import/export/side-effect or absence surface (2) |
| qlib/rl/order_execution/interpreter.py | R / RL-OE | none | qlib.constant, qlib.rl.data.base, qlib.rl.interpreter, qlib.rl.order_execution.state, qlib.typehint, qlib.utils | canonicalize, FullHistoryObs, DummyStateInterpreter, FullHistoryStateInterpreter, CurrentStateObs, CurrentStepStateInterpreter, CategoricalActionInterpreter, TwapRelativeActionInterpreter, _to_int32, _to_float32 (40) |
| qlib/rl/order_execution/network.py | R / RL-OE | none | qlib.rl.order_execution.interpreter, qlib.typehint | Recurrent, Attention (10) |
| qlib/rl/order_execution/policy.py | R / RL-OE | candidate 2/23; core/lib.rs, core/rl_candle_collector.rs, core/rl_candle_dqn_vessel.rs, core/saoe_interpreter.rs | qlib.rl.trainer.trainer | NonLearnablePolicy, AllOne, PPOActor, PPOCritic, PPO, DQN, auto_device, set_weight, chain_dedup (23) |
| qlib/rl/order_execution/reward.py | R / RL-OE | candidate 2/8; core/saoe_reward.rs | qlib.backtest.decision, qlib.rl.order_execution.state, qlib.rl.reward | PAPenaltyReward, PPOReward (8) |
| qlib/rl/order_execution/simulator_qlib.py | R / RL-OE | candidate 2/10; core/live_nested_executor.rs, core/recursive_nested_inner.rs, core/saoe_child_assembly.rs, core/saoe_live_strategy.rs | qlib.backtest, qlib.backtest.decision, qlib.backtest.executor, qlib.rl.data.integration, qlib.rl.order_execution.state, qlib.rl.order_execution.strategy, qlib.rl.simulator | SingleAssetOrderExecution (10) |
| qlib/rl/order_execution/simulator_simple.py | R / RL-OE | none | qlib.backtest.decision, qlib.constant, qlib.rl.data.base, qlib.rl.data.native, qlib.rl.data.pickle_styled, qlib.rl.order_execution.state, qlib.rl.simulator, qlib.rl.utils | SingleAssetOrderExecutionSimple, price_advantage (21) |
| qlib/rl/order_execution/strategy.py | R / RL-OE | candidate 12/29; core/calendar_provider.rs, core/dataframe_append/extended_plan.rs, core/dataframe_append/object_sort.rs, core/dataframe_append/temporal_plan.rs +14 | qlib.backtest, qlib.backtest.decision, qlib.backtest.exchange, qlib.backtest.executor, qlib.backtest.utils, qlib.constant, qlib.rl.data.native, qlib.rl.interpreter, qlib.rl.order_execution.state, qlib.rl.order_execution.utils, qlib.strategy.base, qlib.utils, qlib.utils.index_data, qlib.utils.time | _get_all_timestamps, fill_missing_data, SAOEStateAdapter, SAOEStrategy, ProxySAOEStrategy, SAOEIntStrategy (29) |
| qlib/rl/order_execution/utils.py | R / ACTIVE | candidate 4/4; core/builtin_object_price.rs, core/complex_price_advantage.rs, core/dataframe_append.rs, core/dataframe_append/objects.rs +8 | qlib.backtest.decision, qlib.backtest.executor, qlib.constant | dataframe_append, price_advantage, get_simulator_executor (4) |
| qlib/rl/trainer/__init__.py | S / INIT | none | qlib.rl.trainer.api, qlib.rl.trainer.callbacks, qlib.rl.trainer.trainer, qlib.rl.trainer.vessel | module import/export/side-effect or absence surface (2) |
| qlib/rl/trainer/api.py | R / RL-TRAIN | none | qlib.rl.interpreter, qlib.rl.reward, qlib.rl.simulator, qlib.rl.trainer.trainer, qlib.rl.trainer.vessel, qlib.rl.utils | train, backtest (3) |
| qlib/rl/trainer/callbacks.py | R / RL-TRAIN | candidate 4/33; core/dataframe_append/extended_plan.rs, core/dataframe_append/temporal_plan.rs, core/environment_step.rs, core/numpy_order_indicator.rs +5 | qlib.log, qlib.rl.trainer.trainer, qlib.rl.trainer.vessel, qlib.typehint | Callback, EarlyStopping, MetricsWriter, Checkpoint (33) |
| qlib/rl/trainer/trainer.py | R / RL-TRAIN | candidate 11/25; core/lib.rs, core/position_cash.rs, core/rl_candle_checkpoint.rs, core/rl_candle_collector.rs +9 | qlib.log, qlib.rl.simulator, qlib.rl.trainer.callbacks, qlib.rl.trainer.vessel, qlib.rl.utils, qlib.rl.utils.finite_env, qlib.typehint | Trainer, _wrap_context, _named_collection (25) |
| qlib/rl/trainer/vessel.py | R / RL-TRAIN | candidate 8/31; core/complex_price_advantage.rs, core/dataframe_append/constructor.rs, core/dataframe_append/constructor_tests.rs, core/environment_step.rs +13 | qlib.constant, qlib.log, qlib.rl.interpreter, qlib.rl.reward, qlib.rl.simulator, qlib.rl.trainer.trainer, qlib.rl.utils, qlib.rl.utils.finite_env | SeedIteratorNotAvailable, TrainingVesselBase, TrainingVessel (31) |
| qlib/rl/utils/__init__.py | S / INIT | none | qlib.rl.utils.data_queue, qlib.rl.utils.env_wrapper, qlib.rl.utils.finite_env, qlib.rl.utils.log | module import/export/side-effect or absence surface (2) |
| qlib/rl/utils/data_queue.py | R / RL-ENV | candidate 1/18; core/data_queue.rs | qlib.log | DataQueue (18) |
| qlib/rl/utils/env_wrapper.py | R / RL-ENV | candidate 4/22; core/atomic_nested_inner.rs, core/dataframe_append/categorical_append_tests.rs, core/environment_reset.rs, core/environment_step.rs +17 | qlib.rl.aux_info, qlib.rl.interpreter, qlib.rl.reward, qlib.rl.simulator, qlib.rl.utils.finite_env, qlib.rl.utils.log, qlib.typehint | InfoDict, EnvWrapperStatus, EnvWrapper (22) |
| qlib/rl/utils/finite_env.py | R / RL-ENV | candidate 9/26; core/dataframe_append.rs, core/dataframe_append/block_tests.rs, core/dataframe_append/categorical_append.rs, core/dataframe_append/categorical_append_internal_tests.rs +28 | qlib.rl.utils.log, qlib.typehint | fill_invalid, is_invalid, generate_nan_observation, check_nan_observation, FiniteVectorEnv, FiniteDummyVectorEnv, FiniteSubprocVectorEnv, FiniteShmemVectorEnv, vectorize_env (26) |
| qlib/rl/utils/log.py | R / RL-ENV | candidate 3/65; core/complex_price_advantage.rs, core/environment_step.rs, core/rl_candle_dqn_policy.rs, core/rl_candle_nstep.rs +3 | qlib.log, qlib.rl.utils.env_wrapper | LogLevel, LogCollector, LogWriter, LogBuffer, ConsoleWriter, CsvWriter, PickleWriter, TensorboardWriter, MlflowWriter (65) |

### model-training-inference (16)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/model/__init__.py | S / INIT | none | qlib.model.base | module import/export/side-effect or absence surface (2) |
| qlib/model/base.py | R / MODEL-BASE | none | qlib.data.dataset, qlib.data.dataset.weight, qlib.utils.serial | BaseModel, Model, ModelFT (9) |
| qlib/model/ens/ensemble.py | R / MODEL-ENS | none | qlib.log, qlib.utils | Ensemble, SingleKeyEnsemble, RollingEnsemble, AverageEnsemble (9) |
| qlib/model/ens/group.py | R / MODEL-ENS | none | qlib.model.ens.ensemble | Group, RollingGroup (9) |
| qlib/model/interpret/base.py | R / MODEL-BASE | none | - | FeatureInt, LightGBMFInt (6) |
| qlib/model/meta/__init__.py | S / INIT | none | qlib.model.meta.dataset, qlib.model.meta.task | module import/export/side-effect or absence surface (2) |
| qlib/model/meta/dataset.py | R / MODEL-BASE | none | qlib.model.meta.task, qlib.utils.serial | MetaTaskDataset (5) |
| qlib/model/meta/model.py | R / MODEL-BASE | none | qlib.model.meta.dataset | MetaModel, MetaTaskModel, MetaGuideModel (10) |
| qlib/model/meta/task.py | R / MODEL-BASE | none | qlib.data.dataset, qlib.utils | MetaTask (9) |
| qlib/model/riskmodel/__init__.py | S / INIT | none | qlib.model.riskmodel.base, qlib.model.riskmodel.poet, qlib.model.riskmodel.shrink, qlib.model.riskmodel.structured | module import/export/side-effect or absence surface (2) |
| qlib/model/riskmodel/base.py | R / MODEL-RISK | none | qlib.model.base | RiskModel (9) |
| qlib/model/riskmodel/poet.py | R / MODEL-RISK | none | qlib.model.riskmodel | POETCovEstimator (7) |
| qlib/model/riskmodel/shrink.py | R / MODEL-RISK | none | qlib.model.riskmodel | ShrinkCovEstimator (18) |
| qlib/model/riskmodel/structured.py | R / MODEL-RISK | none | qlib.model.riskmodel | StructuredCovEstimator (7) |
| qlib/model/trainer.py | R / MODEL-BASE | candidate 3/40; core/finite_subprocess.rs, core/rl_checkpoint_callback.rs, core/rl_checkpoint_lossless_callback.rs | qlib.config, qlib.data.dataset, qlib.data.dataset.weight, qlib.log, qlib.model.base, qlib.utils, qlib.utils.paral, qlib.workflow, qlib.workflow.recorder, qlib.workflow.task.manage | _log_task_info, _exe_task, begin_task_train, end_task_train, task_train, Trainer, TrainerR, DelayTrainerR, TrainerRM, DelayTrainerRM (40) |
| qlib/model/utils.py | R / L1-C | none | - | ConcatDataset, IndexSampler (9) |

### experiment-workflow-recording (14)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/workflow/__init__.py | P / WF-CORE | none | qlib.utils, qlib.utils.exceptions, qlib.workflow.exp, qlib.workflow.expm, qlib.workflow.recorder | QlibRecorder, RecorderWrapper (29) |
| qlib/workflow/exp.py | P / WF-CORE | none | qlib.log, qlib.typehint, qlib.workflow.recorder | Experiment, MLflowExperiment (29) |
| qlib/workflow/expm.py | P / WF-CORE | none | qlib.config, qlib.log, qlib.utils.exceptions, qlib.workflow.exp, qlib.workflow.recorder | ExpManager, MLflowExpManager (29) |
| qlib/workflow/online/manager.py | P / WF-ONLINE | none | qlib, qlib.data.data, qlib.log, qlib.model.ens.ensemble, qlib.model.trainer, qlib.utils.serial, qlib.workflow.online.strategy, qlib.workflow.task.collect | OnlineManager (16) |
| qlib/workflow/online/strategy.py | P / WF-ONLINE | none | qlib.log, qlib.model.ens.group, qlib.utils, qlib.workflow.online.utils, qlib.workflow.recorder, qlib.workflow.task.collect, qlib.workflow.task.gen, qlib.workflow.task.utils | OnlineStrategy, RollingStrategy (14) |
| qlib/workflow/online/update.py | P / WF-ONLINE | none | qlib, qlib.data, qlib.data.dataset, qlib.data.dataset.handler, qlib.model, qlib.utils, qlib.workflow.record_temp, qlib.workflow.recorder | RMDLoader, RecordUpdater, DSBasedUpdater, _replace_range, PredUpdater, LabelUpdater (19) |
| qlib/workflow/online/utils.py | P / WF-ONLINE | none | qlib.log, qlib.utils.exceptions, qlib.workflow.online.update, qlib.workflow.recorder, qlib.workflow.task.utils | OnlineTool, OnlineToolR (19) |
| qlib/workflow/record_temp.py | P / WF-CORE | none | qlib.backtest, qlib.contrib.eva.alpha, qlib.contrib.evaluate, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.utils, qlib.utils.data, qlib.utils.exceptions, qlib.utils.time | RecordTemp, SignalRecord, ACRecordTemp, HFSignalRecord, SigAnaRecord, PortAnaRecord, MultiPassPortAnaRecord (48) |
| qlib/workflow/recorder.py | P / WF-CORE | none | qlib.log, qlib.utils.exceptions, qlib.utils.paral, qlib.utils.serial | Recorder, MLflowRecorder (51) |
| qlib/workflow/task/collect.py | P / WF-TASK | none | qlib.log, qlib.utils.exceptions, qlib.utils.serial, qlib.workflow, qlib.workflow.exp, qlib.workflow.recorder | Collector, MergeCollector, RecorderCollector (16) |
| qlib/workflow/task/gen.py | P / WF-TASK | none | qlib.utils, qlib.workflow.task.utils | task_generator, TaskGen, handler_mod, trunc_segments, RollingGen, MultiHorizonGenBase (18) |
| qlib/workflow/task/manage.py | P / WF-TASK | none | qlib, qlib.config, qlib.utils.pickle_utils, qlib.workflow.task.utils | TaskManager, run_task (34) |
| qlib/workflow/task/utils.py | P / WF-TASK | none | qlib.config, qlib.data, qlib.log, qlib.utils, qlib.utils.mod, qlib.workflow | get_mongodb, list_recorders, TimeAdjuster, replace_task_handler_with_cache (18) |
| qlib/workflow/utils.py | P / WF-CORE | none | qlib.log, qlib.workflow, qlib.workflow.recorder | experiment_exit_handler, experiment_exception_hook (4) |

### command-line (2)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/cli/data.py | P / CLI | none | qlib.tests.data | module import/export/side-effect or absence surface (1) |
| qlib/cli/run.py | P / CLI | none | qlib, qlib.config, qlib.log, qlib.model.trainer, qlib.utils, qlib.utils.data | get_path_list, sys_config, render_template, workflow, run (7) |

### contributed-models-workflows (92; part 1 of 3)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/contrib/data/data.py | R / C-DATA | none | qlib.data.data | ArcticFeatureProvider (4) |
| qlib/contrib/data/dataset.py | R / C-DATA | none | qlib.data.dataset, qlib.utils, qlib.utils.data | _to_tensor, _create_ts_slices, _get_date_parse_fn, _maybe_padding, MTSDatasetH (22) |
| qlib/contrib/data/handler.py | R / C-DATA | none | qlib.contrib.data.loader, qlib.data.dataset, qlib.data.dataset.handler, qlib.data.dataset.processor, qlib.utils | check_transform_proc, Alpha360, Alpha360vwap, Alpha158, Alpha158vwap (15) |
| qlib/contrib/data/highfreq_handler.py | R / C-DATA | none | qlib.contrib.data.handler, qlib.data.dataset.handler | HighFreqHandler, HighFreqGeneralHandler, HighFreqBacktestHandler, HighFreqGeneralBacktestHandler, HighFreqOrderHandler, HighFreqBacktestOrderHandler (25) |
| qlib/contrib/data/highfreq_processor.py | R / C-DATA | none | qlib.data.dataset.processor, qlib.data.dataset.utils | HighFreqTrans, HighFreqNorm (9) |
| qlib/contrib/data/highfreq_provider.py | R / C-DATA | none | qlib, qlib.config, qlib.contrib.ops.high_freq, qlib.data, qlib.data.data, qlib.data.dataset.handler, qlib.utils | HighFreqProvider (14) |
| qlib/contrib/data/loader.py | R / C-DATA | none | qlib.data.dataset.loader | Alpha360DL, Alpha158DL (8) |
| qlib/contrib/data/processor.py | R / C-DATA | none | qlib.data.dataset.processor, qlib.log | ConfigSectionProcessor (7) |
| qlib/contrib/data/utils/sepdf.py | R / C-DATA | none | - | align_index, SepDataFrame, SDFLoc, _isinstance (24) |
| qlib/contrib/eva/alpha.py | R / ACTIVE | none | qlib, qlib.utils.paral | calc_long_short_prec, calc_long_short_return, pred_autocorr, pred_autocorr_all, calc_ic, calc_all_ic (9) |
| qlib/contrib/evaluate.py | R / C-ANALYTICS | none | qlib.backtest, qlib.config, qlib.data, qlib.data.dataset.utils, qlib.log, qlib.strategy.base, qlib.utils, qlib.utils.resam | risk_analysis, indicator_analysis, backtest_daily, long_short_backtest, t_run (8) |
| qlib/contrib/evaluate_portfolio.py | R / C-ANALYTICS | none | qlib.data | _get_position_value_from_df, get_position_value, get_position_list_value, get_daily_return_series_from_positions, get_annual_return_from_positions, get_annaul_return_from_return_series, get_sharpe_ratio_from_return_series, get_max_drawdown_from_series, get_turnover_rate, get_beta, get_alpha, get_volatility_from_series, get_rank_ic, get_normal_ic (15) |
| qlib/contrib/meta/__init__.py | S / INIT | none | qlib.contrib.meta.data_selection | module import/export/side-effect or absence surface (2) |
| qlib/contrib/meta/data_selection/__init__.py | S / INIT | none | qlib.contrib.meta.data_selection.dataset, qlib.contrib.meta.data_selection.model | module import/export/side-effect or absence surface (2) |
| qlib/contrib/meta/data_selection/dataset.py | R / C-META | none | qlib.contrib.torch, qlib.data.dataset, qlib.data.dataset.utils, qlib.log, qlib.model.meta.dataset, qlib.model.meta.task, qlib.model.trainer, qlib.utils, qlib.utils.data, qlib.workflow, qlib.workflow.task.gen, qlib.workflow.task.utils | InternalData, MetaTaskDS, MetaDatasetDS (15) |
| qlib/contrib/meta/data_selection/model.py | R / C-META | none | qlib.contrib.meta.data_selection.dataset, qlib.contrib.meta.data_selection.net, qlib.contrib.meta.data_selection.utils, qlib.data.dataset.weight, qlib.log, qlib.model.meta.dataset, qlib.model.meta.model, qlib.model.meta.task, qlib.workflow | TimeReweighter, MetaModelDS (11) |
| qlib/contrib/meta/data_selection/net.py | R / C-META | none | qlib.contrib.meta.data_selection.utils | TimeWeightMeta, PredNet (9) |
| qlib/contrib/meta/data_selection/utils.py | R / C-META | none | qlib.constant, qlib.log | ICLoss, preds_to_weight_with_clamp, SingleMetaBase (8) |
| qlib/contrib/model/__init__.py | S / INIT | none | qlib.contrib.model.catboost_model, qlib.contrib.model.double_ensemble, qlib.contrib.model.gbdt, qlib.contrib.model.linear, qlib.contrib.model.pytorch_add, qlib.contrib.model.pytorch_alstm, qlib.contrib.model.pytorch_gats, qlib.contrib.model.pytorch_gru, qlib.contrib.model.pytorch_lstm, qlib.contrib.model.pytorch_nn, qlib.contrib.model.pytorch_sfm, qlib.contrib.model.pytorch_tabnet, qlib.contrib.model.pytorch_tcn, qlib.contrib.model.xgboost | module import/export/side-effect or absence surface (9) |
| qlib/contrib/model/catboost_model.py | R / C-MODEL | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.data.dataset.weight, qlib.model.base, qlib.model.interpret.base | CatBoostModel (7) |
| qlib/contrib/model/double_ensemble.py | R / C-MODEL | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.model.interpret.base | DEnsembleModel (13) |
| qlib/contrib/model/gbdt.py | R / C-MODEL | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.data.dataset.weight, qlib.model.base, qlib.model.interpret.base, qlib.workflow | LGBModel (7) |
| qlib/contrib/model/highfreq_gdbt_model.py | R / C-MODEL | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.model.base, qlib.model.interpret.base | HFLGBModel (10) |
| qlib/contrib/model/linear.py | R / C-MODEL | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.data.dataset.weight, qlib.log, qlib.model.base | LinearModel (11) |
| qlib/contrib/model/pytorch_adarnn.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | ADARNN, data_loader, get_stock_loader, get_index, AdaRNN, TransferLoss, cosine, ReverseLayerF, Discriminator, adv, CORAL, MMD_loss, Mine_estimator, Mine, pairwise_dist, pairwise_dist_np, pa, kl_div, js (56) |
| qlib/contrib/model/pytorch_add.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_gru, qlib.contrib.model.pytorch_lstm, qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | ADD, ADDModel, Decoder, RevGradFunc, RevGrad (35) |
| qlib/contrib/model/pytorch_alstm.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | ALSTM, ALSTMModel (15) |
| qlib/contrib/model/pytorch_alstm_ts.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.data.dataset.weight, qlib.log, qlib.model.base, qlib.model.utils, qlib.utils | ALSTM, ALSTMModel (15) |
| qlib/contrib/model/pytorch_gats.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_gru, qlib.contrib.model.pytorch_lstm, qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | GATs, GATModel (16) |
| qlib/contrib/model/pytorch_gats_ts.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_gru, qlib.contrib.model.pytorch_lstm, qlib.contrib.model.pytorch_utils, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | DailyBatchSampler, GATs, GATModel (20) |
| qlib/contrib/model/pytorch_general_nn.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.data.dataset.weight, qlib.log, qlib.model.base, qlib.model.utils, qlib.utils | GeneralPTNN (12) |

### contributed-models-workflows (92; part 2 of 3)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/contrib/model/pytorch_gru.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils, qlib.workflow | GRU, GRUModel (14) |
| qlib/contrib/model/pytorch_gru_ts.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_utils, qlib.data.dataset.handler, qlib.data.dataset.weight, qlib.log, qlib.model.base, qlib.model.utils, qlib.utils | GRU, GRUModel (14) |
| qlib/contrib/model/pytorch_hist.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_gru, qlib.contrib.model.pytorch_lstm, qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | HIST, HISTModel (16) |
| qlib/contrib/model/pytorch_igmtf.py | R / C-TORCH-A | none | qlib.contrib.model.pytorch_gru, qlib.contrib.model.pytorch_lstm, qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | IGMTF, IGMTFModel (18) |
| qlib/contrib/model/pytorch_krnn.py | R / C-TORCH-A | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | CNNEncoderBase, KRNNEncoderBase, CNNKRNNEncoder, KRNNModel, KRNN (24) |
| qlib/contrib/model/pytorch_localformer.py | R / C-TORCH-B | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | LocalformerModel, PositionalEncoding, _get_clones, LocalformerEncoder, Transformer (22) |
| qlib/contrib/model/pytorch_localformer_ts.py | R / C-TORCH-B | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | LocalformerModel, PositionalEncoding, _get_clones, LocalformerEncoder, Transformer (22) |
| qlib/contrib/model/pytorch_lstm.py | R / C-TORCH-B | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | LSTM, LSTMModel (14) |
| qlib/contrib/model/pytorch_lstm_ts.py | R / C-TORCH-B | none | qlib.data.dataset.handler, qlib.data.dataset.weight, qlib.log, qlib.model.base, qlib.model.utils, qlib.utils | LSTM, LSTMModel (14) |
| qlib/contrib/model/pytorch_nn.py | R / C-TORCH-B | none | qlib.contrib.meta.data_selection.utils, qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.data.dataset.weight, qlib.log, qlib.model.base, qlib.utils, qlib.workflow | DNNModelPytorch, AverageMeter, Net (20) |
| qlib/contrib/model/pytorch_sandwich.py | R / C-TORCH-B | none | qlib.contrib.model.pytorch_krnn, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | SandwichModel, Sandwich (14) |
| qlib/contrib/model/pytorch_sfm.py | R / C-TORCH-B | none | qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | SFM_Model, SFM, AverageMeter (20) |
| qlib/contrib/model/pytorch_tabnet.py | R / C-TORCH-B | none | qlib.contrib.model.pytorch_utils, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | TabnetModel, FinetuneModel, DecoderStep, TabNet_Decoder, TabNet, GBN, GLU, AttentionTransformer, FeatureTransformer, DecisionStep, make_ix_like, SparsemaxFunction (47) |
| qlib/contrib/model/pytorch_tcn.py | R / C-TORCH-B | none | qlib.contrib.model.pytorch_utils, qlib.contrib.model.tcn, qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | TCN, TCNModel (14) |
| qlib/contrib/model/pytorch_tcn_ts.py | R / C-TORCH-B | none | qlib.contrib.model.pytorch_utils, qlib.contrib.model.tcn, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | TCN, TCNModel (14) |
| qlib/contrib/model/pytorch_tcts.py | R / C-TORCH-B | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | TCTS, MLPModel, GRUModel (15) |
| qlib/contrib/model/pytorch_tra.py | R / C-TORCH-B | none | qlib.constant, qlib.contrib.data.dataset, qlib.log, qlib.model.base | TRAModel, RNN, PositionalEncoding, Transformer, TRA, evaluate, shoot_infs, sinkhorn, loss_fn, minmax_norm, transport_sample, transport_daily, load_state_dict_unsafe, plot (34) |
| qlib/contrib/model/pytorch_transformer.py | R / C-TORCH-B | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | TransformerModel, PositionalEncoding, Transformer (17) |
| qlib/contrib/model/pytorch_transformer_ts.py | R / C-TORCH-B | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.log, qlib.model.base, qlib.utils | TransformerModel, PositionalEncoding, Transformer (17) |
| qlib/contrib/model/pytorch_utils.py | R / C-TORCH-B | none | - | count_parameters (2) |
| qlib/contrib/model/tcn.py | R / C-MODEL | none | - | Chomp1d, TemporalBlock, TemporalConvNet (11) |
| qlib/contrib/model/xgboost.py | R / C-MODEL | none | qlib.data.dataset, qlib.data.dataset.handler, qlib.data.dataset.weight, qlib.model.base, qlib.model.interpret.base | XGBModel (6) |
| qlib/contrib/online/manager.py | R / C-ONLINE | none | qlib.backtest.account, qlib.contrib.online.user, qlib.contrib.online.utils, qlib.utils | UserManager (8) |
| qlib/contrib/online/online_model.py | R / C-ONLINE | none | qlib.contrib.model.base, qlib.data | ScoreFileModel (8) |
| qlib/contrib/online/operator.py | R / C-ONLINE | none | qlib, qlib.contrib.backtest.backtest, qlib.contrib.evaluate, qlib.contrib.online.executor, qlib.contrib.online.manager, qlib.contrib.online.utils, qlib.data, qlib.log, qlib.utils | Operator, run (12) |
| qlib/contrib/online/user.py | R / C-ONLINE | none | qlib.contrib.evaluate, qlib.data, qlib.log | User (6) |
| qlib/contrib/online/utils.py | R / C-ONLINE | none | qlib.backtest.exchange, qlib.config, qlib.data, qlib.log, qlib.utils, qlib.utils.pickle_utils | load_instance, save_instance, create_user_folder, prepare (6) |
| qlib/contrib/ops/high_freq.py | R / C-OPS | none | qlib.data.cache, qlib.data.data, qlib.data.ops, qlib.utils.time | get_calendar_day, get_calendar_minute, DayCumsum, DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut (25) |
| qlib/contrib/report/__init__.py | S / INIT | none | - | module import/export/side-effect or absence surface (2) |
| qlib/contrib/report/analysis_model/__init__.py | S / INIT | none | qlib.contrib.report.analysis_model.analysis_model_performance | module import/export/side-effect or absence surface (2) |
| qlib/contrib/report/analysis_model/analysis_model_performance.py | R / C-REPORT | none | qlib.contrib.report.graph, qlib.contrib.report.utils, qlib.typehint | _group_return, _plot_qq, _pred_ic, _pred_autocorr, _pred_turnover, ic_figure, model_performance_graph (9) |

### contributed-models-workflows (92; part 3 of 3)

| Path | Kind / owner | Support | Deps | Missing top-level contracts (symbols) |
|---|---|---|---|---|
| qlib/contrib/report/analysis_position/__init__.py | S / INIT | none | qlib.contrib.report.analysis_position.cumulative_return, qlib.contrib.report.analysis_position.rank_label, qlib.contrib.report.analysis_position.report, qlib.contrib.report.analysis_position.risk_analysis, qlib.contrib.report.analysis_position.score_ic | module import/export/side-effect or absence surface (2) |
| qlib/contrib/report/analysis_position/cumulative_return.py | R / C-REPORT | none | qlib.contrib.report.analysis_position.parse_position, qlib.contrib.report.graph | _get_cum_return_data_with_position, _get_figure_with_position, cumulative_return_graph (4) |
| qlib/contrib/report/analysis_position/parse_position.py | R / C-REPORT | none | qlib.backtest.profit_attribution | parse_position, _add_label_to_position, _add_bench_to_position, _calculate_label_rank, get_position_data (7) |
| qlib/contrib/report/analysis_position/rank_label.py | R / C-REPORT | none | qlib.contrib.report.analysis_position.parse_position, qlib.contrib.report.graph | _get_figure_with_position, rank_label_graph (3) |
| qlib/contrib/report/analysis_position/report.py | R / C-REPORT | none | qlib.contrib.report.graph | _calculate_maximum, _calculate_mdd, _calculate_report_data, _report_figure, report_graph (6) |
| qlib/contrib/report/analysis_position/risk_analysis.py | R / C-REPORT | none | qlib.contrib.evaluate, qlib.contrib.report.graph | _get_risk_analysis_data_with_report, _get_all_risk_analysis, _get_monthly_risk_analysis_with_report, _get_monthly_analysis_with_feature, _get_risk_analysis_figure, _get_monthly_risk_analysis_figure, risk_analysis_graph (8) |
| qlib/contrib/report/analysis_position/score_ic.py | R / C-REPORT | none | qlib.contrib.report.graph, qlib.contrib.report.utils | _get_score_ic, score_ic_graph (3) |
| qlib/contrib/report/data/ana.py | R / C-REPORT | none | qlib.contrib.eva.alpha, qlib.contrib.report.data.base, qlib.contrib.report.utils, qlib.utils.paral | CombFeaAna, NumFeaAnalyser, ValueCNT, FeaDistAna, FeaInfAna, FeaNanAna, FeaNanAnaRatio, FeaACAna, FeaSkewTurt, FeaMeanStd, RawFeaAna (39) |
| qlib/contrib/report/data/base.py | R / C-REPORT | none | qlib.contrib.report.utils, qlib.log | FeaAnalyser (7) |
| qlib/contrib/report/graph.py | R / C-REPORT | none | - | BaseGraph, ScatterGraph, BarGraph, DistplotGraph, HeatmapGraph, HistogramGraph, SubplotsGraph (30) |
| qlib/contrib/report/utils.py | R / C-REPORT | none | - | sub_fig_generator, guess_plotly_rangebreaks (3) |
| qlib/contrib/rolling/__main__.py | P / C-ROLL | none | qlib, qlib.contrib.rolling.base, qlib.utils.mod | module import/export/side-effect or absence surface (2) |
| qlib/contrib/rolling/base.py | R / C-ROLL | none | qlib, qlib.log, qlib.model.ens.ensemble, qlib.model.trainer, qlib.utils, qlib.utils.data, qlib.workflow, qlib.workflow.record_temp, qlib.workflow.task.collect, qlib.workflow.task.gen, qlib.workflow.task.utils | Rolling (13) |
| qlib/contrib/rolling/ddgda.py | R / C-ROLL | none | qlib.contrib.meta.data_selection.dataset, qlib.contrib.meta.data_selection.model, qlib.contrib.rolling.base, qlib.data.dataset.handler, qlib.model.meta.task, qlib.model.trainer, qlib.typehint, qlib.utils, qlib.utils.pickle_utils, qlib.workflow, qlib.workflow.task.utils | DDGDA (19) |
| qlib/contrib/strategy/__init__.py | S / INIT | none | qlib.contrib.strategy.cost_control, qlib.contrib.strategy.rule_strategy, qlib.contrib.strategy.signal_strategy | module import/export/side-effect or absence surface (2) |
| qlib/contrib/strategy/cost_control.py | R / C-STRAT | none | qlib.contrib.strategy.order_generator, qlib.contrib.strategy.signal_strategy | SoftTopkStrategy (6) |
| qlib/contrib/strategy/optimizer/__init__.py | S / INIT | none | qlib.contrib.strategy.optimizer.base, qlib.contrib.strategy.optimizer.enhanced_indexing, qlib.contrib.strategy.optimizer.optimizer | module import/export/side-effect or absence surface (2) |
| qlib/contrib/strategy/optimizer/enhanced_indexing.py | R / C-STRAT | none | qlib.contrib.strategy.optimizer.base, qlib.log | EnhancedIndexingOptimizer (5) |
| qlib/contrib/strategy/optimizer/optimizer.py | R / C-STRAT | none | qlib.contrib.strategy.optimizer.base | PortfolioOptimizer (22) |
| qlib/contrib/strategy/order_generator.py | R / C-STRAT | none | qlib.backtest.exchange, qlib.backtest.position | OrderGenerator, OrderGenWInteract, OrderGenWOInteract (7) |
| qlib/contrib/strategy/rule_strategy.py | R / C-STRAT | none | qlib.backtest.decision, qlib.backtest.exchange, qlib.backtest.utils, qlib.data.data, qlib.data.dataset.utils, qlib.strategy.base, qlib.utils, qlib.utils.file, qlib.utils.resam | TWAPStrategy, SBBStrategyBase, SBBStrategyEMA, ACStrategy, RandomOrderStrategy, FileOrderStrategy (28) |
| qlib/contrib/strategy/signal_strategy.py | R / C-STRAT | none | qlib.backtest.decision, qlib.backtest.position, qlib.backtest.signal, qlib.contrib.strategy.optimizer, qlib.contrib.strategy.order_generator, qlib.data, qlib.data.dataset, qlib.log, qlib.model.base, qlib.strategy.base, qlib.utils | BaseSignalStrategy, TopkDropoutStrategy, WeightStrategyBase, EnhancedIndexingStrategy (25) |
| qlib/contrib/torch.py | R / C-MODEL | none | - | data_to_tensor (2) |
| qlib/contrib/tuner/config.py | R / C-TUNER | none | - | TunerConfigManager, PipelineExperimentConfig, OptimizationConfig (7) |
| qlib/contrib/tuner/launcher.py | R / C-TUNER | none | qlib.contrib.tuner.config | run (5) |
| qlib/contrib/tuner/pipeline.py | R / C-TUNER | none | qlib.log, qlib.utils | Pipeline (7) |
| qlib/contrib/tuner/space.py | R / C-TUNER | none | - | module import/export/side-effect or absence surface (3) |
| qlib/contrib/tuner/tuner.py | R / C-TUNER | none | qlib.log, qlib.utils.pickle_utils | Tuner, QLibTuner (18) |
| qlib/contrib/workflow/__init__.py | S / INIT | none | qlib.contrib.workflow.record_temp | module import/export/side-effect or absence surface (2) |
| qlib/contrib/workflow/record_temp.py | P / C-WF | none | qlib.contrib.eva.alpha, qlib.data, qlib.log, qlib.workflow.record_temp | MultiSegRecord, SignalMseRecord (11) |

## Reconciliation before any count update

- Re-run the inventory --check after coordinator integration. The two generated CSVs are stale now; updating them is coordinator-owned and was not done.
- Re-audit ownership before dispatch: Python ownership is disjoint in this snapshot, but worker results can create Rust-module overlap.
- Preserve 32/230 (13.91%) as the strict starting point. All 198 appendix rows are unaccepted; no partial support is inferred complete.
