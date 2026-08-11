# Bullet

## lab-0334 实盘目标仓位运行器

`bullet-live` 是可部署到 Linux x86_64 的单一 Rust 二进制。它把 E-Works
`lab-0334/run_conceptual.py` 的默认并行候选路径接到 Parquet 与 CTPD：启动时
完整回放四个连续合约 Parquet，随后以 CTPD `IDX-CFFEX-*` 已完成分钟 K 线和 SSE
Tick 延续状态，并暴露 1Exchange Remote Account Source 所需的账户、持仓与成交历史端点。

默认可信语义是：主策略只使用 `IC8888`、`IH8888`；`IF8888`、`IM8888` 参与
四品种候选池和对应 overlay。候选标签严格在
`label_available_time < prediction_asof_time` 时成熟，同一时点按
`entry_time, trade_type, symbol, trade_id` 顺序仲裁。它不会下实盘订单，只发布
模拟目标持仓。

运行器的安全边界是 fail-closed：CTPD 任一连接断开、超时或出现乱序行情时，公开
目标立即清空；所有品种重新以 `/v1/klines` 校准后才重新发布。Kline 只接受
`closed=true`，与 Parquet 重叠的 Bar 不会重复回放。Parquet 超过 72 小时未更新时
拒绝启动实时推理，而非用断档数据推导仓位。

### 配置

从 [lab0334-live.example.toml](configs/lab0334-live.example.toml) 创建
`/etc/bullet/lab0334.toml`。配置包含完整的 `IC/IF/IH/IM` universe、CTPD 连续指数
与当前目标合约之间的显式映射。`target_instrument_id` 到期前由部署负责人换月；运行器
绝不会把 `8888` 自动猜成可交易合约。

`history_seed_bars` 必须覆盖每个文件的完整可用历史，默认 1,000,000。缩短它会改变
默认仲裁器的成熟标签历史，不是一个可接受的性能开关。

令牌放在权限为 `0600` 的一行文件中：`/etc/bullet/ctpd.token` 与
`/etc/bullet/remote-account.token`。示例明确设置
`allow_unauthenticated = false`；不得在生产环境把目标持仓公开匿名访问。

### 构建与性能门禁

```bash
cargo build --release -p bullet-live
./target/release/bullet-live benchmark 20000
/usr/bin/time -v ./target/release/bullet-live seed-benchmark /etc/bullet/lab0334.toml
```

基准测量预热后一个 CTPD Tick 进入 `Portfolio`，直到同步更新已发布 target state 的完整
进程内热路径，输出 `p50_ns`、`p99_ns` 与 `max_ns`。`p99_ns` 必须低于
100,000,000 才会返回成功。Parquet 启动回放、网络传输和 1Exchange HTTP 轮询不属于
这个 100ms 内核延迟定义，分别在部署验收中检查。

`seed-benchmark` 不读取令牌或连接 CTPD，只回放生产 Parquet；配合
`/usr/bin/time -v` 记录完整启动耗时和峰值 RSS，作为 t3a.small 的资源门禁。

每次合并到 `main` 的 release workflow 还构建完全静态的
`bullet-live-x86_64-unknown-linux-musl` 与 SHA-256 文件，适合 Amazon Linux 2023。

### lab0334 一致性门禁

发布任何推理改动前，必须以同一份 Parquet、同一提交的 E-Works lab0334 生成参考账本，
并运行下列门禁。`candidate_decisions.csv` 是 lab 正常运行输出；第二个 CSV 必须由仓库
附带的导出工具生成，包含仲裁前的全部候选标签，不能以最终成交表替代。

```bash
python3 scripts/export-lab0334-parity-reference.py \
  --lab /path/to/E-Works/labs/lab-0334/run_optimized.py \
  --data /path/to/parquet \
  --output /tmp/lab0334-raw-candidate-labels.csv
cargo run --release -p bullet-live -- verify-parity \
  /etc/bullet/lab0334.toml \
  /path/to/lab-output/candidate_decisions.csv \
  /tmp/lab0334-raw-candidate-labels.csv
```

成功时输出 `parity=pass`。它将 Python 与 Bullet 的候选身份、预测键、成熟标签、所有
仲裁字段、平仓时刻和价格规范化为按候选排序的 JSONL，并逐字节比较。所有有限浮点数在
写入规范账本前固定为 14 位小数（负零归零）；该格式规则在比较前固定，不能在出现差异后
调整。原始决策分支在格式化前已逐字段进入规范账本，因此任何接受/拒绝、仓位、替换或标签
时序差异都会使命令失败。

### Singapore EC2 部署

目标实例仅允许 loopback listener；以 Caddy 或 Nginx 在公开 HTTPS hostname 上反代到
`127.0.0.1:8091`。1Exchange 会拒绝 loopback、私网、重定向和非 HTTPS 的 Remote
Account Source。反代必须保留 Authorization header，并设置长期 SSE read timeout。

```ini
# /etc/systemd/system/bullet-live.service
[Unit]
Description=Bullet lab-0334 target account
After=network-online.target
Wants=network-online.target

[Service]
User=bullet
Group=bullet
ExecStart=/opt/bullet/bullet-live serve /etc/bullet/lab0334.toml
Restart=always
RestartSec=2
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/lib/bullet

[Install]
WantedBy=multi-user.target
```

After installing a checksum-verified release binary and fresh Parquet files:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bullet-live
curl -fsS -H "Authorization: Bearer $(< /etc/bullet/remote-account.token)" \
  https://<bullet-host>/api/accounts
curl -fsS -H "Authorization: Bearer $(< /etc/bullet/remote-account.token)" \
  'https://<bullet-host>/api/positions?account_id=BULLET%2Flab0334-sim'
curl -fsS -H "Authorization: Bearer $(< /etc/bullet/remote-account.token)" \
  'https://<bullet-host>/api/account-history?account_id=BULLET%2Flab0334-sim&limit=100'
```

The endpoint returns finite signed positions. Gross futures exposure is
`notional_value`; `valuation` is the NAV-additive unrealized PnL, so a Fund
does not mistake gross contract exposure for account equity.

### 1Exchange Remote Account Source

After the public HTTPS endpoint has passed its contract tests, register it with an owner access
token held outside the repository:

```bash
curl -fsS -X POST "$ONE_EXCHANGE_URL/api/custom-account-sources" \
  -H "Authorization: Bearer $ONE_EXCHANGE_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"name":"Bullet lab-0334 simulated target","base_url":"https://<bullet-host>","auth_header":"Bearer <remote-account-token>","enabled":true}'
```

Then verify the discovered `BULLET/lab0334-sim` account through the local 1Exchange
`/api/accounts`, `/api/positions`, and `/api/account-history` endpoints. History uses
`TRADE_FILL_V1`, stable `<candidate-id>/open` and `<candidate-id>/close` IDs, and opaque
pagination cursors. It records only candidates actually accepted into the live active book:
rejected candidates and training labels never become fills. `coverage` is an explicit,
incomplete Parquet-plus-CTPD reconstruction window and is fixed for every page of one cursor
walk; it is not a claim of broker execution or a permanently archived account ledger.

## Backtest CLI

Bullet also provides its existing deterministic Rust backtest CLI:

```bash
cargo run --release -p bullet-cli -- run examples/dual_ma.rs --config configs/im8888.toml
```
