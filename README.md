# Bullet

## lab-0344 实时模拟账户

`bullet-live` 是一个可部署的单一 Rust 二进制。它启动时从 Parquet 读取尾部
K 线以拼接状态，随后只消费 CTPD 的认证 `GET /v1/ticks?instrument_id=...`
SSE 行情；每个已完成分钟线触发一次内存内推理。它提供 1Exchange Remote
Custom Account Source 所需的 `GET /api/accounts` 和
`GET /api/positions?account_id=...`。

lab-0344 在 E-Works 中的状态是 `process_exception_not_freezable`：其固定规则
已经得到负结果，且不具备策略冻结或真实下单资格。因此该二进制**只输出目标
模拟账户**，不连接下单接口；每个期货持仓的 `comment` 也会保留此状态。

最小配置（token 文件内容均为单行密钥，不要提交）：

```toml
account_id = "BULLET/lab0344-sim"
bind_address = "127.0.0.1:8091"
history_tail_bars = 600

[ctpd]
base_url = "http://127.0.0.1:8080"
bearer_token_file = "/etc/bullet/ctpd.token"
stale_after_ms = 10000

[remote_account]
allow_unauthenticated = false
bearer_token_file = "/etc/bullet/remote-account.token"

[[instruments]]
symbol = "IF"
ctpd_instrument_id = "IF2609"
parquet = "/var/lib/bullet/IF8888.parquet"
target_contracts = 1
contract_multiplier = 300.0
session_bar_count = 240
last_executable_signal_time = "14:40:00"
```

`ctpd_instrument_id` 是显式的连续合约到当前可交易合约映射。换月必须更新该值，
不能把 `IF8888` 直接当作 CTP 合约订阅。输入失效、SSE 断开或超出
`stale_after_ms` 时，Bullet 会清空该合约目标持仓（fail-closed，失效即归零）。
清空也会丢弃未完成 K 线、当日行号、已见信号和计划仓位；重连只会用新收到的完整
分钟线重新累计，绝不跨断线拼接。CTPD 的 `action_day + update_time` 是自然日边界，
不能用夜盘归属的 `trading_day` 替代。

`history_tail_bars` 必须覆盖一个完整的配置交易日。示例的 CFFEX 股指期货会话有
240 根一分钟线，所以 `600` 满足该约束。启动时运行器会回放尾部最后一个自然日的
所有完成 K 线，恢复“每日第一个信号”、挂起入场和已持有目标；Parquet `date` 必须是
以中国市场本地钟表示的分钟结束时间。`session_bar_count` 和
`last_executable_signal_time` 是可执行性契约：只有第 60 至第 220 行且不晚于 14:40
结束的首信号才可能拥有第 240 行退出。换交易所、交易时间或 Parquet 标注语义时，必须
先依据该来源的完整会话重新配置并重放验证，不能沿用此示例。

```bash
cargo build --release -p bullet-live
./target/release/bullet-live benchmark 20000
./target/release/bullet-live serve /etc/bullet/lab0344.toml
```

性能口径是“已反序列化的 CTPD Tick 进入进程”到“目标持仓状态发布”的单次推理。
它不包含上游网络传输、SSE 等待、Parquet 启动读取或 1Exchange 的 HTTP 拉取。
`benchmark` 会输出 p50/p99/max，并在 p99 达到 100ms 时以失败退出。

### 部署到 1Exchange

`bind_address = "127.0.0.1:8091"` 仅用于将二进制限制在反向代理上游；它不能直接
注册到 1Exchange。生产部署需要一个只解析到公网地址、无重定向的 HTTPS 域名，例如
由 Caddy 或 Nginx 将 `https://bullet-live.example.com` 反向代理到该 loopback 地址。代理
必须保留 `Authorization`，不得把账户接口暴露为匿名服务；`remote-account.token` 的内容
与 1Exchange Custom Account Source 的 `auth_header` 值对应（`Bearer <token>`）。

先在目标机运行二进制并使用 HTTPS 域名做以下只读契约检查：

```bash
curl -fsS -H "Authorization: Bearer $(< /etc/bullet/remote-account.token)" \
  https://bullet-live.example.com/api/accounts
curl -fsS -H "Authorization: Bearer $(< /etc/bullet/remote-account.token)" \
  'https://bullet-live.example.com/api/positions?account_id=BULLET%2Flab0344-sim'
```

再以拥有者的 1Exchange access token 注册来源（令牌内容不写入 shell 历史或仓库）：

```bash
curl -fsS -X POST "$ONE_EXCHANGE_URL/api/custom-account-sources" \
  -H "Authorization: Bearer $ONE_EXCHANGE_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"name":"Bullet lab-0344 simulated target","base_url":"https://bullet-live.example.com","auth_header":"Bearer <remote-account-token>","enabled":true}'
curl -fsS -H "Authorization: Bearer $ONE_EXCHANGE_TOKEN" \
  "$ONE_EXCHANGE_URL/api/accounts"
```

`<remote-account-token>` 应由安全的部署工具代入，不要将它粘贴到配置、文档或终端历史。
注册后还要从 1Exchange 读取该 AccountID 的 `/api/positions`，检查 CNY 汇率覆盖后才能将
它用于 Fund；本运行器不提供成交历史端点，因此不支持 1Exchange 复盘。

Bullet is a Rust product for deterministic strategy backtests.

Its public boundary is deliberately small:

- **`bullet` crate**: the stable API used by a Rust strategy source file.
- **`bullet` CLI**: compiles that source file together with the crate, creates a runnable strategy executable, and runs it against a versioned TOML configuration.

The internal workspace crates are implementation details.

## Run a strategy

```bash
cargo run --release -p bullet-cli -- run examples/dual_ma.rs --config configs/im8888.toml
```

The CLI creates a cached, release-mode `bullet-strategy` executable, prints its location, and runs it. A strategy file therefore remains a normal Rust source file: no Python, no interpreter, no user-managed Cargo workspace.

## Strategy API

A strategy exposes one public factory named `strategy` and implements `bullet::Strategy`.

```rust
use bullet::{BarContext, Order, Strategy};

pub struct BuyOnce;

pub fn strategy() -> BuyOnce { BuyOnce }

impl Strategy for BuyOnce {
    fn on_bar(&mut self, context: BarContext<'_>) -> Order {
        if context.position == 0 { Order::Buy(1) } else { Order::Close }
    }
}
```

`on_bar` executes after all previously submitted orders for that instrument are filled at the new bar open. It may return `Order::None`, `Order::Buy(quantity)`, `Order::Sell(quantity)`, or `Order::Close`. The order becomes pending and fills at the **next bar open**. Positions may be long, flat, or short: buys increase the position, sells decrease it, and `Order::Close` submits the opposing quantity required to flatten it.

## Backtest configuration

Use one TOML file to separate research assumptions from the strategy alpha:

```toml
version = 1

[backtest]
mode = "bar"
initial_cash = 1_000_000.0
currency = "CNY"

[execution]
fill_price = "next_bar_open"
slippage_bps = 1.0

[fees]
mode = "per_contract"
open = 2.3
close = 2.3

[[instruments]]
id = "IM8888"
data = "~/.quant-data/IM8888.parquet"
multiplier = 200.0
margin_rate = 0.12
tick_size = 0.2
```

Add another `[[instruments]]` table for every data file. Bullet merges bars deterministically by timestamp then instrument ID and invokes the strategy once per bar. Relative data paths are resolved from the configuration file; `~` resolves to the current user home directory.

Version 1 accepts only `mode = "bar"`, `fill_price = "next_bar_open"`, and `fees.mode = "per_contract"`. An order that increases absolute exposure is rejected if the resulting portfolio initial-margin requirement exceeds current account equity. Existing positions use their latest close; the filling instrument uses its fill price. Open and close fees apply by the exposure that each fill opens or closes. This makes unsupported tick, same-bar, or custom-fee behavior impossible rather than silently applying a different model.

## Evaluation output

Every run reports a single Bullet-defined record: data bytes, bars, runtime, peak RSS, fills, unfilled final-bar orders, round trips, costs, final position per instrument, cumulative return, CAGR, daily UTC equity Sharpe, maximum drawdown, and CAGR / absolute maximum drawdown. Orders submitted on an instrument’s last bar remain pending because no next-bar open exists; they are reported, but do not affect fills, positions, or performance.

The version-1 equity model uses configured multipliers, zero risk-free rate, `sqrt(252)` Sharpe annualization, configured per-contract fees and adverse basis-point slippage. It does not yet model intraday margin calls, close-today fees, funding, borrow, corporate actions, or tick/level-2 execution.

## Development

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
