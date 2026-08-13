# Bullet

Bullet is an event-driven Rust engine for quantitative strategy research, deterministic backtesting, and live inference integration. A strategy consumes an ordered stream of market events and emits order intent; historical replay and a live runner use the same causal ordering and strategy semantics.

The workspace has two execution surfaces:

- `bullet-cli` is the general-purpose, deterministic Parquet bar backtest runner.
- `bullet-live` is the current live-inference adapter. It reconstructs a strategy state from Parquet, consumes real-time market data, and publishes simulated target positions. Its present implementation is explicitly for the E-Works `lab-0334` model; it does not place broker orders.

An example instrument or a live-model adapter does not define Bullet's strategy universe. The engine boundary is the event stream, strategy contract, execution assumptions, and evaluation or target-state output.

## Engine model

```text
historical data ──> ordered event stream ──> strategy ──> order intent ──> execution state
real-time data ────> ordered event stream ──> live model ─> target state
                         │
                         ├── deterministic replay and backtest
                         └── live inference adapter
```

Historical replay produces reproducible research results. A live adapter is responsible for receiving real-time market events and converting the resulting target state into its declared downstream interface. It must not change event ordering or use information unavailable at the decision time.

## Fixed-capital research replay

`bullet_backtest::fixed_capital` is the component-level research surface for
strategies whose output is a fixed-capital return stream rather than an integer
contract account. It provides:

- complete OHLCV, money, and open-interest bars;
- one deterministic callback containing every instrument at the same timestamp;
- independent, fractional, overlapping component exposures;
- next-bar-open entry, scheduled open/close exit, and a
  pre-registered gap-aware stop;
- a proportional cost fraction on every exposure; and
- component trades, censored exposures, twelve-decimal daily returns, and
  sample-volatility Sharpe metrics.

An order may be `ExitPlan::OpenEnded`. When later causal information arrives,
`ExposureUpdate::ScheduleExit` registers its fill and additional exit or flip
cost; `SetAllocation` replaces an allocation that depends on the observed entry
open. Same-bar close scheduling is applied before the close, while an open exit
must be scheduled earlier. Invalid, duplicate, inactive, or non-causal updates
are rejected.

The evaluation calendar and ordered component list are explicit configuration.
An unregistered component is rejected, while a registered zero-trade component
keeps its zero-valued daily column. This makes canonical CSV comparison stable.

The Parquet timestamp interpretation is mandatory. Production-style files must
declare `Asia/Shanghai`; timezone-less research snapshots require the explicit
`TimestampInterpretation::NaiveAsiaShanghaiWallClock` option. Bullet never
guesses this boundary or silently applies a fallback.

The legacy `Strategy`, `Order`, and self-financing contract-accounting API stays
unchanged for existing CLI strategies. The two accounting models are separate
public surfaces because merging them would change return denominators and cost
semantics.

## Backtest quick start

The bundled dual-moving-average strategy and `IM8888` configuration are examples only.

```bash
cargo run --release -p bullet-cli -- run examples/dual_ma.rs --config configs/im8888.toml
```

The CLI compiles a plain Rust strategy source file into a cached release executable, prints its location, and runs it from a versioned TOML configuration. No Python runtime, interpreter, or user-managed Cargo workspace is required.

## Strategy contract

A strategy exports one public factory named `strategy` and implements `bullet::Strategy`.

```rust
use bullet::{BarContext, Order, Strategy};

pub struct BuyOnce;

pub fn strategy() -> BuyOnce {
    BuyOnce
}

impl Strategy for BuyOnce {
    fn on_bar(&mut self, context: BarContext<'_>) -> Order {
        if context.position == 0 {
            Order::Buy(1)
        } else {
            Order::Close
        }
    }
}
```

`on_bar` is called after all earlier pending orders for that instrument have filled at the new bar's open. It may return `Order::None`, `Order::Buy(quantity)`, `Order::Sell(quantity)`, or `Order::Close`.

Positions may be long, flat, or short. Buys increase the position, sells decrease it, and `Order::Close` submits the opposing quantity needed to flatten it. A submitted order is pending until the next bar open; an order submitted on an instrument's final bar remains unfilled and is reported separately.

## Backtest input and configuration

Version 1 reads one Parquet series per instrument. Each file must contain non-null, chronologically increasing `date` (`Timestamp(Nanosecond)`), `open` (`Float64`), and `close` (`Float64`) columns.

Use TOML to declare replay and portfolio assumptions separately from strategy alpha:

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
id = "EXAMPLE"
data = "path/to/example.parquet"
multiplier = 200.0
margin_rate = 0.12
tick_size = 0.2
```

Add an `[[instruments]]` table for each series. Relative data paths resolve from the configuration file; `~` resolves to the current user's home directory.

Version 1 supports bar replay, next-bar-open fills, and per-contract fees. An order that increases absolute exposure is rejected when the resulting portfolio initial-margin requirement exceeds current equity. Existing positions use their latest close, while the filling instrument uses its fill price. Fees are applied to the exposure opened and closed by each fill.

## Current live-inference adapter

`bullet-live` is a concrete adapter for the E-Works `lab-0334` causal target-position model. It is an implementation on top of Bullet, not the definition of Bullet itself.

At startup, it replays the configured Parquet history; it then recovers and consumes closed CTPD bars and ticks for the configured continuous instruments. It publishes a simulated account's target positions, account history, and account state through the Remote Account API. It does not submit orders to a broker.

Create deployment configuration from [`configs/lab0334-live.example.toml`](configs/lab0334-live.example.toml). The mapping from a continuous `8888` series to an exposed target contract is explicit and deployment-owned; the runner never infers an executable contract from a continuous index. Token files are external to the repository and must be readable only by the service account.

```bash
cargo build --release -p bullet-live
./target/release/bullet-live serve /etc/bullet/lab0334.toml
```

### Live safety and validation

The adapter fails closed: it clears published targets when CTPD is disconnected, stale, or out of order, and only republishes them after all configured instruments are recovered from closed Klines. It refuses a live start with stale Parquet input. Linkit notifications, when configured, are asynchronous and do not alter inference or target state.

Before changing live inference, compare the Rust implementation with an E-Works reference generated from the same Parquet history:

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

Success prints `parity=pass`. The verifier compares normalized candidate identity, maturity labels, arbitration fields, close times, and prices; it rejects any causal or target-state mismatch.

For the live hot-path latency gate and Parquet reconstruction profile:

```bash
./target/release/bullet-live benchmark 20000
/usr/bin/time -v ./target/release/bullet-live seed-benchmark /etc/bullet/lab0334.toml
```

The benchmark fails when p99 inference latency reaches 100 ms. It measures the in-process tick-to-target-state path; startup replay and network time are checked separately by the seed benchmark and deployment validation.

## Evaluation output

Every backtest reports data bytes, bars, runtime, peak RSS, fills, unfilled final-bar orders, round trips, costs, final position per instrument, cumulative return, CAGR, daily UTC equity Sharpe, maximum drawdown, and CAGR / absolute maximum drawdown.

The v1 equity model uses configured multipliers, per-contract fees, adverse basis-point slippage, zero risk-free rate, and `sqrt(252)` Sharpe annualization. It does not model intraday margin calls, close-today fees, funding, borrow, corporate actions, tick or level-2 execution, or broker order routing.

## Workspace architecture

```text
bullet                 stable public strategy API and reporting
├── bullet-backtest    configuration-driven deterministic bar replay
│   └── bullet-data    Parquet market-bar reader
├── bullet-engine      deterministic event dispatch and order-state reduction
└── bullet-core        market, order, bar, fill, and event domain types

bullet-cli             compiles and runs a backtest strategy source file
bullet-live            current live-inference and target-state adapter
```

## Development

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
