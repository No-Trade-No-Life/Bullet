# Bullet

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
