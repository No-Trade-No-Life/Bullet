# Bullet

Bullet is an event-driven Rust foundation for quantitative trading research and execution.

## Architecture

```text
bullet-cli
    |
    +--> bullet-data      -- Parquet OHLC bar input
    +--> bullet-backtest  -- dual-moving-average simulation
    +--> bullet-engine    -- deterministic dispatch, tick execution, and order-state reduction
    +--> bullet-core      -- market, order, bar, and event-envelope domain types
```

- **`bullet-core`** defines validated instruments, floating-point research prices, quantities, orders, OHLC bars, market ticks, fills, and sequenced event envelopes.
- **`bullet-data`** reads Parquet OHLC bars into the shared `Bar` domain type.
- **`bullet-backtest`** runs a long-flat dual-moving-average strategy through `bullet-engine`.
- **`bullet-engine`** records ticks before generated fills, so a decision made from a bar close cannot fill until a later bar open.
- **`bullet-cli`** loads a Parquet file and prints a backtest summary.

## Parquet input

The input is one chronologically ordered instrument series with these required, non-null columns:

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `date` | `Timestamp(Nanosecond)` | Strictly increasing bar timestamp. |
| `open` | `Float64` | Positive, finite next-execution price. |
| `close` | `Float64` | Positive, finite closing price used for the moving averages. |

## Dual-moving-average backtest

```bash
cargo run -p bullet-cli -- path/to/bars.parquet
cargo run -p bullet-cli -- path/to/bars.parquet 20 50
```

The defaults are a 20-bar fast moving average and a 50-bar slow moving average. The strategy is long-flat:

1. After each close, it compares the fast and slow simple moving averages.
2. When fast is above slow and no position is open, it submits a one-unit buy order.
3. When fast is at or below slow and a position is open, it submits a one-unit sell order.
4. `bullet-engine` executes that pending order at the **next bar open**.

The report is Bullet's evaluation record. It includes file bytes, bar count, end-to-end runtime, process peak RSS, fills, completed round trips, ending position, realized P&L, and mark-to-market P&L.

Performance metrics use daily **UTC** close equity, zero risk-free rate, `sqrt(252)` annualization for Sharpe, and a fully funded one-unit initial equity equal to the first close. The report also includes cumulative return, CAGR, annualized Sharpe, maximum drawdown, and CAGR / absolute maximum drawdown. `n/a` means the supplied date range has insufficient observations or no drawdown. It intentionally excludes fees, slippage, contract multiplier, futures margin, corporate actions, sizing, borrow, and live-broker behavior.

## Development

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p bullet-cli -- path/to/bars.parquet 20 50
```
