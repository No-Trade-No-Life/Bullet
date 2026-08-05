# Bullet

Bullet is an event-driven Rust foundation for quantitative trading research and execution.

## Architecture

```text
bullet-cli
    |
    +--> bullet-engine  -- deterministic dispatch and order-state reduction
    |
    +--> bullet-core    -- market, order, and event-envelope domain types
```

- **`bullet-core`** defines validated instruments, prices, quantities, orders, market ticks, fills, and sequenced event envelopes.
- **`bullet-engine`** assigns event sequences in memory and applies submitted-order and filled-order transitions deterministically.
- **`bullet-cli`** publishes a submitted order followed by a fill, then prints the resulting event sequence.

## Development

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p bullet-cli
```

## Initial scope

This initial workspace is deliberately limited to in-memory, deterministic event flow and order lifecycle reduction. It has no market-data adapters, strategy runtime, persistence, networking, broker integration, portfolio accounting, backtesting, or live-trading behavior.
