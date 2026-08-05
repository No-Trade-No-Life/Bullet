# Bullet

Bullet is an event-driven Rust foundation for quantitative trading research and execution.

## Architecture

```text
bullet-cli
    |
    +--> bullet-engine  -- deterministic dispatch, tick execution, and order-state reduction
    |
    +--> bullet-core    -- market, order, and event-envelope domain types
```

- **`bullet-core`** defines validated instruments, prices, quantities, orders, market ticks, fills, and sequenced event envelopes.
- **`bullet-engine`** assigns event sequences in memory, reduces submitted and filled order states, and executes pending orders when a later matching market tick arrives.
- **`bullet-cli`** submits an order, publishes an AAPL tick, and prints the resulting fill sequence.

## Execution semantics

`execute_market_tick_at` records the incoming market tick before creating fills. It fills only orders that were already pending and whose instrument matches the tick. Fills are generated in ascending order ID, making the resulting event stream deterministic.

The initial execution path treats each submitted order as marketable and uses the next matching tick price as its fill price. It is an in-memory simulation boundary, not a broker or live-trading implementation.

## Development

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p bullet-cli
```

## Initial scope

This workspace is deliberately limited to in-memory, deterministic event flow and order lifecycle reduction. It has no market-data adapters, strategy runtime, persistence, networking, broker integration, portfolio accounting, backtesting, or live-trading behavior.
