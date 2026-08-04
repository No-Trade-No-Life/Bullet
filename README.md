# Bullet

![Bullet event-track mark](assets/bullet-mark.svg)

**Bullet** is an event-driven Rust workspace for quantitative research, backtesting, and execution.

## Core invariant

Market events are accepted in one causal sequence. A strategy can observe event `N` and create an order, but that order can only fill at the open of a **later eligible event**. It must never be filled retroactively on event `N` or at a price that was not available after the decision.

This is intentionally encoded at the execution boundary: `fill_at_next_open` rejects an open whose sequence is not strictly greater than the order's creation sequence.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `bullet-core` | Shared event and market primitives |
| `bullet-events` | Causal event-log validation |
| `bullet-data` | Validated OHLCV bars |
| `bullet-features` | Deterministic feature calculations |
| `bullet-strategy` | Signal generation |
| `bullet-execution` | Order and next-open fill semantics |
| `bullet-portfolio` | Position accounting |
| `bullet-backtest` | Backtest wiring and records |
| `bullet-ml` | Feature-vector boundary for ML workflows |
| `bullet-cli` | Minimal command-line entry point |

## Development

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p bullet-cli
```
