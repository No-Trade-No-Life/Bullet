# Neutral strategy evaluation

`bullet-evaluation` is Bullet's neutral replay surface for an externally
supplied integer target path. It is an adapter boundary, not a strategy
implementation.

## Boundary

An external adapter owns:

- model artifacts, feature construction, labels, and model vintages;
- strategy state machines, overlays, leases, and strategy-specific clocks;
- the mapping from a frozen oracle or model output to `TargetDecision` records;
- the explicit decision and execution event times;
- the evaluation calendar and opaque strategy diagnostics.

Bullet owns:

- ordered market and decision validation;
- declared causal-availability validation;
- deterministic target-to-realized-position reduction;
- normalized-exposure open-to-open return accounting;
- one-way cost and optional slippage accounting;
- decision, execution, position, and daily-return ledgers;
- fixed-point canonical values and audit hashes.

The evaluator does not branch on BTC, ETH, SOL, a model family, a state code,
or a lab identifier. Those values are data supplied by the adapter.

## Normalized-exposure accounting

`bullet-evaluation` v1 uses `Accounting::NormalizedExposureV1`:

```text
interval gross return = target_units × (next_price / current_price - 1)
interval cost          = turnover_units × one_way_cost_bps / 10,000
interval net return    = gross return - interval cost
```

`target_units` are normalized exposure multiples. They are not BTC quantity,
contracts, notional currency, or margin. This is the accounting surface needed
by the BTC frozen target path and is intentionally distinct from Bullet's
contract-accounting and fixed-capital surfaces.

The adapter supplies one ordered `MarketPoint` per valuation timestamp. A v1
market stream has strictly increasing timestamps and one point per timestamp;
`EventTime.sequence` is retained in the schema for causal provenance, but equal-
timestamp market batching is not part of v1 accounting.

`TerminalPolicy::Liquidate` adds the final absolute exposure to terminal
turnover and records a terminal execution at the last market point. The
position interval itself remains the pre-liquidation exposure, which keeps the
last open-to-open return and the liquidation cost separately auditable.

## Canonical input schema

`EvaluationInput` contains:

- `schema_version`;
- `EvaluationConfig` with accounting, instrument, costs, terminal policy,
  annualization, and an explicit UTC-midnight calendar;
- sorted `MarketPoint` records containing `EventTime`, instrument, and price;
- sorted `TargetDecision` records.

Each `TargetDecision` contains:

- `decision_time` and `execution_time`;
- `instrument` and integer `target_units`;
- optional `model_vintage`, `signal_id`, `source_id`, and `state_code`;
- opaque `diagnostic_json`;
- `CausalAvailability.source_end` and `available_at`;
- a lowercase SHA-256 `provenance_hash` supplied by the adapter.

A decision must satisfy:

```text
source_end <= available_at <= decision_time < execution_time
```

and its execution time must match a nonterminal market point. Bullet does not
infer a next bar, repair missing prices, clamp a target, or silently shift a
timezone.

## Output ledgers

`EvaluationResult` contains:

- `decision_ledger`: every accepted target decision plus realized units;
- `execution_ledger`: target changes and terminal liquidation, with costs;
- `position_ledger`: one normalized-exposure interval row per adjacent market
  point pair;
- `daily_returns`: explicit calendar rows, including quiet days;
- `metrics`: compounded return, volatility, Sharpe, annualized return,
  drawdown, turnover, costs, and ending target/realized units;
- `audit`: SHA-256 hashes for config, market input, decision input, ledgers,
  and the complete run manifest.

Money-like return values and costs use signed or unsigned integers at scale
`10^12`. Aggregate floating-point reductions are performed before canonical
rounding, matching the existing Bullet twelve-decimal contract. Canonical
integers—not runtime tolerances—are the comparison surface.

## Accounting separation

This surface is separate from both existing backtest surfaces:

1. `bullet-backtest`'s legacy `Strategy` API uses contract-accounting equity,
   margin, fees, and next-bar-open order semantics.
2. `bullet-backtest::fixed_capital` evaluates independent component exposures
   against fixed-capital return denominators.
3. `bullet-evaluation` evaluates a declared normalized target path with explicit
   open-to-open prices and return-space costs.

These surfaces must not be merged. Their denominators and execution/cost
semantics are different.

## Adapter workflow

The intended replication workflow is:

1. verify frozen seed, fixture, source, and model-artifact hashes outside Bullet;
2. extract the reference target path into sorted `TargetDecision` records;
3. map the reference valuation prices into sorted `MarketPoint` records;
4. run a bounded slice through `bullet_evaluation::evaluate`;
5. compare target path, execution, position, daily, metric, and audit ledgers;
6. only then expand to full replay, prefix replay, future-suffix mutation, and
   checkpoint/restart tests.

The `bullet-evaluate` binary is the bounded in-memory JSON boundary. For the
full-scale path, `bullet-evaluate-stream` consumes one config JSON and two
newline-delimited JSON streams:

```bash
cargo run --release -p bullet-evaluation --bin bullet-evaluate-stream -- \
  config.json market.jsonl decisions.jsonl output-dir \
  --checkpoint-every 100000
```

The streaming runner writes decision, execution, position, and daily ledgers as
JSONL files. It persists a binary-safe JSON checkpoint containing the last
market point, target state, day accumulators, input hashes, and output-prefix
hashes. A restart validates those values before appending:

```bash
cargo run --release -p bullet-evaluation --bin bullet-evaluate-stream -- \
  config.json market.jsonl decisions.jsonl output-dir \
  --resume
```

`--stop-after N` is a test-only bounded stop that leaves a restartable
checkpoint. A resumed run must produce byte-identical ledger files and the same
run audit hash as an uninterrupted run. The streaming runner does not change
strategy semantics; it only changes storage and restart boundaries.
