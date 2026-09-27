# Pure-Rust strategy runtime

`bullet-strategy` is the high-level strategy layer above
`bullet-evaluation`. It absorbs protocol plumbing without moving
strategy-specific semantics into the neutral accounting core.

## What the runtime owns

A `StrategyRunner` owns:

- the causal history window;
- the completed-observation to next-execution boundary;
- target-intent metadata defaults;
- causal availability defaults;
- deterministic provenance hashes;
- automatic UTC evaluation-calendar derivation;
- compilation into `EvaluationInput`;
- handoff to the existing Bullet evaluator.

A strategy only implements:

```rust
pub trait Strategy {
    type Error: std::error::Error + Send + Sync + 'static;

    fn decide(
        &mut self,
        context: &DecisionContext<'_>,
    ) -> Result<TargetIntent, Self::Error>;
}
```

`DecisionContext::history` ends at the completed observation. The next
execution timestamp is visible, but the execution market price is not exposed
to the callback. This preserves the causal boundary by construction.

## Example

```rust
struct Momentum;

impl Strategy for Momentum {
    type Error = std::convert::Infallible;

    fn decide(
        &mut self,
        context: &DecisionContext<'_>,
    ) -> Result<TargetIntent, Self::Error> {
        Ok(context
            .target(if context.observation.price > 100.0 { 1 } else { 0 })
            .model_vintage("momentum-v1"))
    }
}

let result = StrategyRunner::new(strategy, market, config)?.run()?;
```

The runner creates the low-level `TargetDecision` records. The strategy does
not manually construct `decision_time`, `execution_time`, causal availability,
or provenance hashes.

## Boundary

The runtime intentionally does not implement:

- model training or feature search;
- ETH, SOL, BTC, or any other asset-specific state machine;
- automatic target clamping;
- hidden timing shifts;
- strategy-specific campaign, lease, overlay, or controller semantics.

Those remain explicit in the strategy implementation. The runtime reduces
repeated integration code while keeping the evaluator strategy-neutral.

## Current surface

The initial pure-Rust runtime compiles an in-memory causal strategy path into
the existing `EvaluationInput` contract and evaluates it with the same ledger,
metrics, and audit implementation used by the streaming evaluator. The
low-level evaluator remains available for frozen-path imports and external
producers; the high-level runtime is the preferred surface for new Rust
strategies.

Run the example with:

```bash
cargo run -p bullet-strategy --example momentum
```
