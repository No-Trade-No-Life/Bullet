# Pure-Rust ML strategies

`bullet-ml` is the pure-Rust ML inference layer above `bullet-strategy`. It
reduces repeated glue code without introducing a Python runtime or moving
asset-specific semantics into Bullet.

## Layers

```text
DecisionContext
    ↓
FeaturePipeline
    ↓
FeatureVector + FeatureSchema
    ↓
Model
    ↓
Prediction
    ↓
TargetMapper
    ↓
TargetIntent
    ↓
bullet-strategy / bullet-evaluation
```

The three boundaries are explicit:

- `FeaturePipeline` consumes only the causal `DecisionContext`;
- `Model` receives a validated, named feature vector and returns a prediction;
- `TargetMapper` converts that prediction into a normalized-exposure target.

## Built-in components

The initial crate provides:

- `RollingPriceFeatures`: causal open-to-open returns for configured lookbacks;
- `LinearModel`: deterministic weighted-sum inference with a serialized model
  metadata and feature-schema contract;
- `ScoreToExposure`: threshold mapping from a score to long, flat, or short
  normalized exposure;
- `WarmupPolicy`: explicit hold or flat behavior before enough history exists.

Feature schemas are hashed and checked before replay. A model cannot be paired
with a different feature ordering accidentally.

## Example

```bash
cargo run -p bullet-ml --example linear_momentum
```

The strategy author supplies the model parameters and mapping policy in Rust;
Bullet supplies causal history, warmup routing, model metadata, ML diagnostics,
provenance, accounting, and audit output.

## Non-goals

`bullet-ml` does not train models, search features, infer labels, or silently
convert Python model objects. Training and export remain external research
steps; the deployed/replayed inference contract is pure Rust and must provide a
stable feature schema and model vintage.
