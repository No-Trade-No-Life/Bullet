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

The crate provides:

- `RollingPriceFeatures`: causal open-to-open returns for configured lookbacks;
- `LinearModel`: deterministic weighted-sum inference with a serialized model
  metadata and feature-schema contract;
- `LogisticRegressionModel`: stable class-1 probability with class order `[0,1]`;
- `HuberRegressionModel`: linear prediction plus persisted fitted scale;
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
convert Python model objects. Training and export belong to `bullet-ml-training` or external research
code; the deployed/replayed inference contract is pure Rust and must provide a
stable feature schema and model vintage.

## Scheduled models

`ScheduledMlStrategy` routes by the completed observation's decision timestamp,
not the next execution timestamp. `ModelWindow` has an explicit fit cutoff and
half-open availability interval. One feature pipeline and mapper survive every
model switch. Gaps, premature use, expired models, overlapping windows, and
conflicting mapper-vintage metadata are rejected rather than silently reusing a
model. `PreprocessedModel` applies each vintage's serialized training-fitted
transform. See [`native-ml-rolling.md`](native-ml-rolling.md).

Rust is the strategy and model-interface language. Implementations can call
native libraries through Rust bindings; no Python interpreter is required.

The two fitted linear artifacts validate schema, parameters, vintage and model
kind before inference. Their JSON roundtrips need no native solver. Training
contracts and examples are in [`linear-estimators.md`](linear-estimators.md).
