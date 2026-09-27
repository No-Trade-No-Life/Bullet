# Pure-Rust ML training

`bullet-ml-training` adds a chronological training layer above `bullet-ml`.
The first backend is SmartCore, enabled by the default `smartcore-backend`
feature. The backend is isolated behind the training crate; Bullet's strategy
runtime and accounting crates do not depend on a particular estimator.

## Dataset contract

A `TrainingExample` declares:

```text
feature_end_ns <= decision_time_ns < label_end_ns
```

The dataset also requires:

- one stable feature schema for every row;
- strictly increasing decision times;
- finite features and targets;
- no empty datasets.

The training layer never shuffles rows. `ChronologicalSplit` and
`WalkForwardPlan` make train/validation windows explicit and expanding.

## Training and inference

The SmartCore backend currently exports deterministic linear models into the
existing `bullet-ml::LinearModel` format. Each `TrainingReport` records:

- solver and model vintage;
- feature-schema SHA-256;
- dataset SHA-256;
- train and validation sample counts;
- train MSE;
- validation MSE, MAE, and R².

This means the training result can be passed directly into `MlStrategy` without
a Python serialization boundary:

```text
TrainingDataset
    → SmartCore trainer
    → bullet_ml::LinearModel
    → MlStrategy
    → StrategyRunner
    → Bullet ledger and audit
```

Run the example with:

```bash
cargo run -p bullet-ml-training --example train_linear
```

## Backend scope

The current release starts with ordinary least squares and ridge regression for
small, transparent tabular models. The trainer/backend boundary leaves room for
classification and neural/tensor backends later without changing the causal
dataset contract or the inference model/strategy interfaces.

Model training remains causal and chronological. Random row shuffling and
random cross-validation are intentionally not part of this surface.
