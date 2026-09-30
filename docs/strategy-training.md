# Rust ML training

`bullet-ml-training` adds a chronological training layer above `bullet-ml`.
SmartCore is enabled by the default `smartcore-backend` feature. The optional
`xgboost-backend` uses the community Rust wrapper and native XGBoost; optional
`linear-backend` adds native L-BFGS-B-backed LogisticRegression and Huber. Rust
interfaces and orchestration are the boundary; native libraries are permitted
without a Python interpreter, SDK, or subprocess. The backend is isolated behind the training crate; Bullet's strategy
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
`WalkForwardPlan` make train/validation windows explicit and expanding. A split
is rejected when **any** training label extends past the first validation
decision timestamp; this is the purging boundary for overlapping forward labels.

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

The default backend provides OLS and ridge regression. The optional native
backend provides binary logistic boosted trees and squared-error boosted-tree
regression. These are distinct algorithms; XGBoost's binary-logistic objective
is not scikit-learn's linear LogisticRegression.

For timestamp-based sliding windows, per-vintage transforms and weights,
model-availability routing, native artifacts, and bounded OOS replay, see
[`native-ml-rolling.md`](native-ml-rolling.md). The existing `ChronologicalSplit`
and expanding `WalkForwardPlan` API remains available for OLS/ridge callers.
Weighted binary linear LogisticRegression and joint coefficient/intercept/scale
Huber training are available through `linear-backend`. Their `RollingTrainer`
implementations export `LogisticRegressionModel` and `HuberRegressionModel`,
respectively; inference and JSON reload require neither the native optimizer
nor Python. See [`linear-estimators.md`](linear-estimators.md).

Model training remains causal and chronological. Random row shuffling and
random cross-validation are intentionally not part of this surface.
