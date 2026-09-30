# Weighted linear LogisticRegression and Huber

The optional `bullet-ml-training/linear-backend` provides two independent
`RollingTrainer` implementations: `LogisticRegressionTrainer` and
`HuberRegressionTrainer`. Training uses `lbfgsb = "=0.1.1"`, the community
L-BFGS-B-C binding. Inference artifacts live in `bullet-ml` and contain only
Rust/serde data; inference does not link the optimizer or execute Python.

This is estimator and rolling-lifecycle integration, **not** BTC/ETH/SOL
retraining, original-model elementwise parity, or a profitability result.
Features, labels, class weights, ensembles, state machines and exposure mapping
remain the strategy/adapter's responsibility.

## Build and bounded example

Requirements: Rust, a C toolchain and libclang for the pinned crate's build-time
bindings. Supported validation targets are macOS ARM64 and Linux x86_64/ARM64
(LP64). No XGBoost installation, OpenMP, Python SDK or Python subprocess is
needed for this feature. Default features still select SmartCore OLS/ridge.

```sh
cargo test -p bullet-ml -p bullet-ml-training \
  --no-default-features --features linear-backend --locked
cargo run -p bullet-ml-training --no-default-features \
  --features linear-backend --example rolling_linear --locked -- \
  logistic 128 target/logistic-oos-demo
cargo run -p bullet-ml-training --no-default-features \
  --features linear-backend --example rolling_linear --locked -- \
  huber 128 target/huber-oos-demo
```

Each output directory must be new. Each run independently fits four rolling
vintages, saves models/transforms/reports, reloads them, and checks the complete
OOS decision input and evaluation ledger exactly. `PERF_PROGRESS` logs expose
per-fold row counts, features, iterations, evaluations and elapsed time. The
summary includes solver workloads; wall-clock times are not part of model or
report hashes. The example bounds input to 64..4096 synthetic daily rows.

The generic integration is:

```text
TrainingDataset + weights + RollingPlan + RollingOptions
  -> train_rolling(..., &mut LogisticRegressionTrainer / HuberRegressionTrainer)
  -> FittedFold<LogisticRegressionModel / HuberRegressionModel>
  -> into_window() -> ScheduledMlStrategy -> StrategyRunner -> evaluator
```

The timestamp, per-label maturity, training-only scaling, weight normalization
and model-availability contracts are unchanged; see
[`native-ml-rolling.md`](native-ml-rolling.md). Models fit from zero independently
at every vintage; this is not online incremental learning or warm starting.

## Mathematical contracts

All features, parameters, targets and positive sample weights are `f64`.
Let `a_i` denote the effective sample weight, `W = sum(a_i)`,
`z_i = x_i dot beta + b`, and `||beta||² = sum(beta_j²)`. Disabling the intercept
fixes `b = 0`; enabling it adds an unpenalized parameter.

### Binary linear LogisticRegression

```text
lambda = 1 / (C * W)
objective = sum(a_i * binary_logloss(y_i, z_i)) / W
            + lambda * ||beta||² / 2
prediction = sigmoid(z), the probability of class 1
```

Labels must be exactly 0 or 1, with at least `minimum_rows_per_class` rows in
each class. Class order is fixed at `[0,1]` and saved in the model. `C` must be
positive finite with representable inverse/normalized regularization. The
implementation uses stable sigmoid and softplus, including a cancellation-safe
positive-label derivative at extreme margins. Overflow/underflow that makes the
normalization unrepresentable is rejected.

Defaults: `C=1`, intercept enabled, minimum one row per class. The intercept is
never penalized. Increasing every sample weight by `k` is equivalent in the
objective to increasing `C` by `k`; therefore choosing `TrainingMeanOne` rather
than `Preserve` deliberately changes raw-weight regularization semantics.
Class-weight construction is external; the backend performs no balancing.

This is linear classification, distinct from XGBoost's `binary:logistic`
boosted trees. Multiclass, L1 and elastic-net are outside this backend's API.

### Concomitant-scale Huber regression

```text
r_i = y_i - z_i
objective = sum(a_i * [sigma + sigma * rho_epsilon(r_i / sigma)])
            + alpha * ||beta||²
rho_epsilon(u) = u²                            if |u| <= epsilon
                 2 * epsilon * |u| - epsilon²  otherwise
sigma >= 10 * f64::EPSILON
```

Coefficients, optional intercept and positive scale are optimized jointly.
Neither intercept nor scale is penalized. `epsilon >= 1` and `alpha >= 0` must
be finite; defaults are `epsilon=1.35`, `alpha=0.0001`, intercept enabled.
Initialization is zero coefficients/intercept and `sigma=1`. This is neither
fixed-delta Huber loss nor pseudo-Huber.

The fitted scale is saved in the inference artifact and diagnostics. Prediction
is `x dot beta + b`, **not** divided by scale. Outliers in the training receipt
are counted by `|r| > epsilon * sigma`. The objective is a weighted sum, not a
weighted mean, so weight normalization also changes Huber's regularization
relative to the data term.

The objective accepts every finite positive trial scale: native line-search
arithmetic can briefly yield a value just below the lower bound by subtraction
roundoff. Bullet neither clips that trial nor changes the bound. Final solver
success still requires the returned scale to satisfy the exact lower bound;
non-positive/non-finite trials fail. A regression test covers the observed
positive sub-bound trial and the active-bound solution.

## Optimizer, convergence and reproducibility

`LbfgsbConfig` exposes bounded solver controls. Defaults:

| Parameter | Logistic | Huber |
| --- | ---: | ---: |
| History size | 10 | 10 |
| Maximum iterations | 1000 | 1000 |
| Maximum objective/gradient evaluations | 15000 | 15000 |
| Projected-gradient tolerance | 1e-4 | 1e-5 |
| Relative-function tolerance | 64 × machine epsilon | 1e7 × machine epsilon |

These are Bullet defaults, not a promise to mirror every sklearn default.
Relative tolerance maps to native `factr = tolerance / machine_epsilon`.
Convergence may be either projected-gradient or relative-function reduction;
a function-reduction stop does not imply the gradient met its tolerance.
Receipts record the actual reason, objective, projected-gradient infinity norm,
iterations and evaluations, along with estimator configuration, weight sum,
initialization, dtype, native version and model SHA-256.

Only native tasks 21 and 22 count as convergence. Invalid inputs, non-finite
results, abnormal line search, exhausted budgets, and infeasible final solutions
return errors without a model. There is no retry with enlarged budgets, fallback
estimator or automatic tolerance repair.

The crate's high-level wrapper does not expose the required budget/termination
control. Bullet uses its existing low-level `setulb` binding through a private
reverse-communication adapter with checked workspace dimensions. The numerical
optimizer is upstream code, not reimplemented in Bullet. Upstream C uses static
work variables, so a process-local mutex covers each entire solve. This protects
Bullet calls only; other code must not bypass the adapter and call the same
non-reentrant native library concurrently.

The pinned native line search allows **20** steps, versus **50** in sklearn
1.5.1's LogisticRegression `lbfgs` path. Summation, BLAS implementation and stop
trajectories can differ. Do not claim SciPy iteration equivalence, original SOTA
model parity, or universal cross-platform byte-identical training.

The released crate is BSD-3-Clause; exact source identity is its version and
published-crate checksum (also recorded in the vendored source manifest):

```text
lbfgsb 0.1.1
70b0d1a3440f0ad7ec488c97ccd03e1d76f0114342be1a5fa17442b0b25e075f
```

Its packaged VCS metadata is marked dirty; the Git revision alone does not
identify a clean source tree. The crate builds bundled reference miniCBLAS and
native solver code statically; it does not require a separately provisioned
optimizer shared library.

### Build-only binding compatibility patch

`vendor/lbfgsb` retains the pinned release's Rust wrapper, native build inputs
and both licenses. A root Cargo patch selects it. The only build-script change
limits bindgen to declarations in `lbfgsb.h`, excluding unrelated system math
APIs. This fixes bindgen 0.65.1's unsupported AArch64 vector-PCS ABI error with
Ubuntu 24.04/glibc 2.39; no compiler macros, native floating-point flags or
numerical source files are changed. The same filter is used on every platform.

`vendor/lbfgsb/SOURCE.json` records each retained upstream file's hash and the
build patch; `SHA256SUMS` is checked in native CI. See its `SOURCE.md` for the
Bullet ML maintainers' ownership and removal criteria. This is a documented,
removable build compatibility patch, not an estimator fallback or replacement.

## Evidence boundaries

The committed fixture comes from the optional development-only
`scripts/reference_linear_estimators.py`, pinned to sklearn 1.5.1, SciPy 1.13.1
and NumPy 1.26.4. It uses synthetic deterministic arithmetic inputs only, not
frozen strategy assets. CI reads the JSON and never executes that script.

There are eight cases: two estimators × intercept on/off × uniform/nonuniform
weights. Each coefficient, intercept, prediction and applicable Huber scale is
checked using tolerances declared before examining Bullet output:

| Quantity | Absolute tolerance |
| --- | ---: |
| Coefficient / intercept | 1e-5 |
| Prediction | 2e-5 |
| Huber scale | 1e-5 |

These checks establish bounded numerical agreement, not bitwise training parity.
Model JSON reload and OOS replay are separate checks requiring exact equality
within each platform, without rounding. Native CI runs both examples plus
linear-only and all-feature tests on Linux x86_64/ARM64; macOS ARM64 is validated
locally. Future suffix mutation must not alter earlier model/report hashes.
Tests also cover analytic gradients, unpenalized weighted-prior intercept,
weight/C equivalence, outliers, active scale bound, malformed artifacts, both
budgets, non-finite values, abnormal native termination and concurrent fits.

Performance evidence is in [`evidence/linear-estimators.json`](evidence/linear-estimators.json).
Each family completed N=256,512,1024 before a 2048-row target was authorized under
a 30-second budget. There are four vintages with N/4 training rows and two
features, and per-fold optimizer workload is recorded. This does not authorize
full BTC/ETH/SOL training or larger feature dimensions.

## Complexity review

Necessary new paths are confined to two objective implementations, optional
intercepts, Huber inlier/outlier residuals, binary classes, the optional native
feature, and invalid-input/budget/native-task boundaries. Each corresponds to a
mathematical contract or explicit failure mode; unit, fixture and replay tests
cover those boundaries. Generic rolling selection, accounting and schedule
routing are reused unchanged. There are no new compatibility fallbacks, asset
switches, preprocessing guesses, inherited optimizer state or parameter search.
The native lock and task adapter are explicit upstream safety/control boundaries.
The single build compatibility patch is the header allowlist described above;
remove it after a pinned upstream release fixes the binding scope and passes
the three-platform numerical, artifact and rolling-replay gates.

Primary references: [lbfgsb 0.1.1](https://docs.rs/lbfgsb/0.1.1/lbfgsb/),
[sklearn 1.5.1 logistic source](https://github.com/scikit-learn/scikit-learn/blob/1.5.1/sklearn/linear_model/_logistic.py),
[sklearn 1.5.1 Huber source](https://github.com/scikit-learn/scikit-learn/blob/1.5.1/sklearn/linear_model/_huber.py).
