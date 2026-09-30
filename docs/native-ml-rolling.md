# Native ML and rolling out-of-sample replay

Bullet's interfaces, feature/strategy callbacks, training orchestration, and
accounting are Rust. Numerical backends may be native C/C++ libraries behind
Rust bindings. The formal training and inference path does not launch Python,
embed an interpreter, or load Python objects.

This release is a library-level integration with a bounded synthetic example.
It is **not** a claim that BTC/ETH/SOL models have been retrained or that frozen
SOTA parity has been reproduced by the training layer.

## Installed capabilities

- Default SmartCore OLS/ridge and the existing expanding, row-count-based API.
- Optional community `xgb = 3.0.6` wrapper, pinned separately from native
  XGBoost **3.2.0**. The linked native version is checked before training and
  when loading an artifact; another native version fails closed.
- CPU `hist`, single native worker thread, an explicit seed, binary-logistic
  boosted trees and squared-error boosted-tree regression.
- Timestamp-based sliding windows; one independently fitted model per fold.
- Per-row label maturity, optional label gap, explicit model activation delay.
- Optional training-only population standardization (`ddof=0`; exactly constant
  columns use scale 1), independent of sample weighting.
- Positive finite sample weights, optionally divided by the **selected training
  rows'** mean weight. No future-window normalization.
- Native JSON artifacts, model/report hash checks, scheduled OOS target replay,
  and training-report hashes included in decision diagnostics/provenance.

XGBoost `binary:logistic` is boosted-tree classification. The separate
[`linear-backend`](linear-estimators.md) provides linear LogisticRegression and
concomitant-scale Huber using the same rolling lifecycle. Multi-output ensembles,
automatic parameter/feature/label search, online train-state checkpoints and a
full-scale streaming trainer remain outside the built-in backend surface.

## Build and run without Python

Prerequisites: Rust, curl, unzip, a C/C++ toolchain and libclang. On macOS ARM64,
install native OpenMP with `brew install libomp`. Linux distributions must
satisfy the supplied binaries' glibc >= 2.28 requirement.

```sh
scripts/prepare-xgboost.sh
cargo --config target/native/xgboost-3.2.0/cargo.toml test \
  -p bullet-ml -p bullet-ml-training --all-features --locked
cargo --config target/native/xgboost-3.2.0/cargo.toml run \
  -p bullet-ml-training --features xgboost-backend \
  --example rolling_xgboost --locked -- 128 target/native-oos-demo
```

The output directory must be new; the example does not overwrite evidence.
It trains four model vintages on synthetic daily observations, persists and
reloads every artifact/transform/report, and compares the entire decision input
and accounting result exactly. Pipeline warm history is seeded only from the
three observations preceding the OOS interval. Outputs include per-fold reports,
model artifacts, `ledger.json`, and `summary.json`.

The preparation script verifies SHA-256 pins for official XGBoost 3.2.0
archives. `.whl` is used only as a ZIP container for native binaries: **no pip,
Conda, Python module extraction, or Python execution**. Linux uses the official
CPU distribution; macOS uses the official ARM64 distribution and system
Homebrew OpenMP. Both macOS dependency edges resolve to the same OpenMP image;
loading separate copies is unsafe. An XGBoost license is retained separately.

On Linux the generated linker configuration uses `DT_RPATH` so Cargo's test-time
`LD_LIBRARY_PATH` cannot substitute an older wrapper-provided XGBoost binary.
The linked version is still independently verified by the backend.

The generated Cargo config supplies the wrapper's required **build-time**
`XGBOOST_LIB_DIR` and linker search/rpath. Bullet applications have no runtime
environment-variable configuration. The native path is local to this checkout;
binaries must not be relocated without a platform-specific native packaging
step. The existing `bullet-live` musl release stays independent of this optional
backend. `native-provenance.txt` records the archive hash and actual linked
native dependency paths. Keep it with experiment evidence. This is not a
portable self-contained ML release package.

## Time contract

Every dataset row is validated, including rows assembled by public Rust struct
fields:

```text
feature_end_ns <= decision_time_ns < label_end_ns
```

`RollingPlan::new(first_fit, prediction_end, history, step, label_gap,
activation_delay)` uses elapsed nanoseconds. For UTC days, use multiples of
86,400,000,000,000; this is not a calendar-month scheduler. For example, a
365-day lookback and 30-day retrain step are independent of the training-row
sampling rate and the strategy decision rate.

For each fold:

```text
training decisions: [fit_asof - history, fit_asof)
eligible labels:    label_end <= fit_asof - label_gap
model availability: [fit_asof + activation_delay, next activation)
```

The last prediction interval is explicitly truncated at `prediction_end`.
There is no hidden time repair, automatic sorting, random shuffle, reuse of an
expired model, or fallback fit. Too few mature rows or insufficient class
coverage returns an error. Callers choose a warmup/evaluation start with enough
training data. `TimeFold` and `RollingPlan::from_folds` allow explicit validated
windows; generated or deserialized plans are validated before fitting.

Every label is checked independently. A late-maturing earlier row can be
excluded while later rows are eligible. The legacy contiguous split API also
now checks **all** training labels rather than just the final row.

The eligibility boundary is the declared label-end timestamp. If a source's
label becomes available later, its adapter must declare that availability
conservatively. Feature, label and raw-weight business semantics remain the
adapter's responsibility. Prefix hashes cannot prove an arbitrary external
feature formula is causal.

Prediction windows are half-open. A frozen adapter using different endpoint
conventions must map those conventions explicitly; Bullet does not infer them.
Activation delay is a declared simulated delay, not a measured asynchronous
training service. A callback sees only the completed observation and cannot use
a future model simply because its next execution time crosses a model boundary.

## Backend and model contracts

`RollingTrainer` receives a validated `FitBatch` containing only selected
training rows, targets, weights and the generated unique vintage. It never
receives validation/future rows. `ArtifactModel` binds the returned model to the
SHA-256 in the trainer's receipt. `FittedFold::into_window` verifies report
integrity, artifact identity, transform, schema and causal bounds before routing.

`XgboostTrainer` validates binary labels and class counts, dimensions, finite
values, strictly positive weights, and conversion to the native `float32`
matrix contract. Overflow and weight underflow are errors. There is no claim
that float64 inputs remain float64 inside XGBoost. The receipt records the
dtype, native parameters, seed, wrapper version and native version.

The native artifact contains schema, model metadata, config, native JSON and its
hash. The artifact is intended for trusted training outputs, not arbitrary
untrusted native model payloads. File JSON serialization is used deliberately:
`xgb 3.0.6`'s buffer-save wrapper does not explicitly NUL-terminate its C config
string. This compatibility path is owned by the native adapter and can be
removed after an upstream fixed release passes identical artifact roundtrips.
The XGBoost adapter uses the community wrapper without a handwritten C ABI.
The linear backend has a private bounded adapter over its community solver
crate's already-generated `setulb` binding.

`ScheduledMlStrategy` owns one pipeline and one mapper across all model windows.
Their state is not reset at retraining boundaries. Warmup policy applies only
to missing feature history, not to missing or expired models. Initial model
absence, schedule gaps, overlap, duplicate vintages and expiry fail closed.

## Evidence and reproducibility

Each `FoldReport` records:

- actual selected sample count and original row-index hash;
- selected dataset/schema hashes, maximum training label end;
- raw/effective weight hashes and weight-normalization policy;
- fitted transform and transformed-feature hash;
- exact fit, label cutoff and prediction bounds;
- backend parameters, versions and model artifact SHA-256;
- weighted training MSE (a probability squared-error measure for binary labels);
- a bundle SHA-256 over the report with its own hash field empty.

The report intentionally excludes future dataset rows. Mutating a future suffix
must leave earlier reports, artifacts and decision prefixes unchanged. A
separate experiment manifest can retain the entire source dataset hash. Model
artifacts plus transforms are inference checkpoints; they are not optimizer,
feature-pipeline, mapper or full research-run checkpoints.

Tests cover sliding boundaries, variable label maturity, feature/schema errors,
future/gap/expired routing, state retention, sample weights, both native
objectives, per-element comparison with the community wrapper, corrupt artifacts,
report/model mismatches, future-suffix mutations and artifact reload replay.
Native CI runs on Linux x86_64 and ARM64 both before and after merging. Exact
comparisons are within each platform; cross-platform bit-identical training and
original Python SOTA model equivalence require separate evidence.

Before large training or replay, collect at least three completed bounded N-T
samples and budget the actual rows × vintages × model-size workload. The example
accepts only 64..4096 rows and is not a full-scale benchmark.

## Complexity review

New branches correspond to actual boundaries: model objective, transform policy,
weight policy, window availability, invalid input, and supported native platform.
There is no fallback estimator, stale-model reuse or automatic parameter repair.
The default/no-native feature path keeps existing evaluator and linear-training
users free of native dependencies; enabling the backend is explicit. The only
upstream compatibility workaround is the file-based serializer described above.
Strategy-specific labels, features, controller state and exposure mapping remain
outside Bullet's evaluator and training core.

Primary dependencies: [community wrapper](https://docs.rs/xgb/3.0.6/xgb/),
[XGBoost C API](https://xgboost.readthedocs.io/en/stable/c.html), and
[native model format](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).
