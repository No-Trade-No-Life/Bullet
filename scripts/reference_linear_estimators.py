#!/usr/bin/env python3
"""Optional, isolated reference-fixture generation; never part of Bullet runtime/CI.

Requires the frozen reference environment: sklearn 1.5.1, scipy 1.13.1,
numpy 1.26.4. Generates synthetic data only and does not read SOTA assets.
Rust tests consume the committed JSON; no Python package is needed to run them.
"""
import json
import pathlib
import warnings
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, LogisticRegression

assert (sklearn.__version__, scipy.__version__, np.__version__) == ("1.5.1", "1.13.1", "1.26.4")
warnings.filterwarnings("error", category=ConvergenceWarning)
# These bounds are declared before examining Bullet optimizer output.
TOLERANCES = {"coefficient_atol": 1e-5, "prediction_atol": 2e-5, "scale_atol": 1e-5}
x = np.array([[(i % 17 - 8) / 3, ((i * 5) % 13 - 6) / 4, ((i * i) % 11 - 5) / 5] for i in range(96)], dtype=np.float64)
weights = np.array([1 + (i % 5) * 0.25 for i in range(96)], dtype=np.float64)
margin = 0.8 * x[:, 0] - 0.6 * x[:, 1] + 0.4 * x[:, 2] + 0.35
y_class = np.array([(i * 37 % 101) < 100 / (1 + np.exp(-v)) for i, v in enumerate(margin)], dtype=np.int64)
y_reg = 1.4 * x[:, 0] - 0.75 * x[:, 1] + 0.3 * x[:, 2] + 2.0
for i in range(len(y_reg)):
    y_reg[i] += 0.15 * ((i * 7) % 9 - 4)
    if i % 13 == 0:
        y_reg[i] += 15 if i % 2 else -15
cases = []
for family in ("logistic", "huber"):
    for intercept in (True, False):
        for weighted in (True, False):
            a = weights if weighted else np.ones(len(x))
            y = y_class if family == "logistic" else y_reg
            if family == "logistic":
                config = dict(C=0.7, fit_intercept=intercept, max_iter=1000, tol=1e-9, solver="lbfgs", penalty="l2")
                model = LogisticRegression(**config).fit(x, y, sample_weight=a)
                coef = model.coef_[0]
                bias = float(model.intercept_[0]) if intercept else 0.0
                prediction = model.predict_proba(x)[:, 1]
                scale = None
            else:
                config = dict(alpha=0.03, epsilon=1.35, fit_intercept=intercept, max_iter=1000, tol=1e-9, warm_start=False)
                model = HuberRegressor(**config).fit(x, y, sample_weight=a)
                coef, bias, scale = model.coef_, float(model.intercept_), float(model.scale_)
                prediction = model.predict(x)
            cases.append(dict(name=f"{family}-intercept-{intercept}-weighted-{weighted}", family=family,
                config=config, weighted=weighted,
                expected=dict(coefficients=coef.tolist(), intercept=bias, scale=scale, predictions=prediction.tolist())))
output = pathlib.Path(__file__).resolve().parents[1] / "crates/bullet-ml-training/tests/fixtures/linear-estimator-oracle.json"
output.write_text(json.dumps(dict(scope="synthetic optimizer numerical agreement, not SOTA model parity",
    versions=dict(sklearn=sklearn.__version__, scipy=scipy.__version__, numpy=np.__version__),
    tolerances=TOLERANCES, inputs=dict(features=x.tolist(), logistic_targets=y_class.astype(float).tolist(), huber_targets=y_reg.tolist(), weighted_weights=weights.tolist(), uniform_weights=np.ones(len(x)).tolist()), cases=cases), indent=2, allow_nan=False) + "\n")
print(output)
