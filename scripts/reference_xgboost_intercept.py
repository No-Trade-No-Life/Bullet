#!/usr/bin/env python3
"""Development-only XGBoost 3.2 sklearn oracle. Rust CI consumes committed JSON."""
import json
from pathlib import Path
import numpy as np
import xgboost
from xgboost import XGBClassifier, XGBRegressor
assert xgboost.__version__ == "3.2.0"
x = np.array([[(i % 17 - 8) / 3, ((i * 5) % 13 - 6) / 4, ((i*i) % 11 - 5) / 5] for i in range(128)], dtype=np.float64)
p = np.array([[(i % 19 - 9) / 3, ((i * 7) % 11 - 5) / 4, ((i*i) % 13 - 6) / 5] for i in range(32)], dtype=np.float64)
yc = np.array([int(i % 7 != 0) for i in range(128)])
yr = 50.0 + 1.5*x[:,0] - 0.7*x[:,1] + 0.2*x[:,2]
w = np.array([1.0+(i % 4)*0.25 for i in range(128)])
params=dict(n_estimators=128,max_depth=3,learning_rate=0.05,min_child_weight=20.0,max_bin=64,reg_lambda=5.0,reg_alpha=0.0,subsample=1.0,colsample_bytree=1.0,random_state=20260903,n_jobs=1,tree_method="hist",verbosity=0)
cases=[]
for kind in ['classification','regression']:
 for weighted in [False,True]:
  weights=w if weighted else np.ones(128)
  y=yc if kind=='classification' else yr
  model=(XGBClassifier(objective='binary:logistic',eval_metric='logloss',**params) if kind=='classification' else XGBRegressor(objective='reg:squarederror',eval_metric='rmse',**params))
  model.fit(x,y,sample_weight=weights,verbose=False)
  predictions=model.predict_proba(p)[:,1] if kind=='classification' else model.predict(p)
  native=json.loads(model.get_booster().save_raw(raw_format='json'))
  cases.append(dict(kind=kind,weighted=weighted,targets=y.tolist(),weights=weights.tolist(),predictions=predictions.tolist(),initial_intercept=native['learner']['learner_model_param']))
out=Path(__file__).resolve().parents[1]/'crates/bullet-ml-training/tests/fixtures/xgboost-intercept-oracle.json'
out.write_text(json.dumps(dict(scope='independent sklearn XGBoost native-initial-intercept regression test, not a strategy',xgboost_version=xgboost.__version__,comparison='exact f32 predictions, no tolerance or rounding',features=x.tolist(),prediction_features=p.tolist(),cases=cases),indent=2,allow_nan=False)+'\n')
