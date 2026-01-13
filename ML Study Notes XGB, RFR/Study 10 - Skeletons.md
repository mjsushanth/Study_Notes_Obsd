
**Mental model:** one `run_experiment(model_ctor, params)` that can swap. 

The experiment does _data → split → fit → eval → log_; the only “model-specific” detail is how construct the estimator and whether you need early stopping callbacks.

```
def run_experiment(X, y, *, model_name: str, model, metrics_fn):
    # 1) Split (time-aware or random)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)

    # 2) Fit
    model.fit(X_tr, y_tr)

    # 3) Predict
    pred_te  = model.predict(X_te)
    pred_val = model.predict(X_val)

    # 4) Metrics
    m_te  = metrics_fn(y_te, pred_te)
    m_val = metrics_fn(y_val, pred_val)

    # 5) Logging / artifacts
    log_params(model_name, model.get_params())
    log_metrics(m_te, prefix="test_")
    log_metrics(m_val, prefix="val_")
    log_model(model, artifact_path="model")

    return m_te, m_val
```

## 2) RandomForestRegressor skeleton (sklearn) — interview-ready

**Key points to mention:** bagging ensemble; no early stopping; CPU parallelism via `n_jobs`; simple `.fit/.predict`.

```
from sklearn.ensemble import RandomForestRegressor

def train_rf_regressor(X_tr, y_tr, *, seed: int):
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,         # CPU parallelism
    )
    rf.fit(X_tr, y_tr)
    return rf

def rf_pipeline(X, y, seed: int):
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data_time_series(X, y)

    model = train_rf_regressor(X_tr, y_tr, seed=seed)

    pred_te  = model.predict(X_te)
    pred_val = model.predict(X_val)

    test_metrics = regression_metrics(y_te, pred_te)
    val_metrics  = regression_metrics(y_val, pred_val)

    # (optional) MLflow logging
    log_all(model, test_metrics, val_metrics, tags={"model.family": "RandomForestRegressor"})
    return test_metrics

```


## 3) XGBoost skeleton A — sklearn wrapper (`XGBClassifier/XGBRegressor`)

This is the “looks like sklearn” version. It’s useful because it integrates nicely with sklearn pipelines / CV, but under the hood you still get GPU if you set the device knobs.

```
from xgboost import XGBClassifier, XGBRegressor

def train_xgb_sklearn(X_tr, y_tr, X_val, y_val, *, task: str, seed: int, use_gpu: bool):
    common = dict(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,     # eta
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",   # key switch
        n_jobs=-1 if not use_gpu else None,    # CPU threads matter only on CPU
    )

    if task == "clf":
        model = XGBClassifier(objective="multi:softmax", num_class=7, eval_metric="mlogloss", **common)
    else:
        model = XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)

    # Early stopping is common with XGBoost; RF doesn’t have it.
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model

```


## 4) XGBoost skeleton B — native training with `DMatrix` + explicit CPU/GPU knobs

This is the one you actually benchmarked. It’s more “systems / performance” friendly because you control data containers, training loop, and evaluation.


```
import xgboost as xgb

def train_xgb_native_dmatrix(X_tr, y_tr, X_val, y_val, *, seed: int, use_gpu: bool):
    # 1) Convert to XGBoost-native containers
    dtr  = xgb.DMatrix(X_tr,  label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 2) Params: same algorithm, different backend
    params = {
        "objective": "multi:softmax",
        "num_class": 7,
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,

        # Backend selection:
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
    }

    # 3) Watchlist enables eval logging / early stopping
    watchlist = [(dtr, "train"), (dval, "val")]

    # 4) Train booster
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=300,
        evals=watchlist,
        verbose_eval=False,
        # early_stopping_rounds=30,  # common in practice
    )
    return booster

def predict_xgb_native(booster, X_te):
    dte = xgb.DMatrix(X_te)
    return booster.predict(dte)

```


### RandomForestRegressor knobs

- `n_estimators`: number of trees. More trees reduce variance and stabilize predictions; runtime scales roughly linearly.
- `max_depth`: controls tree complexity. Deeper trees can fit nonlinearities but overfit; also increases compute.
- `min_samples_leaf`: regularization via minimum leaf size; larger values smooth predictions and reduce variance.
- `n_jobs=-1`: CPU parallelism across trees.

### XGBoost core knobs (these drive both performance and behavior)

- `num_boost_round` / `n_estimators`: number of boosting iterations. More rounds = more capacity; also more compute.
- `max_depth`: tree depth per boosting round; very influential. Depth increases expressiveness and compute per round.
- `eta` / `learning_rate`: shrinkage. Smaller values reduce overfitting risk but require more rounds.
- `subsample`, `colsample_bytree`: stochastic regularization; reduces overfitting, can speed training, and improves robustness.
- `eval_metric`: evaluation metric used for monitoring (logloss, rmse, etc.), not necessarily the training objective.
- `seed`: affects row/feature subsampling; GPU may still be nondeterministic in practice.

### The GPU/CPU execution knobs (the key interview answer)

- `tree_method="hist"`: histogram-based split finding. It’s the scalable method and the one that is GPU-friendly.
- `device="cuda"`: tells XGBoost to run supported operations on the GPU.
- In older idioms: `tree_method="gpu_hist"` explicitly forces GPU split finding; but your run proved `device="cuda"` works.