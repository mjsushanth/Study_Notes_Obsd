

## 3. Questions 16–30

### Q16. What is the exact training objective in XGBoost, including regularization?

For predictions f(x_i) and trees t=1..T:
    Obj = sum_i L(y_i, f(x_i)) + sum_t Omega(h_t)

Regularization term:
    Omega(h) = gamma * T + 0.5 * lambda * sum_j w_j^2

where:
- T = number of leaves in tree h
- w_j = leaf weight of leaf j
- gamma = penalty per leaf (controls tree size)
- lambda = L2 regularization on leaf weights

During each boosting step, XGBoost minimizes:
    Obj_new approx constant + sum_i (g_i * delta_i + 0.5 * h_i * delta_i^2) + reg
where g_i and h_i are gradient and Hessian of L wrt current prediction.

---

### Q17. How does XGBoost compute optimal leaf values, and why are leaves piecewise constant?

Given a leaf j with samples I_j:
    G_j = sum_{i in I_j} g_i
    H_j = sum_{i in I_j} h_i

To minimize quadratic approx:
    sum_{i in I_j} (g_i * w + 0.5 * h_i * w^2) + 0.5 * lambda * w^2

Derivative wrt w:
    G_j + (H_j + lambda) * w = 0

So:
    w_j* = - G_j / (H_j + lambda)

This is a single scalar per leaf, so predictions of that tree are:
    h(x) = w_j* for all x that fall into leaf j

Hence each tree is piecewise constant over its partition of feature space.

---

### Q18. How do you handle imbalanced classification with XGBoost?

Main techniques:
- tune `scale_pos_weight`:
  - roughly `scale_pos_weight = negative_count / positive_count`
  - increases gradient for positive class, balancing loss
- adjust objective and evaluation metric:
  - use `binary:logistic` with AUC/PR-AUC tracking, not just accuracy
- tune regularization:
  - stronger regularization on majority-dominated splits
- use stratified sampling or custom sample weights:
  - weights per instance, not just per class

Important: monitor metrics that reflect imbalance (PR-AUC, recall at fixed precision), not only logloss.

---

### Q19. What are common failure modes or pitfalls when using XGBoost?

Typical issues:
- data leakage:
  - target encoded features built on full dataset
  - cross-validation done incorrectly for time series
- overfitting with too-deep trees and high eta
- using accuracy on imbalanced data instead of AUC/PR metrics
- poor handling of categorical variables (e.g., arbitrary integer encoding)
- mismatch between objective and business metric
- wrong early stopping:
  - using validation from different distribution
  - tuning on test set

Mitigation:
- strict train/val/test separation
- correct CV scheme (k-fold, time-series CV)
- proper categorical handling (one-hot, target encoding with leakage-safe scheme)
- align objective with evaluation metric.

---

### Q20. XGBoost vs LightGBM: what are the main differences?

Both are gradient-boosted tree libraries, but:

XGBoost:
- traditional level-wise tree growth
- strong regularization, second-order objective
- supports exact and histogram split finders
- usually more conservative, stable

LightGBM:
- leaf-wise growth with depth constraints
- gradient-based one-side sampling and exclusive feature bundling
- extremely fast on large sparse datasets
- more prone to overfitting if not constrained (leaf-wise can grow very unbalanced trees)

Rule of thumb:
- XGBoost for stability and mature ecosystem
- LightGBM when speed and very large-scale training matter, with careful regularization.

---

### Q21. XGBoost vs CatBoost: how do they differ, especially for categorical features?

CatBoost:
- designed for categorical-heavy data
- uses ordered target statistics and permutations to avoid target leakage
- handles categorical variables natively; often minimal preprocessing

XGBoost:
- expects numeric features
- categorical variables need manual encoding:
  - one-hot, target encoding, frequency encoding, etc.
- more work to avoid leakage and handle high-cardinality categorical features.

So:
- CatBoost often wins when there are many categoricals.
- XGBoost is more generic and widely adopted; still strong if categorical engineering is done well.

---

### Q22. What are monotonic constraints in XGBoost, and when would you use them?

Monotonic constraints force the model to be:
- non-decreasing in some features
- or non-increasing in others

Configuration: `monotone_constraints` (e.g., `[1, -1, 0]` for three features).

Use cases:
- price should not decrease as "size" increases
- credit risk should not decrease as "number of defaults" increases

Benefits:
- embeds domain knowledge
- improves extrapolation and interpretability
- reduces risk of pathological behavior in regions with sparse data

Internally, splits that violate the monotonic direction are disallowed during tree construction.

---

### Q23. How would you deploy XGBoost in production and think about latency?

Key points:
- trained model is sequence of trees:
  - inference = traverse each tree, sum leaf values, apply link function (e.g., sigmoid)
- model can be:
  - loaded in Python (flask/fastapi)
  - exported to binary / JSON and loaded in C++/Java/Go for low-latency
  - compiled via Treelite / ONNX into optimized runtimes

Latency considerations:
- depth of trees and number of trees
- CPU cache behavior (trees stored in contiguous memory)
- batch vs single-instance inference
- whether shap or explanations are computed online (expensive)

For hard latency budgets, you tune:
- fewer trees
- shallower trees
- maybe LightGBM or Treelite-compiled XGBoost.

---

### Q24. How do you interpret XGBoost predictions with SHAP?

SHAP (SHapley Additive exPlanations):
- computes per-feature contribution for each prediction
- for tree models, TreeSHAP gives exact Shapley values in polynomial time

Usage:
- global explanation:
  - feature importance bar plots (mean absolute SHAP value)
  - dependence plots (SHAP vs feature value)
- local explanation:
  - per-instance bar or waterfall plot showing how features push prediction up/down from base value

Advantage:
- consistent additive explanation:
    f(x) = base_value + sum_j phi_j
  where phi_j are SHAP values per feature

This works very naturally for XGBoost because the model is an additive tree ensemble.

---

### Q25. How does early stopping work in XGBoost, and what are caveats?

Mechanism:
- split data into train and validation
- after each boosting iteration (each new tree), evaluate metric on validation
- if metric does not improve for N rounds (`early_stopping_rounds`), stop training and keep best iteration

Caveats:
- validation set must be representative; otherwise you stop too early or too late
- if you use random CV, must align early stopping with CV folds
- must not use test set as the early stopping validation set
- small eta with early stopping can still underfit if patience is too short

---

### Q26. Give a practical recipe for tuning XGBoost from scratch.

One reasonable approach:
1. Start simple:
   - max_depth = 4 to 6
   - eta = 0.1
   - subsample = colsample_bytree = 0.8
   - n_estimators large, rely on early stopping
2. Tune tree shape:
   - adjust max_depth and min_child_weight
   - deeper + smaller min_child_weight lowers bias, raises variance
3. Tune regularization:
   - sweep lambda, alpha
   - add gamma to penalize unnecessary leaves
4. Tune sampling:
   - adjust subsample, colsample_* to reduce overfitting
5. Finally lower eta:
   - eta = 0.05 or 0.01
   - increase n_estimators with early stopping

Always:
- use CV or holdout based on data regime (time-series vs iid)
- optimize for the metric that matters (AUC, RMSE, etc.).

---

### Q27. How would you use XGBoost for time-series forecasting?

Common pattern: treat it as supervised learning on lag features.

Steps:
- create lagged features: y_{t-1}, y_{t-2}, ..., external covariates
- ensure causal splits:
  - use time-based train/validation splits (no random shuffling)
  - possibly rolling-origin CV
- train XGBoost to predict y_t from lags and covariates
- careful with leakage from the future (do not use features that peek ahead)
- evaluate using time-appropriate metrics (e.g., RMSE, MAPE, SMAPE).

XGBoost is not sequence-aware by design; all memory is encoded in the features you construct.

---

### Q28. How do trees in XGBoost capture feature interactions compared to linear or polynomial models?

Linear model:
- f(x) = w^T x  (only linear terms; interactions must be engineered, e.g., x1*x2)

Polynomial model:
- explicit interaction terms added (x1*x2, x1^2, etc.)

Tree model:
- interactions appear implicitly when splits on multiple features are combined along paths
- a path like:
    if x1 > a and x2 < b then leaf_j
  corresponds to a region where the model behaves differently:
    f(x) = ... + w_j in that rectangle

So tree depth determines maximum interaction order:
- depth 1: main effects only
- depth 2: pairwise interactions
- depth d: up to d-way interactions

Boosting adds many such trees, building a rich library of interaction patterns without manual feature engineering.

---

### Q29. How do custom objectives and evaluation metrics work in XGBoost?

You can pass:
- `obj` (custom objective) taking (preds, dtrain) and returning (grad, hess)
- `feval` (custom eval) taking (preds, dtrain) and returning (name, value, higher_better)

Mechanism:
- For each iteration, XGBoost calls your `obj` to compute gradient g_i and Hessian h_i
- Tree construction uses those g_i, h_i with the same gain formulas
- `feval` is used purely for reporting and early stopping; it does not affect training directly

This lets you implement:
- asymmetric losses
- quantile loss
- custom ranking metrics approximated via differentiable surrogates.

---

### Q30. When is XGBoost a bad choice or clearly inferior?

Cases where XGBoost is not ideal:
- raw images / audio / text sequences:
  - structure is high-dimensional and highly local; CNNs/RNNs/transformers are better
- extremely high-dimensional sparse text (classic bag-of-words):
  - linear models with strong regularization can be simpler and faster
- very small datasets with simple relationships:
  - linear or small models might generalize better and be easier to interpret
- strict latency or memory constraints:
  - though XGBoost can be optimized, very large ensembles may be heavy compared to small neural nets or linear models

General rule:
- XGBoost shines on medium-to-large **tabular** data.
- For modality-rich tasks (vision, language) or ultra-low-latency micro-models, other architectures can dominate.

