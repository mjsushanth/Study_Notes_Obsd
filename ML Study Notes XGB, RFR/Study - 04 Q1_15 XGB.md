

## 1. Compact Mental Model

XGBoost = gradient boosting over decision trees + strong engineering:
- second-order (Newton-style) optimization
- explicit regularization on tree structure and leaf weights
- optimized split search (histograms, sparsity-aware)
- parallelization, cache-aware layout, distributed training

Model is an additive ensemble:
    f(x) = sum_{m=1..M} eta * h_m(x)
where h_m are small CART regression trees.

Goal: sequentially reduce loss by fitting each tree to the negative gradient (or negative residuals) of the previous ensemble.

---

## 2. Fifteen Core Interview Questions

### Q1. What is boosting and how is it different from bagging?

Boosting:
- sequential models
- each new model corrects errors of previous ensemble
- reduces bias while controlling variance

Bagging:
- many independent models on bootstrap samples
- predictions averaged
- reduces variance

Boosting = "correct and refine". Bagging = "average many noisy experts".

---

### Q2. What function class does XGBoost learn?

Additive model of regression trees:
    f(x) = sum( eta * h_m(x) )
Each h_m is a piecewise-constant CART tree.

XGBoost is linear in the space of trees, but highly nonlinear in the raw features.

---

### Q3. How does gradient boosting work conceptually?

At iteration m:
1. Compute pseudo-residuals:
       r_i = - dL(y_i, f(x_i)) / df
   For squared error: r_i = y_i - f(x_i).

2. Fit a regression tree h_m(x) to r_i.

3. Update:
       f_m(x) = f_{m-1}(x) + eta * h_m(x)

Interpretation: gradient descent in function space.

---

### Q4. How is XGBoost different from generic gradient boosting?

Key enhancements:
- second-order info (gradients g_i and Hessians h_i)
- closed-form leaf weights
- regularization (lambda for L2, alpha for L1, gamma for leaf creation)
- optimized/histogram splitting
- sparsity-aware handling of missing values
- large engineering for speed

---

### Q5. How does XGBoost choose splits? What is the Gain formula?

For a node with gradient sum G and Hessian sum H:
Leaf weight:
    w* = - G / (H + lambda)

Score of leaf:
    Score(G,H) = -0.5 * (G^2 / (H + lambda))

For candidate split into Left (G_L, H_L) and Right (G_R, H_R):
    Gain = Score(G_L,H_L) + Score(G_R,H_R) - Score(G,H) - gamma

Choose split with maximum positive Gain. If no positive Gain, stop split.

---

### Q6. Why does XGBoost use second-order (Hessian) terms?

Using Taylor expansion:
    L(y, f + delta) approx L(y,f) + g*delta + 0.5*h*(delta^2)

XGBoost fits tree outputs delta = h_m(x) to minimize the quadratic approximation. Hessian gives curvature, enabling better leaf weights and more stable optimization than gradient-only methods.

---

### Q7. How does XGBoost prevent overfitting?

- tree complexity limits: max_depth, min_child_weight
- gamma: cost for adding a leaf
- regularization: lambda (L2), alpha (L1)
- eta (learning rate): smaller = more stable, less variance
- subsampling: subsample for rows, colsample_* for features
- early stopping

---

### Q8. When choose XGBoost vs Random Forest?

Random Forest:
- many deep trees averaged
- variance reduction
- strong baseline, low tuning needs

XGBoost:
- sequential trees, bias reduction
- more expressive and tunable
- usually superior on structured/tabular data

---

### Q9. How does XGBoost handle missing values?

- sparsity-aware splitting
- for each split, XGBoost learns a default direction (left or right) for missing values
- missingness can itself act as a signal

No imputation required; routing learned contextually at each node.

---

### Q10. Which hyperparameters affect bias vs variance?

High variance --> lower by:
- lower max_depth
- higher min_child_weight
- higher gamma
- higher lambda/alpha
- smaller eta
- stronger subsampling (subsample, colsample_bytree)

High bias --> lower by:
- deeper trees
- lower regularization
- higher eta
- more trees

---

### Q11. Why does small learning rate (eta) improve generalization?

Small eta = smaller steps in function space.
Each tree contributes only a small correction.
Requires more trees but reduces risk of overfitting noise.
Equivalent to gradient descent with small step size.

---

### Q12. How is feature importance computed in XGBoost?

Typical measures:
- Gain: total split gain using a feature (most informative)
- Cover: number of samples split on that feature
- Frequency: how often feature appears in splits

Pitfalls:
- correlated features split importance
- high-cardinality features may appear overly important
- importance is global; local behavior needs SHAP

---

### Q13. Why can XGBoost beat neural nets on tabular data?

Tabular data:
- irregular patterns, non-smooth interactions
- moderate dataset size
- mixed types and sparsity

Trees naturally capture nonlinearity and interactions without scaling. Boosting with regularization handles small-to-medium datasets well. Neural nets often need far more data and careful preprocessing.

---

### Q14. How does XGBoost handle multiple losses (regression, classification, ranking)?

XGBoost only needs gradient g_i and Hessian h_i for each data point:
- regression: mse, mae, huber, etc.
- classification: logistic, softmax
- ranking: pairwise or listwise losses

Tree building is identical once g_i and h_i are defined.

---

### Q15. Systematic approach to fixing overfitting in XGBoost?

1. Reduce tree depth (max_depth).
2. Increase min_child_weight, gamma.
3. Increase regularization (lambda, alpha).
4. Lower eta, raise number of trees.
5. Use row/column subsampling.
6. Enable early stopping.
7. Inspect features: remove leakage, reduce high-cardinality issues.
8. Check SHAP for dominant or suspicious features.

Goal: reduce model capacity and add randomness until train and validation scores converge.

