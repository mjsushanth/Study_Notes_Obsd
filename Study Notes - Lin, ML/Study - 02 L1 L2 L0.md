
**L0, L1, L2, and Elastic Net**. Think of it as your core mental model for anything from Lasso / Ridge to RuleFit and LIME-style local surrogates. ([Unchitta][1])

---

## 1. Why sparse linear models?

Start with a standard linear model:

* Data:
  * `X ∈ R^{n × p}` (n samples, p features)
  * `y ∈ R^n` (target)

* Model:
  * prediction `ŷ = X β` (ignoring intercept for brevity)

Plain least squares solves:

> minimize over β:
> `L(β) = (1/2n) ||y − Xβ||²`

In high-dim regimes (p comparable to or larger than n), this has problems:

1. **Overfitting**: infinitely many β that fit almost perfectly.
2. **Instability**: tiny changes in data produce huge swings in β.
3. **Non-interpretability**: hundreds or thousands of small nonzero coefficients.

Sparse linear models fix this by adding a **penalty** that encourages many coefficients to be exactly 0.

General form:

> minimize over β:
> `L(β) + λ * P(β)`

where `P(β)` is some norm/penalty, and λ ≥ 0 controls the strength.

---

## 2. L0 regularization: the “true” sparsity objective

**L0 penalty** counts how many nonzero coefficients you have:

* `||β||_0 = number of j s.t. β_j ≠ 0`

Objective:

> `min_β (1/2n) ||y − Xβ||² + λ ||β||_0`

Interpretation:

* Each nonzero feature costs you λ units of penalty.
* The optimizer chooses a **subset of features** that best balances error vs. model size.

Conceptually:

* This is the **ideal feature selection** problem.
* You directly ask:

  > “Which subset of features should be in the model?”

But:

* This is **combinatorial and NP-hard** in general (you are searching over 2^p subsets). ([KiltHub][2])

Practical impact:

* Exact L0 is rarely used directly on large p.
* People either:

  * solve greedy approximations (forward selection, stepwise), or
  * use L1 as a convex surrogate (Lasso), or
  * specialized mixed-integer / branch-and-bound solvers for small and medium problems (e.g., SLIM scoring systems). ([arXiv][3])

Mental model:

> L0 is the *gold standard* for sparsity and interpretability: “pay per feature used”.

---

## 3. L1 regularization (Lasso): convex sparsity surrogate

**L1 penalty**:

* `||β||_1 = Σ_j |β_j|`

Objective:

> `min_β (1/2n) ||y − Xβ||² + λ Σ_j |β_j|`

This is the **Lasso**. ([Scribd][4])

Key properties:

1. **Convex** optimization problem → can be solved efficiently with coordinate descent, etc. ([ResearchGate][5])
2. Encourages **exact zeros** in β because |β| has a sharp “corner” at 0 → the gradient / subgradient pushes many coefficients to 0. ([Cross Validated][6])
3. Acts as a **relaxed version of feature selection**:

   * Many β_j = 0 → those features are dropped.
   * Nonzero β_j get shrunk toward 0.

Geometric intuition:

* Unregularized least squares solution is the point that minimizes squared error ellipses.
* L1 penalty creates a **diamond-shaped constraint region** in coefficient space.
* The optimum tends to land on a corner of the diamond → some coefficients exactly zero.

Bias–variance behavior:

* Bias ↑ (estimates are shrunk).
* Variance ↓ (model simpler, more stable).
* Good when p is large and many features are irrelevant.

Use cases:

* High-dimensional sparse problems (text, genomics).
* You want both prediction and **feature selection**.
* As local explainer models (e.g., LIME uses sparse linear models for local explanations). ([Christoph M.][7])

---

## 4. L2 regularization (Ridge): smooth shrinkage, no feature selection

**L2 penalty**:

* `||β||_2² = Σ_j β_j²`

Objective:

> `min_β (1/2n) ||y − Xβ||² + (λ/2) Σ_j β_j²`

This is **Ridge regression**. ([Scribd][4])

Key properties:

1. Also convex and easy to optimize (closed form solution with normal equations).
2. Does **not** promote zeros; instead, shrinks all coefficients **smoothly** toward 0.
3. Great at handling **multicollinearity**:

   * correlated features → LS unstable
   * ridge stabilizes by distributing weight among them.

Geometric intuition:

* L2 constraint is a **sphere/ball** in coefficient space.
* Solution tends to lie on smooth boundary; no sharp corners → rarely hits exactly 0.

Bias–variance behavior:

* Adds bias but can drastically reduce variance.
* Particularly helpful when features are correlated and you care mainly about predictive performance, not feature selection.

Use cases:

* Many moderately relevant features.
* You want predictive stability more than interpretability.
* Operand in logistic regression, GLMs (e.g., ridge logistic).

---

## 5. Elastic Net: interpolating between L1 and L2

**Elastic Net** combines both penalties:

> `min_β (1/2n) ||y − Xβ||² + λ [ α ||β||_1 + (1 − α) (1/2) ||β||_2² ]`
> with `0 ≤ α ≤ 1` controlling the mix.

Special cases:

* α = 1 → pure Lasso (L1 only).
* α = 0 → pure Ridge (L2 only). ([Scribd][4])

Motivation:

1. Lasso struggles with groups of correlated features:

   * tends to pick one and drop others arbitrarily.
2. Elastic Net encourages **grouped selection**:

   * L2 part spreads weight across correlated features.
   * L1 part still pushes many coefficients to 0.

So Elastic Net:

* keeps **sparsity** from L1,
* keeps **stability and grouping** from L2.

Use cases:

* p >> n with strong feature correlations (e.g., many mildly redundant predictors).
* You want a stable sparse model.

---

## 6. How these penalties shape function space and interpretability

Think in terms of what you want out of the model:

### 6.1 If you care about pure predictive performance

* Ridge or Elastic Net often perform more stably than pure Lasso.
* Lasso’s hard zeros can hurt if many features share information. ([Reddit][8])

### 6.2 If you care about interpretability (few features, simple logic)

* L0 would be the ideal, but infeasible in large dimensions.
* L1 is the practical go-to:

  * “Which small subset of variables explains the data best?”

### 6.3 If you want “interpretable but robust”

* Elastic Net with α in (0.3–0.8) is a good compromise:

  * some grouping behavior
  * some exact sparsity
  * less unstable than pure Lasso when features are correlated.

### 6.4 When do we push toward L0 again?

Recent work pushes toward **non-convex penalties** (SCAD, MCP, approximate L0) to recover even sparser and more accurate models, at the cost of harder optimization. ([arXiv][9])

Think:
L0 ⇐ Lp with p<1 ⇐ L1 ⇐ Elastic Net ⇐ L2
as a spectrum from **hard sparsity** to **smooth shrinkage**.

---

## 7. Algorithmic flows (how these are actually solved)

Most practical implementations (e.g., `glmnet`, `skglm`, scikit-learn’s Lasso/Ridge/ElasticNet) use some variant of **coordinate descent**: ([Scribd][4])

1. Fix all β_k except one β_j.
2. Solve the 1D optimization for β_j (which has a closed form update with L1/L2).
3. Cycle over j until convergence.

For L1, the coordinate update uses **soft thresholding**:

* If gradient for β_j is small in magnitude, set β_j to **0**.
* If large, shrink it toward zero by λ.

For L0, there is no convex closed form; algorithms are:

* greedy (forward/backward stepwise), or
* discrete/branch-and-bound, or
* L1 → L0 refinement (e.g., Lass-0). ([arXiv][9])

---

## 8. How this ties back to interpretable ML (RuleFit, LIME, etc.)

Now connect the dots:

* **LIME / local surrogates**:

  * Fit a **local sparse linear model** (via L1) to approximate the black-box around a point.
  * You get a handful of nonzero coefficients → local explanation. ([Christoph M.][7])

* **RuleFit**:

  * Turns tree paths into binary rule features.
  * Then fits a **sparse linear model (L1 or Elastic Net)** on these rule features.
  * The final explanation is a short list of high-weight rules.

* **Scoring systems (SLIM)**:

  * Use L0 + integer constraints to build “add +5 points if condition A” style models. ([arXiv][3])

All of these are variations on the same idea:

> Use sparsity-inducing penalties (L0, L1, Elastic Net) to build **small, human-digestible linear models** in some feature space.

---

## 9. Cheat-sheet style summary

* **L0**:

  * Penalty: number of nonzero β_j
  * Pros: true feature selection, maximal interpretability
  * Cons: combinatorial, NP-hard in general
  * Use: small/medium models with specialized solvers, scoring systems

* **L1 (Lasso)**:

  * Penalty: Σ |β_j|
  * Pros: convex, yields exact zeros, automatic feature selection
  * Cons: unstable with correlated features, biased estimates
  * Use: high-dim sparse, when you want a short list of features

* **L2 (Ridge)**:

  * Penalty: Σ β_j²
  * Pros: stable, handles multicollinearity, simple math
  * Cons: no zeros, no feature selection
  * Use: predictive performance with many correlated features

* **Elastic Net**:

  * Penalty: α * L1 + (1−α) * L2
  * Pros: compromise; group selection; some sparsity, some stability
  * Cons: two hyperparameters (λ, α), slightly more complex tuning
  * Use: p >> n with correlated features; interpretable yet robust models



[1]: https://unchitta.com/wp-content/uploads/2020/03/Interpretability-in-ML-Sparse-Linear-Models.pdf?utm_source=chatgpt.com "Interpretability in ML & Sparse Linear Models - unchitta.com"
[2]: https://kilthub.cmu.edu/ndownloader/files/39356993?utm_source=chatgpt.com "Algorithms for Interpretable High-Dimensional Regression"
[3]: https://arxiv.org/abs/1306.6677?utm_source=chatgpt.com "Supersparse Linear Integer Models for Interpretable ..."
[4]: https://www.scribd.com/document/714982139/Sparse-linear-regression?utm_source=chatgpt.com "Sparse Linear Models with glmnet - Logistic Regression"
[5]: https://www.researchgate.net/publication/360031017_Beyond_L1_Faster_and_Better_Sparse_Models_with_skglm?utm_source=chatgpt.com "Beyond L1: Faster and Better Sparse Models with skglm"
[6]: https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models?utm_source=chatgpt.com "Why L1 norm for sparse models - Cross Validated"
[7]: https://christophm.github.io/interpretable-ml-book/lime.html?utm_source=chatgpt.com "14 LIME – Interpretable Machine Learning"
[8]: https://www.reddit.com/r/learnmachinelearning/comments/1eqp6bc/l1_vs_l2_regularization_which_is_better/?utm_source=chatgpt.com "L1 vs L2 regularization. Which is \"better\"?"
[9]: https://arxiv.org/abs/1511.04402?utm_source=chatgpt.com "Lass-0: sparse non-convex regression by local search"
