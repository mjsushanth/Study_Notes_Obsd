
([Past chat](https://chatgpt.com/c/69342a08-b2c8-8325-bdd4-c84e22996ac5 "Study notes style guide"))([Past chat](https://chatgpt.com/c/69342a08-b2c8-8325-bdd4-c84e22996ac5 "Study notes style guide"))([Past chat](https://chatgpt.com/c/691b62a4-7370-8332-ad7d-6fe18ae67734 "MLFlow lab study - XGB RFR"))([Past chat](https://chatgpt.com/c/691b62a4-7370-8332-ad7d-6fe18ae67734 "MLFlow lab study - XGB RFR"))

Second-order ideas (Hessians/curvature) are very practical in _some_ ML corners; they’re “impractical” mainly when you mean “form the full Hessian in parameter space for a modern deep net and invert it.” The trick is that many successful methods either (i) use **second-order information in a low-dimensional surrogate space**, or (ii) use **structured/approximate curvature** that’s cheap enough.

---

## 1) What “second order” really buys you

Take an objective (L(\theta)). Around the current parameters (\theta), a local quadratic approximation is

[  
L(\theta+\Delta);\approx; L(\theta) + g^\top \Delta + \tfrac{1}{2}\Delta^\top H \Delta  
]

where (g=\nabla L(\theta)) and (H=\nabla^2 L(\theta)) is the Hessian (curvature).  
Minimizing that quadratic gives Newton’s step:

[  
\Delta^\star = - H^{-1} g  
]

Interpretation: **gradient** tells you the direction of steepest increase; **Hessian** tells you how the slope changes as you move—so (H^{-1}) acts like a _geometry-aware rescaling_ of the gradient (big steps in flat directions, small steps in steep directions).

Why it’s “impractical” in deep nets: if (\theta\in\mathbb{R}^d) with (d\sim 10^7), storing (H) is (O(d^2)), and inverting is worse.

---

## 2) Where second order is used _in practice_ (yes: XGBoost / GBDTs)

### The key subtlety

In boosted trees like **XGBoost**, “second order” usually means the **second derivative of the loss w.r.t. the model’s prediction for each datapoint**, not w.r.t. millions of weights. That Hessian is **scalar per example** (or tiny), so it’s cheap.

XGBoost explicitly uses a second-order Taylor expansion of the objective at each boosting round (t). In their notation, with (g_i) and (h_i) as first and second derivatives of the loss for point (i), the per-round objective becomes a sum of terms involving (g_i) and (h_i). ([arXiv](https://arxiv.org/pdf/1603.02754?utm_source=chatgpt.com "XGBoost: A Scalable Tree Boosting System"))

Concretely, for a candidate tree (f_t), XGBoost approximates:

[  
\tilde L^{(t)} = \sum_i \left[g_i f_t(x_i) + \tfrac{1}{2} h_i f_t(x_i)^2\right] + \Omega(f_t)  
]

and then the optimal leaf weight for a leaf that collects instances (I_j) depends on aggregated gradient/Hessian statistics (G_j=\sum_{i\in I_j}g_i), (H_j=\sum_{i\in I_j}h_i). This is why split scoring and leaf values can be computed from **just sums of (g) and (h)**—no giant Hessian matrices. ([arXiv](https://arxiv.org/pdf/1603.02754?utm_source=chatgpt.com "XGBoost: A Scalable Tree Boosting System"))

XGBoost even exposes this directly in custom objectives: you must return both **gradients and Hessians** for the loss. ([xgboost.readthedocs.io](https://xgboost.readthedocs.io/en/latest/tutorials/advanced_custom_obj.html?utm_source=chatgpt.com "Advanced Usage of Custom Objectives"))

### LightGBM (and friends)

LightGBM similarly relies on gradient/Hessian statistics, and accelerates split finding by histogram/bin aggregation of these statistics. ([Michael Brenndoerfer](https://mbrenndoerfer.com/writing/lightgbm-fast-gradient-boosting-leaf-wise-tree-growth-complete-guide-mathematical-foundations-python-implementation?utm_source=chatgpt.com "LightGBM: Fast Gradient Boosting with Leaf-wise Tree Growth"))

### Important distinction

Plain decision trees (CART) pick splits via impurity measures (Gini/entropy/MSE), not Hessians. “Second-order boosting” is specifically about **optimizing a differentiable loss via derivatives**, which standard single trees don’t do.

---

## 3) If second order is so good, why do we get high accuracy with only first order?

Because “optimization efficiency” and “final generalization/accuracy” are not the same thing, and modern ML has structural reasons first-order works extremely well:

### (a) Scale and noise actually help

SGD’s minibatch noise is not just tolerated; it often _helps_ by (i) escaping saddle regions and (ii) biasing toward “flatter” solutions that generalize well. You don’t need an exact Newton step to land in a good basin.

### (b) Overparameterization changes the geometry

Modern nets have many more parameters than constraints. There are typically many global minima (or near-minima) with similar training loss. The hard part isn’t finding _the_ minimum; it’s finding a _good_ one that generalizes. First-order methods are good enough and scale to huge models.

### (c) Cost matters more than iteration count

Newton-type methods can reduce iterations, but each iteration is expensive. First-order steps are cheap and parallelizable; you can afford millions of them.

### (d) The Hessian you “want” is not always stable in deep nets

The true Hessian in deep nets is often indefinite, ill-conditioned, and changes rapidly early in training. Naive second-order steps can be unstable without trust regions/damping, at which point you’re back to approximations.

---

## 4) Are there “partial”/approximate Hessian methods used for real?

Yes—this is a big, active area. The practical variants fall into a few families:

### 4.1 Quasi-Newton (curvature from gradients): L-BFGS

Instead of forming (H), methods like **(L-)BFGS** build an approximation to (H^{-1}) from successive gradient differences. They’re widely used for convex-ish problems (logistic regression, CRFs, smaller nets, fine-tuning in some regimes), but less common for giant deep nets due to memory/line-search complications.

### 4.2 Gauss-Newton / Fisher / Natural gradient (curvature of the model’s output distribution)

For least squares and many probabilistic models, the **Gauss-Newton** matrix (and its probabilistic cousin, the **Fisher Information**) gives a PSD curvature proxy that’s more stable than the raw Hessian. This underlies natural-gradient style training and approximations like K-FAC (block-structured curvature).

### 4.3 Hessian-vector products (HVPs): “second order without the matrix”

You can compute (H v) efficiently via automatic differentiation without explicitly building (H). This enables truncated Newton / conjugate-gradient / trust-region style methods that use curvature implicitly.

### 4.4 Diagonal / block-diagonal Hessian approximations (cheap curvature)

A concrete example: **AdaHessian** estimates (an approximation to) the **diagonal of the Hessian** using stochastic tricks (Hutchinson) and then uses it to adapt step sizes; it’s explicitly a second-order adaptive optimizer. ([arXiv](https://arxiv.org/abs/2006.00719?utm_source=chatgpt.com "ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning"))

### 4.5 “Adaptive first order” (Adam/AdamW) as _preconditioning_, not true Hessian

Adam/AdamW use moving averages of the gradient and squared gradient; that’s not the Hessian, but it behaves like a diagonal preconditioner. It’s best viewed as “first-order + geometry heuristics,” not Newton.

---

## 5) A clean way to resolve your “theoretical vs practical” confusion

Think in **three spaces**:

1. **Parameter space curvature** (true Hessian (\nabla^2_\theta L)): usually too big for deep nets.
    
2. **Output/prediction space curvature** (second derivative w.r.t. predictions): cheap and heavily used in boosted trees (XGBoost/LightGBM). ([arXiv](https://arxiv.org/pdf/1603.02754?utm_source=chatgpt.com "XGBoost: A Scalable Tree Boosting System"))
    
3. **Structured approximations** (diag/block/HVP/Fisher): practical middle ground; used in specialized optimizers and some large-scale training setups. ([arXiv](https://arxiv.org/abs/2006.00719?utm_source=chatgpt.com "ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning"))
    

So yes: second-order ideas are absolutely used today—just not usually as “full Hessian inversion of a 100M-parameter net.”

---
