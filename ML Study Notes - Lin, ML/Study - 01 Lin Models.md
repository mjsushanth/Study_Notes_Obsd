


Nice, this is a great doc to “tie together” all the linear stuff we’ve been hinting at. Let me give you a professor-style pass over the **5.1 Linear Model** section (plus its extensions) from that page, but in our own language and with the mental models you care about. ([FirmAI][1])

---

## 1. Where linear models sit in the ecosystem

In that book, linear models are presented as the **canonical interpretable model**:

* They learn a **linear, monotone relationship** between features `x` and target `y`.
* For one instance `i`:

  `y_i = β0 + β1 x_{i1} + … + βp x_{ip} + ε_i` ([FirmAI][1])

  * `β0` = intercept
  * `βj` = feature j’s weight
  * `ε_i` = residual error

Key idea:
A linear model doesn’t just predict. It says:

> “Your outcome is a sum of feature contributions. I can show you how much each feature pushes the prediction up or down.”

That’s exactly why medicine, social sciences, etc., have lived in this world for decades: they need **effect sizes**, not just accuracy. ([FirmAI][1])

---

## 2. Geometry + objective: what the model *really* is

### 2.1 Geometry view

Stack all your data:

* `X` = `n × p` matrix (rows = examples, columns = features)
* `y` = `n`-dim vector of targets
* `β` = `p+1`-dim (including intercept)

The model prediction is:

`ŷ = X β`

Geometrically:

* Columns of `X` span some subspace of `R^n`.
* Linear regression finds the vector `ŷ` in that subspace that is **closest** to the actual `y` in Euclidean distance.

So it’s literally:

> “Project the observed outcomes `y` onto the column space of `X`.”

This projection interpretation is super important: all the “least squares” stuff is really just **orthogonal projection** in `R^n`.

---

### 2.2 Objective (OLS) and what is being minimized

Ordinary Least Squares solves:

`β_hat = argmin_β Σ_i (y_i – (β0 + Σ_j β_j x_{ij}))^2` ([FirmAI][1])

In vector notation:

`β_hat = argmin_β ||y – Xβ||^2`

So:

* The model **does not** care about absolute accuracy;
* It cares about minimizing **sum of squared residuals**.

Why squares?

* Penalizes large errors more aggressively.
* Gives you analytic solutions and nice math (normal equations).
* Under Gaussian noise assumptions, it’s also the **maximum likelihood** estimator.

---

## 3. How to interpret coefficients like a human

This book is very explicit about **interpretation templates**. ([FirmAI][1])

### 3.1 Numerical features

For a numeric feature `x_k`:

> *“If x_k increases by 1 unit, the expected value of y changes by β_k units, **holding all other features fixed**.”*

That “holding others fixed” clause is not pedantic fluff; it encodes the **ceteris paribus** assumption:

* Linear model gives you **partial effects**.
* You are walking along one axis of the feature space, not changing the others.

This is the core lens: each β_k is the **slope** of the regression hyperplane in that direction.

---

### 3.2 Binary features

Say `x_k` is 0/1:

> *“Changing x_k from 0 (reference) to 1 changes the expected y by β_k, holding all other features fixed.”* ([FirmAI][1])

Examples:

* `"holiday"` vs “not holiday” in the bike rental example.
* `"has_garden"` vs `"no_garden"` for housing prices.

---

### 3.3 Categorical features with many levels

You one-hot encode (or similar scheme):

* A feature with `ℓ` levels gets `ℓ–1` dummy columns.
* One level is the **reference** (e.g., “SUNNY” for weather).
  ([FirmAI][1])

Each dummy coefficient means:

> *“Compared to the reference category, this category shifts y by β_k, holding other features fixed.”*

Again, purely **relative to a reference baseline**.

---

### 3.4 Intercept `β0`

Interpretation in the book:

> “Given all numeric features are zero and categorical features at reference levels, expected outcome is β0.” ([FirmAI][1])

In practice:

* Often meaningless unless you standardize features.
* If features are z-scored, β0 ≈ predicted outcome for an “average” sample.

---

### 3.5 R² and adjusted R²

They bring in R² as:

`R² = 1 – SSE / SST`

Where:

* `SSE = Σ (y_i – ŷ_i)²` = residual sum of squares
* `SST = Σ (y_i – ȳ)²` = total sum of squares ([FirmAI][1])

Interpretation:

> Fraction of variance in y explained by the model’s linear fit.

Problem: R² **always increases** when you add features, even junk. So they recommend **adjusted R²**:

`R²_adj = R² – (1 – R²) * p / (n – p – 1)` ([FirmAI][1])

This penalizes adding many features with little marginal value.

Key takeaway:

* High R²/adjusted R² → model explains a lot of variance → interpreting β’s is meaningful.
* Very low R² → any nicely worded interpretation of coefficients is mostly storytelling.

---

## 4. Assumptions: when the math and interpretation are valid

The book lists 6 classical assumptions and what they buy you. ([FirmAI][1])

### 4.1 Linearity

Assumption:
`E[y | X] = β0 + Σ β_j x_j` (true conditional mean is linear in features).

* Strength: easy interpretation, additive effects.
* Limitation: real world often has nonlinearities and interactions.

Fixes:

* Add interaction terms (e.g., `x1 * x2`).
* Use basis expansions / splines to model nonlinear relationships.

Mental model: the **true surface** may be curved; you’re fitting a plane. If plane fits badly, β’s are still “slopes”, but they’re summarizing a curved surface ⇒ can mislead.

---

### 4.2 Normality of errors

Assumption: residuals `ε_i` are Gaussian.

* Not needed for unbiasedness of β.
* Needed for the **validity of t-tests, p-values, confidence intervals.** ([FirmAI][1])

If violated strongly:

* You can still use the model as a **predictor**.
* But classical inference (significance of β’s) becomes shaky.

---

### 4.3 Homoscedasticity (constant variance)

Assumption: variance of `ε_i` is same across feature space. ([FirmAI][1])

In practice:

* Predicting house prices: residuals for small apartments vs. mansions likely have very different spreads.
* Violation ⇒ OLS still unbiased but no longer **efficient**; confidence intervals wrong.

Fixes:
Robust standard errors, weighted least squares, or transform target.

---

### 4.4 Independence

Assumption: each row is independent (no time-series or grouped correlations). ([FirmAI][1])

* Violated with repeated measures (e.g., multiple visits per patient).
* Then you need mixed-effects models, GEEs, etc.

---

### 4.5 Fixed features (no measurement error in X)

Assumption: features are “given” without noise.

* Very rarely true, but convenient.
* If you really care, you enter the world of **errors-in-variables** models.

Most practical workflows just accept this as an approximation.

---

### 4.6 No multicollinearity

Assumption: features are not highly correlated (no near-linear combinations).

* If two columns of X are almost identical, you can’t uniquely assign effect to each.
* Coefficients become unstable, standard errors blow up. ([FirmAI][1])

This is the classical “do not interpret β’s blindly when features are highly collinear” warning.

---

## 5. Interpretability tricks from the doc

The chapter has some nice, simple devices for **turning β’s into human language and visuals**. ([FirmAI][1])

### 5.1 Interpretation templates

They explicitly propose text templates:

* Numeric feature:

  > “An increase of x_k by one unit increases the expectation for y by β_k units, given all other features stay the same.”

* Categorical feature:

  > “Changing x_k from reference level to this category increases the expectation for y by β_k, given all other features stay the same.”

Once baked into a reporting pipeline, you get **automatic natural language explanations** from the model.

---

### 5.2 Visual parameter interpretation (weight plots)

They suggest weight plots:

* y-axis: feature names
* x-axis: β estimate
* horizontal line segments: confidence intervals around each β
  ([FirmAI][1])

What this gives you:

* Instant view of direction (pos/neg), magnitude.
* Whether CI crosses 0 → roughly, whether the effect is distinguishable from 0 under assumptions.

In your head: it’s like a **forest plot** for linear regression.

---

## 6. Algorithmic flow: how you’d “run” such a model in code

Conceptually, the linear model pipeline from this doc is:

1. **Prepare X, y**

   * Maybe standardize numeric features.
   * Encode categorical features with one-hot (drop one level as reference).

2. **Fit OLS**

   * Solve `β_hat = argmin ||y - Xβ||²` (via closed form or gradient/ALS).
   * Store β_hat, residuals, estimated variance, etc.

3. **Diagnostics**

   * Compute R², adjusted R².
   * Check residual plots → normality, heteroscedasticity, outliers.

4. **Global interpretation**

   * Look at β’s and CIs.
   * Weight plot, table like the bike rentals example (feature, β, std err). ([FirmAI][1])

5. **Local interpretation**

   * For a given instance i:

     * Decompose `ŷ_i` into additive contributions: `β0` + Σ `β_j x_{ij}`.
     * You can literally show: “+110.7 from temperature, –1901.5 from rainy weather, …”

This is why linear models fit perfectly into the **“glass-box model”** slot in the interpretability book:
no need for LIME/SHAP here – the model **is** the explanation.

---

## 7. How this connects to XGBoost / tree ensembles / RuleFit

Just to connect threads in your head:

* Linear models:
  * Single hyperplane; **interpretable β’s**; very rigid inductive bias.

* XGBoost:
  * Many trees, piecewise constant function; highly flexible; strong performance; internal parameters **not directly interpretable**.

* RuleFit:
  * Uses trees/XGBoost to generate **rule features**, then fits a **sparse linear model** on top.
  * Back to linear interpretability, but in a feature space made of rules.

So this LIMO linear-model chapter is the **base language** that RuleFit and all “interpretable surrogates” speak. If you’re comfortable with:

* what β means,
* what “holding others fixed” means,
* what assumptions are needed to interpret β’s as partial effects,

then you can reason comfortably about more exotic hybrids.

---

