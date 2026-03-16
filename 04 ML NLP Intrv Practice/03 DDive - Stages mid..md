
# Stages 3 → 4 → 5 Deep Dive

## SETUP — Load + Inject Synthetic Categorical Columns

The diabetes dataset is ALL numeric — nothing to encode. So we **inject fake categoricals** purely for encoding practice. This is intentional. You're training the hand, not the dataset.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

# Load
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame  
# Columns: age, sex, bmi, bp, s1-s6, target (continuous — regression task)

# ── Inject synthetic categoricals for encoding practice ──
np.random.seed(42)
n = len(df)

df['risk_level']  = np.random.choice(['Low', 'Medium', 'High'], n)         # 3 ordinal categories
df['region']      = np.random.choice(['North','South','East','West'], n)    # 4 nominal categories
df['on_insulin']  = np.random.choice(['Yes', 'No'], n)                     # 2 binary categories
df['smoker']      = np.random.choice([True, False], n)                     # boolean

print(df.shape)       # (442, 15)
print(df.dtypes)
print(df.head(3))
```

**Brain flow:** The real test will give you a messy dataset with mixed types. By injecting categoricals here, you're drilling the real scenario synthetically. Your hands learn the pattern. The dataset doesn't matter. The pattern does.

---

## ─────────────────────────────────────────

## STAGE 3 — ENCODING (5 Methods, Real Intuition)

## ─────────────────────────────────────────

### The Core Question Your Brain Must Answer First:

> _"Does this column have ORDER? Does this column have MEANING as a number?"_

That single question determines which encoder you use. Every time.

---

### Method 1 — LabelEncoder (binary or target column only)

**Intuition:** You have Yes/No or True/False. There are only 2 states. Converting them to 0/1 is mathematically honest because there's no middle ground to misrepresent.

```python
from sklearn.preprocessing import LabelEncoder

df_le = df.copy()  # always work on a copy

le = LabelEncoder()

# Binary string column
df_le['on_insulin'] = le.fit_transform(df_le['on_insulin'])   # Yes→1, No→0

# Boolean column — same thing
df_le['smoker'] = le.fit_transform(df_le['smoker'])           # True→1, False→0

# Verify
print(df_le[['on_insulin', 'smoker']].value_counts())
print(le.classes_)   # shows you what mapped to what — ALWAYS check this
```

**Why NOT for 3+ categories:** If you LabelEncoder on ['Low','Medium','High','Critical'], you get [0,1,2,3]. The model now thinks Critical(3) is 3x more than Low(0). That's a lie unless there's actual ordinal meaning.

**Rewrite test:** Close this. Write LabelEncoder on `on_insulin` and `smoker` from blank.

---

### Method 2 — pd.get_dummies (nominal, 3+ categories, no natural order)

**Intuition:** Region (North/South/East/West) has no order. North isn't "more" than South. So you create a separate binary column for each category. The model treats each as independent evidence.

```python
df_dummies = df.copy()

# Before
print("Before:", df_dummies.shape)          # (442, 15)
print(df_dummies['region'].value_counts())

# One-hot encode
df_dummies = pd.get_dummies(df_dummies, columns=['region'], drop_first=True)

# After
print("After:", df_dummies.shape)           # (442, 18) — added 3 cols, dropped 1 (drop_first)
print([c for c in df_dummies.columns if 'region' in c])
# → ['region_South', 'region_East', 'region_West']  (North is the reference — it's implied)
```

**Why drop_first=True:** If you have North/South/East/West and encode all 4: `not South AND not East AND not West` already tells the model it's North. Keeping the 4th column is redundant AND causes multicollinearity (two columns tell the model the same thing — confuses it). `drop_first=True` removes exactly one column and avoids this.

**What if you have MANY categories (high cardinality)?**

```python
# Check cardinality first
print(df['region'].nunique())  # 4 — fine
# If it's 50+ categories, get_dummies explodes your feature space
# Use target encoding or frequency encoding instead (see Method 5)
```

---

### Method 3 — Ordinal Encoding via map() (when order genuinely matters)

**Intuition:** Low < Medium < High. That order is real information. You want the model to know High is more than Medium which is more than Low. Manual mapping preserves that semantics.

```python
df_ord = df.copy()

# Method A — dict map (explicit, readable, PREFERRED)
ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}
df_ord['risk_level'] = df_ord['risk_level'].map(ordinal_map)

print(df_ord['risk_level'].value_counts())
print(df_ord['risk_level'].dtype)   # int64

# Verify no NaNs crept in (map returns NaN for values not in dict)
print(df_ord['risk_level'].isnull().sum())   # must be 0
```

```python
# Method B — OrdinalEncoder from sklearn (when you have many ordinal columns)
from sklearn.preprocessing import OrdinalEncoder

df_oe = df.copy()
oe = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df_oe[['risk_level']] = oe.fit_transform(df_oe[['risk_level']])
# Note: input must be 2D → double brackets
```

**The trap everyone hits with .map():** If your data has a typo like 'high' (lowercase), the map returns NaN silently. Always `.str.strip().str.capitalize()` before mapping on real data:

```python
df_ord['risk_level'] = df_ord['risk_level'].str.strip().str.capitalize()
df_ord['risk_level'] = df_ord['risk_level'].map({'Low':0,'Medium':1,'High':2})
```

---

### Method 4 — Encode Multiple Columns At Once (test time efficiency)

**Intuition:** The test might give you 5 columns needing encoding. Don't write 5 separate blocks. Write one clean loop.

```python
df_multi = df.copy()

# Separate columns by encoding type
binary_cols  = ['on_insulin', 'smoker']
nominal_cols = ['region']
ordinal_cols = {'risk_level': {'Low':0, 'Medium':1, 'High':2}}

# Encode all binary
le = LabelEncoder()
for col in binary_cols:
    df_multi[col] = le.fit_transform(df_multi[col].astype(str))

# Encode all nominal
df_multi = pd.get_dummies(df_multi, columns=nominal_cols, drop_first=True)

# Encode all ordinal
for col, mapping in ordinal_cols.items():
    df_multi[col] = df_multi[col].map(mapping)

print(df_multi.shape)
print(df_multi.dtypes)
```

**This is the pattern you use when the question has mixed messy columns.** Sort them by type mentally first, then apply each method in one pass.

---

### Method 5 — Feature Engineering (Intermediate/Advanced Playbook)

**Intuition:** Sometimes raw columns aren't informative enough. You create NEW features from existing ones. This is where you show you understand the data, not just the syntax.

```python
df_feat = df.copy()

# Ratio feature — BMI relative to blood pressure (physiological interaction)
df_feat['bmi_bp_ratio'] = df_feat['bmi'] / (df_feat['bp'] + 1e-5)

# Binning a continuous column into categories
df_feat['bmi_category'] = pd.cut(
    df_feat['bmi'],
    bins=[-np.inf, -0.5, 0, 0.5, np.inf],
    labels=['Very Low', 'Low', 'Normal', 'High']
)

# Polynomial feature (interaction term)
df_feat['age_bmi'] = df_feat['age'] * df_feat['bmi']

# Log transform (stabilize skewed distributions)
df_feat['log_s5'] = np.log1p(df_feat['s5'] - df_feat['s5'].min() + 1)

print(df_feat[['bmi_bp_ratio', 'bmi_category', 'age_bmi', 'log_s5']].head())
```

**Why feature engineering matters on the test:** If they ask "improve the model's performance," this is your answer — not just tuning hyperparameters. Creating interaction features shows ML maturity.

---

### Method 6 — Outlier Detection (Intermediate Playbook)

**Intuition:** An outlier is a data point that lies far from the rest. Like one patient with BMI of 800 — that's either a measurement error or a very extreme case. Either way it can destroy your model's learning by pulling the regression line toward it.

```python
df_out = df.copy()

# ── Method A: IQR (Interquartile Range) — the standard ──
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

print(f"Before outlier removal: {df_out.shape}")
df_out = remove_outliers_iqr(df_out, 'bmi')
df_out = remove_outliers_iqr(df_out, 'bp')
print(f"After outlier removal: {df_out.shape}")

# ── Method B: Z-score (when data is normally distributed) ──
from scipy import stats
z_scores = np.abs(stats.zscore(df[['bmi', 'bp', 's1']]))
df_zscore = df[(z_scores < 3).all(axis=1)]
print(f"Z-score filtered: {df_zscore.shape}")
```

**IQR vs Z-score:**

- IQR: robust, doesn't assume normal distribution, PREFERRED for most cases
- Z-score: assumes normality, faster math, use when you know data is Gaussian

---

## ─────────────────────────────────────────

## STAGE 4 — SPLIT (4 Methods)

## ─────────────────────────────────────────

### Setup: Final clean df before splitting

```python
# Use df_multi (has all encoding done)
# Drop any remaining object columns (safety net)
df_final = df_multi.copy()
df_final = df_final.select_dtypes(exclude=['object', 'category'])

# Confirm all numeric
print(df_final.dtypes.value_counts())
print(df_final.isnull().sum().sum())   # must be 0

X = df_final.drop('target', axis=1)
y = df_final['target']
print(f"X: {X.shape}, y: {y.shape}")
```

---

### Method 1 — Basic train_test_split (your default)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80/20 split
    random_state=42      # reproducibility — ALWAYS
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
```

**Intuition:** 80/20 is the industry default. With 442 rows, 80% gives ~354 training samples — enough. 20% gives ~88 test samples — enough to evaluate without wasting data.

---

### Method 2 — Stratified split (when classes are imbalanced)

**Intuition:** Imagine your dataset has 90% "No diabetes" and 10% "Yes diabetes." A random split might accidentally put 95% No in train and only 5% in test. Stratified split guarantees both train and test have the SAME class ratio as the original.

```python
# Create a binary target for demonstration
y_binary = (y > y.median()).astype(int)   # 1 if above median, 0 if below
print(y_binary.value_counts())             # roughly 50/50 here

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary    # ← this is the only difference
)

# Verify class ratios are preserved
print("Train ratio:", y_train_s.value_counts(normalize=True).values)
print("Test ratio: ", y_test_s.value_counts(normalize=True).values)
# Both should show ~50/50
```

**When to use:** Any classification task. Just add `stratify=y`. Costs nothing. Prevents silent data bugs.

---

### Method 3 — Cross-Validation (Advanced Playbook)

**Intuition:** Train/test split is like grading a student on ONE exam. Cross-validation grades them on 5 different exams and averages the score. You get a much more reliable estimate of true performance.

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print(f"CV R² scores: {scores.round(3)}")
print(f"Mean R²: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Intuition on the output:** Mean = how good the model actually is. Std = how stable/consistent it is across different slices. High std means the model is sensitive to which data it sees — unstable.

---

### Method 4 — Scaling (often needed after splitting)

**Intuition:** Your features live on different scales. `age` might range -0.1 to 0.1 (already normalized in diabetes). But in a raw dataset, Age is 20-80, Income is 30000-200000. A model treating distance (like KNN) will think Income is 10000x more important than Age. Scaling fixes this by putting everything on the same scale.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: mean=0, std=1 — use for most ML models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform on train
X_test_scaled  = scaler.transform(X_test)         # ONLY transform on test (never fit)

# MinMaxScaler: range [0,1] — use for neural networks
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm  = mm_scaler.transform(X_test)
```

**THE MOST IMPORTANT RULE IN SCALING:** `fit_transform` on TRAIN only. `transform` on TEST only. If you fit the scaler on test data too, you're leaking future information into your model. This is called **data leakage** — it makes your test accuracy look good but fails in production.

---

## ─────────────────────────────────────────

## STAGE 5 — MODELING (5 Methods)

## ─────────────────────────────────────────

### The Decision Tree in Your Head:

```
Is target continuous?  →  Regression  →  LinearRegression / RandomForestRegressor
Is target categorical? →  Classification → LogisticRegression / RandomForestClassifier / GradientBoosting
Not sure?             →  RandomForest (works for both, just change the class name)
```

Diabetes `target` = continuous disease progression score → **Regression task**

---

### Method 1 — Linear Regression (baseline, always run first)

**Intuition:** Draw the best straight line through your data. Fast, interpretable, tells you if the problem is even linearly solvable. Always run this first as your baseline. If RF beats this by a lot, the data has non-linear patterns.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mse  = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred_lr)

print(f"Linear Regression → RMSE: {rmse:.2f} | R²: {r2:.4f}")

# Bonus: see which features matter most
coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': lr.coef_})
print(coef_df.sort_values('coefficient', ascending=False))
```

---

### Method 2 — Random Forest (your go-to for most tasks)

**Intuition:** 100 decision trees each look at a random subset of features and data. They all vote. Majority wins. The ensemble is dramatically more stable and accurate than any single tree. It handles non-linearity, outliers, and mixed feature types gracefully.

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# n_jobs=-1 uses all CPU cores — always add this
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)
print(f"Random Forest     → RMSE: {rmse_rf:.2f} | R²: {r2_rf:.4f}")

# Feature importances — shows which columns actually matter
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feat_imp.head(10))
```

---

### Method 3 — Gradient Boosting (Advanced Playbook, often best performer)

**Intuition:** Instead of 100 independent trees voting, each new tree CORRECTS the mistakes of the previous tree. Trees are built sequentially, focused on what the model got wrong. Usually outperforms Random Forest — at the cost of being slower and harder to tune.

```python
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,   # how much each tree corrects — smaller = more conservative
    max_depth=3,         # shallow trees prevent overfitting
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb   = r2_score(y_test, y_pred_gb)
print(f"Gradient Boosting → RMSE: {rmse_gb:.2f} | R²: {r2_gb:.4f}")
```

---

### Method 4 — Hyperparameter Tuning with GridSearchCV (Advanced Playbook)

**Intuition:** You're optimizing a recipe. Instead of randomly guessing spice amounts, you systematically try every combination and pick the best one. GridSearchCV does exactly this — it trains the model with every combination of your parameters and cross-validates each.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 5, 10],
    'min_samples_split': [2, 5]
}

rf_base = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,                   # 5-fold cross-validation per combination
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

# Use best model to predict
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Test R²: {r2_score(y_test, y_pred_best):.4f}")
```

---

### Method 5 — Full sklearn Pipeline (Advanced Playbook — most impressive)

**Intuition:** Instead of separate steps you can accidentally apply in the wrong order, a Pipeline chains everything together. Preprocessing → Scaling → Model. No data leakage possible. One `.fit()`, one `.predict()`. Clean and production-grade.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestRegressor(n_estimators=100, random_state=42))
])

# Cross-validate the entire pipeline
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f"Pipeline CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Final fit and evaluate
pipe.fit(X_train, y_train)
y_pred_pipe = pipe.predict(X_test)
print(f"Pipeline Test R²: {r2_score(y_test, y_pred_pipe):.4f}")
```

**Why Pipeline is the right answer on any advanced question:** It prevents data leakage automatically. The scaler `fit`s only on the training fold, never bleeds into test. This is what separates someone who understands ML from someone who just knows the syntax.

---

## COMPARISON TABLE — Run This to See All Methods Side by Side

```python
results = {
    'Linear Regression': (rmse, r2),
    'Random Forest':     (rmse_rf, r2_rf),
    'Gradient Boosting': (rmse_gb, r2_gb),
}

print(f"\n{'Model':<25} {'RMSE':>10} {'R²':>10}")
print("─" * 47)
for name, (r, r2) in results.items():
    print(f"{name:<25} {r:>10.2f} {r2:>10.4f}")
```

---

## IMITATION DRILL FOR STAGES 3→4→5

Three rounds. Each harder.

**Round 1 (Guided):** Run every block above. Read the output. Understand what changed.

**Round 2 (Blind encoding):** New notebook. Load diabetes. Inject the 4 synthetic columns yourself. Encode ALL of them correctly using the right method for each. Split. Fit RandomForest. Print R². No looking.

**Round 3 (Full pipeline blind):** New notebook. Load diabetes. Inject categoricals. Encode. Feature engineer ONE new column. Split with stratify. Fit GradientBoosting. GridSearch with 2 params. Print best params and test R². Under 20 minutes.

If you can do Round 3 in 20 minutes, you are ready for anything Section 1 throws at you.