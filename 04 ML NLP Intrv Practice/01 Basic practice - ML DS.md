


- Basic → encode, clean, linear/logistic regression
- Intermediate → **groupby, outlier detection (IQR), feature engineering, heatmaps, classification eval**
- Advanced → **full Pipelines, GridSearchCV, cross-validation, GradientBoosting, SMOTE**

----

## QUICK REFERENCE CARD (Keep Open During Test)

```
NULLS:    numeric → .fillna(median)   |   categorical → .fillna(mode()[0])
ENCODE:   2 values → LabelEncoder     |   3+ values → get_dummies(drop_first=True)
SPLIT:    train_test_split(X, y, test_size=0.2, random_state=42)
MODEL:    classify → RandomForestClassifier(n_estimators=100, random_state=42)
          regress  → LinearRegression()
EVAL:     classify → classification_report(y_test, y_pred)
          regress  → mean_squared_error + r2_score
```
## — Modeling

**Intuition:** This is where you pick your tool. For this test, you need exactly 3 tools and when to use each:

|Task|Signal words|Use|
|---|---|---|
|Predict yes/no, category|"classify", "predict if", "spam/not spam"|RandomForestClassifier or LogisticRegression|
|Predict a number|"predict price", "estimate age", "forecast sales"|LinearRegression|
|Not sure|Default|RandomForestClassifier (safest, works most places)|


```python 

# Option 1 — Titanic (classic, perfect for classification)
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Option 2 — Iris (simplest possible, no nulls)
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris(as_frame=True)
df = iris.frame

# Option 3 — Diabetes (regression practice)
from sklearn.datasets import load_diabetes
import pandas as pd
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

--------------------------------------------------------------------------------
```
```python 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD ──────────────────────────────────────────────
df = pd.read_csv('titanic.csv')
print(df.shape)
print(df.isnull().sum())

# ── 2. CLEAN ─────────────────────────────────────────────
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)
df.dropna(inplace=True)

# ── 3. ENCODE ────────────────────────────────────────────
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# ── 4. SPLIT ─────────────────────────────────────────────
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 5. MODEL ─────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── 6. EVALUATE ──────────────────────────────────────────
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

```



