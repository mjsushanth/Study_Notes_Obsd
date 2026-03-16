
### Classification 

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

python

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
# max_iter=1000 prevents convergence warnings — always add this
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Regression (predicting a number)

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


### Classification evaluation

```python
from sklearn.metrics import classification_report, accuracy_score

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

- **Precision**: Of everything I labeled as positive, what % was actually positive? (Am I crying wolf?)
- **Recall**: Of all actual positives, what % did I catch? (Am I missing things?)
- **F1**: Harmonic mean of both. The one number that summarizes both.
- **Support**: How many samples in that class

### Regression evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

**Reading the numbers:** 
RMSE is in the same units as your target (if predicting salary in $, RMSE is in $). R² is between 0-1, closer to 1 means better fit.

----


## VARIATIONS 


**Twist 1:** Multiple categorical columns to encode → Just loop or call `get_dummies` with a list: 
`pd.get_dummies(df, columns=['col1','col2','col3'])`


**Twist 2:** They ask for specific metric only (just accuracy, or just F1)

```python
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average='weighted'))
```

**Twist 3:** They give you a regression task (predict house price) → Swap `RandomForestClassifier` → `RandomForestRegressor` or `LinearRegression` → Swap `classification_report` → `mean_squared_error` + `r2_score`



**Twist 4:** They give you test data separately (no train/test split needed)

```python
# They give you train.csv AND test.csv
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Clean both the same way, fit on train, predict on test
model.fit(X_train, y_train)
predictions = model.predict(test_processed)
```



**Twist 5:** They want you to handle class imbalance

```python
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
```