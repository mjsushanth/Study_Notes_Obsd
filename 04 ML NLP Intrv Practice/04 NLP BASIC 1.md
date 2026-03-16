
must know cold:

```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')

# 1. Load
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1','v2']]
df.columns = ['label', 'text']

# 2. Preprocess
stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def clean(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([lem.lemmatize(t) for t in tokens 
                     if t.isalpha() and t not in stop_words])

df['clean'] = df['text'].apply(clean)

# 3. Vectorize + Split + Model + Evaluate
X = TfidfVectorizer(max_features=3000).fit_transform(df['clean'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```