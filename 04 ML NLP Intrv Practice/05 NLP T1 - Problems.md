
PROBLEM 1 — SUBSTRING:
  Basic check:     query.lower() in corpus.lower()
  With position:   corpus.find(query)  → index or -1
  Whole word only: re.search(r'\b' + query + r'\b', corpus)
  Multiple queries: [q in corpus for q in queries]

PROBLEM 2 — TDM REPLACEMENT:
  Step 1: CountVectorizer().fit_transform(corpus + [masked])
  Step 2: cosine_similarity(masked_vec, corpus_vecs)
  Step 3: argsort → top_k most similar docs
  Step 4: Counter(candidate_words).most_common(1)[0][0]
  Upgrade: swap CountVectorizer → TfidfVectorizer (same everything else)


**The conceptual chain you must own:** 

Corpus → TDM (CountVectorizer) → Cosine Similarity → Most Similar Docs → Candidate Words → Counter → Best Replacement

That chain is the entire problem. Every line of code is just one step in that chain. If you can narrate that chain out loud without looking, you understand it.

**The one swap that matters:** `CountVectorizer` → `TfidfVectorizer` is literally one word change. But it means you understand WHY raw counts are naive. That's the kind of thing that impresses in an AI intern assessment — showing you know the limitation of the simpler approach.


# NLP Task 1 — Deep Intuition Pack

## Problem 1: Substring Search | Problem 2: TDM Word Replacement

---

## ══════════════════════════════════════

## PROBLEM 1 — Substring Search

## "Find if the provided string is a substring of a larger string"

---

### THE MENTAL MODEL FIRST

A substring is any contiguous sequence of characters that exists WITHIN a larger string.

```
Larger string:  "the cat sat on the mat"
Query:          "cat sat"       → IS a substring ✓ (contiguous, exists)
Query:          "cat mat"       → NOT a substring ✗ (not contiguous)
Query:          "CAT SAT"       → depends on whether case-sensitive
```

**The brain question you ask first:**

> "Am I searching for exact match? Or fuzzy/partial? Case sensitive or not?"

That determines which of the 4 methods below you use.

---

### Method 1 — Python `in` operator (simplest, most readable)

**Intuition:** Python's `in` is the most direct English-to-code translation. "Is 'cat' IN 'the cat sat'?" — you literally write it that way.

```python
def is_substring(query, corpus):
    return query in corpus

# Test
corpus = "the cat sat on the mat"
print(is_substring("cat sat", corpus))    # True
print(is_substring("cat mat", corpus))    # False
print(is_substring("CAT SAT", corpus))    # False — case sensitive!

# Case-insensitive version
def is_substring_ci(query, corpus):
    return query.lower() in corpus.lower()

print(is_substring_ci("CAT SAT", corpus))  # True
```

**When HackerRank gives you this:** They'll give you N queries against M corpus strings. Always lowercase both sides unless the problem says "case sensitive."

---

### Method 2 — `.find()` and `.index()` (when you need the position too)

**Intuition:** `in` only tells you yes/no. `.find()` tells you WHERE. Returns the starting index if found, -1 if not found. `.index()` is same but raises an exception instead of returning -1 — use `.find()` for safety.

```python
corpus = "the cat sat on the mat"

# find() → returns index or -1
pos = corpus.find("cat sat")
print(pos)              # 4  (starts at index 4)

pos = corpus.find("cat mat")
print(pos)              # -1  (not found)

# Safe pattern for checking + getting position
def find_substring(query, corpus):
    pos = corpus.find(query.lower())
    if pos != -1:
        return f"Found at index {pos}"
    return "Not found"

print(find_substring("cat sat", corpus))   # Found at index 4
print(find_substring("dog", corpus))       # Not found
```

---

### Method 3 — Multiple queries against one corpus (the HackerRank format)

**Intuition:** HackerRank almost never gives you one query. They give you a list. You need a loop. Build the pattern mentally: "for each query, check against corpus."

```python
corpus = "the quick brown fox jumps over the lazy dog"

queries = ["quick brown", "fox jumps", "lazy cat", "the", "OVER"]

# Version 1: simple list of results
results = [(q, q.lower() in corpus.lower()) for q in queries]
for query, found in results:
    print(f"'{query}' → {'Found' if found else 'Not found'}")

# Output:
# 'quick brown' → Found
# 'fox jumps'   → Found
# 'lazy cat'    → Not found
# 'the'         → Found
# 'OVER'        → Found  (case-insensitive)
```

---

### Method 4 — regex for pattern-based substring (when query has wildcards)

**Intuition:** Sometimes the query isn't a fixed string — it's a pattern. "Find any word starting with 'cat'" — that's regex territory. `re.search()` is your tool. It returns a match object (truthy) or None (falsy).

```python
import re

corpus = "the cat sat on the caterpillar mat"

# Find exact word 'cat' (not 'caterpillar')
pattern = r'\bcat\b'   # \b = word boundary
matches = re.findall(pattern, corpus)
print(matches)          # ['cat']

# Find any word starting with 'cat'
pattern2 = r'\bcat\w*'
matches2 = re.findall(pattern2, corpus)
print(matches2)         # ['cat', 'caterpillar']

# Simple search — does pattern exist?
if re.search(r'\bcat\b', corpus):
    print("Found 'cat' as a complete word")
```

**\b is the most important regex token for NLP.** It means "word boundary" — the invisible edge between a word and a space/punctuation. Without it, searching for "cat" also matches "concatenate", "caterpillar", "scat".

---

### FULL SOLUTION — HackerRank-style input/output format

```python
# HackerRank typically gives input via stdin
# Format: first line = corpus, next N lines = queries

import sys

def solve():
    lines = sys.stdin.read().strip().split('\n')
    corpus = lines[0].lower()
    queries = [line.lower() for line in lines[1:]]
    
    for query in queries:
        if query in corpus:
            print("True")
        else:
            print("False")

# For Jupyter testing (simulate stdin)
corpus = "the cat sat on the mat near the vat"
queries = ["cat sat", "on the", "dog ran", "vat"]

for q in queries:
    print("True" if q.lower() in corpus.lower() else "False")
# True
# True
# False
# True
```

---

### IMITATION DRILL — Problem 1

**Round 1:** Run all 4 methods above. Read outputs. **Round 2:** Close. Write the HackerRank-style loop from blank. 10 minutes. **Round 3:** Add case-insensitive handling + regex word boundary version. No looking.

---

---

## ══════════════════════════════════════

## PROBLEM 2 — TDM Word Replacement

## "Use Term Document Matrix to find the best replacement for ------ in a corpus"

## ══════════════════════════════════════

---

### THE MENTAL MODEL — What is a TDM and why does it solve this?

**Term Document Matrix (TDM)** is a table where:

- Each **row** = a document (sentence/paragraph)
- Each **column** = a unique word from the entire vocabulary
- Each **cell** = how many times that word appears in that document

```
Corpus:
  Doc1: "the cat sat on the mat"
  Doc2: "the dog sat on the rug"
  Doc3: "the cat and the dog ran"

TDM:
         and  cat  dog  mat  on  ran  rug  sat  the
  Doc1:   0    1    0    1    1    0    0    1    2
  Doc2:   0    0    1    0    1    0    1    1    2
  Doc3:   1    1    1    0    0    1    0    0    2
```

**Now the word replacement problem:** You have a sentence: "the ------ sat on the mat" You need to find which word best fills ------

**The insight:** Find other documents that are MOST SIMILAR to your masked sentence. The word that appears most in those similar documents is your best replacement.

**Similarity** = how much two document vectors point in the same direction = **cosine similarity**

---

### Step 1 — Build the TDM

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Your corpus
corpus = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat and the dog played",
    "a cat ran across the mat",
    "the dog ran on the grass"
]

# Masked sentence (the query)
masked = "the ------ sat on the mat"

# Add masked sentence to corpus for vectorization
all_docs = corpus + [masked]

# Build TDM
vectorizer = CountVectorizer()
tdm = vectorizer.fit_transform(all_docs)

# Convert to readable DataFrame
feature_names = vectorizer.get_feature_names_out()
tdm_df = pd.DataFrame(tdm.toarray(), columns=feature_names)
print(tdm_df)
```

**Brain flow:** `CountVectorizer` is your TDM builder. It scans all documents, builds a vocabulary, then counts occurrences. `fit_transform` does both: learns vocabulary (fit) + creates the matrix (transform).

---

### Step 2 — Compute Cosine Similarity

**Intuition:** Each document is a vector in high-dimensional space (one dimension per word). Two documents talking about similar things will have similar vectors — they point in a similar direction. Cosine similarity measures the ANGLE between two vectors.

- Angle = 0° → similarity = 1.0 → identical content
- Angle = 90° → similarity = 0.0 → completely different content

```
Why cosine and not Euclidean distance?
Because "the cat sat" and "the cat sat the cat sat" (repeated twice)
would be far apart in Euclidean space but cosine-similar because
the RATIO of words is the same. Cosine is length-invariant.
```

```python
from sklearn.metrics.pairwise import cosine_similarity

# tdm matrix — last row is our masked sentence
tdm_array = tdm.toarray()

# Similarity of masked sentence (last row) against all corpus docs (all rows except last)
masked_vector = tdm_array[-1].reshape(1, -1)
corpus_vectors = tdm_array[:-1]

similarities = cosine_similarity(masked_vector, corpus_vectors)[0]
print("Similarities:", similarities.round(3))

# Find most similar document
most_similar_idx = np.argmax(similarities)
print(f"Most similar doc: '{corpus[most_similar_idx]}'")
print(f"Similarity score: {similarities[most_similar_idx]:.3f}")
```

---

### Step 3 — Extract the replacement word

**Intuition:** Now you know which documents are most similar to your masked sentence. The replacement word is whatever word appears in those similar documents but is NOT already in your masked sentence and NOT the placeholder itself.

```python
# Words already in masked sentence (excluding the placeholder)
masked_words = set(masked.replace('------', '').split())
print("Words in masked:", masked_words)
# {'the', 'sat', 'on', 'mat'}

# Get top-2 most similar documents
top_k = 2
top_indices = np.argsort(similarities)[-top_k:][::-1]
print(f"Top {top_k} similar docs:", [corpus[i] for i in top_indices])

# Count candidate words from similar docs
from collections import Counter

candidate_words = []
for idx in top_indices:
    words = corpus[idx].split()
    for w in words:
        if w not in masked_words and w != '------':
            candidate_words.append(w)

word_counts = Counter(candidate_words)
print("Candidate words:", word_counts)

# Best replacement = most frequent candidate
best_replacement = word_counts.most_common(1)[0][0]
print(f"\nBest replacement for ------: '{best_replacement}'")
# → 'cat' or 'dog' depending on corpus similarity
```

---

### Step 4 — Full Clean Solution

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def find_best_replacement(corpus, masked_sentence, placeholder='------', top_k=2):
    """
    Find best word to replace placeholder in masked_sentence
    using Term Document Matrix + cosine similarity
    """
    # 1. Build TDM on corpus + masked sentence
    all_docs = corpus + [masked_sentence]
    vectorizer = CountVectorizer(stop_words=None)
    tdm = vectorizer.fit_transform(all_docs).toarray()
    
    # 2. Compute similarity of masked sentence against corpus
    masked_vec = tdm[-1].reshape(1, -1)
    corpus_vecs = tdm[:-1]
    similarities = cosine_similarity(masked_vec, corpus_vecs)[0]
    
    # 3. Get top-k most similar documents
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # 4. Collect candidate words — words in similar docs but not in masked sentence
    masked_words = set(masked_sentence.replace(placeholder, '').lower().split())
    
    candidates = []
    for idx in top_indices:
        for word in corpus[idx].lower().split():
            if word not in masked_words:
                candidates.append(word)
    
    # 5. Most frequent candidate = best replacement
    if not candidates:
        return "No replacement found"
    
    best = Counter(candidates).most_common(1)[0][0]
    return best

# ── Test it ──
corpus = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat and the dog played",
    "a cat ran across the mat",
    "the dog ran on the grass"
]

masked = "the ------ sat on the mat"
result = find_best_replacement(corpus, masked)
print(f"Masked:      '{masked}'")
print(f"Replacement: '{result}'")
# → 'cat' (most common word in similar documents)
```

---

### VARIATION — Using TF-IDF instead of raw counts (better version)

**Intuition:** Raw counts treat "the" and "cat" equally. But "the" appears in EVERY document — it's useless for distinguishing similarity. TF-IDF downweights common words automatically. Swapping `CountVectorizer` for `TfidfVectorizer` makes the similarity smarter.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Exact same code — just swap the vectorizer
vectorizer = TfidfVectorizer()
tdm = vectorizer.fit_transform(all_docs).toarray()

# Everything else stays identical
masked_vec = tdm[-1].reshape(1, -1)
corpus_vecs = tdm[:-1]
similarities = cosine_similarity(masked_vec, corpus_vecs)[0]
print("TF-IDF similarities:", similarities.round(3))
```

**Rule of thumb:**

- Use `CountVectorizer` when problem says "TDM" or "Term Document Matrix"
- Use `TfidfVectorizer` when problem says "TF-IDF"
- Both use cosine_similarity the same way — only the vectorizer line changes

---

### THE KEY IMPORTS TO MEMORIZE FOR BOTH PROBLEMS

```python
# Problem 1 — Substring
import re                                          # for regex patterns

# Problem 2 — TDM
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
```

---

### IMITATION DRILL — Problem 2

**Round 1:** Run the full solution. Print the TDM DataFrame. Stare at it. Understand what each row and column means physically.

**Round 2:** Close everything. Write `find_best_replacement()` from blank. Your only allowed reference: the function signature at the top.

**Round 3:** Swap `CountVectorizer` for `TfidfVectorizer` and compare the similarity scores. Ask yourself: did the ranking of most-similar documents change? Why or why not?

**Round 4 (hardest):** Write a version that handles multiple masked sentences in a loop, printing the best replacement for each one. Under 15 minutes, no looking.

---

### MENTAL CHEAT SHEET

```
PROBLEM 1 — SUBSTRING:
  Basic check:     query.lower() in corpus.lower()
  With position:   corpus.find(query)  → index or -1
  Whole word only: re.search(r'\b' + query + r'\b', corpus)
  Multiple queries: [q in corpus for q in queries]

PROBLEM 2 — TDM REPLACEMENT:
  Step 1: CountVectorizer().fit_transform(corpus + [masked])
  Step 2: cosine_similarity(masked_vec, corpus_vecs)
  Step 3: argsort → top_k most similar docs
  Step 4: Counter(candidate_words).most_common(1)[0][0]
  
  Upgrade: swap CountVectorizer → TfidfVectorizer (same everything else)
```