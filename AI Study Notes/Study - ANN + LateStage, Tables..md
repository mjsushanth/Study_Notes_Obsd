

**clean, hierarchical, knowledge-architecture table**.

one sheet_ that organizes everything from Level-0 corpus sharding → Level-1 ANN families → Level-2 retrieval families → Level-3 multi-vector structures → Level-4 MuVeRa refinements. 

know **exactly where each idea lives** in the retrieval stack.


---

## **LEVEL 0 – Corpus-Scale Partitioning (Sharding Layer)**

|**Component**|**What It Operates On**|**Purpose**|**Technical Mechanism**|**Why It Exists**|
|---|---|---|---|---|
|**Logical Shards**|Document collections (e.g., by company, domain, language)|Semantic partitioning|Metadata routing; domain-specific indexes|Avoids unnecessary search across irrelevant domains|
|**Hash / ID-Based Shards**|Document IDs hashed to buckets|Load balancing|Hash functions; consistent hashing|Distributed storage & compute, uniform distribution|
|**Range Shards**|Range-split documents (e.g., sorted by timestamp or numeric key)|Temporal or numeric segmentation|Range partitioning|Efficient for time-bound or numeric-range queries|
|**Broadcast/Fan-out Querying**|All shards|Parallel retrieval then merge|Parallel ANN/BM25 queries|Scale-out across machines|

---

## **LEVEL 1 – ANN Candidate Generation (Vector-Search Layer)**

### **1.1 ANN Families**

|**ANN Family**|**What It Indexes**|**How Routing Works**|**Complexity Benefit**|**Why Choose It**|
|---|---|---|---|---|
|**Tree-based (kd-tree, ball tree)**|Full document embeddings|Split space via coordinate/metric trees|Good in low-dim; bad in high-dim|Simple, but rarely used for 768D embeddings|
|**Hash-based (LSH)**|Document embeddings|Similar vectors → same hash buckets|Sublinear search via hashing|Works for cosine; good for high recall-first|
|**Partition-based (FAISS IVF / IVFPQ)**|Document embeddings + centroids|Query → nearest centroids → search only those lists|Search O(1/K) subset|Most standard ANN; robust, flexible|
|**Quantization-based (PQ, OPQ)**|Residual vectors in compressed forms|Codebooks + product quantization|Memory shrink ×16–32|Massive memory savings with tolerable error|
|**Graph-based (HNSW, NSG)**|Document embeddings|Multi-layer proximity graph walk|Very fast, high recall|State-of-the-art for search latency & recall|

### **1.2 What ANN Actually Performs**

|**ANN Stage**|**What It Chooses**|**Effect**|
|---|---|---|
|**Candidate Routing**|Which vector partitions to probe|Reduces 1M docs → 1k candidates|
|**Distance Estimation**|Approx cosine/dot product|Faster than brute-force|
|**Top-K Selection**|Rank candidate docs|Produces final candidate set|

---

## **LEVEL 2 – Retrieval Families (Scoring Models Layer)**

|**Retrieval Family**|**Representation Unit**|**Query–Doc Interaction**|**Strengths**|**Weaknesses**|
|---|---|---|---|---|
|**Sparse (BM25 / TF–IDF)**|Discrete tokens + posting lists|Term overlap (symbolic)|Exact entity match, rare terms, ultra cheap|No semantics, brittle to paraphrase|
|**Dense (Single-Vector)**|One embedding per doc|ANN dot product / cosine|Semantic recall, paraphrases, simple|Loses multi-aspect structure, compression bottleneck|
|**Multi-Vector (Late Interaction / ColBERT)**|One embedding per token|MaxSim: Σ_i max_j sim(q_i, d_j)|Fine-grained alignment, multi-aspect matching, contextual|Expensive, n×m cost, memory-heavy|
|**Cross-Encoder (Full attention)**|Entire sequences jointly encoded|Query–doc attention inside transformer|Best accuracy|Too slow for large search (O(n×m))|

---

## **LEVEL 3 – Multi-Vector Document Indexing (Token-Level Layer)**

|**Component**|**What It Stores**|**Interaction Cost**|**Purpose**|
|---|---|---|---|
|**Token Embeddings**|{d₁, d₂, …, dₙ}|O(|q|
|**Contextualized Vectors**|BERT-derived representations|Each token disambiguated by context|Solves polysemy (“bank”=river vs finance)|
|**Dimensionality Reduction**|768 → 128 dims|Lowers cost and memory|Keeps most semantic structure|

---

## **LEVEL 4 – MuVeRa Deep Optimizations (Within-Document Layer)**

### 4.1 Centroid Interaction Matching (CIM)

|**Target**|**What It Operates On**|**Why It Works**|**Goal**|
|---|---|---|---|
|**Document Centroid**|Avg of token vectors|Cheap ANN stage|Reduce 1M docs → few hundred|
|**Query Centroid**|Avg of query tokens|Captures rough intent|Route to top ANN buckets|
|**Centroid ANN**|Single vectors|Approx semantic filter|Low-latency coarse search|

---

### 4.2 Offline Document Token Clustering

|**Clustered Unit**|**When It Happens**|**Benefit**|**Effect on MaxSim**|
|---|---|---|---|
|**Document tokens**|Preprocessing time|Skip irrelevant token clusters|Reduce n in O(|

---

### 4.3 Online Query Token Clustering

|**Clustered Unit**|**When It Happens**|**Benefit**|**Effect on MaxSim**|
|---|---|---|---|
|**Query tokens**|At each query|Avoid redundant match computations|Reduce|

---

### 4.4 Combined View: MuVeRa Reduces All Three Dimensions

|**Cost Dimension**|**Naive Late Interaction**|**MuVeRa After Optimizations**|**Technique**|
|---|---|---|---|
|**# Documents scored**|1M|~100–500|Centroid ANN|
|**# Token Clusters per Doc**|n tokens|n/k clusters|Offline clustering|
|**# Query Token Groups**|m tokens|m/r groups|Online clustering|

---

# “Bird’s Eye View” Table—The Entire Retrieval Stack in One Sheet

|**Level**|**Component Type**|**Operates On**|**Primary Outcome**|**Engineering Insight**|
|---|---|---|---|---|
|**0**|Sharding|Corpus partitions|Removes irrelevant large data zones|Distributed systems & metadata routing|
|**1**|ANN|Document vectors|Select top-K candidate documents|Approximate geometry for huge corpora|
|**2**|Sparse / Dense / Late Retrieval|Document representations|Compute ranking scores|Semantics vs exact match trade-offs|
|**3**|Multi-Vector Index|Token embeddings|High-resolution doc modeling|Preserve contextual granularity|
|**4**|MuVeRa Refinements|Token groups + query groups|Make late interaction fast|Structured attention-like pruning|

---

This table gives you clean conceptual navigation:

- **Level 0** = WHERE to search
- **Level 1** = WHICH documents to score
- **Level 2** = HOW to score them
- **Level 3** = WHAT representation supports scoring
- **Level 4** = HOW to make expensive scoring feasible
