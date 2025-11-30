
## ANN Fundamentals + Clustering/Sharding ??

Dense/late models give us vectors. ANN and sharding are the machinery for **searching those vectors sublinearly**. 

### Exact kNN vs Approximate NN: the need for approximation

If you have:
- N document vectors in R^d, a query vector q,

exact top-k search would involve:
- Computing similarity(q, d_i) for all i = 1..N
- Sorting or partially selecting top-k

This is O(N d) per query. For N = 10^7, that’s too slow. ANN (Approximate Nearest Neighbor) indexes accept:

> I’ll give up _perfect_ top-k to get something extremely close, but much faster.


### Major ANN families: mental models
 
should know the structural types.

1. **Tree-based (kd-tree, ball tree)**  
    Space is partitioned recursively along coordinates or in metric balls.  
    Works well in low dimensions (d ~ 10–30).  
    In high dimensions, the curse of dimensionality kills pruning; you end up touching most leaves.

2. **Hash-based (LSH)**  
    Random projections or specialized hash functions map similar vectors into the same buckets with high probability.  
    Query time: compute hashes for q, search only relevant buckets.  
    Useful for some metrics (e.g., Jaccard, cosine via random hyperplanes).

3. **Partition-based / quantization-based (FAISS IVF, IVF-PQ, etc.)**  
    Use **clustering** (k-means) over document vectors to get coarse centroids.  
    For each document vector d, assign it to one or a few centroids; store residuals if using quantization.  
    At query time: compute distances from query to all centroids, select top L centroids, only search documents in those lists.

4. **Graph-based (HNSW, NSG)**  
    Build a proximity graph where each vector is a node connected to its neighbors.  
    Query: start from some entry point(s), walk greedily towards higher-similarity neighbors.  
    This tends to be state-of-the-art in many setups: very fast, high recall, dynamic updates.


S3 Vectors, Milvus, Pinecone, etc., usually implement some combination of graph and partitioning under the hood.


### How clustering enters ANN: coarse quantization as routing

Clustering in ANN (e.g. FAISS IVF) is not about semantics for its own sake; it’s about **routing queries to small subsets** of vectors.

Workflow:

- Run k-means over all document embeddings to get K centroids `{c_1,...,c_K}`.
- For each document embedding d, assign it to some centroid c_k; store d in list L_k.

- At query time:
    - Compute q’s distance to all centroids.
    - Pick the top L centroids (where q is likely to find neighbors).
    - Only search within the union of lists for those L centroids.

This yields a complexity like:
- O(K d) to score q against centroids (K is small, e.g., 4k).
- O((N/K) * L * d) to search a fraction of the dataset (just a few lists).

This is **clustering as a top-level directory for the vector space**. It’s akin to:
- “These documents live on shard 7 because they’re about topic X; we’ll search only shard 7 and 12 for this query.”

The key concept: **coarse quantization**.  
You approximate each vector by its centroid index and a residual; this shrinks the search space and sometimes the storage cost (with further quantization like PQ).


### Sharding: splitting data into physically separate pieces

Sharding is a broader concept: you split your data into multiple shards (machines, indexes) by some scheme.

Two major flavors:

1. **Logical / semantic sharding**  
    You partition by some meaningful dimension:
    - Company → one shard per ticker.
    - Language → English vs French indexes.
    - Domain → news vs academic vs product docs.
    
    Query routing: use metadata to decide which shard(s) to hit.

1. **Hash or range-based sharding**  
    You partition by hashing doc ID or using an index key range.  
    This is more about load balancing and capacity than semantics.

In vector search system, you might have:
- Multiple **vector indexes** across machines, each holding a subset of document embeddings.
- The top-level routing: either broadcast a query to all shards (fan-out) and merge results, or route based on some metadata or coarse centroid.

So:
- **Clustering (within an index)**: route q to a few centroids / partitions to reduce search within that index.
- **Sharding (across indexes)**: route q to a subset of machines or logical partitions.

Both are forms of “don’t look everywhere,” but at different levels of the hierarchy.


### One clean mental diagram

- Level 0: Entire corpus split across **shards** (machines).
- Level 1: Within a shard, documents are partitioned via **ANN structure** (IVF centroids, HNSW graph).
- Level 2: Within each selected document set, documents are represented as **multi-vectors** (token embeddings).
- Level 3: Within each document’s token set, MuVeRa uses **offline clustering** to skip irrelevant token clusters per query.
- Level 4: Within the query tokens, MuVeRa uses **online clustering** to reuse similar token matches.

Sparse BM25 can also exist in parallel at Level 1 as a separate, cheap candidate generator.