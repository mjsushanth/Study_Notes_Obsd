# C01 — JIT Retrieval and the Pre-Filter Dominant Architecture

> **The claim of this note.** A widely-shared post claims you can run RAG over 1.5 TB of legal
> documents with "no vector database" and "practically zero" infrastructure cost. Stripped of
> the marketing, there is a **real and important architecture** underneath it, and it has a
> proper name: **pre-filter dominant retrieval with an ephemeral index**.
>
> The valuable part is not the conclusion ("vector DBs are a trap"). It is the **variable that
> decides it**: what fraction of the corpus a single query actually touches. Get that number
> and the architecture chooses itself.
>
> This note deconstructs the pattern, teaches the two mechanisms that make it possible
> (Parquet statistics pruning, HTTP range reads), derives the cost crossover as a formula, and
> marks precisely where the original claims overreach.
>
> Companion: [[C02 — Precompute vs Query-Time Compute — FinSights against JIT]]

---

## 1. The source, and how to read it

| Attribute | Detail |
| :-- | :-- |
| Claim | 15 M+ Indian court judgments, ~1.5 TB source, ~2 TB processing/indexing |
| Stack | DuckDB + S3 Parquet metadata filtering, single consumer laptop |
| Mechanism | Data at rest in S3; embeddings computed **on the fly** for 3–5 PDFs per query |
| Claimed cost | "practically zero" |
| Genre | Teaser post for a whitepaper — **marketing artifact, not a design doc** |

Read it as a **hypothesis with one strong idea and three unpriced liabilities**. The idea is
sound. The liabilities are the interesting part, because they are exactly what a design review
would surface.

> **Discipline.** When a post claims an order-of-magnitude win, find the variable that was
> silently held constant. Here it is **query volume** — and it is doing all the work.

---

## 2. The inversion — the whole architecture in one diagram

Standard RAG makes **semantic similarity** the primary selector and metadata a refinement.
This design reverses the two.

```
   STANDARD RAG FUNNEL                     PRE-FILTER DOMINANT FUNNEL
   (semantic-primary)                      (metadata-primary)
   ------------------                      --------------------------

   15,000,000 docs                         15,000,000 docs
        |                                       |
   [ bulk embed EVERYTHING ]  <-- $$$$      [ metadata predicate ]  <-- ~free
   [ persist vector index  ]  <-- $$/mo      court + statute + date
        |                                       |
        v                                       v
   ANN search over 15M vectors               ~200 candidate docs
        |                                       |
        v                                       v
   top-k  ~50                               [ embed the survivors ]  <-- cents
        |                                       |    JIT, on demand
        v                                       v
   [ metadata filter ]  <-- refinement       brute-force cosine, in RAM
        |                                       |
        v                                       v
   [ rerank ] -> answer                      top-k -> answer
                                                 |
                                            [ DISCARD INDEX ]
```

Two structural differences, and they are independent choices:

- **Selector order.** Metadata first, semantics second (versus the reverse).
- **Index lifetime.** The vector index is **ephemeral** — built per query, thrown away
  (versus persistent).

> **The post's headline is imprecise.** It is not "no vector search." Vector search still
> happens over the surviving 3–5 documents. It is **no *persistent* vector index**. That
> distinction is what makes the design analysable.

### 2.1 Why legal data rewards this so heavily

The metadata in this domain is unusually **selective** and unusually **available**:

- Court, bench, judge
- Statute / section cited
- Date, year, case type
- Jurisdiction, disposition

A query like *"Supreme Court rulings on Section 498A between 2015 and 2020"* is **almost
entirely a structured predicate**. The semantic component is doing comparatively little work —
it is ranking within an already-tiny candidate set.

```
   WHERE THE RETRIEVAL SIGNAL LIVES  (knowledge-flow view)

   legal research query
        |
        +-- 85-95%  structured  ->  court, statute, date range, case type
        |                            (exact, cheap, no embedding needed)
        |
        +--  5-15%  semantic    ->  "which of these 200 is on point"
                                     (needs embeddings, but only over 200)

   contrast: open-domain Q&A
        |
        +--  0-10%  structured  ->  maybe a date filter, often nothing
        |
        +-- 90-100% semantic    ->  the index IS the retrieval system
```

**This ratio is the whole architecture.** Where metadata carries the signal, the persistent
vector index is largely redundant. Where it does not, removing the index removes retrieval.

---

## 3. Mechanism 1 — Parquet is a queryable format, not just a compressed one

The part worth learning properly, because it is a **capability**, not an opinion. Most people
treat Parquet as "smaller CSV." It is closer to a read-only column store with a built-in index.

### 3.1 Physical anatomy

```
   my_judgments.parquet   (say 1 GB)
   +--------------------------------------------------------------+
   | Row Group 0        (~128 MB)                                 |
   |   Column Chunk: court      [ dict + pages ]                  |
   |   Column Chunk: year       [ dict + pages ]                  |
   |   Column Chunk: text       [ pages ...    ]  <-- the bulk    |
   +--------------------------------------------------------------+
   | Row Group 1        (~128 MB)   ... same layout ...           |
   +--------------------------------------------------------------+
   | ...                                                          |
   +--------------------------------------------------------------+
   | FOOTER  (a few KB)                                           |
   |   schema                                                     |
   |   per row group, per column:                                 |
   |       min, max, null_count, byte_offset, size                |
   |   <-- THIS is the index                                      |
   +--------------------------------------------------------------+
   | 4-byte length | "PAR1"                                       |
   +--------------------------------------------------------------+
```

The footer holds **min/max statistics per column per row group**. That is enough to prove a
row group *cannot* contain a match without reading it.

### 3.2 What actually crosses the network

This is the sequence that makes "query 2 TB from a laptop" true rather than boastful. All of it
rides on HTTP `Range:` headers against S3.

```
   CLIENT (DuckDB / Polars)                        S3
   ------------------------                        --

   1. GET Range: bytes=-8          ------------>   last 8 bytes
      (footer length + magic)      <------------   "…PAR1"

   2. GET Range: bytes=N-M         ------------>   footer only  (~KB)
      (the metadata block)         <------------   min/max stats

   3. evaluate predicate against stats
        year BETWEEN 2015 AND 2020
        -> row groups 0,1,2,5,9  PRUNED
        -> row group 7           possible match

   4. GET Range: <offsets of      ------------>   only the column chunks
      row group 7, cols needed>   <------------   you actually projected

   TOTAL TRANSFERRED: kilobytes of metadata + megabytes of payload
   NOT: 1 GB
```

### 3.3 The three pruning layers, strongest first

| Layer | Mechanism | Eliminates | Requires |
| :-- | :-- | :-- | :-- |
| **Partition pruning** | Directory naming (`year=2019/court=SC/`) | Whole **files**, before any read | Hive-style layout at write time |
| **Row-group pruning** | Footer min/max statistics | **Chunks within** a file | Sorted / clustered data to be effective |
| **Projection pushdown** | Columnar physical layout | **Columns** you did not name | Nothing — free by format |

> **Partition pruning is a *write-time* decision that determines *read-time* cost.** You cannot
> retrofit it with a smarter query. This is the single most under-used lever in object-storage
> analytics.

**Critical caveat, usually omitted:** row-group pruning only works if the data is **clustered**
on the predicate column. If `year` is randomly scattered, every row group's `[min, max]` spans
2006–2025, nothing prunes, and you scan everything. Sorting on write is what makes the
statistics informative.

---

## 4. Mechanism 2 — What DuckDB contributes

DuckDB is not magic; it is a **vectorised, single-node OLAP engine** that happens to have
excellent object-store support. Its contributions here:

- **`httpfs`** — reads S3 objects via HTTP range requests, no download step, no local copy.
- **Predicate + projection pushdown** into the Parquet reader, automatically.
- **Larger-than-memory execution** — streaming, spilling, hash aggregation that does not
  require the working set in RAM.
- **Zero infrastructure** — an in-process library. No cluster, no daemon, no coordinator.

```
   THE "SINGLE NODE IS ENOUGH" CLAIM

   2015 assumption                     2025 reality
   ---------------                     ------------
   100 GB+ = distributed               100 GB+ = one laptop, if columnar
     Spark cluster                       DuckDB / Polars
     coordinator + workers               in-process, vectorised
     shuffle over network                NVMe + range reads
     $$$ idle cost                       $0 idle cost

   The break came from three things arriving together:
     (a) columnar formats with statistics   -> read less
     (b) vectorised execution engines       -> process faster per core
     (c) cheap NVMe + fast object storage   -> spill and stream cheaply
```

> Polars occupies the same niche and is already in the FinSights stack. **The gap is not the
> tool — it is exploiting statistics and partitioning, which is a data-layout skill, not a
> library choice.** See [[C02 — Precompute vs Query-Time Compute — FinSights against JIT]] §7.

---

## 5. The contract view — what changes at the interface

Worth drawing, because the interface reveals the real coupling. The *signature* is identical;
the **cost profile and failure modes are not**.

```
   SHARED INTERFACE
   ----------------
   retrieve(query: str, k: int) -> list[Passage]


   IMPLEMENTATION A — PERSISTENT INDEX (FinSights today)
   .....................................................
   preconditions   : entire corpus already embedded and indexed
   fixed cost      : HIGH  (one-time embed + monthly index storage)
   marginal cost   : LOW   (1 query embedding + k ANN lookups)
   latency         : LOW, predictable        (~5 s pipeline, measured)
   failure mode    : degraded ranking; still returns plausible neighbours
   scales badly in : CORPUS SIZE
   coupling        : write path and read path coupled through the index


   IMPLEMENTATION B — EPHEMERAL INDEX (JIT)
   ........................................
   preconditions   : structured metadata exists and is SELECTIVE
   fixed cost      : ~ZERO (object storage only)
   marginal cost   : HIGH  (download + parse + embed k docs, per query)
   latency         : HIGH, variable          (seconds to minutes)
   failure mode    : filter miss -> ZERO recall, no graceful degradation
   scales badly in : QUERY VOLUME
   coupling        : read path depends on metadata QUALITY, not index freshness
```

> **Same signature, opposite economics.** This is the clean case for putting retrieval behind
> an interface: the two implementations are swappable, and a **tiered** system can hold both at
> once. That is the design [[C02 — Precompute vs Query-Time Compute — FinSights against JIT]] §8
> builds toward.

---

## 6. The decision framework — derive the crossover, do not argue about it

This converts a blog opinion into a tool you can use in a design review.

### 6.1 The formula

```
   Let   N  = documents in corpus
         k  = documents touched per query
         t  = tokens per document
         p  = embedding price per token
         Q  = expected lifetime query count

   BULK  (precompute) :  C_bulk = N * t * p          (once)  + storage
   JIT   (on demand)  :  C_jit  = Q * k * t * p      (recurring)

   Crossover:   Q* = N / k          <-- t and p CANCEL

   Q < Q*  ->  JIT wins
   Q > Q*  ->  precompute wins
```

> **`t` and `p` cancel out.** The crossover does **not** depend on document length or embedding
> price. It depends only on **corpus size divided by per-query footprint**. That is a genuinely
> surprising and useful result — it means you can decide this before you know your model or
> your vendor.

### 6.2 Applied

| System | N | k | **Q\* = N/k** | Touch fraction (k/N) | Verdict |
| :-- | --: | --: | --: | --: | :-- |
| Legal JIT (the post) | 15,000,000 | ~4 | **~3,750,000** | 0.000027 % | JIT wins decisively |
| FinSights (per company) | 4,674 | ~1 | **~4,674** | 0.021 % | contested — see C02 |
| Typical enterprise KB | 50,000 | ~5 | **~10,000** | 0.010 % | JIT until real traffic |
| Open-domain chatbot | 5,000,000 | ~2,000 | **~2,500** | 0.040 % | precompute almost always |

The legal case has a touch fraction **~800× more favourable** than FinSights. That is *why*
the architecture works there. It is a property of the workload, not a universal truth.

### 6.3 The crossover, drawn

```
   cost
    ^
    |                                                    C_jit (slope = k*t*p)
    |                                                  /
    |                                                /
    |                                              /
    |                                            /
    |  C_bulk  ___________________________ _ _ /_ _ _ _ _ _ _ _  (flat after t=0)
    |         |                              /
    |         |                            /
    |         |                          /
    |         |                        /
    |         |                      /
    |    _____|                    /
    |   /                        /
    |  /  <- JIT cheaper here   /  <- precompute cheaper beyond Q*
    | /                       /
    +------------------------X-------------------------------> queries
    0                       Q* = N/k

   The post lives FAR to the left of Q*. That is not luck - it is a
   low-traffic, high-corpus workload, which is the exact shape JIT serves.
```

### 6.4 The correction the post omits — caching moves Q\*

Once a document is JIT-embedded, **cache it**. Document access is almost always power-law
distributed (landmark judgments are cited constantly; the cold tail never gets touched).

```
   NAIVE JIT              q1: embed A,B      q2: embed A,C      q3: embed A,D
   (re-pays for A)        cost: 3            cost: 3            cost: 3

   JIT + CACHE            q1: embed A,B      q2: A hit, embed C q3: A hit, embed D
   (pays once per doc)    cost: 2            cost: 1            cost: 1
                                                  |
                                                  v
                          converges to: "the hot set is precomputed,
                          the cold tail is never embedded at all"
```

> **JIT + cache under power-law access converges on the optimal architecture without anyone
> designing it.** The hot set migrates into a de facto persistent index; the cold tail costs
> nothing. This is strictly better than either pure strategy, and the post does not mention it.

---

## 7. Where the claims overreach — priced honestly

Three items. Stating them precisely is the difference between learning from a post and being
sold by one.

### 7.1 "Infrastructure cost is practically zero" — false as stated

| Line item | Reality | Note |
| :-- | :-- | :-- |
| S3 Standard, 2 TB | **~$47 / month** | at ~$0.023/GB-mo |
| GET / LIST requests | non-zero | metadata filtering is request-heavy by design |
| **Egress** | non-zero | he *downloads* PDFs per query |
| Embedding API | **recurring, per query** | the cost was moved, not removed |
| OCR / parse compute | unpriced | see §7.3 |

He converted a **large fixed cost into a small recurring one** and called the result zero. The
trade is often correct. The label is not.

> **"Zero" almost always means "moved to a line item I am not counting."** Find the line item.

### 7.2 The cost claim he makes is *fair* — concede it clearly

Bulk-embedding 1.5 TB of text is genuinely expensive. At ~4 chars/token, 1.5 TB ≈ **300–400
billion tokens**; at commodity embedding rates (~$0.12/M) that is **$36,000–48,000**. His
"thousands of dollars" is an *understatement*, not hype.

> Conceding a correct claim is what makes the other criticisms credible. The cost wall is real.

### 7.3 The ingestion question is dodged — the load-bearing gap

He implies he skipped the expensive bulk pass. But:

- He filters on **structured metadata for 15 M judgments** — someone extracted that.
- He cites **"~2 TB of processing and indexing data"** — that is a bulk artifact.

So the honest framing:

> **He avoided bulk *embedding*, not bulk *ingestion*.** A full parse/metadata pass over 15 M
> documents already happened. That is the majority of the engineering effort, and it is
> unpriced in the post.

And at query time, if those PDFs lack a text layer, he pays **OCR latency per query** —
plausibly tens of seconds for a long judgment. Either the corpus is already digital (likely for
recent Indian e-filings) or the user-facing latency is far worse than implied. **Unanswered,
and it is the single biggest risk in the design.**

---

## 8. Failure-mode analysis — the asymmetry that matters most

```
   PERSISTENT INDEX + HYBRID SEARCH           PRE-FILTER DOMINANT
   -------------------------------            -------------------

   query -> entity extraction                 query -> metadata predicate
              |                                          |
        +-----+-----+                                     |
        |           |                              (predicate wrong)
   filtered      global                                   |
   search        search                                    v
        |           |                              candidate set = {}
        +-----+-----+                                       |
              |                                             v
         merge + dedup                              ZERO RECALL
              |                                     - no fallback path
              v                                     - confident wrong answer
   filter wrong? -> global arm                      - failure is SILENT
   still returns something
   DEGRADED, not DEAD
```

| | Persistent + hybrid | Pre-filter dominant |
| :-- | :-- | :-- |
| Entity/filter extraction fails | degraded ranking | **zero recall** |
| Recovery path | global arm | none |
| Failure visibility | low quality, detectable | **silent** — looks like "no results exist" |
| Cost of redundancy | pays for a second search arm | none |

> **Pre-filter dominant architectures are brittle exactly where they are cheap.** The metadata
> predicate becomes a **single point of failure for recall**, and its failures are silent — the
> worst combination. Any production version needs a fallback arm, which erodes the cost
> advantage it was chosen for.

---

## 9. When to reach for this — and when not to

**Strong fit:**

- Rich, reliable, **selective** structured metadata exists (or is cheap to derive).
- Queries are naturally scoped by that metadata (legal, medical records, ticketing, logs, email).
- Touch fraction `k/N` is minuscule — well under ~0.01 %.
- Query volume is low, or far below `Q* = N/k`.
- Corpus is large enough that bulk embedding is a real budget line.
- Latency tolerance is generous (research tools, back-office, batch).

**Poor fit:**

- Metadata is absent, sparse, or non-selective.
- Queries are conceptual/thematic — "documents *about* regulatory risk" is not a predicate.
- High query volume, or heavy overlap in documents touched.
- Interactive latency requirements (sub-second).
- Corpus small enough that bulk embedding is trivially cheap — **then this is pure
  over-engineering.**

### 9.1 Review checklist

- [ ] What is `k/N` — the touch fraction — for a *realistic* query?
- [ ] What is `Q* = N/k`, and where does projected lifetime volume sit relative to it?
- [ ] Is the metadata **selective** enough, or merely *present*?
- [ ] Is the data **clustered/partitioned** on the predicate column? (If not, pruning is fiction.)
- [ ] What happens when the predicate is wrong — degraded, or zero recall?
- [ ] Is the failure **visible**? How would I detect a silent recall collapse?
- [ ] Has the **ingestion/parse** pass been priced, not just the embedding pass?
- [ ] Is there a **cache**? Does access follow a power law that would make it effective?
- [ ] What is the **p95 latency**, including download and parse, not just embed?
- [ ] Which cost did I *move*, and which did I actually *remove*?

---

## 10. The transferable principle

> **When you pay for compute is an architectural decision, not an implementation detail.**

Nearly every RAG tutorial assumes index-time compute and never presents it as a choice. This
case study's real contribution is making the axis visible:

```
   THE COMPUTE-TIMING SPECTRUM

   index-time                                          query-time
   <===========================================================>
   |                    |                    |                 |
   embed everything     embed hot set,       embed on demand    embed on demand
   up front             JIT the tail         + cache            no cache
   |                    |                    |                 |
   FinSights            <- the optimum       the post's         naive JIT
   (100% precompute)       lives here        implied design     (wasteful)

   fixed cost:  HIGH  ------------------------------------>  ZERO
   marginal:    LOW   ------------------------------------>  HIGH
   latency:     LOW   ------------------------------------>  HIGH
   scales in:   query volume  <---------------->  corpus size
```

The endpoints are both defensible. **The middle is usually correct and almost never chosen**,
because it requires an interface boundary that a tutorial architecture does not have.

---

## Related notes

- [[C02 — Precompute vs Query-Time Compute — FinSights against JIT]] — the contrast against my own system, with measured numbers
- [[S02h - Measurement as a Design Practice]] — the measurement discipline this note's cost claims lean on
- [[S02i - Higher-Level Design Principles from a Real Deployment]] — P1/P2 on making bad states unrepresentable
- [[S03 - Systems Walkthrough - Deploying a RAG Service to AWS]] — the persistent-index architecture in full
- [[Note 01 - Polars, DB]] — single-node columnar engines

---

*Written 2026-08-13. Source post: Medium, "RAG Without a Vector Database: JIT Retrieval over
1.5TB of Legal Documents" (anirudh.lumia820). Claims about the post's system are **UNVERIFIED**
— no code or whitepaper was available at the time of writing; treat all of its figures as
author-reported. Parquet/DuckDB mechanisms described here are format- and engine-level
behaviour, **VERIFIED** by specification and standard tooling. Cost arithmetic in §6–§7 is
derived, with inputs stated inline.*
