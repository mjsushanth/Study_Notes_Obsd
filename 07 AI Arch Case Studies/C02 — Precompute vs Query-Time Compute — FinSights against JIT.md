# C02 — Precompute vs Query-Time Compute — FinSights against JIT

> **The claim of this note.** FinSights and the JIT legal system are not competing designs. They
> are **two endpoints of one axis** — *when* you pay for embedding compute — and each fails at
> the opposite end. FinSights does not scale in **corpus size**; JIT does not scale in **query
> volume**.
>
> The genuinely important finding is that **FinSights already contains the empirical proof of
> JIT's central premise, and I filed it under the wrong heading.** The filtered-vs-open regime
> numbers in `LLMOPS_TECHNICAL_COMPLIANCE.md` measure exactly how much retrieval signal lives in
> metadata versus semantics — the variable that decides whether a persistent vector index is
> necessary at all. I recorded it as a sensitivity analysis. It was an architectural discovery.
>
> This note contrasts the two systems on measured numbers, identifies where each is genuinely
> stronger, and specifies a **tiered extension** that would let FinSights answer questions about
> 4,649 companies it never indexed.
>
> Companion: [[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]]

---

## 1. The two systems, on the record

| | **FinSights** (mine) | **JIT Legal** (the post) |
| :-- | :-- | :-- |
| Corpus | 25 companies, 2006–2025 | 15 M+ judgments |
| Unit indexed | sentence | document (JIT-chunked) |
| Indexed items | **614,647 vectors** | 0 persistent |
| Source size | Stage 1 ~35.8 MiB text | ~1.5 TB |
| Embedding | Cohere Embed v4, 1024-d | on-the-fly, model unstated |
| Index | S3 Vectors, persistent, cosine | ephemeral, in-RAM, discarded |
| Metadata filter | `cik_int`, `report_year`, `section_name`, `sic`, `sentence_pos` | court, statute, date, case type |
| Retrieval arms | **filtered + global, merged** | filtered only |
| One-time embed cost | **$2.21** (measured) | avoided (~$36–48 k, est.) |
| Ingest cost | **$0.500237** PutVectors (measured) | n/a |
| Idle cost | **$0.2938 / month** (measured) | ~$47 / month (2 TB S3, est.) |
| Per-query cost | **$0.017–$0.06+** (measured) | embed + egress + OCR, unpriced |
| Pipeline latency | **~5.2 s**, retrieval ~90 % of it | seconds to minutes |
| Reranking | tested, **removed on ablation evidence** | not mentioned |
| Eval | Self@1, Hit@5, MRR + BERTScore/BLEURT/ROUGE-L/LLM-judge | none published |

> Every FinSights figure above is **VERIFIED** from project docs. Every JIT figure is
> **author-reported or derived** — see [[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]] §7.

---

## 2. The scaling asymmetry — the one diagram to remember

```
                      CORPUS SIZE  ------------------->
                      25 co.        4,674 co.      15M docs
                        |               |              |
     Q                  |               |              |
     U    100 q  -------+---------------+--------------+-----
     E                  | FinSights     | FinSights    | JIT
     R                  | comfortable   | $550-700     | ONLY
     Y                  |               | one-time     | viable
                        |               |              | option
     V   10k q  --------+---------------+--------------+-----
     O                  | FinSights     | FinSights    | JIT starts
     L                  | ideal         | ideal        | to hurt
     U                  |               |              |
     M    1M q  --------+---------------+--------------+-----
     E                  | FinSights     | FinSights    | JIT
     |                  | ideal         | ideal        | COLLAPSES
     v                  |               |              | (re-embeds
                        |               |              |  corpus ~4x)

     FinSights' wall is ->  CORPUS SIZE   (linear embed cost, hard $ wall)
     JIT's wall is       ->  QUERY VOLUME (linear marginal cost, no ceiling)
```

Neither is "the right playbook." They are **bets on which axis grows**, and the bets are
opposite. Applying `Q* = N / k` from [[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]] §6:

| Scope | N | k | **Q\*** | Reading |
| :-- | --: | --: | --: | :-- |
| JIT legal | 15,000,000 | ~4 | **3,750,000 q** | effectively never crosses |
| FinSights, per company | 4,674 | ~1 | **4,674 q** | **crosses easily** — a demo could |
| FinSights, per sentence | 614,647 | ~30 | ~20,488 q | crosses within a semester |

> **My touch fraction is ~800× worse than the legal system's.** One query consumes one whole
> company (0.021 % of the universe); one legal query consumes four documents out of fifteen
> million (0.000027 %). That single ratio explains why JIT is obviously right for him and
> genuinely arguable for me. **It is a property of the workload, not a virtue of the engineer.**

---

## 3. The compute-timing axis, and where each system sits

```
   WHEN DO YOU PAY?

   index-time  <=================================================>  query-time

   |-----------------|------------------|------------------|-----------------|
   FinSights         tiered:            tiered:            JIT + cache       naive JIT
   100% precompute   hot precomputed    hot precomputed    (converges left)  (wasteful)
                     cold JIT+cached    cold JIT, no cache

   ^                 ^                                                        ^
   |                 |                                                        |
   where I am        where I SHOULD be (§8)                    where the post implies it is

   fixed cost   HIGH ----------------------------------------------------> ZERO
   marginal     LOW  ----------------------------------------------------> HIGH
   p95 latency  LOW  ----------------------------------------------------> HIGH
   ops surface  index freshness, reindex ------> metadata quality, cache warmth
```

**What I never considered:** that this was an axis at all. FinSights is 100 % precompute because
that is what every RAG tutorial does, not because I priced the alternative. The 25-company cap
was partly a *budget* decision — which means **the cost model silently chose my product scope.**
That is the uncomfortable and useful realisation.

---

## 4. The empirical vindication — I already measured JIT's core premise

This is the most valuable section of this note.

JIT's entire viability rests on one claim: **structured metadata carries most of the retrieval
signal, so semantic search over a pre-filtered set is sufficient.** I tested that on my own
corpus and recorded the result as a "filter regime sensitivity analysis."

### 4.1 The measurement

| Regime | Self@1 | Hit@5 | What it means |
| :-- | --: | --: | :-- |
| **Filtered** (metadata predicate applied) | **96.7 %** | **82 %** | semantics ranking within a scoped set |
| **Open** (global ANN, no predicate) | **60.0 %** | **61 %** | semantics doing retrieval unaided |
| **Delta attributable to metadata** | **+36.7 pts** | **+21 pts** | the pre-filter's contribution |

Source: `LLMOPS_TECHNICAL_COMPLIANCE.md` §§41–63, 60-anchor deterministic neighbour tests,
W=5. **VERIFIED.**

### 4.2 Why the delta is that large — and it is not a subtle reason

```
   THE BOILERPLATE PROBLEM  (measured, not theorised)

   Every 10-K contains Item 1A Risk Factors, and it is templated:

     "...may materially and adversely affect our business,
      financial condition, and results of operations..."

   Cosine similarity across DIFFERENT companies:  0.95+

        Apple 1A  ----.
        Tesla 1A  ----+----> all mutually near-identical in embedding space
        NVDA  1A  ----+
        MSFT  1A  ----'

   Consequence for GLOBAL search:
     query about Tesla supply-chain risk
        -> nearest neighbours are boilerplate from 24 OTHER companies
        -> true local context is BURIED
        -> Self@1 collapses 96.7% -> 60.0%

   Consequence for FILTERED search:
     cik_int = TSLA is applied FIRST
        -> the 24 other companies cannot compete at all
        -> boilerplate is no longer a competitor, it is excluded
```

> **On this corpus, global semantic search is not merely weaker — it is actively adversarial.**
> The embedding space is dominated by legal-template similarity, which is exactly the signal a
> financial analyst does *not* want. The metadata filter is not an optimisation. **It is the
> mechanism that makes retrieval work at all.**

### 4.3 The reframe

I wrote this up as *"validates the need for metadata-driven filtering in production RAG."* That
is true but far too modest. The stronger statement my own data supports:

> **On a corpus with high cross-document boilerplate and selective metadata, the persistent
> global vector index contributes the *minority* of retrieval signal. A pre-filter dominant
> architecture is not a cost hack — it is the better retrieval design, and I have the ablation
> to prove it.**

Same measurement. Sensitivity finding versus architectural thesis. **The difference is entirely
in how it is framed** — and that difference is what separates an ML engineer from an architect.

---

## 5. Where FinSights is genuinely ahead

Not flattery — these are real advantages the JIT design lacks, and I should defend them.

### 5.1 Graceful degradation under filter failure

```
   FinSights                                    JIT
   ---------                                    ---
   query                                        query
     |                                            |
   EntityAdapter (fuzzy)                        metadata predicate
     |                                            |
     +---- cik/year/section extracted           (extraction wrong)
     |                                            |
     v                                            v
   +-------------------+                        candidates = {}
   | filtered  | global|                              |
   |   82%     |  61%  |                              v
   +-------------------+                        ZERO RECALL, SILENT
     |                                          "no results exist"
     v                                           (indistinguishable from
   merge + dedup + distance rank                  "nothing was found")
     |
     v
   filter wrong -> global arm STILL RETURNS
   DEGRADED (61%), not DEAD (0%)
```

The global arm I might otherwise call redundant is the **fallback that makes entity-extraction
failure survivable**. JIT has no equivalent. Its cheapness *is* its brittleness.

### 5.2 Measured everything, and acted against my own preference

- **Reranking**: implemented a cross-encoder, ablated it, **removed it** when it degraded
  quality on this corpus (`enable_reranking: false`). Deleting working code on evidence is
  harder than adding it.
- **Window sensitivity**: W=2/3/5 vs 7 → Hit@5 65.0 % → 60.0 %, so ±3 sentences is the
  defensible default rather than a guess.
- **Two-track evaluation**: retrieval (Self@1, Hit@5, MRR) held separate from generation
  (BERTScore, BLEURT, ROUGE-L, LLM-judge). No single metric family carries the whole claim.
- **Cost and latency instrumented in production**: $0.017–$0.06+/query, ~5.2 s pipeline,
  retrieval ~90 % of it (~4,667 ms of ~5,165 ms).

> The post publishes **no evaluation numbers at all.** An architecture without a measured
> quality floor is a hypothesis. Mine has one. That is not a small difference — it is the
> difference between "I built a system" and "I built a system and can prove what it does."

### 5.3 Corrected my own record repeatedly

`4,674 → 25` companies, `203,076 → 614,647` vectors, `"500 MB–2.3 GB" → 64,781,290 bytes`,
`"$1/month ECR" → $0.0643/month`. Documented as corrections with provenance, not quietly
overwritten. See [[S02h - Measurement as a Design Practice]] §7–8.

---

## 6. Where FinSights is exposed

### 6.1 The global arm has never been cost-justified

I know the global arm helps **when the filter fails**. I do not know its **marginal
contribution when the filter succeeds** — and that is a separate, unmeasured question.

```
   UNMEASURED:  of the final top-k that the LLM actually cites,
                what fraction came from the GLOBAL arm on queries
                where the filter was CORRECT?

     if ~0%   -> global arm is pure insurance. Consider making it
                 conditional (fire only on low-confidence extraction).
                 Reclaim its share of the ~4,667 ms retrieve budget.

     if >20%  -> my metadata filter is LOSSIER than the 82% suggests,
                 and the honest read is that Hit@5 82% is propped up
                 by an arm I was about to call redundant.

   EITHER ANSWER IS ACTIONABLE. Not measuring it is the only bad option.
```

This is structurally the **same experiment as the reranking ablation** — which I ran, and acted
on. The precedent exists; I just did not point it at this component.

### 6.2 The 2.19 GiB monolith — solved the symptom, not the shape

`CLAUDE.md` currently encodes this rule:

> *"NEVER `pl.read_parquet()` the VECTORS table (2.19 GiB) ... Eager loading crashes the
> kernel."*

That is a correct **workaround** for a **layout** problem. `scan_parquet` + streaming avoids the
OOM; it does not avoid the scan. Per
[[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]] §3, the format offers a way to
make the problem not exist.

---

## 7. The partitioning lesson — concrete and actionable

My filter columns are *already* the right partition keys. The data is simply not laid out to
exploit them.

```
   TODAY  (monolith - every read scans past everything)
   ------------------------------------------------------
   ML_EMBED_ASSETS/EMBED_VECTORS/cohere_1024d/
     finrag_embeddings_cohere_1024d.parquet      2,293,538,065 B

     query "Apple 2023"  ->  stream past 2.19 GiB
                         ->  row-group stats USELESS if cik/year
                             are scattered across every row group


   PROPOSED  (Hive-partitioned - pruning becomes real)
   ------------------------------------------------------
   ML_EMBED_ASSETS/EMBED_VECTORS/cohere_1024d/
     report_year=2023/cik_int=320193/part-0.parquet    ~ MB
     report_year=2023/cik_int=1318605/part-0.parquet   ~ MB
     report_year=2022/cik_int=320193/part-0.parquet    ~ MB
     ...

     query "Apple 2023"  ->  partition pruning eliminates ALL
                             other directories BEFORE any read
                         ->  touch ONE small file
                         ->  no streaming gymnastics required
```

| | Monolith | Partitioned |
| :-- | :-- | :-- |
| Bytes touched, single-company query | up to 2.19 GiB streamed | single-partition MB |
| Row-group stats useful? | only if globally sorted | **yes, by construction** |
| OOM risk | mitigated by discipline | **structurally absent** |
| Cost of a wrong lazy/eager call | kernel crash | small read |
| Incremental company add | rewrite / append to monolith | **drop in a new directory** |

> **P1 from [[S02i - Higher-Level Design Principles from a Real Deployment]] applies directly:
> prefer "unrepresentable" to "unlikely."** A documented rule saying *never eagerly load this
> file* makes the crash unlikely. Partitioning so no single file is large enough to crash the
> kernel makes it **unrepresentable.** I fixed the instance; the class is still open.

The incremental-add row is the sleeper benefit — it is exactly the primitive §8 needs.

---

## 8. The extension this case study argues for — a tiered FinSights

### 8.1 The economics, in my own measured numbers

```
   Per-company embedding cost
   --------------------------
   614,787 sentences / 25 companies       =  24,591 sentences/company

   at 39.6 tok/sentence  (tiktoken estimate)
       24,591 x 39.6  =    973,804 tok  x $0.12/M  =  $0.117
   at 50.7 tok/sentence  (ACTUAL Bedrock billing observed)
       24,591 x 50.7  =  1,246,764 tok  x $0.12/M  =  $0.150

   + PutVectors ingest ($0.20/GB, 4.069 GB per 1M vectors)
       24,591 vectors -> ~0.100 GB       ->            $0.020

   ===> ~$0.17 per company, all-in, on demand
```

| Strategy | One-time | Per cold query | Reachable companies |
| :-- | --: | --: | --: |
| Today | $2.21 (paid) | $0 | **25** |
| Bulk-embed everything | **+$550–700** | $0 | 4,674 |
| **Tiered (hot 25 + JIT cold)** | **$0** | **~$0.17, once per company** | **4,674** |

> **For 17 cents and one wait, the system could answer a question about any of the 4,649
> companies I excluded.** Then it is cached and free forever. The crossover from
> [[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]] §6 says bulk-embedding only
> wins past ~4,674 cold-company queries — a threshold a portfolio demo will never approach.

Note the measurement lesson embedded above: the **tiktoken estimate understated real billing by
28 %** (39.6 → 50.7 tok/sentence). Both numbers are carried deliberately. Costing this from the
estimate alone would understate by a quarter.

### 8.2 The tier state machine

```
   query -> EntityAdapter -> cik_int
                              |
                              v
                    +---------------------+
                    | is cik in HOT set?  |
                    +---------------------+
                       |               |
                     YES              NO
                       |               |
                       v               v
              serve from S3    +------------------------+
              Vectors index    | is cik in WARM cache?  |
              ~5.2 s           +------------------------+
              $0.017-0.06         |              |
                       ^        YES             NO
                       |          |              |
                       |          v              v
                       |    serve from    +--------------------+
                       |    warm partition| COLD PATH          |
                       |    ~5.2 s        | 1. fetch Stage-1   |
                       |          |       |    sentences (S3)  |
                       |          |       | 2. embed on demand |
                       |          |       | 3. PutVectors      |
                       |          |       | 4. write partition |
                       |          |       | 5. promote -> WARM |
                       |          |       +--------------------+
                       |          |              |
                       |          |         ~$0.17, minutes
                       |          |         (see §9.1 - the blocker)
                       |          |              |
                       |          +--------------+
                       |                         |
                       +-------------------------+
                          promotion on Nth access
                          WARM -> HOT (power-law converges)
```

### 8.3 The interface that makes it possible

The tier decision must live behind **one seam**, or it leaks into every call site.

```
   CONTRACT
   --------
   class VectorSource(Protocol):
       def ensure_available(cik: int, years: list[int]) -> Availability
       def query(vec: list[float], filt: MetadataFilter, k: int) -> list[Hit]

   Availability = READY | WARMING(eta_s) | UNAVAILABLE(reason)


   IMPLEMENTATIONS
   ---------------
   S3VectorsSource      ensure_available -> READY always      (hot, today)
   JITEmbedSource       ensure_available -> WARMING(eta)      (cold, new)
   TieredVectorSource   delegates by cik membership           (the router)


   WHY 'ensure_available' IS A SEPARATE CALL
   -----------------------------------------
   The UI must be able to say "warming NVDA, ~45 s" BEFORE committing
   to a query. Folding this into query() forces either a silent
   multi-minute block or a lie. Separating readiness from retrieval
   is what makes the latency HONEST to the user.
```

> This is the interface boundary FinSights does not currently have — and its absence is exactly
> why 100 % precompute felt like the only option. **The architecture was constrained by a
> missing seam, not by a considered decision.**

---

## 9. Honest risk register for the extension

Written before implementing, because [[S02h - Measurement as a Design Practice]] §9 says state
what the experiment cannot rule out.

### 9.1 P0 — The pipeline is sequential; latency, not cost, is the blocker

```
   24,591 sentences / 96 per batch      =  256 API calls
   observed throughput                  =  ~1.75 s per batch, ONE call in flight
   -------------------------------------------------------------
   sequential wall time  =  256 x 1.75  =  448 s  =  ~7.5 MINUTES per company
```

Unacceptable as an interactive path. And note **the RPM quota is not the constraint** — 256
calls fits inside even the throttled 100 RPM. The constraint is that
`EmbeddingGenerationPipeline` deliberately keeps one call in flight (which is also why
`GlobalRateLimiter` needs no lock).

- With concurrency 10 → **~45 s**. Viable as an explicit "warming" state.
- **This makes concurrency a prerequisite, not an optimisation** — and it invalidates the
  limiter's documented no-lock assumption, so thread safety must be revisited at the same time.

> **Correction to my own earlier reasoning on this:** I framed the pending AWS quota increase
> (RPM 100 → 3,000) as what would unlock a JIT tier. It would not. **The sequential pipeline
> would waste 3,000 RPM exactly as thoroughly as it wastes 100.** Concurrency is the real gate;
> the quota only matters once more than one call is in flight.

### 9.2 P1 — Vector-space consistency across transports

Bins 1–2 came from Bedrock `cohere.embed-v4:0`; Bin 3 from Cohere native `embed-v4.0`. Verified
equivalent on the full table (**L2 norm spread 0.00e+00**). A JIT tier must pin the same
`output_dimension=1024` explicitly — v4 silently defaults to **1536-d**, and the `truncate`
vocabulary differs between transports (`"RIGHT"` on Bedrock vs `NONE|START|END` native). A
mismatch corrupts retrieval **silently**, which is the worst failure class.

### 9.3 P1 — Cold companies have no Stage-1 sentences

The tier math above assumes sentence-level Stage-1 data exists for all 4,674 companies. **It
does not.** 4,674 is the *upstream ETL universe*; Stage 1 holds 25 companies' sentences. So the
cold path is really:

```
   fetch filing -> parse -> sentence-split -> metadata-enrich -> embed
   \_________________ THIS IS THE UNPRICED PART ________________/
```

> **This is the exact same gap I criticised in the post** ([[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]] §7.3):
> *avoiding bulk embedding is not avoiding bulk ingestion.* I nearly reproduced the error in my
> own design one section after naming it. The honest scope is **"JIT-embed companies whose
> Stage-1 sentences already exist"** — and the ETL cost for the rest is **UNMEASURED**.

### 9.4 P2 — Checkpoint path collision

`CHECKPOINT_PATH` is a single global file. Concurrent JIT warm-ups would collide. Already
flagged as P1 in `EMBEDDING_PROVIDER_ABSTRACTION_DESIGN.md`; a tiered system escalates it from
annoying to a **correctness risk**. Scope it by `{provider}_{cik}_{hash(scope)}`.

---

## 10. How this changes what I say about the system

Weak framing (what I had):

> *"I used S3 Vectors because it's 99 % cheaper than managed vector DBs."*

True, but it reads as a vendor choice. Strong framing (what the analysis supports):

> *"I treated **when** to pay for embedding compute as an explicit design axis. FinSights sits at
> full precompute, which is right for a 25-company corpus queried heavily — the crossover is
> corpus-size ÷ per-query-footprint, and mine is ~4,674 cold queries. I also measured how much
> retrieval signal actually lives in metadata versus semantics: filtered retrieval hits 96.7 %
> Self@1 against 60 % for global search, because 10-K risk-factor boilerplate scores 0.95+ cosine
> across companies. So on my corpus the metadata filter is not an optimisation, it's the
> mechanism that makes retrieval work — and I keep the global arm specifically as a
> graceful-degradation path for entity-extraction failure, which a filter-only design doesn't
> have."*

The second version demonstrates four things the first does not: a **decision framework**, a
**measured tradeoff**, a **causal explanation**, and an **acknowledged failure mode**.

> **The measurement was always there. What was missing was the frame that made it an
> architectural argument instead of a table in a compliance document.**

---

## 11. Checklist — carry into the next system

- [ ] Have I priced **both** endpoints of the compute-timing axis, or defaulted to precompute?
- [ ] What is `Q* = N/k`, and where does realistic lifetime volume sit against it?
- [ ] How much retrieval signal is **metadata** vs **semantic**? Have I *ablated* it?
- [ ] Is there **cross-document boilerplate** that makes global search adversarial rather than weak?
- [ ] When the filter/extraction fails: **degraded or dead**? Is the failure **visible**?
- [ ] Is my data **partitioned** on its filter keys, or am I disciplining around a monolith?
- [ ] Did I fix the **instance** or the **class**? (Documented rule vs unrepresentable state.)
- [ ] Which components have I justified only by *presence*, not by **marginal contribution**?
- [ ] Is there a **seam** where a second implementation could be swapped in — or is the
      architecture constrained by a missing interface?
- [ ] For any "cheap" path: did I price **ingestion**, or only the step I optimised?
- [ ] Is the real blocker **cost**, **latency**, or **concurrency**? (They are not interchangeable.)

---

## Related notes

- [[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]] — the pattern, mechanisms, and crossover math
- [[S02h - Measurement as a Design Practice]] — the ablation discipline §4 and §6.1 rest on
- [[S02i - Higher-Level Design Principles from a Real Deployment]] — P1 "unrepresentable vs unlikely," applied in §7
- [[S02j - Empirical Methods and Findings Catalogue]] — the corrected-figures record
- [[S03 - Systems Walkthrough - Deploying a RAG Service to AWS]] — the deployed persistent-index system
- [[S02 - Design Learn from FinSights RAG]] — earlier design-lesson pass
- [[Note 01 - Polars, DB]] — columnar engines, single-node analytics

---

*Written 2026-08-13. FinSights figures **VERIFIED** from `finrag_ml_tg1/CLAUDE.md`,
`LLMOPS_TECHNICAL_COMPLIANCE.md`, `IMPLEMENTATION_GUIDE.md`,
`investigation_analysis/EMPIRICAL_METHODS_AND_FINDINGS.md`, and
`investigation_analysis/EMBEDDING_PROVIDER_ABSTRACTION_DESIGN.md`. Per-company cost in §8.1 and
wall-time in §9.1 are **DERIVED** from those verified inputs, with arithmetic shown. ETL cost for
un-ingested companies (§9.3) is **NOT MEASURED**. All JIT-legal figures are author-reported and
**UNVERIFIED**.*
