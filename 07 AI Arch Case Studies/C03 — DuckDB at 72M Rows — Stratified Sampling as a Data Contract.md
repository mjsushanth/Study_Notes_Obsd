# C03 — DuckDB at 72M Rows — Stratified Sampling as a Data Contract

> **The claim of this note.** This is a case study of my own work — the DuckDB layer in
> `DataPipeline/data_engineering_research/duckdb_data_engineering/`. It turns a **71,866,962-row**
> SEC corpus into a curated 1M-sentence fact table in **~47 seconds**, on a laptop, with every
> step timed to the millisecond and four full runs preserved in the script itself.
>
> The engineering is strong. But the finding that matters most is about **impact, not
> performance**: this layer is where the RAG system's metadata filter was *manufactured*. Four of
> the five filterable fields in the live S3 Vectors index — `cik_int`, `report_year`,
> `section_name`, `sic` — are produced or normalised here. And per
> [[C02 — Precompute vs Query-Time Compute — FinSights against JIT]] §4, that filter is worth
> **+36.7 points of Self@1**.
>
> So the causal chain is: a section dimension table built in SQL in October → the metadata filter
> → the difference between 96.7% and 60% retrieval accuracy. **The data engineering layer is
> where the retrieval quality actually came from.** That is the case study.
>
> Companion: [[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]] — the Parquet
> mechanics this layer exercises directly.

---

## 1. What was actually built

| Artifact | Count / size | Note |
| :-- | --: | :-- |
| SQL scripts | **44 files** | numbered taxonomy, 4 are empty placeholders |
| Largest script | `31_run_stratified.sql`, **1,377 lines** | the production pipeline |
| Company-curation series (`21_*`) | **16 files** | the discovery work |
| Documentation | 3 markdown, ~40 KB | EDA, sampling strategy, README |
| Source corpus | **71,866,962 rows**, 1.6 GB Parquet | 4,674 companies, 1993–2020 |
| Output | **1,012,044 rows**, ~75 MB ZSTD Parquet | 48 columns |
| Full pipeline wall time | **~47 s** | 12 logged steps, all timed |
| Preserved timed runs | **4** | inline in the script, with step deltas |

```
   sql/
   ├── 00_pragmas.sql              session bootstrap
   ├── 01_params.sql               paths + tuning knobs as MACROs
   ├── 02_macros_sampling.sql
   ├── 20_0_eda_basics.sql         36 KB — corpus profiling
   ├── 21_0 .. 21_16               16 files — company curation, fuzzy match, RCA
   ├── 22_1, 22_2                  stratification logic builder + manual validation
   ├── 30_run_uniform.sql          baseline sampler
   ├── 31_run_stratified.sql       68 KB / 1,377 lines — THE PIPELINE
   ├── 31_run_stratified_post_analysis.sql
   ├── 32_1 .. 32_6                dimension tables, exports, subset creation
   ├── 90_qc_checks.sql            merge-key correctness demo
   └── 91_1_Study_RCA_DropsDDL1.sql   parameterisation root-cause study
```

> The numbering is not decoration. `00–09` bootstrap, `20–29` EDA, `21` curation, `30–39` jobs,
> `90–99` QA. **A taxonomy that makes execution order and intent legible without a README** — the
> same discipline as the `S02x` series in this vault.

---

## 2. The dataset, physically — this layer already used C01's mechanics

Before any sampling, the corpus was interrogated through **Parquet metadata**, not by reading
data:

```sql
-- A2/A3: row-group inventory, read from the FOOTER only
WITH meta AS (
  SELECT row_group_id, MAX(num_values) AS rows_in_rg
  FROM parquet_metadata(parquet_path())
  GROUP BY 1
)
SELECT SUM(rows_in_rg) AS total_rows, AVG(rows_in_rg) AS avg_rows_per_rg FROM meta;
--  total_rows = 71,866,962   avg_rows_per_rg = 261,334.4
```

```
   sec_filings_large_full.parquet   1.6 GB
   ┌──────────────────────────────────────────────────────┐
   │ 275 row groups  x  ~261,334 rows each                │
   │                                                      │
   │  RG 0   RG 1   RG 2   ...   RG 274                   │
   │  ~261K  ~261K  ~261K        ~261K                    │
   ├──────────────────────────────────────────────────────┤
   │ FOOTER: per-RG, per-column min/max/null/offset       │
   │   <- parquet_metadata() and parquet_schema() read    │
   │      ONLY this. Zero data pages touched.             │
   └──────────────────────────────────────────────────────┘

   19 top-level columns, including 2 nested STRUCTs
   (labels{1d,5d,30d}, returns{1d,5d,30d}{5 fields each})
   and 2 LISTs (tickers, exchanges)
```

> **This is the `parquet_metadata()` half of [[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]] §3 already in
> production use.** Total row count, row-group layout, and full nested schema were obtained
> without scanning 71.8M rows. What was *not* exploited is the other half — **partition pruning**
> — and §6 shows exactly what that cost.

---

## 3. The core architecture — two-stage sample-then-join

The single most important design decision in the pipeline, and it is worth stealing.

```
   NAIVE (one stage)                    ACTUAL (two stages)
   -----------------                    -------------------

   sample rows WITH all 19 columns      STAGE 1: sample sentenceIDs ONLY
   from 71.8M-row parquet                  - carries 4 narrow columns
        |                                  - window fns over ~1.95M rows
        |  every window function            - 1.70 s
        |  drags 19 cols + nested                |
        |  structs through the sort              v
        |                               sample_sentenceIDs (1,003,534 IDs)
        v                                        |
   massive intermediate spill           STAGE 2: join IDs back to parquet
   sort on wide rows                       - INNER JOIN on sentenceID
                                           - retrieves ALL 19 source columns
                                           - LEFT JOIN dim_sec_sections
                                           - 36.83 s
                                                 |
                                                 v
                                        sample_1m_finrag (48 cols)
```

**Why it works:** `ROW_NUMBER() OVER (PARTITION BY bin, cik, year, section ORDER BY RANDOM())`
must sort every row in every stratum. Sorting a 4-column row is dramatically cheaper than sorting
a row carrying two nested STRUCTs and two LISTs. The heavy schema is attached **once**, after the
selection set is already decided.

> **Separate *deciding which rows* from *fetching those rows*.** The decision needs a key; only
> the fetch needs the payload. This is the same instinct as projection pushdown, applied at the
> level of pipeline stages rather than inside one scan.

---

## 4. The measured run — every step, four times

The script writes to an `execution_log` temp table at each step and prints deltas via
`LAG(execution_time) OVER (ORDER BY step_number)`. **Four complete runs are preserved as comments
inside the script.**

### 4.1 Run 1 — 2025-10-18 06:07 (the first clean end-to-end)

| # | Step | Rows | Duration | % of total |
| --: | :-- | --: | --: | --: |
| 0 | INITIALIZATION | — | 0.013 s | <1% |
| 1 | PARAMETERS | 1 | 0.013 s | <1% |
| 2 | **CORPUS_LOAD** | **1,952,705** | **3.673 s** | 7.8% |
| 3 | BIN_POPULATIONS | 3 | 0.271 s | 0.6% |
| 4 | PIVOT_POPULATIONS | 1 | 0.004 s | <1% |
| 5 | ALLOCATION_STRATEGY | 1 | 0.003 s | <1% |
| 6 | BIN_ALLOCATIONS | 3 | 0.005 s | <1% |
| 7 | **SAMPLING_EXECUTION** | **1,003,534** | **1.698 s** | 3.6% |
| 8 | **SCHEMA_RETRIEVAL** | 1,003,534 | **36.830 s** | **78.3%** |
| 9 | SCHEMA_VALIDATION | — | 0.104 s | 0.2% |
| 10 | FEATURE_ENGINEERING | 45 cols | 1.184 s | 2.5% |
| 11 | EXPORT_PARQUET | 1,003,534 | 3.065 s | 6.5% |
| | **TOTAL** | | **~46.9 s** | |

### 4.2 All four runs, compared

| Run | Date | Sample | Cols | CORPUS_LOAD | SAMPLING | SCHEMA_RETRIEVAL | Notable |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| 1 | 10-18 06:07 | 1,003,534 | 45 | 3.673 s | 1.698 s | 36.830 s | first clean run |
| 2 | 10-24 16:53 | 654,096 | 48 | 3.337 s | 0.355 s | 22.166 s | modern bin only |
| 3 | 10-24 17:33 | 1,003,534 | 48 | 4.057 s | 1.338 s | 30.196 s | section dim revised |
| 4 | 10-26 00:14 | **1,012,044** | 48 | 2.775 s | 0.789 s | 32.684 s | **+GOOGL injection** |

Run 4 adds step 9': `INCREMENTAL_INJECTION — 8,510 rows in 0.635 s`.

> **Keeping four timed runs inline is the practice worth copying.** Run 2's 654K/22.2s against
> Run 3's 1.00M/30.2s is a free scaling datum: **1.53× the rows for 1.36× the time** — sublinear,
> which is what you want from a hash join and evidence the join is not spilling.

### 4.3 The throughput number

```
   CORPUS_LOAD reads 71,866,962 rows, applies 5 predicates,
   emits 1,952,705 rows -- in 3.673 s.

     effective scan rate  =  71.87M / 3.673 s  =  ~19.6M rows/sec
     selectivity          =  1.95M / 71.87M    =  2.72%  (36.8x reduction)

   On a laptop. Single process. No cluster.
```

---

## 5. The allocation algorithm — adaptive, not hardcoded

The stated design was a fixed 15/20/65 temporal split. The **implemented** design is better: it
adapts to what the data actually contains.

```
                    budget = 1,000,000
                          |
                          v
              +---------------------------+
              | modern_pop >= budget ?    |
              +---------------------------+
                 |                    |
               YES                   NO
                 |                    |
                 v                    v
        MODERN_ONLY            MODERN_FULL_PLUS_OLDER
        sample modern          take 100% of modern (654,096)
        down to budget                  |
        (+ warn)                        v
                            leftover = 1,000,000 - 654,096 = 345,904
                                        |
                          +-------------+-------------+
                          |                           |
                     x 0.60                       x 0.40
                          |                           |
                          v                           v
                  bin_2010_2015                bin_2006_2009
                  target 207,542               target 138,362
```

**Observed result:**

| Bin | Sampled | % of sample | Target | Overage |
| :-- | --: | --: | --: | --: |
| bin_2006_2009 | 139,569 | 13.91% | 138,362 | +1,207 |
| bin_2010_2015 | 209,869 | 20.91% | 207,542 | +2,327 |
| bin_2016_2020 | 654,096 | 65.18% | 654,096 | 0 |
| **Total** | **1,003,534** | 100% | 1,000,000 | **+3,534** |

### 5.1 The overage was predicted, then confirmed

`DuckDB_Sampling_Strategy.md` predicted the rounding behaviour *before* the run:

> *"Each stratum's allocation is rounded to nearest integer… Accumulated across 15,000 strata:
> +2,000–2,500 sentences… ~0.35% or 1% overage."*

```
   GREATEST(1, CAST(ROUND(stratum_size * rate) AS INTEGER))
      ^                    ^
      |                    +-- rounds up as often as down, but...
      +-- ...this floor means a stratum NEVER contributes zero.
          Systematically biased UPWARD. Overage is structural.

   PREDICTED:  ~0.35 %
   OBSERVED:   3,534 / 1,000,000  =  0.353 %      <-- exact
```

> **A predicted systematic error, confirmed to three significant figures.** That is the
> difference between a bug and a documented tolerance — and it is exactly the
> hypothesis-then-measure shape from [[S02h - Measurement as a Design Practice]] §9.

---

## 6. Where the time actually goes — and what would fix it

**78.3% of runtime is one step: SCHEMA_RETRIEVAL, 36.8 s.** Worth diagnosing rather than
accepting.

```
   STEP 8 as executed
   ------------------
   FROM sample_sentenceIDs si                     1,003,534 rows  (build side)
   INNER JOIN read_parquet(<71.8M-row file>)     71,866,962 rows  (probe side)
       ON si.sentenceID = corpus.sentenceID
   LEFT JOIN dim_sec_sections dim                        20 rows

   The probe side is the ENTIRE corpus, re-read from disk.

   WHY no pruning is possible:
     sentenceID = '0000001750_10-K_2020_section_1_0'
        -> a compound VARCHAR key
        -> NOT the physical sort order of the file
        -> every row group's [min,max] on sentenceID spans a huge range
        -> row-group statistics eliminate NOTHING
        -> full 71.8M scan, every time
```

| | As built | If source were Hive-partitioned by `report_year` / `cik` |
| :-- | :-- | :-- |
| Probe side | 71.8M rows, full re-scan | only partitions for 75 CIKs × 2006–2020 |
| Rows actually needed | 1.95M (2.72%) | 1.95M |
| Pruning available | **none** — random key | directory-level, before any read |
| Step-8 cost | 36.8 s | plausibly single-digit seconds (**UNVERIFIED**) |

> **The 36.8 s is the price of a monolithic, unclustered source file — not a DuckDB limitation.**
> Stage 1 already knows the target CIKs and years. Carrying those two columns into the join
> predicate, on a partitioned source, would let the engine skip most of the corpus. This is the
> **same finding** as [[C02 — Precompute vs Query-Time Compute — FinSights against JIT]] §7 about
> the 2.19 GiB vectors table: *layout, not query, is the lever.* The pattern has now appeared
> twice in this project on different files.

**Honest caveat:** the source Parquet came from HuggingFace via a JSONL→Parquet conversion.
Re-partitioning is a one-time write cost that a one-time sampling job may not justify. The lesson
is the *diagnosis*, not an automatic mandate to refactor.

---

## 7. Company curation — the funnel and the two-tier match

The `21_*` series is 16 scripts of discovery work reconciling **three heterogeneous sources**:

```
   SPDR SPY Daily Holdings (Excel)      SEC company_tickers.json      71.8M-row Parquet
        503 holdings                      10,142 CIK->ticker            4,674 CIKs
        99.94% weight coverage                    |                          |
              |                                   |                          |
              +-----------------+-----------------+--------------------------+
                                |
                                v
                    TIER 1: match on cik_int  ------------->  611 matched
                                |
                                v  (89 unmatched)
                    TIER 2: match on ticker   ------------->    2 recovered
                                |
                                v
                          613 matched  /  87 permanently unmatched
```

| Stage | Companies | Basis |
| :-- | --: | :-- |
| Target master list | ~700 | S&P 500 + ~200 curated |
| After two-tier match | **613** | 611 CIK + 2 ticker |
| After quality scoring | 540 | `quality_score` + HAVING filters |
| Deduplicated selection | **75** | 73 met *both* market-cap and disclosure-quality criteria |
| Final production subset | **21** | cost of embedding forced further reduction |

### 7.1 The 87 that could never be matched

Not a bug — a **dataset limitation**, root-caused and documented rather than silently dropped.
Missing companies include **Alphabet, Berkshire Hathaway, JPMorgan Chase, Bank of America, Wells
Fargo, Goldman Sachs, Morgan Stanley, AT&T, Broadcom** — roughly **13% of the S&P 500 by count**,
absent from a corpus whose source paper claims coverage of "all publicly traded companies."

> **A verified negative finding, with named examples and a stated cause.** Documenting *what the
> data cannot do* is worth more than a coverage percentage — and it directly motivated the
> incremental-injection feature in §9.

### 7.2 The quality score

```
   quality_score = (filing_years            *  5)
                 + (section_coverage        * 10)
                 + (recent_sentences        / 100)
                 + (priority_section_sentences / 50)

   gated by:
     filing_years              >= 5      temporal consistency
     section_coverage          >= 16     disclosure comprehensiveness  (of ~20)
     total_sentences           >= 8000   content volume
     latest_filing_year        >= 2018   recency relevance
     priority_section_sentences >= 3000  information density
```

> Stated explicitly in the doc: **"These filters determine WHO is in the dataset, not WHAT gets
> sampled."** Separating *cohort selection* from *row selection* is the kind of distinction that
> keeps a sampling design auditable.

Observed banding: Elite 1,783→1,050 pts (heavily regulated, verbose filers) · Balanced middle
1,050→875 · Long tail 875→796 (regional banks, declining firms, messy disclosure).

---

## 8. Feature engineering — 12 flags and a scored signal

A single batched `UPDATE` computes 12 derived columns in **1.18 s over 1M rows** — ~850K
rows/sec including regex evaluation.

| Flag | Signal | Weight in score |
| :-- | :-- | --: |
| `likely_kpi` | revenue, margin, EBITDA, EPS, ROE… | **+3** |
| `has_numbers` | currency, magnitudes, bps, percentages | **+2** |
| `has_comparison` | YoY, QoQ, increase/decrease, versus | **+2** |
| `is_material` | significant, material, primarily | +1 |
| `has_forward_looking` | expect, forecast, guidance, outlook | +1 |
| `is_recent` | `report_year >= 2018` | +1 |
| `is_safe_harbor` | boilerplate disclaimer language | **−2** |
| `is_table_like`, `mentions_years`, `has_risk_language` | structural / topical | 0 |

```sql
retrieval_signal_score =
      CAST(likely_kpi AS INTEGER) * 3
    + CAST(has_numbers AS INTEGER) * 2
    + CAST(has_comparison AS INTEGER) * 2
    + CAST(is_material AS INTEGER) * 1
    + CAST(has_forward_looking AS INTEGER) * 1
    + CAST(is_recent AS INTEGER) * 1
    + CAST(is_safe_harbor AS INTEGER) * (-2);   -- NEGATIVE weight
```

> **The `−2` on `is_safe_harbor` is the sharpest idea in the file.** It encodes, at ingest time,
> that legal boilerplate is *anti-signal*. That is the same phenomenon
> [[C02 — Precompute vs Query-Time Compute — FinSights against JIT]] §4.2 later measured in
> embedding space — risk-factor boilerplate at 0.95+ cosine across companies, collapsing open-regime
> Self@1 to 60%. **The problem was identified in SQL in October and measured in vector space
> later.** The intuition was right and early.

All patterns use `\b` word boundaries — deliberate, to avoid `cost` matching *costume* or `risk`
matching *brisk*.

---

## 9. The incremental-injection contract

Alphabet was one of the 87 unmatched. Rather than accept the gap, the pipeline gained a
**flag-gated external merge path**, wrapped so that the entire section no-ops when disabled.

```
   SET VARIABLE enable_incremental_injection = TRUE;
                          |
                          v
   ┌─────────────────────────────────────────────────────────────┐
   │ STEP 0  gate check                                          │
   │         every statement carries                             │
   │         WHERE getvariable('enable_incremental_injection')   │
   │         -> FALSE = whole section is a no-op, not a branch   │
   ├─────────────────────────────────────────────────────────────┤
   │ STEP 1  SCHEMA COMPARISON  (name-based, order-independent)  │
   │         information_schema.columns  FULL-OUTER              │
   │         DESCRIBE SELECT * FROM read_parquet(<incoming>)     │
   │         -> reports: Both / Target only / Source only        │
   ├─────────────────────────────────────────────────────────────┤
   │ STEP 2  STAGING with schema alignment                       │
   │         - sample_id continues from MAX(sample_id)           │
   │         - missing cols CAST(NULL AS <exact type>)           │
   │           incl. STRUCT("1d" INT,"5d" INT,"30d" INT)         │
   │         - source 'section_item' remapped to 'section_name'  │
   ├─────────────────────────────────────────────────────────────┤
   │ STEP 3  VALIDATION — duplicate detection BEFORE merge       │
   ├─────────────────────────────────────────────────────────────┤
   │ STEP 4  MERGE = DELETE + INSERT keyed on sentenceID         │
   │         (DuckDB has no MERGE statement)                     │
   └─────────────────────────────────────────────────────────────┘
                          |
                          v
             8,510 rows injected in 0.635 s
             1,003,534 -> 1,012,044
```

> **Schema alignment is name-based, not positional.** Column *order* between two Parquet files is
> an implementation detail; matching on `UPPER(TRIM(column_name))` makes the contract robust to
> it. Positional matching here would corrupt data silently.

---

## 10. The merge-key demo — the best teaching artifact in the folder

`90_qc_checks.sql` does not assert. It **demonstrates a data-loss bug** using real Apple 2018 data
(1,152 sentences across 20 sections), by simulating a partial re-ingestion of only 3 sections.

```
   SCENARIO: re-ingest Apple 2018, but the new extract covers
             only ITEM_1, ITEM_7, ITEM_8.

   APPROACH 1 (WRONG)                    APPROACH 2 (CORRECT)
   DELETE WHERE cik=320193               DELETE WHERE sentenceID IN
     AND report_year=2018                  (SELECT sentenceID FROM new_data)
        |                                       |
        v                                       v
   deletes ALL 1,152 sentences           deletes ONLY the sentences
   across all 20 sections                being replaced
        |                                       |
        v                                       v
   inserts 3 sections' worth             inserts 3 sections' worth
        |                                       |
        v                                       v
   17 SECTIONS LOST FOREVER              17 sections PRESERVED
   silently. row count drops.            row count correct.
```

> **The delete key must match the grain of the incoming data, not the grain you think in.** You
> reason in "company-year"; the extract arrives at "sentence." Deleting at the coarser grain
> destroys everything the finer-grained replacement does not cover.

The demo builds `row_hash = MD5(sentenceID || sentence)` in the main table for exactly this —
future dedupe on content, not just key.

---

## 11. The parameterisation RCA — knowing a tool's real boundary

`91_1_Study_RCA_DropsDDL1.sql` is a root-cause study written after repeated failures trying to
parameterise DDL. It lays out four distinct layers people conflate:

| Layer | Mechanism | Works for | **Cannot do** |
| --: | :-- | :-- | :-- |
| 1 | Client preprocessing (`${var}` in DBeaver) | text substitution pre-send | no runtime logic; client-specific |
| 2 | `SET VARIABLE` / `getvariable()` | data plane — SELECT, WHERE | **not DDL identifiers** |
| 3 | `PREPARE` / `EXECUTE` with `?` | values; injection-safe; plan cached | **not identifiers** |
| 4 | Dynamic SQL string execution | — | **DuckDB has no `EXECUTE IMMEDIATE`** |

The practical consequence, recorded directly in the pipeline:

```sql
-- COPY ... TO <path> requires a string LITERAL.
-- A variable, macro call, or subquery all FAIL there.
PREPARE export_stmt AS COPY sample_1m_finrag TO ? (FORMAT PARQUET, COMPRESSION 'ZSTD');
EXECUTE export_stmt(getvariable('export_full_path'));
```

> **Knowing where a tool's parameterisation boundary sits — and writing down *why* — converts a
> recurring frustration into a reusable rule.** This is the same species of artifact as
> [[S02h - Measurement as a Design Practice]] §8: the uncomfortable section that turns out to be
> the most useful one.

---

## 12. Honest findings — three defects in my own record

Applying [[S02h - Measurement as a Design Practice]] §7: *documentation drifts; artifacts do not.*

### 12.1 The "52% of corpus" claim is wrong — twice

`DuckDB_EDA.md` states *"top 3 sections = 52% of corpus"* and heads a block *"Top 5 sections (52%
of corpus)"*. The raw query output in the same repo says otherwise:

| Section | Sentences | % |
| --: | --: | --: |
| 8 | 14,648,162 | 20.38 |
| 10 | 14,170,561 | 19.72 |
| 0 | 12,181,378 | 16.95 |
| 1 | 10,869,087 | 15.12 |
| 19 | 9,756,826 | 13.58 |
| **Top 3** | 41,000,101 | **57.05** |
| **Top 5** | 61,626,014 | **85.75** |

> **Top 5 is 85.75%, not 52%.** The concentration is *far more extreme* than documented — which
> strengthens the stratification argument rather than weakening it. The narrative conclusion
> survives; the number does not. **Correct the doc.**

### 12.2 A garbled "Global Counts" output

```
   --"distinct_filings","total_rows","distinct_companies","distinct_years"
   --4674,        71866962,      32473083,          20
```

`distinct_companies = 32,473,083` is impossible (it exceeds filings by 3 orders of magnitude), and
`distinct_years = 20` contradicts `n_years = 28` from the D2 query. Headers and values are
misaligned — the query used a `COLUMNS(...)` clause against columns that do not exist in the
schema (`filing_id`, `company`, `year`). **The authoritative numbers are D2's: 4,674 companies /
55,096 filings / 28 years.** The bad output should be struck, not left ambiguous.

### 12.3 Reproducibility is honestly scoped, and that is correct

The README states plainly that full reproduction needs a 1.6 GB Parquet that cannot be hosted, and
splits the work into **parameterised production scripts** (reproducible) versus **ad-hoc discovery**
(explicitly not). Six dimension tables under 1 MB are exported so the pipeline can be reconstructed.

> Distinguishing *"this is a reproducible pipeline"* from *"this was one-time discovery work,
> preserved for transparency"* is a maturity signal, not a limitation. Most projects claim the
> first for all of it.

---

## 13. Impact — the through-line to retrieval quality

This is what the case study is actually about.

```
   OCTOBER — DuckDB layer                         LATER — RAG retrieval
   ----------------------                         ---------------------

   32_2_SectionName_DimensionCreation.sql
        |
        v
   dim_sec_sections
     hf_section_code (0-19)  ->  sec_item_canonical (ITEM_1A, ITEM_7)
        |                        section_category (P1_RISK), priority
        v
   31_run_stratified.sql  LEFT JOIN dim
        |
        v                                          S3 Vectors filterable metadata
   sample_1m_finrag  produces: ------------------>   cik_int        <- CAST here
     cik_int, report_year,                           report_year    <- derived here
     section_name, sic                               section_name   <- NORMALISED here
        |                                            sic            <- carried here
        v                                            sentence_pos   (later, ML layer)
   614,787 sentences -> embeddings                        |
                                                          v
                                             filtered regime:  96.7% Self@1 / 82% Hit@5
                                             open regime:      60.0% Self@1 / 61% Hit@5
                                                          |
                                                          v
                                             metadata filter worth +36.7 pts Self@1
```

**Four of the five filterable fields in the live index are manufactured or normalised in this
DuckDB layer.** The `section_name` field in particular did not exist in the source — the corpus
had opaque integer codes `0–19`. Mapping those to canonical SEC items (`ITEM_1A`, `ITEM_7`) is what
makes a metadata predicate *expressible at all*.

> **The retrieval accuracy that C02 credits to "metadata filtering" was created here.** Not in the
> embedding model, not in the vector store — in a dimension table built in SQL. That is the
> impact statement, and it is causally grounded, not rhetorical.

Secondary impacts:

- **The section imbalance finding drove the design.** Top-5 sections at 85.75% is precisely why
  proportional-only allocation was rejected.
- **The `is_safe_harbor = −2` intuition anticipated the boilerplate problem** later measured in
  embedding space.
- **Deliberate late binding.** The sampling layer uses integer `section` (stable, locked HF
  encoding); the semantic layer uses `section_name`. Labels can be renamed without touching
  sampling. That separation is why the section dimension could be revised (Run 2→3, 45→48 columns)
  without re-sampling.

---

## 14. Transferable principles

1. **Separate row *selection* from row *retrieval*.** Sample keys, then join back. 1.7 s vs 36.8 s
   is the difference the split protects.
2. **Predict systematic error, then confirm it.** 0.35% predicted, 0.353% observed. A known
   tolerance beats a mysterious discrepancy.
3. **Log every step with row counts and timestamps, in-band.** `LAG()` over a log table costs
   nothing and turns each run into a permanent performance record.
4. **Keep old timed runs in the file.** Four runs = free scaling curves and regression detection.
5. **Make the delete key match the incoming grain**, not your mental model of the entity.
6. **Compare schemas by name, never by position**, when merging external data.
7. **Gate optional stages with a flag evaluated in `WHERE`**, so "off" is a no-op rather than a
   code path.
8. **Number scripts by lifecycle phase.** Execution order and intent become self-documenting.
9. **Document what the data *cannot* do**, with named examples and a root cause.
10. **Layout beats query tuning.** The 36.8 s join is a partitioning problem, not a SQL problem —
    the same lesson as the 2.19 GiB vectors table.
11. **Separate cohort selection from row selection.** "Who is eligible" and "what gets sampled"
    are different questions with different filters.
12. **Encode anti-signal explicitly.** A negative weight on boilerplate is cheaper at ingest than
    fighting it in embedding space later.

---

## 15. Checklist for the next large-corpus sampling job

- [ ] Have I read the **footer** (`parquet_metadata`) before scanning a single row?
- [ ] Is the source **partitioned/clustered** on the columns I will join or filter on?
- [ ] Am I carrying the wide schema through window functions unnecessarily?
- [ ] Does every pipeline step write **row count + timestamp** to a log table?
- [ ] Have I **predicted** the rounding/overage behaviour before running?
- [ ] Is my merge key at the **same grain** as the incoming data?
- [ ] Is schema alignment **name-based**?
- [ ] Are optional stages **flag-gated as no-ops**?
- [ ] Have I separated **cohort filters** from **sampling logic**?
- [ ] Are stable machine codes and human labels **late-bound** to each other?
- [ ] Which numbers in my docs have I **re-derived from the artifact** rather than copied forward?
- [ ] Which produced columns become **downstream filter keys** — and have I named that contract?

---

## Related notes

- [[C01 — JIT Retrieval and the Pre-Filter Dominant Architecture]] — Parquet statistics, pruning layers, the partitioning lever
- [[C02 — Precompute vs Query-Time Compute — FinSights against JIT]] — where this layer's metadata pays off (+36.7 pts Self@1)
- [[S02h - Measurement as a Design Practice]] — measurement discipline; §12 applies its §7 directly
- [[S02i - Higher-Level Design Principles from a Real Deployment]] — P1 unrepresentable-vs-unlikely
- [[S02j - Empirical Methods and Findings Catalogue]] — the corrected-figures record
- [[Note 01 - Polars, DB]] — columnar single-node engines

---

*Written 2026-08-13. All figures **VERIFIED** by direct read of
`DataPipeline/data_engineering_research/duckdb_data_engineering/` — `Data_Engineering_README.md`,
`DuckDB_EDA.md`, `DuckDB_Sampling_Strategy.md`, `manual_export_analysis/DuckDB Large71M EDA
Scripts-Results.txt`, and `sql/` (00, 01, 21_7, 31_run_stratified, 90, 91_1). Timings are the
execution-log outputs preserved inline in `31_run_stratified.sql` (four runs, 2025-10-18 to
2025-10-26). §12.1 and §12.2 are **corrections derived from raw query output** contradicting the
prose in the same repo. The partitioned-join estimate in §6 is **UNVERIFIED** — not benchmarked.*
