---
title: Databricks Lakehouse - Curriculum and Unit Map
type: curriculum-index
created: 2026-08-13
target: "Databricks / Delta Lake, cloud-agnostic. Medallion + CDC + ACID upserts at volume."
pace: "~1 hr per unit, batchable. Units are self-contained; do 1 or do 4 in a sitting."
tags:
  - study/databricks
  - study/data-engineering
  - domain/lakehouse
---

# 00 - INDEX - Curriculum and Unit Map

Companion notes: [[01 - Translation Table - Innova to Lakehouse]]

**Read [[01 - Translation Table - Innova to Lakehouse]] before Unit A1.** It is the single
highest-leverage note in this folder and it is interview material on its own.

---

## The premise

You are not learning data engineering. You built a four-tier warehouse with CDC replay, SCD2,
158 `MERGE` upserts and per-procedure run auditing at Innova ([[00-audit-summary]]). You are
learning **a different vocabulary and one genuinely new mental model** — distributed execution.

Three real gaps, stated honestly:

1. **Distributed compute.** Oracle ADW hid the machine from you. Spark does not. Everything that
   breaks at 100M rows/day is a shuffle, a skew or a file-size problem.
2. **The lakehouse storage layer.** You got ACID for free from Oracle. On object storage it is a
   transaction log, and you need to be able to explain the mechanism, not just the guarantee.
3. **Git and CI/CD for data.** The Innova audit records this outright: no Git repository, no build
   server, no automated test harness. This is your weakest interview surface. Unit B21 exists
   entirely for it.

## Ground rules

- **Units, not days.** Batch them however your week goes. Ticking a unit means the deliverable
  exists, not that you read it.
- **Every unit ends in writing.** ~10 min putting it in your own words in this folder. That is the
  part that survives three weeks.
- **Translate, never memorize.** Each unit names the Innova thing it maps to. If you cannot say
  "this is my `SP_LOAD_Fact_JobOrder` but ___", you have not finished the unit.
- **No claiming Databricks production experience.** You have Oracle warehouse experience and
  Databricks *proficiency*. That distinction survives a follow-up question; the lie does not.

## Environment - verified 2026-08-13

| Need | Tool | Why |
|---|---|---|
| Spark, streaming, Lakeflow pipelines, Unity Catalog | **Databricks Free Edition**, browser | Community Edition **retired 2026-01-01**. Free Edition has serverless compute and default storage, and needs no cloud account or credit card. Sign-up: `databricks.com/learn/free-edition` |
| Poking Delta internals locally | **`deltalake`** Python package (Rust engine) | Reads and writes Delta with no JVM and no Spark. Lets you open `_delta_log` by hand on the Mac. |
| Local dataframe work | **Polars / DuckDB** | You already have notes on these — [[Note 01 - Polars, DB]] |

Nothing here needs Java, Windows, or an Azure subscription. Proposed conda env is in the project
folder; **do not create it until you have approved the file.**

## Naming currency check - verified 2026-08-13

Material older than ~2025 will teach you names that are now wrong. These cost credibility:

| Stale name | Current name |
|---|---|
| Delta Live Tables / DLT | **Lakeflow Declarative Pipelines** |
| `APPLY CHANGES INTO` (SQL) | **`AUTO CDC INTO`** |
| `dlt.apply_changes()` (Python) | **`dp.create_auto_cdc_flow()`**, from `pyspark import pipelines as dp` |
| Z-ORDER as the layout answer | **Liquid clustering**, recommended for all new Delta and managed Iceberg tables. `ZORDER BY` is not supported on liquid-clustered tables. |
| Community Edition | **Free Edition** |

Re-verify anything version-specific at the time you study it. Do not trust this table in 2027.

---

## BLOCK A - Interview-credible core (A1-A10)

Target: two weeks. After A10 you can hold a 45-minute conversation about medallion, CDC and ACID
upserts without bluffing, and you have six written answers.

| # | Unit | Deliverable | Maps to at Innova |
|---|---|---|---|
| **A1** | Why the lakehouse exists — object storage broke the warehouse, the log fixed it | Note: the problem in 6 sentences | Your `ADWC_LOAD` / `ADWC_USER` / `ADWC_RO` tiers |
| **A2** | The Delta transaction log, opened by hand | Screenshot/paste of a real `_delta_log` JSON commit you caused | Nothing — this is new. Oracle hid it. |
| **A3** | `MERGE INTO` on Delta; optimistic concurrency; idempotent writes | A working merge, run twice, row count unchanged | Your 158 `MERGE`s, the 15-column composite-key merge |
| **A4** | Medallion precisely — what belongs in bronze vs silver vs gold, and the boundary arguments | Your own 4-tier → 3-layer mapping diagram | 4-tier schema, `_STG` / `_STG_MANUAL` twins |
| **A5** | Spark execution model to speaking depth — driver/executor, job/stage/task, narrow vs wide, shuffle | Note: "why 100M rows is a shuffle question" | Nothing. This is the real gap. |
| **A6** | File layout: small-file problem, `OPTIMIZE`, liquid clustering, data skipping | Before/after file counts on a table you compact | Your Oracle table partitioning (14 uses) |
| **A7** | CDC on the lakehouse: Auto Loader, Change Data Feed, `AUTO CDC INTO`, SCD 1 / 2 / bitemporal | Note comparing all three to your replay logic | Bullhorn column-level history replay; 15-min ODS→ADWC job; SCD2 on `FACT_JOBORDER` |
| **A8** | Exactly-once and "zero data loss": checkpoints, idempotency, quarantine, expectations | Note: your Innova reliability story retold in lakehouse terms | `EXCEPTION WHEN OTHERS` per row; `CURRENT_DW_LOG_DATA` (363 refs); `MINUS` residue tracking |
| **A9** | Unity Catalog: three-level namespace, lineage, row filters and column masks | Note mapping UC row filters to your Power BI RLS | Dynamic `USERPRINCIPALNAME()` RLS with `PATHCONTAINS` |
| **A10** | Assemble the answers | **6 written interview answers + one "what I'd do differently now"** | All of the above |

## BLOCK B - Depth and hands-on (B11-B28)

Target: four weeks, but this is the part to stretch if the clock allows.

| # | Unit | Deliverable |
|---|---|---|
| **B11** | Bronze: Auto Loader ingest, schema evolution, rescued data column | Notebook: files landing → bronze table |
| **B12** | Silver: `AUTO CDC INTO` building an SCD2 dimension from a change feed | Notebook: SCD2 table with history you can query |
| **B13** | Gold: aggregate + one hard business rule | Port your nine-level crediting ladder to PySpark window functions |
| **B14** | Structured Streaming properly: triggers, watermarks, state, `availableNow` | Note: batch vs micro-batch vs continuous, and the cost argument |
| **B15** | Performance: AQE, skew joins, broadcast, partition sizing, caching | Note: the five things you check on a slow job, in order |
| **B16** | Read a real Spark UI / query profile on a job you made slow deliberately | Annotated screenshot of a shuffle you diagnosed |
| **B17** | Cost: serverless vs classic compute, Photon, autoscaling, DBU intuition | Note: how you'd size for 100M rows/day and defend the number |
| **B18** | Data quality as a gate: expectations, quarantine tables, failing a pipeline on purpose | Notebook: a bad row that gets quarantined, not dropped |
| **B19** | Delta internals: checkpoints, `VACUUM`, time travel, deletion vectors, schema evolution | Note: what `VACUUM` can destroy and how time travel really works |
| **B20** | Delta vs Iceberg vs Hudi; managed Iceberg in Unity Catalog | Note: the format question, answered without tribalism |
| **B21** | **Git-backed development + Databricks Asset Bundles** | A bundle you deployed. **This closes your stated CI/CD gap.** |
| **B22** | Workflows: task dependencies, retries, alerting, backfills | Note comparing to your Talend Master/child Execution Plans and year-bounded backfill jobs |
| **B23** | dbt on Databricks — appears in a large share of JDs | Note: where dbt ends and Lakeflow begins |
| **B24** | Source-side CDC: Debezium / Kafka concepts, log-based vs query-based vs trigger-based | Note: classify your Bullhorn history replay correctly |
| **B25** | Testing data pipelines: unit tests on transforms, integration tests on pipelines | One real test that fails when you break a transform |
| **B26** | **Migration case study: the Innova Oracle platform → lakehouse** | A written architecture. This is a portfolio artifact and a very strong interview answer. |
| **B27** | Write up B26 with cost and SLA reasoning | 2-page design doc |
| **B28** | Mock interview + gap sweep | List of what you still cannot answer, honestly |

---

## Progress ledger

Append one line per completed unit. Date, unit, what actually landed, what stayed murky.
Murky items feed B28.

| Date | Unit | Landed | Still murky |
|---|---|---|---|
| | | | |
