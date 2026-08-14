---
title: Translation Table - Innova Oracle to Databricks Lakehouse
type: reference
created: 2026-08-13
source_evidence: "[[00-audit-summary]], [[07-strength-points]], [[03-technical-depth]]"
tags:
  - study/databricks
  - work/innova
  - interview/data-engineering
---

# 01 - Translation Table - Innova to Lakehouse

Companion: [[00 - INDEX - Curriculum and Unit Map]]

Every left-column item is **verified from your own artifacts**, not from a résumé. Every
right-column item is what the same idea is called in a Databricks JD. Learn the right column by
walking across from the left, never from scratch.

---

## The core claim this note supports

> "I built medallion architecture, change data capture, and ACID upserts for four years. I built
> them on Oracle Autonomous Data Warehouse rather than on Delta Lake, so what I am adding is the
> distributed execution model and the lakehouse storage layer — not the data engineering."

That sentence is defensible under follow-up questions. Rehearse it in Unit A10.

---

## Architecture

| Innova (verified) | Lakehouse name | Note for the conversation |
|---|---|---|
| `ADWC_LOAD` — staging, raw source shape | **Bronze** | Same job: land it, do not judge it. Bronze additionally keeps the raw file and ingest metadata. |
| `ADWC_USER` — curated star schema, 42 `DIM_*` / 84 `FACT_*` | **Silver** | Conformed, deduplicated, entity-resolved. Your `COMMON_*` conformed field vocabulary (1,780 refs) is textbook silver. |
| `ADWC_RO` — read-only reporting layer, 13 `RPT_*` | **Gold** | Business-level aggregates serving BI. |
| Four environments DEV/QA/UAT/PROD, schemas mirrored `_UAT` / `_DEV` | **Catalogs in Unity Catalog** | UC's three-level `catalog.schema.table` is close to your schema-per-environment convention. Say this — it lands well. |
| Talend orchestration, Master/child Execution Plans | **Lakeflow Jobs / Workflows** | Task dependencies, retries, alerting. Your `Master_BH_Load_Job` is a job DAG. |
| Power BI semantic model on top | **Databricks SQL + BI tool** | Unchanged in shape. |

**You already have a medallion architecture on your résumé.** It is three named tiers with
staging→curated→reporting promotion. Use the word.

## Upserts and transactions

| Innova (verified) | Lakehouse name | What is genuinely different |
|---|---|---|
| `MERGE` upsert, **158 uses across 24 files** | `MERGE INTO` on a Delta table | The statement is nearly identical. What is new: **why** it is atomic on object storage — optimistic concurrency against a transaction log, not row locks. |
| 15-column composite-key `MERGE` into the producer fact, so manual and system rows never collide | Merge condition design / idempotency key | Same concept. This is a *good* interview story about merge-key design. |
| SCD Type 2 via `EFF_START_DATE` / `EFF_END_DATE` / `IS_CURRENT` correlated-subquery end-dating on `FACT_JOBORDER`, `FACT_JOBSUBMISSION` | `AUTO CDC INTO ... STORED AS SCD TYPE 2` | Databricks generates the end-dating for you. You hand-wrote it. **Say that** — it means you know what the abstraction is hiding. |
| 48 physical monthly snapshot tables, Jan 2020 – Dec 2023 | Snapshot pattern / time travel | Delta time travel and `BITEMPORAL` mode make most of those 48 tables unnecessary. Excellent "what I would do differently now" material. |
| `EXECUTE IMMEDIATE` dynamic SQL (665 uses) for schema-agnostic procedures | Parameterized notebooks / DAB variables / widgets | Same intent: one artifact, four environments, zero code edits. |

## CDC

| Innova (verified) | Lakehouse name | Classification |
|---|---|---|
| Column-level replay of Bullhorn's audit-history rows, rebuilding facts change by change | **Log-based CDC, consumed downstream** | You consumed a source system's own change log. That is the same class as Debezium output. |
| 15-minute `CDC_JOB_FROM_ODS_TO_ADWC` | Micro-batch streaming ingest | Your real SLA floor was 15 minutes. Say "daily batch with a 15-minute CDC feed" — it is true and it is not nothing. |
| SFTP-landed Excel feeds picked up by dedicated Talend jobs | **Auto Loader** (`cloudFiles`) | Auto Loader does file discovery, schema inference and evolution, and exactly-once file tracking. |
| Two different text date formats detected and parsed per history row; string-vs-number type drift repaired via lookup | Schema evolution + rescued data column | Your handling is more manual and more precise than the default. Good story. |
| `'[blank]'` sentinel handling, quote escaping before dynamic execution | Data cleansing in bronze→silver | — |

## Reliability - your "zero data loss" story

This is the strongest translation you have, because most candidates cannot describe a real one.

| Innova (verified) | Lakehouse name |
|---|---|
| Per-row `EXCEPTION WHEN OTHERS` so one malformed history row cannot abort the load | **Quarantine pattern** — bad rows diverted, not dropped, pipeline survives |
| `CURRENT_DW_LOG_DATA(SP_NAME, SP_STATUS, TABLE_NAME, LOADED, START, END, RECORDS_COUNT, ERROR_MESSAGE)` — **363 references**, larger procedures logging per stage | **Pipeline observability / event log**. Lakeflow emits this automatically; you built it by hand across 39 procedures. |
| `MINUS` (110), `NOT EXISTS` (38), `NOT IN (SELECT...)` (62) computing the unmatched residue explicitly | **Reconciliation / completeness checks** |
| Sentinel surrogate keys (`999999999`, `'N/A'`) so orphans stay visible and totals reconcile | Referential integrity handling without dropping rows |
| `HAVING COUNT(*) > 1` probes (277 uses / 64 files) shipped beneath the load code | **Data quality expectations** — the test ships with the change |
| `Run_Type = 'VERIFICATION'` non-mutating check mode before a real run | **Dry run / pipeline validate mode** |
| `_BKP<date>` pre-change table snapshots for rollback | **Time travel / `RESTORE TABLE`** — Delta gives you this for free |

## Governance

| Innova (verified) | Lakehouse name |
|---|---|
| Power BI dynamic RLS: `USERPRINCIPALNAME()` against an email-to-person map, resolving a parent-child `PATH`/`PATHCONTAINS` org hierarchy, with `ISSUPERUSER` bypass | **Unity Catalog row filters** + dynamic views using `current_user()` / `is_account_group_member()` |
| Manual-override column `MANUALMAP_BH_USERID` surviving every reload; `manual_load_flag` (295 refs) propagated onto the fact | Governed override / audit lineage |
| Four-tier schema separation with read-only reporting layer | UC catalog and schema grants |

## Things you did NOT do - do not claim these

| Gap | Say instead |
|---|---|
| Databricks, Spark, Delta Lake in production | "Proficient, built pipelines on Free Edition; my production warehouse experience is Oracle ADW." |
| Git, CI/CD, automated test harness | Honest gap. **Close it in Unit B21**, then say "I've since built deployment with Databricks Asset Bundles." Do not say "CI/CD pipelines" about Innova — the audit is explicit that there was no repo or build server. |
| Real-time / streaming | Your floor was a 15-minute CDC job. Say that number. |
| 100M+ rows/day | Volume is **unverified** in your artifacts. Do not attach a number you cannot source. Describe the architecture instead. |
| Power BI model-size optimization | 115 auto-generated `LocalDateTable_*` tables were left on. This is "what I'd do differently", not an achievement. |
| Azure | You have none. Cloud-agnostic Databricks + your AWS sys-design notes is the honest position. |

---

## The three sentences to have ready

1. **Medallion.** "At Innova I ran a three-tier warehouse — staging, curated star schema,
   read-only reporting — across four environments. That is medallion; we called the layers
   `ADWC_LOAD`, `ADWC_USER` and `ADWC_RO`."
2. **ACID upserts.** "About 158 `MERGE` upserts across the load procedures, the heaviest one
   merging into the producer fact on a 15-column composite key so manually adjusted finance rows
   and system rows could never collide. On Delta the statement is the same; what I had to learn was
   how atomicity works on object storage without row locks."
3. **CDC + zero data loss.** "The job-order fact was rebuilt by replaying the ATS's own audit
   history column by column, with each row wrapped so one malformed record couldn't abort the load,
   and a run-audit row written per stage across every procedure — 363 call sites. The unmatched
   residue was computed with `MINUS` and carried forward rather than dropped, so totals always
   reconciled."
