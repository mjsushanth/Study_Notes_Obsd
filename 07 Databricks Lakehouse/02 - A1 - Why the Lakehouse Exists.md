---
title: A1 - Why the Lakehouse Exists
type: study-unit
unit: A1
block: A
est_time: 60min
created: 2026-08-13
status: not-started
tags:
  - study/databricks
  - study/unit
---

# A1 - Why the Lakehouse Exists

Index: [[00 - INDEX - Curriculum and Unit Map]] · Translation: [[01 - Translation Table - Innova to Lakehouse]]

**Goal of this hour.** Be able to say, unprompted, what problem Delta Lake solves and why it needed
solving. This is the framing every other unit hangs off. If you skip it you will learn Databricks
as a pile of product names.

**Prerequisite.** None.

---

## 1. Read (20 min)

### The question to hold in your head

You had Oracle ADW. It gave you transactions, `MERGE`, constraints, indexes, a query planner. Why
would anyone give that up for files in a bucket?

### The three-act history

**Act 1 — the warehouse (yours).** Data lands in a proprietary system that owns both storage and
compute. You get ACID, SQL, and a planner that knows the data. You pay for it in three ways: the
storage is proprietary so only that engine can read it, compute and storage scale together whether
you want that or not, and it is a poor home for anything that is not tabular.

**Act 2 — the data lake.** Dump everything as files (Parquet, JSON, CSV) into cheap object storage
— S3, ADLS, GCS. Any engine can read it. Storage and compute are finally separate. But object
storage has no transactions. Concretely, what breaks:

- A writer crashes halfway. Half the files are new, half are old. Readers see a torn state, and
  there is no rollback.
- Two jobs write the same table concurrently. Last-writer-wins at the *file* level, so you silently
  lose rows.
- You cannot `UPDATE` or `DELETE` a row. Parquet files are immutable. GDPR deletion means rewriting
  partitions by hand.
- Listing a million files to plan a query is itself slow, and object-store listings are only
  eventually consistent.
- No schema enforcement. One bad producer writes a `string` where an `int` was, and the table is
  poisoned.

This is the "data swamp" era. The failure is not storage — it is the **absence of a transaction
boundary**.

**Act 3 — the lakehouse.** Keep the cheap open files. Add one thing: an **ordered log of commits**
sitting beside them, which is the single source of truth about which files are currently part of
the table. That log is `_delta_log/`. Everything Delta gives you falls out of it:

| Guarantee | How the log provides it |
|---|---|
| Atomicity | A commit is one log entry. Until it is written, none of your new files count. Crash = no entry = no change. |
| Isolation (snapshot) | A reader resolves the file list at one log version and reads that. Writers adding versions do not disturb it. |
| Concurrency | Optimistic: write files, then attempt to append log version N+1. If someone else took N+1, re-check for conflict and retry. No locks. |
| `UPDATE` / `DELETE` / `MERGE` | Write new files with the changed data, then commit a log entry that removes the old files and adds the new. Immutability preserved. |
| Time travel | Old log versions still exist, so "the table as of version 12" is just replaying the log to 12. |
| Schema enforcement | Schema is recorded in the log, and writes are checked against it. |
| Fast planning | The log carries per-file min/max statistics, so most files are skipped without being opened. |

**The one-sentence version:** *a lakehouse is a data lake with a transaction log, and the log is
what makes it behave like a warehouse.*

### Why this matters for the JD you're chasing

"Medallion architecture, CDC patterns, ACID-compliant upserts, 100M+ records daily, zero data loss"
is one sentence describing a system that needs a transaction boundary at every layer. That is what
the log is for.

### Where your Innova platform sits

You built the warehouse version of this (Act 1). Your `ADWC_LOAD` → `ADWC_USER` → `ADWC_RO` tiers
are the medallion shape. What Delta changes is *not* the architecture — it is what is underneath
it. Hold on to this: **you are not relearning the architecture, only the substrate.**

## 2. Do (25 min)

1. **Sign up for Databricks Free Edition** — `databricks.com/learn/free-edition`. Verified current
   as of 2026-08-13; Community Edition retired 2026-01-01. No credit card, no cloud account.
   Serverless compute and default storage are included.
2. Open the workspace. Create a notebook. Run:
   ```sql
   CREATE OR REPLACE TABLE a1_scratch (id INT, name STRING);
   INSERT INTO a1_scratch VALUES (1, 'joel'), (2, 'test');
   DESCRIBE DETAIL a1_scratch;
   DESCRIBE HISTORY a1_scratch;
   ```
3. Read the `DESCRIBE HISTORY` output. Every row is a log commit. Note the `operation`,
   `operationMetrics` and `version` columns. **That table is the transaction log, surfaced as SQL.**
4. Now do the thing Oracle never let you see:
   ```sql
   UPDATE a1_scratch SET name = 'joel_m' WHERE id = 1;
   DESCRIBE HISTORY a1_scratch;
   SELECT * FROM a1_scratch VERSION AS OF 1;
   ```
   You just queried the table as it was before your update. There was no backup table.

## 3. Write (10 min)

Add to the bottom of this note, in your own words:

- The lakehouse problem statement in **six sentences**. No product names allowed.
- One sentence on which of your `_BKP<date>` pre-change snapshot tables Delta time travel would
  have made unnecessary, and one sentence on where a `_BKP` table would still be the right call.

## 4. Check yourself

You have finished this unit when you can answer these without notes:

1. What exactly breaks when two jobs write to the same Parquet table on S3 at once?
2. Why does Delta not need row locks?
3. Your Oracle warehouse already had ACID. What did the lakehouse actually buy you?
4. Where does Delta store the min/max statistics that let it skip files?

If any answer is shaky, it goes in the **Still murky** column of the ledger in
[[00 - INDEX - Curriculum and Unit Map]] and gets swept in B28.

---

## My notes

<!-- write here -->
