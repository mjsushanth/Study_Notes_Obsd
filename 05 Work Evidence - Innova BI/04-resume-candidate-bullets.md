---
title: Resume Candidate Bullets
type: resume-draft
audit_date: 2026-08-12
status: evidence-backed draft, pre-amplification
tags:
  - evidence/resume
  - work/innova
---

# 04 - Résumé Candidate Bullets

Back to [[00-audit-summary]] · evidence IDs resolve in [[02-evidence-ledger]] ·
open items in [[05-questions-for-joel]] · handoff brief in [[07-strength-points]]

**These are deliberately un-amplified.** Every number traces to an artifact. The amplification
pass (reframing toward the current BI/AI tooling landscape) comes later and should start from
these, not from the old résumé.

Rules applied: no metric without an evidence ID; no "real-time"; no forecasting; no
revenue/savings impact; no invented user counts; no "leveraged" or "spearheaded". "Fuzzy
matching" **is** permitted — it is confirmed by `EV-65`; only the 88% accuracy figure is barred.

---

## A. Work-experience bullets — two-page master résumé (12)

**Data Engineer / BI Engineer — Innova Solutions, Hyderabad, India — Dec 2019 – Nov 2023**
*(entity lineage: GGK → acquired into ACS / American CyberSystems → renamed Innova Solutions; Innova later acquired Volt Information Sciences)*

1. Built and operated the Oracle Autonomous Data Warehouse platform behind a US staffing group's
   recruiting and finance reporting, integrating **8+ source systems** — Bullhorn ATS (SQL Server
   mirror), Bullhorn and Hiregenics operational data stores, Oracle E-Business Suite across four
   instances, Salesforce, the Coupa API, Oracle HR extracts and SFTP spreadsheet feeds — into a
   staging → curated → read-only schema architecture. `EV-01` `EV-28` `EV-02`

2. Authored **39 Oracle PL/SQL load procedures** (~30,700 lines across the four production
   modules) driving a **42-dimension / 40-plus-fact star schema**, all instrumented into a shared
   run-audit table capturing procedure name, status, target table, row count, start/end timestamps
   and error message. `EV-05` `EV-02` `EV-14`

3. Implemented column-level change-data-capture replay against Bullhorn audit history to rebuild
   job-order and submission facts — resolving two conflicting date formats stored as text,
   remapping IDs that arrive as strings in history but numbers in the live table, and isolating
   failures per row so a single malformed record cannot abort a 1,470-line load. `EV-07`

4. Applied **SCD Type 2** history to job-order and submission facts using effective-date
   end-dating plus a current-record flag, and maintained a **48-table monthly snapshot history**
   of FP&A measures spanning 2020–2023 with parameterized period-scoped and year-scoped
   reload. `EV-37` `EV-18`

5. Designed the producer-crediting algorithm that made revenue attribution correct on a fact
   grained per producer-role: a nine-level designation precedence ladder stored as data, window
   functions to detect and rank multi-role rows, financial measures zeroed on non-primary rows so
   each producer keeps their own scorecard, then a 15-column composite-key `MERGE` back into the
   fact. `EV-08`

6. Built cross-system entity resolution for people and companies across **six systems with no
   shared key** (Bullhorn, Hiregenics, JobDiva/ORCA, Azure AD, Oracle CE-HR, Oracle EBS) using
   cascading multi-round string joins — **six-way match ladders** over normalized, concatenated,
   case-folded and reversed name forms — with window-function collision detection, `HAVING COUNT`
   uniqueness probes, `MINUS`/`NOT EXISTS` residue tracking, sentinel keys so unmatched rows stay
   visible, and a durable manual-override column for exceptions. `EV-65` `EV-66` `EV-67` `EV-68`
   `EV-71` `EV-72` `EV-06`

7. Delivered the recruiter and sales performance semantic model in Power BI: **40 tables, 187
   relationships, 354 DAX measures** (median 13 lines; 66 over 30 lines), three bridge tables for
   many-to-many resolution, a calculation group built in Tabular Editor, and an **85-page /
   1,035-visual** report with mirrored recruiter and business-development navigation. `EV-20`
   `EV-23` `EV-22` `EV-25` `EV-21`

8. Implemented dynamic hierarchical row-level security — identity from the signed-in principal,
   organisational-subtree visibility through a warehouse-maintained parent-child path, and a
   data-driven super-user bypass — and prototyped client-level external RLS with two competing
   Power Query implementations evaluated side by side. `EV-24` `EV-46`

9. Made the whole stack promotable across **four environments (DEV/QA/UAT/PROD) by parameter
   injection rather than code edits**: schema-parameterized PL/SQL built with dynamic SQL,
   **69 versioned Talend job artifacts** deployed through a manifest carrying per-environment
   context parameters and cron triggers, and a semantic model exposing server and schema
   parameters. `EV-29` `EV-11` `EV-26`

10. Replaced off-system finance spreadsheets with a governed override path — automated and manual
    staging twins for every hierarchy dimension and the FP&A fact, a `manual_load_flag` carried
    onto the fact so adjustments stay traceable in reports, and dedicated Talend jobs ingesting
    finance-owned files from SFTP. `EV-19`

11. Owned Power BI platform administration across two tenants after an acquisition: workspace and
    data-source inventory, on-premises gateway upgrade and post-migration verification procedures,
    a production gateway VM dependency audit, a specification for dedicated production BI
    infrastructure (32 GB / 8 cores), and five monitoring reports — one sourcing the **Power BI
    REST API through a service principal** to track workspaces, capacity assignment and refresh
    status. `EV-27` `EV-50` `EV-56` `EV-57` `EV-58`

12. Wrote the Phase 2 High Level Design (glossary, data-flow, source-to-target mapping,
    relationship and measure registers) with its companion test-case pack, kept test IDs
    referenced directly in the load code, and authored a two-track Power BI and SQL training
    curriculum covering the tabular engine, Import versus DirectQuery, dynamic RLS, query
    processing order and window functions. `EV-54` `EV-55` `EV-60`

---

## B. Compressed bullets — one-page, experience-first (6)

**Data Engineer / BI Engineer — Innova Solutions — Dec 2019 – Nov 2023**

1. Built and ran an Oracle Autonomous Data Warehouse integrating 8+ systems (Bullhorn ATS, Oracle
   EBS across four instances, Hiregenics ODS, Salesforce, Coupa API, HR and SFTP feeds) through
   **39 PL/SQL load procedures** over a 42-dimension star schema, with run-level audit logging on
   every procedure. `EV-28` `EV-05` `EV-02` `EV-14`

2. Implemented CDC replay from ATS audit history, SCD Type 2 on transactional facts, and a
   48-table monthly FP&A snapshot history with parameterized period and year reload. `EV-07`
   `EV-37` `EV-18`

3. Designed the producer-crediting algorithm — designation precedence ladder, window-function
   collision detection, measure zeroing on non-primary rows, composite-key `MERGE` — so revenue
   counts once while every producer keeps their own scorecard. `EV-08`

4. Resolved person and client identity across six systems with no shared key via cascading
   six-way fuzzy string-join ladders, window-function duplicate resolution, explicit residue
   tracking and a durable manual-override path. `EV-65` `EV-66` `EV-68` `EV-71`

5. Delivered a 40-table / 187-relationship / **354-measure** Power BI model with dynamic
   hierarchical row-level security, a Tabular Editor calculation group, and an 85-page report
   across dual personas. `EV-20` `EV-24` `EV-25` `EV-21`

6. Made deployment parameter-driven across four environments — **69 versioned Talend artifacts**
   with per-environment context injection, schema-parameterized procedures and a parameterized
   semantic model — and administered Power BI for two tenants, including gateway upgrades and
   REST-API-based refresh and workspace monitoring. `EV-29` `EV-11` `EV-27` `EV-58`

---

## C. Professional summaries (3 versions)

### C1 — Data engineering first

> Data engineer with four years building and operating a production Oracle Autonomous Data
> Warehouse for a US staffing group, plus an MS in the US. Integrated eight-plus source systems —
> Bullhorn ATS, Oracle E-Business Suite, Hiregenics, Salesforce, Coupa — through 39 PL/SQL load
> procedures and 69 versioned Talend jobs across DEV, QA, UAT and production. Comfortable in the
> parts that are actually hard: change-data-capture replay from audit history, SCD Type 2,
> cascading fuzzy-join identity resolution across six systems with no shared key, and revenue
> attribution logic where the fact grain makes naive aggregation wrong. Every pipeline instrumented, every environment promoted by parameter
> rather than by editing code.
> `EV-28` `EV-05` `EV-11` `EV-07` `EV-37` `EV-06` `EV-08` `EV-29`

### C2 — AI/ML plus production data engineering

> MS-trained engineer pairing machine learning work with four years of production data
> engineering. Built the Oracle warehouse and Power BI semantic layer behind a US staffing group's
> recruiting and finance reporting — eight-plus integrated systems, 39 PL/SQL procedures, a
> 354-measure tabular model, row-level security resolved through an organisational hierarchy.
> The value I bring to ML work is the half most teams underestimate: the fuzzy entity resolution,
> the grain discipline, the historical snapshots and the audit logging that decide whether a model
> is trained on data that means what it claims to mean.
> `EV-28` `EV-05` `EV-20` `EV-24` `EV-06` `EV-18` `EV-14`

### C3 — Analytics engineering / BI platform

> Analytics engineer who has owned the whole path from source system to governed semantic model.
> Four years on an Oracle ADW and Power BI platform for a US staffing group: 42-dimension star
> schema, a 354-measure tabular model with dynamic hierarchical row-level security and a
> calculation group, an 85-page report serving two distinct personas, and a governed path for
> finance overrides that kept adjustments inside the warehouse instead of in spreadsheets. Also
> ran the platform itself — two Power BI tenants, on-premises gateway upgrades, REST-API refresh
> and workspace monitoring — and wrote both the design documents and the training curriculum the
> team worked from.
> `EV-02` `EV-20` `EV-24` `EV-25` `EV-21` `EV-19` `EV-27` `EV-58` `EV-54` `EV-60`

---

## D. Phrasing bank — safe vs unsafe

| Do not write | Write instead | Why |
|---|---|---|
| "real-time data pipelines" | "daily batch pipelines with a 15-minute CDC feed" | `EV-04`; production cron is daily at 14:00 IST |
| "incremental refresh in Power BI" | "parameterized incremental loads at the warehouse layer" | `EV-48`; no `RangeStart`/`RangeEnd` exists |
| "6+ fuzzy matching patterns, 88% accuracy" | "6+ fuzzy match patterns via cascading multi-round string joins" — keep the pattern count, drop the percentage | `EV-65`; **the pattern count is confirmed** (199 OR-laddered string predicates, 66 of them six-way). Only the 88% lacks an artifact |
| "built forecasting models" | "budget and target versus actuals variance reporting" | `EV-53`; no forecasting code exists |
| "SCD Type 2 dimensions" | "SCD Type 2 on job-order and submission facts, plus monthly snapshot history for FP&A" | `EV-37` `EV-18`; the history strategies differ by subject area |
| "CI/CD pipelines reducing deployment time 50%" | "controlled promotion across four environments by parameter injection rather than code edits" | `EV-29`; no pipeline definitions, no timing baseline. See Q4 |
| "optimized model performance" | "40-table model with 179 of 187 relationships kept single-direction" | `EV-51`; auto date/time left on and a 520 MB PBIX argue against an optimization claim |
| "25+ dashboards, 500+ metrics" | "354 measures in the primary semantic model; 10 named production reports; 85 report pages" | `EV-20` `EV-09` `EV-21`; these are countable |
| "led a team of 2–4" | "mentored teammates; authored the team's Power BI and SQL training curriculum" | `EV-60` `EV-64`; team size is unevidenced here |
| "drove $X savings / improved margin" | describe the mechanism, not the money | `EV-17`; processing financial data is not evidence of financial impact |
| "PL/SQL packages" | "PL/SQL procedures" | one `CREATE PACKAGE` in the whole corpus |
| "performance tuning with partitioning and parallelism" | "table partitioning and index management around bulk loads" | no `PARALLEL` hints, no materialized views, no stats gathering |

---

## E. Ordering recommendation

Put experience above education, and inside experience order the bullets **A1, A2, A5, A9, A7,
A11, A3, A6, A4, A8, A10, A12**. Rationale: open with platform scope and code volume (credible,
concrete), then lead with the two bullets that show judgment rather than tool use (the crediting
algorithm and the promotion architecture), then the semantic-model scale, then the
platform-administration year — which is the differentiator most BI job descriptions ask for and
which the current résumé omits entirely.
