---
title: Six Strength Points - Handoff Brief
type: handoff-brief
audit_date: 2026-08-12
audience: independent resume agent (standalone - assumes no access to the other notes)
employer_line: "Innova Solutions (formerly ACS / American CyberSystems; GGK acquired into ACS), Hyderabad, India"
role: Data Engineer / BI Engineer
dates: 2019-12 to 2023-11
tags:
  - evidence/handoff
  - work/innova
  - domain/data-engineering
---

# 07 - Six Strength Points (Handoff Brief)

Companion notes: [[00-audit-summary]] · [[02-evidence-ledger]] · [[03-technical-depth]] ·
[[04-resume-candidate-bullets]] · [[05-questions-for-joel]]

**Read this first if you are the résumé agent.** This brief is self-contained. Every number below
was recovered from primary artifacts — source code, deployment manifests, Power BI model
internals, design documents — not from a previous résumé. Where a claim is not provable, it says
so. Nothing here needs softening for accuracy; it may be sharpened for register.

**Employment context.** One employer, one continuous role, Dec 2019 – Nov 2023, Hyderabad.
Corporate lineage: **GGK → acquired into ACS (American CyberSystems) → ACS renamed to Innova
Solutions**; Innova subsequently acquired **Volt Information Sciences** and other firms. Artifacts
carry all three names, which is why the estate spans multiple staffing brands (Innova, Volt,
Diversant, Hiregenics, Ascent) and two Power BI tenants. **Use "Innova Solutions" as the employer
line.** The multi-brand, post-acquisition complexity is a genuine strength, not a
complication — say so.

**Domain.** US staffing and workforce solutions. Recruiters and business development managers
place contractors and permanent hires; the business needs to know, per person and per client,
what activity happened and what money it made. Everything below serves that.

**Platform.** Oracle Autonomous Data Warehouse Cloud (ADWC), fed by Talend, consumed by Power BI,
across four environments (DEV / QA / UAT / PROD).

---

## Point 1 — Cross-system identity resolution through cascading fuzzy string joins

**The problem.** The same human being and the same client company existed in six systems under
six different spellings, with no shared key. Bullhorn (ATS) had one name form, Hiregenics another,
JobDiva/ORCA another, Azure AD another, Oracle CE-HR another, Oracle E-Business Suite another.
Until a person and a client resolved to one identity, recruiter activity could not be joined to
revenue — which means no producer scorecard, no commission calculation, and no client-scoped
security. This was the foundational blocker for every downstream deliverable.

**What was built.** A multi-round cascading match pipeline in Oracle PL/SQL, where each round
joins on a *different derived string construction* of the same entity, progressively resolving
what the previous round could not, deliberately over-matching, and then resolving the resulting
duplication with business rules.

The match strategies materialized as first-class join columns — each one a distinct "fuzzy
pattern":

| Strategy | Column family | Handles |
|---|---|---|
| `first + last` concatenation | `*_USERNAMEJOINFIELD` (626 refs) | the base case |
| `last + first` reversal | `*_LASTFIRSTNAME_JOINFIELD` (254 refs) | systems that store names reversed |
| first + **middle** + last | `COMMON_MIDDLENAMEFIELD` (middlename: 2,530 refs) | records where one system carries a middle name and the other does not |
| preferred/display name | `CEHR_PREFNAME` (113 refs) | HR "preferred name" vs legal name |
| case-normalized variants | `COMMON_USERNAMEFIELD_CCASE`, `COMMON_LFNAMEFIELD_CASE` | initcap vs upper vs raw source casing |
| employee-ID crosswalk | `COMMON_EMPLOYEEID` (91 refs) | where an ID does exist in both systems |
| system-prefixed composite keys | `'DCH '‖id`, `'BH '‖id`, `CONCATKEY_DCHCLIENTS` | keeps provenance inside the surrogate key so a key collision across systems is impossible |
| role + nationality composite | `CONCATKEY_ROLE_NATN` | one person holding two roles must resolve to two target rows |

**Verified scale of the technique.** Across the deduplicated 393-file SQL corpus:

- **8,918 join predicates** parsed, of which **1,609 (18%) join on transformed strings rather than
  on keys** — `TRIM`, `UPPER`, `LOWER`, `SUBSTR`, `INSTR`, `INITCAP`, `LIKE`, `‖` concatenation,
  `NVL` guards.
- **5,527 `LEFT JOIN`s across 207 files** — this is a left-join-cascade codebase by design, not by
  accident. The heaviest single procedures carry 178, 158, 145 and 136 LEFT JOINs.
- **199 multi-alternative OR-laddered string join predicates**, distributed by number of
  alternatives: **66 are six-way ladders**, 25 five-way, 11 four-way, 19 three-way, 2 eight-way,
  1 seven-way. *This is the concrete basis for "6+ fuzzy matching patterns" — the six-way ladder
  is the dominant form.*
- **9,593 `‖` string concatenations** building composite keys.
- **1,780 references to `COMMON_*` conformed fields** — the resolved-identity vocabulary that the
  whole warehouse then joins on.

**A real match predicate**, from the production person-resolution procedure — four alternatives in
one `ON` clause, including the name-order reversal:

```sql
   TRIM(UPPER(NVL(HGC_USERNAMEJOINFIELD,'N/A'))) = TRIM(UPPER(CEHR_PREFNAME))
OR TRIM(UPPER(NVL(BH_USERNAMEJOINFIELD ,'N/A'))) = TRIM(UPPER(CEHR_PREFNAME))
OR TRIM(UPPER(BH_FIRSTNAME)) ‖TRIM(UPPER(BH_LASTNAME))  = TRIM(UPPER(CEHR_LASTNAME))‖TRIM(UPPER(CEHR_FIRSTNAME))
OR TRIM(UPPER(HGC_FIRSTNAME))‖TRIM(UPPER(HGC_LASTNAME)) = TRIM(UPPER(CEHR_LASTNAME))‖TRIM(UPPER(CEHR_FIRSTNAME))
```

**The cascade, as a staged pipeline.** The resolver materializes each round as its own table so
every intermediate result is inspectable and re-runnable — `LKP_HGC_PERSON_DESIGNATIONS` →
`BHMR_HGC_EXCLUSIVE_BH_TBL` (the set only one system knows about) → `BHMR_HGC_UNIONTBL` →
`LKP_HGC_DISTINCTNAMES_LF` → `BHMR_HGC_TO_CEHR_NEWUSER` (the crosswalk) → sequential refinement
passes → `BHRPT_DIM_BHHGC_USERENT_PNL` (the conformed producer dimension). 1,469 lines, with a
run-log row written **per stage**, not per procedure.

**Handling the duplication that loose joins create — the part that makes it engineering rather
than guessing:**

- `ROW_NUMBER() OVER` (201 uses / 41 files) and `COUNT(*) OVER` (127 / 42) to detect and rank
  collisions rather than silently taking the first row.
- `HAVING COUNT(*) > 1` probes (277 / 64 files) written directly beneath the load code, so the
  duplicate check ships with the change.
- `MINUS` (110), `NOT EXISTS` (38), `NOT IN (SELECT ...)` (62) to compute the unmatched residue
  explicitly and carry it forward instead of losing it.
- Anti-joined `UNION ALL`, so a client matched in round 1 is excluded from round 2's population
  and cannot be counted twice.
- Sentinel surrogate keys (`999999999`, `'N/A'`) for the still-unmatched — totals reconcile and
  orphans stay visible rather than vanishing from a report.
- A **manual override column on the match table itself** — `MANUALMAP_BH_USERID` (128 refs) — so a
  human decision about a specific person survives every subsequent reload. This is the mature
  answer to fuzzy matching: automate the 95%, and give the remainder a durable, auditable home.
- Test and house-account exclusion by rule: names matching `(N/A) - NOT AVAILABLE`,
  `HOUSE, HOUSE`, `HOUSE, HG`, `HOUSE, ASCENT`, plus an explicit `BHMR_DUMMY_DATA` exclusion list.

**Ambiguity resolution as a deliberate business rule.** Role classification runs as a cascade of
passes over the crosswalk: department containing `%RECRUIT%` → recruiter flag; `%SALES%` →
business-development flag; **neither → set both flags**, i.e. when the source data cannot classify
a person, credit them into both populations and let the downstream attribution logic (Point 2)
resolve which one actually earns. Customer-success managers are then flagged by name against a
three-way `UNION` of distinct CSM names across job-order and placement tables. Over-match first,
disambiguate with rules second — applied consistently.

**Client-side equivalent.** Normalized exact match between Bullhorn client corporations and Oracle
EBS customer hierarchy, an anti-joined union of matched plus EBS-exclusive customers, then a
curated standardization table for the residue. Recorded working figures: **2,206 matched Bullhorn
client entries resolving to 1,476 distinct EBS customers out of 4,703**, with the ~3,227 residue
computed and tracked. Two documented design decisions worth quoting as evidence of product
judgment: the **Bullhorn** name wins as the display name because the slicer is recruiter-facing,
while the relationship to the fact travels on the **EBS** key because the fact is financial.

**Résumé framing.** "Built cross-system entity resolution for people and companies across six
systems with no shared key, using cascading multi-round string joins — six-way match ladders over
normalized, concatenated and reversed name forms — with window-function collision detection,
explicit unmatched-residue tracking, sentinel keys, and a durable manual-override path for the
exceptions." Safe to say **6+ match patterns**. If an accuracy figure is used it must come from
Joel; the artifacts record match *counts*, not a certified accuracy percentage.

---

## Point 2 — Financial attribution: crediting revenue once across nine producer roles

**The problem.** Up to seven people can hold a role on a single placement — recruiter, recruiter
override, secondary recruiter, BDM, BDM override, secondary BDM, customer success manager. The
producer fact is therefore grained at
`assignment × producer × designation × period × activity_code`, which is *correct* for
scorecards — every producer must see their own row — and *catastrophic* for financials, because
`SUM(revenue)` multiplies the same money by the number of people who touched the deal. Commission
statements and P&L reporting both run off this table.

**The solution.** An attribution algorithm that keeps both properties simultaneously:

1. **Precedence as data, not code.** A nine-level designation ladder — `REC`(1), `RECOVR`(2),
   `BDM`(3), `BDMOVR`(4), `CSM`(5), `SECREC`(6), `TERREC`(7), `SECBDM`(8), `TERBDM`(9) — built as
   an inline ranked set. Adding a new designation is a row, not a deployment.
2. **Detect the collision.**
   `COUNT(*) OVER (PARTITION BY assignment_id, name, person_employeeid, activity_code, period_name)`
   identifies multi-role rows.
3. **Elect the credit holder.** `ROW_NUMBER() OVER (... ORDER BY index_des)` ranks the colliding
   rows by designation precedence.
4. **Zero, don't delete.** For every non-primary row, ~22 financial measures are set to zero —
   revenue, GP, AWGP, burden and its components, direct cost, COGS, total margin, billed FTE,
   hours billable and payable, direct-hire amount, perm fee. The row survives, so each producer
   still appears in their own scorecard with their activity intact; only the money is attributed
   once.
5. **Merge back safely.** A `MERGE` into the production fact on a **15-column composite key**
   including `activity_code`, `max_expenditure_id` and `manual_load_flag` — so manually adjusted
   rows and system-sourced rows can never collide.

Recorded run figures: **341,711 rows** written in the crediting pass; **37,129** test/house rows
excluded. Test-case IDs are referenced in the code header, traceable back to a companion test-case
workbook.

**The metric being attributed.** AWGP — Average Weekly Gross Profit — defined verbatim in the
project's own rule document as *total margin or GP from Oracle timesheets, divided by the number of
fiscal weeks adjusted for holidays, for the selected period*. The denominator is the hard part: it
is holiday-aware and it changes with the user's period selection, so it cannot be a stored
division — it has to be recomputed inside the current filter context against a custom fiscal
calendar.

**Full-cost margin, decomposed.** Burden is not a single number: `medical_burden`, `h1b_burden`,
`marketing_burden`, `burden_perc`, alongside `direct_cost`, `cogs`, `gm_perc`, `gp_per_hour`,
`billed_fte_adj`, `permfee_without_ammort`. **Visa-sponsorship cost (`h1b_burden`) and nationality
are first-class financial attributes** — domain-specific staffing finance that generic BI
candidates will not have touched.

**Two-grain fact design — the actual "dual-path attribution".** The FP&A layer maintains the same
measures at two grains: `..._FPNA_FACT_GLLEVEL_WPNL` keyed on GL transaction date, and
`..._FPNA_FACT_PERIODLEVEL_WPNL` keyed on fiscal period start. Both carry client identity and
producer identity, both are P&L-attributed. A user can drill from period margin to the general
ledger transactions behind it without either fact being compromised to serve both grains.

**Governing the rules across reports.** A single authoritative matrix defines, for **10 named
production reports × 7 producer roles**, who earns AWGP credit, whether direct-hire fees enter GP,
and the amortization window — **6 months live**, 3 months planned. One artifact that a finance
stakeholder and an engineer can both read. This is also the cleanest available answer to
"how many reports?": **10 named production reports** (US and India Commissions Statements,
Ranking & Summits REC and BDM, Client Report REC and BDM, Producers Report REC and BDM,
BH Reporting REC and BDM).

**Résumé framing.** "Designed the revenue-attribution logic for a producer fact grained per
role — a nine-level designation precedence ladder, window-function collision detection, selective
measure zeroing that preserves per-producer visibility, and a 15-column composite-key MERGE — so
gross profit and margin are credited exactly once across up to seven role-holders per placement,
feeding commission statements and P&L reporting." Do **not** attach a revenue or savings figure;
processing financial data is not evidence of financial impact.

---

## Point 3 — The warehouse and its PL/SQL transformation layer

**Scope.** An Oracle Autonomous Data Warehouse with a four-tier schema architecture — `ADWC_LOAD`
(staging) → `ADWC_USER` (curated star schema) → `ADWC_RO` (read-only reporting), each mirrored as
`_UAT` and `_DEV`. **306 distinct schema-qualified objects** and **475 distinct object names**
counted: 84 `FACT_*`, 42 `DIM_*`, 21 `LKP_*`, 13 `RPT_*`, plus source-prefixed families.

**Source systems — eight-plus, not four.** Bullhorn Data Mirror (SQL Server), Bullhorn ODS,
Hiregenics ODS, Oracle E-Business Suite across four instances (prod / UAT / two dev, schemas
`APPS` and a custom `XXACS`), JobDiva/ORCA, Salesforce (opportunity pipeline), the **Coupa API**
(four jobs including a *reverse* write-back integration), Oracle HR extracts for US and India, and
SFTP-landed Excel feeds.

**Code volume.** **39 distinct stored procedures** by name (~34 excluding debug and temp variants),
across four production modules totalling **~30,700 lines** — a mean of roughly 1,000 lines per
procedure, with the largest single procedure at 1,470 lines.

**Techniques verified in the corpus:** `MERGE` upsert (158 uses / 24 files); dynamic SQL via
`EXECUTE IMMEDIATE` (665 / 57) used to make procedures schema-agnostic; window functions
(`OVER (PARTITION BY ...)` 279 / 56, `ROW_NUMBER` 201, `RANK` 35, `LAG`/`LEAD` 21, `LISTAGG` 16);
CTEs; hierarchical `CONNECT BY`; regex functions (142 / 31); table partitioning (14 / 10); index
management around bulk loads; `NVL`/`COALESCE` null handling at 10,038 occurrences across 138
files.

**Two genuinely hard pieces of engineering:**

*Column-level change-data-capture replay.* Rather than trusting a snapshot, the job-order and
submission facts are rebuilt by replaying Bullhorn's own audit-history rows column by column. That
means: branching on column name to know the target datatype; detecting which of **two different
text date formats** a history row used (`%/%/%` versus `____-%-%`) and parsing accordingly;
repairing type drift where `CLIENTCORPORATIONID` and `RESPONSEUSERID` are stored as *strings* in
history but *numbers* in the live table, resolving each back through a lookup; handling a
`'[blank]'` sentinel; escaping quotes before dynamic execution; and wrapping each row in
`EXCEPTION WHEN OTHERS` with the failure logged so **one malformed history row cannot abort the
load**.

*Slowly Changing Dimension Type 2 on transactional facts.* `EFF_START_DATE` / `EFF_END_DATE` /
`IS_CURRENT` end-dating (159 / 114 / 90 references) applied to `FACT_JOBORDER` and
`FACT_JOBSUBMISSION` via correlated-subquery end-dating against their staging twins, with `MERGE`
stamping `LAST_UPDATED_BY` and `LAST_UPDATED_DATE`. Separately, FP&A history uses **48 physical
monthly snapshot tables** spanning Jan 2020 – Dec 2023, with parameterized period-scoped and
year-scoped reload — two different history strategies chosen deliberately per subject area.

**Observability built in, not bolted on.** A shared run-audit table
`CURRENT_DW_LOG_DATA(SP_NAME, SP_STATUS, TABLE_NAME, LOADED, SP_START_DATETIME, SP_END_DATETIME, RECORDS_COUNT, ERROR_MESSAGE)`
is referenced **363 times** — effectively every procedure instruments itself, and the larger
procedures log **per stage** rather than per run. Talend writes its own `CURRENT_LOAD_LOG_DATA`;
row-level change processing writes `Event_log_status`; margin auditing has its own history table.

**Governed manual overrides — the automation story with the most substance.** Every hierarchy
dimension and the FP&A fact carries an automated `_STG` and a human `_STG_MANUAL` twin, with
`manual_load_flag` (295 references) propagated onto the fact so a finance adjustment stays
traceable in the report that shows it. Finance-owned workbooks land on SFTP and are ingested by
dedicated jobs. The result: finance keeps the ability to correct numbers, and the warehouse keeps
the audit trail — instead of corrections living in spreadsheets outside the system.

**Résumé framing.** "Built and operated the Oracle ADW platform behind a US staffing group's
recruiting and finance reporting — 39 PL/SQL load procedures (~30,700 lines) over a four-tier
schema and 42-dimension star model, integrating 8+ source systems, with change-data-capture replay
from ATS audit history, SCD Type 2 on transactional facts, 48-table monthly snapshot history, and
run-level audit logging on every procedure."

---

## Point 4 — Release engineering: one artifact, four environments, zero code edits

**The insight that makes this a strength rather than a chore.** Every environment-specific value —
server, schema, credential, source host, load mode, reporting period — is externalized as a
parameter, at all three layers of the stack. The consequence is that the *same* artifact promotes
from DEV to QA to UAT to PROD without anyone editing code, which is where release errors normally
come from.

| Layer | Mechanism |
|---|---|
| **PL/SQL** | Procedures accept `Schema_Name`, `Load_Type_V`, `Run_Date` and build statements with `EXECUTE IMMEDIATE` (665 uses). `LOAD_TYPE` 246 references, `SCHEMA_NAME` 194. One procedure body runs against `ADWC_LOAD`, `ADWC_USER`, `ADWC_USER_UAT` or `ADWC_USER_DEV` unchanged. |
| **Talend** | **69 distinct job artifacts** as versioned deliverables (`GroupId ggk.demo.jobs`, semantic versions from 0.0.1 up to 0.5.4 on the most-iterated job), deployed through a manifest that injects a JSON `ContextParams` block per environment. |
| **Power BI** | The semantic model exposes `ServerNameParam` plus three required schema parameters and six business control parameters — the same `.pbix` repoints environments without touching a query. |

**The deployment manifest** is a real release artifact with columns `JobActionType`, `Version`,
`ExistingVersion`, `GroupId`, `ArtifactId`, `ContextParams`, `TriggerActionType`, `TriggerId`,
`TriggerCronSyntax`, `Timeout`. Deployed jobs are environment-prefixed — `QA_BI_FTR_*`,
`UAT_BI_FTR_*`, `PROD_BI_*` — so one glance identifies which environment a job belongs to.

**Release discipline that left traces.** A companion tracker lists, **per job**, exactly which
context parameters must change for that environment, with reviewer comments
("Please add read write passwords for BHODS1 environment"). Sibling worksheets are named
`pending uat`, `backup- failed in qa,`, `NotNeeded in Prod`, `PROD Master - Backup`,
`PROD different path` — a QA gate that actually failed things and tracked them. Cutover
instruction, verbatim: *"Please create the trigger and keep in pause. They should not be in active
state."* Pre-change table snapshots (`_BKP<date>`) provided rollback. A dedicated
`Job_Data_Full_Load_Prod_To_Dev` reseeded lower environments from production.

**Orchestration and control jobs, not just extract jobs.** Master/child Execution Plans
(`Master_BH_Load_Job`, `Master_EBS_Inc_Load_Job`, `Master_HGC_Inc_Load_Job`,
`Master_BHODS_ADWC_Inc_Load_Job`, `Master_AdwcLoadSchema_AdwcUserSchema`); full/incremental job
pairs for the same entity; **year-bounded backfill jobs** (`..._FULL_2020`, `..._FULL_2022`,
`Job_T_Person_PartitionFullLoad`, a `partition_year` context parameter) so a historical rebuild
never becomes one monolithic reload; run-metadata jobs (`Job_EP_StartDateTime_New`,
`Job_EP_EndDateTime_New`); a dedicated `Job_Load_Data_Validation` and
`Job_Business_Setup_Data_Differences`; and `Job_Email_Notification_New` for load-status alerting.

**A verification run mode as a first-class load type** — `Run_Type = 'VERIFICATION'` and
`'PRELIMS'`, `Load_Type = 'V'` — meaning the commission pipeline could be run in a non-mutating
check mode before a real run. Combined with fine-grained load flags (`budget_load_flag`,
`actuals_load_flag`, `LoadFlag_MonthlyProc`, `LoadFlag_RegularProc`, `DH_FLAG`,
`Category_Type`, `Business_Unit`, `Commission_DatePeriod`), an operator could rerun exactly the
slice that needed rerunning.

**Connectivity handled properly:** Oracle cloud wallet mutual TLS
(`javax.net.ssl.trustStoreType=SSO`, `cwallet.sso`) to ADW; encrypted JDBC
(`encrypt=true;trustServerCertificate=true;ssl=require`) to SQL Server sources.

**Production schedule, stated accurately:** daily cron Execution Plans at 14:00 and 14:30
Asia/Kolkata, plus one 15-minute CDC job from ODS to the warehouse.

**Résumé framing.** Say "**release engineering and controlled promotion across four environments
by parameter injection**", not "CI/CD pipelines" — there was no Git repository, no build server
and no automated test harness, and an interviewer who asks about branching strategy should not be
met with a gap. The parameterization story is stronger anyway, because most candidates cannot
describe one. **Do not attach a "50% faster deployment" figure** unless Joel can source it; the
mechanism is fully evidenced, the measurement is not.

---

## Point 5 — The Power BI semantic model and dynamic hierarchical security

Recovered by parsing the model file directly, so these are exact counts, not estimates.

| | |
|---|---|
| User tables | **40** |
| Relationships | **187** — 179 many-to-one single-direction, 5 bidirectional, 1 inactive (activated by `USERELATIONSHIP`), 1 many-to-many |
| DAX measures | **354** |
| Calculated columns | **129** |
| Calculation groups | 1 |
| RLS roles | 1, dynamic and hierarchical |
| Power Query queries | 42, including 10 parameters |
| Report pages | **85** |
| Visual containers | **1,035** |
| Compatibility level | 1567 |

**DAX complexity, measured rather than asserted.** Median measure is **13 lines / 395 characters**;
**208 measures exceed 10 lines; 66 exceed 30 lines**; the largest is **159 lines / 8,066
characters**. Function usage across measures and calculated columns: `VAR` 1,678, `CALCULATE`
1,367, `SELECTEDVALUE` 908, `FILTER` 816, `ALL` 500, `ISBLANK` 394, `RETURN` 383, `DATEADD` 234,
`REMOVEFILTERS` 189, `SUMMARIZE` 53, `USERELATIONSHIP` 19. That profile is
filter-context-manipulation-heavy with disciplined variable use — 1,678 `VAR` declarations is a
deliberate practice, not an accident. Time intelligence is hand-rolled with `DATEADD` rather than
`TOTALYTD`, which is the correct choice against a custom holiday-aware fiscal calendar where the
built-in functions do not apply.

**Dynamic hierarchical row-level security — the standout technical detail.** One role, three ideas
in nine lines of DAX: identity taken from the signed-in principal via `USERPRINCIPALNAME()`;
organisational-subtree visibility via `PATHCONTAINS` against a materialized parent-child `PATH`,
so a manager sees their entire reporting line without a recursive join at query time; and a
data-driven `ISSUPERUSER` bypass instead of a second role. The `PATH` column is built and
maintained **in the warehouse** by a dedicated procedure, so the security hierarchy is data that
operations can change, not model code that requires a redeploy.

RLS was designed rather than switched on: a dedicated design workbook with "Design-Analysis,
Thoughts" and "Plan" sheets, an attribute-linked RLS guide, **two competing Power Query
implementations of the client-security table evaluated side by side**, supporting DDL and a
loader procedure, and a separate proof-of-concept report for external client-level RLS.

**A calculation group** — which cannot be authored in Power BI Desktop and required Tabular
Editor (corroborated by a 114 KB Tabular Editor C# script in the same folder) — using
`ISCROSSFILTERED` inside `CALCULATE(..., ALLSELECTED(), ALLEXCEPT(...))` to detect whether a user
filter is active and switch measure behaviour accordingly.

**Deliberate architecture in the Power Query layer.** 31 Oracle sources, several pushing native SQL
down to the database, and only a thin M transformation layer (23 type transforms, 17 row filters,
19 `Text.Trim`, one join) — because the joins belong in PL/SQL where they can be tested and
logged. The reporting date window is computed rather than hardcoded: today's fiscal period end is
resolved from the calendar dimension at refresh time.

**Report craft.** A deliberate information architecture: two persona home pages, then mirrored
BDM/Recruiter pairs of every analytical view (1-on-1, grouped summary, producer's report, details,
actuals vs targets, summary, call details, glossary), then ~40 single-metric drill pages, then
producer financial pages split by role across AWGP, headcount, hours billed, revenue, cost, margin
and billed FTE. 32 bookmarks, 27 action buttons, 73 page-navigation references. Two in-model
glossary tables feed dedicated glossary pages — **self-documenting metric definitions shipped
inside the report**, which is unusually mature and worth calling out.

**Honest caveats to steer around.** Auto date/time was left enabled (115 auto-generated date
tables against 40 real ones), so **do not claim model-size optimization**. There is no incremental
refresh configured — models are full Import — so **do not claim incremental refresh or real-time**.
Four debug/test pages shipped in the UAT file. Every recovered file for this report is named
`..._UAT`, so prefer "delivered to UAT / delivered in phases" over "delivered in production" for
this specific model unless Joel confirms go-live. The models that unambiguously ran in production
are the commission statements, Ranking & Summits, Producers Report, and the APCOB / VSD / DSO
pipelines, which have live production cron triggers.

---

## Point 6 — Owning the Power BI platform across two tenants after an acquisition

**Context.** Innova's acquisition of Volt brought a second, unfamiliar Power BI tenant into scope
alongside the existing ACS/Innova tenant. Joel took it over through a structured knowledge
transfer and then ran both. **This workstream is absent from the current résumé and it is the
single most direct match for BI-platform and analytics-engineering job descriptions.**

**Tenant monitoring built on the Power BI REST API.** A monitoring model authenticates as an Azure
AD **service principal** using the OAuth2 client-credentials flow — a `GetAccessToken()` function
in Power Query M acquiring a bearer token, then `Json.Document(Web.Contents(...))` against the
Power BI REST endpoints — expanding workspace and capacity metadata (`isOnDedicatedCapacity`,
`capacityMigrationStatus`, `state`, `type`) and filtering to active workspaces. A PowerShell path
is documented as an alternative for data-source enumeration. **Five monitoring reports** in total,
covering workspace inventory and dataset refresh status for both tenants, plus CSV and workbook
estate inventories.

**Gateway lifecycle ownership.** An on-premises data gateway upgrade procedure covering admin
prerequisites, a **required VM snapshot before upgrade**, a user communication template, an
expected 1–2 hour downtime window, and — the detail that shows real understanding — the
distinction between gateway-dependent in-network sources (Oracle Financials/HR, the Bullhorn
mirror, the warehouse) and cloud sources that are unaffected. A companion pre- and
post-migration verification procedure covers data-source status testing and gateway cluster
recovery-key handling. *Attribution note for accuracy:* pre-existing Volt documentation was
consolidated and extended rather than written from scratch — the document says so itself. Phrase
as "consolidated and extended", which is still ownership.

**Infrastructure specification.** A dependency audit of the inherited production gateway VM —
Oracle data-access components and client version, the gateway configurator, .NET runtime, the
Azure Monitor agent, and a note that TNS configuration had been placed manually. Then a formal
request for **dedicated production BI infrastructure** (32 GB RAM, 8 cores, 250 GB), argued on its
merits: separate the production gateway from non-production so a development refresh cannot affect
a production report, and support more than two concurrent developers. Joel is named first among
the four users requiring access. He also produced his own effort estimate for the server and
gateway upgrade programme.

**Knowledge transfer in both directions.** *Receiving:* a KT plan with delivery tracking, planned
dates, hosts and recording links, covering workspace and module inventory, gateway setup and
environment, recurring maintenance, refresh-status monitoring, gateway health, gateway upgrades,
cluster admin accounts and recovery keys, and data-source status monitoring. *Giving:* a two-track
technical training curriculum he authored — **Power BI, 42 topics sequenced day by day** (the
mashup and xVelocity engines, connecting to Oracle ADW, file formats including `.pbip`,
gateway-to-dataset mapping, refresh scheduling, Import versus DirectQuery trade-offs, basic *and*
dynamic RLS, drill-through, cross-page filter sync, bookmarks and selection panes, custom visuals)
and **SQL, 33 topics** (relational model, query optimizer and algebrizer, Oracle versus T-SQL
architecture, DDL/DML/DCL/DQL/TCL, set operators, **logical order of query processing**, ranking
and `OVER (PARTITION BY)`, temp tables versus table variables versus CTEs). You do not write that
syllabus unless you can teach it.

**Design authority alongside it.** A versioned High Level Design (v1.3) written explicitly "for
project managers, Development leads and Testing leads", containing the glossary, data-flow,
source-to-target mapping, relationship register and measure register — the document this entire
evidence audit relied on to decode the estate. Paired with a test-case workbook whose test IDs are
referenced directly in the load code, giving requirement → test → implementation traceability.

**Stakeholder troubleshooting.** A sustained run of dated investigations through 2023: cross-system
measure comparisons reconciling the same number computed from two systems, a negative-margin
defect traced from the report back through FP&A to the EBS source, assignment-level variance
tracking, employee-ID mismatch triage, null-designation triage by hierarchy level, and 104 KB of
change notes written two weeks before departure.

**Résumé framing.** "Owned Power BI platform administration across two tenants following an
acquisition — workspace and data-source inventory, on-premises gateway upgrades with pre/post
migration verification, a dedicated production BI server specification, and five monitoring
reports including REST-API-based refresh and capacity tracking via an Azure AD service principal —
and authored the team's High Level Design, test cases and a two-track Power BI and SQL training
curriculum."

---

## Guardrails for the résumé agent

**Safe to state as fact:** every number in this brief. They come from source code, model
internals, deployment manifests and dated design documents.

**Never write these:**

| Do not write | Write instead |
|---|---|
| "real-time" / "streaming" | "daily batch with a 15-minute CDC feed" |
| "incremental refresh in Power BI" | "parameterized incremental loads at the warehouse layer" |
| "CI/CD pipelines" | "release engineering and controlled promotion across four environments" |
| "50% faster deployment" | describe the parameter-injection mechanism; the figure is unsourced |
| "70+ hours/month saved" | describe the governed manual-override path; the figure is unsourced |
| "built forecasting models" | "budget and target versus actuals variance reporting" — no forecasting code exists |
| "optimized model performance" | quote the 187-relationship / 179-single-direction design instead |
| "PL/SQL packages" | "PL/SQL procedures" — there is one package in the whole corpus |
| "partitioning and parallelism tuning" | "table partitioning and index management around bulk loads" — no parallel hints, no materialized views, no stats gathering |
| "led a team of 2–4" | "mentored teammates; authored the team's Power BI and SQL training curriculum" — team size is not evidenced here |
| any revenue, margin or cost-savings impact | describe the mechanism |
| "25+ dashboards, 500+ metrics" | "354 measures in the primary semantic model; 10 named production reports; 85 report pages" |

**Two items only Joel can confirm:** the Innova Idol Award (no artifact exists in these files) and
the size of the reporting-model audience (the security design implies a real hierarchy of users,
but nothing counts them — and workspace or dimension row counts must not be converted into user
counts).

**Suggested ordering** with data engineering at the top: Point 3 (platform and PL/SQL) → Point 1
(identity resolution) → Point 2 (financial attribution) → Point 4 (release engineering) →
Point 5 (semantic model and security) → Point 6 (platform administration and enablement).
