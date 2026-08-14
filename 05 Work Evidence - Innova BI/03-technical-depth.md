---
title: Technical Depth
type: technical-analysis
audit_date: 2026-08-12
tags:
  - evidence/technical-depth
  - work/innova
  - tech/plsql
  - tech/dax
  - tech/power-query
  - tech/talend
  - tech/oracle-adw
---

# 03 - Technical Depth

Back to [[00-audit-summary]] · [[01-project-and-system-map]] · [[02-evidence-ledger]] · [[07-strength-points]]

This note is the technical substance for later amplification. Method notes are stated inline so
every number can be re-derived.

---

## 1. Data architecture

**Shape.** A four-tier Oracle ADW warehouse — `ADWC_LOAD` (staging, source-shaped) →
`ADWC_USER` (curated star schema) → `ADWC_RO` (read-only reporting) — with `_UAT` and `_DEV`
twins of the curated tier. 306 distinct schema-qualified object references were counted across
the corpus; 475 distinct object names in total.

**Modelling patterns actually present:**

- **Conformed dimensions across ATS + HR + financials.** `BHRPT_DIM_USER_ENTITY` reconciles
  Bullhorn corporate users, Hiregenics persons and Oracle HR employees into one producer
  dimension keyed by a composite `CONCAT_JOINKEY` / `CONCATKEY_ROLE_NATIONALITY`.
- **Role-playing facts by physical duplication.** `BHRPT_FACT_PLACEMENT`, `_STARTS`, `_ENDS`
  are three loads of one business event, each with its own date grain, because each needs its
  own active date relationship. `BHRPT_FACT_JOBORDER_PRIMARYUSER` vs `_DATEADDED` is the same
  idea on job orders.
- **Bridge tables for many-to-many.** `BRIDGE_USER` (distinct producers),
  `BRIDGE_BASELINEKEYS` (distinct role/nationality target keys),
  `BRIDGE_PERIOD_AGG_DIMDATE` (period-to-date granularity bridge), `LKP_BRIDGE_CKEYS`.
- **Two-grain financial facts.** `..._FPNA_FACT_GLLEVEL_WPNL` (GL transaction, `GL_DATE`) and
  `..._FPNA_FACT_PERIODLEVEL_WPNL` (fiscal period, `FISCALSTART_DATE`), both P&L-attributed.
  A report can drill from period margin to the GL transactions behind it without one fact table
  trying to serve both grains.
- **Automated / manual staging twins.** Every hierarchy dimension and the FP&A fact has a
  `_STG` and a `_STG_MANUAL`, with `manual_load_flag` carried onto the fact so a manual finance
  adjustment stays visible and attributable downstream.
- **Sentinel-key unmatched handling.** `999999999` and `'N/A'` rather than dropping rows, so
  totals reconcile and orphans are visible instead of silently missing.

**Grain, worked example.** `BHRPT_PRODUCERS_FPNA_ALL` is grained at
`assignment_id × producer (name + person_employeeid) × designation × period_name × activity_code`.
That is *deliberately* finer than "one row per assignment-period" — it must be, because several
producers each hold a role on the same assignment, and each needs to see their own row. The
consequence is that naive `SUM(revenue)` double-counts, which is precisely the problem §5 solves.

---

## 2. ETL and orchestration

**Talend, 69 distinct job artifacts.** Recovered from a deployment manifest, not inferred from
filenames. Structure of the manifest: `JobActionType`, `Version`, `ExistingVersion`, `GroupId`
(`ggk.demo.jobs`), `ArtifactId`, `ContextParams` (a JSON blob injected at deploy time),
`TriggerActionType`, `TriggerId`, `TriggerCronSyntax`, `Timeout`.

**Job taxonomy** (from artifact names):

| Family | Examples | Pattern |
|---|---|---|
| Master orchestrators | `Master_BH_Load_Job`, `Master_EBS_Inc_Load_Job`, `Master_HGC_Inc_Load_Job`, `Master_BHODS_ADWC_Inc_Load_Job`, `Master_AdwcLoadSchema_AdwcUserSchema` | parent jobs sequencing children |
| Full vs incremental pairs | `Job_T_Assignment_Full_Load` / `Job_T_ASSIGNMENT_Inc_Load`; `Full_Load_Job_VSD_Payment_Info` / `Job_Load_VSD_Payment_Info` | same entity, two strategies |
| Year-partitioned reloads | `BH_USER_FACT_JOBORDER_FULL_2020`, `_2022`; `Job_T_Person_PartitionFullLoad` | bounded backfill instead of one giant reload |
| API integrations | `Job_Load_Coupa_API_Expenses_main`, `..._Invoice_combination_main`, `..._Purchase_Order_Combination_main`, `..._Invoice_Reverse_Integration_put_Non_Prod` | REST ingestion **and** write-back |
| SFTP Excel feeds | `Master_ExcelFeed_to_AdwcLoadSchema`, `Job_Load_Excel_to_APCOB_BUDGETS` / `_OpeningCounts` / `_TeamLookup`, `Job_Load_Excel_to_SGnA_ACS_Budgets`, `Load_US_HR_EXTRACT_to_ADWC_LOAD`, `Load_IND_HR_EXTRACT_to_ADWC_LOAD` | governed spreadsheet intake |
| Manual-override intake | `Job_Load_FPnA_Manual_Inserts_Master`, `Job_Load_SGnA_Manual_Inserts_Master`, `Master_ExcelFeed_to_AdwcLoad_Manual_loads` | finance corrections as a pipeline, not a side channel |
| Procedure runners | `Job_Load_SGNA_ProcRun`, `Job_Load_USER_FULL_ProcRun`, `Job_Run_CE_Code`, `Job_Load_User_BHMR_RPT_BHProducers` | Talend as scheduler for PL/SQL |
| Control / observability | `Job_Load_Data_Validation`, `Job_Business_Setup_Data_Differences`, `Job_Email_Notification_New`, `Job_EP_StartDateTime_New`, `Job_EP_EndDateTime_New` | validation and run-metadata as first-class jobs |
| Environment seeding | `Job_Data_Full_Load_Prod_To_Dev`, `ADWC_SRC_PROD_to_ADWC_TGT_Full_Load_Common_Job` | refresh lower environments from prod |
| CDC | `CDC_JOB_FROM_ODS_TO_ADWC` — annotated "15 mints trigger" | the only sub-daily pipeline found |

**Scheduling reality.** Production Execution Plans are daily cron: `PROD_DSO_FACTS_EP` at 14:30
and `PROD_LOAD_APCOB_EP` at 14:00, timezone `Asia/Calcutta` / `Asia/Kolkata`. Plus the 15-minute
CDC job. Nothing streaming, nothing event-driven.

**Two-layer run logging.** Talend writes `CURRENT_LOAD_LOG_DATA`; PL/SQL writes
`CURRENT_DW_LOG_DATA(SP_NAME, SP_STATUS, TABLE_NAME, LOADED, SP_START_DATETIME, SP_END_DATETIME, RECORDS_COUNT, ERROR_MESSAGE)`
— **363 references** across the corpus, i.e. essentially every procedure instruments itself.
Row-level change processing writes `Event_log_status(TableName, ChangeID, StatusMessage, InsertDate)`.
Margin auditing has its own history table, `MARGINSAUDIT_CONSOLIDATED_HIST`.

**Load-mode control is uniform across all three layers** — this is the single most
architecturally coherent thing in the whole estate:

| Layer | Mechanism |
|---|---|
| Talend | `LoadFlag`, `LoadFlag_BHMergeTable:FULL`, `LoadFlag_APCOBFact:INC`, `LoadFlag_MonthlyProc`, `LoadFlag_RegularProc`, `budget_load_flag`, `actuals_load_flag`, `DH_FLAG`, `partition_year`, `Run_Type` (`VERIFICATION` / `PRELIMS`), `Load_Type` (`FULL`/`INC`/`C`/`V`) |
| PL/SQL | `Load_Type_V IN ('INC', ...)`, `Schema_Name`, `Run_Date`, `LAST_RUN_DATE_TIME` (`LOAD_TYPE` 246 refs, `SCHEMA_NAME` 194) |
| Power Query | `LoadSchemaName`, `UserSchemaName`, `ROSchemaName`, `ServerNameParam`, plus 6 business control parameters |

---

## 3. SQL and PL/SQL

**Census method.** 509 `.sql`/`.pls`/`.txt` files from the employment clusters were deduplicated
by MD5 to 393 unique files (9.9 MB), copied to a scratch directory, and pattern-counted with
case-insensitive regex. Counts below are (files containing / total occurrences).

| Technique | Present? | Evidence |
|---|---|---|
| Stored procedures | ✅ | 39 distinct names; `CREATE OR REPLACE PROCEDURE` 75 / 48 files |
| Packages | ⚠️ marginal | 1 occurrence. **Do not claim package-based design** |
| Views | ⚠️ | 2 `CREATE OR REPLACE VIEW`; plus `EBS_AR_CASH_PERM_VIEW` referenced in job names |
| `MERGE` upsert | ✅ | 158 / 24 files |
| Dynamic SQL | ✅ heavily | `EXECUTE IMMEDIATE` 665 / 57 files — used for schema-parameterized DDL/DML |
| Cursors | ✅ | 65 / 33 files |
| `BULK COLLECT` / `FORALL` | ❌ | 0. Row-by-row cursor loops instead |
| Window functions | ✅ | `OVER (PARTITION BY ...)` 279 / 56; `ROW_NUMBER` 98 / 31; `LAG`/`LEAD` 21 / 6; `LISTAGG` 16 / 4 |
| CTEs | ✅ | 12 / 9 files (plus many un-aliased `WITH` forms) |
| `PIVOT` | ✅ light | 3 / 2 |
| Hierarchies | ✅ | `CONNECT BY` 8 / 5 |
| Regex | ✅ | `REGEXP_LIKE/REPLACE/SUBSTR/INSTR/COUNT` 142 / 31 |
| Fuzzy matching (functions) | ❌ | `UTL_MATCH` 0, `SOUNDEX` 0, `Table.FuzzyNestedJoin` 0 |
| **Fuzzy matching (join construction)** | ✅ **heavily** | **the actual technique** — 8,918 join predicates, 1,609 (18%) on transformed strings; 5,527 `LEFT JOIN`s / 207 files; **199 OR-laddered string predicates, 66 of them six-way**; 9,593 `‖` concatenations. See §5 |
| SCD2 columns | ✅ | `EFF_START_DATE` 159, `EFF_END_DATE` 114, `IS_CURRENT` 90, `START_DATE_ACTIVE`/`END_DATE_ACTIVE` 24 each |
| Dedupe logic | ✅ | 183 / 55 files |
| Null handling | ✅ pervasive | `NVL`/`NVL2`/`COALESCE` 10,038 / 138 files |
| Partitioning | ✅ | `PARTITION BY RANGE/LIST/HASH` 14 / 10 |
| Indexing | ✅ | `CREATE INDEX` 30 / 13; index drop/recreate around bulk load (commented) |
| Materialized views | ❌ | 0 |
| Optimizer hints | ❌ | `PARALLEL` 0, `APPEND` 0 |
| Stats gathering | ❌ | `GATHER_TABLE_STATS` 0 |
| Structured exceptions | ⚠️ thin | `EXCEPTION WHEN` 3 / 2; `SQLERRM` 58 / 32; `RAISE_APPLICATION_ERROR` 6 / 2; `DBMS_OUTPUT` 413 / 46 |

**Complexity estimate, with method.** The four production dumps are 9,157 + 12,154 + 8,392 + 986
= **30,689 lines** carrying 30 procedure definitions, i.e. a mean of ~1,020 lines per procedure.
`SP_LOAD_Fact_JobOrder` alone spans 1,470 lines. This is a *size* measurement and is offered as
such — it is not converted into an impact claim.

### The hardest single piece of logic: column-level CDC replay

`SP_LOAD_Fact_JobOrder` reconstructs current state from Bullhorn's audit/history tables rather
than trusting a snapshot. Per changed column it must:

1. Branch on column name to know the target datatype
   (`IF UPPER(v_ColName) IN ('STARTDATE','DATEEND','CUSTOMDATE1','CUSTOMDATE2','CUSTOMDATE3')`).
2. Detect which of two date formats the history row used —
   `LIKE '%/%/%'` vs `LIKE '____-%-%'` — and parse accordingly. The history table stores dates
   as text with inconsistent formats.
3. Repair type drift: `CLIENTCORPORATIONID` and `RESPONSEUSERID` are stored as *strings* in
   history but *numbers* in the live table, so the old value is resolved back to an ID through a
   lookup via `EXECUTE IMMEDIATE (QUERY1) INTO v_correctvalue USING v_Oldvalue`.
4. Handle the `'[blank]'` sentinel and escape embedded quotes before injecting into dynamic SQL.
5. Log the outcome per row to `Event_log_status`, and on `EXCEPTION WHEN OTHERS` log and
   continue — one malformed history row cannot abort a 1,470-line load.

Then SCD2 end-dating: `UPDATE FACT_JOBORDER A SET EFF_END_DATE = (SELECT MIN(EFF_START_DATE) FROM FACT_JOBORDER_STAGE B WHERE A.JOBORDERID = B.JOBORDERID)`
followed by the `IS_CURRENT` flag pass, and a `MERGE` stamping `LAST_UPDATED_BY = USER` and
`LAST_UPDATED_DATE = SYSDATE`.

---

## 4. Power BI, DAX and semantic modelling

All figures below are parsed from the `DataModelSchema` and `Report/Layout` inside
`BH_Reporting_UAT - Rec 2109 (ClientRemodelling R2).pbit` (2023-09-26). No Power BI process was
launched; the `.pbit` is a ZIP and was read as one.

| Metric | Value | Note |
|---|---|---|
| Compatibility level | 1567 | modern tabular |
| Tables | **40 user** (+ 115 auto date) | quote the 40 |
| Relationships | **187** | 179 many-to-one single-direction, 5 bidirectional, 1 one-to-one bidirectional, 1 inactive, 1 many-to-many |
| Measures | **354** | 258 on `BHRPT_DIM_USER_ENTITY`, 44 on `BHRPT_PRODUCERS_FPNA_ALL`, 15 on `BHRPT_CALLS_RECRUITERS` |
| Calculated columns (user) | **129** | plus 690 auto date columns |
| Hierarchies | 115 | mostly auto date |
| Calculation groups | 1 | `UserDefaultCalcGroup` |
| RLS roles | 1 | dynamic, hierarchical |
| M queries | 42 | incl. 10 parameters |
| Report pages | **85** | |
| Visual containers | **1,035** | 598 slicers, 144 cards, 129 images, 72 tables, 27 textboxes, 27 action buttons, 21 shapes, 17 matrices |
| Custom visuals | 6+ | advanceCard, Gantt, heatmap, LinearGauge, multiKpi, … |
| Bookmarks / page-nav refs | 32 / 73 | |
| Package size | 4.0 MB (`.pbit`) / ~520 MB (`.pbix`) | template excludes data |

**DAX complexity, measured.** Over all 354 measure expressions: median **395 characters / 13
lines**, mean 779 characters, maximum **8,066 characters / 159 lines**. **208 measures exceed 10
lines; 66 exceed 30 lines.** Method: character and newline counts on the model's stored
expression strings.

**Function usage** (occurrences across measures + calculated columns):

```
VAR 1678   CALCULATE 1367   SELECTEDVALUE 908   FILTER 816   ALL 500
ISBLANK 394   RETURN 383   DATEADD 234   REMOVEFILTERS 189   FORMAT 70
DIVIDE 58   SUMMARIZE 53   COUNTX 23   DISTINCT 21   USERELATIONSHIP 19
SUMX 17   ADDCOLUMNS 13   CONCATENATEX 11   SWITCH 8   RELATED 8   EARLIER 8
ALLEXCEPT 6   EXCEPT 5   ALLSELECTED 4   LOOKUPVALUE 4   MAXX 3
COALESCE 2   DATESBETWEEN 2   PATH 1   ISFILTERED 1
```

Reading that honestly: this is **filter-context-manipulation-heavy DAX** — `CALCULATE` +
`FILTER` + `ALL`/`REMOVEFILTERS` + `SELECTEDVALUE`, with `VAR`/`RETURN` used pervasively
(1,678 `VAR` declarations is best-practice discipline, not incidental). Time intelligence is
mostly hand-rolled with `DATEADD` (234) rather than `TOTALYTD`/`SAMEPERIODLASTYEAR`, which is
consistent with a custom 4-4-5-style fiscal calendar (`DIM_FIN_PERIODDATE`, `PERIOD_NAME`,
`YEAR_QUARTER`, `FISCALSTART_DATE`, `WEEKEND_DATE`) where the built-in functions don't apply.
`EARLIER` (8) is legacy style. Iterators are used sparingly, which is good for a model this size.

**Dynamic hierarchical RLS** — one role, `UserRole`, filtering `BHRPT_USEREMAILMAPPING_RLS`:

```
If(
  MaxX(Filter('BHRPT_USEREMAILMAPPING_RLS',[EMAIL] = USERPRINCIPALNAME()),
       'BHRPT_USEREMAILMAPPING_RLS'[ISSUPERUSER]) = 0,
  PATHCONTAINS('BHRPT_USEREMAILMAPPING_RLS'[PATH],
               MaxX(Filter('BHRPT_USEREMAILMAPPING_RLS',[EMAIL] = USERPRINCIPALNAME()),
                    'BHRPT_USEREMAILMAPPING_RLS'[NAME])),
  1=1
)
```

Three ideas in nine lines: identity from the signed-in principal; org-subtree visibility via a
materialized parent-child `PATH` so a manager sees their whole reporting line without a recursive
join at query time; and a data-driven super-user escape hatch instead of a second role. The `PATH`
column is built in the warehouse (`SP_LOAD_BHRPT_CLIENTRLS_EMAIL_MAP`,
`BHRPT_CLIENTMAP_RLS_DB`), so the security hierarchy is maintained as data, not as model code.

**Calculation group** `UserDefaultCalcGroup` — two items. `No Filter` = `SELECTEDMEASURE()`;
`Yes Filter` wraps `ISCROSSFILTERED('BHRPT_DIM_USER_ENTITY'[CONCAT_JOINKEY])` inside
`CALCULATE(..., ALLSELECTED(), ALLEXCEPT('DIM_DATE','DIM_DATE'[FULL_DATE]))` to detect whether a
user filter is active and switch behaviour accordingly. Calculation groups cannot be authored in
Power BI Desktop — this required Tabular Editor, corroborated by the 114 KB
`TabularEdt CScript - Exp info.txt`.

**Power Query design.** 42 shared queries, 31 `Oracle.Database` sources, 10 parameters
(`ServerNameParam` + three schema parameters + six business control parameters:
`PLACEMENT_DESGN_CTRL`, `JOBORDER_DESGN_CTRL`, `FPNA_DESGN_CTRL`, `DATE_CTRL_SUBS`,
`DATE_CTRL_JOBS`, `DATE_CTRL_NOTES`). Transformation in M is deliberately thin —
`Table.TransformColumnTypes` 23, `Table.SelectRows` 17, `Text.Trim` 19, `Text.Upper` 11,
`Table.Distinct` 6, one `Table.NestedJoin`, one `Table.UnpivotOtherColumns`, one
`List.Accumulate` — because the joins live in PL/SQL and some queries push native SQL through
`Oracle.Database(server, [Query=...])`. One nice touch: the date window is computed, not
hardcoded — `DIM_DATE_CURR_ENDDATE` resolves today's fiscal period end from
`DIM_FIN_PERIODDATE` via `DateTime.LocalNow()`, and `DIM_DATE` filters
`>= #date(2022,01,01) and <= Table.FirstValue(DIM_DATE_CURR_ENDDATE)`.

**Report craft.** 85 pages in a deliberate information architecture: two persona home pages, then
mirrored BDM/Recruiter pairs for each analytical view (1-On-1, Grouped Summary, Producer's Report,
Details, Actuals vs Targets, Summary, Call Details, Glossary), then ~40 single-metric drill pages,
then Producer financial pages split REC/Sales across AWGP, headcount/starts, hours billed, total
revenue, total cost, total margin and billed FTE. Two in-model glossary tables
(`BHRPT_GLOSSARY_REC`, `BHRPT_GLOSSARY_BDM`) feed dedicated glossary pages — self-documenting
metric definitions inside the report, which is unusually mature.

**Honest weaknesses.** Auto date/time enabled (115 `LocalDateTable_*`), 4 debug/test pages
shipped in the UAT file, 598 slicers indicating duplicated filter panes rather than synced
slicers, and measure names carrying working notes (`#x(NotUsed) FTE Fill Ratio rolling 6`,
`To Date HeadcountRec_Don't use`, `#CC_Debug_RecBillables_HGC`). These are exactly the things to
name as "what I'd rebuild differently" rather than hide.

---

## 5. Data quality and reconciliation

**The crediting-once algorithm, in full.** The problem: `BHRPT_PRODUCERS_FPNA_ALL` is grained per
producer-role, so one assignment-period appears several times (a recruiter, a recruiter override,
a BDM, a CSM). Summing revenue across producers multiplies it. Dropping rows loses each
producer's own scorecard. The solution keeps both:

```sql
-- 1. an explicit precedence ladder, as data
with tmp_desgn_ranks(designation, index_des) as (
  select 'REC',1 from dual union select 'RECOVR',2 from dual union
  select 'BDM',3 from dual union select 'BDMOVR',4 from dual union
  select 'CSM',5 from dual union select 'SECREC',6 from dual union
  select 'TERREC',7 from dual union select 'SECBDM',8 from dual union
  select 'TERBDM',9 from dual )
-- 2. detect and rank the collision
count(*)     over (partition by assignment_id, name, person_employeeid,
                                activity_code, period_name)              as identifier_multroles,
row_number() over (partition by assignment_id, name, person_employeeid,
                                activity_code, period_name
                   order by index_des)                                   as ranker_multroles
-- 3. zero the ~22 financial measures on every non-primary row
case when ranker_multroles > 1 then 0 else revenue end as revenue,   -- and gp, awgp, burden,
                                                                     -- total_margin, cogs, ...
-- 4. merge the corrected rows back on a 15-column composite key
MERGE INTO BHRPT_PRODUCERS_FPNA_ALL fct USING (...) crd_lkp
ON ( fct.ASSIGNMENT_ID = crd_lkp.ASSIGNMENT_ID AND fct.DESIGNATION = ...
     AND fct.PERIOD_NAME = ... AND fct.activity_code = ... AND fct.max_expenditure_id = ...
     AND fct.manual_load_flag = ... )
WHEN MATCHED THEN UPDATE SET fct.revenue = crd_lkp.revenue, ...
```

Three design decisions worth naming: precedence is **data, not code** (a new designation is a
row, not a deployment); non-primary rows are **zeroed, not deleted**, so every producer still
appears in their own scorecard while the financial rollup is correct exactly once; and the merge
key includes `manual_load_flag` and `max_expenditure_id` so manually adjusted and
system-sourced rows never collide. A source comment records `341,711 rows inserted` and
`37129 test/ house assignment and null ID` excluded.

### Entity resolution — cascading fuzzy string joins

**Corrected 2026-08-12.** An earlier pass of this audit concluded the matching was purely
deterministic, on the grounds that no Oracle fuzzy *function* appears anywhere. That was the wrong
test. The fuzziness lives in **how the joins are constructed**, not in a library call — and by that
measure this is one of the most fuzzy-join-heavy codebases you will see.

**Measured over the 393-file corpus:**

| | |
|---|---|
| Join predicates parsed | **8,918** |
| …joining on a transformed string rather than a key | **1,609 (18%)** |
| `LEFT JOIN` (the cascade primitive) | **5,527 across 207 files** — heaviest procedures carry 178, 158, 145, 136 |
| Multi-alternative OR-laddered string predicates | **199** |
| …distribution by alternatives per ladder | **6-way: 66** · 5-way: 25 · 4-way: 11 · 3-way: 19 · 8-way: 2 · 7-way: 1 |
| `‖` string concatenations (composite key building) | **9,593** |
| String functions inside `ON` clauses | `UPPER` 1,427 · `TRIM` 863 · `NVL` 405 · `‖` 392 · `LIKE` 249 · `LOWER` 91 · `SUBSTR` 20 · `INITCAP` 12 · `INSTR` 11 |

**The six-way ladder is the dominant form — 66 instances — which is precisely what "6+ fuzzy
matching patterns" refers to.**

**The match strategies, each materialized as a join column** so a round can be reasoned about,
tested and reused:

| Strategy | Column family | Refs | Handles |
|---|---|---|---|
| `first + last` | `*_USERNAMEJOINFIELD` | 626 | base case |
| `last + first` | `*_LASTFIRSTNAME_JOINFIELD` | 254 | systems storing names reversed |
| with middle name | `COMMON_MIDDLENAMEFIELD` | 2,530 (`middlename`) | one side carries a middle name |
| preferred/display name | `CEHR_PREFNAME` | 113 | HR preferred vs legal name |
| case-normalized variants | `COMMON_USERNAMEFIELD_CCASE`, `COMMON_LFNAMEFIELD_CASE` | 165 | initcap vs upper vs source casing |
| employee-ID crosswalk | `COMMON_EMPLOYEEID` | 91 | where an ID exists on both sides |
| provenance-prefixed composite | `'DCH '‖id`, `'BH '‖id`, `CONCATKEY_DCHCLIENTS` | 235+ | key collisions across systems made impossible |
| designation + department | `COMMON_DESGN_DEPT` | 80 | role classification input |
| role + nationality | `CONCATKEY_ROLE_NATN` | 27 | one person, two roles, two target rows |
| conformed output vocabulary | `COMMON_*` | 1,780 | what the rest of the warehouse joins on |

**A real four-alternative predicate**, including explicit name-order reversal:

```sql
   TRIM(UPPER(NVL(HGC_USERNAMEJOINFIELD,'N/A'))) = TRIM(UPPER(CEHR_PREFNAME))
OR TRIM(UPPER(NVL(BH_USERNAMEJOINFIELD ,'N/A'))) = TRIM(UPPER(CEHR_PREFNAME))
OR TRIM(UPPER(BH_FIRSTNAME)) ‖ TRIM(UPPER(BH_LASTNAME))  = TRIM(UPPER(CEHR_LASTNAME)) ‖ TRIM(UPPER(CEHR_FIRSTNAME))
OR TRIM(UPPER(HGC_FIRSTNAME))‖ TRIM(UPPER(HGC_LASTNAME)) = TRIM(UPPER(CEHR_LASTNAME)) ‖ TRIM(UPPER(CEHR_FIRSTNAME))
```

The `NVL(..., 'N/A')` guard (128 uses) is deliberate: it stops a NULL on either side from silently
matching, which is the classic way a loose join quietly corrupts a dimension.

**The cascade, as materialized stages** (`SP_LOAD_BHMR_BHHGC_USERENTITY`, 1,469 lines) — each round
lands in its own table, so intermediate results stay inspectable and any stage is independently
re-runnable:

```
LKP_HGC_PERSON_DESIGNATIONS        (L127)  designation vocabulary
   → BHMR_HGC_EXCLUSIVE_BH_TBL     (L180)  the set only one system knows about
   → BHMR_HGC_UNIONTBL             (L289)  combined population
   → LKP_HGC_DISTINCTNAMES_LF      (L344)  distinct last-first name forms
   → BHMR_HGC_TO_CEHR_NEWUSER      (L358)  the crosswalk  ← the OR-ladder match lands here
   → 4 sequential refinement UPDATEs (L482-499)
   → BHRPT_DIM_BHHGC_USERENT_PNL   (L537)  conformed producer dimension
   → 7 downstream _WPNL facts      (L645-1232)
```

with **12 `CURRENT_DW_LOG_DATA` inserts on entry and 12+ updates on exit — one per stage**. So the
logging granularity noted in §2 is finer than "per procedure": each cascade round is independently
observable, which is what makes a 1,469-line matcher debuggable at all.

**Resolving what loose joins produce.** A fuzzy join returns either too few rows (residue) or too
many (duplication). Both are handled explicitly rather than hoped away:

| Concern | Technique | Volume |
|---|---|---|
| detect collisions | `ROW_NUMBER() OVER`, `COUNT(*) OVER`, `RANK() OVER` | 201 / 127 / 35 |
| assert uniqueness | `HAVING COUNT(*) > 1` probes beside the load code | 277 / 64 files |
| compute the residue | `MINUS`, `NOT EXISTS`, `NOT IN (SELECT …)` | 110 / 38 / 62 |
| prevent double-counting across rounds | anti-joined `UNION ALL` | — |
| keep orphans visible | sentinel keys `999999999`, `'N/A'` | — |
| exclude junk | `(N/A) - NOT AVAILABLE`, `HOUSE, *` name filters, `BHMR_DUMMY_DATA` list | — |
| retain human decisions | `MANUALMAP_BH_USERID` override column | 128 |

**Ambiguity as a deliberate business rule.** Role classification cascades over the crosswalk:
department like `%RECRUIT%` → recruiter flag; like `%SALES%` → BDM flag; **neither → set both
flags**; customer-success managers flagged by name against a three-way `UNION` of distinct CSM
names across `fact_joborder`, `bhmr_joborder` and `bhmr_placement`. Unclassifiable people are
credited into *both* populations and the crediting ladder above arbitrates which one actually earns.
Over-match first, disambiguate with rules second — the same philosophy as the joins themselves.

**Identity sources spanned** (from join aliases): Bullhorn corporate (`bhcrp`), Hiregenics
(`hgc_R`, `hgc_p`, `hgc_sp`), JobDiva/ORCA staging (`orcastg`), **Azure AD**
(`azd.GIVENNAME‖' '‖azd.SURNAME`, via an `AzureDataAcs_*` extract), Oracle CE-HR (`cehr_x`), and
Oracle EBS (`DIM_CUST_HIER`). Six systems, no shared key.

**Client-side counterpart.** The client matcher is the simpler sibling — normalized exact match,
anti-joined union, curated standardization table for the residue, recorded working figures of
2,206 matched Bullhorn client entries → 1,476 distinct EBS customers out of 4,703. Documented
design intent worth quoting: the Bullhorn name wins as display name because the slicer is
recruiter-facing, while the relationship to the fact travels on the EBS key because the fact is
financial.

**What remains unsourced:** the **88% accuracy** figure. The artifacts record match *counts* and
residue arithmetic, not a certified accuracy measurement.

**Validation and reconciliation practice** (this is where Joel's personal fingerprint is
heaviest — dozens of dated comparison files):

- `having count(*) > 1` duplicate probes written directly beneath the load code, so the check
  travels with the change.
- Cross-system comparison workbooks: `CEHR to HGC,BH Comparison.xlsx`,
  `MichaelAgulair Comparison_BhrptFpna.xlsx`, `BHRPTvsPrdcr.xlsx`, `FPNAvsComm.txt`,
  `BDM SalesPrdcr - BHRPT analysiQry.txt` — the same measure computed from two systems and
  differenced.
- Root-cause investigations: `Ebs Source, Fpna, Milestone Debug -May23 NegativeMargin.txt`
  (a negative-margin defect traced back through FP&A to the EBS source),
  `BHCorpUser EMPID Issues 2006.xlsx`, `FieldHier_RO_DesignationWiseNulls_2604.xlsx`
  (null-designation triage by hierarchy level).
- Validation scripts as artifacts: `BHValidationPROD_1802.sql` (139 KB),
  `BH_RepMeasures_DEBVALIDS.sql` (89 KB), `dso_revChecks.sql`, `sngaChecks_504.sql`,
  `HistChecksDSOsql`, `BHTestQueries_07012023.sql` (111 KB).
- Pipeline-level: `Job_Load_Data_Validation`, `Job_Business_Setup_Data_Differences`,
  `Schema_toValidate` context parameter, and a `Run_Type='VERIFICATION'` load mode.
- `_BKP<date>` table snapshots before schema or logic changes
  (`DIM_FIN_PERIODDATE_BKP2208`, `FACT_JOBORDER_BKP_050123_MJ`,
  `FACT_SGNA_SUMMARY_BKPCR190422`, `DIM_ASSIGNMENT_ID_TASK_ID_BKP3011`) — informal, but a real
  rollback practice.

---

## 6. Forecasting and financial logic

**AWGP — the core metric**, defined verbatim in the rule matrix:

> AWGP = Total Margin or GP (based on timesheets from Oracle) ÷ number of fiscal weeks adjusted
> for holidays (based on the period selection)

The denominator is the interesting part: fiscal weeks *adjusted for holidays*, varying with the
user's period selection. That forces a holiday-aware fiscal calendar
(`DIM_FIN_PERIODDATE` with `WEEKEND_DATE`, `PERIOD_NAME`, `START_DATE`, `END_DATE`) and means the
measure cannot be a simple division — it must recompute the denominator inside the current filter
context, which is consistent with the `SELECTEDVALUE` (908) / `ALLEXCEPT` / `ISCROSSFILTERED`
patterns in the DAX.

**Attribution rules, per report** (`Power BI reports - AWGP logics - Production Stats.xlsx`,
2023-05-03) — a 10 × 7 matrix of which roles earn credit, plus:

| Report | DH in GP? | Amortization |
|---|---|---|
| US Commissions Statement | all 7 roles present | — |
| Ind Commissions Statement | restricted role set | — |
| Ranking & Summits (REC) 2022 | yes | 6 months; override data limited to LOB in Staffing, VMS Staffing, HBS |
| Ranking & Summits (BDM) 2022 | yes | 6 months; same LOB restriction |
| Client Report (REC / BDM) | — | new model in UAT: overrides + secondary amounts |
| Producers Report (REC / BDM) | separate Direct Hire fee metric | 6 months; "AWGP strictly takes GP where category = Non Perm" |
| BH Reporting (REC / BDM) | — | 3-month amortization **planned**, in UAT |

**Full-cost margin.** Burden is decomposed, not aggregate: `medical_burden`, `h1b_burden`,
`marketing_burden`, `burden_perc`, alongside `direct_cost`, `cogs`, `permfee_without_ammort`,
`billed_fte_adj`, `gm_perc`, `gp_per_hour`. `h1b_burden` and `cehr_nationality` as
first-class financial attributes mean visa-sponsorship cost was modelled into margin — a genuinely
domain-specific piece of staffing finance.

**Eligibility and hierarchy logic.** `LKP_ELIG` (commission eligibility), `LKP_SUB_SERV`,
`JOB_LOAD_GGK_RATECARD`, `India_Team_Lead_Size.xlsx` (team-size-driven India commission),
`LKP_HGC_PERSON_DESIGNATIONS`, `DIM_ORG_HIER` / `DIM_FIELD_HIER` / `DIM_CUST_HIER` three-way
hierarchy set with `_STG` and `_STG_MANUAL` twins.

**Budget vs actuals.** `SP_LOAD_RPT_SGNA(actuals_load_flag VARCHAR, ...)` with orchestration
flags `budget_load_flag`, `actuals_load_flag`, `budget_year`, `LoadFlag_MonthlyProc`,
`LoadFlag_RegularProc`; variance materialized by `SP_SGNA_INSERT_VARIANCES_MONTHLY` and
`SP_SGNA_INSERT_TOTALS_MONTHLY` with parameterized category-level rollup
(`SP_temp_Insert_Variances(Category_var, Category_name, L2_specifier)`).

**Forecasting — the honest position.** No statistical forecasting, regression, trend
extrapolation or projection procedure exists anywhere in 393 files. What exists is
**budget/target vs actuals variance reporting**: `LKP_BASELINE_BH` / `LKP_BH_BASELINES` hold
per-person, per-period targets; `BHRPT_FACT_USER_SUMMARIZED_METRICS` holds actuals; the
"Actuals vs Targets" and "BDM/Rec Baselines" report pages compare them. Call it target-versus-
actuals variance analysis. See [[05-questions-for-joel]] Q2.

---

## 7. CI/CD and release engineering

The strongest under-told story. Three layers share **one** promotion mechanism — externalize
every environment-specific value as a parameter, so the *same artifact* moves DEV → QA → UAT →
PROD:

1. **PL/SQL** — procedures take `Schema_Name` and build statements with `EXECUTE IMMEDIATE`
   (665 occurrences), so one procedure body runs against `ADWC_LOAD` / `ADWC_USER` /
   `ADWC_USER_UAT` / `ADWC_USER_DEV` unchanged.
2. **Talend** — jobs are versioned artifacts (`GroupId ggk.demo.jobs`, semantic versions
   0.0.1 → 0.1.3) deployed via a manifest that injects `ContextParams` per environment. The
   companion tracker lists, **per job**, exactly which parameters must change
   (`ADWC_LOAD_PASSWORD, ADWC_USER_PASSWORD, BHODS_Password, ADWC_LOAD_URL`) with reviewer
   comments.
3. **Power BI** — the semantic model exposes `ServerNameParam` plus three required schema
   parameters, so the same `.pbix`/`.pbit` repoints without editing a single query.

**Environment naming convention as a control**: `QA_BI_FTR_<Artifact>`,
`UAT_BI_FTR_<Artifact>`, `PROD_BI_<Artifact>` — one glance tells you which environment a
deployed job belongs to.

**Gates and rollback that actually left traces**: sheets literally named
`backup- failed in qa,`, `pending uat`, `NotNeeded in Prod`, `PROD Master - Backup`,
`CM teamwork QA`, `PROD different path`; `_BKP<date>` table snapshots before change;
`Job_Data_Full_Load_Prod_To_Dev` to reseed lower environments; and the cutover instruction
*"Please create the trigger and keep in pause. They should not be in active state."*

**What is not there.** No Git repository, no Jenkins/Azure DevOps pipeline definition, no
`.yml`, no automated test harness, no schema-migration tool. Version control is filename
discipline (`v1.4`, `v1.62(base)`, `v1.63 v2`, `v1.64`, `HLD v1.3`, dated `_BKP` tables) plus
Talend's artifact versioning. **The claim to make is "release engineering and controlled
promotion across four environments", not "CI/CD pipelines".**

---

## 8. Performance and scale

Stated conservatively, because the evidence is thin here.

**Supported:**
- Model scale: a ~520 MB PBIX with 40 user tables, 187 relationships and 354 measures.
- Code scale: ~30,700 lines across four production procedure dumps.
- History depth: 48 monthly snapshot tables spanning 4 years (2020–2023).
- Row-count signals from source comments: `341,711 rows inserted` (crediting pass);
  4,703 EBS customers / 2,206 matched Bullhorn client entries (client dimension);
  `37,129` test/house rows excluded.
- Deliberate techniques: `MERGE` instead of delete+insert (158 uses), table partitioning
  (14 uses), index management around bulk load, year-bounded backfill jobs
  (`..._FULL_2020`, `..._FULL_2022`, `partition_year`) rather than one monolithic reload,
  `TRUNCATE` + rebuild on staging (439 uses) rather than row-wise deletes.
- Infrastructure sizing owned: a 32 GB / 8-core / 250 GB production BI server specified and
  requested, explicitly to separate production from non-production gateway load and support
  concurrent development.

**Not supported — do not claim:** query runtime improvements, percentage speed-ups, data volume
in GB/TB or row counts for the warehouse as a whole, concurrent user counts, refresh duration,
parallelism, materialized-view acceleration, or statistics-driven tuning. There is no benchmark,
no timing log, and no before/after measurement anywhere in the drive.

---

## 9. Stakeholder and leadership evidence

**Design authority.** `BH Reporting Phase 2 - HLD v1.3` — a versioned design document written
for "project managers, Development leads and Testing leads", containing the glossary that this
entire audit relied on to decode the estate. Version 1.3 implies at least two review rounds.
Paired with `HLD, TestCases for Phase2 - Cred, Splits.xlsx`, and test-case IDs referenced
directly in the code (`-- tstcase : 174475 ( rec, recovr ) ,, 213704 ( rec bdm )`) — requirement
→ test → implementation traceability that Joel maintained himself.

**Business translation.** The AWGP matrix is the clearest artifact of this: a single sheet that
lets a finance stakeholder see, across 10 reports and 7 roles, exactly who gets credited and how
direct-hire fees amortize. That is a technologist writing *for* the business, not documenting for
themselves. Likewise the in-model glossary tables and dedicated glossary report pages.

**Infrastructure ownership.** The server request document specifies purpose, configuration,
named users and required software, and argues the production/non-production separation on its
merits. The dependency audit shows someone inventorying an inherited machine before touching it.
The gateway procedures include user communication templates and downtime expectations.

**Knowledge transfer, both directions.** Receiving: the Volt Power BI administration KT plan with
delivery tracking, planned dates, hosts and recording links. Giving: a two-track training
curriculum (Power BI 42 rows, SQL 33 rows) sequenced by day, teaching not just tool mechanics but
concepts — the mashup and xVelocity engines, Import vs DirectQuery trade-offs, dynamic RLS,
query optimizer and algebrizer, logical order of query processing. You do not write that syllabus
unless you understand the material well enough to teach it.

**Troubleshooting under stakeholder pressure.** A run of dated investigation files through 2023
— named-stakeholder discrepancy comparisons, a negative-margin root cause, assignment-level
variance tracking, `Nov13 BHRPT changes.txt` (104 KB of change notes two weeks before departure),
`ProdBHProducerCode_Before100%Change.txt` (a preserved before-state for a credit-rule change).

**Where the evidence stops.** No performance review, no org chart, no award record, no
delegation artifact. Mentoring of 2–4 people and 35+ interviews remain **plausible but
unevidenced** on this drive. The Innova Idol Award has **no artifact at all** here.
