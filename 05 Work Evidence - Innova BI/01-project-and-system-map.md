---
title: Project and System Map
type: system-map
audit_date: 2026-08-12
tags:
  - evidence/system-map
  - work/innova
  - domain/data-warehouse
  - tech/oracle-adw
  - tech/talend
  - tech/power-bi
---

# 01 - Project and System Map

Back to [[00-audit-summary]] · evidence IDs resolve in [[02-evidence-ledger]] ·
technique detail in [[03-technical-depth]]

All paths are relative to the source root
`.../Work Files - MJS BI/GGK WorkSample Assets/`. Hostnames, IPs, usernames and passwords from
Talend context parameters are generalized throughout.

---

## Platform context shared by every project

One Oracle Autonomous Data Warehouse Cloud (ADWC) instance family with a layered schema design,
fed by Talend, consumed by Power BI. This is the spine every workstream below hangs off.

```
SOURCES                          INGEST            WAREHOUSE LAYERS         SEMANTIC
-------                          ------            ----------------         --------
Bullhorn Data Mirror (SQL Svr)  ┐                 ADWC_LOAD    (staging)
BHODS1 (Oracle ODS)             │                     │  PL/SQL SP_LOAD_*
Hiregenics ODS (HGCODS1)        │                     ▼
Oracle EBS  APPS + XXACS        ├─► Talend Jobs ─► ADWC_USER   (curated       ─► Power BI
  (PROD / UAT / DEV instances)  │   (69 artifacts)     │        star schema)      tabular
JobDiva / ORCA                  │   cron via TMC        ▼                         models
Salesforce (opportunities)      │   Execution Plans  ADWC_RO    (read-only        (Import,
Coupa API (4 jobs)              │                     │        reporting)         Oracle
Oracle HR extracts (US + IND)   │                     ▼                           connector)
SFTP Excel feeds (/apps/ftp/)   ┘                 ADWC_USER_UAT / _DEV
```

**Schema tiering** — 306 distinct schema-qualified object references were counted:
`ADWC_LOAD` 96, `ADWC_USER_UAT` 69, `ADWC_USER` 61, `ADWC_USER_DEV` 46, `ADWC_RO` 32.
The `_UAT` / `_DEV` twins are what makes the whole platform promotable. `EV-01`

**Object families** — 475 distinct object names: `FACT_*` 84, `DIM_*` 42, `LKP_*` 21,
`RPT_*` 13, plus source-prefixed families `HGC_*` 151, `BHRPT_*` 68, `BHMR_*` 36, `EBS_*` 36,
`JD_*` 11. `EV-02`

**Connectivity** — Oracle cloud wallet mutual TLS
(`javax.net.ssl.trustStoreType=SSO&keyStoreType=SSO`, `cwallet.sso`) for ADW; JDBC with
`encrypt=true;trustServerCertificate=true` for the SQL Server Bullhorn mirror. `EV-12`

**Environments** — DEV, QA, UAT, PRD, named in the HLD glossary itself. `EV-03`

---

## Project A — BH Reporting (Bullhorn recruiter & sales performance reporting)

| | |
|---|---|
| **Dates** | Phase 1 ~2022; **Phase 2 HLD v1.3 dated 2023-09-06**; active through 2023-11 |
| **Status at exit** | Phase 2 in UAT (`BH_Reporting_UAT - ...`); AWGP matrix marks BH Reporting model changes "Currently in UAT" as of 2023-05 |
| **Business purpose** | Give recruiters, BDMs and their managers a single scorecard of pipeline activity (submissions, interviews, offers, starts, ends) joined to the financial outcome of each placement, and measure both against per-person baselines/targets |
| **Users (conservative)** | Recruiters, business development managers and their leadership, scoped by row-level security to each person's own org subtree; a super-user flag exists for unrestricted viewers. Exact headcount is **not** evidenced |
| **Source systems** | Bullhorn Data Mirror (SQL Server), BHODS1, Hiregenics ODS, Oracle EBS, Oracle CE/HR extract, JobDiva/ORCA |

### Architecture and data flow

Bullhorn mirror → `ADWC_LOAD` staging (`BHMR_*` tables: `BHMR_CORPORATEUSER`,
`BHMR_CLIENTCORPORATION`, `BHMR_APPOINTMENT`, `BHMR_PLACEMENT`) → PL/SQL transformation into
`BHRPT_*` reporting facts and dimensions in `ADWC_USER` → Power BI Import model over the
`_WPNL` (with-P&L) variants.

### Important tables, grain and keys

| Object | Role | Grain |
|---|---|---|
| `BHRPT_DIM_USER_ENTITY` (`BHRPT_DIM_BHHGC_USERENT_PNL`) | conformed producer dimension across Bullhorn + Hiregenics + Oracle HR | one row per person per role/nationality/P&L key |
| `BHRPT_FACT_PLACEMENT` / `_STARTS` / `_ENDS` | three role-play views of one placement fact, each with its own date join | placement × event date |
| `BHRPT_FACT_JOBORDER_PRIMARYUSER` / `_DATEADDED` | job-order facts joined on primary user vs. creation date | job order × user × `EFF_START_DATE_JOIN` |
| `BHRPT_FACT_JOBSUBMISSION` / `_INTERVIEWS` | submission funnel | submission × candidate × sending user |
| `BHRPT_PRODUCERS_FPNA_ALL` | financial outcome per producer | assignment × producer × designation × period × activity code |
| `BHRPT_FACT_USER_SUMMARIZED_METRICS` | pre-aggregated actuals for actual-vs-target pages | user × period |
| `LKP_BH_BASELINES` | per-person targets | person role/nationality key × period |
| `BRIDGE_USER`, `BRIDGE_BASELINEKEYS`, `BRIDGE_PERIOD_AGG_DIMDATE` | many-to-many resolution bridges | distinct key lists |
| `BHRPT_USEREMAILMAPPING_RLS`, `BHRPT_CLIENTMAP_RLS_DB` | security tables (email → person → org PATH; client entitlement) | one row per user |

Three separate `_STARTS` / `_ENDS` / base placement tables exist because each needs a *different*
active date relationship to `DIM_DATE` — a deliberate role-playing-dimension workaround rather
than relying on inactive relationships alone (though `USERELATIONSHIP` also appears 19 times).
`EV-22`

### Important business logic

- **Crediting once** (`CreditingOnce_Algo.txt`, 2023-06-19): 9-level designation ladder, window
  functions to detect and rank multi-role rows, ~22 measures zeroed on non-primary rows,
  `MERGE` back on a 15-column key. A source comment records `341,711 rows inserted`. `EV-08`
- **AWGP** = total margin/GP from Oracle timesheets ÷ fiscal weeks adjusted for holidays, per
  period selection. Direct-hire fees amortized over **6 months** (3 months planned for BH
  Reporting). `EV-09`
- **Test/house-account exclusion**: `(N/A) - NOT AVAILABLE`, `HOUSE, HOUSE`, `HOUSE, HG`,
  `HOUSE, ASCENT` names filtered; `BHMR_DUMMY_DATA` excluded by ID. `EV-10`
- **Cross-system client identity** — see Project C.
- **Designation/tenure mapping**: `LKP_HGC_PERSON_DESIGNATIONS`,
  `DIM_BH_USERDEPARTMENT_MAPPING`, `FieldHier_RO_DesignationWiseNulls_2604.xlsx` (null-designation
  triage, 2023-04-26).

### Reliability and governance controls

`CURRENT_DW_LOG_DATA` inserts at procedure start and end; `Event_log_status` per changed row;
`EXCEPTION WHEN OTHERS` isolating row failures; `_BKP<date>` snapshot tables before schema
changes; a documented HLD with source-to-target mapping, relationship register and measure
register.

### Outputs

One report, two model variants (Rec and Sales), **85 pages / ~1,040 visual containers**, mirrored
BDM and Recruiter page pairs: Home, 1-On-1 Report, Grouped Summary, Producer's Report, Details,
Actuals vs Targets, Summary, Call Details, Glossary, plus ~40 single-metric drill pages (Starts,
Terms/Ends, Offers Accepted/Rejected, Total Hires, Client/Internal Submissions, Priority A/B/C
Openings, A/B/C Jobs, Notes, Attempts, Candidate Screening, Connects, Meetings, Client Visits,
Inbound/Outbound Call Duration Bucketing, Interviews, Interview/Subs/Hires to A/B/C Jobs) and
explicitly labelled Debug/Test pages. `EV-21`

### Joel's supported role

**Authored or led.** The Phase 2 HLD, the crediting algorithm, the client-remodelling PBIT, the
RLS design workbook (`RLS ModellingWork Client_1109.xlsx`, sheets "Design-Analysis, Thoughts"
and "Plan"), the phase-2 test-case workbook, and the majority of the debugging/reconciliation
notes carry his authorship pattern (first-person design commentary, dated filenames, `_MJS`
table suffixes such as `FACT_SGNA_SUMMARY_MJS`, `FACT_JOBORDER_BKP_050123_MJ`). Some job-order
SQL is tagged with a colleague's initials (`BHM_krthk_joborderScr.sql`) — team-delivered.

### Strongest evidence paths

- `BkpNov23 - Docs, Sql, Reportspbix/BH Reporting Phase 2 - HLD v1.3.xlsx`
- `BkpNov23 - Docs, Sql, Reportspbix/BH_Reporting_UAT - Rec 2109 (ClientRemodelling R2).pbit`
- `BkpNov23 - Docs, Sql, Reportspbix/CreditingOnce_Algo.txt`
- `BkpNov23 - Docs, Sql, Reportspbix/SP LOAD BHRPT PLCINFO BHNEWUSER.txt`
- `17 11 downloads; volt/BH_Reporting_UAT - Sales (Client Based M3).pbix`

---

## Project B — FP&A reporting (Finance Project & Accounting)

| | |
|---|---|
| **Dates** | Artifacts 2021 → 2023-11 |
| **Business purpose** | Attribute revenue, gross profit, burden and margin from Oracle EBS down to the individual producer and client, at both GL-transaction and fiscal-period grain, and hold a per-period history for variance analysis |
| **Users** | Finance/FP&A analysts and staffing leadership. Not further evidenced |
| **Source systems** | Oracle EBS (`APPS`, `XXACS` custom schema, PROD/UAT/DEV), Oracle CE/HR extract, SFTP Excel manual-load feeds |

### Two-grain fact design (this is the real "dual-path attribution")

| Fact | Grain | Date key |
|---|---|---|
| `BHRPT_FPNA_FACT_GLLEVEL_WPNL` | GL transaction | `GL_DATE` |
| `BHRPT_FPNA_FACT_PERIODLEVEL_WPNL` | fiscal period | `FISCALSTART_DATE` |

Both carry `CONCATKEY_DCHCLIENTS` (client identity) and `LFNAME_JOINFIELD_FACT` (producer
identity) and both are P&L-attributed (`_WPNL` = with P&L; see `LKP_REORG_USER_CLIENT_PNL`,
`LKP_DISTINCT_PNL`). `EV-16`

### Measure vocabulary carried on the fact (~50 columns)

`hours`, `hours_payable`, `hours_billable`, `avg_billrate`, `avg_payrate`, `awgp`,
`billed_fte_adj`, `sum_of_bill`, `burden`, `medical_burden`, `h1b_burden`, `marketing_burden`,
`burden_perc`, `direct_cost`, `revenue`, `total_revenue`, `gp`, `total_margin`, `cogs`,
`gm_perc`, `gp_per_hour`, `directhire_amount`, `permfee_without_ammort`, `activity_code`,
`is_diver_hist` (Diversant brand history flag), `manual_load_flag`. `EV-17`

The presence of `h1b_burden` and `cehr_nationality` as first-class financial dimensions is
notable — visa-status cost was modelled into margin.

### History strategy

**48 physical monthly snapshot tables**, `FACT_FIN_FPNA_MEASURES_HISTORY_<MON>_<YYYY>`,
JAN-2020 through DEC-2023, plus `FACT_FIN_FPNA_MEASURES_HISTORY_PART` (partitioned variant) and
`_STG` / `_STG_MANUAL` staging twins. Reload is parameterized: a Talend context passes
`table_name_parameter=FACT_FIN_FPNA_MEASURES_HISTORY` with `period_name_from`, `period_name_to`,
`full_load_flag`, `start_date_param`, `end_date_param`, and separately `partition_year` with
`Load_type=FULL` for a whole-year rebuild. `EV-18`

### Manual-override pattern

Every hierarchy and the FP&A fact has an automated `_STG` and a human `_STG_MANUAL` twin
(`DIM_CUST_HIER_STG` / `_STG_MANUAL`, `DIM_FIELD_HIER_STG` / `_STG_MANUAL`, `DIM_ORG_HIER_STG` /
`_STG_MANUAL`, `FACT_FIN_FPNA_MEASURES_STG` / `_STG_MANUAL`), with `manual_load_flag` carried on
the fact so manual adjustments stay traceable in the report. Finance-controlled Excel files
(`FPnA_Manual_Load.xlsx`, `SGnA_Manual_Load.xlsx`) land on SFTP and are ingested by
`Job_Load_FPnA_Manual_Inserts_Master` / `Job_Load_SGnA_Manual_Inserts_Master`. **This is the
strongest "manual effort eliminated" evidence in the drive** — a governed path for finance
overrides instead of spreadsheets living outside the warehouse. `EV-19`

### Joel's supported role

**Authored.** `FPNA_dimfieldanalysis_0407.xlsx`, `FieldMapping_DivHist_BHFpna.xlsx` (10 MB
field mapping), `Ebs Source, Fpna, Milestone Debug -May23 NegativeMargin.txt`,
`MichaelAgulair Comparison_BhrptFpna.xlsx`, `FPNAvsComm.txt`, `BHRPTvsPrdcr.xlsx` — a sustained
run of reconciliation and root-cause work between EBS source, FP&A output and the commission
engine.

---

## Project C — Cross-system client and person identity resolution

| | |
|---|---|
| **Dates** | 2023-06 → 2023-09 (`Script - LKP_BH_EBS_CLIENTNAMES.txt` 2023-06-28 → `2509 Enhanced - BH EBS ClientMatch Script.txt` 2023-09-25) |
| **Business purpose** | The same customer exists in Bullhorn (ATS) and Oracle EBS (financials) under different names and IDs. Without a reconciled identity, recruiter activity cannot be joined to revenue for that client, and a client-scoped RLS model is impossible |

### The actual algorithm (three stages)

1. **Normalized exact match** — `TRIM(UPPER(customername))` between
   `LKP_BH_EBS_CLIENTNAMES` (Bullhorn side, `CONCATKEY LIKE '%BH%'`) and EBS `DIM_CUST_HIER`
   (deduplicated by `GROUP BY` on normalized name + `OCUSTID`), emitting an `NMATCH_IND` Y/N
   indicator.
2. **`UNION ALL` of matched + EBS-exclusive** — the second leg deliberately excludes EBS
   customers already consumed by stage 1 (`EBS_CUSTOMERID NOT IN (... NMATCH_IND='Y')`), so a
   client appears exactly once.
3. **Curated fallback** — unmatched Bullhorn rows `LEFT JOIN` a hand-maintained standardization
   table (`BH_COMPANY_STANDARDIZATION_TBL_POC1`); still-unmatched rows get sentinel surrogate
   keys `999999999` and `'N/A'` names rather than being dropped.

Supporting decisions recorded in comments: the Bullhorn name **wins** as the display name
(`NVL(MAPPED_BHCUSTNAME, CUSTOMERNAME) AS COMMON_CLIENTNAME`) because the slicer is
recruiter-facing; the relationship to the fact deliberately travels on the **EBS** key, since
the fact is financial. Row counts are recorded inline: 2,206 matched Bullhorn entries resolving
to 1,476 distinct EBS customers out of 4,703 total, ~3,227 expected residue. `EV-06`

Person identity is the mirror problem, solved with the same shape: `BHMR_HGC_TO_CEHR_NEWUSER`,
`SP_LOAD_BHMR_BHHGC_USERENTITY`, `LKP_BHRPT_EMPID_DISTUSER`, and a documented issues log
(`BHCorpUser EMPID Issues 2006.xlsx`, `CEHR to HGC,BH Comparison.xlsx`,
`BH CC - DCH DistinctName matches.xlsx`) — three-way reconciliation between Bullhorn corporate
users, Hiregenics persons and Oracle HR employee IDs. `SP_LOAD_CANDIDATE_DUP_DATA` handles
candidate duplicate detection. `EV-13`

**Caveat, stated plainly:** there is no fuzzy/similarity function anywhere. This is deterministic
normalization plus curated mapping, and the honest description is stronger than the current
résumé claim because it is verifiable.

---

## Project D — Commission engine (US and India)

| | |
|---|---|
| **Dates** | 2021 → 2023 |
| **Business purpose** | Calculate producer commission and bonus from placement financials, with role-based eligibility, direct-hire amortization and period control |
| **Procedures** | `SP_COMMISSION_LOAD`, `SP_COMMISSION_LOAD_WITH_EXP`, `SP_COMMISSION_LOAD_BONUS_CALC`, `SP_LOAD_ACTIVITY_CODE_LIST`, `SP_LOAD_ASSIGNMENT_ID_TASK_ID`, `SP_LOAD_BHRPT_PRODUCERS` (+ `_Debug` twins) — `SP LOAD COMM.txt`, 12,154 lines |
| **Facts** | `FACT_INDIA_COMMISSIONS_DETAILS`, `RPT_FACT_BDM_REC_AMMORT`, `RPT_BDM_REC_AMMORT_WITH_HIST`, `BHRPT_PRODUCERS_FPNA_POST_CRED` |
| **Orchestration control** | `Load_Type` (C/V/FULL/INC), `Load_Flag`, `Category_Type`, `Business_Unit`, `DH_FLAG`, `Commission_DatePeriod`, `Run_Type` = `VERIFICATION` / `PRELIMS` — a **verification run mode is a first-class load type**, i.e. the pipeline could be run in a non-mutating check mode before a real commission run. `EV-15` |
| **Inputs** | `India_Team_Lead_Size.xlsx`, `LKP_ELIG` (eligibility lookup), `LKP_SUB_SERV`, `JOB_LOAD_GGK_RATECARD` |
| **Rule specification** | `Power BI reports - AWGP logics - Production Stats.xlsx` (2023-05-03): 10 production reports × 7 roles × credit/DH/amortization. `EV-09` |
| **Joel's role** | **Contributed / led parts.** `MGAudit_1605fldev.sql`, `MARGINSAUDIT_CONSOLIDATED_HIST`, `SampleData After Crediting Changes.xlsx`, `ProdBHProducerCode_Before100%Change.txt` (a before/after snapshot of a 100%-credit rule change) and `Nov13 BHRPT changes.txt` are his. The engine predates 2023 and was almost certainly multi-author |

---

## Project E — SG&A / Budget reporting

| | |
|---|---|
| **Dates** | 2021-03 (`BnSGNA_TrailDesignWithExcel_03292021.pbix`) → 2022 |
| **Business purpose** | P&L statement and SG&A expense reporting against budget, with variance analysis by category hierarchy |
| **Procedures** | `SP_LOAD_RPT_SGNA`, `_V2`, `_TR`, `_MONTHLYLEVEL`, `SP_SGNA_INSERT_TOTALS_MONTHLY`, `SP_SGNA_INSERT_VARIANCES_MONTHLY`, `SP_TEMP_INSERT_TOTALS`, `SP_TEMP_INSERT_VARIANCES`. Signature: `(actuals_load_flag VARCHAR, ...)` — actuals and budget loaded independently |
| **Tables** | `FACT_SGNA`, `FACT_SGNA_SUMMARY`, `_MONTHLY`, `_WITH_2_LEVELS`, `DIM_CATEGORY_HIERARCHY_SGNA`, `DIM_CATEGORY_DETAILED_SGNA`, `DIM_BUSINESS_UNIT_DTL` |
| **Hierarchical rollup** | `SP_temp_Insert_Totals(Category_var, Category_name, ...)` and `SP_temp_Insert_Variances(Category_var, Category_name, L2_specifier, ...)` — parameterized recursive totalling by category level, plus `CONNECT BY` usage in the corpus |
| **Reports** | `Budget_&_SG&A (pnl statement).pbix`, `Budget_&_SGnA_UAT.pbix`, `BnSGNA checks505.pbix` |
| **Joel's role** | **Authored** `SGNA_NewSubProc_1.sql` (2022-06-17), `sngaChecks_504.sql`, `FACT_SGNA_SUMMARY_MJS` |

---

## Project F — APCOB, VSD and DSO (operational finance pipelines)

| | |
|---|---|
| **APCOB** | Accounts-Payable / Cost-of-Business reporting. `SP_LOAD_FACT_APCOB`, `FACT_COB`, `FACT_COB_HGC_DATA`, Excel feeds for BUDGETS / OpeningCounts / TeamLookup, `Master_APCOB_ExcelFeeds`, `Job_Load_Master_APCOB`. Four PBIX versions v1.4 → v1.64 |
| **VSD** | Vendor/Supplier payment information. `Full_Load_Job_VSD_Payment_Info` (v0.1.2), `Job_Load_VSD_Payment_Info` (v0.1.3), `LOAD_VSD_MASTER_TRACKER` reading `Master_Tracker.xlsx` from SFTP |
| **DSO** | Days Sales Outstanding / AR aging. `SP_LOAD_DSO_REVENUE_TABLES`, `SP_LOAD_EBS_DSO_AGING_HIST`, `SP_LOAD_EBS_AR_AGING_HIST`, `LKP_AGING_STARTS_ON`, `LKP_REVENUE_CREATED_ON`, `DSO_EXTRACT_HIST` |
| **Coupa** | Four API integration jobs: Expenses, Invoice combination, Invoice **Reverse** Integration (write-back), Purchase Order combination |
| **Production schedule** | `PROD_DSO_FACTS_EP` cron 14:30 Asia/Calcutta; `PROD_LOAD_APCOB_EP` cron 14:00 Asia/Kolkata — daily. Handover instruction in the same sheet: *"Please create the trigger and keep in pause. They should not be in active state."* `EV-04` |
| **Joel's role** | **Authored** `APCOB_PROC_UATsql`, `APCOB_Proc_3108.sql`, `procApcob_1309.sql`, `SP_LoadFactAPCOB.sql`, `dso-ddls.sql`, `dsoProcDev.sql`, `dso_revChecks.sql`, `HistChecksDSOsql`, and the deployment-tracking workbooks |

---

## Project G — Power BI platform administration, two tenants (2023)

This is the workstream most absent from the current résumé, and it is the most directly relevant
to BI-platform roles.

| | |
|---|---|
| **Dates** | 2023-07 (`PBI Discussion_2007 VoltTenant, KT, kickoff.txt`) → 2023-11 |
| **Trigger** | An acquisition brought a second Power BI tenant (Volt) into scope alongside the existing Innova/ACS tenant. Joel took over its administration via a structured knowledge transfer |
| **Business purpose** | Keep two tenants' gateways, data sources, workspaces and scheduled refreshes healthy; inventory an unfamiliar estate; upgrade on-premises gateways without breaking in-network reports |

### What was built and written

- **Five monitoring reports** — `MonitorPowerBI.pbix`, `VoltMonitoringPBI.pbix` (10 pages, 117
  visuals), `Volt_DataRefresh_Monitoring.pbix`, `All_Innova_Workspaces.pbix`,
  `All_Workspaces_Volt.pbix`. `MonitorPowerBI` sources the **Power BI REST API** in Power Query
  M via a service principal (OAuth2 `client_credentials`, a `GetAccessToken()` function, Bearer
  header), expanding workspace/capacity JSON (`isOnDedicatedCapacity`,
  `capacityMigrationStatus`, `state`, `type`) and filtering to active workspaces. A companion
  PowerShell approach is referenced for data-source enumeration. `EV-27`
- **`On Premise PowerBI Gateway Upgrade Procedure.docx`** (2023-11-03) — admin prerequisites, a
  required VM snapshot before upgrade, user communication template, expected 1–2 hour downtime
  window, and the distinction between gateway-dependent in-network sources (Oracle FS/HR,
  Bullhorn Data Mirror, the warehouse) and cloud sources unaffected. **Attribution: partly
  compiled.** The embedded announcement email is authored by a Volt IT manager; the surrounding
  procedure is Joel's consolidation. The companion checks document says so explicitly:
  *"Existing documentation prepared by Volt with respect to gateway upgrades"*
- **`PowerBI Gateway checks - Post Upgrade, Migration.docx`** — pre/post-upgrade verification
  procedure, including recovering the gateway cluster with its recovery key
- **`Dependencies Identified.docx`** (2023-10-16) — a software/dependency inventory of the
  production gateway VM: Oracle ODAC and an 11.2.0 Oracle client, `DataGateway` folder with
  `EnterpriseGatewayConfigurator`, .NET 4.8, gateway build Dec-2022, Azure Monitor agent,
  manually placed `TNSNAMES.ORA`. This is the artifact of someone auditing an inherited machine
  before touching it
- **`Details for new server request - ACSBI team.docx`** (2023-09-14) — Joel's own specification
  for a dedicated **production** BI server (32 GB RAM, 8 cores, 250 GB) to separate production
  from non-production gateways and support >2 concurrent developers, naming the four users
  requiring access with himself first. Ownership evidence
- **`Volt PowerbiAdministration_Activities.xlsx`** — a KT plan with sheets `KT Items`, `POC`,
  `Checklist`: workspace/module inventory, gateway setup and environment, refresh-status
  monitoring, gateway health monitoring, gateway upgrades, cluster admin account and recovery
  keys, data-source status monitoring
- **`Volt_Workspaces.csv`** / **`Volt NonPersonal Workspaces.xlsx`** — estate inventory
- **`Estimated Activities - Server, Gateway upgrades..xlsx`** — his own effort estimation
- **`TabularEdt CScript - Exp info.txt`** (114 KB, 2023-10-18) — Tabular Editor C# script
  material, i.e. external-tooling model automation beyond Power BI Desktop

### Joel's supported role

**Owned.** The server request, dependency audit, KT plan, estate inventories, effort estimates
and monitoring reports are his. The pre-existing Volt procedures he consolidated and extended
rather than authored.

---

## Project H — Enablement and hiring

| | |
|---|---|
| **`Training Plan draft v1.0.xlsx`** (2023-10-10) | A structured curriculum Joel authored. **Power BI track** (42 rows, day-by-day): desktop vs service vs PowerApps, the mashup and xVelocity engines, connecting to Oracle ADW, file types incl. `.pbip`, workspaces and access, gateway-to-dataset mapping, refresh scheduling, **Import vs DirectQuery vs other modes**, basic *and* dynamic RLS, visual families, drill-through, filter levels and cross-page sync, interaction control, bookmarks/buttons/selection pane, custom visuals. **SQL track** (33 rows): relational model, query optimizer/algebrizer, Oracle vs T-SQL architecture, DDL/DML/DCL/DQL/TCL, constraints, set operators, **logical order of query processing**, window/ranking functions and `OVER (PARTITION BY ...)`, temp tables vs table variables vs CTEs |
| **`Questions powerbi.xlsx`** (648 KB) + **`Qs Interview.txt`** | Interview question banks — consistent with the "35+ interviews conducted" claim, though the artifacts do not evidence a count |
| **Joel's role** | **Authored** the curriculum. Mentoring of 2–4 person sub-teams is plausible from the KT/training/POC artifacts but **not directly evidenced**; the Innova Idol Award has **no artifact on this drive** |
