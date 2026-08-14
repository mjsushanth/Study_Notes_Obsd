---
title: Work Evidence Audit - Summary
type: audit-summary
audit_date: 2026-08-12
source_root: "/Users/joel/.mounty/Seagate/Personals - 1/Work Files - MJS BI/GGK WorkSample Assets/"
employer: Innova Solutions (GGK -> acquired into ACS/American CyberSystems -> renamed Innova; Innova later acquired Volt), Hyderabad
employment_window: 2019-12 to 2023-11
files_inventoried: 6031
files_deeply_inspected: 62
tags:
  - evidence/audit
  - work/innova
  - domain/data-engineering
  - domain/business-intelligence
---

# 00 - Audit Summary

Read-only evidence audit of Joel's retained work artifacts from Innova Solutions / ACS Group.
Nothing on the source drive was renamed, moved, modified or executed. No database connection,
Power BI refresh, macro, script or installer was run.

**Handoff brief for the résumé agent: [[07-strength-points]]** — six consolidated strength points,
self-contained.

Related: [[01-project-and-system-map]] · [[02-evidence-ledger]] · [[03-technical-depth]] ·
[[04-resume-candidate-bullets]] · [[05-questions-for-joel]] · [[06-file-coverage]]

## Scope actually inspected

| | |
|---|---|
| Total files inventoried (metadata) | **6,031** (5.0 GB) |
| Files opened and read in substance | **62** |
| Files pattern-scanned as a corpus | **393** deduplicated SQL/PL-SQL/notes files (9.9 MB); **8,918 join predicates** parsed in a second pass |
| Power BI packages structurally parsed | **17** (1 `.pbit` fully, incl. 12 MB tabular model) |
| Archives expanded | 0 on source; PBIX/PBIT copies expanded in scratchpad only |
| Files deliberately never opened | Oracle wallets, `.jks`, `.p12`, `.sso`, `.ppk`, exported browser passwords, a password screenshot |

Coverage detail in [[06-file-coverage]].

## What this employment actually was

The baseline description ("Data Engineer, FP&A + SG&A analytics and enterprise data warehousing")
is correct but substantially undersells the scope. The artifacts describe a **single Oracle
Autonomous Data Warehouse platform** serving a US staffing and workforce-solutions group, with
Joel working across the full stack: source ingestion (Talend), warehouse modelling and PL/SQL
transformation, Power BI semantic models, and — in the final year — Power BI *platform
administration* for two tenants after an acquisition.

Decoded from the project's own HLD glossary (not inferred):

- **BH / BHMR** = Bullhorn / Bullhorn Data Mirror (ATS, SQL Server source)
- **HGC** = Hiregenics · **ORCA / JD** = JobDiva · **EBS** = Oracle E-Business Suite
- **ADWC** = Oracle Autonomous Data Warehouse Cloud · **ODS** = Operational Data Source
- **FP&A** = Finance Project and Accounting · **AWGP** = Average Weekly Gross Profit
- Environments: **DEV, QA, UAT, PRD** (four, not two)

## Ten strongest verified discoveries

1. **69 distinct Talend job artifacts** promoted across QA → UAT → PROD, tracked in a
   deployment manifest with `JobActionType`, semantic versions, injected `ContextParams`, cron
   triggers and timeouts. Environment-prefixed labels (`QA_BI_FTR_*`, `UAT_BI_FTR_*`,
   `PROD_BI_*`). This is the hard evidence behind the CI/CD claim — and it is far more concrete
   than "CI/CD improvements". `EV-11`
2. **39 distinct Oracle stored procedures** by name (≈34 excluding `_DEBUG` / `_TEMP` / `_DEV`
   variants), across ~9,000–12,000-line source dumps. Exceeds the "25+" baseline. `EV-05`
3. **A 40-user-table / 187-relationship / 354-measure Power BI tabular model**, recovered in
   full from a `.pbit`. Median measure is 13 lines of DAX; 66 measures exceed 30 lines; the
   largest is 159 lines. `EV-20`
4. **Dynamic hierarchical row-level security**: one RLS role filtering on
   `USERPRINCIPALNAME()` against an email-to-person map, resolving a parent-child
   `PATH`/`PATHCONTAINS` org hierarchy, with an `ISSUPERUSER` bypass flag. `EV-24`
5. **The "crediting once" attribution algorithm** — a 9-level designation rank ladder
   (REC, RECOVR, BDM, BDMOVR, CSM, SECREC, TERREC, SECBDM, TERBDM), `COUNT(*) OVER` to detect
   multi-role rows, `ROW_NUMBER()` by rank to elect the primary credit holder, ~22 financial
   measures zeroed on non-primary rows, then `MERGE` back to the fact on a 15-column composite
   key. Preserves per-producer row visibility while counting revenue once. `EV-08`
6. **A pervasive ETL audit framework**: `CURRENT_DW_LOG_DATA` (SP name, status, table, rowcount,
   start/end datetime, error message) referenced **363 times**; plus `CURRENT_LOAD_LOG_DATA` on
   the Talend side and `Event_log_status` for row-level change replay. Two-layer, not decorative.
   `EV-14`
7. **Column-level change-data-capture replay** in `SP_LOAD_Fact_JobOrder`: cursor loop over
   Bullhorn history rows applying old/new column values, with mixed date-format coercion,
   string-to-numeric ID remapping via lookup tables, `[blank]` sentinel handling, quote
   escaping, and per-row `EXCEPTION WHEN OTHERS` so one bad row cannot abort the load. `EV-07`
8. **An 85-page, 1,035-visual Power BI report** with mirrored BDM/Recruiter navigation, 32
   bookmarks, 27 action buttons and ~40 metric-specific drill pages. Two variants (Rec and
   Sales) at **~520 MB PBIX each**. `EV-21`
9. **Power BI tenant administration via the Power BI REST API** — service-principal OAuth2
   client-credentials flow in Power Query M, enumerating workspaces, capacities and refresh
   status across the Innova and Volt tenants. Five monitoring reports. `EV-27`
10. **A cross-report business-rule matrix** defining, for 10 named production reports, which of
    7 producer roles earn AWGP credit, whether Direct Hire fees enter GP, and the amortization
    window (6 months live, 3 months planned). This document *is* the commission/attribution
    specification. `EV-09`

## Five most promising résumé improvements

1. **Lead with the platform, not the dashboards.** The strongest recoverable story is a
   four-environment Oracle ADW platform ingesting **8+ distinct source systems** (Bullhorn Data
   Mirror on SQL Server, BHODS, Hiregenics ODS, Oracle EBS across 4 instances, Salesforce,
   Coupa API, Oracle HR extracts, SFTP Excel feeds) — not the four in the current claim.
2. **Name the hard algorithms.** "Dual-path P&L attribution" is vague. The recoverable version
   is the 9-level crediting ladder plus two-grain FP&A facts
   (`..._FPNA_FACT_GLLEVEL_WPNL` at GL-date grain and `..._FPNA_FACT_PERIODLEVEL_WPNL` at
   fiscal-period grain). That is a specific, defensible, senior-sounding design.
3. **Claim the deployment engineering properly.** Environment-parameterized procedures
   (`Schema_Name`, `Load_Type_V`) + Talend context injection + parameterized Power Query
   (`ServerNameParam`, three schema params, six control params) = one artifact promotable across
   DEV/QA/UAT/PROD without editing code. That is the real "50% deployment time" mechanism.
4. **Add the platform-administration year.** Gateway upgrade procedures, pre/post-migration
   checklists, a dedicated production BI server specification, two-tenant workspace inventory
   and refresh monitoring, and a written Power BI + SQL training curriculum. None of this is in
   the current résumé and it is exactly what BI-platform roles hire for.
5. **Use the honest DAX/model metrics** (354 measures, 129 calculated columns, 187
   relationships, 40 user tables, calculation group, dynamic RLS) instead of "25+ dashboards
   covering 500+ metrics", which the evidence neither confirms nor contradicts.

## Major contradictions and risks

| Risk | Detail |
|---|---|
| **"SCD Type 2" is true but not where you'd expect** | `EFF_START_DATE` / `EFF_END_DATE` / `IS_CURRENT` end-dating is implemented on **fact** tables (`FACT_JOBORDER`, `FACT_JOBSUBMISSION` and their `_STAGE` twins) via correlated-subquery end-dating. The FP&A history is *not* SCD2 — it is **48 physical monthly snapshot tables** (`FACT_FIN_FPNA_MEASURES_HISTORY_<MON>_<YYYY>`, Jan 2020–Dec 2023). Say "SCD2 on transactional facts plus monthly snapshot history", not "SCD2 dimensions". |
| **"6+ fuzzy patterns" is CONFIRMED — corrected 2026-08-12** | An earlier pass of this audit wrongly flagged this claim after finding no Oracle fuzzy *functions* (`UTL_MATCH`, `SOUNDEX`). That was the wrong test. The fuzzy matching here is **cascading multi-round string joins**: **8,918 join predicates, 1,609 (18%) joining on transformed strings rather than keys; 5,527 `LEFT JOIN`s across 207 files; 199 multi-alternative OR-laddered string predicates, of which 66 are six-way ladders** (plus 25 five-way, 11 four-way, 2 eight-way). Joins are deliberately loose, produce subsets or duplication, and the duplication is then resolved by business rules — `ROW_NUMBER`/`COUNT(*) OVER` collision detection, `HAVING COUNT(*)>1` probes, `MINUS`/`NOT EXISTS` residue tracking, sentinel keys, and a `MANUALMAP_BH_USERID` override column. **"6+ fuzzy match patterns" is now Strongly Supported.** Only the **88% accuracy figure** still lacks an artifact. `EV-06` `EV-65`
| **Nothing supports "real-time"** | Production cron triggers fire **once daily at 14:00 / 14:30 Asia/Kolkata**. The fastest thing found is one 15-minute CDC job (`CDC_JOB_FROM_ODS_TO_ADWC`). No `RangeStart`/`RangeEnd` anywhere, so **no Power BI incremental refresh was configured**; models are full Import. "Daily batch with a 15-minute CDC feed" is the defensible phrasing. |
| **"11+ programs" needs a definition** | Defensible counts exist but differ by unit: 10 named production reports, ~69 Talend jobs, 5 FTP project folders (`Fpna_Project`, `Commissions_Project`, `FM_Reporting`, …), or the workstream names APCOB / VSD / DSO / SG&A / Commissions / BH Reporting / DH Payment / Ranking & Summits. Pick one unit and state it. |
| **Model hygiene is a genuine weak spot** | Auto date/time was left on: **115 auto-generated `LocalDateTable_*` tables** against 40 real ones, and 690 of the 819 calculated columns are auto date columns. This is very likely a large part of why the PBIX is 520 MB. Do not claim model-size optimization. It is, however, excellent "what I would do differently now" material. |
| **Some artifacts are other people's work** | `jobs-def (praveens master file).xlsx` is named for a colleague; the gateway-upgrade email text is authored by a Volt IT manager; `BHM_krthk_joborderScr.sql` carries another initials tag. The Volt gateway checks document says outright "Existing documentation prepared by Volt". Attribution is handled per-row in [[02-evidence-ledger]]. |
| **Draft vs production** | The HLD's own `ETL - Talend Jobs` sheet contains only "TBD:". Several `POC_*`, `_Debug`, `_TEMP`, `_DEV`, `_BKP*` and `Untitled*.sql` artifacts are explicitly exploratory. The AWGP matrix marks Client Report and BH Reporting model changes as **"Currently in UAT"** as of May 2023 — those two were not yet production at that date. |

## Security finding - act on this

The source drive contains **live-looking secrets in plaintext**. They are named here so they can
be dealt with; no secret value is reproduced in any file in this folder.

1. An **Azure AD application client secret** and client ID hardcoded in the Power Query M of
   `BkpNov23 - Docs, Sql, Reportspbix/MonitorPowerBI.pbix` (service principal used for the
   Power BI REST API). **Rotate / confirm the app registration is deleted.**
2. A **SendGrid SMTP API key** field and internal SMTP host in the Talend context parameters
   inside `Jobwise Context Change - PROD ( APCOB, VSD, DSO ).xlsx`.
3. **Oracle cloud wallets** (`cwallet.sso`, `.p12`, `.jks`, `ewallet`) for `DWPRD1`, `DWDEV1`
   and `ADW` under `17 11 downloads; volt/OneDrive_2_11-17-2023/WALLET zips, fs/`.
4. `OneDrive_1_11-29-2023/ChromePasses_2802Exp.csv` — an exported browser password list — and
   `devuser01_nonprd_1.ppk`, a private key.
5. `SQL SAVED_FILES, SCRIPTS/awdc_prd_passwords.PNG` — a screenshot whose filename asserts it
   holds production passwords.
6. Internal hostnames, IPs, service names, schema names and DB usernames throughout the Talend
   context parameters. All are redacted or generalized in this dossier.

None of these were opened. Items 1 and 2 were encountered incidentally while parsing M code and
workbook cells.

## Recommended next investigation

1. ~~Open the 520 MB PBIX in Power BI Desktop~~ — **closed 2026-08-12: not possible.** Joel is on
   macOS and Power BI Desktop is Windows-only; Tabular Editor and DAX Studio are likewise
   Windows-only, and installing software is out of scope. The `DataModel` part is
   XPress9/VertiPaq-compressed with no stdlib decompressor. **Accept the gap:** the `.pbit`
   already yielded the complete model *schema* (tables, relationships, all 354 measures, RLS,
   calculation group, M queries). What is permanently unrecoverable here is only **row counts and
   per-column compression sizes**. Consequence: make no data-volume or row-count claim about the
   semantic model. BH Reporting needs no further investigation — the artifact coverage is already
   sufficient.
2. **`SP LOAD COMM.txt` (12,154 lines)** — the commission engine. I mapped its 10 procedures and
   sampled the crediting logic but did not read the bonus calculation
   (`SP_COMMISSION_LOAD_BONUS_CALC`) or the expenditure-based variant in depth. Highest
   remaining concentration of unexploited business logic.
3. **`SQL Developer/SqlHistory/` (119 XML files, ~1,478 nodes)** — Joel's own executed-query
   history. Timestamps are present but not in the attribute form I probed. Parsing these
   correctly would give a *dated* record of hands-on activity, which is the single most
   defensible form of "I personally wrote this" evidence in the whole drive.
4. **`Untitled*.sql` (~40 files, up to 116 KB)** — unnamed scratch files from the SQL Developer
   session store. Pattern-scanned as part of the corpus but not individually read; likely
   contains further validation and reconciliation queries.
