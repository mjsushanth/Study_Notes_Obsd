---
title: File Coverage
type: coverage-record
audit_date: 2026-08-12
files_inventoried: 6031
files_deeply_read: 61
files_pattern_scanned: 393
tags:
  - evidence/coverage
  - work/innova
---

# 06 - File Coverage

Back to [[00-audit-summary]] · [[02-evidence-ledger]] · [[07-strength-points]]

This note exists so that no claim elsewhere in this dossier rests on an implied "I read
everything". **I did not read every file.** 6,031 files were inventoried by metadata; 393 were
pattern-scanned as a corpus; **61 were read in substance**. Everything below is the honest
accounting.

Source root: `/Users/joel/.mounty/Seagate/Personals - 1/Work Files - MJS BI/GGK WorkSample Assets/`

---

## Method, in order

1. **Full recursive inventory** — `find` + `stat` producing modified-date, size and path for all
   6,031 non-AppleDouble files (5.0 GB) into a scratch index. No source file touched beyond
   `stat`/`read`.
2. **Date-histogram triage.** 4,870 files carry a `2023-12` mtime — a bulk drive-copy date, so
   their timestamps are worthless. 4,753 of those sit under `17 11 downloads; volt/23FebDL Backups`.
   The remaining ~1,160 files retain real 2021–2023 dates and were prioritized.
3. **Cluster classification** into employment artifacts vs third-party study material vs tool
   detritus vs credentials.
4. **Corpus deduplication.** 509 `.sql`/`.pls`/`.txt` files from the employment clusters →
   **393 unique by MD5** (9.9 MB). Copies were made into the scratch directory with sanitized
   names so `grep -r` could run reliably over paths containing spaces and commas.
5. **Pattern census** over the 393 with ~36 case-insensitive technique regexes, recording both
   files-containing and total occurrences.
6. **Structural parse of Power BI packages** — treated as ZIP archives, copies expanded in
   scratch. `DataModelSchema` (UTF-16 JSON), `DataMashup` (a nested OPC package behind an 8-byte
   header), `Report/Layout` (UTF-16 JSON) parsed with Python's stdlib.
7. **Purpose-built XLSX reader.** No `openpyxl` or `pandas` on this machine and installing
   software is out of scope, so a minimal reader was written against `xl/workbook.xml`,
   `xl/sharedStrings.xml` and the sheet XML.
8. **DOCX text extraction** by stripping tags from `word/document.xml`.
9. **Second pass, 2026-08-12 (after Joel's clarification on fuzzy matching)** — a join-predicate
   parse across all 393 corpus files: every `ON` clause extracted up to the next
   `JOIN`/`WHERE`/`GROUP BY`/`UNION`/`;` boundary (8,918 predicates), then classified by whether it
   joins on a transformed string and by how many `OR` alternatives it carries. This produced
   `EV-65` – `EV-72` and reversed the audit's earlier conclusion about fuzzy matching. One
   additional file was read closely at this point:
   `2023 J,F downloads/SP_LOAD_BHMR_BHHGC_USERENTITY - PROD 02022023.txt` (1,469 lines — the
   person-resolution cascade, its four-alternative match predicate and its role-classification
   passes).

No SQL was executed. No database was connected. No macro, script, installer or `.exe` was run. No
Power BI process was launched and nothing was refreshed. Nothing on the source was renamed, moved,
modified or deleted.

---

## Directories inspected

| Directory | Files | Depth of inspection |
|---|---|---|
| `BkpNov23 - Docs, Sql, Reportspbix/` | 81 | **Full listing + ~30 files read.** Densest genuine cluster; the HLD, procedure dumps, crediting algorithm, client-match script, PBIT, Volt monitoring reports and governance docs all live here |
| `SQL SAVED_FILES, SCRIPTS/` (+ `SomeCode Scrpts_1/`, `2023JANFEB newSqls_DiverBHetc/`) | 483 | **Full listing + whole-corpus pattern scan.** ~10 files read in substance. Heavy internal duplication (root ↔ `SomeCode Scrpts_1` ↔ `2023JANFEB` are largely the same files) |
| `OneDrive_1_11-29-2023/` | 17 | **Full listing; 4 read.** Mostly duplicates of `BkpNov23` plus credential files that were skipped |
| `2023 J,F downloads/` | 35 | **Full listing; 2 read** (`SP_LOAD_BHMR_BHHGC_USERENTITY - PROD 02022023.txt`, `JobOrder_Script_Final (1).sql`), rest in the pattern corpus |
| `OldTest_Notepads/` | 14 | Listing + pattern corpus only |
| `Halfhour buckets/` | 6 | Listing only; flagged as an open question |
| `17 11 downloads; volt/` (top level) | 6 | **Both 520 MB PBIX structurally parsed** (layout only — see limitations) |
| `17 11 downloads; volt/OneDrive_2_11-17-2023/` | 117 | Listing only. Contains the wallet archives — deliberately untouched |
| `17 11 downloads; volt/23FebDL Backups/` | 4,753 | **Listing + targeted subdirectory analysis.** `2022 Downloads older/` (20 PBIX, 19 SQL) parsed; `Roaming - scripts/SQL Developer/` triaged; the rest is IDE state |
| `Study Samples, Docs, etc/` | 508 | **Indexed only, by your instruction.** Confirmed third-party |
| Source root loose files | 14 | **Full listing; 3 read** (`Details for new server request - ACSBI team.docx`, `RLS ModellingWork Client_1109.xlsx`, `Volt PowerbiAdministration_Activities.xlsx`) |

---

## Files read in substance (61)

**Executable logic — PL/SQL and SQL (11)**
`SP LOAD BHRPT PLCINFO BHNEWUSER.txt` (9,157 lines; `SP_LOAD_Fact_JobOrder` read closely) ·
`SP LOAD COMM.txt` (12,154 lines; procedure map + sampled) ·
`SPLOAD CEHR DSO LOADS.txt` (8,392 lines; procedure map) ·
`SPLOADUSER BH ETC.txt` (986 lines) · `ProdBHProducerCode_Before100%Change.txt` (4,229 lines) ·
`Nov13 BHRPT changes.txt` (1,890 lines) · `CreditingOnce_Algo.txt` (**read in full**) ·
`2509 Enhanced - BH EBS ClientMatch Script.txt` (**read in full**) ·
`BH ClientRLS NewProc 1905.sql` · `SP_LOAD_BHMR_BHHGC_USERENTITY - PROD 02022023.txt` ·
`JobOrder_Script_Final (1).sql`

**Power BI packages structurally parsed (17)**
`BH_Reporting_UAT - Rec 2109 (ClientRemodelling R2).pbit` — **fully parsed**: 12.1 MB
`DataModelSchema` (all 354 measures, 129 calculated columns, 187 relationships, RLS role,
calculation group), `DataMashup` (407-line M, 42 queries), 7.2 MB `Report/Layout` (85 pages,
1,035 visuals) ·
`BH_Reporting_UAT - Sales (Client Based M3).pbix` and `... M2 ...` (layout + M; **model not
readable**) ·
`MonitorPowerBI.pbix` (M read — REST API) · `VoltMonitoringPBI.pbix` ·
`Volt_DataRefresh_Monitoring.pbix` · `All_Innova_Workspaces.pbix` · `All_Workspaces_Volt.pbix` ·
`POC_Report_ExtClientRLS.pbix` · `RN.pbix` · `APCOB reporting v1.64.pbix` · `v1.4.pbix` ·
`Budget_&_SGnA_UAT.pbix` · `Budget_&_SG&A (pnl statement).pbix` · `DH_Payment_UAT.pbix` ·
`US_Staffing_Sales_Ranking_Summit_Report.pbix` · `US_Staffing_Recruiter_Ranking_Summit_Report.pbix` ·
`BnSGNA checks505.pbix`

**Workbooks read (16)**
`BH Reporting Phase 2 - HLD v1.3.xlsx` (sheet inventory + Introduction, Reporting - Info,
Reporting - Tables, ETL - Talend Jobs, HLD - Data Flow Diagram) ·
`HLD, TestCases for Phase2 - Cred, Splits.xlsx` (sheet inventory) ·
`Power BI reports - AWGP logics - Production Stats.xlsx` (**read in full**) ·
`jobs-def.xlsx` (**read in full**) · `jobs-def (praveens master file).xlsx` ·
`Jobwise Context Change - PROD ( APCOB, VSD, DSO ).xlsx` (13 sheets; 3 read in detail) ·
`EP Trigger Timings ( APCOB, VSD, DSO ).xlsx` (**read in full**) ·
`Training Plan draft v1.0.xlsx` (**both tracks read**) ·
`Volt PowerbiAdministration_Activities.xlsx` (**read in full**) ·
`Guide on PowerBI RLS - Attribute linked.xlsx` · `RLS ModellingWork Client_1109.xlsx` ·
`Estimated Activities - Server, Gateway upgrades..xlsx` · `ERP, HCM Analysis.xlsx` ·
`FieldMapping_DivHist_BHFpna.xlsx` · `BH - EBS Mapping 2109.xlsx` · `Plans (BH).xlsx`
(last four: sheet structure only)

**Documents read (4)**
`Dependencies Identified.docx` · `On Premise PowerBI Gateway Upgrade Procedure.docx` ·
`PowerBI Gateway checks - Post Upgrade, Migration.docx` ·
`Details for new server request - ACSBI team.docx`

**Corpus-level (13 further files quoted from pattern hits)** — `BHROTables_DDL.sql`,
`dso-ddls.sql`, `dsoProcDev.sql`, `dso_revChecks.sql`, `HistChecksDSOsql`,
`BHMRSCD_ConvTrails.sql`, `BHValidationPROD_1802.sql`, `BH_RepMeasures_DEBVALIDS.sql`,
`SGNA_NewSubProc_1.sql`, `sngaChecks_504.sql`, `BHTestQueries_07012023.sql`,
`BHM_krthk_joborderScr.sql`, `CreateScriptsTalend_bhods_loadDev.sql` — read as regex context
windows, not end to end.

---

## Files sampled (pattern-scanned, not read)

**393 deduplicated `.sql` / `.pls` / `.txt` files (9.9 MB)** from the five employment clusters.
Every technique count in [[03-technical-depth]] §3 derives from these. Individually most were
never opened — including ~40 `Untitled*.sql` files up to 116 KB from the SQL Developer session
store, which likely hold further validation and reconciliation queries. Flagged as remaining
high-value work in [[00-audit-summary]].

**Extension coverage across the drive** (all / employment clusters only):

| | all | employment | note |
|---|---|---|---|
| `.sql` | 516 | 457 | 393 unique after dedupe; all pattern-scanned |
| `.txt` | 134 | 110 | in the corpus |
| `.xlsx` / `.xlsm` | 196 | 179 | 16 read; **160 employment workbooks not opened** |
| `.pbix` | 218 | 34 | 16 parsed; 184 are third-party samples |
| `.pbit` | 6 | 1 | the one work `.pbit` fully parsed |
| `.docx` | 129 | 46 | 4 read; **42 employment documents not opened** |
| `.pdf` | 68 | 57 | **0 read** |
| `.pptx` | 14 | 11 | **0 read** |
| `.csv` | 31 | 17 | 2 listed, 0 parsed |
| `.xml` | 2,293 | 2,293 | 2,884 files under the SQL Developer profile are IDE state; **119 `SqlHistory` XMLs were counted but not parsed** |
| `.pls` | 9 | 9 | in the corpus |

---

## Files skipped, and why

**Deliberately never opened — security and privacy (≈821 files matched credential patterns):**

| Item | Reason |
|---|---|
| Oracle wallets for `DWPRD1`, `DWDEV1`, `ADW` (`cwallet.sso`, `ewallet`, `.p12`, `.jks`, 179 `.sso` + 179 `.p12` + 358 `.jks` matches across nested copies) | Live database credentials. Existence and instance names recorded; contents not read |
| 358 `.ora` files (`tnsnames.ora`, `sqlnet.ora`) | Connection strings and internal hostnames |
| `OneDrive_1_11-29-2023/ChromePasses_2802Exp.csv` and `_2Bk.csv` | Exported browser passwords |
| `OneDrive_1_11-29-2023/devuser01_nonprd_1.ppk` | Private key |
| `SQL SAVED_FILES, SCRIPTS/awdc_prd_passwords.PNG` | Filename asserts production passwords |
| `ExportedzzConn_sqldev23feb.json` | Saved SQL Developer connection definitions |

Two secrets were encountered **incidentally** while parsing legitimate artifacts (an Azure AD
client secret inside `MonitorPowerBI.pbix`'s M code; a SendGrid API key field in a Talend context
cell). Neither value appears anywhere in this dossier. Both are flagged for rotation in
[[00-audit-summary]].

**Third-party learning material — indexed only, per your instruction (508 files, 910 MB):**
`Study Samples, Docs, etc/PowerBI Study, Crubal, examples/DAX Fridays/` (101 PBIX),
`.../Community Downloads/` (60 PBIX), `.../PowerbiPyth Trails/PowerBIExamples-master/`,
`Study Samples, Docs, etc/SQL training codes/` (54 SQL), and the 2021-dated
`PowerBI samples , reports/` tutorial files (Northwind, COVID-19 tracker, LOOKUPVALUE, RANKX,
TREATAS, running totals). Downloaded samples cannot evidence what Joel built. **Note:** I did not
verify file-by-file that no work PBIX is misfiled into these trees — you chose "index only", and
the 2021 dates and tutorial titles are consistent with third-party origin.

**Tool detritus — not evidence:** 2,884 files under
`23FebDL Backups/Roaming - scripts/SQL Developer/system20.4.0.379.2205/` (IDE preferences,
toolbars, caches, window state), plus `.wsmode`, `.wstcref`, `.settings`, `.lck`, `.jdb`, `.jws`
files.

**Executables and installers — never run, per the safety rules:** 7 `.exe`, 3 `.msi`, plus `.dll`
and `.map` files.

**Not reached for time:** 57 employment-cluster PDFs, 11 PPTX, 42 employment DOCX, ~160
employment XLSX, 17 CSV, 26 PNG screenshots. Several of these are likely valuable — see
"remaining high-value areas".

---

## Files that failed to parse

| File | Reason |
|---|---|
| `BH_Reporting_UAT - Sales (Client Based M3).pbix` and `... M2 ...` — the `DataModel` part | Power BI stores the tabular model as XPress9/VertiPaq-compressed binary. The report layout and (for M2) the M queries parsed fine; **the model itself is unreadable without loading it into Power BI or a Tabular Object Model client.** **Permanently closed 2026-08-12:** Joel is on macOS, and Power BI Desktop, Tabular Editor and DAX Studio are all Windows-only. The `.pbit` already yielded the complete model *schema*, so the only unrecoverable facts are fact-table row counts and per-column compression sizes. Make no data-volume claim about the model. See Q10 in [[05-questions-for-joel]] |
| `DataModel` part of the other 15 PBIX | Same limitation. For those, only report layout and (where present) M were recovered |
| M queries in 15 of 17 PBIX | `DataMashup` extraction returned no `.m` part for most PBIX (the mashup is embedded differently once a model is loaded). Only the `.pbit` and `BH ... M2` yielded M source |
| `HLD - Data Flow Diagram` and `ETL - Talend Jobs` sheets of the HLD | The data-flow diagram is a **drawing object**, not cells — my reader recovers cell text only, so the diagram content is unrecovered. `ETL - Talend Jobs` genuinely contains only the text "TBD:" (a real gap in the document, not a parse failure) |
| `Introduction` sheet, partial | Long merged narrative cells recovered; some formatting-only rows returned row indices |
| 119 `SqlHistory/*.xml` | Counted (≈1,478 nodes) but **not parsed**. Timestamps are present in a form my probe did not match. Parsing these properly would yield a *dated* record of Joel's own executed queries — the strongest possible personal-authorship evidence on the drive |
| `DailyNotes_Tracker.txt` | 0 bytes |

## Archives not expanded

**72 employment-cluster `.zip`, 2 `.7z`, 1 `.rar` were left unexpanded.** The four largest at the
source root (`BkpNov23 - Docs, Sql, Reportspbix.zip` 28.7 MB, `OneDrive_1_11-29-2023.zip` 10 MB,
`2023 J,F downloads.zip` 8.4 MB, `SQL SAVED_FILES, SCRIPTS.zip` 3 MB) are byte-for-byte the
already-extracted sibling folders and expanding them would have added nothing. **Not verified as
duplicates and therefore genuinely unknown:** `17 11 downloads; volt/JM_PERSONALS LATEST.zip`,
`MSdownload volt OD.zip`, `23FebDL Backups.zip`, `OneDrive_2_11-17-2023.zip`, and the wallet
archives under `WALLET zips, fs/` (the last deliberately, on privacy grounds).

PBIX/PBIT packages *were* expanded — as copies in the scratch directory, never on the source.

---

## Remaining high-value areas

Ranked by expected return:

1. ~~The 520 MB Sales-side PBIX model~~ — **closed, not possible on macOS** (see the parse-failure
   table above). Deprioritized entirely: BH Reporting artifact coverage is already the deepest of
   any workstream here.
2. **`SqlHistory/` (119 XML)** — dated proof of personally executed work. Now the highest-value
   remaining item.
3. **`SP LOAD COMM.txt` beyond the procedure map** — 12,154 lines, the densest remaining
   concentration of business logic (bonus calculation, expenditure-based commission variant).
4. **57 unread employment PDFs and 11 PPTX** — most likely to contain BRDs, requirement
   documents, sign-off records and status decks, i.e. the *acceptance and stakeholder* evidence
   this audit is thinnest on. Filenames were not individually reviewed.
5. **~160 unread employment workbooks** — the KPI-definition, mapping and reconciliation tier.
   `HLD, TestCases for Phase2 - Cred, Splits.xlsx` (860 KB) and
   `BH Reporting Phase 2 - HLD v1.3.xlsx` remaining sheets (`Technical HLD - Data Elements`,
   `Data Elements Definitions`, `Reporting - Measures`, `Reporting - Relationships`) are the top
   two — a full measure and data-element register would let the "500+ metrics" claim be replaced
   with a counted one.
6. **~40 `Untitled*.sql`** — unnamed scratch files, pattern-scanned but not read.
7. **`JM_PERSONALS LATEST.zip`** — unexamined and the name suggests it is Joel's own collection.
8. **`Halfhour buckets/` + `HalfHourCounts_Analysis_0303_wWorkHours.csv`** — an unexplained
   analytical workstream (Q-list, lower stakes).
