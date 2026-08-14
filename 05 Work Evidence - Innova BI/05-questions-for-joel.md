---
title: Questions for Joel
type: open-questions
audit_date: 2026-08-12
tags:
  - evidence/open-questions
  - work/innova
---

# 05 - Questions for Joel

Back to [[00-audit-summary]] · [[02-evidence-ledger]] · [[04-resume-candidate-bullets]] · [[07-strength-points]]

Only questions whose answer would change what goes on the page. Ordered by how much is at stake.

---

## Q1 — Where does the "88% accuracy" entity-resolution figure come from?

> **Partly resolved 2026-08-12.** Joel clarified that "fuzzy patterns" means cascading multi-round
> string-based `LEFT JOIN` matching — loose joins that intentionally return subsets or duplicates,
> with the duplication then resolved by business rules — not Oracle fuzzy *functions*. **Verified
> and confirmed:** 8,918 join predicates, 1,609 (18%) on transformed strings, 5,527 `LEFT JOIN`s
> across 207 files, and **199 OR-laddered string predicates of which 66 are six-way** — which is
> exactly the "6+ patterns". See `EV-65` through `EV-72`. **The pattern-count claim is cleared for
> use.** Only the question below remains open.

**What is still unresolved: the 88% figure specifically.**

**Artifact that raised it.** The client matcher records working arithmetic in its own comments —
*"Y = 2206; 4703 minus 2206 // but there are only 1476 distinct EBS customers matching to 2206 BH
entries. 4703 - 1476 = 3227 approx expected"*. These are match *counts* and residue arithmetic, not
an accuracy measurement. 2,206 of 4,703 is 47%; 1,476 of a smaller revenue-bearing subset could
plausibly land near 88%.

**Why it matters.** The pattern count is now defensible. A percentage that cannot be sourced is the
one remaining soft spot, and it is the kind of number an interviewer asks to see derived.

**Sub-questions:** Was 88% measured on the *person* matcher rather than the client matcher? Was it
coverage (share of clients or producers successfully resolved) rather than accuracy (share resolved
*correctly*)? Was it measured against a manually verified sample, and does that sample survive
anywhere? Did it come from a status report or a stakeholder review rather than the code?

**Evidence currently leans:** **yes** on 6+ fuzzy patterns (confirmed). **Unresolved** on 88% — and
the honest fallback is to state the technique without a percentage, which is already strong.

## Q2 — Was there real forecasting, or was it budget/target variance reporting?

**Artifact that raised it.** No forecasting, regression, trend or projection logic exists in the
corpus. What exists is `LKP_BASELINE_BH` / `LKP_BH_BASELINES` (per-person, per-period **targets**),
`BHRPT_FACT_USER_SUMMARIZED_METRICS` (actuals), and report pages named "Actuals vs Targets",
"BDM Baselines", "Rec Baselines". Separately, SG&A has `budget_load_flag` / `actuals_load_flag`
and `SP_SGNA_INSERT_VARIANCES_MONTHLY`.

**Why it matters.** "Built forecasting" invites questions about method, horizon and error metrics
that the artifacts cannot answer. "Budget/target variance reporting" is accurate and still senior.

**What it could support.** If forecasting genuinely existed, it deserves its own bullet with the
method named. If not, the variance-reporting framing replaces it cleanly.

**Sub-questions:** Were the baselines/targets *computed* (from trailing performance, ramp curves,
seasonality) or *supplied* by finance as an input file? Was there a pipeline projection —
open-jobs-to-expected-starts, or DSO forecasting off aging buckets?

**Evidence currently leans:** **no** on statistical forecasting. **Yes** on budget/target variance
reporting.

---

## Q3 — What is behind "70+ hours per month saved"?

**Artifact that raised it.** The mechanism is clearly visible — `Job_Load_FPnA_Manual_Inserts_Master`,
`Job_Load_SGnA_Manual_Inserts_Master`, `Master_ExcelFeed_to_AdwcLoad_Manual_loads`, the
`_STG_MANUAL` staging twins, and finance-owned files (`FPnA_Manual_Load.xlsx`,
`SGnA_Manual_Load.xlsx`, `Master_Tracker.xlsx`, `BudgetFile.xlsx`) landing on SFTP. Also
`Volt PowerbiAdministration_Activities.xlsx` lists recurring manual monitoring tasks that the five
monitoring reports would displace. But **no timesheet, effort estimate or before/after
measurement** exists — the closest is `Estimated Activities - Server, Gateway upgrades..xlsx`,
which estimates *project* effort, not recurring savings.

**Why it matters.** It is the only "impact" number on the résumé. Unsupported, it is the first
thing a careful interviewer probes.

**What it could support.** A defensible efficiency bullet — but the units must be preserved.
Manual effort eliminated, elapsed run-time reduction and deployment-time reduction are three
different claims and the current résumé blends them.

**Sub-questions:** Which specific manual process, and who was doing it before — a finance analyst
rekeying spreadsheets, or you running reports by hand? Was 70 hours estimated by you, agreed with
a stakeholder, or in a status report? Which reports/pipelines does it cover?

**Evidence currently leans:** **yes** that meaningful manual work was automated. **Unresolved** on
the 70-hour figure.

---

## Q4 — What was actually measured for "50% deployment time reduction"?

**Artifact that raised it.** The mechanism is unusually well evidenced (`EV-29`, `EV-11`):
parameter injection across all three layers, 69 versioned Talend artifacts, per-job context-change
tracking, environment-prefixed naming. But there is **no Git repo, no pipeline definition file, no
build log and no timing baseline** anywhere on the drive. `Estimated Activities - Server, Gateway
upgrades..xlsx` is the only effort artifact and it concerns infrastructure work.

**Why it matters.** "CI/CD" invites questions about Git branching, build agents and automated
tests, which this estate did not have. "Controlled promotion across four environments by
parameter injection" is both true and impressive to a data-platform interviewer.

**What it could support.** A concrete before/after release bullet, if you remember the numbers —
e.g. "reduced a manual per-job context reconfiguration of N parameters across 69 jobs from X hours
to Y".

**Sub-questions:** Before parameterization, was promotion done by editing job internals in Talend
Studio per environment? Was the 50% about a release window, or your own per-job effort? Was there
ever a Git or SVN repository for the Talend project or the SQL?

**Evidence currently leans:** **yes** on parameterized controlled promotion, **no** on CI/CD in
the pipeline-automation sense, **unresolved** on the 50%.

---

## Q5 — How many people used the BH Reporting model, and at what level?

**Artifact that raised it.** The RLS design implies a real audience — a per-person email map with a
parent-child organisational `PATH`, an `ISSUPERUSER` flag, and mirrored BDM/Recruiter navigation
across 85 pages. `Volt_Workspaces.csv` and `Volt NonPersonal Workspaces.xlsx` inventory
workspaces. But nothing counts *users*, and workspace or dimension row counts must not be
converted into user counts.

**Why it matters.** Audience size is the most common interview follow-up to any dashboard claim,
and it is currently unanswerable.

**What it could support.** A conservative reach statement, e.g. "used by recruiting and sales
leadership across N US business units" or "row-level security scoped to N producers".

**Sub-questions:** Roughly how many people were in the RLS email map? Which management levels
consumed it — team leads, regional leads, executives? Was it in a Premium capacity or Pro
workspace? Did an actual QBR run off this model, or off separate decks?

**Evidence currently leans:** **unresolved**. A designed-for-many system with no headcount
evidence.

---

## Q6 — Which of the 10 named reports reached production, and when?

**Artifact that raised it.** `Power BI reports - AWGP logics - Production Stats.xlsx` (2023-05-03)
names 10 reports but marks **Client Report (REC/BDM)** as "New Model in UAT" and **BH Reporting
(REC/BDM)** as "3 month amortization planned. Currently in UAT". Every recovered BH Reporting file
is named `BH_Reporting_UAT - ...` right up to 2023-09-27, two months before departure. The HLD is
"Phase 2".

**Why it matters.** Design documents proposing deployment are not evidence of production. If the
flagship 85-page model never left UAT, the bullet needs "delivered to UAT" or "delivered in
phases", not "delivered in production".

**What it could support.** Correct tense and status on the strongest single deliverable, and a
defensible answer to "was it actually live?"

**Sub-questions:** Did BH Reporting Phase 2 go live before Nov 2023? Did the 3-month amortization
ship? Was Phase 1 in production the whole time, with Phase 2 an enhancement? Which of the 10 were
already live in 2022?

**Evidence currently leans:** **yes** for US/India Commissions Statements, Ranking & Summits 2022,
Producers Report and APCOB/VSD/DSO (production cron triggers exist). **Unresolved / leaning
UAT-only at exit** for BH Reporting Phase 2 and the Client Reports.

---

## Q7 — What is the correct employer and client naming? — **RESOLVED**

> **Resolved 2026-08-12 by Joel.** The lineage is: **GGK → acquired by / folded into ACS (American
> CyberSystems) → ACS renamed itself Innova Solutions.** Innova then acquired **Volt Information
> Sciences** and other companies. This is one continuous employment through two corporate identity
> changes and at least one major acquisition — which explains every naming artifact on the drive:
> the "GGK WorkSample" folder and `ggk.demo.jobs` Talend GroupId (earliest era), the `acsicorp.com`
> / `acs.net` domains and "ACSBI team" documents (middle era), and the Volt tenant, Volt gateway
> work and Volt workspace inventories (final era).

**Use `Innova Solutions` as the employer line**, optionally with "(formerly ACS / American
CyberSystems)" on first mention. Dec 2019 – Nov 2023 is a single unbroken role.

**Framing opportunity, not a problem.** The multi-brand estate — Innova, Volt, Diversant
(`is_diver_hist`), Hiregenics, Ascent (`DIM_CUST_HIER_ASC_2202`, `DIM_FIN_PERIODDATE_ASCENT`) —
is direct evidence of building a warehouse and BI platform that had to absorb acquired companies'
data and, in Volt's case, an entire second Power BI tenant. Post-acquisition data integration is a
senior, marketable capability and it should be said out loud rather than left implicit.

**Only residual check:** whether naming the former employer's clients or acquired brands is
constrained by any agreement Joel signed. Naming the *employer* is normal résumé practice; naming
*client companies* by name is the part worth a moment's thought.

## Q8 — Were the Talend jobs themselves yours to claim?

**Artifact that raised it.** The drive contains deployment **manifests** (`jobs-def.xlsx`,
`Jobwise Context Change - PROD ...xlsx`) but **no Talend job internals** — no `.item` files, no
job XML, no `contexts` folder. One manifest is named `jobs-def (praveens master file).xlsx`. So the
recoverable evidence proves you tracked, versioned and promoted 69 jobs; it does not prove you
*built* them.

**Why it matters.** "Built 69 Talend jobs" and "managed promotion of 69 Talend jobs across four
environments" are very different claims, and only the second is currently provable.

**What it could support.** Either an authorship claim (if you can say which jobs you built) or a
release-ownership claim, which is itself valuable and rarer.

**Sub-questions:** Which jobs did you author versus inherit? Was Talend Studio development yours,
or were you primarily the PL/SQL and Power BI owner consuming jobs others built? Who owned the
Talend Management Console?

**Evidence currently leans:** **yes** for release ownership and context management (strong).
**Unresolved** for job authorship — the PL/SQL and Power BI authorship evidence is much stronger.

---

## Q9 — Does the Innova Idol Award exist as a record?

**Artifact that raised it.** No certificate, email, screenshot or mention anywhere on the drive.
The claim appears only in the baseline description given to this audit.

**Why it matters.** An unverifiable award is low-risk but zero-value if challenged; with a date and
citation it becomes a credible differentiator.

**What it could support.** A dated award line with what it was for.

**Sub-questions:** Do you have the certificate or announcement email in personal mail? What year,
and what was it recognising?

**Evidence currently leans:** **unresolved** — absent from this drive, which does not mean absent.

---

## Q10 — Can you still open the 520 MB PBIX offline? — **CLOSED, not possible**

> **Closed 2026-08-12.** Joel is on macOS. Power BI Desktop is Windows-only, and so are Tabular
> Editor and DAX Studio. Installing software is out of scope. The `DataModel` part inside the PBIX
> is XPress9/VertiPaq-compressed with no Python-stdlib decompressor, so there is no local route to
> it.

**What this costs, precisely.** Nothing about the model's *design* — the `.pbit` already gave the
complete schema: 40 user tables, 187 relationships, all 354 measures with their DAX, 129 calculated
columns, the RLS role, the calculation group and all 42 Power Query queries. What is permanently
unrecoverable is only **fact-table row counts and per-column compression sizes**.

**Consequence for the résumé:** make no data-volume, row-count or model-memory claim about the
semantic model. Use the design metrics instead, which are stronger evidence of skill than a row
count anyway. The only row-count figures available anywhere are the ones recorded in code comments
(341,711 rows in the crediting pass; 4,703 EBS customers; 37,129 excluded test rows) and those are
already in use.

**BH Reporting needs no further investigation.** Artifact coverage for it is the deepest of any
workstream in this audit.

## Also worth a moment (lower stakes)

- **`SP_LOAD_FACT_OPPORTUNITIES_SALESFORCE`** — was Salesforce opportunity/pipeline reporting a
  real workstream you owned? It is a named procedure but has no supporting documents, and adding
  Salesforce to the source list is a cheap credibility win if genuine.
- **Coupa integration** including a *reverse* write-back job
  (`Job_Load_Coupa_API_Invoice_Reverse_Integration_put_Non_Prod`) — bidirectional API integration
  is a strong bullet if it was yours; otherwise leave it in the platform description.
- **`0505_OCIMigrationAspect_BIBaremetals.txt`** and `ERP, HCM Analysis.xlsx` — was there an OCI
  migration or an ERP/HCM assessment you contributed to? Migration experience is highly marketable
  and currently unrepresented.
- **`Halfhour buckets/` + `HalfHourCounts_Analysis_0303_wWorkHours.csv`** (2023-02) — a
  half-hour-bucket call/activity analysis with work-hours adjustment. Looks like genuine analytical
  work (time bucketing, business-hours normalization) that nothing else references. What was it for?
- **`Plans (BH).xlsx`** (95 KB, 2023-06-07) and `DailyNotes_Tracker.txt` (0 bytes) — was there a
  project plan you owned with dates and owners? Would firm up the leadership evidence.
