---
title: FDE Story Bank - Client Facing Translations
type: interview-prep
role: Forward Deployed Engineer
created: 2026-08-12
tags:
  - interview/fde
  - interview/story-bank
  - career/2026
---

# 01 - Story Bank: Client-Facing Translations

Companion notes: [[00 - FDE Core Positioning and Screen Prep]] · [[02 - FDE Technical Rundowns and Scenarios]]

Every story here is anchored to verified evidence from [[07-strength-points]]. Numbers in **bold**
are recovered from primary artifacts and safe to state.

**Scope correction, 2026-08-12.** An earlier version of this file said you had no external-client
experience. That was wrong. You worked directly with client-side finance and audit teams - Privia
Health among them, plus other companies whose contingent-workforce programs ran through Hiregenics -
on billing, assignment numbers and revenue reconciliation, largely over Slack. You also worked
across a live company boundary during the Volt acquisition. **Story 8 is that story, and it is now
your lead answer to any client-facing question.** Full reasoning in Part 3 of
[[00 - FDE Core Positioning and Screen Prep]].

**How to use this:** each story has an exec version (30-45 seconds, business-first), a technical
version (2-3 minutes, for an engineer), the anchor numbers, and the follow-up questions it invites.
Learn the shape, not the wording.

---

## The register shift (read before the stories)

Consultants and engineers describe the same work in different orders. The FDE test is whether you
can switch on demand.

| Engineer register | Consultant register |
|---|---|
| "I built a cascading fuzzy-join resolver" | "The client couldn't tell which salesperson earned what, because the same person existed six times across six systems. I fixed the identity problem first, because nothing downstream works until that's solved." |
| "SCD Type 2 on transactional facts" | "They needed to see what the number *was* at the time a decision was made, not just what it is today." |
| "I zeroed 22 measures on non-primary rows" | "Seven people can touch one deal. Everybody needed to see their own activity, but the company could only book the revenue once. So I kept every row and only moved the money." |
| "Window-function collision detection" | "When the matching is loose on purpose, you get duplicates. The engineering is in catching them deliberately instead of silently taking the first row." |
| "Manual override column" | "Automate 95% and give the exceptions a permanent, auditable home. Finance keeps the ability to correct a number; the system keeps the audit trail. That's the trade that actually gets adopted." |

**The universal move: lead with the business problem, land on the technical decision, close with the
trade-off you accepted.** Three beats. Every time.

---

## Story 1 - The identity problem (your strongest consulting story)

**Use for:** post-merger integration, messy data, "tell me about a hard technical problem,"
roll-ups, carve-outs, anything about integrating acquired companies.

### Exec version

> "The company was a roll-up - it had grown by acquiring other staffing firms, and it eventually
> acquired Volt. So the same recruiter and the same client company existed in six different systems
> under six different spellings, with no shared ID anywhere. Until you solve that, nothing works:
> you can't pay commissions, you can't build a producer scorecard, you can't even scope who's
> allowed to see which client's data. It was the blocker underneath every other deliverable, so I
> built the resolution layer first."

### Technical version

> "There was no key to join on, so I built a multi-round cascading match. Each round joins on a
> different derived construction of the same entity - first-plus-last, last-plus-first for systems
> that store names reversed, first-middle-last for records where only one side carries a middle
> name, HR preferred name versus legal name, case-normalized variants, an employee-ID crosswalk
> where an ID actually exists in both, and system-prefixed composite keys so a collision across
> systems is structurally impossible.
>
> The important part isn't the matching, it's what you do with the mess it creates. The joins are
> deliberately loose - they over-match on purpose - so I used `ROW_NUMBER` and `COUNT(*) OVER` to
> detect collisions and rank them rather than silently taking the first row, `HAVING COUNT(*) > 1`
> probes written directly under the load code so the duplicate check ships with the change, and
> `MINUS` and `NOT EXISTS` to compute the unmatched residue explicitly and carry it forward instead
> of losing it. Unmatched records got sentinel keys so totals still reconciled and orphans stayed
> visible in the report instead of vanishing.
>
> And then the part I'd argue is the actual answer to fuzzy matching: a manual-override column on
> the match table itself. Automate the 95%, and give the remaining 5% a durable home where a human
> decision about a specific person survives every subsequent reload."

### Anchor numbers

- **8,918** join predicates parsed across the corpus; **1,609 (18%)** join on transformed strings
  rather than keys
- **5,527** LEFT JOINs across **207** files - a left-join-cascade codebase by design
- **199** multi-alternative OR-laddered predicates: **66 six-way**, 25 five-way, 11 four-way
- **9,593** string concatenations building composite keys
- Client side: **2,206** matched Bullhorn entries resolving to **1,476** distinct EBS customers out
  of **4,703**, with the ~**3,227** residue tracked explicitly
- The resolver itself: **1,469 lines**, with a run-log row written **per stage**, not per procedure

### Why this wins for FDE

Every PE roll-up and every carve-out has this exact problem. You have solved it at production scale
in a real post-acquisition estate. Most candidates have read about entity resolution; you have the
join-predicate counts.

### Follow-ups to expect

- *"How did you know the matching was right?"* → Residue tracking. I computed the unmatched set
  explicitly with `MINUS`/`NOT EXISTS` and carried it forward, so "how many did we fail to match"
  was always a number I could show, not a guess. Be honest: **the 88% accuracy figure on your
  resume has no artifact behind it.** Give counts, not accuracy.
- *"What would you do differently now?"* → Good answer available: the role-classification cascade
  sets *both* flags when the source can't classify someone, deferring the decision downstream. It
  works, but it pushes ambiguity into the attribution layer. I'd now make that an explicit
  "unclassified" state with its own review queue rather than a double-credit default.

---

## Story 2 - Crediting revenue once (the stakeholder-conflict story)

**Use for:** competing stakeholder requirements, sales compensation, "tell me about a time
requirements conflicted," business-rule design, executive translation.

### Exec version

> "On any given placement, up to seven people could legitimately claim credit - recruiter, a
> recruiter override, a secondary recruiter, business development, a BD override, a secondary BD,
> and a customer success manager. Sales needed every one of those people to see their own activity
> on their own scorecard. Finance needed the revenue counted exactly once, because the same table
> fed both commission statements and the P&L. Those two requirements are in direct conflict if you
> model it naively - a `SUM` multiplies the money by however many people touched the deal.
>
> The design that satisfied both: keep every row, and only move the money."

### Technical version

> "Precedence became data rather than code - a nine-level designation ladder, recruiter through
> tertiary BD, built as a ranked set so adding a new designation is a row insert, not a deployment.
> Then `COUNT(*) OVER` partitioned on assignment, person, activity code and period to detect the
> collision, `ROW_NUMBER` ordered by that precedence rank to elect the primary credit holder, and
> then - this is the bit people don't expect - I *zero* about 22 financial measures on the
> non-primary rows rather than deleting them. Revenue, gross profit, AWGP, burden and its
> components, direct cost, COGS, margin, billed FTE, hours. The row survives, so the producer still
> appears in their own scorecard with their activity intact. Only the money is attributed once.
> Then a `MERGE` back into the production fact on a 15-column composite key including activity code
> and a manual-load flag, so a hand-adjusted row and a system-sourced row can never collide."

### Anchor numbers

- **9** designation levels; up to **7** role-holders per placement
- ~**22** financial measures zeroed on non-primary rows
- **341,711** rows written in the crediting pass; **37,129** test/house rows excluded
- **15-column** composite key on the MERGE
- Governed by a matrix covering **10 named production reports × 7 producer roles**

### Why this wins for FDE

This is the purest "boardroom and codebase" story you own. Two executives wanted contradictory
things, both were right, and the resolution was a design decision - not a compromise, not a
meeting. Sales comp and margin attribution are also directly on the PE value-creation map.

### Follow-ups to expect

- *"How did you get finance and sales to agree?"* → Honest answer: I built the matrix. One artifact
  defining, for ten named reports and seven roles, who earns credit, whether direct-hire fees enter
  gross profit, and the amortization window - six months live. A finance stakeholder and an
  engineer could both read it. The agreement followed from making the rule visible, not from
  winning an argument.
- *"What did you push back on?"* → The definition of AWGP. It's gross profit over fiscal weeks
  adjusted for holidays, for the selected period - which means the denominator changes with the
  user's own filter selection. It can't be a stored division. I had to explain to finance why the
  number couldn't just be a column, and recompute it inside filter context against a custom fiscal
  calendar.

---

## Story 3 - Six months of contradictory requirements (the ambiguity story)

**Use for:** "difficult stakeholder," "ambiguous requirements," "how do you handle scope change,"
delivery under pressure.

### The shape

> "For about six months the finance requirements kept moving, the underlying data had real quality
> problems, and some of the business logic they were asking for actually contradicted itself. You
> couldn't solve that by writing more SQL. The job was to find the broken calculation, work out
> what the intended behavior actually was, get a decision on it, track the defect, and *still* keep
> visible delivery moving - because a stakeholder who sees nothing ship for six months stops
> believing you.
>
> So I split it. I stayed hands-on wherever the ambiguity or the data quality actually threatened
> the system, and I let teammates own the parts where they'd learn something. And I deliberately
> looked for smaller wins - dashboard features that were genuinely useful and could land while the
> big correctness problems were still open. Raising a contradiction is better than silently
> encoding one. The failure mode is a system that quietly implements a rule nobody agreed to."

### Why this wins for FDE

The JD says *"manage expectations"* and *"ask thoughtful questions."* This story is both, and it
is the closest thing you have to a consulting-engagement narrative. Note that the six-month
duration comes from your recollection, so use it only where it matters.

### Where to be careful

Do not claim staffing authority, an approved executive roadmap, budget ownership, or a final
adoption metric. The mentoring is real; the formal management is not.

---

## Story 4 - Taking over a second tenant after an acquisition (the PMI story)

**Use for:** post-merger integration, ownership, "tell me about picking up something unfamiliar,"
platform work, enablement.

### Exec version

> "When Innova acquired Volt, a second Power BI tenant came into scope - unfamiliar estate, nobody
> internal who owned it. I took it over through a structured knowledge transfer and then ran both
> tenants. Inventory first: what workspaces exist, what's on dedicated capacity, what's actually
> refreshing. Then the operational risk: the on-premises gateway, which is the single point of
> failure between the cloud reports and everything still living in the network."

### Technical version

> "For monitoring I built against the Power BI REST API - a service principal on Azure AD with an
> OAuth2 client-credentials flow, acquiring a bearer token in Power Query M and expanding workspace
> and capacity metadata across both tenants. Five monitoring reports covering workspace inventory
> and dataset refresh status.
>
> For the gateway I wrote the upgrade procedure - admin prerequisites, a required VM snapshot
> before the upgrade, a user communication template, an expected one-to-two-hour downtime window,
> and the distinction that actually matters operationally: which sources are gateway-dependent
> because they're in-network - Oracle Financials, HR, the Bullhorn mirror, the warehouse - versus
> the cloud sources that are unaffected. If you don't know that split, you either take down more
> than you need to or you miss something.
>
> I also specified dedicated production BI infrastructure and argued it on its merits - separate
> production from non-production so a development refresh can't degrade a production report, and
> support more than two concurrent developers."

### Anchor numbers

- **2** Power BI tenants across **5** staffing brands
- **5** monitoring reports built on the REST API
- Infrastructure spec: **32 GB RAM, 8 cores, 250 GB**, with the justification written out
- Training curriculum authored: **42 Power BI topics**, **33 SQL topics**

### Accuracy note

Pre-existing Volt documentation was consolidated and extended, not written from scratch - the
document says so itself. Say **"consolidated and extended."** That is still ownership, and being
precise about it is exactly the instinct a consulting firm is checking for.

---

## Story 5 - FinSights (your AI credibility story)

**Use for:** the RAG/LLM questions, "what have you built recently," evaluation, cost discipline.

### Exec version

> "I built a retrieval system over SEC 10-K filings that answers analyst-style questions with
> citations back to the source. The part I'd emphasize isn't that it works - it's that I measured
> it. About two cents per query, per-stage latency tracked, and an evaluation suite rather than a
> demo. If you're putting this in front of a client, 'it seems good' isn't an answer."

### Technical version

> "The architecture combines KPI table extraction, fuzzy entity extraction, and vector retrieval
> through S3 Vectors on AWS. The embedding pipeline ran about 1,850 vectors a minute with
> checkpointing, token and rate controls, and live cost tracking. Retrieval used query variants,
> filtered/global/union regimes, sentence-window expansion, dedup, and stratified distance
> selection. Deployment separated LLM orchestration, a FastAPI service, and a Streamlit front end,
> running on ECS.
>
> The finding I'd actually lead with: I tested cross-encoder reranking rather than assuming it, and
> the ablation showed it *degraded* results on this corpus. So I didn't ship it. That's the habit
> that matters more than any individual component - reranking is supposed to help, it's in every
> reference architecture, and on this data it didn't."

### Anchor numbers

- ~**1,850** vectors/minute embedding throughput
- ~**$0.02** inference cost per query
- **0.82-0.86** BERTScore F1 on business-realistic tests
- Validated with deterministic neighbor tests plus BLEURT-20, ROUGE-L, BERTScore, LLM judges

### Be ready for the honest limitation

If they ask about scale: *"It's a single deployed service with a thin front end - it doesn't have
multi-tenant serving, VPC and load-balancer architecture, or real concurrency work behind it. That's
the gap between a system I built end to end and one that's been through production traffic."*

Saying this yourself, before they find it, converts a weakness into evidence of calibration.

---

## Story 6 - The multi-agent system (your differentiator)

**Use for:** agentic workflows, orchestration, MCP, "what's the most technically interesting thing
you've built."

### Exec version

> "I built a multi-agent debate system that runs entirely on my own machine - thirteen nodes,
> agents that argue positions against retrieved evidence. Two things make it more than a demo: it
> survives crashes and resumes without duplicating work, and the tools are exposed over Model
> Context Protocol so the same capability runs in-process or over a protocol boundary."

### Technical version

> "Thirteen-node LangGraph state machine with gates, cycles, map-reduce fan-out, typed reducers,
> and SQLite checkpointing. I deliberately forced a recursion failure to verify a partial run could
> resume without duplicating turns - which a happy-path demo never tests.
>
> Two MCP servers exposing nine tools and seven resources behind a shared adapter, so the same
> retrieval capability is callable either way. Measured transport latency: 15.3 ms in-process
> versus 23.1 ms over MCP. Retrieval was hybrid BM25-plus-dense with reciprocal rank fusion over
> 14,195 argument passages, grounding each debate turn.
>
> The constraint that made it interesting: serving about 34 GB of quantized model weights on 24 GB
> of unified memory, using phase-separated residency so only the models needed for the current
> phase are resident."

### Why this matters for this specific job

The JD calls agentic workflows "highly desirable." A consultancy selling agentic solutions to PE
portfolio companies needs people who have actually shipped one. **Most candidates with the
client-facing years you lack have never written an MCP server.** This is the card that offsets your
gap - play it early and concretely.

---

## Story 8 - Working with client-side finance and audit teams (your lead client-facing story)

**Use for:** "external customers," "client-facing experience," stakeholder management, "a time you
had to explain something technical to a business audience," reconciling disputed numbers.

**Read this before using it:** the shape below is correct and the framing is right. Two or three
concrete beats are still missing and only you can supply them - see *What to fill in* at the end.
Do not improvise details about a named company under pressure. Fill them in once, properly, then
use it everywhere.

### Exec version

> "A lot of my stakeholder work was with client-side finance and audit teams rather than internal
> ones. Hiregenics ran contingent-workforce programs for client companies - Privia Health was one -
> and those clients had their own finance people, their own auditors, their own billing cycles and
> their own assignment numbers. When a number looked wrong, it was their number and their money, so
> it came back fast and it came back directly, usually over Slack rather than through an account
> manager.
>
> That's a different job from internal reporting. Internally you can say 'I'll look at it this
> week.' With a client reconciling a billing figure against their own system, you need to
> acknowledge it the same day, explain what the difference actually is, and be clear about which
> number answers which question."

### Technical version

> "The reconciliation work was the substance of it. The same figure computed from two systems
> disagrees, and the job is finding *which step* it diverges at - date range, entity inclusion,
> whether house and test accounts are in or out, recognition timing, whether a manual adjustment
> happened downstream of the load. I did a sustained run of those investigations through 2023,
> including tracing a negative-margin defect from the report back through the FP&A layer to the
> Oracle EBS source.
>
> The fix that mattered wasn't any individual reconciliation - it was making adjustment governable.
> Every hierarchy dimension and the FP&A fact carries an automated staging table and a human
> `_STG_MANUAL` twin, with a `manual_load_flag` propagated onto the fact - 295 references to it -
> so a finance adjustment stays traceable in the report that displays it. Client finance keeps the
> ability to correct a number; the warehouse keeps the audit trail. Before that, corrections lived
> in spreadsheets outside the system, which is unauditable and it's exactly what an external auditor
> asks about."

### The boundary point, if they probe whether these were "really" external

Say this plainly and without defensiveness:

> "Fair question, and the honest answer is the lines were genuinely blurred - that's what a live
> roll-up looks like from inside. GGK went into ACS, ACS became Innova, Innova acquired Volt. On
> paper some of those people ended up under one group eventually. In practice, when I was working
> with them, they were a different company with different systems, their own milestones, their own
> revenue and billing, and no shared history with us. Program clients like Privia Health were never
> internal at all. If anything I undersold that for a while - I'd been calling it internal work
> because we shared a parent by the time I left."

That answer is strong *because* it concedes the complexity instead of flattening it. It also
demonstrates the exact judgment a consultancy is hiring for: knowing who your actual counterparty
is.

### Anchor numbers

- `manual_load_flag` propagated onto the fact - **295** references
- **10** named production reports governed by the shared business-rule matrix, including client
  reports for both recruiter and BDM views
- Dated reconciliation investigations across **2023**, plus **104 KB** of change notes written in
  the two weeks before departure
- A proof-of-concept report for **external client-level row-level security**

### Why this wins

It converts your single biggest apparent gap into a strength, and the register is already right -
"acknowledge same day, explain which number answers which question" is how consultants talk about
client escalation. It also pairs directly with Scenario C in
[[02 - FDE Technical Rundowns and Scenarios]], which is the same story as a live exercise.

### What to fill in - only you know these

1. **One specific Privia Health (or other named client) interaction, start to finish.** What did
   they ask for or dispute, what did you find, what did you change? One vivid beat beats three
   general claims.
2. **Who exactly you were talking to** - client finance analyst, controller, program manager,
   external auditor? Titles make it real.
3. **Whether you ever ran the call** versus participating in it. Both are usable; they are
   different claims and you should know which one you are making.
4. **Any other named external clients** you are comfortable naming.

---

## Story 7 - The design decision that shows product judgment

**Use for:** "tell me about a trade-off," "how do you decide between two valid options."

Small story, disproportionately effective, because it is exactly how consultants think.

> "When I resolved Bullhorn client companies against the Oracle EBS customer hierarchy, I had two
> different names for the same company and had to pick which one wins. I split it: the **Bullhorn**
> name wins as the display name, because the slicer is recruiter-facing and recruiters know clients
> by the name in their ATS. But the relationship to the fact table travels on the **EBS** key,
> because the fact is financial and finance reconciles to EBS.
>
> Same entity, two representations, chosen by who's looking at it. If I'd picked one name for both,
> I'd have either confused every recruiter or broken finance's reconciliation."

That is a one-minute story that demonstrates you optimize for the human on the other end rather
than for schema purity. It is the single most FDE-flavored thing in your evidence file.

---

## Questions to ask them (have four ready)

Asking good questions is half of a client-facing screen. These are calibrated to this specific
firm.

1. **"What does a typical engagement look like in duration and team shape - am I embedded with one
   portfolio company for months, or moving across several?"** Shows you understand the model and
   tests the burnout risk the Reddit thread flagged.
2. **"How much of the work is net-new build versus untangling an inherited data estate after an
   acquisition?"** This is the question that lets you deploy Story 1.
3. **"Where do FDEs here go after two or three years?"** Directly from the Reddit advice - it tests
   whether FDEs are treated as first-class or as rebranded support. If they cannot answer it, that
   is real signal.
4. **"When you deploy agentic solutions into a portfolio company, who owns it after you leave -
   your team, or theirs?"** Sophisticated question. It is about adoption and handover, which is the
   actual hard part of consulting delivery, and it signals you think past the demo.

Optional fifth, if the conversation is going well:
**"What's the most common reason an engagement here goes sideways?"** People answer this honestly
more often than you would expect, and it tells you what the job is really like.

---

## Anti-patterns - do not do these

- **Do not use the word "just."** "I just built a small RAG thing." You did not. Watch for this in
  your own speech; it is your most consistent tell.
- **Do not describe your client work as internal.** This cost you a submission on 2026-08-12. When
  a counterparty is a different legal entity with their own billing, milestones and revenue, they
  are a client - regardless of what the org chart looked like a year later. Lead with Story 8.
- **Do not pre-emptively concede a requirement.** State what you did and let them judge whether it
  qualifies. You are not prone to overclaiming; you are prone to disqualifying yourself before
  anyone asks you to.
- **Do not lead with the BI/dashboard summary on this role.** Lead with the platform and the
  integration problem. Dashboards sound like reporting; platform integration sounds like consulting.
- **Do not quote unsourced numbers** (50% deployment, 70+ hours, 88% accuracy). Mechanism over
  metric. See Part 7 of [[00 - FDE Core Positioning and Screen Prep]].
- **Do not say "real-time."** Daily batch with a 15-minute CDC feed.
- **Do not oversell FinSights as production-scale.** Name the limitation yourself.
