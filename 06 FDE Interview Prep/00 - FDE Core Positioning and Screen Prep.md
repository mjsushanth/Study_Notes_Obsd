---
title: FDE Core Positioning and Screen Prep
type: interview-prep
role: Forward Deployed Engineer
client: "Data analytics + technology consulting firm serving private equity firms and portfolio companies (client name undisclosed by Jobot)"
recruiter: "Forrest Mack, Executive Manager, Jobot (Newport Beach, CA)"
posting_range: "$160,000 - $300,000"
locations: "Boston, MA +4 (Charleston, SC confirmed as one)"
applied: 2026-08-12
created: 2026-08-12
tags:
  - interview/fde
  - job/jobot
  - career/2026
---

# 00 - Core Positioning and Screen Prep

Companion notes: [[01 - FDE Story Bank - Client Facing Translations]] · [[02 - FDE Technical Rundowns and Scenarios]]

---

## Part 0 - Read this first: the immediate gate is an AI screener

"Jeni, Virtual Recruiting Assistant" is **not a person**. It is an automated voice screener, 8-12
minutes, and it is the gate before any human at Jobot or the client reads your file.

**What that changes about how you answer:**

An AI screener is not evaluating nuance. It is almost certainly walking the requirements list in
order and scoring whether your spoken answer contains a match. This has two consequences:

1. **Never answer a requirement question with a bare "no."** A human recruiter hears "no, but
   here's the adjacent thing" and follows the thread. An automated screener that gets "no" logs a
   miss and moves on. Every answer needs the qualifying substance *inside the same answer*, in the
   first two sentences, before the system decides you have moved on.
2. **Say the keywords out loud.** Python. SQL. RAG. AI agents. Agentic workflows. Cloud-native.
   APIs. Data pipelines. LLMs. Stakeholders. Requirements. Architecture. These are almost certainly
   the tokens being matched. You have every one of them legitimately - so use the actual words
   rather than a synonym or a story that implies them.

**The seven questions it will almost certainly ask,** taken straight off the requirements list,
with your prepared answers below in [Part 4](#part-4---the-ai-screen-answer-bank).

---

## Part 1 - What this job actually is

Strip the title. This is an **implementation consultant / solutions architect at a
PE-focused analytics consultancy**, and the Reddit read you found was correct. That is not a
downgrade - it is a much better description of what you would actually do day to day, and it makes
the role far more legible against your background than the FDE label does.

**The business model, which you should understand before you speak to anyone:**

A private equity firm buys companies (portfolio companies, "portcos"), holds them 3-7 years, tries
to raise their value, and sells. Value goes up through EBITDA growth and multiple expansion. PE
firms hire consultancies like this client to make the portcos measurably better, fast, because the
hold clock is running. That is why the JD says "measurable outcomes" and "improve decision-making
across sales, operations, marketing, pricing, and customer strategy" - those are the standard
EBITDA levers.

**What that means for the engineering work:** short engagements, messy inherited data estates,
non-technical executive stakeholders who need a number they can act on, and a hard deadline set by
someone else's investment thesis. Not greenfield. Not elegant. Fast, correct, defensible.

**Vocabulary you should recognize instantly** (do not force these into answers - just do not be
caught blank when someone says one):

| Term | Meaning |
|---|---|
| Portco / portfolio company | A company the PE fund owns |
| Platform vs add-on | The first acquisition in a sector vs. subsequent bolt-ons merged into it |
| Roll-up | Strategy of buying many small companies and merging them into one |
| Value creation plan (VCP) | The written thesis for how this company gets more valuable |
| 100-day plan | The initial post-acquisition sprint of changes |
| PMI / post-merger integration | Merging systems, data, and processes after an acquisition |
| Carve-out | Separating a division out of a parent company, including untangling its data |
| Diligence (commercial / tech / QoE) | Pre-purchase investigation; QoE = quality of earnings |
| EBITDA | Earnings before interest, taxes, depreciation, amortization - the headline profit number |
| Operating partner | The PE firm's internal person who works hands-on with portcos |
| Hold period / exit | How long they own it / the sale that ends it |

---

## Part 2 - Your positioning thesis (the one thing to get right)

You have been framing your Innova work as "internal BI for a big services company." That framing
is costing you this interview. Here is the accurate reframe:

> **Innova Solutions is itself a roll-up.** GGK was acquired into ACS (American CyberSystems), ACS
> was renamed Innova, and Innova then acquired Volt Information Sciences. The data estate you
> worked on spanned five staffing brands - Innova, Volt, Diversant, Hiregenics, Ascent - across two
> Power BI tenants, eight-plus source systems, and four Oracle EBS instances.
>
> **You spent four years doing post-merger data integration on a multi-brand roll-up.** That is
> precisely the engagement this consultancy sells to private equity.

This is not spin. It is the literal corporate lineage recorded in
[[00-audit-summary]] and [[07-strength-points]]. You lived the exact problem their clients hire
them to solve, from the inside, for four years.

**Three specific things you did that are PE-consultancy work by another name:**

1. **Post-acquisition system consolidation.** You took over an unfamiliar second Power BI tenant
   after the Volt acquisition through structured knowledge transfer, then ran both - workspace
   inventory, gateway lifecycle, refresh monitoring via the Power BI REST API with an Azure AD
   service principal, and a formal specification for dedicated production BI infrastructure. That
   is PMI work.
2. **Cross-system identity resolution with no shared key.** Six systems, six spellings of the same
   human being and the same client company. This is the single most universal problem in every
   roll-up and every carve-out, and it is the thing that blocks all downstream reporting. You built
   the cascading multi-round resolver: 8,918 join predicates parsed, 1,609 joining on transformed
   strings, 199 OR-laddered multi-alternative predicates of which 66 are six-way ladders, with
   window-function collision detection, explicit unmatched-residue tracking, sentinel keys, and a
   durable manual-override column.
3. **Revenue attribution across contested credit.** Up to seven people can claim credit on one
   placement. You built the nine-level precedence ladder that credits the money exactly once while
   preserving every producer's visibility into their own activity - feeding commission statements
   and P&L. Sales compensation and margin attribution are core PE value-creation territory.

**Your one-sentence positioning:**

> "I spent four years building the data platform for a staffing group that was itself a roll-up -
> eight-plus source systems, five acquired brands, two tenants - working with client-side finance
> and audit teams on billing and revenue while the acquisitions were still in flight. Then two
> years on the AI side, shipping a deployed RAG system on AWS and a multi-agent orchestration
> system. Messy post-acquisition data estates with real commercial stakes on the other end of the
> conversation is the environment I actually know."

---

## Part 3 - Honest gap map, line by line

Do this exercise before you talk to anyone, because knowing exactly where you are short lets you
answer confidently instead of defensively.

| JD requirement | Your position | Verdict |
|---|---|---|
| 5+ years in SWE / DE / MLE / Solutions Eng / FDE | ~4 years at Innova (Dec 2019 - Nov 2023) + M.S. in AI (Jan 2024 - Dec 2025) with two substantial builds | **Just under.** Say "four years professional data engineering plus two years of graduate AI work, so about six years in the field with a gap for the degree." Honest, and closes the distance without lying. |
| Experience with **external** customers/clients | **Yes - this was mis-scoped in an earlier draft of this file.** Client-side finance and audit teams on billing, assignment numbers and revenue reconciliation, including Privia Health and other companies running contingent-workforce programs through Hiregenics. Day-to-day contact over Slack, not mediated by an account manager. Plus cross-company work during the Volt acquisition, where the counterparties were legally another company's employees. | **Real, and you must claim it.** See Part 4 and Story 8. |
| Leading discovery sessions, technical workshops, architecture discussions | Authored the team's High Level Design v1.3 "for project managers, Development leads and Testing leads." Authored a two-track training curriculum - 42 Power BI topics, 33 SQL topics - and delivered it. Ran structured knowledge-transfer sessions during the Volt tenant handover. Conducted 35+ interviews. | **Genuinely strong, and you have been underselling it.** Workshops and architecture docs delivered to a technical audience is exactly this bullet - it was just internal. |
| Strong Python and SQL | ~30,700 lines of PL/SQL across 39 procedures; Python throughout FinSights and the LangGraph system | **Clear strength.** |
| Cloud-native applications, APIs, data pipelines, AI-enabled solutions | FinSights on AWS (ECS, S3 Vectors, FastAPI); Oracle ADW platform, Talend orchestration, Coupa API integration including a reverse write-back; Power BI REST API with OAuth2 service principal | **Clear strength.** |
| LLMs, RAG, AI agents, agentic workflows ("highly desirable") | FinSights end-to-end RAG; 13-node LangGraph multi-agent system with two custom MCP servers exposing 9 tools and 7 resources | **Strongest single card you hold.** Most candidates with 5 years of consulting delivery have not built either. |
| Communicate with executives and engineers | Business-rule matrix readable by both a finance stakeholder and an engineer; self-documenting in-report glossary pages; six months of contradictory finance requirements resolved across multiple entities' finance teams | **Strong.** |
| Willingness to travel to customer sites | **Unknown - you have not told me** | **Decide before any screen.** See Part 5. |

**Score: seven clear strengths, one near-miss on years, one unanswered logistics question.**

That is a competitive profile for a role paying $160-300k. The only genuine shortfall is the
five-year line, and you close most of that by counting the graduate build work honestly.

**Standing correction to carry into every future application.** An earlier version of this file
scored you as having no external-client experience. That was wrong, and it came from reading your
evidence audit's phrase "finance and operations stakeholders" as meaning internal - the audit
described stakeholders by *function*, because organizational boundary was never the question it
set out to answer. The facts:

- **Hiregenics runs contingent-workforce programs for client companies.** A company like Privia
  Health whose staffing ran through HGC is a **client**, not a colleague. Their finance and audit
  people asking you about billing, assignment numbers and revenue are external client stakeholders
  by any definition a consulting firm would use.
- **External audit teams are external.** Reconciling a number with an auditor carries stakes and
  consequences that internal reporting does not.
- **An in-flight acquisition is a company boundary, not a team boundary.** During the Volt
  integration the people on the other side were legally another company's employees, unfamiliar,
  with their own systems, milestones and interests. Their IT manager authored the gateway
  documentation you consolidated and extended. That is cross-organizational delivery.
- **Each of those programs had its own milestones, billing, revenue and assignment numbers.**
  Separate commercial stakes, not a shared internal P&L.

Never re-narrow this. The instinct to describe a company boundary as "basically internal because we
were all under one group by then" is exactly what cost you the first pass on this req.

---

## Part 4 - The AI screen answer bank

Speak these. Do not read them verbatim - they are structured so the keyword lands in the first
sentence and the substance follows.

### Q: "Do you have 5+ years of experience in software engineering, data engineering, or a related field?"

> "Yes - about four years of professional data engineering at Innova Solutions, from December 2019
> through November 2023, plus two years of graduate AI engineering work at Northeastern where I
> built and deployed production systems. So roughly six years in the field, with the degree in the
> middle."

*Never say "only four."* The graduate work was building, not coursework, and it counts.

### Q: "Do you have experience working directly with external customers or clients to define technical requirements?"

Answer it plainly and with names. **Do not hedge this any more.**

> "Yes, in a few different shapes. On the client side, I worked directly with external finance and
> audit teams on billing, assignment numbers and revenue reconciliation - Privia Health was one,
> and there were others running contingent-workforce programs through our Hiregenics business. That
> was day-to-day contact, a lot of it over Slack, not routed through an account manager. Each of
> those programs had its own milestones and its own revenue, so when a number was wrong it was
> their number, not an internal report.
>
> And separately, during the Volt acquisition I was working across an actual company boundary -
> people who were legally another company's employees, who'd never worked with us, with their own
> systems and their own way of doing things. I took over their reporting tenant through a structured
> knowledge transfer and consolidated documentation their IT manager had written. That's
> cross-organizational delivery, whatever the org chart said six months later."

### Q: "Have you led customer discovery sessions, technical workshops, or architecture discussions?"

> "Yes. I authored our High Level Design document - version 1.3, written for project managers,
> development leads and testing leads - which was the architecture reference for the whole
> warehouse. I authored and delivered a two-track training curriculum, 42 topics on Power BI and 33
> on SQL. And I ran the knowledge-transfer sessions when we absorbed a second reporting tenant from
> an acquired company, which meant sitting with their people to work out what their estate actually
> did before I could own it."

### Q: "How strong are your Python and SQL skills?"

> "Both are core. On SQL, I wrote about 30,700 lines of PL/SQL across 39 stored procedures against
> an Oracle Autonomous Data Warehouse - window functions, MERGE upserts, dynamic SQL, change-data-
> capture replay. On Python, everything in my recent AI work: I built a retrieval-augmented
> generation system deployed on AWS with FastAPI, and a multi-agent orchestration system in
> LangGraph."

### Q: "Do you have experience with LLMs, RAG, AI agents, or agentic workflows?"

Lean in hard here. This is your differentiator.

> "Yes, directly. I built FinSights, a retrieval-augmented generation system over SEC 10-K filings,
> deployed on AWS - S3 Vectors for the vector store, FastAPI service layer, with cost and latency
> instrumentation and a full evaluation suite. I also built a multi-agent system on LangGraph -
> thirteen nodes with cycles, checkpointing and crash recovery - and wrote two Model Context
> Protocol servers exposing nine tools and seven resources behind a shared adapter, so the same
> retrieval capability runs in-process or over MCP."

### Q: "Are you willing to travel to customer sites?"

> "Yes." *(Assuming you are - see Part 5. If you are genuinely not, say "yes, occasional travel is
> fine" only if true. Do not commit to something you will refuse later.)*

### Q: "What are your compensation expectations?"

**Do not repeat $100,000.** See Part 6.

> "Based on the scope of the role and the Boston market, I'm targeting the $150,000 to $180,000
> range on base, and I'm flexible depending on the overall package and level."

### Q: "Are you authorized to work in the United States?"

> "Yes, I'm authorized to work in the US and I don't require sponsorship right now."

Both true per your approved screening answers. If it asks whether you will *ever* need sponsorship,
or asks about citizenship or clearance - that is a different question with a different approved
answer, and it goes to a human, not an automated screener. Answer citizenship honestly ("no") if
directly asked; do not volunteer it.

---

## Part 5 - The two decisions you must make before the screen

### 1. Travel and location

The req spans Boston MA + 4 locations, and Jeni's message routed you to **Charleston, SC**. Two
possibilities: it is a multi-site req and Charleston is where the current need is, or the automated
router simply matched you to the wrong location. Either way:

- If Charleston is a **relocation** ask, decide now whether that is a yes.
- If the role is Boston-hybrid with travel, that is a much easier yes.
- **Ask Forrest directly which location this specific req is for.** It is a completely normal
  question and it also signals you are treating the process seriously.

### 2. Your OPT timeline versus a consulting firm

You are authorized now, EAD valid **Jul 26 2026 - Feb 20 2027**. A consultancy that bills you to
clients will care about continuity of authorization, and STEM OPT requires an E-Verify-enrolled
employer. Two things follow:

- Do **not** raise sponsorship in the AI screen. It is not asked, and it is not the screener's
  decision.
- **Do** get clarity from Forrest before a client interview about whether the end client is
  E-Verify enrolled, because that binds later. A staffing-adjacent consultancy very likely is.
- Anything beyond Feb 20 2027 - STEM OPT timing, H-1B, cap-exempt status - goes to your ISSO or an
  attorney, not to a recruiter and not to me.

---

## Part 6 - The compensation correction

You told an intake form **$100,000**. This req posts at **$160,000 - $300,000**.

That gap is large enough to actively hurt you in two ways: a contingency recruiter earns a
percentage of your placement salary, so a low anchor gives Forrest a reason to route you to cheaper
reqs; and a number that far below band can read as a self-assessment that you are junior for the
role.

**Fix it before the screen if you can, or in the screen itself.** Suggested text to Forrest:

> "Quick correction on something from my intake - I put $100k as a floor before I'd seen the band
> on this req. Realistically I'm targeting $150-180k base for this scope. Didn't want a stale
> number sitting in the file."

**Why $150-180k and not $200k+:** the top of that band is for someone with the client-facing years
you do not have yet. Asking mid-band is credible; asking top-of-band invites the comparison you
would lose. Mid-band is also 50-80% above the number currently in your file, which is the point.

---

## Part 7 - Resume claims that will not survive probing

Flagging this because it is a real interview risk, not a nitpick. Your registered resume PDFs
carry claims that the 2026-08-12 primary-source evidence audit could **not** substantiate. If an
interviewer probes one of these and you cannot back it, it damages everything else you said.

| Claim on resume | Status | What to say if probed |
|---|---|---|
| "CI/CD practices, 50% deployment time reduction" | Mechanism verified, figure unsourced. There was no Git repo, no build server, no automated test harness. | Describe the mechanism: parameter injection across DEV/QA/UAT/PROD, 69 versioned Talend job artifacts, deployment manifest with per-environment context params. Call it "release engineering and controlled promotion," not CI/CD. **Never** get caught claiming CI/CD and then having no branching strategy answer. |
| "70+ hours per month saved" | Unsourced | Describe the governed manual-override path instead. |
| "88% entity resolution accuracy" | Technique fully verified; the accuracy figure has no artifact | Give counts, which are real: 2,206 matched Bullhorn client entries resolving to 1,476 distinct EBS customers out of 4,703, with the ~3,227 residue explicitly tracked. |
| "25+ dashboards, 500+ metrics" | Neither confirmed nor contradicted | Use the exact recovered numbers: 354 DAX measures, 10 named production reports, 85 report pages, 1,035 visuals, 40 tables, 187 relationships. Stronger anyway. |
| "Led sub-teams of 2-4" | Team size not evidenced | "Mentored teammates and authored the team's Power BI and SQL training curriculum." |
| "Real-time" anything | Contradicted - daily cron at 14:00/14:30 IST plus one 15-min CDC job | "Daily batch with a 15-minute change-data-capture feed." |
| "Built forecasting models" | No forecasting code exists | "Budget and target versus actuals variance reporting." |

**Rule for the whole interview: when a number is shaky, describe the mechanism instead.** A
specific mechanism you can explain in depth beats a round number you cannot defend, every single
time - and consultants can smell an undefendable number from across a table.

---

## Part 8 - The mental reframe, since you said you were scared

You wrote: *"I actually have NO architect or solutions architect level experience... I worked on
the dev end where I implemented things people told me to."*

Against your own evidence file, that is false, and it is worth correcting in your own head before
you speak to anyone.

Nobody told you to build a nine-level designation precedence ladder to solve revenue double-
counting. Nobody handed you a spec for a cascading six-way fuzzy match ladder with sentinel keys
and a manual-override column. Nobody specified that the Bullhorn name should win as the display
name while the relationship travels on the EBS key. You made those calls. **You made them, and the
reason you can't see them as architecture is that you were the only one in the room, so nobody
ever handed you the title.**

You also told yourself you had no client experience, and that turned out to be a framing error
rather than a fact. See the standing correction in Part 3.

What you are actually missing, after both corrections, is narrow:

- You have not worked inside a consultancy, so you have not run a formal engagement lifecycle -
  scoping document, statement of work, billable hours, a defined end date.
- You have not built a system with multi-tenant serving, VPC/ALB, and real concurrency. FinSights
  is one service with a thin frontend, and you are right that it is.

That is the whole list. Both are learnable in the role, and the second one is the only technical
item on it.

**The pattern worth noticing, because it will keep costing you money:** twice in two days you
described your own work in the narrowest possible terms - "I just implemented what people told me
to," "internal stakeholders only" - and both times the evidence contradicted you. You are not
prone to overclaiming. You are prone to pre-emptively conceding things nobody asked you to concede.
In a screening conversation that reads as a disqualification, because a recruiter takes your
self-assessment at face value - they have no reason not to.

Describe what happened. Let them decide whether it counts.
